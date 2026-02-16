"""Training script for multimodal JEPA with attention-based interaction."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import v2
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from multimodal_models import MultimodalLeJEPAModel
from retrieval_eval import LabelRetrievalEvaluator


# Simplified ViewAugmenter
class ViewAugmenter:
    """Create multiple augmented views."""
    
    def __init__(
        self,
        num_views: int = 2,
        noise_std: float = 0.1,
        image_shape: Tuple[int, int, int] = (1, 28, 28),
        device: str = "cuda",
        normalize_mean: float = 0.1307,
        normalize_std: float = 0.3081,
        **kwargs,
    ):
        self.num_views = num_views
        self.noise_std = noise_std
        self.image_shape = image_shape
        self.device = device
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        
        self.spatial_transform = v2.RandomAffine(
            degrees=kwargs.get("rotation_deg", 20.0),
            translate=(
                kwargs.get("translation_px", 3) / image_shape[1],
                kwargs.get("translation_px", 3) / image_shape[2],
            ),
            scale=kwargs.get("scale_range", (0.9, 1.1)),
            interpolation=v2.InterpolationMode.BILINEAR,
        )
        
        self.color_transform = v2.Compose([
            v2.ColorJitter(
                brightness=kwargs.get("brightness_range", (0.85, 1.15)),
                contrast=kwargs.get("contrast_range", (0.8, 1.2)),
            ),
            v2.RandomApply(
                [v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                p=kwargs.get("blur_prob", 0.25),
            ),
        ])
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, input_dim) -> (B, num_views, input_dim)"""
        b, _ = x.shape
        channels, height, width = self.image_shape
        x_img = x.view(b, channels, height, width).to(self.device)
        
        views = []
        for _ in range(self.num_views):
            view = self.spatial_transform(x_img)
            view = self.color_transform(view)
            view = view.view(b, -1)
            
            if self.noise_std > 0:
                view = view + torch.randn_like(view) * self.noise_std
            
            view = (view - self.normalize_mean) / self.normalize_std
            views.append(view.unsqueeze(1))
        
        return torch.cat(views, dim=1)


def create_mnist_transform(image_shape: Tuple[int, int, int] = (1, 28, 28)):
    """Create transform for MNIST."""
    _, height, width = image_shape
    transforms = [v2.ToTensor()]
    if height != 28 or width != 28:
        transforms.insert(0, v2.Resize((height, width)))
    return v2.Compose(transforms)


@dataclass
class ExperimentConfig:
    """Configuration for multimodal retrieval experiment."""
    
    # Model architecture
    emb_dim: int = 192  # Transformer embedding dimension
    proj_dim: int = 64  # Projection head output dimension
    num_classes: int = 10
    lamb: float = 0.05
    depth: int = 6  # Number of transformer blocks
    num_heads: int = 3  # Number of attention heads
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.1
    use_cls_token: bool = False
    
    # Data
    image_shape: Tuple[int, int, int] = (1, 16, 16)
    patch_size: int = 8  # Patch size (32x32 image with 4x4 patches = 64 patches)
    normalize_mean: float = 0.1307
    normalize_std: float = 0.3081
    
    # Training
    num_epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    use_multimodal: bool = True
    
    # Augmentation
    num_views: int = 2
    noise_std: float = 0.1
    rotation_deg: float = 20.0
    translation_px: int = 3
    scale_range: Tuple[float, float] = (0.9, 1.1)
    brightness_range: Tuple[float, float] = (0.85, 1.15)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    blur_prob: float = 0.25
    label_drop_prob: float = 0.5
    
    # Evaluation
    eval_every: int = 5
    linear_probe_epochs: int = 20
    linear_probe_lr: float = 1e-2
    pca_dim: int = 32
    mi_k: int = 5
    info_eval_max_samples: Optional[int] = None
    attention_rollout_alpha: float = 0.5
    compute_attention_rollout: bool = True
    
    # System
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42
    data_dir: str = "./data"
    
    @property
    def input_dim(self):
        return self.image_shape[0] * self.image_shape[1] * self.image_shape[2]


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging."""
    logger = logging.getLogger("multimodal_jepa")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fh = logging.FileHandler(output_dir / "training.log")
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def set_seed(seed: int):
    """Set random seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mnist_data(config: ExperimentConfig):
    """Load MNIST data."""
    transform = create_mnist_transform(config.image_shape)
    
    train_dataset = datasets.MNIST(
        root=config.data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=config.data_dir, train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    
    # Prepare test data
    test_images = []
    test_labels = []
    for imgs, labels in test_loader:
        test_images.append(imgs.view(imgs.shape[0], -1))
        test_labels.append(labels)
    test_images = torch.cat(test_images, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    
    return train_loader, test_loader, test_images, test_labels


def extract_features(model: nn.Module, data: torch.Tensor, batch_size: int, device: str):
    """Extract features for linear probe."""
    model.eval()
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features = []
    
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            emb, _ = model.encoder.encode_image(batch)
            feats = emb[:, 0, :]
            features.append(feats.detach().cpu())
    
    return torch.cat(features, dim=0)


def train_linear_probe(
    model, train_data, train_labels, test_data, test_labels,
    epochs: int, lr: float, batch_size: int, device: str
):
    """Train linear probe."""
    train_features = extract_features(model, train_data, batch_size, device)
    test_features = extract_features(model, test_data, batch_size, device)
    
    feat_dim = train_features.shape[1]
    probe = nn.Sequential(
        nn.BatchNorm1d(feat_dim, affine=False),
        nn.Linear(feat_dim, 10),
    ).to(device)
    
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    probe.train()
    for _ in range(epochs):
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(probe(feats), labels)
            loss.backward()
            optimizer.step()
    
    probe.eval()
    with torch.no_grad():
        logits = probe(test_features.to(device))
        preds = logits.argmax(dim=1)
        acc = (preds == test_labels.to(device)).float().mean().item()
    
    return acc


def initialize_metrics_log(log_path: Path):
    """Initialize CSV log."""
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "loss_total", "loss_inv", "loss_sigreg",
            "linear_probe_acc",
            "retrieval_m1_acc", "retrieval_m1_top3", "retrieval_m1_sim",
            "retrieval_m2_acc", "retrieval_m2_top3", "retrieval_m2_sim",
            "mi_M_Z0", "mi_M_Z1",
            "cmi_M_Z1_given_Z0", "cmi_Z0_Z1_given_M",
            "attn_L_to_P", "attn_P_to_L",
            "rollout_L_to_P", "rollout_P_to_L",
        ])


def append_metrics_log(
    log_path,
    epoch,
    loss_dict,
    linear_acc=None,
    retrieval_results=None,
    info_metrics: Optional[Dict[str, float]] = None,
):
    """Append to CSV log."""
    with log_path.open("a", newline="") as f:
        writer = csv.writer(f)
        m1 = retrieval_results.get("method1", {}) if retrieval_results else {}
        m2 = retrieval_results.get("method2", {}) if retrieval_results else {}
        info_metrics = info_metrics or {}
        writer.writerow([
            epoch, loss_dict["total"], loss_dict["inv"], loss_dict["sigreg"],
            linear_acc,
            m1.get("accuracy"), m1.get("top3_accuracy"), m1.get("mean_correct_similarity"),
            m2.get("accuracy"), m2.get("top3_accuracy"), m2.get("mean_correct_similarity"),
            info_metrics.get("mi_M_Z0"), info_metrics.get("mi_M_Z1"),
            info_metrics.get("cmi_M_Z1_given_Z0"), info_metrics.get("cmi_Z0_Z1_given_M"),
            info_metrics.get("attn_L_to_P"), info_metrics.get("attn_P_to_L"),
            info_metrics.get("rollout_L_to_P"), info_metrics.get("rollout_P_to_L"),
        ])


def _standardize_and_pca(z: np.ndarray, n_components: int) -> np.ndarray:
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z)
    if n_components is None or z_scaled.shape[1] <= n_components:
        return z_scaled
    pca = PCA(n_components=n_components, random_state=0)
    return pca.fit_transform(z_scaled)


def _ksg_mi(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """KSG MI estimator for continuous-continuous variables."""
    n = x.shape[0]
    if n <= k:
        return float("nan")

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    joint = np.concatenate([x, y], axis=1)

    nbrs_joint = NearestNeighbors(metric="chebyshev", n_neighbors=k + 1).fit(joint)
    dist, _ = nbrs_joint.kneighbors(joint)
    eps = dist[:, k] - 1e-15

    nbrs_x = NearestNeighbors(metric="chebyshev").fit(x)
    nbrs_y = NearestNeighbors(metric="chebyshev").fit(y)
    nx = np.array([len(idx) - 1 for idx in nbrs_x.radius_neighbors(x, eps, return_distance=False)])
    ny = np.array([len(idx) - 1 for idx in nbrs_y.radius_neighbors(y, eps, return_distance=False)])

    nx = torch.tensor(nx + 1, dtype=torch.float32)
    ny = torch.tensor(ny + 1, dtype=torch.float32)
    return float(torch.digamma(torch.tensor(k, dtype=torch.float32))
                 + torch.digamma(torch.tensor(n, dtype=torch.float32))
                 - (torch.digamma(nx) + torch.digamma(ny)).mean())


def _mixed_mi_discrete_continuous(m: np.ndarray, z: np.ndarray, k: int = 5) -> float:
    """Ross (2014) kNN MI estimator for discrete-continuous variables."""
    m = np.asarray(m)
    z = np.asarray(z, dtype=np.float64)
    n = z.shape[0]
    unique, counts = np.unique(m, return_counts=True)
    min_count = counts.min()
    if min_count <= 1:
        return float("nan")
    k_eff = min(k, min_count - 1)
    if k_eff < 1:
        return float("nan")

    eps = np.zeros(n, dtype=np.float64)
    for val in unique:
        idx = np.where(m == val)[0]
        z_sub = z[idx]
        nbrs = NearestNeighbors(metric="chebyshev", n_neighbors=k_eff + 1).fit(z_sub)
        dist, _ = nbrs.kneighbors(z_sub)
        eps[idx] = dist[:, k_eff] - 1e-15

    nbrs_all = NearestNeighbors(metric="chebyshev").fit(z)
    m_counts = np.array([len(ids) - 1 for ids in nbrs_all.radius_neighbors(z, eps, return_distance=False)])

    n_x = np.array([counts[np.where(unique == val)[0][0]] for val in m])
    n_x = torch.tensor(n_x, dtype=torch.float32)
    m_counts = torch.tensor(m_counts + 1, dtype=torch.float32)

    return float(
        torch.digamma(torch.tensor(k_eff, dtype=torch.float32))
        + torch.digamma(torch.tensor(n, dtype=torch.float32))
        - (torch.digamma(n_x) + torch.digamma(m_counts)).mean()
    )


def _collect_test_embeddings(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    max_samples: Optional[int] = None,
    with_attention: bool = True,
    rollout_alpha: float = 0.5,
    compute_rollout: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Collect Z0, Z1 embeddings and attention flow metrics from test set."""
    model.eval()
    z0_list: List[np.ndarray] = []
    z1_list: List[np.ndarray] = []
    m_list: List[np.ndarray] = []
    attn_metrics = {
        "attn_L_to_P": 0.0,
        "attn_P_to_L": 0.0,
        "rollout_L_to_P": 0.0,
        "rollout_P_to_L": 0.0,
        "num_samples": 0,
    }

    num_patches = model.encoder.num_patches
    has_cls = model.encoder.backbone.use_cls_token
    label_index = (1 if has_cls else 0) + num_patches
    patch_indices = np.arange((1 if has_cls else 0), (1 if has_cls else 0) + num_patches)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.shape[0], -1).to(device)
            labels = labels.to(device)

            emb0, _ = model.encoder.encode_image(images)
            emb1, _ = model.encoder.encode_multimodal(images, labels)

            z0_list.append(emb0.squeeze(1).cpu().numpy())
            z1_list.append(emb1.squeeze(1).cpu().numpy())
            m_list.append(labels.cpu().numpy())

            if with_attention:
                attn_maps = model.encoder.backbone.get_attention_maps()
                if all(attn is not None for attn in attn_maps):
                    attn_l2p, attn_p2l, roll_l2p, roll_p2l = _compute_attention_flow(
                        attn_maps,
                        label_index=label_index,
                        patch_indices=patch_indices,
                        rollout_alpha=rollout_alpha,
                        compute_rollout=compute_rollout,
                    )
                    batch_size = images.size(0)
                    attn_metrics["attn_L_to_P"] += attn_l2p * batch_size
                    attn_metrics["attn_P_to_L"] += attn_p2l * batch_size
                    attn_metrics["rollout_L_to_P"] += roll_l2p * batch_size
                    attn_metrics["rollout_P_to_L"] += roll_p2l * batch_size
                    attn_metrics["num_samples"] += batch_size

            if max_samples is not None and sum(len(arr) for arr in m_list) >= max_samples:
                break

    z0 = np.concatenate(z0_list, axis=0)
    z1 = np.concatenate(z1_list, axis=0)
    m = np.concatenate(m_list, axis=0)

    if max_samples is not None and z0.shape[0] > max_samples:
        z0 = z0[:max_samples]
        z1 = z1[:max_samples]
        m = m[:max_samples]

    if attn_metrics["num_samples"] > 0:
        for key in ["attn_L_to_P", "attn_P_to_L", "rollout_L_to_P", "rollout_P_to_L"]:
            attn_metrics[key] /= attn_metrics["num_samples"]

    return z0, z1, m, attn_metrics


def _compute_attention_flow(
    attn_maps: List[torch.Tensor],
    label_index: int,
    patch_indices: np.ndarray,
    rollout_alpha: float = 0.5,
    compute_rollout: bool = True,
) -> Tuple[float, float, float, float]:
    """Compute attention flow metrics from attention maps."""
    attn_l2p_vals = []
    attn_p2l_vals = []
    rollout_l2p_vals = []
    rollout_p2l_vals = []

    for attn in attn_maps:
        patch_idx = torch.tensor(patch_indices, device=attn.device)
        attn_l2p = attn[:, :, label_index, patch_idx].sum(dim=-1)
        attn_p2l = attn[:, :, patch_idx, label_index].mean(dim=-1)
        attn_l2p_vals.append(attn_l2p.mean(dim=1))
        attn_p2l_vals.append(attn_p2l.mean(dim=1))

    attn_l2p_epoch = torch.cat(attn_l2p_vals, dim=0).mean().item()
    attn_p2l_epoch = torch.cat(attn_p2l_vals, dim=0).mean().item()

    if compute_rollout:
        attn_roll = []
        for attn in attn_maps:
            attn_avg = attn.mean(dim=1)
            eye = torch.eye(attn_avg.size(-1), device=attn_avg.device).unsqueeze(0)
            attn_tilde = rollout_alpha * attn_avg + (1 - rollout_alpha) * eye
            attn_tilde = attn_tilde / attn_tilde.sum(dim=-1, keepdim=True)
            attn_roll.append(attn_tilde)

        rollout = attn_roll[0]
        for mat in attn_roll[1:]:
            rollout = mat @ rollout

        patch_idx = torch.tensor(patch_indices, device=rollout.device)
        roll_l2p = rollout[:, label_index, patch_idx].sum(dim=-1)
        roll_p2l = rollout[:, patch_idx, label_index].mean(dim=-1)
        rollout_l2p_epoch = roll_l2p.mean().item()
        rollout_p2l_epoch = roll_p2l.mean().item()
    else:
        rollout_l2p_epoch = float("nan")
        rollout_p2l_epoch = float("nan")

    return attn_l2p_epoch, attn_p2l_epoch, rollout_l2p_epoch, rollout_p2l_epoch


def evaluate_information_metrics(
    model: nn.Module,
    test_loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Compute MI/CMI metrics and attention flows on the test set."""
    z0, z1, m, attn_metrics = _collect_test_embeddings(
        model,
        test_loader,
        device,
        max_samples=config.info_eval_max_samples,
        with_attention=True,
        rollout_alpha=config.attention_rollout_alpha,
        compute_rollout=config.compute_attention_rollout,
    )

    z0_red = _standardize_and_pca(z0, config.pca_dim)
    z1_red = _standardize_and_pca(z1, config.pca_dim)

    mi_m_z0 = _mixed_mi_discrete_continuous(m, z0_red, k=config.mi_k)
    mi_m_z1 = _mixed_mi_discrete_continuous(m, z1_red, k=config.mi_k)
    mi_m_z0z1 = _mixed_mi_discrete_continuous(m, np.concatenate([z0_red, z1_red], axis=1), k=config.mi_k)
    cmi_m_z1_given_z0 = mi_m_z0z1 - mi_m_z0

    cmi_z0_z1_given_m_vals = []
    for label in np.unique(m):
        idx = np.where(m == label)[0]
        if idx.size <= config.mi_k:
            continue
        mi_val = _ksg_mi(z0_red[idx], z1_red[idx], k=config.mi_k)
        cmi_z0_z1_given_m_vals.append((idx.size / m.size) * mi_val)
    cmi_z0_z1_given_m = float(np.nansum(cmi_z0_z1_given_m_vals)) if cmi_z0_z1_given_m_vals else float("nan")

    return {
        "mi_M_Z0": mi_m_z0,
        "mi_M_Z1": mi_m_z1,
        "cmi_M_Z1_given_Z0": cmi_m_z1_given_z0,
        "cmi_Z0_Z1_given_M": cmi_z0_z1_given_m,
        "attn_L_to_P": attn_metrics.get("attn_L_to_P"),
        "attn_P_to_L": attn_metrics.get("attn_P_to_L"),
        "rollout_L_to_P": attn_metrics.get("rollout_L_to_P"),
        "rollout_P_to_L": attn_metrics.get("rollout_P_to_L"),
    }


def train_epoch(model, train_loader, optimizer, augmenter, config, device):
    """Train one epoch."""
    model.train()
    total_loss = total_inv = total_sigreg = 0.0
    num_batches = 0
    
    for images, labels in train_loader:
        images = images.view(images.shape[0], -1).to(device)
        labels = labels.to(device)
        views = augmenter(images)
        
        optimizer.zero_grad()
        loss_dict = model.compute_loss(
            views, 
            labels=labels if config.use_multimodal else None,
            use_multimodal=config.use_multimodal,
            label_drop_prob=config.label_drop_prob,
        )
        
        loss_dict["total"].backward()
        optimizer.step()
        
        total_loss += loss_dict["total"].item()
        total_inv += loss_dict["inv"].item()
        total_sigreg += loss_dict["sigreg"].item()
        num_batches += 1
    
    return {
        "total": total_loss / num_batches,
        "inv": total_inv / num_batches,
        "sigreg": total_sigreg / num_batches,
    }


def evaluate_retrieval(model, test_loader, config, device, max_samples=1000):
    """Evaluate retrieval using methods from retrieval_eval."""
    model.eval()

    batch_size = test_loader.batch_size or config.batch_size
    max_batches = None
    if max_samples is not None:
        max_batches = max(1, math.ceil(max_samples / batch_size))

    evaluator = LabelRetrievalEvaluator(model, num_classes=config.num_classes, device=device)
    return evaluator.evaluate_retrieval(test_loader, max_batches=max_batches, use_emb=True)


def run_experiment(config: ExperimentConfig, output_dir: Path, logger):
    """Run experiment."""
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    logger.info("=" * 80)
    logger.info("MULTIMODAL JEPA WITH ATTENTION-BASED INTERACTION")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Multimodal training: {config.use_multimodal}")
    logger.info(f"Image shape: {config.image_shape}")
    logger.info(f"Patch size: {config.patch_size}")
    logger.info(f"Transformer: depth={config.depth}, heads={config.num_heads}, dim={config.emb_dim}")
    logger.info("=" * 80)
    
    logger.info("[1] Loading MNIST...")
    train_loader, test_loader, test_images, test_labels = load_mnist_data(config)
    logger.info(f"Train: {len(train_loader.dataset)}, Test: {len(test_images)}")
    
    logger.info("[2] Initializing model...")
    model = MultimodalLeJEPAModel(
        input_dim=config.input_dim,
        emb_dim=config.emb_dim,
        proj_dim=config.proj_dim,
        num_classes=config.num_classes,
        lamb=config.lamb,
        use_vit=True,
        image_shape=config.image_shape,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        patch_size=config.patch_size,
        drop_path_rate=config.drop_path_rate,
        use_cls_token=config.use_cls_token,
    ).to(device)
    
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    augmenter = ViewAugmenter(
        num_views=config.num_views,
        noise_std=config.noise_std,
        image_shape=config.image_shape,
        device=device,
        normalize_mean=config.normalize_mean,
        normalize_std=config.normalize_std,
        rotation_deg=config.rotation_deg,
        translation_px=config.translation_px,
        scale_range=config.scale_range,
        brightness_range=config.brightness_range,
        contrast_range=config.contrast_range,
        blur_prob=config.blur_prob,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate) \
        if config.optimizer.lower() == "adam" else \
        torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    
    metrics_log = output_dir / "metrics.csv"
    initialize_metrics_log(metrics_log)
    
    logger.info("[3] Training...")
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        loss_dict = train_epoch(model, train_loader, optimizer, augmenter, config, device)
        epoch_time = time.time() - epoch_start
        
        logger.info(
            f"Epoch {epoch+1}/{config.num_epochs} | "
            f"Loss: {loss_dict['total']:.4f} | "
            f"Inv: {loss_dict['inv']:.4f} | "
            f"SIG: {loss_dict['sigreg']:.4f} | "
            f"{epoch_time:.1f}s"
        )
        
        linear_acc = None
        retrieval_results = None
        info_metrics = None

        logger.info("  Computing information + attention metrics on test set...")
        info_metrics = evaluate_information_metrics(model, test_loader, config, device)
        logger.info(
            "  MI(M;Z0): {:.4f} | MI(M;Z1): {:.4f} | CMI(M;Z1|Z0): {:.4f} | CMI(Z0;Z1|M): {:.4f}".format(
                info_metrics["mi_M_Z0"],
                info_metrics["mi_M_Z1"],
                info_metrics["cmi_M_Z1_given_Z0"],
                info_metrics["cmi_Z0_Z1_given_M"],
            )
        )
        logger.info(
            "  Attn L->P: {:.4f} | P->L: {:.4f} | Rollout L->P: {:.4f} | Rollout P->L: {:.4f}".format(
                info_metrics["attn_L_to_P"],
                info_metrics["attn_P_to_L"],
                info_metrics["rollout_L_to_P"],
                info_metrics["rollout_P_to_L"],
            )
        )
        
        if (epoch + 1) % config.eval_every == 0 or epoch == config.num_epochs - 1:
            logger.info(f"  Evaluating...")
            
            train_data = []
            train_labels_list = []
            for imgs, lbls in train_loader:
                train_data.append(imgs.view(imgs.shape[0], -1))
                train_labels_list.append(lbls)
            train_data = torch.cat(train_data, dim=0)
            train_labels_tensor = torch.cat(train_labels_list, dim=0)
            
            linear_acc = train_linear_probe(
                model, train_data, train_labels_tensor, test_images, test_labels,
                config.linear_probe_epochs, config.linear_probe_lr, 256, device
            )
            
            retrieval_results = evaluate_retrieval(
                model, test_loader, config, device, 1000
            )
            
            logger.info(f"  Linear: {linear_acc:.4f}")
            logger.info(
                f"  M1: {retrieval_results['method1']['accuracy']:.4f} "
                f"(top3: {retrieval_results['method1']['top3_accuracy']:.4f})"
            )
            logger.info(
                f"  M2: {retrieval_results['method2']['accuracy']:.4f} "
                f"(top3: {retrieval_results['method2']['top3_accuracy']:.4f})"
            )
        
        append_metrics_log(metrics_log, epoch + 1, loss_dict, linear_acc, retrieval_results, info_metrics)
        
        if (epoch + 1) % 10 == 0 or epoch == config.num_epochs - 1:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pt")
    
    logger.info("=" * 80)
    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no_multimodal", action="store_true")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=3)
    parser.add_argument("--emb_dim", type=int, default=192)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--label_drop_prob", type=float, default=0.5)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--pca_dim", type=int, default=32)
    parser.add_argument("--mi_k", type=int, default=5)
    parser.add_argument("--info_eval_max_samples", type=int, default=None)
    parser.add_argument("--attention_rollout_alpha", type=float, default=0.5)
    parser.add_argument("--no_attention_rollout", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    
    config = ExperimentConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_multimodal=not args.no_multimodal,
        depth=args.depth,
        num_heads=args.num_heads,
        emb_dim=args.emb_dim,
        patch_size=args.patch_size,
        label_drop_prob=args.label_drop_prob,
        eval_every=args.eval_every,
        pca_dim=args.pca_dim,
        mi_k=args.mi_k,
        info_eval_max_samples=args.info_eval_max_samples,
        attention_rollout_alpha=args.attention_rollout_alpha,
        compute_attention_rollout=not args.no_attention_rollout,
        seed=args.seed,
        data_dir=args.data_dir,
    )
    
    run_experiment(config, output_dir, logger)


if __name__ == "__main__":
    main()