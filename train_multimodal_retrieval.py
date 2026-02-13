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
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import v2

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
        ])


def append_metrics_log(log_path, epoch, loss_dict, linear_acc=None, retrieval_results=None):
    """Append to CSV log."""
    with log_path.open("a", newline="") as f:
        writer = csv.writer(f)
        m1 = retrieval_results.get("method1", {}) if retrieval_results else {}
        m2 = retrieval_results.get("method2", {}) if retrieval_results else {}
        writer.writerow([
            epoch, loss_dict["total"], loss_dict["inv"], loss_dict["sigreg"],
            linear_acc,
            m1.get("accuracy"), m1.get("top3_accuracy"), m1.get("mean_correct_similarity"),
            m2.get("accuracy"), m2.get("top3_accuracy"), m2.get("mean_correct_similarity"),
        ])


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
        
        append_metrics_log(metrics_log, epoch + 1, loss_dict, linear_acc, retrieval_results)
        
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
        seed=args.seed,
        data_dir=args.data_dir,
    )
    
    run_experiment(config, output_dir, logger)


if __name__ == "__main__":
    main()