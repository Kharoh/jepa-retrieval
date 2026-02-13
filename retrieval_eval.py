"""Label retrieval evaluation for multimodal JEPA with attention."""

from __future__ import annotations

from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class LabelRetrievalEvaluator:
    """Evaluator for label retrieval experiments."""
    
    def __init__(self, model, num_classes: int = 10, device: str = "cuda"):
        self.model = model
        self.num_classes = num_classes
        self.device = device
        
    @torch.no_grad()
    def retrieval_method_1(
        self,
        images: torch.Tensor,
        true_labels: torch.Tensor,
        use_emb: bool = True
    ) -> Dict[str, float]:
        """Retrieval Method 1: Image-only vs Image+Label matching.
        
        For each image:
        1. Encode image only -> get embedding
        2. For each possible label, encode image+label -> get embedding  
        3. Find which image+label embedding is closest to image-only
        4. Compare retrieved label to ground truth
        
        The label token interacts with patches through attention!
        """
        self.model.eval()
        images = images.to(self.device)
        true_labels = true_labels.to(self.device)
        
        B = images.shape[0]
        
        # Encode image only (no label token in transformer)
        img_emb, img_proj = self.model.encoder.encode_image(images)
        img_feat = img_emb[:, 0, :] if use_emb else img_proj[:, 0, :]  # (B, D)
        
        retrieved_labels = []
        similarities = []
        
        for i in range(B):
            img_i = images[i:i+1]  # (1, V, D)
            img_feat_i = img_feat[i:i+1]  # (1, D)
            
            # Try all possible labels
            all_label_feats = []
            for label in range(self.num_classes):
                label_tensor = torch.tensor([label], device=self.device)
                # Label token is added and interacts via attention
                multimodal_emb, multimodal_proj = self.model.encoder.encode_multimodal(
                    img_i, label_tensor
                )
                # Use mean pooling over all tokens (patches + label after attention)
                feat = multimodal_emb.mean(dim=1) if use_emb else multimodal_proj.mean(dim=1)
                all_label_feats.append(feat)
            
            all_label_feats = torch.cat(all_label_feats, dim=0)  # (num_classes, D)
            
            # Compute similarities
            sims = F.cosine_similarity(
                img_feat_i.expand_as(all_label_feats), 
                all_label_feats, 
                dim=-1
            )
            
            retrieved_label = sims.argmax().item()
            retrieved_labels.append(retrieved_label)
            similarities.append(sims[true_labels[i]].item())
        
        retrieved_labels = torch.tensor(retrieved_labels, device=self.device)
        accuracy = (retrieved_labels == true_labels).float().mean().item()
        
        # Top-k accuracy
        top3_acc = self._compute_topk_accuracy_method1(
            images, true_labels, k=3, use_emb=use_emb
        )
        
        return {
            "accuracy": accuracy,
            "top3_accuracy": top3_acc,
            "mean_correct_similarity": float(np.mean(similarities)),
            "retrieved_labels": retrieved_labels.cpu().numpy(),
        }
    
    @torch.no_grad()
    def retrieval_method_2(
        self,
        images: torch.Tensor,
        true_labels: torch.Tensor,
        use_emb: bool = True
    ) -> Dict[str, float]:
        """Retrieval Method 2: Patch-only vs Label-only embedding matching.
        
        For each image:
        1. Encode image patches only -> get embedding
        2. Encode all possible label tokens only -> get embeddings
        3. Find which label embedding is closest to patch embedding
        4. Compare retrieved label to ground truth
        
        Note: Label embeddings here are standalone, not conditioned on images.
        """
        self.model.eval()
        images = images.to(self.device)
        true_labels = true_labels.to(self.device)
        
        B = images.shape[0]
        
        # Encode images (patches only, no label token)
        img_emb, img_proj = self.model.encoder.encode_image(images)
        img_feat = img_emb.mean(dim=1) if use_emb else img_proj.mean(dim=1)  # (B, D)
        
        # Encode all label tokens (standalone, no image context)
        all_labels = torch.arange(self.num_classes, device=self.device)
        label_emb, label_proj = self.model.encoder.encode_label(all_labels)
        label_feat = label_emb if use_emb else label_proj  # (num_classes, D)
        
        # Compute similarities for all images vs all labels
        sims = F.cosine_similarity(
            img_feat.unsqueeze(1).expand(-1, self.num_classes, -1),
            label_feat.unsqueeze(0).expand(B, -1, -1),
            dim=-1
        )
        
        retrieved_labels = sims.argmax(dim=-1)
        accuracy = (retrieved_labels == true_labels).float().mean().item()
        
        # Top-k accuracy
        topk_indices = sims.topk(k=3, dim=-1).indices
        top3_acc = (topk_indices == true_labels.unsqueeze(1)).any(dim=-1).float().mean().item()
        
        # Similarity for correct labels
        correct_sims = sims[torch.arange(B), true_labels].cpu().numpy()
        
        return {
            "accuracy": accuracy,
            "top3_accuracy": top3_acc,
            "mean_correct_similarity": float(np.mean(correct_sims)),
            "retrieved_labels": retrieved_labels.cpu().numpy(),
            "similarity_matrix": sims.cpu().numpy(),
        }
    
    @torch.no_grad()
    def _compute_topk_accuracy_method1(
        self, 
        images: torch.Tensor, 
        true_labels: torch.Tensor, 
        k: int = 3,
        use_emb: bool = True
    ) -> float:
        """Compute top-k accuracy for method 1."""
        B = images.shape[0]
        img_emb, img_proj = self.model.encoder.encode_image(images)
        img_feat = img_emb[:, 0, :] if use_emb else img_proj[:, 0, :]
        
        correct = 0
        for i in range(B):
            img_i = images[i:i+1]
            img_feat_i = img_feat[i:i+1]
            
            all_label_feats = []
            for label in range(self.num_classes):
                label_tensor = torch.tensor([label], device=self.device)
                multimodal_emb, multimodal_proj = self.model.encoder.encode_multimodal(
                    img_i, label_tensor
                )
                feat = multimodal_emb.mean(dim=1) if use_emb else multimodal_proj.mean(dim=1)
                all_label_feats.append(feat)
            
            all_label_feats = torch.cat(all_label_feats, dim=0)
            sims = F.cosine_similarity(
                img_feat_i.expand_as(all_label_feats), 
                all_label_feats, 
                dim=-1
            )
            
            topk_labels = sims.topk(k=k).indices
            if true_labels[i] in topk_labels:
                correct += 1
        
        return correct / B
    
    def evaluate_retrieval(
        self,
        data_loader: DataLoader,
        max_batches: int = None,
        use_emb: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate both retrieval methods on a dataset."""
        self.model.eval()
        
        method1_results = {"accuracy": [], "top3_accuracy": [], "mean_correct_similarity": []}
        method2_results = {"accuracy": [], "top3_accuracy": [], "mean_correct_similarity": []}
        
        for batch_idx, (images, labels) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            m1 = self.retrieval_method_1(images, labels, use_emb=use_emb)
            method1_results["accuracy"].append(m1["accuracy"])
            method1_results["top3_accuracy"].append(m1["top3_accuracy"])
            method1_results["mean_correct_similarity"].append(m1["mean_correct_similarity"])
            
            m2 = self.retrieval_method_2(images, labels, use_emb=use_emb)
            method2_results["accuracy"].append(m2["accuracy"])
            method2_results["top3_accuracy"].append(m2["top3_accuracy"])
            method2_results["mean_correct_similarity"].append(m2["mean_correct_similarity"])
        
        return {
            "method1": {k: float(np.mean(v)) for k, v in method1_results.items()},
            "method2": {k: float(np.mean(v)) for k, v in method2_results.items()},
        }
    
    def compute_confusion_matrix(
        self,
        images: torch.Tensor,
        true_labels: torch.Tensor,
        method: int = 2,
        use_emb: bool = True
    ) -> np.ndarray:
        """Compute confusion matrix for retrieval."""
        if method == 1:
            results = self.retrieval_method_1(images, true_labels, use_emb=use_emb)
        else:
            results = self.retrieval_method_2(images, true_labels, use_emb=use_emb)
        
        retrieved = results["retrieved_labels"]
        true = true_labels.cpu().numpy()
        
        confusion = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for t, r in zip(true, retrieved):
            confusion[t, r] += 1
        
        return confusion
