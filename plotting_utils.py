"""Plotting utilities for multimodal JEPA results."""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path



def plot_training_curves(metrics_csv: str, output_path: str = None):
    """Plot training loss and retrieval accuracy curves."""
    df = pd.read_csv(metrics_csv)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    ax = axes[0, 0]
    ax.plot(df["epoch"], df["loss_total"], label="Total Loss", marker="o")
    ax.plot(df["epoch"], df["loss_inv"], label="Invariance Loss", marker="s")
    ax.plot(df["epoch"], df["loss_sigreg"], label="SIGReg Loss", marker="^")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Components")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Linear probe accuracy
    ax = axes[0, 1]
    mask = df["linear_probe_acc"].notna()
    ax.plot(df.loc[mask, "epoch"], df.loc[mask, "linear_probe_acc"], 
            label="Linear Probe", marker="o", linewidth=2)
    ax.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5, label="Random (10%)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Linear Probe Accuracy (Representation Quality)")
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Retrieval accuracy - Method 1
    ax = axes[1, 0]
    mask = df["retrieval_m1_acc"].notna()
    ax.plot(df.loc[mask, "epoch"], df.loc[mask, "retrieval_m1_acc"], 
            label="Top-1", marker="o", linewidth=2)
    ax.plot(df.loc[mask, "epoch"], df.loc[mask, "retrieval_m1_top3"], 
            label="Top-3", marker="s", linewidth=2)
    ax.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5, label="Random Top-1 (10%)")
    ax.axhline(y=0.3, color="gray", linestyle=":", alpha=0.5, label="Random Top-3 (30%)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Retrieval Method 1: Image vs Image+Label")
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Retrieval accuracy - Method 2
    ax = axes[1, 1]
    mask = df["retrieval_m2_acc"].notna()
    ax.plot(df.loc[mask, "epoch"], df.loc[mask, "retrieval_m2_acc"], 
            label="Top-1", marker="o", linewidth=2)
    ax.plot(df.loc[mask, "epoch"], df.loc[mask, "retrieval_m2_top3"], 
            label="Top-3", marker="s", linewidth=2)
    ax.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5, label="Random Top-1 (10%)")
    ax.axhline(y=0.3, color="gray", linestyle=":", alpha=0.5, label="Random Top-3 (30%)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Retrieval Method 2: Patch vs Label")
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()



def plot_confusion_matrix(confusion: np.ndarray, output_path: str = None, title: str = "Confusion Matrix"):
    """Plot retrieval confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        confusion, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10),
        ax=ax
    )
    
    ax.set_xlabel("Retrieved Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved confusion matrix to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison(metrics_csv_baseline: str, metrics_csv_multimodal: str, output_path: str = None):
    """Compare baseline (no labels) vs multimodal training."""
    df_baseline = pd.read_csv(metrics_csv_baseline)
    df_multimodal = pd.read_csv(metrics_csv_multimodal)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Method 1 comparison
    ax = axes[0]
    mask = df_baseline["retrieval_m1_acc"].notna()
    ax.plot(df_baseline.loc[mask, "epoch"], df_baseline.loc[mask, "retrieval_m1_acc"], 
            label="Baseline (no labels)", marker="o", linewidth=2)
    
    mask = df_multimodal["retrieval_m1_acc"].notna()
    ax.plot(df_multimodal.loc[mask, "epoch"], df_multimodal.loc[mask, "retrieval_m1_acc"], 
            label="Multimodal training", marker="s", linewidth=2)
    
    ax.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5, label="Random (10%)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Method 1: Image vs Image+Label")
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Method 2 comparison
    ax = axes[1]
    mask = df_baseline["retrieval_m2_acc"].notna()
    ax.plot(df_baseline.loc[mask, "epoch"], df_baseline.loc[mask, "retrieval_m2_acc"], 
            label="Baseline (no labels)", marker="o", linewidth=2)
    
    mask = df_multimodal["retrieval_m2_acc"].notna()
    ax.plot(df_multimodal.loc[mask, "epoch"], df_multimodal.loc[mask, "retrieval_m2_acc"], 
            label="Multimodal training", marker="s", linewidth=2)
    
    ax.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5, label="Random (10%)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Method 2: Patch vs Label")
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot multimodal JEPA results")
    parser.add_argument("--metrics_csv", type=str, required=True, help="Path to metrics CSV")
    parser.add_argument("--output_dir", type=str, default="./plots", help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_training_curves(
        args.metrics_csv,
        output_path=str(output_dir / "training_curves.png")
    )
