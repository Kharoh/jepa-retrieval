"""Plot per-epoch information and attention metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def _plot_lines(df: pd.DataFrame, columns: List[str], title: str, ylabel: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for col in columns:
        if col not in df.columns:
            continue
        mask = df[col].notna()
        if mask.any():
            ax.plot(df.loc[mask, "epoch"], df.loc[mask, col], marker="o", label=col)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(metrics_csv: str, output_dir: str) -> None:
    df = pd.read_csv(metrics_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_lines(
        df,
        ["mi_M_Z0", "mi_M_Z1"],
        "Mutual Information Diagnostics",
        "MI (nats)",
        out_dir / "mi_diagnostics.png",
    )

    _plot_lines(
        df,
        ["cmi_M_Z1_given_Z0", "cmi_Z0_Z1_given_M"],
        "Conditional Mutual Information",
        "CMI (nats)",
        out_dir / "cmi_metrics.png",
    )

    _plot_lines(
        df,
        ["attn_L_to_P", "attn_P_to_L"],
        "Attention Flow (Direct)",
        "Attention mass",
        out_dir / "attention_flow_direct.png",
    )

    if "rollout_L_to_P" in df.columns or "rollout_P_to_L" in df.columns:
        _plot_lines(
            df,
            ["rollout_L_to_P", "rollout_P_to_L"],
            "Attention Flow (Rollout)",
            "Attention mass",
            out_dir / "attention_flow_rollout.png",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot MI/CMI and attention metrics")
    parser.add_argument("--metrics_csv", type=str, required=True, help="Path to metrics CSV")
    parser.add_argument("--output_dir", type=str, default="./plots", help="Directory to save plots")
    args = parser.parse_args()

    plot_metrics(args.metrics_csv, args.output_dir)
