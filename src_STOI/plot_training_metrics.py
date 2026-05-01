from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def _to_float(value: str) -> Optional[float]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return float(text)


def _read_metrics_csv(path: Path) -> Dict[str, List[Optional[float]]]:
    rows: Dict[str, List[Optional[float]]] = {
        "epoch": [],
        "train_loss": [],
        "train_mae": [],
        "val_loss": [],
        "val_mae": [],
        "val_corr": [],
        "lr": [],
        "best_score": [],
        "new_best": [],
    }
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows["epoch"].append(_to_float(row.get("epoch", "")))
            rows["train_loss"].append(_to_float(row.get("train_loss", "")))
            rows["train_mae"].append(_to_float(row.get("train_mae", "")))
            rows["val_loss"].append(_to_float(row.get("val_loss", "")))
            rows["val_mae"].append(_to_float(row.get("val_mae", "")))
            rows["val_corr"].append(_to_float(row.get("val_corr", "")))
            rows["lr"].append(_to_float(row.get("lr", "")))
            rows["best_score"].append(_to_float(row.get("best_score", "")))
            rows["new_best"].append(_to_float(row.get("new_best", "")))
    return rows


def _scatter_new_best(ax: plt.Axes, x: List[float], y: List[Optional[float]], new_best: List[Optional[float]]) -> None:
    xb: List[float] = []
    yb: List[float] = []
    for xi, yi, bi in zip(x, y, new_best):
        if yi is None or bi is None:
            continue
        if bi >= 0.5:
            xb.append(xi)
            yb.append(yi)
    if xb:
        ax.scatter(xb, yb, marker="*", s=110, label="new best", zorder=3)


def _plot_line(
    ax: plt.Axes,
    x: List[float],
    y: List[Optional[float]],
    label: str,
    *,
    linewidth: float = 2.0,
) -> bool:
    xp: List[float] = []
    yp: List[float] = []
    for xi, yi in zip(x, y):
        if yi is None:
            continue
        xp.append(xi)
        yp.append(yi)
    if not xp:
        return False
    ax.plot(xp, yp, label=label, linewidth=linewidth)
    return True


def build_plots(metrics_csv: Path, out_dir: Path, dpi: int) -> None:
    rows = _read_metrics_csv(metrics_csv)
    epochs = [e for e in rows["epoch"] if e is not None]
    if not epochs:
        raise ValueError(f"CSV пустой или без колонок epoch: {metrics_csv}")
    x = [float(e) for e in epochs]

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Loss
    fig, ax = plt.subplots(figsize=(8, 5))
    has_train = _plot_line(ax, x, rows["train_loss"], "train_loss")
    has_val = _plot_line(ax, x, rows["val_loss"], "val_loss")
    _scatter_new_best(ax, x, rows["val_loss"] if has_val else rows["train_loss"], rows["new_best"])
    ax.set_title("Training / Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    if has_train or has_val:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_loss.png", dpi=dpi)
    plt.close(fig)

    # 2) MAE
    fig, ax = plt.subplots(figsize=(8, 5))
    has_train = _plot_line(ax, x, rows["train_mae"], "train_mae")
    has_val = _plot_line(ax, x, rows["val_mae"], "val_mae")
    ax.set_title("Training / Validation MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    ax.grid(alpha=0.25)
    if has_train or has_val:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_mae.png", dpi=dpi)
    plt.close(fig)

    # 3) Correlation (val)
    fig, ax = plt.subplots(figsize=(8, 5))
    has_corr = _plot_line(ax, x, rows["val_corr"], "val_corr")
    ax.set_title("Validation Correlation (r)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Correlation")
    ax.grid(alpha=0.25)
    if has_corr:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_val_corr.png", dpi=dpi)
    plt.close(fig)

    # 4) Learning rate
    fig, ax = plt.subplots(figsize=(8, 5))
    has_lr = _plot_line(ax, x, rows["lr"], "lr")
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.grid(alpha=0.25)
    if has_lr:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_lr.png", dpi=dpi)
    plt.close(fig)

    # 5) Overview (2x2)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    _plot_line(axs[0, 0], x, rows["train_loss"], "train_loss")
    _plot_line(axs[0, 0], x, rows["val_loss"], "val_loss")
    axs[0, 0].set_title("Loss")
    axs[0, 0].grid(alpha=0.25)
    axs[0, 0].legend()

    _plot_line(axs[0, 1], x, rows["train_mae"], "train_mae")
    _plot_line(axs[0, 1], x, rows["val_mae"], "val_mae")
    axs[0, 1].set_title("MAE")
    axs[0, 1].grid(alpha=0.25)
    axs[0, 1].legend()

    _plot_line(axs[1, 0], x, rows["val_corr"], "val_corr")
    axs[1, 0].set_title("Validation Correlation")
    axs[1, 0].grid(alpha=0.25)
    axs[1, 0].legend()

    _plot_line(axs[1, 1], x, rows["lr"], "lr")
    axs[1, 1].set_title("Learning Rate")
    axs[1, 1].grid(alpha=0.25)
    axs[1, 1].legend()

    for ax in axs.reshape(-1):
        ax.set_xlabel("Epoch")

    fig.tight_layout()
    fig.savefig(out_dir / "training_overview.png", dpi=dpi)
    plt.close(fig)


def main() -> None:
    default_csv = Path("./checkpoints_src_stoi_net/train_metrics.csv")
    default_out = Path("./checkpoints_src_stoi_net/plots")

    parser = argparse.ArgumentParser(description="Build training plots from train_metrics.csv")
    parser.add_argument("--metrics-csv", type=Path, default=default_csv, help=f"Path to CSV (default: {default_csv})")
    parser.add_argument("--out-dir", type=Path, default=default_out, help=f"Directory for PNG plots (default: {default_out})")
    parser.add_argument("--dpi", type=int, default=160, help="PNG DPI (default: 160)")
    args = parser.parse_args()

    metrics_csv = args.metrics_csv.expanduser()
    out_dir = args.out_dir.expanduser()
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")

    build_plots(metrics_csv=metrics_csv, out_dir=out_dir, dpi=args.dpi)
    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
