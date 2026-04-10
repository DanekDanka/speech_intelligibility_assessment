from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src_STOI.test import arrays_from_pairs, load_prediction_file, pairs_from_payload


def summarize(pred: np.ndarray, tgt: np.ndarray) -> dict:
    err = pred - tgt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    if pred.size > 1:
        c = np.corrcoef(pred, tgt)[0, 1]
        r = float(c) if not np.isnan(c) else float("nan")
    else:
        r = float("nan")
    out = {"mae": mae, "rmse": rmse, "pearson_r": r, "n": int(pred.size)}
    try:
        from scipy.stats import spearmanr

        out["spearman_r"] = float(spearmanr(pred, tgt).correlation)
    except Exception:
        out["spearman_r"] = None
    return out


def plot_scatter(pred: np.ndarray, tgt: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(tgt, pred, s=8, alpha=0.35, edgecolors="none")
    lims = [min(tgt.min(), pred.min()), max(tgt.max(), pred.max())]
    ax.plot(lims, lims, "k--", lw=1, label="ideal")
    ax.set_xlabel("Target STOI")
    ax.set_ylabel("Predicted STOI")
    ax.set_title("Predicted vs target")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_residuals(pred: np.ndarray, tgt: np.ndarray, out_path: Path) -> None:
    err = pred - tgt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(err, bins=40, color="steelblue", edgecolor="white", alpha=0.9)
    ax.axvline(0.0, color="k", ls="--", lw=1)
    ax.set_title("Prediction error (pred − target)")
    ax.set_xlabel("Error")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate STOI prediction file (model-agnostic)")
    p.add_argument("--predictions", type=str, required=True, help="JSON from train.py export")
    p.add_argument("--output-dir", type=str, default="./stoi_eval_plots")
    p.add_argument("--metrics-json", type=str, default=None, help="Write metrics to this path")
    args = p.parse_args()

    payload = load_prediction_file(args.predictions)
    pairs = pairs_from_payload(payload)
    pred, tgt = arrays_from_pairs(pairs)
    metrics = summarize(pred, tgt)

    out_dir = Path(args.output_dir)
    plot_scatter(pred, tgt, out_dir / "scatter_pred_vs_target.png")
    plot_residuals(pred, tgt, out_dir / "residual_hist.png")

    print(json.dumps(metrics, indent=2))
    if args.metrics_json:
        Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
