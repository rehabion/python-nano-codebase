"""Metrics, comparison tables and publication-ready plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def classification_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    """Compute the standard panel of binary-classification metrics.

    MCC and balanced accuracy are included because nanotoxicity datasets are
    frequently class-imbalanced, where raw accuracy is misleading.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    # Probabilistic metrics need both classes present in y_true.
    if y_score is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["pr_auc"] = average_precision_score(y_true, y_score)
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def leaderboard(results: dict[str, dict]) -> pd.DataFrame:
    """Build a sorted comparison table from per-model result dicts.

    Each value in ``results`` is expected to carry a ``cv_mean`` / ``cv_std``
    mapping and a ``holdout`` metric mapping.
    """
    rows = []
    for name, res in results.items():
        row = {"model": name}
        for k, v in res.get("cv_mean", {}).items():
            row[f"cv_{k}"] = v
        for k, v in res.get("cv_std", {}).items():
            row[f"cv_{k}_std"] = v
        for k, v in res.get("holdout", {}).items():
            row[f"holdout_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    sort_key = "holdout_roc_auc" if "holdout_roc_auc" in df.columns else "cv_roc_auc"
    if sort_key in df.columns:
        df = df.sort_values(sort_key, ascending=False, na_position="last")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Plots (matplotlib, Agg-safe)
# --------------------------------------------------------------------------- #

def _ensure_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_roc_curves(roc_data: dict[str, tuple], out_path: Path) -> None:
    """ROC overlay for every model. ``roc_data[name] = (y_true, y_score)``."""
    plt = _ensure_mpl()
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (y_true, y_score) in roc_data.items():
        if y_score is None or len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (held-out test set)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_model_comparison(board: pd.DataFrame, out_path: Path,
                          metric: str = "holdout_roc_auc") -> None:
    """Bar chart of a chosen metric across models."""
    if metric not in board.columns:
        return
    plt = _ensure_mpl()
    data = board.dropna(subset=[metric]).sort_values(metric)
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(data) + 2))
    ax.barh(data["model"], data[metric], color="#3b7dd8")
    ax.set_xlabel(metric)
    ax.set_xlim(0, 1)
    ax.set_title(f"Model comparison by {metric}")
    for y, v in enumerate(data[metric]):
        ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred, out_path: Path, title: str) -> None:
    """Confusion matrix heat-map for the best model."""
    plt = _ensure_mpl()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["non-toxic", "cytotoxic"])
    ax.set_yticks([0, 1], ["non-toxic", "cytotoxic"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
