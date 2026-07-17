"""
src/utils.py
Shared helper utilities used across all modules.

Includes:
- Directory setup & verification
- Loss / metric curve plotting
- Scaler save/load wrappers
- Logging helpers
"""

import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# ── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a configured logger with timestamped console output.

    Args:
        name:  Logger name (typically __name__ of the calling module).
        level: Logging level (default INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── Directory Setup ───────────────────────────────────────────────────────────

def ensure_dirs(*paths: str) -> None:
    """
    Create directories if they do not already exist.

    Args:
        *paths: One or more directory paths to create.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)


def verify_project_structure(base_dir: str = ".") -> dict:
    """
    Check that required project directories exist and report status.

    Args:
        base_dir: Root of the fraud_detection_project (default current dir).

    Returns:
        Dict mapping directory name -> exists (bool).
    """
    required = [
        "data/raw",
        "data/processed",
        "models",
        "notebooks",
        "src",
    ]
    status = {}
    for rel in required:
        full = os.path.join(base_dir, rel)
        status[rel] = os.path.isdir(full)
    return status


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_loss_curves(
    train_losses: list,
    val_losses: list,
    title: str,
    save_dir: str = "plots"
) -> None:
    """
    Plot training vs. validation loss curves and save the figure.

    A diverging gap (train ↓, val ↑) indicates overfitting.
    Both curves decreasing together is healthy.

    Args:
        train_losses: List of per-epoch training losses.
        val_losses:   List of per-epoch validation losses.
        title:        Module title used in the plot and filename.
        save_dir:     Directory where the PNG is saved.
    """
    ensure_dirs(save_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Training Loss",   color="#4C72B0", linewidth=2)
    ax.plot(val_losses,   label="Validation Loss", color="#DD8452", linewidth=2)
    ax.set_xlabel("Epoch",         fontsize=13)
    ax.set_ylabel("Loss",          fontsize=13)
    ax.set_title(f"{title} — Loss Curves\n(no divergence = no overfitting)", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    fname = os.path.join(save_dir, f"{title.lower().replace(' ', '_')}_loss_curve.png")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved loss curve -> {fname}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    title: str,
    save_dir: str = "plots"
) -> None:
    """
    Plot a labelled confusion matrix heatmap and save it.

    Args:
        cm:           2D confusion matrix array (sklearn.metrics.confusion_matrix output).
        class_names:  List of class labels (e.g. ['Legit', 'Fraud']).
        title:        Used in the plot title and filename.
        save_dir:     Directory where the PNG is saved.
    """
    ensure_dirs(save_dir)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title(f"{title} — Confusion Matrix", fontsize=13)
    fname = os.path.join(save_dir, f"{title.lower().replace(' ', '_')}_confusion_matrix.png")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved confusion matrix -> {fname}")


# ── Scaler Helpers ────────────────────────────────────────────────────────────

def save_scaler(scaler, path: str = "models/scaler.pkl") -> None:
    """Persist a fitted sklearn scaler to disk."""
    ensure_dirs(os.path.dirname(path) or ".")
    joblib.dump(scaler, path)
    print(f"  Scaler saved -> {path}")


def load_scaler(path: str = "models/scaler.pkl"):
    """Load a previously saved sklearn scaler from disk."""
    return joblib.load(path)


# ── Metrics Summary ───────────────────────────────────────────────────────────

def save_metrics(metrics: dict, path: str = "models/metrics.json") -> None:
    """
    Save a dictionary of evaluation metrics as a JSON file.

    Args:
        metrics: Dict of metric names -> values.
        path:    Output JSON file path.
    """
    ensure_dirs(os.path.dirname(path) or ".")
    metrics["timestamp"] = datetime.utcnow().isoformat()
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved -> {path}")


if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("Utils module loaded successfully.")
    status = verify_project_structure(".")
    for d, exists in status.items():
        logger.info(f"  {'✓' if exists else '✗'} {d}")
