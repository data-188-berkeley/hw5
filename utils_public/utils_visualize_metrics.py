from typing import Any, Optional

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


def plot_train_val_test_curves(
    losses_train: np.ndarray,
    losses_val: np.ndarray,
    losses_test: np.ndarray,
    metrics_train: np.ndarray,
    metrics_val: np.ndarray,
    metrics_test: np.ndarray,
    epochs_train: np.ndarray,
    epochs_val: np.ndarray,
    fig: Optional[Figure] = None,
) -> Figure:
    """Plot train/val/test metrics.
    For train/val metrics, assumes that each loss/metric is calculated at end of an epoch.
    Num epochs for train vs val are allowed to be different.

    Args:
        losses_train (np.ndarray):
        losses_val (np.ndarray):
        losses_test (np.ndarray):
        metrics_train (np.ndarray):
        metrics_val (np.ndarray):
        metrics_test (np.ndarray):
        epochs_train:
            len must match losses_train, metrics_train.
        epochs_val:
            len must match losses_val, metrics_val.
        fig: If given, add plots as subplots to this Figure.
            Else, create a new Figure.
    Returns:
        Figure: figure.
    """
    # Plotting the loss curves
    if fig is not None:
        ax_loss = fig.add_subplot(nrows=1, ncols=2)
        ax_metric = fig.add_subplot(nrows=2, ncols=2)
    else:
        # Create new Figure and Axes
        fig, (ax_loss, ax_metric) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    ax_loss.plot(
        epochs_train, losses_train, "r", linewidth=1, marker="o", label="Training loss"
    )
    ax_loss.plot(epochs_val, losses_val, "g", linewidth=1, marker="o", label="Val loss")
    ax_loss.plot(
        epochs_train, losses_test, "b", linewidth=1, marker="o", label="Test loss"
    )

    # Add details to the plot
    ax_loss.set_title("Training/Val/Test Loss", fontsize=12)
    ax_loss.set_xlabel("Epochs", fontsize=10)
    ax_loss.set_ylabel("Loss Value", fontsize=10)
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True)

    # Plot metrics curves
    ax_metric.plot(
        epochs_train,
        metrics_train,
        "r",
        linewidth=1,
        marker="o",
        label="Training metrics",
    )
    ax_metric.plot(
        epochs_val, metrics_val, "g", linewidth=1, marker="o", label="Val metrics"
    )
    ax_metric.plot(
        epochs_train, metrics_test, "b", linewidth=1, marker="o", label="Test metrics"
    )

    # Add details to the plot
    ax_metric.set_title("Training/Val/Test Metrics", fontsize=12)
    ax_metric.set_xlabel("Epochs", fontsize=10)
    ax_metric.set_ylabel("Metric value", fontsize=10)
    ax_metric.legend(fontsize=8)
    ax_metric.grid(True)

    return fig


def plot_train_val_test_meta(
    train_metas: list[dict[str, Any]],
    val_metas: list[dict[str, Any]],
    test_meta: dict[str, Any],
    fig_suptitle: Optional[str] = None,
) -> Figure:
    """Plots train/val/test metrics, such as:
    Loss curves, and metrics (ex: SacreBLEU scores).

    Args:
        train_metas (list[dict[str, Any]]): See: `train_loop()`
        val_metas (list[dict[str, Any]]): See: `train_loop()`
            Note that len(val_metas) doesn't have to match len(train_metas), due to
                `val_every_n_epochs` kwarg for `train_loop()`.
        test_meta (dict[str, Any]): See: `train_loop()`.
        fig_suptitle (Optional[str], optional): If given, provide a suptitle
            for the output Figure.
            Defaults to None.

    Returns:
        Figure: figure.
    """
    # sort by epoch (increasing) to simplify downstream code (just in case)
    train_metas_srt = sorted(train_metas, key=lambda thing: thing["epoch"])
    val_metas_srt = sorted(val_metas, key=lambda thing: thing["epoch"])

    train_losses = []
    train_epochs = []

    for train_meta in train_metas_srt:
        train_losses.append(train_meta["avg_loss"])
        train_epochs.append(train_meta["epoch"])
    val_losses = []
    val_metrics = []
    val_epochs = []
    for val_meta in val_metas_srt:
        val_losses.append(val_meta["avg_loss"])
        val_metrics.append(val_meta["score_sacre_bleu"])
        val_epochs.append(val_meta["epoch"])

    train_losses_np = np.array(train_losses, dtype=np.float32)
    train_metrics_np = np.full_like(train_losses_np, np.nan)
    train_epochs_np = np.array(train_epochs, dtype=np.int64)
    val_losses_np = np.array(val_losses, dtype=np.float32)
    val_metrics_np = np.array(val_metrics, dtype=np.float32)
    val_epochs_np = np.array(val_epochs, dtype=np.int64)

    # Assumes that test loss/metrics are only calculated at end of final epoch
    test_losses_np = np.full_like(train_losses_np, np.nan)
    test_losses_np[-1] = test_meta["avg_loss"]
    test_metrics_np = np.full_like(train_losses_np, np.nan)
    test_metrics_np[-1] = test_meta["score_sacre_bleu"]

    fig = plot_train_val_test_curves(
        losses_train=train_losses_np,
        losses_val=val_losses_np,
        losses_test=test_losses_np,
        metrics_train=train_metrics_np,
        metrics_val=val_metrics_np,
        metrics_test=test_metrics_np,
        epochs_train=train_epochs_np,
        epochs_val=val_epochs_np,
    )
    if fig_suptitle is not None:
        fig.suptitle(fig_suptitle)
    return fig
