from typing import Any

from matplotlib import pyplot as plt
import numpy as np


def plot_train_val_metrics(
    train_meta: dict[str, Any],
    val_meta: dict[str, Any],
    suptitle: str = "",
    acc_set_ylim_0_1: bool = True,
):
    """Plots train/val metrics.

    Args:
        train_meta (dict[str, Any]): See: train_loop()
        val_meta (dict[str, Any]): See: train_loop()
        suptitle (str, optional): Figure suptitle. Defaults to "".
        acc_set_ylim_0_1: If True, set the train/val acc y limits from 0.0 to 1.0.
            Useful if we know that the "accuracy" metric is truly bounded from [0.0, 1.0].
            But, sometimes the "accuracy" metric is actually some other metric, so we don't
            always want to do this. (Odd, I know)
    Returns:
        fig:
    """
    fig, axd = plt.subplot_mosaic([["loss", "train_acc", "val_acc"]], figsize=(15, 5))
    if suptitle:
        fig.suptitle(suptitle)

    epochs = np.linspace(0, 1, len(train_meta["losses"])) * train_meta["num_epochs"]

    axd["loss"].plot(epochs, train_meta["losses"])
    axd["loss"].set_title("Train Loss")
    axd["loss"].set_xlabel("Epoch")
    axd["loss"].set_ylabel("Loss")
    axd["loss"].grid(True)

    axd["train_acc"].plot(epochs, train_meta["accuracy"])
    axd["train_acc"].set_title("Train Accuracy")
    axd["train_acc"].set_xlabel("Epoch")
    axd["train_acc"].set_ylabel("Accuracy")
    if acc_set_ylim_0_1:
        axd["train_acc"].set_ylim(bottom=0.0, top=1.0)
    axd["train_acc"].grid(True)

    axd["val_acc"].plot(val_meta["accuracy_per_epoch"])
    axd["val_acc"].set_title("Val Accuracy")
    axd["val_acc"].set_xlabel("Epoch")
    axd["val_acc"].set_ylabel("Mean Accuracy")
    if acc_set_ylim_0_1:
        axd["val_acc"].set_ylim(bottom=0.0, top=1.0)
    axd["val_acc"].grid(True)

    return fig


def plot_train_val_metrics_cmp(
    train_meta: dict[str, Any],
    val_meta: dict[str, Any],
    train_meta_v2: dict[str, Any],
    val_meta_v2: dict[str, Any],
    train_meta_v3: dict[str, Any],
    val_meta_v3: dict[str, Any],
    suptitle: str = "",
):
    fig, axd = plt.subplot_mosaic([["loss", "train_acc", "val_acc"]], figsize=(15, 5))
    if suptitle:
        fig.suptitle(suptitle)

    epochs = np.linspace(0, 1, len(train_meta["losses"])) * train_meta["num_epochs"]
    epochs_v2 = (
        np.linspace(0, 1, len(train_meta_v2["losses"])) * train_meta_v2["num_epochs"]
    )
    epochs_v3 = (
        np.linspace(0, 1, len(train_meta_v3["losses"])) * train_meta_v3["num_epochs"]
    )

    axd["loss"].plot(
        epochs,
        train_meta["losses"],
        "r",
        linewidth=1,
        marker=".",
        label="Loss (V1)",
    )
    axd["loss"].plot(
        epochs_v2,
        train_meta_v2["losses"],
        "g",
        linewidth=1,
        marker="o",
        label="Loss (V2)",
    )
    axd["loss"].plot(
        epochs_v3,
        train_meta_v3["losses"],
        "b",
        linewidth=1,
        marker="*",
        label="Loss (V3)",
    )
    axd["loss"].set_title("Train Loss")
    axd["loss"].set_xlabel("Epoch")
    axd["loss"].set_ylabel("Loss")
    axd["loss"].set_ylim(bottom=0.0)
    axd["loss"].legend(fontsize=8)
    axd["loss"].grid(True)

    axd["train_acc"].plot(
        epochs,
        train_meta["accuracy"],
        "r",
        linewidth=1,
        marker=".",
        label="Accuracy (V1)",
    )
    axd["train_acc"].plot(
        epochs_v2,
        train_meta_v2["accuracy"],
        "g",
        linewidth=1,
        marker="o",
        label="Accuracy (V2)",
    )
    axd["train_acc"].plot(
        epochs_v3,
        train_meta_v3["accuracy"],
        "b",
        linewidth=1,
        marker="*",
        label="Accuracy (V3)",
    )
    axd["train_acc"].set_title("Train Accuracy")
    axd["train_acc"].set_xlabel("Epoch")
    axd["train_acc"].set_ylabel("Accuracy")
    axd["train_acc"].set_ylim(bottom=0.0, top=1.0)
    axd["train_acc"].legend(fontsize=8)
    axd["train_acc"].grid(True)

    axd["val_acc"].plot(
        val_meta["accuracy_per_epoch"],
        "r",
        linewidth=1,
        marker=".",
        label="Val Acc (V1)",
    )
    axd["val_acc"].plot(
        val_meta_v2["accuracy_per_epoch"],
        "g",
        linewidth=1,
        marker="o",
        label="Val Acc (V2)",
    )
    axd["val_acc"].plot(
        val_meta_v3["accuracy_per_epoch"],
        "b",
        linewidth=1,
        marker="*",
        label="Val Acc (V3)",
    )
    axd["val_acc"].set_title("Val Accuracy")
    axd["val_acc"].set_xlabel("Epoch")
    axd["val_acc"].set_ylabel("Mean Accuracy")
    axd["val_acc"].set_ylim(bottom=0.0, top=1.0)
    axd["val_acc"].legend(fontsize=8)
    axd["val_acc"].grid(True)

    return fig
