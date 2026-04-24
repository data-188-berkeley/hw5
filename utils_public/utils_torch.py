import hashlib
import json
from typing import Any, Callable
from typing import Optional
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure


def get_model_device(model: torch.nn.Module) -> torch.device:
    # Heuristic: use the device of the first model parameter
    # Note: this won't work correctly for models that are sharded at
    # the op level (ie FSDP), but is good enough for this class, which
    # mainly focuses on single-node, single-GPU training.
    return next(model.parameters()).device


def count_model_parameters(model: torch.nn.Module, only_trainable: bool = True) -> int:
    """Counts the number of trainable parameters in a model."""
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        # Ex: count frozen parameters as well
        return sum(p.numel() for p in model.parameters())


def check_metas_train_test(
    out_metas_train: dict,
    out_metas_test: dict,
    num_train_batches: int,
    log_every_n_steps: int,
    num_train_epochs: int,
) -> bool:
    """Perform a few checks on out_metas_train, out_metas_test.
    These aren't comprehensive checks, but good enough for a pre-submission check to
    catch common issues early.

    Args:
        out_metas_train (dict): Output of `train_epochs()`
        out_metas_test (dict): Output of `train_epochs()`
        num_train_batches (int): Number of train dataloader batches.
        log_every_n_steps (int): See: `train_epochs()`
        num_train_epochs (int): See: `train_epochs()`.

    Raises:
        RuntimeError: If any check fails.

    Returns:
        bool: If True, we passed all checks.
    """
    # Validate that out_metas (train, test) look correct (not comprehensive)
    if len(out_metas_train) != len(out_metas_test):
        raise RuntimeError(
            f"out_metas should be the same for both train and test: {len(out_metas_train)} vs {len(out_metas_test)}"
        )
    if len(out_metas_train) != num_train_epochs:
        raise RuntimeError(
            f"out_metas_train should have {num_train_epochs} entries, found {len(out_metas_train)}"
        )
    # we always test after each train epoch
    if len(out_metas_test) != num_train_epochs:
        raise RuntimeError(
            f"out_metas_test should have {num_train_epochs} entries, found {len(out_metas_test)}"
        )

    # Train meta checks
    expected_keys_train_meta = ["loss", "ind_batch", "dur_total_secs", "tput_total"]
    # +1 for logging first batch
    # ceil() (not floor()) since we log the final batch
    expected_num_losses_train = math.ceil(num_train_batches / log_every_n_steps) + 1
    for ind_epoch, out_meta_train in enumerate(out_metas_train):
        for expected_key in expected_keys_train_meta:
            if expected_key not in out_meta_train:
                raise RuntimeError(
                    f"epoch={ind_epoch} '{expected_key}' not found in out_meta_train, keys: {out_meta_train.keys()}"
                )
        if len(out_meta_train["ind_batch"]) != len(out_meta_train["loss"]):
            raise RuntimeError(
                f"epoch={ind_epoch} len mismatch: len(ind_batch)={len(out_meta_train['ind_batch'])} vs len(loss)={len(out_meta_train['loss'])}"
            )
        if out_meta_train["ind_batch"][0] != 0:
            raise RuntimeError(
                f"First ind_batch must always be 0, was: {out_meta_train['ind_batch'][0]}"
            )
        if out_meta_train["ind_batch"][-1] != num_train_batches - 1:
            raise RuntimeError(
                f"Expected last ind_batch to be {num_train_batches - 1}, was: {out_meta_train['ind_batch'][-1]}"
            )
        if len(out_meta_train["loss"]) != expected_num_losses_train:
            raise RuntimeError(
                f"epoch={ind_epoch} expected {expected_num_losses_train} losses, got: {len(out_meta_train['loss'])}"
            )

    # Test meta checks
    expected_keys_test_meta = [
        "test_loss",
        "test_accuracy",
        "dur_total_secs",
        "tput_total",
    ]
    for ind_epoch, out_meta_test in enumerate(out_metas_test):
        for expected_key in expected_keys_test_meta:
            if expected_key not in out_meta_test:
                raise RuntimeError(
                    f"epoch={ind_epoch} '{expected_key}' not found in out_meta_test, keys: {out_meta_test.keys()}"
                )
    return True


def visualize_image_classification_dataset(
    dataset: torch.utils.data.Dataset,
    selected_class_names: Optional[list[str]] = None,
    num_examples_per_class: int = 5,
    figsize: tuple[int, int] = (10, 10),
) -> Figure:
    """Visualizes an image classification dataset.
    Assumes that dataset emits images in the form (img: torch.Tensor, label: int), where
        img.shape=[C, H, W], and label is an integer corresponding to the class
        names in `dataset.classes`.
    Args:
        dataset: Dataset to visualize.
        selected_class_names: If given, restrict visualization to just these
            class names. If None, visualize all classes.
        num_examples_per_class: Visualize this many examples from each class.
        figsize: Figure size. (float width, float height).
    """
    # (Acknowledgement) Gemini was used to help generate this function, via the Colab+Gemini integration product
    # feature. Unfortunately the exact prompt is lost, but from memory it roughly was:
    # "Using matplotlib, generate a visualization of the CIFAR10 dataset, where we show 5 examples from 5 different classes in a 5x5 grid."
    # EricK then did additional refactoring to produce the final function.

    # Get class names from the dataset
    class_names = dataset.classes
    print(f"dataset classes: {class_names}")

    # Select classes to visualize
    if not selected_class_names:
        # All classes
        selected_class_names = list(class_names)
    selected_classes_indices = [
        dataset.classes.index(name) for name in selected_class_names
    ]

    # Prepare to store images for plotting
    images_to_plot = []

    # Collect N images for each selected class
    is_img_grayscale = False
    for class_idx in selected_classes_indices:
        class_images = []
        for img, label in dataset:
            if label == class_idx:
                # Convert from (C, H, W) to (H, W, C) and to numpy
                if img.shape[0] == 1:
                    is_img_grayscale = True
                class_images.append(img.permute(1, 2, 0).numpy())
                if len(class_images) == num_examples_per_class:
                    break
        images_to_plot.append(class_images)

    # Create a num_classes x num_examples_per_class grid for plotting
    fig, axes = plt.subplots(
        len(selected_class_names), num_examples_per_class, figsize=figsize
    )

    for i, class_images in enumerate(images_to_plot):
        for j, img in enumerate(class_images):
            if is_img_grayscale:
                axes[i, j].imshow(img, cmap="gray")
            else:
                axes[i, j].imshow(img)
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_title(selected_class_names[i])

    plt.suptitle(
        f"Dataset Viz: {num_examples_per_class} Examples from {len(selected_class_names)} Classes",
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout()
    return fig


def visualize_image_classification_dataloader(
    dataloader: torch.utils.data.DataLoader,
    first_n_batches: int = 5,
    figsize: tuple[int, int] = (10, 10),
):
    """Visualizes the first few batches of an image classification dataloader.
    Assumes that dataloader emits images in the form (img: torch.Tensor, label: int), where
        img.shape=[B, C, H, W], and label is an integer corresponding to the class
        names in `dataset.classes`.
    Args:
        dataloader: DataLoader to visualize.
        first_n_batches: Visualize first n batches.
        figsize: Figure size. (float width, float height).
    """
    # Get class names from the dataset
    class_names = dataloader.dataset.classes
    print(f"dataset classes: {class_names}")

    # Prepare to store images for plotting
    # [i][j] -> np.ndarray img, shape=[H, W, C]
    images_to_plot: list[list[np.ndarray]] = []
    labels_to_plot: list[list[str]] = []

    # Collect 5 images+labels from each batch
    is_img_grayscale = False
    for ind_batch, (X, y) in enumerate(dataloader):
        if ind_batch >= first_n_batches:
            break
        # X.shape=[B, C, H, W]
        # y.shape=[B]
        if X.shape[1] == 1:
            is_img_grayscale = True
        batch_imgs = []
        batch_labels = []
        for i in range(5):
            # Convert from (C, H, W) to (H, W, C) and to numpy
            img_i = X[i, :, :, :].permute(1, 2, 0).numpy()
            batch_imgs.append(img_i)
            # labels
            batch_labels.append(f"'{class_names[y[i]]}' (y={y[i]})")
        images_to_plot.append(batch_imgs)
        labels_to_plot.append(batch_labels)

    # Create a num_batches x 5 grid for plotting
    fig, axes = plt.subplots(first_n_batches, 5, figsize=figsize)

    for i, (batch_images, batch_labels) in enumerate(
        zip(images_to_plot, labels_to_plot)
    ):
        for j, (img, label_str) in enumerate(zip(batch_images, batch_labels)):
            if is_img_grayscale:
                axes[i, j].imshow(img, cmap="gray")
            else:
                axes[i, j].imshow(img)
            axes[i, j].axis("off")
            axes[i, j].set_title(f"Batch={i}\n{label_str}")

    plt.suptitle(
        f"Dataloader Viz: 5 Examples from first {first_n_batches} batches",
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout()
    return fig


def plot_image_gallery(
    images: list[np.ndarray], titles: list[str], n_row: int = 3, n_col: int = 4
) -> Figure:
    """Plots images+titles to figure.

    Args:
        images (list[np.ndarray]): shape=[num_images, img_height, img_width], or
            [num_images, img_height, img_width, 3]
        titles (list[str]): len=num_images.
        n_row (int, optional): Number of rows in figure. Defaults to 3.
        n_col (int, optional): Number of cols in figure. Defaults to 4.

    Returns:
        Figure: fig_out.
    """
    # inspired by:
    #   https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html
    # Helper function to plot a gallery of portraits
    if len(images) != len(titles):
        raise RuntimeError(
            f"num images must match num text titles: {len(images)} vs {len(titles)}"
        )
    fig = plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    fig.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        axes_subplot = fig.add_subplot(n_row, n_col, i + 1)

        axes_subplot.imshow(images[i], cmap=plt.cm.gray)
        axes_subplot.set_title(titles[i], size=12)
        axes_subplot.set_xticks(())
        axes_subplot.set_yticks(())
    return fig


def visualize_predictions(
    pred_logits: np.ndarray,
    X: np.ndarray,
    num_chans_img: int,
    h_img: int,
    w_img: int,
    y: np.ndarray,
    ind_class_to_viz: int,
    class_name_to_viz: str,
    top_k: int = 5,
):
    """Visualizes the predictions of an image classifier.

    Args:
        pred_logits (np.ndarray): Model predictions. shape=[num_samples, num_classes].
        X (np.ndaray): Test images. shape=[num_samples, num_pixels], arranged row-by-row,
            in [C, H, W] -> [C*H*W] order.
        num_chans_img: Number of image channels. Should be 1, these are grayscale images.
        h_img: Image height, in pixels.
        w_img: Image width, in pixels.
        y (np.ndarray): Test labels. shape=[num_samples].
        ind_class_to_viz: Which ground-truth class to visualize predictions for.
        top_k: How many examples to visualize.

    Returns:
        fig: matplotlib Figure instance.
    """
    # shape=[num_samples, num_classes]
    pred_probs = _softmax_normalize(pred_logits)
    pred_labels = pred_probs.argmax(axis=1)
    mask_correct_preds = pred_labels == y

    # Visualize top k most confident true positives
    tp_indices = (mask_correct_preds == 1) & (y == ind_class_to_viz)
    tp_probs = pred_probs[tp_indices, ind_class_to_viz]

    sorted_inds = np.argsort(tp_probs)[::-1]
    sorted_inds_top_k = sorted_inds[:top_k]
    top_tp_probs = tp_probs[sorted_inds_top_k]

    Xtp = X[tp_indices, :]
    Xviz = Xtp[sorted_inds_top_k, :]

    yviz = y[tp_indices][sorted_inds_top_k]

    if num_chans_img > 1:
        # permute [b, c, h, w] -> [b, h, w, c] for matplotlib imshow()
        images = Xviz.reshape([Xviz.shape[0], num_chans_img, h_img, w_img]).transpose(
            [0, 2, 3, 1]
        )
    else:
        images = Xviz.reshape([Xviz.shape[0], h_img, w_img])

    if images.shape[0] < top_k:
        # skip, not enough true positives to visualize
        print(f"uhoh, not enough, skipping: {images.shape}, {sum(tp_indices)}")
        fig_tp = None
    else:
        titles = [
            f"p={top_tp_probs[i]:.3f} y_gt={yviz[i]}" for i in range(len(top_tp_probs))
        ]

        fig_tp = plot_image_gallery(images, titles, n_row=1, n_col=5)
        fig_tp.suptitle(
            f"Top {top_k} most confident true positives (ind_class={ind_class_to_viz} {class_name_to_viz})"
        )

    # Visualize top k least confident false negatives ("hard" examples)
    # Sort false negatives by their predicted prob for `digit`.
    # Take top k lowest scores, and visualize them.
    fn_indices = (mask_correct_preds == 0) & (y == ind_class_to_viz)

    fn_probs = pred_probs[fn_indices, ind_class_to_viz]

    sorted_inds_top_k = np.argsort(fn_probs)[:top_k]
    top_fn_probs = fn_probs[sorted_inds_top_k]

    # what did we predict?
    pred_label_fn = pred_labels[fn_indices][sorted_inds_top_k]
    pred_prob_fn = pred_probs[fn_indices, :][sorted_inds_top_k, pred_label_fn]

    Xviz = X[fn_indices, :][sorted_inds_top_k, :]
    yviz = y[fn_indices][sorted_inds_top_k]

    if num_chans_img > 1:
        # permute [b, c, h, w] -> [b, h, w, c] for matplotlib imshow()
        images = Xviz.reshape([Xviz.shape[0], num_chans_img, h_img, w_img]).transpose(
            [0, 2, 3, 1]
        )
    else:
        images = Xviz.reshape([Xviz.shape[0], h_img, w_img])

    if images.shape[0] < top_k:
        # skip, not enough true positives to visualize
        print(f"uhoh, not enough, skipping: {images.shape}, {sum(tp_indices)}")
        fig_fn = None
    else:
        titles = [
            f"\np_gt={top_fn_probs[i]:.3f} y_gt={yviz[i]}\npred={pred_label_fn[i]} ({pred_prob_fn[i]:.3f})"
            for i in range(len(top_fn_probs))
        ]

        fig_fn = plot_image_gallery(images, titles, n_row=1, n_col=5)
        fig_fn.suptitle(
            f"Top {top_k} least confident false negatives (ind_class={ind_class_to_viz} {class_name_to_viz})\n"
        )

        # make room for suptitle
        fig_fn.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.8, hspace=0.35)
    return fig_tp, fig_fn


def _softmax_normalize(vals: np.ndarray) -> np.ndarray:
    """Perform softmax normalization of input vals. The normalization
    is done across the last dimension.
    Args:
        vals: shape=[d0, d1, ..., dim].
    Returns:
        vals_norm: shape=[d0, d1, ..., dim].
    """
    vals_exp = np.exp(vals)
    norm_factor = np.sum(vals_exp, axis=-1)
    # note: need `norm_factor[:, np.newaxis]` to broadcast the division
    #   across the axis=1 of vals_exp
    #   otherwise, this line produces an error, due to numpy not knowing
    #   how to divide an array with shape=[batchsize, k] with an array with
    #   shape=[k].
    #     vals_exp / norm_factor
    return vals_exp / (norm_factor[:, np.newaxis])


def visualize_image_classifier_preds(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: list[str],
    image_shape: tuple[int, int, int],
    top_k: int = 5,
    dataloader_viz: Optional[torch.utils.data.DataLoader] = None,
):
    """Visualizes an image classifier model on an input dataloader.
    Args:
        model: image classifier model. Must accept an image (shape=[B, C, H, W]), and its forward()
            should return the predicted logits (shape=[B, num_classes]).
        dataloader: Dataloader to run visualize predictions on.
        class_names: len(num_classes), the human-readable class names for each class index.
        image_shape: (C, H, W).
        top_k: show this many examples for each class.
        dataloader_viz:
            Important: the order of this must be exactly the same as `dataloader`.
                Ex: ensure that `shuffle=False` is set for both!
    """
    num_classes = len(class_names)
    target_device = get_model_device(model)
    # Gather predictions
    # pre-allocate tensors to improve performance
    print(f"Running inference on {len(dataloader.dataset)} samples...")
    tic_infer = time.time()
    pred_logits = torch.zeros(size=(len(dataloader.dataset), num_classes))
    gt_labels = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.long)
    test_images = torch.zeros(
        size=(len(dataloader.dataset),) + tuple(image_shape), dtype=torch.float32
    )
    with torch.no_grad():
        i1 = 0
        for X, y in dataloader:
            # X.shape=[B, C, H, W]
            # y.shape=[B]
            i2 = i1 + X.shape[0]
            # pred.shape=[B, num_classes]
            pred = model(X.to(device=target_device))
            pred_logits[i1:i2, :] = pred.to(device="cpu")
            gt_labels[i1:i2] = y
            test_images[i1:i2, :] = X
            i1 = i2
    dur_infer = time.time() - tic_infer
    tput_infer = len(dataloader.dataset) / dur_infer
    print(
        f"Finished inference on {len(dataloader.dataset)} samples ({dur_infer:.2f} secs, throughput={tput_infer:.2f} samples/sec)"
    )

    if dataloader_viz is not None:
        # Populate test images (ie without preprocessing applied, like mean/std standardization)
        test_images_viz = torch.zeros(
            size=(len(dataloader_viz.dataset),) + tuple(image_shape),
            dtype=torch.float32,
        )
        i1 = 0
        for X, y in dataloader_viz:
            i2 = i1 + X.shape[0]
            test_images_viz[i1:i2, :] = X
            i1 = i2
    else:
        test_image_viz = test_images

    # Visualize predictions
    for ind_class, class_name in enumerate(class_names):
        visualize_predictions(
            pred_logits=pred_logits.numpy(),
            X=test_images_viz.numpy(),
            num_chans_img=image_shape[0],
            h_img=image_shape[1],
            w_img=image_shape[2],
            y=gt_labels.numpy(),
            ind_class_to_viz=ind_class,
            class_name_to_viz=class_name,
            top_k=top_k,
        )


# Don't modify this helper function.
def train_epochs(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    train_epoch_fn: Callable,
    test_dataloader: torch.utils.data.DataLoader,
    test_fn: Callable,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    log_every_n_steps: int = 100,
) -> tuple[list[dict], list[dict]]:
    """Trains a model on a training dataset, and evaluates on a test set, for a
    given number of epochs.
    Args:
        model:
        train_dataloader:
        train_epoch_fn: Function that trains the model for one epoch. For exact input/output
            format, see: `train_epoch()` in the notebook.
        test_dataloader:
        test_fn: Function that evaluates the model on a test set. For exact input/output
            format, see: `test()` in the notebook.
        loss_fn:
        optimizer:
        num_epochs: How many epochs to train for.
        log_every_n_steps: See: `train_epoch()`

    Returns:
        out_metas_train: list[dict]. Useful training metadata. See problem spec
            to learn exactly what this should contain.
        out_metas_test: list[dict]. Useful test metadata. See problem spec
            to learn exactly what this should contain.
    """
    out_metas_train = []
    out_metas_test = []
    tic_start = time.time()
    for t in range(num_epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        out_meta_train = train_epoch_fn(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            log_every_n_steps=log_every_n_steps,
        )
        out_meta_test = test_fn(test_dataloader, model, loss_fn)
        out_metas_train.append(out_meta_train)
        out_metas_test.append(out_meta_test)
    dur = time.time() - tic_start
    print(f"Finished train_epochs(), {dur:.4f} secs")
    return out_metas_train, out_metas_test


def get_hash_filename(params: dict[str, Any], prefix="model", suffix=".pth") -> str:
    # 1. Sort keys to ensure consistent hashing for same parameters
    params_str = json.dumps(params, sort_keys=True)

    # 2. Generate SHA256 hash
    hash_object = hashlib.sha256(params_str.encode("utf-8"))
    hash_str = hash_object.hexdigest()

    # 3. Return sanitized string
    return f"{prefix}_{hash_str[:16]}{suffix}"


def visualize_mae_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    image_shape: tuple[int, int, int],
    patchify: Callable,
    unpatchify: Callable,
    num_batches_viz: int = 1,
    figsize: tuple[int, int] = (8, 40),
    img_transform_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
):
    """Visualize MAE predictions.

    Args:
        model (torch.nn.Module): _description_
        dataloader (torch.utils.data.DataLoader): _description_
        image_shape (tuple[int, int, int]): _description_
        patchify (Callable): _description_
        unpatchify (Callable): _description_
        num_batches_viz (int, optional): _description_. Defaults to 1.
        figsize (tuple[int, int], optional): (width, height). Defaults to (10, 10).
        img_transform_fn (Optional[Callable[[torch.Tensor], torch.Tensor]], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    target_device = get_model_device(model)
    # Gather predictions
    # pre-allocate tensors to improve performance
    num_samples_viz = num_batches_viz * dataloader.batch_size
    print(
        f"Running inference on {num_batches_viz} batches ({num_samples_viz} samples total)..."
    )
    tic_infer = time.time()

    # shape=[n, channels, height, width]
    pred_images = torch.zeros(
        size=(
            num_samples_viz,
            image_shape[0],
            image_shape[1],
            image_shape[2],
        ),
        dtype=torch.float32,
    )
    gt_images = torch.zeros(
        size=(
            num_samples_viz,
            image_shape[0],
            image_shape[1],
            image_shape[2],
        ),
        dtype=torch.float32,
    )
    # masks.shape=[num_samples_viz, num_patches], where 0 means not-masked, 1 means masked
    masks = None
    losses = torch.zeros(size=(num_samples_viz,), dtype=torch.float32)
    with torch.no_grad():
        i1 = 0
        for ind_batch, (X, y) in enumerate(dataloader):
            if ind_batch >= num_batches_viz:
                break
            X = X.to(device=target_device)
            # X.shape=[B, C, H, W]
            # y.shape=[B]
            i2 = i1 + X.shape[0]
            # gt_image_patches.shape=[batch, num_patches, patch_size * patch_size * channels]
            gt_image_patches = patchify(X)

            # predicted_patches: shape=[batch, num_patches, patch_size * patch_size * channels]
            # mask.shape=[batch, num_patches], where 0 means not-masked, 1 means masked
            predicted_patches, mask = model(X)
            # loss_cur_batch.shape=[batch]
            loss_cur_batch = (
                torch.mean(
                    torch.mean(
                        torch.square(gt_image_patches - predicted_patches), dim=-1
                    )
                    * mask,
                    dim=-1,
                )
            ).to(device="cpu")
            pred_images[i1:i2, :, :, :] = unpatchify(predicted_patches).to(device="cpu")
            gt_images[i1:i2, :, :, :] = X.to(device="cpu")
            if masks is None:
                masks = torch.zeros(
                    size=(num_samples_viz, gt_image_patches.shape[1]),
                    dtype=torch.bool,
                )
            masks[i1:i2, :] = mask.to(device="cpu")
            losses[i1:i2] = loss_cur_batch

            i1 = i2
    dur_infer = time.time() - tic_infer
    tput_infer = num_samples_viz / dur_infer
    print(
        f"Finished inference on {num_samples_viz} samples ({dur_infer:.2f} secs, throughput={tput_infer:.2f} samples/sec)"
    )

    if img_transform_fn is not None:
        pred_images = img_transform_fn(pred_images)
        gt_images = img_transform_fn(gt_images)

    # also apply mask
    gt_images_patched = patchify(gt_images)
    gt_images_masked = unpatchify(gt_images_patched * ~masks.unsqueeze(2))

    # blend pred-patches with gt non-masked
    pred_images_blend = unpatchify(
        (patchify(pred_images) * (masks.unsqueeze(2)))
        + (patchify(gt_images) * ~masks.unsqueeze(2))
    )

    # Visualize images
    # matplotlib expects images to be (H, W, C) and in numpy, and between [0, 1]
    pred_images = torch.clamp(pred_images, min=0.0, max=1.0)
    gt_images = torch.clamp(gt_images, min=0.0, max=1.0)
    gt_images_masked = torch.clamp(gt_images_masked, min=0.0, max=1.0)
    pred_images_blend = torch.clamp(pred_images_blend, min=0.0, max=1.0)

    pred_images_plt = pred_images.permute(0, 2, 3, 1).numpy()
    gt_images_plt = gt_images.permute(0, 2, 3, 1).numpy()
    gt_images_masked_plt = gt_images_masked.permute(0, 2, 3, 1).numpy()
    pred_images_blend_plt = pred_images_blend.permute(0, 2, 3, 1).numpy()
    is_image_grayscale = image_shape[0] == 1

    # Create a n x 4 grid for plotting
    fig, axes = plt.subplots(pred_images_plt.shape[0], 4, figsize=figsize)

    for ind in range(pred_images_plt.shape[0]):
        pred_image = pred_images_plt[ind, :, :, :]
        pred_image_blend = pred_images_blend_plt[ind, :, :, :]
        gt_image = gt_images_plt[ind, :, :, :]
        gt_image_masked = gt_images_masked_plt[ind, :, :, :]
        if is_image_grayscale:
            axes[ind, 0].imshow(pred_image, cmap="gray")
            axes[ind, 1].imshow(pred_image_blend, cmap="gray")
            axes[ind, 2].imshow(gt_image, cmap="gray")
            axes[ind, 3].imshow(gt_image_masked, cmap="gray")
        else:
            axes[ind, 0].imshow(pred_image)
            axes[ind, 1].imshow(pred_image_blend)
            axes[ind, 2].imshow(gt_image)
            axes[ind, 3].imshow(gt_image_masked)

        axes[ind, 0].axis("off")
        axes[ind, 1].axis("off")
        axes[ind, 2].axis("off")
        axes[ind, 3].axis("off")
        axes[ind, 0].set_title(
            f"ind={ind + 1}/{pred_images_plt.shape[0]} Pred(raw) loss={losses[ind].item():.4f}"
        )
        axes[ind, 1].set_title("Pred (blend)")
        axes[ind, 2].set_title("GT")
        axes[ind, 3].set_title("GT (masked)")

    fig.suptitle(
        "Dataset Viz: MAE reconstructions.",
        fontsize=16,
        y=1.02,
    )
    fig.tight_layout()
    return fig


def create_undo_img_standardization(
    mean: tuple[float, float, float], std: tuple[float, float, float]
) -> Callable[[torch.Tensor], torch.Tensor]:
    # aka inverse of transforms.Normalize(mean, std)
    mean = torch.tensor(mean)
    std = torch.tensor(std)

    def inner(images_norm: torch.Tensor) -> torch.Tensor:
        return images_norm * std.view(3, 1, 1) + mean.view(3, 1, 1)

    return inner
