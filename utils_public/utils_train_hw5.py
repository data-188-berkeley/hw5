from typing import Any, Callable, Optional
import time
import os

from tqdm import trange, tqdm
import torch
import numpy as np

from utils_public.utils_torch import get_model_device


def train_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader_train: torch.utils.data.DataLoader,
    dataloader_val: torch.utils.data.DataLoader,
    num_epochs: int,
    outpath_best_val: str,
    outpath_best_val_meta: str,
    log_every_n_steps: int = 100,
    is_jupyter_notebook: bool = True,
    is_mae: bool = False,
    patchify: Optional[Callable] = None,
    flip_val_acc_cmp: bool = False,
    use_amp: bool = False,
    grad_scaler: Optional[torch.amp.GradScaler] = None,
    profiler: Optional[Any] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Trains an image classification model.

    Args:
        model (torch.nn.Module): Model.
        optimizer (torch.optim.Optimizer): Optimizer.
        dataloader_train (torch.utils.data.DataLoader): Train dataloader.
        dataloader_val (torch.utils.data.DataLoader): Val dataloader.
        num_epochs (int): Num train epochs.
        log_every_n_steps (int, optional): Log every N steps. Defaults to 100.
        flip_val_acc_cmp: If True, this will treat val_acc as a metric where
            "lower is better" (rather than default "higher is better").
            Ex: useful for MAE training, where val_acc is reconstruction loss,
            not accuracy.
    Returns:

    """
    if is_jupyter_notebook:
        # Annoying: apparently tqdm needs a separate import to work correctly in
        # notebooks.
        from tqdm.notebook import trange, tqdm
    else:
        from tqdm import trange, tqdm
    torch_device = get_model_device(model)
    # Move model to target device (eg GPU if available)
    model.to(torch_device)

    total_steps = 0
    losses = []
    train_acc = []
    all_val_acc = []
    if not flip_val_acc_cmp:
        best_val_acc = 0
    else:
        # Ex: for reconstruction loss, this needs to start at +Inf
        best_val_acc = float("Inf")

    train_meta = {
        "num_epochs": num_epochs,
    }
    val_meta = {
        "num_epochs": num_epochs,
    }

    tic_total = time.time()
    epoch_iterator = trange(num_epochs)
    for epoch in epoch_iterator:
        # Train
        model.train()
        data_iterator = tqdm(dataloader_train)
        tic_epoch_train = time.time()
        for x, y in data_iterator:
            total_steps += 1
            x, y = (
                x.to(device=torch_device, non_blocking=True),
                y.to(device=torch_device, non_blocking=True),
            )
            with torch.autocast(
                device_type=torch_device.type, dtype=torch.float16, enabled=use_amp
            ):
                if not is_mae:
                    logits = model(x)
                    loss = torch.mean(torch.nn.functional.cross_entropy(logits, y))
                    accuracy = torch.mean((torch.argmax(logits, dim=-1) == y).float())
                else:
                    image_patches = patchify(x)
                    predicted_patches, mask = model(x)
                    loss = (
                        torch.sum(
                            torch.mean(
                                torch.square(image_patches - predicted_patches), dim=-1
                            )
                            * mask
                        )
                        / mask.sum()
                    )
                    accuracy = loss
            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()

            # perftip: avoid calling `Tensor.item()` until after the backwards call, to avoid
            # unnecessarily blocking computation
            accuracy_item = accuracy.item()

            data_iterator.set_postfix(loss=loss.item(), train_acc=accuracy_item)

            if total_steps % log_every_n_steps == 0:
                losses.append(loss.item())
                train_acc.append(accuracy_item)
            if profiler is not None:
                profiler.step()
        dur_epoch_train = time.time() - tic_epoch_train

        train_tput = len(dataloader_train.dataset) / dur_epoch_train

        # Validation
        val_acc = []
        model.eval()
        tic_val = time.time()
        for x, y in dataloader_val:
            x, y = x.to(torch_device), y.to(torch_device)
            with torch.no_grad():
                if not is_mae:
                    logits = model(x)
                    accuracy = torch.mean(
                        (torch.argmax(logits, dim=-1) == y).float()
                    ).item()
                else:
                    image_patches = patchify(x)
                    predicted_patches, mask = model(x)
                    loss = (
                        torch.sum(
                            torch.mean(
                                torch.square(image_patches - predicted_patches), dim=-1
                            )
                            * mask
                        )
                        / mask.sum()
                    )
                    accuracy = loss.item()
            val_acc.append(accuracy)
        dur_val = time.time() - tic_val
        tput_val = len(dataloader_val.dataset) / dur_val

        mean_val_acc = np.mean(val_acc).item()
        all_val_acc.append(mean_val_acc)

        print(
            f"[epoch={epoch + 1}/{num_epochs}] "
            f"(Train) dur={dur_epoch_train:.2f}s. tput={train_tput:.2f} exs/s "
            f"(Val) dur={dur_val:.2f}s tput={tput_val:.2f} exs/s. val_acc={mean_val_acc:.6f} (best_val_acc={best_val_acc:.6f})"
        )

        # Save best model
        is_cur_best = mean_val_acc > best_val_acc
        if flip_val_acc_cmp:
            # Ex: val_acc is reconstruction loss, not accuracy, so flip comparison
            is_cur_best = mean_val_acc < best_val_acc
        if is_cur_best:
            best_val_acc = mean_val_acc
            os.makedirs(os.path.dirname(outpath_best_val), exist_ok=True)
            torch.save(
                {
                    # hack: if model is compiled via `torch.compile()`, we don't want to emit the
                    #   _orig_mod.* prefix in state dict (complicates loading the model)
                    "weights": getattr(model, "_orig_mod", model).state_dict(),
                    "epoch": epoch,
                },
                outpath_best_val,
            )
            print(
                f"Saved best val model to: {outpath_best_val} (val_acc={mean_val_acc:.6f} vs prev_best_val_acc={best_val_acc:.6f}) (flip_val_acc_cmp={flip_val_acc_cmp})"
            )

        epoch_iterator.set_postfix(val_acc=mean_val_acc, best_val_acc=best_val_acc)

    train_meta["losses"] = losses
    train_meta["accuracy"] = train_acc

    val_meta["accuracy_per_epoch"] = all_val_acc
    dur_total = time.time() - tic_total
    print(f"Training complete. {dur_total:.2f} secs. best_val_acc={best_val_acc:.6f}")

    # also save train/val meta to disk
    torch.save(
        {
            "train_meta": train_meta,
            "val_meta": val_meta,
            "dur_total": dur_total,
            "best_val_acc": best_val_acc,
        },
        outpath_best_val_meta,
    )
    print(f"Saved train/val meta to: {outpath_best_val_meta}")

    return train_meta, val_meta


def load_if_exists_else_train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader_train: torch.utils.data.DataLoader,
    dataloader_val: torch.utils.data.DataLoader,
    num_epochs: int,
    outpath_best_val: str,
    outpath_best_val_meta: str,
    log_every_n_steps: int = 100,
    is_jupyter_notebook: bool = True,
    force_retrain: bool = False,
    is_mae: bool = False,
    patchify: Optional[Callable] = None,
    flip_val_acc_cmp: bool = False,
    use_amp: bool = False,
    grad_scaler: Optional[torch.amp.GradScaler] = None,
    profiler: Optional[Any] = None,
):
    if (
        os.path.exists(outpath_best_val) and os.path.exists(outpath_best_val_meta)
    ) and not force_retrain:
        # load model from disk
        load_d1 = torch.load(outpath_best_val)
        if hasattr(model, "_orig_mod"):
            # hack: torch.compile adds _orig_mod to state dict
            model._orig_mod.load_state_dict(load_d1["weights"])
        else:
            model.load_state_dict(load_d1["weights"])
        print(
            f"Loaded model weights from {outpath_best_val} (epoch={load_d1['epoch']})"
        )
        # load train/val meta
        load_d2 = torch.load(outpath_best_val_meta)
        train_meta, val_meta = load_d2["train_meta"], load_d2["val_meta"]
        print(f"Loaded train/val meta from: {outpath_best_val_meta}")
        return train_meta, val_meta
    else:
        return train_loop(
            model=model,
            optimizer=optimizer,
            dataloader_train=dataloader_train,
            dataloader_val=dataloader_val,
            num_epochs=num_epochs,
            outpath_best_val=outpath_best_val,
            outpath_best_val_meta=outpath_best_val_meta,
            log_every_n_steps=log_every_n_steps,
            is_jupyter_notebook=is_jupyter_notebook,
            is_mae=is_mae,
            patchify=patchify,
            flip_val_acc_cmp=flip_val_acc_cmp,
            use_amp=use_amp,
            grad_scaler=grad_scaler,
            profiler=profiler,
        )
