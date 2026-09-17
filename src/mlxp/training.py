from typing import cast

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import get_device


def train_mlp(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    loss_fn: nn.Module = nn.MSELoss,
    device: torch.device | None = None,
    val_split: float = 0.2,
    seed: int = 42,
    patience: int = 10,
    verbose: bool = True,
    task: str = "regression",
) -> dict[str, list[float] | torch.Tensor]:
    """Train `model` on (x, y) with mini-batch gradient descent.

    `val_split` fraction of the data is held out for validation, chosen
    deterministically from `seed`; `seed` also drives the per-epoch
    minibatch shuffling (via an explicit `torch.Generator`, independent
    of the global RNG that governs the model's own weight
    initialization). Training stops early once validation loss fails to
    improve for `patience` consecutive epochs, and the model is left
    with the weights from its best validation epoch. When `verbose`,
    train/val loss (and accuracy, for classification) is printed after
    every epoch.

    `task` is "regression" or "classification":
    - "regression": `loss_fn` defaults to `MSELoss`. Targets are
      standardized (zero mean, unit variance, from training-split
      statistics) before training, so the loss and its gradients stay
      well-conditioned regardless of the target's raw scale; the model
      is therefore trained to predict standardized targets.
      "train_loss"/"val_loss" are rescaled back to the original target
      units for interpretability, and "y_mean"/"y_std" are included in
      the returned dict so callers can unstandardize the model's raw
      predictions (`prediction * y_std + y_mean`).
    - "classification": `loss_fn` defaults to `CrossEntropyLoss`; `y`
      must be integer class indices. "train_acc"/"val_acc" (fraction
      correct) are included in the returned dict.

    Returns a dict with "train_loss", "val_loss", plus
    "y_mean"/"y_std" (regression) or "train_acc"/"val_acc" (classification).
    """
    if task not in ("regression", "classification"):
        raise ValueError(f"Unknown task: {task!r}. Use 'regression' or 'classification'.")
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()

    device = device or get_device()
    model.to(device)

    n_val = int(x.size(0) * val_split)
    perm = torch.randperm(x.size(0), generator=torch.Generator().manual_seed(seed))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    y_mean = None
    y_std = None
    loss_rescale = 1.0
    if task == "regression":
        y_mean = y_train.mean(dim=0, keepdim=True)
        y_std = y_train.std(dim=0, keepdim=True).clamp_min(1e-8)
        y_train = (y_train - y_mean) / y_std
        y_val = (y_val - y_mean) / y_std
        # Rescales the standardized MSE back to the original target units.
        loss_rescale = y_std.pow(2).mean().item()

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    shuffle_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=shuffle_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        n_train_seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            n_train_seen += xb.size(0)
            if task == "classification":
                running_correct += (preds.argmax(dim=-1) == yb).sum().item()
        train_losses.append(running_loss / n_train_seen * loss_rescale)
        if task == "classification":
            train_accs.append(running_correct / n_train_seen)

        model.eval()
        running_val_loss = 0.0
        running_val_correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                running_val_loss += loss_fn(preds, yb).item() * xb.size(0)
                if task == "classification":
                    running_val_correct += (preds.argmax(dim=-1) == yb).sum().item()
        val_loss = running_val_loss / len(val_dataset) * loss_rescale
        val_losses.append(val_loss)
        if task == "classification":
            val_accs.append(running_val_correct / len(val_dataset))

        if verbose:
            message = (
                f"epoch {epoch + 1}/{epochs} "
                f"- train_loss: {train_losses[-1]:.4f} "
                f"- val_loss: {val_loss:.4f}"
            )
            if task == "classification":
                message += f" - train_acc: {train_accs[-1]:.4f} - val_acc: {val_accs[-1]:.4f}"
            print(message)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    result: dict[str, list[float] | torch.Tensor] = {
        "train_loss": train_losses,
        "val_loss": val_losses,
    }
    if task == "regression":
        result["y_mean"] = cast(torch.Tensor, y_mean)
        result["y_std"] = cast(torch.Tensor, y_std)
    else:
        result["train_acc"] = train_accs
        result["val_acc"] = val_accs
    return result
