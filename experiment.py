import torch
import torch.nn as nn
import torch.optim as optim


def train(
    model,
    train_loader,
    num_epochs: int = 1,
    clip_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    use_dp: bool = True,
):
    """
    Train the model for `num_epochs` using MSE loss.

    When use_dp=True, applies DP-SGD after each backward pass:
      1. Clip the total gradient L2 norm to `clip_norm` (sensitivity bound).
      2. Add Gaussian noise scaled by (noise_multiplier * clip_norm / batch_size).
    A higher noise_multiplier gives stronger privacy but hurts accuracy.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for _ in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()

            if use_dp:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                batch_size = X_batch.size(0)
                for param in model.parameters():
                    if param.grad is not None:
                        noise = torch.randn_like(param.grad)
                        noise.mul_(noise_multiplier * clip_norm / batch_size)
                        param.grad.add_(noise)

            optimizer.step()


def evaluate(model, test_loader):
    """Return (MSE, MAE) averaged per sample over the test set."""
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    n = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch)
            total_mse += torch.nn.functional.mse_loss(preds, y_batch, reduction="sum").item()
            total_mae += torch.abs(preds - y_batch).sum().item()
            n += y_batch.size(0)

    return total_mse / n, total_mae / n
