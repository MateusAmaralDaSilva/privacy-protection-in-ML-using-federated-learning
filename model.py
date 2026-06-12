import torch.nn as nn

from dataset import LOOKBACK

N_FEATURES = 12          # features per timestep (8 cyclical + snap + is_event + price_norm + sales_norm)
INPUT_DIM = LOOKBACK * N_FEATURES  # 28 * 12 = 336


class DemandForecaster(nn.Module):
    """MLP that predicts next-day store sales from a 28-day feature window."""

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)
