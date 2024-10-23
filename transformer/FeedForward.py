import torch.nn as nn

from .config import dropout


# NOTE: FeedForward module:
class FeedForward(nn.Module):
    def __init__(self, n) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, 4 * n),
            nn.ReLU(),
            nn.Linear(4 * n, n),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
