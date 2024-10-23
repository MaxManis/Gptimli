import torch as pytorch
import torch.nn as nn

from .config import blockSize, device, dropout, nEmbd
from .TransformerBlock import Block


# NOTE: Encoder block part:
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.device = device
        self.tokenEmbeddingTable = nn.Embedding(vocabSize, nEmbd)
        self.positionEmbeddingTable = nn.Embedding(blockSize, nEmbd)
        self.blocks = nn.ModuleList(
            [
                Block(numOfHeads=nHeadsCount, numOfEmbd=nEmbd)
                for _ in range(nLayersCount)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T = x.shape

        positionEmbd = self.positionEmbeddingTable(
            pytorch.arange(T, device=self.device)
        ).unsqueeze(0).expand(B, T, -1)
        tokenEmbed = self.tokenEmbeddingTable(x)  # (B, T, C)
        x = tokenEmbed + positionEmbd
        x = self.dropout(x)
        for block in self.blocks:
            out = block(x, x, x)

        return out
