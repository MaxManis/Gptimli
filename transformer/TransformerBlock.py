import torch as pytorch
import torch.nn as nn

from .config import blockSize, dropout, nEmbd
from .FeedForward import FeedForward
from .MultiHeadSelfAttention import MultiHeadSelfAttention


# NOTE: Block module:
class Block(nn.Module):
    # transformer block: communication flollows the computation:
    def __init__(self, numOfHeads, numOfEmbd, mask=None) -> None:
        super().__init__()
        headSize = numOfEmbd // numOfHeads
        self.selfAttention = MultiHeadSelfAttention(numOfHeads, headSize, mask)
        self.feedForward = FeedForward(numOfEmbd)
        self.layerNorm1 = nn.LayerNorm(
            numOfEmbd
        )  # same as BatchLayerNormalization1D class at the top
        self.layerNorm2 = nn.LayerNorm(numOfEmbd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, queries, earlyReturn=False):
        attention = self.selfAttention(values, keys, queries)
        x = self.layerNorm1(attention + queries)
        x = self.dropout(x)

        if earlyReturn is True:
            return x

        ff = self.feedForward(x)
        out = self.layerNorm2(ff + x)
        out = self.dropout(out)
        return out
