import torch as pytorch
import torch.nn as nn

from .config import dropout, nEmbd
from .SelfAttentionHead import SelfAttentionHead


# NOTE: MultiHeadSelfAttention module
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, headSize, numOfHeads, mask=None) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttentionHead(headSize, mask) for _ in range(numOfHeads)]
        )
        self.projection = nn.Linear(nEmbd, nEmbd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, queries):
        out = pytorch.cat([head(values, keys, queries) for head in self.heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out
