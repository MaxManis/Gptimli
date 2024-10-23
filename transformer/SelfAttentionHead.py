import torch as pytorch
import torch.nn as nn
from torch.nn import functional as F

from .config import blockSize, dropout, nEmbd


# NOTE: SelfAttentionHead module
class SelfAttentionHead(nn.Module):
    # one head of self-attention:

    def __init__(self, headSize, mask=None) -> None:
        super().__init__()
        self.key = nn.Linear(nEmbd, headSize, bias=False)
        self.query = nn.Linear(nEmbd, headSize, bias=False)
        self.value = nn.Linear(nEmbd, headSize, bias=False)
        self.register_buffer("tril", pytorch.tril(pytorch.ones(blockSize, blockSize)))
        self.dropout = nn.Dropout(dropout)
        self.isMask = mask

    def forward(self, values, keys, queries):
        B, T, C = queries.shape
        k = self.key(keys)
        q = self.query(queries)

        # compute attention scores:
        weight = q @ k.transpose(-2, -1) * C**-0.5

        if self.isMask is not None:
            weight = weight.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)

        v = self.value(values)
        out = weight @ v
        return out
