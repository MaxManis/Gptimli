import torch as pytorch
import torch.nn as nn
from torch.nn import functional as F

from .config import (blockSize, device, dropout, nEmbd, nHeadsCount,
                     nLayersCount)
from .TransformerBlock import Block


# NOTE: Bigram based LanguageModel GptIMLI itself:
class GptImliModel(nn.Module):
    def __init__(self, vocabSize) -> None:
        super().__init__()

        self.tokenEmbeddingTable = nn.Embedding(vocabSize, nEmbd)
        self.positionEmbeddingTable = nn.Embedding(blockSize, nEmbd)
        #self.encoder = Encoder()
        self.blocksMasked = nn.ModuleList(
            [
                Block(numOfHeads=nHeadsCount, numOfEmbd=nEmbd, mask=True)
                for _ in range(nLayersCount)
            ]
        )
        #self.blocks = nn.ModuleList(
        #    [
        #        Block(numOfHeads=nHeadsCount, numOfEmbd=nEmbd)
        #        for _ in range(nLayersCount)
        #    ]
        #)
        self.dropout = nn.Dropout(dropout)
        self.layerNormFinal = nn.LayerNorm(nEmbd)
        self.lmHead = nn.Linear(nEmbd, vocabSize)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tokenEmbed = self.tokenEmbeddingTable(idx)  # (B, T, C)
        positionEmbd = self.positionEmbeddingTable(pytorch.arange(T, device=device)).unsqueeze(0).expand(B, T, -1)
        x = tokenEmbed + positionEmbd
        
        x = self.dropout(x)

        #for block in self.blocksMasked:
        #    x = block(x, x, x, True)

        # x = self.blocksMasked(x, x, x, True)

        #encoderOut = self.encoder(idx)
        encoderOut = x

        for block in self.blocksMasked:
            x = block(x, encoderOut, encoderOut)

        #x = self.blocks2(x, encoderOut, encoderOut)
        x = self.layerNormFinal(x)
        logits = self.lmHead(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, maxNewTokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(maxNewTokens):
            # crop idx to the last blockSize tokens:
            idx_cond = idx[:, -blockSize:]
            # get the predictions
            logits, loss = self(idx=idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idxNext = pytorch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = pytorch.cat((idx, idxNext), dim=1)  # (B, T+1)
        return idx
