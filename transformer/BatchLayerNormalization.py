import torch as pytorch
import torch.nn as nn


# NOTE: BatchLayerNormalization1D module:
# the same thing as a pytorch.nn.LayerNorm() module
class BatchLayerNormalization1D:
    def __init__(self, dim, esp=1e-5) -> None:
        self.esp = esp
        self.gamma = pytorch.ones(dim)
        self.beta = pytorch.ones(dim)

    def __call__(self, x):
        # calc the forward pass
        xMean = x.mean(1, keepdim=True)  # batch mean
        xVar = x.var(1, keepdim=True)  # batch variance
        xHat = (x - xMean) / pytorch.sqrt(
            xVar + self.esp
        )  # normailize to unit variance
        out = self.gamma * xHat + self.beta
        return out

    def parameters(self):
        return [self.gamma, self.beta]
