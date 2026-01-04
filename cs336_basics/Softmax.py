import torch
import torch.nn as nn
from torch.nn import init
from .Linear import Linear


class Softmax(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为 (...,)

        Returns:
            输出张量，形状为 (...,)
        """
        weights = torch.exp(x - torch.max(x))
        t = torch.sum(weights, dim=self.dim, keepdim=True)
        return weights / t
