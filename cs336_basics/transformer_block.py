import torch
import torch.nn as nn
from .RMSNorm import RMSNorm
from .CausalMultiHeadSelfAttention import CausalMultiHeadSelfAttention
from .RotaryPositionalEmbedding import RotaryPositionalEmbedding
from .SwiGLU import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: int, device=None):
        """
        Transtormer 块，它把包含多头注意力机制的一些组件包装在一起，形成一个完整的 Transformer 块。
        Args:
            d_model: int: 输入的维度,也就是d_model
            num_heads: int: 头的数量
            d_ff: int: 前馈神经网络的维度
            max_seq_len: int: 最大序列长度
            theta: float: 底数超参数
        """
        super().__init__()
        self.rms_norm1 = RMSNorm(d_model=d_model, device=device)
        self.rms_norm2 = RMSNorm(d_model=d_model, device=device)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len, device=device)
        self.swi_glu = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为 (..., seq_len, d_model)

        Returns:
            输出张量，形状为 (..., seq_len, d_model)
        """
        # 输入 → RMSNorm1 → 多头自注意力（含RoPE） → 残差连接 → RMSNorm2 → 前馈网络 → 残差连接 → 输出
        norm1_x = self.rms_norm1(x)
        token_positions = torch.arange(x.shape[-2], device=x.device)
        delta1_x = x + self.attn(norm1_x, token_positions)
        norm2_x = self.rms_norm2(delta1_x)
        delta2_x = delta1_x + self.swi_glu(norm2_x)
        return delta2_x
