import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class ReGLU(nn.Module):
    """
    References:
        Shazeer et al., "GLU Variants Improve Transformer," 2020.
        https://arxiv.org/abs/2002.05202
    """

    def reglu(self, x: Tensor) -> Tensor:
        assert x.shape[-1] % 2 == 0
        x, gate = x.chunk(2, dim=-1)
        return x * F.relu(gate)

    def forward(self, x: Tensor) -> Tensor:
        return self.reglu(x)


class GeGLU(nn.Module):
    """
    References:
        Shazeer et al., "GLU Variants Improve Transformer," 2020.
        https://arxiv.org/abs/2002.05202
    """

    def geglu(self, x: Tensor) -> Tensor:
        assert x.shape[-1] % 2 == 0
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

    def forward(self, x: Tensor) -> Tensor:
        return self.geglu(x)


class SwiGLU(nn.Module):
    """
    References:
        Shazeer et al., "GLU Variants Improve Transformer," 2020.
        https://arxiv.org/abs/2002.05202
    """

    def swiglu(self, x: Tensor) -> Tensor:
        assert x.shape[-1] % 2 == 0
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)

    def forward(self, x: Tensor) -> Tensor:
        return self.swiglu(x)
