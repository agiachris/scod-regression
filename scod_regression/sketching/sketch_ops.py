import torch
from torch import nn
import numpy as np
from abc import abstractmethod

from ..utils.utils import idct


class SketchOperator(nn.Module):
    @abstractmethod
    def __init__(
        self, d: int, N: int, device: torch.DeviceObjType = torch.device("cpu")
    ) -> None:
        """Implements d x N linear operator for sketching.

        args:
            d: first matrix dimension size
            N: second matrix dimension size
        """
        super().__init__()
        self._d = d
        self._N = N
        self._device = device

    @abstractmethod
    def forward(self, M: torch.TensorType, transpose: bool = False) -> torch.TensorType:
        """Computes right multiplication by M (S @ M). If transpose,
        computes transposed left multiplication by M (M @ S.T).
        """
        raise NotImplementedError


class GaussianSketchOp(SketchOperator):
    def __init__(self, d, N, device=torch.device("cpu")):
        super().__init__(d, N, device)
        self.test_matrix = nn.Parameter(
            torch.randn(d, N, dtype=torch.float, device=device), requires_grad=False
        )

    @torch.no_grad()
    def forward(self, M, transpose=False):
        assert M.dim() == 3, "M must be of dimension 3"
        if transpose:
            return M @ self.test_matrix.t()
        return self.test_matrix @ M


class SRFTSketchOp(SketchOperator):
    def __init__(self, d, N, device=torch.device("cpu")):
        super().__init__(d, N, device)
        self.D = nn.Parameter(
            2 * (torch.rand(N, device=device) > 0.5).float() - 1, requires_grad=False
        )
        self.P = np.random.choice(N, d)

    @torch.no_grad()
    def forward(self, M, transpose=False):
        assert M.dim() == 3, "M must be of dimension 3"
        if transpose:
            M = M.transpose(2, 1)
        result = idct((self.D[:, None] * M).transpose(2, 1))
        result = result.transpose(2, 1)[:, self.P, :]
        if transpose:
            return result.transpose(2, 1)
        return result
