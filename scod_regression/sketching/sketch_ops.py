import torch
from torch import nn, Tensor
import numpy as np
from abc import abstractmethod

from ..utils.utils import idct, batch_idct


class SketchOperator(nn.Module):
    @abstractmethod
    def __init__(self, d: int, N: int, device: torch.device = torch.device("cpu")) -> None:
        """Implements d x N linear operator for sketching.

        args:
            d: first matrix dimension size
            N: second matrix dimension size
            device: torch.device to compute sketch
        """
        super().__init__()
        self._d = d
        self._N = N
        self._device = device

    @abstractmethod
    def forward(self, M: Tensor, transpose: bool = False) -> Tensor:
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
        if transpose:
            try:
                M = M.t()
            except:
                M = M.transpose(2, 1)

        if M.dim() == 3:
            result = batch_idct((self.D[:, None] * M).transpose(2, 1))
            result = result.transpose(2, 1)[:, self.P, :]
        elif M.dim() == 2:
            result = idct((self.D[:, None] * M).t()).t()[self.P, :]
        elif M.dim() == 1:
            result = idct(self.D * M)[self.P]
        else:
            raise ValueError("Matrix is not of dimension 1, 2, 3")

        if transpose:
            try:
                result = result.t()
            except:
                result = result.transpose(2, 1)

        return result
