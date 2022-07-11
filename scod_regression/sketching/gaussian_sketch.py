from typing import Type, Tuple

import torch
from torch import Tensor

from .sketch_ops import SketchOperator, GaussianSketchOp
from ..utils.utils import fixed_rank_eig_approx


class SinglePassPCA:
    def __init__(
        self,
        num_params: int,
        num_eigs: int,
        num_samples: int,
        sketch_op_cls: Type[SketchOperator] = GaussianSketchOp,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Computes a sketch of AA^T when presented columns of A sequentially.
        Then uses eigenvalue decomp of sketch to compute rank num_eigs range basis.

        args:
            num_params: number of trainable neural network parameters
            num_eigs: desired rank of matrix sketch approximation
            num_samples: sketch size T (default: 6 * num_eigs + 4)
            sketch_op_cls: sketch operator class (default: GaussianSketchOp)
            device: torch.device to compute low-rank approximation
        """
        self._N = num_params
        self._k = num_eigs
        self._T = num_samples
        self._device = device

        # Construct sketching operators
        self._r = max(self._k + 2, (self._T - 1) // 3)
        self._s = self._T - self._r
        self._Om = sketch_op_cls(self._r, self._N, device=self._device)
        self._Psi = sketch_op_cls(self._s, self._N, device=self._device)

        # Sketch data
        self._Y = torch.zeros(self._N, self._r, dtype=torch.float, device=self._device)
        self._W = torch.zeros(self._s, self._N, dtype=torch.float, device=self._device)

    @torch.no_grad()
    def low_rank_update(self, v: Tensor) -> None:
        """Processes a batch of columns of matrix A.

        args:
            v: matrix of size (B x N x d)
        """
        assert v.dim() == 3, "v must be of dimension 3"
        v = v.to(self._device)
        self._Y += torch.sum(v @ self._Om(v.transpose(2, 1), transpose=True), dim=0)
        self._W += torch.sum(self._Psi(v) @ v.transpose(2, 1), dim=0)

    @torch.no_grad()
    def eigs(self) -> Tuple[Tensor, Tensor]:
        """Returns a basis for the range of the top-2k left singular vectors.

        returns:
            D: top-2k eigenvalues in ascending order
            U: top-2k eigenvectors in ascending order
        """
        self._Y = self._Y.cpu()
        self._W = self._W.cpu()
        self._Om.cpu()
        self._Psi.cpu()
        U, D = fixed_rank_eig_approx(self._Y, self._W, self._Psi, 2 * self._k)
        return D, U
