from typing import Type

import torch

from .sketch_ops import SketchOperator, SRFTSketchOp
from .gaussian_sketch import SinglePassPCA


class SRFTSinglePassPCA(SinglePassPCA):
    """Computes a subsampled randomized fourier transform sketch of AA^T
    when presented columns of A sequentially. Then uses eigen decomp of
    sketch to compute rank r range basis
    """

    def __init__(
        self,
        num_params: int,
        num_eigs: int,
        num_samples: int,
        sketch_op_cls: Type[SketchOperator] = SRFTSketchOp,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Computes a sketch of AA^T when presented columns of A sequentially.
        Then uses eigenvalue decomp of sketch to compute rank num_eigs range basis.

        args:
            num_params: number of trainable neural network parameters
            num_eigs: desired rank of matrix sketch approximation
            num_samples: sketch size T (default: 6 * num_eigs + 4)
            sketch_op_cls: sketch operator class (default: SRFTSketchOp)
            device: torch.device to compute low-rank approximation
        """
        super().__init__(
            num_params,
            num_eigs,
            num_samples,
            sketch_op_cls=sketch_op_cls,
            device=device,
        )
