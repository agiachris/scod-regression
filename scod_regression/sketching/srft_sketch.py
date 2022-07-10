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
        *args,
        sketch_op_cls: Type[SketchOperator] = SRFTSketchOp,
        device: torch.device = torch.device("cpu")
    ) -> None:
        """Computes a sketch of AA^T when presented columns of A sequentially.
        Then uses eigenvalue decomp of sketch to compute rank num_eigs range basis.

        args:
            *args: positional arguments of SinglePassPCA
            sketch_op_cls: sketch operator class (default: SRFTSketchOp)
            device: torch.device to compute low-rank approximation
        """
        super().__init__(*args, sketch_op_cls=sketch_op_cls, device=device)
