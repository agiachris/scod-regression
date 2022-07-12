from typing import (
    Optional,
    Type,
    Union,
    Iterable,
    Iterator,
    Callable,
    Tuple,
    List,
    Dict,
)

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, IterableDataset, DataLoader
from functorch import make_functional_with_buffers, jacrev, vmap
from functorch._src.make_functional import FunctionalModuleWithBuffers

from .distributions.distribution import DistributionLayer
from .distributions.normal import NormalMeanParamLayer
from .sketching import SinglePassPCA
from .utils.utils import flatten


class SCOD(nn.Module):
    @property
    def functional_model(
        self,
    ) -> Tuple[FunctionalModuleWithBuffers, Iterator[nn.Parameter], Dict[str, Optional[Tensor]],]:
        """Get functorch functional model. Set parameter gradients to None."""
        for p in self._fparams:
            if p.grad is not None:
                p.grad = None
        return self._fmodel, self._fparams, self._fbuffers

    @functional_model.setter
    def functional_model(
        self,
        x: Tuple[
            FunctionalModuleWithBuffers,
            Iterator[nn.Parameter],
            Dict[str, Optional[Tensor]],
        ],
    ):
        """Set functorch functional model."""
        self._fmodel, self._fparams, self._fbuffers = x

    def __init__(
        self,
        model: nn.Module,
        output_dist_cls: Type[DistributionLayer] = NormalMeanParamLayer,
        sketch_cls: Type[SinglePassPCA] = SinglePassPCA,
        use_empirical_fischer: bool = False,
        num_eigs: int = 10,
        num_samples: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Wraps a trained model with functionality for adding epistemic uncertainty estimation.
        Accelerated with batched dataset processing and forward pass functionality.

        args:
            model: base PyTorch model to equip with uncertainty metric
            output_dist_cls: distributions.DistributionLayer subclass output probability distribution
            sketch_cls: matrix sketching algorithm class (Gaussian or SRFT)
            use_empirical_fischer: weight sketch samples by loss
            num_eigs: low-rank estimate of the dataset Fischer (K)
            num_samples: sketch size (T)
            device: torch.device to store matrix sketch parameters
        """
        super().__init__()
        self._model = model
        self._output_dist = output_dist_cls()
        self._sketch_cls = sketch_cls
        self._use_empirical_fischer = use_empirical_fischer
        self._num_eigs = num_eigs
        self._num_samples: int = num_samples if num_samples is not None else self._num_eigs * 6 + 4
        self._device = device

        # Setup functional model
        self.functional_model = make_functional_with_buffers(self._model)
        self._num_params = int(sum(p.numel() for p in self._fparams if p.requires_grad))

        # batched Jacobian function transforms are dynamically setup
        self._compute_batched_jacobians: Optional[Callable[..., Tuple[Tensor, Tensor]]] = None
        self._in_dims: Optional[Tuple[Optional[int], ...]] = None

        # SCOD parameters
        self._gauss_newton_eigs = nn.Parameter(
            data=torch.zeros(self._num_eigs, device=self._device), requires_grad=False
        )
        self._gauss_newton_basis = nn.Parameter(
            data=torch.zeros(self._num_params, self._num_eigs, device=self._device),
            requires_grad=False,
        )
        self._configured = nn.Parameter(torch.zeros(1, dtype=torch.bool), requires_grad=False)

    def save(self, path: str) -> None:
        """Save SCOD parameters."""
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def load(self, path: str) -> None:
        """Load SCOD parameters and instantiate functional model."""
        state_dict = torch.load(path)
        super().load_state_dict(state_dict, strict=False)
        self.functional_model = make_functional_with_buffers(self._model)

    def process_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        input_keys: Optional[List[str]] = None,
        target_key: Optional[str] = None,
        dataloader_kwargs: Dict = {},
        inputs_only: bool = False,
    ) -> None:
        """Summarizes information about training data by logging gradient directions
        seen during training and forming an orthonormal basis with Gram-Schmidt.
        Directions not seen during training are taken to be irrelevant to data,
        and used for detecting generalization.

        args:
            dataset: torch.utils.data.<Dataset/IterableDataset> returning a list, tuple or dictionary
            input_keys: List[str] of keys to extract inputs if dataset returns a dictionary (default: None)
            target_key: str key to extract targets if the dataset returns a dictionary (default: None)
            dataloader_kwargs: dictionary of kwargs for torch.utils.data.DataLoader class (default: {})
            inputs_only: the dataset only returns inputs
        """

        # Iterable dataset assumed to implement batching internally
        if isinstance(dataset, IterableDataset):
            if "batch_size" in dataloader_kwargs:
                del dataloader_kwargs["batch_size"]
            if "shuffle" in dataloader_kwargs:
                del dataloader_kwargs["shuffle"]
        dataloader = DataLoader(dataset, **dataloader_kwargs)

        # Incrementally build new sketch from samples
        self.functional_model = make_functional_with_buffers(self._model)
        sketch = self._sketch_cls(
            self._num_params, self._num_eigs, self._num_samples, device=self._device
        )
        for sample in dataloader:
            inputs, targets, batch_size = self._format_sample(
                sample, input_keys, target_key, inputs_only
            )
            # Compute test weight Fischer: L_w = J_f.T @ L_theta
            L_w, _ = self._compute_jacobians_outputs(inputs, targets, batch_size)
            sketch.low_rank_update(L_w)

        # Compute and store top-k eigenvalues and eigenvectors
        del L_w
        eigs, basis = sketch.eigs()
        del sketch
        self._gauss_newton_eigs.data = torch.clamp_min(eigs[-self._num_eigs :], min=0).to(
            self._device
        )
        self._gauss_newton_basis.data = basis[:, -self._num_eigs :].to(self._device)
        self._configured.data = torch.ones(1, dtype=torch.bool).to(self._device)

    def forward(
        self,
        sample: Union[Tensor, Tuple[Tensor, ...], List[Tensor], Dict[str, Tensor]],
        input_keys: Optional[List[str]] = None,
        detach: bool = True,
        mode: int = 0,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Computes the desired uncertainty quantity of samples, e.g., the posterior predictive
        variance or the local KL-divergence of the model on the test input.

        args:
            sample: batch of samples with type torch.Tensor, tuple, list or dict
            input_keys: List[str] of keys to extract inputs if dataset returns a dictionary (default: None)
            detach: remove jacobians and model outputs from the computation graph
            mode: int defining the return uncertainty metrics from SCOD (default: 0)

        returns:
            outputs: predicted model outputs (B x d)
            variance: posterior predictive variance of shape (B x d)
            uncertainty: local KL-divergence scalar of size (B x 1)
        """
        assert self._configured, "Must call self.process_dataset() before self.forward()"

        inputs, _, batch_size = self._format_sample(sample, input_keys, inputs_only=True)
        L_w, outputs = self._compute_jacobians_outputs(inputs, None, batch_size, detach=detach)

        if mode == 0:
            variance, uncertainty = self._predictive_variance_and_kl_divergence(L_w)
        elif mode == 1:
            variance, uncertainty = self._posterior_predictive_variance(L_w), None
        elif mode == 2:
            variance, uncertainty = None, self._local_kl_divergence(L_w)
        else:
            raise NotImplementedError(f"Specified mode {mode} not in [0, 1, 2]")

        return outputs, variance, uncertainty

    def _predictive_variance_and_kl_divergence(self, L: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the variance of the posterior predictive distribution and the local
        KL-divergence of the output distribution against the posterior weight distribution.

        Note: While JT and L are both of shape (B x N x d), they are only identical
        if the output distribution has unit variance, rendering self._output_dist.apply_sqrt_F()
        negligible. The code-base assumes this case, and hence uses JT and L interchangeably.

        args:
            L: test weight Fischer with shape (B x N x d)

        returns:
            S: posterior predictive variance of shape (B x d)
            E: local KL-divergence scalar of size (B x 1)
        """
        UT_L = self._gauss_newton_basis.t() @ L
        D = (self._gauss_newton_eigs / (1 + self._gauss_newton_eigs))[:, None]
        S = L.transpose(2, 1) @ L - UT_L.transpose(2, 1) @ (D * UT_L)
        E = torch.sum(L**2, dim=(1, 2)) - torch.sum((torch.sqrt(D) * UT_L) ** 2, dim=(1, 2))
        return torch.diagonal(S, dim1=1, dim2=2), E.unsqueeze(-1)

    def _posterior_predictive_variance(self, JT: Tensor) -> Tensor:
        """Computes the variance of the posterior predictive distribution.

        args:
            JT: transposed Jacobian tensor of shape (B x N x d)

        returns:
            S: posterior predictive variance of shape (B x d)
        """
        UT_JT = self._gauss_newton_basis.t() @ JT
        D = (self._gauss_newton_eigs / (1 + self._gauss_newton_eigs))[:, None]
        S = JT.transpose(2, 1) @ JT - UT_JT.transpose(2, 1) @ (D * UT_JT)
        return torch.diagonal(S, dim1=1, dim2=2)

    def _local_kl_divergence(self, L: Tensor) -> Tensor:
        """Computes the local KL-divergence of the output distribution against the
        posterior weight distribution.

        args:
            L: test weight Fischer with shape (B x N x d)

        returns:
            E: local KL-divergence scalar of size (B x 1)
        """
        UT_L = self._gauss_newton_basis.t() @ L
        D = torch.sqrt((self._gauss_newton_eigs / (1 + self._gauss_newton_eigs)))[:, None]
        E = torch.sum(L**2, dim=(1, 2)) - torch.sum((D * UT_L) ** 2, dim=(1, 2))
        return E.unsqueeze(-1)

    def _compute_jacobians_outputs(
        self,
        inputs: List[Tensor],
        targets: Optional[Tensor],
        batch_size: int,
        detach: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """Computes the test or empirical weight Fischer of a batch of samples
        and the model outputs.

        args:
            inputs: model input tensors
            targets: ground truth target tensors
            batch_size: number of samples
            detach: remove jacobians and outputs from computation graph

        returns:
            jacobians: Jacobians of size (B x N x d)
            outputs: model predictions parameterizing the output distribution of size (B x d)
        """
        in_dims = (None,) * 3 + ((0,) if targets is not None else (None,)) + (0,) * len(inputs)
        if self._compute_batched_jacobians is None or in_dims != self._in_dims:
            # Setup batched Jacobian function transforms
            self._compute_batched_jacobians = vmap(
                func=jacrev(self._compute_fischer_stateless_model, argnums=1, has_aux=True),
                in_dims=in_dims,
            )
            self._in_dims = in_dims
        jacobians, outputs = self._compute_batched_jacobians(
            *self.functional_model, targets, *inputs
        )
        jacobians = self._format_jacobian(jacobians, batch_size, outputs.size(-1))
        assert jacobians.size() == (batch_size, self._num_params, outputs.size(-1))
        if detach:
            jacobians, outputs = jacobians.detach(), outputs.detach()
        return jacobians, outputs

    def _compute_fischer_stateless_model(
        self,
        fmodel: FunctionalModuleWithBuffers,
        params: Tuple[nn.Parameter],
        buffers: Dict[str, Optional[Tensor]],
        target: Tensor,
        *input: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute the models weight Fischer for a single sample. There are two cases, below:
        1) Test weight Fischer: contribution C = (J_l.T @ J_l), J_l = d(-log p(y|x))/dw
        2) Empirical weight Fischer: contribution C = (J_f.T @ L_theta @ L_theta.T @ J_f), J_f = df(x)/dw

        args:
            fmodel: functional form of model casted from nn.Module
            params: parameters of functional model
            buffers: buffers of the functional model
            target: grouth truth target tensor
            *input: model input tensors

        returns:
            pre_jacobian: factor by which to compute the weight Jacobian of size (d)
            output: model predictions parameterizing the output distribution of size (d)
        """
        input = tuple(x.unsqueeze(0) for x in input)
        outputs = fmodel(params, buffers, *input)
        pre_jacobians = (
            self._output_dist.apply_sqrt_F(outputs)
            if not self._use_empirical_fischer
            else -self._output_dist.log_prob(outputs, target.unsqueeze(0))
        )

        return pre_jacobians.squeeze(0), outputs.squeeze(0)

    def _format_sample(
        self,
        x: Union[Tensor, Tuple[Tensor, ...], List[Tensor], Dict[str, Tensor]],
        input_keys: Optional[List[str]] = None,
        target_key: Optional[str] = None,
        inputs_only: bool = False,
    ) -> Tuple[List[Tensor], Optional[Tensor], int]:
        """Format dataset sample to be used by model and loss functions.

        args:
            x: batch of samples with type Tensor, tuple, list or dict
            input_keys: List[str] of keys to extract inputs if dataset returns a dictionary (default: None)
            target_key: str key to extract targets if the dataset returns a dictionary (default: None)

        returns:
            inputs: model input tensors
            targets: grouth truth target tensors
            batch_size: number of samples
        """
        if isinstance(x, Tensor):
            inputs, targets = [x.to(self._device)], None
        elif isinstance(x, tuple) or isinstance(x, list):
            x = [_x.to(self._device) for _x in flatten(x)]
            inputs, targets = x[:-1], x[-1]
            if inputs_only:
                inputs, targets = inputs + [targets], None
        elif isinstance(x, dict):
            assert input_keys is not None, "Require keys to extract inputs"
            inputs = [x[k].to(self._device) for k in input_keys]
            targets = x[target_key].to(self._device) if target_key is not None else None
        else:
            raise TypeError("x must be of type torch.Tensor, dict, tuple or list")
        batch_size = inputs[0].size(0)

        if inputs_only:
            assert targets is None, "targets must be None"
        else:
            assert not (
                self._use_empirical_fischer and targets is None
            ), "Require targets to compute empirical Fischer"

        return inputs, targets, batch_size

    @staticmethod
    def _format_jacobian(x: Iterable[Tensor], batch_size: int, output_dim: int) -> Tensor:
        """Returns flattenned, contiguous Jacobian.

        args:
            x: List of parameter Jacobians of size (B x d x n1 x n2 x ...)
            batch_size: number of samples
            output_dim: dimension of the output

        returns:
            x: formatted Jacobian of shape (B x N x d)
        """
        x = torch.cat([_x.contiguous().view(batch_size, output_dim, -1) for _x in x], dim=-1)
        return x.transpose(2, 1)
