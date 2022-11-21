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
    Mapping,
    Any,
)

import inspect
import numpy as np  # type: ignore
import torch  # type: ignore
from torch import nn, Tensor
from torch.utils.data import Dataset, IterableDataset, DataLoader
from functorch import make_functional_with_buffers, jacrev, vmap
from functorch._src.make_functional import FunctionalModuleWithBuffers

from .distributions.distribution import DistributionLayer
from .distributions.normal import NormalMeanParamLayer
from .sketching import SinglePassPCA
from .utils import tensors


class SCOD(nn.Module):
    """Sketching curvature for out-of-distribution detection (SCOD) class.
    Wraps a trained model with functionality for epistemic uncertainty quantification.
    Accelerated with batched dataset processing and forward pass functionality."""

    @property
    def device(self) -> torch.device:
        """Get current torch device."""
        return self._device

    @property
    def functional_model(
        self,
    ) -> Tuple[
        FunctionalModuleWithBuffers,
        Iterator[nn.Parameter],
        Iterator[nn.Parameter],
        Dict[str, Optional[Tensor]],
    ]:
        """Get functorch functional model."""
        for p in self._fgrad_params:
            if p.grad is not None:
                p.grad = None
        return self._fmodel[0], self._fstatic_params, self._fgrad_params, self._fbuffers

    @functional_model.setter
    def functional_model(
        self,
        functional_model: Tuple[Any, Iterator[nn.Parameter], Dict[str, Optional[Tensor]]],
    ):
        """Set functorch functional model."""
        fmodel, fparams, self._fbuffers = functional_model

        # Ensure frozen parameters come before requires grad parameters
        filter = [not p.requires_grad for p in fparams]
        m = sum(filter)
        if not all(x for x in filter[:m]) or any(x for x in filter[m:]):
            raise ValueError("Frozen and requires gradient layers cannot be interleaved.")

        self._fstatic_params = tuple(p for p in fparams if not p.requires_grad)
        self._fgrad_params = tuple(p for p in fparams if p.requires_grad)

        # Put fmodel in a list to avoid registering it as a child module
        # torch.nn.Module._apply() does not work with child FunctionModules
        self._fmodel = [fmodel]

    def __init__(
        self,
        model: nn.Module,
        output_agg_func: Optional[Union[str, Callable]] = None,
        output_dist_cls: Type[DistributionLayer] = NormalMeanParamLayer,
        sketch_cls: Type[SinglePassPCA] = SinglePassPCA,
        use_empirical_fischer: bool = False,
        num_eigs: int = 10,
        num_samples: Optional[int] = None,
        prior_scale: float = 1.0,
        device: Optional[Union[str, torch.device]] = None,
        checkpoint: Optional[str] = None,
    ) -> None:
        """Construct SCOD.

        args:
            model: Base PyTorch model to equip with uncertainty metric.
            output_agg_func: Output aggregation function if model outputs an Iterable.
            output_dist_cls: Distributions.DistributionLayer subclass output probability distribution.
            sketch_cls: Matrix sketching algorithm class (Gaussian or SRFT).
            use_empirical_fischer: Weight sketch samples by loss.
            num_eigs: Low-rank estimate of the dataset Fischer (K).
            num_samples: Sketch size (T).
            prior_scale: Gaussian isotropic prior variance.
            device: torch.device to store matrix sketch parameters.
            checkpoint: SCOD checkpoint with precomputed weights.
        """
        super().__init__()
        self._model = model
        self._output_agg_func: Optional[
            Callable[[Tensor, int], Union[Tensor, Tuple[Tensor, Tensor]]]
        ] = (
            getattr(torch, output_agg_func) if isinstance(output_agg_func, str) else output_agg_func
        )
        self._output_dist = output_dist_cls()
        self._sketch_cls = sketch_cls
        self._use_empirical_fischer = use_empirical_fischer
        self._num_eigs = num_eigs
        self._num_samples = num_samples if num_samples is not None else self._num_eigs * 6 + 4
        self._prior_scale = prior_scale
        self._device = tensors.device(device)

        # Setup functional model.
        self.functional_model = make_functional_with_buffers(self._model)
        self._num_params = int(sum(p.numel() for p in self._fgrad_params))

        # batched Jacobian function transforms are dynamically setup.
        self._compute_batched_jacobians: Optional[Callable[..., Tuple[Tensor, Tensor]]] = None
        self._in_dims: Optional[Tuple[Optional[int], ...]] = None

        # SCOD parameters.
        self._gauss_newton_eigs = nn.Parameter(
            data=torch.zeros(self._num_eigs, device=self._device), requires_grad=False
        )
        self._gauss_newton_basis = nn.Parameter(
            data=torch.zeros(self._num_params, self._num_eigs, device=self._device),
            requires_grad=False,
        )
        self._configured = nn.Parameter(torch.zeros(1, dtype=torch.bool), requires_grad=False)

        # Load checkpoint.
        if checkpoint is not None:
            self.load(checkpoint)

        self.to(self.device)

    def save(self, path: str) -> None:
        """Save SCOD parameters."""
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def load(self, path: str) -> None:
        """Load SCOD parameters."""
        state_dict = torch.load(path, map_location=self._device)
        self.load_state_dict(state_dict)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False) -> None:
        """Load SCOD state dict.

        args:
            state_dict: torch state dictionary.
        """
        super().load_state_dict(state_dict, strict=strict)
        self.functional_model = make_functional_with_buffers(self._model)

    def _apply(self, fn) -> "SCOD":
        """Called internally by torch to move tensors to a device."""
        super()._apply(fn)
        self.functional_model = make_functional_with_buffers(self._model)
        return self

    def to(self, device: Union[str, torch.device]) -> "SCOD":
        """Move SCOD module and nn.Parameters to device."""
        self._device = tensors.device(device)
        super().to(self.device)
        return self

    def train_mode(self) -> None:
        """Transfer model to train mode."""
        self._model.train()
        self.functional_model = make_functional_with_buffers(self._model)

    def eval_mode(self) -> None:
        """Transfer model to evaluation mode."""
        self._model.eval()
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
            dataset: torch.utils.data.<Dataset/IterableDataset> returning a list, tuple or dictionary.
            input_keys: List[str] of keys to extract inputs if dataset returns a dictionary.
            target_key: String key to extract targets if the dataset returns a dictionary.
            dataloader_kwargs: Dictionary of kwargs for torch.utils.data.DataLoader class.
            inputs_only: Dataset only returns inputs.
        """

        # Iterable dataset assumed to implement batching internally.
        if isinstance(dataset, IterableDataset):
            dataloader = iter(dataset)
        elif isinstance(dataset, Dataset):
            dataset = DataLoader(dataset, **dataloader_kwargs)
        else:
            raise ValueError("Dataset must be one of torch Dataset or IterableDataset.")

        # Incrementally build new sketch from samples.
        self.functional_model = make_functional_with_buffers(self._model)
        sketch = self._sketch_cls(
            self._num_params, self._num_eigs, self._num_samples, device=self._device
        )
        for sample in dataloader:
            inputs, targets, batch_size = self._format_sample(
                sample, input_keys, target_key, inputs_only
            )
            # Compute test weight Fischer: L_w = J_f.T @ L_theta.
            L_w, _ = self._compute_jacobians_outputs(inputs, targets, batch_size)
            sketch.low_rank_update(L_w)

        # Compute and store top-k eigenvalues and eigenvectors.
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
        sample: Union[
            Tensor,
            np.ndarray,
            Tuple[Union[Tensor, np.ndarray], ...],
            List[Union[Tensor, np.ndarray]],
            Dict[str, Union[Tensor, np.ndarray]],
        ],
        input_keys: Optional[List[str]] = None,
        detach: bool = True,
        mode: int = 0,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Computes the desired uncertainty quantity of samples, e.g., the posterior predictive
        variance or the local KL-divergence of the model on the test input.

        args:
            sample: Batch of samples with type torch.Tensor, tuple, list or dict.
            input_keys: List[str] of keys to extract inputs if dataset returns a dictionary.
            detach: Remove jacobians and model outputs from the computation graph.
            mode: Int defining the return uncertainty metrics from SCOD.

        returns:
            outputs: Predicated model outputs (B x d).
            variance: Posterior predictive variance of shape (B x d).
            uncertainty: Local KL-divergence scalar of size (B x 1).
        """
        if not self._configured:
            raise ValueError("Must call self.process_dataset() before self.forward().")

        inputs, _, batch_size = self._format_sample(sample, input_keys, inputs_only=True)
        L_w, outputs = self._compute_jacobians_outputs(inputs, None, batch_size, detach=detach)

        if mode == 0:
            variance, uncertainty = self._predictive_variance_and_kl_divergence(L_w)
        elif mode == 1:
            variance, uncertainty = self._posterior_predictive_variance(L_w), None
        elif mode == 2:
            variance, uncertainty = None, self._local_kl_divergence(L_w)
        elif mode == 3:
            variance, uncertainty = None, None
        else:
            raise NotImplementedError(f"Specified mode {mode} not in [0, 1, 2, 3].")

        return outputs, variance, uncertainty

    def _predictive_variance_and_kl_divergence(self, L: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the variance of the posterior predictive distribution and the local
        KL-divergence of the output distribution against the posterior weight distribution.

        Note: While JT and L are both of shape (B x N x d), they are only identical
        if the output distribution has unit variance, rendering self._output_dist.apply_sqrt_F()
        negligible. The code-base assumes this case, and hence uses JT and L interchangeably.

        args:
            L: Test weight Fischer with shape (B x N x d).

        returns:
            S: Posterior predictive variance of shape (B x d).
            E: Local KL-divergence scalar of size (B x 1).
        """
        UT_L = self._gauss_newton_basis.t() @ L
        D = (self._gauss_newton_eigs / (1 / self._prior_scale + self._gauss_newton_eigs))[:, None]
        S = self._prior_scale * (L.transpose(2, 1) @ L - UT_L.transpose(2, 1) @ (D * UT_L))
        E = self._prior_scale * (
            torch.sum(L**2, dim=(1, 2)) - torch.sum((torch.sqrt(D) * UT_L) ** 2, dim=(1, 2))
        )
        return torch.diagonal(S, dim1=1, dim2=2), E.unsqueeze(-1)

    def _posterior_predictive_variance(self, JT: Tensor) -> Tensor:
        """Computes the variance of the posterior predictive distribution.

        args:
            JT: Transposed Jacobian tensor of shape (B x N x d).

        returns:
            S: Posterior predictive variance of shape (B x d).
        """
        UT_JT = self._gauss_newton_basis.t() @ JT
        D = (self._gauss_newton_eigs / (1 / self._prior_scale + self._gauss_newton_eigs))[:, None]
        S = self._prior_scale * (JT.transpose(2, 1) @ JT - UT_JT.transpose(2, 1) @ (D * UT_JT))
        return torch.diagonal(S, dim1=1, dim2=2)

    def _local_kl_divergence(self, L: Tensor) -> Tensor:
        """Computes the local KL-divergence of the output distribution against the
        posterior weight distribution.

        args:
            L: Test weight Fischer with shape (B x N x d).

        returns:
            E: Local KL-divergence scalar of size (B x 1).
        """

        UT_L = self._gauss_newton_basis.t() @ L
        D = torch.sqrt(
            (self._gauss_newton_eigs / (1 / self._prior_scale + self._gauss_newton_eigs))
        )[:, None]
        E = self._prior_scale * (
            torch.sum(L**2, dim=(1, 2)) - torch.sum((D * UT_L) ** 2, dim=(1, 2))
        )
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
            inputs: Model input tensors.
            targets: Ground truth target tensors.
            batch_size: Number of samples.
            detach: Remove jacobians and outputs from computation graph.

        returns:
            jacobians: Jacobians of size (B x N x d).
            outputs: Model predictions parameterizing the output distribution of size (B x d).
        """
        forward_signature = inspect.signature(self._model.forward)
        num_args = len(forward_signature.parameters)
        if num_args == 1 and len(inputs) > 1:
            inputs = [torch.cat(inputs, dim=-1)]
        elif len(inputs) < num_args:
            raise ValueError("self._model.forward() expects more inputs than provided.")

        in_dims = (None,) * 4 + ((0,) if targets is not None else (None,)) + (0,) * len(inputs)
        if self._compute_batched_jacobians is None or in_dims != self._in_dims:
            # Setup batched Jacobian function transforms.
            self._compute_batched_jacobians = vmap(
                func=jacrev(self._compute_fischer_stateless_model, argnums=2, has_aux=True),
                in_dims=in_dims,
            )
            self._in_dims = in_dims
        assert self._compute_batched_jacobians is not None

        jacobians, outputs = self._compute_batched_jacobians(
            *self.functional_model, targets, *inputs
        )
        jacobians = self._format_jacobian(jacobians, batch_size, outputs.size(-1))

        if not jacobians.size() == (batch_size, self._num_params, outputs.size(-1)):
            raise ValueError(f"Failed to parse jacobian of size {jacobians.size()}.")

        if detach:
            jacobians, outputs = jacobians.detach(), outputs.detach()

        return jacobians, outputs

    def _compute_fischer_stateless_model(
        self,
        fmodel: FunctionalModuleWithBuffers,
        fstatic_params: Tuple[nn.Parameter],
        fgrad_params: Tuple[nn.Parameter],
        fbuffers: Dict[str, Optional[Tensor]],
        target: Tensor,
        *input: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute the models weight Fischer for a single sample. There are two cases, below:
        1) Test weight Fischer: contribution C = (J_l.T @ J_l), J_l = d(-log p(y|x))/dw
        2) Empirical weight Fischer: contribution C = (J_f.T @ L_theta @ L_theta.T @ J_f), J_f = df(x)/dw

        args:
            fmodel: Functional form of model casted from nn.Module.
            fstatic_params: Functional model parameters that are frozen.
            fgrad_params: Functional model parameters that require gradients.
            fbuffers: Buffers of the functional model.
            target: Grouth truth target tensor.
            *input: Model input tensors.

        returns:
            pre_jacobian: Factor by which to compute the weight Jacobian of size (d).
            output: Model predictions parameterizing the output distribution of size (d).
        """
        input = tuple(x.unsqueeze(0) for x in input)
        outputs = fmodel(fstatic_params + fgrad_params, fbuffers, *input)
        outputs = self._format_output(outputs)
        pre_jacobians = (
            self._output_dist.apply_sqrt_F(outputs)
            if not self._use_empirical_fischer
            else -self._output_dist.log_prob(outputs, target.unsqueeze(0))
        )

        return pre_jacobians.squeeze(0), outputs.squeeze(0)

    def _format_sample(
        self,
        x: Union[
            Tensor,
            np.ndarray,
            Tuple[Union[Tensor, np.ndarray], ...],
            List[Union[Tensor, np.ndarray]],
            Dict[str, Union[Tensor, np.ndarray]],
        ],
        input_keys: Optional[List[str]] = None,
        target_key: Optional[str] = None,
        inputs_only: bool = False,
    ) -> Tuple[List[Tensor], Optional[Tensor], int]:
        """Format dataset sample to be used by model and loss functions.

        args:
            x: Batch of samples with type Tensor, tuple, list or dict.
            input_keys: List[str] of keys to extract inputs if dataset returns a dictionary.
            target_key: String key to extract targets if the dataset returns a dictionary.
            inputs_only: Dataset only returns inputs.

        returns:
            inputs: Model input tensors.
            targets: Grouth truth target tensors.
            batch_size: Number of samples.
        """
        x = tensors.to(x, self._device)
        if isinstance(x, Tensor):
            inputs, targets = [x], None
        elif isinstance(x, tuple) or isinstance(x, list):
            x = [_x for _x in tensors.flatten(x)]
            inputs, targets = x[:-1], x[-1]
            if inputs_only:
                inputs, targets = inputs + [targets], None
        elif isinstance(x, dict):
            assert input_keys is not None, "Require keys to extract inputs."
            inputs = [x[k] for k in input_keys]
            targets = x[target_key] if target_key is not None else None
        else:
            raise TypeError("x must be of type torch.Tensor, dict, tuple or list.")
        batch_size = inputs[0].size(0)

        if inputs_only:
            assert targets is None, "targets must be None."
        else:
            assert not (
                self._use_empirical_fischer and targets is None
            ), "Require targets to compute empirical Fischer."

        return inputs, targets, batch_size

    def _format_output(self, x: Union[Tensor, Tuple[Tensor, ...], List[Tensor]]) -> Tensor:
        """Returns formatted output.

        args:
            x: Tensor or Iterable of Tensors.
            output_agg_func: Output aggregation function if model outputs an Iterable.

        returns:
            x: Formatted output of shape (1 x d).
        """
        if isinstance(x, (tuple, list)):
            x = torch.stack(x, dim=0)

        if x.dim() == 2 and x.size(0) > 1:
            if self._output_agg_func is None:
                raise ValueError(
                    "output_agg_func must be a torch function if model outputs an Iterable."
                )
            x = self._output_agg_func(x, dim=0)
            if not isinstance(x, Tensor):
                try:
                    x: Tensor = getattr(x, "values")
                except AttributeError:
                    raise ValueError(
                        f"Aggregation function {self._output_agg_func} output of type {type(x)} cannot be parsed."
                    )

        if x.dim() == 1:
            x = x.unsqueeze(0)

        return x

    @staticmethod
    def _format_jacobian(x: Iterable[Tensor], batch_size: int, output_dim: int) -> Tensor:
        """Returns flattenned, contiguous Jacobian.

        args:
            x: List of parameter Jacobians of size (B x d x n1 x n2 x ...).
            batch_size: Number of samples.
            output_dim: Dimension of the output.

        returns:
            x: Formatted Jacobian of shape (B x N x d).
        """
        x = torch.cat([_x.contiguous().view(batch_size, output_dim, -1) for _x in x], dim=-1)
        return x.transpose(2, 1)
