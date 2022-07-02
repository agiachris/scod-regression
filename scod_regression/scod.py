from typing import Dict, List, Tuple, Union

import torch
from torch import nn
from torch.utils.data import Dataset, IterableDataset, DataLoader
from functorch import make_functional_with_buffers, jacrev, vmap
from functorch._src.make_functional import FunctionalModuleWithBuffers

from .utils.constants import _BASE_SCOD_CONFIG
from . import distributions
from . import sketching


class SCOD(nn.Module):

    def __init__(self,
                 model: nn.Module,
                 config: Dict=_BASE_SCOD_CONFIG,
                 device: torch.DeviceObjType=torch.device("cpu")
                 ) -> None:
        """Wraps a trained model with functionality for adding epistemic uncertainty estimation.
        Accelerated with batched dataset processing and forward pass functionality.

        args:
            model: base PyTorch model to equip with uncertainty metric
            output_dist: distributions.DistributionLayer output probability distribution
            config: SCOD hyperparamters / configuration
            device: torch.DeviceObjType to store matrix sketch parameters
        """
        super().__init__()
        self._model = model
        self._device = device
        self._config = _BASE_SCOD_CONFIG.copy()
        self._config.update(config)
        
        # Neural network parameters to sketch
        trainable_params = list(filter(lambda x: x.requires_grad, self._model.parameters()))
        self._num_params = int(sum(p.numel() for p in trainable_params))

        # Output distribution
        self._output_dist = vars(distributions)[self._config["output_dist"]]
        self._use_empirical_fischer = self._config["use_empirical_fischer"]
        
        # Matrix sketching
        self._sketch_cls = vars(sketching)[self._config["sketch"]]
        self._num_eigs = self._config["num_eigs"]       
        self._num_samples = self._config["num_samples"] \
            if self._config["num_samples"] is not None else self._num_eigs * 6 + 4
        
        # SCOD parameters
        self._gauss_newton_eigs = nn.Parameter(
            data=torch.zeros(self._num_eigs, device=self._device),
            requires_grad=False
        )
        self._gauss_newton_basis = nn.Parameter(
            data=torch.zeros(self._num_params, self._num_eigs, device=self._device),
            requires_grad=False
        )

        # Attributes assigned upon self.process_dataset call
        self._fmodel = None
        self._params = None
        self._buffers = None
        self._compute_batched_jacobians = None
        self._configured = False

    @property
    def functional_model(self):
        """Get functorch functional model. Set parameter gradients to None.
        """
        for p in self._params:
            if p.grad is not None:
                p.grad = None
        return self._fmodel, self._params, self._buffers

    def process_dataset(self, 
                        dataset: Dataset,
                        input_keys: List[str]=None,
                        target_key: str=None,
                        dataloader_kwargs: Dict={}
                        ) -> None:
        """Summarizes information about training data by logging gradient directions
        seen during training and forming an orthonormal basis with Gram-Schmidt.
        Directions not seen during training are taken to be irrelevant to data, 
        and used for detecting generalization.

        args:
            dataset: torch.utils.data.<Dataset/IterableDataset> returning a tuple or dictionary
            input_keys: List[str] of keys to extract inputs if dataset returns a dictionary (default: None)
            target_key: str key to extract targets if the dataset returns a dictionary (default: None)
            dataloader_kwargs: dictionary of kwargs for torch.utils.data.DataLoader class (default: {})
        """

        # Iterable dataset assumed to implement batching internally
        if isinstance(dataset, IterableDataset):
            if "batch_size" in dataloader_kwargs: del dataloader_kwargs["batch_size"]
            if "shuffle" in dataloader_kwargs: del dataloader_kwargs["shuffle"]
        dataloader = DataLoader(dataset, **dataloader_kwargs)

        # Instantiate sketch and functional model
        sketch = self._sketch_cls(
            self._num_params, 
            self._num_eigs, 
            self._num_samples, 
            device=self._device
        )
        self._fmodel, self._params, self._buffers = make_functional_with_buffers(self._model)

        # Incrementally build sketch from samples
        for _, sample in enumerate(dataloader):
            inputs, targets, batch_size = self._format_sample(sample, input_keys, target_key)           
            
            # Create batched Jacobian function transforms
            if not self._configured:
                self._compute_batched_jacobians = vmap(
                    func=jacrev(self._compute_fischer_stateless_model, argnums=1, has_aux=True),
                    in_dims=(None,) * 3 + ((0,) if targets is not None else (None,)) + (0,) * len(inputs)
                )                

            # Compute test weight Fischer: L_w = J_f.T @ L_theta
            L_w, _ = self._compute_jacobians_outputs(inputs, targets, batch_size)
            sketch.low_rank_update(L_w)
        
        # Compute and store top-k eigenvalues and eigenvectors
        del L_w; eigs, basis = sketch.eigs(); del sketch
        self._gauss_newton_eigs.data = torch.clamp_min(eigs[-self._num_eigs:], min=torch.zeros(1)).to(self._device)
        self._gauss_newton_basis.data = basis[:, -self._num_eigs:].to(self._device)
        self._configured = True

    def forward(self, 
                sample,
                input_keys: List[str]=None,
                detach: bool=True,
                quantity: str="all",
                ) -> Dict[torch.Tensor]:
        """Computes the desired uncertainty quantity of samples, e.g., the posterior predictive 
        variance or the local KL-divergence of the model on the test input.

        args:
            sample: batch of tuple or dictionary samples
            input_keys: List[str] of keys to extract inputs if dataset returns a dictionary (default: None)
            detach: remove jacobians and model outputs from the computation graph
            quantity: the desired output quantities from SCOD (default: "all")

        returns:
            outputs: predicted model outputs
            variance: posterior predictive variance of shape (B x d)
            uncertainty: local KL-divergence scalar of size (B x 1)
        """
        assert self._configured, "Must call self.process_dataset() before self.forward()"

        inputs, _, batch_size = self._format_sample(sample, input_keys=input_keys)
        L_w, outputs = self._compute_jacobians_outputs(inputs, None, batch_size, detach=detach)
        
        if quantity == "all":
            variance, uncertainty = self._predictive_variance_and_kl_divergence(L_w)
        elif quantity == "variance":
            variance, uncertainty = self._posterior_predictive_variance(L_w), None
        elif quantity == "local_kl":
            variance, uncertainty = None, self._local_kl_divergence(L_w)
        else:
            raise NotImplementedError(f'Requested quantity not in ["all", "variance", "local_kl"]')

        return outputs, variance, uncertainty

    def _predictive_variance_and_kl_divergence(self, L: torch.Tensor) -> torch.Tensor:
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
        E = torch.sum(L**2, dim=(1, 2)) - torch.sum((torch.sqrt(D) * UT_L)**2, dim=(1, 2))
        return torch.diagonal(S, dim1=1, dim2=2), E

    def _posterior_predictive_variance(self, JT: torch.Tensor) -> torch.Tensor:
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

    def _local_kl_divergence(self, L: torch.Tensor) -> torch.Tensor:
        """Computes the local KL-divergence of the output distribution against the 
        posterior weight distribution.

        args:
            L: test weight Fischer with shape (B x N x d)
        
        returns:
            E: local KL-divergence scalar of size (B x 1)
        """
        UT_L = self._gauss_newton_basis.t() @ L
        D = torch.sqrt((self._gauss_newton_eigs / (1 + self._gauss_newton_eigs)))[:, None]
        E = torch.sum(L**2, dim=(1, 2)) - torch.sum((D * UT_L)**2, dim=(1, 2))
        return E

    def _compute_jacobians_outputs(self,
                                   inputs: List[torch.Tensor],
                                   targets: torch.Tensor,
                                   batch_size: int,
                                   detach: bool=True
                                   ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        
        jacobians, outputs = self._compute_batched_jacobians(*self.functional_model, targets, *inputs)
        jacobians = self._format_jacobian(jacobians, batch_size, outputs.size(-1))
        assert jacobians.dim() == 3 and jacobians.size() == (batch_size, self._num_params, outputs.size(-1))
        return jacobians.detach(), outputs.detach() if detach else jacobians, outputs

    def _compute_fischer_stateless_model(self,
                                         fmodel: FunctionalModuleWithBuffers, 
                                         params: Tuple[nn.Parameter],
                                         buffers: Tuple[torch.Tensor],
                                         target = torch.Tensor,
                                         *input: Tuple[torch.Tensor],
                                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the models weight Fischer for a single sample. There are two cases, below:
        1) Test weight Fischer: contribution C = (J_l.T @ J_l), J_l = d(-log p(y|x))/dw
        2) Empirical weight Fischer: contribution C = (J_f.T @ L_theta @ L_theta.T @ J_f), J_f = df(x)/dw

        args:
            fmodel: functional form of model casted from nn.Module
            params: parameters of functional model
            buffers: buffers of the functional model
            target: grouth truth target tensor
            *input: tuple of model input tensors
        
        returns:
            pre_jacobian: factor by which to compute the weight Jacobian of size (d)
            output: model predictions parameterizing the output distribution of size (d)
        """
        input = [x.unsqueeze(0) for x in input]
        outputs = fmodel(params, buffers, *input)
        pre_jacobians = self._output_dist.apply_sqrt_F(outputs) if not self._use_empirical_fischer \
            else -self._output_dist.validated_log_prob(outputs, target.unsqueeze(0))
        
        return pre_jacobians.squeeze(0), outputs.squeeze(0)

    def _format_sample(self, 
                       x: Union[Dict[torch.Tensor], Tuple[torch.Tensor]], 
                       input_keys: List[str]=None, 
                       target_key: str=None
                       ) -> Tuple[List[torch.Tensor], torch.Tensor, int]:
        """Format dataset sample to be used by model and loss functions.

        args:
            x: dataset sample of type tuple or dict
            input_keys: List[str] of keys to extract inputs if dataset returns a dictionary (default: None)
            target_key: str key to extract targets if the dataset returns a dictionary (default: None)

        returns:
            inputs: model input tensors
            targets: grouth truth target tensors
            batch_size: number of samples
        """
        if isinstance(x, dict):
            assert input_keys is not None, "Require keys to extract inputs"
            inputs = [x[k].to(self._device) for k in input_keys]
            targets = x[target_key] if target_key is not None else None
            batch_size = inputs[0].size(0)
        elif isinstance(x, tuple):
            inputs = [x[0].to(self._device)]
            targets = x[1].to(self._device)
            batch_size = inputs.size(0)
        else:
            raise TypeError("x must be of type dict or tuple")
        
        assert not (self._use_empirical_fischer and targets is None), "Require targets to compute empirical Fischer"
        return inputs, targets, batch_size

    @staticmethod
    def _format_jacobian(x: Tuple[torch.Tensor],
                         batch_size: int,
                         output_dim: int
                         ) -> torch.Tensor:
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
