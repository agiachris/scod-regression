from typing import Dict, List, Tuple, Union, Callable

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
        self._model_ensemble = getattr(model, "ensemble_output", False)
        self._device = device
        self._config = _BASE_SCOD_CONFIG.copy()
        self._config.update(config)
        
        # Neural network parameters to sketch
        self._trainable_params = list(filter(lambda x: x.requires_grad, self._model.parameters()))
        self._num_params = int(sum(p.numel() for p in self._trainable_params))

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

        # Call self.process_dataset before self.forward
        self._configured = False

    def process_dataset(self, 
                        dataset: Dataset,
                        input_keys: List[str]=None,
                        target_key: str=None,
                        dataloader_kwargs: Dict={}
                        ) -> None:
        """Summarizes information about training data by logging gradient directions
        seen during training, and then using gram schmidt of these to form
        an orthonormal basis. directions not seen during training are 
        taken to be irrelevant to data, and used for detecting generalization

        args:
            dataset: torch.utils.data.<Dataset/IterableDataset> returning a tuple or dictionary
            input_keys: List[str] of keys to extract inputs if dataset returns a dictionary (default: None)
            target_keys: str key to extract outputs if the dataset returns a dictionary (default: None)
            dataloader_kwargs: Dict of kwargs for torch.utils.data.DataLoader class (default: {})
        """

        # Iterable dataset assumed to implement batching internally
        if isinstance(dataset, IterableDataset):
            if "batch_size" in dataloader_kwargs: del dataloader_kwargs["batch_size"]
            if "shuffle" in dataloader_kwargs: del dataloader_kwargs["shuffle"]
        dataloader = DataLoader(dataset, **dataloader_kwargs)

        # Instantiate sketch
        sketch = self._sketch_cls(
            self._num_params, 
            self._num_eigs, 
            self._num_samples, 
            device=self._device
        )

        # Make functional model
        fmodel, params, buffers = make_functional_with_buffers(self._model)
        output_dim = params[-1].size(-1)

        # Incrementally build sketch from samples
        for i, sample in enumerate(dataloader):
            
            inputs, labels, batch_size = \
                self._format_sample(sample, input_keys=input_keys, target_key=target_key)           

            # Batched Jacobian function transforms
            in_dims = (None, None, None) + (0,) * len(inputs)
            in_dims += (0,) if labels is not None else (None,)
            compute_weight_jacobian = jacrev(self._compute_fischer_stateless_model, argnums=1)
            compute_batched_weight_jacobian = vmap(compute_weight_jacobian, in_dims=in_dims)
            
            # Compute L_w.T = L_theta.T @ J_f
            L_w = compute_batched_weight_jacobian(fmodel, params, buffers, *inputs, labels=labels)
            L_w = self._format_grad(L_w, batch_size, output_dim)
            assert L_w.dim() == 3 and L_w.size() == (batch_size, output_dim, self._num_params)
            sketch.low_rank_update(L_w.transpose(2, 1))
        
        # Compute top-k eigenvalue and basis
        del L
        eigs, basis = sketch.eigs()
        del sketch

        # Store top-k eigenvalue and basis
        self._gauss_newton_eigs.data = torch.clamp_min(eigs[-self._num_eigs:], min=torch.zeros(1)).to(self._device)
        self._gauss_newton_basis.data = basis[:, -self._num_eigs:].to(self._device)
        self._configured = True
    
    def _compute_fischer_stateless_model(self,
                                         fmodel: FunctionalModuleWithBuffers, 
                                         params: Tuple[nn.Parameter],
                                         buffers: Tuple[torch.Tensor], 
                                         *input: Tuple[torch.Tensor],
                                         label: torch.Tensor=None,
                                         ) -> torch.Tensor:
        """Compute the models weight Fischer for a single sample. There are two cases:
        1) Use output Fischer: constribution C = (J_l.T @ J_l), J_l = d(-log p(y|x))/dw
        2) Use empirical Fischer: constribution C = (J_f.T @ L_theta @ L_theta.T @ J_f), J_f = df(x)/dw

        args:
            fmodel: functional form of model casted from nn.Module
            params: parameters of functional model
            buffers: buffers of the functional model
            *input: tuple of input tensors
            label: target tensors to compute loss (default: None)
        
        returns:
            pre_jac_factor: factor to compute the weight Jacobian via reverse auto-differentiation
        """
        input = [x.unsqueeze(0) for x in input]
        theta = fmodel(params, buffers, *input)
        pre_jac_factor = -self._output_dist.validated_log_prob(theta, label.unsqueeze(0)) \
            if self._use_empirical_fischer else self._output_dist.apply_sqrt_F(theta)
        return pre_jac_factor.squeeze(0)


    # def forward(self, inputs, 
    #                   input_keys=None, 
    #                   n_eigs=None, 
    #                   Meps=5000, 
    #                   compute_unc=True,
    #                   compute_var=False,
    #                   detach=True):
    #     """Assumes inputs are of shape (N, input_dims...)
    #     where N is the batch dimension,
    #           input_dims... are the dimensions of a single input

    #     returns 
    #         mu = model(inputs) -- shape (N, 1)
    #         unc = hessian based uncertainty estimates shape (N)
    #         var = posterior predictive distribution variance
    #     """
    #     assert self._configured, "Must call process_dataset first before using model for predictions."

    #     if n_eigs is None: 
    #         n_eigs = self.num_eigs

    #     # Make forward and backward pass differentable processes
    #     if detach: autograd_hacks.disable_computation_graph()
    #     else: autograd_hacks.enable_computation_graph()

    #     if isinstance(inputs, dict):
    #         assert input_keys is not None, "Require keys to extract inputs"
    #         inputs = [inputs[k] for k in input_keys]
    #         mu = self._model(*inputs)
    #     elif isinstance(inputs, tuple): mu = self._model(*inputs)
    #     else: mu = self._model(inputs)
    #     mu = self._format_output(mu)
        
    #     unc = None
    #     if compute_unc:
    #         theta = self.output_dist.apply_sqrt_F(mu, exact=True)
    #         L = self._get_weight_jacobian(theta.mean(0), mu.size(0), detach=detach)
    #         unc = self.projector.compute_distance(L.transpose(2, 1),
    #                                               self.proj_type,
    #                                               n_eigs=n_eigs,
    #                                               Meps=Meps)

    #     var = None
    #     if compute_var:
    #         J = self._get_weight_jacobian(mu.mean(0), mu.size(0), detach=detach)
    #         var = self.projector.compute_distance(J.transpose(2, 1),
    #                                               "batch_posterior_pred_var",
    #                                               n_eigs=n_eigs,
    #                                               Meps=Meps)

    #     return dict(output=self.output_dist.output(mu), unc=unc, var=var)

    # def _get_weight_jacobian(self, vec, batch_size, detach=True):
    #     """Returns b x d x nparam matrix, with each row of each d x nparam matrix being d(vec[i])/d(weights)
    #     """
    #     assert vec.dim() == 1
    #     grad_vecs = []
    #     autograd_hacks.clear_model_gradients(self._model)
    #     for j in range(vec.size(0)):
    #         vec[j].backward(retain_graph=True, create_graph=not detach)
    #         autograd_hacks.compute_grad1(self._model)
    #         g = self._get_grad_vec(batch_size)
    #         if detach: g = g.detach()
    #         grad_vecs.append(g)
    #         autograd_hacks.clear_model_gradients(self._model)

    #     return torch.stack(grad_vecs).transpose(1, 0)

    # def _get_grad_vec(self, batch_size):
    #     """Returns gradient of NN parameters flattened into a vector
    #     assumes backward() has been called so each parameters grad attribute
    #     has been updated
    #     """
    #     grads = [p.grad1.contiguous().view(batch_size, -1) for p in self.trainable_params]
    #     return torch.cat(grads, dim=1)

    def _format_sample(self, 
                       x: Union[Dict[torch.Tensor], Tuple[torch.Tensor]], 
                       input_keys: List[str]=None, 
                       target_key: str=None
                       ) -> Tuple[List[torch.Tensor], torch.Tensor, int]:
        """Format dataset sample to be used by model and loss functions.

        args:
            x: dataset sample of type tuple or dict
            input_keys: List[str] of keys to extract inputs if dataset returns a dictionary (default: None)
            target_keys: str key to extract outputs if the dataset returns a dictionary (default: None)

        returns:
            inputs: input tensors to the model
            labels: target tensors to compute losses
            batch_size: size of the batch
        """
        if isinstance(x, dict):
            assert input_keys is not None, "Require keys to extract inputs"
            assert not (self._use_empirical_fischer and target_key is None), "Dataset does not provide labels"
            inputs = [x[k].to(self._device) for k in input_keys]
            labels = x[target_key] if target_key is not None else None
            batch_size = inputs[0].size(0)

        elif isinstance(x, tuple):
            assert input_keys is None, "Keys cannot be used to extract inputs from a tuple"
            inputs = [x[0].to(self._device)]
            labels = x[1].to(self._device)
            batch_size = inputs.size(0)
    
        else:
            raise TypeError("x must be of type dict or tuple")
        
        return inputs, labels, batch_size

    @staticmethod
    def _format_output(x: torch.Tensor, 
                       batch_size: int=None,
                       model_ensemble: bool=False
                       ) -> torch.Tensor:
        """Format model output.

        args:
            x: model output of size(s): (B), (B x d), (E x B x d), (E x B), (E x d)
            batch_size: batch size of model input (default: None)
            model_ensemble: first model output dimension is ensemble size (E)
        
        returns:
            x: formatted model output of size (B x d)
        """
        if model_ensemble:
            assert x.size(0) == 1, "Cannot format output for model ensembles"
            x = x.squeeze(0)
        if batch_size is not None:
            assert x.size(0) == batch_size
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return x

    @staticmethod
    def _format_grad(x: torch.Tensor,
                     batch_size: int,
                     output_dim: int
                     ) -> torch.Tensor:
        """Returns flattenned, contiguous Jacobian.
        
        args: 
            x: Jacobian of size (B x d x n1 x n2 x ...)
            batch_size: batch_size
            output_dim: dimension of the output

        returns:
            g: formatted Jacobian of shape (B x d x N)
        """
        x = [_x.contiguous().view(batch_size, output_dim, -1) for _x in x]
        g = torch.cat(x, dim=-1)
        return g
