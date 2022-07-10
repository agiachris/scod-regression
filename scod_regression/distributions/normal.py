from typing import Tuple

import torch
from torch import nn, distributions

from .distribution import DistributionLayer
from .extended_distribution import ExtendedDistribution


class NormalMeanParamLayer(DistributionLayer):
    """
    Implements Normal RV parameterized by the mean. The variance is a parameter of the layer.
    """

    def __init__(self, init_log_variance: torch.Tensor = torch.zeros(1)) -> None:
        super().__init__()
        self.log_variance = nn.Parameter(init_log_variance)

    @property
    def std_dev(self) -> torch.Tensor:
        return torch.exp(0.5 * self.log_variance)

    @property
    def var(self) -> torch.Tensor:
        return torch.exp(self.log_variance)

    def forward(self, z: torch.Tensor) -> distributions.Distribution:
        return distributions.Independent(
            distributions.Normal(loc=z, scale=self.std_dev.broadcast_to(z.size())),
            reinterpreted_batch_ndims=1,
        )

    def marginalize_gaussian(
        self, z_mean: torch.Tensor, z_var: torch.Tensor
    ) -> distributions.Distribution:
        combined_std_dev = torch.sqrt(self.std_dev**2 + z_var)
        return distributions.Independent(
            distributions.Normal(loc=z_mean, scale=combined_std_dev),
            reinterpreted_batch_ndims=1,
        )

    def marginalize_samples(
        self, z_samples: torch.Tensor, batch_idx: int = 0
    ) -> distributions.Distribution:
        combined_mean = z_samples.mean(batch_idx)
        combined_std_dev = torch.sqrt(
            (z_samples**2).mean(batch_idx) - combined_mean**2 + self.var
        )
        return distributions.Independent(
            distributions.Normal(loc=combined_mean, scale=combined_std_dev),
            reinterpreted_batch_ndims=1,
        )

    def apply_sqrt_F(self, z: torch.Tensor) -> torch.Tensor:
        return z / self.std_dev

    def metric(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return ((y - z) ** 2).sum(-1)


class NormalMeanDiagVarParamLayer(DistributionLayer):
    """
    Implements Normal RV where both mean and log var are input as parameters
    Assumes first half of z is mean, second half is log_var
    Only performs Fisher computation around mean.
    """

    def _get_mean_logvar(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N = z.shape[-1]
        assert N % 2 == 0
        return z[..., : N // 2], z[..., N // 2 :]

    def forward(self, z: torch.Tensor) -> distributions.Distribution:
        mean, logvar = self._get_mean_logvar(z)
        return distributions.Independent(
            distributions.Normal(loc=mean, scale=torch.exp(0.5 * logvar)),
            reinterpreted_batch_ndims=1,
        )

    def marginalize_gaussian(
        self, z_mean: torch.Tensor, z_var: torch.Tensor
    ) -> distributions.Distribution:
        mean_mean, logvar_mean = self._get_mean_logvar(z_mean)
        mean_var, logvar_var = self._get_mean_logvar(z_var)

        combined_std_dev = torch.sqrt(torch.exp(logvar_mean) + mean_var)
        return distributions.Independent(
            distributions.Normal(loc=mean_mean, scale=combined_std_dev),
            reinterpreted_batch_ndims=1,
        )

    def marginalize_samples(
        self, z_samples: torch.Tensor, batch_idx: int = 0
    ) -> distributions.Distribution:
        mean_samples, logvar_samples = self._get_mean_logvar(z_samples)
        var_mean = torch.exp(logvar_samples).mean(batch_idx)
        combined_mean = mean_samples.mean(batch_idx)
        combined_std_dev = torch.sqrt(
            (z_samples**2).mean(batch_idx) - combined_mean**2 + var_mean
        )
        return distributions.Independent(
            distributions.Normal(loc=combined_mean, scale=combined_std_dev),
            reinterpreted_batch_ndims=1,
        )

    def apply_sqrt_F(self, z: torch.Tensor) -> torch.Tensor:
        mean, logvar = self._get_mean_logvar(z)
        return mean / torch.exp(0.5 * logvar).detach()

    def metric(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mean, logvar = self._get_mean_logvar(z)
        return ((y - mean) ** 2).sum(-1)


class Normal(distributions.normal.Normal, ExtendedDistribution):
    """
    Implemetns a Normal distribution specified by mean and standard deviation
    Only supports diagonal variance matrices
    """

    def __init__(self, loc, scale, validate_args=None):
        """Creates Normal distribution object

        Args:
            loc (_type_): mean of normal distribution
            scale (_type_): standard deviation of normal distribution
        """
        super().__init__(loc, scale, validate_args=validate_args)

    def apply_sqrt_F(self, vec):
        """
        if the Fisher matrix in terms of self._param is LL^T,
        return L^T vec, blocking gradients through L

        Here, we assume only loc varies, and do not consider cov as a backpropable parameter

        F = Sigma^{-1}

        """
        return vec / self.stddev.detach()

    def marginalize(self, diag_var):
        """
        returns an approximation to the marginal distribution if the parameter
        used to initialize this distribution was distributed according to a Gaussian
        with a diagonal variance as given

        inputs:
            diag_var: variance of parameter (d,)
        """
        stdev = torch.sqrt(self.variance + diag_var)
        return Normal(loc=self.mean, scale=stdev)

    def merge_batch(self):
        diag_var = (
            torch.mean(self.mean**2, dim=0)
            - self.mean.mean(dim=0) ** 2
            + self.variance.mean(dim=0)
        )
        return Normal(loc=self.mean.mean(dim=0), scale=torch.sqrt(diag_var))

    def metric(self, y):
        return torch.mean(torch.sum((y - self.mean) ** 2, dim=-1))

    def validated_log_prob(self, labels):
        # all labels are in the support of a Normal distribution
        return self.log_prob(labels)
