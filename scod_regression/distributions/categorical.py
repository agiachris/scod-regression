import numpy as np
import torch
from torch import distributions

from .distribution import DistributionLayer
from .extended_distribution import ExtendedDistribution


class CategoricalLogitLayer(DistributionLayer):
    """
    Implements Categorical distribution parameterized by logits
    """

    def forward(self, z: torch.Tensor) -> distributions.Distribution:
        return distributions.Categorical(logits=z)

    def marginalize_gaussian(
        self, z_mean: torch.Tensor, z_var: torch.Tensor
    ) -> distributions.Distribution:
        kappa = 1.0 / torch.sqrt(1.0 + np.pi / 8 * z_var)
        return distributions.Categorical(logits=kappa * z_mean)

    def marginalize_samples(
        self, z_samples: torch.Tensor, batch_idx: int = 0
    ) -> distributions.Distribution:
        probs = torch.softmax(z_samples, -1).mean(dim=batch_idx)
        return distributions.Categorical(probs=probs)

    def apply_sqrt_F(self, z: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(z, -1).detach()
        z_bar = (p * z).sum(-1, keepdim=True)
        return torch.sqrt(p) * (z - z_bar)

    def metric(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes 0-1 classification error
        """
        return (torch.argmax(z, dim=-1) != y).float()


class Categorical(distributions.categorical.Categorical, ExtendedDistribution):
    """
    Implements Categorical distribution specified by either probabilities directly or logits.
    """

    def __init__(self, probs=None, logits=None, validate_args=None):
        self.use_logits = False
        if probs is None:
            self.use_logits = True
        super().__init__(probs=probs, logits=logits, validate_args=validate_args)

    def apply_sqrt_F(self, vec):
        """
        if the Fisher matrix in terms of self._param is LL^T,
        return L^T vec, blocking gradients through L

        if self._param is probs, then
            F = (diag(p^{-1}))
        if self._param is logits, then
            F = (diag(p) - pp^T) = LL^T
        """
        p = self.probs.detach()
        if self.use_logits:
            vec_bar = torch.sum(p * vec, dim=-1, keepdim=True)
            return torch.sqrt(p) * (vec - vec_bar)
        else:
            return vec / (torch.sqrt(p) + 1e-8)

    def marginalize(self, diag_var):
        """
        returns an approximation to the marginal distribution if the parameter
        used to initialize this distribution was distributed according to a Gaussian
        with a diagonal variance as given

        inputs:
            diag_var: variance of parameter (d,)
        """
        if self.use_logits:
            # TODO: allow selecting this via an argument
            # probit approx
            kappa = 1.0 / torch.sqrt(1.0 + np.pi / 8 * diag_var)
            scaled_logits = kappa * self.logits
            dist = Categorical(logits=scaled_logits)

            # laplace bridge
            # d = diag_var.shape[-1]
            # sum_exp = torch.sum(torch.exp(-self.logits), dim=-1, keepdim=True)
            # alpha = 1. / diag_var * (1 - 2./d + torch.exp(self.logits)/(d**2) * sum_exp)
            # dist = distributions.Dirichlet(alpha)
            # return distributions.Categorical(probs=torch.nan_to_num(dist.mean, nan=1.0))
        else:
            p = self.probs  # gaussian posterior in probability space is not useful
            return Categorical(probs=p)
        return dist

    def validated_log_prob(self, labels):
        """
        computes log prob, ignoring contribution from labels which are out of the support
        """
        valid_idx = self.support().check(labels)
        # raise error if there are no valid datapoints in the batch
        if torch.sum(valid_idx) == 0:
            return torch.zeros_like(valid_idx).float()

        # construct dist keeping only valid slice
        labels_valid = labels[valid_idx]
        if self.use_logits:
            logits_valid = self.logits[valid_idx, ...]
            return Categorical(logits=logits_valid).log_prob(labels_valid)
        else:
            probs_valid = self.probs[valid_idx, ...]
            return Categorical(probs=probs_valid).log_prob(labels_valid)

    def merge_batch(self):
        p_mean = self.probs.mean(dim=0)
        return Categorical(probs=p_mean)

    def metric(self, y):
        """
        classification error (1- accuracy)
        """
        return (torch.argmax(self.probs, dim=-1) != y).float()
