import numpy as np
import torch
from torch import distributions

from .distribution import DistributionLayer
from .extended_distribution import ExtendedDistribution


class BernoulliLogitsLayer(DistributionLayer):
    """
    Implements Bernoulli RV parameterized by logits.
    """

    def forward(self, z: torch.Tensor) -> distributions.Distribution:
        return distributions.Bernoulli(logits=z)

    def marginalize_gaussian(
        self, z_mean: torch.Tensor, z_var: torch.Tensor
    ) -> distributions.Distribution:
        kappa = 1.0 / torch.sqrt(1.0 + np.pi / 8 * z_var)
        return distributions.Bernoulli(logits=kappa * z_mean)

    def marginalize_samples(
        self, z_samples: torch.Tensor, batch_idx: int = 0
    ) -> distributions.Distribution:
        probs = torch.sigmoid(z_samples).mean(dim=batch_idx)
        return distributions.Bernoulli(probs=probs)

    def apply_sqrt_F(self, z: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(z)
        L = torch.sqrt(p * (1 - p)) + 1e-8  # for stability
        return L * z

    def metric(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes 0-1 classification error
        """
        return ((z >= 0) != y).float()


class Bernoulli(distributions.Bernoulli, ExtendedDistribution):
    """
    Implements Bernoulli RV specified either through a logit or a probability directly.
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
            F = 1/(p(1-p))
        if self._param is logits, then
            F = p(1-p), where p = sigmoid(logit)
        """
        p = self.probs.detach()
        L = torch.sqrt(p * (1 - p)) + 1e-10  # for stability

        if self.use_logits:
            return L * vec
        else:
            return vec / L

    def marginalize(self, diag_var):
        """
        returns an approximation to the marginal distribution if the parameter
        used to initialize this distribution was distributed according to a Gaussian
        with mean and a diagonal variance as given

        inputs:
            diag_var: variance of parameter (1,)
        """
        if self.use_logits:
            kappa = 1.0 / torch.sqrt(1.0 + np.pi / 8 * diag_var)
            p = torch.sigmoid(kappa * self.logits)
        else:
            p = self.probs  # gaussian posterior in probability space is not useful
        return Bernoulli(probs=p)

    def merge_batch(self):
        p_mean = self.probs.mean(dim=0)
        return Bernoulli(probs=p_mean)

    def metric(self, y):
        """
        classification error (1- accuracy)
        """
        return ((self.probs >= 0.5) != y).float()

    def validated_log_prob(self, labels):
        """
        computes log prob, ignoring contribution from labels which are out of the support
        """
        valid_idx = self.support.check(labels)
        # raise error if there are no valid datapoints in the batch
        assert torch.sum(valid_idx) > 0

        # construct dist keeping only valid slice
        labels_valid = labels[valid_idx]
        if self.use_logits:
            logits_valid = self.logits[valid_idx]
            return Bernoulli(logits=logits_valid).log_prob(labels_valid)
        else:
            probs_valid = self.probs[valid_idx]
            return Bernoulli(probs=probs_valid).log_prob(labels_valid)
