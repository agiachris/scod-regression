import torch
from torch import nn, distributions
from abc import abstractmethod


class DistributionLayer(nn.Module):
    """
    A layer mapping network output to a distribution object.
    """

    @abstractmethod
    def forward(self, z: torch.Tensor) -> distributions.Distribution:
        """
        Returns torch.Distribution object specified by z

        Args:
            z (torch.Tensor): parameter of output distribution
        """
        raise NotImplementedError

    @abstractmethod
    def marginalize_gaussian(
        self, z_mean: torch.Tensor, z_var: torch.Tensor
    ) -> distributions.Distribution:
        r"""
        If z \sim N(z_mean, z_var), estimates p(y) = E_z [ p(y \mid z) ]

        Args:
            z_mean (torch.Tensor): mean of z
            z_var (torch.Tensor): diagonal variance of z (same size as z_mean)

        Returns:
            distributions.Distribution: p(y)
        """
        raise NotImplementedError

    @abstractmethod
    def marginalize_samples(
        self, z_samples: torch.Tensor, batch_idx: int = 0
    ) -> distributions.Distribution:
        r"""
        Given samples of z, estimates p(y) = E_z [ p(y \mid z) ] over empirical distribution

        Args:
            z_samples (torch.Tensor): [..., M, ..., z_dim]
            batch_idx (int): index over which to marginalize, assumed to be 0

        Returns:
            distributions.Distribution: p(y)
        """
        raise NotImplementedError

    @abstractmethod
    def apply_sqrt_F(self, z: torch.Tensor) -> torch.Tensor:
        r"""
        The Fisher Information Matrix of the output distribution is given by
            $$ F = E_{y \sim p(y | z)}[ d^2/dz^2 \log p (y \mid z)] $$
        If we factor F(z) = L(z) L(z)^T
        This function returns [ L(z)^T ].detach() @ z

        Args:
            z (torch.Tensor): parameter of p(y | z)

        Returns:
            torch.Tensor: parameter scaled by the square root of the fisher matrix, [ L(z)^T ].detach() z
        """
        raise NotImplementedError

    @abstractmethod
    def metric(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Returns a metric of Error(y,z), e.g. MSE for regression, 0-1 error for classification

        Args:
            z (torch.Tensor): parameter of distribution
            y (torch.Tensor): target

        Returns:
            torch.Tensor: Error(y, z)
        """
        raise NotImplementedError

    def log_prob(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Returns log p( y | z)

        Args:
            z (torch.Tensor): parameter of distribution
            y (torch.Tensor): target

        Returns:
            torch.Tensor: log p ( labels | dist )
        """
        return self.forward(z).log_prob(y)

    def validated_log_prob(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Checks if each element of y is in the support of the distribution specified by z
        Computes mean log_prob only on the elements

        Args:
            z (torch.Tensor): parameter of distribution
            y (torch.Tensor): target

        Returns:
            torch.Tensor: log p ( valid_y | corresponding_z )
        """
        dist = self.forward(z)
        valid_idx = dist.support.check(y)
        # raise error if there are no valid datapoints in the batch
        assert torch.sum(valid_idx) > 0

        # construct dist keeping only valid slice
        valid_y = y[valid_idx]
        valid_z = z[valid_idx]
        return self.forward(valid_z).log_prob(valid_y)
