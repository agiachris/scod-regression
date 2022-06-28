from abc import abstractmethod


class ExtendedDistribution:
    """
    AbstractBaseClass
    """

    @abstractmethod
    def apply_sqrt_F(self, vec):
        """
        if the Fisher matrix in terms of self._param is LL^T,
        return L^T vec, blocking gradients through L

        Args:
            vec (torch.Tensor): vector

        Returns:
            L^T vec (torch.Tensor)
        """
        raise NotImplementedError

    @abstractmethod
    def marginalize(self, diag_var):
        """
        returns an approximation to the marginal distribution if the parameter
        used to initialize this distribution was distributed according to a Gaussian
        with mean and a diagonal variance as given

        inputs:
            diag_var: variance of parameter (1,)
        """
        raise NotImplementedError

    @abstractmethod
    def merge_batch(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        raise NotImplementedError

    @abstractmethod
    def metric(self, y):
        """
        Computes an alternative metric for an obervation under this distribution,
        e.g., MSE for Normal distribution, or 0-1 error for categorical and bernoulli distributions.

        Returns:
            torch.Tensor:
        """
        raise NotImplementedError

    @abstractmethod
    def validated_log_prob(self, labels):
        """Returns log prob after throwing out labels which are outside
        the support of the distribution.

        Args:
            labels (torch.Tensor): observations of the distribution

        Returns:
            torch.Tensor: log p ( labels | dist )
        """
        raise NotImplementedError
