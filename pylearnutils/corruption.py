import numpy as np
from pylearn2.corruption import Corruptor

class SparseBinomialCorruptor(Corruptor):
    """
    A binomial corruptor that adds 1 to inputs with probability
    0 < `corruption_level` < 1.
    """

    def _corrupt(self, x):
        """
        Corrupts a single tensor_like object.

        Parameters
        ----------
        x : tensor_like
            Theano symbolic representing a (mini)batch of inputs to be
            corrupted, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        corrupted : tensor_like
            Theano symbolic representing the corresponding corrupted input.
        """
        ncol = x.shape[0]
        indices = np.random.choice(ncol, size=int(ncol*self.corruption_level))
        x[indices] = 1
        return x
