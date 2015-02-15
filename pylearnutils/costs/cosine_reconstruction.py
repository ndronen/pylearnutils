"""
"""

from theano import tensor as T
import theano.sparse
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin

class MeanCosineReconstructionError(DefaultDataSpecsMixin, Cost):
    """
    """

    def cost(x, y):
        """
        """
        return T.dot(x, y)/(T.sqrt(T.sum(x**2))*T.sqrt(T.sum(y**2)))

    def expr(self, model, data, *args, **kwargs):
        """
        """
        self.get_data_specs(model)[0].validate(data)
        X = data
        return self.cost(X, model.reconstruct(X))
