from theano import tensor
from pylearn2.costs.cost import Cost
from pylearn2.space import CompositeSpace

class NearestNeighborCost(Cost):
    """
    Cost for training autoencoders to reconstruct the nearest
    neighbor of the training examples.  For background, see:

        http://arxiv.org/abs/1407.7906
    """
    @staticmethod
    def cost(target, output):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError

    def expr(self, model, data, *args, **kwargs):
        """
        .. todo::

            WRITEME
        """
        self.get_data_specs(model)[0].validate(data)
        X, Y = data
        try:
            return self.cost(Y, model.reconstruct(X))
        except AttributeError:
            return self.cost(Y, model.fprop(X))

    def get_data_specs(self, model):
        """
        When training an autoencoder to reconstruct the input, the output
        space has the same dimension as the input space.  If we were to
        use DefaultDataSpecsMixin.get_data_specs, we would get an error
        during training when the monitoring callbacks fire because something
        is expecting the target (viz. the matrix of nearest neighbors) to 
        have the same number of dimensions as number of units in the model's
        hidden layer.  This method forces the output space and the input 
        space to be identical.

        Parameters
        ----------
        model : pylearn2.models.Model
        """
        space = CompositeSpace([model.get_input_space(),
            model.get_input_space()])
        sources = (model.get_input_source(), model.get_target_source())
        return (space, sources)

class MeanSquaredReconstructionError(NearestNeighborCost):
    """
    .. todo::

        WRITEME
    """

    @staticmethod
    def cost(target, output):
        """
        .. todo::

            WRITEME
        """
        return ((target - output) ** 2).sum(axis=1).mean()

class MeanBinaryCrossEntropy(NearestNeighborCost):
    """
    .. todo::

        WRITEME
    """

    @staticmethod
    def cost(target, output):
        """
        .. todo::

            WRITEME
        """
        return tensor.nnet.binary_crossentropy(output, target).sum(axis=1).mean()
