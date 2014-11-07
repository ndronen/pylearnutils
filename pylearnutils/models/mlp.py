import numpy as np
import theano
from pylearn2.models.mlp import Layer
from pylearn2.utils import wraps, sharedX

class LayerDelegator(Layer):
    def __init__(self, layer):
        self.layer = layer
        super(LayerDelegator, self).__init__()

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        self.layer.set_input_space(space)

    @wraps(Layer.get_params)
    def get_params(self):
        return self.layer.get_params()

    @wraps(Layer.get_mlp)
    def get_mlp(self):
        return self.layer.get_mlp()

    @wraps(Layer.set_mlp)
    def set_mlp(self, mlp):
        self.layer.set_mlp(mlp)

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
            state=None, targets=None):
        return self.layer.get_layer_monitoring_channels(
                state_below, state, targets)

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        return self.layer.fprop(state_below)

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):
        return self.layer.cost(Y, Y_hat)

    @wraps(Layer.cost_from_cost_matrix)
    def cost_from_cost_matrix(self, cost_matrix):
        return self.layer.cost_from_cost_matrix(cost_matrix)

    @wraps(Layer.cost_matrix)
    def cost_matrix(self, Y, Y_hat):
        return self.layer.cost_matrix(Y, Y_hat)

    @wraps(Layer.get_weights)
    def get_weights(self):
        return self.layer.get_weights()

    @wraps(Layer.set_weights)
    def set_weights(self, mlp):
        self.layer.set_weights(mlp)

    @wraps(Layer.get_biases)
    def get_biases(self):
        return self.layer.get_biases()

    @wraps(Layer.set_biases)
    def set_biases(self):
        self.layer.set_biases()

    @wraps(Layer.get_weights_format)
    def get_weights_format(self):
        return self.layer.get_weights_format()

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeff):
        return self.layer.get_weight_decay()

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeff):
        return self.layer.get_l1_weight_decay()

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

class WeightsFromNpyFileLayer(LayerDelegator):
    """
    The default behavior of pylearn2 classes is to initialize weight
    matrices with random values.  If you want more control over the
    weights, you can create a dense matrix out-of-band, save it to disc,
    then load it like so:

        !obj:pylearnutils.models.mlp.WeightsFromNpyFileLayer {
            path: 'W.npy',
            layer: !obj:pylearn2.models.mlp.Tanh {
                ...
            }
        },

    Parameters
    ----------
    path : str
        Path to a .npy file containing a dense numpy weight matrix.
    layer : pylearn2.models.mlp.Layer
        A Layer instance (e.g. Tanh, RectifiedLinear)
    """
    def __init__(self, path, layer, **kwargs):
        super(WeightsFromNpyFileLayer, self).__init__(layer)
        self.path = path

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        self.layer.set_input_space(space)
        W = np.load(self.path)
        W = W.astype(theano.config.floatX)
        self.layer.set_weights(W)

    def __getattr__(self, name):
        return getattr(self.layer, name)

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
