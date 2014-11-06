import numpy as np

from pylearn2.models.mlp import Layer
# from pylearn2.utils import sharedX

def set_weights_from_npy_file(path, layer):
    """
    The default behavior of pylearn2 classes is to initialize weight
    matrices with random values.  If you want more control over the
    weights, you can create a dense matrix out-of-band, save it to disc,
    then load it like so:

        !obj:pylearnutils.models.set_weights_from_npy_file {
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
    W = np.load(path)
    layer.set_weights(W)
    return layer
