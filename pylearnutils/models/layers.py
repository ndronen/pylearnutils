"""
Experimental layers.
"""
__authors__ = "Nicholas Dronen"
__copyright__ = "Copyright 2014-2015, Nicholas Dronen"
__credits__ = ["Nicholas Dronen"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicholas Dronen <ndronen@gmail.com>"

import logging
import numpy as np
import theano.tensor as T
from pylearn2.models.mlp import Layer, Linear
from pylearn2.utils import wraps

logger = logging.getLogger(__name__)

class Elementwise(Linear):
    """
    Traditionally, the least computation perfomed by a linear layer is
    a matrix multiplication (i.e. dot product) of a weight matrix `W`
    and an input vector `x`.  Consequently, if `W` is nhid X nin and, say,
    `x` is nin X 1, the output of such a layer will be nhid X 1.
    This makes sense when `x` is dense.

    Other computations are possible when `x` is sparse.  Instead of
    computing the dot product of `W` and `x`, we can use `x` as a
    mask for selecting rows of `W` and perform operations on the
    resulting set of rows.  For lack of a better term, we currently
    call this an element-wise layer.  

    This implementation provides element-wise sum, product, mean, min
    and max.  (When `x` is binary, element-wise sum is equivalent to 
    dot product.)

    Parameters
    ----------
    operation : str
        The element-wise operation to perform: "sum", "product",
        "mean", "max", or "min".
    kwargs : dict
        Keyword arguments to pass to `Linear` class constructor
        or to implementation of operations.  Element-wise max (min)
        can be called with a boolean `use_abs` that determines whether the
        the max (min) is taken over the absolute values of `x`.

    Notes
    ----------
    This subclasses `Linear` only because it is convenient to do so.
    """
    def __init__(self, operation, **kwargs):
        super(Elementwise, self).__init__(**kwargs)
        self.scanner = OuterScanner(InnerScanner(operation))

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        W, = self.transformer.get_params()
        return self.scanner.scan(W, state_below)

    @wraps(Layer.cost)
    def cost(self, *args, **kwargs):
        raise NotImplementedError()

class InnerScanner(object):
    """
    WRITEME
    """
    def __init__(self, operation):
        self.operation = operation
        if not hasattr(self, self.operation):
            raise ValueError("'operation' " + str(operation) + 
                    " is invalid.  " + str(type(self)) + " doesn't have a " +
                    "method named " + str(self.operation))

    def prod(self, W_row_index, accumulator, W):
        """
        Initial value of accumulator: ones (but this will produce
        incorrect results if `x` is 0s.
        WRITEME
        """
        return T.prod(accumulator, W[W_row_index, :])

    def sum(self, W_row_index, accumulator, W):
        """
        Initial value of accumulator: zeros.
        WRITEME
        """
        return T.sum(accumulator, W[W_row_index, :])

    def min(self, W_row_index, accumulator, W):
        """
        Initial value of accumulator: np.finfo(theano.config.floatX).max
        WRITEME
        """
        return T.min(accumulator, W[W_row_index, :])

    def max(self, W_row_index, accumulator, W):
        """
        Initial value of accumulator: zeros.
        WRITEME
        """
        return T.max(accumulator, W[W_row_index, :])

    def get_outputs_info(self, W):
        """
        WRITEME
        """
        if self.operation == "prod":
            return T.ones_like(W[0, :])
        elif self.operation == "sum":
            return T.zeros_like(W[0, :])
        elif self.operation == "min":
            return T.ones_like(W[0, :]) * np.fifo(theano.config.floatX) 
        elif self.operation == "max":
            return T.zeros_like(W[0, :])

    def scan(self, state_below_row_index, accumulator, W, state_below):
        """
        WRITEME
        """
        W_row_indices = (state_below[state_below_row_index, :] > 0).nonzero()
    
        results, updates = theano.scan(
            fn=getattr(self, self.operation),
            outputs_info=self.get_outputs_info(),
            ###############################################################
            # The lambda expression will get one row index for W
            # corresponding to each of the non-zero entries in this
            # particular row of state_below.
            ###############################################################
            sequences=W_row_indices,
            non_sequences=W)
    
        return results[-1]

class OuterScanner(object):
    """
    WRITEME
    """
    def __init__(self, inner_scanner):
        self.inner_scanner = inner_scanner

    def scan(self, state_below):
        """
        WRITEME
        """
        ###################################################################
        # Pylearn2 has weight matrices W with shape nin X nhid, inputs x
        # that are nobs X nin, and the matrix multiplication is done by
        # T.dot(x, W).
        ###################################################################
        state_below_row_indices = T.arange(state_below.shape[0])
    
        results, _= theano.scan(
            ###############################################################
            # The arguments to this function are (1) the current index into x,
            # (2) an accumulator, (3) a weight matrix, and (4) x itself.
            # x is a matrix of inputs, with one row per input example,
            # so when we call scan() here, we ask the function to compute the
            # element-wise product of W and each individual row of x.
            ###############################################################
            fn=self.inner_scanner.scan,
            outputs_info=inner_scanner.get_outputs_info(),
            sequences=state_below_row_indices,
            non_sequences=[W, state_below])
    
        return results
