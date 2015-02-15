"""
Experimental layer.  This doesn't work yet.  I don't know if it's necessary yet.
When learning word embeddings, you want back propagation only to affect the
words used in the forward pass.  It may be that
pylearn2.sandbox.nlp.models.nlp.ProjectionLayer does the right thing with respect
to back propagation.  I need to run some tests to know.
"""
__authors__ = "Nicholas Dronen"
__copyright__ = "Copyright 2014-2015, Nicholas Dronen"
__credits__ = ["Nicholas Dronen"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicholas Dronen <ndronen@gmail.com>"

import unittest
import logging
import numpy as np
import theano
import theano.tensor as T
from pylearn2.models.mlp import Layer, Linear
from pylearn2.utils import wraps

logger = logging.getLogger(__name__)

class MaskingLayer(Linear):
    """
    WRITEME

    Parameters
    ----------
    layer: pylearn2.models.mlp.Layer-like
        The inner layer.  The masking layer will mask the parameter
        updates of the inner layer.
    kwargs : dict
        Keyword arguments to pass to `Linear` class constructor....

    Notes
    ----------
    This subclasses `Linear` only because it is convenient to do so.
    """
    def __init__(self, layer, **kwargs):
        super(MaskingLayer, self).__init__(**kwargs)
        self.layer = layer
        self.mask = None

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        assert state_below.ndim == 2

        # The MatrixMul class in pylearn2 does T.dot(x, self.W), so x
        # must be (number of examples) X (width of input).  Here x
        # is called `state_below`.  We want the mask to be the same
        # shape as W, as we're going to multiply the mask and W
        # elementwise.  So the mask needs to (width of input) X (number
        # of hidden units).  Taking the columnar sum of x will yield a
        # 1-row matrix with a non-zero value for the inputs of x that
        # are non-zero.  Binarizing that row, repeating it as many 
        # times as there are hidden units, then transposing it should
        # yield a mask with the desired shape and values.

        # Take columnar sum.
        mask = state_below.sum(axis=0)
        # Binarize it.
        mask = T.set_subtensor(mask[mask != 0], 1)
        # Repeat it 
        mask = mask.repeat(self.layer.get_output_space().dim, axis=0)
        # Transpose it.
        mask = mask.T

        self.mask_weights = mask
        self.mask = mask

        return self.layer.fprop(state_below)


    @wraps(Layer.cost)
    def cost(self, *args, **kwargs):
        raise NotImplementedError()
