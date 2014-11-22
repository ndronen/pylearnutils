"""
Experimental layers.
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
        "max", "min", or "mean".
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
        self.operation = operation

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        assert state_below.ndim == 2

        W, = self.transformer.get_params()
        if self.operation == 'prod':
            result = elemwise_prod(W, state_below)
        elif self.operation == 'sum':
            result = elemwise_sum(W, state_below)
        elif self.operation == 'min':
            result = elemwise_min(W, state_below)
        elif self.operation == 'max':
            result = elemwise_max(W, state_below)
        elif self.operation == 'mean':
            result = elemwise_mean(W, state_below)
        else:
            raise ValueError("Unknown operation type: " +
                    str(self.operation))

        ###################################################################
        # TODO: change the elemwise_* functions to return a binary #
        # mask denoting the features (the rows of the weight matrix) # that
        # were selected by this batch of inputs.  Then use that # mask to
        # modify the next parameter update so that the only parameters
        # that are updated are those that belong to features that were
        # used to represent this batch of inputs.
        # 
        # Note: if we assign to self.mask_weights and self.weights, then
        # Linear._modify_updates will ensure that only the weights of the
        # unmasked features are updated.
        ###################################################################

        # self.mask_weights = mask
        # self.mask = mask

        return result.astype(theano.config.floatX)

    @wraps(Layer.cost)
    def cost(self, *args, **kwargs):
        raise NotImplementedError()

def elemwise_prod_np(W, x):
    """
    Find the non-zero rows of x (the input to the network).
    If a row is a 0 vector, the sum will be 0, and applying
    that value as a mask to W will cause W to be 0.  Then
    multiplying the masked W by each row of x will cause
    the undesired rows of W to become 0 vectors.  At that
    point any operation (prod, sum, min, max) can be applied
    along the columns of the masked W.
    """
    prod = np.zeros((x.shape[0], W.shape[1]))
    for i in range(x.shape[0]):
        mask = x[i, :]
        mask = mask.reshape((mask.shape[0], 1))
        temp = mask * W
        temp[mask.flatten() == 0, :] = 1
        prod[i, :] = np.prod(temp, axis=0)

    rowsums = x.sum(axis=1)
    prod[rowsums == 0, :] = 0
    return prod

def elemwise_prod(W, x):
    def scanfun(x_row_index, W, x):
        mask = x[x_row_index, :]
        mask = T.reshape(mask, (x.shape[1], 1))
        masked = mask * W
        # Set to 1 all rows of `masked` where the mask is 0.
        mask = T.flatten(mask)
        zeros = (mask - 1).nonzero()
        masked = T.set_subtensor(masked[zeros, :], 1)

        # If necessary, replace the 1-vector rows with 0s.
        return T.switch(
                T.eq(T.sum(mask), 0),
                T.zeros((1, W.shape[1])),
                T.prod(masked, axis=0)).flatten()

    # Iterate over the rows of the input.
    x_row_indices = T.arange(x.shape[0])

    results, updates = theano.scan(
            fn=scanfun,
            outputs_info=None,
            sequences=x_row_indices,
            non_sequences=[W, x])

    return results

def elemwise_sum(W, x):
    def scanfun(x_row_index, W, x):
        mask = x[x_row_index, :]
        mask = T.reshape(mask, (x.shape[1], 1))
        masked = mask * W
        return T.sum(masked, axis=0).flatten()

    # Iterate over the rows of the input.
    x_row_indices = T.arange(x.shape[0])

    results, updates = theano.scan(
            fn=scanfun,
            outputs_info=None,
            sequences=x_row_indices,
            non_sequences=[W, x])

    return results

def elemwise_min(W, x):
    def scanfun(x_row_index, W, x):
        mask = x[x_row_index, :]
        mask = T.reshape(mask, (x.shape[1], 1))
        masked = mask * W
        # Set to 1 all rows of `masked` where the mask is 0.
        mask = T.flatten(mask)
        zeros = (mask - 1).nonzero()
        masked = T.set_subtensor(masked[zeros, :], 10)

        # If necessary, replace the large-number rows with 0s.
        return T.switch(
                T.eq(T.sum(mask), 0),
                T.zeros((1, W.shape[1])),
                T.min(masked, axis=0)).flatten()

    # Iterate over the rows of the input.
    x_row_indices = T.arange(x.shape[0])

    results, updates = theano.scan(
            fn=scanfun,
            outputs_info=None,
            sequences=x_row_indices,
            non_sequences=[W, x])

    return results

def elemwise_max(W, x):
    def scanfun(x_row_index, W, x):
        mask = x[x_row_index, :]
        mask = T.reshape(mask, (x.shape[1], 1))
        masked = mask * W
        mask = T.flatten(mask)

        # If necessary, replace the large-number rows with 0s.
        return T.switch(
                T.eq(T.sum(mask), 0),
                T.zeros((1, W.shape[1])),
                T.max(masked, axis=0)).flatten()

    # Iterate over the rows of the input.
    x_row_indices = T.arange(x.shape[0])

    results, updates = theano.scan(
            fn=scanfun,
            outputs_info=None,
            sequences=x_row_indices,
            non_sequences=[W, x])

    return results

def elemwise_mean(W, x):
    def scanfun(x_row_index, W, x):
        mask = x[x_row_index, :]
        mask = T.reshape(mask, (x.shape[1], 1))
        masked = mask * W
        mask = T.flatten(mask)
        nnz = mask.nonzero()[0].shape[0]
        #mean = T.sum(masked, axis=0).flatten()/nnz
        return T.switch(
                # If this row of the input is all zeros, then
                T.eq(T.sum(mask), 0),
                # return a vector of 0s, else
                T.zeros((1, W.shape[1])),
                # return the average across the non-zero rows.
                (T.sum(masked, axis=0)/nnz)).flatten()

    # Iterate over the rows of the input.
    x_row_indices = T.arange(x.shape[0])

    results, updates = theano.scan(
            fn=scanfun,
            outputs_info=None,
            sequences=x_row_indices,
            non_sequences=[W, x])

    return results

class TestElementwise(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(seed=17)
        self.nin = 5
        self.nhid = 3
        self.W = self.rng.uniform(size=(self.nin, self.nhid))
        self.W = self.W.astype(theano.config.floatX)

    def test_elemwise_prod_np(self):
        """
        Notes
        --------
        Two rows of inputs, one that has a single non-zero entry,
        one that has no non-zero entries.
        """
        x = np.zeros((2, self.nin))
        x[0, 0] = 1
        prod = elemwise_prod_np(self.W, x)

        self.assertTrue(prod.shape == (2, self.nhid))
        self.assertTrue(np.all(prod[0, :] == self.W[0, :]))
        self.assertTrue(np.all(prod[1, :] == np.zeros((1, self.nhid))))

    def test_elemwise_prod(self):
        """
        Notes
        --------
        Two rows of inputs, one that has a single non-zero entry,
        one that has no non-zero entries.
        """
        x = np.zeros((2, self.nin))
        x = x.astype(np.int8)
        x[0, 0] = 1

        Wt = T.matrix('W')
        xt = T.matrix('x')

        results = elemwise_prod(Wt, xt)
        f = theano.function([Wt, xt], outputs=results)
        prod = f(self.W, x)

        self.assertTrue(prod.shape == (2, self.nhid))
        self.assertTrue(np.all(prod[0, :] == self.W[0, :]))
        self.assertTrue(np.all(prod[1, :] == np.zeros((1, self.nhid))))

    def test_elemwise_sum(self):
        """
        Notes
        --------
        Two rows of inputs, one that has a single non-zero entry,
        one that has no non-zero entries.
        """
        x = np.zeros((2, self.nin))
        x = x.astype(np.int8)
        x[0, 0] = 1
        x[0, 1] = 1

        Wt = T.matrix('W')
        xt = T.matrix('x')
        results = elemwise_sum(Wt, xt)
        f = theano.function([Wt, xt], outputs=results)
        sum = f(self.W, x)

        self.assertTrue(sum.shape == (2, self.nhid))
        self.assertTrue(np.all(sum[0, :] == np.sum(self.W[0:2, :], axis=0)))
        self.assertTrue(np.all(sum[1, :] == np.zeros((1, self.nhid))))

    def test_elemwise_min(self):
        """
        Notes
        --------
        Two rows of inputs, one that has a single non-zero entry,
        one that has no non-zero entries.
        """
        x = np.zeros((2, self.nin))
        x = x.astype(np.int8)
        x[0, 0] = 1
        x[0, 4] = 1

        Wt = T.matrix('W')
        xt = T.matrix('x')
        results = elemwise_min(Wt, xt)
        f = theano.function([Wt, xt], outputs=results)
        min = f(self.W, x)

        self.assertTrue(min.shape == (2, self.nhid))
        self.assertTrue(np.all(min[0, :] == np.min(self.W[[0,4], :], axis=0)))
        self.assertTrue(np.all(min[1, :] == np.zeros((1, self.nhid))))

    def test_elemwise_mean(self):
        """
        Notes
        --------
        Two rows of inputs, one that has a single non-zero entry,
        one that has no non-zero entries.
        """
        x = np.zeros((2, self.nin))
        x = x.astype(np.int8)
        x[0, 0] = 1
        x[0, 4] = 1

        Wt = T.matrix('W')
        xt = T.matrix('x')
        results = elemwise_mean(Wt, xt)
        f = theano.function([Wt, xt], outputs=results)
        mean = f(self.W, x)

        self.assertTrue(mean.shape == (2, self.nhid))
        self.assertTrue(np.all(mean[0, :] == np.mean(self.W[[0,4], :], axis=0)))
        self.assertTrue(np.all(mean[1, :] == np.zeros((1, self.nhid))))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).setLevel(logging.INFO)
    unittest.main()
