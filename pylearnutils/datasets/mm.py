# -*- coding: utf-8 -*-
"""
A dataset for Matrix Market files.
"""
__authors__ = "Nicholas Dronen"
__copyright__ = "Copyright 2013, Nicholas Dronen"
__credits__ = ["Nicholas Dronen"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicholas Dronen"
__email__ = "ndronen@gmail.com"

import scipy.io 

from pylearn2.datasets.sparse_dataset import SparseDataset
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

from scipy.sparse import issparse

class SparseMatrixMarket(SparseDataset):
    """
    A dataset for Market Matrix files.

    Parameters
    ----------
    path : The path to the Market Matrix file.
    """
    def __init__(self, path, transpose=False):
        """
        .. todo::

            WRITEME
        """
        X = scipy.io.mmread(path)
        if transpose:
            X = X.T
        X = X.tocsr()
        super(SparseMatrixMarket, self).__init__(
                from_scipy_sparse_dataset=X)

class MatrixMarket(DenseDesignMatrix):
    """
    A dataset for Market Matrix files.

    Parameters
    ----------
    path : The path to the Market Matrix file.
    """
    def __init__(self, path, transpose=False):
        """
        .. todo::

            WRITEME
        """
        X = scipy.io.mmread(path)
        X = X.todense()
        if transpose:
            X = X.T
        super(MatrixMarket, self).__init__(X=X)

