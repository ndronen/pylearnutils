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

class MatrixMarket(SparseDataset):
    """
    A dataset for Market Matrix files.

    Parameters
    ----------
    path : The path to the Market Matrix file.
    """
    def __init__(self, path):
        """
        .. todo::

            WRITEME
        """
        X = scipy.io.mmread(path)
        super(MatrixMarket, self).__init__(
                from_scipy_sparse_dataset=X)

