# -*- coding: utf-8 -*-

__author__ = 'Nicholas Dronen'
__email__ = 'ndronen@gmail.com'
__version__ = '0.1.0'

import theano
import numpy as np
import cPickle

def load_model(model_file):
    f = open(model_file)
    return cPickle.load(f)

def predict(model_file, X):
    model = load_model(model_file)
    return Predictor(model).predict(X)

def encode(model_file, X):
    model = load_model(model_file)
    return Encoder(model).encode(X)

class Predictor(object):
    """
    General comments about the class.
    """
    def __init__(self, model, return_all=False):
        """
        Constructor documentation.
        model: what the model is.
        """
        self.model = model
        X = theano.tensor.matrix('X')
        outputs = model.fprop(X, return_all=return_all)
        self.f = theano.function([X], outputs=outputs)

    def predict(self, X):
        """
        Predict function documentation.
        X: what X is.
        """
        return self.f(X)

class Encoder(object):
    """
    General comments about the class.
    """
    def __init__(self, model):
        """
        Constructor documentation.
        model: what the model is.
        """
        self.model = model

    def encode(self, X):
        """
        Predict function documentation.
        X: what X is.
        """
        X = theano.shared(
            np.asarray(X, dtype=theano.config.floatX),
            borrow=True)
        try: 
            f = theano.function([], outputs=self.model.encode(X))
            return f()
        except AttributeError:
            # Assume that the model is an MLP, that the topmost layer was
            # trained to reconstruct the input, and that we want the output
            # of the penultimate layer.
            f = theano.function([], outputs=self.model.fprop(X, return_all=True))
            return f()[-2]
