import theano.tensor as T

def relu(x):
    return T.switch(x > 0, 0, x)
