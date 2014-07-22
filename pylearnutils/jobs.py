from jobman.tools import DD
import numpy as np

"""
Borrowed from

    http://vdumoulin.github.io/articles/pylearn2-jobman/.
"""

def extract_everything(train_obj):
    dd = DD()
    channels = train_obj.model.monitor.channels
    for k in channels.keys():
        values = channels[k].val_record
        dd[k] = values
    return dd

def log_uniform(low, high):
    """
    Generates a number that's uniformly distributed in the log-space
    between `low` and `high`

    Parameters
    ----------
    low : float
        Lower bound of the randomly generated number
    high : float
        Upper bound of the randomly generated number

    Returns
    -------
    rval : float
    	Random number uniformly distributed in the log-space specified
    	by `low` and `high`
    """
    log_low = np.log(low)
    log_high = np.log(high)

    log_rval = np.random.uniform(log_low, log_high)
    rval = float(np.exp(log_rval))

    return rval
