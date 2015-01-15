import joblib
import numpy as np
from pylearn2.train_extensions import TrainExtension

class MonitorKeepModelEveryEpoch(TrainExtension):
    def __init__(self, save_prefix, save_freq=1):
        self.save_prefix = save_prefix
        self.save_freq = save_freq
        self.i = 1

    def on_monitor(self, model, dataset, algorithm):
        if self.i % self.save_freq == 0:
            save_path = "{0}_{1:04d}.joblib".format(self.save_prefix, self.i)
            joblib.dump(model, save_path)
        self.i += 1


class SaveWeightsWithinEpoch(object):
    def __init__(self, save_prefix, save_freq=1):
        self.save_prefix = save_prefix
        self.save_freq = save_freq
        self.param_name = 'h1_W'
        self.i = 1

    def __call__(self, sgd):
        if self.i % self.save_freq == 0:
            save_path = "{0}_{1:04d}.npy".format(self.save_prefix, self.i)
            names = [x.name for x in sgd.params]
            i = names.index(self.param_name)
            np.save(save_path, sgd.params[i].get_value())
        self.i += 1
