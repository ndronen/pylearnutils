from pylearn2.train_extensions import TrainExtension

import joblib

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