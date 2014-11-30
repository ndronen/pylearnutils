from pylearn2.train_extensions import TrainExtension
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest

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

class SaveBestAfterKEpochs(MonitorBasedSaveBest):
    """
    It seems that TrainExtension, from which MonitorBasedSaveBest,
    is a new-style class, so I'd expect to be able to call super
    from this subclass.  I had a bit of trouble getting that to work
    so I'm invoking the superclass methods in an old-style way here.
    """
    def __init__(self, k, channel_name, save_path):
        MonitorBasedSaveBest.__init__(self, channel_name, save_path)
        self.k = k
        self.epoch = 0

    def on_monitor(self, model, dataset, algorithm):
        if self.epoch > self.k:
            return MonitorBasedSaveBest.on_monitor(self, model, dataset, algorithm)
        self.epoch += 1
