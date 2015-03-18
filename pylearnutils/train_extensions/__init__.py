import joblib
import numpy as np
from pylearn2.train_extensions import TrainExtension
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest

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

class EpochBasedScheduledNoiseController(TrainExtension):
    """
    Increases or decreases the level of noise in the training data
    after each fixed number of epochs.  The current implementation
    is a hack just to move things along.

    Ultimately the knowledge of the noise-adding protocol (here,
    a fraction of the sentence to permute) should be decoupled
    from the schedule.  The Dataset chould, for instance, have a method
    that returns an object that knows how to change the kind or level
    of noise that is added to the training data.  This TrainExtension
    would then only need to get that object and call a method to make
    it change the noise-adding protocol.
    """
    def __init__(self, num_epochs):
        self.__dict__.update(locals())
        del self.self
        self.i = 1

    def on_monitor(self, model, dataset, algorithm):
        if self.i % self.num_epochs == 0:
            next_fraction = dataset.get_next_fraction()
            dataset.set_fraction(next_fraction)
        self.i += 1

class SaveWeightsWithinEpoch(object):
    """
    Saves the weights of model parameters at specified intervals.
    
    Include an instance of this class in the list passed to the
    update_callbacks argument of the SGD class.
    """
    def __init__(self, save_prefix, param_name, save_freq=1):
        self.__dict__.update(locals())
        self.i = 1

    def __call__(self, sgd):
        if self.i % self.save_freq == 0:
            save_path = "{0}_{1:04d}.npy".format(self.save_prefix, self.i)
            names = [x.name for x in sgd.params]
            i = names.index(self.param_name)
            np.save(save_path, sgd.params[i].get_value())
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


