import bisect
import json

import tensorflow as tf
from tensorflow.keras.callbacks import (
    CSVLogger, 
    LearningRateScheduler, 
    ModelCheckpoint,
)
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from utils import get_frozen_params, get_params, get_results_dir, is_file


PARAMS = get_params('Train')
VERSION = PARAMS['version']
if PARAMS['switch']:
    PARAMS = get_frozen_params(
        'Train',
        version=VERSION,
    )


def savemodel():
    if PARAMS['use_cloud']:
        savemodel_dir = (
            PARAMS['gcs_results'].rstrip('/') + f'/{str(VERSION)}'
        )
    else:
        savemodel_dir = (
            get_results_dir(PARAMS['dataset']) 
            + f'savemodel/{str(VERSION)}'
        )    
    return Checkpoint(savemodel_dir)

def csvlogger(restore=False):
    logs_dir = get_results_dir(PARAMS['dataset']) + 'logs/'
    is_file(
        logs_dir, 
        filename=f'logs_v{VERSION}.csv', 
        restore=restore,
    )
    return CSVLogger(
        filename=logs_dir + f'logs_v{VERSION}.csv',
        append=True,
    )

def lr_schedule_per_step():
    return LRSchedulerPerStep(schedule_per_step)

def lr_schedule():
    return LRScheduler(schedule)

def schedule_per_step(step, learning_rate):
    if PARAMS['num_replicas'] > 1:
        if PARAMS['scale'] == 0.:
            lr_per_replica = PARAMS['learning_rate']
        else:
            scale = PARAMS['num_replicas'] * PARAMS['scale']
            lr_per_replica = PARAMS['learning_rate'] * scale
    else:
        lr_per_replica = PARAMS['learning_rate']
    return ExponentialDecay(
            initial_learning_rate=lr_per_replica,
            decay_steps=PARAMS['decay_steps'],
            decay_rate=PARAMS['decay_rate'],
            staircase=True,
            )(step)

def schedule(epoch, learning_rate):
    if PARAMS['num_replicas'] > 1:
        if PARAMS['scale'] == 0.:
            lr_per_replica = PARAMS['learning_rate']
        else:
            scale = PARAMS['num_replicas'] * PARAMS['scale']
            lr_per_replica = PARAMS['learning_rate'] * scale
    else:
        lr_per_replica = PARAMS['learning_rate']

    decay_epochs = sorted(PARAMS['decay_epochs'])
    idx = bisect.bisect_right(decay_epochs, epoch)
    return lr_per_replica * PARAMS['decay_factor']**idx


class LRSchedulerPerStep(LearningRateScheduler):
    def __init__(
            self, 
            schedule, 
            verbose=0,
        ):
        super(LRSchedulerPerStep, self).__init__(schedule, verbose)
        
        params = get_params('Train')
        self.version = params['version']
        if params['switch']:
            self.params = get_frozen_params(
                'Train', 
                version=self.version,
            )
        else:
            self.params = params 
        self.track_every = self.params['track_every']
        self.steps_per_execution = self.params['steps_per_execution']
        self.strategy = self.params['strategy']

        self.step = 1
        results_dir = get_results_dir(self.params['dataset'])
        self.trackers_file = results_dir + 'trackers.json'

    def on_epoch_begin(self, epoch, logs=None):
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        super(LRSchedulerPerStep, self).on_epoch_end(epoch, logs)

        with open(self.trackers_file, 'r') as f:
            trackers = json.load(f)
        trackers[f"v{self.version}"]["epoch"] = epoch + 1
        with open(self.trackers_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(
                trackers, 
                ensure_ascii=False, 
                indent=4,
            ))

    def on_train_batch_begin(self, batch, logs=None):
        super(LRSchedulerPerStep, self).on_epoch_begin(self.step, logs)
        
    def on_train_batch_end(self, batch, logs=None):
        if (self.step - 1) % self.track_every == 0:
            with open(self.trackers_file, 'r') as f:
                trackers = json.load(f)
            trackers[f"v{self.version}"]["step"] = self.step
            with open(self.trackers_file, 'w', encoding='utf-8') as f:
                f.write(json.dumps(
                    trackers, 
                    ensure_ascii=False, 
                    indent=4,
                ))
        if self.strategy != 'tpu':
            self.step += self.steps_per_execution
        else:
            self.step += 1

class LRScheduler(LearningRateScheduler):
    def __init__(self, scheduler, verbose=0):
        super(LRScheduler, self).__init__(scheduler, verbose)

        params = get_params('Train')
        self.version = params['version']
        if params['switch']:
            self.params = get_frozen_params(
                'Train', 
                version=self.version,
            )
        else:
            self.params = params 

        results_dir = get_results_dir(self.params['dataset'])
        self.trackers_file = results_dir + 'trackers.json'

    def on_epoch_end(self, epoch, logs=None):
        super(LRScheduler, self).on_epoch_end(epoch, logs)
        
        with open(self.trackers_file, 'r') as f:
            trackers = json.load(f)
        trackers[f"v{self.version}"]["epoch"] = epoch + 1
        with open(self.trackers_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(
                trackers, 
                ensure_ascii=False, 
                indent=4,
            ))

class Checkpoint(ModelCheckpoint):
    def __init__(
            self, 
            filepath,
            monitor='val_pck',
            save_best_only=True,
            mode='max',
            save_weights_only=False,
            verbose=0,
        ):
        super(Checkpoint, self).__init__(
            filepath, 
            monitor=monitor, 
            save_best_only=save_best_only, 
            mode=mode, 
            save_weights_only=save_weights_only,
            verbose=verbose,
        )
        params = get_params('Train')
        self.version = params['version']
        if params['switch']:
            self.params = get_frozen_params(
                'Train', 
                version=self.version,
            )
        else:
            self.params = params 
        
        self.best_pck = -1.
        results_dir = get_results_dir(self.params['dataset'])
        self.trackers_file = results_dir + 'trackers.json'

    def on_epoch_end(self, epoch, logs=None):
        super(Checkpoint, self).on_epoch_end(epoch, logs)
        if self.best_pck < self.best:
            self.best_pck = self.best
            with open(self.trackers_file, 'r') as f:
                trackers = json.load(f)
            trackers[f"v{self.version}"]["best"] = self.best_pck
            with open(self.trackers_file, 'w', encoding='utf-8') as f:
                f.write(json.dumps(
                    trackers, 
                    ensure_ascii=False, 
                    indent=4,
                ))





