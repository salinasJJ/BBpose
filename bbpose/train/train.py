import json
import math
import os

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from ingestion.ingest import DataGenerator
from models import model as cnn
from preprocess.preprocess import DataPreprocessor
from train import callbacks
from train.losses import JointsMSE
from train.metrics import PercentageOfCorrectKeypoints
from utils import (
    force_update, 
    freeze_cfg, 
    get_frozen_params, 
    get_params,
    get_results_dir,
    is_file,
    reload_modules,
)


def run(restore=False):
    force_update({'switch':False})
    reload_modules(cnn, callbacks)

    params = get_params('Train')
    version = params['version']
    if restore:
        force_update({'switch':True})
        reload_modules(cnn, callbacks)

        params = get_frozen_params(
            'Train', 
            version=version,
        )

        results_dir = get_results_dir(params['dataset'])
    else:
        freeze_cfg(version=version)
        results_dir = get_results_dir(params['dataset'])
        
        skeleton = {}
        skeleton[f'v{version}'] = {
            'best': 0.0, 
            'epoch': 0, 
            'step': 0,
        }
        is_file(results_dir, 'trackers.json')
        with open(results_dir + 'trackers.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(
                skeleton, 
                ensure_ascii=False, 
                indent=4,
            ))
    
    tf.config.run_functions_eagerly(params['is_eager'])

    if params['strategy'] == 'default':
        strategy = tf.distribute.get_strategy()
    elif params['strategy'] == 'mirrored':
        strategy = tf.distribute.MirroredStrategy()
    elif params['strategy'] == 'tpu':
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=params['tpu_address']
        )
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        
    force_update({'num_replicas':strategy.num_replicas_in_sync})
    reload_modules(callbacks)
    
    if params['strategy'] != 'tpu':
        if params['mixed_precision']:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        steps_per_execution = params['steps_per_execution']
        assert params['track_every'] % steps_per_execution == 0, \
            "'track_every' must be a multiple of 'steps_per_execution'"
    else:
        steps_per_execution = None

    assert params['img_size'] == 4 * params['hm_size'], \
        "'hm_size' must be 4 times smaller than 'img_size'"

    generator = DataGenerator()
    preprocessor = DataPreprocessor()
    if params['use_records']:
        train_records, val_records = generator.load_records()
        train_tables, val_tables = preprocessor.read_records(
            train_records,
            val_records,
        )
    else:
        assert params['strategy'] != 'tpu', \
            "TPUStrategy only supports TFRecords as input"
        if os.path.isdir(params['img_dir']):
            train_tables, val_tables = generator.load_datasets()
        else:
            print(f"{params['img_dir']} does not exist")
    
    train_dataset, val_dataset = preprocessor.get_datasets(
        train_tables, 
        val_tables,
    )

    if params['schedule_per_step']:
        lr_callback = callbacks.lr_schedule_per_step()
        lr_object = callbacks.LRSchedulerPerStep
    else:
        lr_callback = callbacks.lr_schedule()
        lr_object = callbacks.LRScheduler
    
    callbacks_list = [
        lr_callback, 
        callbacks.savemodel(), 
        callbacks.csvlogger(restore),
    ]

    if params['use_records']:
        steps_per_epoch = params['train_size'] // params['batch_per_replica']
        validation_steps = int(math.ceil(
            params['val_size'] / params['batch_per_replica']
        ))
    else:
        steps_per_epoch = train_dataset.cardinality().numpy()
        validation_steps = val_dataset.cardinality().numpy()

    if restore:
        if params['use_cloud']:
            savemodel_dir = (
                params['gcs_results'].rstrip('/') + f'/{str(version)}'
            )
        else:
            savemodel_dir = results_dir + f'savemodel/{str(version)}'
            
        if tf.io.gfile.isdir(savemodel_dir):
            print('Restoring...')
            with open(results_dir + 'trackers.json', 'r') as f:
                trackers = json.load(f)

            callbacks_list[1].best = trackers[f"v{version}"]["best"]
            initial_epoch = int(trackers[f"v{version}"]["epoch"])
            if params['schedule_per_step']: 
                callbacks_list[0].step = int(trackers[f"v{version}"]["step"])

            with strategy.scope():
                model = tf.keras.models.load_model(
                    savemodel_dir, 
                    custom_objects=get_custom_objects(
                        params['schedule_per_step'],
                    ),
                )
                model = get_compiled_model(model, steps_per_execution)
        else:
            print('No SaveModel found, creating a new model from ' \
                  'scratch...')
            initial_epoch = 0
            with strategy.scope():
                model = cnn.get_model()  
                model = get_compiled_model(model, steps_per_execution)
    else:
        print('Creating a new model...')
        initial_epoch = 0
        with strategy.scope():
            model = cnn.get_model()
            model = get_compiled_model(model, steps_per_execution)

    force_update({'switch':False})

    model.fit(
        train_dataset, 
        validation_data=val_dataset,
        initial_epoch=initial_epoch,
        epochs=params['num_epochs'],
        callbacks=callbacks_list,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )
    return model

def get_compiled_model(model, steps_per_execution):
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(), 
        loss=JointsMSE(),
        metrics=[PercentageOfCorrectKeypoints()],
        steps_per_execution=steps_per_execution,
    )
    return model

def get_custom_objects(schedule_per_step):
    if schedule_per_step:
        lr_object = callbacks.LRSchedulerPerStep
    else:
        lr_object = callbacks.LRScheduler
        
    return {
        "JointsMSE":JointsMSE,
        "PercentageOfCorrectKeypoints":PercentageOfCorrectKeypoints,
        "LRScheduler":lr_object,
        "Checkpoint":callbacks.Checkpoint,
    }






