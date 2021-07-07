import math
import os

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from ingestion.ingest import DataGenerator
from preprocess.preprocess import DataPreprocessor
from test.metrics import PCKh
from train.train import get_custom_objects
from utils import (
    force_update, 
    get_frozen_params, 
    get_params,
    get_results_dir,
)


SEEN_EVERY = 20


def run():
    force_update({'switch':True})

    version=get_params('Test')['version']
    params = get_frozen_params(
        'Test', 
        version=version,
    )

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

    if params['mixed_precision']:
        if params['strategy'] != 'tpu':
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

    assert params['img_size'] == 4 * params['hm_size'], \
        "'hm_size' must be 4 times smaller than 'img_size'"

    generator = DataGenerator()
    preprocessor = DataPreprocessor()
    if params['use_records']:
        test_records = generator.load_test_records()
        test_tables = preprocessor.read_test_records(test_records)
    else:
        assert params['strategy'] != 'tpu', \
            "TPUStrategy only supports TFRecords as input"
        if os.path.isdir(params['img_dir']):    
            test_tables = generator.load_test_dataset()
        else:
            print(f"{params['img_dir']} does not exist")
    
    test_dataset = preprocessor.get_test_dataset(test_tables)

    if params['use_records']:
        test_steps = int(math.ceil(
            params['val_size'] / params['batch_per_replica']
        ))
    else:
        test_steps = test_dataset.cardinality().numpy()

    if params['use_cloud']:
        savemodel_dir = (
            params['gcs_results'].rstrip('/') + f'/{str(version)}'
        )
    else:
        savemodel_dir = (
            get_results_dir(params['dataset']) 
            + f'savemodel/{str(version)}'
        )
        
    if tf.io.gfile.isdir(savemodel_dir):
        with strategy.scope():
            print('Loading model weights...')
            model = tf.keras.models.load_model(
                savemodel_dir, 
                custom_objects=get_custom_objects(
                    params['schedule_per_step'],
                ),
            )
        pckh = PCKh()

        print("Evaluating...")
        for enum, elements in test_dataset.enumerate():
            images, center, scale = elements
            if enum % SEEN_EVERY == 0:
                print(f'{enum.numpy()} / {test_steps}')
            preds = model.predict(images)
            preds = preds[:, -1, :, :, :]

            if enum == 0:
                predictions = pckh.get_final_predictions(
                    preds, 
                    center, 
                    scale,
                )
            else:
                temp = pckh.get_final_predictions(
                    preds, 
                    center, 
                    scale,
                )
                predictions = tf.concat([predictions, temp], axis=0)

        results = pckh.get_results(predictions)
        print('\nResults...')
        for k, v in results.items():
            print(f"{k}:{v}")
        pckh.save_results(results)

        force_update({'switch':False,})
    else:
        force_update({'switch':False,})
        print('No SaveModel was found.')



    










