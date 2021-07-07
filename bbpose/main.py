import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from ingestion.ingest import DataGenerator
from test import test
from train import train
from utils import (
    force_update,
    reload_modules,
    reset_default_params,
    set_media_directory,
    set_model_version, 
    update_params,
)


def str_to_bool(v):
    if isinstance(v, bool):
       return v
    elif v.lower() in ['true', 't', '1']:
        return True
    elif v.lower() in ['false', 'f', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(
        description="Human pose estimation model."
    )
    subparsers = parser.add_subparsers(
        dest='subparser_name', 
        help='sub-command help',
    )

    update_parser = subparsers.add_parser(
        'update', 
        help="updates the model parameters",
    )
    update_parser.add_argument(
        '--version', type=int, required=True,
        help="model version number",
    )
    update_parser.add_argument(
        '--img_dir', type=str, default='/PATH/TO/IMG/DIR', 
        help="directory where image files can be found",
    )
    update_parser.add_argument(
        '--dataset', type=str, default='mpii',
        help="currently, only 'mpii' is supported",
    )
    update_parser.add_argument(
        '--use_records', type=str_to_bool, default=False,
        help="whether to generate and use tfrecords or not",
    )
    update_parser.add_argument(
        '--use_cloud', type=str_to_bool, default=False,
        help="whether to retrieve data from a remote GCS location or not",
    )
    update_parser.add_argument(
        '--img_size', type=int, default=256, 
        help="one of 128, 192, 256, 320, 384, 448, or 512 (spatial length)",
    )
    update_parser.add_argument(
        '--hm_size', type=int, default=64,
        help="one of 32, 48, 64, 80, 96, 112, or 128 (spatial length)",
    )
    update_parser.add_argument(
        '--num_stacks', type=int, default=4, 
        help="one of 2, 4, or 8 (number of hourglass stacks)",
    )
    update_parser.add_argument(
        '--batch_per_replica', type=int, default=24,
        help="number of samples passed to each replica",
    )
    update_parser.add_argument(
        '--download_images', type=str_to_bool, default=False,
        help="whether to download image dataset or not",
    )
    update_parser.add_argument(
        '--toy_set', type=str_to_bool, default=False,
        help="whether to use a smaller dataset to experiment with or not",
    )
    update_parser.add_argument(
        '--toy_samples', type=int, default=250,
        help="number of samples per toy dataset",
    )
    update_parser.add_argument(
        '--examples_per_record', type=int, default=250,
        help="number of examples per tfrecord file",
    )
    update_parser.add_argument(
        '--interleave_num', type=int, default=2,
        help="number of records to simultaneously interleave",
    )
    update_parser.add_argument(
        '--interleave_block', type=int, default=1,
        help="number of consecutive elements from each record",
    )
    update_parser.add_argument(
        '--gcs_data', type=str, default='',
        help="remote GCS location where tfrecords may be found",
    )
    update_parser.add_argument(
        '--mean', type=float, nargs=3,
        default=[0.4624228775501251, 0.44416481256484985, 0.4025438725948334],
        help="mean values calculated (from dataset) for each channel (rgb)",
    )  
    update_parser.add_argument(
        '--sigma', type=int, default=1, 
        help="std used for gaussian kernel",
    )
    update_parser.add_argument(
        '--arch', type=str, default='softgate', 
        help="one of 'softgate' or 'hourglass' (network architectures)")
    update_parser.add_argument(
        '--num_filters', type=int, default=144,
        help="any multiple of 8 (number of channels used at each layer)",
    )
    update_parser.add_argument(
        '--initializer', type=str, default='glorot_uniform',
        help=(
            "one of 'glorot_normal', 'glorot_uniform', 'he_normal', or "
            "'he_uniform' (used in each layer)"
        ),
    )
    update_parser.add_argument(
        '--momentum', type=float, default=0.9,
        help="momentum for moving average used in each batch norm layer",
    )
    update_parser.add_argument(
        '--epsilon', type=float, default=0.001,
        help="value added to batch norm's variance to avoid division by zero",
    )
    update_parser.add_argument(
        '--dropout_rate', type=float, default=0.2,
        help="dropout performed after each hourglass",
    )
    update_parser.add_argument(
        '--is_eager', type=str_to_bool, default=False,
        help="whether to run tf.functions in eager mode or not",
    )
    update_parser.add_argument(
        '--strategy', type=str, default='default',
        help="one of 'default', 'mirrored', or 'tpu' (distributed training)",
    )
    update_parser.add_argument(
        '--tpu_address', type=str, default='',
        help="remote location of TPU device(s)"
    )
    update_parser.add_argument(
        '--gcs_results', type=str, default='',
        help="remote GCS location where savedmodels may be found",
    )
    update_parser.add_argument(
        '--mixed_precision', type=str_to_bool, default=False,
        help="whether to use both 16 and 32 bit floating-point types or not",
    )
    update_parser.add_argument(
        '--num_epochs', type=int, default=200,
        help="number of epochs to train for",
    )
    update_parser.add_argument(
        '--steps_per_execution', type=int, default=1,
        help="number of batches to run through each tf.function",
    )
    update_parser.add_argument(
        '--track_every', type=int, default=32,
        help="how often to save current iteration/step for use with LR schedule",
    )
    update_parser.add_argument(
        '--threshold', type=float, default=0.5,
        help="pck threshold",
    )
    update_parser.add_argument(
        '--decay_epochs', type=int, nargs='*', default=[75, 100, 150], 
        help="epoch at which to apply a decay factor to",
    )
    update_parser.add_argument(
        '--decay_factor', type=float, default=0.2,
        help="factor to decay the learning rate by",
    )
    update_parser.add_argument(
        '--learning_rate', type=float, default=0.00025,
        help="initial learning rate",
    )
    update_parser.add_argument(
        '--scale', type=float, default=0.0,
        help="learning rate scaler used in distributed training",
    )
    update_parser.add_argument(
        '--schedule_per_step', type=str_to_bool, default=False,
        help="whether to use a per step schedule (exponential decay) or not",
    )
    update_parser.add_argument(
        '--decay_rate', type=float, default=0.96,
        help="decay rate used in exponential schedule",
    )
    update_parser.add_argument(
        '--decay_steps', type=int, default=2000,
        help="decay frequency used in exponential schedule",
    )

    force_parser = subparsers.add_parser(
        'force',
        help="updates hidden params",
    )
    force_parser.add_argument(
        '--train_size', type=int, default=0,
        help="number of elements in the train dataset",
    )
    force_parser.add_argument(
        '--val_size', type=int, default=0,
        help="number of elemenets in the validation dataset",
    )

    reset_parser = subparsers.add_parser(
        'reset',
        help="resets params to default values",
    )
    reset_parser.add_argument(
        '--defaults', type=str, default='bulat',
        help="one of 'bulat' or 'newell' (set of default params)",
    )

    ingest_parser = subparsers.add_parser(
        'ingest', 
        help="retrieves data and generates datasets from scratch",
    )
    ingest_parser.add_argument(
        '--setup', action='store_true',
        help="whether to run setup script or not",
    )
    ingest_parser.add_argument(
        '--generate', action='store_true',
        help="whether to generate the datasets or not"
    )

    train_parser = subparsers.add_parser(
        'train',
        help="trains the model",
    )
    train_parser.add_argument(
        '--restore', action='store_true',
        help="whether to restore a saved model or not",
    )

    test_parser = subparsers.add_parser(
        'test',
        help="evaluates a trained model",
    )

    args = vars(parser.parse_args())

    if args['subparser_name'] == 'reset':
        reset_default_params(args['defaults'])
        print('Params reset to default values.')
    
    elif args['subparser_name'] == 'force':
        args.pop('subparser_name')
        force_update(args)

    elif args['subparser_name'] == 'update':
        args.pop('subparser_name')
        set_model_version(args.pop('version'))
        set_media_directory(args.pop('img_dir'))        
        update_params(args)
        print('Update complete.')

    elif args['subparser_name'] == 'ingest':
        if args['setup']:
            generator = DataGenerator(setup=True)
        else:
            generator = DataGenerator()
        if args['generate']:
            generator.generate()
        else:
            pass

    elif args['subparser_name'] == 'train':
        reload_modules(train)
        train.run(restore=args['restore'])

    elif args['subparser_name'] == 'test':
        reload_modules(test)
        test.run()
    
if __name__ == '__main__':
    main()






















