from configparser import ConfigParser, ExtendedInterpolation
from importlib import reload
import os
import shutil
import subprocess
import sys


BB_DIR = os.path.dirname(os.path.realpath(__file__)) + '/'
CFG = BB_DIR + 'configs/'
MAIN = 'bulat'
MPII = {
    'data':'https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/',
    'annos':'https://github.com/bearpaw/pytorch-pose/raw/master/',
}
DATASETS = {
    'mpii':{
        'data':MPII['data']+'mpii_human_pose_v1.tar.gz',
        'annotations':MPII['annos']+'data/mpii/mpii_annotations.json',
        'detections':MPII['annos']+'evaluation/data/detections_our_format.mat',
    },
}
SECTIONS = [
    'Ingestion', 
    'Preprocess', 
    'Models',
    'Train',
    'Test'
]
REQUIRED = [
    'version',
    'img_dir',
]
TYPES = [
    'string', 
    'integer', 
    'float', 
    'boolean', 
    'list',
]
DEFAULTS = {
    'newell':{
        'string':{
            'img_dir':'/PATH/TO/IMG/DIR',
            'dataset':'mpii',
            'gcs_data':'',
            'arch':'hourglass',
            'initializer':'glorot_normal',
            'strategy':'default',
            'tpu_address':'',
            'gcs_results':'',
        },
        'integer':{
            'version':0,
            'img_size':256,
            'hm_size':64,
            'num_stacks': 8,
            'batch_per_replica':8, 
            'num_replicas':1,
            'train_size':0,
            'val_size':0,
            'toy_samples':250,
            'examples_per_record':250,
            'interleave_num':-1,
            'interleave_block':1,
            'sigma':1,
            'num_filters':256,
            'num_epochs':100,
            'steps_per_execution':1,
            'track_every':32,
            'decay_steps':2000, 
        }, 
        'float':{
            'momentum':0.9,
            'epsilon':0.001,
            'dropout_rate':0.2,
            'threshold':0.5,
            'decay_factor':0.1,
            'learning_rate':0.00025,
            'scale':0.0,
            'decay_rate':0.96,
        },
        'boolean':{
            'use_records':False,
            'use_cloud':False,
            'switch':False,
            'download_images':False,
            'toy_set':False,
            'is_eager':False,
            'mixed_precision':False,
            'schedule_per_step':False,
        },
        'list':{
            'mean':[
                0.4624228775501251, 
                0.44416481256484985, 
                0.4025438725948334
            ],
            'decay_epochs':[60, 90],
        }       
    },
    'bulat':{
        'string':{
            'img_dir':'/PATH/TO/IMG/DIR',
            'dataset':'mpii',
            'gcs_records':'',
            'arch':'softgate',
            'initializer':'glorot_uniform',
            'strategy':'default',
            'tpu_address':'',
            'gcs_results':'',
        },
        'integer':{
            'version':0,
            'img_size':256,
            'hm_size':64,
            'num_stacks': 4,
            'batch_per_replica':24,
            'num_replicas':1,
            'train_size':0,
            'val_size':0,
            'toy_samples':250,
            'examples_per_record':250,
            'interleave_num':2,
            'interleave_block':1,
            'sigma':1,
            'num_filters':144,
            'num_epochs':200,
            'steps_per_execution':1,
            'track_every':32,
            'decay_steps':2000, 
        }, 
        'float':{
            'momentum':0.9,
            'epsilon':0.001,
            'dropout_rate':0.2,
            'threshold':0.5,
            'decay_factor':0.2,
            'learning_rate':0.00025,
            'scale':0.0,
            'decay_rate':0.96,
        },
        'boolean':{
            'use_records':False,
            'use_cloud':False,
            'switch':False,
            'download_images':False,
            'toy_set':False,
            'is_eager':False,
            'mixed_precision':False,
            'schedule_per_step':False,
        },
        'list':{
            'mean':[
                0.4624228775501251, 
                0.44416481256484985, 
                0.4025438725948334
            ],
            'decay_epochs':[75, 100, 150],
        }
    }
}
IGNORE = [
    'switch',
    'num_replicas',
    'train_size',
    'val_size',
]
DUPLICATES = {
    'DEFAULT':[],
    'Ingestion':[],
    'Preprocess':[],
    'Models':[],
    'Train':[],
    'Test':[
        'toy_set',
        'is_eager',
        'strategy',
        'tpu_address',
        'gcs_results',
        'mixed_precision',
        'threshold',
        'schedule_per_step',
    ],
}
FILTERS = {
    'strategy':[
        'default',
        'mirrored',
        'tpu',
    ],
    'dataset':[
        'mpii',
    ],
    'img_size':[128, 192, 256, 320, 384, 448, 512],
    'hm_size':[32, 48, 64, 80, 96, 112, 128],
    'num_stacks':[2, 4, 8],
    'arch':[
        'hourglass', 
        'softgate', 
    ],
    'initializer':[
        'glorot_normal', 
        'glorot_uniform', 
        'he_normal', 
        'he_uniform',
    ],
}


def get_params(*args, cfg=CFG+'config.cfg'):
    config = ConfigParser(
        interpolation=ExtendedInterpolation(),
    )
    config.read(cfg)
    params = {}
    for section in args:
        assert isinstance(section, str), f"'{section}' must be a string"
        if section.capitalize() in SECTIONS:
            for option in config.options(section):
                params[option] = eval(config.get(
                    section, 
                    option,
                ))
        else:
            print((
                f"Section '{section}' was not found in the configuration file."
            ))
            print(f"Available sections: {SECTIONS}")
    return params

def update_config(*args, cfg=CFG+'config.cfg'):
    config = ConfigParser(
        interpolation=ExtendedInterpolation(),
    )
    config.read(cfg)
    for arg in args:
        config.set(
            arg[0], 
            arg[1], 
            arg[2],
        )
    with open(cfg, 'w') as f:
        config.write(f)

def set_model_version(version):
    assert isinstance(version, int), f"'{version}' must be an integer"

    update_params({
        REQUIRED[0]:version,
    })

def set_media_directory(media_dir):
    assert isinstance(media_dir, str), f"'{media_dir}' must be a string"

    update_config((
        'DEFAULT', 
        REQUIRED[1], 
        '\'' + media_dir.rstrip('/') + '/\'',
    ))

def update_params(param_dict, cfg=CFG+'config.cfg'):
    assert isinstance(param_dict, dict), f"'{param_dict}' must be a dictionary"
    assert isinstance(cfg, str), f"'{cfg}' must be a string"

    force_update({'switch':False})

    config = ConfigParser(
        interpolation=ExtendedInterpolation()
    )
    config.read(cfg)

    params_dict = filter_params(param_dict)
    for param, value in param_dict.items():
        if param in config.defaults().keys():
            conditional_update(
                'DEFAULT',
                param,
                value,
                cfg=cfg,
            )
        elif config.has_option(SECTIONS[0], param):
            conditional_update(
                SECTIONS[0],
                param,
                value,
                cfg=cfg,
            )
        elif config.has_option(SECTIONS[1], param):
            conditional_update(
                SECTIONS[1],
                param,
                value,
                cfg=cfg,
            )
        elif config.has_option(SECTIONS[2], param):
            conditional_update(
                SECTIONS[2],
                param,
                value,
                cfg=cfg,
            )
        elif config.has_option(SECTIONS[3], param):
            conditional_update(
                SECTIONS[3],
                param,
                value,
                cfg=cfg,
            )
        elif config.has_option(SECTIONS[4], param):
            conditional_update(
                SECTIONS[4],
                param,
                value,
                cfg=cfg,
            )  

def filter_params(param_dict):
    param_list = get_param_list()

    for param, value in param_dict.copy().items():
        if param not in param_list:
            print(f"Unknown param: '{param}'")
            param_dict.pop(param)
            continue

        if not type_check(param, value):
            type_message(param, value)
            param_dict.pop(param)
            continue

        if param in IGNORE:
            print(f"'{param}' is not an adjustable parameter.\n")
            param_dict.pop(param)
        elif param == 'initializer':
            if value not in FILTERS[param]:
                print(
                    f"Recommended '{param}' values include: {FILTERS[param]}."
                )
                print((
                    "Please check with the tensorflow documentation first to " 
                    "determine if your requested initializer is supported. If " 
                    "not, you may receive an error at runtime."
                ))
                print((
                    'https://www.tensorflow.org/api_docs/python/tf/keras/'
                    'initializers\n'
                ))
        elif param == 'num_filters':
            if value % 8 != 0: 
                print((
                    f"Supported '{param}' values include only multiples of 8. "
                ))
                print(f"'{param}' was not updated.\n")
                param_dict.pop(param)
        elif param == 'scale':
            if value > 1.0:
                print((
                    "Warning: It is not recommended to scale the learning rate "
                    "with a value greater than one."
                ))
        elif param == 'strategy':
            generic_filter(param_dict, param)
        elif param == 'dataset':
            generic_filter(param_dict, param)
        elif param == 'img_size':
            generic_filter(param_dict, param)
        elif param == 'hm_size':
            generic_filter(param_dict, param)  
        elif param == 'num_stacks':
            generic_filter(param_dict, param)     
        elif param == 'arch':
            generic_filter(param_dict, param)
    return param_dict

def conditional_update(
        section, 
        param,
        value,
        cfg=CFG+'config.cfg',
    ):
    if param in DUPLICATES[section]:
        pass
    elif param in DEFAULTS[MAIN]['string']:
        update_config(
            (section, f'{param}', f"'{value}'"),
            cfg=cfg,
        )
    else:
        update_config(
            (section, f'{param}', f'{value}'),
            cfg=cfg,
        )

def get_param_list():
    param_list = []
    for t in TYPES:
        param_list += [k for k in DEFAULTS[MAIN][t]]
    return param_list

def type_check(
        param, 
        value,
    ):
    if isinstance(value, bool):
        if param in DEFAULTS[MAIN]['boolean']:
            return True
    elif isinstance(value, list):
        if param in DEFAULTS[MAIN]['list']:
            list_type = type(
                DEFAULTS[MAIN]['list'][param][0]
            )
            if all(isinstance(elem, list_type) for elem in value):
                return True        
    elif isinstance(value, int):
        if param in DEFAULTS[MAIN]['integer']:
            return True
    elif isinstance(value, float):
        if param in DEFAULTS[MAIN]['float']:
            return True
    elif isinstance(value, str):
        if param in DEFAULTS[MAIN]['string']:
            return True
    return False

def type_message(param, value):
    if isinstance(value, list):
        list_type = type(
            DEFAULTS[MAIN]['list'][param][0]
            ).__name__
        print((
            f"'{param}' must be of type 'list' containing elements of type "
            f"'{list_type}'"
        ))
        print(f"'{param}' was not updated.\n")
    else:
        param_type = get_type(param)
        print(f"'{param}' must be of type '{param_type}'")
        print(f"'{param}' was not updated.\n")
    
def generic_filter(param_dict, param):
    if param_dict[param] not in FILTERS[param]:
        print(f"Supported '{param}' values include: {FILTERS[param]}.")
        print(f"'{param}' was not updated.\n")
        param_dict.pop(param)  

def get_type(param):
    for k, v in DEFAULTS[MAIN].items():
        if param in v:
            return k

def force_update(param_dict):
    config = ConfigParser(
        interpolation=ExtendedInterpolation(),
    )
    config.read(CFG + 'config.cfg')

    for param in param_dict:
        if param in config.defaults().keys():
            if param in IGNORE:
                update_config((
                    'DEFAULT', 
                    f'{param}', 
                    f'{param_dict[param]}',
                ))

def view_config(version='current'):
    if version == 'current':
        params = get_params(*SECTIONS)
    elif isinstance(version, int):
        try:
            params = get_frozen_params(
                *SECTIONS,
                version=version,
            )
        except:
            return f"No config file for version '{version}' exists."
    else:
        return 'Invalid entry.'
    
    for ignore in IGNORE:
        params.pop(ignore)
    return params

def reset_default_params(defaults=MAIN):
    default_dict = {}
    ignore_dict = {}
    for k, v in DEFAULTS[defaults].items():
        for nested_k, nested_v in v.items():
            if nested_k not in IGNORE:
                default_dict[nested_k] = nested_v
            else:
                ignore_dict[nested_k] = nested_v
    update_params(default_dict)
    force_update(ignore_dict)

def freeze_cfg(version):
    shutil.copyfile(
        CFG + 'config.cfg', 
        CFG + f'freeze_v{version}.cfg',
    )

def get_frozen_params(*section, version=0):
    return get_params(
        *section, 
        cfg=CFG + f'freeze_v{version}.cfg',
    )

def update_frozen_params(param_dict, version=0):
    assert isinstance(param_dict, dict), f"'{param_dict}' must be a dictionary"
    assert isinstance(version, int), f"'{version}' must be an integer"

    update_params(
        param_dict,
        cfg=CFG + f'freeze_v{version}.cfg',
    )

def reload_modules(*args):
    for arg in args:
        reload(arg)

def is_file(
        path, 
        filename, 
        restore=False,
    ):
    if os.path.isdir(path):
        if os.path.isfile(path + filename):
            if restore == False:
                open(path + filename, 'w').close() 
            else:
                pass
        else:
            open(path + filename, 'w').close()
    else:
        os.mkdir(path)
        open(path + filename, 'w').close()

def is_dir(path):
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)
        
def get_results_dir(dataset_name):
    return get_module_dir('results/', dataset_name)

def get_data_dir(dataset_name):
    return get_module_dir('data/', dataset_name)

def get_module_dir(module, dataset_name):
    is_dir(BB_DIR + module)
    module_dir = os.path.abspath(os.path.join(
        BB_DIR, 
        module,
        dataset_name,
    ))
    is_dir(module_dir)
    return module_dir + '/'

def call_bash(command, message=None):
    try:
        output = subprocess.check_output(
            command, 
            shell=True, 
            stderr=subprocess.STDOUT,
        )
        if message is not None:
            print(message) 
        return output
    except subprocess.CalledProcessError as e:
        print(e.output)
        print("Exiting program...")
        sys.exit()




