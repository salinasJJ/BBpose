import math
import os
import shutil

import pandas as pd
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

from utils import (
    DATASETS, 
    call_bash, 
    force_update,
    get_data_dir,
    get_frozen_params, 
    get_params, 
    is_dir,
)


ANNOTATIONS = [
    'img_paths', 
    'people_index', 
    'joint_self', 
    'objpos', 
    'scale_provided',
]
ID_COLUMNS = [
    'file_name', 
    'person_N',
]
JNT_COLUMNS = [
    'r_ankle_X', 'r_ankle_Y', 'r_knee_X', 'r_knee_Y', 'r_hip_X', 'r_hip_Y',
    'l_hip_X', 'l_hip_Y', 'l_knee_X', 'l_knee_Y', 'l_ankle_X', 'l_ankle_Y',
    'pelvis_X', 'pelvis_Y', 'thorax_X', 'thorax_Y', 'neck_X', 'neck_Y',
    'head_X', 'head_Y', 'r_wrist_X', 'r_wrist_Y', 'r_elbow_X', 'r_elbow_Y',
    'r_shoulder_X', 'r_shoulder_Y', 'l_shoulder_X', 'l_shoulder_Y', 'l_elbow_X',
    'l_elbow_Y', 'l_wrist_X', 'l_wrist_Y',
]
VIS_COLUMNS = [ 
    'r_ankle_vis', 'r_knee_vis', 'r_hip_vis', 'l_hip_vis', 'l_knee_vis', 
    'l_ankle_vis', 'pelvis_vis', 'thorax_vis', 'neck_vis', 'head_vis',
    'r_wrist_vis', 'r_elbow_vis', 'r_shoulder_vis', 'l_shoulder_vis',
    'l_elbow_vis', 'l_wrist_vis',
]
CS_COLUMNS = [
    'center_X', 
    'center_Y', 
    'scale',
]
EXCLUDE = [
    ('012545809.jpg', 2),
]
SEEN_EVERY = 100


class DataGenerator():
    def __init__(self, setup=False):
        params = get_params('Ingestion')
        if params['switch']:
            self.params = get_frozen_params(
                'Ingestion', 
                version=params['version'],
            )
        else:
            self.params = params
        self.dataset_dir = get_data_dir(self.params['dataset'])
        if self.params['toy_set']:
            self.data_dir = self.dataset_dir + 'data/toy/'
            self.tfds_dir = self.dataset_dir + 'tfds/toy/'
            self.records_dir = self.dataset_dir + 'records/toy/'
        else:
            self.data_dir = self.dataset_dir + 'data/full/'
            self.tfds_dir = self.dataset_dir + 'tfds/full/'
            self.records_dir = self.dataset_dir + 'records/full/'
        
        self.script_dir = (
            os.path.dirname(os.path.realpath(__file__)) + '/scripts/'
        )
        if setup:
            self.get_data()

    def get_data(self):
        print('Downloading data files...')
        data_url = DATASETS[self.params['dataset']]['data']
        annotations_url = DATASETS[self.params['dataset']]['annotations']
        detections_url = DATASETS[self.params['dataset']]['detections']

        status = call_bash(
            command = (
                f"'{self.script_dir + 'data.sh'}' "
                f"-D '{self.data_dir}' "
                f"-T '{self.tfds_dir}' "
                f"-R '{self.records_dir}' "
                f"-j {data_url} " 
                f"-a {annotations_url} "
                f"-m {detections_url} "
                f"-I '{self.params['img_dir']}' "
                f"-r {self.params['use_records']} "
                f"-d {self.params['dataset']} "
                f"-i {self.params['download_images']} "
            ),
            message='Data files downloaded.\n',
        )
        if status:
            for s in status.decode('utf-8').split("\n"):
                print(s)

    def generate(self):
        print("Generating 'train' csv file...")
        train_df = self._json_to_pandas('train')
        self._pandas_to_csv(train_df, split='train')
        print("Generating 'val' csv file...")
        val_df = self._json_to_pandas('val')
        self._pandas_to_csv(val_df, split='val')

        train_dataset = self._csv_to_tfds('train')
        val_dataset = self._csv_to_tfds('val')
        if self.params['use_records']:
            print(f'Generating Tensorflow Records...')
            self._create_records(train_dataset, 'train')
            self._create_records(val_dataset, 'val')

            shutil.rmtree(self.dataset_dir + 'tfds/')
        else:
            print(f'Generating Tensorflow Datasets...')
            self.save_dataset(train_dataset, 'train')
            self.save_dataset(val_dataset, 'val')

    def _json_to_pandas(self, split):
        return self._get_dataframe(
            self._get_annos(split),
        )
        
    def _get_annos(self, split):
        annos = pd.read_json(self.data_dir + 'annotations.json')
        if split == 'train':
            annos = annos[annos.isValidation == 0]
        elif split == 'val':
            annos = annos[annos.isValidation == 1]  
        annos = annos.drop(
            'isValidation', 
            axis=1
            ).reset_index(drop=True)
        return annos.loc[:, ANNOTATIONS]

    def _get_dataframe(self, dataframe):
        df = pd.DataFrame(
            columns=ID_COLUMNS + JNT_COLUMNS + VIS_COLUMNS + CS_COLUMNS
        )
        total = dataframe.shape[0]
        for idx, annos in dataframe.iterrows():
            if annos[0] == EXCLUDE[0][0] and annos[1] == EXCLUDE[0][1]:
                continue

            ids = [0 for _ in range(2)]
            jnts = []
            vis = []
            cs = [0 for _ in range(3)]
            for a in annos[2]:
                jnts.append(a[0])
                jnts.append(a[1])
                vis.append(a[2])
            
            ids = pd.DataFrame([ids], columns=ID_COLUMNS)
            jnts = pd.DataFrame([jnts], columns=JNT_COLUMNS)
            vis = pd.DataFrame([vis], columns=VIS_COLUMNS)
            cs = pd.DataFrame([cs], columns=CS_COLUMNS)

            ids.file_name = annos[0]
            ids.person_N = annos[1]

            if annos[3][0] != -1:
                annos[3][1] = annos[3][1] + 15 * annos[4]
                annos[4] = annos[4] * 1.25
            cs.center_X = annos[3][0]
            cs.center_Y = annos[3][1]
            cs.scale = annos[4]
            
            temp = pd.concat(
                [ids, jnts, vis, cs], 
                axis=1,
            )
            df = pd.concat([df, temp])

            if idx % SEEN_EVERY == 0:
                print(f'{idx}/{total}')
            if self.params['toy_set']:
                if idx == self.params['toy_samples'] - 1:
                    break
        return df.reset_index(drop=True)
    
    def _pandas_to_csv(self, dataframe, split):
        print(f"Saving '{split}' csv file to: {self.data_dir}\n")
        dataframe.to_csv(
            self.data_dir + split + '.csv', 
            index=False,
        )
        
    def _csv_to_tfds(self, split):    
        return self._map_dataset(
            self._get_dataset(split)
        )
 
    def _get_dataset(self, split):
        return tf.data.experimental.CsvDataset(
            self.data_dir + split + '.csv',
            DataGenerator._get_component_dtypes(),
            header=True
        )
    
    @staticmethod
    def _get_component_dtypes():
        field_dtypes = [tf.string for _ in range(2)]
        return field_dtypes + [tf.float32 for _ in range(51)]  

    def _map_dataset(self, dataset):
        return dataset.map(
            DataGenerator._get_components, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=True,
        )

    @staticmethod
    def _get_components(*elements):
        file_name = elements[0]
        person_N = elements[1]
        identifier = tf.strings.join(
            inputs=[
                file_name, 
                person_N,
            ], 
            separator='-',
        )
        joints = tf.reshape(
            tf.stack(elements[2:34]), 
            shape=[16,2],
        )
        weights = tf.stack(elements[34:50])
        center = tf.stack(elements[50:52])
        scale = elements[52]
        return identifier, joints, weights, center, scale

    def _create_records(self, dataset, split):
        dataset_size = self._get_dataset_size(split)
        total_records = math.ceil(
            dataset_size / self.params['examples_per_record']
        )
        for record_num in range(total_records):
            skip = record_num * self.params['examples_per_record']
            take = self.params['examples_per_record']

            self._write_to_record(
                dataset.skip(skip).take(take),
                split,
                record_num,
            )
            print(f"{record_num}/{total_records - 1}")
        print(f"Saving '{split}' records to: {self.records_dir}")
    
    def _write_to_record(
            self, 
            dataset, 
            split,
            record_num,
        ):
        is_dir(self.records_dir + split)
        record = self.records_dir + split + f'/{split}_{record_num}.tfrecord'
        with tf.io.TFRecordWriter(record) as writer:
            for identifier, joints, weights, center, scale in dataset:
                filename = (
                    identifier.numpy().decode().split('.jpg')[0] + '.jpg'
                )
                path = self.params['img_dir'] + filename
                raw_image =  open(path, 'rb').read()

                example = self._serialize_example(
                    raw_image,
                    joints,
                    weights,
                    center,
                    scale,
                )
                writer.write(example)

    def _serialize_example(
            self, 
            raw_image, 
            joints,
            weights,
            center,
            scale,
        ):
        feature = {
            'raw_image':DataGenerator._bytes_feature(raw_image),
            'joints':DataGenerator._bytes_feature(
                tf.io.serialize_tensor(joints),
            ),
            'weights':DataGenerator._bytes_feature(
                tf.io.serialize_tensor(weights),
            ),
            'center':DataGenerator._bytes_feature(
                tf.io.serialize_tensor(center),
            ),
            'scale':DataGenerator._float_feature(scale),
        }
        example = tf.train.Example(
            features=tf.train.Features(feature=feature)
        )
        return example.SerializeToString()
    
    @staticmethod
    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[value]),
        )

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(
            float_list=tf.train.FloatList(value=[value]),
        )
    
    def _get_dataset_size(self, split):
        csv_file = self.data_dir + split + '.csv'
        dataset_size = call_bash(
                command = f"wc -l < {csv_file}",
            )
        dataset_size = int(dataset_size.decode('utf-8').strip("\n")) - 1
        force_update(
            {f"{split}_size":dataset_size},
        )
        return dataset_size

    def save_dataset(self, dataset, split):
        print(f"Saving '{split}' dataset to: {self.tfds_dir}")
        if os.path.isdir(self.tfds_dir + split):
            shutil.rmtree(self.tfds_dir + split)
        
        tf.data.experimental.save(
            dataset, 
            self.tfds_dir + split,
        )
        
    def load_datasets(self):
        train_dataset = self._load_dataset('train')
        val_dataset = self._load_dataset('val')
        return train_dataset, val_dataset

    def load_test_dataset(self):
        return self._load_dataset('val')

    def _load_dataset(self, split):
        return tf.data.experimental.load(
            path=self.tfds_dir + split,
            element_spec=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(16, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(16,), dtype=tf.float32),
                tf.TensorSpec(shape=(2,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
            ),
        )

    def load_records(self):
        train_records = self._load_records('train')
        val_records = self._load_records('val')
        return train_records, val_records
    
    def load_test_records(self):
        return self._load_records('test')
    
    def _load_records(self, split):
        records = self._get_record_names(split)
        if split == 'test':
            records = sorted(
                records, 
                key=lambda r: int(r.split('val_')[1].split('.tf')[0]),
            )
            return tf.data.TFRecordDataset(records)
        else:
            ds = tf.data.Dataset.from_tensor_slices(records)
            ds = ds.shuffle(
                buffer_size=tf.cast(
                    tf.shape(records)[0], 
                    tf.int64,
                ),
            )
            ds = ds.repeat()
            return ds.interleave(
                tf.data.TFRecordDataset,
                cycle_length=self.params['interleave_num'],
                block_length=self.params['interleave_block'],
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                deterministic=False,
            )
    
    def _get_record_names(self, split):
        if split == 'test':
            split = 'val'
        if self.params['use_cloud']:
            records = tf.io.gfile.glob((
                self.params['gcs_data'].rstrip('/') 
                + '/'
                + split
                + f'/{split}_*.tfrecord'
            ))
        else:
            records = tf.io.gfile.glob(
                self.records_dir + split + f'/{split}*.tfrecord'
            )
        return records











