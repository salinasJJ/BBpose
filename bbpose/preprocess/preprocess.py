import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import tensorflow_addons as tfa

from preprocess.utils import coord_transform
from utils import get_frozen_params, get_params, update_config


NUM_JOINTS = 16
PAIRED_JOINTS = [
    [0,5], [1,4], [2,3], [10,15], [11,14], [12,13]
]
SINGLE_JOINTS = [6, 7, 8, 9]
PI = 3.14159265


class DataPreprocessor():
    def __init__(self):
        params = get_params('Preprocess')
        if params['switch']:
            self.params = get_frozen_params(
                'Preprocess', 
                version=params['version'],
            )
        else:
            self.params = params

        self.batch_size = tf.cast(
            self.params['batch_per_replica'] * self.params['num_replicas'],
            tf.int64,
        )
        self.random_generator = tf.random.Generator.from_seed(1)

    def read_records(self, train_records, val_records):
        train_record_dataset = self._map_records(train_records)
        val_record_dataset = self._map_records(val_records)
        return train_record_dataset, val_record_dataset

    def read_test_records(self, test_records):
        return self._map_records(test_records)

    def _map_records(self, dataset):
        ds = dataset.map(
            self._parse_example,
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        ds = ds.map(
            self._parse_features,
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        return ds

    def _parse_example(self, example):
        feature_description = {
            'raw_image':tf.io.FixedLenFeature([], tf.string),
            'joints':tf.io.FixedLenFeature([], tf.string),
            'weights':tf.io.FixedLenFeature([], tf.string),
            'center':tf.io.FixedLenFeature([], tf.string),
            'scale':tf.io.FixedLenFeature([], tf.float32)
        }
        features = tf.io.parse_single_example(example, feature_description)
        return features
  
    def _parse_features(self, features):
        raw_image = features['raw_image']
        joints = tf.io.parse_tensor(features['joints'], tf.float32)
        weights = tf.io.parse_tensor(features['weights'], tf.float32)
        center = tf.io.parse_tensor(features['center'], tf.float32)
        scale = tf.cast(features['scale'], tf.float32)
        return raw_image, joints, weights, center, scale

    def get_datasets(self, train_table, val_table):
        train_dataset = self._train_process(train_table)
        validation_dataset = self._validation_process(val_table)
        return train_dataset, validation_dataset
    
    def get_test_dataset(self, test_table):
        return self._test_process(test_table)

    def _train_process(self, dataset):
        if self.params['use_records']:
            buffer_size = self.params['train_size']
        else:
            buffer_size = dataset.cardinality()

        ds = dataset.shuffle(
            tf.cast(buffer_size + 1, tf.int64)
        )
        ds = ds.map(
            self._parsing_function, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        ds = ds.map(
            self._augmentation_function, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        ds = ds.map(
            self._generating_function, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        ds = ds.map(
            lambda im, hm, wt, c, s: (im, hm, wt),
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        ds = ds.map(
            lambda im, hm, wt: (Rescaling(scale=1./255)(im), hm, wt),
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        ds = ds.map(
            lambda im, hm, wt: (im - self.params['mean'], hm, wt),
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        ds = ds.map(
            self._expansion_function, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        ds = ds.batch(
            self.batch_size, 
            drop_remainder=False if self.params['use_records'] else True,
        )
        ds = ds.prefetch(1)
        return ds

    def _validation_process(self, dataset):
        if self.params['use_records']:
            buffer_size = self.params['val_size']
        else:
            buffer_size = dataset.cardinality()

        ds = dataset.shuffle(
            tf.cast(buffer_size + 1, tf.int64)
        )
        ds = ds.map(
            self._parsing_function, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        ds = ds.map(
            lambda im, j, wt, c, s: (im, j, wt, c, s, tf.zeros([])),
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )        
        ds = ds.map(
            self._generating_function, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        ds = ds.map(
            lambda im, hm, wt, c, s: (im, hm, wt),
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        ds = ds.map(
            lambda im, hm, wt: (Rescaling(scale=1./255)(im), hm, wt),
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        ds = ds.map(
            lambda im, hm, wt: (im - self.params['mean'], hm, wt),
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        ds = ds.map(
            self._expansion_function, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=False,
        )
        ds = ds.batch(
            self.batch_size,
            drop_remainder=False if self.params['use_records'] else True,
        )
        ds = ds.prefetch(1)
        return ds

    def _test_process(self, dataset):
        ds = dataset.map(
            self._parsing_function, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=True,
        )
        ds = ds.map(
            lambda im, j, wt, c, s: (im, j, wt, c, s, tf.zeros([])),
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=True,
        )
        ds = ds.map(
            self._generating_function, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=True,
        )
        ds = ds.map(
            lambda im, hm, wt, c, s: (im, c, s),
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=True,
        )
        ds = ds.map(
            lambda im, c, s: (Rescaling(scale=1./255)(im), c, s),
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=True,
        )
        ds = ds.map(
            lambda im, c, s: (im - self.params['mean'], c, s),
            num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=True,
        )
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(1)
        return ds

    @tf.function
    def _parsing_function(
            self, 
            source,
            joints, 
            weights, 
            center, 
            scale,
        ):
        image = self._get_image(source)
        return image, joints, weights, center, scale

    def _get_image(self, source):
        if self.params['use_records']:
            image = source
        else:
            source = tf.strings.substr(
                source, 
                pos=0, 
                len=13,
            )
            path = tf.strings.join([
                self.params['img_dir'],
                source,
            ])
            image = tf.io.read_file(path)
        return tf.io.decode_jpeg(image)

    @tf.function
    def _augmentation_function(
            self, 
            image, 
            joints, 
            weights, 
            center, 
            scale,
        ):
        image, joints, weights, center = self._random_flipping(
            image, 
            joints, 
            weights, 
            center,
        )
        image = self._random_color_jittering(image)
        image = tf.clip_by_value(
            image, 
            clip_value_min=0, 
            clip_value_max=255,
        )
        scale = self._random_scaling(scale)
        rotation = self._random_rotating()

        return image, joints, weights, center, scale, rotation

    def _random_flipping(
            self, 
            image, 
            joints, 
            weights, 
            center,
        ):
        random_chance = self.random_generator.uniform(
            shape=[], 
            minval=0, 
            maxval=2, 
            dtype=tf.int32,
        )
        if random_chance == 0:
            image = tf.image.flip_left_right(image)
            width = tf.cast(tf.shape(image)[1], tf.float32)
            joint_weights = tf.TensorArray(
                tf.float32, 
                size=NUM_JOINTS, 
                dynamic_size=False, 
                clear_after_read=False,
            )
            for j in tf.range(NUM_JOINTS):
                joint_weights = joint_weights.write(
                    index=j, 
                    value=[
                        width - joints[j,0], 
                        joints[j,1], 
                        weights[j],
                    ],
                )
            paired_joints = tf.constant(PAIRED_JOINTS)
            single_joints = tf.constant(SINGLE_JOINTS)

            joint_weights_flipped = tf.TensorArray(
                tf.float32, 
                size=NUM_JOINTS, 
                dynamic_size=False,
            )
            for sj in single_joints:
                joint_weights_flipped = joint_weights_flipped.write(
                    index=sj, 
                    value=joint_weights.read(sj),
                )
            for pj in paired_joints:
                temp = joint_weights.read(pj[0])
                joint_weights_flipped = joint_weights_flipped.write(
                    index=pj[0], 
                    value=joint_weights.read(pj[1]),
                )
                joint_weights_flipped = joint_weights_flipped.write(
                    index=pj[1], 
                    value=temp,
                )
            joint_weights_flipped = joint_weights_flipped.stack()
            joints = joint_weights_flipped[:,:2]
            weights = joint_weights_flipped[:,2]

            center_flipped = tf.TensorArray(
                tf.float32, 
                size=1, 
                dynamic_size=False,
            )
            center_flipped = center_flipped.write(
                index=0, 
                value=[
                    width - center[0], 
                    center[1]
                ],
            )
            center = tf.squeeze(center_flipped.stack())
            return image, joints, weights, center
        else:
            return image, joints, weights, center
    
    def _random_color_jittering(self, image):
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(
            image, 
            lower=0.5, 
            upper=1.5,
        )
        # Option 2: Remove random contrast.
        return image 

    def _random_scaling(self, scale):
        random_number = self.random_generator.normal([1])
        scale = (
            scale *
            tf.clip_by_value(
                0.25 * random_number + 1, 
                clip_value_min=0.75, 
                clip_value_max=1.25,
            )
        )
        return scale[0]

    def _random_rotating(self):
        random_chance = self.random_generator.uniform(
            shape=[], 
            minval=0, 
            maxval=3, 
            dtype=tf.int32,
        )
        random_number = self.random_generator.normal([1])

        if random_chance == 0:              
            rotation = tf.abs(tf.clip_by_value(
                random_number * 15, 
                clip_value_min=-30, 
                clip_value_max=30,
            ))
        elif random_chance == 1:
            rotation = (
                360.0 - 
                tf.abs(tf.clip_by_value(
                    random_number * 15, 
                    clip_value_min=-30, 
                    clip_value_max=30,
                ))
            )
        else:
            rotation = tf.zeros([1])
        # Option 2: Change (random_number * 15) to (random_number * 30).
        return rotation[0]

    @tf.function
    def _generating_function(
            self, 
            image, 
            joints, 
            weights, 
            center, 
            scale, 
            rotation,
        ):
        image = self._get_cropped_image(
            image, 
            center, 
            scale, 
            rotation,
        )
        heatmap, weights = self._get_heatmap(
            joints, 
            weights, 
            center, 
            scale, 
            rotation,
        )
        image = tf.reshape(
            image, 
            shape=[
                self.params['img_size'], 
                self.params['img_size'], 
                3,
            ],
        )     
        heatmap = tf.reshape(
            heatmap, 
            shape=[
                self.params['hm_size'],
                self.params['hm_size'],
                NUM_JOINTS,
            ],
        )   
        return image, heatmap, weights, center, scale
    
    def _get_cropped_image(
            self, 
            image, 
            center, 
            scale, 
            rotation,
        ):
        height = tf.cast(tf.shape(image)[0], tf.float32)
        width = tf.cast(tf.shape(image)[1], tf.float32)
        sf = scale * 200.0 / tf.cast(self.params['img_size'], tf.float32)
        if sf < 2:
            return self._crop_image(
                image, 
                center, 
                scale, 
                rotation,
            )
        else:
            max_size = tf.cast(
                tf.floor(tf.maximum(height, width) / sf), 
                tf.int32,
            )
            new_height = tf.cast(tf.floor(height / sf), tf.int32)
            new_width = tf.cast(tf.floor(width / sf), tf.int32)
            if max_size < 2:
                return tf.zeros([
                    self.params['img_size'], 
                    self.params['img_size'], 
                    tf.shape(image)[2],
                ])
            else:
                image = tf.image.resize(image, size=[new_height, new_width])
                center = center / sf
                scale = scale / sf
                return self._crop_image(
                    image, 
                    center, 
                    scale, 
                    rotation,
                )

    def _crop_image(
            self, 
            image, 
            center, 
            scale, 
            rotation,
        ):
        ul_coords = coord_transform(
            tf.zeros([2]), 
            center, 
            scale, 
            rotation=tf.zeros([]),
            invert=tf.ones([], tf.int32),
            size=tf.cast(self.params['img_size'], tf.float32),
        )
        br_coords = coord_transform(
            tf.cast(
                [self.params['img_size'], self.params['img_size']], 
                tf.float32,
            ), 
            center, 
            scale, 
            rotation=tf.zeros([]),
            invert=tf.ones([], tf.int32),
            size=tf.cast(self.params['img_size'], tf.float32),
        )
        pad = tf.norm(
            tf.cast(br_coords - ul_coords, tf.float32)
        )
        pad = pad / 2
        pad = pad - tf.cast(br_coords[1] - ul_coords[1], tf.float32) / 2
        pad = tf.cast(pad, tf.int32)

        if rotation != 0.:
            ul_coords = ul_coords - pad
            br_coords = br_coords + pad

        x_min = tf.maximum(0, ul_coords[0])
        x_max = tf.minimum(tf.shape(image)[1], br_coords[0])
        y_min = tf.maximum(0, ul_coords[1])
        y_max = tf.minimum(tf.shape(image)[0], br_coords[1])
        x_min_margin = tf.maximum(0, -ul_coords[0])
        x_max_margin = (
            tf.minimum(br_coords[0], tf.shape(image)[1]) 
            - ul_coords[0]
        )
        y_min_margin = tf.maximum(0, -ul_coords[1]) 
        y_max_margin = (
            tf.minimum(br_coords[1], tf.shape(image)[0]) 
            - ul_coords[1]
        )
        
        if x_max_margin < x_min_margin:
            temp = x_max_margin
            x_max_margin = x_min_margin
            x_min_margin = temp
            temp = x_min
            x_min = x_max
            x_max = temp

        top = y_min_margin
        bottom = (br_coords[1] - ul_coords[1]) - y_max_margin
        left = x_min_margin
        right = (br_coords[0] - ul_coords[0]) - x_max_margin
            
        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_image = tf.pad(
            cropped_image, 
            paddings=[
                [top, bottom], 
                [left, right], 
                [0, 0],
            ],
        )
        if rotation != 0.:
            cropped_image = tfa.image.rotate(
                cropped_image, 
                angles=rotation * (PI / 180),
            )
            cropped_image = cropped_image[pad:-pad, pad:-pad]
        return tf.image.resize(
            cropped_image, 
            size=[
                self.params['img_size'],
                self.params['img_size'],
            ],
        )

    def _get_heatmap(
            self, 
            joints, 
            weights, 
            center, 
            scale, 
            rotation,
        ):
        hm_size = tf.cast(self.params['hm_size'], tf.float32)
        hm = tf.TensorArray(
            tf.float32, 
            size=NUM_JOINTS, 
            dynamic_size=False,
        )
        w = tf.TensorArray(
            tf.float32, 
            size=NUM_JOINTS, 
            dynamic_size=False,
        )
        for j in tf.range(NUM_JOINTS):
            if joints[j, 1] > 0:
                joints_transformed = coord_transform(
                    joints[j,:] + 1, 
                    center, 
                    scale, 
                    rotation, 
                    tf.zeros([], tf.int32),
                    hm_size,
                )
                heatmap, visible = self._draw_heatmap(joints_transformed - 1)
                hm = hm.write(j, value=heatmap)
                w = w.write(j, value=weights[j] * visible)
            else:
                hm = hm.write(j, value=tf.zeros([hm_size, hm_size]))
                w = w.write(j, value=weights[j])
        heatmap = tf.transpose(hm.stack(), perm=[1,2,0])
        weights = w.stack()
        return heatmap, weights

    def _draw_heatmap(self, joints):
        coords = tf.TensorArray(
            tf.int32, 
            size=2, 
            dynamic_size=False,
        )
        coords = coords.write(
            index=0, 
            value=[
                tf.cast(joints[0] - 3 * self.params['sigma'], tf.int32),
                tf.cast(joints[1] - 3 * self.params['sigma'], tf.int32),
            ],
        )
        coords = coords.write(
            index=1,
            value=[
                tf.cast(joints[0] + 3 * self.params['sigma'] + 1, tf.int32),
                tf.cast(joints[1] + 3 * self.params['sigma'] + 1, tf.int32,),
            ],
        )
        coords = coords.stack()
        if (
            coords[0,0] >= self.params['hm_size']
            or coords[0,1] >= self.params['hm_size']
            or coords[1,0] < 0
            or coords[1,1] < 0
        ):
            heatmap = tf.zeros([
                self.params['hm_size'], 
                self.params['hm_size'],
            ])
            visible = tf.zeros([])
            return heatmap, visible

        gaussian = self._get_gaussian(coords)    
        padding = self._get_padding(coords)
        heatmap = tf.pad(gaussian, padding)
        visible = tf.ones([])
        return heatmap, visible
    
    def _get_gaussian(self, coords):
        size = tf.constant(6. * self.params['sigma'] + 1.)
        x = tf.range(
            start=0., 
            limit=size, 
            delta=1,
        )
        y = tf.expand_dims(x, 1)
        x0 = y0 = tf.floor(size / 2)
        gaussian = tf.exp(
            - ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.params['sigma'] ** 2)
        )

        x_min = tf.maximum(0, -coords[0,0])
        x_max = tf.minimum(coords[1,0], self.params['hm_size']) - coords[0,0]
        y_min = tf.maximum(0, -coords[0,1])
        y_max = tf.minimum(coords[1,1], self.params['hm_size']) - coords[0,1]

        return gaussian[y_min:y_max, x_min:x_max]

    def _get_padding(self, coords):
        x_min = tf.maximum(0, coords[0,0])
        x_max = tf.minimum(coords[1,0], self.params['hm_size'])
        y_min = tf.maximum(0, coords[0,1])
        y_max = tf.minimum(coords[1,1], self.params['hm_size'])

        top = y_min if y_min != 0 else 0
        bottom = self.params['hm_size'] - y_max
        left = x_min if x_min != 0 else 0
        right = self.params['hm_size'] - x_max

        padding = tf.TensorArray(
            tf.int32, 
            size=2, 
            dynamic_size=False,
        )
        padding = padding.write(0, value=[top, bottom])
        padding = padding.write(1, value=[left, right])
        return padding.stack()

    @tf.function
    def _expansion_function(
            self, 
            image, 
            heatmap, 
            weights,
        ):
        weights = tf.expand_dims(weights, axis=0)
        weights = tf.expand_dims(weights, axis=0)
        weights = tf.tile(
            weights, 
            multiples=[
                self.params['hm_size'], 
                self.params['hm_size'], 
                1,
            ],
        )
        return image, heatmap, weights








        