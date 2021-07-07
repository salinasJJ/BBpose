import json

import tensorflow as tf
import scipy.io

from preprocess.utils import coord_transform
from train.utils import get_max_indices
from utils import (
    get_data_dir, 
    get_frozen_params, 
    get_params, 
    get_results_dir,
    is_file,
)


NUM_JOINTS = 16
PRECISION = 0.6


class PCKh():
    def __init__(self):
        self.params = get_frozen_params(
            'Test', 
            version=get_params('Test')['version'],
        )
        self.results_dir = get_results_dir(self.params['dataset'])
        self.dataset_dir = get_data_dir(self.params['dataset'])
        if self.params['toy_set']:
            self.data_dir = self.dataset_dir + 'data/toy/'
        else:
            self.data_dir = self.dataset_dir + 'data/full/'

        self.num_joints = NUM_JOINTS

    @tf.function
    def get_final_predictions(
            self, 
            preds, 
            center, 
            scale,
        ):
        coords = self._get_coords(preds)

        batch_size = tf.shape(preds)[0]
        predictions = tf.TensorArray(
            tf.float32, 
            size=batch_size, 
            dynamic_size=False,
        )
        for n in tf.range(batch_size):
            predictions = predictions.write(
                index=n, 
                value=self._transform_coords(
                    coords[n], 
                    center[n], 
                    scale[n])
                )
        return predictions.stack()

    def _get_coords(self, preds):
        max_coords = get_max_indices(preds)
        batch_size = tf.shape(preds)[0]
        example_coords = tf.TensorArray(
            tf.float32, 
            size=batch_size, 
            dynamic_size=False,
        )
        for n in tf.range(batch_size):
            joint_coords = tf.TensorArray(
                tf.float32, 
                size=self.num_joints, 
                dynamic_size=False,
            )
            for j in tf.range(self.num_joints):
                hm = preds[n, :, :, j]
                max_x = tf.cast(tf.floor(max_coords[n, 0, j]), tf.int32)
                max_y = tf.cast(tf.floor(max_coords[n, 1, j]), tf.int32)
                if (
                    max_x > 1 
                    and max_x < self.params['hm_size'] 
                    and max_y > 1 
                    and max_y < self.params['hm_size']
                ):
                    diff = tf.TensorArray(
                        tf.float32, 
                        size=1, 
                        dynamic_size=False,
                    )
                    diff = diff.write(
                        index=0, 
                        value=[
                            hm[max_y - 1][max_x] - hm[max_y - 1][max_x - 2], 
                            hm[max_y][max_x - 1] - hm[max_y - 2][max_x - 1],
                        ],
                    )
                    diff = diff.stack()
                    diff = tf.squeeze(diff)
                    joint_coords = joint_coords.write(
                        index=j, 
                        value=max_coords[n,:,j] + tf.sign(diff) * 0.25,
                    )
                else:
                    joint_coords = joint_coords.write(j, max_coords[n,:,j])
            joint_coords = joint_coords.stack()
            example_coords = example_coords.write(n, joint_coords)
        coords = example_coords.stack() + 0.5
        return tf.transpose(coords, perm=[0,2,1])

    def _transform_coords(
            self, 
            coords, 
            center, 
            scale,
        ):
        coords_transformed = tf.TensorArray(
            tf.float32, 
            size=self.num_joints, 
            dynamic_size=False,
        )
        for j in tf.range(self.num_joints):
            temp = tf.cast(
                coord_transform(
                    coords[0:2, j], 
                    center, 
                    scale,
                    rotation=tf.zeros([]),
                    invert=tf.ones([], tf.int32),
                    size=tf.cast(self.params['hm_size'], tf.float32),
                ),
                tf.float32,
            )
            coords_transformed = coords_transformed.write(j, temp)
        coords_transformed = coords_transformed.stack()
        return tf.transpose(coords_transformed, [1,0])

    def get_results(self, y_pred):
        y_true = scipy.io.loadmat(self.data_dir + 'detections.mat')
        joints_missing = tf.cast(y_true['jnt_missing'], tf.float32)
        hm_true = tf.cast(y_true['pos_gt_src'], tf.float32)
        headboxes = tf.cast(y_true['headboxes_src'], tf.float32)
        if self.params['toy_set']:
            joints_missing = joints_missing[...,:y_pred.shape[0]]
            hm_true = hm_true[...,:y_pred.shape[0]]
            headboxes = headboxes[...,:y_pred.shape[0]]

        hm_pred = tf.transpose(y_pred, perm=[2, 1, 0])

        head_norm = tf.norm(
            headboxes[1, :, :] - headboxes[0, :, :], 
            axis=0,
        ) 
        visible = 1 - joints_missing
        distances = tf.norm(hm_pred - hm_true, axis=1) / (head_norm * PRECISION)
        distances = distances * visible

        below_threshold = tf.less(distances, self.params['threshold'])
        below_threshold = tf.cast(below_threshold, tf.float32) * visible

        total_visible = tf.reduce_sum(visible, axis=1)
        results = tf.reduce_sum(below_threshold, axis=1) / total_visible 
        return self._get_results_dict(results * 100.)

    def _get_results_dict(self, results):
        return {
            'head':float(
                results[9].numpy()
            ), 
            'shoulder':float(
                (results[12] + results[13]).numpy() / 2
            ),
            'elbow':float(
                (results[11] + results[14]).numpy() / 2
            ),
            'wrist':float(
                (results[10] + results[15]).numpy() / 2
            ),
            'hip':float(
                (results[2] + results[3]).numpy() / 2
            ),
            'knee':float(
                (results[1] + results[4]).numpy() / 2
            ),
            'ankle':float(
                (results[0] + results[5]).numpy() / 2
            ),
            'mean':float(
                tf.reduce_mean(tf.concat(
                    values=[
                        results[:6],
                        results[8:],
                    ],
                    axis=0,
                )).numpy()
            ),
        }
  
    def save_results(self, results_dict):
        is_file(self.results_dir, 'results.json')
        json_file = self.results_dir + 'results.json'
        print(f"\nSaving results to {json_file}")
      
        results = {}
        results[f"v{self.params['version']}"] = results_dict
        with open(json_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(
                results, 
                ensure_ascii=False, 
                indent=4,
            ))

    def load_results(self):
        with open(self.results_dir + 'results.json', 'r') as f:
            results = json.load(f)
        try:
            return results[f"v{self.params['version']}"]
        except:
            return (
                f"No test results were found for version "
                f"'{self.params['version']}'."
            )









