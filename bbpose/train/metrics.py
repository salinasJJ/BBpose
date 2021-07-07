import tensorflow as tf
from tensorflow.keras import metrics

from train.utils import get_max_indices
from utils import get_frozen_params, get_params


NUM_JOINTS = 16
EXCLUDE = 6
PRECISION = 0.1


class PercentageOfCorrectKeypoints(metrics.Metric):
    def __init__(self, name='pck', **kwargs):
        super(PercentageOfCorrectKeypoints, self).__init__(name=name)
        
        params = get_params('Train')
        if params['switch']:
            self.params = get_frozen_params(
                'Train', 
                version=params['version'],
            )
        else:
            self.params = params
        
        self.num_stacks = self.params['num_stacks']
        self.hm_size = self.params['hm_size']
        self.threshold = self.params['threshold']
        self.num_joints = NUM_JOINTS - EXCLUDE

        self.accuracy = tf.Variable(0.0, trainable=False)
        self.num_batches = tf.Variable(0.0, trainable=False)

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def get_config(self):
        config = {
            'num_stacks':self.num_stacks,
            'hm_size':self.hm_size,
            'threshold':self.threshold,
            'num_joints':self.num_joints,
        }
        base_config = super(PercentageOfCorrectKeypoints, self).get_config()    
        return dict(
            list(base_config.items()) + list(config.items())
        )

    @tf.function
    def update_state(
            self, 
            y_true, 
            y_pred, 
            sample_weight=None,
        ):
        tf.debugging.assert_equal(
            sample_weight, 
            None,
            message='sample_weight is not required',
        )
        hm_pred = tf.concat(
            values=[
                y_pred[...,:6], 
                y_pred[...,10:12], 
                y_pred[...,14:],
            ],
            axis=4,
        )
        hm_pred = hm_pred[:, self.num_stacks-1, :, :, :]
        hm_true = tf.concat(
            values=[
                y_true[...,:6], 
                y_true[...,10:12], 
                y_true[...,14:],
            ], 
            axis=3,
        )

        batch_acc = self._get_batch_acc(hm_pred, hm_true)
        self.accuracy.assign_add(batch_acc)
        self.num_batches.assign_add(1.)

    def result(self):
        return self.accuracy / self.num_batches

    def reset_state(self):
        self.accuracy.assign(0.)
        self.num_batches.assign(0.)

    def _get_batch_acc(self, hm_pred, hm_true):
        pred_max = get_max_indices(hm_pred)
        true_max = get_max_indices(hm_true)
        l2_distances = self._get_l2_distances(pred_max, true_max)

        batch_acc = tf.zeros([])
        visible_joints = tf.zeros([])

        joint_acc = tf.TensorArray(
            tf.float32, 
            size=self.num_joints+1, 
            dynamic_size=False, 
            clear_after_read=False,
        )
        for j in tf.range(self.num_joints):
            joint_acc = joint_acc.write(
                index=j, 
                value=self._get_joint_acc(l2_distances[:,j]),
            )
            if joint_acc.read(j) >= 0.:
                batch_acc = batch_acc + joint_acc.read(j)
                visible_joints = visible_joints + 1.

        if visible_joints != 0.:
            joint_acc = joint_acc.write(
                index=self.num_joints, 
                value=batch_acc / visible_joints,
            )
            joint_acc = joint_acc.stack()
            return joint_acc[-1]
        else:
            return tf.zeros([])

    def _get_l2_distances(self, pred_max, true_max):
        batch_size = tf.shape(true_max)[0]
        distances = tf.TensorArray(
            tf.float32, 
            size=batch_size, 
            dynamic_size=False,
        )
        for n in tf.range(batch_size):
            dist = tf.TensorArray(
                tf.float32, 
                size=self.num_joints, 
                dynamic_size=False,
            )
            for j in tf.range(self.num_joints):
                if true_max[n, 0, j] > 1 and true_max[n, 1, j] > 1:
                    dist = dist.write(
                        index=j, 
                        value=(
                            tf.norm(pred_max[n, :, j] - true_max[n, :, j]) 
                            / (self.hm_size * PRECISION)
                        )
                    )
                else:
                    dist = dist.write(j, -1.)
            dist = dist.stack()
            distances = distances.write(n, dist)
        return distances.stack()

    def _get_joint_acc(self, joint_dists):
        visible = joint_dists[tf.not_equal(joint_dists, -1.)]
        if tf.size(visible) > 0:
            below_thr = tf.less(visible, self.threshold)
            below_thr = tf.reduce_sum(
                tf.cast(below_thr, tf.float32)
            )
            return  below_thr / tf.cast(tf.size(visible), tf.float32)
        else:
            return -tf.ones([])
        














