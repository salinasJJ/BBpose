import tensorflow as tf
from tensorflow.keras import losses

from utils import get_frozen_params, get_params


NUM_JOINTS = 16


class JointsMSE(losses.Loss):
    def __init__(
            self, 
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, 
            name='joints_mse',
            **kwargs
        ):
        super(JointsMSE, self).__init__(reduction=reduction, name=name)

        params = get_params('Train')
        if params['switch']:
            self.params = get_frozen_params(
                'Train', 
                version=params['version'],
            )
        else:
            self.params = params

        self.batch_size = self.params['batch_per_replica']
        self.num_stacks = self.params['num_stacks']
        self.hm_size = self.params['hm_size']
        self.num_joints = NUM_JOINTS

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def get_config(self):
        config = {
            'num_stacks':self.num_stacks,
            'hm_size':self.hm_size,
            'batch_size':self.batch_size,
            'num_joints':self.num_joints,
        }
        base_config = super(JointsMSE, self).get_config()    
        return dict(
            list(base_config.items()) + list(config.items())
        )

    @tf.function
    def __call__(
            self, 
            y_true, 
            y_pred, 
            sample_weight=None,
        ):
        hm_pred = tf.reshape(
            y_pred, 
            shape=[
                self.batch_size, 
                self.num_stacks, 
                tf.square(self.hm_size), 
                self.num_joints,
            ],
        )
        hm_true = tf.reshape(
            y_true, 
            shape=[
                self.batch_size, 
                tf.square(self.hm_size), 
                self.num_joints,
            ],
        )  
        weights = tf.reshape(
            sample_weight, 
            shape=[
                self.batch_size, 
                tf.square(self.hm_size), 
                self.num_joints,
            ],
        )    
        
        loss = tf.zeros([self.batch_size])
        for s in tf.range(self.num_stacks):
            j_loss = tf.zeros([self.batch_size])
            for j in tf.range(self.num_joints):
                weighted_pred = weights[:,:,j] * hm_pred[:,s,:,j]
                weighted_true = weights[:,:,j] * hm_true[:,:,j]
                j_loss = (
                    j_loss 
                    + 0.5 
                    * tf.keras.losses.MSE(weighted_true, weighted_pred)
                )
            loss = loss + j_loss / self.num_joints
        return loss



        
