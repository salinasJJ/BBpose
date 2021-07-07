import tensorflow as tf
from tensorflow.keras import layers

from utils import get_frozen_params, get_params


PARAMS = get_params('Models')
if PARAMS['switch']:
    PARAMS = get_frozen_params(
        'Models',
        version=PARAMS['version'],
    )


def residual_block(inputs, filters):
    x = layers.BatchNormalization(
        momentum=PARAMS['momentum'], 
        epsilon=PARAMS['epsilon'], 
        scale=False,
        )(inputs)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        filters=int(filters/2), 
        kernel_size=(1,1), 
        strides=(1,1), 
        kernel_initializer=PARAMS['initializer'],
        )(x)
    x = layers.BatchNormalization(
        momentum=PARAMS['momentum'], 
        epsilon=PARAMS['epsilon'], 
        scale=False,
        )(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        filters=int(filters/2), 
        kernel_size=(3,3), 
        strides=(1,1), 
        padding='same',
        kernel_initializer=PARAMS['initializer'],
        )(x)
    x = layers.BatchNormalization(
        momentum=PARAMS['momentum'], 
        epsilon=PARAMS['epsilon'], 
        scale=False,
        )(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        filters=int(filters),
        kernel_size=(1,1), 
        strides=(1,1), 
        kernel_initializer=PARAMS['initializer'],
        )(x)
    # skip connection
    if inputs.shape[3] == x.shape[3]:
        skip = inputs
    else:
        skip = layers.Conv2D(
            filters=int(filters), 
            kernel_size=(1,1), 
            strides=(1,1), 
            kernel_initializer=PARAMS['initializer'],
            )(inputs)
    return tf.math.add_n([x, skip])

def encoder_to_decoder(inputs, depth):
    filters = tf.cast(PARAMS['num_filters'], tf.int32)

    skip = residual_block(inputs, filters)
    x = layers.MaxPool2D(pool_size=(2,2))(inputs)
    x = residual_block(x, filters)
    if depth > 1:
        x = encoder_to_decoder(x, depth-1)
    else:
        x = residual_block(x, filters)
    x = residual_block(x, filters)
    x = layers.UpSampling2D(size=(2,2))(x)
    return layers.Add()([skip, x])



