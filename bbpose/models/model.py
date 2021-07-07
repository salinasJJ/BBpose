import tensorflow as tf
from tensorflow.keras import layers

from models import hourglass, softgate
from utils import get_frozen_params, get_params, reload_modules


PARAMS = get_params('Models')
if PARAMS['switch']:
    PARAMS = get_frozen_params(
        'Models',
        version=PARAMS['version'],
    )

NUM_JOINTS = 16
DEPTH = 4


def block(inputs, filters):
    if PARAMS['arch'] == 'hourglass':
        return hourglass.residual_block(inputs, filters)
    elif PARAMS['arch'] == 'softgate':
        return softgate.skip_block(inputs, filters)

def bottleneck(inputs, final=tf.constant(0)):
    if PARAMS['arch'] == 'hourglass':
        x = hourglass.encoder_to_decoder(inputs, tf.constant(DEPTH))
    elif PARAMS['arch'] ==  'softgate':
        e_out, skip1, skip2, skip3, skip4 = softgate.encoder(inputs)
        x = softgate.decoder(e_out, skip1, skip2, skip3, skip4)

    x = layers.Dropout(rate=PARAMS['dropout_rate'])(x)
    x = block(x, tf.cast(PARAMS['num_filters'], tf.int32))

    x = layers.Conv2D(
        filters=PARAMS['num_filters'],
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

    scores = layers.Conv2D(
        filters=NUM_JOINTS,
        kernel_size=(1,1),
        strides=(1,1),
        kernel_initializer=PARAMS['initializer'],
        )(x)
    if final == 0:
        x_lower = layers.Conv2D(
            filters=PARAMS['num_filters'],
            kernel_size=(1,1),
            strides=(1,1),
            kernel_initializer=PARAMS['initializer'],
            )(scores)
        x_upper = layers.Conv2D(
            filters=PARAMS['num_filters'],
            kernel_size=(1,1),
            strides=(1,1),
            kernel_initializer=PARAMS['initializer'],
            )(x)
        x = layers.Add()([
            x_lower, 
            x_upper, 
            inputs,
        ])
        return scores, x
    else:
        return scores

def network(inputs):
    x = layers.ZeroPadding2D(padding=(3,3))(inputs)
    x = layers.Conv2D(
        filters=PARAMS['num_filters']//4,
        kernel_size=(7,7),
        strides=(2,2),
        kernel_initializer=PARAMS['initializer'],
        )(x)
    x = layers.BatchNormalization(
        momentum=PARAMS['momentum'], 
        epsilon=PARAMS['epsilon'], 
        scale=False,
        )(x)
    x = layers.ReLU()(x)

    x = block(x, tf.cast(PARAMS['num_filters'] / 2, tf.int32))
    x = layers.MaxPool2D(
        pool_size=(2,2), 
        strides=(2,2),
        )(x)
    x = block(x, tf.cast(PARAMS['num_filters'], tf.int32))
    x = block(x, tf.cast(PARAMS['num_filters'],tf.int32))

    hm1, x = bottleneck(x)
    if PARAMS['num_stacks'] == 8:
        hm2, x = bottleneck(x)
        hm3, x = bottleneck(x)
        hm4, x = bottleneck(x)
        hm5, x = bottleneck(x)
        hm6, x = bottleneck(x)
        hm7, x = bottleneck(x)
        hm8 = bottleneck(x, final=tf.constant(1))
        return tf.stack(
            [hm1, hm2, hm3, hm4, hm5, hm6, hm7, hm8], 
            axis=1,
        )
    elif PARAMS['num_stacks'] == 4:
        hm2, x = bottleneck(x)
        hm3, x = bottleneck(x)
        hm4 = bottleneck(x, final=tf.constant(1))
        return tf.stack(
            [hm1, hm2, hm3, hm4], 
            axis=1,
        )
    else:
        hm2 = bottleneck(x, final=tf.constant(1))
        return tf.stack(
            [hm1, hm2], 
            axis=1,
        )

def get_model():
    if PARAMS['arch'] == 'hourglass':
        reload_modules(hourglass)
    elif PARAMS['arch'] == 'softgate':
        reload_modules(softgate)

    inputs = tf.keras.layers.Input([
        PARAMS['img_size'], 
        PARAMS['img_size'], 
        3,
    ])
    outputs = network(inputs)
    outputs = layers.Activation(
        'linear', 
        dtype='float32',
        )(outputs)
    return tf.keras.Model(inputs, outputs)







