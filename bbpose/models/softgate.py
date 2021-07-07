import tensorflow as tf
from tensorflow.keras import layers

from utils import get_frozen_params, get_params


PARAMS = get_params('Models')
if PARAMS['switch']:
    PARAMS = get_frozen_params(
        'Models',
        version=PARAMS['version'],
    )


def skip_block(inputs, filters):
    x1 = layers.BatchNormalization(
        momentum=PARAMS['momentum'], 
        epsilon=PARAMS['epsilon'], 
        scale=False,
        )(inputs)
    x1 = layers.ReLU()(x1)
    x1 = layers.Conv2D(
        filters=int(filters/2), 
        kernel_size=(3,3), 
        strides=(1,1), 
        padding='same',
        kernel_initializer=PARAMS['initializer'],
        )(x1)
    
    x2 = layers.BatchNormalization(
        momentum=PARAMS['momentum'], 
        epsilon=PARAMS['epsilon'], 
        scale=False,
        )(x1)
    x2 = layers.ReLU()(x2)
    x2 = layers.Conv2D(
        filters=int(filters/4), 
        kernel_size=(3,3), 
        strides=(1,1), 
        padding='same',
        kernel_initializer=PARAMS['initializer'],
        )(x2)

    x3 = layers.BatchNormalization(
        momentum=PARAMS['momentum'], 
        epsilon=PARAMS['epsilon'], 
        scale=False,
        )(x2)
    x3 = layers.ReLU()(x3)
    x3 = layers.Conv2D(
        filters=int(filters/4),
        kernel_size=(3,3), 
        strides=(1,1), 
        padding='same',
        kernel_initializer=PARAMS['initializer'],
        )(x3)
    x = layers.concatenate(
        [x1, x2, x3], 
        axis=3,
    )

    if inputs.shape[3] == x.shape[3]:
        skip = layers.DepthwiseConv2D(
            kernel_size=1, 
            strides=(1,1), 
            depthwise_initializer=PARAMS['initializer'],
            )(inputs)
    else:
        skip = layers.Conv2D(
            filters=int(filters), 
            kernel_size=(1,1), 
            strides=(1,1), 
            kernel_initializer=PARAMS['initializer'],
            )(inputs)
    return tf.math.add_n([x, skip])

def encoder(inputs):
    skip1 = skip_block(inputs, filters=inputs.shape[3])
    e1 = layers.MaxPool2D(pool_size=(2,2))(inputs)
    e1 = skip_block(e1, filters=e1.shape[3])
    
    skip2 = skip_block(e1, filters=e1.shape[3])
    e2 = layers.MaxPool2D(pool_size=(2,2))(e1)
    e2 = skip_block(e2, filters=e2.shape[3]//2) 

    skip3 = skip_block(e2, filters=e2.shape[3])
    e3 = layers.MaxPool2D(pool_size=(2,2))(e2)
    e3 = skip_block(e3, filters=e3.shape[3]) 

    skip4 = skip_block(e3, filters=e3.shape[3])
    e4 = layers.MaxPool2D(pool_size=(2,2))(e3)
    e4 = skip_block(e4, filters=e4.shape[3]) 

    return e4, skip1, skip2, skip3, skip4

def decoder(e_out, skip1, skip2, skip3, skip4):
    d4 = layers.UpSampling2D(size=(2,2))(e_out)
    d4 = layers.concatenate(
        [skip4, d4], 
        axis=3,
    )
    d4 = layers.Conv2D(
        filters=int(d4.shape[3]/2), 
        kernel_size=(3,3), 
        strides=(1,1), 
        padding='same',
        kernel_initializer=PARAMS['initializer'],
        )(d4)
    d4 = skip_block(d4, filters=d4.shape[3])            

    d3 = layers.UpSampling2D(size=(2,2))(d4)
    d3 = layers.concatenate(
        [skip3, d3], 
        axis=3,
    )
    d3 = layers.Conv2D(
        filters=int(d3.shape[3]/2), 
        kernel_size=(3,3), 
        strides=(1,1), 
        padding='same',
        kernel_initializer=PARAMS['initializer'],
        )(d3)
    d3 = skip_block(d3, filters=d3.shape[3]*2)            

    d2 = layers.UpSampling2D(size=(2,2))(d3)
    d2 = layers.concatenate(
        [skip2, d2], 
        axis=3,
    )
    d2 = layers.Conv2D(
        filters=int(d2.shape[3]/2), 
        kernel_size=(3,3), 
        strides=(1,1), 
        padding='same',
        kernel_initializer=PARAMS['initializer'],
        )(d2)
    d2 = skip_block(d2, filters=d2.shape[3])           

    d1 = layers.UpSampling2D(size=(2,2))(d2)
    d1 = layers.concatenate(
        [skip1, d1], 
        axis=3,
    )
    d1 = layers.Conv2D(
        filters=int(d1.shape[3]/2), 
        kernel_size=(3,3), 
        strides=(1,1), 
        padding='same',
        kernel_initializer=PARAMS['initializer'],
        )(d1)
    return skip_block(d1, filters=d1.shape[3])            


