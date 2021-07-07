import tensorflow as tf


def get_max_indices(heatmaps):
    batch_size = tf.shape(heatmaps)[0]
    hm_size = tf.cast(tf.shape(heatmaps)[2], tf.float32)
    num_joints = tf.shape(heatmaps)[3]
    tf.debugging.assert_equal(
        tf.rank(heatmaps), 
        4, 
        message='heatmaps should be of rank 4',
    )

    hm_reshaped = tf.reshape(heatmaps, [batch_size, -1, num_joints])
    max_vals = tf.reduce_max(hm_reshaped, axis=1)
    max_vals = tf.reshape(max_vals, [batch_size, 1, num_joints])

    max_idx = tf.argmax(hm_reshaped, axis=1)
    max_idx = tf.reshape(max_idx, [batch_size, 1, num_joints])
    max_idx = tf.cast(
        tf.tile(
            max_idx + 1, 
            multiples=[1, 2, 1],
        ), 
        tf.float32,
    )

    max_indices = tf.TensorArray(
        tf.float32, 
        size=2, 
        dynamic_size=False,
    )
    max_indices = max_indices.write(
        index=0, 
        value=(max_idx[:, 0, :] - 1) % hm_size + 1,
    )
    max_indices = max_indices.write(
        index=1, 
        value=tf.floor((max_idx[:, 1, :] - 1) / hm_size) + 1,
    )
    max_indices = max_indices.stack()
    max_indices = tf.transpose(max_indices, perm=[1,0,2])

    mask = tf.cast(
        tf.greater(max_vals, 0.0), 
        tf.float32,
    )
    mask = tf.tile(mask, multiples=[1,2,1])
    return max_indices * mask







