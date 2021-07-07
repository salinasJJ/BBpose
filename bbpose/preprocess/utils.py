import tensorflow as tf


PI = 3.14159265


def coord_transform(
        coords, 
        center, 
        scale, 
        rotation, 
        invert,
        size,
    ):
    transfromation_matrix = get_transformation_matrix(
        center, 
        scale, 
        rotation, 
        invert, 
        size,
    )
    new_coords = tf.TensorArray(
        tf.float32, 
        size=3, 
        dynamic_size=False,
    )
    new_coords = new_coords.write(0, value=coords[0] - 1.)
    new_coords = new_coords.write(1, value=coords[1] - 1.)
    new_coords = new_coords.write(2, value=1.)
    new_coords = new_coords.stack()
    new_coords = tf.tensordot(
        transfromation_matrix, 
        new_coords,
        axes=1,
    )
    return tf.cast(new_coords[:2], tf.int32) + 1

def get_transformation_matrix(
        center, 
        scale, 
        rotation, 
        invert, 
        size,
    ):
    transfromation_matrix = tf.TensorArray(
        tf.float32, 
        size=3, 
        dynamic_size=False,
    )
    transfromation_matrix = transfromation_matrix.write(
        index=0, 
        value=[
            size / (scale * 200.), 
            0., 
            size * (-center[0] / (scale * 200.) + 0.5),
        ],
    )
    transfromation_matrix = transfromation_matrix.write(
        index=1, 
        value=[
            0., 
            size / (scale * 200.), 
            size * (-center[1] / (scale * 200.) + 0.5),
        ],
    )
    transfromation_matrix = transfromation_matrix.write(2, value=[0., 0., 1.])
    transfromation_matrix = transfromation_matrix.stack()

    if rotation != 0.:
        transfromation_matrix = rotate_transformation_matrix(
            transfromation_matrix,
            rotation,
            size,
        )
    if invert == 1:
        transfromation_matrix = tf.linalg.inv(transfromation_matrix)
    return transfromation_matrix

def rotate_transformation_matrix(
        transfromation_matrix, 
        rotation, 
        size,
    ):
    rotation = -rotation
    sn = tf.sin(rotation * (PI / 180))
    csn = tf.cos(rotation * (PI / 180))
    
    rot_matrix = tf.TensorArray(
        tf.float32, 
        size=3, 
        dynamic_size=False,
    )
    rot_matrix = rot_matrix.write(0, value=[csn, -sn, 0])
    rot_matrix = rot_matrix.write(1, value=[sn, csn, 0])
    rot_matrix = rot_matrix.write(2, value=[0, 0, 1])
    rot_matrix = rot_matrix.stack()
    
    tr_matrix = tf.TensorArray(
        tf.float32, 
        size=3, 
        dynamic_size=False,
    )
    tr_matrix = tr_matrix.write(0, value=[1, 0, -size / 2])
    tr_matrix = tr_matrix.write(1, value=[0, 1, -size / 2])
    tr_matrix = tr_matrix.write(2, value=[0, 0, 1])
    tr_matrix = tr_matrix.stack()
    
    inv_matrix = tf.TensorArray(
        tf.float32, 
        size=3, 
        dynamic_size=False,
    )
    inv_matrix = inv_matrix.write(0, value=[1, 0, size / 2])
    inv_matrix = inv_matrix.write(1, value=[0, 1, size / 2])
    inv_matrix = inv_matrix.write(2, value=[0, 0, 1])
    inv_matrix = inv_matrix.stack()

    transfromation_matrix = tf.tensordot(
        tr_matrix, 
        transfromation_matrix, 
        axes=1,
    )
    transfromation_matrix = tf.tensordot(
        rot_matrix, 
        transfromation_matrix, 
        axes=1,
    )
    transfromation_matrix = tf.tensordot(
        inv_matrix, 
        transfromation_matrix, 
        axes=1,
    )
    return transfromation_matrix


