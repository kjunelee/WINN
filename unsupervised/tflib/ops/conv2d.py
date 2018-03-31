import tflib as lib

import numpy as np
import tensorflow as tf
from tflib.ops.sn import spectral_normed_weight

_default_weightnorm = False
def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

_weights_stdev = None
def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev

def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None

def scope_has_variables(scope):
    return len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)) > 0

def spectrally_normed_weight(name, W, update_collection=tf.GraphKeys.UPDATE_OPS):
    W_shape = W.shape.as_list()
    number_filters = W_shape[-1]

    # reshape into 2D.
    W_reshaped = tf.reshape(W, [-1, number_filters])

    # initialize u as a random vector.
    u = tf.get_variable(name+"u", (1, number_filters), initializer=tf.truncated_normal_initializer(), trainable=False)

    new_v = tf.nn.l2_normalize(tf.matmul(u, tf.transpose(W_reshaped)), 1)
    new_u = tf.nn.l2_normalize(tf.matmul(new_v, W_reshaped), 1)
    new_u = tf.stop_gradient(new_u)
    new_v = tf.stop_gradient(new_v)

    sigma = tf.reduce_sum(tf.matmul(new_u, tf.transpose(W_reshaped)) * new_v, axis=1)
    W_bar = W_reshaped / sigma
    W_bar = tf.reshape(W_bar, W_shape)

    tf.add_to_collection(update_collection, u.assign(new_u))
    return W_bar

def Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=True, mask_type=None, stride=1, weightnorm=None, biases=True, gain=1., padding = 'SAME', spectral_normed = False, update_collection=None, bias_init = 0.0):
    """
    inputs: tensor of shape (batch size, num channels, height, width)
    mask_type: one of None, 'a', 'b'

    returns: tensor of shape (batch size, num channels, height, width)
    """
    with tf.name_scope(name) as scope:
        #### This part is added for SN ###
        #if scope_has_variables(scope):
        #    scope.reuse_variables()
        #### This part is added for SN ###
            
        if mask_type is not None:
            mask_type, mask_n_channels = mask_type

            mask = np.ones(
                (filter_size, filter_size, input_dim, output_dim), 
                dtype='float32'
            )
            center = filter_size // 2

            # Mask out future locations
            # filter shape is (height, width, input channels, output channels)
            mask[center+1:, :, :, :] = 0.
            mask[center, center+1:, :, :] = 0.

            # Mask out future channels
            for i in xrange(mask_n_channels):
                for j in xrange(mask_n_channels):
                    if (mask_type=='a' and i >= j) or (mask_type=='b' and i > j):
                        mask[
                            center,
                            center,
                            i::mask_n_channels,
                            j::mask_n_channels
                        ] = 0.


        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        fan_in = input_dim * filter_size**2
        fan_out = output_dim * filter_size**2 / (stride**2)

        if mask_type is not None: # only approximately correct
            fan_in /= 2.
            fan_out /= 2.

        if he_init:
            filters_stdev = np.sqrt(4./(fan_in+fan_out))
        else: # Normalized init (Glorot & Bengio)
            filters_stdev = np.sqrt(2./(fan_in+fan_out))

        if _weights_stdev is not None:
            filter_values = uniform(
                _weights_stdev,
                (filter_size, filter_size, input_dim, output_dim)
            )
        else:
            filter_values = uniform(
                filters_stdev,
                (filter_size, filter_size, input_dim, output_dim)
            )

        # print "WARNING IGNORING GAIN"
        filter_values *= gain

        filters = lib.param(name+'.Filters', filter_values)

        if weightnorm==None:
            weightnorm = _default_weightnorm
        if weightnorm:
            norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0,1,2)))
            target_norms = lib.param(
                name + '.g',
                norm_values
            )
            with tf.name_scope('weightnorm') as scope:
                norms = tf.sqrt(tf.reduce_sum(tf.square(filters), reduction_indices=[0,1,2]))
                filters = filters * (target_norms / norms)

        if mask_type is not None:
            with tf.name_scope('filter_mask'):
                filters = filters * mask
        if spectral_normed:
            norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0,1,2,3)))
            target_norms = lib.param(
                name + '.sng',
                norm_values
            )
            """
            result = tf.nn.conv2d(
                input=inputs, 
                filter=target_norms*spectral_normed_weight(name, filters, num_iters=1, update_collection=update_collection), 
                strides=[1, 1, stride, stride],
                padding=padding,
                data_format='NCHW'
            )
            """
            result = tf.nn.conv2d(
                input=inputs, 
                filter=spectral_normed_weight(name, filters, num_iters=7, update_collection=tf.GraphKeys.UPDATE_OPS),
                strides=[1, 1, stride, stride],
                padding=padding,
                data_format='NCHW'
            )
            #spectrally_normed_weight(name, filters, update_collection=tf.GraphKeys.UPDATE_OPS),

        else:
            result = tf.nn.conv2d(
                input=inputs, 
                filter=filters, 
                strides=[1, 1, stride, stride],
                padding=padding,
                data_format='NCHW'
            )
        
        if biases:
            _biases = lib.param(
                name+'.Biases',
                np.ones(output_dim, dtype='float32') * bias_init
            )

            result = tf.nn.bias_add(result, _biases, data_format='NCHW')


        return result
