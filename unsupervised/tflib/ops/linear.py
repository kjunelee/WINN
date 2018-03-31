import tflib as lib

import numpy as np
import tensorflow as tf
from tflib.ops.sn import spectral_normed_weight

_default_weightnorm = False
def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

def disable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = False

_weights_stdev = None
def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev

def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None

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

def Linear(
        name, 
        input_dim, 
        output_dim, 
        inputs,
        biases=True,
        initialization=None,
        weightnorm=None,
        gain=1., spectral_normed = False
    ):
    """
    initialization: None, `lecun`, 'glorot', `he`, 'glorot_he', `orthogonal`, `("uniform", range)`
    """
    with tf.name_scope(name) as scope:

        def uniform(stdev, size):
            if _weights_stdev is not None:
                stdev = _weights_stdev
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        if initialization == 'lecun':# and input_dim != output_dim):
            # disabling orth. init for now because it's too slow
            weight_values = uniform(
                np.sqrt(1./input_dim),
                (input_dim, output_dim)
            )

        elif initialization == 'glorot' or (initialization == None):

            weight_values = uniform(
                np.sqrt(2./(input_dim+output_dim)),
                (input_dim, output_dim)
            )

        elif initialization == 'he':

            weight_values = uniform(
                np.sqrt(2./input_dim),
                (input_dim, output_dim)
            )

        elif initialization == 'glorot_he':

            weight_values = uniform(
                np.sqrt(4./(input_dim+output_dim)),
                (input_dim, output_dim)
            )

        elif initialization == 'orthogonal' or \
            (initialization == None and input_dim == output_dim):
            
            # From lasagne
            def sample(shape):
                if len(shape) < 2:
                    raise RuntimeError("Only shapes of length 2 or more are "
                                       "supported.")
                flat_shape = (shape[0], np.prod(shape[1:]))
                 # TODO: why normal and not uniform?
                a = np.random.normal(0.0, 1.0, flat_shape)
                u, _, v = np.linalg.svd(a, full_matrices=False)
                # pick the one with the correct shape
                q = u if u.shape == flat_shape else v
                q = q.reshape(shape)
                return q.astype('float32')
            weight_values = sample((input_dim, output_dim))
        
        elif initialization[0] == 'uniform':
        
            weight_values = np.random.uniform(
                low=-initialization[1],
                high=initialization[1],
                size=(input_dim, output_dim)
            ).astype('float32')

        else:

            raise Exception('Invalid initialization!')

        weight_values *= gain

        weight = lib.param(
            name + '.W',
            weight_values
        )

        if weightnorm==None:
            weightnorm = _default_weightnorm
        if weightnorm:
            norm_values = np.sqrt(np.sum(np.square(weight_values), axis=0))
            # norm_values = np.linalg.norm(weight_values, axis=0)

            target_norms = lib.param(
                name + '.g',
                norm_values
            )

            with tf.name_scope('weightnorm') as scope:
                norms = tf.sqrt(tf.reduce_sum(tf.square(weight), reduction_indices=[0]))
                weight = weight * (target_norms / norms)

        # if 'Discriminator' in name:
        #     print "WARNING weight constraint on {}".format(name)
        #     weight = tf.nn.softsign(10.*weight)*.1

        if inputs.get_shape().ndims == 2:
            if spectral_normed:
                norm_values = np.sqrt(np.sum(np.square(weight_values), axis=(0,1)))
                # norm_values = np.linalg.norm(weight_values, axis=0)

                target_norms = lib.param(
                    name + '.sng',
                    norm_values
                )

                result = tf.matmul(inputs, spectral_normed_weight(name, weight, num_iters=7, update_collection=tf.GraphKeys.UPDATE_OPS))
                
            else:
                result = tf.matmul(inputs, weight)
        else:
            reshaped_inputs = tf.reshape(inputs, [-1, input_dim])
            if spectral_normed:
                norm_values = np.sqrt(np.sum(np.square(weight_values), axis=(0,1)))
                # norm_values = np.linalg.norm(weight_values, axis=0)

                target_norms = lib.param(
                    name + '.sng',
                    norm_values
                )
                result = tf.matmul(inputs, spectral_normed_weight(name, weight, num_iters=7, update_collection=tf.GraphKeys.UPDATE_OPS))
            else:
                result = tf.matmul(reshaped_inputs, weight)
            result = tf.reshape(result, tf.pack(tf.unpack(tf.shape(inputs))[:-1] + [output_dim]))

        if biases:
            result = tf.nn.bias_add(
                result,
                lib.param(
                    name + '.b',
                    np.zeros((output_dim,), dtype='float32')
                )
            )

        return result