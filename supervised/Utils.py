import os
import gzip
import numpy as np
import tensorflow as tf
import scipy.misc

def log(log_file_path, line):
    '''
    Write one line of log into screen and file.
        log_file_path: Path of log file.
        string:        String to write in log file.
    '''
    with open(log_file_path, 'a+') as f:
        f.write(line + '\n')
        f.flush()
    print(line)    

def read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def download_mnist(mnist_dir = './mnist'):
    if not os.path.exists(mnist_dir):
        os.mkdir(mnist_dir)
        import urllib
        urllib.urlretrieve('http://cseweb.ucsd.edu/~weijian/static/datasets/mnist/t10k-images-idx3-ubyte.gz', 
                           './mnist/t10k-images-idx3-ubyte.gz')
        urllib.urlretrieve('http://cseweb.ucsd.edu/~weijian/static/datasets/mnist/t10k-labels-idx1-ubyte.gz', 
                           './mnist/t10k-labels-idx1-ubyte.gz')
        urllib.urlretrieve('http://cseweb.ucsd.edu/~weijian/static/datasets/mnist/train-images-idx3-ubyte.gz', 
                           './mnist/train-images-idx3-ubyte.gz')
        urllib.urlretrieve('http://cseweb.ucsd.edu/~weijian/static/datasets/mnist/train-labels-idx1-ubyte.gz', 
                           './mnist/train-labels-idx1-ubyte.gz')
    
def load_train_data():
    mnist_dir = './mnist'
    download_mnist(mnist_dir)
    train_image_path = os.path.join(mnist_dir, 'train-images-idx3-ubyte.gz')
    train_label_path = os.path.join(mnist_dir, 'train-labels-idx1-ubyte.gz')

    with gzip.open(train_image_path) as image_stream, gzip.open(train_label_path) as label_stream:
        magic_image, magic_label = read32(image_stream), read32(label_stream)
        if magic_image != 2051 or magic_label != 2049:
            raise ValueError('Invalid magic number')

        image_count, label_count = read32(image_stream), read32(label_stream)
        row_count = read32(image_stream)
        col_count = read32(image_stream)

        label_buffer = label_stream.read(label_count)
        train_labels = np.frombuffer(label_buffer, dtype=np.uint8)

        image_buffer = image_stream.read(row_count * col_count * image_count)
        train_images = np.frombuffer(image_buffer, dtype=np.uint8)
        train_images = train_images.reshape(image_count, row_count, col_count, 1)

        return train_images, train_labels.astype(np.int32)


def load_test_data():
    mnist_dir = './mnist'
    download_mnist(mnist_dir)
    test_image_path = os.path.join(mnist_dir, 't10k-images-idx3-ubyte.gz')
    test_label_path = os.path.join(mnist_dir, 't10k-labels-idx1-ubyte.gz')

    with gzip.open(test_image_path) as image_stream, gzip.open(test_label_path) as label_stream:
        magic_image, magic_label = read32(image_stream), read32(label_stream)
        if magic_image != 2051 or magic_label != 2049:
            raise ValueError('Invalid magic number')

        image_count, label_count = read32(image_stream), read32(label_stream)
        row_count = read32(image_stream)
        col_count = read32(image_stream)

        label_buffer = label_stream.read(label_count)
        test_labels = np.frombuffer(label_buffer, dtype=np.uint8)

        image_buffer = image_stream.read(row_count * col_count * image_count)
        test_images = np.frombuffer(image_buffer, dtype=np.uint8)
        test_images = test_images.reshape(image_count, row_count, col_count, 1)

        return test_images, test_labels.astype(np.int32)

def extract_data(train_raw_images, train_raw_labels, count):
    # FIXME: Currently there is no exception handler for insufficient images.
    train_images = []
    train_labels = []
    #shuffle = np.random.permutation(train_raw_images.shape[0])    # Enable shuffling.
    shuffle = np.arange(train_raw_images.shape[0])                 # Disable shuffling.
    each_cat_count = count // 10
    cat_dict = {}
    for i in xrange(10):
        cat_dict[i] = 0
    idx = 0
    while (True):
        if cat_dict[train_raw_labels[shuffle[idx]]] < each_cat_count:
            cat_dict[train_raw_labels[shuffle[idx]]] += 1
            train_images.append(train_raw_images[shuffle[idx]])
            train_labels.append(train_raw_labels[shuffle[idx]])
        idx += 1
        if len(train_images) == count:
            break
    return np.array(train_images), np.array(train_labels)
        
# Normalize images
def normalize(images):
    '''
    Normalize the intensity values from [0, 255] into [-1, 1].
        images: Image array to normalize. Require each intensity value
                ranges from 0 to 255.
    Return normalized image array.
    '''
    return 1.0 * np.array(images) / 255 * 2.0 - 1.0

# Unnormalize images
def unnormalize(images):
    '''
    Unnormalize the intensity values from [-1, 1] to [0, 255].
        images: Image array to unnormalize. Require each intensity value 
                ranges from -1 to 1.
    Return unnormalized image array.
    '''
    return (images + 1.0) / 2.0 * 255

def save_batch_images_to_path(image_shape, images, images_path):
    height, width, channels = image_shape
    side_len = int(np.sqrt(images.shape[0]))
    merged_image = np.zeros((side_len * height, side_len * width))
    
    for i, image in enumerate(images):
        m = i // side_len
        n = i % side_len
        merged_image[m * height : (m + 1) * height, 
                     n * width : (n + 1) * width] = image
    
    scipy.misc.imsave(name=images_path, arr=merged_image)

def conv2d(input_, output_dim, 
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev),
                            regularizer=tf.contrib.layers.l2_regularizer(0.0002))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev),
                                 regularizer=tf.contrib.layers.l2_regularizer(0.0002))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
        
def swish(x, leak=0.2):
    return x * tf.sigmoid(x)

def fully_connected(scope, input_layer, output_dim):
    input_dim = input_layer.get_shape().as_list()[-1]
    
    with tf.variable_scope(scope):
        fc_weight = tf.get_variable(
            'fc_weight',
            shape = [input_dim, output_dim],
            dtype = tf.float32,
            initializer = tf.contrib.layers.variance_scaling_initializer(),
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.0002)            
        )

        fc_bias = tf.get_variable(
            'fc_bias',
            shape = [output_dim],
            dtype = tf.float32,
            initializer = tf.constant_initializer(0.0)
        )

        output_layer = tf.matmul(input_layer, fc_weight) + fc_bias

        return output_layer

def avg_pool(scope, input_layer, ksize=None, strides=[1, 2, 2, 1]):
    if ksize is None:
        ksize = strides

    with tf.variable_scope(scope):
        output_layer = tf.nn.avg_pool(input_layer, ksize, strides, 'VALID')
        return output_layer
