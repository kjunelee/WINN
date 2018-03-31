# utils.py
# This file contains utility functions, including directory creation and traverse, image generating, reading, saving, merging, normalizing, unnormalizing, etc.

import os
import scipy
import numpy as np

def log(log_file_path, string):
    '''
    Write one line of log into screen and file.
        log_file_path: Path of log file.
        string:        String to write in log file.
    '''
    with open(log_file_path, 'a+') as f:
        f.write(string + '\n')
        f.flush()
    print(string)
    
def mkdir_if_not_exists(path):
    '''
    Create directory if it does not exist.
        path:           Path of directory.
    '''
    if not os.path.exists(path):
        os.mkdir(path)
        
def get_images_path_in_directory(path):
    '''
    Get path of all images recursively in directory filtered by extension list.
        path: Path of directory contains images.
    Return path of images in selected directory.
    '''
    images_path_in_directory = []
    image_extensions = ['.png', '.jpg']
    
    for root_path, directory_names, file_names in os.walk(path):
        for file_name in file_names:
            lower_file_name = file_name.lower()
            if any(map(lambda image_extension: 
                       lower_file_name.endswith(image_extension), 
                       image_extensions)):
                images_path_in_directory.append(os.path.join(root_path, file_name))

    return images_path_in_directory

def save_unnormalized_image(image, path):
    '''
    Save one image.
        image: Unnormalized images array. The count of images 
               should match the size and the intensity values range
               from 0 to 255. Format: [height, width, channels]
        path:  Path of merged image.
    '''
    # Attention: Here we should not use the following way to save image.
    #     scipy.misc.imsave(path, image)
    # Because it automatically scale the intensity value in image
    # from [min(image), max(image)] to [0, 255]. It should be
    # the reason behind the issue reported by Kwonjoon Lee, which states 
    # the intensity value in demo in ICL/IGM paper is much near 0 or 255.
    scipy.misc.toimage(arr = image, cmin = 0, cmax = 255).save(path)
    
def save_unnormalized_images(images, size, path):
    '''
    Merge multiple unnormalized images into one and save it.
        images: Unnormalized images array. The count of images 
                should match the size and the intensity values range
                from 0 to 255. Format: [count, height, width, channels]
        size:   Number of images to merge. 
                Format: (vertical_count, horizontal_count).
        path:   Path of merged image.
    '''
    merged_image = merge(images, size)
    # Attention: Here we should not use the following way to save image.
    #     scipy.misc.imsave(path, merged_image)
    # Because it automatically scale the intensity value in merged_image
    # from [min(merged_image), max(merged_image)] to [0, 255]. It should be
    # the reason behind the issue reported by Kwonjoon Lee, which states 
    # the intensity value in demo in ICL/IGM paper is much near 0 or 255.
    scipy.misc.toimage(arr = merged_image, cmin = 0, cmax = 255).save(path)
    
def load_unnormalized_image(path):
    '''
    Load a RGB image and do not normalize. Each intensity value is from 
    0 to 255 and then it is converted into 32-bit float.
        path: Path of image file.
    Return image array.
    '''
    return scipy.misc.imread(path, mode = 'RGB').astype(np.float32)

def merge(images, size):
    '''
    Merge several images into one.
        size: Number of images to merge. 
              Format: (vertical_count, horizontal_count)
    Return merged image array.
    '''
    count, height, width, channels = images.shape
    vertical_count, horizontal_count = size
    if not (vertical_count * horizontal_count == count):
        raise ValueError("Count of images does not match size.")
        
    # Merged image looks like
    #     [ ][ ][ ]
    #     [ ][ ][ ]
    #     [ ][ ][ ]
    # when size = [3, 3].
    merged_image = np.zeros((height * vertical_count, 
                             width * horizontal_count, 
                             channels))
    for i, image in enumerate(images):
        m = i // vertical_count
        n = i % vertical_count
        merged_image[m * height : (m + 1) * height, 
                     n * width : (n + 1) * width, :] = image
    return merged_image

def normalize(images):
    '''
    Normalize the intensity values from [0, 255] into [-1, 1].
        images: Image array to normalize. Require each intensity value
                ranges from 0 to 255.
    Return normalized image array.
    '''
    return 1.0 * np.array(images) / 255 * 2.0 - 1.0

def unnormalize(images):
    '''
    Unnormalize the intensity values from [-1, 1] to [0, 255].
        images: Image array to unnormalize. Require each intensity value 
                ranges from -1 to 1.
    Return unnormalized image array.
    '''
    return (images + 1.0) / 2.0 * 255

def gen_unnormalized_random_images(image_shape, count):
    '''
    Generate unnormalized image with random intensity values. Each intensity
    value ranges from 0 to 255.
        image_shape: Shape of an image. Format: [height, width, channels]
        count:       Number of random images to generate.
    Return array of generated random images.
    '''
    height, width, channels = image_shape
    intermediate_images = np.random.normal(loc = 0, scale = 0.3, 
                          size = [count, height, width, channels])
    intermediate_images = intermediate_images - intermediate_images.min()
    return intermediate_images / intermediate_images.max() * 255.0 

def gen_unnormalized_uniform_random_images(image_shape, count):
    '''
    Generate unnormalized image with random intensity values. Each intensity
    value ranges from 0 to 255.
        image_shape: Shape of an image. Format: [height, width, channels]
        count:       Number of random images to generate.
    Return array of generated random images.
    '''
    height, width, channels = image_shape
    intermediate_images = np.random.uniform(low=-1.0, high=1.0,
                          size = [count, height, width, channels])
    intermediate_images = intermediate_images - intermediate_images.min()
    return intermediate_images / intermediate_images.max() * 255.0 

def get_image_shape(path):
    '''
    Get shape of image. Format: [height, width, channels]. In fact, all images
    are regarded as color images, thus, channels is always 3.
        path: Path of image file.
    Return array of image shape.
    '''
    image = scipy.misc.imread(path)
    [height, width, channels] = image.shape
    return [height, width, channels]