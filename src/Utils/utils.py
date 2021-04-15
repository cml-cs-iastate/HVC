import os
import numpy as np
import tensorflow as tf


def euclidean_distance(tensor, p, squared=False, axis_=-2):
    if squared:
        return tf.reduce_sum(tf.square(tensor - p), axis=axis_)
    return tf.sqrt(tf.reduce_sum(tf.square(tensor - p), axis=axis_))


def distance_to_similarity(distances):
    loss = 1/(1 + distances)
    return loss


def log10(t):
    numerator = tf.math.log(t)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def custom_weights_init(weights_shape):
    if weights_shape[0] <= weights_shape[1]:
        print('ERROR: the HVC layer has more neurones that the output layer!')
    weights = np.full(weights_shape, -0.1)
    vc_per_class = weights_shape[0] // weights_shape[1]
    for idx in range(weights_shape[0]):
        if idx // vc_per_class < weights_shape[1]:
            idx_one = idx // vc_per_class
        else:
            idx_one = weights_shape[1] - 1
        weights[idx, idx_one] = 1
    output = tf.constant_initializer(weights)
    return output


def get_layer_weights(model, layer_name):
    return model.get_layer(layer_name).get_weights()


def set_layer_weights(model, layer_name, weights):
    model.get_layer(layer_name).set_weights(weights)


def get_layer_names(model):
    layer_names = list()
    for i, layer in enumerate(model.layers):
        layer_names.append(layer.name)
    return layer_names


def get_ordered_file_names(generator):
    current_index = ((generator.batch_index-1) * generator.batch_size)
    if current_index < 0:
        if generator.samples % generator.batch_size > 0:
            current_index = max(0, generator.samples - generator.samples % generator.batch_size)
        else:
            current_index = max(0, generator.samples - generator.batch_size)
    index_array = generator.index_array[current_index:current_index + generator.batch_size].tolist()
    image_file_names = [generator.filepaths[idx] for idx in index_array]
    return image_file_names


def combine_heatmaps(input_y, hvc_heatmap, vc_per_class):
    label = np.argmax(input_y)
    start = label * vc_per_class
    end = (label + 1) * vc_per_class
    hvc_heatmap_combined = np.max(hvc_heatmap[start:end, :, :, :], axis=0)
    return hvc_heatmap_combined


def store_coordinates(VC_coordinates, file_name, coordinates):
    t = dict()
    t['x1_vc'] = coordinates[0]
    t['y1_vc'] = coordinates[1]
    t['x2_vc'] = coordinates[2]
    t['y2_vc'] = coordinates[3]
    VC_coordinates[file_name] = t
    return VC_coordinates


def get_file_name_from_path(path):
    head, tail = os.path.split(path)
    return tail


def get_file_name_and_extension(file_name):
    f_name, extension = os.path.splitext(file_name)
    return f_name, extension
