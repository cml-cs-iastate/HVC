import os
import numpy as np
import pickle as pkl
import tensorflow as tf
from collections import defaultdict

from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models


def closest_patch_to_vc(batch_size, num_vc, distances):
    """
    return the (x, y) index of the closest patch for each vc
    :param batch_size: the batch size
    :param num_vc: the number of vc
    :param distances: distances of the vc with every patch  # shape [batch_size, height, width, num_vc]
    """
    distances_transpose = tf.transpose(distances, perm=[0, 3, 1, 2])
    output_1 = tf.cast(tf.argmin(distances_transpose, axis=-1), dtype=tf.int32)
    output_2 = tf.cast(tf.argmin(tf.reduce_min(distances_transpose, axis=-1), axis=-1), dtype=tf.int32)
    range_vc = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(num_vc), axis=0), [batch_size, 1]), axis=2)
    range_batch = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, num_vc]), axis=2)
    m = tf.concat([range_batch, tf.concat([range_vc, tf.expand_dims(output_2, axis=2)], axis=2)], axis=2)
    o = tf.gather_nd(indices=m, params=output_1)
    indices = tf.concat([tf.expand_dims(output_2, axis=2), tf.expand_dims(o, axis=2)], axis=2)
    return indices


def vc_heat_map_generating(input_tensor, distances, num_vc, p_h, p_w):
    """
    Generate a heat map for the closest patch of the convolutional output that correspond to every vc.
    :param input_tensor: TensorFlow tensor of input samples         # shape [batch_size, height, width, depth]
    :param distances: distances of the vc with every patch          # shape [batch_size, height, width, num_vc]
    :param num_vc: int the number of vc
    """
    batch_size, height, width, depth = input_tensor.shape
    # The indices of the closest patch to the vc for each image in the batch
    indices = closest_patch_to_vc(batch_size, num_vc, distances)
    # Generate the heat-map of the convolutional layer before the VCL
    range_vc = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(num_vc), axis=0), [batch_size, 1]), axis=2)
    range_batch = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, num_vc]), axis=2)
    m = tf.expand_dims(tf.concat([range_batch, tf.concat([range_vc, indices], axis=2)], axis=2), axis=2)
    ones = tf.ones(shape=(batch_size, num_vc, 1), dtype=tf.int32)
    heat_map = tf.scatter_nd(indices=m, updates=ones, shape=[batch_size, num_vc, height, width])

    if p_h == 2 and p_w == 2:
        vc_num = heat_map.shape[1]
        a = tf.concat([tf.zeros([batch_size, vc_num, height, 1], dtype=tf.int32), heat_map], axis=3)[:, :, :, :-1]
        b = tf.concat([tf.zeros([batch_size, vc_num, 1, width], dtype=tf.int32), heat_map], axis=2)[:, :, :-1, :]
        c = tf.concat([tf.zeros([batch_size, vc_num, 1, width], dtype=tf.int32), a], axis=2)[:, :, :-1, :]
        heat_map = heat_map + a + b + c

    closest_distance = tf.reshape(tf.reduce_min(distances, axis=[1, 2]), [-1, num_vc])

    return heat_map, closest_distance


def vc_interpretation(input_images, heat_map):
    """
    return the part of the image that related to the vc
    :param input_images: Numpy array of the image                       # shape = [batch_size, img_height, img_width, depth]
    :param heat_map: Numpy array of convolutional output heat-map       # shape = [batch_size, num_vc, height, width]
    """

    input_images_shape = np.shape(input_images)
    img_batch_size, img_height, img_width, img_depth = input_images_shape[0], input_images_shape[1], input_images_shape[2], input_images_shape[3]
    heat_map_shape = np.shape(heat_map)
    batch_size, num_vc, height, width = heat_map_shape[0], heat_map_shape[1], heat_map_shape[2], heat_map_shape[3]
    # Upsample the heat-map to the image size
    model = models.Sequential()
    model.add(layers.UpSampling2D(size=(img_height//height, img_width//width), input_shape=[height, width, 1], interpolation='nearest'))
    # Combine the the batch_size and the num_vc before upsampling and separate after upsampling
    heat_map = np.expand_dims(heat_map, axis=4)
    upsampled = list()
    for i in range(batch_size):
        output_upsampled = model.predict(heat_map[i, :, :, :, :])
        upsampled.append(np.expand_dims(np.squeeze(output_upsampled, axis=-1), axis=0))

    upsampled = np.concatenate(upsampled, axis=0)

    if img_height - np.shape(upsampled)[2] != 0:
        ones = np.ones(shape=(batch_size, num_vc, img_height - np.shape(upsampled)[2], np.shape(upsampled)[3]))
        upsampled = np.concatenate([upsampled, ones], axis=2)
    if img_width - np.shape(upsampled)[3] != 0:
        ones = np.ones(shape=(batch_size, num_vc, np.shape(upsampled)[2], img_width - np.shape(upsampled)[3]))
        upsampled = np.concatenate([upsampled, ones], axis=3)

    hvc_heatmap = np.tile(np.expand_dims(upsampled, axis=4), [1, 1, 1, 1, 3])
    input_images_broadcast = np.tile(np.expand_dims(input_images, axis=1), [1, num_vc, 1, 1, 1])

    # Project the heat-map to the input images
    closest_patches = np.multiply(input_images_broadcast, hvc_heatmap)
    return closest_patches, hvc_heatmap


def store_memberships_dict(image_file_names, sub_model_num, batch_size, num_classes, num_vc, membership_distance, membership_index, sub_model_directory, membership_dict_name):
    """
    store the membership dictionary between the visual concepts of each layers from layer 5 to layer 1
    :param image_file_names: a list with the image names file in the batch
    :param sub_model_num: the number of the sub_model being used
    :param batch_size:
    :param num_classes:
    :param num_vc:
    :param membership_distance: the membership distance from the current visual concept to the next layer visual concept
    :param membership_index: the membership index from the current visual concept to the next layer visual concept
    :param sub_model_directory:
    :param membership_dict_name:
    """
    if os.path.exists(membership_dict_name):
        with open(membership_dict_name, 'rb') as f:
            membership_dict = pkl.load(f)
    else:
        membership_dict = dict()

    for i in range(batch_size):
        # image name as the key of the top dict()
        membership_dict.setdefault(image_file_names[i], dict())
        # sub_model number as the key of the sub_dict()
        membership_dict[image_file_names[i]].setdefault(sub_model_num, list())
        for j in range(num_vc // num_classes):
            membership_dict[image_file_names[i]][sub_model_num].append((j, int(membership_index[i][j]), membership_distance[i][j]))

    # save the membership_dict for further use
    with open(membership_dict_name, 'wb') as f:
        pkl.dump(membership_dict, f, protocol=pkl.HIGHEST_PROTOCOL)


def save_visualize_patches(input_y, input_y_predictions, image_file_names, closest_patches, closest_distance, sub_model_num, coordinates, save_img_patches=False):
    """
    visualize the results of the heat-map images
    :param input_y: the true labels of the input images                         # shape = [batch_size, num_classes]
    :param input_y_predictions: the predicted labels                            # shape = [batch_size, num_classes]
    :param image_file_names: list of files name in the patch                    # shape = [batch_size]
    :param closest_patches: Numpy array of the interpreted images               # shape = [batch_size, num_vc, img_height, img_width, img_depth]
    :param closest_distance: numpy array of similarity scores                   # shape = [batch_size, num_vc]
    :param sub_model_num: the number of the sub_model being run
    :param coordinates: the x, y indices of the closest patch to each vc        # shape [batch_size, num_vc, 4]
    :param save_img_patches: if True, save images patches to the disk
    """
    input_images_shape = np.shape(closest_patches)
    batch_size, num_vc = input_images_shape[0], input_images_shape[1]

    sub_model_directory = f'sub_model_{sub_model_num}'

    # create output directory to store the patches to the disk
    if not os.path.exists(sub_model_directory):
        os.makedirs(sub_model_directory)

    # list to store the top 10 patches per class at each layer
    top_patches_per_class_file = sub_model_directory + f'/top_patches_per_class_{sub_model_num}.pkl'
    if os.path.exists(top_patches_per_class_file):
        with open(top_patches_per_class_file, 'rb') as f:
            top_patches_per_class = pkl.load(f)
    else:
        top_patches_per_class = defaultdict(list)

    # list to store the top patch per vc at each layer
    top_patch_per_vc_file = sub_model_directory + f'/top_patches_per_vc_{sub_model_num}.pkl'
    if os.path.exists(top_patch_per_vc_file):
        with open(top_patch_per_vc_file, 'rb') as f:
            top_patches_per_vc = pkl.load(f)
    else:
        top_patches_per_vc = dict()

    # iterate for each image
    for image_idx in range(batch_size):

        t_label = np.argmax(input_y[image_idx])
        p_label = np.argmax(input_y_predictions[image_idx])

        image_file_name = image_file_names[image_idx]

        class_directory = sub_model_directory + f'/class_{t_label}'
        if not os.path.exists(class_directory):
            os.makedirs(class_directory)

        # iterate for each vc
        for vc_idx in range(num_vc):

            # store the interpretation of the correctly classified images
            if t_label == p_label:

                # store only the vc of the target class
                if vc_idx // 10 == t_label:

                    image_name = f'{image_file_name}_vc_{vc_idx % 10}_distance{closest_distance[image_idx][vc_idx]}.jpg'
                    x1, y1, x2, y2 = coordinates[image_idx, vc_idx, :]

                    # save the patch to the disk
                    if save_img_patches:
                        image.save_img(path=class_directory + '/' + image_name, x=closest_patches[image_idx, vc_idx, x1:x2, y1:y2, :])

                    # Append the patch file name with the distance score to the top_patches_per_class dict()
                    top_patches_per_class[t_label].append((image_name, closest_distance[image_idx][vc_idx], image_file_names[image_idx], x1, y1, x2, y2))

                    # Append the patch file name with the distance score to the top_patches_per_vc dict()
                    top_patches_per_vc.setdefault(vc_idx, dict())
                    top_patches_per_vc[vc_idx].setdefault(t_label, list())
                    top_patches_per_vc[vc_idx][t_label].append((image_name, closest_distance[image_idx][vc_idx], image_file_names[image_idx], x1, y1, x2, y2))

    # Get the closest 10 patches to each class at every layer
    temp_top_patches_per_class = defaultdict(list)
    for k, v in top_patches_per_class.items():
        temp_top_patches_per_class[k] = sorted(v, key=lambda x: x[1])[:10]

    # Get the closest 10 patches for each vc at each class for every layer
    temp_top_patches_per_vc = dict()
    for k, v in top_patches_per_vc.items():
        for kk, vv in v.items():
            temp_top_patches_per_vc.setdefault(k, dict())
            temp_top_patches_per_vc[k].setdefault(kk, list())
            temp_top_patches_per_vc[k][kk] = sorted(vv, key=lambda x: x[1])[:10]

    # save the temp_top_patches_per_class dict for further use
    with open(top_patches_per_class_file, 'wb') as f:
        pkl.dump(temp_top_patches_per_class, f, protocol=pkl.HIGHEST_PROTOCOL)

    # save the temp_top_patches_per_vc dict for further use
    with open(top_patch_per_vc_file, 'wb') as f:
        pkl.dump(temp_top_patches_per_vc, f, protocol=pkl.HIGHEST_PROTOCOL)


def save_patches(input_y, input_y_predictions, image_file_names, closest_patches, closest_distance, sub_model_num, coordinates):
    """
    visualize the results of the heat-map images
    :param input_y: the true labels of the input images                         # shape = [batch_size, num_classes]
    :param input_y_predictions: the predicted labels                            # shape = [batch_size, num_classes]
    :param image_file_names: list of files name in the patch                    # shape = [batch_size]
    :param closest_patches: Numpy array of the interpreted images               # shape = [batch_size, num_vc, img_height, img_width, img_depth]
    :param closest_distance: numpy array of similarity scores                   # shape = [batch_size, num_vc]
    :param sub_model_num: the number of the sub_model being run
    :param coordinates: the x, y indices of the closest patch to each vc        # shape [batch_size, num_vc, 4]
    """
    input_images_shape = np.shape(closest_patches)
    batch_size, num_vc = input_images_shape[0], input_images_shape[1]

    sub_model_directory = f'sub_model_{sub_model_num}'

    # create output directory to store the patches to the disk
    if not os.path.exists(sub_model_directory):
        os.makedirs(sub_model_directory)

    # iterate for each image
    for image_idx in range(batch_size):

        t_label = np.argmax(input_y[image_idx])
        p_label = np.argmax(input_y_predictions[image_idx])

        image_file_name = image_file_names[image_idx]

        class_directory = sub_model_directory + f'/class_{t_label}'
        if not os.path.exists(class_directory):
            os.makedirs(class_directory)

        # iterate for each vc
        for vc_idx in range(num_vc):

            # store the interpretation of the correctly classified images
            if t_label == p_label:

                # store only the vc of the target class
                if vc_idx // 10 == t_label:

                    image_name = f'{image_file_name}_vc_{vc_idx % 10}_distance{closest_distance[image_idx][vc_idx]}.jpg'
                    x1, y1, x2, y2 = coordinates[image_idx, vc_idx, :]

                    # save the patch to the disk
                    fname = class_directory + '/' + image_name
                    image.save_img(path=fname, x=closest_patches[image_idx, vc_idx, x1:x2, y1:y2, :])