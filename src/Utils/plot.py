import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_loss_accuracy(history, output_directory, model_acc_file='model_accuracy.png', model_loss_file='model_loss.png'):
    print('History Metrics:', history.keys())
    # summarize history for accuracy
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(output_directory, model_acc_file), dpi=300)
    plt.show()
    plt.clf()
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(output_directory, model_loss_file), dpi=300)
    plt.show()
    plt.clf()


def visualize_vc(input_x, input_y, input_y_predictions, image_file_names, closest_patches, closest_distance, num_classes, vc_per_class, num_vc):
    """
    visualize the results of the heat-map images. This function will present the results for all the class in the dataset
    :param input_x: Numpy array of original images                              # shape = [batch_size, img_height, img_width, img_depth]
    :param input_y: the true labels of the input images                         # shape = [batch_size, num_classes]
    :param input_y_predictions: the predicted labels                            # shape = [batch_size, num_classes]
    :param image_file_names: list of files name in the patch                    # shape = [batch_size]
    :param closest_patches: Numpy array of the interpreted images               # shape = [batch_size, num_vc, img_height, img_width, img_depth]
    :param closest_distance: numpy array of closest distance of vc to a patch   # shape = [batch_size, num_vc]
    :param num_classes:
    :param vc_per_class:
    :param num_vc:
    """
    input_images_shape = np.shape(closest_patches)
    batch_size = input_images_shape[0]
    for image_idx in range(batch_size):
        # Visualize input and output
        fig, axes = plt.subplots(num_classes, vc_per_class + 1)
        fig.subplots_adjust(hspace=0.5)
        axes[0, 0].imshow(input_x[image_idx])
        axes[0, 0].get_xaxis().set_visible(False)
        axes[0, 0].get_yaxis().set_visible(False)
        axes[0, 0].set_title('Original image')
        for i in range(1, num_classes):
            axes[i, 0].get_xaxis().set_visible(False)
            axes[i, 0].get_yaxis().set_visible(False)
        # plot every vc for image sample
        for vc_idx in range(num_vc):
            x_axis = vc_idx // vc_per_class
            y_axis = vc_idx % vc_per_class
            axes[x_axis, y_axis+1].imshow(closest_patches[image_idx, vc_idx, :, :, :])
            axes[x_axis, y_axis+1].get_xaxis().set_visible(False)
            axes[x_axis, y_axis+1].get_yaxis().set_visible(False)
            axes[x_axis, y_axis+1].set_title(closest_distance[image_idx][vc_idx])
        fig.suptitle(f'Original image and Visual Concepts Patches. True label: {np.argmax(input_y[image_idx])} Predicted Label: {np.argmax(input_y_predictions[image_idx])}\n'
                     f'File path: {image_file_names[image_idx]}')
        plt.show()


def visualize_combined_vc(input_x, input_y, input_y_predictions, image_file_names, hvc_heatmap, vc_per_class):
    """
    visualize the results of the heat-map images (combined vc for each image)
    :param input_x: Numpy array of original images                              # shape = [batch_size, img_height, img_width, img_depth]
    :param input_y: the true labels of the input images                         # shape = [batch_size, num_classes]
    :param input_y_predictions: the predicted labels                            # shape = [batch_size, num_classes]
    :param image_file_names: list of files name in the patch                    # shape = [batch_size]
    :param hvc_heatmap: Numpy array of the upsampled heat-map                   # shape = [batch_size, num_vc, img_height, img_width, img_depth]
    :param vc_per_class:
    """
    input_images_shape = np.shape(hvc_heatmap)
    batch_size = input_images_shape[0]
    for image_idx in range(batch_size):
        # Visualize input and output
        fig, axes = plt.subplots(1, 2)
        fig.subplots_adjust(hspace=0.5)
        axes[0].imshow(input_x[image_idx])
        axes[0].get_xaxis().set_visible(False)
        axes[0].get_yaxis().set_visible(False)
        axes[0].set_title('Original image')
        # Get starting and ending index of the VC weights
        label = int(np.argmax(input_y[image_idx]))
        start = label * vc_per_class
        end = (label + 1) * vc_per_class
        # shape [img_height, img_width, img_depth]
        combined_patches_interpretation = np.multiply(input_x[image_idx], np.max(hvc_heatmap[image_idx, start:end, :, :, :], axis=0))
        axes[1].imshow(combined_patches_interpretation)
        axes[1].get_xaxis().set_visible(False)
        axes[1].get_yaxis().set_visible(False)
        axes[1].set_title('combined Visual Concepts interpretation')
        fig.suptitle(f'Original image and Visual Concepts Patches. True label: {np.argmax(input_y[image_idx])} '
                     f'Predicted Label: {np.argmax(input_y_predictions[image_idx])}\n'
                     f'File path: {image_file_names[image_idx]}')
        plt.show()


def visualize_vc_rect(input_x, input_y, input_y_predictions, coordinates, num_vc, vc_per_class, sub_model_num, image_file_names):
    batch_size = np.shape(input_x)[0]
    for image_idx in range(batch_size):
        t_label = np.argmax(input_y[image_idx])
        p_label = np.argmax(input_y_predictions[image_idx])
        # show the interpretation of the correctly classified images
        if t_label == p_label:
            # plot every vc for image sample
            for vc_idx in range(num_vc):
                # show only the vc of the target class
                if vc_idx // vc_per_class == t_label:
                    plt.imshow(input_x[image_idx])
                    # get (x,y) coordinates
                    x1, y1, x2, y2 = coordinates[image_idx, vc_idx, :]
                    gap_x = x2 - x1
                    gap_y = y2 - y1
                    # Get the current reference
                    ax = plt.gca()
                    # Create a rectangle patch
                    # This function reflect the x and y dim
                    rect = patches.Rectangle((y1, x1), gap_y, gap_x, linewidth=8, edgecolor='b', facecolor='none')
                    # Add the patch to the Axes
                    ax.add_patch(rect)
                    # plt.show()
            # create output directory to store the rectangular images to the disk
            sub_model_directory = f'sub_model_{sub_model_num}_rectangular_VC'
            if not os.path.exists(sub_model_directory):
                os.makedirs(sub_model_directory)
            class_directory = sub_model_directory + f'/class_{t_label}'
            if not os.path.exists(class_directory):
                os.makedirs(class_directory)
            image_file_name = image_file_names[image_idx]
            image_name = f'{image_file_name}_vc_{vc_idx}.jpg'
            fname = class_directory + '/' + image_name
            plt.savefig(fname)
            plt.clf()
