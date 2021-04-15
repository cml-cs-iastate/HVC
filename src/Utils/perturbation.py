import numpy as np

from src.Utils.utils import combine_heatmaps


def perturb_input(input_x, input_y, hvc_heatmap, vc_per_class, remove_important=False):
    """
    Perturb the input images by zero out the patches of the hvc_heatmap
    :param input_x: Numpy array of original images                              # shape = [img_height, img_width, img_depth]
    :param input_y: the true labels of the input images                         # shape = [num_classes]
    :param hvc_heatmap: Numpy array of the upsampled heat-map                   # shape = [num_vc, img_height, img_width, img_depth]
    :param vc_per_class: number of visual concepts per class
    :param remove_important: if True, remove the important part, False otherwise
    """
    if len(np.shape(hvc_heatmap)) == 4:
        # Get the combined hvc_heatmap
        hvc_heatmap_combined = combine_heatmaps(input_y, hvc_heatmap, vc_per_class)
    elif len(np.shape(hvc_heatmap)) == 2:
        # This one for grad_cam heat map to expand it to image dim
        hvc_heatmap_combined = np.tile(np.expand_dims(hvc_heatmap, axis=-1), [1, 1, 3])
    # Inverse the hvc_heatmap from 0 to 1 and from 1 to 0
    if remove_important:
        hvc_heatmap_combined = np.ones_like(hvc_heatmap_combined) - hvc_heatmap_combined
    input_x_perturb = input_x * hvc_heatmap_combined
    return input_x_perturb
