from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from keract import keract
from matplotlib import pyplot as plt

from models.constrained_layer import Constrained3DKernelMinimal
# from models.mobile_net import MobileNetBase


def get_patch_location_mask(img_data, max_std_dev, min_std_dev, patch_dimensions, patches_type):
    """
    This method extracts the upto specified number of patches per image. Note that this method can return 0 patches
    if the homogeneity criteria is not met. We extract non-overlapping patches with strides same as patch sizes.
    :param patches_type:
    :param img_data: a numpy image
    :param min_std_dev: 1x3 numpy array, per channel threshold. Any patch with threshold lesser than the
    min_std_threshold will be rejected.
    :param max_std_dev: 1x3 numpy array, per channel threshold. Any patch with threshold greater than the
    max_std_threshold will be rejected.
    :param patch_dimensions: The size of the patch to extract, for example (128, 128)
    :return: array of extracted patches, and an empty list if no patches matched the homogeneity criteria
    """
    homogeneous_patches = []
    non_homogeneous_patches = []

    patch = namedtuple('WindowSize', ['width', 'height'])(*patch_dimensions)
    stride = namedtuple('Strides', ['width_step', 'height_step'])(1, 1)
    image = namedtuple('ImageSize', ['width', 'height'])(img_data.shape[1], img_data.shape[0])
    num_channels = 3

    # Choose the patches
    for row_idx in range(patch.height, image.height, stride.height_step):
        for col_idx in range(patch.width, image.width, stride.width_step):
            img_patch = img_data[(row_idx - patch.height):row_idx, (col_idx - patch.width):col_idx]
            std_dev = np.std(img_patch.reshape(-1, num_channels), axis=0)
            if np.prod(np.less_equal(std_dev, max_std_dev)) and \
                    np.prod(np.greater_equal(std_dev, min_std_dev)):
                homogeneous_patches.append((std_dev, img_patch, row_idx, col_idx))
            else:
                non_homogeneous_patches.append((std_dev, img_patch, row_idx, col_idx))

    if patches_type == 'homogeneous':
        zero_mask = np.zeros(img_data.shape[:2])
        for item in homogeneous_patches:
            zero_mask[item[2], item[3]] = 1
    elif patches_type == 'non_homogeneous':
        zero_mask = np.zeros(img_data.shape[:2])
        for item in non_homogeneous_patches:
            zero_mask[item[2], item[3]] = 1
    else:
        raise ValueError(f'Invalid option for `patches_type`: {str(patches_type)}')

    return zero_mask


def load_model(model_path):
    # noinspection PyProtectedMember
    custom_objects = {
        'Constrained3DKernelMinimal': Constrained3DKernelMinimal,
        # '_hard_swish': MobileNetBase._hard_swish,
        # '_relu6': MobileNetBase._relu6
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)


if __name__ == '__main__':
    image_path = Path(
        r'/scratch/p288722/datasets/vision/all_frames/D03_Huawei_P9/D03_V_flat_move_0001/D03_V_flat_move_0001-00001.png')

    img = cv2.imread(str(image_path))
    img = np.float32(img) / 255.0

    h_map = get_patch_location_mask(
        img,
        max_std_dev=np.array([0.02, 0.02, 0.02]),
        min_std_dev=np.array([0.005, 0.005, 0.005]),
        patch_dimensions=(5, 5),
        patches_type='homogeneous'  # homogeneous or non_homogeneous
    )
    h_map = h_map[2:-2, 2:-2]
    img_c = img[2:-2, 2:-2, :]
    homogeneous_map = np.repeat(h_map[:, :, np.newaxis], 3, axis=2)

    plt.figure()
    plt.imshow(homogeneous_map, interpolation='nearest')
    plt.title('Mask - Homogeneous Set to 1')
    plt.show()

    model = load_model(model_path=Path(r'/scratch/p288722/runtime_data/scd-videos/dev_const_layer/'
                                       r'50_frames_28d_64_pred/mobile_net/models/ConstNet_guru/fm-e00019.h5'))
    # Weights of the first constrained layer filter
    filters, biases = model.layers[1].get_weights()

    constrained_activations = keract.get_activations(model, img[np.newaxis, ...], layer_names=['constrained_layer'],
                                                     nodes_to_evaluate=None, output_format='simple', nested=False,
                                                     auto_compile=True)['constrained_layer'][0]
    act = constrained_activations
    act = (act - act.min()) / (act.max() - act.min())

    plt.figure()
    plt.imshow(act)
    plt.title('Constrained Activation Map')
    plt.show()

    non_homogeneous_map = np.ones_like(homogeneous_map) - homogeneous_map
    # non_homogeneous_map = np.repeat(nh_map[:, :, np.newaxis], 3, axis=2)

    # Multiply the images
    regions_from_img = np.multiply(img_c, homogeneous_map)
    regions_from_act = np.multiply(act, non_homogeneous_map)
    mixed_input = regions_from_img + regions_from_act

    plt.figure()
    plt.imshow(mixed_input)
    plt.title('Preprocessed input')
    plt.show()

    print('')
