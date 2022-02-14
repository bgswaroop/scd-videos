import itertools
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import ImageFile

from dataset.prnu import extract_single

AUTOTUNE = tf.data.experimental.AUTOTUNE
ImageFile.LOAD_TRUNCATED_IMAGES = True

tf.config.run_functions_eagerly(True)


# tf.data.experimental.enable_debug_mode()
# GlcmProperties = namedtuple('GlcmProperties', ['File',
#                                                'mean_contrast', 'max_contrast',
#                                                'mean_dissimilarity', 'max_dissimilarity',
#                                                'mean_homogeneity', 'max_homogeneity',
#                                                'mean_energy', 'max_energy',
#                                                'mean_correlation', 'max_correlation'])
#
#
# def get_glcm_properties(img_path):
#     from skimage.feature import greycomatrix, greycoprops
#     # https://scikit-image.org/docs/0.18.x/api/skimage.feature.html?highlight=greycomatrix#greycomatrix
#
#     img_rgb = Image.open(img_path)  # reading image as grayscale
#     img = img_rgb.convert('L')
#     img = np.asarray(img, dtype=np.uint8)
#     glcm = greycomatrix(img, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256)
#     p1 = greycoprops(glcm, prop='contrast')
#     p2 = greycoprops(glcm, prop='dissimilarity')
#     p3 = greycoprops(glcm, prop='homogeneity')
#     p4 = greycoprops(glcm, prop='energy')
#     p5 = greycoprops(glcm, prop='correlation')
#
#     # from matplotlib import pyplot as plt
#     # plt.figure()
#     # plt.imshow(img_rgb)
#     # plt.title(f'Homogeneity score - {score}')
#     # plt.show()
#     # plt.close()
#
#     properties = GlcmProperties(str(img_path),
#                                 np.mean(p1), np.max(p1), np.mean(p2), np.max(p2), np.mean(p3), np.max(p3),
#                                 np.mean(p4), np.max(p4), np.mean(p5), np.max(p5))
#     return properties


class DataFactory:

    def __init__(self, input_dir, batch_size=32, height=480, width=800, homogeneous_frames=None):

        with open(Path(input_dir).joinpath('train.json'), 'r') as f:
            self.train_data = json.load(f)
            self.class_names = np.array(sorted(self.train_data.keys()))
            self.train_data = list(itertools.chain.from_iterable(self.train_data.values()))
            # self.train_data = self._filter_images_based_on_homogeneity(self.train_data, homogeneous_frames)
            random.seed(108)
            random.shuffle(self.train_data)

        with open(Path(input_dir).joinpath('val.json'), 'r') as f:
            self.val_data = json.load(f)
            self.val_data = sorted(itertools.chain.from_iterable(self.val_data.values()))
            # self.val_data = self._filter_images_based_on_homogeneity(self.val_data, homogeneous_frames)

        with open(Path(input_dir).joinpath('test.json'), 'r') as f:
            self.test_data = json.load(f)
            self.test_data = sorted(itertools.chain.from_iterable(self.test_data.values()))
            # self.test_data = self._filter_images_based_on_homogeneity(self.test_data, homogeneous_frames)

        self.batch_size = batch_size
        self.img_width = width
        self.img_height = height
        self.seed = 108  # To allow reproducibility

    # @staticmethod
    # def _filter_images_based_on_homogeneity(img_paths, homogeneous_frames):
    #     if homogeneous_frames is None:
    #         return img_paths
    #     pool = Pool(cpu_count())
    #     homogeneity_scores = pool.map(get_image_homogeneity, img_paths)
    #     if homogeneous_frames is True:
    #         img_paths = [x for x, h in zip(img_paths, homogeneity_scores) if h >= 0.5]
    #     elif homogeneous_frames is False:
    #         img_paths = [x for x, h in zip(img_paths, homogeneity_scores) if h < 0.5]
    #     pool.close()
    #     return img_paths

    def get_class_names(self):
        return self.class_names

    def get_tf_input_dim(self):
        return tf.constant((self.img_height, self.img_width), tf.dtypes.int32)

    def process_path(self, file_path):
        # label = self.get_label(file_path)
        label = tf.py_function(self.get_label, [file_path], tf.float32)
        img = self.load_img(file_path)
        return img, label

    def get_label(self, file_path):
        # file_parts = tf.strings.split(file_path, os.path.sep)
        # file_parts = file_path.numpy().decode("utf-8").split(os.path.sep)
        file_parts = file_path.numpy().decode("utf-8").split(os.path.sep)
        class_name = file_parts[-3]
        one_hot_vec = tf.cast(class_name == self.class_names, dtype=tf.dtypes.float32, name="labels")
        return one_hot_vec

    @staticmethod
    def get_file_name(file_path):
        file_parts = tf.strings.split(file_path, os.path.sep)
        file_name = file_parts[-1]
        return file_name

    def load_img(self, file_path, resize_dim=None, ):
        img = tf.io.read_file(file_path)
        try:
            img = tf.image.decode_png(img, channels=3)
        except Exception as e:
            print(f'Issue decoding the png image - {file_path}\n')
            raise e

        img = tf.py_function(self.center_crop, [img], tf.uint8)
        img_r = tf.expand_dims(img[:, :, 0], -1)
        img_g = tf.expand_dims(img[:, :, 1], -1)
        img_b = tf.expand_dims(img[:, :, 2], -1)

        img_r = tf.py_function(extract_single, [img_r], tf.float32)
        img_g = tf.py_function(extract_single, [img_g], tf.float32)
        img_b = tf.py_function(extract_single, [img_b], tf.float32)

        img = tf.stack([img_r, img_g, img_b], axis=-1)

        # img = tf.image.convert_image_dtype(img, tf.dtypes.float32)

        # img = img[:, :, 1]  # Considering only the Green color channel
        # img = tf.expand_dims(img, -1)   # Adding back the num_channels axis : (height, width, num_channels)

        # img = tfio.experimental.color.rgb_to_lab(img)
        # img = (img + tf.constant([0.0, 128, 128], shape=(1, 1, 3))) / tf.constant([100.0, 255, 255], shape=(1, 1, 3))

        # Set to CNN Input Dimensions
        # if resize_dim is None:
        #     resize_dim = self.get_tf_input_dim()
        # img = tf.image.resize(img, size=resize_dim)

        # if self.center_crop:
        # explicitly renaming the variables to avoid confusion
        # img_height, img_width, _ = img.get_shape().as_list()
        # crop_height, crop_width = self.img_height, self.img_width
        # print(' ')

        # if img_height <= crop_height or img_width <= crop_width:
        #     # rotate image clockwise 90 degrees and check further on if it matches the crop dimensions
        #     img = tf.image.rot90(img)
        #     img_height, img_width, _ = img.get_shape().as_list()

        # if img_height <= crop_height or img_width <= crop_width:
        #     pass    # rotate anti-clockwise 90 degrees and perform resize
        # try:
        #     img = tf.image.crop_to_bounding_box(image=img,
        #                                         offset_height=int(img_height / 2 - crop_height / 2),
        #                                         offset_width=int(img_width / 2 - crop_width / 2),
        #                                         target_height=480,
        #                                         target_width=800)
        # except ValueError:
        #     img = tf.image.rot90(img)
        #     img_height, img_width, _ = img.get_shape().as_list()
        #     img = tf.image.crop_to_bounding_box(image=img,
        #                                         offset_height=int(img_height / 2 - crop_height / 2),
        #                                         offset_width=int(img_width / 2 - crop_width / 2),
        #                                         target_height=480,
        #                                         target_width=800)
        #
        # # BM3D Denoising
        # im_denoised = tf.py_function(func=bm3d, inp=[img, 0.02], Tout=tf.float32)
        # additive_noise = img - im_denoised
        # img = additive_noise

        return img

    def get_tf_train_data(self, category):
        t_start = time.time()
        file_path_ds = tf.data.Dataset.from_tensor_slices(self.train_data)
        if category == 'native':
            ds_list = [x for x in file_path_ds if
                       ('WA' not in x.numpy().decode("utf-8")) and ('YT' not in x.numpy().decode("utf-8"))]
            file_path_ds = tf.data.Dataset.from_tensor_slices(ds_list)
        elif category == 'whatsapp':
            ds_list = [x for x in file_path_ds if 'WA' in x.numpy().decode("utf-8")]
            file_path_ds = tf.data.Dataset.from_tensor_slices(ds_list)
        elif category == 'youtube':
            ds_list = [x for x in file_path_ds if 'YT' in x.numpy().decode("utf-8")]
            file_path_ds = tf.data.Dataset.from_tensor_slices(ds_list)

        print(f"Found {len(list(file_path_ds))} images in ({int(time.time() - t_start)} sec.)")

        print(f"\nPrinting first 2 elements of dataset:\n")
        for element in file_path_ds.take(2):
            k = self.process_path(element)
            print(k, element)
            break

        # Load actual images and create labels accordingly
        labeled_ds = file_path_ds.map(self.process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # labeled_ds = self.pre_process(labeled_ds)

        print(f"\nFinished creating labeled dataset ({int(time.time() - t_start)} sec.)\n")

        # Determine number of total elements
        num_elements = tf.data.experimental.cardinality(labeled_ds).numpy()
        print(f"\ntotal number elements: {num_elements} ({int(time.time() - t_start)} sec.)\n")

        # Set batch and prefetch preferences
        labeled_ds = labeled_ds.batch(self.batch_size, drop_remainder=False)
        labeled_ds = labeled_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        num_batches = math.ceil(num_elements / self.batch_size)
        return labeled_ds, num_batches

    def get_tf_evaluation_data(self, category, mode):
        t_start = time.time()

        if mode == 'test':
            file_path_ds = tf.data.Dataset.from_tensor_slices(self.test_data)
        elif mode == 'val':
            file_path_ds = tf.data.Dataset.from_tensor_slices(self.val_data)
        elif mode == 'train':
            file_path_ds = tf.data.Dataset.from_tensor_slices(self.train_data)
        else:
            raise ValueError('Invalid mode')

        if category == 'native':
            ds_list = [x for x in file_path_ds if
                       ('WA' not in x.numpy().decode("utf-8")) and ('YT' not in x.numpy().decode("utf-8"))]
            file_path_ds = tf.data.Dataset.from_tensor_slices(ds_list)
        elif category == 'whatsapp':
            ds_list = [x for x in file_path_ds if 'WA' in x.numpy().decode("utf-8")]
            file_path_ds = tf.data.Dataset.from_tensor_slices(ds_list)
        elif category == 'youtube':
            ds_list = [x for x in file_path_ds if 'YT' in x.numpy().decode("utf-8")]
            file_path_ds = tf.data.Dataset.from_tensor_slices(ds_list)

        print(f"Found {len(list(file_path_ds))} images in ({int(time.time() - t_start)} sec.)")

        # Create labeled dataset by loading the image and estimating the label
        labeled_ds = file_path_ds.map(self.process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # labeled_ds = self.pre_process(labeled_ds)

        labeled_ds = labeled_ds.batch(self.batch_size, drop_remainder=False)
        labeled_ds = labeled_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        print(f"Finished loading test frames ({int(time.time() - t_start)} sec.)")

        # Create dataset of file names which is necessary for evaluation
        # filename_ds = file_path_ds.map(self.get_file_name, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return file_path_ds, labeled_ds

    def get_tf_val_data(self, category):
        return self.get_tf_evaluation_data(category, mode='val')

    def get_tf_test_data(self, category):
        return self.get_tf_evaluation_data(category, mode='test')

    @staticmethod
    def get_labels(ds):
        # Credits to Guru
        labels_ds = ds.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices(y))
        ground_truth_labels = np.array(list(labels_ds.as_numpy_iterator())).astype(np.int32)
        return ground_truth_labels

    # @staticmethod
    # def bm3d_noise(sample):
    #     img, label = sample
    #     noise = img - tf.convert_to_tensor(bm3d(img.numpy(), sigma_psd=0.02))
    #     tf.py_function(random_rotate_image, [image], [tf.float32])
    #     return noise, img
    #
    # def pre_process(self, labeled_ds):
    #     # ds = tfds.as_numpy(labeled_ds)
    #     # ds = map(lambda x: (x[0] - bm3d(x[0], sigma_psd=0.02), x[1]), ds)
    #     # ds = tf.data.Dataset.from_tensor_slices(ds)
    #     ds = labeled_ds.map(self.bm3d_noise, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #     return ds

    def center_crop(self, img):
        img = tf.convert_to_tensor(img.numpy())
        img_height, img_width, _ = img.get_shape().as_list()
        crop_height, crop_width = self.img_height, self.img_width

        try:
            img = tf.image.crop_to_bounding_box(image=img,
                                                offset_height=int(img_height / 2 - crop_height / 2),
                                                offset_width=int(img_width / 2 - crop_width / 2),
                                                target_height=480,
                                                target_width=800)
        except ValueError:
            img = tf.image.rot90(img)
            img_height, img_width, _ = img.get_shape().as_list()
            img = tf.image.crop_to_bounding_box(image=img,
                                                offset_height=int(img_height / 2 - crop_height / 2),
                                                offset_width=int(img_width / 2 - crop_width / 2),
                                                target_height=480,
                                                target_width=800)
        return img

    def center_crop_wrapper(self, img, label):
        # explicitly renaming the variables to avoid confusion
        img = tf.py_function(self.center_crop, img, tf.float32)
        return img, label

    def pre_process(self, labeled_ds):
        """
        Center crop the dataset
        :param labeled_ds:
        :return:
        """
        ds = labeled_ds.map(self.center_crop_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return ds

    def get_distance_matrix(self):
        intra_class_distance = np.ndarray(shape=(len(self.class_names), len(self.class_names)))
        for row_id, device in enumerate(self.class_names):
            brand = device.split('_')[1]
            for col_id, device in enumerate(self.class_names):
                if row_id == col_id:
                    intra_class_distance[row_id][col_id] = 0.0
                elif brand in device:
                    intra_class_distance[row_id][col_id] = 0.00001
                else:
                    intra_class_distance[row_id][col_id] = 0.99999

        intra_class_distance = tf.convert_to_tensor(intra_class_distance)
        # intra_class_distance = intra_class_distance / tf.math.reduce_sum(intra_class_distance, axis=1, keepdims=True)
        return tf.cast(intra_class_distance, dtype=tf.float32)
