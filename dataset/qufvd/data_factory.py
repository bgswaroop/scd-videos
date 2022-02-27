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

from .frame_selection import get_frames_dataset

AUTOTUNE = tf.data.experimental.AUTOTUNE
ImageFile.LOAD_TRUNCATED_IMAGES = True
tf.config.run_functions_eagerly(True)


class DataFactory:

    def __init__(self, args):

        self.train_data = get_frames_dataset('Training', args)
        self.train_data = list(itertools.chain.from_iterable(self.train_data.values()))
        # self.train_data = self._filter_images_based_on_homogeneity(self.train_data, homogeneous_frames)
        random.seed(108)
        random.shuffle(self.train_data)

        self.val_data = get_frames_dataset('Validation', args)
        self.val_data = sorted(itertools.chain.from_iterable(self.val_data.values()))
        # self.val_data = self._filter_images_based_on_homogeneity(self.val_data, homogeneous_frames)

        self.test_data = get_frames_dataset('Testing', args)
        self.test_data = sorted(itertools.chain.from_iterable(self.test_data.values()), reverse=True)
        # self.test_data = self._filter_images_based_on_homogeneity(self.test_data, homogeneous_frames)
        self.all_I_frames_dir = args.all_I_frames_dir
        self.class_names = self.get_class_names()

        self.batch_size = args.batch_size
        self.img_width = args.width
        self.img_height = args.height
        self.seed = 108  # To allow reproducibility

    def get_class_names(self):
        devices = []
        for model in self.all_I_frames_dir.joinpath(rf'FrameDatabaseTraining').glob('*'):
            for device in model.glob('*'):
                classname = f'{model.name}_{device.name}'
                devices.append(classname)
        sorted_devices = sorted(devices, key=lambda x: int(x.split('Device')[-1]))
        return np.array(sorted_devices)

    def get_tf_input_dim(self):
        return tf.constant((self.img_height, self.img_width), tf.dtypes.int32)

    def process_path(self, file_path):
        label = tf.py_function(self.get_label, [file_path], tf.float32)
        img = self.load_img(file_path)
        return img, label

    def get_label(self, file_path):
        file_path = file_path.numpy().decode('ascii')
        file_parts = file_path.split(os.path.sep)
        class_name = f'{file_parts[-3]}_{file_parts[-2]}'
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

        # Correct image orientation and perform center crop
        img = tf.image.convert_image_dtype(img, tf.dtypes.float32)
        img = tf.py_function(self.center_crop, [img], tf.float32)
        return img

    def get_tf_train_data(self, category):
        t_start = time.time()
        file_path_ds = tf.data.Dataset.from_tensor_slices(self.train_data)
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

    def center_crop(self, img):
        img = tf.convert_to_tensor(img.numpy())
        img_height, img_width, _ = img.get_shape().as_list()

        # Correcting image orientation
        if img_height > img_width:
            img = tf.image.rot90(img)
            img_height, img_width = img_width, img_height

        # Perform center crop
        crop_height, crop_width = self.img_height, self.img_width
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
