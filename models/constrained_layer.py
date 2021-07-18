import os
import random

import cv2
import numpy as np
import tensorflow as tf
from keras.constraints import Constraint
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical


class CombineInputsWithConstraints(tf.keras.layers.Layer):
    def __init__(self, kernel_size=5,
                 min_threshold=0.005,
                 max_threshold=0.02, **kwargs):
        kwargs['name'] = 'combine_inputs_with_constraints'
        kwargs['trainable'] = False

        super(CombineInputsWithConstraints, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def build(self, input_shape):
        self.p = self.kernel_size // 2  # inner padding
        _, self.img_height, self.img_width, self.num_channels = input_shape

    def call(self, cnn_inputs, constrained_activations):
        """
        This method is being implemented assuming that the input tensor dimensions correspond to:
        batch_size x height x width x num_channels
        :param cnn_inputs: inputs to the CNN
        :param constrained_activations: constrained net outputs
        :return:
        """

        k = self.kernel_size
        patches = tf.image.extract_patches(cnn_inputs, [1, k, k, 1], [1, 1, 1, 1], [1, 1, 1, 1], 'VALID')
        p = tf.reshape(patches, shape=(
            -1, self.img_height - 2 * self.p, self.img_width - 2 * self.p, k, k, self.num_channels))
        std_dev = tf.math.reduce_std(p, axis=[3, 4])
        homo_mask = tf.reduce_prod(tf.multiply(tf.cast(std_dev >= self.min_threshold, tf.float32),
                                               tf.cast(std_dev <= self.max_threshold, tf.float32)), axis=3)
        homo_mask = tf.stack([homo_mask] * 3, axis=3)
        non_homo_mask = tf.ones_like(homo_mask) - homo_mask

        valid_inputs = cnn_inputs[:, self.p:-self.p, self.p:-self.p, :]
        min_val = tf.reduce_min(constrained_activations, axis=[1, 2, 3], keepdims=True)
        max_val = tf.reduce_max(constrained_activations, axis=[1, 2, 3], keepdims=True)
        min_max_activations = (constrained_activations - min_val) / (max_val - min_val)

        pre_precessed_output = tf.multiply(homo_mask, valid_inputs) + tf.multiply(non_homo_mask, min_max_activations)

        return pre_precessed_output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'kernel_size': self.kernel_size,
            'min_threshold': self.min_threshold,
            'max_threshold': self.max_threshold,
        })
        return config


class Constrained3DKernelMinimal(Constraint):
    def __init__(self, const_type='guru'):
        super(Constrained3DKernelMinimal, self).__init__()
        self.const_type = const_type

    def __call__(self, w):
        """
        =========================================================================================
        === Same functionality as Constrained3DKernel but without extensive in-line comments. ===
        =========================================================================================

        This custom weight constraint implements the constrained convolutional layer for RGB-images.
        However, this is not as straightforward as it should be. Since TensorFlow prohibits us from
        assigning new values directly to the weight matrix, we need a trick to update its values.
        This trick consists of multiplying the weight matrix by so-called 'mask matrices'
        to get the desired results.

        For example, if we want to set the center values of the weight matrix to zero, we first create a
        mask matrix of the same size consisting of only ones, except at the center cells. The center cells
        will have as value one. After multiplying the weight matrix with this mask matrix,
        we obtain a 'new' weight matrix, where the center values are set to zero but with the remaining values
        untouched.

        More information about this problem:
        #https://github.com/tensorflow/tensorflow/issues/14132

        The incoming weight matrix 'w' is a 4D array of shape (x, y, z, n_filters) where (normally):
        x = 5
        y = 5
        z = 3, since we're using RGB images
        n_filters = 3

        This means there are 3 filters in total with each filter being 3-dimensional.
       """
        if self.const_type == 'guru':
            return self.__constraint_positive_surrounding_weights(w)
        elif self.const_type == 'derrick':
            return self.__constraint_derrick_et_al_with_transpose(w)
        elif self.const_type == 'bug':
            return self.__constraint_derrick_et_al_with_reshape(w)
        else:
            raise ValueError('Invalid constraint type')

    def get_config(self):
        return {}

    @staticmethod
    def __constraint_positive_surrounding_weights(w):
        center = w.shape[0] // 2

        # Determine the min and max values for the non-center pixels
        center_zero_mask = np.ones(w.shape)
        center_zero_mask[center, center, :, :] = 0
        w = w * center_zero_mask
        center_max_mask = np.zeros(w.shape)
        center_max_mask[center, center, :, :] = np.inf
        w_max_center = tf.math.add(w, center_max_mask)
        w_min = tf.reduce_min(w_max_center, axis=[0, 1], keepdims=True)

        center_min_mask = np.zeros(w.shape)
        center_min_mask[center, center, :, :] = -np.inf
        w_min_center = tf.math.add(w, center_min_mask)
        w_max = tf.reduce_max(w_min_center, axis=[0, 1], keepdims=True)

        # Reducing the min by a small value to avoid zeros in the weight matrix
        w_min -= tf.random.uniform([1], minval=0.001, maxval=0.1)

        # 1. Perform min max normalization
        w = tf.math.divide(tf.math.subtract(w, w_min), w_max - w_min)

        # 2. Perform l1 normalization
        center_zero_mask = np.ones(w.shape)
        center_zero_mask[center, center, :, :] = 0
        w *= center_zero_mask
        w = tf.math.divide(w, tf.reduce_sum(w, axis=[0, 1], keepdims=True))

        # 3. Set the center value to -1
        center_one_mask = np.zeros(w.shape)
        center_one_mask[center, center, :, :] = 1
        w = tf.math.subtract(w, center_one_mask)

        return w

    @staticmethod
    def __constraint_derrick_et_al_with_reshape(w):
        w_original_shape = w.shape
        w = w * 10000  # scale by 10k to prevent numerical issues

        # 1. Reshaping of 'w'
        x, y, z, n_kernels = w_original_shape[0], w_original_shape[1], w_original_shape[2], w_original_shape[3]
        center = x // 2  # Determine the center cell on the xy-plane.
        new_shape = [n_kernels, z, x, y]
        w = tf.reshape(w, new_shape)

        # 2. Set center values of 'w' to zero by multiplying 'w' with mask-matrix
        center_zero_mask = np.ones(new_shape)
        center_zero_mask[:, :, center, center] = 0
        w *= center_zero_mask

        # 3. Normalize values w.r.t xy-planes
        xy_plane_sum = tf.reduce_sum(w, [2, 3], keepdims=True)  # Recall new shape of w: (n_kernels, z, y, x).
        w = tf.math.divide(w, xy_plane_sum)  # Divide each element by its corresponding xy-plane sum-value

        # 4. Set center values of 'w' to negative one by subtracting mask-matrix from 'w'
        center_one_mask = np.zeros(new_shape)
        center_one_mask[:, :, center, center] = 1
        w = tf.math.subtract(w, center_one_mask)

        # Reshape 'w' to original shape and return
        w = tf.reshape(w, w_original_shape)
        return w

    @staticmethod
    def __constraint_derrick_et_al_with_transpose(w):
        w_original_shape = w.shape
        w = w * 10000  # scale by 10k to prevent numerical issues

        # 1. Reshaping of 'w'
        x, y, z, n_kernels = w_original_shape[0], w_original_shape[1], w_original_shape[2], w_original_shape[3]
        center = x // 2  # Determine the center cell on the xy-plane.
        new_shape = [n_kernels, z, x, y]
        w = tf.transpose(w, [3, 2, 0, 1])

        # 2. Set center values of 'w' to zero by multiplying 'w' with mask-matrix
        center_zero_mask = np.ones(new_shape)
        center_zero_mask[:, :, center, center] = 0
        w *= center_zero_mask

        # 3. Normalize values w.r.t xy-planes
        xy_plane_sum = tf.reduce_sum(w, [2, 3], keepdims=True)  # Recall new shape of w: (n_kernels, z, y, x).
        w = tf.math.divide(w, xy_plane_sum)  # Divide each element by its corresponding xy-plane sum-value

        # 4. Set center values of 'w' to negative one by subtracting mask-matrix from 'w'
        center_one_mask = np.zeros(new_shape)
        center_one_mask[:, :, center, center] = 1
        w = tf.math.subtract(w, center_one_mask)

        # Reshape 'w' to original shape and return
        w = tf.transpose(w, [2, 3, 1, 0])
        return w


def __create_model(input_shape):
    model = Sequential([
        Conv2D(filters=3, kernel_size=5, strides=(1, 1), input_shape=input_shape, padding='valid',
               kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.0001, maxval=1, seed=108),
               kernel_constraint=Constrained3DKernelMinimal(),
               name="constrained_layer"),
        Flatten(),
        Dense(2, activation=tf.keras.activations.softmax)
    ])

    return model


def __get_friendly_weights(model, layer_index=0):
    """
    Transforming HWCN - Height x width x channels x filters into NCHW - filters x channels x height x width
    """
    weights = model.layers[layer_index].get_weights()[0]
    return tf.transpose(weights, [3, 2, 0, 1])


def __train(x, y):
    print_weight_info = True
    input_shape = (256, 256, 3)
    model = __create_model(input_shape)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.00001, momentum=0.95, decay=0.0005)

    for i in range(10):
        with tf.GradientTape() as tape:
            # Make prediction
            prediction = model(x)
            # Calculate loss
            loss = categorical_crossentropy(y, prediction)
            print(f"Epoch: {i}, Loss: {np.average(loss)}")
            # Calculate gradients
            grads = tape.gradient(loss, model.trainable_variables)
            # Update weights by applying gradients
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if print_weight_info:
        constr_weights = __get_friendly_weights(model, layer_index=0)
        # print(f"\nEntire weight-matrix (iter {i+1}):\n {constr_weights}")
        print(f"\nFilter 1:\n {constr_weights[0]}")
        print(f"\nFilter 1, XY-axis 1:\n {constr_weights[0][0]}")
        print(f"Filter 1, XY-axis 1 sum-value: {tf.reduce_sum(constr_weights[0][0])}")
        print(f"\nFilter 1, XY-axis 2:\n {constr_weights[0][1]}")
        print(f"Filter 1, XY-axis 2 sum-value: {tf.reduce_sum(constr_weights[0][1])}")
        print(f"\nFilter 1, XY-axis 3:\n {constr_weights[0][2]}")
        print(f"Filter 1, XY-axis 3 sum-value: {tf.reduce_sum(constr_weights[0][2])}")


def __get_train_data():
    DATA_DIR = "../_miscellaneous/test_constrained_layer/PetImages"
    CATEGORIES = ["Dog", "Cat"]
    IMG_SIZE = 256

    training_data = []

    for category in CATEGORIES:
        path = os.path.join(DATA_DIR, category)  # Path to cats or dogs dir
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])

    random.shuffle(training_data)
    return training_data


def __preprocess_data(train_data):
    X = []
    y = []
    IMG_SIZE = 256

    print("training data size: ", len(train_data))
    for features, label in train_data:
        X.append(features)
        y.append(label)

    # X is an numpy array. First parameter of reshape (-1) is necessary to indicate to take all the features we have.
    # The last parameter is in this case a 1 since we are using grayscale. By colored images this has to be set to 3.
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    # Normalize values within range 0,1
    X = X / 255.0
    # One-hot encoding
    y = to_categorical(y, 2)

    return X, y


if __name__ == "__main__":
    # inputs, targets = __preprocess_data(train_data=__get_train_data())
    # __train(x=inputs, y=targets)

    cnn_inputs = np.random.uniform(low=0, high=1, size=(2, 480, 800, 3))
    cnn_inputs = tf.convert_to_tensor(cnn_inputs, dtype=tf.float32)
    activations = np.random.uniform(low=-1, high=0, size=(2, 476, 796, 3))
    activations = tf.convert_to_tensor(activations, dtype=tf.float32)
    output = CombineInputsWithConstraints()(cnn_inputs, activations)
    # f = Constrained3DKernelMinimal()
    # new_weights = f(weights)

    print(' ')