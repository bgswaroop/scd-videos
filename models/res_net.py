from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D

from . import BaseNet, Constrained3DKernelMinimal


class ResNet(BaseNet):
    def __init__(self, num_batches, global_results_dir, const_type, model_path=None):
        super().__init__(num_batches, global_results_dir, const_type, model_path)

    def create_model(self, num_outputs, height, width, model_name=None, use_pretrained=True):
        self.model_name = model_name  # fixme: This should ideally be in the __init__
        input_shape = (height, width, 3)

        self.model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=True, weights=None,
                                                                input_shape=input_shape, classes=num_outputs)
        if use_pretrained:
            pretrained = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet',
                                                                    input_shape=input_shape, pooling='avg')

            for idx in range(len(pretrained.layers)):
                self.model.layers[idx].set_weights(pretrained.layers[idx].get_weights())

        if self.const_type:
            self.model = Sequential([
                tf.keras.layers.InputLayer(input_shape=input_shape),
                Conv2D(filters=3, kernel_size=5, strides=(1, 1), padding="same",
                       kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.0001, maxval=1, seed=108),
                       kernel_constraint=Constrained3DKernelMinimal(self.const_type),
                       name="constrained_layer"),
                self.model
            ])

        self.compile()
        return self.model


if __name__ == '__main__':
    net = ResNet(num_batches=10, global_results_dir=Path('.'), const_type=None, model_path=None)
    m = net.create_model(num_outputs=28, height=480, width=800, model_name=None, use_pretrained=True)

    print(' ')
