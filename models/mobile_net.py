from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D

from . import BaseNet, Constrained3DKernelMinimal, PPCCELoss


class MobileNet(BaseNet):
    def __init__(self, num_batches, global_results_dir, const_type, model_path=None, lr=0.1):
        super().__init__(num_batches, global_results_dir, const_type, model_path, lr)
        assert const_type in {'bug', 'guru', 'derrick'}
        self.const_type = const_type

    def create_model(self, num_outputs, height, width, distance_matrix, model_name=None, use_pretrained=True):
        self.model_name = model_name  # fixme: This should ideally be in the __init__
        input_shape = (height, width, 3)

        self.model = tf.keras.applications.MobileNetV3Small(input_shape=input_shape, include_top=True,
                                                            weights=None, classes=num_outputs,
                                                            include_preprocessing=False)
        if use_pretrained:
            pretrained = tf.keras.applications.MobileNetV3Small(input_shape=input_shape, include_top=False,
                                                                weights='imagenet', include_preprocessing=False)
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

        self.distance_matrix = distance_matrix
        self.compile()
        return self.model

    # def compile(self):
    #     self.model.compile(loss=PPCCELoss(self.distance_matrix),
    #                        optimizer=tf.keras.optimizers.SGD(learning_rate=self.lr, momentum=0.95, decay=0.0005),
    #                        metrics=["acc"],
    #                        run_eagerly=True
    #                        )
    #     self.model.run_eagerly = True


if __name__ == '__main__':
    net = MobileNet(num_batches=10, global_results_dir=Path('.'), const_type=None, model_path=None)
    m = net.create_model(num_outputs=28, height=480, width=800, class_names=None, model_name=None, use_pretrained=True)

    print(' ')
