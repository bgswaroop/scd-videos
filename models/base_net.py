import abc
import os

import numpy as np
import tensorflow as tf

tf.config.run_functions_eagerly(True)
from tensorflow.keras.callbacks import TensorBoard

from utils.callbacks import WarmUpCosineDecayScheduler
from utils.callbacks import PredictionsCallback


class BaseNet(abc.ABC):
    def __init__(self, num_batches, global_results_dir, const_type=None, model_path=None, lr=0.1):
        self.num_batches = num_batches
        self.model = None
        self.model_path = None
        if model_path is not None:
            self.set_model(model_path)
        self.lr = lr

        self.verbose = False
        self.model_name = None

        # Constrained layer properties
        assert const_type in {None, 'guru', 'derrick'}
        self.const_type = const_type
        self.constrained_n_filters = 3
        self.constrained_kernel_size = 5

        # Results directories
        self.global_save_models_dir = global_results_dir.joinpath('models')
        self.global_tensorflow_dir = global_results_dir.joinpath('tensorboard')

        # Fix the RNGs
        np.random.seed(108)
        tf.compat.v1.set_random_seed(108)
        tf.random.set_seed(108)

    def set_model(self, model_path):
        # Path is e.g. ~/constrained_net/fm-e00001.h5
        path_splits = model_path.split(os.sep)
        model_name = path_splits[-2]

        self.model_path = model_path
        self.model_name = model_name

        if self.model is None:
            from models import Constrained3DKernelMinimal, CombineInputsWithConstraints, \
                SupervisedContrastiveLoss, PPCCELoss
            custom_objects = {
                'Constrained3DKernelMinimal': Constrained3DKernelMinimal,
                'CombineInputsWithConstraints': CombineInputsWithConstraints,
                'SupervisedContrastiveLoss': SupervisedContrastiveLoss,
                'PPCCELoss': PPCCELoss,
            }
            self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        else:
            self.model.load_weights(model_path)

    def create_model(self, **kwargs):
        raise NotImplementedError('method create_model is not implemented')

    def compile(self):
        # custom_loss = self.make_custom_loss(self.model)
        self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                           optimizer=tf.keras.optimizers.SGD(learning_rate=self.lr, momentum=0.95, decay=0.0005),
                           metrics=["acc"],
                           run_eagerly=True
                           )
        self.model.run_eagerly = True

    def get_tensorboard_path(self):
        if self.model_name is None:
            raise ValueError("Model has no name specified. This is required in order to save TensorBoard log-files.")

        # Create directory if not exists
        path = self.global_tensorflow_dir.joinpath(self.model_name)
        path.mkdir(exist_ok=True, parents=True)

        return path

    def get_save_model_path(self, file_name):
        if self.model_name is None:
            raise ValueError("Model has no name specified. This is required in order to save checkpoints.")

        # Create directory if not exists
        path = self.global_save_models_dir.joinpath(self.model_name)
        path.mkdir(exist_ok=True, parents=True)

        # Append file name and return
        return path.joinpath(file_name)

    def __get_initial_epoch(self):
        # Means we train from scratch
        if self.model_path is None:
            return 0

        path = self.model_path
        file_name = path.split(os.sep)[-1]

        if file_name is None:
            return 0

        file_name = file_name.split(".")[0]
        splits = file_name.split("-")
        for split in splits:
            if split.startswith("e"):
                epoch = split.strip("e")
                return int(epoch)

        return 0

    def train(self, train_ds, val_ds, epochs=1):
        if self.model is None:
            raise ValueError("Cannot start training! self.model is None!")

        initial_epoch = self.__get_initial_epoch()
        epochs += initial_epoch

        callbacks = self.get_callbacks(train_ds, val_ds, epochs, initial_epoch)

        self.model.fit(train_ds,
                       epochs=epochs,
                       initial_epoch=initial_epoch,
                       validation_data=val_ds,
                       callbacks=callbacks,
                       workers=12,
                       use_multiprocessing=True)

    def get_callbacks(self, train_ds, val_ds, epochs, completed_epochs):
        default_file_name = "fm-e{epoch:05d}.h5"
        save_model_path = self.get_save_model_path(default_file_name)

        save_model_cb = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path,
                                                           verbose=0,
                                                           save_weights_only=False,
                                                           save_freq='epoch')  # period=1 (for older ver of TensorFlow)

        tensorboard_cb = TensorBoard(log_dir=str(self.get_tensorboard_path()), update_freq='batch')

        # lr_callback = tf.keras.callbacks.LearningRateScheduler(
        #     schedule=tf.keras.optimizers.schedules.ExponentialDecay(
        #         initial_learning_rate=0.001, decay_steps=1, decay_rate=0.96
        #     ),
        #     # schedule=CosineDecay(
        #     #     initial_learning_rate=0.001, decay_steps=40, alpha=0.0, name=None
        #     # ),
        #     verbose=1)

        steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        warm_up_epochs = 0.25 if epochs < 3 else 3
        lr_callback = WarmUpCosineDecayScheduler(learning_rate_base=self.lr,
                                                 total_steps=epochs * steps_per_epoch,
                                                 global_step_init=completed_epochs * steps_per_epoch,
                                                 warmup_learning_rate=0,
                                                 warmup_steps=int(warm_up_epochs * steps_per_epoch),
                                                 hold_base_rate_steps=0,
                                                 verbose=1)

        # print_predictions_cb = PredictionsCallback(train_ds=train_ds, val_ds=val_ds)

        return [save_model_cb, tensorboard_cb, lr_callback]

    def evaluate(self, test_ds, model_path=None):
        if model_path is not None:
            self.model = tf.keras.models.load_model(model_path)
        elif self.model is None:
            raise ValueError("No model available")
        test_loss, test_acc = self.model.evaluate(test_ds)
        return test_acc, test_loss

    def predict(self, dataset, load_model=None):
        if load_model is not None:
            self.model = tf.keras.models.load_model(load_model)
        elif self.model is None:
            raise ValueError("No model available")

        test_ds = dataset.get_test_data()
        predicted_labels = self.model.predict_class(test_ds)
        true_labels = dataset.get_labels(test_ds)
        return true_labels, predicted_labels

    def print_model_summary(self):
        if self.model is None:
            print("Can't print model summary, self.model is None!")
        else:
            print(f"\nSummary of model:\n{self.model.summary()}")
