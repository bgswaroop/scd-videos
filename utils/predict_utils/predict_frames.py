import os
from multiprocessing import Pool, cpu_count

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from keract import keract
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dataset import DataFactory
# from dataset.data_factory import get_glcm_properties


class FramePredictor:

    def __init__(self, model_dir, model_file_name, result_dir, input_dir, homogeneity_csv,
                 weights_only=False, model_class=None):
        self.model_dir = model_dir
        self.model_file_name = model_file_name
        self.result_dir = result_dir
        self.input_dir = input_dir
        self.homogeneity_csv = homogeneity_csv

        # Load model
        from models import Constrained3DKernelMinimal, CombineInputsWithConstraints, PPCCELoss
        model_path = os.path.join(model_dir, model_file_name)
        custom_objects = {
            'Constrained3DKernelMinimal': Constrained3DKernelMinimal,
            'CombineInputsWithConstraints': CombineInputsWithConstraints,
            'PPCCELoss': PPCCELoss,
        }
        if weights_only:  # fixme: deprecated functionality
            model_class.model.load_weights(model_path)
            self.model = model_class.model
        else:
            self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

    def start(self, test_ds, filenames):
        output_file = self.get_output_file()
        return self._predict_and_save(test_ds, filenames, output_file)

    def get_output_file(self):
        output_file = f"{self.model_file_name.split('.')[0]}_F_predictions.csv"
        return os.path.join(self.result_dir, output_file)

    def __plot_constrained_filter(self):

        matplotlib.rcParams.update({'font.size': 15})
        # retrieve weights from the second hidden layer
        filters, biases = self.model.layers[0].get_weights()
        # filters = filters.reshape((3, 3, 5, 5))
        filters = filters.transpose([3, 2, 0, 1])
        fig, axs = plt.subplots(3, 3, figsize=(15, 15), dpi=300, sharey=True, sharex=True)
        # Set the range for the plots
        # v_min = np.min(filters)
        # v_max = np.max(filters)
        v_min = v_max = None
        for i in range(3):
            for j in range(3):
                # Plot the filter weights as an image
                im = axs[i][j].imshow(filters[i][j], vmin=v_min, vmax=v_max, cmap='viridis')
                divider = make_axes_locatable(axs[i][j])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
                # Print the filter weights to the plot
                for (y, x), label in np.ndenumerate(filters[i][j]):
                    axs[i][j].text(x, y, round(label, 2), ha='center', va='center')
        axs[0][0].set_ylabel('Filter 1')
        axs[1][0].set_ylabel('Filter 2')
        axs[2][0].set_ylabel('Filter 3')
        axs[0][0].title.set_text('Channel 1')
        axs[0][1].title.set_text('Channel 2')
        axs[0][2].title.set_text('Channel 3')
        plt.tight_layout()
        plt.show()
        plt.cla()
        plt.clf()
        plt.close()

    def __plot_constrained_activations(self, test_ds):

        for index, element in enumerate(test_ds.take(10)):
            activations = keract.get_activations(self.model, element[0].numpy(), layer_names=['constrained_layer'],
                                                 nodes_to_evaluate=None, output_format='simple', nested=False,
                                                 auto_compile=True)
            plt.figure()

            fig, axs = plt.subplots(2, 4, dpi=300, figsize=(12, 4), sharex=True, sharey=True)
            axs[0][0].imshow(element[0].numpy().reshape((480, 800, 3)))
            axs[0][1].imshow(element[0].numpy().reshape((480, 800, 3))[:, :, 0])
            axs[0][2].imshow(element[0].numpy().reshape((480, 800, 3))[:, :, 1])
            axs[0][3].imshow(element[0].numpy().reshape((480, 800, 3))[:, :, 2])

            axs[0][0].title.set_text('RGB')
            axs[0][0].set_ylabel('Inputs')
            axs[0][1].title.set_text('R')
            axs[0][2].title.set_text('G')
            axs[0][3].title.set_text('B')

            v = np.array(activations['constrained_layer']).reshape((476, 796, 3))
            act = (v - v.min()) / (v.max() - v.min())
            axs[1][0].imshow(act)
            axs[1][1].imshow(act[:, :, 0])
            axs[1][2].imshow(act[:, :, 1])
            axs[1][3].imshow(act[:, :, 2])
            axs[1][0].set_ylabel('Constrained\nactivations')

            # plt.colorbar()
            plt.tight_layout()
            plt.savefig(f'{index}.png')
            plt.show()
            plt.cla()
            plt.clf()
            plt.close()

    def __predict_frames(self, test_ds):

        # self.__plot_constrained_filter()
        # self.__plot_constrained_activations(test_ds)

        # self.model.compile()
        # evaluate_dict = self.model.evaluate(test_ds, return_dict=True)
        softmax_scores = self.model.predict(test_ds, verbose=1)

        # layer_name = 'conv2d_25'
        # intermediate_layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer(layer_name).output)
        # outputs_before_softmax = intermediate_layer_model.predict(test_ds, verbose=1)

        actual_labels = DataFactory.get_labels(test_ds)

        # Use reduction type 'None' to create array of losses for each prediction
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        cce_losses = cce(actual_labels, softmax_scores).numpy()

        # Both actual and predicted labels are in one-hot vector form
        actual_labels = [np.argmax(x) for x in actual_labels]
        predictions = [np.argmax(x) for x in softmax_scores]
        prediction_losses = [x for x in cce_losses]

        return actual_labels, predictions, prediction_losses, softmax_scores

    def _predict_and_save(self, test_ds, filenames, output_file):
        true_labels, predicted_labels, losses, softmax_scores = self.__predict_frames(test_ds=test_ds)
        df = pd.DataFrame(list(zip(filenames, true_labels, predicted_labels, losses, softmax_scores)),
                          columns=["File", "True Label", "Predicted Label", "Loss", "Softmax Scores"])
        df.to_csv(output_file, index=False)

        # Add the homogeneity score computations to the data_frame
        # df = self.__compute_homogeneity_score(df)
        df.to_csv(output_file, index=False)

        return output_file

    def __compute_homogeneity_score(self, pred_df):
        homo_df = pd.read_csv(self.homogeneity_csv)
        pred_df = pd.merge(homo_df, pred_df, on='File')
        pred_df = pred_df.loc[:, ~pred_df.columns.str.contains('^Unnamed')]
        return pred_df
