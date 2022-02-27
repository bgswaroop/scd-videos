import sys
from pathlib import Path

import matplotlib
import numpy as np
import tensorflow as tf
from keract import keract
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

sys.path.insert(1, '/home/p288722/git_code/scd-videos')

from dataset.vision.data_factory import DataFactory
from models import Constrained3DKernelMinimal, CombineInputsWithConstraints, PPCCELoss


def load_model(model_path):
    # noinspection PyProtectedMember
    custom_objects = {
        'Constrained3DKernelMinimal': Constrained3DKernelMinimal,
        'CombineInputsWithConstraints': CombineInputsWithConstraints,
        'PPCCELoss': PPCCELoss
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)


def get_constrained_activations():
    maps = []
    for element in eval_ds:
        maps.append(keract.get_activations(model, element[0].numpy(), layer_names=['constrained_layer'],
                                           nodes_to_evaluate=None, output_format='simple', nested=False,
                                           auto_compile=True)['constrained_layer'][0]
                    )
    return maps


def compute_correlation_map(activation_maps):
    num_images = len(activation_maps)
    num_filters = 3
    pcc_map = np.zeros(shape=(num_images, num_images, num_filters))

    # Memoization for PCC Map
    data = np.array([x[1] for x in activation_maps])
    a = data.transpose([0, 3, 1, 2])
    b = a.reshape((num_images, num_filters, -1))

    data = data.transpose([0, 3, 1, 2]).reshape((num_images, num_filters, -1))
    data_std = data.std(axis=2)
    data_mean_sub = data - data.mean(axis=2, keepdims=True)
    num_features = len(data[0, 0, :])

    for c in range(num_filters):
        for x in tqdm(range(0, num_images)):
            for y in range(x + 1, num_images):  # fixme: x+1 is to make diagonals zero (for better visualization)
                pcc_map[y][x][c] = pcc_map[x][y][c] = np.dot(data_mean_sub[x, c], data_mean_sub[y, c]) / (
                        data_std[x, c] * data_std[y, c] * num_features)

    return pcc_map


def plot_correlations(plot_name):
    matplotlib.rcParams.update({'font.size': 15})
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=300, sharey=True, sharex=True)
    # Set the range for the plots
    # v_min = np.min(filters)
    # v_max = np.max(filters)
    v_min = v_max = None
    for i in range(3):
        # Plot the filter weights as an image
        im = axs[i].imshow(pcc[:, :, i], vmin=v_min, vmax=v_max, cmap='viridis')
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    # axs[0].set_ylabel('Constrained Filters')
    axs[0].title.set_text('Channel 1')
    axs[1].title.set_text('Channel 2')
    axs[2].title.set_text('Channel 3')
    plt.tight_layout()
    plt.savefig(str(Path(r'/scratch/p288722/runtime_data/scd-videos/plots').joinpath(plot_name)))
    plt.show()
    plt.cla()
    plt.clf()
    plt.close()


if __name__ == '__main__':
    # 1. Load the dataset
    dataset = DataFactory(input_dir=Path(r'/scratch/p288722/datasets/vision/2_devices/bal_50_frames'),
                          batch_size=1, height=480, width=800)
    filename_ds, eval_ds = dataset.get_tf_evaluation_data(category=None, mode='test')

    # 2a. Load the model
    model = load_model(model_path=Path(r'/scratch/p288722/runtime_data/scd-videos/dev_combine_layer/'
                                       r'200_frames_8d_64/mobile_net/models/ConstNet_guru/fm-e00020.h5'))

    # Weights of the first constrained layer filter
    filters, biases = model.layers[1].get_weights()

    # 3. Run the predictions and save the results (along with class labels)
    activations = get_constrained_activations()

    # 4. Compute pair-wise pearson's co-relation
    labels = [int(str(x.numpy())[3:5]) for x in filename_ds]
    sorted_data = sorted(zip(labels, activations), key=lambda x: -x[0])
    pcc = compute_correlation_map(sorted_data)

    # 5. Plot the results
    plot_correlations(plot_name='pcc_mobile_net_guru_2d_train_set5.png')
