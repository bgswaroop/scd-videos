import argparse
import shutil
from pathlib import Path

from matplotlib import pyplot as plt


def make_combined_plots_horizontal(plots):
    plt.figure()
    fig, axs = plt.subplots(2, 1, figsize=(7, 5), dpi=300, sharex=True)

    plt.rcParams.update({'font.size': 9})
    num_epochs_after_max = 100

    palette = {'MobileNet (VISION)': ('tab:green', '-'),
               'MobileNet - Constrained (VISION)': ('tab:green', '--'),
               'ResNet (VISION)': ('tab:orange', '-'),
               'ResNet - Constrained (VISION)': ('tab:orange', '--'),
               'MobileNet (QUFVD)': ('tab:purple', '-'),
               'MobileNet - Constrained (QUFVD)': ('tab:purple', '--'),
               }

    for plot_name in plots:
        if plots[plot_name].exists():
            color = palette[plot_name][0]
            style = palette[plot_name][1]
            with open(plots[plot_name], 'r') as f:
                lines = sorted(f.readlines()[2:])
            test_loss = [float(x.split(',')[3]) for x in lines]
            test_acc = [float(x.split(',')[1]) for x in lines]

            max_elem = max(zip(test_acc, [-x for x in test_loss]))
            max_elem = (max_elem[0], -max_elem[1])
            max_acc_indices = set([i for i, x in enumerate(test_acc) if x == max_elem[0]])
            min_loss_indices = set([i for i, x in enumerate(test_loss) if x == max_elem[1]])
            index = min(max_acc_indices.intersection(min_loss_indices))

            test_loss = test_loss[:index + num_epochs_after_max]
            test_acc = test_acc[:index + num_epochs_after_max]
            epochs = list(range(1, len(test_acc) + 1))

            axs[0].plot(epochs, test_acc, label=plot_name, alpha=0.7, color=color, linestyle=style)
            color = axs[0].lines[-1].get_color()
            axs[0].scatter(index + 1, test_acc[index], color=color, s=12)

    for plot_name in plots:
        if plots[plot_name].exists():
            color = palette[plot_name][0]
            style = palette[plot_name][1]
            with open(plots[plot_name], 'r') as f:
                lines = sorted(f.readlines()[2:])
            test_loss = [float(x.split(',')[3]) for x in lines]
            test_acc = [float(x.split(',')[1]) for x in lines]

            max_elem = max(zip(test_acc, [-x for x in test_loss]))
            max_elem = (max_elem[0], -max_elem[1])
            max_acc_indices = set([i for i, x in enumerate(test_acc) if x == max_elem[0]])
            min_loss_indices = set([i for i, x in enumerate(test_loss) if x == max_elem[1]])
            index = min(max_acc_indices.intersection(min_loss_indices))

            test_loss = test_loss[:index + num_epochs_after_max]
            test_acc = test_acc[:index + num_epochs_after_max]
            epochs = list(range(1, len(test_loss) + 1))

            axs[1].plot(epochs, test_loss, label=plot_name, alpha=0.7, color=color, linestyle=style)
            color = axs[1].lines[-1].get_color()
            axs[1].scatter(index + 1, test_loss[index], color=color, s=12)

    # plt.suptitle('Epoch-wise accuracy and loss on the test set')

    # axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('Validation  accuracy')
    # axs[0].set_ylim([0.20, 0.70])
    # axs[0].set_xlim([0, 20])

    axs[1].set_xlabel('epochs')
    axs[1].set_ylabel('Validation  loss')
    axs[1].set_yscale('log')
    # axs[1].set_ylim([0.20, 0.37])
    axs[1].set_xlim([0, 41])

    plt.subplots_adjust(wspace=0.05, hspace=0.08)

    # plt.xlabel('epochs')

    # axes = plt.gca()
    # plt.ylabel('accuracy')
    # axes.set_ylim([0.40, 0.74])

    # plt.title('Epoch-wise loss on test data')
    # plt.ylabel('loss')
    # axes.set_ylim([0.218, 0.31])

    # axs[1].legend(loc='upper right')
    axs[0].grid(linewidth=0.2, axis='x', linestyle='--')
    axs[1].grid(linewidth=0.2, axis='x', linestyle='--')

    plt.legend(bbox_to_anchor=(0.98, 1.45), loc='upper right', borderaxespad=0., framealpha=0.95)
    # plt.tight_layout()

    plt.show()
    plt.cla()
    plt.clf()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_summary', type=Path, help='Path to validation summary')
    args = parser.parse_args()

    # plot_data = {'MobileNet - Constrained': args.val_summary}
    # make_combined_plots_horizontal(plot_data)

    plot_data = {
        'MobileNet (VISION)': Path(
            r'/scratch/p288722/runtime_data/scd_videos_first_revision/06_I_frames_bs64/'
            r'50_frames/mobile_net/models/MobileNet_50_I_frames_ccrop_run2/predictions_50_frames_val/'
            r'videos/V_prediction_stats.csv'),
        'MobileNet (QUFVD)': Path(
            r'/scratch/p288722/runtime_data/scd_videos_first_revision/'
            r'14_qufvd/all_frames/mobile_net/models/MobileNet_all_I_frames_ccrop_run1/'
            r'predictions_all_frames/videos/V_prediction_stats.csv'),
        'ResNet (VISION)': Path(
            r'/scratch/p288722/runtime_data/scd_videos_first_revision/06_I_frames_bs32/'
            r'50_frames/res_net/models/ResNet_50_I_frames_ccrop_run2/predictions_50_frames_val/'
            r'videos/V_prediction_stats.csv'),
        'MobileNet - Constrained (VISION)': Path(
            r'/scratch/p288722/runtime_data/scd_videos_first_revision/11_constraints_bs64/'
            r'50_frames/mobile_net/models/MobileNet_50_I_frames_ccrop_run1_Const/'
            r'predictions_50_frames_val/videos/V_prediction_stats.csv'),
        'MobileNet - Constrained (QUFVD)': Path(
            r'/scratch/p288722/runtime_data/scd_videos_first_revision/'
            r'14_qufvd/all_frames/mobile_net/models/'
            r'MobileNet_all_I_frames_ccrop_run1_Const/predictions_all_frames/'
            r'videos/V_prediction_stats.csv'),
        'ResNet - Constrained (VISION)': Path(
            r'/scratch/p288722/runtime_data/scd_videos_first_revision/11_constraints_bs64/'
            r'50_frames/res_net/models/ResNet_50_I_frames_ccrop_run1_Const/'
            r'predictions_50_frames_val/videos/V_prediction_stats.csv'),

    }
    make_combined_plots_horizontal(plot_data)
