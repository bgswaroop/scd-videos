from pathlib import Path
from matplotlib import pyplot as plt
import tensorflow as tf
from tqdm import tqdm


def classwise_distribution_of_videos():
    dataset_root = Path(r'/data/p288722/datasets/qufvd/IFrameForEvalution20Class')
    plt.figure()
    for split in ['Training', 'Validation', 'Testing']:
        devices = dataset_root.joinpath(f'FrameDatabase{split}').glob('*/*')
        x, y = [], []
        for device in devices:
            class_id = int(device.name.split('Device')[-1])
            files = list(device.glob('*'))
            video_ids = {str(x).split('-')[-3] for x in files}
            videos_count = len(video_ids)

            x.append(class_id)
            y.append(videos_count)
        plt.plot(x, y, label=f'{split}')
        print(f'{split} videos per device: {y}')

    plt.legend()
    plt.xlabel('Device ID')
    plt.ylabel('Count')
    plt.title('Distribution of videos in QUFVD dataset')
    plt.tight_layout()
    plt.show()


def distribution_of_I_frames():
    dataset_root = Path(r'/data/p288722/datasets/qufvd/IFrameForEvalution20Class')
    plt.figure()
    for split, align in zip(['Training', 'Validation', 'Testing'], ['left', 'mid', 'right']):
        devices = dataset_root.joinpath(f'FrameDatabase{split}').glob('*/*')
        x = []
        for device in devices:
            files = list(device.glob('*'))
            video_ids = {str(x).split('-')[-3] for x in files}
            i_frame_counts = [len(list(device.glob(f'*{x}*'))) for x in video_ids]

            x.extend(i_frame_counts)

        plt.hist(x=x, bins=100, label=f'{split}', align=align, rwidth=10, stacked=True, log=True)

    plt.legend()
    plt.xlabel('Number of I-frames')
    plt.ylabel('Count')
    plt.title('Distribution of I-frames in QUFVD dataset')
    plt.tight_layout()
    plt.show()


def distribution_of_I_frame_orientation():
    dataset_root = Path(r'/data/p288722/datasets/qufvd/IFrameForEvalution20Class')
    from collections import Counter
    plt.figure()

    # distribution = []
    for split in ['Training', 'Validation', 'Testing']:
        filepaths = list(dataset_root.joinpath(f'FrameDatabase{split}').glob('*/*/*'))

        img_sizes = [str(tf.image.decode_jpeg(tf.io.read_file(str(x))).get_shape().as_list()) for x in tqdm(filepaths)]
        print(f'Number of {split} images: {len(img_sizes)}')
        # distribution = Counter(img_sizes)

    plt.hist(x=img_sizes, bins=100, label=f'{split}', rwidth=10, stacked=True, log=True)
    # No. train         - 49,029
    # No. validation    - 12,191
    # No. test          - 15,311

    plt.xlabel('Number of I-frames')
    plt.ylabel('Count')
    plt.title('Distribution of I-frames in QUFVD dataset')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    classwise_distribution_of_videos()
    # distribution_of_I_frames()
    # distribution_of_I_frame_orientation()
