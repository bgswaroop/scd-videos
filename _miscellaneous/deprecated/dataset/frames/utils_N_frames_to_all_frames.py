import json
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import Counter
# from dataset.data_factory import get_glcm_properties, GlcmProperties


def n_frames_to_all_frames():
    source_split_file = r'/scratch/p288722/datasets/vision/old_baseline_split/bal_50_frames/test.json'
    dest_split_file = r'/scratch/p288722/datasets/vision/old_baseline_split/bal_all_frames/test.json'

    with open(source_split_file) as f:
        n_frames_ds = json.load(f)

    all_frames_ds = {}
    for device in sorted(n_frames_ds):
        videos = sorted(set([Path(x).parent for x in n_frames_ds[device]]))
        all_frames_ds[device] = []
        for video in videos:
            all_frames_ds[device].extend([str(x) for x in sorted(video.glob('*.png'))])

    with open(dest_split_file, 'w+') as f:
        json.dump(all_frames_ds, f, indent=2)


def all_frames_to_n_frames(n=50, max_threshold=0.85):
    dataset_root = r'/scratch/p288722/datasets/vision/'
    for split in ['train', 'val', 'test']:
        source_split_file = dataset_root + rf'old_baseline_split/bal_all_I_frames/{split}.json'
        dest_split_file = dataset_root + rf'/old_baseline_split/bal_{n}_I_frames/{split}.json'
        df = pd.read_csv(dataset_root + rf'old_baseline_split/bal_all_I_frames/frame_properties.csv')

        with open(source_split_file) as f:
            all_frames_ds = json.load(f)

        n_frames_ds = {}
        count = 0
        np.random.RandomState(108)

        videos_insufficient_frames = []
        for device in tqdm(sorted(all_frames_ds)):
            videos = sorted(set([Path(x).parent for x in all_frames_ds[device]]))
            count += len(videos)
            n_frames_ds[device] = []
            for video in videos:
                # get a dataframe with all the frames belonging to the current video
                video_df = df[df['File'].str.contains(str(video))]
                # homo_df = video_df[(video_df['mean_energy'] <= max_threshold)]
                if len(video_df) >= n:
                    homo_df = video_df.sample(n=n)
                else:
                    homo_df = video_df
                # homo_df = video_df.sort_values(by='mean_energy', ascending=True)[:n]
                # if len(homo_df) != n:
                #     homo_df = video_df.sort_values(by='max_homogeneity', ascending=True)[:n]
                if len(homo_df) != n:
                    print(f'WARNING: Insufficient number of frames for video - {video}')
                n_frames_ds[device].extend([x for x in sorted(homo_df['File'].values)])

        print(f'Total number of videos - {count}')
        print(f'Number of under represented videos - {len(videos_insufficient_frames)}')
        print(videos_insufficient_frames)

        Path(dest_split_file).parent.mkdir(exist_ok=True, parents=True)
        with open(dest_split_file, 'w+') as f:
            json.dump(n_frames_ds, f, indent=2)


# def compute_homogeneity_scores_for_50_frames_dataset():
#     # source_split_dir = Path(r'/scratch/p288722/datasets/vision/old_baseline_split/bal_all_frames/')
#     source_split_dir = Path(r'/scratch/p288722/datasets/vision/homo_frames_split/bal_50_frames_max_glcm_0.85')
#     csv_file = source_split_dir.joinpath('homogeneity_scores.csv')
#
#     glcm_properties = []
#     pool = Pool(cpu_count())
#     for split in ['test', 'val', 'train']:
#         with open(source_split_dir.joinpath(f'{split}.json')) as f:
#             n_frames_ds = json.load(f)
#         for device in tqdm(sorted(n_frames_ds)):
#             frames = n_frames_ds[device]
#             glcm_properties += pool.map(get_glcm_properties, frames)
#             df = pd.DataFrame(glcm_properties)
#             df.to_csv(str(csv_file))
#     pool.close()
#     # df.to_csv(str(csv_file))


# def compute_homogeneity_scores_for_the_entire_dataset():
#     # source_split_dir = Path(r'/scratch/p288722/datasets/vision/old_baseline_split/bal_all_frames/')
#     # csv_file = source_split_dir.joinpath('homogeneity_scores.csv')
#     #
#     # df = pd.DataFrame({
#     #     'File': [],
#     #     'mean_homogeneity': [],
#     #     'max_homogeneity': [],
#     # })
#     # all_frames = []
#     # all_h_scores = []
#     #
#     # if csv_file.exists():
#     #     df = pd.read_csv(str(csv_file))
#     #     all_frames = list(df['File'].values)
#     #     all_h_scores = list(zip(df['mean_homogeneity'].values, df['max_homogeneity'].values))
#     #
#     # pool = Pool(cpu_count())
#     # for split in ['train', 'val', 'test']:
#     #     with open(source_split_dir.joinpath(f'{split}.json')) as f:
#     #         n_frames_ds = json.load(f)
#     #     for device in tqdm(sorted(n_frames_ds)):
#     #         frames = n_frames_ds[device]
#     #
#     #         if not set(all_frames).intersection(frames):
#     #             h_scores = pool.map(get_glcm_properties, frames)  # fixme: This will no longer work
#     #             all_frames.extend(frames)
#     #             all_h_scores.extend(h_scores)
#     #
#     #             df = pd.DataFrame({
#     #                 'File': all_frames,
#     #                 'mean_homogeneity': [x[0] for x in all_h_scores],
#     #                 'max_homogeneity': [x[1] for x in all_h_scores],
#     #             })
#     #             df.to_csv(str(csv_file))
#     # pool.close()
#     # df.to_csv(str(csv_file))
#     import argparse
#     import math
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--task_id', type=int, default=0, required=False)
#     args = parser.parse_args()
#
#     num_parts = 200
#     assert 0 <= args.task_id <= num_parts - 1
#
#     print(f'Current task id - {args.task_id}\n')
#
#     source_split_dir = Path(r'/scratch/p288722/datasets/vision/old_baseline_split/bal_all_I_frames')
#     # csv_file = source_split_dir.joinpath(f'frame_properties_{str(args.task_id).zfill(3)}.csv')
#     csv_file = source_split_dir.joinpath(f'frame_properties.csv')
#
#     files = []
#     for split in ['test', 'val', 'train']:
#         with open(source_split_dir.joinpath(f'{split}.json')) as f:
#             n_frames_ds = json.load(f)
#             files += [y for x in n_frames_ds for y in n_frames_ds[x]]
#
#     # Parallel (multiple jobs)
#     # chunks = [files[x:x + len(files) // num_parts] for x in range(0, len(files), math.ceil(len(files) / num_parts))]
#     # glcm_properties = map(get_glcm_properties, chunks[args.task_id])
#
#     # Parallel (single job)
#     pool = Pool(cpu_count())
#     glcm_properties = pool.map(get_glcm_properties, files)
#     pool.close()
#
#     df = pd.DataFrame(glcm_properties)
#     df.to_csv(str(csv_file))


def predict_N_most_homogeneous(
        csv_file=r'/scratch/p288722/runtime_data/scd_videos_first_revision/01_fine_tune/all_frames_pred/mobile_net/'
                 r'models/MobileNet/predictions_all_frames/frames/fm-e00016_F_predictions.csv'):
    df_frame_predictions = pd.read_csv(csv_file)

    # mis_classified = df_frame_predictions[df_frame_predictions['True Label'] != df_frame_predictions['Predicted Label']]
    # correctly_classified = df_frame_predictions[
    #     df_frame_predictions['True Label'] == df_frame_predictions['Predicted Label']]
    #
    # from matplotlib import pyplot as plt
    # plt.figure()
    # plt.hist(mis_classified['homogeneity_score_max'], bins=1000, alpha=0.5, log=False, color='r',
    #          label='mis-classifications')
    # plt.hist(correctly_classified['homogeneity_score_max'], bins=1000, alpha=0.5, log=False, color='g',
    #          label='correct classification')
    # plt.title('Distribution of predictions')
    # plt.ylabel('Count')
    # plt.xlabel('max homogeneity score ')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    # plt.close()
    #
    # plt.figure()
    # plt.hist(mis_classified['homogeneity_score_mean'], bins=1000, alpha=0.5, log=False, color='r',
    #          label='mis-classifications')
    # plt.hist(correctly_classified['homogeneity_score_mean'], bins=1000, alpha=0.5, log=False, color='g',
    #          label='correct classification')
    # plt.title('Distribution of predictions')
    # plt.ylabel('Count')
    # plt.xlabel('mean homogeneity score ')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    # plt.close()

    # df_frame_predictions = compute_homogeneity_score(df_frame_predictions)
    # df_frame_predictions.to_csv(csv_file)

    videos = df_frame_predictions["File"].str.split("-").str[0].unique()
    print(f"Total number of videos to classify: {len(videos)}.")
    range_mid_point = 0.5389

    video_predictions = []
    for video in videos:
        # Get frame predictions for current video
        df_video_predictions = df_frame_predictions[df_frame_predictions["File"].str.contains(video)]
        class_idx = df_video_predictions["True Label"].iloc[0]
        n_frames = len(df_video_predictions)

        # Majority Vote
        # df_device_vote = df_video_predictions["Predicted Label"].value_counts().rename_axis('class').reset_index(
        #     name='vote_count')

        print(' ')
        n = 50  # num_frames_to_consider
        temp_df = df_video_predictions.sort_values(by='homogeneity_score_mean', ascending=False)[:n]
        df_device_vote = temp_df["Predicted Label"].value_counts().rename_axis('class').reset_index(name='vote_count')

        temp_df = df_video_predictions.sort_values(by='homogeneity_score_max', ascending=False)[:n]
        df_device_vote = temp_df["Predicted Label"].value_counts().rename_axis('class').reset_index(name='vote_count')

        distance = np.abs(df_video_predictions['homogeneity_score_mean'] - range_mid_point)
        df_video_predictions.assign(dist_from_selected_homo_score=distance)
        temp_df = df_video_predictions.sort_values(by='dist_from_selected_homo_score')[:n]
        df_device_vote = temp_df["Predicted Label"].value_counts().rename_axis('class').reset_index(name='vote_count')


def get_ideal_range():
    from matplotlib import pyplot as plt

    csv_file = r'/scratch/p288722/runtime_data/scd_videos_first_revision/01_fine_tune/all_frames_pred/mobile_net/' \
               r'models/MobileNet/predictions_all_frames/frames/fm-e00016_F_predictions.csv'
    df = pd.read_csv(csv_file)
    mis_classified = df[df['True Label'] != df['Predicted Label']]
    correctly_classified = df[df['True Label'] == df['Predicted Label']]

    num_bins = 1000
    false_dist = np.histogram(mis_classified['homogeneity_score_max'], bins=num_bins, range=(0, 1))
    true_dist = np.histogram(correctly_classified['homogeneity_score_max'], bins=num_bins, range=(0, 1))

    score_range = 0.1
    window_length = int(num_bins * score_range)
    diff_count = np.subtract(true_dist[0], false_dist[0])
    # lb = true_dist[1][0:-window_length]
    # ub = true_dist[1][window_length:]
    # c = np.convolve(diff_count, np.ones(window_length, dtype=int), 'valid')

    df_stats = pd.DataFrame({
        'lower_bound': true_dist[1][0:-window_length],
        'upper_bound': true_dist[1][window_length:],
        'mean_diff_count': np.convolve(diff_count, np.ones(window_length, dtype=int), 'valid'),
    })
    selected_range_df = df_stats[df_stats['mean_diff_count'] == df_stats['mean_diff_count'].max()]
    range_mid_point = float((selected_range_df['lower_bound'] + selected_range_df['upper_bound']) / 2)
    print(range_mid_point)


def test_the_validity_of_images():
    import tensorflow as tf
    source_split_file = r'/scratch/p288722/datasets/vision/old_baseline_split/bal_all_frames/test.json'

    with open(source_split_file) as f:
        all_frames_ds = json.load(f)
        all_frames_ds = [y for x in all_frames_ds for y in all_frames_ds[x]]

    for file_path in tqdm(all_frames_ds):
        img = tf.io.read_file(file_path)
        try:
            tf.image.decode_png(img, channels=3)
        except Exception as e:
            print(f'Issue decoding the png image - {file_path}\n')
            raise e


def analyze_data():
    equally_spaced_dir = Path(r'/scratch/p288722/datasets/vision/old_baseline_split/bal_50_frames/')
    equally_spaced_csv_file = equally_spaced_dir.joinpath('homogeneity_scores.csv')
    homo_dir = Path(r'/scratch/p288722/datasets/vision/energy_frames_split/bal_50_mean_energy')
    homo_dir_csv_file = homo_dir.joinpath('frame_properties.csv')

    df1 = pd.read_csv(equally_spaced_csv_file)
    df2 = pd.read_csv(homo_dir_csv_file)

    print(' ')
    # Test set frames
    split = 'test'
    with open(equally_spaced_dir.joinpath(f'{split}.json')) as f:
        ds1 = json.load(f)
        ds1_files = [y for x in ds1 for y in ds1[x]]
    with open(homo_dir.joinpath(f'{split}.json')) as f:
        ds2 = json.load(f)
        ds2_files = [y for x in ds2 for y in ds2[x]]

    df1_flat = df1[df1['File'].str.contains('flat')]
    df1_indoor = df1[df1['File'].str.contains('indoor')]
    df1_outdoor = df1[df1['File'].str.contains('outdoor')]
    df2_flat = df2[df2['File'].str.contains('flat')]
    df2_indoor = df2[df2['File'].str.contains('indoor')]
    df2_outdoor = df2[df2['File'].str.contains('outdoor')]

    from matplotlib import pyplot as plt

    prop = 'mean_energy'
    data_range = None
    log = False
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
    ax1.set_title('Equally spaced frames')
    ax1.hist(df1_flat[prop].values, bins=100, range=data_range, label='flat', color='r', alpha=0.5, log=log)
    ax1.hist(df1_indoor[prop].values, bins=100, range=data_range, label='indoor', color='g', alpha=0.5, log=log)
    ax1.hist(df1_outdoor[prop].values, bins=100, range=data_range, label='outdoor', color='b', alpha=0.5, log=log)
    ax2.set_title('Homogeneous frames (< 0.85)')
    ax2.hist(df2_flat[prop].values, bins=100, range=data_range, label='flat', color='r', alpha=0.5, log=log)
    ax2.hist(df2_indoor[prop].values, bins=100, range=data_range, label='indoor', color='g', alpha=0.5, log=log)
    ax2.hist(df2_outdoor[prop].values, bins=100, range=data_range, label='outdoor', color='b', alpha=0.5, log=log)
    ax1.set_ylabel('count')
    ax1.set_xlabel(prop)
    ax2.set_xlabel(prop)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

    prop = 'mean_contrast'
    data_range = None
    log = False
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
    ax1.set_title('Equally spaced frames')
    ax1.hist(df1_flat[prop].values, bins=100, range=data_range, label='flat', color='r', alpha=0.5, log=log)
    ax1.hist(df1_indoor[prop].values, bins=100, range=data_range, label='indoor', color='g', alpha=0.5, log=log)
    ax1.hist(df1_outdoor[prop].values, bins=100, range=data_range, label='outdoor', color='b', alpha=0.5, log=log)
    ax2.set_title('Homogeneous frames (< 0.85)')
    ax2.hist(df2_flat[prop].values, bins=100, range=data_range, label='flat', color='r', alpha=0.5, log=log)
    ax2.hist(df2_indoor[prop].values, bins=100, range=data_range, label='indoor', color='g', alpha=0.5, log=log)
    ax2.hist(df2_outdoor[prop].values, bins=100, range=data_range, label='outdoor', color='b', alpha=0.5, log=log)
    ax1.set_ylabel('count')
    ax1.set_xlabel(prop)
    ax2.set_xlabel(prop)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

    prop = 'mean_dissimilarity'
    data_range = None
    log = False
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
    ax1.set_title('Equally spaced frames')
    ax1.hist(df1_flat[prop].values, bins=100, range=data_range, label='flat', color='r', alpha=0.5, log=log)
    ax1.hist(df1_indoor[prop].values, bins=100, range=data_range, label='indoor', color='g', alpha=0.5, log=log)
    ax1.hist(df1_outdoor[prop].values, bins=100, range=data_range, label='outdoor', color='b', alpha=0.5, log=log)
    ax2.set_title('Homogeneous frames (< 0.85)')
    ax2.hist(df2_flat[prop].values, bins=100, range=data_range, label='flat', color='r', alpha=0.5, log=log)
    ax2.hist(df2_indoor[prop].values, bins=100, range=data_range, label='indoor', color='g', alpha=0.5, log=log)
    ax2.hist(df2_outdoor[prop].values, bins=100, range=data_range, label='outdoor', color='b', alpha=0.5, log=log)
    ax1.set_ylabel('count')
    ax1.set_xlabel(prop)
    ax2.set_xlabel(prop)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

    prop = 'mean_homogeneity'
    data_range = None
    log = False
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
    ax1.set_title('Equally spaced frames')
    ax1.hist(df1_flat[prop].values, bins=100, range=data_range, label='flat', color='r', alpha=0.5, log=log)
    ax1.hist(df1_indoor[prop].values, bins=100, range=data_range, label='indoor', color='g', alpha=0.5, log=log)
    ax1.hist(df1_outdoor[prop].values, bins=100, range=data_range, label='outdoor', color='b', alpha=0.5, log=log)
    ax2.set_title('Homogeneous frames (< 0.85)')
    ax2.hist(df2_flat[prop].values, bins=100, range=data_range, label='flat', color='r', alpha=0.5, log=log)
    ax2.hist(df2_indoor[prop].values, bins=100, range=data_range, label='indoor', color='g', alpha=0.5, log=log)
    ax2.hist(df2_outdoor[prop].values, bins=100, range=data_range, label='outdoor', color='b', alpha=0.5, log=log)
    ax1.set_ylabel('count')
    ax1.set_xlabel(prop)
    ax2.set_xlabel(prop)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

    prop = 'mean_correlation'
    data_range = None
    log = False
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
    ax1.set_title('Equally spaced frames')
    ax1.hist(df1_flat[prop].values, bins=100, range=data_range, label='flat', color='r', alpha=0.5, log=log)
    ax1.hist(df1_indoor[prop].values, bins=100, range=data_range, label='indoor', color='g', alpha=0.5, log=log)
    ax1.hist(df1_outdoor[prop].values, bins=100, range=data_range, label='outdoor', color='b', alpha=0.5, log=log)
    ax2.set_title('Homogeneous frames (< 0.85)')
    ax2.hist(df2_flat[prop].values, bins=100, range=data_range, label='flat', color='r', alpha=0.5, log=log)
    ax2.hist(df2_indoor[prop].values, bins=100, range=data_range, label='indoor', color='g', alpha=0.5, log=log)
    ax2.hist(df2_outdoor[prop].values, bins=100, range=data_range, label='outdoor', color='b', alpha=0.5, log=log)
    ax1.set_ylabel('count')
    ax1.set_xlabel(prop)
    ax2.set_xlabel(prop)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

    print(' ')


def examine_frame_dimensions():
    equally_spaced_dir = Path(r'/scratch/p288722/datasets/vision/old_baseline_split/bal_50_frames/')
    equally_spaced_csv_file = equally_spaced_dir.joinpath('homogeneity_scores.csv')

    df1 = pd.read_csv(equally_spaced_csv_file)
    df1 = df1[df1['File'].str.contains('00001.png')]  # selecting only the first frame for every video

    df1_flat = df1[df1['File'].str.contains('flat')]
    df1_indoor = df1[df1['File'].str.contains('indoor')]
    df1_outdoor = df1[df1['File'].str.contains('outdoor')]

    flat_counter = Counter([Image.open(x).size for x in df1_flat['File'].values])
    indoor_counter = Counter([Image.open(x).size for x in df1_indoor['File'].values])
    outdoor_counter = Counter([Image.open(x).size for x in df1_outdoor['File'].values])

    labels = sorted(list(flat_counter.keys()) + list(indoor_counter.keys()) + list(outdoor_counter.keys()))
    flat_counts = [flat_counter[x] for x in labels]
    indoor_counts = [indoor_counter[x] for x in labels]
    outdoor_counts = [outdoor_counter[x] for x in labels]

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    plt.figure()
    # rects1 = plt.bar([str(x) for x in labels], flat_counts, width, label='flat')
    # rects2 = plt.bar([str(x) for x in labels], indoor_counts, width, label='indoor')
    rects3 = plt.bar([str(x) for x in labels], outdoor_counts, width, label='outdoor')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('Video count')
    plt.xlabel('Video resolution (width, height)')
    plt.title('Distribution of videos by resolution')
    plt.xticks([str(x) for x in labels], rotation=45)
    plt.legend()

    plt.bar_label(rects3, padding=1)
    # plt.bar(rects2, padding=1)
    # plt.bar(rects3, padding=1)

    plt.tight_layout()
    plt.show()
    plt.close()

    print(' ')

    plt.figure()

    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


def analyze_I_frames_distribution():
    file = Path(r'/scratch/p288722/datasets/vision/old_baseline_split/bal_all_I_frames/train.json')
    with open(file) as f:
        ds = json.load(f)
    print(' ')
    all_frames = [Path(y) for x in ds for y in ds[x]]

    video_labels_native = [x.parent for x in all_frames if ('WA' not in str(x) and 'YT' not in str(x))]
    video_labels_wa = [x.parent for x in all_frames if 'WA' in str(x)]
    video_labels_yt = [x.parent for x in all_frames if 'YT' in str(x)]

    counts_native = Counter(video_labels_native)
    counts_wa = Counter(video_labels_wa)
    counts_yt = Counter(video_labels_yt)

    c_native = Counter(counts_native.values())
    c_wa = Counter(counts_wa.values())
    c_yt = Counter(counts_yt.values())

    plt.figure()
    plt.bar(list(c_native.keys()), c_native.values())
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == "__main__":
    analyze_I_frames_distribution()
    # examine_frame_dimensions()
    # run_flow()
    # predict_N_most_homogeneous()
    # get_ideal_range()
    # compute_homogeneity_scores_for_the_entire_dataset()
    # all_frames_to_n_frames(n=50, max_threshold=0.85)
    # test_the_validity_of_images()
    # compute_homogeneity_scores_for_50_frames_dataset()
    # analyze_data()

    # csv_file = r'/scratch/p288722/datasets/vision/old_baseline_split/bal_all_frames/homogeneity_scores.csv'
    # df = pd.read_csv(csv_file)
    # print(' ')

    # equally_spaced_dir = Path(r'/scratch/p288722/datasets/vision/old_baseline_split/bal_all_frames/')
    # split = 'test'
    # with open(equally_spaced_dir.joinpath(f'{split}.json')) as f:
    #     ds1 = json.load(f)
    #     ds1_files = [y for x in ds1 for y in ds1[x]]

    # print(' ')

    # source_split_dir = Path(r'/scratch/p288722/datasets/vision/old_baseline_split/bal_all_frames')
    # csv_files = sorted(source_split_dir.glob('frame_properties_*.csv'))
    # dfs = []
    # for file in csv_files:
    #     temp_df = pd.read_csv(file)
    #     dfs.append(temp_df)
    # df = pd.concat(dfs)
    # df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # df.to_csv(source_split_dir.joinpath('frame_properties.csv'))
    # print(' ')
