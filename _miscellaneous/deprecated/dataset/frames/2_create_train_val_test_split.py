import argparse
import json
import math
import random
from pathlib import Path

from tqdm import tqdm

if __name__ == "__main__":
    """
    This script creates a balanced dataset.
    The dataset consists of 28 camera devices. The videos belonging to each device can be categorized into three 
    scenarios - Flat, Indoor, and Outdoor. Furthermore, there can be three possible versions of each video - 
    Native, WhatsApp (WA), and YouTube (YT). Note that, Native versions are present for all videos. The corresponding
    WA and YT versions may not always be present in the dataset.
    
    This script creates a balanced dataset as follows:
    For each device:
    - Use 6 native videos (2 Flat, 2 Indoor, and 2 Outdoor) for train + corresponding WA and YT versions
    - Use 6 native videos (2 Flat, 2 Indoor, and 2 Outdoor) for test + corresponding WA and YT versions
    - Use 3 native videos (1 Flat, 1 Indoor, and 1 Outdoor) for validation + corresponding WA and YT versions
    
    Note - The above scheme is subject to availability of the videos in the VISION dataset. This strategy of dataset
    split ensures that the videos are equally distributed both with respect to scenarios and compression types.
    """

    parser = argparse.ArgumentParser(
        description='Balance datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_frames', type=int, required=True,
                        help='Number of frames to sample from each video. A special value of "-1" indicates to '
                             'copy all the available frames (results in imbalanced data)')
    parser.add_argument('--dest_path', type=Path, required=True, help='enter the destination path')
    parser.add_argument('--frames_dataset', type=Path, default=Path(r'/scratch/p288722/datasets/vision/all_I_frames'))

    args = parser.parse_args()
    num_frames = args.num_frames
    dest_dir = args.dest_path
    frames_dataset = args.frames_dataset

    assert num_frames == -1 or num_frames > 0, "Invalid number of frames"
    assert frames_dataset.exists(), "frames_dataset doesn't exists"

    output_dir = dest_dir.joinpath(f'bal_all_frames') if num_frames == -1 else dest_dir.joinpath(
        f'bal_{num_frames}_frames')
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir.joinpath('train.json')
    val_file = output_dir.joinpath('val.json')
    test_file = output_dir.joinpath('test.json')

    random.seed(108)

    train_frame_paths = {}
    val_frame_paths = {}
    test_frame_paths = {}


    def __count_missing_WA_YT_videos(x):
        if 'flat' in x.name:
            missing_vids_count = 2 - int(x.parent.joinpath(x.name.replace('_flat_', '_flatWA_')).exists()) - int(
                x.parent.joinpath(x.name.replace('_flat_', '_flatYT_')).exists())
        elif 'indoor' in x.name:
            missing_vids_count = 2 - int(x.parent.joinpath(x.name.replace('_indoor_', '_indoorWA_')).exists()) - int(
                x.parent.joinpath(x.name.replace('_indoor_', '_indoorYT_')).exists())
        elif 'outdoor' in x.name:
            missing_vids_count = 2 - int(x.parent.joinpath(x.name.replace('_outdoor_', '_outdoorWA_')).exists()) - int(
                x.parent.joinpath(x.name.replace('_outdoor_', '_outdoorYT_')).exists())
        else:
            raise ValueError('Unable to determine the video scenario')
        return missing_vids_count


    def add_WA_YT_versions(native_videos):
        extended_videos = []
        for v in native_videos:
            for scenario in ['flat', 'indoor', 'outdoor']:
                if scenario in v.name:
                    break
            if v.parent.joinpath(v.name.replace(f'_{scenario}_', f'_{scenario}WA_')).exists():
                extended_videos.append(v.parent.joinpath(v.name.replace(f'_{scenario}_', f'_{scenario}WA_')))
            if v.parent.joinpath(v.name.replace(f'_{scenario}_', f'_{scenario}YT_')).exists():
                extended_videos.append(v.parent.joinpath(v.name.replace(f'_{scenario}_', f'_{scenario}YT_')))

        return native_videos + extended_videos


    def frame_selection(videos, num_frames_to_select):
        all_frames = []
        for video_path in videos:
            frames = sorted([str(x) for x in video_path.glob('*')])

            if num_frames_to_select == -1:
                # Select all the available frames
                selected_frames = frames
            else:
                available_frames_count = len(frames)
                # assert num_frames_to_select <= available_frames_count
                if num_frames_to_select > available_frames_count:
                    print(f'Warning: Fewer than {num_frames_to_select} frames are available for the '
                          f'video: {video_path.name}. Consists of only {available_frames_count} frames')

                # Select `num_frames_to_select` frames equally spaced in time
                selected_frames = []
                time_interval = max(1, int(math.floor(available_frames_count / num_frames_to_select)))
                for index in range(0, available_frames_count, time_interval):
                    selected_frames.append(frames[index])
                selected_frames = selected_frames[:num_frames_to_select]
            all_frames.extend(selected_frames)

        return all_frames


    for device in tqdm(sorted(list(frames_dataset.glob('*')))):
        val_frame_paths[device.name] = []
        test_frame_paths[device.name] = []

        train_videos, val_videos, test_videos = [], [], []

        all_videos = list(device.glob('*'))
        random.shuffle(all_videos)
        flat = sorted([x for x in all_videos if '_flat_' in x.name], key=__count_missing_WA_YT_videos)
        indoor = sorted([x for x in all_videos if '_indoor_' in x.name], key=__count_missing_WA_YT_videos)
        outdoor = sorted([x for x in all_videos if '_outdoor_' in x.name], key=__count_missing_WA_YT_videos)

        # Add 6 native videos to train from all three scenarios - 2 per scenario
        train_videos.extend(flat[:2])
        train_videos.extend(indoor[:2])
        train_videos.extend(outdoor[:2])

        # Add 6 native videos to test from all three scenarios (wherever possible) - 2 per scenario
        test_videos.extend(flat[2:4])
        test_videos.extend(indoor[2:4])
        test_videos.extend(outdoor[2:4])

        # Add 3 native videos to val from all three scenarios (wherever possible) - 1 per scenario
        val_videos.extend(flat[4:5])
        val_videos.extend(indoor[4:5])
        val_videos.extend(outdoor[4:5])

        # augment with corresponding WA and YT versions
        train_videos = add_WA_YT_versions(train_videos)
        test_videos = add_WA_YT_versions(test_videos)
        val_videos = add_WA_YT_versions(val_videos)

        # update the dictionary by choosing num_frames
        train_frame_paths[device.name] = frame_selection(train_videos, num_frames_to_select=num_frames)
        test_frame_paths[device.name] = frame_selection(test_videos, num_frames_to_select=num_frames)
        val_frame_paths[device.name] = frame_selection(val_videos, num_frames_to_select=num_frames)
        print(f'{device.name} : train - {len(train_videos)}, test - {len(test_videos)}, val - {len(val_videos)}')

    with open(train_file, 'w') as f:
        f.write(json.dumps(train_frame_paths, indent=2))
    with open(test_file, 'w') as f:
        f.write(json.dumps(test_frame_paths, indent=2))
    with open(val_file, 'w') as f:
        f.write(json.dumps(val_frame_paths, indent=2))
