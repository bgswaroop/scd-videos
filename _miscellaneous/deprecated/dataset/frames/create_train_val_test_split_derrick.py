import argparse
import json
import random
from pathlib import Path

import math
from tqdm import tqdm


def extract_frames(args):
    output_dir = args.dest_dir.joinpath(f'bal_all_frames') if args.num_frames == -1 else args.dest_dir.joinpath(
        f'bal_{args.num_frames}_frames')
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir.joinpath('train.json')
    val_file = output_dir.joinpath('val.json')
    test_file = output_dir.joinpath('test.json')

    random.seed(108)

    train_dict = {}
    val_dict = {}
    test_dict = {}

    def __get_frames(videos, num_frames):
        all_frames = []
        for video in videos:
            frames = sorted([str(x) for x in video.glob('*')])
            if num_frames == -1:
                selected_frames = frames
            else:
                selected_frames = []
                for index in range(0, len(frames), int(math.floor(len(frames) / num_frames))):
                    selected_frames.append(frames[index])
                selected_frames = selected_frames[:num_frames]
            all_frames.extend(selected_frames)

        return all_frames

    def __get_200_frames(videos, ref_dir):
        all_frames = []
        ref_frame_names = [x.stem for x in ref_dir.glob('*')]
        for v in videos:
            selected_frames = sorted([str(x) for x in v.glob('*') if x.stem in ref_frame_names])
            all_frames.extend(selected_frames)
        return all_frames

    validation_set_1 = {}
    validation_set_2 = {}
    remaining_videos_dict = {}

    all_scenarios = ['_flat_', '_indoor_', '_outdoor_', '_flatWA_', '_indoorWA_', '_outdoorWA_',
                     '_flatYT_', '_indoorYT_', '_outdoorYT_']

    for device in tqdm(sorted(list(args.frames_dataset.glob('*')))):
        test_dict[device.name] = []

        train_videos, val_videos, test_videos = [], [], []

        all_videos = list(device.glob('*'))
        random.shuffle(all_videos)

        ref_train_dir = Path(r'/data/p288722/from_f118170/vbdi/datasets/Exp.III/balanced_ds_28D/train')
        ref_test_dir = Path(r'/data/p288722/from_f118170/vbdi/datasets/Exp.III/balanced_ds_28D/test')

        # 7 native videos in train
        reference_train_videos = set([x.name.split('-')[0] for x in ref_train_dir.glob(f'{device.name}/*.jpg')])
        train_videos.extend(sorted([x for x in all_videos if x.name in reference_train_videos]))

        # 6 native videos in test
        reference_test_videos = set([x.name.split('-')[0] for x in ref_test_dir.glob(f'{device.name}/*.jpg')])
        test_videos.extend(sorted([x for x in all_videos if x.name in reference_test_videos]))

        # 3 native videos in val
        remaining_videos = list(set(all_videos).difference(train_videos).difference(test_videos))
        random.shuffle(remaining_videos)

        # Create a dictionary of remaining videos
        remaining_videos_dict[device.name] = {
            scenario: [x for x in remaining_videos if scenario in str(x)] for scenario in all_scenarios
        }
        # Add upto 1 video per scenario in val_1
        validation_set_1[device.name] = {
            scenario: remaining_videos_dict[device.name][scenario][:1] for scenario in all_scenarios
        }
        # Update the remaining videos
        for scenario in all_scenarios:
            remaining_videos_dict[device.name][scenario] = list(
                set(remaining_videos_dict[device.name][scenario]).difference(validation_set_1[device.name][scenario])
            )
        # Add upto 1 video per scenario in val_2
        validation_set_2[device.name] = {
            scenario: remaining_videos_dict[device.name][scenario][:1] for scenario in all_scenarios
        }

        # remaining_videos = remaining_videos[:6]

        # flat_remaining = [x for x in remaining_videos if 'flat' in str(x)]
        # indoor_remaining = [x for x in remaining_videos if 'indoor' in str(x)]
        # outdoor_remaining = [x for x in remaining_videos if 'outdoor' in str(x)]
        # print(f'{device.name}, {len(flat_remaining)}, {len(indoor_remaining)}, {len(outdoor_remaining)}')

        # flat_native_remaining = [x for x in remaining_videos if '_flat_' in str(x)]
        # flat_WA_remaining = [x for x in remaining_videos if '_flatWA_' in str(x)]
        # flat_YT_remaining = [x for x in remaining_videos if '_flatYT_' in str(x)]
        # print(f'{len(flat_native_remaining)}, {len(flat_WA_remaining)}, {len(flat_YT_remaining)}')

        # indoor_native_remaining = [x for x in remaining_videos if '_indoor_' in str(x)]
        # indoor_WA_remaining = [x for x in remaining_videos if '_indoorWA_' in str(x)]
        # indoor_YT_remaining = [x for x in remaining_videos if '_indoorYT_' in str(x)]
        # print(f'{len(indoor_native_remaining)}, {len(indoor_WA_remaining)}, {len(indoor_YT_remaining)}')

        # outdoor_native_remaining = [x for x in remaining_videos if '_outdoor_' in str(x)]
        # outdoor_WA_remaining = [x for x in remaining_videos if '_outdoorWA_' in str(x)]
        # outdoor_YT_remaining = [x for x in remaining_videos if '_outdoorYT_' in str(x)]
        # print(f'{len(outdoor_native_remaining)}, {len(outdoor_WA_remaining)}, {len(outdoor_YT_remaining)}')

        # val_videos.extend(remaining_videos[:6])
        #
        # update the dictionary by choosing num_frames
        if args.num_frames == 200:
            train_dict[device.name] = __get_200_frames(train_videos, ref_train_dir.joinpath(device.name))
            test_dict[device.name] = __get_200_frames(test_videos, ref_test_dir.joinpath(device.name))
        else:
            train_dict[device.name] = __get_frames(train_videos, args.num_frames)
            test_dict[device.name] = __get_frames(test_videos, args.num_frames)

        # # Unbalanced Validation set
        # unbal_val_dict[device.name] = __get_frames(val_videos, num_frames)

        print(f'{device.name} : train - {len(train_videos)}, test - {len(test_videos)}, val - {len(val_videos)}')

    validation_videos_1 = {}
    for device in validation_set_1:
        for scenario in validation_set_1[device]:
            video = validation_set_1[device][scenario]
            if video:
                if scenario in validation_videos_1:
                    validation_videos_1[scenario].extend(video)
                else:
                    validation_videos_1[scenario] = video

    validation_videos_2 = {}
    for device in validation_set_2:
        for scenario in validation_set_2[device]:
            video = validation_set_2[device][scenario]
            if video:
                if scenario in validation_videos_2:
                    validation_videos_2[scenario].extend(video)
                else:
                    validation_videos_2[scenario] = video

    validation_videos = []
    for scenario in validation_videos_2:
        x = validation_videos_1[scenario]
        y = validation_videos_2[scenario][: 39 - len(validation_videos_1[scenario])]
        validation_videos.extend(x + y)

    for device in tqdm(sorted(list(args.frames_dataset.glob('*')))):
        val_videos = [x for x in validation_videos if device.name in str(x)]
        val_dict[device.name] = __get_frames(val_videos, args.num_frames)

    with open(train_file, 'w') as f:
        f.write(json.dumps(train_dict, indent=2))
    with open(test_file, 'w') as f:
        f.write(json.dumps(test_dict, indent=2))
    with open(val_file, 'w') as f:
        f.write(json.dumps(val_dict, indent=2))


def run_flow():
    """
        This script creates a balanced dataset for 28 devices.
        For each device 6x3 videos are included in train, followed by test and 1x3 in validation set
    """

    parser = argparse.ArgumentParser(
        description='Balance datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_frames', type=int, required=True, help='enter num frames to copy')
    parser.add_argument('--dest_path', type=str, required=True, help='enter the destination path')
    parser.add_argument('--source_path', type=str, required=True, help='enter the source path')

    args = parser.parse_args()

    extract_frames(args)


if __name__ == "__main__":
    run_flow()
