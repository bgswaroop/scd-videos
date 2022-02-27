import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Generate dataset split files')
    parser.add_argument('--all_I_frames_dir', type=Path, help='Input directory of extracted I frames')
    parser.add_argument('--all_frames_dir', type=Path, help='Input directory of extracted frames')
    parser.add_argument('--dest_frame_splits_dir', type=Path, required=True,
                        help='Output directory to save the train, val, and test splits')
    parser.add_argument('--frame_selection', type=str, required=True, choices=['equally_spaced', 'first_N'])
    parser.add_argument('--frame_type', type=str, required=True, choices=['I', 'all'])
    parser.add_argument('--fpv', type=int, required=True, help='max number of frames per video')
    args = parser.parse_args()

    if args.frame_type == 'I':
        assert args.all_I_frames_dir.exists(), 'Input directory does not exists!'
    elif args.frame_type == 'all':
        assert args.all_frames_dir.exists(), 'Input directory does not exists!'

    args.dest_frame_splits_dir.mkdir(parents=True, exist_ok=True)
    return args


def frame_selection(args, device, video):
    if args.frame_type == 'I':
        dataset = args.all_I_frames_dir
    elif args.frame_type == 'all':
        dataset = args.all_frames_dir
    all_frames = [str(x) for x in sorted(dataset.joinpath(device).joinpath(video).glob('*'))]

    if args.fpv > len(all_frames):
        print(f'Warning: Fewer than {args.fpv} frames are available for the video: {video}. '
              f'Consists of only {len(all_frames)} frames')

    selected_frames = []
    if args.fpv == -1 or args.fpv > len(all_frames):  # Select all frames
        selected_frames = all_frames
    elif args.frame_selection == 'equally_spaced':
        uniformly_distributed_indices = np.unique(np.linspace(0, len(all_frames), args.fpv, endpoint=False).astype(int))
        selected_frames = [all_frames[x] for x in uniformly_distributed_indices][:args.fpv]
    elif args.frame_selection == 'first_N':
        selected_frames = all_frames[:args.fpv]

    return selected_frames


def get_frames_dataset(split, args):
    video_level_split = Path(__file__).resolve().parent.joinpath(f'split/{split}_videos.json')
    with open(video_level_split) as f:
        videos_per_device = json.load(f)

    frames_per_device = {}
    for device in sorted(videos_per_device):
        selected_frames = []
        for video in videos_per_device[device]:
            selected_frames.extend(frame_selection(args, device, video))
        frames_per_device[device] = selected_frames

    return frames_per_device


def generate_dataset_split_files(args):
    for split in ['train', 'val', 'test']:
        frames_per_device = get_frames_dataset(split, args)
        frame_split = args.dest_frame_splits_dir.joinpath(f'{split}_frames.json')
        with open(frame_split, 'w+') as f:
            json.dump(frames_per_device, f, indent=2)


def are_two_frame_splits_same(split1, split2):
    with open(split1) as f1, open(split2) as f2:
        split1 = json.load(f1)
        split2 = json.load(f2)

    for device1, device2 in zip(sorted(split1), sorted(split2)):
        elements = set(split1[device1]).symmetric_difference(split2[device2])
        if len(elements) != 0:
            return False
    return True


def run_flow():
    args = parse_args()
    generate_dataset_split_files(args)


if __name__ == '__main__':
    run_flow()
    print(are_two_frame_splits_same(
        split1=r'/scratch/p288722/datasets/vision/splits/I_frames/train_frames.json',
        split2=r'/scratch/p288722/datasets/vision/I_frame_splits/bal_50_frames/train.json'
    ))
