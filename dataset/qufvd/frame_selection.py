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
    models = args.all_I_frames_dir.joinpath(rf'FrameDatabase{split}')
    from collections import OrderedDict
    frames_per_device = {}
    for model in models.glob('*'):
        for device in model.glob('*'):
            filepaths = [str(x) for x in sorted(device.glob('*'))]
            classname = f'{model.name}_{device.name}'
            frames_per_device[classname] = filepaths
            # todo: implement frame selection (default select all frames)
    sorted_devices = sorted(frames_per_device, key=lambda x: int(x.split('Device')[-1]))
    frames_per_device = OrderedDict({x: frames_per_device[x] for x in sorted_devices})

    return frames_per_device


def generate_dataset_split_files(args):
    for split in ['Training', 'Validation', 'Testing']:
        frames_per_device = get_frames_dataset(split, args)
        frame_split = args.dest_frame_splits_dir.joinpath(f'{split}_frames.json')
        with open(frame_split, 'w+') as f:
            json.dump(frames_per_device, f, indent=2)


def run_flow():
    args = parse_args()
    generate_dataset_split_files(args)


if __name__ == '__main__':
    run_flow()
