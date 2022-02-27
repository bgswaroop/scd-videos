import argparse
import json
import random
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def extract_homogeneous_patches(image, patch_size, num_patches, min_std_dev_threshold, max_std_dev_threshold):
    homogeneous_patches = []
    non_homogeneous_patches = []

    patch = namedtuple('WindowSize', ['width', 'height'])(*patch_size)
    stride = namedtuple('Strides', ['width_step', 'height_step'])(patch.width // 4, patch.height // 4)
    image_size = namedtuple('ImageSize', ['width', 'height'])(image.shape[1], image.shape[0])
    num_channels = 3

    # Choose the patches
    for row_idx in range(patch.height, image_size.height, stride.height_step):
        for col_idx in range(patch.width, image_size.width, stride.width_step):
            img_patch = image[(row_idx - patch.height):row_idx, (col_idx - patch.width):col_idx]
            std_dev = np.std(img_patch.reshape(-1, num_channels), axis=0)
            if np.prod(np.less_equal(std_dev, max_std_dev_threshold)) and \
                    np.prod(np.greater_equal(std_dev, min_std_dev_threshold)):
                homogeneous_patches.append((std_dev, img_patch, row_idx, col_idx))
            else:
                non_homogeneous_patches.append((std_dev, img_patch, row_idx, col_idx))

    selected_patches = homogeneous_patches
    # Filter out excess patches
    if len(homogeneous_patches) > num_patches:
        random.seed(108)
        indices = random.sample(range(len(homogeneous_patches)), num_patches)
        selected_patches = [homogeneous_patches[x] for x in indices]
    # Add additional patches
    elif len(homogeneous_patches) < num_patches:
        num_additional_patches = num_patches - len(homogeneous_patches)
        non_homogeneous_patches.sort(key=lambda x: np.mean(x[0]))
        selected_patches.extend(non_homogeneous_patches[:num_additional_patches])

    return selected_patches


def extract_patches(frame_paths, num_patches_per_frame, dest_patches_dir):
    dest_patches_dir.mkdir(parents=True, exist_ok=True)
    all_patch_paths = []
    for frame_path in tqdm(frame_paths):
        if len(list(dest_patches_dir.glob(f'{Path(frame_path).stem}*'))) < num_patches_per_frame:
            frame = cv2.imread(frame_path)
            patches = extract_homogeneous_patches(image=frame, patch_size=(128, 128), num_patches=num_patches_per_frame,
                                                  min_std_dev_threshold=0.005, max_std_dev_threshold=0.02)
            patch_paths = []
            for index, patch in enumerate(patches):
                patch_name = f'{Path(frame_path).stem}-{str(index).zfill(3)}-' \
                             f'{str(patch[2]).zfill(4)}-{str(patch[3]).zfill(4)}.png'
                patch_path = str(dest_patches_dir.joinpath(patch_name))
                cv2.imwrite(patch_path, patch[1])
                patch_paths.append(patch_path)

            all_patch_paths.extend(patch_paths)
        else:
            patch_paths = sorted([str(x) for x in dest_patches_dir.glob(f'{Path(frame_path).stem}*')])
            all_patch_paths.extend(patch_paths[:num_patches_per_frame])

    return all_patch_paths


def create_patch_dataset(source_frames_split, dest_patches_split, dest_patches_dir, num_patches_per_frame, device_id):
    for file_path in list(source_frames_split.glob('val.json')):
        dest_file = dest_patches_split.joinpath(file_path.name)

        if not dest_file.exists():
            print(f'Processing the source file: {str(file_path)}')

            with open(file_path) as f:
                frames_json_dict = json.load(f)
                devices_list = sorted(list(frames_json_dict.keys()))

            if device_id:
                device_name = devices_list[device_id]
                extract_patches(
                    frame_paths=frames_json_dict[device_name],
                    num_patches_per_frame=num_patches_per_frame,
                    dest_patches_dir=dest_patches_dir.joinpath(device_name)
                )
            else:
                patches_json_dict = {}
                for device_id, device_name in enumerate(devices_list):
                    patch_paths = extract_patches(
                        frame_paths=frames_json_dict[device_name],
                        num_patches_per_frame=num_patches_per_frame,
                        dest_patches_dir=dest_patches_dir.joinpath(device_name)
                    )
                    patches_json_dict[device_name] = patch_paths

                dest_patches_split.mkdir(parents=True, exist_ok=True)
                with open(dest_file, 'w+') as f:
                    json.dump(patches_json_dict, f, indent=2)
                    print(f'Saved the file: {str(dest_file)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract and save patches',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device_id', type=int, default=None, required=False, help='The index of the device')
    parser.add_argument('--ppf', type=int, default=1, required=False, help='Number of patches per frame')

    args = parser.parse_args()
    d_id = args.device_id
    ppf = args.ppf

    create_patch_dataset(
        source_frames_split=Path(rf'/scratch/p288722/datasets/vision/8_devices/bal_50_frames'),
        dest_patches_split=Path(rf'/scratch/p288722/datasets/vision/8_devices/bal_50_frames_{ppf}ppf'),
        dest_patches_dir=Path(rf'/scratch/p288722/datasets/vision/all_patches'),
        num_patches_per_frame=ppf,
        device_id=d_id,
    )
