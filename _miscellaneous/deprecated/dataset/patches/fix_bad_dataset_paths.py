import json
import os
from pathlib import Path

from tqdm import tqdm


def fix_paths(source_dir, dest_dir):
    for device_dir in source_dir.glob('*'):
        video_names = sorted(set([str(x).split('-')[0].split('/')[-1] for x in device_dir.glob('*')]))
        for video_name in tqdm(video_names):
            target_dir = dest_dir.joinpath(device_dir.name).joinpath(video_name)
            target_dir.mkdir(parents=True, exist_ok=True)
            for patch_path in device_dir.glob(f'{video_name}*'):
                os.rename(str(patch_path), str(target_dir.joinpath(patch_path.name)))


def fix_dataset_views(source_dir):
    for file_path in source_dir.glob('*.json'):
        with open(file_path) as f:
            json_dict = json.load(f)

        for device_name in tqdm(json_dict):
            patch_paths = json_dict[device_name]
            video_names = sorted(set([str(x).split('-')[0].split('/')[-1] for x in patch_paths]))
            modified_patch_paths = []
            for video_name in video_names:
                patch_paths_per_video = sorted([x for x in patch_paths if video_name in x])
                patch_paths_per_video = ['/'.join(x.split('/')[:-1]) + '/' + video_name + '/' + x.split('/')[-1] for x
                                         in patch_paths_per_video]

                modified_patch_paths.extend(patch_paths_per_video)
            json_dict[device_name] = modified_patch_paths

        with open(file_path, 'w+') as f:
            json.dump(json_dict, f, indent=2)


if __name__ == '__main__':
    # fix_paths(
    #     source_dir=Path(r'/scratch/p288722/datasets/vision/__all_patches'),
    #     dest_dir=Path(r'/scratch/p288722/datasets/vision/all_patches')
    # )

    fix_dataset_views(
        source_dir=Path(r'/scratch/p288722/datasets/vision/8_devices/bal_50_frames_50ppf')
    )
