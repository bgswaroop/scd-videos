import json
from pathlib import Path


def count_train_val_test_videos():
    dataset_views = Path(r'/scratch/p288722/datasets/vision/splits/bal_50_frames')
    for split in ['train', 'val', 'test']:
        with open(dataset_views.joinpath(f'{split}.json')) as f:
            dataset = json.load(f)
        videos = {Path(y).parent.name for _, x in dataset.items() for y in x}
        frames = [y for _, x in dataset.items() for y in x]
        print(f'Number of videos in {split}: {len(videos)}')
        print(f'Number of video frames in {split}: {len(frames)}')


def create_videos_split():
    dataset_views = Path(r'/scratch/p288722/datasets/vision/I_frame_splits/bal_50_frames')
    destination_dir = Path(r'/home/p288722/git_code/scd_videos_first_revision/dataset/split')
    for split in ['train', 'val', 'test', 'unbal_val']:
        with open(dataset_views.joinpath(f'{split}.json')) as f:
            dataset = json.load(f)
        videos = {d: sorted({Path(y).parent.name for y in x}) for d, x in dataset.items()}
        # frames = [y for _, x in dataset.items() for y in x]
        # print(f'Number of videos in {split}: {len(videos)}')
        # print(f'Number of video frames in {split}: {len(frames)}')
        with open(destination_dir.joinpath(f'{split}_videos.json'), 'w+') as f:
            json.dump(videos, f, indent=2)


if __name__ == '__main__':
    # count_train_val_test_videos()
    create_videos_split()
