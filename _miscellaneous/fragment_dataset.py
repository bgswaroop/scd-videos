import json
from pathlib import Path


def split_dataset(input_dataset_dir, num_fragments, output_dataset_dir):
    """
    :param input_dataset_dir:
    :param num_fragments: int
    :param output_dataset_dir:
    :return:
    """

    dest_split = {index: {} for index in range(num_fragments)}

    for split in ['train', 'val', 'test']:
        for index in dest_split:
            dest_split[index][split] = {}

        with open(input_dataset_dir.joinpath(f'{split}.json')) as f:
            file_paths_dict = json.load(f)
        for device in file_paths_dict:
            img_paths = sorted(file_paths_dict[device])
            num_images = len(img_paths)
            step_size = int(num_images / num_fragments)
            if step_size:
                interval = [(x, x + step_size) for x in range(0, num_images, step_size)]
                interval = interval[:9] + [(interval[9][0], num_images)]
                for index, i in enumerate(interval):
                    dest_split[index][split][device] = img_paths[i[0]: i[1]]
            else:
                for index in dest_split:
                    dest_split[index][split][device] = []

    for index in dest_split:
        for split in dest_split[index]:
            output_dataset_dir.joinpath(f'{index}').mkdir(exist_ok=True, parents=True)
            with open(output_dataset_dir.joinpath(f'{index}/{split}.json'), 'w+') as f:
                json.dump(dest_split[index][split], f)


if __name__ == '__main__':
    split_dataset(
        input_dataset_dir=Path(r'/scratch/p288722/datasets/vision/bal_all_frames/'),
        num_fragments=10,
        output_dataset_dir=Path(r'/scratch/p288722/datasets/vision/bal_all_frames_fragments/')
    )