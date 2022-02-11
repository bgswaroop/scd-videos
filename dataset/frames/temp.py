import argparse
import json
from pathlib import Path

import tensorflow as tf
from tqdm import tqdm


def change_scratch_to_data():
    """
    This is needed when transferring dataset view files from peregrine to a40 server
    :return:
    """
    root_dir = Path(r'/data/p288722/datasets/vision/I_frame_splits')
    for file in root_dir.glob('*/*.json'):
        with open(file) as f:
            json_data = json.load(f)
        for device in json_data:
            json_data[device] = [x.replace('scratch', 'data') for x in json_data[device]]
        with open(file, 'w+') as f:
            json.dump(json_data, f, indent=2)


def correct_frame_disorientation():
    """
    Identify videos with disoriented frames. That is videos where few frames are horizontal while the rest
    have vertical orientation. Also test if all the video frames from a video are of the same dimensions.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=0, help='device index')
    p = parser.parse_args()
    assert 0 <= p.device_id <= 27, 'There are only 28 devices, indices must be in [0, 27]'

    dataset_path = Path(r'/scratch/p288722/datasets/vision/all_I_frames')
    devices = sorted(dataset_path.glob('*'))[p.device_id]
    found = False
    for video in tqdm(sorted(devices.glob('*'))):
        video_frames = sorted(video.glob('*.png'))
        frame_dimensions = {tf.image.decode_png(tf.io.read_file(str(x)), channels=3).numpy().shape for x in
                            video_frames}
        if len(frame_dimensions) != 1:
            found = True
            print(video, frame_dimensions)
            with open(f'disorientation_{p.device_id}.txt', 'a+') as f:
                f.write(f'{video} - {frame_dimensions}\n')

    if found:
        print('Disorientation is found!')
    else:
        print('All good!')


if __name__ == '__main__':
    correct_frame_disorientation()
