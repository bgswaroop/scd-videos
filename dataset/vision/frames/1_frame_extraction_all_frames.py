import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def start(device):
    print(f"\n{device} | Start")
    videos_processed = 0
    device_time_start = time.time()

    input_device_folder = Path(INPUT_DIR).joinpath(device)
    all_videos = list(input_device_folder.glob('videos/*/*'))
    total_nb_videos = len(all_videos)

    for video_path in tqdm(all_videos):
        print(f"processing {video_path}")
        video_time_start = time.time()

        # The video's name without extension is used for assigning names to the frames.
        video_name = video_path.stem
        frames_saved = video_to_frames(video_name, str(video_path), device, True)
        videos_processed += 1

        video_time_end = time.time()
        print(f"Finished video {video_name} ({videos_processed}/{total_nb_videos}) in "
              f"{int(video_time_end - video_time_start)} seconds. {frames_saved} Frames are saved.")

    device_time_end = time.time()
    print(f"{device} | Finished {total_nb_videos} videos in {int(device_time_end - device_time_start)} seconds.")


def video_to_frames(video_name, video_path, device, verbose=True):
    # Create video output dir
    output_dir = Path(OUTPUT_DIR).joinpath(device).joinpath(video_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Start capturing the feed
    cap = cv2.VideoCapture(video_path)

    # Frame rate per second
    frame_rate = np.floor(cap.get(cv2.CAP_PROP_FPS))

    # Total number of frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Orientation
    # orientation_auto = cap.get(cv2.CAP_PROP_ORIENTATION_AUTO)
    # orientation_meta = cap.get(cv2.CAP_PROP_ORIENTATION_META)

    if verbose:
        print(f"Video: {video_name}, #frames: {num_frames}, FPS: {frame_rate}, #frames to save: {num_frames}.")

    frames_saved = 0
    count = 0

    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()

        # Frame is available
        if ret:
            # Get current frame id
            frame_id = cap.get(1)
            frame_path = output_dir.joinpath(f"{video_name}-" + "%#05d.png" % frame_id)
            if not frame_path.exists():
                print(frame_path)
                cv2.imwrite(str(frame_path), frame)
            frames_saved = frames_saved + 1
        count += 1

        # if (frames_saved >= number_of_frames_to_save or count >= video_length):
        if count >= num_frames:
            # Release the feed
            if cap.isOpened():
                cap.release()
            break

    return frames_saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract and save video frames',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to VISION dataset (input directory consisting of folders per device)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save the frames')
    parser.add_argument('--devices', type=str, required=False,
                        help='Only extract frames of these devices (separated by a \',\')')
    parser.add_argument('--device_id', type=int, required=False, help='Specify device ID')
    args = parser.parse_args()

    """
    --input_dir="/data/p288722/VISION/dataset" --output_dir="/scratch/p288722/datasets/vision/all_frames" --device_id=0
    """

    # input dir is the path to the VISION dataset in its original structure
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    DEVICES = ['D01_Samsung_GalaxyS3Mini',
               'D02_Apple_iPhone4s',
               'D03_Huawei_P9',
               'D04_LG_D290',
               'D05_Apple_iPhone5c',
               'D06_Apple_iPhone6',
               'D07_Lenovo_P70A',
               'D08_Samsung_GalaxyTab3',
               'D09_Apple_iPhone4',
               'D10_Apple_iPhone4s',
               'D11_Samsung_GalaxyS3',
               'D12_Sony_XperiaZ1Compact',
               'D14_Apple_iPhone5c',
               'D15_Apple_iPhone6',
               'D16_Huawei_P9Lite',
               'D18_Apple_iPhone5c',
               'D19_Apple_iPhone6Plus',
               'D24_Xiaomi_RedmiNote3',
               'D25_OnePlus_A3000',
               'D26_Samsung_GalaxyS3Mini',
               'D27_Samsung_GalaxyS5',
               'D28_Huawei_P8',
               'D29_Apple_iPhone5',
               'D30_Huawei_Honor5c',
               'D31_Samsung_GalaxyS4Mini',
               'D32_OnePlus_A3003',
               'D33_Huawei_Ascend',
               'D34_Apple_iPhone5']

    start(device=DEVICES[args.device_id])
