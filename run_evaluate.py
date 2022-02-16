import argparse
import os
import random
from pathlib import Path

import tensorflow as tf

from dataset import DataFactory
from utils.predict_utils import (FramePredictionStatistics, FramePredictionVis, FramePredictor,
                                 VideoPredictionStatistics, VideoPredictionVis, VideoPredictor)


def get_models_files(input_dir, models_to_process):
    # Get all files (i.e. models) from input directory
    files_list = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    files_list = sorted(files_list)
    print(f"Found {len(files_list)} files in {input_dir}: {files_list}")

    if models_to_process:
        print(f"'Models' argument is set. Only the following models will be evaluated: {models_to_process.split(',')}")
        model_split = models_to_process.split(',')
        return [file for file in files_list if file in model_split]

    return files_list


def get_result_dir(input_dir, primary_suffix, secondary_suffix):
    if primary_suffix:
        output_dir = os.path.join(input_dir, f"predictions_{primary_suffix}")
    elif secondary_suffix:
        output_dir = os.path.join(input_dir, f"predictions_{secondary_suffix}")
    else:
        output_dir = os.path.join(input_dir, f"predictions")

    # Create output directory is not exists
    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(e)
            raise ValueError(f"Error during creation of output directory")

    frames_output_dir = os.path.join(output_dir, "frames")
    videos_output_dir = os.path.join(output_dir, "videos")
    plots_output_dir = os.path.join(output_dir, "plots")
    if not os.path.isdir(frames_output_dir):
        os.makedirs(frames_output_dir)
    if not os.path.isdir(videos_output_dir):
        os.makedirs(videos_output_dir)
    if not os.path.isdir(plots_output_dir):
        os.makedirs(plots_output_dir)

    return output_dir, frames_output_dir, videos_output_dir, plots_output_dir


def none_or_bool(value):
    if value == 'None':
        return None
    return bool(int(value))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate the CNNs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to directory consisting of .h5-models (to use for predicting)')
    parser.add_argument('--models', type=str,
                        help='Models within input dir (*.h5) to evaluate. Separate models by a ","')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use to make predictions')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--height', type=int, default=480, help='Height of CNN input dimension [default: 480]')
    parser.add_argument('--width', type=int, default=800, help='Width of CNN input dimension [default: 800]')
    parser.add_argument('--frames_per_video', type=int, default=50,
                        help='to determine #frames to sample for creating multi-frame dataset for the ConvNet')
    parser.add_argument('--category', type=str, help='enter "native", "whatsapp", or "youtube"')
    parser.add_argument('--suffix', type=str, help='enter suffix string for the predictions folder')
    parser.add_argument('--gpu_id', type=int, default=None, help='Choose the available GPU devices')
    parser.add_argument('--epoch', type=int, default=-1, help='Choose the epoch')
    parser.add_argument('--eval_set', type=str, required=True, default='val', help='Evaluation set - val or test')
    parser.add_argument('--homogeneity_csv', type=str, default=r'/scratch/p288722/datasets/vision/old_baseline_split/'
                                                               r'bal_all_frames/homogeneity_scores.csv')
    parser.add_argument('--homo_or_not', type=bool, default=None,
                        help='Filter dataset by homogeneous frames if `True`, or'
                             'Filter dataset by non-homogeneous frames if `False`, or '
                             'Do not perform any filtering - set to `None`')

    p = parser.parse_args()

    if p.gpu_id is not None:
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices):
            tf.config.set_visible_devices(physical_devices[p.gpu_id], device_type='GPU')
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    return p


def run_flow():
    p = parse_args()

    model_name = p.input_dir.split(os.path.sep)[-1]
    model_files = get_models_files(p.input_dir, p.models)
    random.shuffle(model_files)
    _, frames_res_dir, videos_res_dir, plots_res_dir = get_result_dir(p.input_dir,
                                                                      primary_suffix=p.category,
                                                                      secondary_suffix=p.suffix)
    if p.epoch > 0:
        model_files = [x for x in model_files if str(p.epoch).zfill(5) in x]

    print(f"Found {len(model_files)} files for model {model_name}")

    dataset = DataFactory(p.dataset, p.batch_size, p.height, p.width, p.frames_per_video, p.homo_or_not)
    if p.eval_set == 'val':
        filename_ds, eval_ds = dataset.get_tf_val_data(category=p.category)
    elif p.eval_set == 'test':
        filename_ds, eval_ds = dataset.get_tf_test_data(category=p.category)
    else:
        raise ValueError('Invalid evaluation set')
    # List containing only the file names of items in evaluation set
    # eval_ds_filepaths = [x.decode("utf-8") for x in filename_ds.as_numpy_iterator()]
    eval_ds_filepaths = [[Path(x[0].decode("utf-8")).name, Path(x[1].decode("utf-8")).name, Path(
        x[2].decode("utf-8")).name] for x in filename_ds.as_numpy_iterator()]

    for model_file in model_files:
        frame_predictor = FramePredictor(model_dir=p.input_dir, model_file_name=model_file,
                                         result_dir=frames_res_dir, input_dir=p.dataset,
                                         homogeneity_csv=p.homogeneity_csv)
        video_predictor = VideoPredictor(model_file_name=model_file, result_dir=videos_res_dir)

        frames_results = Path(frame_predictor.get_output_file())
        videos_results = Path(video_predictor.get_output_file())

        if not (frames_results.exists() and videos_results.exists()):
            print(f"{model_file} | Start prediction process")

            if not frames_results.exists():
                print(f"{model_file} | Start predicting frames")
                frame_predictor.start(test_ds=eval_ds, filenames=eval_ds_filepaths)
                print(f"{model_file} | Predicting frames completed")

            if not videos_results.exists():
                print(f"{model_file} | Start predicting videos")
                video_predictor.start(frame_prediction_file=str(frames_results))
                print(f"{model_file} | Predicting videos completed")

    print(f"Creating Statistics and Visualizations ...")
    # Create Frame Prediction Statistics
    fps = FramePredictionStatistics(result_dir=frames_res_dir)
    frame_stats = fps.start()
    print(f"Frame Prediction Statistics Completed")

    vps = VideoPredictionStatistics(result_dir=videos_res_dir)
    video_stats = vps.start()
    print(f"Video Prediction Statistics Completed")

    fpv = FramePredictionVis(result_dir=plots_res_dir)
    fpv.start(frame_stats)
    print(f"Frame Prediction Visualization Completed")

    vpv = VideoPredictionVis(result_dir=plots_res_dir, model_name=model_name)
    vpv.start(video_stats)
    print(f"Video Prediction Visualization Completed")


if __name__ == "__main__":
    run_flow()
