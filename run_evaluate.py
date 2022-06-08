import argparse
import os
from pathlib import Path

import tensorflow as tf
from timeit import default_timer as timer

import dataset
from utils.predict_utils import (FramePredictionStatistics, FramePredictionVis, FramePredictor,
                                 VideoPredictionStatistics, VideoPredictionVis, VideoPredictor)


def get_models_files(input_dir, models_to_process):
    # Get all files (i.e. models) from input directory
    files_list = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    files_list = sorted(files_list)  # picking the last trained model
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

    # Dataset params
    parser.add_argument('--dataset_name', type=str, choices=['vision', 'qufvd'])
    parser.add_argument('--eval_set', type=str, required=True, default='val', choices=['val', 'test'])
    parser.add_argument('--all_I_frames_dir', type=Path, help='Input directory of extracted I frames')
    parser.add_argument('--all_frames_dir', type=Path, help='Input directory of extracted frames')
    parser.add_argument('--frame_selection', type=str, default='equally_spaced', choices=['equally_spaced', 'first_N'])
    parser.add_argument('--frame_type', type=str, default='I', choices=['I', 'all'])
    parser.add_argument('--fpv', type=int, default=50, help='max number of frames per video (set -1 for all frames)')
    parser.add_argument('--category', type=str, choices=["native", "whatsapp", "youtube", "None"])

    # ConvNet params
    parser.add_argument('--height', type=int, default=480, help='Height of CNN input dimension [default: 480]')
    parser.add_argument('--width', type=int, default=800, help='Width of CNN input dimension [default: 800]')
    parser.add_argument('--net_type', type=str, default='mobile',
                        choices=['mobile', 'effv2', 'misl', 'res', 'mobile_supcon', 'resnet_supcon', 'eff_supcon'])

    # Evaluation params
    parser.add_argument('--epoch', type=int, default=-1, help='Choose the epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to directory consisting of .h5-models (to use for predicting)')
    parser.add_argument('--models', type=str,
                        help='Models within input dir (*.h5) to evaluate. Separate models by a ","')

    # General params
    parser.add_argument('--suffix', type=str, help='enter suffix string for the predictions folder')
    parser.add_argument('--gpu_id', type=int, default=None, help='Choose the available GPU devices')

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
    _, frames_res_dir, videos_res_dir, plots_res_dir = get_result_dir(p.input_dir,
                                                                      primary_suffix=p.category,
                                                                      secondary_suffix=p.suffix)
    if p.epoch > 0:
        model_files = [x for x in model_files if str(p.epoch).zfill(5) in x]
    print(f"Found {len(model_files)} files for model {model_name}")

    if p.dataset_name == 'vision':
        data_factory = dataset.vision.DataFactory(p)
    elif p.dataset_name == 'qufvd':
        data_factory = dataset.qufvd.DataFactory(p)
    else:
        raise ValueError(f'Invalid option {p.dataset_name}')

    if p.eval_set == 'val':
        filename_ds, eval_ds = data_factory.get_tf_val_data(category=p.category)
    elif p.eval_set == 'test':
        filename_ds, eval_ds = data_factory.get_tf_test_data(category=p.category)
    else:
        raise ValueError('Invalid evaluation set')
    # List containing only the file names of items in evaluation set
    onehot_ground_truths = data_factory.get_labels(eval_ds)
    eval_ds_filepaths = [x.decode("utf-8") for x in filename_ds.as_numpy_iterator()]

    for model_file in model_files:
        # start = timer()
        frame_predictor = FramePredictor(model_dir=p.input_dir, model_file_name=model_file, result_dir=frames_res_dir)
        video_predictor = VideoPredictor(model_file_name=model_file, result_dir=videos_res_dir,
                                         dataset_name=p.dataset_name)

        frames_results = Path(frame_predictor.get_output_file())
        videos_results = Path(video_predictor.get_output_file())

        if not (frames_results.exists() and videos_results.exists()):
            print(f"{model_file} | Start prediction process")

            if not frames_results.exists():
                print(f"{model_file} | Start predicting frames")
                frame_predictor.start(eval_ds, eval_ds_filepaths, onehot_ground_truths)
                print(f"{model_file} | Predicting frames completed")

            if not videos_results.exists():
                print(f"{model_file} | Start predicting videos")
                video_predictor.start(frame_prediction_file=str(frames_results))
                print(f"{model_file} | Predicting videos completed")
        # end = timer()
        # print(f'Elapsed time: {end - start} sec')

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
