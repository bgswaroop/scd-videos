from pathlib import Path
import pandas as pd


def compute_model_level_labels(df_frame_predictions):
    dataset_dir = Path(r'/scratch/p288722/datasets/vision/all_I_frames')
    device_labels = [x.name for x in sorted(dataset_dir.glob('*'))]
    model_labels = sorted({x[4:] for x in device_labels})
    model_label_to_id = {x: idx for idx, x in enumerate(model_labels)}
    device_to_model_map = {idx: model_label_to_id[x[4:]] for idx, x in enumerate(device_labels)}

    df_frame_predictions['True Label'] = [device_to_model_map[x] for x in df_frame_predictions['True Label']]
    df_frame_predictions['Predicted Label'] = [device_to_model_map[x] for x in df_frame_predictions['Predicted Label']]

    return df_frame_predictions


if __name__ == '__main__':
    frame_predictions = Path(r'/scratch/p288722/runtime_data/scd_videos_first_revision/06_I_frames_bs64/50_frames_pred/'
                             r'mobile_net/models/MobileNet_50_I_frames_ccrop_run2/predictions_50_frames/frames/'
                             r'fm-e00012_F_predictions.csv')
    d = compute_model_level_labels(pd.read_csv(frame_predictions))
    print(' ')
