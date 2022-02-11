import json
import os
from pathlib import Path

import pandas as pd
import numpy as np


class VideoPredictor:

    def __init__(self, model_file_name=None, result_dir=None):
        self.model_file_name = model_file_name
        self.result_dir = result_dir
        # Top k predicted classes will be saved in csv-file
        self.top_k_predictions = 3

    def start(self, frame_prediction_file):
        # Read file with frame predictions
        df_frame_predictions = pd.read_csv(frame_prediction_file)
        if len(df_frame_predictions) == 0:
            print(f"No frame predictions found in {str(frame_prediction_file)}")
            return
        return self._predict_and_save(df_frame_predictions)

    def _predict_and_save(self, df_frame_predictions, output_file=None):
        # Predict videos
        # df_frame_predictions = self.__compute_homogeneity_score(df_frame_predictions)
        # df_frame_predictions.to_csv(f_pred_file, index=False)
        video_predictions = self.__get_predictions(df_frame_predictions=df_frame_predictions)

        # Create dataframe based on video_predictions
        df_results = pd.DataFrame(video_predictions, columns=self.__get_columns(self.top_k_predictions))

        if not output_file:
            output_file = self.get_output_file()
        df_results.to_csv(output_file, index=False)

        return output_file

    def get_output_file(self):
        # Remove .h5 extension
        output_file_name = self.model_file_name.split('.')[0]
        output_file = os.path.join(self.result_dir, f"{output_file_name}_V_predictions.csv")
        return output_file

    @staticmethod
    def get_platform(filename):
        if "YT" in filename:
            return "YT"
        elif "WA" in filename:
            return "WA"
        else:
            return "original"

    def __get_predictions(self, df_frame_predictions):
        # Determine unique videos
        # videos = df_frame_predictions["File"].str.split("-").str[0].unique()
        videos = sorted({x.split(',')[0].strip('[]\\\'').split('-')[0] for x in df_frame_predictions["File"]})
        print(f"Total number of videos to classify: {len(videos)}.")

        max_softmax = [[float(y) for y in x.strip('[]').split()] for x in df_frame_predictions['Softmax Scores']]
        max_softmax = [float(max(np.array(x))) for x in max_softmax]
        df_frame_predictions = df_frame_predictions.assign(max_softmax=max_softmax)

        # range_mid_point = 0.6150  # best max homogeneity score (test set)
        # range_mid_point = 0.5389    # best mean homogeneity score (test set)

        # # add the niqe_scores to the dataframe
        # with open(r'/scratch/p288722/datasets/vision/I_frame_metrics/all_I_frames.json', 'r') as f:
        #     scores = json.load(f)
        #     fp = df_frame_predictions
        #     fp = fp.assign(niqe=[scores[Path(x.File).name]['niqe'] for _, x in fp.iterrows()])
        #     fp = fp.assign(piqe=[scores[Path(x.File).name]['piqe'] for _, x in fp.iterrows()])
        #     fp = fp.assign(brisque=[scores[Path(x.File).name]['brisque'] for _, x in fp.iterrows()])
        #     fp = fp.assign(mean_homogeneity=[scores[Path(x.File).name]['mean_homogeneity'] for _, x in fp.iterrows()])
        #     fp = fp.assign(mean_energy=[scores[Path(x.File).name]['mean_energy'] for _, x in fp.iterrows()])
        #     fp = fp.assign(entropy=[scores[Path(x.File).name]['entropy'] for _, x in fp.iterrows()])
        #     df_frame_predictions = fp
        #
        # header = ','.join(['Video Name'] + [str(x) for x in range(1, 51)]) + '\n'
        # gt_data, pred_data, softmax_data, niqe_data = [header], [header], [header], [header]
        # piqe_data, brisque_data = [header], [header]
        # mean_homogeneity_data, mean_energy_data, entropy_data = [header], [header], [header]
        # for video in videos:
        #     df_video_predictions = df_frame_predictions[df_frame_predictions["File"].str.contains(video)]
        #     video_name = Path(video).name
        #     ground_truth = [str(x) for x in df_video_predictions["True Label"]]
        #     predicted = [str(x) for x in df_video_predictions["Predicted Label"]]
        #     softmax_scores = [x.strip('[]').split()[int(idx)] for x, idx in
        #                       zip(df_video_predictions["Softmax Scores"], predicted)]
        #     niqe = [str(x) for x in df_video_predictions["niqe"]]
        #     piqe = [str(x) for x in df_video_predictions["piqe"]]
        #     brisque = [str(x) for x in df_video_predictions["brisque"]]
        #     mean_homogeneity = [str(x) for x in df_video_predictions["mean_homogeneity"]]
        #     mean_energy = [str(x) for x in df_video_predictions["mean_energy"]]
        #     entropy = [str(x) for x in df_video_predictions["entropy"]]
        #
        #     gt_data.append(','.join([video_name] + ground_truth) + '\n')
        #     pred_data.append(','.join([video_name] + predicted) + '\n')
        #     softmax_data.append(','.join([video_name] + softmax_scores) + '\n')
        #     niqe_data.append(','.join([video_name] + niqe) + '\n')
        #     piqe_data.append(','.join([video_name] + piqe) + '\n')
        #     brisque_data.append(','.join([video_name] + brisque) + '\n')
        #     mean_homogeneity_data.append(','.join([video_name] + mean_homogeneity) + '\n')
        #     mean_energy_data.append(','.join([video_name] + mean_energy) + '\n')
        #     entropy_data.append(','.join([video_name] + entropy) + '\n')
        #
        # dest_dir = Path(r'/scratch/p288722/datasets/vision/I_frame_metrics')
        # with open(dest_dir.joinpath('gt_data.csv'), 'w+') as f:
        #     f.writelines(gt_data)
        # with open(dest_dir.joinpath('pred_data.csv'), 'w+') as f:
        #     f.writelines(pred_data)
        # with open(dest_dir.joinpath('softmax_data.csv'), 'w+') as f:
        #     f.writelines(softmax_data)
        # with open(dest_dir.joinpath('niqe_data.csv'), 'w+') as f:
        #     f.writelines(niqe_data)
        # with open(dest_dir.joinpath('piqe_data.csv'), 'w+') as f:
        #     f.writelines(piqe_data)
        # with open(dest_dir.joinpath('brisque_data.csv'), 'w+') as f:
        #     f.writelines(brisque_data)
        # with open(dest_dir.joinpath('mean_homogeneity_data.csv'), 'w+') as f:
        #     f.writelines(mean_homogeneity_data)
        # with open(dest_dir.joinpath('mean_energy_data.csv'), 'w+') as f:
        #     f.writelines(mean_energy_data)
        # with open(dest_dir.joinpath('entropy_data.csv'), 'w+') as f:
        #     f.writelines(entropy_data)

        video_predictions = []
        for video in videos:
            # Get frame predictions for current video
            df_video_predictions = df_frame_predictions[df_frame_predictions["File"].str.contains(video)]
            # Class index
            class_idx = df_video_predictions["True Label"].iloc[0]

            # Use all frames to classify a video
            n_frames = len(df_video_predictions)

            # # Weighted Majority by Niqe scores
            # df_device_vote = df_video_predictions.groupby(['Predicted Label'])['niqe'].sum().rename_axis(
            #     'class').reset_index(name='vote_count')

            # # Compute the average of max_softmax (grouped by predicted label)
            # df_device_vote = df_video_predictions.groupby(['Predicted Label'])['max_softmax'].mean().rename_axis(
            #     'class').reset_index(name='vote_count')

            # a = df_video_predictions.groupby(['Predicted Label'])['max_softmax'].mean().rename_axis(
            #     'class').reset_index(name='vote_count')
            # b = df_video_predictions.groupby(['Predicted Label'])['max_softmax'].sum().rename_axis(
            #     'class').reset_index(name='vote_count')
            # df_device_vote = pd.DataFrame(a.values * b.values, columns=a.columns, index=a.index).astype({'class': int})

            # Select n_frames random rows. This can be used to experiment with predictions by using
            # different number of frames per video.
            # if n_frames < len(df_video_predictions):
            #     df_video_predictions = df_video_predictions.sample(n=n_frames)

            # softmax_scores = []
            # for index in np.ravel(df_video_predictions["Softmax Scores"].axes):
            #     softmax_scores.append(
            #         np.array([float(x) for x in df_video_predictions["Softmax Scores"][index].
            #                  replace('\n', '').replace('[', '').replace(']', '').split(' ') if x is not '']))
            # sum_softmax = np.sum(softmax_scores, axis=0)
            # top_3_classes = np.argsort(-1 * sum_softmax)[:3]
            # top_3_confidence = [round(x, 6) for x in sum_softmax[top_3_classes] / sum(sum_softmax)]
            #
            # # Get platform (i.e. original/Whatsapp (WA)/Youtube (YT) by using filename of first row.
            # platform = self.get_platform(df_video_predictions["File"].iloc[0])
            #
            # video_result = [video, platform, n_frames, class_idx]
            # for item in zip(top_3_classes, top_3_confidence):
            #     video_result.extend(item)

            # Majority Vote
            df_device_vote = df_video_predictions["Predicted Label"].value_counts().rename_axis('class').reset_index(
                name='vote_count')

            # n = 50  # num_frames_to_consider
            # df = df_video_predictions.sort_values(by='homogeneity_score_mean', ascending=False)[:n]
            # df_device_vote = df["Predicted Label"].value_counts().rename_axis('class').reset_index(name='vote_count')

            # distance = np.abs(df_video_predictions['mean_homogeneity'] - range_mid_point).values
            # df_video_predictions = df_video_predictions.assign(dist_from_selected_homo_score=distance)
            # df = df_video_predictions.sort_values(by='dist_from_selected_homo_score')[:n]
            # df_device_vote = df["Predicted Label"].value_counts().rename_axis('class').reset_index(name='vote_count')

            # df = df_video_predictions.sort_values(by='max_homogeneity', ascending=False)[:n]
            # df_device_vote = df["Predicted Label"].value_counts().rename_axis('class').reset_index(name='vote_count')

            # # Weighted Majority by Homogeneity score
            # df_device_vote = df_video_predictions.groupby(['Predicted Label'])['homogeneity_score'].sum().rename_axis(
            #     'class').reset_index(name='vote_count')

            # n = 50  # num_frames_to_consider
            # temp_df = df_video_predictions.sort_values(by='homogeneity_score_mean', ascending=False)[:n]
            # df_device_vote = temp_df["Predicted Label"].value_counts().rename_axis('class').reset_index(
            #     name='vote_count')
            #
            # temp_df = df_video_predictions.sort_values(by='homogeneity_score_max', ascending=False)[:n]
            # df_device_vote = temp_df["Predicted Label"].value_counts().rename_axis('class').reset_index(
            #     name='vote_count')

            # Select top 3 devices
            df_top_votes = df_device_vote.nlargest(self.top_k_predictions, ['vote_count'])

            # Get platform (i.e. original/Whatsapp (WA)/Youtube (YT) by using filename of first row.
            platform = self.get_platform(df_video_predictions["File"].iloc[0])

            video_result = [video, platform, n_frames, class_idx]
            for row_idx, row in df_top_votes.iterrows():
                label = int(row["class"])
                vote_count = int(row["vote_count"])
                confidence = round(vote_count / n_frames, 3)

                video_result.append(label)
                video_result.append(confidence)

            # If all frames predicted the same label, we only have one row in df_top_votes.
            if len(df_top_votes) < self.top_k_predictions:
                i = len(df_top_votes)
                while i < 3:
                    video_result.append(None)
                    video_result.append(None)
                    i += 1

            video_loss = df_video_predictions["Loss"].mean()
            video_result.append(video_loss)

            video_predictions.append(video_result)

        return video_predictions

    @staticmethod
    def __get_columns(k):
        columns = ['filename', 'platform', 'n_frames_for_prediction', 'true_class']

        for i in range(k):
            # e.g. top1_class
            columns.append(f"top{i + 1}_class")
            # e.g. top1_conf
            columns.append(f"top{i + 1}_conf")

        columns.append('Loss')

        return columns
