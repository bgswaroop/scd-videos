#!/bin/bash
#SBATCH --job-name=mn-200Iframes-ccrop
#SBATCH --time=24:00:00
#SBATCH --mem=64g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
# --reservation=infsys
#SBATCH --cpus-per-task=12
#SBATCH --array=1
#SBATCH --dependency=afterany:22972735_1

# create the following directory manually
#SBATCH --chdir=/scratch/p288722/runtime_data/scd_videos_first_revision/09_triplets_bs128
#SBATCH --output=slurm-%j-%x.out
#SBATCH --error=slurm-%j-%x.out

#module load cuDNN/8.0.4.30-CUDA-11.1.1
#module load TensorFlow/2.5.0-fosscuda-2020b

module purge
module load OpenBLAS/0.3.15-GCC-10.3.0
module load CUDAcore/11.1.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/p288722/softwares/cuda/lib64
source /data/p288722/python_venv/scd_videos_first_revision/bin/activate
#export TF_GPU_ALLOCATOR=cuda_malloc_async

num_frames=200
base_dir=$(pwd)
splits_dir="/scratch/p288722/datasets/vision/I_frame_splits/bal_all_frames"
#splits_dir="/scratch/p288722/datasets/vision/I_frame_splits/bal_${num_frames}_frames"
net="mobile"
const_type="None"

homo_or_not="None" # also change the model_name accordingly

case ${net} in
  "mobile") model_name="MobileNet_${num_frames}_I_frames_ccrop_run${SLURM_ARRAY_TASK_ID}" ;;
  "eff") model_name="EfficientNet" ;;
  "misl") model_name="MISLNet" ;;
  "mobile_supcon") model_name="MobileNet_ft" ;;
  "resnet_supcon") model_name="ResNet_ft" ;;
  "eff_supcon") model_name="EfficientNet_ft" ;;
  *) exit 1 ;;
esac
case ${const_type} in
  "derrick") model_name="${model_name}_Const" ;;
  "guru") model_name="${model_name}_Const_Pos" ;;
  "None") ;;
  *) exit 1 ;;
esac

#python3 /home/p288722/git_code/scd_videos_first_revision/run_train.py --homo_or_not=None --net_type=mobile --dataset=/scratch/p288722/datasets/vision/I_frame_splits/bal_50_frames --epochs=20 --lr=0.1 --batch_size=64 --height=480 --width=800 --use_pretrained=1 --gpu_id=0 --const_type=None --model_name=MobileNet_50_I_frames_ccrop_run1 --global_results_dir=/scratch/p288722/runtime_data/scd_videos_first_revision/07_ppcce/50_frames/mobile_net
#python3 /home/p288722/git_code/scd_videos_first_revision/run_evaluate.py --homo_or_not=None --eval_set=val --dataset=/scratch/p288722/datasets/vision/I_frame_splits/bal_50_frames --batch_size=64 --height=480 --width=800 --gpu_id=0 --suffix=50_frames_val --input_dir=/scratch/p288722/runtime_data/scd_videos_first_revision/06_I_frames/50_frames/mobile_net/models/MobileNet_50_I_frames_ccrop_run1
#python3 /home/p288722/git_code/scd_videos_first_revision/_miscellaneous/plots/validation_plots.py --val_summary=/scratch/p288722/runtime_data/scd_videos_first_revision/06_I_frames/50_frames/mobile_net/models/MobileNet_50_I_frames_ccrop_run1/predictions_50_frames_val/videos/V_prediction_stats.csv
#python3 /home/p288722/git_code/scd_videos_first_revision/run_evaluate.py --homo_or_not=None --eval_set=test --dataset=/scratch/p288722/datasets/vision/I_frame_splits/bal_50_frames --batch_size=64 --height=480 --width=800 --gpu_id=0 --suffix=50_frames --input_dir=/scratch/p288722/runtime_data/scd_videos_first_revision/06_I_frames/50_frames_pred/mobile_net/models/MobileNet_50_I_frames_ccrop_run1

#python3 /home/p288722/git_code/scd_videos_first_revision/run_train.py --frames_per_video=${num_frames} --homo_or_not=${homo_or_not} --net_type=${net} --dataset=${splits_dir} --epochs=20 --lr=0.1 --batch_size=128 --height=480 --width=800 --use_pretrained=1 --gpu_id=0 --const_type=${const_type} --model_name="${model_name}" --global_results_dir="${base_dir}/${num_frames}_frames/${net}_net"
python3 /home/p288722/git_code/scd_videos_first_revision/run_evaluate.py --frames_per_video=${num_frames} --homo_or_not=${homo_or_not} --eval_set="val" --dataset=${splits_dir} --batch_size=128 --height=480 --width=800 --gpu_id=0 --suffix="${num_frames}_frames_val" --input_dir="${base_dir}/${num_frames}_frames/${net}_net/models/${model_name}"
python3 /home/p288722/git_code/scd_videos_first_revision/_miscellaneous/plots/validation_plots.py --val_summary="${base_dir}/${num_frames}_frames/${net}_net/models/${model_name}/predictions_${num_frames}_frames_val/videos/V_prediction_stats.csv"
python3 /home/p288722/git_code/scd_videos_first_revision/run_evaluate.py --frames_per_video=${num_frames} --homo_or_not=${homo_or_not} --eval_set="test" --dataset=${splits_dir} --batch_size=128 --height=480 --width=800 --gpu_id=0 --suffix="${num_frames}_frames" --input_dir="${base_dir}/${num_frames}_frames_pred/${net}_net/models/${model_name}"

#python3 /home/p288722/git_code/scd_videos_first_revision/run_train.py --homo_or_not=None --net_type=mobile --dataset=/data/p288722/datasets/vision/I_frame_splits/bal_all_frames --epochs=20 --lr=0.1 --batch_size=64 --height=480 --width=800 --use_pretrained=1 --gpu_id=0 --const_type=None --model_name=MobileNet_50_I_frames_ccrop_run1 --global_results_dir=/data/p288722/runtime_data/scd_videos_first_revision/09_triplet_inputs/50_frames/mobile_net
#python3 /home/p288722/git_code/scd_videos_first_revision/run_evaluate.py --homo_or_not=None --eval_set=val --dataset=/data/p288722/datasets/vision/I_frame_splits/bal_all_frames --batch_size=64 --height=480 --width=800 --gpu_id=0 --suffix=50_frames_val --input_dir=/data/p288722/runtime_data/scd_videos_first_revision/09_triplet_inputs/50_frames/mobile_net/models/MobileNet_50_I_frames_ccrop_run1
#python3 /home/p288722/git_code/scd_videos_first_revision/_miscellaneous/plots/validation_plots.py --val_summary=/data/p288722/runtime_data/scd_videos_first_revision/09_triplet_inputs/50_frames/mobile_net/models/MobileNet_50_I_frames_ccrop_run1/predictions_50_frames_val/videos/V_prediction_stats.csv
#python3 /home/p288722/git_code/scd_videos_first_revision/run_evaluate.py --homo_or_not=None --eval_set=test --dataset=/data/p288722/datasets/vision/I_frame_splits/bal_all_frames --batch_size=64 --height=480 --width=800 --gpu_id=0 --suffix=50_frames --input_dir=/data/p288722/runtime_data/scd_videos_first_revision/09_triplet_inputs/50_frames_pred/mobile_net/models/MobileNet_50_I_frames_ccrop_run1

#python3 /home/p288722/git_code/scd_videos_first_revision/run_evaluate.py --homo_or_not=None --eval_set=test --dataset=/data/p288722/datasets/vision/I_frame_splits/bal_all_frames --batch_size=64 --height=480 --width=800 --gpu_id=0 --suffix=1_frame --frames_per_video=1 --input_dir=/data/p288722/runtime_data/scd_videos_first_revision/09_triplet_inputs/50_frames_pred/mobile_net/models/MobileNet_50_I_frames_ccrop_run1
#python3 /home/p288722/git_code/scd_videos_first_revision/run_evaluate.py --homo_or_not=None --eval_set=test --dataset=/data/p288722/datasets/vision/I_frame_splits/bal_all_frames --batch_size=64 --height=480 --width=800 --gpu_id=0 --suffix=5_frames --frames_per_video=5 --input_dir=/data/p288722/runtime_data/scd_videos_first_revision/09_triplet_inputs/50_frames_pred/mobile_net/models/MobileNet_50_I_frames_ccrop_run1
#python3 /home/p288722/git_code/scd_videos_first_revision/run_evaluate.py --homo_or_not=None --eval_set=test --dataset=/data/p288722/datasets/vision/I_frame_splits/bal_all_frames --batch_size=64 --height=480 --width=800 --gpu_id=0 --suffix=10_frames --frames_per_video=10 --input_dir=/data/p288722/runtime_data/scd_videos_first_revision/09_triplet_inputs/50_frames_pred/mobile_net/models/MobileNet_50_I_frames_ccrop_run1
#python3 /home/p288722/git_code/scd_videos_first_revision/run_evaluate.py --homo_or_not=None --eval_set=test --dataset=/data/p288722/datasets/vision/I_frame_splits/bal_all_frames --batch_size=64 --height=480 --width=800 --gpu_id=0 --suffix=25_frames --frames_per_video=25 --input_dir=/data/p288722/runtime_data/scd_videos_first_revision/09_triplet_inputs/50_frames_pred/mobile_net/models/MobileNet_50_I_frames_ccrop_run1

#python3 /home/p288722/git_code/scd_videos_first_revision/run_evaluate.py --homo_or_not=None --eval_set=test --dataset=/data/p288722/datasets/vision/I_frame_splits/bal_all_frames --batch_size=64 --height=480 --width=800 --gpu_id=0 --suffix=100_frames --frames_per_video=100 --input_dir=/data/p288722/runtime_data/scd_videos_first_revision/09_triplet_inputs/50_frames_pred/mobile_net/models/MobileNet_50_I_frames_ccrop_run1
#python3 /home/p288722/git_code/scd_videos_first_revision/run_evaluate.py --homo_or_not=None --eval_set=test --dataset=/data/p288722/datasets/vision/I_frame_splits/bal_all_frames --batch_size=64 --height=480 --width=800 --gpu_id=0 --suffix=250_frames --frames_per_video=250 --input_dir=/data/p288722/runtime_data/scd_videos_first_revision/09_triplet_inputs/50_frames_pred/mobile_net/models/MobileNet_50_I_frames_ccrop_run1
#python3 /home/p288722/git_code/scd_videos_first_revision/run_evaluate.py --homo_or_not=None --eval_set=test --dataset=/data/p288722/datasets/vision/I_frame_splits/bal_all_frames --batch_size=64 --height=480 --width=800 --gpu_id=0 --suffix=500_frames --frames_per_video=500 --input_dir=/data/p288722/runtime_data/scd_videos_first_revision/09_triplet_inputs/50_frames_pred/mobile_net/models/MobileNet_50_I_frames_ccrop_run1
