#!/bin/bash
#SBATCH --job-name=mn-50Iframes-prnu
#SBATCH --time=24:00:00
#SBATCH --mem=64g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
# --reservation=infsys
#SBATCH --cpus-per-task=12
#SBATCH --array=1
# --dependency=afterok:22266219

# create the following directory manually
#SBATCH --chdir=/scratch/p288722/runtime_data/scd_videos_first_revision/10_prnu_bs64
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

num_frames=50
base_dir=$(pwd)
splits_dir="/scratch/p288722/datasets/vision/I_frame_splits/bal_${num_frames}_frames"
#splits_dir="/scratch/p288722/datasets/vision/rand_frames_split/bal_${num_frames}_frames"
net="mobile"
const_type="None"

homo_or_not="None" # also change the model_name accordingly

case ${net} in
  "mobile") model_name="MobileNet_${num_frames}_I_frames_prnu_run${SLURM_ARRAY_TASK_ID}" ;;
  "eff") model_name="EfficientNet_${num_frames}_I_frames_prnu_run${SLURM_ARRAY_TASK_ID}" ;;
  "misl") model_name="MISLNet_${num_frames}_I_frames_prnu_run${SLURM_ARRAY_TASK_ID}" ;;
  "res") model_name="ResNet_${num_frames}_I_frames_prnu_run${SLURM_ARRAY_TASK_ID}" ;;
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

echo python3 /home/p288722/git_code/scd_videos_first_revision/run_train.py --homo_or_not=${homo_or_not} --net_type=${net} --dataset=${splits_dir} --epochs=20 --lr=0.1 --batch_size=64 --height=480 --width=800 --use_pretrained=1 --gpu_id=0 --const_type=${const_type} --model_name="${model_name}" --global_results_dir="${base_dir}/${num_frames}_frames/${net}_net"
echo python3 /home/p288722/git_code/scd_videos_first_revision/run_evaluate.py --homo_or_not=${homo_or_not} --eval_set="val" --dataset=${splits_dir} --batch_size=64 --height=480 --width=800 --gpu_id=0 --suffix="${num_frames}_frames_val" --input_dir="${base_dir}/${num_frames}_frames/${net}_net/models/${model_name}"
echo python3 /home/p288722/git_code/scd_videos_first_revision/_miscellaneous/plots/validation_plots.py --val_summary="${base_dir}/${num_frames}_frames/${net}_net/models/${model_name}/predictions_${num_frames}_frames_val/videos/V_prediction_stats.csv"
echo python3 /home/p288722/git_code/scd_videos_first_revision/run_evaluate.py --homo_or_not=${homo_or_not} --eval_set="test" --dataset=${splits_dir} --batch_size=64 --height=480 --width=800 --gpu_id=0 --suffix="${num_frames}_frames" --input_dir="${base_dir}/${num_frames}_frames_pred/${net}_net/models/${model_name}"


python3 /home/p288722/git_code/scd_videos_first_revision/run_train.py --homo_or_not=${homo_or_not} --net_type=${net} --dataset=${splits_dir} --epochs=20 --lr=0.1 --batch_size=64 --height=480 --width=800 --use_pretrained=1 --gpu_id=0 --const_type=${const_type} --model_name="${model_name}" --global_results_dir="${base_dir}/${num_frames}_frames/${net}_net"
python3 /home/p288722/git_code/scd_videos_first_revision/run_evaluate.py --homo_or_not=${homo_or_not} --eval_set="val" --dataset=${splits_dir} --batch_size=64 --height=480 --width=800 --gpu_id=0 --suffix="${num_frames}_frames_val" --input_dir="${base_dir}/${num_frames}_frames/${net}_net/models/${model_name}"
python3 /home/p288722/git_code/scd_videos_first_revision/_miscellaneous/plots/validation_plots.py --val_summary="${base_dir}/${num_frames}_frames/${net}_net/models/${model_name}/predictions_${num_frames}_frames_val/videos/V_prediction_stats.csv"
python3 /home/p288722/git_code/scd_videos_first_revision/run_evaluate.py --homo_or_not=${homo_or_not} --eval_set="test" --dataset=${splits_dir} --batch_size=64 --height=480 --width=800 --gpu_id=0 --suffix="${num_frames}_frames" --input_dir="${base_dir}/${num_frames}_frames_pred/${net}_net/models/${model_name}"

