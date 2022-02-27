#!/bin/bash

#SBATCH --job-name=all_frames
#SBATCH --time=12:00:00
#SBATCH --mem=4GB
# --partition=short
#SBATCH --array=0-27

module purge
module load OpenBLAS/0.3.15-GCC-10.3.0

source /data/p288722/python_venv/scd_videos_first_revision/bin/activate

python3 /home/p288722/git_code/scd_videos_first_revision/dataset/frames/1_frame_extraction_all_frames.py --input_dir="/data/p288722/datasets/VISION/dataset" --output_dir="/scratch/p288722/datasets/vision/all_frames" --device_id=${SLURM_ARRAY_TASK_ID}
#python3 /home/p288722/git_code/scd_videos_first_revision/dataset/frames/1_frame_extraction_I_frames.py --input_dir="/data/p288722/datasets/VISION/dataset" --output_dir="/scratch/p288722/datasets/vision/all_I_frames" --device_id="${SLURM_ARRAY_TASK_ID}"
