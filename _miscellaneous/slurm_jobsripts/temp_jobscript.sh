#!/bin/bash

#SBATCH --job-name=check_frames
#SBATCH --time=04:00:00
#SBATCH --mem=4GB
# --partition=short
#SBATCH --array=11-13,20,27

module purge
module load OpenBLAS/0.3.15-GCC-10.3.0
source /data/p288722/python_venv/scd_videos_first_revision/bin/activate

python3 /home/p288722/git_code/scd_videos_first_revision/dataset/frames/temp.py --device_id="${SLURM_ARRAY_TASK_ID}"
