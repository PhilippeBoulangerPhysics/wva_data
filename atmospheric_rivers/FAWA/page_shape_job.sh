#!/bin/bash
#SBATCH --array=60-96%12   # Run jobs 60 to 300, but only 10 at once
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G            # Memory per SINGLE job
#SBATCH --time=03:30:00      # Time per SINGLE month
#SBATCH --job-name=diag_moist_T85
#SBATCH --output=/scratch/philbou/outerr/%x-%j.out
#SBATCH --error=/scratch/philbou/outerr/%x-%j.err
#SBATCH --account=def-rfajber

# $SLURM_ARRAY_TASK_ID is the number (60, 61, ... 300)



source /home/philbou/.bashrc 
conda activate pro_env

cd /home/philbou/projects/def-rfajber/philbou/FAWA

python worker.py $SLURM_ARRAY_TASK_ID
