#!/bin/bash
#SBATCH --ntasks=36
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0-3:00
#SBATCH --job-name=diag_precip_age_shape_T85
#SBATCH --output=/scratch/philbou/outerr/%x-%j.out
#SBATCH --error=/scratch/philbou/outerr/%x-%j.err
#SBATCH --account=def-rfajber


source /home/philbou/.bashrc 
conda activate pro_env

cd /home/philbou/projects/def-rfajber/philbou/process_run

srun python worker.py