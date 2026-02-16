#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1             # Run 1 main script
#SBATCH --cpus-per-task=4     # Give that script 6 cores to use
#SBATCH --mem-per-cpu=5G      # 10G per core = 60G total RAM
#SBATCH --time=02:00:00
#SBATCH --job-name=cummul_300
#SBATCH --output=/scratch/philbou/outerr/%x-%j.out
#SBATCH --error=/scratch/philbou/outerr/%x-%j.err
#SBATCH --account=def-rfajber


source /home/philbou/.bashrc 
conda activate pro_env

cd /home/philbou/projects/def-rfajber/philbou/abstractEGU

srun python composite.py