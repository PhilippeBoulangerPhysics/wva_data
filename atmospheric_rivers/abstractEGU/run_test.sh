#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6    # 24 Parallel workers
#SBATCH --mem=96G             # 4GB per worker (Safe margin)
#SBATCH --job-name=comp_testT85      
#SBATCH --output=/scratch/philbou/outerr/%x-%j.out
#SBATCH --error=/scratch/philbou/outerr/%x-%j.err
#SBATCH --account=def-rfajber


source /home/philbou/.bashrc 
conda activate pro_env

cd /home/philbou/projects/def-rfajber/philbou/abstractEGU

python composite.py