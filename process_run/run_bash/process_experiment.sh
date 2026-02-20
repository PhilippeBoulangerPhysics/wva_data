#!/bin/bash
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=6G
#SBATCH --time=3-00:00
#SBATCH --job-name=RT44_sst_0_processing_20years
#SBATCH --output=/scratch/philbou/outerr/experiment_process/%x-%j.out
#SBATCH --error=/scratch/philbou/outerr/experiment_process/%x-%j.err
#SBATCH --account=def-rfajber

#conda init bash
#conda activate isca_env

source /home/philbou/.bashrc 
conda activate pro_env

cd /home/philbou/projects/def-rfajber/philbou/masters/process_run

python process_experiment.py RT42_sst_0_bucket 360 600

