#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0-01:00
#SBATCH --job-name=add_new_diags_realistic_continents_T85_2moments_rrtm_qflux
#SBATCH --output=/scratch/philbou/outerr/experiment_process/%x-%j.out
#SBATCH --error=/scratch/philbou/outerr/experiment_process/%x-%j.err
#SBATCH --account=def-rfajber

#conda init bash
#conda activate isca_env

source /home/philbou/.bashrc 
conda activate pro_env

cd /home/philbou/projects/def-rfajber/philbou/wva_data/process_run

python create_new_diag.py realistic_continents_T85_2moments_rrtm_qflux 24 36

