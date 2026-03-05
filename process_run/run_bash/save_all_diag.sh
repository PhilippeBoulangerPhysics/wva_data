#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=0-12:00
#SBATCH --job-name=save_all_diags_realistic_continents_T85_2moments_rrtm_qflux
#SBATCH --output=/scratch/philbou/outerr/experiment_process/%x-%j.out
#SBATCH --error=/scratch/philbou/outerr/experiment_process/%x-%j.err
#SBATCH --account=def-rfajber

#conda init bash
#conda activate isca_env

source /home/philbou/.bashrc 
conda activate pro_env

cd /home/philbou/projects/def-rfajber/philbou/wva_data/process_run

python save_all_diag.py realistic_continents_T85_2moments_rrtm_qflux_0.1_0 60 71

