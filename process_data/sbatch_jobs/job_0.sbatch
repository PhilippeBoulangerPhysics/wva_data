#!/bin/bash
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=20G
#SBATCH --time=1-00:00
#SBATCH --job-name=PROT42_sst_0
#SBATCH --output=/scratch/philbou/outerr/%x-%j.out
#SBATCH --error=/scratch/philbou/outerr/%x-%j.err
#SBATCH --account=def-rfajber
#SBATCH --mail-user=philippe.boulanger@mail.mcgill.ca
#SBATCH --mail-type=ALL

export GFDL_BASE=/home/philbou/Isca
export GFDL_ENV=narval.ifort
export GFDL_WORK=/scratch/philbou/isca_work
export GFDL_DATA=/scratch/philbou/isca_data
source /home/philbou/.bashrc
conda activate pro_env
cd /home/philbou/projects/def-rfajber/philbou/process_run/

python process_run.py RT42_sst_0_bucket
