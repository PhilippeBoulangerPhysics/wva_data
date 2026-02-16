#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=5G
#SBATCH --time=0-1:00
#SBATCH --job-name=precip_age_0test
#SBATCH --output=/scratch/philbou/outerr/%x-%j.out
#SBATCH --error=/scratch/philbou/outerr/%x-%j.err
#SBATCH --account=def-rfajber

#not sure if this is needed but just in case 
# directory of the Isca source code
export GFDL_BASE=/home/philbou/Isca 
# &quot;environment&quot; configuration for emps-gv4
export GFDL_ENV=narval.ifort
# temporary working directory used in running the model
export GFDL_WORK=/scratch/philbou/isca_work
# directory for storing model output
export GFDL_DATA=/scratch/philbou/isca_data

#conda init bash
#conda activate isca_env

source /home/philbou/.bashrc 
conda activate pro_env

cd /home/philbou/projects/def-rfajber/philbou/process_run

python process_run.py
#python precipitation_age.py "RT42_sst_m3_5yrs" 
#python precipitation_age.py "RT42_sst_3_5yrs" 
#python precipitation_age.py "RT42_sst_m4_5yrs" 
#python precipitation_age.py "RT42_sst_2_5yrs" 
#python precipitation_age.py "RT42_sst_m2_5yrs" 
#python precipitation_age.py "RT42_sst_0_5yrs" 
#python precipitation_age.py "RT42_sst_4_5yrs" 
