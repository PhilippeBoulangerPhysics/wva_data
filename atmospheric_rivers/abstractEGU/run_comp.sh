#!/bin/bash
#SBATCH --job-name=fast_diag
#SBATCH --array=0-29           # 30 jobs total (0 to 29)
#SBATCH --cpus-per-task=1      # 1 CPU is enough since we run sequentially
#SBATCH --mem=5G               # 8GB is plenty (runs 1 month at a time)
#SBATCH --time=03:00:00        # Give enough time for 10 months (approx 10-15m each?)
#SBATCH --output=/scratch/philbou/outerr/diag_batch_%a.out  # Separate logs for each batch
#SBATCH --error=/scratch/philbou/outerr/diag_batch_%a.err

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
START_MONTH_BASE=300
BATCH_SIZE=10
# ---------------------------------------------------------

# Calculate the range for THIS job
# Example: If Array ID is 0 -> Start = 300
# Example: If Array ID is 1 -> Start = 310
MY_START=$(( START_MONTH_BASE + (SLURM_ARRAY_TASK_ID * BATCH_SIZE) ))
MY_END=$(( MY_START + BATCH_SIZE - 1 ))

echo "Job Array ID: $SLURM_ARRAY_TASK_ID"
echo "Processing Batch: Month $MY_START to $MY_END"

source /home/philbou/.bashrc 
conda activate pro_env

cd /home/philbou/projects/def-rfajber/philbou/abstractEGU

# Loop through the batch
for (( month=MY_START; month<=MY_END; month++ ))
do
    echo "--------------------------------------"
    echo "Running Month: $month"
    
    # Run your python script for ONE month
    # -u forces prints to appear immediately in the log
    python -u compo_worker.py $month
    
    # Check if python failed
    if [ $? -ne 0 ]; then
        echo "ERROR: Month $month failed!"
        # Optional: exit 1  # Uncomment to stop the whole batch on error
    fi
done

echo "Batch Complete."