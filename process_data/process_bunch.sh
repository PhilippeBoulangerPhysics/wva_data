#!/bin/bash

# === CONFIGURATION ===
SCRIPT="process_run.py"
SBATCH_TEMPLATE_DIR="./sbatch_jobs"
mkdir -p "$SBATCH_TEMPLATE_DIR"

# === ENVIRONMENT SETUP BLOCK ===
read -r -d '' ENV_BLOCK << 'EOF'
export GFDL_BASE=/home/philbou/Isca
export GFDL_ENV=narval.ifort
export GFDL_WORK=/scratch/philbou/isca_work
export GFDL_DATA=/scratch/philbou/isca_data
source /home/philbou/.bashrc
conda activate pro_env
cd /home/philbou/projects/def-rfajber/philbou/process_run/

EOF

# === GENERATE AND SUBMIT JOBS FOR RANGE -4 TO 4 (INCLUDE 0) ===
for i in 0 ; do
    if (( i < 0 )); then
        abs=${i#-}            # absolute value of negative number
        JOB_SUFFIX="m${abs}"  # e.g. m4, m3, m2, m1
    else
        JOB_SUFFIX="${i}"     # e.g. 0, 1, 2, 3, 4
    fi

    JOB_NAME="PROT42_sst_${JOB_SUFFIX}"
    JOB_FILE="$SBATCH_TEMPLATE_DIR/job_${JOB_SUFFIX}.sbatch"

    cat <<EOF > "$JOB_FILE"
#!/bin/bash
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=20G
#SBATCH --time=1-00:00
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=/scratch/philbou/outerr/%x-%j.out
#SBATCH --error=/scratch/philbou/outerr/%x-%j.err
#SBATCH --account=def-rfajber
#SBATCH --mail-user=philippe.boulanger@mail.mcgill.ca
#SBATCH --mail-type=ALL

$ENV_BLOCK

python $SCRIPT RT42_sst_${JOB_SUFFIX}_bucket
EOF
    sbatch "$JOB_FILE"
    echo "Submitted $JOB_FILE with argument $i (job name: $JOB_NAME)"
done