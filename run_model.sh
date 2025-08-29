#!/bin/bash
#SBATCH --job-name=gridsearch           # Slurm job name
#SBATCH --output=logs/%x_%j.out         # Standard output log: job name + job ID
#SBATCH --error=logs/%x_%j.err          # Standard error log
#SBATCH --time=02:00:00                 # Max runtime (hh:mm:ss)
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --cpus-per-task=4               # Request 4 CPU cores for the task
#SBATCH --mem=16G                       # Request 16 GB RAM
#SBATCH --partition=rtx6000             # Partition/queue name

# -------------------------
# Load environment
# -------------------------
source ~/.bashrc
conda activate myenv

# -------------------------
# Read JSON-encoded parameters from sbatch argument
# -------------------------
PARAM_JSON="$1"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job $SLURM_JOB_ID — Params: $PARAM_JSON"

# -------------------------
# Execute main.py with given parameters
# -------------------------
python parallel.py --param_json "$PARAM_JSON"