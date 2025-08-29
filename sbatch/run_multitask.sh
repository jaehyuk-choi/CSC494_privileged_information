#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Activate your environment
eval "$(conda shell.bash hook)"
conda activate myenv

# The first argument is the JSON payload
PARAM_JSON="$1"

# Execute the multitask script with this JSON
python multitask.py --param_json "$PARAM_JSON" --num_runs 10