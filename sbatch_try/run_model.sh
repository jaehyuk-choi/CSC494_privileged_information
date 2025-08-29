#!/bin/bash
#SBATCH --job-name=gradient_boosting
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --mem=10G
#SBATCH --time=6:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

eval "$(conda shell.bash hook)"
conda activate myenv    # 여기를 네 실제 conda env 이름으로 교체!

model=$1; shift

if [ "$model" != "GradientBoosting" ]; then
    echo "Unknown model: $model"
    exit 1
fi

learning_rate=$1; shift
n_estimators=$1; shift
max_depth=$1; shift

python train.py \
    --learning_rate $learning_rate \
    --n_estimators $n_estimators \
    --max_depth $max_depth
