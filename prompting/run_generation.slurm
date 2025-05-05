#!/bin/bash
#SBATCH --job-name=generate_aug_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=36:00:00
#SBATCH --qos=cpu_qos
#SBATCH --partition=cpu
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.e

eval "$(conda shell.bash hook)"

conda activate test_cuda

python explore_data.py

