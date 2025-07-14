#!/bin/bash
#SBATCH --job-name=realistic_SLM_model
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=test_output.out
#SBATCH --mail-user=urajnis@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --mem=128G

srun python -u realistic_maker.py "$@"