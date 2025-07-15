#!/bin/bash
#SBATCH --job-name=realistic_SLM_model
#SBATCH --cpus-per-task=32
#SBATCH --time=8:00:00
#SBATCH --output=test_output.out
#SBATCH --mail-user=urajnis@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --mem=128G

module load python/3.13.2

srun python -u realistic_maker.py "$@"