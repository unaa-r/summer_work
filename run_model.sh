#!/bin/bash
#SBATCH --job-name=dip_maker
#SBATCH --cpus-per-task=64
#SBATCH --time=4:00:00
#SBATCH --output=test_output.out
#SBATCH --mail-user=urajnis@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --mem=128G

module load python/3.13.2

srun python -u dip_maker.py "$@"