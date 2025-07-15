#!/bin/bash
#SBATCH --job-name=dip_maker
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00
#SBATCH --output=debugging_output.out
#SBATCH --mail-user=urajnis@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --mem=128G

srun python -u dip_maker.py "$@"