#!/bin/bash
#SBATCH --job-name=dip_maker
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00
#SBATCH --output=debugging_output.out
#SBATCH --mail-user=urajnis@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --mem=128G

module load python/3.13.2

srun python -u realistic_maker.py "$@"