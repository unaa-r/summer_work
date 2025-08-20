#!/bin/bash
#SBATCH --job-name=interfaces_aug19a
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=interfaces_aug19a.out
#SBATCH --mail-user=n2costa@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --mem=128G

module load python

srun python -u interfaces_aug6_runa.py --b 8300.0 --sigma_s 10.0 --dband 0.05 --pband 1 --output interfaces_aug19a --A 18033.7