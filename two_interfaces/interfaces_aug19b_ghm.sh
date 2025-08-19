#!/bin/bash
#SBATCH --job-name=interfaces_aug19b_ghm
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=interfaces_aug19b_ghm.out
#SBATCH --mail-user=n2costa@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --mem=128G

module load python

srun python -u interfaces_aug6_runb.py --b 8300.0 --sigma_s 10.0 --dband 0.005 --pband 0.1 --output interfaces_aug19b_ghm --A 180337