#!/bin/bash
#SBATCH --job-name=two_interfaces_jul29_nv
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=two_interfaces_jul29_nv.out
#SBATCH --mail-user=n2costa@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --mem=128G

module load python

srun python -u interfaces_jul23.py --b 8300.0 --sigma_s 10.0 --dband 0.005 --pband 0.1 --output two_interfaces_jul29