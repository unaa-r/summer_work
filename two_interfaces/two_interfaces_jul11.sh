#!/bin/bash
#SBATCH --job-name=two_interfaces_jul11_ghm
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH --output=two_interfaces_jul11_ghm.out
#SBATCH --mail-user=n2costa@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --mem=64G

module load python

srun python -u interfaces_jul9.py --b 8300.0 --sigma_s 10.0 --output 2int_lin_jul11
