#!/bin/bash
#SBATCH --job-name=two_interfaces_jul15_ghm
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=two_interfaces_jul15_nv.out
#SBATCH --mail-user=n2costa@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --mem=64G

module load python

srun python -u interfaces_jul15.py --b 8300.0 --sigma_s 10.0 --output 2int_jul15