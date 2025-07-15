#!/bin/bash
#SBATCH --job-name=two_interfaces_jul14nv
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=two_interfaces_jul14nv.out
#SBATCH --mail-user=n2costa@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --mem=128G

module load python

srun python -u interfaces_jul9.py --b 8300.0 --sigma_s 10.0 --output 2int_lin_jul14nv