#!/bin/bash
#SBATCH --job-name=two_interfaces_jul22_nv
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=two_interfaces_jul22_nv.out
#SBATCH --mail-user=n2costa@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --mem=128G

module load python

srun python -u interfaces_jul21.py --b 8300.0 --sigma_s 10.0 --dband 1.0 --pband 1.0 --output 2int_jul22_nv_dband1.0
srun python -u interfaces_jul21.py --b 8300.0 --sigma_s 10.0 --dband 0.5 --pband 1.0 --output 2int_jul22_nv_dband0.5
srun python -u interfaces_jul21.py --b 8300.0 --sigma_s 10.0 --dband 0.1 --pband 0.2 --output 2int_jul22_nv_dband0.1
srun python -u interfaces_jul21.py --b 8300.0 --sigma_s 10.0 --dband 0.05 --pband 0.1 --output 2int_jul22_nv_dband0.05
srun python -u interfaces_jul21.py --b 8300.0 --sigma_s 10.0 --dband 0.01 --pband 0.05 --output 2int_jul22_nv_dband0.01
srun python -u interfaces_jul21.py --b 8300.0 --sigma_s 10.0 --dband 0.005 --pband 0.01 --output 2int_jul22_nv_dband0.005
srun python -u interfaces_jul21.py --b 8300.0 --sigma_s 10.0 --dband 0.001 --pband 0.002 --output 2int_jul22_nv_dband0.001
srun python -u interfaces_jul21.py --b 8300.0 --sigma_s 10.0 --dband 0.0005 --pband 0.001 --output 2int_jul22_nv_dband0.0005