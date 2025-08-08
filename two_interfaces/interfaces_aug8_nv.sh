#!/bin/bash
#SBATCH --job-name=two_interfaces_aug8_nv
#SBATCH --array=0-9
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=two_interfaces_aug8_nv.out
#SBATCH --mail-user=n2costa@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --mem=128G

module load python

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" params.txt)

read A band <<< "$line"

echo "Running with A=$A, bandwidth=$band"

outfile = "two_interfaces_aug8_A${A}_band${band}"

srun python -u interfaces_aug6.py --b 8300.0 --sigma_s 10.0 --dband $band --pband 0.1 --output outfile --A $A