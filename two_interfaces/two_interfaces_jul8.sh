#!/bin/bash
#SBATCH --job-name=two_interfaces_jul8_ghm
#SBATCH --cpus-per-task=32
#SBATCH --time=6:00:00
#SBATCH --output=two_interfaces_jul18_ghm.out
#SBATCH --mail-user=n2costa@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --mem=16G

module load python

srun python -u interfere_long.py --b 8300.0 --sigma_s 10.0 --output 100ps_s10.0
#srun python -u interfere_long.py --b 7904.761904761905 --sigma_s 10.5 --output 100ps_s10.5
srun python -u interfere_long.py --b 7545.454545454545 --sigma_s 11.0 --output 100ps_s11.0
#srun python -u interfere_long.py --b 7217.391304347826 --sigma_s 11.5 --output 100ps_s11.5
srun python -u interfere_long.py --b 6916.666666666667 --sigma_s 12.0 --output 100ps_s12.0
#srun python -u interfere_long.py --b 6640.0 --sigma_s 12.5 --output 100ps_s12.5
srun python -u interfere_long.py --b 6384.615384615385 --sigma_s 13.0 --output 100ps_s13.0

#srun python -u interfere_long.py --b 83000.0 --sigma_s 10.0 --output 1000ps_s10.0
#srun python -u interfere_long.py --b 79047.61904761905 --sigma_s 10.5 --output 1000ps_s10.5
#srun python -u interfere_long.py --b 75454.54545454546 --sigma_s 11.0 --output 1000ps_s11.0
#srun python -u interfere_long.py --b 72173.91304347826 --sigma_s 11.5 --output 1000ps_s11.5
#srun python -u interfere_long.py --b 69166.66666666667 --sigma_s 12.0 --output 1000ps_s12.0
#srun python -u interfere_long.py --b 66400.0 --sigma_s 12.5 --output 1000ps_s12.5
#srun python -u interfere_long.py --b 63846.153846153844 --sigma_s 13.0 --output 1000ps_s13.0

# srun python -u interfere_long.py --b 830.0 --sigma_s 10.0 --output 10ps_s10.0
# srun python -u interfere_long.py --b 790.4761904761905 --sigma_s 10.5 --output 10ps_s10.5
# srun python -u interfere_long.py --b 754.5454545454545 --sigma_s 11.0 --output 10ps_s11.0
# srun python -u interfere_long.py --b 721.7391304347826 --sigma_s 11.5 --output 10ps_s11.5
# srun python -u interfere_long.py --b 691.6666666666666 --sigma_s 12.0 --output 10ps_s12.0
# srun python -u interfere_long.py --b 664.0 --sigma_s 12.5 --output 10ps_s12.5
# srun python -u interfere_long.py --b 638.4615384615385 --sigma_s 13.0 --output 10ps_s13.0