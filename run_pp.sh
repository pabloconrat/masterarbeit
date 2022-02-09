#!/bin/sh -l
#SBATCH -J EMIL_pp              # Specify job name
#SBATCH -p prepost             # Use partition prepost
#SBATCH -N 1                   # Specify number of nodes
#SBATCH -n 3                   # Specify max. number of tasks to be invoked
#SBATCH --mem-per-cpu=5300     # Set memory required per allocated CPU
#SBATCH -t 01:00:00            # Set a limit on the total run time
#SBATCH -A bd1022              # Charge resources on this project account
#SBATCH -o EMIL_pp_%j.out          # File name for standard and error output
module load python3

python3 /home/b/b381739/masterarbeit/postprocessing.py EMIL_sf6_w2_a4