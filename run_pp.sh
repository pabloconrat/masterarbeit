#!/bin/sh -l
#SBATCH -A bd1022
#SBATCH --partition=compute
#SBATCH --time=01:00:00

module load python3

python3 /home/b/b381739/masterarbeit/postprocessing.py EMIL_sf6_w2_a4 