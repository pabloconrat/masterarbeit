#!/bin/bash 
#SBATCH -J EMIL_pp             # Specify job name
#SBATCH -p shared              # Use partition prepost
#SBATCH --mem=25000
#SBATCH -t 01:00:00            # Set a limit on the total run time
#SBATCH -A bd1022              # Charge resources on this project account
#SBATCH -o EMIL_pp_%j.out      # File name for standard and error output

python3 /home/b/b381739/masterarbeit/postprocessing.py EMIL_SW120f100 --years 2000 2001 --eofs

