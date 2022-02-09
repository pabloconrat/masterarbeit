#!bin/python3
#SBATCH -J EMIL_pp              # Specify job name
#SBATCH -p prepost             # Use partition prepost
#SBATCH -N 1                   # Specify number of nodes
#SBATCH -n 6                   # Specify max. number of tasks to be invoked
#SBATCH --mem-per-cpu=5300     # Set memory required per allocated CPU
#SBATCH -t 01:00:00            # Set a limit on the total run time
#SBATCH -A bd1022              # Charge resources on this project account
#SBATCH -o EMIL_pp_%j.out          # File name for standard and error output

import os
import numpy as np
import xarray as xr 
import argparse

def read_model_output(infiles):
    ds_list = []
    for file in infiles:
        ds_file = xr.open_dataset(file, engine='netcdf4')[var_list]
        ds_file = ds_file.isel(plev=slice(85,90))
        ds_list.append(ds_file)
    ds = xr.concat(ds_list, dim='time')
    return ds

parser = argparse.ArgumentParser()
parser.add_argument('exp_name')
parser.add_argument('--ml', action='store_true', default=False)
parser.add_argument('--eofs', action='store_true', default=False)
args = vars(parser.parse_args())

exp_name=args['exp_name']
ml=args['ml']
eof_analysis_wanted=args['eofs']
output_ending = 'vaxtra.nc'

inpath=f'/work/bd1022/b381739/{exp_name}'
outpath=f'/work/bd1022/b381739/{exp_name}/postprocessed'
if ml:
    outpath=f'/work/bd1022/b381739/{exp_name}/postprocessed_ml'
    output_ending='emil.nc'
#inpath=f'/mnt/c/Users/pablo/Nextcloud/3_Mastersemester/Masterarbeit/test_files'
#outpath=f'/mnt/c/Users/pablo/Nextcloud/3_Mastersemester/Masterarbeit/test_files/postprocessed'

try: 
    os.makedirs(outpath)
except FileExistsError:
    pass

os.chdir(inpath)
model_files = [fi for fi in os.listdir(inpath) if fi.endswith(output_ending) and fi.startswith(f'{exp_name}_2')]

var_list = ['vm1','tm1','aps']

md = read_model_output(model_files)
md.to_netcdf(f'{exp_name}_sfc_pl_vT.nc')