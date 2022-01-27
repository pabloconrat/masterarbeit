#!bin/python3
#SBATCH -J EMIL_pp              # Specify job name
#SBATCH -p prepost             # Use partition prepost
#SBATCH -N 1                   # Specify number of nodes
#SBATCH -n 1                   # Specify max. number of tasks to be invoked
#SBATCH --mem-per-cpu=5300     # Set memory required per allocated CPU
#SBATCH -t 01:00:00            # Set a limit on the total run time
#SBATCH -A bd1022              # Charge resources on this project account
#SBATCH -o EMIL_pp.o%j          # File name for standard and error output

import os
import numpy as np
import xarray as xr 
import argparse
from eofs.xarray import Eof

parser = argparse.ArgumentParser()
parser.add_argument('exp_name')
parser.add_argument('--eofs', action='store_true', default=False)
args = vars(parser.parse_args())

exp_name=args['exp_name']
eof_analysis_wanted=args['eofs']
inpath=f'/work/bd1022/b381739/{exp_name}'
outpath=f'/work/bd1022/b381739/{exp_name}/postprocessed'
#inpath=f'/mnt/c/Users/pablo/Nextcloud/3_Mastersemester/Masterarbeit/test_files'
#outpath=f'/mnt/c/Users/pablo/Nextcloud/3_Mastersemester/Masterarbeit/test_files/postprocessed'

try: 
    os.makedirs(outpath)
except FileExistsError:
    pass

os.chdir(inpath)
model_files = [fi for fi in os.listdir(inpath) if fi.endswith('vaxtra.nc')]

var_list = ['um1', 'vm1', 'vervel', 'tm1', 'aps', 'geopot_p']

def read_model_output(infiles):
    ds_list = []
    var_list_sel = []
    test_file = xr.open_dataset(infiles[0], engine='netcdf4')
    for var in var_list:
        if var in test_file:
            var_list_sel.append(var)

    for file in infiles:
        ds_list.append(xr.open_dataset(file, engine='netcdf4')[var_list_sel])
    ds = xr.concat(ds_list, dim='time')
    return ds

# model data
md = read_model_output(model_files).sortby('time').sel(time=slice('2000', '2003'))
print('model data read in')

if eof_analysis_wanted:
    # deseasonalize, detrend, crop, area-weight
    x = md.sel(lat=slice(90,20)).aps.groupby("time.month") - md.sel(lat=slice(90,20)).aps.groupby("time.month").mean()
    x = x - x.mean('time')
    x = x * np.sqrt(np.cos(np.deg2rad(x.lat)))

    
    solver = Eof(x)
    eofs = solver.eofs()
    pcs = solver.pcs(pcscaling=1)
    var_explained = solver.varianceFraction()

    eofs['var_exp'] = var_explained
    eofs.to_netcdf(f'{outpath}/{exp_name}_eofs.nc')
    pcs.to_netcdf(f'{outpath}/{exp_name}_pcs.nc')


# zonal averages and deviations
md_zm = md.mean('lon')
md_anom = md - md_zm
print('zonal averages and anomalies computed')

# momentum transport
vu_mt = md_zm.vm1 * md_zm.um1 * np.cos(np.radians(md_zm.lat))
vu_et = (md_anom.vm1 * md_anom.um1 ).mean('lon') * np.cos(np.radians(md_zm.lat))
print('momentum transport calculated')

# heat transport
vT_mt = md_zm.vm1 * md_zm.tm1 * np.cos(np.radians(md_zm.lat))
vT_et = (md_anom.vm1 * md_anom.tm1).mean('lon') * np.cos(np.radians(md_zm.lat))
print('heat transport calculated')

# eddy kinetic energy
eke = ((md_anom.vm1**2).mean('lon')
       + (md_anom.um1**2).mean('lon')) * 0.5
md_zm['eke'] = eke

vu = xr.Dataset({'vu_mt': vu_mt, 'vu_et': vu_et})
vT = xr.Dataset({'vT_mt': vT_mt, 'vT_et': vT_et})

md.sel(lat=slice(90,0)).sel(plev=[200,500,800,1000], method='nearest').to_netcdf(f'{outpath}/{exp_name}_pl_sel.nc')
md_zm.to_netcdf(f'{outpath}/{exp_name}_zm_pp.nc')
vu.to_netcdf(f'{outpath}/{exp_name}_vu_pp.nc')
vT.to_netcdf(f'{outpath}/{exp_name}_vT_pp.nc')
print('\t done')
