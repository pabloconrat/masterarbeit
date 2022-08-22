
import os
import numpy as np
import xarray as xr 
import argparse
from pathlib import Path


def read_model_output(infiles, var_list_sel=None, fillna=False):
    ds_list = []
    for file in infiles:
        if var_list_sel is None:
            ds_file = xr.open_dataset(file, engine='netcdf4')
        else:
            ds_file = xr.open_dataset(file, engine='netcdf4')[var_list_sel]

        if fillna:
            ds_file.fillna(value=fillna_values)
        ds_list.append(ds_file)
    ds = xr.combine_by_coords(ds_list, combine_attrs='drop_conflicts', data_vars='minimal')
    for ds_f in ds_list:
        ds_f.close()
    return ds

def merge_pp_files(path, types):
    for tp in types:
        files = [fi for fi in os.listdir(path) if fi.endswith(f'_{tp}.nc')]
        if files:
            ds = read_model_output(files)
            ds.to_netcdf(f'{exp_name}_{tp}.nc')
            [os.remove(file) for file in files]
    return

parser = argparse.ArgumentParser()
parser.add_argument('spinup_name')
parser.add_argument('spinon_name')
parser.add_argument('exp_name')
args = vars(parser.parse_args())

exp_name=args['exp_name']
spinup_name=args['spinup_name']
spinon_name=args['spinon_name']
output_endings=('eke_fft.nc','pl_sel.nc','transports_int_pp.nc','transports_pp.nc', 'zm_pp.nc', 'ep_pp.nc', 'zmom_eq_pp.nc', 'zmom_tem_eq_pp.nc')


inpath=f'/work/bd1022/b381739/{exp_name}/postprocessed'
inpath_sp=f'/work/bd1022/b381739/{spinup_name}/postprocessed'
inpath_sp2=f'/work/bd1022/b381739/{spinon_name}/postprocessed'
outpath=inpath

os.chdir(inpath)
exp_files = [fi for fi in os.listdir(inpath) if fi.endswith(output_endings)]
os.chdir(inpath_sp)
spinup_files = [fi for fi in os.listdir(inpath_sp) if fi.endswith(output_endings)]
os.chdir(inpath_sp2)
spinon_files = [fi for fi in os.listdir(inpath_sp2) if fi.endswith(output_endings)]

exp_files.sort()
spinup_files.sort()
spinon_files.sort()

print(spinup_files)
print(spinon_files)
print(exp_files)

assert len(spinup_files) == len(spinon_files) == len(exp_files), 'Postprocessed output does not match'

ds_list_e = []
ds_list_s = []
ds_list_s2 = []
ds_list_f = []

print('opening spinup files from: ' + inpath_sp)
os.chdir(inpath_sp)
for i in range(len(spinup_files)):
    ds_list_s.append(xr.open_dataset(spinup_files[i]))

os.chdir(inpath_sp2)
print('opening spinon files from: ' + inpath_sp2)
for i in range(len(spinon_files)):
    ds_list_s2.append(xr.open_dataset(spinon_files[i]))

print('opening files from: ' + inpath)
os.chdir(inpath)
for i in range(len(exp_files)):
    ds_list_e.append(xr.open_dataset(exp_files[i]))

print('calculating start and transition dates')
trans_time = ds_list_e[0].time[0]
start_time = ds_list_s[0].time[0]
init_time = ds_list_s2[0].time[0]

print(f'start/end spinup: {ds_list_s[i].time[0].values}, {ds_list_s[i].time[-1].values}')
print(f'start/end spinon: {ds_list_s2[i].time[0].values}, {ds_list_s2[i].time[-1].values}')
print(f'start/end exp: {ds_list_e[i].time[0].values}, {ds_list_e[i].time[-1].values}')

if 'trans_date' in ds_list_e[0]:
    raise RuntimeError('Output has already been merged!')

print('concatenating files')
for i in range(len(exp_files)):

    ds_spinup_select = ds_list_s[i].sel(time=slice(start_time, init_time - np.timedelta64(1, 'D')))
    ds_spinon_select = ds_list_s2[i].sel(time=slice(init_time, trans_time - np.timedelta64(1, 'D')))

    ds_list_f.append(xr.concat([ds_spinup_select, ds_spinon_select, ds_list_e[i]], dim='time'))
    ds_list_e[i].close()

os.chdir(outpath)
print('writing output')
for i,ds in enumerate(ds_list_f):
    ds['trans_date'] = trans_time
    # Save output from one year before the transition onwards
    ds.sel(
        time=slice(trans_time.values - np.timedelta64(365, 'D'), ds.time[-1].values)
    ).to_netcdf(f'{exp_files[i]}')