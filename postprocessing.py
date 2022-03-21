#!bin/python3
#SBATCH -J EMIL_pp             # Specify job name
#SBATCH -p shared         # Use partition prepost
#SBATCH --mem=20000
#SBATCH -t 01:00:00            # Set a limit on the total run time
#SBATCH -A bd1022              # Charge resources on this project account
#SBATCH -o EMIL_pp_%j.out      # File name for standard and error output

import os
import numpy as np
import xarray as xr 
import argparse

def xr_spatial_fft_analysis(data):
    
    dx = data.lon.values[1] - data.lon.values[0]
    dim='k'
    result =  xr.apply_ufunc(
        np.fft.fft, data, input_core_dims=[['lon']], output_core_dims=[[dim]],
        kwargs={'norm':'ortho'}
    )
    k = np.fft.fftfreq(result.k.shape[0], d=dx/(360))
    result[dim] = k
    return result

def save_complex(data_array, *args, **kwargs):
    ds = xr.Dataset({'real': data_array.real, 'imag': data_array.imag})
    return ds.to_netcdf(*args, **kwargs)

def read_model_output(infiles):
    ds_list = []
    for file in infiles:
        ds_file = xr.open_dataset(file, engine='netcdf4')[var_list_sel]
        ds_file.fillna(value=fillna_values)
        ds_list.append(ds_file)
    ds = xr.combine_by_coords(ds_list, combine_attrs='drop_conflicts', data_vars='minimal')
    for ds_f in ds_list:
        ds_f.close()
    return ds

def postprocessing(model_files, year):

    files_in_year = [mf for mf in model_files if year in mf ]
    files_in_year.sort()
    # model data
    md = read_model_output(files_in_year).sortby('time')
    print(f'{year}: model data read in')

    if eof_analysis_wanted:
        from eofs.xarray import Eof
        # deseasonalize, detrend, crop, area-weight
        x = md.sel(lat=slice(90,20)).aps.groupby("time.month") - md.sel(lat=slice(90,20)).aps.groupby("time.month").mean()
        x = x - x.mean('time')
        x = x * np.sqrt(np.cos(np.deg2rad(x.lat)))

        solver = Eof(x)
        eofs = solver.eofs()
        pcs = solver.pcs(pcscaling=1)
        var_explained = solver.varianceFraction()

        eofs['var_exp'] = var_explained
        eofs.to_netcdf(f'{outpath}/{exp_name}_{year}_eofs.nc')
        pcs.to_netcdf(f'{outpath}/{exp_name}_{year}_pcs.nc')


    # zonal averages and deviations
    md_zm = md.mean('lon')
    md_anom = md - md_zm
    print(f'{year}: zonal averages and anomalies computed')
    cos_lat = np.cos(np.radians(md_zm.lat))

    # momentum transport
    vu_mt = md_zm.vm1 * md_zm.um1 * cos_lat
    vu_et = (md_anom.vm1 * md_anom.um1 ).mean('lon') * cos_lat
    print(f'{year}: momentum transport calculated')

    # heat transport
    vT_mt = md_zm.vm1 * md_zm.tm1 * cos_lat
    vT_et = (md_anom.vm1 * md_anom.tm1).mean('lon') * cos_lat
    print(f'{year}: heat transport calculated')

    wu_mt = md_zm.vervel * md_zm.um1 * cos_lat
    wu_et = (md_anom.vervel * md_anom.um1).mean('lon') * cos_lat
    print(f'{year}: vertical transport of zonal momentum calculated')

    transports = xr.Dataset({'vu_mt': vu_mt, 'vu_et': vu_et, 'vT_mt': vT_mt, 'vT_et': vT_et, 'wu_mt': wu_mt, 'wu_et': wu_et})

    # dry static energy transport
    cp = 1004
    g = 9.81
    if 'geopot_p' in md:
        dse = cp * md.tm1 + md.geopot_p
        dse_anom = dse - dse.mean('lon')
        vdse_mt = md_zm.vm1 * dse.mean('lon') * cos_lat
        vdse_et = (md_anom.vm1 * dse_anom).mean('lon') * cos_lat
        transports['vdse_mt'] = vdse_mt
        transports['vdse_et'] = vdse_et


    # eddy kinetic energy

    eke = ((md.vm1**2) + (md.um1**2)) * 0.5
    eke_fft = xr_spatial_fft_analysis(eke)
    eke_fft
    if ml is True:
        save_complex(eke_fft.sel(k=slice(0,18), lev=slice(60,90)), f'{outpath}/{exp_name}_{year}_eke_fft.nc')
    else:
        save_complex(eke_fft.sel(k=slice(0,18), plev=slice(100,1000)), f'{outpath}/{exp_name}_{year}_eke_fft.nc')
    eke_zm = eke.mean('lon')
    md_zm['eke'] = eke_zm


    if ml is False:
        md.sel(lat=slice(90,0)).sel(plev=[200,300,500,700,850], method='nearest').to_netcdf(f'{outpath}/{exp_name}_{year}_pl_sel.nc')    
    else:
        md.sel(lat=slice(90,0)).sel(lev=[70,74,80,83,85]).to_netcdf(f'{outpath}/{exp_name}_{year}_ml_sel.nc')    
    md_zm.to_netcdf(f'{outpath}/{exp_name}_{year}_zm_pp.nc')
    transports.to_netcdf(f'{outpath}/{exp_name}_{year}_transports_pp.nc')

    print(f'{year}: \t done')
    return

parser = argparse.ArgumentParser()
parser.add_argument('exp_name')
parser.add_argument('--ml', action='store_true', default=False)
parser.add_argument('--eofs', action='store_true', default=False)
args = vars(parser.parse_args())

exp_name=args['exp_name']
eof_analysis_wanted=args['eofs']
ml=args['ml']
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
years = [model_file[15:19] for model_file in model_files]
years = list(set(years))
years.sort()
print(exp_name)
print(years)

var_list = ['um1', 'vm1', 'vervel', 'tm1', 'aps', 'geopot_p', 'hyam', 'hybm']

var_list_sel = []
ds_file = xr.open_dataset(model_files[0], engine='netcdf4')
for var in var_list:
    if var in ds_file:
        var_list_sel.append(var)

ds_file.close()
print(var_list_sel)

# set NaN in the wind fields to zero to reduce spurius averages of only a few points
fillna_values = {"um1": 0, "vm1": 0, 'vervel': 0}

for year in years:
    postprocessing(model_files, year)
