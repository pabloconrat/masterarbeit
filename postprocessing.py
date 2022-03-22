#!bin/python3
#SBATCH -J EMIL_pp             # Specify job name
#SBATCH -p shared              # Use partition prepost
#SBATCH --mem=22000
#SBATCH -t 01:00:00            # Set a limit on the total run time
#SBATCH -A bd1022              # Charge resources on this project account
#SBATCH -o EMIL_pp_%j.out      # File name for standard and error output

import os
import numpy as np
import xarray as xr 
import argparse

cp = 1004
g = 9.81
r_e = 6.37e6

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

def read_model_output(infiles, var_list_sel):
    ds_list = []
    for file in infiles:
        ds_file = xr.open_dataset(file, engine='netcdf4')[var_list_sel]
        ds_file.fillna(value=fillna_values)
        ds_list.append(ds_file)
    ds = xr.combine_by_coords(ds_list, combine_attrs='drop_conflicts', data_vars='minimal')
    for ds_f in ds_list:
        ds_f.close()
    return ds

def postprocessing_pl(model_files, year, pl_var_list_sel):

    files_in_year = [mf for mf in model_files if year in mf ]
    files_in_year.sort()
    # model data
    md = read_model_output(files_in_year, var_list_sel=pl_var_list_sel).sortby('time')
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

    tps = xr.Dataset({'vu_mt': vu_mt, 'vu_et': vu_et, 'vT_mt': vT_mt, 'vT_et': vT_et, 'wu_mt': wu_mt, 'wu_et': wu_et})

    # dry static energy transport

    if 'geopot_p' in md:
        dse = cp * md.tm1 + md.geopot_p
        dse_anom = dse - dse.mean('lon')
        vdse_mt = md_zm.vm1 * dse.mean('lon') * cos_lat
        vdse_et = (md_anom.vm1 * dse_anom).mean('lon') * cos_lat
        tps['vdse_mt'] = vdse_mt
        tps['vdse_et'] = vdse_et


    # eddy kinetic energy

    eke = ((md.vm1**2) + (md.um1**2)) * 0.5
    eke_fft = xr_spatial_fft_analysis(eke)
    save_complex(eke_fft.sel(k=slice(0,18), plev=slice(100,1000)), f'{outpath}/{exp_name}_{year}_eke_fft.nc')
    eke_zm = eke.mean('lon')
    md_zm['eke'] = eke_zm


    md.sel(lat=slice(90,0)).sel(plev=[200,300,500,700,850], method='nearest').to_netcdf(f'{outpath}/{exp_name}_{year}_pl_sel.nc')    
    md_zm.to_netcdf(f'{outpath}/{exp_name}_{year}_zm_pp.nc')
    tps.to_netcdf(f'{outpath}/{exp_name}_{year}_transports_pp.nc')

    print(f'{year}: \t done')
    md.close()
    md_anom.close()
    return

def postprocessing_ml(model_files, year, ml_var_list_sel, r_e=r_e, g=g):

    files_in_year = [mf for mf in model_files if year in mf ]
    files_in_year.sort()
    # model data
    md = read_model_output(files_in_year, var_list_sel=ml_var_list_sel).sortby('time')
    print(f'ml - {year}: model data read in')

    md['p'] = md.hyam + md.hybm * md.aps
    nlevs = md.lev.size

    dp_top = (md.p.isel(lev=0) + md.p.isel(lev=1).values)/2 - md.p.isel(lev=0).values
    dp_mid = ((md.p.isel(lev=slice(1,nlevs-1)) + md.p.isel(lev=slice(2,nlevs)).values)/2 -
              (md.p.isel(lev=slice(0,nlevs-2)).values + md.p.isel(lev=slice(1,nlevs-1)).values)/2)
    dp_bottom = md.aps - (md.p.isel(lev=nlevs-1) + md.p.isel(lev=nlevs-2).values)/2
    dp = xr.concat([dp_top, dp_mid, dp_bottom], dim='lev')

    md['dp'] = dp

    # zonal averages and deviations
    md_zm = md.mean('lon')
    md_anom = md - md_zm

    print(f'ml - {year}: zonal averages and anomalies computed')
    cos_lat = np.cos(np.radians(md_zm.lat))

    # momentum transport
    vu_mt = (md_zm.vm1 * md_zm.um1 * md_zm.dp).sum('lev') * cos_lat * 2 * np.pi * r_e / g
    vu_et = (md_anom.vm1 * md_anom.um1 * dp).sum('lev').mean('lon') * cos_lat * 2 * np.pi * r_e / g
    print(f'ml - {year}: momentum transport calculated')

    # heat transport
    vT_mt = (md_zm.vm1 * md_zm.tm1 * md_zm.dp).sum('lev') * cos_lat * 2 * np.pi * r_e / g
    vT_et = (md_anom.vm1 * md_anom.tm1 * dp).sum('lev').mean('lon') * cos_lat * 2 * np.pi * r_e / g
    print(f'ml - {year}: heat transport calculated')

    wu_mt = (md_zm.vervel * md_zm.um1 * md_zm.dp).sum('lev') * cos_lat * 2 * np.pi * r_e / g
    wu_et = (md_anom.vervel * md_anom.um1 * dp).sum('lev').mean('lon') * cos_lat * 2 * np.pi * r_e / g
    print(f'ml - {year}: vertical transport of zonal momentum calculated')

    # join all transports in one dataset
    tps = xr.Dataset({'vu_mt': vu_mt, 'vu_et': vu_et, 'vT_mt': vT_mt, 'vT_et': vT_et, 'wu_mt': wu_mt, 'wu_et': wu_et})

    # dry static energy transport
    cp = 1004
    g = 9.81
    if 'geopot' in md:
        dse = cp * md.tm1 + md.geopot
        dse_anom = dse - dse.mean('lon')
        vdse_mt = (md_zm.vm1 * dse * dp).sum('lev').mean('lon') * cos_lat * 2 * np.pi * r_e / g
        vdse_et = (md_anom.vm1 * dse_anom * dp).sum('lev').mean('lon') * cos_lat * 2 * np.pi * r_e / g
        tps['vdse_mt'] = vdse_mt
        tps['vdse_et'] = vdse_et
        print(f'ml - {year}: DSE transport calculated')


    # eddy kinetic energy

    eke = ((md.vm1**2) + (md.um1**2)) * 0.5
    eke_fft = xr_spatial_fft_analysis(eke)
    #save_complex(eke_fft.sel(k=slice(0,18), lev=slice(60,90)), f'{outpath}/{exp_name}_{year}_eke_fft.nc')
    eke_zm = eke.mean('lon')
    md_zm['eke'] = eke_zm

    #md.sel(lat=slice(90,0)).sel(lev=[70,74,80,83,85]).to_netcdf(f'{outpath}/{exp_name}_{year}_ml_sel.nc')    
    #md_zm.to_netcdf(f'{outpath}/{exp_name}_{year}_zm_ml_pp.nc')
    tps.to_netcdf(f'{outpath}/{exp_name}_{year}_transports_int_pp.nc')

    print(f'ml - {year}: \t done')
    md.close()
    md_anom.close()
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
ml_output_ending='emil.nc'

inpath=f'/work/bd1022/b381739/{exp_name}'
outpath=f'/work/bd1022/b381739/{exp_name}/postprocessed'


#inpath=f'/mnt/c/Users/pablo/Nextcloud/3_Mastersemester/Masterarbeit/test_files'
#outpath=f'/mnt/c/Users/pablo/Nextcloud/3_Mastersemester/Masterarbeit/test_files/postprocessed'

try: 
    os.makedirs(outpath)
except FileExistsError:
    pass

os.chdir(inpath)
pl_files = [fi for fi in os.listdir(inpath) if fi.endswith(output_ending) and fi.startswith(f'{exp_name}_2')]
ml_files = [fi for fi in os.listdir(inpath) if fi.endswith(ml_output_ending) and fi.startswith(f'{exp_name}_2')]
years = [model_file[15:19] for model_file in pl_files]
years = list(set(years))
years.sort()
print(exp_name)
print(years)

var_list = ['um1', 'vm1', 'vervel', 'tm1', 'aps', 'geopot_p', 'geopot', 'hyam', 'hybm']

pl_var_list_sel = []
ml_var_list_sel = []
pl_file = xr.open_dataset(pl_files[0], engine='netcdf4')
ml_file = xr.open_dataset(ml_files[0], engine='netcdf4')
for var in var_list:
    if var in pl_file:
        pl_var_list_sel.append(var)
    if var in ml_file:
        ml_var_list_sel.append(var)

pl_file.close()
ml_file.close()
print(f'pressure level variables: {pl_var_list_sel}')
print(f'model level variables: {ml_var_list_sel}')

# set NaN in the wind fields to zero to reduce spurius averages of only a few points
fillna_values = {"um1": 0, "vm1": 0, 'vervel': 0}

for year in years:
    #postprocessing_pl(pl_files, year, pl_var_list_sel)
    postprocessing_ml(ml_files, year, ml_var_list_sel)
