#!bin/python3
#SBATCH -J EMIL_pp             # Specify job name
#SBATCH -p shared              # Use partition prepost
#SBATCH --mem=22000
#SBATCH -t 01:00:00            # Set a limit on the total run time
#SBATCH -A bd1022              # Charge resources on this project account
#SBATCH -o EMIL_pp_%j.out      # File name for standard and error output

from cmath import cos
import os
from re import U
import aostools.climate
import numpy as np
import xarray as xr 
import argparse
from pathlib import Path
from own_functions import calc_da_derivative, interp_p_to_theta, omega_earth, r_e, cp, g0, r_air, p00


def xr_spatial_fft_analysis(data):
    
    dx = data.lon.values[1] - data.lon.values[0]
    dim='k'
    result =  xr.apply_ufunc(
        np.fft.fft, data, input_core_dims=[['lon']], output_core_dims=[[dim]],
        kwargs={'norm':'ortho'},
        dask='parallelized',
        dask_gufunc_kwargs={'output_sizes':{'k':data.lon.size}}
    )
    k = np.fft.fftfreq(result.k.shape[0], d=dx/(360))
    result[dim] = k
    return result

def save_complex(data_array, *args, **kwargs):
    ds = xr.Dataset({'real': data_array.real, 'imag': data_array.imag})
    return ds.to_netcdf(*args, **kwargs)

def read_model_output_old(infiles, var_list_sel=None, fillna=False):
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

def read_model_output(infiles, var_list_sel=None, fillna=False):
    if var_list_sel is None:
        ds = xr.open_mfdataset(infiles, combine='by_coords')
    else:
        ds = xr.open_mfdataset(infiles, combine='by_coords')[var_list_sel]
    return ds

def merge_pp_files(path, types):
    for tp in types:
        files = [fi for fi in os.listdir(path) if fi.endswith(f'_{tp}.nc')]
        if files:
            ds = read_model_output(files)
            ds.to_netcdf(f'{exp_name}_{tp}.nc')
            [os.remove(file) for file in files]
    return

def postprocessing_pl(model_files, year, pl_var_list_sel):

    files_in_year = [mf for mf in model_files if year in mf ]
    files_in_year.sort()
    # model data
    md = read_model_output(files_in_year, var_list_sel=pl_var_list_sel, fillna=True).sortby('time').load()
    print(f'{year}: model data read in')
    nplevs = md.plev.size
    dp_top = (md.plev.isel(plev=0) + md.plev.isel(plev=1).values)/2 - md.plev.isel(plev=0).values
    dp_mid = ((md.plev.isel(plev=slice(1,nplevs-1)) + md.plev.isel(plev=slice(2,nplevs)).values)/2 -
              (md.plev.isel(plev=slice(0,nplevs-2)).values + md.plev.isel(plev=slice(1,nplevs-1)).values)/2)
    dp_bottom = md.aps - (md.plev.isel(plev=nplevs-1) + md.plev.isel(plev=nplevs-2).values)/2
    dp = xr.concat([dp_top, dp_mid, dp_bottom], dim='plev')

    if eof_analysis_wanted:
        # deseasonalize, detrend, crop, area-weight
        # Geopotential option
        #x = md.sel(lat=slice(90,20)).sel(plev=500, method='nearest').geopot_p 
        # Vertically integrated wind option
        u_int = (md.um1 * dp).sel(plev=slice(50,850)).mean(['plev','lon'])
        x = u_int
        x = x - x.mean('time')
        x = x * np.sqrt(np.cos(np.deg2rad(x.lat)))
        x.to_dataset(name='x').to_netcdf(f'{outpath}/{exp_name}_{year}_x_temp.nc')

    print(f'{year}: zonal averages and anomalies computed')
    lat_rad = np.radians(md.lat)
    lon_rad = np.radians(md.lon)
    cos_lat = np.cos(lat_rad)
    f = 2*omega_earth*np.sin(lat_rad)

    # PV
    md['theta'] = md.tm1 * (p00/md.plev) ** (r_air/cp)
    cos_lat = np.cos(np.radians(md.lat))
    # calculate gradient of pot. temperature w.r.t. pressure
    dtheta_dp = md.theta.differentiate('plev')/100
    # calculate the absolute vorticity in vertical (pressure) direction
    zeta_a = ((calc_da_derivative(md.vm1, lon_rad, coord_name='lon') - calc_da_derivative(md.um1 * cos_lat, lat_rad, coord_name='lat'))
              /(r_e * cos_lat) + f)
    # calculate PV twisting term
    pv_twist = (md.um1.differentiate('plev')/100 * calc_da_derivative(md.theta, lat_rad, coord_name='lat')
                - md.vm1.differentiate('plev')/100 * calc_da_derivative(md.theta, lon_rad, coord_name='lon')/cos_lat)
    # rescale to PVU
    md['pv'] = -1e6 * g0 * (zeta_a * dtheta_dp + pv_twist/r_e)
    print(f'{year}: PV calculated')

    # zonal averages and deviations
    md_zm = md.mean('lon')
    md_anom = md - md_zm

    # momentum transport
    vu_mt = md_zm.vm1 * md_zm.um1 * cos_lat
    vu_et = (md_anom.vm1 * md_anom.um1 ).mean('lon') * cos_lat
    print(f'{year}: momentum transport calculated')

    wu_mt = md_zm.vervel * md_zm.um1 #* cos_lat
    wu_et = (md_anom.vervel * md_anom.um1).mean('lon') #* cos_lat
    print(f'{year}: vertical transport of zonal momentum calculated')

    # heat transport
    vT_mt = md_zm.vm1 * md_zm.tm1 * cos_lat
    vT_et = (md_anom.vm1 * md_anom.tm1).mean('lon') * cos_lat
    print(f'{year}: heat transport calculated')

    # PV transport
    vq_mt = md_zm.vm1 * md_zm.pv * cos_lat
    vq_et = (md_anom.vm1 * md_anom.pv).mean('lon') * cos_lat

    # Interpolate heat transport to 700 hPa 
    vT_mt_700 = vT_mt.interp(plev=700)
    vT_et_700 = vT_et.interp(plev=700)
 
    tps = xr.Dataset({'vu_mt': vu_mt, 'vu_et': vu_et, 'vT_mt': vT_mt, 'vT_et': vT_et, 'wu_mt': wu_mt, 'wu_et': wu_et, 'vq_mt': vq_mt, 'vq_et': vq_et})
    tps_ml = xr.Dataset({'vT_mt': vT_mt_700, 'vT_et': vT_et_700})

    # dry static energy transport

    if 'geopot_p' in md:
        dse = cp * md.tm1 + md.geopot_p
        dse_anom = dse - dse.mean('lon')
        vdse_mt = md_zm.vm1 * dse.mean('lon') * cos_lat
        vdse_et = (md_anom.vm1 * dse_anom).mean('lon') * cos_lat
        tps['vdse_mt'] = vdse_mt
        tps['vdse_et'] = vdse_et
        print(f'{year}: DSE transport calculated')


    # eddy kinetic energy
    eke = ((md.vm1**2) + (md.um1**2)) * 0.5
    eke_fft = xr_spatial_fft_analysis(eke)
    save_complex(eke_fft.sel(k=slice(0,18), plev=slice(100,1000)), f'{outpath}/{exp_name}_{year}_eke_fft.nc')
    eke_zm = eke.mean('lon')
    md_zm['eke'] = eke_zm
    print(f'{year}: EKE calculated')

    # EP flux computation
    md = md.reindex({'lat':np.flipud(md.lat)})
    ep_cart1, ep_cart2, div1, div2 = aostools.climate.ComputeEPfluxDivXr(
        md.um1, md.vm1, md.tm1, pres='plev', w=md.vervel/100, do_ubar=True
    )
    ep = ep_cart1.to_dataset(name='ep_cart1')
    ep['ep_cart2'] = ep_cart2
    ep['div1'] = div1
    ep['div2'] = div2
    md = md.reindex({'lat':np.flipud(md.lat)})
    ep = ep.reindex({'lat':np.flipud(ep.lat)})
    #ep = ep.resample(time='1Y').mean('time')
    print(f'{year}: EP fluxes calculated')

    # calculate v_star and vervel_star for the TEM formulated equation
    vtheta_et = vT_et * (p00/md_zm.plev) ** (r_air/cp)
    dtheta_dp = md_zm.theta.differentiate('plev') / 100
    d_vtheta_et_dp = (vtheta_et/dtheta_dp).differentiate('plev') / 100 # rescale to Pa
    d_vtheta_et_dphi = 1/(r_e * cos_lat) * ((vtheta_et * cos_lat)/dtheta_dp).differentiate('lat') * 360 / (2*np.pi) # rescale to radians

    v_star = md_zm.vm1 - d_vtheta_et_dp
    vervel_star = md_zm.vervel + d_vtheta_et_dphi

    # Zonal Momentum Equation
    u_zm_eq = md_zm.um1.differentiate('time', datetime_unit="s").to_dataset(name='dudt')
    u_zm_eq['u_adv_phi'] = md_zm.vm1/(r_e * cos_lat) * calc_da_derivative(md_zm.um1 * cos_lat, lat_rad, coord_name='lat') 
    u_zm_eq['u_adv_p'] = md_zm.vervel * calc_da_derivative(md_zm.um1, md_zm.plev * 100)
    u_zm_eq['f_v'] = 2 * omega_earth * np.sin(lat_rad) * md_zm.vm1 
    u_zm_eq['vu_et'] = 1/(r_e * cos_lat**2) * calc_da_derivative(vu_et * cos_lat, lat_rad, coord_name='lat') # vu_et has one cos_lat in it
    u_zm_eq['wu_et'] = calc_da_derivative((md_anom.vervel * md_anom.um1).mean('lon'), md_zm.plev * 100)
    u_zm_eq['residual'] = u_zm_eq.dudt + u_zm_eq.u_adv_phi + u_zm_eq.u_adv_p - u_zm_eq.f_v + u_zm_eq.vu_et + u_zm_eq.wu_et
    print(f'{year}: zonal momentum equation calculated')

    # TEM Zonal Momentum Equation
    u_zm_tem = md_zm.um1.differentiate('time', datetime_unit="s").to_dataset(name='dudt')
    u_zm_tem['v_star'] = v_star
    u_zm_tem['vervel_star'] = vervel_star
    u_zm_tem['wu_mt'] = vervel_star * calc_da_derivative(md_zm.um1, md_zm.plev * 100)
    u_zm_tem['vo_mt'] = v_star  * (1/(r_e * cos_lat) * calc_da_derivative(md_zm.um1 * cos_lat, lat_rad, coord_name='lat') - f)
    u_zm_tem['dphi_vu_et'] = - 1/(r_e * cos_lat**2) * calc_da_derivative(vu_et * cos_lat**2, lat_rad, coord_name='lat')
    u_zm_tem['u_adv_p_rev'] = calc_da_derivative(md_zm.um1, md_zm.plev*100) * d_vtheta_et_dphi # 1/cos_lat in d_vtheta...
    #         ^ alternative: dphi_vpthp-dp_u
    #u_zm_tem['tan_phi_vu_et'] = np.tan(np.radians(md_zm.lat)) * vu_et/cos_lat / r_e
    u_zm_tem['vo_et'] = d_vtheta_et_dp * (f - 1/(r_e * cos_lat) * calc_da_derivative(md_zm.um1 * cos_lat, lat_rad, coord_name='lat'))
    u_zm_tem['dp_wu_et'] = - calc_da_derivative(wu_et, md_zm.plev*100)
    u_zm_tem['residual'] = (
        u_zm_tem.dudt + u_zm_tem.wu_mt + u_zm_tem.vo_mt 
        - u_zm_tem.dphi_vu_et - u_zm_tem.u_adv_p_rev
        - u_zm_tem.vo_et - u_zm_tem.dp_wu_et
    )
    
    # interpolate to isentropes
    md_isent_zm = interp_p_to_theta(md_zm[['um1', 'vm1', 'tm1', 'pv', 'theta']])
    md_isent_zm['aps'] = md_zm.aps
    md_isent_sel = interp_p_to_theta(md[['um1', 'vm1', 'tm1', 'pv', 'theta']], theta_levels=theta_levels_pl_sel)
    md_isent_sel['aps'] = md.aps

    md.sel(plev=[200,300,500,700,850], method='nearest').to_netcdf(f'{outpath}/{exp_name}_{year}_pl_sel.nc')    
    md_zm.to_netcdf(f'{outpath}/{exp_name}_{year}_zm_pp.nc')
    md_isent_zm.to_netcdf(f'{outpath}/{exp_name}_{year}_zm_isent_pp.nc')
    md_isent_sel.to_netcdf(f'{outpath}/{exp_name}_{year}_il_sel_pp.nc')
    tps.to_netcdf(f'{outpath}/{exp_name}_{year}_transports_pp.nc')
    tps_ml.to_netcdf(f'{outpath}/{exp_name}_{year}_transports_int_pp.nc')
    ep.to_netcdf(f'{outpath}/{exp_name}_{year}_ep_pp.nc')
    u_zm_eq.to_netcdf(f'{outpath}/{exp_name}_{year}_zmom_eq_pp.nc')
    u_zm_tem.to_netcdf(f'{outpath}/{exp_name}_{year}_zmom_tem_eq_pp.nc')

    print(f'{year}: \t done')
    md.close()
    md_anom.close()
    return

def postprocessing_ml(model_files, year, ml_var_list_sel, r_e=r_e, g=g0):

    files_in_year = [mf for mf in model_files if year in mf ]
    files_in_year.sort()
    # model data
    md = read_model_output(files_in_year, var_list_sel=ml_var_list_sel, fillna=True).sortby('time')
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
    vu_mt = (md_zm.vm1 * md_zm.um1 * md_zm.dp).mean('lev') * cos_lat
    vu_et = (md_anom.vm1 * md_anom.um1 * dp).mean(['lon', 'lev']) * cos_lat 
    print(f'ml - {year}: momentum transport calculated')

    # vertical momentum transport
    wu_mt = (md_zm.vervel * md_zm.um1 * md_zm.dp).mean('lev') * cos_lat 
    wu_et = (md_anom.vervel * md_anom.um1 * dp).mean(['lon', 'lev']) * cos_lat 
    print(f'ml - {year}: vertical transport of zonal momentum calculated')

    # join all transports in one dataset
    tps = xr.Dataset({'vu_mt': vu_mt, 'vu_et': vu_et, 'wu_mt': wu_mt, 'wu_et': wu_et})

    # dry static energy transport
    if 'geopot' in md:
        dse = cp * md.tm1 + md.geopot
        dse_anom = dse - dse.mean('lon')
        vdse_mt = (md_zm.vm1 * dse * dp).sum('lev').mean('lon') * cos_lat * 2 * np.pi * r_e / g0
        vdse_et = (md_anom.vm1 * dse_anom * dp).sum('lev').mean('lon') * cos_lat * 2 * np.pi * r_e / g0
        tps['vdse_mt'] = vdse_mt
        tps['vdse_et'] = vdse_et
        print(f'ml - {year}: DSE transport calculated')

    # eddy kinetic energy
    #eke = ((md.vm1**2) + (md.um1**2)) * 0.5
    #eke_fft = xr_spatial_fft_analysis(eke)
    #save_complex(eke_fft.sel(k=slice(0,18), lev=slice(60,90)), f'{outpath}/{exp_name}_{year}_eke_fft.nc')
    #eke_zm = eke.mean('lon')
    #md_zm['eke'] = eke_zm

    #md.sel(lat=slice(90,0)).sel(lev=[70,74,80,83,85]).to_netcdf(f'{outpath}/{exp_name}_{year}_ml_sel.nc')    
    #md_zm.to_netcdf(f'{outpath}/{exp_name}_{year}_zm_ml_pp.nc')
    tps.to_netcdf(f'{outpath}/{exp_name}_{year}_transports_int_pp.nc', mode='a')

    print(f'ml - {year}: \t done')
    md.close()
    md_anom.close()
    return

parser = argparse.ArgumentParser()
parser.add_argument('exp_name')
parser.add_argument('--ml', action='store_true', default=False)
parser.add_argument('--eofs', action='store_true', default=False)
parser.add_argument('--years', nargs='*', default=None)
args = vars(parser.parse_args())

exp_name=args['exp_name']
eof_analysis_wanted=args['eofs']
ml=args['ml']
output_ending = 'vaxtra.nc'
ml_output_ending='emil.nc'

inpath=f'/work/bd1022/b381739/{exp_name}'
outpath=f'/work/bd1022/b381739/{exp_name}/postprocessed'

print(inpath)
print(f'... Model level analysis wanted? {ml}')
if args['years'] is not None:
    years=args['years']
else:
    years = list(set([fi[15:19] for fi in os.listdir(inpath) if fi.endswith(output_ending)])).sort()
    print(f'...years: {years}')


#inpath=f'/mnt/c/Users/pablo/Nextcloud/3_Mastersemester/Masterarbeit/test_files'
#outpath=f'/mnt/c/Users/pablo/Nextcloud/3_Mastersemester/Masterarbeit/test_files/postprocessed'

try: 
    os.makedirs(outpath)
except FileExistsError:
    [f.unlink() for f in Path(outpath).glob("*") if f.is_file()] 

separator = ( 14  - len(exp_name) + 1) * '_'
filecheck_list = [exp_name + separator + year for year in years]
filecheck_tuple = tuple(filecheck_list)

os.chdir(inpath)
pl_files = [fi for fi in os.listdir(inpath) if fi.endswith(output_ending) and fi.startswith(filecheck_tuple)]
ml_files = [fi for fi in os.listdir(inpath) if fi.endswith(ml_output_ending) and fi.startswith(filecheck_tuple)]

print(f'... experiment name: {exp_name}')
print(f'... years to analyze: {years}')

var_list = ['um1', 'vm1', 'vervel', 'tm1', 'aps', 'geopot_p', 'geopot', 'hyam', 'hybm']
theta_levels_pl_sel = np.array([275.0, 285.0, 300.0, 315.0, 330.0, 350.0])

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
print(f'... pressure level variables: {pl_var_list_sel}')
print(f'... model level variables: {ml_var_list_sel}')

# set NaN in the wind fields to zero to reduce spurius averages of only a few points
fillna_values = {"um1": 0, "vm1": 0, 'vervel': 0}

for year in years:
    postprocessing_pl(pl_files, year, pl_var_list_sel)
    if ml:
        postprocessing_ml(ml_files, year, ml_var_list_sel)

os.chdir(outpath)

if eof_analysis_wanted:

    files = [fi for fi in os.listdir(os.getcwd()) if fi.endswith('x_temp.nc')]
    ds = read_model_output(files).load()
    from eofs.xarray import Eof

    solver = Eof(ds.x)
    eofs = solver.eofs().to_dataset(name='eofs')
    pcs = solver.pcs(pcscaling=1)
    var_explained = solver.varianceFraction()
    eofs['var_exp'] = var_explained
    eofs.to_netcdf(f'{outpath}/{exp_name}_eofs.nc')
    pcs.to_netcdf(f'{outpath}/{exp_name}_pcs.nc')

    for file in files:
        os.remove(file)

print('...Merging output')
endings = ['transports_pp', 'pl_sel', 'zm_pp', 'eke_fft', 'transports_int_pp', 'ep_pp', 'zmom_eq_pp', 'zmom_tem_eq_pp', 'zm_isent_pp', 'il_sel_pp']
merge_pp_files(outpath, endings)
print('===|Done|===')