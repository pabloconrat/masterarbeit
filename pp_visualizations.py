import numpy as np
import matplotlib.pyplot as plt
import proplot as pplt
import xarray as xr
import os
import sys
import jinja2
import argparse

if '/project/meteo/work/Pablo.Conrat/code/' not in sys.path:
    sys.path.append('/project/meteo/work/Pablo.Conrat/code/')

import pytropd.functions as tropdf
#from own_functions import vertical_weights, weights
from visualization import plot_zm_climatologies, plot_transports, plot_hor_fields, plot_ep_flux_div, plot_EKE_spectral, plot_spectral_vd, plot_hayashi_spectra, plot_wave_persistency, plot_theta_profiles, plot_t_profiles
import dataclasses
from dataclasses import dataclass

import warnings
warnings.simplefilter(action='ignore')

parser = argparse.ArgumentParser()
parser.add_argument('exp_name')
parser.add_argument('--ref_name', action='store' , type=str, default=None)
parser.add_argument('--plot_format', action='store' , type=str, default='png')
args = vars(parser.parse_args())

## Arguments
exp_name = args['exp_name']
ref_name = args['ref_name']
plot_format = args['plot_format']
ylims=[1013,10]
yscale='linear'

## Functions

@dataclass
class PresentationObject:
    title: str = None
    comment: str = None
    plots: list = dataclasses.field(default_factory=list)

def get_template(template_path):
    """get Jinja2 template file"""
    search_path = ['.', 'templates']

    loader = jinja2.FileSystemLoader(search_path)
    environment = jinja2.Environment(loader=loader)
    return environment.get_template(template_path)

class ChangeDirectory:
    """Context manager for changing the current working directory"""
    def __init__(self, new_path):
        self.new_path = os.path.expanduser(new_path)
        self.saved_path = os.getcwd()

    def __enter__(self):
        os.chdir(self.new_path)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved_path)

def read_complex(*args, **kwargs):
    ds = xr.open_dataset(*args, **kwargs)
    return ds['real'] + ds['imag'] * 1j

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

def calc_transport_fourier(va, da):
    vfft = xr_spatial_fft_analysis(va)
    dfft = xr_spatial_fft_analysis(da)
    vd_t = (np.real(dfft)*np.real(vfft) + np.imag(dfft)*np.imag(vfft))

    return vd_t

def compute_harmonics(data, m=None):

    """
    time series  time series to approximate
    m            number of harmonics
    """

    # Set parapeters and set time series to right format
    ts = np.array(data)    
    time = np.arange(1, len(ts)+1, 1)
    n = len(ts)

    if m is None:
        m = int(data.size/2) # uses the fact that int() takes the int closer to zero
    
    # Initialize Coefficient vectors
    A_k = np.zeros(m)
    B_k = np.zeros(m)
    C_k = np.zeros(m)
    data_t = np.ones(len(ts)) * np.mean(ts)
    
    A_k[0] = np.mean(ts)
    # Optional: phase angles
#    phi_k = np.zeros(n)

    if m >= 1:
        for i in np.arange(1, m, 1):
            # Precompute sine and cosine terms
            cos_term = np.cos(2*np.pi*(i)*time/n)
            sin_term = np.sin(2*np.pi*(i)*time/n)

            # Compute Coefficients for Fourier Series
            A_k[i] = 2/n * sum(ts*cos_term)
            B_k[i] = 2/n * sum(ts*sin_term)

            # Optional: Compute phase angles
    #        if A_k[i] > 0:
    #            phi_k[i] = np.arctan(A_k[i]/B_k[i])
    #        elif A_k[i] < 0:
    #            phi_k[i] = np.arctan(A_k[i]/B_k[i]) + np.pi
    #        else:
    #            (A_k[i]/B_k[i]) = np.pi/2

            # Add new term to series
            data_t += (A_k[i] * cos_term + B_k[i] * sin_term)

    # Compute amplitude for information about spectrum
    C_k = np.sqrt(A_k**2 + B_k**2)

    return A_k, B_k, C_k, data_t


def compute_hayashi_spectra(data, k, w):
    """
    data = data(t,lon)
    k = no. of zonal wavenumbers
    w = no. of wavenumbers in time
    
    Steps:
    1) compute time dependent Fourier coefficients C(k,t), S(k,t) from zonal Fourier transformations
    2) compute Fourier coefficients of C and S
    3) compute cospectra and quadrature spectrum to calculate the power spectrum
    """
    # prep
    
    n_t = data.shape[0]
    
    # zonal fourier coefficients 
    c_kt = np.zeros([n_t,k])
    s_kt = np.zeros([n_t,k])
    
    # temporal fourier coefficients for c_kt (cos term)
    a_c = np.zeros([k,w])
    b_c = np.zeros([k,w])
    # temporal fourier coefficients for s_kt (sin term)
    a_s = np.zeros([k,w])
    b_s = np.zeros([k,w])
    
    cosp_c = np.zeros([k,w])
    cosp_s = np.zeros([k,w])
    quadsp = np.ones([k,w])
    
    for i in range(data.shape[0]):
        # 1)
        c_kt[i], s_kt[i] = compute_harmonics(data[i], m=k)[:2]
    
    for j in range(k):
        # 2) 
        a_c[j], b_c[j] = compute_harmonics(c_kt[:,j], m=w)[:2]
        
        a_s[j], b_s[j] = compute_harmonics(s_kt[:,j], m=w)[:2]
        
        # 3)
        cosp_c[j] = 0.5 * (a_c[j] **2 + b_c[j]**2)
        cosp_s[j] = 0.5 * (a_s[j] **2 + b_s[j]**2)
        quadsp[j] = 0.5 * (b_c[j]*a_s[j] - a_c[j]*b_s[j])
        
    hayashi_spectra = np.zeros([k, w*2+1])

    hayashi_spectra[:,:w] = np.flip((cosp_c + cosp_s - 2 * quadsp) * 0.25, axis=1)
    hayashi_spectra[:,w+1:] = (cosp_c + cosp_s + 2 * quadsp) * 0.25
    hayashi_spectra[:,w] = np.nan
    return hayashi_spectra


def calculate_wave_persistency_coeffs(data, resample_freqs_numeral, m):
    """
    m = no of wavenumbers to include in Fourier series
    """
    # compute coefficients for all timesteps
    a_k = np.zeros([len(resample_freqs), m])
    b_k = np.zeros([len(resample_freqs), m])
    c_k = np.zeros([len(resample_freqs), m])

    for i in range(len(resample_freqs)):
        """ resample:
        if i < len(resample_freqs):
            ds = data.resample(time=resample_freqs[i], skipna=True).mean('time')
        elif i == len(resample_freqs):
            ds = data.mean('time')
        """
        # rolling mean:
        if i == 0:
            ds = data
        else:
            ds = data.rolling(time=int(resample_freqs_numeral[i])).mean('time').dropna('time')
        a_kh = np.zeros([ds.time.size, m])
        b_kh = np.zeros([ds.time.size, m])
        c_kh = np.zeros([ds.time.size, m])
        for t in range(ds.time.size):
            ds_sel = ds.isel(time=t)
            a_kh[t], b_kh[t], c_kh[t] = compute_harmonics(ds_sel.values, m=m)[:3]

        a_k[i] = np.mean(a_kh, axis=0)
        b_k[i] = np.mean(b_kh, axis=0)
        c_k[i] = np.mean(c_kh, axis=0)
    
    return a_k, b_k, c_k

## Data loading
indir = f'/project/meteo/work/Pablo.Conrat/Masterarbeit/{exp_name}'
outdir = f'/project/meteo/work/Pablo.Conrat/Masterarbeit/summaries'
tempdir = f'/project/meteo/work/Pablo.Conrat/Masterarbeit/'

os.chdir(indir)

ds_zm = xr.open_dataset(f'{exp_name}_zm_pp.nc').sortby('time')
ds_pls = xr.open_dataset(f'{exp_name}_pl_sel.nc').sortby('time')
ds_tp = xr.open_dataset(f'{exp_name}_transports_pp.nc').sortby('time')
ds_eke = read_complex(f'{exp_name}_eke_fft.nc').sortby('time')
ds_tp_int = xr.open_dataset(f'{exp_name}_transports_int_pp.nc').sortby('time')


if ref_name is not None:
    indir_ref = f'/project/meteo/work/Pablo.Conrat/Masterarbeit/{ref_name}'
    os.chdir(indir_ref)

    ds_zm_ref = xr.open_dataset(f'{ref_name}_zm_pp.nc').sortby('time')
    ds_pls_ref = xr.open_dataset(f'{ref_name}_pl_sel.nc').sortby('time')
    ds_tp_ref = xr.open_dataset(f'{ref_name}_transports_pp.nc').sortby('time')

os.chdir(tempdir)
template_path = 'template.md.j2' # path of template
md_template = get_template(template_path)
presentation_list = []

os.chdir(outdir)


## Vertical and horizontal weights

vert_weights = np.zeros(ds_zm.plev.size)

plevs = ds_zm.plev.values

vert_weights[0] = (plevs[1] + plevs[0])/2 - plevs[0]
vert_weights[1:-1] = (plevs[2:] + plevs[1:-1])/2 - (plevs[1:-1] + plevs[:-2])/2
vert_weights[-1] = 1.013e3 - (plevs[-1] + plevs[-2])/2
vert_weights = vert_weights/np.sum(vert_weights)

weights_da = xr.DataArray(vert_weights, dims=['plev'], coords=dict(plev=(['plev'], ds_zm.plev.data)))

hor_weights = np.cos(np.radians(ds_zm.lat))/np.sum(np.cos(np.radians(ds_zm.lat)))

## Climatologies

ds = ds_zm.mean('time')
psi = tropdf.TropD_Calculate_StreamFunction(ds.vm1.values.T, 
                                            ds.lat.values,
                                            ds.plev.values)

r_air = 287
cp = 1003
ds_zm['psi'] = (['plev', 'lat'], psi.T)
ds_zm['theta'] = ds_zm.tm1 * (1013/ds_zm.plev) ** (r_air/cp)

fig, ax, lat_stj, lat_edj = plot_zm_climatologies(ds_zm, ds_tp)

fig_path = f'{exp_name}_zm_fields.{plot_format}'
fig.save(fig_path)

comment = (
    f'Experiment: {exp_name}. '
    f'Fields are averaged over time and longitude. \n\n'
    f'Latitude of NH STJ: {round(lat_stj[0], 2)}\n\n'
    f'Latitude of NH EDJ: {round(lat_edj[0], 2)}\n\n'
    f'Latitude of SH STJ: {round(lat_stj[1], 2)}\n\n'
    f'Latitude of SH EDJ: {round(lat_edj[1], 2)}\n'
)

presentation_list.append(
    PresentationObject(title=f'zonal mean climatologies', comment=comment, plots=[fig_path])
)


if ref_name is not None:
    
    ds = ds_zm_ref.mean('time')
    psi = tropdf.TropD_Calculate_StreamFunction(ds.vm1.values.T, 
                                                ds.lat.values,
                                                ds.plev.values)
    ds_zm_ref['psi'] = (['plev', 'lat'], psi.T)
    ds_zm_ref['theta'] = ds_zm_ref.tm1 * (1013/ds_zm_ref.plev) ** (r_air/cp)

    fig, ax = plot_zm_climatologies(ds_zm, ds_tp, ds_zm_ref, ds_tp_ref)[:2]

    fig_path = f'{exp_name}-ref_{ref_name}_zm_fields.{plot_format}'
    fig.save(fig_path)

    comment = (
        f"Fields are averaged over time and longitude. Comparison between {exp_name} and {ref_name}."
        f"Line contours show {exp_name}'s climatology. Filled contours show the difference to the reference scenario."
    )

    presentation_list.append(
        PresentationObject(title=f'comparison with reference scenario', comment=comment, plots=[fig_path])
    )

#### Potential temperatures

fig, ax = plot_theta_profiles(ds_zm.sel(plev=slice(10,1000)), exp_name)
fig2, ax2 = plot_t_profiles(ds_zm.sel(plev=slice(10,1000)), exp_name)

fig_path = f'{exp_name}_theta_cmp.{plot_format}'
fig.save(fig_path)
fig_path2 = f'{exp_name}_t_cmp.{plot_format}'
fig2.save(fig_path2)

comment = (
    f"Mean over time and longitude. Map of (potential) temperature and two selected profiles in the tropics and subtropics."
)

presentation_list.append(
    PresentationObject(title=f'stability', comment=comment, plots=[fig_path, fig_path2])
)

## Transports 

fig, ax = plot_transports(ds_tp, weights_da)

fig_path = f'{exp_name}_fluxes.{plot_format}'
fig.save(fig_path)

comment = (
    f"Zonally averaged, vertically integrated (pressure weighting) transports of momentum, heat, and dry static energy."
    f"All plots show mean and eddy transports as well as their sum."
)

presentation_list.append(
    PresentationObject(title=f'transports', comment=comment, plots=[fig_path])
)


## Horizontal fields

#plot_hor_fields(ds_pls, 'vm1',anom=True)
#plot_hor_fields(ds_pls, 'tm1',anom=True)
fig, ax = plot_hor_fields(ds_pls, 'geopot_p')
plot_hor_fields(ds_pls, 'um1',anom=True, fig=fig, ax=ax, savefig=False)

fig_path = f'{exp_name}_hor_fields.{plot_format}'
fig.save(fig_path)

comment = (
    f"Selected horizontal fields."
)

presentation_list.append(
    PresentationObject(title=f'Horizontal fields', comment=comment, plots=[fig_path])
)

## EP-Flux Divergence

uv = (ds_tp.vu_et + ds_tp.vu_mt).sel(plev=slice(1,900), lat=slice(85,0)).mean('time')
uw = (ds_tp.wu_et + ds_tp.wu_mt).sel(plev=slice(1,900), lat=slice(85,0)).mean('time')
vt = (ds_tp.vT_et + ds_tp.vT_mt).sel(plev=slice(1,900), lat=slice(85,0)).mean('time')
u = ds_zm.um1.sel(plev=slice(1,900), lat=slice(85,0)).mean('time')
t = ds_zm.tm1.sel(plev=slice(1,900), lat=slice(85,0)).mean('time')

fig, ax = plot_ep_flux_div(uv, uw, vt, u, t)

fig_path = f'{exp_name}_ep_flux_divergence.{plot_format}'
fig.save(fig_path)

comment = (
    f"EP-Flux divergence (Frederiks code, yet to be adapted)."
)

presentation_list.append(
    PresentationObject(title=f'EP flux divergence', comment=comment, plots=[fig_path])
)


## EKE Fourier Analysis 


fig, ax = plot_EKE_spectral(ds_eke.sel(lat=slice(90,0)).mean('lat'), 'NH')   
fig2, ax2 = plot_EKE_spectral(ds_eke.sel(lat=slice(0,-90)).mean('lat'), 'SH')   

fig_path = f'{exp_name}_eke_hem_avg_nh.{plot_format}'
fig.save(fig_path)
fig_path2 = f'{exp_name}_eke_hem_avg_sh.{plot_format}'
fig2.save(fig_path2)

comment = (
    f"EKE averaged over the hemispheres, weighted by the cosine of latitude. EKE(p,k) plotted."
)

presentation_list.append(
    PresentationObject(title=f'EKE', comment=comment, plots=[fig_path, fig_path2])
)

## vT & vu Fourier Analysis

ds_va = ds_pls.vm1 - ds_pls.vm1.mean(['lon'])
ds_ua = ds_pls.um1 - ds_pls.um1.mean(['lon'])
ds_ta = ds_pls.tm1 - ds_pls.tm1.mean(['lon'])

vT_t = calc_transport_fourier(ds_va.mean('time'), ds_ta.mean('time'))
vu_t = calc_transport_fourier(ds_va.mean('time'), ds_ua.mean('time'))

lat_sel = 45
plev_sel = 700
fig, ax = plot_spectral_vd(vT_t.sel(lat=lat_sel, plev=plev_sel, method='nearest'), title=f'Heat transport at {int(lat_sel)}°N, {int(plev_sel)} hPa')
fig_path = f'{exp_name}_heat_transport_tavg_{lat_sel}_{plev_sel}.{plot_format}'
fig.save(fig_path)

lat_sel = 20
plev_sel = 200
fig, ax = plot_spectral_vd(vu_t.sel(lat=lat_sel, plev=plev_sel, method='nearest'), title=f'Momentum transport at {int(lat_sel)}°N, {int(plev_sel)} hPa')
fig_path2 = f'{exp_name}_momentum_transport_tavg_{lat_sel}_{plev_sel}.{plot_format}'
fig.save(fig_path2)

comment = (
    f"Heat transport per wavenumber at specific latitude and height levels, zonally averaged."
)

presentation_list.append(
    PresentationObject(title=f'Transports (spectral)', comment=comment, plots=[fig_path, fig_path2])
)

## Hayashi Spectra

lat_slice = slice(50,20)
ds_v = ds_pls.vm1.sel(plev=ds_pls.plev[0], lat=lat_slice).weighted(weights=hor_weights).mean(['lat'])
ds_va = ds_v - ds_v.mean(['lon'])

k = 30
w = 140

ds_va_samples = ds_va.resample(time='180D', closed='left', label='left')
#ds_va_samples = ds_va_samples.where(ds_va_samples.count() >= 180)
n_samples = ds_va_samples.count().time.size
hayashi_spectra = np.zeros([n_samples, k, w*2+1])
i = 0
for t,ds in ds_va_samples:
    if ds.time.size != 180:
        continue
    hayashi_spectra[i] = compute_hayashi_spectra(ds.values, k, w)
    i += 1

ks = np.arange(k)
ws = np.arange(-w, w+1, 1)
freqs = (ws * 2*np.pi) / 180 # 1/d

da_hs = xr.DataArray(hayashi_spectra,
                     dims=['time', 'k', 'f'],
                     coords=dict(k = (['k'], ks),
                                 f = (['f'], freqs),
                                 time = (['time'], ds_va_samples.count().time.data))
                    )
da_hs = da_hs.mean('time').rolling({'k':3}).mean()

fig, ax = plot_hayashi_spectra(da_hs)
fig_path = f'{exp_name}_hayashi_spectra_v_extratropics.{plot_format}'
fig.save(fig_path)

comment = (
    f"Hayashi spectra over the extratropical (20°N to 50°N, weighted by cos(lat)) meridional wind anomalies."
    f"Frequencies are given in 1/d and positive frequencies represent westward moving waves."
    f"Careful: results seem a bit weird!"
)

presentation_list.append(
    PresentationObject(title=f'EKE', comment=comment, plots=[fig_path])
)

## Persistency of Wavenumbers

ds = ds_va.resample(time='QS-DEC').mean('time')
resample_freqs = ['1D','3D', '1W', '2W', '4W', '8W', '16W', '32W', '1Y', '2Y', '3Y']
resample_freqs_numeral = np.array([1/7, 3/7, 1, 2, 4, 8, 16, 32, 52, 104, 156]) * 7

kmax = 15

a_k, b_k, c_k = calculate_wave_persistency_coeffs(ds_va, resample_freqs_numeral, kmax)
fig, ax = plot_wave_persistency(ds, c_k, resample_freqs_numeral)
fig_path = f'{exp_name}_wave_persistence_abs.{plot_format}'
fig.savefig(fig_path)

fig, ax = plot_wave_persistency(ds, c_k, resample_freqs_numeral, relative=True)
fig_path2 = f'{exp_name}_wave_persistence_rel.{plot_format}'
fig.savefig(fig_path2)


comment = (
    f"Amplitude of Fourier coefficients for different resample frequencies"
)

presentation_list.append(
    PresentationObject(title=f'heat transport (spectral)', comment=comment, plots=[fig_path, fig_path2])
)

plt.close('all')

###############################
### Writing out to .md file ###
###############################

with ChangeDirectory(outdir):
    with open(f"./{exp_name}_summary.md", 'w') as md_out:
        md_out.write(md_template.render(
            presentation_list=presentation_list,
        ))