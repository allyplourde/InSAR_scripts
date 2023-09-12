import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sarlab.met import parse_ec_dir#, get_param
from scipy.constants import c
from scipy.ndimage import uniform_filter, generic_filter
from scipy.stats import skew
import scipy.ndimage.morphology as morph
from scipy.interpolate import griddata
import os
import glob
import re
from scipy import special
import scipy.optimize as opt
from datetime import datetime, timedelta
from osgeo import gdal
from adjustText import adjust_text
import pickle

from sarlab import multiprocess_gx as mp
from met_utils import getECMetData, dailyERAatSite
from sarlab.gammax import readBin, MLI_Par, writeBin, read_ras, DEM_Par, run, SLC_stack, SLC_Par
from sarlab.gammax.utils import exec as _exec
from cr_phase_to_deformation import get_itab_diffs, deformation_to_phase, phase_to_deformation, unpack_cr_coords, extract_annulus
from u76_process import mk_water_mask
from clicky import launch_clicky

### UP NEXT,
# weight coherance
## scatter
#estimates over whole scene, average -> examine how bias changes with mask
#does surfaces compare to era5?
#university of colorada wrapper
#useful for excluding vegetation
#    vegetation leads to unrealistic snow holding height
#if inc_mask is good
#    use model to make swe variations \
#
#peak to peak with and without deformation model

## CLICK -> QUALITY MASK -> MIDDLE

# global constants
CM_TO_MM = 10; MM_TO_CM = 0.1; CM_TO_M = 0.01; MM_TO_M = 0.001; M_TO_CM = 100; M_TO_MM = 1000

# Met. Directories
inuvik_met_dir = '/local-scratch/users/aplourde/met_data/env_canada/Inuvik/'
tv_met_dir = '/local-scratch/users/aplourde/met_data/env_canada/TrailValley/'
era5_met_dir = '/local-scratch/users/aplourde/met_data/era5/'

# Logger Data Directory
data_dir = '/local-scratch/users/aplourde/field_data/'
snow_files = glob.glob(data_dir + '/*/*snow_depth_processed.csv')
tilt_files = glob.glob(data_dir + '/*/*inclinometer_processed.csv')

# RS2 STACK
RS2_working_dir = '/local-scratch/users/aplourde/HDS/projects/southern_ITH/RS2_U76_D/'
#RS2_sub_dir = 'crop_sites/'; RS2_master = '20180827'
RS2_sub_dir = 'full_scene/'; RS2_master = '20180827'
ingest_cfg = {'password': 'S1m0nF7@SER', 'polarizations': 'HH'}                                                     #2
RS2_stack = SLC_stack(dirname=RS2_working_dir + RS2_sub_dir, name='inuvik_postcr', master=RS2_master, looks_hr=(2,3), looks_lr=(12,18), multiprocess=True, rdcdem_refine=True, skipmode='None', ingest_cfg=ingest_cfg)

#TSX STACK
TSX_working_dir = '/local-scratch/users/aplourde/HDS/projects/southern_ITH/TSX_SM39_D/'
TSX_sub_dir = 'full_scene_crop/'; TSX_master = '20210903'
TSX_stack = SLC_stack(dirname=TSX_working_dir + TSX_sub_dir, name='tsx_southernITH', master=TSX_master, looks_hr=(2, 2), looks_lr=(7, 6), multiprocess=False, rdcdem_refine=True, skipmode='None', ingest_cfg=ingest_cfg)

# SLA STACK
SLA_working_dir = '/local-scratch/users/jaysone/projects_active/inuvik/RS2_SLA27_D/'
SLA_sub_dir = 'full_scene/'; SLA_master = '20160811'
SLA_stack = SLC_stack(dirname=SLA_working_dir + SLA_sub_dir, name='sla_inuvik', master=SLA_master, looks_hr=(3, 12), looks_lr=(11, 44), multiprocess=False, rdcdem_refine=True, skipmode='None', ingest_cfg=ingest_cfg)

# config
SLA_win = 1
SLA_look = (3,12)
SLA_look_str = str(SLA_look[0]) + '_' + str(SLA_look[1])
SLA_cfg = {'stack': SLA_stack, 'looks':SLA_look, 'looks_str':SLA_look_str,
       'swe_range': (-0.1, 0.1), 'swe_del':0.002,
       #'swe_est_dir': SLA_stack._dir +'swe_est_' + SLA_look_str + '/', 'rho': 0.3,
       'swe_est_dir': TSX_stack._dir +'sla_swe_est_' + SLA_look_str + '/gdiff/not_inc_mask_full/', 'rho': 0.3,
       'era5_dir': era5_met_dir + 'delta_snow_depth/RS2_SLA27_D/' + SLA_sub_dir + 'swe_' + SLA_look_str + '/',
       #'swe_est_dir': TSX_stack._dir +'sla_swe_est_' + SLA_look_str + '/gdiff/', 'rho': 0.3,
       'win': np.asarray((188, 204)), 'extent': (0, 1000, 0, 1000),
       'frequency': 5.4049992e+09, 'incidence_angle': np.deg2rad(49.0569),
       'plt_range':(0.22, 0.28),
       'working_dir': SLA_working_dir, 'sub_dir': SLA_sub_dir, 'master':SLA_master,
       #'sample_ifg': '20120317_20120410.diff',
       'sample_ifg': '20200204_20200228.diff',
       'itab': SLA_working_dir + SLA_sub_dir + 'itab_snow_lf'
       }
RS2_win = 2
#RS2_look = (12, 18); RS2_win1000 = (28, 27)
RS2_look = (2,3); RS2_win1000 = (169, 161)
#RS2_look = (1,1); RS2_win1000 = (338, 482)
RS2_look_str = str(RS2_look[0]) + '_' + str(RS2_look[1])
RS2_cfg = {'stack': RS2_stack, 'looks':RS2_look, 'looks_str':RS2_look_str,
       'swe_range': (-0.1, 0.1), 'swe_del':0.002,
       'swe_est_dir': RS2_stack._dir +'swe_est_' + RS2_look_str + '/tdxgauss3/win'+str(RS2_win)+'/water_mask_full/', 'rho': 0.3,
       'era5_dir': era5_met_dir + 'delta_snow_depth/RS2_U76_D/' + RS2_sub_dir + 'swe_' + RS2_look_str + '/',
       'win': np.asarray((RS2_win1000[0]*RS2_win, RS2_win1000[1]*RS2_win)), 'extent': (0, 1000*RS2_win, 0, 1000*RS2_win),
       'frequency': 5.4049992e+09, 'incidence_angle': np.deg2rad(26.7438),
       'plt_range':(0.16, 0.23),
       'working_dir': RS2_working_dir, 'sub_dir': RS2_sub_dir, 'master':RS2_master,
       #'sample_ifg': '20200206_20200301.diff',
       'sample_ifg': '20141128_20141222.diff.adf',
       'itab': RS2_working_dir + RS2_sub_dir + 'itab_lf_winter'
       }
TSX_win = 1
#TSX_look = (1, 1); TSX_win1000 = (455, 402)
TSX_look = (2, 2); TSX_win1000 = (228, 201)
#TSX_look = (7,6); TSX_win1000 = (65, 67)
TSX_look_str = str(TSX_look[0]) + '_' + str(TSX_look[1])
TSX_cfg = {'stack': TSX_stack, 'looks':TSX_look, 'looks_str':TSX_look_str,
       'swe_range':(-0.1, 0.1), 'swe_del':0.002,
       'swe_est_dir': TSX_stack._dir+'swe_est_' + TSX_look_str + '/win'+str(TSX_win)+'/water_mask_full/', 'rho': 0.3,
       'era5_dir': era5_met_dir + 'delta_snow_depth/TSX_SM39_D/' + TSX_sub_dir + 'swe_' + TSX_look_str + '/',
       'win': np.asarray((TSX_win1000[0]*TSX_win, TSX_win1000[1]*TSX_win)), 'extent': (0, 1000*TSX_win, 0, 1000*TSX_win),
       'frequency': 9.6499984e+09, 'incidence_angle': np.deg2rad(24.1774),
       'plt_range':(0.32, 0.38),
       'working_dir': TSX_working_dir, 'sub_dir': TSX_sub_dir, 'master':TSX_master,
       #'sample_ifg': '20191216_HH_20191227_HH.diff',
       'sample_ifg': '20180213_HH_20180224_HH.diff.adf',
       'itab': TSX_working_dir + TSX_sub_dir + 'itab_lf_snow_mq'
       }

cfg = TSX_cfg

# Constants
snow_density = cfg['rho']
incidence_angle = cfg['incidence_angle']
frequency = cfg['frequency']
wavelength = c / frequency
annulus = 250

fig_width_full = 7.16
extent = cfg['extent']
vmin = cfg['plt_range'][0]
vmax = cfg['plt_range'][1]
win1000 = cfg['win']



# SAR Directory
sar_dir = cfg['working_dir']
sub_dir = cfg['sub_dir']
dem_dir = sar_dir + sub_dir + 'dem_' + cfg['looks_str'] +'/'
ifg_dir = sar_dir + sub_dir + 'diff_' + cfg['looks_str'] +'/'
#ifg_dir = sar_dir + sub_dir + 'diff_hr/'

master_par = MLI_Par(dem_dir + '../rmli_' + cfg['looks_str'] + '/rmli_' + cfg['looks_str'] + '.ave.par')
water_mask = read_ras(sar_dir + sub_dir + 'rmli_' + cfg['looks_str'] + '/water_mask_' + cfg['looks_str'] + '.ras')[0].T == 0
#no_motion_mask = read_ras(sar_dir + sub_dir + 'no_motion_mask_' + cfg['looks_str'] + '.ras')[0].T != 0
#exclude_mask = read_ras(dem_dir + 'inc_mask.ras')[0].T == 0
#exclude_mask = read_ras('/local-scratch/users/aplourde/SLA_masks/inc_mask.ras')[0].T == 0
#cfg['exclude_mask'] = exclude_mask
cfg['exclude_mask'] = water_mask
cfg['watermask'] = water_mask
#cfg['class_map'] = np.array(~cfg['watermask'], dtype=np.int32) + np.array(no_motion_mask, dtype=np.int32)


cfg['diff_ext'] = '.diff.adf'

itab = cfg['itab']
RSLC_tab = sar_dir + sub_dir + 'RSLC_tab'

cr_file = sar_dir + sub_dir + 'cr_loc_' + cfg['looks_str'] + '.txt'
coords = unpack_cr_coords(cr_file)
#coords = None

#dem_dir = '/local-scratch/users/aplourde/HDS/projects/southern_ITH/dems/tdx/dem_3_12/'
#dem_dir = '/local-scratch/users/aplourde/HDS/projects/southern_ITH/dems/tdx/dem_2_3/'
#dem_dir = '/local-scratch/users/aplourde/HDS/projects/southern_ITH/dems/tdx/dem_7_6/'

def snow_from_met(met_dir=inuvik_met_dir, show_plot=False):
    dict = parse_ec_dir(met_dir, freq='daily')

    snow_on_ground = dict['Snow on Grnd (cm)']
    total_snow = dict['Total Snow (cm)']
    dates = dict['Date/Time']
    dates = pd.to_datetime(dates)

    if show_plot:
        plt.plot(dates, snow_on_ground)
        plt.show()

    return pd.DataFrame(index=dates, data={'snow_cm':snow_on_ground})


def snow_real_permittivity(rho):
    # from Leinss 2015 eq(3)
    # rho is snow density in g/cm^3
    if rho > 1.:
        raise ValueError('rho>1 is not allowed!')
    if rho <= 0.41:
        a1 = 1.5995  # cm^3/g
        a3 = 1.861  # cm^9/g^3
        eps = 1 + a1 * rho + a3 * rho ** 3
    else:
        rho_ice = 0.917  # g/cm^3
        eps_h = 1.005
        eps_ice = 3.179
        eps = ((1 - rho / rho_ice) * (eps_h) ** (1.0 / 3) + rho / rho_ice * eps_ice ** (1.0 / 3)) ** 3
    return eps


def sensitivity_to_topo(rho, theta, lamb, label=None):

    eps = snow_real_permittivity(rho)
    print(theta)
    thetas = np.linspace(0, 60, 100)
    alphas = np.linspace(0, 25, 100)

    zetas = []

    #dim = master_par.dim
    #xi_im = readBin(dem_dir + 'topo/dphi_dswe.rdc', dim, 'float32')
    #inc_im = readBin(dem_dir + 'topo/inc.rdc', dim, 'float32')
    #slope_im = readBin(dem_dir + 'topo/slope.rdc', dim, 'float32')
    #phis = xi_im * 0.01

    #plt.imshow(slope_im.T)
    #plt.colorbar()
    #plt.scatter(np.rad2deg(inc_im.flatten()), phis.flatten())
    #plt.xlabel("Local Incident Angle (Deg)")
    #plt.ylabel("Phase due to 10mm SWE")
    #plt.show()

    a_grid, t_grid = np.meshgrid(alphas, thetas)
    phis = np.zeros_like(a_grid)
    angle_component = np.zeros_like(a_grid)

    """
    for i in range(len(alphas)):
        for j in range(len(thetas)):
            slope = np.deg2rad(alphas[i])
            inc = np.deg2rad(thetas[j])
            xi = 4 * np.pi / lamb / rho * (np.sqrt(eps - 1 + np.cos(inc) ** 2) - np.cos(inc)) * np.cos(slope)
            angle_component[i, j] = (np.sqrt(eps - 1 + np.cos(inc) ** 2) - np.cos(inc)) * np.cos(slope)
            phis[i, j] = xi * 0.01  # 10 mm SWE
    """
    xis = []
    phis = []
    for theta in thetas:
        #slope = np.deg2rad(alpha)
        slope = np.deg2rad(5)
        inc = np.deg2rad(theta)
        xi = 4 * np.pi / lamb / rho * (np.sqrt(eps - 1 + np.cos(inc) ** 2) - np.cos(inc)) * np.cos(slope)
        xis.append(xi)
        phis.append(xi *0.01) # 10 mm SWE

    plt.plot(thetas, phis)
    plt.xlabel(r'Local Incident Angle $\theta$ (degrees)')
    plt.ylabel('Phase due to 10 mm SWE (rad)')
    y_ticks = (np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4)
    y_tick_labels = ['π', '5π/4', '3π/2', '7π/4']  # Format the labels as fractions of pi
    plt.yticks(y_ticks, y_tick_labels)
    plt.show()
    """
    plt.subplot(121)
    plt.xlabel('local incidence angle (deg)')
    plt.ylabel('Phase due to 10 mm SWE (rad)')
    plt.scatter(t_grid.flatten(), phis)
    plt.subplot(122)
    plt.scatter(a_grid.flatten(), phis)
    plt.xlabel('slope angle (deg)')
    plt.ylabel('phase due to 10 mm SWE(rad)')
    plt.show()
    """
    plt.scatter(angle_component.flatten(), phis.flatten())
    plt.show()
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(a_grid, t_grid, phis, cmap='viridis')

    # Add labels and a colorbar
    ax.set_xlabel(r'Slope $\alpha$ (degrees)')
    ax.set_ylabel(r'Incident Angle $\theta$ (degrees)')
    ax.set_zlabel('Phi Values')
    plt.title('3D Surface Plot of Phis')
    fig.colorbar(surf, label=r'$\phi$ due to 10mm SWE (rad)')

    plt.show()

    zetas = []
    alpha = 5
    thetas = np.linspace(15, 60, 100)
    for theta in thetas:
        theta = np.deg2rad(theta)
        zetas.append(4 * np.pi / lamb / rho * (np.sqrt(eps - 1 + np.cos(theta) ** 2) - np.cos(theta)) * np.cos(np.deg2rad(alpha)))

    plt.plot(thetas, zetas, label='alpha = 5 deg')
    zetas = []
    alpha = 10
    thetas = np.linspace(15, 60, 100)
    for theta in thetas:
        theta = np.deg2rad(theta)
        zetas.append(4 * np.pi / lamb / rho * (np.sqrt(eps - 1 + np.cos(theta) ** 2) - np.cos(theta)) * np.cos(
            np.deg2rad(alpha)))

    plt.plot(thetas, zetas, label='alpha = 10 deg')
    zetas = []
    alpha = 15
    thetas = np.linspace(15, 60, 100)
    for theta in thetas:
        theta = np.deg2rad(theta)
        zetas.append(4 * np.pi / lamb / rho * (np.sqrt(eps - 1 + np.cos(theta) ** 2) - np.cos(theta)) * np.cos(
            np.deg2rad(alpha)))

    plt.plot(thetas, zetas, label='alpha = 15 deg')
    plt.xlabel('incidence angle (deg)')
    plt.ylabel('sensitivity (rad/mm)')


def mk_dphi_dswe(dem_dir, looks_str='1_1', ifg=None):
    # Phase Sensitivity to SWE as a function of
    # slope and local incidence angle
    dem_par = DEM_Par(dem_dir + 'seg.dem_par')
    dim = master_par.dim

    #inc = compute_local_incidence_angle(dem_dir, dim)
    slope = readBin(dem_dir + 'topo/slope.rdc', dim, 'float32')
    inc = readBin(dem_dir + 'topo/inc.rdc', dim, 'float32')
    #print(np.nanmean(inc))
    #inc = incidence_angle
    #print(inc)
    #inc = inc*2

    slope = np.deg2rad(slope)
    eps = snow_real_permittivity(snow_density)

    dphi_dswe = 4*np.pi/wavelength/snow_density*(np.sqrt(eps-1+np.cos(inc)**2)-np.cos(inc)) * np.cos(slope)

    if ifg:
        c_phi = readBin(ifg, dim, 'complex64')
        phi = np.angle(c_phi)
        plt.figure(figsize=(12,18))
        plt.subplot(121)
        plt.title(f"Interferogram\n{os.path.basename(ifg).split('.')[0]}")
        plt.imshow(phi.T, cmap='hsv', alpha=0.5)
        plt.colorbar()
        plt.subplot(122)
    plt.title('Phase sensitivity to SWE\n(rad/mm)')
    plt.imshow(dphi_dswe.T/1000, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()
    #sla = readBin('/local-scratch/users/jaysone/projects_active/inuvik/RS2_SLA27_D/full_scene/dem_fr/dphi_dswe.rdc',
    #              [12230, 27064], 'float32')
    #plt.imshow(sla.T/1000)
    #plt.colorbar()
    plt.show()


    dphi_dswe = nan_fill(dphi_dswe)
    writeBin(dem_dir + 'topo/dphi_dswe.rdc', dphi_dswe)
    #writeBin('/local-scratch/users/aplourde/dphi_dswe.rdc', dphi_dswe)


def compute_local_incidence_angle(dem_dir, looks_str='1_1'):

    dem_par = DEM_Par(dem_dir + 'seg.dem_par')
    #dim = dem_par.dim

    dim = master_par.dim

    slope = readBin(dem_dir + 'topo/slope.rdc', dim, 'float32')
    aspect = readBin(dem_dir + 'topo/aspect.rdc', dim, 'float32')
    slope = np.deg2rad(slope)

    water_mask = exclude_mask
    slope[water_mask] = 0.
    slope = nan_fill(slope)
    aspect[water_mask] = 0.

    lv_theta = readBin(dem_dir + 'topo/lv_theta.rdc', dim, 'float32')
    lv_phi = readBin(dem_dir + 'topo/lv_phi.rdc', dim, 'float32')

    print('Computing local incidence angle.')
    # #compute terrain slope normal vector in local ENU system
    slope_cos = np.cos(slope)
    slope_sin = np.sin(slope)
    aspect_cos = np.cos(aspect)
    aspect_sin = np.sin(aspect)
    t_u = slope_cos
    t_n = slope_sin * aspect_cos
    t_e = slope_sin * aspect_sin
    # #compute look vector in local ENU system
    lv_theta_cos = np.cos(lv_theta)
    l_u = np.sin(lv_theta)
    l_n = lv_theta_cos * np.sin(lv_phi)
    l_e = lv_theta_cos * np.cos(lv_phi)

    # #compute incidence angle
    inc = np.arccos(l_u * t_u + l_n * t_n + l_e * t_e)
    writeBin(dem_dir + 'inc.rdc', inc)

    return inc


def snow_depth_to_phase(Ds, rho, lamb, theta):
    eps = snow_real_permittivity(rho)
    phi = (4 * np.pi) / (lamb) * Ds * (np.sqrt(eps - 1 + np.cos(theta) ** 2) - np.cos(theta))
    return phi


def phase_to_snow_depth(phase, rho, lamb, theta):
    eps = snow_real_permittivity(rho)
    depth = (phase * lamb) / (4 * np.pi * (np.sqrt(eps - 1 + np.cos(theta) ** 2) - np.cos(theta)))
    return depth


def phi_snow_depth_map(Ds, dem_dir, dim, lamb, rho, show_plot=False):

    inc_map = readBin(dem_dir + 'inc.rdc', dim, 'float32')

    eps = snow_real_permittivity(rho)
    phi = snow_depth_to_phase(Ds, rho, lamb, inc_map)
    #phi = (4*np.pi)/(lamb) * Ds * (np.sqrt(eps-1+np.cos(inc_map)**2)-np.cos(inc_map))

    if show_plot:
        plt.subplot(131)
        plt.title('Local Incidence Angle with Respect to\nTopographic Slope')
        plt.imshow(inc_map.T)
        plt.colorbar()
        plt.subplot(132)
        plt.title("Phase due to {} cm of\nSnow Accumulation".format(Ds * M_TO_CM))
        plt.imshow(phi.T)
        plt.colorbar()
        plt.subplot(133)
        plt.title("Wrapped Phase due to {} cm of\nSnow Accumulation".format(Ds * M_TO_CM))
        plt.imshow(np.angle(np.exp(1j*phi.T)), vmin=-np.pi, vmax=np.pi, cmap='hsv', alpha=0.5)
        plt.colorbar()
        plt.show()
    return phi


def nan_fill(arr_, win = (3,3)):
    print('nan_fill...')
    arr = np.copy(arr_)
    bad_mask = ~np.isfinite(arr)
    num_bad = np.sum(bad_mask)
    ii=0
    while(num_bad > 0):
        print('iteration:', ii+1, ', found', num_bad, 'bad values.')
        arr[bad_mask] = 0
        norm_filt = uniform_filter((~bad_mask).astype(float), win)
        norm_filt[norm_filt < 1/win[0]/win[1]]=0
        arr_filt = uniform_filter(arr, win)/norm_filt
        arr[bad_mask] = arr_filt[bad_mask]
        bad_mask = ~np.isfinite(arr)
        num_bad = np.sum(bad_mask)
        ii+=1

    return arr


# Jayson Eppler 2022
def phase_coh(phi, win, normalize=True):
    if normalize:
        #this should be set to false only for data that has already been normalized as a time saver
        phi/=np.abs(phi)
    phi_filt = np.empty_like(phi)
    phi[~np.isfinite(phi)]=0
    phi_filt.real = uniform_filter(phi.real, win)
    phi_filt.imag = uniform_filter(phi.imag, win)
    coh = np.abs(phi_filt)
    return coh


# Jayson Eppler 2022
def est_swe(phi_full, xi_full, swe_range=(-0.1, 0.1), swe_del=0.002, win=(64, 64), exclude_mask_full=None, weighted=False, point_xy=None):

    if point_xy is not None:
        cmin = point_xy - win // 2
        cmax = cmin + win

        print(cmin, cmax)
        print(phi_full.shape)

        # extract local patch
        phi = phi_full[cmin[0]:cmax[0], cmin[1]:cmax[1]]
        xi = xi_full[cmin[0]:cmax[0], cmin[1]:cmax[1]]

        exclude_mask = exclude_mask_full[cmin[0]:cmax[0], cmin[1]:cmax[1]]
        #plt.imshow(exclude_mask.T)
        #plt.show()
        #exclude_mask = mask_patch.copy()
        phi[exclude_mask] = np.nan
        #kernal = np.ones_like(phi)
        #if kernal is not None:
        #    phi[~kernal] = np.nan

    else:
        phi = phi_full
        xi = xi_full
        exclude_mask = exclude_mask_full

    n_interp=3
    n_interp_rad = (n_interp - 1) // 2
    valid_thresh = 0 #need at least this fraction of valid points under esimation window to be considered a valid estimation
    swes = swe_range[0] + np.arange(int((swe_range[1] - swe_range[0]) / swe_del)) * swe_del
    #np.flip(swes)
    n_swes = len(swes)
    dim = phi.shape
    print(dim)
    if not weighted:
        #no weights so just normalize each phasor to unit length
        phi = phi / np.abs(phi)
    phi[np.isnan(phi)] = 0.
    if exclude_mask is not None:
        phi[exclude_mask] = 0.
        #exclude_mask = None
    valid_mask = (phi != 0)
    abs_filt = uniform_filter(np.abs(phi), win)
    #abs_filt = uniform_filter(np.abs(phi), win) ## TODO: weight filter
    #plt.figure()
    #plt.imshow(valid_mask_filt.T)
    # valid_mask_final = valid_mask_filt > valid_thresh
    valid_mask_final = valid_mask
    valid_mask=None
    interp_points = np.zeros((n_interp, dim[0], dim[1]), dtype='float32') - 1.  # n_interp points in vicinity of peak
    idx_max = np.zeros(dim, dtype=int) - 1
    buffer = np.zeros((n_interp, dim[0], dim[1]), dtype='float32')  # last n_interp points
    for ii in np.arange(n_swes):
        # print(ii+1, 'of', n_swes)
        phi_demod_ii = phi * np.exp(-1j * swes[ii] * xi)  # Calculate Phase demodulated by SWE
        coh_ii = phase_coh(phi_demod_ii, win, normalize=False) / abs_filt  # Calculate associated coherance
        if ii < n_interp:
            # initial phase, just populate buffer
            buffer[ii, :, :] = coh_ii
        if ii >= n_interp:
            # later phase, shift buffer, then add latest entry to end
            buffer[0:-1, :, :] = buffer[1:, :, :]
            buffer[-1, :, :] = coh_ii

            # find targets with new max coh
            update_mask = buffer[n_interp_rad, :, :] > interp_points[n_interp_rad, :, :]
            """
            plt.subplot(121)
            plt.title(f"SWE: {swes[ii]}")
            plt.imshow(update_mask.T)
            plt.subplot(122)
            plt.title("Coherance")
            plt.imshow(coh_ii.T)
            plt.show()
            """
            # print(np.sum(update_mask), 'new peak coherences found.')
            # update interp_points for those targets with new max coh
            interp_points[:, update_mask] = buffer[:, update_mask]
            idx_max[update_mask] = ii - n_interp_rad
            center_idx_max = ii - n_interp_rad
            """
            plt.subplot(121)
            plt.title("Max Index")
            plt.imshow(idx_max.T)
            plt.subplot(122)
            plt.title("Update Mask")
            plt.imshow(update_mask.T, cmap='Greys_r')
            plt.show()
            """
    buffer = None
    #phi=None
    #xi=None
    phi_demod_ii=None
    coh_ii = None
    update_mask=None
    valid_mask_filt=None
    # coh_max = interp_points[n_interp_rad,:,:]
    swe_est = np.reshape(swes[idx_max], dim)
    idx_max=None

    """
    pnts = get_site_centers()
    plt.subplot(121)
    plt.title(f"Swe Est\nSite 1: {swe_est[pnts[0][0], pnts[1][0]]*M_TO_MM:.2f}; Site 2: {swe_est[pnts[0][1], pnts[1][1]]*M_TO_MM:.2f}")
    plt.imshow(swe_est.T, vmin=-0.06, vmax=0.06, cmap='RdBu')
    plt.colorbar()
    plt.scatter(pnts[0], pnts[1], color='red')
    """

    # quadratic components (hardcoded for n_interp = 3)
    a = interp_points[1, :, :]
    b = -0.5 * interp_points[0, :, :] + 0.5 * interp_points[2, :, :]
    c = 0.5 * interp_points[0, :, :] - 1.0 * interp_points[1, :, :] + 0.5 * interp_points[2, :, :]
    interp_points=None
    dx = -b / 2 / c
    # handle cases where center point is not the highest
    dx[dx < -1] = -1
    dx[dx > 1] = 1.

    swe_est = swe_est + dx * swe_del
    coh_max = a + b * dx + c * dx ** 2
    a = None; b=None; c=None; dx=None

    swe_est[~valid_mask_final] = np.nan
    coh_max[~valid_mask_final] = np.nan

    """
    plt.subplot(122)
    plt.title(f"Swe Est\nSite 1: {swe_est[pnts[0][0], pnts[1][0]]*M_TO_MM:.2f}; Site 2: {swe_est[pnts[0][1], pnts[1][1]]*M_TO_MM:.2f}")
    plt.imshow(swe_est.T, vmin=-0.06, vmax=0.06, cmap='RdBu')
    plt.colorbar()
    plt.scatter(pnts[0], pnts[1], color='red')
    plt.show()
    """

    return swe_est, coh_max, xi, phi, exclude_mask


def swe_stack(diffs=None, met_dir=inuvik_met_dir, method=None, pt_xy=None):

    dim = master_par.dim

    if diffs is None:

        files = glob.glob(ifg_dir + '*' + cfg['diff_ext'])
        print(files)
        diffs = get_itab_diffs(files, itab, RSLC_tab)
        print(diffs)

    xi = readBin(dem_dir + 'topo/dphi_dswe.rdc', dim, 'float32')

    for ifg in diffs:

        root = os.path.basename(ifg).split('.')[0]

        m_date, s_date = parse_ifg_dates(root)

        ### HDS SWE Estimates ###
        phi = readBin(ifg, dim, 'complex64')
        #water_mask = cfg['watermask']
        mask = cfg['exclude_mask']
        #phi[mask] = 0.

        if os.path.exists(cfg['swe_est_dir'] + root + '.diff.swe'):
            print("file already exists, skipping...")
            continue
        swe_est, coh_max, xi_patch, phi_patch, mask_patch = est_swe(phi, xi, swe_range=cfg['swe_range'], swe_del=cfg['swe_del'], win=cfg['win'], exclude_mask_full=mask,
                                   weighted=False, point_xy=pt_xy)

        ### Point SWE Estimates ###
        if method == 'point':
            #snow_df = snow_from
            snow_init = snow_df[snow_df.index == pd.to_datetime(m_date)]['snow_cm'].values[0]
            snow_end = snow_df[snow_df.index == pd.to_datetime(s_date)]['snow_cm'].values[0]
            snow_accumulation_cm = (snow_end - snow_init)
            snow_accumulation_m = snow_accumulation_cm * 0.01
            phi_theory = phi_snow_depth_map(snow_accumulation_m, dem_dir, dim, wavelength, snow_density, show_plot=0)

        ### Map SWE Estimates
        if method == 'era5':
            snow_df = snow_from_met(met_dir, show_plot=False)
            sde = era5_met_dir + 'delta_snow_depth/crop_sites/' + m_date + '_' + s_date + '.sde'
            sde_im = readBin(sde, dim, 'float32')
            snow_accumulation_m = np.nanmean(sde_im)
            snow_accumulation_cm = snow_accumulation_m * 100
            phi_theory = phi_snow_depth_map(sde_im, dem_dir, dim, wavelength, snow_density, show_plot=0)
            cphi_theory = np.exp(1j * phi_theory)
            swe_est_theory, coh_max = est_swe(cphi_theory, xi, swe_range=(-0.1, 0.1), swe_del=0.002, win=(64, 64),
                                              exclude_mask=None, weighted=False)
            sde_est_theory = np.nanmean(swe_est_theory / snow_density)

        ### Plot Swe Est
        if len(diffs) < 5 and False:
            #plt.subplot(141)
            #plt.title(f'Iterferogram: {m_date} - {s_date}')
            #plt.imshow(np.angle(phi.T), vmin=-np.pi, vmax=np.pi)
            #plt.colorbar()
            plt.subplot(121)
            plt.title('Theoretical Interferogram\ndue to ERA-5 Snow Accumulation\n(Aveage of {:.2}cm)'.format(snow_accumulation_cm))
            plt.imshow(np.angle(cphi_theory.T))
            plt.colorbar()
            #plt.subplot(143)
            #plt.title('Estimated SWE from HDS Interferogram\nScene Average: {:.2}\n(Est. Snow Depth = {:.2f}cm)'.format(np.nanmean(swe_est), np.nanmean(swe_est)*100 / snow_density))
            #plt.imshow(swe_est.T, vmin=-0.1, vmax=0.1, cmap='RdBu')
            #plt.colorbar()
            plt.subplot(122)
            plt.title('Estimated SWE from Theoretical Interferogram\nScene Average: {:.2}\n(Est. Snow Depth = {:.2f}cm)'.format(np.nanmean(swe_est_theory), sde_est_theory*100))
            plt.imshow(swe_est_theory.T, vmin=-0.2, vmax=0.2, cmap='RdBu')
            plt.colorbar()
            #plt.savefig(out_dir + root + '.png')
            plt.show()
        else:
            print(cfg['swe_est_dir'] + root + cfg['diff_ext'] + '.swe')
            writeBin(cfg['swe_est_dir'] + root + cfg['diff_ext'] + '.swe', swe_est)
            writeBin(cfg['swe_est_dir'] + root + cfg['diff_ext'] + '.swe.coh', coh_max)
            writeBin(cfg['swe_est_dir'] + root + cfg['diff_ext'] + '.patch', phi_patch)
    writeBin(cfg['swe_est_dir'] + 'patch.xi', xi_patch)
    print(np.asarray(mask_patch, dtype=float))
    writeBin(cfg['swe_est_dir'] + 'patch.mask', np.asarray(mask_patch, dtype=float))


def combine_snow_with_EC_and_tilt():

    out = {}
    for (tf, sf) in zip(tilt_files, snow_files):
        print(tf, sf)
        site = re.search(r'site_.', tf).group(0)

        tdf = pd.read_csv(tf, index_col=0)
        sdf = pd.read_csv(sf, index_col=0)

        tdf.index = pd.to_datetime(tdf.index)
        sdf.index = pd.to_datetime(sdf.index)

        sdf = sdf[sdf.index >= first_snow]
        sdf = sdf[sdf.index < last_snow]
        tdf = tdf[tdf.index >= first_snow]
        tdf = tdf[tdf.index < last_snow]
        print(sdf)

        #sdf.snow_depth_cm = sdf.snow_depth_cm - sdf.snow_depth_cm.loc[first_snow]
        if first_snow in tdf.index.values:
            tdf.dh1_mm = tdf.dh1_mm - tdf.dh1_mm.loc[first_snow]


        if 'Snow on Grnd (cm)_x' not in list(sdf):
            sdf = getECMetData(sdf, met_station="Inuvik")
            sdf = getECMetData(sdf, met_station="TrailValley")

        #sdf.loc[sdf['Snow on Grnd (cm)_x'].isna(), 'snow_depth_cm'] = np.nan
        sdf['snow_sub_heave'] = sdf.snow_depth_cm #- tdf.dh1_mm

        """
        sdf['heave'] = np.nan
        sdf['dh1_mm'] = np.nan

        tilt = 'dh1_mm'
        count_nan = 0
        zero = 0

        for index, row in sdf.iterrows():
            sdf.dh1_mm.loc[index] = tdf.dh1_mm.loc[tdf.index == index]

            if pd.isna(row.snow_depth_cm):
                count_nan += 1
                if count_nan > 30:
                    zero = 0
            else:
                if zero == 0 and count_nan > 30:
                    zero = row.snow_depth_cm
                    if tilt is not None:
                        try:
                            tilt_zero = tdf[tilt].loc[index] * 0.1
                            print(f"tilt zero set to {tilt_zero}")
                        except:
                            print(f"Error: index {index} not found in {site}")

                new_val = row.snow_depth_cm - zero
                if new_val < 0:
                    zero = row.snow_depth_cm
                    #sdf.snow_depth_cm[index] = np.nan
                    sdf.snow_depth_cm[index] = 0
                    if tilt is not None:
                        try:
                            tilt_zero = tdf[tilt].loc[index] * 0.1
                            sdf.heave[index] = np.nan
                        except:
                            print(f"Error: index {index} not found in {site}")
                else:
                    count_nan = 0
                    sdf.snow_depth_cm[index] = row.snow_depth_cm - zero
                    if tilt is not None:
                        try:
                            sdf.heave[index] = tdf[tilt].loc[index] * 0.1 - tilt_zero
                        except:
                            print(f"Error: index {index} not found in {site}")
            if tilt is not None:
                sdf.snow_sub_heave[index] = row.snow_depth_cm - zero - sdf.heave[index]
        """

        out[site] = sdf
    return out


def plot_snow_with_tilt(swe=None):
    start_date = pd.to_datetime('2022-07-21')
    end_date = pd.to_datetime('2023-08-01')
    n_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    xtics = pd.date_range(start=start_date, end=end_date, freq='M').tolist()
    data = combine_snow_with_EC_and_tilt()

    """
    for site in data:
        print(data[site])
        plt.title(site)
        #plt.plot(data[site].index, data[site].snow_depth_cm, label="Measured Snow Depth")
        #plt.plot(data[site].index, data[site].snow_sub_heave, label="Corrected Snow Depth")
        plt.plot(data[site].index, data[site].snow_depth_cm - data[site].snow_sub_heave, label=f"Correction - {site}")
        plt.plot(data[site].index, data[site].heave, label=f"Heave - {site}")
        plt.legend()
        plt.show()
        data[site].to_csv(f"/local-scratch/users/aplourde/field_data/{site}/{site}_combined.csv")
        #return
    """

    era5 = dailyERAatSite()

    style_map = {'site_1': {'color': '#e41a1c', 'linestyle': '-', 'label': "Site 1: Homogenous 1"},
                 'site_2': {'color': '#377eb8', 'linestyle': '-', 'label': "Site 2: Low Ground"},
                 'site_3': {'color': '#4daf4a', 'linestyle': '--', 'label': "Site 3: High Ground"},
                 'site_4': {'color': '#ff7f00', 'linestyle': '--', 'label': "Site 4: Homogenous 2"},
                 'site_5': {'color': '#984ea3', 'linestyle': '-.', 'label': "Site 5: Hill Top"},
                 'site_6': {'color': '#a65628', 'linestyle': '-.', 'label': "Site 6: Jimmy Lake"}
                 }

    fig, axes = plt.subplots(1, 1, sharex='col', sharey=False, gridspec_kw={'wspace': 0, 'hspace': 0})

    for site in data:
        if site == 'site_4':
            continue
        #data[site] = data[site][data[site].index >= pd.to_datetime(start_date)]
        axes.plot(data[site].index, data[site].snow_depth_cm,
                     style_map[site]['linestyle'], color=style_map[site]['color'], label=style_map[site]['label'])

    axes.plot(data['site_1'].index, data['site_1']['Snow on Grnd (cm)_x'], color='black', label = 'Env. Canada - Inuvik')
    axes.plot(data['site_1'].index, data['site_1']['Snow on Grnd (cm)_y'], '--', color='black', label='Env. Canada - Trail Valley')
    list(era5)
    axes.set_ylabel('Snow Depth (cm)')
    #axes[0].set_ylim(dlim)

    class XFormatter:
        def __call__(self, x, pos=None):
            return '' if pos % 2 else f'{x:.1f}'

    xlabels = []
    for i, label in enumerate(xtics):
        xlabels.append('') if i % 3 else xlabels.append(label.strftime('%Y-%m'))
    axes.set_xlabel("Date")
    axes.set_xticks(xtics)
    axes.set_xticklabels(xlabels)
    axes.set_xlim(start_date, end_date)

    axes.legend()
    #axes[1].legend(handles=[t, p])

    plt.show()

    """
    
    # fig, axes = plt.subplots(131, sharex='col', sharey=False, gridspec_kw = {'wspace':0, 'hspace':0})
    subplot_titles = {'site_1': 'Site 1 - Homogenous',
                      'site_2': 'Site 2 - Low Ground',
                      'site_3': 'Site 3 - High Ground'}
    plot_titles = ['2019-2020', '2020-2021', '2021-2022']

    years = {'2019-2020': pd.to_datetime(['20190901', '20200601']),
             '2020-2021': pd.to_datetime(['20200901', '20210601']),
             '2021-2022': pd.to_datetime(['20210901', '20220601'])}

    data_by_year = {}

    for y in years:
        data_by_year[y] = {}
        for site in data:
            df = data[site]
            df = df[(df.index > years[y][0]) & (df.index < years[y][1])]
            if ~df.snow_depth_cm.isna().all():
                data_by_year[y][site] = df

    for year, sites in data_by_year.items():

        df5 = era5[era5.index > years[year][0]]
        df5 = df5[df5.index < years[year][1]]
        if swe is not None:
            sar_swe = swe.copy()
            sar_swe['site_1'] = sar_swe['site_1'][sar_swe['site_1']['slave'] > years[year][0]]
            sar_swe['site_1'] = sar_swe['site_1'][sar_swe['site_1']['slave'] < years[year][1]]
            print(sar_swe['site_1'])
        fig, axes = plt.subplots(len(sites), 1, sharex=False, sharey=False, figsize=(8, 8))
        # fig.suptitle(year)
        ss = 0
        for site, df in sites.items():
            tilt = 'dh1_mm'
            if len(sites) > 1:
                if len(sar_swe[site]) > 0:
                    print(sar_swe[site]['slave'].values[0], df['Snow on Grnd (cm)_x'][df.index == sar_swe[site]['slave'].values[0]].values[0])
                    sar_swe[site]['delta_sde'] = sar_swe[site]['delta_sde']*100 + df['Snow on Grnd (cm)_x'][df.index == sar_swe[site]['slave'].values[0]].values[0]
                axes[ss].scatter(sar_swe[site]['slave'], sar_swe[site]['delta_sde'], color='lime', marker='*', label="SAR SWE")
                axes[ss].scatter(df.index, df.snow_depth_cm, color='darkred', s=4, label="Ultra-Sonic Sensor (Zeroed at Freezeback)")
                if tilt is not None:
                    axes[ss].scatter(df.index, df.snow_sub_heave, color='red', s=4, label="Ultra-Sonic Sensor - Heave Corrected")
                    axes[ss].plot(df.index, df.heave, '--', color='grey', label="Inclinometer (Zeroed at Freezeback)")
                axes[ss].plot(df.index, df['Snow on Grnd (cm)_x'], color='black', label="Inuvik (EC)")
                axes[ss].plot(df.index, df['Snow on Grnd (cm)_y'], color='blue', label="Trail Valley (EC)")
                axes[ss].plot(df5.index, df5['snow_depth_m']*100, color='green', label="ERA5 Reanalysis")
                #axes[ss].set_ylim([-5, 80])
                axes[ss].set_xlim(years[year])
                axes[ss].set_ylabel('centimers (cm)')
                axes[ss].set_title(subplot_titles[site])
                ss += 1
                axes[1].legend()
            else:
                axes.scatter(sar_swe[site]['slave'], sar_swe[site]['delta_sde'] * 100, color='lime', marker='*',
                                 label="SAR SWE")
                axes.scatter(df.index, df.snow_depth_cm, color='darkred', s=4,
                                 label="Ultra-Sonic Sensor (Zeroed at Freezeback)")
                if tilt is not None:
                    axes.scatter(df.index, df.snow_sub_heave, color='red', s=4,
                                     label="Ultra-Sonic Sensor - Heave Corrected")
                    axes.plot(df.index, df.heave, '--', color='grey', label="Inclinometer (Zeroed at Freezeback)")
                axes.plot(df.index, df['Snow on Grnd (cm)_x'], color='black', label="Inuvik (EC)")
                axes.plot(df.index, df['Snow on Grnd (cm)_y'], color='blue', label="Trail Valley (EC)")
                axes.plot(df5.index, df5['snow_depth_m'] * 100, color='green', label="ERA5 Reanalysis")
                #axes.set_ylim([-5, 80])
                axes.set_ylabel('centimers (cm)')
                axes.set_title(subplot_titles[site])
                axes.legend()

        plt.show()
        plt.close()
    """


def site_swe(ifg_swe_estimates=None, show_results=False, plt_annulus=False, site='site_1', input_coords=None):

    dim = master_par.dim

    #point_data = combine_snow_with_EC_and_tilt()
    #point_data = point_data['site_1']
    #point_data.snow_depth_cm = point_data.snow_depth_cm.interpolate()

    xi_im = readBin(dem_dir + 'topo/dphi_dswe.rdc', dim, 'float32')

    if ifg_swe_estimates is None:
        files = glob.glob(cfg['swe_est_dir'] + '*' + cfg['diff_ext'] + '.swe')
        ifg_swe_estimates = get_itab_diffs(files, itab, RSLC_tab)

    if input_coords is None:
        if site == 'site_1':
            site_center = np.mean([coords['cr1'], coords['cr2']], axis=0)
            site_center = [[int(px) for px in site_center]]
        elif site == 'site_2' or site == 'site_3':
            site_center = [coords['cr5']]
    elif input_coords == 'all':
        site_center = None
    else:
        site_center = input_coords

    swe_estimates = []

    for file in ifg_swe_estimates:
        path, root = os.path.split(file)
        root = root.split('.')[0]

        m_str, s_str = parse_ifg_dates(root)
        m_date = pd.to_datetime(m_str)
        s_date = pd.to_datetime(s_str)

        #swe_im = readBin(file + '.interp', dim, 'float32')
        swe_im = readBin(file, dim, 'float32')
        cc_im = readBin(file + '.coh', dim, 'float32')

        cc = []
        swe_est = []
        if site_center is not None:
            for coord in site_center:
                swe, mask = extract_annulus(swe_im, coord, 0, annulus, show_plot=0)
                #coord_cc, cc_mask = extract_annulus(cc_im, coord, 0, annulus, show_plot=0)
                #xi, xi_mask = extract_annulus(xi_im, coord, 0, annulus, show_plot=0)
                #phi, phi_mask = extract_annulus(phi_im, coord, 0, annulus, show_plot=0)

                coord_swe = np.nanmean(swe[mask])
                coord_cc = cc_im[mask]

                cc.append(cc_im[mask])
                swe_est.append(swe[mask])
        else:

            #plt.imshow(swe_im.T, cmap='RdBu')
            #plt.colorbar()
            #plt.show()

            swe_est = swe_im
            cc = cc_im
            
        swe_stats = [np.nanmean(swe_est), np.nanstd(swe_est), np.nanmin(swe_est), np.nanmax(swe_est)]
        print(swe_stats)

        rswe = get_relative_swe(m_str, s_str)

        swe_estimates.append((m_date, s_date, np.nanmean(cc), swe_stats, rswe['ERA5-Reanalysis'], rswe['site_1'],
                              rswe['site_2'], rswe['site_3'], rswe['EC-Inuvik'], rswe['EC-TrailValley']))

        if plt_annulus:
            an_lim_x = [site_center[0][0] - 250, site_center[0][0] + 250]
            an_lim_y = [site_center[0][1] + 250, site_center[0][1] - 250]
            plt.figure(figsize=[18, 12])
            plt.subplot(221)
            plt.title(f"Phi\n{m_str}_{s_str}")
            plt.imshow(phi_im[mask], vmin=-0.2, vmax=0.2, cmap='hsv', alpha=0.5)
            plt.xlim(an_lim_x)
            plt.ylim(an_lim_y)
            plt.subplot(222)
            plt.title(f"Coherance\n{m_str}_{s_str}")
            #plt.imshow(cc_im[mask].T, cmap='Greys_r', vmin=0, vmax=1)
            plt.xlim(an_lim_x)
            plt.ylim(an_lim_y)
            plt.subplot(223)
            plt.title(f"SWE Estimate\n{root}")
            plt.imshow(swe.T, vmin=-0.2, vmax=0.2, cmap='RdBu')
            plt.colorbar()
            plt.xlim(an_lim_x)
            plt.ylim(an_lim_y)
            plt.subplot(144)
            plt.title("Phi versus Xi")
            plt.scatter(phi_im[mask], xi_im[mask])
            plt.xlabel("Xi")
            plt.ylabel("Phi")
            plt.show()
            #plt.savefig('/local-scratch/users/aplourde/snow/estimates/' + m_str + '_' + s_str +'.png')
            #plt.close()
            #break


    swe_est_df = pd.DataFrame(swe_estimates, columns=['master', 'slave', 'mean_coh', 'swe_est', 'swe_era5', 'swe_site_1', 'swe_site_2', 'swe_site_3', 'swe_inuvik', 'swe_trailValley'])
    #swe_est_df['hds_era5_error'] = swe_est_df['swe_hds'].str[0] - swe_est_df['swe_era5']
    #swe_est_df['hds_site_1_error'] = swe_est_df['swe_hds'].str[0] - swe_est_df['swe_site_1']

    res = swe_est_df
    #print(res)

    if show_results:

        #print(f"Error between estimate and ERA5: }")
        err_era5 = np.abs(swe_est_df['swe_est'].str[0] - swe_est_df['swe_era5'])
        err_inuv = np.abs(swe_est_df['swe_est'].str[0] - swe_est_df['swe_inuvik'])
        err_tlvl = np.abs(swe_est_df['swe_est'].str[0] - swe_est_df['swe_trailValley'])
        err_site1 = np.abs(swe_est_df['swe_est'].str[0] - swe_est_df['swe_site_1'])
        err_site2 = np.abs(swe_est_df['swe_est'].str[0] - swe_est_df['swe_site_2'])
        err_site3 = np.abs(swe_est_df['swe_est'].str[0] - swe_est_df['swe_site_3'])

        errs = pd.DataFrame(data={'err_era5': err_era5,
                                      'err_inuv': err_inuv,
                                      'err_tlvl': err_tlvl,
                                      'err_site1': err_site1,
                                      'err_site2': err_site2,
                                      'err_site3': err_site3})

        err_df = pd.DataFrame({'Mean':errs.mean(),
                            'Total': errs.sum(),
                            'Std': errs.std(),
                            'Min': errs.min(),
                            'Max': errs.max()})
        print(err_df)

        ### Correlation ###
        parameter1 = 'swe_est'
        parameter2 = 'swe_era5'

        swe_est_df = swe_est_df.dropna(subset=[parameter2])

        plt.figure()

        plt.title(f"{parameter2} vs {parameter1}")
        plt.fill_between([0, 0, 0.1], [0, 0.1, 0.1], facecolor='green', alpha=0.1)
        plt.fill_between([0, 0, 0.1], [0, -0.1, -0.1], facecolor='red', alpha=0.1)
        plt.fill_between([0, 0, -0.1], [0, -0.1, -0.1], facecolor='green', alpha=0.1)
        plt.fill_between([0, 0, -0.1], [0, 0.1, 0.1], facecolor='red', alpha=0.1)
        plt.plot([-0.1, 0.1], [-0.1, 0.1], color="black")

        plt.scatter(swe_est_df[parameter2], swe_est_df['swe_est'].str[0], c=swe_est_df['mean_coh'], cmap='RdYlGn')#, vmin=0, vmax=1)
        clb = plt.colorbar()

        norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='RdYlGn')
        colors = np.array([(mapper.to_rgba(c)) for c in swe_est_df['mean_coh']])

        for (x, y, e, c) in zip(swe_est_df[parameter2], swe_est_df['swe_est'].str[0], swe_est_df['swe_est'].str[1], colors):
            plt.errorbar(x, y, e, lw=1, capsize=3, color=c)


        plt.xlabel(f"{parameter2}")
        plt.ylabel(f"{parameter1}")


        #annotations = []
        #for idx, row in swe_est_df.iterrows():
        #    annotations.append(plt.text(row['swe_est'].str[0], row[parameter2], row['master'].strftime('%Y%m%d'), fontsize=8))#, arrowprops=dict(arrowstyle='-',color='red')))
        #adjust_text(annotations, swe_est_df['swe_est'].str[0].copy(), swe_est_df[parameter2].copy(), expand_text=(1, 1), expand_objects=(2,2), expand_points=(2,2), arrowprops=dict(arrowstyle='-',color='black', alpha=0.5, shrinkA=0.05))#, time_lim=1)#, force_static=(2,2), expand=(1.05, 10), , arrowprops=dict(facecolor='black', arrowstyle='->'))


        ### Histogram ###
        plt.figure()
        classes = np.array(cfg['exclude_mask'], dtype=int)
        classed_histogram(xi_im, classes)


        ### Timeseries ###
        plt.figure()
        timeseries_swe(res)


        ## mask ##
        plt.figure()
        mask = np.array(~cfg['exclude_mask'], dtype=int)
        plt.imshow(mask.T, cmap='Greys_r')
        plt.show()



    return res


def classed_histogram(xi, class_map):
    #print(class_map, type(class_map), class_map.shape)
    num_classes = np.ptp(class_map) + 1
    #print(f"NUM CLASSES: {num_classes}")
    num_samples_per_class = [len(xi[class_map==cc]) for cc in range(num_classes)]
    ylim=0
    for kk in range(num_classes):
        #print(f"KK: {kk}")
        plt.subplot(1, num_classes, kk+1)
        n, b, p = plt.hist(xi[class_map == kk], range=[np.nanmin(xi), np.nanmax(xi)], bins=25)
        # plt.hist(xi_patch.flatten(), bins=100)
        #plt.xlim([np.nanmin(xi), np.nanmax(xi)])
        #plt.ylim([0, ylim])
        plt.xlabel(f"Xi - Class {kk}")
        plt.ylabel("Num Samples")
        #print(n, b, p)
        if np.max(n) > ylim:
            ylim = round(np.max(n) + 1000, -3)
    for kk in range(num_classes):
        plt.subplot(1, num_classes, kk+1)
        plt.ylim([0, ylim])
    #plt.show()


def point_swe_clicky_func(pt_xy, show_results=True):

    dim = master_par.dim

    files = glob.glob(ifg_dir + '*' + cfg['diff_ext'])
    diffs = get_itab_diffs(files, itab, RSLC_tab)

    xi = readBin(dem_dir + 'topo/dphi_dswe.rdc', dim, 'float32')

    swe_estimates = []

    for ifg in diffs:
        root = os.path.basename(ifg).split('.')[0]

        m_date, s_date = parse_ifg_dates(root)

        ### HDS SWE Estimates ###
        phi = readBin(ifg, dim, 'complex64')
        #mask = cfg['watermask']
        #mask = cfg['class_map']
        mask = cfg['exclude_mask']
        #mask = no_motion_mask

        swe_est, coh_max, xi_patch, phi_patch, mask_patch = est_swe(phi, xi, swe_range=cfg['swe_range'],
                                                                    swe_del=cfg['swe_del'], win=cfg['win'],
                                                                    exclude_mask_full=mask,
                                                                    weighted=False, point_xy=pt_xy)

        xi_im = xi_patch
        exclude_mask = mask_patch

        xi_masked = xi_im[~exclude_mask]
        xi_demean = xi_masked-np.nanmean(xi_masked)

        xpts = np.asarray([np.min(xi_demean), np.max(xi_demean)])
        rswe = get_relative_swe(m_date, s_date, show_results=0)

        swe_im = swe_est
        cphi_im = phi_patch
        cphi_im[exclude_mask] = 0
        cphi_masked = cphi_im[~exclude_mask]

        phi_demean = np.angle(cphi_masked * np.conj(np.nanmean(cphi_masked)))
        phi_demean -= np.nanmean(phi_demean)

        #cc = phase_coh(cphi_im, win1000)
        #cc = np.abs(np.mean(cphi_im))/np.mean((~exclude_mask).astype(float))
        cc = coh_max

        #if np.nanmean(cc) < 0.1:
        #    continue

        swe_est_refined = swe_im

        #swe_weighted = np.average(swe_est[~exclude_mask], weights=cc[~exclude_mask])
        swe_weighted = np.nanmean(swe_im)


        #swe_stats = [np.nanmean(swe_est), np.nanstd(swe_est), np.nanmin(swe_est), np.nanmax(swe_est)]
        swe_stats = [swe_weighted, np.nanstd(swe_est), np.nanmin(swe_est), np.nanmax(swe_est)]

        swe_estimates.append((m_date, s_date, np.nanmean(cc), swe_stats, rswe['ERA5-Reanalysis'], rswe['site_1'],
                              rswe['site_2'], rswe['site_3'], rswe['EC-Inuvik'], rswe['EC-TrailValley']))



    swe_est_df = pd.DataFrame(swe_estimates,
                              columns=['master', 'slave', 'mean_coh', 'swe_est', 'swe_era5', 'swe_site_1', 'swe_site_2',
                                       'swe_site_3', 'swe_inuvik', 'swe_trailValley'])

    res = swe_est_df

    if show_results:
        ### Correlation ###
        parameter1 = 'swe_est'
        parameter2 = 'swe_inuvik'

        swe_est_df = swe_est_df.dropna(subset=[parameter2])

        plt.figure()
        #plt.subplot(121)
        plt.title(f"{parameter2} vs {parameter1}")
        plt.fill_between([0, 0, 0.1], [0, 0.1, 0.1], facecolor='green', alpha=0.1)
        plt.fill_between([0, 0, 0.1], [0, -0.1, -0.1], facecolor='red', alpha=0.1)
        plt.fill_between([0, 0, -0.1], [0, -0.1, -0.1], facecolor='green', alpha=0.1)
        plt.fill_between([0, 0, -0.1], [0, 0.1, 0.1], facecolor='red', alpha=0.1)
        plt.plot([-0.1, 0.1], [-0.1, 0.1], color="black")

        plt.scatter(swe_est_df[parameter2], swe_est_df['swe_est'].str[0], c=swe_est_df['mean_coh'], cmap='RdYlGn', vmin=0, vmax=1)
        clb = plt.colorbar()

        norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='RdYlGn')
        colors = np.array([(mapper.to_rgba(c)) for c in swe_est_df['mean_coh']])

        for (x, y, e, c) in zip(swe_est_df[parameter2], swe_est_df['swe_est'].str[0], swe_est_df['swe_est'].str[1], colors):
            plt.errorbar(x, y, e, lw=1, capsize=3, color=c)

        plt.xlabel(f"{parameter2}")
        plt.ylabel(f"{parameter1}")

        plt.figure()
        class_patch = np.array(mask_patch, dtype=np.int)
        classed_histogram(xi_patch, class_patch)

        plt.figure()
        timeseries_swe(res)
        plt.clf()

    return res


def point_swe(pt_xy, ifg_swe_estimates=None, show_results=False, plt_patch=False):

    est_dim = cfg['win']
    dim = master_par.dim

    xi_im = readBin(cfg['swe_est_dir'] + 'patch.xi', est_dim, 'float32')
    exclude_mask = readBin(cfg['swe_est_dir'] + 'patch.mask', est_dim, 'float32') != 0

    xi_masked = xi_im[~exclude_mask]
    xi_demean = xi_masked-np.nanmean(xi_masked)

    xpts = np.asarray([np.min(xi_demean), np.max(xi_demean)])

    if ifg_swe_estimates is None:
        files = glob.glob(cfg['swe_est_dir'] + '*.diff.adf.swe')
        ifg_swe_estimates = get_itab_diffs(files, itab, RSLC_tab)

    swe_estimates = []

    for file in ifg_swe_estimates:
        path, root = os.path.split(file)
        root = root.split('.')[0]

        m_str, s_str = parse_ifg_dates(root)
        m_date = pd.to_datetime(m_str)
        s_date = pd.to_datetime(s_str)

        rswe = get_relative_swe(m_str, s_str, show_results=0)

        swe_im = readBin(file, est_dim, 'float32')
        cphi_im = readBin(cfg['swe_est_dir'] + root + '.diff.adf.patch', est_dim, 'complex64')
        cphi_im[exclude_mask] = 0
        cphi_masked = cphi_im[~exclude_mask]

        phi_demean = np.angle(cphi_masked * np.conj(np.nanmean(cphi_masked)))
        phi_demean -= np.nanmean(phi_demean)

        #cc = phase_coh(cphi_im, win1000)
        #cc = np.abs(np.mean(cphi_im))/np.mean((~exclude_mask).astype(float))
        cc = readBin(file + '.coh', est_dim, 'float32')

        if np.nanmean(cc) < 0.2:
            print("DROPPING ", file)
            continue

        swe_est = swe_im
        swe_est_refined = swe_im

        #swe_weighted = np.average(swe_est[~exclude_mask], weights=cc[~exclude_mask])
        swe_weighted = np.nanmean(swe_im)


        #swe_stats = [np.nanmean(swe_est), np.nanstd(swe_est), np.nanmin(swe_est), np.nanmax(swe_est)]
        swe_stats = [swe_weighted, np.nanstd(swe_est), np.nanmin(swe_est), np.nanmax(swe_est)]

        swe_estimates.append((m_date, s_date, np.nanmean(cc), swe_stats, rswe['ERA5-Reanalysis'],
                              rswe['site_1'], rswe['site_2'], rswe['site_3'], rswe['site_4'], rswe['site_5'], rswe['site_6'],
                              rswe['EC-Inuvik'], rswe['EC-TrailValley']))

        if plt_patch:
            phi_full = readBin(ifg_dir + root + cfg['diff_ext'], dim, 'complex64')
            xi_full = xi = readBin(dem_dir + 'topo/dphi_dswe.rdc', dim, 'float32')
            exclude_mask_full = cfg['watermask']

            swes, coh = swe_est_clicky_func(pt_xy, phi_full, xi_full, exclude_mask_full, plots_for_paper=False)

            plt.figure(figsize=[16, 16])
            plt.subplot(321)
            plt.title(f"Phi - {root}")
            plt.imshow(np.angle(cphi_im).T, vmin=-np.pi, vmax=np.pi, cmap='hsv', alpha=0.5)
            plt.colorbar()
            plt.subplot(322)
            plt.xlabel("Sensitivity - Xi")
            #plt.imshow(xi_im.T)
            plt.hist(xi_im[~exclude_mask].flatten(), bins=100)
            plt.subplot(323)
            plt.title(f"SWE Estimate {np.nanmean(swe_im):.3f}")
            plt.imshow(swe_im.T, vmin=cfg['swe_range'][0], vmax=cfg['swe_range'][1], cmap='RdBu')
            plt.colorbar()
            plt.subplot(324)
            plt.plot(xpts / 1000., xpts * swe_weighted, color='tab:red',
                     label=r'$\hat s_w$ = {:02.0f} mm'.format(swe_weighted * 1000.))
            plt.hist2d(xi_demean / 1000., phi_demean, bins=100, cmap='Greys')
            plt.xlim([np.min(xi_demean) / 1000, np.max(xi_demean) / 1000])
            plt.ylim((-np.pi, np.pi))
            plt.xlabel('Zero mean dry-snow ' r'phase sensitivity, $\~\xi$ [radians/mm]')
            plt.ylabel(r'Centered phase, $\~\Phi$ [radians]')
            plt.legend()

            plt.subplot(325)
            plt.title("Coherance")
            plt.imshow(cc.T, vmin=0, vmax=1, cmap='Greys_r')
            plt.colorbar()

            plt.subplot(326)
            plt.plot(swes, coh, 'k')
            plt.plot([swe_weighted * 1000., swe_weighted * 1000.], [0., np.nanmean(cc) * 1.1], '--',
                     color='tab:red')
            plt.text((swe_weighted + 0.005) * 1000., 0.0,
                     r'$\hat s_w$ = {:02.0f} mm'.format(swe_weighted * 1000.), horizontalalignment='left',
                     verticalalignment='bottom', size=8)
            plt.xlabel(r'Correcting $\Delta$SWE [mm]')
            plt.ylabel('Spectral magnitude')

            plt.show()
            # plt.savefig('/local-scratch/users/aplourde/snow/estimates/' + m_str + '_' + s_str +'.png')
            # plt.close()
            # break

    swe_est_df = pd.DataFrame(swe_estimates,
                              columns=['master', 'slave', 'mean_coh', 'swe_est', 'swe_era5', 'swe_site_1', 'swe_site_2',
                                       'swe_site_3', 'swe_site_4', 'swe_site_5', 'swe_site_6', 'swe_inuvik', 'swe_trailValley'])
    # swe_est_df['hds_era5_error'] = swe_est_df['swe_hds'].str[0] - swe_est_df['swe_era5']
    # swe_est_df['hds_site_1_error'] = swe_est_df['swe_hds'].str[0] - swe_est_df['swe_site_1']

    res = swe_est_df

    if show_results:
        scatter_swe(swe_est_df)
        plt.figure()
        timeseries_swe(res)
        plt.clf()

        ### Histogram ###
        """
        plt.hist(swe_est_df[parameter2] - swe_est_df['swe_est'].str[0])
        plt.xlabel("SWE Error")
        plt.ylabel("Num Samples")
        plt.show()
        """
    return res


def scatter_swe(swe_est_df):
        ### Correlation ###
        parameter1 = 'swe_est'
        #parameter2 = 'swe_inuvik'
        parameter2 = 'swe_site_1_insitu'

        swe_est_df = swe_est_df.dropna(subset=[parameter2])

        plt.figure(figsize=(6,6))
        #plt.subplot(211)


        plt.fill_between([0, 0, 100], [0, 100, 100], facecolor='green', alpha=0.1)
        plt.fill_between([0, 0, 100], [0, -100, -100], facecolor='red', alpha=0.1)
        plt.fill_between([0, 0, -100], [0, -100, -100], facecolor='green', alpha=0.1)
        plt.fill_between([0, 0, -100], [0, 100, 100], facecolor='red', alpha=0.1)
        plt.plot([-100, 100], [-100, 100], color="black")

        #plt.scatter(swe_est_df[parameter2], swe_est_df['swe_est'].str[0], c=swe_est_df['mean_coh'], cmap='RdYlGn', vmin=0, vmax=1)
        plt.scatter(swe_est_df[parameter2]*M_TO_MM, swe_est_df['swe_est']*M_TO_MM, c=swe_est_df['mean_coh'], cmap='RdYlGn', vmin=0, vmax=1)

        clb = plt.colorbar(label='Coherence')

        norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='RdYlGn')
        colors = np.array([(mapper.to_rgba(c)) for c in swe_est_df['mean_coh']])

        #for (x, y, e, c) in zip(swe_est_df[parameter2], swe_est_df['swe_est'].str[0], swe_est_df['swe_est'].str[1], colors):
        #    plt.errorbar(x, y, e, lw=1, capsize=3, color=c)
        for (x, y, e, c) in zip(swe_est_df[parameter2]*M_TO_MM, swe_est_df['swe_est']*M_TO_MM, swe_est_df['swe_est_std']*M_TO_MM, colors):
            plt.errorbar(x, y, e, lw=1, capsize=3, color=c)

        # plt.scatter(swe_est_df['swe_est'].str[0], swe_est_df['swe_inuvik'], label='HDS vs Inuvik')
        # plt.scatter(swe_est_df['swe_est'].str[0], swe_est_df['swe_trailValley'], label='HDS vs trailValley')
        # plt.scatter(swe_est_df['swe_est'].str[0], swe_est_df['swe_era5'], label='HDS vs era5')
        # plt.scatter(swe_est_df['swe_est'].str[0], swe_est_df['swe_site_1'], label='HDS vs Site1')
        # plt.legend()

        #plt.xlabel(f"{parameter2}")
        #plt.title(f"{parameter2} vs {parameter1}")
        # plt.ylabel(f"{parameter1}")
        plt.title("Snow Water Equivalent (SWE)\n(mm)")
        #plt.xlabel("Inuvik")
        plt.xlabel("Site 1 In-situ")
        plt.ylabel("Site 1 InSAR")
        #plt.ylabel("Region with High Variability in $\\xi$")


        # annotations = []
        # for idx, row in swe_est_df.iterrows():
        #    annotations.append(plt.text(row['swe_est'].str[0], row[parameter2], row['master'].strftime('%Y%m%d'), fontsize=8))#, arrowprops=dict(arrowstyle='-',color='red')))
        # adjust_text(annotations, swe_est_df['swe_est'].str[0].copy(), swe_est_df[parameter2].copy(), expand_text=(1, 1), expand_objects=(2,2), expand_points=(2,2), arrowprops=dict(arrowstyle='-',color='black', alpha=0.5, shrinkA=0.05))#, time_lim=1)#, force_static=(2,2), expand=(1.05, 10), , arrowprops=dict(facecolor='black', arrowstyle='->'))

        plt.show()


def get_swe_stack(pnts):

    dim = master_par.dim
    r_grid, az_grid = np.mgrid[0:dim[0], 0:dim[1]]

    files = glob.glob(cfg['swe_est_dir'] + '*.diff.adf.swe')
    itab_files = get_itab_diffs(files, itab, RSLC_tab)

    if os.path.exists(f'/local-scratch/users/aplourde/cache/swe_est_df_{os.path.basename(itab)}_water_mask.pkl'):
        with open(f'/local-scratch/users/aplourde/cache/swe_est_df_{os.path.basename(itab)}_water_mask.pkl', 'rb') as file:
            swe_est_df = pickle.load(file)
    else:
        swe_est_df = pd.DataFrame()

        for file in itab_files:
            swe_im = readBin(file, dim, 'float32')
            cc = readBin(file + '.coh', dim, 'float32')
            nan_mask = np.isnan(swe_im)

            if os.path.exists(file + '.int'):
                print(file)
                swe_int = readBin(file + '.int', dim, 'float32')
            else:
                r_interp = r_grid[~nan_mask]
                az_interp = az_grid[~nan_mask]
                im_interp = swe_im[~nan_mask]
                swe_int = griddata((az_interp, r_interp), im_interp, (az_grid, r_grid))
                swe_int = swe_int.reshape(swe_im.shape)

                writeBin(file + '.int', swe_int)

            #plt.imshow(swe_int.T)
            #plt.scatter(pt_xy[0], pt_xy[1])
            #plt.title(f"{swe_int[int(pt_xy[0])][int(pt_xy[1])]}")
            #plt.show()

            path, root = os.path.split(file)
            root = root.split('.')[0]

            m_str, s_str = parse_ifg_dates(root)
            m_date = pd.to_datetime(m_str)
            s_date = pd.to_datetime(s_str)

            rswe = get_relative_swe(m_str, s_str, show_results=0)

            swe_estimates = pd.Series({'master': m_date,
                                       'slave': s_date,
                                       'swe_era5': rswe['ERA5-Reanalysis'],
                                       'swe_inuvik': rswe['EC-Inuvik'],
                                       'swe_trailValley': rswe['EC-TrailValley'],
                                       'swe_site_1_insitu': rswe['site_1'],
                                       'swe_site_2_insitu': rswe['site_2'],
                                       'swe_site_3_insitu': rswe['site_3'],
                                       'swe_site_4_insitu': rswe['site_4'],
                                       'swe_site_5_insitu': rswe['site_5'],
                                       'swe_site_6_insitu': rswe['site_6']
                                       })

            pt_swe = {}
            pt_coh = {}
            for i, row in pnts.iterrows():
                pt_x = int(row['x_loc'])
                pt_y = int(row['y_loc'])
                label = row['label']

                swe_int_patch, swe_int_mask = extract_annulus(swe_int, [pt_x, pt_y], 10, 100, show_plot=0)
                cc_patch, cc_mask = extract_annulus(cc, [pt_x, pt_y], 10, 100, show_plot=0)

                #swe_estimates['swe_'+label+'_insar'] = swe_int[pt_x]
                swe_estimates['swe_' + label + '_insar'] = np.nanmean(swe_int_patch[swe_int_mask])
                swe_estimates['swe_'+label+'_coh'] = np.nanmean(cc_patch[swe_int_mask])
                swe_estimates['swe_' + label + '_std'] = np.nanstd(swe_int_patch[swe_int_mask])

            swe_est_df = pd.concat([swe_est_df, swe_estimates.to_frame().T], ignore_index=True)
        with open(f'/local-scratch/users/aplourde/cache/swe_est_df_{os.path.basename(itab)}_water_mask.pkl', 'wb') as file:
            print("dumping file...")
            pickle.dump(swe_est_df, file)
    #print(swe_est_df)
    #return

    tabulate_results(swe_est_df)

    swe_est_df['swe_est'] = swe_est_df['swe_site_1_insar']
    swe_est_df['mean_coh'] = swe_est_df['swe_site_1_coh']
    swe_est_df['swe_est_std'] = swe_est_df['swe_site_1_std']

    scatter_swe(swe_est_df)

    #plt.figure()
    #timeseries_swe(swe_est_df)
    #plt.clf()


def timeseries_swe(df):
    print(df)
    df['delta_swe_est'] = 0
    df['delta_swe_inuvik'] = 0
    df['delta_swe_trailValley'] = 0
    df['delta_swe_era5'] = 0
    for idx, row in df.iterrows():
        if idx == 0:
            continue

        #print(row['master'], df['slave'].iloc[idx-1])
        if row['master'] == df['slave'].iloc[idx-1]:
            #df.loc[idx, 'delta_swe_est'] = row['swe_est'][0] + df['delta_swe_est'].iloc[idx-1]
            df.loc[idx, 'delta_swe_est'] = row['swe_est'] + df['delta_swe_est'].iloc[idx - 1]
            df.loc[idx, 'delta_swe_inuvik'] = row['swe_inuvik'] + df['delta_swe_inuvik'].iloc[idx - 1]
            df.loc[idx, 'delta_swe_trailValley'] = row['swe_trailValley'] + df['delta_swe_trailValley'].iloc[idx - 1]
            df.loc[idx, 'delta_swe_era5'] = row['swe_era5'] + df['delta_swe_era5'].iloc[idx - 1]

    df['delta_sde_est'] = df['delta_swe_est'] / snow_density * M_TO_CM
    df['delta_sde_inuvik'] = df['delta_swe_inuvik'] / snow_density * M_TO_CM
    df['delta_sde_trailValley'] = df['delta_swe_trailValley'] / snow_density * M_TO_CM
    df['delta_sde_era5'] = df['delta_swe_era5'] / snow_density * M_TO_CM

    df['slave'] = pd.to_datetime(df['slave'])
    dates = df['slave'].dt.to_pydatetime()
    plt.plot(dates, df['delta_sde_inuvik'], '.-', label="Inuvik", color='black')
    plt.plot(dates, df['delta_sde_trailValley'], '.-', label="trailValley", color='grey')
    plt.plot(dates, df['delta_sde_era5'], '.-', label="era5", color='lightgrey')

    plt.plot(dates, df['delta_sde_est'], label="Estimated SWE", color='blue')
    plt.scatter(dates, df['delta_sde_est'], c=df['mean_coh'], cmap='RdYlGn', vmin=0, vmax=1)
    import matplotlib.dates as mdates
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.legend()
    plt.ylabel("Snow Depth (cm)")
    plt.show()


def tabulate_results(df):
    print(df)
    df.master = pd.to_datetime(df.master)
    df.slave = pd.to_datetime(df.slave)
    seasons = [['20171001', '20180331'],
               ['20181001', '20190331'],
               ['20191001', '20200331'],
               ['20201001', '20210331'],
               ['20211001', '20220331'],
               ['20221001', '20230331']]

    table_cols = [f"{s[0][:-4]}-{s[1][:-4]}" for s in seasons]
    seasonal_table = pd.DataFrame(columns=table_cols)

    for season in seasons:
        start, end = pd.to_datetime(season)
        scol = f"{start.year}-{end.year}"

        df_temp = pd.DataFrame()
        df_temp = df[df.master >= start]
        df_temp = df_temp[df_temp.slave <= end]

        cols = list(df_temp)

        df_temp['num_pairs'] = 1

        for col in cols:
            if 'swe' in col and 'coh' not in col:
                new_col = col.replace('swe', 'sde')
                df_temp[new_col] = df_temp[col] / snow_density * M_TO_CM
                df_temp.drop(columns=[col], inplace=True)
            else:
                df_temp.drop(columns=[col], inplace=True)

        #print(df_temp.sum())
        seasonal_table[scol] = df_temp.sum()

    print(seasonal_table)

    error_table = pd.DataFrame()



def phi_due_to_snow_and_heave(sensor_cfg):

    depth = np.linspace(0, 0.1, 100)  # 1 mm to 100 cm
    depth = np.array(depth)

    lamb = c / sensor_cfg['frequency']

    phi_snow = snow_depth_to_phase(depth, snow_density, lamb, sensor_cfg['incidence_angle'])
    phi_snow = np.array(phi_snow)
    phi_swe = phi_snow/snow_density
    cphi_heave, phi_heave = deformation_to_phase(depth, sensor_cfg['frequency'], sensor_cfg['incidence_angle'])
    phi_heave = np.array(phi_heave)

    def closest_index(arr, target):
        absolute_diff = np.abs(arr - target)
        return np.argmin(absolute_diff)
    phi_snow_wrap = closest_index(phi_snow, np.pi)
    phi_swe_wrap = closest_index(phi_swe, np.pi)
    phi_heave_wrap = closest_index(phi_heave, np.pi)

    plt.plot(depth*100, phi_snow, color='blue', label='Phase due to snow depth')
    plt.plot(depth * 100, phi_swe, color='green', label='Phase due to SWE')
    plt.plot(depth*100, phi_heave, color='red', label='Phase due to heave')
    plt.plot([0, np.max(depth*100)], [np.pi, np.pi], 'k--', label='pi')
    plt.scatter(depth[phi_snow_wrap]*100, phi_snow[phi_snow_wrap], color='blue')
    plt.scatter(depth[phi_swe_wrap] * 100, phi_swe[phi_swe_wrap], color='green')
    plt.scatter(depth[phi_heave_wrap] * 100, phi_heave[phi_heave_wrap], color='red')
    plt.annotate(f"{depth[phi_snow_wrap] * 100:.2f}", (depth[phi_snow_wrap] * 100, phi_snow[phi_snow_wrap]+1))
    plt.annotate(f"{depth[phi_swe_wrap] * 100:.2f}", (depth[phi_swe_wrap] * 100 + 0.5, phi_swe[phi_swe_wrap] + 0.5))
    plt.annotate(f"{depth[phi_heave_wrap] * 100:.2f}", (depth[phi_heave_wrap] * 100 - 1, phi_heave[phi_heave_wrap] + 1))
    plt.legend()
    plt.title(f"Radar Frequency: {sensor_cfg['frequency']/1e9:.2f} GHz\nInclination Angle: {np.rad2deg(sensor_cfg['incidence_angle']):.2f}")
    plt.xlabel("Height (cm)")
    plt.ylabel("Phase (rad)")
    plt.ylim((-1,20))
    plt.show()

    snow_wrap = phase_to_snow_depth(np.pi, snow_density, lamb, sensor_cfg['incidence_angle'])
    heave_wrap = phase_to_deformation(np.pi, sensor_cfg['frequency'], sensor_cfg['incidence_angle'])
    print(f"Snow phase wrap: {snow_wrap}")
    print(f"Heave phase wrap {heave_wrap}")

    phi_snow = snow_depth_to_phase(0.1, snow_density, lamb, sensor_cfg['incidence_angle'])
    cphi_heave, phi_heave = deformation_to_phase(0.04, sensor_cfg['frequency'], sensor_cfg['incidence_angle'])
    print(f"Phase due to 10cm snow accumulation: {phi_snow}")
    print(f"PHase due to 4cm of vertical heave: {phi_heave}")
    print(f"Combined phase: {phi_snow + phi_heave}")
    snow = phase_to_snow_depth(phi_snow + phi_heave, snow_density, lamb, sensor_cfg['incidence_angle'])
    print(f"Snow overestimate: {snow}")


def li_and_pomeroy(U, T, dt_hours):
    """ Calculate P:
    U: wind speed vector
    T: air temperature
    dt_hours: total time
    """
    #from Li and Pomeroy, 1997
    #I = np.log(np.arange(6, 148, 6))
    I = np.log(dt_hours) # snow age index (time since snow deposition)
    U_bar = 0.365*T + 0.00706*(T**2) + 0.9*I +11.2 # average wind speed as a function of air temperature and snow age index
    delta = 0.145*T + 0.00196*(T**2) + 4.3 # variance of the wind speed
    z = (U-U_bar)/delta
    if np.isscalar(z):
        P = 0.5+0.5*special.erf(z/(2**0.5))
    else:
        P = [0.5+0.5*special.erf(z_elem/(2**0.5)) for z_elem in z]
    return P


def blown_snow_index2(dict):
    ns = len(dict['2t'])
    bsi = np.zeros(ns)
    for ii in np.arange(1, ns):
        bsi[ii] = li_and_pomeroy(dict['10m'][ii], dict['2t'][ii], 1)
        if dict['sd'][ii] < 0.001:
            bsi[ii] = 0.0
        if dict['10m'][ii]<3.0:
            bsi[ii] = 0.0 #recommended in subsequent paper by authors
    return bsi


def get_era5_pairwise_data(dict, acq_dates, start_date, end_date, acq_fractional_day=0, plot=True):
    len_dict = len(dict['times'])
    start_time = datetime.strptime(start_date, '%Y%m%d')
    acq_times = [datetime.strptime(date, '%Y%m%d')+timedelta(acq_fractional_day) for date in acq_dates]
    acq_idx = [np.rint((acq_time - start_time).days).astype(np.int) for acq_time in acq_times]
    nd = len(acq_dates)
    max_sd = np.zeros((nd, nd)) + np.inf
    min_sd = np.zeros((nd, nd))+ np.inf
    max_2t = np.zeros((nd, nd))+ np.inf
    delta_sd = np.zeros((nd, nd))+ np.inf
    cum_sm = np.zeros((nd,nd))+ np.inf
    cum_bsi = np.zeros((nd,nd))+ np.inf
    cum_bsi_vec = np.zeros((nd, nd), dtype=complex) + np.inf
    snow_seasons = np.zeros((nd, nd))+ np.inf
    mask = np.zeros((nd,nd))
    for ii in range(nd):
        for jj in range(ii,nd):
            if (acq_idx[ii] < len_dict) and  (acq_idx[jj] < len_dict) and acq_idx[ii]>=0 and acq_idx[jj] >=0:
                max_sd[ii, jj] = max(dict['sd'][acq_idx[ii]], dict['sd'][acq_idx[jj]])
                max_sd[jj, ii] = max_sd[ii, jj]
                min_sd[ii, jj] = min(dict['sd'][acq_idx[ii]], dict['sd'][acq_idx[jj]])
                min_sd[jj, ii] = min_sd[ii, jj]
                max_2t[ii, jj] = np.max(dict['2t'][acq_idx[ii]:acq_idx[jj]+1])
                max_2t[jj, ii] = max_2t[ii, jj]
                delta_sd[ii, jj] =dict['sd'][acq_idx[jj]] - dict['sd'][acq_idx[ii]]

                delta_sd[jj, ii] = -1*delta_sd[ii, jj]
                #cum_sm[ii, jj] = np.sum(dict['smlt'][acq_idx[ii]:acq_idx[jj]+1])
                cum_sm[jj, ii] = -1*cum_sm[ii, jj]
                cum_bsi[ii, jj] = np.sum(dict['bsi'][acq_idx[ii]:acq_idx[jj]+1])
                cum_bsi[jj, ii] = 0#-1* cum_bsi[ii, jj]
                cum_bsi_vec[ii, jj] = np.sum(dict['bsi'][acq_idx[ii]:acq_idx[jj]+1]*np.exp(1j*dict['10d'][acq_idx[ii]:acq_idx[jj]+1]/180*np.pi))
                cum_bsi_vec[jj, ii] = 0

                snow_seasons[ii, jj] = np.prod(dict['seasonal_snow_mask'][acq_idx[ii]:acq_idx[jj]+1])
                snow_seasons[jj, ii] = snow_seasons[ii, jj]
                mask[ii, jj] = 1
            #else: uses data not present in dict results so just skip
            if ii == jj:
                mask[ii, jj] = 0
    # cum_bsi = cum_bsi*snow_seasons
    # delta_sd = delta_sd*snow_seasons
    # max_2t *=snow_seasons

    #snow_ifg_mask = snow_seasons*(min_sd > 0.001)*(cum_sm<0.001)*mask
    snow_ifg_mask = (min_sd > 0.001) * (cum_sm < 0.001) * mask
    # within_summer_ifg_mask = (max_sd < 0.001) * (cum_sm < 0.001) * mask
    # across_summer_ifg_mask = (max_sd < 0.001) * (cum_sm > 0.001) * mask

    era5_pairwise = {}
    era5_pairwise['mask'] = mask
    era5_pairwise['max_sd'] = max_sd
    era5_pairwise['min_sd'] = min_sd
    era5_pairwise['max_2t'] = max_2t
    era5_pairwise['delta_sd'] = delta_sd
    era5_pairwise['cum_sm'] = cum_sm
    era5_pairwise['cum_bsi'] = cum_bsi
    era5_pairwise['cum_bsi_vec_mag'] = np.abs(cum_bsi_vec)
    era5_pairwise['cum_bsi_vec_dir'] = np.angle(cum_bsi_vec)*180/np.pi
    era5_pairwise['snow_seasons'] = snow_seasons

    #network selection keys
    era5_pairwise['snow_free'] = max_sd < 0.001
    era5_pairwise['snow_free'][mask == 0] = False
    era5_pairwise['dry_snow_intra'] = np.logical_and(min_sd > 0.001, cum_sm < 0.001)
    era5_pairwise['dry_snow_intra'][mask == 0] = False
    # era5_pairwise['ifg_types'] = snow_ifg_mask*1 + within_summer_ifg_mask*2 + across_summer_ifg_mask*3

    if plot:
        #all
        plt.figure()
        for ii, key in enumerate(era5_pairwise):
            masked_array = np.ma.array(era5_pairwise[key], mask=(era5_pairwise['mask']==0))
            cmap = cm.jet
            cmap.set_bad('black', 1.)
            plt.subplot(3, len(era5_pairwise)//3+1, ii+1)
            plt.imshow(masked_array.T, interpolation='nearest', cmap=cmap)
            plt.title(key)
            #plt.colorbar()


        plot_list = ['cum_sm', 'delta_sd', 'snow_seasons', 'min_sd', 'cum_bsi', 'cum_bsi_vec_mag', 'cum_bsi_vec_dir', 'snow_free', 'dry_snow_intra']
        plt.figure()
        for ii, key in enumerate(plot_list):
            masked_array = np.ma.array(era5_pairwise[key], mask=(era5_pairwise['mask']==0))
            cmap = cm.jet
            cmap.set_bad('black', 1.)
            plt.subplot(2, len(plot_list)//2+1, ii+1)
            plt.imshow(masked_array.T, interpolation='nearest', cmap=cmap)
            plt.title(key)
            #plt.colorbar()

        #ones for Fringe 2017 talk with ifg selection mask
        plt.figure()
        for ii, key in enumerate(plot_list):
            masked_array = np.ma.array(era5_pairwise[key], mask=(snow_ifg_mask==0))
            cmap = cm.jet
            cmap.set_bad('black', 1.)
            plt.subplot(2, len(plot_list)//2+1, ii+1)
            plt.imshow(masked_array.T, interpolation='nearest', cmap=cmap)
            plt.title(key)
            #plt.colorbar()

    return era5_pairwise


def blown_snow_analysis(dates):

    heading = -163.0993187   #degrees
    #fig_width = 8

    start_date = '20180101'
    end_date = '20221231'
    #start_date = '20120101'
    #end_date = '20170601'

    fractional_day = 0.63
    #era5_data = get_inuvik_RS2_SLA27D_era5()
    era5_data = pd.read_csv(f'{era5_met_dir}U76_scene_center_snow_series_{start_date}-{end_date}.csv')
    #dates = ['20210107', '20210114']
    """dates = ['20120105', '20120129', '20120222', '20120317', '20120410', '20120504', '20120528', '20120621', '20120715',
             '20120808',
             '20120901', '20120925', '20121019', '20121112', '20121206', '20121230', '20130123', '20130216',
             '20130312', '20130405', '20130429', '20130523', '20130616', '20130710', '20130803', '20130827',
             '20130920', '20131014', '20131107', '20131225', '20140118', '20140424', '20140518', '20140611',
             '20140705', '20140729', '20140822', '20140915', '20141009', '20141102', '20141126', '20141220',
             '20150113', '20150206', '20150302', '20150513', '20150606', '20150724', '20150817', '20150910',
             '20151004', '20151028', '20151121', '20151215', '20160108', '20160225', '20160320', '20160624',
             '20160718', '20160811', '20160904', '20160928', '20161022', '20161115', '20161209', '20170102',
             '20170126', '20170219', '20170315', '20170408', '20170502', '20170526', '20170619', '20170713',
             '20170806', '20170830', '20170923', '20171017', '20171204', '20171228', '20180214', '20180310',
             '20180403', '20180427', '20180521', '20180614', '20180708', '20180801', '20180825', '20180918',
             '20181012', '20181105', '20181129', '20181223', '20190116', '20190209', '20190305', '20190329',
             '20190422', '20190516', '20190609', '20190703', '20190727', '20190913', '20191007', '20191124',
             '20200111', '20200204', '20200228', '20200323', '20200416']"""


    #replace bsi with version2
    #era5_data['bsi_old'] = era5_data['bsi']
    era5_data['bsi'] = blown_snow_index2(era5_data)*24

    dts = [datetime.strptime(date, '%Y%m%d') + timedelta(fractional_day) for date in dates]

    era5_data_pairwise = get_era5_pairwise_data(era5_data, dates, start_date, end_date, fractional_day, plot=False)
    # plt.figure()
    # plt.plot(dts[:-1], np.diagonal(era5_data_pairwise['cum_bsi'], offset=1))
    # plt.plot(dts[:-1], np.diagonal(era5_data_pairwise['cum_bsi_vec_mag'], offset=1))

    #plt.figure(figsize=(fig_width, 3))
    print(np.diagonal(era5_data_pairwise['cum_bsi_vec_mag'], offset=1)[0])
    print(np.nanmean(np.diagonal(era5_data_pairwise['cum_bsi_vec_mag'], offset=1)[1:]))
    #plt.polar(np.diagonal(era5_data_pairwise['cum_bsi_vec_dir'], offset=1)[0] / 180 * np.pi,
    #          np.diagonal(era5_data_pairwise['cum_bsi_vec_mag'], offset=1)[0], '*', label='20120105_20120129')
    plt.polar(np.diagonal(era5_data_pairwise['cum_bsi_vec_dir'], offset=1)[1:]/180*np.pi,
              np.diagonal(era5_data_pairwise['cum_bsi_vec_mag'], offset=1)[1:], '*', label='BSI at 24-day intervals')
    plt.gca().set_theta_direction(-1)
    plt.gca().set_theta_zero_location('N')
    los_heading = np.exp(1j*np.deg2rad(heading) + np.pi/2)
    plt.polar((los_heading, los_heading+np.pi), (60, 60), '--',label = 'SAR look direction axis')
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.show()
    #plt.savefig(fig_dir + 'bsi.png')


def make_terrain_maps(looks_str='1_1'):
    #stack = cfg['stack']
    #looks_str = '2_3'

    #create terrain related maps from source DEM
    rho = 0.3 #normalized snow density

    topo_dir = '/local-scratch/users/aplourde/HDS/projects/southern_ITH/dems/tdx/'
    init_fn = topo_dir + 'topo_indices_new/dem_cropped_new_gauss3.tif'
    utm_fn = topo_dir + 'topo_indices_new/dem_cropped_utm.tif'
    #init_fn = dem_dir + 'topo/dem_cropped.tif'
    #utm_fn = dem_dir + 'topo/dem_cropped_utm.tif'
    slope_utm_fn = topo_dir + 'topo_indices_new/slope_utm.tif'
    aspect_utm_fn = topo_dir + 'topo_indices_new/aspect_utm.tif'
    aspect_n_utm_fn = topo_dir + 'topo_indices_new/aspect_n_utm.tif'
    aspect_e_utm_fn = topo_dir + 'topo_indices_new/aspect_e_utm.tif'
    tpi_utm_fn = topo_dir + 'topo_indices_new/tpi_raw.tif'
    basins_utm_fn = topo_dir + 'topo_indices_new/all_basins.tif'

    rmli_dir = sar_dir + 'rmli_' + looks_str + '/'
    out_dir = dem_dir + 'topo/'
    par = MLI_Par(sar_dir + sub_dir + 'rmli_'+looks_str+ '/rmli_'+looks_str+'.ave.par')
    dem_par = DEM_Par(dem_dir+'seg.dem_par')

    gc_map = dem_dir+'gc_fine'
    dem_fn = dem_dir+'seg.dem'
    lamb = c / frequency

    """
    print('Converting basins map to rdc.')
    #par.geotiff2rdc(basins_utm_fn, out_dir + 'basins_all.rdc', dem_par, gc_map, interp_mode=1)

    print('Converting source DEM to UTM.')
    warp_to_utm_args = ['gdalwarp', '--config', 'GDAL_CACHEMAX', '500', '-wm', '500', '-t_srs','EPSG:32608','-r', 'cubicspline', '-of', 'GTiff', '-overwrite'] + [init_fn] + [utm_fn]
    result = _exec(warp_to_utm_args)


    print('Making slope map.')
    slope_args = ['gdaldem', 'slope', '-of', 'GTiff', '-b', '1', '-s', '1.0'] + [utm_fn] + [slope_utm_fn]
    _exec(slope_args)
    par.geotiff2rdc(slope_utm_fn, out_dir + 'slope_init.rdc', dem_par, gc_map)
    return
    """
    slope = np.deg2rad(readBin(out_dir+'slope_init.rdc', par.dim, 'float32'))

    #set water areas to have zero slope
    water_mask = cfg['watermask']
    #print(np.sum(water_mask))
    print(water_mask.shape, slope.shape)
    slope[water_mask] = 0.

    #fill DEM blunder areas
    #blunders = read_ras(dem_dir + 'blunders_'+looks_str+'.ras')[0].T == 0
    #blunders = morph.binary_dilation(blunders, iterations=5)

    #slope[blunders] = np.nan
    slope = nan_fill(slope)
    """
    writeBin(out_dir+'slope.rdc', np.rad2deg(slope))

    print('Making aspect map.')
    aspect_args = ['gdaldem', 'aspect', '-of', 'GTiff', '-b', '1', '-alg', 'ZevenbergenThorne'] + [utm_fn] + [aspect_utm_fn]
    result = _exec(aspect_args)
    #read in GDAL computed aspect angle and convert to north and east basis components (required for warping)
    aspect = get_band(aspect_utm_fn)
    aspect_n = np.cos(np.deg2rad(aspect))
    aspect_e = np.sin(np.deg2rad(aspect))
    in_raster = gdal.Open(aspect_utm_fn)
    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(aspect_e_utm_fn, aspect.shape[1], aspect.shape[0], 1, gdal.GDT_Float32)
    out_raster.SetGeoTransform(in_raster.GetGeoTransform())
    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(aspect_e)
    out_raster.SetProjection(in_raster.GetProjection())
    out_band.FlushCache()
    out_raster = None
    out_band = None
    in_raster = gdal.Open(aspect_utm_fn)
    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(aspect_n_utm_fn, aspect.shape[1], aspect.shape[0], 1, gdal.GDT_Float32)
    out_raster.SetGeoTransform(in_raster.GetGeoTransform())
    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(aspect_n)
    out_raster.SetProjection(in_raster.GetProjection())
    out_band.FlushCache()
    out_raster = None
    out_band = None
    #par.geotiff2rdc(aspect_n_utm_fn, out_dir+'aspect_n.rdc', dem_par, gc_map)
    #return
    #par.geotiff2rdc(aspect_e_utm_fn, out_dir+'aspect_e.rdc', dem_par, gc_map)
    #return
    """
    aspect_n = readBin(out_dir + 'aspect_n.rdc', par.dim, 'float32')
    aspect_e = readBin(out_dir + 'aspect_e.rdc', par.dim, 'float32')

    #fill blunders in aspect components
    #aspect_n[blunders] = np.nan
    aspect_n = nan_fill(aspect_n)
    #aspect_e[blunders] = np.nan
    aspect_e = nan_fill(aspect_e)

    #compute and write aspect
    aspect = np.angle(aspect_n+1j*aspect_e)
    aspect[water_mask] = 0.
    writeBin(out_dir + 'aspect.rdc', aspect)
    #"""
    aspect = readBin(out_dir+'aspect.rdc', par.dim, 'float32')
    """
    print('Computing SAR look vectors.')
    print('look_vector', par, None, dem_par, dem_fn, dem_dir+'lv_theta', dem_dir+'lv_phi')
    run('look_vector', par, None, dem_par, dem_fn, dem_dir+'topo/lv_theta', dem_dir+'topo/lv_phi')
    #print('geocode', gc_map, dem_dir + 'lv_theta', dem_par.dim[0], dem_dir+'lv_theta.rdc', par.dim[0], par.dim[1], 0, 0, '-', '-', 2, 32, '-')
    #print('geocode', gc_map, dem_dir + 'lv_phi', dem_par.dim[0], dem_dir + 'lv_phi.rdc', par.dim[0], par.dim[1], 0, 0, '-', '-', 2, 32, '-')
    #return
    #run('geocode', gc_map, dem_dir + 'lv_theta', dem_par.dim[0], dem_dir+'lv_theta.rdc', par.dim[0], par.dim[1], 0, 0, None, None, 2, 32, None)
    #run('geocode', gc_map, dem_dir + 'lv_phi', dem_par.dim[0], dem_dir + 'lv_phi.rdc', par.dim[0], par.dim[1], 0, 0, None, None, 2, 32, None)
    """
    lv_theta = readBin(dem_dir+'topo/lv_theta.rdc', par.dim, 'float32')
    lv_phi = readBin(dem_dir+'topo/lv_phi.rdc', par.dim, 'float32')

    print('Computing local incidence angle.')
    # #compute terrain slope normal vector in local ENU system
    slope_cos = np.cos(slope)
    slope_sin = np.sin(slope)
    aspect_cos = np.cos(aspect)
    aspect_sin = np.sin(aspect)
    t_u = slope_cos
    t_n = slope_sin*aspect_cos
    t_e = slope_sin*aspect_sin
    # #compute look vector in local ENU system
    lv_theta_cos = np.cos(lv_theta)
    l_u = np.sin(lv_theta)
    l_n = lv_theta_cos*np.sin(lv_phi)
    l_e = lv_theta_cos*np.cos(lv_phi)

    # #compute incidence angle
    inc = np.arccos(l_u*t_u + l_n*t_n + l_e*t_e)
    writeBin(out_dir + 'inc.rdc', inc)

    inc = readBin(out_dir + 'inc.rdc', par.dim, 'float32')

    print('Computing heave sensitivity.')
    _dphi_dheave = dphi_dheave(inc, lamb/2)
    writeBin(out_dir + 'dphi_dheave.rdc', _dphi_dheave)

    print('Computing topo sensitivity of dry snow phase.')
    _dphi_dswe_init = dphi_dswe(slope, inc, rho, lamb)

    #_dphi_dswe_init[blunders] = np.nan
    _dphi_dswe = nan_fill(_dphi_dswe_init)

    writeBin(out_dir + 'dphi_dswe.rdc', _dphi_dswe)

    #_dphi_dswe_init = dphi_dswe(slope, inc, rho, lamb)
    #_dphi_dswe = nan_fill(_dphi_dswe_init)

    print('Computing down-slope sensitivity.')
    #compute down-slope direction vector
    d_u = slope_sin
    d_n = slope_cos*aspect_cos
    d_e = slope_cos*aspect_sin

    #compute down-slope projection to los
    dphi_ddownslope = 4*np.pi/lamb*(-1*(l_u*d_u + l_n*d_n + l_e*d_e))
    writeBin(out_dir + 'dphi_ddownslope.rdc', dphi_ddownslope)

    #compute down-slope projection to los
    dphi_ddownslope_slope_scaled = 4*np.pi/lamb*(-1*(l_u*d_u + l_n*d_n + l_e*d_e))*np.tan(slope)
    writeBin(out_dir + 'dphi_ddownslope_slope_scaled.rdc', dphi_ddownslope_slope_scaled)

    #remap tpi map
    print('Making tpi map.')
    #par.geotiff2rdc(tpi_utm_fn, out_dir + 'tpi.rdc', dem_par, gc_map)


def dphi_dswe(u, inc, rho, lamb):

    eps = snow_real_permittivity(rho)
    xi = 4*np.pi/lamb/rho*(np.sqrt(eps-1+np.cos(inc)**2)-np.cos(inc)) * np.cos(u)
    return xi


def dphi_dheave(inc, lamb):
    _d = -4*np.pi/lamb*np.cos(inc)
    return _d


def get_band(fname,crop=None):
    ds = gdal.Open(fname)
    band_obj = ds.GetRasterBand(1)
    arr = band_obj.ReadAsArray()
    if crop is not None:
        arr = arr[crop[0]:crop[0]+crop[2], crop[1]:crop[1]+crop[3]]
    return arr


def get_relative_swe(date1, date2, show_results=False):

    res = {}

    d1 = pd.to_datetime(date1)
    d2 = pd.to_datetime(date2)

    if show_results:
        print(f"Relative Snow Accumulation\n({date1} - {date2})")

    # env canada - Inuvik
    ec_inuvik = getECMetData(pd.DataFrame(index=[d1,d2]), met_station='Inuvik')
    s1 = ec_inuvik.loc[d1]['Snow on Grnd (cm)']
    s2 = ec_inuvik.loc[d2]['Snow on Grnd (cm)']
    rsd = s2 - s1   # relative snow depth
    rsw = (rsd*CM_TO_MM) * snow_density  # relative swe
    if show_results:
        print(f"Inuvik: {rsd:.2f} cm [{rsw:.2f} mm SWE]")
    res['EC-Inuvik'] = rsw * MM_TO_M
    #print(f"\nInuvik: {rsd} ({s1} - {s2}")
    #res['EC-Inuvik'] = rsd #cm

    # env canada - Trail Valley
    ec_trailValley = getECMetData(pd.DataFrame(index=[d1, d2]), met_station='TrailValley')
    s1 = ec_trailValley.loc[d1]['Snow on Grnd (cm)']
    s2 = ec_trailValley.loc[d2]['Snow on Grnd (cm)']
    rsd = s2 - s1  # relative snow depth
    rsw = (rsd * CM_TO_MM) * snow_density  # relative swe
    if show_results:
        print(f"Trail Valley: {rsd:.2f} cm [{rsw:.2f} mm SWE]")
    res['EC-TrailValley'] = rsw * MM_TO_M
    #res['EC-TrailValley'] = rsd #cm

    # era5
    sde = cfg['era5_dir'] + date1 + '_' + date2 + '.sde'
    sde_im = readBin(sde, master_par.dim, 'float32')
    rsd = np.nanmean(sde_im) * M_TO_CM  # relative snow depth
    rsw = (rsd * CM_TO_MM) * snow_density  # relative swe
    if show_results:
        print(f"ERA5: {rsd:.2f} cm [{rsw:.2f} mm SWE]")
    res['ERA5-Reanalysis'] = rsw * MM_TO_M
    #res['ERA5-Reanalysis'] = rsd #cm


    # sites
    for sf in snow_files:
        site = re.search(r'site_.', sf).group(0)
        sdf = pd.read_csv(sf, index_col=0)
        sdf.index = pd.to_datetime(sdf.index)
        try:
            s1 = sdf.loc[d1]['snow_sub_heave']
            s2 = sdf.loc[d2]['snow_sub_heave']
            rsd = s2 - s1  # relative snow depth
            rsw = (rsd * CM_TO_MM) * snow_density  # relative swe
            if show_results:
                print(f"{site}: {rsd:.2f} cm [{rsw:.2f} mm SWE]")
            res[site] = rsw * MM_TO_M
            #res[site] = rsd #cm
        except:
            if show_results:
                print(f"{site}: {np.nan}")
            res[site] = np.nan

    #print(res)

    return res


# Jayson Eppler 2022
def swe_est_clicky_func(pt_xy, phi_full, xi_full, exclude_mask_full, plots_for_paper=False):
    #swe_range = (-0.05, 0.08)
    swe_range = (-0.5, 0.8)
    swe_del = 0.002
    #swe_range = cfg['swe_range']
    #swe_del = cfg['swe_del']
    win = win1000
    swes = swe_range[0] + np.arange(int((swe_range[1] - swe_range[0]) / swe_del)) * swe_del
    swe_render_lim = np.max(np.abs(swe_range))
    n_swes = len(swes)

    win = np.asarray(win)
    cmin = pt_xy - win // 2
    cmax = cmin + win

    # extract local patch
    phi = phi_full[cmin[0]:cmax[0], cmin[1]:cmax[1]]
    xi = xi_full[cmin[0]:cmax[0], cmin[1]:cmax[1]]

    exclude_mask = exclude_mask_full[cmin[0]:cmax[0], cmin[1]:cmax[1]]
    phi[exclude_mask]=0

    n_interp = 3
    n_interp_rad = (n_interp - 1) // 2

    interp_points = np.zeros((n_interp), dtype='float32') - 1.  # n_interp points in vicinity of peak
    idx_max =  -1
    buffer = np.zeros((n_interp), dtype='float32')  # last n_interp points
    coh = np.zeros(n_swes)
    for ii in np.arange(n_swes):
        # print(ii+1, 'of', n_swes)
        phi_demod_ii = phi * np.exp(-1j * swes[ii] * xi)
        coh[ii] = np.abs(np.mean(phi_demod_ii))/np.mean((~exclude_mask).astype(float))
        if ii < n_interp:
            # initial phase, just populate buffer
            buffer[ii] = coh[ii]
        if ii >= n_interp:
            # later phase, shift buffer, then add latest entry to end
            buffer[0:-1] = buffer[1:]
            buffer[-1] = coh[ii]
            # find targets with new max coh
            update_mask = buffer[n_interp_rad] > interp_points[n_interp_rad]
            # print(np.sum(update_mask), 'new peak coherences found.')
            # update interp_points for those targets with new max coh
            if update_mask:
                interp_points[:] = buffer[:]
                idx_max = ii - n_interp_rad


    #coh_max = interp_points[n_interp_rad,:,:]
    swe_est = swes[idx_max]
    #quadratic components (hardcoded for n_interp = 3)
    a = interp_points[1]
    b = -0.5*interp_points[0] + 0.5*interp_points[2]
    c = 0.5*interp_points[0] -1.0*interp_points[1] + 0.5*interp_points[2]
    dx = -b/2/c
    #handle cases where center point is not the highest
    dx = np.clip(dx, -1, 1)
    swe_est_refined = swe_est + dx*swe_del
    coh_max_refined = a + b*dx + c*dx**2
    #print('swe_est_refined, coh_max_refined:', swe_est_refined, coh_max_refined)

    phi_masked = phi[~exclude_mask]
    phi_demean = np.angle(phi_masked*np.conj(np.mean(phi_masked)))
    phi_demean -= np.mean(phi_demean)

    xi_masked = xi[~exclude_mask]
    xi_demean = xi_masked-np.mean(xi_masked)

    swe_est_corr = np.mean(phi_demean*xi_demean)/np.mean(xi_demean**2)
    #print('swe_est_corr:', swe_est_corr)

    """obsolete
    def lad_cost(x, v, w):
        cost = np.sum(np.abs(x[0]*v - w))
        print('swe, cost', x[0], cost)
        return cost
    def fit_lad_swe(swe_init, xi_demean, phi_demean):
        scale = 1
        x_init = (swe_init/scale)
        bounds = ((-1./scale, 1./scale),)
        gtol = 1e-13; ftol = 1e-6;
        maxiter = 200
        #minimize_options = {'gtol': gtol, 'ftol': ftol, 'maxiter': maxiter}
        minimize_options = None
        opt_result = opt.minimize(lad_cost, x_init, args=(xi_demean, phi_demean), options=minimize_options, bounds=bounds)
        swe = opt_result.x[0]*scale

        print('*** Final solution ***')
        print('swe = ', swe)
        #return swe
        return swe_init
    swe_est_lad = fit_lad_swe(swe_est_corr, xi_demean, phi_demean)
    print('swe_est_lad:', swe_est_lad)
    """

    fig = plt.figure(figsize=(fig_width_full, 4.5))
    gs = gridspec.GridSpec(ncols=6, nrows=2, figure=fig)
    fig.add_subplot(gs[1,0:3])

    plt.hist2d(xi_demean/1000., phi_demean, bins=100, cmap='Greys')

    xpts = np.asarray([np.min(xi_demean), np.max(xi_demean)])
    #plt.plot(xpts, xpts * swe_est_corr, color='tab:blue', label = r'$\hat s$ = {:05.3f} m'.format(swe_est_corr))
    plt.plot(xpts/1000., xpts*swe_est_refined, color='tab:red', label=r'$\hat s_w$ = {:02.0f} mm'.format(swe_est_refined*1000.))
    #plt.plot(xpts, xpts*swe_est_lad, color='tab:green', label=r'$\hat s_L1$ = {:05.3f} m'.format(swe_est_lad))
    plt.xlabel('Zero mean dry-snow ' r'phase sensitivity, $\~\xi$ [radians/mm]')
    plt.ylabel(r'Centered phase, $\~\Phi$ [radians]')
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(np.pi))
    plt.xlim([np.min(xi_demean)/1000, np.max(xi_demean)/1000])
    plt.ylim((-np.pi, np.pi))
    plt.legend(loc='lower right')
    #plt.title(r'Regression of $\~\Phi$ vs. $\~\xi$')
    plt.title(r'2D-histogram of $\~\Phi$ vs. $\~\xi$')


    def format_func(value, tick_number):
        # find number of multiples of pi
        N = int(np.round(value / np.pi))
        if N == 0:
            return "0"
        elif N == 1:
            return r"$\pi$"
        elif N == -1:
            return r"-$\pi$"
        else:
            return r"${0}\pi$".format(N)

    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_func))

    fig.add_subplot(gs[1, 3:])
    plt.title('Periodogram')
    plt.plot(swes*1000., coh, 'k')
    plt.plot([swe_est_refined*1000., swe_est_refined*1000.], [0., coh_max_refined*1.1], '--', color='tab:red')
    plt.text((swe_est_refined+0.005)*1000., 0.0, r'$\hat s_w$ = {:02.0f} mm'.format(swe_est_refined*1000.) , horizontalalignment='left', verticalalignment='bottom', size=8)

    plt.xlim((-0.3*1000, 0.3*1000))
    plt.ylim((0., coh_max_refined*1.1))
    plt.xlabel(r'Correcting $\Delta$SWE [mm]')
    plt.ylabel('Spectral magnitude')


    fig.add_subplot(gs[0, 0:2])
    plt.title(r'SWE sensitivity $\xi$')
    im = plt.imshow(xi[::-1, :].T /1000., vmin=vmin, vmax=vmax, extent=extent)
    plt.xlabel('Ground range position [m]')
    plt.ylabel('Azimuth position [m]')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ticks = [vmin, vmax])
    cbar.ax.get_yaxis().labelpad = 0
    cbar.ax.set_ylabel('radians/mm', rotation=90)

    fig.add_subplot(gs[0, 2:4])
    plt.title('Uncorrected phase')
    im = plt.imshow(np.angle(phi)[::-1, :].T, vmin=-np.pi, vmax=np.pi, extent=extent, cmap='hsv', alpha=0.5)
    plt.xlabel('Ground range position [m]')
    plt.gca().axes.yaxis.set_ticklabels([])
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ticks = [-np.pi, np.pi])
    cbar.ax.set_yticklabels([r'-$\pi$', r'$\pi$'])
    cbar.ax.get_yaxis().labelpad = 0
    cbar.ax.set_ylabel('radians', rotation=90)

    fig.add_subplot(gs[0, 4:])
    plt.title(r'$\hat s_w$ corrected phase')
    im = plt.imshow(np.angle(phi*np.exp(-1j*swe_est_refined*(xi-np.mean(xi))))[::-1, :].T, vmin=-np.pi, vmax=np.pi, extent=extent, cmap='hsv', alpha=0.5)
    plt.xlabel('Ground range position [m]')
    plt.gca().axes.yaxis.set_ticklabels([])
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ticks = [-np.pi, np.pi])
    cbar.ax.set_yticklabels([r'-$\pi$', r'$\pi$'])
    cbar.ax.get_yaxis().labelpad = 0
    cbar.ax.set_ylabel('radians', rotation=90)


    plt.tight_layout()
    if plots_for_paper:
        plt.savefig(fig_dir + 'inset_example.png')

    plt.show()

    return swes*1000, coh


# Jayson Eppler 2022 (updated)
def swe_est_clicky(pnts=None, ifg_fname=None, xi_fname=None):
    #launch click tool and perform estimation for one window, showing cost function behavior
    #stack = cfg['stack']
    looks_str = cfg['looks_str']
    #swe_est_dir = cfg['swe_est_dir']

    dim = master_par.dim

    #xi = get_xi(cfg)
    xi = readBin(dem_dir + 'topo/dphi_dswe.rdc', dim, 'float32')
    #xi = uniform_filter(xi, (12,12))

    #plt.imshow(xi.T/1000, vmin=vmin, vmax=vmax)
    #plt.colorbar()
    #plt.show()

    # define mask to exclude from estimation
    #exclude_mask = get_water_mask(cfg) #just exclude water areas since the blunders were masked out and filled already



    if xi_fname:
        quality_map = readBin(xi_fname, dim, 'float32')
        pltkwargs = dict(vmin=0, vmax=15, interpolation='nearest', cmap='RdYlGn')

        launch_clicky(quality_map, pnts, pltkwargs, point_swe_clicky_func)
        return

    elif ifg_fname is None:
        ifg_fname = ifg_dir + '20220113_HH_20220124_HH.diff'
        #swe_fname = out_dir + '20220113_HH_20220124_HH.diff.swe'

    pltkwargs = dict(vmin=-np.pi, vmax=np.pi, interpolation='nearest', cmap='hsv', alpha=0.5)

    phi = readBin(ifg_fname, dim, 'complex64')
    #swe_est_refined = readBin(swe_fname, dim, 'float32')
    #coh_max_refined = readBin(swe_fname+'.coh', dim, 'float32')

    print(np.angle(phi).shape)
    launch_clicky(np.angle(phi), pnts, pltkwargs, swe_est_clicky_func, phi, xi, exclude_mask)


# Jayson Eppler 2022
def monte_carlo_mpinstance(coh, xi,  swe_range=(-0.1, 0.1), swe_del = 0.002, win=(64,64), exclude_mask=None):
    np.random.seed()
    swe_render_lim = np.max(np.abs(swe_range))
    dim = coh.shape

    x1 = np.random.randn(dim[0], dim[1]).astype(np.float32) + 1j*np.random.randn(dim[0], dim[1]).astype(np.float32)
    x2 = np.random.randn(dim[0], dim[1]).astype(np.float32) + 1j*np.random.randn(dim[0], dim[1]).astype(np.float32)
    x2 = (coh*x1 + (1 - coh**2)**0.5*x2)
    phi_sim = np.conj(x1)*x2
    x1=None
    x2=None

    swe_est_refined, _ = est_swe(phi_sim, xi, swe_range=swe_range, swe_del=swe_del, win=win, exclude_mask=exclude_mask)
    return swe_est_refined


# Jayson Eppler 2022
def monte_carlo(cfg, pair_name=None, plot = False):
    #monte carlo estimation of SWE variance and bias
    stack = cfg['stack']
    looks_str = cfg['looks_str']
    win=cfg['win']
    swe_est_dir = cfg['swe_est_dir']
    swe_range = cfg['swe_range']
    swe_del = cfg['swe_del']

    swes = swe_range[0] + np.arange(int((swe_range[1] - swe_range[0]) / swe_del)) * swe_del

    if pair_name is None:
        #form list using glob and then recurse back in same function for each pair found
        swe_fnames = glob.glob(swe_est_dir + '????????_????????' + '.diff.swe')
        pair_names = [os.path.basename(swe_fname)[0:17] for swe_fname in swe_fnames]
        print('Found', len(pair_names),'. Performing Monte Carlo on each one...')
        for pair_name in pair_names:
            print(pair_name, '...')
            monte_carlo(cfg, pair_name=pair_name, plot=plot)
        return

    swe_render_lim = 60. #mm

    max_gb = 64.
    n_iter = 40
    crop = None
    #crop = [0,0,1000,1000]

    swe_fname = swe_est_dir+pair_name+'.diff.swe'
    sim_sd_fname = swe_fname + '.sim_sd'

    par = MLI_Par(stack._dir + 'rmli_' + looks_str + '/rmli_' + looks_str + '.ave.par')
    dim = par.dim

    #xi = get_xi(cfg)
    xi = readBin(dem_dir + 'topo/dphi_dswe.rdc', dim, 'float32')

    #exclude_mask = get_water_mask(cfg)  # just exclude water areas since the blunders were masked out and filled alread
    exclude_mask = cfg['watermask']

    if crop is not None:
        exclude_mask = exclude_mask[crop[0]:crop[2], crop[1]:crop[3]]
    # get standard deviation of xi
    xi_masked = np.copy(xi)
    xi_masked[exclude_mask] = 0
    good_mask = np.ones_like(xi)
    good_mask[exclude_mask] = 0
    good_mask_filtered = uniform_filter(good_mask, win)
    xi_sd = np.sqrt(uniform_filter(xi_masked ** 2, win) / good_mask_filtered - (
                uniform_filter(xi_masked, win) / good_mask_filtered) ** 2)

    if crop is not None:
        dim = crop[2:]
    else:
        dim = par.dim
    if not os.path.exists(sim_sd_fname):
        coh = readBin(swe_fname+'.coh', par.dim, 'float32', crop=crop)

        multiprocess=False
        #generate all the simulated swe estimates
        #job_gb = 1.6 * dim[0] * dim[1]/(4076*2255)
        job_gb = 15
        nodes_max = int(max_gb/job_gb)
        #TODO fix the mp code so I can use nodes max!!!
        p = mp.ProcessingPool(multiprocess=multiprocess)
        res = p.map(lambda ii: monte_carlo_mpinstance(coh, xi,  swe_range=swe_range, swe_del=swe_del, win=win, exclude_mask=exclude_mask), np.arange(n_iter))

        swe_sims = np.asarray(res)
        swe_sd = np.var(swe_sims, axis=0)**0.5
        swe_bias = np.mean(swe_sims, axis=0)

        #writeBin(sim_sd_fname, swe_sd)
        #render_png(swe_sd.T * 1000, sim_sd_fname + '.png', vmin=0, vmax=10)
        plt.show(swe_sd.T * 1000)#, sim_sd_fname + '.png', vmin=0, max=10)
    else:
        swe_sd = readBin(sim_sd_fname, dim, 'float32')

    if plot:
        swe_est = readBin(swe_fname, dim, 'float32')
        coh_max = readBin(swe_fname+'.coh', dim, 'float32')
        nan_mask = (~np.isfinite(swe_est)) | (~np.isfinite(swe_sd))
        #bad_mask = nan_mask | (swe_sd*1000 > 5)
        bad_mask = nan_mask | (swe_est <= swes[4]) | (swe_est >= swes[-4])
        #bad_mask = np.reshape(bad_mask, dim)
        extent = np.asarray([0, par['range_pixel_spacing']*par['range_samples']/np.sin(np.deg2rad(par['incidence_angle'])),0, par['azimuth_pixel_spacing']*par['azimuth_lines']])/1000

        plt.figure(figsize=(fig_width_full, 4.))

        plt.subplot(2,2,1)
        cmap = cm.get_cmap('RdBu')
        cmap.set_bad(color='grey')
        swe_est[bad_mask] = np.nan
        #im=plt.imshow(np.ma.array(swe_est, mask=(np.reshape(bad_mask, dim)))[::-1, :].T*1000, cmap='RdBu', vmin=-swe_render_lim,
        #           vmax=swe_render_lim, extent=extent)
        im=plt.imshow(swe_est[::-1, :].T*1000, cmap='RdBu', vmin=-swe_render_lim,vmax=swe_render_lim, extent=extent)
        plt.title(r'Estimated $\Delta$SWE')
        #plt.xlabel('Ground range position [km]')
        plt.xticks([])
        plt.ylabel('Azimuth\nposition [km]')
        plt.yticks([0, 5, 10])
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, ticks = [-swe_render_lim, swe_render_lim])
        cbar.ax.get_yaxis().labelpad = 0
        cbar.ax.set_ylabel('mm SWE', rotation=90)

        plt.subplot(2,2,2)
        cmap = cm.get_cmap('viridis')
        cmap.set_bad(color='grey')
        coh_max[bad_mask]=np.nan
        im = plt.imshow(coh_max[::-1, :].T, vmin=0, vmax=1, extent=extent, cmap=cmap)
        plt.title('Residual Phase Coherence')
        #plt.xlabel('Ground range position [km]')
        plt.xticks([])
        #plt.ylabel('Azimuth\nposition [km]')
        #plt.yticks([0, 5, 10])
        plt.yticks([])
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, ticks = [0, 1])
        cbar.ax.get_yaxis().labelpad = 0
        cbar.ax.set_ylabel('Coherence', rotation=90)

        plt.subplot(2,2,3)
        xi_sd[exclude_mask]=np.nan
        im=plt.imshow(xi_sd[::-1, :].T, vmin=0, vmax=20, extent=extent, cmap=cmap)
        plt.title(r'SWE sensitivity ($\xi$) std. dev.')
        plt.xlabel('Ground range position [km]')
        plt.ylabel('Azimuth\nposition [km]')
        plt.yticks([0, 5, 10])
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, ticks = [0, 20])
        cbar.ax.get_yaxis().labelpad = 0
        cbar.ax.set_ylabel('radians/mm', rotation=90)

        plt.subplot(2,2,4)
        swe_sd[bad_mask]=np.nan
        im=plt.imshow(swe_sd[::-1, :].T * 1000., vmin=0, vmax=10, extent=extent, cmap=cmap)
        plt.title(r'Estimated $\Delta$SWE std. dev.')
        plt.xlabel('Ground range position [km]')
        #plt.ylabel('Azimuth\nposition [km]')
        #plt.yticks([0, 5, 10])
        plt.yticks([])
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, ticks = [0, 10])
        cbar.ax.get_yaxis().labelpad = 0
        cbar.ax.set_ylabel('mm SWE', rotation=90)



        # plt.subplot(2,2,3)

        # im=plt.imshow(swe_est[::-1, :].T*1000, cmap = cmap, vmin=-swe_render_lim, vmax=swe_render_lim, extent=extent)
        # plt.title(r'Estimated $\Delta$SWE ')
        # plt.xlabel('Ground range position [km]\n(c)')
        # plt.ylabel('Azimuth\nposition [km]')
        # plt.yticks([0, 5, 10])
        # divider = make_axes_locatable(plt.gca())
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # cbar = plt.colorbar(im, cax=cax, ticks = [-swe_render_lim, swe_render_lim])
        # cbar.ax.get_yaxis().labelpad = 0
        # cbar.ax.set_ylabel('mm SWE', rotation=90)


        plt.tight_layout()
        fig_dir = '/local-scratch/users/aplourde/'
        plt.savefig(fig_dir + 'monte_carlo.png')

        # plt.figure()
        # swe_range = (-0.05, 0.08)
        # swe_del = 0.002
        # swes = swe_range[0] + np.arange(int((swe_range[1] - swe_range[0]) / swe_del)) * swe_del
        # bad_mask2 = nan_mask.flatten() | (swe_est.flatten() <= swes[4]) | (swe_est.flatten() >= swes[-4])
        # im = plt.imshow(np.ma.array(swe_est, mask=(np.reshape(bad_mask2, dim)))[::-1, :].T * 1000, cmap='RdBu',
        #                 vmin=-swe_render_lim * 1000,
        #                 vmax=swe_render_lim * 1000, extent=extent)


def parse_ifg_dates(root):
    pattern = r"\d{8}"  # regular expression pattern for date extraction

    # Find all occurrences of the pattern in the string
    dates = re.findall(pattern, root)
    m_date = dates[0]
    s_date = dates[1]
    return m_date, s_date


def get_site_centers():

    df = pd.read_csv(cr_file)

    # Calculate mean x_loc and y_loc values for each site
    site_means = df.groupby('site')[['x_loc', 'y_loc']].mean()
    site_means['label'] = ['site_' + str(i) for i in site_means.index]

    # Merge the mean values back into the DataFrame and center the coordinates
    df = df.merge(site_means, left_on='site', right_index=True, suffixes=('', '_mean'))

    return site_means


def meta_stats(num_points=None):

    if num_points is None:
        size = master_par.dim[0] * master_par.dim[1]
        # select 1% of the possible points
        num_points = int(size*0.001)

    # create random sample of points
    print(master_par.dim)
    randarray = [np.random.randint(0, master_par.dim[0], size=num_points), np.random.randint(0, master_par.dim[1], size=num_points)]
    randarray = np.asarray(randarray).T

    # get SWE estimate for annulus at each point
    res = site_swe(show_results=True, plt_annulus=False, input_coords=randarray)
    timeseries_swe(res)

    # Calculate statistics

    # create scatter and timeseries based on mean/median with std error bars

    pass


def xi_histograms(coords, cfg):
    # xi histograms
    site_1_center = np.mean([coords['cr1'], coords['cr2']], axis=0)
    site_1_center = [int(px) for px in site_center]
    site_2_center = [int(px) for px in coords['cr3']]
    #good_coh = [250, 475] #[371, 783]  # [486, 643]
    good_coh = [3298, 2798]
    sla_file = SLA_cfg['working_dir'] + SLA_cfg['sub_dir'] + 'cr_loc_' + SLA_cfg['looks_str'] + '.txt'
    coords = unpack_cr_coords(cr_file)
    sla_site = [int(px) for px in coords['cr1']]

    ### DO SITE 1 ###
    swe_stack(pt_xy=site_1_center)
    est_dim = cfg['win']
    xi_im = readBin(cfg['swe_est_dir'] + 'patch.xi', est_dim, 'float32')
    exclude_mask = readBin(cfg['swe_est_dir'] + 'patch.mask', est_dim, 'float32') != 0

    plt.subplot(241)
    plt.title(f"TSX Site 1 - Xi Map\n{TSX_win ** 2}km2")
    plt.imshow(xi_im.T, vmin=cfg['plt_range'][0] * 1000, vmax=cfg['plt_range'][1] * 1000)
    plt.colorbar()
    plt.subplot(242)
    plt.title("TSX Site 1 - Xi Distribution")
    plt.hist(xi_im[~exclude_mask].flatten(), bins=100)

    ### DO SITE 2 ###
    swe_stack(pt_xy=site_2_center)
    est_dim = cfg['win']
    xi_im = readBin(cfg['swe_est_dir'] + 'patch.xi', est_dim, 'float32')
    exclude_mask = readBin(cfg['swe_est_dir'] + 'patch.mask', est_dim, 'float32') != 0

    plt.subplot(243)
    plt.title(f"TSX Site 2 - Xi Map\n{TSX_win ** 2}km2")
    plt.imshow(xi_im.T, vmin=cfg['plt_range'][0] * 1000, vmax=cfg['plt_range'][1] * 1000)
    plt.colorbar()
    plt.subplot(244)
    plt.title("TSX Site 2 - Xi Distribution")
    plt.hist(xi_im[~exclude_mask].flatten(), bins=100)

    ### DO SITE GOOD COH ###
    swe_stack(pt_xy=good_coh)
    est_dim = cfg['win']
    xi_im = readBin(cfg['swe_est_dir'] + 'patch.xi', est_dim, 'float32')
    exclude_mask = readBin(cfg['swe_est_dir'] + 'patch.mask', est_dim, 'float32') != 0

    plt.subplot(245)
    plt.title(f"TSX Good Coh - Xi Map\n{TSX_win ** 2}km2")
    plt.imshow(xi_im.T, vmin=cfg['plt_range'][0] * 1000, vmax=cfg['plt_range'][1] * 1000)
    plt.colorbar()
    plt.subplot(246)
    plt.title("TSX Good Coh - Xi Distribution")
    plt.hist(xi_im[~exclude_mask].flatten(), bins=100)

    ### DO SITE SLA ###
    cfg = SLA_cfg
    # swe_stack(pt_xy=sla_site)
    est_dim = cfg['win']
    xi_im = readBin(cfg['swe_est_dir'] + '/patch.xi', est_dim, 'float32')
    exclude_mask = readBin(cfg['swe_est_dir'] + '/patch.mask', est_dim, 'float32') != 0

    plt.subplot(247)
    plt.title(f"Spotlight - Xi Map\n{SLA_win ** 2}km2")
    plt.imshow(xi_im.T, vmin=cfg['plt_range'][0] * 1000, vmax=cfg['plt_range'][1] * 1000)
    plt.colorbar()
    plt.subplot(248)
    plt.title("Spotlight - Xi Distribution")
    plt.hist(xi_im[~exclude_mask].flatten(), bins=100)

    plt.show()


def mk_temperal_coherance_mask():
    ifgs = glob.glob(ifg_dir + '*' + cfg['diff_ext'])
    ifgs = get_itab_diffs(ifgs, cfg['itab'], RSLC_tab)

    pnts = get_site_centers()

    dim = master_par.dim
    csum = np.zeros(dim, dtype='complex64')
    for ifg in ifgs:
        print(ifg)
        root = os.path.basename(ifg).split('.')[0]

        m_date, s_date = parse_ifg_dates(root)

        #cphi is the complex conjugate of slcs
        cphi = readBin(ifg, dim, 'complex64')
        phi = np.angle(cphi)

        cphi /= np.abs(cphi)
        cphi[~np.isfinite(cphi)] = 0
        csum += cphi


    temporal_coherance = np.abs(csum / len(ifgs))
    #temporal_coherance[exclude_mask] = np.nan


    plt.subplot(121)
    plt.imshow(temporal_coherance.T, cmap='Greys_r', vmin=0, vmax=1)
    plt.colorbar()
    plt.subplot(122)
    plt.scatter(pnts[0], pnts[1], color='red')
    temporal_coherance = uniform_filter(temporal_coherance, [3,3])
    temporal_coherance[temporal_coherance < 0.2] = 0
    temporal_coherance[temporal_coherance != 0] = 1
    plt.imshow(temporal_coherance.T, cmap='Greys_r', vmin=0, vmax=1)
    plt.colorbar()
    plt.show()


def create_kilometer_axis_labels(ax, win, nticks=4):

    dim = master_par.dim

    # Define the physical size of your image in kilometers
    image_width_km = dim[0]/win[0]
    image_height_km = dim[1]/win[1]

    # Calculate the scaling factors for x and y axes
    x_scale_factor = image_width_km / dim[0]
    y_scale_factor = image_height_km / dim[1]

    # Calculate the desired number of ticks in kilometers
    num_ticks = nticks  # Adjust the number of ticks as needed

    # Calculate the tick positions and labels
    x_tick_positions = np.linspace(0, image_width_km, num_ticks)
    y_tick_positions = np.linspace(0, image_height_km, num_ticks)

    # Set the x-axis and y-axis ticks and labels
    ax.set_xticks(np.linspace(0, dim[0], num_ticks))
    ax.set_xticklabels([f'{int(val)} km' for val in x_tick_positions])
    ax.set_yticks(np.linspace(0, dim[1], num_ticks))
    ax.set_yticklabels([f'{int(val)} km' for val in y_tick_positions])

    return


def mk_xi_quality_mask():

    dim = master_par.dim
    xi = readBin(dem_dir + 'topo/dphi_dswe.rdc', dim, 'float32')

    exclude_mask = cfg['watermask']
    #xi_masked = xi
    #xi_masked[exclude_mask] = np.nan

    hist, bins = np.histogram(xi[~exclude_mask], bins=100)
    xi_ave = np.mean(xi[~exclude_mask])
    xi_std = np.std(xi[~exclude_mask])
    xi_min = np.min(xi[~exclude_mask])
    xi_max = np.max(xi[~exclude_mask])
    xi_range = xi_max - xi_min
    xi_skew = skew(xi[~exclude_mask])
    print(xi_ave, xi_std, xi_min, xi_max, xi_skew)

    xi_demean = xi-np.mean(xi[~exclude_mask])
    #cfg['win'] = (10,10)
    xi_ave = uniform_filter(xi_demean, cfg['win'])
    xi_sqave = uniform_filter(xi_demean**2, cfg['win'])
    xi_std = np.sqrt(xi_sqave - xi_ave**2)

    xi_ave2 = uniform_filter(xi_demean, [10,cfg['win'][0]])
    xi_sqave2 = uniform_filter(xi_demean**2, [10,cfg['win'][0]])
    xi_std2 = np.sqrt(xi_sqave2 - xi_ave2**2)
    xi_ave3 = uniform_filter(xi_demean, [cfg['win'][0],10])
    xi_sqave3 = uniform_filter(xi_demean**2, [cfg['win'][0],10])
    xi_std3 = np.sqrt(xi_sqave3 - xi_ave3**2)
    #xi_skew = generic_filter(xi_demean, skew, cfg['win'])

    win = 3
    #xi_skew = np.empty_like(xi_demean)
    #for ii in range(xi_demean.shape[0]):
    #    for jj in range(xi_demean.shape[1]):
    #        print(xi_demean[ii,jj])
    #        # set bounds
    #        x1 = np.max([0, ii - cfg['win'][0] // 2 ])
    #        x2 = np.min([ii + cfg['win'][0] // 2 + 1, xi_demean.shape[0]])
    #        y1 = np.max([0, jj - cfg['win'][1] // 2])
    #        y2 = np.min([jj + cfg['win'][1] // 2 + 1, xi_demean.shape[1]])
    #        chip = xi_demean[x1:x2, y1:y2]
    #        xi_skew[ii,jj] = skew(chip.flatten())
    #xi_demean[exclude_mask] = np.nan
    #xi_std[exclude_mask] = np.nan
    #xi_std2[exclude_mask] = np.nan
    #xi_std3[exclude_mask] = np.nan
    #xi_skew[exclude_mask] = np.nan

    fig, axes = plt.subplots(1, 5, sharex='col', sharey=False, gridspec_kw={'wspace': 0, 'hspace': 0})
    cr_pnts = get_site_centers()


    im1 = axes[0].imshow(xi_demean.T, vmin=-40, vmax=40)
    axes[0].scatter(cr_pnts['x_loc'], cr_pnts['y_loc'], c='k', s=6)

    texts = []
    for i, row in cr_pnts.iterrows():
        texts.append(axes[0].text(row['x_loc'], row['y_loc'], "site "+str(i), fontsize=12, ha='center', va='bottom'))

    #texts = [axes[0].text(cr_pnts['x_loc'][i], cr_pnts['y_loc'][i], cr_pnts.index[i], ha='center', va='center') for i in range(len(cr_pnts))]

    # Use adjustText to automatically adjust text positions to avoid overlap
    adjust_text(texts, ax=axes[0], autoalign='xy', arrowprops=dict(arrowstyle='-', color='k'))

    create_kilometer_axis_labels(axes[0], TSX_win1000)
    plt.colorbar(im1, ax=axes[0], label="De-meaned Sensitivity - Xi\n{rad/mm}")

    im2 = axes[1].imshow(xi_std.T, vmin=0, vmax=15, cmap='RdYlGn')
    axes[1].scatter(cr_pnts['x_loc'], cr_pnts['y_loc'], c='k', s=6)
    create_kilometer_axis_labels(axes[1], TSX_win1000)
    fig.colorbar(im2, ax=axes[1], label="Standard Deviation of the Sensitivity")


    inc_rad = readBin(dem_dir + 'topo/inc.rdc', master_par.dim, 'float32')
    inc_deg = np.rad2deg(inc_rad)

    #plt.title("Inc. Mask")
    inc_mask = np.zeros_like(inc_deg)
    inc_mask[inc_deg>25] = 1
    inc_mask[cfg['watermask']] = 0
    im3 = axes[2].imshow(inc_mask.T, cmap='Greys_r')
    axes[2].scatter(cr_pnts['x_loc'], cr_pnts['y_loc'], c='r', s=6)
    create_kilometer_axis_labels(axes[2], TSX_win1000)
    plt.colorbar(im3, ax=axes[2], label="Inclination Mask")


    slope_deg = readBin(dem_dir + 'topo/slope.rdc', master_par.dim, 'float32')
    slope_rad = np.deg2rad(slope_deg)
    slope_mask = np.zeros_like(slope_deg)
    slope_mask[slope_deg > 5] = 1
    slope_mask[cfg['watermask']] = 0
    im4 = axes[3].imshow(slope_mask.T, cmap='Greys_r')
    axes[3].scatter(cr_pnts['x_loc'], cr_pnts['y_loc'], c='r', s=6)
    create_kilometer_axis_labels(axes[3], TSX_win1000)
    plt.colorbar(im4, ax=axes[3], label="Slope Mask")

    eps = snow_real_permittivity(0.3)
    angle_component = (np.sqrt(eps - 1 + np.cos(inc_rad) ** 2) - np.cos(inc_rad)) * np.cos(slope_rad)
    ac_mask = np.zeros_like(angle_component)
    ac_mask[angle_component > 0.260] = 1
    ac_mask[cfg['watermask']] = 0
    im5 = axes[4].imshow(ac_mask.T, cmap='Greys_r')
    axes[4].scatter(cr_pnts['x_loc'], cr_pnts['y_loc'], c='r', s=6)
    create_kilometer_axis_labels(axes[4], TSX_win1000)
    plt.colorbar(im5, ax=axes[4], label="Slope Mask")

    plt.show()

    #xi_std[~xi_qual] = 0
    writeBin(dem_dir + 'xi_std', xi_std)
    #writeBin(dem_dir + 'topo/xi_skew_map.rdc', xi_skew)
    #writeBin(dem_dir + 'topo/xi.mask', xi_qual)
    writeBin(dem_dir + 'inc_mask', inc_mask)
    writeBin(dem_dir + 'slope_mask', slope_mask)
    writeBin(dem_dir + 'ac_mask_260', ac_mask)


def swe_scene_analysis():
    dim = master_par.dim
    r_grid, az_grid = np.mgrid[0:dim[0], 0:dim[1]]

    xi_im = readBin(dem_dir + 'topo/dphi_dswe.rdc', dim, 'float32')

    files = glob.glob(cfg['swe_est_dir'] + '*' + cfg['diff_ext'] + '.swe')
    print(cfg['swe_est_dir'])
    ifg_swe_estimates = get_itab_diffs(files, itab, RSLC_tab)
    #ifg_swe_estimates = []
    swe_estimates = []

    for file in ifg_swe_estimates:
        path, root = os.path.split(file)

        m_str, s_str = parse_ifg_dates(root.split('.')[0])
        m_date = pd.to_datetime(m_str)
        s_date = pd.to_datetime(s_str)

        swe_im = readBin(file, dim, 'float32')
        swe_im_water = readBin(cfg['swe_est_dir'] + '../water_mask_full/' + root, dim, 'float32')
        swe_era5 = readBin(cfg['era5_dir'] + m_str + '_' + s_str + '.sde', dim, 'float32') * snow_density

        nan_mask = np.isnan(swe_im)

        r_interp = r_grid[~nan_mask]
        az_interp = az_grid[~nan_mask]
        im_interp = swe_im[~nan_mask]
        swe_int = griddata((az_interp, r_interp), im_interp, (az_grid, r_grid))
        swe_int = swe_int.reshape(swe_im.shape)


        plt.figure(figsize=(16, 8))
        plt.subplot(141)
        plt.title("Estimated SWE")
        plt.imshow(swe_im.T, cmap='RdBu', vmin=cfg['swe_range'][0], vmax=cfg['swe_range'][1])
        plt.colorbar()
        #plt.axis('off')
        plt.gca().xaxis.set_major_formatter('')
        plt.gca().yaxis.set_major_formatter('')
        plt.subplot(142)
        plt.title("Interpolated")
        plt.imshow(swe_int.T, cmap='RdBu', vmin=cfg['swe_range'][0], vmax=cfg['swe_range'][1])
        #plt.imshow(swe_im.T, cmap='RdBu', vmin=cfg['swe_range'][0], vmax=cfg['swe_range'][1])
        plt.colorbar()
        #plt.axis('off')
        plt.gca().xaxis.set_major_formatter('')
        plt.gca().yaxis.set_major_formatter('')
        plt.subplot(143)
        plt.title("Water Masked")
        plt.imshow(swe_im_water.T, cmap='RdBu', vmin=cfg['swe_range'][0], vmax=cfg['swe_range'][1])
        plt.colorbar()
        #plt.axis('off')
        plt.gca().xaxis.set_major_formatter('')
        plt.gca().yaxis.set_major_formatter('')
        plt.subplot(144)
        plt.title("ERA5")
        plt.imshow(swe_era5.T, cmap='RdBu', vmin=cfg['swe_range'][0], vmax=cfg['swe_range'][1])
        plt.colorbar()
        #plt.axis('off')
        plt.gca().xaxis.set_major_formatter('')
        plt.gca().yaxis.set_major_formatter('')

        plt.suptitle(f"{m_str}_{s_str}")
        #plt.show()
        print(cfg['swe_est_dir'] + m_str + '_' + s_str + '.png')
        plt.savefig(cfg['swe_est_dir'] + m_str + '_' + s_str + '.png')
        plt.close()

        writeBin(cfg['swe_est_dir'] + root + '.interp', swe_int)

    from avi_timeseries import mk_avi
    files = glob.glob(cfg['swe_est_dir'] + '*.png')
    files = sorted(files)

    mk_avi(files, 'full_scene_swe', cfg['swe_est_dir'])


if __name__ == "__main__":

    looks = cfg['looks_str']

    """
    ifgs = glob.glob(ifg_dir + '*' + cfg['diff_ext'])
    ifgs = get_itab_diffs(ifgs, itab, RSLC_tab)
    """
    #looks = "18_16"
    #mk_water_mask(sar_dir + sub_dir + 'rmli_' + looks +'/', sar_dir + sub_dir, None, look_str=looks)
    #make_terrain_maps(looks_str=looks)

    #plot_snow_with_tilt()

    #compute_local_incidence_angle(dem_dir, looks_str=looks)

    #sensitivity_to_topo(snow_density, incidence_angle, wavelength, label='Spotlight True Parameters')
    #sensitivity_to_topo(snow_density, RS2_cfg['incidence_angle'], c / RS2_cfg['frequency'], label='Ultrafine True Parameters')
    #sensitivity_to_topo(snow_density, TSX_cfg['incidence_angle'], c / TSX_cfg['frequency'], label='TerraSAR True Parameters')
    #sensitivity_to_topo(snow_density, incidence_angle, c / (2 * frequency), label='Spotlight 2f')
    #sensitivity_to_topo(snow_density, incidence_angle / 2, wavelength, label='Spotlight inc/2')
    #sensitivity_to_topo(snow_density, incidence_angle / 2, c / (2 * frequency), label='Spotlight 2f, inc/2')
    #plt.legend()
    #plt.show()

    #mk_dphi_dswe(dem_dir, looks_str=looks, ifg=ifg_dir + cfg['sample_ifg'])
    #mk_dphi_dswe(dem_dir, looks_str=looks)

    #swe_stack(diffs=[ifg_dir + cfg['sample_ifg']])

    #swe_stack()
    coords = get_site_centers()
    hi_xi_tsx_2_2 = pd.Series({'x_loc': 1799, 'y_loc': 5402, 'label': 'high_xi'})
    coords = coords.append(hi_xi_tsx_2_2, ignore_index=True)
    get_swe_stack(coords)
    #swe_stack(pt_xy=test_point)

    #sample_est = [out_dir + '20201214_20210107.hds.swe']
    #site_swe(sample_est, plt_annulus=False)
    #res = site_swe(show_results=True, plt_annulus=False, input_coords='all')
    #res = site_swe(show_results=True, plt_annulus=False, input_coords=[371, 783])
    #res = site_swe(show_results=True, plt_annulus=False, input_coords=[[486, 643]])

    #xi_histograms(coords, RS2_cfg)
    #test_point = [876,4300]
    #res = point_swe([250, 475], show_results=True, plt_patch=0)
    #res = point_swe(test_point, show_results=True, plt_patch=0)
    #timeseries_swe(res)

    #phi_due_to_snow_and_heave(SLA_cfg)

    #rslcs = glob.glob(sar_dir + sub_dir + 'rslc/*.rslc')
    #dates = [os.path.basename(file.split('.')[0]) for file in rslcs]
    #blown_snow_analysis(dates)

    # clicky!
    #m_date, s_date = parse_ifg_dates(cfg['sample_ifg'])
    #get_relative_swe(m_date, s_date, show_results=True)

    #swe_est_clicky(pnts=get_site_centers(), ifg_fname=ifg_dir + cfg['sample_ifg'])
    #swe_est_clicky(pnts=get_site_centers(), xi_fname=dem_dir + 'xi_std')


    # monte carlo
    #pair_name = cfg['sample_ifg'].split('.')[0]
    #monte_carlo(cfg, pair_name, plot=1)

    # meta stats
    #meta_stats(num_points=10)

    #xi quality
    #mk_xi_quality_mask()

    #temporal coherance
    #mk_temperal_coherance_mask()

    #swe_est_clicky(pnts=get_site_centers(), xi_fname=dem_dir + 'topo/xi_std_map.rdc')


    #phi_snow_depth_map(0.05, dem_dir + 'topo/', master_par.dim, wavelength, snow_density, show_plot=True)


    #interpolate swe maps
    #swe_scene_analysis()








