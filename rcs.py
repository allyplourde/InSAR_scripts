import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os

from sarlab.gammax import *

from cr_phase_to_deformation import extract_annulus
from ptarg import ptarg_rslcs, PtargPar


working_dir = '/local-scratch/users/aplourde/HDS/projects/southern_ITH/RS2_U76_D/'
sub_dir = 'full_scene/'; master = '20180827'

ingest_cfg = {'password': 'S1m0nF7@SER', 'polarizations': 'HH'}
stack = SLC_stack(dirname=working_dir + sub_dir,name='inuvik_RS2_U76_D', master=master, looks_hr=(2,3), looks_lr=(12,18), multiprocess=True, rdcdem_refine=True, skipmode='None', ingest_cfg=ingest_cfg)

# Constants
# Corner Reflector Dimensions
l = 0.6477  # tihedral height in meters (25.5 inches)
a = 0.508  # bottom length dimension in meters (20 inches)
as2 = a * np.sqrt(2)  # bottom width dimension in meters (28 inches)
Aeff = (a ** 2) / np.sqrt(3)


def rcs_to_dB(rcs):
    return 10 * np.log10(rcs)


def calculate_theoretical_RCS(a, lamb, theta=None, phi=None):

    # Geometrical Optics method for a trihedral Corner Reflector
    # (RCS Handbook, Ruck (1970)

    if theta is None and phi is None:
        rcs = (4 * np.pi * (a ** 4)) / (3 * (lamb ** 2))
        rcs_dBsm = rcs_to_dB(rcs)
        return rcs_dBsm
    elif theta is None:
        theta = np.deg2rad(54.74)
    elif phi is None:
        phi = np.deg2rad(45)

    angle = np.cos(theta) + np.sin(theta)*(np.sin(phi) + np.cos(phi))
    rcs = (4*np.pi)/(lamb**2) * (a**4) * (angle - 2*(angle**(-1)))**2
    rcs_dBsm = rcs_to_dB(rcs)

    #Aeff = (a ** 2) / np.sqrt(3)
    #rcs_dB = rcs_dBsm - 10*np.log10(Aeff)

    return rcs_dBsm


def phase_error(a, lamb, Apx, theta, slc, dim, coords, cr='cr1', show_plot=False):

    phi_theoretical = calculate_theoretical_RCS(a, lamb)

    basename = os.path.basename(slc)
    root = basename.split('.')[0]
    date = pd.to_datetime(root)

    rc = coords['r_px_' + cr[-1]][coords['Date'] == int(root)].values[0]
    ac = coords['az_px_' + cr[-1]][coords['Date'] == int(root)].values[0]

    if ~np.isnan(rc):
        rc = int(rc)
        ac = int(ac)
        ptr_par = PtargPar(working_dir + sub_dir + 'ptarg_slc/' + root + '_' + cr + '.slc.ptr_par')

        im = readBin(slc, dim, 'complex64')
        r, az = np.meshgrid(range(rc - 1, rc + 2), range(ac - 1, ac + 2))

        if show_plot:
            plt.imshow(np.abs(im.T)) #[rc - 25:rc + 5, ac - 5:ac + 25]
            plt.scatter(r, az, color='red')
            plt.title(root)
            plt.xlim([rc-25, rc+5])
            plt.ylim([ac+25, ac-5])
            plt.show()

        phi_targ = np.sum(abs(im[r, az]))

        phi_clutter, mask = extract_annulus(im, [rc, ac], 2, 5, show_plot=0)
        clutter_dB = 2*rcs_to_dB(np.mean(np.abs(phi_clutter[mask])))

        # integrated point target energy
        E_clt = np.sum(np.abs(phi_clutter)**2)  # sigma nought
        N_clt = np.sum(mask)
        E_n = phi_targ
        N_cr = len(r)
        E_cr = E_n - (N_cr / (N_clt)) * E_clt  # eqn 14 from Gray 1990

        phi_est = E_cr * Apx
        phi_est_dB = rcs_to_dB(phi_est)

        mle = (ptr_par._r_mle + ptr_par._az_mle)/2
        mle_dB = rcs_to_dB(mle)
        r3dB = ptr_par._r_3dB_width   # dBm
        a3dB = ptr_par._az_3dB_width  # dBm
        imle = mle_dB + r3dB + a3dB  # Area in dB, equivalent to E*A


        print("Theoretical RCS: {:.4f} dB".format(phi_theoretical))
        print("Measured RCS: {:.4f} dB".format(phi_est_dB))
        print("Integrated Main Lobe Energy: {:.4f} dB".format(imle))
        print("RCS Loss: {:.4f} dB".format(phi_theoretical - phi_est_dB))
        print("Mean Clutter: {:.4f} dB".format(clutter_dB))


        SCR = phi_theoretical - clutter_dB - 10*np.log10(Apx)
        SCR_est = phi_est_dB - clutter_dB - 10 * np.log10(Apx)
        SCR_corr = 10*np.log10(E_cr/(E_clt/N_clt))
        print("Theoretical SCR: {:.4f} dB".format(SCR))
        print("Measured SCR: {:.4f} dB".format(SCR_est))
        print("Measured SCR corrected for clutter: {:.4f} dB".format(SCR_corr))

        SCR_m2 = 10**(SCR/10)
        SCR_m2_est = 10**(SCR_est/10)
        SCR_m2_corr = 10**(SCR_corr/10)

        phi_err = 1/np.sqrt(2*SCR_m2)
        phi_err_est = 1/np.sqrt(2*SCR_m2_corr)
        d_err = ((phi_err * lamb) / (4 * np.pi)) * 1000
        d_err_est = ((phi_err_est * lamb) / (4 * np.pi)) * 1000
        dv_err = d_err / np.cos(theta)
        dv_err_est = d_err_est / np.cos(theta)
        print("Theoretical phase error: {:.4f} rad".format(phi_err))
        print("Theoretical LOS error: {:.4f} mm".format(d_err))
        print("Measured phase error: {:.4f} rad".format(phi_err_est))
        print("Measured LOS error: {:.4f} mm".format(d_err_est))
        print("Measured vertical error: {:.4f} mm".format(dv_err_est))

        return {'slc': [root],
                'rcs_theory': [phi_theoretical],
                'rcs_measured': [phi_est_dB],
                'rcs_imle': [imle],
                'loss': [phi_theoretical - phi_est_dB],
                'clutter': [clutter_dB],
                'scr_theory': [SCR],
                'scr_measured': [SCR_est],
                'scr_corr': [SCR_corr],
                'phi_err_theory': [phi_err],
                'phi_err_measured': [phi_err_est],
                'los_err_theory': [d_err],
                'los_err_measured': [d_err_est],
                'vert_err_theory': [dv_err],
                'vert_err_measured': [dv_err_est]}
    else:
        return {'slc': [root],
                'rcs_theory': [np.nan],
                'rcs_measured': [np.nan],
                'rcs_imle': [np.nan],
                'loss': [np.nan],
                'clutter': [np.nan],
                'scr_theory': [np.nan],
                'scr_measured': [np.nan],
                'scr_corr': [np.nan],
                'phi_err_theory': [np.nan],
                'phi_err_measured': [np.nan],
                'los_err_theory': [np.nan],
                'los_err_measured': [np.nan],
                'vert_err_theory': [np.nan],
                'vert_err_measured': [np.nan]}

def find_closest_arg(arr, val):
    """Find the argument (i.e., index) of the closest value to val in arr"""
    idx = []
    for i in range(len(arr)):
        if i > 10 and i < 90:
            if abs(arr[i] - val) < 0.1:
                idx.append(i)

    return idx


def model_RCS_deviations(a, lamb, show_plots = False):

    # calculate max RCS
    theta_ideal = np.deg2rad(54.74)
    phi_ideal = np.deg2rad(45)

    max_rcs_dB = calculate_theoretical_RCS(a, lamb, theta_ideal, phi_ideal)

    # create theta, phi meshgrid
    thetas = np.linspace(theta_ideal - (np.pi/4), theta_ideal + (np.pi/4), 100)
    phis = np.linspace(phi_ideal - (np.pi/4), phi_ideal + (np.pi/4), 100)
    rcs_var_theta = []
    for theta in thetas:
        rcs_var_theta.append(calculate_theoretical_RCS(a, lamb, theta, phi_ideal))
    rcs_var_phi = []
    for phi in phis:
        rcs_var_phi.append(calculate_theoretical_RCS(a, lamb, theta_ideal, phi))

    # find 3 dB
    rcs_3dB = max_rcs_dB - 3
    theta_idx = find_closest_arg(rcs_var_theta, rcs_3dB)
    phi_idx = find_closest_arg(rcs_var_phi, rcs_3dB)

    if show_plots:
        fig, ax = plt.subplots(1, 2, sharex=False, sharey=False)
        ax[0].set_title('RCS as a function of Elevation Angle')
        ax[0].plot(np.rad2deg(thetas), rcs_var_theta)
        ax[0].scatter(np.rad2deg(thetas), rcs_var_theta)
        ax[0].scatter(np.rad2deg(theta_ideal), max_rcs_dB)
        ax[0].axvspan(np.rad2deg(thetas[theta_idx[0]]), np.rad2deg(thetas[theta_idx[1]]), color='lightskyblue', alpha=.25)
        #plt.annotate('Max RCS at theta=54.74', (np.rad2deg(theta_ideal)+1, max_rcs_dBsm))
        ax[0].set_xlabel('Elevation Angle (deg)')
        ax[0].set_ylabel('RCS (dB)')

        #plt.subplot(122)
        ax[1].set_title('RCS as a function of Azimuth Angle')
        ax[1].plot(np.rad2deg(phis), rcs_var_phi)
        ax[1].scatter(np.rad2deg(phis), rcs_var_phi)
        ax[1].scatter(np.rad2deg(phi_ideal), max_rcs_dB)
        ax[1].axvspan(np.rad2deg(phis[phi_idx[0]]), np.rad2deg(phis[phi_idx[1]]), color='lightskyblue', alpha=.25)
        ax[1].set_xlabel('Azimuth Angle (deg)')
        ax[1].set_ylabel('RCS (dB)')
        plt.show()

        thetas = np.linspace(theta_ideal - (np.pi / 8), theta_ideal + (np.pi / 8), 100)
        phis = np.linspace(phi_ideal - (np.pi / 8), phi_ideal + (np.pi / 8), 100)
        Theta, Phi = np.meshgrid(thetas, phis)
        RCS_vals = np.zeros(Theta.shape)
        for i in range(Theta.shape[0]):
            for j in range(Theta.shape[1]):
                _theta = Theta[i][j]
                _phi = Phi[i][j]
                RCS_vals[i][j] = calculate_theoretical_RCS(a, _theta, _phi)

        fig = plt.figure()
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        ax.plot_surface(np.rad2deg(Theta), np.rad2deg(Phi), RCS_vals, rstride=1, cstride=1, cmap='viridis', alpha=.5)
        #ax.scatter(w_pred, b_pred, L_min, color='black', s=24)
        #ax.text(w_pred, b_pred, L_min, "Minimum Loss", color='black')
        ax.set_xlabel('Elevation Angle (deg)')
        ax.set_ylabel('Azimuth Angle (deg)')
        ax.set_zlabel('RCM (dBsm)')
        plt.show()

    return max_rcs_dB


def get_all_rcs(slcs, coords, cr='cr1'):
    df = pd.DataFrame(columns=['slc', 'rcs_theory', 'rcs_measured', 'rcs_imle'])

    for slc in slcs:
        par = SLC_Par(slc + '.par')
        frequency = par['radar_frequency']
        wavelength = c / frequency
        range_pixel_spacing = par['range_pixel_spacing']
        azimuth_pixel_spacing = par['azimuth_pixel_spacing']
        incidence_angle = np.deg2rad(par['incidence_angle'])
        Apx = (azimuth_pixel_spacing * range_pixel_spacing) / np.sin(incidence_angle)

        out = phase_error(a, wavelength, Apx, incidence_angle, slc, par.dim, coords, cr=cr)
        out = pd.DataFrame(out)
        df = pd.concat([df, out], ignore_index=True)

    df.to_csv(working_dir + sub_dir + 'rcs_' + cr + '.csv')



if __name__ == "__main__":

    #max_RCS_dB = model_RCS_deviations(a, wavelength)

    #print("Max RCS: {:.4f} dB".format(max_RCS_dB))
    #print("Pixel Area: {:.4f} m2".format(Apx))

    target_cr = 'cr5'
    if target_cr in ['cr1', 'cr2']:
        cr_file = working_dir + sub_dir + 'slc_phase_site1.csv'
    elif target_cr in ['cr3', 'cr4', 'cr5', 'cr6']:
        cr_file = working_dir + sub_dir + 'slc_phase_site2.csv'

    coords = pd.read_csv(cr_file)
    slcs = stack.slc_list()
    pt_slcs = ptarg_rslcs(target_cr, slcs)

    get_all_rcs(pt_slcs, coords, target_cr)
    rcs_measurements = pd.read_csv(working_dir + sub_dir + 'rcs_' + target_cr + '.csv')
    rcs_measurements['date'] = [pd.to_datetime(val, format='%Y%m%d') for val in rcs_measurements['slc'].values]
    print(rcs_measurements)
    plt.scatter(rcs_measurements['date'], rcs_measurements['rcs_theory'], label='Theoretical')
    plt.scatter(rcs_measurements['date'], rcs_measurements['rcs_measured'], label='Measured')
    plt.scatter(rcs_measurements['date'], rcs_measurements['rcs_imle'], label='Calculated from Ptarg')
    plt.ylabel('RCS (dB)')
    plt.title(f"Corner Reflector RCS\n{target_cr}")
    plt.legend()
    plt.show()

    #Todo: put error bars on insar tilt plots



