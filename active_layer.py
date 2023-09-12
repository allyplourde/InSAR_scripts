from sarlab.met import parse_ec_dir#, get_param
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
from scipy.interpolate import interp1d
from sarlab.gammax import name_root
import datetime
import glob
import os
import re
from cr_phase_to_deformation import *
from sarlab.gammax import SLC_stack, readBin
import pandas as pd

### RS2 ###
working_dir = '/local-scratch/users/aplourde/HDS/projects/southern_ITH/RS2_U76_D/'
sub_dir = 'crop_sites/'; master = '20180827'

### TSX ###
"""
working_dir = '/local-scratch/users/aplourde/TSX/'
sub_dir = 'crop_sites/'; master = '20190726_HH'
"""

ingest_cfg = {'password': 'S1m0nF7@SER', 'polarizations': 'HH'}
stack = SLC_stack(dirname=working_dir + sub_dir,name='inuvik_RS2_U76_D', master=master, looks_hr=(2,3), looks_lr=(12,18), multiprocess=True, rdcdem_refine=True, skipmode='None', ingest_cfg=ingest_cfg)

cr_loc = working_dir + sub_dir + 'cr_loc.txt'

def active_layer_def_template(dir, rslc_list, out_fname, temp_offset = 4., alpha=1, show_plot=False):
    filt_wid = 31
    filt_wid = 61

    #read air temperature history
    dict = parse_ec_dir(dir, freq='daily')

    #throw out nans at end of record
    dict['Mean Temp (°C)'] = dict['Mean Temp (°C)'][0:4084]
    dict['Date/Time'] = dict['Date/Time'][0:4084]

    temp_air = np.asarray(dict['Mean Temp (°C)'])
    npts = len(temp_air)
    good_mask = np.isfinite(temp_air)
    temp_air = interp1d(np.arange(npts)[good_mask], temp_air[good_mask])(np.arange(npts))

    #apply emperical offset to infer ground temperature
    temp_gnd = temp_air + temp_offset

    #smooth to enable estimation of stable zero crossing times
    temp_gnd_sm = savgol_filter(temp_gnd, filt_wid, 1)
    npts = len(temp_gnd_sm)

    #get cumulative freezing/thawing degree days from zero crossing
    temp_cum = np.zeros(npts)
    for ii in np.arange(1, npts):
        temp_cum[ii]= temp_cum[ii-1]+temp_gnd[ii-1]
        if np.sign(temp_gnd_sm[ii-1]) != np.sign(temp_gnd_sm[ii]):
            temp_cum[ii] = 0

    #now get deformation curve (assumed proportional to freezing depth)
    deform = np.zeros(npts)
    ref_current =0
    valid_range_start_idx = -1
    for ii in np.arange(1, npts):
        if np.sign(temp_gnd_sm[ii-1]) != np.sign(temp_gnd_sm[ii]):
            ref_current = deform[ii-1]
            if ((temp_gnd_sm[ii]) > 0 and (valid_range_start_idx < 0)):
                valid_range_start_idx = ii

        deform[ii] = ref_current -1*np.sqrt(np.abs(temp_cum[ii]))*np.sign(temp_cum[ii]) * alpha

        # deformation is upper bounded
        deform[ii] = min(deform[ii], 0)

    if show_plot:
        plt.subplot(311)
        plt.title('Ground Temperature')
        plt.plot(temp_gnd)
        plt.subplot(312)
        plt.title('Ground Temperature (Smoothed)')
        plt.plot(temp_gnd_sm)
        #plt.plot(temp_cum)
        plt.subplot(313)
        plt.title('Deformation Curve')
        plt.plot(deform)
        #plt.savefig('active_layer.png')
        #plt.show()

    #build interpolation table
    dts = [(dt -  dict['Date/Time'][0]).total_seconds() for dt in dict['Date/Time']]
    deform_table = interp1d(dts[valid_range_start_idx:], deform[valid_range_start_idx:])
    rslc_roots = [name_root(rslc_fname) for rslc_fname in rslc_list]
    dts_rslc = [(datetime.datetime.strptime(rslc_root, '%Y%m%d')-dict['Date/Time'][0]).total_seconds() for rslc_root in rslc_roots]

    deform_rslc = deform_table(dts_rslc)

    with open(out_fname, 'w') as file:
        for ii in np.arange(len(rslc_list)):
            file.write('%s %.3f\n' % (rslc_roots[ii], deform_rslc[ii]))
        file.close()
    return rslc_roots, deform_rslc


def read_active_layer_template(fname):
    dates=[]
    vals = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            dates += [line.split()[0]]
            vals += [float(line.split()[1])]
    vals = np.asarray(vals)
    return dates, vals


def download_env_canada_met_data(met_dir, start_year, end_year, freq='daily'):
    cdir = os.getcwd()
    os.chdir(met_dir)
    start_year = str(start_year)
    end_year = str(end_year)
    if freq == 'daily':
        time_frame = '2'
    elif freq == 'hourly':
        time_frame = '1'

    query = 'for year in `seq %s %s`; do for month in `seq 1 1`;do wget --content-disposition "https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=41883&Year=${year}&Month=${month}&Day=$14&timeframe=%s&submit= Download+Data"; done; done' % (start_year, end_year, time_frame)

    os.system(query)
    os.system('rm *csv.*')  # delete duplicates

    os.chdir(cdir)


def active_layer_phase(def_model, show_plot=False):
    dates = def_model[0]
    vals = def_model[1]

    phis = []
    cphis = []
    rel_phis = []

    prev_phi = deformation_to_phase(vals[0])
    for val in vals:
        cphi, phi = deformation_to_phase(val)
        phis.append(phi)
        cphis.append(cphi)
        rel_phis.append(phi-prev_phi)
        prev_phi =phi

    for i in range(len(dates)):
        if dates[i] == '20180827':
            zero = phis[i]
    phis = phis-zero
    cphis = np.exp(1j*(np.angle(cphis) - zero))

    dates = pd.to_datetime(dates)
    if show_plot:
        ax1.plot(dates, np.angle(cphis), '-o', label='Wrapped Model', color='green')
        ax1.plot(dates, phis, '-o', label='Model', color='grey')

    master = dates[:-1]
    slave = dates[1:]
    if show_plot:
        for i in range(len(master)):
            #plt.plot([master[i], slave[i]], [0, rel_phis[1:][i]])
            pass

    return pd.DataFrame(data={'master': master, 'slave': slave, 'model': cphis[1:], 'rel_phi': rel_phis[1:]})


def get_cr_error_bar(dates, cr):

    err = []
    rcs_df = pd.read_csv(working_dir + 'full_scene/rcs_' + cr + '.csv')
    for date in dates:
        row = rcs_df[rcs_df['slc'] == int(date.strftime('%Y%m%d'))]
        err.append(row['phi_err_measured'].values[0])
    return np.array(err)



def test_model(rslcs, par, model, itab, method='rslc', crs=['cr1', 'cr2'], label=None, plot=False, sub_flt = False, annulus = None, diff_dir='diff_fr/'):

    rslc_diff = None
    file_dir = working_dir + sub_dir + 'rslc/'

    if os.path.exists(file_dir + 'cr_diff_cm.csv') and False:
        rslc_diff = pd.read_csv(file_dir + 'cr_diff_cm.csv')
        rslc_diff[crs[0] + '_diff'] = [np.complex(strval) for strval in rslc_diff[crs[0] + '_diff']]
        rslc_diff[crs[1] + '_diff'] = [np.complex(strval) for strval in rslc_diff[crs[1] + '_diff']]
    else:
        if method in ['.flt', '.diff', '.diff.adf']:    # use interferometric results

            # grab files from directory containing common master interferograms
            files = glob.glob(working_dir + sub_dir + diff_dir + '*' + method)

            ### Seperate CM ###
            """
            if 'cr1' in crs:
                itab = working_dir + sub_dir + 'itab_cm_site1'
            elif 'cr3' in crs:
                itab = working_dir + sub_dir + 'itab_cm_site2'
            """

            ### Single CM ###
            itab = working_dir + sub_dir + itab
            """
            if "TSX" in working_dir:
                itab = working_dir + sub_dir + 'itab_cm_20190726_reduced'
            else:
                itab = working_dir + sub_dir + 'itab_cm_20190729'
            itab = working_dir + sub_dir + 'itab_cm_site1'
            """
            RSLC_tab = working_dir + sub_dir + 'RSLC_tab'
            diffs = get_itab_diffs(files, itab, RSLC_tab)

            cra_df = cr_phase(crs[0], cr_loc, diffs, par.dim, dtype='complex64', annulus=annulus)
            crf_df = cr_phase(crs[1], cr_loc, diffs, par.dim, dtype='complex64', annulus=annulus)

            cra_df['crf_diff'] = crf_df['delta_phi']
            diff = cra_df.rename(columns={'delta_phi': 'cra_diff'})

        elif 'hds' in method:
            # grab files from hds svd directory

            if 'refine' in method:
                files = glob.glob('/local-scratch/users/aplourde/quadzilla_desktop/def_rerefine/*.vert.2d')
            if 'def_dhrm' in method:
                files = glob.glob('/local-scratch/users/aplourde/quadzilla_desktop/def_dhrm/*.vert.2d')
            else:
                files = glob.glob(working_dir + sub_dir + 'hds/svd/*svd.2d')

            cra_df = cr_svd(crs[0], cr_loc, files, par.dim, dtype='float32')
            crf_df = cr_svd(crs[1], cr_loc, files, par.dim, dtype='float32')
            cra_df['crf_diff'] = crf_df['def_mm']
            diff = cra_df.rename(columns={'def_mm': 'cra_diff'})
            diff['cr_diff'] = diff['cra_diff'] - diff['crf_diff']
            cphi, phi = deformation_to_phase(diff['cr_diff'])

            #
            diff = diff[diff['slave'] >= pd.to_datetime('20180920')]

            ax2.scatter(diff['slave'], diff['cr_diff'])
            ax2.plot(diff['slave'], diff['cr_diff'], label=method)
            ax2.plot([pd.to_datetime('20180827'), pd.to_datetime('20201201')], [0, 0], color='black')

            return

        else: # generate interferometric phase from slc

            cra_df = mk_diff_cm(rslcs, par, crs[0], cr_loc, file_dir, phase_from=method, sub_flt=sub_flt)
            crf_df = mk_diff_cm(rslcs, par, crs[1], cr_loc, file_dir, phase_from=method, sub_flt=sub_flt)

            if sub_flt:
                cra_df['crf_diff'] = crf_df['flt_phi']
                diff = cra_df.rename(columns={'flt_phi': 'cra_diff'})
            else:
                cra_df['crf_diff'] = crf_df['delta_phi']
                diff = cra_df.rename(columns={'delta_phi': 'cra_diff'})

            #diff.to_csv(file_dir + 'cr_diff_cm.csv')

    diff['master'] = pd.to_datetime(diff['master'])
    diff['slave'] = pd.to_datetime(diff['slave'])

    diff = pd.merge(diff, model, on=['slave'], how='left')
    if 'master_x' in list(diff):
        diff = diff.rename(columns={'master_x': 'master'})

    diff = diff[diff['slave'] >= pd.to_datetime('20180920')]

    diff['cr_diff'] = diff['cra_diff']*np.conj(diff['crf_diff'])


    mean = np.mean(diff['cr_diff'][diff['slave']>='20180920'])

    #diff['demean'] = diff['cr_diff']*np.conj(mean)
    #diff['demod'] = diff['cr_diff'] * np.conj(diff['model'])  # subtract model
    diff['demod'] = np.angle(diff['cr_diff']) - np.angle(diff['model'])
    #diff['demod'] = diff['cr_diff'] * diff['model']  # add model
    #diff['demod'] = diff['demean'] * diff['model']

    if plot:

        ### Error Bars ###
        """
        err_s1 = get_cr_error_bar(diff['slave'], crs[0])
        err_m1 = get_cr_error_bar(diff['master'], crs[0])
        err1 = np.sqrt(err_s1**2 + err_m1**2)

        err_s2 = get_cr_error_bar(diff['slave'], crs[1])
        err_m2 = get_cr_error_bar(diff['master'], crs[1])
        err2 = np.sqrt(err_s2**2 + err_m2**2)

        err = np.sqrt(err1**2 + err2**2)
        ave_err = np.mean(err)
        """
        err = np.zeros(len(diff['slave']))

        if label is None:
            if method == 'ptr_par':
                label = 'ptarg'
            else:
                label = method
            if sub_flt:
                label = label + ' - flt'

        # DATA #
        new_row = {'master': diff['master'].values[0], 'slave': diff['master'].values[0], 'cr_diff':0}
        diff = diff.append(new_row, ignore_index=True)
        diff = diff.sort_values(by=['slave'])

        err = np.insert(err, 0, 0)
        print(err.shape)

        ### Unwrap ###
        """
        data = np.unwrap(np.angle(diff['cr_diff']), discont=3.5)
        if 'cr1' in crs:
            data = data + 6.05
        """
        data = np.angle(diff['cr_diff'])
        if 'lf' in itab:
            data = np.angle(np.exp(1j*data.cumsum()))
        ax1.errorbar(diff['slave'], data, fmt='-o', yerr=err, label=label)
        #ax1.scatter(diff['slave'], diff['demod'], label='demod', color='red')
        # ax1.scatter(diff['slave'], np.angle(diff['demod']), label='demod', color='red')
        #ax1.scatter(diff['slave'], np.angle(diff['demean']), label='demean', color='lightblue')


        # META #
        ax1.plot([pd.to_datetime('20180827'), pd.to_datetime('20201201')], [np.pi, np.pi], '--', color='black')
        ax1.plot([pd.to_datetime('20180827'), pd.to_datetime('20201201')], [0,0], color='black')
        ax1.plot([pd.to_datetime('20180827'), pd.to_datetime('20201201')], [-np.pi, -np.pi], '--', color='black')
        #ax1.plot([pd.to_datetime('20180827'),pd.to_datetime('20180827')], [-3*np.pi, 3*np.pi], '--', color='black')

    #ave_residual = np.sum(np.abs(np.angle(diff['demod'][diff['master'] >= pd.to_datetime('20180827')])))
    #ave_residual = np.sum(np.abs(diff['demod'][diff['slave'] >= pd.to_datetime('20180920')]))
    ave_residual = np.sum(np.abs(diff['demod']))
    #ave_residual = np.nanmean(np.abs(diff['demod'][diff['slave'] >= pd.to_datetime('20180920')]))

    return ave_residual


def tilt_demod(rslcs, end_date = '20201201', show_plot = False):

    tilt_site1 = pd.read_csv("/local-scratch/users/aplourde/field_data/site_1/site_1_inclinometer_processed.csv")
    tilt_site1.index = pd.to_datetime(tilt_site1[list(tilt_site1)[0]])
    if end_date is not None:
        tilt_site1 = tilt_site1[tilt_site1.index < pd.to_datetime(end_date)]
    tilt_phi = []
    for val in tilt_site1.dh1_mm:
        cphi, phi = deformation_to_phase(val / 1000)
        tilt_phi.append(phi)
    filt_wid = 15
    tilt_phi_sm = savgol_filter(tilt_phi, filt_wid, 1)
    tilt_sm = savgol_filter(tilt_site1.dh1_mm, filt_wid, 1)

    tilt_cphi_sm = np.exp(1j*(tilt_phi_sm))

    if show_plot or True:
        ax2.plot(tilt_site1.index, tilt_sm, color='black')
        ax1.plot(tilt_site1.index, tilt_phi_sm, color='black')
        #ax1.plot(tilt_site1.index, np.angle(tilt_cphi_sm), color='gray', label='wrapped tilt')

    rslc_tilt = []
    rslc_dates = []
    for rslc in rslcs:
        root = os.path.basename(rslc)
        date = pd.to_datetime(re.findall(r"\d{8}", root)[0])
        idx = np.argwhere(tilt_site1.index == date)
        if idx:
            rslc_tilt.append(tilt_cphi_sm[idx][0][0])
            rslc_dates.append(date)

    master = rslc_dates[:-1]
    master = np.insert(master, 0, pd.to_datetime('20180827'))
    slave = rslc_dates
    rslc_tilt = rslc_tilt

    if show_plot:
        ax1.scatter(slave, np.angle(rslc_tilt), color='gray')

    return pd.DataFrame(data = {'master':master, 'slave':slave, 'model':rslc_tilt})


if __name__ == "__main__":

    sar_dir = '/local-scratch/users/aplourde/HDS/projects/southern_ITH/RS2_U76_D/'
    met_dir = '/local-scratch/users/aplourde/met_data/env_canada/'
    out_fname = sar_dir + 'active_layer.txt'
    rslc_list = stack.rslc_list()
    master_par = stack.master_slc_par
    rslc_list = ptarg_rslcs('cr1', rslc_list)

    fig, ax1 = plt.subplots()
    plt.title('Deformation Timeseries')
    ax2 = ax1.twinx()

    site1 = ['cr1', 'cr2']
    site2 = ['cr3', 'cr4']
    site3 = ['cr3', 'cr6']
    site = site3

    #download_env_canada_met_data(met_dir, start_year=2014, end_year=2021) #TODO STATION
    #def_model = active_layer_def_template(met_dir, rslc_list, out_fname, temp_offset=4., alpha=0.001, show_plot=True)
    #active_layer_def_template(met_dir, rslc_list, out_fname, temp_offset=4., alpha=1.)

    """
    alphas = np.arange(0.0001, 0.0015, 0.0001)
    best_res = 100
    best_alpha = 0.001
    best_model = None
    for alpha in alphas:
        def_model = active_layer_def_template(met_dir, rslc_list, out_fname, temp_offset=3., alpha=alpha)
        phi_model = active_layer_phase(def_model, show_plot=1)
        res = test_model(rslc_list, master_par, phi_model, method='.diff', crs=site1, label='mod')
        if res < best_res:
            best_res = res
            best_alpha = alpha
            best_model = phi_model
    print(best_alpha, best_res)
    """

    show_model = False
    show_residuals = True
    """
    def_model = active_layer_def_template(met_dir, rslc_list, out_fname, temp_offset=3., alpha=best_alpha, show_plot=show_model)
    phi_model = active_layer_phase(def_model, show_plot=1)

    res = test_model(rslc_list, master_par, best_model, method='.diff', crs=site1, label='site1', plot=show_residuals)
    print(f"Residual: {res}")
    """

    tilt_model = tilt_demod(rslc_list)
    ### SLC Compare ###
    """
    res = test_model(rslc_list, master_par, tilt_model, method='ptr_par', plot=show_residuals)
    res = test_model(rslc_list, master_par, tilt_model, method='rslc_pixel', plot=show_residuals)
    res = test_model(rslc_list, master_par, tilt_model, method='slc_pixel', plot=show_residuals)
    """

    ### Flat Earth Compare ###
    """
    res = test_model(rslc_list, master_par, tilt_model, method='ptr_par', plot=show_residuals, sub_flt = 1)
    res = test_model(rslc_list, master_par, tilt_model, method='rslc_pixel', plot=show_residuals, sub_flt = 1)
    res = test_model(rslc_list, master_par, tilt_model, method='.flt', plot=show_residuals)
    """

    ### Diff Compare ###
    """
    res = test_model(rslc_list, master_par, tilt_model, method='.flt', crs=site, plot=show_residuals)
    res = test_model(rslc_list, master_par, tilt_model, method='.diff', crs=site, plot=show_residuals)
    #res = test_model(rslc_list, master_par, tilt_model, method='.diff', crs=site, plot=show_residuals, annulus = 10)
    res = test_model(rslc_list, master_par, tilt_model, method='.diff.adf', crs=site, plot=show_residuals)
    """

    ### Site Compare ###
    """
    res = test_model(rslc_list, master_par, tilt_model, method='.diff', crs=site1, label='RS2 - site1', plot=show_residuals)
    res = test_model(rslc_list, master_par, tilt_model, method='.diff', crs=site2, label='RS2 - site2', plot=show_residuals)
    res = test_model(rslc_list, master_par, tilt_model, method='.diff', crs=site3, label='RS2 - site3', plot=show_residuals)

    ### TSX ###
    working_dir = '/local-scratch/users/aplourde/TSX/'
    sub_dir = 'crop_sites/';
    master = '20190726_HH'
    cr_loc = working_dir + sub_dir + 'cr_loc.txt'
    stack = SLC_stack(dirname=working_dir + sub_dir, name='inuvik_RS2_U76_D', master=master, looks_hr=(2, 3),
                      looks_lr=(12, 18), multiprocess=True, rdcdem_refine=True, skipmode='None', ingest_cfg=ingest_cfg)
    rslc_list = stack.rslc_list()
    master_par = stack.master_slc_par
    rslc_list = ptarg_rslcs('cr1', rslc_list)

    res = test_model(rslc_list, master_par, tilt_model, method='.diff', crs=site1, label='TSX - site1', plot=show_residuals)
    res = test_model(rslc_list, master_par, tilt_model, method='.diff', crs=site2, label='TSX - site2', plot=show_residuals)
    res = test_model(rslc_list, master_par, tilt_model, method='.diff', crs=site3, label='TSX - site3', plot=show_residuals)
    #res = test_model(rslc_list, master_par, tilt_model, method='.diff', crs=['cr3', 'cr5'], label='snow', plot=show_residuals)
    """


    ### HDS Compare ###
    """
    res = test_model(rslc_list, master_par, tilt_model, method='.diff', plot=show_residuals)
    res = test_model(rslc_list, master_par, tilt_model, method='hds', plot=show_residuals)
    res = test_model(rslc_list, master_par, tilt_model, method='hds - refine', plot=show_residuals)
    res = test_model(rslc_list, master_par, tilt_model, method='hds - def_dhrm', plot=show_residuals)
    """


    ### Baseline Compare ###
    res = test_model(rslc_list, master_par, tilt_model, itab='itab_cm_site1', method='.diff', crs=site1, label='RS2 - site1', plot=show_residuals)
    sub_dir = 'new_crop_sites/'
    res = test_model(rslc_list, master_par, tilt_model, itab='itab_lf', method='.diff', crs=site1, label='RS2 - site1', plot=show_residuals, diff_dir='diff_2/')
    res = test_model(rslc_list, master_par, tilt_model, itab='itab_lf', method='.diff', crs=site1, label='RS2 - site1', plot=show_residuals, diff_dir='diff_3/')
    res = test_model(rslc_list, master_par, tilt_model, itab='itab_lf', method='.diff', crs=site1, label='RS2 - site1', plot=show_residuals, diff_dir='diff_8/')
    res = test_model(rslc_list, master_par, tilt_model, itab='itab_lf', method='.diff', crs=site1, label='RS2 - site1', plot=show_residuals, diff_dir='diff_9/')
    res = test_model(rslc_list, master_par, tilt_model, itab='itab_lf', method='.diff', crs=site1, label='RS2 - site1', plot=show_residuals, diff_dir='diff/')

    ax1.set_ylabel('Phase (rad)')
    ax2.set_ylabel('Deformation (mm)')
    ax1.legend()
    plt.show()

