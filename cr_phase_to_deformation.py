from sarlab.gammax import *
import pandas as pd
import numpy as np
import os
import re
import glob
from scipy.constants import c
from ptarg import ptarg_rslcs
from scipy.stats import linregress
start_date = pd.to_datetime('20180827')
end_date = pd.to_datetime('20220101')

#working_dir = "/local-scratch/users/aplourde/RS2/HDS/projects/jayson_hds/test_files/"
#working_dir = '/local-scratch/users/aplourde/HDS/projects/southern_ITH/RS2_U76_D/crop_sites/hds/'
working_dir = '/local-scratch/users/aplourde/HDS/projects/southern_ITH/RS2_U76_D/'
sub_dir = 'crop_sites/'
#sub_dir = 'full_scene/'
cr_file = working_dir + sub_dir + 'cr_loc.txt'
#pt_file = working_dir + 'pt.rendermode0.ras'
#DIM = [996, 2592] #jayson
#dim = [1498, 2200] #my summer sites
master = '20170808'

ingest_cfg = {'password': 'S1m0nF7@SER', 'polarizations': 'HH'}
stack = SLC_stack(dirname=working_dir + sub_dir,name='inuvik_RS2_U76_D', master=master, looks_hr=(2,3), looks_lr=(12,18), multiprocess=True, rdcdem_refine=True, skipmode='None', ingest_cfg=ingest_cfg)


def unpack_cr_coords(cr_file):
    crs = pd.read_csv(cr_file)

    coords = {'cr1': (crs[crs['cr'] == 1]['x_loc'].values[0], crs[crs['cr'] == 1]['y_loc'].values[0]),
              'cr2': (crs[crs['cr'] == 2]['x_loc'].values[0], crs[crs['cr'] == 2]['y_loc'].values[0]),
              'cr3': (crs[crs['cr'] == 3]['x_loc'].values[0], crs[crs['cr'] == 3]['y_loc'].values[0]),
              'cr4': (crs[crs['cr'] == 4]['x_loc'].values[0], crs[crs['cr'] == 4]['y_loc'].values[0]),
              'cr5': (crs[crs['cr'] == 5]['x_loc'].values[0], crs[crs['cr'] == 5]['y_loc'].values[0]),
              'cr6': (crs[crs['cr'] == 6]['x_loc'].values[0], crs[crs['cr'] == 6]['y_loc'].values[0])}

    return coords


def visualize_file(file, dtype, range, visualize_sites=False, savefig=False):

    basename = os.path.basename(file)
    im = readBin(file, DIM, dtype)
    if 'complex' in dtype:
        im = np.angle(im)/360

    coords = unpack_cr_coords(cr_file)

    if visualize_sites:
        m_date = pd.to_datetime(basename.split('.')[0].split('_')[0])
        site1 = np.mean([coords['cr1'], coords['cr2']], axis=0).astype(int)
        site2 = np.mean([coords['cr3'], coords['cr4'], coords['cr5'], coords['cr6']], axis=0).astype(int)

        w = 25
        s = 2

        if m_date >= pd.to_datetime('20180827'):
            plt.title('Site 1')
            plt.imshow(im.T*13*13, cmap=cm.Spectral, vmax=range[1], vmin=range[0])
            plt.colorbar()
            plt.scatter(coords['cr1'][0], coords['cr1'][1], color='red', s=s)
            plt.scatter(coords['cr2'][0], coords['cr2'][1], color='red', s=s)
            plt.xlim(site1[0]-w, site1[0]+w)
            plt.ylim(site1[1]+w, site1[1]-w)
            plt.suptitle(basename)
            if savefig:
                plt.savefig('ifg_sites/' + basename + 'site1.png', orientation='landscape')
                plt.clf()
                plt.close()
            else:
                plt.show()

        if m_date >= pd.to_datetime('20190729'):
            plt.title('Site 2')
            plt.imshow(im.T, cmap=cm.Spectral)
            plt.colorbar()
            plt.scatter(coords['cr3'][0], coords['cr3'][1], color='red', s=s)
            plt.scatter(coords['cr4'][0], coords['cr4'][1], color='red', s=s)
            plt.scatter(coords['cr5'][0], coords['cr5'][1], color='red', s=s)
            plt.scatter(coords['cr6'][0], coords['cr6'][1], color='red', s=s)
            plt.xlim(site2[0]-w, site2[0]+w)
            plt.ylim(site2[1]+w, site2[1]-w)
            plt.suptitle(basename)
            if savefig:
                plt.savefig('ifg_sites/' + basename + 'site2.png', orientation='landscape')
                plt.clf()
                plt.close()
            else:
                plt.show()
    else:
        plt.imshow(im.T * 11 * 17, cmap=cm.Greys_r, vmax=range[1], vmin=range[0])
        plt.colorbar()
        plt.scatter(coords['cr1'][0], coords['cr1'][1], color='red', s=4)
        plt.scatter(coords['cr2'][0], coords['cr2'][1], color='red', s=4)
        plt.scatter(coords['cr3'][0], coords['cr3'][1], color='red', s=4)
        plt.scatter(coords['cr4'][0], coords['cr4'][1], color='red', s=4)
        plt.scatter(coords['cr5'][0], coords['cr5'][1], color='red', s=4)
        plt.scatter(coords['cr6'][0], coords['cr6'][1], color='red', s=4)
        plt.suptitle(basename)
        if savefig:
            plt.savefig('ifg_sites/' + basename + '.png', orientation='landscape')
        else:
            plt.show()
    plt.clf()
    plt.close()


def cr_rslc(rslcs, par, cr, cr_file, file_dir, phase_from='rslc_pixel'):
    rslc_phase = []
    for rslc in rslcs:
        basename = os.path.basename(rslc)
        date = basename.split('.')[0]

        if phase_from == 'rslc_pixel':
            im = readBin(file_dir + basename, par.dim, 'complex64')
            coords = unpack_cr_coords(cr_file)[cr]
            #im_phi = np.angle(im)
            cr_phi = im[coords[0], coords[1]]
            rslc_phase.append((date, np.asarray(cr_phi)))

        if phase_from == 'ptr_par':
            phi = np.deg2rad(np.loadtxt(file_dir + date + '_' + cr + '.rslc.ptr_par', usecols=[5]))
            #phi = np.angle(np.exp(1j * phi))  # wrapped phase?
            rslc_phase.append((date, np.asarray(np.exp(1j*phi))))

        if phase_from == 'annulus':
            blocksize = 10
            im = readBin(file_dir + basename, par.dim, 'complex64')
            coords = unpack_cr_coords(cr_file)[cr]
            #im_phi = np.angle(im)
            annulus = im[coords[0] - blocksize:coords[0] + blocksize - 1,
                      coords[1] - blocksize:coords[1] + blocksize - 1]
            annulus[blocksize - 2:blocksize + 1, blocksize - 2:blocksize + 1] = 0
            # plt.imshow(cr1_annulus)
            # plt.show()
            # break
            rslc_phase.append((date, np.asarray(annulus)))

        if phase_from == 'slc_pixel':
            slc_df = pd.read_csv('/local-scratch/users/aplourde/HDS/projects/southern_ITH/RS2_U76_D/full_scene/slc_phase_site1.csv')
            phi = slc_df['phi_' + cr[-1]][slc_df['Date'] == int(date)].values
            if len(phi) == 0:
                phi = 0
            else:
                phi = np.exp(1j*phi[0])
            rslc_phase.append((date, np.asarray(phi)))

    return rslc_phase


def extract_annulus(im, center, inner_radius, outer_radius, show_plot=False):
    """
    Extracts the annulus around a pixel in an image.
    :param im: (numpy.ndarray) Input image
    :param center: (tuple) Coordinates of the center pixel.
    :param inner_radius: (int) Inner radius of the annulus (in pixel units)
    :param outer_radius: (int) Outer radius of the annulus (in pixel units)
    :Returns:
    numpy.ndarray: Annulus image
    """
    # Get the coordinates of the annulus pixels
    x, y = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
    distance = np.sqrt((x-center[1])**2 + (y-center[0])**2)
    annulus_mask = np.logical_and(distance >= inner_radius, distance <= outer_radius)

    # Extract the annulus pixels
    annulus = np.zeros_like(im)
    annulus[annulus_mask] = im[annulus_mask]



    if show_plot:
        #plt.imshow(im.T)
        #plt.imshow(distance.T)
        #plt.imshow(annulus_mask.T)
        plt.imshow(annulus.T)
        #plt.scatter(center[0], center[1], color='red')
        #plt.colorbar()
        plt.show()

    return annulus, annulus_mask

def cr_phase(cr, cr_file, files, dim, dtype = 'float32', annulus=None, show_plot=False):
    # extract cr_phase from interferogram

    coords = unpack_cr_coords(cr_file)[cr]
    phi_cr = []

    for file in files:

        im = readBin(file, dim, dtype)

        root = os.path.basename(file)
        ifg = root.split('.')[0]

        #master, slave = ifg.split('_')
        dates = re.findall(r"\d{8}", root)

        master = pd.to_datetime(dates[0])
        slave = pd.to_datetime(dates[1])

        if annulus is not None:

            annulus_phi, mask = extract_annulus(im, coords, 2, annulus, show_plot=show_plot)
            delta_phi = np.mean(annulus_phi[mask])

        else:

            delta_phi = im[coords[0], coords[1]]

        #### COMMON MASTER 20190729 ####
        """
        if "TSX" in file:
            new_master = pd.to_datetime('20190726')
        else:
            new_master = pd.to_datetime('20190729')
        if master == new_master:
            slave = slave
        elif slave == new_master:
            slave = master
            delta_phi = np.conj(delta_phi)
        master = new_master
        """

        phi_cr.append((master, slave, delta_phi))


        #double check coords
        if show_plot:
            #im[im == 0] = np.nan
            im = readBin(working_dir + sub_dir + 'rmli_1_1/rmli_1_1.ave', dim, 'float32')
            plt.imshow(im.T, cmap=cm.Greys_r) #[cr1_x-100:cr1_x+100, cr1_y-100:cr1_y+100]
            plt.scatter(coords[0], coords[1], color='red')
            plt.show()
            break

    cr_df = pd.DataFrame(data=phi_cr, columns=['master', 'slave', 'delta_phi'])
    cr_df = cr_df.sort_values(by='slave')
    cr_df = cr_df.sort_values(by='slave')
    return cr_df


def cr_svd(cr, cr_file, files, dim, dtype='float32'):
    # extract cr_phase from interferogram
    coords = unpack_cr_coords(cr_file)[cr]

    def_cr = []

    for file in files:

        im = readBin(file, dim, dtype)

        base = os.path.basename(file)
        root = base.split('.')[0]
        slave = pd.to_datetime(root)

        def_mm = im[coords[0], coords[1]] * 10 #* 1000
        def_cr.append((slave, def_mm))


    cr_df = pd.DataFrame(data=def_cr, columns=['slave', 'def_mm'])
    cr_df = cr_df.sort_values(by='slave')
    cr_df = cr_df.reset_index(drop=True)
    master = cr_df['slave'].iloc[0]
    cr_df['master'] = master

    return cr_df


def mk_diff_cm(rslcs, par, cr, cr_file, file_dir, phase_from='rslc_pixel', sub_flt=True):

    if phase_from == 'rslc_pixel':
        rslc_phase = cr_rslc(rslcs, par, cr, cr_file, file_dir, phase_from='rslc_pixel')
    elif phase_from == 'slc_pixel':
        rslc_phase = cr_rslc(rslcs, par, cr, cr_file, file_dir, phase_from='slc_pixel')
    elif phase_from == 'ptr_par':
        file_dir = file_dir + '../ptarg/'
        pt_rslcs = ptarg_rslcs(cr, rslcs)
        rslc_phase = cr_rslc(pt_rslcs, par, cr, cr_file, file_dir, phase_from='ptr_par')

    df = pd.DataFrame(rslc_phase, columns=['rslc', 'phi'])
    df = df.sort_values(by='rslc')

    phi = np.asarray(df['phi'])
    master_phi = df['phi'][df['rslc'] == '20180827'].values
    slave_phi = phi[1:]

    diff_phi = master_phi*np.conj(slave_phi)
    diff_phi = [val for val in diff_phi]

    master = df.rslc.values[0]
    slave = df.rslc.values[1:]

    diff_df = pd.DataFrame(data={'master': master, 'slave': slave, 'delta_phi': diff_phi})
    diff_df['master'] = pd.to_datetime(diff_df['master'])
    diff_df['slave'] = pd.to_datetime(diff_df['slave'])

    #remove flat earth phase
    if sub_flt:
        diff_df = subtract_flat_earth(file_dir, diff_df, cr, cr_file, par)

    return diff_df


def mk_diff(rslcs, par, cr, cr_file, file_dir, phase_from='rslc_pixel'):

    rslc_phase = cr_rslc(rslcs, par, cr, cr_file, file_dir, phase_from='rslc_pixel')

    df = pd.DataFrame(rslc_phase, columns=['rslc', 'phi'])
    df = df.sort_values(by='rslc')
    phi = np.asarray(df['phi'])
    master_phi = phi[:-1]
    slave_phi = phi[1:]
    #diff_phi = np.flip(np.conj(np.flip(phi)), axis=0)  # slc1 - slc2
    diff_phi = master_phi*np.conj(slave_phi)
    #dp = np.asarray([1j*phi for phi in diff_phi])
    #wrap = np.angle(np.exp(diff_phi))

    master = df.rslc.values[:-1]
    slave = df.rslc.values[1:]

    diff_df = pd.DataFrame(data={'master': master, 'slave': slave, 'delta_phi': diff_phi})
    diff_df['master'] = pd.to_datetime(diff_df['master'])
    diff_df['slave'] = pd.to_datetime(diff_df['slave'])

    return diff_df


def get_itab_diffs(diffs, itab, SLC_tab):
    itab = np.loadtxt(itab, dtype='int')
    stab = np.loadtxt(SLC_tab, dtype='str')

    itab_diffs = []

    for i in itab:
        mslc = os.path.basename(stab[i[0]-1][0]).split('.')[0]
        sslc = os.path.basename(stab[i[1]-1][0]).split('.')[0]

        for diff in diffs:
            if mslc in diff and sslc in diff:
                itab_diffs.append(diff)

    return itab_diffs


def convert_1d_to_2d(files, pt_file):
    # convert 1d hds files to 2d interferograms

    ras_pts = read_ras(pt_file)[0]

    pts = np.sum(ras_pts, axis=2) / (255*3)  # convert to binary

    n_points = int(np.sum(pts))

    guessing_n_points = True
    #n_points = 1931243
    for file in files:
        im_1d = readBin(file, (n_points, 1), 'float32')
        pts[np.nonzero(pts)] = im_1d.T
        im_2d = pts
        writeBin(file + '.2d', im_2d.T)


def phase_to_deformation(phase, frequency=5.4049992e+09, incidence_angle=np.deg2rad(26.9)):
    # estimate deformation in meters from interferometric phase

    # TODO individual par files!
    wavelength = c / frequency
    # sign convention: positive when phase shift is measured in upward direction
    def_m = phase * (wavelength/(4*np.pi))/np.cos(np.radians(incidence_angle))

    return def_m


def deformation_to_phase(deformation, frequency=5.4049992e+09, incidence_angle=np.deg2rad(26.9)):
    #frequency = 5.4049992e+09
    # TODO individual par files!
    wavelength = c / frequency
    #incidence_angle = 26.9
    # sign convention: positive when phase shift is measured in upward direction
    phi = deformation * np.cos(np.radians(incidence_angle)) * (4*np.pi) / wavelength
    cphi = np.exp(1j*(phi))

    return cphi, phi


def visualize_interferogram_stack(files, dtype):

    for file in files:
        visualize_file(file, dtype, [-np.pi, np.pi], visualize_sites=True, savefig=True)


def subtract_flat_earth(file_dir, diff_df, cr, cr_file, par):

    file_dir = file_dir + '../diff_fr/'
    flat_earth_models = glob.glob(file_dir + '*.flat_earth')
    flt_df = cr_phase(cr, cr_file, flat_earth_models, par.dim, dtype='complex64', show_plot=False)

    flt_df['slave'] = pd.to_datetime(flt_df['slave'])

    diff_df['flt_phi'] = np.zeros(len(diff_df))

    for i, date in diff_df['slave'].iteritems():

        phi = flt_df['delta_phi'][flt_df['slave'] == date].values
        flt_phi = diff_df['delta_phi'][i] * np.conj(phi)

        if len(flt_phi) > 0:
            diff_df['flt_phi'][i] = flt_phi[0]
        else:
            diff_df['flt_phi'][i] = np.nan

    return diff_df


def compare_slc_phase():
    slc_df = pd.read_csv('/local-scratch/users/aplourde/HDS/projects/southern_ITH/RS2_U76_D/full_scene/slc_phase_site1.csv')

    rslcs = stack.rslc_list()
    pt_rslcs = ptarg_rslcs('cr1', rslcs)
    master_par = stack.master_slc_par
    rslc_val = cr_rslc(pt_rslcs, master_par, 'cr1', cr_file, working_dir + sub_dir + 'rslc/',
                                  phase_from='rslc_pixel')
    rslc_df = pd.DataFrame(data=rslc_val)
    rslc_df[0] = pd.to_datetime(rslc_df[0])
    rslc_df = rslc_df.sort_values(by=0)

    rslc_dates = pd.to_datetime(rslc_df[0])
    rslc_phis = [np.angle(val) for val in rslc_df[1].values]
    slc_df['Date'] = pd.to_datetime(slc_df['Date'].values, format='%Y%m%d')
    slc_df = slc_df.sort_values(by='Date')
    slc_phis = slc_df['phi_1']
    slc_dates = slc_df['Date']

    m, b, r_value, p_value, std_err = linregress(slc_phis, rslc_phis)
    x = np.linspace(-np.pi, np.pi, 10)
    y = m * x + b
    plt.title('SLC versus RSLC phase')
    plt.scatter(slc_phis, rslc_phis)
    plt.xlabel('slc')
    plt.ylabel('rslc')
    plt.plot([-np.pi, np.pi], [-np.pi, np.pi], '--', color='grey', label='expected')
    plt.plot(x, y, label='R_val = ' + str(r_value)[:4])
    for i, date in enumerate(slc_dates):
        plt.annotate(date.strftime('%m/%y'), (slc_phis[i], rslc_phis[i]))
    plt.legend()
    plt.show()
    print(p_value)


def compare_cr_annulus(cr, show_plot=False):
    """
    Correlation plot between corner reflector pixel and surroudning annulus
    :param cr: (str) corner reflector label, eg 'cr1'
    :return:
    """
    master_par = stack.master_slc_par
    files = glob.glob(working_dir + sub_dir + 'diff/*.diff')
    itab = working_dir + sub_dir + 'itab_postcr_lf'
    RSLC_tab = working_dir + sub_dir + 'RSLC_tab'
    postcr_files = get_itab_diffs(files, itab, RSLC_tab)

    roots = [os.path.basename(f) for f in postcr_files]
    cr_df = cr_phase(cr, cr_file, postcr_files, master_par.dim, dtype='complex64')
    an_df = cr_phase(cr, cr_file, postcr_files, master_par.dim, dtype='complex64', annulus=10, show_plot=0)

    cr_df = cr_df.sort_values(by='slave')
    cr_df = cr_df.reset_index(drop=True)
    cr_df = cr_df.rename(columns={'delta_phi': 'cr_phi'})
    an_df = an_df.sort_values(by='slave')
    an_df = an_df.reset_index(drop=True)
    an_df = an_df.rename(columns={'delta_phi': 'an_phi'})

    df = pd.merge(cr_df, an_df, on=['master', 'slave'], how='left')

                            #  start date               #  end date
    snow_dates = {'2014': [pd.to_datetime('20141008'), pd.to_datetime('20150601')],
                  '2015': [pd.to_datetime('20151001'), pd.to_datetime('20160601')],
                  '2016': [pd.to_datetime('20161001'), pd.to_datetime('20170601')],
                  '2017': [pd.to_datetime('20171009'), pd.to_datetime('20180526')],
                  '2018': [pd.to_datetime('20180922'), pd.to_datetime('20190518')],
                  '2019': [pd.to_datetime('20191006'), pd.to_datetime('20200523')],
                  '2020': [pd.to_datetime('20201009'), pd.to_datetime('20210515')]}

    snow_ind = []
    for year in snow_dates.keys():
        snow = cr_df[cr_df['slave'] >= snow_dates[year][0]]
        snow = snow.index[snow['slave'] <= snow_dates[year][1]].values
        for i in snow:
            snow_ind.append(i)

    snow_df = df.iloc[snow_ind]
    nosnow_df = df.drop(snow_ind)

    if show_plot:
        plt.figure(1)
        plt.subplot(121)
        m, b, r_value, p_value, std_err = linregress(np.angle(snow_df['cr_phi']), np.angle(snow_df['an_phi']))
        x = np.linspace(-np.pi, np.pi, 10)
        y = m * x + b
        plt.scatter(np.angle(snow_df['cr_phi']), np.angle(snow_df['an_phi']))
        plt.plot([-np.pi, np.pi], [-np.pi, np.pi], '--', color='grey', label='R=1')
        plt.plot(x, y, label='R = ' + str(round(r_value, 2)))
        for i, date in enumerate(snow_df['master']):
            plt.annotate(date.strftime('%m/%y'), (np.angle(snow_df['cr_phi'])[i], np.angle(snow_df['an_phi'])[i]))
        plt.title('Snow')
        plt.xlabel('CR phase')
        plt.ylabel('Annulus phase')
        plt.legend()

        plt.subplot(122)
        m, b, r_value, p_value, std_err = linregress(np.angle(nosnow_df['cr_phi']), np.angle(nosnow_df['an_phi']))
        x = np.linspace(-np.pi, np.pi, 10)
        y = m * x + b
        plt.scatter(np.angle(nosnow_df['cr_phi']), np.angle(nosnow_df['an_phi']))
        plt.plot([-np.pi, np.pi], [-np.pi, np.pi], '--', color='grey', label='R=1')
        plt.plot(x, y, label='R = ' + str(round(r_value, 2)))
        for i, date in enumerate(nosnow_df['master']):
            plt.annotate(date.strftime('%m/%y'), (np.angle(nosnow_df['cr_phi'])[i], np.angle(nosnow_df['an_phi'])[i]))
        plt.title('No Snow')
        plt.xlabel('CR phase')
        plt.ylabel('Annulus phase')
        plt.legend()

        plt.suptitle(cr)
        plt.show()

    return snow_df, nosnow_df

    """
    plt.figure(2)
    m, b, r_value, p_value, std_err = linregress(np.angle(df['cr_phi']), np.angle(df['an_phi']))
    x = np.linspace(-np.pi, np.pi, 10)
    y = m * x + b
    plt.scatter(np.angle(df['cr_phi']), np.angle(df['an_phi']))
    plt.plot([-np.pi, np.pi], [-np.pi, np.pi], '--', color='grey', label='R=1')
    plt.plot(x, y, label='R = ' + str(round(r_value, 2)))
    plt.title(cr)
    plt.xlabel('CR phase')
    plt.ylabel('Annulus phase')
    plt.legend()
    plt.show()
    """

def extract_phase_from_slc(cr='cr1'):
    pixel_loc = pd.read_csv(working_dir + sub_dir + 'slc_phase_site2.csv')
    print(pixel_loc)
    slcs = ptarg_rslcs(cr, stack.slc_list())
    for slc in slcs:
        par = SLC_Par(slc+'.par')
        im = readBin(slc, par.dim, 'complex64')

        root = os.path.basename(slc)
        date = root.split('.')[0]


        row = pixel_loc[pixel_loc['Date'] == int(date)]
        cols = list(pixel_loc)
        r_cols = [col for col in cols if 'r_px' in col]
        az_cols = [col for col in cols if 'az_px' in col]
        mag_cols = [col for col in cols if 'Mag' in col]
        phi_cols = [col for col in cols if 'Phi' in col]

        r_c = row[r_cols].values
        az_c = row[az_cols].values

        # verify pixel location
        """
        r_center = int(round(np.nanmean(row[r_cols].values), 0))
        az_center = int(round(np.nanmean(row[az_cols].values), 0))

        plt.imshow(20*np.log10(np.abs(im.T)), vmin=-5)
        plt.scatter(r_c, az_c, color='red')
        plt.xlim([r_center-30, r_center+20])
        plt.ylim([az_center+20, az_center-20])
        plt.title(date)
        plt.show()
        """

        r_c = np.array(r_c).T
        az_c = np.array(az_c).T
        for r, az, m, p in zip(r_c, az_c, mag_cols, phi_cols):
            if ~np.isnan(r):
                mag = np.abs(im[int(r), int(az)])
                phi = np.angle(im[int(r), int(az)])

                row[m] = mag
                row[p] = phi

        pixel_loc.loc[pixel_loc['Date'] == int(date)] = row

    pixel_loc.to_csv(working_dir + sub_dir + 'slc_phase_site2.csv')


if __name__ == "__main__":

    ### CORRELATION PLOTS ###
    """
    cr1_snow, cr1_nosnow = compare_cr_annulus('cr1')
    cr2_snow, cr2_nosnow = compare_cr_annulus('cr2')

    #snow
    plt.subplot(121)
    plt.title('Snow')
    cr_diff = cr1_snow['cr_phi'] * np.conj(cr2_snow['cr_phi'])
    an_diff = cr1_snow['an_phi'] * np.conj(cr2_snow['an_phi'])

    cr_an_diff = cr1_snow['cr_phi'] * np.conj(cr1_snow['an_phi'])
    cr_an2_diff = cr1_snow['cr_phi'] * np.conj(cr2_snow['an_phi'])

    m, b, r_value, p_value, std_err = linregress(np.angle(cr_diff), np.angle(an_diff))
    x = np.linspace(-np.pi, np.pi, 10)
    y = m * x + b
    plt.plot([-np.pi, np.pi], [-np.pi, np.pi], '--', color='grey', label='R=1')
    #plt.plot(x, y, label='R = ' + str(round(r_value, 2)))
    plt.scatter(np.angle(cr_diff), np.angle(an_diff), label='(cr1 - cr2) vs (an1 - an2)')
    plt.scatter(np.angle(cr_diff), np.angle(cr_an_diff), label='(cr1 - cr2) vs (cr1-an1)')
    plt.scatter(np.angle(cr_diff), np.angle(cr_an2_diff), label='(cr1 - cr2) vs (cr1-an2)')
    plt.legend()

    #nosnow
    plt.subplot(122)
    plt.title('No Snow')
    cr_diff = cr1_nosnow['cr_phi'] * np.conj(cr2_nosnow['cr_phi'])
    an_diff = cr1_nosnow['an_phi'] * np.conj(cr2_nosnow['an_phi'])

    cr_an_diff = cr1_nosnow['cr_phi'] * np.conj(cr1_nosnow['an_phi'])
    cr_an2_diff = cr1_nosnow['cr_phi'] * np.conj(cr2_nosnow['an_phi'])

    m, b, r_value, p_value, std_err = linregress(np.angle(cr_diff), np.angle(an_diff))
    x = np.linspace(-np.pi, np.pi, 10)
    y = m * x + b
    plt.plot([-np.pi, np.pi], [-np.pi, np.pi], '--', color='grey', label='R=1')
    #plt.plot(x, y, label='R = ' + str(round(r_value, 2)))
    plt.scatter(np.angle(cr_diff), np.angle(an_diff), label='(cr1 - cr2) vs (an1 - an2)')
    plt.scatter(np.angle(cr_diff), np.angle(cr_an_diff), label='(cr1 - cr2) vs (cr1-an1)')
    plt.scatter(np.angle(cr_diff), np.angle(cr_an2_diff), label='(cr1 - cr2) vs (cr1-an2)')
    plt.legend()
    plt.show()
    """


    ### PHASE FROM SLC TO CSV ###
    extract_phase_from_slc(cr='cr3')









