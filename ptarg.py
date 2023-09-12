from sarlab.gammax import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.constants import c
import os
import re
import pandas as pd

"""
working_dir = '/local-scratch/users/aplourde/RS2/post_cr_installation/'
master = '20190822'
#sub_dir = 'full_scene/'
#sub_dir = 'full_scene_tresmask/'
#sub_dir = 'crop_site1_only/'
#sub_dir = 'crop_large/'
sub_dir = 'ptarg_site1/'; target_dir = working_dir + 'ptarg_site1/rslc/'
#sub_dir = 'ptarg_site2/'; target_dir = working_dir + 'ptarg_site2/rslc/'
"""
"""
working_dir ='/local-scratch/users/jaysone/projects_active/inuvik/RS2_U76_D/'
sub_dir = 'small/'; master = '20170808'
"""
#"""
working_dir = '/local-scratch/users/aplourde/HDS/projects/southern_ITH/RS2_U76_D/'
#sub_dir = 'full_scene_summer/'; master = '20170808'
#sub_dir = 'full_scene_winter/'; master = '20191009'
sub_dir = 'full_scene/'; master = '20180827'
#sub_dir = 'old/full_scene_masked/'; master = '20170808'
#sub_dir = 'temp/'; master = '20180827'
#sub_dir = 'crop_sites/'; master = '20170808'
#sub_dir = 'crop_sites_sb/'; master = '20170808'
#sub_dir = 'crop_slc/'; master = '20170808'
#sub_dir = 'crop_slc_post_cr/'; master = '20190729'
target_dir = working_dir + sub_dir + 'ptarg/'
cr_loc = working_dir + sub_dir + 'cr_loc.txt'
#"""

ingest_cfg = {'password': 'S1m0nF7@SER', 'polarizations': 'HH'}
stack = SLC_stack(dirname=working_dir + sub_dir,name='inuvik_RS2_U76_D', master=master, looks_hr=(2,3), looks_lr=(12,18), multiprocess=True, rdcdem_refine=True, skipmode='None', ingest_cfg=ingest_cfg)


### ptr_par keys
class PtargPar:
    def __init__(self, ptr_file):

        vals = np.loadtxt(ptr_file)

        self._r_pix = vals[0]
        self._az_pix = vals[1]
        self._r_pos = vals[2]
        self._az_pos = vals[3]
        self._pk_mag = vals[4]
        self._pk_phase_deg = vals[5]
        self._r_3dB_width = vals[6]
        self._az_3dB_width = vals[7]
        self._r_10dB_width = vals[8]
        self._az_10dB_width = vals[9]
        self._r_pslr_dB = vals[10]
        self._az_pslr_dB = vals[11]
        self._r_islr_dB = vals[12]
        self._az_islr_dB = vals[13]
        if len(vals) > 14:
            self._r_sle = vals[14]
            self._r_mle = vals[15]
            self._az_sle = vals[16]
            self._az_mle = vals[17]

def unpack_cr_coords(cr_file):
    crs = pd.read_csv(cr_file)

    coords = {'cr1': (crs[crs['cr'] == 1]['x_loc'].values[0], crs[crs['cr'] == 1]['y_loc'].values[0]),
              'cr2': (crs[crs['cr'] == 2]['x_loc'].values[0], crs[crs['cr'] == 2]['y_loc'].values[0]),
              'cr3': (crs[crs['cr'] == 3]['x_loc'].values[0], crs[crs['cr'] == 3]['y_loc'].values[0]),
              'cr4': (crs[crs['cr'] == 4]['x_loc'].values[0], crs[crs['cr'] == 4]['y_loc'].values[0]),
              'cr5': (crs[crs['cr'] == 5]['x_loc'].values[0], crs[crs['cr'] == 5]['y_loc'].values[0]),
              'cr6': (crs[crs['cr'] == 6]['x_loc'].values[0], crs[crs['cr'] == 6]['y_loc'].values[0])}

    return coords

"""
def amplitude_dispersion ():
    rslc_list = stack.rslc_list()
    n = len(rslc_list)

    im_ave = []
    im_std = []
    cr1 = []
    cr2 = []
    random_p1 = []
    par = SLC_Par(rslc_list[0] + '.par')
    ave_mag = np.zeros(par.dim)
    for rslc in rslc_list:
        if "20190331" in str(rslc):
            print("SKIP!")
            n = n-1
            continue
        par = SLC_Par(rslc + '.par')

        im = readBin(rslc, par.dim, 'complex64')
        run('cpx_to_real', rslc, rslc + '.int', par.dim[0], 2)
        #run('dispwr', rslc + '.int', par.dim[0])
        #run('dismph', rslc, par.dim[0])

        mag = readBin(rslc+'.int', par.dim, 'float32')

        cr1.append(mag[cr_crop_large['x'][0], cr_crop_large['y'][0]])
        cr2.append(mag[cr_crop_large['x'][1], cr_crop_large['y'][1]])
        mid_x = int((cr_crop_large['x'][0]+cr_crop_large['x'][1])/2)
        mid_y = int((cr_crop_large['y'][0]+cr_crop_large['y'][1])/2)
        random_p1.append(mag[mid_x, mid_y])

        cr = 1
        plt.imshow(np.transpose(mag[cr_crop_large['x'][cr]-3:cr_crop_large['x'][cr]+5, cr_crop_large['y'][cr]-4:cr_crop_large['y'][cr]+4]))
        plt.colorbar()
        plt.savefig(rslc+'.crop.png')
        plt.close()

        im_ave.append(np.mean(mag))
        im_std.append(np.std(mag))

        ave_mag += mag

    ave_mag = ave_mag / n

    # plt.imshow(np.transpose(mag[cr_crop_large['x'][0]-1:cr_crop_large['x'][0]+1, cr_crop_large['y'][0]-1:cr_crop_large['y'][0]+1]))
    plt.imshow(np.transpose(ave_mag))
    plt.colorbar()
    plt.savefig('ave_mag.png')
    plt.close()

    #im_ave = np.log10(im_ave)
    #cr1 = np.log10(cr1)
    #cr2 = np.log10(cr2)
    #random_p1 = np.log10(random_p1)

    DA_1 = np.std(cr1)/np.mean(cr1)
    DA_2 = np.std(cr2) / np.mean(cr2)
    DA_ave = np.std(im_ave)/np.mean(im_ave)
    DA_r1 = np.std(random_p1)/np.mean(random_p1)

    print(DA_ave, DA_1, DA_2, DA_r1)

    #plt.hist(im_ave)
    plt.hist(cr1, bins=10)
    plt.hist(cr2, bins=10)
    plt.hist(random_p1, bins=10)
    plt.ylabel("Frequency")
    plt.xlabel("Magnitude")
    plt.savefig('hist.png')
    plt.close()

    plt.plot(range(n), im_ave, label="image average")
    plt.plot(range(n), cr1, label="cr1")
    plt.plot(range(n), cr2, label="cr2")
    plt.plot(range(n), random_p1, label="midpoint")
    plt.legend()
    plt.savefig('im_ave.png')
"""

def point_target(coords, rslc_list, cr='cr1', label_cr = True):

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for rslc in rslc_list:

        #if "20190331" in str(rslc):
        #    print("SKIP!")
        #    n = n-1
        #    continue
        par = SLC_Par(rslc + '.par')

        if label_cr:
            target_rslc = target_dir + os.path.basename(rslc)[:-5] + '_' + cr + '.rslc'
        else:
            target_rslc = target_dir + os.path.basename(rslc)

        run('ptarg_SLC', par, rslc, coords[cr][0], coords[cr][1], target_rslc + '.output', target_rslc + '.r_plot',
            target_rslc + '.az_plot', target_rslc + '.ptr_par', 16, 1, 1)

        run('rasSLC', target_rslc + '.output', 1024, 1, 1023, None, None, None, None, 1, 0, None, target_rslc + '.sans_phase.ras')
        run('rasSLC', target_rslc + '.output', 1024, 1024, 2046, None, None, None, None, 1, 0, None, target_rslc + '.ras')

        run('cp_data', target_rslc + '.output', target_rslc, 4096, 2049, None, None, None, None, None)


def point_target_slc(coord_csv, slc_list, cr = 'cr1', label_cr = True):

    target_dir = working_dir + sub_dir + 'ptarg_slc/'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    coords = pd.read_csv(coord_csv)

    for slc in slc_list:

        #if "20190331" in str(rslc):
        #    print("SKIP!")
        #    n = n-1
        #    continue

        root = os.path.basename(slc)
        date = root.split('.')[0]
        par = SLC_Par(slc + '.par')

        rc = coords['r_px_' + cr[-1]][coords['Date'] == int(date)].values[0]
        ac = coords['az_px_' + cr[-1]][coords['Date'] == int(date)].values[0]

        if ~np.isnan(rc):

            if label_cr:
                target_slc = target_dir + date + '_' + cr + '.slc'
            else:
                target_slc = target_dir + root

            osf = 16

            run('ptarg_SLC', par, slc, rc, ac, target_slc + '.output', target_slc + '.r_plot',
                target_slc + '.az_plot', target_slc + '.ptr_par', osf, 1, 1)

            run('rasSLC', target_slc + '.output', osf*64, 1, osf*64-1, None, None, None, None, 1, 0, None, target_slc + '.sans_phase.ras')
            run('rasSLC', target_slc + '.output', osf*64, osf*64, 2*(osf*64-1), None, None, None, None, 1, 0, None, target_slc + '.ras')

            run('cp_data', target_slc + '.output', target_slc, 4*16*osf, 2*(osf*64)+1, None, None, None, None, None)


def overwrite_par_files():
    rslc_list = stack.rslc_list()
    n = len(rslc_list)

    for rslc in rslc_list:
        if "20190331" in str(rslc):
            print("SKIP!")
            n = n-1
            continue
        par = SLC_Par(rslc + '.par')

        new_range = par['range_samples'] * 16
        new_azimuth = par['azimuth_lines'] * 16
        new_range_ps = par['range_pixel_spacing'] / 16
        new_azimuth_ps = par['azimuth_pixel_spacing'] / 16

        run('set_value', rslc + '.par', rslc + '.par.new', 'range_samples', new_range, 0)
        os.rename(rslc + '.par.new', rslc + '.par')
        run('set_value', rslc + '.par', rslc + '.par.new', 'azimuth_lines', new_azimuth, 0)
        os.rename(rslc + '.par.new', rslc + '.par')
        run('set_value', rslc + '.par', rslc + '.par.new', 'range_pixel_spacing', new_range_ps, 0)
        os.rename(rslc + '.par.new', rslc + '.par')
        run('set_value', rslc + '.par', rslc + '.par.new', 'azimuth_pixel_spacing', new_azimuth_ps, 0)
        os.rename(rslc+'.par.new', rslc+'.par')

"""#obsolete
Phase varies significantly within PT glob, should use peak phase instead
def diff_phase_analysis(show_plot = False):
    
    diff = sorted(glob.glob(working_dir + 'ptarg_site1/diff_fr/*.diff'))
    print(diff)
    dates = []
    signal = []
    for ifg in diff:
        im = readBin(ifg, [1024, 1024], 'complex64')

        phi_im = np.angle(im)

        wn_factor = 0.25
        cr1 = phi_im[cr_ptarg['x'][0]-int(wn_factor*cr_ptarg['diameter'][0]):
                          cr_ptarg['x'][0]+int(wn_factor*cr_ptarg['diameter'][0]),
                          cr_ptarg['y'][0]-int(wn_factor*cr_ptarg['diameter'][0]):
                          cr_ptarg['y'][0]+int(wn_factor*cr_ptarg['diameter'][0])]

        cr2 = phi_im[cr_ptarg['x'][1]-int(wn_factor*cr_ptarg['diameter'][1]):
                          cr_ptarg['x'][1]+int(wn_factor*cr_ptarg['diameter'][1]),
                          cr_ptarg['y'][1]-int(wn_factor*cr_ptarg['diameter'][1]):
                          cr_ptarg['y'][1]+int(wn_factor*cr_ptarg['diameter'][1])]


        cr1 = np.mean(cr1)
        cr2 = np.mean(cr2)

        signal.append(np.angle(np.exp((cr1 - cr2)*1j)))

        filename = os.path.basename(ifg)
        filename = filename.split('.')[0]
        master_date, slave_date = filename.split('_')
        dates.append((master_date, slave_date))


    if show_plot:
        plt.plot(range(len(signal)), signal)
        #plt.legend()
        plt.show()
        plt.close()

    return dates, signal
"""


def ptarg_plots():
    file = working_dir + sub_dir + 'rslc/' + '20190518.rslc.r_plot'
    relative_peak_location = np.loadtxt(file, dtype='float', usecols = (0,))
    signal_intensity_dB = np.loadtxt(file, dtype='float', usecols = (1,))
    signal_intensity_linear = np.loadtxt(file, dtype='float', usecols = (2,))
    signal_phase = np.loadtxt(file, dtype='float', usecols = (3,))

    plt.plot(relative_peak_location, signal_intensity_dB)
    plt.plot(relative_peak_location, signal_intensity_linear)
    plt.plot(relative_peak_location, signal_phase)
    plt.xlabel("Relative_Peak_Location")
    plt.ylabel("Signal Intensity(dB)")
    plt.savefig('r_plot.png')

    print(np.min(relative_peak_location), np.max(relative_peak_location))

"""old
def plot_subpixel_location(rslcs, cr):

    ave_cr1 = [[],[]]
    ave_cr2 = [[],[]]
    r_pixels = []
    az_pixels = []
    for rslc in rslcs:
        filename = os.path.basename(rslc)
        par = SLC_Par(rslc + '.par')
        if cr is None:
            ptr_par_1 = target_dir + filename[:-5] + '_cr1.rslc' + '.ptr_par'
            ptr_par_2 = target_dir + filename[:-5] + '_cr2.rslc' + '.ptr_par'
            r1, az1 = np.loadtxt(ptr_par_1, usecols=[0, 1])
            r2, az2 = np.loadtxt(ptr_par_2, usecols=[0, 1])

            ave_cr1[0].append(r1)
            ave_cr1[1].append(az1)
            ave_cr2[0].append(r2)
            ave_cr2[1].append(az2)

            # normalize

            cr1 = (5121.511523809523, 5745.647285714287)
            cr2 = (5110.293476190476, 5762.533571428571)


            r1 -= cr1[0]
            az1 -= cr1[1]
            r2 -= cr2[0]
            az2 -= cr2[1]


            r_pix = np.asarray([r1, r2])
            az_pix = np.asarray([az1, az2])
        else:
            ptr_par = target_dir + filename[:-5] + '_' + cr + '.rslc' + '.ptr_par'
            r_pix, az_pix = np.loadtxt(ptr_par, usecols=[0, 1])
            r_pixels.append(r_pix)
            az_pixels.append(az_pix)

        date = pd.to_datetime(filename[:-5])

        # single rslc plts
        #im = readBin(rslc, par.dim, 'complex64')
        #amp = np.absolute(im)
        #plt.imshow(amp.T, cmap=cm.Greys_r, vmin=0, vmax=2)

        # convert to slant to ground range
        range_pixel_spacing = 1.330834   #m
        azimuth_pixel_spacing = 2.076483   #m
        incidence_angle = 26.9903   #degrees
        ground_range_pixel_spacing = range_pixel_spacing / np.sin(np.radians(incidence_angle))

        r_pix = r_pix * ground_range_pixel_spacing
        az_pix = az_pix * azimuth_pixel_spacing

        print(r_pix, az_pix)

        summer = [5, 6, 7, 8, 9]
        winter = [1, 2, 3, 4, 10, 11, 12]
        spring = []
        fall = []
        #spring = [3,4,5]
        #summer = [6,7,8]
        #fall = [9, 10, 11]
        #winter = [12, 1, 2]

        if cr is not None:
            if date.month in spring:
                plt.scatter(r_pix, az_pix, marker='x', color='lime')
            elif date.month in summer:
                plt.scatter(r_pix, az_pix, marker='x', color='purple')
            elif date.month in fall:
                plt.scatter(r_pix, az_pix, marker='x', color='orange')
            elif date.month in winter:
                plt.scatter(r_pix, az_pix, marker='x', color='cyan')
            else:
                return
            if date.year == 2018 and date.month == 9 and cr == 1:
                plt.annotate(str(date.year)[-1] + str(date.month), (r_pix-0.05, az_pix+0.01), color='white')
            else:
                plt.annotate(str(date.year)[-1] + str(date.month), (r_pix+0.01, az_pix+0.01), color='white')
        else:

            if date.month in spring:
                plt.scatter(r_pix[0], az_pix[0], marker='x', color='lime')
                plt.scatter(r_pix[1], az_pix[1], marker='.', color='lime')
            elif date.month in summer:
                plt.scatter(r_pix[0], az_pix[0], marker='x', color='purple')
                plt.scatter(r_pix[1], az_pix[1], marker='.', color='purple')
            elif date.month in fall:
                plt.scatter(r_pix[0], az_pix[0], marker='x', color='orange')
                plt.scatter(r_pix[1], az_pix[1], marker='.', color='orange')
            elif date.month in winter:
                plt.scatter(r_pix[0], az_pix[0], marker='x', color='cyan')
                plt.scatter(r_pix[1], az_pix[1], marker='.', color='cyan')
            else:
                plt.scatter(r_pix[0], az_pix[0], marker='x', color='blue')
                plt.scatter(r_pix[1], az_pix[1], marker='x', color='red')

            plt.scatter(r_pix[0], az_pix[0], marker='x', color='blue')
            plt.scatter(r_pix[1], az_pix[1], marker='.', color='orange')
            plt.annotate(str(date.year)[-1] + str(date.month), (r_pix[0], az_pix[0]))
            plt.annotate(str(date.year)[-1] + str(date.month), (r_pix[1], az_pix[1]))


    if cr is not None:
        plt.title(
            'Subpixel Locations - ' + cr + '\nLabel Convention: YYY[Y]-[MM]-DD (eg. 2018-09-20 labelled 89)')
        im = readBin(working_dir + sub_dir + '/rmli_fr/ave.rmli', par.dim, 'float32')
        print(im.T.shape)
        plt.imshow(im.T, cmap=cm.Greys_r, extent = [0, (im.T.shape[1]-1) * ground_range_pixel_spacing, (im.T.shape[0]-1) * azimuth_pixel_spacing, 0], aspect = 1)
        #plt.scatter(r_pix, az_pix, marker='x', color='lime', label='May - Sept')
        #plt.scatter(r_pix, az_pix, marker='x', color='cyan', label='Oct - April')
    else:
        plt.title(
            'Relative Subpixel Locations\n(normalized to average position of CR)')
        plt.scatter(r_pix[0], az_pix[0], marker='x', color='blue', label='Anchored')
        plt.scatter(r_pix[1], az_pix[1], marker='x', color='red', label='Floating')
    if cr == 'cr1':
        pass
        #plt.xlim([5120, 5123])
        #plt.ylim([5744, 5747])
        #plt.xlim(np.round(np.asarray([5120, 5122.5]) * ground_range_pixel_spacing))
        #plt.ylim(np.round(np.asarray([5744, 5747]) * azimuth_pixel_spacing))
        #plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0),
        #                     useOffset=np.round(5120 * ground_range_pixel_spacing))
        #plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useOffset=np.round(5744 * azimuth_pixel_spacing))
    elif cr == 'cr2':
        plt.xlim(np.round(np.asarray([5109, 5112]) * ground_range_pixel_spacing))
        plt.ylim(np.round(np.asarray([5761, 5764]) * azimuth_pixel_spacing))
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0),
                             useOffset=np.round(5109 * ground_range_pixel_spacing))
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useOffset=np.round(5761 * azimuth_pixel_spacing))
    #plt.colorbar()
    plt.xlabel('Ground range position (m)')
    plt.ylabel('Azimuth position (m)')
    #plt.ticklabel_format(axis='both', style='scientific', scilimits=(0, 0), useLocale=15000)

    plt.grid()
    plt.legend()

    plt.show()
    #plt.savefig(working_dir + 'ptarg_vs_ovr/subpixel_pngs/' + filename[:-5] + '_fr_cr' + str(cr) + '.png')
    plt.close()


    print('average cr1 coords: ({},{})'.format(np.mean(ave_cr1[0]), np.mean(ave_cr1[1])))
    print('average cr2 coords: ({},{})'.format(np.mean(ave_cr2[0]), np.mean(ave_cr2[1])))
"""


def subpixel_locations(rslcs, ptarg_dir=target_dir, cr='cr1', show_plot=False, label=False):
    subpixel_loc = {'RSLC': [], 'r_pix': [], 'az_pix': []}
    for rslc in rslcs:
        if "20190331" in str(rslc):
            #poorly registered scene, remove as outlier
            continue
        filename = os.path.basename(rslc)

        par = SLC_Par(rslc + '.par')

        ptr_par = ptarg_dir + filename[:-5] + '_' + cr + '.rslc' + '.ptr_par'
        r, az = np.loadtxt(ptr_par, usecols=[0, 1])

        subpixel_loc['RSLC'].append(filename)
        subpixel_loc['r_pix'].append(r)
        subpixel_loc['az_pix'].append(az)

    ave_r_pix = np.mean(subpixel_loc['r_pix'])
    ave_az_pix = np.mean(subpixel_loc['az_pix'])

    def cr_pix_loc(ave_pix):
        if round(ave_pix) - ave_pix > 0:
            target_pix = [int(ave_pix) + 0.5, int(ave_pix )+1.5]
        else:
            target_pix = [int(ave_pix) - 0.5, int(ave_pix) + 0.5]
        return target_pix

    target_r_pix = cr_pix_loc(ave_r_pix)
    target_az_pix = cr_pix_loc(ave_az_pix)

    subpixel_loc = pd.DataFrame(data=subpixel_loc)
    # if r, az not in ave, throw it away!
    filtered_loc = subpixel_loc[subpixel_loc['r_pix'] > target_r_pix[0]]
    filtered_loc = filtered_loc[subpixel_loc['r_pix'] < target_r_pix[1]]
    filtered_loc = filtered_loc[subpixel_loc['az_pix'] > target_az_pix[0]]
    filtered_loc = filtered_loc[subpixel_loc['az_pix'] < target_az_pix[1]]

    rslcs_to_keep = filtered_loc['RSLC'].values

    if show_plot:
        try:
            im = readBin(working_dir + sub_dir + 'rmli_1_1/rmli_1_1.ave', par.dim, 'float32')
        except:
            im = readBin(working_dir + sub_dir + 'rmli_fr/ave.rmli', par.dim, 'float32')
        plt.scatter(subpixel_loc['r_pix'], subpixel_loc['az_pix'])
        plt.imshow(im.T, vmax=2)
        plt.colorbar()
        plt.xlim(np.mean(subpixel_loc['r_pix'])-3, np.mean(subpixel_loc['r_pix'])+3)
        plt.ylim(np.mean(subpixel_loc['az_pix'])-3, np.mean(subpixel_loc['az_pix'])+3)
        r_mean = np.mean(subpixel_loc['r_pix'])
        r_std = np.std(subpixel_loc['r_pix'])
        az_mean = np.mean(subpixel_loc['az_pix'])
        az_std = np.std(subpixel_loc['az_pix'])
        plt.title('{m} - {cr}\nRange: u = {ru:.4f}, std = {rs:.4f}\nAzimuth: u = {azu:.4f}, std = {azs:.4f}'.format(m = master, cr=cr, ru=r_mean, rs=r_std, azu=az_mean, azs=az_std))
        if label:
            for i, rslc in enumerate(subpixel_loc['RSLC']):
                plt.annotate(rslc.split('.')[0], (subpixel_loc['r_pix'][i], subpixel_loc['az_pix'][i]))
        plt.show()

    return rslcs_to_keep


def phase_analysis(show_plot = False):

    rslcs = stack.rslc_list()
    rslcs = sorted([os.path.basename(rslc) for rslc in rslcs])

    signal = []
    dates = []
    for i in range(len(rslcs)-1):
        # 0: r_pix, 1: az_pix, 4:pk_mag, 5:pk_phase_deg,
        cr1_m = np.deg2rad(np.loadtxt(target_dir + rslcs[i][:-5] + '_cr1.rslc.ptr_par', usecols=[5]))
        cr1_s = np.deg2rad(np.loadtxt(target_dir + rslcs[i+1][:-5] + '_cr1.rslc.ptr_par', usecols=[5]))
        cr2_m = np.deg2rad(np.loadtxt(target_dir + rslcs[i][:-5] + '_cr2.rslc.ptr_par', usecols=[5]))
        cr2_s = np.deg2rad(np.loadtxt(target_dir + rslcs[i+1][:-5] + '_cr2.rslc.ptr_par', usecols=[5]))
        print(cr1_m)

        phi_cr1 = np.exp((cr1_m - cr1_s)*1j)
        phi_cr2 = np.exp((cr2_m - cr2_s)*1j)

        phi_cr1 = np.angle(phi_cr1)
        phi_cr2 = np.angle(phi_cr2)

        # subtract floating from anchored
        signal.append(np.angle(np.exp((phi_cr1 - phi_cr2)*1j)))
        dates.append((rslcs[i][:-5], rslcs[i + 1][:-5]))

    if show_plot:
        plt.title('Signal = Floating - Anchored')
        plt.plot(range(len(signal)), signal)
        plt.legend()
        plt.show()
        plt.close()

    return dates, signal


def phase_analysis_full_scene(show_plot=False, ptype = 'cr', offset = None, blocksize = None):
    # using ptarg coordinates, take the average in the interferogram
    # i.e. if the cr is located at the same pixel in both master and slave,
    # the phase remains unchanged, else the phase is the average between
    # the phase in the master rslc pixel and the phase in the slave rslc pixel
    # probably not a great solution.

    #force sub_dir
    sub_dir = 'full_scene/'
    diff = sorted(glob.glob(working_dir + sub_dir + '/diff_fr/*.diff'))

    dim = (7621, 11430)
    dates = []
    signal = []

    for ifg in diff:
        filename = os.path.basename(ifg)
        im = readBin(ifg, dim, 'complex64')
        phi_im = np.angle(im)

        m, s = filename[:-5].split('_')
        m_cr1_coords = (np.loadtxt(target_dir + m + '_cr1.rslc.ptr_par', usecols=[0, 1]))
        s_cr1_coords = (np.loadtxt(target_dir + s + '_cr1.rslc.ptr_par', usecols=[0, 1]))
        m_cr2_coords = (np.loadtxt(target_dir + m + '_cr2.rslc.ptr_par', usecols=[0, 1]))
        s_cr2_coords = (np.loadtxt(target_dir + s + '_cr2.rslc.ptr_par', usecols=[0, 1]))

        if ptype == 'cr_ave':
            phi_cr1 = np.mean([phi_im[round(m_cr1_coords[0]), round(m_cr1_coords[1])],
                                phi_im[round(s_cr1_coords[0]), round(s_cr1_coords[1])]])
            phi_cr2 = np.mean([phi_im[round(m_cr2_coords[0]), round(m_cr2_coords[1])],
                                phi_im[round(s_cr2_coords[0]), round(s_cr2_coords[1])]])
            signal.append(np.angle(np.exp((phi_cr2 - phi_cr1)*1j)))
        if ptype == 'cr_monly':
            phi_cr1 = phi_im[round(m_cr1_coords[0]), round(m_cr1_coords[1])]
            phi_cr2 = phi_im[round(m_cr2_coords[0]), round(m_cr2_coords[1])]
            signal.append(np.angle(np.exp((phi_cr2 - phi_cr1)*1j)))
        if ptype == 'cr1_ave':
            phi_cr1 = np.mean([phi_im[round(m_cr1_coords[0]), round(m_cr1_coords[1])],
                               phi_im[round(s_cr1_coords[0]), round(s_cr1_coords[1])]])
            signal.append(phi_cr1)
        if ptype == 'cr2_ave':
            phi_cr2 = np.mean([phi_im[round(m_cr2_coords[0]), round(m_cr2_coords[1])],
                                phi_im[round(s_cr2_coords[0]), round(s_cr2_coords[1])]])
            signal.append(phi_cr2)
        if ptype == 'cr1_monly':
            phi_cr1 = phi_im[round(m_cr1_coords[0]), round(m_cr1_coords[1])]
            signal.append(phi_cr1)
        if ptype == 'cr2_monly':
            phi_cr2 = phi_im[round(m_cr2_coords[0]), round(m_cr2_coords[1])]
            signal.append(phi_cr2)

        if ptype == 'block':
            print(offset, blocksize)
            if blocksize == 0:
                phi_cr1 = phi_im[round(m_cr1_coords[0])+offset, round(m_cr1_coords[1])+offset]
            else:
                phi_cr1 = np.nanmean(phi_im[round(m_cr1_coords[0])+offset:round(m_cr1_coords[0])+offset+blocksize,
                                     round(m_cr1_coords[1])+offset:round(m_cr1_coords[1])+offset+blocksize])
            print(phi_cr1)
            signal.append(phi_cr1)

        if show_plot:
            plt.imshow(phi_im.T)
            plt.scatter(m_cr1_coords[0], m_cr1_coords[1], marker='x', color='white')
            plt.scatter(m_cr2_coords[0], m_cr2_coords[1], marker='x', color='white')
            plt.scatter(s_cr1_coords[0], s_cr1_coords[1], marker='x', color='black')
            plt.scatter(s_cr2_coords[0], s_cr2_coords[1], marker='x', color='black')

            plt.title(phi_cr1)
            plt.colorbar()
            plt.show()

        dates.append((m, s))

    return dates, signal


def ptarg_rslcs(target_cr, rslcs):
    if target_cr in ['cr1', 'cr2']:
        cr_installation_date = pd.to_datetime('20180827')
    elif target_cr in ['cr3', 'cr4', 'cr5', 'cr6']:
        cr_installation_date = pd.to_datetime('20190729')
    else:
        cr_installation_date = None

    pt_rslcs = []
    for rslc in rslcs:
        if cr_installation_date is not None:
            root = os.path.basename(rslc)
            date = re.findall(r"\d{8}", root)[0]
            if pd.to_datetime(date) >= cr_installation_date:
                pt_rslcs.append(rslc)

    return pt_rslcs


def plot_phase_blocks(slcs, cr):
    slc_loc = pd.read_csv('/local-scratch/users/aplourde/HDS/projects/southern_ITH/RS2_U76_D/full_scene/slc_phase_site1.csv')

    for slc in slcs:
        filename = os.path.basename(slc)
        par = SLC_Par(slc + '.par')
        im = readBin(slc, par.dim, 'complex64')
        phi = np.angle(im)
        mag = np.log10(np.abs(im))

        r_px_1 = slc_loc['r_px_1'][slc_loc['Date'] == int(filename.split('.')[0])].values[0]
        az_px_1 = slc_loc['az_px_1'][slc_loc['Date'] == int(filename.split('.')[0])].values[0]
        plt.scatter(r_px_1, az_px_1, color='red')
        r_px_2 = slc_loc['r_px_2'][slc_loc['Date'] == int(filename.split('.')[0])].values[0]
        az_px_2 = slc_loc['az_px_2'][slc_loc['Date'] == int(filename.split('.')[0])].values[0]
        plt.scatter(r_px_2, az_px_2, color='red')

        slc_loc['phi_1'][slc_loc['Date'] == int(filename.split('.')[0])] = phi[r_px_1][az_px_1]
        slc_loc['phi_2'][slc_loc['Date'] == int(filename.split('.')[0])] = phi[r_px_2][az_px_2]

        plt.title(filename +'\n'+ str(phi[r_px_1][az_px_1])[:6] +', '+ str(phi[r_px_2][az_px_2])[:6])
        plt.imshow(phi.T)#, vmin=-1)
        plt.colorbar()
        plt.xlim([r_px_2-5, r_px_1+5])
        plt.ylim([az_px_2+5, az_px_1-5])
        plt.show()
    #slc_loc.to_csv('/local-scratch/users/aplourde/HDS/projects/southern_ITH/RS2_U76_D/full_scene/slc_phase_site1.csv')


def plot_ptr_mag(target_dir, cr):

    magnitudes = []
    dates = []
    ptr_pars = glob.glob(target_dir + '*' + cr + '*.ptr_par')
    for par in ptr_pars:

        basename = os.path.basename(par)
        root = basename.split('.')[0].split('_')[0]

        ptr_par = PtargPar(par)
        magnitudes.append(ptr_par._pk_mag)
        dates.append(pd.to_datetime(root))

    plt.scatter(dates, magnitudes)
    plt.ylabel("Magnitude (dB)")
    plt.show()

if __name__ == "__main__":

    coords = unpack_cr_coords(cr_loc)
    target_cr = 'cr5'
    rslcs = stack.rslc_list()
    slcs = stack.slc_list()

    pt_rslcs = ptarg_rslcs(target_cr, rslcs)
    pt_slcs = ptarg_rslcs(target_cr, slcs)

    #point_target(coords, pt_rslcs, cr=target_cr)
    if target_cr in ['cr1', 'cr2']:
        point_target_slc(working_dir + sub_dir + 'slc_phase_site1.csv', pt_slcs, cr=target_cr)
    elif target_cr in ['cr3', 'cr4', 'cr5', 'cr6']:
        point_target_slc(working_dir + sub_dir + 'slc_phase_site2.csv', pt_slcs, cr=target_cr)

    plot_ptr_mag(working_dir + sub_dir + 'ptarg_slc/', target_cr)

    #amplitude_dispersion()

    #overwrite_par_files()

    #ptarg_plots()

    #subpixel_locations(pt_rslcs, cr=target_cr, show_plot=1, label=0)

    #phase_analysis(show_plot=True)
    #phase_analysis_full_scene(show_plot=False)

    #plot_phase_blocks(pt_slcs, coords[target_cr])




