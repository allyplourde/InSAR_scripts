from sarlab.gammax import *
import matplotlib.pyplot as plt
from scipy.constants import c
import os
import pandas as pd

master = '20190822'
working_dir = '/local-scratch/users/aplourde/RS2_ITH/post_cr_installation/' #<- only makes sense for post
sub_dir = 'full_scene/'
#sub_dir = 'crop_site1_only/'
#sub_dir = 'crop_large/'
#sub_dir = 'ptarg_site1/'
#sub_dir = 'ptarg_site2/'
target_dir = working_dir + 'ptarg_site1/rslc/'
#target_dir = working_dir + 'ptarg_site2/rslc/'
ingest_cfg = {'password': 'S1m0nF7@SER', 'polarizations': 'HH'}
stack = SLC_stack(dirname=working_dir + sub_dir,name='inuvik_RS2_U76_D', master=master, looks_hr=(2,3), looks_lr=(12,18), multiprocess=True, rdcdem_refine=True, skipmode='None', ingest_cfg=ingest_cfg)

cr_full_scene = {'cr': [1, 2, 3, 4, 5, 6],
                 'x': [5121, 5110, 0, 0, 5019, 0],
                 'y': [5746, 5763, 0, 0, 4129, 0],
                 'type': ['anchored', 'floating']
                 }

cr_crop_large = {'cr': [1, 2, 3, 4, 5, 6],
                 'x': [671, 343],
                 'y': [1996, 2013],
                 'type': ['anchored', 'floating']
                 }

cr_ptarg = {'cr': [1, 2, 3, 4, 5, 6],
            'x': [521, 340, 765, 785, 526, 178],
            'y': [506, 778, 270, 657, 507, 371],
            'diameter': [25, 25, 25, 25, 25, 25],
            'type': ['anchored', 'floating', 'anchored', 'floating', 'not mounted', 'floating']
            }

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


def point_target(coords, cr=1):
    rslc_list = stack.rslc_list()
    diff_list = glob.glob(working_dir + sub_dir + 'diff_fr/*.diff')
    n = len(rslc_list)

    for rslc in rslc_list:
        if "20190331" in str(rslc):
            print("SKIP!")
            n = n-1
            continue
        par = SLC_Par(rslc + '.par')

        #os.chdir(working_dir + sub_dir)
        #print(os.getcwd())
        target_rslc = target_dir + os.path.basename(rslc)[:-5] + '_cr' + str(cr) + '.rslc'

        #print('ptarg', rslc, par.dim[0], coords['x'][cr-1], coords['y'][cr-1], target_rslc, target_rslc + '.r_plot', target_rslc + '.az_plot', 0, 1)
        #run('ptarg', rslc, par.dim[0], coords['x'][cr-1], coords['y'][cr-1], target_rslc+'.output', target_rslc + '.r_plot', target_rslc + '.az_plot', 0, 1, 1)
        #run('cpx_to_real', target_rslc, target_rslc + 'intensity', par.dim[0], 2)
        run('ptarg_SLC', par, rslc, coords['x'][cr - 1], coords['y'][cr - 1], target_rslc + '.output', target_rslc + '.r_plot',
              target_rslc + '.az_plot', target_rslc + '.ptr_par', 16, 1, 1)

        run('rasSLC', target_rslc + '.output', 1024, 1, 1023, None, None, None, None, 1, 0, None, target_rslc + '.sans_phase.ras')
        run('rasSLC', target_rslc + '.output', 1024, 1024, 2046, None, None, None, None, 1, 0, None, target_rslc + '.ras')

        #run('cp_data', target_rslc+'.output', target_rslc, 4096, 1, 2048, None, None, None, None)
        run('cp_data', target_rslc + '.output', target_rslc, 4096, 2049, None, None, None, None, None)


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

def ptarg_diff_analysis():

    diff = glob.glob(working_dir + sub_dir + 'diff_fr/*.diff')
    print(diff)
    phi_cr = {1: [], 2: []}
    #phi_cr = {3: [], 4: [], 5: [], 6: []}
    dates = {'master':[], 'slave':[]}
    im_phi = []
    for ifg in diff:
        im = readBin(ifg, [1024,1024], 'complex64')

        phi_im = np.angle(im)

        crs = [0,1]
        #crs = [2,3,4,5]
        for cr in crs:
            wn_factor = 0.25
            cr_phi = phi_im[cr_ptarg['x'][cr]-int(wn_factor*cr_ptarg['diameter'][cr]):
                          cr_ptarg['x'][cr]+int(wn_factor*cr_ptarg['diameter'][cr]),
                          cr_ptarg['y'][cr]-int(wn_factor*cr_ptarg['diameter'][cr]):
                          cr_ptarg['y'][cr]+int(wn_factor*cr_ptarg['diameter'][cr])]

            phi_cr[cr+1].append(np.mean(cr_phi))
            wn_factor = 2
            if cr+1 == 2:
                im_phi.append(np.mean(phi_im[cr_ptarg['x'][cr]-int(wn_factor*cr_ptarg['diameter'][cr]):
                          cr_ptarg['x'][cr]+int(wn_factor*cr_ptarg['diameter'][cr]),
                          cr_ptarg['y'][cr]-int(wn_factor*cr_ptarg['diameter'][cr]):
                          cr_ptarg['y'][cr]+int(wn_factor*cr_ptarg['diameter'][cr])]))

        filename = os.path.basename(ifg)
        filename = filename.split('.')[0]
        master_date, slave_date = filename.split('_')
        master_date = pd.to_datetime(master_date, format="%Y%m%d")
        slave_date = pd.to_datetime(slave_date, format="%Y%m%d")
        dates['master'].append(master_date)
        dates['slave'].append(slave_date)


    #phi_anchored = phi_cr[3]
    phi_anchored = phi_cr[1]
    #phi_floating = [phi_cr[4], phi_cr[5], phi_cr[6]]
    phi_floating = [phi_cr[2]]

    diff = []
    for floating_cr in phi_floating:
        diff.append(np.asarray(floating_cr) - np.asarray(phi_anchored))

    #plt.plot(range(len(phi_anchored)), phi_anchored, label='anchored')
    #plt.plot(range(len(phi_floating)), phi_floating, label='floating')
    #plt.plot(range(len(diff)), diff, label='phase difference')
    #plt.legend()
    #plt.savefig('test_HC.png')


    df = pd.DataFrame(data = {"master": dates['master'],
                              "slave": dates['slave'],
                              "phi_anchored": phi_anchored,
                              "phi_floating": phi_floating[0],
                              "floating_sub_anchored": diff[0]
                              #"phi_floating_4": phi_floating[0],
                              #"phi_floating_5": phi_floating[1],
                              #"phi_floating_6": phi_floating[2],
                              #"floating_sub_anchored_4": diff[0],
                              #"floating_sub_anchored_5": diff[1],
                              #"floating_sub_anchored_6": diff[2],
                              })
    df = df.sort_values(by=['master']).reset_index().drop(columns=['index'])

    print(df)


    df.to_csv("phi_ptarg_site1_slc_10m.csv")

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

def cr_versus_background():
    #get cr_phase from .csv
    df = pd.read_csv("phi_ptarg_HC_site1_slc_10m.csv")

    #reduce to no-snow

    df['master'] = pd.to_datetime(df['master'],format='%Y-%m-%d')
    df['slave'] = pd.to_datetime(df['slave'],format='%Y-%m-%d')
    df = df.loc[df['master'] > pd.to_datetime('20190401',format='%Y%m%d')]
    df = df.loc[df['slave'] < pd.to_datetime('20201001', format='%Y%m%d')]

    cr_anchored = np.asarray(df['phi_anchored'])
    cr_floating = np.asarray(df['phi_floating'])

    ifgs = []
    for i in range(len(df)):
        master_date = str(df['master'].iloc[i]).split(' ')[0].replace('-','')
        slave_date = str(df['slave'].iloc[i]).split(' ')[0].replace('-','')
        ifgs.append(master_date + '_' + slave_date + '.diff.adf')

    phi = cr_floating #- cr_anchored

    #calculate background phase
    #open diffs
    im_bg = []
    im_cr = []
    im_full = []
    for i in range(len(df)):
        #diff = working_dir + 'crop_site1_only/' + 'diff_fr/' + ifgs[i]
        diff = working_dir + 'ptarg_site1/' + 'diff_fr/' + ifgs[i]
        im = readBin(diff, [1024,1024], 'complex64')
        im = np.angle(im)
        #subtract atmosphere
        cr = 1
        wn_factor = 0.25
        og = im[cr_ptarg['x'][cr]-int(wn_factor*cr_ptarg['diameter'][cr]):
                          cr_ptarg['x'][cr]+int(wn_factor*cr_ptarg['diameter'][cr]),
                          cr_ptarg['y'][cr]-int(wn_factor*cr_ptarg['diameter'][cr]):
                          cr_ptarg['y'][cr]+int(wn_factor*cr_ptarg['diameter'][cr])]
        wn_factor = 1.25
        full_cr = im[cr_ptarg['x'][cr]-int(wn_factor*cr_ptarg['diameter'][cr]):
                          cr_ptarg['x'][cr]+int(wn_factor*cr_ptarg['diameter'][cr]),
                          cr_ptarg['y'][cr]-int(wn_factor*cr_ptarg['diameter'][cr]):
                          cr_ptarg['y'][cr]+int(wn_factor*cr_ptarg['diameter'][cr])]
        wn_factor = 3
        background = im[cr_ptarg['x'][cr]-int(wn_factor*cr_ptarg['diameter'][cr]):
                          cr_ptarg['x'][cr]+int(wn_factor*cr_ptarg['diameter'][cr]),
                          cr_ptarg['y'][cr]-int(wn_factor*cr_ptarg['diameter'][cr]):
                          cr_ptarg['y'][cr]+int(wn_factor*cr_ptarg['diameter'][cr])]
        wn_factor = 1.25
        background[cr_ptarg['x'][cr]-int(wn_factor*cr_ptarg['diameter'][cr]):
                          cr_ptarg['x'][cr]+int(wn_factor*cr_ptarg['diameter'][cr]),
                          cr_ptarg['y'][cr]-int(wn_factor*cr_ptarg['diameter'][cr]):
                          cr_ptarg['y'][cr]+int(wn_factor*cr_ptarg['diameter'][cr])] = np.nan


        im_full.append(np.mean(im))
        im_bg.append(np.mean(background))
        im_cr.append(np.mean(full_cr))


    fig, ax = plt.subplots()
    plt.plot(df['slave'], phi, label='floating CR')
    plt.plot(df['slave'], im_bg, label=r'10 $m^2$')
    plt.plot(df['slave'], im_full, label=r'0.040 $km^2$')
    every_nth = 2
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    plt.legend()
    plt.savefig('phi_compare_hc.png')

def read_ptarg_output_files():

    rslcs = stack.rslc_list()

    for rslc in rslcs:
        print(rslc)
        r_file = rslc + '.r_plot'
        r_data = np.loadtxt(r_file, dtype = 'float', skiprows=0, usecols = (0,1,2,3)) #shape(1024, 4)
        r_dict = {'relative_peak_location':[i[0] for i in r_data],
                  'signal_intensity_dB': [i[1] for i in r_data],
                  'signal_intensity_linear': [i[2] for i in r_data],
                  'signal_phase': [i[3] for i in r_data]}
        a_file = rslc + '.az_plot'
        a_data = np.loadtxt(a_file, dtype = 'float', skiprows=0, usecols = (0,1,2,3)) #shape(1024, 4)
        a_dict = {'relative_peak_location':[i[0] for i in a_data],
                  'signal_intensity_dB': [i[1] for i in a_data],
                  'signal_intensity_linear': [i[2] for i in a_data],
                  'signal_phase': [i[3] for i in a_data]}

        plt.plot(r_dict['relative_peak_location'], r_dict['signal_intensity_dB'], label='range')
        plt.plot(a_dict['relative_peak_location'], a_dict['signal_intensity_dB'], label='azimuth')
        plt.xlabel('relative_peak_location')
        plt.ylabel('signal_intensity_dB')
        plt.title('Sub Pixel Location')
        plt.legend()
        #plt.show()
        plt.close()

        #print(np.min(r_dict['relative_peak_location']), np.max(r_dict['relative_peak_location']), np.mean(r_dict['relative_peak_location']))
        r_imax = np.argmax(r_dict['signal_intensity_dB'])
        a_imax = np.argmax(a_dict['signal_intensity_dB'])
        print(r_imax, r_data[r_imax])
        print(a_imax, a_data[a_imax])

        cr = (r_imax, a_imax)

        im = readBin(rslc, [1024, 1024], 'complex64')
        amp = np.abs(im)

        plt.imshow(amp.T)
        plt.scatter(cr[0], cr[1], marker='x', color = 'r')
        plt.title(os.path.basename(rslc))
        #plt.show()
        plt.close()


        #convert 502, 522 to 64x64
        cr64 = (cr[0] / 16, cr[1] / 16)
        print(cr64)

        #open 64x64 rslc
        dir64 = working_dir + 'ovr_site1/rslc/'
        filename = os.path.basename(rslc)

        rslc64 = glob.glob(dir64 + filename)[0]
        im64 = readBin(rslc64, [64, 64], 'complex64')
        amp64 = np.absolute(im64)
        plt.imshow(amp64.T)
        plt.scatter(cr64[0], cr64[1], marker='x', color='r')
        plt.title(filename + ' Amplitude\n(64x64)')
        plt.xlim([31,34])
        plt.ylim([30,33])
        plt.colorbar()
        #plt.show()
        plt.savefig(working_dir + 'ptarg_vs_ovr/subpixel_pngs/' + filename[:-5] + '_x64_cr1.png')
        plt.close()

def print_ptarg_calls(coords, cr=1):
    rslc_list = stack.rslc_list()
    n = len(rslc_list)

    for rslc in rslc_list:
        if "20190331" in str(rslc):
            print("SKIP!")
            n = n-1
            continue
        par = SLC_Par(rslc + '.par')
        target_rslc = target_dir + os.path.basename(rslc) + '.cr' + str(cr)

        #print('ptarg', rslc, par.dim[0], coords['x'][cr-1], coords['y'][cr-1], target_rslc, target_rslc + '.r_plot',
        #  target_rslc + '.az_plot', 0, 1)

        print('ptarg_SLC', par, rslc, coords['x'][cr - 1], coords['y'][cr - 1], target_rslc, target_rslc + '.r_plot',
              target_rslc + '.az_plot', target_rslc + '.ptr_par', 16, 1, 1)

def parse_ptarg_output(cr = 1):
    output_file = working_dir + 'ptarg_vs_ovr/ptarg_outputs_cr' + str(cr) + '.txt'

    rslcs = []
    range_peak_positions = []
    azimuth_peak_positions = []
    interpolated_peak_powers = []
    interpolated_phases = []
    interpolated_phases_grad = []

    with open(output_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'input SLC data file' in line:
                rslc = line.split(' ')[-1].split('/')[-1][:-1]
            if 'SLC range_peak_position' in line:
                range_peak_position = line.split(' ')[4]
            if 'SLC azimuth_peak_position' in line:
                azimuth_peak_position = line.split(' ')[2]
            if 'SLC interpolated_peak_power_value' in line:
                interpolated_peak_power = line.split(' ')[3]
            if 'interpolated peak phase (no phase gradients)' in line:
                interpolated_phase = line.split(' ')[-1][:-1]
            if 'interpolated peak phase (with phase gradients' in line:
                interpolated_phase_grad = line.split(' ')[-1][:-1]
            if 'elapsed time' in line:
                #print(rslc, range_peak_position, azimuth_peak_position, interpolated_phase, interpolated_phase_grad)
                rslcs.append(rslc)
                range_peak_positions.append(range_peak_position)
                azimuth_peak_positions.append(azimuth_peak_position)
                interpolated_peak_powers.append(interpolated_peak_power)
                interpolated_phases.append(interpolated_phase)
                interpolated_phases_grad.append(interpolated_phase_grad)


    df = pd.DataFrame(data = {'rslc':rslcs,
                              'range_peak_position': range_peak_positions,
                              'azimuth_peak_position': azimuth_peak_positions,
                              'interpolated_peak_power': interpolated_peak_powers,
                              'interpolated_phase': interpolated_phases,
                              'interpolated_phase_with_gradiant': interpolated_phases_grad})
    #print(df)
    df.to_csv(working_dir + 'ptarg_vs_ovr/ptarg_parsed_output_cr' + str(cr) + '.csv')

def plot_subpixel_location(cr=1):
    rslcs = stack.rslc_list()
    locs = pd.read_csv(working_dir + 'ptarg_vs_ovr/ptarg_parsed_output_cr' + str(cr) + '.csv')

    for rslc in rslcs:
        filename = os.path.basename(rslc)
        par = SLC_Par(rslc + '.par')

        print(rslc, par.dim)
        im = readBin(rslc, par.dim, 'complex64')

        amp = np.absolute(im)
        plt.imshow(amp.T, vmin=0, vmax=2)

        loc = locs[locs['rslc'] == filename]
        plt.scatter(loc.range_peak_position, loc.azimuth_peak_position, marker='x', color='r')

        plt.title(filename + ' Amplitude\n(full_scene)')
        if cr == 1:
            plt.xlim([5120, 5123])
            plt.ylim([5744, 5747])
        elif cr == 2:
            plt.xlim([5109, 5112])
            plt.ylim([5761, 5764])
        plt.colorbar()
        if False:
            plt.show()
            break
        plt.savefig(working_dir + 'ptarg_vs_ovr/subpixel_pngs/' + filename[:-5] + '_fr_cr' + str(cr) + '.png')
        plt.close()


def phase_analysis(show_plot = False):

    cr1 = pd.read_csv(working_dir + 'ptarg_vs_ovr/ptarg_parsed_output_cr1.csv')
    cr2 = pd.read_csv(working_dir + 'ptarg_vs_ovr/ptarg_parsed_output_cr2.csv')

    #ensure dataframes are sorted
    cr1 = cr1.sort_values('rslc')
    cr2 = cr2.sort_values('rslc')

    signal = []
    dates = []

    for i in range(len(cr1)-1):
        phi_cr1 = np.exp((cr1.interpolated_phase.iloc[i] - cr1.interpolated_phase.iloc[i + 1])*1j)
        phi_cr2 = np.exp((cr2.interpolated_phase.iloc[i] - cr2.interpolated_phase.iloc[i + 1])*1j)

        phi_cr1 = np.angle(phi_cr1)
        phi_cr2 = np.angle(phi_cr2)

        # subtract floating from anchored
        signal.append(np.angle(np.exp((phi_cr1 - phi_cr2)*1j)))
        #signal.append(phi_cr1 - phi_cr2)
        dates.append((cr1.rslc.iloc[i][:-5], cr1.rslc.iloc[i + 1][:-5]))

    if show_plot:
        plt.title('Signal = Floating - Anchored')
        plt.plot(range(len(signal)), signal)
        plt.legend()
        plt.show()
        plt.close()

    return dates, signal

def flat_earther():
    #get flat earth phase

    flt_removed = glob.glob(working_dir + sub_dir + 'diff_fr/*.flt')
    for ifg in flt_removed:
        int = ifg[:-4] + '.int'
        flat_earth = ifg[:-4] + '.flat_earth'

        run('subtract_phase', int, ifg, flat_earth, 7621)
        run('rasmph', flat_earth, 7621)


if __name__ == "__main__":


    #amplitude_dispersion()
    point_target(cr_full_scene, cr=1)
    #overwrite_par_files()
    #ptarg_diff_analysis()
    #ptarg_plots()
    #cr_versus_background()

    #read_ptarg_output_files()

    #print_ptarg_calls(cr_full_scene, cr=1)

    #parse_ptarg_output(cr=1)

    #plot_subpixel_location(cr=2)

    #phase_analysis()

    #flat_earther()



