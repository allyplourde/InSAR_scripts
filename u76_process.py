from sarlab.gammax import *
import matplotlib.pyplot as plt
import os
from scipy.ndimage import uniform_filter
import scipy.ndimage.morphology as morph
import glob
import xml.etree.ElementTree as ET

from cr_phase_to_deformation import get_itab_diffs

#working_dir = '/local-scratch/users/aplourde/RS2_ITH/post_cr_installation/'
#sub_dir = 'full_scene/'
#working_dir = '/local-scratch/users/aplourde/HDS/projects/southern_ITH/RS2_U76_D/'
#sub_dir = 'crop_sites/'; master = '20180827'
#sub_dir = 'full_scene/'; master = '20180827'
working_dir = '/local-scratch/users/jaysone/projects_active/inuvik/RS2_SLA27_D/'
sub_dir = 'full_scene/'; master= '20160811'
#sub_dir = 'temp/'; master = '20180827'
#water_mask = 'watermask_1_1_eroded.ras'
water_mask = None
ingest_cfg = {'password': 'S1m0nF7@SER', 'polarizations': 'HH'}
stack = SLC_stack(dirname=working_dir + sub_dir, name='inuvik_postcr', reg_mask=water_mask, master=master, looks_hr=(3,12), looks_lr=(12,18), multiprocess=False, rdcdem_refine=True, skipmode='None', ingest_cfg=ingest_cfg)
# tiny_stack = SLC_stack(dirname=working_dir+'tiny/', name='inuvik_RS2_U76D', master=master, looks_hr=(4, 6), looks_lr=(12, 18),
#                       multiprocess=False, rdcdem_refine=True, skipmode='refresh', ingest_cfg = ingest_cfg)


#itab = np.loadtxt(working_dir + sub_dir + 'itab_cm_20190729', dtype='int', skiprows=0, usecols=(0, 1))
itab = np.loadtxt(working_dir + sub_dir + 'itab_snow_lf', dtype='int', skiprows=0, usecols=(0, 1))
itab = itab-1

#itab_summer_season = np.loadtxt(working_dir + sub_dir + '20190822.itab',  dtype = 'int', skiprows=0, usecols = (0,1))
#itab_summer_season = itab_summer_season - 1
#itab_goodcc = np.loadtxt(working_dir + sub_dir + '20190822_goodcc.itab',  dtype = 'int', skiprows=0, usecols = (0,1))
#itab_goodcc = itab_goodcc - 1


def full_processing(looks='fr'):

    #stack.ingest()
    #stack.register()
    #stack.mk_mli_all(looks = looks)
    #stack.rdc_dem(looks=looks)
    #stack._diff_dir_fr = 'diff_lf_winter/'
    #stack._diff_dir_hr = 'diff_hr_lf_winter/'
    stack.mk_diff(looks=looks, itab=itab)# itab='all', itab_year_to_year)


def get_refdem():
    # Create stack
    #ingest_cfg_tmp = {'password': 'S1m0nF7@SER', 'polarizations': 'HH'}
    #stack_tmp = SLC_stack(dirname=working_dir + sub_dir, master=master, slcreg_refine=True, rdcdem_refine=True, ingest_cfg=ingest_cfg_tmp)

    # raw/ -> slc/
    #stack_tmp.ingest()

    # master slc par file
    #master_par = stack_tmp._slc_dir + stack_tmp._master + '.slc.par'

    #master_par = os.path.join(working_dir, master_par)
    #print(master_par)

    # print possible multi-look values
    #print(getOptimalLooks(master_par))

    stack.ref_dem(dem_name='arctic', _high_res=False)
    #stack.ref_dem()


def subset():
    #subset_dir = working_dir + 'crop_site1_only/'
    #subset_st = stack.subset((4450, 5200, 1000, 1000), subset_dir)
    #subset_dir = working_dir + 'crop_large/'
    #subset_st = stack.subset((4450, 3750, 1000, 2500), subset_dir)
    #subset_dir = working_dir + 'ptarg_site1/'
    #subset_st = stack.subset((5089, 5714, 64, 64), subset_dir)
    #subset_dir = working_dir + 'ptarg_site2/'
    #subset_st = stack.subset((4987, 4097, 64, 64), subset_dir)
    subset_dir = working_dir + 'ovr_site1/'
    #subset_st = stack.subset((5085, 5710, 72, 72), subset_dir)
    subset_st = stack.subset((5089, 5714, 64, 64), subset_dir)
                                                                                               #2
    subset_st = SLC_stack(dirname=subset_dir, name='inuvik_RS2_U76D', master=master,  looks_hr=(3, 3),
                      looks_lr=(12, 18), multiprocess=True)
    subset_st.mk_mli_all()
    itab_cm = [[0, ii+1] for ii in np.arange(subset_st.n_scenes-1)]
    rslc_pair_list = subset_st.getPairFilenames(itab_cm)
    #getPhaseSimList(rslc_pair_list, subset_dir+'diff_fr/', skipmode=None, multiprocess=True, looks=(1, 1), dem_rdc=subset_st._dir + subset_st._dem_dir_fr + 'dem.rdc')
    #x=1


def unwrap(looks = 'fr', method = 'mcf'):
    diff_list = glob.glob(working_dir + sub_dir + "diff_"+looks+"/*.diff.adf")
    diff_cc_list = glob.glob(working_dir + sub_dir + "diff_"+looks+"/*.diff.adf.cc")

    cc_file = working_dir + sub_dir + 'diff_' + looks + '/ccfile.txt'
    if not os.path.exists(cc_file):
        diff_cc_list = np.asarray(diff_cc_list)
        diff_cc_list.tofile(cc_file, sep='\n', format='%s')

    rslc_list = stack.rslc_list()
    par = SLC_Par(rslc_list[0] + '.par')

    #unwrap from road
    if looks == 'hr':
        width = int(par.dim[0] / 2)
        starting_pixel = [1630, 1290]
    elif looks == 'fr':
        width = par.dim[0]
        starting_pixel = [3260, 3870]
    else:
        return

    ave_cc = working_dir + sub_dir + "diff_" + looks + "/ave.diff.adf.cc"
    print(('ave_image', cc_file, width, ave_cc))
    run('ave_image', cc_file, width, ave_cc)
    run('rascc', ave_cc, None, width, ave_cc + '.ras')

    if method == 'mcf':
        for diff in diff_list:
            run('mcf', diff, diff+'.cc', None, diff+'.unw', width)
            run('ras8_float', diff + '.unw', None, width, diff + '.unw.ras')
    elif method == 'grasses':

        for diff in diff_list:

            # create flag file
            corr_thr = 0.7
            run('corr_flag', diff + '.cc', diff + '.flag70', width, corr_thr)
            run('residue_cc', diff, diff + '.flag70', width)
            run('tree_cc', diff + '.flag70', width)
            run('grasses', diff, diff+'.flag70', diff+'.unw', width,
                None, None, None, None, starting_pixel[0], starting_pixel[1], 1)
            if looks == 'hr':
                corr_thr = 0.3
                run('corr_flag', diff + '.cc', diff + '.flag30', width, corr_thr)
                run('residue_cc', diff, diff + '.flag30', width)
                run('tree_cc', diff + '.flag30', width)
                run('grasses', diff, diff + '.flag30', diff + '.unw', width,
                    None, None, None, None, starting_pixel[0], starting_pixel[1], 1)
                #corr_thr = 0.2
                #run('corr_flag', diff + '.cc', diff + '.flag20', width, corr_thr)
                #run('residue_cc', diff, diff + '.flag20', width)
                #run('tree_cc', diff + '.flag20', width)
                #run('grasses', diff, diff + '.flag20', diff + '.unw', width,
                #    None, None, None, None, starting_pixel[0], starting_pixel[1], 1)


            run('rasrmg', diff + '.unw', None, width, None, None, None, None, None, None, None, None, None, None,
                diff + '.unw.ras', diff + '.cc', None, corr_thr)
    else:
        return


def refilter(looks = 'fr'):

    ifg_list = glob.glob(working_dir + sub_dir + 'diff_' + looks + '/*.diff.adf')

    for ifg_diff in ifg_list:
        offpar_file = ifg_diff[:-9] + '.off'
        off_par = OFF_Par(filename=offpar_file)
        ifg_adf = Interferogram(filename=ifg_diff + '.adf', par=off_par)
        cc_adf = ifg_diff + '.adf.cc'

        adf_alpha = 0.4
        adf_nfft = 64
        adf_ncc = 15
        adf_step = adf_nfft/8
        adf_wfrac = 0.2  # adf filter settings

        print('adf', ifg_diff, ifg_adf, cc_adf, off_par.dim[0],
                       adf_alpha, adf_nfft, adf_ncc, adf_step, None, None,
                       adf_wfrac)

        result = run('adf', ifg_diff, ifg_adf, cc_adf, off_par.dim[0],
                       adf_alpha, adf_nfft, adf_ncc, adf_step, None, None,
                       adf_wfrac)
        ifg_adf.ras()


def extract_flat_earth(looks='fr'):

    ifgs = glob.glob(working_dir + sub_dir + 'diff_' + looks + '/*.int')
    master_par = stack.master_slc_par

    for ifg in ifgs:

        basename = os.path.basename(ifg)
        root = basename.split('.')[0]

        flt = working_dir + sub_dir + 'diff_' + looks + '/' + root + '.flt'
        tmp = working_dir + sub_dir + 'diff_' + looks + '/' + root + '.tmp'
        flat_earth = working_dir + sub_dir + 'diff_' + looks + '/' + root + '.flat_earth'


        flat_earth_removed = readBin(flt, master_par.dim, 'complex64')
        writeBin(tmp, np.angle(flat_earth_removed))
        run('subtract_phase', ifg, tmp, flat_earth, master_par.dim[0], 1)
        run('rasmph', flat_earth, master_par.dim[0], None, None, None, None, None, None, None, flat_earth + '.ras', 0)



#generates a water mask based on temporal min amplitude image
#stolen from jayson
def mk_water_mask(indir, outdir, dates, look_str='1_1', iterations=1, SLA=False):
    #hand-picked to speed up and improve statistics
    #fnames = [indir + date + '.rmli' for date in dates]
    print(indir)
    print(outdir)
    fnames = sorted(glob.glob(indir + '*.rmli'))
    par = MLI_Par(indir+'rmli_' + look_str + '.ave.par')
    rmli_min = np.zeros(par.dim)
    N=0
    for fname in fnames:
        print('reading ', fname, ' ...')
        rmli = readBin(fname, par.dim, 'float32')
        rmli[rmli == 0] = 1000.
        rmli = uniform_filter(rmli, 5)
        if N==0:
            rmli_min = rmli
        else:
            rmli_min = np.minimum(rmli_min, rmli)
        N+=1

    rmli_min[rmli_min <= 0] = np.min(rmli_min[rmli_min > 0])
    writeBin(outdir + 'min.rmli', rmli_min)
    rmli_min = readBin(outdir + 'min.rmli', par.dim, 'float32')
    if SLA:
        #spotlight so special treatment for variable noise floor
        thresh_dB =10*np.log10(np.min(rmli_min[0:4000,:], axis=0))+3
        plt.figure()
        plt.plot(thresh_dB)
        plt.ylim((-30, None))
    else:
        thresh_dB = -18.

    rmli_min_db = 10*np.log10(rmli_min)
    water_mask_init = np.zeros(par.dim)
    for ii in np.arange(par.dim[0]):
        water_mask_init[ii, rmli_min_db[ii,:] > thresh_dB] = 1.
    #water_mask_init[rmli_min_db < -21] = 1.
    water_mask = morph.binary_closing(morph.binary_opening(water_mask_init, iterations=iterations), iterations=iterations)
    writeBin(outdir + 'dem_' + look_str + '/water_mask_' + look_str, water_mask.astype('float32'))
    plt.figure()
    ax1 = plt.subplot(3,1,1)
    plt.imshow(rmli_min_db.T, vmin=-30, vmax=-10)
    plt.subplot(3,1, 2, sharex=ax1, sharey=ax1)
    plt.imshow(water_mask_init.T, vmin=0, vmax=1.)
    plt.subplot(3,1, 3, sharex=ax1, sharey=ax1)
    plt.imshow(water_mask.T, vmin=0, vmax=1.)
    #render_png((water_mask.T).astype('float'),outdir + 'water_mask_autogen.png', vmin=0, vmax=1.)

    #plt.figure()
    #plt.hist(rmli_min_db.flatten(), bins=np.linspace(-50., 0., 200))

    plt.show()


def get_lut_gains(lut):
    #parse the xml
    tree = ET.parse(lut)
    root = tree.getroot()

    gainstr = root.find('gains').text
    gainsplt = gainstr.split(' ')
    gains = [float(g) for g in gainsplt]

    return np.asarray(gains)


def raw_to_sigma0(working_dir, show_plot=False):
    slcs = stack.slc_list()
    import rasterio
    import rasterio.plot as rplt
    for slc in slcs:
        par = SLC_Par(filename= slc + '.par')
        root = os.path.basename(slc)
        date = root.split('.')[0]
        #if date == '20180827':
        if True:
            slc_im = readBin(slc, par.dim, 'complex64')
            rmli_im = readBin(working_dir + 'full_scene/rmli_1_1/' + date + '.rmli', [7621, 11393], 'float32')
            rslc_im = readBin(working_dir + 'full_scene/rslc/' + date + '.rslc', [7621, 11393], 'complex64')
            raw = working_dir + 'full_scene/raw/' + date + '/imagery_HH.tif'
            tif = rasterio.open(raw)

            real = tif.read(1)
            imag = tif.read(2)
            data = real + 1j * imag
            lut = working_dir + 'full_scene/raw/' + date + '/lutSigma.xml'
            gains = get_lut_gains(lut)

            sigma0 = np.zeros(data.T.shape)
            for i, range_sample in enumerate(data.T):
                sigma0[i] = np.abs(range_sample)**2 / (gains[i]**2)#+ 1j*np.angle(range_sample)

            sig0_file = working_dir + 'full_scene/sigma0/' + date + '.sigma0'

            writeBin(sig0_file, sigma0)


            if show_plot:
                xlim = [5020, 5120]
                ylim = [5900, 5820]
                ymax = data.shape[0]
                xmax = data.shape[1]
                plt.subplot(131)
                plt.title('RMLI')
                #plt.imshow(20*np.log10(np.abs(slc_im.T)))
                plt.imshow(10*np.log10(rmli_im.T), vmin=-20, vmax=10)
                plt.colorbar()
                plt.xlim([5020, 5120])
                plt.ylim([5780, 5650])
                plt.subplot(132)
                plt.title('RSLC')
                #plt.imshow(np.abs(data))
                plt.imshow(20 * np.log10(np.abs(rslc_im.T)), vmin=-20, vmax=10)
                plt.colorbar()
                plt.xlim([5020, 5120])
                plt.ylim([5780, 5650])
                #plt.xlim([xmax - lim for lim in xlim])
                #plt.ylim(ylim)
                plt.subplot(133)
                plt.title('Sigma0 (dB)')
                plt.imshow(10*np.log10(sigma0.T), vmin=-20, vmax=10)
                plt.colorbar()
                plt.xlim([xmax - lim for lim in xlim])
                plt.ylim(ylim)
                plt.show()


def mk_coherance_maps(looks='fr'):
    files = glob.glob(working_dir + sub_dir + stack._diff_dir_hr + '*diff.cc')
    print(stack._master)
    print(stack._dir + stack._rmli_dir_hr  + stack._master + '.rmli.par')
    master_par = MLI_Par(stack._dir + stack._rmli_dir_hr + stack._master + '.rmli.par')

    itab_file = stack._dir + 'itab_lf_winter'
    RSLC_tab = stack._dir + 'RSLC_tab'

    files = get_itab_diffs(files, itab_file, RSLC_tab)

    sum = np.zeros(master_par.dim)
    for f in files:
        im = readBin(f, master_par.dim, 'float32')
        #plt.imshow(im.T, cmap='Greys_r')
        #plt.show()
        #break

        sum += im

    ave = sum / len(files)
    plt.imshow(ave.T, cmap='RdYlGn')
    plt.colorbar()
    plt.show()

    writeBin(stack._dir + stack._rmli_dir_hr + 'ave_coh_' + itab_file + '.rmli', ave)


if __name__ == '__main__':

    look = 'hr'

    #print(getOptimalLooks(stack.master_slc_par))
    """ Main Tool Chain """
    #subset()
    #get_refdem()
    full_processing(looks = look)
    #refilter(looks = look)
    #unwrap(looks = look, method = None)

    #extract_flat_earth(look)
    ### .flt is flat earth removed xxxsubtract_flat_earth(look)xxx

    #raw_to_sigma0(working_dir)

    """ Utils """
    #mk_coherance_maps(looks=look)






