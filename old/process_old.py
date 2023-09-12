from sarlab.gammax import *
#from jinsar.slimsar import *
import matplotlib.pyplot as plt
#from jinsar.utils import jaysone_proj_dir, cr_stack_analysis
from scipy.constants import c
import os
#from inuvik_processing import cr_analysis
from sarlab.tools import lsq
import pylab as pl
import math

master = '20190822'
working_dir = '/local-scratch/users/aplourde/RS2_ITH/'
sub_dir = 'full_scene/'
#sub_dir = 'full_scene/summer_season/'
#sub_dir = 'crop_site1_only/'
#sub_dir = 'crop_site1_only/summer_season/'
#sub_dir = 'crop_large/'
#sub_dir = 'ptarg/'
#sub_dir = 'site1/'
#sub_dir = 'ptarg_site1/'
water_mask = 'watermask_1_1.ras'
#water_mask = None
ingest_cfg = {'password': 'S1m0nF7@SER', 'polarizations': 'HH'}                                                     #2
stack = SLC_stack(dirname=working_dir + sub_dir, name='inuvik_postcr', reg_mask=water_mask, master=master, looks_hr=(2,3), looks_lr=(12,18), multiprocess=True, rdcdem_refine=True, skipmode='None', ingest_cfg=ingest_cfg)
# tiny_stack = SLC_stack(dirname=working_dir+'tiny/', name='inuvik_RS2_U76D', master=master, looks_hr=(4, 6), looks_lr=(12, 18),
#                       multiprocess=False, rdcdem_refine=True, skipmode='refresh', ingest_cfg = ingest_cfg)


itab_first_winter = 14+np.asarray(itab_all(5))
#itab_year_to_year = np.asarray([[0,12],[1,13],[2,14],[3,15],[4,16],[6,17],[7,18],[8,19]])#,[10,21]])
itab_year_to_year = np.asarray([[0,10],
                                [1,11],
                                [2,12],
                                [11,19],
                                [19,31],
                                [31,45],
                                [45,58],
                                [58,70]])#,[10,21]])

#itab_summer_season = np.loadtxt(working_dir + sub_dir + '20190822.itab',  dtype = 'int', skiprows=0, usecols = (0,1))
#itab_summer_season = itab_summer_season - 1


def full_processing_j():
    #stack._slc_mask=np.zeros(53)
    #stack._slc_mask[45:]=1
    #stack = SLC_stack(master=master, looks_hr=(2,3), looks_lr=(12,18), multiprocess=True)
    stack.ingest()
    #aster_par = stack._slc_dir + stack._master + '.slc.par'
    #output = stack.register()
    #stack._reg_itab = itab_merge(itab_merge(itab_merge(itab_lf(66, lag=1), itab_lf(66, lag=2)) , itab_lf(66, lag=3)),itab_cm(66, 31))
    ##output = stack.register_network(mode_crop='master')
    #stack.mk_diff(itab=np.asarray(itab_lf(13))+19, looks='hr', cc_win=7, cc_wgt=0)

    x=1
    #stack.mk_mli_all(looks = ['hr'])
    #stack.ref_dem(dem_name='arctic', _high_res=True)
    #stack.rdc_dem(looks=('hr'))
    #stack.mk_diff(itab=itab_first_winter, looks='hr', cc_win=7, cc_wgt=0)
    #stack.mk_diff(itab='lf', looks='lr', cc_win=7, cc_wgt=0)
    #print(getOptimalLooks(master_par))

def full_processing(looks = 'fr'):
    #stack.ingest()
    #stack.register()
    #stack.mk_mli_all(looks = looks)
    #stack.rdc_dem(looks = looks)
    stack.mk_diff(looks = looks, itab = 'lf')# itab='all', itab_year_to_year)

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

    #stack.ref_dem(dem_name='arctic')
    stack.ref_dem()


def ras_shape(ras_file):
    res = exec(['file', ras_file]).stdout
    shape = (int(res.split(',')[1].split('x')[0]), int(res.split(',')[1].split('x')[1]))
    print(shape)
    return shape

def rmli_width(rmli_file):
    file=open(rmli_file + '.par')
    for line in file:
        fields = line.split()
        if len(fields) > 0:
            if fields[0].strip(':') == 'range_samples':
                return int(fields[1])
    return 0

def geocode_rmli_list(rmli_list, suffix, looks = 'fr'):
    if looks == 'lr':
        dem_dir = working_dir + sub_dir + stack._dem_dir_lr
    elif looks == 'hr':
        dem_dir = working_dir + sub_dir + stack._dem_dir_hr
    elif looks == 'fr':
        dem_dir = working_dir + sub_dir + stack._dem_dir_fr

    gc_map = dem_dir + 'gc_map'
    dem_par = DEM_Par(filename = dem_dir+'seg.dem_par')
    print(dem_par, dem_par.dim[0])
    print('\nGeocoding results...')
    p = mp.ProcessingPool(multiprocess=stack._multiprocess)

    #geocode_back rmli_hr/ave.rmli 3810 dem_hr/gc_map rmli_hr/ave.rmli.geocode 3215

    #p.map(lambda rmli_file: run('geocode_back', rmli_file, rmli_width(rmli_file), gc_map, os.path.splitext(rmli_file)[0] + suffix + '.geo', dem_par.dim[0], None, 1, 0), rmli_list)
    p.map(lambda rmli_file: run('geocode_back', rmli_file, ras_shape(rmli_file)[0], gc_map, os.path.splitext(rmli_file)[0] + suffix + '.geo', dem_par.dim[0], None, 1, 0), rmli_list)
    print('...done geocoding results.\n')

def geocode_ras_list(ras_list, looks = 'fr'):
    if looks == 'lr':
        dem_dir = working_dir + sub_dir + stack._dem_dir_lr
    elif looks == 'hr':
        dem_dir = working_dir + sub_dir + stack._dem_dir_hr
    elif looks == 'fr':
        dem_dir = working_dir + sub_dir + stack._dem_dir_fr

    gc_map = dem_dir + 'gc_map'
    dem_par = DEM_Par(filename = dem_dir+'seg.dem_par')
    p = mp.ProcessingPool(multiprocess=stack._multiprocess)
    #geocode_back 20170901_20170925.diff.adf 7622 ../dem_fr/gc_map 20170901_20170925.diff.adf.geo 15504 - 1

    #geocode_back rmli_hr/ave.rmli 3810 dem_hr/gc_map rmli_hr/ave.rmli.geocode 3215

    #for i in range(len(ras_list)):
    #    print(ras_list[i], ras_shape(ras_list[i])[0], gc_map, os.path.splitext(ras_list[i])[0]+'.geo.ras', dem_par.dim[0])
    p.map(lambda ras_file: run('geocode_back', ras_file, ras_shape(ras_file)[0], gc_map, os.path.splitext(ras_file)[0] + '.geo.ras', dem_par.dim[0], None, 1, 0), ras_list)
    print('...done geocoding results.\n')

def geocode_results(looks = 'fr', format = 'rmli'):
    if format == 'diff':
        diff_dir = working_dir + sub_dir + 'diff_' + looks + '/'
        suffix = '.diff.adf'
        #geocode_rmli_list(glob.glob(diff_dir + '*.diff.adf.ras'), suffix, looks = looks)
        geocode_ras_list(glob.glob(diff_dir + '*.diff.adf.cc.ras'), looks = looks)
    elif format == 'rmli':
        rmli_dir = working_dir + sub_dir + 'rmli_' + looks + '/'
        suffix = '.rmli'
        geocode_rmli_list(glob.glob(rmli_dir + '*.rmli'), suffix, looks=looks)
        #geocode_ras_list(glob.glob(rmli_dir + 'ave.rmli.cr_only.ras'), looks=looks)
        #geocode_ras_list(glob.glob(rmli_dir + 'ave.rmli'), looks=looks)
    else:
        #file_to_geocode = working_dir + sub_dir + 'diff_fr/ave.diff.adf.cc'
        file_to_geocode = working_dir + sub_dir + water_mask
        dem_dir = working_dir + sub_dir + 'dem_fr/'
        gc_map = dem_dir + 'gc_map'
        dem_par = DEM_Par(filename=dem_dir + 'seg.dem_par')
        run('geocode_back', file_to_geocode, 7621, gc_map, file_to_geocode + '.geo', dem_par.dim[0], None, 1, 0)
        run('data2geotiff', dem_par, file_to_geocode + '.geo', 2, file_to_geocode + '.geo'[0] + 'tif')

def convert_to_geotiff(looks = 'fr', type = 'rmli'):
    if type == 'diff':
        diff_dir = working_dir + sub_dir + 'diff_' + looks + '/'
        dem_dir = working_dir + sub_dir + 'dem_' + looks + '/'
        dem_par = DEM_Par(filename=dem_dir + 'seg.dem_par')
        geo_list = glob.glob(diff_dir + '*.geo.ras')
        print("Converting to geotiff...")
        p = mp.ProcessingPool(multiprocess=stack._multiprocess)
        p.map(lambda geo_file: run('data2geotiff', dem_par, geo_file, 2, os.path.splitext(geo_file)[0] + '.tif'), geo_list)
    else:
        rmli_dir = working_dir + sub_dir + 'rmli_' + looks + '/'
        dem_dir = working_dir + sub_dir + 'dem_' + looks + '/'
        dem_par = DEM_Par(filename = dem_dir+'seg.dem_par')
        geo_list = glob.glob(rmli_dir + '*.geo')
        print("Converting to geotiff...")
        p = mp.ProcessingPool(multiprocess=stack._multiprocess)
        p.map(lambda geo_file: run('data2geotiff', dem_par, geo_file, 2, os.path.splitext(geo_file)[0]+'.tif'), geo_list)

    print('... done converting to geotiff.\n')

    # data2geotiff dem_hr/seg.dem rmli_hr/ave.rmli.geocode 2 rmli_hr/ave.rmli.geocode.tif^C

def stack_kml():
    master_par = SLC_Par(filename = stack._slc_dir + stack._master + '.slc.par')
    master_par.kml(hgt=100.0, filename = 'master.kml')


def mean_coherence():
    itab = itab_lf(37)
    #lf_mask = np.asarray([0,1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,0,1,1,0,0,1,0,0,0,1,0,0,0,1,1,0,1,1]).astype(np.bool)
    #lf_idx_list = (1,3,4,5,6,8,9,10,11,12,14,15,16,17,19,20,23,27,31,32,34,35)
    lf_idx_list_summer = (1, 8, 9, 10, 11, 12,  19, 20, 31, 32)
    lf_idx_list_winter = (3, 4, 5, 6, 14, 15, 16, 17, 23, 27, 34, 35)
    itab = np.asarray(itab)
    pair_names = get_pair_names(stack.getPairFilenames(itab))
    cc_filenames = [stack._dir + stack._diff_dir_fr + pair_name + '.cc' for pair_name in pair_names]
    npair = len(pair_names)
    cc_summer_mean = None
    cc_winter_mean = None
    for ii in np.arange(len(pair_names)):
        pair_name = pair_names[ii]
        cc_filename = stack._dir + stack._diff_dir_fr + pair_name + '.cc'
        off_par = OFF_Par(stack._dir + stack._diff_dir_fr + pair_name + '.off')
        cc = CC_Img(filename=cc_filename, par=off_par).array
        if ii in lf_idx_list_summer:
            print('summer', cc_filename)
            if cc_summer_mean is None:
                cc_summer_mean = cc/len(lf_idx_list_summer)
            else:
                cc_summer_mean += cc/len(lf_idx_list_summer)
        if ii in lf_idx_list_winter:
            print('winter', cc_filename)
            if cc_winter_mean is None:
                cc_winter_mean = cc / len(lf_idx_list_winter)
            else:
                cc_winter_mean += cc / len(lf_idx_list_winter)
    render_png(cc_summer_mean.T, 'cc_summer_mean.png',
               cmap=plt.get_cmap('inferno'), vmin=0., vmax=1.)
    render_png(cc_summer_mean.T, 'cc_summer_mean.png',
               cmap=plt.get_cmap('inferno'), vmin=0., vmax=1.)
    # plt.imshow(np.fliplr(cc_summer_mean.T), vmin=0., vmax=1.)
    # plt.figure()
    # plt.imshow(np.fliplr(cc_winter_mean.T), vmin=0., vmax=1.)
    # plt.show()

    x=1


def lf_triplets():
    out_dir=stack._dir+'triplets_vlr/'
    triplet_tab = triplet_tab_lf(len(stack.rslc_list()))
    triplet_filenames = stack.getTripletFilenames(triplet_tab)
    print(triplet_filenames)
    getTripletList(triplet_filenames, out_dir, looks=(24,36), render_range=[-np.pi/6, np.pi/6])
    montage_args = ['montage', '-label', '''%f''', out_dir+'*.png', '-geometry', '''+4+4>''', '-flop', '-tile', 'x6', stack._name+'_phase_triplets_vlr.png']
    res1 = exec(montage_args)


def site_selection():
    fname_pt =jaysone_proj_dir() + 'inuvik/test_readbin_pt/pt'
    fname_seasonal = jaysone_proj_dir() + 'inuvik/test_readbin_pt/tdpcorr_refine'
    fname_linear = jaysone_proj_dir() + 'test_readbin_pt/ldcorr_refine'
    fname_resid = jaysone_proj_dir() + 'test_readbin_pt/resid_meanabs'
    fig_file = jaysone_proj_dir() + 'test_readbin_pt/sensor_placement_mask_figure.png'

    pt = read_pt(fname_pt)
    dim = np.asarray((2640, 11412))
    looks = (66,100)
    dim_ml = dim//looks
    def_seasonal = readBin(fname_seasonal, dim, 'float32', pt=pt).T
    def_linear = readBin(fname_linear, dim, 'float32', pt=pt).T
    resid = readBin(fname_resid, dim, 'float32', pt=pt).T
    sd_seasonal = np.zeros(dim)
    sd_linear = np.zeros(dim)
    frac_finite = np.zeros(dim)
    for ii in np.arange(dim_ml[0]):
        for jj in np.arange(dim_ml[1]):
            x_min = int(ii*looks[0])
            y_min = int(jj*looks[1])
            x_max = x_min + looks[0]
            y_max = y_min + looks[1]
            sd_seasonal[x_min:x_max,y_min:y_max] = np.nanvar(def_seasonal[x_min:x_max,y_min:y_max])**0.5
            sd_linear[x_min:x_max,y_min:y_max] = np.nanvar(def_linear[x_min:x_max,y_min:y_max]) ** 0.5
            frac_finite[x_min:x_max,y_min:y_max] = np.sum(np.isfinite(def_linear[x_min:x_max,y_min:y_max]))/looks[0]/looks[1]
            #def_seasonal[x_min:x_max, y_min:y_max] = np.nanmean(def_seasonal[x_min:x_max, y_min:y_max])
            #def_linear[x_min:x_max, y_min:y_max] = np.nanmean(def_linear[x_min:x_max, y_min:y_max])
            #resid[x_min:x_max, y_min:y_max] = np.nanmean(resid[x_min:x_max, y_min:y_max])

    #plt.imshow(var_seasonal.T, vmin=0)
    plt.figure()
    ax=plt.subplot(161)
    plt.hist(sd_seasonal[np.isfinite(sd_seasonal)].flatten(), bins=100)
    ax.set_title('sd_seasonal')
    ax =plt.subplot(162)
    plt.hist(sd_linear[np.isfinite(sd_linear)].flatten(), bins=100)
    ax.set_title('sd_linear')
    ax =plt.subplot(163)
    plt.hist(def_seasonal[np.isfinite(def_seasonal)].flatten(), bins=100)
    ax.set_title('def_seasonal')
    ax =plt.subplot(164)
    plt.hist(def_linear[np.isfinite(def_linear)].flatten(), bins=100)
    ax.set_title('def_linear')
    ax =plt.subplot(165)
    plt.hist(resid[np.isfinite(resid)].flatten(), bins=100)
    ax.set_title('resid')
    ax =plt.subplot(166)
    plt.hist(frac_finite[np.isfinite(frac_finite)].flatten(), bins=100)
    ax.set_title('frac_finite')

    mask_sd_seasonal = sd_seasonal < 0.0004
    mask_sd_linear = sd_linear < 0.002
    mask_def_seasonal = np.abs(def_seasonal) < 0.0005
    mask_def_linear = np.abs(def_linear) < 0.002
    mask_resid = resid < 1.
    mask_frac_finite = frac_finite > 0.8

    plt.figure()
    plt.imshow((np.isfinite(def_seasonal).astype(np.int) + frac_finite).T)

    plt.show()

    plt.figure()
    ax=plt.subplot(161)
    plt.imshow(sd_seasonal.T)
    ax.set_title('sd_seasonal')
    ax =plt.subplot(162)
    plt.imshow(sd_linear.T)
    ax.set_title('sd_linear')
    ax =plt.subplot(163)
    plt.imshow(def_seasonal.T)
    ax.set_title('def_seasonal')
    ax =plt.subplot(164)
    plt.imshow(def_linear.T)
    ax.set_title('def_linear')
    ax =plt.subplot(165)
    plt.imshow(resid.T)
    ax.set_title('resid')
    ax =plt.subplot(166)
    plt.imshow(frac_finite.T)
    ax.set_title('frac_finite')


    plt.figure()
    ax=plt.subplot(161)
    plt.imshow(mask_sd_seasonal.T)
    ax.set_title('sd_seasonal')
    ax =plt.subplot(162)
    plt.imshow(mask_sd_linear.T)
    ax.set_title('sd_linear')
    ax =plt.subplot(163)
    plt.imshow(mask_def_seasonal.T)
    ax.set_title('def_seasonal')
    ax =plt.subplot(164)
    plt.imshow(mask_def_linear.T)
    ax.set_title('def_linear')
    ax =plt.subplot(165)
    plt.imshow(mask_resid.T)
    ax.set_title('resid')
    ax =plt.subplot(166)
    plt.imshow(mask_frac_finite.T)
    ax.set_title('frac_finite')




    mask = mask_sd_seasonal*mask_sd_linear  *mask_resid*mask_frac_finite
    plt.figure()
    plt.imshow(mask.T)
    plt.savefig(fig_file, dpi=1000)

    plt.show()
    x=1


def subset():
    #subset_dir = working_dir + 'crop_site1_only/'
    #subset_st = stack.subset((4450, 5200, 1000, 1000), subset_dir)
    #subset_dir = working_dir + 'crop_large/'
    #subset_st = stack.subset((4450, 3750, 1000, 2500), subset_dir)
    subset_dir = working_dir + 'ptarg_site1/'
    subset_st = stack.subset((5089, 5714, 64, 64), subset_dir)
    #subset_dir = working_dir + 'site2/'
    #subset_st = stack.subset((4987, 4097, 64, 64), subset_dir)
                                                                                               #2
    subset_st = SLC_stack(dirname=subset_dir, name='inuvik_RS2_U76D', master=master,  looks_hr=(3, 3),
                      looks_lr=(12, 18), multiprocess=True)
    subset_st.mk_mli_all()
    itab_cm = [[0, ii+1] for ii in np.arange(subset_st.n_scenes-1)]
    rslc_pair_list = subset_st.getPairFilenames(itab_cm)
    #getPhaseSimList(rslc_pair_list, subset_dir+'diff_fr/', skipmode=None, multiprocess=True, looks=(1, 1), dem_rdc=subset_st._dir + subset_st._dem_dir_fr + 'dem.rdc')
    #x=1


def boo():
    hoo = '''*** Range and azimuth offset polynomial estimation ***
*** Copyright 2019, Gamma Remote Sensing, v3.4 8-Feb-2019 clw/uw ***
offset estimates: /local-scratch/users/jaysone/projects_active/inuvik/RS2_U76_D/full_scene/rslc//20170808_20140707.offs
cross-correlation or SNR data: /local-scratch/users/jaysone/projects_active/inuvik/RS2_U76_D/full_scene/rslc//20170808_20140707.snr
ISP offset parameter file: /local-scratch/users/jaysone/projects_active/inuvik/RS2_U76_D/full_scene/rslc/20170808_20140707.off_par.init
culled offsets (fcomplex): /local-scratch/users/jaysone/projects_active/inuvik/RS2_U76_D/full_scene/rslc//20170808_20140707.coffs

culled list of offsets and cross-correlation (text format): /local-scratch/users/jaysone/projects_active/inuvik/RS2_U76_D/full_scene/rslc//20170808_20140707.coffsets

number of offset polynomial parameters: 4: a0 + a1*x + a2*y + a3*x*y
number of range samples: 32  number of azimuth samples: 64
number of samples in offset map: 2048
range sample spacing: 243  azimuth sample spacing: 179
solution: 688 offset estimates accepted out of 2048 samples

range fit SVD singular values:        1.60079e+08   1.33877e+00   1.70210e+04   9.86859e+03
azimuth fit SVD singular values:      1.60079e+08   1.33877e+00   1.70210e+04   9.86859e+03
range offset poly. coeff.:               60.19649   4.30331e-04  -8.84781e-06  -6.54165e-09
azimuth offset poly. coeff.:            263.88784  -1.14231e-04  -9.02518e-05   1.37281e-08
model fit std. dev. (samples) range:   1.8779  azimuth:   2.0601
range, azimuth error thresholds:     4.6949     5.1501
cross-correlation threshold:     0.1000

range fit SVD singular values:        1.52919e+08   1.31589e+00   1.67053e+04   9.32565e+03
azimuth fit SVD singular values:      1.52919e+08   1.31589e+00   1.67053e+04   9.32565e+03

*** Improved least-squares polynomial coefficients 1 ***
solution: 651 offset estimates accepted out of 2048 samples
range offset poly. coeff.:               60.09040   4.68911e-04  -4.08181e-06  -3.68809e-09
azimuth offset poly. coeff.:            263.65642  -7.23428e-05  -5.48753e-05   9.73095e-09
model fit std. dev. (samples) range:   1.2937  azimuth:   1.5640
range, azimuth error thresholds:     3.2343     3.9101
cross-correlation threshold:     0.1000

range fit SVD singular values:        1.49139e+08   1.27683e+00   1.61284e+04   8.91828e+03
azimuth fit SVD singular values:      1.49139e+08   1.27683e+00   1.61284e+04   8.91828e+03

*** Improved least-squares polynomial coefficients 2 ***
solution: 605 offset estimates accepted out of 2048 samples
range offset poly. coeff.:               60.28965   4.44731e-04  -3.73155e-05   3.69074e-09
azimuth offset poly. coeff.:            263.70829  -1.13323e-04  -6.31754e-05   1.33531e-08
model fit std. dev. (samples) range:   0.9825  azimuth:   1.3260
range, azimuth error thresholds:     2.4562     3.3149
cross-correlation threshold:     0.1000

range fit SVD singular values:        1.48196e+08   1.25282e+00   1.57900e+04   8.73203e+03
azimuth fit SVD singular values:      1.48196e+08   1.25282e+00   1.57900e+04   8.73203e+03

*** Improved least-squares polynomial coefficients 3 ***
solution: 578 offset estimates accepted out of 2048 samples
range offset poly. coeff.:               60.19587   4.85405e-04  -1.96935e-05  -1.43951e-09
azimuth offset poly. coeff.:            263.69871  -1.26438e-04  -6.62990e-05   1.55611e-08
model fit std. dev. (samples) range:   0.8715  azimuth:   1.2355
range, azimuth error thresholds:     2.1788     3.0888
cross-correlation threshold:     0.1000

range fit SVD singular values:        1.44980e+08   1.23378e+00   1.55360e+04   8.46429e+03
azimuth fit SVD singular values:      1.44980e+08   1.23378e+00   1.55360e+04   8.46429e+03

*** Improved least-squares polynomial coefficients 4 ***
solution: 548 offset estimates accepted out of 2048 samples
range offset poly. coeff.:               60.13542   5.19578e-04  -1.57834e-05  -3.63121e-09
azimuth offset poly. coeff.:            263.70332  -1.34701e-04  -6.92403e-05   1.71977e-08
model fit std. dev. (samples) range:   0.7626  azimuth:   1.1621
range, azimuth error thresholds:     1.9064     2.9052
cross-correlation threshold:     0.1000

range fit SVD singular values:        1.43209e+08   1.21510e+00   1.54103e+04   8.39063e+03
azimuth fit SVD singular values:      1.43209e+08   1.21510e+00   1.54103e+04   8.39063e+03

*** Improved least-squares polynomial coefficients 5 ***
solution: 527 offset estimates accepted out of 2048 samples
range offset poly. coeff.:               60.16852   5.25414e-04  -2.26481e-05  -3.67521e-09
azimuth offset poly. coeff.:            263.75607  -1.46366e-04  -7.34500e-05   1.74918e-08
model fit std. dev. (samples) range:   0.6890  azimuth:   1.1052
range, azimuth error thresholds:     1.7226     2.7629
cross-correlation threshold:     0.1000

range fit SVD singular values:        1.42869e+08   1.21249e+00   1.53344e+04   8.31067e+03
azimuth fit SVD singular values:      1.42869e+08   1.21249e+00   1.53344e+04   8.31067e+03

*** Improved least-squares polynomial coefficients 6 ***
solution: 515 offset estimates accepted out of 2048 samples
range offset poly. coeff.:               60.14997   5.35978e-04  -1.94668e-05  -4.96623e-09
azimuth offset poly. coeff.:            263.76588  -1.42039e-04  -7.71843e-05   1.74334e-08
model fit std. dev. (samples) range:   0.6587  azimuth:   1.0710

total number of culling iterations: 6
final solution: 515 offset estimates accepted out of 2048 samples

final range offset poly. coeff.:             60.14997   5.35978e-04  -1.94668e-05  -4.96623e-09
final range offset poly. coeff. errors:   5.43279e-01   1.39866e-04   8.55053e-05   1.94728e-08

final azimuth offset poly. coeff.:            263.76588  -1.42039e-04  -7.71843e-05   1.74334e-08
final azimuth offset poly. coeff. errors:   8.83301e-01   2.27404e-04   1.39021e-04   3.16602e-08

final model fit std. dev. (samples) range:   0.6587  azimuth:   1.0710

binary culled offsets: /local-scratch/users/jaysone/projects_active/inuvik/RS2_U76_D/full_scene/rslc//20170808_20140707.coffs
updating ISP offset parameters: /local-scratch/users/jaysone/projects_active/inuvik/RS2_U76_D/full_scene/rslc/20170808_20140707.off_par.init

user time (s):         0.010
system time (s):       0.000
elapsed time (s):      0.010
'''


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

        result12 = run('adf', ifg_diff, ifg_adf, cc_adf, off_par.dim[0],
                       adf_alpha, adf_nfft, adf_ncc, adf_step, None, None,
                       adf_wfrac)
        ifg_adf.ras()


# Part 3 static atmospheric correction
def Static_atmosphere_ALL(looks = 'fr'):
    inf_list=glob.glob(working_dir + sub_dir + 'diff_'+looks+'/*.adf.unw.hc')
    cc_list = glob.glob(working_dir + sub_dir + 'diff_' + looks + '/*.adf.cc')
    #inf_list = glob.glob('diff_hr/20180731_20180811.diff.adf')
    #cc_list = glob.glob('diff_hr/20180731_20180822.cc')

    mlipar = MLI_Par(filename=working_dir + sub_dir + 'rmli_hr/ave.rmli.par')
    rdcDEM = DEM(filename=working_dir + sub_dir + 'dem_hr/dem.rdc', par=mlipar)
    dem = rdcDEM.array
    [mask, colormap] = read_ras(working_dir + sub_dir + 'diff_'+looks+'/ave.diff.adf.cc.ras')

    jval = np.where(mask > 0)
    mask[jval] = 1

    print(inf_list)

    for idx, inf in enumerate(inf_list):
        _dir, _inf = os.path.split(inf)
        prefix = _inf[0:17]
        print(prefix)
        ipar = OFF_Par(filename=working_dir + sub_dir + 'diff_'+looks+'/' + prefix + '.off')
        intf_obj = Interferogram(filename=inf, par=ipar)
        intf_out_obj = Interferogram(filename=inf + '.satm', par=ipar)

        cc_obj = Img(filename=cc_list[idx], par=mlipar)
        cc = cc_obj.array
        intf = intf_obj.array  # Read

        #cc = readBin(cc_list[idx], (3810,3810), 'float32')
        #intf = readBin(inf, (3810,3810), 'complex64')

        # intf_Crop=np.copy(intf[1:800,1:1800])
        # cc_Crop=np.copy(cc[1:800,1:1800])
        # dem_Crop=np.copy(dem[1:800,1:1800])
        cc_Crop = cc * mask.T

        idx_valid = np.where(cc_Crop >= 0.8)
        phase_data = intf[idx_valid]
        mean_phasor = np.mean(phase_data)
        phase_data = phase_data * np.conj(mean_phasor)
        # phase_data *=np.exp(math.pi*0.5j)
        phase_data = np.angle(phase_data)
        hgt_data = dem[idx_valid]
        mean_hgt = np.mean(hgt_data)
        hgt_data -= mean_hgt

        LsqSL = lsq.lsq_fit(phase_data, hgt_data, degree=1)

        #stc_crr= np.exp(-1j*((dem-mean_hgt)*LsqSL[0][0]+LsqSL[0][1]))*mean_phasor
        hgt_cor = lsq.poly(dem - mean_hgt, LsqSL[0])
        stc_crr = np.exp(-1j * hgt_cor[0])
        intf_out_obj.array = intf * stc_crr * np.conj(mean_phasor)  # *np.exp(math.pi*0.5j)
        i = np.arange(-1000, 1100, 100)
        j = ((i) * LsqSL[0][0]) + LsqSL[0][1]
        plt.plot(i, j, linestyle='solid');
        plt.plot(hgt_data, phase_data, ',');
        plt.savefig('atm_test.png')
        intf_out_obj.push()
        intf_out_obj.ras()




if __name__ == '__main__':

    look = 'hr'

    #cr_loc()
    #subset()
    #active_layer_def_template(working_dir + '../met_data/', tiny_stack.rslc_list(), working_dir+'../active_layer.txt')

    #get_refdem()
    #full_processing_j()

    full_processing(looks = look)

    #refilter(looks = look)

    #unwrap(looks = look, method = None)

    #Static_atmosphere_ALL(looks = look)

    geocode_results(format = None)
    #geocode_results(looks = 'hr', format = 'diff')
    #geocode_results(looks = 'lr')

    #convert_to_geotiff()
    #convert_to_geotiff(looks = 'hr', type = 'diff')
    #convert_to_geotiff(looks = 'lr')

    #testsite_cr_analysis()
    #stack_kml()
    #mean_coherence()
    #lf_triplets()
    #stack.plot_2d_baselines(itab=itab_lf(stack.n_scenes), filename=stack._name+'_baseplot.png')


    #remove_atm()


    #base_calc slc_tab rslc/20190729.rslc.par 20190729.bperp itab 1
    #rslc rslc rslc.par slc_tab
