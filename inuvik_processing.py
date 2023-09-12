import numpy as np
import os
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy import constants
import glob
import copy
import jinsar.met_data.snow as snow
from sarlab.tools import render_png
from sarlab.gammax import SLC_stack, DEM_Par, itab_all, itab_merge, write_itab, itab_from_matrix, DIFF_Par, readBin, writeBin,\
    write_ras, read_ras, itab_remove_duplicates, itab_cm, get_pair_name_root, read_pt, cmap_gamma
from jinsar.utils import func_cached, cr_stack_analysis, G_matrix, M_matrix, itab_idx_list
from jinsar.permafrost import active_layer_def_template, read_active_layer_template

#this is for the 2021 analysis of the Inuvik U76D and SLA27D stacks for the various papers
#paper specific analysis should NOT go here!!!

# Allison Plourde version

working_dir = '/local-scratch/users/aplourde/RS2_ITH/full_scene/'
master_U76D = '20170808'
master_SLA27D = '20160811'
ingest_cfg = {'polarizations': None}
stack_U76D = SLC_stack(dirname=working_dir ,name='inuvik_RS2_U76_D', master=master_U76D, looks_hr=(2,3), looks_lr=(12,18), multiprocess=True, rdcdem_refine=True, skipmode='None', ingest_cfg=ingest_cfg)
#stack_SLA27D = SLC_stack(dirname=jaysone_proj_dir() + 'inuvik/RS2_SLA27_D/full_scene/',name='inuvik_RS2_SLA27_D', master=master_SLA27D, looks_hr=(1,4), looks_lr=(11,44), multiprocess=True, rdcdem_refine=True, skipmode='None', ingest_cfg=ingest_cfg)
#stack_U76D_small = copy.copy(stack_U76D)
#stack_U76D_small._dir = jaysone_proj_dir() + 'inuvik/RS2_U76_D/small/'
#stack_U76D_big = copy.copy(stack_U76D)
#stack_U76D_big._dir = jaysone_proj_dir() + 'inuvik/RS2_U76_D/big/'

dates_all_U76D = ['20140707','20140824','20140917','20141128','20141222','20150115','20150208','20150328',
                '20150515','20150608','20150726','20150819','20150912','20151006','20151123','20151217',
                '20160110','20160203','20160227','20160813','20160906','20160930','20161024','20161211',
                '20170104','20170128','20170221','20170317','20170504','20170528','20170621','20170808',
                '20170901','20170925','20171206','20171230','20180123','20180216','20180312','20180405',
                '20180429','20180523','20180616','20180710','20180803','20180827','20180920','20181014',
                '20181201','20181225','20190118','20190211','20190307','20190331','20190424','20190518',
                '20190611','20190729','20190822','20190915','20191009','20191126','20200113','20200206',
                '20200301','20200325','20200418','20200512','20200605','20200629','20200816','20200909',
                '20201003','20201027','20201214','20210107']

dates_all_SLA27D = ['20120222', '20120317', '20120410', '20120504', '20120528', '20120621', '20120715', '20120808',
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
             '20200111', '20200204', '20200228', '20200323', '20200416', '20200510', '20200603', '20200627',
             '20200814', '20200907', '20201001', '20201025', '20201212', '20210105', '20210129']


fractional_day_U76D = 0.66
fractional_day_SLA27D = 0.63
inuvik_RS2_U76D_scene_centre = [-133.6274042, 68.5861113]
#inuvik_RS2_SLA27D_scene_centre = [-133.6444803, 68.3595694]
#ITH CR pixel positions (ref RS2_U76_D/full_scene/rslc/20170808.rslc)
cr_loc = {}
cr_loc['CR_A1'] = np.asarray([5049, 5680]) #1
cr_loc['CR_F1'] = np.asarray([5038, 5697]) #2
cr_loc['CR_A2'] = np.asarray([4963, 4048]) #3
cr_loc['CR_F2'] = np.asarray([4964, 4072]) #4
cr_loc['CR_F3'] = np.asarray([4926, 4054]) #6
cr_loc['CR_F4'] = np.asarray([4948, 4063]) #5

"""
cr_loc_small = copy.deepcopy(cr_loc)
subset_offset_small = np.asarray((4500, 3492))
for key in cr_loc_small:
    cr_loc_small[key]-=subset_offset_small

cr_loc_big = copy.deepcopy(cr_loc)
subset_offset_big = np.asarray((276, 216))
for key in cr_loc_big:
    cr_loc_big[key]-=subset_offset_big
"""

def print_cr_list(cr_loc_, rad=1):
    for key in cr_loc_:
        print(cr_loc_[key][0] - rad, cr_loc_[key][1] - rad, 2*rad+1, 2*rad+1)



def get_inuvik_RS2_U76D_era5():
    return func_cached(snow.get_era5_snow_study_series, inuvik_RS2_U76D_scene_centre, '20120201', '20210131')

def get_inuvik_RS2_SLA27D_era5():
    return func_cached(snow.get_era5_snow_study_series, inuvik_RS2_SLA27D_scene_centre, '20110701', '20210131')


def gen_phase_sim_for_cr_analysis():
    pass


def cr_analysis():
    stack = stack_U76D
    win_sz = np.asarray((19,19))
    cr_size = 0.45 #m leg length

    #two sets of CRs were installed
    #the first set was installed 20180819 (first slc was 20180827)
    #the second set was installed 20190707 (first slc was 20190729)
    #so I need to get a set of single-look common-master .sim_unw files from 20180827 to present (total of 32)
    #I currently have 20 slc since the second installation
    #I have enough to get a height error estimate for each CR and do a topographic phase correction


    #get list of sim_unw files to flatten the SLCs with


    sim_unw_list = [stack._dir + 'diff_fr/'+'20140707_'+name_root(rslcname)+'.sim_unw' for rslcname in stack.rslc_list()]
    for ii in np.arange(len(sim_unw_list)):
        if name_root(stack.rslc_list()[ii]) == '20140707':
            sim_unw_list[ii] = None

    print(stack.rslc_list())
    stack_res_FSE = cr_stack_analysis(stack.rslc_list()[-8:], sim_unw_list[-8:], cr_loc_FSE, win_sz, cr_size, 'ITH FSE CR')
    stack_res_ANW = cr_stack_analysis(stack.rslc_list()[-8:], sim_unw_list[-8:], cr_loc_ANW, win_sz, cr_size, 'ITH_ANW_CR')

    #compute relative vertical deformation between two CRs
    par = stack.master_slc_par
    wavelength = constants.c/par['radar_frequency']
    phi_diff = stack_res_FSE['phi_cr']-stack_res_ANW['phi_cr']
    phi_diff -=phi_diff[0]
    phi_diff = np.angle(np.exp(1j*phi_diff))
    phi_diff_unw = phunwrap(phi_diff)
    def_vert = phi_diff_unw*(-wavelength/4/np.pi)/np.cos(np.radians(par['incidence_angle']))

    #compute relative deformation between anchored CR and floating clutter
    phi_diff_clut = stack_res_FSE['phi_clut']-stack_res_ANW['phi_cr']
    phi_diff_clut -=phi_diff_clut[0]
    phi_diff_clut = np.angle(np.exp(1j*phi_diff_clut))
    phi_diff_clut_unw = phunwrap(phi_diff_clut)
    def_vert_clut = phi_diff_clut_unw*(-wavelength/4/np.pi)/np.cos(np.radians(par['incidence_angle']))

    #compute mean and difference offsets in rng and azimuth
    drg_mean = (stack_res_FSE['drg'] + stack_res_ANW['drg'])/2
    daz_mean = (stack_res_FSE['daz'] + stack_res_ANW['daz'])/2

    var_drg_mean = np.mean(((drg_mean - np.mean(drg_mean))**2))
    var_daz_mean = np.mean(((daz_mean - np.mean(daz_mean))**2))

    drg_diff = (stack_res_FSE['drg'] - stack_res_ANW['drg'])
    daz_diff = (stack_res_FSE['daz'] - stack_res_ANW['daz'])

    var_drg_diff = np.mean(((drg_diff - np.mean(drg_diff))**2))
    var_daz_diff = np.mean(((daz_diff - np.mean(daz_diff))**2))


    #compare clutter phase solutions for the two patches
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(stack_res_FSE['phi_clut'])
    plt.plot(stack_res_ANW['phi_clut'])
    plt.title('ITH CR pair wrapped clutter phase comparison')
    plt.xlabel('Acquisition #')
    plt.ylabel('Wrapped phase [radians]')
    plt.legend(['FSE', 'ANW'])


    plt.subplot(2, 2, 2)
    plt.plot(def_vert)
    #TODO is this really the reverse!!!!!
    plt.plot(-def_vert_clut)
    plt.title('ITH CR pair relative vertical displacement time series.')
    plt.xlabel('Acquisition #')
    plt.ylabel('Vertical Deformation [m]')
    plt.legend(['CR def', 'Clutter def'])

    plt.subplot(2, 2, 3)
    plt.plot(drg_mean); plt.ylim((-1,1))
    plt.plot(drg_diff)
    plt.title('ITH CR pair sub-pixel range offsets.')
    plt.xlabel('Acquisition #')
    plt.ylabel('Range distance [pixels]')
    plt.legend(['mean', 'difference'])

    plt.subplot(2,2,4)
    plt.plot(daz_mean); plt.ylim((-1,1))
    plt.plot(daz_diff)
    plt.title('ITH CR pair sub-pixel azimuth offsets.')
    plt.xlabel('Acquisition #')
    plt.ylabel('Azimuth distance [pixels]')
    plt.legend(['mean', 'difference'])


    print('stddev_drg_mean, stddev_daz_mean:', np.sqrt(var_drg_mean/2), np.sqrt(var_daz_mean/2))
    print('stddev_drg_diff, stddev_daz_diff:', np.sqrt(var_drg_diff/2), np.sqrt(var_daz_diff/2))
    plt.show()



def phunwrap(phi):
    #simple 1D phase unwrapper
    phi_unw = np.cumsum(np.angle(np.exp(1j*(phi[1:] - phi[0:-1]))))
    phi_unw = np.insert(phi_unw, 0, 0)
    return phi_unw

# def cr_loc():
#     #manual read off of the cr pixel locations from the ave.rmli.cr_only.geo.ras image and conversion to geo position
#     ras_loc_ANW = np.asarray((5533, 7270))
#     ras_loc_FSE = np.asarray((5544, 7293))
#
#     dempar_file = dir_U76D + 'full_scene/dem_hr/seg.dem_par'
#     par = DEM_Par(dempar_file)
#
#     dem_corner = np.asarray((par['corner_lon'], par['corner_lat']))
#     dem_post = np.asarray((par['post_lon'], par['post_lat']))
#
#     loc_ANW = dem_corner + ras_loc_ANW*dem_post
#     loc_FSE = dem_corner + ras_loc_FSE*dem_post
#
#     print('CR location Report:')
#     print('loc_ANW [deg lon, deg lat]:', loc_ANW)
#     print('loc_FSE [deg lon, deg lat]:', loc_FSE)



def plot_seasonal_network(era5_data_pairwise):
    #plots the results of the ECMWF based network selection
    import matplotlib.cm
    boo = np.zeros(era5_data_pairwise['snow_free'].shape)
    boo[era5_data_pairwise['snow_free']] = 1
    boo[era5_data_pairwise['dry_snow_intra']] = -1
    plt.figure()
    masked_array = np.ma.array(boo, mask=(era5_data_pairwise['mask'] == 0))
    cmap = matplotlib.cm.jet
    cmap.set_bad('black', 1.)
    plt.imshow(masked_array.T, interpolation='nearest', cmap=cmap)



def RS2_SLA27D_ecmwf_itabs():
    # construct and write itab files for RS2_SLA27_D processing based on ECMWF
    era5_data = get_inuvik_RS2_SLA27D_era5()
    era5_data_pairwise = snow.get_era5_pairwise_data(era5_data, dates_all_SLA27D, fractional_day_SLA27D)

    plot_seasonal_network(era5_data_pairwise)
    plt.show()

    itab_snow_free = itab_from_matrix(era5_data_pairwise['snow_free'])
    itab_coherent_snow_init = itab_from_matrix(era5_data_pairwise['dry_snow_intra'])

    # tack on dry snow from end of 2020 (currently ERA5 Land does not include them in queury result
    itab_coherent_snow = itab_merge(itab_coherent_snow_init, (np.asarray(itab_all(4)) + 116).tolist())

    # add connecting edges to link snow periods with snow-free periods (these will be incoherent but will form 'invertable network'
    itab_connectors = [[3, 4],[9, 10],[23, 26],[35, 36],[47, 48],[59, 60],[73, 75],[87, 88],[102, 103],[114, 116]]
    itab_connected = itab_merge(itab_merge(itab_snow_free, itab_coherent_snow),
                                itab_connectors)  # use this one for InSAR processing

    if False:
        write_itab(dir_SLA27D + 'full_scene/itab_snow_free', itab_snow_free)
        write_itab(dir_SLA27D + 'full_scene/itab_coherent_snow', itab_coherent_snow)
        write_itab(dir_SLA27D + 'full_scene/itab_connected', itab_connected)
    x=1

def RS2_SLA27_itabs():
    pass

RS2_U76D = {}
dates_all = dates_all_U76D
RS2_U76D['summer_idx'] = [1,2,3,9,10,11,12,13,14,20,21,22,30,31,32,33,34,42,43,44,45,46,47,48,56,57,58,59,60,61,69,70,71,72,73]
RS2_U76D['w2014_2015_idx'] = [4,5,6,7,8]
RS2_U76D['w2015_2016_idx'] = [15,16,17,18,19]
RS2_U76D['w2016_2017_idx'] = [24,25,26,27,28,29]
RS2_U76D['w2017_2018_idx'] = [35,36,37,38,39,40,41]
RS2_U76D['w2018_2019_idx'] = [49,50,51,52,53]
RS2_U76D['w2019_2020_idx'] = [62,63,64,65,66,67]
RS2_U76D['w2020_2021_idx'] = [74,75,76]
RS2_U76D['winter_idx'] = RS2_U76D['w2014_2015_idx'] + RS2_U76D['w2015_2016_idx'] + RS2_U76D['w2016_2017_idx'] + RS2_U76D['w2017_2018_idx'] + \
                RS2_U76D['w2018_2019_idx'] + RS2_U76D['w2019_2020_idx'] + RS2_U76D['w2020_2021_idx']

def RS2_U76D_itabs():
    stack = stack_U76D
    S = RS2_U76D
    ns = 76
    master_idx = 32

    itab_summer = indexed_itab(itab_all(len(S['summer_idx'])), S['summer_idx'])
    itab_winter = []
    for key in S:
        if 'w2' in key:
            itab_winter += indexed_itab(itab_all(len(S[key])), S[key])
    itab_winter = itab_remove_duplicates(itab_winter)

    itab_master_cm = itab_cm(ns+1, master_idx-1) #zero-indexed
    itab_master_cm = [(pair[0]+1, pair[1]+1) for pair in itab_master_cm] #one-indexed
    #itab_full = itab_remove_duplicates(itab_summer+itab_winter+itab_master_cm)
    itab_mid_quality=read_itab(stack._dir+'itab_mid_quality')
    itab_mst = read_itab(stack._dir + 'itab_mst')
    itab_mid_quality_mst_cm = itab_remove_duplicates(itab_mid_quality+itab_master_cm+itab_mst)
    itab_mid_quality_mst = itab_remove_duplicates(itab_mid_quality+itab_mst)
    itab_missing = itab_find_missing(itab_mid_quality_mst_cm, dates_all, stack._dir + 'base_adjust_all/', '.diff_init.ras', zero_indexed=False)

    if False:
        write_itab(stack_U76D._dir + 'itab_missing', itab_missing, zero_indexed=False)
        write_itab(stack_U76D._dir + 'itab_mid_quality_mst_cm', itab_mid_quality_mst_cm, zero_indexed=False)
        write_itab(stack_U76D._dir + 'itab_mid_quality_mst', itab_mid_quality_mst, zero_indexed=False)
    # pairnames = get_pair_names(itab_full, dates_all)
    # pairnames_txtlist = ''
    # for ii in np.arange(len(pairnames)):
    #     pairnames_txtlist+=(str(itab_full[ii][0]) + ' ' + str(itab_full[ii][1]) + ' ' + pairnames[ii] + '\n')
    x=1


def indexed_itab(itab, idxs):
    #creates itab with same network as one supplied but using indx list instead of assumed 1-N
    itab_new = []
    for pair in itab:
        itab_new += [(idxs[pair[0]], idxs[pair[1]])]
    return itab_new

def get_pair_names(itab, names):
    names = sorted(names)
    pairs = [names[pair[0]-1]+'_'+ names[pair[1]-1] for pair in itab]
    return pairs

def itab_find_missing(itab, datelist, prefix, suffix, zero_indexed=True):
    #find itab entries that are not found using the patter prefix+master_slave+suffix
    itab_new = []
    extra = 0
    if not zero_indexed:
        extra=1
    for pair in itab:
        pattern = prefix+datelist[pair[0]-extra]+'_'+datelist[pair[1]-extra]+suffix
        matches = glob.glob(pattern)
        if len(matches) == 0:
            itab_new+=[pair]
    return itab_new


# def RS2_U76D_ecmwf_itabs():
#     #construct and write itab files for RS2_U76_D processing based on ECMWF
#
#     #first use ECMWF to approximate a network
#     era5_data = get_inuvik_RS2_U76D_era5()
#     era5_data_pairwise = snow.get_era5_pairwise_data(era5_data, dates_all_U76D, fractional_day_U76D)
#
#     itab_snow_free = itab_from_matrix(era5_data_pairwise['snow_free'])
#     itab_coherent_snow_init = itab_from_matrix(era5_data_pairwise['dry_snow_intra'])
#
#     #tack on dry snow from end of 2020 (currently ERA5 Land does not include them in queury result
#     itab_coherent_snow = itab_merge(itab_coherent_snow_init, (np.asarray(itab_all(4)) + 73).tolist())
#
#     #add connecting edges to link snow periods with snow-free periods (these will be incoherent but will form 'invertable network'
#     itab_connectors = [[2, 3],[12, 13],[20, 22], [33, 34], [46, 47], [59, 60], [72, 73]]
#     itab_connected = itab_merge(itab_merge(itab_snow_free, itab_coherent_snow), itab_connectors) #use this one for InSAR processing
#
#     if False:
#         write_itab(dir_U76D+'full_scene/itab_snow_free', itab_snow_free)
#         write_itab(dir_U76D + 'full_scene/itab_coherent_snow', itab_coherent_snow)
#         write_itab(dir_U76D + 'full_scene/itab_connected', itab_connected)
#     x=1

def coherency_mtx():
    #finish this with mask parameter and move to stack_processing module as part of stack class
    names = stack_U76D.getAllPairFilenames()

def mk_non_motion_map(stack=stack_U76D, debug=False):
    coh_thresh = 0.8
    mostly_coherent_lf_list_U76D = ['20140824_20140917','20141128_20141222','20141222_20150115','20150726_20150819','20150819_20150912',
                               '20150912_20151006','20151123_20151217','20151217_20160110','20160110_20160203','20160110_20160227',
                               '20160203_20160227','20160813_20160906','20161211_20170104','20170808_20170901','20170901_20170925',
                               '20171206_20171230','20171230_20180123','20180803_20180827','20180827_20180920','20180920_20181014',
                               '20181225_20190118','20190729_20190822','20190822_20190915','20200206_20200301','20200816_20200909',
                               '20200909_20201003','20201214_20210107']
    mostly_coherent_lf_list_SLA27D = [] #put them here one you have them
    if stack is stack_U76D:
        mostly_coherent_lf_list = mostly_coherent_lf_list_U76D
        dir = stack_U76D._dir
        lookstr = '_12_18'
    elif stack is stack_SLA27D:
        mostly_coherent_lf_list = mostly_coherent_lf_list_SLA27D
        dir =  stack_SLA27D._dir
        lookstr = '_11_44'

    diff_dir = dir + 'base_adjust_all/'
    ifg_ext = '.diff'

    #just need any ras mask file with valid colormap..
    dummy_mask_filename = '/local-scratch/common/quadzilla_data_mount/jaysone/umiujaq/full_scene/watermask_1_6.ras'
    _, cmap_wm = read_ras(dummy_mask_filename)

    parname = diff_dir + mostly_coherent_lf_list[0] + '.diff_par'
    par = DIFF_Par(parname)
    coh_cum = np.zeros(par.dim, dtype=complex)
    mag_cum = np.zeros(par.dim, dtype=float)
    for ii, pair_name in enumerate(mostly_coherent_lf_list):
        ifg_name = diff_dir + pair_name + ifg_ext
        atm_name = diff_dir + 'atm_screens/' + pair_name + '.atm'
        ifg0 = readBin(ifg_name, par.dim, '>c8')
        atm_screen = readBin(atm_name, par.dim, '>f4')
        ifg = ifg0*np.exp(-1j*atm_screen)
        ifg_mean = np.mean(ifg)
        ifg_demean = ifg/ifg_mean*np.abs(ifg_mean)
        if debug:
            plt.figure()
            plt.hist(np.angle(ifg_demean.flatten()[np.abs(ifg_demean.flatten()) != 0]), bins=180)
            plt.title(mostly_coherent_lf_list[ii])
            plt.show()
        coh_cum += ifg_demean
        mag_cum += np.abs(ifg_demean)
    coh_temporal = np.abs(coh_cum)/mag_cum
    coh_temporal[mag_cum == 0] = 0
    coh_mask = (coh_temporal > coh_thresh)
    coh_mask = ndimage.binary_opening(coh_mask, structure=np.ones((2, 2)))
    plt.figure()
    plt.hist(coh_temporal.flatten(), bins=200)
    plt.figure()
    plt.subplot(1,2,1)
    plt.title('Temporal coherence')
    plt.imshow(coh_temporal.T)
    plt.subplot(1, 2, 2)
    plt.title('Temporal coherence mask (>0.8)')
    plt.imshow(coh_mask.T)

    no_motion_mask = coh_mask
    write_ras(dir + 'no_motion_mask'+lookstr+'.ras', no_motion_mask.T*255,cmap=cmap_wm)
    plt.show()
    x=1

def read_itab(fname):
    itab=[]
    with open(fname, 'r') as f:
        for line in f.readlines():
            vals = [int(val) for val in line.split()]
            itab += [vals[0:2]]
    return itab



def get_pair_names(stack, itabname):
    itab = read_itab(stack._dir+ itabname)
    rslc_filenames = stack.rslc_list()
    pair_names = [get_pair_name_root(rslc_filenames[pair[0]-1], rslc_filenames[pair[1]-1]) for pair in itab]
    return pair_names

def coh_clustering_sandbox():
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    stack = stack_U76D_small
    itab_name = 'itab_mid_quality'
    npts_sample = 10000
    n_clusters = 8
    #n_az = 10
    dim = stack.dim()
    pair_names = get_pair_names(stack, itab_name)
    n_pairs = len(pair_names)
    coh = np.zeros((n_pairs, dim[0], dim[1]), dtype=np.float32)
    for ii in np.arange(n_pairs):
        coh[ii,:,:] = readBin(stack._dir+'diff/'+ pair_names[ii]+'.cc', stack.dim(), 'float32')

    shape_orig = coh.shape

    #coh = coh[:, :, 0:n_az]
    #dim = (dim[0], n_az)

    #coh_sample = np.zeros((n_pairs, npts_sample), dtype=np.float32)
    idx_sample = np.linspace(0, dim[0]*dim[1]-1, npts_sample).astype(np.int)
    coh = coh.reshape(n_pairs, dim[0]*dim[1])
    coh_sample = coh[:, idx_sample]

    n_components=3
    pca = PCA(n_components=n_components)
    coh_dr = pca.fit_transform(coh.T).T
    coh_dr = coh_dr.reshape(n_components,dim[0],dim[1])
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    coh = coh.reshape(shape_orig)

    plt.figure()
    plt.hist2d(coh_dr[0,:,:].flatten(), coh_dr[1,:,:].flatten(), bins=200)

    plt.figure()
    plt.plot(pca.explained_variance_ratio_)
    plt.ylim((0, None))
    plt.ylabel('Explained variance ratio')
    plt.xlabel('PCA component #')
    plt.figure()
    for ii in np.arange(n_components):
        plt.subplot(1, n_components, ii+1)
        plt.imshow(np.squeeze(coh_dr[ii, :, :]).T)
    plt.show()


    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coh_sample.T)
    classes = kmeans.predict(coh.T)
    classes = classes.reshape(dim[0], dim[1])
    plt.imshow(classes.T, cmap='tab10')
    plt.figure()
    plt.hist(classes.flatten(),bins=10)
    plt.show()

    x=1

def expand_from_sparse(img_sparse, pt, dim, fill=np.nan):
    img = np.empty(dim)
    img.fill(fill)
    img[pt[:, 0], pt[:, 1]] = img_sparse
    return img


def tcr_correlation_sandbox():
    stack = stack_U76D_big
    idx_lists = RS2_U76D
    locs = cr_loc_big
    itab_name = 'itab_mid_quality'
    pt_name = stack._dir + 'hds/pt'
    #first TCR set appears 20180827 which is itab idx #46
    #idx_start = 46
    #tcrs = ['CR_A1', 'CR_F1']
    #loc = cr_loc_small['CR_A1']
    #second TCR set appears 20190729 which is itab idx #58
    idx_start = 58
    tcrs = ['CR_A1', 'CR_A2', 'CR_F1', 'CR_F2', 'CR_F3', 'CR_F4']

    n_tcrs = len(tcrs)
    dim = stack.dim()
    pair_names_init = get_pair_names(stack, itab_name)
    n_pairs_init = len(pair_names_init)
    #pt = read_pt(pt_name)
    #npts = pt.shape[0]

    #load itab and then only keeps pairs that see the installed TCR (based on idx_start)
    itab_init = read_itab(stack._dir + itab_name)
    #itab = []
    pair_names = []
    for ii in np.arange(n_pairs_init):
        pair = itab_init[ii]
        if (pair[0] >= idx_start) and (pair[1] >= idx_start): #all one-indexed
            #itab += [pair]
            if pair[0] in idx_lists['summer_idx'] and pair[1] in idx_lists['summer_idx']:
                pair_names += [pair_names_init[ii]]
    n_pairs = len(pair_names)
    print(n_pairs, ' valid pairs out of ', n_pairs_init, ' possible pairs.')

    # #find the pt indices of the TCRs..
    # cr_pt_idx = {}
    cr_pt_idx_wrapped = {}
    # for ii in np.arange(npts):
    #     for tcr in tcrs:
    #         if (pt[ii,0] == locs[tcr][0]) and (pt[ii,1] == locs[tcr][1]):
    #             #found it!
    #             cr_pt_idx[tcr] = ii
    #             cr_pt_idx_wrapped[tcr] = ii

    for tcr in tcrs:
        cr_pt_idx_wrapped[tcr] = locs[tcr][0] * dim[1] + locs[tcr][1]


    coh_sum = np.zeros((n_tcrs, dim[0]*dim[1]), dtype=np.complex64)
    abs_sum = np.zeros((n_tcrs, dim[0]*dim[1]), dtype=np.float32)
    for ii in np.arange(n_pairs):
        print('Processing pair ', ii+1, ' of ', n_pairs)
        print('reading...')
        phi_wrapped_ii = (readBin(stack._dir + 'diff/' + pair_names[ii] + '.diff.natm.hds', dim, 'complex64')).flatten()
        print('updating running temporal coherence...')
        for jj in np.arange(n_tcrs):
            phi_wrapped_ref_ii_jj = phi_wrapped_ii[cr_pt_idx_wrapped[tcrs[jj]]]
            phi_wrapped_ii_demod_jj = phi_wrapped_ii*np.conj(phi_wrapped_ref_ii_jj)
            coh_sum[jj, :] += phi_wrapped_ii_demod_jj
            abs_sum[jj, :] += np.abs(phi_wrapped_ii_demod_jj)

    cohs = np.abs(coh_sum)/abs_sum
    print('Max cohs: ', np.max(cohs, axis=1))
    for ii in np.arange(n_tcrs):
        print('\nTCR Differential Coherences:')
        for jj in np.arange(n_tcrs):
            print(tcrs[ii], tcrs[jj], cohs[ii,cr_pt_idx_wrapped[tcrs[jj]]])

    print('rendering results...')
    for ii in np.arange(n_tcrs):
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(np.reshape(cohs[ii,:], dim).T[:, ::-1], vmin=0, vmax=1., interpolation='bilinear')
        #plt.title('coh for '+ tcrs[ii])
        plt.subplot(1,2,2)
        plt.hist(cohs[ii,:].flatten(), bins=100)
        plt.title('coh for ' + tcrs[ii])

    plt.show()

def phase_ica_sandbox():
    from sklearn.decomposition import FastICA
    phase_centering = 'mean'
    stack = stack_U76D_big
    idx_lists = RS2_U76D
    itab_name = 'itab_mid_quality'
    pt_name = stack._dir + 'hds/pt'
    selection = 'all'
    n_components=4
    npts_sample = 1000000 #number of spatial points to uniformly subsample and base the analysis on

    dim = stack.dim()
    pair_names_init = get_pair_names(stack, itab_name)
    n_pairs_init = len(pair_names_init)
    pt = read_pt(pt_name)
    npts = pt.shape[0]
    npts_sample = np.min((npts_sample, npts))

    _, active_layer_idd = read_active_layer_template(stack._dir + '../active_layer.txt')

    itab_init = read_itab(stack._dir+ itab_name)

    itab = []
    pair_names=[]
    #enforce summer only
    for ii in np.arange(n_pairs_init):
        pair = itab_init[ii]
        if (selection == 'winter') and (pair[0] not in idx_lists['summer_idx']) and (pair[1] not in idx_lists['summer_idx']):
            itab += [pair]
            pair_names += [pair_names_init[ii]]
        if (selection == 'summer') and (pair[0] in idx_lists['summer_idx']) and (pair[1] in idx_lists['summer_idx']):
            itab += [pair]
            pair_names += [pair_names_init[ii]]
        if (selection != 'winter') and (selection != 'summer'):
            itab += [pair]
            pair_names += [pair_names_init[ii]]

    n_pairs = len(pair_names)

    itab = [[pair[0]-1, pair[1]-1] for pair in itab ] #convert from one-indexed to zero-indexed
    #itab = itab[0:6]

    #matrix to go from pairwise to per scene...
    #M, idx_list = design_matrix(itab)
    M = M_matrix(itab)
    #idx_list = itab_idx_list(itab)
    n_scenes = M.shape[1]
    #read all explanatory variables (baselines, acq times, active_layer value, swe)
    #bperps
    bperp = stack.bperps(dir='base_adjust_all/')#[idx_list]
    bperp=(bperp - bperp[0])[1:]

    #delta time in days
    datetimes = stack.datetimes()
    dt = np.asarray([int((datetime-datetimes[0]).total_seconds()/86400. + 0.5) for datetime in datetimes])
    dt = dt#[idx_list]
    dt = (dt-dt[0])[1:]

    #active layer template difference
    _, active_layer_idd = read_active_layer_template(stack._dir + '../active_layer.txt')
    active_layer_idd = active_layer_idd#[idx_list]
    active_layer_idd = (active_layer_idd - active_layer_idd[0])[1:]

    #dt2 = np.matmul(M_inv, np.matmul(M, dt))
    #bperp2 = np.matmul(M_inv, np.matmul(M, bperp))
    #active_layer_idd2 = np.matmul(M_inv, np.matmul(M, active_layer_idd))

    inversion_order = 1 #controls SVD behavior (0: invert positions, 1: invert velocities, 2: invert accelerations)
    G = G_matrix(M, inversion_order)
    dt2 = np.matmul(G, np.matmul(M, dt))
    bperp2 = np.matmul(G, np.matmul(M, bperp))
    active_layer_idd2 = np.matmul(G, np.matmul(M, active_layer_idd))


    dt3 = np.matmul(M, dt)
    bperp3 = np.matmul(M, bperp)
    active_layer_idd3 = np.matmul(M, active_layer_idd)

    idx_sample = np.linspace(0, npts - 1, npts_sample).astype(np.int)

    #print(pair_names[5], pair_names[70], pair_names[82])
    if True:
        phi_sample = np.zeros((n_pairs, npts_sample), dtype=np.float32)
        for ii in np.arange(n_pairs):
            fname = stack._dir+'hds/select/'+ pair_names[ii]+'.diff.natm.hds.unw'
            print('Sampling igram ', ii+1, ' of ', n_pairs, ' : ', fname)
            phi_ii = np.squeeze(readBin(fname, (npts,1), 'float32'))
            if phase_centering == 'mode':
                hist, edges = np.histogram(phi_ii, bins=int(npts_sample / 10000))
                idx_max = np.argmax(hist)
                mode = 0.5 * (edges[idx_max] + edges[idx_max + 1])
                phi_ii -= mode
            elif phase_centering == 'mean':
                phi_ii -= np.mean(phi_ii)
            phi_sample[ii,:] = phi_ii[idx_sample]

        print('Performing Fast ICA...')
        ica = FastICA(n_components=n_components, random_state=0).fit(phi_sample.T)

        print('Computing component maps...')
        mixing_pairwise = ica.mixing_
        ica_components = np.zeros((n_components, npts), dtype=np.float32)

        for ii in np.arange(n_pairs):
            print('Updating ICA components with igram ', ii + 1, ' of ', n_pairs)
            phi_ii = np.squeeze(readBin(stack._dir+'hds/select/'+ pair_names[ii]+'.diff.natm.hds.unw', (npts,1), 'float32'))
            ica_components += np.outer(ica.components_[:,ii], phi_ii)


        print('Saving results.')
        np.save(stack._dir + 'analysis/ica.npy', ica_components)
        np.save(stack._dir + 'analysis/ica.mixing.npy', mixing_pairwise)


    else:
        print('Loading already computed results.')
        ica_components = np.load(stack._dir + 'analysis/ica.npy')
        mixing_pairwise = np.load(stack._dir + 'analysis/ica.mixing.npy')

    mixing = np.matmul(G, mixing_pairwise)


    print('Plotting results...')
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(dt2)
    plt.title('Acq. day v2')
    plt.subplot(3,1,2)
    plt.plot(bperp2)
    plt.title('Bperp v2')
    plt.subplot(3,1,3)
    plt.plot(active_layer_idd2)
    plt.title('Active layer IDD v2')

    # plt.figure()
    # plt.subplot(4,3,1)
    # plt.plot(dt)
    # plt.title('Acq. day')
    # plt.subplot(4,3,4)
    # plt.plot(bperp)
    # plt.title('Bperp')
    # plt.subplot(4,3,7)
    # plt.plot(active_layer_idd)
    # plt.title('Active layer IDD')
    #
    # plt.subplot(4,3,2)
    # plt.plot(dt2)
    # plt.title('Acq. day v2')
    # plt.subplot(4,3,5)
    # plt.plot(bperp2)
    # plt.title('Bperp v2')
    # plt.subplot(4,3,8)
    # plt.plot(active_layer_idd2)
    # plt.title('Active layer IDD v2')
    #
    # plt.subplot(4,3,3)
    # plt.plot(dt3)
    # plt.title('Acq. day v3')
    # plt.subplot(4,3,6)
    # plt.plot(bperp3)
    # plt.title('Bperp v3')
    # plt.subplot(4,3,9)
    # plt.plot(active_layer_idd3)
    # plt.title('Active layer IDD v3')

    # plt.figure()
    # plt.title('Pairwise mixing')
    # for ii in np.arange(n_components):
    #     plt.subplot(n_components,1,ii+1)
    #     plt.plot(mixing_pairwise[:,ii])


    plt.figure()
    plt.title('Pairwise mixing')
    for ii in np.arange(n_components):
        plt.subplot(n_components,1,ii+1)
        plt.plot(mixing_pairwise[:,ii])

    plt.figure()
    plt.title('Per-scene mixing')
    for ii in np.arange(n_components):
        plt.subplot(n_components,1,ii+1)
        plt.plot(mixing[:,ii])

    component_modes = np.zeros(n_components)
    for ii in np.arange(n_components):
        ica_component_ii = np.copy(ica_components[ii, :])
        hist, edges = np.histogram(ica_component_ii, bins=int(npts_sample/10000))
        idx_max = np.argmax(hist)
        component_modes[ii] = 0.5*(edges[idx_max] + edges[idx_max+1])

    for ii in np.arange(n_components):
        render_edge_percentile = 0.1
        #plt.subplot(1, n_components, ii+1)
        plt.figure()
        #render_dist = np.max((component_modes[ii] - np.min(ica_components[ii, :]), np.max(ica_components[ii, :]) - component_modes[ii]))
        render_dist = np.max((component_modes[ii] - np.percentile(ica_components[ii, :], render_edge_percentile), np.percentile(ica_components[ii, :], (100-render_edge_percentile)) - component_modes[ii]))
        plt.imshow(expand_from_sparse(ica_components[ii, :], pt, dim).T[:, ::-1], cmap='RdBu', vmin = component_modes[ii]-render_dist, vmax=component_modes[ii]+render_dist)
        plt.title('Mixing Component ' + str(ii+1) + ' of '+ str(n_components))

    plt.figure()
    for ii in np.arange(n_components):
        plt.subplot(n_components,1,ii+1)
        plt.hist(ica_components[ii, :].flatten()-component_modes[ii], bins=int(npts_sample/10000))

    n_vars=3
    explanatory_variables = np.zeros((n_scenes, n_vars))
    explanatory_variables[:, 0] = dt2
    explanatory_variables[:, 1] = bperp2
    explanatory_variables[:, 2] = active_layer_idd2

    corr_mat = np.zeros((n_vars, n_components))
    for ii in np.arange(n_vars):
        for jj in np.arange(n_components):
            x = explanatory_variables[:, ii]
            x = x - np.mean(x)
            x/=np.var(x)**0.5
            y = mixing[:,jj]
            y = y - np.mean(y)
            y/=np.var(y)**0.5
            corr_mat[ii, jj] = np.sum(x*y)/n_pairs
    print(corr_mat)

    plt.show()

    x=1




