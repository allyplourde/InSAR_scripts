from sarlab.gammax import *
import pandas as pd
import numpy as np
import os
import glob
import shutil

#working_dir = '/local-scratch/users/aplourde/HDS/projects/southern_ITH/RS2_U76_D/full_scene/'
#working_dir = '/local-scratch/users/aplourde/HDS/projects/southern_ITH/RS2_U76_D/new_crop_sites/'
#working_dir = '/local-scratch/users/aplourde/HDS/projects/southern_ITH/TSX_SM39_D/crop_sites/'
working_dir = '/local-scratch/users/aplourde/HDS/projects/southern_ITH/TSX_SM39_D/full_scene_crop/'
#working_dir = '/sarlab/orca/akplourd/projects/southern_ITH/RS2_U76_D/full_scene/'

#master_par = 'rslc/20170808.rslc.par'
master_par = 'rslc/20210903.rslc.par'

def small_baselines(dir):
    os.chdir(dir)
    SLC_tab = 'RSLC_tab'
    out = 'lf' + '.bperp'
    itab = 'itab_lf'
    #out = 'sb_D90' + '.bperp'
    #itab = 'itab_sb_D90'
    itab_type = 1   #0: lf, 1:all
    pltflg = 1      #0: None, 1: png, 2: screen
    bperp_min = 0
    bperp_max = 5000 #m
    delta_T_min = 10
    delta_T_max = 13
    #delta_T_min = 300
    #delta_T_max = 400  #days

    cmd = 'base_calc', SLC_tab, master_par, out, itab, itab_type, pltflg, bperp_min, bperp_max, delta_T_min, delta_T_max
    #print('base_calc', SLC_tab, master_par, out, itab, itab_type, pltflg, bperp_min, bperp_max, delta_T_min, delta_T_max)
    cmd = str(cmd).replace(',','')
    os.system(cmd)


def itab_all(dir):
    os.chdir(dir)
    SLC_tab = 'SLC_tab'
    master_par = 'rslc/20181014.rslc.par'
    out = 'all.bperp'
    itab = 'itab_all'
    itab_type = 1  # 0: lf, 1:all
    pltflg = 0  # 0: None, 1: png, 2: screen
    bperp_min = 0
    bperp_max = 400000  # m
    delta_T_min = 0
    delta_T_max = 365000  # days

    cmd = 'base_calc', SLC_tab, master_par, out, itab, itab_type, pltflg, bperp_min, bperp_max, delta_T_min, delta_T_max
    # print('base_calc', SLC_tab, master_par, out, itab, itab_type, pltflg, bperp_min, bperp_max, delta_T_min, delta_T_max)
    cmd = str(cmd).replace(',', '')
    os.system(cmd)


def itab_by_date(dir, out, start_date = '20170101', end_date = None):
    SLC_tab = np.loadtxt(dir + 'SLC_tab', dtype=np.str)  # should be in order

    slcs = [x[0] for x in SLC_tab]
    dates = [s.split('/')[1][:-4] for s in slcs]
    pd_dates = pd.to_datetime(dates)

    new_dates = None
    if start_date is not None:
        new_dates = pd_dates>pd.to_datetime(start_date)

    if end_date is not None:
        if new_dates is not None:
            new_dates = np.multiply(new_dates, pd_dates<pd.to_datetime(end_date))
        else:
            new_dates = pd_dates<pd.to_datetime(end_date)

    SLC_tab_new = SLC_tab[new_dates]
    np.savetxt(dir + 'SLC_tab' + '_' + out, SLC_tab_new, fmt='%s')

    small_baselines(dir, custom = '_' + out)


def itab_lf(dir, start_date = None):
    RSLC_tab = np.loadtxt(dir + 'RSLC_tab', dtype='str')

    count = 1
    rslc = 1
    itab_lf = []
    for i in range(len(RSLC_tab)-1):
        if start_date is not None:
            date = pd.to_datetime(os.path.basename(RSLC_tab[i][0].split('.')[0]))
            if date >= start_date:
                itab_lf.append([rslc, rslc+1, count, 1])
                count += 1
        else:
            itab_lf.append([rslc, rslc+1, count, 1])
            count += 1
        rslc += 1
    itab_lf = np.asarray(itab_lf)

    np.savetxt(working_dir + 'itab_lf', itab_lf, fmt='%s')


def small_baselines_custom(dir):
    SLC_tab = np.loadtxt(dir + 'SLC_tab', dtype=np.str) #should be in order

    slcs = [x[0] for x in SLC_tab]
    dates = [s.split('/')[1][:-4] for s in slcs]
    pd_dates = pd.to_datetime(dates)
    print(pd_dates)

    #temporal
    #create itab for all slcs with temporal baselines <= 1 year
    #itab format
    #master slc     slave slc   ifg#    1?
    count = 0
    tb = []
    for i in range(len(pd_dates)):
        for j in range(len(pd_dates)-i):
            temporal_baseline = pd_dates[j+i] - pd_dates[i]
            if temporal_baseline.days < 365 and temporal_baseline.days > 0:
                count += 1
                tb.append((i+1, j+i+1, count, 1))
    print(tb)

    #geometrical
    #TODO


def itab_cm(dir, itab_name, master_date = '20180827'):
    RSLC_tab = np.loadtxt(dir + 'RSLC_tab', dtype='str')

    count = 1
    itab_cm = []

    for i, rslc in enumerate(RSLC_tab):
        if master_date in rslc[0]:
            cm = i+1
            break

    for i in range(len(RSLC_tab)):
        if i+1 < cm:
            #pass
            itab_cm.append([i+1, cm, count, 1])
        elif i+1 > cm:
            itab_cm.append([cm, i+1, count, 1])
        count += 1

    itab_cm = np.asarray(itab_cm)

    np.savetxt(working_dir + itab_name, itab_cm, fmt='%s')


def itab_seasonal_filter(itab_dir, itab, filtered_name, months):

    itab_to_be_filtered = np.loadtxt(itab_dir + itab, dtype='int')
    RSLC_tab = np.loadtxt(itab_dir + 'RSLC_tab', dtype='str')
    new_itab = []

    rslcs = [x[0] for x in RSLC_tab]
    dates = [r.split('/')[1][:-4] for r in rslcs]
    pd_dates = pd.to_datetime(dates)

    count = 1

    for i in itab_to_be_filtered:
        mslc = os.path.basename(RSLC_tab[i[0]-1][0]).split('.')[0]
        sslc = os.path.basename(RSLC_tab[i[1]-1][0]).split('.')[0]

        if 'TSX' in itab_dir:
            mslc = mslc.split('_HH')[0]
            sslc = sslc.split('_HH')[0]

        m_date = pd.to_datetime(mslc)
        s_date = pd.to_datetime(sslc)

        if m_date.month in months and s_date.month in months:
            new_itab.append([[i[0]][0], [i[1]][0], count, 1])

    np.savetxt(itab_dir + filtered_name, new_itab, fmt='%s')


def combine_itabs(itab_dir, itab1, itab2, new_name):

    new_itab = []
    RSLC_tab = np.loadtxt(itab_dir + 'RSLC_tab', dtype='str')

    itab1 = np.loadtxt(itab_dir + itab1, dtype='int')
    itab2 = np.loadtxt(itab_dir + itab2, dtype='int')

    # combine the itabs
    new_itab = np.concatenate((itab1, itab2))
    new_itab[:, 2] = 1

    # remove duplicates
    new_itab = np.unique(new_itab, axis=0)

    # reorder
    new_itab = new_itab[new_itab[:, 0].argsort()]
    new_itab[:, 2] = np.arange(1, new_itab.shape[0] + 1)

    np.savetxt(working_dir + new_name, new_itab, fmt='%s')


def itab_coherance_filter(itab_dir, itab, itab_result, par, min_coh = 0.3, diff_dir='base_adjust_all/', ext='.cc', mask=None):

    itab_to_be_filtered = np.loadtxt(working_dir + itab, dtype='int')
    RSLC_tab = np.loadtxt(itab_dir + 'RSLC_tab', dtype='str')
    new_itab = []

    rslcs = [x[0] for x in RSLC_tab]
    roots = [os.path.basename(r).split('.')[0] for r in rslcs]
    print(roots)

    if mask is not None:
        mask = read_ras(mask)[0].T
    else:
        mask = np.ones(par.dim)


    cc_dir = working_dir + diff_dir
    for i in itab_to_be_filtered:
        mslc = roots[i[0]-1]
        sslc = roots[i[1]-1]

        try:
            cc_file = os.path.join(cc_dir, mslc + '_' + sslc + ext)
            #diff_file = os.path.join(cc_dir, mslc + '_' + sslc + '.diff_init.ras')
            diff_file = os.path.join(cc_dir, mslc + '_' + sslc + '.diff.ras')

            cc = readBin(cc_file, par.dim, 'float32')
            phi = read_ras(diff_file)[0].T
        except:
            print(cc_file)
            print(f"Warning: {mslc}_{sslc} not found, skipping...")
            continue
        print(f"{mslc}_{sslc}")
        mean_cc = np.nanmean(cc[mask > 0])


        print(mean_cc)
        if mean_cc > min_coh:
            print(f"putting {mslc}_{sslc} with mean coherence {mean_cc} into itab.")
            new_itab.append(i)
            plt.figure(figsize=(12, 12))
            plt.subplot(121)
            plt.title(f"{mslc}_{sslc}\n{mean_cc}")
            plt.imshow(cc.T, cmap='Greys_r', vmin=0, vmax=1)
            plt.subplot(122)
            plt.imshow(phi.T)
            #plt.show()
        else:
            #shutil.copy(working_dir + 'base_adjust_all/indv_corr_backup/' + mslc + '_' + sslc + '.diff',
            #            working_dir + 'base_adjust_all/' + mslc + '_' + sslc + '.diff_init',)
            #shutil.copy(working_dir + 'base_adjust_all/indv_corr_backup/' + mslc + '_' + sslc + '.diff.ras',
            #            working_dir + 'base_adjust_all/' + mslc + '_' + sslc + '.diff_init.ras',)
            pass

    new_itab = np.asarray(new_itab)
    new_itab = new_itab[new_itab[:, 0].argsort()]
    new_itab[:, 2] = np.arange(1, new_itab.shape[0] + 1)

    np.savetxt(working_dir + itab_result, new_itab, fmt='%s')


def remove_scenes_from_itab(itab_dir, itab, date, itab_result):
    itab_to_be_filtered = np.loadtxt(working_dir + itab, dtype='int')
    RSLC_tab = np.loadtxt(itab_dir + 'RSLC_tab', dtype='str')
    new_itab = []

    rslcs = [x[0] for x in RSLC_tab]
    roots = [os.path.basename(r).split('.')[0] for r in rslcs]
    print(roots)

    for i in itab_to_be_filtered:
        mslc = roots[i[0]-1]
        sslc = roots[i[1]-1]

        if mslc == date or sslc == date:
            continue
        else:
            new_itab.append(i)

    new_itab = np.asarray(new_itab)
    new_itab = new_itab[new_itab[:, 0].argsort()]
    new_itab[:, 2] = np.arange(1, new_itab.shape[0] + 1)

    np.savetxt(working_dir + itab_result, new_itab, fmt='%s')


if __name__ == "__main__":

    #small_baselines(working_dir)
    #itab_by_date(working_dir, '2017-2019')
    #itab_all(working_dir)
    #itab_lf(working_dir)
    #itab_cm(working_dir, itab_name = 'itab_cm_20190726', master_date='20190726')
    #itab_seasonal_filter(working_dir, "itab_lf", "itab_lf_snow", [10, 11, 12, 1, 2, 3])
    #itab_seasonal_filter(working_dir, "itab_lf", "itab_lf_snowfree", [6, 7, 8, 9])

    #combine_itabs(working_dir, 'itab_sb_D90', 'itab_sb_Y1', 'itab_Y1_D90')

    look_str = '2_2'
    master_par = SLC_Par(working_dir + 'rmli_' + look_str + '/rmli_' + look_str + '.ave.par')
    #water_mask = "/local-scratch/users/aplourde/HDS/projects/southern_ITH/RS2_U76_D/watermask_12_18_eroded.ras"
    water_mask = working_dir + 'dem_' + look_str + '/watermask_' + look_str + '.ras'
    #RS2 min_coh_low = 0.03;    min_coh_mid = 0.3;    min_coh_high = 0.5
    min_coh_low = 0.2; min_coh_mid = 0.3; min_coh_high = 0.4
    itab_coherance_filter(working_dir, "itab_lf_snow", "itab_lf_snow_mq", master_par, diff_dir='diff_' + look_str + '/', min_coh=min_coh_mid, mask=None, ext='.diff.adf.cc')

    #remove_scenes_from_itab(working_dir, "itab_Y1_D90_mid_quality_summer", "20180312", "itab_Y1_D90_mid_quality_summer_bdrm")
    #remove_scenes_from_itab(working_dir, "itab_Y1_D90_mid_quality_summer_bdrm", "20190729", "itab_Y1_D90_mid_quality_summer_bdrm")