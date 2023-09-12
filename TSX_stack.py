from sarlab.gammax import SLC_stack, SLC_Par, getOptimalLooks, MLI_Par, readBin, writeBin
import matplotlib.pyplot as plt
import os
import re
import tarfile
import glob
import subprocess
import numpy as np

from cr_phase_to_deformation import get_itab_diffs


working_dir = '/local-scratch/users/aplourde/HDS/projects/southern_ITH/TSX_SM39_D/'
#working_dir = '/local-scratch/users/aplourde/TSX/'
#sub_dir = 'full_scene/'; master = '20210903_HH'
#sub_dir = 'full_scene_crop/'; master = '20210903'
sub_dir = 'crop_sites/'; master = '20210903'
#sub_dir = 'test/'; master = '20220411_HH'
#sub_dir = 'test_crop_sites/'; master = '20220411_HH'
water_mask = None

ingest_cfg = {'password': 'S1m0nF7@SER', 'polarizations': 'HH'}
stack = SLC_stack(dirname=working_dir + sub_dir, name='tsx_southernITH', reg_mask=None, master=master, looks_hr=(2, 2), looks_lr=(18, 16), multiprocess=False, rdcdem_refine=True, skipmode='None', ingest_cfg=ingest_cfg)

itab = np.loadtxt(stack._dir + 'itab_lf_snow', dtype='int', skiprows=0, usecols=(0, 1))
itab = itab-1
#itab = np.asarray([[0,1]])

def preproc_tsx():

    zipfiles = glob.glob(working_dir + sub_dir + 'raw/*.tar.gz')
    paths = working_dir + sub_dir + 'paths.txt'
    slc_dir = working_dir + sub_dir + 'slc/'

    with open(working_dir + sub_dir + 'paths.txt', 'w') as f:
        for zipfile in zipfiles:
            if os.path.exists(zipfile.split('.tar.gz')[0]):
                # file already unzipped, add to paths.txt
                data = glob.glob(zipfile.split('.')[0] + '/TSX-1.SAR.L1B/*/')
                f.write(data[0][:-1] + '\n')
            else:
                # unzip the product
                tar = tarfile.open(zipfile, "r:gz")
                tar.extractall(working_dir + sub_dir + 'raw/')
                tar.close()

                data = glob.glob(zipfile.split('.')[0] + '/TSX-1.SAR.L1B/*/')

                f.write(data[0][:-1] + '\n')
        f.close()

    os.chdir(working_dir + sub_dir)

    logfile = 'preproc.log'

    if not os.path.exists(logfile):
        with open(logfile, 'w') as f:
            f.close()

    print('TX_SLC_preproc', paths, slc_dir, logfile)
    preproc = ['TX_SLC_preproc', paths, slc_dir, logfile]

    subprocess.run(preproc, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def full_processing(looks='fr'):

    #preproc_tsx()
    #stack.ingest()

    #stack.register()

    stack.mk_mli_all(looks=looks)

    #stack.rdc_dem(looks=looks)
    #stack.mk_diff(looks=looks, itab=itab)  # itab='all', itab_year_to_year)


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

    look = 'lr'

    #looks, aspects = getOptimalLooks(stack.master_slc_par)
    #for look, aspect in zip(looks, aspects):
    #    print(look, aspect)

    """ Main Tool Chain """
    full_processing(looks=look)

    """ Subsetting """
    #crop_sites = [1138, 12700, 1235, 2250]
    #full_scene_crop = [0, 6000, 6000, 12000]
    #stack.subset(full_scene_crop, working_dir + 'full_scene_crop/')

    """ Utils """
    #mk_coherance_maps(looks='hr')


