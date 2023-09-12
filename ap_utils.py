from sarlab.gammax import readBin, read_ras, writeBin, run
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage as ndimage
import os

import matplotlib.cm as cm
import matplotlib.colors as col

from PIL import Image

from sarlab.gammax import *
import re


#working_dir = '/local-scratch/users/aplourde/HDS/projects/southern_ITH/RS2_U76_D/'
working_dir = '/local-scratch/users/aplourde/HDS/projects/southern_ITH/TSX_SM39_D/'
#sub_dir = 'post_cr_installation/'
#sub_dir = 'full_scene/'
sub_dir = 'full_scene_crop/'
#working_dir = '/local-scratch/users/aplourde/DSP_project/small/'
#sub_dir = ''

def mk_base_plot():

    if not os.path.exists(working_dir + sub_dir + '*SLC_tab'):
        os.chdir(working_dir + sub_dir)
        run('mk_tab', 'rslc', 'rslc', 'rslc.par', 'RSLC_tab')

    run('base_calc', '*SLC_tab', )



def write_watermask(im, filename, width):

    writeBin(filename, im)
    run('rascc_bw', filename, None, width, None, None, None, None, None, 0, 1, None, None, None, filename + '.ras')
    #run('ras2ras', filename + '_tmp.ras', filename + '.ras', 'gray.cm')
    #os.remove(filename + '_tmp.ras')


def erode_watermask(filename, out_file, iters=25):

    mask = read_ras(filename)[0]

    struct = ndimage.generate_binary_structure(2,2)
    mask_eroded = ndimage.morphology.binary_dilation(mask, structure=struct, iterations=iters)


    fig = plt.subplot(121)
    plt.imshow(mask, cmap=cm.Greys_r)
    fig = plt.subplot(122)
    plt.imshow(mask_eroded, cmap=cm.Greys_r)
    plt.show()

    mask_eroded = mask_eroded.astype('float')
    out = mask_eroded.T

    write_watermask(out, out_file, out.shape[0])


def downsample_watermask(filename, dsamp_r, dsamp_az, out_file):


    image = Image.open(filename)

    r_dim = int(image.size[0]/dsamp_r)
    az_dim = int(image.size[1]/dsamp_az)
    downsampled_image = image.resize((r_dim, az_dim), resample=Image.BILINEAR)

    # Save the downsampled image

    downsampled_image = np.asarray(downsampled_image).astype('float').T

    plt.imshow(downsampled_image)
    plt.show()
    write_watermask(downsampled_image, out_file, downsampled_image.shape[0])


def resample_watermask(filename, look_str, out_file):


    image = Image.open(filename)

    par = SLC_Par(working_dir + sub_dir + 'rmli_' + look_str + '/rmli_' + look_str + '.ave.par')
    resampled_image = image.resize(par.dim, resample=Image.BILINEAR)

    # Save the downsampled image

    resampled_image = np.asarray(resampled_image).astype('float').T

    plt.imshow(resampled_image)
    plt.show()
    write_watermask(resampled_image, out_file, resampled_image.shape[0])


def phasecmap():
    cyan = '#00ffff'
    magenta = '#ff00ff'
    yellow = '#ffff00'
    red = '#ff0000'
    anglemap = col.LinearSegmentedColormap.from_list(
        'anglemap', [red, magenta, cyan, yellow, red], N=256, gamma=1)
    return anglemap

def crop_ras(filename, outname, crop_param):

    im = read_ras(filename)[0]

    x = crop_param[0]
    x_dim = crop_param[0] + crop_param[2]
    y = crop_param[1]
    y_dim = crop_param[1] + crop_param[3]

    cropped = im.T[x:x_dim, y:y_dim]

    plt.imshow(cropped.T)
    plt.show()

    out = cropped.astype('float')
    print(out.shape)
    write_watermask(out, outname, out.shape[0])


if __name__ == "__main__":

    #erode_watermask(working_dir + sub_dir + 'dem_7_6/topo/xi_mask.ras', working_dir + sub_dir + 'dem_7_6/topo/xi_mask_eroded', iters=3)

    #downsample_watermask(working_dir + sub_dir + 'dem_1_1/water_mask_1_1.ras', 2, 2, working_dir + sub_dir + 'dem_2_2/water_mask_2_2')

    resample_watermask(working_dir + sub_dir + 'dem_7_6/water_mask_7_6_tdxgauss3.ras', '2_2', working_dir + sub_dir  + 'dem_2_2/water_mask_2_2_tdxgauss3')

    #crop_ras(working_dir + 'watermask_1_1_eroded.ras', working_dir + 'watermask_1_1_eroded_sites', [4270, 3700, 1498, 2200])

    #mk_base_plot()

