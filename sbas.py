import numpy as np
from sarlab.gammax import *


def temporal_filter(ifgs):
    return ifgs


def spatial_filter(ifgs):
    return ifgs


def calculate_displacement(ifgs):
    #minimum mean squared error?
    return ifgs

def matrix_SVD(matrix):
    U, s, Vh = np.linalg.svd(matrix)
    return U, s, Vh


def process(diff_dir):
    ifgs = glob.glob(diff_dir + '*.diff')

    #subtract interferograms
    sifgs = ifgs[1:] - ifgs[:-1]

    #filter the network
    tmp_flt = temporal_filter(sifgs)
    spat_flt = spatial_filter(tmp_flt)

    #displacement map
    disp_map = calculate_displacement(spat_flt)