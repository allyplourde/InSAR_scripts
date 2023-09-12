from sarlab.gammax import *
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from numpy.linalg import lstsq
import pandas as pd
import os
from scipy.constants import c
import shutil

master = '20190822'
working_dir = '/local-scratch/users/aplourde/RS2_ITH/post_cr_installation/'
sub_dir = 'full_scene/'
#sub_dir = 'full_scene/summer_season/'
#sub_dir = 'crop_site1_only/'
#sub_dir = 'crop_site1_only/summer_season/'
#sub_dir = 'crop_large/'
#sub_dir = 'ptarg_site1/'
ingest_cfg = {'password': 'S1m0nF7@SER', 'polarizations': 'HH'}
stack = SLC_stack(dirname=working_dir + sub_dir,name='inuvik_RS2_U76_D', master=master, looks_hr=(2,3), looks_lr=(12,18), multiprocess=True, rdcdem_refine=True, skipmode='None', ingest_cfg=ingest_cfg)

#-----------------------------------------------------------------------------------------------------------------------
def read_bperp_file(bperp_file):
    #read all fields from bperp file

    master_unsort = np.loadtxt(bperp_file, dtype = 'str', skiprows=0, usecols = (1,))
    #master_unsort = np.array([name[2:-1] for name in master_unsort])


    slave_unsort = np.loadtxt(bperp_file, dtype = 'str', skiprows=0, usecols = (2,))
    #slave_unsort = np.array([name[2:-1] for name in slave_unsort])


    dt_unsort = np.loadtxt(bperp_file, dtype='float', skiprows=0, usecols = (4,))
    bperp_unsort = np.loadtxt(bperp_file, dtype='float', skiprows=0, usecols = (3,))

    ifg_names_unsort = []
    for i in range(len(master_unsort)):
        ifg_names_unsort.append(master_unsort[i]+'_'+slave_unsort[i])
    ifg_names_unsort = np.asarray(ifg_names_unsort)

    #apply pair name sorting
    sort_key = np.argsort(ifg_names_unsort)
    ifg_names = ifg_names_unsort[sort_key]
    master = master_unsort[sort_key]
    slave = slave_unsort[sort_key]
    dt = dt_unsort[sort_key]
    bperp = bperp_unsort[sort_key]

    return ifg_names, master, slave, dt, bperp


#-----------------------------------------------------------------------------------------------------------------------
def read_ifgs(glob_string, dim):
    #read all interferograms matching glob string and return as single array
    ifgs = sorted(glob(glob_string))
    n_ifgs = len(ifgs)
    m_pix = dim[0]*dim[1]
    phi = np.zeros((n_ifgs, m_pix))
    cc = np.zeros((n_ifgs, m_pix))
    for i, ifg in enumerate(ifgs):

        try:
            #im = readBin(ifg, dim, 'float32').flatten()
            #phi[i, :] = im
            im = readBin(ifg, dim, 'complex64').flatten()
            phi[i, :] = np.angle(im)
            cc_im = readBin(ifg[:-4] + '.cc', dim, 'float32').flatten()
            cc[i, :] = cc_im
        except:
            #print("here")
            phi[i, :] = np.zeros(dim).flatten()
            cc[i, :] = np.zeros(dim).flatten()


    return phi, cc

def find_a(phi, bperp, cc, plot = False, pixel = None, method=None):

    y = []
    x = []
    w = []
    a = []
    if pixel is not None:
        ifg = phi.shape[0]
        iter = [pixel]
    else:
        ifg, pixel = phi.shape
        iter = range(pixel)
    for j in iter:
        print(j)
        for i in range(ifg):
            #j = 222
            phi_j = phi[i][j]
            cc_j = cc[i][j]

            if method == 'nonzero' and phi_j == 0:
                # do something
                phi_j = np.nan

            y.append(phi_j)
            x.append(bperp[i])
            w.append(cc_j)

        x = np.asarray(x) #bperp
        y = np.asarray(y) #phi
        w = np.asarray(w) #cc
        mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(w)
        x = x[mask]
        y = y[mask]
        w = w[mask]
        M = x[:, np.newaxis]*[0, 1]
        W = np.sqrt(np.diag(w))
        #p, res, rnk, s = lstsq(M,y, rcond=None)
        p, res, rnk, s = lstsq(np.dot(W,M), np.dot(y,W))
        a.append(p[1])

        if plot:
            plt.plot(x,y, 'o')
            xx = np.linspace(min(x), max(x), 100)
            yy = p[1]*xx
            plt.plot(xx,yy)
            plt.title("Pixel {}".format(j))
            plt.xlabel("Perpendicular Baseline")
            plt.ylabel("Phi")
            plt.savefig("a_est.png")
            return

        y = []
        x = []
        w = []

    return a

def height_error_map(phi, bperp, a, dim, L, theta_deg, R, ifg_names):

    h_err = []
    ave_height_error = np.ones(dim)
    for i in range(len(bperp)):
        phi_H = (-1) * a * bperp[i]
        height_error = (-1) * phi_H * L * R * np.sin(np.radians(theta_deg)) / (4 * np.pi * bperp[i])
        height_error = np.asarray(height_error)
        h_err.append(height_error)

        height_error = height_error.reshape(dim)
        ave_height_error += height_error

        #writeBin('results/' + ifg_names[i] + '.herr', height_error)
    
    #average height error map
    ave_height_error = ave_height_error / len(ifg_names)
    plt.imshow(height_error)#, vmin = -25.0, vmax = 25.0)
    plt.colorbar()
    plt.title("Mean Height Error (m)")
    plt.savefig("ave_herr.png")
    plt.close()

    h_err = np.asarray(h_err)

    #sl.writeBin('results/' + 'ave.herr', height_error)

    return h_err, ave_height_error.flatten()
    
def calculate_deformation(phi, cc, bperp, a, dim, ifg_names, ifg_number = None, pixel_number = None):
    cum_phi = []
    dates = []
    if ifg_number is None and pixel_number is None:
        ifg, pixel = phi.shape
        phi_0 = np.ones(phi.shape)
        for p in range(pixel):
            for i in range(ifg):
                #if cc[i][p] < 0.3:
                #    phi_0[i][p] = np.nan
                #else:
                #    phi_0[i][p] = phi[i][p] - a[p] * bperp[i]
                #    #print("here")
                phi_0[i][p] = phi[i][p] - a[p] * bperp[i]
    elif ifg_number and pixel_number is None:
        ifg, pixel = phi.shape
        phi_0 = np.ones(a.shape)
        for p in range(pixel):
            phi_0[p] = phi[ifg_number][p] - a[p] * bperp[ifg_number]

        plt.imshow(phi_0.reshape(dim))
        plt.title("Phase with subtracted Height Error\n{}".format(ifg_names[ifg_number]))
        plt.colorbar()
        plt.savefig('testifg.png')

    elif ifg_number is None and pixel_number:

        mask, dates = get_leapfrog_indices(ifg_names)
        phi = phi[mask]
        bperp = bperp[mask]

        ifg, pixel = phi.shape
        phi_0 = np.ones(bperp.shape)
        for i in range(ifg):
            phi_0[i] = phi[i][pixel_number] - a[pixel_number] * bperp[i]
        
        cum_phi = np.cumsum(phi_0)
        cum_phi = np.insert(cum_phi, 0, 0)
        #plt.plot(range(len(phi_0)),phi_0)


        plt.title("Cumulative Phase Difference of Leap Frog Interferograms\nPixel {}".format(pixel_number))
        ax = plt.gca()
        ax.plot(range(len(cum_phi)),cum_phi)
        ymin, ymax = ax.get_xlim()
        ax.set_xticks(np.round(np.linspace(ymin, ymax, 5), 2))
        ax.set_xlabel("Date")
        ax.set_ylabel("Phase (rad)")
        plt.savefig('524288.png')
    else:
        print("Error: enter valid ifg or valid pixel but not both.")
    mask, dates = get_leapfrog_indices(ifg_names)
    return phi_0#, cum_phi, dates

def calculate_deformation_lstsq(phi, bperp, a, dim, ifg_names, pixel_number = None):


    ifg, pixel = phi.shape
    phi_0 = np.ones(bperp.shape)
    for i in range(ifg):
        phi_0[i] = phi[i][pixel_number] - a[pixel_number] * bperp[i]

    M = get_leapfrog_matrix(ifg_names)

    y = phi_0
    #M = x[:, np.newaxis]*[0, 1]
    p, res, rnk, s = lstsq(M,y, rcond=None)
    phi_lf = np.zeros(bperp.shape)
    for row in range(len(M)):
        #if row > 0:
        #    phi_lf[row] = phi_lf[row-1]
        for i in range(len(M[row])):
            if M[row][i] == 1:
                phi_lf[row] += phi_0[i]

    mask, dates = get_leapfrog_indices(ifg_names)

    phi_lf = phi_lf[mask]
    phi_0 = phi_0[mask]
    
    plt.plot(range(len(phi_lf)), phi_lf)
    plt.plot(range(len(phi_0)), phi_0)
    plt.show()

def get_leapfrog_indices(ifg_names):
    ifg = [name.split('.')[0] for name in ifg_names]

    master = [date.split('_')[0] for date in ifg]
    slave = [date.split('_')[1] for date in ifg]
    
    lf_index = []
    lf_index.append(0)
    current_master = master[0]
    for i in range(len(master)):
        if master[i] == current_master:
            pass
        else:
            current_master = master[i]
            lf_index.append(i)

    dates = [master[0]]
    for i in lf_index:
        dates.append(slave[i])

    dates = pd.to_datetime(dates, format=("%Y%m%d"))

    print("NUM LF DATES: ", len(dates))
    
    return lf_index, dates

def get_leapfrog_matrix(ifg_names):
    ifg = [name.split('.')[0] for name in ifg_names]

    master = ['20' + date.split('_')[0] for date in ifg]
    slave = [date.split('_')[1] for date in ifg]
    
    lf_matrix = np.zeros((len(ifg), len(ifg)))


    current_master = master[0]
    current_master_index = 0
    val = 0
    for i in range(len(master)):
        if master[i] == current_master:
            ind = i - val
            while ind >= current_master_index:
                lf_matrix[i][ind] = 1
                ind -= 1
        else:
            current_master = master[i]
            current_master_index = i
            val = i - current_master_index
            lf_matrix[i][current_master_index] = 1

    #dates = [master[0]]
    #for i in lf_matrix:
    #    dates.append(slave[i])

    #dates = pd.to_datetime(dates, format=("%Y%m%d"))
    
    return lf_matrix
    

def find_lf(ifg_names):
    mask, dates = get_leapfrog_indices(ifg_names)
    lf_ifg_names = ifg_names[mask]
    for ifg in lf_ifg_names:
        files = glob(working_dir + sub_dir + 'diff_'+look+'/' + ifg + '*')
        for file in files:
            basename = os.path.basename(file)
            shutil.copyfile(file, working_dir + sub_dir + 'diff_'+look+'/lf/' + basename)


#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    look = 'fr'
    ext = '.diff'
    rslc_dir = working_dir + sub_dir + 'rslc/'
    print('mk_tab', rslc_dir, '.rslc', '.rslc.par', 'SLC_tab')
    #run('mk_tab', rslc_dir, '.rslc', '.rslc.par', 'SLC_tab')

    #generate bperp
    SLC_tab = working_dir+sub_dir + 'SLC_tab'
    master_par = working_dir+sub_dir + 'rslc/'+master+'.rslc.par'
    out = working_dir + sub_dir + master + '.bperp'
    itab = working_dir + sub_dir + master + '.itab'
    print('base_calc', SLC_tab, master_par, out, itab, 1, 2)
    #run('base_calc', SLC_tab, master_par, out, itab, 1, 2)

    #parameters of the scene
    master_par = SLC_Par(master_par)
    f = master_par['radar_frequency']
    theta_deg = master_par['incidence_angle']  #scene centre incidence angle (degrees)
    R = master_par['center_range_slc']     #scene centre range (m)

    bperp_file = working_dir + sub_dir + master + '.bperp'

    theta = theta_deg*np.pi/180.0   #convert scene centre incidence angle to radians
    L = c/f     #SAR wavelength (m)
    if look == 'fr':
        dim = master_par.dim
    elif look == 'hr':
        dim = (333,333)
        #dim = (3810, 3810)

    #read unwrapped interferograms in sorted pairname order
    phi, cc = read_ifgs(working_dir + sub_dir + 'diff_'+look+'/*'+ext, dim)

    #read fields from bperp file
    ifg_names, master, slave, dt, bperp = read_bperp_file(bperp_file)
    ifg_names = ifg_names[:-1]

    master = master[:-1]
    slave = slave[:-1]
    dt = dt[:-1]
    bperp = bperp[:-1]

    a_file = working_dir + sub_dir + 'a_est.txt'
    if os.path.exists(a_file) and False:
        a = np.loadtxt(a_file, dtype = 'str', skiprows=0, usecols = (0,))
        a = [float(x) for x in a]
        a = np.asarray(a)
    else:
        print("finding a...")
        a = find_a(phi, bperp, cc)
        np.savetxt(a_file, a)
        a = [float(x) for x in a]
        a = np.asarray(a)
    h_err, ave_h_err = height_error_map(phi, bperp, a, dim, L, theta_deg, R, ifg_names)

    """
    standard_dev = []
    for j in range(len(h_err[0])):
        val = []
        for i in range(len(h_err)):
            val.append(h_err[i][j])
        standard_dev.append(np.std(val))
        #print(np.mean(val), np.min(val), np.max(val), np.std(val))
    print(np.mean(standard_dev))
    """



    print("calculating deformation...")
    phi_0 = calculate_deformation(phi, cc, bperp, a, dim, ifg_names)


    ifg, pixel = phi.shape
    for i in range(ifg):
        phase = phi_0[i]#.reshape(dim)
        complex = [np.cos(phi) + 1j*np.sin(phi) for phi in phase]
        new_diff = np.asarray(complex).reshape(dim)
        writeBin(working_dir+sub_dir+'diff_'+look+'/'+ifg_names[i]+ext+'.hc', new_diff)
        run('rasmph', working_dir+sub_dir+'diff_'+look+'/'+ifg_names[i]+ext+'.hc', dim[0], None, None, None, None, None, None, None, working_dir+sub_dir+'diff_'+look+'/'+ifg_names[i]+ext+'.hc' + '.ras', 0)


    #find_lf(ifg_names)


    """
    #QUESTION 2
    a_nonzero = df['a_nonzero']
    if False:
        find_a(phi, bperp, df, label='a_nonzero')
        h_err, ave_h_err = height_error_map(phi, bperp, a_nonzero, dim, L, theta_deg, R, ifg_names)
        phi_0 = calculate_deformation(phi, bperp, a_nonzero, dim, ifg_names, ifg_number = 25)
    
    #QUESTION 3
    if True:
        pn = 2624
        phi_0, cum_phi, dates = calculate_deformation(phi, bperp, a, dim, ifg_names, ifg_number = None, pixel_number = pn)

        #QUESTION 4
        m_to_cm = 100
        # x = (L * cum_phi) / (2 * np.pi * np.cos(np.radians(theta_deg))) * m_to_cm
        x = ( (L * cum_phi) / (2 * np.pi) ) * m_to_cm
        ax = plt.gca()
        ax.plot(dates, x)
        plt.title("Cumulative Deformation of Leap Frog Interferograms\nPixel {}".format(pn))
        ymin, ymax = ax.get_xlim()
        ax.set_xticks(np.round(np.linspace(ymin, ymax, 5), 2))
        ax.set_xlabel("Date")
        ax.set_ylabel("Deformation (cm)")
        plt.show()

    #QUESTION 5
    if False:
        phi_0, cum_phi, dates = calculate_deformation(phi, bperp, a_nonzero, dim, ifg_names, ifg_number = None, pixel_number = 72269)

        m_to_cm = 100
        x = (L * cum_phi) / (2 * np.pi * np.cos(np.radians(theta_deg))) * m_to_cm
        plt.plot(dates, x)
        plt.title("Cumulative Deformation\n of Leap Frog Interferograms")
        plt.xlabel("Date")
        plt.ylabel("Deformation (cm)")
        plt.show()

    df.to_csv("a_values.csv")

    #QUESTION 7
    if False:
        calculate_deformation_lstsq(phi, bperp, a_nonzero, dim, ifg_names, pixel_number = 72269)
        
        m_to_cm = 100
        x = (L * cum_phi) / (2 * np.pi * np.cos(np.radians(theta_deg))) * m_to_cm
        plt.plot(dates, x)
        plt.title("Cumulative Deformation\n of Leap Frog Interferograms")
        plt.xlabel("Date")
        plt.ylabel("Deformation (cm)")
        plt.show()
        



    #QUESTION 8
    # improve phase modelling with coherence weight
    """

