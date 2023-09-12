from sarlab.gammax import *
import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import rice
from skimage.transform import resize

from cr_phase_to_deformation import get_itab_diffs


working_dir = '/local-scratch/users/aplourde/HDS/projects/southern_ITH/RS2_U76_D/'
sub_dir = 'crop_sites/'; master = '20180827'

ingest_cfg = {'password': 'S1m0nF7@SER', 'polarizations': 'HH'}                                                     #2
stack = SLC_stack(dirname=working_dir + sub_dir, name='inuvik_postcr', reg_mask=None, master=master, looks_hr=(2,3), looks_lr=(12,18), multiprocess=True, rdcdem_refine=True, skipmode='None', ingest_cfg=ingest_cfg)


def synthetic_topographic_phase(B, dim, dem, inc, R):

    freq = 5.4049992e+09
    lamb = c / freq
    h = dem
    theta = inc

    # Altitude of Ambiguity
    alt_ambiguity = np.nanmean(lamb * R * np.sin(theta) / (2 * B))
    print(alt_ambiguity)

    phi = (4*np.pi*B*h) / (lamb*R*np.sin(theta))

    """
    plt.subplot(121)
    #plt.imshow(np.rad2deg(inc.T), cmap='terrain')
    plt.imshow(dem.T, cmap='terrain')
    #plt.imshow((dem - np.nanmean(dem)).T, cmap='terrain')
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(phi_c.T, cmap=cm.rainbow)
    plt.colorbar()
    plt.show()
    """

    # return the wrapped phase
    return np.exp(1j*phi)

def phase_noise_model(dim, phi_c, rmli, cc):

    # standard circular gaussian
    #noise_factor = np.random.rand()
    u1 = np.random.normal(size=dim, scale=0.01)
    u2 = np.random.normal(size=dim, scale=0.01)
    print(np.mean(u1), np.std(u1), np.min(u1), np.max(u1))

    # coherence
    rho = cc

    # amplitude
    A = rmli**2
    print(np.mean(A), np.std(A), np.min(A), np.max(A))

    B = A*rho*np.exp(-1j*phi_c)
    C = A*np.sqrt(1-rho**2)

    z1 = np.zeros(u1.shape)
    z2 = np.zeros(u2.shape, dtype=np.complex)
    for i in range(dim[0]):
        for j in range(dim[1]):

            u = np.array([[u1[i, j]], [u2[i, j]]], dtype=object)

            cholesky = [[A[i, j], 0], [B[i, j], C[i, j]]]
            cholesky = np.array(cholesky, dtype=object)

            z = np.matmul(cholesky, u)

            z1[i, j] = z[0][0]
            z2[i, j] = z[1][0]


    return z2



def distribution_stats(files, par, dtype='float32'):

    #mean coherance
    ims = []
    #count = 0
    for file in files:
        im = readBin(file, par.dim, 'float32')
        ims.append(im)

    ims = np.array(ims)
    print(ims.shape)

    mean = np.mean(ims, axis=0)
    ave_cc = working_dir + sub_dir + 'diff_fr/ave.flt.cc'
    writeBin(ave_cc, mean)
    run('rascc_bw', ave_cc, None, par.dim[0])

    std = np.std(ims, axis=0)
    std_cc = working_dir + sub_dir + 'diff_fr/std.flt.cc'
    writeBin(std_cc, std)
    run('rascc_bw', std_cc, None, par.dim[0])

    # Fit a Rice distribution to the data using maximum likelihood estimation
    """
    s = np.zeros(par.dim)
    sigma = np.zeros(par.dim)
    loc = np.zeros(par.dim)
    for i in range(par.dim[0]):
        for j in range(par.dim[1]):
            x = ims[:, i, j]
            s[i, j], loc[i, j], sigma[i, j] = rice.fit(x)

    print(s.shape)
    # Calculate the mean and variance of the fitted distribution
    mean = np.sqrt(2) * sigma + loc
    var = s ** 2 + 2 * sigma ** 2

    print(mean.shape)
    plt.imshow(mean.T)
    plt.show()

    coh = abs(mean)**2 / (abs(mean)**2 + var)

    print(coh.shape)
    plt.imshow(coh.T)
    plt.show()
    """

def mk_coherence(ifgs, par):

    for ifg in ifgs:
        root = os.path.basename(ifg)
        dates = root.split('.')[0]
        master, slave = dates.split('_')
        rmli_master = working_dir + sub_dir + 'rmli_1_1/' + master + '.rmli'
        rmli_slave = working_dir + sub_dir + 'rmli_1_1/' + slave + '.rmli'
        cc = ifg + '.cc'

        cc_win = 3
        cc_wgt = 1
        run('cc_wave', ifg, rmli_master, rmli_slave, cc, par.dim[0], cc_win, cc_win, cc_wgt)
        run('rascc_bw', cc, None, par.dim[0])


if __name__ == "__main__":

    master_par = stack.master_slc_par
    dem = readBin(working_dir + sub_dir + 'dem_1_1/seg.dem.rdc', master_par.dim, 'float32')
    inc = readBin(working_dir + sub_dir + 'dem_1_1/inc.rdc', master_par.dim, 'float32')
    R = readBin(working_dir + sub_dir + 'dem_1_1/range', master_par.dim, 'float32')
    rmli = readBin(working_dir + sub_dir + 'rmli_1_1/rmli_1_1.ave', master_par.dim, 'float32')

    plt.subplot(121)
    phi_c = synthetic_topographic_phase(100, master_par.dim, dem, inc, R)
    plt.title("Baseline = 100 m\n Altitude of Ambiguity = 110.7 m")
    plt.imshow(np.angle(phi_c.T), cmap='rainbow')
    plt.colorbar()
    plt.subplot(122)
    phi_c = synthetic_topographic_phase(400, master_par.dim, dem, inc, R)
    plt.title("Baseline = 400 m\n Altitude of Ambiguity = 27.7 m")
    plt.imshow(np.angle(phi_c.T), cmap='rainbow')
    plt.colorbar()
    plt.show()
    """

    #ifgs = glob.glob(working_dir + sub_dir +'diff_fr/*.flt')  # flat earth corrected ifgs
    #mk_coherence(ifgs, master_par)

    #ccs = glob.glob(working_dir + sub_dir + 'diff_fr/*.flt.cc')
    #itab = working_dir + sub_dir + 'itab_lf'
    #RSLC_tab = working_dir + sub_dir + 'RSLC_tab'
    #ccs = get_itab_diffs(ccs, itab, RSLC_tab)

    cc = readBin(working_dir + sub_dir + 'hds/diff_coh_hds', master_par.dim, 'float32')

    # clean interferometric phase
    baselines = np.random.rand(2000) * 500
    #baselines = np.random.rand(1) * 500
    #baselines = [100]
    #for i, b in enumerate(baselines):
    files = glob.glob("/local-scratch/users/aplourde/deeplearningcourse/project/newtrain/labels/*ifg.dsmp")

    for f in files:
        print(f)
        #phi_c = synthetic_topographic_phase(b, master_par.dim, dem, inc, R)
        rmli = resize(rmli, (128, 128), anti_aliasing=True)
        cc = resize(cc, (128, 128), anti_aliasing=True)
        phi_c = readBin(f, [128,128], "complex64")
        #phi_n = phase_noise_model([128, 128], phi_c, rmli, cc)

        """ """
        plt.subplot(131)
        plt.imshow(np.angle(phi_c.T))
        plt.colorbar()
        plt.subplot(132)
        plt.imshow(np.angle(phi_n.T))
        plt.colorbar()
        plt.subplot(133)
        
        plt.imshow(np.angle(phi_nn.T))
        plt.colorbar()
        plt.show()
        break
        """ """

        noise = np.random.normal(0, np.pi, phi_c.shape)
        phi_nn = phi_c + noise

        #s = str(i)
        #padded = '0' * (4 - len(s)) + s
        #print(padded)


        feature_dir = "/local-scratch/users/aplourde/deeplearningcourse/project/newtrain/features/"
        #label_dir = "/local-scratch/users/aplourde/deeplearningcourse/project/newtrain/labels/"
        #writeBin(feature_dir + padded + '.ifg', phi_n)
        #writeBin(label_dir + padded + '.ifg', phi_c)
        #run('rasmph', feature_dir + padded + '.ifg', master_par.dim[0])
        #run('rasmph', label_dir + padded + '.ifg', master_par.dim[0])

        root = os.path.basename(f)
        writeBin(feature_dir + root, phi_nn)
        #run('rasmph', feature_dir + padded + '.ifg', master_par.dim[0])
        #run('rasmph', label_dir + padded + '.ifg', master_par.dim[0])
    """