import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sarlab.gammax import *
from scipy.ndimage import filters as snf

master = '20190822'
working_dir = '/local-scratch/users/aplourde/RS2_ITH/post_cr_installation/ovr_site1/'
ingest_cfg = {'password': 'S1m0nF7@SER', 'polarizations': 'HH'}                                                     #2
stack = SLC_stack(dirname=working_dir, name='inuvik_postcr', master=master, looks_hr=(3,3), looks_lr=(12,12), multiprocess=True, rdcdem_refine=True, skipmode='None', ingest_cfg=ingest_cfg)

#boxcar coherence with amplitude
def boxcarcoherence(d,i1,i2,ws=5,window=np.array([])):
    a1 = np.sqrt(i1)
    a2 = np.sqrt(i2)
    d  = d/np.abs(d)*a1*a2

    if not window.any():
        window=np.ones((ws,ws))
    window = np.array(window).ravel()

    dr = snf.generic_filter(d.real, lambda x: np.sum(x*window), size=ws, mode="nearest")

    di = snf.generic_filter(d.imag, lambda x: np.sum(x*window), size=ws, mode="nearest")

    a1 = snf.generic_filter(i1, lambda x: np.sum(x*window), size=ws, mode="nearest")

    a2 = snf.generic_filter(i2, lambda x: np.sum(x*window), size=ws, mode="nearest")

    f = (dr+1j*di)/np.sqrt(a1*a2)
    c = np.abs(f)
    return c,f

def oversample_rslcs(rslcs, bspline = False):

    for rslc in rslcs:
        #rslc = SLC(filename=file, par=SLC_Par(file + '.par'))
        #print(rslc.filename, rslc.par)
        if bspline:
            run('SLC_ovr', rslc, rslc+'.par', rslc+'.ovr.b', rslc+'.ovr.b.par', '16', '16', 1, 4)
            run('rasSLC', rslc+'.ovr.b', 1024)
        else:
            run('SLC_ovr', rslc, rslc+'.par', rslc+'.ovr', rslc+'.ovr.par', '16', '16', 0, 4)
            run('rasSLC', rslc+'.ovr', 1024)

def plot_ifg(rslcs):

    ptarg_dir = '/local-scratch/users/aplourde/RS2_ITH/post_cr_installation/ptarg_site1/'
    plt_dir = '/local-scratch/users/aplourde/RS2_ITH/post_cr_installation/ptarg_vs_ovr/ifg_pngs/'

    #64x64
    #ext = '_64x64'
    #ext_len = 5
    #rslc_names = [os.path.basename(rslc) for rslc in rslcs]

    #ovr
    #ext = '_ovr'
    #ext_len = 13
    #rslc_names = [os.path.basename(rslc) + '.ovr' for rslc in rslcs]

    #ptarg
    ext = '_ptarg'
    rslc_names = [os.path.basename(rslc)[:-5] + '_cr1.rslc' for rslc in rslcs]
    ext_len = 9

    rslcs_sorted = sorted(rslc_names)

    for i in range(len(rslcs_sorted)-1):
        if ext == '_64x64':
            c1 = readBin(working_dir + 'rslc/' + os.path.basename(rslcs_sorted[i]), [64 ,64], 'complex64')
            c2 = readBin(working_dir + 'rslc/' + os.path.basename(rslcs_sorted[i+1]), [64, 64], 'complex64')
        elif ext == '_ovr':
            c1 = readBin(working_dir + 'rslc/' + os.path.basename(rslcs_sorted[i]), [1024 ,1024], 'complex64')
            c2 = readBin(working_dir + 'rslc/' + os.path.basename(rslcs_sorted[i+1]), [1024, 1024], 'complex64')
        elif ext == '_ptarg':
            c1 = readBin(ptarg_dir + 'rslc/' + os.path.basename(rslcs_sorted[i]), [1024 ,1024], 'complex64')
            c2 = readBin(ptarg_dir + 'rslc/' + os.path.basename(rslcs_sorted[i+1]), [1024, 1024], 'complex64')
        else:
            pass

        ph = np.angle(c1 * np.conj(c2))

        plt.imshow(ph.T)
        plt.title(rslcs_sorted[i] + ' - ' + rslcs_sorted[i+1])
        plt.colorbar()
        #plt.show()
        plt.savefig(plt_dir + rslcs_sorted[i][:-ext_len] + '_' + rslcs_sorted[i+1][:-ext_len] + ext + '.png')
        plt.close()

        #c, f = boxcarcoherence(ph, np.abs(c1*c1), np.abs(c2*c2), ws=2)
        #plt.imshow(c.T, cm.Greys_r)
        #plt.title('Coherence (64x64)')
        #plt.savefig(plt_dir + 'coherence_64x64.png')
        #plt.show()
        #plt.close()

def ptarg_vs_ovr(rslcs):

    ptarg_dir = '/local-scratch/users/aplourde/RS2_ITH/post_cr_installation/ptarg_site1/'
    plt_dir = '/local-scratch/users/aplourde/RS2_ITH/post_cr_installation/ptarg_vs_ovr/ifg_pngs/ptarg_vs_ovr/'

    rslcs = [os.path.basename(rslc) for rslc in rslcs]

    ovr_ext_len = 9
    ptarg_ext_len = 5


    for rslc in rslcs:
        c1 = readBin(working_dir + 'rslc/' + rslc + '.ovr', [1024, 1024], 'complex64')
        c2 = readBin(ptarg_dir + 'rslc/' + rslc[:-5] + '_cr1.rslc', [1024, 1024], 'complex64')


        ph = np.angle(c1 * np.conj(c2))
        intensity = np.abs(c1 * np.conj(c2))

        plt.imshow(ph.T)
        plt.title('SLC_ovr vs ptarg\n' + rslc)
        plt.colorbar()
        plt.savefig(plt_dir + rslc + '.png')
        #plt.show()
        plt.close()

        print(np.min(ph), np.max(ph), np.mean(ph))

    #c, f = boxcarcoherence(ph, np.abs(c1*c1), np.abs(c2*c2), ws=2)
    #plt.imshow(c.T, cm.Greys_r)
    #plt.title('Coherence (64x64)')
    #plt.savefig(plt_dir + 'coherence_64x64.png')
    #plt.show()


if __name__ == "__main__":

    rslcs = glob.glob(working_dir + 'rslc/*.ovr')
    #oversample_rslcs(rslcs, bspline = True)

    #plot_ifg(rslcs)

    #ptarg_vs_ovr(rslcs)

    for rslc in rslcs:
        print(rslc)
        run('cp_data', rslc, rslc[:-4], 4096, None, None, None, None, None, None)
        run('rasSLC', rslc[:-4], 1024, None, None, None, None, None, None, 1, 0, None, rslc[:-4] + '.ras')


