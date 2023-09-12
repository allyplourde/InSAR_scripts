import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy import constants
from scipy.interpolate import RectBivariateSpline
from sarlab.gammax import SLC_Par, SLC_stack, readBin, writeBin, run

master = '20170808'
ingest_cfg = {'password': 'S1m0nF7@SER', 'polarizations': 'HH'}

c = constants.c

#TODO TODAY: take subset of hds over site #1, compute vertical deflection over 1 year (20190915,20200909)
# compare to tilt logger.
# Download sentinel-1 imagery, see if corner reflectors are still standing by creating an average rmli.
# Read CR articles



#import sys
#sys.path.append('/home/akplourd/dev/sarlab-dev/users/jaysone/')

look = 'fr'
#working_dir = '/local-scratch/users/aplourde/RS2_ITH/20190611-20191009/'
working_dir = '/local-scratch/users/aplourde/RS2_ITH/full_scene/'
#working_dir = '/local-scratch/users/jaysone/projects_active/inuvik/RS2_U76_D/small/'

stack = SLC_stack(dirname=working_dir,name='inuvik_RS2_U76_D', master=master, looks_hr=(2,3), looks_lr=(12,18),
                  multiprocess=True, rdcdem_refine=True, skipmode='None', ingest_cfg=ingest_cfg)

par = stack.master_slc_par
wavelength = constants.c / par['radar_frequency']
print(wavelength, par['incidence_angle'])

ftol = 1e-12
gtol = 1e-12
minimize_options = {'ftol': ftol, 'gtol': gtol}

if look == 'hr':
    diff_dir = working_dir + 'diff_hr/'
    rmli_dir = working_dir + 'rmli_hr/'
    width = 3811
    crop23 = [2445, 1315, 40, 40]

elif look == 'fr':
    diff_dir = working_dir + 'diff_fr/'
    #diff_dir = working_dir + 'diff/'
    rmli_dir = working_dir + 'rmli_fr/'
    rslc_dir = working_dir + 'rslc/'
    #width = 7623
    #height = 11422
    width = 7621
    height = 11393
    #width = 996
    #height = 2592
    crop23 = [4890, 3980, 100, 80]
    crop1 = [5000, 5660, 100, 100]

diffs = sorted(glob(diff_dir + '*diff.adf'))
#diffs = sorted(glob(diff_dir + '*diff.natm.hds'))
diffs = [os.path.basename(diff) for diff in diffs]

ave_rmli = sorted(glob(rmli_dir + '*ave.rmli'))
ave_rmli = [os.path.basename(rmli) for rmli in ave_rmli]

rslcs = sorted(glob(rslc_dir + '*.rslc'))
rslcs = [os.path.basename(rslc) for rslc in rslcs]

sims = sorted(glob(diff_dir + '*sim_unw'))
sims = [os.path.basename(sim) for sim in sims]


cr_loc = pd.read_csv('cr_loc_full_scene.csv')
print(cr_loc)
subset_offset_small = np.asarray((4500, 3492))
for i in range(len(cr_loc)):
    cr_loc['x_fr'][i]-=subset_offset_small[0]
    cr_loc['y_fr'][i]-=subset_offset_small[1]

#model the temporal history of the tcr in clutter through a stack of SLC images
def cr_stack_analysis(slcnames, sim_unw_names, cr_loc, win_sz, cr_size, label):
    ns = len(sim_unw_names)
    print("ns: ", ns)
    chip_corner = cr_loc - (win_sz - 1) / 2
    slcs = np.zeros((ns, win_sz[0], win_sz[1]), dtype='complex128')
    res=[]
    sim_unw_cr = np.zeros(ns)
    slcs_clutter = np.zeros((ns, win_sz[0], win_sz[1]), dtype='complex128')
    for ii in np.arange(ns):
        print(ii+1, ' of ', ns)
        rslc = working_dir + 'rslc/' + slcnames[ii]
        sim_unw = working_dir + 'diff_' + look + '/' + sim_unw_names[ii]
        par = SLC_Par(rslc+'.par')
        print(rslc, sim_unw)
        slcs[ii,:,:] = readBin(rslc,par.dim,'complex64', crop=(chip_corner[0], chip_corner[1], win_sz[0], win_sz[1]))
        if sim_unw_names[ii] is not None:
            sim_unw_ii = readBin(sim_unw,par.dim,'float32', crop=(chip_corner[0], chip_corner[1], win_sz[0], win_sz[1]))
        else:
            sim_unw_ii = np.zeros(win_sz, dtype='float')

        res_ii, slcs_clutter[ii,:,:] = fit_cr_model(slcs[ii, :, :], par, cr_loc)
        res.append(res_ii)

        # topo correction
        slcs_clutter[ii,:,:] *= np.exp(1j * sim_unw_ii)
        #interpolate the sim phase at the subpixel location
        spline = RectBivariateSpline(np.arange(win_sz[0])-(win_sz[0]-1)/2, np.arange(win_sz[1])-(win_sz[1]-1)/2, sim_unw_ii)
        sim_unw_cr[ii] = spline.ev(res[ii].x[0], res[ii].x[1])
    ret = {}
    ret['drg'] = np.asarray([res[ii].x[0] for ii in np.arange(ns)])
    ret['daz'] = np.asarray([res[ii].x[1] for ii in np.arange(ns)])
    ret['sigm0_clut'] = np.asarray([res[ii].x[2] for ii in np.arange(ns)])
    ret['rcs_cr'] = np.asarray([res[ii].x[3] for ii in np.arange(ns)])
    ret['phi_cr'] = np.asarray([res[ii].x[4] for ii in np.arange(ns)])
    ret['phi_cr'] = np.angle(np.exp(1j*(ret['phi_cr']+sim_unw_cr)))
    ret['sim_unw_cr'] = sim_unw_cr

    wavelength = constants.c/par['radar_frequency']
    rcs_cs_theor = 4*np.pi*cr_size**4/3/wavelength**2*np.ones(ns)

    #estimate complex coherence for the clutter
    #TODO (2021) fix this clutter estimation - it should just use corner quadrants rather than using this model which is prone to bias issues
    clutter_coh = chip_coh(slcs_clutter)

    #phase triangulation
    #TODO get proper t_vec
    t_vec = 24.*np.arange(ns)
    #coh_model = Coh_Model_3param(t_vec, win_sz[0]*win_sz[1])
    #pta_fit = coh_model.fit(clutter_coh, cost='mle', phase_init='lf')
    #ret['phi_clut'] = pta_fit.theta

    if False:

        plt.figure()
        """
        plt.subplot(2,3,1)
        plt.plot(ret['drg'], marker='o'); plt.ylim((-1,1))
        plt.plot(ret['daz'], marker='o'); plt.ylim((-1,1))
        plt.title(label + ' sub-pixel 2D offset')
        plt.xlabel('Acquisition #')
        plt.ylabel('Offset distance [pixels]')
        plt.legend(['range','azimuth' ])
        """
        """
        #plt.subplot(2,3,2)
        plt.plot(10 * np.log10(rcs_cs_theor), marker='o')
        plt.plot(10 * np.log10(ret['rcs_cr']), marker='o')
        plt.plot(10 * np.log10(ret['sigm0_clut']), marker='o'); plt.ylim((-20,30))
        rcs_clut = ret['sigm0_clut']*pixel_area_ground(par)
        plt.plot(10 * np.log10(ret['rcs_cr']/rcs_clut), marker='o')
        plt.title(label + ' Cross-sections')
        plt.xlabel('Acquisition #')
        plt.ylabel('Cross-section [dB m^2]')
        plt.legend(['CR theoretical','CR estimated', 'Clutter [sigma0]', 'CR Signal-to-clutter ratio' ])
        """
        #plt.subplot(2,3,3)
        plt.plot(ret['phi_cr'], marker='o');plt.ylim((-np.pi, np.pi))
        #plt.plot(ret['phi_clut'], marker='o');
        plt.ylim((-np.pi, np.pi))
        plt.title(label + ' CR and Clutter Phases')
        plt.xlabel('Acquisition #')
        plt.ylabel('Phase [radians]')
        plt.legend(['CR phase', 'Clutter phase'])
        """
        plt.subplot(2,3,4)
        plt.imshow(np.abs(clutter_coh), vmin=0, vmax=1)
        plt.title(label + ' Clutter absolute coherence')

        plt.subplot(2,3,5)
        plt.imshow(np.angle(clutter_coh), vmin=-np.pi, vmax=np.pi)
        plt.title(label + ' Clutter phase')
        """
        plt.savefig('results.png')

    return ret

#fit model from single slc image
def fit_cr_model(slc, par, cr_loc):
    dim = np.asarray(slc.shape).astype(int)
    ampl = np.abs(slc)
    # get 2D index of sample with max amplitude
    max_idx = np.unravel_index(ampl.argmax(), ampl.shape)
    cen = max_idx

    #estimate clutter as annulus mean
    sigm0_clut_init = (np.sum(ampl**2)- np.sum((ampl[cen[0]-2:cen[0]+3, cen[1]-2:cen[1]+3])**2))/(dim[0]*dim[1]-25)
    sigm0_clut_lims = (sigm0_clut_init * 0.01, sigm0_clut_init * 100.)

    #estimate corner reflector rcs as sum of center block
    rcs_cr_init = np.sum((ampl[cen[0] - 2:cen[0] + 3, cen[1] - 2:cen[1] + 3]) ** 2 - sigm0_clut_init)*pixel_area_ground(par)
    rcs_cr_init = np.max((rcs_cr_init, sigm0_clut_init**pixel_area_ground(par)))
    rcs_cr_lims = (rcs_cr_init*0.01,rcs_cr_init*100.)

    phi_cr_init = np.angle(slc[max_idx])
    phi_cr_lims = (None, None)

    offset_init = (0., 0.)
    offset_lims = (-1.,1.)

    #just use a fixed Doppler centroid
    fdc_init = _fdc(par, cr_loc)
    fdc_lims = (fdc_init,fdc_init)

    x_init = (offset_init[0],offset_init[1],sigm0_clut_init, rcs_cr_init, phi_cr_init, fdc_init)
    bounds = (offset_lims,offset_lims, sigm0_clut_lims, rcs_cr_lims, phi_cr_lims, fdc_lims)

    opt_result = opt.minimize(fit_cr_model_cost, x_init, bounds=bounds, args=(slc, par, cr_loc),
                              options=minimize_options)
    print(x_init)
    #wrap cr phase angle since it was unconstrained during fit
    opt_result.x[4] = np.angle(np.exp(1j*opt_result.x[4]))
    print(opt_result)
    x = opt_result.x
    drg = x[0]; daz = x[1]; sigm0_clut = x[2]; rcs_cr = x[3]; phi_cr = x[4]; fdc = x[5]
    cr_signal = cr_signal_model(drg, daz, rcs_cr, phi_cr, fdc, par, slc.shape)
    clutter_signal = slc - cr_signal
    opt_result['clutter_signal'] = clutter_signal

    print('sigm0_clut/sigm0_clut_init, rcs_cr/rcs_cr_init, phi_cr-phi_cr_init, fdc-fdc_init', sigm0_clut/sigm0_clut_init, rcs_cr/rcs_cr_init, np.angle(np.exp(1j*(phi_cr-phi_cr_init))),fdc-fdc_init)
    return opt_result, clutter_signal

def pixel_area_ground(par):
    A = par['range_pixel_spacing'] * par['azimuth_pixel_spacing']/np.sin(np.radians(par['incidence_angle']))
    return A

#Doppler centroid for a given point in the slc
def _fdc(par, loc):
    fdp = par['doppler_polynomial'];
    fdp_dot = par['doppler_poly_dot'];
    fdp_ddot = par['doppler_poly_ddot']
    a0 = fdp[0];
    a1 = fdp[1];
    a2 = fdp[2];
    a3 = fdp[3];
    b0 = fdp_dot[0];
    b1 = fdp_dot[1];
    c0 = fdp_ddot[0]

    r_loc = loc[0]*par['range_pixel_spacing']
    t_loc = loc[1]/par['prf']
    fdc_ = a0 + b0 * t_loc + c0 * t_loc ** 2 + (a1 + b1 * t_loc) * r_loc + a2 * r_loc ** 2

    return fdc_

#log-likelihood cost function
def fit_cr_model_cost(x, slc, par, cr_loc):
    #cost is the log likelihood of the ampl chip given the model
    drg = x[0]; daz = x[1]; sigm0_clut = x[2]; rcs_cr = x[3]; phi_cr = x[4]; fdc = x[5]
    dim = slc.shape

    #compute deterministic signal function of the CR including 2D shift, amplitude, phase offset and Doppler phase ramp
    cr_signal = cr_signal_model(drg, daz, rcs_cr, phi_cr, fdc, par, dim)

    #compute clutter corrected for CR signal and convert from sigma_0 to absolute RCS units
    clutter_signal = (slc - cr_signal)

    #full likelihood function is sum of indidual samples modeled as clutter plus cr_signal
    ll = dim[0] * dim[1] * (-np.log(np.pi * np.sqrt(2*sigm0_clut))) - np.sum(np.abs(clutter_signal) ** 2 / (2*sigm0_clut))
    cost = -ll

    return cost

#deterministic signal model for a cr
def cr_signal_model(drg, daz, rcs_cr, phi_cr, fdc, par, win_sz):
    dim = win_sz
    # compute deterministic amplitude function of the CR
    # induce a 2d shift by applying scaled phase ramp to the spectrum
    rg_ramp = -2 * np.pi * drg / dim[0] * np.outer((np.arange(dim[0]) - (dim[0] - 1) / 2), np.ones(dim[1]))
    az_ramp = -2 * np.pi * daz / dim[1] * np.outer(np.ones(dim[0]), (np.arange(dim[1]) - (dim[1] - 1) / 2))
    phase_ramp = rg_ramp + az_ramp
    phase_ramp = np.fft.ifftshift(phase_ramp)
    spect = spectrum(par, dim)
    cr_f = spect.astype('complex64') * np.exp(1j * phase_ramp)
    cr_signal = np.fft.fftshift(np.fft.ifft2(cr_f))

    cr_signal_normalization = np.sum(np.abs(np.fft.ifft2(spect))**2)
    cr_signal *= np.sqrt(rcs_cr/cr_signal_normalization/pixel_area_ground(par))
    cr_signal = cr_signal * np.exp(1j * phi_cr)

    #apply Doppler phase ramp - ref Scheiber paper
    az_ramp2 = 2*np.pi*fdc/par['prf']*np.outer(np.ones(dim[0]), ((np.arange(win_sz[1]) - (win_sz[1]-1)/2)-daz))
    cr_signal *= np.exp(1j*az_ramp2)
    return cr_signal

def chip_coh(slc_all):
    ns = slc_all.shape[0]
    dim = slc_all.shape[1:]
    cov = np.zeros((ns, ns), dtype=np.complex64)
    for ii in np.arange(dim[0]):
        for jj in np.arange(dim[1]):
            z = slc_all[:, ii, jj]
            cov += np.outer(z, z.conj())
    cov /= dim[0]*dim[1]
    sigma_diag = np.sqrt(np.abs(np.diag(cov)))
    coh = cov / (np.outer(sigma_diag, sigma_diag))
    return coh

def spectrum(par, win_sz):
    return _spectrum(par['chirp_bandwidth'], par['range_pixel_spacing'], par['azimuth_proc_bandwidth'], par['prf'], par.dim[0], par.dim[1], par['sensor'], win_sz)

def _spectrum(chirp_bandwidth, range_pixel_spacing, azimuth_proc_bandwidth, prf, dim0, dim1, sensor, win_sz):
    rg_bw_factor = chirp_bandwidth/(c/range_pixel_spacing/2.)
    az_bw_factor = azimuth_proc_bandwidth/prf

    win_rg, win_az = proc_windows(int(rg_bw_factor*dim0), int(az_bw_factor*dim1), {'sensor':sensor})

    rg_psd = np.zeros(dim0)
    az_psd = np.zeros(dim1)
    rg_npad = (len(rg_psd)-len(win_rg))//2
    az_npad = (len(az_psd)-len(win_az))//2
    rg_psd[rg_npad:rg_npad+len(win_rg)] = win_rg
    az_psd[az_npad:az_npad+len(win_az)] = win_az
    rg_psd = np.fft.ifftshift(rg_psd)
    az_psd = np.fft.ifftshift(az_psd)

    rg_acf = np.fft.ifft(rg_psd)
    az_acf = np.fft.ifft(az_psd)

    rg_acf_win = np.fft.ifftshift(np.roll(rg_acf, win_sz[0]//2)[0:win_sz[0]])
    az_acf_win = np.fft.ifftshift(np.roll(az_acf, win_sz[1] // 2)[0:win_sz[1]])

    rg_psd_win = np.abs(np.fft.fft(rg_acf_win))
    az_psd_win = np.abs(np.fft.fft(az_acf_win))
    psd_win_2d = np.outer(rg_psd_win, az_psd_win)

    if False:
    #interpolation method produces similar but slightly different results...
        rg_psd_win2 = interpolate.interp1d(np.arange(dim0), np.fft.fftshift(rg_psd))((dim0-1)/(win_sz[0]-1)*np.arange(win_sz[0]))
        az_psd_win2 = interpolate.interp1d(np.arange(dim1), np.fft.fftshift(az_psd))((dim1-1)/(win_sz[1]-1)*np.arange(win_sz[1]))
        psd_win_2d2 = np.outer(rg_psd_win2, az_psd_win2)

        plt.figure()
        plt.imshow(psd_win_2d)
        plt.figure()
        plt.plot(rg_psd_win)
        plt.plot(np.fft.ifftshift(rg_psd_win2))
        plt.figure()
        plt.plot(az_psd_win)
        plt.plot(np.fft.ifftshift(az_psd_win2))

        plt.show()

    return psd_win_2d

def proc_windows(N_rg, N_az, slc_par):
    if 'RADARSAT-2' in slc_par['sensor']:
        win_rg = np.abs(kaiser_window(N_rg, 2.4))
        win_az = np.abs(kaiser_window(N_az, 2.4))
        if 'Spotlight' in slc_par['sensor']:
            win_az = np.abs(kaiser_window(N_az, 3.5))
    elif (('TSX' in slc_par['sensor']) or ('TDX' in slc_par['sensor'])):
        win_rg = np.abs(hamming_window(N_rg, 0.6))
        win_az = np.abs(hamming_window(N_az, 0.6))
    elif ('mexicocity_S1' in slc_par['sensor']):
        win_rg = np.abs(hamming_window(N_rg, 0.75))
        win_az = np.abs(hamming_window(N_az, 0.70))
    else:
        raise ValueError('unrecognized mode')
    return win_rg, win_az

def hamming_window(N, alpha=25./46.):
    beta = (1-alpha)/2
    return alpha + 2*beta*np.cos(2*np.pi/N*np.arange(N))

def kaiser_window(N, beta):
    win = np.kaiser(N,beta) #Kaiser window for processed bandwidth
    #not sure why Kaiser generates a nan on the end..but fix it
    if(np.isnan(win[-1])):
        win[-1] = win[-2]
    return win

def process_diff(crop = False, looks = 'hr'):
    interferograms = []
    if crop:
        ave_angle = np.zeros((crop23[3], crop23[2]))
    else:
        ave_angle = np.zeros((height,width))
    for i, diff in enumerate(diffs):
        print(diff)
        diff_im = readBin(diff_dir + diff, (height,width), 'complex64')

        interferograms.append(diff.split('.')[0])

        diff_values = []
        for i in range(len(cr_loc)-1):
            if looks == 'hr':
                val = diff_im[cr_loc['y_hr'][i]][cr_loc['x_hr'][i]]
            elif looks == 'fr':
                val = diff_im[cr_loc['y_fr'][i]][cr_loc['x_fr'][i]]
            angle = np.angle(val)
            diff_values.append(angle)

        if crop:
            cropped = diff_im[crop23[1]:crop23[1]+crop23[3],crop23[0]:crop23[0]+crop23[2]]
            diff_phase = np.angle(cropped)
            diff_vert = diff_phase * (-wavelength / 4 / np.pi) / np.cos(np.radians(par['incidence_angle']))
            ave_angle += diff_phase

            diff_values.append(np.mean(diff_vert))

        else:
            diff_phase = np.angle(diff_im)
            diff_vert = diff_phase * (-wavelength / 4 / np.pi) / np.cos(np.radians(par['incidence_angle']))
            if '20140707' in diff:
                pass
            else:
                run('raspwr', '/local-scratch/users/aplourde/pngs/fr/'+ diff, width, None, None, None, None, None, None, None, 'pngs/fr/'+ diff + '.ras', None)
                run('raspwr', '/local-scratch/users/aplourde/pngs/fr/vert_' + diff, width, None, None, None, None, None, None, None, 'pngs/fr/' + diff + '.vert.ras', None)

            #writeBin('pngs/fr/'+ diff + '.ras', diff_phase)
            #writeBin('pngs/fr/vert_' + diff, diff_vert)
            ave_angle += diff_phase

            diff_values.append(np.mean(diff_vert))

        cr_loc['dv_' + diff] = diff_values

        if True:
            plt.figure()
            plt.imshow(diff_vert)
            plt.colorbar()
            plt.savefig('pngs/'+looks+'/'+diff+'.png', dpi=1200)

    plt.figure()
    ave_angle = ave_angle / len(diffs)
    plt.imshow(ave_angle)
    plt.savefig('pngs/'+looks+'/ave_angle.png', dpi=1200)
    cr_loc['dv_ave'] = cr_loc[list(cr_loc)[7:-1]].mean(axis=1)
    cr_loc['dv_std'] = cr_loc[list(cr_loc)[7:-2]].std(axis=1)
    #print(cr_loc[['cr_id', 'type', 'dv_ave', 'dv_std']])
    print(cr_loc)
    cr_loc.to_csv("cr_phase_withadf.csv")
    print(np.mean(ave_angle), np.std(ave_angle))

    var_angle = np.zeros(ave_angle.shape)
    for i,diff in enumerate(diffs):
        diff_im = readBin(diff_dir + diff, (height,width), 'complex64')
        if crop:
            cropped = diff_im[crop23[1]:crop23[1] + crop23[3], crop23[0]:crop23[0] + crop23[2]]
            diff_phase = np.angle(cropped)
        else:
            diff_phase = np.angle(diff_im)

        var_angle += (diff_phase - ave_angle)**2

    var_angle = var_angle/len(diffs)
    std_angle = np.sqrt(var_angle)
    print(np.mean(std_angle))
    plt.figure()
    plt.imshow(std_angle, cm.Greys_r)
    plt.savefig('pngs/'+looks+'/std_angle.png', dpi=1200)

    print(interferograms)

def process_ave_rmli(crop = False, looks='hr'):
    for i, rmli in enumerate(ave_rmli):
        print(rmli)
        rmli_im = readBin(rmli_dir + rmli, width, 'float')
        plt.figure()

        for i in range(len(cr_loc)):
            if looks == 'hr':
                rmli_im[cr_loc['y_hr'][i]][cr_loc['x_hr'][i]] = 0
            if looks == 'fr':
                rmli_im[cr_loc['y_fr'][i]][cr_loc['x_fr'][i]] = 0

        if crop:
            cropped = rmli_im[crop23[1]:crop23[1]+crop23[3],crop23[0]:crop23[0]+crop23[2]]
            plt.imshow(cropped, cm.Greys_r)
        else:

            plt.imshow(rmli_im, cm.Greys_r)
        plt.savefig('pngs/'+looks+'/ave_rmli.png')

def phunwrap(phi):
    #simple 1D phase unwrapper
    phi_unw = np.cumsum(np.angle(np.exp(1j*(phi[1:] - phi[0:-1]))))
    phi_unw = np.insert(phi_unw, 0, 0)
    return phi_unw

def process_rslc():

    win_sz = np.asarray((19,19))
    cr_size = 0.45 #m leg length
    ANW_loc = [cr_loc['x_'+look][0],cr_loc['y_' + look][0]]
    FSE_loc = [cr_loc['x_' + look][1], cr_loc['y_' + look][1]]
    print(ANW_loc, FSE_loc, win_sz, cr_size)
    #print(stack.rslc_list())
    ANW = cr_stack_analysis(rslcs, sims, ANW_loc, win_sz, cr_size, 'ANW')
    FSE = cr_stack_analysis(rslcs, sims, FSE_loc, win_sz, cr_size, 'FSE')

    #compute relative vertical deformation between two CRs
    par = stack.master_slc_par
    wavelength = constants.c/par['radar_frequency']
    phi_diff = FSE['phi_cr']-ANW['phi_cr']
    phi_diff -=phi_diff[0]
    phi_diff = np.angle(np.exp(1j*phi_diff))
    phi_diff_unw = phunwrap(phi_diff)
    def_vert = phi_diff_unw*(-wavelength/4/np.pi)/np.cos(np.radians(par['incidence_angle']))

    # compute relative deformation between anchored CR and floating clutter

    phi_diff_clut = FSE['phi_clut'] - ANW['phi_cr']
    phi_diff_clut -= phi_diff_clut[0]
    phi_diff_clut = np.angle(np.exp(1j * phi_diff_clut))
    phi_diff_clut_unw = phunwrap(phi_diff_clut)
    def_vert_clut = phi_diff_clut_unw * (-wavelength / 4 / np.pi) / np.cos(np.radians(par['incidence_angle']))

    # compute mean and difference offsets in rng and azimuth
    drg_mean = (FSE['drg'] + ANW['drg']) / 2
    daz_mean = (FSE['daz'] + ANW['daz']) / 2

    var_drg_mean = np.mean(((drg_mean - np.mean(drg_mean)) ** 2))
    var_daz_mean = np.mean(((daz_mean - np.mean(daz_mean)) ** 2))

    drg_diff = (FSE['drg'] - ANW['drg'])
    daz_diff = (FSE['daz'] - ANW['daz'])

    var_drg_diff = np.mean(((drg_diff - np.mean(drg_diff)) ** 2))
    var_daz_diff = np.mean(((daz_diff - np.mean(daz_diff)) ** 2))

    # compare clutter phase solutions for the two patches
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(FSE['phi_clut'])
    plt.plot(ANW['phi_clut'])
    plt.title('ITH CR pair wrapped clutter phase comparison')
    plt.xlabel('Acquisition #')
    plt.ylabel('Wrapped phase [radians]')
    plt.legend(['FSE', 'ANW'])

    plt.subplot(2, 2, 2)
    plt.plot(def_vert)
    # TODO is this really the reverse!!!!!
    #plt.plot(-def_vert_clut)
    plt.title('ITH CR pair relative vertical displacement time series.')
    plt.xlabel('Acquisition #')
    plt.ylabel('Vertical Deformation [m]')
    plt.legend(['CR def', 'Clutter def'])

    plt.subplot(2, 2, 3)
    plt.plot(drg_mean);
    plt.ylim((-1, 1))
    plt.plot(drg_diff)
    plt.title('ITH CR pair sub-pixel range offsets.')
    plt.xlabel('Acquisition #')
    plt.ylabel('Range distance [pixels]')
    plt.legend(['mean', 'difference'])

    plt.subplot(2, 2, 4)
    plt.plot(daz_mean);
    plt.ylim((-1, 1))
    plt.plot(daz_diff)
    plt.title('ITH CR pair sub-pixel azimuth offsets.')
    plt.xlabel('Acquisition #')
    plt.ylabel('Azimuth distance [pixels]')
    plt.legend(['mean', 'difference'])

    print('stddev_drg_mean, stddev_daz_mean:', np.sqrt(var_drg_mean / 2), np.sqrt(var_daz_mean / 2))
    print('stddev_drg_diff, stddev_daz_diff:', np.sqrt(var_drg_diff / 2), np.sqrt(var_daz_diff / 2))
    plt.savefig('results.png')



if __name__ == "__main__":

    # read in images with crop
    process_diff(crop=False,looks=look)
    #process_ave_rmli(crop=True, looks=look)
    #process_rslc()

    # display

    # make average