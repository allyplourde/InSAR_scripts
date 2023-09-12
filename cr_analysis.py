#this module contains code to model a triangular corner reflector embeded in clutter from SLC data

from sarlab.gammax import *
from jinsar.ds_modelling import spectrum, Coh_Model_3param
import scipy.optimize as opt
from scipy import constants
import numpy as np
from numba import jit


ftol = 1e-12
gtol = 1e-12
minimize_options = {'ftol': ftol, 'gtol': gtol}

#model the temporal history of the tcr in clutter through a stack of SLC images
def cr_stack_analysis(slcnames, sim_unw_names, cr_loc, win_sz, cr_size, label):
    ns = len(slcnames)
    chip_corner = cr_loc - (win_sz - 1) / 2
    slcs = np.zeros((ns, win_sz[0], win_sz[1]), dtype='complex128')
    res=[]
    sim_unw_cr = np.zeros(ns)
    slcs_clutter = np.zeros((ns, win_sz[0], win_sz[1]), dtype='complex128')
    for ii in np.arange(ns):
        print(ii+1, ' of ', ns)
        par = SLC_Par(slcnames[ii]+'.par')
        slcs[ii,:,:] = readBin(slcnames[ii],par.dim,'complex64', crop=(chip_corner[0], chip_corner[1], win_sz[0], win_sz[1]))
        if sim_unw_names[ii] is not None:
            sim_unw_ii = readBin(sim_unw_names[ii],par.dim,'float32', crop=(chip_corner[0], chip_corner[1], win_sz[0], win_sz[1]))
        else:
            sim_unw_ii = np.zeros(win_sz, dtype='float')

        res_ii, slcs_clutter[ii,:,:] = fit_cr_model(slcs[ii, :, :], par, cr_loc)
        res.append(res_ii)

        # topo correction
        slcs_clutter[ii,:,:] *= np.exp(1j * sim_unw_ii)
        #interpolate the sim phase at the subpixel location
        spline = scipy.interpolate.RectBivariateSpline(np.arange(win_sz[0])-(win_sz[0]-1)/2, np.arange(win_sz[1])-(win_sz[1]-1)/2, sim_unw_ii)
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
    coh_model = Coh_Model_3param(t_vec, win_sz[0]*win_sz[1])
    pta_fit = coh_model.fit(clutter_coh, cost='mle', phase_init='lf')
    ret['phi_clut'] = pta_fit.theta

    if True:
        plt.figure()
        plt.subplot(2,3,1)
        plt.plot(ret['drg'], marker='o'); plt.ylim((-1,1))
        plt.plot(ret['daz'], marker='o'); plt.ylim((-1,1))
        plt.title(label + ' sub-pixel 2D offset')
        plt.xlabel('Acquisition #')
        plt.ylabel('Offset distance [pixels]')
        plt.legend(['range','azimuth' ])

        plt.subplot(2,3,2)
        plt.plot(10 * np.log10(rcs_cs_theor), marker='o')
        plt.plot(10 * np.log10(ret['rcs_cr']), marker='o')
        plt.plot(10 * np.log10(ret['sigm0_clut']), marker='o'); plt.ylim((-20,30))
        rcs_clut = ret['sigm0_clut']*pixel_area_ground(par)
        plt.plot(10 * np.log10(ret['rcs_cr']/rcs_clut), marker='o')
        plt.title(label + ' Cross-sections')
        plt.xlabel('Acquisition #')
        plt.ylabel('Cross-section [dB m^2]')
        plt.legend(['CR theoretical','CR estimated', 'Clutter [sigma0]', 'CR Signal-to-clutter ratio' ])

        plt.subplot(2,3,3)
        plt.plot(ret['phi_cr'], marker='o');plt.ylim((-np.pi, np.pi))
        plt.plot(ret['phi_clut'], marker='o');
        plt.ylim((-np.pi, np.pi))
        plt.title(label + ' CR and Clutter Phases')
        plt.xlabel('Acquisition #')
        plt.ylabel('Phase [radians]')
        plt.legend(['CR phase', 'Clutter phase'])

        plt.subplot(2,3,4)
        plt.imshow(np.abs(clutter_coh), vmin=0, vmax=1)
        plt.title(label + ' Clutter absolute coherence')

        plt.subplot(2,3,5)
        plt.imshow(np.angle(clutter_coh), vmin=-np.pi, vmax=np.pi)
        plt.title(label + ' Clutter phase')

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

def pixel_area_ground(par):
    A = par['range_pixel_spacing'] * par['azimuth_pixel_spacing']/np.sin(np.radians(par['incidence_angle']))
    return A

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


def cr_orientation(slc_par_fname, dem_par_fname, declination, lon, lat):
    # all input and displayed units are in degrees
    # east declinations are positive (west are negative)
    par = SLC_Par(filename=slc_par_fname)

    def format_azimuth_angle(az_angle):
        return ((az_angle - 180.) % 360.) - 180.

    cr_azimuth_geo = format_azimuth_angle(par['heading'] + 270.)
    cr_azimuth_mag = format_azimuth_angle(cr_azimuth_geo - declination)

    hgt = 100.  # m
    nrg = 132 * 12.
    re = par['earth_radius_below_sensor'] + hgt
    sec = par['sar_to_earth_center']
    rg = par['near_range_slc'] + nrg * par['range_pixel_spacing']
    cr_inc_angle = np.degrees(np.pi - np.arccos((rg ** 2 + re ** 2 - sec ** 2) / (2 * rg * re)))

    # calc. angle between CR boresight and bottom plate
    # p1 = np.asarray((1,0,0))
    # p2 = np.asarray((0,1,0))
    # p3 = np.asarray((0,0,1))
    # p_bore = (p1 + p2 + p3)
    # p_bottom_plate = (p1 + p2)
    # del_elev = np.degrees(np.arccos(np.dot(p_bore,p_bottom_plate)/np.linalg.norm(p_bore)/np.linalg.norm(np.linalg.norm(p_bottom_plate))))

    del_elev = np.degrees(np.arctan(2 ** -0.5))

    bore_elev = 90. - cr_inc_angle
    base_plate_elev = bore_elev - del_elev
    base_plate_dip = -base_plate_elev

    print('CR Azimuth Orientation:')
    print('Satellite heading (geo, degrees):', par['heading'])
    print('Local magnetic declination (east, degrees):', declination)
    print('CR azimuth angle (geo, degrees):', cr_azimuth_geo)
    print('CR azimuth angle (mag, degrees):', cr_azimuth_mag)
    print('CR long edge angle (mag, degrees):', cr_azimuth_mag - 90.)

    print('CR Elevation:')
    print('Local incidence angle:', cr_inc_angle)
    print('Elevation angle between CR boresight and bottom plate (degrees):', del_elev)
    print('Bottom plate dip angle (degrees, +ve angle means plate sloping down):', base_plate_dip)


@jit(nopython=True)
def tcr_cross_section(side_len, lamb, elev_angle, az_angle):
    # compute tcr cross-section (sigm) according to Bonkowski et al 1953 ...
    # elev angle is relative to horizontal, az_angle relative to boresight
    # angles in radians
    # unit vectors for the three tcr edges
    e1 = np.asarray((1., 0., 0.))
    e2 = np.asarray((0., 1., 0.))
    e3 = np.asarray((0., 0., 1.))

    # unit look vector
    lx = np.cos(np.abs(az_angle) + np.pi / 4)
    ly = np.sin(np.abs(az_angle) + np.pi / 4)
    lz = np.tan(elev_angle)
    lv = np.asarray((lx, ly, lz))
    lv = lv / np.linalg.norm(lv)

    b = side_len
    l_m_n = np.sort(np.asarray((np.dot(e1, lv), np.dot(e2, lv), np.dot(e3, lv))))
    l = l_m_n[0];
    m = l_m_n[1];
    n = l_m_n[2]
    if l + m <= n:
        A = 4 * l * m * b ** 2 / (l + m + n)
    else:
        A = (l + m + n - 2 / (l + m + n)) * b ** 2
    sigm = 4 * np.pi / lamb ** 2 * A ** 2
    return sigm