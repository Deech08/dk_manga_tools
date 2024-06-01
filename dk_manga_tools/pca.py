# PCA functionality from Zach Pace

import logging
import warnings
import astropy.units as u
import numpy as np
from astropy.io import fits

import os

from speclite import filters
from astropy.cosmology import WMAP9
from astropy.table import Table


directory = os.path.dirname(__file__)

# pca_data_dir_old= os.path.join(os.environ['SAS_BASE_DIR'], 'dr17', 'manga', 'spectro', 'mangapca', 'CSPs_CKC14_MaNGA_20190215-1')
pca_dr17_dir = os.path.join(os.environ['SAS_BASE_DIR'], 'dr17', 'manga', 'spectro', 'mangapca', '1.1.0')
spec_unit = 1e-17 * u.erg / u.s / u.cm**2. / u.AA
absmag_sun_band = {'u': 6.39, 'g': 5.12, 'r': 4.64, 'i': 4.53, 'z': 4.51, 'V': 4.81}


def PCA_eigendata(vec_file = None, pca_data_dir = None):
    """
    Get eigenvector data

    Parameters
    ----------
    vec_file: 'str'
        filename of eigenvector data
    """
    if vec_file is None:
        if pca_data_dir is None:
            pca_data_dir = pca_dr17_dir
        vec_file = os.path.join(pca_data_dir, "pc_vecs-1.1.0.fits")

    # Load vectors:
    with fits.open(vec_file) as vec_data:
        mean_spec = vec_data["MEAN"].data 
        evec_spec = vec_data["EVECS"].data
        lam_spec = vec_data["LAM"].data
    return mean_spec, evec_spec, lam_spec


def PCA_Spectrum(plateifu = None, filename = None, vec_file = None, 
                vec_data = None, pca_data_dir = None):
    """
    Construct PCA spectrum from eigenvectors for specified galaxy

    Parameters
    ----------
    plateifu: 'str', list, optional, must be keyword
        plate-ifu of galaxy desired
    filename: 'str', optional, must be keyword
        pca data file to read in, ignores plateifu if provided
    vec_file: 'str', optional, must be keyword
        pca data file containing eigenvectors
    vec_data: 'tuple', optional, must be keyword
        eigenvector data (mean_spec, evec_spec, lam_spec)
    """
    

    if vec_data is None:
        if pca_data_dir is None:
            pca_data_dir = pca_dr17_dir

        mean_spec, evec_spec, lam_spec = PCA_eigendata(vec_file = vec_file, pca_data_dir = pca_data_dir)
    else:
        mean_spec, evec_spec, lam_spec = vec_data

    if not isinstance(lam_spec, u.Quantity):
        lam_spec *= u.AA


    if filename is None:
        if plateifu is None:
            raise ValueError("No input file or plateifu provided")
        else:
            filename = os.path.join(pca_data_dir, plateifu, "{}_res.fits".format(plateifu))

    # Load PCA Data for Galaxy
    with fits.open(filename) as pca_data:
        pca_norm = pca_data["NORM"].data
        pca_calpha = pca_data["CALPHA"].data


    spectrum = pca_norm * (np.einsum('al,a...->l...', evec_spec, pca_calpha) + mean_spec[:,None,None]) * spec_unit

    return spectrum, lam_spec

def PCA_mag(filter_obs, dapall = None, maps = None, plateifu = None, filename = None, vec_file = None, 
                vec_data = None, pca_data_dir = None):
    """
    Return absolute AB Magnitude in filter provided

    Parameters
    ----------
    dapall: 'Table', 'dict'
        DAPALL file data
    filter_obs: 'str', 'speclite.filters.FilterSequence'
        observational filter to use
    plateifu: 'str', list, optional, must be keyword
        plate-ifu of galaxy desired
    filename: 'str', optional, must be keyword
        pca data file to read in, ignores plateifu if provided
    vec_file: 'str', optional, must be keyword
        pca data file containing eigenvectors
    vec_data: 'tuple', optional, must be keyword
        eigenvector data (mean_spec, evec_spec, lam_spec)
    """

    # Check filter status

    if pca_data_dir is None:
        pca_data_dir = pca_dr17_dir

    # CHECK PLATEIFU
    if plateifu is None:
        if maps is None:
            plateifu = dapall["plateifu"]
        else:
            plateifu = maps.plateifu

    if filter_obs.__class__ is not filters.FilterSequence:
        if filter_obs in ["GALEX-NUV", "GALEX-FUV"]:
            wav, resp = np.loadtxt("{}/data/GALEX_GALEX.NUV.dat".format(directory)).T
            galex_nuv = filters.FilterResponse(
                wavelength = wav * u.Angstrom,
                response = resp, meta=dict(group_name='GALEX', band_name='NUV'))

            wav, resp = np.loadtxt("{}/data/GALEX_GALEX.FUV.dat".format(directory)).T
            galex_fuv = filters.FilterResponse(
                wavelength = wav * u.Angstrom,
                response = resp, meta=dict(group_name='GALEX', band_name='FUV'))
        try:
            filter_obs = filters.load_filters(filter_obs)
        except ValueError:
            logging.warnings("Invalid filter, using default of 'sdss2010-i'")
            filter_obs = filters.load_filters("sdss2010-i")

    if filename is None:
        if plateifu is None:
            raise ValueError("No input file or plateifu provided")
        else:
            filename = os.path.join(pca_data_dir, "v3_1_1", "3.1.0", plateifu.split("-")[0], "mangapca-{}.fits".format(plateifu))

    spectrum, wlen = PCA_Spectrum(plateifu = plateifu, filename = filename, 
                                  vec_file = vec_file, vec_data = vec_data, pca_data_dir = pca_data_dir)


    mag = filter_obs.get_ab_magnitudes(spectrum, wlen, axis = 0)[filter_obs.names[0]].data * u.ABmag
    if maps is None:
        mag_abs = mag - WMAP9.distmod(dapall["nsa_zdist"])
    else:
        mag_abs = mag - WMAP9.distmod(maps.nsa["zdist"])

    return mag_abs

def PCA_stellar_mass(maps = None, dapall=None, plateifu = None, filename = None, vec_file = None, 
                vec_data = None, pca_data_dir = None, goodfrac_channel = 2, 
                goodfrac_thresh = .0001, use_mask = True):
    """
    Return absolute AB Magnitude in filter provided

    Parameters
    ----------
    dapall: 'Table', 'dict'
        DAPALL file data
    plateifu: 'str', list, optional, must be keyword
        plate-ifu of galaxy desired
    filename: 'str', optional, must be keyword
        pca data file to read in, ignores plateifu if provided
    vec_file: 'str', optional, must be keyword
        pca data file containing eigenvectors
    vec_data: 'tuple', optional, must be keyword
        eigenvector data (mean_spec, evec_spec, lam_spec)
    """
    if pca_data_dir is None:
        pca_data_dir = pca_dr17_dir
    if filename is None:
        if plateifu is None:
            if maps is None:
                plateifu = dapall["plateifu"]
            else:
                plateifu = maps.plateifu
        if pca_data_dir is None:
            pca_data_dir = pca_dr17_dir
        filename = os.path.join(pca_data_dir, "v3_1_1", "3.1.0", plateifu.split("-")[0], "mangapca-{}.fits".format(plateifu))


    filter_obs = filters.load_filters("sdss2010-i")
    i_mag_abs = PCA_mag(filter_obs, maps = maps, dapall = dapall, plateifu = plateifu, filename = filename, 
                        vec_file = vec_file, vec_data = vec_data, pca_data_dir = pca_data_dir)

    sun_i = absmag_sun_band["i"] * u.ABmag
    i_sol_lum = 10**(-0.4 * (i_mag_abs - sun_i).value)

    with fits.open(filename) as pca_data:
        MLi = pca_data["MLi"].data
        mask = pca_data["MASK"].data.astype(bool)
        goodfrac = pca_data["GOODFRAC"].data[goodfrac_channel]
    m_star = i_sol_lum * 10**MLi
    mask = mask | (goodfrac < goodfrac_thresh )

    mask_shaped = np.zeros_like(m_star, dtype = bool)
    if mask_shaped.shape[0] == 3:
        mask_shaped[0,:,:] = mask
        mask_shaped[1,:,:] = mask
        mask_shaped[1,:,:] = mask
    else:
        mask_shaped = mask

    m_star_masked = np.ma.masked_array(m_star * u.solMass, mask = mask_shaped)

    return m_star_masked



def PCA_MLi(maps = None, dapall=None, plateifu = None, filename = None, pca_data_dir = None):
    """
    Return absolute Mass to Light Ratio in i-band

    Parameters
    ----------
    dapall: 'Table', 'dict'
        DAPALL file data
    plateifu: 'str', list, optional, must be keyword
        plate-ifu of galaxy desired
    filename: 'str', optional, must be keyword
        pca data file to read in, ignores plateifu if provided
    vec_file: 'str', optional, must be keyword
        pca data file containing eigenvectors
    vec_data: 'tuple', optional, must be keyword
        eigenvector data (mean_spec, evec_spec, lam_spec)
    """
    if pca_data_dir is None:
        pca_data_dir = pca_dr17_dir
    if filename is None:
        if plateifu is None:
            if maps is None:
                plateifu = dapall["plateifu"]
            else:
                plateifu = maps.plateifu
        else:
            filename = os.path.join(pca_data_dir, "v3_1_1", "3.1.0", plateifu.split("-")[0], "mangapca-{}.fits".format(plateifu))

    # Load PCA Data for Galaxy
    if filename is None:
        filename = os.path.join(pca_data_dir, "v3_1_1", "3.1.0", plateifu.split("-")[0], "mangapca-{}.fits".format(plateifu))
    with fits.open(filename) as pca_data:
        MLi = pca_data["MLi"].data

    return MLi

def PCA_info(maps, name, dapall = None, plateifu = None, filename = None, pca_data_dir = None, 
    masked = True, goodfrac_channel = 2, goodfrac_thresh = 0.0001):
    """
    Return specific PCA CSP based info

    Parameters
    ----------
    name: `string`
        name of info to get from PCA data, 
        see https://data.sdss.org/datamodel/files/MANGA_PCA/PCAY_VER/CSPs/CSPs.html
    """

    if pca_data_dir is None:
        pca_data_dir = pca_dr17_dir

    if filename is None:
        if plateifu is None:
            if maps is None:
                plateifu = dapall["plateifu"]
            else:
                plateifu = maps.plateifu
        else:
            filename = os.path.join(pca_data_dir, "v3_1_1", "3.1.0", plateifu.split("-")[0], "mangapca-{}.fits".format(plateifu))

    # get main info
    with fits.open(filename) as pca_data:
        header = pca_data[0].header
        mask = pca_data["MASK"].data 
        mask_header = pca_data["MASK"].header  
        good_frac = pca_data["GOODFRAC"].data
        modelnum = pca_data["MODELNUM"].data

        galmask = mask == 0
        galmask &= good_frac[goodfrac_channel] > goodfrac_thresh

        if maps is None:
            maps = DKMaps(plateifu)

        if not hasattr(maps, "pca_csp"):
            maps.load_PCA_CSP(pca_data_dir = pca_data_dir)

        info = np.array([maps.pca_csp[name][ell] for ell in modelnum])

        if masked:
            return np.ma.masked_array(data = info, mask = ~galmask)
        else:
            return info


def PCA_zpres_info(name, dapall=None, maps = None, plateifu = None, filename = None, pca_data_dir = None, 
    masked = True, goodfrac_channel = 2, 
                goodfrac_thresh = .0001):
    """
    Return absolute Mass to Light Ratio in i-band

    Parameters
    ----------
    dapall: 'Table', 'dict'
        DAPALL file data
    name: `string` or `int`
        name or index of info to get
        see Data model 
        https://data.sdss.org/sas/mangawork/manga/sandbox/mangapca/zachpace/mangapca.html
    plateifu: 'str', list, optional, must be keyword
        plate-ifu of galaxy desired
    filename: 'str', optional, must be keyword
        pca data file to read in, ignores plateifu if provided
    """
    if filename is None:
        if plateifu is None:
            if maps is None:
                plateifu = dapall["plateifu"]
            else:
                plateifu = maps.plateifu
        else:
            filename = os.path.join(pca_data_dir, plateifu, "{}_zpres.fits".format(plateifu))

    # Load PCA Data for Galaxy
    if filename is None:
        filename = os.path.join(pca_data_dir, plateifu, "{}_zpres.fits".format(plateifu))
    with fits.open(filename) as pca_data:
        try:
            info = pca_data[name].data
            if masked:
                mask = pca_data["MASK"].data.astype(bool)
                goodfrac = pca_data["GOODFRAC"].data[goodfrac_channel]
        except KeyError:
            info = np.full(pca_data[1].data.shape, np.nan)

            if masked:
                return np.ma.masked_array(info, mask = np.ones_like(info, dtype = bool))
            else:
                return info

    if masked:
        mask = mask | (goodfrac < goodfrac_thresh )
        mask_shaped = np.zeros_like(info, dtype = bool)
        if mask_shaped.shape[0] == 3:
            mask_shaped[0,:,:] = mask
            mask_shaped[1,:,:] = mask
            mask_shaped[1,:,:] = mask
        else:
            mask_shaped = mask
        info = np.ma.masked_array(info, mask = mask_shaped)

    return info






def load_CSP_data(pca_data_dir = None):
    if pca_data_dir is None:
        pca_data_dir = pca_dr17_dir

    A_csps = []
    d1 = []
    dtb = []
    fbhb = []
    gamma = []
    logzsol = []
    nburst = []
    sbss = []
    tb = []
    tf = []
    theta = []
    tt = []
    f20 = []
    f50 = []
    f100 = []
    f200 = []
    f500 = []
    f1G = []
    mwa = []
    mstar = []
    sigma = []
    Na_D = []
    Hdelta_A = []
    Hgamma_A = []
    Dn4000 = []
    D4000 = []
    Ca_HK = []
    MLr = []
    MLi = []
    MLz = []
    MLV = []
    Cgr = []
    Cri = []
    Cgi = []
    logQHpersolmass = []

    for i in range(0,40):
        with fits.open(os.path.join(pca_data_dir, "CSPs", "CSPs_{}.fits".format(i))) as csps:
            A_csps.append(csps[1].data['A'].T)
            d1.append(csps[1].data['d1'])
            dtb.append(csps[1].data['dtb'].T)
            fbhb.append(csps[1].data['fbhb'])
            gamma.append(csps[1].data['gamma'])
            logzsol.append(csps[1].data['logzsol'])
            nburst.append(csps[1].data['nburst'])
            sbss.append(csps[1].data['sbss'])
            tb.append(csps[1].data['tb'].T)
            tf.append(csps[1].data['tf'])
            theta.append(csps[1].data['theta'])
            tt.append(csps[1].data['tt'])
            f20.append(csps[1].data['F_20M'])
            f50.append(csps[1].data['F_50M'])
            f100.append(csps[1].data['F_100M'])
            f200.append(csps[1].data['F_200M'])
            f500.append(csps[1].data['F_500M'])
            f1G.append(csps[1].data['F_1G'])
            mwa.append(csps[1].data['MWA'])
            mstar.append(csps[1].data['mstar'])
            sigma.append(csps[1].data['sigma'])
            Na_D.append(csps[1].data['Na_D'])
            Hdelta_A.append(csps[1].data['Hdelta_A'])
            Hgamma_A.append(csps[1].data['Hgamma_A'])
            Dn4000.append(csps[1].data['Dn4000'])
            D4000.append(csps[1].data['D4000'])
            Ca_HK.append(csps[1].data['Ca_HK'])
            MLr.append(csps[1].data['MLr'])
            MLi.append(csps[1].data['MLi'])
            MLz.append(csps[1].data['MLz'])
            MLV.append(csps[1].data['MLV'])
            Cgr.append(csps[1].data['Cgr'])
            Cri.append(csps[1].data['Cri'])
            Cgi.append(csps[1].data['Cgi'])
            logQHpersolmass.append(csps[1].data['logQHpersolmass'])

    A_csps = np.hstack(A_csps)
    d1 = np.hstack(d1)
    dtb = np.hstack(dtb)
    fbhb = np.hstack(fbhb)
    gamma = np.hstack(gamma)
    logzsol = np.hstack(logzsol)
    nburst = np.hstack(nburst)
    sbss = np.hstack(sbss)
    tb = np.hstack(tb)
    tf = np.hstack(tf)
    theta = np.hstack(theta)
    tt = np.hstack(tt)
    f20 = np.hstack(f20)
    f50 = np.hstack(f50)
    f100 = np.hstack(f100)
    f200 = np.hstack(f200)
    f500 = np.hstack(f500)
    f1G = np.hstack(f1G)
    mwa = np.hstack(mwa)
    mstar = np.hstack(mstar)
    sigma = np.hstack(sigma)
    Na_D = np.hstack(Na_D)
    Hdelta_A = np.hstack(Hdelta_A)
    Hgamma_A = np.hstack(Hgamma_A)
    Dn4000 = np.hstack(Dn4000)
    D4000 = np.hstack(D4000)
    Ca_HK = np.hstack(Ca_HK)
    MLr = np.hstack(MLr)
    MLi = np.hstack(MLi)
    MLz = np.hstack(MLz)
    MLV = np.hstack(MLV)
    Cgr = np.hstack(Cgr)
    Cri = np.hstack(Cri)
    Cgi = np.hstack(Cgi)
    logQHpersolmass = np.hstack(logQHpersolmass)


    return Table({
        "A":A_csps.T, 
        "d1":d1,
        "dtb":dtb.T,
        "fbhb":fbhb,
        "gamma":gamma,
        "logzsol":logzsol,
        "nburst":nburst,
        "sbss":sbss,
        "tb":tb.T,
        "tf":tf,
        "theta":theta,
        "tt":tt,
        "F_20M":f20,
        "F_50M":f50,
        "F_100M":f100,
        "F_200M":f200,
        "F_500M":f500,
        "F_1G":f1G,
        "MWA":mwa,
        "mstar":mstar,
        "sigma":sigma,
        "Na_D":Na_D,
        "Hdelta_A":Hdelta_A,
        "Hgamma_A":Hgamma_A,
        "Dn4000":Dn4000,
        "Ca_HK":Ca_HK,
        "MLr":MLr,
        "MLi":MLi,
        "MLz":MLz,
        "MLiV":MLV,
        "Cgr":Cgr,
        "Cri":Cri,
        "Cgi":Cgi,
        "logQHpersolmass":logQHpersolmass
        })






