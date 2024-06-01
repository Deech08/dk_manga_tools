# PCA functionality from Zach Pace

import logging
import warnings
import astropy.units as u
import numpy as np
from astropy.io import fits

import os

from speclite import filters
from astropy.cosmology import WMAP9


directory = os.path.dirname(__file__)

pca_vec_data_dir_old= os.path.join(os.environ['SAS_BASE_DIR'], 'dr17', 'manga', 'spectro', 'mangapca', 'CSPs_CKC14_MaNGA_20190215-1')
pca_dr17_dir = os.path.join(os.environ['SAS_BASE_DIR'], 'dr17', 'manga', 'spectro', 'mangapca', '1.1.0')
spec_unit = 1e-17 * u.erg / u.s / u.cm**2. / u.AA
absmag_sun_band = {'u': 6.39, 'g': 5.12, 'r': 4.64, 'i': 4.53, 'z': 4.51, 'V': 4.81}


def PCA_eigendata(vec_file = None, pca_vec_data_dir = None):
    """
    Get eigenvector data

    Parameters
    ----------
    vec_file: 'str'
        filename of eigenvector data
    """
    if vec_file is None:
        if pca_vec_data_dir is None:
            pca_vec_data_dir = pca_dr17_dir
        vec_file = os.path.join(pca_vec_data_dir, "pc_vecs-1.1.0.fits")

    # Load vectors:
    with fits.open(vec_file) as vec_data:
        mean_spec = vec_data["MEAN"].data 
        evec_spec = vec_data["EVECS"].data
        lam_spec = vec_data["LAM"].data
    return mean_spec, evec_spec, lam_spec


def PCA_Spectrum(plateifu = None, filename = None, vec_file = None, 
                vec_data = None, pca_vec_data_dir = None, pca_data_dir = None):
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
        if pca_vec_data_dir is None:
            pca_vec_data_dir = pca_dr17_dir

        mean_spec, evec_spec, lam_spec = PCA_eigendata(vec_file = vec_file, pca_vec_data_dir = pca_vec_data_dir)
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
            filename = os.path.join(pca_data_dir, plateifu, "{}_res.fits".format(plateifu))

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
    if filename is None:
        if plateifu is None:
            if maps is None:
                plateifu = dapall["plateifu"]
            else:
                plateifu = maps.plateifu
        else:
            filename = os.path.join(pca_data_dir, plateifu, "{}_res.fits".format(plateifu))

    # Load PCA Data for Galaxy
    if filename is None:
        filename = os.path.join(pca_data_dir, plateifu, "{}_res.fits".format(plateifu))
    with fits.open(filename) as pca_data:
        MLi = pca_data["MLi"].data

    return MLi

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















