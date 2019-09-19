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

pca_vec_data_dir_master = os.path.join(os.environ['SAS_BASE_DIR'], 'mangawork', 'manga', 'sandbox', 'mangapca', 'zachpace', 'CSPs_CKC14_MaNGA_20190215-1')

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
            pca_vec_data_dir = pca_vec_data_dir_master
        vec_file = os.path.join(pca_vec_data_dir, "pc_vecs.fits")

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
            pca_vec_data_dir = pca_vec_data_dir_master

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

def PCA_mag(dapall, filter_obs, plateifu = None, filename = None, vec_file = None, 
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
    if filter_obs.__class__ is not filters.FilterSequence:
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
    mag_abs = mag - WMAP9.distmod(dapall["nsa_zdist"])

    return mag_abs

def PCA_stellar_mass(dapall, plateifu = None, filename = None, vec_file = None, 
                vec_data = None, pca_data_dir = None):
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
            plateifu = dapall["plateifu"]
        else:
            filename = os.path.join(pca_data_dir, plateifu, "{}_res.fits".format(plateifu))


    filter_obs = filters.load_filters("sdss2010-i")
    i_mag_abs = PCA_mag(dapall, filter_obs, plateifu = plateifu, filename = filename, 
                        vec_file = vec_file, vec_data = vec_data, pca_data_dir = pca_data_dir)

    sun_i = absmag_sun_band["i"] * u.ABmag
    i_sol_lum = 10**(-0.4 * (i_mag_abs - sun_i).value)

    # Load PCA Data for Galaxy
    if filename is None:
        filename = os.path.join(pca_data_dir, plateifu, "{}_res.fits".format(plateifu))
    with fits.open(filename) as pca_data:
        MLi = pca_data["MLi"].data

    m_star = i_sol_lum * 10**MLi
    return m_star * u.solMass



def PCA_MLi(dapall, plateifu = None, filename = None, pca_data_dir = None):
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
            plateifu = dapall["plateifu"]
        else:
            filename = os.path.join(pca_data_dir, plateifu, "{}_res.fits".format(plateifu))

    # Load PCA Data for Galaxy
    if filename is None:
        filename = os.path.join(pca_data_dir, plateifu, "{}_res.fits".format(plateifu))
    with fits.open(filename) as pca_data:
        MLi = pca_data["MLi"].data

    return MLi
















