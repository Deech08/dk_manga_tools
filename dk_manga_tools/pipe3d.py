# PCA functionality from Zach Pace

import logging
import warnings
import astropy.units as u
import numpy as np
from astropy.table import Table
from astropy.io import fits

import os

from speclite import filters
from astropy.cosmology import WMAP9


directory = os.path.dirname(__file__)

pipe3d_dir_master = os.path.join(os.environ['SAS_BASE_DIR'], 'dr17', 'manga', 'spectro', 'pipe3d', 'v3_1_1', '3.1.1')


def load_PIPE3D_SSP(plateifu, pipe3d_dir = None):
    """
    Get full PIPE3D data table from given plateifu
    """
    if pipe3d_dir is None:
        pipe3d_dir = pipe3d_dir_master
    plateifu = str(plateifu)
    plate,_ = plateifu.split("-")
    filename = os.path.join(pipe3d_dir, plate, "manga-{}.Pipe3D.cube.fits.gz".format(plateifu))
    with fits.open(filename) as hdulist:
        header = hdulist[1].header
        data = hdulist[1].data

    ind = np.arange(header["NAXIS3"])
    desc = []
    type_info []
    unit = []
    for ell in ind:
        desc.append(header["DESC_{}".format(ell)])
        unit.append(header["UNITS_{}".format(ell)])
        type_info.append(header["TYPE_{}".format(ell)])


    ssp_info_table = Table({
        "IND":ind,
        "KEY":[
            "V", 
            "CONT_SEG",
            "CONT_DEZON",
            "MED_INT_12772-12705",
            "STD_INT_12772-12705",
            "LOG10_LW_AGE",
            "LOG10_MW_AGE",
            "ERR_LOG10_AGE",
            "LOG10_LW_Z",
            "LOG10_MW_Z",
            "ERR_LOG10_Z",
            "MEAN_DUST_ATTN",
            "ERR_MEAN_DUST_ATTN",
            "VEL_STELLAR",
            "ERR_VEL_STELLAR",
            "VEL_DISP_STELLAR",
            "ERR_VEL_DISP_STELLAR",
            "MEAN_ML_STELLAR",
            "LOG10_MASS_DENS_STELLAR",
            "LOG10_MASS_DENS_STELLAR_DC",
            "ERR_LOG10_MASS_DENS_STELLAR"
        ],
        "DESCRIPTION":desc,
        "TYPE":type_info,
        "UNIT":unit,
        })

    ssp_table = Table(names = ssp_info_table["KEY"], 
                      data = [data[ell,:,:] for ell in ind])

    return ssp_info_table, ssp_table
    









