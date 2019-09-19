# Core functionality

import logging

from marvin import config
from astropy.io import fits
import astropy.units as u
from astropy.cosmology import WMAP9

import numpy as np

from extinction import fm07 as extinction_law

from marvin.tools.image import Image
from marvin.tools.maps import Maps
from marvin.tools.cube import Cube

from .DKAnalogMixin import DKAnalogMixin

import os

from astropy.table import Table

class DK_MWAnalogs(DKAnalogMixin):
    """
    Core DAP and DRP Data product holder

    Parameters
    ----------
    filename_drp: 'str', optional, must be keyword
        filename of DRP file
    filename_dap: 'str', optional, must be keyword
        filename of DAP file
    drpver: 'str', optional, must be keyword
        DRP Version to load
    dapver: 'str', optional, must be keyword
        DAP Version to load
    filename_targets: 'str', optional, must be keyword
        MaNGA Target list file
    filename_gz: 'str', optional, must be keyword
        Galaxy Zoo Morphological Classifications file
    sersic: 'bool', optional must be keyword
        if True, uses Sersic Mass

    """

    def __init__(self, filename_drp = None, filename_dap = None, 
                 drpver = None, dapver = None, 
                 filename_targets = None, filename_gz = None, sersic = False,
                 **kwargs):
        # Get or set filenames for DRP all file
        if filename_drp is None:
            if drpver is None:
                self.drpver, _ = config.lookUpVersions()
                logging.warning("Using DRP Version: {}".format(self.drpver))
            else:
                self.drpver = drpver
            self.filename_drp = os.path.join(os.environ['SAS_BASE_DIR'], 'mangawork', 'manga', 'spectro', 'redux',
                           self.drpver, 'drpall-{}.fits'.format(self.drpver))
        else:
            self.filename_drp  = filename_drp

        # Get or set filenames for DAP all file
        if filename_dap is None:
            if dapver is None:
                _, self.dapver = config.lookUpVersions()
                logging.warning("Using DAP Version: {}".format(self.dapver))
            else:
                self.dapver = dapver
            self.filename_dap = os.path.join(os.environ['SAS_BASE_DIR'], 'mangawork', 'manga', 'spectro', 'analysis',
                           self.drpver, self.dapver, 'dapall-{0}-{1}.fits'.format(self.drpver,self.dapver))
        else:
            self.filename_dap = filename_dap

        # Get or set filename for Target File List
        if filename_targets is None:
            self.filename_targets = os.path.join(os.environ['SAS_BASE_DIR'], 'mangawork', 'manga', 'target', 
                            'v1_2_27', 'MaNGA_targets_extNSA_tiled_ancillary.fits')
        else:
            self.filename_targets = filename_targets

        # Get or set filename for Galaxy Zoo VAC
        if filename_gz is None:
            self.filename_gz = os.path.join(os.environ['SAS_BASE_DIR'], 'dr15', 'manga', 'morphology', 
                            'galaxyzoo', 'MaNGA_gz-v1_0_1.fits')
        else:
            self.filename_gz = filename_gz


        # Load Data
        try:
            self.drp = Table.read(self.filename_drp)
        except FileNotFoundError:
            logging.warning("DRP File not found")
            self.drp = Table()
        try:
            self.dap = Table.read(self.filename_dap)
        except FileNotFoundError:
            logging.warning("DAP File not found")
            self.dap = Table()

        try:
            self.targets = Table.read(self.filename_targets)
        except FileNotFoundError:
            logging.warning("Target Data File not found")
            self.targets = Table()

        try:
            self.gz = Table.read(self.filename_gz)
        except FileNotFoundError:
            logging.warning("Galaxy Zoo Morphology Data File not found")
            self.gz = Table()

        self.sersic = sersic

        # Set Ind Dictionary of Targets by MangaID
        self.ind_dict_target = dict((k.rstrip(),i) for i,k in enumerate(self.targets['MANGAID']))

        # Set some Milky Way Stellar Mass Estimates
        self.mw_stellar_mass = 6.43 * 10**10 * u.solMass
        self.mw_stellar_mass_err = 0.63 * 10**10 * u.solMass

        self.mw_stellar_mass_jbh = 5.0 * 10**10 * u.solMass
        self.mw_stellar_mass_jbh_err = 1.0 * 10**10 * u.solMass


        self.targets_gz = self.targets_in_gz()

        # Get default mass matches:
        self.determine_mass_mwa(reset = True, sersic = sersic)
        self.determine_mass_mwa(reset = True, barred = False, sersic = sersic)
        self.determine_mass_mwa(reset = True, no_morph = True, sersic = sersic)
        self.determine_mass_mwa(reset = True, jbh = True, sersic = sersic)
        self.determine_mass_mwa(reset = True, jbh = True, barred = False, sersic = sersic)
        self.determine_mass_mwa(reset = True, jbh = True, no_morph = True, sersic = sersic)

        # Get Images
        self.get_mass_images(return_images = False)

        # DAP Star Formation Rates
        self.mass_mwa_sfr = (self.mass_mwa_dap["SFR_TOT"] * u.solMass / u.yr * u.littleh**-2).to(u.solMass / u.yr, u.with_H0(WMAP9.H0))
        self.mass_mwa_nobar_sfr = (self.mass_mwa_nobar_dap["SFR_TOT"] * u.solMass / u.yr * u.littleh**-2).to(u.solMass / u.yr, u.with_H0(WMAP9.H0))

        self.mass_jbh_mwa_sfr = (self.mass_jbh_mwa_dap["SFR_TOT"] * u.solMass / u.yr * u.littleh**-2).to(u.solMass / u.yr, u.with_H0(WMAP9.H0))
        self.mass_jbh_mwa_nobar_sfr = (self.mass_jbh_mwa_nobar_dap["SFR_TOT"] * u.solMass / u.yr * u.littleh**-2).to(u.solMass / u.yr, u.with_H0(WMAP9.H0))

        # Set default sample
        self.dk_sample_sfr_mask = self.mass_mwa_sfr > 0.1 * u.solMass / u.yr
        self.dk_sample_nobar_sfr_mask = self.mass_mwa_nobar_sfr > 0.1 * u.solMass / u.yr

        self.dk_sample_jbh_sfr_mask = self.mass_jbh_mwa_sfr > 0.1 * u.solMass / u.yr
        self.dk_sample_jbh_nobar_sfr_mask = self.mass_jbh_mwa_nobar_sfr > 0.1 * u.solMass / u.yr

        # Core Sample
        self.dk_sample = self.mass_mwa_dap[self.dk_sample_sfr_mask]
        self.dk_sample_sfr = self.mass_mwa_sfr[self.dk_sample_sfr_mask]

        self.dk_sample_jbh = self.mass_jbh_mwa_dap[self.dk_sample_jbh_sfr_mask]
        self.dk_sample_jbh_sfr = self.mass_jbh_mwa_sfr[self.dk_sample_jbh_sfr_mask]

        # Control Sample
        self.dk_sample_nobar = self.mass_mwa_nobar_dap[self.dk_sample_nobar_sfr_mask]
        self.dk_sample_nobar_sfr = self.mass_mwa_nobar_sfr[self.dk_sample_nobar_sfr_mask]

        self.dk_sample_jbh_nobar = self.mass_jbh_mwa_nobar_dap[self.dk_sample_jbh_nobar_sfr_mask]
        self.dk_sample_jbh_nobar_sfr = self.mass_jbh_mwa_nobar_sfr[self.dk_sample_jbh_nobar_sfr_mask]

        # Sort by z
        argsort = np.argsort(self.dk_sample["NSA_Z"])
        self.dk_sample = self.dk_sample[argsort]
        self.dk_sample_sfr = self.dk_sample_sfr[argsort]

        argsort = np.argsort(self.dk_sample_nobar["NSA_Z"])
        self.dk_sample_nobar = self.dk_sample_nobar[argsort]
        self.dk_sample_nobar_sfr = self.dk_sample_nobar_sfr[argsort]

        # Set Full Barred and Unbarred Sample
        self.barred_sample = self.get_barred_galaxies_dap()
        argsort = np.argsort(self.barred_sample["NSA_Z"])
        self.barred_sample = self.barred_sample[argsort]
        self.nonbarred_sample = self.get_barred_galaxies_dap(nonbarred = True)
        argsort = np.argsort(self.nonbarred_sample["NSA_Z"])
        self.nonbarred_sample = self.nonbarred_sample[argsort]


    def targets_in_drp(self):
        """
        return Table of Targets that are in the DRP ALL FILE
        """
        data_targets_drp_ind = [self.ind_dict_target[x] for x in self.drp['mangaid']]
        return self.targets[data_targets_drp_ind]

    def targets_in_dap(self):
        """
        return Table of Targets that are in the DRP ALL FILE
        """
        data_targets_dap_ind = [self.ind_dict_target[x] for x in self.dap['MANGAID']]
        return self.targets[data_targets_dap_ind]

    def targets_in_gz(self):
        """
        return Table of Targets that are in the Galaxy Zoo Catalog
        """
        data_targets_gz_ind = [self.ind_dict_target[x] for x in self.gz['MANGAID']]
        return self.targets[data_targets_gz_ind]

    def get_barred_galaxies_mask(self):
        """
        Return Barred Galaxy Mask from self.gz
        """
        return (self.gz['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & \
                (self.gz['t02_edgeon_a05_no_debiased'] > 0.715) & \
                (self.gz['t02_edgeon_a05_no_count'] >= 20) & \
                (self.gz['t03_bar_a06_bar_debiased'] >= 0.8)

    def get_nonbarred_galaxies_mask(self):
        """
        Return Non-Barred Galaxy Mask from self.gz
        """
        return (self.gz['t01_smooth_or_features_a02_features_or_disk_debiased'] > 0.430) & \
                (self.gz['t02_edgeon_a05_no_debiased'] > 0.715) & \
                (self.gz['t02_edgeon_a05_no_count'] >= 20) & \
                (self.gz['t03_bar_a06_bar_debiased'] <= 0.2)

    def get_barred_galaxies_dap(self, nonbarred = False):
        """
        Return DAP Table for barred/nonbarred galaxies
        """
        bar_mask = self.get_barred_galaxies_mask()
        no_bar_mask = self.get_nonbarred_galaxies_mask()
        barred_IDs = [mangaid.decode("utf-8").rstrip() for mangaid in self.targets_gz[bar_mask]["MANGAID"].data]
        nonbarred_IDs = [mangaid.decode("utf-8").rstrip() for mangaid in self.targets_gz[no_bar_mask]["MANGAID"].data]
        ind_dict_drp = dict((k,i) for i,k in enumerate(self.drp['mangaid']))
        barred_in_drp = set(ind_dict_drp).intersection(barred_IDs)
        bar_in_drp_ind = [ind_dict_drp[x] for x in barred_in_drp]
        nonbarred_in_drp = set(ind_dict_drp).intersection(nonbarred_IDs)
        nonbar_in_drp_ind = [ind_dict_drp[x] for x in nonbarred_in_drp]
        barred_plateifus = [plateifu.decode("utf").rstrip() for plateifu in self.drp[bar_in_drp_ind]["plateifu"].data]
        nonbarred_plateifus = [plateifu.decode("utf").rstrip() for plateifu in self.drp[nonbar_in_drp_ind]["plateifu"].data]
        ind_dict_dap = dict((k,i) for i,k in enumerate(self.dap['PLATEIFU']))

        bad_barred_ind = ind_dict_dap["10507-12705"]
        barred_sample_dap_ind = np.array([ind_dict_dap[plateifu] for plateifu in barred_plateifus])
        good_barred_mask = np.array([ind != bad_barred_ind for ind in barred_sample_dap_ind])
        barred_sample_dap_ind = barred_sample_dap_ind[good_barred_mask]

        bad_nonbarred_inds = [ind_dict_dap[bad] for bad in ["8332-12704", "8616-3704", "10498-12704"]]
        nonbarred_sample_dap_ind = np.array([ind_dict_dap[plateifu] for plateifu in nonbarred_plateifus])
        good_nonbarred_mask = np.array([ind not in bad_nonbarred_inds for ind in nonbarred_sample_dap_ind])
        nonbarred_sample_dap_ind = nonbarred_sample_dap_ind[good_nonbarred_mask]

        barred_sample = self.dap[barred_sample_dap_ind]
        nonbarred_sample = self.dap[nonbarred_sample_dap_ind]


        if nonbarred:
            return nonbarred_sample
        else:
            return barred_sample


    def get_stellar_mass_mwa_mask(self, sersic = False):
        """
        Return mask of galaxies in self.targets_gz that fit within stellar mass range of McMillan (2011) MW Stellar Mass estimates
        """
        if sersic:
            key = "NSA_SERSIC_MASS"
        else:
            key = 'NSA_ELPETRO_MASS'

        NSA_STELLAR_MASS = (10**self.targets_gz[key] * u.solMass* u.littleh**-2).to(u.solMass, u.with_H0(WMAP9.H0))
        return (NSA_STELLAR_MASS < (self.mw_stellar_mass + self.mw_stellar_mass_err)) & \
                            (NSA_STELLAR_MASS > (self.mw_stellar_mass - self.mw_stellar_mass_err))

    def get_jbh_stellar_mass_mwa_mask(self, sersic = False):
        """nsa_elpetro_mass
        Return mask of galaxies in self.targets_gz that fit within stellar mass range of JBH (2016) MW Stellar Mass estimates
        """
        if sersic:
            key = "NSA_SERSIC_MASS"
        else:
            key = 'NSA_ELPETRO_MASS'
        NSA_STELLAR_MASS = (10**self.targets_gz[key] * u.solMass* u.littleh**-2).to(u.solMass, u.with_H0(WMAP9.H0))
        return (NSA_STELLAR_MASS < (self.mw_stellar_mass_jbh + self.mw_stellar_mass_jbh_err)) & \
                            (NSA_STELLAR_MASS > (self.mw_stellar_mass_jbh - self.mw_stellar_mass_jbh_err))


    def determine_mass_mwa(self, reset = False, jbh = False, barred = True, no_morph = False, sersic = False):
        """
        Return dap and drp entries with Stellar Mass within MW range

        Parameters
        ----------
        reset: 'bool', optional, must be keyword
            if True, resets self.dap_mass_mwa to this
        jbh: 'bool', optional, must be keyword
            if True, uses JBH Stellarr Mass estimate
        barred: 'bool', optional, must be keyword
            if True, returns barred galaxies only
            if False, returns non-barred galaxies only
        no_morph: 'bool', optional, must be keyword
            if True, doesn't consider morphology information and returns only mass cuts
        sersic: 'bool', if True, uses sersic mass
        """
        if no_morph:
            if reset:
                if not jbh:
                    self.mass_targets = self.targets_gz[self.get_stellar_mass_mwa_mask(sersic = sersic)]
                else:
                    self.mass_jbh_targets = self.targets_gz[self.get_jbh_stellar_mass_mwa_mask(sersic = sersic)]
            else:
                if not jbh:
                    return self.targets_gz[self.get_stellar_mass_mwa_mask(sersic = sersic)]
                else:
                    return self.targerts_gz[self.get_jbh_stellar_mass_mwa_mask(sersic = sersic)]
        else:
            if not jbh:
                if barred:
                    mwa_gz = self.gz[np.logical_and(self.get_stellar_mass_mwa_mask(sersic = sersic), self.get_barred_galaxies_mask())]
                else:
                    mwa_gz = self.gz[np.logical_and(self.get_stellar_mass_mwa_mask(sersic = sersic), self.get_nonbarred_galaxies_mask())]
            else:
                if barred:
                    mwa_gz = self.gz[np.logical_and(self.get_jbh_stellar_mass_mwa_mask(sersic = sersic), self.get_barred_galaxies_mask())]
                else:
                    mwa_gz = self.gz[np.logical_and(self.get_jbh_stellar_mass_mwa_mask(sersic = sersic), self.get_nonbarred_galaxies_mask())]
            ind_dict_dap = dict((k,i) for i,k in enumerate(self.dap['MANGAID']))
            inter_bar_stellar_dap = set(ind_dict_dap).intersection(mwa_gz['MANGAID'])
            bar_stellar_dap_ind = [ind_dict_dap[x] for x in inter_bar_stellar_dap]
            bar_stellar_drp_ind = self.dap['DRPALLINDX'][bar_stellar_dap_ind]

            if reset:
                if not jbh:
                    if barred:
                        self.mass_mwa_dap = self.dap[bar_stellar_dap_ind]
                        self.mass_mwa_drp = self.drp[bar_stellar_drp_ind]
                    else:
                        self.mass_mwa_nobar_dap = self.dap[bar_stellar_dap_ind]
                        self.mass_mwa_nobar_drp = self.drp[bar_stellar_drp_ind]
                else:
                    if barred:
                        self.mass_jbh_mwa_dap = self.dap[bar_stellar_dap_ind]
                        self.mass_jbh_mwa_drp = self.drp[bar_stellar_drp_ind]
                    else:
                        self.mass_jbh_mwa_nobar_dap = self.dap[bar_stellar_dap_ind]
                        self.mass_jbh_mwa_nobar_drp = self.drp[bar_stellar_drp_ind]
                
            return self.dap[bar_stellar_dap_ind], self.drp[bar_stellar_drp_ind]


    def get_mass_images(self, return_images = True, barred = True, jbh = False):
        """
        Get images of Mass based MWAs

        Parameters
        ----------

        return_images: 'bool', optional, must be keyword
            if True, returns images
            if False, adds images to class
        jbh: 'bool', optional, must be keyword
            if True, uses JBH Stellarr Mass estimate
        barred: 'bool', optional, must be keyword
            if True, returns barred galaxies only
            if False, returns non-barred galaxies only
        """
        if jbh:
            if barred:
                images = Image.from_list(self.mass_jbh_mwa_dap["PLATEIFU"])
                if return_images:
                    return images
                else:
                    self.mass_jbh_mwa_images = images
            else:
                images = Image.from_list(self.mass_jbh_mwa_nobar_dap["PLATEIFU"])
                if return_images:
                    return images
                else:
                    self.mass_jbh_mwa_nobar_images = images
        else:
            if barred:
                images = Image.from_list(self.mass_mwa_dap["PLATEIFU"])
                if return_images:
                    return images
                else:
                    self.mass_mwa_images = images
            else:
                images = Image.from_list(self.mass_mwa_nobar_dap["PLATEIFU"])
                if return_images:
                    return images
                else:
                    self.mass_mwa_nobar_images = images


















