import logging
import warnings

from marvin.tools.maps import Maps

import astropy.units as u
import astropy.wcs as wcs
import astropy.constants as constants
from astropy.cosmology import WMAP9

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

pal = sns.color_palette('colorblind')

from extinction import fm07 as extinction_law
import matplotlib
import os
import glob

from .pca import PCA_stellar_mass
from .pca import PCA_MLi
from .gz3d_fits import gz3d_fits


# Dictionary of chanell labels:
labels_dict = {
    'ha_6564' : r"$H\alpha$ $\lambda 6564 \AA$ ({})",
    'hb_4862' : r"$H\beta$ $\lambda 4862 \AA$ ({})",
    'oii_3727' : r"[OII] $\lambda 3727 \AA$ ({})",
    'oii_3729' : r"[OII] $\lambda 3729 \AA$ ({})",
    'oiii_4960' : r"[OIII] $\lambda 4960 \AA$ ({})",
    'oiii_5008' : r"[OIII] $\lambda 5008 \AA$ ({})",
    'hei_5877' : r"He I $\lambda 5877 \AA$ ({})",
    'oi_6302' : r"[OI] $\lambda 6302 \AA$ ({})",
    'oi_6365' : r"[OI] $\lambda 6365 \AA$ ({})",
    'nii_6549' : r"[NII] $\lambda 6549 \AA$ ({})",
    'nii_6585' : r"[NII] $\lambda 6585 \AA$ ({})",
    'nii_6718' : r"[NII] $\lambda 6718 \AA$ ({})",
    'nii_6732' : r"[NII] $\lambda 6732 \AA$ ({})",
    'r_re' : r"$R / R_e$",
    'elliptical_radius' : r"Elliptical Radius ({})" 
}


class DKMapsMixin(object):
    """
    Mixin Functionality
    """
    def balmer_Av(self, expected_ratio = 2.92, snr_min = None):
        """
        Estimate Av from balmer Decrement

        Parameters
        ----------
        expected_ratio: 'number', optional, must be keyword
            Expected ha/hb ratio in Energy Units
        snr_min: 'number', optional, must be keyword
            Minimum SNR Threshold to use


        """

        ha = self['emline gflux ha']
        hb = self['emline gflux hb']

        ha_masked = ha.masked
        hb_masked = hb.masked

        if snr_min is not None:
            ha_masked.mask |= ha.snr < snr_min
            hb_masked.mask |= hb.snr < snr_min

        hahb = ha_masked / hb_masked
        AV = 2.68 * np.log(hahb / expected_ratio)
        AV.mask |= AV < 0.


        return AV

    def deredden(self, name, Av = None, **kwargs):
        """
        De-redden a Map

        Parameters
        ----------
        name: 'str'
            name of Maps key to Map that will be dereddened
        Av: 'list' or 'np.ma.masked_array' or 'np.array', optional, must be keyword
            Av values to use
            if not provided, it is estimated using the balmer decrememnt
        kwargs: 'dict', optional, must be keywords
            keywords passed to balmer_Av
        """
        if not hasattr(self, name):
            raise ValueError("cannot find a good match for '{}'. Your input value is too ambiguous.".format(name))

        og_map = self[name]

        if Av is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Av = self.balmer_Av(**kwargs)

        wave = np.array([og_map.datamodel.channel.name[-4:]], dtype = np.float)
        A_v_to_A_lambda = extinction_law(wave, 1.)
        A_lambda = Av * A_v_to_A_lambda

        return og_map * 10**(0.4 * A_lambda)

    def luminosity(self, name, lum_distz, deredden = False, **kwargs):
        """
        Calculate Emission Line Luminosity

        Parameters
        ----------
        name: 'str'
            name of Maps key to Map that will be dereddened
        lum_distz: 'number', 'u.Quantity'
            luminosity distance in units of Mpc / littleh
        deredden: 'bool', optional, must be keyword
            if True, dereddens flux first
        kwargs: 'dict', optional, must be keyword
            keywords passed to deredden

        """
        if not isinstance(lum_distz, u.Quantity):
            logging.warning("No units provided for Luminosity Distance, assuming u.Mpc / u.littleh")
            lum_distz *= u.Mpc / u.littleh
        
        if deredden:
            flux = self.deredden(name, **kwargs)
        else:
            flux = self[name]
        lum = 4. * np.pi * flux.value * flux.unit * lum_distz**2
        return lum.to(u.erg / u.s / u.pix, u.with_H0(WMAP9.H0))

    def plot_bpt_nii(self, **kwargs):
        """
        plots NII/HA BPT Diagram in style similar to Krishnarao+19, with option to overlay Milky Way Data

        Parameters
        ----------
        kwargs: 'dict', optional, must be keyword
            passed to 'dk_manga_tools.bpt.bpt_nii'
        """
        from .bpt import bpt_nii
        return bpt_nii(self, **kwargs)

    def plot_radial_emline(self, name, ax = None, deredden = False, 
        Re = True, snr_min = None, log10 = False, deredden_kwargs = {}, **kwargs):
        """
        Plots emline as a funtion of R/Re

        Parameters
        ----------
        name: 'str', 'list'
            name of Maps key to Map that will be dereddened
            if list, uses ratio of two emlines
        ax: 'matplotlib.pyplot.figure.axes': optional, must be keyword
            Matplotlib Axes instance to plot on
        deredden: 'bool', optional, must be keyword
            if True, dereddens flux first
        Re: 'bool', optional, must be keyword
            if True, plots x axis as R/Re
            if False, plots x axis as R
        snr_min: 'number', optional, must be keyword
            Minimum SNR to mask out
        log10: 'bool', optional, must be keyword
            if True, plots log_10 of the value
            only if ratios are plotted
        deredden_kwargs: 'dict', optional, must be keywords:
            Keywords passed to dereddedn
        kwargs: 'optional', must be keyword
            keywords passed to scatter plot
        """
        if (name.__class__ is list) | (name.__class__ is tuple):
            assert len(name) == 2, "too many emline entries provided"
            name, denom_name = name
            ratios = True
        else:
            ratios = False

        if deredden:
            flux = self.deredden(name, **deredden_kwargs)
            if ratios:
                flux_denom = self.deredden(denom_name, **deredden_kwargs)
        else:
            flux = self[name]
            if ratios:
                flux_denom = self[denom_name]

        if Re:
            radius = self['spx ellcoo r_re']
        else:
            radius = self['spx ellcoo radius']

        # Default SNR_MIN
        if snr_min is None:
            snr_min = 2.

        flux_masked = flux.masked
        flux_masked.mask |= flux.snr <= snr_min
        if ratios:
            flux_denom_masked = flux_denom.masked
            flux_denom_masked.mask |= flux_denom.snr <= snr_min
            flux_denom_masked.mask |= flux_denom_masked <= 0.


        try: 
            y_label = labels_dict[flux.datamodel.channel.name]
        except KeyError:
            y_label = r"" + flux.datamodel.channel.name + "({})"

        if ratios:
            try: 
                y_label_denom = labels_dict[flux_denom.datamodel.channel.name]
            except KeyError:
                y_label_denom = r"" + flux_denom.datamodel.channel.name

            if y_label[-4:] == "({})":
                y_label = y_label[:-4]
            if y_label_denom[-4:] == "({})":
                y_label_denom = y_label_denom[:-4]

            y_label = y_label + " / " + y_label_denom
            if log10:
                y_label = r"$Log_{{10}}$ " + y_label

        x_label = labels_dict[radius.datamodel.channel.name]

        # Check if axes are created
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        # Default kwargs:
        if ("color" not in kwargs) | ("c" not in kwargs):
            color = pal[0]
        if ratios:
            flux_ratio = flux_masked / flux_denom_masked
            if log10:
                flux_ratio = np.log10(flux_ratio)
            ax.scatter(radius.value.flatten(), flux_ratio.flatten(), **kwargs)
            ax.set_ylabel(y_label, fontsize = 12)
        else:
            ax.scatter(radius.value.flatten(), flux_masked.flatten(), **kwargs)
            ax.set_ylabel(y_label.format(flux.unit.to_string("latex")), fontsize = 12)

        ax.set_xlabel(x_label.format(radius.unit.to_string("latex")), fontsize = 12)

        return ax

    def plot_violin_bpt_nii(self, ax = None,
                        deredden = False, Re = True, 
                        snr_min = None, deredden_kwargs = {},
                        **kwargs):
        """
        Plots categorical BPT Classification as a funtion of R/Re

        Parameters
        ----------
        ax: 'matplotlib.pyplot.figure.axes': optional, must be keyword
            Matplotlib Axes instance to plot on
        deredden: 'bool', optional, must be keyword
            if True, dereddens flux first
        Re: 'bool', optional, must be keyword
            if True, plots x axis as R/Re
            if False, plots x axis as R
        snr_min: 'number', optional, must be keyword
            Minimum SNR to mask out
        deredden_kwargs: 'dict', optional, must be keywords:
            Keywords passed to dereddedn
        kwargs: 'optional', must be keyword
            keywords passed to sns.violinplot
        """


        if Re:
            radius = self['spx ellcoo r_re']
        else:
            radius = self['spx ellcoo radius']

        bpt_classifications = self.plot_bpt_nii(return_figure = False, 
                                                snr_min = snr_min, 
                                                deredden = deredden, 
                                                deredden_kwargs = deredden_kwargs)

        # Default kwargs
        if "palette" not in kwargs:
            kwargs["palette"] = [pal[1], pal[0], pal[9], pal[4]]

        if "saturation" not in kwargs:
            kwargs["saturation"] = 1.5

        if "inner" not in kwargs:
            kwargs["inner"] = 'quartile'


        # Check if axes are created
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        
        # Violin Plots
        sf_x = radius[bpt_classifications["sf"]["nii"]].flatten().value
        sf_y = ["SF"] * len(sf_x)

        comp_x = radius[bpt_classifications["comp"]["nii"]].flatten().value
        comp_y = ["Composite"] * len(comp_x)

        agn_x = radius[bpt_classifications["agn"]["nii"]].flatten().value
        agn_y = ["AGN"] * len(agn_x)

        liner_x = radius[bpt_classifications["liner"]["nii"]].flatten().value
        liner_y = ["LI(N)ER"] * len(liner_x)

        # invalid_x = radius[bpt_classifications["invalid"]["nii"]].flatten().value
        # invalid_y = ["invalid"] * len(invalid_x)

        x = np.concatenate((sf_x, comp_x, agn_x, liner_x), axis = None)
        y = np.concatenate((sf_y, comp_y, agn_y, liner_y), axis = None)

        data = {"x":x, "y":y}

        sns.violinplot(ax = ax, x = "x", 
                       y = "y", data = data, **kwargs,
                       )
        x_label = labels_dict[radius.datamodel.channel.name]
        ax.set_xlabel(x_label.format(radius.unit.to_string("latex")), fontsize = 12)

        if ("alpha" in kwargs) | ("zorder" in kwargs):
            artists = ax.get_default_bbox_extra_artists()
            for artist in artists:
                if artist.__class__ is matplotlib.collections.PolyCollection:
                    if "alpha" in kwargs:
                        artist.set_alpha(kwargs["alpha"])
                    if "zorder" in kwargs:
                        artist.set_zorder(kwargs["zorder"])

                        






        return ax




    def get_radial_bpt_counts(self, radial_bin, binned_counts = False, bin_width = None,
                              snr_min = None, deredden = False, deredden_kwargs = {}, radial_norm = 1.,
                              pool = None, set_up = True, add_to = None, keep_pool = False):
        """
        Get number of spaxels in each BPT classification within a specified radial bin

        Parameters
        ----------
        radial_bin: 'number', 'list'
            radius value in terms of R_e 

        binned_counts: 'bool', optional, must be keyword
            if True, counts spaxels within bins, instead of cumulatively
        bin_width: 'number', optional, must be keyword
            bin width to use if binned_counts is True
        deredden: 'bool', optional, must be keyword
            if True, dereddens flux first
        snr_min: 'number', optional, must be keyword
            Minimum SNR to mask out
        deredden_kwargs: 'dict', optional, must be keyword
            Keywords passed to dereddedn
        radial_norm: 'number', 'u.Quantity', optional, must be keyword
            normalization value for radius in Re
        pool: 'multiprocessing.pool', optional, must be keyword
            pool with a map method for multiprocessing capabilities
        set_up: 'bool', optional, must be keyword
            Default is True
            used to regulate recursion - not usually used by end-user
        add_to: 'dict', optional, must be keyword
            Dictionary of counts to add values onto
        keep_pool: 'bool', optional, must be keyword
            if True, does not close pool
            not usually used by end-user

        Returns
        -------
        counts: 'dict'
            Dictionary of counts per each classification and Total spaxels count
        """



        if (radial_bin.__class__ in [np.ndarray, list, tuple]) & (set_up):
            from functools import partial
            # Map multiple radial bin values as output
            partial_func = partial(self.get_radial_bpt_counts, 
                                  snr_min = snr_min, 
                                  binned_counts = binned_counts, 
                                  bin_width = bin_width,
                                  deredden = deredden, 
                                  deredden_kwargs = deredden_kwargs, 
                                  set_up = False)
            if pool is not None:
                try: 
                    res = pool.map(partial_func, radial_bin)
                except AttributeError:
                    logging.warning("Invalid Pool, pool has no map method.")
                    res = map(partial_func, radial_bin)
                else: 
                    if keep_pool is False:
                        pool.close()
            else:
                res = map(partial_func, radial_bin)

            counts_sub = [*res]
            counts = {}
            for k in counts_sub[0]:
                counts[k] = np.array(list(d[k] for d in counts_sub))

            if add_to is not None:
                tot_counts = {}
                for k in counts:
                    tot_counts[k] = np.sum(list(d[k] for d in [counts, add_to]), axis = 0)
                return tot_counts
            else:
                return counts
        else:
            # initialize dictionary
            counts = {}

            if binned_counts:
                if bin_width is None:
                    bin_width = 0.2 #R_e
                within_radius = (self['spx ellcoo r_re'].value <= radial_bin / radial_norm + bin_width/2.) 
                within_radius &= (self['spx ellcoo r_re'].value > radial_bin / radial_norm - bin_width/2.)            
            else:
                # Get radius values from Map
                within_radius = self['spx ellcoo r_re'].value <= radial_bin * radial_norm

            # Get bpt_classificaitons
            bpt_classificaitons = self.plot_bpt_nii(return_figure = False, 
                                                    snr_min = snr_min, 
                                                    deredden = deredden, 
                                                    deredden_kwargs = deredden_kwargs)

            counts["sf"] = (bpt_classificaitons["sf"]["nii"] & within_radius).sum()
            counts["comp"] = (bpt_classificaitons["comp"]["nii"] & within_radius).sum()
            counts["agn"] = (bpt_classificaitons["agn"]["nii"] & within_radius).sum()
            counts["liner"] = (bpt_classificaitons["liner"]["nii"] & within_radius).sum()
            counts["invalid"] = (bpt_classificaitons["invalid"]["nii"] & within_radius).sum()
            counts["total"] = counts["sf"] + counts["comp"] + counts["agn"] + counts["liner"] + counts["invalid"]

            if add_to is not None:
                tot_counts = {}
                for k in counts:
                    tot_counts[k] = np.sum(list(d[k] for d in [counts, add_to]), axis = 0)
                return add_to
            else:
                return counts





    def get_PCA_stellar_mass(self, pca_data_dir = None, **kwargs):
        """
        Return PCA Stellar Mass map with errors in u.solmass
        """
        if pca_data_dir is None:
            pca_data_dir = os.path.join(os.environ['SAS_BASE_DIR'], 'mangawork', 'manga', 'sandbox', 'mangapca', 'zachpace', 'CSPs_CKC14_MaNGA_20190215-1',
                           self.dapall["versdrp3"], self.dapall["versdap"], 'results')
        return PCA_stellar_mass(self.dapall, pca_data_dir = pca_data_dir, **kwargs)


    def get_deproj_pixel_area(self):
        """
        Return deprojected pixel area of Galaxy in u.pc**2
        """

        alpha = .13 # axis ratio of a perfectly edge-on system

        inclination = np.arccos(
          np.sqrt(
            (self.dapall['nsa_elpetro_ba']**2. - alpha**2.)/(1 - alpha**2.)))
        D = self.dapall["nsa_zdist"] * constants.c / WMAP9.H0
        proj_pix_area = wcs.utils.proj_plane_pixel_area(self.wcs) *u.deg**2

        return (proj_pix_area.to(u.sr).value * D**2).to(u.pc**2) / np.cos(inclination)

    def get_PCA_stellar_mass_density(self, **kwargs):
        """
        Return PCA stellar mass / deprojected pixel area in units of u.solMass * u.pc**-2
        """
        m_star = self.get_PCA_stellar_mass(**kwargs)
        area = self.get_deproj_pixel_area()
        return m_star / area

    def get_PCA_MLi(self, pca_data_dir = None, **kwargs):
        """
        Return PCA Stellar Mass-to-Light ratio in the i-band
        """
        if pca_data_dir is None:
            pca_data_dir = os.path.join(os.environ['SAS_BASE_DIR'], 'mangawork', 'manga', 'sandbox', 'mangapca', 'zachpace', 'CSPs_CKC14_MaNGA_20190215-1',
                           self.dapall["versdrp3"], self.dapall["versdap"], 'results')
        return PCA_MLi(self.dapall, pca_data_dir = pca_data_dir, **kwargs)

    def get_bar_mask(self, galaxyzoo3d_dir = None, vote_threshold = None, **kwargs):
        """
        If available get Galaxy Zoo 3D Bar Spaxel Mask

        Parameters
        ----------
        galaxyzoo3d_dir: 'str', optional, must be keyword
            Directory to find data files
        vote_threshold: 'number', optional, must be keyword
            Vote threshold to consider; default of 0.2 (20%)
        """


        if galaxyzoo3d_dir is None:
            galaxyzoo3d_dir = "/Users/dk/sas/mangawork/manga/sandbox/galaxyzoo3d/v2_0_0/"

        if vote_threshold is None:
            vote_threshold = 0.2


        filename = glob.glob(galaxyzoo3d_dir+"{}*.fits.gz".format(self.mangaid))
        if filename == []:
            logging.warning("No Galaxy Zoo 3D Data Available for this Galaxy!")
            return np.zeros(self["emline gflux ha"].value.shape[0:], dtype = bool)
        else:
            data = gz3d_fits(filename[0], maps = self)
            data.make_all_spaxel_masks()
            if np.all(data.bar_mask_spaxel == 0):
                logging.warning("No Bar Mask Available for this Galaxy!")
                return np.zeros(self["emline gflux ha"].value.shape[0:], dtype = bool)
            else:
                try:
                    bar_mask = data.bar_mask_spaxel >= data.metadata["GZ2_bar_votes"] * vote_threshold
                except KeyError:
                    bar_mask = data.bar_mask_spaxel >= data.metadata["GZ_bar_votes"] * vote_threshold

                if np.any(bar_mask):
                    return bar_mask
                else:
                    logging.warning("No Bar Mask Available for this Galaxy above vote threshold!")
                    return np.zeros(self["emline gflux ha"].value.shape[0:], dtype = bool)

    def get_map(self,  *args, snr_min = None, **kwargs):
        """
        Retrives map with a minimum SNR cut applied
        """
        if snr_min is None:
            snr_min = 3.0

        m = self.getMap(*args, **kwargs)

        m_masked = m.masked
        m_masked.mask |= m.snr <= snr_min

        return m_masked
























class DKMaps(DKMapsMixin, Maps):
    """
    Wrapper Class for custom functionality with Marvin Maps

    Parameters
    ----------
    kwargs: 'dict', optional, must be keyword
        keywords passed to 'marvin.tools.maps.Maps'
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

