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

from astropy.coordinates import CartesianRepresentation, CylindricalRepresentation, Angle

pal = sns.color_palette('colorblind')

from extinction import fm07 as extinction_law
import matplotlib
import os
import glob

from .pca import PCA_stellar_mass
from .pca import PCA_MLi
from .pca import PCA_zpres_info
from .pca import PCA_mag
from .gz3d_fits import gz3d_fits

from .timeslice_utils import timecube, agemap, Total, metmap, SFH


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

from scipy.spatial import ConvexHull

def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


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

        og_map = self[name].masked

        if Av is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Av = self.balmer_Av(**kwargs)

        wave = np.array([self[name].datamodel.channel.name[-4:]], dtype = np.float)
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
            flux = self[name].masked

        flux_unit = self[name].unit
        lum = 4. * np.pi * flux.data * flux_unit * lum_distz**2
        lum_out = lum.to(u.erg / u.s / u.pix, u.with_H0(WMAP9.H0))
        return np.ma.masked_array(data = lum_out, mask = flux.mask)

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
            pca_data_dir = os.path.join(os.environ['SAS_BASE_DIR'], 'dr17', 'manga', 'spectro', 'mangapca', 'CSPs_CKC14_MaNGA_20190215-1',
                           'v2_5_3', '2.3.0', 'results')
        return PCA_stellar_mass(self.dapall, pca_data_dir = pca_data_dir, **kwargs)

    def get_timeslice_mass(self, timeslice_dir = None, **kwargs):
        """
        Return TimeSlice Mass Map in units of u.solMass
        """
        if timeslice_dir is None:
            timeslice_dir = "/Users/dk/sas/mangawork/manga/sandbox/starlight/MPL9_Spirals_noSpecs/"

        ffn = "{}{}_E-n.fits".format(timeslice_dir, self.plateifu)

        tc = timecube(ffn, weight_type='current_mass')
        return tc.sum_im * u.solMass

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

        smsd = np.ma.masked_array(data = m_star/area, mask = m_star.mask)
        return m_star / area

    def get_timeslice_mass_density(self, **kwargs):
        """
        Return TimeSlice stellar mass / deprojected pixel area in units of u.solMass * u.pc**-2
        """
        m = self.get_timeslice_mass(**kwargs)
        area = self.get_deproj_pixel_area()
        return m/area

    def get_timeslice_mean_age(self, timeslice_dir = None, weight_type = "light"):
        """
        Return TimeSlice mean weighted age
        """
        if timeslice_dir is None:
            timeslice_dir = "/Users/dk/sas/mangawork/manga/sandbox/starlight/MPL9_Spirals_noSpecs/"

        ffn = "{}{}_E-n.fits".format(timeslice_dir, self.plateifu)
        tc = timecube(ffn, weight_type = weight_type)
        return agemap(Total(tc))

    def get_timeslice_metallicity(self, timeslice_dir = None, weight_type = "light"):
        """
        Return TimeSlice mean weighted age
        """
        if timeslice_dir is None:
            timeslice_dir = "/Users/dk/sas/mangawork/manga/sandbox/starlight/MPL9_Spirals_noSpecs/"

        ffn = "{}{}_E-n.fits".format(timeslice_dir, self.plateifu)
        tc = timecube(ffn, weight_type = weight_type)
        return metmap(Total(tc))
    def get_timeslice_SFH(self, timeslice_dir = None, weight_type = "initial_mass"):
        """
        Return TimeSlice SFH_age and SFH_sfrs
        """
        if timeslice_dir is None:
            timeslice_dir = "/Users/dk/sas/mangawork/manga/sandbox/starlight/MPL9_Spirals_noSpecs/"

        ffn = "{}{}_E-n.fits".format(timeslice_dir, self.plateifu)
        tc = timecube(ffn, weight_type = weight_type)
        return SFH(tc)



    def get_PCA_MLi(self, pca_data_dir = None, **kwargs):
        """
        Return PCA Stellar Mass-to-Light ratio in the i-band
        """
        if pca_data_dir is None:
            pca_data_dir = os.path.join(os.environ['SAS_BASE_DIR'], 'dr17', 'manga', 'spectro', 'mangapca', 'CSPs_CKC14_MaNGA_20190215-1',
                           'v2_5_3', '2.3.0', 'results')
        return PCA_MLi(self.dapall, pca_data_dir = pca_data_dir, **kwargs)

    def get_PCA_zpres_info(self, name, pca_data_dir = None, **kwargs):
        """
        Return additional specified info from PCA
        """
        if pca_data_dir is None:
            pca_data_dir = os.path.join(os.environ['SAS_BASE_DIR'], 'dr17', 'manga', 'spectro', 'mangapca', 'CSPs_CKC14_MaNGA_20190215-1',
                           'v2_5_3', '2.3.0', 'results')
        return PCA_zpres_info(self.dapall, name, pca_data_dir = pca_data_dir, **kwargs)

    def get_PCA_mag(self, filter_obs, pca_data_dir = None, **kwargs):
        """
        Return PCA mag in specified filter
        """
        if pca_data_dir is None:
            pca_data_dir = os.path.join(os.environ['SAS_BASE_DIR'], 'dr17', 'manga', 'spectro', 'mangapca', 'CSPs_CKC14_MaNGA_20190215-1',
                           'v2_5_3', '2.3.0', 'results')
        return PCA_mag(self.dapall, filter_obs, pca_data_dir = pca_data_dir, **kwargs)


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
            galaxyzoo3d_dir = os.path.join(os.environ['SAS_BASE_DIR'], 'dr17', 'manga', 'morphology', 'galaxyzoo3d', 'v4_0_0')

        if vote_threshold is None:
            vote_threshold = 0.2


        filename = glob.glob(galaxyzoo3d_dir+"*{}*.fits.gz".format(self.mangaid))
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


    def get_bar_coords(self, bar_mask = None, flip = False, bar_radius = None, **kwargs):
        """
        Determines bar angle based on min bounding box and returns Coordinate Frame in cylindrical 
        coordinates scaled by the bar_length

        Parameters
        ----------

        bar_mask: `np.array`, optional
            bar_mask boolean array
            if not provided, will attempt to get
        **kwargs:
            passed onto self.get_bar_mask if used
        """

        if bar_mask is None:
            bar_mask = self.get_bar_mask(**kwargs).flatten()

        try:
            assert bar_mask.sum() > 0
        except AssertionError:
            nan_arr = np.full(self.spx_ellcoo_r_re.shape, np.nan)
            bar_coords = CylindricalRepresentation(rho = nan_arr, phi = nan_arr*u.deg, z = nan_arr)
            return bar_coords
            # raise ValueError("bar_mask does not identify any spaxels in the bar!")

        # Define Coordinates in Galaxy Frame
        cyl = CylindricalRepresentation(rho = self.spx_ellcoo_r_re, 
                                 phi = self.spx_ellcoo_elliptical_azimuth * u.deg, 
                                 z = np.zeros_like(self.spx_ellcoo_r_re))
        # Convert to Cartesian in Galaxy Frame
        cart = cyl.to_cartesian()

        bar_x = cart.x.flatten()[bar_mask]
        bar_y = cart.y.flatten()[bar_mask]

        # Determine Bar Angle
        points_cart = np.array([bar_x, bar_y]).T
        bbox = minimum_bounding_rectangle(points_cart)
        xx,yy = bbox[0,:]
        dists = np.array([((xx - xx2)**2 + (yy-yy2)**2) for (xx2,yy2) in bbox])
        args = np.argsort(dists)
        xx2,yy2= bbox[args,:][2,:]
        xx,yy = bbox[args,:][0,:]
        slope = (yy2 - yy) / (xx2 - xx)
        bar_angle = np.arctan2((yy2 - yy), (xx2-xx))


        # C = np.cov(np.vstack([bar_x,
        #            bar_y]))
        # w, v = np.linalg.eig(C)
        # inx = w.argsort()[::-1]
        # w, v = w[inx], v[:, inx]

        # w_12 = w[:2]
        # v_12 = v[:, :2]

        # Determine new r_bar rho-frame
        l_or_w = np.sqrt((bbox[0,0] - bbox[1,0])**2 + (bbox[0,1] - bbox[1,1])**2)
        l_or_w2 = np.sqrt((bbox[1,0] - bbox[3,0])**2 + (bbox[1,1] - bbox[3,1])**2)
        
        if bar_radius is None:
            bar_radius = np.max([l_or_w, l_or_w2])/2.
        new_center_x, new_center_y = bar_x.mean(), bar_y.mean()
        new_x = cart.x - new_center_x
        new_y = cart.y - new_center_y

        new_cart = CartesianRepresentation(x = new_x, y = new_y, z = cart.z)
        new_bar_coords = CylindricalRepresentation.from_cartesian(new_cart)

        med_x, med_y = np.median(bar_x), np.median(bar_y)

        # Check Angle Values and Fix
        # bar_angle = np.arctan2(v_12[0,1],v_12[0,0])
        if (bar_x>med_x).sum() >= (bar_x<med_x).sum():
            if np.median(bar_y[bar_x>med_x]) >med_y:
                try:
                    assert bar_angle > 0
                except AssertionError:
                    bar_angle *= -1
            elif np.median(bar_y[bar_x>med_x]) < med_y:
                try:
                    assert bar_angle < 0
                except AssertionError:
                    bar_angle *= -1
        else:
            if np.median(bar_y[bar_x<med_x]) <med_y:
                try:
                    assert bar_angle > 0
                except AssertionError:
                    bar_angle *= -1
            elif np.median(bar_y[bar_x<med_x]) > med_y:
                try:
                    assert bar_angle < 0
                except AssertionError:
                    bar_angle *= -1

        if (np.abs(bar_angle) > np.pi/2.):
            bar_angle = np.pi - bar_angle

        # Determine New Phi-frame
        new_phi = Angle(new_bar_coords.phi - (bar_angle) * u.rad)

        if flip:
            new_phi *= -1

        
        # bar_radius = np.max(self.spx_ellcoo_r_re.value.flatten()[bar_mask])
        new_rho = new_bar_coords.rho / bar_radius

        bar_coords = CylindricalRepresentation(rho = new_rho, phi = new_phi, z = cyl.z)

        return bar_coords



    def mean_intensity_v_phi(self, map_name, 
                         bin_width_phi = None, step_size_phi = None, 
                         min_rho = None, max_rho = None, 
                         bar_coords = None, estimator = None, 
                         return_errors = False, snr_min = None, 
                         wrap = True,
                         **kwargs):
        """
        Find mean intensity of specified map along bins in azimuth
        
        Parameters
        ----------
        map_name: `str`
            name of map attribute to use
        bin_width_phi: `u.Quantity`, `number`, optional, must be keyword
            width of bins along azimuth angle
            defualt units of deg
        step_size_phi: `u.Quantity`, `number`, optional, must be keyword
            step size along azimuth angle
            defualt units of deg
        min_rho: `number`, optional, must be keyword
            minimum radius to consider 
            default to 1 R_bar
        max_rho: `number`, optional, must be keyword
            maximum radius to consider
            default to 2 R_bar
        bar_coords: `astropy.coordinates.CylindricalRepresentation`, optional, must be keyword
            bar coordinate frame
            if not given, will try to get
        estimator: `str`, optional, must be keyword
            'mean' or 'median'
        return_errors: `bool`, must be keyword
            if True, also returns errors
        snr_min: `number`, optional, must be keyword
            min SNR to use
            default to 3
        wrap: `bool`, optional, must be keyword
            if True, will only consider 0-180 degrees, wrapping
            if False, central_phis span 0 to 360 degrees
        kwargs:
            passed onto get_bar_coords if used
        """
        if bin_width_phi is None:
            bin_width_phi = 10 * u.deg
        elif not hasattr(bin_width_phi, "unit"):
            bin_width_phi *= u.deg
            logging.warning("No units specified for bin_width_phi, assuming u.deg")
        
        if step_size_phi is None:
            step_size_phi = 2.5 * u.deg
        elif not hasattr(step_size_phi, "unit"):
            step_size_phi *= u.deg
            logging.warning("No units specified for step_size_phi, assuming u.deg")
        
        if min_rho is None:
            min_rho = 1.2 
        if max_rho is None:
            max_rho = 2.
            
        if bar_coords is None:
            bar_coords = self.get_bar_coords(**kwargs)
            
        if estimator is None:
            estimator = "mean"
        elif estimator not in ["mean", "median"]:
            estimator = "mean"
            logging.warning("estimator not recognized, using mean")

        if estimator is "mean":
            estimator_function = np.ma.mean
        else:
            estimator_function = np.ma.median
            
        if snr_min is None:
            snr_min = 3.0


        if wrap:    
            central_phi = np.arange(0, 
                                    180,
                                    step_size_phi.to(u.deg).value) * u.deg
        else:
            central_phi = np.arange(0, 
                                    360,
                                    step_size_phi.to(u.deg).value) * u.deg
        
        # Make radial mask
        radial_mask = bar_coords.rho < max_rho
        radial_mask &= bar_coords.rho > min_rho

        # Check map_name
        if map_name is "stellar_mass":
            map_data = self.get_PCA_stellar_mass()
            nan_mask = np.isnan(map_data)
            map_data.mask |= nan_mask
            map_unit = map_data.data.unit
            map_data = np.ma.masked_array(data = map_data.data.value[0,:,:], 
                mask = map_data.mask[0,:,:])
        elif map_name is "smsd":
            map_data = self.get_PCA_stellar_mass_density()
            nan_mask = np.isnan(map_data)
            map_data.mask |= nan_mask
            map_unit = map_data.data.unit
            map_data = np.ma.masked_array(data = map_data.data.value[0,:,:], 
                mask = map_data.mask[0,:,:])
        elif map_name is "Av":
            map_data = self.balmer_Av(snr_min = snr_min)
            map_unit = 1.
        else:
            try:
                # Get Map Attribute to average
                map_data = self.get_map(map_name, snr_min = snr_min)
                map_unit = self.datamodel["emline_gflux_ha"].unit
            except ValueError:
                try:
                    map_data = self.get_PCA_zpres_info(map_name)
                except FileNotFoundError:
                    map_data = np.full(self.get_map("emline_gflux_ha").shape, np.nan)
                    map_data = np.ma.masked_array(data = map_data, mask = np.isnan(map_data))
                if map_data.shape[0] == 3:
                    map_data = map_data[0,:,:]

                map_data = np.ma.masked_array(data= map_data.data, 
                    mask = map_data.mask | np.isnan(map_data.data))
                map_unit = 1.
        
        average_values = np.zeros_like(central_phi.value)
        if estimator is "mean":
            error_values = np.zeros_like(average_values)
        else:
            error_values = np.zeros((len(average_values),2))
            
        for ell, phi in enumerate(central_phi):
            if wrap:
                az_mask = bar_coords.phi.wrap_at("180d") <= phi + bin_width_phi
                az_mask &= bar_coords.phi.wrap_at("180d") > phi - bin_width_phi
                az_mask |= ((bar_coords.phi.wrap_at("180d") <= -180*u.deg - (phi - bin_width_phi)) & 
                            (bar_coords.phi.wrap_at("180d") > -180*u.deg + (phi + bin_width_phi)))

            else:
                az_mask = bar_coords.phi.wrap_at("360d") <= phi + bin_width_phi
                az_mask &= bar_coords.phi.wrap_at("360d") > phi - bin_width_phi
            
            current_mask = az_mask & radial_mask
            
            
            average_values[ell] = estimator_function(map_data[current_mask])
            if (estimator is "mean") & (return_errors):
                error_values[ell] = np.std(map_data[current_mask])
            elif (estimator is "median") & (return_errors):
                error_values[ell,:] = np.percentile(map_data[current_mask].flatten(), (16,84))
                
        if return_errors:
            return average_values * map_unit, central_phi, error_values * map_unit
        else:
            return average_values * map_unit, central_phi
    
    




















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

