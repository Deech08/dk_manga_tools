import logging
import warnings

from marvin.tools.maps import Maps

import astropy.units as u

from astropy.table import Table

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

pal = sns.color_palette('colorblind')

import os

directory = os.path.dirname(__file__)

# Helper Functions for building a bpt diagram ala Krishnarao et al. (2019)

def nii_sf_line(log_nii_ha = None):
    """
    Returns the Star Formation Classication line of Kauffmann+03 (log_oiii_hb)

    Parameters
    ----------
    log_nii_ha: 'list', 'np.array'
        Log base 10 values of nii/ha line ratio in Energy Units along X axis
    """
    if log_nii_ha is None:
        log_nii_ha = np.linspace(-1.5,0.)
    from marvin.utils.dap.bpt import kewley_sf_nii
    return kewley_sf_nii(log_nii_ha)

def nii_comp_line(log_nii_ha = None):
    """
    Returns the Composite Classification line of Kewley+01 (log_oiii_hb)

    Parameters
    ----------
    log_nii_ha: 'list', 'np.array'
        Log base 10 values of nii/ha line ratio in Energy Units along X axis
    """
    if log_nii_ha is None:
        log_nii_ha = np.linspace(-1.5,0.3)
    from marvin.utils.dap.bpt import kewley_comp_nii
    return kewley_comp_nii(log_nii_ha)

def nii_agn_line(log_nii_ha = None):
    """
    Returns the AGN/LI(N)ER Classification line of Schawinski+07 (log_oiii_hb)

    Parameters
    ----------
    log_nii_ha: 'list', 'np.array'
        Log base 10 values of nii/ha line ratio in Energy Units along X axis
    """
    if log_nii_ha is None:
        log_nii_ha = np.linspace(-0.180,1.5)
    return 1.05*log_nii_ha + 0.45

def draw_classification_lines(ax, sf_kwargs = {}, comp_kwargs = {}, agn_kwargs = {}, **kwargs):
    """
    Draw classification lines onto nii_ha BPT Diagram

    Parameters
    ----------
    ax: 'matplotlib.pyplot.figure.axes'
        axes to plot lines on
    sf_kwargs: 'dict', optional, must be keyword
        kwargs to pass to ax.plot for the SF Classification Line
    comp_kwargs: 'dict', optional, must be keyword
        kwargs to pass to ax.plot for the COMP Classification Line
    agn_kwargs: 'dict', optional, must be keyword
        kwargs to pass to ax.plot for the AGN/LI(NER Classification Line
    **kwargs: 'dict', optional, must be keyword
        universal line kwargs passed to all
        If kwarg is set to a 3 element array, then they are each passed to 
        SF, Comp, AGN, in that order
    """

    # Check kwargs:

    # Default zorder
    if "zorder" not in kwargs:
        kwargs["zorder"] = 0

    for keyword in kwargs:
        if (kwargs[keyword].__class__ is tuple) | ((kwargs[keyword].__class__ is list)):
            # Should have 3 entries
            if len(kwargs[keyword]) is 3:
                sf_kwargs[keyword], comp_kwargs[keyword], agn_kwargs[keyword] = kwargs[keyword]
            if len(kwargs[keyword]) is 1:
                if keyword not in sf_kwargs:
                    sf_kwargs[keyword] = kwargs[keyword]
                if keyword not in comp_kwargs:
                    comp_kwargs[keyword] = kwargs[keyword]
                if keyword not in agn_kwargs:
                    agn_kwargs[keyword] = kwargs[keyword]
        else:
            if keyword not in sf_kwargs:
                sf_kwargs[keyword] = kwargs[keyword]
            if keyword not in comp_kwargs:
                comp_kwargs[keyword] = kwargs[keyword]
            if keyword not in agn_kwargs:
                agn_kwargs[keyword] = kwargs[keyword]

    # Default colors:
    if "color" not in sf_kwargs:
        sf_kwargs["color"] = pal[1]
    if "color" not in comp_kwargs:
        comp_kwargs["color"] = pal[0]
    if "color" not in agn_kwargs:
        agn_kwargs["color"] = pal[-1]

    # Default Labels:
    if "label" not in sf_kwargs:
        sf_kwargs["label"] = "Kauffmann+03"
    if "label" not in comp_kwargs:
        comp_kwargs["label"] = "Kewley+01"
    if "label" not in agn_kwargs:
        agn_kwargs["label"] = "Schawinski+07"



    # SF Line
    x = np.linspace(-1.5,0.)
    ax.plot(x, nii_sf_line(x), **sf_kwargs)

    # Comp Line
    x = np.linspace(-1.5, 0.3)
    ax.plot(x, nii_comp_line(x), **comp_kwargs)

    # AGN / LI(N)ER Line
    x = np.linspace(-0.180,1.5)
    ax.plot(x, nii_agn_line(x), **agn_kwargs)

    return ax

def plot_mw_tilted_disk(ax, error_bar_kwargs = {}, arrow_kwargs = {}, **kwargs):
    """
    Plots data on bpt diagram from Tilted Disk in Milky Way ala Krishnarao+19

    Parameters
    ----------
    ax 'matplotlib.pyplot.figure.axes'
        axes to plot lines on
    error_bar_kwargs: 'dict', optional, must be keyword
        kwargs passed to plotting error bars
    arrow_kwargs: 'dict', optional, must be keyword
        kwargs passed to ax.arrow for upper limit error bars
    **kwargs: 'dict', optional, must be keywords
        passed to ax.scatter for data points
    """
    # Default point colors
    if ("c" not in kwargs) & ("color" not in kwargs) & ("cmap" not in kwargs):
        kwargs["color"] = pal[2]
    if ("c" not in error_bar_kwargs) & ("color" not in error_bar_kwargs):
        error_bar_kwargs["c"] = kwargs["color"]

    # Default zorder
    if "zorder" not in kwargs:
        if "zorder" in error_bar_kwargs:
            kwargs["zorder"] = error_bar_kwargs["zorder"] + 1
        else:
            kwargs["zorder"] = 2
            error_bar_kwargs["zorder"] = 1
    else:
        if "zorder" not in error_bar_kwargs:
            error_bar_kwargs["zorder"] = kwargs["zorder"] - 1




    # Default Size
    if "s" not in kwargs:
        kwargs["s"] = 75

    # Default error bar line width
    if "lw" not in error_bar_kwargs:
        error_bar_kwargs["lw"] = 1
    if ("alpha" in kwargs) and ("alpha" not in error_bar_kwargs):
        error_bar_kwargs["alpha"] = kwargs["alpha"]

    if ("alpha" in kwargs) and ("alpha" not in arrow_kwargs):
        arrow_kwargs["alpha"] = kwargs["alpha"]

        # Same zorder and color for arrow
    arrow_kwargs["zorder"] = error_bar_kwargs["zorder"]
    arrow_kwargs["color"] = error_bar_kwargs["c"]


    # Default arrow width and head width
    if "width" not in arrow_kwargs:
        arrow_kwargs["width"] = 0.001
    if "head_width" not in arrow_kwargs:
        arrow_kwargs["head_width"] = 0.03

    # Default Label
    if "label" not in kwargs:
        kwargs["label"] = "Tilted Disk"

    # Load Data

    bpt_data_filepath = os.path.join(directory, "mw_data/WHAM_BPT_DATA_021119.fits")
    bpt_data = Table.read(bpt_data_filepath)

    # Error Bars
    for ell in range(len(bpt_data["log_nii_ha"])):
        ax.plot([bpt_data["log_nii_ha_lower"][ell], bpt_data["log_nii_ha_upper"][ell]], 
                [bpt_data["log_oiii_hb"][ell], bpt_data["log_oiii_hb"][ell]], 
                **error_bar_kwargs)
        y_axis_errs = [bpt_data["log_oiii_hb_upper"][ell], bpt_data["log_oiii_hb_lower"][ell]]
        if np.isnan(y_axis_errs[1]):
            position = [bpt_data["log_nii_ha"][ell], bpt_data["log_oiii_hb_upper"][ell], 
                       0., 
                        -2 * (bpt_data["log_oiii_hb_upper"][ell] - bpt_data["log_oiii_hb"][ell])]
            ax.arrow(position[0], position[1], position[2], position[3],
                    **arrow_kwargs)
        else:
            ax.plot([bpt_data["log_nii_ha"][ell], bpt_data["log_nii_ha"][ell]], 
                    y_axis_errs, 
                    **error_bar_kwargs)

    # Data Points
    mws = ax.scatter(bpt_data["log_nii_ha"], bpt_data["log_oiii_hb"], **kwargs)

    return mws, ax

def plot_mw_scutum_cloud(ax, **kwargs):
    """
    Plots data on bpt diagram from Scutum Star Cloud in Milky Way ala Krishnarao+19

    Parameters
    ----------
    ax 'matplotlib.pyplot.figure.axes'
        axes to plot lines on
    **kwargs: 'dict', optional, must be keywords
        passed to ax.scatter for data points
    """

    # Default point colors
    if ("c" not in kwargs) & ("color" not in kwargs) & ("cmap" not in kwargs):
        kwargs["color"] = pal[2]

    # Default marker
    if "marker" not in kwargs:
        kwargs["marker"] = 'P'

    # Default zorder
    if "zorder" not in kwargs:
        kwargs["zorder"] = 2

    # Default Size
    if "s" not in kwargs:
        kwargs["s"] = 75

    # Default label
    if "label" not in kwargs:
        kwargs["label"] = "Scutum Cloud"

    # Load Data

    scutum_data_filepath = os.path.join(directory, "mw_data/MadsenScutumRatios.fits")
    scutum_data = Table.read(scutum_data_filepath)

    mads = ax.scatter(np.log10(scutum_data["nii_ha"]), np.log10(scutum_data["oiii_hb"]), 
                  **kwargs)

    return mads, ax

def plot_mw_upper_feature(ax, error_bar_kwargs = {}, arrow_kwargs = {}, **kwargs):
    """
    Plots data on bpt diagram from Upper Feature in Milky Way ala Krishnarao+19

    Parameters
    ----------
    ax 'matplotlib.pyplot.figure.axes'
        axes to plot lines on
    **kwargs: 'dict', optional, must be keywords
        passed to ax.scatter for data points
    """

    # Default point colors
    if ("c" not in kwargs) & ("color" not in kwargs) & ("cmap" not in kwargs):
        kwargs["color"] = pal[2]

    # Default marker
    if "marker" not in kwargs:
        kwargs["marker"] = '*'

    # Default zorder
    if "zorder" not in kwargs:
        kwargs["zorder"] = 2

    # Default Size
    if "s" not in kwargs:
        kwargs["s"] = 75

    # Default label
    if "label" not in kwargs:
        kwargs["label"] = "Upper Feature"

    # Default error bar line width
    if "lw" not in error_bar_kwargs:
        error_bar_kwargs["lw"] = 1

    # Default arrow width and head width
    if "width" not in arrow_kwargs:
        arrow_kwargs["width"] = 0.001
    if "head_width" not in arrow_kwargs:
        arrow_kwargs["head_width"] = 0.03

    if ("c" not in error_bar_kwargs) & ("color" not in error_bar_kwargs):
        error_bar_kwargs["c"] = kwargs["color"]

    if ("alpha" in kwargs) and ("alpha" not in error_bar_kwargs):
        error_bar_kwargs["alpha"] = kwargs["alpha"]

    # Load Data

    upper_data_filepath = os.path.join(directory, "mw_data/BPT_DATA_MW_UPPER.fits")
    upper_data = Table.read(upper_data_filepath)

    upper_data_nohi_filepath = os.path.join(directory, "mw_data/NOHI_BPT_DATA.fits")
    upper_nohi_data = Table.read(upper_data_nohi_filepath)

    xx = [upper_data["log_nii_ha"][:], upper_nohi_data["log_nii_ha"][:]]
    yy = [upper_data["log_oiii_hb"][:], upper_nohi_data["log_oiii_hb"][:]]

    uppers = ax.scatter(upper_data["log_nii_ha"], upper_data["log_oiii_hb"], **kwargs)
    kwargs["label"] = None
    uppers = ax.scatter(upper_nohi_data["log_nii_ha"], upper_nohi_data["log_oiii_hb"], **kwargs)

    # Error Bars
    bpt_data = upper_data
    for ell in range(len(bpt_data["log_nii_ha"])):
        ax.plot([bpt_data["log_nii_ha_lower"][ell], bpt_data["log_nii_ha_upper"][ell]], 
                [bpt_data["log_oiii_hb"][ell], bpt_data["log_oiii_hb"][ell]], 
                **error_bar_kwargs)
        y_axis_errs = [bpt_data["log_oiii_hb_upper"][ell], bpt_data["log_oiii_hb_lower"][ell]]
        if np.isnan(y_axis_errs[1]):
            position = [bpt_data["log_nii_ha"][ell], bpt_data["log_oiii_hb_upper"][ell], 
                       0., 
                        -2 * (bpt_data["log_oiii_hb_upper"][ell] - bpt_data["log_oiii_hb"][ell])]
            ax.arrow(position[0], position[1], position[2], position[3],
                    **arrow_kwargs)
        else:
            ax.plot([bpt_data["log_nii_ha"][ell], bpt_data["log_nii_ha"][ell]], 
                    y_axis_errs, 
                    **error_bar_kwargs)

    # Error Bars
    bpt_data = upper_nohi_data
    for ell in range(len(bpt_data["log_nii_ha"])):
        ax.plot([bpt_data["log_nii_ha_lower"][ell], bpt_data["log_nii_ha_upper"][ell]], 
                [bpt_data["log_oiii_hb"][ell], bpt_data["log_oiii_hb"][ell]], 
                **error_bar_kwargs)
        y_axis_errs = [bpt_data["log_oiii_hb_upper"][ell], bpt_data["log_oiii_hb_lower"][ell]]
        if np.isnan(y_axis_errs[1]):
            position = [bpt_data["log_nii_ha"][ell], bpt_data["log_oiii_hb_upper"][ell], 
                       0., 
                        -2 * (bpt_data["log_oiii_hb_upper"][ell] - bpt_data["log_oiii_hb"][ell])]
            ax.arrow(position[0], position[1], position[2], position[3],
                    **arrow_kwargs)
        else:
            ax.plot([bpt_data["log_nii_ha"][ell], bpt_data["log_nii_ha"][ell]], 
                    y_axis_errs, 
                    **error_bar_kwargs)


    return uppers, ax

def plot_mw_nii_bars(ax, snr_min = None, shaded_kwargs = {}, **kwargs):
    """
    Plots vertical lines and bars on bpt Diagram for Tilted Disk where only 
    NII/HA line is detected

    Parameters
    ----------
    ax 'matplotlib.pyplot.figure.axes'
        axes to plot lines on
    snr_min: 'number', optional, must be keyword
        Miniumum sigma detection level to plot
        Default of 2 sigma
    shaded_kwargs: 'dict', optional, must be keyword
        kwargs passed to ax.fill_betweenx for shaded error boxes
    **kwargs: 'dict', optional, must be keywords
        passed to ax.plot for lines
    """

    # Default line color
    if "color" not in kwargs:
        kwargs["color"] = pal[4]
    if "facecolor" not in shaded_kwargs:
        shaded_kwargs["facecolor"] = kwargs["color"]

    # Default zorder
    if "zorder" not in kwargs:
        if "zorder" in shaded_kwargs:
            kwargs["zorder"] = shaded_kwargs["zorder"] + 1
        else:
            kwargs["zorder"] = 2
            shaded_kwargs["zorder"] = 1
    else:
        if "zorder" not in shaded_kwargs:
            shaded_kwargs["zorder"] = kwargs["zorder"] - 1


    # Default alpha
    if "alpha" not in kwargs:
        kwargs["alpha"] = 0.5
    if "alpha" not in shaded_kwargs:
        shaded_kwargs["alpha"] = 0.1

    # Default line style
    if "ls" not in kwargs:
        kwargs["ls"] = ':'

    # Default line width
    if "lw" not in kwargs:
        kwargs["lw"] = 2

    # Default SNR
    if snr_min is None:
        snr_min = 2.

    # Default Label
    if "label" not in kwargs:
        kwargs["label"] = r'$>{0:2.1f}\sigma$ Tilted Disk'.format(snr_min)

    # Load Data
    nii_ha_data_filepath = os.path.join(directory, "mw_data/WHAM_NII_HA_DATA_021219.fits")
    nii_ha_data = Table.read(nii_ha_data_filepath)

    # SNR Cut Data
    snr_cut = (nii_ha_data["NII_SIGMA_LEVEL"] > snr_min) & (nii_ha_data["HA_SIGMA_LEVEL"] > snr_min)
    # in Tilted Disk OIII/HB points cut
    oiii_hb_cut = np.ones(len(nii_ha_data), dtype = bool)
    oiii_hb_cut[0] = False
    oiii_hb_cut[6] = False
    oiii_hb_cut[7] = False
    nii_ha_data = nii_ha_data[snr_cut & oiii_hb_cut]

    for ell, entry in enumerate(nii_ha_data):
        if ell == 0:
            ax.plot([entry["log_nii_ha"], entry["log_nii_ha"]], [-2-ell/3.,2], 
                **kwargs)
            del kwargs["label"]
        else:
            ax.plot([entry["log_nii_ha"], entry["log_nii_ha"]], [-2-ell/3.,2], 
                **kwargs)
        ax.fill_betweenx([-2,2], 
                         [entry["log_nii_ha_lower"], entry["log_nii_ha_lower"]], 
                         x2 = [entry["log_nii_ha_upper"], entry["log_nii_ha_upper"]], 
                         **shaded_kwargs)

    return ax

def plot_dkbpt(ax, 
               tilted_disk = True, 
               scutum_cloud = True, 
               nii_bars = True, 
               upper_feature = True,
               tilted_disk_kwargs = {}, 
               scutum_cloud_kwargs = {}, 
               nii_bars_kwargs = {}, 
               upper_feature_kwargs = {},
               classification_kwargs = {},
               **kwargs):
    """
    Plots Default bpt diagram of Milky Way Data ala Krishnarao+17

    Parameters
    ----------
    ax 'matplotlib.pyplot.figure.axes'
        axes to plot lines on
    tilted_disk: 'bool', optional, must be keyword
        if True, includes Milky Way Tilted Disk Points
    scutum_cloud: 'bool', optional, must be keyword
        if True, includes Milky Way Scutum Star Cloud Points
    nii_bars: 'bool', optional, must be keyword
        if True, includes NII/Ha Tilted Disk Vertical Bars
    upper_feature: 'bool', optional, must be keyword
        if True, includes Upper Feature points
    tilted_disk_kwargs: 'dict', optional, must be keyword
        kwargs passed to plot_mw_tilted_disk
    scutum_cloud_kwargs: 'dict', optional, must be keyword
        kwargs passed to plot_mw_scutum_cloud
    nii_bars_kwargs: 'dict', optional, must be keyword
        kwargs passed to plot_mw_nii_bars
    upper_feature_kwargs: 'dict', optional, must be keyword
        kwargs passed to plot_mw_upper_feature
    classification_kwargs: 'dict', optional, must be keyword
        kwargs passed to draw_classification_lines
    kwargs: 'dict', optional, must be keyword
        kwargs passed to tilted_disk and scutum cloud if not in them already
    """
    ax = draw_classification_lines(ax, **classification_kwargs)

    for keyword in kwargs:
        if keyword not in tilted_disk_kwargs:
            tilted_disk_kwargs[keyword] = kwargs[keyword]
        if keyword not in scutum_cloud_kwargs:
            scutum_cloud_kwargs[keyword] = kwargs[keyword]

    if tilted_disk:
        mws, ax = plot_mw_tilted_disk(ax, **tilted_disk_kwargs)

    if scutum_cloud:
        mads, ax = plot_mw_scutum_cloud(ax, **scutum_cloud_kwargs)

    if nii_bars:
        ax = plot_mw_nii_bars(ax, **nii_bars_kwargs)

    if upper_feature:
        uppers, ax = plot_mw_upper_feature(ax, **upper_feature_kwargs)

    ax = scale_to_dkbpt(ax)

    ax.set_xlabel(r'$log_{10}$([NII] $\lambda$ 6583/H$\alpha$)',fontsize=12)
    ax.set_ylabel(r'$log_{10}$([OIII] $\lambda$ 5007/H$\beta$)',fontsize=12)

    return ax



def scale_to_dkbpt(ax):
    """
    Scales x and y axis to match Krishnarao+19 BPT Diagram

    Parameters
    ----------
    ax 'matplotlib.pyplot.figure.axes'
        axes to plot lines on
    """
    Xmin, Xmax         = -1.2, 1.0
    Ymin, Ymax         = -1.2, 1.0
    ax.set_xlim([Xmin, Xmax])
    ax.set_ylim([Ymin, Ymax])

    ax.set_aspect("equal")
    return ax

def bpt_nii(maps, ax = None, 
            snr_min = None,
            deredden = False, 
            return_figure = True,
            plot_map = False,
            plot_kde = False,
            radial_bin = None,
            return_data = None,
            overplot_dk_bpt = False,
            dk_bpt_kwargs = {},
            deredden_kwargs = {},
            classification_kwargs = {}, 
            **kwargs):
    """
    NII / HA BPT Diagram ala Krishnarao+19 for MaNGA Galaxy

    Parameters
    ----------
    maps: 'marvin.tools.maps.Maps' or 'dk_marvin_tools.DKMaps.DKMaps'
        MaNGA Maps Data
    ax: 'matplotlib.pyplot.figure.axes', optional, must be keyword
        Matplotlib axes to plot on
    snr_min: 'number', optional, must be keyword
        min SNR to use for data
    deredden: 'bool', optional, must be keyword
        if True, will deredden emission lines
    return_figure: 'bool', optional, must be keyword
        if False, will return BPT Classifications as a dictionary
    plot_map: 'bool', optional, must be keyword
        if True, will instead plot a map color coded by classifications
    plot_kde: 'bool', optional, must be keyword
        if True, will plot kde plot instead of scatter plot
    radial_bin: 'list' or 'tuple' or 'np.ndarray', optional, must be keyword
        if given, only plots points within provided radial bin in terms of R/R_e
    return_data: 'bool', optional, must be keyword
        if True, returns data instead of plotting
    overplot_dk_bpt: 'bool', optional, must be keyword
        if True, overplots dk_bpt
    dk_bpt_kwargs: 'dict', optional, must be keyword
        kwargs passed to dk_bpt
    deredden_kwargs: 'dict', optional, must be keyword
        kwargs passed to deredden
    classification_kwargs: 'dict', optional, must be keyword
        kwargs passed to draw_classification_lines
    kwargs: 'dict', optional, must be keywords
        keywords to pass to scatter plot of BPT points
        or keywords to pass to map.plot of plot_map
    """
    # Get EmLine Data
    if deredden:
        try:
            ha = maps.deredden("emline gflux ha", **deredden_kwargs)
        except AttributeError:
            logging.warning("provided maps object does not have a deredden method. Skipping dereddening process.")
            ha = maps["emline gflux ha"]
            hb = maps["emline gflux hb"]
            oiii = maps["emline glfux oiii 5007"]
            nii = maps["emline glflux nii 6585"]
        else:
            hb = maps.deredden("emline glfux hb", **deredden_kwargs)
            oiii = maps.deredden("emline glfux oiii 5007", **deredden_kwargs)
            nii = maps.deredden("emline glfux nii 6585", **deredden_kwargs)
    else:
        ha = maps["emline gflux ha"]
        hb = maps["emline gflux hb"]
        oiii = maps["emline glfux oiii 5007"]
        nii = maps["emline glflux nii 6585"]

    if snr_min is None:
        snr_min = 3.

    # Get masked Data
    ha_masked = ha.masked
    hb_masked = hb.masked
    oiii_masked = oiii.masked
    nii_masked = nii.masked

    # SNR Cut
    ha_masked.mask |= ha.snr < snr_min
    hb_masked.mask |= hb.snr < snr_min
    oiii_masked.mask |= oiii.snr < snr_min
    nii_masked.mask |= nii.snr < snr_min

    ha_masked.mask |= ha.ivar == 0
    hb_masked.mask |= hb.ivar == 0
    oiii_masked.mask |= oiii.ivar == 0
    nii_masked.mask |= nii.ivar == 0

    # Mask Negative Flux
    ha_masked.mask |= ha_masked.data <= 0
    hb_masked.mask |= hb_masked.data <= 0
    oiii_masked.mask |= oiii_masked.data <= 0
    nii_masked.mask |= nii_masked.data <= 0

    # masked Logarithms
    log_oiii_hb = np.ma.log10(oiii_masked / hb_masked)
    log_nii_ha = np.ma.log10(nii_masked / ha_masked)

    # Calculate Masks for classification regions
    sf_mask_nii = (log_oiii_hb < nii_sf_line(log_nii_ha)) & (log_nii_ha < 0.05)

    comp_mask = ((log_oiii_hb > nii_sf_line(log_nii_ha)) & (log_nii_ha < 0.05))  & \
                ((log_oiii_hb < nii_comp_line(log_nii_ha)) & (log_nii_ha < 0.465))

    sub_agn_mask_nii = (log_oiii_hb > nii_comp_line(log_nii_ha)) | (log_nii_ha > 0.465)

    agn_mask_nii = sub_agn_mask_nii & (nii_agn_line(log_nii_ha) < log_oiii_hb)

    liner_mask_nii = sub_agn_mask_nii & (nii_agn_line(log_nii_ha) > log_oiii_hb)

    invalid_mask = ha_masked.mask | oiii_masked.mask | nii_masked.mask | hb_masked.mask

    sf_classification = {"nii": sf_mask_nii}
    comp_classification = {"nii": comp_mask}
    agn_classification = {"nii": agn_mask_nii}
    liner_classification = {"nii": liner_mask_nii}
    invalid_classification = {"nii": invalid_mask}

    bpt_return_classification = {'sf': sf_classification,
                                 'comp': comp_classification,
                                 'agn': agn_classification,
                                 'liner': liner_classification,
                                 'invalid': invalid_classification}

    if not return_figure:
        return bpt_return_classification
    elif plot_map:
        # Make image
        bpt_image = np.empty(ha.shape)
        bpt_image[:] = np.nan
        # Star Forming
        bpt_image[bpt_return_classification['sf']['nii']] = 0.5
        # Comp
        bpt_image[bpt_return_classification['comp']['nii']] = 1.5
        # Seyfert
        bpt_image[bpt_return_classification['agn']['nii']] = 2.5
        # LINER
        bpt_image[bpt_return_classification['liner']['nii']] = 3.5
        # Ambiguous
        bpt_image[bpt_return_classification['invalid']['nii']] = 4.5
        
        bpt_image = np.ma.masked_array(bpt_image, 
                                       mask = np.isnan(bpt_image))


        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure


        if "cmap" not in kwargs:
            kwargs["cmap"] = ListedColormap(sns.color_palette([pal[1], pal[0], pal[9], pal[4], pal[8]]))
        if "title" not in kwargs:
            kwargs["title"] = "NII BPT Classification Map"


        fig, ax, cb = ha.plot(fig = fig, ax = ax, value = bpt_image, 
            return_cb = True, cbrange = [0,5], **kwargs)

        cb.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5])
        cb.ax.set_yticklabels(['Star Forming', 
                                 'Composite', 
                                 'AGN', 
                                 'LI(N)ER', 
                                 'Invalid'])
        cb.set_label(r'BPT Classification', fontsize = 14)
        cb.ax.tick_params(labelsize=12) 

        return ax, cb

    elif plot_kde:
        # KDE Map 
        if return_data:
            radius = maps['spx ellcoo r_re']
            return radius, log_nii_ha, log_oiii_hb
        else:

            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111)

            # Draw classification lines
            ax = draw_classification_lines(ax, **classification_kwargs)

            # Default kwargs
            # Default colormap
            if "cmap" not in kwargs:
                kwargs["cmap"] = "plasma"

            if "zorder" not in kwargs:
                kwargs["zorder"] = 0

            if "shade" not in kwargs:
                kwargs["shade"] = True

            if "shade_lowest" not in kwargs:
                kwargs["shade_lowest"] = False

            if radial_bin is not None:
                radius = maps['spx ellcoo r_re']
                within = radius.value >= radial_bin[0]
                within &= radius.value <= radial_bin[1]

                ax = sns.kdeplot(log_nii_ha[(np.invert(log_nii_ha.mask | log_oiii_hb.mask) & within)], 
                                 log_oiii_hb[(np.invert(log_nii_ha.mask | log_oiii_hb.mask) & within)], 
                                 ax = ax, **kwargs)
            else:    
                ax = sns.kdeplot(log_nii_ha[np.invert(log_nii_ha.mask | log_oiii_hb.mask)], 
                                 log_oiii_hb[np.invert(log_nii_ha.mask | log_oiii_hb.mask)], 
                                 ax = ax, **kwargs)

            ax.set_xlabel(r'$log_{10}$([NII] $\lambda$ 6585/H$\alpha$)',fontsize=12)
            ax.set_ylabel(r'$log_{10}$([OIII] $\lambda$ 5007/H$\beta$)',fontsize=12)

            if overplot_dk_bpt:
                ax = plot_dkbpt(ax, **dk_bpt_kwargs)

            ax = scale_to_dkbpt(ax)

            return ax



    else:
        # Do the plotting

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        # Draw classification lines
        ax = draw_classification_lines(ax, **classification_kwargs)

        # Defautl kwargs
        # Default colors
        if ("c" not in kwargs) & ("color" not in kwargs) & ("cmap" not in kwargs):
            # Default radial colormapping
            radius = maps['spx ellcoo r_re']
            kwargs["c"] = radius
            kwargs["cmap"] = sns.dark_palette(pal[8], as_cmap=True)

        # Default vmin/vmax
        if "c" in kwargs:
            if "vmin" not in kwargs:
                kwargs["vmin"] = 0.
            if "vmax" not in kwargs:
                kwargs["vmax"] = 3.

        # Default Size
        if "s" not in kwargs:
            kwargs["s"] = 5

        # plot the points
        pts = ax.scatter(log_nii_ha, log_oiii_hb, **kwargs)

        ax.set_xlabel(r'$log_{10}$([NII] $\lambda$ 6585/H$\alpha$)',fontsize=12)
        ax.set_ylabel(r'$log_{10}$([OIII] $\lambda$ 5007/H$\beta$)',fontsize=12)

        if overplot_dk_bpt:
            ax = plot_dkbpt(ax, **dk_bpt_kwargs)

        ax = scale_to_dkbpt(ax)

        return pts, ax


















