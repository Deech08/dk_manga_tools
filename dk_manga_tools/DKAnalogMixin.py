import logging

from marvin.tools.maps import Maps
from marvin.tools.image import Image
import astropy.units as u
from astropy.cosmology import WMAP9

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
pal = sns.color_palette('colorblind', 10)

from .DKMaps import DKMaps

from astropy.table.row import Row

from marvin import config

import os
import warnings


class DKSamplePlotter():
    """
    Class to interactivly plot MaNGA DK Sample Data
    Click on an image in the main frame - plots below are updated to correspond that image / galaxy

    Rescale map image using scaling plot point to click on new min/max values
    Plot different map images using bottom table to click on map options

    Parameters
    ----------

    fig: 'matplotlib.figure' 
        figure containing maps
    lgs:   'matplotlib.gridspec'
        figure gridspec instance to draw on
    data:   'DK_MWAnalogs'
        MW_Analogs class object containing dk_sample
    scale_ax: 'matplotlib.figure.axes'
        axes instance for the scaling min/max clicker
    map_ax: 'matplotlib.figure.axes'
        axes instance for the map selection clicker
    snr_min: 'number'
        min SNR Threshold for maps and points

    """
    def __init__(self, fig, gs, data = None, scale_ax = None, map_ax = None, snr_min = None):
        self.fig = fig
        self.gs = gs
        self.data = data
        self.scale_ax = scale_ax
        self.new_scale = (0,10)
        self.map_ax = map_ax
        self.AV = False
        self.line = "emline gflux ha"
        self.images = Image.from_list(self.data.dk_sample["PLATEIFU"])
        self.snr_min = snr_min
        if not data.sersic:
            self.n_gal = 21
            self.image_axes = self.fig._get_axes()[0:self.n_gal]
            self.big_image_gs = gs[6:8, 0:2]
            self.bpt_map_gs = gs[5:8, 2:]
            self.radial_nii_gs = gs[8:10, 0:2]
            self.violin_gs = gs[8:10, 2:]
            self.bpt_nii_gs = gs[10:13, 1:]
            self.radial_oiii_gs = gs[10:13,0]
            self.map_gs = gs[14:17, 2:]
            self.scale_gs = gs[14:17, 0]
        else:
            self.n_gal = 14
            self.image_axes = self.fig._get_axes()[0:self.n_gal]
            self.big_image_gs = gs[4:6, 0:2]
            self.bpt_map_gs = gs[3:6, 2:]
            self.radial_nii_gs = gs[6:8, 0:2]
            self.violin_gs = gs[6:8, 2:]
            self.bpt_nii_gs = gs[8:11, 1:]
            self.radial_oiii_gs = gs[8:11,0]
            self.map_gs = gs[12:15, 2:]
            self.scale_gs = gs[12:15, 0]

        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def on_click(self, event):
        if event.button is 1: # left mouse click
            if event.inaxes not in self.image_axes: 
                if event.inaxes is self.scale_ax:
                    # Re scale map image
                    self.old_scale = self.new_scale[:]
                    self.new_scale = [event.xdata, event.ydata]
                    if self.new_scale[0] < 0.:
                        if self.new_scale[1] < 0.:
                            self.new_scale = [0,10]
                        else:
                            self.new_scale[0] = 0.

                    axes = self.fig._get_axes()[-3:]
                    for axi in axes:
                        axi.remove()
                    
                    ax = self.fig.add_subplot(self.map_gs)
                    line = self.maps[self.line]
                    if self.AV:
                        ax = line.plot(fig = self.fig, ax = ax, 
                                            cbrange = self.new_scale,
                                            return_cb = True, 
                                            sky_coords = True,
                                            snr_min = self.snr_min,
                                            title = r'$A_V$',
                                            patch_kws = {'hatch':'xx', 
                                                         'facecolor':'white', 
                                                         'edgecolor':'grey'}, 
                                            cmap = "Reds", value = self.AV_value
                                      )
                    else:
                        ax = line.plot(fig = self.fig, ax = ax, 
                                            cbrange = self.new_scale,
                                            return_cb = True, 
                                            sky_coords = True, 
                                            snr_min = self.snr_min,
                                            patch_kws = {'hatch':'xx', 
                                                         'facecolor':'white', 
                                                         'edgecolor':'grey'}, 
                                            cmap = "Reds"
                                      )

                    self.scale_ax = self.fig.add_subplot(self.scale_gs)
                    self.scale_ax.set_xlim([-1,11])
                    self.scale_ax.set_ylim((-1,31))
                    self.scale_ax.scatter([self.new_scale[0]], [self.new_scale[1]])
                    self.scale_ax.set_xlabel("Min Color Scale")
                    self.scale_ax.set_ylabel("Max Color Scale")
                    
                    self.fig.canvas.draw()
                elif event.inaxes is self.map_ax:
                    # Re draw map image
                    click_point_x, click_point_y = event.xdata, event.ydata
                    if (click_point_x < 0.2) & (click_point_x > 0):
                        # Ha
                        self.line = 'emline gflux ha'
                        self.AV = False
                        if (click_point_y > 0.5):
                            line = self.maps['emline gflux ha']
                        else:
                            line = self.maps.deredden('emline gflux ha', snr_min = self.snr_min)
                    elif (click_point_x < 0.4) & (click_point_x > 0.2):
                        # Hb
                        self.line = 'emline gflux hb'
                        self.AV = False
                        if (click_point_y > 0.5):
                            line = self.maps['emline gflux hb']
                        else:
                            line = self.maps.deredden('emline gflux hb', snr_min = self.snr_min)
                    elif (click_point_x < 0.6) & (click_point_x > 0.4):
                        # NII
                        self.line = 'emline gflux nii 6585'
                        self.AV = False
                        if (click_point_y > 0.5):
                            line = self.maps['emline gflux nii 6585']
                        else:
                            line = self.maps.deredden('emline gflux nii 6585', snr_min = self.snr_min)
                    elif (click_point_x < 0.8) & (click_point_x > 0.6):
                        # OII
                        self.line = 'emline gflux oii 3727'
                        self.AV = False
                        if (click_point_y > 0.5):
                            line = self.maps['emline gflux oii 3727']
                        else:
                            line = self.maps.deredden('emline gflux oii 3727', snr_min = self.snr_min)
                        
                    elif (click_point_x < 1.) & (click_point_x > 0.8):
                        # AV
                        self.AV = True
                        self.line = 'emline gflux ha'
                        self.AV_value = self.maps.balmer_Av(snr_min = self.snr_min)
                        line = self.maps["emline gflux ha"]
                    
                    axes = self.fig._get_axes()[-3:]
                    for axi in axes:
                        axi.remove()
                    
                    ax = self.fig.add_subplot(self.map_gs)
                    if self.AV:
                        ax = line.plot(fig = self.fig, ax = ax, 
                                            cbrange = self.new_scale,
                                            return_cb = True, 
                                            sky_coords = True,
                                            snr_min = self.snr_min,
                                            title = r'$A_V$',
                                            patch_kws = {'hatch':'xx', 
                                                         'facecolor':'white', 
                                                         'edgecolor':'grey'}, 
                                            cmap = "Reds", value = self.AV_value
                                      )
                    else:
                        ax = line.plot(fig = self.fig, ax = ax, 
                                            cbrange = self.new_scale,
                                            return_cb = True, 
                                            snr_min = self.snr_min,
                                            sky_coords = True, 
                                            patch_kws = {'hatch':'xx', 
                                                         'facecolor':'white', 
                                                         'edgecolor':'grey'}, 
                                            cmap = "Reds"
                                      )

                    self.scale_ax = self.fig.add_subplot(self.scale_gs)
                    self.scale_ax.set_xlim([-1,11])
                    self.scale_ax.set_ylim((-1,31))
                    self.scale_ax.scatter([self.new_scale[0]], [self.new_scale[1]])
                    self.scale_ax.set_xlabel("Min Color Scale")
                    self.scale_ax.set_ylabel("Max Color Scale")
                    
                    self.fig.canvas.draw()
                    
                else:
                    return # if click is not on image, nothing happens
                    
            else:
                axes_match = [self.fig._get_axes()[i] is event.inaxes for i in range(15)]
                ell = np.where(axes_match)[0][0]
                
                axes = self.fig._get_axes()[-11:]
                for axi in axes:
                    axi.remove()

                
                im = self.images[ell]
                ax = self.fig.add_subplot(self.big_image_gs, projection = im.wcs)
                ax.imshow(im.data, origin = 'lower')
                im.overlay_hexagon(ax, color=pal[1], linewidth=1)
                ax.grid(False)
                ax.set_title(self.data.dk_sample["PLATEIFU"][ell])
                
                self.maps = DKMaps(plateifu = self.data.dk_sample["PLATEIFU"][ell])
                
                ax = self.fig.add_subplot(self.radial_nii_gs)
                ax = self.maps.plot_radial_emline(["emline gflux nii 6585", "emline gflux ha"], s = 5, ax = ax, 
                                 log10 = True, c = self.maps['spx ellcoo r_re'].flatten(), cmap = 'viridis', snr_min = self.snr_min)
                

                ax = self.fig.add_subplot(self.violin_gs)
                ax = self.maps.plot_violin_bpt_nii(inner = 'quartile', alpha = 0.2, ax = ax, snr_min = self.snr_min)
                ax = self.maps.plot_violin_bpt_nii(inner = None, scale = 'count', ax = ax, snr_min = self.snr_min)
                ax.yaxis.tick_right()
                
                ax = self.fig.add_subplot(self.bpt_map_gs)
                ax, cb = self.maps.plot_bpt_nii(ax = ax, plot_map = True)
                
                ax = self.fig.add_subplot(self.bpt_nii_gs)
                pts, ax = self.maps.plot_bpt_nii(ax = ax, overplot_dk_bpt=True, dk_bpt_kwargs={"s": 25}, snr_min = self.snr_min)
                plt.colorbar(pts, label = r'$R/R_e$')
                
                axy = self.fig.add_subplot(self.radial_oiii_gs, sharey = ax)
                ylim = axy.get_ylim()
                axy = self.maps.plot_radial_emline(["emline gflux oiii 5008", "emline gflux hb"], 
                                                   s = 5, ax = axy,
                                                   log10 = True, c = self.maps['spx ellcoo r_re'].flatten(), 
                                                   cmap = 'viridis', snr_min = self.snr_min)
                axy.set_ylim(ylim)
                
                ax = self.fig.add_subplot(self.map_gs)
                ha = self.maps['emline gflux ha']
                ax = ha.plot(fig = self.fig, ax = ax, 
                                        cbrange = self.new_scale,
                                        return_cb = True, 
                                        sky_coords = True, 
                                        snr_min = self.snr_min,
                                        patch_kws = {'hatch':'xx', 
                                                     'facecolor':'white', 
                                                     'edgecolor':'grey'}, 
                                        cmap = "Reds")
                
                self.scale_ax = self.fig.add_subplot(self.scale_gs)
                self.scale_ax.set_xlim([-1,11])
                self.scale_ax.set_ylim((-1,31))
                self.scale_ax.scatter([self.new_scale[0]], [self.new_scale[1]])
                self.scale_ax.set_xlabel("Min Color Scale")
                self.scale_ax.set_ylabel("Max Color Scale")

                
                self.fig.canvas.draw()



class DKAnalogMixin():
    """
    Mixin Class for DK_MWAnalogs Class
    """

    def basic_SFR(self, sample = None, balmer_dered = False):
        """
        Basic Star Formation Rate Calculation

        Parameters
        ----------
        sample: 'dict', or 'astropy.table.Table' optional, must be keyword
            DAP for the Sample that you want to get SFR for
        balmer_dered: 'bool', optional, must be keyword
            if True, uses Balmer Decrement to deredden ha intensity
        """
        if sample is None:
            # Use Default Sample
            sample = self.dk_sample
        elif ("PLATEIFU" not in sample) and ("plateifu" not in sample):
            raise TypeError("Sample is not a DAP Entry or Dictionary")

        def ha_deredden(ha, hb, snr_mask = None, Av_mask = None):
            """
            Calculate A_V and deredden Halpha 

            Parameters
            ----------

            ha: 'marvin.tools.maps.Maps'
                H-Alpha Map
            hb: 'marvin.tools.maps.Maps'
                H-Beta Map
            snr_mask: 'number', optional, must be keyword
                Signal to Noise Mask to use
            Av_mask: 'number', optional, must be keyword
                Max value of Av (actually A_ha) to consider
            """
            hahb = ha.value / hb.value
            A_ha = A_ha_balmer(hahb)
            mask = np.isinf(A_ha)
            mask |= np.isnan(A_ha)
            mask |= A_ha < 0
            if Av_mask is not None:
                mask |= A_ha > Av_mask
            if snr_mask is not None:
                mask |= ha.snr < snr_mask
                mask |= hb.snr < snr_mask
            return ha * 10**(0.4 * A_ha), mask


        


        def SFR_Single(dap):
            """
            Single Entry calculation

            Parameters
            ----------
            dap: 'dict' or 'astropy.table.Table'
                DAP entry
            """
            maps = Maps(plateifu = dap["PLATEIFU"])
            ha = maps['emline sflux ha']
            hb = maps['emline sflux hb']
            ha_dered, mask = ha_deredden(ha, hb)
            ldz = dap["LDIST_Z"] * u.Mpc / u.littleh
            lha = ha_lum(ha_dered.value * ha_dered.unit, ldz).to(u.erg / u.s / u.pix, u.with_H0(WMAP9.H0))
            sfr = simple_SFR(lha)
            mask |= np.isnan(sfr)
            return np.ma.masked_array(sfr, mask = mask)

    def sample_click_plots(self, fig = None, snr_min = None):
        """
        Interactive Plots with DK_MWAnalog Sample
        """
        if fig is None:
            fig = plt.figure(figsize = (8,35))

        if snr_min is None:
            snr_min = 2.

        if self.sersic:
            gs = fig.add_gridspec(24,4, wspace = 0.3, hspace = 0.6)
        else:
            gs = fig.add_gridspec(26,4, wspace = 0.3, hspace = 0.6)
        ind1 = 0
        images = Image.from_list(self.dk_sample["PLATEIFU"])

        for ell, im in enumerate(images):
            ind2 = ell % 4
            ax = fig.add_subplot(gs[ind1, ind2], projection = im.wcs)

            ax.imshow(im.data, origin = 'lower')
            im.overlay_hexagon(ax, color=pal[1], linewidth=1)
            ax.grid(False)

            if ind2 == 3:
                ind1+=1

        if not self.sersic:
            n_gal = 21
            image_axes = fig._get_axes()[0:n_gal]
            big_image_gs = gs[6:8, 0:2]
            bpt_map_gs = gs[5:8, 2:]
            radial_nii_gs = gs[8:10, 0:2]
            violin_gs = gs[8:10, 2:]
            bpt_nii_gs = gs[10:13, 1:]
            radial_oiii_gs = gs[10:13,0]
            map_gs = gs[14:17, 2:]
            scale_gs = gs[14:17, 0]
            map_ax = fig.add_subplot(gs[17,:])
        else:
            n_gal = 14
            image_axes = fig._get_axes()[0:n_gal]
            big_image_gs = gs[4:6, 0:2]
            bpt_map_gs = gs[3:6, 2:]
            radial_nii_gs = gs[6:8, 0:2]
            violin_gs = gs[6:8, 2:]
            bpt_nii_gs = gs[8:11, 1:]
            radial_oiii_gs = gs[8:11,0]
            map_gs = gs[12:15, 2:]
            scale_gs = gs[12:15, 0]   
            map_ax = fig.add_subplot(gs[15,:])

        ell = 0
        im = images[ell]
        ax = fig.add_subplot(big_image_gs, projection = im.wcs)
        ax.imshow(im.data, origin = 'lower')
        im.overlay_hexagon(ax, color=pal[1], linewidth=1)
        ax.grid(False)

        ax = fig.add_subplot(bpt_map_gs)
        maps = DKMaps(plateifu = self.dk_sample["PLATEIFU"][ell])
        ax, cb = maps.plot_bpt_nii(ax = ax, plot_map = True)

        ax = fig.add_subplot(radial_nii_gs)
        ax = maps.plot_radial_emline(["emline gflux nii 6585", "emline gflux ha"], s = 5, ax = ax, 
                                         log10 = True, c = maps['spx ellcoo r_re'].flatten(), cmap = 'viridis')


        ax = fig.add_subplot(violin_gs)
        ax = maps.plot_violin_bpt_nii(inner = 'quartile', alpha = 0.2, ax = ax)
        ax = maps.plot_violin_bpt_nii(inner = None, scale = 'count', ax = ax)
        ax.yaxis.tick_right()

        ax = fig.add_subplot(bpt_nii_gs)
        pts, ax = maps.plot_bpt_nii(ax = ax, overplot_dk_bpt=True, dk_bpt_kwargs={"s": 25})
        plt.colorbar(pts, label = r'$R/R_e$')


        axy = fig.add_subplot(radial_oiii_gs, sharey = ax)
        ylim = axy.get_ylim()
        axy = maps.plot_radial_emline(["emline gflux oiii 5008", "emline gflux hb"], s = 5, ax = axy,
                                        log10 = True, c = maps['spx ellcoo r_re'].flatten(), cmap = 'viridis')
        axy.set_ylim(ylim)

        ax = fig.add_subplot(map_gs)
        ha = maps['emline gflux ha']
        ax = ha.plot(fig = fig, ax = ax, 
                                cbrange = (0.,10.),
                                return_cb = True, 
                                sky_coords = True,
                     snr_min = 2.,
                                patch_kws = {'hatch':'xx', 
                                             'facecolor':'white', 
                                             'edgecolor':'grey'}, 
                                cmap = "Reds")

        scale_ax = fig.add_subplot(scale_gs)
        scale_ax.set_xlim([-1,11])
        scale_ax.set_ylim((-1,31))
        scale_ax.scatter([0], [10])
        scale_ax.set_xlabel("Min Color Scale")
        scale_ax.set_ylabel("Max Color Scale")


        # Create a Rectangle patch
        rectha = Rectangle((0,0.5),0.2,0.5,
                         linewidth=1,edgecolor='b',facecolor='r', alpha = 0.5)
        recthb = Rectangle((0.2,0.5),0.2,0.5,
                         linewidth=1,edgecolor='b',facecolor='b', alpha = 0.5)
        rectnii = Rectangle((0.4,0.5),0.2,0.5,
                         linewidth=1,edgecolor='b',facecolor='r', alpha = 0.5)
        rectoii = Rectangle((0.6,0.5),0.2,0.5,
                         linewidth=1,edgecolor='b',facecolor='b', alpha = 0.5)
        rectav = Rectangle((0.8,0.),0.2,1.0,
                         linewidth=1,edgecolor='b',facecolor='g', alpha = 0.5)

        # Add the patch to the Axes
        map_ax.add_patch(rectha)
        map_ax.add_patch(recthb)
        map_ax.add_patch(rectnii)
        map_ax.add_patch(rectoii)
        map_ax.add_patch(rectav)

        # Create a Rectangle patch
        rectha = Rectangle((0,0.),0.2,0.5,
                         linewidth=1,edgecolor='b',facecolor='r', alpha = 0.5)
        recthb = Rectangle((0.2,0.),0.2,0.5,
                         linewidth=1,edgecolor='b',facecolor='b', alpha = 0.5)
        rectnii = Rectangle((0.4,0.),0.2,0.5,
                         linewidth=1,edgecolor='b',facecolor='r', alpha = 0.5)
        rectoii = Rectangle((0.6,0.),0.2,0.5,
                         linewidth=1,edgecolor='b',facecolor='b', alpha = 0.5)


        # Add the patch to the Axes
        map_ax.add_patch(rectha)
        map_ax.add_patch(recthb)
        map_ax.add_patch(rectnii)
        map_ax.add_patch(rectoii)


        map_ax.text(0.1, 0.75, r"H$\alpha$", 
                    horizontalalignment='center', 
                    verticalalignment='center')
        map_ax.text(0.1, 0.25, r"H$\alpha$ Dered", 
                    horizontalalignment='center', 
                    verticalalignment='center')

        map_ax.text(0.3, 0.75, r"H$\beta$", 
                    horizontalalignment='center', 
                    verticalalignment='center')
        map_ax.text(0.3, 0.25, r"H$\beta$ Dered", 
                    horizontalalignment='center', 
                    verticalalignment='center')

        map_ax.text(0.5, 0.75, r"$[NII] \lambda 6585$", 
                    horizontalalignment='center', 
                    verticalalignment='center')
        map_ax.text(0.5, 0.25, r"$[NII] \lambda 6585$ Dered", 
                    horizontalalignment='center', 
                    verticalalignment='center')

        map_ax.text(0.7, 0.75, r"$[OII] \lambda 3727$", 
                    horizontalalignment='center', 
                    verticalalignment='center')
        map_ax.text(0.7, 0.25, r"$[OII] \lambda 3727$ Dered", 
                    horizontalalignment='center', 
                    verticalalignment='center')

        map_ax.text(0.9, 0.5, r"$A_V (Balmer)$", 
                    horizontalalignment='center', 
                    verticalalignment='center')

        map_ax.xaxis.set_ticks([])
        map_ax.yaxis.set_ticks([])


        return DKSamplePlotter(fig, gs, data= self, scale_ax = scale_ax, map_ax = map_ax, snr_min = snr_min)

    def get_radial_bpt_counts(self, radial_bin, sample = None, fraction = False, 
                              cumulative = False, percentiles = None, radial_norm = [1],
                              combine_comp_liner = False, return_all = False, **kwargs):
        """
        Returns dictionary of radial BPT classifications cumulative for a sample

        Parameters
        ----------
        radial_bin: 'number', 'list'
            radius value in terms of R_e 

        sample: 'astropy.Table', optional, must be keyword
            DAP data table of sample to use
            Defaulot to dk_sample
        fraction: 'bool', optional, must be keyword
            if True, returns fraction instead of counts
        cumulative: 'bool', optional, must be keyword
            if True, returns cumulative values
            if False, returns counts/percentiles for each galaxy
        percentiles: 'tuple', 'list', optional, must be keyword
            percentile values to return
        radial_norm: 'number', 'u.Quantity', optional, must be keyword
            normalization value for radius in Re
        combine_comp_liner: 'bool', optional, must be keyword
            if True, combines comp and liner classifications into one
        return_all: 'bool', optional, must be keyword
            if True, return all galaxy values
        kwargs: 'dict', optional, must be keyword
            keywords passed to DKMaps.get_radial_bpt_counts
        """

        if sample is None:
            sample = self.dk_sample

        if len(radial_norm) != len(sample):
        	if len(radial_norm) == 1:
	        	radial_norm = np.ones(len(sample)) * radial_norm[0]
	        else:
	        	raise ValueError("radial_norm does not have same size as the specified sample.")


        if cumulative:

            # Initialize counts dictionary
            counts = {"sf":np.zeros_like(radial_bin), 
                      "comp":np.zeros_like(radial_bin), 
                      "agn":np.zeros_like(radial_bin), 
                      "liner":np.zeros_like(radial_bin), 
                      "invalid":np.zeros_like(radial_bin), 
                      "total":np.zeros_like(radial_bin)}

            kwargs["add_to"] = counts

            if sample.__class__ != Row:
                if "pool" in kwargs:
                    kwargs["keep_pool"] = True
                for ell, plateifu in enumerate(sample["PLATEIFU"]):
                    maps = DKMaps(plateifu = plateifu)

                    kwargs["add_to"] = maps.get_radial_bpt_counts(radial_bin, radial_norm = radial_norm[ell], **kwargs)

                if "pool" in kwargs:
                    # Close Pool when finished
                    kwargs["pool"].close()
            else:
                maps = DKMaps(plateifu = sample["PLATEIFU"])
                kwargs["add_to"] = maps.get_radial_bpt_counts(radial_bin, radial_norm = radial_norm[0], **kwargs)

            counts = kwargs["add_to"]
            if combine_comp_liner:
                counts["liner"] += counts["comp"]
            if fraction:
                fractions = {}
                fractions["sf"] = [np.divide(x, tot) for x, tot in zip(counts["sf"],counts["total"])]
                fractions["comp"] = [np.divide(x, tot) for x, tot in zip(counts["comp"],counts["total"])]
                fractions["agn"] = [np.divide(x, tot) for x, tot in zip(counts["agn"],counts["total"])]
                fractions["liner"] = [np.divide(x, tot) for x, tot in zip(counts["liner"],counts["total"])]
                fractions["invalid"] = [np.divide(x, tot) for x, tot in zip(counts["invalid"],counts["total"])]
                return fractions
            else:
                return counts

        else:
            total_counts = []
            if sample.__class__ != Row:
                if "pool" in kwargs:
                    kwargs["keep_pool"] = True
    
                for ell, plateifu in enumerate(sample["PLATEIFU"]):
                    maps = DKMaps(plateifu = plateifu)
                    counts = maps.get_radial_bpt_counts(radial_bin, radial_norm = radial_norm[ell], **kwargs)
                    if combine_comp_liner:
                        counts["liner"] += counts["comp"]
                    if fraction:
                        fractions = {}
                        fractions["sf"] = [np.divide(x, tot) for x, tot in zip(counts["sf"],counts["total"])]
                        fractions["comp"] = [np.divide(x, tot) for x, tot in zip(counts["comp"],counts["total"])]
                        fractions["agn"] = [np.divide(x, tot) for x, tot in zip(counts["agn"],counts["total"])]
                        fractions["liner"] = [np.divide(x, tot) for x, tot in zip(counts["liner"],counts["total"])]
                        fractions["invalid"] = [np.divide(x, tot) for x, tot in zip(counts["invalid"],counts["total"])]
                        total_counts.append(fractions)
                    else:
                        total_counts.append(counts)



                if "pool" in kwargs:
                    # Close Pool when finished
                    kwargs["pool"].close()
            else:
                maps = DKMaps(plateifu = sample["PLATEIFU"])
                counts = maps.get_radial_bpt_counts(radial_bin, radial_norm = radial_norm[0], **kwargs)
                if combine_comp_liner:
                    counts["liner"] += counts["comp"]
                if fraction:
                    fractions = {}
                    fractions["sf"] = [np.divide(x, tot) for x, tot in zip(counts["sf"],counts["total"])]
                    fractions["comp"] = [np.divide(x, tot) for x, tot in zip(counts["comp"],counts["total"])]
                    fractions["agn"] = [np.divide(x, tot) for x, tot in zip(counts["agn"],counts["total"])]
                    fractions["liner"] = [np.divide(x, tot) for x, tot in zip(counts["liner"],counts["total"])]
                    fractions["invalid"] = [np.divide(x, tot) for x, tot in zip(counts["invalid"],counts["total"])]
                    total_counts.append(fractions)
                else:
                    total_counts.append(counts)

            if fraction:
                if percentiles is None:
                    percentiles = (16,50,84)

                perper = {}
                perpers = {}
                for k in total_counts[0]:
                    perper[k] = np.vstack(list(d[k] for d in total_counts))
                    perpers[k] = np.percentile(perper[k], percentiles, axis = 0)

                if return_all:
                    return perper
                else:
                    return perpers
            else:
                return total_counts


    def get_all_bpt_points(self, sample = None, set_as_attribute = False, return_ndarray = False, **kwargs):
        """
        Gets all bpt points from sample in a single list

        Parameters
        ----------
        sample: 'astropy.Table', optional, must be keyword
            DAP data table of sample to use
            Defaulot to dk_sample
        set_as_attribute: 'bool', optional, must be keyword
            if True, will set output data as an attribute to the class
            used for future plot creation / manipulation
        return_ndarray: 'bool', optional, must be keyword
            if True, returns data as numpy array
        """
        if sample is None:
            sample = self.dk_sample

        # Initialize variables
        if return_ndarray:
            data = []
        radius = np.array([])
        log_nii_ha = np.array([])
        log_oiii_hb = np.array([])
        nii_mask = np.array([])
        oiii_mask = np.array([])

        if sample.__class__ != Row:
            for plateifu in sample["PLATEIFU"]:
                maps = DKMaps(plateifu = plateifu)
                rad, nii, oiii = maps.plot_bpt_nii(plot_kde = True, return_data = True, **kwargs)
                if return_ndarray:
                    data.append((rad.value.flatten(), 
                        np.ma.masked_array(nii.flatten(), mask = nii.mask.flatten()), 
                        np.ma.masked_array(oiii.flatten(), mask = oiii.mask.flatten())))
                else:
                    radius = np.hstack((radius, rad.value.flatten()))
                    log_nii_ha = np.hstack((log_nii_ha, nii.flatten()))
                    log_oiii_hb = np.hstack((log_oiii_hb, oiii.flatten()))
                    nii_mask = np.hstack((nii_mask, nii.mask.flatten()))
                    oiii_mask = np.hstack((oiii_mask, oiii.mask.flatten()))
        else:
            maps = DKMaps(plateifu = sample["PLATEIFU"])
            rad, nii, oiii = maps.plot_bpt_nii(plot_kde = True, return_data = True, **kwargs)
            if return_ndarray:
                data.append((rad.value.flatten(), 
                    np.ma.masked_array(nii.flatten(), mask = nii.mask.flatten()), 
                    np.ma.masked_array(oiii.flatten(), mask = oiii.mask.flatten())))
            else:
                radius = np.hstack((radius, rad.value.flatten()))
                log_nii_ha = np.hstack((log_nii_ha, nii.flatten()))
                log_oiii_hb = np.hstack((log_oiii_hb, oiii.flatten()))
                nii_mask = np.hstack((nii_mask, nii.mask.flatten()))
                oiii_mask = np.hstack((oiii_mask, oiii.mask.flatten()))

        if return_ndarray:
            data = np.array(data)
            return data
        else:
            nii_masked = np.ma.masked_array(log_nii_ha, mask = nii_mask)
            oiii_masked = np.ma.masked_array(log_oiii_hb, mask = oiii_mask)

            if set_as_attribute:
                if sample is self.dk_sample:
                    self.sample_radius = radius
                    self.sample_log_nii_ha = nii_masked
                    self.sample_log_oiii_hb = oiii_masked
                else:
                    self.nobar_sample_radius = radius
                    self.nobar_sample_log_nii_ha = nii_masked
                    self.nobar_sample_log_oiii_hb = oiii_masked
            else:
                return radius, nii_masked, oiii_masked

    def plot_bpt_kde(self, ax = None, jointplot = False, sample = None, radial_bin = None, 
                     overplot_dk_bpt = False, dk_bpt_kwargs = {}, 
                     classification_kwargs = {}, **kwargs):
        """
        Plots BPT diagram with kdeplot from sample

        Parameters
        ----------
        ax: 'matplotlib.pyplot.figure.axes', optional, must be keyword
            Matplotlib axes to plot on
        jointplot: 'bool', optional, must be keyword
            if True, does a seaborn jointplot
        sample: 'astropy.Table', optional, must be keyword
            DAP data table of sample to use
            Defaulot to dk_sample
            can also be tuple containing (radius, log_nii_ha, log_oiii_hb)
        radial_bin: 'list' or 'tuple' or 'np.ndarray', optional, must be keyword
            if given, only plots points within provided radial bin in terms of R/R_e
        overplot_dk_bpt: 'bool', optional, must be keyword
            if True, overplots dk_bpt
        dk_bpt_kwargs: 'dict', optional, must be keyword
            kwargs passed to dk_bpt
        classification_kwargs: 'dict', optional, must be keyword
            kwargs passed to draw_classification_lines
        kwargs: 'dict', optional, must be keyword
            keywords passed to sns.kdeplot
        """
        if sample is None:
            if hasattr(self, "sample_radius"):
                radius = self.sample_radius
                log_nii_ha = self.sample_log_nii_ha
                log_oiii_hb = self.sample_log_oiii_hb
            else:
                radius, log_nii_ha, log_oiii_hb = self.get_all_bpt_points()
        elif sample.__class__ is tuple:
            radius, log_nii_ha, log_oiii_hb = sample
        elif sample is self.dk_sample_nobar:
            if hasattr(self, "nobar_sample_radius"):
                radius = self.nobar_sample_radius
                log_nii_ha = self.nobar_sample_log_nii_ha
                log_oiii_hb = self.nobar_sample_log_oiii_hb
            else:
                radius, log_nii_ha, log_oiii_hb = self.get_all_bpt_points(sample = sample)
        else:
            radius, log_nii_ha, log_oiii_hb = self.get_all_bpt_points(sample = sample)

        if not jointplot:
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111)


            # Draw classification lines
            from .bpt import draw_classification_lines, scale_to_dkbpt
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
                within = radius >= radial_bin[0]
                within &= radius <= radial_bin[1]

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
                from .bpt import plot_dkbpt
                ax = plot_dkbpt(ax, **dk_bpt_kwargs)

            ax = scale_to_dkbpt(ax)

            return ax
        else:
            # Default kwargs
            if "joint_kws" not in kwargs:
                kwargs["joint_kws"] = {}

            if "cmap" not in kwargs["joint_kws"]:
                kwargs["joint_kws"]["cmap"] = "plasma"

            if "zorder" not in kwargs:
                kwargs["zorder"] = 0

            if "kind" not in kwargs:
                kwargs["kind"] = "kde"

            if ("shade" not in kwargs) & (kwargs["kind"] == "kde"):
                kwargs["shade"] = True

            if ("shade_lowest" not in kwargs) & (kwargs["kind"] == "kde"):
                kwargs["shade_lowest"] = False


            if radial_bin is not None:
                within = radius >= radial_bin[0]
                within &= radius <= radial_bin[1]

                g = sns.jointplot(log_nii_ha[(np.invert(log_nii_ha.mask | log_oiii_hb.mask) & within)], 
                                 log_oiii_hb[(np.invert(log_nii_ha.mask | log_oiii_hb.mask) & within)],
                                 **kwargs)
            else:
                g = sns.jointplot(log_nii_ha[np.invert(log_nii_ha.mask | log_oiii_hb.mask)], 
                                 log_oiii_hb[np.invert(log_nii_ha.mask | log_oiii_hb.mask)], 
                                 **kwargs)
            # Draw classification lines
            from .bpt import draw_classification_lines, scale_to_dkbpt
            ax = draw_classification_lines(g.ax_joint, **classification_kwargs)

            ax.set_xlabel(r'$log_{10}$([NII] $\lambda$ 6585/H$\alpha$)',fontsize=12)
            ax.set_ylabel(r'$log_{10}$([OIII] $\lambda$ 5007/H$\beta$)',fontsize=12)

            if overplot_dk_bpt:
                from .bpt import plot_dkbpt
                ax = plot_dkbpt(ax, **dk_bpt_kwargs)

            ax = scale_to_dkbpt(ax)

            return g

    def get_PCA_stellar_mass(self, sample = None, pca_data_dir = None, **kwargs):
        """
        Return PCA stellar masses for sample of galaxies as a numpy array in solMass

        Parameters
        ----------
        sample:
            Sample to get data for
        """

        if sample is None:
            sample = self.dk_sample

        if pca_data_dir is None:
            pca_data_dir = os.path.join(os.environ['SAS_BASE_DIR'], 'mangawork', 'manga', 'sandbox', 'zachpace', 'CSPs_CKC14_MaNGA_20190215-1',
                                        self.drpver, self.dapver, 'results')


        data = []

        if sample.__class__ != Row:
            for plateifu in sample["PLATEIFU"]:
                maps = DKMaps(plateifu = plateifu)
                mstar = maps.get_PCA_stellar_mass(**kwargs).value
                data.append((mstar[0,:,:].flatten(), mstar[1,:,:].flatten(), mstar[2,:,:].flatten()))
        else:
            maps = DKMaps(plateifu = sample["PLATEIFU"])
            mstar = maps.get_PCA_stellar_mass(**kwargs).value
            data.append((mstar[0,:,:].flatten(), mstar[1,:,:].flatten(), mstar[2,:,:].flatten()))
        
        return np.array(data)


    def get_PCA_stellar_mass_density(self, sample = None, pca_data_dir = None, **kwargs):
        """
        Return PCA stellar mass densities for sample of galaxies as a numpy array
        in units of solMass / pc**2

        Parameters
        ----------
        sample:
            Sample to get data for
        """
        if sample is None:
            sample = self.dk_sample

        if pca_data_dir is None:
            pca_data_dir = os.path.join(os.environ['SAS_BASE_DIR'], 'mangawork', 'manga', 'sandbox', 'zachpace', 'CSPs_CKC14_MaNGA_20190215-1',
                                        self.drpver, self.dapver, 'results')

        data = []

        if sample.__class__ != Row:
            for plateifu in sample["PLATEIFU"]:
                maps = DKMaps(plateifu = plateifu)
                mstar = maps.get_PCA_stellar_mass_density(**kwargs).value
                data.append((mstar[0,:,:].flatten(), mstar[1,:,:].flatten(), mstar[2,:,:].flatten()))
        else:
            maps = DKMaps(plateifu = sample["PLATEIFU"])
            mstar = maps.get_PCA_stellar_mass_density(**kwargs).value
            data.append((mstar[0,:,:].flatten(), mstar[1,:,:].flatten(), mstar[2,:,:].flatten()))
        
        return np.array(data)

    def get_bar_masks(self, sample = None, galaxyzoo3d_dir = None, supress_warnings = True, **kwargs):
        """
        Return bar masks from GalaxyZoo 3D

        Parameters
        ----------
        sample:
            Sample to get data for
        galaxyzoo3d_dir: 'str', optional, must be keyword
            Directory to find data files
        supress_warnings: 'bool', optional, must be keyword
            supresses warnings for galaxies that don't have data;
            default of True
        """
        from sdss_access import AccessError
        from marvin.core.exceptions import MarvinError

        if sample is None:
            sample = self.dk_sample

        if galaxyzoo3d_dir is None:
            galaxyzoo3d_dir = "/Users/dk/sas/mangawork/manga/sandbox/galaxyzoo3d/v2_0_0/"

        data = []


        if sample.__class__ != Row:
            if "PLATEIFU" in sample.keys():
                for plateifu in sample["PLATEIFU"]:
                    try:
                        maps = DKMaps(plateifu = plateifu)
                    except AccessError:
                        logging.warning("AccessError raised - skipping Galaxy with plateifu {}".format(plateifu))
                        mask = np.array([np.nan])
                    else:
                        if supress_warnings:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                mask = maps.get_bar_mask(galaxyzoo3d_dir = galaxyzoo3d_dir, **kwargs)
                        else:
                            mask = maps.get_bar_mask(galaxyzoo3d_dir = galaxyzoo3d_dir, **kwargs)

                    data.append(mask.flatten())
            else:
                for mangaid in sample["MANGAID"]:
                    try:
                        maps = DKMaps(mangaid = mangaid.rstrip())
                    except AccessError:
                        logging.warning("AccessError raised - skipping Galaxy with MaNGA-ID {}".format(mangaid.rstrip()))
                        mask = np.array([np.nan])
                    except MarvinError:
                        logging.warning("MarvinError raised - skipping Galaxy with MaNGA-ID {}".format(mangaid.rstrip()))
                        mask = np.array([np.nan])
                    else:
                        if supress_warnings:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                mask = maps.get_bar_mask(galaxyzoo3d_dir = galaxyzoo3d_dir, **kwargs)
                        else:
                            mask = maps.get_bar_mask(galaxyzoo3d_dir = galaxyzoo3d_dir, **kwargs)

                    data.append(mask.flatten())
        else:
            maps = DKMaps(plateifu = sample["PLATEIFU"])
            if supress_warnings:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mask = maps.get_bar_mask(galaxyzoo3d_dir = galaxyzoo3d_dir, **kwargs)
            else:
                mask = maps.get_bar_mask(galaxyzoo3d_dir = galaxyzoo3d_dir, **kwargs)
            data.append(mask.flatten())
        
        return np.array(data)

    def get_all_bpt_masks(self, sample = None, **kwargs):
        """
        Return bpt masks from nii bpt diagram

        Parameters
        ----------
        sample:
            Sample to get data for

        """
        if sample is None:
            sample = self.dk_sample

        sf = []
        liner = []
        agn = []
        comp = []
        invalid = []


        if sample.__class__ != Row:
            for plateifu in sample["PLATEIFU"]:
                maps = DKMaps(plateifu = plateifu)
                data = maps.plot_bpt_nii(return_figure = False, **kwargs)
                sf.append(data["sf"]["nii"].flatten())
                comp.append(data["comp"]["nii"].flatten())
                liner.append(data["liner"]["nii"].flatten())
                agn.append(data["agn"]["nii"].flatten())
                invalid.append(data["invalid"]["nii"].flatten())
        else:
            maps = DKMaps(plateifu = sample["PLATEIFU"])
            data = maps.plot_bpt_nii(return_figure = False, **kwargs)
            sf.append(data["sf"]["nii"].flatten())
            comp.append(data["comp"]["nii"].flatten())
            liner.append(data["liner"]["nii"].flatten())
            agn.append(data["agn"]["nii"].flatten())
            invalid.append(data["invalid"]["nii"].flatten())
        
        return np.array(sf), np.array(comp), np.array(liner), np.array(agn), np.array(invalid)

    def get_all_bar_coords(self, sample = None, bar_masks = None, **kwargs):
        """
        Return bar_coords from bar masks 

        Parameters
        ----------
        sample:
            sample to get data for
        bar_masks: `tuple, list`
            bar masks (flattened, same len as sample)

        """
        if sample is None:
            sample = self.dk_sample

        bar_coords = []

        if sample.__class__ != Row:
            if bar_masks is not None:
                for plateifu, mask in zip(sample["PLATEIFU"], bar_masks):
                    maps = DKMaps(plateifu = plateifu)
                    bar_coords.append(maps.get_bar_coords(bar_mask = mask))
            else:
                for plateifu in sample["PLATEIFU"]:
                    maps = DKMaps(plateifu = plateifu)
                    bar_coords.append(maps.get_bar_coords(**kwargs))

        else:
            maps = DKMaps(plateifu = sameple["PLATEIFU"])
            if bar_masks is not None:
                bar_coords.append(maps.get_bar_coords(bar_mask = bar_masks))
            else:
                bar_coords.append(maps.get_bar_coords(**kwargs))

        return np.array(bar_coords)

    

    def downloadList(inputlist, dltype='cube', **kwargs):
        """Download a list of MaNGA objects.

        Uses sdss_access to download a list of objects
        via rsync.  Places them in your local sas path mimicing
        the Utah SAS.

        i.e. $SAS_BASE_DIR/mangawork/manga/spectro/redux

        Can download cubes, rss files, maps, modelcubes, mastar cubes,
        png images, default maps, or the entire plate directory.
        dltype=`dap` is a special keyword that downloads all DAP files.  It sets binmode
        and daptype to '*'

        Parameters:
            inputlist (list):
                Required.  A list of objects to download.  Must be a list of plate IDs,
                plate-IFUs, or manga-ids
            dltype ({'cube', 'maps', 'modelcube', 'dap', image', 'rss', 'mastar', 'default', 'plate'}):
                Indicated type of object to download.  Can be any of
                plate, cube, image, mastar, rss, map, modelcube, or default (default map).
                If not specified, the dltype defaults to cube.
            release (str):
                The MPL/DR version of the data to download.
                Defaults to Marvin config.release.
            bintype (str):
                The bin type of the DAP maps to download. Defaults to *
            binmode (str):
                The bin mode of the DAP maps to download. Defaults to *
            n (int):
                The plan id number [1-12] of the DAP maps to download. Defaults to *
            daptype (str):
                The daptype of the default map to grab.  Defaults to *
            dir3d (str):
                The directory where the images are located.  Either 'stack' or 'mastar'. Defaults to *
            verbose (bool):
                Turns on verbosity during rsync
            limit (int):
                A limit to the number of items to download
            test (bool):
                If True, tests the download path construction but does not download

        Returns:
            If test=True, returns the list of full filepaths that will be downloaded
        """

        from marvin.core.exceptions import MarvinError, MarvinUserWarning


        try:
            from sdss_access import RsyncAccess, AccessError
        except ImportError:
            RsyncAccess = None

        try:
            from sdss_access.path import Path
        except ImportError:
            Path = None

        assert isinstance(inputlist, (list, np.ndarray)), 'inputlist must be a list or numpy array'

        # Get some possible keywords
        # Necessary rsync variables:
        #   drpver, plate, ifu, dir3d, [mpl, dapver, bintype, n, mode]
        verbose = kwargs.get('verbose', None)
        as_url = kwargs.get('as_url', None)
        release = kwargs.get('release', config.release)
        drpver, dapver = config.lookUpVersions(release=release)
        bintype = kwargs.get('bintype', '*')
        binmode = kwargs.get('binmode', None)
        daptype = kwargs.get('daptype', '*')
        dir3d = kwargs.get('dir3d', '*')
        n = kwargs.get('n', '*')
        limit = kwargs.get('limit', None)
        test = kwargs.get('test', None)

        # check for sdss_access
        if not RsyncAccess:
            raise MarvinError('sdss_access not installed.')

        # Assert correct dltype
        dltype = 'cube' if not dltype else dltype
        assert dltype in ['plate', 'cube', 'mastar', 'modelcube', 'dap', 'rss', 'maps', 'image',
                          'default', 'pca_mli'], ('dltype must be one of plate, cube, mastar, '
                                       'image, rss, maps, modelcube, dap, default')

        assert binmode in [None, '*', 'MAPS', 'LOGCUBE'], 'binmode can only be *, MAPS or LOGCUBE'

        # Assert correct dir3d
        if dir3d != '*':
            assert dir3d in ['stack', 'mastar'], 'dir3d must be either stack or mastar'

        # Parse and retrieve the input type and the download type
        idtype = parseIdentifier(inputlist[0])
        if not idtype:
            raise MarvinError('Input list must be a list of plates, plate-ifus, or mangaids')

        # Set download type
        if dltype == 'cube':
            name = 'mangacube'
        elif dltype == 'rss':
            name = 'mangarss'
        elif dltype == 'default':
            name = 'mangadefault'
        elif dltype == 'plate':
            name = 'mangaplate'
        elif dltype == 'maps':
            # needs to change to include DR
            if '4' in release:
                name = 'mangamap'
            else:
                name = 'mangadap5'
                binmode = 'MAPS'
        elif dltype == 'modelcube':
            name = 'mangadap5'
            binmode = 'LOGCUBE'
        elif dltype == 'dap':
            name = 'mangadap5'
            binmode = '*'
            daptype = '*'
        elif dltype == 'mastar':
            name = 'mangamastar'
        elif dltype == 'image':
            if check_versions(drpver, 'v2_5_3'):
                name = 'mangaimagenew'
            else:
                name = 'mangaimage'

        # check for public release
        is_public = 'DR' in release
        rsync_release = release.lower() if is_public else None


        # create rsync
        rsync_access = RsyncAccess(label='marvin_download', verbose=verbose, public=is_public, release=rsync_release)
        rsync_access.remote()


        # Add objects
        for item in inputlist:
            if idtype == 'mangaid':
                try:
                    plateifu = mangaid2plateifu(item)
                except MarvinError:
                    plateifu = None
                else:
                    plateid, ifu = plateifu.split('-')
            elif idtype == 'plateifu':
                plateid, ifu = item.split('-')
            elif idtype == 'plate':
                plateid = item
                ifu = '*'

        if dltype == "pca_mli":
            source = "rsync://sdss@dtn01.sdss.org/sas/mangawork/manga/sandbox/mangapca/zachpace/CSPs_CKC14_MaNGA_20190215-1/{0}/{1}/results/{2}-{3}/*".format(drpver, 
                dapver, plateid, ifu)
            location = "mangawork/manga/sandbox/mangapca/zachpace/CSPs_CKC14_MaNGA_20190215-1/{0}/{1}/results/{2}-{3}/*".format(drpver, 
                dapver, plateid, ifu)


            rsync_access.add(name, plate=plateid, drpver=drpver, ifu=ifu, dapver=dapver, dir3d=dir3d,
                             mpl=release, bintype=bintype, n=n, mode=binmode, daptype=daptype)

        # set the stream
        try:
            rsync_access.set_stream()
        except AccessError as e:
            raise MarvinError('Error with sdss_access rsync.set_stream. AccessError: {0}'.format(e))

        # get the list and download
        listofitems = rsync_access.get_urls() if as_url else rsync_access.get_paths()

        # print download location
        item = listofitems[0] if listofitems else None
        if item:
            ver = dapver if dapver in item else drpver
            dlpath = item[:item.rfind(ver) + len(ver)]
            if verbose:
                print('Target download directory: {0}'.format(dlpath))

        if test:
            return listofitems
        else:
            rsync_access.commit(limit=limit)












