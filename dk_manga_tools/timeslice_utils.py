# FOR HANDLING TOM'S STARLIGHT OUTPUT FILES
# Contact/queries: Thomas.Peterken@nottingham.ac.uk or tom_peterken@hotmail.co.uk
#
# v2.0.3: 2/6/20
# - Changed universe_age default to be 10^17 years (i.e. the option below is now switched off by default).
# - Allowed Starlight fits files without "spectra" extension.  If Spectra is called in this case, it will return None.
#
# v2.0.2: 28/04/2020
# - Added universe_age option to define the absolute maximum limit of stellar population ages possible in smoothing (in years).  Setting this to a very large number will use the SSPs to define this 
#
# v2.0.1: 22/04/2020
# - Fixed minor typo in interpolation between SFH sampling points.
# - Fixed error when requesting a Cumulative of the oldest extant age.  Will now return a TimeSlice of the oldest age.
# - Removed warning when requesting a slice outside the age limits because the excessive messages were annoying me.



import numpy as np
import os
from astropy.io import fits
import bisect

   
    
    
class timecube(object):
    
    def __init__(self, cubename, weight_type='light', default_gaussw=0.3, metallicity_gaussw=None, age_cutoff=3*10**7, universe_age=10**17, interp_points=(30,15), smooth_by_sfh=True):
        """Generates a file object to analyse the Starlight output located at cubename.
        
        :param cubename: Full path to your Starlight output cube.
        :param weight_type: 'light', 'current_mass', or 'initial_mass'.
        :param default_gaussw: Width of smoothing Gaussian used for smoothing in age.
        :param metallicity_gaussw: Width of smoothing Gaussian used for smoothing in metallicity.  If None, smoothing will be the same in both dimensions.
        :param age_cutoff: Minimum age, below which the Starlight SSP weights are ignored.  Recommended not to change this.
        :param interp_points: Tuple of number of sampling points in (age,metallicity) space to interpolate and smooth over.
        :param smooth_by_sfh: If True and weight_type is a mass type, the interpolation and smoothing is done in SFRs and converted back to mass weights.  If False, interpolation and smoothing is done on the SSP weights (less physically meaningful).  Results will be very similar!  Unused if weight_type='light'.
        :attributes:
            .map_av: 2D map of Av values, in dex.
            .map_veldisp: 2D map of stellar velocity dispersions measured by Starlight, in km/s.  !!Starlight was not configured for optimal kinematics, so this should not be used for science purposes!!
            .map_redchi2: 2D map of reduced Chi^2 of the final fit.
            .bpm: Bad pixel mask (True for bad pixels).   This is a very simple mask and False values does not imply that the data contained are trustworthy; merely that there are data there.
            .sum_im: 2D map of the total flux or mass contained in each spaxel.
            ._weighttype_: Same as input weight_type.
            ._age_gaussw_: Width of age-smoothing Gaussian, in dex.  This is equal to the input default_gaussw.
            ._met_gaussw_: Width of metallicity-smoothing Gaussian, in dex.  If metallicity_gaussw was not specified, this is equal to default_gaussw.  Otherwise it is equal to metallicity_gaussw.
            ._rawcube_: The raw data array of SSP light- or mass-weights.
            ._rawages_: The age values of the SSP array, in years.
            ._rawmets_: The metallicities of the SSP array, in [M/H].
            ._intcube_: The interpolated and smoothed data array.
            ._intages_: List of age sampling points of the smoothed data array.
            ._intmets_: List of metallicity sampling points of the smoothed data array.
            ._sfhcube_: Smoothed datacube of SFRs (rather than weights).
            ._specarray_: Array of observed, model, and weight spectra from the Starlight fits.
            
            """
        
        assert weight_type=='light' or weight_type=='initial_mass' or weight_type=='current_mass', \
            "weight_type must be 'light', 'initial_mass', or 'current_mass'."
            
        assert os.path.exists(cubename), \
            cubename+" does not exist."
            
            
        # Open fits cube
        fcube = fits.open(cubename)
        extslist = [n[1] for n in fcube.info(output=False)]        
        assert set(extslist) == set(['AGES', 'AV', 'CHI2', 'LIGHT', 'INITIAL_MASS', 'CURRENT_MASS', 'METS', 'PRIMARY', 'SPECTRA', 'VEL', 'VEL DISP', 'YAV']) or set(extslist) == set(['AGES', 'AV', 'CHI2', 'LIGHT', 'INITIAL_MASS', 'CURRENT_MASS', 'METS', 'PRIMARY', 'VEL', 'VEL DISP', 'YAV']), cubename+" extension list is unexpected!  Are you using the mass-corrected Starlight files?"
        

        # Load SSP weights cube, list of ages and metallicities
        self._rawages_ = fcube['AGES'].data
        self._rawmets_ = fcube['METS'].data
        self.map_av = fcube['AV'].data
        if 'YAV' in extslist:
            self.map_yav = fcube['YAV'].data
        self.map_veldisp = fcube['VEL DISP'].data
        self.map_redchi2 = fcube['CHI2'].data
        if 'SPECTRA' in extslist:
            self._specarray_ = fcube['SPECTRA'].data
        else:
            self._specarray_ = None
        self._rawcube_ = fcube[weight_type].data
        fcube.close()
        
        self._weighttype_ = weight_type
        self._age_gaussw_ = default_gaussw
        
        if metallicity_gaussw==None:
            self._met_gaussw_ = default_gaussw
        else:
            self._met_gaussw_ = metallicity_gaussw
        

        # Create pixel mask (True for bad pixels)
        self.bpm = np.sum(self._rawcube_, axis=(2,3))==0


        # Designate which SSP coordinates are invalid and which are bad
        metgrid_SSP,logagegrid_SSP = np.meshgrid(self._rawmets_, np.log10(self._rawages_))
        noSSPs = np.sum(self._rawcube_, axis=(0,1))==0 # Indicates all the (age,met) SSP coords which have no weight, so are cut out or don't exist in the parameter space
        
        invalidSSPs = np.sum(self._rawcube_, axis=(0,1))==0
        invalidSSPs[:,self._rawmets_==-0.4] = 0
        invalidSSPs[:,self._rawmets_==-0.41] = 0 # This indicates all the regions of parameter space that we don't explore!
        
        badagemets = invalidSSPs^noSSPs # This is therefore all of the SSP coordinates that we do not wish to take account for in smoothing

        
        
        # Cut young ages off
        if age_cutoff >= np.min(self._rawages_):
            agecuts = self._rawages_<=age_cutoff
            agews = 1-agecuts
            agews = _create4dweights(self._rawcube_, agews, ax=2)
            self._rawcube_ = self._rawcube_*agews
        self.sum_im = np.sum(self._rawcube_, axis=(2,3))
        
        
        # Define full region of parameter space
        if universe_age <= np.max(self._rawages_):
            universe_age = np.max(self._rawages_)+10**7
        
        ageboxw = np.log10(self._rawages_[-1])-np.log10(self._rawages_[-2])
        old_lim = np.min((10**(np.log10(self._rawages_[-1])+ageboxw), universe_age))
        self._universe_age_ = old_lim
        lmboxw = self._rawmets_[1]-self._rawmets_[0]
        hmboxw = self._rawmets_[-1]-self._rawmets_[-2]
        metboxw = np.min([lmboxw,hmboxw])
        hi_lim = self._rawmets_[-1]+metboxw
        lo_lim = self._rawmets_[0]-metboxw


        # Find the boundaries between the SSPs in age-space
        eff_ages = self._rawages_[self._rawages_>=age_cutoff]
        SSPagebounds = np.concatenate(([age_cutoff],[10**(0.5*(np.log10(eff_ages[an])+np.log10(eff_ages[an+1]))) for an in range(len(eff_ages)-1)],[old_lim]))
        SSPagebounds = np.pad(SSPagebounds, (1+len(self._rawages_)-len(SSPagebounds),0), 'constant', constant_values=0)
        self.SSPagebounds = SSPagebounds
            
        
        
        # Assign the weight cube to be the star-formation history cube
        if 'mass' in weight_type and smooth_by_sfh:
            dts = np.diff(SSPagebounds)            
            dtcube = _create4dweights(self._rawcube_, dts, ax=2)
            weightcube = self._rawcube_/dtcube
            weightcube[dtcube==0] = 0
                        
        else:
            weightcube = self._rawcube_
            
        self._weightcube_ = weightcube
            
            
            
            
        # Create grid of interpolated points
        agebasebounds = np.linspace(np.log10(age_cutoff), np.log10(old_lim), num=interp_points[0]+1)
        logagebase = np.array([0.5*(agebasebounds[n]+agebasebounds[n+1]) for n in range(interp_points[0])])
        metbasebounds = np.linspace(lo_lim, hi_lim, num=interp_points[1]+1)
        metbase = np.array([0.5*(metbasebounds[n]+metbasebounds[n+1]) for n in range(interp_points[1])])
            
        smooth_arr = np.zeros((len(self.map_av),len(self.map_av),interp_points[0],interp_points[1]), dtype=np.float)
                    
                   
        # Smooth the interpolated array
        for an,a in enumerate(logagebase):
            for mn,m in enumerate(metbase):
                ws = _gauss2d(self._rawmets_, np.log10(self._rawages_), m, a, self._met_gaussw_, self._age_gaussw_)
                ws[badagemets] = 0
                ws_4d = np.expand_dims(ws, axis=2)
                ws_4d = np.expand_dims(ws_4d, axis=3)
                ws_4d = np.swapaxes(ws_4d, 0, 2)
                ws_4d = np.swapaxes(ws_4d, 1, 3)
                ws_4d = np.pad(ws_4d, ((0,len(self.map_av)-1),(0,len(self.map_av)-1),(0,0),(0,0)), 'maximum')
#                if interp_bad[0,0,an,mn]==0:
#                    ws_4d[interp_bad==1] = 0
                smooth_arr[:,:,an,mn] = np.nansum(ws_4d*weightcube, axis=(2,3))/np.nansum(ws_4d, axis=(2,3))
                    
                
        
                    
        
        # Create a "fudge factor" array to make the totals add up
        sumar = np.expand_dims(self.sum_im, 2)
        sumar = np.expand_dims(sumar, 3)
        sumar = np.pad(sumar, ((0,0),(0,0),(0,len(logagebase)-1),(0,len(metbase)-1)), 'maximum')
        
                
        # If we're using the SFH, convert back to mass weights. Otherwise, make a SFH cube
        interp_dts = np.diff(10**agebasebounds)
        interp_dtcube = _create4dweights(smooth_arr, interp_dts, ax=2)
        
        if 'mass' in weight_type and smooth_by_sfh:
            intcube = smooth_arr*interp_dtcube
        else:
            intcube = smooth_arr
                        
            
        sumint = np.expand_dims(np.sum(intcube, axis=(2,3)), 2)
        sumint = np.expand_dims(sumint, 3)
        sumint = np.pad(sumint, ((0,0),(0,0),(0,len(logagebase)-1),(0,len(metbase)-1)), 'maximum')        
        self._intcube_ = intcube*sumar/sumint
            
        if 'mass' in weight_type and smooth_by_sfh:
            self._sfhcube_ = smooth_arr*sumar/sumint
        else:
            self._sfhcube_ = smooth_arr*sumar/(sumint*interp_dtcube)
            
        self._intages_ = 10**logagebase
        self._intmets_ = metbase
        
        
        
      
        

    
        
        
    
class TimeSlice(object):
    """Creates a Time-Slice object.  This will be linearly interpolated from the smoothed array of light or mass weights in the given timecube() object.
    
    :param cube: An instance of timecube().
    :param age: The age (in years) of the timeslice.
    :attributes:
        :ages: The age of the returned timeslice.  If the requested age is outside of the available ages, this will be the closest available age.
        :array: TimeSlice array.  This is the data that image(), metmap(), and agemap() work with.
        :_cube_: The specified timecube() instance.
    """
    
    def __init__(self, cube, age):
        
        age = _checkage(age, cube._intages_)
        self.ages = [age]
    
        self.array = _arrofage(cube, age, 'older') + _arrofage(cube, age, 'younger')
        self._cube_ = cube
        
        
        
        
class Range(object):
    """Creates a range Time-Slice object: i.e. all time-slices within the age range.  The smoothed weights from the timecube() within the specified age range are extracted.  The end "slices" of the array are linearly interpolated from the smoothed weights.
    
    :param cube: An instance of timecube().
    :param ages: A tuple-like of the age boundaries (in years).  Can be [min,max] or [max,min].
    :attributes:
        :ages: An array of the ages used in the interpolated data.
        :array: TimeSlice array.  This is the data that image(), metmap(), and agemap() work with.
        :_cube_: The specified timecube() instance.
    """
    
    def __init__(self, cube, ages):
        
        lo_age = _checkage(np.min(ages), cube._intages_)
        hi_age = _checkage(np.max(ages), cube._intages_)
        
        assert lo_age < hi_age, \
            "Ages must be different"
            
        # Find the "end slices"
        lo_arr = _arrofage(cube, lo_age, 'younger')
        hi_arr = _arrofage(cube, hi_age, 'older')
        
        # Find the main cube
        ages,centre_arr = _arrofrange(cube, lo_age, hi_age)
        
        self.ages = np.concatenate(([lo_age],ages,[hi_age]))
        self.array = np.concatenate((lo_arr, centre_arr, hi_arr), axis=2)
        self._cube_ = cube
        
        
        
        
def Cumulative(cube, age):
    """Creates a cumulative Time-Slice object.  A special case of Range().  The smoothed weights from the timecube() of all points older than the specified age range are extracted.  The end "slices" of the array are linearly interpolated from the smoothed weights.
    
    :param cube: An instance of timecube().
    :param ages: The age boundary (in years).
    :attributes:
        :ages: An array of the ages used in the interpolated data.
        :array: TimeSlice array.  This is the data that image(), metmap(), and agemap() work with.
        :_cube_: The specified timecube() instance.
    """
    
    if age==np.max(cube._intages_):
        cim = TimeSlice(cube, age)
    else:
        cim = Range(cube, [age, np.max(cube._intages_)])
    
    return cim
    
    
    
        
def Total(cube):
    """Creates a total Time-Slice object.  A special case of Cumulative().  The smoothed weights from the timecube() within the greatest possible age range are extracted.
    
    :param cube: An instance of timecube().
    :attributes:
        :ages: An array of the ages used in the interpolated data.
        :array: TimeSlice array.  This is the data that image(), metmap(), and agemap() work with.
        :_cube_: The specified timecube() instance.
    """
    
    return Range(cube, [np.min(cube._intages_), np.max(cube._intages_)])
        
        
        
      
        

    
        
        
    
def SFH(cube, mask=None):
    """Measures the star-formation history of the timecube() object.  timecube()'s weight_type MUST be a mass type.
    
    :param cube: An instance of timecube().
    :param mask: A pixel mask.  This can be a tuple of a specific coordinate (y,x), or can be a 2D image, in which case mask=0 will be ignored, and all mask>0 will be given equal weighting.
    :return:
        :ages: List of ages at which the SFR is sampled, in years.
        :SFH: Total SFR contained within the mask at each age sampling, in Msol/yr.
    """
    
    if type(cube._sfhcube_)==str:
        print('SFH of a light-weighted cube is undefined,  Please use a mass extension in the cube, with smooth_by_sfh=True!')
        return None
    else:
        sfhcube = np.sum(cube._sfhcube_, axis=3) # Remove metallicity dimension
        maskedcube = _maskobj(sfhcube, mask)
        return cube._intages_, maskedcube
        
        
        
        
def Spec(cube, mask=None, spec_type='model'):
    """Retrieves the spectrum of a timecube() object.
    
    :param cube: An instance of timecube().
    :param mask: A pixel mask.  This can be a tuple of a specific coordinate (y,x), or can be a 2D image, in which case mask=0 will be ignored, and all mask>0 will be given equal weighting.
    :param spec_type: If "model", returns Starlight's best-fit spectrum.  If "obs" or "observed", returns the pre-fitting emission-subtracted input spectrum.  If "weight", returns the weight spectrum given by Starlight to each wavelength in the fits.
    :return:
        :waves: Wavelengths at which the spectrum is sampled, in Angstrom.
        :spectrum: Total flux contained within the mask at each wavelength, in MaNGA flux units.
    """
    
    if cube._specarray_ == None:
        print('Cube has no spectra!  Ask Tom for the version with spectra.')
        return None
    else:
        if spec_type=='model':
            spec_in = 2
        elif spec_type=='obs' or spec_type=='observed':
            spec_in = 1
        elif spec_type=='weight':
            spec_in = 3
        else:
            print('spec_type must be "obs"/"observed", "model", or "weight"!')
            spec_in = 0
            return None
            
        if spec_in > 0:
            coords = np.unravel_index(np.argmin(cube.bpm), cube.bpm.shape)
            waves = cube._specarray_[coords[0],coords[1],0,:]
            speccube = cube._specarray_[:,:,spec_in,:]
            return waves, _maskobj(speccube,mask)
            
        
        
        
      
        

    
        
        
    
def metmap(slice_obj):
    """Measures the mean metallicity map of a timeslice instance.
    
    :param slice_obj: An instance of TimeSlice(), Range(), Cumulative() or Total().
    :return:
        :metmap: Light- or mass-weighted (depending on weight_type in the timecube()) mean metallicity of the slice_obj at each spaxel, in [M/H].
    """
    
    return _metfromarr(slice_obj.array, slice_obj._cube_._intmets_)
    
    
    
    
def agemap(slice_obj):
    """Measures the 10^{mean log(age)} map of a timeslice instance.
    
    :param slice_obj: An instance of TimeSlice(), Range(), Cumulative() or Total().
    :return:
        :agemap: Light- or mass-weighted (depending on weight_type in the timecube()) mean age of the slice_obj at each spaxel, in years.
    """
    
    return _agefromarr(slice_obj.array, slice_obj._cube_._intages_)
    
    
    
    
def image(slice_obj):
    """Measures the image map of a timeslice instance.
    
    :param slice_obj: An instance of TimeSlice(), Range(), Cumulative() or Total().
    :return:
        :image: Total light or mass (depending on weight_type in the timecube()) contained in the slice_obj at each spaxel, in MaNGA flux units or in Msol.
    """
    
    return _imfromarr(slice_obj.array)
        
        
        
      
        

    
        
        
    
def _maskobj(obj, mask): # Masks an object and returns the object summed in the spatial dimensions
    """Masks an object, and returns the object summed in the spatial dimensions.
    
    :param obj: Any 2D, 3D, or 4D array.
    :param mask: A spaxel mask to apply.  This can be a 2D array (where mask=0 is removed and all other values are given equal weighting), a single-spaxel coordinate tuple, or None (in which case all spaxels are weighted equally)
    :return:
        :masked_obj: (N-2)-dimension array where N is the dimensionality of obj.  Spatial coordinates flattened.
    """
    
    if np.sum(mask)==None:
        masked_obj = obj
    else:
        
        obj_shape = np.shape(obj)
        numdims = len(obj_shape)
        
        if type(mask)==tuple: # For specific spaxel coordinates, make a 2D mask
            m = np.zeros((obj_shape[0],obj_shape[1]), dtype=np.int)
            m[mask[0],mask[1]] = 1
            mask = m
        else: # For an array, set to binary array
            mask[mask!=0] = 1

        # Make the mask the correct dimensions
        if numdims==3:
            mask = np.expand_dims(mask, 2)
            mask = np.pad(mask, ((0,0),(0,0),(0,obj_shape[2]-1)), 'maximum')
        elif numdims==4:
            mask = np.expand_dims(mask, 2)
            mask = np.expand_dims(mask, 3)
            mask = np.pad(mask, ((0,0),(0,0),(0,obj_shape[2]-1),(0,obj_shape[3]-1)), 'maximum')
            
        # Mask the object
        masked_obj = obj*mask
        
    return np.nansum(masked_obj, axis=(0,1))
            
        
        
    
def _metfromarr(arr, metlist):
    """Measures the mean metallicity map of a timeslice array.
    
    :param arr: A timeslice instance's data array.
    :param metlist: List of metallicity sampling points of array.
    :return:
        :metmap: Light- or mass-weighted (depending on weight_type in the timecube()) mean metallicity of the slice_obj at each spaxel, in [M/H].
    """
    
    arr = np.sum(arr, axis=2) # Sum in age direction
    arrshape = np.shape(arr)
    
    arw = np.expand_dims(metlist, 0)
    arw = np.expand_dims(arw, 0)
    arw = np.pad(arw, ((0,arrshape[0]-1),(0,arrshape[1]-1),(0,0)), 'maximum')
    
    return np.sum(arw*arr, axis=2)/np.sum(arr, axis=2)
    
    
    
    
def _agefromarr(arr, agelist):
    """Measures the mean age map of a timeslice array.
    
    :param arr: A timeslice instance's data array.
    :param agelist: List of age sampling points of array.
    :return:
        :agemap: Light- or mass-weighted (depending on weight_type in the timecube()) mean metallicity of the slice_obj at each spaxel, in years.
    """
    
    arr = np.sum(arr, axis=3) # Remove metallicities
    arrshape = np.shape(arr)
    
    arw = np.expand_dims(np.log10(agelist), 0)
    arw = np.expand_dims(arw, 0)
    arw, np.pad(arw, ((0,arrshape[0]-1),(0,arrshape[1]-1),(0,0)), 'maximum')
    
    return 10**(np.sum(arw*arr, axis=2)/np.sum(arr, axis=2))
    
    
    
    
def _checkage(age, agelist):
    """Checks that an age value isn't outside the range of possible values.  If it is, replaces with closest valid age.
    
    :param age: An age, in years.
    :param agelist: List of age sampling points.
    :return:
        :cor_age: If age is valid, cor_age=age.  Otherwise cor_age is the closest valid age to age.
    """
    
    if age < np.min(agelist):
#        print('WARNING: Your requested age '+str(age)+' is too small.  Replacing with '+str(np.int(np.min(agelist)))+'.  Try altering the cutoff_age in the cube?')
        age = np.min(agelist)
    elif age > np.max(agelist):
#        print('WARNING: Your requested age '+str(age)+' is too large.  Replacing with '+str(np.int(np.max(agelist)))+'.')
        age = np.max(agelist)
    return age
    
    
    
def _arrofage(cube, age, side):
    """ONE-SIDED interpolation of a cube to a specific age slice.  Interpolates linearly in log-space.
    
    :param cube: A timecube() instance.
    :param age: Age to sample cube at, in years.
    :param side: If "older", finds the weighted flux or mass of the cube at the nearest valid sampling point older than age.  Otherwise, finds the weighted flux or mass of the cube at the nearest valid sampling point younger than age.
    :return:
        :arr: The (slice at age)'s contribution from the sampling point [side] than [age],
    """
    
    if age == np.min(cube._intages_):
        arr = cube._intcube_[:,:,[0],:] # WxHxMet
    elif age == np.max(cube._intages_):
        arr = cube._intcube_[:,:,[-1],:]
    else:        
    
        # Find the correct index in age list
        arr_ind = bisect.bisect(cube._intages_, age)
        
        # Find the relative weights to each side
        w_lo = np.log10(cube._intages_[arr_ind])-np.log10(age)
        w_hi = np.log10(age)-np.log10(cube._intages_[arr_ind-1])
        
        if side=='older':
            arr = (w_lo*cube._intcube_[:,:,[arr_ind-1],:]) / (w_lo+w_hi)
        else:
            arr = (w_hi*cube._intcube_[:,:,[arr_ind],:]) / (w_lo+w_hi)
            
    return arr
    
    
    
    
def _arrofrange(cube, lo_age, hi_age):
    """Total flux/mass array of a cube within an age range.
    
    :param cube: A timecube() instance.
    :param lo_age: Lower bound of ages, in years.  Exclusive.
    :param hi_age: Upper bound of ages, in years.  Exclusive.
    :return:
        :agelist: The age sampling in years of the data array.
        :data: The smoothed data array within the age range.
    """
    
    agelist = cube._intages_
    agemask = (agelist<hi_age) & (agelist>lo_age) # Absolute inequalities because we want to avoid double-counting end slices
    age_indices = [i for i, val in enumerate(agemask) if val]
    return agelist[age_indices], cube._intcube_[:,:,age_indices,:]




def _imfromarr(arr):
    """Total flux/mass within a data array.
    
    :param arr: A timeslice's data array.
    :return:
        :im: The total flux/mass image.
    """
    
    return np.sum(arr, axis=(2,3))
    
    
    

def _gauss(x, m, s):
    """Defines a Gaussian.
    
    :param x: Sampling points.
    :param m: Centre of Gaussian profile.
    :param s: Width of Gaussian.
    :return: Gaussian weights of length x.
    """
    
    return (1/(s*np.sqrt(2*np.pi)))*np.exp(-np.power(x-m,2) / (2*np.power(s,2)))
    
    
    
    
def _gauss2d(x, y, mx, my, sx, sy):
    """Defines a Gaussian.
    
    :params x, y: Sampling points.
    :params mx, my: Centre of Gaussian profile.
    :params sx, sy: Width of Gaussian.
    :return: Gaussian weights of length x.
    """
    
    xg,yg = np.meshgrid(x,y)
    
    return (1/(2*np.pi*sx*sy))*np.exp(-( (np.power(xg-mx,2) / (2*np.power(sx,2))) + (np.power(yg-my,2) / (2*np.power(sy,2))) ))
    
    

      
def _create4dweights(cube, ageweights, ax=2):
    """Turns 1D weights for each age into a 4D array of weights corresponding to each SSP.
    
    :param ageweights: Array of weights for each age.
    :return: 4D array of each weight.
    """
    
#    assert ax==0 or ax==2 or ax==3, \
#        "Error: ax="+str(ax)+".  Use ax=2 or ax=3."
    
    for axnum in range(1,4):
        ageweights = np.expand_dims(ageweights, axis=axnum) # Expands to 4D array
    ageweights = np.swapaxes(ageweights, 0, ax) # Puts the "age" axis into axis 2 (like weightar)
    
    shapediff = np.subtract(np.shape(cube),np.shape(ageweights))
    padshape = ((0,shapediff[0]),(0,shapediff[1]),(0,shapediff[2]),(0,shapediff[3]))
    ageweights = np.pad(ageweights, padshape, 'maximum')
    
    return ageweights