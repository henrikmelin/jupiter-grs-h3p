from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import h3ppy
#import time
import spiceypy as spice
import glob
import JWSTSolarSystemPointing as jssp
import ch4
reload(ch4)
from astropy.io import fits
import sys
import os
from astropy.table import Table

import warnings
warnings.filterwarnings("ignore")

# Load the JWST and Jupiter kernels
spice.furnsh('kernels/naif0012.tls')
spice.furnsh('kernels/pck00010.tpc') 
spice.furnsh('kernels/de430.bsp')
spice.furnsh('kernels/jup310.bsp')
spice.furnsh('kernels/jwst_horizons_20211225_20240221_v01.bsp')

# Request the CH4 line data from Hitran
#ch4list = ch4.get_hitran_table()
ch4list = Table.read('ch4_line_list.txt', format='ascii')

class JWSTExtractIons() : 
    def __init__(self, filename, ch4list, outdir = '.') : 
    
        self.geo      = jssp.JWSTSolarSystemPointing(filename)
        self.geometry = self.geo.full_fov()
        self.h3p      = h3ppy.h3p()
        self.ch4      = ch4.methane_fitting(ch4list)

        self.filename = filename
        self.outdir   = outdir
        
#        self.ncores = ncores
        self.cube = False
        self.line = False

    def get_sigma(self, wave, spec) : 
        '''
            Construct an error (sigma) array for the CH4 fitting.
            For the purpose of fitting the CH4 spectra whilst ignoring the H3+ wavelengths. 
        '''

        # Create a normalised H3+ spectrum at some resonable temperature
        self.h3p.set(wavelength = wave, temperature = 600, density = 1e16, R = 2700)
        m = self.h3p.model()
        m /= np.max(m)

        # Generate the sigma array
        sigma = np.zeros(wave.shape[0])
        sigma[:] = 0.05 * np.max(spec) + m * 0.5 * np.max(spec)
        return sigma

    def ch4_subfit(self, wave, spec) : 

        sigma = self.get_sigma(wave, spec)
        fit   = self.ch4.fit(wave, spec, sigma)
        vars  = self.ch4.vars
        errs  = self.ch4.errs
        # What about all those fitted parameters? 
        return fit, vars, errs

#    @staticmethod
    def ch4_pixel_fit(self, x, y) : 

        wave = self.geo.get_wavelength()

        # Correct the data using the calculated Doppler velocity
        corr = self.geometry[self.geo.map['doppler'], x, y]

        # Convert Mjy to W/cm2/sr/micron
        #spec = self.geo.convert(wave, self.geo.im[:, x, y]) # / self.geo.dm.meta.photometry.pixelarea_steradians
        spec = self.geo.im[:, x, y] # / self.geo.dm.meta.photometry.pixelarea_steradians
        spec[np.isnan(spec)] = 0

#        return wave, spec

        # Do nothing if there's no data
        if (np.nansum(spec) == 0) : return False, False

        waves     = []
        specs     = []
        residuals = []
        fits      = []
        ch4s      = []
        self.fits_vars = []
        self.fits_errs = []


        # There are four regions for which we wanna extract data
        for i in range(4) : 
            if (i == 0) : whl = np.argwhere((wave > 3.29) & (wave < 3.31))
            if (i == 1) : whl = np.argwhere((wave > 3.37) & (wave < 3.4))
            #if (i == 2) : whl = np.argwhere((wave > 3.4) & (wave < 3.458))
            if (i == 2) : whl = np.argwhere((wave > 3.4) & (wave < 3.43))
            if (i == 3) : whl = np.argwhere((wave > 3.528) & (wave < 3.56))

            # Extract the subregions
            subwave = wave[whl.flatten()]
            subspec = spec[whl.flatten()]

            # Just to make sure there are no nans in the data
            subspec[np.isnan(subspec)] = 0

            # Convert to proper units
            intensity = self.geo.convert(subwave / corr, subspec) * 1e4
            #intensity = subspec

            # Just to make sure there are no nans in the data
            intensity[np.isnan(intensity)] = 0

            # Fit the subregions
            fit, vars, errs = self.ch4_subfit(subwave / corr, intensity)
            methane = self.ch4.methane_only()

            # What's left after the fit should largely be H3+
            residual = intensity - fit

            # Save everything 
            for j, w in np.ndenumerate(subwave) : 
                waves.append(subwave[j] / corr)
                specs.append(intensity[j])
                residuals.append(residual[j])
                fits.append(fit[j])
                ch4s.append(methane[j])

            # Also save the fitted CH4 parameters, although largely nonsense
            for j, v in np.ndenumerate(vars) : 
                self.fits_vars.append(vars[j])                
                self.fits_errs.append(errs[j])    
                
        if (self.cube == True) : 
            self.ret_fits[:, x, y]      = np.array(fits)
            self.ret_specs[:, x, y]     = np.array(specs)
            self.ret_residuals[:, x, y] = np.array(residuals)
            self.ret_waves[:, x, y]     = np.array(waves)

            self.ret_vars[:, x, y]      = np.array(self.fits_vars)
            self.ret_errs[:, x, y]      = np.array(self.fits_errs)

        if (self.line == True) : 
            self.ret_fits[:, y]      = np.array(fits)
            self.ret_specs[:, y]     = np.array(specs)
            self.ret_residuals[:, y] = np.array(residuals)
            self.ret_waves[:, y]     = np.array(waves)
            self.ret_methane[:, y]     = np.array(ch4s)

            self.ret_vars[:, y]      = np.array(self.fits_vars)
            self.ret_errs[:, y]      = np.array(self.fits_errs)


        return np.array(waves), np.array(residuals)

    def fit_cube(self) : 

        # Fit a (random) pixel to get the size of the resultant spectrum
        wave, spec = self.ch4_pixel_fit(10, 10)

        # We also want to store the fit variables
        vars = np.array(self.fits_vars)

        self.cube = True

        # This is the return arrays
        self.ret_fits      = np.zeros((spec.shape[0], self.geo.im.shape[1], self.geo.im.shape[2]))
        self.ret_specs     = np.zeros((spec.shape[0], self.geo.im.shape[1], self.geo.im.shape[2]))
        self.ret_residuals = np.zeros((spec.shape[0], self.geo.im.shape[1], self.geo.im.shape[2]))
        self.ret_waves     = np.zeros((spec.shape[0], self.geo.im.shape[1], self.geo.im.shape[2]))

        self.ret_vars = np.zeros((vars.shape[0], self.geo.im.shape[1], self.geo.im.shape[2]))
        self.ret_errs = np.zeros((vars.shape[0], self.geo.im.shape[1], self.geo.im.shape[2]))

        # Iterate over spaxels
        for x in np.arange(self.geo.im.shape[1]) : 
            print(str(x) + ' ', end=' ')
            for y in np.arange(self.geo.im.shape[2]) : 
                wave, spec = self.ch4_pixel_fit(x, y)

        self.cube = False

        self.save_ch4_fits()

    def fit_line(self, x) : 

        # Fit a (random) pixel to get the size of the resultant spectrum
        wave, spec = self.ch4_pixel_fit(10, 10)

        # We also want to store the fit variables
        vars = np.array(self.fits_vars)

        self.line = True

        # This is the return arrays
        self.ret_fits      = np.zeros((spec.shape[0], self.geo.im.shape[2]))
        self.ret_methane   = np.zeros((spec.shape[0], self.geo.im.shape[2]))
        self.ret_specs     = np.zeros((spec.shape[0], self.geo.im.shape[2]))
        self.ret_residuals = np.zeros((spec.shape[0], self.geo.im.shape[2]))
        self.ret_waves     = np.zeros((spec.shape[0], self.geo.im.shape[2]))

        self.ret_vars = np.zeros((vars.shape[0], self.geo.im.shape[2]))
        self.ret_errs = np.zeros((vars.shape[0], self.geo.im.shape[2]))

        # Iterate over spaxels
        for y in np.arange(self.geo.im.shape[2]) : 
            wave, spec = self.ch4_pixel_fit(x, y)

        self.line = False

        self.save_ch4_fits()

    def save_ch4_fits(self) : 
        filename     = os.path.basename(self.filename)
        filename_out = filename.replace('.fits', '_ch4_fits.fits')

        # Create multi extension fits file
        hdu1 = fits.PrimaryHDU(self.ret_residuals, header = self.geo.hdr)
        hdu2 = fits.ImageHDU(self.ret_fits)
        hdu3 = fits.ImageHDU(self.ret_specs)
        hdu4 = fits.ImageHDU(self.ret_waves)
        hdu5 = fits.ImageHDU(self.ret_methane)
        hdu6 = fits.ImageHDU(self.ret_vars)
        hdu7 = fits.ImageHDU(self.ret_errs)

        hdul = fits.HDUList([hdu1, hdu2, hdu3, hdu4, hdu5, hdu6, hdu7])
        hdul.writeto(self.outdir + '/' + filename_out, overwrite = True)
        print(filename_out)

    def flatfield_data(self, flatfield) : 
        im = self.geo.im
        for i in range(im.shape[0]) :
            im[i, :, :] = im[i, :, :] / flatfield[i, :, :] 
        im[np.isnan(im)] = 0
        self.geo.im = im


if (len(sys.argv) == 1) : 
    print('Usage: ERS_GRS_extract_ions.py file_nbr line_nbr ')
else : 

    datadir = 'data/G395H/'
    files = sorted(glob.glob(datadir + '*s3d.fits'))

    flatfields = fits.getdata('G395H_flatfield.fits')

    im = fits.getdata(files[0])
    print('IFU lines: ' + str(im.shape[1]))
    print('IFU length: ' + str(im.shape[1] * im.shape[2]))

    line = int(sys.argv[2]) - 1
    file = files[int(sys.argv[1]) - 1]
    print('Processing ' + file)
    print('Processing line ' + str(line))
    outdir  = 'output/line_' + str(line)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    ions = JWSTExtractIons(file, ch4list, outdir = outdir)
    ions.flatfield_data(flatfields)
    ions.fit_line(line)






