import h3ppy
import numpy as np
from astropy.io import fits
import glob
import sys


line = int(sys.argv[1]) - 1
datadir = 'output/line_' + str(line) + '/'
files = sorted(glob.glob(datadir + '*ch4_fits.fits'))
print(datadir)
print(files)

im = fits.getdata(files[0])
ret = np.zeros((im.shape[1], 8))

#wave = its.getdata('wavelength.fits')

for f in files : 
    h3p = h3ppy.h3p()
    fname = f.replace('_ch4_fits.fits', '_h3p_fits.fits')
    im = fits.getdata(f)
    waves = fits.getdata(f, ext = 3)

    for y in range(im.shape[1]) : 
        spec = im[:, y]
        wave = waves[:, y]
        if (np.nansum(spec) == 0) : continue
        h3p.set(temperature = 700, density = 1e16, R = 2700, data = spec, wavelength = wave)
        #fit = h3p.fit(params_to_fit = ['sigma_0', 'offset_0']) 
        #fit = h3p.fit(params_to_fit = ['temperature', 'density']) 
        fit = h3p.fit() #   params_to_fit = ['temperature', 'density']) 
        vars, errs = h3p.get_results(verbose = False)
        if not vars : continue

        ret[y, 0] = vars['temperature']
        ret[y, 1] = errs['temperature']
        ret[y, 2] = vars['density']
        ret[y, 3] = errs['density']
        ret[y, 4] = vars['sigma_0']
        ret[y, 5] = vars['offset_0']
        ret[y, 6] = vars['background_0']
        ret[y, 7] = h3p.total_emission()
    
    fits.writeto(fname, ret, overwrite = True)
