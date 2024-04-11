from codecs import latin_1_decode
import numpy as np
from jwst import datamodels
from astropy.io import fits
import spiceypy as spice
from scipy.interpolate import griddata
from astropy import units as u
from astropy import constants as const
import logging

class JWSTSolarSystemPointing: 

    def __init__(self, file, emission_altitude = 0, target = '', arcsec_limit = 0, 
            radec_offset = [0.0, 0.0], observatory = 'JWST', fixed_slit = 0 ) : 
        '''
        A class that calculates the geometry of JWST Solar System observations

        Parameters
        ----------
        file : string
            The filename of the observation 

        emission_altitude : float
            The altitude above the 1 bar level in km to do the projection at

        target : string
            Specify a target within the frame. The default is set in the datamodels. 
            E.g. when looking at Jupiter data you may want map a moon instead.
            
        arcsec_limit : float
            Limit the geometry calculations withing a radii of the target centre. This can 
            speed things up somewhat.

        radec_offset : float array
            Shift the pixel coordinates in RA and DEC in arcseconds. The JWST pointing is only accurate to ~0.1"
            and so a shift may sometimes be appropriate. 

        mslit : intercept
            Specify which fixed slit to use

        '''

        # Configure logging
        logging.basicConfig(level=logging.WARNING)

        self.arcsec_limit = arcsec_limit
        self.radec_offset = radec_offset

        # Load the appropriate data model
        # Gonna need the right datamodel for each set of data        
        # https://jwst-pipeline.readthedocs.io/en/latest/jwst/datamodels/index.html
        self.hdr = fits.getheader(file, 'PRIMARY')

        # Store the datamodel        
        self.datamodel    = self.hdr['DATAMODL']
        self.dm  = getattr(datamodels, self.datamodel)(file) 

        # Store the data - need to treat the MultiSlitModel differently
        if (self.datamodel == 'MultiSlitModel') :
            self.dm_master = self.dm
            self.dm = self.dm.slits[fixed_slit]
        self.im  = self.dm.data.copy()

        self.observatory = observatory
        if (target) : self.target = target
        else : self.target = self.dm.meta.target.catalog_name

        self.instrument  = self.dm.meta.instrument.name
        self.framestring = 'IAU_' + self.target
        self.iref        = 'J2000'
        self.abcorr      = 'LT'
        
        self.id_obs      = spice.bodn2c(self.observatory)
        self.id_target   = spice.bodn2c(self.target)            
        
        # Determine the mid-point of the observation
        self.obs_start   = self.hdr['DATE-BEG']
        self.obs_end     = self.hdr['DATE-END']
        self.et_start    = spice.str2et(self.hdr['DATE-BEG'])
        self.et_end      = spice.str2et(self.hdr['DATE-END'])
        self.et          = (self.et_start + self.et_end) / 2.0
        
        # Generate human readable versions of the mid-point et
        self.obs_mid_doy = spice.et2utc(self.et, 'D', 0).replace('// ', '')
        self.obs_mid_iso = spice.et2utc(self.et, 'ISOC', 0).replace('T', ' ')

        # Define the output names and what are angles
        self.keys   = ['lat', 'lon', 'lat_limb', 'lon_limb', 'lat_graphic', 'phase', 'emission', 'incidence', 'azimuth', 'localtime', 'distance_limb', 'distance_rings', 'lon_rings', 'ra', 'dec', 'radial_velocity', 'doppler', 'localtime_limb']
        self.angles = ['lat', 'lon', 'lat_limb', 'lon_limb', 'lat_graphic', 'phase', 'emission', 'incidence', 'azimuth', 'lon_rings']

        # Create a reciprocal map to the keys
        self.map = {}
        for key, value in enumerate(self.keys) : self.map[value] = key

        self.set_emission_altitude(emission_altitude)
        self.target_location()
        
    def set_emission_altitude(self, emission_altitude) : 
        """Set the altitude of the reference spheroid, relative to IAU 1 bar surface, in km."""        

        self.emission_altitude = emission_altitude
        # Get the radius of the planet + optional altitude offset
        self.radii = spice.bodvar(self.id_target, 'RADII', 3)
        self.radii[0] = self.radii[0] + self.emission_altitude * self.radii[1] / self.radii[2]
        self.radii[1] = self.radii[1] + self.emission_altitude * self.radii[1] / self.radii[2]
        self.radii[2] = self.radii[2] + self.emission_altitude
        self.flattening = ( self.radii[0] - self.radii[2] ) / self.radii[0]     
        
    def target_location(self) : 
        
        # Get the position of the target relative to the obervatory
        self.pos_target, self.light_time = spice.spkpos(self.target, self.et, 
                        self.iref, self.abcorr, self.observatory)
        
        ###
        point, epoch, vector = spice.subpnt('NEAR POINT/ELLIPSOID', self.target, self.et, self.framestring, self.abcorr, self.observatory)
        distance, ra, dec = spice.recrad(point)
    
                
        self.ra_sub = np.rad2deg(ra) % 360
        self.dec_sub = np.rad2deg(dec)

        distance, lat, lon = spice.reclat(point)
        self.lon_sub = np.rad2deg(lat) 
        self.lat_sub = np.rad2deg(lon)
        ###
        
        
        # Convert position to distance, RA, dec
        self.distance, ra, dec = spice.recrad(self.pos_target)
        d, self.lon_obs, self.lat_obs = spice.reclat(self.pos_target)
        self.ra_target = np.rad2deg(ra)
        self.dec_target = np.rad2deg(dec)
        
        # Create the conversion from J2000 to the target frame
        self.i2p   = spice.pxform(self.iref, self.framestring, self.et - self.light_time)
        self.scloc = np.matmul(-self.i2p, self.pos_target) 
        
        # Get the subsolar coordinates
        point, epoch, vector = spice.subslr('NEAR POINT/ELLIPSOID', self.target, self.et, self.framestring, self.abcorr, self.observatory)
        distance, self.lon_sun, self.lat_sun = spice.reclat(point)

        self.lon_sun_orig = self.lon_sun
        self.lon_sun = np.rad2deg(self.lon_sun) % 360
        self.lat_sun = np.rad2deg(self.lat_sun)


        
        
    def pixel_params(self, ra, dec, vector_target = '') : 
        '''
        lat : degrees
            Planetocsentric latitude
        lon : degrees
            West Longitude 
        distance_limb : km
            The distance between the pixel and the 1 bar limb. The 1 bar limb is defined as 0 km, 
            and negative distances are on the limb on the planet, positive ones are above the limb.
            Note that, e.g. if you want to project data to a different altitude, use the emission_altitude
            keyword in the initialisation of the gometry object.
        lat_limb : degrees
            Planetocentric latitude of the point on the limb closest to the pixel look vector.
        lon_limb : degrees
            West longitude of the point on the limb closest to the pixel look vector.
        lat_graphic : degrees
            Planetgraphic latitude.
        phase : degrees
            Phase angle
        emissions : degrees
            Emission angle
        incidence : degrees
            Incidence angle
        azimuth : degrees
            Azimuth angle
        localtime : decimal hours
            The localtime of a pixel
        distance_rings : km
            The distance from the centre of the planet in the equatorial (ring) plane
        lon_rings : degrees
            The West longitude of the the point on the equatorial (ring) plane
        ra : degrees
            Right Acension 
        dec : degrees
            Declination
        radial_velocity : km/s
            Radial velocity of the surface point relative to the observer. Positive
            values correspond to motion awaty from the observer.
        doppler : dimensionless
            Doppler factor calculated from radial velocity. Calculated as
            sqrt((1 + v/c)/(1 - v/c)) where v is radial velocity away from the observer
            and c is the speed of light. To correct the wavelength scale, divide the scale 
            provided by JWSTSolarSystemPointing.get_wavelength() with the doppler number.
        '''
        # Set up the return variable
        ret = {}
        for key in self.keys : 
            ret[key] = np.nan
                
        # Get the pixel RA and DEC from the datamodel
        #if (len(self.im.shape) == 2) : coords = self.dm.meta.wcs(x, y)
        #else : coords = self.dm.meta.wcs(x, y, 10)
        
        # If we are only doing a radius around the target
        if (self.arcsec_limit != 0) : 
            dist = np.sqrt((self.ra_target - ra)**2 + (self.dec_target - dec)**2)*3600.0
            if (dist > self.arcsec_limit) : return ret
        
        if (vector_target == '') : 
            # Calculate a look vector based on the coordinates and convert to target frame
            vector_J2000  = spice.radrec(self.distance, np.deg2rad(ra), np.deg2rad(dec))
            vector_target = np.matmul(self.i2p, vector_J2000)

        # Get the closest point on the vector to the planet
        origin = np.array([0.0, 0.0, 0.0])
        nearpoint, rayradius = spice.nplnpt(self.scloc, vector_target, origin)

        if (np.sum(nearpoint) != 0) :
            # Calculate the point in the surface closest to that point
            normal = spice.surfpt(origin, nearpoint, self.radii[0], self.radii[1], self.radii[2])

            # Get the latitude and longitud e of the point on the limb
            d, ret['lon_limb'], ret['lat_limb'] = spice.reclat(nearpoint)        
        
            # Calculate the height above the limb 
            ret['distance_limb'] = rayradius - spice.vnorm(normal)
            
            # Get the localtimes of the occ lons, and convert to decimal hours 
            hr, min, sc, time, ampm = spice.et2lst(self.et - self.light_time, self.id_target, ret['lon_limb'], 'PLANETOCENTRIC')
            ret['localtime_limb'] = hr + min / 60.0 + sc / 3600.0

        # Now get the ring-plane projection
        ringplane = spice.nvc2pl(np.array([0.0, 0.0, 1.0]), 0.0)
        nxpt, ring_intercept = spice.inrypl(self.scloc, vector_target, ringplane)        
        ret['distance_rings'], ret['lon_rings'], lat_rings = spice.reclat(ring_intercept)
        
#        ret['distance_rings'] = spice.vnorm(ring_intercept)
        
        # Test if the pixel vector intersects with our target surface 
        try: 
            point = spice.surfpt(self.scloc, vector_target, self.radii[0], self.radii[1], self.radii[2])
            intercept = True
        except: 
            intercept = False    
    
        if (intercept) : 
        
            # Get some angles 
            ret['phase'], ret['incidence'], ret['emission'] = spice.illum(self.target, self.et , self.abcorr, self.observatory, point)

            # From these angles, calculate the azimut angle (as defined in the NEMESIS manual)
            # https://nemesiscode.github.io/manuals.html
            # Based on zcalc_aziang.pro
            a = np.cos(ret['phase']) - np.cos(ret['emission']) * np.cos(ret['incidence'])
            b = np.sqrt(1.0 - np.cos(ret['emission'])**2) * np.sqrt(1.0 - np.cos(ret['incidence'])**2)            
            ret['azimuth'] = np.pi - np.arccos(a/b)

            # Calculate the planetocentric coordinates 
            distance, ret['lon'], ret['lat'] = spice.reclat(point)
            #ret['distance'] = spice.vnorm(self.scloc - point)
            
             # Calculate the planetographic coordinates
            lon_graphic, ret['lat_graphic'], bodyintercept = spice.recpgr(self.target, point, self.radii[0], self.flattening)
            
            # Get the localtime, and convert to decimal hours 
            hr, min, sc, time, ampm = spice.et2lst(self.et - self.light_time, self.id_target, ret['lon'], 'PLANETOCENTRIC')
            ret['localtime'] = float(hr) + float(min) / 60.0 + float(sc) / 3600.0
 #           ret['localtime'] = self.calculate_localtime(ret['lon'])

 #           sun_angle_diff = (np.rad2deg(ret['lon']) - np.rad2deg(self.lon_sun_orig)) % 360
  #          if (sun_angle_diff > 180) : sun_angle_diff -= 360.0
#            print(sun_angle_diff, ret['lon'], self.lon_sun_orig)
   #         ret['localtime'] = 12.0 + (sun_angle_diff) / 15.0
#
            #print(ret['localtime'])
            #print(hr, min, sc, time, ampm)
            # Get the radial velocity for doppler shift calculation
            state, lt = spice.spkcpt(
                trgpos=point,
                trgctr=self.target,
                trgref=self.framestring,
                et=self.et,
                outref=self.iref,
                refloc='OBSERVER',
                abcorr=self.abcorr,
                obsrvr=self.observatory,
            )
            position = state[:3]
            velocity = state[3:]
            # dot the velocity with the normalised position vector to get radial component
            radial_velocity = np.dot(position, velocity) / np.linalg.norm(position)
            # calculate doppler shift factor from radial velocity
            beta = radial_velocity / spice.clight()
            doppler = np.sqrt((1 + beta) / (1 - beta))
            ret['radial_velocity'] = radial_velocity
            ret['doppler'] = doppler


        # For the angles, convert radians to degrees
        for key in self.angles : 
            if (ret[key] != np.nan) : ret[key] = np.rad2deg(ret[key])

        # Makes sure longitudes wrap 0 to 360, spice returns the Earth-like -180 to 180. 
        # All longitudes are specifically West! 
        longitudes = ['lon', 'lon_limb', 'lon_rings']
        for key in longitudes : ret[key] = (360 - ret[key]) % 360

        return ret
        
    def calculate_localtime(self, lon) : 

        # Calculate the angel between the sub-solar longitude and the longitude
        sun_angle_diff = (np.rad2deg(lon) - np.rad2deg(self.lon_sun_orig)) % 360
    
        # Wrap the numbrs -180 < diff < 180
        if (sun_angle_diff > 180) : sun_angle_diff -= 360.0

        # Calculate the local-time
        localtime = 12.0 + (sun_angle_diff) / 15.0
        return localtime

    
    def data_to_header(self) : 
        hdr = self.hdr
        for i, key in enumerate(self.keys) : 
            hdr['KEY_' + str(i)] = key
        return hdr
        
    def full_fov_fixed_slit(self) : 
        '''
            Calculate the geometry for the fixed slits.
        '''
    
        sz = np.flip(self.im.shape)

        # Get the RA and Dec from the datamodel
        valx = np.arange(0, sz[1])
#        detector_to_world = self.dm.meta.wcs.get_transform('slit_frame', 'world')
        coords = self.dm.meta.wcs(valx, valx)
#        coords = detector_to_world(valx, valx, valx)
        self.ras = coords[0] #(np.reshape(coords[0], (sz[1], sz[0])))
        self.decs = coords[1] #(np.reshape(coords[1], (sz[1], sz[0])))

        # Apply any shift in RA and DEC in arcseconds
        self.ras += self.radec_offset[0] / 3600.0
        self.decs += self.radec_offset[1] / 3600.0

        # Make our output array with extra room for RA and Dec
        output = np.zeros([len(self.keys) + 2, sz[1]])
        for x in range(sz[1]) :
            ret = self.pixel_params(self.ras[x], self.decs[x])
            for i, key in enumerate(ret) : 
                output[i, x] = ret[key]

        # Add RA and Dec
        output[self.keys.index('ra'), : :] = self.ras
        output[self.keys.index('dec'), : :] = self.decs

        self.geometry_cube = output
        return output
    
    def get_ra_dec(self) : 
        '''
            Get the RA and DEC coordinates from the WCS datamodel        
        '''
        sz = np.flip(self.im.shape)

        valx = np.zeros([sz[1], sz[0]])
        valy = np.zeros([sz[1], sz[0]])
        for x in range(sz[1]) : 
            for y in range(sz[0]) : 
                valx[x, y] = x
                valy[x, y] = y

        if (len(self.im.shape) == 2) : 
            coords = self.dm.meta.wcs(valy.flatten(), valx.flatten())
        else : coords = self.dm.meta.wcs(valy.flatten(), valx.flatten(), 10)

        self.ras = (np.reshape(coords[0], (sz[1], sz[0])))
        self.decs = (np.reshape(coords[1], (sz[1], sz[0])))

        return self.ras, self.decs
    
    def full_fov(self) : 
        '''
            Calculate the geometry for an observation. 
        '''
        sz = np.flip(self.im.shape)

        valx = np.zeros([sz[1], sz[0]])
        valy = np.zeros([sz[1], sz[0]])
        for x in range(sz[1]) : 
            for y in range(sz[0]) : 
                valx[x, y] = x
                valy[x, y] = y
        
        if (len(self.im.shape) == 2) : 
            coords = self.dm.meta.wcs(valy.flatten(), valx.flatten())
        else : coords = self.dm.meta.wcs(valy.flatten(), valx.flatten(), 10)
        self.ras = (np.reshape(coords[0], (sz[1], sz[0])))
        self.decs = (np.reshape(coords[1], (sz[1], sz[0])))

        # Apply any shift in RA and DEC in arcseconds
        self.ras += self.radec_offset[0] / 3600.0
        self.decs += self.radec_offset[1] / 3600.0

        # Make our output array with extra room for RA and Dec
        output = np.zeros([len(self.keys) + 2, sz[1], sz[0]])
        for x in range(sz[1]) :
            for y in range(sz[0]) :
                ret = self.pixel_params(self.ras[x, y], self.decs[x, y])
                for i, key in enumerate(ret) : 
                    output[i, x, y] = ret[key]

        # Add RA and Dec
        output[self.keys.index('ra'), : :] = self.ras
        output[self.keys.index('dec'), : :] = self.decs

        self.geometry_cube = output
        return output
    
    def get_param(self, key) :
        if (key in self.map) : 
            return self.geometry_cube[self.map[key], :, :]
        else : 
            logging.error('Error in get_param(): key "' + str(key) + '" does not exist! Available keys are: ' + ', '.join(self.keys))
            return False

    def get_wavelength(self, xpixel = 0, ypixel = 0) : 
        '''
            Get the wavelength scale from the datamodels.
        '''
        wave_pixels = np.arange(self.im.shape[0])
        
        # The different datamodels for the different settings take diferent types of input,
        # so will have to treat them differently.
        if ((self.datamodel == 'MultiSlitModel') | (self.datamodel == 'SlitModel')) :  
            spatial_pixels = np.arange(self.im.shape[0])           
            waves = []
            for i in range(int(np.ceil(self.im.shape[1] / self.im.shape[0]))) : 
                wave_pixels = np.arange(self.im.shape[0]) + i * self.im.shape[0]
                ras, decs, wave = self.dm.meta.wcs(wave_pixels, spatial_pixels)
                waves.append(wave)
            wave = np.ravel(np.array(waves))
            wave = wave[np.isnan(wave) == False]
        else : ras, decs, wave = self.dm.meta.wcs(ypixel, xpixel, wave_pixels)
        return wave

    def get_spk_coverage(self, spkfile) : 
        '''
            Return the date interval for which an SPK file is valid. 
        '''
        cover        = spice.spkcov(spkfile, self.id_obs)

        # The JWST reconstructed kernels only have one window, so using 0
        time         = spice.wnfetd(cover, 0)

        # Format the start and end times
        window_start = spice.timout(time[0], "YYYY MON DD HR:MN:SC")
        window_end   = spice.timout(time[1], "YYYY MON DD HR:MN:SC")

        return window_start, window_end

    def convert(self, wave, spec) :
        '''
            Convert from mJy/sr to Wm-2sr-1micron-1. 
        '''
        c = 2.99792458e+8
        spec = spec.copy() / (wave * 1e-6)**2 * c * 1.0e-26
        return spec
        
    def get_delta_ra_dec_arcsec(self) : 
 
        dra = (self.geometry_cube[13, :, :] - self.ra_target) * 3600.0
        ddec = (self.geometry_cube[14, :, :] - self.dec_target) * 3600.0
        return dra, ddec

    def save_spx(self, x, y, wstart = 0, wend = 0, fwhm = 4.0/2600.0, erradd = 0.0) : 
        ''' Create a basic NEMESIS spx input file '''
        wave     = self.get_wavelength(xpixel = x, ypixel = y) 
        spec     = self.convert(wave, self.im[:, y, x] / 3.33000e+08) 
        error    = self.convert(wave, self.dm.err[:, y, x] / 3.33000e+08) 
        lat      = self.geometry_cube[self.map['lat_graphic'], y, x]
        lon      = self.geometry_cube[self.map['lon'], y, x]
        phase    = self.geometry_cube[self.map['phase'], y, x]
        emission = self.geometry_cube[self.map['emission'], y, x]
        azimuth  = self.geometry_cube[self.map['azimuth'], y, x]

        # Add a multipicative error correction
        error += erradd 

        if (wend > 0) : 
            spec  = spec[wave > wstart]
            error = error[wave > wstart]
            wave  = wave[wave > wstart]
            spec  = spec[wave < wend]
            error = error[wave < wend]
            wave  = wave[wave < wend]

        ngeom = 1
        nconv = wave.shape[0]
        nav = 1
        wgeom = 1

        # Generate the header
        header = []
        header.append([fwhm, lat, lon, ngeom])
        header.append([nconv])
        header.append([nav])
        header.append([lat, lon, phase, emission, azimuth, wgeom])

        filename = self.dm.meta.observation.obs_id + '_lon_{:.2f}'.format(lon) + '_lat_{:.2f}.spx'.format(lat)  
        with open(filename, 'w') as f:
            for line in header : 
                f.write('\t'.join('{:.3f}'.format(x) for x in line))
                f.write('\n')
            for i, w in np.ndenumerate(wave) : 
                line = [wave[i], spec[i], error[i]]
                f.write('\t'.join('{:.6e}'.format(x) for x in line))
                f.write('\n')
                
                
                
                
                
                
                
                
class JWSTSolarSystemPlanning(JWSTSolarSystemPointing) : 
    def __init__(self, obstime, emission_altitude = 0, target = '', arcsec_limit = 0, 
            radec_offset = [0.0, 0.0], observatory = 'JWST' ) : 
        '''
        A class that calculates the geometry of planned (future) JWST Solar System observations

        Parameters
        ----------
        file : string
            The filename of the observation 

        emission_altitude : float
            The altitude above the 1 bar level in km to do the projection at

        target : string
            Specify a target within the frame. The default is set in the datamodels. 
            E.g. when looking at Jupiter data you may want map a moon instead.
            
        arcsec_limit : float
            Limit the geometry calculations withing a radii of the target centre. This can 
            speed things up somewhat.

        radec_offset : float array
            Shift the pixel coordinates in RA and DEC in arcseconds. The JWST pointing is only accurate to ~0.1"
            and so a shift may sometimes be appropriate. 

        '''

        # Configure logging
        logging.basicConfig(level=logging.WARNING)

        self.arcsec_limit = arcsec_limit
        self.radec_offset = radec_offset

        # Load the appropriate data model
        # Gonna need the right datamodel for each set of data        
        # https://jwst-pipeline.readthedocs.io/en/latest/jwst/datamodels/index.html
#        self.hdr = fits.getheader(file, 'PRIMARY')
#        model    = self.hdr['DATAMODL']
#        self.dm  = getattr(datamodels, model)(file) 
#        self.im  = self.dm.data.copy()

        self.observatory = observatory
        if (target) : self.target = target
 #       else : self.target = self.dm.meta.target.catalog_name

#        self.instrument  = self.dm.meta.instrument.name
        self.framestring = 'IAU_' + self.target
        self.iref        = 'J2000'
        self.abcorr      = 'LT'
        
        self.id_obs      = spice.bodn2c(self.observatory)
        self.id_target   = spice.bodn2c(self.target)            
        
        # Determine the mid-point of the observation
#        self.obs_start   = self.hdr['DATE-BEG']
#        self.obs_end     = self.hdr['DATE-END']
#        self.et_start    = spice.str2et(self.hdr['DATE-BEG'])
#        self.et_end      = spice.str2et(self.hdr['DATE-END'])
        self.et          = spice.str2et(obstime)
        
        # Generate human readable versions of the mid-point et
        self.obs_mid_doy = spice.et2utc(self.et, 'D', 0).replace('// ', '')
        self.obs_mid_iso = spice.et2utc(self.et, 'ISOC', 0).replace('T', ' ')

        # Define the output names and what are angles
        self.keys   = ['lat', 'lon', 'lat_limb', 'lon_limb', 'lat_graphic', 'phase', 'emission', 'incidence', 'azimuth', 'localtime', 'distance_limb', 'distance_rings', 'lon_rings', 'ra', 'dec', 'radial_velocity', 'doppler', 'dra', 'ddec']
        self.angles = ['lat', 'lon', 'lat_limb', 'lon_limb', 'lat_graphic', 'phase', 'emission', 'incidence', 'azimuth', 'lon_rings']

        # Create a reciprocal map to the keys
        self.map = {}
        for key, value in enumerate(self.keys) : self.map[value] = key

        self.set_emission_altitude(emission_altitude)
        self.target_location()
        
        
        self.centre = self.pixel_params(self.ra_target, self.dec_target)

        # Set some defaults  
        self.rotation = 0
        self.mosaic   = False
        
        
        timeline = {}
        timeline['visit_slew'] = 0.0
        timeline['visit_overheads_before'] = 0.0
        timeline['visit_overheads_after'] = 0.0
        timeline['guide_star_acquisition'] = 0.0
        timeline['filter'] = 0.0
        timeline['oss_compile'] = 0.0
        timeline['msa_change'] = 0.0
        timeline['mechanism_move'] = 0.0
        timeline['exposure_overhead_before'] = 0.0
        timeline['exposure_overhead_after'] = 0.0
        timeline['exposure'] = 0.0
  
    def generate_fov_ra_dec(self, fov_size, steps = 100) : 

        arcsec  = np.arange(-fov_size / 2.0, fov_size/2.0, fov_size / steps)
        degrees = arcsec / 3600.0
        ra      = self.ra_target + degrees
        dec     = self.dec_target + degrees
        return ra, dec
        
    def generate_fov(self, fov_size, steps = 100) : 
    
        ra, dec = self.generate_fov_ra_dec(fov_size, steps = steps)
        ras, decs = np.meshgrid(ra, dec)

        # Make our output array with extra room for RA and Dec
        output = np.zeros([len(self.keys) + 4, steps, steps])
        for x in range(steps) :
            for y in range(steps) :
                ret = self.pixel_params(ras[x, y], decs[x, y])
                for i, key in enumerate(ret) : 
                    output[i, x, y] = ret[key]

        # Add RA and Dec
        output[self.keys.index('ra'), : :] = ras
        output[self.keys.index('dec'), : :] = decs
        
        # Add a ra and dee difference from the centre of target
        output[self.keys.index('dra'), : :] = (ras - self.ra_target) * 3600.0
        output[self.keys.index('ddec'), : :] = (decs - self.dec_target) * 3600.0
        
        return output
        
    def fov_geometry(self, ras, decs) : 
    
        # Make our output array with extra room for RA and Dec
        output = np.zeros([len(self.keys) + 2, ras.shape[0], ras.shape[1]])
        for x in range(ras.shape[0]) :
            for y in range(ras.shape[1]) :
                ret = self.pixel_params(ras[x, y], decs[x, y])
                for i, key in enumerate(ret) : 
                    output[i, x, y] = ret[key]

        # Add RA and Dec
        output[self.keys.index('ra'), : :] = ras
        output[self.keys.index('dec'), : :] = decs
    
        return output
        
    def instrument_fov(self, instrument_name = '', geometry = 'corners') :

        if (instrument_name != '') : self.instrument_name = instrument_name
        
        
        if (self.instrument_name == 'nirspec_ifu') : 
            xs = 3.0 / 3600.0 / 2.0
            if (geometry == 'corners') : 
                fov_ra  = np.array([-xs, -xs, +xs, +xs, -xs])
                fov_dec = np.array([-xs, +xs, +xs, -xs, -xs])
            elif (geometry == 'full') : 
                ra = np.arange(-xs, xs, 0.1/3600.0)
                dec = np.arange(-xs, xs, 0.1/3600.0)    
                fov_ra, fov_dec = np.meshgrid(ra, dec)
            self.rotation = 138.5

        elif (self.instrument_name == 'miri_mrs_ch1') : 
            xs = 3.2 / 3600.0 / 2.0
            ys = 3.7 / 3600.0 / 2.0
            if (geometry == 'corners') : 
                fov_ra  = np.array([-xs, -xs, +xs, +xs, -xs])
                fov_dec = np.array([-ys, +ys, +ys, -ys, -ys])
            elif (geometry == 'full') : 
                ra = np.arange(-xs, xs, 0.176 / 3600.0)
                dec = np.arange(-ys, ys, 0.196 / 3600.0)    
                fov_ra, fov_dec = np.meshgrid(ra, dec)
            self.rotation = 8.4

        elif (self.instrument_name == 'miri_mrs_ch2') : 
            xs = 4.0 / 3600.0 / 2.0
            ys = 4.8 / 3600.0 / 2.0
            if (geometry == 'corners') : 
                fov_ra  = np.array([-xs, -xs, +xs, +xs, -xs])
                fov_dec = np.array([-ys, +ys, +ys, -ys, -ys])
            elif (geometry == 'full') : 
                ra = np.arange(-xs, xs, 0.277 / 3600.0)
                dec = np.arange(-ys, ys, 0.196 / 3600.0)    
                fov_ra, fov_dec = np.meshgrid(ra, dec)
            self.rotation = 8.2

        elif (self.instrument_name == 'miri_mrs_ch3') : 
            xs = 5.2 / 3600.0 / 2.0
            ys = 6.2 / 3600.0 / 2.0
            if (geometry == 'corners') : 
                fov_ra  = np.array([-xs, -xs, +xs, +xs, -xs])
                fov_dec = np.array([-ys, +ys, +ys, -ys, -ys])
            elif (geometry == 'full') : 
                ra = np.arange(-xs, xs, 0.387 / 3600.0)
                dec = np.arange(-ys, ys, 0.245 / 3600.0)    
                fov_ra, fov_dec = np.meshgrid(ra, dec)
            self.rotation = 7.5
            
        elif (self.instrument_name == 'miri_mrs_ch4') : 
            xs = 6.6 / 3600.0 / 2.0
            ys = 7.7 / 3600.0 / 2.0
            if (geometry == 'corners') : 
                fov_ra  = np.array([-xs, -xs, +xs, +xs, -xs])
                fov_dec = np.array([-ys, +ys, +ys, -ys, -ys])
            elif (geometry == 'full') : 
                ra = np.arange(-xs, xs, 0.645 / 3600.0)
                dec = np.arange(-ys, ys, 0.273 / 3600.0)    
                fov_ra, fov_dec = np.meshgrid(ra, dec)
            self.rotation = 8.3
            
        else : 
            print('Available instrument modes are:')
            modes = ['nirspec_ifu', 
                     'miri_mrs_ch1', 
                     'miri_mrs_ch2', 
                     'miri_mrs_ch3', 
                     'miri_mrs_ch3']
            print(modes)

        return fov_ra, fov_dec

    def set(self, **kwargs) : 
        for key, value in kwargs.items() :
            if (key == 'instrument_name') : self.instrument_name = value
            if (key == 'dither_pattern') : self.pattern_name = value
            if (key == 'rotation') : self.rotation = value

         #   if (key == 'time_visit_slew') : self.time_visit_slew = value
         #   if (key == 'time_visit_overhead') : self.time_visit_overhead = value
         #   if (key == 'time_filter') : self.time_visit_overhead = value
         #   if (key == 'time_oss_compile') : self.time_oss_compile = value
            
    def rotate(self, x, y, xo, yo, theta): #rotate x,y around xo,yo by theta (rad)
        xr = np.cos(np.deg2rad(theta))*(x-xo)-np.sin(np.deg2rad(theta))*(y-yo)   + xo
        yr = np.sin(np.deg2rad(theta))*(x-xo)+np.cos(np.deg2rad(theta))*(y-yo)  + yo
        return xr,yr

    def dither_pattern(self, pattern_name = '') : 

        cycle_file = 'models/JWST_NIRSpec_cycling.txt'

        if (pattern_name != '') : self.pattern_name = pattern_name

        if (self.pattern_name == 'nirspec_ifu_4_point_nod') : 
            dither_ra  = np.array([-0.9077, 0.7980, 0.6437, -0.7521]) / 3600.0
            dither_dec = np.array([-0.7635, 0.8718, -0.8357, 0.9441]) / 3600.0
    
        elif (self.pattern_name == 'nirspec_ifu_4_point_dither') : 
            dither_ra  = np.array([-0.2328, 0.1292, 0.0259, -0.0776]) / 3600.0
            dither_dec = np.array([-0.0774, 0.1855, -0.1333, 0.2415]) / 3600.0
        
        elif (self.pattern_name == 'nirspec_ifu_cycle_small') : 
            data       = np.genfromtxt(cycle_file)
            dither_ra  = data[:, 1] / 3600.0
            dither_dec = data[:, 2] / 3600.0

        elif (self.pattern_name == 'nirspec_ifu_cycle_medium') : 
            data       = np.genfromtxt(cycle_file)
            dither_ra  = data[:, 3] / 3600.0
            dither_dec = data[:, 4] / 3600.0

        elif (self.pattern_name == 'nirspec_ifu_cycle_large') : 
            data       = np.genfromtxt(cycle_file)
            dither_ra  = data[:, 5] / 3600.0
            dither_dec = data[:, 6] / 3600.0
            
        elif (self.pattern_name == 'none') : 
            dither_ra  = [0]
            dither_dec = [0]
            
        return dither_ra, dither_dec

#    def read_cycle_pattern(filename = 'models/JWST_NIRSpec_cycling.txt') : 


    def set_mosaic(self, nrows = 1, ncols = 1, row_overlap = 0.1, col_overlap = 0.1, 
            row_shift = 0.0, col_shift = 0.0) : 


        self.mosaic             = True
        self.mosaic_nrows       = nrows
        self.mosaic_ncols       = ncols
        self.mosaic_row_overlap = row_overlap
        self.mosiac_col_overlap = col_overlap
        self.mosaic_row_shift   = row_shift
        self.mosaic_col_shift   = col_shift

    def get_mosaic(self, nrow, ncol, geometry = 'corners') : 
        #print(nrow, ncol)
        fov_ra, fov_dec = self.instrument_fov(geometry = geometry)
        
        # Distance to the centre of of one mosaic tile
        dist_ra  = (fov_ra[2] - fov_ra[1]) * (1 - self.mosaic_row_overlap) 
        dist_dec = (fov_dec[1] - fov_dec[0]) * (1 - self.mosiac_col_overlap) 
        
#        dist_ra *= 1 + np.sin(np.deg2rad(self.mosaic_row_shift)) 
#        dist_dec *= 1 + np.sin(np.deg2rad(self.mosaic_col_shift)) 
        
#        print(row_shift, col_shift)
        
        row_shift = np.sin(np.deg2rad(self.mosaic_row_shift)) * dist_ra * (ncol - self.mosaic_nrows/2)
        col_shift = np.sin(np.deg2rad(self.mosaic_col_shift)) * dist_dec * (nrow - self.mosaic_nrows/2)
        
        
        shift_ra = (dist_ra) * (nrow + 0.5 - self.mosaic_nrows/2) + row_shift
        shift_dec = (dist_dec) * (ncol + 0.5 - self.mosaic_ncols/2) + col_shift
 
        #shift_ra += np.sin(np.deg2rad(self.mosaic_row_shift)) * dist_ra
 
        
        off_ra = fov_ra +  shift_ra
        off_dec = fov_dec + shift_dec

        off_ra, off_dec = self.rotate(off_ra, off_dec, 0, 0, self.rotation)

        # Store the mosaic pointing 
        self.mosaic_ra  = self.pointing_ra + off_ra
        self.mosaic_dec = self.pointing_dec + off_dec

        mosaic_ra  = (self.pointing_ra + off_ra - self.ra_target) * 3600
        mosaic_dec = (self.pointing_dec + off_dec - self.dec_target) * 3600
        
        return mosaic_ra, mosaic_dec

    def get_dither(self, nbr, output = 'darcsec', geometry = 'corners') : 

        # Get the instrument FOV and dither patterh
        fov_ra, fov_dec         = self.instrument_fov(geometry = geometry)
        pattern_ra, pattern_dec = self.dither_pattern()
#        print(fov_ra, pattern_ra)
#        print(self.pointing_ra, self.pointing_dec)
#        print(self.ra_target, self.dec_target)
        #print(pattern_ra)

        off_ra = fov_ra +  pattern_ra[nbr]
        off_dec = fov_dec + pattern_dec[nbr]

        off_ra, off_dec = self.rotate(off_ra, off_dec, 0, 0, self.rotation)

        #if (self.mosaic == True) : 
        pointing_ra = self.pointing_ra
        pointing_dec = self.pointing_dec
        #else : 
        #    pointing_ra = self.mosaic_ra
        #    pointing_dec = self.mosaic_dec


        if (output == 'darcsec') :
            
            dither_ra  = (pointing_ra + off_ra - self.ra_target) * 3600
            dither_dec = (pointing_dec + off_dec - self.dec_target) * 3600
    
#            dither_ra  = (self.pointing_ra + fov_ra + pattern_ra[nbr] - self.ra_target) * 3600
#            dither_dec = (self.pointing_dec + fov_dec + pattern_dec[nbr] - self.dec_target) * 3600
    
        if (output == 'radec') : 
            dither_ra  = pointing_ra + off_ra
            dither_dec = pointing_dec + off_dec
    
    
        # Rotate the FOV
    #    dra = self.pointing_ra - self.ra_target
    #    ddec = self.pointing_dec - self.dec_target
    #    dither_ra, dither_dec = self.rotate(dither_ra, dither_dec, dra, ddec, self.rotation)
    
#            dither_corners_ra = (ifu_ra + dither_ra[j] - plan.ra_target ) * 3600
#        dither_corners_dec = (ifu_dec + dither_dec[j] - plan.dec_target ) * 3600
    
        return np.array(dither_ra), np.array(dither_dec)
    
    def target_level_1(self) : 
        self.pointing_ra, self.pointing_dec = self.ra_target, self.dec_target
        return self.ra_target, self.dec_target
    
    def target_radius_lat(self, lat) : 

        a = self.radii[0]
        b = self.radii[2]
        
        theta = np.deg2rad(lat)
        r = a * b / (np.sqrt(a**2 * np.sin(theta)**2 + b**2 * np.cos(theta)**2))

        return r
            
    def target_level_2_lon_lat(self, lon, lat) : 

        r = self.target_radius_lat(lat)

        torus_longitude = (360 - lon) % 360
        rectan = spice.latrec(r, np.deg2rad(torus_longitude), np.deg2rad(lat))
        
        
        p2 = np.matmul(rectan, self.i2p)
        vector_target = spice.vadd(self.pos_target, p2)
        range, ra, dec = spice.recrad(vector_target)
        
#        vector_target2 = spice.vadd(p2, self.pos_target)
        
#        params = self.pixel_params(0, 0, vector_target = vector_target)
#        print(params)

        
        self.pointing_ra, self.pointing_dec = np.rad2deg(ra), np.rad2deg(dec)
        
        return self.pointing_ra, self.pointing_dec
        
        
    def target_level_2_torus(self, radius, lon, lat) : 
        east_longitude_centre = 360.0 - self.centre['lon']
        torus_longitude = (east_longitude_centre + lon) % 360
        rectan = spice.latrec(radius, np.deg2rad(torus_longitude), np.deg2rad(lat))
        p2 = np.matmul(rectan, self.i2p)
        vector_target = spice.vadd(self.pos_target, p2)
            
        
        range, ra, dec = spice.recrad(vector_target)
        
        self.pointing_ra, self.pointing_dec = np.rad2deg(ra), np.rad2deg(dec)
        
        return np.rad2deg(ra), np.rad2deg(dec)        
    
    def project_lat_lon_array(self, lons, lats) :

        ras = np.zeros(lons.shape[0])
        decs = np.zeros(lons.shape[0])
        for i, lon in enumerate(lons) : 
            ras[i], decs[i] = self.lon_lat_to_dra_ddec(lons[i], lats[i])
        return ras[ras != np.nan], decs[decs != np.nan]

        
    def plot_lon_lat_contour(self, ax, fov, line_spacing = 10, labels = False) :
    
    
        cs1 = ax.contour(fov[17], fov[18], fov[0], levels=np.arange(-90, 90, line_spacing), colors= 'black', linestyles='dotted', origin = 'lower', linewidths = 0.5)
        if labels : 
            ax.clabel(cs1, cs1.levels)
        lons = fov[1]
        #print('lons', np.nanmax(lons), np.nanmin(lons))
        if (np.nanmax(lons) - np.nanmin(lons) > 356) :
            lons[lons > 180] -= 360
            #print(self.centre['lon'])
        ax.contour(fov[17], fov[18], lons, levels=np.arange(-500, 500, line_spacing), linestyles = 'dotted', colors= 'black', linewidths = 0.5)
        ax.contour(fov[17], fov[18], fov[10], levels=[0], colors= 'black', linewidths = 1.2)
#        CS = ax.contour(fov[17], fov[18], fov[10], levels=[5000, 10000], linestyles='--', colors= 'orange', linewidths = 0.7)
#        fmt = '%i km'
        #ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=6)

        #ax.contour(fov[17], fov[18], fov[9], levels=[6, 18], linestyles='--', colors= 'grey', linewidths = 0.7)

        ax.set(aspect = 'equal')
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set(xlabel = '$\Delta$ Right Ascension (arcsec)', ylabel = '$\Delta$ Declination (arcsec)')\

        return ax    
    
    def lon_lat_to_dra_ddec(self, lon, lat) : 
        ra, dec = self.target_level_2_lon_lat(lon, lat)
        params = self.pixel_params(ra, dec)
        ra = (ra - self.ra_target) * 3600
        dec = (dec - self.dec_target) * 3600
#        print(params['emission'])
#        if (params['emission'] != np.nan) : 
 #           if (np.nanmin(params['emission']) > 90) : 
        #east_longitude_centre = 360.0 - self.centre['lon']
        #torus_longitude = (east_longitude_centre + lon) % 360 
 
        londiff = (lon - self.centre['lon'])
        #print('Lon diff ', lon, londiff, np.abs(londiff))
        if (np.abs(londiff) > 90 ) : 
                ra, dec = np.nan, np.nan
        return ra, dec
 
 
    
    
    
    
def load_kernels(kdir = '/Users/hpm5/Documents/Data/kernels/') : 
    '''
        Load the kernels required to get the JWST pointing for the giant planets. 
    '''
    # Load the JWST and Jupiter kernels
    spice.furnsh(kdir + 'naif0012.tls')
    spice.furnsh(kdir + 'pck00010.tpc') 
    spice.furnsh(kdir + 'de430.bsp')
    spice.furnsh(kdir + 'jup310.bsp')
    spice.furnsh(kdir + 'sat452.bsp')
    spice.furnsh(kdir + 'ura115.bsp')
    spice.furnsh(kdir + 'nep102.bsp')
    spice.furnsh(kdir + 'jwst_pred.bsp')
    spice.furnsh(kdir + 'jwst_rec.bsp')
    


