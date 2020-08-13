# -*- coding: UTF-8 -*-
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

pf = fits.open('HFI_SkyMap_545_2048_R3.01_full.fits')
position = SkyCoord(8.4744770, -56.3420895, frame='galactic', unit='deg')  # GLON,deg     GLAT,deg
# position = SkyCoord(334.4436414, -35.7155681, frame='fk5', unit='deg', equinox='J2000.0') #  RA,deg       Dec,deg
size = (400, 400)
wcs = WCS(pf[0].header)
cutout = Cutout2D(pf[0].data, position, size, wcs=wcs)

pf[0].data = cutout.data
pf[0].header['CRPIX1'] = cutout.wcs.wcs.crpix[0]
pf[0].header['CRPIX2'] = cutout.wcs.wcs.crpix[1]
pf.writeto('PSZ2_G008.47-56.34_545_GHz.fits')
pf.close()
