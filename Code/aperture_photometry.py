# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 23:18:39 2023

@author: lucal
"""

import numpy as np
from photutils.aperture import aperture_photometry, CircularAperture


def star_mag (data, position):
    """
    data : array [[], []]
        Background must be already substracted 
    position : array [(x1,y1) , (x2,y2)]
        x,y pixel position of the object (for each picture).

    Returns
    phot_table : array
        [0]: id: number of the position/picture (starts at 1)
        [1]: xcenter position for this id
        [2]: ycenter position for this id
        [3+i]: aperture sum i: sum of the data inside aperture with radii[i]

    """
    radii = [1.0, 2.0, 3.0, 4.0, 5.0]
    aperture = [CircularAperture(position, r=r) for r in radii]
    
    #without subpixels its slightly more accurate but much slower!
    #subpixels=n --> the higher n, the more accurate
    phot_table = aperture_photometry(data, aperture, method='subpixel', 
                                     subpixels=5)    
    #format:
    for col in phot_table.colnames:
        phot_table[col].info.format = '%.8g'
    
    return phot_table

def star_mag_idealradius (data, position):
    """
    data : array [[], []]
        Background must be already substracted 
    position : array [(x1,y1) , (x2,y2)]
        x,y pixel position of the object (for each picture).

    Returns
    phot_table : array
        [0]: id: number of the position/picture (starts at 1)
        [1]: xcenter position for this id
        [2]: ycenter position for this id
        [3]: aperture sum: sum of the data inside aperture with radius r
    r: float
        ideal radius r of the aperture == star radius
    """
    control = 1
    r, phot_table = 0.0
    
    while (control):
        temp = phot_table
        r += 1.0
        #aperture gets bigger by factor x --> aperture sum must get bigger by factor x/2
        factor = r**2/(r-1)**2
        limit = 1/2*factor  
        
        aperture = CircularAperture(position, r)
        
        #without subpixels its slightly more accurate but much slower!
        #subpixels=n --> the higher n, the more accurate
        phot_table = aperture_photometry(data, aperture, method='subpixel', 
                                         subpixels=5)    
        #format:
        phot_table['aperture_sum'].info.format = '%.8g'
        
        if (r > 1 and np.mean((phot_table[3] - temp[3]) / phot_table[3]) < limit):
            control = 0
            phot_table = temp
            r -= 1

    return phot_table, r
    

