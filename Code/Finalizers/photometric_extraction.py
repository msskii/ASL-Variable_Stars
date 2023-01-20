#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 13:35:01 2023

@author: Gian
"""

from photutils.aperture import CircularAperture
from photutils.aperture import aperture_photometry
import numpy as np


def photometric_extraction(data, pos, posref_list, ref_mag_list):
    ''' from the data it extracts the flux from the target star
    positioned at pos and compares it to a list of reference stars
    which are at posref_list in the data 
    data = matrix with photon counts
    pos = tuple with x,y coordinates of star
    posref_list = list of tuples with x,y coordinates
    ref_mag_list = list of magnitudes of ref stars according to posref_list
    '''
    aperturestar = CircularAperture(pos, r=3.0)
    apertureref_list = [CircularAperture(posref_list[i], r=3.0) for i in np.arange(len(posref_list))]
    StarTab = aperture_photometry(data,aperturestar)
    RefTab = [aperture_photometry(data,apertureref_list[i]) for i in np.arange(len(posref_list))]
    
    # simplest algorithm for magnitude extraction:¨¨
    # mag of target = sum_ref(mag_ref * flux_targ/flux_ref)
    flux_targ = StarTab[0][3]
    flux_ref = [RefTab[i][3] for i in np.arange(len(RefTab))]
    return(sum(ref_mag_list/flux_ref * flux_targ))
