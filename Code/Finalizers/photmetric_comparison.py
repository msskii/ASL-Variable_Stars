#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:35:04 2023

@author: Gian
"""

import numpy as np
from peakcenter import peak_center

def photometric_extraction(data, corner, cornerref_list, ref_mag_list):
    '''extracts the flux from a data array for a star with 
    corner coordinates "corner" and reference stars at "cornerref_list"
    each with magnitude "ref_mag_list"'''
    
    starsum = peak_center(data,corner)
    refstarsum = np.zeros_like(ref_mag_list)
    for i in np.arange(ref_mag_list.size):
        refstarsum[i] = peak_center(data,cornerref_list)
    correctionfactor = np.ones_like(ref_mag_list)
    for i in np.arange(ref_mag_list.size):
        
    
    