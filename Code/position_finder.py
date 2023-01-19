# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 17:59:07 2023

@author: lucal
"""

import numpy as np
from photutils.segmentation import detect_threshold
from photutils.detection import find_peaks

def star_position (data, sigma):
    """
    data : array [[], []]
    
    sigma : array (one float for each picture)
        number of standard deviations per pixel above the background for which 
        to consider a pixel as possibly being part of a source.

    Returns
    Position of all stars (/peaks) on the images

    """
    thresholds = [detect_threshold(data[i], nsigma=sigma[i]) for i in sigma.size()]
    
    peaks = [find_peaks(data[i], thresholds[i]) for i in sigma.size()]
    
    return peaks

print (x)
