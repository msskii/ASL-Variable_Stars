#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 17:26:09 2023

@author: Gian
"""

from astroscrappy import detect_cosmics
import numpy as np

def cosmicrayrejection(indata, badpixelmap,sigclip=4.5):
    '''cite https://github.com/astropy/astroscrappy for detect_cosmics fn'''
    ''' the fn takes an array of data and returns an array of cosmicray 
    corrected data '''
    outdata = indata.copy()
    for i in np.arange(indata[:,0,0].size):
        outdata[i] = detect_cosmics(indata[i],badpixelmap,sigclip=sigclip)[1]
    return outdata
