#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 23:07:21 2023

@author: Gian
"""

import numpy as np
from badpixelinterpolation import badpixelinterpolation
from badpixelmapping import badpixelmapping

def masterflatnormed(flat_data, dark_data, badpixelmap):
    ''' outputs the normed masterflat given raw flat and dark data.
    also includes badpixel removal '''
    badpixelmap = badpixelmapping(flat_data,dark_data)
    masterflatbad = np.median(flat_data-dark_data, axis=0)
    masterflat = badpixelinterpolation(masterflatbad,badpixelmap)
    flatavg = np.sum(masterflat/masterflat.size)
    masterflatnorm = masterflat/flatavg
    return masterflatnorm