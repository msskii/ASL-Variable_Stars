#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 22:39:02 2023

@author: Gian
"""
import numpy as np
from badpixelinterpolation import badpixelinterpolation

def masterdark(data,badpixelmap):
    '''given a numpy array of dark frames the masterdark is returned'''
    gooddata = badpixelinterpolation(data,badpixelmap)
    mstdark = np.median(gooddata, axis=0)
    return mstdark