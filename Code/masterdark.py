#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 22:39:02 2023

@author: Gian
"""
import numpy as np
from badpixelinterpolation import badpixelinterpolation
import Fits_reader as fr

def masterdark(data_index,badpixelmap):
    '''given a numpy array of dark frames the masterdark is returned'''
    data = fr.reader(data_index)
    gooddata = badpixelinterpolation(data,badpixelmap)
    mstdark = np.median(gooddata, axis=0)
    return mstdark
