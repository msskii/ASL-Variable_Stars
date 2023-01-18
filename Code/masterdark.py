#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 22:39:02 2023

@author: Gian
"""
import numpy as np
from badpixelinterpolation import badpixelinterpolation

def masterdark(data):
    '''given a numpy array of dark frames the masterdark is returned'''
    mstdark = np.median(data, axis=0)
    return mstdark