#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 18:52:37 2023

@author: Gian
"""
import numpy as np
# Steps in data reduction:
#   import .fit data (done)
#   make master dark and master flat (done)
#   make bad pixel map (done)
#   remove bad pixels from frames (done)
#   dark subtraction and flat division (done)
#   cosmic ray rejection
#   background subtraction
#   astrometric calibration
#   photometric calibration

##########################################

# import .fit data
import os
import time
from astropy.io import fits
import Fits_reader



start = time.time()

####
# create badpixelmap (array with 0 if pixel can be used and 1 if not)
from badpixelmapping import badpixelmapping
badpixelmap = badpixelmapping()


print("done Badpixel", time.time() - start)

# create masterdark and masterflat
from masterdark import masterdark
from masterflatnormed import masterflatnormed
mstrdark_4s = masterdark(0,badpixelmap)
mstrflatn = masterflatnormed(data[3],data[1])

print("done Masterflat", time.time() - start)

# dark subtraction, flat division and badpixel removal
from badpixelinterpolation import badpixelinterpolation
from darkflatsubtraction import darkflatsubtraction
good_data = badpixelinterpolation(dat,badpixelmap)
science_data = darkflatsubtraction(good_data,mstrdark,mstrflatn)

print("done badpixelinterpolation", time.time() - start)
