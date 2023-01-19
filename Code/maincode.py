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

# create badpixelmap (array with 0 if pixel can be used and 1 if not)
from badpixelmapping import badpixelmapping
badpixelmap = badpixelmapping()


print("done Badpixel", time.time() - start)

# create masterdark and masterflat
from masterdark import masterdark
from masterflatnormed import masterflatnormed
mstrdark_4s = masterdark(0,badpixelmap)
mstrdark_4s = masterdark(1,badpixelmap)

mstrflatn = masterflatnormed(data[3],data[1])

print("done Masterflat", time.time() - start)

# dark subtraction, flat division and badpixel removal
from badpixelinterpolation import badpixelinterpolation
from darkflatsubtraction import darkflatsubtraction
# good_TVLyn_4s = np.zeros(data[4].shape)
# good_TVLyn_10s = np.zeros(data[5].shape)
# for i in np.arange(data[4][:,0,0].size):
#     good_TVLyn_4s[i] = badpixelinterpolation(data[4][i],badpixelmap)
# for i in np.arange(data[5][:,0,0].size):
#     good_TVLyn_10s[i] = badpixelinterpolation(data[5][i],badpixelmap)
####### badpixel to be fixed
### change TVLyn data names to good_TVLyn after bad pixel has been done in the darkflatsubtraction section (replace data 4 and data 5!

science_TVLyn_4s = darkflatsubtraction(data[4],mstrdark_4s,mstrflatn)
science_TVLyn_10s = darkflatsubtraction(data[5],mstrdark_10s,mstrflatn)
#### edit above lines

print("done badpixelinterpolation", time.time() - start)

# cosmic ray rejection
from cosmicrayrejection import cosmicrayrejection
science_TVLyn_4s_nocosm = cosmicrayrejection(science_TVLyn_4s,badpixelmap)
science_TVLyn_10s_nocosm = cosmicrayrejection(science_TVLyn_10s,badpixelmap)
#variable sigclip can be added (default 4.5), lower values will flag more pixels as cosmic rays
