#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 18:52:37 2023

@author: Gian
"""
import os

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
import time
from astropy.io import fits
import Fits_reader as fr



start = time.time()

# create badpixelmap (array with 0 if pixel can be used and 1 if not)
from badpixelmapping import badpixelmapping
badpixelmap = badpixelmapping()


print("done Badpixel", time.time() - start)

# create masterdark and masterflat
from masterdark import masterdark
from masterflatnormed import masterflatnormed
mstrdark_4s = masterdark(0,badpixelmap)
mstrdark_10s = masterdark(1,badpixelmap)

mstrflatn = masterflatnormed(3, 1, badpixelmap)

print("done Masterflat", time.time() - start)

# dark subtraction, flat division and badpixel removal
from badpixelinterpolation import badpixelinterpolation, multiplebadinterpol
from darkflatsubtraction import darkflatsubtraction

good_TVLyn_4s = multiplebadinterpol(4, badpixelmap)
good_TVLyn_10s = multiplebadinterpol(5, badpixelmap)

####### badpixel to be fixed

science_TVLyn_4s = darkflatsubtraction(good_TVLyn_4s,mstrdark_4s,mstrflatn)
science_TVLyn_10s = darkflatsubtraction(good_TVLyn_10s,mstrdark_10s,mstrflatn)

import gc
del good_TVLyn_4s
del good_TVLyn_10s
gc.collect()

print("done badpixelinterpolation", time.time() - start)

# cosmic ray rejection
# from cosmicrayrejection import cosmicrayrejection
# science_TVLyn_4s_nocosm = cosmicrayrejection(science_TVLyn_4s,badpixelmap)
# science_TVLyn_10s_nocosm = cosmicrayrejection(science_TVLyn_10s,badpixelmap)
# variable sigclip can be added (default 4.5), lower values will flag more pixels as cosmic rays

# background subtraction
from background_subtraction import subtract_background
science_TVLyn_4s_nobkg = subtract_background(science_TVLyn_4s)
science_TVLyn_10s_nobkg = subtract_background(science_TVLyn_10s)

del science_TVLyn_4s
del science_TVLyn_10s
gc.collect()

print("done background subtraction", time.time() - start)

headers_4 = fr.read_headers(4)
headers_10 = fr.read_headers(5)

for d4, h4 in zip(science_TVLyn_4s_nobkg, headers_4):
    fr.writer(d4, os.path.join("01 - TV Lyn", "4s", "Cleaned"), h4)

for d10, h10 in zip(science_TVLyn_10s_nobkg, headers_10):
    fr.writer(d10, os.path.join("01 - TV Lyn", "10s", "Cleaned"), h10)

from Finalizers.aligner import align_write
align_write("01 - TV Lyn/4s")
align_write("01 - TV Lyn/10s")

from plotter import plot
testplot = fr.processed_reader(1)[0]
plot(testplot,"TV Lyn 4s exposure - processed",vmax=70)
testplot2 = fr.processed_reader(3)[0]
plot(testplot2,"TV Lyn 10s exposure - processed",vmax=70)
from Finalizers.star_pos_finder import star_position_finder
print(star_position_finder(testplot,200))


