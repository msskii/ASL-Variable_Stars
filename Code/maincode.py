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
from enum import Enum

class Fits(Enum):
    Darks_4s = 0
    Darks_10s = 1
    Flats_0dot5s = 2
    Flats_10s = 3
    TV_Lyn_4s = 4
    TV_Lyn_10s = 5
    W_Uma = 6
# we take 10s flats and ignore the .5s ones

paths = ["01 - Darks/4 Seconds", "01 - Darks/10 Seconds", "02 - Flats/5 Seconds", "02 - Flats/10 Seconds", "03 - Measurements/01 - TV Lyn/4s", "03 - Measurements/01 - TV Lyn/10s", "03 - Measurements/02 - W Uma"]

script_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(script_path, os.pardir, 'data')


data = np.zeros(len(Fits), dtype=object)
head = np.zeros(len(Fits),dtype=object)

start = time.time()

for i in range(len(Fits)):
    FITpath = os.path.join(data_path, paths[i])
    #print("-------------------")
    #print(FITpath)
    filenames = os.listdir(FITpath)
    filenames.sort()
    fits_data = np.zeros((len(filenames),3600,4500))
    fits_head = np.zeros(len(filenames),dtype=object)
    #print(filenames)
    for j in np.arange(len(filenames)):
       fits_data[j] = fits.getdata(os.path.join(FITpath, filenames[i]),ext=0)
       fits_head[j] = fits.getheader(os.path.join(FITpath, filenames[i]),ext=0)
    data[i] = fits_data
    head[i] = fits_head


print(time.time() - start)

# create badpixelmap (array with 0 if pixel can be used and 1 if not)
from badpixelmapping import badpixelmapping
badpixelmap = badpixelmapping(data[3],data[1])


print("done Badpixel", time.time() - start)

# create masterdark and masterflat
from masterdark import masterdark
from masterflatnormed import masterflatnormed
mstrdark_4s = masterdark(data[0],badpixelmap)
mstrdark_10s = masterdark(data[1],badpixelmap)
mstrflatn = masterflatnormed(data[3],data[1])

print("done Masterflat", time.time() - start)

# dark subtraction, flat division and badpixel removal
from badpixelinterpolation import badpixelinterpolation
from darkflatsubtraction import darkflatsubtraction
good_TVLyn_4s = np.zeros(data[4].shape)
good_TVLyn_10s = np.zeros(data[5].shape)
for i in np.arange(data[4][:,0,0].size):
    good_TVLyn_4s[i] = badpixelinterpolation(data[4][i],badpixelmap)
for i in np.arange(data[5][:,0,0].size):
    good_TVLyn_10s[i] = badpixelinterpolation(data[5][i],badpixelmap)

science_TVLyn_4s = darkflatsubtraction(good_TVLyn_4s,mstrdark_4s,mstrflatn)
science_TVLyn_10s = darkflatsubtraction(good_TVLyn_10s,mstrdark_10s,mstrflatn)

print("done badpixelinterpolation", time.time() - start)
