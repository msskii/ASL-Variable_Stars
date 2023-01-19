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
from astropy.io import fits
from enum import Enum

class Fits(Enum):
    Darks_4s = 0
    Darks_10s = 1
    Flats_0dot5s = 2
    Flats_10s = 3
    TV_Lyn = 4
    W_Uma = 5

paths = ["01 - Darks/4 Seconds", "01 - Darks/10 Seconds", "02 - Flats/5 Seconds", "02 - Flats/10 Seconds", "03 - Measurements/01 - TV Lyn", "03 - Measurements/02 - W U"]

script_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(script_path, os.pardir, 'data')


data = np.zeros(len(Fits), dtype=object)
head = np.zeros(len(Fits),dtype=object)

for i in range(len(Fits)):
    FITpath = os.path.join(data_path, paths[i])
    print("-------------------")
    print(FITpath)
    filenames = os.listdir(FITpath)
    filenames.sort()
    fits_data = np.zeros((len(filenames),3500,4500))
    fits_head = np.zeros(len(filenames),dtype=object)
    print(filenames)
    for j in np.arange(len(filenames)):
       fits_data[j] = fits.getdata(data_path + filenames[i],ext=0)
       fits_head[j] = fits.getheader(data_path + filenames[i],ext=0)
    data[i] = fits_data
    head[i] = fits_head

# data[i] is list of Dark, Flat, or Science Frames according to Enum index

#temporary variables; to be removed later
#in the next three sections the function input names have to be changed accordingly
dark_data = 0
flat_data = 0
dat = 0
####
# create badpixelmap (array with 0 if pixel can be used and 1 if not)
from badpixelmapping import badpixelmapping
badpixelmap0_5s = 0#badpixelmapping(flat_data,dark_data) #add this if dark 0.5s have been done
badpixelmap10s = badpixelmapping(data[3],data[1])
badpixelmap = np.invert(np.invert(badpixelmap0_5s) * np.invert(badpixelmap10s))

# create masterdark and masterflat
from masterdark import masterdark
from masterflatnormed import masterflatnormed
mstrdark = masterdark(dark_data)
mstrflatn = masterflatnormed(flat_data,dark_data)

# dark subtraction, flat division and badpixel removal
from badpixelinterpolation import badpixelinterpolation
from darkflatsubtraction import darkflatsubtraction
good_data = badpixelinterpolation(dat,badpixelmap)
science_data = darkflatsubtraction(good_data,mstrdark,mstrflatn)
