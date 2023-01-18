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
#   (cosmic ray rejection)
#   background subtraction
#   astrometric calibration
#   photometric calibration

##########################################

# import .fit data
import os
from astropy.io import fits

script_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(script_path, os.pardir, 'data')

filenames = os.listdir(data_path)
filenames.sort()

print(filenames)


# total path is userspecific+data path
filenames = os.listdir(data_path)
filenames.sort()

data = np.zeros((len(filenames),4096,4096)) # img size has to be edited
head = np.zeros(len(filenames),dtype=object)
for i in np.arange(len(filenames)):
    data[i] = fits.getdata(data_path + filenames[i],ext=0)
    head[i] = fits.getheader(data_path + filenames[i],ext=0)
