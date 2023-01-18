#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 18:52:37 2023

@author: Gian
"""
import numpy as np
# Steps in data reduction:
#   import .fit data
#   make master dark and master flat
#   make bad pixel map
#   remove bad pixels from frames
#   dark subtraction and flat division
#   (cosmic ray rejection)
#   background subtraction
#   astrometric calibration
#   photometric calibration

##########################################

# import .fit data
import os
from astropy.io import fits

data_path = '/Users/Gian/Documents/ASL Telescope/SFR_experiment_Laxhuber_Klasovita_copy/'
processed_path = '/Users/Gian/Documents/ASL Telescope/SFR_processed/'
filenames = os.listdir(data_path)
filenames.sort()

data = np.zeros((len(filenames),4096,4096)) # img size has to be edited
head = np.zeros(len(filenames),dtype=object)
for i in np.arange(len(filenames)):
    data[i] = fits.getdata(data_path + filenames[i],ext=0)
    head[i] = fits.getheader(data_path + filenames[i],ext=0)


