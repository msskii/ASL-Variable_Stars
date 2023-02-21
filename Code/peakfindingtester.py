#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 08:17:26 2023

@author: Gian
"""

from photutils.detection import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from astropy.io import fits
import Fits_reader as fr
import grunggers_image_processing as pk
import plotter as plot

#raw_data = fr.processed_reader(2)
headers = fr.processed_read_headers(2)
# data = 86561*raw_data
# data1 = 86561*data[0]
# data2 = 86561*data[1]
# data3 = 86561*data[50]

# dataproblem = 86561*data[33]


#coortopleft_down = pk.peak_finder_it(data,2079,2060)

# start = time.time()
# peaks = pk.peak_finder_it(np.array([dataproblem]),2079,2060)
# print(time.time()-start,peaks)
# peaks = find_peaks(data1,0.001,npeaks=4)
# peaks_x = np.zeros(4)
# peaks_y = np.zeros(4)    #  2110,2060
# for i in np.arange(4):
#     peaks_x[i] = peaks[i][0]
#     peaks_y[i] = peaks[i][1]
# plot.plot(data3,"title",vmax=70,show=False)
# plt.scatter(peaks[0][1],peaks[0][0],s=1,c='red')
# plt.scatter(1000,2000)