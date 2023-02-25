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

# raw_data = fr.read_headers(4)
# headers = fr.processed_read_headers(0)
# print(raw_data)
data = fr.processed_reader(-1)
# data = 86561*raw_data
# data1 = 86561*data[0]
# data2 = 86561*data[1]
# data3 = 86561*data[50]

# dataproblem = 86561*data[33]
#peaks = np.zeros(23,dtype=tuple)
# peaks = pk.peak_finder_it(data,2800,3450)

# WUma right
# peaks[:10] = pk.peak_finder_it(data[:10],2800,3640)
# peaks[10:] = pk.peak_finder_it(data[10:],2660,3735)

# WUma up -> take this as radius check!!
# peaks[:8] = pk.peak_finder_it(data[:8],2880,3490)
# peaks[8:] = pk.peak_finder_it(data[8:],2840,3650)

# WUma left
# peaks[:10] = pk.peak_finder_it(data[:10],2800,3470)
# peaks[10:] = pk.peak_finder_it(data[10:],2750,3500)

# WUma down
# peaks[:8] = pk.peak_finder_it(data[:8],2765,3490)
# peaks[8:] = pk.peak_finder_it(data[8:],2590,3650)



# WUmaref right
# peaks[:11] = pk.peak_finder_it(data[:11],760,1035)
# peaks[11:] = pk.peak_finder_it(data[11:],630,1110)

# WUmaref up -> take this as radius check!!
# peaks[:11] = pk.peak_finder_it(data[:11],800,950)
# peaks[11:] = pk.peak_finder_it(data[11:],730,1050)

# WUmaef left
# peaks[:11] = pk.peak_finder_it(data[:11],760,900)
peaks1 = pk.peak_finder(data[-1],630,1000,50)

# WUmaref down
# peaks[:11] = pk.peak_finder_it(data[:11],720,950)
peaks2 = pk.peak_finder(data[-1],550,1100,50)



#coortopleft_down = pk.peak_finder_it(data,2079,2060)
# for i in [0 to 8]: (2700,3450) yields bottom
# (2784, 3493), (2788, 3494), (2781, 3494), (2775, 3500),
#        (2780, 3498), (2768, 3508), (2780, 3520), (2770, 3534),
#        (2770, 3544)
# -> ideal startx starty for first 9 entries for down would be: (2765,3490)
# for i in [rest]: "" this yields left
# (2767, 3556), (2760, 3562), (2746, 3574),
# (2730, 3578), (2720, 3585), (2708, 3593), (2699, 3601),
# (2700, 3608), (2690, 3614), (2690, 3622), (2679, 3632),
# (2665, 3638), (2669, 3645), (2659, 3652)
# -> ideal startx starty for entries after 9 for left would be: (2700,3550)

#-> ideal start for after 9 bottom: (2590,3650)


# start = time.time()
# peaks = pk.peak_finder_it(np.array([dataproblem]),2079,2060)
# print(time.time()-start,peaks)
# peaks = find_peaks(data1,0.001,npeaks=4)
# peaks_x = np.zeros(23)
# peaks_y = np.zeros(23)    #  2110,2060
# for i in np.arange(23):
#     peaks_x[i] = peaks[i][0]
#     peaks_y[i] = peaks[i][1]
#     plot.plot(data[i],str(i),vmax=70,show=False)
#     plt.scatter(peaks[i][1],peaks[i][0],s=1,c='red')
#     plt.show()
# plt.scatter(1000,2000)
peaks1_x = peaks1[0]
peaks1_y = peaks1[1]
peaks2_x = peaks2[0]
peaks2_y = peaks2[1]
plot.plot(data[-1],"title",vmax=70,show=False)
plt.scatter(peaks1[1],peaks1[0],s=1,c='red')
plt.scatter(peaks2[1],peaks2[0],s=1,c='red')
plt.show()
print(peaks1,peaks2)