#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 14:07:50 2023

@author: Gian
"""
import numpy as np
import plotter as plot
import Fits_reader as fr

TVLyn_data = 86561*fr.processed_reader(2)
headers_10 = fr.read_headers(5)

# reference star initial estimate of positions (x,y)
# star bl :  (1428.5,1643) # HIP 36761
mag_bl = 10.48 #[0.04]
# TV Lyn :  (2536,1500)
# star tl :  (1857,2500) # TYC 3409-2187-1
mag_tl = 11.45 #[0.09]
# star tr :  (3214,2886) # TYC 3409-2474-1
mag_tr = 11.38 #[0.08]

from grunggers_image_processing import peak_finder_it
coortopleft = peak_finder_it(TVLyn_data,2000,2000)
coortopright = peak_finder_it(TVLyn_data,2500,3500)
coorbotleft = peak_finder_it(TVLyn_data,1500,1500)
coorTVLyn = peak_finder_it(TVLyn_data,1200,2800)

# photometric calibration
from Finalizers.photometric_extraction import photometric_extraction
TVLynmag_list = np.zeros(coorTVLyn[:,0].size)
for i in np.arange(coorTVLyn[:,0].size):
    TVLynmag_list[i] = photometric_extraction(TVLyn_data[i],coorTVLyn[i],[coortopleft[i],coortopright[i],coorbotleft[i]],np.array([mag_tl,mag_tr,mag_bl]))


# light curve plot
from time_extractor import time_extract
import matplotlib.pyplot as plt
time_list = time_extract(headers_10)
plt.plot(time_list,TVLynmag_list)

import matplotlib.pyplot as plt
plot.plot(TVLyn_data[-2],"it works!")
xy = np.zeros((coortopleft.size, 2))
for i in np.arange(coortopleft.size):
    xy[i] = coortopleft[i]
x = xy[:,0]
y = xy[:,1]
plt.scatter(y,x)