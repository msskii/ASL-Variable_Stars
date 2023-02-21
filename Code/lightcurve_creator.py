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
#headers_10 = fr.processed_read_headers(2)
headers_10 = fr.read_headers(5)
headers_10 = np.delete(headers_10,np.array([1,20,21,36,37,38,54,58,61]))

# reference star initial estimate of positions (x,y)
# star bl :  (1428.5,1643) # HIP 36761
mag_bl = 10.48 #[0.04]
# TV Lyn :  (2536,1500)
# star tl :  (1857,2500) # TYC 3409-2187-1
mag_tl = 11.45 #[0.09]
# star tr :  (3214,2886) # TYC 3409-2474-1
mag_tr = 11.38 #[0.08]

refmaglist = np.array([mag_tl,mag_tr,mag_bl])

from grunggers_image_processing import peak_finder_it
# coortopleft = peak_finder_it(TVLyn_data,2000,2000)
coortopleft_down = peak_finder_it(TVLyn_data,2079,2060)
#print("tld done")
coortopleft_up = peak_finder_it(TVLyn_data,2110,2060)
#print("tlu done")
coortopleft_left = peak_finder_it(TVLyn_data,2096,2030)
coortopleft_right = peak_finder_it(TVLyn_data,2091,2074)
coortopleft = np.array([coortopleft_left,coortopleft_right,coortopleft_up,coortopleft_down])

# coortopright = peak_finder_it(TVLyn_data,2500,3500)
coortopright_right = peak_finder_it(TVLyn_data,2517,3450)
coortopright_left = peak_finder_it(TVLyn_data,2519,3390)
coortopright_down = peak_finder_it(TVLyn_data,2490,3429)
coortopright_up = peak_finder_it(TVLyn_data,2540,3420)
coortopright = np.array([coortopright_left,coortopright_right,coortopright_up,coortopright_down])

# coorbotleft = peak_finder_it(TVLyn_data,1500,1500)
coorbotleft_left = peak_finder_it(TVLyn_data,1298,1550)
coorbotleft_right = peak_finder_it(TVLyn_data,1283,1610)
coorbotleft_up = peak_finder_it(TVLyn_data,1310,1577)
coorbotleft_down = peak_finder_it(TVLyn_data,1250,1582)
coorbotleft = np.array([coorbotleft_left,coorbotleft_right,coorbotleft_up,coorbotleft_down])

# coorTVLyn = peak_finder_it(TVLyn_data,1200,2800)
coorTVLyn_left = peak_finder_it(TVLyn_data,1128,2680)
coorTVLyn_right = peak_finder_it(TVLyn_data,1130,2730)
coorTVLyn_up = peak_finder_it(TVLyn_data,1140,2705)
coorTVLyn_down = peak_finder_it(TVLyn_data,1090,2713)
coorTVLyn = np.array([coorTVLyn_left,coorTVLyn_right,coorTVLyn_up,coorTVLyn_down])

refstarcoor = np.array([coortopleft,coortopright,coorbotleft])
# photometric calibration
from Finalizers.photmetric_comparison import photometric_extraction_it, photometric_extraction
m_TVLyn_list = photometric_extraction_it(TVLyn_data,coorTVLyn,refstarcoor,refmaglist)


# from Finalizers.photometric_extraction import photometric_extraction
# TVLynmag_list = np.zeros(coorTVLyn[:,0].size)
# for i in np.arange(coorTVLyn[:,0].size):
#     TVLynmag_list[i] = photometric_extraction(TVLyn_data[i],coorTVLyn[i],[coortopleft[i],coortopright[i],coorbotleft[i]],np.array([mag_tl,mag_tr,mag_bl]))


# light curve plot
from time_extractor import time_extract
import matplotlib.pyplot as plt
time_list = time_extract(headers_10)
plt.scatter(time_list,m_TVLyn_list)

# import matplotlib.pyplot as plt
# plot.plot(TVLyn_data[-2],"it works!")
# xy = np.zeros((coortopleft.size, 2))
# for i in np.arange(coortopleft.size):
#     xy[i] = coortopleft[i]
# x = xy[:,0]
# y = xy[:,1]
# plt.scatter(y,x)