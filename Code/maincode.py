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

good_WUma = multiplebadinterpol(6, badpixelmap)

####### badpixel to be fixed

science_TVLyn_4s = darkflatsubtraction(good_TVLyn_4s,mstrdark_4s,mstrflatn)
science_TVLyn_10s = darkflatsubtraction(good_TVLyn_10s,mstrdark_10s,mstrflatn)

science_WUma = darkflatsubtraction(good_WUma, mstrdark_10s, mstrflatn)

import gc
del good_TVLyn_4s
del good_TVLyn_10s
del good_WUma
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

science_WUma_nobkg = subtract_background(science_WUma)

del science_TVLyn_4s
del science_TVLyn_10s
del science_WUma
gc.collect()

print("done background subtraction", time.time() - start)

headers_4 = fr.read_headers(4)
headers_10 = fr.read_headers(5)
headers_wuma = fr.read_headers(6)

for d4, h4 in zip(science_TVLyn_4s_nobkg, headers_4):
    fr.writer(d4, os.path.join("01 - TV Lyn", "4s", "Cleaned"), h4)

for d10, h10 in zip(science_TVLyn_10s_nobkg, headers_10):
    fr.writer(d10, os.path.join("01 - TV Lyn", "10s", "Cleaned"), h10)

for dw, hw in zip(science_WUma_nobkg, headers_wuma):
    fr.writer(dw, os.path.join("02 - W UMa", "Cleaned"), hw)

# from Finalizers.aligner import align_write
# align_write("01 - TV Lyn/4s")
# align_write("01 - TV Lyn/10s")

from plotter import plot
#testplot = fr.processed_reader(1)[0]
#plot(testplot,"TV Lyn 4s exposure - processed",vmax=70)
# testplot2 = fr.processed_reader(3)[0]
# plot(testplot2,"TV Lyn 10s exposure - processed",vmax=70)
# from Finalizers.star_pos_finder import star_position_finder
# stars = star_position_finder(testplot2,700)
# x = np.array([stars[i][0] for i in np.arange(len(stars))])
# y = np.array([stars[i][1] for i in np.arange(len(stars))])
# import matplotlib.pyplot as plt
# plt.scatter(x,y)



# ########
# # reference star initial estimate of positions (x,y)
# # star bl :  (1428.5,1643) # HIP 36761
# mag_bl = 10.48 #[0.04]
# # TV Lyn :  (2536,1500)
# # star tl :  (1857,2500) # TYC 3409-2187-1
# mag_tl = 11.45 #[0.09]
# # star tr :  (3214,2886) # TYC 3409-2474-1
# mag_tr = 11.38 #[0.08]

# # assuming star_finder(data_list, initial guess for 4 stars,threshold)
# # outputs four lists of (x,y) coordinates for each data_list data matrix and onelist per star
# from grunggers_image_processing import peak_finder_it
# coortopleft = peak_finder_it(science_TVLyn_10s_nobkg,2000,2000)
# coortopright = peak_finder_it(science_TVLyn_10s_nobkg,2500,3500)
# coorbotleft = peak_finder_it(science_TVLyn_10s_nobkg,1500,1500)
# coorTVLyn = peak_finder_it(science_TVLyn_10s_nobkg,1200,2800)

# # photometric calibration
# from Finalizers.photometric_extraction import photometric_extraction
# TVLynmag_list = np.zeros(coorTVLyn[:,0].size)
# for i in np.arange(coorTVLyn[:,0].size):
#     TVLynmag_list[i] = photometric_extraction(science_TVLyn_10s_nobkg[i],coorTVLyn[i],[coortopleft[i],coortopright[i],coorbotleft[i]],np.array([mag_tl,mag_tr,mag_bl]))

# # light curve plot
# from time_extractor import time_extract
# import matplotlib.pyplot as plt
# time_list = time_extract(headers_10)
# plt.plot(time_list,TVLynmag_list)


# import matplotlib.pyplot as plt
# plot(science_TVLyn_10s_nobkg[-2],"it works!")
# xy = np.zeros((coortopleft.size, 2))
# for i in np.arange(coortopleft.size):
#     xy[i] = coortopleft[i]
# x = xy[:,0]
# y = xy[:,1]
# plt.scatter(y,x)
