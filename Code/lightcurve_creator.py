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

# light curve plot
from time_extractor import time_extract
import matplotlib.pyplot as plt
time_list = time_extract(headers_10)
plt.scatter(time_list,m_TVLyn_list)

from scipy import optimize
time_list_arr = np.zeros(time_list.size)
for i in np.arange(time_list.size):
    time_list_arr[i] = time_list[i]
def test_func(x, a, b, c):
    return a * np.sin(b * x) + c

# plt.scatter(time_list_arr/1000,np.ones_like(time_list_arr)) 
# plt.scatter(np.array([0.35,0.8,1.4,2,2.7,3.25,4]),np.ones(7),color='orange')
# plt.grid(which='both') 
# plt.show()


p1 = np.where(time_list_arr<350)
p2 = np.where((time_list_arr<800)&(time_list_arr>350))
p3 = np.where((time_list_arr<1400)&(time_list_arr>800))
p4 = np.where((time_list_arr<2000)&(time_list_arr>1400))
p5 = np.where((time_list_arr<2700)&(time_list_arr>2000))
p6 = np.where((time_list_arr<3250)&(time_list_arr>2700))
p7 = np.where((time_list_arr<4000)&(time_list_arr>3250))
p8 = np.where(time_list_arr>4000)

t1 = np.median(time_list_arr[p1])
Datap1 = m_TVLyn_list[p1]
t2 = np.median(time_list_arr[p2])
Datap2 = m_TVLyn_list[p2]
t3 = np.median(time_list_arr[p3])
Datap3 = m_TVLyn_list[p3]
t4 = np.median(time_list_arr[p4])
Datap4 = m_TVLyn_list[p4]
t5 = np.median(time_list_arr[p5])
Datap5 = m_TVLyn_list[p5]
t6 = np.median(time_list_arr[p6])
Datap6 = m_TVLyn_list[p6]
t7 = np.median(time_list_arr[p7])
Datap7 = m_TVLyn_list[p7]
t8 = np.median(time_list_arr[p8])
Datap8 = m_TVLyn_list[p8]

D1 = np.median(Datap1)
D2 = np.median(Datap2)
D3 = np.median(Datap3)
D4 = np.median(Datap4)
D5 = np.median(Datap5)
D6 = np.median(Datap6)
D7 = np.median(Datap7)
D8 = np.median(Datap8)

t = np.array([t1,t2,t3,t4,t5,t6,t7,t8])
D = np.array([D1,D2,D3,D4,D5,D6,D7,D8])
plt.scatter(t,D)

params, params_covariance = optimize.curve_fit(test_func, t, D,
                                                p0=[0.1, 2*np.pi/8000, 11.52],bounds=(np.array([0.8,1/10000,10]),np.array([1,6/5000,12])))

print(params)
plt.scatter(np.arange(4000,step=10),test_func(np.arange(4000,step=10),params[0],params[1],params[2]))
