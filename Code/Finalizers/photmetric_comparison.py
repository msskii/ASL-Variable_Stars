#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:35:04 2023

@author: Gian
"""

import numpy as np
from peakcenter import peak_center_zonein

def photometric_extraction(data, corners, cornersref_list, ref_mag_list,borderthickness=1):
    '''extracts the flux from a data array for a star with 
    corner coordinates "corner" and reference stars at "cornerref_list"
    each with magnitude "ref_mag_list"'''
    # ma - mb = -2.5 log(Na/Nb)
    # ma = mb -2.5 log(Na/Nb)
    starsum = peak_center_zonein(data,corners,borderthickness=borderthickness)
    #print("wtf?",ref_mag_list)
    refstarsum = np.zeros_like(ref_mag_list)
    #print(refstarsum)
    for i in np.arange(ref_mag_list.size):
        #print("refzonein", cornersref_list)
        refstarsum[i] = peak_center_zonein(data,cornersref_list[i],borderthickness=borderthickness)
    mstar = np.zeros_like(ref_mag_list)
    for i in np.arange(ref_mag_list.size):
        mstar[i] = ref_mag_list[i] - 2.5 * np.log10(starsum/refstarsum[i])
    sqsum = 0
    mstaravg = np.sum(mstar)/mstar.size
    for i in np.arange(ref_mag_list.size):
        sqsum = sqsum + (mstar[i]-mstaravg)**2
    stddev = np.sqrt(sqsum/ref_mag_list.size)
    return mstaravg,mstar,stddev

def photometric_extraction_it(data_list,corners_list,cornersref_list_list,ref_mag_list,borderthickness=1):
    '''data_list is a list of data matrices, corners_list = starcoor and cornersref_list_list = [star1coor,sta2coor,...]
    where starcoor = [coorleft,coorright,coorup,coordown] and coordirec = [coordir for data mat i, for all i].
    The function performs photometric extraction for a set of data matrices, outputs the average of the list of magnitudes'''
    N = data_list[:,0,0].size # nr of data matrices
    magnitude_list = np.zeros(N)
    stddev_list = np.zeros(N)
    mag_list_3star = np.zeros((N,3))
    for j in np.arange(N):
        #print("extraction step: ", j)
        #print(cornersref_list_list[:,:,j][0])
        magnitude_list[j],mag_list_3star[j,:],stddev_list[j] = photometric_extraction(data_list[j], corners_list[:,j], cornersref_list_list[:,:,j], ref_mag_list,borderthickness=borderthickness)
    return magnitude_list,mag_list_3star,stddev_list
    
    
    
    