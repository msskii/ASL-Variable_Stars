#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:35:04 2023

@author: Gian
"""

import numpy as np
from peakcenter import peak_center_zonein

def photometric_extraction(data, corners, cornersref_list, ref_mag_list):
    '''extracts the flux from a data array for a star with 
    corner coordinates "corner" and reference stars at "cornerref_list"
    each with magnitude "ref_mag_list"'''
    # ma - mb = -2.5 log(Na/Nb)
    # ma = mb -2.5 log(Na/Nb)
    starsum = peak_center_zonein(data,corners)
    #print("wtf?",ref_mag_list)
    refstarsum = np.zeros_like(ref_mag_list)
    #print(refstarsum)
    for i in np.arange(ref_mag_list.size):
        #print("refzonein", cornersref_list)
        refstarsum[i] = peak_center_zonein(data,cornersref_list[i])
    mstar = np.zeros_like(ref_mag_list)
    for i in np.arange(ref_mag_list.size):
        mstar[i] = ref_mag_list[i] - 2.5 * np.log10(starsum/refstarsum[i])
    return np.sum(mstar)/mstar.size

def photometric_extraction_it(data_list,corners_list,cornersref_list_list,ref_mag_list):
    '''data_list is a list of data matrices, corners_list = starcoor and cornersref_list_list = [star1coor,sta2coor,...]
    where starcoor = [coorleft,coorright,coorup,coordown] and coordirec = [coordir for data mat i, for all i].
    The function performs photometric extraction for a set of data matrices, outputs the average of the list of magnitudes'''
    N = data_list[:,0,0].size # nr of data matrices
    magnitude_list = np.zeros(N)
    for j in np.arange(N):
        print("extraction step: ", j)
        print(cornersref_list_list[:,:,j][0])
        magnitude_list[j] = photometric_extraction(data_list[j], corners_list[:,j], cornersref_list_list[:,:,j], ref_mag_list)
    return magnitude_list
    
    
    
    