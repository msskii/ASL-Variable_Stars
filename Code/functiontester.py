#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 00:04:24 2023

@author: Gian
"""
import numpy as np
#data_path = 
#data = fits.getdata(data_path + "",ext=0)
from masterdark import masterdark
data = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[10,20,30],[40,50,60],[70,80,90]],[[100,200,300],[400,500,600],[700,800,900]]])
badpixelmap = np.zeros((3,3))
print(masterdark(data,badpixelmap))