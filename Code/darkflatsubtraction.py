#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 23:18:07 2023

@author: Gian
"""

import numpy as np
import Fits_reader as fr

def darkflatsubtraction(data_index, masterdark,masterflatnorm):
    ''' takes a raw data frame subtracts dark and divides by normed flat'''
    data = fr.reader(data_index)
    reduceddata = (data-masterdark)/masterflatnorm
    return reduceddata