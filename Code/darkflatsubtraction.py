#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 23:18:07 2023

@author: Gian
"""

import numpy as np

def darkflatsubtraction(data, masterdark,masterflatnorm):
    ''' takes a raw data frame subtracts dark and divides by normed flat'''
    reduceddata = (data-masterdark)/masterflatnorm
    return reduceddata