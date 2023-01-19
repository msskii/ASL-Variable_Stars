#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 17:26:09 2023

@author: Gian
"""

from astroscrappy import detect_cosmics

def cosmicrayrejection(indata, badpixelmap):
    '''cite https://github.com/astropy/astroscrappy for detect_cosmics fn'''
    return detect_cosmics(indata,badpixelmap)
