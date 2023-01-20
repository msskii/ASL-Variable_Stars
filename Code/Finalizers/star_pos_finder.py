#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:15:02 2023

@author: Gian
"""

from photutils.detection import find_peaks
import numpy as np

def star_position_finder(data,threshold):
    return find_peaks(data,threshold)