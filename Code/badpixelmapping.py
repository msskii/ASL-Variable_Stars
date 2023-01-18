#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 22:51:14 2023

@author: Gian
"""
import numpy as np

def badpixelmapping(flat_data, dark_data):
    '''takes flat data (and dark data) from ccd and identifies outliers
    (5 sigma) that should be excluded in the science frames'''
    masterflat = np.median(flat_data-dark_data, axis=0)
    flatavg = np.sum(masterflat/masterflat.size)
    flatsig = np.sqrt(np.sum((masterflat-np.sum(masterflat)/masterflat.size)**2)/masterflat.size)
    badpixelmap = np.invert(np.invert(masterflat < flatavg-5*flatsig) * np.invert(masterflat > flatavg + 5*flatsig))
    return badpixelmap


