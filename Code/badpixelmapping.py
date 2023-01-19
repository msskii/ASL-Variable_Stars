#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 22:51:14 2023

@author: Gian
"""
import numpy as np
import Fits_reader as fr

def flat():
    flat_data = fr.reader(1)
    dark_data = fr.reader(3)
    flat_data -= dark_data
    print("subdone")
    return np.median(flat_data, axis=0)

def badpixelmapping():
    masterflat = flat()
    print(masterflat.shape)
    flatavg = np.mean(masterflat)
    print(flatavg.shape)
    flatsig = np.sqrt(np.sum((masterflat-np.sum(masterflat)/masterflat.size)**2)/masterflat.size)
    badpixelmap = np.invert(np.invert(masterflat < flatavg-5*flatsig) * np.invert(masterflat > flatavg + 5*flatsig))
    return badpixelmap
