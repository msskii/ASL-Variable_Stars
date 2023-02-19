#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 15:10:22 2023

@author: Gian
"""

import numpy as np
from skimage import draw

def peak_center(data,corner,search_radius=20,aperture_increase = 2,threshold=50):
    c = corner
    rx, ry = draw.circle_perimeter(c[0], c[1], radius=search_radius, shape=data.shape)
    sumsave = 0
    sumnext = 0
    xsav,ysav = rx[0],ry[0]
    for xo,yo in zip(rx,ry):
        mx, my = draw.disk((xo, yo), search_radius+aperture_increase)
        sumnext = np.sum(data[mx,my])
        if(sumnext>=sumsave):
            xsav,ysav = xo,yo
        sumsave = sumnext
    return sumnext


