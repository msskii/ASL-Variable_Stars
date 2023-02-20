#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 15:10:22 2023

@author: Gian
"""

import numpy as np
from skimage import draw
from threepointcircle import threepointcircledefinition

def peak_center_hammerthrow(data,corner,search_radius=20,aperture_increase = 2,threshold=50):
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

def peak_center_zonein(data,corners,borderthickness = 1):
    '''corners is a list of 4 corners that touch the star. 
    Given a data frame with a star surrounded by corners,
    the function overlays a circular aperture with additional
    border space around the star and sums the photon count from the star.'''
    cornerlist = corners[:-1]
    cornercheck = corners[-1]
    Mx,My,r = threepointcircledefinition(cornerlist)
    R = np.sqrt((cornercheck[0]-Mx)**2 + (cornercheck[1]-My)**2)
    Rr = np.max(np.array([r,R])) + borderthickness
    print("Overshoot: ", Rr-r)
    rx,ry = draw.disk(int(Mx+0.5),int(My+0.5),int(Rr+1),shape=data.shape)
    Totalcount = np.sum(data[rx,ry])
    return Totalcount
    
    
    
