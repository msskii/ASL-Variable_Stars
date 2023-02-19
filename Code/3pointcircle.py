#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 18:30:16 2023

@author: Gian
"""

import numpy as np

def 3pointcircle(3points):
    '''finds radius and center coordinates of a circle
    given three non-collinear points (x,y-tuples of 
    points given in 3points) on the circle'''
    (x1,y1)-(x0,y0) -> xstep,ystep
    y = ax+b -> a = ystep/xstep -> b = y0-ax0
    if 3points
    p1 = 3points[0]
    p2 = 3points[1]
    p3 = 3points[2]
    secant1 = lambda x: ((p2[1]-p1[1])/(p2[0]-p1[0]))*(x-p1[0]) + p1[1]
    secant2 = lambda x: ((p3[1]-p1[1])/(p3[0]-p1[0]))*(x-p1[0]) + p1[1]
    