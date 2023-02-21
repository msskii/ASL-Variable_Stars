#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 18:30:16 2023

@author: Gian
"""

import numpy as np
import matplotlib.pyplot as plt

def threepointcircledefinition(points):
    '''finds radius and center coordinates of a circle
    given three non-collinear points (x,y-tuples of 
    points given in 3points) on the circle'''
    Px = points[0][0]
    Py = points[0][1]
    Qx = points[1][0]
    Qy = points[1][1]
    Rx = points[2][0]
    Ry = points[2][1]
    xc = (Py*(Rx**2+Ry**2-Qx**2-Qy**2) + Ry*(Qx**2+Qy**2-Px**2-Py**2) + Qy*(Px**2+Py**2-Rx**2-Ry**2))/(2*(Qy*(Px-Rx)+Py*(Rx-Qx)+Ry*(Qx-Px)))
    yc = (Px**2+Py**2-Qx**2-Qy**2)/(2*(Py-Qy))-(Qx-Px)/(Qy-Py)*xc
    r = np.sqrt((Px-xc)**2 + (Py-yc)**2)
    return xc,yc,r

# #test
# P = (1,2)
# Q = (2,3)
# R = (10,33)
# points = np.array([P,Q,R])
# Mx,My,r = threepointcircledefinition(points)
# plt.scatter(P[0],P[1])
# plt.scatter(Q[0],Q[1])
# plt.scatter(R[0],R[1])
# # (y-My)^2 + (x-Mx)^2 = r^2 -> (y-My) = pm sqrt(r^2-(x-Mx)^2)
# circleyp = lambda Mx,My,r,x: My + np.sqrt(r**2 - (x-Mx)**2)
# circleym = lambda Mx,My,r,x: My - np.sqrt(r**2 - (x-Mx)**2)
# plt.scatter(Mx,My)
# x = np.linspace(-r,r,100)
# plt.scatter(x,circleyp(Mx,My,r,x),s=1)
# plt.scatter(x,circleym(Mx,My,r,x),s=1)
# plt.show()
