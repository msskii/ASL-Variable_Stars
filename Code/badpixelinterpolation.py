#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:12:34 2022

@author: Gian
"""
import numpy as np
import warnings
import time

x0 = [-1,-1,-1,0,0,1,1,1]
y0 = [-1,0,1,-1,1,-1,0,1]


def badpixelinterpolation(data, badpixelmap):
    start = time.time()
    Bx,By = np.where(badpixelmap)
    xbound = badpixelmap[:,0].size
    ybound = badpixelmap[0,:].size

    for x,y in zip(Bx,By):
        pixels = 0
        value = 0
        for dx,dy in zip(x0,y0):
            if(x+dx >= 0 and x+dx <= xbound and y+dy >= 0 and y+dy <= ybound):
                if(not badpixelmap[x+dx,y+dy]):
                    pixels += 1
                    value += data[x+dx,y+dy]
        if(pixels == 0):
            warnings.warn("Too many bad badpixels")
            pixels = 1
        data[x,y] = (value / pixels)
    print(time.time() - start)
    return  data

def multiplebadinterpol(data_index, badpixelmap):
    data = fr.reader(data_index)
    ret = np.zeros(data[4].shape)
    for i in np.arange(data[:,0,0].size):
        ret[i] = badpixelinterpolation(data[i],badpixelmap)
    return ret



def badpixelinterpolationOld(data,badpixelmap):
    '''given a frame (data) containing badpixels (badpixelmap),
    the function outputs the same frame with the badpixels excluded and
    replaced with data interpolated from the closest surrounding non-bad pixels.'''
    x,y = np.where(badpixelmap)
    new = data.copy()
    xbound = badpixelmap[0,:].size
    ybound = badpixelmap[:,0].size
    for i in np.arange(x.size):
        n = int(x[i])
        m = int(y[i])
        foundinterpol = False
        layer = 1
        count = 0
        savex = np.array([])
        savey = np.array([])
        while not foundinterpol:
            for j in np.arange(n-layer,n+1+layer):
                if j==n-layer or j==n+layer:
                    for k in np.arange(m-layer,m+1+layer):
                        if j<xbound and k<ybound and j>=0 and k>=0 and not badpixelmap[j,k]:
                            count = count+1
                            savex = np.append(savex,np.array([j]))
                            savey = np.append(savey,np.array([k]))
                else:
                    for k in np.array([int(m-layer),int(m+layer)]):
                        if j<xbound and k<ybound and j>=0 and k>=0 and not badpixelmap[j,k]:
                            count = count+1
                            savex = np.append(savex,np.array([j]))
                            savey = np.append(savey,np.array([k]))
            layer = layer + 1
            if not count < 2:
                    foundinterpol = True
        interpolations = np.array([])
        for l in np.arange(savex.size):
            interpolations = np.append(interpolations,data[int(savex[l]),int(savey[l])])
        new[n,m] = np.sum(interpolations)/interpolations.size
    return new

# function test:
# badpixeldata = np.array([[1,2,3],[3,400,2],[50000,1,2]])
# badpixelmask = np.array([[False,False,False],[False,True,False],[True,False,False]])

# print(badpixelinterpolation(badpixeldata,badpixelmask))
