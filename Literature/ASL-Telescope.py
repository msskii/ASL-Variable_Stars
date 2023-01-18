#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 13:11:25 2022

@author: Gian
"""
import os
from astropy.io import fits
import numpy as np
from astroscrappy import detect_cosmics


def badpixelinterpolation(data,badpixelmap):
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




data_path = '/Users/Gian/Documents/ASL Telescope/SFR_experiment_Laxhuber_Klasovita_copy/'
processed_path = '/Users/Gian/Documents/ASL Telescope/SFR_processed/'
filenames = os.listdir(data_path)
filenames.sort()
filenames.remove('.DS_Store')

# all files are matrices of size 4096x4096
data = np.zeros((len(filenames),4096,4096))
head = np.zeros(len(filenames),dtype=object)
for i in np.arange(len(filenames)):
    data[i] = fits.getdata(data_path + filenames[i],ext=0)
    head[i] = fits.getheader(data_path + filenames[i],ext=0)

dark0_5s = np.array([s for s in filenames if "dark" in s and "0_5s" in s],dtype=object)
dark0_5sdata = np.zeros((len(dark0_5s),4096,4096))
dark0_5shead = np.zeros(len(dark0_5s),dtype=object)
for i in np.arange(dark0_5s.size):
    dark0_5sdata[i] = data[filenames.index(dark0_5s[i])]
    dark0_5shead[i] = head[filenames.index(dark0_5s[i])]

dark_3s = np.array([s for s in filenames if "dark" in s and "3s" in s],dtype=object)
dark_3sdata = np.zeros((len(dark_3s),4096,4096))
dark_3shead = np.zeros(len(dark_3s),dtype=object)
for i in np.arange(dark_3s.size):
    dark_3sdata[i] = data[filenames.index(dark_3s[i])]
    dark_3shead[i] = head[filenames.index(dark_3s[i])]

dark_120s = np.array([s for s in filenames if "dark" in s and "120s" in s],dtype=object)
dark_120sdata = np.zeros((len(dark_120s),4096,4096))
dark_120shead = np.zeros(len(dark_120s),dtype=object)
for i in np.arange(dark_120s.size):
    dark_120sdata[i] = data[filenames.index(dark_120s[i])]
    dark_120shead[i] = head[filenames.index(dark_120s[i])]

darkflat = np.array([s for s in filenames if "Dark" in s and "Flat" in s],dtype=object)
darkflatdata = np.zeros((len(darkflat),4096,4096))
darkflathead = np.zeros(len(darkflat),dtype=object)
for i in np.arange(darkflat.size):
    darkflatdata[i] = data[filenames.index(darkflat[i])]
    darkflathead[i] = head[filenames.index(darkflat[i])]

flatHa1 = np.array([s for s in filenames if "Flat" in s and "Ha" in s],dtype=object)
flatHa1data = np.zeros((len(flatHa1),4096,4096))
flatHa1head = np.zeros(len(flatHa1),dtype=object)
for i in np.arange(flatHa1.size):
    flatHa1data[i] = data[filenames.index(flatHa1[i])]
    flatHa1head[i] = head[filenames.index(flatHa1[i])]
    
flatHa2 = np.array([s for s in filenames if "Flat" in s and "Ha" in s and "Try2" in s],dtype=object)
flatHa2data = np.zeros((len(flatHa2),4096,4096))
flatHa2head = np.zeros(len(flatHa2),dtype=object)
for i in np.arange(flatHa2.size):
    flatHa2data[i] = data[filenames.index(flatHa2[i])]
    flatHa2head[i] = head[filenames.index(flatHa2[i])]

M101Ha = np.array([s for s in filenames if "M101" in s and "Try1" in s],dtype=object)
M101Hadata = np.zeros((len(M101Ha),4096,4096))
M101Hahead = np.zeros(len(M101Ha),dtype=object)
for i in np.arange(M101Ha.size):
    M101Hadata[i] = data[filenames.index(M101Ha[i])]
    M101Hahead[i] = head[filenames.index(M101Ha[i])]

M101Haref = np.array([s for s in filenames if "M101" in s and "Ref" in s],dtype=object)
M101Harefdata = np.zeros((len(M101Haref),4096,4096))
M101Harefhead = np.zeros(len(M101Haref),dtype=object)
for i in np.arange(M101Haref.size):
    M101Harefdata[i] = data[filenames.index(M101Haref[i])]
    M101Harefhead[i] = head[filenames.index(M101Haref[i])]

M33Ha = np.array([s for s in filenames if "M33" in s and "Try1" in s],dtype=object)
M33Hadata = np.zeros((len(M33Ha),4096,4096))
M33Hahead = np.zeros(len(M33Ha),dtype=object)
for i in np.arange(M33Ha.size):
    M33Hadata[i] = data[filenames.index(M33Ha[i])]
    M33Hahead[i] = head[filenames.index(M33Ha[i])]

M33Haref = np.array([s for s in filenames if "M33" in s and "Ref" in s],dtype=object)
M33Harefdata = np.zeros((len(M33Haref),4096,4096))
M33Harefhead = np.zeros(len(M33Haref),dtype=object)
for i in np.arange(M33Haref.size):
    M33Harefdata[i] = data[filenames.index(M33Haref[i])]
    M33Harefhead[i] = head[filenames.index(M33Haref[i])]

M74Ha = np.array([s for s in filenames if "M74" in s and "Try1" in s],dtype=object)
M74Hadata = np.zeros((len(M74Ha),4096,4096))
M74Hahead = np.zeros(len(M74Ha),dtype=object)
for i in np.arange(M74Ha.size):
    M74Hadata[i] = data[filenames.index(M74Ha[i])]
    M74Hahead[i] = head[filenames.index(M74Ha[i])]

M74Haref = np.array([s for s in filenames if "M74" in s and "Ref" in s],dtype=object)
M74Harefdata = np.zeros((len(M74Haref),4096,4096))
M74Harefhead = np.zeros(len(M74Haref),dtype=object)
for i in np.arange(M74Haref.size):
    M74Harefdata[i] = data[filenames.index(M74Haref[i])]
    M74Harefhead[i] = head[filenames.index(M74Haref[i])]

# for overview all filename groups listed:
#   dark0_5s  dark_3s  dark_120s  darkflat  
#   flatHa1  flatHa2  M101Ha  M33Ha  M74Ha
#   M101Haref  M33Haref  M74Haref

dark0_5smedbad = np.median(dark0_5sdata, axis=0)       # .5s
dark_3smedbad = np.median(dark_3sdata, axis=0)         # 3s
dark_120smedbad = np.median(dark_120sdata, axis=0)     # 120s
darkflatmedbad = np.median(darkflatdata, axis=0)       # 1.5s
#flatHa1med = np.median(flatHa1data, axis=0)         # 1.5s
#flatHa2med = np.median(flatHa2data, axis=0)         # 1.5s
M101Hamed = np.median(M101Hadata, axis=0)           # 120s
M33Hamed = np.median(M33Hadata, axis=0)             # 120s
M74Hamed = np.median(M74Hadata, axis=0)             # 120s
M101Harefmed = np.median(M101Harefdata, axis=0)     # 3s
M33Harefmed = np.median(M33Harefdata, axis=0)       # 3s
M74Harefmed = np.median(M74Harefdata, axis=0)       # .5s

masterflat1 = np.median(flatHa1data-darkflatmedbad, axis=0)
masterflat2 = np.median(flatHa2data-darkflatmedbad, axis=0)

flatsigma1 = np.sqrt(np.sum((masterflat1-np.sum(masterflat1)/masterflat1.size)**2)/masterflat1.size)

flatavg1 = np.sum(masterflat1/masterflat1.size)
flatavg2 = np.sum(masterflat2/masterflat2.size)
flatsig1 = np.sqrt(np.sum((masterflat1-np.sum(masterflat1)/masterflat1.size)**2)/masterflat1.size)
flatsig2 = np.sqrt(np.sum((masterflat2-np.sum(masterflat2)/masterflat2.size)**2)/masterflat2.size)

badpixel1mask = np.invert(np.invert(masterflat1 < flatavg1-5*flatsig1) * np.invert(masterflat1 > flatavg1 + 5*flatsig1))
badpixel2mask = np.invert(np.invert(masterflat2 < flatavg2-5*flatsig2) * np.invert(masterflat2 > flatavg2 + 5*flatsig2))
# combining the two bad pixel maps to eliminate all bad pixels:
badpixelmask = np.invert(np.invert(badpixel1mask) * np.invert(badpixel2mask))


masterflatnormbadpixel = (masterflat1/flatavg1 + masterflat2/flatavg2)/2

masterflatnorm = badpixelinterpolation(masterflatnormbadpixel,badpixelmask)

masterdark0_5s = badpixelinterpolation(dark0_5smedbad,badpixelmask)
masterdark_3s = badpixelinterpolation(dark_3smedbad,badpixelmask)
masterdark_120s = badpixelinterpolation(dark_120smedbad,badpixelmask)

# with the median images one can subtract flat and darks

# M101
M101sciencebad = np.median((M101Hadata-dark_120smedbad)/(masterflatnorm), axis=0)
M101science = badpixelinterpolation(M101sciencebad,badpixelmask)
# M101ref
M101refsciencebad = np.median((M101Harefdata-dark_3smedbad)/(masterflatnorm), axis=0)
M101refscience = badpixelinterpolation(M101refsciencebad,badpixelmask)
# M33
M33sciencebad = np.median((M33Hadata-dark_120smedbad)/(masterflatnorm), axis=0)
M33science = badpixelinterpolation(M33sciencebad,badpixelmask)
# M33ref
M33refsciencebad = np.median((M33Harefdata-dark_3smedbad)/(masterflatnorm), axis=0)
M33refscience = badpixelinterpolation(M33refsciencebad,badpixelmask)
# M74
M74sciencebad = np.median((M74Hadata-dark_120smedbad)/(masterflatnorm), axis=0)
M74science = badpixelinterpolation(M74sciencebad,badpixelmask)
# M74ref
M74refsciencebad = np.median((M74Harefdata-dark0_5smedbad)/(masterflatnorm), axis=0)
M74refscience = badpixelinterpolation(M74refsciencebad,badpixelmask)


#Unfinished cosmic ray detection and source detection:
M101mask, M101noray = detect_cosmics(M101science, inmask=None, inbkg=None, invar=None, sigclip=4.5,
                      sigfrac=0.3, objlim=5.0, gain=1.0, readnoise=6.5,
                      satlevel=65536.0, niter=4, sepmed=True,
                      cleantype='meanmask', fsmode='median', psfmodel='gauss',
                      psffwhm=2.5, psfsize=7, psfk=None, psfbeta=4.765,
                      verbose=False)

from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.segmentation import detect_threshold, detect_sources
from photutils.utils import circular_footprint
sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
threshold = detect_threshold(M101science, nsigma=2.0, sigma_clip=sigma_clip)
segment_img = detect_sources(M101science, threshold, npixels=10)
footprint = circular_footprint(radius=10)
mask = segment_img.make_source_mask(footprint=footprint)
mean, median, std = sigma_clipped_stats(M101science, sigma=3.0, mask=mask)
#print((mean, median, std)) 
 
#for i in np.arange(M74Hahead.size): print(M74Hahead[i]['EXPTIME'])

#fits.writeto(processed_path + 'test.fit', median_image, M33Hahead[0], overwrite=True)
