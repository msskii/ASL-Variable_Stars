#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 14:07:50 2023

@author: Gian
"""
import numpy as np
import plotter as plot
import Fits_reader as fr
import time

TVLyn_data = 86561*fr.processed_reader(2)
WUma_data = fr.processed_reader(-1)
#headers_10 = fr.processed_read_headers(2)
headers_10 = fr.read_headers(5)
headers_10 = np.delete(headers_10,np.array([1,20,21,36,37,38,54,58,61]))
headers_wuma = fr.read_headers(6)

# reference star initial estimate of positions (x,y)
# star bl :  (1428.5,1643) # HIP 36761
mag_bl = 10.48 #[0.04]
# TV Lyn :  (2536,1500)
# star tl :  (1857,2500) # TYC 3409-2187-1
mag_tl = 11.45 #[0.09]
# star tr :  (3214,2886) # TYC 3409-2474-1
mag_tr = 11.38 #[0.08]

# ref star wuma : HD 83728
mag_wumaref = 8.87 # in visible wavelengths


refmaglist = np.array([mag_tl,mag_tr,mag_bl])
refmag_wuma = np.array([mag_wumaref])

from grunggers_image_processing import peak_finder_it

# WUma right
coorWuma_right = np.zeros(23,dtype=tuple)
coorWuma_right[:10] = peak_finder_it(WUma_data[:10],2800,3640)
coorWuma_right[10:] = peak_finder_it(WUma_data[10:],2660,3735)

# WUma up -> take this as radius check!!
coorWuma_up = np.zeros(23,dtype=tuple)
coorWuma_up[:8] = peak_finder_it(WUma_data[:8],2880,3490)
coorWuma_up[8:] = peak_finder_it(WUma_data[8:],2840,3650)

# WUma left
coorWuma_left = np.zeros(23,dtype=tuple)
coorWuma_left[:10] = peak_finder_it(WUma_data[:10],2800,3470)
coorWuma_left[10:] = peak_finder_it(WUma_data[10:],2750,3500)

# WUma down
coorWuma_down = np.zeros(23,dtype=tuple)
coorWuma_down[:8] = peak_finder_it(WUma_data[:8],2765,3490)
coorWuma_down[8:] = peak_finder_it(WUma_data[8:],2590,3650)

coorWuma = np.array([coorWuma_left,coorWuma_right,coorWuma_down,coorWuma_up])

# WUmaref right
coorWumaref_right = np.zeros(23,dtype=tuple)
coorWumaref_right[:11] = peak_finder_it(WUma_data[:11],760,1035)
coorWumaref_right[11:] = peak_finder_it(WUma_data[11:],630,1110)

# WUmaref up -> take this as radius check!!
coorWumaref_up = np.zeros(23,dtype=tuple)
coorWumaref_up[:11] = peak_finder_it(WUma_data[:11],800,950)
coorWumaref_up[11:] = peak_finder_it(WUma_data[11:],730,1050)

# WUmaef left
coorWumaref_left = np.zeros(23,dtype=tuple)
coorWumaref_left[:11] = peak_finder_it(WUma_data[:11],760,900)
coorWumaref_left[11:] = peak_finder_it(WUma_data[11:],630,1000)

# WUmaref down
coorWumaref_down = np.zeros(23,dtype=tuple)
coorWumaref_down[:11] = peak_finder_it(WUma_data[:11],720,950)
coorWumaref_down[11:-1] = peak_finder_it(WUma_data[11:-1],590,1050)
coorWumaref_down[-1] = peak_finder_it(np.array([WUma_data[-1]]),550,1100)[0]


coorWumaref = np.array([np.array([coorWumaref_left,coorWumaref_right,coorWumaref_down,coorWumaref_up])])

# coortopleft = peak_finder_it(TVLyn_data,2000,2000)
coortopleft_down = peak_finder_it(TVLyn_data,2079,2060)
#print("tld done")
coortopleft_up = peak_finder_it(TVLyn_data,2110,2060)
#print("tlu done")
coortopleft_left = peak_finder_it(TVLyn_data,2096,2030)
coortopleft_right = peak_finder_it(TVLyn_data,2091,2074)
coortopleft = np.array([coortopleft_left,coortopleft_right,coortopleft_up,coortopleft_down])

# coortopright = peak_finder_it(TVLyn_data,2500,3500)
coortopright_right = peak_finder_it(TVLyn_data,2517,3450)
coortopright_left = peak_finder_it(TVLyn_data,2519,3390)
coortopright_down = peak_finder_it(TVLyn_data,2490,3429)
coortopright_up = peak_finder_it(TVLyn_data,2540,3420)
coortopright = np.array([coortopright_left,coortopright_right,coortopright_up,coortopright_down])

# coorbotleft = peak_finder_it(TVLyn_data,1500,1500)
coorbotleft_left = peak_finder_it(TVLyn_data,1298,1550)
coorbotleft_right = peak_finder_it(TVLyn_data,1283,1610)
coorbotleft_up = peak_finder_it(TVLyn_data,1310,1577)
coorbotleft_down = peak_finder_it(TVLyn_data,1250,1582)
coorbotleft = np.array([coorbotleft_left,coorbotleft_right,coorbotleft_up,coorbotleft_down])

# coorTVLyn = peak_finder_it(TVLyn_data,1200,2800)
coorTVLyn_left = peak_finder_it(TVLyn_data,1128,2680)
coorTVLyn_right = peak_finder_it(TVLyn_data,1130,2730)
coorTVLyn_up = peak_finder_it(TVLyn_data,1140,2705)
coorTVLyn_down = peak_finder_it(TVLyn_data,1090,2713)
coorTVLyn = np.array([coorTVLyn_left,coorTVLyn_right,coorTVLyn_up,coorTVLyn_down])



refstarcoor = np.array([coortopleft,coortopright,coorbotleft])
# photometric calibration
from Finalizers.photmetric_comparison import photometric_extraction_it, photometric_extraction
m_TVLyn_list,stddev_TVLyn_list = photometric_extraction_it(TVLyn_data,coorTVLyn,refstarcoor,refmaglist)
m_WUma,stddev_WUma_list = photometric_extraction_it(WUma_data,coorWuma,coorWumaref,refmag_wuma)
print(stddev_TVLyn_list)

# light curve plot
from time_extractor import time_extract,time_extract_wuma
import matplotlib.pyplot as plt

time_list = time_extract(headers_10)

from scipy import optimize
time_list_arr = np.zeros(time_list.size)
for i in np.arange(time_list.size):
    time_list_arr[i] = time_list[i]
def test_func(x, a, b, c):
    return a * np.sin(b * x) + c

p1 = np.where(time_list_arr<350)
p2 = np.where((time_list_arr<800)&(time_list_arr>350))
p3 = np.where((time_list_arr<1400)&(time_list_arr>800))
p4 = np.where((time_list_arr<2000)&(time_list_arr>1400))
p5 = np.where((time_list_arr<2700)&(time_list_arr>2000))
p6 = np.where((time_list_arr<3250)&(time_list_arr>2700))
p7 = np.where((time_list_arr<4000)&(time_list_arr>3250))
p8 = np.where(time_list_arr>4000)

t1 = np.median(time_list_arr[p1])
Datap1 = m_TVLyn_list[p1]
t2 = np.median(time_list_arr[p2])
i2,j2 = time_list_arr[p2][4],time_list_arr[p2][5]
Datap2 = m_TVLyn_list[p2]
t3 = np.median(time_list_arr[p3])
i3 = time_list_arr[p3][4]
Datap3 = m_TVLyn_list[p3]
t4 = np.median(time_list_arr[p4])
i4,j4 = time_list_arr[p4][4],time_list_arr[p4][5]
Datap4 = m_TVLyn_list[p4]
t5 = np.median(time_list_arr[p5])
i5,j5 = time_list_arr[p5][3],time_list_arr[p5][4]
Datap5 = m_TVLyn_list[p5]
t6 = np.median(time_list_arr[p6])
i6 = time_list_arr[p6][3]
Datap6 = m_TVLyn_list[p6]
t7 = np.median(time_list_arr[p7])
i7,j7 = time_list_arr[p7][4],time_list_arr[p7][5]
Datap7 = m_TVLyn_list[p7]
t8 = np.median(time_list_arr[p8])
i8,j8 = time_list_arr[p8][3],time_list_arr[p8][4]
Datap8 = m_TVLyn_list[p8]

D1 = np.median(Datap1)
dataerr1 = stddev_TVLyn_list[p1][4]
D2 = np.median(Datap2)
dataerr2 = np.sqrt((stddev_TVLyn_list[p2][4])**2+(stddev_TVLyn_list[p2][5])**2)
D3 = np.median(Datap3)
dataerr3 = stddev_TVLyn_list[p3][4]
D4 = np.median(Datap4)
dataerr4 = np.sqrt((stddev_TVLyn_list[p4][4])**2+(stddev_TVLyn_list[p4][5])**2)
D5 = np.median(Datap5)
dataerr5 = np.sqrt((stddev_TVLyn_list[p5][3])**2+(stddev_TVLyn_list[p5][4])**2)
D6 = np.median(Datap6)
dataerr6 = stddev_TVLyn_list[p6][3]
D7 = np.median(Datap7)
dataerr7 = np.sqrt((stddev_TVLyn_list[p7][4])**2+(stddev_TVLyn_list[p7][5])**2)
D8 = np.median(Datap8)
dataerr8 = np.sqrt((stddev_TVLyn_list[p8][3])**2+(stddev_TVLyn_list[p8][4])**2)


wmean1 = np.sum(stddev_TVLyn_list[p1]*Datap1)/np.sum(stddev_TVLyn_list[p1])
wmean2 = np.sum(stddev_TVLyn_list[p2]*Datap2)/np.sum(stddev_TVLyn_list[p2])
wmean3 = np.sum(stddev_TVLyn_list[p3]*Datap3)/np.sum(stddev_TVLyn_list[p3])
wmean4 = np.sum(stddev_TVLyn_list[p4]*Datap4)/np.sum(stddev_TVLyn_list[p4])
wmean5 = np.sum(stddev_TVLyn_list[p5]*Datap5)/np.sum(stddev_TVLyn_list[p5])
wmean6 = np.sum(stddev_TVLyn_list[p6]*Datap6)/np.sum(stddev_TVLyn_list[p6])
wmean7 = np.sum(stddev_TVLyn_list[p7]*Datap7)/np.sum(stddev_TVLyn_list[p7])
wmean8 = np.sum(stddev_TVLyn_list[p8]*Datap8)/np.sum(stddev_TVLyn_list[p8])

std1 = np.std(Datap1)
std2 = np.std(Datap2)
std3 = np.std(Datap3)
std4 = np.std(Datap4)
std5 = np.std(Datap5)
std6 = np.std(Datap6)
std7 = np.std(Datap7)
std8 = np.std(Datap8)

wstd1 = std1 * np.sqrt(np.sum((stddev_TVLyn_list[p1]/np.sum(stddev_TVLyn_list[p1]))**2))
wstd2 = std2 * np.sqrt(np.sum((stddev_TVLyn_list[p2]/np.sum(stddev_TVLyn_list[p2]))**2))
wstd3 = std3 * np.sqrt(np.sum((stddev_TVLyn_list[p3]/np.sum(stddev_TVLyn_list[p3]))**2))
wstd4 = std4 * np.sqrt(np.sum((stddev_TVLyn_list[p4]/np.sum(stddev_TVLyn_list[p4]))**2))
wstd5 = std5 * np.sqrt(np.sum((stddev_TVLyn_list[p5]/np.sum(stddev_TVLyn_list[p5]))**2))
wstd6 = std6 * np.sqrt(np.sum((stddev_TVLyn_list[p6]/np.sum(stddev_TVLyn_list[p6]))**2))
wstd7 = std7 * np.sqrt(np.sum((stddev_TVLyn_list[p7]/np.sum(stddev_TVLyn_list[p7]))**2))
wstd8 = std8 * np.sqrt(np.sum((stddev_TVLyn_list[p8]/np.sum(stddev_TVLyn_list[p8]))**2))

medstd1 = np.sqrt((wstd1 * 1.2533)**2 + dataerr1**2)
medstd2 = np.sqrt((wstd2 * 1.2533)**2 + dataerr2**2)
medstd3 = np.sqrt((wstd3 * 1.2533)**2 + dataerr3**2)
medstd4 = np.sqrt((wstd4 * 1.2533)**2 + dataerr4**2)
medstd5 = np.sqrt((wstd5 * 1.2533)**2 + dataerr5**2)
medstd6 = np.sqrt((wstd6 * 1.2533)**2 + dataerr6**2)
medstd7 = np.sqrt((wstd7 * 1.2533)**2 + dataerr7**2)
medstd8 = np.sqrt((wstd8 * 1.2533)**2 + dataerr8**2)
medstd = np.array([medstd1,medstd2,medstd3,medstd4,medstd5,medstd6,medstd7,medstd8])

t = np.array([t1,t2,t3,t4,t5,t6,t7,t8])
D = np.array([D1,D2,D3,D4,D5,D6,D7,D8])

#params, params_covariance = optimize.curve_fit(test_func, t, D, p0=[0.1, 2*np.pi/8000, 11.52],bounds=(np.array([0.8,1/10000,10]),np.array([1,6/5000,12])))

def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])
    numerical_guess = np.array([0.042*1.5,2*np.pi*0.0041/2,-0.8,11.5])
    
    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    #popt, pcov = optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    popt, pcov = optimize.curve_fit(sinfunc, tt, yy, p0=guess)#,bounds=(np.array([0,1/10000,-1000,10]),np.array([0.2,0.01,1000,12])))
    A, w, p, c = popt
    f = w/(2.*np.pi)
    corr_w = 1/2
    corr_A = 1.5
    corr_p = 1.5
    fitfunc = lambda t: A*corr_A * np.sin(w*t*corr_w + p+corr_p) + c
    #fitfunc = lambda t: A* np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}


# skewed sine function:
def skewsinfunc(t, A, w, p, c, n):
    n = int(n)
    import scipy.special as sp
    S = np.ones(n,dtype=object)
    for k in np.arange(n):
        S[k] = (sp.binom(2*n,n-(k+1)))/(sp.binom(2*n,n)) * np.sin((k+1) * w * t + p)/(k+1)
        print(k,": ",(sp.binom(2*n,n-(k+1)))/(sp.binom(2*n,n))/(k+1),(k+1) * w, p)
    return A*np.sum(S,axis=0)+c

# res = fit_skewed_sin(t, D)
# print( "Amplitude=%(amp)s, Angular freq.=%(omega)s, phase=%(phase)s, offset=%(offset)s, n=%(n)s, Max. Cov.=%(maxcov)s" % res )

#TV Lyn plot
#   plot only data:
#plt.scatter(time_list/60/60,m_TVLyn_list,s=15,label="TVLyn Magnitudes")
plt.errorbar(time_list/60/60,m_TVLyn_list,yerr=stddev_TVLyn_list,capsize=2,ls='',elinewidth=0.5,ecolor="black",fmt='ob',ms=3,label="TVLyn Magnitudes")
#plt.scatter(t/60/60, D, s=30,c="red", label="Median data")
plt.title("TVLyn light curve")
plt.xlabel("Time t[hrs]")
plt.ylabel("Apparent Magnitude m[-]")
plt.grid()
plt.legend(loc="best")
plt.show()

#   plot data with medians:
# approximate error of median as error of weighted mean times 1.2533 according
# to: https://influentialpoints.com/Training/standard_error_of_median.htm
# the error of the median is given by the sqrt of the above value squared + the measurement uncertainty/stddeviation of the median element itself squared

#plt.scatter(time_list/60/60,m_TVLyn_list,s=15,label="TVLyn Magnitudes")
plt.errorbar(time_list/60/60,m_TVLyn_list,yerr=stddev_TVLyn_list,errorevery=2,capsize=2,ls='',elinewidth=0.5,fmt='ob',ms=2,label="TVLyn Magnitudes")
plt.errorbar(t/60/60, D, ms=3,fmt='or', yerr=medstd,capsize=2,ls='',elinewidth=0.5,label="Median data")
plt.title("TVLyn light curve")
plt.xlabel("Time t[hrs]")
plt.ylabel("Apparent Magnitude m[-]")
plt.grid()
plt.legend(loc="best")
plt.show()

#   sine fit:
time_list = time_extract(headers_10)
tt = np.linspace(0,4500,10000)
tth = tt/60/60
#plt.scatter(time_list/60/60,m_TVLyn_list,s=15,label="TVLyn Magnitudes")
plt.errorbar(time_list/60/60,m_TVLyn_list,yerr=stddev_TVLyn_list,errorevery=2,capsize=2,ls='',elinewidth=0.5,fmt='ob',ms=2,label="TVLyn Magnitudes")
plt.errorbar(t/60/60, D, ms=2,fmt='oc', yerr=medstd,capsize=2,ls='',elinewidth=0.5,label="Median data")
plt.plot(tth,0.09*np.sin(0.002*tt-0.7)+11.54,"g-", label="Sine curve fit")
plt.title("TV Lyn Light Curve")
plt.xlabel("Time t[hrs]")
plt.ylabel("Apparent Magnitude m[-]")
plt.grid()
plt.legend(loc="best")

sine_fit = lambda t: 0.09*np.sin(0.002*t-0.7)+11.54
fi = sine_fit(t)
yi = D
xi = t
n = t.size
Rsq_sinemodel = 1-np.sum((yi-fi)**2)/np.sum((yi-np.sum(yi)/n)**2) 
Rsq_sinemodel = round(Rsq_sinemodel,3)
plt.text(0.0,12.41,r'$R^2$ = '+str(Rsq_sinemodel))
plt.show()

#   skewed sine fit:
time_list = time_extract(headers_10)
tt = np.linspace(0,4500,10000)
tth = tt/60/60
#plt.scatter(time_list/60/60,m_TVLyn_list,s=15,label="TVLyn Magnitudes")
plt.errorbar(time_list/60/60,m_TVLyn_list,yerr=stddev_TVLyn_list,errorevery=2,capsize=2,ls='',elinewidth=0.5,fmt='ob',ms=2,label="TVLyn Magnitudes")
plt.errorbar(t/60/60, D, ms=2,fmt='oc', yerr=medstd,capsize=2,ls='',elinewidth=0.5,label="Median data")
plt.plot(tth, skewsinfunc(tt,0.11,0.002,-0.7,11.54,4), "r-", label="Skewed-sine curve fit", linewidth=2)
plt.title("TV Lyn Light Curve")
plt.xlabel("Time t[hrs]")
plt.ylabel("Apparent Magnitude m[-]")
plt.grid()
plt.legend(loc="best")

skewed_sine_fit = lambda t: skewsinfunc(t,0.11,0.002,-0.7,11.54,4)
fi = skewed_sine_fit(t)
yi = D
xi = t
n = t.size
Rsq_skewmodel = 1-np.sum((yi-fi)**2)/np.sum((yi-np.sum(yi)/n)**2) 
Rsq_skewmodel = round(Rsq_skewmodel,3)
plt.text(0.0,12.41,r'$R^2$ = '+str(Rsq_skewmodel))
plt.show()

#   both sine and skewed sine fit:
#   sine fit:
time_list = time_extract(headers_10)
tt = np.linspace(0,4500,10000)
tth = tt/60/60
#plt.scatter(time_list/60/60,m_TVLyn_list,s=15,label="TVLyn Magnitudes")
plt.errorbar(time_list/60/60,m_TVLyn_list,yerr=stddev_TVLyn_list,errorevery=2,capsize=2,ls='',elinewidth=0.5,fmt='ob',ms=2,label="TVLyn Magnitudes")
plt.errorbar(t/60/60, D, ms=2,fmt='oc', yerr=medstd,capsize=2,ls='',elinewidth=0.5,label="Median data")
plt.plot(tth,0.09*np.sin(0.002*tt-0.7)+11.54,"g-", label="Sine curve fit")
plt.plot(tth, skewsinfunc(tt,0.11,0.002,-0.7,11.54,4), "r-", label="Skewed-sine curve fit", linewidth=2)
plt.title("TV Lyn Light Curve")
plt.xlabel("Time t[hrs]")
plt.ylabel("Apparent Magnitude m[-]")
plt.grid()
plt.legend(loc="best")
plt.show()

# our fit function result is an angular frequency of 0.002 Hz which corresponds
# to a period of 3141.6 sec or 0.87 hrs


# WUma plot:
time_list_wuma = time_extract_wuma(headers_wuma)
plt.scatter(time_list_wuma/60/60,m_WUma)
plt.grid()
plt.title("W Uma Light Curve")
plt.xlabel("Time t[hrs]")
plt.ylabel("Apparent Magnitude m[-]")
plt.show()