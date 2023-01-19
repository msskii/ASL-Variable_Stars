import numpy as np
from astropy.io import fits
from enum import Enum
import os

class Fits(Enum):
    Darks_4s = 0
    Darks_10s = 1
    Flats_0dot5s = 2
    Flats_10s = 3
    TV_Lyn = 4
    W_Uma = 5
    Special = 6
# we take 10s flats and ignore the .5s ones

paths = [os.path.join("01 - Darks", "4 Seconds"),os.path.join("01 - Darks", "10 Seconds"), os.path.join("02 - Flats", "5 Seconds"), os.path.join("02 - Flats", "10 Seconds"), os.path.join("03 - Measurements","01 - TV Lyn"), os.path.join("03 - Measurements","02 - W Uma"),"XX - Special"]

script_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(script_path, os.pardir, 'data')


def reader(i):
    FITpath = os.path.join(data_path, paths[i])
    filenames = os.listdir(FITpath)
    filenames.sort()
    fits_data = np.zeros((len(filenames),3600,4500))
    #fits_head = np.zeros(len(filenames),dtype=object)
    for j in np.arange(len(filenames)):
       fits_data[j] = fits.getdata(os.path.join(FITpath, filenames[i]),ext=0)
       #fits_head[j] = fits.getheader(os.path.join(FITpath, filenames[i]),ext=0)
    return fits_data
