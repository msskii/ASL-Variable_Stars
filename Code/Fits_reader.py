import numpy as np
from astropy.io import fits
from enum import Enum
import os

class Fits(Enum):
    Darks_4s = 0
    Darks_10s = 1
    Flats_0dot5s = 2
    Flats_10s = 3
    TV_Lyn_4s = 4
    TV_Lyn_10s = 5
    W_Uma = 6
    Special = 7
# we take 10s flats and ignore the .5s ones

paths = [os.path.join("01 - Darks", "4 Seconds"),os.path.join("01 - Darks", "10 Seconds"), os.path.join("02 - Flats", "5 Seconds"), os.path.join("02 - Flats", "10 Seconds"), os.path.join("03 - Measurements","01 - TV Lyn","4s"), os.path.join("03 - Measurements","01 - TV Lyn","10s"), os.path.join("03 - Measurements","02 - W Uma"),"XX - Special"]
processed_paths = [os.path.join("01 - TV Lyn", "4s", "Aligned"), os.path.join("01 - TV Lyn", "4s","Cleaned"), os.path.join("01 - TV Lyn", "10s", "Aligned"), os.path.join("01 - TV Lyn", "10s","Cleaned"),os.path.join("02 - W Uma", "Aligned"), os.path.join("02 - W Uma", "Averaged"),os.path.join("02 - W Uma", "Cleaned")]

script_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(script_path, os.pardir, 'data')
data_write = os.path.join(script_path, os.pardir, 'data', '04 - Edited Images')

def listdir_nohidden(path):
    files = list()
    for f in os.listdir(path):
        if not f.startswith('.'):
            files.append(f)
    return files

def reader(i):
    FITpath = os.path.join(data_path, paths[i])
    filenames = listdir_nohidden(FITpath)
    filenames.sort()
    #print(filenames)
    fits_data = np.zeros((len(filenames),3600,4500), dtype=np.uint16)
    for j in np.arange(len(filenames)):
       fits_data[j] = fits.getdata(os.path.join(FITpath, filenames[j]),ext=0)
    return fits_data

def read_headers(i):
    FITpath = os.path.join(data_path, paths[i])
    filenames = listdir_nohidden(FITpath)
    filenames.sort()
    fits_head = np.zeros(len(filenames),dtype=object)
    for j in np.arange(len(filenames)):
        fits_head[j] = transform_timestamp(fits.getheader(os.path.join(FITpath, filenames[j]), ext=0)["DATE-OBS"])
    return fits_head

def processed_read_headers(i):
    FITpath = os.path.join(data_write, processed_paths[i])
    filenames = listdir_nohidden(FITpath)
    filenames.sort()
    fits_head = np.zeros(len(filenames),dtype=object)
    for j in np.arange(len(filenames)):
        #fits_head[j] = transform_timestamp(fits.getheader(os.path.join(FITpath, filenames[j]), ext=0)["DATE"])
        fits_head[j] = fits.getheader(os.path.join(FITpath, filenames[j]), ext=0)
    return fits_head


def transform_timestamp(raw):
    return raw.replace("T", "_")\
            .replace(":", "_").replace(".000", "")

def writer(data, path, date):
    write_to = os.path.join(data_write, path)
    fits.writeto(os.path.join(write_to, "mes_" + date + ".FITS"), data, overwrite=True)

def processed_reader(i):
    FITpath = os.path.join(data_write, processed_paths[i])
    filenames = listdir_nohidden(FITpath)
    filenames.sort()
    #print(filenames)
    fits_data = np.zeros((len(filenames),3600,4500))
    for j in np.arange(len(filenames)):
       fits_data[j] = fits.getdata(os.path.join(FITpath, filenames[j]),ext=0)
    return fits_data
