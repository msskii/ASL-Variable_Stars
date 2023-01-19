from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
import numpy as np

def subtract_background(data):
    """Subtracts background noise from data frame"""
    ret = np.zeros(data.shape)
    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = MedianBackground()
    
    for i in np.arange(data[:,0,0].size):
        bkg = Background2D(data, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        ret[i] = data - bkg.background

    return ret
