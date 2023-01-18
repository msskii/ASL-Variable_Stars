from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground

def subtract_background(data):
    """Subtracts background noise from data frame"""
    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    return data - bkg.background
