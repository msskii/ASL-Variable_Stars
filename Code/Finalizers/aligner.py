from astropy.io import fits
import astroalign as aa
import os


def align_write(target_measurement):
    """Takes cleaned target measurement and aligns all images to match."""

    source_dir = "../data/04 - Processed Data/" + target_measurement + "/Cleaned/"
    target_dir = "../data/04 - Processed Data/" + target_measurement + "/Aligned/"

    source = os.listdir(source_dir)
    master = fits.getdata(source_dir + source[0])
    for i, file in enumerate(source[1:]):
        transformed, _ = aa.register(master, fits.getdata(source_dir + file))
        fits.writeto(target_dir + "al_" + file, transformed)

    print("[+] Alignment completed.")
