from astropy.io import fits
import numpy as np
import os


def average_write(target_measurement):
    """Takes aligned images of target measurement and averages to yield one corrected picture."""

    filter = lambda lst: [x for x in lst if x.startswith("mes") and x.count("_") == 3]

    source_dir = "../data/04 - Edited Images/" + target_measurement + "/Aligned/"
    target_dir = "../data/04 - Edited Images/" + target_measurement + "/Averaged/"

    source = filter(os.listdir(source_dir))

    prefixes = {'_'.join(x.split("_")[:-1]) for x in source}

    for prefix in prefixes:

        print("[+] Averaging over measurement \"" + prefix + "\".")

        files = [x for x in source if x.startswith(prefix)]
        mat = fits.getdata(source_dir + files[0])
        for f in files[1:]:
            mat = np.add(mat, fits.getdata(source_dir + f))
        mat = np.divide(mat, len(files))
        fits.writeto(target_dir + "avg_" + prefix + ".FITS", mat)

    print("[+] Averaging completed.")
