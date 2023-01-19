from astropy.io import fits
import astroalign as aa
import os


def align_write(target_measurement):

    filter = lambda lst: [x for x in lst if x.startswith("mes") and x.count("_") == 3]

    source_dir = "../data/04 - Edited Images/" + target_measurement + "/Cleaned/"
    target_dir = "../data/04 - Edited Images/" + target_measurement + "/Aligned/"

    source = filter(os.listdir(source_dir))

    exposures = {x.split("s")[1].split("_")[1] + "s" for x in source}

    for id in exposures:
        print("[+] Aligning exposure time: " + id)
        files = [x for x in source if id in x]
        master = fits.getdata(source_dir + files[0])
        for i, file in enumerate(files[1:]):
            transformed, _ = aa.register(master, fits.getdata(source_dir + file))
            fits.writeto(target_dir + "al_" + file, transformed)

    print("[+] Alignment completed.")
