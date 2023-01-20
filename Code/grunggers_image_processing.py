import numpy as np
from skimage import draw


def peak_finder(data, startx, starty, threshold, box_radius=5, abort=500):
    highestsum = -100
    for r in range(abort):
        rx, ry = draw.circle_perimeter(startx, starty, radius=r+1, shape=data.shape)
        for xo,yo in zip(rx,ry):
            mx, my = draw.circle(xo, yo, box_radius)
            mean = np.median(data[mx,my])
            if(mean >= threshold):
                return (xo, yo)
            if(mean > highestsum):
                highestsum = mean
        print(r, highestsum)
    return (0,0)

#test


from astropy.io import fits
import matplotlib.pyplot as plt

data = fits.getdata("/Users/Keanu/Documents/GitHub/ASL-Variable_Stars/data/04 - Edited Images/01 - TV Lyn/10s/Cleaned/mes_2023-01-18_02_56_56.FITS",ext=0)
#plt.plot(data, "test")
data = np.array(data, dtype=np.float64) - float(np.median(data))
print(data.shape)
peak = peak_finder(data,1400,2600, 50)
print(peak)


plt.imshow(data, cmap='gray', vmin=0, vmax=300)
plt.scatter(peak[1], peak[0])
plt.title("title")
plt.show()
