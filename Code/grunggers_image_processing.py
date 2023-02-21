import numpy as np
from skimage import draw

#
def peak_finder(data, startx, starty, threshold, box_radius=10, abort=500):
    for r in range(abort):
        rx, ry = draw.circle_perimeter(startx, starty, radius=r+1, shape=data.shape)
        for xo,yo in zip(rx,ry):
            mx, my = draw.disk((xo, yo), box_radius)
            mean = np.median(data[mx,my])
            if(mean >= threshold):
                #print("found", r)
                return (xo, yo)
    return (0,0)


def peak_finder_it(data, startx, starty, threshold=50, box_radius=10, abort=500):
    pos = np.zeros(len(data[:,0,0]), dtype=object)
    for i in range(len(data[:,0,0])):
        #print("data entry: ", i)
        pos[i] = peak_finder(data[i], startx, starty, threshold, box_radius=box_radius, abort=abort)
        #(startx, starty) = pos[i]
    return pos
