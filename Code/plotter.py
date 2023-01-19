import matplotlib.pyplot as plt
import numpy as np


def plot(data, title):
    data = np.array(data, dtype=np.float64) - float(np.median(data))
    plt.imshow(data, cmap='gray', vmin=0, vmax=200)
    plt.colorbar()
    plt.title(title)
    plt.show()
