from pyteomics import mzml
import os
import numpy as np
import pandas as pd
import time
from scipy import signal
from matplotlib import pyplot as plt
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

folder = 'C:/Base/MS_Files/FE/mzMLs/'
#file = 'C:/Base/MS_Files/FE/mzMLs/190701_F200_yM1.mzML'
outfolder = 'C:/Base/data/FE/'

ppmwindow = 30
# intensityfilter = 1e6

def ecdf(data):
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x,y)

for f in os.listdir(folder):
    file = ''.join((folder, f))
    print(file)
    msrun = mzml.MzML(file)
    masses = []
    intensities = []
    for spec in msrun:
        # inds = spec['intensity array'] > intensityfilter
        # masses.extend(spec['m/z array'][inds])
        masses.extend(spec['m/z array'])
        # intensities.extend(spec['intensity array'])
    msrun.close()

    masses = np.asarray(masses)
    # intensities = np.asarray(intensities)
    
    x, y = ecdf(masses)
    plt.plot(x,y,'.', color='orange')
    plt.vlines(masses.mean(), ymin=0, ymax=1, color='purple', label='Mean')
    plt.vlines(np.median(masses), ymin=0, ymax=1, color='green', label='Median')
    plt.hlines(0.5, xmin=0, xmax=x.max())
    plt.legend()
    plt.show()
    
    # x, y = ecdf(intensities)
    # plt.plot(x,y,'.', color='orange')
    # plt.vlines(intensities.mean(), ymin=0, ymax=1, color='purple', label='Mean')
    # plt.vlines(np.median(intensities), ymin=0, ymax=1, color='green', label='Median')
    # plt.hlines(0.5, xmin=0, xmax=x.max())
    # plt.xlim(0,5000000)
    # plt.legend()
    # plt.show()