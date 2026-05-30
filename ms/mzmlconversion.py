import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from colour import Color
import psutil
import asyncio
import aiofiles
from pyteomics import mzml
import csv
import bisect
import heapq
import fcntl #this will need to be portalocker on other operating systems
from time import time
import pandas as pd
import gc
import concurrent.futures
import multiprocessing as mp
from collections import Counter, defaultdict
from scipy import sparse, integrate, spatial, stats, special
from pandas.api.types import CategoricalDtype
from sklearn.neighbors import NearestNeighbors
from more_itertools import sort_together
from functools import partial
from pickleshare import PickleShareDB
import math
import zlib
import lmdb
import random
import itertools
import subprocess
import string
import pickle
import sys
import os
import warnings
#warnings.filterwarnings('error')
np.set_printoptions(suppress=True)
gc.enable()

#for more https://matplotlib.org/stable/tutorials/introductory/customizing.html
if os.uname()[1] == 'toaster':
    plt.rcParams['figure.dpi'] = 180
elif os.uname()[1] == 'box':
    plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.facecolor'] = 'gray'
plt.rcParams['figure.facecolor'] = 'gray'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['ytick.labelcolor'] = 'white'
plt.rcParams['xtick.labelcolor'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['xtick.color'] = 'white'
chexes = ['#ffffff',
        '#8ff6ff',
        '#ff9f9c',
        '#2ded8d',
        '#fbffb3',
        '#ea68f2',
        '#7d26ff',
        ]
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', chexes)
#n = 0
#for c in chexes:
#    plt.plot([n, n+1], [n, n+1], color=c, label=c)
#    n += 1
#plt.legend()
#plt.show()

folder = '/home/sfo/store/data/PXD017618/'
files = [i for i in os.listdir(folder) if i.endswith('.raw')]

outputfolder = ''.join((folder, 'mzMLs'))
if not os.path.isdir(outputfolder):
    os.mkdir(outputfolder)

nprocs = os.cpu_count()

def conversion(folder, f, centroid=False):
    #removed the -it from the docker call to make this run via python
    #if centroid:
    #    #dockercmd = f'docker run --rm -e WINEDEBUG=-all -v {folder}:/data chambm/pwiz-skyline-i-agree-to-the-vendor-licenses wine msconvert --mzML --filter "peakPicking true 1-" --outdir=mzMLs/ {f}'
    #dockercmd = f'docker run --rm -e WINEDEBUG=-all -v {folder}:/data chambm/pwiz-skyline-i-agree-to-the-vendor-licenses wine msconvert --mzML --filter "peakPicking true 1" --outdir=mzMLs/ {f}'
    #else:
    dockercmd = f'docker run --rm -e WINEDEBUG=-all -v {folder}:/data chambm/pwiz-skyline-i-agree-to-the-vendor-licenses wine msconvert --mzML --outdir=mzMLs/ {f}'
    subprocess.run(dockercmd, shell=True, check=True)

partial_conversion = partial(conversion, folder)

with mp.Pool(nprocs) as pool:
    pool.map(partial_conversion, files)
