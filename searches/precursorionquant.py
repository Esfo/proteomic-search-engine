import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from colour import Color
from pyteomics import mzml
from time import time
import pandas as pd
import gc
import concurrent.futures
import multiprocessing as mp
from collections import Counter, defaultdict
from textwrap import wrap
from scipy import sparse, integrate, spatial, stats, special, signal
from pandas.api.types import CategoricalDtype
from sklearn.neighbors import NearestNeighbors
from more_itertools import sort_together
from distinctipy import distinctipy as dp
from functools import partial
import math
import lmdb
import random
import itertools
import pickle
import sys
import os
import warnings
#warnings.filterwarnings("error")
np.set_printoptions(suppress=True)
#np.warnings.filterwarnings('ignore')
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

proton = 1.007276554940804
chargetolerance = 0.1
minpoints = 3
mzmlfile = '/home/sfo/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
#mzmlfile = '/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
nprocs = 8

basefolder = '/'.join((mzmlfile.split('/')[:-2]))
basefile = mzmlfile.split('/')[-1].split('.mzML')[0]

processinglocation = '/'.join((basefolder, 'fileprocessing', basefile))
librarylocation = '/home/sfo/data/proteomics/fastas/search-db'

proteomefile = '/home/sfo/data/proteomics/fastas/proteomes/Human_Homo_sapien.fasta'
proteome = proteomefile.split('/')[-1].split('.')[0]

environment_partial = partial(lmdb.Environment, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=99999, max_spare_txns=1, lock=True)

linesbyscanbysubformulafile = '/'.join((processinglocation, 'linesbyscanbysubformula.pickle'))
with open(linesbyscanbysubformulafile, 'rb') as pick:
    linesbyscanbysubformula = pickle.load(pick)
#linesbyscanbysubformula = {} #subformula: scan: [lines]

linesofscansfile = '/'.join((processinglocation, 'linesofscans.pickle'))
with open(linesofscansfile, 'rb') as pick:
    linesofscans = pickle.load(pick)
#linesofscans = {} #scan: [lines]

mslevelfile = '/'.join((processinglocation, 'centroid.ms2.pickle'))
with open(mslevelfile, 'rb') as pick:
    ms2scans = pickle.load(pick)
#scanmasses = {} #scan: [[masses], [intensities]]

regionfile = '/'.join((processinglocation, 'regions.pickle'))
with open(regionfile, 'rb') as pick:
    regions = pickle.load(pick)
#regions as [minmass, maxmass, mintime, maxtime, # datapoints, peakarea, maxintensity, wmean, lineid]

def nearest_neighbors(baselist, flylist):
    baseind = 0
    matches = {} #flyind: baseind
    mininds = {} #baseind: flyind
    mindists = {} #baseind: mindist
    for nf, f in enumerate(flylist):
        mindist = math.inf
        for n, b in enumerate(baselist[baseind:]):
            dist = abs(b-f)
            if dist < mindist:
                minind = n + baseind
                mindist = dist
            else:
                break
        matches[nf] = minind
        mindists[nf] = mindist
        baseind = minind
    #removing redundant matches, only matching whatever is actually nearest to something
    outmatches = {} #baseind: flyind
    outmin = defaultdict(lambda: np.inf)
    for f, b in matches.items():
        if mindists[f] < outmin[b]:
            outmin[b] = mindists[f]
            outmatches[b] = f
    return outmatches

scanlist = set()
for subformula, scans in linesbyscanbysubformula.items():
    scanlist.update(scans)
scanlist = list(scanlist)

allintensities = []
for scan, v in ms2scans.items():
    allintensities.extend(v['intensity array'])

#scan: line: precursor intensity
#scan: total percursor %
#percents = []
shoulds = []
#haves = []
sumintensities = []
for scan in scanlist:
    scanlines = linesofscans[scan]
    linemasses = regions[scanlines,7]
    ms2scan = ms2scans[scan]
    intensities = ms2scan['intensity array']
    masses = ms2scan['m/z array']
    matches = nearest_neighbors(masses, linemasses.tolist())
    massinds, lineinds = map(list, zip(*matches.items()))
    #percents.extend(intensities[massinds] / intensities.sum())
    shoulds.append(len(linemasses))
    #haves.append(len(massinds))
    sumintensities.append(intensities.sum())

