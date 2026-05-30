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
import string
import pickle
import sys
import os
import warnings
#warnings.filterwarnings("error")
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

#mzmlfile = '/home/sfo/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
#mzmlfile = '/home/sfo/store/flowcharacterizations/round5/mzMLs/20210312_E5_CG_high_tw1.mzML'
mzmlfile = '/home/sfo/store/flowcharacterizations/round3/mzMLs/200901_1s-dyn-300-200_R0.mzML'
#mzmlfile = '/home/sfo/store/data/PXD051214/mzMLs/JMM-6.mzML'
#mzmlfile = '/home/sfo/store/data/PXD035665/mzMLs/GradientAmount_HeLa_4ug_240min_R01.mzML'
#mzmlfile = '/home/sfo/store/data/PXD017618/mzMLs/20161222_Q1_MD_colQ1-33_Ecoli_Mat_B3.mzML'
basefolder = '/'.join((mzmlfile.split('/')[:-2]))
basefile = mzmlfile.split('/')[-1].split('.mzML')[0]

#investigationlocation = '/'.join((basefolder, 'investigations'))
#if not os.path.isdir(investigationlocation):
#    os.mkdir(investigationlocation)
#
#processinglocation = '/'.join((investigationlocation, basefile))
#if not os.path.isdir(processinglocation):
#    os.mkdir(processinglocation)

def better_histogram(numbers, nbins):
    nt = time()
    numbers = np.sort(numbers).tolist()
    binboundaries = np.geomspace(min(numbers), max(numbers), nbins+1)
    print(time() - nt, 'sorting')
    nt = time()
    biniter = iter(binboundaries.tolist())
    outbins = Counter()
    currentbin = next(biniter)
    for n in numbers:
        while True:
            if n > currentbin:
                currentbin = next(biniter)
            else:
                break
        #matchingbin = np.where(n <= binboundaries)[0][0]
        outbins[currentbin] += 1
    print(time() - nt, 'organizing')
    nt = time()
    binedges = sorted(outbins.keys())
    counts = [outbins[edge] for edge in binedges]
    binwidths = np.diff(binedges)
    finalbin = binwidths[-1] / (binwidths[0] / binwidths[1])
    binwidths = binwidths.tolist()
    binwidths.append(finalbin)
    plt.bar(binedges, counts, width=binwidths, align='edge', edgecolor='black', log=True, alpha=0.8)
    plt.xscale('log')
    plt.yscale('log')
    print(time() - nt, 'plotting')
    return outbins

msrun = mzml.MzML(mzmlfile, dtype=np.float64)
nprocs = os.cpu_count()

pcounts = defaultdict(dict)
massdifferences = defaultdict(list) #ms level: [differences between ajacent masses]

nt = time()

def mass_differences(scan):
    mslevel = scan['ms level']
    massdiffs = np.diff(scan['m/z array']).tolist()
    return mslevel, massdiffs

mscounts = Counter()
for mslevel, diffs in msrun.map(mass_differences):
    massdifferences[mslevel].extend(diffs)
    mscounts[mslevel] += 1
print('ms level scans:')
print(mscounts)


#for scan in msrun:
#    mslevel = scan['ms level']
#    massdiffs = np.diff(scan['m/z array']).tolist()
#    massdifferences[mslevel].extend(massdiffs)
    #scanparams = scan['scanList']['scan'][0]
    #scanbounds = scanparams['scanWindowList']['scanWindow'][0]
    #if 'mass resolving power' in scanparams:
    #    if 'mass resolving power' not in pcounts[mslevel]:
    #        pcounts[mslevel]['mass resolving power'] = Counter()
    #    pcounts[mslevel]['mass resolving power'][scanparams['mass resolving power']] += 1
    #if 'filter string' in scanparams:
    #    if 'filter string' not in pcounts[mslevel]:
    #        pcounts[mslevel]['filter string'] = Counter()
    #    pcounts[mslevel]['filter string'][scanparams['filter string']] += 1
    #if 'ion injection time' in scanparams:
    #    if 'ion injection time' not in pcounts[mslevel]:
    #        pcounts[mslevel]['ion injection time'] = []
    #    pcounts[mslevel]['ion injection time'].append(scanparams['ion injection time'].real)
    #if 'intensity sums' not in pcounts[mslevel]:
    #    pcounts[mslevel]['intensity sums'] = []
    #intensities = scan['intensity array']
    #pcounts[mslevel]['intensity sums'].append(intensities.sum())
    #if 'n masses' not in pcounts[mslevel]:
    #    pcounts[mslevel]['n masses'] = []
    #pcounts[mslevel]['n masses'].append(intensities.size)
    #if 'mass bounds' not in pcounts[mslevel]:
    #    pcounts[mslevel]['mass bounds'] = Counter()
    #massboundstring = str(scanbounds['scan window lower limit'].real) + '-' + str(scanbounds['scan window upper limit'].real)
    #pcounts[mslevel]['mass bounds'][massboundstring] += 1
    #if 'precursorList' in scan:
    #    fragparams = scan['precursorList']['precursor'][0]
    #    if 'activation' not in pcounts[mslevel]:
    #        pcounts[mslevel]['activation'] = Counter()
    #    pcounts[mslevel]['activation'][list(fragparams['activation'])[0]] += 1

print(time() - nt, 'file processing')

for mslevel, diffs in massdifferences.items():
    counts = better_histogram(diffs, 30)
    plt.title('ms level' + str(mslevel))
    plt.xlabel('adjacent mass differences')
    plt.show()
    for k, v in counts.most_common(10):
        print(k, v)
    print(f'~ ms level {mslevel} tops')
    for k in sorted(counts)[:10]:
        print(k, counts[k])
    print(f'~ ms level {mslevel} mins')
