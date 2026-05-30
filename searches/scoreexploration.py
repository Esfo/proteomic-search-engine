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
from decimal import Decimal, getcontext
import tempfile
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

mzmlfile = '/home/sfo/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
#mzmlfile = '/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
basefolder = '/'.join((mzmlfile.split('/')[:-2]))
basefile = mzmlfile.split('/')[-1].split('.mzML')[0]

processinglocation = '/'.join((basefolder, 'fileprocessing', basefile))
fragmentlocation = '/'.join((basefolder, 'fileprocessing', basefile, 'fragments'))
scanalytelocation = '/'.join((basefolder, 'fileprocessing', basefile, 'scanalytegroups'))
librarylocation = '/home/sfo/data/proteomics/fastas/search-db'
csvfilename = '/'.join((processinglocation, 'fragment.matches'))
proteome = 'Human_Homo_sapien'
nprocs = 8
proton = 1.007276554940804
dividingthreshold = 0.8
ppmtol = 25
ppmmod = ppmtol / 1000000

#subformularankingsfile = '/'.join((processinglocation, 'subformularankings.csv'))
#peptidetimeseriesfile = '/'.join((processinglocation, 'peptidetimeseries.csv'))
testrankingsfile = '/'.join((processinglocation, 'testrankings.csv'))
peptiderankingsfile = '/'.join((processinglocation, 'peptiderankings.csv'))
altids = '/home/sfo/store/flowcharacterizations/round3/crux-output/200901_fR_400.percolator.target.peptides.txt'

def better_histogram(numbers, nbins):
    numbers = np.sort(numbers).tolist()
    #binboundaries = np.geomspace(min(numbers), max(numbers), nbins+1)
    binboundaries = np.linspace(min(numbers), max(numbers), nbins+1)
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
    binedges = sorted(outbins.keys())
    counts = [outbins[edge] for edge in binedges]
    #binwidths = np.diff(binedges)
    #finalbin = binwidths[-1] / (binwidths[0] / binwidths[1])
    #binwidths = binwidths.tolist()
    #binwidths.append(finalbin)
    binwidths = np.diff(sorted(outbins)).min() / 2
    plt.bar(binedges, counts, width=binwidths, align='edge', edgecolor='black', log=True, alpha=0.8)
    #plt.xscale('log')
    #plt.yscale('log')
    return

def better_histogram2(numbers, nbins):
    numbers = np.sort(numbers).tolist()
    binboundaries = np.geomspace(min(numbers), max(numbers), nbins+1)
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
    binedges = sorted(outbins.keys())
    counts = [outbins[edge] for edge in binedges]
    binwidths = np.diff(binedges)
    finalbin = binwidths[-1] / (binwidths[0] / binwidths[1])
    binwidths = binwidths.tolist()
    binwidths.append(finalbin)
    plt.bar(binedges, counts, width=binwidths, align='edge', edgecolor='black', log=True, alpha=0.8)
    plt.xscale('log')
    plt.yscale('log')
    return

#peptideheaders = ['sequence', 'analyteid', 'score', 'ion_coverage', 'scan_indices']
df = pd.read_csv(peptiderankingsfile)

adf = pd.read_csv(altids, delimiter='\t')
adf.sort_values('percolator q-value', inplace=True)
aseqs = set(adf.loc[:,'sequence'])

#all scores are unique?
sdf = df.sort_values('score')

ndf = pd.read_csv(testrankingsfile, converters={'score': Decimal})

ss = sdf.loc[:,'score'].to_numpy()[::-1]
ns = ndf.loc[:,'score'].to_numpy().astype(float)
print((ss == ns).all())

inds = ss != ns
s1 = ss[inds]
n1 = ns[inds]

inds = ns != np.sort(ns)
n1 = ns[inds]
n2 = np.sort(ns)[inds]
print(inds.sum())


ns = ndf.loc[:,'score'].to_numpy().astype(float)
(ns == np.sort(ns)).all()
#passes when loading in score as Decimal

#not everything matches perfectly
#this is because pandas is interpreting the values as floats -> precision errors
#ndf is being read as strings and im converting to floats in numpy -> more accuracy i guess
