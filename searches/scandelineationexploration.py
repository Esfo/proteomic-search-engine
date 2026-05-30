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

mslevelfile = '/'.join((processinglocation, 'centroid.ms2.pickle'))
with open(mslevelfile, 'rb') as pick:
    ms2scans = pickle.load(pick)
#scanmasses = {} #scan: [[masses], [intensities]]

linesbyscanbysubformulafile = '/'.join((processinglocation, 'linesbyscanbysubformula.pickle'))
with open(linesbyscanbysubformulafile, 'rb') as pick:
    linesbyscanbysubformula = pickle.load(pick)
#linesbyscanbysubformula = {} #subformula: scan: [lines]

aminoacidcomposition = {
        'A': {'C': 3, 'H': 5, 'N': 1, 'O': 1},
        'R': {'C': 6, 'H': 12, 'N': 4, 'O': 1},
        'N': {'C': 4, 'H': 6, 'N': 2, 'O': 2},
        'D': {'C': 4, 'H': 5, 'N': 1, 'O': 3},
        'C': {'C': 3, 'H': 5, 'N': 1, 'O': 1, 'S': 1},
        'Q': {'C': 5, 'H': 8, 'N': 2, 'O': 2},
        'E': {'C': 5, 'H': 7, 'N': 1, 'O': 3},
        'G': {'C': 2, 'H': 3, 'N': 1, 'O': 1},
        'H': {'C': 6, 'H': 7, 'N':3, 'O': 1},
        'I': {'C': 6, 'H': 11, 'N':1, 'O': 1},
        'L': {'C': 6, 'H': 11, 'N':1, 'O': 1},
        'K': {'C': 6, 'H': 12, 'N': 2, 'O': 1},
        'M': {'C': 5, 'H': 9, 'N':1, 'O': 1, 'S': 1},
        'F': {'C': 9, 'H': 9, 'N':1, 'O': 1},
        'P': {'C': 5, 'H': 7, 'N':1, 'O': 1},
        'S': {'C': 3, 'H': 5, 'N':1, 'O': 2},
        'T': {'C': 4, 'H': 7, 'N':1, 'O': 2},
        'W': {'C': 11, 'H': 10, 'N': 2, 'O': 1},
        'Y': {'C': 9, 'H': 9, 'N': 1, 'O': 2},
        'V': {'C': 5, 'H': 9, 'N': 1, 'O': 1}
        }

elementalmasses = { #isotope: mass
            'H1': 1.00782503223,
            'H2': 2.01410177812,
            'C12': 12.0000000, 
            'C13': 13.00335483507,
            'N14': 14.00307400443,
            'N15': 15.00010889888,
            'O16': 15.99491461957,
            'O17': 16.99913175650,
            'O18': 17.99915961286,
            'S32': 31.9720711744,
            'S33': 32.9714589098,
            'S34': 33.967867004,
            'S36': 35.96708071}

monoisotopickeys = { #element: monoisotopic element
        'H': 'H1',
        'C': 'C12',
        'N': 'N14',
        'O': 'O16',
        'S': 'S32'}

scanlist = set()
for subformula, scans in linesbyscanbysubformula.items():
    scanlist.update(scans)

viablems2scans = {} #scan: [[masses], [intensities]]
for scan in list(scanlist):
    viablems2scans[scan] = np.array([ms2scans[scan]['m/z array'].tolist(), ms2scans[scan]['intensity array'].tolist()])

aminomasses = {k: sum(elementalmasses[monoisotopickeys[i]] * j for i, j in comp.items()) for k, comp in aminoacidcomposition.items()}
water = elementalmasses['H1'] * 2 + elementalmasses['O16']

massarray = np.array(list(aminomasses.values()))

#scoring ions of scans:
#min/max mass range via amino acids for the MS2 range, any AAs can be used
#use these AAs to then determine a potential delimiter process for dividing the ions into "groups", 100da straight might be too simple
#top ion is 1 / (number of ions in group being tested), which is the most intense
#15 ions becomes: 1/15, 2/14, 3/13, etc as they decrease in intensity
#OR maybe it can just be that ion's percentage of the total ions in that group

maxlen = 0
maxind = -1
maxint = 0
for k, v in viablems2scans.items():
    mza, intensities = v
    if len(mza) > maxlen:
        maxlen = len(mza)
        #maxind = k
    if intensities.mean() > maxint:
        maxint = intensities.mean()
        maxind = k

mza, intensities = viablems2scans[maxind]

maxmass = mza.max()
minmass = mza.min()

masslevels = []
masslevels.append(massarray.round())
for _ in range(np.ceil(round(maxmass) / massarray.min()).astype(int) - 1):
    newlevel = (masslevels[-1] + massarray[:,None]).flatten()
    roundlevel = np.round(newlevel).astype(int)
    newlevel = np.unique(roundlevel)
    masslevels.append(newlevel)

levelranges = [[i.min(), i.max()] for i in masslevels]
flatranges = np.array(list(itertools.chain.from_iterable(levelranges)))
flatranges = np.sort(flatranges[flatranges <= maxmass])

firstind = (flatranges <= minmass).sum() - 1
flatranges = flatranges[firstind:]
flatranges[0] = np.floor(minmass)
flatranges = flatranges.astype(int).tolist()

#it's going to be worth testing other matrices
# - cutting levelranges off once it hits maxmass and taking all those indices might work too, limiting to 16 again
# - check the raw 100 distance

#i'm hard-coding 16 to be the smallest range, its the most common difference from the matrix of differences of massarray from itself, and it seems like a reasonable minimum i suppose
while True:
    removal = False
    for n in range(len(flatranges)-1):
        l = flatranges[n]
        r = flatranges[n+1]
        diff = r - l
        if diff < 16:
            removal = True
            break
    if removal:
        flatranges.remove(r)
    else:
        break

maxmass = mza.max()
if maxmass - flatranges[-1] < 16:
    flatranges[-1] = int(np.ceil(maxmass))
else:
    flatranges.append(int(np.ceil(maxmass)))

plt.vlines(flatranges, ymin=0, ymax=intensities.max(), linewidth=0.5, color='black')
plt.bar(mza, intensities, width=4)
plt.show()


#if all intensities are similar percentage wise to the sum, it should be worth less
#ions that stand out get rewarded

#rangebounds = np.stack((flatranges[:-1], flatranges[1:]), axis=1).tolist()
#rangecounts = []
#for n, (l, r) in enumerate(rangebounds):
#    secintensities = intensities[np.logical_and(mza >= l, mza < r)]


#below i'm exploring properties of scans, 
#there's ~5600 peptide IDs and ~5k protein IDs via crux for fr400
#^i should be able to see that kind of viability here

intensitymeans = {} #scan: mean intensity of scan
intensitysums = {} #scan: sum intensity of scan
ioncount = {} #scan: number of ions
countabovemean = {} #scan: number of ions above mean intensity
sumintensityabovemean = {} #scan: sum intensity of ions above mean
percentcountabovemean = {} #scan: % of ions above mean intensity
percentintensityabovemean = {} #scan: % of total scan intensity held by ions above mean intensity
averagemasses = {} #scan: average mass
weightedaveragemasses = {} #scan: weighted by intensity
for k, v in viablems2scans.items():
    mza, intensities = v
    meanintensity = intensities.mean()
    sumintensity = intensities.sum()
    numberofions = len(intensities)
    intensitymeans[k] = meanintensity
    intensitysums[k] = sumintensity
    ioncount[k] = numberofions
    abovemean = intensities[intensities > meanintensity]
    sumintensityabovemean[k] = abovemean.sum()
    countabovemean[k] = len(abovemean)
    percentcountabovemean[k] = len(abovemean) / numberofions
    percentintensityabovemean[k] = abovemean.sum() / sumintensity
    averagemasses[k] = mza.mean()
    weightedaveragemasses[k] = (mza * intensities).sum() / intensities.sum()

plt.plot(intensitymeans.values(), intensitysums.values(), '.')
plt.show()

