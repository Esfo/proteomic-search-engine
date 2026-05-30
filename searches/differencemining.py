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
from collections import Counter, defaultdict
from scipy import sparse, integrate, spatial, stats, special
from sklearn.neighbors import NearestNeighbors, KernelDensity
from distinctipy import distinctipy as dp
import random
import itertools
import pickle
import sys
import os
gc.enable()

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
        '#e85d58',
        '#b88cfa',
        '#f5972c',
        '#2ded8d',
        '#4bc8f2',
        '#ea68f2',
        ]
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', chexes)
#n = 0
#for c in chexes:
#    plt.plot([n, n+1], [n, n+1], color=c)
#    n += 1
#plt.show()

mzmlfile = '/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
basefolder = '/'.join((mzmlfile.split('/')[:-2]))
basefile = mzmlfile.split('/')[-1].split('.mzML')[0]

processedroot = '/'.join((basefolder, 'fileprocessing', basefile))
librarylocation = '/home/sfo/data/proteomics/fastas/isotope-arrays/human_isotopes-6-50_miss-1_ss50'

matchfile = '/'.join((processedroot, 'distributionmatches.pairs.pickle'))
with open(matchfile, 'rb') as pick:
    matchpairs = pickle.load(pick)

matchfile = '/'.join((processedroot, 'distributionmatches.conjoined.pickle'))
with open(matchfile, 'rb') as pick:
    distributionmassdict, distributionintensities = pickle.load(pick)

#first up: change librarygeneration and the end of chargehandling to save distributions as a 2xn numpy array (or whichever orientation takes less memory), the current dict format is more expensive memory/disk wise and isn't even needed. it'll be [[masses], [intensities]]

#for the below calculations you should either compare matched-to-nonmatched differences with matched-to-matched differences, or the positive/negative differences generated below as means of taking out known differences ie amino acids and things of that sort. I think one of these strategies will work, in some way, but I need to figure out which direction to take these perspectives in.

matcheddists = set(j for i, j in matchpairs.values())

#collecting masses at every increase in intensity
matchedgroups = []
nonmatchedgroups = []
for k, masses in distributionmassdict.items():
    intensities = distributionintensities[k].tolist()
    diffs = [intensities[i+1] - intensities[i] >= 0 for i in range(len(intensities)-1)]
    diffs.insert(0, True)
    candidates = masses[diffs].tolist()
    if k in matcheddists:
        matchedgroups.extend(zip(candidates, itertools.repeat(k, len(candidates))))
    else:
        nonmatchedgroups.extend(zip(candidates, itertools.repeat(k, len(candidates))))

#for density: take each point and expand their range by the minimum distance available (or maybe this shouldn't be a min if you're going after expansion rates), and check for the number of things becoming encompassed per distance, so not just a number within, but a rate of expansion -> use as density surrogate
#I want to try a numpy approach to this with concurrency

nmsorted, nmkeys = zip(*sorted(nonmatchedgroups, key=lambda x: x[0]))
msorted, mkeys = zip(*sorted(matchedgroups, key=lambda x: x[0]))

nt = time()
nmiter = iter(nmsorted)
searchrange = 500 #daltons, unimod masses range from -156 to 3550
nmradius = []
#differencecounter = Counter()
differences = []

dm = msorted[0]
lm = next(nmiter)
adist = abs(dm - lm)
while adist >= searchrange: #making sure the matched and nonmatched are within the same range before starting
    lm = next(nmiter)
    adist = abs(dm - lm)

nmradius.append(lm)
for dm in msorted:
    #this can be faster by calculating min/max differences from dm then applying it in the next 2 loops as < or >
    adist = abs(dm - lm)
    while adist <= searchrange:
        nmradius.append(lm)
        lm = next(nmiter)
        adist = abs(dm - lm)
    lremovals = []
    for lr in nmradius: #nmradius is in order so this works
        if abs(lr - dm) > searchrange:
            lremovals.append(lr)
        else:
            break
    for lr in lremovals:
        nmradius.remove(lr)
    for lr in nmradius:
        differences.append(lr-dm)
        #differencecounter[round(lr - dm, 4)] += 1
print(time() - nt)
differences = np.array(differences)

saverloc = '/'.join((processedroot, 'distribution.differences.pickle'))
with open(saverloc, 'wb') as pick:
    pickle.dump(differences, pick)
