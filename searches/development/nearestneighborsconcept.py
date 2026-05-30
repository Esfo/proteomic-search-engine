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
from distinctipy import distinctipy as dp
from functools import partial
from pickleshare import PickleShareDB
from bisect import bisect_left, bisect_right
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
    plt.rcParams['figure.dpi'] = 160
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

def radius_neighbors(baselist, flylist):
    b = 0
    pool = []
    matches = [] #flylist locations, baselist indices
    biter = enumerate(baselist)
    for f in flylist:
        ftol = f * ppmmod
        fmin = f - ftol
        fmax = f + ftol
        removals = []
        submatches = []
        for pi, pb in pool:
            if pb < fmin:
                removals.append([pi, pb])
            elif pb <= fmax:
                submatches.append(pi)
        for r in removals:
            pool.remove(r)
        while b <= fmax:
            try:
                i, b = next(biter)
                if b >= fmin:
                    pool.append([i, b])
                    if b <= fmax:
                        submatches.append(i)
            except StopIteration:
                break
        matches.append(submatches)
    return matches

def radius_neighbors_dictout(baselist, flylist):
    b = 0
    pool = []
    matches = {} #flylist index: [baselist indices]
    biter = enumerate(baselist)
    for fn, f in enumerate(flylist):
        ftol = f * ppmmod
        fmin = f - ftol
        fmax = f + ftol
        removals = []
        submatches = []
        for pi, pb in pool:
            if pb < fmin:
                removals.append([pi, pb])
            elif pb <= fmax:
                submatches.append(pi)
        for r in removals:
            pool.remove(r)
        while b <= fmax:
            try:
                i, b = next(biter)
                if b >= fmin:
                    pool.append([i, b])
                    if b <= fmax:
                        submatches.append(i)
            except StopIteration:
                break
        #matches.append(submatches)
        if submatches:
            matches[fn] = submatches
    return matches

def radius_neighbors_dictout_optimized(baselist, flylist, ppmmod):
    matches = {}  # flylist index: [baselist indices]
    b_index = 0  # Index to track position in baselist
    len_baselist = len(baselist)

    for fn, f in enumerate(flylist):
        ftol = f * ppmmod
        fmin = f - ftol
        fmax = f + ftol
        submatches = []

        # Skip baselist elements that are less than fmin
        while b_index < len_baselist and baselist[b_index] < fmin:
            b_index += 1

        # Check elements in range [fmin, fmax]
        b_check = b_index  # Temporary index for checking in range
        while b_check < len_baselist and baselist[b_check] <= fmax:
            submatches.append(b_check)
            b_check += 1

        if submatches:
            matches[fn] = submatches

    return matches

def radius_neighbors_dictout_binary_search(baselist, flylist, ppmmod):
    matches = {}
    for fn, f in enumerate(flylist):
        ftol = f * ppmmod
        fmin = f - ftol
        fmax = f + ftol
        
        # Use binary search to find the relevant range in baselist
        start_idx = bisect_left(baselist, fmin)
        end_idx = bisect_right(baselist, fmax)
        
        # Only add to matches if there are indices within range
        if start_idx < end_idx:
            matches[fn] = list(range(start_idx, end_idx))
            
    return matches

def radius_neighbors_dictout_efficient(baselist, flylist, ppmmod):
    matches = {}
    start_index = 0  # Initialize start index for baselist
    blen = len(baselist)

    for fn, f in enumerate(flylist):
        ftol = f * ppmmod
        fmin = f - ftol
        fmax = f + ftol
        submatches = []

        # Move the start index forward to skip irrelevant elements
        while start_index < blen and baselist[start_index] < fmin:
            start_index += 1

        # Find matching elements within the tolerance range
        for i in range(start_index, blen):
            if baselist[i] > fmax:
                break  # Stop searching if beyond the tolerance range
            submatches.append(i)
        
        # Only add to matches if submatches are found
        if submatches:
            matches[fn] = submatches

    return matches

def radius_neighbors_dictout_new_approach(baselist, flylist, ppmmod):
    matches = {}
    last_start_index = 0  # Track the start index to avoid redundant scans

    for fn, f in enumerate(flylist):
        ftol = f * ppmmod
        fmin = f - ftol
        fmax = f + ftol

        # Use binary search to find the first index in baselist >= fmin
        start_index = bisect.bisect_left(baselist, fmin, lo=last_start_index)
        submatches = []

        # Scan from start_index to find all valid matches
        for i in range(start_index, len(baselist)):
            if baselist[i] > fmax:
                break  # Exit the loop if baselist[i] is beyond the acceptable range
            submatches.append(i)

        if submatches:
            matches[fn] = submatches

        last_start_index = start_index  # Update last_start_index for the next iteration

    return matches

import bisect

def radius_neighbors_dictout_enhanced(baselist, flylist, ppmmod):
    matches = {}
    
    for fn, f in enumerate(flylist):
        ftol = f * ppmmod
        fmin = f - ftol
        fmax = f + ftol

        # Find the start and end indices in baselist that fall within the tolerance range
        start_index = bisect.bisect_left(baselist, fmin)
        end_index = bisect.bisect_right(baselist, fmax)

        # If there are matches, record their indices
        if start_index != end_index:
            matches[fn] = list(range(start_index, end_index))

    return matches



def kd_neighbors(baselist, flylist):
    radius = (flylist * ppmmod).flatten()
    nn = spatial.KDTree(np.array(baselist))
    out = nn.query_ball_point(flylist, r=radius)
    return out

def nearest_neighbors(baselist, flylist):
    baseind = 0
    indices = []
    distances = []
    for f in flylist:
        mindist = math.inf
        for n, b in enumerate(baselist[baseind:]):
            dist = abs(b-f)
            if dist < mindist:
                minind = n + baseind
                mindist = dist
            else:
                break
        indices.append(minind)
        distances.append(mindist)
        baseind = minind
    return np.array(distances), np.array(indices)

#slow no matter what????
def nearest_neighbors2(baselist, flylist):
    baseind = 0
    indices = []
    distances = []
    for f in flylist.tolist():
        mindist = math.inf
        for n, b in enumerate(baselist[baseind:].tolist()):
            dist = abs(b-f)
            if dist < mindist:
                minind = n + baseind
                mindist = dist
            else:
                break
        indices.append(minind)
        distances.append(mindist)
        baseind = minind
    return np.array(distances), np.array(indices)

#the nearest concept isnt the same as the last
#both lists must be sorted prior
#iterate the flylist -> check for nearest
#once the distance increases away, you've found it
#when this happens you need to iterate from the last 2 baselist points, starting with the one that was just found to be the last winning match
#so instead of having a pool, you would just have the normal baselist and retain the index to start at


def kd_nearest(baselist, flylist):
    nn = spatial.KDTree(baselist[:,None])
    dists, inds = nn.query(flylist[:,None])
    return dists, inds

ppmtol = 25 #ppm
ppmmod = ppmtol / 1000000

nopes = []
outputtimes = []
outputlengths = []
for _ in range(100):
    #assuming pre-sorted and optimal formats
    times = []

    baseint = np.random.randint(10000, 500000)
    flyint = np.random.randint(10000, 500000)
    basemax = np.random.uniform(10, 10000)
    flymax = np.random.uniform(10, 10000)
    outputlengths.append([baseint, flyint])
    
    baselist = np.sort(np.random.uniform(0,basemax, size=baseint))[:,None] #match to these, return their indices
    flylist = np.sort(np.random.uniform(0,flymax, size=flyint))[:,None] #match these and return baselist indices AT each flylist index they match to
    
    #nt = time()
    #kdout = kd_neighbors(baselist, flylist)
    #st = time() - nt
    #times.append(st)
    
    baselist = baselist.flatten()
    flylist = flylist.flatten()
    
    #nt = time()
    #flymatches = radius_neighbors(baselist, flylist)
    #st = time() - nt
    #times.append(st)
    #
    nt = time()
    fly2 = radius_neighbors_dictout(baselist, flylist)
    st = time() - nt
    times.append(st)

    nt = time()
    
    nt = time()
    fly3 = radius_neighbors_dictout(baselist.tolist(), flylist.tolist())
    st = time() - nt
    times.append(st)

    nt = time()
    #fly3 = radius_neighbors_dictout_optimized(baselist, flylist, ppmmod)
    #fly3 = radius_neighbors_dictout_binary_search(baselist, flylist, ppmmod)
    #fly3 = radius_neighbors_dictout_efficient(baselist, flylist, ppmmod)
    #fly3 = radius_neighbors_dictout_new_approach(baselist, flylist, ppmmod)
    #fly3 = radius_neighbors_dictout_enhanced(baselist, flylist, ppmmod)
    #st = time() - nt
    #times.append(st)
    #
    #if fly2 != fly3:
    #    print('nope')
    #    break

    #nt = time()
    #dists1, inds1 = nearest_neighbors(baselist, flylist)
    #st = time() - nt
    #times.append(st)

    ##nt = time()
    ##dists2, inds2 = nearest_neighbors2(baselist, flylist)
    ##st = time() - nt
    ##times.append(st)

    #nt = time()
    #dists2, inds2 = kd_nearest(baselist, flylist)
    #st = time() - nt
    #times.append(st)
    
    outputtimes.append(times)

    #kdout = kdout.tolist()
    #if kdout != flymatches:
    #    print('mismatch!!!')
    #    nopes.append([baselist, flylist])

outputtimes = np.array(outputtimes)
outputlengths = np.array(outputlengths)

plt.plot(outputtimes[:,0], outputtimes[:,1], '.')
plt.plot([0,1],[0,1], '-')
plt.show()

#plt.plot(outputtimes[:,0], outputlengths[:,0], '.')
#plt.plot(outputtimes[:,1], outputlengths[:,1], '.')
#plt.show()
