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

def radius_neighbors_ppm_tolerance(baselist, flylist, ppmmod):
    b = 0
    pool = []
    matches = {} #flylist index: [baselist indices]
    biter = enumerate(baselist)
    for fn, fly in enumerate(flylist):
        ftol = fly * ppmmod
        fmin = fly - ftol
        fmax = fly + ftol
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
        if submatches:
            matches[fn] = submatches
    return matches

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
    return distances, indices

#iterate flylist
#full iterate baselist
#keep track of only 2 baselist values max at any time: the current and the last - no pool
#if baseval < flymin: go to the next baseval and check that as the new closest
#keep iterating basevals until you find the closest
#and if the new distance == the last distance, put both together in a list
#all values are stored in the end as a list

def nearest_neighbors_ppm_tolerance(baselist, flylist, ppmmod):
    baseind = 0
    indices = {} #flyindex: [baseindex] or [baseindex1, baseindex2]
    distances = {} #flyindex: minimum distance matched
    blen = baselist.size
    for fn, f in enumerate(flylist):
        mindist = math.inf
        distpass = False
        ftol = f * ppmmod
        fmin = f - ftol
        fmax = f + ftol
        basepass = True
        while basepass:
            for bn, b in enumerate(baselist[baseind:]):
                if b < fmin:
                    #b is too low
                    #find the next b that's above fmin
                    #this step saves speed in some outlier datasets
                    baseind = np.searchsorted(baselist, fmin)
                    if baseind == blen:
                        #baselist is finished, function is over
                        return distances, indices
                    else:
                        #iterate to the next b
                        break
                elif b <= fmax:
                    #b is between fmin and fmax
                    dist = abs(b-f)
                    if dist < mindist:
                        #winner winner chicken dinner, but don't pass go, don't collect $200 just yet
                        minind = [bn + baseind]
                        mindist = dist
                        distpass = True
                        basepass = False
                    elif dist == mindist:
                        #two symmetrical distances, nothing will be closer so move to the next f
                        minind = [minind[0], bn + baseind]
                        break
                    else:
                        #b > fmax and b > minind, move to the next f
                        break
                else:
                    #b is higher than fmax, iterate to the next f
                    basepass = False
                    break
        if distpass:
            indices[fn] = minind
            distances[fn] = mindist
            baseind = minind[-1]
    return distances, indices

def nearest_neighbors_ppm_tolerance_ss(baselist, flylist, ppmmod):
    indices = {} #baseindex: [flyindex] or [flyindex1, flyindex2]
    distances = {} #baseindex: distance
    for bn, rightfn in enumerate(np.searchsorted(flylist, baselist).tolist()): #iter the tolist for profiling
        b = baselist[bn]
        btol = b * ppmmod
        bmin = b - btol
        bmax = b + btol
        
        leftfn = rightfn - 1 #worse case scenario this is -1 -> left = False
        left = False
        leftf = flylist[leftfn]
        if leftf > bmin and leftf < bmax:
            left = True
        
        right = False
        try:
            rightf = flylist[rightfn]
            if rightf > bmin and rightf < bmax:
                right = True
        except IndexError:
            #rightfn == len(flylist), the iteration is over
            if not left:
                return distances, indices

        if left and right:
            leftdist = b - leftf
            rightdist = rightf - b
            if leftdist < rightdist:
                indices[bn] = [leftfn]
                distances[bn] = leftdist
            elif rightdist < leftdist:
                indices[bn] = [rightfn]
                distances[bn] = rightdist
            elif leftdist == rightdist:
                indices[bn] = [leftfn, rightfn]
                distances[bn] = leftdist
        elif left:
            leftdist = b - leftf
            indices[bn] = [leftfn]
            distances[bn] = leftdist
        elif right:
            rightdist = rightf - b
            indices[bn] = [rightfn]
            distances[bn] = rightdist
    return distances, indices

def nearest_neighbors_ppm_tolerance_ss2(baselist, flylist, ppmmod):
    indices = {} #baseindex: [flyindex] or [flyindex1, flyindex2]
    distances = {} #baseindex: distance
    for bn, rightfn in enumerate(np.searchsorted(flylist, baselist)):
        b = baselist[bn]
        btol = b * ppmmod
        bmin = b - btol
        bmax = b + btol
        
        leftfn = rightfn - 1 #worse case scenario this is -1 -> left = False
        left = False
        leftf = flylist[leftfn]
        if leftf > bmin and leftf < bmax:
            left = True
        
        right = False
        try:
            rightf = flylist[rightfn]
            if rightf > bmin and rightf < bmax:
                right = True
        except IndexError:
            #rightfn == len(flylist), the iteration is over
            if not left:
                return distances, indices

        if left and right:
            leftdist = b - leftf
            rightdist = rightf - b
            if leftdist < rightdist:
                indices[bn] = [leftfn]
                distances[bn] = leftdist
            elif rightdist < leftdist:
                indices[bn] = [rightfn]
                distances[bn] = rightdist
            elif leftdist == rightdist:
                indices[bn] = [leftfn, rightfn]
                distances[bn] = leftdist
        elif left:
            leftdist = b - leftf
            indices[bn] = [leftfn]
            distances[bn] = leftdist
        elif right:
            rightdist = rightf - b
            indices[bn] = [rightfn]
            distances[bn] = rightdist
    return distances, indices

# Assuming baselist is pre-sorted to avoid redundant sorting

#is it faster to compare two big lists?
#or is it faster to compare a bunch of smaller lists to one larger one?

#matches = self.radius_neighbors_ppm_tolerance(chargedfragments, ms2masses)

ppmtol = 25 #ppm
ppmmod = ppmtol / 1000000

outputtimes = []
for t in range(100):
    #assuming pre-sorted and optimal formats
    times = []

    baseint = np.random.randint(10000, 500000)
    flyint = np.random.randint(10000, 500000)
    basemax = np.random.uniform(10, 10000)
    flymax = np.random.uniform(10, 10000)
    
    baselist = np.sort(np.random.uniform(0, basemax, size=baseint)) #match to these, return their indices
    flylist = np.sort(np.random.uniform(0, flymax, size=flyint)) #match these and return baselist indices AT each flylist index they match to
    
    #baselist = baselist.tolist()
    #flylist = flylist.tolist()
    
    #nt = time()
    ##flymatches = radius_neighbors_ppm_tolerance(baselist, flylist, ppmmod)
    #flymatches = radius_neighbors_ppm_tolerance(flylist, baselist, ppmmod)
    #st = time() - nt
    #times.append(st)

    ##baselist = np.array(baselist)
    ##baselist = baselist.tolist()
    #flylist = np.array(flylist)
    #
    #nt = time()
    ##flymatches = nearest_neighbors(baselist, flylist)
    #flymatches = nearest_neighbors(flylist, baselist)
    #st = time() - nt
    #times.append(st)
    #
    ##baselist = baselist.tolist()
    ##flylist = np.array(flylist)
    #
    #nt = time()
    #ndists, nmatches = nearest_neighbors_ppm_tolerance(flylist, baselist, ppmmod)
    #st = time() - nt
    #times.append(st)
    #
    ##flylist = flylist.tolist()
    #flylist = np.array(flylist)
    #baselist = np.array(baselist)
    
    nt = time()
    n2dists, n2matches = nearest_neighbors_ppm_tolerance_ss(baselist, flylist, ppmmod)
    st = time() - nt
    times.append(st)
    
    nt = time()
    n2dists, n2matches = nearest_neighbors_ppm_tolerance_ss2(baselist, flylist, ppmmod)
    st = time() - nt
    times.append(st)
    
    outputtimes.append(times)
    print('trial', t, 'done')
    #if nmatches != n2matches or ndists != n2dists:
    #    print('problem found')
    #    break

outputtimes = np.array(outputtimes)

plt.boxplot(outputtimes)
plt.show()
