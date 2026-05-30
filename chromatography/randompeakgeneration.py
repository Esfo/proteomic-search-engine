import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from colour import Color
#from pyteomics import mzml
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
#from distinctipy import distinctipy as dp
from functools import partial
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

npeaks = 200
minpoints = 3
maxpoints = 12
noiselevel = 0.95
meanmax = 45
meanmulmax = 1e7
distrange = 10
peakfill = 0.2

#npeaks = 10
#minpoints = 5
#maxpoints = 20
#noiselevel = 0.9
#meanmax = 3
#meanmulmax = 1e6
#distrange = 5
#peakfill = 1


mindist = 8 #this is an index distance, min distance between allowable apexes
boxcarlength = 5 #n data points on either side of a 0 point to use in an averaging for both max and boundary finding, total points used is this number x2. Any of the encompassed data points that fall below this groups average are masked prior to any processes.
#make fusion peaks, add more than 1 window at a specific location

#notes:
#the only legitemate failures stem from the maxes not being properly elucidated in minpoiint_reduction. This can have varying causes. The major cause of this failing is actually when some of the random data generation puts to many 0s next to each other - this wouldn't happpen in real data to the best of my understanding. Other times there are non-gaussian blobs that are next to good-looking peaks that don't get picked up as easily. Meh, it happens. It's not the most realistic data in that form anyways, the blobs are not my worry.

def boolcount(b, counts=False):
    bc = np.where(np.diff(b, prepend=True))[0]
    bc = np.append(bc, len(b))
    if counts:
        bc = np.diff(bc, prepend=0)
    if bc.size % 2:
        bc = np.append(bc, 0)
    return bc.reshape(bc.size//2,2)

#def old_minpoint_reduction(array, mindist):
#    #this approach fails when the maxes are at the absolute beginning or end of an array
#    extramaxes = set()
#    mask = np.repeat(False, array.size)
#    while True:
#        narray = array[~mask]
#
#        forwardmaxcheck = np.append(narray[:-1] > narray[1:], False)
#        #backwardmaxcheck = np.append(False, narray[1:] > narray[:-1])
#        backwardmaxcheck = np.append(forwardmaxcheck[0], narray[1:] > narray[:-1])
#        forwardmaxcheck[-1] = backwardmaxcheck[-1]
#
#        forwardmincheck = np.append(narray[:-1] < narray[1:], False)
#        #backwardmincheck = np.append(False, narray[1:] < narray[:-1])
#        backwardmincheck = np.append(forwardmincheck[0], narray[1:] < narray[:-1])
#        forwardmincheck[-1] = backwardmincheck[-1]
#
#        newmask = np.logical_and(forwardmincheck, backwardmincheck)
#        mins = np.where(newmask)[0]
#        maxes = np.where(np.logical_and(forwardmaxcheck, backwardmaxcheck))[0]
#        extremadistances = (np.abs(maxes - mins.reshape(-1,1)) < mindist)
#        
#        maxestoholdonto = maxes[~extremadistances.any(axis=0)]
#        if maxestoholdonto.size > 0:
#            maxestoholdonto = (maxestoholdonto + mask.cumsum()[~mask][maxestoholdonto]).tolist()
#            extramaxes.update(maxestoholdonto)
#        
#        adjacentextremas = extremadistances.any()
#        if adjacentextremas:
#            maskinds = np.argwhere(~mask)[np.argwhere(newmask)].flatten()
#            mask[maskinds] = True
#        else:
#            break
#    
#    if not maxes.size:
#        maxes = narray.argmax() #this seems like an easier way to allow for maxes at the first or last point
#    
#    fmaxes = maxes + mask.cumsum()[~mask][maxes]
#    fmaxes = np.unique(np.append(fmaxes, list(extramaxes))).astype(int)
#    return fmaxes

def boundary_finding(fmaxes, array):
    fmaxiter = fmaxes.copy().tolist()
    fmaxiter = np.append(0, fmaxiter)
    fmaxiter = np.append(fmaxiter, len(array)-1)
    peakbounds = []
    for n, l in enumerate(fmaxiter[:-1]):
        r = fmaxiter[n+1] + 1
        if n > 0:
            rightseries = array[l:r]
            rightacc = np.minimum.accumulate(rightseries)
            rtrimmer = rightseries <= rightacc
            rightestimate = np.trim_zeros(rtrimmer, trim='b').size
            nr = l + rightestimate
            rightseries = array[l:nr]
            rcutoff = np.where(rightseries == rightseries.min())[0][0]
            rightbound = l + rcutoff + 1
            peakbounds[-1].append(rightbound)
        
        if n < len(fmaxiter[:-1]) - 1:
            leftseries = array[l:r]
            leftacc = np.flip(np.minimum.accumulate(np.flip(leftseries)))
            ltrimmer = leftseries <= leftacc
            leftestimate = np.trim_zeros(ltrimmer, trim='f').size
            nl = r - leftestimate
            leftseries = array[nl:r]
            lcutoff = np.where(leftseries == leftseries.min())[0][-1]
            leftbound = nl + lcutoff
            peakbounds.append([leftbound])
    
    peakbounds = np.asarray(peakbounds)
    peakparameters = np.vstack((peakbounds[:,0], fmaxes, peakbounds[:,1])).transpose()
    peakparameters = np.unique(peakparameters, axis=0)
    return peakparameters

#depracated
#def boxcar_noise_replacement(array, boxcarlength=5):
#    narray = array.copy()
#    scans = np.arange(len(array))
#    indvector = np.linspace(scans-boxcarlength, scans+boxcarlength, num=boxcarlength*2+1, axis=1).astype(int)
#    indvector[indvector < 0] = 0
#    indvector[indvector >= len(array) - 1] = len(array) - 1
#    filtermeans = array[indvector].mean(axis=1)
#    meanfiltermeans = filtermeans[indvector].mean(axis=1) #applying the lowest mean that each individual point is involved with
#    #filterboolmap = np.logical_or(array < meanfiltermeans, array > meanfiltermeans)
#    filterboolmap = np.array([[True] * len(array)]).flatten()
#    
#    #baselineforwardmincheck = np.append(array[:-1] == array[1:], False)
#    #baselinebackwardmincheck = np.append(False, array[1:] == array[:-1])
#    #baselinemins = np.where(np.logical_and(baselineforwardmincheck, baselinebackwardmincheck))[0]
#    
#    filterboolmap[baselinemins] = False
#    filterboolmap[0] = False
#    filterboolmap[-1] = False
#    
#    narray[filterboolmap] = meanfiltermeans[filterboolmap]
#    return narray

#depracated
#def boxcar_noise_reduction(array, boxcarlength=5):
#    scans = np.arange(len(array))
#    indvector = np.linspace(scans-boxcarlength, scans+boxcarlength, num=boxcarlength*2+1, axis=1).astype(int)
#    indvector[indvector < 0] = 0
#    indvector[indvector >= len(array) - 1] = len(array) - 1
#    filtermeans = array[indvector].mean(axis=1)
#    minfiltermeans = filtermeans[indvector].min(axis=1) #applying the lowest mean that each individual point is involved with
#    filterboolmap = array < minfiltermeans
#    filteredinds = scans[filterboolmap]
#    noisemask = np.repeat(True, len(array))
#    noisemask[filteredinds] = False
#    noisemask[0] = True
#    noisemask[-1] = True
#    return noisemask

#if:
#less than the lowest mean within distance
#-> replace value with the mean that it's closest in value to, for any mean that it's involved with
#ignore any values that are == to either neighbor
#first/last of array shouldn't be touched I suppose?

#depracated
#def scan_handler(array, **kwargs):
#    noisemask = boxcar_noise_reduction(array, boxcarlength=boxcarlength)
#    noisereducedarray = array[noisemask]
#    maxes = minpoint_reduction(noisereducedarray, mindist=mindist)
#    peakparameters = boundary_finding(maxes, noisereducedarray)
#    
#    peakparameters = peakparameters + (~noisemask).cumsum()[noisemask][peakparameters]
#    peakparameters = peakparameters.tolist()
#    
#    #tightening up just in case because noisemask can still mask 0s that may be more optimal
#    for n in range(len(peakparameters)):
#        l, m, r = peakparameters[n]
#        
#        ltrimdiff = len(array[l:m]) - len(np.trim_zeros(array[l:m], trim='f'))
#        l = l + ltrimdiff - 1
#        peakparameters[n][0] = l
#        
#        rtrimdiff = len(array[m:r+1]) - len(np.trim_zeros(array[m:r+1], trim='b'))
#        r = r - rtrimdiff + 1
#        peakparameters[n][2] = r
#        
#        m = l + array[l:r+1].argmax()
#        peakparameters[n][1] = m
#    
#    peakparameters = np.asarray(peakparameters)
#    return peakparameters, noisemask

def minpoint_reduction(barray, mindist):
    extramaxes = set()
    mask = np.repeat(False, barray.size)
    #narray = array.copy()
    while True:
        narray = barray[~mask]

        forwardmaxcheck = np.append(narray[:-1] > narray[1:], False)
        #backwardmaxcheck = np.append(False, narray[1:] > narray[:-1])
        backwardmaxcheck = np.append(forwardmaxcheck[0], narray[1:] > narray[:-1])
        forwardmaxcheck[-1] = backwardmaxcheck[-1]

        forwardmincheck = np.append(narray[:-1] < narray[1:], False)
        #backwardmincheck = np.append(False, narray[1:] < narray[:-1])
        backwardmincheck = np.append(forwardmincheck[0], narray[1:] < narray[:-1])
        forwardmincheck[-1] = backwardmincheck[-1]

        newmask = np.logical_and(forwardmincheck, backwardmincheck)
        mins = np.where(newmask)[0]
        #mins = np.where(np.logical_and(forwardmincheck, backwardmincheck))[0]
        maxes = np.where(np.logical_and(forwardmaxcheck, backwardmaxcheck))[0]
        #extremadistances = (np.abs(maxes - mins.reshape(-1,1)) < mindist)
        #extremadistances = (np.abs(maxes - maxes.reshape(-1,1)) < mindist)
        #np.fill_diagonal(extremadistances, False)
        extremas = np.sort(np.append(mins, maxes))
        #textremas = extremas + mask.cumsum()[~mask][extremas] #using true distance didn't work out well for large peaks, over-found too many
        #extremadistances = (np.abs(np.diff(extremas)) < mindist) #brings forth incorrect distances
        extremadistances = (np.abs(extremas - extremas[:,None]) < mindist)
        np.fill_diagonal(extremadistances, False)
        
        separatedextremas = extremas[~extremadistances.any(axis=0)]
        if separatedextremas.size > 0:
            maxestomaintain = separatedextremas[np.isin(separatedextremas, maxes)]
            maxestomaintain = (maxestomaintain + mask.cumsum()[~mask][maxestomaintain]).tolist()
            extramaxes.update(maxestomaintain)
            minstomaintain = separatedextremas[np.isin(separatedextremas, mins)]
            newmask[minstomaintain] = False
            if minstomaintain.size > 0:
                mins = np.delete(mins, np.where(mins == minstomaintain[:,None])[1])
        
        adjacentextremas = extremadistances.any()
        if adjacentextremas and mins.size > 0:
            maskinds = np.argwhere(~mask)[np.argwhere(newmask)].flatten()
            mask[maskinds] = True
        else:
            break
    
    if not maxes.size:
        maxes = narray.argmax()
    
    fmaxes = maxes + mask.cumsum()[~mask][maxes]
    fmaxes = np.unique(np.append(fmaxes, list(extramaxes))).astype(int)
    return fmaxes

def boxcar_mean_replacement(array, boxcarlength):
    narray = array.copy()
    scans = np.arange(len(array))
    indvector = np.linspace(scans-boxcarlength, scans+boxcarlength, num=boxcarlength*2+1, axis=1).astype(int)
    indvector[indvector < 0] = 0
    indvector[indvector >= len(array) - 1] = len(array) - 1
    filtermeans = array[indvector].mean(axis=1)
    #minfiltermeans = filtermeans[indvector].min(axis=1) #applying the lowest mean that each individual point is involved with
    #filterboolmap = narray <= minfiltermeans
    filterboolmap = narray <= filtermeans #this seems to work better than when only replacing things < minfiltermeans. I actually have no idea why.
    narray[filterboolmap] = filtermeans[filterboolmap] #this needs to be applied selectively or else the tops of the peaks can even-out when the boxcarlength considered is wider than the peak - which prevents any maxes from being found
    return narray

#def axis_peaks(array, boxcarlength, mindist):
def axis_peaks(array, mindist):
    #boxcararray = boxcar_mean_replacement(array, boxcarlength)
    maxes = minpoint_reduction(array, mindist)
    peakparameters = boundary_finding(maxes, array) #reverted the r + 1 into this function
    peakparameters = peakparameters.tolist()
    
    #peakparameters = peakparameters + (~noisemask).cumsum()[noisemask][peakparameters]
    #peakparameters = peakparameters.tolist()
    
    #finalparameters = []
    #trimming zeros that can come from the boxcar transforms
    #for l, m, r in peakparameters:
    #    while array[l] >= array[l+1]:
    #        l += 1
    #    
    #    while array[r] >= array[r-1]:
    #        r -= 1
    #    r += 1 #setting up for slice indexing
    #    
    #    if r > l:
    #        m = array[l:r].argmax() + l
    #        finalparameters.append([l, m, r])
    
    #return finalparameters, boxcararray
    return peakparameters

def negligible_difference(a):
    while (np.diff(a) == 0).any():
        a[1:][np.diff(a) == 0] += 0.00000001
    return a

def integration_limit_solving(b, m, c, a):
    return ((-1*b) + np.sqrt((b**2) - (4*(0.5*m)*(-0.5*m*(c**2) - (b*c) - a)))) / m

array = np.zeros(np.random.randint(1,distrange))
for p in range(npeaks):
    points = np.random.randint(minpoints, maxpoints)
    #n points plays a role in determining mm
    mv =  np.random.randint(1,meanmax)
    #mm = np.random.randint(1,meanmulmax) * maxpoints
    mm = np.random.randint(1,meanmulmax)
    #window = (signal.windows.gaussian(points, std=mv, sym=False)**mm)*mm
    window = signal.windows.gaussian(points, std=mv, sym=False)*mm
    if peakfill < 1:
        zerofills = np.random.choice(np.arange(len(window)), size=np.random.randint(0, int(window.size*(1-peakfill))))
        window[zerofills] *= 0.00001
    dist = np.random.randint(-distrange,distrange)
    zlength = abs(dist + len(array) + points)
    zpolarity = points + dist
    if dist >= 0:
        zeros = np.zeros(points + dist)
        array = np.hstack((array, zeros, window))
        array[-points:] += window
    else:
        if zlength < 0:
            zeros = np.zeros(abs(zlength))
            array = np.hstack((window, zeros, array))
        elif zpolarity < 0:
            zeros = np.zeros(points - zpolarity)
            array = np.hstack((window, zeros, array))
        elif zpolarity >= 0:
            zeros = np.zeros(points - (points + dist))
            array = np.hstack((array, zeros, window))
        else:
            print('???? what')
            carray = array.copy()
            cwindow = window.copy()
            print(zpolarity, zlength, points, mv, mm, dist)
            print('~~')

randomfactor = np.random.uniform(low=1-noiselevel, high=1+noiselevel, size=len(array))
array = array * randomfactor


#xvals = np.arange(len(array))
#plt.plot(array)
#plt.vlines(xvals[array ==0 ], 0, array.max(), color='black', linewidth=0.5)
#plt.vlines(xvals[mins], 0, array.max(), color='orange', linewidth=0.5)
#plt.vlines(xvals[maxes], 0, array.max(), color='purple', linewidth=0.5)
#plt.vlines(xvals[baselinemins], 0, array.max(), color='green', linewidth=0.5)
#plt.plot(narray, color='red')
#plt.show()


#Acounting for noisy data where zeros, and extremely low values, often appear in the midst of a real peak where 
#scans = np.arange(len(array))
#indvector = np.linspace(scans-boxcarlength, scans+boxcarlength, num=boxcarlength*2+1, axis=1).astype(int)
#indvector[indvector < 0] = 0
#indvector[indvector >= len(array) - 1] = len(array) - 1
#filtermeans = array[indvector].mean(axis=1)
#minfiltermeans = filtermeans[indvector].min(axis=1) #applying the lowest mean that each individual point is involved with
#filterboolmap = array < minfiltermeans
#filteredinds = scans[filterboolmap]
#noisemask = np.repeat(True, len(array))
#noisemask[filteredinds] = False
#
#
#noisereducedarray = array[noisemask]

#peakdiffs = np.diff(noisereducedarray)
#increasing = peakdiffs > 0
#decreasing = peakdiffs < 0
#data = increasing != decreasing
#zeros = increasing == decreasing
#direction = np.diff((increasing.cumsum() - decreasing.cumsum()))
#position = direction.cumsum()


#narray = noisereducedarray.copy()
#extramaxes = set()
#mask = np.repeat(False, narray.size)
#while True:
#    narray = noisereducedarray[~mask]
#
#    forwardmaxcheck = np.append(narray[:-1] > narray[1:], False)
#    backwardmaxcheck = np.append(False, narray[1:] > narray[:-1])
#
#    forwardmincheck = np.append(narray[:-1] < narray[1:], False)
#    backwardmincheck = np.append(False, narray[1:] < narray[:-1])
#
#    newmask = np.logical_and(forwardmincheck, backwardmincheck)
#    mins = np.where(newmask)[0]
#    maxes = np.where(np.logical_and(forwardmaxcheck, backwardmaxcheck))[0]
#    extremadistances = (np.abs(maxes - mins.reshape(-1,1)) < mindist)
#    adjacentextremas = extremadistances.any()
#    
#    maxestoholdonto = maxes[~extremadistances.any(axis=0)]
#    if maxestoholdonto.size > 0:
#        maxestoholdonto = (maxestoholdonto + mask.cumsum()[~mask][maxestoholdonto]).tolist()
#        extramaxes.update(maxestoholdonto)
#    
#    if adjacentextremas:
#        maskinds = np.argwhere(~mask)[np.argwhere(newmask)].flatten()
#        mask[maskinds] = True
#    else:
#        break
#
#fmaxes = maxes + mask.cumsum()[~mask][maxes]
#fmaxes = np.unique(np.append(fmaxes, list(extramaxes)))
#
#fmaxiter = fmaxes.copy().tolist()
#if fmaxes[0] != 0:
#    fmaxiter = np.append(0, fmaxiter)
#if fmaxes[-1] != len(noisereducedarray) - 1:
#    fmaxiter = np.append(fmaxiter, len(noisereducedarray)-1)
#
#peakbounds = []
#for n, l in enumerate(fmaxiter[:-1]):
#    r = fmaxiter[n+1] + 1
#    if n > 0:
#        rightseries = noisereducedarray[l:r]
#        rightacc = np.minimum.accumulate(rightseries)
#        rtrimmer = rightseries <= rightacc
#        rightestimate = np.trim_zeros(rtrimmer, trim='b').size
#        nr = l + rightestimate
#        rightseries = noisereducedarray[l:nr]
#        rcutoff = np.where(rightseries == rightseries.min())[0][0]
#        rightbound = l + rcutoff
#        peakbounds[-1].append(rightbound)
#    
#    if n < len(fmaxiter[:-1]) - 1:
#        leftseries = noisereducedarray[l:r]
#        leftacc = np.flip(np.minimum.accumulate(np.flip(leftseries)))
#        ltrimmer = leftseries <= leftacc
#        leftestimate = np.trim_zeros(ltrimmer, trim='f').size
#        nl = r - leftestimate
#        leftseries = noisereducedarray[nl:r]
#        lcutoff = np.where(leftseries == leftseries.min())[0][-1]
#        leftbound = nl + lcutoff
#        peakbounds.append([leftbound])
#
#peakbounds = np.asarray(peakbounds)
#
#finalpeaks = np.vstack((peakbounds[:,0], fmaxes, peakbounds[:,1])).transpose()
#finalpeaks = np.unique(finalpeaks, axis=0)
#
#peakparameters = finalpeaks + (~noisemask).cumsum()[noisemask][finalpeaks]
#peakparameters = peakparameters.tolist()
#
##tightening up just in case because noisemask can still mask 0s that may be more optimal
#for n in range(len(peakparameters)):
#    l, m, r = peakparameters[n]
#    
#    ltrimdiff = len(array[l:m]) - len(np.trim_zeros(array[l:m], trim='f'))
#    if ltrimdiff > 1:
#        print('left tightened')
#        l = l + ltrimdiff - 1
#        peakparameters[n][0] = l
#
#    rtrimdiff = len(array[m:r+1]) - len(np.trim_zeros(array[m:r+1], trim='b'))
#    if rtrimdiff > 1:
#        print('right tightened')
#        r = r - rtrimdiff + 1
#        peakparameters[n][2] = r
#
#    nm = l + array[l:r+1].argmax()
#    if m != nm:
#        print('max readjusted')
#    peakparameters[n][1] = nm
#
#peakparameters = np.asarray(peakparameters)

#peakparameters, boxcararray = axis_peaks(array, boxcarlength=boxcarlength, mindist=mindist)
peakparameters = axis_peaks(array, mindist=mindist)

peakparameters = np.array(peakparameters)
lefts = peakparameters[:,0]
rights = peakparameters[:,2]
emaxes = peakparameters[:,1]

scans = np.arange(len(array))
fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True)
ax.bar(scans, array, 2, color='white')
#ax.plot(scans, boxcararray, '.', markersize=0.5, color='darkorange')
#plt.show()

#mx = scans[~mask]
#fig, ax = plt.subplots(1, 1, figsize=(6,6), sharex=True)

#ax[0].plot(mx, narray, '.', markersize=0.5)
#ax[0].vlines(mx[maxes], 0, narray[maxes], color='black', linewidth=0.4)

#ax[1].plot(scans, array, '.', markersize=0.5)
ax.vlines(scans[emaxes], 0, array.max(), color='black', linewidth=0.4)
ax.vlines(scans[lefts], 0, array[emaxes], color='cyan', linewidth=0.4, alpha=1)
ax.vlines(scans[rights-1], 0, array[emaxes], color='orange', linewidth=0.4, alpha=1)
ax.hlines(array[emaxes], scans[lefts], scans[rights-1], linewidth=0.4, color='black')
#plt.xlim(2300,2700)
#plt.yscale('log')
plt.show()
print(len(peakparameters), '/', npeaks, 'peaks found')

nsplits = 10

def peak_area_split(peak, xvals, nsplits):
    maxind = peak.argmax()
    
    leftxvals = xvals[:maxind+1]
    rightxvals = xvals[maxind:]
    
    slopes = np.diff(peak) / np.diff(xvals)
    intercepts = peak[:-1] - (xvals[:-1] * slopes)
    
    cumarea = integrate.cumulative_trapezoid(peak, xvals, initial=0)
    areadiffs = np.diff(cumarea, prepend=0)
    
    leftslopes = slopes[:maxind+1]
    rightslopes = slopes[maxind:]
    
    if leftslopes.size > 0:
        leftintercepts = intercepts[:maxind+1]
        leftspots = peak[:maxind+1]
        leftxvals = xvals[:maxind+1]
        leftareadiffs = areadiffs[:maxind+1]
        leftcumarea = leftareadiffs.cumsum()
        leftsplits = np.linspace(leftcumarea[0], leftcumarea[-1], nsplits+1)[1:-1]
        leftinds = (leftcumarea < leftsplits.reshape(-1,1)).sum(axis=1) - 1
        leftslopesofinterest = leftslopes[leftinds]
        leftinterceptsofinterest = leftintercepts[leftinds]
        leftxvalsofinterest = leftxvals[leftinds]
        leftareasofinterest = leftsplits - leftcumarea[leftinds]
        leftsplitxvals = integration_limit_solving(leftinterceptsofinterest, leftslopesofinterest, leftxvalsofinterest, leftareasofinterest)
        leftsplityvals = leftsplitxvals * leftslopesofinterest + leftinterceptsofinterest
        leftsplitxvals = np.hstack((leftxvals[0], leftsplitxvals, leftxvals[-1]))
        leftsplityvals = np.hstack((leftspots[0], leftsplityvals, leftspots[-1]))
    else:
        leftsplitxvals = np.repeat(np.nan, nsplits+1)
        leftsplityvals = np.repeat(np.nan, nsplits+1)
    
    if rightslopes.size > 0:
        rightintercepts = intercepts[maxind:]
        rightspots = peak[maxind:]
        rightxvals = xvals[maxind:]
        rightareadiffs = areadiffs[maxind+1:]
        rightcumarea = np.append(0, rightareadiffs.cumsum())
        rightsplits = np.linspace(rightcumarea[0], rightcumarea[-1], nsplits+1)[1:-1]
        rightinds = (rightcumarea < rightsplits.reshape(-1,1)).sum(axis=1) - 1
        rightslopesofinterest = rightslopes[rightinds]
        rightinterceptsofinterest = rightintercepts[rightinds]
        rightxvalsofinterest = rightxvals[rightinds]
        rightareasofinterest = rightsplits - rightcumarea[rightinds]
        rightsplitxvals = integration_limit_solving(rightinterceptsofinterest, rightslopesofinterest, rightxvalsofinterest, rightareasofinterest)
        rightsplityvals = rightsplitxvals * rightslopesofinterest + rightinterceptsofinterest
        rightsplitxvals = np.hstack((rightxvals[0], rightsplitxvals, rightxvals[-1]))
        rightsplityvals = np.hstack((rightspots[0], rightsplityvals, rightspots[-1]))
    else:
        rightsplitxvals = np.repeat(np.nan, nsplits+1)
        rightsplityvals = np.repeat(np.nan, nsplits+1)
    
    return leftsplitxvals.tolist(), leftsplityvals.tolist(), rightsplitxvals.tolist(), rightsplityvals.tolist()

#params = peakparameters[0]
for params in peakparameters:
    peak = array[params[0]:params[2]]
    xvals = np.arange(len(peak))
    lx, ly, rx, ry = peak_area_split(peak, xvals, nsplits)

#lineval = 0.0001
#nvals = 10
#
#linexvals = np.linspace(xvals - lineval, xvals + lineval, nvals)
#lineyvals = slopes * linexvals[:,:-1] + intercepts
#
#leftlinexvals = linexvals[:,:maxind+1]
#leftlineyvals = lineyvals[:,:maxind+1]
#
#rightlinexvals = linexvals[:,maxind:]
#rightlineyvals = lineyvals[:,maxind:]
#
##plt.plot(linexvals[:,:-1], lineyvals, '--', linewidth=0.3)
#plt.plot(leftlinexvals, leftlineyvals, '--', linewidth=0.3, color='green')
#plt.plot(rightlinexvals[:,:-1], rightlineyvals, '--', linewidth=0.3, color='purple')
#plt.plot(xvals, peak, '.-', linewidth=0.7, color='black', markersize=2)
#plt.vlines(xvals[maxind], 0, peak.max(), linewidth=0.5, color='black')
#plt.show()
