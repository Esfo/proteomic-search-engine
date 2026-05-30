import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from colour import Color
#from pyteomics import mzml
from time import time
import pandas as pd
import operator
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

npeaks = 2
minpoints = 15
maxpoints = 280
noiselevel = 0.8
meanmax = 20
meanmulmax = 1e5
distrange = 20
#peakfill = 0.2

#npeaks = 10
#minpoints = 5
#maxpoints = 20
#noiselevel = 0.9
#meanmax = 3
#meanmulmax = 1e6
#distrange = 5
#peakfill = 1

walkbuffermax = 100

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
            rightbound = l + rcutoff
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
        maxes = narray.argmax() #this seems like an easier way to allow for maxes at the first or last point, in case nothing is found (this case being why)
    
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

def axis_peaks(array, mindist):
    #boxcararray = boxcar_mean_replacement(array, boxcarlength)
    maxes = minpoint_reduction(array, mindist)
    peakparameters = boundary_finding(maxes, array)
    peakparameters = peakparameters.tolist()
    return peakparameters

def group_handler(group, tk, mindist):
    #glen = len(group)
    group = np.array(list(zip(*group)))
    #gwmean = np.average(group[0], weights=group[2])
    #gtimes = np.sort(group[1])
    #gstart = gtimes[0]
    #gend = gtimes[-1]
    #groupinfo = {}
    #groupinfo[tk] = [gstart, gend, gwmean, glen]
    #
    peaks = axis_peaks(group[2], mindist) #this version doesn't have the r + 1 ind b/c of making rp below
    endpeaks = {}
    endpeaks[tk] = []
    for p in peaks:
        rp = group[1,p].tolist()
        wmean = np.average(group[0,p[0]:p[2]+1], weights=group[2,p[0]:p[2]+1])
        rp.append(wmean)
        endpeaks[tk].append(rp)
    #return endpeaks, groupinfo
    return endpeaks

def negligible_difference(a):
    while (np.diff(a) == 0).any():
        a[1:][np.diff(a) == 0] += 0.00000001
    return a

def integration_limit_solving(b, m, c, a):
    return ((-1*b) + np.sqrt((b**2) - (4*(0.5*m)*(-0.5*m*(c**2) - (b*c) - a)))) / m

def max_finding(array):
    forwardmaxcheck = np.append(array[:-1] > array[1:], False)
    #backwardmaxcheck = np.append(False, array[1:] > array[:-1])
    backwardmaxcheck = np.append(forwardmaxcheck[0], array[1:] > array[:-1])
    forwardmaxcheck[-1] = backwardmaxcheck[-1]
    maxes = np.where(forwardmaxcheck & backwardmaxcheck)[0]
    return maxes

def min_finding(array):
    forwardmincheck = np.append(array[:-1] < array[1:], False)
    #backwardmincheck = np.append(False, array[1:] < array[:-1])
    backwardmincheck = np.append(forwardmincheck[0], array[1:] < array[:-1])
    forwardmincheck[-1] = backwardmincheck[-1]
    mins = np.where(forwardmincheck & backwardmincheck)[0]
    return mins

array = np.zeros(np.random.randint(1,distrange))
for p in range(npeaks):
    points = np.random.randint(minpoints, maxpoints)
    #n points plays a role in determining mm
    mv =  np.random.randint(1,meanmax)
    #mm = np.random.randint(1,meanmulmax) * maxpoints
    mm = np.random.randint(1,meanmulmax)
    #window = (signal.windows.gaussian(points, std=mv, sym=False)**mm)*mm
    window = signal.windows.gaussian(points, std=mv, sym=False)*mm
    #if peakfill < 1:
    #    zerofills = np.random.choice(np.arange(len(window)), size=np.random.randint(0, int(window.size*(1-peakfill))))
    #    window[zerofills] *= 0.00001
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

#alternatively set walkbuffer to a static value
walkbuffer = np.random.uniform(0, walkbuffermax)
walkval = np.random.uniform(0, walkbuffer)
forwardwalkingbackground = []
for i in range(array.size):
    walkbuffer = np.random.uniform(0, walkbuffermax)
    walkval += np.random.uniform(-walkbuffer, walkbuffer)
    while walkval < 0:
        walkval += np.random.uniform(0, walkbuffer)
    forwardwalkingbackground.append(walkval)

backwardwalkingbackground = []
for i in range(array.size):
    walkval += np.random.uniform(-walkbuffer, walkbuffer)
    while walkval < 0:
        walkval += np.random.uniform(0, walkbuffer)
    backwardwalkingbackground.append(walkval)

forwardwalkingbackground = np.array(forwardwalkingbackground)
backwardwalkingbackground = np.array(backwardwalkingbackground)[::-1]

walkingbackground = forwardwalkingbackground + backwardwalkingbackground

array += walkingbackground

randomfactor = np.random.uniform(low=1-noiselevel, high=1+noiselevel, size=len(array))
array *= randomfactor


#xvals = np.arange(len(array))
#plt.plot(array)
#plt.vlines(xvals[array ==0 ], 0, array.max(), color='black', linewidth=0.5)
#plt.vlines(xvals[mins], 0, array.max(), color='orange', linewidth=0.5)
#plt.vlines(xvals[maxes], 0, array.max(), color='purple', linewidth=0.5)
#plt.vlines(xvals[baselinemins], 0, array.max(), color='green', linewidth=0.5)
#plt.plot(narray, color='red')
#plt.show()

def positions(tarray):
    peakdiffs = np.diff(tarray)
    increasing = peakdiffs > 0

    nincreasing = np.append(increasing, False)
    ndecreasing = np.zeros(len(nincreasing)).astype(int)
    ndecreasing[nincreasing] += 1
    ndecreasing[~nincreasing] -= 1
    position = ndecreasing.cumsum()
    if position.min() < 0:
        position -= position.min()
    position += 1
    inverseposition = position.max() + 1 - position
    return position, inverseposition


def transf(tarray, passes=1):
    position, inverseposition = positions(tarray)
    #for _ in range(passes):
    meannorm = []
    plen = len(position)
    for n, i in enumerate(inverseposition):
        lp = n - i
        rp = n + i + 1
        if lp < 0:
            lp = 0
        if rp > plen:
            rp = plen
        newmean = np.mean(tarray[lp:rp])
        meannorm.append(newmean)
    meannorm = np.array(meannorm)
    
    mposition, minverseposition = positions(meannorm)
    
    for _ in range(passes):
        sarg = meannorm.argsort()[::-1]
        meannorm = meannorm / mposition
        
        for n, sind in enumerate(meannorm[sarg]):
            reduceinds = sarg[n:][meannorm[sarg][n:] > sind]
            meannorm[reduceinds] = sind
            #tarray[reduceinds] = sind - np.random.uniform(size=len(reduceinds))
    
    return meannorm


def transf(array):
    position, inverseposition = positions(array)
    
    meannorm = []
    plen = len(position)
    for n, i in enumerate(inverseposition):
        lp = n - i
        rp = n + i + 1
        if lp < 0:
            lp = 0
        if rp > plen:
            rp = plen
        newmean = np.mean(array[lp:rp])
        meannorm.append(newmean)
    meannorm = np.array(meannorm)
    
    mposition, minverseposition = positions(meannorm)
    
    sarg = meannorm.argsort()[::-1]
    normarray = meannorm / mposition
    newnorm = normarray.copy()
    
    for n, sind in enumerate(newnorm[sarg]):
        reduceinds = sarg[n:][newnorm[sarg][n:] > sind]
        newnorm[reduceinds] = sind - np.random.uniform(size=len(reduceinds))
        
    return newnorm

def counting_sum_cutoff(array):
    mbool = array <= array[:,None]
    countsums = mbool.sum(axis=0) / array.size
    #mdsum = array.sum()
    #sumcounts = []
    #for mb in mbool:
    #    sumcounts.append(array[mb].sum() / mdsum)
    #sumcounts = np.array(sumcounts)
    #the sumcounts below is generally the same thing, but will differ at values where mbool would have given duplicate entries, doesn't change anything major enough to change the result
    sumcounts = array.cumsum() / array.sum()
    mincomboind = (countsums + sumcounts).argmin()
    mincombo = array[mincomboind]
    #moving average of average of dists under mincombo
    explicitcutoff = array[array <= mincombo].mean()
    return explicitcutoff

peakdiffs = np.diff(array)
increasing = peakdiffs > 0
#decreasing = peakdiffs < 0
#data = increasing != decreasing
#zeros = increasing == decreasing
#direction = np.diff((increasing.cumsum() - decreasing.cumsum()))
#position = direction.cumsum()
#
#if position.min() <= 0:
#    position += abs(position.min()) + 1
##position[position <= 0] = 1
#
#inverseposition = position.max() + 1 - position
#i want to weight a boxcar smoothing based on positions or its inverse

#slices = [slice(-2), slice(1,-1), slice(2,None)]
#prints = [':-2]', '1:-1', '2:']

position, inverseposition = positions(array)

meannorm = []
plen = len(position)
for n, i in enumerate(inverseposition):
    lp = n - i
    rp = n + i + 1
    if lp < 0:
        lp = 0
    if rp > plen:
        rp = plen
    newmean = np.mean(array[lp:rp])
    meannorm.append(newmean)
meannorm = np.array(meannorm)

peakdiffs = np.diff(meannorm)
increasing = peakdiffs > 0

mposition, minverseposition = positions(meannorm)

sarg = meannorm.argsort()[::-1]
normarray = meannorm / mposition
newnorm = normarray.copy()

nlen = len(newnorm)
for n, sind in enumerate(newnorm[sarg]):
    reduceinds = sarg[n:][newnorm[sarg][n:] >= sind]
    #newnorm[reduceinds] = sind
    newnorm[reduceinds] = sind - np.random.uniform(size=len(reduceinds))
    #try:
    #    newnorm[reduceinds] = newnorm[sarg][n+1]
    #except IndexError:
    #    newnorm[reduceinds] = 0

#I want to try this transform in [maybe reverse] order of [inverse or normal] position value on the fly, the transformed values will used for future transformations.
meannorm2 = []
plen = len(position)
for n, i in enumerate(position):
    lp = n - i
    rp = n + i + 1
    if lp < 0:
        lp = 0
    if rp > plen:
        rp = plen
    newmean = np.mean(newnorm[lp:rp])
    meannorm2.append(newmean)
meannorm2 = np.array(meannorm2)

#when boundary finding using the cumulative minimum, you should also keep track of a cumulative maximum since each new minimum
scans = np.arange(len(array))

peakparameters = axis_peaks(array, mindist=mindist)

peakparameters = np.array(peakparameters)
lefts = peakparameters[:,0]
rights = peakparameters[:,2]
emaxes = peakparameters[:,1]
arraymins = min_finding(array)
arraymaxfinds = max_finding(np.delete(array, arraymins))
arraymaxes = np.delete(scans, arraymins)[arraymaxfinds]
udiffs = np.append(0,np.abs(np.diff(array)))
ucutoff = counting_sum_cutoff(udiffs)



#~~~~

#fig, ax = plt.subplots(nrows=4, figsize=(6,8), sharex=True)
#
#ax[0].plot(np.delete(scans, arraymins), np.delete(array, arraymins), '-', linewidth=0.3, color='orangered', alpha=0.3)
#ax[0].plot(scans, array, '.', markersize=0.1, color='whitesmoke')
#ax[0].plot(scans, array, '-', linewidth=0.3, color='aqua')
#ax[0].vlines(scans[emaxes], 0, array[emaxes], color='black', linewidth=0.4)
#ax[0].vlines(scans[lefts], 0, array[emaxes], color='greenyellow', linewidth=0.4, alpha=0.5)
#ax[0].vlines(scans[rights], 0, array[emaxes], color='yellow', linewidth=0.4, alpha=0.5)
#ax[0].vlines(np.where(udiffs >= ucutoff)[0], 0, array.max(), color='black', alpha=0.3, linewidth=0.2)
#ax[0].hlines(array[emaxes], scans[lefts], scans[rights-1], linewidth=0.4, color='black')
#ax[0].set_yscale('log')
#
#
#minpointreducedarray = np.delete(array, arraymins)
#reducedscans = np.delete(scans, arraymins)
#
#peakparameters = axis_peaks(minpointreducedarray, mindist=mindist)
#
#peakparameters = np.array(peakparameters)
#lefts = peakparameters[:,0]
#rights = peakparameters[:,2]
#emaxes = peakparameters[:,1]
#udiffs = np.append(0,np.abs(np.diff(minpointreducedarray)))
#ucutoff = counting_sum_cutoff(udiffs)
#
#ax[1].plot(reducedscans, minpointreducedarray, '-', linewidth=0.4, color='whitesmoke')
#ax[1].vlines(reducedscans[emaxes], 0, minpointreducedarray[emaxes], color='black', linewidth=0.4)
#ax[1].vlines(reducedscans[lefts], 0, minpointreducedarray[emaxes], color='greenyellow', linewidth=0.4, alpha=0.5)
#ax[1].vlines(reducedscans[rights], 0, minpointreducedarray[emaxes], color='yellow', linewidth=0.4, alpha=0.5)
#ax[1].vlines(reducedscans[np.where(udiffs >= ucutoff)[0]], 0, minpointreducedarray.max(), color='black', alpha=0.3, linewidth=0.2)
#ax[1].hlines(minpointreducedarray[emaxes], reducedscans[lefts], reducedscans[rights-1], linewidth=0.4, color='black')
#ax[1].set_yscale('log')
#
#
#
##tx = ax[0].twinx()
##tx.plot(normarray, '-', color='fuchsia', linewidth=0.3)
##tx.set_yscale('log')
#
#firsts = len(peakparameters)
#
#peakparameters = axis_peaks(meannorm, mindist=mindist)
#
#peakparameters = np.array(peakparameters)
#lefts = peakparameters[:,0]
#rights = peakparameters[:,2]
#emaxes = peakparameters[:,1]
#arraymins = min_finding(meannorm)
#arraymaxfinds = max_finding(np.delete(meannorm, arraymins))
#arraymaxes = np.delete(scans, arraymins)[arraymaxfinds]
#udiffs = np.append(0,np.abs(np.diff(meannorm)))
#ucutoff = counting_sum_cutoff(udiffs)
#
#ax[2].plot(np.delete(scans, arraymins), np.delete(meannorm, arraymins), '-', linewidth=0.3, color='orangered', alpha=0.3)
#ax[2].plot(meannorm, '-', linewidth=0.3, color='white')
#ax[2].vlines(scans[emaxes], 0, meannorm[emaxes], color='black', linewidth=0.4)
#ax[2].vlines(scans[lefts], 0, meannorm[emaxes], color='greenyellow', linewidth=0.4, alpha=0.5)
#ax[2].vlines(scans[rights], 0, meannorm[emaxes], color='yellow', linewidth=0.4, alpha=0.5)
#ax[2].vlines(np.where(udiffs >= ucutoff)[0], 0, meannorm.max(), color='black', alpha=0.3, linewidth=0.2)
#ax[2].hlines(meannorm[emaxes], scans[lefts], scans[rights-1], linewidth=0.4, color='black')
#ax[2].set_yscale('log')
#
#seconds = len(peakparameters)
#
#peakparameters = axis_peaks(newnorm, mindist=mindist)
#
#peakparameters = np.array(peakparameters)
#lefts = peakparameters[:,0]
#rights = peakparameters[:,2]
#emaxes = peakparameters[:,1]
#arraymins = min_finding(newnorm)
#arraymaxfinds = max_finding(np.delete(newnorm, arraymins))
#arraymaxes = np.delete(scans, arraymins)[arraymaxfinds]
#udiffs = np.append(0,np.abs(np.diff(newnorm)))
#ucutoff = counting_sum_cutoff(udiffs)
#
#ax[3].plot(np.delete(scans, arraymins), np.delete(newnorm, arraymins), '-', linewidth=0.3, color='orangered', alpha=0.3)
#ax[3].plot(normarray, '-', linewidth=0.5, alpha=0.5, color='fuchsia')
#ax[3].plot(newnorm, '-', linewidth=0.5, alpha=0.5, color='cyan')
#ax[3].vlines(scans[emaxes], 0, newnorm[emaxes], color='black', linewidth=0.4)
#ax[3].vlines(scans[lefts], 0, newnorm[emaxes], color='greenyellow', linewidth=0.4, alpha=0.5)
#ax[3].vlines(scans[rights], 0, newnorm[emaxes], color='yellow', linewidth=0.4, alpha=0.5)
#ax[3].vlines(np.where(udiffs >= ucutoff)[0], 0, newnorm.max(), color='black', alpha=0.3, linewidth=0.2)
#ax[3].hlines(newnorm[emaxes], scans[lefts], scans[rights-1], linewidth=0.4, color='black')
#ax[3].set_yscale('log')

#peakparameters = axis_peaks(meannorm2, mindist=mindist)
#
#peakparameters = np.array(peakparameters)
#lefts = peakparameters[:,0]
#rights = peakparameters[:,2]
#emaxes = peakparameters[:,1]
#arraymins = min_finding(meannorm2)
#arraymaxfinds = max_finding(np.delete(meannorm2, arraymins))
#arraymaxes = np.delete(scans, arraymins)[arraymaxfinds]
#udiffs = np.append(0,np.abs(np.diff(meannorm2)))
#ucutoff = counting_sum_cutoff(udiffs)
#
#ax[3].plot(np.delete(scans, arraymins), np.delete(meannorm2, arraymins), '-', linewidth=0.3, color='orangered', alpha=0.3)
#ax[3].plot(meannorm2, '-', linewidth=0.5, alpha=0.5, color='lime')
#ax[3].vlines(scans[emaxes], 0, newnorm.max(), color='black', linewidth=0.4)
#ax[3].vlines(scans[lefts], 0, newnorm[emaxes], color='greenyellow', linewidth=0.4, alpha=0.5)
#ax[3].vlines(scans[rights], 0, meannorm2[emaxes], color='yellow', linewidth=0.4, alpha=0.5)
#ax[3].vlines(np.where(udiffs >= ucutoff)[0], 0, meannorm2.max(), color='black', alpha=0.3, linewidth=0.2)
#ax[3].hlines(meannorm2[emaxes], scans[lefts], scans[rights-1], linewidth=0.4, color='black')
#ax[3].set_yscale('log')
#
#tx = ax[3].twinx()
#tx.plot(positions(meannorm2)[0], linewidth=0.5, alpha=0.5, color='lightcyan')

#plt.suptitle(p)
#plt.show()
#print(firsts, '/', npeaks, 'peaks found')
#print(seconds, '/', npeaks, 'peaks found')
#print(len(peakparameters), '/', npeaks, 'peaks found')
#
#print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

#This is a nice transformation! really levels out the bottom super well. There can be some things that pop up as little spikes from the noisy areas that originally were some of the lowest actual points across the chromatogram. My solution for this will be to never let anything be higher than its original ranking. Each point can never have a higher value than something ranked 1 higher in array[s].argsort(). It will be forcefully made equal... will this cause flat surfaces? Might be smarter to make it x-1 or something. This should prevent the spikes though.


#So essentially, weighting the boxcars by inverseposition is a great idea. The boundaries always overestimate the actual boundaries of the peak, so that can be used as a legit initial range to find a max within, and to find boundaries within.
#The boundaries of the peak get wider because the use of the mean forces them outward.
#^The only exception to the boundaries thing I've seen is when the boundaries of 2 peaks directly overlap. This can easily be made to trigger a different event where you find the min between two maxes.
#^There could also be, for both this and boundary finding in general, a post-cleanup step to try and clean up where noise might have been allowed within the boundary.

#now do the position normalizing after the meannorm transform to flatten the baselines

#reduce minpoints
#find position -> normalize
#maxes are peaks?
#or maybe find position first?

#usedarray = meannorm
#udiffs = np.append(0,np.abs(np.diff(usedarray)))
#ucutoff = counting_sum_cutoff(udiffs)
#
#fig, ax = plt.subplots(figsize=(6,4))
#ax.plot(usedarray, '-', color='white', linewidth=0.3)
#ax.vlines(np.where(udiffs >= ucutoff)[0], 0, usedarray.max(), color='black', alpha=0.3, linewidth=0.2)
#plt.show()

#arrayma = np.cumsum(array) / (np.arange(array.size) + 1)
#difftoma = np.diff(arrayma)
#diffstobeat = np.cumsum(difftoma) / (np.arange(difftoma.size) + 1)



elists = {} #[fA, fA/r, fB, tA, tA/r, tB, count] #where r is the previous A

efuncs = {}
efuncs[0] = operator.lt
efuncs[1] = operator.gt
efuncs[2] = lambda x, y: y / x #for decreasing
efuncs[3] = lambda x, y: x / y #for increasing

outvals = []
for n, a in enumerate(array.tolist()):
    if n == 0:
        #initialization - will be a necessary step for new lines, this needs to be in the loop because of the 2-step initialization process
        #^maybe keep line length as a single value in a dict to +=1, wouldn't need much memory
        elists[0] = [a, 0, 0, a, 0, 0] #decreasing
        elists[1] = [a, 0, 0, a, 0, 0] #increasing
    else:
        if n == 1:
            if a > elists[0][0]:
                activekey = 1
                opposingkey = 0
                ratio = a / elists[0][0]
                #this offers a flaw if the first two values has some crazy ratio
            else:
                activekey = 0
                opposingkey = 1
                ratio = elists[0][0] / a
            elists[0][1] = ratio
            elists[1][1] = ratio
            elists[0][4] = ratio
            elists[1][4] = ratio
        outval = 0
        switch = False
        if efuncs[activekey](a, elists[activekey][0]): #new extrema found in same polarity
            elists[activekey][2] += 1
            ratio = efuncs[activekey+2](a, elists[activekey][0])
            if ratio > elists[activekey][1]: #new ratio beats old ratio
                outval = 1
                elists[activekey][1] = ratio
                elists[activekey][3] = a #reset templist
                #elists[activekey][5] = 0
                elists[activekey][0] = a
                elists[activekey][2] = 0
                #elists[activekey][4] = ratio
                #switch = True
            #output value here is 1
        #no switching, always collect both, reset the opposite at a rate faster than it grows?
        elif efuncs[opposingkey](a, elists[activekey][0]): #opposing extrema found
            elists[activekey][5] += 1
            ratio = efuncs[activekey+2](elists[activekey][0], a)
            #ratio = efuncs[activekey+2](elists[activekey][3], a) #should it be?
            #if efuncs[opposingkey](a, elists[opposingkey][0]): #new temp beyond old extrema
                #switch = True
            #if ratio > elists[opposingkey][0]:
            if ratio > elists[activekey][4]:
                outval = 1
                switch = True
            #if ratio > elists[opposingkey][1]:
            if (elists[activekey][5]) > elists[activekey][2]: #if temp accumulation > main accumulation
            #    #reset the opposite here
                switch = True
            #elists[activekey][4] = ratio
            #elists[activekey][3] = a
                #elists[opposingkey][1] = ratio
                elists[activekey][3] = a
                elists[activekey][5] = 0
                elists[activekey][0] = a
                #elistsactivegkey][2] = 0
                elists[activekey][4] = ratio
        #elif elists[activekey][4] > elists[activekey][2]: #if temp accumulates more extrema, switch, nah
        #    switch = True
        if outval > 0:
            if switch:
                outvals.append([n, outval, opposingkey, ratio, int(switch)])
            else:
                outvals.append([n, outval, activekey, ratio, int(switch)])
        if switch:
            opposingkey = activekey
            activekey = abs(1-activekey)

increasing = [i for i in outvals if i[2] == 1]
decreasing = [i for i in outvals if i[2] == 0]

incmatch = [i for i in increasing if i[4] == 0]
incswitch = [i for i in increasing if i[4] == 1]

decmatch = [i for i in decreasing if i[4] == 0]
decswitch = [i for i in decreasing if i[4] == 1]

incmatch, incswitch, decmatch, decswitch = np.array(incmatch), np.array(incswitch), np.array(decmatch), np.array(decswitch)

fig, ax = plt.subplots(nrows=1, figsize=(6,2), sharex=True)

ax.plot(np.delete(scans, arraymins), np.delete(array, arraymins), '-', linewidth=0.3, color='orangered', alpha=0.3)
ax.plot(scans, array, '.', markersize=0.1, color='whitesmoke')
ax.plot(scans, array, '-', linewidth=0.3, color='aqua')
ax.vlines(scans[emaxes], 0, array[emaxes], color='black', linewidth=0.4)
ax.vlines(scans[lefts], 0, array[emaxes], color='greenyellow', linewidth=0.4, alpha=0.5)
ax.vlines(scans[rights], 0, array[emaxes], color='yellow', linewidth=0.4, alpha=0.5)
ax.hlines(array[emaxes], scans[lefts], scans[rights-1], linewidth=0.4, color='black')

#ax.vlines(np.where(udiffs >= ucutoff)[0], 0, array.max(), color='black', alpha=0.3, linewidth=0.2)

if decmatch.size > 0:
    ax.plot(scans[decmatch[:,0].astype(int)], array[decmatch[:,0].astype(int)], 's', color='red', alpha=0.5)
if incmatch.size > 0:
    ax.plot(scans[incmatch[:,0].astype(int)], array[incmatch[:,0].astype(int)], 's', color='green', alpha=0.5)
if decswitch.size > 0:
    ax.plot(scans[decswitch[:,0].astype(int)], array[decswitch[:,0].astype(int)], '*', color='red', alpha=0.5)
if incswitch.size > 0:
    ax.plot(scans[incswitch[:,0].astype(int)], array[incswitch[:,0].astype(int)], '*', color='green', alpha=0.5)

ax.set_yscale('log')

#ax[1].plot(scans, arrayma)
#ax[1].plot(scans[1:], difftoma)
#ax[1].plot(scans[1:], diffstobeat)

plt.show()
print(len(outvals))



elists = {} #[fA, fA/r, fB, tA, tA/r, tB, count] #where r is the previous A

efuncs = {}
efuncs[0] = operator.lt
efuncs[1] = operator.gt
efuncs[2] = lambda x, y: y / x #for decreasing
efuncs[3] = lambda x, y: x / y #for increasing

outvals = []
for n, a in enumerate(array.tolist()):
    if n == 0:
        #initialization - will be a necessary step for new lines, this needs to be in the loop because of the 2-step initialization process
        #^maybe keep line length as a single value in a dict to +=1, wouldn't need much memory
        elists[0] = [a, 0, 0, a, 0, 0] #decreasing
        elists[1] = [a, 0, 0, a, 0, 0] #increasing
    else:
        if n == 1:
            if a > elists[0][0]:
                activekey = 1
                opposingkey = 0
                ratio = a / elists[0][0]
                #this offers a flaw if the first two values has some crazy ratio
            else:
                activekey = 0
                opposingkey = 1
                ratio = elists[0][0] / a
            elists[0][1] = ratio
            elists[1][1] = ratio
            elists[0][4] = ratio
            elists[1][4] = ratio
        outval = 0
        switch = False
        if efuncs[activekey](a, elists[activekey][0]): #new extrema found in same polarity
            elists[activekey][2] += 1
            ratio = efuncs[activekey+2](a, elists[activekey][0])
            if ratio > elists[activekey][1]: #new ratio beats old ratio
                outval = 1
                elists[activekey][1] = ratio
                elists[activekey][3] = a #reset templist
                #elists[activekey][5] = 0
                elists[activekey][0] = a
                elists[activekey][2] = 0
                #elists[activekey][4] = ratio
                switch = True
            #output value here is 1
        #no switching, always collect both, reset the opposite at a rate faster than it grows?
        elif efuncs[opposingkey](a, elists[activekey][0]): #opposing extrema found
            elists[activekey][5] += 1
            ratio = efuncs[activekey+2](elists[activekey][0], a)
            #ratio = efuncs[activekey+2](elists[activekey][3], a) #should it be?
            #if efuncs[opposingkey](a, elists[opposingkey][0]): #new temp beyond old extrema
                #switch = True
            #if ratio > elists[opposingkey][0]:
            if ratio > elists[activekey][4]:
                outval = 1
                #switch = True
                elists[activekey][3] = a
                elists[activekey][5] = 0
                elists[activekey][0] = a
                #elistsactivegkey][2] = 0
                elists[activekey][4] = ratio
            #if ratio > elists[opposingkey][1]:
            elif (elists[activekey][5]) > elists[activekey][2]: #if temp accumulation > main accumulation
            #    #reset the opposite here
                switch = True
            #elists[activekey][4] = ratio
            #elists[activekey][3] = a
                #elists[opposingkey][1] = ratio
                #elists[activekey][3] = a
                #elists[activekey][5] = 0
                #elists[activekey][0] = a
                ##elistsactivegkey][2] = 0
                #elists[activekey][4] = ratio
        #elif elists[activekey][4] > elists[activekey][2]: #if temp accumulates more extrema, switch, nah
        #    switch = True
        if outval > 0:
            if switch:
                outvals.append([n, outval, opposingkey, ratio, int(switch)])
            else:
                outvals.append([n, outval, activekey, ratio, int(switch)])
        if switch:
            opposingkey = activekey
            activekey = abs(1-activekey)
            #elists[activekey] = [a, ratio, 0, a, ratio, 0]

increasing = [i for i in outvals if i[2] == 1]
decreasing = [i for i in outvals if i[2] == 0]

incmatch = [i for i in increasing if i[4] == 0]
incswitch = [i for i in increasing if i[4] == 1]

decmatch = [i for i in decreasing if i[4] == 0]
decswitch = [i for i in decreasing if i[4] == 1]

incmatch, incswitch, decmatch, decswitch = np.array(incmatch), np.array(incswitch), np.array(decmatch), np.array(decswitch)

fig, ax = plt.subplots(nrows=1, figsize=(6,2), sharex=True)

ax.plot(np.delete(scans, arraymins), np.delete(array, arraymins), '-', linewidth=0.3, color='orangered', alpha=0.3)
ax.plot(scans, array, '.', markersize=0.1, color='whitesmoke')
ax.plot(scans, array, '-', linewidth=0.3, color='aqua')
ax.vlines(scans[emaxes], 0, array[emaxes], color='black', linewidth=0.4)
ax.vlines(scans[lefts], 0, array[emaxes], color='greenyellow', linewidth=0.4, alpha=0.5)
ax.vlines(scans[rights], 0, array[emaxes], color='yellow', linewidth=0.4, alpha=0.5)
ax.hlines(array[emaxes], scans[lefts], scans[rights-1], linewidth=0.4, color='black')

#ax.vlines(np.where(udiffs >= ucutoff)[0], 0, array.max(), color='black', alpha=0.3, linewidth=0.2)

if decmatch.size > 0:
    ax.plot(scans[decmatch[:,0].astype(int)], array[decmatch[:,0].astype(int)], 's', color='red', alpha=0.5)
if incmatch.size > 0:
    ax.plot(scans[incmatch[:,0].astype(int)], array[incmatch[:,0].astype(int)], 's', color='green', alpha=0.5)
if decswitch.size > 0:
    ax.plot(scans[decswitch[:,0].astype(int)], array[decswitch[:,0].astype(int)], '*', color='red', alpha=0.5)
if incswitch.size > 0:
    ax.plot(scans[incswitch[:,0].astype(int)], array[incswitch[:,0].astype(int)], '*', color='green', alpha=0.5)

ax.set_yscale('log')

#ax[1].plot(scans, arrayma)
#ax[1].plot(scans[1:], difftoma)
#ax[1].plot(scans[1:], diffstobeat)

plt.show()
print(len(outvals))





elists = {} #[fA, fA/r, fB, tA, tA/r, tB, count] #where r is the previous A

efuncs = {}
efuncs[0] = operator.lt
efuncs[1] = operator.gt
efuncs[2] = lambda x, y: y / x #for decreasing
efuncs[3] = lambda x, y: x / y #for increasing

outvals = []
for n, a in enumerate(array.tolist()):
    if n == 0:
        #initialization - will be a necessary step for new lines, this needs to be in the loop because of the 2-step initialization process
        #^maybe keep line length as a single value in a dict to +=1, wouldn't need much memory
        elists[0] = [a, 0, 0, a, 0, 0] #decreasing
        elists[1] = [a, 0, 0, a, 0, 0] #increasing
    else:
        if n == 1:
            if a > elists[0][0]:
                activekey = 1
                opposingkey = 0
                ratio = a / elists[0][0]
                #this offers a flaw if the first two values has some crazy ratio
            else:
                activekey = 0
                opposingkey = 1
                ratio = elists[0][0] / a
            elists[0][1] = ratio
            elists[1][1] = ratio
            elists[0][4] = ratio
            elists[1][4] = ratio
        outval = 0
        switch = False
        if efuncs[activekey](a, elists[activekey][0]): #new extrema found in same polarity
            elists[activekey][2] += 1
            ratio = efuncs[activekey+2](a, elists[activekey][0])
            if ratio > elists[activekey][1]: #new ratio beats old ratio
                outval = 1
                elists[activekey][1] = ratio
                elists[activekey][3] = a #reset templist
                #elists[activekey][5] = 0
                elists[activekey][0] = a
                elists[activekey][2] = 0
                #elists[activekey][4] = ratio
                #switch = True
            #output value here is 1
        #no switching, always collect both, reset the opposite at a rate faster than it grows?
        elif efuncs[opposingkey](a, elists[activekey][0]): #opposing extrema found
            elists[activekey][5] += 1
            ratio = efuncs[activekey+2](elists[activekey][0], a)
            #ratio = efuncs[activekey+2](elists[activekey][3], a) #should it be?
            #if efuncs[opposingkey](a, elists[opposingkey][0]): #new temp beyond old extrema
                #switch = True
            #if ratio > elists[opposingkey][0]:
            if ratio > elists[activekey][4]:
                outval = 1
                #switch = True
                elists[activekey][3] = a
                elists[activekey][5] = 0
                elists[activekey][0] = a
                #elistsactivegkey][2] = 0
                elists[activekey][4] = ratio
            #if ratio > elists[opposingkey][1]:
            elif (elists[activekey][5]) > elists[activekey][2]: #if temp accumulation > main accumulation
            #    #reset the opposite here
                switch = True
            #elists[activekey][4] = ratio
            #elists[activekey][3] = a
                #elists[opposingkey][1] = ratio
                #elists[activekey][3] = a
                #elists[activekey][5] = 0
                #elists[activekey][0] = a
                ##elistsactivegkey][2] = 0
                #elists[activekey][4] = ratio
        #elif elists[activekey][4] > elists[activekey][2]: #if temp accumulates more extrema, switch, nah
        #    switch = True
        if outval > 0:
            if switch:
                outvals.append([n, outval, opposingkey, ratio, int(switch)])
            else:
                outvals.append([n, outval, activekey, ratio, int(switch)])
        if switch:
            opposingkey = activekey
            activekey = abs(1-activekey)
            #elists[activekey] = [a, ratio, 0, a, ratio, 0]

increasing = [i for i in outvals if i[2] == 1]
decreasing = [i for i in outvals if i[2] == 0]

incmatch = [i for i in increasing if i[4] == 0]
incswitch = [i for i in increasing if i[4] == 1]

decmatch = [i for i in decreasing if i[4] == 0]
decswitch = [i for i in decreasing if i[4] == 1]

incmatch, incswitch, decmatch, decswitch = np.array(incmatch), np.array(incswitch), np.array(decmatch), np.array(decswitch)

fig, ax = plt.subplots(nrows=1, figsize=(6,2), sharex=True)

ax.plot(np.delete(scans, arraymins), np.delete(array, arraymins), '-', linewidth=0.3, color='orangered', alpha=0.3)
ax.plot(scans, array, '.', markersize=0.1, color='whitesmoke')
ax.plot(scans, array, '-', linewidth=0.3, color='aqua')
ax.vlines(scans[emaxes], 0, array[emaxes], color='black', linewidth=0.4)
ax.vlines(scans[lefts], 0, array[emaxes], color='greenyellow', linewidth=0.4, alpha=0.5)
ax.vlines(scans[rights], 0, array[emaxes], color='yellow', linewidth=0.4, alpha=0.5)
ax.hlines(array[emaxes], scans[lefts], scans[rights-1], linewidth=0.4, color='black')

#ax.vlines(np.where(udiffs >= ucutoff)[0], 0, array.max(), color='black', alpha=0.3, linewidth=0.2)

if decmatch.size > 0:
    ax.plot(scans[decmatch[:,0].astype(int)], array[decmatch[:,0].astype(int)], 's', color='red', alpha=0.5)
if incmatch.size > 0:
    ax.plot(scans[incmatch[:,0].astype(int)], array[incmatch[:,0].astype(int)], 's', color='green', alpha=0.5)
if decswitch.size > 0:
    ax.plot(scans[decswitch[:,0].astype(int)], array[decswitch[:,0].astype(int)], '*', color='red', alpha=0.5)
if incswitch.size > 0:
    ax.plot(scans[incswitch[:,0].astype(int)], array[incswitch[:,0].astype(int)], '*', color='green', alpha=0.5)

ax.set_yscale('log')

#ax[1].plot(scans, arrayma)
#ax[1].plot(scans[1:], difftoma)
#ax[1].plot(scans[1:], diffstobeat)

plt.show()
print(len(outvals))



elists = {} #[fA, fA/r, fB, tA, tA/r, tB, count] #where r is the previous A

efuncs = {}
efuncs[0] = operator.lt
efuncs[1] = operator.gt
efuncs[2] = lambda x, y: y / x #for decreasing
efuncs[3] = lambda x, y: x / y #for increasing

outvals = []
for n, a in enumerate(array.tolist()):
    if n == 0:
        #initialization - will be a necessary step for new lines, this needs to be in the loop because of the 2-step initialization process
        #^maybe keep line length as a single value in a dict to +=1, wouldn't need much memory
        elists[0] = [a, 0, 0, a, 0, 0] #decreasing
        elists[1] = [a, 0, 0, a, 0, 0] #increasing
    else:
        if n == 1:
            if a > elists[0][0]:
                activekey = 1
                opposingkey = 0
                ratio = a / elists[0][0]
                #this offers a flaw if the first two values has some crazy ratio
            else:
                activekey = 0
                opposingkey = 1
                ratio = elists[0][0] / a
            elists[0][1] = ratio
            elists[1][1] = ratio
            elists[0][4] = ratio
            elists[1][4] = ratio
        outval = 0
        switch = False
        vset = False
        if efuncs[activekey](a, elists[activekey][0]): #new extrema found in same polarity
            elists[activekey][2] += 1
            ratio = efuncs[activekey+2](a, elists[activekey][0])
            if ratio > elists[activekey][1]: #new ratio beats old ratio
                outval = 1
                elists[activekey][1] = ratio
                #elists[activekey][3] = a #reset templist
                #elists[activekey][5] = 0
                elists[activekey][0] = a
                elists[activekey][2] = 0
                #elists[activekey][4] = ratio
                #switch = True
            #output value here is 1
        #no switching, always collect both, reset the opposite at a rate faster than it grows?
        elif efuncs[opposingkey](a, elists[activekey][0]): #opposing extrema found
            elists[activekey][5] += 1
            ratio = efuncs[activekey+2](elists[activekey][0], a)
            #ratio = efuncs[activekey+2](elists[activekey][3], a) #should it be?
            #if efuncs[opposingkey](a, elists[opposingkey][0]): #new temp beyond old extrema
                #switch = True
            #if ratio > elists[opposingkey][0]:
            if ratio > elists[activekey][4]:
                outval = 1
                #switch = True
                vset = True
                #elists[activekey][3] = a
                #elists[activekey][5] = 0
                ##elists[activekey][0] = a
                ##elistsactivegkey][2] = 0
                #elists[activekey][4] = ratio
            #if ratio > elists[opposingkey][1]:
            if (elists[activekey][5]) > elists[activekey][2]: #if temp accumulation > main accumulation
            #    #reset the opposite here
                #outval = 1
                switch = True
                vset = True
            #elists[activekey][4] = ratio
            #elists[activekey][3] = a
                #elists[opposingkey][1] = ratio
                #elists[activekey][3] = a
                #elists[activekey][5] = 0
                ##elists[activekey][0] = a
                ###elistsactivegkey][2] = 0
                #elists[activekey][4] = ratio
            if vset:
                elists[activekey][3] = a
                elists[activekey][5] = 0
                elists[activekey][0] = a
                ##elistsactivegkey][2] = 0
                elists[activekey][4] = ratio
        #elif elists[activekey][4] > elists[activekey][2]: #if temp accumulates more extrema, switch, nah
        #    switch = True
        if outval > 0:
            if switch:
                outvals.append([n, outval, opposingkey, ratio, int(switch)])
            else:
                outvals.append([n, outval, activekey, ratio, int(switch)])
        if switch:
            opposingkey = activekey
            activekey = abs(1-activekey)
            #elists[activekey] = [a, ratio, 0, a, ratio, 0]

increasing = [i for i in outvals if i[2] == 1]
decreasing = [i for i in outvals if i[2] == 0]

incmatch = [i for i in increasing if i[4] == 0]
incswitch = [i for i in increasing if i[4] == 1]

decmatch = [i for i in decreasing if i[4] == 0]
decswitch = [i for i in decreasing if i[4] == 1]

incmatch, incswitch, decmatch, decswitch = np.array(incmatch), np.array(incswitch), np.array(decmatch), np.array(decswitch)

fig, ax = plt.subplots(nrows=1, figsize=(6,2), sharex=True)

ax.plot(np.delete(scans, arraymins), np.delete(array, arraymins), '-', linewidth=0.3, color='orangered', alpha=0.3)
ax.plot(scans, array, '.', markersize=0.1, color='whitesmoke')
ax.plot(scans, array, '-', linewidth=0.3, color='aqua')
ax.vlines(scans[emaxes], 0, array[emaxes], color='black', linewidth=0.4)
ax.vlines(scans[lefts], 0, array[emaxes], color='greenyellow', linewidth=0.4, alpha=0.5)
ax.vlines(scans[rights], 0, array[emaxes], color='yellow', linewidth=0.4, alpha=0.5)
ax.hlines(array[emaxes], scans[lefts], scans[rights-1], linewidth=0.4, color='black')

#ax.vlines(np.where(udiffs >= ucutoff)[0], 0, array.max(), color='black', alpha=0.3, linewidth=0.2)

if decmatch.size > 0:
    ax.plot(scans[decmatch[:,0].astype(int)], array[decmatch[:,0].astype(int)], 's', color='red', alpha=0.5)
if incmatch.size > 0:
    ax.plot(scans[incmatch[:,0].astype(int)], array[incmatch[:,0].astype(int)], 's', color='green', alpha=0.5)
if decswitch.size > 0:
    ax.plot(scans[decswitch[:,0].astype(int)], array[decswitch[:,0].astype(int)], '*', color='red', alpha=0.5)
if incswitch.size > 0:
    ax.plot(scans[incswitch[:,0].astype(int)], array[incswitch[:,0].astype(int)], '*', color='green', alpha=0.5)

ax.set_yscale('log')

#ax[1].plot(scans, arrayma)
#ax[1].plot(scans[1:], difftoma)
#ax[1].plot(scans[1:], diffstobeat)

plt.show()
print(len(outvals))



signaltracker = {}
signaltracker[1] = [0, 0, 0] #increasing
signaltracker[0] = [0, 0, 0] #decreasing

efuncs = {}
efuncs[1] = operator.gt
efuncs[0] = operator.lt
efuncs[2] = lambda x, y: y / x #for decreasing
efuncs[3] = lambda x, y: x / y #for increasing

adders = {}
adders[1] = 1
adders[0] = -1

#[cummin, n since reset, cummax, n since reset]
#only reset the opposing when the main stops accumulating for deadsignal
#deadsignal accumulates, and only subracts if a max is found
#^maybe is should subtract the amount since the last max, rather than 1 or the whole thing
#and maybe the points to find relevant, within the scope of the moving model, would be when the deadsignal count is negative!
#if the count is negative, only a -1 for every datapoint, if it's positive, you can subtract the datapoint distance since the last extrema? But subtracting multiple points specifically should be maxed out to prevent this causing the number to go negative.. and it also might not be so great if the distance is pretty much equal to deadsignal.
#and perhaps the opposing recording process can happen early if a larger than previously seen difference occurs? This would also be immediately thrown away if there's a new matching extrema.

#as long as the max values don't go below the original temp minimum in the main list, the reset temp values should only be 

outvals = []
for n, a in enumerate(array.tolist()):
    if n == 0:
        heldvalue = a
    else:
        if n == 1:
            if a > heldvalue:
                direction = 1
            else:
                direction = 0
            signaltracker[direction][0] = a
            signaltracker[direction][1] += a
        else:
            switch = False
            if direction > 0:
                if a > signaltracker[direction][0]: #new extrema found in same polarity
                    signaltracker[direction][0] = a
                    signaltracker[direction][1] += a
                    if signaltracker[direction][1] > a:
                    #if efuncs[direction](-1*signaltracker[direction][1], a):
                        signaltracker[direction][2] += 1
                        #signaltracker[direction][2] += adders[direction]
                        outvals.append([n, signaltracker[direction][1], direction, signaltracker[direction][2]])
                    elif signaltracker[direction][2] > 0:
                        signaltracker[direction][2] -= 1
                        #signaltracker[direction][2] -= adders[direction]
                else:
                    switch = True
            else:
                if a < signaltracker[direction][0]: #extrema not beat
                    signaltracker[direction][0] = a
                    signaltracker[direction][1] += a
                    if signaltracker[direction][1] > a:
                    #if efuncs[direction](-1*signaltracker[direction][1], a):
                        signaltracker[direction][2] += 1
                        #signaltracker[direction][2] += adders[direction]
                        outvals.append([n, signaltracker[direction][1], direction, signaltracker[direction][2]])
                    elif signaltracker[direction][2] > 0:
                        signaltracker[direction][2] -= 1
                        #signaltracker[direction][2] -= adders[direction]
                else:
                    switch = True
                    #else:
                        #signaltracker[direction][2] -= adders[direction]
                #elif efuncs[~direction](a, signaltracker[~direction][0]): #opposing extrema found
                    #if difference is notable? If I track differences, not supported yet
                #new opposing beyond old extrema?
            if switch:
                signaltracker[direction][1] -= a
                if signaltracker[direction][1] < 0:
                    direction = abs(1-direction)
                    signaltracker[direction][0] = a
                    #signaltracker[direction][1] = 0
                    signaltracker[direction][1] += a
                    signaltracker[direction][2] = 0
                    #outvals.append([n, signaltracker[direction][1], direction, signaltracker[direction][2]])
                #outvals.append([n, signaltracker[direction][1], direction])

fig, ax = plt.subplots(nrows=1, figsize=(6,2), sharex=True)

#ax.plot(np.delete(scans, arraymins), np.delete(array, arraymins), '-', linewidth=0.3, color='orangered', alpha=0.3)
ax.plot(scans, array, '.', markersize=0.1, color='whitesmoke')
ax.plot(scans, array, '-', linewidth=0.3, color='aqua')
#ax.vlines(scans[emaxes], 0, array[emaxes], color='black', linewidth=0.4)
#ax.vlines(scans[lefts], 0, array[emaxes], color='greenyellow', linewidth=0.4, alpha=0.5)
#ax.vlines(scans[rights], 0, array[emaxes], color='yellow', linewidth=0.4, alpha=0.5)
#ax.hlines(array[emaxes], scans[lefts], scans[rights-1], linewidth=0.4, color='black')

#ax.vlines(np.where(udiffs >= ucutoff)[0], 0, array.max(), color='black', alpha=0.3, linewidth=0.2)

vcols = ['green' if i[2] == 1 else 'red' for i in outvals]
outvals = np.array(outvals)

tx = ax.twinx()
tx.scatter(outvals[:,0], outvals[:,3], c=vcols, marker='.')

ax.set_yscale('log')
#tx.set_yscale('symlog')

#ax[1].plot(scans, arrayma)
#ax[1].plot(scans[1:], difftoma)
#ax[1].plot(scans[1:], diffstobeat)

plt.show()
print(len(outvals))




signaltracker = {}
signaltracker[1] = [0, 0] #increasing
signaltracker[0] = [0, 0] #decreasing
signalcount = 0

efuncs = {}
efuncs[1] = operator.gt
efuncs[0] = operator.lt
efuncs[2] = lambda x, y: y / x #for decreasing
efuncs[3] = lambda x, y: x / y #for increasing

adders = {}
adders[1] = 1
adders[0] = -1

#[cummin, n since reset, cummax, n since reset]
#only reset the opposing when the main stops accumulating for deadsignal
#deadsignal accumulates, and only subracts if a max is found
#^maybe is should subtract the amount since the last max, rather than 1 or the whole thing
#and maybe the points to find relevant, within the scope of the moving model, would be when the deadsignal count is negative!
#if the count is negative, only a -1 for every datapoint, if it's positive, you can subtract the datapoint distance since the last extrema? But subtracting multiple points specifically should be maxed out to prevent this causing the number to go negative.. and it also might not be so great if the distance is pretty much equal to deadsignal.
#and perhaps the opposing recording process can happen early if a larger than previously seen difference occurs? This would also be immediately thrown away if there's a new matching extrema.

#as long as the max values don't go below the original temp minimum in the main list, the reset temp values should only be

outvals = []
for n, a in enumerate(array.tolist()):
    if n == 0:
        heldvalue = a
    else:
        if n == 1:
            if a > heldvalue:
                direction = 1
            else:
                direction = 0
            signaltracker[direction][0] = a
            signaltracker[direction][1] += a
        else:
            switch = False
            if direction > 0:
                if a > signaltracker[direction][0]: #new extrema found in same polarity
                    signaltracker[direction][0] = a
                    signaltracker[direction][1] += a
                    if signaltracker[direction][1] > a:
                    #if efuncs[direction](-1*signaltracker[direction][1], a):
                        signalcount += 1
                        #signalcount += adders[direction]
                        outvals.append([n, signaltracker[direction][1], direction, signalcount])
                    elif signalcount > 0:
                        signalcount -= 1
                        #signalcount -= adders[direction]
                else:
                    switch = True
            else:
                if a < signaltracker[direction][0]: #extrema not beat
                    signaltracker[direction][0] = a
                    signaltracker[direction][1] += a
                    if signaltracker[direction][1] > a:
                    #if efuncs[direction](-1*signaltracker[direction][1], a):
                        signalcount -= 1
                        #signalcount += adders[direction]
                        outvals.append([n, signaltracker[direction][1], direction, signalcount])
                    elif signalcount > 0:
                        signalcount -= 1
                        #signalcount -= adders[direction]
                else:
                    switch = True
                    #else:
                        #signalcount -= adders[direction]
                #elif efuncs[~direction](a, signaltracker[~direction][0]): #opposing extrema found
                    #if difference is notable? If I track differences, not supported yet
                #new opposing beyond old extrema?
            if switch:
                signaltracker[direction][1] -= a
                if signaltracker[direction][1] < 0:
                    direction = abs(1-direction)
                    signaltracker[direction][0] = a
                    #signaltracker[direction][1] = 0
                    signaltracker[direction][1] += a
                    #signalcount = 0
                    #outvals.append([n, signaltracker[direction][1], direction, signalcount])
                #outvals.append([n, signaltracker[direction][1], direction])

fig, ax = plt.subplots(nrows=1, figsize=(6,2), sharex=True)

#ax.plot(np.delete(scans, arraymins), np.delete(array, arraymins), '-', linewidth=0.3, color='orangered', alpha=0.3)
ax.plot(scans, array, '.', markersize=0.1, color='whitesmoke')
ax.plot(scans, array, '-', linewidth=0.3, color='aqua')
#ax.vlines(scans[emaxes], 0, array[emaxes], color='black', linewidth=0.4)
#ax.vlines(scans[lefts], 0, array[emaxes], color='greenyellow', linewidth=0.4, alpha=0.5)
#ax.vlines(scans[rights], 0, array[emaxes], color='yellow', linewidth=0.4, alpha=0.5)
#ax.hlines(array[emaxes], scans[lefts], scans[rights-1], linewidth=0.4, color='black')

#ax.vlines(np.where(udiffs >= ucutoff)[0], 0, array.max(), color='black', alpha=0.3, linewidth=0.2)

vcols = ['green' if i[2] == 1 else 'red' for i in outvals]
outvals = np.array(outvals)

tx = ax.twinx()
tx.scatter(outvals[:,0], outvals[:,3], c=vcols, marker='.')

ax.set_yscale('log')
#tx.set_yscale('symlog')

#ax[1].plot(scans, arrayma)
#ax[1].plot(scans[1:], difftoma)
#ax[1].plot(scans[1:], diffstobeat)

plt.show()
print(len(outvals))


#under increasing movement:
#if it's lower, is it higher than the last minimum in the templist? +1 if yes -> record vallue
#If it's higher, is it higher than the last maximum? +1 if yes -> record value
#don't record the value if it's not a yes, therefore no +1
#and vice-versa for decreasing
#^except that a -1 gets recorded when decreasing
#^HOWEVER, the shift is negative from decreasing, but the value, 1, is still what's being contributed to total score growth (for all isotopomers)
#^actually, for the value being contributed, it can only be a 2, it becomes a 2 by the minimums and maximums both going in the correct direction. If a 1 is output, then it doesn't add to total growth -> noise should be ignored from this process happening.
#^so the 2 can be a temporary 2 and turn to a 1 if a higher low value is succeded by a lower low value. The opportunity gets reset if a new higher maximum is found still, and the 'too-low' lower value from before is the lower value to beat I suppose.
#IF no minimum value occurs between two maxes, then the 2 is maintained.
#IF a non-new-max is found after a low, the 2 is dropped temporarily? This does a -1 to the growth value, but the process following this gets more complicated:
#   - the 2 stays live because the non-growth increasing value isn't 0
#   - If it's being determined that the direction is switching, then the decreasing values can contribute to the growth value, but if the switch doesn't stay immediately permenant, then the next few switched values that WOULD gain a 2 don't gain that 2, the processes have to cancel each other out. So some kind of fake-gain cache is set up -> but this is ALSO a 2-dimensional process [for each direction].
#   - this fake-gain cache is something that also has to be overcome [maybe] if you want to determine something is a peak, and to have an actual growth value, otherwise the noise wouldn't cancel itself out so nicely I think.
#^this total score growth gets -1 if the mins/maxes don't align. Floor is 0 for this value.
#an isotopomer is only 'searching' for a partner when a 2 hits, it can still be found by others though I suppose. Maybe I should allow it to also search bordering timepoints?
#if the value is continuing to appear WITHIN the last 2 min/maxes determines latency, the latency can feed off of the growth value?
#continuing timepoints can keep their library index in the pool of useable indices as long as they keep continuing. As new things, that started after/at the same time as something else, find more matches, they officially survive, and their index can be looked up in the future. But only things that survive in the future can be looked up in the past.

#this is essentially a game of peripherals, what peripherals to keep track of matters.

#on the fly, you should be noting that:
#IF a value goes up, it's GOING to be a maximum, until proven otherwise
#and vice-versa
#new maximums are always found after new minimums, and vice-versa
#the starting basis of each peak is to find the absolute max
#the ending basis is to decline from it, so the decline process technically starts at every new maximum, and so does the new max process.
#once the apex is found, the quest is for the next absolute minimum, 

#find absolute maximums, and let the switch degrade twice as fast as they were gathered. So each datapoint gets you a 1/2 datapoint extension in deadsignal. Round up on the first match, then ignore on the second.
#the 2x degradation idea works, but I need to treat line initiation differently than line continuation, like the other moving model.
#For initilization: you collect distant maxes until the number of maxes you have can fill in the gaps between all the individual maxes.
#example:
#0, 3(m), 2, 3, 4(m), 2, 1, 5(m) can become a group because there's 3 maxes and 3 things between points at most. If there was like 20 points between the first and second, that one would just be dropped if the front ones could make it. Adjacent maxes also count together. So if it's a double increase instead of a dip, that's +2 total baby!
#maybe ditch the 2x degradation, from ^then on, if the points can't find anything new in the time it takes 

#the pool: all the potential isotopomers looking for a place to hang out. These ones can also search for lines that...

maxval = 0 #the cumulative max
deadspace = 0 #n of adjacent non-max values
spacepool = [] #ordered
peaksurvival = False #initialization vs collection
peakcoords = [0, 0] #[first, last] indices

outvals = []
for n, a in enumerate(array.tolist()):
    if a > maxval:
        maxval = a
        spacepool.append(deadspace)
        deadspace = 0
        peakcoords[1] = n
        if not peaksurvival and n != peakcoords[0]:
            #initialization stage
            plen = len(spacepool)
            if plen > max(spacepool):
                peaksurvival = True
            else:
                #can it work by excluding the earlier ones?
                frontpeak = [n for n in range(len(spacepool)) if max(spacepool[n:]) < plen - n]
                if frontpeak:
                    cropind = frontpeak[0]
                    #there may be some initial points that don't make it in here, but it doesn't matter because this is an on-the-fly model, the isotopomer switch can only be flipped when the model is absolutely sure.
                    peakcoords[0] += sum(i if i > 0 else 1 for i in spacepool[:cropind])
                    spacepool = spacepool[cropind:]
                    peaksurvival = True

    else:
        deadspace += 1
    if peaksurvival:
        if deadspace > len(spacepool):
            #kill the peak
            maxval = a
            maxspace = 0
            deadspace = 0
            spacepool = []
            peaksurvival = False
            peakcoords = [n, n]
        else:
            #append uniqueid to a list where isotopomer deconvolution is processed
            #or, print an outlist value to visualize
            outvals.append(n)
            pass

fig, ax = plt.subplots(nrows=1, figsize=(6,2), sharex=True)

ax.plot(scans, array, '-', linewidth=0.1, color='whitesmoke')
ax.plot(scans, array, '.', markersize=0.2, color='whitesmoke')

outvals = np.array(outvals)

ax.plot(outvals, array[outvals], '.', color='green')

ax.set_yscale('log')

plt.show()
print(len(outvals))
