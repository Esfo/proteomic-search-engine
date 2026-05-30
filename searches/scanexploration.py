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

mzmlfile = '/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
msrun = mzml.MzML(mzmlfile, dtype=np.float64)

basefolder = '/'.join((mzmlfile.split('/')[:-2]))
basefile = mzmlfile.split('/')[-1].split('.mzML')[0]

processinglocation = '/'.join((basefolder, 'fileprocessing', basefile))
librarylocation = '/home/sfo/data/proteomics/fastas/search-db'
proteome = 'Human_Homo_sapien'
nprocs = 8
subisotopomericdepth = 0.8
proton = 1.007276554940804
dividingthreshold = 0.1

environment_partial = partial(lmdb.Environment, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=9999, max_spare_txns=1, lock=True)

linesofscansfile = '/'.join((processinglocation, 'linesofscans.pickle'))
with open(linesofscansfile, 'rb') as pick:
    linesofscans = pickle.load(pick)
#linesofscans = defaultdict(list) #ms2 scan index: [lineuids within window]

scansoflinesfile = '/'.join((processinglocation, 'scansoflines.pickle'))
with open(scansoflinesfile, 'rb') as pick:
    scansoflines = pickle.load(pick)
#scansoflines = defaultdict(list) #lineuid: [ms2 scan indices]

analytefile = '/'.join((processinglocation, 'analytefactors.pickle'))
with open(analytefile, 'rb') as pick:
    analytekeys, analytedistributions, analytesbydistribution, distributionsoflines, linesofdistributions, linesofanalytes = pickle.load(pick)
#analytekeys = defaultdict(dict) #analyte id: distid: charge, this can be one or multiple distributions based on whether it has multiple charge states
#analytedistributions = {} #analyte id: [[weighted means [via intensity] across isotopomers from every charge state if there are multiple], [AUC of merged isotopomers]]
#analytesbydistribution = {} #distid: analyte id
#distributionsoflines: lineuid: distid
#linesofdistributions: distid: [lineuids ordered by mass]

lineintensitiesofscansfile = '/'.join((processinglocation, 'lineintensitiesofscans.pickle'))
with open(lineintensitiesofscansfile, 'rb') as pick:
    lineintensitiesofscans = pickle.load(pick)
#lineintensitiesofscans = defaultdict(lambda: defaultdict(dict)) #scan: line: raw area of ms1 line, average of 2 flanking points

linepercentagesofscansfile = '/'.join((processinglocation, 'linepercentagesofscans.pickle'))
with open(linepercentagesofscansfile, 'rb') as pick:
    linepercentagesofscans = pickle.load(pick)
#linepercentagesofscans = defaultdict(lambda: defaultdict(dict)) #scan: line: % of scan intensity input

def intersection_merge(mergable_items):
    sn = 0
    itemgroups = defaultdict(set) #group: [members]
    groupsofitems = {} #member: group
    for items in mergable_items:
        locs = set()
        for i in items:
            if i in groupsofitems:
                locs.add(groupsofitems[i])
        if locs:
            joiner = min(locs)
            if len(locs) > 1:
                for oldlocs in locs.difference([joiner]):
                    for ol in itemgroups[oldlocs]:
                        groupsofitems[ol] = joiner
                    itemgroups[joiner].update(itemgroups.pop(oldlocs))
        else:
            joiner = sn
            sn += 1
        itemgroups[joiner].update(items)
        for i in items:
            groupsofitems[i] = joiner
    return list(itemgroups.values())

distsofscans = {}
for scan, lines in linesofscans.items():
    dists = set(distributionsoflines[i] for i in lines)
    distsofscans[scan] = tuple(dists)

mergeddistributions = list(map(tuple, intersection_merge(distsofscans.values())))

linesbyscangroup = defaultdict(set) #scangroup: [lines]
scansbyscangroup = defaultdict(set) #scangroup: [scans]
mergedscans = []
for n, md in enumerate(mergeddistributions):
    allscans = set()
    for dist in md:
        lines = linesofdistributions[dist]
        for line in lines:
            if line in scansoflines:
                linescans = scansoflines[line]
                linesbyscangroup[n].add(line)
                scansbyscangroup[n].update(linescans)
                allscans.update(linescans)
    mergedscans.append(tuple(sorted(allscans))) #this sorting was for troubleshooting the comparison of this to the old method, its not necessary

def radius_neighbors(baselist, flylist, ppmmod):
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
        if submatches:
            matches[fn] = submatches
    return matches

#find a scangroup with 2 lines from the same dist
#and also a group with 2 lines from different dists
#a least 2 scans in both

#2 lines different dists:
#see if you can differentiate their change in ms% via their fragment intensity changes

#2 lines from the same dist:
#do the fragment ions here have more nearest neighbor matches than 2 lines from different dists?

ppmtolerance = 25

#fishing:
t = 0
for n, scangroup in enumerate(mergedscans):
    dists = set([distributionsoflines[i] for i in linesbyscangroup[n]])
    lines = linesbyscangroup[n]
    if len(lines) == len(scangroup) == len(dists) == 2:
    #if len(lines) == len(scangroup) == 2 and len(dists) == 1:
        lscan = scangroup[0]
        lms = msrun[lscan]
        rscan = scangroup[1]
        rms = msrun[rscan]
        tree = spatial.KDTree(lms['m/z array'][:,None])
        radius = rms['m/z array'] / 1000000 * ppmtolerance
        matches = tree.query_ball_point(rms['m/z array'][:,None], r=radius)
        nmatches = sum(1 for i in matches.tolist() if i)
        #match index: [lintensity, rintensity, ratio] #only works if the number of matches == 1 when it exists
        #when assessing this for more than 2 scans:
        #i think you'll want to sort all ms2 scans involved by rt
        #^then normalize l/r intensity ratios to use for the same lines across scans
        #i also want to compare the total intensity of either scan to the line %s
        matchratios = []
        for mn, mm in enumerate(matches.tolist()):
            rintensity = rms['intensity array'][mn]
            for m in mm:
                lintensity = lms['intensity array'][m]
                ratio = lintensity / rintensity
                matchratios.append([lintensity, rintensity, ratio])
                
        #correlate %s of the intensities of overlaps from scan to scan
        if t > 15:
            for i in sorted(matchratios, key=lambda x: sum([x[0], x[1]])):
                print(i)
            print('~')
            print(linepercentagesofscans[lscan])
            print(linepercentagesofscans[rscan])
            print(lineintensitiesofscans[lscan])
            print(lineintensitiesofscans[rscan])
            print('~')
            break
        t += 1

#the ratio as an initial indicator of likelihood of being worthy of entropy
#percent of the total intensity of the scan as a weight of how much entropy
#these two things together are going to define my intensity metric
#because some ratios are good with shit intensity
#some intensities are so far off when the ratio should be 1 and thats an indicator of something being bad
#but some intensities are far off when the ratio is 2x or something, which would be more of a "good" indicator
#i need to set an expectation for allowing these indices to be entropy catchers
#and it needs to reverse the ratio of anything < 1, while also changing the game of distributions based around 1 rather than larger than it

#this data is impossible, it says nothing
#i need to make the existing entrop engine less combinatoric-heavy, i think its possible
#doing this by intensity should only be done for bigger verifiable cases
#it would be dumb to use intensity as the first go-to
#it should be the secondary layer, sure
#like if you can see a clear increase and/or decline of a signal across multiple scans
#and that could ONLY match one thing, then go for it, label that as higher priority logic
#but these instances should be fished out, most things can't be accepted into this because the data is just shit at indicating anything in this regard
#as the main idea is to still NOT be WRONG

#aka
#the intensity inferences should be supplementary
#reduce the complexity of the existing entropy engine
#those are pretty much the best guard rails you're going to get
#then work on organizing the scangroups that have more scans and lines in their respective linear time scales with overlapping ion indices and fish for intensity-based entropy




#the total intensity of a fragment scan is but a fraction of the input ms1 intensity
#1/10ish
#^would be interesting to see the distribution of these across all scans

#if the intensities show that it makes ~sense
#then its good to work with
#if not, AND if the intensities are below what sensible things are available -> noise connection, no need to assign value here for the entropy process
#exact ratios might be useless
#i need to inspect more complicated scenarios to see how simultaneous ms1 increase and decrease to a scan is handled with overlapping masses across scans
#^i need some quantitative assessment of whether the decrease could fit within the opposite of the increase
#and it most cases it might be possible -> an easy indicator of what could be an allowable double fragment match for 2 different sequences!




#scaninfo = [] #[ms1 intensity input, ms2 intensity sum, ratio of the 2]
#for scan, ld in lineintensitiesofscans.items():
#    ms1sum = sum(ld.values())
#    scanintensities = msrun[scan]['intensity array']
#    precursorinfo = msrun[scan]['precursorList']['precursor'][0]
#    selectionwindow = precursorinfo['isolationWindow']
#    precmass = selectionwindow['isolation window target m/z'].real
#    scansum = scanintensities.sum()
#    maxpercent = scanintensities.max() / scansum
#    ratio = scansum / ms1sum
#    npoints = len(scanintensities)
#    output = [ms1sum, scansum, ratio, maxpercent, npoints, precmass]
#    scaninfo.append(output)
#scaninfo = np.array(scaninfo)
#
#nbins = np.sqrt(len(scaninfo)).astype(int)
#
#plt.hist(scaninfo[:,0], bins=nbins)
#plt.show()
#
#plt.hist(scaninfo[:,1], bins=nbins)
#plt.show()
#
#plt.hist(scaninfo[:,2], bins=nbins)
#plt.show()
#
#plt.hist(scaninfo[:,3], bins=nbins)
#plt.show()
#
#plt.hist(scaninfo[:,4], bins=nbins)
#plt.show()
#
#plt.hist(scaninfo[:,5], bins=nbins)
#plt.show()
#
#plt.plot(scaninfo[:,0], scaninfo[:,1], '.')
#plt.show()
#
#plt.plot(scaninfo[:,0], scaninfo[:,2], '.')
##plt.yscale('log')
##plt.xscale('log')
#plt.show()
#
#plt.plot(scaninfo[:,0], scaninfo[:,4], '.')
#plt.show()
#
#plt.plot(scaninfo[:,0], scaninfo[:,5], '.')
#plt.show()
#
#plt.plot(scaninfo[:,1], scaninfo[:,4], '.')
#plt.show()
#
#plt.plot(scaninfo[:,1], scaninfo[:,5], '.')
#plt.show()
#
#plt.plot(scaninfo[:,2], scaninfo[:,4], '.')
#plt.show()
#
#plt.plot(scaninfo[:,2], scaninfo[:,5], '.')
#plt.show()
#
#plt.plot(scaninfo[:,3], scaninfo[:,4], '.')
#plt.show()
#
#plt.plot(scaninfo[:,3], scaninfo[:,5], '.')
#plt.show()
#
##compare these to the mass of line first
##figure out how to append line mass
#
#
##^the ones with unreasonable looking ratios all have shit for actual intensity
##this isn't aa bad measure of the weight of important of a scan
#scaninfo[:,2].mean()
#(scaninfo[:,2] * scaninfo[:,1]).sum() / scaninfo[:,1].sum()
#(scaninfo[:,2] * scaninfo[:,0]).sum() / scaninfo[:,0].sum()
##~10% holds true, less when you weight by ms1 input intensity
