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
from bisect import bisect
import heapq
from time import time
import pandas as pd
import gc
import concurrent.futures
import multiprocessing as mp
from collections import Counter, defaultdict
from scipy import sparse, integrate, spatial, stats, special
from pandas.api.types import CategoricalDtype
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
import os
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

basefolder = '/'.join((mzmlfile.split('/')[:-2]))
basefile = mzmlfile.split('/')[-1].split('.mzML')[0]

processinglocation = '/'.join((basefolder, 'fileprocessing', basefile))
nprocs = os.cpu_count()
global ppmtol
ppmtol = 25

linesofscansfile = '/'.join((processinglocation, 'linesofscans.pickle'))
with open(linesofscansfile, 'rb') as pick:
    global linesofscans
    linesofscans = pickle.load(pick)
#linesofscans = defaultdict(list) #ms2 scan index: [lineuids within window]

scansoflinesfile = '/'.join((processinglocation, 'scansoflines.pickle'))
with open(scansoflinesfile, 'rb') as pick:
    global scansoflines
    scansoflines = pickle.load(pick)
#linesofscans = defaultdict(list) #ms2 scan index: [lineuids within window]

analytefile = '/'.join((processinglocation, 'analytefactors.pickle'))
with open(analytefile, 'rb') as pick:
    distributionsoflines, linesofdistributions = pickle.load(pick)[3:5]
#distributionsoflines: lineuid: distid
#linesofdistributions: distid: [lineuids ordered by mass

mslevelfile = '/'.join((processingdirectory, 'centroid.ms2.pickle'))
with open(mslevelfile, 'rb') as pick:
    ms2scans = pickle.load(pick)

def intersection_merge(mergable_items):
    sn = 0
    itemgroups = defaultdict(set) #groupn: [members]
    groupsofitems = {} #line: groupn
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

nt = time()

#distsofscans = {}
#for scan, lines in linesofscans.items():
#    dists = set(distributionsoflines[i] for i in lines)
#    distsofscans[scan] = tuple(dists)

#mergeddistributions = list(map(tuple, intersection_merge(distsofscans.values())))

mergedlines = intersection_merge(linesofscans.values())
mergedscans = [tuple(set(itertools.chain(*[scansoflines[j] for j in i]))) for i in mergedlines]

print(len(list(chain.from_iterable(mergedscans))), 'relevant scans')
print(len(mergedscans), 'scan groups')
print(time() - nt, 'organizing scangroups')

#msrun = mzml.MzML(mzmlfile, dtype=np.float64)

nt = time()

primarycount = 0
global massesandindices
massesandindices = {} #scan: [[masses], [primary indices]]
global scansbyprimaryind
scansbyprimaryind = {} #primary: scan
indexofprimaryinds = {} #primary: mass index in scan
primariesofscansbyindex = defaultdict(dict) #scan: index: primary
#for scan in msrun:
for scanindex in ms2scans:
    #if scan['ms level'] == 2:
    #scanindex = scan['index']
    if scanindex in linesofscans:
        scanmasses, intensities = ms2scans[scanindex].values()
        scanmasses = scanmasses.tolist()
        mlen = len(scanmasses)
        scanlist = itertools.repeat(scanindex, mlen)
        inds = np.arange(mlen)
        primaries = (inds + primarycount).tolist()
        inds = inds.tolist()
        primariesofscansbyindex[scanindex].update(zip(inds, primaries))
        scansbyprimaryind.update(zip(primaries, scanlist))
        indexofprimaryinds.update(zip(primaries, inds))
        massesandindices[scanindex] = [scanmasses, primaries]
        primarycount += mlen

print(time() - nt, 'primary mass indexing')

#for n, group in enumerate(mergedscans):
def boundary_calculation(group):
    mainind = 0
    primarytomainindex = {} #primary index: main index
    maintoprimaryindex = defaultdict(list) #main index: [primary indices]
    mainindicesbyscan = defaultdict(list) #scan: [main indices], previously known as scandict
    scansbymainindices = defaultdict(set)
    scans, masses, primaryinds = [], [], []
    for scan in group:
        scanmasses, primaries = massesandindices[scan]
        scanlist = itertools.repeat(scan, len(scanmasses))
        primaryinds.extend(primaries)
        masses.extend(scanmasses)
        scans.extend(scanlist)
    masses = np.array(masses)[:,None]
    radius = (masses * ppmtol).flatten() / 1000000
    nn = spatial.KDTree(masses)
    matches = nn.query_ball_point(masses, radius).tolist()
    groupableinds = list(map(tuple, intersection_merge(matches)))
    for gi in groupableinds:
        for g in gi:
            primaryind = primaryinds[g]
            primarytomainindex[primaryind] = mainind
            maintoprimaryindex[mainind].append(primaryind)
            scan = scansbyprimaryind[primaryind]
            mainindicesbyscan[scan].append(mainind)
            scansbymainindices[mainind].add(scan)
        mainind += 1
    
    ms1entropy = defaultdict(lambda: Counter()) #ms2line: line: count
    #compare scansbymainindices to linesofscans
    for mainindex, scans in scansbymainindices.items():
        for scan in scans:
            ms1lines = linesofscans[scan]
            for line in ms1lines:
                ms1scans = set(scansoflines[line])
                ms1diffs = len(ms1scans.difference(scans))
                ms1entropy[mainindex][line] -= ms1diffs
    
    assignment_dict = {}
    for mainindex, assignablems1lines in ms1entropy.items():
        primaries = maintoprimaryindex[mainindex]
        for primary in primaries:
            #get scan: scansbyprimaryind -> get lines
            scan = scansbyprimaryind[primary]
            assessablems1lines = linesofscans[scan]
            assessabledict = Counter()
            for a in assessablems1lines:
                if a in assignablems1lines:
                    assessabledict[a] = assignablems1lines[a]
            assessablerankings = assessabledict.most_common(len(assessabledict))
            if len(assessablerankings) > 1:
                if assessablerankings[0][1] == assessablerankings[1][1]:
                    #find all top ranks
                    toprank = assessablerankings[0][1]
                    toplines = []
                    for ms1line, rank in assessablerankings:
                        if rank == toprank:
                            toplines.append(ms1line)
                        else:
                            break
                    toplines = tuple(toplines)
                    assignment_dict[primary] = toplines
                else:
                    #lone top rank
                    #assign primary to ms1 line
                    assignment_dict[primary] = assessablerankings[0][0]
            else:
                #lone top rank
                #assign primary to ms1 line
                assignment_dict[primary] = assessablerankings[0][0]
    return assignment_dict

nt = time()

assignmentresults = {}
#this won't write to assignmentresults but its good for a class
with mp.Pool(nprocs) as pool:
    outputs = pool.map(boundary_calculation, mergedscans)
    for output in outputs:
        assignmentresults.update(output)

print(len(assignmentresults))
print(time() - nt, 'grouped masses')

assignmentresultsfile = '/'.join((processinglocation, 'assignmentresults.pickle'))
with open(assignmentresultsfile, 'wb') as pick:
    pickle.dump(assignmentresults, pick)

primariesofscansbyindexfile = '/'.join((processinglocation, 'primariesofscansbyindex.pickle'))
with open(primariesofscansbyindexfile, 'wb') as pick:
    pickle.dump(primariesofscansbyindex, pick)

indexofprimaryindsfile = '/'.join((processinglocation, 'indexofprimaryinds.pickle'))
with open(indexofprimaryindsfile, 'wb') as pick:
    pickle.dump(indexofprimaryinds, pick)

scansbyprimaryindfile = '/'.join((processinglocation, 'scansbyprimaryind.pickle'))
with open(scansbyprimaryindfile, 'wb') as pick:
    pickle.dump(scansbyprimaryind, pick)
