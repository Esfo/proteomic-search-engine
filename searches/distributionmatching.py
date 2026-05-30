import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from colour import Color
from pyteomics import mzml
from time import time
import lmdb
import pandas as pd
import gc
from functools import partial
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from collections import Counter, defaultdict
import concurrent.futures
import threading
from scipy import sparse, integrate, spatial, stats, special
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
#proton = 1.00727647

#all the memory management for shit, nothing can be made concurrent

mzmlfile = '/home/sfo/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
basefolder = '/'.join((mzmlfile.split('/')[:-2]))
basefile = mzmlfile.split('/')[-1].split('.mzML')[0]

processinglocation = '/'.join((basefolder, 'fileprocessing', basefile))
#librarylocation = '/home/sfo/data/proteomics/fastas/isotope-arrays/human_isotopes-6-50_miss-1_ss50'
librarylocation = '/home/sfo/data/proteomics/fastas/search-db'
#proteome = proteomefile.split('/')[-1].split('.')[0]
proteome = 'Human_Homo_sapien-NoTremb'
#proteome = 'Human_Homo_sapien'
ppmtolerance = 20

environment_partial = partial(lmdb.Environment, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=99999, max_spare_txns=1, lock=True)

scansbyanalytefile = '/'.join((processinglocation, 'scansbyanalyte.pickle'))
with open(scansbyanalytefile, 'rb') as pick:
    scansbyanalyte = pickle.load(pick)
#scansbyanalyte = defaultdict(list) #analyteid: [spectra across all lines and charge states]

scansoflinesfile = '/'.join((processinglocation, 'scansoflines.pickle'))
with open(scansoflinesfile, 'rb') as pick:
    scansoflines = pickle.load(pick)
#scansoflines = defaultdict(list) #lineuid: [ms2 scan indices]

analytefile = '/'.join((processinglocation, 'analytefactors.pickle'))
with open(analytefile, 'rb') as pick:
    analytekeys, analytedistributions, analytesbydistribution, distributionsoflines, linesofdistributions, linesofanalytes = pickle.load(pick)
#analytekeys = defaultdict(dict) #analyte id: distid: charge, this can be one or multiple distributions based on whether it has multiple charge states
#analytedistributions = defaultdict(dict) #analyte id: ordered masses: AUC of merged isotopomers, weighted means [via intensity] across isotopomers from every charge state, isotopomer datapoints merged across each charge state - if there are any
#analytesbydistribution = {} #distid: analyte id

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

def intersection_merge(mergable_items):
    sn = 0
    groupsofitems = {} #member: group
    itemgroups = defaultdict(set) #group: [members]
    for items in mergable_items:
        locs = set()
        for i in items:
            if i in groupsofitems:
                locs.add(groupsofitems[i])
        if locs:
            joiner = min(locs)
            if len(locs) > 1:
                for oldloc in locs.difference([joiner]):
                    for ol in itemgroups[oldloc]:
                        groupsofitems[ol] = joiner
                    itemgroups[joiner].update(itemgroups.pop(oldloc))
        else:
            joiner = sn
            sn += 1
        itemgroups[joiner].update(items)
        for i in items:
            groupsofitems[i] = joiner
    return list(itemgroups.values())

nt = time()

sumabundances = {}
with environment_partial(librarylocation) as env:
    formuladb = env.open_db(('proteomes.formulalist').encode())
    with env.begin(write=False) as txn:
        with txn.cursor(formuladb) as cursor:
            pulledformulas = eval(cursor.get(proteome.encode()).decode())
    getkeys = [i.encode() for i in pulledformulas]
    sums = env.open_db('distributions.sum'.encode())
    with env.begin(write=False) as txn:
        with txn.cursor(sums) as cursor:
            for k, v in cursor.getmulti(getkeys):
                out = np.frombuffer(v)
                out = out.reshape(2, out.size//2)
                sumabundances[k.decode()] = out

print(time() - nt, 'formulas loaded')
nt = time()

librarykeys = []
librarymasses = []
librarymassdict = {} #lid: [masses]
librarypositions = {} #lid: [indices]
#libraryintensities = {} #lid: [intensities]
libraryintensityranks = {} #lid: [intensityranks]
#librarydirections = {} #lid: [increasing/decreasing, max=0]
for f, (masses, intensities) in sumabundances.items():
    k = formulaidentifiers[f]
    librarymassdict[k] = masses
    librarypositions[k] = list(range(masses.size))
    #libraryintensities[k] = intensities
    intensityranks = np.abs(intensities.argsort().argsort() - intensities.size + 1)
    #maxloc = intensities.argmax()
    #leftdirections = (np.diff(intensities[:maxloc+1]) > 0).tolist()
    #rightdirections = (np.diff(intensities[maxloc:]) > 0).tolist()
    #directions = [1 if i else -1 for i in leftdirections] + [0] + [1 if i else -1 for i in rightdirections]
    #librarydirections[k] = directions
    libraryintensityranks[k] = intensityranks
    librarykeys.extend(itertools.repeat(k, masses.size))
    librarymasses.extend(masses.tolist())

librarykeys = np.array(librarykeys)
librarymasses = np.array(librarymasses)

librarykeys = librarykeys[librarymasses.argsort()]
librarymasses = np.sort(librarymasses)

distributionkeys = []
distributionmasses = []
distributionmassdict = {} #did: [masses]
#distributionintensities = {} #did: [intensities]
distributionintensityranks = {} #did: [intensityranks]
#distributiondirections = {} #did: [+/- directions]
for k, (masses, intensities) in analytedistributions.items():
    distributionmassdict[k] = masses
    #distributionintensities[k] = intensities
    intensityranks = np.abs(intensities.argsort().argsort() - intensities.size + 1)
    #maxloc = intensities.argmax()
    #leftdirections = (np.diff(intensities[:maxloc+1]) > 0).tolist()
    #rightdirections = (np.diff(intensities[maxloc:]) > 0).tolist()
    #directions = [1 if i else -1 for i in leftdirections] + [0] + [1 if i else -1 for i in rightdirections]
    #distributiondirections[k] = directions
    distributionintensityranks[k] = intensityranks
    distributionkeys.extend(itertools.repeat(k, masses.size))
    distributionmasses.extend(masses.tolist())

distributionkeys = np.array(distributionkeys)
distributionmasses = np.array(distributionmasses)

distributionkeys  = distributionkeys[distributionmasses.argsort()]
distributionmasses = np.sort(distributionmasses)

#radius = distributionmasses / 1000000 * ppmtolerance
ppmmod = ppmtolerance / 1000000

#switch the train to distmatches?
#lmtree = spatial.KDTree(librarymasses[:,None])
#matches1 = lmtree.query_ball_point(distributionmasses[:,None], radius, workers=8)
matches = radius_neighbors(librarymasses.tolist(), distributionmasses.tolist(), ppmmod)

#matchorganizer1 = defaultdict(list)
#for dk, lkeys in zip(distributionkeys, matches1):
#    matchorganizer1[dk].extend(librarykeys[lkeys])
matchorganizer = defaultdict(list)
for k, lkeys in matches.items():
    dk = distributionkeys[k]
    matchorganizer[dk].extend(librarykeys[lkeys])

for k in list(matchorganizer):
    matchorganizer[k] = np.array(list(set(matchorganizer[k])))

#analytesbyformula = defaultdict(set) #formula: [analyteids in scansbyanalytes]
#linepositionsofanalytes = defaultdict(lambda: defaultdict(set)) #analyteid in scansbyanalytes: position: [lines in scansoflines]
linepositionsbyformula = defaultdict(lambda: defaultdict(set)) #formula: matchable library positions of a distribution: [lines that have ms2 scans]

librarymatchesbyanalyte = defaultdict(list) #distribution key: [library keys]

lmatches = 0
dmatches = 0
for dk, lkeys in matchorganizer.items():
    if dk in scansbyanalyte:
        dmasses = distributionmassdict[dk]
        dsize = dmasses.size
        tx = 0
        for lk in lkeys.tolist(): #can this loop be made concurrent?
            lmasses = librarymassdict[lk]
            lsize = lmasses.size
            
            leftoffset = int(round(lmasses.tolist()[0] - dmasses.tolist()[0]))
            if leftoffset > 0:
                li = 0
                rmax = dsize - leftoffset
                maxsize = min(lsize, rmax)
                ri = leftoffset
            elif leftoffset == 0:
                li = 0
                ri = 0
                maxsize = min(lsize, dsize)
            else: #< 0
                li = -leftoffset
                ri = 0
                lmax = lsize - li
                maxsize = min(lmax, dsize)
            le = li + maxsize
            lrange = le - li
            #if lrange > 1: #at least 2 matches
            lintranks = libraryintensityranks[lk][li:le]
            if 0 in lintranks: #the top library rank is included
                re = ri + maxsize
                dorders = distributionintensityranks[dk][ri:re].tolist()
                lorders = libraryintensityranks[lk][li:le].tolist()
                orderdiffs = [abs(i-j) for i, j in zip(dorders, lorders)]
                allowance = sum(orderdiffs)
                if allowance == 0: #complete heirarchical match
                    librarymatchesbyanalyte[dk].append(lk)
                    distlines = linesofanalytes[dk][ri:re]
                    positions = librarypositions[lk][li:le]
                    formula = distributionidentifiers[lk]
                    for lines, pos in zip(distlines, positions):
                        for line in lines:
                            if line in scansoflines:
                                #analytesbyformula[formula].add(dk)
                                #linepositionsofanalytes[dk][pos].add(line)
                                #looks like the right way to do this is:
                                #formula: position: line
                                linepositionsbyformula[formula][pos].add(line)
                    tx += 1
        if tx > 0:
            lmatches += tx
            dmatches += 1

print(time() - nt, 'matches assembled')
print('library matches:', lmatches)
print('dist matches:', dmatches)

#for k, v in analytesbyformula.items():
#    analytesbyformula[k] = tuple(v)
#analytesbyformula = dict(analytesbyformula)
#
#for k, v in linepositionsofanalytes.items():
#    for sk, sv in v.items():
#        v[sk] = tuple(sv)
#    linepositionsofanalytes[k] = dict(v)
#linepositionsofanalytes = dict(linepositionsofanalytes)

for k, v in linepositionsbyformula.items():
    for sk, sv in v.items():
        v[sk] = tuple(sv)
    linepositionsbyformula[k] = dict(v)
linepositionsbyformula = dict(linepositionsbyformula)

#analytesbyformulafile = '/'.join((processinglocation, 'analytesbyformula.pickle'))
#with open(analytesbyformulafile, 'wb') as pick:
#    pickle.dump(analytesbyformula, pick)
#
#linepositionsofanalytesfile = '/'.join((processinglocation, 'linepositionsofanalytes.pickle'))
#with open(linepositionsofanalytesfile, 'wb') as pick:
#    pickle.dump(linepositionsofanalytes, pick)

#linepositionsbyformulafile = '/'.join((processinglocation, 'linepositionsbyformula.pickle'))
#with open(linepositionsbyformulafile, 'wb') as pick:
#    pickle.dump(linepositionsbyformula, pick)
