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

processingdirectory = '/'.join((basefolder, 'fileprocessing', basefile)) + '/'
#librarylocation = '/home/sfo/data/proteomics/fastas/isotope-arrays/human_isotopes-6-50_miss-1_ss50'
librarylocation = '/home/sfo/data/proteomics/fastas/search-db'
#proteome = proteomefile.split('/')[-1].split('.')[0]
#proteome = 'Human_Homo_sapien-NoTremb'
proteome = 'Human_Homo_sapien-NoTremb'
ppmtolerance = 20

environment_partial = partial(lmdb.Environment, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=99999, max_spare_txns=1, lock=True)

scansbyanalytefile = ''.join((processingdirectory, 'scansbyanalyte.pickle'))
with open(scansbyanalytefile, 'rb') as pick:
    scansbyanalyte = pickle.load(pick)
#scansbyanalyte = defaultdict(list) #analyteid: [spectra across all lines and charge states]

scansoflinesfile = ''.join((processingdirectory, 'scansoflines.pickle'))
with open(scansoflinesfile, 'rb') as pick:
    scansoflines = pickle.load(pick)
#scansoflines = defaultdict(list) #lineuid: [ms2 scan indices]

linesofscansfile = ''.join((processingdirectory, 'linesofscans.pickle'))
with open(linesofscansfile, 'rb') as pick:
    linesofscans = pickle.load(pick)
#linesofscans = defaultdict(list) #ms2 scan index: [lineuids within window]

analytefile = ''.join((processingdirectory, 'analytefactors.pickle'))
with open(analytefile, 'rb') as pick:
    analytekeys, analytedistributions, analytesbydistribution, distributionsoflines, linesofdistributions, linesofanalytes = pickle.load(pick)
#analytekeys = defaultdict(dict) #analyte id: distid: charge, this can be one or multiple distributions based on whether it has multiple charge states
#analytedistributions = defaultdict(dict) #analyte id: ordered masses: AUC of merged isotopomers, weighted means [via intensity] across isotopomers from every charge state, isotopomer datapoints merged across each charge state - if there are any
#analytesbydistribution = {} #distid: analyte id

sumabundances = {} #formula: [sum abundance dist]
maxabundances = {} #formula: [full abundance dist]
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
    maxes = env.open_db('distributions.max'.encode())
    with env.begin(write=False) as txn:
        with txn.cursor(maxes) as cursor:
            for k, v in cursor.getmulti(getkeys):
                out = np.frombuffer(v)
                maxabundances[k.decode()] = out

proton = 1.007276554940804

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

librarykeys = []
librarymasses = []
librarymassdict = {} #lid: [masses]
librarymaxranks = {} #n: [max intensity ranks]
librarypositions = {} #lid: [indices]
libraryintensityranks = {} #lid: [intensityranks]
libraryidentifier = {} #n: formula
for n, (f, (masses, intensities)) in enumerate(sumabundances.items()):
    libraryidentifier[n] = f
    librarymassdict[n] = masses
    librarypositions[n] = list(range(masses.size))
    intensityranks = np.abs(intensities.argsort().argsort() - intensities.size + 1)
    libraryintensityranks[n] = intensityranks
    librarykeys.extend(itertools.repeat(n, masses.size))
    librarymasses.extend(masses.tolist())
    
    maxes = maxabundances[f]
    maxintensityranks = np.abs(maxes.argsort().argsort() - maxes.size + 1)
    librarymaxranks[n] = maxintensityranks

librarykeys = np.array(librarykeys)
librarymasses = np.array(librarymasses)

librarykeys = librarykeys[librarymasses.argsort()]
librarymasses = np.sort(librarymasses)

distributionkeys = []
distributionmasslist = []
distributionintensityranks = {} #did: [intensityranks]
distributionmassdict = {} #did: [masses]
for k, (masses, intensities) in analytedistributions.items():
    intensityranks = np.abs(intensities.argsort().argsort() - intensities.size + 1)
    distributionmassdict[k] = masses
    distributionintensityranks[k] = intensityranks
    distributionkeys.extend(itertools.repeat(k, masses.size))
    distributionmasslist.extend(masses.tolist())

distributionkeys = np.array(distributionkeys)
distributionmasslist = np.array(distributionmasslist)

distributionkeys  = distributionkeys[distributionmasslist.argsort()]
distributionmasslist = np.sort(distributionmasslist)

ppmmod = ppmtolerance / 1000000

matches = radius_neighbors(librarymasses.tolist(), distributionmasslist.tolist(), ppmmod)

matchorganizer = defaultdict(list) #distributionkeys: [library formulas]
for k, lkeys in matches.items():
    dk = distributionkeys[k]
    matchorganizer[dk].extend(librarykeys[lkeys])

for k in list(matchorganizer):
    matchorganizer[k] = np.array(list(set(matchorganizer[k])))

linepositionsbyformula = defaultdict(lambda: defaultdict(set)) #formula: matchable library positions of a distribution: [lines that have ms2 scans]
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
            sumlintranks = libraryintensityranks[lk][li:le]
            maxlintranks = librarymaxranks[lk][li:le]
            if 0 in sumlintranks or 0 in maxlintranks: #the top library rank is included
                re = ri + maxsize
                dorders = distributionintensityranks[dk][ri:re].tolist()
                lsorders = sumlintranks.tolist()
                lmorders = maxlintranks.tolist()
                sdorderdiffs = [abs(i-j) for i, j in zip(dorders, lsorders)] #library sum to dist
                mdorderdiffs = [abs(i-j) for i, j in zip(dorders, lmorders)] #dist to library max
                lmorderdiffs = [abs(i-j) for i, j in zip(lmorders, lsorders)] #library sum to max
                sdallowance = sum(sdorderdiffs)
                mdallowance = sum(mdorderdiffs)
                liballowance = sum(lmorderdiffs)
                #dints = distributionintensities[dk][ri:re].tolist()
                #lints = libraryintensities[lk][li:le].tolist()
                #dsum = sum(dints)
                #lsum = sum(lints)
                #dnorm = [i / dsum for i in dints]
                #lnorm = [i / lsum for i in lints]
                #intensitydiff = [d - l for d, l in zip(dnorm, lnorm)]
                #idmean = sum(intensitydiff) / maxsize
                #intensitydiffs = [abs(idmean - i) for i in intensitydiff]
                #meanintensitydiff = sum(intensitydiffs) / maxsize
                #if allowance == 0: #complete heirarchical match
                if sdallowance <= liballowance or mdallowance <= liballowance:
                    #zeroscores.append(meanintensitydiff)
                    #librarymatchesbydistribution[dk].append(lk)
                    #distlines = linesofdistributions[dk][ri:re]
                    distlines = linesofanalytes[dk][ri:re]
                    positions = librarypositions[lk][li:le]
                    formula = libraryidentifier[lk]
                    for lines, pos in zip(distlines, positions):
                        for line in lines:
                            if line in scansoflines:
                                linepositionsbyformula[formula][pos].add(line)
                    tx += 1
                #else:
                #    #keep scores of other distributions
                #    nonzeroscores.append(meanintensitydiff)
        if tx > 0:
            lmatches += tx
            dmatches += 1

for k, v in linepositionsbyformula.items():
    for sk, sv in v.items():
        v[sk] = tuple(sv)
    linepositionsbyformula[k] = dict(v)
linepositionsbyformula = dict(linepositionsbyformula)

print('library matches:', lmatches)
print('distribution matches:', dmatches)

#linepositionsbyformulafile = ''.join((processingdirectory, 'linepositionsbyformula.pickle'))
#with open(linepositionsbyformulafile, 'wb') as pick:
#    pickle.dump(linepositionsbyformula, pick)
