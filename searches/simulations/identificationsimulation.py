import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from colour import Color
import psutil
import asyncio
import aiofiles
from decimal import Decimal
from pyteomics import mzml
import csv
from bisect import bisect
import heapq
from time import time
import pandas as pd
import re
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
import dill
dill.settings['recurse'] = True
mp.util.pickle = dill
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

#make a fake retention times, discrete points
#make fake analytes out of fake peak distributions with fake charges
#make fake scans and scan windows, overlap certain analytes via scan window
#assign intensities to each analyte within a scan window
#generate fake fragments of a scan, randomizing a +/- of # fragments and intensities based on the difference of the flanking ms1 intensities

#to make a fake peak:
#generate random floats from 0-1
#sort them in order
#take +/- 1/2 of the number of datapoints (random)
#make the second half of the peak, reverse sort
#apply noise filter for random +/- %

nprocs = os.cpu_count()

nproteins = 300
minproteinlength = 200
maxproteinlength = 800
minpeptidelength = 6
maxpeptidelength = 20 #decreasing this decreases uniques + entropic boundary clarity?
missedcleavages = 1

cut = [{'K':1, 'R':1}, {'K':'P', 'R':'P'}] #trypsin

proton = 1.007276554940804

nanalytes = 300
scanradius = 0.5 #m/z window on either side of a target mass
scansperdistribution = 1 #>= 1 guarantees that number on every distribution, < 1 will randomly choose which ones receive a scan event
ppmtolerance = 25 #ppm, fragment ions within this window of another will be grouped
intensityvariation = 0.1 #random variation in ms1 distribution intensities
minimumabundance = 0.01 #with the current function, as long as this is low the the dists will be pretty decent, i don't care to switch this function atm
dividingthreshold = 0.1

#ms1
minrt = 0
maxrt = 120
pointsperminute = 80
massmin = 300
massmax = 2000
minpoints = 5 #datapoints
maxpoints = 190
minisotopomers = 2
maxisotopomers = 8
minintensity = 1e2
maxintensity = 1e10

#ms2
ions = 'by'
subisotopomericdepth = 0.7

chargestates = { #n analytes: probability
        1: 0.9,
        2: 0.1
        }

maxchargedistancepercent = 0.05 #distance between isotopomers
charges = { #charge: probability
        1: 0.05,
        2: 0.9,
        3: 0.05
        }

rtpoints = pointsperminute*maxrt
retentiontimes = np.linspace(minrt, maxrt, num=rtpoints)

proton = 1.007276554940804

elementalprobabilities = { #isotope: abundance
        'H1': 0.999885,
        'H2': 0.000115,
        'C12': 0.9893,
        'C13': 0.0107,
        'N14': 0.99636,
        'N15': 0.00364,
        'O16': 0.99757,
        'O17': 0.00038,
        'O18': 0.00205,
        'S32': 0.9499,
        'S33': 0.0075,
        'S34': 0.0425,
        'S36': 0.0001}

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

#elementvector = [0 for _ in elementalmasses]
#elementlist = list(elementalmasses)
#vectorpositions = {k: n for n, k in enumerate(elementlist)}
#elementpositions = {n: k for n, k in enumerate(elementlist)}

vectorrangesbyelement = {'H': range(0,2),
                         'C': range(2,4),
                         'N': range(4,6),
                         'O': range(6,9),
                         'S': range(9,13)}

nonmonoisotopicelements = {'H2', 'C13', 'N15', 'O17', 'O18', 'S33', 'S34', 'S36'}

isotopesbyelement = { #element: isotopes
        'H': ('H1', 'H2'),
        'C': ('C12', 'C13'),
        'N': ('N14', 'N15'),
        'O': ('O16', 'O17', 'O18'),
        'S': ('S32', 'S33', 'S34', 'S36')}

monoisotopickeys = { #element: monoisotopic element
        'H': 'H1',
        'C': 'C12',
        'N': 'N14',
        'O': 'O16',
        'S': 'S32'}

nonmonoisotopicgroups = { #element: nonmonoisotopic elements
        'H': ('H2',),
        'C': ('C13',),
        'N': ('N15',),
        'O': ('O17', 'O18'),
        'S': ('S33', 'S34', 'S36')}

elementvectors = {}
vectorpositions = {}
elementpositions = {}
for e, isos in isotopesbyelement.items():
    elementvectors[e] = [0 for _ in range(len(isos))]
    vectorpositions[e] = {k: n for n, k in enumerate(isos)}
    elementpositions[e] = {n: k for n, k in enumerate(isos)}

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

#to selectively pick ions below, you can iterate these dicts in order and cumulatively combine until you hit an ion you want to generate, then use those cumulative +/-s as just a single dict entry each in fragmentation_compositions. so you would generate the dicts you plan on using in this file
nfragmentcompositions = {}
nfragmentcompositions['a'] = Counter({'C': -1, 'O': -1})
nfragmentcompositions['b'] = Counter()
nfragmentcompositions['c'] = Counter({'N': 1, 'H': 3})

cfragmentcompositions = {}
cfragmentcompositions['x'] = Counter({'C': 1, 'O': 2})
cfragmentcompositions['y'] = Counter({'H': 2, 'O': 1})
cfragmentcompositions['z'] = Counter({'N': -1, 'H': -1, 'O': 1})
#cfragmentcompositions['z•'] = Counter({'N': -1, 'O': 1})

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

def individual_element_binomial_walk(dividingthreshold, e, acount):
    #elementalorganizer = defaultdict(list)
    elementlist = []
    mainheap = []
    vectorsets = defaultdict(set) #element: set of used vectors
    #for e, acount in atomiccomposition.items():
    mk = monoisotopickeys[e]
    nvector = elementvectors[e].copy()
    nvector[vectorpositions[e][mk]] += acount
    if len(isotopesbyelement[e]) > 2:
        baseprob = elementalprobabilities[mk] ** acount
        preheap = []
        preheap.append([baseprob, acount * elementalmasses[mk], e, nvector.copy()])
        greater = True
        lastprob = baseprob
        while greater:
            greater = False
            for iso in nonmonoisotopicgroups[e]:
                newelementvector = nvector.copy()
                newelementvector[vectorpositions[e][mk]] -= 1
                if newelementvector[vectorpositions[e][mk]] > -1:
                    newelementvector[vectorpositions[e][iso]] += 1
                    vectorsets[e].add(tuple(newelementvector))
                    pn = 0
                    newelementmass = 0
                    newelementprob = 1
                    for n, c in enumerate(newelementvector):
                        loopiso = elementpositions[e][n]
                        newelementmass += elementalmasses[loopiso] * c
                        newelementprob *= elementalprobabilities[loopiso]**c
                        if loopiso in nonmonoisotopicelements:
                            newelementprob *= math.comb(acount-pn, c)
                            pn += c
                    preheap.append([newelementprob, newelementmass, e, newelementvector.copy()])
                    if newelementprob > lastprob:
                        lastprob = newelementprob
                        greater = True
        preheap = sorted(preheap)
        maxiso = preheap[-1]
        maxprob, m, e, nv = maxiso
        elementlist.append([-1, maxprob, m, e, nv])
        maxprob *= -1
        preheap = preheap[:-1]
        for h in preheap:
            r = h[0] / maxprob
            h.insert(0, r)
            heapq.heappush(mainheap, h)
        for iso in nonmonoisotopicgroups[e]:
            v = nv.copy()
            v[vectorpositions[e][mk]] -= 1
            if v[vectorpositions[e][mk]] > -1:
                v[vectorpositions[e][iso]] += 1
                tuplevec = tuple(v)
                if tuplevec not in vectorsets[e]:
                    vectorsets[e].add(tuplevec)
                    pn = 0
                    newelementmass = 0
                    newelementprob = 1
                    for n, c in enumerate(v):
                        loopiso = elementpositions[e][n]
                        newelementmass += elementalmasses[loopiso] * c
                        newelementprob *= elementalprobabilities[loopiso]**c
                        if loopiso in nonmonoisotopicelements:
                            newelementprob *= math.comb(acount-pn, c)
                            pn += c
                    heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, v.copy()])
    else:
        preheap = []
        baseprob = elementalprobabilities[mk] ** acount
        preheap.append([baseprob, acount * elementalmasses[mk], e, nvector.copy()])
        greater = True
        lastprob = baseprob
        iso = nonmonoisotopicgroups[e][0]
        while greater:
            greater = False
            nvector[vectorpositions[e][mk]] -= 1
            if nvector[vectorpositions[e][mk]] > -1:
                nvector[vectorpositions[e][iso]] += 1
                vectorsets[e].add(tuple(nvector))
                pn = 0
                newelementmass = 0
                newelementprob = 1
                for n, c in enumerate(nvector):
                    loopiso = elementpositions[e][n]
                    newelementmass += elementalmasses[loopiso] * c
                    newelementprob *= elementalprobabilities[loopiso]**c
                    if loopiso in nonmonoisotopicelements:
                        newelementprob *= math.comb(acount-pn, c)
                        pn += c
                preheap.append([newelementprob, newelementmass, e, nvector.copy()])
                if newelementprob > lastprob:
                    lastprob = newelementprob
                    greater = True
        preheap = sorted(preheap)
        maxiso = preheap[-1]
        maxprob, m, e, nv = maxiso
        elementlist.append([-1, maxprob, m, e, nv])
        maxprob *= -1
        preheap = preheap[:-1]
        for h in preheap:
            r = h[0] / maxprob
            h.insert(0, r)
            heapq.heappush(mainheap, h)
        v = nv.copy()
        v[vectorpositions[e][mk]] -= 1
        if v[vectorpositions[e][mk]] > -1:
            v[vectorpositions[e][iso]] += 1
            tuplevec = tuple(v)
            if tuplevec not in vectorsets[e]:
                vectorsets[e].add(tuplevec)
                pn = 0
                newelementmass = 0
                newelementprob = 1
                for n, c in enumerate(v):
                    loopiso = elementpositions[e][n]
                    newelementmass += elementalmasses[loopiso] * c
                    newelementprob *= elementalprobabilities[loopiso]**c
                    if loopiso in nonmonoisotopicelements:
                        newelementprob *= math.comb(acount-pn, c)
                        pn += c
                heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, v.copy()])
    
    cutoff = -maxprob * dividingthreshold

    r, p, m, e, v = heapq.heappop(mainheap)
    elementlist.append([r, p, m, e, v])
    if len(isotopesbyelement[e]) > 2:
        while p > cutoff:
            for iso in nonmonoisotopicgroups[e]:
                newelementvector = v.copy()
                newelementvector[vectorpositions[e][mk]] -= 1
                if newelementvector[vectorpositions[e][mk]] > 0:
                    newelementvector[vectorpositions[e][iso]] += 1
                    tuplevec = tuple(newelementvector)
                    if tuplevec not in vectorsets[e]:
                        vectorsets[e].add(tuplevec)
                        pn = 0
                        newelementmass = 0
                        newelementprob = 1
                        for n, c in enumerate(newelementvector):
                            loopiso = elementpositions[e][n]
                            newelementmass += elementalmasses[loopiso] * c
                            newelementprob *= elementalprobabilities[loopiso]**c
                            if loopiso in nonmonoisotopicelements:
                                newelementprob *= math.comb(acount-pn, c)
                                pn += c
                        heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, newelementvector.copy()])
            r, p, m, e, v = heapq.heappop(mainheap)
            elementlist.append([r, p, m, e, v])
    else:
        iso = nonmonoisotopicgroups[e][0]
        while p > cutoff:
            nvector = v.copy()
            nvector[vectorpositions[e][mk]] -= 1
            if nvector[vectorpositions[e][mk]] > 0:
                nvector[vectorpositions[e][iso]] += 1
                tuplevec = tuple(nvector)
                if tuplevec not in vectorsets[e]:
                    vectorsets[e].add(tuplevec)
                    pn = 0
                    newelementmass = 0
                    newelementprob = 1
                    for n, c in enumerate(nvector):
                        loopiso = elementpositions[e][n]
                        newelementmass += elementalmasses[loopiso] * c
                        newelementprob *= elementalprobabilities[loopiso]**c
                        if loopiso in nonmonoisotopicelements:
                            newelementprob *= math.comb(acount-pn, c)
                            pn += c
                    heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, nvector.copy()])
            r, p, m, e, v = heapq.heappop(mainheap)
            elementlist.append([r, p, m, e, v])
    heapq.heapify(elementlist)
    return elementlist

def descending_partial_products(dividingthreshold, elementalorganizer):
    for k in elementalorganizer:
        heapq.heapify(elementalorganizer[k])
    
    mainpool = defaultdict(list) #things already popped from elementalorganizer
    for k in elementalorganizer:
        mainpool[k].append(heapq.heappop(elementalorganizer[k]))
    
    formula = ''
    maxprob = 1
    mainmass = 0
    finalabundances = {} #subformula: prob
    for b in sorted(mainpool):
        for r, p, m, e, v in mainpool[b]:
            for n, c in enumerate(v):
                if c > 0:
                    formula += f'{elementpositions[e][n]}({c})'
            maxprob *= p
            mainmass += m

    finalabundances[formula] = [mainmass, maxprob]

    cutoff = maxprob * dividingthreshold
    mainheap = list(itertools.chain(*elementalorganizer.values()))
    heapq.heapify(mainheap)

    multinomialpath = [] #sublists not in mainpool
    probabilityranking = [] #representative lists of ratio probability to sort multinomialpath
    while mainheap:
        r, p, m, e, v = heapq.heappop(mainheap)
        baseiter = {k: v for k, v in mainpool.items() if k != e}
        baseiter[e] = [[r, p, m, e, v]]
        
        formula = ''
        prob = 1
        mass = 0
        for b in sorted(baseiter):
            for sr, sp, sm, se, sv in baseiter[b]:
                for n, c in enumerate(sv):
                    if c > 0:
                        formula += f'{elementpositions[se][n]}({c})'
                prob *= sp
                mass += sm
        
        finalabundances[formula] = [mass, prob]
        if prob < cutoff:
            break
        
        ind = bisect(probabilityranking, r)
        probabilityranking.insert(ind, r)
        multinomialpath.insert(ind, [r, p, m, e, v])
        
        checkedcombos = set()
        for path in multinomialpath.copy():
            multielement = False
            match path[1]:
                case list():
                    multielement = True
                    sepool = set()
                    sepool.add(e)
                    seformulas = []
                    multipath = []
                    nsr = 1
                    for sr, sp, sm, se, sv in path[1:]:
                        if se not in sepool:
                            nsr *= sr
                            sepool.add(se)
                            sef = ''
                            for n, c in enumerate(sv):
                                if c > 0:
                                    sef += f'{elementpositions[se][n]}({c})'
                            seformulas.append(sef)
                            multipath.append([sr, sp, sm, se, sv])
                    checkformula = ''.join((sorted(seformulas)))
                    if checkformula in checkedcombos:
                        continue
                    else:
                        checkedcombos.add(checkformula)
                    if len(multipath) == 0:
                        continue
                case _:
                    sr, sp, sm, se, sv = path
                    sef = ''.join((f'{se}{str(n)}{(val)}' for n, val in enumerate(sv)))
                    if sef in checkedcombos:
                        continue
                    else:
                        checkedcombos.add(sef)
                    if se == e:
                        continue
                    nsr = sr
            newratio = nsr * r
            if newratio > 0:
                newratio *= -1
            if -newratio >= dividingthreshold:
                if multielement:
                    seformula = ''
                    newprob = 1
                    newmass = 0
                    newiter = {k: v for k, v in baseiter.items() if k not in sepool}
                    newiter[e] = [[r, p, m, e, v]]
                    for ir, ip, im, ie, iv in multipath:
                        newiter[ie] = [[ir, ip, im, ie, iv]]
                    for b in sorted(newiter):
                        for ir, ip, im, ie, iv in newiter[b]:
                            for n, c in enumerate(iv):
                                if c > 0:
                                    seformula += f'{elementpositions[ie][n]}({c})'
                            newprob *= ip
                            newmass += im
                else:
                    newiter = {k: v for k, v in baseiter.items() if k != se}
                    newiter[se] = [[sr, sp, sm, se, sv]]
                    seformula = ''
                    newprob = 1
                    newmass = 0
                    for b in sorted(newiter):
                        for ir, ip, im, ie, iv in newiter[b]:
                            for n, c in enumerate(iv):
                                if c > 0:
                                    seformula += f'{elementpositions[ie][n]}({c})'
                            newprob *= ip
                            newmass += im
                if newprob >= cutoff:
                    finalabundances[seformula] = [newmass, newprob]
                    if multielement:
                        ind = bisect(probabilityranking, newratio)
                        probabilityranking.insert(ind, newratio)
                        multinomialpath.insert(ind, [newratio, *multipath])
                    else:
                        ind = bisect(probabilityranking, newratio)
                        probabilityranking.insert(ind, newratio)
                        multinomialpath.insert(ind, [newratio, [sr, sp, sm, se, sv], [r, p, m, e, v]])
            else:
                break

    subformulas, massesandabundances = list(zip(*finalabundances.items()))
    subformulas = np.array(subformulas, dtype='S')
    massesandabundances = np.array(massesandabundances).transpose()
    #sorting by mass
    subformulas = subformulas[massesandabundances[0].argsort()].tolist()
    massesandabundances = massesandabundances[:,massesandabundances[0].argsort()]
    return subformulas, massesandabundances

def distribution_generation(formula, atomiccomposition):
    elementalorganizer = {} #element: [[preheaps]]
    for e, acount in atomiccomposition.items():
        elementstring = e + str(acount)
        elementlist = individual_element_binomial_walk(dividingthreshold, e, acount)
        elementalorganizer[e] = elementlist.copy() #don't need to copy the insides
    subformulas, massesandabundances = descending_partial_products(dividingthreshold, elementalorganizer)
    return formula, subformulas, massesandabundances

def seqsplit_dict(tseqs, cut, minlength, maxlength, missedcleavages):
    '''
    Cuts peptide sequences using two dicts, one of enzyme cut sites and the other of spots they can't hit.
    Input is {protein id: sequence}. Output is a dict as {protein id: [list of peptides]}
    '''
    cutsequences = tseqs.copy()
    for site in cut[0]:
        if cut[0][site] == 0:
            splitstring = r''.join(('(?=[', site, '](?!', cut[1][site], '))')) if cut[1][site] else r''.join(('(?=[', site, '])'))
        elif cut[0][site] == 1:
            splitstring = r''.join(('(?<=[', site, '](?!', cut[1][site], '))')) if cut[1][site] else r''.join(('(?<=[', site, '])'))
        cutsequences = {key: re.split(splitstring, cutsequences[key]) for key in cutsequences} if type(list(cutsequences.values())[0]) is str else {key: [k for j in [re.split(splitstring, i) for i in cutsequences[key]] for k in j] for key in cutsequences}
    slist = {}
    for s in cutsequences:
        slist[s] = []
        midlist = {}
        midlist[s] = []
        for y in range(missedcleavages+1):
            midlist[s].extend([''.join((cutsequences[s][i:i+y+1])) for i in range(len(cutsequences[s]))])
        if cutsequences[s][0].startswith('M'): #N-terminal cleavage, makes the cleaved and uncleaved version of the n-terminal peptide
            midlist[s].extend([''.join((cutsequences[s][0][1:], ''.join((cutsequences[s][1:y+1])))) for y in range(missedcleavages+1)])
        slist[s].extend(list(filter(lambda x: maxlength >= len(x) >= minlength, set(midlist[s]))))
        slist[s].sort(key=lambda x: (tseqs[s].find(x), len(x)))
    return slist

def fragment_element_binomial_walk(e, acount, fragprobabilities):
    #elementalorganizer = defaultdict(list)
    nvector = []
    fragmentvectorpositions = {} #iso: position in vector, replacing nvectorpositions
    fragmentelementpositions = {} #position: iso
    maxinitial = 0
    for n, (iso, prob) in enumerate(fragprobabilities.items()):
        nvector.append(0)
        fragmentvectorpositions[iso] = n
        fragmentelementpositions[n] = iso
        if prob > maxinitial:
            maxinitial = prob
            mk = iso
    lesserfragmentisotopes = [i for i in fragprobabilities if i != mk] #replacing nonmonoisotopicgroups
    elementlist = []
    mainheap = []
    vectorsets = defaultdict(set) #element: set of used vectors
    #for e, acount in atomiccomposition.items():
    #mk = monoisotopickeys[e]
    #nvector = elementvectors[e].copy()
    #nvector[nvectorpositions[e][mk]] += acount
    nvector[fragmentvectorpositions[mk]] += acount
    #if len(isotopesbyelement[e]) > 2:
    flen = len(fragprobabilities)
    if flen > 2:
        baseprob = fragprobabilities[mk] ** acount
        preheap = []
        preheap.append([baseprob, acount * elementalmasses[mk], e, nvector.copy()])
        greater = True
        lastprob = baseprob
        while greater:
            greater = False
            for iso in lesserfragmentisotopes:
                newelementvector = nvector.copy()
                newelementvector[fragmentvectorpositions[mk]] -= 1
                if newelementvector[fragmentvectorpositions[mk]] > -1:
                    newelementvector[fragmentvectorpositions[iso]] += 1
                    vectorsets[e].add(tuple(newelementvector))
                    pn = 0
                    newelementmass = 0
                    newelementprob = 1
                    for n, c in enumerate(newelementvector):
                        loopiso = fragmentelementpositions[n]
                        newelementmass += elementalmasses[loopiso] * c
                        newelementprob *= fragprobabilities[loopiso]**c
                        #if loopiso in nonmonoisotopicelements:
                        if n > 0:
                            newelementprob *= math.comb(acount-pn, c)
                            pn += c
                    preheap.append([newelementprob, newelementmass, e, newelementvector.copy()])
                    if newelementprob > lastprob:
                        lastprob = newelementprob
                        greater = True
        preheap = sorted(preheap)
        maxiso = preheap[-1]
        maxprob, m, e, nv = maxiso
        elementlist.append([-1, maxprob, m, e, nv])
        maxprob *= -1
        preheap = preheap[:-1]
        for h in preheap:
            r = h[0] / maxprob
            h.insert(0, r)
            heapq.heappush(mainheap, h)
        for iso in lesserfragmentisotopes:
            v = nv.copy()
            v[fragmentvectorpositions[mk]] -= 1
            if v[fragmentvectorpositions[mk]] > -1:
                v[fragmentvectorpositions[iso]] += 1
                tuplevec = tuple(v)
                if tuplevec not in vectorsets[e]:
                    vectorsets[e].add(tuplevec)
                    pn = 0
                    newelementmass = 0
                    newelementprob = 1
                    for n, c in enumerate(v):
                        loopiso = fragmentelementpositions[n]
                        newelementmass += elementalmasses[loopiso] * c
                        newelementprob *= fragprobabilities[loopiso]**c
                        #if loopiso in nonmonoisotopicelements:
                        if n > 0:
                            newelementprob *= math.comb(acount-pn, c)
                            pn += c
                    heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, v.copy()])
    else:
        preheap = []
        baseprob = fragprobabilities[mk] ** acount
        preheap.append([baseprob, acount * elementalmasses[mk], e, nvector.copy()])
        greater = True
        lastprob = baseprob
        iso = lesserfragmentisotopes[0]
        while greater:
            greater = False
            nvector[fragmentvectorpositions[mk]] -= 1
            if nvector[fragmentvectorpositions[mk]] > -1:
                nvector[fragmentvectorpositions[iso]] += 1
                vectorsets[e].add(tuple(nvector))
                pn = 0
                newelementmass = 0
                newelementprob = 1
                for n, c in enumerate(nvector):
                    loopiso = fragmentelementpositions[n]
                    newelementmass += elementalmasses[loopiso] * c
                    newelementprob *= fragprobabilities[loopiso]**c
                    #if loopiso in nonmonoisotopicelements:
                    if n > 0:
                        newelementprob *= math.comb(acount-pn, c)
                        pn += c
                preheap.append([newelementprob, newelementmass, e, nvector.copy()])
                if newelementprob > lastprob:
                    lastprob = newelementprob
                    greater = True
        preheap = sorted(preheap)
        maxiso = preheap[-1]
        maxprob, m, e, nv = maxiso
        elementlist.append([-1, maxprob, m, e, nv])
        maxprob *= -1
        preheap = preheap[:-1]
        for h in preheap:
            r = h[0] / maxprob
            h.insert(0, r)
            heapq.heappush(mainheap, h)
        v = nv.copy()
        v[fragmentvectorpositions[mk]] -= 1
        if v[fragmentvectorpositions[mk]] > -1:
            v[fragmentvectorpositions[iso]] += 1
            tuplevec = tuple(v)
            if tuplevec not in vectorsets[e]:
                vectorsets[e].add(tuplevec)
                pn = 0
                newelementmass = 0
                newelementprob = 1
                for n, c in enumerate(v):
                    loopiso = fragmentelementpositions[n]
                    newelementmass += elementalmasses[loopiso] * c
                    newelementprob *= fragprobabilities[loopiso]**c
                    #if loopiso in nonmonoisotopicelements:
                    if n > 0:
                        newelementprob *= math.comb(acount-pn, c)
                        pn += c
                heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, v.copy()])
    
    cutoff = -maxprob * dividingthreshold

    r, p, m, e, v = heapq.heappop(mainheap)
    elementlist.append([r, p, m, e, v])
    if flen > 2:
        while p > cutoff:
            for iso in lesserfragmentisotopes:
                newelementvector = v.copy()
                newelementvector[fragmentvectorpositions[mk]] -= 1
                if newelementvector[fragmentvectorpositions[mk]] > 0:
                    newelementvector[fragmentvectorpositions[iso]] += 1
                    tuplevec = tuple(newelementvector)
                    if tuplevec not in vectorsets[e]:
                        vectorsets[e].add(tuplevec)
                        pn = 0
                        newelementmass = 0
                        newelementprob = 1
                        for n, c in enumerate(newelementvector):
                            loopiso = fragmentelementpositions[n]
                            newelementmass += elementalmasses[loopiso] * c
                            newelementprob *= fragprobabilities[loopiso]**c
                            #if loopiso in nonmonoisotopicelements:
                            if n > 0:
                                newelementprob *= math.comb(acount-pn, c)
                                pn += c
                        heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, newelementvector.copy()])
            r, p, m, e, v = heapq.heappop(mainheap)
            elementlist.append([r, p, m, e, v])
            try:
                r, p, m, e, v = heapq.heappop(mainheap)
                elementlist.append([r, p, m, e, v])
            except IndexError:
                #mainheap is empty, this can happen when count is low and probabilities are evenly split. When this happened it was in the below loop, but I'll keep this here too just in case
                break
    else:
        iso = lesserfragmentisotopes[0]
        while p > cutoff:
            nvector = v.copy()
            nvector[fragmentvectorpositions[mk]] -= 1
            if nvector[fragmentvectorpositions[mk]] > 0:
                nvector[fragmentvectorpositions[iso]] += 1
                tuplevec = tuple(nvector)
                if tuplevec not in vectorsets[e]:
                    vectorsets[e].add(tuplevec)
                    pn = 0
                    newelementmass = 0
                    newelementprob = 1
                    for n, c in enumerate(nvector):
                        loopiso = fragmentelementpositions[n]
                        newelementmass += elementalmasses[loopiso] * c
                        newelementprob *= fragprobabilities[loopiso]**c
                        #if loopiso in nonmonoisotopicelements:
                        if n > 0:
                            newelementprob *= math.comb(acount-pn, c)
                            pn += c
                    heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, nvector.copy()])
            try:
                r, p, m, e, v = heapq.heappop(mainheap)
                elementlist.append([r, p, m, e, v])
            except IndexError:
                #mainheap is empty, this can happen when count is low and probabilities are evenly split
                break
    heapq.heapify(elementlist)
    return elementlist, fragmentelementpositions

def fragment_descending_partial_products(elementalorganizer, fragmentpositions):
    mainpool = defaultdict(list) #things already popped from elementalorganizer
    for k in elementalorganizer:
        mainpool[k].append(heapq.heappop(elementalorganizer[k]))

    subformulas = []
    sumabundances = []
    massnumberindices = {} #mass number: index in the 2 above lists
    
    formula = ''
    maxprob = 1
    mainmass = 0
    massnumber = 0
    for b in sorted(mainpool):
        for r, p, m, e, v in mainpool[b]:
            for n, c in enumerate(v):
                if c > 0:
                    iso = fragmentpositions[e][n]
                    massnumber += int(iso[1:]) * c
                    formula += f'{iso}({c})'
            maxprob *= p
            mainmass += m

    massnumberindices[massnumber] = 0
    subformulas.append(formula)
    sumabundances.append([mainmass * maxprob, maxprob])
    
    cutoff = maxprob * dividingthreshold
    mainheap = list(itertools.chain(*elementalorganizer.values()))
    heapq.heapify(mainheap)
    
    vectorpool = set()
    multinomialpath = [] #sublists not in mainpool
    probabilityranking = [] #representative lists of ratio probability to sort multinomialpath
    while mainheap:
        r, p, m, e, v = heapq.heappop(mainheap)
        baseiter = {k: v for k, v in mainpool.items() if k != e}
        baseiter[e] = [(r, p, m, e, v)]
        
        formula = ''
        prob = 1
        mass = 0
        massnumber = 0
        for b in sorted(baseiter):
            for sr, sp, sm, se, sv in baseiter[b]:
                for n, c in enumerate(sv):
                    if c > 0:
                        iso = fragmentpositions[se][n]
                        massnumber += int(iso[1:]) * c
                        formula += f'{iso}({c})'
                prob *= sp
                mass += sm
        
        try:
            index = massnumberindices[massnumber]
            subformulas[index] += '-' + formula
            sumabundances[index][0] += mass * prob
            sumabundances[index][1] += prob
        except KeyError: #not in there
            index = len(massnumberindices)
            massnumberindices[massnumber] = index
            subformulas.append(formula)
            sumabundances.append([mass * prob, prob])
        if prob < cutoff:
            break
        
        tsv = tuple(v)
        if tsv not in vectorpool:
            ind = bisect(probabilityranking, r)
            probabilityranking.insert(ind, r)
            multinomialpath.insert(ind, (r, p, m, e, v))
            vectorpool.add(tsv)
        
        checkedcombos = set()
        for path in multinomialpath.copy():
            multielement = False
            match path[1]:
                case tuple():
                    multielement = True
                    sepool = set()
                    sepool.add(e)
                    seformulas = []
                    multipath = []
                    nsr = 1
                    for sr, sp, sm, se, sv in path[1:]:
                        if se not in sepool:
                            nsr *= sr
                            sepool.add(se)
                            sef = ''
                            for n, c in enumerate(sv):
                                if c > 0:
                                    sef += f'{fragmentpositions[se][n]}({c})'
                            seformulas.append(sef)
                            multipath.append((sr, sp, sm, se, sv))
                    checkformula = ''.join((sorted(seformulas)))
                    if checkformula in checkedcombos:
                        continue
                    else:
                        checkedcombos.add(checkformula)
                    if len(multipath) == 0:
                        continue
                case _:
                    sr, sp, sm, se, sv = path
                    sef = ''.join((f'{se}{str(n)}{(val)}' for n, val in enumerate(sv)))
                    if sef in checkedcombos:
                        continue
                    else:
                        checkedcombos.add(sef)
                    if se == e:
                        continue
                    nsr = sr
            newratio = nsr * r
            if newratio > 0:
                newratio *= -1
            if -newratio >= dividingthreshold:
                if multielement:
                    seformula = ''
                    newprob = 1
                    newmass = 0
                    newmassnum = 0
                    newiter = {k: v for k, v in baseiter.items() if k not in sepool}
                    newiter[e] = [(r, p, m, e, v)]
                    for ir, ip, im, ie, iv in multipath:
                        newiter[ie] = [(ir, ip, im, ie, iv)]
                    for b in sorted(newiter):
                        for ir, ip, im, ie, iv in newiter[b]:
                            for n, c in enumerate(iv):
                                if c > 0:
                                    iso = fragmentpositions[ie][n]
                                    newmassnum += int(iso[1:]) * c
                                    seformula += f'{iso}({c})'
                                    #seformula += f'{fragmentpositions[ie][n]}({c})'
                            newprob *= ip
                            newmass += im
                else:
                    newiter = {k: v for k, v in baseiter.items() if k != se}
                    newiter[se] = [(sr, sp, sm, se, sv)]
                    seformula = ''
                    newprob = 1
                    newmass = 0
                    newmassnum = 0
                    for b in sorted(newiter):
                        for ir, ip, im, ie, iv in newiter[b]:
                            for n, c in enumerate(iv):
                                if c > 0:
                                    iso = fragmentpositions[ie][n]
                                    newmassnum += int(iso[1:]) * c
                                    seformula += f'{iso}({c})'
                                    #seformula += f'{fragmentpositions[ie][n]}({c})'
                            newprob *= ip
                            newmass += im
                if newprob >= cutoff:
                    #finalabundances[seformula] = [newmass, newprob]
                    try:
                        index = massnumberindices[newmassnum]
                        subformulas[index] += '-' + seformula
                        sumabundances[index][0] += newmass * newprob
                        sumabundances[index][1] += newprob
                    except KeyError: #not in there
                        index = len(massnumberindices)
                        massnumberindices[newmassnum] = index
                        subformulas.append(seformula)
                        sumabundances.append([newmass * newprob, newprob])
                    if multielement:
                        ind = bisect(probabilityranking, newratio)
                        probabilityranking.insert(ind, newratio)
                        multinomialpath.insert(ind, (newratio, *multipath))
                    else: #this is rarely ever needed, but it is needed
                        newmulti = []
                        tsv = tuple(sv)
                        #should this one be first? does it matter? i don't believe it does
                        if tsv not in vectorpool:
                            newmulti.append((sr, sp, sm, se, sv))
                            vectorpool.add(tsv)
                        tvv = tuple(v)
                        if tvv not in vectorpool:
                            newmulti.append((r, p, m, e, v))
                            vectorpool.add(tvv)
                        if newmulti:
                            ind = bisect(probabilityranking, newratio)
                            probabilityranking.insert(ind, newratio)
                            multinomialpath.insert(ind, (newratio, *newmulti))
            else:
                break

    #subformulas, massesandabundances = list(zip(*finalabundances.items()))
    subformulas = np.array(subformulas, dtype='S')
    #massesandabundances = np.array(massesandabundances).transpose()
    #massesandabundances = np.array(massesandabundances) #this is smaller in memory footprint somehow when its [[m, a], [m, a]] rather than [[m, m], [a, a]] for larger arrays
    massesandabundances = np.array(sumabundances)
    massesandabundances[:,0] /= massesandabundances[:,1]
    #sorting by mass
    subformulas = subformulas[massesandabundances[:,0].argsort()].tolist()
    massesandabundances = massesandabundances[massesandabundances[:,0].argsort()]
    return subformulas, massesandabundances

def fragmentation_compositions(seq):
    fragments = {}

    #calculate the compositions of the n-term fragments
    fragcomp_n = {}
    for n, aa in enumerate(seq[:-1]):  
        aa_composition = aminoacidcomposition[aa]
        for k in aa_composition:
            fragcomp_n[k] = fragcomp_n.get(k, 0) + aa_composition.get(k, 0)
        for ion, modcomp in ndict.items():
            fragment_composition = fragcomp_n.copy()
            for k in modcomp:
                fragment_composition[k] = fragment_composition.get(k, 0) + modcomp.get(k, 0)
            fragments[ion + str(n + 1)] = fragment_composition

    #calculate the compositions of the c-term fragments
    fragcomp_c = {}
    for n, aa in enumerate(seq[::-1][:-1]): 
        aa_composition = aminoacidcomposition[aa]
        for k in aa_composition:
            fragcomp_c[k] = fragcomp_c.get(k, 0) + aa_composition.get(k, 0)
        for ion, modcomp in cdict.items():
            fragment_composition = fragcomp_c.copy()
            for k in modcomp:
                fragment_composition[k] = fragment_composition.get(k, 0) + modcomp.get(k, 0)
            fragments[ion + str(n + 1)] = fragment_composition

    return fragments

#this only catches nearest 2 at most
def nearest_neighbors_ppm_tolerance(baselist, flylist):
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
                return indices

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
    return indices

def timeshift(experimental, theoretical):
    #normalize both arrays to preserve their relationship
    tlen = len(theoretical)
    ratiolen = tlen - 1
    thmean = sum(theoretical) / tlen
    exmean = sum(experimental) / tlen
    scalefactor = thmean / exmean
    exnorm = [e * scalefactor for e in experimental]
    
    #calculate relative differences
    relativedifferences = [abs((e - t) / t) for e, t in zip(exnorm, theoretical)]
    
    #calculate ratios between consecutive points
    thratio = [theoretical[i+1] / theoretical[i] for i in range(ratiolen)]
    exratio = [exnorm[i+1] / exnorm[i] for i in range(ratiolen)]
    
    #compare ratios
    ratiodiffs = [abs(t - e) / t for t, e in zip(thratio, exratio)]
    
    #combine relative differences and ratio differences
    reldiffmean = sum(relativedifferences) / tlen
    ratiodiffmean = sum(ratiodiffs) / ratiolen
    combinationdiffs = (reldiffmean + ratiodiffmean) / 2
    
    return combinationdiffs

def difference_maximization(arr, double):
    #function for refined two-phase greedy difference maximization
    sarr, sdouble = map(list, zip(*sorted(zip(arr, double))))
    
    #initialize sequence with the largest and smallest elements
    sequence = [sarr.pop(0), sarr.pop(-1)]
    sequencedouble = [sdouble.pop(0), sdouble.pop(-1)]
    
    while sarr:
        #compute the difference for adding either the next smallest or next largest element
        min_value, max_value = sarr[0], sarr[-1]
        
        #compare adding to both ends with the smallest and largest elements
        add_to_left_diff_min = abs(min_value - sequence[0])
        add_to_right_diff_min = abs(min_value - sequence[-1])
        
        add_to_left_diff_max = abs(max_value - sequence[0])
        add_to_right_diff_max = abs(max_value - sequence[-1])
        
        #decide to place the minimum or maximum value based on the maximum possible gain
        if add_to_left_diff_min >= add_to_right_diff_min and add_to_left_diff_min >= add_to_left_diff_max and add_to_left_diff_min >= add_to_right_diff_max:
            sequence.insert(0, sarr.pop(0))
            sequencedouble.insert(0, sdouble.pop(0))
        elif add_to_right_diff_min >= add_to_left_diff_min and add_to_right_diff_min >= add_to_left_diff_max and add_to_right_diff_min >= add_to_right_diff_max:
            sequence.append(sarr.pop(0))
            sequencedouble.append(sdouble.pop(0))
        elif add_to_left_diff_max >= add_to_right_diff_max and add_to_left_diff_max >= add_to_left_diff_min and add_to_left_diff_max >= add_to_right_diff_min:
            sequence.insert(0, sarr.pop(-1))
            sequencedouble.insert(0, sdouble.pop(-1))
        else:
            sequence.append(sarr.pop(-1))
            sequencedouble.append(sdouble.pop(-1))
    
    return sequence, sequencedouble

def sequence_geometry(seq, ioncoverage):
    slen = len(seq)
    maxncoverage = 0
    maxccoverage = 0
    dividers = set()
    ntermcoverage = []
    ctermcoverage = []
    for ion in ioncoverage:
        iontype = ion[0]
        ioncount = int(ion[1:])
        if iontype in 'abc': #nterm
            dividers.add(ioncount)
            pseq = seq[:ioncount]
            ntermcoverage.append(ioncount)
            if ioncount > maxccoverage:
                maxccoverage = ioncount
        elif iontype in 'xyz': #cterm
            dividers.add(slen - ioncount)
            pseq = seq[slen-ioncount:]
            ctermcoverage.append(slen-ioncount)
            if ioncount > maxncoverage:
                maxncoverage = ioncount
    dividers = sorted(dividers)
    if maxncoverage + maxccoverage >= slen:
        coverageweight = 1 / slen
    else:
        coverageweight = 1 / (maxncoverage + maxccoverage)

    #isolation counts need to be robust against redundant pseqs
    ind = 0
    ddiff = np.diff(dividers, prepend=0).tolist()
    #dividerstring = ''
    partialseqs = defaultdict(int) #index-pseq: count #the index safeguards against multiple isolations of the same partial sequence, the defaultdict rather than a Counter keeps the keys in order so i can view it easier in regards to the order of the sequence
    for d in ddiff:
        pseq = seq[ind:ind+d]
        #dividerstring += pseq + '|'
        ntermcovers = [i for i in ntermcoverage if i > ind]
        ctermcovers = [i for i in ctermcoverage if i <= ind]
        covers = len(ntermcovers) + len(ctermcovers)
        if covers > 0:
            label = str(ind) + '-' + pseq
            partialseqs[label] += covers
        ind += d
    pseq = seq[ind:]
    #dividerstring += pseq
    ntermcovers = [i for i in ntermcoverage if i > ind]
    ctermcovers = [i for i in ctermcoverage if i <= ind]
    covers = len(ntermcovers) + len(ctermcovers)
    if covers > 0:
        label = str(ind) + '-' + pseq
        partialseqs[label] += covers
    pairsum = 0
    isolationlengthweight = 1
    for indseq, count in partialseqs.items():
        ind, pseq = indseq.split('-')
        ind = int(ind)
        #i could use this index to weight based on distance from the ends i guess?
        #isolationlengthweight *= len(pseq) / len(seq)
        isolationlengthweight *= 1 / len(pseq) / len(seq) #this 1 / provides an additional layer of geometric success to this scheme, grants success where there was previously failure
    dividerweight = 1 / len(partialseqs)
    return dividerweight * isolationlengthweight * coverageweight

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

def shuffle_string(s):
    char_list = list(s)
    random.shuffle(char_list)
    return ''.join(char_list)

def unique_permutations_count(sequence):
    # Count the frequency of each element in the sequence
    freq = Counter(sequence)

    # Calculate the total number of permutations
    total_permutations = math.factorial(len(sequence))

    # Divide by the factorial of the frequency of each element to account for repetitions
    for count in freq.values():
        total_permutations //= math.factorial(count)

    return total_permutations

pid = 0
proteins = {} #proteinid: sequence
for protein in range(nproteins):
    length = np.random.randint(minproteinlength, maxproteinlength-1)
    seq = 'M' + ''.join((np.random.choice(list(aminoacidcomposition), length).tolist()))
    proteins[pid] = seq
    pid += 1

peptidesbyprotein = seqsplit_dict(proteins, cut, minpeptidelength, maxpeptidelength, missedcleavages)
peptides = list(set(itertools.chain(*peptidesbyprotein.values())))

proteinsbypeptide = defaultdict(list)
for protein, peps in peptidesbyprotein.items():
    for pep in peps:
        proteinsbypeptide[pep].append(protein)

def formula_consolidation(seq):
    atomiccomposition = Counter()
    for aa in seq:
        atomiccomposition += aminoacidcomposition[aa]
    #no OH loss on last residue, no H lost on first residue
    atomiccomposition['H'] += 2
    atomiccomposition['O'] += 1
    formulastring = ''.join((''.join((k, str(v))) for k, v in atomiccomposition.items()))
    return formulastring, atomiccomposition, seq

nt = time()

formulasbyseq = {} #seq: formula
atomiccompositions = {} #formula string: atomic composition dict
seqsbyformula = defaultdict(list) #formula string: [seqs]
with mp.Pool(nprocs) as pool:
    for formulastring, atomiccomposition, seq in pool.map(formula_consolidation, peptides):
        formulasbyseq[seq] = formulastring
        seqsbyformula[formulastring].append(seq)
        atomiccompositions[formulastring] = atomiccomposition

print(time() - nt, len(formulasbyseq), 'sequences consolidated')

#it will be important to test the power of +/-% variation in these MS1 isotopomers vs my distribution matching scheme
#getting these %'s from real data would be a great step, then applying that % here later to check how it does

t3 = time()

abundances = {}
with mp.Pool(nprocs) as pool:
    formulastrings, subformulas, massesandabundances = zip(*pool.starmap(distribution_generation, atomiccompositions.items()))

abundances = dict(zip(formulastrings, massesandabundances))
abundanceformulas = dict(zip(formulastrings, subformulas))

print(time() - t3, len(abundances), 'isotopic distributions total')
nt = time()

sumabundances = {} #formula: [[wmean masses], [sumabundances]]
subisodepthqualifiers = {} #formula: [[top n subisos of proton locations]]
condensationcoordinates = {} #formula: [# isotopomers per proton-step]
for formula, distribution in abundances.items():
    subformulas = abundanceformulas[formula]
    massgroups = defaultdict(list) #massnumber: [masses]
    intensitygroups = defaultdict(list) #massnumber: [abundances]
    masses, intensities = distribution
    for n, s in enumerate(subformulas):
        s = s.decode()
        massnumber = 0
        for ss in s.split(')')[:-1]:
            i1, i2 = map(int, ss[1:].split('('))
            massnumber += i1 * i2
        massgroups[massnumber].append(masses[n])
        intensitygroups[massnumber].append(intensities[n])
    meansofmasses = []
    sumsofabundances = []
    subisodepthindices = [] #coordinates that reset for each proton location
    condensationindices = []
    for mn, m in massgroups.items():
        condensationindices.append(len(m))
        a = intensitygroups[mn]
        totalabundance = sum(a)
        weightedmass = 0
        cumulativeabundance = 0
        subinds = []
        for n, (sm, sa) in enumerate(zip(m, a)):
            weightedmass += (sm * sa) / totalabundance
            cumulativeabundance += sa
            if cumulativeabundance / totalabundance >= subisotopomericdepth:
                #requires intensity being sorted by abundance
                subinds.append(n)
        meansofmasses.append(weightedmass)
        sumsofabundances.append(totalabundance)
        subisodepthindices.append(tuple(subinds))
    sumabundancedist = np.array([meansofmasses, sumsofabundances])
    sumabundances[formula] = sumabundancedist
    condensationcoordinates[formula] = np.array(condensationindices)
    subisodepthqualifiers[formula] = subisodepthindices

print(time() - nt, 'sum abundances made')

#group peptides by protein -> change to seqsplit_dict
#pick what peptide will be present and make the distributions
peptideanalytes = random.sample(peptides, nanalytes)
analytedistributions = defaultdict(lambda: [[], []]) #analyte id: [[wmean of ordered masses across isos across charge states], [AUC of merged isotopomers]]
analytesbydistribution = {} #distid: analyteid
linesofanalytes = {} #analyteid: [[lines across charge states at this position]]

#for the sake of this simulation i'll only make distributions that have their highest isotopomer first, there isn't a need for a deep enough simulation yet to warrant caring about this
peaks = {} #distid: [peakids]
trackedgroups = {} #lineid: [[masses], [rts], [intensities]]
chargesoflines = {} #lineid: charge
distributionsoflines = defaultdict(list) #lineid: distids
distributioncharges = {} #distid: charge
linesofdistributions = defaultdict(list) #distid: [lines]
positionsoflines = defaultdict(list) #distid: [proton locations as positions of the sumabundance dict they came from]
analytekeys = defaultdict(dict) #analyte id: distid: charge
regions = [] #[target mass, mintime, maxtime, lineid, distid, analyteid, charge]

peptidesofanalytes = {} #analyteid: sequence
peptidesofdistributions = {} #distid: peptide
peptidesoflines = {} #line: peptide
formulasbydist = {} #distid: formula

lineid = 0
distid = 0
for analyteid, peptide in enumerate(peptideanalytes):
    formula = formulasbyseq[peptide]
    peptidesofanalytes[analyteid] = peptide
    sumabundancedistribution = sumabundances[formula]
    meanmasses, summedintensities = sumabundancedistribution
    #pick a top N of points
    if len(sumabundancedistribution) >= maxisotopomers:
        maxisos = maxisotopomers
    else:
        maxisos = len(sumabundancedistribution)
    ntopisos = np.random.randint(minisotopomers, maxisos+1)
    rt = np.random.choice(retentiontimes)
    rtind = np.where(retentiontimes == rt)[0][0]
    initialcharge = random.choices(list(charges.keys()), weights=charges.values())[0]
    ncharges = random.choices(list(chargestates.keys()), weights=chargestates.values())[0]
    cdirection = np.random.randint(2) #1 up, 0 down
    chargeinds = {} #charge: [indices of visible masses]
    while len(chargeinds) < ncharges:
        newmasses = (meanmasses + proton * initialcharge) / initialcharge
        #why were these like this? i think it was wrong
        #minbound = meanmasses >= massmin
        #maxbound = meanmasses <= massmax
        minbound = newmasses >= massmin
        maxbound = newmasses <= massmax
        if minbound.any() and maxbound.any():
            chargeinds[initialcharge] = np.where(np.logical_and(minbound, maxbound))[0]
        else:
            break
        if cdirection > 0: #charge increases
            initialcharge += 1
        else: #charge decreases
            initialcharge -= 1
        if initialcharge < 1:
            break
    
    distcheck = False
    for charge, massinds in chargeinds.items():
        currentmasses = (meanmasses[massinds] + proton * charge) / charge #no mass variation, not needed yet
        nisos = np.random.randint(minisotopomers, ntopisos+1)
        ntopisos = nisos
        ndatapoints = np.random.randint(minpoints, maxpoints)
        seedintensity = np.random.uniform(minintensity, maxintensity)
        baseintensities = summedintensities[massinds]
        distintensities = baseintensities * np.random.uniform(1-intensityvariation, 1+intensityvariation, size=len(massinds))
        datapoints = (ndatapoints * distintensities).astype(int)
        distributioncharges[distid] = charge
        for n, (d, m, i) in enumerate(zip(datapoints, currentmasses, distintensities)):
            if d < minpoints:
                #plugging in a minlinepoints concept here
                #1 is too low for the array funcions below
                break
            halfd = d // 2
            lsub = rtind - halfd
            if lsub < 0:
                lp = rtind
            else:
                lp = halfd
            radd = rtind + halfd
            if radd > rtpoints - 1:
                rp = rtpoints - rtind - 1
            else:
                rp = halfd
            rts = retentiontimes[rtind-lp:rtind+rp]
            
            # based gpt telling me how to generate a peak of area X
            b = rts.mean()  # Mean (center of the peak)
            w = d #width factor of the peak
            c = (rts.max() - rts.min()) / w  # Standard deviation (width of the peak)
            # Find amplitude 'a' for the given desired area
            a = (i * seedintensity) / (np.sqrt(2 * np.pi) * c)
            # Generate Gaussian peak using the Gaussian function
            points = a * np.exp(-((rts - b) ** 2) / (2 * c ** 2))
            # Calculate the actual area under the generated Gaussian peak for verification
            pointvariation = np.random.uniform(1-intensityvariation, 1+intensityvariation, size=points.size)
            points *= pointvariation
            auc = np.trapezoid(points, rts)
            #check = (i * seedintensity) / auc
            #print(check) #it only fails when the number of datapoints is small, like d=2
            
            nonzerointensities = points > 0
            points = points[nonzerointensities]
            rts = rts[nonzerointensities]
            masses = np.repeat(m, points.size) #static until i need this to be more complex
            regions.append([m, min(rts), max(rts), lineid, distid, analyteid, charge, auc])
            trackedgroups[lineid] = np.array([masses, rts, points])
            distributionsoflines[lineid] = distid
            linesofdistributions[distid].append(lineid)
            positionsoflines[distid].append(n)
            analytekeys[analyteid][distid] = charge
            chargesoflines[lineid] = charge
            formulasbydist[distid] = formula
            peptidesoflines[lineid] = peptide
            lineid += 1
            distcheck = True
        if distcheck:
            analytesbydistribution[distid] = analyteid
            peptidesofdistributions[distid] = peptide
            distid += 1
    if len(chargeinds) > 1:
        chargeintensities = np.sort(np.random.uniform(size=len(analytekeys[analyteid])))[::-1]
        for n, did in enumerate(analytekeys[analyteid]):
            for line in linesofdistributions[did]:
                trackedgroups[line][2] *= chargeintensities[n]
        #make analytedistributions
    if distcheck:
        #i'm just going to use the real masses and not take a weighted mean, i don't have a need for it yet
        analyteorganizer = defaultdict(lambda: defaultdict(float)) #iso position: rt: abundance
        analytelineorganizer = defaultdict(list) #position: [lineuids]
        for did, charge in analytekeys[analyteid].items():
            lines = linesofdistributions[did]
            for line in lines:
                masses, rt, intensities = trackedgroups[line]
                basemass = (masses[0] * charge) - (proton * charge)
                position = np.abs(meanmasses - basemass).argmin()
                for m, r, i in zip(*trackedgroups[line]):
                    analyteorganizer[position][r] += i
                analytelineorganizer[position].append(line)
        linesofanalytes[analyteid] = []
        for position, adict in analyteorganizer.items():
            times, intensities = list(zip(*adict.items()))
            auc = np.trapezoid(intensities, times)
            #these masses seem to come out sorted so i don't need to sort them
            analytedistributions[analyteid][0].append(meanmasses[position])
            analytedistributions[analyteid][1].append(auc)
            linesofanalytes[analyteid].append(analytelineorganizer[position])
        analytedistributions[analyteid] = np.array(analytedistributions[analyteid])

regions = np.array(regions)
regions = regions[regions[:,3].argsort()] #sorted by line index

#cols = dp.get_colors(len(analytekeys))
#for n, (analyteid, dists) in enumerate(analytekeys.items()):
#    for distid, charge in dists.items():
#        for lineid in linesofdistributions[distid]:
#            masses, rts, points = trackedgroups[lineid]
#            plt.plot(masses, rts, '.', color=cols[n], markersize=1)
#plt.show()

outratios = set()
for distid, (masses, rts, intensities) in trackedgroups.items():
    if len(rts) != len(masses):
        outratios.add(len(masses) / len(rts))
        print(distid)
print(len(outratios))

#apply scans to analytes
nsamples = int(scansperdistribution * len(linesofdistributions))
samples = np.random.choice(list(linesofdistributions), size=nsamples)

areas = {} #lineid: area, #I'll use this to add more random behavior to the spectra generation
allintensities = set()
for line, (masses, rts, intensities) in trackedgroups.items():
    allintensities.update(intensities)
    area = np.trapezoid(intensities, rts)
    areas[line] = area
allintensities = np.sort(list(allintensities))

areasorts = sorted(areas.items(), key=lambda x: x[1])
arearanks = {} #lineid: rank
for n, (k, a) in enumerate(reversed(areasorts)):
    arearanks[k] = n / len(areasorts) * maxpoints / minpoints

slope = (maxpoints - minpoints) / (allintensities[-1] - allintensities[0])
intercept = maxpoints - allintensities[-1] * slope
interpolatedpoints = dict(zip(allintensities, allintensities * slope + intercept + 100))

#insert ms2 library generation here
ionlist = list(ions)
ndict = {k: nfragmentcompositions[k] for k in ionlist if k in nfragmentcompositions}
cdict = {k: cfragmentcompositions[k] for k in ionlist if k in cfragmentcompositions}
#generate fragments, embed fragmentation patterns later, do random selection atm

#making the "rules"
#dictate pairs that are the most likely to fragment
#as each pair, where a fragment occurs, figure out which pairs NEXT to it are less likely o pop, and that would "bubble" up to make the breaking point the HIGHER prop breakage point next to it.
#a subtle buildup model
#ie if PE is a high prob breaker, and EA is a low prob one, then PEA would have an even higher likelihood of breaking at the PE as the EA is strong by comparison
#so set up every amino acid pair to have a prob, then see if the patterns you find in the entropy match what you originally had
#this doesn't handle different subisos with any tact though

#i'll do "individual bonders" first, then derive pair strengths from it
individualaminoacidstrengths = {} #aa: strength, as a prob, higher prob = stronger bond
for aa in aminoacidcomposition:
    individualaminoacidstrengths[aa] = np.random.uniform()

pairstrengths = {} #ordered string of the pair: strength
for pair in itertools.combinations_with_replacement(aminoacidcomposition, 2):
    if len(set(pair)) > 1:
        pairvalue = np.prod([individualaminoacidstrengths[i] for i in pair])
        discrepancy = np.random.uniform(-0.1*pairvalue,0.1*pairvalue)
        pairstrengths[''.join((sorted(pair)))] = pairvalue + discrepancy
        pairstrengths[''.join((sorted(pair)[::-1]))] = pairvalue - discrepancy
    else: #2 of the same
        pairstrengths[''.join((pair))] = np.prod([individualaminoacidstrengths[i] for i in pair])

peptidefrags = {} #peptide: {fragdict}
peptidefragprobs = defaultdict(dict) #peptide: ion: prob
#set up the probabilities that can be sampled below
for peptide in peptideanalytes:
    frags = fragmentation_compositions(peptide)
    peptidefrags[peptide] = frags
    aminopairs = [peptide[i:i+2] for i in range(len(peptide)-1)]
    bondstrengths = np.array([pairstrengths[i] for i in aminopairs])
    bondstrengths = bondstrengths / bondstrengths.sum()
    plen = len(peptide)
    workinglen = plen - 1
    fragionlist = list(frags)
    ionpairs = [[fragionlist[i], fragionlist[-i-1]] for i in range(len(fragionlist) // 2)] #this doesn't work if including every kind of fragment ion atm
    for p, ip in zip(bondstrengths.tolist(), ionpairs):
        pairprobs = np.random.uniform(size=2) / p
        #pairprobs = pairprobs / pairprobs.sum() #removing adds more clear distinction between higher/lower probs later
        for i, pp in zip(ip, pairprobs.tolist()):
            peptidefragprobs[peptide][i] = pp
    psum = sum(peptidefragprobs[peptide].values())
    for i, p in peptidefragprobs[peptide].items():
        peptidefragprobs[peptide][i] /= psum
    #each bondstrength at its INDEX applies to that POSITION for each fragment, ie 1 -> b1/y8
    #then a separate process determines whether the b/y will get the brunt of the probability, or all

ms2ppmvariance = 12 #ppm

ms2lists = defaultdict(list) #previous ms1 scan: [ms2 scan ids]
rtids = {} #ms2id: prior ms1 rt
ms2scans = {} #ms2id: [[masses], [intensities]]
ms2massidentities = defaultdict(lambda: defaultdict(list)) #ms2id: line: [masses]
ms2massesoflines = defaultdict(lambda: defaultdict(list)) #line: scan: [masses]
#ms2neighbors = {} #ms2id: current nn model
#ms2massindices = defaultdict(dict) #ms2id: index: [[masses], [intensities], [lineids]]
linesofms2 = defaultdict(list) #ms2id: [lines]
distsofms2 = defaultdict(list) #ms2id: [dists]
#ionsoflines = defaultdict(dict) #lineid: ms2id: [[masses], [intensities]]
precursorcoordinates = [] #[rt of previous ms1, lower mass bound, upper mass bound, ms2 scan index]
ms2scanbounds = {} #ms2 scan: [rt of previous ms1, lower mass bound, upper mass bound]
#linesinscans = defaultdict(set) #lineuid: [lines that have been added to ms2scans]

#these 2 below can have duplicate peptide/scans in their lists because of different isotopomers/lines being from the same dist aka peptide
peptidesofscans = defaultdict(list) #ms2 scan: [peptides]
scansofpeptides = defaultdict(list) #peptide: [ms2 scans]

linesbyprimaryind = {} #primary index: line
scansbyprimaryind = {} #primary ind: scan

ms2id = 0 #aka scan
primaryind = 0
for dist in samples.tolist():
    peptide = peptidesofdistributions[dist]
    charge = distributioncharges[dist]
    lines = linesofdistributions[dist]
    line = np.random.choice(lines)
    linesubisoposition = positionsoflines[dist][lines.index(line)]
    formula = formulasbydist[dist]
    subformulas = [i.decode() for i in abundanceformulas[formula]]
    conlengths = condensationcoordinates[formula]
    conends = conlengths.cumsum()
    constarts = conends - conlengths
    #for line in lines:
    linemasses, rts, lineintensities = trackedgroups[line]
    rtsample = np.random.choice(rts[:-1]) #the new scan will be after this one
    intensityind = np.where(rts == rtsample)[0][0]
    rtind = np.where(retentiontimes == rtsample)[0][0]
    targetmass = regions[line][0]
    rtids[ms2id] = rtind #this needs to be before the if statement
    #generate spectra
    #i'm going to work off the ms1 limitations until i need something more complicated
    #and i don't even need intensities yet, so whatever
    #ms2lists[rtind].append(ms2id)
    #rtids[ms2id] = rtind
    #ipoint = int(interpolatedpoints[lineintensities[intensityind]])
    #apoint = int(arearanks[line])
    #if apoint < ipoint:
    #    npoints = np.random.randint(apoint, ipoint)
    #elif ipoint < apoint:
    #    npoints = np.random.randint(ipoint, apoint)
    #else:
    #    npoints = apoint
    #targetmass = regions[line][0]
    #masses = np.random.uniform(massmin, massmax, size=npoints).tolist()
    #baseintensities = np.random.uniform(minintensity, trackedgroups[line][2][intensityind], size=npoints)
    #intensities = (baseintensities * (baseintensities / trackedgroups[line][2].max())).tolist()
    ##adapt fragmentation below to something more sequence-based
    #masses = np.random.uniform(massmin, massmax, size=npoints).tolist()
    #^make this into a random sample with the assigned %s
    #better yet, make each of the intensities select little pieces of the total intensity at their probabilities
    #the little pieces can be a list of generated values between 2 integers
    #the width of those 2 integers will dictate the effectiveness of this model as a concept, and whether the 2 values, that would hypothetically be derived from real data if you were to model a simulation at the same size, would be an indication of whether this model makes sense
    #i'll input a small distance between the 2 at first so the model works well
    #as random slices get larger it gets more sporadic, i wouldn't expect it to be too sporadic, but i guess scan-to-scan variability of the same analyte being sampled > 1x might be valuable information for determining this factor
    #and i do know it as a matter of fact, as you can see via PRM data the signal is very stable, fragment rankings are fairly consistent and only get slightly more inconsistent when there's a lot of small ones
    #so its almost like there's an area of stability and also instability
    #the stable area of fragmentation is the first ~2-7 or so identifiable fragment ions that rise to the top
    #the rest below it do come out in somewhat stable orders but if you have a lot of them that are ~proximally all just as unlikely then they can get the competition of noise down below the relevant signals
    #then focus on consistency first when it comes to the entropy concept
    #also, an uptick in noise that corresponds with an uptick in ms1 intensity of a line is an easy pick, basically
    #it might also be good to see just how much raw entropy can be represented properly via the % of total intensity in the ms2 spectrum vs ms1 intensities
    #overall you try to best match both the hypothetical entropy plus the intensity % of each line
    baseintensity = trackedgroups[line][2][intensityind].astype(int) #add random variation?
    fragrolls = np.random.multinomial(baseintensity, list(peptidefragprobs[peptide].values()), size=1)[0].tolist()
    fragmin = min(fragrolls)
    #if fragmin > 0:
    ms2lists[rtind].append(ms2id)
    masses = []
    intensities = []
    #fragions = []
    bi = constarts[linesubisoposition]
    for sq in subisodepthqualifiers[formula][linesubisoposition]:
        subindex = bi + sq
        subformula = subformulas[subindex]
        isocounts = set()
        competing = set()
        competitors = {}
        isosums = {}
        for ss in subformula.split(')')[:-1]:
            iso, c = ss.split('(')
            c = int(c)
            e = iso[0]
            if e in isocounts:
                competing.add(e)
                competitors[e][iso] = c
                isosums[e] += c
            else:
                isocounts.add(e)
                competitors[e] = {iso: c}
                isosums[e] = c
        isoprobs = defaultdict(dict) #e: probs of that element
        for e, v in competitors.items():
            if e in competing:
                for iso, c in v.items():
                    prob = c / isosums[e]
                    isoprobs[e][iso] = prob
            else:
                for iso in v:
                    isoprobs[e][iso] = 1
        for ion, roll in zip(peptidefragprobs[peptide], fragrolls):
            roll += 1
            fragcomp = peptidefrags[peptide][ion]
            elementalorganizer = {} #element: [[iso heaps]]
            fragmentpositions = {} #element: position: iso
            for e, c in fragcomp.items():
                if e in isoprobs: #element is in subformula
                    fragprobs = isoprobs[e]
                    if len(fragprobs) > 1:
                        elementlist, positions = fragment_element_binomial_walk(e, c, fragprobs)
                        elementalorganizer[e] = elementlist.copy()
                        fragmentpositions[e] = positions
                    else: #no need for cache, only 1 iso
                        #iso, c = list(fragprobs.items())[0] #WOOPS
                        iso = list(fragprobs)[0]
                        elementalorganizer[e] = [[-1, 1, elementalmasses[iso]*c, e, [c]]]
                        fragmentpositions[e] = {0: iso}
            fragformulas, massesandabundances = fragment_descending_partial_products(elementalorganizer, fragmentpositions)
            massesandabundances[:,0] += proton #making all ions are single-charge for now
            keepers = massesandabundances[:,1] * roll >= fragmin #simple intensity filter
            masses.extend(massesandabundances[keepers,0].tolist())
            intensities.extend((massesandabundances[keepers,1] * roll).tolist())
            #fragions.extend(itertools.repeat(ion, keepers.sum()))
    npoints = len(intensities)
    primaryinds = np.arange(npoints) + primaryind
    primaryinds = primaryinds.tolist()
    masses = np.array(masses)
    massvariation = masses / 1000000 * ms2ppmvariance / 2
    masses = masses + np.random.uniform(-massvariation, massvariation, size=masses.size)
    intensities = np.array(intensities)[masses.argsort()].tolist()
    masses = np.sort(masses).tolist()
    ms2scans[ms2id] = [masses, intensities]
    ms2massidentities[ms2id][line] = masses
    ms2massesoflines[line][ms2id] = masses
    peptidesofscans[ms2id].append(peptide)
    scansofpeptides[peptide].append(ms2id)
    for p in primaryinds:
        linesbyprimaryind[p] = line
        scansbyprimaryind[p] = ms2id
    primaryind += npoints
    precursorcoordinates.append([rtsample, targetmass-scanradius, targetmass+scanradius, ms2id])
    ms2scanbounds[ms2id] = [rtsample, targetmass-scanradius, targetmass+scanradius]
    ms2id += 1

t2 = time()

precursorcoordinates = sorted(precursorcoordinates, key=lambda x: x[0]) #sorted by rt
regiter = regions[regions[:,1].argsort()].tolist() #sorted by starting time

regioniter = iter(regiter)

regminrt = -1

linesofscans = defaultdict(list) #ms2 scan index: [lineuids within window]
scansoflines = defaultdict(list) #lineuid: [ms2 scan indices]

regpool = []
for pc in precursorcoordinates:
    prt = pc[0]
    pminmass = pc[1]
    pmaxmass = pc[2]
    precid = pc[3]
    while regminrt < prt: #add more regs to regpool
        #no region rts and precursor rts are the same values, don't need <=
        try:
            reg = next(regioniter)
        except StopIteration: #regiter reached the end before precursorcoordinates, which is within the realm of expectations
            break
        regminrt = reg[1]
        #regmaxrt = reg[2]
        regid = int(reg[3])
        regpool.append(regid)
    regremovals = []
    for r in regpool:
        treg = regions[r]
        trmaxrt = treg[2]
        if trmaxrt < prt:
            regremovals.append(r)
    for r in regremovals:
        regpool.remove(r)
    for r in regpool: #assess reg masses across pc masses
        treg = regions[r]
        trminmass = treg[0]
        trmaxmass = treg[0] #homogenous ms1 masses atm
        if pminmass <= trmaxmass and pmaxmass >= trminmass:
            tminrt = treg[1]
            if tminrt < prt:
                linesofscans[precid].append(r)
                scansoflines[r].append(precid)

for k, v in linesofscans.items():
    linesofscans[k] = tuple(sorted(v))
linesofscans = dict(linesofscans)

for k, v in scansoflines.items():
    scansoflines[k] = tuple(v)
scansoflines = dict(scansoflines)

print(time() - t2, 'determined line-window overlaps')
t3 = time()

#it might be worth visualizing these to see if these are linemodel errors
blankscans = len(precursorcoordinates) - len(linesofscans)
if blankscans > 0:
    blankpercent = blankscans / len(precursorcoordinates)
    print('your instrument produced', blankscans, f'MS2 scans that targeted nothing within the minimum point threshhold of {minpoints},', f'{round(blankpercent, 4)}% of all scans')

precursordict = {}
for pc in precursorcoordinates:
    precursordict[pc[-1]] = pc[:-1]

scansbyanalyte = defaultdict(list) #analyteid: [scans across all lines and charge states]
scanalytecharges = defaultdict(dict) #analyteid: scan: charge
#spectralsamplings = defaultdict(lambda: defaultdict(dict)) #scan: line: % by area of ms1 lines
lineintensitiesofscans = defaultdict(lambda: defaultdict(dict)) #scan: line: raw area of ms1 line, average of 2 flanking points
linepercentagesofscans = defaultdict(lambda: defaultdict(dict)) #scan: line: % of scan intensity input
scansums = defaultdict(float) #scan: sum area used in lineintensitiesofscans
#isotopomerpositionsofanalytes = defaultdict(set) #analyteid: isotopomer coordinate from max

#these 3 below are keeping track of which line from which distribution [at each charge] gives the most intense MS2 sampling based on MS1 intensity
maxintensitysampleofdists = defaultdict(float) #distid: max intensity
maxintensitylinesofdists = {} #distid: (line, scan)
premaxsampledistributionsoflines = {} #distid: line

for line, scans in scansoflines.items():
    try:
        distid = distributionsoflines[line]
        analyteid = analytesbydistribution[distid]
        #lines = linesofdistributions[distid]
        #maxline = regions[lines,5].argmax()
        #linecoordinate = lines.index(line) - maxline
        charge = analytekeys[analyteid][distid]
    except KeyError: #line is in nodists
        analyteid = -line
        distid = -1
        #linecoordinate = 0
        charge = 0
    scansbyanalyte[analyteid].extend(scans)
    linegroup = trackedgroups[line]
    linetimes = linegroup[:,1]
    linemasses = linegroup[:,0]
    lineintensity = linegroup[:,2]
    lmax = linemasses.max()
    lmin = linemasses.min()
    for scan in scans:
        pcoords = precursordict[scan]
        rt = pcoords[0]
        #this assumes there is a left and right intensity, i guess there would be though?
        #PROBLEM here, this assumes the time difference is the same between the two
        #^there should be a time-based extrapolation here
        #i'm leaving it for later because its simple and probably won't change much
        leftintensity = lineintensity[linetimes <= rt][-1]
        rightintensity = lineintensity[linetimes >= rt][0]
        sampleintensity = (leftintensity + rightintensity) / 2
        minmass = pcoords[1]
        maxmass = pcoords[2]
        if not lmin > minmass and lmax < maxmass:
            #if the overlap doesn't fully encompass all mass points, normalize by the % mass overlap
            #idc for slight mass shifts, i'm just going by the range, the shifts would be too annoying to incorporate, probably not worth my time
            #this assumes there's no realistic way for the line mass to fully encompass the scans mass window, which there shouldn't be unless the line model screws up
            if lmax > maxmass:
                percentoverlap = (maxmass - lmin) / (lmax - lmin)
            else:
                percentoverlap = (lmax - minmass) / (lmax - lmin)
            sampleintensity *= percentoverlap
        if distid >= 0:
            if sampleintensity > maxintensitysampleofdists[distid]:
                maxintensitysampleofdists[distid] = sampleintensity
                maxintensitylinesofdists[distid] = line, scan
                premaxsampledistributionsoflines[distid] = line
        lineintensitiesofscans[scan][line] = sampleintensity
        scansums[scan] += sampleintensity
        scanalytecharges[analyteid][scan] = charge
    #isotopomerpositionsofanalytes[analyteid].add(linecoordinate)
scansbyanalyte = dict(scansbyanalyte)

#turning areas into percents
for scan, lines in lineintensitiesofscans.items():
    #samplesum = 0
    #for analyteid, positions in analytes.items():
    #    for position, area in positions.items():
    #        samplesum += area
    #for analyteid, positions in analytes.items():
    #    for position, area in positions.items():
    for line in lines:
        linepercentagesofscans[scan][line] = lineintensitiesofscans[scan][line] / scansums[scan]
    linepercentagesofscans[scan] = dict(linepercentagesofscans[scan])
    lineintensitiesofscans[scan] = dict(lines) #can't pickle double default dicts
lineintensitiesofscans = dict(lineintensitiesofscans)
linepercentagesofscans = dict(linepercentagesofscans)

maxsampledistributionsoflines = {} #line: distid
for distid, line in premaxsampledistributionsoflines.items():
    maxsampledistributionsoflines[line] = distid

print(time() - t3, 'weighted scan window hits by intensity')

print(len(linesofscans), 'linesofscans')
print(len(scansoflines), 'scansoflines')

#adding other lines to ms2scans
for scan, lines in linesofscans.items():
    rtind = rtids[scan]
    for line in lines:
        if line not in ms2massidentities[scan]:
            dist = distributionsoflines[line]
            peptide = peptidesofdistributions[dist]
            linesubisoposition = positionsoflines[dist][linesofdistributions[dist].index(line)]
            formula = formulasbydist[dist]
            subformulas = [i.decode() for i in abundanceformulas[formula]]
            conlengths = condensationcoordinates[formula]
            conends = conlengths.cumsum()
            constarts = conends - conlengths
            #append other lines fragments to the scan
            linemasses, rts, lineintensities = trackedgroups[line]
            intensityind = np.where(rts == retentiontimes[rtind])[0][0]
            #rtids[scan] = rtind
            #ipoint = int(interpolatedpoints[lineintensities[intensityind]])
            #apoint = int(arearanks[line])
            #if apoint < ipoint:
            #    npoints = np.random.randint(apoint, ipoint)
            #elif ipoint < apoint:
            #    npoints = np.random.randint(ipoint, apoint)
            #else:
            #    npoints = apoint
            #masses = np.random.uniform(massmin, massmax, size=npoints).tolist()
            #baseintensities = np.random.uniform(minintensity, trackedgroups[line][2][intensityind], size=npoints)
            #intensities = baseintensities * (baseintensities / trackedgroups[line][2].max())
            #targetmass = regions[line][0]
            #primaryinds = np.arange(npoints) + primaryind
            #primaryinds = primaryinds.tolist()
            #ms2scans[scan][0].extend(masses)
            #ms2scans[scan][1].extend(intensities)
            #ms2scans[scan][2].extend(primaryinds)
            #ms2massidentities[scan][line] = masses
            #for p in primaryinds:
            #    linesbyprimaryind[p] = line
            #    scansbyprimaryind[p] = scan
            #primaryind += npoints
            baseintensity = lineintensities[intensityind].astype(int) #add random variation?
            fragrolls = np.random.multinomial(baseintensity, list(peptidefragprobs[peptide].values()), size=1)[0].tolist()
            fragmin = min(fragrolls)
            #if fragmin > 0:
            ms2lists[rtind].append(scan)
            masses = []
            intensities = []
            #fragions = []
            bi = constarts[linesubisoposition]
            for sq in subisodepthqualifiers[formula][linesubisoposition]:
                subindex = bi + sq
                subformula = subformulas[subindex]
                isocounts = set()
                competing = set()
                competitors = {}
                isosums = {}
                for ss in subformula.split(')')[:-1]:
                    iso, c = ss.split('(')
                    c = int(c)
                    e = iso[0]
                    if e in isocounts:
                        competing.add(e)
                        competitors[e][iso] = c
                        isosums[e] += c
                    else:
                        isocounts.add(e)
                        competitors[e] = {iso: c}
                        isosums[e] = c
                isoprobs = defaultdict(dict) #e: probs of that element
                for e, v in competitors.items():
                    if e in competing:
                        for iso, c in v.items():
                            prob = c / isosums[e]
                            isoprobs[e][iso] = prob
                    else:
                        for iso in v:
                            isoprobs[e][iso] = 1
                for ion, roll in zip(peptidefragprobs[peptide], fragrolls):
                    roll += 1
                    fragcomp = peptidefrags[peptide][ion]
                    elementalorganizer = {} #element: [[iso heaps]]
                    fragmentpositions = {} #element: position: iso
                    for e, c in fragcomp.items():
                        if e in isoprobs: #element is in subformula
                            fragprobs = isoprobs[e]
                            if len(fragprobs) > 1:
                                elementlist, positions = fragment_element_binomial_walk(e, c, fragprobs)
                                elementalorganizer[e] = elementlist.copy()
                                fragmentpositions[e] = positions
                            else:
                                iso = list(fragprobs)[0]
                                elementalorganizer[e] = [[-1, 1, elementalmasses[iso]*c, e, [c]]]
                                fragmentpositions[e] = {0: iso}
                    fragformulas, massesandabundances = fragment_descending_partial_products(elementalorganizer, fragmentpositions)
                    massesandabundances[:,0] += proton #making all ions are single-charge for now
                    keepers = massesandabundances[:,1] * roll >= fragmin #simple intensity filter
                    masses.extend(massesandabundances[keepers,0].tolist())
                    intensities.extend((massesandabundances[keepers,1] * roll).tolist())
            npoints = len(intensities)
            primaryinds = np.arange(npoints) + primaryind
            primaryinds = primaryinds.tolist()
            masses = np.array(masses)
            massvariation = masses / 1000000 * ms2ppmvariance / 2
            masses = masses + np.random.uniform(-massvariation, massvariation, size=masses.size)
            intensities = np.array(intensities)[masses.argsort()].tolist()
            masses = np.sort(masses).tolist()
            if scan in ms2scans:
                ms2scans[scan][0].extend(masses)
                ms2scans[scan][1].extend(intensities)
                #ms2scans[scan][2].extend(primaryinds)
            else:
                ms2scans[scan] = [masses, intensities]
            ms2massidentities[scan][line] = masses
            ms2massesoflines[line][scan] = masses
            peptidesofscans[scan].append(peptide)
            scansofpeptides[peptide].append(scan)
            for p in primaryinds:
                linesbyprimaryind[p] = line
                scansbyprimaryind[p] = scan
            primaryind += npoints

for k, v in ms2scans.items():
    ms2scans[k] = np.array(v)

#nothing interesting really
#things = 0
#for line, scandict in ms2massesoflines.items():
#    if len(scandict) > 1:
#        lens = []
#        mset = set()
#        for scan, masses in scandict.items():
#            lens.append(len(masses))
#            mset.update(masses)
#        print(lens, len(mset))
#        if len(mset) > max(lens):
#            things += 1

#ms1 distribution matching

nt = time()

librarykeys = []
librarymasses = []
librarymassdict = {} #lid: [masses]
librarypositions = {} #lid: [indices]
#libraryintensities = {} #lid: [intensities]
libraryintensityranks = {} #lid: [intensityranks]
#librarydirections = {} #lid: [increasing/decreasing, max=0]
for f, (masses, intensities) in sumabundances.items():
    librarymassdict[f] = masses
    librarypositions[f] = list(range(masses.size))
    #libraryintensities[k] = intensities
    intensityranks = np.abs(intensities.argsort().argsort() - intensities.size + 1)
    #maxloc = intensities.argmax()
    #leftdirections = (np.diff(intensities[:maxloc+1]) > 0).tolist()
    #rightdirections = (np.diff(intensities[maxloc:]) > 0).tolist()
    #directions = [1 if i else -1 for i in leftdirections] + [0] + [1 if i else -1 for i in rightdirections]
    #librarydirections[k] = directions
    libraryintensityranks[f] = intensityranks
    librarykeys.extend(itertools.repeat(f, masses.size))
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
                    for lines, pos in zip(distlines, positions):
                        for line in lines:
                            if line in scansoflines:
                                #analytesbyformula[formula].add(dk)
                                #linepositionsofanalytes[dk][pos].add(line)
                                #looks like the right way to do this is:
                                #formula: position: line
                                linepositionsbyformula[lk][pos].add(line)
                    tx += 1
        if tx > 0:
            lmatches += tx
            dmatches += 1

for k, v in linepositionsbyformula.items():
    for sk, sv in v.items():
        v[sk] = tuple(sv)
    linepositionsbyformula[k] = dict(v)
linepositionsbyformula = dict(linepositionsbyformula)

print(time() - nt, 'matches assembled')
print('library matches:', lmatches)
print('dist matches:', dmatches)

success = []
failure = []
for analyteid, lkeys in librarymatchesbyanalyte.items():
    if any([peptidesofanalytes[analyteid] in seqsbyformula[formula] for formula in lkeys]):
        success.append(analyteid)
    else:
        failure.append(analyteid)

#measure how ~10%, etc, random variation changes the rank order of theoretical distribution abundances in this simulation
#and can you catch them with the 1 allowance n-1 dallowance scheme? or something by lrange?

slen = len(success)
flen = len(failure)
print(slen, 'ms1 success')
print(flen, 'ms1 failure')
print(slen / (slen + flen), '%')
#GOOD! failures are small, but they still exist with 0 intervention, why?
#one analytedistribution had a negative value for an intensity slot
#^not sure why, the trackedgroups/distribution info for it was fine
#^i'm guessing the other errors are of the same nature but i've only found intensity rank mismatches from a brief inspection
#^i'll overhaul it at some point but its not a priority atm, its too rare to care
#i'm assuming i'm getting less real matches than the total number i chose to inject because they only have 1 line to match and the above scheme requires > 1

nt = time()

formulabysortedseq = {} #sortedseq: formula
seqsbysortedseq = defaultdict(set) #sortedseq: [seqs]
for formula, seqs in seqsbyformula.items():
    for seq in seqs:
        sortedseq = ''.join((sorted(seq)))
        seqsbysortedseq[sortedseq].add(seq)
        formulabysortedseq[sortedseq] = formula

#for now this only handles trytic peptides
#but i should note if c or n terminal AAs are relevant to the digest and handle it here
#make them kwargs and default them to false, it should either be a [0] or [-1] slice, if they exist then slice those out first via case/match
fulldecoyset = set() #all decoy sequences
seqswithdecoysbyformula = defaultdict(list) #formula: [seqs + decoys]
for sortedseq, seqs in seqsbysortedseq.items():
    decoys = set()
    slen = len(seqs)
    #make seqgroups based on first/last AA depending on enzyme
    seqgroups = defaultdict(lambda: defaultdict(list)) #position (0 or -1): AA: [seqs]
    for seq in seqs:
        #this would need to be:
        #if seq.startswith/endswith and make double groups for when both AAs apply
        seqgroups[-1][seq[-1]].append(seq)
    for position, aas in seqgroups.items():
        for aa, subseqs in aas.items():
            #if len(aa) > 1: double-group i suppose?
            initialseq = subseqs[0][:-1]
            setlen = len(set(initialseq))
            if setlen > 1:
                subdecoys = set()
                sublen = len(subseqs)
                permax = unique_permutations_count(initialseq) #the -1 is considering K or R ending is consistent
                for seq in subseqs:
                    #tryptic only atm
                    endchar = seq[-1]
                    shortseq = seq[:-1]
                    while True:
                        decoy = shuffle_string(shortseq) + endchar
                        if decoy not in subdecoys and decoy not in decoys and decoy not in seqs:
                            subdecoys.add(decoy)
                            break
                        if len(subdecoys) + sublen == permax:
                            #all potential sequences already made
                            break
                decoys.update(subdecoys)
            #else: #setlen == 1 and sublen == 1
                #the sequence only has one AA, no decoys possible, whatever
                #break
    seqs.update(decoys)
    fulldecoyset.update(decoys)
    seqswithdecoysbyformula[formulabysortedseq[sortedseq]].extend(seqs.copy())

seqswithdecoysbyformula = dict(seqswithdecoysbyformula)
print(time() - nt, 'decoys generated')

nt = time()

probtracker = {} #prob string: prob index

probabilityorganizer = defaultdict(dict) #prob index: iso: prob
#^there's still some redundancy in here, 99%+ of it is carbon. the reason is that two different subformula compositions can form the same ratios/probabilities, its not a big deal tbh, the dict is less than 2000 in length
matchprobabilities = defaultdict(list) #subformula: [prob indices] #subformula here instead of match index bc the prob comp is tied to subformulas

subformulasubindices = defaultdict(list) #subformula: [sub match indices]
submatchsequences = {} #submatchindex: sequence
elementsofprobabilityindices = {} #prob index: e

linesbysubformula = defaultdict(set) #subformula: [lines that have ms2 scans]
subformulapercent = defaultdict(dict) #subformula: sequence: (subiso abundance rank, subiso abundance)
subformulasofsequencedistribution = defaultdict(dict) #dist: seq: subformula

group = [] #a single dividedgroup

probindex = 0
submatchindex = 0
mainmatchindex = 0
for formula, positions in linepositionsbyformula.items():
    qualifiers = subisodepthqualifiers[formula]
    conlengths = condensationcoordinates[formula]
    conends = conlengths.cumsum()
    constarts = conends - conlengths
    subformulas = [i.decode() for i in abundanceformulas[formula]]
    massesandintensities = abundances[formula]
    theoreticalabundances = massesandintensities[1]
    for position, lines in positions.items():
        for seq in seqswithdecoysbyformula[formula]:
            bi = constarts[position]
            for qualrank, sq in enumerate(qualifiers[position]):
                subindex = bi + sq
                sformula = subformulas[subindex]
                subformulapercent[sformula][seq] = qualrank, theoreticalabundances[subindex]
                linesbysubformula[sformula].update(lines)
                for line in lines:
                    if qualrank == 0:
                        if line in maxsampledistributionsoflines:
                            distid = maxsampledistributionsoflines[line]
                            subformulasofsequencedistribution[distid][seq] = sformula
                    group.append(sformula)
                    group.append(seq)
                subformulasubindices[sformula].append(submatchindex)
                submatchsequences[submatchindex] = seq
                submatchindex += 1
                if sformula not in matchprobabilities:
                    #setting up subformula-specific probabilities
                    isocounts = set()
                    competing = set()
                    competitors = {}
                    isosums = {}
                    for ss in sformula.split(')')[:-1]:
                        iso, c = ss.split('(')
                        c = int(c)
                        e = iso[0]
                        if e in isocounts:
                            competing.add(e)
                            competitors[e][iso] = c
                            isosums[e] += c
                        else:
                            isocounts.add(e)
                            competitors[e] = {iso: c}
                            isosums[e] = c
                    for e, v in competitors.items():
                        isoprobs = {}
                        if e in competing:
                            for iso, c in v.items():
                                prob = c / isosums[e]
                                isoprobs[iso] = prob
                            probstring = '/'.join(('/'.join((k, str(v))) for k, v in isoprobs.items()))
                            if probstring in probtracker:
                                foundprobindex = probtracker[probstring]
                                matchprobabilities[sformula].append(foundprobindex)
                            else:
                                probtracker[probstring] = probindex
                                probabilityorganizer[probindex] = isoprobs
                                matchprobabilities[sformula].append(probindex)
                                elementsofprobabilityindices[probindex] = e
                                probindex += 1
                        else:
                            #don't need to make a new index for every time something has no competition
                            for iso in v:
                                isoprobs[iso] = 1
                            if e not in probabilityorganizer:
                                probstring = tuple(isoprobs.items())
                                probtracker[probstring] = e
                                probabilityorganizer[e] = isoprobs
                                elementsofprobabilityindices[e] = e
                            matchprobabilities[sformula].append(e)
    mainmatchindex += 1

probabilityorganizer = dict(probabilityorganizer)
matchprobabilities = dict(matchprobabilities)
subformulasubindices = dict(subformulasubindices)

linesbyscanbysubformula = {} #subformula: scan: [lines]
for sformula, lines in linesbysubformula.items():
    linesbyscan = defaultdict(list)
    for line in lines:
        for scan in scansoflines[line]:
            linesbyscan[scan].append(line)
    for k, v in linesbyscan.items():
        linesbyscan[k] = tuple(v)
    linesbyscan = dict(linesbyscan)
    linesbyscanbysubformula[sformula] = linesbyscan

for subformula, seqs in subformulapercent.items():
    subformulapercent[subformula] = dict(subformulapercent[subformula])

print(time() - nt, 'submatch organization')
nt = time()

scanlist = set()
for subformula, scans in linesbyscanbysubformula.items():
    scanlist.update(scans)

scoredms2scans = {} #scan: [[masses], [intensities]]
for scan in list(scanlist):
    scoredms2scans[scan] = ms2scans[scan].tolist()

intensityaverages = {} #scan: average intensity
for scan, (masses, intensities) in scoredms2scans.items():
    intensityaverages[scan] = np.mean(intensities)

aminomasses = {k: sum(elementalmasses[monoisotopickeys[i]] * j for i, j in comp.items()) for k, comp in aminoacidcomposition.items()}
water = elementalmasses['H1'] * 2 + elementalmasses['O16']

massarray = np.array(list(aminomasses.values()))

#scoring ions of scans:
#min/max mass range via amino acids for the MS2 range, any AAs can be used
#use these AAs to then determine a potential delimiter process for dividing the ions into "groups", 100da straight might be too simple
#top ion is 1 / (number of ions in group being tested), which is the most intense
#15 ions becomes: 1/15, 2/14, 3/13, etc as they decrease in intensity
#OR maybe it can just be that ion's percentage of the total ions in that group

fullmaxmass = 0
for mza, intensities in scoredms2scans.values():
    if max(mza) > fullmaxmass:
        fullmaxmass = max(mza)

masslevels = []
masslevels.append(massarray.round())
for _ in range(np.ceil(round(fullmaxmass) / massarray.min()).astype(int) - 1):
    newlevel = (masslevels[-1] + massarray[:,None]).flatten()
    roundlevel = np.round(newlevel).astype(int)
    newlevel = np.unique(roundlevel)
    masslevels.append(newlevel)

levelranges = [[i.min(), i.max()] for i in masslevels]
flatranges = np.sort(list(itertools.chain.from_iterable(levelranges)))
#it's going to be worth testing other matrices
# - cutting levelranges off once it hits maxmass and taking all those indices might work too, limiting to 16 again
# - check the raw 100 distance

nlowers = []
for scan, (mza, intensities) in scoredms2scans.items():
    maxmass = max(mza)
    minmass = min(mza)
    mza = np.array(mza)
    intensities = np.array(intensities)
    
    scanranges = flatranges[flatranges <= maxmass]
    
    firstind = (scanranges <= minmass).sum() - 1
    scanranges = scanranges[firstind:]
    scanranges[0] = np.floor(minmass)
    scanranges = scanranges.astype(int).tolist()
    
    #i'm hard-coding 16 to be the smallest range, its the most common difference from the matrix of differences of massarray from itself, and it seems like a reasonable minimum i suppose
    while True:
        removal = False
        for n in range(len(scanranges)-1):
            l = scanranges[n]
            r = scanranges[n+1]
            diff = r - l
            if diff < 16:
                removal = True
                break
        if removal:
            scanranges.remove(r)
        else:
            break
    
    maxmass = max(mza)
    if maxmass - scanranges[-1] < 16:
        scanranges[-1] = int(np.ceil(maxmass) + 1)
    else:
        scanranges.append(int(np.ceil(maxmass)) + 1)
    
    #if all intensities are similar percentage wise to the sum, it should be worth less
    #ions that stand out get rewarded
    
    rangescores = []
    rangebounds = np.stack((scanranges[:-1], scanranges[1:]), axis=1).tolist()
    for n, (l, r) in enumerate(rangebounds):
        secintensities = intensities[np.logical_and(mza >= l, mza < r)]
        secpercents = secintensities / secintensities.sum()
        secranks = secpercents.size - secintensities.argsort().argsort()
        secranksadj = secranks - 1
        secratios = secranks / (secranks.size - secranksadj)
        secscores = secratios / secpercents
        rangescores.extend(secscores.tolist())
        nlowers.append(len(secscores[secscores < 1]))
    scoredms2scans[scan].append(rangescores)
    scoredms2scans[scan] = np.array(scoredms2scans[scan])

print(time() - nt, 'scans scored')

searchtime = 0
filtertime = 0
fraglens = 0
scanlens = 0
chargeiterations = 0
nt = time()
positioncache = {}
elementalcache = {}
descentcache = {}
groupseqs = []
groupsubformulas = []
#postfragmenttypes = Counter()
#postfragmentcounts = defaultdict(lambda: Counter())
initialmatches = 0
finalmatches = 0
subformulaoutput = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))))) #seq: distid: line: scan: subformula: ion charge: ion: [metrics]
for member in group:
    if '(' in member:
        groupsubformulas.append(member)
    else:
        groupseqs.append(member)
fragments = {}
for seq in groupseqs:
    fragments[seq] = fragmentation_compositions(seq)
for subformula in groupsubformulas:
    probindices = {elementsofprobabilityindices[i]: probabilityorganizer[i] for i in matchprobabilities[subformula]}
    subindices = subformulasubindices[subformula]
    output, fragmasses = [], []
    for submatchindex in subindices:
        seq = submatchsequences[submatchindex]
        for ion, fragcomp in fragments[seq].items():
            elementalorganizer = {} #element: [[iso heaps]]
            fragmentpositions = {} #element: position: iso
            fragstrings = ''
            for e, c in fragcomp.items():
                fragprobs = probindices[e]
                fragstring = str(c) + '/' + '/'.join(('/'.join((k, str(v))) for k, v in probindices[e].items()))
                fragstrings += fragstring
                if len(fragprobs) > 1:
                    #try/except is faster than an if/else, so i might as well
                    try:
                        elementlist = elementalcache[fragstring]
                        positions = positioncache[fragstring]
                    except KeyError: #not in cache
                        elementlist, positions = fragment_element_binomial_walk(e, c, fragprobs)
                        elementalcache[fragstring] = elementlist
                        positioncache[fragstring] = positions
                    elementalorganizer[e] = elementlist.copy()
                    fragmentpositions[e] = positions
                else: #no need for cache, only 1 iso
                    iso = list(fragprobs)[0]
                    elementalorganizer[e] = [[-1, 1, elementalmasses[iso]*c, e, [c]]]
                    fragmentpositions[e] = {0: iso}
            try:
                fragformulas, massesandabundances = descentcache[fragstrings]
            except KeyError: #not done prior
                fragformulas, massesandabundances = fragment_descending_partial_products(elementalorganizer, fragmentpositions)
                descentcache[fragstrings] = fragformulas, massesandabundances
            for n, (m, i) in enumerate(massesandabundances.tolist()):
                out = (seq, ion, fragformulas[n].decode(), n, i)
                output.append(out)
                fragmasses.append(m)
    fragmasses, output = zip(*sorted(zip(fragmasses, output)))
    fragmasses = np.array(fragmasses)
    fraglens += fragmasses.size
    st = time()
    for scan, lines in linesbyscanbysubformula[subformula].items():
        if len(lines) > 1:
            #analyteid = '_'.join((str(analytesbydistribution[distributionsoflines[i]]) for i in lines))
            chargeset = set(chargesoflines[i] for i in lines)
            ##if len(chargeset) > 1: -> test passes
            ##    print('PROBLEM, chargeset len > 1')
            maxcharge = max(chargeset)
            #CHECK if ^this is ever different, i'm pretty sure its always the same charge, there shouldn't be different ones
            #^because even if the same subformula is in the same scan more than once, it will never be of a different charge than itself in another distribution..
            #linestring = '_'.join((str(i) for i in lines))
            linesofmatchdistributions = defaultdict(list) #distid: [lines]
            for line in lines:
                linesofmatchdistributions[distributionsoflines[line]].append(line)
        else:
            line = lines[0]
            #analyteid = analytesbydistribution[distributionsoflines[lines]]
            maxcharge = chargesoflines[line]
            #linestring = str(lines)
            linesofmatchdistributions = {distributionsoflines[line]: [line]}
        #put fragmasses here -> append precursor ion of the line? because i dont want to calculate precursors via the above dists, but i want to match them
        #^i might just make a special precursor search because this gets too retarded
        #i'm removing precursors from the above calculations for now
        ms2masses, ms2intensities, ms2scores = scoredms2scans[scan]
        scanlens += ms2masses.size
        #ms2masses = ms2masses.tolist()
        outputorganizer = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))) #sequence: charge: ion: fragrank: [[metrics],]
        for charge in range(1, maxcharge+1):
            chargeiterations += 1
            chargedfragments = ((fragmasses + proton * charge) / charge)
            #i'm going to ditch the radius neighbors for a nearest neighbors concept instead, ought to have a minor speedboost at least
                #when MS resolution is low -> you won't get anything close enough to have more than 1 thing in a radius
                #if its high -> you CAN have this, but you should also expect the masses to be accurate
                    #i can see this in the ms1 vs ms2 data for the fr400 file
                #this will match at most 2 ions if they both have the same distance to a theoretical fragment
            matches = nearest_neighbors_ppm_tolerance(chargedfragments, ms2masses)
            #matches = radius_neighbors_ppm_tolerance(chargedfragments.tolist(), ms2masses)
            for fragindex, scanindices in matches.items(): #frag index: [mass index] or [mass index 1, mass index 2]
                #a scanmass can match to multiple generated fragment ions
                for scanindex in scanindices:
                    experimentalmass = ms2masses[scanindex]
                    ionscore = ms2scores[scanindex]
                    experimentalintensity = ms2intensities[scanindex]
                    theoreticalmass = chargedfragments[fragindex]
                    #ppmerror = ((experimentalmass - theoreticalmass) / experimentalmass) * 1000000 #maybe this could be done somewhere below instead
                    seq, ion, fragformula, fragrank, theoreticalabundance = output[fragindex]
                    metrics = [fragrank, theoreticalmass, experimentalmass, theoreticalabundance, experimentalintensity, scanindex, fragformula, ionscore]
                    outputorganizer[seq][charge][ion][fragrank].append(metrics)
                    initialmatches += 1
        nst = time()
        for seq, ioncharges in outputorganizer.items():
            for ioncharge, ions in ioncharges.items():
                for ion, fragranks in ions.items():
                    fcount = 0
                    for fragrank in sorted(fragranks):
                        if fragrank == fcount:
                            fcount += 1
                    if fcount > 0:
                        combinationoutputs = defaultdict(list) #either 0: [single ion] or 1: [multiple ions]
                        #^multiple ions will always be chosen over the single in ranked order
                        #^if there's multiple single ions, whichever is closer in intensity to the average intensity of that scan will be chosen
                        if fcount > 1:
                            #process multiple potential fragiso ranks
                            #iterate products i guess
                            #assemble a list of the number of total combinations, check if len > 1 or not
                            isoiterators = {}
                            for c in range(fcount):
                                isoiterators[c] = fragranks[c]
                            #this is determining which out of all possible frag iso combos makes the best distribution for this matched frag dist
                            for rankcombos in itertools.product(*isoiterators.values()):
                                #assess on charge distance, its consistency, and abundance modeling
                                rankcombinations = sorted(rankcombos) #not guaranteed to be most-intense ion as first mass
                                scorearray = np.array([i[:4] for i in rankcombinations])
                                rankpairs = np.array([(rankcombinations[i][6], rankcombinations[i+1][6]) for i in range(len(rankcombinations)-1)])
                                theoreticalabundances = scorearray[:,2]
                                experimentalintensities = scorearray[:,3]
                                lowerthbounds = theoreticalabundances[:-1] / theoreticalabundances[1:]
                                #the closer the max / min ratio of theoretical abundance are to 1, the more accurate the ratio between the 2 needs to be ->> square the theoretical ratio, that's the limit for the experimental
                                #the min would be half the initial ratio as long as its > 1? maybe
                                #and the max would be the squared value
                                upperthbounds = lowerthbounds ** 2 #this decreases lower bounds and increases upper ones.. i think this is fine?
                                #model by mass but rank by ranks i guess?
                                #start at the 0 to 1 rank linkage
                                #and if linkages dont always connect next to each other you can always link to a rank previously seen in this ranking
                                extimeshift = experimentalintensities[:-1] / experimentalintensities[1:]
                                acceptablepairs = np.sort(rankpairs[np.logical_and(extimeshift > lowerthbounds, extimeshift < upperthbounds)]).tolist()
                                try:
                                    firstgroup = acceptablepairs[0]
                                except IndexError:
                                    #useless, no good matches here
                                    continue
                                isoindices = set()
                                #the first two ranks MUST have the correct rank order intensities, or else it just won't be taken, take only the top rank instead
                                if firstgroup == [0, 1]:
                                    #good, start distassembling
                                    isoindices.update(firstgroup)
                                    #from these rankpairs just accumulate anything adjacent to whatevers already in there
                                    for l, r in acceptablepairs[1:]:
                                        if l in isoindices or r in isoindices:
                                            #connect anything adjacent
                                            isoindices.add(l)
                                            isoindices.add(r)
                                        else:
                                            #finished accumulating
                                            break
                                else:
                                    #mismatch, no good
                                    if 0 in firstgroup:
                                        #take 0
                                        combinationoutputs[0].append([abs(intensityaverages[scan]-rankcombos[0][3]), rankcombos[0]])
                                        continue
                                    else:
                                        #useless
                                        continue
                                finalindices = sorted(isoindices)
                                #score the dist and add it to the final list
                                massdiffs = np.diff(scorearray[finalindices,1])
                                avgmdiff = np.abs(massdiffs.mean() - massdiffs).mean() #mass distance consistency measure
                                theoreticalabundances = scorearray[finalindices,2].tolist()
                                experimentalintensities = scorearray[finalindices,3].tolist()
                                shiftdeviance = timeshift(experimentalintensities, theoreticalabundances) #time-series comparison
                                #shiftdeviance = linregress(experimentalintensities, theoreticalabundances).pvalue
                                combinationoutputs[1].append([avgmdiff * shiftdeviance, [rankcombos[i] for i in finalindices]])
                        else: #fcount == 1
                            fullmetrics = fragranks[0]
                            if len(fullmetrics) > 1:
                                #multiple matches to this fragrank
                                #pick whichever is closer in intensity to the average intensity of the scan
                                scanav = intensityaverages[scan]
                                avdiffs = [abs(scanav-i[3]) for i in fullmetrics]
                                minav = min(avdiffs)
                                finalmetric = fullmetrics[avdiffs.index(minav)]
                                combinationoutputs[0].append([minav, finalmetric])
                            else:
                                #single, take it
                                minav = abs(intensityaverages[scan] - fullmetrics[0][3])
                                combinationoutputs[0].append([minav, fullmetrics[0]])
                        #with either of these results below i'm assuming an equal score would only be given to matches that are exactly the same
                        if 1 in combinationoutputs:
                            #sort and pick best
                            selection = min(combinationoutputs[1])
                        elif 0 in combinationoutputs:
                            #take whichever is nearest to the mean intensity of the scan
                            selection = min(combinationoutputs[0])
                        else:
                            #got nothing
                            continue
                        for distid, alines in linesofmatchdistributions.items():
                            for line in alines:
                                #just a quick test -> test passes -> the order is different now but it should still be fine
                                #if seq in subformulaoutput:
                                #    if distid in subformulaoutput[seq]:
                                #        if scan in subformulaoutput[seq][distid]:
                                #            if line in subformulaoutput[seq][distid][scan]:
                                #                if subformula in subformulaoutput[seq][distid][scan][line]:
                                #                    if ioncharge in subformulaoutput[seq][distid][scan][line][subformula]:
                                #                        if ion in subformulaoutput[seq][distid][scan][line][subformula][ioncharge]:
                                #                            if selection == subformulaoutput[seq][distid][scan][subformula][ioncharge][ion]:
                                #                                print('selection present')
                                #                            else:
                                #                                print('different selection present')
                                subformulaoutput[seq][analytesbydistribution[distid]][distid][line][scan][subformula][ioncharge][ion] = selection
                    else:
                        #nada
                        continue
        filtertime += time() - nst
    searchtime += time() - st
fragtime = time() - nt - searchtime
searchtime -= filtertime
sct = time()
peptideleveloutput = []
distributionleveloutput = []
scanleveloutput = []
for seq, analyteids in subformulaoutput.items():
    for analyteid, distributions in analyteids.items():
        analytescore = 1
        analyteionscore = 0
        analyteppm = 0
        analyteintensity = 0
        ioncoverage = set()
        scanindexstring = ''
        intensityratios = []
        abundanceratios = []
        for distid, lines in distributions.items():
            distioncoverage = set()
            #this ion superset samples the most intense ion of a distribution and determines a top-down superset of all the fragmenting ions to be imposed on every other subformula and MS2 sampling taken at lesser intensities, if an ion doesn't show up here then it's not allowed to contribute to the rest of the scoring/ID process as its inconsistent and probably not real
            #so ie this assumes all subformulas fragment similar enough for it to matter despite slight isotopic differences - which i think should be ok
            #starting with the line and scan where this distribution sampled the largest MS1 intensity
            line, scan = maxintensitylinesofdists[distid]
            #if scan in scans:
            if line in lines:
                #lines = scans[scan]
                scans = lines[line]
                #if line in lines:
                if scan in scans:
                    #this is what should be the most abundant subformula at that position
                    subformula = subformulasofsequencedistribution[distid][seq]
                    subformulas = scans[scan]
                    if subformula in subformulas:
                        ioncharges = subformulas[subformula]
                        ionsuperset = defaultdict(set) #charge: [ions]
                        for ioncharge, ions in ioncharges.items():
                            #add to these ions to the superset of this identification instance
                            ionsuperset[ioncharge].update(ions)
                            ioncoverage.update(ions)
                            distioncoverage.update(ions)
                            #for ion, metrics in ions.items():
                    else:
                        #no superset to be made, the supposed best match isn't there
                        continue
                else:
                    continue
            else:
                continue
            scanorder = []
            ms1intensities = []
            ms2intensities = []
            scanlineintensitiesbyion = defaultdict(lambda: Counter()) #line-scan: ion: intensity
            distppm = 0
            distscore = 1
            distionscore = 0
            distintensity = 0
            #for scan, lines in scans.items():
            for line, scans in lines.items():
                #for line, subformulas in lines.items():
                for scan, subformulas in scans.items():
                    fragmentindices = set() #all fragmass indices in a scan
                    linescan = str(line) + '-' + str(scan)
                    #sort subformulas using subformulapercent, take only adjacent matches, if something has no superset matches -> break
                    subformulalist = sorted((subformulapercent[i][seq], i) for i in subformulas)
                    #subformulalist = sorted((*subformulapercent[i][seq], subformula) for i in subformulas)
                    #^which is faster?
                    #main scoring mechanisms:
                        #fragdist multiple -> here
                        #cross-scan consistency -> here as a time series across scans
                        #sequence geometry -> here
                        #cross-subformula entropy -> here
                        #intensity pair entropy -> here -> nah im not implementing this
                        #MS1/MS2 intensity entropy by scan % -> next script -> groups into dists i suppose
                    #aiming to make lower scores better in every case to multiply them all together
                    subformulamassindices = defaultdict(lambda: defaultdict(dict)) #ioncharge: ion: subformula: [masses]
                    subformulaintensities = defaultdict(lambda: defaultdict(dict)) #ioncharge: ion: subformula: intensity sum
                    abundanceofsubformulas = {} #subformula: abundance
                    #fragdistmultiple = 1
                    #fragioncount = 0
                    fragmentintensities = set() #assuming each intensity is unique which is actually false apparently, but within scans maybe more probable
                    #ionsubformulastring = '' #subformula^subformularank%ioncharge&ion&scanindex&ppmerror_ioncharge&ion...-subformula&... in order of subformula abundance
                    #gonna capture intensities across ions and check a timeshift
                    qcount = 0
                    sumppm = 0
                    sumintensity = 0
                    scanionscore = 0
                    scanions = set()
                    for (qualrank, abundance), subformula in subformulalist:
                        #this is multiplying the fragdist scores and assembling info to be used for everything else
                        #iterating and applying multiples in order of decreasing subformula abundance
                        if qualrank == qcount:
                            #substring = '-' + subformula + '^' + str(qualrank) + '%'
                            ionmatches = False
                            ioncharges = subformulas[subformula]
                            abundanceofsubformulas[subformula] = abundance
                            for ioncharge, ions in ioncharges.items():
                                if ioncharge in ionsuperset:
                                #if True:
                                    for ion, metrictuple in ions.items():
                                        if ion in ionsuperset[ioncharge]:
                                        #if True:
                                            scanions.add(ion)
                                            fragdistscore, metrics = metrictuple
                                            fragintensitysum = 0
                                            fragmassindices = []
                                            match metrics[0]:
                                                case list():
                                                    #metrics = sorted(metrics)
                                                    for fragrank, theoreticalmass, experimentalmass, theoreticalabundance, experimentalintensity, scanindex, fragformula, ionscore in metrics:
                                                        ppmerror = ((experimentalmass - theoreticalmass) / experimentalmass) * 1000000
                                                        #substring += f'_{ioncharge}&{ion}&{fragrank}&{scanindex}&{round(ppmerror,8)}&{round(theoreticalmass,8)}&{round(experimentalmass,8)}&{round(theoreticalabundance,8)}&{round(experimentalintensity,8)}&{fragformula}'
                                                        fragintensitysum += experimentalintensity
                                                        fragmassindices.append(scanindex)
                                                        fragmentintensities.add(experimentalintensity)
                                                        #if qualrank == 0:
                                                        scanlineintensitiesbyion[linescan][ion] += experimentalintensity
                                                        sumppm += abs(ppmerror) * experimentalintensity
                                                        sumintensity += experimentalintensity
                                                        scanionscore += math.log(ionscore)
                                                        finalmatches += 1
                                                case float:
                                                    fragrank, theoreticalmass, experimentalmass, theoreticalabundance, experimentalintensity, scanindex, fragformula, ionscore = metrics
                                                    ppmerror = ((experimentalmass - theoreticalmass) / experimentalmass) * 1000000
                                                    #substring += f'_{ioncharge}&{ion}&{fragrank}&{scanindex}&{round(ppmerror,8)}&{round(theoreticalmass,8)}&{round(experimentalmass,8)}&{round(theoreticalabundance,8)}&{round(experimentalintensity,8)}&{fragformula}'
                                                    fragintensitysum += experimentalintensity
                                                    fragmassindices.append(scanindex)
                                                    fragmentintensities.add(experimentalintensity)
                                                    #if qualrank == 0:
                                                    scanlineintensitiesbyion[linescan][ion] += experimentalintensity
                                                    sumppm += abs(ppmerror) * experimentalintensity
                                                    sumintensity += experimentalintensity
                                                    scanionscore += math.log(ionscore)
                                                    finalmatches += 1
                                            #fragdistmultiple *= fragdistscore
                                            #if ioncharge in subformulamassindices: -> test passes
                                            #    if ion in subformulamassindices[ioncharge]:
                                            #        if subformula in subformulamassindices[ioncharge][ion]:
                                            #            print('problem')
                                            subformulamassindices[ioncharge][ion][subformula] = fragmassindices
                                            subformulaintensities[ioncharge][ion][subformula] = fragintensitysum
                                            fragmentindices.update(fragmassindices)
                                            #ionmatches = True
                                            #fragioncount += 1
                            #if ionmatches:
                            #    ionsubformulastring += substring
                            qcount += 1
                        else:
                            #descending order of subformulas is finished
                            break
                    #for ion in ioncoverage:
                    #    #tracking fragment ions to infer the likelihood of their surroundings
                    #    iontype = ion[0]
                    #    postfragmenttypes[iontype] += 1
                    #    if iontype in 'abc': #nfrags
                    #        ioncount = int(ion[1:])
                    #        npartialseq = seq[ioncount-countrange:ioncount]
                    #        if len(npartialseq) == countrange: #partialseq isn't cut off by the end of the sequence
                    #            postfragmentcounts['n'][npartialseq] = 1
                    #        cpartialseq = seq[ioncount:ioncount+countrange+1]
                    #        if len(cpartialseq) == countrange: #partialseq isn't cut off by the end of the sequence
                    #            postfragmentcounts['n'][cpartialseq] = 1
                    #    elif iontype in 'xyz': #cfrags
                    #        ioncount = len(seq) - int(ion[1:])
                    #        npartialseq = seq[ioncount-countrange:ioncount]
                    #        if len(npartialseq) == countrange: #partialseq isn't cut off by the end of the sequence
                    #            postfragmentcounts['c'][npartialseq] = 1
                    #        cpartialseq = seq[ioncount:ioncount+countrange+1]
                    #        if len(cpartialseq) == countrange: #partialseq isn't cut off by the end of the sequence
                    #            postfragmentcounts['c'][cpartialseq] = 1
                    
                    #if len(fragmentintensities) > 1:
                    if len(scanions) > 1:
                        distionscore += scanionscore
                        avgppm = sumppm / sumintensity
                        distppm += sumppm
                        distintensity += sumintensity
                        
                        indexstring = '/'.join((map(str, sorted(fragmentindices))))
                        scanindexstring += f'{scan}[{indexstring}]'
                        
                        ms1intensities.append(lineintensitiesofscans[scan][line])
                        matchedintensitysum = sum(fragmentintensities)
                        ms2intensities.append(matchedintensitysum)
                        scanorder.append(scan)
                        
                        #intensity scoring based on whether the entirety of the intensity is kept by just 1 ion or if its well-dispersed (which is presumably better)
                        #case: 1 ion takes up 20% of the total intensity of 9 ions -> 1/(.8*8)
                        #case: 1 ion takes up 80% of the total intensity of 4 ions -> 1/(.2*3)
                        #etc
                        #matchesintensitymax = max(fragmentintensities)
                        #intensitydispersion = 1 / (((matchedintensitysum - matchesintensitymax) / matchedintensitysum) * (len(fragmentintensities) - 1)) #intensity dispersion
                        #invertedsum = 1 / matchedintensitysum #fuck it i'm doing it
                        #analytescore *= intensitydispersion * invertedsum
                        #analytescore *= avgppm
                        scangeometry = sequence_geometry(seq, scanions)
                        #scanoutput = f'{seq},{analyteid},{distid},{line},{scan},{scanionscore},{avgppm},{intensitydispersion},{invertedsum}'
                        scanoutput = f'{seq},{analyteid},{distid},{line},{scan},{indexstring},{scanionscore},{avgppm},{matchedintensitysum},{scangeometry}'
                        scanleveloutput.append(scanoutput)
                    #else:
                    #    #intensitycoverage = 1
                    #    if sumintensity:
                    #        scanoutput = f'{seq},{analyteid},{distid},{line},{scan},{scanionscore},{avgppm},1,1'
                    #    else:
                    #        scanoutput = f'{seq},{analyteid},{distid},{line},{scan},{scanionscore},0,1,1'
                    #if fragmentindices:
                    
                        #assess cross-subformula entropy if its present
                        #maybe these don't need to be subformulas that are only at the same location, this could be done across all lines? you'd need to normalize by line intensities
                        for ioncharge, ions in subformulamassindices.items():
                            #this is taking all mass-shifted ion matches and puttig them in a time series as identifying the same ion from different subformulas at different masses is something i consider to be good evidence of a match
                            #scanions.update(ions)
                            for ion, ionsubformulas in ions.items():
                                if len(ionsubformulas) > 1:
                                    for l, r in itertools.combinations(ionsubformulas, 2):
                                        #it could possibly be slower to make these into sets while adding them to subformulamassindices but i think this wouldn't happen enough to make that faster, test it later if you need
                                        if not set(ionsubformulas[l]).intersection(ionsubformulas[r]):
                                            #all indices are different -> compare total intensities
                                            intensityratio = subformulaintensities[ioncharge][ion][l] / subformulaintensities[ioncharge][ion][r]
                                            abundanceratio = abundanceofsubformulas[l] / abundanceofsubformulas[r]
                                            #if abundanceratio > 1:
                                            #    #maintain the scaling abundanceratio to be between 0 and 1
                                            #    intensityratio = 1 / intensityratio
                                            #    abundanceratio = 1 / abundanceratio
                                            intensityratios.append(intensityratio)
                                            abundanceratios.append(abundanceratio)
                                        else:
                                            print(len(set(ionsubformulas[l]).intersection(ionsubformulas[r])), 'overlapped shifts')
                                            #if you can't do either of those because the indices overlap -> don't add the comparison to the score
                                            #i'm not going to do max-max comparisons because not all distributions of the same ion across subformulas are similar enough in nature for this to be wise
                        
                        #if len(intensityratios) > 1:
                        #    #i guess i could make this > 2 and make a 2 option but i'll be lazy for now
                        #    #this maximizes the adjacent differences of each matched ion reflected across the abundance of the subformula it belongs to
                        #    #it turns it into a hard-to-fake time series, then i use a timeshift concept to score the ion matches based on their experimental intensity ratios compared to the theoretical ones that ought to be present based on theoretical subformula abundance
                        #    #print(ilen, 'length intensity / abundance ratios') #not seen this happen yet
                        #    maximalorderedabundances, maximalorderedintensities = difference_maximization(abundanceratios, intensityratios)
                        #    maximaldiffsubformularatiotimeshift = timeshift(maximalorderedintensities, maximalorderedabundances)
                        #    #maximaldiffsubformularatiotimeshift = linregress(maximalorderedintensities, maximalorderedabundances).pvalue
                        #    analytescore *= maximaldiffsubformularatiotimeshift
                        #    
                        #    #rawsubformularatiotimeshift = timeshift(intensityratios, abundanceratios)
                        #    
                        #    #sortedabundanceratios, sortedintensityratios = zip(*sorted(zip(abundanceratios, intensityratios)))
                        #    #sortedsubformularatiotimeshift = timeshift(sortedintensityratios, sortedabundanceratios)
                        #    #sortedsubformularatiotimeshift = linregress(sortedintensityratios, sortedabundanceratios).pvalue
                        #    
                        #    scanoutput += f',{maximaldiffsubformularatiotimeshift}'
                        #    scanleveloutput.append(scanoutput)
                        #else:
                        #    #if not scanoutput.endswith('0,1,1'):
                        #    #    scanoutput += ',1,1,1,1'
                        #    #    scanleveloutput.append(scanoutput)
                        #    scanoutput += ',1'
                        #    scanleveloutput.append(scanoutput)
                        #elif ilen == 1:
                        #    #idk if this one is a good idea?
                        #    analytescore *= abs(intensityratios[0] - abundanceratios[0]) / abundanceratios[0]
                        #else:
                        #    #length 0
                        #    subformulashiftdeviance = 1
                        #subformulaidentificationmultiple = 1 / qcount #idk about this one, can't necessarily verify the subformula presence with this
                        #finalscore = geometry * subformulashiftdeviance * intensitycoverage
                        #finaldistoutput = f'{seq},{analyteid},{distid},{scan},{line},{ionsubformulastring},{qcount+1},{ilen},{fragioncount},{len(ioncoverage)},{len(fragmentindices)},{matchedintensitysum},{geometry},{subformulashiftdeviance},{intensitycoverage},{finalscore}' + '\n'
                        #subformuladistoutput.append(finaldistoutput)
            analyteionscore += distionscore
            if distintensity:
                analyteintensity += distintensity
                avgdistppm = distppm / distintensity
            if len(ms2intensities) > 1:
                #scanlineintensitiesbyion = defaultdict(lambda: defaultdict(lambda: defaultdict(float))) #line-scan: ion: intensity
                ctshifts = 1
                for l, r in itertools.combinations(scanlineintensitiesbyion, 2):
                    llen = len(scanlineintensitiesbyion[l])
                    rlen = len(scanlineintensitiesbyion[r])
                    keys = set(scanlineintensitiesbyion[l]).intersection(scanlineintensitiesbyion[r])
                    if len(keys) > 1:
                        llist, rlist = [], []
                        for k in keys:
                            llist.append(scanlineintensitiesbyion[l][k])
                            rlist.append(scanlineintensitiesbyion[r][k])
                        iontimeshift = timeshift(llist, rlist)
                        #iontimeshift = linregress(llist, rlist).pvalue
                        if iontimeshift > 0:
                            ctshifts *= iontimeshift
                #this is a cross-scan consistency of matched fragion intensity that doubles as MS1 time series entropy
                scanorder, ms1intensities, ms2intensities = zip(*sorted(zip(scanorder, ms1intensities, ms2intensities)))
                intensitytimeshift = timeshift(ms2intensities, ms1intensities)
                #intensitytimeshift = linregress(ms2intensities, ms1intensities).pvalue
                distributiongeometry = sequence_geometry(seq, distioncoverage)
                analytescore *= intensitytimeshift * ctshifts
                #analytescore *= ctshifts
                analyteppm += distppm
                #should probably factor in the length of each series for this timeshift i guess
                distoutput = f'{seq},{analyteid},{distid},{Decimal(intensitytimeshift).to_eng_string()},{distributiongeometry},{ctshifts}'
                distributionleveloutput.append(distoutput)
        if len(ioncoverage) > 1 and analytescore < 1:
            ilen = len(intensityratios)
            if ilen > 1:
                maximalorderedabundances, maximalorderedintensities = difference_maximization(abundanceratios, intensityratios)
                subformularatiotimeshift = timeshift(maximalorderedintensities, maximalorderedabundances)
                analytescore *= subformularatiotimeshift
            elif ilen == 1:
                subformularatiotimeshift = abs(intensityratios[0] - abundanceratios[0]) / abundanceratios[0]
            else:
                subformularatiotimeshift = 1
            #applying geometric logic to the coverage of the sequence based on the matched ions
            ioncoverage = list(ioncoverage)
            analytegeometry =  sequence_geometry(seq, ioncoverage)
            analytescore *= analytegeometry
            avganalyteppm = analyteppm / analyteintensity
            analytescore *= avganalyteppm
            ioncoveragestring = '/'.join(map(str, sorted(ioncoverage)))
            #finalscore = Decimal(1) / Decimal(analytescore)
            analytescore *= analyteionscore
            finalscore = Decimal(analytescore)
            outputstring = f'{seq},{analyteid},{ioncoveragestring},{scanindexstring},{finalscore.to_eng_string()},{analytegeometry},{subformularatiotimeshift}'
            peptideleveloutput.append(outputstring)
