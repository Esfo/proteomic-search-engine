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

nproteins = 30000
minproteinlength = 200
maxproteinlength = 800
minpeptidelength = 6
maxpeptidelength = 20 #decreasing this decreases uniques + entropic boundary clarity?
missedcleavages = 1

cut = [{'K':1, 'R':1}, {'K':'P', 'R':'P'}] #trypsin

proton = 1.007276554940804

nanalytes = 30000
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

elementvector = [0 for _ in elementalmasses]
elementlist = list(elementalmasses)
vectorpositions = {k: n for n, k in enumerate(elementlist)}
elementpositions = {n: k for n, k in enumerate(elementlist)}

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

def distribution_generation(formula, atomiccomposition):
    natomicsample = Counter()
    for e, c in atomiccomposition.items():
        for iso in isotopesbyelement[e]:
            natomicsample[iso] = elementalprobabilities[iso] * c

    basevector = elementvector.copy()
    for e, v in isotopesbyelement.items():
        sumcount = 0
        for iso in v:
            startcount = round(natomicsample[iso])
            #^i tried generating random peptides in order to find a split that came out to 0.5 here but I'm unable to find an example where it actually happens. Hypothetically, if it does, there would be an equal probability between two isotopes, and this wouldn't handle that well. they'll be generated in the below loop regardless I suppose.
            basevector[vectorpositions[iso]] += startcount
            sumcount += startcount
        if sumcount < atomiccomposition[e]:
            #mono isotope rounded down but nothing could fill the gap, add to mono isotope
            basevector[vectorpositions[monoisotopickeys[e]]] += 1
        elif sumcount > atomiccomposition[e]:
            #this has never happened, and probably won't
            #multiple things rounded up.. presumably because there's an even split
            #subtracting from the mono isotope would be the easiest move rather than identifying which non-mono isotope was added
            basevector[vectorpositions[monoisotopickeys[e]]] -= 1
            print(seq, 'generated an erroneous example i\'ve been trying to catch with isotopic distribution rounding')
    
    branchkey = 0 #this is used as a basis for accessing subheaps
    
    baseprob = 1
    basemass = 0
    baseoffsets = defaultdict(int)
    branchprobabilitiesbyelement = defaultdict(lambda: defaultdict(lambda: 1)) #branchkey: e: prob
    branchmassesbyelement = defaultdict(lambda: defaultdict(int)) #branchkey: e: mass
    for n, c in enumerate(basevector):
        iso = elementpositions[n]
        if c > 0:
            e = iso[0]
            prob = elementalprobabilities[iso]**c
            mass = elementalmasses[iso] * c
            baseprob *= prob
            basemass += mass
            branchprobabilitiesbyelement[branchkey][e] *= prob
            branchmassesbyelement[branchkey][e] += mass
            if iso in nonmonoisotopicelements:
                combfactor = math.comb(atomiccomposition[e] - baseoffsets[e], c)
                baseprob *= combfactor
                branchprobabilitiesbyelement[branchkey][e] *= combfactor
                baseoffsets[e] += c
    
    expansiondirections = {} #branchkey: 1 or -1, a negative direction is only given to nonmonisotopic elements that are in basecomposition generated above
    isotopesbyisodirection = {} #isodirection: iso
    isodirectionsbyelement = defaultdict(list) #e: [isodirections]
    opposingdirections = {} #isodirection: isodirection of the same element moving in the other direction, I don't let these coexist within the same subheap
    
    isodirection = 0
    for e, isos in nonmonoisotopicgroups.items():
        if e in atomiccomposition:
            for iso in isos:
                #positive direction
                isotopesbyisodirection[isodirection] = iso
                isodirectionsbyelement[e].append(isodirection)
                expansiondirections[isodirection] = 1
                isodirection += 1
                if basevector[vectorpositions[iso]] > 0:
                    #negative direction
                    isotopesbyisodirection[isodirection] = iso
                    isodirectionsbyelement[e].append(isodirection)
                    expansiondirections[isodirection] = -1
                    #there are two directions for this iso from basecomp
                    opposingdirections[isodirection] = isodirection - 1
                    opposingdirections[isodirection - 1] = isodirection
                    isodirection += 1
    
    finalprobabilities = {} #branchkey: abundance
    finalmasses = {} #branchkey: mass
    
    finalprobabilities[branchkey] = baseprob
    finalmasses[branchkey] = basemass
    
    branchcount = 1
    currentbranchkeys = [] #list of all currently unexplored branchkeys
    priorbranch = {} #branchkey: branchkey of branch that generated this branch
    branchopposers = defaultdict(set) #branchkey: set of non-compatible isodirections
    branchprobabilities = defaultdict(dict) #branchkey: isodirection: combined element prob
    branchmasses = defaultdict(dict) #branchkey: isodirection: combined element mass
    branchcompositions = {} #branchcount: branchvector
    branchisodirections = {} #branchkey: tailored version of isotopesbyisodirection
    vectorsets = set()
    
    branchisodirections[branchkey] = isotopesbyisodirection.copy()
    branchcompositions[branchkey] = basevector
    
    for isodirection, iso in isotopesbyisodirection.items():
        e = iso[0]
        acount = atomiccomposition[e]
        mk = monoisotopickeys[e]
        direction = expansiondirections[isodirection]
        newbasecomp = basevector.copy()
        isopos = vectorpositions[iso]
        monopos = vectorpositions[mk]
        newbasecomp[isopos] += direction
        newbasecomp[monopos] -= direction
        vectorsets.add(tuple(newbasecomp))
        n = 0
        newelementmass = 0
        newelementprob = 1
        for en in vectorrangesbyelement[e]:
            c = newbasecomp[en]
            if c > 0:
                loopiso = elementpositions[en]
                newelementmass += elementalmasses[loopiso] * c
                newelementprob *= elementalprobabilities[loopiso]**c
                if loopiso in nonmonoisotopicelements:
                    newelementprob *= math.comb(acount-n, c)
                    n += c
        newprob = baseprob / branchprobabilitiesbyelement[branchkey][e] * newelementprob
        if newprob >= minimumabundance:
            newmass = basemass - branchmassesbyelement[branchkey][e] + newelementmass
            #add everything to new branchthings via branchcount!
            branchprobabilitiesbyelement[branchcount] = branchprobabilitiesbyelement[branchkey].copy()
            branchprobabilitiesbyelement[branchcount][e] = newelementprob
            branchmassesbyelement[branchcount] = branchmassesbyelement[branchkey].copy()
            branchmassesbyelement[branchcount][e] = newelementmass
            branchcompositions[branchcount] = newbasecomp
            finalprobabilities[branchcount] = newprob
            finalmasses[branchcount] = newmass
            priorbranch[branchcount] = branchkey
            branchisodirections[branchcount] = branchisodirections[branchkey].copy()
            if newbasecomp[vectorpositions[iso]] == 0:
                #negative direction ended, remove that isodirection in the future branches
                #del branchisodirections[branchkey][isodirection]
                del branchisodirections[branchcount][isodirection]
                #branchopposers[branchcount].add(isodirection)
            if newbasecomp[vectorpositions[mk]] == 0:
                #end of a positive direction, remove all isodirections of that element in the future branches
                for shk in isodirectionsbyelement[e]:
                    try:
                        #del branchisodirections[branchkey][shk]
                        del branchisodirections[branchcount][shk]
                    except KeyError:
                        pass
                    #branchopposers[branchcount].add(shk)
            if isodirection in opposingdirections:
                #remove opposing direction from isodirections
                branchopposers[branchcount].add(opposingdirections[isodirection])
                #try:
                #    del branchisodirections[branchcount][isodirection]
                #except KeyError:
                #    #previously deleted above
                #    pass
            currentbranchkeys.append(branchcount)
            branchcount += 1
    
    while currentbranchkeys:
        nextbranchkeys = []
        branches, branchkeysbyvector = [], {}
        for branchkey in currentbranchkeys:
            passing = False
            prior = priorbranch[branchkey]
            #generate all potential vectors here
            #for isodirection, iso in branchisodirections[prior].items():
            for isodirection, iso in branchisodirections[branchkey].items():
                if isodirection not in branchopposers[branchkey]:
                    e = iso[0]
                    mk = monoisotopickeys[e]
                    direction = expansiondirections[isodirection]
                    newbasecomp = branchcompositions[branchkey].copy()
                    isopos = vectorpositions[iso]
                    monopos = vectorpositions[mk]
                    newbasecomp[isopos] += direction
                    newbasecomp[monopos] -= direction
                    compvector = tuple(newbasecomp)
                    if compvector in vectorsets:
                        if compvector in branchkeysbyvector:
                            #merge branchopposers
                            foundkey = branchkeysbyvector[compvector]
                            branchopposers[foundkey].update(branchopposers[branchkey])
                            #branchopposers is the easiest mechanism to use here because merging branchisodirections would be a via-negation process so symmetric-differencing and deleting would be involved and it might be annoying and tedious
                    else:
                        branches.append([branchcount, branchkey, isodirection])
                        branchkeysbyvector[compvector] = branchcount
                        branchcompositions[branchcount] = newbasecomp
                        vectorsets.add(compvector)
                        branchcount += 1
                        passing = True
        for branchkey, prior, isodirection in branches:
            iso = isotopesbyisodirection[isodirection]
            e = iso[0]
            acount = atomiccomposition[e]
            newbasecomp = branchcompositions[branchkey]
            n = 0
            newelementmass = 0
            newelementprob = 1
            for en in vectorrangesbyelement[e]:
                c = newbasecomp[en]
                if c > 0:
                    loopiso = elementpositions[en]
                    newelementmass += elementalmasses[loopiso] * c
                    newelementprob *= elementalprobabilities[loopiso]**c
                    if loopiso in nonmonoisotopicelements:
                        newelementprob *= math.comb(acount-n, c)
                        n += c
            newprob = finalprobabilities[prior] / branchprobabilitiesbyelement[prior][e] * newelementprob
            if newprob >= minimumabundance:
                mk = monoisotopickeys[e]
                newmass = finalmasses[prior] - branchmassesbyelement[prior][e] + newelementmass
                #modify basecomp via elementcomp
                #add everything to new branchthings via branchkey!
                branchprobabilitiesbyelement[branchkey] = branchprobabilitiesbyelement[prior].copy()
                branchprobabilitiesbyelement[branchkey][e] = newelementprob
                branchmassesbyelement[branchkey] = branchmassesbyelement[prior].copy()
                branchmassesbyelement[branchkey][e] = newelementmass
                branchopposers[branchkey] = branchopposers[prior].copy()
                finalprobabilities[branchkey] = newprob
                #finalformulas[branchkey] = subformula
                finalmasses[branchkey] = newmass
                priorbranch[branchkey] = prior
                #copy branchopposers
                branchisodirections[branchkey] = branchisodirections[prior].copy()
                if newbasecomp[vectorpositions[iso]] == 0:
                    #negative direction ended, remove that isodirection in the future branches
                    #del branchisodirections[prior][isodirection]
                    del branchisodirections[branchkey][isodirection]
                    #branchopposers[branchkey].add(isodirection)
                elif newbasecomp[vectorpositions[mk]] == 0:
                    #end of a positive direction, remove all isodirections of that element in the future branches
                    for shk in isodirectionsbyelement[e]:
                        try:
                            #del branchisodirections[prior][shk]
                            del branchisodirections[branchkey][shk]
                        except KeyError:
                            pass
                        #branchopposers[branchkey].add(isodirection)
                if isodirection in opposingdirections:
                    #remove opposing direction from isodirections
                    branchopposers[branchkey].add(opposingdirections[isodirection])
                    #try:
                    #    del branchisodirections[branchkey][isodirection]
                    #except KeyError:
                    #    pass
                nextbranchkeys.append(branchkey)
        currentbranchkeys = nextbranchkeys.copy()
    
    massesandabundances = [[], []]
    formulas = []
    for k, m in finalmasses.items():
        subformula = ''
        for n, c in enumerate(branchcompositions[k]):
            if c > 0:
                subformula += f'{elementpositions[n]}({c})'
        massesandabundances[0].append(m)
        massesandabundances[1].append(finalprobabilities[k])
        formulas.append(subformula)

    #sorting everything by mass
    massesandabundances = np.array(massesandabundances)
    formulas = np.array(formulas, dtype='S')
    formulas = formulas[massesandabundances[0].argsort()].tolist()
    massesandabundances = massesandabundances[:,massesandabundances[0].argsort()]
    return formula, formulas, massesandabundances

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

def fragment_element_binomial_walk(dividingthreshold, e, acount, fragprobabilities):
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

def fragment_descending_partial_products(dividingthreshold, elementalorganizer, fragmentpositions):
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

def fragmentation_compositions(aminocomps, nfrags, cfrags, seq):
    fragments = {}

    #calculate the compositions of the n-term fragments
    fragcomp_n = {}
    for n, aa in enumerate(seq[:-1]):  
        aa_composition = aminocomps[aa]
        for k in aa_composition:
            fragcomp_n[k] = fragcomp_n.get(k, 0) + aa_composition.get(k, 0)
        for ion, modcomp in nfrags.items():
            fragment_composition = fragcomp_n.copy()
            for k in modcomp:
                fragment_composition[k] = fragment_composition.get(k, 0) + modcomp.get(k, 0)
            fragments[ion + str(n + 1)] = fragment_composition

    #calculate the compositions of the c-term fragments
    fragcomp_c = {}
    for n, aa in enumerate(seq[::-1][:-1]): 
        aa_composition = aminocomps[aa]
        for k in aa_composition:
            fragcomp_c[k] = fragcomp_c.get(k, 0) + aa_composition.get(k, 0)
        for ion, modcomp in cfrags.items():
            fragment_composition = fragcomp_c.copy()
            for k in modcomp:
                fragment_composition[k] = fragment_composition.get(k, 0) + modcomp.get(k, 0)
            fragments[ion + str(n + 1)] = fragment_composition

    return fragments

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
            trackedgroups[lineid] = [masses, rts, points]
            distributionsoflines[lineid] = distid
            linesofdistributions[distid].append(lineid)
            positionsoflines[distid].append(n)
            analytekeys[analyteid][distid] = charge
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
individualaminoacidstrengths = {} #aa: strength, as a prob, higher prob = stronger bonder
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
    frags = fragmentation_compositions(aminoacidcomposition, ndict, cdict, peptide)
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
                        elementlist, positions = fragment_element_binomial_walk(dividingthreshold, e, c, fragprobs)
                        elementalorganizer[e] = elementlist.copy()
                        fragmentpositions[e] = positions
                    else: #no need for cache, only 1 iso
                        #iso, c = list(fragprobs.items())[0] #WOOPS
                        iso = list(fragprobs)[0]
                        elementalorganizer[e] = [[-1, 1, elementalmasses[iso]*c, e, [c]]]
                        fragmentpositions[e] = {0: iso}
            fragformulas, massesandabundances = fragment_descending_partial_products(dividingthreshold, elementalorganizer, fragmentpositions)
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
    ms2scans[ms2id] = [masses, intensities, primaryinds]
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

#assigning relative intensity %'s of each MS1 distribution for each MS2 window
scansbyanalyte = defaultdict(list) #analyteid: [scans across all lines and charge states]
scanalytecharges = defaultdict(dict) #analyteid: scan: charge
spectralsamplings = defaultdict(lambda: defaultdict(dict)) #scan: line: % by area of ms1 lines
spectrascansums = defaultdict(float) #scan: sum area used in spectralsamplings
for line, scans in scansoflines.items():
    try:
        distid = distributionsoflines[line]
        analyteid = analytesbydistribution[distid]
        charge = analytekeys[analyteid][distid]
    except KeyError: #line is in nodists
        analyteid = -line
        charge = 0
    scansbyanalyte[analyteid].extend(scans)
    linegroup = trackedgroups[line]
    linetimes = linegroup[1]
    linemasses = linegroup[0]
    lineintensity = linegroup[2]
    lmax = linemasses.max()
    lmin = linemasses.min()
    for scan in scans:
        pcoords = precursordict[scan]
        rt = pcoords[0]
        #this assumes there is a left and right intensity, i guess there would be though?
        #this is >= and <= atm because of homogenous ms1 masses atm
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
        spectralsamplings[scan][line] = sampleintensity
        spectrascansums[scan] += sampleintensity
        scanalytecharges[analyteid][scan] = charge
scansbyanalyte = dict(scansbyanalyte)

#turning areas into percents
for scan, lines in spectralsamplings.items():
    for line in lines:
        spectralsamplings[scan][line] /= spectrascansums[scan]
    spectralsamplings[scan] = dict(lines) #can't pickle double default dicts
spectralsamplings = dict(spectralsamplings)

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
                                elementlist, positions = fragment_element_binomial_walk(dividingthreshold, e, c, fragprobs)
                                elementalorganizer[e] = elementlist.copy()
                                fragmentpositions[e] = positions
                            else:
                                iso = list(fragprobs)[0]
                                elementalorganizer[e] = [[-1, 1, elementalmasses[iso]*c, e, [c]]]
                                fragmentpositions[e] = {0: iso}
                    fragformulas, massesandabundances = fragment_descending_partial_products(dividingthreshold, elementalorganizer, fragmentpositions)
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
                ms2scans[scan][2].extend(primaryinds)
            else:
                ms2scans[scan] = [masses, intensities, primaryinds]
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

k = 0
formulaidentifiers = {}
distributionidentifiers = {}

librarykeys = []
librarymasses = []
librarymassdict = {} #lid: [masses]
librarypositions = {} #lid: [indices]
libraryintensityranks = {} #lid: [intensityranks]
for f, (masses, intensities) in sumabundances.items():
    #k = formulaidentifiers[f]
    formulaidentifiers[f] = k
    distributionidentifiers[k] = f
    librarymassdict[k] = masses
    librarypositions[k] = list(range(masses.size))
    intensityranks = np.abs(intensities.argsort().argsort() - intensities.size + 1)
    libraryintensityranks[k] = intensityranks
    librarykeys.extend(itertools.repeat(k, masses.size))
    librarymasses.extend(masses.tolist())
    k += 1

librarykeys = np.array(librarykeys)
librarymasses = np.array(librarymasses)

distributionkeys = []
distributionmasses = []
distributionmassdict = {} #did: [masses]
distributionintensityranks = {} #did: [intensityranks]
for k, (masses, intensities) in analytedistributions.items():
    distributionmassdict[k] = masses
    intensityranks = np.abs(intensities.argsort().argsort() - intensities.size + 1)
    distributionintensityranks[k] = intensityranks
    distributionkeys.extend(itertools.repeat(k, masses.size))
    distributionmasses.extend(masses.tolist())

distributionkeys = np.array(distributionkeys)
distributionmasses = np.array(distributionmasses)

radius = distributionmasses / 1000000 * ppmtolerance

lmtree = spatial.KDTree(librarymasses[:,None])
matches = lmtree.query_ball_point(distributionmasses[:,None], radius, workers=8)

matchorganizer = defaultdict(list)
for dk, lkeys in zip(distributionkeys, matches):
    matchorganizer[dk].extend(librarykeys[lkeys])

for k in list(matchorganizer):
    matchorganizer[k] = np.array(list(set(matchorganizer[k])))

linepositionsbyformula = defaultdict(lambda: defaultdict(set)) #formula: matchable library positions of a distribution: [lines that have ms2 scans]
distributionmatches = defaultdict(list)
lmatches = 0
dmatches = 0
for dk, lkeys in matchorganizer.items():
    if dk in scansbyanalyte:
        dmasses = distributionmassdict[dk]
        dsize = dmasses.size
        tx = 0
        for lk in lkeys.tolist():
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
                    distlines = linesofanalytes[dk][ri:re]
                    positions = librarypositions[lk][li:le]
                    formula = distributionidentifiers[lk]
                    distributionmatches[dk].append(lk)
                    for lines, pos in zip(distlines, positions):
                        for line in lines:
                            if line in scansoflines:
                                linepositionsbyformula[formula][pos].add(line)
                    tx += 1
        if tx > 0:
            lmatches += tx
            dmatches += 1
    if dk in distributionmatches:
        distributionmatches[dk] = np.array(distributionmatches[dk])

for k, v in linepositionsbyformula.items():
    for sk, sv in v.items():
        v[sk] = tuple(sv)
    linepositionsbyformula[k] = dict(v)
linepositionsbyformula = dict(linepositionsbyformula)

print(time() - t2, 'matches assembled')
print('library matches:', lmatches)
print('distribution matches:', dmatches)

success = []
failure = []
for analyteid, lkeys in distributionmatches.items():
    formulas = [distributionidentifiers[i] for i in lkeys.tolist()]
    if any([peptidesofanalytes[analyteid] in seqsbyformula[formula] for formula in formulas]):
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


#the basis of the entropy concept:
#you need to differentiate between the fragment ions of competing analytes using scan logic
#to do this, iterate combinations of scans representative of grouped analyteids

#in analytesofscans you have individual scans that show which analytes can be differentiated and where
#^but its not just those scans, those scans belong to a scangroup that does the same thing
#iterating scangroups gives you all unique scangroups
#plug a scangroup into analyteidsbyscangroup to get the analyteids that that scangroup can identify, then use scangroupsbyanalyteid to find all other relevant scangroups
#there's an inefficiency in this because you might be doing redundant work by iterating the same competing scangroup twice, maybe just add a set that indicates these have been seen already?

#intersection merge scangroups, then collect all of their analytes and scangroups within the loop that processed the intersection merged groups?
#^a problem arises here when different lines of a distribution are causing these to merge together, because tbh, different isos will probably have different frag patterns
#so my reference chain needs to reference lines of a distribution, and dists of an analyte, but the LOGIC needs to reference dists/analytes
#^its worth doing a brief exploration into whether a +/- proton from the fragmentation of one iso to another make sense
#break this down into logically-related groups for dists, and organizational overlapping lines of scans
#so from withiin this loop i can dictate logical groups across dists and charge states - which basically link an ID to the state of the identification of all related ions, AND i can link scans to lines to figure out which lines to then intersection merge in a new list that mimicks the concept of scangroups but now uses lines instead of analyteids or dists
#^so you won't have to charge segregate the way you do below with chargelists?
#i guess you would also want to differentiate between ions of lines from the same distributions - this would be useful when different lines then overlap with lines of other distributions, to make the differentiation easier

#intersection merge the lines -> differentiate ions
#intersection merge the distributions/analytes -> logical actors
#then iterate the intersection merged groups in order to reclassify different meta-groups for iteration in order to compare scan masses via nn
#^for the nn comparisons you would only need to iterate mergedlines
#^this approach would assume each individual proton location of a dist produces isotopomers that are differentiable from those at another proton location - which i need to do an exploration for, but i'm pretty sure it will be true

nt = time()

#spectrabyformula = defaultdict(lambda: defaultdict(set)) #formula: analyteid: scan
#for analyteid, samples in scanalytecharges.items():
#    if analyteid in distributionmatches: #not in nodists
#        maxfragmentcharge = max(samples.values()) - 1
#        unlinkedmatches = distributionmatches[analyteid]
#        for formula in unlinkedmatches.tolist(): #matches from the library
#            #if distid % 2: #not needed, distids are formulas in this simulation
#                #distid -= 1
#            #formula = distributionidentifiers[distid]
#            #generatedsequences.update(seqsbyformula[formula])
#            for scan in samples:
#                spectrabyformula[formula][analyteid].add(scan)
##generatedsequences = list(generatedsequences)
#
##pectrabyformula = {}
##temp = spectrabyformula.items()
#for k, v in spectrabyformula.items():
#    for sk, sv in v.items():
#        #spectrabyformula[k] = {sk: list(sv)}
#        spectrabyformula[k][sk] = tuple(sv)
#    spectrabyformula[k] = dict(spectrabyformula[k])
#spectrabyformula = dict(spectrabyformula)
#
#allfragmentscans = set()
#scangroups = set()
##analytegroups = set()
#analytesofscans = defaultdict(set) #scan: [analyteids]
#scangroupsbyscan = defaultdict(set) #scan: [scangroups]
#analyteidsbyscangroup = defaultdict(set) #scangroup: {analyteids}
#scangroupsbyanalyteid = defaultdict(set) #analyteid: [scangroups]
#for formula, sample in spectrabyformula.items():
#    for analyteid, sids in sample.items():
#        tsids = tuple(sorted(sids))
#        #analyteidsbyscangroup[tsids].add(analyteid) #this was here previously but i think it should be below in the loop that limits scangroup association by charge
#        chargelists = defaultdict(list)
#        allfragmentscans.update(sids)
#        #analytegroups.add(tsids)
#        for sid in sids:
#            chargelists[scanalytecharges[analyteid][sid]].append(sid)
#            analytesofscans[sid].add(analyteid)
#        for sids in chargelists.values():
#            tsids = tuple(sorted(sids))
#            scangroupsbyanalyteid[analyteid].add(tsids)
#            analyteidsbyscangroup[tsids].add(analyteid)
#            scangroups.add(tsids)
#            scanlines = []
#            for sid in sids:
#                scangroupsbyscan[sid].add(tsids)
#
#scangroups = tuple(scangroups)
##analytegroups = tuple(analytegroups)
#
##mergedlines = list(map(tuple, intersection_merge(linesofscans.values())))
#oldmergedscans = list(map(tuple, map(sorted, intersection_merge(scangroups))))
##mergedanalytes = list(map(tuple, intersection_merge(analytegroups)))

#the above loop via spectrabyformula is an older method i used, but it gives different results than the below, which should be the same thing...
#so it turns out the above was missing a few small edge cases, not sure what they are but the below is a superset of the above, and the new additions to the below are legit, so that's good enough for me

distsofscans = {}
for scan, lines in linesofscans.items():
    dists = set(distributionsoflines[i] for i in lines)
    distsofscans[scan] = tuple(dists)

mergeddistributions = list(map(tuple, intersection_merge(distsofscans.values())))

distributionmergedscans = []
for md in mergeddistributions:
    allscans = set()
    for dist in md:
        lines = linesofdistributions[dist]
        for line in lines:
            if line in scansoflines:
                linescans = scansoflines[line]
                allscans.update(linescans)
    distributionmergedscans.append(tuple(sorted(allscans))) #this sorting was for troubleshooting the comparison of this to the old method, its not necessary

print(len(distributionmergedscans) == len(intersection_merge(distributionmergedscans)), 'result for distributionmergedscan test')

mergedlines = intersection_merge(linesofscans.values())
mergedscans = [tuple(sorted(set(itertools.chain(*[scansoflines[j] for j in i])))) for i in mergedlines]

print(len(mergedscans) == len(intersection_merge(mergedscans)), 'result for mergedscan test')

#oldmergedscans = set(oldmergedscans)
#mergedscans = set(mergedscans)
#mismatchers = mergedscans.symmetric_difference(oldmergedscans)

#competingscans = set()
#overlappingscangroups = defaultdict(set) #scangroup: {other scangroups}
#for sids in mergedgroups:
#    for sid in sids:
#        for outersids in scangroupsbyscan[sid]:
#            if outersids != sids:
#                overlappingscangroups[sids].add(outersids)
#                competingscans.update(sids)
#                competingscans.update(outersids)

#competingscans = list(itertools.chain(*[i for i in mergedscans if len(i) > 1]))

#competinglines = set()
#competingscans = set()
#for lines in mergedlines:
#    if len(lines) > 1:
#        competinglines.update(lines)
#        for line in lines:
#            competingscans.update(scansoflines[line])
#
#nocompetition = allfragmentscans.difference(competingscans)
#competinglines = tuple(competinglines)
#competingscans = tuple(competingscans)
#print(len(competinglines), 'lines in competition')
#print(len(competingscans), 'scans in competition')

print(time() - nt, 'organizing scangroups')

#iterating mergedlines, use scansoflines to get the scan patterns
#now figure out how to derive comparison logic from this
#^i think you could get the base set of scans and use that as the starting point
#^then derive the set differences across the scangroups if they have an intersection
#^maybe if they don't have an intersection then you can use that as the basis for a variability measure?

#not needed
#for lines in mergedlines:
#    scans = [scansoflines[i] for i in lines]
#    scanpositions = defaultdict(list) #scan: [positions]
#    for n, ss in enumerate(scans):
#        for s in ss:
#            scanpositions[s].append(n)
#    mscans = intersection_merge(scans)
#    if len(scans) > 5:
#        print('success')
#        break

#the data was bad for chatGPT because nothing was matching via nn
#make the fragments pick from a pool that assigns a probability based on ~abundance and apply it to that abundance


#so, again:
#iterate scangroups to get a scangroup
#input scangroup into analyteidsbyscangroup to get relevant individual analyteids
#   > input the scans in the scangroup into analyteofscans to get secondary analyteids
#input all analyteids into scangroupsbyanalyteid to get secondary scangroups that are represented by that scangroup
#while loop until finished:
#   > input each scan from the secondary scangroups into scangroupsbyanalyteids to get tertiary analyteids within that scan
#   > input the tertiary analyteids from those scans into analyteidsbyscangroup to get more analyteids


#not needed
#aequals = 0
#adiffs = 0
#sequals = 0
#sdiffs = 0
#for scangroup in scangroups:
#    allscans = set()
#    allscans.update(scangroup)
#    analytes = set()
#    analytes.update(analyteidsbyscangroup[scangroup])
#    slen = len(allscans)
#    alen = len(analytes)
#    for scan in scangroup:
#        newanalytes = analytesofscans[scan]
#        analytes.update(newanalytes)
#        for a in newanalytes:
#            for subscan in scangroupsbyanalyteid[a]:
#                allscans.update(subscan)
#    if alen == len(analytes):
#        aequals += 1
#    else:
#        adiffs += 1
#    if slen == len(allscans):
#        sequals += 1
#    else:
#        sdiffs += 1



#KNN -> mainindicesofscans as scan: KNN inds

#original indexing process:
nt = time()

mainind = 0
linesbymainindex = defaultdict(set) #main index: [lines]
maintoprimaryindex = defaultdict(list) #main index: [primary indices]
primarytomainindex = {} #primary index: main index
mainindicesofscans = defaultdict(list) #scan: [main indices], previously known as scandict
scangroupbyline = defaultdict(list)
#for group, masses in massesofscangroups.items():
mscancounts = [] #counting the number of masses vs the number of merged masses
scansbymainindices = defaultdict(set) #main index: [scans]
for n, group in enumerate(mergedscans):
    masses, primaryinds = [], []
    for scan in group:
        try:
            m, i, p = ms2scans[scan]
            masses.extend(m)
            primaryinds.extend(p)
        except KeyError:
            pass
    if masses:
        masses = np.array(masses)[:,None]
        radius = (masses * ppmtolerance).flatten() / 1000000
        nn = spatial.KDTree(masses)
        matches = nn.query_ball_point(masses, radius).tolist()
        groupableinds = list(map(tuple, intersection_merge(matches)))
        mscancounts.append([n, len(masses), len(groupableinds)])
        for gi in groupableinds:
            for g in gi:
                primaryind = primaryinds[g]
                primarytomainindex[primaryind] = mainind
                maintoprimaryindex[mainind].append(primaryind)
                line = linesbyprimaryind[primaryind]
                linesbymainindex[mainind].add(line)
                scangroupbyline[line] = n
                scan = scansbyprimaryind[primaryind]
                mainindicesofscans[scan].append(mainind)
                scansbymainindices[mainind].add(scan)
            mainind += 1
mscancounts = np.array(mscancounts)

#plt.plot(mscancounts[:,1], mscancounts[:,2], '.')
#plt.show()

#with these latest modifications, only adding to ms2scans on the latter part that i do it on
#number of library matches goes up
#and the total wrong/incorrect/not incorrect go up too
#wtf
#its consistent

print(time() - nt, 'mass indexing')

incorrectindices = []
print(len(mergedscans), 'mergedscans')
print(sum(1 for i in mergedscans if len(i) > 1), '> length 1')
print(max(len(i) for i in mergedscans), 'max group length')

ms1entropy = defaultdict(lambda: Counter()) #ms2line: line: [union, ms1diff, ms2diff, symdiff]
ms2entropy = defaultdict(lambda: Counter()) #ms2line: line: [union, ms1diff, ms2diff, symdiff]
symdiffentropy = defaultdict(lambda: Counter()) #ms2line: line: [union, ms1diff, ms2diff, symdiff]
unionentropy = defaultdict(lambda: Counter()) #ms2line: line: [union, ms1diff, ms2diff, symdiff]
#newentropylist = []
#newentropylabels = []

primaryguesses = defaultdict(list) #primary: [ms1, ms2, union, classical] as indices
ms1assignments = defaultdict(set) #either 0, 1 or 2, based on below: [primaries]
ms2assignments = defaultdict(set) #either 0, 1 or 2, based on below: [primaries]
unionassignments = defaultdict(set) #either 0, 1 or 2, based on below: [primaries]
classicassignments = defaultdict(set) #^same
#primary: [ms1, ms2, union, classical] as indices
#correct -> 0
#incorrect -> 1
#not incorrect -> 2

#compare scansbymainindices to linesofscans
for mainindex, scans in scansbymainindices.items():
    #total ms1 line scans vs total ms2 line scans -> union to get overlap
    #differences + symmetric difference
    for scan in scans:
        ms1lines = linesofscans[scan]
        for line in ms1lines:
            ms1scans = set(scansoflines[line])
            ms1diffs = len(ms1scans.difference(scans))
            ms2diffs = len(scans.difference(ms1scans))
            scanunion = len(ms1scans.union(scans))
            ms1entropy[mainindex][line] -= ms1diffs
            ms2entropy[mainindex][line] -= ms2diffs
            symdiffentropy[mainindex][line] -= ms1diffs + ms1diffs
            unionentropy[mainindex][line] += scanunion
            #newentropylist.append([ms1diffs, ms2diffs, ms1diffs + ms2diffs, scanunion])
            #newentropylabels.append([mainindex, line])

assignmentresults = {} #primary: [results]
for mainindex, assignablems1lines in ms1entropy.items():
    primaries = maintoprimaryindex[mainindex]
    #for m, i, primary, scan, section, baseprimary in tg.tolist():
    for primary in primaries:
        #get scan: scansbyprimaryind -> get lines
        scan = scansbyprimaryind[primary]
        assessablems1lines = linesofscans[scan]
        #assessabledict = Counter(dict(zip(assessablems1lines, map(assignablems1lines.get, assessablems1lines))))
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
                assignmentresults[primary] = toplines
            else:
                #lone top rank
                #assign primary to ms1 line
                assignmentresults[primary] = assessablerankings[0][0]
        else:
            #lone top rank
            #assign primary to ms1 line
            assignmentresults[primary] = assessablerankings[0][0]
print(time() - nt)

incorrectcount = Counter()
notincorrectcount = Counter()

correct = 0
incorrect = 0
notincorrect = 0
for primary, results in assignmentresults.items():
    trueresult = linesbyprimaryind[primary]
    match results:
        case tuple():
            #outcome = set(results).intersection(trueresult)
            outcome = trueresult in results
            if outcome:
                notincorrect += 1
                notincorrectcount[len(results)] += 1
                primaryguesses[primary].append(2)
                ms1assignments[2].add(primary)
            else:
                incorrect += 1
                incorrectcount[len(results)] += 1
                primaryguesses[primary].append(1)
                ms1assignments[1].add(primary)
        case int():
            if results == trueresult:
                correct += 1
                primaryguesses[primary].append(0)
                ms1assignments[0].add(primary)
            else:
                incorrect += 1
                incorrectcount[1] += 1
                primaryguesses[primary].append(1)
                ms1assignments[1].add(primary)

print('MS1 Entropy')
print('correct:', correct)
print('incorrect:', incorrect)
print('not incorrect:', notincorrect)
print(correct + incorrect + notincorrect)
print('~')

assignmentresults = {} #primary: [results]
for mainindex, assignablems1lines in ms2entropy.items():
    primaries = maintoprimaryindex[mainindex]
    #for m, i, primary, scan, section, baseprimary in tg.tolist():
    for primary in primaries:
        #get scan: scansbyprimaryind -> get lines
        scan = scansbyprimaryind[primary]
        assessablems1lines = linesofscans[scan]
        #assessabledict = Counter(dict(zip(assessablems1lines, map(assignablems1lines.get, assessablems1lines))))
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
                assignmentresults[primary] = toplines
            else:
                #lone top rank
                #assign primary to ms1 line
                assignmentresults[primary] = assessablerankings[0][0]
        else:
            #lone top rank
            #assign primary to ms1 line
            assignmentresults[primary] = assessablerankings[0][0]
print(time() - nt)

incorrectcount = Counter()
notincorrectcount = Counter()

correct = 0
incorrect = 0
notincorrect = 0
for primary, results in assignmentresults.items():
    trueresult = linesbyprimaryind[primary]
    match results:
        case tuple():
            #outcome = set(results).intersection(trueresult)
            outcome = trueresult in results
            if outcome:
                notincorrect += 1
                notincorrectcount[len(results)] += 1
                primaryguesses[primary].append(2)
                ms2assignments[2].add(primary)
            else:
                incorrect += 1
                incorrectcount[len(results)] += 1
                primaryguesses[primary].append(1)
                ms2assignments[1].add(primary)
        case int():
            if results == trueresult:
                correct += 1
                primaryguesses[primary].append(0)
                ms2assignments[0].add(primary)
            else:
                incorrect += 1
                incorrectcount[1] += 1
                primaryguesses[primary].append(1)
                ms2assignments[1].add(primary)

print('MS2 Entropy:')
print('correct:', correct)
print('incorrect:', incorrect)
print('not incorrect:', notincorrect)
print(correct + incorrect + notincorrect)
print('~')

assignmentresults = {} #primary: [results]
for mainindex, assignablems1lines in symdiffentropy.items():
    primaries = maintoprimaryindex[mainindex]
    #for m, i, primary, scan, section, baseprimary in tg.tolist():
    for primary in primaries:
        #get scan: scansbyprimaryind -> get lines
        scan = scansbyprimaryind[primary]
        assessablems1lines = linesofscans[scan]
        #assessabledict = Counter(dict(zip(assessablems1lines, map(assignablems1lines.get, assessablems1lines))))
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
                assignmentresults[primary] = toplines
            else:
                #lone top rank
                #assign primary to ms1 line
                assignmentresults[primary] = assessablerankings[0][0]
        else:
            #lone top rank
            #assign primary to ms1 line
            assignmentresults[primary] = assessablerankings[0][0]
print(time() - nt)

incorrectcount = Counter()
notincorrectcount = Counter()

correct = 0
incorrect = 0
notincorrect = 0
for primary, results in assignmentresults.items():
    trueresult = linesbyprimaryind[primary]
    match results:
        case tuple():
            #outcome = set(results).intersection(trueresult)
            outcome = trueresult in results
            if outcome:
                notincorrect += 1
                notincorrectcount[len(results)] += 1
            else:
                incorrect += 1
                incorrectcount[len(results)] += 1
        case int():
            if results == trueresult:
                correct += 1
            else:
                incorrect += 1
                incorrectcount[1] += 1

print('Symdiff')
print('correct:', correct)
print('incorrect:', incorrect)
print('not incorrect:', notincorrect)
print(correct + incorrect + notincorrect)
print('~')

assignmentresults = {} #primary: [results]
for mainindex, assignablems1lines in unionentropy.items():
    primaries = maintoprimaryindex[mainindex]
    #for m, i, primary, scan, section, baseprimary in tg.tolist():
    for primary in primaries:
        #get scan: scansbyprimaryind -> get lines
        scan = scansbyprimaryind[primary]
        assessablems1lines = linesofscans[scan]
        #assessabledict = Counter(dict(zip(assessablems1lines, map(assignablems1lines.get, assessablems1lines))))
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
                assignmentresults[primary] = toplines
            else:
                #lone top rank
                #assign primary to ms1 line
                assignmentresults[primary] = assessablerankings[0][0]
        else:
            #lone top rank
            #assign primary to ms1 line
            assignmentresults[primary] = assessablerankings[0][0]
print(time() - nt)

incorrectcount = Counter()
notincorrectcount = Counter()

correct = 0
incorrect = 0
notincorrect = 0
for primary, results in assignmentresults.items():
    trueresult = linesbyprimaryind[primary]
    match results:
        case tuple():
            #outcome = set(results).intersection(trueresult)
            outcome = trueresult in results
            if outcome:
                notincorrect += 1
                notincorrectcount[len(results)] += 1
                primaryguesses[primary].append(2)
                unionassignments[2].add(primary)
            else:
                incorrect += 1
                incorrectcount[len(results)] += 1
                primaryguesses[primary].append(1)
                unionassignments[1].add(primary)
        case int():
            if results == trueresult:
                correct += 1
                primaryguesses[primary].append(0)
                unionassignments[0].add(primary)
            else:
                incorrect += 1
                incorrectcount[1] += 1
                primaryguesses[primary].append(1)
                unionassignments[1].add(primary)

print('Union')
print('correct:', correct)
print('incorrect:', incorrect)
print('not incorrect:', notincorrect)
print(correct + incorrect + notincorrect)
print('~')

#you only need to rank at the mainindex level, this below can be indented
#make individual entropyorganizers out of each diff + the union, what can you tell from any of these?
#then, if the multi-rank process and/or its derivative actually improve upon these 3, that's valuable information
#WHEN any of the above metrics work out, why? what do their organizers look like?
#^do the good results overlap? and are they distinguishable?
#this strategy is trying to link ms2 lines with ms1 lines, as opposed to the aboves linking a primary to an ms1 line
#assign mainindex to ms1line(s) then find the % of primaries that turn out to be right
#newentropyranks = stats.rankdata(test, method='dense', axis=0) - 1
#for (mainindex, ms1line), ranks in zip(newentropylabels, newentropyranks.tolist()):

#if there are redundant line-tuples in maininds am i missing data to iterate? -> YES
#i can use the union-labels as LOW-RISK BOUNDS -> these won't be non-incorrect
#also it seemed like the simple dual-combo setup was getting the same results when there was only 1 mergedgroup > length 1?

#aim for
#14683 mergedscans
#5803 > length 1
#101 max group length

#if there are redundant line-tuples in maininds am i missing data to iterate? -> YES
#i can use the union-labels as LOW-RISK BOUNDS -> these won't be non-incorrect, nvm
#^doesn't look promising but i should do a more in-depth comparison - NAH
#if the length of maininds is 2, then will this still work via iterlength and coiterlength? i think this screws up and skips those? no it should be fine, things have changed since the while-loop days

#the 1-1 dual comparison was faster, provided ~better to ~equal results as the fully-reciprocal comparison, so i'm rolling with it
#i suppose i should try and derive confidence from the counts, and perhaps pull in a few of the lower-counted values to get a lower bound estimate?

#i want to look at the differences in count of the CORRECT lines for each line that got it wrong
#^and how many others beat out the correct line
#^plus how many other lines are within that same error distance past the true line
#closest 2nd to the correct ones
#distance of correct to incorrect for the non-correct matches

#lastly, link the below with the indexing above and multiprocess from there
#i want to do a more serious mass-comparison from scan-to-scan in real data and compare what i've simulated here to this too
#^i need to find out how consistent the masses are across both lines and distributions because that seems to be the #1 thing that this entire process banks on, they ought to be pretty consistent though aye -> Skipping this for now, it should be consistent enough otherwise nothing would work, plus i've seen enough PRM data to know its fine, i don't know consistency across different isotopomers or whether they fragment completely different but i can figure this out later
#^in scanexploration i want to see how many frag intensities i can find that are consistent with the ms1 intensity flux, because jack shit looked to be accurate, maybe i can assign confidence this way?

#overall
#a lot of the incorrect lines are wrong, afaik, because there's like 3 correct answers and it only has 2 of them
#things that are completely wrong seem to be that way for logical reasons, which is the best you can ask

nt = time()

entropyorganizer = defaultdict(lambda: defaultdict(int)) #mainind: line: count
for scangroup in mergedscans:
    lineindices = {}
    massindices = {}
    for n, scan in enumerate(scangroup):
        lineindices[n] = set(linesofscans[scan])
        massindices[n] = set(mainindicesofscans[scan])
    blocked = set()
    if n > 0:
        for itercomb in massindices:
            iterset = lineindices[itercomb]
            for coitercomb in massindices:
                if itercomb != coitercomb:
                    combinationtuple = tuple(sorted([itercomb, coitercomb]))
                    coiterset = lineindices[coitercomb]
                    combintersection = iterset.intersection(coiterset)
                    itercombinds = massindices[itercomb]
                    coitercombinds = massindices[coitercomb]
                    if combintersection:
                        if combinationtuple not in blocked:
                            blocked.add(combinationtuple)
                            mainindintersection = itercombinds.intersection(coitercombinds)
                            if mainindintersection:
                                combintersection = tuple(combintersection)
                                for ind in mainindintersection:
                                    for c in combintersection:
                                        entropyorganizer[ind][c] += 1
                        iterdiff = iterset.difference(coiterset)
                        if iterdiff:
                            itermaininds = itercombinds.difference(coitercombinds)
                            label = tuple(iterset.difference(coiterset)) #automatically sorts
                        else:
                            #everything from the iter is within the coiter, mark everything as being from the itercomb key
                            itermaininds = itercombinds
                            label = tuple(iterset)
                        for ind in itermaininds:
                            for l in label:
                                entropyorganizer[ind][l] += 1
    else: #no competition
        for n, inds in massindices.items():
            for line in lineindices[n]:
                for mi in inds:
                    entropyorganizer[mi][line] += 1

print(time() - nt, 'entropy estimated')

#results = []
#for mainind, counts in entropyorganizer.items():
#    trueindices = linesbymainindex[mainind] #should be the answers...
#    counts = Counter(counts)
#    mostcommon = counts.most_common(len(counts))
#    if len(mostcommon) > 1: #> 1 result
#        if mostcommon[0][1] == mostcommon[1][1]:
#            maxcount = mostcommon[0][1]
#            #iterate and collect all
#            grouping = []
#            for line, c in mostcommon:
#                if c == maxcount:
#                    match line:
#                        case int():
#                            grouping.append(line)
#                        case tuple():
#                            for l in line:
#                                grouping.append(l)
#                else:
#                    break
#            result = tuple(set(grouping))
#            if trueindices.issubset(result):
#                outcome = tuple(result)
#            else:
#                #incorrect outcome
#                outcome = -1
#        else:
#            outcome = mostcommon[0][0]
#            result = mostcommon[0][0]
#            if outcome not in trueindices or len(trueindices) > 1:
#                #incorrect outcome via not matching everything
#                outcome = -2
#    else:
#        outcome = mostcommon[0][0]
#        result = mostcommon[0][0]
#        match outcome:
#            case int():
#                if outcome not in trueindices or len(trueindices) > 1:
#                    #incorrect outcome
#                    outcome = -3
#            case tuple():
#                if not trueindices.issubset(outcome):
#                    outcome = -4
#        #elif not trueindices.issubset(outcome):
#        #    outcome = -4
#    results.append([mainind, tuple(trueindices), outcome, result])
#
#nwrongs = 0
#correctoutcomes = 0
#notincorrectoutcomes = 0
#notincorrectdistances = Counter()
#badoutcomesbadmatches = Counter()
#badoutcomesgoodmatches = Counter()
#for r in results:
#    mainind, trueinds, outcome, result = r
#    if type(result) == int:
#        result = set((result,))
#    else:
#        result = set(result)
#    if type(trueinds) == int:
#        trueinds = set((trueinds,))
#    else:
#        trueinds = set(trueinds)
#    if type(outcome) is int and outcome < 0: #bad outcome
#        goodlength = len(trueinds.intersection(result))
#        badoutcomesgoodmatches[goodlength] += 1
#        badlength = len(trueinds.symmetric_difference(result))
#        badoutcomesbadmatches[badlength] += 1
#        nwrongs += 1
#    else: #good outcome
#        if trueinds == result:
#            correctoutcomes += 1
#        else:
#            notincorrectoutcomes += 1
#            distance = len(result.difference(trueinds))
#            notincorrectdistances[distance] += 1
#
#print(f'total correct: {correctoutcomes}')
#print(f'total wrong: {nwrongs}')
#print(f'total not incorrect {notincorrectoutcomes}')
#
#failures = []
#failcount = 0
#somewhatwrong = 0
#completelywrong = 0
#for mainind, lines in entropyorganizer.items():
#    trueindices = linesbymainindex[mainind]
#    tintersect = trueindices.intersection(lines)
#    if len(tintersect) != len(trueindices):
#        if len(tintersect) == len(lines):
#            somewhatwrong += 1
#        elif len(tintersect) == 0:
#            completelywrong += 1
#        failcount += 1
#        failures.append(mainind)
#
#singles = 0
#multiples = 0
#entropicboundaries = {} #mainind: [lines]
#for mainind, counts in entropyorganizer.items():
#    counts = Counter(counts)
#    mostcommon = counts.most_common(len(counts))
#    if len(mostcommon) > 1: #> 1 result
#        if mostcommon[0][1] == mostcommon[1][1]:
#            maxcount = mostcommon[0][1]
#            #iterate and collect all
#            grouping = []
#            for line, c in mostcommon:
#                if c == maxcount:
#                    match line:
#                        case int():
#                            grouping.append(line)
#                        case tuple():
#                            for l in line:
#                                grouping.append(l)
#                else:
#                    break
#            outcome = tuple(grouping)
#            multiples += 1
#        else:
#            outcome = mostcommon[0][:1]
#            singles += 1
#    else:
#        outcome = mostcommon[0][:1]
#        singles += 1
#    entropicboundaries[mainind] = outcome
#
#print(time() - nt, 'entropic boundaries')
#print(f'singles: {singles}, multiples: {multiples}')

print('~')
print('second measure:')

assignmentresults = {} #primary: [results]
for ms2line, assignablems1lines in entropyorganizer.items():
    primaries = maintoprimaryindex[ms2line]
    #for m, i, primary, scan, section, baseprimary in tg.tolist():
    for primary in primaries:
        #get scan: scansbyprimaryind -> get lines
        scan = scansbyprimaryind[primary]
        assessablems1lines = linesofscans[scan]
        #assessabledict = Counter(dict(zip(assessablems1lines, map(assignablems1lines.get, assessablems1lines))))
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
                assignmentresults[primary] = toplines
            else:
                #lone top rank
                #assign primary to ms1 line
                assignmentresults[primary] = assessablerankings[0][0]
        else:
            #lone top rank
            #assign primary to ms1 line
            assignmentresults[primary] = assessablerankings[0][0]
print(time() - nt)

#make trackedgroups numpy arrays
#entropyorganizer counts everything
#assess ms2lines by section -> iterate entropyorganizer
#from ms2lines get the primaries and sections
#line: section: primaries

#assess how many primaries are correct/incorrect

incorrectcount = Counter()
notincorrectcount = Counter()
incorrectdifferences = Counter()
incorrectsinglediffs = 0
notincorrectdifferences = Counter()
notincorrectsinglelengths = Counter()
notincorrectmultilengths = Counter()

correct = 0
incorrect = 0
notincorrect = 0
for primary, results in assignmentresults.items():
    trueresult = linesbyprimaryind[primary]
    primaryscan = scansbyprimaryind[primary]
    linesofprimaryscan = linesofscans[primaryscan]
    match results:
        case tuple():
            #outcome = set(results).intersection(trueresult)
            outcome = trueresult in results
            if outcome:
                notincorrect += 1
                notincorrectcount[len(results)] += 1
                primaryguesses[primary].append(2)
                classicassignments[2].add(primary)
                notincorrectdifferences[len(linesofprimaryscan) - len(results)] += 1
                if len(linesofprimaryscan) - len(results) == 0:
                    notincorrectsinglelengths[len(results)] += 1
                else:
                    notincorrectmultilengths[len(results)] += 1
            else:
                incorrect += 1
                incorrectcount[len(results)] += 1
                primaryguesses[primary].append(1)
                classicassignments[1].add(primary)
                incorrectdifferences[len(linesofprimaryscan) - len(results)] += 1
        case int():
            if results == trueresult:
                correct += 1
                primaryguesses[primary].append(0)
                classicassignments[0].add(primary)
            else:
                incorrect += 1
                incorrectcount[1] += 1
                primaryguesses[primary].append(1)
                classicassignments[1].add(primary)
                incorrectsinglediffs += 1

print('correct:', correct)
print('incorrect:', incorrect)
print('not incorrect:', notincorrect)
print(correct + incorrect + notincorrect)

plt.bar(incorrectcount.keys(), incorrectcount.values())
plt.show()

plt.bar(notincorrectcount.keys(), notincorrectcount.values())
plt.show()

plt.bar(incorrectdifferences.keys(), incorrectdifferences.values())
plt.show()

plt.bar(notincorrectdifferences.keys(), notincorrectdifferences.values())
plt.show()

plt.bar(notincorrectsinglelengths.keys(), notincorrectsinglelengths.values())
plt.show()

plt.bar(notincorrectmultilengths.keys(), notincorrectmultilengths.values())
plt.show()
#^looks like it can narrow a lot of stuff down pretty well, just not a majority

print(incorrectsinglediffs, 'incorrectsinglediffs')

#ms1 == symdiff, so i'll leave symdiff out
#primaryguesses = defaultdict(list) #primary: [ms1, ms2, union, classical] as indices
#primary: [ms1, ms2, union, classical] as indices
#correct -> 0
#incorrect -> 1
#not incorrect -> 2
#find all unique combinations -> count
#other lists with the same indices:
#percent of total counts -> get rank of percentages
#number of total counts -> get rank of counts


tupletest = [tuple(i) for i in primaryguesses.values()]
tguess = Counter(tupletest)
rawproducts = set(itertools.product([0,1,2], repeat=4))
print(len(tguess), 'vs.', len(rawproducts.difference(tguess)))

#fro doing shit like len(ms1assignments[0].intersection(classicassignments[0]))
#i can tell for the most part, they overlap a lot in what is correct/incorrect etc
#its definitely worth an exploration at some point into how i might use cleverness or a multirank in order to better take some of the high correct # from the classic approach into the ms1 approach which has less correct but WAY less incorrect -> all siphoned into not incorrect. its good enough for now, i'm going to move forward
#there might be a good way to squeeze in a decision-tree like exploration, but the different kind of matches that have the same outcome, ie [0, 1, 1, 1] == [2, 0 , 0, 0] from the data perspective need to be grouped prior, and idk if this will be of benefit when i look at the actual numbers


#fine = 0
#failcount = 0
#somewhatwrong = 0
#are there maininds in scans that don't have their determined boundarylines?
#for mainind, lines in entropicboundaries.items():
#    indscans = scansbymainindices[mainind]
#    boundaryscans = tuple(set(itertools.chain(*[scansoflines[i] for i in lines])))
#    if indscans.difference(boundaryscans):
#        print('mainind:', mainind)
#        print('lines:', lines)
#        print('indscans:', indscans)
#        print('boundaryscans:', boundaryscans)
#        print('entropyorganizer:', entropyorganizer[mainind])
#        print('linesbymainindex:', linesbymainindex[mainind])
#        print('lines of scans:', [linesofscans[i] for i in indscans])
#        print('scans of ^lines:', [{j: scansoflines[j] for j in linesofscans[i]} for i in indscans])
#        break

#so basically, the logic engine is fine, kind of
#but there's heirarchies i'm missing
#i think i need to start from the group of lines present in each scan
#and work down to eliminate whichever ones make sense to eliminate
#i keep score, but i don't allow logic to dominate when it should
#i change the nature of the competition when i add +1 to each thing
#in reality, there should be at least ONE line from each of the scans
#no excuse to miss lines in a scan when the supposed identified line isn't in there!

#i also want to use pure differences and pure unions and see the difference
#plus i want to do the union only once, so block it the 2nd time when the iter/coiter reverse
#also just do the iterdiff not the coiterdiff, as the iteration repeats the opposite iter/coiter

#lines can only compete WITHIN scans! albeit using the information across them
#im not setting the baseline

#so basically
#i might implement all ^this later
#but if i come to a mainind in a scan that doesn't have its boundary lines, i'll just allow it to match whatever
#atm its more important to move onto the ms2 line model
#i'm not focusing on the boundary aspect right now because when its wrong, its just going to be double-wrong when i try and assume the wrong thing
#^ie, even if i try to set up a baseline "must compete in a certain scan" constraint, it looks like (from the data itself) that it will tend to be wrong anyways!



#i need to find out where the failed primaryinds come from
#lines that are in multiple scans?
#do they show up correctly?
#do they literally show up in a way that makes them look like the entropyorganizer has it right???
#^right? but removing the minimum cutoff didn't cause problems

#a weakness of this logical inference plan is when a minor, tiny, intensity fragment is paired with a major fragment in another scan - it could probably throw the logic off
#the natural evolution of this process would then be to compare fragment ion intensities of different scans and try to pair fragments to a line via the line's intensity at that moment

#so current plan:
#library generation with random probs for fragments
    #- i need to track line positions to know which fragments to call upon
#ms2 intensity-based entropy
#fix ms1 distribution matching
    #- for whatever metric it is in the ms1 distribution matching process where you can have a max of n incorrect intensity ranks -> this should instead become a metric that plugs into the ms2 identifying process. ie something that has 1 rank off is better than 5. might want to check all of them though, i get a feeling its better to work with what might be mismatches.
#go pipe deconvoluted dists as information into the real ms2 ion matching process
#^as well as start caching theoretical distribution results
#fragment entropy ranking!

#a potentially more consistent way to generate fragment ions is to make a fragment pool beforehand? and randomly assign some fragments to be more or less probable a priori. this way you can achieve some level of consistency
#^i'll do this when i implement sequences
#it may be better to input a real proteome than random peptides, or rather, you can compare the two. it doesn't seem that random peptides have enough redundancy, they're mostly all unique
#^might be a cool comparison, generate the same # of AAs across fake proteins/peptides and see how often certain sequences come up, the difference of unique sequences
#^a good monte carlo

#intensity-entropy:
#lines need to have > 1 scan i suppose?
#^no, but they need to have a line that has > 1 scan, they can have other lines as long as at least one has > 1 scans to it
#maybe i could think of this as fragment-centric? check all scans relative to each fragment and determine which it should likely belong to

 
#post-entropy:
#i can enhance this whole simulation by generating random peptides, then producing their fragments
#then try to id the fragments based simply on whats visible, this is a worthwhile endeavor
#it might also be worth simulating the line model here when dealing with means vs performance, idk if this will apply well to real data after but the door is open

#implement noise via  density concept for ms1
#implement random noise to random ms2 scans
#implement random noise distributions in the ms1 level too, that will have random noise ms2 fragments - BUT these distributions should be based on organic molecules and follow the rules of chemistry
#you should try simulating both higher and lower resolution subisodepths than what you "allow" for the library, what are the impacts?









#~

#re-vamping with line-model and multiprocessing (later i guess):
#planning for a new mainindexing process:
#simple line model from one to the next of k=1 nn?
#make trackedgroups with mainind: [mass, intensity, primaryind] or just primaryind?
#may need to make scansofprimaryinds
#output of the line model will be {mainind: (scans)}
#this will cross with {scan: (lines)}
#after this, reduce the groups of maininds with the same scanpatterns
#trackedgroups needs to be made via adjacent rts, but it also needs to be formed within overlapping mass ranges
#^you'll need to keep a running window estimate of each mass origin
#if the scan window completely leaves what an original trackedgroup was working off of, make a new trackedgroup -> or rather, allow for a merge between whatever previous overlapping scans had this and the newer ones moving forward, they can share the middleman in this case because who knows if its actually from that, all that would be certain is that the window is now different
#
#i might be able to pre-define mass boundaries for the line model
#but that doesnt tell me how to have the entropy engine handle the things that fall between 2+ boundaries
#track where it starts, the range -> follow it -> if it leaves the range, start a new one that also matches the ~previous range
#you'd allow for a half-identification of the ion in the middle i guess
#where other ions can only be matched once, and it depends on which distributions are being matched
#
#or maybe i should keep this ppm radius intersection merge THEN apply a line model from within the merges? -> Nah the scangroup is already doing this
#you may see legit differences across the mass ranges
#
#the question im not able to ask, as a singe mainind appears across 2 mass ranges, is ARE they related?!
#intensity probably won't answer the question
#i can't do entropy OTF to determine whether the lines involved likely are the same, and even if i could then it wouldn't ALWAYS work, it would need more scans which i can't rely on
#maybe a scanexploration-like comparison across scans via nn to determine a minimum cutoff for the ms2 line model?
#mass is likely the best indicator
#and i can perhaps normalize the distances by intensity
#so from here, to scanexploration, to get a legitimate comparison either way - NAH
#
#so basically
#keep this initial nn grouping at ~some ppm
#then from within the group
#guarantee only one signal per scan
#and also minimize mass distance of all matches
#make it so there's potentially multiple real "lines" that come out of any one grouping if appropriate
#and if there's, say, 3 scans with 3 datapoints, and 2 of them connect WAY better, their intensity should indicate mass accuracy
#so if the thing that has larger distance to match masses has more intensity, i guess its a bad match? this sounds weird
#another thing to consider is that the total ppm tolerance should be the maximum diameter, and if linking masses makes a groups diameter larger than this, remove the outliers
#a signal can never leave the original ms1 scan range it was found in, it can connect to ranges overlapping that - but it MUST remain within it - which is like a line model statement
#i may have room here for an average mass convergence as a method of cutting off a bad match?
#
#i want to keep it the same as it is rn, but it still needs to factor in scan windows
#basically, a line model, with the ppm limit as entered
#line collection will stop when every line where the primaryind started is no longer present -> if it continues then mark it as a potential double
#^because there's no reason each mainind only needs to be attributed to one thing across multiple scans if you can determine its not just one thing - BUT you can just split them into 2 things at that point and draw a "vague split" wherever new lines enter the scan window
#
#so basically:
#ms2 line starts, track ms1 lines, track scan window, and if either of these end the ms2 line ends
#the ms2 line model will be able to SHARE datapoints! ie if two non-overlapping scans happen to have lines that overlap in a third window, and they both share a mass that shows up in all 3 scans, that third datapoint can be present in both ms2 lines and identified twice
#so there's sharing redundancy
#if an ms2 line tries to continue when none of the original lines do, the tracked ms1 line additions into the data will be used as starting points for a new line with shared datapoints
#
#so essentially you can keep a line through complicated differences by splitting it into SECTIONS
#each SECTION will have its own constraints that require an identification based on the ms1 lines present
#the section is represented by the ms1 line combinations
#if the mass range of a scan overlaps with previously handled scan windows, then it creates a different SECTION
#the sections will be the constraints for boundaries, requiring a solution to be made in each one
#the best found solutions will come out on top from the existing process
#then each section will go down the ladder in entropyorganizer and pick the best guess it can get
#so i dont end up needing to differentiate a trackedgroup by lines then, seeing as they have sections, it can all still be one mainind, and the output boundaries will be based on sections which will be specific scans with specifis solutions
#^so then if two existing lines that dont have overlapping scan windows, match to the same ms2 point that then overlaps with both of these windows, you can merge the groups
#i'll merge the sections after the line model i guess, instead of corrections
#you can intersection merge them based on a primary ind match while allowing multiple ms2 lines to match the same thing as long as they came from diferent scan windows?
#BUT, there can only be one primaryind per scan, this plus the merging might be tricky
#^but i guess this will just be the correct/merging process, pick the closest one?
#
#ok so there's a deeper problem that occurs in the entropic boundary results:
#when you have multiple lines that are the determined entropic boundaries of a mainind
#BUT, you're matching that mainind in a scan where some of the determined lines don't exist
#THEN you can match that mainind to the subset of lines in the entropic boundaries
#AND you can later match the same mainind in a different scan to a different line of the same boundary
#in fact that mainind can always be matched to ALL of the lines of that boundary
#because thats how it works
#so i dont need to track which lines are relevant to that mainind
#i dont need to track ms1 lines for any mainind
#i can just match that mainind to different line/peptide combinations later and thats fine
#because the identifications across all scans for any lines fragment match will always remain consistent
#and you cant disprove its a multi-matching ion
#so i dont need to care if im labeling the same mass across different lines as the same mainind?
#BUT then the problem arises
#what if from the determined entropic boundaries
#the mainind is present but NONE of the determined lines are!
#does this ever happen?
#IF so, then you need to ADD other lines to the boundaries BECAUSE of those scans
#i dont use a defensive boundary identification?
#
#i need to test if ALL of the lines determined in the boundaries are ALWAYS present wherever the mainind persists
#IF not, then i need to add other lines to the boundaries
#i can pick the extra-and-necessary lines off from the rank-order of entropyorganizer
#test this in the simulation and the real data
#if masses: #no empty lists in ms2scans anymore, shouldn't need this


#nt = time()

#signals can merge and coalesce into merges ONLY when there are intersecting lines in the section
#no need for mass ranges
#so redundant matches needs to be re-done based on whether a signal matches to an ms2 line of 2 different sections

#so if two matches align, but no lines do, make a different line
#BUT, should you not determine this prior? what if the nn matches to a line with no overlaps when it has one that both overlaps and aligns that didn't match?
#so basically, and this is to my advantage with the nn setup i've made, simply select relevant baselist that overlaps with the current section
#^just keep track of their sections in a separate array

#for ms2 line removals you can use the ms1 scan window
#have a second grouprange-like factor that tracks the min/max ms1 window of an ms2 line
#and if the window is touched a total of minpoint times or whatever, then finalize that ms2 line

#i'll need to keep a running list of ms2 lineids to substitute for trackedma because that is the only thing that won't be compatible

#fullentropyorganizer = defaultdict(lambda: defaultdict(int)) #ms2line: line: count
#
#ms1assignmentresults = {} #primary: [results]
#ms2assignmentresults = {} #primary: [results]
#symdiffassignmentresults = {} #primary: [results]
#unionassignmentresults = {} #primary: [results]
#
#minmovinginds = 10 #parameter
#deadsignal = 20
#
##uidcount = 0
#roundcutoff = 0
#tglens = []
#
#sectionindex = 0
#mainind = 0
#entropylength = 0
#oldentropylength = len(entropyorganizer)
##linesbymainindex = defaultdict(set) #main index: [lines]
##maintoprimaryindex = defaultdict(list) #main index: [primary indices]
##primarytomainindex = {} #primary index: main index
##ms2linesofscans = defaultdict(list) #scan: [main indices], previously known as scandict
##scangroupbyline = defaultdict(list)
##scansbymainindices = defaultdict(set) #main index: [scans]
#primaryassignmentresults = {} #primary: assigned ms1 line(s)
#for groupn, group in enumerate(mergedscans):
#    #group is pre-sorted
#    linesofprimaries = {} #primary: line
#    primariesbyline = defaultdict(list) #line: [primaries]
#    scancount = group[0]
#    #scan = msrun[scancount]
#    #mza = scan['m/z array']
#    #intensities = scan['intensity array']
#    mza, intensities, primaries = ms2scans[scancount]
#    primaries = primaries.astype(int)
#    previousdata = mza.copy()
#    primarybases = np.arange(primaries.size)
#    
#    ms1lines = linesofscans[scancount]
#    scanlist = np.repeat(scancount, mza.size)
#    sectionlist = np.repeat(sectionindex, mza.size)
#    
#    coords = np.stack((mza, intensities, primaries, scanlist, sectionlist, primarybases), axis=1).reshape(mza.size, 1, 6).tolist()
#    
#    #previousindices = np.arange(mza.size) + uidcount
#    previousindices = np.arange(mza.size)
#    uids = previousindices.tolist()
#    #uidcount += len(mza)
#    uidcount = max(uids)
#
#    ms2trackedgroups = {} #ms2 line: [[mass, i ntensity, primary, scan, section]] #primaries should be sorted by mass and scan, so by sorting by primary you always sort by time
#    #trackedma = {} #latest moving average mass of trackedgroup: lineuid
#    linedeletioncounter = defaultdict(int) #lineuid: notmatched count
#    #^do you need a line deletion counter? i guess so
#    #^i would expect this to matter more for DIA data, and less for DDA because of the need to associate it with an ms1line and the section used for doing so
#    #^im going to stick with it for now
#    groupmovingaverages = {} #ms2 line: latest moving average of line
#    groupdifftoma = {} #ms2 line: moving difference to moving average
#    #mainindex: massrange -> all masses will be live? or you'll index-select which masses to match based on the window
#    #so instead of the mass range index ill use the section to dictate what is connected where
#    #massrangeindices = {} #mass range index???: [main indices] #NOPE
#    groupdifftoma = {} #ms2 line: moving difference to moving average
#    groupranges = {} #main index: [minimum scan mass, maximum scan mass]
#    windowranges = {} #main index: [minimum ms1 window mass, maximum ms1 window mass]
#    #mainindices are ms2lines
#    
#    sectionbyprimaryindex = {} #primary: section, for later calling this up during/pre ranking
#    sectionofms1linegroups = {} #(sorted ms1 lines): section, check if it exists first - if not then make a new one
#    sectionsoflines = defaultdict(set) #ms2 line: [sections], this is used to test for line intersection (which acts as mass range overlap) OTF
#    ms1linesofsections = {} #section: [ms1 lines]
#    #mainbyprimaryindex = {} #primary: main
#    #im not going to use a model tracker because the main point of it for ms1 processing is to determine when and where the sprayer is bubbling, but here i dont expect to see the same differences because in ms2 scans i pretty much expect MOST scans to have more novel masses than not? i guess maybe i could still visualize this to confirm, but i dont have a need for it now and i can always do it later
#    accumulatedwindowlines = defaultdict(set) #ms2 line: [all ms1 lines that have been interacted with]
#    scansofsections = defaultdict(list) #section [scans]
#    
#    sectionofms1linegroups[ms1lines] = sectionindex
#    ms1linesofsections[sectionindex] = tuple(ms1lines)
#    #for line in ms1lines:
#    #    sectionsoflines[line].add(sectionindex)
#    #^supposed to be ms2 lines
#    accumulatedwindowlines.update(zip(uids, itertools.repeat(set(ms1lines), mza.size)))
#    
#    flatmasslist = mza.tolist()
#    #trackedma.update(zip(flatmasslist, uids))
#    ms2trackedgroups.update(zip(uids, coords))
#    groupmovingaverages.update(zip(uids, flatmasslist))
#    elen = len(uids)
#    groupdifftoma.update(zip(uids, np.zeros(elen).tolist()))
#    groupranges.update(zip(uids, np.stack((mza, mza), axis=1)))
#    windowranges.update(zip(uids, np.repeat([ms2scanbounds[scancount][1:]], mza.size, axis=0)))
#    sectionbyprimaryindex.update(zip(primaries, sectionlist.tolist()))
#    sectionsoflines.update(((u, set((sectionindex,))) for u in uids))
#    scansofsections[sectionindex].append(scancount)
#    linesofprimaries.update(zip(primaries, uids))
#    primariesbyline.update(zip(uids, [[i] for i in primaries]))
#    
#    #linesofscans is already tuple-sorted
#    #primaries can be made here for the real process, and just taken from above in this simulation
#    #i'm just going to run forward with this for now, check the wides later and see if it performs better at keeping them narrow than the old mainindex process
#    for scancount in group[1:]:
#        trackedkeys = {} #latest mass in a trackedgroup: lineid
#        
#        #scan = msrun[scancount]
#        #mza = scan['m/z array']
#        #intensities = scan['intensity array']
#        #^in the real deal i can assign primaries here OTF
#        mza, intensities, primaries = ms2scans[scancount]
#        primaries = primaries.astype(int)
#        primarybases = np.arange(primaries.size)
#        massesbyprimary = dict(zip(primaries.tolist(), mza.tolist()))
#        intensitiesbyprimary = dict(zip(primaries.tolist(), intensities.tolist()))
#        primarybasebyprimary = dict(zip(primaries.tolist(), primarybases.tolist()))
#        
#        ms1lines = linesofscans[scancount]
#        scanlist = np.repeat(scancount, mza.size)
#        scanbounds = ms2scanbounds[scancount][1:]
#        lowerscanbound, upperscanbound = scanbounds
#        
#        if ms1lines in sectionofms1linegroups:
#            workingsectionindex = sectionofms1linegroups[ms1lines]
#        else:
#            sectionindex += 1
#            sectionofms1linegroups[ms1lines] = sectionindex
#            #for line in ms1lines:
#            #    sectionsoflines[line].add(sectionindex)
#            ms1linesofsections[sectionindex] = tuple(ms1lines)
#            workingsectionindex = sectionindex
#        scansofsections[workingsectionindex].append(scancount)
#        
#        sectionlist = np.repeat(workingsectionindex, mza.size)
#        sectionbyprimaryindex.update(zip(primaries, sectionlist.tolist()))
#        
#        #get ms1 lines
#        #determine if the section exists
#        #check for line intersections
#        #intersectedinds = []
#        nonintersectedinds = []
#        fullintersectedinds = []
#        indicesofaccumulation = defaultdict(list) #accumulation instance: [intersected indices]
#        for n, p in enumerate(previousindices.tolist()):
#            #its determined whether these actually get added to accumulatedwindow when a match/nonmatch is made
#            intersectedinstance = accumulatedwindowlines[p].intersection(ms1lines)
#            if intersectedinstance:
#                indicesofaccumulation[tuple(sorted(intersectedinstance))].append(n)
#                fullintersectedinds.append(p)
#            else:
#                nonintersectedinds.append(n)
#        #^here, i'm going to make separate lists within a dict when determining intersections, and organize it by what is being intersected with, ie accumulatedwindowlines
#        
#        #for each unique accumulation will come a separate process of matching
#        #then organize primaries by their accumulated instances, accumulation instance: [primaries]
#        #^and intersection merge these to find which ms2lines need to be merged
#        #newmodelremovals = []
#        
#        #i want it so that one mass from mza can match two masses in different intersectedinstances
#        #but i dont want an intersectedinstance to match two different mza masses
#        #but, in a contradictory way, this allows a single mass, that's now in two lines, to match two different things later in a single following scan
#        #UNLESS you can merge the two lines OTF, a post-merge would screw everything up, but if you can salvage OTF then you might be good to roll
#        
#        #in a a practical sense:
#        # - across different intersectedinstances, there can be redundant catches
#        # - within a single intersectedinstance, there can be no redundancy
#        # - this is already handled by the redundant bit below
#        #my real problem comes later when mixing found/unfound across the loop
#        #i'll organize everything post-redundancy reduction to be outside of the intersectedinstance loop
#        
#        fullmassdists = []
#        fullfoundinds = []
#        fullfoundprimaries = []
#        for intersectedinstance, intersectedinds in indicesofaccumulation.items():
#            intersectedmasses = previousdata[intersectedinds] #trackedmasses with ms1 lines that are relevant to this ms2 scan
#            intersectedkeys = previousindices[intersectedinds] #indices of the above^
#            
#            baseind = 0
#            catches = []
#            massdists = []
#            matchedprimaries = []
#            #a k=1 nearest neighbors for signal-processing
#            #this is iterating over numpy arrays because its slower to conver them to lists and slower to index a list
#            #picking whatevers closer in intensity might start to fail as a concept if the ms1 scans are more spaced out, or perhaps boxcar'd
#            for fn, f in enumerate(mza.tolist()):
#                mindist = np.inf
#                for n, b in enumerate(intersectedmasses[baseind:]):
#                    ikey = tuple(accumulatedwindowlines[intersectedkeys[n]])
#                    dist = abs(b-f)
#                    if dist < mindist:
#                        minind = n + baseind
#                        mindist = dist
#                    elif dist == mindist:
#                        #two new masses have symmetrical distances to existing moving average
#                        #choose whichever is within the original lines range
#                        #currentind = trackedma[b]
#                        currentind = intersectedkeys[n]
#                        linerange = groupranges[currentind]
#                        othermass = intersectedmasses[minind]
#                        currentmatch = b > linerange[0] and b < linerange[1]
#                        othermatch = othermass > linerange[0] and othermass < linerange[1]
#                        if currentmatch and not othermatch:
#                            #new match wins
#                            minind = n + baseind
#                            mindist = dist
#                            #no other distances will be closer
#                            break
#                        elif othermatch and not currentmatch:
#                            #old match wins
#                            #no other distances will be closer
#                            break
#                        else:
#                            #either both or neither are within the range
#                            #switch from comparing masses to comparing intensities
#                            currentintensity = ms2trackedgroups[currentind][-1][1]
#                            otherintensity = ms2trackedgroups[minind][-1][1]
#                            massintensity = intensities[fn]
#                            cabs = abs(massintensity - currentintensity)
#                            oabs = abs(massintensity - otherintensity)
#                            if cabs < oabs:
#                                #current mass wins out
#                                minind = n + baseind
#                                mindist = dist
#                                #no other masses will be closer
#                                break
#                            else:
#                                #other mass wins out
#                                #no other masses will be closer
#                                break
#                    else:
#                        break
#                matchedprimaries.append(primaries[fn])
#                massdists.append(mindist)
#                catches.append(minind)
#                baseind = minind
#            massdists = np.array(massdists)
#            catches = np.array(catches)
#            
#            #found = intersectedmasses[catches] #matched group moving averages of existing lines
#            foundinds = intersectedkeys[catches]
#            #if (catches != foundinds).any():
#            #    print('catches != foundinds')
#            #^this worked, catches is a good thing
#            #uf, ufc = np.unique(found, return_counts=True)
#            uf, ufc = np.unique(foundinds, return_counts=True)
#            ub = ufc > 1
#            redundants = np.any(ub)
#            #finding redundant matches
#            #if there are redundant matches from different sections:
#            #either merge those ms2 lines here, or set them up to merge later and allow them both the chance to match
#            #set them up for merging later only if they do match
#            #if they're from the same section, ditch them and keep only the closest one
#            #i may be able to handle this in the nearest neighbors itself..?
#            
#            #deleting redundant indices that were found
#            removals = []
#            if redundants:
#                for umatch in uf[ub].tolist():
#                    mwhere = np.where(foundinds == umatch)[0]
#                    mwdists = massdists[mwhere]
#                    mwdargmin = mwdists.argmin()
#                    removals.extend(np.delete(mwhere, mwdargmin).tolist())
#            
#            finalmassdists = np.delete(massdists, removals)
#            finalprimaries = np.delete(matchedprimaries, removals)
#            finalfoundinds = np.delete(foundinds, removals)
#            
#            fullmassdists.extend(finalmassdists.tolist())
#            fullfoundinds.extend(finalfoundinds.tolist())
#            fullfoundprimaries.extend(finalprimaries.tolist())
#            #fullfound.extend(found.tolist())
#            #fullremovals.extend(removals)
#        #set1diff(fullremovals, fullfoundinds) #to get the definite removals
#        #everything in foundinds is already good to go
#        #you may need to carry primaries and matchedlines in the above process
#        #fullfound = list(map(groupmovingaverages.get, fullfoundinds))
#        #fmza = list(map(massesbyprimary.get, fullfoundprimaries))
#        #fint = list(map(intensitiesbyprimary.get, fullfoundprimaries))
#        
#        #removing redundant matches -> the line ended for these ones, or its skipping an index
#        #flen = len(fmza)
#        #fscanlist = np.repeat(scancount, flen)
#        #fsectionlist = np.repeat(workingsectionindex, flen)
#        #coords = np.stack((fmza, fint, fullfoundprimaries, fscanlist, fsectionlist), axis=1).reshape(flen, 1, 5).tolist()
#        
#        primaryorganizer = defaultdict(list) #primary: [foundinds]
#        for fi, p in zip(fullfoundinds, fullfoundprimaries):
#            primaryorganizer[p].append(fi)
#        
#        #fullremovals = np.setdiff1d(previousindices, fullfoundinds)
#        removalprimaries = np.setdiff1d(primaries, fullfoundprimaries).tolist()
#        
#        if removalprimaries:
#            #things that had a redundant match, and weren't taken, from mza are put up as new lines
#            #excluded = mza[removals]
#            ebases = list(map(primarybasebyprimary.get, removalprimaries))
#            excluded = list(map(massesbyprimary.get, removalprimaries))
#            elen = len(excluded)
#            #retentiontimes = np.repeat(rt, elen)
#            #eint = intensities[removals]
#            eint = list(map(intensitiesbyprimary.get, removalprimaries))
#            #ecoords = np.stack((excluded, retentiontimes, eint), axis=1).reshape(elen, 1, 3).tolist()
#            ecoords = np.stack((excluded, eint, removalprimaries, np.repeat(scancount, elen), np.repeat(workingsectionindex, elen), ebases), axis=1).reshape(elen, 1, 6).tolist()
#            uids = (np.arange(elen) + uidcount).tolist()
#            uidcount += elen
#            ms2trackedgroups.update(zip(uids, ecoords))
#            trackedkeys.update(zip(excluded, uids))
#            #trackedma.update(zip(excluded, uids))
#            groupmovingaverages.update(zip(uids, excluded))
#            groupdifftoma.update(zip(uids, np.zeros(elen).tolist()))
#            groupranges.update(zip(uids, np.stack((excluded, excluded), axis=1)))
#            windowranges.update(zip(uids, np.repeat([scanbounds], elen, axis=0)))
#            linesofprimaries.update(zip(removalprimaries, uids))
#            primariesbyline.update(zip(uids, [[i] for i in removalprimaries]))
#            sectionsoflines.update(((u, set((workingsectionindex,))) for u in uids))
#            #lastmatchtime.update(zip(uids, np.repeat(rt, len(uids))))
#        #else:
#            #coords = np.stack((fmza, retentiontimes, fint), axis=1).reshape(fmza.size, 1, 3).tolist()
#            #coords = np.stack((mza, intensities, primaries, scanlist, sectionlist), axis=1).reshape(mza.size, 1, 5).tolist()
#        #found = found.flatten().tolist() #does this need to flatten?
#        #foundinds = foundinds.tolist()
#        
#        #THIS CUTOFF PROCESS WILL NEED TO CHANGE?!
#        #^not if it only bases things on matches groups, it should be fine aye
#        sorteddistances = np.sort(fullmassdists) #this moved the cutoff up a little bit, higher mass-range lines seemed to be slightly better connected, there's more mass variation up there it seems...
#        mbool = sorteddistances <= sorteddistances[:,None] #isn't there an easier way to do this?, isn't this just arange?
#        countsums = mbool.sum(axis=0) / sorteddistances.size
#        #mbool = stats.rankdata(sorteddistances, method='max')[::-1] #^yep, but it's just slightly different because this is a ranking and not counting the things beyond it - so i'll skip this for now - the last of a dense rank needs a +1?
#        #the sumcounts below is generally the same thing, but will differ at values where mbool would have given duplicate entries, doesn't change anything major enough to change the result
#        sumcounts = sorteddistances.cumsum() / sorteddistances.sum()
#        #^this will throw a warning
#        if sumcounts.size > 0:
#            if np.isnan(sumcounts).any():
#                sumcounts = np.nan_to_num(sumcounts, 0)
#            mincomboind = (countsums + sumcounts).argmin()
#            mincombo = sorteddistances[mincomboind]
#            #moving average of average of dists under mincombo
#            explicitcutoff = sorteddistances[sorteddistances <= mincombo].mean()
#            roundcutoff = (roundcutoff * scancount + explicitcutoff) / (scancount + 1)
#        
#        #modifying things that are already being tracked
#        #fmzaremovals = []
#        #foundremovals = []
#        foundindremovals = []
#        #for c, f, tid in zip(coords, found, foundinds): #could this loop be reduced to just foundinds while i generate everything else OTF?
#        for primary, foundinds in primaryorganizer.items():
#            nf = massesbyprimary[primary]
#            nint = intensitiesbyprimary[primary]
#            pbase = primarybasebyprimary[primary]
#            c = [nf, nint, primary, scancount, workingsectionindex, pbase]
#            modify = []
#            for tid in foundinds:
#                f = groupmovingaverages[tid]
#                #modify = False
#                #nf = c[0]
#                d = abs(f-nf)
#                #tid = trackedma[f]
#                tgroup = ms2trackedgroups[tid]
#                tlen = len(tgroup)
#                lastmass = tgroup[-1][0]
#                rmin, rmax = groupranges[tid]
#                grange = rmax - rmin
#                rep = False
#                if tid == -1: #change -1 to the trackedkey of interest
#                    rep = True
#                #this is for when the moving decision fails for something within the existing range, ain't no thang
#                rangepass = nf <= rmax and nf >= rmin
#                distancepass = abs(nf - lastmass) < grange / 2
#                if rangepass or distancepass:
#                    oldma = groupmovingaverages[tid]
#                    nma = (oldma * tlen + nf) / (tlen + 1)
#                    nmadiff = abs(oldma - nma)
#                    groupmovingaverages[tid] = nma
#                    madiff = groupdifftoma[tid]
#                    groupdifftoma[tid] = (madiff * (tlen - 1) + nmadiff) / tlen
#                    #ndiff = (madiff * (tlen - 1) + nmadiff) / tlen
#                    #modify = True
#                    modify.append(tid)
#                    if rep:
#                        print(1, nf, nmadiff, madiff, rmax-rmin)
#                #generally, this is good for on-the-fly decision making when the moving target is outside the existing mass range. This dominates later on, where it's more robust
#                elif tlen >= minmovinginds:
#                    oldma = groupmovingaverages[tid]
#                    madiff = groupdifftoma[tid]
#                    nma = (oldma * tlen + nf) / (tlen + 1)
#                    nmadiff = abs(oldma - nma)
#                    if rep:
#                        print(2, nf, nmadiff, madiff, rmax-rmin)
#                    #if nmadiff <= np.mean(madiff): #max(madiff) + (2*np.mean(madiff)):
#                    #if nmadiff <= np.mean(madiff):
#                    if nmadiff <= madiff:
#                        groupmovingaverages[tid] = nma
#                        groupdifftoma[tid] = (madiff * (tlen - 1) + nmadiff) / tlen
#                        #ndiff = (madiff * (tlen - 1) + nmadiff) / tlen
#                        #modify = True
#                        modify.append(tid)
#                else:
#                    if tlen > 1:
#                        grange = rmax - rmin
#                        if d <= roundcutoff + grange:
#                            oldma = groupmovingaverages[tid]
#                            nma = (oldma * tlen + nf) / (tlen + 1)
#                            nmadiff = abs(oldma - nma)
#                            groupmovingaverages[tid] = nma
#                            madiff = groupdifftoma[tid]
#                            groupdifftoma[tid] = (madiff * (tlen - 1) + nmadiff) / tlen
#                            #ndiff = (madiff * (tlen - 1) + nmadiff) / tlen
#                            #modify = True
#                            modify.append(tid)
#                            if rep:
#                                print(3, nf, nmadiff, madiff, grange)
#                    else: #first one's not free, but comes at a discount
#                        #if (d - roundcutoff) - d * roundcutoff <= roundcutoff: #1154745
#                        if d <= roundcutoff * 2: #1154797 and way less lenient
#                            oldma = groupmovingaverages[tid]
#                            nma = (oldma * tlen + nf) / (tlen + 1)
#                            nmadiff = abs(oldma - nma)
#                            groupmovingaverages[tid] = nma
#                            groupdifftoma[tid] = nmadiff
#                            modify.append(tid)
#                            #modify = True
#                            if rep:
#                                print(4, nf, nmadiff, madiff, grange)
#            #below needs a different section
#            if len(modify) > 0:
#                if len(modify) > 1:
#                    #merge ms2lines first
#                    basetid = min(modify)
#                    baselen = len(ms2trackedgroups[basetid])
#                    modify.remove(basetid)
#                    mas = 0
#                    diffs = 0
#                    lengths = 0
#                    for tid in modify:
#                        poppedgroup = ms2trackedgroups.pop(tid)
#                        plen = len(poppedgroup)
#                        ms2trackedgroups[basetid].extend(poppedgroup)
#                        #mas += trackedma.pop(tid)*plen
#                        diffs += groupdifftoma.pop(tid)*plen
#                        lengths += plen
#                        tidmin, tidmax = groupranges[tid]
#                        if tidmin < groupranges[basetid][0]:
#                            groupranges[basetid][0] = tidmin
#                        if tidmax > groupranges[basetid][1]:
#                            groupranges[basetid][1] = tidmax
#                        wtidmin, wtidmax = windowranges[tid]
#                        if wtidmin < windowranges[basetid][0]:
#                            windowranges[basetid][0] = wtidmin
#                        if wtidmax > windowranges[basetid][1]:
#                            windowranges[basetid][1] = wtidmax
#                        sectionsoflines[basetid].update(sectionsoflines.pop(tid))
#                        accumulatedwindowlines[basetid].update(accumulatedwindowlines.pop(tid))
#                        tidprimaries = primariesbyline.pop(tid)
#                        primariesbyline[basetid].extend(tidprimaries)
#                        for tidprimary in tidprimaries:
#                            #linesofprimaries[tidprimary].remove(tid)
#                            linesofprimaries[tidprimary] = basetid
#                        #tidsections = sectionsoflines.pop(tid) #doing this twice?
#                        #sectionsoflines[basetid].update(tidsections)
#                        #these below are for ms1 lines, they don't need to change
#                        #for tidsec in tidsections:
#                        #    oldtidseclines = tuple(sorted(ms1linesofsections[tidsec]))
#                        #    ms1linesofsections[tidsec].remove(tid)
#                        #    ms1linesofsections[tidsec].add(basetid)
#                        #    newtidseclines = tuple(sorted(ms1linesofsections[tidsec]))
#                        #    sectionofms1linegroups[newtidseclines] = sectionofms1linegroups.pop(oldtidseclines)
#                    #just average the 3 new groupmas/diffs, good enough i think
#                    #append the new datapoint to the new mergedline
#                    
#                    sectionsoflines[basetid].add(workingsectionindex)
#                    #merge lines -> update groupma and groupdifftoma
#                    primariesbyline[basetid].append(primary)
#                    linesofprimaries[primary] = basetid
#                    trackedkeys[nf] = basetid
#                    #trackedma[nma] = trackedma.pop(f)
#                    #trackedscancount[tid] += 1
#                    #trackedlength[tid] += 1
#                    #for n, ci in enumerate(c):
#                    #    ms2trackedgroups[tid][n].append(ci)
#                    groupmovingaverages[basetid] = mas / lengths
#                    groupdifftoma[basetid] = diffs / lengths
#                    ms2trackedgroups[basetid].append(c)
#                    #if linedeletioncounter[tid] > 0:
#                    #linedeletioncounter[tid] -= 1 #1661209
#                    #linedeletioncounter[tid] = 0 #1593013
#                    #linedeletioncounter[tid] //= 2 #1606868
#                    #lastmatchtime[uidcount] = rt
#                    linedeletioncounter[basetid] /= 2
#                    #linedeletiontime[tid] /= 2
#                    accumulatedwindowlines[basetid].update(ms1lines)
#                    #^these two options are very different in philosophy. The -1 doesn't allow things to be able to extend themselves very far. The =0 allows for a lifeline every time a new point is added, this one assumes that the end is not going to be hard to find. There's only a difference of 4% between the two trackedgroup lengths listed next to the numbers above^.
#                    if nf < rmin:
#                        groupranges[basetid][0] = nf
#                    if nf > rmax:
#                        groupranges[basetid][1] = nf
#                    #gmin, gmax = groupranges[basetid]
#                    #grange = gmax - gmin
#                    #if grange > widestmassrange:
#                    #    widestmassrange = grange
#                    ms2linelowerscanbound, ms2lineupperscanbound = windowranges[basetid]
#                    if lowerscanbound < ms2linelowerscanbound:
#                        windowranges[basetid][0] = lowerscanbound
#                    if upperscanbound > ms2lineupperscanbound:
#                        windowranges[basetid][1] = upperscanbound
#                else: #1 match, no merges
#                    tid = modify[0]
#                    sectionsoflines[tid].add(workingsectionindex)
#                    primariesbyline[tid].append(primary)
#                    linesofprimaries[primary] = tid
#                    trackedkeys[nf] = tid
#                    ms2trackedgroups[tid].append(c)
#                    linedeletioncounter[tid] /= 2
#                    #linedeletiontime[tid] /= 2
#                    accumulatedwindowlines[tid].update(ms1lines)
#                    #^these two options are very different in philosophy. The -1 doesn't allow things to be able to extend themselves very far. The =0 allows for a lifeline every time a new point is added, this one assumes that the end is not going to be hard to find. There's only a difference of 4% between the two trackedgroup lengths listed next to the numbers above^.
#                    if nf < rmin:
#                        groupranges[tid][0] = nf
#                    if nf > rmax:
#                        groupranges[tid][1] = nf
#                    #gmin, gmax = groupranges[tid]
#                    #grange = gmax - gmin
#                    #if grange > widestmassrange:
#                    #    widestmassrange = grange
#                    ms2linelowerscanbound, ms2lineupperscanbound = windowranges[tid]
#                    if lowerscanbound < ms2linelowerscanbound:
#                        windowranges[tid][0] = lowerscanbound
#                    if upperscanbound > ms2lineupperscanbound:
#                        windowranges[tid][1] = upperscanbound
#            else:
#                #now i need to keep these from forming unless ALL instances of a previousdata point fails? no, this is per mza point, ?
#                #iterate all ^above groups of a single mza point at once, then do this below if none of them pass
#                sectionsoflines[uidcount].add(workingsectionindex)
#                primariesbyline[uidcount].append(primary)
#                linesofprimaries[primary] = line
#                ms2trackedgroups[uidcount] = [c]
#                trackedkeys[nf] = uidcount
#                #trackedma[nf] = uidcount
#                groupmovingaverages[uidcount] = nf
#                groupdifftoma[uidcount] = 0 #this zero won't bog down any averages, same principle new mechanics
#                groupranges[uidcount] = np.array([nf, nf])
#                windowranges[uidcount] = scanbounds.copy() #do i need to copy?
#                #trackedscancount[uidcount] += 1
#                #trackedlength[uidcount] += 1
#                #lastmatchtime[uidcount] = rt
#                accumulatedwindowlines[uidcount].update(ms1lines)
#                uidcount += 1
#                #foundremoval = groupmovingaverages[tid]
#                #foundremovals.append(foundremoval)
#                foundindremovals.append(tid)
#        
#        #for fr in foundremovals:
#        #    found.remove(fr)
#        for fr in foundindremovals:
#            fullfoundinds.remove(fr)
#        #fullfoundinds.extend(foundinds)
#        #nonmatchedmasses = np.setdiff1d(intersectedmasses, found)
#        #nonmatchedinds = np.array(list(map(trackedkeys.get, nonmatchedmasses.tolist())))
#        #I NEED TO FIX THIS
#        #its not indexing right, find a way to get the keys
#        #nonmatchedinds = np.setdiff1d(intersectedkeys, foundinds)
#        #nonmatchedmasses = intersectedmasses[nonmatchedinds]
#        nonmatchedinds = np.setdiff1d(fullintersectedinds, fullfoundinds)
#        nonmatchedmasses = np.array([list(map(groupmovingaverages.get, nonmatchedinds.tolist()))])
#        
#        newmodeladditions = []
#        #things from previousdata not in found gets +1 to linedeletioncounter
#        for n, (nm, linekey) in enumerate(zip(nonmatchedmasses.tolist(), nonmatchedinds.tolist())): #could this loop be concurrent?
#            #linekey = trackedkeys[nm]
#            #linekey = trackedma[nm]
#            linedeletioncounter[linekey] += 1
#            #linedeletiontime[linekey] = rt - lastmatchtime[linekey]
#            #if linedeletioncounter[linekey] > deadsignal:
#            #    #determine, out of all matched and nonmatchedmasses, which fall into a +/- subisomax distance to this movingma, put the lineuids together in a list to be later intersection_merged for line corrections
#            #    #newmodelremovals.append(n)
#            #    ms2trackedgroups[linekey] = np.array(ms2trackedgroups[linekey]) #more efficient memory storage now that it doesn't need to be appended, not much speed compromised but it is a little slower
#            #else:
#            if linedeletioncounter[linekey] <= deadsignal:
#                newmodeladditions.append(linekey)
#        
#        #a second loop here repeating the ^above but for things not intersected -> check if they fit within the ms1 window range and add to line deletion counter if so
#        nonintersectedmasses = previousdata[nonintersectedinds]
#        nonintersectedkeys = previousindices[nonintersectedinds]
#        for n, (nm, linekey) in enumerate(zip(nonintersectedmasses.tolist(), nonintersectedkeys.tolist())): #could this loop be concurrent?
#            ms2linelowerscanbound, ms2lineupperscanbound = windowranges[linekey]
#            if upperscanbound > ms2linelowerscanbound and lowerscanbound < ms2lineupperscanbound:
#                #linekey = trackedkeys[nm]
#                #linekey = trackedma[nm]
#                linedeletioncounter[linekey] += 1
#                #linedeletiontime[linekey] = rt - lastmatchtime[linekey]
#                #if linedeletioncounter[linekey] > deadsignal:
#                #    #determine, out of all matched and nonmatchedmasses, which fall into a +/- subisomax distance to this movingma, put the lineuids together in a list to be later intersection_merged for line corrections
#                #    #newmodelremovals.append(n)
#                #    ms2trackedgroups[linekey] = np.array(ms2trackedgroups[linekey]) #more efficient memory storage now that it doesn't need to be appended, not much speed compromised but it is a little slower
#                #else:
#                if linedeletioncounter[linekey] <= deadsignal:
#                    newmodeladditions.append(linekey)
#        
#        #wides.append(widestmassrange)
#        #nmremovals = np.unique(newmodelremovals)
#        #nonmatchedmasses = np.delete(nonmatchedmasses, nmremovals)
#        #nonmatchedinds = np.delete(nonmatchedinds, nmremovals)
#        newmodeladditions = np.unique(newmodeladditions)
#        newmodeladditionmasses = list(map(groupmovingaverages.get, newmodeladditions.tolist()))
#        currentmasskeys = list(map(trackedkeys.get, mza.flatten().tolist()))
#        currentmasses = np.array(list(map(groupmovingaverages.get, currentmasskeys)))
#        #previousdata = np.concatenate((currentmasses, nonmatchedmasses, nonintersectedmasses))
#        previousdata = np.concatenate((currentmasses, newmodeladditionmasses, nonintersectedmasses))
#        #previousindices = np.concatenate((currentmasskeys, nonmatchedinds, nonintersectedkeys))
#        previousindices = np.concatenate((currentmasskeys, newmodeladditions, nonintersectedkeys))
#        previousindices = previousindices[previousdata.argsort()]
#        previousdata = np.sort(previousdata)
#        scancount += 1
#    
#    maxlen = 0
#    ms2linesofscans = defaultdict(list) #scan: [ms2lines]
#    scansofms2lines = defaultdict(set)
#    for ms2line, tg in ms2trackedgroups.items():
#        scans = set(i[3] for i in tg)
#        ms2trackedgroups[ms2line] = np.array(tg)
#        if len(tg) > maxlen:
#            maxlen = len(tg)
#        for scan in scans:
#            ms2linesofscans[scan].append(ms2line)
#            scansofms2lines[ms2line].add(scan)
#    tglens.append(maxlen)
#    
#    entropyorganizer = defaultdict(lambda: defaultdict(int)) #ms2line: line: count
#    #i need to make a ms2linesofscans and everything should just plug in after that
#    #i'll probably need to introduce minor ppm-based variation in ms2 signals
#    lineindices = {}
#    massindices = {}
#    for n, scan in enumerate(group):
#        lineindices[n] = set(linesofscans[scan])
#        massindices[n] = set(ms2linesofscans[scan])
#    blocked = set()
#    if n > 0:
#        for itercomb in massindices:
#            iterset = lineindices[itercomb]
#            for coitercomb in massindices:
#                if itercomb != coitercomb:
#                    combinationtuple = tuple(sorted([itercomb, coitercomb]))
#                    coiterset = lineindices[coitercomb]
#                    combintersection = iterset.intersection(coiterset)
#                    itercombinds = massindices[itercomb]
#                    coitercombinds = massindices[coitercomb]
#                    if combintersection:
#                        if combinationtuple not in blocked:
#                            blocked.add(combinationtuple)
#                            mainindintersection = itercombinds.intersection(coitercombinds)
#                            if mainindintersection:
#                                combintersection = tuple(combintersection)
#                                for ind in mainindintersection:
#                                    for c in combintersection:
#                                        entropyorganizer[ind][c] += 1
#                        iterdiff = iterset.difference(coiterset)
#                        if iterdiff:
#                            itermaininds = itercombinds.difference(coitercombinds)
#                            label = tuple(iterset.difference(coiterset)) #automatically sorts
#                        else:
#                            #everything from the iter is within the coiter, mark everything as being from the itercomb key
#                            itermaininds = itercombinds
#                            label = tuple(iterset)
#                        for ind in itermaininds:
#                            for l in label:
#                                entropyorganizer[ind][l] += 1
#    else: #no competition
#        for n, inds in massindices.items():
#            for line in lineindices[n]:
#                for mi in inds:
#                    entropyorganizer[mi][line] += 1
#    entropylength += len(entropyorganizer)
#    #fullentropyorganizer.update(entropyorganizer)
#    
#    percentbyline = {} #ms2line: % good primaries
#    for ms2line, assignablems1lines in entropyorganizer.items():
#        tg = ms2trackedgroups[ms2line]
#        for m, i, primary, scan, section, baseprimary in tg.tolist():
#            assessablems1lines = ms1linesofsections[section]
#            #assessabledict = Counter(dict(zip(assessablems1lines, map(assignablems1lines.get, assessablems1lines))))
#            assessabledict = Counter()
#            for a in assessablems1lines:
#                if a in assignablems1lines:
#                    assessabledict[a] = assignablems1lines[a]
#            assessablerankings = assessabledict.most_common(len(assessabledict))
#            if len(assessablerankings) > 1:
#                if assessablerankings[0][1] == assessablerankings[1][1]:
#                    #find all top ranks
#                    toprank = assessablerankings[0][1]
#                    toplines = []
#                    for ms1line, rank in assessablerankings:
#                        if rank == toprank:
#                            toplines.append(ms1line)
#                        else:
#                            break
#                    toplines = tuple(toplines)
#                    primaryassignmentresults[primary] = toplines
#                else:
#                    #lone top rank
#                    #assign primary to ms1 line
#                    primaryassignmentresults[primary] = assessablerankings[0][0]
#            else:
#                #lone top rank
#                #assign primary to ms1 line
#                primaryassignmentresults[primary] = assessablerankings[0][0]
#
#    ms1entropy = defaultdict(lambda: Counter()) #ms2line: line: [union, ms1diff, ms2diff, symdiff]
#    ms2entropy = defaultdict(lambda: Counter()) #ms2line: line: [union, ms1diff, ms2diff, symdiff]
#    symdiffentropy = defaultdict(lambda: Counter()) #ms2line: line: [union, ms1diff, ms2diff, symdiff]
#    unionentropy = defaultdict(lambda: Counter()) #ms2line: line: [union, ms1diff, ms2diff, symdiff]
#    
#    #compare scansbymainindices to linesofscans
#    for ms2line, scans in scansofms2lines.items():
#        #total ms1 line scans vs total ms2 line scans -> union to get overlap
#        #differences + symmetric difference
#        for scan in scans:
#            ms1lines = linesofscans[scan]
#            for line in ms1lines:
#                ms1scans = set(scansoflines[line])
#                ms1diffs = len(ms1scans.difference(scans))
#                ms2diffs = len(scans.difference(ms1scans))
#                scanunion = len(ms1scans.union(scans))
#                ms1entropy[ms2line][line] -= ms1diffs
#                ms2entropy[ms2line][line] -= ms2diffs
#                symdiffentropy[ms2line][line] -= ms1diffs + ms1diffs
#                unionentropy[ms2line][line] += scanunion
#    
#    for ms2line, assignablems1lines in ms1entropy.items():
#        primaries = primariesbyline[ms2line]
#        #for m, i, primary, scan, section, baseprimary in tg.tolist():
#        for primary in primaries:
#            #get scan: scansbyprimaryind -> get lines
#            scan = scansbyprimaryind[primary]
#            assessablems1lines = linesofscans[scan]
#            #assessabledict = Counter(dict(zip(assessablems1lines, map(assignablems1lines.get, assessablems1lines))))
#            assessabledict = Counter()
#            for a in assessablems1lines:
#                if a in assignablems1lines:
#                    assessabledict[a] = assignablems1lines[a]
#            assessablerankings = assessabledict.most_common(len(assessabledict))
#            if len(assessablerankings) > 1:
#                if assessablerankings[0][1] == assessablerankings[1][1]:
#                    #find all top ranks
#                    toprank = assessablerankings[0][1]
#                    toplines = []
#                    for ms1line, rank in assessablerankings:
#                        if rank == toprank:
#                            toplines.append(ms1line)
#                        else:
#                            break
#                    toplines = tuple(toplines)
#                    ms1assignmentresults[primary] = toplines
#                else:
#                    #lone top rank
#                    #assign primary to ms1 line
#                    ms1assignmentresults[primary] = assessablerankings[0][0]
#            else:
#                #lone top rank
#                #assign primary to ms1 line
#                ms1assignmentresults[primary] = assessablerankings[0][0]
#    
#    for ms2line, assignablems1lines in ms2entropy.items():
#        primaries = primariesbyline[ms2line]
#        #for m, i, primary, scan, section, baseprimary in tg.tolist():
#        for primary in primaries:
#            #get scan: scansbyprimaryind -> get lines
#            scan = scansbyprimaryind[primary]
#            assessablems1lines = linesofscans[scan]
#            #assessabledict = Counter(dict(zip(assessablems1lines, map(assignablems1lines.get, assessablems1lines))))
#            assessabledict = Counter()
#            for a in assessablems1lines:
#                if a in assignablems1lines:
#                    assessabledict[a] = assignablems1lines[a]
#            assessablerankings = assessabledict.most_common(len(assessabledict))
#            if len(assessablerankings) > 1:
#                if assessablerankings[0][1] == assessablerankings[1][1]:
#                    #find all top ranks
#                    toprank = assessablerankings[0][1]
#                    toplines = []
#                    for ms1line, rank in assessablerankings:
#                        if rank == toprank:
#                            toplines.append(ms1line)
#                        else:
#                            break
#                    toplines = tuple(toplines)
#                    ms2assignmentresults[primary] = toplines
#                else:
#                    #lone top rank
#                    #assign primary to ms1 line
#                    ms2assignmentresults[primary] = assessablerankings[0][0]
#            else:
#                #lone top rank
#                #assign primary to ms1 line
#                ms2assignmentresults[primary] = assessablerankings[0][0]
#    
#    for ms2line, assignablems1lines in symdiffentropy.items():
#        primaries = primariesbyline[ms2line]
#        #for m, i, primary, scan, section, baseprimary in tg.tolist():
#        for primary in primaries:
#            #get scan: scansbyprimaryind -> get lines
#            scan = scansbyprimaryind[primary]
#            assessablems1lines = linesofscans[scan]
#            #assessabledict = Counter(dict(zip(assessablems1lines, map(assignablems1lines.get, assessablems1lines))))
#            assessabledict = Counter()
#            for a in assessablems1lines:
#                if a in assignablems1lines:
#                    assessabledict[a] = assignablems1lines[a]
#            assessablerankings = assessabledict.most_common(len(assessabledict))
#            if len(assessablerankings) > 1:
#                if assessablerankings[0][1] == assessablerankings[1][1]:
#                    #find all top ranks
#                    toprank = assessablerankings[0][1]
#                    toplines = []
#                    for ms1line, rank in assessablerankings:
#                        if rank == toprank:
#                            toplines.append(ms1line)
#                        else:
#                            break
#                    toplines = tuple(toplines)
#                    symdiffassignmentresults[primary] = toplines
#                else:
#                    #lone top rank
#                    #assign primary to ms1 line
#                    symdiffassignmentresults[primary] = assessablerankings[0][0]
#            else:
#                #lone top rank
#                #assign primary to ms1 line
#                symdiffassignmentresults[primary] = assessablerankings[0][0]
#    
#    for ms2line, assignablems1lines in unionentropy.items():
#        primaries = primariesbyline[ms2line]
#        #for m, i, primary, scan, section, baseprimary in tg.tolist():
#        for primary in primaries:
#            #get scan: scansbyprimaryind -> get lines
#            scan = scansbyprimaryind[primary]
#            assessablems1lines = linesofscans[scan]
#            #assessabledict = Counter(dict(zip(assessablems1lines, map(assignablems1lines.get, assessablems1lines))))
#            assessabledict = Counter()
#            for a in assessablems1lines:
#                if a in assignablems1lines:
#                    assessabledict[a] = assignablems1lines[a]
#            assessablerankings = assessabledict.most_common(len(assessabledict))
#            if len(assessablerankings) > 1:
#                if assessablerankings[0][1] == assessablerankings[1][1]:
#                    #find all top ranks
#                    toprank = assessablerankings[0][1]
#                    toplines = []
#                    for ms1line, rank in assessablerankings:
#                        if rank == toprank:
#                            toplines.append(ms1line)
#                        else:
#                            break
#                    toplines = tuple(toplines)
#                    unionassignmentresults[primary] = toplines
#                else:
#                    #lone top rank
#                    #assign primary to ms1 line
#                    unionassignmentresults[primary] = assessablerankings[0][0]
#            else:
#                #lone top rank
#                #assign primary to ms1 line
#                unionassignmentresults[primary] = assessablerankings[0][0]
#    
#print(time() - nt)
#
##make ms2trackedgroups numpy arrays
##entropyorganizer counts everything
##assess ms2lines by section -> iterate entropyorganizer
##from ms2lines get the primaries and sections
##line: section: primaries
#
##assess how many primaries are correct/incorrect
#
#incorrectcount = Counter()
#notincorrectcount = Counter()
#
#correct = 0
#incorrect = 0
#notincorrect = 0
#for primary, results in primaryassignmentresults.items():
#    trueresult = linesbyprimaryind[primary]
#    match results:
#        case tuple():
#            #outcome = set(results).intersection(trueresult)
#            outcome = trueresult in results
#            if outcome:
#                notincorrect += 1
#                notincorrectcount[len(results)] += 1
#            else:
#                incorrect += 1
#                incorrectcount[len(results)] += 1
#        case int():
#            if results == trueresult:
#                correct += 1
#            else:
#                incorrect += 1
#                incorrectcount[1] += 1
#print('correct:', correct)
#print('incorrect:', incorrect)
#print('not incorrect:', notincorrect)
#
##there's MORE results, why??? and it still doesnt equal the length of whats in linesbyprimaryind...
#print(correct + incorrect + notincorrect, '/', len(linesbyprimaryind))
#print(entropylength, 'vs.', oldentropylength)
#
##the below has one small thing wrong with it, its only getting like 97% of all primaries atm but idc to figure it out because EVERYTHING is being outperformed by the simple intersection_merging within a ppm radius above
##which is kind of infuriating but hey i guess i'll use it
##now i can just explore the simpler process done further above without making any ms2lines because they don't actually provide any benefit
#
#incorrectcount = Counter()
#notincorrectcount = Counter()
#
#correct = 0
#incorrect = 0
#notincorrect = 0
#for primary, results in ms1assignmentresults.items():
#    trueresult = linesbyprimaryind[primary]
#    match results:
#        case tuple():
#            #outcome = set(results).intersection(trueresult)
#            outcome = trueresult in results
#            if outcome:
#                notincorrect += 1
#                notincorrectcount[len(results)] += 1
#            else:
#                incorrect += 1
#                incorrectcount[len(results)] += 1
#        case int():
#            if results == trueresult:
#                correct += 1
#            else:
#                incorrect += 1
#                incorrectcount[1] += 1
#
#print('MS1 Entropy')
#print('correct:', correct)
#print('incorrect:', incorrect)
#print('not incorrect:', notincorrect)
#print(correct + incorrect + notincorrect)
#print('~')
#
#incorrectcount = Counter()
#notincorrectcount = Counter()
#
#correct = 0
#incorrect = 0
#notincorrect = 0
#for primary, results in ms2assignmentresults.items():
#    trueresult = linesbyprimaryind[primary]
#    match results:
#        case tuple():
#            #outcome = set(results).intersection(trueresult)
#            outcome = trueresult in results
#            if outcome:
#                notincorrect += 1
#                notincorrectcount[len(results)] += 1
#            else:
#                incorrect += 1
#                incorrectcount[len(results)] += 1
#        case int():
#            if results == trueresult:
#                correct += 1
#            else:
#                incorrect += 1
#                incorrectcount[1] += 1
#
#print('MS2 Entropy:')
#print('correct:', correct)
#print('incorrect:', incorrect)
#print('not incorrect:', notincorrect)
#print(correct + incorrect + notincorrect)
#print('~')
#
#incorrectcount = Counter()
#notincorrectcount = Counter()
#
#correct = 0
#incorrect = 0
#notincorrect = 0
#for primary, results in symdiffassignmentresults.items():
#    trueresult = linesbyprimaryind[primary]
#    match results:
#        case tuple():
#            #outcome = set(results).intersection(trueresult)
#            outcome = trueresult in results
#            if outcome:
#                notincorrect += 1
#                notincorrectcount[len(results)] += 1
#            else:
#                incorrect += 1
#                incorrectcount[len(results)] += 1
#        case int():
#            if results == trueresult:
#                correct += 1
#            else:
#                incorrect += 1
#                incorrectcount[1] += 1
#
#print('Symdiff')
#print('correct:', correct)
#print('incorrect:', incorrect)
#print('not incorrect:', notincorrect)
#print(correct + incorrect + notincorrect)
#print('~')
#
#incorrectcount = Counter()
#notincorrectcount = Counter()
#
#correct = 0
#incorrect = 0
#notincorrect = 0
#for primary, results in unionassignmentresults.items():
#    trueresult = linesbyprimaryind[primary]
#    match results:
#        case tuple():
#            #outcome = set(results).intersection(trueresult)
#            outcome = trueresult in results
#            if outcome:
#                notincorrect += 1
#                notincorrectcount[len(results)] += 1
#            else:
#                incorrect += 1
#                incorrectcount[len(results)] += 1
#        case int():
#            if results == trueresult:
#                correct += 1
#            else:
#                incorrect += 1
#                incorrectcount[1] += 1
#
#print('Union')
#print('correct:', correct)
#print('incorrect:', incorrect)
#print('not incorrect:', notincorrect)
#print(correct + incorrect + notincorrect)
#print('~')

#plot
#x-axis is length of [not] [in]correct tuple outcome
#y-axis is number of occurrences

#plt.bar(incorrectcount.keys(), incorrectcount.values())
#plt.show()
#
#plt.bar(notincorrectcount.keys(), notincorrectcount.values())
#plt.show()
#
#lencounts = Counter(len(i) for i in linesofscans.values())

#visualize linesofscans by the number of lines vs scan

#does the entropy organizer perform better/worse than just counting which signals overlap with each scan that an ms1line is in?
# - count the lines of times an ms1line coindices with an ms2line
# - +1 for each match
# - -1 for each non-match
#   > and the non-match can be either an ms1line that outnumbers the m2line, or vice versa
# - if the 
#for every scan of an ms1 line, is the ms2 line there?
#for every scan of th ms2 line, is the ms1 line there?
# - n points of ms2 and no ms1
# - n points of ms1 and no ms2
#   > are these ^two combinable into one also?
# - n points of ms1 and ms2
#^combine these into individual ranking systems
#MULTIRANK?!
#in the multirank i also want to try assessing the rate of ascent into the ranking
# - the rate of ascent can be (highest rank) / (final rank) or its inverse, it doesn't matter
#then do a SECOND multirank with the final rank AND the average rate of ascent to get some kind of normalization?
#if this leaves me with 2 different entropyorganizers then i can compare those too

#i also want to assess
#on the assignments that are outright wrong
#how many ms1 lines are in those scans? does it tend to be a lot?

#you'll need to explore the multiple counters on the level of whats assessable i guess?

#the original setup gets the most correct
#while the ms1diffs gets the least wrong
#where can i find a compromise between these to figure out when to accept something from the original, and when to reject it?
