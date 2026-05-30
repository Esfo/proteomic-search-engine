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


#this is basically a simulation that doesn't deal with RTs, and the overlap can be more "flexible" because of this. this also makes it unrealistic when compared to the things the real simulation is meant to represent but that fine here because there isn't a need for this to be perfect. only the ms2 style fragment ions need to be weeded out through comparison

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

#indicesandprobabilities = {0: {0, 1, 2, 3},
#               1: {4, 5},
#               2: {6, 7, 8},
#               3: {9, 10, 11, 12, 13},
#               4: {14, 15, 16, 17}} #line: {correct mass indices}

debugging = False
debuggingiter = False

nlines = 20
matchingradius = 25 #ppm
maxscansamples = 4

minimumabundance = 0.01
dividingthreshold = 0.1
subisotopomericdepth = 0.7
ions = 'by'

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

def distribution_generation(atomiccomposition):
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
    return formulas, massesandabundances

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

nt = time()
finalcounts = []
#j = 0
#while j < 1000:
#j += 1

#lines = [0, 1, 2, 3, 4]
lines = np.arange(nlines)

#scansfromlines = [[0, 1],
#         [1],
#         [2, 3],
#         [2, 3, 4],
#         [3, 4]] #line (as the index): [scans]
scansfromlines = [] #line as index, [scans]
for n in range(nlines):
    nscans = np.random.randint(1, maxscansamples)
    scans = np.random.choice(lines, size=nscans).tolist()
    scansfromlines.append(scans)
    if debugging:
        print(f'line: {n}, scans: {scans}')
if debugging:
    print('~')

#linesofscans = {0: tuple([0]),
#                  1: (0, 1),
#                  2: (2, 3),
#                  3: (2, 3, 4),
#                  4: (4, 3)} #scan: [lines]
scansoflines = defaultdict(set) #line: [scans]
linesofscans = defaultdict(list) #scan: [lines]
peptidesoflines = {} #line: peptide
for line, scans in enumerate(scansfromlines):
    peptidesoflines[line] = ''.join((random.choices(list(aminoacidcomposition.keys()), k=np.random.randint(6,50))))
    for scan in scans:
        linesofscans[scan].append(line)
        scansoflines[line].add(scan)
for k, v in linesofscans.items():
    linesofscans[k] = tuple(set(v))
    if debugging:
        print(f'scan: {k}, lines: {v}')
if debugging:
    print('~')

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

ionlist = list(ions)
ndict = {k: nfragmentcompositions[k] for k in ionlist if k in nfragmentcompositions}
cdict = {k: cfragmentcompositions[k] for k in ionlist if k in cfragmentcompositions}

peptidefrags = {} #peptide: {fragdict}
peptidefragprobs = defaultdict(dict) #peptide: ion: prob
ms1dists = {} #peptide: [masses, abundances]
peptidesubformulas = {} #peptide: subformula
#set up the probabilities that can be sampled below
for peptide in set(peptidesoflines.values()):
    atomiccomposition = Counter()
    for aa in peptide:
        atomiccomposition += aminoacidcomposition[aa]
    #no OH loss on last residue, no H lost on first residue
    atomiccomposition['H'] += 2
    atomiccomposition['O'] += 1
    subformulas, massesandabundances = distribution_generation(atomiccomposition)
    massgroups = defaultdict(list) #massnumber: [masses]
    intensitygroups = defaultdict(list) #massnumber: [abundances]
    masses, intensities = massesandabundances
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
    ms1dists[peptide] = sumabundancedist
    maxloc = np.array(sumsofabundances).argmax()
    subquals = condensationindices[maxloc]
    conlengths = np.array(condensationindices)
    conends = conlengths.cumsum()
    constarts = conends - conlengths
    lineposition = subisodepthindices[subquals][0] + constarts[subquals]
    subformula = subformulas[lineposition]
    peptidesubformulas[peptide] = subformula.decode()
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

#do this simulation but with peptides + frag masses

currentind = 0
massesoflines = {} #line: [masses]
indicesandprobabilities = {} #line: [mass inds]
linesbyprimaryind = {} #primary ind: line
for n in range(nlines):
    #num = np.random.randint(minlone, maxlone)
    #masses = np.random.uniform(low=10, high=20, size=num).tolist()
    baseintensity = np.random.uniform(1e2, 1e5)
    fragrolls = np.random.multinomial(baseintensity, list(peptidefragprobs[peptide].values()), size=1)[0].tolist()
    fragmin = min(fragrolls)
    peptide = peptidesoflines[n]
    subformula = peptidesubformulas[peptide]
    masses = []
    #ms2lists[rtind].append(ms2id)
    #masses = []
    #intensities = []
    #fragions = []
    #for sq in subisodepthqualifiers[formula][linesubisoposition]:
    #subindex = bi + sq
    #subformula = subformulas[subindex]
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
        if roll > 0:
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
                        iso = list(fragprobs)[0]
                        elementalorganizer[e] = [[-1, 1, elementalmasses[iso]*c, e, [c]]]
                        fragmentpositions[e] = {0: iso}
            fragformulas, massesandabundances = fragment_descending_partial_products(dividingthreshold, elementalorganizer, fragmentpositions)
            massesandabundances[:,0] += proton #making all ions are single-charge for now
            keepers = massesandabundances[:,1] * roll >= fragmin #simple intensity filter
            masses.extend(massesandabundances[keepers,0].tolist())
            #intensities.extend((massesandabundances[keepers,1] * roll).tolist())
            #fragions.extend(itertools.repeat(ion, keepers.sum()))
    #npoints = len(intensities)
    #primaryinds = np.arange(npoints) + primaryind
    #primaryinds = primaryinds.tolist()
    #ms2scans[ms2id] = [masses, intensities, primaryinds]
    #ms2massidentities[ms2id][line] = masses
    #ms2massesoflines[line][ms2id] = masses
    #for p in primaryinds:
    #    linesbyprimaryind[p] = line
    #    scansbyprimaryind[p] = ms2id
    #primaryind += npoints
    massesoflines[n] = masses
    num = len(masses)
    inds = np.arange(num) + currentind
    #nmainions = np.random.randint(minmainfrags, maxmainfrags)
    #probs = np.random.uniform(size=num)
    #while True:
    #    probs *= probs
    #    test = probs / probs.sum()
    #    if np.sort(test)[-nmainions:].sum() >= threshold:
    #        break
    #probs = probs / probs.sum() #basic stuff for now
    indicesandprobabilities[n] = inds.tolist()
    for i in inds.tolist():
        linesbyprimaryind[i] = n
    currentind += num
    if debugging:
        print(f'line: {n}, inds: {inds}')
if debugging:
    print('~')

mergedscans = intersection_merge(scansfromlines)
mergedscans = list(map(tuple, mergedscans))

mergedgroupbyline = {} #line: merged index
mergedscansbyindex = {} #merged ind: [scangroup]
for n, group in enumerate(mergedscans):
    mergedscansbyindex[n] = group
    for scan in group:
        for line in linesofscans[scan]:
            mergedgroupbyline[line] = n


#linesbymainind = defaultdict(dict) #group: index: line
#primarymainindices = defaultdict(dict) #group: mass: index within massesofscangroups
#massesofscangroups = defaultdict(list) #scangroup: [masses]
#for line, masses in massesoflines.items():
#    group = mergedgroupbyline[line]
#    grouplen = len(massesofscangroups[group])
#    massesofscangroups[group].extend(masses)
#    for m in masses:
#        primarymainindices[group][m] = grouplen
#        linesbymainind[group][grouplen] = line
#        grouplen += 1

#2 indices
# - primary - index of the mass in the scan
# - main - KNN reduced index

#start:
#generate masses, primary inds, and their probs
#next loop:
#reindex masses of scangroups 
#KNN reduction

mainind = 0
linesbymainindex = defaultdict(set) #main index: [lines]
maintoprimaryindex = defaultdict(list) #main index: [primary indices]
primarytomainindex = {} #primary index: main index
scangroupbyline = defaultdict(list)
#for group, masses in massesofscangroups.items():
for n, group in enumerate(mergedscans):
    linelist = list(itertools.chain(*set(linesofscans[i] for i in group)))
    masses = list(itertools.chain(*[massesoflines[i] for i in linelist]))
    inds = list(itertools.chain(*[indicesandprobabilities[i] for i in linelist]))
    masses = np.array(masses)[:,None]
    radius = (masses * matchingradius).flatten() / 1000000
    nn = spatial.KDTree(masses)
    matches = nn.query_ball_point(masses, radius).tolist()
    groupableinds = list(map(tuple, intersection_merge(matches)))
    for gi in groupableinds:
        for g in gi:
            primaryind = inds[g]
            primarytomainindex[primaryind] = mainind
            maintoprimaryindex[mainind].append(primaryind)
            line = linesbyprimaryind[primaryind]
            linesbymainindex[mainind].add(line)
            scangroupbyline[line] = n
            #
            #intial = primarymainindices[group][masses[g][0]]
            #line = linesbymainind[group][primary]
            #linesbymainindex[n].append(line)
            #indicesandprobabilities[line].append(n)
        mainind += 1

#mainindicesbyscan = {0: {0, 1, 2, 3},
#            1: {0, 1, 4, 5},
#            2: {6, 7, 8, 9, 10, 12, 13},
#            3: {7, 8},
#            4: {9, 10, 11, 13, 14, 15, 16, 17}} #scan: mass indices

mainindicesbyscan = defaultdict(set) #scan: {mass indices}
truelineindex = defaultdict(set) #mainid: [primary lines involved]
scansbymainindices = defaultdict(set)
for scan, slines in linesofscans.items():
    for line in slines:
        sampleinds = indicesandprobabilities[line]
        #num = len(sampleinds)
        #nsample = np.random.randint(minlone-1, num)
        #while True:
        #    try:
        #        sampleinds = np.random.choice(inds, p=probs, size=nsample, replace=False)
        #        break
        #    except ValueError: #fewer non-zero entries in p than size
        #        num -= 1
        #        nsample = np.random.randint(minlone-1, num)
        maininds = [primarytomainindex[i] for i in sampleinds]
        mainindicesbyscan[scan].update(maininds)
        for m in maininds:
            truelineindex[m].add(line)
            scansbymainindices[m].add(scan)

#i need to visualize the primary -> mainind connections in here and the original
#first look at individual maininds
#then look at scangroups

lens = [len(prims) for mains, prims in truelineindex.items()]


if debugging:
    for k, v in mainindicesbyscan.items():
        print(f'scan: {k}, inds: {v}')
    print('~')

#this is going to simplify the problem by using integers instead of floats, the match radius can be factored in once this concept is understood better

#so with scans 0 and 1 involving lines 0 and 1:
#where they have the same indices is from line 0
#where they differ in scan 0 is from line 0
#where they differ in scan 1 is from line 1

#maybe introduce an order of confidence?
#ie, however many scans you see something in, and if you can attribute it to a line, or to some linked group, then that's its order of confidence
#something that is present in all 10 scans of some analyte gets a degree of confidence 10

#iterate scansfromlines
#from the scan indices get the lines involved
#do combinatorics with all scans, output pipes into the set differences of line groups in linesofscans with all scans being considered
#then increase the combinatoric factor until it == the # of scans within the scangroup
#output logic is labeled, in a dictionary where the set difference of scans being compared is the key
#and the count is kept for the matching of each mass index, and the count is separated by lines
#^when multiple lines get an equal count for an index, it implies indifferentiable logic, so its assigned to all lines/scans/whatever within that group aka ambiguous amongst them
#^(in the future perhaps intensity can be taken into account here?)
#^NO! not here at least
#intensity will be taken into account during the ranking process when fragments are matched to a frame-controlled set of ions based on ms1 intensity %
#there are great logical arguments for why a solo-sampled line with a high ms1 intensity had a higher likelihood of not having some massive new collection of fragments in a shared scan where its intensity is much less, but i shouldn't need to assert it here
#the logic should fall into place nicely later imo
#^although this might be a nice concept to visualize for a human

#the purpose of this isn't to be "right" about everything, its about not being wrong about ANYTHING
#if anything is uncertain, it needs to be marked as uncertain
#uncertainty is the way of avoiding being wrong
#anything that can be marked correctly as a biproduct, is a pretty nice deal

#if debugging:
#print(f'mergedscans: {mergedscans}')
#print(f'mergedscan length: {len(mergedscans)}')

incorrectindices = []
print(len(mergedscans), 'mergedscans')
print(sum(1 for i in mergedscans if len(i) > 1), '> length 1')

#nt = time()
#
##if there are redundant line-tuples in maininds am i missing data to iterate? -> YES
##i can use the union-labels as LOW-RISK BOUNDS -> these won't be non-incorrect
##^doesn't look promising but i should do a more in-depth comparison
#
#entropyorganizer = defaultdict(lambda: defaultdict(int)) #mainind: line: count
#for scangroup in mergedscans:
#    maininds = {linesofscans[i]: set(mainindicesbyscan[i]) for i in scangroup} #(lines): [mass indices]
#    if len(maininds) > 1:
#        #iterate a combination of size n, and co-iterate combinations of the (total length - n) to compare each group
#        maxlen = len(maininds)
#        iterlength = maxlen - 1
#        coiterlength = maxlen - iterlength
#        #i tried a lot of different combinatoric avenues for analyzing different angles of the same information, this turned out to be the best way. Just a single-pass of each individual mainind vs every other one as a group
#        for itercomb in itertools.combinations(maininds, iterlength):
#            for coitercomb in itertools.combinations(maininds, coiterlength):
#                if itercomb != coitercomb:
#                    iterset = set()
#                    for i in itercomb:
#                        match i:
#                            case int():
#                                iterset.add(i)
#                            case tuple():
#                                iterset.update(i)
#                    coiterset = set()
#                    for c in coitercomb:
#                        match c:
#                            case int():
#                                coiterset.add(c)
#                            case tuple():
#                                coiterset.update(c)
#                    combintersection = iterset.intersection(coiterset)
#                    itercombinds = set(itertools.chain(*[maininds[i] for i in itercomb]))
#                    coitercombinds = set(itertools.chain(*[maininds[c] for c in coitercomb]))
#                    if combintersection:
#                        mainindintersection = itercombinds.intersection(coitercombinds)
#                        if mainindintersection:
#                            combintersection = tuple(combintersection)
#                            #if len(combintersection) == 1:
#                            #    combintersection = combintersection[0]
#                            #match combintersection:
#                            #    case int():
#                            #        for ind in mainindintersection:
#                            #            entropyorganizer[ind][combintersection] += 1
#                            #    case tuple():
#                            #        for ind in mainindintersection:
#                            #            for c in combintersection:
#                            #                entropyorganizer[ind][c] += 1
#                            for ind in mainindintersection:
#                                for c in combintersection:
#                                    entropyorganizer[ind][c] += 1
#                        iterdiff = iterset.difference(coiterset)
#                        #^should i be checking if the union != the difference?
#                        if iterdiff:
#                            #difference exists, take the diff of the maininds
#                            itermaininds = itercombinds.difference(coitercombinds)
#                            #label = tuple(iterset.union(coiterset)) #automatically sorts
#                            label = tuple(iterset.difference(coiterset)) #automatically sorts
#                        else:
#                            #everything from the iter is within the coiter, mark everything as being from the itercomb key
#                            itermaininds = itercombinds
#                            #label = itercomb
#                            label = tuple(iterset)
#                        #match label:
#                        #    case int():
#                        #        for ind in itermaininds:
#                        #            entropyorganizer[ind][label] += 1
#                        #    case tuple():
#                        #        for ind in itermaininds:
#                        #            for l in label:
#                        #                entropyorganizer[ind][l] += 1
#                        for ind in itermaininds:
#                            for l in label:
#                                entropyorganizer[ind][l] += 1
#                        coiterdiff = coiterset.difference(iterset)
#                        if coiterdiff:
#                            #difference exists, take the diff of the maininds
#                            coitermaininds = coitercombinds.difference(itercombinds)
#                            #label = tuple(coiterset.union(iterset))
#                            label = tuple(coiterset.difference(iterset))
#                        else:
#                            #everything from the coiter is within the iter, mark everything as being from the coitercomb key
#                            coitermaininds = coitercombinds
#                            #label = coitercomb
#                            label = tuple(coiterset)
#                        #match label:
#                        #    case int():
#                        #        for ind in coitermaininds:
#                        #            entropyorganizer[ind][label] += 1
#                        #    case tuple():
#                        #        for ind in coitermaininds:
#                        #            for l in label:
#                        #                entropyorganizer[ind][l] += 1
#                        for ind in coitermaininds:
#                            for l in label:
#                                entropyorganizer[ind][l] += 1
#    else: #no competition
#        for lines, inds in maininds.items():
#            for mi in inds:
#                for line in lines:
#                    entropyorganizer[mi][line] += 1
#
#print(len(entropyorganizer), sum(len(v) for v in entropyorganizer.values()))
#print(time() - nt, 'single-pass reciprocating combinatoric (no non-overlap additions) finished')
#labelfailure = False
#for k, v in entropyorganizer.items():
#    for sv in v:
#        if type(sv) is tuple:
#            labelfailure = True
#print(labelfailure, 'labelfailure')
#
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
#incorrections = []
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
#        incorrections.append(mainind)
#    else: #good outcome
#        if trueinds == result:
#            correctoutcomes += 1
#        else:
#            notincorrectoutcomes += 1
#            distance = len(result.difference(trueinds))
#            notincorrectdistances[distance] += 1
#
#incorrectindices.append(incorrections)
#
##finalcounts.append([nwrongs, correctoutcomes, notincorrectoutcomes])
#
##plt.bar(*list(zip(*notincorrectdistances.items())))
##plt.title('not incorrect')
##plt.show()
##
##plt.bar(*list(zip(*badoutcomesbadmatches.items())))
##plt.title('bad outcomes bad matches')
##plt.show()
##
##plt.bar(*list(zip(*badoutcomesgoodmatches.items())))
##plt.title('bad outcomes good matches')
##plt.show()
#
##plt.hist(list(zip(*finalcounts)), bins=100, label=['wrong', 'correct', 'not incorrect'])
##plt.legend()
##plt.title(f'nlines: {nlines}, minlone: {minlone}, maxlone: {maxlone}, radius: {matchingradius}, mainfrags: {minmainfrags}-{maxmainfrags}, threshold: {threshold}', fontsize=10)
##plt.show()
#
#print(f'total wrong: {nwrongs}')
#print(f'total correct: {correctoutcomes}')
#print(f'total not incorrect {notincorrectoutcomes}')
#print('-------------------------------------------')
#
#nt = time()
#
##if there are redundant line-tuples in maininds am i missing data to iterate? -> YES
##i can use the union-labels as LOW-RISK BOUNDS -> these won't be non-incorrect
##^doesn't look promising but i should do a more in-depth comparison
##if the length of maininds is 2, then will this still work via iterlength and coiterlength? i think this screws up and skips those? no it should be fine
#
#entropyorganizer = defaultdict(lambda: defaultdict(int)) #mainind: line: count
#for scangroup in mergedscans:
#    #maininds = {linesofscans[i]: set(mainindicesbyscan[i]) for i in scangroup} #(lines): [mass indices]
#    lineindices = {}
#    massindices = {}
#    for n, scan in enumerate(scangroup):
#        lineindices[n] = linesofscans[scan]
#        massindices[n] = set(mainindicesbyscan[scan])
#    #if len(maininds) > 1:
#    if n > 0:
#        #iterate a combination of size n, and co-iterate combinations of the (total length - n) to compare each group
#        maxlen = n + 1
#        iterlength = maxlen - 1
#        coiterlength = maxlen - iterlength
#        #i tried a lot of different combinatoric avenues for analyzing different angles of the same information, this turned out to be the best way. Just a single-pass of each individual mainind vs every other one as a group
#        #for itercomb in itertools.combinations(maininds, iterlength):
#        for itercomb in itertools.combinations(massindices, iterlength):
#            iterset = set()
#            for i in itercomb:
#                #lines = lineindices[i]
#                #match lines:
#                #    case int():
#                #        iterset.add(lines)
#                #    case tuple():
#                #        iterset.update(lines)
#                iterset.update(lineindices[i])
#            #for coitercomb in itertools.combinations(maininds, coiterlength):
#            for coitercomb in itertools.combinations(massindices, coiterlength):
#                if itercomb != coitercomb: #for dual length 1 comparisons
#                    coiterset = set()
#                    for c in coitercomb:
#                        #lines = lineindices[c]
#                        #match lines:
#                        #    case int():
#                        #        coiterset.add(lines)
#                        #    case tuple():
#                        #        coiterset.update(lines)
#                        coiterset.update(lineindices[c])
#                    combintersection = iterset.intersection(coiterset)
#                    if combintersection:
#                        #itercombinds = set(itertools.chain(*[maininds[i] for i in itercomb]))
#                        itercombinds = set(itertools.chain(*[massindices[i] for i in itercomb]))
#                        #coitercombinds = set(itertools.chain(*[maininds[c] for c in coitercomb]))
#                        coitercombinds = set(itertools.chain(*[massindices[c] for c in coitercomb]))
#                        mainindintersection = itercombinds.intersection(coitercombinds)
#                        if mainindintersection:
#                            combintersection = tuple(combintersection)
#                            #if len(combintersection) == 1:
#                            #    combintersection = combintersection[0]
#                            #match combintersection:
#                            #    case int():
#                            #        for ind in mainindintersection:
#                            #            entropyorganizer[ind][combintersection] += 1
#                            #    case tuple():
#                            #        for ind in mainindintersection:
#                            #            for c in combintersection:
#                            #                entropyorganizer[ind][c] += 1
#                            for ind in mainindintersection:
#                                for c in combintersection:
#                                    entropyorganizer[ind][c] += 1
#                        iterdiff = iterset.difference(coiterset)
#                        #^should i be checking if the union != the difference?
#                        if iterdiff:
#                            #difference exists, take the diff of the maininds
#                            itermaininds = itercombinds.difference(coitercombinds)
#                            #label = tuple(iterset.union(coiterset)) #automatically sorts
#                            label = tuple(iterset.difference(coiterset)) #automatically sorts
#                        else:
#                            #everything from the iter is within the coiter, mark everything as being from the itercomb key
#                            itermaininds = itercombinds
#                            #label = itercomb
#                            label = tuple(iterset)
#                        #match label:
#                        #    case int():
#                        #        for ind in itermaininds:
#                        #            entropyorganizer[ind][label] += 1
#                        #    case tuple():
#                        #        for ind in itermaininds:
#                        #            for l in label:
#                        #                entropyorganizer[ind][l] += 1
#                        for ind in itermaininds:
#                            for l in label:
#                                entropyorganizer[ind][l] += 1
#                        coiterdiff = coiterset.difference(iterset)
#                        if coiterdiff:
#                            #difference exists, take the diff of the maininds
#                            coitermaininds = coitercombinds.difference(itercombinds)
#                            #label = tuple(coiterset.union(iterset))
#                            label = tuple(coiterset.difference(iterset))
#                        else:
#                            #everything from the coiter is within the iter, mark everything as being from the coitercomb key
#                            coitermaininds = coitercombinds
#                            #label = coitercomb
#                            label = tuple(coiterset)
#                        #match label:
#                        #    case int():
#                        #        for ind in coitermaininds:
#                        #            entropyorganizer[ind][label] += 1
#                        #    case tuple():
#                        #        for ind in coitermaininds:
#                        #            for l in label:
#                        #                entropyorganizer[ind][l] += 1
#                        for ind in coitermaininds:
#                            for l in label:
#                                entropyorganizer[ind][l] += 1
#    else: #no competition
#        #for line, inds in maininds.items():
#        for n, inds in massindices.items():
#            for line in lineindices[n]:
#                for mi in inds:
#                    entropyorganizer[mi][line] += 1
#
#print(len(entropyorganizer), sum(len(v) for v in entropyorganizer.values()))
#print(time() - nt, 'reindexed single-pass reciprocating combinatoric (no non-overlap additions) finished')
#labelfailure = False
#for k, v in entropyorganizer.items():
#    for sv in v:
#        if type(sv) is tuple:
#            labelfailure = True
#print(labelfailure, 'labelfailure')
#
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
#incorrections = []
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
#        incorrections.append(mainind)
#    else: #good outcome
#        if trueinds == result:
#            correctoutcomes += 1
#        else:
#            notincorrectoutcomes += 1
#            distance = len(result.difference(trueinds))
#            notincorrectdistances[distance] += 1
#
#incorrectindices.append(incorrections)
#
#print(f'total wrong: {nwrongs}')
#print(f'total correct: {correctoutcomes}')
#print(f'total not incorrect {notincorrectoutcomes}')
#print('-------------------------------------------')
#
##nt = time()
##
##entropyorganizer = defaultdict(lambda: defaultdict(int)) #mainind: line: count
##for scangroup in mergedscans:
##    maininds = {linesofscans[i]: set(mainindicesbyscan[i]) for i in scangroup} #(lines): [mass indices]
##    if len(maininds) > 1:
##        #iterate a combination of size n, and co-iterate combinations of the (total length - n) to compare each group
##        maxlen = len(maininds)
##        #iterlength = maxlen - 1
##        iterlength = 1
##        while iterlength < maxlen:
##            coiterlength = maxlen - iterlength
##            for itercomb in itertools.combinations(maininds, iterlength):
##                iterset = set()
##                for i in itercomb:
##                    match i:
##                        case int():
##                            iterset.add(i)
##                        case tuple():
##                            iterset.update(i)
##                #itercomb = itercomb[0] #why was this here???
##                for coitercomb in itertools.combinations(maininds, coiterlength):
##                    coiterset = set()
##                    for c in coitercomb:
##                        match c:
##                            case int():
##                                coiterset.add(c)
##                            case tuple():
##                                coiterset.update(c)
##                    #coitercomb = coitercomb[0] #same with this???
##                    if itercomb != coitercomb:
##                        combintersection = iterset.intersection(coiterset)
##                        itercombinds = set(itertools.chain(*[maininds[i] for i in itercomb]))
##                        coitercombinds = set(itertools.chain(*[maininds[c] for c in coitercomb]))
##                        if combintersection:
##                            #mainindintersection = tuple(maininds[itercomb].intersection(maininds[coitercomb])) #assign this to everything in combintersection
##                            mainindintersection = itercombinds.intersection(coitercombinds)
##                            if mainindintersection:
##                                combintersection = tuple(combintersection)
##                                if len(combintersection) == 1:
##                                    combintersection = combintersection[0]
##                                #for ind in mainindintersection:
##                                #    entropyorganizer[ind][combintersection] += 1
##                                match combintersection:
##                                    case int():
##                                        for ind in mainindintersection:
##                                            entropyorganizer[ind][combintersection] += 1
##                                    case tuple():
##                                        for ind in mainindintersection:
##                                            for c in combintersection:
##                                                entropyorganizer[ind][c] += 1
##                            #get the difference and intersection or something?
##                            #of coiter and iter sets as well as maininds values
##                            iterdiff = iterset.difference(coiterset)
##                            #^should i be checking if the union != the difference?
##                            if iterdiff:
##                                #difference exists, take the diff of the maininds
##                                #itermaininds = tuple(maininds[itercomb].difference(maininds[coitercomb]))
##                                itermaininds = itercombinds.difference(coitercombinds)
##                                #label = tuple(iterset.union(coiterset)) #automatically sorts
##                                #why the fuck is the label a union while this thing is for diffs???
##                                #should i be getting the union, +1ing those, then +1ing different things for the diffs?
##                                label = tuple(iterset.difference(coiterset)) #automatically sorts
##                            else:
##                                #everything from the iter is within the coiter, mark everything as being from the itercomb key
##                                #itermaininds = tuple(maininds[itercomb])
##                                itermaininds = itercombinds
##                                label = itercomb
##                            #for ind in itermaininds:
##                            #    entropyorganizer[ind][label] += 1
##                            match label:
##                                case int():
##                                    for ind in itermaininds:
##                                        entropyorganizer[ind][label] += 1
##                                case tuple():
##                                    for ind in itermaininds:
##                                        for l in label:
##                                            entropyorganizer[ind][l] += 1
##                            coiterdiff = coiterset.difference(iterset)
##                            if coiterdiff:
##                                #difference exists, take the diff of the maininds
##                                #coitermaininds = tuple(maininds[coitercomb].difference(maininds[itercomb]))
##                                coitermaininds = coitercombinds.difference(itercombinds)
##                                #label = tuple(coiterset.union(iterset))
##                                label = tuple(coiterset.difference(iterset))
##                            else:
##                                #everything from the coiter is within the iter, mark everything as being from the coitercomb key
##                                coitermaininds = coitercombinds
##                                label = coitercomb
##                            #for ind in coitermaininds:
##                            #    entropyorganizer[ind][label] += 1
##                            match label:
##                                case int():
##                                    for ind in coitermaininds:
##                                        entropyorganizer[ind][label] += 1
##                                case tuple():
##                                    for ind in coitermaininds:
##                                        for l in label:
##                                            entropyorganizer[ind][l] += 1
##                        #else: #no intersection, +1 to either for what they have
##                        #    for i in iterset:
##                        #        for ic in itercombinds:
##                        #            entropyorganizer[ic][i] += 1
##                        #    for c in coiterset:
##                        #        for cc in coitercombinds:
##                        #            entropyorganizer[cc][i] += 1
##            iterlength += 1
##    else: #no competition
##        for line, inds in maininds.items():
##            for mi in inds:
##                entropyorganizer[mi][line] += 1
##
##print(time() - nt, 'reciprocating combinatoric (no non-overlap additions) finished')
##
##results = []
##for mainind, counts in entropyorganizer.items():
##    trueindices = linesbymainindex[mainind] #should be the answers...
##    counts = Counter(counts)
##    mostcommon = counts.most_common(len(counts))
##    if len(mostcommon) > 1: #> 1 result
##        if mostcommon[0][1] == mostcommon[1][1]:
##            maxcount = mostcommon[0][1]
##            #iterate and collect all
##            grouping = []
##            for line, c in mostcommon:
##                if c == maxcount:
##                    match line:
##                        case int():
##                            grouping.append(line)
##                        case tuple():
##                            for l in line:
##                                grouping.append(l)
##                else:
##                    break
##            result = tuple(set(grouping))
##            if trueindices.issubset(result):
##                outcome = tuple(result)
##            else:
##                #incorrect outcome
##                outcome = -1
##        else:
##            outcome = mostcommon[0][0]
##            result = mostcommon[0][0]
##            if outcome not in trueindices or len(trueindices) > 1:
##                #incorrect outcome via not matching everything
##                outcome = -2
##    else:
##        outcome = mostcommon[0][0]
##        result = mostcommon[0][0]
##        match outcome:
##            case int():
##                if outcome not in trueindices or len(trueindices) > 1:
##                    #incorrect outcome
##                    outcome = -3
##            case tuple():
##                if not trueindices.issubset(outcome):
##                    outcome = -4
##        #elif not trueindices.issubset(outcome):
##        #    outcome = -4
##    results.append([mainind, tuple(trueindices), outcome, result])
##
##nwrongs = 0
##correctoutcomes = 0
##notincorrectoutcomes = 0
##notincorrectdistances = Counter()
##badoutcomesbadmatches = Counter()
##badoutcomesgoodmatches = Counter()
##incorrections = []
##for r in results:
##    mainind, trueinds, outcome, result = r
##    if type(result) == int:
##        result = set((result,))
##    else:
##        result = set(result)
##    if type(trueinds) == int:
##        trueinds = set((trueinds,))
##    else:
##        trueinds = set(trueinds)
##    if type(outcome) is int and outcome < 0: #bad outcome
##        goodlength = len(trueinds.intersection(result))
##        badoutcomesgoodmatches[goodlength] += 1
##        badlength = len(trueinds.symmetric_difference(result))
##        badoutcomesbadmatches[badlength] += 1
##        nwrongs += 1
##        incorrections.append(mainind)
##    else: #good outcome
##        if trueinds == result:
##            correctoutcomes += 1
##        else:
##            notincorrectoutcomes += 1
##            distance = len(result.difference(trueinds))
##            notincorrectdistances[distance] += 1
##
##incorrectindices.append(incorrections)
##
##print(f'total wrong: {nwrongs}')
##print(f'total correct: {correctoutcomes}')
##print(f'total not incorrect {notincorrectoutcomes}')
##print('-------------------------------------------')
#
#nt = time()
#
#entropyorganizer = defaultdict(lambda: defaultdict(int)) #mainind: line: count
#for scangroup in mergedscans:
#    maininds = {linesofscans[i]: set(mainindicesbyscan[i]) for i in scangroup} #(lines): [mass indices]
#    if len(maininds) > 1:
#        #iterate a combination of size n, and co-iterate combinations of the (total length - n) to compare each group
#        maxlen = len(maininds)
#        iterlength = maxlen - 1
#        #while iterlength < maxlen:
#        coiterlength = maxlen - iterlength
#        #for itercomb in itertools.combinations(maininds, iterlength):
#        for itercomb in maininds:
#            iterset = set()
#            for i in itercomb:
#                match i:
#                    case int():
#                        iterset.add(i)
#                    case tuple():
#                        iterset.update(i)
#            for coitercomb in maininds:
#                if itercomb != coitercomb:
#                    coiterset = set()
#                    for c in coitercomb:
#                        match c:
#                            case int():
#                                coiterset.add(c)
#                            case tuple():
#                                coiterset.update(c)
#                    combintersection = iterset.intersection(coiterset)
#                    itercombinds = maininds[itercomb]
#                    coitercombinds = maininds[coitercomb]
#                    if combintersection:
#                        mainindintersection = itercombinds.intersection(coitercombinds)
#                        if mainindintersection:
#                            combintersection = tuple(combintersection)
#                            #if len(combintersection) == 1:
#                            #    combintersection = combintersection[0]
#                            #match combintersection:
#                            #    case int():
#                            #        for ind in mainindintersection:
#                            #            entropyorganizer[ind][combintersection] += 1
#                            #    case tuple():
#                            #        for ind in mainindintersection:
#                            #            for c in combintersection:
#                            #                entropyorganizer[ind][c] += 1
#                            for ind in mainindintersection:
#                                for c in combintersection:
#                                    entropyorganizer[ind][c] += 1
#                        #get the difference and intersection or something?
#                        #of coiter and iter sets as well as maininds values
#                        iterdiff = iterset.difference(coiterset)
#                        #^should i be checking if the union != the difference?
#                        if iterdiff:
#                            #difference exists, take the diff of the maininds
#                            #itermaininds = tuple(maininds[itercomb].difference(maininds[coitercomb]))
#                            itermaininds = itercombinds.difference(coitercombinds)
#                            #label = tuple(iterset.union(coiterset)) #automatically sorts
#                            #why the fuck is the label a union while this thing is for diffs???
#                            #should i be getting the union, +1ing those, then +1ing different things for the diffs?
#                            label = tuple(iterset.difference(coiterset)) #automatically sorts
#                        else:
#                            #everything from the iter is within the coiter, mark everything as being from the itercomb key
#                            #itermaininds = tuple(maininds[itercomb])
#                            itermaininds = itercombinds
#                            #label = itercomb
#                            label = tuple(iterset)
#                        #for ind in itermaininds:
#                        #    entropyorganizer[ind][label] += 1
#                        #match label:
#                        #    case int():
#                        #        for ind in itermaininds:
#                        #            entropyorganizer[ind][label] += 1
#                        #    case tuple():
#                        #        for ind in itermaininds:
#                        #            for l in label:
#                        #                entropyorganizer[ind][l] += 1
#                        for ind in itermaininds:
#                            for l in label:
#                                entropyorganizer[ind][l] += 1
#                        coiterdiff = coiterset.difference(iterset)
#                        if coiterdiff:
#                            #difference exists, take the diff of the maininds
#                            #coitermaininds = tuple(maininds[coitercomb].difference(maininds[itercomb]))
#                            coitermaininds = coitercombinds.difference(itercombinds)
#                            #label = tuple(coiterset.union(iterset))
#                            label = tuple(coiterset.difference(iterset))
#                        else:
#                            #everything from the coiter is within the iter, mark everything as being from the coitercomb key
#                            coitermaininds = coitercombinds
#                            #label = coitercomb
#                            label = tuple(coiterset)
#                        #for ind in coitermaininds:
#                        #    entropyorganizer[ind][label] += 1
#                        #match label:
#                        #    case int():
#                        #        for ind in coitermaininds:
#                        #            entropyorganizer[ind][label] += 1
#                        #    case tuple():
#                        #        for ind in coitermaininds:
#                        #            for l in label:
#                        #                entropyorganizer[ind][l] += 1
#                        for ind in coitermaininds:
#                            for l in label:
#                                entropyorganizer[ind][l] += 1
#                    #else: #no intersection, +1 to either for what they have
#                    #    for i in iterset:
#                    #        for ic in itercombinds:
#                    #            entropyorganizer[ic][i] += 1
#                    #    for c in coiterset:
#                    #        for cc in coitercombinds:
#                    #            entropyorganizer[cc][i] += 1
#            #iterlength += 1
#    else: #no competition
#        for lines, inds in maininds.items():
#            for mi in inds:
#                for line in lines:
#                    entropyorganizer[mi][line] += 1
#
#print(len(entropyorganizer), sum(len(v) for v in entropyorganizer.values()))
#print(time() - nt, 'simple dual combinatoric finished')
#labelfailure = False
#for k, v in entropyorganizer.items():
#    for sv in v:
#        if type(sv) is tuple:
#            labelfailure = True
#print(labelfailure, 'labelfailure')
#
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
#incorrections = []
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
#        incorrections.append(mainind)
#    else: #good outcome
#        if trueinds == result:
#            correctoutcomes += 1
#        else:
#            notincorrectoutcomes += 1
#            distance = len(result.difference(trueinds))
#            notincorrectdistances[distance] += 1
#
#incorrectindices.append(incorrections)
#
#print(f'total wrong: {nwrongs}')
#print(f'total correct: {correctoutcomes}')
#print(f'total not incorrect {notincorrectoutcomes}')
#print('-------------------------------------------')
#
#nt = time()
#
#entropyorganizer = defaultdict(lambda: defaultdict(int)) #mainind: line: count
#for scangroup in mergedscans:
#    #maininds = {linesofscans[i]: set(mainindicesbyscan[i]) for i in scangroup} #(lines): [mass indices]
#    lineindices = {}
#    massindices = {}
#    for n, scan in enumerate(scangroup):
#        lineindices[n] = linesofscans[scan]
#        massindices[n] = set(mainindicesbyscan[scan])
#    if n > 0:
#        #iterate a combination of size n, and co-iterate combinations of the (total length - n) to compare each group
#        maxlen = n + 1
#        iterlength = maxlen - 1
#        #while iterlength < maxlen:
#        coiterlength = maxlen - iterlength
#        #for itercomb in itertools.combinations(maininds, iterlength):
#        for itercomb in massindices:
#            iterset = set(lineindices[itercomb])
#            for coitercomb in massindices:
#                if itercomb != coitercomb:
#                    coiterset = set(lineindices[coitercomb])
#                    combintersection = iterset.intersection(coiterset)
#                    itercombinds = massindices[itercomb]
#                    coitercombinds = massindices[coitercomb]
#                    if combintersection:
#                        mainindintersection = itercombinds.intersection(coitercombinds)
#                        if mainindintersection:
#                            combintersection = tuple(combintersection)
#                            #if len(combintersection) == 1:
#                            #    combintersection = combintersection[0]
#                            #match combintersection:
#                            #    case int():
#                            #        for ind in mainindintersection:
#                            #            entropyorganizer[ind][combintersection] += 1
#                            #    case tuple():
#                            #        for ind in mainindintersection:
#                            #            for c in combintersection:
#                            #                entropyorganizer[ind][c] += 1
#                            for ind in mainindintersection:
#                                for c in combintersection:
#                                    entropyorganizer[ind][c] += 1
#                        #get the difference and intersection or something?
#                        #of coiter and iter sets as well as maininds values
#                        iterdiff = iterset.difference(coiterset)
#                        #^should i be checking if the union != the difference?
#                        if iterdiff:
#                            #difference exists, take the diff of the maininds
#                            #itermaininds = tuple(maininds[itercomb].difference(maininds[coitercomb]))
#                            itermaininds = itercombinds.difference(coitercombinds)
#                            #label = tuple(iterset.union(coiterset)) #automatically sorts
#                            #why the fuck is the label a union while this thing is for diffs???
#                            #should i be getting the union, +1ing those, then +1ing different things for the diffs?
#                            label = tuple(iterset.difference(coiterset)) #automatically sorts
#                        else:
#                            #everything from the iter is within the coiter, mark everything as being from the itercomb key
#                            #itermaininds = tuple(maininds[itercomb])
#                            itermaininds = itercombinds
#                            #label = itercomb
#                            label = tuple(iterset)
#                        #for ind in itermaininds:
#                        #    entropyorganizer[ind][label] += 1
#                        #match label:
#                        #    case int():
#                        #        for ind in itermaininds:
#                        #            entropyorganizer[ind][label] += 1
#                        #    case tuple():
#                        #        for ind in itermaininds:
#                        #            for l in label:
#                        #                entropyorganizer[ind][l] += 1
#                        for ind in itermaininds:
#                            for l in label:
#                                entropyorganizer[ind][l] += 1
#                        coiterdiff = coiterset.difference(iterset)
#                        if coiterdiff:
#                            #difference exists, take the diff of the maininds
#                            #coitermaininds = tuple(maininds[coitercomb].difference(maininds[itercomb]))
#                            coitermaininds = coitercombinds.difference(itercombinds)
#                            #label = tuple(coiterset.union(iterset))
#                            label = tuple(coiterset.difference(iterset))
#                        else:
#                            #everything from the coiter is within the iter, mark everything as being from the coitercomb key
#                            coitermaininds = coitercombinds
#                            #label = coitercomb
#                            label = tuple(coiterset)
#                        #for ind in coitermaininds:
#                        #    entropyorganizer[ind][label] += 1
#                        #match label:
#                        #    case int():
#                        #        for ind in coitermaininds:
#                        #            entropyorganizer[ind][label] += 1
#                        #    case tuple():
#                        #        for ind in coitermaininds:
#                        #            for l in label:
#                        #                entropyorganizer[ind][l] += 1
#                        for ind in coitermaininds:
#                            for l in label:
#                                entropyorganizer[ind][l] += 1
#                    #else: #no intersection, +1 to either for what they have
#                    #    for i in iterset:
#                    #        for ic in itercombinds:
#                    #            entropyorganizer[ic][i] += 1
#                    #    for c in coiterset:
#                    #        for cc in coitercombinds:
#                    #            entropyorganizer[cc][i] += 1
#            #iterlength += 1
#    else: #no competition
#        for n, inds in massindices.items():
#            for line in lineindices[n]:
#                for mi in inds:
#                    entropyorganizer[mi][line] += 1
#
#print(len(entropyorganizer), sum(len(v) for v in entropyorganizer.values()))
#print(time() - nt, 'reindexed simple dual combinatoric finished')
#labelfailure = False
#for k, v in entropyorganizer.items():
#    for sv in v:
#        if type(sv) is tuple:
#            labelfailure = True
#print(labelfailure, 'labelfailure')
#
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
#incorrections = []
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
#        incorrections.append(mainind)
#    else: #good outcome
#        if trueinds == result:
#            correctoutcomes += 1
#        else:
#            notincorrectoutcomes += 1
#            distance = len(result.difference(trueinds))
#            notincorrectdistances[distance] += 1
#
#incorrectindices.append(incorrections)
#
#print(f'total wrong: {nwrongs}')
#print(f'total correct: {correctoutcomes}')
#print(f'total not incorrect {notincorrectoutcomes}')
#print('-------------------------------------------')
#
##nt = time()
##
##entropyorganizer = defaultdict(lambda: defaultdict(int)) #mainind: line: count
##for scangroup in mergedscans:
##    maininds = {linesofscans[i]: set(mainindicesbyscan[i]) for i in scangroup} #(lines): [mass indices]
##    if len(maininds) > 1:
##        #iterate a combination of size n, and co-iterate combinations of the (total length - n) to compare each group
##        maxlen = len(maininds)
##        iterlength = maxlen - 1
##        while iterlength < maxlen:
##            coiterlength = maxlen - iterlength
##            for itercomb in itertools.combinations(maininds, iterlength):
##                iterset = set()
##                for i in itercomb:
##                    match i:
##                        case int():
##                            iterset.add(i)
##                        case tuple():
##                            iterset.update(i)
##                #itercomb = itercomb[0] #why was this here???
##                for coitercomb in itertools.combinations(maininds, coiterlength):
##                    coiterset = set()
##                    for c in coitercomb:
##                        match c:
##                            case int():
##                                coiterset.add(c)
##                            case tuple():
##                                coiterset.update(c)
##                    #coitercomb = coitercomb[0] #same with this???
##                    if itercomb != coitercomb:
##                        combintersection = iterset.intersection(coiterset)
##                        itercombinds = set(itertools.chain(*[maininds[i] for i in itercomb]))
##                        coitercombinds = set(itertools.chain(*[maininds[c] for c in coitercomb]))
##                        if combintersection:
##                            #mainindintersection = tuple(maininds[itercomb].intersection(maininds[coitercomb])) #assign this to everything in combintersection
##                            mainindintersection = itercombinds.intersection(coitercombinds)
##                            if mainindintersection:
##                                combintersection = tuple(combintersection)
##                                if len(combintersection) == 1:
##                                    combintersection = combintersection[0]
##                                #for ind in mainindintersection:
##                                #    entropyorganizer[ind][combintersection] += 1
##                                match combintersection:
##                                    case int():
##                                        for ind in mainindintersection:
##                                            entropyorganizer[ind][combintersection] += 1
##                                    case tuple():
##                                        for ind in mainindintersection:
##                                            for c in combintersection:
##                                                entropyorganizer[ind][c] += 1
##                            #get the difference and intersection or something?
##                            #of coiter and iter sets as well as maininds values
##                            iterdiff = iterset.difference(coiterset)
##                            #^should i be checking if the union != the difference?
##                            if iterdiff:
##                                #difference exists, take the diff of the maininds
##                                #itermaininds = tuple(maininds[itercomb].difference(maininds[coitercomb]))
##                                itermaininds = itercombinds.difference(coitercombinds)
##                                #label = tuple(iterset.union(coiterset)) #automatically sorts
##                                #why the fuck is the label a union while this thing is for diffs???
##                                #should i be getting the union, +1ing those, then +1ing different things for the diffs?
##                                label = tuple(iterset.difference(coiterset)) #automatically sorts
##                            else:
##                                #everything from the iter is within the coiter, mark everything as being from the itercomb key
##                                #itermaininds = tuple(maininds[itercomb])
##                                itermaininds = itercombinds
##                                label = itercomb
##                            #for ind in itermaininds:
##                            #    entropyorganizer[ind][label] += 1
##                            match label:
##                                case int():
##                                    for ind in itermaininds:
##                                        entropyorganizer[ind][label] += 1
##                                case tuple():
##                                    for ind in itermaininds:
##                                        for l in label:
##                                            entropyorganizer[ind][l] += 1
##                            coiterdiff = coiterset.difference(iterset)
##                            if coiterdiff:
##                                #difference exists, take the diff of the maininds
##                                #coitermaininds = tuple(maininds[coitercomb].difference(maininds[itercomb]))
##                                coitermaininds = coitercombinds.difference(itercombinds)
##                                #label = tuple(coiterset.union(iterset))
##                                label = tuple(coiterset.difference(iterset))
##                            else:
##                                #everything from the coiter is within the iter, mark everything as being from the coitercomb key
##                                coitermaininds = coitercombinds
##                                label = coitercomb
##                            #for ind in coitermaininds:
##                            #    entropyorganizer[ind][label] += 1
##                            match label:
##                                case int():
##                                    for ind in coitermaininds:
##                                        entropyorganizer[ind][label] += 1
##                                case tuple():
##                                    for ind in coitermaininds:
##                                        for l in label:
##                                            entropyorganizer[ind][l] += 1
##                        #else: #no intersection, +1 to either for what they have
##                        #    for i in iterset:
##                        #        for ic in itercombinds:
##                        #            entropyorganizer[ic][i] += 1
##                        #    for c in coiterset:
##                        #        for cc in coitercombinds:
##                        #            entropyorganizer[cc][i] += 1
##            iterlength += 1
##        #comb-size iters
##        for itersize in range(1, maxlen - 1):
##            baseiters = list(itertools.combinations(maininds, itersize))
##            for itercomb, coitercomb in itertools.combinations(baseiters, 2):
##                iterset = set()
##                for i in itercomb:
##                    match i:
##                        case int():
##                            iterset.add(i)
##                        case tuple():
##                            iterset.update(i)
##                coiterset = set()
##                for c in coitercomb:
##                    match c:
##                        case int():
##                            coiterset.add(c)
##                        case tuple():
##                            coiterset.update(c)
##                #coitercomb = coitercomb[0] #same with this???
##                combintersection = iterset.intersection(coiterset)
##                itercombinds = set(itertools.chain(*[maininds[i] for i in itercomb]))
##                coitercombinds = set(itertools.chain(*[maininds[c] for c in coitercomb]))
##                if combintersection:
##                    #mainindintersection = tuple(maininds[itercomb].intersection(maininds[coitercomb])) #assign this to everything in combintersection
##                    mainindintersection = itercombinds.intersection(coitercombinds)
##                    if mainindintersection:
##                        combintersection = tuple(combintersection)
##                        if len(combintersection) == 1:
##                            combintersection = combintersection[0]
##                        #for ind in mainindintersection:
##                        #    entropyorganizer[ind][combintersection] += 1
##                        match combintersection:
##                            case int():
##                                for ind in mainindintersection:
##                                    entropyorganizer[ind][combintersection] += 1
##                            case tuple():
##                                for ind in mainindintersection:
##                                    for c in combintersection:
##                                        entropyorganizer[ind][c] += 1
##                    #get the difference and intersection or something?
##                    #of coiter and iter sets as well as maininds values
##                    iterdiff = iterset.difference(coiterset)
##                    #^should i be checking if the union != the difference?
##                    if iterdiff:
##                        #difference exists, take the diff of the maininds
##                        #itermaininds = tuple(maininds[itercomb].difference(maininds[coitercomb]))
##                        itermaininds = itercombinds.difference(coitercombinds)
##                        #label = tuple(iterset.union(coiterset)) #automatically sorts
##                        #why the fuck is the label a union while this thing is for diffs???
##                        #should i be getting the union, +1ing those, then +1ing different things for the diffs?
##                        label = tuple(iterset.difference(coiterset)) #automatically sorts
##                    else:
##                        #everything from the iter is within the coiter, mark everything as being from the itercomb key
##                        #itermaininds = tuple(maininds[itercomb])
##                        itermaininds = itercombinds
##                        label = itercomb
##                    #for ind in itermaininds:
##                    #    entropyorganizer[ind][label] += 1
##                    match label:
##                        case int():
##                            for ind in itermaininds:
##                                entropyorganizer[ind][label] += 1
##                        case tuple():
##                            for ind in itermaininds:
##                                for l in label:
##                                    entropyorganizer[ind][l] += 1
##                    coiterdiff = coiterset.difference(iterset)
##                    if coiterdiff:
##                        #difference exists, take the diff of the maininds
##                        #coitermaininds = tuple(maininds[coitercomb].difference(maininds[itercomb]))
##                        coitermaininds = coitercombinds.difference(itercombinds)
##                        #label = tuple(coiterset.union(iterset))
##                        label = tuple(coiterset.difference(iterset))
##                    else:
##                        #everything from the coiter is within the iter, mark everything as being from the coitercomb key
##                        coitermaininds = coitercombinds
##                        label = coitercomb
##                    #for ind in coitermaininds:
##                    #    entropyorganizer[ind][label] += 1
##                    match label:
##                        case int():
##                            for ind in coitermaininds:
##                                entropyorganizer[ind][label] += 1
##                        case tuple():
##                            for ind in coitermaininds:
##                                for l in label:
##                                    entropyorganizer[ind][l] += 1
##                #else: #no intersection, +1 to either for what they have
##                #    for i in iterset:
##                #        for ic in itercombinds:
##                #            entropyorganizer[ic][i] += 1
##                #    for c in coiterset:
##                #        for cc in coitercombinds:
##                #            entropyorganizer[cc][i] += 1
##    else: #no competition
##        for line, inds in maininds.items():
##            for mi in inds:
##                entropyorganizer[mi][line] += 1
##
##print(time() - nt, 'multicombinatoric (neither non-overlap addition) entropy estimated')
##
##results = []
##for mainind, counts in entropyorganizer.items():
##    trueindices = linesbymainindex[mainind] #should be the answers...
##    counts = Counter(counts)
##    mostcommon = counts.most_common(len(counts))
##    if len(mostcommon) > 1: #> 1 result
##        if mostcommon[0][1] == mostcommon[1][1]:
##            maxcount = mostcommon[0][1]
##            #iterate and collect all
##            grouping = []
##            for line, c in mostcommon:
##                if c == maxcount:
##                    match line:
##                        case int():
##                            grouping.append(line)
##                        case tuple():
##                            for l in line:
##                                grouping.append(l)
##                else:
##                    break
##            result = tuple(set(grouping))
##            if trueindices.issubset(result):
##                outcome = tuple(result)
##            else:
##                #incorrect outcome
##                outcome = -1
##        else:
##            outcome = mostcommon[0][0]
##            result = mostcommon[0][0]
##            if outcome not in trueindices or len(trueindices) > 1:
##                #incorrect outcome via not matching everything
##                outcome = -2
##    else:
##        outcome = mostcommon[0][0]
##        result = mostcommon[0][0]
##        match outcome:
##            case int():
##                if outcome not in trueindices or len(trueindices) > 1:
##                    #incorrect outcome
##                    outcome = -3
##            case tuple():
##                if not trueindices.issubset(outcome):
##                    outcome = -4
##        #elif not trueindices.issubset(outcome):
##        #    outcome = -4
##    results.append([mainind, tuple(trueindices), outcome, result])
##
##nwrongs = 0
##correctoutcomes = 0
##notincorrectoutcomes = 0
##notincorrectdistances = Counter()
##badoutcomesbadmatches = Counter()
##badoutcomesgoodmatches = Counter()
##incorrections = []
##for r in results:
##    mainind, trueinds, outcome, result = r
##    if type(result) == int:
##        result = set((result,))
##    else:
##        result = set(result)
##    if type(trueinds) == int:
##        trueinds = set((trueinds,))
##    else:
##        trueinds = set(trueinds)
##    if type(outcome) is int and outcome < 0: #bad outcome
##        goodlength = len(trueinds.intersection(result))
##        badoutcomesgoodmatches[goodlength] += 1
##        badlength = len(trueinds.symmetric_difference(result))
##        badoutcomesbadmatches[badlength] += 1
##        nwrongs += 1
##        incorrections.append(mainind)
##    else: #good outcome
##        if trueinds == result:
##            correctoutcomes += 1
##        else:
##            notincorrectoutcomes += 1
##            distance = len(result.difference(trueinds))
##            notincorrectdistances[distance] += 1
##
##incorrectindices.append(incorrections)
##
##print(f'total wrong: {nwrongs}')
##print(f'total correct: {correctoutcomes}')
##print(f'total not incorrect {notincorrectoutcomes}')
##print('-------------------------------------------')
##
##nt = time()
##
##entropyorganizer = defaultdict(lambda: defaultdict(int)) #mainind: line: count
##for scangroup in mergedscans:
##    maininds = {linesofscans[i]: set(mainindicesbyscan[i]) for i in scangroup} #(lines): [mass indices]
##    if len(maininds) > 1:
##        #iterate a combination of size n, and co-iterate combinations of the (total length - n) to compare each group
##        maxlen = len(maininds)
##        #iterlength = maxlen - 1
##        #while iterlength < maxlen:
##        for iterlength in range(1, maxlen-1):
##            #coiterlength = maxlen - iterlength
##            for itercomb in itertools.combinations(maininds, iterlength):
##                iterset = set()
##                for i in itercomb:
##                    match i:
##                        case int():
##                            iterset.add(i)
##                        case tuple():
##                            iterset.update(i)
##                for coiterlength in range(iterlength, maxlen-1):
##                    for coitercomb in itertools.combinations(maininds, coiterlength):
##                        coiterset = set()
##                        for c in coitercomb:
##                            match c:
##                                case int():
##                                    coiterset.add(c)
##                                case tuple():
##                                    coiterset.update(c)
##                        #coitercomb = coitercomb[0] #same with this???
##                        if itercomb != coitercomb:
##                            combintersection = iterset.intersection(coiterset)
##                            itercombinds = set(itertools.chain(*[maininds[i] for i in itercomb]))
##                            coitercombinds = set(itertools.chain(*[maininds[c] for c in coitercomb]))
##                            if combintersection:
##                                #mainindintersection = tuple(maininds[itercomb].intersection(maininds[coitercomb])) #assign this to everything in combintersection
##                                mainindintersection = itercombinds.intersection(coitercombinds)
##                                if mainindintersection:
##                                    combintersection = tuple(combintersection)
##                                    if len(combintersection) == 1:
##                                        combintersection = combintersection[0]
##                                    #for ind in mainindintersection:
##                                    #    entropyorganizer[ind][combintersection] += 1
##                                    match combintersection:
##                                        case int():
##                                            for ind in mainindintersection:
##                                                entropyorganizer[ind][combintersection] += 1
##                                        case tuple():
##                                            for ind in mainindintersection:
##                                                for c in combintersection:
##                                                    entropyorganizer[ind][c] += 1
##                                #get the difference and intersection or something?
##                                #of coiter and iter sets as well as maininds values
##                                iterdiff = iterset.difference(coiterset)
##                                #^should i be checking if the union != the difference?
##                                if iterdiff:
##                                    #difference exists, take the diff of the maininds
##                                    #itermaininds = tuple(maininds[itercomb].difference(maininds[coitercomb]))
##                                    itermaininds = itercombinds.difference(coitercombinds)
##                                    #label = tuple(iterset.union(coiterset)) #automatically sorts
##                                    #why the fuck is the label a union while this thing is for diffs???
##                                    #should i be getting the union, +1ing those, then +1ing different things for the diffs?
##                                    label = tuple(iterset.difference(coiterset)) #automatically sorts
##                                else:
##                                    #everything from the iter is within the coiter, mark everything as being from the itercomb key
##                                    #itermaininds = tuple(maininds[itercomb])
##                                    itermaininds = itercombinds
##                                    label = itercomb
##                                #for ind in itermaininds:
##                                #    entropyorganizer[ind][label] += 1
##                                match label:
##                                    case int():
##                                        for ind in itermaininds:
##                                            entropyorganizer[ind][label] += 1
##                                    case tuple():
##                                        for ind in itermaininds:
##                                            for l in label:
##                                                entropyorganizer[ind][l] += 1
##                                coiterdiff = coiterset.difference(iterset)
##                                if coiterdiff:
##                                    #difference exists, take the diff of the maininds
##                                    #coitermaininds = tuple(maininds[coitercomb].difference(maininds[itercomb]))
##                                    coitermaininds = coitercombinds.difference(itercombinds)
##                                    #label = tuple(coiterset.union(iterset))
##                                    label = tuple(coiterset.difference(iterset))
##                                else:
##                                    #everything from the coiter is within the iter, mark everything as being from the coitercomb key
##                                    coitermaininds = coitercombinds
##                                    label = coitercomb
##                                #for ind in coitermaininds:
##                                #    entropyorganizer[ind][label] += 1
##                                match label:
##                                    case int():
##                                        for ind in coitermaininds:
##                                            entropyorganizer[ind][label] += 1
##                                    case tuple():
##                                        for ind in coitermaininds:
##                                            for l in label:
##                                                entropyorganizer[ind][l] += 1
##                            #else: #no intersection, +1 to either for what they have
##                            #    for i in iterset:
##                            #        for ic in itercombinds:
##                            #            entropyorganizer[ic][i] += 1
##                            #    for c in coiterset:
##                            #        for cc in coitercombinds:
##                            #            entropyorganizer[cc][i] += 1
##            iterlength += 1
##    else: #no competition
##        for line, inds in maininds.items():
##            for mi in inds:
##                entropyorganizer[mi][line] += 1
##
##print(time() - nt, 'length-rotation embedded multicombinatoric (no non-overlapping additions) entropy estimated')
##
##results = []
##for mainind, counts in entropyorganizer.items():
##    trueindices = linesbymainindex[mainind] #should be the answers...
##    counts = Counter(counts)
##    mostcommon = counts.most_common(len(counts))
##    if len(mostcommon) > 1: #> 1 result
##        if mostcommon[0][1] == mostcommon[1][1]:
##            maxcount = mostcommon[0][1]
##            #iterate and collect all
##            grouping = []
##            for line, c in mostcommon:
##                if c == maxcount:
##                    match line:
##                        case int():
##                            grouping.append(line)
##                        case tuple():
##                            for l in line:
##                                grouping.append(l)
##                else:
##                    break
##            result = tuple(set(grouping))
##            if trueindices.issubset(result):
##                outcome = tuple(result)
##            else:
##                #incorrect outcome
##                outcome = -1
##        else:
##            outcome = mostcommon[0][0]
##            result = mostcommon[0][0]
##            if outcome not in trueindices or len(trueindices) > 1:
##                #incorrect outcome via not matching everything
##                outcome = -2
##    else:
##        outcome = mostcommon[0][0]
##        result = mostcommon[0][0]
##        match outcome:
##            case int():
##                if outcome not in trueindices or len(trueindices) > 1:
##                    #incorrect outcome
##                    outcome = -3
##            case tuple():
##                if not trueindices.issubset(outcome):
##                    outcome = -4
##        #elif not trueindices.issubset(outcome):
##        #    outcome = -4
##    results.append([mainind, tuple(trueindices), outcome, result])
##
##nwrongs = 0
##correctoutcomes = 0
##notincorrectoutcomes = 0
##notincorrectdistances = Counter()
##badoutcomesbadmatches = Counter()
##badoutcomesgoodmatches = Counter()
##incorrections = []
##for r in results:
##    mainind, trueinds, outcome, result = r
##    if type(result) == int:
##        result = set((result,))
##    else:
##        result = set(result)
##    if type(trueinds) == int:
##        trueinds = set((trueinds,))
##    else:
##        trueinds = set(trueinds)
##    if type(outcome) is int and outcome < 0: #bad outcome
##        goodlength = len(trueinds.intersection(result))
##        badoutcomesgoodmatches[goodlength] += 1
##        badlength = len(trueinds.symmetric_difference(result))
##        badoutcomesbadmatches[badlength] += 1
##        nwrongs += 1
##        incorrections.append(mainind)
##    else: #good outcome
##        if trueinds == result:
##            correctoutcomes += 1
##        else:
##            notincorrectoutcomes += 1
##            distance = len(result.difference(trueinds))
##            notincorrectdistances[distance] += 1
##
##incorrectindices.append(incorrections)
##
##print(f'total wrong: {nwrongs}')
##print(f'total correct: {correctoutcomes}')
##print(f'total not incorrect {notincorrectoutcomes}')
##print('-------------------------------------------')
#
#incorrectcounts = Counter()
#for group in incorrectindices:
#    for g in group:
#        incorrectcounts[g] += 1

nt = time()

entropyorganizer = defaultdict(lambda: defaultdict(int)) #mainind: line: count
for scangroup in mergedscans:
    lineindices = {}
    massindices = {}
    for n, scan in enumerate(scangroup):
        lineindices[n] = set(linesofscans[scan])
        massindices[n] = set(mainindicesbyscan[scan])
    if n > 0:
        blocked = set()
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
                        #coiterdiff = coiterset.difference(iterset)
                        #if coiterdiff:
                        #    #difference exists, take the diff of the maininds
                        #    coitermaininds = coitercombinds.difference(itercombinds)
                        #    label = tuple(coiterset.difference(iterset))
                        #else:
                        #    #everything from the coiter is within the iter, mark everything as being from the coitercomb key
                        #    coitermaininds = coitercombinds
                        #    label = tuple(coiterset)
                        #for ind in coitermaininds:
                        #    for l in label:
                        #        entropyorganizer[ind][l] += 1
    else: #no competition
        for n, inds in massindices.items():
            for line in lineindices[n]:
                for mi in inds:
                    entropyorganizer[mi][line] += 1

print(time() - nt, 'entropy estimated')

results = []
for mainind, counts in entropyorganizer.items():
    trueindices = linesbymainindex[mainind] #should be the answers...
    counts = Counter(counts)
    mostcommon = counts.most_common(len(counts))
    if len(mostcommon) > 1: #> 1 result
        if mostcommon[0][1] == mostcommon[1][1]:
            maxcount = mostcommon[0][1]
            #iterate and collect all
            grouping = []
            for line, c in mostcommon:
                if c == maxcount:
                    match line:
                        case int():
                            grouping.append(line)
                        case tuple():
                            for l in line:
                                grouping.append(l)
                else:
                    break
            result = tuple(set(grouping))
            if trueindices.issubset(result):
                outcome = tuple(result)
            else:
                #incorrect outcome
                outcome = -1
        else:
            outcome = mostcommon[0][0]
            result = mostcommon[0][0]
            if outcome not in trueindices or len(trueindices) > 1:
                #incorrect outcome via not matching everything
                outcome = -2
    else:
        outcome = mostcommon[0][0]
        result = mostcommon[0][0]
        match outcome:
            case int():
                if outcome not in trueindices or len(trueindices) > 1:
                    #incorrect outcome
                    outcome = -3
            case tuple():
                if not trueindices.issubset(outcome):
                    outcome = -4
        #elif not trueindices.issubset(outcome):
        #    outcome = -4
    results.append([mainind, tuple(trueindices), outcome, result])

nwrongs = 0
correctoutcomes = 0
notincorrectoutcomes = 0
notincorrectdistances = Counter()
badoutcomesbadmatches = Counter()
badoutcomesgoodmatches = Counter()
for r in results:
    mainind, trueinds, outcome, result = r
    if type(result) == int:
        result = set((result,))
    else:
        result = set(result)
    if type(trueinds) == int:
        trueinds = set((trueinds,))
    else:
        trueinds = set(trueinds)
    if type(outcome) is int and outcome < 0: #bad outcome
        goodlength = len(trueinds.intersection(result))
        badoutcomesgoodmatches[goodlength] += 1
        badlength = len(trueinds.symmetric_difference(result))
        badoutcomesbadmatches[badlength] += 1
        nwrongs += 1
    else: #good outcome
        if trueinds == result:
            correctoutcomes += 1
        else:
            notincorrectoutcomes += 1
            distance = len(result.difference(trueinds))
            notincorrectdistances[distance] += 1

print(f'total wrong: {nwrongs}')
print(f'total correct: {correctoutcomes}')
print(f'total not incorrect {notincorrectoutcomes}')

singles = 0
multiples = 0
entropicboundaries = {} #mainind: [lines]
for mainind, counts in entropyorganizer.items():
    counts = Counter(counts)
    mostcommon = counts.most_common(len(counts))
    if len(mostcommon) > 1: #> 1 result
        if mostcommon[0][1] == mostcommon[1][1]:
            maxcount = mostcommon[0][1]
            #iterate and collect all
            grouping = []
            for line, c in mostcommon:
                if c == maxcount:
                    match line:
                        case int():
                            grouping.append(line)
                        case tuple():
                            for l in line:
                                grouping.append(l)
                else:
                    break
            outcome = tuple(grouping)
            multiples += 1
        else:
            outcome = mostcommon[0][:1]
            singles += 1
    else:
        outcome = mostcommon[0][:1]
        singles += 1
    entropicboundaries[mainind] = outcome

print(time() - nt, 'entropic boundaries')
print(f'singles: {singles}, multiples: {multiples}')

fine = 0
failcount = 0
somewhatwrong = 0
broken = False
#are there maininds in scans that don't have their determined boundarylines?
for mainind, lines in entropicboundaries.items():
    indscans = scansbymainindices[mainind]
    boundaryscans = tuple(set(itertools.chain(*[scansoflines[i] for i in lines])))
    if indscans.difference(boundaryscans):
    #    for scan in indscans:
    #        lines = linesofscans[scan]
    #        if len(set(lines).intersection(entropyorganizer[mainind])) == 0:
        if len(linesbymainindex[mainind].intersection(entropyorganizer[mainind])) < len(linesbymainindex[mainind]):
                print('mainind:', mainind)
                print('lines:', lines)
                print('indscans:', indscans)
                print('boundaryscans:', boundaryscans)
                print('entropyorganizer:', entropyorganizer[mainind])
                print('linesbymainindex:', linesbymainindex[mainind])
                print('lines of scans:', [linesofscans[i] for i in indscans])
                print('scans of ^lines:', [{j: scansoflines[j] for j in linesofscans[i]} for i in indscans])
                broken = True
                break
        if broken:
            break

#in terms of how this played out, having more consistent fragmentation patterns decreased the number of wrongs, and increased the number of correct and not incorrect!
#a great outcome
#still, see if you can find anything specific about the wrong line matches to weed them out without knowing

#for fake fragment-databasing, generate the entire spectrum of fragments and apply a probability to most of them
#>generate them all
#>sort them randomly
#>use this order as the basis for iteration, restart loop until finished
#>random factors are:
# - increase/decreasing % with each iteration
# - max jump to the next %

#~
#so after introducing fragmented peptide masses, the complexity has overloaded the entropy engine, and the results end up coming out non-ideally
#it was still a childrens toy
#the complexity of the merging masses are greater than the complexity of their differences
#there's something to be said about merging masses that are within the same scan, i'm not in favor of it
#within each scangroup or perhaps across them all is a good starting point for determining an acceptable distance of each mergable match
#there's also something to say about doing this pre vs post deconvolution and i think i'm going to ditch the convolution and search the raw spectrum for specific charge states
#there's also something to say about missing a single ion of a distribution in one scan and matching it in another -> ie it doesn't show up, flickers out of one of them -> but anyways, your deconvolution would be too rigid to deal with this anyways? its something that can be handled in the entropy engine by accepting that dist in both scans because it has a full match somewhere at least -> this can be handled easier by searching the charge states rather than deconvoluting

#find the closest mass to each other mass across scans, and within each scan maybe?
#and treat it like a network
#intersection merge the results and work
#ONLY check combos from line pairs that directly overlap in a scan, don't check across every comparable scan in the scangroup
#^this way even if a single mass is in every scan of non-overlapping yet linked spectra, you can still distinguish how much of each line contributes to it in their relative scans via intensity modeling
#^which sounds like the next simulation i'll make after this, to try to guess the implicit input of a value contributed to by multiple sources across a linear strip of things that all contribute -> and what makes it feasible is their ms1 mass intensity as the input representation
#^so what you'll end up iterating are maininds while referring to the intensity of each primaryind for the modeling
#^you would actually look at multiple maininds at once, whichever ones overlap across the scans in a scangroup
#you would also, by this point, narrow down individual ion patterns that follow each line in every scan it goes to
#which lines become included in the measurement are dictated by logic and the rules of the logic can either expel an ion from a weighing process or determine how much signal is there relative to each input based on maybe about > 3 points of logic?
#because basically, if you have a 1:1 ratio of suspects, then it could be either that's correct, so then leave it as a wild-match post-entropy. if you get something that is 1:1:1:1 across like 4 lines in different scans that only connects 2 at a time, then it becomes a different kind of logic machine you can try and infer values from what they should be
#when observing a specific split. and compared to other potenial splits, how does the contribution of this one (as a potential legitimate match) work for the intensity-based entropy of the entire scan?

#first set up an intensity-based entropy system that classifies which intensities belong to which line, of like 2 lines or something, an easy guess
#simle preset mass values and intensities
#add in 1-2 overlaps and model how to quantify their correct values
#then make it more complex and see what kind of performance you get
#figure out where to put in the "wild card" determiner for single overlaps

#modifying this original version first?
#change the mechanism of fragment prob generation, does this give you a different answer?
#also determine where a line leaves no trace whenever its supposed to, where are the bad apples that might be missing and interupting the logic through a loss of info?
