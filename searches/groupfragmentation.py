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

mzmlfile = '/home/sfo/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
msrun = mzml.MzML(mzmlfile, dtype=np.float64)

basefolder = '/'.join((mzmlfile.split('/')[:-2]))
basefile = mzmlfile.split('/')[-1].split('.mzML')[0]

processinglocation = '/'.join((basefolder, 'fileprocessing', basefile))
fragmentlocation = '/'.join((basefolder, 'fileprocessing', basefile, 'fragments'))
librarylocation = '/home/sfo/data/proteomics/fastas/search-db'
csvfilename = '/'.join((fragmentlocation, 'fragments'))
proteome = 'Human_Homo_sapien'
nprocs = os.cpu_count()
proton = 1.007276554940804
dividingthreshold = 0.1
ppmtol = 25
ppmmod = ppmtol / 1000000

ions = 'by'

if not os.path.isdir(fragmentlocation):
    os.mkdir(fragmentlocation)

environment_partial = partial(lmdb.Environment, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=9999, max_spare_txns=1, lock=True)

scanmassesfile = '/'.join((processinglocation, 'scanmasses.pickle'))
with open(scanmassesfile, 'rb') as pick:
    scanmasses = pickle.load(pick)
#scanmasses = {} #scan: [masses]

analytefile = '/'.join((processinglocation, 'analytefactors.pickle'))
with open(analytefile, 'rb') as pick:
    analytesbydistribution, distributionsoflines = pickle.load(pick)[2:4]
#analytesbydistribution = {} #distid: analyte id
#distributionsoflines = {} #lineid: distid

linesbyscanbysubformulafile = '/'.join((processinglocation, 'linesbyscanbysubformula.pickle'))
with open(linesbyscanbysubformulafile, 'rb') as pick:
    linesbyscanbysubformula = pickle.load(pick)
#linesbyscanbysubformula = {} #subformula: scan: [lines]

divisionfile = '/'.join((processinglocation, 'dividedgroups.pickle'))
with open(divisionfile, 'rb') as pick:
    dividedgroups = pickle.load(pick)

elementsofprobindicesfile = '/'.join((processinglocation, 'elementsofprobabilityindices.pickle'))
with open(elementsofprobindicesfile, 'rb') as pick:
    elementsofprobabilityindices = pickle.load(pick)
#elementsofprobabilityindices = {} #prob index: e

probabilityorganizerfile = '/'.join((processinglocation, 'probabilityorganizer.pickle'))
with open(probabilityorganizerfile, 'rb') as pick:
    probabilityorganizer = pickle.load(pick)
#probabilityorganizer = defaultdict(dict) #prob index: iso: prob

matchprobfile = '/'.join((processinglocation, 'matchprobabilities.pickle'))
with open(matchprobfile, 'rb') as pick:
    matchprobabilities = pickle.load(pick)
#matchprobabilities = defaultdict(list) #subformula: [prob indices]

subformulasubindsfile = '/'.join((processinglocation, 'subformulasubindices.pickle'))
with open(subformulasubindsfile, 'rb') as pick:
    subformulasubindices = pickle.load(pick)
#subformulasubindices = defaultdict(list) #subformula: [sub match indices]

submatchsequencesfile = '/'.join((processinglocation, 'submatchsequences.pickle'))
with open(submatchsequencesfile, 'rb') as pick:
    submatchsequences = pickle.load(pick)
#submatchsequences = {} #submatchindex: sequence

#mainindexfile = '/'.join((processinglocation, 'mainindicesbysubindex.pickle'))
#with open(mainindexfile, 'rb') as pick:
#    mainindexbysubindex = pickle.load(pick)
##mainindexbysubindex = defaultdict(list) #sub match index: main match index

#analytesbymainindexfile = '/'.join((processinglocation, 'analytesbymainindex.pickle'))
#with open(analytesbymainindexfile, 'rb') as pick:
#    analytesbymainindex = pickle.load(pick)
##analytesbymainindex = {} #main match index: [scans]

#scansbyanalytefile = '/'.join((processinglocation, 'scansbyanalyte.pickle'))
#with open(scansbyanalytefile, 'rb') as pick:
#    scansbyanalyte = pickle.load(pick)
##scansbyanalyte = defaultdict(list) #analyteid: [scans across all lines and charge states]

#submatchpositionsfile = '/'.join((processinglocation, 'submatchpositions.pickle'))
#with open(submatchpositionsfile, 'rb') as pick:
#    submatchpositions = pickle.load(pick)
##submatchpositions = {} #submatch index: [distribution position, subiso position]

with environment_partial(librarylocation) as env:
    aas = env.open_db('aminoacids'.encode())
    with env.begin(write=False) as txn:
        with txn.cursor(aas) as cursor:
            aaget = cursor.get(proteome.encode()).decode()
            aminoacidcomposition = eval(aaget)

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

nonmonoisotopicelements = {'H2', 'C13', 'N15', 'O17', 'O18', 'S33', 'S34', 'S36'}

isotopesbyelement = {
        'H': ('H1', 'H2'),
        'C': ('C12', 'C13'),
        'N': ('N14', 'N15'),
        'O': ('O16', 'O17', 'O18'),
        'S': ('S32', 'S34', 'S33', 'S36')} #in order of abundance

elementvectors = {}
nvectorpositions = {}
elementpositions = {}
for e, isos in isotopesbyelement.items():
    elementvectors[e] = [0 for _ in range(len(isos))]
    nvectorpositions[e] = {k: n for n, k in enumerate(isos)}
    elementpositions[e] = {n: k for n, k in enumerate(isos)}

monoisotopickeys = {
        'H': 'H1',
        'C': 'C12',
        'N': 'N14',
        'O': 'O16',
        'S': 'S32'}

nonmonoisotopicgroups = {
        'H': ('H2',),
        'C': ('C13',),
        'N': ('N15',),
        'O': ('O17', 'O18'),
        'S': ('S33', 'S34', 'S36')}

#loaded above
#aminoacidcomposition = {
#        'A': {'C': 3, 'H': 5, 'N': 1, 'O': 1},
#        'R': {'C': 6, 'H': 12, 'N': 4, 'O': 1},
#        'N': {'C': 4, 'H': 6, 'N': 2, 'O': 2},
#        'D': {'C': 4, 'H': 5, 'N': 1, 'O': 3},
#        'C': {'C': 3, 'H': 5, 'N': 1, 'O': 1, 'S': 1},
#        'Q': {'C': 5, 'H': 8, 'N': 2, 'O': 2},
#        'E': {'C': 5, 'H': 7, 'N': 1, 'O': 3},
#        'G': {'C': 2, 'H': 3, 'N': 1, 'O': 1},
#        'H': {'C': 6, 'H': 7, 'N':3, 'O': 1},
#        'I': {'C': 6, 'H': 11, 'N':1, 'O': 1},
#        'L': {'C': 6, 'H': 11, 'N':1, 'O': 1},
#        'K': {'C': 6, 'H': 12, 'N': 2, 'O': 1},
#        'M': {'C': 5, 'H': 9, 'N':1, 'O': 1, 'S': 1},
#        'F': {'C': 9, 'H': 9, 'N':1, 'O': 1},
#        'P': {'C': 5, 'H': 7, 'N':1, 'O': 1},
#        'S': {'C': 3, 'H': 5, 'N':1, 'O': 2},
#        'T': {'C': 4, 'H': 7, 'N':1, 'O': 2},
#        'W': {'C': 11, 'H': 10, 'N': 2, 'O': 1},
#        'Y': {'C': 9, 'H': 9, 'N': 1, 'O': 2},
#        'V': {'C': 5, 'H': 9, 'N': 1, 'O': 1}
#        }

nfragmentcompositions = {}
nfragmentcompositions['a'] = Counter({'C': -1, 'O': -1})
nfragmentcompositions['b'] = Counter()
nfragmentcompositions['c'] = Counter({'N': 1, 'H': 3})

cfragmentcompositions = {}
cfragmentcompositions['x'] = Counter({'C': 1, 'O': 2})
cfragmentcompositions['y'] = Counter({'H': 2, 'O': 1})
cfragmentcompositions['z'] = Counter({'N': -1, 'H': -1, 'O': 1})
#cfragmentcompositions['z•'] = Counter({'N': -1, 'O': 1})

#staticmods = {
#        #'C': {'H': 5, 'C': 3, 'N':1, 'O':1}, #acrylamide
#        }
#
#for aa, sad in staticmods.items():
#    for saa, sav in sad.items():
#        aminoacidcomposition[aa][saa] += sav
#
##need to modify this organization to allow more than one type of mod on the same AA
#variablemods = {
#        'C': {'H': 5, 'C': 3, 'N':1, 'O':1}, #acrylamide
#        }
#
##there should be more than enough in the alphabet for any reasonable number of variable mods I suppose
#newmods = {} #new AA letter: its atomic composition
#modifiers = defaultdict(set) #existing AA: [new AA letters]
#modoriginals = {} #new AA letter: existing AA
#variablecharacters = string.ascii_lowercase
#for vn, (va, vad) in enumerate(variablemods.items()):
#    representativecharacter = variablecharacters[vn]
#    modifiers[va].add(representativecharacter)
#    newmods[representativecharacter] = vad
#    modoriginals[representativecharacter] = va
#    aminoacidcomposition[representativecharacter] = aminoacidcomposition[va].copy()
#    for vaa, vav in vad.items():
#        aminoacidcomposition[representativecharacter][vaa] += vav

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
        #matches.append(submatches)
        if submatches:
            matches[fn] = submatches
    return matches

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
                fc = fragment_composition.get(k, 0) + modcomp.get(k, 0)
                if fc > 0:
                    fragment_composition[k] = fc
                else:
                    del fragment_composition[k]
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
                fc = fragment_composition.get(k, 0) + modcomp.get(k, 0)
                if fc > 0:
                    fragment_composition[k] = fc
                else:
                    del fragment_composition[k]
            fragments[ion + str(n + 1)] = fragment_composition

    return fragments

#both of the fragment distribution-related functions have been fact-checked via brute-force, they're LEGIT
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

#both of the fragment distribution-related functions have been fact-checked via brute-force, they're LEGIT
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
            ind = bisect.bisect(probabilityranking, r)
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
                        ind = bisect.bisect(probabilityranking, newratio)
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
                            ind = bisect.bisect(probabilityranking, newratio)
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

def group_fragmentation(dividingthreshold, csvfilename, frag_func, group, count):
    searchtime = 0
    fraglens = 0
    scanlens = 0
    nt = time()
    positioncache = {}
    elementalcache = {}
    descentcache = {}
    finaloutput = []
    groupseqs = []
    groupsubformulas = []
    for member in group:
        if '(' in member:
            groupsubformulas.append(member)
        else:
            groupseqs.append(member)
    fragments = {}
    for seq in groupseqs:
        fragments[seq] = frag_func(seq)
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
                        #try/except if faster than an if/else, so i might as well
                        try:
                            elementlist = elementalcache[fragstring]
                            positions = positioncache[fragstring]
                        except KeyError: #not in cache
                            elementlist, positions = fragment_element_binomial_walk(dividingthreshold, e, c, fragprobs)
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
                    fragformulas, massesandabundances = fragment_descending_partial_products(dividingthreshold, elementalorganizer, fragmentpositions)
                    descentcache[fragstrings] = fragformulas, massesandabundances
                for n, (m, i) in enumerate(massesandabundances.tolist()):
                    out = (seq, fragformulas[n].decode(), ion, submatchindex, i)
                    #out = (seq, fragformulas[n].decode(), ion, submatchindex, i, mainindexbysubmatchindex[submatchindex]) #do i need main index???
                    output.append(out)
                    fragmasses.append(m)
        fragmasses.sort()
        fraglens += len(fragmasses)
        #to do a better search, i might group the masses the same way i do for entropy first, it would be faster, right? it might do multiple scans at once like this while matching to a truer "entity" that can also be dynamically varied (as in varied depending on whether they're actually grouped with other scans in the same way EVERY time) depending on which scangroup i want to select
        st = time()
        for scan, lines in linesbyscanbysubformula[subformula].items():
            if len(lines) > 1:
                analyteid = '-'.join((str(analytesbydistribution[distributionsoflines[i]]) for i in lines))
                lines = '-'.join((str(i) for i in lines))
            else:
                lines = lines[0]
                analyteid = analytesbydistribution[distributionsoflines[lines]]
            fragscan = scanmasses[scan]
            scanlens += len(fragscan)
            matches = radius_neighbors(fragscan, fragmasses, ppmmod)
            for n, m in matches.items(): #fragmass index: [scan indices]
                mstring = '-'.join((str(i) for i in m))
                out = (*output[n], analyteid, lines, scan, fragmasses[n], mstring)
                #^mstring here has the indices of the matched masses in a scan
                outstring = ','.join((str(i) for i in out)) + '\n' #smaller memory footprint + needs to be written like this later, only ~1-2 microsecond cost
                finaloutput.append(outstring)
        searchtime += time() - st
    fragtime = time() - nt - searchtime
    with open(csvfilename + '.' + str(count) + '.matches.csv', 'w') as f:
        for piece in finaloutput:
            f.write(piece)
    print(f'{round(time() - nt,4)} - group {count} saved, fragtime: {round(fragtime, 4)}, searchtime: {round(searchtime,4)}, generated fragments: {fraglens}, scan masses: {scanlens}, matches: {len(finaloutput)}')
    #return finaloutput

def group_processing(csvfilename, dividingthreshold, elementsofprobabilityindices, probabilityorganizer, matchprobabilities, subformulasubindices, submatchsequences, linesbyscanbysubformula, analytesbydistribution, distributionsoflines, scanmasses, dividedgroups, ions):
    #positioncache = {} #fragtuple: iso: position
    #elementalcache = {} #fragtuple: [[iso heaps]]
    #descentcache = {} #fragtuple group: (subformulas, massesandabundances)
    #descentcounter = Counter()

    #this ends up being slower than doing it linearly???? and i don't know why using normal dicts works so well, but it does cause more memory to be taken up, and total memory fluctuates up and down for some reason. BUT IT WORKS!
    #positioncache = mp.Manager().dict() #fragtuple: iso: position
    #elementalcache = mp.Manager().dict() #fragtuple: [[iso heaps]]
    #descentcache = mp.Manager().dict() #fragtuple group: (subformulas, massesandabundances)
    #group_fragmentation_partial = partial(group_fragmentation, dividingthreshold, positioncache, elementalcache, descentcache)
    ionlist = list(ions)
    ndict = {k: nfragmentcompositions[k] for k in ionlist if k in nfragmentcompositions}
    cdict = {k: cfragmentcompositions[k] for k in ionlist if k in cfragmentcompositions}
    frag_func_partial = partial(fragmentation_compositions, aminoacidcomposition, ndict, cdict)
    
    group_fragmentation_partial = partial(group_fragmentation, dividingthreshold, csvfilename, frag_func_partial)
    #group_fragmentation_partial = partial(group_fragmentation, dividingthreshold, csvfilename)

    with mp.Pool(nprocs) as pool:
        for count, divgroup in enumerate(dividedgroups):
            if count < 70:
                pool.apply_async(group_fragmentation_partial, args=(divgroup, count))
        #matchlist = list(itertools.chain(*pool.map(group_fragmentation_partial, divgroup)))
        pool.close()
        pool.join()
    
    #nt = time()
    #with open(csvfilename + '.' + str(count) + '.csv', 'a', newline='') as f:
    #    writer = csv.writer(f)
    #    writer.writerows(matchlist)
    #print(time() - nt, count, 'saved')
    

#8 divgroups
#600s for divthresh of 0.6 nn included
#1000s for 0.1
#there's no sum dists for the frag dist output...
#might need that
#sum dists attained and 867s for 0.1, an improvement!

#12 divgroups, 0.4 divthresh: 700s
#new order + combining: 704
#no combining just ordering: 744, SLOWER!

#initial: 754
#NEW FUNC: 477!!! even while browsing to then downloading 2 torrents
#organized now: 448 while fullscreen watching a video and doing other shit!!!

#msrun.reset()
#
#def mass_generation(scan):
#    if scan['ms level'] == 2:
#        ind = scan['index']
#        masses = scan['m/z array']
#        #intensities = scan['intensity array']
#        #neighbors[ind] = spatial.KDTree(masses[:,None])
#        #neighbors[ind] = masses.tolist()
#        return ind, masses.tolist()
#
#neighbors = {}
#for output in msrun.map(lambda scan: mass_generation(scan), processes=nprocs):
#    match output:
#        case None:
#            pass
#        case tuple:
#            ind, masses = output
#            neighbors[ind] = masses

#remove any csvs in this folder first
#for file in os.listdir(processinglocation):
#    if file.endswith('.matches.csv'):
#        os.remove('/'.join((processinglocation, file)))

#headers = ['sequence', 'subformula', 'ion', 'subindex', 'theoretical_abundance', 'mainindex', 'analyteid', 'scan', 'theoretical_mass', 'distribution_position']

group_processing_partial = partial(group_processing, csvfilename, dividingthreshold, elementsofprobabilityindices, probabilityorganizer, matchprobabilities, subformulasubindices, submatchsequences, linesbyscanbysubformula, analytesbydistribution, distributionsoflines, scanmasses)

print(len(dividedgroups), 'groups')
nt = time()
#for count, mergedsequences in enumerate(dividedgroups):
group_processing_partial(dividedgroups, ions)
print(time() - nt, '- total')
