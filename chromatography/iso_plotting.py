import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from colour import Color
from pyteomics import mzml
from time import time
import pandas as pd
import lmdb
import heapq
from functools import partial
import gc
import bisect
import concurrent.futures
import multiprocessing as mp
from multiprocessing.managers import BaseManager, DictProxy
from collections import Counter, defaultdict
from textwrap import wrap
from scipy import sparse, integrate, spatial, stats, special
from pandas.api.types import CategoricalDtype
from sklearn.neighbors import NearestNeighbors
from more_itertools import sort_together
from distinctipy import distinctipy as dp
import math
import sqlitedict as sq
import random
import itertools
import pickle
import sys
import os
import warnings
#warnings.filterwarnings("error")
np.set_printoptions(suppress=True)
#np.warnings.filterwarnings('ignore') #depricated i guess
np.testing.suppress_warnings(forwarding_rule='always')
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


#loaderloc = '/store/flowcharacterizations/round3/DDAs/fileprocessing/200901_fR_400.pickle'
#with open(loaderloc, "rb") as pick:
#    regions, trackedgroups, modeltracking, timearra, roundcutoff = pickle.load(pick)

nprocs = os.cpu_count()

mzmlfile = '/home/sfo/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
#mzmlfile = '/home/sfo/store/data/PXD035665/mzMLs/GradientAmount_HeLa_4ug_240min_R01.mzML'
#mzmlfile = '/home/sfo/store/data/PXD051214/mzMLs/JMM-6.mzML'

basefolder = '/'.join((mzmlfile.split('/')[:-2]))
basefile = mzmlfile.split('/')[-1].split('.mzML')[0]

processinglocation = '/'.join((basefolder, 'fileprocessing', basefile))
librarylocation = '/home/sfo/data/proteomics/fastas/search-db'
#librarylocation = '/home/sfo/data/proteomics/fastas/iodoacetamide-search'

proteomefile = '/home/sfo/data/proteomics/fastas/proteomes/Human_Homo_sapien.fasta'
proteome = proteomefile.split('/')[-1].split('.')[0]

proton = 1.007276554940804

msrun = mzml.MzML(mzmlfile, dtype=np.float64)

regionfile = '/'.join((processinglocation, 'regions.pickle'))
with open(regionfile, 'rb') as pick:
    regions = pickle.load(pick)
#regions as [minmass, maxmass, mintime, maxtime, # datapoints, peakarea, maxintensity, wmean, lineid]

loaderloc = '/'.join((processinglocation, 'trackedgroups.pickle'))
with open(loaderloc, 'rb') as pick:
    trackedgroups = pickle.load(pick)
#trackedgroups = {} #line: [[mass, rt, intensity]]

nodistfile = '/'.join((processinglocation, 'nodists.pickle'))
with open(nodistfile, 'rb') as pick:
    nodists = pickle.load(pick)
#keys of distributions not found in distributions

analytefile = '/'.join((processinglocation, 'analytefactors.pickle'))
with open(analytefile, 'rb') as pick:
    analytekeys, analytedistributions, analytesbydistribution, distributionsoflines, linesofdistributions, linesofanalytes = pickle.load(pick)
#analytekeys = defaultdict(dict) #analyte id: distid: charge, this can be one or multiple distributions based on whether it has multiple charge states
#analytedistributions = defaultdict(dict) #analyte id: ordered masses: AUC of merged isotopomers, weighted means [via intensity] across isotopomers from every charge state, isotopomer datapoints merged across each charge state - if there are any
#analytesbydistribution = {} #distid: analyte id

distributionchargesfile = '/'.join((processinglocation, 'distributioncharges.pickle'))
with open(distributionchargesfile, 'rb') as pick:
    distributioncharges = pickle.load(pick)
#distributioncharges = {} #distid: charge

linesofscansfile = '/'.join((processinglocation, 'linesofscans.pickle'))
with open(linesofscansfile, 'rb') as pick:
    global linesofscans
    linesofscans = pickle.load(pick)
#linesofscans = defaultdict(list) #ms2 scan index: [lineuids within window]

environment_partial = partial(lmdb.Environment, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=99999, max_spare_txns=1, lock=True)

with environment_partial(librarylocation) as env:
    #defaults = env.open_db('defaults'.encode())
    #with env.begin(write=False) as txn:
    #    with txn.cursor(defaults) as cursor:
    #        samplesize = int(cursor.get('samplesize'.encode()).decode())
    parameters = env.open_db('isofactors'.encode())
    with env.begin(write=False) as txn:
        with txn.cursor(parameters) as cursor:
            parameterbytes = cursor.get(proteome.encode())
            parameterdict = dict(eval(parameterbytes.decode()))
            subisomax = float(parameterdict['subisomax'])


rgblow = lambda: (np.random.uniform(low=0.6, high=1), np.random.uniform(low=0.9, high=1), np.random.uniform(low=0.7, high=1))
rgbhigh = lambda: (np.random.uniform(low=0.9, high=1), np.random.uniform(low=0, high=0.1), np.random.uniform(low=0.8, high=0.9))
deepset = lambda: (np.random.uniform(low=0.2, high=1), np.random.uniform(low=0.4, high=1), np.random.uniform(low=0.1, high=0.8))

##for color development
#colorset = lambda: (np.random.uniform(low=0.2, high=1), np.random.uniform(low=0.4, high=1), np.random.uniform(low=0.1, high=0.8))
#cn = 100
#x = np.arange(cn)
#y = x
#colorboard = [colorset() for _ in range(cn)]
#plt.scatter(x,y,c=colorboard)
#plt.show()

dividingthreshold = 0.01
subisotopomericdepth = 0.8
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

nfragmentcompositions = {}
nfragmentcompositions['a'] = Counter({'C': -1, 'O': -1})
nfragmentcompositions['b'] = Counter()
nfragmentcompositions['c'] = Counter({'N': 1, 'H': 3})

cfragmentcompositions = {}
cfragmentcompositions['x'] = Counter({'C': 1, 'O': 2})
cfragmentcompositions['y'] = Counter({'H': 2, 'O': 1})
cfragmentcompositions['z'] = Counter({'N': -1, 'H': -1, 'O': 1})
#cfragmentcompositions['z•'] = Counter({'N': -1, 'O': 1})

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
        
        ind = bisect.bisect(probabilityranking, r)
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
                        ind = bisect.bisect(probabilityranking, newratio)
                        probabilityranking.insert(ind, newratio)
                        multinomialpath.insert(ind, [newratio, *multipath])
                    else:
                        ind = bisect.bisect(probabilityranking, newratio)
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

def distribution_generation(dividingthreshold, atomiccomposition):
    elementalorganizer = {} #element: [[preheaps]]
    for e, acount in atomiccomposition.items():
        elementstring = e + str(acount)
        elementlist = individual_element_binomial_walk(dividingthreshold, e, acount)
        elementalorganizer[e] = elementlist.copy() #don't need to copy the insides
    subformulas, massesandabundances = descending_partial_products(dividingthreshold, elementalorganizer)
    return subformulas, massesandabundances

def distributions_of_seqs(originalseq):
    atomiccomposition = Counter()
    for aa in originalseq:
        atomiccomposition += aminoacidcomposition[aa]
    #no OH loss on last residue, no H lost on first residue
    atomiccomposition['H'] += 2
    atomiccomposition['O'] += 1
    formulastring = ''.join((''.join((k, str(v))) for k, v in atomiccomposition.items()))

    subformulas, massesandabundances = distribution_generation(dividingthreshold, atomiccomposition)
    subformulas = [i.decode() for i in subformulas]

    massgroups = defaultdict(list) #massnumber: [masses]
    intensitygroups = defaultdict(list) #massnumber: [abundances]
    masses, abundances = massesandabundances
    for n, s in enumerate(subformulas):
        massnumber = 0
        for ss in s.split(')')[:-1]:
            i1, i2 = map(int, ss[1:].split('('))
            massnumber += i1 * i2
        massgroups[massnumber].append(masses[n])
        intensitygroups[massnumber].append(abundances[n])

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
        #cumulatively adding intensities to determine which subisos can be added for ms2 searches
        subisocontinuation = True
        for n, (sm, sa) in sorted(enumerate(zip(m, a)), key=lambda x: -x[1][1]):
            weightedmass += sm * sa
            cumulativeabundance += sa
            cumpercent = cumulativeabundance / totalabundance
            if cumpercent <= subisotopomericdepth:
                subinds.append(n)
            elif subisocontinuation and cumpercent >= subisotopomericdepth:
                #subisodepth threshold breached, end it with this one
                subinds.append(n)
                subisocontinuation = False
            #    break
            #NO, stupid, you stopped the weighted mass means from being right by breaking here
        meansofmasses.append(weightedmass / totalabundance)
        sumsofabundances.append(totalabundance)
        subisodepthindices.append(tuple(subinds))
    sumabundancedist = np.array([meansofmasses, sumsofabundances])
    return sumabundancedist

def coordinate_generation(scan):
    if scan['ms level'] == 2:
        precursorinfo = scan['precursorList']['precursor'][0]
        selectionwindow = precursorinfo['isolationWindow']
        precmass = selectionwindow['isolation window target m/z'].real
        lowerbound = precmass - selectionwindow['isolation window lower offset'].real
        upperbound = precmass + selectionwindow['isolation window upper offset'].real
        #trainind references index of newtrain, -> get mass -> input to trackedma -> lineuid
        scanlist = scan['scanList']['scan'][0]
        #windowbounds = scanlist['scanWindowList']['scanWindow'][0]
        #lwbound = windowbounds['scan window lower limit'].real
        #uwbound = windowbounds['scan window upper limit'].real
        rt = scanlist['scan start time'].real
        scindex = int(scan['index'])
        #bounddict = {scindex: [rt, lwbound, uwbound]}
        #coordinates = [rt, precmass, lowerbound, upperbound, scindex]
        coordinates = [rt, lowerbound, upperbound, scindex]
        #return coordinates, bounddict
        return coordinates

precursorcoordinates = [] #[rt of previous ms1, lower mass bound, upper mass bound, ms2 scan index]
for output in msrun.map(lambda scan: coordinate_generation(scan), processes=nprocs):
    match output:
        case list():
            coords = output
            precursorcoordinates.append(coords)
precursorcoordinates = np.array(precursorcoordinates)

def arg_coord_rectangle_overlap(rec, coords):
    tops, bottoms, lefts, rights = coords.transpose()
    c1 = rec[2] < rights
    c2 = rec[3] > lefts
    c3 = rec[0] < bottoms
    c4 = rec[1] > tops
    overlaps = np.argwhere(c1 & c2 & c3 & c4)
    return overlaps.flatten()

def arg_coord_overlap(rec, coords):
    times, bottoms, tops, scans = coords.transpose()
    lowermass, uppermass, starttime, endtime = rec
    c1 = lowermass < tops
    c2 = uppermass > bottoms
    c3 = times < endtime
    c4 = times > starttime
    overlaps = np.argwhere(c1 & c2 & c3 & c4)
    return overlaps.flatten()

def iso_by_location(starttime, endtime, lowermass, uppermass):
    boundrec = [lowermass, uppermass, starttime, endtime]
    plotkeys = arg_coord_rectangle_overlap(boundrec, regions[:,:4]).tolist()
    
    lowermass, uppermass, starttime, endtime = boundrec
    timeadd = 0
    massadd = 0
    boundrec[0] -= massadd
    boundrec[1] += massadd
    boundrec[2] -= timeadd
    boundrec[3] += timeadd
    
    plotkeys = arg_coord_rectangle_overlap(boundrec, regions[:,:4]).tolist()
    solodists = defaultdict(lambda: defaultdict(list)) #charge: distid: lines
    nodistplotters = []
    for p in plotkeys:
        if p in distributionsoflines:
            dist = distributionsoflines[p]
            solodists[distributioncharges[dist]][dist].append(p) #not all lines of a dist are necessarily included
        else:
            nodistplotters.append(p)
    ngroups = 0
    for distcharge, linesbydistribution in solodists.items():
        for dist, lines in linesbydistribution.items():
            if len(lines) > 1:
                ngroups += 1
    cols = dp.get_colors(ngroups)
    
    cn = 0
    distrank = 0 #arbitrary ranking order on these
    fig, ax = plt.subplots(nrows=3, figsize=(10, 12), sharex=True)
    for distcharge, linesbydistribution in solodists.items():
        for dist, lines in linesbydistribution.items():
            if len(lines) > 1:
                col = cols[cn]
                low, high = rgblow(), rgbhigh()
                #cmap = colors.LinearSegmentedColormap.from_list('rgb', [low, high])
                cn += 1
                for line in lines:
                    a = np.array(trackedgroups[line]).transpose()
                    ax[2].scatter(a[0], a[1], marker='.', color=col, s=0.3, alpha=0.3)
                    ax[2].plot(a[0], a[1], '-', color=col, linewidth=0.2, alpha=0.8)
                    fw = np.ptp(a[1])
                    #ax[2].text(a[0][-1], a[1][-1], str(fw.round(3)), color='white', fontsize=4)
                fmasses = [regions[i,7] for i in lines]
                fints = [regions[i,5] for i in lines]
                ax[0].bar(fmasses, fints, color=col, alpha=0.5, width=0.01)
                for line in lines:
                    fx = regions[line,7]
                    fy = regions[line, 5]
                    ax[0].text(fx, fy + fy * 0.03, str(line), color='white', fontsize=8)
                ax[1].hlines(distrank, min(fmasses), max(fmasses), color=col, linewidth=0.6)
                for ln, line in enumerate(lines):
                    odd = False
                    if ln % 2:
                        odd = True
                    vert = regions[line, 7]
                    npoints = regions[line, 4]
                    ax[1].vlines(vert, distrank - 0.1, distrank + 0.1, color=col, linewidth=0.6)
                    if odd:
                        ax[1].text(vert, distrank + 0.2, str(round(vert, 2)), fontsize=6, ha='center', color='white')
                    else:
                        ax[1].text(vert, distrank - 0.4, str(round(vert, 2)), fontsize=6, ha='center', color='white')
                vi = np.sort(lines)
                if vi.size > 2:
                    vstacklines = np.stack((vi[:-1], vi[1:]), axis=1)
                    vstack = regions[vstacklines,7]
                    editspots = np.diff(vstack) < subisomax
                    if editspots.any():
                        ewheres = np.where(editspots)[0].tolist()
                        for ew in ewheres:
                            subpair = vstacklines[ew].tolist()
                            subints = [regions[i,5] for i in subpair]
                            winint = subints.index(max(subints))
                            winner = subpair[winint]
                            if ew > 0:
                                #edit 1 before ew
                                vstacklines[ew-1,1] = winner
                            if ew < len(vstack) - 1:
                                #edit 1 after ew
                                vstacklines[ew+1,0] = winner
                        vstacklines = np.delete(vstacklines, ewheres, axis=0)
                else:
                    vstack = vi.reshape(1, -1)
                vdiffs = np.diff(vstack)
                vflat = sorted(vstack.flatten().tolist())
                labelspots = np.mean(vstack, axis=1).tolist()
                for ls, lp in zip(labelspots, vstack.tolist()):
                    labeldiff = np.diff(lp)[0].round(4)
                    chargedist = (proton/distcharge - labeldiff).round(4)
                    lstring = ' ~ '.join((str(distcharge), str(labeldiff), str(chargedist)))
                    #ax[1].text(ls, distrank - 0.2, lstring, fontsize=4, ha='center', color='white')
                #heightcounter += 1
                distrank += 1
            else:
                #nodists, basically
                ndmasses = [regions[i,7] for i in lines]
                ndints = [regions[i,5] for i in lines]
                ax[0].bar(ndmasses, ndints, alpha=0.5, color='white', width=0.01)
                for nd in lines:
                    fx = regions[nd, 7]
                    fy = regions[nd, 5]
                    a = np.array(trackedgroups[nd]).transpose()
                    ax[0].text(fx, fy + fy * 0.03, str(nd), color='white', fontsize=4)
                    ax[2].scatter(a[0], a[1], marker='.', color='white', s=0.3, alpha=0.3)
                    ax[2].plot(a[0], a[1], '-', color='white', linewidth=0.2, alpha=0.8)
                    #ax[2].text(a[0][-1], a[1][-1], str(nd), color='white', fontsize=4)
    if nodistplotters:
        #these ones were < minpoints
        ndmasses = [regions[i,7] for i in nodistplotters]
        ndints = [regions[i,5] for i in nodistplotters]
        ax[0].bar(ndmasses, ndints, alpha=0.5, color='white', width=0.01)
        for nd in nodistplotters:
            fx = regions[nd, 7]
            fy = regions[nd, 5]
            a = np.array(trackedgroups[nd]).transpose()
            if fy > 0:
                #otherwise the plot goes off the wall down low
                ax[0].text(fx, fy + fy * 0.03, str(nd), color='white', fontsize=4)
            ax[2].scatter(a[0], a[1], marker='.', color='white', s=0.3, alpha=0.3)
            ax[2].plot(a[0], a[1], '-', color='white', linewidth=0.2, alpha=0.8)
            #ax[2].text(a[0][-1], a[1][-1], str(nd), color='white', fontsize=4)
    ax[0].set_yscale('log')
    ax[0].set_ylabel('intensity')
    ax[2].set_ylabel('minutes')
    ax[1].set_ylabel('distribution rank')
    ax[2].set_xlabel('m/z')
    for label in ax[2].get_xticklabels():
        #label.set_ha("right")
        label.set_rotation(-45)
    ncols = 6
    #ax[0].legend(title='Charge', loc='upper left', bbox_to_anchor=(0, 1.3 + (0.1 * cn / ncols)), ncol=ncols)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)
    plt.show()
    fig.clf()
    plt.close()

def location_plot(starttime, endtime, lowermass, uppermass):
    boundrec = [lowermass, uppermass, starttime, endtime]
    plotkeys = arg_coord_rectangle_overlap(boundrec, regions[:,:4]).tolist()
    
    fig, ax = plt.subplots(figsize=(6,4), facecolor='gray', sharex=True)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_facecolor('gray')
    for k in plotkeys:
        a = trackedgroups[k].transpose()
        low, high = rgblow(), rgbhigh()
        cmap = colors.LinearSegmentedColormap.from_list('rgb', [low, high])
        ax.scatter(a[1], a[0], marker='o', c=a[2], s=0.02, alpha=1, cmap=cmap)
        if a.size > 0:
            ax.plot(a[1], a[0], '-', color=low, linewidth=0.2, alpha=1)
    ax.set_xlabel('minutes')
    ax.set_ylabel('m/z')
    fig.tight_layout()
    plt.show()
    fig.clf()
    plt.close()

def location_by_line(line, massadd=2, timeadd=2):
    
    boundrec = [np.inf, 0, np.inf, 0]
    minmass = min(trackedgroups[line][:,0])
    maxmass = max(trackedgroups[line][:,0])
    mintime = min(trackedgroups[line][:,1])
    maxtime = max(trackedgroups[line][:,1])
    if minmass < boundrec[0]:
        boundrec[0] = minmass
    if maxmass > boundrec[1]:
        boundrec[1] = maxmass
    if mintime < boundrec[2]:
        boundrec[2] = mintime
    if maxmass > boundrec[3]:
        boundrec[3] = maxtime
    
    lowermass, uppermass, starttime, endtime = boundrec
    boundrec[0] -= massadd
    boundrec[1] += massadd
    boundrec[2] -= timeadd
    boundrec[3] += timeadd

    plotkeys = arg_coord_rectangle_overlap(boundrec, regions[:,:4]).tolist()
    ms2coords = precursorcoordinates[arg_coord_overlap(boundrec, precursorcoordinates)]
    
    fig, ax = plt.subplots(figsize=(6,4), facecolor='gray', sharex=True)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_facecolor('gray')
    for k in plotkeys:
        a = trackedgroups[k].transpose()
        low, high = rgblow(), rgbhigh()
        cmap = colors.LinearSegmentedColormap.from_list('rgb', [low, high])
        ax.scatter(a[1], a[0], marker='o', c=a[2], s=0.02, alpha=1, cmap=cmap)
        if a.size > 0:
            ax.plot(a[1], a[0], '-', color=low, linewidth=0.2, alpha=1)
    
    for m in ms2coords.tolist():
        if m[3] == m:
            col = 'cyan'
        else:
            col = 'darkorange'
        ax.vlines(m[0], m[1], m[2], color=col, alpha=0.4)
    
    ax.set_xlabel('minutes')
    ax.set_ylabel('m/z')

def limited_location_plot(starttime, endtime, lowermass, uppermass):
    boundrec = [lowermass, uppermass, starttime, endtime]
    plotkeys = arg_coord_rectangle_overlap(boundrec, regions[:,:4]).tolist()

    fig, ax = plt.subplots(figsize=(6,4), facecolor='gray', sharex=True)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_facecolor('gray')
    for k in plotkeys:
        a = trackedgroups[k].transpose()
        ainds = np.logical_and.reduce((a[1] <= endtime, a[1] >= starttime, a[0] >= lowermass, a[0] <= uppermass))
        a = a[:,ainds]
        low, high = rgblow(), rgbhigh()
        cmap = colors.LinearSegmentedColormap.from_list('rgb', [low, high])
        ax.scatter(a[1], a[0], marker='o', c=a[2], s=0.02, alpha=1, cmap=cmap)
        if a.size > 0:
            ax.plot(a[1], a[0], '-', color=low, linewidth=0.2, alpha=1)
    ax.set_xlabel('minutes')
    ax.set_ylabel('m/z')
    plt.close()

def location_by_ms2(ms2scan, massadd=2, timeadd=2):
    scanlines = linesofscans[ms2scan]
    
    boundrec = [np.inf, 0, np.inf, 0]
    for k in scanlines:
        minmass = min(trackedgroups[k][:,0])
        maxmass = max(trackedgroups[k][:,0])
        mintime = min(trackedgroups[k][:,1])
        maxtime = max(trackedgroups[k][:,1])
        if minmass < boundrec[0]:
            boundrec[0] = minmass
        if maxmass > boundrec[1]:
            boundrec[1] = maxmass
        if mintime < boundrec[2]:
            boundrec[2] = mintime
        if maxmass > boundrec[3]:
            boundrec[3] = maxtime
    
    lowermass, uppermass, starttime, endtime = boundrec
    boundrec[0] -= massadd
    boundrec[1] += massadd
    boundrec[2] -= timeadd
    boundrec[3] += timeadd

    plotkeys = arg_coord_rectangle_overlap(boundrec, regions[:,:4]).tolist()
    ms2coords = precursorcoordinates[arg_coord_overlap(boundrec, precursorcoordinates)]
    
    fig, ax = plt.subplots(figsize=(6,4), facecolor='gray', sharex=True)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_facecolor('gray')
    for k in plotkeys:
        a = trackedgroups[k].transpose()
        low, high = rgblow(), rgbhigh()
        cmap = colors.LinearSegmentedColormap.from_list('rgb', [low, high])
        ax.scatter(a[1], a[0], marker='o', c=a[2], s=0.02, alpha=1, cmap=cmap)
        if a.size > 0:
            ax.plot(a[1], a[0], '-', color=low, linewidth=0.2, alpha=1)
    
    for m in ms2coords.tolist():
        if m[3] == ms2scan:
            col = 'cyan'
        else:
            col = 'darkorange'
        ax.vlines(m[0], m[1], m[2], color=col, alpha=0.4)
    
    ax.set_xlabel('minutes')
    ax.set_ylabel('m/z')
    fig.tight_layout()
    plt.show()
    fig.clf()
    plt.close()

def seq_to_distribution_comparison(distid, seq):
    smasses, sabundances = distributions_of_seqs(seq)
    linesofdistribution = linesofdistributions[distid]
    distcharge = distributioncharges[distid]
    distrank = 0 #arbitrary ranking order on these
    fig, ax = plt.subplots(nrows=2, figsize=(10, 8), sharex=True)
    col = 'green'
    low, high = rgblow(), rgbhigh()
    cmap = colors.LinearSegmentedColormap.from_list('rgb', [low, high])
    for line in linesofdistribution:
        a = np.array(trackedgroups[line]).transpose()
        ax[1].plot(a[0], a[1], '-', color=low, linewidth=0.5, alpha=0.3)
        ax[1].scatter(a[0], a[1], marker='.', c=a[1], s=5, alpha=0.9, cmap=cmap)
        fw = np.ptp(a[1])
        fmasses = regions[line,7]
        fints = regions[line,5]
        ax[0].bar(fmasses, fints, color=col, alpha=0.5, width=0.01)
        fx = regions[line,7]
        fy = regions[line, 5]
        #ax[0].text(fx, fy + fy * 0.03, str(line), color='white', fontsize=8)
    #ax[0].set_yscale('log')
    t = ax[0].twinx()
    t.bar((smasses + (proton * distcharge)) / distcharge, sabundances, color='red', alpha=0.3, width=0.01, label=seq)
    ax[0].set_ylabel('intensity')
    ax[1].set_ylabel('minutes')
    ax[1].set_xlabel('m/z')
    for label in ax[1].get_xticklabels():
        #label.set_ha("right")
        label.set_rotation(-45)
    fig.subplots_adjust(hspace=0.05)
    plt.legend()
    plt.show()
    fig.clf()
    plt.close()
