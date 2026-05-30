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

#list of peptides to be searched via ms2
pepfragsfile = '/'.join((processinglocation, 'fragment.peptides.pickle'))
with open(pepfragsfile, 'rb') as pick:
    generatedsequences = pickle.load(pick)

isotopomerpositionsfile = '/'.join((processinglocation, 'isotopomersbypositions.pickle'))
with open(isotopomerpositionsfile, 'rb') as pick:
    isotopomerpositionsofanalytes = pickle.load(pick)
#isotopomerpositionsofanalytes = defaultdict(set) #analyteid: isotopopmer coordinate from max

spectrabyformulafile = '/'.join((processinglocation, 'spectrabyformula.pickle'))
with open(spectrabyformulafile, 'rb') as pick:
    spectrabyformula = pickle.load(pick)
#spectrabyformula = defaultdict(lambda: defaultdict(set)) #formula: analyteid: scan

#divisionfile = '/'.join((processinglocation, 'dividedformulas.pickle'))
#with open(divisionfile, 'rb') as pick:
#    dividedformulas = pickle.load(pick)

#divkeys = dividedformulas[0]
#spectrabyformula = {i:spectrabyformula[i] for i in divkeys}

seqsbyformula = {} #formula: [seqs]
abundances = {} #formula: [[masses], [intensities]]
abundanceformulas = {} #formula: subformulas
condensationcoordinates = {} #formula: [# isotopomers per proton-step]
subisodepthqualifiers = {} #formula: [[top n subisos of proton locations]]
encodedkeys = [i.encode() for i in spectrabyformula]
with environment_partial(librarylocation) as env:
    seqdb = '.'.join(('seqsbyformula', proteome))
    seqs = env.open_db(seqdb.encode())
    with env.begin(write=False) as txn:
        with txn.cursor(seqs) as cursor:
            for k, v in cursor:
                key = k.decode()
                value = eval(v.decode())
                seqsbyformula[key] = value
    aas = env.open_db('aminoacids'.encode())
    with env.begin(write=False) as txn:
        with txn.cursor(aas) as cursor:
            aaget = cursor.get(proteome.encode()).decode()
            aminoacidcomposition = eval(aaget)
    ddb = env.open_db('distributions.formulas'.encode())
    with env.begin(write=False) as txn:
        with txn.cursor(ddb) as cursor:
            for k, v in cursor.getmulti(encodedkeys):
                abundanceformulas[k.decode()] = eval(v.decode())
    condensationdb = env.open_db('distributions.condensation'.encode())
    with env.begin(write=False) as txn:
        with txn.cursor(condensationdb) as cursor:
            for k, v in cursor.getmulti(encodedkeys):
                condensationcoordinates[k.decode()] = np.frombuffer(v, dtype=int)
    subisoqualdb = env.open_db('distributions.subisoqualifiers'.encode())
    with env.begin(write=False) as txn:
        with txn.cursor(subisoqualdb) as cursor:
            for k, v in cursor.getmulti(encodedkeys):
                subisodepthqualifiers[k.decode()] = eval(v.decode())
    fulldb = env.open_db('distributions.full'.encode())
    with env.begin(write=False) as txn:
        with txn.cursor(fulldb) as cursor:
            for k, v in cursor.getmulti(encodedkeys):
                out = np.frombuffer(v)
                out = out.reshape(2, out.size//2)
                abundances[k.decode()] = out
    #defaults = env.open_db('defaults'.encode())
    #with env.begin(write=False) as txn:
    #    with txn.cursor(defaults) as cursor:
    #        minimumabundance = float(cursor.get('minimumabundance'.encode()).decode())

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

staticmods = {
        #'C': {'H': 5, 'C': 3, 'N':1, 'O':1}, #acrylamide
        }

for aa, sad in staticmods.items():
    for saa, sav in sad.items():
        aminoacidcomposition[aa][saa] += sav

#need to modify this organization to allow more than one type of mod on the same AA
variablemods = {
        'C': {'H': 5, 'C': 3, 'N':1, 'O':1}, #acrylamide
        }

#there should be more than enough in the alphabet for any reasonable number of variable mods I suppose
newmods = {} #new AA letter: its atomic composition
modifiers = defaultdict(set) #existing AA: [new AA letters]
modoriginals = {} #new AA letter: existing AA
variablecharacters = string.ascii_lowercase
for vn, (va, vad) in enumerate(variablemods.items()):
    representativecharacter = variablecharacters[vn]
    modifiers[va].add(representativecharacter)
    newmods[representativecharacter] = vad
    modoriginals[representativecharacter] = va
    aminoacidcomposition[representativecharacter] = aminoacidcomposition[va].copy()
    for vaa, vav in vad.items():
        aminoacidcomposition[representativecharacter][vaa] += vav

#def fragmentation_compositions(seq):
#    fragments = {}
#    fragcomp = Counter()
#    for n, aa in enumerate(seq[:-1]): #n-term
#        fragcomp += aminoacidcomposition[aa]
#        for ion, modcomp in nfragmentcompositions.items():
#            fragments[ion + str(n + 1)] = fragcomp + modcomp
#    fragcomp = Counter()
#    for n, aa in enumerate(seq[::-1][:-1]): #c-term
#        fragcomp += aminoacidcomposition[aa]
#        for ion, modcomp in cfragmentcompositions.items():
#            fragments[ion + str(n + 1)] = fragcomp + modcomp
#    return fragments

#chatgpt actually gave me an optimized version and it works
def fragmentation_compositions(seq):
    fragments = {}

    # Calculate the compositions of the n-term fragments
    fragcomp_n = {}
    for n, aa in enumerate(seq[:-1]):  
        aa_composition = aminoacidcomposition[aa]
        for k in aa_composition:
            fragcomp_n[k] = fragcomp_n.get(k, 0) + aa_composition.get(k, 0)
        for ion, modcomp in nfragmentcompositions.items():
            fragment_composition = fragcomp_n.copy()
            for k in modcomp:
                fragment_composition[k] = fragment_composition.get(k, 0) + modcomp.get(k, 0)
            fragments[ion + str(n + 1)] = fragment_composition

    # Calculate the compositions of the c-term fragments
    fragcomp_c = {}
    for n, aa in enumerate(seq[::-1][:-1]): 
        aa_composition = aminoacidcomposition[aa]
        for k in aa_composition:
            fragcomp_c[k] = fragcomp_c.get(k, 0) + aa_composition.get(k, 0)
        for ion, modcomp in cfragmentcompositions.items():
            fragment_composition = fragcomp_c.copy()
            for k in modcomp:
                fragment_composition[k] = fragment_composition.get(k, 0) + modcomp.get(k, 0)
            fragments[ion + str(n + 1)] = fragment_composition

    return fragments

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

#both of the fragment distribution-related functions have been fact-checked via brute-force, they're LEGIT
def fragment_descending_partial_products(dividingthreshold, elementalorganizer, fragmentpositions):
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
                    formula += f'{fragmentpositions[e][n]}({c})'
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
                        formula += f'{fragmentpositions[se][n]}({c})'
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
                                    sef += f'{fragmentpositions[se][n]}({c})'
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
                                    seformula += f'{fragmentpositions[ie][n]}({c})'
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
                                    seformula += f'{fragmentpositions[ie][n]}({c})'
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
    subformulas = subformulas[massesandabundances[0].argsort()].tolist()
    massesandabundances = massesandabundances[:,massesandabundances[0].argsort()]
    return subformulas, massesandabundances

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

#nfragmentcompositions2 = {}
#nfragmentcompositions2['a'] = {'C': -1, 'O': -1}
#nfragmentcompositions2['b'] = {}
#nfragmentcompositions2['c'] = {'N': 1, 'H': 3}
#
#cfragmentcompositions2 = {}
#cfragmentcompositions2['x'] = {'C': 1, 'O': 2}
#cfragmentcompositions2['y'] = {'H': 2, 'O': 1}
#cfragmentcompositions2['z'] = {'N': -1, 'H': -1, 'O': 1}
#
##this function is incorrect, its off by 1 somewhere, but it was slower than using Counters
#def fragmentation_compositions2(nfragmentcompositions, cfragmentcompositions, aminoacidcomposition, seq):
#    #maybe instead of doing counter arithmatic, you can just pre-program the +/- of each element for each ion in order, it would probably be faster
#    fragments = {}
#    fragcomp = {}
#    for n, aa in enumerate(seq[:-1]): #n-term
#        for e, c in aminoacidcomposition[aa].items():
#            if e in fragcomp:
#                fragcomp[e] += c
#            else:
#                fragcomp[e] = c
#        #b -> no mod
#        #a -> -1C -1O
#        #c -> +1C +1O +1N +3H
#        #undo
#            lastdict = fragcomp
#            for ion, modcomp in nfragmentcompositions.items():
#                key = ion + str(n + 1)
#                #fragments[key] = {}
#                fragments[key] = lastdict.copy()
#                for e, c in modcomp.items():
#                    if e in fragments[key]:
#                        fragments[key][e] + c
#                    else:
#                        fragments[key][e] = c
#                lastdict = fragments[key]
#                #fragments[ion + str(n + 1)] = fragcomp + modcomp
#    fragcomp = {}
#    for n, aa in enumerate(seq[::-1][:-1]): #c-term
#        for e, c in aminoacidcomposition[aa].items():
#            if e in fragcomp:
#                fragcomp[e] += c
#            else:
#                fragcomp[e] = c
#            lastdict = fragcomp
#            for ion, modcomp in cfragmentcompositions.items():
#                key = ion + str(n + 1)
#                #fragments[key] = {}
#                fragments[key] = lastdict.copy()
#                for e, c in modcomp.items():
#                    if e in fragments[key]:
#                        fragments[key][e] + c
#                    else:
#                        fragments[key][e] = c
#                lastdict = fragments[key]
#                #fragments[ion + str(n + 1)] = fragcomp + modcomp
#    return fragments

#def fragmentation_formulas(nfragmentcompositions, cfragmentcompositions, aminoacidcomposition, seq):
#    fraglist = []
#    fragcomp = Counter()
#    for n, aa in enumerate(seq[:-1]): #n-term
#        fragcomp += aminoacidcomposition[aa]
#        for ion, modcomp in nfragmentcompositions.items():
#            #ionformula = ''
#            ionformula = ion + str(n + 1) + '-'
#            ilen = len(ionformula)
#            for element, count in fragcomp.items():
#                if element in modcomp:
#                    newcount = int(count) + modcomp[element]
#                    if newcount > 0:
#                        ionformula += element
#                        ionformula += str(newcount)
#                else:
#                    ionformula += element
#                    ionformula += str(count)
#            fraglist.append(ionformula)
#    fragcomp = Counter()
#    for n, aa in enumerate(seq[::-1][:-1]): #c-term
#        fragcomp += aminoacidcomposition[aa]
#        for ion, modcomp in cfragmentcompositions.items():
#            #ionformula = ''
#            ionformula = ion + str(n + 1) + '-'
#            ilen = len(ionformula)
#            for element, count in fragcomp.items():
#                if element in modcomp:
#                    newcount = int(count) + modcomp[element]
#                    if newcount > 0:
#                        ionformula += element
#                        ionformula += str(newcount)
#                else:
#                    ionformula += element
#                    ionformula += str(count)
#            fraglist.append(ionformula)
#    return fraglist

#optimized below
#def max_estimation(c, eprobs):
#    countsum = 0
#    isoprobs = {}
#    arraycounts = []
#    arrayisotopes = []
#    while eprobs:
#        p, iso = heapq.heappop(eprobs)
#        p *= -1
#        isoprobs[iso] = p
#        roundestimate = round(c * p)
#        arraycounts.append(roundestimate)
#        arrayisotopes.append(iso)
#        countsum += roundestimate
#    testcomps = [arraycounts.copy() for _ in arraycounts]
#    if countsum < c:
#        for n, t in enumerate(testcomps):
#            t[n] += 1
#    elif countsum > c:
#        for n, t in enumerate(testcomps):
#            if t[n] > 0:
#                t[n] -= 1
#            else:
#                t[0] -= 1
#                t[n] += 1
#    else:
#        for n, t in enumerate(testcomps[:-1]):
#            t[0] -= 1
#            t[n+1] += 1
#    
#    probvals = []
#    #massvals = []
#    probvectors = []
#    for comp in testcomps:
#        pn = 0
#        newprob = 1
#        #newmass = 0
#        for n, count in enumerate(comp):
#            newprob *= isoprobs[arrayisotopes[n]] ** count
#            #newmass += elementalmasses[arrayisotopes[n]] * count
#            if n > 0:
#                newprob *= math.comb(c-pn, count)
#                pn += count
#        probvals.append(newprob)
#        #massvals.append(newmass)
#        probvectors.append(comp.copy())
#    maxprob = max(probvals)
#    maxind = probvals.index(maxprob)
#    #maxmass = massvals[maxind]
#    maxmass = 0
#    for n, count in enumerate(probvectors[maxind]):
#        maxmass += elementalmasses[arrayisotopes[n]] * count
#    #maxvec = probvectors[maxind]
#    #print(arrayisotopes)
#    #print(maxvec)
#    #print(round(maxmass, 5), round(maxprob, 5))
#    #print('~')
#    return maxmass, maxprob

#term_partial = partial(term_fragmentation, nfragmentcompositions, aminoacidcomposition, reverse=False)
#
#nt = time()
#with mp.Pool(nprocs) as pool:
#    fragmentsbyseq = dict(pool.map(term_partial, generatedsequences))
#print(time() - nt, 'n-term fragments generated')

#seq = 'AQVAFKKMVQGVLQFAVCDTAAAGQLVK'

#fragmentation_partial = partial(fragmentation_compositions, nfragmentcompositions, cfragmentcompositions, aminoacidcomposition)
#fragment_formula_partial = partial(fragmentation_formulas, nfragmentcompositions, cfragmentcompositions, aminoacidcomposition)


#i'm going to start off assuming that the same fragments should have the same relative prevalence regardless of isotopic location or composition
#so it would be a good idea to base a metric off finding the most true relative ratios of fragments across ions 
#so then make a way to visualize a line, its ms2 hit points, other dists that aren't currently described by the plots but are in other hits, and the consistency of fragment ratios of different isotopomers
#also make a way to visualize two potential sequence matches next to each other
#on the technical side, facilitate the matching process

msrun.reset()
ppmtol = 25
ncap = 1000

allsamples = set()
for sample in spectrabyformula.values():
    for sids in sample.values():
        allsamples.update(sids)

neighbors = {}
allsamples = tuple(allsamples)
for sid in allsamples:
    scan = msrun[sid]
    masses = scan['m/z array']
    intensities = scan['intensity array']
    neighbors[sid] = spatial.KDTree(masses[:,None])

#n = 0
#total = 1000
#testspecs = {}
#for formula, samples in reversed(spectrabyformula.items()):
#    testspecs[formula] = samples
#    n += 1
#    if n > total:
#        break
#
#sc = Counter({k:len(list(v.values())[0]) for k, v in spectrabyformula.items()})
#testspecs = {i:spectrabyformula[i] for i in list(zip(*sc.most_common(total)))[0]}


print('started')
nt = time()
#i think isotopomerpositionsofanalytes could use formulas as a key instead, confirm all the lists of things sharing a formula are the same thing
#spectrabyseq = {} #seq: ion-subindex: [[masses], [frags]]
#spectrabyformula = defaultdict(lambda: defaultdict(set)) #formula: analyteid: [scans]
#isotopomerpositionsofanalytes = defaultdict(set) #analyteid: isotopopmer coordinate from max
#fragmatchlist = []
#cachecounts = Counter()
#subcounts = Counter()
#fragcounts = Counter()
#subformulasbyanalyteid = defaultdict(lambda: Counter()) #analyteid: subformula: count


mainindexbysubindex = defaultdict(list) #sub match index: main match index
scansbymainindex = {} #main match index: [scans]

probtracker = {} #prob string: prob index
#fragtracker = {} #element frag comp: frag index
#matchtracker = {} #sub match index: main match index

probabilityorganizer = defaultdict(dict) #prob index: iso: prob
#^there's still some redundancy in here, 99%+ of it is carbon. the reason is that two different subformula compositions can form the same ratios/probabilities, its not a big deal tbh, the dict is less than 2000 in length
#fragmentorganizer = {} #frag index: 'element' + 'count'
#matchorganizer = defaultdict(list) #main match index: [sub match indices]

#matchbase = {} #sub match index: [[seq, ion, analyteid]]
#matchfragments = defaultdict(list) #sub match index: [frag indices] -> make into tuple?
matchprobabilities = defaultdict(list) #subformula: [prob indices] #subformula here instead of match index bc the prob comp is tied to subformulas

#mainindexformulas = {} #main index: main formula
subformulasubindices = defaultdict(list) #subformula: [sub match indices]
submatchsequences = {} #submatchindex: sequence
submatchsubformulas = {} #submatchindex: subformula
elementsofprobabilityindices = {} #prob index: e

#anything with a sub match index key is way too enormous to hold on disk
#link mainindex to formula -> which links it to seq
#seq can generate the fragments
#maybe mainindexbysubindex will lead to a seq instead?

#fragredundancy = Counter()

probindex = 0
#fragindex = 0
submatchindex = 0
mainmatchindex = 0
for formula, samples in list(spectrabyformula.items()):
    qualifiers = subisodepthqualifiers[formula]
    conlengths = condensationcoordinates[formula]
    conends = conlengths.cumsum()
    constarts = conends - conlengths
    subformulas = [i.decode() for i in abundanceformulas[formula]]
    massesandintensities = abundances[formula]
    #i'm taking full when i should only take max, this can ease you out of using condensationcoordinates and positions too
    intensities = massesandintensities[1]
    sumintensities = np.array([intensities[s:e].sum() for s, e in zip(constarts.tolist(), conends.tolist())])
    maxintensityindex = sumintensities.argmax()
    positions = set()
    for analyteid, sids in samples.items():
        positions.update(isotopomerpositionsofanalytes[analyteid]) #needs to be replaced so theoretical positions are matching to experimental -> distributionpairing
        scansbymainindex[mainmatchindex] = sids
    #mainindexformulas[mainmatchindex] = formula
    positions = [i + maxintensityindex for i in positions]
    for seq in seqsbyformula[formula]:
        #fragions = []
        #fragints = []
        #fragmasses = []
        #fragindices = []
        #fragpositions = []
        #fragments = fragmentation_compositions(seq)
        #for ion, counts in fragments.items():
        #    fragstring = ''.join((''.join((k, str(v))) for k, v in counts.items()))
        #    fragredundancy[fragstring] += 1
        for p in positions:
            try:
                bi = constarts[p]
                #ei = conends[p]
            except IndexError:
                #an example of this came up where an isotopomer of a distribution that had an MS2 scan wasn't in the theoretical distribution, the distributions didn't match that well, and I think I'll just ignore it here for now and work with what I can
                continue
            subquals = qualifiers[p]
            for sq in subquals:
                subindex = bi + sq
                sformula = subformulas[subindex]
                #formulaindex = '-'.join((seq, sformula))
                mainindexbysubindex[submatchindex] = mainmatchindex
                submatchsubformulas[submatchindex] = sformula
                subformulasubindices[sformula].append(submatchindex)
                submatchsequences[submatchindex] = seq
                submatchindex += 1
                #subformulasbyanalyteid[analyteid][sformula] += 1
                #subcounts[sformula] += 1
                #setting up subformula-specific probabilities
                fragint = intensities[subindex] #i need to record frag intensities here
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
                #fragprobs = {}
                if sformula not in matchprobabilities:
                    for e, v in competitors.items():
                        #probstring = ''.join((''.join((iso, '(', str(c), ')')) for iso, c in competitors[e].items()))
                        isoprobs = {}
                        if e in competing:
                            for iso, c in v.items():
                                prob = c / isosums[e]
                                #fragprobs[iso] = prob
                                #probabilityorganizer[probindex][iso] = prob
                                isoprobs[iso] = prob
                                #probtracker[probstring] = probindex
                            #matchprobabilities[sformula].append(probindex)
                            #probindex += 1
                            #probtuple = tuple(isoprobs.items())
                            probstring = '/'.join(('/'.join((k, str(v))) for k, v in isoprobs.items()))
                            if probstring in probtracker:
                                #probabilityorganizer[foundprobindex][iso] = prob #already there
                                foundprobindex = probtracker[probstring]
                                matchprobabilities[sformula].append(foundprobindex)
                            else:
                                probtracker[probstring] = probindex
                                probabilityorganizer[probindex] = isoprobs
                                matchprobabilities[sformula].append(probindex)
                                elementsofprobabilityindices[probindex] = e
                                probindex += 1
                        else:
                            for iso in v:
                                #don't need to make a new index for every time this is hit
                                #prob = c / isosums[e]
                                #fragprobs[iso] = prob
                                #probabilityorganizer[e][probindex][iso] = 1
                                #probabilityorganizer[e][iso] = 1 #some unneeded redundancy of action here still but whatever
                                isoprobs[iso] = 1
                                #probtracker[probstring] = probindex
                                #probtracker[probstring] = e
                                #matchprobabilities[sformula].append(e)
                                #probindex += 1
                            if e not in probabilityorganizer:
                                #probstring = tuple(isoprobs.items())
                                probstring = '/'.join(('/'.join((k, str(v))) for k, v in isoprobs.items()))
                                probtracker[probstring] = e
                                probabilityorganizer[e] = isoprobs
                                elementsofprobabilityindices[e] = e
                            matchprobabilities[sformula].append(e)
                #testing for the use of a cache here
                #elementbase = defaultdict(list) #e: [iso, count]
                #for iso, c in fragprobs.items():
                #    elementbase[iso[0]].append((iso, c))
                #for ion, fragcomp in fragments.items():
                #    matchbase[submatchindex] = [seq, ion, analyteid]
                #    for e, c in fragcomp.items():
                #        if c > 0:
                #            fragstring = e + str(c)
                #            if fragstring in fragtracker:
                #                foundfragindex = fragtracker[fragstring]
                #                matchfragments[submatchindex].append(foundfragindex)
                #            else:
                #                fragtracker[fragstring] = fragindex
                #                fragmentorganizer[fragindex] = fragstring
                #                matchfragments[submatchindex].append(fragindex)
                #                fragindex += 1
                #        #cachecounts[(tuple(sorted(elementbase[e])), c)] += 1
                #        #fragmentorganizer[e][fragindex] = c
                #    matchtracker[submatchindex] = mainmatchindex
                #    matchorganizer[mainmatchindex].append(submatchindex)
                #    submatchindex += 1
                    #fragformula = ''.join(((''.join((k, str(v))) for k, v in fragcomp.items())))
                    #fragcounts[fragformula] += 1
                    #fm, fi = max_fragment(fragprobs, fragcomp, fragint)
                    #fragmasses.append(fm)
                    #fragints.append(fi)
                    #fragpositions.append(p)
                    #fragindices.append(subindex)
                    #fragions.append(ion)
        #fragmasses = np.array(fragmasses)[:,None]
        #fragtol = (fragmasses / 1000000 * ppmtol).flatten()
        #for analyteid, sids in samples.items():
        #    for sid in sids:
        #        matches = neighbors[sid].query_ball_point(fragmasses, fragtol)
        #        for n, m in enumerate(matches.tolist()):
        #            for sm in m:
        #                mlist = (seq, sm, fragions[n], fragmasses[n][0], fi, sid, analyteid, fragpositions[n], fragindices[n])
        #                fragmatchlist.append(mlist)
        #within each model's results, remove redundant matches? should i? what if two frags match to the same ion with good result, its ambiguous, no?
    mainmatchindex += 1
print(time() - nt, 'linear')

mergablesequences = []
for smi, sformula in submatchsubformulas.items():
    seq = submatchsequences[smi]
    mergablesequences.append([seq, sformula])

#i'm merging these here so i don't have to generate the same set of fragment ions for a single sequence more than once later
mergedsequences = list(map(tuple, intersection_merge(mergablesequences))) #could you do this with submatchformulas?

#the memory capacity of the below function will be dependent upon the looseness of ms1 fittings, which might then be again refined by allowing what's been seen 
nt = time()

positioncache = {} #fragtuple: iso: position
elementalcache = {} #fragtuple: [[iso heaps]]
descentcache = {} #fragtuple group: (subformulas, massesandabundances)

fragcounts = Counter()
elementcounts = Counter()

#formulachunking can now be for mergedsequences because the cache makes me run out of memory
#i'm switching the development of this file and that into a new one to create mergedsequences
#then a new file will generate fragment ions
ctn = 0 #23% mem start -> 33% end u guess?
for group in mergedsequences[:100000]:
    ctn += 1
    groupseqs = []
    groupsubformulas = []
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
        sequences = list(set(submatchsequences[i] for i in subindices))
        for seq in sequences:
            for ion, fragcomp in fragments[seq].items():
                elementalorganizer = {} #element: [[iso heaps]]
                fragpositions = {} #element: position: iso
                #fragtuples = () #fucking with tuples was way slower for some reason despite actually making the strings taking longer. tuples were also massive memory overhead
                fragstrings = ''
                for e, c in fragcomp.items():
                    fragprobs = probindices[e]
                    #fragtuple = (tuple(probindices[e].items()), c)
                    fragstring = str(c) + '/' + '/'.join(('/'.join((k, str(v))) for k, v in probindices[e].items()))
                    elementcounts[fragstring] += 1
                    #fragtuples += fragtuple #fragment dict are consistently ordered so this should work fine
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
                        fragpositions[e] = positions
                    else: #no need for cache, only 1 iso
                        #iso, c = list(fragprobs.items())[0]
                        iso = list(fragprobs)[0]
                        elementalorganizer[e] = [[-1, 1, elementalmasses[iso]*c, e, [c]]]
                        fragpositions[e] = {0: iso}
                try:
                    subformulas, massesandabundances = descentcache[fragstrings]
                except KeyError: #not done prior
                    subformulas, massesandabundances = fragment_descending_partial_products(dividingthreshold, elementalorganizer, fragpositions)
                    descentcache[fragstrings] = subformulas, massesandabundances
                fragcounts[fragstrings] += 1
        #mainindices = list(set(mainindexbysubindex[i] for i in subindices)) #its always length 1 as far as i can see
        #for ind in mainindices:
        #    scan = scansbymainindex[i]
print(time() - nt, 'frag dists')

#nt = time()
##i think isotopomerpositionsofanalytes could use formulas as a key instead, confirm all the lists of things sharing a formula are the same thing
##spectrabyseq = {} #seq: ion-subindex: [[masses], [frags]]
##spectrabyformula = defaultdict(lambda: defaultdict(set)) #formula: analyteid: [scans]
##isotopomerpositionsofanalytes = defaultdict(set) #analyteid: isotopopmer coordinate from max
#fragmatchlist = []
#cachecounts = Counter()
#subcounts = Counter()
#fragcounts = Counter()
#subformulasbyanalyteid = defaultdict(lambda: Counter()) #analyteid: subformula: count
#for formula, samples in list(spectrabyformula.items())[:10000]:
#    qualifiers = subisodepthqualifiers[formula]
#    conlengths = condensationcoordinates[formula]
#    conends = conlengths.cumsum()
#    constarts = conends - conlengths
#    subformulas = [i.decode() for i in abundanceformulas[formula]]
#    massesandintensities = abundances[formula]
#    #i'm taking full when i should only take max, this can ease you out of using condensationcoordinates and positions too
#    intensities = massesandintensities[1]
#    sumintensities = np.array([intensities[s:e].sum() for s, e in zip(constarts.tolist(), conends.tolist())])
#    maxintensityindex = sumintensities.argmax()
#    positions = set()
#    for analyteid, sids in samples.items():
#        positions.update(isotopomerpositionsofanalytes[analyteid])
#    positions = [i + maxintensityindex for i in positions]
#    for seq in seqsbyformula[formula]:
#        fragions = []
#        fragints = []
#        fragmasses = []
#        fragindices = []
#        fragpositions = []
#        fragments = fragmentation_compositions(seq)
#        for p in positions:
#            try:
#                bi = constarts[p]
#                #ei = conends[p]
#            except IndexError:
#                #an example of this came up where an isotopomer of a distribution that had an MS2 scan wasn't in the theoretical distribution, the distributions didn't match that well, and I think I'll just ignore it here for now and work with what I can
#                continue
#            subquals = qualifiers[p]
#            for sq in subquals:
#                subindex = bi + sq
#                sformula = subformulas[subindex]
#                subformulasbyanalyteid[analyteid][sformula] += 1
#                subcounts[sformula] += 1
#                #setting up subformula-specific probabilities
#                fragint = intensities[subindex]
#                isocounts = set()
#                competing = set()
#                competitors = {}
#                isosums = {}
#                for ss in sformula.split(')')[:-1]:
#                    iso, c = ss.split('(')
#                    c = int(c)
#                    e = iso[0]
#                    if e in isocounts:
#                        competing.add(e)
#                        competitors[e][iso] = c
#                        isosums[e] += c
#                    else:
#                        isocounts.add(e)
#                        competitors[e] = {iso: c}
#                        isosums[e] = c
#                fragprobs = {}
#                for e, v in competitors.items():
#                    if e in competing:
#                        for iso, c in competitors[e].items():
#                            fragprobs[iso] = c / isosums[e]
#                    else:
#                        for iso in v:
#                            fragprobs[iso] = 1
#                #testing for the use of a cache here
#                elementbase = defaultdict(list) #e: [iso, count]
#                for iso, c in fragprobs.items():
#                    elementbase[iso[0]].append((iso, c))
#                for ion, fragcomp in fragments.items():
#                    for e, c in fragcomp.items():
#                        cachecounts[(tuple(sorted(elementbase[e])), c)] += 1
#                    fragformula = ''.join(((''.join((k, str(v))) for k, v in fragcomp.items())))
#                    fragcounts[fragformula] += 1
#                    fm, fi = max_fragment(fragprobs, fragcomp, fragint)
#                    fragmasses.append(fm)
#                    fragints.append(fi)
#                    fragpositions.append(p)
#                    fragindices.append(subindex)
#                    fragions.append(ion)
#        fragmasses = np.array(fragmasses)[:,None]
#        fragtol = (fragmasses / 1000000 * ppmtol).flatten()
#        for analyteid, sids in samples.items():
#            for sid in sids:
#                matches = neighbors[sid].query_ball_point(fragmasses, fragtol)
#                for n, m in enumerate(matches.tolist()):
#                    for sm in m:
#                        mlist = (seq, sm, fragions[n], fragmasses[n][0], fi, sid, analyteid, fragpositions[n], fragindices[n])
#                        fragmatchlist.append(mlist)
#        #within each model's results, remove redundant matches? should i? what if two frags match to the same ion with good result, its ambiguous, no?
#print(time() - nt, 'linear')







#df = pl.DataFrame(fragmatchlist)
#probably won't need polars

def frag_match(formula, samples):
    fragmatches = []
    qualifiers = subisodepthqualifiers[formula]
    conlengths = condensationcoordinates[formula]
    conends = conlengths.cumsum()
    constarts = conends - conlengths
    subformulas = [i.decode() for i in abundanceformulas[formula]]
    massesandintensities = abundances[formula]
    intensities = massesandintensities[1]
    sumintensities = np.array([intensities[s:e].sum() for s, e in zip(constarts.tolist(), conends.tolist())])
    maxintensityindex = sumintensities.argmax()
    positions = set()
    for analyteid, sids in samples.items():
        positions.update(isotopomerpositionsofanalytes[analyteid])
    positions = [i + maxintensityindex for i in positions]
    for seq in seqsbyformula[formula]:
        fragions = []
        fragints = []
        fragmasses = []
        fragindices = []
        fragpositions = []
        fragments = fragmentation_compositions(seq)
        for p in positions:
            try:
                bi = constarts[p]
                #ei = conends[p]
            except IndexError:
                #an example of this came up where an isotopomer of a distribution that had an MS2 scan wasn't in the theoretical distribution, the distributions didn't match that well, and I think I'll just ignore it here for now and work with what I can
                continue
            subquals = qualifiers[p]
            for sq in subquals:
                subindex = bi + sq
                sformula = subformulas[subindex]
                #setting up subformula-specific probabilities
                fragint = intensities[subindex]
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
                fragprobs = {}
                for e, v in competitors.items():
                    if e in competing:
                        for iso, c in competitors[e].items():
                            fragprobs[iso] = c / isosums[e]
                    else:
                        for iso in v:
                            fragprobs[iso] = 1
                for ion, fragcomp in fragments.items():
                    fm, fi = max_fragment(fragprobs, fragcomp, fragint)
                    fragmasses.append(fm)
                    fragints.append(fi)
                    fragpositions.append(p)
                    fragindices.append(subindex)
                    fragions.append(ion)
        fragmasses = np.array(fragmasses)[:,None] + proton #assuming 1+ for now
        fragtol = (fragmasses / 1000000 * ppmtol).flatten()
        for sid in samples[analyteid]:
            #dists, inds = nn.query(fragmasses, workers=1)
            matches = neighbors[sid].query_ball_point(fragmasses, fragtol)
        for analyteid, sids in samples.items():
            for sid in sids:
                matches = neighbors[sid].query_ball_point(fragmasses, fragtol)
                for n, m in enumerate(matches.tolist()):
                    for sm in m:
                        mlist = (seq, sm, fragions[n], fragmasses[n][0], fi, sid, analyteid, fragpositions[n], fragindices[n])
                        fragmatches.append(mlist)
    return fragmatches

#nt = time()
#fragmatchlist = []
#with mp.Pool(nprocs) as pool:
#    for ol in pool.starmap(frag_match, spectrabyformula.items()):
#        fragmatchlist.extend(ol)
#print(time() - nt, 'multiprocessed')
#42 mem start

#def frag_csv(file, rows):
#    with open(file, 'a', newline='') as f:
#        writer = csv.writer(f)
#        for row in rows:
#            line = ','.join((map(str, row))) + '\n'
#            writer.writerow(line)
#
#filename = '/'.join((processinglocation, 'fragment.matches.csv'))
#if os.path.isfile(filename):
#    os.remove(filename)
#
#with open(filename, 'wb') as pick:
#    pickle.dump(fragmatchlist, pick)

#nt = time()
#fragmatchlist = []
#with mp.Pool(nprocs) as pool:
#    results = []
#    for k, v in spectrabyformula.items():
#        result = pool.apply_async(frag_match, args=(k, v))
#        results.append(result)
#    for result in results:
#        ol = result.get()
#        fragmatchlist.extend(ol)
#        if psutil.virtual_memory().percent > 80:
#            #asyncio.run(write_to_csv(filename, fragmatchlist))
#            #loop = asyncio.get_event_loop()
#            #loop.run_until_complete(write_to_csv(filename, fragmatchlist))
#            await write_to_csv(filename, fragmatchlist)
#            fragmatchlist.clear()
#            gc.collect()
#print(time() - nt, 'apply_async\'d')

#def frag_match(constarts, subformulas, qualifiers, intensities, positions, fragments, samples, seq):
#    outlist = []
#    fragions = []
#    fragints = []
#    fragmasses = []
#    fragindices = []
#    fragpositions = []
#    for p in positions:
#        try:
#            bi = constarts[p]
#            #ei = conends[p]
#        except IndexError:
#            #an example of this came up where an isotopomer of a distribution that had an MS2 scan wasn't in the theoretical distribution, the distributions didn't match that well, and I think I'll just ignore it here for now and work with what I can
#            continue
#        subquals = qualifiers[p]
#        for sq in subquals:
#            subindex = bi + sq
#            sformula = subformulas[subindex]
#            #setting up subformula-specific probabilities
#            fragint = intensities[subindex]
#            isocounts = set()
#            competing = set()
#            competitors = {}
#            isosums = {}
#            for ss in sformula.split(')')[:-1]:
#                iso, c = ss.split('(')
#                c = int(c)
#                e = iso[0]
#                if e in isocounts:
#                    competing.add(e)
#                    competitors[e][iso] = c
#                    isosums[e] += c
#                else:
#                    isocounts.add(e)
#                    competitors[e] = {iso: c}
#                    isosums[e] = c
#            fragprobs = {}
#            for e, v in competitors.items():
#                if e in competing:
#                    for iso, c in competitors[e].items():
#                        fragprobs[iso] = c / isosums[e]
#                else:
#                    for iso in v:
#                        fragprobs[iso] = 1
#            for ion, fragcomp in fragments.items():
#                fm, fi = max_fragment(fragprobs, fragcomp, fragint)
#                fragmasses.append(fm)
#                fragints.append(fi)
#                fragpositions.append(p)
#                fragindices.append(subindex)
#                fragions.append(ion)
#    fragmasses = np.array(fragmasses)[:,None]
#    fragtol = (fragmasses / 1000000 * ppmtol).flatten()
#    for analyteid, sids in samples.items():
#        for sid in sids:
#            matches = neighbors[sid].query_ball_point(fragmasses, fragtol)
#            for n, m in enumerate(matches.tolist()):
#                for sm in m:
#                    mlist = (seq, sm, fragions[n], fragmasses[n][0], fi, sid, analyteid, fragpositions[n], fragindices[n])
#                    outlist.append(mlist)
#    return outlist
#
#nt = time()
##i think isotopomerpositionsofanalytes could use formulas as a key instead, confirm all the lists of things sharing a formula are the same thing
##spectrabyseq = {} #seq: ion-subindex: [[masses], [frags]]
##spectrabyformula = defaultdict(lambda: defaultdict(set)) #formula: analyteid: [scans]
##isotopomerpositionsofanalytes = defaultdict(set) #analyteid: isotopopmer coordinate from max
#fraginputs = []
#fragmatchlist = []
#for formula, samples in testspecs.items():
#    qualifiers = subisodepthqualifiers[formula]
#    conlengths = condensationcoordinates[formula]
#    conends = conlengths.cumsum()
#    constarts = conends - conlengths
#    subformulas = [i.decode() for i in abundanceformulas[formula]]
#    massesandintensities = abundances[formula]
#    intensities = massesandintensities[1]
#    sumintensities = np.array([intensities[s:e].sum() for s, e in zip(constarts.tolist(), conends.tolist())])
#    maxintensityindex = sumintensities.argmax()
#    positions = set()
#    for analyteid, sids in samples.items():
#        positions.update(isotopomerpositionsofanalytes[analyteid])
#    positions = [i + maxintensityindex for i in positions]
#    for seq in seqsbyformula[formula]:
#        fragments = fragmentation_compositions(seq)
#        fraginputs.append((constarts, subformulas, qualifiers, intensities, positions, fragments, samples, seq))
#
#print(time() - nt, 'inputs prepared')
#
#    ##memory check
#    #if psutil.virtual_memory().percent > 70:
#    #    #this concept would be flawed if you collect everything in a list thats still in memory, it HAS to dump to disk or this won't work
#    #    break
#with mp.Pool(nprocs) as pool:
#    for ol in pool.starmap(frag_match, fraginputs):
#        #totalmatches.update(tm)
#        fragmatchlist.extend(ol)
#
##        with mp.Pool(nprocs) as pool:
##        result = pool.apply_async(frag_match, args=(constarts, subformulas, qualifiers, intensities, positions, fragments, samples, seq))
##        results.append(result)
##    #insert memory handling here
##    if psutil.virtual_memory().percent > 90:
##        fragmatchlist = []
##        for result in results:
##            fragmatchlist.extend(result.get())
##        frag_csv(filename, fragmatchlist)
##        results.clear()
##        fragmatchlist.clear()
##        print(time() - nt, 'mid-save')
##fragmatchlist = []
##for result in results:
##    fragmatchlist.extend(result.get())
##frag_csv(filename, fragmatchlist)
##results.clear()
##fragmatchlist.clear()
#print(time() - nt, 'finished')


#maybe pre-process all the inputs beforehand, then run through the multiprocessing in chunks?

#nt = time()
#totalmatches = defaultdict(lambda: Counter()) #analyteid: sequence: nmatches
#with concurrent.futures.ProcessPoolExecutor(nprocs) as executor:
#    futures = []
#    for formula, samples in testspecs.items():
#        futures.append(executor.submit(frag_match, formula, samples))
#    for f in concurrent.futures.as_completed(futures):
#        totalmatches.update(f.result())
#print(time() - nt, 'concurrent.futures')

#this doesn't normalize sub-spectra per original subformula intensity
#for si, spec in seqspectra.items():
#    for ion, ma in spec.items():
#        plt.bar(ma[0], ma[1], width=0.5, alpha=0.5)
#plt.show()

#seqmasses = np.array(seqmasses)
#seqints = np.array(seqints)
#
#seqints = seqints[seqmasses.argsort()]
#seqmasses = np.sort(seqmasses)
#
#massdiffs = np.diff(seqmasses)
#massmeans = (seqmasses[:-1] + seqmasses[1:]) / 2
#
#fig, ax = plt.subplots(nrows=2, figsize=(6,8), sharex=True)
#ax[0].bar(seqmasses, seqints, width=0.2)
#ax[1].bar(massmeans, massdiffs, width=0.2)
#fig.subplots_adjust(hspace=0.05)
#ax[0].set_yscale('log')
#ax[1].set_ylim(0,1.2)
#plt.show()


#collect [seq, ion, experimental mass, experimental mass ind (this will be from the deconvoluted data), raw mass distance error, theoretical abundance, spectra intensity, scan, analyteid, position]
#i won't be able to get spectra intensity or mass distance yet b/c of how i'm using the nn
