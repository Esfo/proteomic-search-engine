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

#mzmlfile = '/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
mzmlfile = '/home/sfo/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
#mzmlfile = '/home/sfo/store/data/PXD035665/mzMLs/GradientAmount_HeLa_4ug_240min_R01.mzML'
msrun = mzml.MzML(mzmlfile, dtype=np.float64)

basefolder = '/'.join((mzmlfile.split('/')[:-2]))
basefile = mzmlfile.split('/')[-1].split('.mzML')[0]

processingdirectory = '/'.join((basefolder, 'fileprocessing', basefile)) + '/'
librarylocation = '/home/sfo/data/proteomics/fastas/search-db'
#librarylocation = '/home/sfo/data/proteomics/fastas/iodoacetamide-search/'
proteome = 'Human_Homo_sapien-NoTremb'
nprocs = 8
subisotopomericdepth = 0.5
proton = 1.007276554940804
dividingthreshold = 0.1
ions = 'by'

nt = time()

environment_partial = partial(lmdb.Environment, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=9999, max_spare_txns=1, lock=True)

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

t2 = time()

linepositionsbyformulafile = ''.join((processingdirectory, 'linepositionsbyformula.pickle'))
with open(linepositionsbyformulafile, 'rb') as pick:
    linepositionsbyformula = pickle.load(pick)
#linepositionsbyformula = defaultdict(lambda: defaultdict(set)) #formula: matchable library positions of a distribution: [lines that have ms2 scans]

scansbyanalytefile = ''.join((processingdirectory, 'scansbyanalyte.pickle'))
with open(scansbyanalytefile, 'rb') as pick:
    scansbyanalyte = pickle.load(pick)
#scansbyanalyte = defaultdict(list) #analyteid: [scans across all lines and charge states]

scansoflinesfile = ''.join((processingdirectory, 'scansoflines.pickle'))
with open(scansoflinesfile, 'rb') as pick:
    scansoflines = pickle.load(pick)
#scansoflines = defaultdict(list) #lineuid: [ms2 scan indices]

maxsampledistributionsoflinesfile = ''.join((processingdirectory, 'maxsampledistributionsoflines.pickle'))
with open(maxsampledistributionsoflinesfile, 'rb') as pick:
    maxsampledistributionsoflines = pickle.load(pick)
#maxsampledistributionsoflines = {} #line: distid

encodedkeys = [i.encode() for i in linepositionsbyformula]

seqsbyformula = {} #formula: [seqs]
abundances = {} #formula: [[masses], [intensities]]
abundanceformulas = {} #formula: subformulas
condensationcoordinates = {} #formula: [# isotopomers per proton-step]
subisodepthqualifiers = {} #formula: [[top n subisos of proton locations]]
with environment_partial(librarylocation) as env:
    ddb = env.open_db('distributions.formulas'.encode())
    with env.begin(write=False) as txn:
        with txn.cursor(ddb) as cursor:
            for k, v in cursor.getmulti(encodedkeys):
                abundanceformulas[k.decode()] = eval(v.decode())
    condensationdb = env.open_db('distributions.condensationcoordinates'.encode())
    with env.begin(write=False) as txn:
        with txn.cursor(condensationdb) as cursor:
            for k, v in cursor.getmulti(encodedkeys):
                condensationcoordinates[k.decode()] = np.frombuffer(v, dtype=int)
    subisoqualdb = env.open_db('distributions.subisodepthqualifiers'.encode())
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
    proteomedb = env.open_db((proteome + '.seqsbyformula').encode())
    with env.begin(write=False) as txn:
        with txn.cursor(proteomedb) as cursor:
            for k, v in cursor.getmulti(encodedkeys):
                seqsbyformula[k.decode()] = eval(v.decode())

t0 = time()

mergables = []
for formula, positions in linepositionsbyformula.items():
    qualifiers = subisodepthqualifiers[formula]
    conlengths = condensationcoordinates[formula]
    conends = conlengths.cumsum()
    constarts = conends - conlengths
    subformulas = [i.decode() for i in abundanceformulas[formula]]
    for position, lines in positions.items():
        for seq in seqsbyformula[formula]:
            bi = constarts[position]
            for qualrank, sq in enumerate(qualifiers[position]):
                subindex = bi + sq
                sformula = subformulas[subindex]
                descriptor = '/'.join((formula, str(position)))
                mergables.append((sformula, seq, descriptor))

mergables = list(set(mergables))
firstmerge = map(tuple, intersection_merge(mergables))
#^merging by seqs and subformulas as a first layer of redundancy reduction
#^this yields a large, somewhat unusable number of groups that would cause a lot of pain for the high redundancy of isotopic compositions to calculate later
#the second layer will be by isotopic composition by individual elements

print(time() - t0, 'initial merge')
t1 = time()
#custom intersection merge to limit the size of each group to whatevers written below
#makes for good memory management later when generating fragments
limiter = 3000 / len(ions)
sn = 0
groupsofitems = {} #iso-member: group
itemgroups = defaultdict(set) #group: [members]
subitemgroups = defaultdict(set) #group: [isotopic compositions]
for items in firstmerge:
    locs = set()
    subitems = set()
    for i in items:
        if ')' in i and '/' not in i:
            subgroups = defaultdict(list) #element: [subiso comps]
            for split in i.split(')')[:-1]:
                splitval = 0
                #for handling elements with multiple letters
                while True:
                    if split[splitval].isalpha():
                        splitval += 1
                    else:
                        break
                e = split[:splitval]
                if e == 'C':
                    subgroups[e].append(split)
            for e, group in subgroups.items():
                output = ')'.join((group)) + ')'
                if output in groupsofitems:
                    locs.add(groupsofitems[output])
                subitems.add(output)
    if locs:
        joiner = min(locs)
        if len(locs) > 1:
            for oldloc in locs.difference([joiner]):
                for ol in subitemgroups[oldloc]:
                    groupsofitems[ol] = joiner
                itemgroups[joiner].update(itemgroups.pop(oldloc))
                subitemgroups[joiner].update(subitemgroups.pop(oldloc))
    else:
        joiner = sn
        sn += 1
    itemgroups[joiner].update(items)
    subitemgroups[joiner].update(subitems)
    for i in subitems:
        groupsofitems[i] = joiner
    if len(itemgroups[joiner]) >= limiter:
        for member in subitemgroups[joiner]:
            #by deleting the old locs it will force them incoming items into new groups
            del groupsofitems[member]

initialdividedgroups = list((map(tuple, itemgroups.values())))

print(time() - t1, 'processable fragment groups assembled')
t2 = time()

dividedgroups = []
for divgroup in initialdividedgroups:
    halfwaylist = []
    sublinepositionsbyformula = defaultdict(lambda: defaultdict(set)) #formula: matchable library positions of a distribution: [lines that have ms2 scans]
    for i in divgroup:
        if '/' in i:
            formula, position = i.split('/')
            position = int(position)
            sublinepositionsbyformula[formula][position].update(linepositionsbyformula[formula][position])
        else:
            halfwaylist.append(i)
    for k, v in sublinepositionsbyformula.items():
        for sk, sv in v.items():
            v[sk] = tuple(sv)
        sublinepositionsbyformula[k] = dict(v)
    sublinepositionsbyformula = dict(sublinepositionsbyformula)
    dividedgroups.append((halfwaylist, sublinepositionsbyformula))

print(time() - t2, 'dividedgroups assembled')

print(time() - t2, 'submatch organization completed')
t3 = time()

divisionfile = ''.join((processingdirectory, 'dividedgroups.pickle'))
with open(divisionfile, 'wb') as pick:
    pickle.dump(dividedgroups, pick)

formulasfile = ''.join((processingdirectory, 'encodedformulas.pickle'))
with open(formulasfile, 'wb') as pick:
    pickle.dump(encodedkeys, pick)

#elementsofprobindicesfile = ''.join((processingdirectory, 'elementsofprobabilityindices.pickle'))
#with open(elementsofprobindicesfile, 'wb') as pick:
#    pickle.dump(elementsofprobabilityindices, pick)
#
#probabilityorganizerfile = ''.join((processingdirectory, 'probabilityorganizer.pickle'))
#with open(probabilityorganizerfile, 'wb') as pick:
#    pickle.dump(probabilityorganizer, pick)
#
#matchprobfile = ''.join((processingdirectory, 'matchprobabilities.pickle'))
#with open(matchprobfile, 'wb') as pick:
#    pickle.dump(matchprobabilities, pick)
#
#subformulasubindsfile = ''.join((processingdirectory, 'subformulasubindices.pickle'))
#with open(subformulasubindsfile, 'wb') as pick:
#    pickle.dump(subformulasubindices, pick)
#
#submatchsequencesfile = ''.join((processingdirectory, 'submatchsequences.pickle'))
#with open(submatchsequencesfile, 'wb') as pick:
#    pickle.dump(submatchsequences, pick)
#
#linesbyscanbysubformulafile = ''.join((processingdirectory, 'linesbyscanbysubformula.pickle'))
#with open(linesbyscanbysubformulafile, 'wb') as pick:
#    pickle.dump(linesbyscanbysubformula, pick)
#
#subformulapercentfile = ''.join((processingdirectory, 'subformulapercent.pickle'))
#with open(subformulapercentfile, 'wb') as pick:
#    pickle.dump(subformulapercent, pick)
#
#subformulasofsequencedistributionfile = ''.join((processingdirectory, 'subformulasofsequencedistribution.pickle'))
#with open(subformulasofsequencedistributionfile, 'wb') as pick:
#    pickle.dump(subformulasofsequencedistribution, pick)
##subformulasofsequencedistribution = {} #seq: dist: subformula
