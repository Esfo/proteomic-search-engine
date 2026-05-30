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

processinglocation = '/'.join((basefolder, 'fileprocessing', basefile))
librarylocation = '/home/sfo/data/proteomics/fastas/search-db'
#librarylocation = '/home/sfo/data/proteomics/fastas/iodoacetamide-search/'
proteome = 'Human_Homo_sapien'
nprocs = 8
subisotopomericdepth = 0.8
proton = 1.007276554940804
dividingthreshold = 0.1

nt = time()

environment_partial = partial(lmdb.Environment, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=9999, max_spare_txns=1, lock=True)

linepositionsbyformulafile = '/'.join((processinglocation, 'linepositionsbyformula.pickle'))
with open(linepositionsbyformulafile, 'rb') as pick:
    linepositionsbyformula = pickle.load(pick)
#linepositionsbyformula = defaultdict(lambda: defaultdict(set)) #formula: matchable library positions of a distribution: [lines that have ms2 scans]

scansoflinesfile = '/'.join((processinglocation, 'scansoflines.pickle'))
with open(scansoflinesfile, 'rb') as pick:
    scansoflines = pickle.load(pick)
#scansoflines = defaultdict(list) #lineuid: [ms2 scan indices]

maxsampledistributionsoflinesfile = '/'.join((processinglocation, 'maxsampledistributionsoflines.pickle'))
with open(maxsampledistributionsoflinesfile, 'rb') as pick:
    maxsampledistributionsoflines = pickle.load(pick)
#maxsampledistributionsoflines = {} #line: distid

seqswithdecoysbyformulafile = '/'.join((processinglocation, 'seqswithdecoysbyformula.pickle'))
with open(seqswithdecoysbyformulafile, 'rb') as pick:
    seqswithdecoysbyformula = pickle.load(pick)
#seqswithdecoysbyformula = defaultdict(list) #formula: [seqs + decoys]

encodedkeys = [i.encode() for i in linepositionsbyformula]

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

print(time() - nt, 'initialized')

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

#mainindexbysubindex = defaultdict(list) #sub match index: main match index
#scansofmainindices = defaultdict(set) #main index: [scans]

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
#submatchsubformulas = {} #submatchindex: subformula
elementsofprobabilityindices = {} #prob index: e
#submatchpositions = {} #submatch index: [distribution position, subiso position]

linesbysubformula = defaultdict(set) #subformula: [lines that have ms2 scans]

#subformulasoflines = defaultdict(set) #line: [subformulas]
#subformularank = defaultdict(dict) #sequence: subformula: descending subiso rank, lower int = more relevant subiso
#subformulapercent = defaultdict(dict) #sequence: subformula: ms1 subiso abundance #the below version produces a smaller dict
subformulapercent = defaultdict(dict) #subformula: sequence: (subiso abundance rank, subiso abundance)

subformulasofsequencedistribution = defaultdict(dict) #dist: seq: subformula

mergables = []

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
    #mainindexformulas[mainmatchindex] = formula
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
                    #for scan in scansoflines[line]:
                    #outgroup = []
                    #subgroups = defaultdict(list) #element: [subiso comps]
                    #for split in sformula.split(')')[:-1]:
                    #    e = split[0]
                    #    if e == 'C':
                    #        subgroups[e].append(split)
                    #for e, group in subgroups.items():
                    #    outgroup.append(')'.join((group)) + ')subsplit')
                    #mergables.append(f'{sformula}-{seq}-{scan}-{'-'.join((outgroup))}') #trying to preserve memory here
                    #mergables.append(f'{sformula}-{seq}-{'-'.join((outgroup))}') #trying to preserve memory here
                mergables.append(f'{sformula}-{seq}') #trying to preserve memory here
                #    subformulasoflines[line].add(sformula)
                #mainindexbysubindex[submatchindex] = mainmatchindex
                #submatchsubformulas[submatchindex] = sformula
                subformulasubindices[sformula].append(submatchindex)
                submatchsequences[submatchindex] = seq
                #submatchpositions[submatchindex] = position, sq
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

#mainindexbysubindex = dict(mainindexbysubindex)
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
#nt = time()

#n seqs > n subformulas > n scans
#subformulabyscanbyseq = defaultdict(lambda: defaultdict(list)) #scan: subformula: [seqs]
#mergables = []
#for smi, subformula in submatchsubformulas.items():
#    seq = submatchsequences[smi]
#    for scan in linesbyscanbysubformula[subformula]:
#        mergables.append([subformula, seq, scan])
    #THIS IS WRONG
    #these scans dont necessarily relate to these sequences

#print(time() - nt, 'mergables made')

#nt = time()
#
#mergables = list(set(mergables))
#
##custom intersection merge to limit the size of each group to whatevers written below
##makes for good memory management within fragmentwriter
##members can be scans, subformulas, or sequences
##the output size should always only be ~2x the limiter size at most - actually think its limiter * n of original locs, i only see < the ~2x though, idk why
#limiter = 1000000 #this is roughly the max of whats hit in the fr_400 file, so im basing it off this
#sn = 0
#groupsofitems = {} #member: group
##itemgroups = defaultdict(set) #group: [members]
#dividedlayers = defaultdict(lambda: defaultdict(lambda: defaultdict(list))) #loc: scans: subformula: [seqs]
#splitgroups = defaultdict(set) #loc: [iso splits], doing these separate
#groupsize = defaultdict(int) #loc: current size
#for items in mergables:
#    items = items.split('-')
#    items[2] = int(items[2])
#    subformula, seq, scan, *splits = items
#    locs = set()
#    for i in items:
#        if i in groupsofitems:
#            locs.add(groupsofitems[i])
#    if locs:
#        joiner = locs.pop()
#        for oldloc in locs:
#            #for ol in itemgroups[oldloc]:
#            for oscan in dividedlayers[oldloc]:
#                #groupsofitems[oscan] = joiner
#                groupsize[joiner] += 1
#                for osubformula in dividedlayers[oldloc][oscan]:
#                    groupsofitems[osubformula] = joiner
#                    groupsize[joiner] += 1
#                    for oseq in dividedlayers[oldloc][oscan][osubformula]:
#                        groupsofitems[oseq] = joiner
#                        dividedlayers[joiner][oscan][osubformula].append(oseq)
#                        groupsize[joiner] += 1
#            for s in splitgroups[oldloc]:
#                groupsofitems[s] = joiner
#            splitgroups[joiner].update(splitgroups[oldloc])
#            #itemgroups[joiner].update(itemgroups.pop(oldloc))
#            del dividedlayers[oldloc]
#            del groupsize[oldloc]
#            del splitgroups[oldloc]
#    else:
#        joiner = sn
#        sn += 1
#    #itemgroups[joiner].update(items)
#    dividedlayers[joiner][scan][subformula].append(seq)
#    splitgroups[joiner].update(splits)
#    groupsize[joiner] += 3
#    #for i in items:
#    #    groupsofitems[i] = joiner
#    #excluding scans from the merging process
#    groupsofitems[subformula] = joiner
#    groupsofitems[seq] = joiner
#    for s in splits:
#        groupsofitems[s] = joiner
#    if groupsize[joiner] >= limiter:
#        #by deleting the old locs it will force them incoming items into new groups
#        for oscan in dividedlayers[joiner]:
#            for osubformula in dividedlayers[joiner][oscan]:
#                for oseq in dividedlayers[joiner][oscan][osubformula]:
#                    try:
#                        del groupsofitems[oseq]
#                    except KeyError:
#                        #this can have redunancy across other layers
#                        pass
#                try:
#                    del groupsofitems[osubformula]
#                except KeyError:
#                    #this can have redunancy across other layers
#                    pass
#            #del groupsofitems[oscan]
#        for s in splitgroups[joiner]:
#            del groupsofitems[s]
#
#for loc, scans in dividedlayers.items():
#    for scan, subformulas in scans.items():
#        for subformula, seqs in subformulas.items():
#            dividedlayers[loc][scan][subformula] = tuple(set(seqs))
#        dividedlayers[loc][scan] = dict(dividedlayers[loc][scan])
#    dividedlayers[loc] = dict(dividedlayers[loc])
#dividedlayers = dict(dividedlayers)
#
#dividedlayers = tuple(dividedlayers.values())
#
#print(time() - nt, 'dividedlayers made')

nt = time()

mergables = list(set(mergables))
firstmerge = map(tuple, intersection_merge(i.split('-') for i in mergables))
#^merging by seqs and subformulas as a first layer of redundancy reduction
#^this yields a large, somewhat unusable number of groups that would cause a lot of pain for the high redundancy of isotopic compositions to calculate later
#the second layer will be by isotopic composition by individual elements

#custom intersection merge to limit the size of each group to whatevers written below
#makes for good memory management within fragmentwriter
limiter = 40000 / len(ions) #this is roughly the max of whats hit in the fr_400 file, so im basing it off this
#^i could probably base this on the memory of this machine too and scale that based on whatever machine specs is running the analysis
#^i can also base this on the number of ions, this amount is fine for b+y, so it should be 2x then divide by 2, or divide by the number of ions
sn = 0
groupsofitems = {} #iso-member: group
itemgroups = defaultdict(set) #group: [members]
subitemgroups = defaultdict(set) #group: [isotopic compositions]
for items in firstmerge:
    locs = set()
    subitems = set()
    for i in items:
        if ')' in i:
            subgroups = defaultdict(list) #element: [subiso comps]
            for split in i.split(')')[:-1]:
                e = split[0]
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

dividedgroups = list(itemgroups.values())

print(time() - nt, 'dividedgroups made')

#1100 allowing all iso subsplits
#909 allowing only carbon! its a winner, still a LITTLE slower than optimal but hey oh well

divisionfile = '/'.join((processinglocation, 'dividedgroups.pickle'))
with open(divisionfile, 'wb') as pick:
    pickle.dump(dividedgroups, pick)

elementsofprobindicesfile = '/'.join((processinglocation, 'elementsofprobabilityindices.pickle'))
with open(elementsofprobindicesfile, 'wb') as pick:
    pickle.dump(elementsofprobabilityindices, pick)

probabilityorganizerfile = '/'.join((processinglocation, 'probabilityorganizer.pickle'))
with open(probabilityorganizerfile, 'wb') as pick:
    pickle.dump(probabilityorganizer, pick)

matchprobfile = '/'.join((processinglocation, 'matchprobabilities.pickle'))
with open(matchprobfile, 'wb') as pick:
    pickle.dump(matchprobabilities, pick)

subformulasubindsfile = '/'.join((processinglocation, 'subformulasubindices.pickle'))
with open(subformulasubindsfile, 'wb') as pick:
    pickle.dump(subformulasubindices, pick)

#mainindexformulasfile = '/'.join((processinglocation, 'mainindexformulas.pickle'))
#with open(mainindexformulasfile, 'wb') as pick:
#    pickle.dump(mainindexformulas, pick)

submatchsequencesfile = '/'.join((processinglocation, 'submatchsequences.pickle'))
with open(submatchsequencesfile, 'wb') as pick:
    pickle.dump(submatchsequences, pick)

#mainindexfile = '/'.join((processinglocation, 'mainindicesbysubindex.pickle'))
#with open(mainindexfile, 'wb') as pick:
#    pickle.dump(mainindexbysubindex, pick)

#submatchpositionsfile = '/'.join((processinglocation, 'submatchpositions.pickle'))
#with open(submatchpositionsfile, 'wb') as pick:
#    pickle.dump(submatchpositions, pick)

linesbyscanbysubformulafile = '/'.join((processinglocation, 'linesbyscanbysubformula.pickle'))
with open(linesbyscanbysubformulafile, 'wb') as pick:
    pickle.dump(linesbyscanbysubformula, pick)

#submatchsubformulasfile = '/'.join((processinglocation, 'submatchsubformulas.pickle'))
#with open(submatchsubformulasfile, 'wb') as pick:
#    pickle.dump(submatchsubformulas, pick)

#subformularankfile = '/'.join((processinglocation, 'subformularank.pickle'))
#with open(subformularankfile, 'wb') as pick:
#    pickle.dump(subformularank, pick)

subformulapercentfile = '/'.join((processinglocation, 'subformulapercent.pickle'))
with open(subformulapercentfile, 'wb') as pick:
    pickle.dump(subformulapercent, pick)

subformulasofsequencedistributionfile = '/'.join((processinglocation, 'subformulasofsequencedistribution.pickle'))
with open(subformulasofsequencedistributionfile, 'wb') as pick:
    pickle.dump(subformulasofsequencedistribution, pick)
#subformulasofsequencedistribution = {} #seq: dist: subformula
