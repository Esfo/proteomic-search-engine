import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from colour import Color
import psutil
import asyncio
import aiofiles
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

#make:
lines = [0, 1, 2, 3, 4, 5]
scans = [1, 2, 3, 4]
linesofscans = {0: [0, 1, 2], #scan: [lines]
                1: [1, 2],
                2: [3, 4, 5],
                3: [3, 4, 5],
                4: [4, 5]}
scansoflines = defaultdict(list) #line: [scans]
for scan, lines in linesofscans.items():
    for line in lines:
        scansoflines[line].append(scan)
massesoflines = {0: [0, 1, 2, 3, 4, 5, 6, 7, 8], #line: [primaries]
                 1: [9, 10, 11, 12, 13],
                 2: [14, 15, 16, 17, 18, 19, 20],
                 3: [21, 22, 23, 24, 25, 26, 27, 28, 29],
                 4: [30, 31, 32, 33],
                 5: [34, 35, 36, 37, 38, 39, 40]}

for scan, lines in linesofscans.items():
    linesofscans[scan] = tuple(sorted(lines))
for line, scans in scansoflines.items():
    scansoflines[line] = tuple(sorted(scans))

primaryindsofscans = defaultdict(list) #scan: [primaryinds]
linesbyprimaryind = {} #primary ind: line
for line, primaries in massesoflines.items():
    for scan in scansoflines[line]:
        primaryindsofscans[scan].extend(massesoflines[line])
    for primary in primaries:
        linesbyprimaryind[primary] = line
primariesofmain = {0: [0], #mainind: [primaries]
                  1: [1],
                  2: [2, 11],
                  3: [3],
                  4: [4, 17],
                  5: [5],
                  6: [6],
                  7: [7],
                  8: [8],
                  9: [9],
                  10: [10, 15],
                  11: [12],
                  12: [13],
                  13: [14],
                  14: [16],
                  15: [18],
                  16: [19],
                  17: [20],
                  18: [21],
                  19: [22, 31, 35],
                  20: [23],
                  21: [24],
                  22: [25],
                  23: [26, 39],
                  24: [27],
                  25: [28],
                  26: [29],
                  27: [30],
                  28: [32],
                  29: [33],
                  30: [34],
                  31: [36],
                  32: [37],
                  33: [38],
                  34: [40]}

mainofprimaries = {}
for main, primaries in primariesofmain.items():
    for primary in primaries:
        mainofprimaries[primary] = main
mainindicesbyscan = defaultdict(set) #scan: [maininds]
scansbymainindices = defaultdict(set)
scansbyprimaryind = {}
maintoprimaryindex = defaultdict(list) #main: [primaries]
for scan, primaries in primaryindsofscans.items():
    for primary in primaries:
        mainindicesbyscan[scan].add(mainofprimaries[primary])
        scansbyprimaryind[primary] = scan
        maintoprimaryindex[mainofprimaries[primary]].append(primary)
for scan, maininds in mainindicesbyscan.items():
    mainindicesbyscan[scan] = list(maininds)
    for mainind in maininds:
        scansbymainindices[mainind].add(scan)

linesbymainindex = {} #mainind: [lines] THIS IS SECRET INFORMATION ONLY TO BE USED IN TESTING
for main, primaries in primariesofmain.items():
    linesbymainindex[main] = set(linesbyprimaryind[i] for i in primaries)

mergedlines = intersection_merge(linesofscans.values())
mergedscans = [tuple(set(itertools.chain(*[scansoflines[j] for j in i]))) for i in mergedlines]

print(len(mergedscans) == len(intersection_merge(mergedscans)), 'result for mergedscan test')

#im thinking of these as something more akin to boundaries
#this is a pretty fair estimate of correctedness with some pieces that slip through the cracks
#im thinking a simpler, more conservative bound would be to just assign each main indice to the lines of those scans, period

nt = time()

entropyorganizer = defaultdict(lambda: defaultdict(int)) #mainind: line: count
for scangroup in mergedscans:
    lineindices = {}
    massindices = {}
    for n, scan in enumerate(scangroup):
        lineindices[n] = set(linesofscans[scan])
        massindices[n] = set(mainindicesbyscan[scan])
    if n > 0:
        for itercomb in massindices:
            iterset = lineindices[itercomb]
            for coitercomb in massindices:
                if itercomb != coitercomb:
                    coiterset = lineindices[coitercomb]
                    combintersection = iterset.intersection(coiterset)
                    itercombinds = massindices[itercomb]
                    coitercombinds = massindices[coitercomb]
                    if combintersection:
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
                        coiterdiff = coiterset.difference(iterset)
                        if coiterdiff:
                            #difference exists, take the diff of the maininds
                            coitermaininds = coitercombinds.difference(itercombinds)
                            label = tuple(coiterset.difference(iterset))
                        else:
                            #everything from the coiter is within the iter, mark everything as being from the coitercomb key
                            coitermaininds = coitercombinds
                            label = tuple(coiterset)
                        for ind in coitermaininds:
                            for l in label:
                                entropyorganizer[ind][l] += 1
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

nt = time()

entropyorganizer = defaultdict(lambda: defaultdict(int)) #mainind: line: count
for scangroup in mergedscans:
    lineindices = {}
    massindices = {}
    for n, scan in enumerate(scangroup):
        lineindices[n] = set(linesofscans[scan])
        massindices[n] = set(mainindicesbyscan[scan])
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

assignmentresults = {} #primary: [results]
for ms2line, assignablems1lines in entropyorganizer.items():
    primaries = primariesofmain[ms2line]
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
print('correct:', correct)
print('incorrect:', incorrect)
print('not incorrect:', notincorrect)
print(correct + incorrect + notincorrect)

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
        else:
            outcome = mostcommon[0][:1]
    else:
        outcome = mostcommon[0][:1]
    entropicboundaries[mainind] = outcome

for mainind, lines in entropicboundaries.items():
    indscans = scansbymainindices[mainind]
    boundaryscans = tuple(set(itertools.chain(*[scansoflines[i] for i in lines])))
    if indscans.difference(boundaryscans):
        print('mainind:', mainind)
        print('lines:', lines)
        print('indscans:', indscans)
        print('boundaryscans:', boundaryscans)
        print('entropyorganizer:', entropyorganizer[mainind])
        print('linesbymainindex:', linesbymainindex[mainind])
        print('lines of scans:', [linesofscans[i] for i in indscans])
        print('scans of ^lines:', [{j: scansoflines[j] for j in linesofscans[i]} for i in indscans])
        break

ms1entropy = defaultdict(lambda: Counter()) #ms2line: line: [union, ms1diff, ms2diff, symdiff]

primaryguesses = defaultdict(list) #primary: [ms1, ms2, union, classical] as indices
ms1assignments = defaultdict(set) #either 0, 1 or 2, based on below: [primaries]
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
            ms1entropy[mainindex][line] -= ms1diffs
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
    print('primary:', primary)
    print('results:', results)
    print('true:', trueresult)
    print('~')

print('MS1 Entropy')
print('correct:', correct)
print('incorrect:', incorrect)
print('not incorrect:', notincorrect)
print(correct + incorrect + notincorrect)
print('~')
