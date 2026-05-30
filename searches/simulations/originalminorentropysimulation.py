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
minlone = 8 #must be > 1
maxlone = 15
matchingradius = 25 #ppm
threshold = 0.84 #% of fragments that need to be held within nmainions
minmainfrags = 3 #^min of nmainions
maxmainfrags = 5


nt = time()
finalcounts = []
j = 0
while j < 1000:
    j += 1

#lines = [0, 1, 2, 3, 4]
    lines = np.arange(nlines)

#scansfromlines = [[0, 1],
#         [1],
#         [2, 3],
#         [2, 3, 4],
#         [3, 4]] #line (as the index): [scans]
    scansfromlines = [] #line as index, [scans]
    for n in range(nlines):
        nscans = np.random.randint(1, 4)
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
    linesofscans = defaultdict(list) #scan: [lines]
    scansoflines = defaultdict(list)
    for line, scans in enumerate(scansfromlines):
        for scan in scans:
            linesofscans[scan].append(line)
            scansoflines[line].append(scan)
    for k, v in linesofscans.items():
        linesofscans[k] = tuple(set(v))
        if debugging:
            print(f'scan: {k}, lines: {v}')
    if debugging:
        print('~')

    currentind = 0
    massesoflines = {} #line: [masses]
    indicesandprobabilities = {} #line: [[mass inds], [probs]]
    linesbyprimaryind = {} #primary ind: line
    for n in range(nlines):
        num = np.random.randint(minlone, maxlone)
        masses = np.random.uniform(low=10, high=20, size=num).tolist()
        massesoflines[n] = masses
        inds = np.arange(num) + currentind
        nmainions = np.random.randint(minmainfrags, maxmainfrags)
        probs = np.random.uniform(size=num)
        while True:
            probs *= probs
            test = probs / probs.sum()
            if np.sort(test)[-nmainions:].sum() >= threshold:
                break
        probs = probs / probs.sum() #basic stuff for now
        indicesandprobabilities[n] = [inds.tolist(), probs.tolist()]
        for i in inds.tolist():
            linesbyprimaryind[i] = n
        currentind += num
        if debugging:
            print(f'line: {n}, inds: {inds}')
    if debugging:
        print('~')

    mergedlines = intersection_merge(linesofscans.values())
    mergedscans = [tuple(set(itertools.chain(*[scansoflines[j] for j in i]))) for i in mergedlines]

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
        inds = list(itertools.chain(*[indicesandprobabilities[i][0] for i in linelist]))
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
    for scan, slines in linesofscans.items():
        for line in slines:
            inds, probs = indicesandprobabilities[line]
            num = len(inds)
            nsample = np.random.randint(minlone-1, num)
            while True:
                try:
                    sampleinds = np.random.choice(inds, p=probs, size=nsample, replace=False)
                    break
                except ValueError: #fewer non-zero entries in p than size
                    num -= 1
                    nsample = np.random.randint(minlone-1, num)
            maininds = [primarytomainindex[i] for i in sampleinds]
            mainindicesbyscan[scan].update(maininds)
            for m in maininds:
                truelineindex[m].add(line)

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

    nt = time()

    entropyorganizer = defaultdict(lambda: defaultdict(int)) #mainind: line: count
    for scangroup in mergedscans:
        maininds = {linesofscans[i]: set(mainindicesbyscan[i]) for i in scangroup} #(lines): [mass indices]
        if len(maininds) > 1:
            #iterate a combination of size n, and co-iterate combinations of the (total length - n) to compare each group
            maxlen = len(maininds)
            iterlength = maxlen - 1
            seencomparisons = set() #with the intersection_merge capping the loop, i don't think this is necessary? no it still is, just within this group now, the combinations can be symmetrical?
            while iterlength < maxlen:
                coiterlength = maxlen - iterlength
                for itercomb in itertools.combinations(maininds, iterlength):
                    iterset = set()
                    for i in itercomb:
                        match i:
                            case int():
                                iterset.add(i)
                            case tuple():
                                iterset.update(i)
                    #itercomb = itercomb[0] #why was this here???
                    for coitercomb in itertools.combinations(maininds, coiterlength):
                        coiterset = set()
                        for c in coitercomb:
                            match c:
                                case int():
                                    coiterset.add(c)
                                case tuple():
                                    coiterset.update(c)
                        #coitercomb = coitercomb[0] #same with this???
                        if itercomb != coitercomb:
                            #combstrings = 'x'.join((str(itercomb), str(coitercomb))), 'x'.join((str(coitercomb), str(itercomb)))
                            #if combstrings[0] not in seencomparisons:
                            #    seencomparisons.update(combstrings)
                            combintersection = iterset.intersection(coiterset)
                            itercombinds = set(itertools.chain(*[maininds[i] for i in itercomb]))
                            coitercombinds = set(itertools.chain(*[maininds[c] for c in coitercomb]))
                            if combintersection:
                                #mainindintersection = tuple(maininds[itercomb].intersection(maininds[coitercomb])) #assign this to everything in combintersection
                                mainindintersection = itercombinds.intersection(coitercombinds)
                                if mainindintersection:
                                    combintersection = tuple(combintersection)
                                    if len(combintersection) == 1:
                                        combintersection = combintersection[0]
                                    #for ind in mainindintersection:
                                    #    entropyorganizer[ind][combintersection] += 1
                                    match combintersection:
                                        case int():
                                            for ind in mainindintersection:
                                                entropyorganizer[ind][combintersection] += 1
                                        case tuple():
                                            for ind in mainindintersection:
                                                for c in combintersection:
                                                    entropyorganizer[ind][c] += 1
                                #get the difference and intersection or something?
                                #of coiter and iter sets as well as maininds values
                                iterdiff = iterset.difference(coiterset)
                                #^should i be checking if the union != the difference?
                                if iterdiff:
                                    #difference exists, take the diff of the maininds
                                    #itermaininds = tuple(maininds[itercomb].difference(maininds[coitercomb]))
                                    itermaininds = itercombinds.difference(coitercombinds)
                                    #label = tuple(iterset.union(coiterset)) #automatically sorts
                                    #why the fuck is the label a union while this thing is for diffs???
                                    #should i be getting the union, +1ing those, then +1ing different things for the diffs?
                                    label = tuple(iterset.difference(coiterset)) #automatically sorts
                                else:
                                    #everything from the iter is within the coiter, mark everything as being from the itercomb key
                                    #itermaininds = tuple(maininds[itercomb])
                                    itermaininds = itercombinds
                                    label = itercomb
                                #for ind in itermaininds:
                                #    entropyorganizer[ind][label] += 1
                                match label:
                                    case int():
                                        for ind in itermaininds:
                                            entropyorganizer[ind][label] += 1
                                    case tuple():
                                        for ind in itermaininds:
                                            for l in label:
                                                entropyorganizer[ind][l] += 1
                                coiterdiff = coiterset.difference(iterset)
                                if coiterdiff:
                                    #difference exists, take the diff of the maininds
                                    #coitermaininds = tuple(maininds[coitercomb].difference(maininds[itercomb]))
                                    coitermaininds = coitercombinds.difference(itercombinds)
                                    #label = tuple(coiterset.union(iterset))
                                    label = tuple(coiterset.difference(iterset))
                                else:
                                    #everything from the coiter is within the iter, mark everything as being from the coitercomb key
                                    coitermaininds = coitercombinds
                                    label = coitercomb
                                #for ind in coitermaininds:
                                #    entropyorganizer[ind][label] += 1
                                match label:
                                    case int():
                                        for ind in coitermaininds:
                                            entropyorganizer[ind][label] += 1
                                    case tuple():
                                        for ind in coitermaininds:
                                            for l in label:
                                                entropyorganizer[ind][l] += 1
                            else: #no intersection, +1 to either for what they have
                                for i in iterset:
                                    for ic in itercombinds:
                                        entropyorganizer[ic][i] += 1
                                for c in coiterset:
                                    for cc in coitercombinds:
                                        entropyorganizer[cc][i] += 1
                iterlength += 1
        else: #no competition
            for line, inds in maininds.items():
                for mi in inds:
                    entropyorganizer[mi][line] += 1

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

    midcount = [nwrongs, correctoutcomes, notincorrectoutcomes]

    nt = time()

    entropyorganizer = defaultdict(lambda: defaultdict(int)) #mainind: line: count
    for scangroup in mergedscans:
        maininds = {linesofscans[i]: set(mainindicesbyscan[i]) for i in scangroup} #(lines): [mass indices]
        if len(maininds) > 1:
            #iterate a combination of size n, and co-iterate combinations of the (total length - n) to compare each group
            maxlen = len(maininds)
            iterlength = maxlen - 1
            seencomparisons = set() #with the intersection_merge capping the loop, i don't think this is necessary? no it still is, just within this group now, the combinations can be symmetrical?
            #while iterlength < maxlen:
            coiterlength = maxlen - iterlength
            #for itercomb in itertools.combinations(maininds, iterlength):
            for itercomb in maininds:
                for coitercomb in maininds:
                    if itercomb != coitercomb:
                        iterset = set()
                        for i in itercomb:
                            match i:
                                case int():
                                    iterset.add(i)
                                case tuple():
                                    iterset.update(i)
                        coiterset = set()
                        for c in coitercomb:
                            match c:
                                case int():
                                    coiterset.add(c)
                                case tuple():
                                    coiterset.update(c)
                        combintersection = iterset.intersection(coiterset)
                        itercombinds = maininds[itercomb]
                        coitercombinds = maininds[coitercomb]
                        if combintersection:
                            mainindintersection = itercombinds.intersection(coitercombinds)
                            if mainindintersection:
                                combintersection = tuple(combintersection)
                                if len(combintersection) == 1:
                                    combintersection = combintersection[0]
                                match combintersection:
                                    case int():
                                        for ind in mainindintersection:
                                            entropyorganizer[ind][combintersection] += 1
                                    case tuple():
                                        for ind in mainindintersection:
                                            for c in combintersection:
                                                entropyorganizer[ind][c] += 1
                            #get the difference and intersection or something?
                            #of coiter and iter sets as well as maininds values
                            iterdiff = iterset.difference(coiterset)
                            #^should i be checking if the union != the difference?
                            if iterdiff:
                                #difference exists, take the diff of the maininds
                                #itermaininds = tuple(maininds[itercomb].difference(maininds[coitercomb]))
                                itermaininds = itercombinds.difference(coitercombinds)
                                #label = tuple(iterset.union(coiterset)) #automatically sorts
                                #why the fuck is the label a union while this thing is for diffs???
                                #should i be getting the union, +1ing those, then +1ing different things for the diffs?
                                label = tuple(iterset.difference(coiterset)) #automatically sorts
                            else:
                                #everything from the iter is within the coiter, mark everything as being from the itercomb key
                                #itermaininds = tuple(maininds[itercomb])
                                itermaininds = itercombinds
                                label = itercomb
                            #for ind in itermaininds:
                            #    entropyorganizer[ind][label] += 1
                            match label:
                                case int():
                                    for ind in itermaininds:
                                        entropyorganizer[ind][label] += 1
                                case tuple():
                                    for ind in itermaininds:
                                        for l in label:
                                            entropyorganizer[ind][l] += 1
                            coiterdiff = coiterset.difference(iterset)
                            if coiterdiff:
                                #difference exists, take the diff of the maininds
                                #coitermaininds = tuple(maininds[coitercomb].difference(maininds[itercomb]))
                                coitermaininds = coitercombinds.difference(itercombinds)
                                #label = tuple(coiterset.union(iterset))
                                label = tuple(coiterset.difference(iterset))
                            else:
                                #everything from the coiter is within the iter, mark everything as being from the coitercomb key
                                coitermaininds = coitercombinds
                                label = coitercomb
                            #for ind in coitermaininds:
                            #    entropyorganizer[ind][label] += 1
                            match label:
                                case int():
                                    for ind in coitermaininds:
                                        entropyorganizer[ind][label] += 1
                                case tuple():
                                    for ind in coitermaininds:
                                        for l in label:
                                            entropyorganizer[ind][l] += 1
                        #else: #no intersection, +1 to either for what they have
                        #    for i in iterset:
                        #        for ic in itercombinds:
                        #            entropyorganizer[ic][i] += 1
                        #    for c in coiterset:
                        #        for cc in coitercombinds:
                        #            entropyorganizer[cc][i] += 1
                #iterlength += 1
        else: #no competition
            for line, inds in maininds.items():
                for mi in inds:
                    entropyorganizer[mi][line] += 1

    #print(time() - nt, 'entropy estimated')

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

    midcount.extend([nwrongs, correctoutcomes, notincorrectoutcomes])

    finalcounts.append(midcount)

finalcounts = np.array(finalcounts)

plt.hist(list(zip(*finalcounts[:,:4])), bins=100, label=['wrong', 'correct', 'not incorrect'])
plt.legend()
plt.title(f'nlines: {nlines}, minlone: {minlone}, maxlone: {maxlone}, radius: {matchingradius}, mainfrags: {minmainfrags}-{maxmainfrags}, threshold: {threshold}', fontsize=10)
plt.show()

plt.hist(list(zip(*finalcounts[:,3:])), bins=100, label=['wrong', 'correct', 'not incorrect'])
plt.legend()
plt.title(f'nlines: {nlines}, minlone: {minlone}, maxlone: {maxlone}, radius: {matchingradius}, mainfrags: {minmainfrags}-{maxmainfrags}, threshold: {threshold}', fontsize=10)
plt.show()


print('-------------------------------------------')
print(f'total wrong: {finalcounts[:,0].sum()} vs. {finalcounts[:,3].sum()}')
print(f'total correct: {finalcounts[:,1].sum()} vs. {finalcounts[:,4].sum()}')
print(f'total not incorrect {finalcounts[:,2].sum()} vs. {finalcounts[:,5].sum()}')

print(time() - nt, 'finished')

#plt.bar(*list(zip(*notincorrectdistances.items())))
#plt.title('not incorrect')
#plt.show()
#
#plt.bar(*list(zip(*badoutcomesbadmatches.items())))
#plt.title('bad outcomes bad matches')
#plt.show()
#
#plt.bar(*list(zip(*badoutcomesgoodmatches.items())))
#plt.title('bad outcomes good matches')
#plt.show()

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
