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
import fcntl #this will need to be portalocker on other operating systems
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
#mzmlfile = '/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
basefolder = '/'.join((mzmlfile.split('/')[:-2]))
basefile = mzmlfile.split('/')[-1].split('.mzML')[0]

processinglocation = '/'.join((basefolder, 'fileprocessing', basefile))
fragmentlocation = '/'.join((basefolder, 'fileprocessing', basefile, 'fragments'))
scanalytelocation = '/'.join((basefolder, 'fileprocessing', basefile, 'scanalytegroups'))
librarylocation = '/home/sfo/data/proteomics/fastas/search-db'
csvfilename = '/'.join((processinglocation, 'fragment.matches'))
proteome = 'Human_Homo_sapien'
nprocs = 8
proton = 1.007276554940804
dividingthreshold = 0.8
ppmtol = 25
ppmmod = ppmtol / 1000000

linesofscansfile = '/'.join((processinglocation, 'linesofscans.pickle'))
with open(linesofscansfile, 'rb') as pick:
    linesofscans = pickle.load(pick)
#linesofscans = defaultdict(list) #scan [lines]

analytefile = '/'.join((processinglocation, 'analytefactors.pickle'))
with open(analytefile, 'rb') as pick:
    analytesbydistribution, distributionsoflines = pickle.load(pick)[2:4]
#analytesbydistribution = {} #distid: analyte id
#distributionsoflines = {} #lineid: distid

linepercentagesofscansfile = '/'.join((processinglocation, 'linepercentagesofscans.pickle'))
with open(linepercentagesofscansfile, 'rb') as pick:
    linepercentagesofscans = pickle.load(pick)
#linepercentagesofscans = defaultdict(lambda: defaultdict(dict)) #scan: line: % of scan intensity input

lineintensitiesofscansfile = '/'.join((processinglocation, 'lineintensitiesofscans.pickle'))
with open(lineintensitiesofscansfile, 'rb') as pick:
    lineintensitiesofscans = pickle.load(pick)
#lineintensitiesofscans = defaultdict(lambda: defaultdict(dict)) #scan: line: total ms1 scan intensity "input"

chargesoflinesfile = '/'.join((processinglocation, 'chargesoflines.pickle'))
with open(chargesoflinesfile, 'rb') as pick:
    chargesoflines = pickle.load(pick)
#chargesoflines = line: charge

#submatchsubformulasfile = '/'.join((processinglocation, 'submatchsubformulas.pickle'))
#with open(submatchsubformulasfile, 'rb') as pick:
#    submatchsubformulas = pickle.load(pick)
##submatchsubformulas = {} #submatchindex: subformula

subformularankfile = '/'.join((processinglocation, 'subformularank.pickle'))
with open(subformularankfile, 'rb') as pick:
    subformularank = pickle.load(pick)
#subformularank = defaultdict(dict) #sequence: subformula: descending subiso rank, lower int = more relevant subiso

subformulapercentfile = '/'.join((processinglocation, 'subformulapercent.pickle'))
with open(subformulapercentfile, 'rb') as pick:
    subformulapeercent = pickle.load(pick)
#subformulapercent = defaultdict(dict) #sequence: subformula: %
#^this will quantify competition across subisos of the same line

#primariesofscansbyindexfile = '/'.join((processinglocation, 'primariesofscansbyindex.pickle'))
#with open(primariesofscansbyindexfile, 'rb') as pick:
#    primariesofscansbyindex = pickle.load(pick)
##indexofprimaryinds = {} #primary: mass index in scan

environment_partial = partial(lmdb.Environment, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=9999, max_spare_txns=1, lock=True)

librarysequences = []
with environment_partial(librarylocation) as env:
    seqdb = '.'.join(('seqsbyformula', proteome))
    seqs = env.open_db(seqdb.encode())
    with env.begin(write=False) as txn:
        with txn.cursor(seqs) as cursor:
            for k, v in cursor:
                key = k.decode()
                value = eval(v.decode())
                #seqsbyformula[key] = value
                for subval in value:
                    #formulasbyseq[subval] = key
                    librarysequences.append(subval)

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

msrun = mzml.MzML(mzmlfile, dtype=np.float64)

scanmasses = {} #scan: [[masses], [intensities]]
for scan, lines in linesofscans.items():
    linesofscans[scan] = tuple(sorted(lines))
    scanmasses[scan] = np.stack((msrun[scan]['m/z array'], msrun[scan]['intensity array']))

headers = ['theoretical_mass', 'ppm_error', 'theoretical_abundance', 'scan_index', 'fragformula', 'scan', 'line', 'sequence', 'subformula', 'ion', 'charge', 'fragrank']

nt = time()

files = [i for i in os.listdir(scanalytelocation) if i.startswith('scanalytes.')]
dmeasures = Counter()
dmaxes = Counter()
linesubcounts = defaultdict(set) #line: [subformulas]
seqmeasures = []
allerrors = []

fragmentcounts = defaultdict(lambda: defaultdict(lambda: Counter())) #ion type: size: partial seq: count
fragmentioncounts = defaultdict(lambda: defaultdict(lambda: Counter())) #ion type: size: partial seq: count
#partialseqcounter = defaultdict(lambda: Counter()) #partialseq: seq: count
distancefragmentcounts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: Counter()))) #ion type: AA: distance away: AA: count
#sequencescanioncounts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: Counter()))) #seq: scan: scanindex: ion type: count ##doesnt show much tbh, everything seems scattershot, like that's what you'll get, even the precursor ions show as many hits as the others (????) despite only having 1 match per sequence
ioncounts = Counter() #ion type: count
sequencecoverage = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))) #seq: analyteid: scan: line-%intensity: [ion-scanindex-partialseqs]
iondata = defaultdict(list) #ion-type: [error, intensity]

rangecount = 4
countrange = range(rangecount)

#partialseqsbylibraryseq = defaultdict(list) #seq: [partialseqs] #takes way too much memory
librarysequencecounts = {} #size: seq: count
for size in countrange:
    size += 1
    librarysequencecounts[size] = Counter(itertools.chain(*[[seq[n:n+size] for n in range(len(seq)-size+1)] for seq in librarysequences]))
    #for lseq in librarysequences:
    #    for pseq in [lseq[n:n+size] for n in range(len(lseq)-size+1)]:
    #        partialseqsbylibraryseq[lseq].append(pseq)

size += 1
librarysequencecounts[size] = Counter(itertools.chain(*[[seq[n:n+size] for n in range(len(seq)-size+1)] for seq in librarysequences]))

sequencecounter = Counter() #seq: count

breaker = False
for f in files:
    ft = time()
    filename = '/'.join((scanalytelocation, f))
    fragmentorganizer = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))))) #scan: analyteid: seq: line: subformula (as submatchindex): ion: charge: fragrank: ppm error
    #fragmentorganizer = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))))) #scan: analyteid: seq: line: ion: fragrank: charge: [extras]
    with open(filename, 'r') as w:
        for i in w.readlines():
            row = i.split(',')
            thmass = float(row[0])
            ppmerror = float(row[1])
            thabundance = float(row[2])
            scanindex = int(row[3])
            fragformula = row[4]
            scan = int(row[5])
            line = int(row[6])
            seq = row[7]
            subformula = row[8]
            ion = row[9]
            charge = int(row[10])
            fragrank = int(row[11].strip())
            analyteid = analytesbydistribution[distributionsoflines[line]]
            exmass, exabundance = scanmasses[scan][:,scanindex]
            flist = [thmass, thabundance, ppmerror, exmass, exabundance, scanindex, fragformula]
            #might want to check if flist is in that thing below, duplicates in fragments now since re-organizing subformulalinegrouping
            fragmentorganizer[scan][analyteid][seq][subformula][ion][charge][line][fragrank].append(flist)
            linesubcounts[line].add(subformula)
    #break
    #for scan, analyteids in fragmentorganizer.items():
    #    alen = len(analyteids)
    #    dmeasures['analyteids'] += alen
    #    if alen > dmaxes['analyteids']:
    #        dmaxes['analyteids'] = alen
    #    for analyteid, seqs in analyteids.items():
    #        slen = len(seqs)
    #        dmeasures['seqs'] += slen
    #        if slen > dmaxes['seqs']:
    #            dmaxes['seqs'] = slen
    #        for seq, lines in seqs.items():
    #            llen = len(lines)
    #            dmeasures['lines'] += llen
    #            if llen > dmaxes['lines']:
    #                dmaxes['lines'] = llen
    #            for line, subformulas in lines.items():
    #                sublen = len(subformulas)
    #                dmeasures['subformula'] += sublen
    #                if sublen > dmaxes['subformula']:
    #                    dmaxes['subformula'] = sublen
    #                for subformula, ions in subformulas.items():
    #                    ilen = len(ions)
    #                    dmeasures['ions'] += ilen
    #                    if ilen > dmaxes['ions']:
    #                        dmaxes['ions'] = ilen
    #                    for ion, fragranks in ions.items():
    #                        flen = len(fragranks)
    #                        dmeasures['fragranks'] += flen
    #                        if flen > dmaxes['fragranks']:
    #                            dmaxes['fragranks'] = flen
    #                        for fragrank, charges in fragranks.items():
    #                            clen = len(charges)
    #                            dmeasures['charges'] += clen
    #                            if clen > dmaxes['charges']:
    #                                dmaxes['charges'] = clen
    #                            for charge, matches in charges.items():
    #                                mlen = len(matches)
    #                                dmeasures['matches'] += mlen
    #                                if mlen > dmaxes['matches']:
    #                                    dmaxes['matches'] = mlen
    for scan, analyteids in fragmentorganizer.items():
        #i think the 1st/2nd/3rd+ priorities can be:
        # - scans with > 1 line with an identifiable seq
        # - scans with 1 line/identifiable seq and non identifiable lines
        # - scans with 1 identifiable line
        
        #you don't need to compete any matches vs non-identified lines/nodist lines yet
        #^NO! i should definitely be using unidentified lines here
        #this is THE COMBINATORICS ROUND!
        #and that includes figuring out ms1 <-> ms2 entropy relations!
        
        #match by line then compete by seq
        #consider all unidentified lines to be a part of the same entity when comparing ms1 <-> ms2 intensities
        matchswitches = defaultdict(lambda: defaultdict(list)) #matchswitch: #[rank for ascending sort, [quant array, becomes numpy array], [string + int list]] ##its a set for now because sequences might have the same subisos, im iterating redundantly
    #    akeys = [tuple(sorted(i.keys())) for i in analyteids.values()]
    #    if set(tuple(sorted(i)) for i in intersection_merge(akeys)) != set(akeys):
    #        print(f, scan)
    #        breaker = True
    #        break
    #if breaker:
    #    break
        for analyteid, seqs in analyteids.items():
            #for seq, lines in seqs.items():
            for seq, subformulas in seqs.items():
                sequencecounter[seq] += 1
                #seqlist = []
                #^i need to move this list to be done at the line level, and compare intensity of the lines to intensities/errors of the matches
                #for line, subformulas in lines.items():
                for subformula, ions in subformulas.items():
                    seqlist = []
                    #for subformula, ions in subformulas.items():
                    for ion, charges in ions.items():
                        #for ion, charges in ions.items():
                        for charge, lines in charges.items():
                            scanindices = ''
                            totalintensity = 0
                            fragcount = 0
                            totalerror = 0
                            #for charge, fragranks in charges.items():
                            for line, fragranks in lines.items():
                                for fragrank, matches in fragranks.items():
                                    for matchlist in matches:
                                        #seqlist.append(matchlist[2])
                                        scanindex = matchlist[5]
                                        #sequencescanioncounts[seq][scan][scanindex][iontype] += 1 #this was the wrong place for this
                                        scanindices += str(scanindex) + '-'
                                        totalintensity += matchlist[4]
                                        totalerror += abs(matchlist[2])
                                        fragcount += 1
                            ionerror = totalerror / fragcount
                            scanindices = scanindices[:-1]
                            stringintensity = str(round(totalintensity, 4))
                            if ion == 'precursor':
                                iontype = 'precursor'
                                label = 'precursor' + '-' + scanindices + '-' + seq + '-' + stringintensity
                            else:
                                iontype = ion[0]
                                ionposition = int(ion[1:])
                                if iontype in 'abc':
                                    label = subformula + '-' + ion + '-' + scanindices + '-' + seq[:ionposition] + '-' + stringintensity
                                else:
                                    label = subformula + '-' + ion + '-' + scanindices + '-' + seq[-ionposition:] + '-' + stringintensity
                            #iondata[iontype].append([ionerror, totalintensity])
                            linelabel = str(line) + '-' + str(round(linepercentagesofscans[scan][line], 4))
                            sequencecoverage[seq][analyteid][scan][linelabel].append(label)
                            ioncounts[iontype] += 1
                            if ion != 'precursor':
                                #we'll just ignore this for now i guess
                                if iontype in 'abc': #nfrags
                                    ioncount = int(ion[1:])
                                    for cr in countrange:
                                        npartialseq = seq[ioncount-cr:ioncount+1]
                                        if len(npartialseq) == cr + 1: #partialseq isn't cut off by the end of the sequence
                                            fragmentcounts['n'][cr+1][npartialseq] += 1
                                            fragmentioncounts[iontype][cr+1][npartialseq] += 1
                                            distancefragmentcounts['n'][npartialseq[0]][cr][npartialseq[-1]] += 1
                                        cpartialseq = seq[ioncount:ioncount+cr+1]
                                        if len(cpartialseq) == cr + 1: #partialseq isn't cut off by the end of the sequence
                                            fragmentcounts['n'][cr+1][cpartialseq] += 1
                                            fragmentioncounts[iontype][cr+1][cpartialseq] += 1
                                            distancefragmentcounts['n'][cpartialseq[-1]][cr][cpartialseq[0]] += 1
                                else: #cfrags
                                    ioncount = len(seq) - int(ion[1:])
                                    for cr in countrange:
                                        npartialseq = seq[ioncount-cr-1:ioncount]
                                        if len(npartialseq) == cr + 1: #partialseq isn't cut off by the end of the sequence
                                            fragmentcounts['c'][cr+1][npartialseq] += 1
                                            fragmentioncounts[iontype][cr+1][npartialseq] += 1
                                            distancefragmentcounts['c'][npartialseq[0]][cr][npartialseq[-1]] += 1
                                        cpartialseq = seq[ioncount:ioncount+cr+1]
                                        if len(cpartialseq) == cr + 1: #partialseq isn't cut off by the end of the sequence
                                            fragmentcounts['c'][cr+1][cpartialseq] += 1
                                            fragmentioncounts[iontype][cr+1][cpartialseq] += 1
                                            distancefragmentcounts['c'][cpartialseq[-1]][cr][cpartialseq[0]] += 1
                    #if len(seqlist) > 1:
                    #    allerrors.extend(seqlist)
                    #    abserrors = np.abs(seqlist)
                    #    absconsistency = np.abs(abserrors - abserrors[:,None]).sum()
                    #    sarray = np.array(seqlist)
                    #    consistency = np.abs(sarray - sarray[:,None]).sum()
                    #    seqmeasures.append([scan, seq, sum(seqlist), abserrors.sum(), len(seqlist), sum(seqlist) / len(seqlist), abserrors.sum() / len(seqlist), absconsistency, np.ptp(seqlist), np.ptp(abserrors), consistency, lineintensitiesofscans[scan][line]])
        #places to explore:
        # - best error match -> single seq
        # - consistency? somehow?
        # - best ms1 entropy match -> for multiple seqs in a scan
    #    if any(len(v) > 1 for v in matchswitches.values()):
    #        print(f, scan)
    #        breaker = True
    #        break
    #if breaker:
    #    break
                            #dist switch: [ascending switch cycles]
                            #match switch: length number: [length number of dist matches]
                        #first, take products of all distributions, dont link different charge states
                        #then, take products of all versions of each distribution, with them even being absent in each permutation, 
                    #line: subformula: ion: fragrank: charge: [matches]
                    #AND you also have to iterate other potential versions of different fragranks
                    #and ignore anything above the charge in chargesoflines
                #FILTER BAD SUBFORMULA RANKS PRIOR TO THIS FILE -> collect false positives?
                #then later when i do inject false positives via seqs, i can collect which of these actually meet dist fragrank and subformula rank requirements and consider them as different layers of false positives?
            #invoke competition across subisotopomers where you can, their effect should be the theoretical % accounted for, right?
        #^when you iterate through each group you get the analyteid as enumerated position
        #then from within each seq grouping you do MORE combinatorics at the dist level
        #intermediaryproducts = defaultdict(list) #matchswitch: [final product lists]
        #for switchkey, matches in matchswitches.items():
        #    fragranks = sorted(matches) #should already be sorted?
        #    for places in list(range(len(fragranks)+1)):
        #        #a blank is included this way, as an option to leave this dist out
        #        intermediarymatches = {i: matches[i] for i in range(places)}
        #        intermediaryproducts[switchkey].append(list(itertools.product(*intermediarymatches.values())))
            #new idea: make a numpy array that included every layer of every dist
            #remove the highest rank frags from the dists 1 at a time through an OTF optimization process, a directed walk that touches on the combinatoric fringes but never goes full tilt until it finds a good direction
            #this will need to happen side by side with the other co-matches of a scan, you can probably use one big array though, and keep track of which row belongs to which analyteid/seq/line/subformula/charge/ion/scanindex etc
            #maybe different optimization processes for error minimization, consistency maximization, subiso competition, and line competition?
            #metrics are error and consistency:
                #simply sort the errors and use this as their priority for the optimization
                #whichever two ADJACENT + DESCENDING (highest must be included) fragranks show the most consistency can be ranked first, and descending ranks from there are allowed to be included in the ranking, and this can compound outwards across dists and frags
            #maybe it doesn't need to be a definite "quantity", you can have higher confidence in part of the IDs and less confidence in others
            #the optimization can be built around the confidence invoked from error/consistency
            #higher intensity should == more accuracy, experimental intensity that is
        #so if combinatorics won't work -> make an optimization process
        #i think i can take all of the matches and put them into a numpy array where i deselect specific indices in order of their error/consistency rank while measuring some kind of value along a curve
        #the one hang up i have is when two different lines match to the same mass, how should i consider that intensity for the entropy process?
        #analyteid: seq: line: switchkey: -> subformula products
    #    analytelist = list(analyteids)
    #    for seqcomb in itertools.product(*(analyteids[i] for i in analyteids)):
    #        for an, seq in enumerate(seqcomb):
    #            aid = analytelist[an]
    #            for line, subformulas in analyteids[analyteid][seq].items():
    #    if len(analyteids) > 1:
    #        #combinatorics
    #    else: #length == 1
    #        #maybe do the seqs 1 at a time then combine them where appropriate?
    #        #rather than hope there isn't redundancy? i mean there probably isn't redundancy
    #        for analyteid, seqs in analyteids.items():
    #            for seq, lines in seqs.items():
    print(time() - ft, f)
    #break

print(time() - nt, 'total')

def better_histogram(numbers, nbins):
    nt = time()
    numbers = np.sort(numbers).tolist()
    binboundaries = np.geomspace(min(numbers), max(numbers), nbins+1)
    print(time() - nt, 'sorting')
    nt = time()
    biniter = iter(binboundaries.tolist())
    outbins = Counter()
    currentbin = next(biniter)
    for n in numbers:
        while True:
            if n > currentbin:
                currentbin = next(biniter)
            else:
                break
        #matchingbin = np.where(n <= binboundaries)[0][0]
        outbins[currentbin] += 1
    print(time() - nt, 'organizing')
    nt = time()
    binedges = sorted(outbins.keys())
    counts = [outbins[edge] for edge in binedges]
    binwidths = np.diff(binedges)
    finalbin = binwidths[-1] / (binwidths[0] / binwidths[1])
    binwidths = binwidths.tolist()
    binwidths.append(finalbin)
    plt.bar(binedges, counts, width=binwidths, align='edge', edgecolor='black', log=True, alpha=0.8)
    plt.xscale('log')
    plt.yscale('log')
    print(time() - nt, 'plotting')
    return outbins

#seqarray = np.array(seqmeasures)
#useqs = np.unique(seqarray, axis=0)
#
##uniform...
#better_histogram(allerrors, 200)
#plt.show()
#
##error sums
#better_histogram(useqs[:,2].astype(float), 200)
#plt.title('error sums')
#plt.show()
#
##absolute error sums
#better_histogram(useqs[:,3].astype(float), 200)
#plt.title('absolute error sums')
#plt.show()
#
##lengths
#better_histogram(useqs[:,4].astype(int), 200)
#plt.title('number of ions per seq')
#plt.show()
#
##average errors
#better_histogram(useqs[:,5].astype(float), 200)
#plt.title('average errors')
#plt.show()
#
##average absolute errors
#better_histogram(useqs[:,6].astype(float), 200)
#plt.title('average absolute errors')
#plt.show()
#
#print('average error:', useqs[:,5].astype(float).mean())
#print('average absolute error:', useqs[:,6].astype(float).mean())
#
##consistency by sum of absolute differences
#better_histogram(useqs[:,7].astype(float), 200)
#plt.title('consistency by sum of absolute differences')
#plt.show()
#
#better_histogram(useqs[:,10].astype(float), 200)
#plt.title('consistency by sum of differences')
#plt.show()
#
##ranges of errors
#better_histogram(useqs[:,8].astype(float), 200)
#plt.title('ranges of errors')
#plt.show()
#
##ranges of absolute errors
#better_histogram(useqs[:,9].astype(float), 200)
#plt.title('ranges of absolute errors')
#plt.show()
#
##ms1 inputs
#better_histogram(np.unique(useqs[:,11].astype(float)), 200)
#plt.title('ms1 inputs')
#plt.show()
#
#labels = ['scan', 'seq', 'sum of errors', 'sum of absolute errors', 'ion matches per seq', 'average error', 'average absolute error', 'sum of absolute differences', 'range of errors', 'range of absolute errors', 'sum of differences', 'ms1 intensity']
##blocked = set([0, 1])
##blockedpairs = set([(2, 3), (5, 6), (8, 9)])
##indices = list(range(len(labels)))
##
###takes more than 24h and i need more blockedpairs for things that are related
##for pair in itertools.combinations(indices, 2):
##    if pair not in blockedpairs:
##        l, r = pair
##        if l not in blocked and r not in blocked:
##            for i in indices:
##                if i not in blocked and i != l and i != r:
##                    c = useqs[:,i].astype(float)
##                    x = useqs[:,l].astype(float)[c.argsort()]
##                    y = useqs[:,r].astype(float)[c.argsort()]
##                    c = np.sort(c)
##                    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(10, 10))
##                    scatter0 = ax[0][0].scatter(x, y, c=c)
##                    scatter1 = ax[0][1].scatter(x, y, c=c, norm=colors.LogNorm())
##                    ax[0][0].set_ylabel(labels[r])
##                    cbar0 = fig.colorbar(scatter0, ax=ax[0][0])
##                    cbar1 = fig.colorbar(scatter1, ax=ax[0][1])
##                    cbar1.set_label(labels[i])
##                    c = useqs[:,i].astype(float)
##                    x = useqs[:,l].astype(float)[c.argsort()[::-1]]
##                    y = useqs[:,r].astype(float)[c.argsort()[::-1]]
##                    c = np.sort(c)[::-1]
##                    scatter2 = ax[1][0].scatter(x, y, c=c)
##                    scatter3 = ax[1][1].scatter(x, y, c=c, norm=colors.LogNorm())
##                    ax[1][0].set_xlabel(labels[l])
##                    ax[1][1].set_xlabel(labels[l])
##                    ax[1][0].set_ylabel(labels[r])
##                    cbar2 = fig.colorbar(scatter2, ax=ax[1][0])
##                    cbar3 = fig.colorbar(scatter3, ax=ax[1][1])
##                    cbar3.set_label(labels[i])
##                    fig.tight_layout()
##                    plt.show()
##                    fig.clf()
##                    plt.close()
##                    gc.collect()
#
##i think everything im doing now is harder than it needs to be
##i can find which factors constrain other factors
##(break clustering still an option)
##then with the constraints, turn towards optimization
##inject false positives
##and combinatorically switch constraints around to determine which constraints are the most important and how to use them
#
#labeldict = {
#        0: 'scan',
#        1: 'seq',
#        2: 'sum of errors',
#        3: 'sum of absolute errors',
#        4: 'ion matches per seq',
#        5: 'average error',
#        6: 'average absolute error',
#        7: 'sum of absolute differences',
#        8: 'range of errors',
#        9: 'range of absolute errors',
#        10: 'sum of differences',
#        11: 'ms1 intensity'
#        }
#
##range of absolute errors x sum of differences by ion matches per seq
#l = 9
#r = 10
#i = 4
#c = useqs[:,i].astype(float)
#x = useqs[:,l].astype(float)[c.argsort()]
#y = useqs[:,r].astype(float)[c.argsort()]
#c = np.sort(c)
#fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(10, 10))
#scatter0 = ax[0][0].scatter(x, y, c=c)
#scatter1 = ax[0][1].scatter(x, y, c=c, norm=colors.LogNorm())
#ax[0][0].set_ylabel(labels[r])
#cbar0 = fig.colorbar(scatter0, ax=ax[0][0])
#cbar1 = fig.colorbar(scatter1, ax=ax[0][1])
#cbar1.set_label(labels[i])
#c = useqs[:,i].astype(float)
#x = useqs[:,l].astype(float)[c.argsort()[::-1]]
#y = useqs[:,r].astype(float)[c.argsort()[::-1]]
#c = np.sort(c)[::-1]
#scatter2 = ax[1][0].scatter(x, y, c=c)
#scatter3 = ax[1][1].scatter(x, y, c=c, norm=colors.LogNorm())
#ax[1][0].set_xlabel(labels[l])
#ax[1][1].set_xlabel(labels[l])
#ax[1][0].set_ylabel(labels[r])
#cbar2 = fig.colorbar(scatter2, ax=ax[1][0])
#cbar3 = fig.colorbar(scatter3, ax=ax[1][1])
#cbar3.set_label(labels[i])
#fig.tight_layout()
#plt.show()
#fig.clf()
#plt.close()
#gc.collect()
#
##avg abs error x sum of differences by ion matches per seq
#l = 6
#r = 10
#i = 4
#c = useqs[:,i].astype(float)
#x = useqs[:,l].astype(float)[c.argsort()]
#y = useqs[:,r].astype(float)[c.argsort()]
#c = np.sort(c)
#fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(10, 10))
#scatter0 = ax[0][0].scatter(x, y, c=c)
#scatter1 = ax[0][1].scatter(x, y, c=c, norm=colors.LogNorm())
#ax[0][0].set_ylabel(labels[r])
#cbar0 = fig.colorbar(scatter0, ax=ax[0][0])
#cbar1 = fig.colorbar(scatter1, ax=ax[0][1])
#cbar1.set_label(labels[i])
#c = useqs[:,i].astype(float)
#x = useqs[:,l].astype(float)[c.argsort()[::-1]]
#y = useqs[:,r].astype(float)[c.argsort()[::-1]]
#c = np.sort(c)[::-1]
#scatter2 = ax[1][0].scatter(x, y, c=c)
#scatter3 = ax[1][1].scatter(x, y, c=c, norm=colors.LogNorm())
#ax[1][0].set_xlabel(labels[l])
#ax[1][1].set_xlabel(labels[l])
#ax[1][0].set_ylabel(labels[r])
#cbar2 = fig.colorbar(scatter2, ax=ax[1][0])
#cbar3 = fig.colorbar(scatter3, ax=ax[1][1])
#cbar3.set_label(labels[i])
#fig.tight_layout()
#plt.show()
#fig.clf()
#plt.close()
#gc.collect()
#
##avg abs error x range of abs errors by ms1 intensity
#l = 6
#r = 9
#i = 11
#c = useqs[:,i].astype(float)
#x = useqs[:,l].astype(float)[c.argsort()]
#y = useqs[:,r].astype(float)[c.argsort()]
#c = np.sort(c)
#fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(10, 10))
#scatter0 = ax[0][0].scatter(x, y, c=c)
#scatter1 = ax[0][1].scatter(x, y, c=c, norm=colors.LogNorm())
#ax[0][0].set_ylabel(labels[r])
#cbar0 = fig.colorbar(scatter0, ax=ax[0][0])
#cbar1 = fig.colorbar(scatter1, ax=ax[0][1])
#cbar1.set_label(labels[i])
#c = useqs[:,i].astype(float)
#x = useqs[:,l].astype(float)[c.argsort()[::-1]]
#y = useqs[:,r].astype(float)[c.argsort()[::-1]]
#c = np.sort(c)[::-1]
#scatter2 = ax[1][0].scatter(x, y, c=c)
#scatter3 = ax[1][1].scatter(x, y, c=c, norm=colors.LogNorm())
#ax[1][0].set_xlabel(labels[l])
#ax[1][1].set_xlabel(labels[l])
#ax[1][0].set_ylabel(labels[r])
#cbar2 = fig.colorbar(scatter2, ax=ax[1][0])
#cbar3 = fig.colorbar(scatter3, ax=ax[1][1])
#cbar3.set_label(labels[i])
#fig.tight_layout()
#plt.show()
#fig.clf()
#plt.close()
#gc.collect()
#
##sum of errors x avg error by sum of differences OR by ion matches per seq -> same pattern
#l = 2
#r = 5
##i = 10
#i = 4
#c = useqs[:,i].astype(float)
#x = useqs[:,l].astype(float)[c.argsort()]
#y = useqs[:,r].astype(float)[c.argsort()]
#c = np.sort(c)
#fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(10, 10))
#scatter0 = ax[0][0].scatter(x, y, c=c)
#scatter1 = ax[0][1].scatter(x, y, c=c, norm=colors.LogNorm())
#ax[0][0].set_ylabel(labels[r])
#cbar0 = fig.colorbar(scatter0, ax=ax[0][0])
#cbar1 = fig.colorbar(scatter1, ax=ax[0][1])
#cbar1.set_label(labels[i])
#c = useqs[:,i].astype(float)
#x = useqs[:,l].astype(float)[c.argsort()[::-1]]
#y = useqs[:,r].astype(float)[c.argsort()[::-1]]
#c = np.sort(c)[::-1]
#scatter2 = ax[1][0].scatter(x, y, c=c)
#scatter3 = ax[1][1].scatter(x, y, c=c, norm=colors.LogNorm())
#ax[1][0].set_xlabel(labels[l])
#ax[1][1].set_xlabel(labels[l])
#ax[1][0].set_ylabel(labels[r])
#cbar2 = fig.colorbar(scatter2, ax=ax[1][0])
#cbar3 = fig.colorbar(scatter3, ax=ax[1][1])
#cbar3.set_label(labels[i])
#fig.tight_layout()
#plt.show()
#fig.clf()
#plt.close()
#gc.collect()

#for ion, counts in fragmentcounts.items():
#    for count, pseqs in counts.items():
#        #x, y = zip(*pseqs.most_common(len(pseqs)))
#        x, y = zip(*pseqs.most_common(100))
#        ts = str(count) + ' - ' + ion
#        #if count <= 3:
#        #these plots are GOOD! they show the anti-conservative behavior
#        #VERY SLOW at > 3 for the full dataset
#        plt.bar(x, y)
#        plt.title(ts)
#        plt.xticks(rotation=90)
#        plt.show()
#        print(ts, 'max value:', x[0], '@', y[0])

partialfrequencies = Counter()
for seq, count in sequencecounter.items():
    for size in countrange:
        size += 1
        for pseq in [seq[n:n+size] for n in range(len(seq)-size+1)]:
            partialfrequencies[pseq] += count

for size in countrange:
    size += 1
    for ion in fragmentioncounts:
        xs = []
        ys = []
        dvals = []
        pseqcount = Counter()
        for pseq, count in fragmentioncounts[ion][size].items():
            freqval = count / partialfrequencies[pseq]
            #xs.append(libval)
            #ys.append(count)
            #pscount[pseq] = count / libval / len(partialseqcounter[pseq])
            pseqcount[pseq] = freqval
            dvals.append(freqval)
            #this is close but its not a lead yet, some pseqs match the same ions across multiple ms1 subformulas, i need to weed that out
        ts = str(size) + ' ' + ion
        #xs = np.array(xs)
        #ys = np.array(ys)
        distvals = np.array(dvals)
        better_histogram(distvals, 100)
        plt.title(ts)
        plt.show()
        #plt.plot(xs, ys, '.')
        #x_min, x_max = plt.xlim()
        #y_min = x_min
        #y_max = x_max
        #plt.plot([x_min, x_max], [y_min, y_max], 'r-', linewidth=2)
        #plt.title(ts)
        #plt.xlabel('library count')
        #plt.ylabel('fragment count')
        #plt.yscale('log')
        #plt.xscale('log')
        #plt.show()

#do different seqs have the same amount of each ion set?
#i need to figure out why they're so god damn even, even y's and b's have a lot of similarity in places

#also look at which sequences are over the 1:1 line and determine if there's a bias into how many times they were searched for or something

#seq: pseq: count
#maybe coverage can be used in the break clusters?

#consistency just across lines in a single scan isn't really any indicator
#but consistency across lines across all scans is a good indicator

#i'm coming to appreciate that even something like collision energy us suspect to being a normal distribution
#its a prenomenon
#some of it comes out harder, some comes out softer
#and i think the acxz ions are justified
#we'll see what the final analysis says about their presence

#for ion, data in iondata.items():
#    x, y = np.array(data).astype(float).transpose()
#    #plt.plot(x, y, '.')
#    #plt.title(ion)
#    #plt.xlabel('ppm error')
#    #plt.ylabel('intensity')
#    #plt.yscale('log')
#    #plt.show()
#    #plt.close()
#    #gc.collect()
#    print(ion, x.mean(), y.mean())
#^the only interesting conclusion was that average precursor intensity was much higher than everything else

#the frequency of the partialseq compared to its existence in the matches
#do what you did to the library to generate seqs but with matches
#account for the number of scans i guess
#use libseq: partialseqs -> += 1, just count seq occurrences and do this after while counting partial seq occurrences from the matches

#~
#i want to directly compare the results of distributions across n/c term frags at each size

for size in countrange:
    size += 1
    for pair in itertools.combinations(fragmentioncounts, 2):
        l, r = pair
        xs = {}
        ys = {}
        xpseqs = set()
        ypseqs = set()
        pseqs = defaultdict(lambda: Counter())
        for pseq, count in fragmentioncounts[l][size].items():
            freqval = count / partialfrequencies[pseq]
            xs[pseq] = freqval
            xpseqs.add(pseq)
            pseqs[l][pseq] = freqval
        for pseq, count in fragmentioncounts[r][size].items():
            freqval = count / partialfrequencies[pseq]
            ys[pseq] = freqval
            ypseqs.add(pseq)
            pseqs[r][pseq] = freqval
        overlappers = xpseqs.intersection(ypseqs)
        pxs = [xs.get(i) for i in overlappers]
        pys = [ys.get(i) for i in overlappers]
        plt.plot(pxs, pys, '.')
        plt.title(str(size) + ' - ' + l + ' - ' + r)
        plt.xlabel(l)
        plt.ylabel(r)
        plt.show()
        plt.close()
        gc.collect()

#pseq: set(scanindices) -> intersection merge these then use the length of the set as the count
#primaryinds
#no you cant do it by primaryind, if 2 seqs have the same pseq and match the same scanindex then it would undercount it

#look at the size-3 plot and extract each AA and do a meta 1/2-size?

for size in countrange:
    size += 1
    for pair in itertools.combinations(fragmentcounts, 2):
        l, r = pair
        xs = {}
        ys = {}
        xpseqs = set()
        ypseqs = set()
        pseqs = defaultdict(lambda: Counter())
        for pseq, count in fragmentcounts[l][size].items():
            freqval = count / partialfrequencies[pseq]
            xs[pseq] = freqval
            xpseqs.add(pseq)
            pseqs[l][pseq] = freqval
        for pseq, count in fragmentcounts[r][size].items():
            freqval = count / partialfrequencies[pseq]
            ys[pseq] = freqval
            ypseqs.add(pseq)
            pseqs[r][pseq] = freqval
        overlappers = xpseqs.intersection(ypseqs)
        pxs = [xs.get(i) for i in overlappers]
        pys = [ys.get(i) for i in overlappers]
        plt.plot(pxs, pys, '.')
        plt.title(str(size) + ' - ' + l + ' - ' + r)
        plt.xlabel(l)
        plt.ylabel(r)
        plt.show()
        plt.close()
        gc.collect()

#when considering the 1, 2 and 3 sized partial sequences
#there's an area under these 3d curves
#most visible on size 3
#interpret the third dimension as density
#i believe this informs on the true ratio of n and c term fragments in the data
#using a different fragmentation methed might show different patterns at this step
#it would be interesting to profile the different methods and energies

#perhaps i can just switch this hypothesis around, which 3-length seqs are likely NOT obviously fragmenters?
