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
#lineintensitiesofscans = defaultdict(lambda: defaultdict(dict)) #scan: line: % of scan intensity input

scanmassesfile = '/'.join((processinglocation, 'scanmasses.pickle'))
with open(scanmassesfile, 'rb') as pick:
    scanmasses = pickle.load(pick)
#scanmasses = {} #scan: [[masses], [intensities]]

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

sumintensitiesofscans = {} #scan: total intensity
for scan, (masses, intensities) in scanmasses.items():
    sumintensitiesofscans[scan] = intensities.sum()

headers = ['theoretical_mass', 'ppm_error', 'theoretical_abundance', 'scan_index', 'fragformula', 'scan', 'line', 'sequence', 'subformula', 'ion', 'charge', 'fragrank']

nt = time()

files = [i for i in os.listdir(scanalytelocation) if i.startswith('scanalytes.')]
dmeasures = Counter()
dmaxes = Counter()
linesubcounts = defaultdict(set) #line: [subformulas]
seqmeasures = []
allerrors = []

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
            fragmentorganizer[scan][analyteid][seq][line][subformula][ion][charge][fragrank].append([thmass, thabundance, ppmerror, exmass, exabundance, scanindex, fragformula])
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
        for analyteid, seqs in analyteids.items():
            for seq, lines in seqs.items():
                seqlist = []
                for line, subformulas in lines.items():
                    for subformula, ions in subformulas.items():
                        for ion, charges in ions.items():
                            for charge, fragranks in charges.items():
                                #switchkey = subformula + '-' + ion + '-' + str(charge)
                                #if switchkey not in matchswitches:
                                #    for fragrank, matches in fragranks.items():
                                #        for matchlist in matches:
                                #            switchlist = [fragrank, matchlist[:5], matchlist[5:]]
                                #            #if switchlist not in matchswitches[switchkey]:
                                #            matchswitches[switchkey][fragrank].append(switchlist)
                                for fragrank, matches in fragranks.items():
                                    for matchlist in matches:
                                        seqlist.append(matchlist[2])
                if len(seqlist) > 1:
                    allerrors.extend(seqlist)
                    abserrors = np.abs(seqlist)
                    absconsistency = np.abs(abserrors - abserrors[:,None]).sum()
                    sarray = np.array(seqlist)
                    consistency = np.abs(sarray - sarray[:,None]).sum()
                    seqmeasures.append([scan, seq, sum(seqlist), abserrors.sum(), len(seqlist), sum(seqlist) / len(seqlist), abserrors.sum() / len(seqlist), absconsistency, np.ptp(seqlist), np.ptp(abserrors), consistency])
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

seqarray = np.array(seqmeasures)
useqs = np.unique(seqarray, axis=0)

#uniform...
better_histogram(allerrors, 200)
plt.show()

#error sums
better_histogram(useqs[:,2].astype(float), 200)
plt.title('error sums')
plt.show()

#absolute error sums
better_histogram(useqs[:,3].astype(float), 200)
plt.title('absolute error sums')
plt.show()

#lengths
better_histogram(useqs[:,4].astype(int), 200)
plt.title('number of ions per seq')
plt.show()

#average errors
better_histogram(useqs[:,5].astype(float), 200)
plt.title('average errors')
plt.show()

#average absolute errors
better_histogram(useqs[:,6].astype(float), 200)
plt.title('average absolute errors')
plt.show()

print('average error:', useqs[:,5].astype(float).mean())
print('average absolute error:', useqs[:,6].astype(float).mean())

#consistency by sum of absolute differences
better_histogram(useqs[:,7].astype(float), 200)
plt.title('consistency by sum of absolute differences')
plt.show()

better_histogram(useqs[:,10].astype(float), 200)
plt.title('consistency by sum of differences')
plt.show()

#ranges of errors
better_histogram(useqs[:,8].astype(float), 200)
plt.title('ranges of errors')
plt.show()

#ranges of absolute errors
better_histogram(useqs[:,9].astype(float), 200)
plt.title('ranges of absolute errors')
plt.show()

#seqmeasures.append([scan, seq, sum(seqlist), abserrors.sum(), len(seqlist), sum(seqlist) / len(seqlist), abserrors.sum() / len(seqlist), consistency, np.ptp(seqlist), np.ptp(abserrors)])

labels = ['scan', 'seq', 'sum of errors', 'sum of absolute errors', 'ion matches per seq', 'average error', 'average absolute error', 'sum of absolute differences', 'range of errors', 'range of absolute errors', 'sum of differences']
blocked = set([0, 1])
blockedpairs = set([(2, 3), (5, 6), (8, 9)])
indices = list(range(len(labels)))

for pair in itertools.combinations(indices, 2):
    if pair not in blockedpairs:
        l, r = pair
        if l not in blocked and r not in blocked:
            plt.plot(useqs[:,l].astype(float), useqs[:,r].astype(float), '.')
            plt.xlabel(labels[l])
            plt.ylabel(labels[r])
            plt.show()
