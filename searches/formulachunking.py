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

#if not librarylocation.endswith('/'):
#    librarylocation = librarylocation + '/'
#fragmentlocation = ''.join((librarylocation, 'fragments/'))
#if not os.path.isdir(fragmentlocation):
#    os.mkdir(fragmentlocation)

spectrabyformulafile = '/'.join((processinglocation, 'spectrabyformula.pickle'))
with open(spectrabyformulafile, 'rb') as pick:
    spectrabyformula = pickle.load(pick)
#spectrabyformula = defaultdict(lambda: defaultdict(set)) #formula: analyteid: scan

environment_partial = partial(lmdb.Environment, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=9999, max_spare_txns=1, lock=True)

seqsbyformula = {} #formula: [seqs]
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

def scan_counting(scan):
    if scan['ms level'] == 2:
        nions = len(scan['m/z array'])
        ind = scan['index']
        return {ind: nions}
    else:
        return {}

#counting the number of ions in each ms2 scan
scancounts = {}
for output in msrun.map(lambda scan: scan_counting(scan), processes=nprocs):
    scancounts.update(output)

totalions = sum(scancounts.values())

#making a somewhat arbitrary measure of complexity for each group in spectabyformula
formulacounts = {} #formula: sum(nions from all scans) * nseqs * avg seqlength
for formula, sample in spectrabyformula.items():
    seqs = seqsbyformula[formula]
    avgseqlen = np.mean([len(i) for i in seqs])
    nseqs = len(seqs)
    #nscans = len(list(sample.values())[0])
    scanions = 0
    for analyteid, sids in sample.items():
        for sid in sids:
            scanions += scancounts[sid]
    count = scanions * nseqs * avgseqlen
    formulacounts[formula] = count

countvalues = np.sort(list(formulacounts.values()))
totalcounts = countvalues.sum()

#reverse organization
formulasbycount = defaultdict(list)
for formula, count in formulacounts.items():
    formulasbycount[count].append(formula)

#determine the number of bins based on the initial test dataset as a baseline
#totalcounts = 55123176624 for an 8gb fragmatchlist, make 1gb chunks, therefore make 8 final groups here
#bin != final group
#make the same number of bins, 8. then subdivide each bin 8 times and make the subdivisions ~even in their subdivision total counts
#pair subidivions from each bin together in a way that minimizes their overall difference in division total count across all final groups

basis = 55530938547 #this value is also going to shift depending on the hardware
#^keep this as a standard but also include an option for this to be modified via script input, some people might have denser/richer data and an input here would be used to prevent memory problems.
basisbins = 8

#ratio to adjust the scale
adjustment = totalcounts / basis
nbins = round(basisbins * adjustment)
ndivisors = nbins - 1

#current output is an 8gb fragment list, i want to split them into 1gb pieces assuming all my counts here equal to that, i can use that as a basis for dividing future files
#^might go wrong b/c i'm not representing ms2 ions here at all, but meh what else can i do atm
#totalcounts / 8gb = 6941367318.375
#i guess i can incorporate the number of ms2 scans here too
#totalcounts / totalions = 4.659340143007824
#incorporate system memory too
#psutil.virtual_memory().total
#i'm not going to care about available atm i guess? this would be a good spot to though, given that this single script is really light
#maybe count the ions in each scan and use that?

#use a 20% safety margin, or base the safety margin on the density of the ms2 scans? ions per scan would come in handy here
#divide into groups that have an even spread of each number of counts
#^so group the counts into ~clusters and distribute the clusters evenly
#also order the most abundant counts to be evenly spaced out within each division

#minimization of differences:
# - keep each bin/division as a sorted list
# - from the bin with the largest amount of sum total distance from all of the other bins:
#   > if this bin also has the least amount of members in it, then favor taking large numbers to fix the difference
#   > if this bin also has the most amount of members in it, favor taking groups of smaller numbers to fix the difference
#  -> subtract these members and distribute them amongst the other bins in a way that minimzes the sum total difference. AND the original bin they came from is still a candidate for where they can go? probably won't need that given the assessment of what bins leave will probably account for this.

minbinbound = min(countvalues) - 1
maxbinbound = max(countvalues) + 1
binsize = (maxbinbound - minbinbound) / nbins
binboundaries = minbinbound + (np.arange(ndivisors) + 1) * binsize
binboundaries = np.insert(binboundaries, 0, minbinbound)
binboundaries = np.insert(binboundaries, binboundaries.shape[0], maxbinbound)
pairedboundaries = np.stack((binboundaries[:-1], binboundaries[1:]), axis=1)

binnedgroups = {} #bin: countvalue
for n, (l, r) in enumerate(pairedboundaries.tolist()):
    binnedgroups[n] = countvalues[np.logical_and(countvalues >= l, countvalues < r)][::-1] #reverse the sorting for future iterations

#for k, v in binnedgroups.items():
#    print(k, len(v))
#0 432169
#1 1709
#2 283
#3 83
#4 33
#5 11
#6 5
#7 1
#there will always be more on the lower side it seems?

#so i'll start from the bottom while dividing out what there is
#maybe i can start organizing from the top to figure out what needs to be distributed?
#i can keep the total count, or the total count of each bin, handy, in order to leave voids that can be filled as i go up the bins
#actually yeah, maybe i should start from the top and use the bottom pieces to make up for the differences, that might be easier than distributing the largest things last.
#^this does heavily rely on there being more of the lower bins, always, so i should print this as output from the engine and put in a check for if this ever doesn't happen.
#as you go through the divisions, add things from the current bin into the division with the lowest sum, and iterate through the bin's countvalues in order from highest to lowest
#the idea of iterating the counts in descending order is based on the idea that the differences between divisions should get lower and lower by doing this

divisions = {k: [] for k in binnedgroups}
dsums = np.array([sum(v) for v in divisions.values()])
for b, counts in reversed(binnedgroups.items()):
    for c in counts.tolist():
        dkey = dsums.argmin()
        divisions[dkey].append(c)
        dsums[dkey] += c

#order the divisions to spread out the large ones amongst the smaller
#pad the largest values based on nprocs i suppose

#start with the highest - they're already ordered like this
#iterate all the potential iteration spots, add an n+1 for each insertion, bisect
ordereddivisions = defaultdict(list)
for k, div in divisions.items():
    ordereddivisions[k].insert(0, div.pop(0))
    while div:
        n = 0
        try:
            for c in range(len(ordereddivisions[k])+1):
                ordereddivisions[k].insert(c+n, div.pop(0))
                n += 1
        except IndexError:
            break

#explanation
#for k, v in ordereddivisions.items():
#    plt.plot(v, '-')
#    plt.show()

#converting the counts into formulas and distributing them across divisions
dividedformulas = defaultdict(list) #divisionkey: [formulas]
for k, div in ordereddivisions.items():
    for c in div:
        dividedformulas[k].append(heapq.heappop(formulasbycount[c]))

dividedformulas = {k: tuple(v) for k, v in dividedformulas.items()}

divisionfile = '/'.join((processinglocation, 'dividedformulas.pickle'))
with open(divisionfile, 'wb') as pick:
    pickle.dump(dividedformulas, pick)



#a higher level of automating for the memory management:
#make one big order to spread out memory
#iterate the entire list, and measure when memory > some value OTF?



#i think this binning+division concept might make a for a good excercise for kids, and maybe the counts concept belongs in there too

#a random tidbit:
#a histogram of countvalues, and a straight plot of them give an inverse relationship
#in the histogram its the minor values that skyrocket in abundance
#but in the raw line chart, the latter values have a much higher magnitude
#in terms of MS data in think it might be interesting to compare the ratio of the binned lower abundance of this kind of distribution vs the exponential nature of the higher values
#^and this doesn't have to be done here at this step? but i think it also can too, i think in general this might be a worthwhile kind of analysis to make, and i just need to figure out where to insert it.


#this file is deprecated
#i don't use the exact strategy in this file anymore because i no longer chunk spectrabyformula
#i chunk mergablesequences now
