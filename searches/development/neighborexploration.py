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

mzmlfile = '/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'

basefolder = '/'.join((mzmlfile.split('/')[:-2]))
basefile = mzmlfile.split('/')[-1].split('.mzML')[0]

processinglocation = '/'.join((basefolder, 'fileprocessing', basefile))
librarylocation = '/home/sfo/data/proteomics/fastas/search-db'
proteome = 'Human_Homo_sapien'
nprocs = 8
ppmtol = 25
proton = 1.007276554940804


linepositionsbyformulafile = '/'.join((processinglocation, 'linepositionsbyformula.pickle'))
with open(linepositionsbyformulafile, 'rb') as pick:
    linepositionsbyformula = pickle.load(pick)
#linepositionsbyformula = defaultdict(lambda: defaultdict(set)) #formula: matchable library positions of a distribution: [lines that have ms2 scans]

analytefile = '/'.join((processinglocation, 'analytefactors.pickle'))
with open(analytefile, 'rb') as pick:
    analytekeys = pickle.load(pick)[0]
#analytekeys = defaultdict(dict) #analyte id: distid: charge, this can be one or multiple distributions based on whether it has multiple charge states

scanalytefile = '/'.join((processinglocation, 'scanalytes.pickle'))
with open(scanalytefile, 'rb') as pick:
    scanalytecharges = pickle.load(pick)
#scanalytecharges = defaultdict(dict) #analyteid: scan: charge

isotopomerpositionsfile = '/'.join((processinglocation, 'isotopomersbypositions.pickle'))
with open(isotopomerpositionsfile, 'rb') as pick:
    isotopomerpositionsofanalytes = pickle.load(pick)
#isotopomerpositionsofanalytes = defaultdict(set) #analyteid: isotopopmer coordinate from max

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

#environment_partial = partial(lmdb.Environment, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=9999, max_spare_txns=1, lock=True)
#
#getkeys = []
#abundances = {} #formula: [[masses], [intensities]]
#with environment_partial(librarylocation) as env:
#    formuladb = '.'.join(('formulaidentifier', proteome))
#    formulas = env.open_db(formuladb.encode())
#    with env.begin(write=False) as txn:
#        with txn.cursor(formulas) as cursor:
#            for k, v in cursor:
#                getkeys.append(k)
#    fulldb = env.open_db('distributions.full'.encode())
#    with env.begin(write=False) as txn:
#        with txn.cursor(fulldb) as cursor:
#            for k, v in cursor.getmulti(getkeys):
#                out = np.frombuffer(v)
#                out = out.reshape(2, out.size//2)
#                abundances[k.decode()] = out


#maybe linesofscans and scansoflines would be useful?

nt = time()
msrun = mzml.MzML(mzmlfile, dtype=np.float64)

masslist = [] #[target mass, scan]
massdict = {} #scan: [masses]
intensitydict = {} #scan: [intensities]
for scan in msrun:
    if scan['ms level'] == 2:
        ind = scan['index']
        massdict[ind] = scan['m/z array'][:,None]
        intensitydict[ind] = scan['intensity array']
        m = scan['precursorList']['precursor'][0]['isolationWindow']['isolation window target m/z'].real
        masslist.append([m, ind])

masslist = sorted(masslist)
print(time() - nt, 'mass info')

nt = time()
#as an initiating concept, keep track of which distributions can be used as alternate comparisons for each individual distribution's group of scans
#making the lists that determine which alternate dists can be used as counterfactuals
masswindow = 2
current = []
overlaps = defaultdict(list) #scan [scans within an overlapping mass range]
for m, ind in masslist:
    removals = []
    #ind = massinds[m]
    for cm, c in current:
        #cm = masstargets[c]
        diff = m - cm
        if diff >= masswindow:
            removals.append([cm, c])
        else:
            overlaps[ind].append(c)
            overlaps[c].append(ind)

    for r in removals:
        current.remove(r)
    current.append([m, ind])

#for i, o in overlaps.items():
#    m = masstargets[i]
#    l = len(o)
#    plt.plot(m, l, '.', color='white', alpha=0.4)
#plt.show()

print(time() - nt, 'overlap windows')

allsamples = set()
analytesofscans = defaultdict(set) #scan: [analyteids]
mergedscans = set()
scangroupsbyscan = defaultdict(set)
analyteidsbyscangroup = defaultdict(set) #scangroup: {analyteids}
for formula, sample in spectrabyformula.items():
    for analyteid, sids in sample.items():
        tsids = tuple(sorted(sids))
        chargelists = defaultdict(list)
        analyteidsbyscangroup[tsids].add(analyteid)
        allsamples.update(sids)
        for sid in sids:
            chargelists[scanalytecharges[analyteid][sid]].append(sid)
            analytesofscans[sid].add(analyteid)
        for sids in chargelists.values():
            tsids = tuple(sorted(sids))
            mergedscans.add(tsids)
            for sid in sids:
                scangroupsbyscan[sid].add(tsids)

mergedscans = tuple(mergedscans)

nocompetition = set()
hascompetition = set()
overlappingscangroups = defaultdict(set) #scangroup: {other scangroups}
for sids in mergedscans:
    for sid in sids:
        for outersids in scangroupsbyscan[sid]:
            if outersids != sids:
                overlappingscangroups[sids].add(outersids)
                hascompetition.update(sids)
                hascompetition.update(outersids)

#for scans without entropic competition, you don't need to compare any NN stuff
#^only need to compare the relevant combinations


#list off sids by ads while keeping track of which are scan-overlapping vs window overlapping
#compare the distribution of similarities
#within the distribution of window overlaps of all scans of all distributions in a linked group @ a charge, make the display of the scan overlaps distinct

nt = time()
output = {} #pair: [rank diffs, # matches]
seenpairs = set()
for sid in allsamples:
    masses = massdict[sid]
    intensities = intensitydict[sid]
    #neighbors[sid] = spatial.KDTree(masses)
    neighbor = spatial.KDTree(masses)
    #compare to every spectra within overlapping mass window rather than every spectra
    for o in overlaps[sid]:
        sortedkey = tuple(sorted((o, sid)))
        if sortedkey not in seenpairs:
            om = massdict[o]
            oi = intensitydict[o]
            fragtol = (om / 1000000 * ppmtol).flatten()
            dists, inds = neighbor.query(om)
            tolerated = dists <= fragtol
            matches = om[tolerated]
            ranks = intensities[inds[tolerated]].argsort().argsort()
            oranks = oi[tolerated].argsort().argsort()
            diffs = np.abs(ranks - oranks).sum()
            seenpairs.add(sortedkey)
            output[sortedkey] = [diffs, len(matches)]
print(time() - nt, 'comparisons')

#below is just to show percentiles, its not a useful analysis for the actual engine imo
nt = time()
scancomparisons = []
for sample in spectrabyformula.values():
    for analyteid, sids in sample.items():
        outdiffs = defaultdict(list)
        outsums = defaultdict(list)
        for sid in sids:
            background = overlaps[sid]
            for b in background:
                sortedkey = tuple(sorted((b, sid)))
                out = output[sortedkey]
                if b not in sids:
                    outdiffs[sid].append(out[1])
                    outsums[sid].append(out[0])
        for k, v in outdiffs.items():
            outdiffs[k] = np.array(v)
        for k, v in outsums.items():
            outsums[k] = np.array(v)
        for pair in itertools.combinations(sids, 2):
            l, r = pair
            sortedkey = tuple(sorted(pair))
            if scanalytecharges[analyteid][l] == scanalytecharges[analyteid][r]:
                try:
                    indiff, insum = output[sortedkey]
                except KeyError: #overlap > masswindow
                    m = massdict[l]
                    intensities = intensitydict[l]
                    neighbor = spatial.KDTree(m)
                    om = massdict[r]
                    oi = intensitydict[r]
                    fragtol = (om / 1000000 * ppmtol).flatten()
                    dists, inds = neighbor.query(om)
                    tolerated = dists <= fragtol
                    matches = om[tolerated]
                    ranks = intensities[inds[tolerated]].argsort().argsort()
                    oranks = oi[tolerated].argsort().argsort()
                    indiff = np.abs(ranks - oranks).sum()
                    insum = len(matches)
                #percentile calculations
                #bigger number is better for all percentiles, aka closer to 1
                diffpercentile = (indiff < outdiffs[l]).sum() / outdiffs[l].size
                sumpercentile = (insum > outsums[l]).sum() / outsums[l].size
                scancomparisons.append([analyteid, l, diffpercentile, sumpercentile])
                diffpercentile = (indiff < outdiffs[r]).sum() / outdiffs[r].size
                sumpercentile = (insum > outsums[r]).sum() / outsums[r].size
                scancomparisons.append([analyteid, r, diffpercentile, sumpercentile])
    #compare across analyteid's here? use sids as a key?
print(time() - nt, 'percentiles')

scancomparisons = pd.DataFrame(scancomparisons)
scancomparisons.columns = ['analyteid', 'sid', 'diffperc', 'sumperc']
scancomparisons = scancomparisons.drop_duplicates()

#this one is properly bimmodal
#it would be good to check proton-window shifting with this, to see if the positive results improve
scancomparisons.loc[:,'diffperc'].plot.hist(bins=100)
plt.yscale('log')
plt.show()

#might just barely be bimodal, can't really see on the data used for this exploration but there's a decent bump at the end near 1 that gives enough hope
scancomparisons.loc[:,'sumperc'].plot.hist(bins=100)
plt.yscale('log')
plt.show()

#seenpairs original spot
#scancomparisons.shape
#Out[2]: (17104476, 4)
#scancomparisons.drop_duplicates().shape
#Out[3]: (92033, 4)
#^same without it, it was in fact not doing anything
#next put seen above all the loops and block any seenpairs for everything


#on assessing the rank diffs and match counts:
#if you have like 2 matches, and the ranks are perfect, this is easily going to be error prone
#or rather:
#   > low matches with perfect rank alignment is a bad sign
#   > high matches with bad rank alignment is 2nd best
#   > high matches with good rank alignment is best
#i also want to compare scan groups of other analyteid's within the same formula-match
#   > if these are just ~average, and therefore easy to distinguish from one another, then that's a good thing for the ID matching process i guess
#what's annoying is that there are probably different formulas with the same distribution rank order, or just for whatever coverage there is, as well as the sample analyteid/scanid combinations
#^organize it like this, by analyteid and scanid?
#^can you potentially note which scans are more similar and use that as a way to differentiate from slightly different matches? ie if one analyteid/scanid combination is a superset of another, can you use any kind of reasoning to allow or deny the super/subset?
#organize as {linkedgroup: [scanids]} and {scanid: [analyteids]}?



#question: do analyteid/charge groups change due to some scan behavior?
#lengthtest = defaultdict(lambda: Counter())
#for formula, sample in spectrabyformula.items():
#    for analyteid, sids in sample.items():
#        lengthtest[analyteid][sids] += 1
#
#lengthcounts = Counter()
#for analyteid, counts in lengthtest.items():
#    clen = len(counts)
#    lengthcounts[clen] += 1
##lengthcounts
##Out[23]: Counter({1: 10176})
#answer: no

#^meaning, there's a lot of redundant information in spectrabyformula, you can index scangroups in the future for a memory reduction



#intersection merge the scanids, then intersection merge the analyteid's based on scanid groups as 2 layers of connection/information?
#^i think these groups will be exactly the same
#instead i want to both merge every group of sids in sample and not. these two layers should be interesting?
#^i also want to keep a charge-inclusive non-merged group to compare to others
#i want to be able to figure out whether the final merged groups have added a TON of other analyteids or scans into their group.
#i think i'll also need to only group these at the same charge state, currently non charge state matching scanids are readily accessible

scangroups = []
mergedscans = []
basescangroups = []
basemergedscans = []
lengthcounter = Counter()
analyteidsbyscangroup = defaultdict(set)
#formulasbyanalyteid = defaultdict(set)
scangroupsbyscan = defaultdict(set)
for formula, sample in spectrabyformula.items():
    chargelists = defaultdict(lambda: defaultdict(list))
    for analyteid, sids in sample.items():
        tsids = tuple(sids)
        analyteidsbyscangroup[tsids].add(analyteid)
        #formulasbyanalyteid[analyteid].add(formula)
        for sid in sids:
            chargelists[analyteid][scanalytecharges[analyteid][sid]].append(sid)
            scangroupsbyscan[sid].add(tsids)
    mergemids = defaultdict(list)
    for subsample in chargelists.values():
        for charge, sids in subsample.items():
            scangroups.append(sids)
            mergemids[charge].extend(sids)
    for sids in mergemids.values():
        mergedscans.append(sids)
    basescangroups.extend(sample.values())
    basemergedscans.append(list(itertools.chain(*sample.values())))

scangroups = list(map(tuple, scangroups))
mergedscans = list(map(tuple, mergedscans))
basescangroups = list(map(tuple, basescangroups))
basemergedscans = list(map(tuple, basemergedscans))

allscans = list(set(itertools.chain(*scangroups)))

groups = defaultdict(set)
for bsg in basescangroups:
    for g in bsg:
        groups[g].add(bsg)

#~384mil combinations if i compare every spectra to every other one

#i want to be able to correct the masses for any slight positional shifts within the ms1 targeting of an ms2 scan, ie if something targets the 2nd as opposed to the 1st or 3rd isotopomer of an ms1 distribution. i'd want to be able to +/- shift the ms2 fragment ions to pre-normalize this before doing the NN, but I think this might pose a few problems?

#collect # matches
#sum of rank intensity differences across each matched ion, the rank array should be sorted by mass

#~
#compare scans of distributions
#compare the amount NON-matched from either scan -> % matched?, % has its own pitfalls here but i might add it
#compare formula matches across analyteid's, each group of scanids should still be more related to the differentiable analyteids of either group
#compare groups of formula analyteid's that overlap in composition, as determined from the intersection merger above
#   > if the same exact {analyteid: [scan group]} is present for another formula, they can be compared. it means these theoretical distributions have the same relative ranks
#   > different [scan groups] do not need to be compared when they overlap. when these occur they indicate different distributions are overlapping on certain scans. what you can gain from comparing these is confidence that the two distributions within the same scans are differentiable. the metric should indicate HOW differetiable. compare the entirety of the pool of what ions were matched in either scangroup, sort and organize them into a new NN model then see how much either relate to each other. might this be a good way to pre-organize matches? i'd also like to compare the number of matches to the relative intensities of either distribution. perhaps whichever peptides match these pre-conditional biases best might be scored better?
#perhaps scans that should share a distribution but are getting low percentile matches indicates a bad charge-state assessment. definitely likely to mean incoming bad data, charge-state is one potential route i suppose, only one i can think of off the top though.


#~
#do i care to compare:
#different formulas across an analyteid? - not here, that's what the other part of the engine is for, it should rank which spectra match best
#different analyteids across a formula? - #same answer as above
#different sid-groups across an overlapping sid group - yes, this is where i can get an initial estimate of entropy and what matches should make sense when overlapping distributions are present
#   > for each distribution/analyteid, for all the sids, any other sidgroup of another distribution is referenced in its entirety based on the distribution it represents
#   > apply intensity-based weighted-averaging of matching masses within a distribution -> this will make up a centralized mass dataset of that distribution
#   > with the central masses, compare the matches of each distribution to those of the scans. do the non-overlapping scans provide more entropy for their respective distributions? ie do the masses that match in the non-overlapping scans provide a better match of what's still unmatched in the scan-pool? and does the extra summed intensity from those matches better match the intensity samplings?
#   > for redundant matches, the summed intensity when its included in both vs excluded in both is the confidence interval on the entropy?

#the output will then be broken into which scans can be compared via entropic competition, and which ones needs to rely purely on identification tactics.
#when something with multiple sampled charge states with entropic competition, these ought to be linked somehow, but that's for the other part of the engine, here they can just be assessed at the independent charge states
