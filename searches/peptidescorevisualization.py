import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from colour import Color
from pyteomics import mzml
from time import time
import pandas as pd
import heapq
from bisect import bisect
import gc
import concurrent.futures
import multiprocessing as mp
from collections import Counter, defaultdict
from textwrap import wrap
from scipy import sparse, integrate, spatial, stats, special, signal
from pandas.api.types import CategoricalDtype
from sklearn.neighbors import NearestNeighbors
from distinctipy import distinctipy as dp
from functools import partial
import math
import lmdb
import random
import itertools
import pickle
import sys
import os
import warnings
#warnings.filterwarnings("error")
np.set_printoptions(suppress=True)
#np.warnings.filterwarnings('ignore')
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

librarylocation = '/home/sfo/data/proteomics/fastas/search-db/'
proteome = 'Human_Homo_sapien'
processingdirectory = '/home/sfo/store/flowcharacterizations/round3/fileprocessing/200901_fR_400/'
ppmtol = 25
ions = 'by'

peptidefilename = processingdirectory + 'peptiderankings.csv'
distributionfilename = processingdirectory + 'distributionrankings.csv'
scanfilename = processingdirectory + 'linerankings.csv'
scanfilename = processingdirectory + 'scanrankings.csv'
subformulafilename = processingdirectory + 'subformularankings.csv'
ppmmod = ppmtol / 1000000
countrange = 3 #number of flanking AAs to cluster by

divisionfile = ''.join((processingdirectory, 'dividedgroups.pickle'))
with open(divisionfile, 'rb') as pick:
    dividedgroups = pickle.load(pick)

distributionchargesfile = ''.join((processingdirectory, 'distributioncharges.pickle'))
with open(distributionchargesfile, 'rb') as pick:
    distributioncharges = pickle.load(pick)
#distributioncharges = line: charge

distributionsoflinemasksfile = ''.join((processingdirectory, 'distributionsoflinemasks.pickle'))
with open(distributionsoflinemasksfile, 'rb') as pick:
    distributionsoflinemasks = pickle.load(pick)
#distributionsoflinemasks = {} #linemask: distid

linemasksbyscanbysubformulafile = ''.join((processingdirectory, 'linemasksbyscanbysubformula.pickle'))
with open(linemasksbyscanbysubformulafile, 'rb') as pick:
    linemasksbyscanbysubformula = pickle.load(pick)
#linemasksbyscanbysubformula = {} #subformula: scan: [lines]

scoredscansfile = ''.join((processingdirectory, 'scored.ms2.pickle'))
with open(scoredscansfile, 'rb') as pick:
    scoredms2scans = pickle.load(pick)
#scoredms2scans = {} #scan: [[masses], [intensities], [ion scores]]

intensityaverages = {} #scan: average intensity
for scan, (masses, intensities, scores) in scoredms2scans.items():
    intensityaverages[scan] = np.mean(intensities)

elementsofprobindicesfile = ''.join((processingdirectory, 'elementsofprobabilityindices.pickle'))
with open(elementsofprobindicesfile, 'rb') as pick:
    elementsofprobabilityindices = pickle.load(pick)
#elementsofprobabilityindices = {} #prob index: e

probabilityorganizerfile = ''.join((processingdirectory, 'probabilityorganizer.pickle'))
with open(probabilityorganizerfile, 'rb') as pick:
    probabilityorganizer = pickle.load(pick)
#probabilityorganizer = defaultdict(dict) #prob index: iso: prob

matchprobfile = ''.join((processingdirectory, 'matchprobabilities.pickle'))
with open(matchprobfile, 'rb') as pick:
    matchprobabilities = pickle.load(pick)
#matchprobabilities = defaultdict(list) #subformula: [prob indices]

subformulasubindsfile = ''.join((processingdirectory, 'subformulasubindices.pickle'))
with open(subformulasubindsfile, 'rb') as pick:
    subformulasubindices = pickle.load(pick)
#subformulasubindices = defaultdict(list) #subformula: [sub match indices]

submatchsequencesfile = ''.join((processingdirectory, 'submatchsequences.pickle'))
with open(submatchsequencesfile, 'rb') as pick:
    submatchsequences = pickle.load(pick)
#submatchsequences = {} #submatchindex: sequence

subformulapercentfile = ''.join((processingdirectory, 'subformulapercent.pickle'))
with open(subformulapercentfile, 'rb') as pick:
    subformulapercent = pickle.load(pick)
#subformulapercent = defaultdict(dict) #subformula: sequence: (subiso abundance rank, subiso abundance)

maxintensitylinesofdistsfile = ''.join((processingdirectory, 'maxintensitylinesofdists.pickle'))
with open(maxintensitylinesofdistsfile, 'rb') as pick:
    maxintensitylinesofdists = pickle.load(pick)
#maxintensitylinesofdists = defaultdict(dict) #distid: (line, scan)

subformulasofsequencedistributionfile = ''.join((processingdirectory, 'subformulasofsequencedistribution.pickle'))
with open(subformulasofsequencedistributionfile, 'rb') as pick:
    subformulasofsequencedistribution = pickle.load(pick)
#subformulasofsequencedistribution = {} #seq: dist: subformula

#lineintensitiesofscansfile = ''.join((processingdirectory, 'lineintensitiesofscans.pickle'))
#with open(lineintensitiesofscansfile, 'rb') as pick:
#    lineintensitiesofscans = pickle.load(pick)
##lineintensitiesofscans = defaultdict(lambda: defaultdict(dict)) #scan: line: raw area of ms1 line, average of 2 flanking points

linemaskpositionsbyformulafile = ''.join((processingdirectory, 'linemaskpositionsbyformula.pickle'))
with open(linemaskpositionsbyformulafile, 'rb') as pick:
    linemaskpositionsbyformula = pickle.load(pick)
#linemaskpositionsbyformula = defaultdict(lambda: defaultdict(set)) #formula: matchable library positions of a distribution: [lines that have ms2 scans]

chargesoflinemasksfile = ''.join((processingdirectory, 'chargesoflinemasks.pickle'))
with open(chargesoflinemasksfile, 'rb') as pick:
    chargesoflinemasks = pickle.load(pick)

linesofscansfile = ''.join((processingdirectory, 'linesofscans.pickle'))
with open(linesofscansfile, 'rb') as pick:
    linesofscans = pickle.load(pick)

seqswithdecoysbyformulafile = ''.join((processingdirectory, 'seqswithdecoysbyformula.pickle'))
with open(seqswithdecoysbyformulafile, 'rb') as pick:
    seqswithdecoysbyformula = pickle.load(pick)
#seqswithdecoysbyformula = defaultdict(list) #formula: [seqs + decoys]

distributionsoflinemasksfile = ''.join((processingdirectory, 'distributionsoflinemasks.pickle'))
with open(distributionsoflinemasksfile, 'rb') as pick:
    distributionsoflinemasks = pickle.load(pick)
#distributionsoflinemasks = {} #linemask: distid

linesbylinemaskfile = ''.join((processingdirectory, 'linesbylinemask.pickle'))
with open(linesbylinemaskfile, 'rb') as pick:
    linesbylinemask = pickle.load(pick)

positionsbyformulabyline = defaultdict(lambda: defaultdict(set))
for formula, positions in linemaskpositionsbyformula.items():
    for position, lines in positions.items():
        for line in lines:
            positionsbyformulabyline[line][formula].add(position)

environment_partial = partial(lmdb.Environment, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=99999, max_spare_txns=1, lock=True)

encodedkeys = [i.encode() for i in linemaskpositionsbyformula]

abundances = {} #formula: [[masses], [intensities]]
abundanceformulas = {} #formula: subformulas
condensationcoordinates = {} #formula: [# isotopomers per proton-step]
subisodepthqualifiers = {} #formula: [[top n subisos of proton locations]]
with environment_partial(librarylocation) as env:
    aas = env.open_db('aminoacids'.encode())
    with env.begin(write=False) as txn:
        with txn.cursor(aas) as cursor:
            aaget = cursor.get(proteome.encode()).decode()
            aminoacidcomposition = eval(aaget)
    defaults = env.open_db('defaults'.encode())
    with env.begin(write=False) as txn:
        with txn.cursor(defaults) as cursor:
            dividingthreshold = float(cursor.get('dividingthreshold'.encode()).decode())
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

#elementvector = [0 for _ in elementalmasses]
#elementlist = list(elementalmasses)
#vectorpositions = {k: n for n, k in enumerate(elementlist)}
#elementpositions = {n: k for n, k in enumerate(elementlist)}

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

#to selectively pick ions below, you can iterate these dicts in order and cumulatively combine until you hit an ion you want to generate, then use those cumulative +/-s as just a single dict entry each in fragmentation_compositions. so you would generate the dicts you plan on using in this file
nfragmentcompositions = {}
nfragmentcompositions['a'] = Counter({'C': -1, 'O': -1})
nfragmentcompositions['b'] = Counter()
nfragmentcompositions['c'] = Counter({'N': 1, 'H': 3})

cfragmentcompositions = {}
cfragmentcompositions['x'] = Counter({'C': 1, 'O': 2})
cfragmentcompositions['y'] = Counter({'H': 2, 'O': 1})
cfragmentcompositions['z'] = Counter({'N': -1, 'H': -1, 'O': 1})
#cfragmentcompositions['z•'] = Counter({'N': -1, 'O': 1})

ionlist = list(ions)
ndict = {k: nfragmentcompositions[k] for k in ionlist if k in nfragmentcompositions}
cdict = {k: cfragmentcompositions[k] for k in ionlist if k in cfragmentcompositions}


#this only catches nearest 2 at most
def nearest_neighbors_ppm_tolerance(baselist, flylist):
    indices = {} #baseindex: [flyindex] or [flyindex1, flyindex2]
    distances = {} #baseindex: distance
    for bn, rightfn in enumerate(np.searchsorted(flylist, baselist).tolist()): #iter the tolist for profiling
        b = baselist[bn]
        btol = b * ppmmod
        bmin = b - btol
        bmax = b + btol
        
        leftfn = rightfn - 1 #worse case scenario this is -1 -> left = False
        left = False
        leftf = flylist[leftfn]
        if leftf > bmin and leftf < bmax:
            left = True
        
        right = False
        try:
            rightf = flylist[rightfn]
            if rightf > bmin and rightf < bmax:
                right = True
        except IndexError:
            #rightfn == len(flylist), the iteration is over
            if not left:
                return indices

        if left and right:
            leftdist = b - leftf
            rightdist = rightf - b
            if leftdist < rightdist:
                indices[bn] = [leftfn]
                distances[bn] = leftdist
            elif rightdist < leftdist:
                indices[bn] = [rightfn]
                distances[bn] = rightdist
            elif leftdist == rightdist:
                indices[bn] = [leftfn, rightfn]
                distances[bn] = leftdist
        elif left:
            leftdist = b - leftf
            indices[bn] = [leftfn]
            distances[bn] = leftdist
        elif right:
            rightdist = rightf - b
            indices[bn] = [rightfn]
            distances[bn] = rightdist
    return indices

#full radius allowing more than 2 matches
def radius_neighbors_ppm_tolerance(baselist, flylist):
    #the ppm on this currently goes out of bounds (by a lot), and i probably won't use this function anyways, replacing it for the above
    f = 0
    pool = []
    matches = {} #baselist index: [flylist indices]
    fiter = enumerate(flylist)
    for bn, b in enumerate(baselist):
        btol = b * ppmmod
        bmin = b - btol
        bmax = b + btol
        removals = []
        submatches = []
        for fi, pf in pool:
            if pf < bmin:
                removals.append([fi, pf])
            elif pf <= bmax:
                submatches.append(fi)
        for r in removals:
            pool.remove(r)
        while f <= bmax:
            try:
                i, f = next(fiter)
                if f >= bmin:
                    pool.append([i, b])
                    if f <= bmax:
                        submatches.append(i)
            except StopIteration:
                break
        if submatches:
            matches[bn] = submatches
    return matches

def timeshift(experimental, theoretical):
    #normalize both arrays to preserve their relationship
    tlen = len(theoretical)
    ratiolen = tlen - 1
    thmean = sum(theoretical) / tlen
    exmean = sum(experimental) / tlen
    scalefactor = thmean / exmean
    exnorm = [e * scalefactor for e in experimental]
    
    #calculate relative differences
    relativedifferences = [abs((e - t) / t) for e, t in zip(exnorm, theoretical)]
    
    #calculate ratios between consecutive points
    thratio = [theoretical[i+1] / theoretical[i] for i in range(ratiolen)]
    exratio = [exnorm[i+1] / exnorm[i] for i in range(ratiolen)]
    
    #compare ratios
    ratiodiffs = [abs(t - e) / t for t, e in zip(thratio, exratio)]
    
    #combine relative differences and ratio differences
    reldiffmean = sum(relativedifferences) / tlen
    ratiodiffmean = sum(ratiodiffs) / ratiolen
    combinationdiffs = (reldiffmean + ratiodiffmean) / 2
    
    return combinationdiffs

def difference_maximization(arr, double):
    #function for refined two-phase greedy difference maximization
    sarr, sdouble = map(list, zip(*sorted(zip(arr, double))))
    
    #initialize sequence with the largest and smallest elements
    sequence = [sarr.pop(0), sarr.pop(-1)]
    sequencedouble = [sdouble.pop(0), sdouble.pop(-1)]
    
    while sarr:
        #compute the difference for adding either the next smallest or next largest element
        min_value, max_value = sarr[0], sarr[-1]
        
        #compare adding to both ends with the smallest and largest elements
        add_to_left_diff_min = abs(min_value - sequence[0])
        add_to_right_diff_min = abs(min_value - sequence[-1])
        
        add_to_left_diff_max = abs(max_value - sequence[0])
        add_to_right_diff_max = abs(max_value - sequence[-1])
        
        #decide to place the minimum or maximum value based on the maximum possible gain
        if add_to_left_diff_min >= add_to_right_diff_min and add_to_left_diff_min >= add_to_left_diff_max and add_to_left_diff_min >= add_to_right_diff_max:
            sequence.insert(0, sarr.pop(0))
            sequencedouble.insert(0, sdouble.pop(0))
        elif add_to_right_diff_min >= add_to_left_diff_min and add_to_right_diff_min >= add_to_left_diff_max and add_to_right_diff_min >= add_to_right_diff_max:
            sequence.append(sarr.pop(0))
            sequencedouble.append(sdouble.pop(0))
        elif add_to_left_diff_max >= add_to_right_diff_max and add_to_left_diff_max >= add_to_left_diff_min and add_to_left_diff_max >= add_to_right_diff_min:
            sequence.insert(0, sarr.pop(-1))
            sequencedouble.insert(0, sdouble.pop(-1))
        else:
            sequence.append(sarr.pop(-1))
            sequencedouble.append(sdouble.pop(-1))
    
    return sequence, sequencedouble

def sequence_geometry(seq, ioncoverage):
    slen = len(seq)
    maxncoverage = 0
    maxccoverage = 0
    dividers = set()
    ntermcoverage = []
    ctermcoverage = []
    for ion in ioncoverage:
        iontype = ion[0]
        ioncount = int(ion[1:])
        if iontype in 'abc': #nterm
            dividers.add(ioncount)
            pseq = seq[:ioncount]
            ntermcoverage.append(ioncount)
            if ioncount > maxccoverage:
                maxccoverage = ioncount
        elif iontype in 'xyz': #cterm
            dividers.add(slen - ioncount)
            pseq = seq[slen-ioncount:]
            ctermcoverage.append(slen-ioncount)
            if ioncount > maxncoverage:
                maxncoverage = ioncount
    dividers = sorted(dividers)
    coverageweight = 1 / (maxncoverage + maxccoverage)

    #isolation counts need to be robust against redundant pseqs
    ind = 0
    ddiff = np.diff(dividers, prepend=0).tolist()
    #dividerstring = ''
    partialseqs = defaultdict(int) #index-pseq: count #the index safeguards against multiple isolations of the same partial sequence, the defaultdict rather than a Counter keeps the keys in order so i can view it easier in regards to the order of the sequence
    for d in ddiff:
        pseq = seq[ind:ind+d]
        #dividerstring += pseq + '|'
        ntermcovers = [i for i in ntermcoverage if i > ind]
        ctermcovers = [i for i in ctermcoverage if i <= ind]
        covers = len(ntermcovers) + len(ctermcovers)
        if covers > 0:
            label = str(ind) + '-' + pseq
            partialseqs[label] += covers
        ind += d
    pseq = seq[ind:]
    #dividerstring += pseq
    ntermcovers = [i for i in ntermcoverage if i > ind]
    ctermcovers = [i for i in ctermcoverage if i <= ind]
    covers = len(ntermcovers) + len(ctermcovers)
    if covers > 0:
        label = str(ind) + '-' + pseq
        partialseqs[label] += covers
    pairsum = 0
    matchcounts = len(set(ioncoverage))
    isolationlengthweight = 1
    for indseq, count in partialseqs.items():
        ind, pseq = indseq.split('-')
        ind = int(ind)
        #i could use this index to weight based on distance from the ends i guess?
        #isolationlengthweight *= len(pseq) / len(seq)
        isolationlengthweight *= 1 / len(pseq) / len(seq) #this 1 / provides an additional layer of geometric success to this scheme, grants success where there was previously failure
        plen = len(pseq)
        matchcounts += plen * count
    dividerweight = 1 / len(partialseqs)
    out1 = 1 / (dividerweight + isolationlengthweight + coverageweight)
    out2 = (1 / dividerweight) + (1 / isolationlengthweight) + (1 / coverageweight)
    return out1, out2, matchcounts

def fragmentation_compositions(seq):
    fragments = {}

    #calculate the compositions of the n-term fragments
    fragcomp_n = {}
    for n, aa in enumerate(seq[:-1]):  
        aa_composition = aminoacidcomposition[aa]
        for k in aa_composition:
            fragcomp_n[k] = fragcomp_n.get(k, 0) + aa_composition.get(k, 0)
        for ion, modcomp in ndict.items():
            fragment_composition = fragcomp_n.copy()
            for k in modcomp:
                fc = fragment_composition.get(k, 0) + modcomp.get(k, 0)
                if fc > 0:
                    fragment_composition[k] = fc
                else:
                    del fragment_composition[k]
            fragments[ion + str(n + 1)] = fragment_composition
    
    #calculate the compositions of the c-term fragments
    fragcomp_c = {}
    for n, aa in enumerate(seq[::-1][:-1]): 
        aa_composition = aminoacidcomposition[aa]
        for k in aa_composition:
            fragcomp_c[k] = fragcomp_c.get(k, 0) + aa_composition.get(k, 0)
        for ion, modcomp in cdict.items():
            fragment_composition = fragcomp_c.copy()
            for k in modcomp:
                fc = fragment_composition.get(k, 0) + modcomp.get(k, 0)
                if fc > 0:
                    fragment_composition[k] = fc
                else:
                    del fragment_composition[k]
            fragments[ion + str(n + 1)] = fragment_composition
    
    #aa = seq[0]
    #aa_composition = aminoacidcomposition[aa]
    #for k in aa_composition:
    #    fragcomp_c[k] = fragcomp_c.get(k, 0) + aa_composition.get(k, 0)
    #fragcomp_c['H'] += 2
    #fragcomp_c['O'] += 1
    #fragments['precursor'] = fragcomp_c
    
    return fragments

def fragment_element_binomial_walk(e, acount, fragprobabilities):
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
    nvector[fragmentvectorpositions[mk]] += acount
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

def fragment_descending_partial_products(elementalorganizer, fragmentpositions):
    mainpool = defaultdict(list) #things already popped from elementalorganizer
    for k in elementalorganizer:
        mainpool[k].append(heapq.heappop(elementalorganizer[k]))

    subformulas = []
    sumabundances = []
    massnumberindices = {} #mass number: index in the 2 above lists
    
    formula = ''
    maxprob = 1
    mainmass = 0
    massnumber = 0
    for b in sorted(mainpool):
        for r, p, m, e, v in mainpool[b]:
            for n, c in enumerate(v):
                if c > 0:
                    iso = fragmentpositions[e][n]
                    massnumber += int(iso[1:]) * c
                    formula += f'{iso}({c})'
            maxprob *= p
            mainmass += m
    
    massnumberindices[massnumber] = 0
    subformulas.append(formula)
    sumabundances.append([mainmass * maxprob, maxprob])
    
    cutoff = maxprob * dividingthreshold
    mainheap = list(itertools.chain(*elementalorganizer.values()))
    heapq.heapify(mainheap)
    
    vectorpool = set()
    multinomialpath = [] #sublists not in mainpool
    probabilityranking = [] #representative lists of ratio probability to sort multinomialpath
    while mainheap:
        r, p, m, e, v = heapq.heappop(mainheap)
        baseiter = {k: v for k, v in mainpool.items() if k != e}
        baseiter[e] = [(r, p, m, e, v)]
        
        formula = ''
        prob = 1
        mass = 0
        massnumber = 0
        for b in sorted(baseiter):
            for sr, sp, sm, se, sv in baseiter[b]:
                for n, c in enumerate(sv):
                    if c > 0:
                        iso = fragmentpositions[se][n]
                        massnumber += int(iso[1:]) * c
                        formula += f'{iso}({c})'
                prob *= sp
                mass += sm
        
        try:
            index = massnumberindices[massnumber]
            subformulas[index] += '-' + formula
            sumabundances[index][0] += mass * prob
            sumabundances[index][1] += prob
        except KeyError: #not in there
            index = len(massnumberindices)
            massnumberindices[massnumber] = index
            subformulas.append(formula)
            sumabundances.append([mass * prob, prob])
        if prob < cutoff:
            break
        
        tsv = tuple(v)
        if tsv not in vectorpool:
            ind = bisect(probabilityranking, r)
            probabilityranking.insert(ind, r)
            multinomialpath.insert(ind, (r, p, m, e, v))
            vectorpool.add(tsv)
        
        checkedcombos = set()
        for path in multinomialpath.copy():
            multielement = False
            match path[1]:
                case tuple():
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
                            multipath.append((sr, sp, sm, se, sv))
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
                    newmassnum = 0
                    newiter = {k: v for k, v in baseiter.items() if k not in sepool}
                    newiter[e] = [(r, p, m, e, v)]
                    for ir, ip, im, ie, iv in multipath:
                        newiter[ie] = [(ir, ip, im, ie, iv)]
                    for b in sorted(newiter):
                        for ir, ip, im, ie, iv in newiter[b]:
                            for n, c in enumerate(iv):
                                if c > 0:
                                    iso = fragmentpositions[ie][n]
                                    newmassnum += int(iso[1:]) * c
                                    seformula += f'{iso}({c})'
                            newprob *= ip
                            newmass += im
                else:
                    newiter = {k: v for k, v in baseiter.items() if k != se}
                    newiter[se] = [(sr, sp, sm, se, sv)]
                    seformula = ''
                    newprob = 1
                    newmass = 0
                    newmassnum = 0
                    for b in sorted(newiter):
                        for ir, ip, im, ie, iv in newiter[b]:
                            for n, c in enumerate(iv):
                                if c > 0:
                                    iso = fragmentpositions[ie][n]
                                    newmassnum += int(iso[1:]) * c
                                    seformula += f'{iso}({c})'
                            newprob *= ip
                            newmass += im
                if newprob >= cutoff:
                    try:
                        index = massnumberindices[newmassnum]
                        subformulas[index] += '-' + seformula
                        sumabundances[index][0] += newmass * newprob
                        sumabundances[index][1] += newprob
                    except KeyError: #not in there
                        index = len(massnumberindices)
                        massnumberindices[newmassnum] = index
                        subformulas.append(seformula)
                        sumabundances.append([newmass * newprob, newprob])
                    if multielement:
                        ind = bisect(probabilityranking, newratio)
                        probabilityranking.insert(ind, newratio)
                        multinomialpath.insert(ind, (newratio, *multipath))
                    else: #this is rarely ever needed, but it is needed
                        newmulti = []
                        tsv = tuple(sv)
                        #should this one be first? does it matter? i don't believe it does
                        if tsv not in vectorpool:
                            newmulti.append((sr, sp, sm, se, sv))
                            vectorpool.add(tsv)
                        tvv = tuple(v)
                        if tvv not in vectorpool:
                            newmulti.append((r, p, m, e, v))
                            vectorpool.add(tvv)
                        if newmulti:
                            ind = bisect(probabilityranking, newratio)
                            probabilityranking.insert(ind, newratio)
                            multinomialpath.insert(ind, (newratio, *newmulti))
            else:
                break

    subformulas = np.array(subformulas, dtype='S')
    massesandabundances = np.array(sumabundances)
    massesandabundances[:,0] /= massesandabundances[:,1]
    #sorting by intensity
    subformulas = subformulas[massesandabundances[:,1].argsort()[::-1]].tolist()
    massesandabundances = massesandabundances[massesandabundances[:,1].argsort()[::-1]]
    return subformulas, massesandabundances

#pick a scan
#get all the lines in that scan
#search every sequence for every line
#list the scores, visualize the top one
#reverse linepositionsbyformula
#line: formula: positions
#derive seqs and subformulas from these
#pick from maxintensitylinesofdists?
originalscan = 13904

groupseqs = set()
groupsubformulas = set()
for line in linesofscans[originalscan]:
    for formula, positions in positionsbyformulabyline[line].items():
        qualifiers = subisodepthqualifiers[formula]
        conlengths = condensationcoordinates[formula]
        conends = conlengths.cumsum()
        constarts = conends - conlengths
        subformulas = [i.decode() for i in abundanceformulas[formula]]
        massesandintensities = abundances[formula]
        theoreticalabundances = massesandintensities[1]
        #mainindexformulas[mainmatchindex] = formula
        for position in positions:
            for seq in seqswithdecoysbyformula[formula]:
                groupseqs.add(seq)
                bi = constarts[position]
                for qualrank, sq in enumerate(qualifiers[position]):
                    subindex = bi + sq
                    sformula = subformulas[subindex]
                    groupsubformulas.add(sformula)


searchtime = 0
filtertime = 0
fraglens = 0
scanlens = 0
chargeiterations = 0
nt = time()
positioncache = {}
elementalcache = {}
descentcache = {}
#postfragmenttypes = Counter()
#postfragmentcounts = defaultdict(lambda: Counter())
initialmatches = 0
finalmatches = 0
subformulaoutput = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))) #seq: distid: linemask: scan: subformula: ion charge: ion: [metrics]
fragments = {}
for seq in groupseqs:
    fragments[seq] = fragmentation_compositions(seq)
for subformula in groupsubformulas:
    probindices = {elementsofprobabilityindices[i]: probabilityorganizer[i] for i in matchprobabilities[subformula]}
    subindices = subformulasubindices[subformula]
    output, fragmasses = [], []
    for submatchindex in subindices:
        seq = submatchsequences[submatchindex]
        for ion, fragcomp in fragments[seq].items():
            elementalorganizer = {} #element: [[iso heaps]]
            fragmentpositions = {} #element: position: iso
            fragstrings = ''
            for e, c in fragcomp.items():
                fragprobs = probindices[e]
                fragstring = str(c) + '/' + '/'.join(('/'.join((k, str(v))) for k, v in probindices[e].items()))
                fragstrings += fragstring
                if len(fragprobs) > 1:
                    #try/except is faster than an if/else, so i might as well
                    try:
                        elementlist = elementalcache[fragstring]
                        positions = positioncache[fragstring]
                    except KeyError: #not in cache
                        elementlist, positions = fragment_element_binomial_walk(e, c, fragprobs)
                        elementalcache[fragstring] = elementlist
                        positioncache[fragstring] = positions
                    elementalorganizer[e] = elementlist.copy()
                    fragmentpositions[e] = positions
                else: #no need for cache, only 1 iso
                    iso = list(fragprobs)[0]
                    elementalorganizer[e] = [[-1, 1, elementalmasses[iso]*c, e, [c]]]
                    fragmentpositions[e] = {0: iso}
            try:
                fragformulas, massesandabundances = descentcache[fragstrings]
            except KeyError: #not done prior
                fragformulas, massesandabundances = fragment_descending_partial_products(elementalorganizer, fragmentpositions)
                descentcache[fragstrings] = fragformulas, massesandabundances
            for n, (m, i) in enumerate(massesandabundances.tolist()):
                out = (seq, ion, fragformulas[n].decode(), n, i)
                output.append(out)
                fragmasses.append(m)
    fragmasses, output = zip(*sorted(zip(fragmasses, output)))
    fragmasses = np.array(fragmasses)
    fraglens += fragmasses.size
    st = time()
    for scan, linemasks in linemasksbyscanbysubformula[subformula].items():
        if len(linemasks) > 1:
            chargeset = set(chargesoflinemasks[i] for i in linemasks)
            #if len(chargeset) > 1: #-> test passes
            #    print('PROBLEM, chargeset len > 1')
            maxcharge = max(chargeset)
            #CHECK if ^this is ever different, i'm pretty sure its always the same charge, there shouldn't be different ones
            #^because even if the same subformula is in the same scan more than once, it will never be of a different charge than itself in another distribution..
            #linestring = '_'.join((str(i) for i in lines))
            linemasksofmatchdistributions = defaultdict(list) #distid: [lines]
            for linemask in linemasks:
                linemasksofmatchdistributions[distributionsoflinemasks[linemask]].append(linemask)
        else:
            linemask = linemasks[0]
            maxcharge = chargesoflinemasks[linemask]
            #linestring = str(lines)
            linemasksofmatchdistributions = {distributionsoflinemasks[linemask]: [linemask]}
        #put fragmasses here -> append precursor ion of the line? because i dont want to calculate precursors via the above dists, but i want to match them
        #^i might just make a special precursor search because this gets too retarded
        #i'm removing precursors from the above calculations for now
        ms2masses, ms2intensities, ms2scores = scoredms2scans[scan]
        scanlens += ms2masses.size
        #ms2masses = ms2masses.tolist()
        outputorganizer = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))) #sequence: charge: ion: fragrank: [[metrics],]
        for charge in range(1, maxcharge+1):
            chargeiterations += 1
            chargedfragments = ((fragmasses + proton * charge) / charge)
            #i'm going to ditch the radius neighbors for a nearest neighbors concept instead, ought to have a minor speedboost at least
                #when MS resolution is low -> you won't get anything close enough to have more than 1 thing in a radius
                #if its high -> you CAN have this, but you should also expect the masses to be accurate
                    #i can see this in the ms1 vs ms2 data for the fr400 file
                #this will match at most 2 ions if they both have the same distance to a theoretical fragment
            matches = nearest_neighbors_ppm_tolerance(chargedfragments, ms2masses)
            #matches = radius_neighbors_ppm_tolerance(chargedfragments.tolist(), ms2masses)
            for fragindex, scanindices in matches.items(): #frag index: [mass index] or [mass index 1, mass index 2]
                #a scanmass can match to multiple generated fragment ions
                for scanindex in scanindices:
                    experimentalmass = ms2masses[scanindex]
                    ionscore = ms2scores[scanindex]
                    experimentalintensity = ms2intensities[scanindex]
                    theoreticalmass = chargedfragments[fragindex]
                    #ppmerror = ((experimentalmass - theoreticalmass) / experimentalmass) * 1000000 #maybe this could be done somewhere below instead
                    seq, ion, fragformula, fragrank, theoreticalabundance = output[fragindex]
                    metrics = [fragrank, theoreticalmass, experimentalmass, theoreticalabundance, experimentalintensity, scanindex, fragformula, ionscore]
                    outputorganizer[seq][charge][ion][fragrank].append(metrics)
                    initialmatches += 1
        nst = time()
        for seq, ioncharges in outputorganizer.items():
            for ioncharge, ions in ioncharges.items():
                for ion, fragranks in ions.items():
                    fcount = 0
                    for fragrank in sorted(fragranks):
                        if fragrank == fcount:
                            fcount += 1
                    if fcount > 0:
                        combinationoutputs = defaultdict(list) #either 0: [single ion] or 1: [multiple ions]
                        #^multiple ions will always be chosen over the single in ranked order
                        #^if there's multiple single ions, whichever is closer in intensity to the average intensity of that scan will be chosen
                        if fcount > 1:
                            #process multiple potential fragiso ranks
                            #iterate products i guess
                            #assemble a list of the number of total combinations, check if len > 1 or not
                            isoiterators = {}
                            for c in range(fcount):
                                isoiterators[c] = fragranks[c]
                            #this is determining which out of all possible frag iso combos makes the best distribution for this matched frag dist
                            for rankcombos in itertools.product(*isoiterators.values()):
                                #assess on charge distance, its consistency, and abundance modeling
                                rankcombinations = sorted(rankcombos) #not guaranteed to be most-intense ion as first mass
                                scorearray = np.array([i[:4] for i in rankcombinations])
                                rankpairs = np.array([(rankcombinations[i][6], rankcombinations[i+1][6]) for i in range(len(rankcombinations)-1)])
                                theoreticalabundances = scorearray[:,2]
                                experimentalintensities = scorearray[:,3]
                                lowerthbounds = theoreticalabundances[:-1] / theoreticalabundances[1:]
                                #the closer the max / min ratio of theoretical abundance are to 1, the more accurate the ratio between the 2 needs to be ->> square the theoretical ratio, that's the limit for the experimental
                                #the min would be half the initial ratio as long as its > 1? maybe
                                #and the max would be the squared value
                                upperthbounds = lowerthbounds ** 2 #this decreases lower bounds and increases upper ones.. i think this is fine?
                                #model by mass but rank by ranks i guess?
                                #start at the 0 to 1 rank linkage
                                #and if linkages dont always connect next to each other you can always link to a rank previously seen in this ranking
                                extimeshift = experimentalintensities[:-1] / experimentalintensities[1:]
                                acceptablepairs = np.sort(rankpairs[np.logical_and(extimeshift > lowerthbounds, extimeshift < upperthbounds)]).tolist()
                                try:
                                    firstgroup = acceptablepairs[0]
                                except IndexError:
                                    #useless, no good matches here
                                    continue
                                isoindices = set()
                                #the first two ranks MUST have the correct rank order intensities, or else it just won't be taken, take only the top rank instead
                                if firstgroup == [0, 1]:
                                    #good, start distassembling
                                    isoindices.update(firstgroup)
                                    #from these rankpairs just accumulate anything adjacent to whatevers already in there
                                    for l, r in acceptablepairs[1:]:
                                        if l in isoindices or r in isoindices:
                                            #connect anything adjacent
                                            isoindices.add(l)
                                            isoindices.add(r)
                                        else:
                                            #finished accumulating
                                            break
                                else:
                                    #mismatch, no good
                                    if 0 in firstgroup:
                                        #take 0
                                        combinationoutputs[0].append([abs(intensityaverages[scan]-rankcombos[0][3]), rankcombos[0]])
                                        continue
                                    else:
                                        #useless
                                        continue
                                finalindices = sorted(isoindices)
                                #score the dist and add it to the final list
                                massdiffs = np.diff(scorearray[finalindices,1])
                                avgmdiff = np.abs(massdiffs.mean() - massdiffs).mean() #mass distance consistency measure
                                theoreticalabundances = scorearray[finalindices,2].tolist()
                                experimentalintensities = scorearray[finalindices,3].tolist()
                                shiftdeviance = timeshift(experimentalintensities, theoreticalabundances) #time-series comparison
                                #shiftdeviance = linregress(experimentalintensities, theoreticalabundances).pvalue
                                combinationoutputs[1].append([avgmdiff * shiftdeviance, [rankcombos[i] for i in finalindices]])
                        else: #fcount == 1
                            fullmetrics = fragranks[0]
                            if len(fullmetrics) > 1:
                                #multiple matches to this fragrank
                                #pick whichever is closer in intensity to the average intensity of the scan
                                scanav = intensityaverages[scan]
                                avdiffs = [abs(scanav-i[3]) for i in fullmetrics]
                                minav = min(avdiffs)
                                finalmetric = fullmetrics[avdiffs.index(minav)]
                                combinationoutputs[0].append([minav, finalmetric])
                            else:
                                #single, take it
                                minav = abs(intensityaverages[scan] - fullmetrics[0][3])
                                combinationoutputs[0].append([minav, fullmetrics[0]])
                        #with either of these results below i'm assuming an equal score would only be given to matches that are exactly the same
                        if 1 in combinationoutputs:
                            #sort and pick best
                            selection = min(combinationoutputs[1])
                        elif 0 in combinationoutputs:
                            #take whichever is nearest to the mean intensity of the scan
                            selection = min(combinationoutputs[0])
                        else:
                            #got nothing
                            continue
                        for distid, alinemasks in linemasksofmatchdistributions.items():
                            for linemask in alinemasks:
                                subformulaoutput[seq][distid][linemask][scan][subformula][ioncharge][ion] = selection
                    else:
                        #nada
                        continue
        filtertime += time() - nst
    searchtime += time() - st
fragtime = time() - nt - searchtime
searchtime -= filtertime
sct = time()
#scanleveloutput = []
peptideleveloutput = []
scanleveloutput = defaultdict(list) #scan: [info]
ionleveloutput = []
for seq, distributions in subformulaoutput.items():
    for distid, linemasks in distributions.items():
        #ioncoverage = set()
        #scanindexstring = ''
        #intensityratios = []
        #abundanceratios = []
        #distributionfragmentindices = defaultdict(set) #scan: [scan indices]
        #this ion superset samples the most intense ion of a distribution and determines a top-down superset of all the fragmenting ions to be imposed on every other subformula and MS2 sampling taken at lesser intensities, if an ion doesn't show up here then it's not allowed to contribute to the rest of the scoring/ID process as its inconsistent and probably not real
        #so ie this assumes all subformulas fragment similar enough for it to matter despite slight isotopic differences - which i think should be ok
        #starting with the line and scan where this distribution sampled the largest MS1 intensity
        #line, scan = maxintensitylinesofdists[distid]
        ##if scan in scans:
        #if line in lines:
        #    #lines = scans[scan]
        #    scans = lines[line]
        #    #if line in lines:
        #    if scan in scans:
        #        #this is what should be the most abundant subformula at that position
        #        subformula = subformulasofsequencedistribution[distid][seq]
        #        subformulas = scans[scan]
        #        #if subformula in subformulas:
        #        ioncharges = subformulas[subformula]
        #        ionsuperset = defaultdict(set) #charge: [ions]
        #        for ioncharge, ions in ioncharges.items():
        #            #add to these ions to the superset of this identification instance
        #            ionsuperset[ioncharge].update(ions)
        #                #for ion, metrics in ions.items():
        #        else:
        #            #no superset to be made, the supposed best match isn't there
        #            continue
        #    else:
        #        continue
        #else:
        #    continue
        #scanorder = []
        #ms1intensities = []
        #ms2intensities = []
        #scanlineintensitiesbyion = defaultdict(lambda: Counter()) #line-scan: ion: intensity
        #distributionppm = 0
        #distributionionscore = 0
        #distributionintensity = 0
        #for scan, lines in scans.items():
        for linemask, scans in linemasks.items():
            #for line, subformulas in lines.items():
            for scan, subformulas in scans.items():
                fragmentindices = set() #all fragmass indices in a scan
                linescan = str(linemask) + '-' + str(scan)
                #sort subformulas using subformulapercent, take only adjacent matches, if something has no superset matches -> break
                subformulalist = sorted((subformulapercent[i][seq], i) for i in subformulas)
                #subformulalist = sorted((*subformulapercent[i][seq], subformula) for i in subformulas)
                #^which is faster?
                #main scoring mechanisms:
                    #fragdist multiple -> here
                    #cross-scan consistency -> here as a time series across scans
                    #sequence geometry -> here
                    #cross-subformula entropy -> here
                    #intensity pair entropy -> here -> nah im not implementing this
                    #MS1/MS2 intensity entropy by scan % -> next script -> groups into dists i suppose
                #aiming to make lower scores better in every case to multiply them all together
                #subformulamassindices = defaultdict(lambda: defaultdict(dict)) #ioncharge: ion: subformula: [masses]
                #subformulaintensities = defaultdict(lambda: defaultdict(dict)) #ioncharge: ion: subformula: intensity sum
                #abundanceofsubformulas = {} #subformula: abundance
                #fragdistmultiple = 1
                #fragioncount = 0
                #fragmentintensities = set() #assuming each intensity is unique which is actually false apparently, but within scans maybe more probable
                #ionsubformulastring = '' #subformula^subformularank%ioncharge&ion&scanindex&ppmerror_ioncharge&ion...-subformula&... in order of subformula abundance
                #gonna capture intensities across ions and check a timeshift
                qcount = 0
                for (qualrank, abundance), subformula in subformulalist:
                    #this is multiplying the fragdist scores and assembling info to be used for everything else
                    #iterating and applying multiples in order of decreasing subformula abundance
                    if qualrank == qcount:
                        ionmatches = False
                        ioncharges = subformulas[subformula]
                        for ioncharge, ions in ioncharges.items():
                            #if ioncharge in ionsuperset:
                            if True:
                                sortedions = sorted(ions.items(), key=lambda x: x[1][0]) #sorting by fragdist score, so the fragment indices go to better ion first in case they overlap
                                for ion, metrictuple in sortedions:
                                    #if ion in ionsuperset[ioncharge]:
                                    if True:
                                        fragdistscore, metrics = metrictuple
                                        #i only want to take these ions if i haven't seen that scanindex in this scan yet? or i need a way to determine which i prefer it to be labeled as
                                        match metrics[0]:
                                            case list():
                                                metrics = sorted(metrics) #sorting by fragrank
                                                for fragrank, theoreticalmass, experimentalmass, theoreticalabundance, experimentalintensity, scanindex, fragformula, ionscore in metrics:
                                                    #if scanindex not in distributionfragmentindices[scan]:
                                                    #distributionfragmentindices[scan].add(scanindex)
                                                    ppmerror = ((experimentalmass - theoreticalmass) / experimentalmass) * 1000000
                                                    abserror = abs(ppmerror)
                                                    scanleveloutput[scan].append([ionscore, abserror, experimentalintensity, ion, scanindex, seq, distid, linemask])
                                                    ionleveloutput.append([ionscore, abserror, experimentalintensity, ion, scan, scanindex, seq, distid, linemask])
                                                    finalmatches += 1
                                            case float:
                                                #if scanindex not in distributionfragmentindices[scan]:
                                                #distributionfragmentindices[scan].add(scanindex)
                                                fragrank, theoreticalmass, experimentalmass, theoreticalabundance, experimentalintensity, scanindex, fragformula, ionscore = metrics
                                                ppmerror = ((experimentalmass - theoreticalmass) / experimentalmass) * 1000000
                                                abserror = abs(ppmerror)
                                                scanleveloutput[scan].append([ionscore, abserror, experimentalintensity, ion, scanindex, seq, distid, linemask])
                                                ionleveloutput.append([ionscore, abserror, experimentalintensity, ion, scan, scanindex, seq, distid, linemask])
                                                finalmatches += 1
                        qcount += 1
                    else:
                        #descending order of subformulas is finished
                        break
        if ionleveloutput:
            ionleveloutput = sorted(ionleveloutput, key=lambda x: (-x[0], x[1]))
            filteredoutput = list(filter(lambda x: x[0] > 0, ionleveloutput))
            if filteredoutput:
                removals = []
                lastindex = -1
                for i in filteredoutput:
                    if i[5] == lastindex:
                        removals.append(i)
                    else:
                        lastindex = i[5]
                for r in removals:
                    filteredoutput.remove(r)
                
                ppmerror = 0
                fullionscore = 0
                sumintensity = 0
                ioncoverage = set()
                scanindices = defaultdict(set) #scan: [scanindices]
                for ionscore, abserror, experimentalintensity, ion, scan, scanindex, seq, distid, linemask in filteredoutput:
                    ioncoverage.add(ion)
                    fullionscore += ionscore
                    ppmerror += abserror
                    sumintensity += experimentalintensity
                    scanindices[scan].add(scanindex)
                
                scanindexstring = ''
                for scan, indices in scanindices.items():
                    indexstring = '/'.join((map(str, sorted(indices))))
                    scanindexstring += f'{scan}[{indexstring}]'
                
                dg1, dg2, dg3 =  sequence_geometry(seq, ioncoverage)
                ioncoveragestring = '/'.join(map(str, sorted(ioncoverage)))
                outputstring = [seq, distid, ioncoveragestring, scanindexstring, fullionscore, ppmerror, sumintensity, dg1, dg2, dg3]
                peptideleveloutput.append(outputstring)

df = pd.DataFrame(peptideleveloutput, columns=['sequence', 'distid', 'ion_coverage', 'scan_indices', 'dist_ion_score', 'dist_ppm_error', 'dist_match_intensity', 'dg1', 'dg2', 'dg3'])
df.loc[:,'nions'] = df.loc[:,'ion_coverage'].apply(lambda x: x.count('/') + 1)

df.sort_values('dist_ion_score', ascending=False, inplace=True)

mza, intensities, scores = scoredms2scans[originalscan]
plt.bar(mza, intensities, width=2)
plt.show()

#distribution matches as floating differences
#by dist size
#by total rank offset
#these are all dists that can have a floating cutoff i guess
#of things that have 0 offset, things of lesser floating diff should be alright
