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

#isotopomerpositionsfile = '/'.join((processinglocation, 'isotopomersbypositions.pickle'))
#with open(isotopomerpositionsfile, 'rb') as pick:
#    isotopomerpositionsofanalytes = pickle.load(pick)
##isotopomerpositionsofanalytes = defaultdict(set) #analyteid: isotopopmer coordinate from max

#spectrabyformulafile = '/'.join((processinglocation, 'spectrabyformula.pickle'))
#with open(spectrabyformulafile, 'rb') as pick:
#    spectrabyformula = pickle.load(pick)
##spectrabyformula = defaultdict(lambda: defaultdict(set)) #line: position: [formulas]

#analytesbyformulafile = '/'.join((processinglocation, 'analytesbyformula.pickle'))
#with open(analytesbyformulafile, 'rb') as pick:
#    analytesbyformula = pickle.load(pick)
##analytesbyformula = defaultdict(set) #formula: [analyteids]
#
#linepositionsofanalytesfile = '/'.join((processinglocation, 'linepositionsofanalytes.pickle'))
#with open(linepositionsofanalytesfile, 'rb') as pick:
#    linepositionsofanalytes = pickle.load(pick)
##linepositionsofanalytes = defaultdict(lambda: defaultdict(set)) #analyteid: position: [lines]

linepositionsbyformulafile = '/'.join((processinglocation, 'linepositionsbyformula.pickle'))
with open(linepositionsbyformulafile, 'rb') as pick:
    linepositionsbyformula = pickle.load(pick)
#linepositionsbyformula = defaultdict(lambda: defaultdict(set)) #formula: matchable library positions of a distribution: [lines that have ms2 scans]

scansbyanalytefile = '/'.join((processinglocation, 'scansbyanalyte.pickle'))
with open(scansbyanalytefile, 'rb') as pick:
    scansbyanalyte = pickle.load(pick)
#scansbyanalyte = defaultdict(list) #analyteid: [scans across all lines and charge states]

scansoflinesfile = '/'.join((processinglocation, 'scansoflines.pickle'))
with open(scansoflinesfile, 'rb') as pick:
    scansoflines = pickle.load(pick)
#scansoflines = defaultdict(list) #lineuid: [ms2 scan indices]

#divisionfile = '/'.join((processinglocation, 'dividedgroups.pickle'))
#with open(divisionfile, 'rb') as pick:
#    dividedgroups = pickle.load(pick)

#divkeys = dividedgroups[0]
#spectrabyformula = {i:spectrabyformula[i] for i in divkeys}

#encodedkeys = set()
#for line, positions in spectrabyformula.items():
#    for pos, formulas in positions.items():
#        encodedkeys.update(formulas)
encodedkeys = [i.encode() for i in linepositionsbyformula]

seqsbyformula = {} #formula: [seqs]
abundances = {} #formula: [[masses], [intensities]]
abundanceformulas = {} #formula: subformulas
condensationcoordinates = {} #formula: [# isotopomers per proton-step]
subisodepthqualifiers = {} #formula: [[top n subisos of proton locations]]
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

#aminoacidcomposition = {
#        'A': {'C': 3, 'H': 5, 'N': 1, 'O': 1},
#        'R': {'C': 6, 'H': 12, 'N': 4, 'O': 1},
#        'N': {'C': 4, 'H': 6, 'N': 2, 'O': 2},
#        'D': {'C': 4, 'H': 5, 'N': 1, 'O': 3},
#        'C': {'C': 3, 'H': 5, 'N': 1, 'O': 1, 'S': 1},
#        'Q': {'C': 5, 'H': 8, 'N': 2, 'O': 2},
#        'E': {'C': 5, 'H': 7, 'N': 1, 'O': 3},
#        'G': {'C': 2, 'H': 3, 'N': 1, 'O': 1},
#        'H': {'C': 6, 'H': 7, 'N':3, 'O': 1},
#        'I': {'C': 6, 'H': 11, 'N':1, 'O': 1},
#        'L': {'C': 6, 'H': 11, 'N':1, 'O': 1},
#        'K': {'C': 6, 'H': 12, 'N': 2, 'O': 1},
#        'M': {'C': 5, 'H': 9, 'N':1, 'O': 1, 'S': 1},
#        'F': {'C': 9, 'H': 9, 'N':1, 'O': 1},
#        'P': {'C': 5, 'H': 7, 'N':1, 'O': 1},
#        'S': {'C': 3, 'H': 5, 'N':1, 'O': 2},
#        'T': {'C': 4, 'H': 7, 'N':1, 'O': 2},
#        'W': {'C': 11, 'H': 10, 'N': 2, 'O': 1},
#        'Y': {'C': 9, 'H': 9, 'N': 1, 'O': 2},
#        'V': {'C': 5, 'H': 9, 'N': 1, 'O': 1}
#        }

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

#i'm going to start off assuming that the same fragments should have the same relative prevalence regardless of isotopic location or composition
#so it would be a good idea to base a metric off finding the most true relative ratios of fragments across ions 
#so then make a way to visualize a line, its ms2 hit points, other dists that aren't currently described by the plots but are in other hits, and the consistency of fragment ratios of different isotopomers
#also make a way to visualize two potential sequence matches next to each other
#on the technical side, facilitate the matching process

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


print(time() - nt, 'initialized')
nt = time()

#i think isotopomerpositionsofanalytes could use formulas as a key instead, confirm all the lists of things sharing a formula are the same thing

mainindexbysubindex = defaultdict(list) #sub match index: main match index
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

mainindexformulas = {} #main index: main formula
subformulasubindices = defaultdict(list) #subformula: [sub match indices]
submatchsequences = {} #submatchindex: sequence
submatchsubformulas = {} #submatchindex: subformula
elementsofprobabilityindices = {} #prob index: e
#submatchpositions = {} #submatch index: [distribution position, subiso position]

linesbysubformula = defaultdict(set) #subformula: [lines that have ms2 scans]

#anything with a sub match index key is way too enormous to hold on disk
#link mainindex to formula -> which links it to seq
#seq can generate the fragments
#maybe mainindexbysubindex will lead to a seq instead?

subformulasoflines = defaultdict(set) #line: [subformulas]
subformularank = defaultdict(dict) #sequence: subformula: descending subiso rank, lower int = more relevant subiso
subformulapercent = defaultdict(dict) #sequence: subformula: % #this doesn't need a position-level key because subformulas are organized by line later on, so only the correct subisos will be referenced, i also dont need to normalize these yet

probindex = 0
#fragindex = 0
submatchindex = 0
mainmatchindex = 0
#for formula, analyteids in spectrabyformula.items():
for formula, positions in linepositionsbyformula.items():
    qualifiers = subisodepthqualifiers[formula]
    conlengths = condensationcoordinates[formula]
    conends = conlengths.cumsum()
    constarts = conends - conlengths
    subformulas = [i.decode() for i in abundanceformulas[formula]]
    massesandintensities = abundances[formula]
    intensities = massesandintensities[1]
    #for analyteid in analytes:
    mainindexformulas[mainmatchindex] = formula
    for position, lines in positions.items():
        for seq in seqsbyformula[formula]:
            #for p, lines in linepositionsofanalytes[analyteid].items():
            #try:
            bi = constarts[position]
            #except IndexError:
            #    #an example of this came up where an isotopomer of a distribution that had an MS2 scan wasn't in the theoretical distribution, the distributions didn't match that well, and I think I'll just ignore it here for now and work with what I can
            #    continue
            for qualrank, sq in enumerate(qualifiers[position]):
                subindex = bi + sq
                #fragint = intensities[subindex]
                sformula = subformulas[subindex]
                subformularank[seq][sformula] = qualrank
                subformulapercent[seq][sformula] = intensities[subindex]
                linesbysubformula[sformula].update(lines)
                for line in lines:
                    subformulasoflines[line].add(sformula)
                mainindexbysubindex[submatchindex] = mainmatchindex
                submatchsubformulas[submatchindex] = sformula
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

mainindexbysubindex = dict(mainindexbysubindex)
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

print(time() - nt, 'submatch organization')

#otherlinesbyscanbysubformula = defaultdict(lambda: defaultdict(list)) #subformula: scan: [lines] to test the above
#for line, subformulas in subformulasoflines.items():
#    linescans = scansoflines[line]
#    for subformula in subformulas:
#        for scan in linescans:
#            otherlinesbyscanbysubformula[subformula][scan].append(line)
#for k, v in otherlinesbyscanbysubformula.items():
#    for sk, sv in v.items():
#        v[sk] = tuple(sorted(sv))
#    otherlinesbyscanbysubformula[k] = dict(v)
#otherlinesbyscanbysubformula = dict(otherlinesbyscanbysubformula)
#no problems! to re-test you need to sort the tuples in linesbyscanbysubformula

#are there multiple lines in any scans that both contribute the same ms1 subformula to the same ms2 scan?
#^yes and they come from different analyteids so that's alright i guess
##TESTING:
#analytefile = '/'.join((processinglocation, 'analytefactors.pickle'))
#with open(analytefile, 'rb') as pick:
#    analytesbydistribution, distributionsoflines = pickle.load(pick)[2:4]
#
#for sformula, lines in linesbysubformula.items():
#    analyteids = tuple([analytesbydistribution[distributionsoflines[i]] for i in lines])
#    if len(analyteids) != len(set(analyteids)):
#        print(sformula)
#
#for f in linesbyscanbysubformula:
#    for scan, lines in linesbyscanbysubformula[f].items():
#        if len(lines) != len(set(lines)):
#            analyteids = tuple([analytesbydistribution[distributionsoflines[i]] for i in lines])
#            print(f, scan, lines, analyteids)
#different lines of the same analyteid are producing the same subformula for fragmentation?

#0!
#nopes = []
#for formula, scanlines in linesbyscanbysubformula.items():
#    for scan, lines in scanlines.items():
#        if len(lines) > 1:
#            analyteids = tuple([analytesbydistribution[distributionsoflines[i]] for i in lines])
#            if len(analyteids) > len(set(analyteids)):
#                nopes.append(formula)


#for k, v in scansofmainindices.items():
#    scansofmainindices[k] = tuple(v)

#mergablesequences = []
#for smi, sformula in submatchsubformulas.items():
#    seq = submatchsequences[smi]
#    mergablesequences.append((seq, sformula))
#mergablesequences = list(set(mergablesequences))
#so what would a merge of scans and analyteids get you? similar to above
#i feel like there's a good reason to do that

#i'm merging these here so i don't have to generate the same set of fragment ions for a single sequence more than once later
#mergedsequences = list(map(tuple, intersection_merge(mergablesequences))) #could you do this with submatchformulas?

#split sequences prior to merge?
#split subformuals prior to merge - i think this would have the biggest impact
splitmergables = []
#for seq, subformula in mergablesequences:
for smi, subformula in submatchsubformulas.items():
    seq = submatchsequences[smi]
    #mergablesequences.append((seq, sformula))
    outgroup = [seq, subformula]
    subgroups = defaultdict(list) #element: [subiso comps]
    for split in subformula.split(')')[:-1]:
        e = split[0]
        if e == 'C':
            subgroups[e].append(split)
    for e, group in subgroups.items():
        outgroup.append(')'.join((group)) + ')-subsplit')
    splitmergables.append(tuple(outgroup))

#intersection-merging:
# - carbon comps
# - sequences
# - subformulas
#dividedgroups = intersection_merge(splitmergables)

#custom intersection merge to limit the size of each group to whatevers written below
#makes for good memory management, sacrifices some non-redundancy but its super minimal and entirely worth it
sn = 0
groupsofitems = {} #member: group
itemgroups = defaultdict(set) #group: [members]
for items in splitmergables:
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
    if len(itemgroups[joiner]) >= 2500:
        for member in itemgroups[joiner]:
            #by deleting the old locs it will force them incoming items into new groups
            del groupsofitems[member]

dividedgroups = list(itemgroups.values())

for n, group in enumerate(dividedgroups):
    newgroup = tuple(i for i in group if not i.endswith('subsplit'))
    dividedgroups[n] = newgroup

#def scan_counting(scan):
#    if scan['ms level'] == 2:
#        nions = len(scan['m/z array'])
#        ind = scan['index']
#        return {ind: nions}
#    else:
#        return {}
#
#
#msrun.reset()
#
##counting the number of ions in each ms2 scan
#scancounts = {}
#for output in msrun.map(lambda scan: scan_counting(scan), processes=nprocs):
#    scancounts.update(output)
#
#totalions = sum(scancounts.values())
#
##making a somewhat arbitrary measure of complexity for each group in spectabyformula
#groupcounts = {} #formula: sum(nions from all scans) * nseqs * avg seqlength
##for formula, sample in spectrabyformula.items(): #REPLACE with mergablesequences
#for n, group in enumerate(newmerges):
#    groupseqs = []
#    groupsubformulas = []
#    for member in group:
#        if '(' in member:
#            groupsubformulas.append(member)
#        else:
#            groupseqs.append(member)
#    #adapt here
#    avgseqlen = np.mean([len(i) for i in groupseqs])
#    nseqs = len(groupseqs)
#    nsubs = len(groupsubformulas)
#    #nscans = len(list(sample.values())[0])
#    scanions = 0
#    for subformula in groupsubformulas:
#        subindices = subformulasubindices[subformula]
#        mainindices = list(set(mainindexbysubindex[i] for i in subindices)) #its always length 1 as far as i can see
#        for i in mainindices:
#            scans = scansofmainindices[i]
#            for scan in scans:
#                scanions += scancounts[scan]
#    count = scanions * nseqs * avgseqlen * nsubs
#    groupcounts[n] = count
#
#countvalues = np.sort(list(groupcounts.values()))
#totalcounts = countvalues.sum()
#
##reverse organization
#groupindexbycount = defaultdict(list)
#for n, count in groupcounts.items():
#    groupindexbycount[count].append(n)
#
##determine the number of bins based on the initial test dataset as a baseline
##totalcounts = 55123176624 for an 8gb fragmatchlist, make 1gb chunks, therefore make 8 final groups here
##bin != final group
##make the same number of bins, 8. then subdivide each bin 8 times and make the subdivisions ~even in their subdivision total counts
##pair subidivions from each bin together in a way that minimizes their overall difference in division total count across all final groups
#
#basis = 1.7138050002012634e+18 #this value is also going to shift depending on the hardware i guess
##^keep this as a standard but also include an option for this to be modified via script input, some people might have denser/richer data and an input here would be used to prevent memory problems.
##basisbins = 50
#basisbins = 20
#
###ratio to adjust the scale
#adjustment = totalcounts / basis
#nbins = round(basisbins * adjustment)
#ndivisors = nbins - 1
#binsize = np.ceil(len(newmerges) / nbins).astype(int)
#
##test if an additional merging of unrelated groups into ~1/2 the number of total groups would benefit speed
##dividedgroups = [newmerges[i:i+binsize] for i in range(0, len(newmerges), binsize)]
#
##maxfixlength = 10 #max aggregation
##for n, divgroup in enumerate(dividedgroups):
##    divgroup = sorted(divgroup, key=lambda x: -len(x))
###    lens = np.array([len(i) for i in divgroup])
###    fixstart = np.where(lens <= minfixlength)[0][0]
###    newgroup = divgroup[:fixstart]
###    combinedgroups = [tuple(itertools.chain(*divgroup[n:n+maxfixlength])) for n in range(fixstart, len(divgroup), maxfixlength)]
###    newgroup += combinedgroups
###    newgroup = sorted(newgroup, key=lambda x: -len(x))
###    dividedgroups[n] = newgroup
##    dividedgroups[n] = divgroup
#
##current output is an 8gb fragment list, i want to split them into 1gb pieces assuming all my counts here equal to that, i can use that as a basis for dividing future files
##^might go wrong b/c i'm not representing ms2 ions here at all, but meh what else can i do atm
##totalcounts / 8gb = 6941367318.375
##i guess i can incorporate the number of ms2 scans here too
##totalcounts / totalions = 4.659340143007824
##incorporate system memory too
##psutil.virtual_memory().total
##i'm not going to care about available atm i guess? this would be a good spot to though, given that this single script is really light
##maybe count the ions in each scan and use that?
#
##use a 20% safety margin, or base the safety margin on the density of the ms2 scans? ions per scan would come in handy here
##divide into groups that have an even spread of each number of counts
##^so group the counts into ~clusters and distribute the clusters evenly
##also order the most abundant counts to be evenly spaced out within each division
#
##minimization of differences:
## - keep each bin/division as a sorted list
## - from the bin with the largest amount of sum total distance from all of the other bins:
##   > if this bin also has the least amount of members in it, then favor taking large numbers to fix the difference
##   > if this bin also has the most amount of members in it, favor taking groups of smaller numbers to fix the difference
##  -> subtract these members and distribute them amongst the other bins in a way that minimzes the sum total difference. AND the original bin they came from is still a candidate for where they can go? probably won't need that given the assessment of what bins leave will probably account for this.
#
#minbinbound = min(countvalues) - 1
#maxbinbound = max(countvalues) + 1
#binsize = (maxbinbound - minbinbound) / nbins
#binboundaries = minbinbound + (np.arange(ndivisors) + 1) * binsize
#binboundaries = np.insert(binboundaries, 0, minbinbound)
#binboundaries = np.insert(binboundaries, binboundaries.shape[0], maxbinbound)
#pairedboundaries = np.stack((binboundaries[:-1], binboundaries[1:]), axis=1)
#
#binnedgroups = {} #bin: countvalue
#for n, (l, r) in enumerate(pairedboundaries.tolist()):
#    binnedgroups[n] = countvalues[np.logical_and(countvalues >= l, countvalues < r)][::-1] #reverse the sorting for future iterations
#
##for k, v in binnedgroups.items():
##    print(k, len(v))
##0 432169
##1 1709
##2 283
##3 83
##4 33
##5 11
##6 5
##7 1
##there will always be more on the lower side it seems?
#
##so i'll start from the bottom while dividing out what there is
##maybe i can start organizing from the top to figure out what needs to be distributed?
##i can keep the total count, or the total count of each bin, handy, in order to leave voids that can be filled as i go up the bins
##actually yeah, maybe i should start from the top and use the bottom pieces to make up for the differences, that might be easier than distributing the largest things last.
##^this does heavily rely on there being more of the lower bins, always, so i should print this as output from the engine and put in a check for if this ever doesn't happen.
##as you go through the divisions, add things from the current bin into the division with the lowest sum, and iterate through the bin's countvalues in order from highest to lowest
##the idea of iterating the counts in descending order is based on the idea that the differences between divisions should get lower and lower by doing this
#
#divisions = {k: [] for k in binnedgroups}
#dsums = np.array([sum(v) for v in divisions.values()])
#for b, counts in reversed(binnedgroups.items()):
#    for c in counts.tolist():
#        dkey = dsums.argmin()
#        divisions[dkey].append(c)
#        dsums[dkey] += c
#
##order the divisions to spread out the large ones amongst the smaller
##pad the largest values based on nprocs i suppose
#
##start with the highest - they're already ordered like this
##iterate all the potential iteration spots, add an n+1 for each insertion, bisect
#ordereddivisions = defaultdict(list)
#for k, div in divisions.items():
#    ordereddivisions[k].insert(0, div.pop(0))
#    while div:
#        n = 0
#        try:
#            for c in range(len(ordereddivisions[k])+1):
#                ordereddivisions[k].insert(c+n, div.pop(0))
#                n += 1
#        except IndexError:
#            break
#
##explanation
##for k, v in ordereddivisions.items():
##    plt.plot(v, '-')
##    plt.show()
#
##converting the counts into formulas and distributing them across divisions
#dividedgroups = defaultdict(list) #divisionkey: [formulas]
#for k, div in ordereddivisions.items():
#    for c in div:
#        dividedgroups[k].append(newmerges[heapq.heappop(groupindexbycount[c])])
#
#dividedgroups = [list(v) for v in dividedgroups.values()]


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

mainindexfile = '/'.join((processinglocation, 'mainindicesbysubindex.pickle'))
with open(mainindexfile, 'wb') as pick:
    pickle.dump(mainindexbysubindex, pick)

#submatchpositionsfile = '/'.join((processinglocation, 'submatchpositions.pickle'))
#with open(submatchpositionsfile, 'wb') as pick:
#    pickle.dump(submatchpositions, pick)

linesbyscanbysubformulafile = '/'.join((processinglocation, 'linesbyscanbysubformula.pickle'))
with open(linesbyscanbysubformulafile, 'wb') as pick:
    pickle.dump(linesbyscanbysubformula, pick)

submatchsubformulasfile = '/'.join((processinglocation, 'submatchsubformulas.pickle'))
with open(submatchsubformulasfile, 'wb') as pick:
    pickle.dump(submatchsubformulas, pick)

subformularankfile = '/'.join((processinglocation, 'subformularank.pickle'))
with open(subformularankfile, 'wb') as pick:
    pickle.dump(subformularank, pick)

subformulapercentfile = '/'.join((processinglocation, 'subformulapercent.pickle'))
with open(subformulapercentfile, 'wb') as pick:
    pickle.dump(subformulapercent, pick)
