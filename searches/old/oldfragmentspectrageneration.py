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
from mpire import WorkerPool
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
filename = '/'.join((processinglocation, 'fragment.matches.csv'))
proteome = 'Human_Homo_sapien'
nprocs = 8
subisotopomericdepth = 0.8
proton = 1.007276554940804
ppmtol = 25

environment_partial = partial(lmdb.Environment, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=9999, max_spare_txns=1, lock=True)

isotopomerpositionsfile = '/'.join((processinglocation, 'isotopomersbypositions.pickle'))
with open(isotopomerpositionsfile, 'rb') as pick:
    isotopomerpositionsofanalytes = pickle.load(pick)
#isotopomerpositionsofanalytes = defaultdict(set) #analyteid: isotopopmer coordinate from max

spectrabyformulafile = '/'.join((processinglocation, 'spectrabyformula.pickle'))
with open(spectrabyformulafile, 'rb') as pick:
    #basespectrabyformula = pickle.load(pick)
    spectrabyformula = pickle.load(pick)
#spectrabyformula = defaultdict(lambda: defaultdict(set)) #formula: analyteid: scan

divisionfile = '/'.join((processinglocation, 'dividedformulas.pickle'))
with open(divisionfile, 'rb') as pick:
    dividedformulas = pickle.load(pick)
#dividedformulas = defaultdict(list) #divisionkey: [formulas]

#generate csv headers

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

elementvector = [0 for _ in elementalmasses]
elementlist = list(elementalmasses)
vectorpositions = {k: n for n, k in enumerate(elementlist)}
elementpositions = {n: k for n, k in enumerate(elementlist)}
vectorrangesbyelement = {'H': range(0,2),
                         'C': range(2,4),
                         'N': range(4,6),
                         'O': range(6,9),
                         'S': range(9,13)}

nonmonoisotopicelements = {'H2', 'C13', 'N15', 'O17', 'O18', 'S33', 'S34', 'S36'}

isotopesbyelement = {
        'H': ('H1', 'H2'),
        'C': ('C12', 'C13'),
        'N': ('N14', 'N15'),
        'O': ('O16', 'O17', 'O18'),
        'S': ('S32', 'S34', 'S33', 'S36')} #in order of abundance

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

with environment_partial(librarylocation) as env:
    aas = env.open_db('aminoacids'.encode())
    with env.begin(write=False) as txn:
        with txn.cursor(aas) as cursor:
            aaget = cursor.get(proteome.encode()).decode()
            aminoacidcomposition = eval(aaget)

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


def fragmentation_compositions(seq):
    fragments = {}

    # Calculate the compositions of the n-term fragments
    fragcomp_n = {}
    for n, aa in enumerate(seq[:-1]):  
        aa_composition = aminoacidcomposition[aa]
        for k in aa_composition:
            fragcomp_n[k] = fragcomp_n.get(k, 0) + aa_composition.get(k, 0)
        for ion, modcomp in nfragmentcompositions.items():
            fragment_composition = fragcomp_n.copy()
            for k in modcomp:
                fragment_composition[k] = fragment_composition.get(k, 0) + modcomp.get(k, 0)
            fragments[ion + str(n + 1)] = fragment_composition

    # Calculate the compositions of the c-term fragments
    fragcomp_c = {}
    for n, aa in enumerate(seq[::-1][:-1]): 
        aa_composition = aminoacidcomposition[aa]
        for k in aa_composition:
            fragcomp_c[k] = fragcomp_c.get(k, 0) + aa_composition.get(k, 0)
        for ion, modcomp in cfragmentcompositions.items():
            fragment_composition = fragcomp_c.copy()
            for k in modcomp:
                fragment_composition[k] = fragment_composition.get(k, 0) + modcomp.get(k, 0)
            fragments[ion + str(n + 1)] = fragment_composition

    return fragments

def max_estimation(count, probarray):
    subestimates = [round(count * -p) for p in probarray]
    total = sum(subestimates)
    diff = total - count

    if diff < 0:
        testcomps = [[e + (i == n) for i, e in enumerate(subestimates)] for n in range(len(subestimates))]
    elif diff > 0:
        #testcomps = [[e - (i == n or e == 0) for i, e in enumerate(subestimates)] for n in range(len(subestimates))] #was replaced idk why
        testcomps = [[e - (i == n and e > 0) for i, e in enumerate(subestimates)] for n in range(len(subestimates))]
    else:
        testcomps = [subestimates[:-1] + [subestimates[-1]]] + [[e - (i == n) + (i == n+1) for i, e in enumerate(subestimates)] for n in range(len(subestimates)-1)]

    maxprob = -float('inf')
    maxvec = None

    for comp in testcomps:
        log_newprob = sum(c * math.log(-p) for p, c in zip(probarray, comp))  # compute the log of newprob
        pn = 0
        for nn, c in enumerate(comp):
            if nn > 0:
                log_newprob += math.log(math.comb(count - pn, c))  # use the log of comb
                pn += c
        if log_newprob > maxprob:
            maxprob = log_newprob
            maxvec = comp
    
    return maxvec, math.exp(maxprob)

def max_fragment(fragprobs, fragcomp, fragint):
    estimates = defaultdict(list)
    isoorders = defaultdict(list)
    for iso, prob in fragprobs.items():
        e = iso[0]
        ind = bisect(estimates[e], -prob)
        estimates[e].insert(ind, -prob)
        isoorders[e].insert(ind, iso)
    mass = 0
    prob = fragint
    for e, c in fragcomp.items():
        if c > 0:
            if len(estimates[e]) > 1:
                #print(e)
                maxvec, elementprob = max_estimation(c, estimates[e])
                for n, a in enumerate(maxvec):
                    mass += elementalmasses[isoorders[e][n]] * a
                prob *= elementprob
            else:
                #p = estimates[e][0]
                iso = isoorders[e][0]
                mass += elementalmasses[iso] * c
    return mass, prob

def frag_match(formula, samples):
#def frag_match(neighbors, formula, samples, qualifiers, conlengths, subformulas, massesandintensities):
    fragmatches = []
    qualifiers = subisodepthqualifiers[formula]
    conlengths = condensationcoordinates[formula]
    conends = conlengths.cumsum()
    constarts = conends - conlengths
    subformulas = [i.decode() for i in abundanceformulas[formula]]
    #subformulas = [i.decode() for i in subformulas]
    massesandintensities = abundances[formula]
    intensities = massesandintensities[1]
    sumintensities = np.array([intensities[s:e].sum() for s, e in zip(constarts.tolist(), conends.tolist())])
    maxintensityindex = sumintensities.argmax()
    positions = set() #the dist positions to check
    for analyteid, sids in samples.items():
        positions.update(isotopomerpositionsofanalytes[analyteid])
    positions = [i + maxintensityindex for i in positions]
    for seq in seqsbyformula[formula]:
        fragions = []
        fragints = []
        fragmasses = []
        fragindices = []
        fragpositions = []
        outdict = Counter()
        fragments = fragmentation_compositions(seq)
        #this position matching could potentially be wrong under less strict distributionmatching schemes. i want this to become more flexible.
        for p in positions:
            try:
                bi = constarts[p]
                #ei = conends[p]
            except IndexError:
                #an example of this came up where an isotopomer of a distribution that had an MS2 scan wasn't in the theoretical distribution, the distributions didn't match that well, and I think I'll just ignore it here for now and work with what I can
                #in the future you'll need to generate single proton-location ms1 isos to make up for this, but you might be able to do it above outside the loop
                continue
            subquals = qualifiers[p]
            for sq in subquals:
                subindex = bi + sq
                #print(f'{formula} {analyteid} {seq} {p} {subindex}')
                #just append subindex?
                #you only need one of these, i think i re-iterate the same thing too many times below
                #^for each individual seq, yeah
                #^no, sometimes things loop twice but idk why
                sformula = subformulas[subindex]
                outdict[sformula] += 1
                #setting up subformula-specific probabilities
                fragint = intensities[subindex]
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
                fragprobs = {}
                for e, v in competitors.items():
                    if e in competing:
                        for iso, c in competitors[e].items():
                            fragprobs[iso] = c / isosums[e]
                    else:
                        for iso in v:
                            fragprobs[iso] = 1
                for ion, fragcomp in fragments.items():
                    fm, fi = max_fragment(fragprobs, fragcomp, fragint)
                    fragmasses.append(fm)
                    fragints.append(fi)
                    fragpositions.append(p)
                    fragindices.append(subindex)
                    fragions.append(ion)
        #if max(outdict.values()) > 1:
        #    print(f'{seq}, {analyteid}, {formula}, {max(outdict.values()}), ~~')
        fragmasses = np.array(fragmasses)[:,None] + proton #assuming 1+ for now
        fragtol = (fragmasses / 1000000 * ppmtol).flatten()
        #for sid in samples[analyteid]:
        #    #dists, inds = nn.query(fragmasses, workers=1)
        #    matches = neighbors[sid].query_ball_point(fragmasses, fragtol)
        for analyteid, sids in samples.items():
            for sid in sids:
                matches = neighbors[sid].query_ball_point(fragmasses, fragtol)
                for n, m in enumerate(matches.tolist()):
                    for sm in m:
                        mlist = (seq, sm, fragions[n], fragmasses[n][0], fi, sid, analyteid, fragpositions[n], fragindices[n])
                        fragmatches.append(mlist)
    return fragmatches

#moving forward:
#convert input into deconvoluted form
#include [ion-type, ion-position, intensity rank, experimental intensity, experimental mass]
#explore simulating cofragmenting ions, see if any 1+, 2+ misidentifications occur
#for identified dists of each scan, check the 2nd theoretical isotopomer position and remove things that don't work
# ^perhaps use that as a chance to change the csv to a pickle and delete the csv
#i need to add a parameter of distributionid, but i guesss this is sampleid
# - so basically, the csv will be used to organize on multiple levels, in order of operation:
#   > sequence combinations compete over a scan group (aka scans on the same distribution/charge state/analyteid)
#       -> i'm thinking the individual-sequence optimization, to ignore ~bad matches could take place at the combination level to minimize co-entropy
#   > scan groups compete over sequences that are in other scan groups
# - ie, the linked scan of dists within a window need to make their entropy metrics. afterwards, where other linked scan groups have the same sequences as other groups - these need to also compete, and their competition will form some kind of hierarchy aka ranking
#i also need to know the charges of each analyteid in a scan, i think this is scanalytecharges, maybe use this to determine if a match can be made to a specific ion once ms2 deconvolution is added
# ^ perhaps organize them by charge to begin with? and somehow that lets you do the nn on everything at once still?
#co-entropy and entropy itself has a problem when there's no competition. an MS1 distribution that can only belong to one sequence, but isn't actually that sequence, might pose problems here?

#to consider for identification judgement:
#number of matches, optimized squared diff error, break clusters
#do the break clusters first and use that as a dataset to reference when going over matches

def fragment_divisions(divgroup, divkeys):
    #spectrabyformula = {i:basespectrabyformula[i] for i in divkeys}
    fspectrabyformula = {i:spectrabyformula[i] for i in divkeys}

    #global seqsbyformula
    #global abundances
    #global abundanceformulas
    #global condensationcoordinates
    #global subisodepthqualifiers
    #seqsbyformula = {} #formula: [seqs]
    #abundances = {} #formula: [[masses], [intensities]]
    #abundanceformulas = {} #formula: subformulas
    #condensationcoordinates = {} #formula: [# isotopomers per proton-step]
    #subisodepthqualifiers = {} #formula: [[top n subisos of proton locations]]
    #encodedkeys = [i.encode() for i in spectrabyformula]
    #with environment_partial(librarylocation) as env:
    #    seqdb = '.'.join(('seqsbyformula', proteome))
    #    seqs = env.open_db(seqdb.encode())
    #    with env.begin(write=False) as txn:
    #        with txn.cursor(seqs) as cursor:
    #            for k, v in cursor:
    #                key = k.decode()
    #                value = eval(v.decode())
    #                seqsbyformula[key] = value
    #    ddb = env.open_db('distributions.formulas'.encode())
    #    with env.begin(write=False) as txn:
    #        with txn.cursor(ddb) as cursor:
    #            for k, v in cursor.getmulti(encodedkeys):
    #                abundanceformulas[k.decode()] = eval(v.decode())
    #    condensationdb = env.open_db('distributions.condensation'.encode())
    #    with env.begin(write=False) as txn:
    #        with txn.cursor(condensationdb) as cursor:
    #            for k, v in cursor.getmulti(encodedkeys):
    #                condensationcoordinates[k.decode()] = np.frombuffer(v, dtype=int)
    #    subisoqualdb = env.open_db('distributions.subisoqualifiers'.encode())
    #    with env.begin(write=False) as txn:
    #        with txn.cursor(subisoqualdb) as cursor:
    #            for k, v in cursor.getmulti(encodedkeys):
    #                subisodepthqualifiers[k.decode()] = eval(v.decode())
    #    fulldb = env.open_db('distributions.full'.encode())
    #    with env.begin(write=False) as txn:
    #        with txn.cursor(fulldb) as cursor:
    #            for k, v in cursor.getmulti(encodedkeys):
    #                out = np.frombuffer(v)
    #                out = out.reshape(2, out.size//2)
    #                abundances[k.decode()] = out
    #print(len(seqsbyformula), 'formulas')

    #msrun.reset()
    #
    #allsamples = set()
    #for sample in spectrabyformula.values():
    #    for sids in sample.values():
    #        allsamples.update(sids)

    #nt = time()
    #global neighbors
    #neighbors = {}
    #allsamples = tuple(allsamples)
    #for sid in allsamples:
    #    scan = msrun[sid]
    #    masses = scan['m/z array']
    #    intensities = scan['intensity array']
    #    neighbors[sid] = spatial.KDTree(masses[:,None])
    #print(time() - nt, 'neighbors')
    
    #frag_match_partial = partial(frag_match, neighbors)

    #funcs = []
    #for formula, samples in spectrabyformula.items():
    #    funcs.append(partial(frag_match, neighbors, formula, samples, subisodepthqualifiers[formula], condensationcoordinates[formula], abundanceformulas[formula], abundances[formula]))
    #
    #nt = time()
    #fragmatchlist = []
    #with concurrent.futures.ProcessPoolExecutor(nprocs) as executor:
    #    futures = []
    #    for func in funcs:
    #        futures.append(executor.submit(func))
    #    for f in concurrent.futures.as_completed(futures):
    #        fragmatchlist.append(f.result())
    #print(time() - nt)
    #
    #nt = time()
    #results = []
    #fragmatchlist = []
    #with mp.Pool(nprocs) as pool:
    #    #args = []
    #    for formula, samples in spectrabyformula.items():
    #        #args.append([formula, samples, subisodepthqualifiers[formula], condensationcoordinates[formula], abundanceformulas[formula], abundances[formula]])
    #        results.append(pool.apply_async(frag_match_partial, args=(formula, samples, subisodepthqualifiers[formula], condensationcoordinates[formula], abundanceformulas[formula], abundances[formula])))
    #    for r in results:
    #        fragmatchlist.append(r.get())
    #print(time() - nt)

    nt = time()
    #fragmatchlist = []
    #with mp.Pool(nprocs) as pool:
    with WorkerPool(n_jobs=nprocs) as pool:
        #for ol in pool.starmap(frag_match_partial, args):
        #for ol in pool.starmap(frag_match, fspectrabyformula.items()):
        #    fragmatchlist.extend(ol)
        fragmatchlist = pool.map(frag_match, fspectrabyformula.items())
    #fragmatchlist = []
    #with concurrent.futures.ProcessPoolExecutor(nprocs) as ex:
    #    futures = []
    #    for arg in args:
    #        futures.append(ex.submit(frag_match_partial, *arg))
    #    for f in concurrent.futures.as_completed(futures):
    #        fragmatchlist.append(f.result())
    print(time() - nt, 'raw process time')
    
    nt = time()
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(fragmatchlist)
    print(time() - nt, 'saved')
    
    #del seqsbyformula
    #del abundances
    #del abundanceformulas
    #del condensationcoordinates
    #del subisodepthqualifiers
    #del neighbors

seqsbyformula = {} #formula: [seqs]
abundances = {} #formula: [[masses], [intensities]]
abundanceformulas = {} #formula: subformulas
condensationcoordinates = {} #formula: [# isotopomers per proton-step]
subisodepthqualifiers = {} #formula: [[top n subisos of proton locations]]
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
print(len(seqsbyformula), 'formulas')

msrun = mzml.MzML(mzmlfile, dtype=np.float64)

allsamples = set()
for sample in spectrabyformula.values():
    for sids in sample.values():
        allsamples.update(sids)

nt = time()
global neighbors
neighbors = {}
allsamples = tuple(allsamples)
for sid in allsamples:
    scan = msrun[sid]
    masses = scan['m/z array']
    intensities = scan['intensity array']
    neighbors[sid] = spatial.KDTree(masses[:,None])
print(time() - nt, 'neighbors')

#make csv headers
with open(filename, 'w') as f:
    writer = csv.writer(f)
    line = ['sequence', 'match', 'ion', 'theoretical-mass', 'frag-intensity', 'scanid', 'analyteid', 'fragposition', 'subindex']
    writer.writerow(line)

nt = time()
for divgroup, divkeys in dividedformulas.items():
    it = time()
    fragment_divisions(divgroup, divkeys)
    print(divgroup, '-', time() - it)
    print('~~~')
    break
print(time() - nt, 'total')

#collect [seq, ion, experimental mass, experimental mass ind (this will be from the deconvoluted data), raw mass distance error, theoretical abundance, spectra intensity, scan, analyteid, position]
#i won't be able to get spectra intensity or mass distance yet b/c of how i'm using the nn

#check the memory impact of seqsbyformula, abundances, etc, when you load the entire dataset of them. this might not be such a bad impact and it would probably contribute to speed.
#^yes, its totally worth changing it
#3gb vs 2.24 base cost of full vs partial loading

#from deconvoluted ms2 scan info, i can present intensity rank directly within the output list, dealing with it later gets tricky


#i'm going to start off assuming that the same fragments should have the same relative prevalence regardless of isotopic location or composition
#so it would be a good idea to base a metric off finding the most true relative ratios of fragments across ions 
#so then make a way to visualize a line, its ms2 hit points, other dists that aren't currently described by the plots but are in other hits, and the consistency of fragment ratios of different isotopomers
#also make a way to visualize two potential sequence matches next to each other
#on the technical side, facilitate the matching process







#your coordinate system might be wrong?
#the distance from max should always be based on experimental distribution
#then, match the theoretical dist to the right line by mass
