from collections import defaultdict, Counter
import itertools
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from time import time
import random
import profile
import bisect
import heapq
import math
import os

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
        '#e85d58',
        '#b88cfa',
        '#f5972c',
        '#2ded8d',
        '#4bc8f2',
        '#ea68f2',
        ]
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', chexes)
#n = 0
#for c in chexes:
#    plt.plot([n, n+1], [n, n+1], color=c)
#    n += 1
#plt.show()

#source:
#https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl
#https://physics.nist.gov/cgi-bin/cuu/Value?mp
#^on this page, values in parenthesis break the summing to 1

proton = 1.007276554940804

nelementalprobabilities = {
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

nelementalmasses = {
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

nmajorisotopemasses = {
            'H': 1.00782503223,
            'C': 12.0000000, 
            'N': 14.00307400443,
            'O': 15.99491461957,
            'S': 31.9720711744}

elementvector = [0 for _ in nelementalmasses]
elementlist = list(nelementalmasses)
vectorpositions = {k: n for n, k in enumerate(elementlist)}
elementpositions = {n: k for n, k in enumerate(elementlist)}
vectorslicesbyelement = {'H': slice(0,2),
                         'C': slice(2,4),
                         'N': slice(4,6),
                         'O': slice(6,9),
                         'S': slice(9,14)}
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

elementvectors = {}
nvectorpositions = {}
nelementpositions = {}
for e, isos in isotopesbyelement.items():
    elementvectors[e] = [0 for _ in range(len(isos))]
    nvectorpositions[e] = {k: n for n, k in enumerate(isos)}
    nelementpositions[e] = {n: k for n, k in enumerate(isos)}

nmassadditions = {}
for k, v in nelementalmasses.items():
    nmassadditions[k] = v - nmajorisotopemasses[k[0]]
    

elementalmasses = {
            'H': {1: 1.00782503223, 2: 2.01410177812},
            'C': {12: 12.00000000, 13: 13.00335483507},
            'N': {14: 14.00307400443, 15: 15.00010889888},
            'O': {16: 15.99491461957, 17: 16.99913175650, 18: 17.99915961286},
            'S': {32: 31.9720711744, 33: 32.9714589098, 34: 33.967867004, 36: 35.96708071}}

elementalprobabilities = {
        'H': {1: 0.999885, 2: 0.000115},
        'C': {12: 0.9893, 13: 0.0107}, 
        'N': {14: 0.99636, 15: 0.00364},
        'O': {16: 0.99757, 17: 0.00038, 18: 0.00205},
        'S': {32: 0.9499, 33: 0.0075, 34: 0.0425, 36: 0.0001}}

isotopes = {}
majorisotopemasses = {}
for k, v in elementalmasses.items():
    probs = elementalprobabilities[k]
    isotopes[k] = {}
    n = 0
    for sk, sv in v.items():
        isotopes[k][sv] = probs[sk]
        if n == 0:
            majorisotopemasses[k] = sv
            n += 1

massadditions = defaultdict(dict) #element: mass addition: probability
isotopomersbyaddition = defaultdict(dict) #element: mass addition: isotopomer
for e, i in isotopes.items():
    maxprob = max(i.values())
    dominantisotope = [k for k, v in i.items() if v == maxprob][0]
    for m, p in i.items():
        addmass = m - dominantisotope
        massadditions[e][addmass] = p
        
        elecount = [k for k, v in elementalprobabilities[e].items() if v == p][0]
        isotopomersbyaddition[e][addmass] = elecount
massadditions = dict(massadditions)
isotopomersbyaddition = dict(isotopomersbyaddition)
    
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

#!~~~~~ New model below

#next_dist_gen is unfinished
def next_dist_gen(atomiccomposition, minimumabundance):
    atomiccomposition = Counter()
    for aa in seq:
        atomiccomposition += aminoacidcomposition[aa]
    atomiccomposition['H'] += 2
    atomiccomposition['O'] += 1
    natomicsample = Counter()
    for e, c in atomiccomposition.items():
        for iso in isotopesbyelement[e]:
            natomicsample[iso] = nelementalprobabilities[iso] * c
    
    basecomposition = {}
    for e, v in isotopesbyelement.items():
        sumcount = 0
        for si in v:
            startcount = round(natomicsample[si])
            #^i tried generating random peptides in order to find a split that came out to 0.5 here but I'm unable to find an example where it actually happens. Hypothetically, if it does, there would be an equal probability between two isotopes, and this wouldn't handle that well. they'll be generated in the below loop regardless I suppose.
            basecomposition[si] = startcount
            #if startcount > 0:
            sumcount += startcount
                #basecomposition[si] = startcount
        if sumcount < atomiccomposition[e]:
            #mono isotope rounded down but nothing could fill the gap, add to mono isotope
            #basecomposition[monoisotopickeys[e]] += 1
            basecomposition[monoisotopickeys[e]] += 1
        elif sumcount > atomiccomposition[e]:
            #this has never happened, and probably won't
            #multiple things rounded up.. presumably because there's an even split
            #subtracting from the mono isotope would be the easiest move rather than identifying which non-mono isotope was added
            #basecomposition[monoisotopickeys[e]] -= 1
            basecomposition[monoisotopickeys[e]] -= 1
            print(seq, 'generated an erroneous example i\'ve been trying to catch with isotopic distribution rounding')
    
    
    branchkey = 0 #this is used as a basis for accessing subheaps
    
    #below calculates the mass and abundance of the most abundant isotopomer, and organizes the binomials its made from
    baseprob = 1
    basemass = 0
    baseoffsets = defaultdict(int)
    branchprobabilitiesbyelement = defaultdict(lambda: defaultdict(lambda: 1)) #branchkey: e: prob
    branchmassesbyelement = defaultdict(lambda: defaultdict(int)) #branchkey: e: mass
    #branchcompositionbyelement = defaultdict(lambda: defaultdict(lambda: Counter())) #branchkey: e: {composition dict}
    branchcompositionbyelement = defaultdict(lambda: defaultdict(dict)) #branchkey: e: {composition dict}
    for iso, c in basecomposition.items():
        if c > 0:
            e = iso[0]
            prob = nelementalprobabilities[iso]**c
            mass = nelementalmasses[iso] * c
            baseprob *= prob
            basemass += mass
            branchprobabilitiesbyelement[branchkey][e] *= prob
            branchmassesbyelement[branchkey][e] += mass
            branchcompositionbyelement[branchkey][e][iso] = c
            if iso in nonmonoisotopicelements:
                combfactor = special.comb(atomiccomposition[e] - baseoffsets[e], c, exact=True)
                baseprob *= combfactor
                branchprobabilitiesbyelement[branchkey][e] *= combfactor
                baseoffsets[e] += c
    
    baseformula = ''.join((sorted(f'{k}({v})' for k, v in basecomposition.items() if v > 0)))
    
    expansiondirections = {} #branchkey: 1 or -1, a negative direction is only given to nonmonisotopic elements that are in basecomposition generated above
    isotopesbyisodirection = {} #isodirection: iso
    isodirectionsbyelement = defaultdict(list) #e: [isodirections]
    opposingdirections = {} #isodirection: isodirection of the same element moving in the other direction, I don't let these coexist within the same subheap
    
    isodirection = 0
    #i think i can separate supheapkeys from initial probability calculations
    #in here i should aim to generate all subheap probabilities, and if they're abundances passes -> put their composition in for the next round
    for e, isos in nonmonoisotopicgroups.items():
        if e in atomiccomposition:
            for iso in isos:
                #positive direction
                isotopesbyisodirection[isodirection] = iso
                isodirectionsbyelement[e].append(isodirection)
                expansiondirections[isodirection] = 1
                isodirection += 1
                if basecomposition[iso] > 0:
                    #negative direction
                    isotopesbyisodirection[isodirection] = iso
                    isodirectionsbyelement[e].append(isodirection)
                    expansiondirections[isodirection] = -1
                    #there are two directions for this iso from basecomp
                    opposingdirections[isodirection] = isodirection - 1
                    opposingdirections[isodirection - 1] = isodirection
                    isodirection += 1
    
    finalprobabilities = {} #branchkey: abundance
    finalformulas = {} #branchkey: subformula
    finalmasses = {} #branchkey: mass
    
    finalprobabilities[branchkey] = baseprob
    finalformulas[branchkey] = baseformula
    finalmasses[branchkey] = basemass
    
    branchcount = 1
    currentbranchkeys = [] #list of all currently unexplored branchkeys
    priorbranch = {} #branchkey: branchkey of branch that generated this branch
    branchopposers = defaultdict(set) #branchkey: set of non-compatible isodirections
    branchprobabilities = defaultdict(dict) #branchkey: isodirection: combined element prob
    branchmasses = defaultdict(dict) #branchkey: isodirection: combined element mass
    branchcompositions = defaultdict(dict) #branchcount: isotope: count
    branchisodirections = {} #branchkey: tailored version of isotopesbyisodirection
    subformulasets = set()
    subformulasets.add(baseformula)
    
    branchisodirections[branchkey] = isotopesbyisodirection.copy()
    
    #in here i should aim to generate all subheap probabilities, and if they're abundances passes -> put their composition in for the next round
    for isodirection, iso in isotopesbyisodirection.items():
    #for e, isos in nonmonoisotopicgroups.items():
        e = iso[0]
        acount = atomiccomposition[e]
        mk = monoisotopickeys[e]
        direction = expansiondirections[isodirection]
        newelementcomp = branchcompositionbyelement[branchkey][e].copy()
        if iso in newelementcomp and newelementcomp[iso] > 0:
            newelementcomp[iso] += direction
        else:
            newelementcomp[iso] = 1
        newelementcomp[mk] -= direction
        n = 0
        newelementmass = 0
        newelementprob = 1
        for loopiso, c in newelementcomp.items():
            if c > 0:
                newelementmass += nelementalmasses[loopiso] * c
                newelementprob *= nelementalprobabilities[loopiso]**c
                if loopiso in nonmonoisotopicelements:
                    #newelementprob *= special.comb(acount-n, c, exact=True)
                    newelementprob *= math.comb(acount-n, c)
                    n += c
        newprob = baseprob / branchprobabilitiesbyelement[branchkey][e] * newelementprob
        if newprob >= minimumabundance:
            newmass = basemass - branchmassesbyelement[branchkey][e] + newelementmass
            #modify basecomp via elementcomp
            newbasecomp = basecomposition.copy()
            newbasecomp[iso] += direction
            newbasecomp[mk] -= direction
            #subformula = ''.join((f'{k}({newbasecomp[k]})' for k in sorted(newbasecomp) if newbasecomp[k] > 0))
            subformula = ''.join((sorted(f'{k}({v})' for k, v in newbasecomp.items() if v > 0)))
            subformulasets.add(subformula)
            #add everything to new branchthings via branchcount!
            branchprobabilitiesbyelement[branchcount] = branchprobabilitiesbyelement[branchkey].copy()
            branchprobabilitiesbyelement[branchcount][e] = newelementprob
            branchmassesbyelement[branchcount] = branchmassesbyelement[branchkey].copy()
            branchmassesbyelement[branchcount][e] = newelementmass
            branchcompositions[branchcount] = newbasecomp
            branchcompositionbyelement[branchcount] = branchcompositionbyelement[branchkey].copy()
            branchcompositionbyelement[branchcount][e] = newelementcomp
            finalprobabilities[branchcount] = newprob
            finalformulas[branchcount] = subformula
            finalmasses[branchcount] = newmass
            priorbranch[branchcount] = branchkey
            if newbasecomp[iso] == 0:
                #negative direction ended, remove that isodirection in the future branches
                del branchisodirections[branchkey][isodirection]
            if newbasecomp[mk] == 0:
                #end of a positive direction, remove that isodirection in the future branches
                del branchisodirections[branchkey][isodirection]
            if isodirection in opposingdirections:
                #remove opposing direction from isodirections
                branchopposers[branchcount].add(opposingdirections[isodirection])
            currentbranchkeys.append(branchcount)
            branchcount += 1
    
    while currentbranchkeys:
        nextbranchkeys = []
        for branchkey in currentbranchkeys:
            prior = priorbranch[branchkey]
            branchisodirections[branchkey] = branchisodirections[prior].copy()
            for isodirection, iso in branchisodirections[prior].items():
                e = iso[0]
                acount = atomiccomposition[e]
                mk = monoisotopickeys[e]
                direction = expansiondirections[isodirection]
                newelementcomp = branchcompositionbyelement[branchkey][e].copy()
                if iso in newelementcomp and newelementcomp[iso] > 0:
                    newelementcomp[iso] += direction
                else:
                    newelementcomp[iso] = 1
                newelementcomp[mk] -= direction
                n = 0
                newelementmass = 0
                newelementprob = 1
                for loopiso, c in newelementcomp.items():
                    if c > 0:
                        newelementmass += nelementalmasses[loopiso] * c
                        newelementprob *= nelementalprobabilities[loopiso]**c
                        if loopiso in nonmonoisotopicelements:
                            #newelementprob *= special.comb(acount-n, c, exact=True)
                            newelementprob *= math.comb(acount-n, c)
                            #^now most time is spent on this comb function, is there a way to summarize it using the info at hand? stackexchange question
                            n += c
                newprob = finalprobabilities[branchkey] / branchprobabilitiesbyelement[branchkey][e] * newelementprob
                if newprob >= minimumabundance:
                    newbasecomp = branchcompositions[branchkey].copy()
                    newbasecomp[iso] += direction
                    newbasecomp[mk] -= direction
                    subformula = ''.join((sorted(f'{k}({v})' for k, v in newbasecomp.items() if v > 0)))
                    if subformula not in subformulasets:
                        subformulasets.add(subformula)
                        newmass = finalmasses[branchkey] - branchmassesbyelement[branchkey][e] + newelementmass
                        #modify basecomp via elementcomp
                        #add everything to new branchthings via branchcount!
                        branchprobabilitiesbyelement[branchcount] = branchprobabilitiesbyelement[branchkey].copy()
                        branchprobabilitiesbyelement[branchcount][e] = newelementprob
                        branchmassesbyelement[branchcount] = branchmassesbyelement[branchkey].copy()
                        branchmassesbyelement[branchcount][e] = newelementmass
                        branchcompositions[branchcount] = newbasecomp
                        branchcompositionbyelement[branchcount] = branchcompositionbyelement[branchkey].copy()
                        branchcompositionbyelement[branchcount][e] = newelementcomp
                        finalprobabilities[branchcount] = newprob
                        finalformulas[branchcount] = subformula
                        finalmasses[branchcount] = newmass
                        priorbranch[branchcount] = branchkey
                        if newbasecomp[iso] == 0:
                            #negative direction ended, remove that isodirection in the future branches
                            del branchisodirections[branchkey][isodirection]
                        elif newbasecomp[mk] == 0:
                            #end of a positive direction, remove that isodirection in the future branches
                            del branchisodirections[branchkey][isodirection]
                        if isodirection in opposingdirections:
                            #remove opposing direction from isodirections
                            branchopposers[branchcount].add(opposingdirections[isodirection])
                        nextbranchkeys.append(branchcount)
                        branchcount += 1
        currentbranchkeys = nextbranchkeys.copy()
    
    massesandabundances = [[], []]
    formulas = []
    for k, m in finalmasses.items():
        massesandabundances[0].append(m)
        massesandabundances[1].append(finalprobabilities[k])
        formulas.append(finalformulas[k])
    
    #sorting everything by mass
    massesandabundances = np.array(massesandabundances)
    formulas = np.array(formulas, dtype='S')
    formulas = formulas[massesandabundances[1].argsort()[::-1]].tolist()
    massesandabundances = massesandabundances[:,massesandabundances[1].argsort()[::-1]]
    return formulas, massesandabundances

#this is inaccurate at higher minimumabundances, it misses some isotopomers so its a failure, they show up at higher nominal (lower threshold) minimumabundances
def vector_dist_gen(seq, minimumabundance, return_formulas=True):
    atomiccomposition = Counter()
    for aa in seq:
        atomiccomposition += aminoacidcomposition[aa]
    atomiccomposition['H'] += 2
    atomiccomposition['O'] += 1
    natomicsample = Counter()
    for e, c in atomiccomposition.items():
        for iso in isotopesbyelement[e]:
            natomicsample[iso] = nelementalprobabilities[iso] * c

    basevector = elementvector.copy()
    for e, v in isotopesbyelement.items():
        sumcount = 0
        for iso in v:
            startcount = round(natomicsample[iso])
            #^i tried generating random peptides in order to find a split that came out to 0.5 here but I'm unable to find an example where it actually happens. Hypothetically, if it does, there would be an equal probability between two isotopes, and this wouldn't handle that well. they'll be generated in the below loop regardless I suppose.
            basevector[vectorpositions[iso]] += startcount
            sumcount += startcount
        if sumcount < atomiccomposition[e]:
            #mono isotope rounded down but nothing could fill the gap, add to mono isotope
            basevector[vectorpositions[monoisotopickeys[e]]] += 1
        elif sumcount > atomiccomposition[e]:
            #this has never happened, and probably won't
            #multiple things rounded up.. presumably because there's an even split
            #subtracting from the mono isotope would be the easiest move rather than identifying which non-mono isotope was added
            basevector[vectorpositions[monoisotopickeys[e]]] -= 1
            print(seq, 'generated an erroneous example i\'ve been trying to catch with isotopic distribution rounding')


    branchkey = 0 #this is used as a basis for accessing subheaps

    baseprob = 1
    basemass = 0
    baseoffsets = defaultdict(int)
    branchprobabilitiesbyelement = defaultdict(lambda: defaultdict(lambda: 1)) #branchkey: e: prob
    branchmassesbyelement = defaultdict(lambda: defaultdict(int)) #branchkey: e: mass
    for n, c in enumerate(basevector):
        iso = elementpositions[n]
        if c > 0:
            e = iso[0]
            prob = nelementalprobabilities[iso]**c
            mass = nelementalmasses[iso] * c
            baseprob *= prob
            basemass += mass
            branchprobabilitiesbyelement[branchkey][e] *= prob
            branchmassesbyelement[branchkey][e] += mass
            if iso in nonmonoisotopicelements:
                combfactor = math.comb(atomiccomposition[e] - baseoffsets[e], c)
                baseprob *= combfactor
                branchprobabilitiesbyelement[branchkey][e] *= combfactor
                baseoffsets[e] += c

    expansiondirections = {} #branchkey: 1 or -1, a negative direction is only given to nonmonisotopic elements that are in basecomposition generated above
    isotopesbyisodirection = {} #isodirection: iso
    isodirectionsbyelement = defaultdict(list) #e: [isodirections]
    opposingdirections = {} #isodirection: isodirection of the same element moving in the other direction, I don't let these coexist within the same subheap

    isodirection = 0
    for e, isos in nonmonoisotopicgroups.items():
        if e in atomiccomposition:
            for iso in isos:
                #positive direction
                isotopesbyisodirection[isodirection] = iso
                isodirectionsbyelement[e].append(isodirection)
                expansiondirections[isodirection] = 1
                isodirection += 1
                if basevector[vectorpositions[iso]] > 0:
                    #negative direction
                    isotopesbyisodirection[isodirection] = iso
                    isodirectionsbyelement[e].append(isodirection)
                    expansiondirections[isodirection] = -1
                    #there are two directions for this iso from basecomp
                    opposingdirections[isodirection] = isodirection - 1
                    opposingdirections[isodirection - 1] = isodirection
                    isodirection += 1

    finalprobabilities = {} #branchkey: abundance
    finalmasses = {} #branchkey: mass

    finalprobabilities[branchkey] = baseprob
    finalmasses[branchkey] = basemass

    branchcount = 1
    currentbranchkeys = [] #list of all currently unexplored branchkeys
    priorbranch = {} #branchkey: branchkey of branch that generated this branch
    branchopposers = defaultdict(set) #branchkey: set of non-compatible isodirections
    branchprobabilities = defaultdict(dict) #branchkey: isodirection: combined element prob
    branchmasses = defaultdict(dict) #branchkey: isodirection: combined element mass
    branchcompositions = {} #branchcount: branchvector
    branchisodirections = {} #branchkey: tailored version of isotopesbyisodirection
    vectorsets = set()

    branchisodirections[branchkey] = isotopesbyisodirection.copy()
    branchcompositions[branchkey] = basevector

    for isodirection, iso in isotopesbyisodirection.items():
        e = iso[0]
        acount = atomiccomposition[e]
        mk = monoisotopickeys[e]
        direction = expansiondirections[isodirection]
        newbasecomp = basevector.copy()
        isopos = vectorpositions[iso]
        monopos = vectorpositions[mk]
        newbasecomp[isopos] += direction
        newbasecomp[monopos] -= direction
        vectorsets.add(tuple(newbasecomp))
        n = 0
        newelementmass = 0
        newelementprob = 1
        for en in vectorrangesbyelement[e]:
            c = newbasecomp[en]
            if c > 0:
                loopiso = elementpositions[en]
                newelementmass += nelementalmasses[loopiso] * c
                newelementprob *= nelementalprobabilities[loopiso]**c
                if loopiso in nonmonoisotopicelements:
                    newelementprob *= math.comb(acount-n, c)
                    n += c
        newprob = baseprob / branchprobabilitiesbyelement[branchkey][e] * newelementprob
        if newprob >= minimumabundance:
            newmass = basemass - branchmassesbyelement[branchkey][e] + newelementmass
            #add everything to new branchthings via branchcount!
            branchprobabilitiesbyelement[branchcount] = branchprobabilitiesbyelement[branchkey].copy()
            branchprobabilitiesbyelement[branchcount][e] = newelementprob
            branchmassesbyelement[branchcount] = branchmassesbyelement[branchkey].copy()
            branchmassesbyelement[branchcount][e] = newelementmass
            branchcompositions[branchcount] = newbasecomp
            finalprobabilities[branchcount] = newprob
            finalmasses[branchcount] = newmass
            priorbranch[branchcount] = branchkey
            branchisodirections[branchcount] = branchisodirections[branchkey].copy()
            if newbasecomp[vectorpositions[iso]] == 0:
                #negative direction ended, remove that isodirection in the future branches
                #del branchisodirections[branchkey][isodirection]
                del branchisodirections[branchcount][isodirection]
                #branchopposers[branchcount].add(isodirection)
            if newbasecomp[vectorpositions[mk]] == 0:
                #end of a positive direction, remove all isodirections of that element in the future branches
                for shk in isodirectionsbyelement[e]:
                    try:
                        #del branchisodirections[branchkey][shk]
                        del branchisodirections[branchcount][shk]
                    except KeyError:
                        pass
                    #branchopposers[branchcount].add(shk)
            if isodirection in opposingdirections:
                #remove opposing direction from isodirections
                branchopposers[branchcount].add(opposingdirections[isodirection])
                #try:
                #    del branchisodirections[branchcount][isodirection]
                #except KeyError:
                #    #previously deleted above
                #    pass
            currentbranchkeys.append(branchcount)
            branchcount += 1

    while currentbranchkeys:
        nextbranchkeys = []
        branches, branchkeysbyvector = [], {}
        for branchkey in currentbranchkeys:
            passing = False
            prior = priorbranch[branchkey]
            #generate all potential vectors here
            #for isodirection, iso in branchisodirections[prior].items():
            for isodirection, iso in branchisodirections[branchkey].items():
                if isodirection not in branchopposers[branchkey]:
                    e = iso[0]
                    mk = monoisotopickeys[e]
                    direction = expansiondirections[isodirection]
                    newbasecomp = branchcompositions[branchkey].copy()
                    isopos = vectorpositions[iso]
                    monopos = vectorpositions[mk]
                    newbasecomp[isopos] += direction
                    newbasecomp[monopos] -= direction
                    compvector = tuple(newbasecomp)
                    if compvector in vectorsets:
                        if compvector in branchkeysbyvector:
                            #merge branchopposers
                            foundkey = branchkeysbyvector[compvector]
                            branchopposers[foundkey].update(branchopposers[branchkey])
                            #branchopposers is the easiest mechanism to use here because merging branchisodirections would be a via-negation process so symmetric-differencing and deleting would be involved and it might be annoying and tedious
                    else:
                        branches.append([branchcount, branchkey, isodirection])
                        branchkeysbyvector[compvector] = branchcount
                        branchcompositions[branchcount] = newbasecomp
                        vectorsets.add(compvector)
                        branchcount += 1
                        passing = True
        for branchkey, prior, isodirection in branches:
            iso = isotopesbyisodirection[isodirection]
            e = iso[0]
            acount = atomiccomposition[e]
            newbasecomp = branchcompositions[branchkey]
            n = 0
            newelementmass = 0
            newelementprob = 1
            for en in vectorrangesbyelement[e]:
                c = newbasecomp[en]
                if c > 0:
                    loopiso = elementpositions[en]
                    newelementmass += nelementalmasses[loopiso] * c
                    newelementprob *= nelementalprobabilities[loopiso]**c
                    if loopiso in nonmonoisotopicelements:
                        newelementprob *= math.comb(acount-n, c)
                        n += c
            newprob = finalprobabilities[prior] / branchprobabilitiesbyelement[prior][e] * newelementprob
            if newprob >= minimumabundance:
                mk = monoisotopickeys[e]
                newmass = finalmasses[prior] - branchmassesbyelement[prior][e] + newelementmass
                #modify basecomp via elementcomp
                #add everything to new branchthings via branchkey!
                branchprobabilitiesbyelement[branchkey] = branchprobabilitiesbyelement[prior].copy()
                branchprobabilitiesbyelement[branchkey][e] = newelementprob
                branchmassesbyelement[branchkey] = branchmassesbyelement[prior].copy()
                branchmassesbyelement[branchkey][e] = newelementmass
                branchopposers[branchkey] = branchopposers[prior].copy()
                finalprobabilities[branchkey] = newprob
                #finalformulas[branchkey] = subformula
                finalmasses[branchkey] = newmass
                priorbranch[branchkey] = prior
                #copy branchopposers
                branchisodirections[branchkey] = branchisodirections[prior].copy()
                if newbasecomp[vectorpositions[iso]] == 0:
                    #negative direction ended, remove that isodirection in the future branches
                    #del branchisodirections[prior][isodirection]
                    del branchisodirections[branchkey][isodirection]
                    #branchopposers[branchkey].add(isodirection)
                elif newbasecomp[vectorpositions[mk]] == 0:
                    #end of a positive direction, remove all isodirections of that element in the future branches
                    for shk in isodirectionsbyelement[e]:
                        try:
                            #del branchisodirections[prior][shk]
                            del branchisodirections[branchkey][shk]
                        except KeyError:
                            pass
                        #branchopposers[branchkey].add(isodirection)
                if isodirection in opposingdirections:
                    #remove opposing direction from isodirections
                    branchopposers[branchkey].add(opposingdirections[isodirection])
                    #try:
                    #    del branchisodirections[branchkey][isodirection]
                    #except KeyError:
                    #    pass
                nextbranchkeys.append(branchkey)
        currentbranchkeys = nextbranchkeys.copy()

    if return_formulas:
        massesandabundances = [[], []]
        formulas = []
        for k, m in finalmasses.items():
            subformula = ''
            for n, c in enumerate(branchcompositions[k]):
                if c > 0:
                    subformula += f'{elementpositions[n]}({c})'
            massesandabundances[0].append(m)
            massesandabundances[1].append(finalprobabilities[k])
            formulas.append(subformula)

        #sorting everything by mass
        massesandabundances = np.array(massesandabundances)
        formulas = np.array(formulas, dtype='S')
        formulas = formulas[massesandabundances[0].argsort()[::-1]].tolist()
        massesandabundances = massesandabundances[:,massesandabundances[0].argsort()[::-1]]
        return formulas, massesandabundances
    
    massesandabundances = [[], []]
    for k, m in finalmasses.items():
        massesandabundances[0].append(m)
        massesandabundances[1].append(finalprobabilities[k])
    return massesandabundances

def binomial_walk(seq, nisos):
    atomiccomposition = Counter()
    for aa in seq:
        atomiccomposition += aminoacidcomposition[aa]
    #no OH loss on last residue, no H lost on first residue
    atomiccomposition['H'] += 2
    atomiccomposition['O'] += 1

    finaloutput = defaultdict(list)
    mainheap = []
    vectorsets = defaultdict(set) #element: set of used vectors
    nelements = len(atomiccomposition)
    maxabundances = {} #element: highest abundance of that element
    lastprobs = {} #element: last probability of that element
    for e, acount in atomiccomposition.items():
        mk = monoisotopickeys[e]
        nvector = elementvectors[e].copy()
        nvector[nvectorpositions[e][mk]] += acount
        if len(isotopesbyelement[e]) > 2:
            baseprob = nelementalprobabilities[mk] ** acount
            preheap = []
            preheap.append([baseprob, acount * nelementalmasses[mk], e, nvector.copy()])
            greater = True
            lastprob = baseprob
            while greater:
                greater = False
                for iso in nonmonoisotopicgroups[e]:
                    newelementvector = nvector.copy()
                    newelementvector[nvectorpositions[e][mk]] -= 1
                    if newelementvector[nvectorpositions[e][mk]] > -1:
                        newelementvector[nvectorpositions[e][iso]] += 1
                        vectorsets[e].add(tuple(newelementvector))
                        pn = 0
                        newelementmass = 0
                        newelementprob = 1
                        for n, c in enumerate(newelementvector):
                            loopiso = nelementpositions[e][n]
                            newelementmass += nelementalmasses[loopiso] * c
                            newelementprob *= nelementalprobabilities[loopiso]**c
                            if loopiso in nonmonoisotopicelements:
                                newelementprob *= math.comb(acount-pn, c)
                                pn += c
                        preheap.append([newelementprob, newelementmass, e, newelementvector.copy()])
                        if newelementprob > lastprob:
                            lastprob = newelementprob
                            greater = True
            preheap = sorted(preheap)
            maxiso = preheap[-1]
            maxprob, m, e, nv = maxiso
            #finaloutput[e].append([maxprob, m, nv])
            finaloutput[e].append([maxprob, m])
            maxprob *= -1
            maxabundances[e] = maxprob
            preheap = preheap[:-1]
            for h in preheap:
                r = h[0] / maxprob
                h.insert(0, r)
                heapq.heappush(mainheap, h)
            for iso in nonmonoisotopicgroups[e]:
                v = nv.copy()
                v[nvectorpositions[e][mk]] -= 1
                if v[nvectorpositions[e][mk]] > -1:
                    v[nvectorpositions[e][iso]] += 1
                    tuplevec = tuple(v)
                    if tuplevec not in vectorsets[e]:
                        vectorsets[e].add(tuplevec)
                        pn = 0
                        newelementmass = 0
                        newelementprob = 1
                        for n, c in enumerate(v):
                            loopiso = nelementpositions[e][n]
                            newelementmass += nelementalmasses[loopiso] * c
                            newelementprob *= nelementalprobabilities[loopiso]**c
                            if loopiso in nonmonoisotopicelements:
                                newelementprob *= math.comb(acount-pn, c)
                                pn += c
                        heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, v.copy()])
                        #heapq.heappush(mainheap, [-newelementprob, newelementmass, e, v.copy()])
        else:
            preheap = []
            baseprob = nelementalprobabilities[mk] ** acount
            lastprobs[e] = baseprob
            preheap.append([baseprob, acount * nelementalmasses[mk], e, nvector.copy()])
            greater = True
            lastprob = baseprob
            while greater:
                greater = False
                iso = nonmonoisotopicgroups[e][0]
                nvector[nvectorpositions[e][mk]] -= 1
                if nvector[nvectorpositions[e][mk]] > -1:
                    nvector[nvectorpositions[e][iso]] += 1
                    vectorsets[e].add(tuple(nvector))
                    pn = 0
                    newelementmass = 0
                    newelementprob = 1
                    for n, c in enumerate(nvector):
                        loopiso = nelementpositions[e][n]
                        newelementmass += nelementalmasses[loopiso] * c
                        newelementprob *= nelementalprobabilities[loopiso]**c
                        if loopiso in nonmonoisotopicelements:
                            newelementprob *= math.comb(acount-pn, c)
                            pn += c
                    preheap.append([newelementprob, newelementmass, e, nvector.copy()])
                    if newelementprob > lastprob:
                        lastprob = newelementprob
                        greater = True
            preheap = sorted(preheap)
            maxiso = preheap[-1]
            maxprob, m, e, nv = maxiso
            #finaloutput[e].append([maxprob, m, nv])
            finaloutput[e].append([maxprob, m])
            #etracker[e] += 1
            maxprob *= -1
            maxabundances[e] = maxprob
            preheap = preheap[:-1]
            for h in preheap:
                r = h[0] / maxprob
                h.insert(0, r)
                heapq.heappush(mainheap, h)
            v = nv.copy()
            v[nvectorpositions[e][mk]] -= 1
            if v[nvectorpositions[e][mk]] > -1:
                v[nvectorpositions[e][iso]] += 1
                tuplevec = tuple(v)
                if tuplevec not in vectorsets[e]:
                    vectorsets[e].add(tuplevec)
                    pn = 0
                    newelementmass = 0
                    newelementprob = 1
                    for n, c in enumerate(v):
                        loopiso = nelementpositions[e][n]
                        newelementmass += nelementalmasses[loopiso] * c
                        newelementprob *= nelementalprobabilities[loopiso]**c
                        if loopiso in nonmonoisotopicelements:
                            newelementprob *= math.comb(acount-pn, c)
                            pn += c
                    heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, v.copy()])
                    #heapq.heappush(mainheap, [-newelementprob, newelementmass, e, v.copy()])

    r, p, m, e, v = heapq.heappop(mainheap)
    #finaloutput[e].append([p, m, v])
    finaloutput[e].append([p, m])
    isotracker = 2 #starts here, not above b/c that's just "1"
    while True:
        acount = atomiccomposition[e]
        mk = monoisotopickeys[e]
        if len(isotopesbyelement[e]) > 2:
            for iso in nonmonoisotopicgroups[e]:
                newelementvector = v.copy()
                newelementvector[nvectorpositions[e][mk]] -= 1
                if newelementvector[nvectorpositions[e][mk]] > 0:
                    newelementvector[nvectorpositions[e][iso]] += 1
                    tuplevec = tuple(newelementvector)
                    if tuplevec not in vectorsets[e]:
                        vectorsets[e].add(tuplevec)
                        pn = 0
                        newelementmass = 0
                        newelementprob = 1
                        for n, c in enumerate(newelementvector):
                            loopiso = nelementpositions[e][n]
                            newelementmass += nelementalmasses[loopiso] * c
                            newelementprob *= nelementalprobabilities[loopiso]**c
                            if loopiso in nonmonoisotopicelements:
                                newelementprob *= math.comb(acount-pn, c)
                                pn += c
                        heapq.heappush(mainheap, [newelementprob / maxabundances[e], newelementprob, newelementmass, e, newelementvector.copy()])
        else:
            iso = nonmonoisotopicgroups[e][0]
            nvector = v.copy()
            nvector[nvectorpositions[e][mk]] -= 1
            if nvector[nvectorpositions[e][mk]] > 0:
                nvector[nvectorpositions[e][iso]] += 1
                tuplevec = tuple(nvector)
                if tuplevec not in vectorsets[e]:
                    vectorsets[e].add(tuplevec)
                    pn = 0
                    newelementmass = 0
                    newelementprob = 1
                    for n, c in enumerate(nvector):
                        loopiso = nelementpositions[e][n]
                        newelementmass += nelementalmasses[loopiso] * c
                        newelementprob *= nelementalprobabilities[loopiso]**c
                        if loopiso in nonmonoisotopicelements:
                            newelementprob *= math.comb(acount-pn, c)
                            pn += c
                    heapq.heappush(mainheap, [newelementprob / maxabundances[e], newelementprob, newelementmass, e, nvector.copy()])
        r, p, m, e, v = heapq.heappop(mainheap)
        finaloutput[e].append([p, m])
        if p == 0:
            break
        isotracker += 1
        if isotracker >= nisos:
            break

    massesandabundances = [[], []]
    for prod in itertools.product(*finaloutput.values()):
        mass = 0
        prob = 1
        for p, m in prod:
            prob *= p
            mass += m
        massesandabundances[0].append(mass)
        massesandabundances[1].append(prob)
    massesandabundances = np.array(massesandabundances)
    massesandabundances = massesandabundances[:,massesandabundances[1].argsort()[::-1]][:,:nisos]
    return massesandabundances

def binomial_walk2(seq, dividingthreshold):
    atomiccomposition = Counter()
    for aa in seq:
        atomiccomposition += aminoacidcomposition[aa]
    #no OH loss on last residue, no H lost on first residue
    atomiccomposition['H'] += 2
    atomiccomposition['O'] += 1

    finaloutput = defaultdict(list)
    mainheap = []
    vectorsets = defaultdict(set) #element: set of used vectors
    nelements = len(atomiccomposition)
    maxabundances = {} #element: highest abundance of that element
    lastprobs = {} #element: last probability of that element
    for e, acount in atomiccomposition.items():
        mk = monoisotopickeys[e]
        nvector = elementvectors[e].copy()
        nvector[nvectorpositions[e][mk]] += acount
        if len(isotopesbyelement[e]) > 2:
            baseprob = nelementalprobabilities[mk] ** acount
            preheap = []
            preheap.append([baseprob, acount * nelementalmasses[mk], e, nvector.copy()])
            greater = True
            lastprob = baseprob
            while greater:
                greater = False
                for iso in nonmonoisotopicgroups[e]:
                    newelementvector = nvector.copy()
                    newelementvector[nvectorpositions[e][mk]] -= 1
                    if newelementvector[nvectorpositions[e][mk]] > -1:
                        newelementvector[nvectorpositions[e][iso]] += 1
                        vectorsets[e].add(tuple(newelementvector))
                        pn = 0
                        newelementmass = 0
                        newelementprob = 1
                        for n, c in enumerate(newelementvector):
                            loopiso = nelementpositions[e][n]
                            newelementmass += nelementalmasses[loopiso] * c
                            newelementprob *= nelementalprobabilities[loopiso]**c
                            if loopiso in nonmonoisotopicelements:
                                newelementprob *= math.comb(acount-pn, c)
                                pn += c
                        preheap.append([newelementprob, newelementmass, e, newelementvector.copy()])
                        if newelementprob > lastprob:
                            lastprob = newelementprob
                            greater = True
            preheap = sorted(preheap)
            maxiso = preheap[-1]
            maxprob, m, e, nv = maxiso
            #finaloutput[e].append([maxprob, m, nv])
            finaloutput[e].append([maxprob, m, e, nv])
            maxprob *= -1
            maxabundances[e] = maxprob
            preheap = preheap[:-1]
            for h in preheap:
                r = h[0] / maxprob
                h.insert(0, r)
                heapq.heappush(mainheap, h)
            for iso in nonmonoisotopicgroups[e]:
                v = nv.copy()
                v[nvectorpositions[e][mk]] -= 1
                if v[nvectorpositions[e][mk]] > -1:
                    v[nvectorpositions[e][iso]] += 1
                    tuplevec = tuple(v)
                    if tuplevec not in vectorsets[e]:
                        vectorsets[e].add(tuplevec)
                        pn = 0
                        newelementmass = 0
                        newelementprob = 1
                        for n, c in enumerate(v):
                            loopiso = nelementpositions[e][n]
                            newelementmass += nelementalmasses[loopiso] * c
                            newelementprob *= nelementalprobabilities[loopiso]**c
                            if loopiso in nonmonoisotopicelements:
                                newelementprob *= math.comb(acount-pn, c)
                                pn += c
                        heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, v.copy()])
                        #heapq.heappush(mainheap, [-newelementprob, newelementmass, e, v.copy()])
        else:
            preheap = []
            baseprob = nelementalprobabilities[mk] ** acount
            lastprobs[e] = baseprob
            preheap.append([baseprob, acount * nelementalmasses[mk], e, nvector.copy()])
            greater = True
            lastprob = baseprob
            while greater:
                greater = False
                iso = nonmonoisotopicgroups[e][0]
                nvector[nvectorpositions[e][mk]] -= 1
                if nvector[nvectorpositions[e][mk]] > -1:
                    nvector[nvectorpositions[e][iso]] += 1
                    vectorsets[e].add(tuple(nvector))
                    pn = 0
                    newelementmass = 0
                    newelementprob = 1
                    for n, c in enumerate(nvector):
                        loopiso = nelementpositions[e][n]
                        newelementmass += nelementalmasses[loopiso] * c
                        newelementprob *= nelementalprobabilities[loopiso]**c
                        if loopiso in nonmonoisotopicelements:
                            newelementprob *= math.comb(acount-pn, c)
                            pn += c
                    preheap.append([newelementprob, newelementmass, e, nvector.copy()])
                    if newelementprob > lastprob:
                        lastprob = newelementprob
                        greater = True
            preheap = sorted(preheap)
            maxiso = preheap[-1]
            maxprob, m, e, nv = maxiso
            #finaloutput[e].append([maxprob, m, nv])
            finaloutput[e].append([maxprob, m, e, nv])
            #etracker[e] += 1
            maxprob *= -1
            maxabundances[e] = maxprob
            preheap = preheap[:-1]
            for h in preheap:
                r = h[0] / maxprob
                h.insert(0, r)
                heapq.heappush(mainheap, h)
            v = nv.copy()
            v[nvectorpositions[e][mk]] -= 1
            if v[nvectorpositions[e][mk]] > -1:
                v[nvectorpositions[e][iso]] += 1
                tuplevec = tuple(v)
                if tuplevec not in vectorsets[e]:
                    vectorsets[e].add(tuplevec)
                    pn = 0
                    newelementmass = 0
                    newelementprob = 1
                    for n, c in enumerate(v):
                        loopiso = nelementpositions[e][n]
                        newelementmass += nelementalmasses[loopiso] * c
                        newelementprob *= nelementalprobabilities[loopiso]**c
                        if loopiso in nonmonoisotopicelements:
                            newelementprob *= math.comb(acount-pn, c)
                            pn += c
                    heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, v.copy()])
                    #heapq.heappush(mainheap, [-newelementprob, newelementmass, e, v.copy()])
    
    cutoff = {e:-i * dividingthreshold for e, i in maxabundances.items()}
    
    r, p, m, e, v = heapq.heappop(mainheap)
    #finaloutput[e].append([p, m, v])
    finaloutput[e].append([p, m, e, v])
    #isotracker = 2 #starts here, not above b/c that's just "1"
    #while True:
    while p > cutoff[e]:
        acount = atomiccomposition[e]
        mk = monoisotopickeys[e]
        if len(isotopesbyelement[e]) > 2:
            for iso in nonmonoisotopicgroups[e]:
                newelementvector = v.copy()
                newelementvector[nvectorpositions[e][mk]] -= 1
                if newelementvector[nvectorpositions[e][mk]] > 0:
                    newelementvector[nvectorpositions[e][iso]] += 1
                    tuplevec = tuple(newelementvector)
                    if tuplevec not in vectorsets[e]:
                        vectorsets[e].add(tuplevec)
                        pn = 0
                        newelementmass = 0
                        newelementprob = 1
                        for n, c in enumerate(newelementvector):
                            loopiso = nelementpositions[e][n]
                            newelementmass += nelementalmasses[loopiso] * c
                            newelementprob *= nelementalprobabilities[loopiso]**c
                            if loopiso in nonmonoisotopicelements:
                                newelementprob *= math.comb(acount-pn, c)
                                pn += c
                        heapq.heappush(mainheap, [newelementprob / maxabundances[e], newelementprob, newelementmass, e, newelementvector.copy()])
        else:
            iso = nonmonoisotopicgroups[e][0]
            nvector = v.copy()
            nvector[nvectorpositions[e][mk]] -= 1
            if nvector[nvectorpositions[e][mk]] > 0:
                nvector[nvectorpositions[e][iso]] += 1
                tuplevec = tuple(nvector)
                if tuplevec not in vectorsets[e]:
                    vectorsets[e].add(tuplevec)
                    pn = 0
                    newelementmass = 0
                    newelementprob = 1
                    for n, c in enumerate(nvector):
                        loopiso = nelementpositions[e][n]
                        newelementmass += nelementalmasses[loopiso] * c
                        newelementprob *= nelementalprobabilities[loopiso]**c
                        if loopiso in nonmonoisotopicelements:
                            newelementprob *= math.comb(acount-pn, c)
                            pn += c
                    heapq.heappush(mainheap, [newelementprob / maxabundances[e], newelementprob, newelementmass, e, nvector.copy()])
        r, p, m, e, v = heapq.heappop(mainheap)
        finaloutput[e].append([p, m, e, v])
        #if p == 0:
        #    break
        #isotracker += 1
        #if isotracker >= nisos:
        #    break

    massesandabundances = [[], []]
    subformulas = []
    for prod in itertools.product(*finaloutput.values()):
        mass = 0
        prob = 1
        formula = ''
        for p, m, e, v in prod:
            prob *= p
            mass += m
            for n, c in enumerate(v):
                if c > 0:
                    formula += f'{nelementpositions[e][n]}({c})'
        massesandabundances[0].append(mass)
        massesandabundances[1].append(prob)
        subformulas.append(formula)
    massesandabundances = np.array(massesandabundances)
    subformulas = np.array(subformulas, dtype='S')
    subformulas = subformulas[massesandabundances[1].argsort()[::-1]].tolist()
    massesandabundances = massesandabundances[:,massesandabundances[1].argsort()[::-1]]
    return subformulas, massesandabundances

def elemental_binomial_walk(atomiccomposition, dividingthreshold):
    elementalorganizer = defaultdict(list)
    mainheap = []
    vectorsets = defaultdict(set) #element: set of used vectors
    nelements = len(atomiccomposition)
    lastprobs = {} #element: last probability of that element
    for e, acount in atomiccomposition.items():
        mk = monoisotopickeys[e]
        nvector = elementvectors[e].copy()
        nvector[nvectorpositions[e][mk]] += acount
        if len(isotopesbyelement[e]) > 2:
            baseprob = nelementalprobabilities[mk] ** acount
            preheap = []
            preheap.append([baseprob, acount * nelementalmasses[mk], e, nvector.copy()])
            greater = True
            lastprob = baseprob
            while greater:
                greater = False
                for iso in nonmonoisotopicgroups[e]:
                    newelementvector = nvector.copy()
                    newelementvector[nvectorpositions[e][mk]] -= 1
                    if newelementvector[nvectorpositions[e][mk]] > -1:
                        newelementvector[nvectorpositions[e][iso]] += 1
                        vectorsets[e].add(tuple(newelementvector))
                        pn = 0
                        newelementmass = 0
                        newelementprob = 1
                        for n, c in enumerate(newelementvector):
                            loopiso = nelementpositions[e][n]
                            newelementmass += nelementalmasses[loopiso] * c
                            newelementprob *= nelementalprobabilities[loopiso]**c
                            if loopiso in nonmonoisotopicelements:
                                newelementprob *= math.comb(acount-pn, c)
                                pn += c
                        preheap.append([newelementprob, newelementmass, e, newelementvector.copy()])
                        if newelementprob > lastprob:
                            lastprob = newelementprob
                            greater = True
            preheap = sorted(preheap)
            maxiso = preheap[-1]
            maxprob, m, e, nv = maxiso
            elementalorganizer[e].append([-1, maxprob, m, e, nv])
            maxprob *= -1
            preheap = preheap[:-1]
            for h in preheap:
                r = h[0] / maxprob
                h.insert(0, r)
                heapq.heappush(mainheap, h)
            for iso in nonmonoisotopicgroups[e]:
                v = nv.copy()
                v[nvectorpositions[e][mk]] -= 1
                if v[nvectorpositions[e][mk]] > -1:
                    v[nvectorpositions[e][iso]] += 1
                    tuplevec = tuple(v)
                    if tuplevec not in vectorsets[e]:
                        vectorsets[e].add(tuplevec)
                        pn = 0
                        newelementmass = 0
                        newelementprob = 1
                        for n, c in enumerate(v):
                            loopiso = nelementpositions[e][n]
                            newelementmass += nelementalmasses[loopiso] * c
                            newelementprob *= nelementalprobabilities[loopiso]**c
                            if loopiso in nonmonoisotopicelements:
                                newelementprob *= math.comb(acount-pn, c)
                                pn += c
                        heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, v.copy()])
        else:
            preheap = []
            baseprob = nelementalprobabilities[mk] ** acount
            lastprobs[e] = baseprob
            preheap.append([baseprob, acount * nelementalmasses[mk], e, nvector.copy()])
            greater = True
            lastprob = baseprob
            while greater:
                greater = False
                iso = nonmonoisotopicgroups[e][0]
                nvector[nvectorpositions[e][mk]] -= 1
                if nvector[nvectorpositions[e][mk]] > -1:
                    nvector[nvectorpositions[e][iso]] += 1
                    vectorsets[e].add(tuple(nvector))
                    pn = 0
                    newelementmass = 0
                    newelementprob = 1
                    for n, c in enumerate(nvector):
                        loopiso = nelementpositions[e][n]
                        newelementmass += nelementalmasses[loopiso] * c
                        newelementprob *= nelementalprobabilities[loopiso]**c
                        if loopiso in nonmonoisotopicelements:
                            newelementprob *= math.comb(acount-pn, c)
                            pn += c
                    preheap.append([newelementprob, newelementmass, e, nvector.copy()])
                    if newelementprob > lastprob:
                        lastprob = newelementprob
                        greater = True
            preheap = sorted(preheap)
            maxiso = preheap[-1]
            maxprob, m, e, nv = maxiso
            elementalorganizer[e].append([-1, maxprob, m, e, nv])
            maxprob *= -1
            preheap = preheap[:-1]
            for h in preheap:
                r = h[0] / maxprob
                h.insert(0, r)
                heapq.heappush(mainheap, h)
            v = nv.copy()
            v[nvectorpositions[e][mk]] -= 1
            if v[nvectorpositions[e][mk]] > -1:
                v[nvectorpositions[e][iso]] += 1
                tuplevec = tuple(v)
                if tuplevec not in vectorsets[e]:
                    vectorsets[e].add(tuplevec)
                    pn = 0
                    newelementmass = 0
                    newelementprob = 1
                    for n, c in enumerate(v):
                        loopiso = nelementpositions[e][n]
                        newelementmass += nelementalmasses[loopiso] * c
                        newelementprob *= nelementalprobabilities[loopiso]**c
                        if loopiso in nonmonoisotopicelements:
                            newelementprob *= math.comb(acount-pn, c)
                            pn += c
                    heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, v.copy()])
        
        cutoff = -maxprob * dividingthreshold

        r, p, m, e, v = heapq.heappop(mainheap)
        elementalorganizer[e].append([r, p, m, e, v])
        while p > cutoff:
            if len(isotopesbyelement[e]) > 2:
                for iso in nonmonoisotopicgroups[e]:
                    newelementvector = v.copy()
                    newelementvector[nvectorpositions[e][mk]] -= 1
                    if newelementvector[nvectorpositions[e][mk]] > 0:
                        newelementvector[nvectorpositions[e][iso]] += 1
                        tuplevec = tuple(newelementvector)
                        if tuplevec not in vectorsets[e]:
                            vectorsets[e].add(tuplevec)
                            pn = 0
                            newelementmass = 0
                            newelementprob = 1
                            for n, c in enumerate(newelementvector):
                                loopiso = nelementpositions[e][n]
                                newelementmass += nelementalmasses[loopiso] * c
                                newelementprob *= nelementalprobabilities[loopiso]**c
                                if loopiso in nonmonoisotopicelements:
                                    newelementprob *= math.comb(acount-pn, c)
                                    pn += c
                            heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, newelementvector.copy()])
            else:
                iso = nonmonoisotopicgroups[e][0]
                nvector = v.copy()
                nvector[nvectorpositions[e][mk]] -= 1
                if nvector[nvectorpositions[e][mk]] > 0:
                    nvector[nvectorpositions[e][iso]] += 1
                    tuplevec = tuple(nvector)
                    if tuplevec not in vectorsets[e]:
                        vectorsets[e].add(tuplevec)
                        pn = 0
                        newelementmass = 0
                        newelementprob = 1
                        for n, c in enumerate(nvector):
                            loopiso = nelementpositions[e][n]
                            newelementmass += nelementalmasses[loopiso] * c
                            newelementprob *= nelementalprobabilities[loopiso]**c
                            if loopiso in nonmonoisotopicelements:
                                newelementprob *= math.comb(acount-pn, c)
                                pn += c
                        heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, nvector.copy()])
            r, p, m, e, v = heapq.heappop(mainheap)
            elementalorganizer[e].append([r, p, m, e, v])
    return elementalorganizer

def descending_partial_products(elementalorganizer, dividingthreshold):
    for k in elementalorganizer:
        heapq.heapify(elementalorganizer[k])

    mainpool = defaultdict(list) #things already popped from elementalorganizer
    for k in elementalorganizer:
        mainpool[k].append(heapq.heappop(elementalorganizer[k]))

    formula = ''
    maxprob = 1
    mainmass = 0
    finalabundances = {} #subformula: prob
    for b in sorted(mainpool):
        for r, p, m, e, v in mainpool[b]:
            for n, c in enumerate(v):
                if c > 0:
                    formula += f'{nelementpositions[e][n]}({c})'
            maxprob *= p
            mainmass += m

    finalabundances[formula] = [mainmass, maxprob]

    cutoff = maxprob * dividingthreshold
    mainheap = list(itertools.chain(*elementalorganizer.values()))
    heapq.heapify(mainheap)

    multinomialpath = [] #sublists not in mainpool
    probabilityranking = [] #representative lists of ratio probability to sort multinomialpath
    while mainheap:
        r, p, m, e, v = heapq.heappop(mainheap)
        baseiter = {k: v for k, v in mainpool.items() if k != e}
        baseiter[e] = [[r, p, m, e, v]]
        
        formula = ''
        prob = 1
        mass = 0
        for b in sorted(baseiter):
            for sr, sp, sm, se, sv in baseiter[b]:
                for n, c in enumerate(sv):
                    if c > 0:
                        formula += f'{nelementpositions[se][n]}({c})'
                prob *= sp
                mass += sm
        
        if prob < cutoff:
            break
        finalabundances[formula] = [mass, prob]
        
        ind = bisect.bisect(probabilityranking, r)
        probabilityranking.insert(ind, r)
        multinomialpath.insert(ind, [r, p, m, e, v])
        
        checkedcombos = set()
        for path in multinomialpath.copy():
            multielement = False
            match path[1]:
                case list():
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
                                    sef += f'{nelementpositions[se][n]}({c})'
                            seformulas.append(sef)
                            multipath.append([sr, sp, sm, se, sv])
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
                    newiter = {k: v for k, v in baseiter.items() if k not in sepool}
                    newiter[e] = [[r, p, m, e, v]]
                    for ir, ip, im, ie, iv in multipath:
                        newiter[ie] = [[ir, ip, im, ie, iv]]
                    for b in sorted(newiter):
                        for ir, ip, im, ie, iv in newiter[b]:
                            for n, c in enumerate(iv):
                                if c > 0:
                                    seformula += f'{nelementpositions[ie][n]}({c})'
                            newprob *= ip
                            newmass += im
                else:
                    newiter = {k: v for k, v in baseiter.items() if k != se}
                    newiter[se] = [[sr, sp, sm, se, sv]]
                    seformula = ''
                    newprob = 1
                    newmass = 0
                    for b in sorted(newiter):
                        for ir, ip, im, ie, iv in newiter[b]:
                            for n, c in enumerate(iv):
                                if c > 0:
                                    seformula += f'{nelementpositions[ie][n]}({c})'
                            newprob *= ip
                            newmass += im
                if newprob >= cutoff:
                    finalabundances[seformula] = [newmass, newprob]
                    if multielement:
                        ind = bisect.bisect(probabilityranking, newratio)
                        probabilityranking.insert(ind, newratio)
                        multinomialpath.insert(ind, [newratio, *multipath])
                    else:
                        ind = bisect.bisect(probabilityranking, newratio)
                        probabilityranking.insert(ind, newratio)
                        multinomialpath.insert(ind, [newratio, [sr, sp, sm, se, sv], [r, p, m, e, v]])
            else:
                break

    subformulas, massesandabundances = list(zip(*finalabundances.items()))
    subformulas = np.array(subformulas, dtype='S')
    massesandabundances = np.array(massesandabundances).transpose()
    subformulas = subformulas[massesandabundances[1].argsort()[::-1]].tolist()
    massesandabundances = massesandabundances[:,massesandabundances[1].argsort()[::-1]]
    return subformulas, massesandabundances

def stringprob(formula):
    baseoffsets = defaultdict(int)
    fmass = 0
    fprob = 1
    for ep in formula.split(')')[:-1]: 
        iso, c = ep.split('(')
        c = int(c)
        e = iso[0]
        prob = nelementalprobabilities[iso]**c
        mass = nelementalmasses[iso] * c
        fprob *= prob
        fmass += mass
        if iso in nonmonoisotopicelements:
            combfactor = special.comb(atomiccomposition[e] - baseoffsets[e], c, exact=True)
            fprob *= combfactor
            baseoffsets[e] += c
    return fmass, fprob

dividingthreshold = 0.05
minimumabundance = 0.01
samplesize = 100
#seq = 'GTSWGLPASKTITTMIDGPQDLRVVAVTPTTLELGWLRPQAEVDR'
#seq = 'IGMGINRYNAQLFRPITNFSCAARCMMHVRCGDLIIDSHQILWFCTTSVNYDDQEPVPIWALQLICCRCYIHTEAWHAMCSEVKHTDTLAAHCVECWL'
#
#atomiccomposition = Counter()
#for aa in seq:
#    atomiccomposition += aminoacidcomposition[aa]
##no OH loss on last residue, no H lost on first residue
#atomiccomposition['H'] += 2
#atomiccomposition['O'] += 1
#formulastring = ''.join((''.join((k, str(v))) for k, v in atomiccomposition.items()))
#monoisotopicmass = sum(majorisotopemasses[k]*v for k, v in atomiccomposition.items())
#
#abundances, subformulas = distribution_generation(atomiccomposition, samplesize)
##nsubformulas, nabundances = next_dist_gen(atomiccomposition, minimumabundance) #comes out wrong
#vsubformulas, vabundances = vector_dist_gen(atomiccomposition, minimumabundance)
#alen = len(vabundances[0])
#babundances = binomial_walk(seq, alen)

#for vm, vp, bm, bp, m, p in zip(*vabundances, *babundances, *abundances.tolist()):
#    print(round(vm, 5), round(vp, 5))
#    print(round(bm, 5), round(bp, 5))
#    print(round(m, 5), round(p, 5))
#    if not np.isclose(vm, m):
#        print('no mass')
#    if not np.isclose(vp, p):
#        print('no prob')
#    print('~~~')
#
#for bm, bp, m, p in zip(*babundances, *abundances.tolist()):
#    print(round(bm, 5), round(bp, 5))
#    print(round(m, 5), round(p, 5))
#    if not np.isclose(bm, m):
#        print('no mass')
#    if not np.isclose(bp, p):
#        print('no prob')
#    print('~~~')
#
#for bm, bp, vm, vp in zip(*babundances, *vabundances.tolist()):
#    print(round(bm, 5), round(bp, 5))
#    print(round(vm, 5), round(vp, 5))
#    if not np.isclose(bm, vm):
#        print('no mass')
#    if not np.isclose(bp, vp):
#        print('no prob')
#    print('~~~')

#np.array(subformulas)[abundances[1].argsort()] #last has the highest intensity
#plt.bar(abundances[0], abundances[1], width=0.01)
#plt.show()

def bs_checker(elementalorganizer):
    output = []
    for prod in itertools.product(*elementalorganizer.values()):
        #formulas = []
        prob = 1
        mass = 0
        for r, p, m, e, v in prod:
            #formula += ''.join(([str(e) + f'{n}' + f'({i})' for n, i in enumerate(v) if i > 0]))
            #formula = ''
            #for n, c in enumerate(v):
            #    if c > 0:
            #        formula += f'{nelementpositions[e][n]}({c})'
            #formulas.append(formula)
            prob *= p
            mass += m
        #formula = ''.join((sorted(formulas)))
        output.append((mass, prob))
    return output



bt = time()
n = 2
screwups1, screwups2, screwups3 = [], [], []
times, lengths = [], []
seqlist = []
for _ in range(n):
    seq = ''.join((random.choices(list(aminoacidcomposition.keys()), k=np.random.randint(6,50))))
    atomiccomposition = Counter()
    for aa in seq:
        atomiccomposition += aminoacidcomposition[aa]
    atomiccomposition['H'] += 2
    atomiccomposition['O'] += 1
    st = []
    #nt = time()
    #abundances, subformulas = distribution_generation(atomiccomposition, samplesize)
    #end = time() - nt
    #st.append(end)
    nt = time()
    vsubformulas, vabundances = vector_dist_gen(seq, minimumabundance)
    end = time() - nt
    st.append(end)
    alen = len(vabundances[0])
    nt = time()
    #nformulas, nabundances = next_dist_gen(seq, minimumabundance)
    babundances = binomial_walk(seq, alen)
    end = time() - nt
    st.append(end)
    lengths.append(alen)
    seqlist.append(seq) 
    dividingthreshold = babundances[1].min() / babundances[1].max()
    nt = time()
    elementalorganizer = elemental_binomial_walk(atomiccomposition, dividingthreshold)
    formulas, abundances = descending_partial_products(elementalorganizer, dividingthreshold)
    end = time() - nt
    st.append(end)
    nt = time()
    #nformulas, nabundances = next_dist_gen(seq, minimumabundance)
    bsubformulas2, babundances2 = binomial_walk2(seq, dividingthreshold)
    end = time() - nt
    st.append(end)
    times.append(st)
    #if not np.isclose(nabundances, vabundances).all():
    #    screwups.append(seq)
    if not np.isclose(vabundances, babundances[:,:vabundances.shape[1]]).all():
        screwups1.append(seq)
    if not np.isclose(abundances, babundances[:,:abundances.shape[1]]).all():
        screwups2.append(seq)
    if not np.isclose(babundances, babundances2[:,:babundances.shape[1]]).all():
        screwups3.append(seq)
print(time() - bt)

lengths = np.array(lengths)
seqlist = np.array(seqlist)
times = np.array(times)
ntimes = times.transpose().tolist()

#not completely accurate, some times get convoluted from background processes
plt.plot(ntimes[2], lengths, '.', color='white')
plt.plot(ntimes[3], lengths, '.', color='crimson', alpha=0.5)
plt.show()

plt.plot(times[:,2], times[:,3], '.')
plt.show()
print(len(screwups1), len(screwups2), len(screwups3), 'screwups')

rates = times / lengths[:,None]

print('times', times.sum(axis=0))
print('rates', rates.sum(axis=0))

#question:
#does the 2nd highest isotopomer always show up in an initial iteration?
#
#if so, i also want to know if these are always the 3+ highest or whatever, as i'm pretty sure they would be close in a lot of cases
#
#+1 their final rank in a global counter
#keep a list of [expectation, true] for each rank pairing
#
#if they don't go to the highest individual subisotopomer, do they follow the pattern of the highest proton locations?
#
#~
#it might also be good to calculate all elementstring/heaps first
#this would work for ms1 dists, not ms2 dists?

#i want to re-do distribution_management to produce only sum distributions - DONE
#and i want to use subformulas to determine which proton locations to combine under - DONE
#subisotopomeric distances are coming out negative, what's wrong here? - DONE
#also, newinclimit and steplimit seem unacceptable now - currently dealing with it via hard coding it instead, 0.5 and 0.1 work fine
#somethings going wrong in distribution management - DONE

#i'm going to generate elementstrings before calling descending-partial-products
#then i can fastcopy each reference
#hopefully subisovals stops being negative
#ALL DONE
