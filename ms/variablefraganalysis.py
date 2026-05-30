from collections import defaultdict, Counter
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from itertools import zip_longest
from time import time
import random
import heapq
import math
import os
import sys
sys.setrecursionlimit(10000)

#this file was used to compare the generation of single max abundance theoretical ms2 fragments to previously used computations to make sure the output matches

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

aminoacidcomposition = { #amino acid: element: count
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


def vector_gen(seq, minimumabundance):
    atomiccomposition = Counter()
    for aa in seq:
        atomiccomposition += aminoacidcomposition[aa]
    atomiccomposition['H'] += 2
    atomiccomposition['O'] += 1

    natomicsample = Counter()
    for e, c in atomiccomposition.items():
        for iso in isotopesbyelement[e]:
            natomicsample[iso] = elementalprobabilities[iso] * c

    basevector = elementvector.copy()
    for e, v in isotopesbyelement.items():
        sumcount = 0
        for iso in v:
            startcount = round(natomicsample[iso])
            #^i tried generating random peptides in order to find a split that came out to 0.5 here but I'm unable to find an example where it actually happens. Hypothetically, if it does, there would be an equal probability between two isotopes, and this wouldn't handle that well. they'll be generated in the below loop regardless I suppose.
            #basecomposition[iso] = startcount
            basevector[vectorpositions[iso]] += startcount
            #if startcount > 0:
            sumcount += startcount
                #basecomposition[iso] = startcount
        if sumcount < atomiccomposition[e]:
            #mono isotope rounded down but nothing could fill the gap, add to mono isotope
            #basecomposition[monoisotopickeys[e]] += 1
            #basecomposition[monoisotopickeys[e]] += 1
            basevector[vectorpositions[monoisotopickeys[e]]] += 1
        elif sumcount > atomiccomposition[e]:
            #this has never happened, and probably won't
            #multiple things rounded up.. presumably because there's an even split
            #subtracting from the mono isotope would be the easiest move rather than identifying which non-mono isotope was added
            #basecomposition[monoisotopickeys[e]] -= 1
            #basecomposition[monoisotopickeys[e]] -= 1
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
            prob = elementalprobabilities[iso]**c
            mass = elementalmasses[iso] * c
            baseprob *= prob
            basemass += mass
            branchprobabilitiesbyelement[branchkey][e] *= prob
            branchmassesbyelement[branchkey][e] += mass
            #branchcompositionbyelement[branchkey][e][iso] = c
            if iso in nonmonoisotopicelements:
                #combfactor = special.comb(atomiccomposition[e] - baseoffsets[e], c, exact=True)
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
                #if basecomposition[iso] > 0:
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
                newelementmass += elementalmasses[loopiso] * c
                newelementprob *= elementalprobabilities[loopiso]**c
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
            #finalformulas[branchcount] = subformula
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
        #else:
            #if it it doesn't work with the most abundant isotopomer then it won't work with anything, remove this iso from all possibilities moving forward
            #del branchisodirections[branchkey][isodirection]
            #branchopposers[branchkey].add(isodirection) #prevent it in future loops
            #^THIS IS NOT TRUE, some things passed over here might still end up being relevant

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
            #if passing:
            #    branchisodirections[branchkey] = branchisodirections[prior].copy()
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
                    newelementmass += elementalmasses[loopiso] * c
                    newelementprob *= elementalprobabilities[loopiso]**c
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

    massesandabundances = [[], []]
    branchformulas = {}
    formulas = []
    for k, m in finalmasses.items():
        subformula = ''
        for n, c in enumerate(branchcompositions[k]):
            if c > 0:
                subformula += f'{elementpositions[n]}({c})'
        massesandabundances[0].append(m)
        massesandabundances[1].append(finalprobabilities[k])
        formulas.append(subformula)
        branchformulas[subformula] = k

    massesandabundances = np.array(massesandabundances)
    formulas = np.array(formulas, dtype='S')
    formulas = formulas[massesandabundances[1].argsort()[::-1]].tolist()
    massesandabundances = massesandabundances[:,massesandabundances[1].argsort()[::-1]].tolist()
    return formulas, massesandabundances

nfragmentcompositions = {}
nfragmentcompositions['a'] = Counter({'C': -1, 'O': -1})
nfragmentcompositions['b'] = Counter()
nfragmentcompositions['c'] = Counter({'N': 1, 'H': 3})

cfragmentcompositions = {}
cfragmentcompositions['x'] = Counter({'C': 1, 'O': 2})
cfragmentcompositions['y'] = Counter({'H': 2, 'O': 1})
cfragmentcompositions['z'] = Counter({'N': -1, 'H': -1, 'O': 1})
#cfragmentcompositions['z•'] = Counter({'N': -1, 'O': 1})

def fragmentation_compositions(seq):
    fragments = {}
    fragcomp = Counter()
    for n, aa in enumerate(seq[:-1]): #n-term
        fragcomp += aminoacidcomposition[aa]
        for ion, modcomp in nfragmentcompositions.items():
            fragments[ion + str(n + 1)] = fragcomp + modcomp
    fragcomp = Counter()
    for n, aa in enumerate(seq[::-1][:-1]): #c-term
        fragcomp += aminoacidcomposition[aa]
        for ion, modcomp in cfragmentcompositions.items():
            fragments[ion + str(n + 1)] = fragcomp + modcomp
    return fragments

def fragment_dists(fragmentprobabilities, fragmentcomposition, minimumabundance, fragint):
    #this uses fragmentcomposition in place of atomiccomposition, and fragmentprobabilities in place of elementalprobabilities
    #atomiccomposition = Counter()
    #for aa in seq:
    #    atomiccomposition += aminoacidcomposition[aa]
    #atomiccomposition['H'] += 2
    #atomiccomposition['O'] += 1

    #natomicsample = Counter()
    #for e, c in fragmentcomposition.items():
    #    for iso in isotopesbyelement[e]:
    #        natomicsample[iso] = fragmentprobabilities[iso] * c

    basevector = elementvector.copy()
    for e, v in isotopesbyelement.items():
        sumcount = 0
        for iso in v:
            if iso in fragmentprobabilities:
                if fragmentprobabilities[iso] < 1:
                    #startcount = round(natomicsample[iso])
                    startcount = round(fragmentprobabilities[iso] * fragmentcomposition[e])
                else:
                    startcount = fragmentcomposition[e]
                basevector[vectorpositions[iso]] += startcount
                sumcount += startcount
        if sumcount < fragmentcomposition[e]:
            basevector[vectorpositions[monoisotopickeys[e]]] += 1
        elif sumcount > fragmentcomposition[e]:
            basevector[vectorpositions[monoisotopickeys[e]]] -= 1
            #print(e, 'overestimated')
            #print(sumcount)
            #print(basevector)


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
            mass = elementalmasses[iso] * c
            basemass += mass
            branchmassesbyelement[branchkey][e] += mass
            if fragmentprobabilities[iso] < 1:
                prob = fragmentprobabilities[iso]**c
                baseprob *= prob
                branchprobabilitiesbyelement[branchkey][e] *= prob
                #branchcompositionbyelement[branchkey][e][iso] = c
                #if iso in nonmonoisotopicelements:
                #combfactor = special.comb(fragmentcomposition[e] - baseoffsets[e], c, exact=True)
                combfactor = math.comb(fragmentcomposition[e] - baseoffsets[e], c)
                baseprob *= combfactor
                branchprobabilitiesbyelement[branchkey][e] *= combfactor
                baseoffsets[e] += c
    
    expansiondirections = {} #branchkey: 1 or -1, a negative direction is only given to nonmonisotopic elements that are in basecomposition generated above
    isotopesbyisodirection = {} #isodirection: iso
    isodirectionsbyelement = defaultdict(list) #e: [isodirections]
    opposingdirections = {} #isodirection: isodirection of the same element moving in the other direction, I don't let these coexist within the same subheap

    isodirection = 0
    #for e, isos in nonmonoisotopicgroups.items():
    #    if e in fragmentcomposition:
    #        for iso in isos:
    #            if iso in fragmentprobabilities:
    #                if fragmentprobabilities[iso] < 1:
    for iso, p in fragmentprobabilities.items():
        if p < 1:
            if iso in nonmonoisotopicelements:
                #positive direction
                isotopesbyisodirection[isodirection] = iso
                isodirectionsbyelement[e].append(isodirection)
                expansiondirections[isodirection] = 1
                isodirection += 1
                #if basecomposition[iso] > 0:
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
        acount = fragmentcomposition[e]
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
                newelementmass += elementalmasses[loopiso] * c
                if fragmentprobabilities[loopiso] < 1:
                    newelementprob *= fragmentprobabilities[loopiso]**c
                    #if loopiso in nonmonoisotopicelements:
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
            #finalformulas[branchcount] = subformula
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
        #else:
            #if it it doesn't work with the most abundant isotopomer then it won't work with anything, remove this iso from all possibilities moving forward
            #del branchisodirections[branchkey][isodirection]
            #branchopposers[branchkey].add(isodirection) #prevent it in future loops
            #^THIS IS NOT TRUE, some things passed over here might still end up being relevant

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
            #if passing:
            #    branchisodirections[branchkey] = branchisodirections[prior].copy()
        for branchkey, prior, isodirection in branches:
            iso = isotopesbyisodirection[isodirection]
            e = iso[0]
            acount = fragmentcomposition[e]
            newbasecomp = branchcompositions[branchkey]
            n = 0
            newelementmass = 0
            newelementprob = 1
            for en in vectorrangesbyelement[e]:
                c = newbasecomp[en]
                if c > 0:
                    loopiso = elementpositions[en]
                    newelementmass += elementalmasses[loopiso] * c
                    if fragmentprobabilities[loopiso] < 1:
                        newelementprob *= fragmentprobabilities[loopiso]**c
                        #if loopiso in nonmonoisotopicelements:
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
    massesandabundances = [[], []]
    branchformulas = {}
    #formulas = []
    for k, m in finalmasses.items():
        #subformula = ''
        #for n, c in enumerate(branchcompositions[k]):
        #    if c > 0:
        #        subformula += f'{elementpositions[n]}({c})'
        massesandabundances[0].append(m)
        massesandabundances[1].append(finalprobabilities[k] * fragint)
        #formulas.append(subformula)
        #branchformulas[subformula] = k

    massesandabundances = np.array(massesandabundances)
    #formulas = np.array(formulas, dtype='S')
    #formulas = formulas[massesandabundances[1].argsort()[::-1]].tolist()
    massesandabundances = massesandabundances[:,massesandabundances[1].argsort()[::-1]].tolist()
    #return formulas, massesandabundances
    #formulas might not be accurate because of fragmentprobabilities? i haven't checked this
    return massesandabundances

def max_estimation(c, eprobs):
    countsum = 0
    isoprobs = {}
    arraycounts = []
    arrayisotopes = []
    while eprobs:
        p, iso = heapq.heappop(eprobs)
        p *= -1
        isoprobs[iso] = p
        roundestimate = round(c * p)
        arraycounts.append(roundestimate)
        arrayisotopes.append(iso)
        countsum += roundestimate
    testcomps = [arraycounts.copy() for _ in arraycounts]
    if countsum < c:
        for n, t in enumerate(testcomps):
            t[n] += 1
    elif countsum > c:
        for n, t in enumerate(testcomps):
            if t[n] > 0:
                t[n] -= 1
            else:
                t[0] -= 1
                t[n] += 1
    else:
        for n, t in enumerate(testcomps[:-1]):
            t[0] -= 1
            t[n+1] += 1
    
    probvals = []
    massvals = []
    probvectors = []
    for comp in testcomps:
        pn = 0
        newprob = 1
        newmass = 0
        for n, count in enumerate(comp):
            newprob *= isoprobs[arrayisotopes[n]] ** count
            newmass += elementalmasses[arrayisotopes[n]] * count
            if n > 0:
                newprob *= math.comb(c-pn, count)
                pn += count
        probvals.append(newprob)
        massvals.append(newmass)
        probvectors.append(comp.copy())
    maxprob = max(probvals)
    maxind = probvals.index(maxprob)
    maxmass = massvals[maxind]
    #maxvec = probvectors[maxind]
    #print(arrayisotopes)
    #print(maxvec)
    #print(round(maxmass, 5), round(maxprob, 5))
    #print('~')
    return maxmass, maxprob

def max_fragment(fragprobs, fragcomp, fragint):
    estimates = defaultdict(list)
    for iso, prob in fragprobs.items():
        heapq.heappush(estimates[iso[0]], [-prob, iso])
    mass = 0
    prob = fragint
    for e, c in fragcomp.items():
        if len(estimates[e]) > -1:
            #print(e)
            elementmass, elementprob = max_estimation(c, estimates[e])
            mass += elementmass
            prob *= elementprob
        else:
            p, iso = estimates[e]
            mass += elementalmasses[iso] * c
            #prob *= elementalprobabilities[iso] ** c
    return mass, prob

proton = 1.007276554940804

#generate random peptides
#make ms1 dists, organize them into proton positions
#grab ~1-2 random iso positions from this, coordinates via max
#generate all fragments
#input comp probabilities generated outside of this function, that are based on the ms1 isotopomer its being derived from
#input raw elemental composition of the fragment

nseqs = 1
maxchoice = 3
minimumabundance = 0.001
subisotopomericdepth = 0.8

maxfrags = []

info = []
times = []
depths = []
for _ in range(nseqs):
    seq = ''.join((random.choices(list(aminoacidcomposition.keys()), k=np.random.randint(6,50))))
    #generate fragments
    fragments = fragmentation_compositions(seq)
    #generate ms1 distributions
    formulas, massesandabundances = vector_gen(seq, minimumabundance)
    masses, intensities = np.array(massesandabundances)
    intensities = intensities[masses.argsort()]
    sortedformulas = np.array([i.decode() for i in formulas])[masses.argsort()]
    masses = np.sort(masses)
    #organize proton locations
    maxmass = masses[intensities.argmax()]
    csteps = masses - masses.min()
    maxstep = masses.size
    steprange = proton * np.arange(maxstep)
    stepclasses = np.abs(csteps - steprange[:,None]).argmin(axis=0)
    cinds, counts = np.unique(stepclasses, return_counts=True)
    csplit = counts.cumsum().tolist()
    ci = 0
    splitmasses, splitintensities, splitforms = [], [], []
    for cs in csplit:
        msplit = masses[ci:cs]
        isplit = intensities[ci:cs]
        fsplit = sortedformulas[ci:cs]
        msorted = msplit[isplit.argsort()[::-1]]
        fsorted = fsplit[isplit.argsort()[::-1]]
        isorted = np.sort(isplit)[::-1]
        #isum = sum(isplit)
        #inorm = [i/isum for i in isplit]
        inorm = isorted.cumsum() / isorted.sum()
        depthcut = np.where(inorm >= subisotopomericdepth)[0][0]
        intensitypicks = isorted[:depthcut+1]
        topinds = np.where(isplit == intensitypicks[:,None])[1].tolist()
        #ic = 0
        #for n, i in enumerate(inorm):
        #    ic += i
        #    if ic >= subisotopomericdepth:
        #        depths.append((n+1) / len(inorm))
        splitmasses.append(msorted)
        splitforms.append(fsorted)
        splitintensities.append(isorted)
        ci = cs
    #make a random + adjacent choice of 1-2 from the top 4, that's good enough
    options = list(range(len(splitforms)))
    nchoices = len(options) + 1
    while nchoices > len(options):
        nchoices = np.random.randint(1, maxchoice + 1)
    selection = np.random.choice(options, size=nchoices, replace=False) #i don't care about adjacency i suppose
    for s in selection:
        sf = splitforms[s][0]
        sm = splitmasses[s][0]
        si = splitintensities[s][0]
        #determine competetive isotopes and generate fragment isotope probabilities
        isocounts = set()
        competing = set()
        competitors = {}
        isosums = {}
        for ss in sf.split(')')[:-1]:
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
        isoprobs = {}
        for e, v in competitors.items():
            if e in competing:
                for iso, c in competitors[e].items():
                    isoprobs[iso] = c / isosums[e]
            else:
                for iso in v:
                    isoprobs[iso] = 1
        for ion, fragcomp in fragments.items():
            fragprobs = {k: v for k, v in isoprobs.items() if k[0] in fragcomp}
            #^^^generating a minimum set of these a priori would be SUPER DOPE, maybe in the fragments function? it would definitely save time on the actual fragspec script
            t1 = time()
            fragdist = fragment_dists(fragprobs, fragcomp, minimumabundance, si)
            r1 = time() - t1
            t2 = time()
            mainrep = max_fragment(fragprobs, fragcomp, si)
            r2 = time() - t2
            times.append([r1, r2])
            info.append([seq, ion])
            maxfrags.append(mainrep)
            passed = True
            if not np.isclose(mainrep[0], fragdist[0]).any():
                passed = False
            if not np.isclose(mainrep[1], fragdist[1]).any():
                passed = False
            if not passed:
                print(seq, ion)

#times = np.array(times)
#plt.plot(times[:,0], times[:,1], '.')
#plt.show()
#
#plt.hist(times[:,0], bins=1000, label='old', alpha=0.4)
#plt.hist(times[:,1], bins=1000, label='new', alpha=0.4)
#plt.vlines(times[:,1].mean(), 0, 10e6)
#plt.legend()
#plt.yscale('log')
#plt.xscale('log')
#plt.show()

#post-completion comparison:
#if you only use the highest subosotopomer, and generate the highest 2 fragisos of that to determine whether left/right is a dominant position, does this match the reality of the true fragment distribution calculated from all subisotopomers?

#an exploration to do:
#do ms1 subisotopomers basically produce the same-shaped fragment distributions?
#^and does the 2nd highest of both max and sum theoretical frag dists show up in the same spot?

#pick a sequence from the loop above and determine whether all the subisos from a proton location share the same fragment distribution patterns

#s = 2
#distsbyionformula = defaultdict(dict) #ion: subformula: fragdist
#for sf, sm, si in zip(splitforms[s], splitmasses[s], splitintensities[s]):
#    #determine competetive isotopes and generate fragment isotope probabilities
#    isocounts = set()
#    competing = set()
#    competitors = {}
#    isosums = {}
#    for ss in sf.split(')')[:-1]:
#        iso, c = ss.split('(')
#        c = int(c)
#        e = iso[0]
#        if e in isocounts:
#            competing.add(e)
#            competitors[e][iso] = c
#            isosums[e] += c
#        else:
#            isocounts.add(e)
#            competitors[e] = {iso: c}
#            isosums[e] = c
#    isoprobs = {}
#    for e, v in competitors.items():
#        if e in competing:
#            for iso, c in competitors[e].items():
#                isoprobs[iso] = c / isosums[e]
#        else:
#            for iso in v:
#                isoprobs[iso] = 1
#    for ion, fragcomp in fragments.items():
#        fragprobs = {k: v for k, v in isoprobs.items() if k[0] in fragcomp}
#        fragdist = fragment_dists(fragprobs, fragcomp, minimumabundance, si)
#        #mainrep = max_fragment(fragprobs, fragcomp, si)
#        distsbyionformula[ion][sf] = fragdist
#
#failed = []
#for ion, dist in distsbyionformula.items():
#    fig, ax = plt.subplots(figsize=(5,6))
#    for f, ma in dist.items():
#        plt.bar(ma[0], ma[1], width=0.1, label=f, alpha=0.4)
#    plt.legend(bbox_to_anchor=(1,0.5))
#    plt.show()
#    distints = []
#    distmasses = []
#    distintranks = []
#    distmassranks = []
#    for f, ma in dist.items():
#        masses, intensities = np.array(ma)
#        intensities = intensities[masses.argsort()]
#        masses = np.sort(masses)
#        maxmass = masses[intensities.argmax()]
#        distmasses.append(masses.tolist())
#        distints.append(intensities.tolist())
#        csteps = masses - masses.min()
#        maxstep = masses.size
#        steprange = proton * np.arange(maxstep)
#        stepclasses = np.abs(csteps - steprange[:,None]).argmin(axis=0)
#        distintranks.append(intensities.argsort()[::-1].tolist())
#        distmassranks.append(stepclasses.tolist())
#    #check that distranks all line up, in mass order
#    #check that mass proton locations are the same in mass order
#    for mr in zip_longest(*distmassranks):
#        mr = [i for i in mr if i]
#        if len(set(mr)) > 1:
#            failed.append(ion)
#    for ir in zip_longest(*distintranks):
#        ir = [i for i in ir if i]
#        if len(set(ir)) > 1:
#            failed.append(ion)

#so what this tells me is that there's quite a lot of variation within subisotopomer fragments, ie mass differences of like 0.01-0.02 daltons across proton locations for fragments from the same subiso groups 
#i think a good strategy moving forward would be to set a threshold for the search, like 0.9, and use that to take 90% of total intensity from any subisotopomer positions, probably as like the top ~2 suboisotopomers. the ratios from fragments across subiso fragments might be comparable via intensities
#this will massively reduce the load of the search compared to what i'm currently doing.
#subisotopomeric_depth = 0.9
#^this can be set up in libraryaddition, and will yield 3 distributions now: full, and sum, and depth-limited
#the full will have all the information from minimumabundance, sum will be used for dist matching, and depth-limited will be used for fragmentation



#a quick question, do summing and maxing ms1 dists give the same rank order?
#i've probably answered this before (see below), but no, they don't, it probably shifts +1 due to the highest subiso always being on something other than the max, or tending to be, when it does shift at least it probably shifts in the direction of the highest subiso
#however, i'm pretty sure my leniency with the distributionmatching allows for what the difference would be, a shift of 1, it would be good to double-check this
#^and potentially make this leniency into an input
#let's confirm the leniency works that way below

nseqs = 100
minimumabundance = 0.001

info = []
times = []
depths = []
nopes = []
for _ in range(nseqs):
    seq = ''.join((random.choices(list(aminoacidcomposition.keys()), k=np.random.randint(6,50))))
    #generate ms1 distributions
    formulas, massesandabundances = vector_gen(seq, minimumabundance)
    masses, intensities = np.array(massesandabundances)
    intensities = intensities[masses.argsort()]
    sortedformulas = np.array([i.decode() for i in formulas])[masses.argsort()]
    masses = np.sort(masses)
    #organize proton locations
    maxmass = masses[intensities.argmax()]
    csteps = masses - masses.min()
    maxstep = masses.size
    steprange = proton * np.arange(maxstep)
    stepclasses = np.abs(csteps - steprange[:,None]).argmin(axis=0)
    cinds, counts = np.unique(stepclasses, return_counts=True)
    csplit = counts.cumsum().tolist()
    ci = 0
    splitmasses, splitintensities, splitforms = [], [], []
    for cs in csplit:
        msplit = masses[ci:cs]
        isplit = intensities[ci:cs]
        fsplit = sortedformulas[ci:cs]
        msplit = msplit[isplit.argsort()[::-1]].tolist()
        fsplit = fsplit[isplit.argsort()[::-1]].tolist()
        isplit = np.sort(isplit)[::-1].tolist()
        splitmasses.append(msplit)
        splitforms.append(fsplit)
        splitintensities.append(isplit)
        ci = cs
    sums = []
    maxes = []
    for si in splitintensities:
        sums.append(sum(si))
        maxes.append(max(si))
    masses = [np.mean(i) for i in splitmasses]
    sumsort = np.argsort(np.argsort(sums))
    maxsort = np.argsort(np.argsort(maxes))
    sumsort = sumsort.max() - sumsort
    maxsort = maxsort.max() - maxsort
    distdiffs = [abs(m-s) for m, s in zip(maxsort, sumsort)]
    allowance = sum(distdiffs) - 1
    #if not np.array_equal(sumsort, maxsort):
    #    print(nope, seq)
    if allowance >= 0:
        nopes.append(seq)
