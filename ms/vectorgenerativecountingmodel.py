from collections import defaultdict, Counter
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from time import time
import random
import heapq
import math
import os
import sys
sys.setrecursionlimit(10000)

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


#element masses and natural abundances on earth in various dicts that link their characteristics
#for the older function i use straightforward names
#if i re-adapted something for the new function i probably put an 'n' in front of the name
#elementalprobabilities -> elementalprobabilities
#you'll probably need to recycle from some of these
#any elements not included aren't needed for the function, just organic elements: C, N, O, S, H
#you don't have to worry about passing any of these dicts through functions, in my workflow I import them from another file where they're stored

#source:
#https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl
#^on this page, values in parenthesis break the summing to 1


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
#vectorslicesbyelement = {'H': slice(0,2),
#                         'C': slice(2,4),
#                         'N': slice(4,6),
#                         'O': slice(6,9),
#                         'S': slice(9,14)}
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

#function inputs:
samplesize = 100
seq = 'GTSWGLPASKTITTMIDGPQDLRVVAVTPTTLELGWLRPQAEVDR'
seq = 'MGTPQSLQWVASQLEYQNLV'
seq = 'CTISSELFWYATAFKYFHWHWL'
seq = 'VFETWFD'
seq = 'MMALMACSGQAWNMVLAKCDCYIHVAMHIMLVTALDECPCGLSGM'
seq = 'GRYADDYLHHYNTLSMPVE'
seq = 'TVKNMNGWGPIHCNNTCDRSDACPGMTMPMWRDFGCMTESMHMMCATM'
seq = 'EIPIQDVAIQDFTCDGNEESSCQLSPRCPEQCTCMETVVRCSNK'
seq = 'IGMGINRYNAQLFRPITNFSCAARCMMHVRCGDLIIDSHQILWFCTTSVNYDDQEPVPIWALQLICCRCYIHTEAWHAMCSEVKHTDTLAAHCVECWL'

minimumabundance = 0.02

atomiccomposition = Counter()
for aa in seq:
    atomiccomposition += aminoacidcomposition[aa]
#no OH loss on last residue, no H lost on first residue
atomiccomposition['H'] += 2
atomiccomposition['O'] += 1
#^make sure to +H2O to whatever sequence composition there is in your function, its just some water chemistry that gets added to either end, I don't add it to the individual pieces above because they end up linked together like a string of legos and they don't have room for the water molecule in between them.

natomicsample = Counter()
for e, c in atomiccomposition.items():
    for iso in isotopesbyelement[e]:
        natomicsample[iso] = elementalprobabilities[iso] * c

#basecomposition = Counter()
#basecomposition = {}
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

#below calculates the mass and abundance of the most abundant isotopomer, and organizes the binomials its made from
baseprob = 1
basemass = 0
baseoffsets = defaultdict(int)
branchprobabilitiesbyelement = defaultdict(lambda: defaultdict(lambda: 1)) #branchkey: e: prob
branchmassesbyelement = defaultdict(lambda: defaultdict(int)) #branchkey: e: mass
#branchcompositionbyelement = defaultdict(lambda: defaultdict(lambda: Counter())) #branchkey: e: {composition dict}
#branchcompositionbyelement = defaultdict(lambda: defaultdict(dict)) #branchkey: e: {composition dict}
#for iso, c in basecomposition.items():
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

#baseformula = ''.join((f'{k}({basecomposition[k]})' for k in sorted(basecomposition) if basecomposition[k] > 0))
#baseformula = ''.join((sorted(f'{k}({v})' for k, v in basecomposition.items() if v > 0)))
#baseformula = ''
#for n, c in enumerate(basevector):
#    if c > 0:
#        baseformula += f'{elementpositions[n]}({c})'

#the general workflow alternates isotopic composition of individual elements, and to do this i distinguish positive and negative directions as different 'branches' that go in those directions.
#ie: something with {O16: 5, O18: 2} could become {O16: 6, O18: 1} or {O16: 4, O18: 3} depending on the direction associated with that O18. O16 doesn't have a isodirection because only nonmonoisotopic elements get added/subtracted from the composition.
#isodirections are used to keep track of an element + direction combination

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
#finalformulas = {} #branchkey: subformula
finalmasses = {} #branchkey: mass

finalprobabilities[branchkey] = baseprob
#finalformulas[branchkey] = baseformula
finalmasses[branchkey] = basemass

branchcount = 1
currentbranchkeys = [] #list of all currently unexplored branchkeys
priorbranch = {} #branchkey: branchkey of branch that generated this branch
branchopposers = defaultdict(set) #branchkey: set of non-compatible isodirections
branchprobabilities = defaultdict(dict) #branchkey: isodirection: combined element prob
branchmasses = defaultdict(dict) #branchkey: isodirection: combined element mass
#branchcompositions = defaultdict(dict) #branchcount: isotope: count
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


#so i think the input should be a vector, ie [0, 0, 0, 0, 0, 0, 0, etc] where each index is reserved for an iso
#this would enable making strings without sorting, and you could add tuples to a set to iterate with less redundancy in the while loop
#a problem being that i would have a harder time organizing isodirections
#i think pos and negative directions would need to be done separately in different while loops

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

#sorting everything by mass
massesandabundances = np.array(massesandabundances)
formulas = np.array(formulas, dtype='S')
formulas = formulas[massesandabundances[1].argsort()[::-1]].tolist()
massesandabundances = massesandabundances[:,massesandabundances[1].argsort()[::-1]].tolist()

for m, a, f in zip(massesandabundances[0], massesandabundances[1], formulas):
    print(m, a)
    print(f)
    print('~')
