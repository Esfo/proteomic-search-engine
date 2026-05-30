from collections import defaultdict, Counter
import numpy as np
import itertools
from scipy import special
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from time import time
import bisect
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

elementvectors = {}
nvectorpositions = {}
nelementpositions = {}
for e, isos in isotopesbyelement.items():
    elementvectors[e] = [0 for _ in range(len(isos))]
    nvectorpositions[e] = {k: n for n, k in enumerate(isos)}
    nelementpositions[e] = {n: k for n, k in enumerate(isos)}


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
seq = 'GTSWGLPASKTITTMIDGPQDLRVVAVTPTTLELGWLRPQAEVDR'
seq = 'MGTPQSLQWVASQLEYQNLV'
seq = 'CTISSELFWYATAFKYFHWHWL'
seq = 'VFETWFD'
seq = 'MMALMACSGQAWNMVLAKCDCYIHVAMHIMLVTALDECPCGLSGM'
seq = 'GRYADDYLHHYNTLSMPVE'
seq = 'TVKNMNGWGPIHCNNTCDRSDACPGMTMPMWRDFGCMTESMHMMCATM'
seq = 'EIPIQDVAIQDFTCDGNEESSCQLSPRCPEQCTCMETVVRCSNK'
seq = 'IGMGINRYNAQLFRPITNFSCAARCMMHVRCGDLIIDSHQILWFCTTSVNYDDQEPVPIWALQLICCRCYIHTEAWHAMCSEVKHTDTLAAHCVECWL'
#seq = 'TVCYSPN'

minimumabundance = 0.01

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
formulas = formulas[massesandabundances[1].argsort()[::-1]].tolist()
vectormassesandabundances = massesandabundances[:,massesandabundances[1].argsort()[::-1]].tolist()


#~~~~~~~~~~~~~~~new function below

#nisos = len(vectormassesandabundances[0])
#
#atomiccomposition = Counter()
#for aa in seq:
#    atomiccomposition += aminoacidcomposition[aa]
##no OH loss on last residue, no H lost on first residue
#atomiccomposition['H'] += 2
#atomiccomposition['O'] += 1
##^make sure to +H2O to whatever sequence composition there is in your function, its just some water chemistry that gets added to either end, I don't add it to the individual pieces above because they end up linked together like a string of legos and they don't have room for the water molecule in between them.
#
##this startup is strategized around organic element that have 99-1 distributions, it would be inefficient for something like a 50-50
##this function is alright, its slow still, a proper way to do this faster would be to include all of the isotopes into a single heap again, but track percent change from last prob, it stops when the length of the cartesian product (list(itertools.product(*elements))) reaches the length of nisos, count the length with this:
##but this might be for later, i want to get the decreasing isos on the road
#
#elementinfo = defaultdict(list)
#for e, acount in atomiccomposition.items():
#        mk = monoisotopickeys[e]
#        nvector = elementvectors[e].copy()
#        nvector[nvectorpositions[e][mk]] += acount
#        if len(isotopesbyelement[e]) > 2:
#            vectorsets = set()
#            mainheap = [[elementalprobabilities[mk] ** acount, acount * elementalmasses[mk], nvector.copy()]] #prob, compkey
#            for iso in nonmonoisotopicgroups[e]:
#                newelementvector = nvector.copy()
#                newelementvector[nvectorpositions[e][mk]] -= 1
#                newelementvector[nvectorpositions[e][iso]] += 1
#                vectorsets.add(tuple(newelementvector))
#                n = 0
#                newelementmass = 0
#                newelementprob = 1
#                for n, c in enumerate(newelementvector):
#                    loopiso = nelementpositions[e][n]
#                    newelementmass += elementalmasses[loopiso] * c
#                    newelementprob *= elementalprobabilities[loopiso]**c
#                    if loopiso in nonmonoisotopicelements:
#                        newelementprob *= math.comb(acount-n, c)
#                        n += c
#                heapq.heappush(mainheap, [-newelementprob, newelementmass, newelementvector.copy()]) #does this need to copy?
#            n = 0
#            decreasers = 0
#            #lastprob = 1 #first one auto-decreases, so nisos gets an informal +1 here
#            outputprobs = []
#            p, m, v = heapq.heappop(mainheap)
#            while decreasers <= nisos:
#                elementinfo[e].append([-p, m, v])
#                for iso in nonmonoisotopicgroups[e]:
#                    newelementvector = v.copy()
#                    newelementvector[nvectorpositions[e][mk]] -= 1
#                    newelementvector[nvectorpositions[e][iso]] += 1
#                    tuplevec = tuple(newelementvector)
#                    if tuplevec not in vectorsets:
#                        vectorsets.add(tuplevec)
#                        n = 0
#                        newelementmass = 0
#                        newelementprob = 1
#                        for n, c in enumerate(newelementvector):
#                            loopiso = nelementpositions[e][n]
#                            newelementmass += elementalmasses[loopiso] * c
#                            newelementprob *= elementalprobabilities[loopiso]**c
#                            if loopiso in nonmonoisotopicelements:
#                                newelementprob *= math.comb(acount-n, c)
#                                n += c
#                        heapq.heappush(mainheap, [-newelementprob, newelementmass, newelementvector.copy()]) #does this need to copy?
#                lastprob = p
#                p, m, v = heapq.heappop(mainheap)
#                if p == 0:
#                    break
#                if p > lastprob:
#                    decreasers += 1
#                else:
#                    decreasers = 0
#        else:
#            lastprob = elementalprobabilities[mk] ** acount
#            elementinfo[e].append([lastprob, acount * elementalmasses[mk], nvector.copy()])
#            iso = nonmonoisotopicgroups[e][0]
#            decreasers = 0
#            while decreasers < nisos:
#                nvector[nvectorpositions[e][mk]] -= 1
#                nvector[nvectorpositions[e][iso]] += 1
#                n = 0
#                newelementmass = 0
#                newelementprob = 1
#                for n, c in enumerate(nvector):
#                    loopiso = nelementpositions[e][n]
#                    newelementmass += elementalmasses[loopiso] * c
#                    newelementprob *= elementalprobabilities[loopiso]**c
#                    if loopiso in nonmonoisotopicelements:
#                        newelementprob *= math.comb(acount-n, c)
#                        n += c
#                elementinfo[e].append([newelementprob, newelementmass, nvector.copy()])
#                if newelementprob == 0:
#                    break
#                if newelementprob < lastprob:
#                    decreasers += 1
#                else:
#                    decreasers = 0
#                lastprob = newelementprob
#
##above is functional but not finished
##second new part below this ~~~
#
#def mtest(acount, e='C'):
#    mk = monoisotopickeys[e]
#    nvector = elementvectors[e].copy()
#    nvector[nvectorpositions[e][mk]] += acount
#    lastprob = elementalprobabilities[mk] ** acount
#    elementinfo = []
#    elementinfo.append([lastprob, acount * elementalmasses[mk], nvector.copy()])
#    iso = nonmonoisotopicgroups[e][0]
#    decreasers = 0
#    while decreasers < 5:
#        nvector[nvectorpositions[e][mk]] -= 1
#        nvector[nvectorpositions[e][iso]] += 1
#        n = 0
#        newelementmass = 0
#        newelementprob = 1
#        for n, c in enumerate(nvector):
#            loopiso = nelementpositions[e][n]
#            newelementmass += elementalmasses[loopiso] * c
#            newelementprob *= elementalprobabilities[loopiso]**c
#            if loopiso in nonmonoisotopicelements:
#                newelementprob *= math.comb(acount-n, c)
#                n += c
#        elementinfo.append([newelementprob, newelementmass, nvector.copy()])
#        if newelementprob == 0:
#            break
#        if newelementprob < lastprob:
#            decreasers += 1
#        else:
#            decreasers = 0
#        lastprob = newelementprob
#    return sorted(elementinfo)[-1][2][0]
#
#
#def test(n):
#    p_C12 = elementalprobabilities['C12']
#    p_C13 = elementalprobabilities['C13']
#    k_max = math.floor((n+1) * p_C13)
#    #print(f"The most abundant isotopomer of C{n} is expected to have {n - k_max} C-12 atoms and {k_max} C-13 atoms.")
#    return n - k_max

#output = []
#for n in range(99999):
#    n += 1
#    to = test(n)
#    tm = mtest(n)
#    output.append([n, to, tm])


#~~~

nisos = len(vectormassesandabundances[0])

atomiccomposition = Counter()
for aa in seq:
    atomiccomposition += aminoacidcomposition[aa]
#no OH loss on last residue, no H lost on first residue
atomiccomposition['H'] += 2
atomiccomposition['O'] += 1

binomialendpoints = defaultdict(list)
mainheap = []
vectorsets = defaultdict(set) #element: set of used vectors
nelements = len(atomiccomposition)
#etracker = defaultdict(int) #element: count of popped isotopes of this element
maxabundances = {} #element: highest abundance of that element
lastprobs = {} #element: last probability of that element
for e, acount in atomiccomposition.items():
    mk = monoisotopickeys[e]
    nvector = elementvectors[e].copy()
    nvector[nvectorpositions[e][mk]] += acount
    if len(isotopesbyelement[e]) > 2:
        baseprob = elementalprobabilities[mk] ** acount
        preheap = []
        preheap.append([baseprob, acount * elementalmasses[mk], e, nvector.copy()])
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
                        newelementmass += elementalmasses[loopiso] * c
                        newelementprob *= elementalprobabilities[loopiso]**c
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
        binomialendpoints[e].append([maxprob, m, nv])
        #binomialendpoints[e].append([maxprob, m])
        #etracker[e] += 1
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
                        newelementmass += elementalmasses[loopiso] * c
                        newelementprob *= elementalprobabilities[loopiso]**c
                        if loopiso in nonmonoisotopicelements:
                            newelementprob *= math.comb(acount-pn, c)
                            pn += c
                    heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, v.copy()])
                    #heapq.heappush(mainheap, [-newelementprob, newelementmass, e, v.copy()])
    else:
        preheap = []
        baseprob = elementalprobabilities[mk] ** acount
        lastprobs[e] = baseprob
        preheap.append([baseprob, acount * elementalmasses[mk], e, nvector.copy()])
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
                    newelementmass += elementalmasses[loopiso] * c
                    newelementprob *= elementalprobabilities[loopiso]**c
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
        binomialendpoints[e].append([maxprob, m, nv])
        #binomialendpoints[e].append([maxprob, m])
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
                    newelementmass += elementalmasses[loopiso] * c
                    newelementprob *= elementalprobabilities[loopiso]**c
                    if loopiso in nonmonoisotopicelements:
                        newelementprob *= math.comb(acount-pn, c)
                        pn += c
                heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, v.copy()])
                #heapq.heappush(mainheap, [-newelementprob, newelementmass, e, v.copy()])

mass = 0
prob = 1
massesandabundances = [[], []]
for e, a in binomialendpoints.items():
    p, m, v = a[0]
    prob *= p
    mass += m
massesandabundances[0].append(mass)
massesandabundances[1].append(prob)

r, p, m, e, v = heapq.heappop(mainheap)
binomialorganizer = defaultdict(list)
binomialorganizer[e].append([r, p, m, v])
#heapq.heappush(binomialorganizer[e], [r, p, m, v])
#binomialorganizer[e].append([p, m])
#etracker[e] += 1
#isotracker = 2 #starts here, not above b/c that's just "1"
isotracker = 1
#unfinished = True
#productlength = 1
#for t in etracker.values():
#    productlength *= t
#if productlength >= nisos:
#    unfinished = False
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
                #using vectorsets like this poses a problem, if something has been seen before, then that branchline just ends
                #^no it doesn't because i'm iterating and pushing all isos in the nonmono group
                if tuplevec not in vectorsets[e]:
                    vectorsets[e].add(tuplevec)
                    pn = 0
                    newelementmass = 0
                    newelementprob = 1
                    for n, c in enumerate(newelementvector):
                        loopiso = nelementpositions[e][n]
                        newelementmass += elementalmasses[loopiso] * c
                        newelementprob *= elementalprobabilities[loopiso]**c
                        if loopiso in nonmonoisotopicelements:
                            newelementprob *= math.comb(acount-pn, c)
                            pn += c
                    heapq.heappush(mainheap, [newelementprob / maxabundances[e], newelementprob, newelementmass, e, newelementvector.copy()])
                    #heapq.heappush(mainheap, [-newelementprob, newelementmass, e, newelementvector.copy()])
    else:
        #keep the mainheap pushes as a list until you're done with greater, then sort to fix their ratios
        #then add the highest of each to the popped final info, and check if nisos > 1
        #this needs a vectorset implementation because initial setup happens in a linear fashion, only the last one would need to be popped, and if its the highest - it will, otherwise, who cares
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
                    newelementmass += elementalmasses[loopiso] * c
                    newelementprob *= elementalprobabilities[loopiso]**c
                    if loopiso in nonmonoisotopicelements:
                        newelementprob *= math.comb(acount-pn, c)
                        pn += c
                heapq.heappush(mainheap, [newelementprob / maxabundances[e], newelementprob, newelementmass, e, nvector.copy()])
                #heapq.heappush(mainheap, [-newelementprob, newelementmass, e, nvector.copy()])
    r, p, m, e, v = heapq.heappop(mainheap)
    binomialorganizer[e].append([r, p, m, v])
    for e, l in binomialorganizer[e].items():
        for 
    #heapq.heappush(binomialorganizer[e], [r, p, m, v])
    #binomialorganizer[e].append([p, m])
    #etracker[e] += 1
    #productlength = 1
    #for t in etracker.values():
    #    productlength *= t
    #if productlength >= nisos * 2:
    #    unfinished = False
    isotracker += 1
    if isotracker >= nisos or p == 0:
        break

#^is it faster to push binomialorganizer OTF or to heapify here after the loop?

#there's got to be some way of arranging this through the pattern
#start on the highest iso of each element, b/c its length is 3
#KEEP the ratios in binomialorganizer, and use the ratios as the way of knowing which is next
#its going to be a cumulative matrix that you're applying each calculation to.
#OTF ranking of initial isotopes.

#for the faster/simple estimate, do the exploration to determine the # of each element you need to cross in order to check each possibility and use that as boolean logic

for prod in itertools.product(*binomialorganizer.values()):
    mass = 0
    prob = 1
    for p, m, v in prod:
        prob *= p
        mass += m
    massesandabundances[0].append(mass)
    massesandabundances[1].append(prob)
massesandabundances = np.array(massesandabundances)
massesandabundances = massesandabundances[:,massesandabundances[1].argsort()[::-1]][:,:nisos].tolist()

for vm, vp, m, p in zip(*vectormassesandabundances, *massesandabundances):
    print(round(vm, 5), round(vp, 5))
    print(round(m, 5), round(p, 5))
    if not np.isclose(vm, m):
        print('no mass')
    if not np.isclose(vp, p):
        print('no prob')
    print('~~~')

#i suppose i can test if the 1.5 or 2 process produces all the desired isotopes
#any prospects for a single max abundance solver?

#for finding a reasonable cutoff time:
#when integrating into the finalized version, count the integration length from the lowest isotopomer
#ie, when a new isotopomer is inserted anywhere other than the bottom of the abundance ranking, count how far away it is from the bottom. this should give you insight into how to cut this off correctly.

#start from natomicsample estimate and go in both directions
#if nisos=4, then you can calculate 4 binomials of each element, the 4 highest i suppose, and all the answers will be in there
#see below:
#In [51]: .8*.8*.2
#Out[51]: 0.12800000000000003
#
#In [52]: .8*.75*.2
#Out[52]: 0.12000000000000002
#
#In [53]: .8*.8*.15
#Out[53]: 0.09600000000000002
#
#In [54]: .75/.8
#Out[54]: 0.9375
#
#In [55]: .15/.2
#Out[55]: 0.7499999999999999
#
#In [56]: .096/.128
#Out[56]: 0.75
#
#In [57]: 0.12/.128
#Out[57]: 0.9375
#you can measure the impact of the changing binomial as percentages, 0.8 -> 0.75 is a smaller % change and therefore has less of an impact then 0.2 -> 1.5, and the exact % change is applicable to the final multinomial
#In [59]: .128*(.75/.8)
#Out[59]: 0.12
#gets you the newest multinomial
