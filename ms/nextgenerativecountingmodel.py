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
#elementalprobabilities -> nelementalprobabilities
#you'll probably need to recycle from some of these
#any elements not included aren't needed for the function, just organic elements: C, N, O, S, H
#you don't have to worry about passing any of these dicts through functions, in my workflow I import them from another file where they're stored

#source:
#https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl
#^on this page, values in parenthesis break the summing to 1


nelementalprobabilities = { #isotope: abundance
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

nelementalmasses = { #isotope: mass
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

isotopesbyelement = { #element: isotopes in order of abundance
        'H': ('H1', 'H2'),
        'C': ('C12', 'C13'),
        'N': ('N14', 'N15'),
        'O': ('O16', 'O17', 'O18'),
        'S': ('S32', 'S34', 'S33', 'S36')}

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

elementalmasses = { #element: atomic number: mass
            'H': {1: 1.00782503223, 2: 2.01410177812},
            'C': {12: 12.00000000, 13: 13.00335483507},
            'N': {14: 14.00307400443, 15: 15.00010889888},
            'O': {16: 15.99491461957, 17: 16.99913175650, 18: 17.99915961286},
            'S': {32: 31.9720711744, 33: 32.9714589098, 34: 33.967867004, 36: 35.96708071}}

elementalprobabilities = { #element: atomic number: abundance
        'H': {1: 0.999885, 2: 0.000115},
        'C': {12: 0.9893, 13: 0.0107}, 
        'N': {14: 0.99636, 15: 0.00364},
        'O': {16: 0.99757, 17: 0.00038, 18: 0.00205},
        'S': {32: 0.9499, 33: 0.0075, 34: 0.0425, 36: 0.0001}}

isotopes = {} #element: mass of isotope: abundance of isotope
majorisotopemasses = {} #element: monoisotopic mass
for k, v in elementalmasses.items():
    probs = elementalprobabilities[k]
    isotopes[k] = {}
    n = 0
    for sk, sv in v.items():
        isotopes[k][sv] = probs[sk]
        if n == 0:
            majorisotopemasses[k] = sv
            n += 1

massadditions = defaultdict(dict) #element: (isotopic mass - monoisotopic mass): abundance
isotopomersbyaddition = defaultdict(dict) #element: (isotopic mass - monoisotopic mass): atomic number
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


#this function below currently does something rather inefficient, it calculates the entire probability of each molecule every single time
#in reality you only need to modify the probability of the molecule that came before it in the recursion, which is something like two multiplications + a combination calculation, ie you modify the multinomial with binomials
def limited_multinomial(elementcount):
    prob = 1
    mass = 0
    for e, v in elementcount.items():
        #v is the {mass addition: count} of isotopes of a single element
        vals = v.values()
        vsum = sum(vals)
        if vsum:
            n = 0
            csum = 0
            for m, c in v.items():
                if c > 0:
                    if m > 0:
                        mass += m * c
                        prob *= special.comb(vsum-n, c, exact=True)
                        #n += 1
                        n += c
                        #^whether i use n += c or n += 1 doesn't change the output, but I swear it should... and I think n += c is the right way of doing it afaik
                    prob *= massadditions[e][m]**c
                    csum += c
                if csum >= vsum: #slight speed boost, time saver
                    break
    return prob, mass

#this recursive function below alternates the isotopic composition of elements via  +/- 1 and passes input to itself if the new mass hasn't been seen already (you can redundantly calculate the same thing twice like this, but the second 'if' statement below stops it)
#it assumes all possible masses derived from any isotope combination are unique - which for my purposes, they are
#the 'samplesize' cutoff here it something I would rather change to 'minimumabundance', samplesize is currently 1/minimumabundance. If I wanted to find all isotopomers with an abundance greater than 0.05, my samplesize would need to be 20, but I don't see/remember any need for it to not just be the actual abundance

#woopsy, wrong one
#def expansion_organizer(elementcount, elementpriority, samplesize, abundanceprobs):
#    fullprob, currentmass = limited_multinomial(elementcount)
#    if fullprob * samplesize > 1:
#        subformulastring = ''.join((''.join((f'{e}{isotopomersbyaddition[e][m]}({c})' for m, c in v.items() if c > 0)) for e, v in elementcount.items())) #this outputs a formula that describes the individual isotopes involved
#        print(subformulastring, fullprob)
#        abundanceprobs[currentmass][subformulastring] = fullprob
#        for e, m, tp in elementpriority:
#            if currentmass + m not in abundanceprobs: #relying on all combinatorics of these masses being unique, i'm pretty sure they end up that way because of how all massadditions are unique, any calculatable overlap would probably be further away than any sequence length limitations would realistically allow
#                elementcount[e][0] -= 1 #i only remove from monoisotopic counts, combinatorics handles the rest
#                elementcount[e][m] += 1
#                aprobs = expansion_organizer(elementcount, elementpriority, samplesize, abundanceprobs)
#                elementcount[e][0] += 1
#                elementcount[e][m] -= 1
#                if not aprobs:
#                    break
#                #^breaking here should be a speedup but the elementpriority should be shifting as you add more isotopes to an element so i think this SHOULD give an incorrect answer, but it doesn't seem to, which is strange
#                abundanceprobs.update(aprobs)
#            #else:
#                #new mass and abundance have already been calculated prior
#    return abundanceprobs

def expansion_organizer(massadditions, elementcount, samplesize, isotopomersbyaddition):
    abundanceprobs = defaultdict(dict)
    fullprob, standingmass = limited_multinomial(elementcount)
    if fullprob * samplesize > 1:
        subformulastring = ''.join((''.join((f'{e}{isotopomersbyaddition[e][m]}({c})' for m, c in v.items() if c > 0)) for e, v in elementcount.items())) #this outputs a formula that describes the individual isotopes involved
        for e, v in elementcount.items():
            if v[0] > 0:
                for m, p in v.items():
                    if m > 0:
                        if standingmass + m not in abundanceprobs: #relying on all combinatorics of these masses being unique, i'm pretty sure they end up that way because of how all massadditions are unique, any calculatable overlap would probably be way further away than any of the length limitations would realistically allow
                            elementcount[e][0] -= 1
                            elementcount[e][m] += 1
                            abundanceprobs.update(expansion_organizer(massadditions, elementcount, samplesize, isotopomersbyaddition))
                            elementcount[e][0] += 1
                            elementcount[e][m] -= 1
            #else:
                #you don't need to worry about these because the combination of all non-0 mass addition spots get hit combinatorically like this, for as many as there are - given it passes samplesize
        abundanceprobs[standingmass][subformulastring] = fullprob
    return abundanceprobs

def distribution_generation(seq, samplesize):
    atomiccomposition = Counter()
    for aa in seq:
        atomiccomposition += aminoacidcomposition[aa]
    #no OH loss on last residue, no H lost on first residue
    atomiccomposition['H'] += 2
    atomiccomposition['O'] += 1
    #^make sure to +H2O to whatever sequence composition there is in your function, its just some water chemistry that gets added to either end, I don't add it to the individual pieces above because they end up linked together like a string of legos and they don't have room for the water molecule in between them.
    monoisotopicmass = sum(majorisotopemasses[k]*v for k, v in atomiccomposition.items())
    
    elementcount = {} #element: (isotopic mass - monoisotopic mass): count
    for e, c in atomiccomposition.items():
        elementcount[e] = {0: c}
    for e, v in massadditions.items():
        if e in atomiccomposition:
            for m, c in v.items():
                if m > 0:
                    elementcount[e][m] = 0
    
    #this below only lists elements present in the peptide, sometimes there's no Sulfur
    #elementpriority = [] #[element, mass addition, binomial probability]
    #for element, count in atomiccomposition.items():
    #    for m, p in massadditions[element].items():
    #        if m > 0:
    #            elementpriority.append([element, m, p*count])
    #elementpriority = sorted(elementpriority, key=lambda x: -x[2])
    #^this sorts isotopes in order of the most likely binomial to be the next highest abundance isotopomer
    #^this isn't necessary for this recursive approach because I don't re-calculate these on the fly, when in reality it would be necessary to do that if you want it to be useful
    
    abundanceprobs = {}
    #this while loop is here because of my own stupidity. This is necessary because in this function I start calculating isotopes at the monoisotopic mass. For larger molecules, the monoisotopic mass can actually be really small (and below my abundance threshold) so the functions above fail to trigger any isotopomer calculations at all
    while not abundanceprobs:
        #abundanceprobs.update(expansion_organizer(elementcount, elementpriority, samplesize, defaultdict(dict)))
        abundanceprobs.update(expansion_organizer(massadditions, elementcount, samplesize, isotopomersbyaddition))
        samplesize *= 2
    
    #below adds the monoisotopic mass to the mass additions delivered from expansion_organizer to get the final masses
    massesandabundances = [[], []]
    formulas = []
    for m, fp in abundanceprobs.items():
        for f, p in fp.items(): #length of fp will always be 1 because of currentmass+m blocking in expansion_organizer
            massesandabundances[0].append(monoisotopicmass + m)
            massesandabundances[1].append(p)
            formulas.append(f)
    
    #sorting everything by mass
    massesandabundances = np.array(massesandabundances)
    formulas = np.array(formulas, dtype='S')
    formulas = formulas[massesandabundances[0].argsort()].tolist()
    massesandabundances = massesandabundances[:,massesandabundances[0].argsort()]
    return massesandabundances, formulas


#function inputs:
samplesize = 100
seq = 'GTSWGLPASKTITTMIDGPQDLRVVAVTPTTLELGWLRPQAEVDR'
#^i'll anchor the scaling price to this sequence and samplesize/minimumabundance specifically, this is on the longer end of what i'll need to process
#longer sequences have more elements -> more multinomial mechanics -> and therefore more room for speed improvement, so there's more room for you to earn!

abundances, subformulas = distribution_generation(seq, samplesize)


#output for comparison with the heap workflow below
#np.sort(abundances[1])
#np.array(subformulas)[abundances[1].argsort()] #last has the highest abundance


plt.bar(abundances[0], abundances[1], width=0.01)
plt.show()

#profiling?!
#
#bt = time()
#n = 0
#times, lengths = [], []
#while n < 99999:
#    seq = ''.join((random.choices(list(aminoacidcomposition.keys()), k=np.random.randint(6,50))))
#    nt = time()
#    abundances, subformulas = distribution_generation(seq, samplesize)
#    end = time() - nt
#    times.append(end)
#    lengths.append(len(subformulas))
#    n += 1
#print(time() - bt)
#
#plt.plot(lengths, times)
#plt.show()

#!~~~~~ New function below

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
        natomicsample[iso] = nelementalprobabilities[iso] * c

#basecomposition = Counter()
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
            #combfactor = special.comb(atomiccomposition[e] - baseoffsets[e], c, exact=True)
            combfactor = math.comb(atomiccomposition[e] - baseoffsets[e], c)
            baseprob *= combfactor
            branchprobabilitiesbyelement[branchkey][e] *= combfactor
            baseoffsets[e] += c

#baseformula = ''.join((f'{k}({basecomposition[k]})' for k in sorted(basecomposition) if basecomposition[k] > 0))
baseformula = ''.join((sorted(f'{k}({v})' for k, v in basecomposition.items() if v > 0)))

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

#^this above loop will be the initiator
#reintroduce priorbranch, what's done via base here will be done via priors later
#basically stick ^that thing in a while loop, and an upper level will iterate a list of current branchkeys, the list of branchkeys gets refreshed every round after being completely assessed

#for blocking opposers:
#this will be the first thing in the isotopesbyisodirection loop
#check if isodirection not in branchopposers[branchkey]
#also copy opposers from the prior
#^this might not be ideal actually, i should be micromanaging isotopesbyisodirection PER branch, because if something doesn't work once -> get rid of it and don't try calculating it again IN THAT BRANCH. AND i think maybe only remove things from positive direction branches? actually i do think its fine to remove from negatives
#^yes im ok with this because everything technically comes from something else that's higher up on the chain, so directions and additions should be managed like this
#so in essence opposingdireciton swill operate on branchkey level, and branchisodirections will operate on priors

#how to stop redundancy?
#^+/- direction to the basecomp copy first, make the string, then allow everything to pass afterwards
#^this could be a problem if the strings aren't made perfectly identical
#you could do an alternative version that doesn't produce subformulas and lookup the mass or prob to check for redundancy, list lookup might be feasible due to size



while currentbranchkeys:
    nextbranchkeys = []
    for branchkey in currentbranchkeys:
        prior = priorbranch[branchkey]
        branchisodirections[branchkey] = branchisodirections[prior].copy()
        for isodirection, iso in branchisodirections[prior].items():
            if isodirection not in branchopposers[branchkey]:
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
                            #^most time is spent on this comb function, is there a way to summarize it using the info at hand? stackexchange question
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
                            #del branchcompositionbyelement[branchcount][e][iso]
                            #del branchcompositions[branchcount][iso]
                        elif newbasecomp[mk] == 0:
                            #end of a positive direction, remove that isodirection in the future branches
                            del branchisodirections[branchkey][isodirection]
                            #del branchcompositionbyelement[branchcount][e][mk]
                            #del branchcompositions[branchcount][iso]
                        if isodirection in opposingdirections:
                            #remove opposing direction from isodirections
                            branchopposers[branchcount].add(opposingdirections[isodirection])
                        nextbranchkeys.append(branchcount)
                        branchcount += 1
    currentbranchkeys = nextbranchkeys.copy()


#so i think the input should be a vector, ie [0, 0, 0, 0, 0, 0, 0, etc] where each index is reserved for an iso
#this would enable making strings without sorting, and you could add tuples to a set to iterate with less redundancy in the while loop
#a problem being that i would have a harder time organizing isodirections
#i think pos and negative directions would need to be done separately in different while loops

massesandabundances = [[], []]
formulas = []
for k, m in finalmasses.items():
    massesandabundances[0].append(m)
    massesandabundances[1].append(finalprobabilities[k])
    formulas.append(finalformulas[k])

#sorting everything by mass
massesandabundances = np.array(massesandabundances)
formulas = np.array(formulas, dtype='S')
formulas = formulas[massesandabundances[0].argsort()].tolist()
massesandabundances = massesandabundances[:,massesandabundances[0].argsort()]
