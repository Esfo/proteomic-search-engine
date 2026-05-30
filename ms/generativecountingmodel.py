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

basecomposition = Counter()
fullcomposition = {}
for e, v in isotopesbyelement.items():
    sumcount = 0
    for si in v:
        startcount = round(natomicsample[si])
        #^i tried generating random peptides in order to find a split that came out to 0.5 here but I'm unable to find an example where it actually happens. Hypothetically, if it does, there would be an equal probability between two isotopes, and this wouldn't handle that well. they'll be generated in the below loop regardless I suppose.
        fullcomposition[si] = startcount
        if startcount > 0:
            sumcount += startcount
            basecomposition[si] = startcount
    if sumcount < atomiccomposition[e]:
        #mono isotope rounded down but nothing could fill the gap, add to mono isotope
        basecomposition[monoisotopickeys[e]] += 1
        fullcomposition[monoisotopickeys[e]] += 1
    elif sumcount > atomiccomposition[e]:
        #this has never happened, and probably won't
        #multiple things rounded up.. presumably because there's an even split
        #subtracting from the mono isotope would be the easiest move rather than identifying which non-mono isotope was added
        basecomposition[monoisotopickeys[e]] -= 1
        fullcomposition[monoisotopickeys[e]] -= 1
        print(seq, 'generated an erroneous example i\'ve been trying to catch with isotopic distribution rounding')


branchkey = 0 #this is used as a basis for accessing subheaps

#below calculates the mass and abundance of the most abundant isotopomer, and organizes the binomials its made from
baseprob = 1
basemass = 0
baseoffsets = defaultdict(int)
branchprobabilitiesbyelement = defaultdict(lambda: defaultdict(lambda: 1)) #branchkey: e: prob
branchmassesbyelement = defaultdict(lambda: defaultdict(int)) #branchkey: e: mass
for iso, c in basecomposition.items():
    e = iso[0]
    prob = nelementalprobabilities[iso]**c
    mass = nelementalmasses[iso] * c
    baseprob *= prob
    basemass += mass
    branchprobabilitiesbyelement[branchkey][e] *= prob
    branchmassesbyelement[branchkey][e] += mass
    if iso in nonmonoisotopicelements:
        combfactor = special.comb(atomiccomposition[e] - baseoffsets[e], c, exact=True)
        baseprob *= combfactor
        branchprobabilitiesbyelement[branchkey][e] *= combfactor
        baseoffsets[e] += c

#the general workflow alternates isotopic composition of individual elements, and to do this i distinguish positive and negative directions as different 'branches' that go in those directions.
#ie: something with {O16: 5, O18: 2} could become {O16: 6, O18: 1} or {O16: 4, O18: 3} depending on the direction associated with that O18. O16 doesn't have a subheapkey because only nonmonoisotopic elements get added/subtracted from the composition.
#subheapkeys are used to keep track of an element + direction combination

expansiondirections = {} #branchkey: 1 or -1, a negative direction is only given to nonmonisotopic elements that are in basecomposition generated above
isotopesbysubheapkey = {} #subheapkey: iso
subheapkeysbyelement = defaultdict(list) #e: [subheapkeys]
opposingdirections = {} #subheapkey: subheapkey of the same element moving in the other direction, I don't let these coexist within the same subheap
branchprobabilities = defaultdict(dict) #branchkey: subheapkey: combined element prob
branchmasses = defaultdict(dict) #branchkey: subheapkey: combined element mass

#the below calculates the hypothetical abundance/probability of what the NEXT isotopomer would be IF that specific isotope was added. HOWEVER, this probability is pre-normalized for the final calculation of: (oldprobability / oldbinomial * newbinomial), where i use newelementprob to name newbinomial. This allows the subheap ranking to stay true when a nonmonoisotopic element of a negative direction goes to zero (ie, there would be no probability to calculate, and trying to estimate it from the monoisotopic element gives an incorrect ranking)
subheapkey = 0
for e, isos in nonmonoisotopicgroups.items():
    if e in atomiccomposition:
        acount = atomiccomposition[e]
        mk = monoisotopickeys[e]
        elementcomp = {k: fullcomposition[k] for k in isotopesbyelement[e]}
        for iso in isos:
            newcomp = elementcomp.copy()
            newcomp[iso] += 1
            newcomp[mk] -= 1
            n = 0
            newelementmass = 0
            newelementprob = 1
            for loopiso, c in newcomp.items():
                newelementmass += nelementalmasses[loopiso] * c
                newelementprob *= nelementalprobabilities[loopiso]**c
                if loopiso in nonmonoisotopicelements:
                    newelementprob *= special.comb(acount-n, c, exact=True)
                    n += c
            branchmasses[branchkey][subheapkey] = newelementmass - branchmassesbyelement[branchkey][e]
            branchprobabilities[branchkey][subheapkey] = -newelementprob / branchprobabilitiesbyelement[branchkey][e]
            
            isotopesbysubheapkey[subheapkey] = iso
            subheapkeysbyelement[e].append(subheapkey)
            expansiondirections[subheapkey] = 1
            subheapkey += 1
            if iso in basecomposition:
                newcomp = elementcomp.copy()
                newcomp[iso] -= 1
                newcomp[mk] += 1
                n = 0
                newelementmass = 0
                newelementprob = 1
                for loopiso, c in newcomp.items():
                    newelementmass += nelementalmasses[loopiso] * c
                    newelementprob *= nelementalprobabilities[loopiso]**c
                    if loopiso in nonmonoisotopicelements:
                        newelementprob *= special.comb(acount-n, c, exact=True)
                        n += c
                branchmasses[branchkey][subheapkey] = newelementmass - branchmassesbyelement[branchkey][e]
                branchprobabilities[branchkey][subheapkey] = -newelementprob / branchprobabilitiesbyelement[branchkey][e]
                
                isotopesbysubheapkey[subheapkey] = iso
                subheapkeysbyelement[e].append(subheapkey)
                expansiondirections[subheapkey] = -1
                #there are two directions for this iso from basecomp
                opposingdirections[subheapkey] = subheapkey - 1
                opposingdirections[subheapkey - 1] = subheapkey
                subheapkey += 1

subheaps = {} #branchkey: [isoprob, subheapkey] heap

initialsubheap = [(i[1], i[0]) for i in branchprobabilities[branchkey].items()]

subheaps[branchkey] = initialsubheap
heapq.heapify(subheaps[branchkey])

branchcount = 1
totalprob = baseprob
populationcoverage = 0.9

massesandabundances = [[], []]
branchcompositions = {} #branchcount: isotope: count
branchcompositions[branchkey] = basecomposition

massesandabundances[0].append(basemass)
massesandabundances[1].append(baseprob)

#pop the subheap of branchkey to put its new rep in mainheap
elementprob, subheapkey = heapq.heappop(subheaps[branchkey])
iso = isotopesbysubheapkey[subheapkey]
e = iso[0]

newprob = baseprob * elementprob #the second part of the pre-normalized calculation above
newmass = basemass + branchmasses[branchkey][subheapkey]

mainheap = [[newprob, newmass, branchkey]]


#MAKING NEW SUBHEAP
direction = expansiondirections[subheapkey]
mk = monoisotopickeys[e]
newcomp = basecomposition.copy()
#these newcomp below are the modifications done previously to generate this subheap prob
newcomp[mk] -= 1 * direction
newcomp[iso] += 1 * direction

if newcomp[mk] == 0:
    #end of a positive direction line has been reached, no more isos of this monoiso will be allowed to continue in this subheap
    #i suppose this won't necessarily be a problem when considering if combinations of nonmonoisos might come into play, i think they should all rise through other position-direction subheaps if they do end up coming up rather than having to do weird acrobatics here
    isos = isotopesbyelement[e]
    branchprobabilities[branchcount] = branchprobabilities[branchkey].copy()
    #remove all isos of an element from where they need to be removed from
    for loopiso in isos:
        if not newcomp[loopiso]:
            try:
                del newcomp[loopiso]
            except KeyError:
                pass
    for shk in subheapkeysbyelement[e]:
        try:
            del branchprobabilities[branchcount][shk]
        except KeyError:
            pass
    nextsubheap = [(i[1], i[0]) for i in branchprobabilities[branchcount].items()]
    subheaps[branchcount] = nextsubheap
    heapq.heapify(subheaps[branchcount])
    
    #popping the new subheap and pushing to mainheap
    elementprob, subheapkey = heapq.heappop(subheaps[branchcount])
    iso = isotopesbysubheapkey[subheapkey]
    e = iso[0]

    newprob = baseprob * elementprob
    newmass = basemass - branchmasses[branchkey][subheapkey]
    subheaps[branchcount] = nextsubheap
    branchmasses[branchcount] = branchmasses[branchkey].copy()
    #del branchmasses[branchcount][subheapkey]
    branchcompositions[branchcount] = newcomp
    #these masses/probs will never be called upon or changed
    branchprobabilitiesbyelement[branchcount] = branchprobabilitiesbyelement[branchkey].copy()
    #del branchprobabilitiesbyelement[branchcount][e]
    branchmassesbyelement[branchcount] = branchmassesbyelement[branchkey].copy()
    del branchmassesbyelement[branchcount][e]
    #opposing directions would have already been removed here
    heapq.heappush(mainheap, [newprob, newmass, branchcount]) #updating the main heap
elif newcomp[iso] == 0:
    #the end of a negative direction line has been reached, the monoiso will not be put into the subheap, a new subheap need not be made?
    #newprob = 1
    #while newprob >= -p:
    del newcomp[iso]
    branchprobabilities[branchcount] = branchprobabilities[branchkey].copy()
    del branchprobabilities[branchcount][subheapkey]
    nextsubheap = [(i[1], i[0]) for i in branchprobabilities[branchcount].items()]
    subheaps[branchcount] = nextsubheap
    heapq.heapify(subheaps[branchcount])
    
    #popping the new subheap and pushing to mainheap
    elementprob, subheapkey = heapq.heappop(subheaps[branchcount])
    iso = isotopesbysubheapkey[subheapkey]
    e = iso[0]

    newprob = baseprob * elementprob
    newmass = basemass - branchmasses[branchkey][subheapkey]
    subheaps[branchcount] = nextsubheap
    branchmasses[branchcount] = branchmasses[branchkey].copy()
    #del branchmasses[branchcount][subheapkey]
    branchcompositions[branchcount] = newcomp
    #these masses/probs will never be called upon or changed
    branchprobabilitiesbyelement[branchcount] = branchprobabilitiesbyelement[branchkey].copy()
    #del branchprobabilitiesbyelement[branchcount][e]
    branchmassesbyelement[branchcount] = branchmassesbyelement[branchkey].copy()
    #del branchmassesbyelement[branchcount][e]
    #opposing directions would have already been removed here
    heapq.heappush(mainheap, [newprob, newmass, branchcount]) #updating the main heap
elif newcomp[iso] > 0:
    #newprob = 1
    #while newprob >= -p:
    #nc = newcomp[iso]
    #these newcomp modifications below are to generate the next step away from the currently popped branch
    elementcomp = {k: newcomp[k] for k in isotopesbyelement[e] if k in newcomp}
    elementcomp[mk] -= 1 * direction
    elementcomp[iso] += 1 * direction
    n = 0
    newelementmass = 0
    newelementprob = 1
    for loopiso, c in elementcomp.items():
        newelementmass += nelementalmasses[loopiso] * c
        newelementprob *= nelementalprobabilities[loopiso]**c
        if loopiso in nonmonoisotopicelements:
            newelementprob *= special.comb(acount-n, c, exact=True)
            n += c
    newprob = newelementprob / elementprob
    branchcompositions[branchcount] = newcomp
    branchprobabilities[branchcount] = branchprobabilities[branchkey].copy()
    branchprobabilities[branchcount][subheapkey] = newprob
    branchmasses[branchcount] = branchmasses[branchkey].copy()
    branchmasses[branchcount][subheapkey] = newelementmass - branchmasses[branchkey][subheapkey]
    
    if subheapkey in opposingdirections:
        opposer = opposingdirections[subheapkey]
        try:
            del branchprobabilities[branchcount][opposer]
        except KeyError:
            #it isn't in there
            pass
    
    nextsubheap = [(i[1], i[0]) for i in branchprobabilities[branchcount].items()]
    subheaps[branchcount] = nextsubheap
    heapq.heapify(subheaps[branchcount])
    
    #popping the new subheap and pushing to mainheap
    elementprob, subheapkey = heapq.heappop(subheaps[branchcount])
    iso = isotopesbysubheapkey[subheapkey]
    e = iso[0]

    newprob = baseprob * elementprob
    newmass = basemass + branchmasses[branchkey][subheapkey]
    branchprobabilitiesbyelement[branchcount] = branchprobabilitiesbyelement[branchkey].copy()
    branchprobabilitiesbyelement[branchcount][e] = newelementprob
    branchmassesbyelement[branchcount] = branchmassesbyelement[branchkey].copy()
    branchmassesbyelement[branchcount][e] = newelementmass
    
    heapq.heappush(mainheap, [newprob, newmass, branchcount]) #updating the main heap
#else:
    #negative direction is finished, no need to continue it


#the masses and abundance/probabilities produced in the loop below are INCORRECT and I don't know why. The two abundance/probabilities initiated above ARE correct, they show up as the first two in massesandabundances.
p, m, branchkey = heapq.heappop(mainheap)
while totalprob < populationcoverage:
    branchcount += 1
    #e = iso[0]
    bcomp = branchcompositions[branchkey]
    massesandabundances[0].append(m)
    massesandabundances[1].append(-p)
    #if the subheap is done, you don't need to pop it, its copy-spawned subheap would have been made in the previous iteration
    if subheaps[branchkey]:
        #pop the subheap of branchkey to put its new rep in mainheap
        #while newprob >= -p: #i don't need this???
        elementprob, subheapkey = heapq.heappop(subheaps[branchkey])
        iso = isotopesbysubheapkey[subheapkey]
        e = iso[0]
        mk = monoisotopickeys[e]
        
        newprob = p * elementprob
        #^i don't think a newprob greater than the current p will arise here if i make a while loop to prevent it in the new subheap generation below, so i probably wouldn't need the while-loops to be here IF they were needed
        newmass = m + branchmasses[branchkey][subheapkey]
        heapq.heappush(mainheap, [-newprob, newmass, branchkey]) #updating the main heap
        
        #making new subheap
        direction = expansiondirections[subheapkey]
        newcomp = bcomp.copy()
        #these newcomp below are the modifications done previously to generate this subheap prob
        newcomp[iso] += 1 * direction
        newcomp[mk] -= 1 * direction
        if newcomp[mk] == 0:
            #end of a positive direction line has been reached, no more isos of this monoiso will be allowed to continue in this subheap
            #i suppose this won't necessarily be a problem when considering if combinations of nonmonoisos might come into play, i think they should all rise through other position-direction subheaps if they do end up coming up rather than having to do weird acrobatics here
            isos = isotopesbyelement[e]
            branchprobabilities[branchcount] = branchprobabilities[branchkey].copy()
            #remove all isos of an element from where they need to be removed from
            for loopiso in isos:
                if not newcomp[loopiso]:
                    try:
                        del newcomp[loopiso]
                    except KeyError:
                        pass
            for shk in subheapkeysbyelement[e]:
                try:
                    del branchprobabilities[branchcount][shk]
                except KeyError:
                    pass
            nextsubheap = [(i[1], i[0]) for i in branchprobabilities[branchcount].items()]
            subheaps[branchcount] = nextsubheap
            heapq.heapify(subheaps[branchcount])
            
            #popping the new subheap and pushing to mainheap
            elementprob, subheapkey = heapq.heappop(subheaps[branchcount])
            iso = isotopesbysubheapkey[subheapkey]
            e = iso[0]

            newprob = p * elementprob
            newmass = m - branchmasses[branchkey][subheapkey]
            subheaps[branchcount] = nextsubheap
            branchmasses[branchcount] = branchmasses[branchkey].copy()
            #del branchmasses[branchcount][subheapkey]
            branchcompositions[branchcount] = newcomp
            #these masses/probs will never be called upon or changed
            branchprobabilitiesbyelement[branchcount] = branchprobabilitiesbyelement[branchkey].copy()
            #del branchprobabilitiesbyelement[branchcount][e]
            branchmassesbyelement[branchcount] = branchmassesbyelement[branchkey].copy()
            del branchmassesbyelement[branchcount][e]
            #opposing directions would have already been removed here
            heapq.heappush(mainheap, [newprob, newmass, branchcount]) #updating the main heap
        elif newcomp[iso] == 0:
            #the end of a negative direction line has been reached, the monoiso will not be put into the subheap, a new subheap need not be made?
            #newprob = 1
            #while newprob >= -p: #i don't need this???
            del newcomp[iso]
            branchprobabilities[branchcount] = branchprobabilities[branchkey].copy()
            del branchprobabilities[branchcount][subheapkey]
            nextsubheap = [(i[1], i[0]) for i in branchprobabilities[branchcount].items()]
            subheaps[branchcount] = nextsubheap
            heapq.heapify(subheaps[branchcount])
            
            #popping the new subheap and pushing to mainheap
            elementprob, subheapkey = heapq.heappop(subheaps[branchcount])
            iso = isotopesbysubheapkey[subheapkey]
            e = iso[0]

            newprob = p * elementprob
            newmass = m - branchmasses[branchkey][subheapkey]
            subheaps[branchcount] = nextsubheap
            branchmasses[branchcount] = branchmasses[branchkey].copy()
            #del branchmasses[branchcount][subheapkey]
            branchcompositions[branchcount] = newcomp
            #these masses/probs will never be called upon or changed
            branchprobabilitiesbyelement[branchcount] = branchprobabilitiesbyelement[branchkey].copy()
            #del branchprobabilitiesbyelement[branchcount][e]
            branchmassesbyelement[branchcount] = branchmassesbyelement[branchkey].copy()
            #del branchmassesbyelement[branchcount][e]
            #opposing directions would have already been removed here
            heapq.heappush(mainheap, [newprob, newmass, branchcount]) #updating the main heap
        elif newcomp[iso] > 0:
            #newprob = 1
            #while newprob >= -p: #i don't need this???
            #these newcomp modifications below are to generate the next step away from the currently popped branch
            elementcomp = {k: newcomp[k] for k in isotopesbyelement[e] if k in newcomp}
            elementcomp[mk] -= 1 * direction
            elementcomp[iso] += 1 * direction
            n = 0
            newelementmass = 0
            newelementprob = 1
            for loopiso, c in elementcomp.items():
                newelementmass += nelementalmasses[loopiso] * c
                newelementprob *= nelementalprobabilities[loopiso]**c
                if loopiso in nonmonoisotopicelements:
                    newelementprob *= special.comb(acount-n, c, exact=True)
                    n += c
            try:
                newprob = newelementprob / elementprob
                branchcompositions[branchcount] = newcomp
                branchprobabilities[branchcount] = branchprobabilities[branchkey].copy()
                branchprobabilities[branchcount][subheapkey] = newprob
                branchmasses[branchcount] = branchmasses[branchkey].copy()
                branchmasses[branchcount][subheapkey] = newelementmass - branchmasses[branchkey][subheapkey]
                
                if subheapkey in opposingdirections:
                    opposer = opposingdirections[subheapkey]
                    try:
                        del branchprobabilities[branchcount][opposer]
                    except KeyError:
                        #it isn't in there
                        pass
                
                nextsubheap = [(i[1], i[0]) for i in branchprobabilities[branchcount].items()]
                subheaps[branchcount] = nextsubheap
                heapq.heapify(subheaps[branchcount])
                #popping the new subheap and pushing to mainheap
                elementprob, subheapkey = heapq.heappop(subheaps[branchcount])
                iso = isotopesbysubheapkey[subheapkey]
                e = iso[0]

                newprob = p * elementprob
                newmass = m + branchmasses[branchkey][subheapkey]
                branchprobabilitiesbyelement[branchcount] = branchprobabilitiesbyelement[branchkey].copy()
                branchprobabilitiesbyelement[branchcount][e] = newelementprob
                branchmassesbyelement[branchcount] = branchmassesbyelement[branchkey].copy()
                branchmassesbyelement[branchcount][e] = newelementmass
                
                heapq.heappush(mainheap, [-newprob, newmass, branchcount]) #updating the main heap
            except ZeroDivisionError:
                #partial binomial is too small!
                pass
            
        #else:
            #negative direction is finished, no need to continue it
    #prepare for next loop
    try:
        p, m, branchkey = heapq.heappop(mainheap)
        totalprob -= p
    except IndexError:
        break


#these were notes I wrote to myself upon realizing I'm a failure, feel free to use the ideas if you find them helpful:

#so this concept has failed
#you need to use the older-style cutoff otherwise you go deeper into more and more useless territory, the new-style cutoff isn't actually realistic, and the old happened to hit the nail on the head

#i'm still thinking something non-recursive that can hold onto existing compositions that iteratively generates new subheaps 

#mainheap pops for highest iso -> new subheap formed with this iso
#all probabilities are made for that subheap
#you then exhaustively extend into every possible subheap from there, while returning subformulastring in a set for lookup
#anything in the subheap that's below minimumabundance doesn't need to be pushed into the exhausive search of any subheaps coming after that subheap. and obviously isn't looked into for a new subheap as well
#^i don't think this makes sense

#the subheaps aren't compatible due to underflow, but they could still be kept around for ease of calculation as long as your minimumabundance doesn't give it any problems, I think the cutoff issue is obviously what caused the underflow.
#i think the subheaps could also be replaced by a elementcompositionsbybranchkey dict where you're indexing the compositions for a lighter calculation in order to modify finalized abundances from one isotopomer to the next
#^in this take, the isotopomers no longer need to be output in descending order or abundance, but I might need a list full of generators if i'm not using recursion here.


#~~~~
#base -> make initial subheap
#pop initial subheap 100% of the way -> round 1
#from round 1, make all the next potential subeaps, pop all of them all the way, anything under minimumabundance is left alone
#make next rounds all the same way, subheaps can be generated from 
