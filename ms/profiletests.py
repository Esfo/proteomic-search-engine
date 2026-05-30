from collections import defaultdict, Counter
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
import profile
import heapq
import math
import os

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


@profile
def new_dist_gen(atomiccomposition, minimumabundance):
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
    baseformula = ''.join((f'{k}({basecomposition[k]})' for k in sorted(basecomposition) if basecomposition[k] > 0))
    
    #below calculates the mass and abundance of the most abundant isotopomer, and organizes the binomials its made from
    baseprob = 1
    basemass = 0
    baseoffsets = defaultdict(int)
    branchprobabilitiesbyelement = defaultdict(lambda: defaultdict(lambda: 1)) #branchkey: e: prob
    branchmassesbyelement = defaultdict(lambda: defaultdict(int)) #branchkey: e: mass
    branchcompositionbyelement = defaultdict(lambda: defaultdict(lambda: Counter())) #branchkey: e: {composition dict}
    for iso, c in basecomposition.items():
        e = iso[0]
        prob = nelementalprobabilities[iso]**c
        mass = nelementalmasses[iso] * c
        baseprob *= prob
        basemass += mass
        branchprobabilitiesbyelement[branchkey][e] *= prob
        branchmassesbyelement[branchkey][e] += mass
        branchcompositionbyelement[branchkey][e][iso] += c
        if iso in nonmonoisotopicelements:
            combfactor = special.comb(atomiccomposition[e] - baseoffsets[e], c, exact=True)
            baseprob *= combfactor
            branchprobabilitiesbyelement[branchkey][e] *= combfactor
            baseoffsets[e] += c
    expansiondirections = {} #branchkey: 1 or -1, a negative direction is only given to nonmonisotopic elements that are in basecomposition generated above
    isotopesbysubheapkey = {} #subheapkey: iso
    subheapkeysbyelement = defaultdict(list) #e: [subheapkeys]
    opposingdirections = {} #subheapkey: subheapkey of the same element moving in the other direction, I don't let these coexist within the same subheap
    
    #i think i can separate supheapkeys from initial probability calculations
    subheapkey = 0
    for e, isos in nonmonoisotopicgroups.items():
        if e in atomiccomposition:
            for iso in isos:
                #positive direction
                isotopesbysubheapkey[subheapkey] = iso
                subheapkeysbyelement[e].append(subheapkey)
                expansiondirections[subheapkey] = 1
                subheapkey += 1
                if iso in basecomposition:
                    #negative direction
                    isotopesbysubheapkey[subheapkey] = iso
                    subheapkeysbyelement[e].append(subheapkey)
                    expansiondirections[subheapkey] = -1
                    #there are two directions for this iso from basecomp
                    opposingdirections[subheapkey] = subheapkey - 1
                    opposingdirections[subheapkey - 1] = subheapkey
                    subheapkey += 1
    
    finalprobabilities = {} #branchkey: abundance
    finalformulas = {} #branchkey: subformula
    finalmasses = {} #branchkey: mass
    
    finalprobabilities[branchkey] = baseprob
    finalformulas[branchkey] = baseformula
    finalmasses[branchkey] = basemass
    
    branchcount = 1
    currentbranchkeys = [] #list of all currently unexplored branchkeys
    priorbranch = {} #branchkey: branchkey of branch that generated this branch
    branchopposers = defaultdict(set) #branchkey: set of non-compatible subheapkeys
    branchprobabilities = defaultdict(dict) #branchkey: subheapkey: combined element prob
    branchmasses = defaultdict(dict) #branchkey: subheapkey: combined element mass
    branchcompositions = defaultdict(dict) #branchcount: isotope: count
    branchsubheapkeys = {} #branchkey: tailored version of isotopesbysubheapkey
    subformulasets = set()
    
    branchsubheapkeys[branchkey] = isotopesbysubheapkey.copy()
    
    for subheapkey, iso in isotopesbysubheapkey.items():
        e = iso[0]
        acount = atomiccomposition[e]
        mk = monoisotopickeys[e]
        direction = expansiondirections[subheapkey]
        newelementcomp = branchcompositionbyelement[branchkey][e].copy()
        newelementcomp[iso] += direction
        newelementcomp[mk] -= direction
        n = 0
        newelementmass = 0
        newelementprob = 1
        for loopiso, c in newelementcomp.items():
            newelementmass += nelementalmasses[loopiso] * c
            newelementprob *= nelementalprobabilities[loopiso]**c
            if loopiso in nonmonoisotopicelements:
                newelementprob *= special.comb(acount-n, c, exact=True)
                n += c
        newprob = baseprob / branchprobabilitiesbyelement[branchkey][e] * newelementprob
        if newprob >= minimumabundance:
            newmass = basemass - branchmassesbyelement[branchkey][e] + newelementmass
            #modify basecomp via elementcomp
            newbasecomp = basecomposition.copy()
            newbasecomp[iso] += direction
            newbasecomp[mk] -= direction
            subformula = ''.join((f'{k}({newbasecomp[k]})' for k in sorted(newbasecomp) if newbasecomp[k] > 0))
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
                #negative direction ended, remove that subheapkey in the future branches
                del branchsubheapkeys[branchkey][subheapkey]
            if newbasecomp[mk] == 0:
                #end of a positive direction, remove that subheapkey in the future branches
                del branchsubheapkeys[branchkey][subheapkey]
            if subheapkey in opposingdirections:
                #remove opposing direction from subheapkeys
                branchopposers[branchcount].add(opposingdirections[subheapkey])
            currentbranchkeys.append(branchcount)
            branchcount += 1
    
    while currentbranchkeys:
        nextbranchkeys = []
        for branchkey in currentbranchkeys:
            prior = priorbranch[branchkey]
            for subheapkey, iso in branchsubheapkeys[prior].items():
                if subheapkey not in branchopposers[branchkey]:
                    e = iso[0]
                    acount = atomiccomposition[e]
                    mk = monoisotopickeys[e]
                    direction = expansiondirections[subheapkey]
                    newbasecomp = branchcompositions[branchkey].copy()
                    newbasecomp[iso] += direction
                    newbasecomp[mk] -= direction
                    subformula = ''.join((f'{k}({newbasecomp[k]})' for k in sorted(newbasecomp) if newbasecomp[k] > 0))
                    if subformula not in subformulasets:
                        subformulasets.add(subformula)
                        newelementcomp = branchcompositionbyelement[branchkey][e].copy()
                        newelementcomp[iso] += direction
                        newelementcomp[mk] -= direction
                        n = 0
                        newelementmass = 0
                        newelementprob = 1
                        for loopiso, c in newelementcomp.items():
                            newelementmass += nelementalmasses[loopiso] * c
                            newelementprob *= nelementalprobabilities[loopiso]**c
                            if loopiso in nonmonoisotopicelements:
                                newelementprob *= special.comb(acount-n, c, exact=True)
                                n += c
                        newprob = finalprobabilities[branchkey] / branchprobabilitiesbyelement[branchkey][e] * newelementprob
                        if newprob >= minimumabundance:
                            branchsubheapkeys[branchkey] = branchsubheapkeys[prior].copy()
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
                                #negative direction ended, remove that subheapkey in the future branches
                                del branchsubheapkeys[branchkey][subheapkey]
                            elif newbasecomp[mk] == 0:
                                #end of a positive direction, remove that subheapkey in the future branches
                                del branchsubheapkeys[branchkey][subheapkey]
                            if subheapkey in opposingdirections:
                                #remove opposing direction from subheapkeys
                                branchopposers[branchcount].add(opposingdirections[subheapkey])
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
    formulas = formulas[massesandabundances[0].argsort()].tolist()
    massesandabundances = massesandabundances[:,massesandabundances[0].argsort()]
    return formulas, massesandabundances

if '__name__' == '__main__':
    minimumabundance = 0.01

    seq = 'GTSWGLPASKTITTMIDGPQDLRVVAVTPTTLELGWLRPQAEVDR'

    atomiccomposition = Counter()
    for aa in seq:
        atomiccomposition += aminoacidcomposition[aa]
    #no OH loss on last residue, no H lost on first residue
    atomiccomposition['H'] += 2
    atomiccomposition['O'] += 1

    nsubformulas, nabundances = new_dist_gen(atomiccomposition, minimumabundance)
