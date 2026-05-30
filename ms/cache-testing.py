from collections import defaultdict, Counter
import itertools
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from functools import partial
import multiprocessing as mp
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

def descending_partial_products(dividingthreshold, mainformula, elementalorganizer):
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
        
        finalabundances[formula] = [mass, prob]
        if prob < cutoff:
            break
        
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
    subformulas = subformulas[massesandabundances[0].argsort()].tolist()
    massesandabundances = massesandabundances[:,massesandabundances[0].argsort()]
    return mainformula, subformulas, massesandabundances

def elemental_binomial_walk(dividingthreshold, mainformula, atomiccomposition):
    elementalorganizer = defaultdict(list)
    mainheap = []
    vectorsets = defaultdict(set) #element: set of used vectors
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
    return descending_partial_products(dividingthreshold, mainformula, elementalorganizer)

def individual_element_binomial_walk(dividingthreshold, e, acount):
    #elementalorganizer = defaultdict(list)
    elementlist = []
    mainheap = []
    vectorsets = defaultdict(set) #element: set of used vectors
    #for e, acount in atomiccomposition.items():
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
        elementlist.append([-1, maxprob, m, e, nv])
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
        preheap.append([baseprob, acount * nelementalmasses[mk], e, nvector.copy()])
        greater = True
        lastprob = baseprob
        iso = nonmonoisotopicgroups[e][0]
        while greater:
            greater = False
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
        elementlist.append([-1, maxprob, m, e, nv])
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
    elementlist.append([r, p, m, e, v])
    if len(isotopesbyelement[e]) > 2:
        while p > cutoff:
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
            r, p, m, e, v = heapq.heappop(mainheap)
            elementlist.append([r, p, m, e, v])
    else:
        iso = nonmonoisotopicgroups[e][0]
        while p > cutoff:
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
            elementlist.append([r, p, m, e, v])
    return elementlist

#def individual_element_binomial_walk2(dividingthreshold, elementstring):
#    e = elementstring[0]
#    acount = int(elementstring[1:])
#    #elementalorganizer = defaultdict(list)
#    elementlist = []
#    mainheap = []
#    vectorsets = defaultdict(set) #element: set of used vectors
#    #for e, acount in atomiccomposition.items():
#    mk = monoisotopickeys[e]
#    nvector = elementvectors[e].copy()
#    nvector[nvectorpositions[e][mk]] += acount
#    if len(isotopesbyelement[e]) > 2:
#        baseprob = nelementalprobabilities[mk] ** acount
#        preheap = []
#        preheap.append([baseprob, acount * nelementalmasses[mk], e, nvector.copy()])
#        greater = True
#        lastprob = baseprob
#        while greater:
#            greater = False
#            for iso in nonmonoisotopicgroups[e]:
#                newelementvector = nvector.copy()
#                newelementvector[nvectorpositions[e][mk]] -= 1
#                if newelementvector[nvectorpositions[e][mk]] > -1:
#                    newelementvector[nvectorpositions[e][iso]] += 1
#                    vectorsets[e].add(tuple(newelementvector))
#                    pn = 0
#                    newelementmass = 0
#                    newelementprob = 1
#                    for n, c in enumerate(newelementvector):
#                        loopiso = nelementpositions[e][n]
#                        newelementmass += nelementalmasses[loopiso] * c
#                        newelementprob *= nelementalprobabilities[loopiso]**c
#                        if loopiso in nonmonoisotopicelements:
#                            newelementprob *= math.comb(acount-pn, c)
#                            pn += c
#                    preheap.append([newelementprob, newelementmass, e, newelementvector.copy()])
#                    if newelementprob > lastprob:
#                        lastprob = newelementprob
#                        greater = True
#        preheap = sorted(preheap)
#        maxiso = preheap[-1]
#        maxprob, m, e, nv = maxiso
#        elementlist.append([-1, maxprob, m, e, nv])
#        maxprob *= -1
#        preheap = preheap[:-1]
#        for h in preheap:
#            r = h[0] / maxprob
#            h.insert(0, r)
#            heapq.heappush(mainheap, h)
#        for iso in nonmonoisotopicgroups[e]:
#            v = nv.copy()
#            v[nvectorpositions[e][mk]] -= 1
#            if v[nvectorpositions[e][mk]] > -1:
#                v[nvectorpositions[e][iso]] += 1
#                tuplevec = tuple(v)
#                if tuplevec not in vectorsets[e]:
#                    vectorsets[e].add(tuplevec)
#                    pn = 0
#                    newelementmass = 0
#                    newelementprob = 1
#                    for n, c in enumerate(v):
#                        loopiso = nelementpositions[e][n]
#                        newelementmass += nelementalmasses[loopiso] * c
#                        newelementprob *= nelementalprobabilities[loopiso]**c
#                        if loopiso in nonmonoisotopicelements:
#                            newelementprob *= math.comb(acount-pn, c)
#                            pn += c
#                    heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, v.copy()])
#    else:
#        preheap = []
#        baseprob = nelementalprobabilities[mk] ** acount
#        preheap.append([baseprob, acount * nelementalmasses[mk], e, nvector.copy()])
#        greater = True
#        lastprob = baseprob
#        while greater:
#            greater = False
#            iso = nonmonoisotopicgroups[e][0]
#            nvector[nvectorpositions[e][mk]] -= 1
#            if nvector[nvectorpositions[e][mk]] > -1:
#                nvector[nvectorpositions[e][iso]] += 1
#                vectorsets[e].add(tuple(nvector))
#                pn = 0
#                newelementmass = 0
#                newelementprob = 1
#                for n, c in enumerate(nvector):
#                    loopiso = nelementpositions[e][n]
#                    newelementmass += nelementalmasses[loopiso] * c
#                    newelementprob *= nelementalprobabilities[loopiso]**c
#                    if loopiso in nonmonoisotopicelements:
#                        newelementprob *= math.comb(acount-pn, c)
#                        pn += c
#                preheap.append([newelementprob, newelementmass, e, nvector.copy()])
#                if newelementprob > lastprob:
#                    lastprob = newelementprob
#                    greater = True
#        preheap = sorted(preheap)
#        maxiso = preheap[-1]
#        maxprob, m, e, nv = maxiso
#        elementlist.append([-1, maxprob, m, e, nv])
#        maxprob *= -1
#        preheap = preheap[:-1]
#        for h in preheap:
#            r = h[0] / maxprob
#            h.insert(0, r)
#            heapq.heappush(mainheap, h)
#        v = nv.copy()
#        v[nvectorpositions[e][mk]] -= 1
#        if v[nvectorpositions[e][mk]] > -1:
#            v[nvectorpositions[e][iso]] += 1
#            tuplevec = tuple(v)
#            if tuplevec not in vectorsets[e]:
#                vectorsets[e].add(tuplevec)
#                pn = 0
#                newelementmass = 0
#                newelementprob = 1
#                for n, c in enumerate(v):
#                    loopiso = nelementpositions[e][n]
#                    newelementmass += nelementalmasses[loopiso] * c
#                    newelementprob *= nelementalprobabilities[loopiso]**c
#                    if loopiso in nonmonoisotopicelements:
#                        newelementprob *= math.comb(acount-pn, c)
#                        pn += c
#                heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, v.copy()])
#    
#    cutoff = -maxprob * dividingthreshold
#
#    r, p, m, e, v = heapq.heappop(mainheap)
#    elementlist.append([r, p, m, e, v])
#    while p > cutoff:
#        if len(isotopesbyelement[e]) > 2:
#            for iso in nonmonoisotopicgroups[e]:
#                newelementvector = v.copy()
#                newelementvector[nvectorpositions[e][mk]] -= 1
#                if newelementvector[nvectorpositions[e][mk]] > 0:
#                    newelementvector[nvectorpositions[e][iso]] += 1
#                    tuplevec = tuple(newelementvector)
#                    if tuplevec not in vectorsets[e]:
#                        vectorsets[e].add(tuplevec)
#                        pn = 0
#                        newelementmass = 0
#                        newelementprob = 1
#                        for n, c in enumerate(newelementvector):
#                            loopiso = nelementpositions[e][n]
#                            newelementmass += nelementalmasses[loopiso] * c
#                            newelementprob *= nelementalprobabilities[loopiso]**c
#                            if loopiso in nonmonoisotopicelements:
#                                newelementprob *= math.comb(acount-pn, c)
#                                pn += c
#                        heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, newelementvector.copy()])
#            r, p, m, e, v = heapq.heappop(mainheap)
#            elementlist.append([r, p, m, e, v])
#        else:
#            iso = nonmonoisotopicgroups[e][0]
#            nvector = v.copy()
#            nvector[nvectorpositions[e][mk]] -= 1
#            if nvector[nvectorpositions[e][mk]] > 0:
#                nvector[nvectorpositions[e][iso]] += 1
#                tuplevec = tuple(nvector)
#                if tuplevec not in vectorsets[e]:
#                    vectorsets[e].add(tuplevec)
#                    pn = 0
#                    newelementmass = 0
#                    newelementprob = 1
#                    for n, c in enumerate(nvector):
#                        loopiso = nelementpositions[e][n]
#                        newelementmass += nelementalmasses[loopiso] * c
#                        newelementprob *= nelementalprobabilities[loopiso]**c
#                        if loopiso in nonmonoisotopicelements:
#                            newelementprob *= math.comb(acount-pn, c)
#                            pn += c
#                    heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, nvector.copy()])
#        r, p, m, e, v = heapq.heappop(mainheap)
#        elementlist.append([r, p, m, e, v])
#    return elementlist

def distribution_generation(dividingthreshold, elementalcache, mainformula, atomiccomposition):
    elementalorganizer = {} #element: [[preheaps]]
    for e, acount in atomiccomposition.items():
        elementstring = e + str(acount)
        #try/except if faster than an if/else, so i might as well
        try:
            elementlist = elementalcache[elementstring]
        except KeyError: #not in cache
            elementlist = individual_element_binomial_walk(dividingthreshold, e, acount)
            elementalcache[elementstring] = elementlist
        elementalorganizer[e] = elementlist.copy()
    mainformula, subformulas, massesandabundances = descending_partial_products(dividingthreshold, mainformula, elementalorganizer)
    return mainformula, subformulas, massesandabundances

#def fast_nested_copy(elementlist):
#    t = []
#    for h in elementlist:
#        t.append(h.copy())
#    return t

#def heap_copy(k, v):
#    return k, {i[0]: fast_nested_copy(elementpreheaps[i]) for i in v}

dividingthreshold = 0.4
subisotopomericdepth = 0.8

elementset = set()
elementstringsbyformula = {} #mainformula: elementstrings
atomiccompositions = {} #mainformula: element: count
while len(atomiccompositions) < 500:
    seq = ''.join((random.choices(list(aminoacidcomposition.keys()), k=np.random.randint(6,50))))
    atomiccomposition = Counter()
    for aa in seq:
        atomiccomposition += aminoacidcomposition[aa]
    atomiccomposition['H'] += 2
    atomiccomposition['O'] += 1
    
    elementstrings = [''.join((k, str(v))) for k, v in atomiccomposition.items()]
    mainformula = ''.join(elementstrings)
    atomiccompositions[mainformula] = atomiccomposition
    elementstringsbyformula[mainformula] = elementstrings
    elementset.update(elementstrings)


#nt = time()
#linearmainformulas, linearsubformulas, linearmassesandabundances = zip(*[elemental_binomial_walk(dividingthreshold, mainformula, atomiccomposition) for mainformula, atomiccomposition in atomiccompositions.items()])
#print('linear', time() - nt)

#elementalcache = {} #elementstring: [[preheaps]]
#nt = time()
#joinedmainformulas, joinedsubformulas, joinedmassesandabundances = zip(*[distribution_generation(dividingthreshold, mainformula, atomiccomposition) for mainformula, atomiccomposition in atomiccompositions.items()])
#print('cached', time() - nt)

#elementalcache = mp.Manager().dict() #elementstring: [[preheaps]]
elementalcache = {} #this was way better for caching, doesn't matter if there's any collisions because they'll just write the same value anyways
distribution_generation_partial = partial(distribution_generation, dividingthreshold, elementalcache)
nt = time()
with mp.Pool(8) as pool:
    joinedmainformulas, joinedsubformulas, joinedmassesandabundances = zip(*pool.starmap(distribution_generation_partial, atomiccompositions.items()))
print('cache multiprocessing', time() - nt)

#linearabundances = dict(zip(linearmainformulas, linearmassesandabundances))
#linearabundanceformulas = dict(zip(linearmainformulas, linearsubformulas))

joinedabundances = dict(zip(joinedmainformulas, joinedmassesandabundances))
joinedabundanceformulas = dict(zip(joinedmainformulas, joinedsubformulas))

#nt = time()
#
#elementpreheaps = {} #elementstring: [[preheap under current dividingthreshold]]
#for es in list(elementset):
#    elementpreheaps[es] = individual_element_binomial_walk2(dividingthreshold, es)
#
#preheapsbyformula = {k:{i[0]: fast_nested_copy(elementpreheaps[i]) for i in v} for k, v in elementstringsbyformula.items()}
#
#preheapabundances = {} #mainformula: [[m], [a]]
#preheapsubformulas = {} #mainformula: subformulas
#for mainformula, elementalorganizer in preheapsbyformula.items():
#    mainformula, subformulas, massesandabundances = descending_partial_products(dividingthreshold, mainformula, elementalorganizer)
#    preheapabundances[mainformula] = massesandabundances
#    preheapsubformulas[mainformula] = subformulas
#
#print('splitup linear', time() - nt)
#
#walk_partial = partial(individual_element_binomial_walk2, dividingthreshold)
#partial_descending_partial_products = partial(descending_partial_products, dividingthreshold)
#nt = time()
#
#elementpreheaps = {} #elementstring: [[preheap under current dividingthreshold]]
#for es in list(elementset):
#    elementpreheaps[es] = individual_element_binomial_walk2(dividingthreshold, es)
##with mp.Pool(8) as pool:
##    elementpreheaps = dict(pool.map(walk_partial, list(elementset)))
##print(time() - nt) #linear was 10x quicker, never enough individual elementstrings to need to multiprocess
#
##preheapsbyformula = {k:{i[0]: fast_nested_copy(elementpreheaps[i]) for i in v} for k, v in elementstringsbyformula.items()}
#with mp.Pool(8) as pool:
#    preheapsbyformula = dict(pool.starmap(heap_copy, elementstringsbyformula.items()))
##a little faster than linear
#
##preheapabundances = {} #mainformula: [[m], [a]]
##preheapsubformulas = {} #mainformula: subformulas
##for mainformula, elementalorganizer in preheapsbyformula.items():
##    mainformula, subformulas, massesandabundances = descending_partial_products(dividingthreshold, mainformula, elementalorganizer)
##    preheapabundances[mainformula] = massesandabundances
##    preheapsubformulas[mainformula] = subformulas
#
#with mp.Pool(8) as pool:
#    mainformulalist, subformulalist, massesandabundanceslist = zip(*pool.starmap(partial_descending_partial_products, preheapsbyformula.items()))
#preheapabundances = dict(zip(mainformulalist, massesandabundanceslist))
#preheapsubformulas = dict(zip(mainformulalist, subformulalist))
#
#print('splitup multi', time() - nt)

#screwups = []
#for k, v in joinedabundances.items():
#    if (v != linearabundances[k]).all():
#        screwups.append(k)
#print(len(screwups))

sumabundances = {} #formula: [[weighted mean masses], [sum abundances]]
condensationcoordinates = {} #formula: [coords]
for formula, subformulas in joinedabundanceformulas.items():
    massgroups = defaultdict(list) #massnumber: [masses]
    intensitygroups = defaultdict(list) #massnumber: [abundances]
    masses, intensities = joinedabundances[formula]
    for n, s in enumerate(subformulas):
        s = s.decode()
        massnumber = 0
        for ss in s.split(')')[:-1]:
            i1, i2 = map(int, ss[1:].split('('))
            massnumber += i1*i2
        massgroups[massnumber].append(masses[n])
        intensitygroups[massnumber].append(intensities[n])
    meansofmasses = []
    sumsofabundances = []
    subisodepthindices = [] #index as sum abundances, sublist filled with those above subisotopomericdepth
    condensationindices = []
    for mn, m in massgroups.items():
        condensationindices.append(len(m))
        a = intensitygroups[mn]
        totalabundance = sum(a)
        weightedmass = 0
        cumulativeabundance = 0
        subinds = []
        for n, (sm, sa) in enumerate(zip(m, a)):
            weightedmass += (sm * sa) / totalabundance
            cumulativeabundance += sa
            if cumulativeabundance / totalabundance >= subisotopomericdepth:
                #requires intensity being sorted by abundance
                subinds.append(n)
        meansofmasses.append(weightedmass)
        sumsofabundances.append(totalabundance)
        subisodepthindices.append(tuple(subinds))
    sumabundancedistribution = np.array([meansofmasses, sumsofabundances])
    sumabundances[formula] = sumabundancedistribution
    condensationcoordinates[formula] = condensationindices
    subisodiffs = set(i-masses[n+1] for n, i in enumerate(masses.tolist()[:-1]))
    subisodiffs = [i for i in subisodiffs if i < 0.5]
    abundancedelta = [intensities[n+1]/i for n, i in enumerate(intensities.tolist()[:-1])]
    intensitypercdiff = [abs(intensities[n+1]-i) / (intensities[n+1]+i) / 2 for n, i in enumerate(intensities.tolist()[:-1])]
    #manage a newinclimit and steplimit
    #steplimit -> highest increase
    #newinclimit -> highest increase after a decrease
    #decreaselimit??? lowest decrease??
    #i've previously focussed on these above, but maybe it would be worth my while to extend this thought process into the entire distribution. and what i mean by that is that when a step of 0.5 occurs, its never had anything below a 0.3 increase below it, for example, maybe this would make more sense?

    #old func
    maxmass = masses[intensities.argmax()]
    csteps = masses - masses.min()
    maxstep = masses.size
    steprange = proton * np.arange(maxstep)
    stepclasses = np.abs(csteps - steprange[:,None]).argmin(axis=0)
    cinds, counts = np.unique(stepclasses, return_counts=True)
    csplit = counts.cumsum().tolist()
    subisodiffs = []
    majordiffs = []
    #i'd prefer sum to be the baseline and max to be the subiso exception but this can be generated all the same using the information this already makes, if needed
    stepsplit = []
    summedintensities = []
    majorintensities = []
    maxmasses = []
    meanmasses = []
    subisodepthindices = []
    ci = 0
    newincmax = 0
    fulldiffmax = 0
    if counts.max() > 1: #subisos exist
        condensationcoordinates2 = []
        for cs in csplit:
            #consider seeing if getting rid of max dists is worth it after ms2 matches comme out
            condensationcoordinates2.append(cs-ci)
            splitmasses = masses[ci:cs]
            splitints = intensities[ci:cs]
            isorted = np.sort(splitints)[::-1]
            inorm = isorted.cumsum() / isorted.sum()
            depthcut = np.where(inorm >= subisotopomericdepth)[0][0]
            intensitypicks = isorted[:depthcut+1]
            topinds = np.where(splitints == intensitypicks[:,None])[1].tolist()
            meanmass = (splitmasses * splitints).sum() / splitints.sum()
            meanmasses.append(meanmass)
            maxmasses.append(splitmasses.max())
            stepsplit.append(splitmasses.tolist())
            summedintensities.append(splitints.sum())
            majorintensities.append(splitints.max())
            subisodepthindices.append(topinds)
            ci = cs
        maxabundancedistribution = np.array([maxmasses, majorintensities])
        summedintensities = np.array(summedintensities)
        majorintensities = np.array(majorintensities)
        msums = majorintensities[:-1] + majorintensities[1:]
        majordiffs.extend((np.diff(majorintensities) / msums / 2).tolist())
        decreasing = False
        for ii in range(len(majorintensities)-1):
            i1 = majorintensities[ii]
            i2 = majorintensities[ii+1]
            if i2 > i1:
                if decreasing:
                    newinc = abs(i2 - i1) / (i2 + i1) / 2
                    if newinc > newincmax:
                        newincmax = newinc
            if i1 > i2:
                decreasing = True
        for step in stepsplit:
            if len(step) > 1:
                rawdiffs = np.diff(step).tolist()
                subisodiffs.extend(rawdiffs)
    else:
        condensationcoordinates2 = [1 for i in range(maxstep)]
        summedintensities = intensities
        meanmasses = masses
        msums = summedintensities[:-1] + summedintensities[1:]
        majordiffs.extend((np.diff(summedintensities) / msums / 2).tolist())
        maxabundancedistribution = None
        #no subisodepth here???
    sumabundancedistribution = np.array([meanmasses, summedintensities])
    diffmax = np.abs(majordiffs).max()
    if diffmax > fulldiffmax:
        fulldiffmax = diffmax
    decreasing = False
    for ii in range(len(summedintensities)-1):
        i1 = summedintensities[ii]
        i2 = summedintensities[ii+1]
        if i2 > i1:
            if decreasing:
                newinc = abs(i2 - i1) / (i2 + i1) / 2
                if newinc > newincmax:
                    newincmax = newinc
        if i1 > i2:
            decreasing = True

lens = []
percs = []
means = []
edicts = []
for k, v in joinedabundances.items():
    percs.append(v[1].sum())
    lens.append(len(v[1]))
    means.append(v[0].mean())
    
    edict = {}
    element = 0
    estring = ''
    for n, i in enumerate(k):
        if i.isalpha():
            if n > 0:
                edict[element] = int(estring)
                estring = ''
            element = i
        else:
            estring += str(i)
    edict[element] = int(estring)
    edicts.append(edict)

plt.hist(percs, bins=100)
plt.title('sum generated distribution abundance')
plt.show()

plt.hist(lens, bins=100)
plt.title('length of generated distribution')
plt.show()

plt.plot(lens, percs, '.')
plt.xlabel('length of generated distribution')
plt.ylabel('sum generated distribution abundance')
plt.show()

#mass vs. # of isotopomers
plt.plot(lens, means, '.')
plt.ylabel('mean generated mass')
plt.xlabel('length of generated distribution')
plt.show()

#number of each element vs. # of isotopomers
for e in monoisotopickeys:
    outcount = []
    for d in edicts:
        try:
            outcount.append(d[e])
        except KeyError:
            outcount.append(0)
    plt.plot(lens, outcount, '.')
    plt.xlabel('length of generated distribution')
    plt.ylabel(f'count of {e}')
    plt.show()
