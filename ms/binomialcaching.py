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

elementalprobabilities = {
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

elementalmasses = {
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

def newtest(atomiccomposition, dividingthreshold):
    elementaloutput = defaultdict(list) #element: [[prob, comp]]
    for e, count in atomiccomposition.items():
        #if e+acount in cache, for ms1 dists
        probarray = [elementalprobabilities[i] for i in isotopesbyelement[e]]
        initialcomp = [round(count * p) for p in probarray]
        total = sum(initialcomp)
        diff = total - count

        if diff < 0:
            testcomps = [[e + (i == n) for i, e in enumerate(initialcomp)] for n in range(len(initialcomp))]
        elif diff > 0:
            #testcomps = [[e - (i == n or e == 0) for i, e in enumerate(initialcomp)] for n in range(len(initialcomp))] #was replaced idk why
            testcomps = [[e - (i == n and e > 0) for i, e in enumerate(initialcomp)] for n in range(len(initialcomp))]
        else:
            #testcomps = [initialcomp[:-1] + [initialcomp[-1]]] + [[e - (i == n) + (i == n+1) for i, e in enumerate(initialcomp)] for n in range(len(initialcomp)-1)] #stopped working on a specific example, that somehow happened, it was also a pretty basic example. this is also > 2x slower than what's below
            testcomps = []
            # Always include the original composition
            testcomps.append(initialcomp.copy())
            # If there is no difference, we need to adjust the composition
            # For each element in the initial composition
            for i in range(len(initialcomp)):
                # If the current element can give a "unit"
                if initialcomp[i] > 0:
                    # For each element that can receive a "unit"
                    for j in range(len(initialcomp)):
                        if i != j:  # Cannot be the same element
                            # Copy the initial composition
                            newcomp = initialcomp.copy()
                            # Move a "unit" from i to j
                            newcomp[i] -= 1
                            newcomp[j] += 1
                            # Add the new composition to the list
                            testcomps.append(newcomp)
        
        #maxprob = -float('inf')
        #maxvec = None

        #make heaps
        mainheap = []
        visited = set()
        for comp in testcomps:
            prob = sum(c * math.log(p) for p, c in zip(probarray, comp))  # compute the log of newprob
            pn = 0
            for nn, c in enumerate(comp):
                if nn > 0:
                    prob += math.log(math.comb(count - pn, c))  # use the log of comb
                    pn += c
            tcomp = tuple(comp)
            mainheap.append((-np.exp(prob), tcomp))
            visited.add(tcomp)
            #if prob > maxprob:
            #    maxprob = prob
            #    maxvec = comp

        maxprob = -min(mainheap)[0]
        cutoff = maxprob * dividingthreshold
        
        heapq.heapify(mainheap)
        #pop heaps
        #return maxvec, math.exp(maxprob)
        #expanding directions in 2 dimensions is easy
            #- make one of both directions
            #- keep generating the direction with the previous highest until its lower
        #expanding directions in 3+ gets annoying
            #- you may be able to keep generating new testcomps while excluding a specific one based on which position was changed last
            #- good gpt question
        
        # Convert the start list to a tuple for set operations
        #initialcomp = tuple(initialcomp)
        #prob, comp = heapq.heappop(mainheap)
        # Calculate the starting probability
        #start_probability = calculate_probability(start_state, probarray)
        #initialprob = sum(c * math.log(-p) for p, c in zip(probarray, initialcomp))  # compute the log of newprob
        #pn = 0
        #for nn, c in enumerate(initialcomp):
        #    if nn > 0:
        #        initialprob += math.log(math.comb(count - pn, c))  # use the log of comb
        #        pn += c
        ## Initialize a heap with the starting state and its probability
        #mainheap = [(math.exp(initialprobability), initialcomp)]
        # Initialize a set with the starting state to keep track of visited states

        # List to hold the final states and their probabilities
        finalprobs = []
        sumprob = 0

        # Heap based BFS over the states of the list
        while mainheap:
            # Pop the state with the highest probability
            prob, comp = heapq.heappop(mainheap)
            #determine rate of loss to total
            # Add the current state and its probability to the final states (negate probability back)
            finalprobs.append((prob, comp))
            if -prob < cutoff:
                break
            
            ## Iterate over each index in the state
            #for i in range(len(comp) - 1):
            #    # Only generate new states if the current element is not '0'
            #    if comp[i] > 0:
            #        # Increment the next element and decrement the current one
            #        newcomp = list(comp)
            #        newcomp[i] -= 1
            #        newcomp[i + 1] += 1 #this is outright fraud, needs to be fixed
            #        newcomp = tuple(newcomp)
            #        # Check if we have already visited this state
            #        if newcomp not in visited:
            #            # Calculate the new probability
            #            #newprob = calculate_probability(newcomp, probarray)
            #            newprob = sum(c * math.log(p) for p, c in zip(probarray, newcomp))  # compute the log of newprob
            #            pn = 0
            #            for nn, c in enumerate(newcomp):
            #                if nn > 0:
            #                    newprob += math.log(math.comb(count - pn, c))  # use the log of comb
            #                    pn += c
            #            # Add the new state to the heap and visited set
            #            heapq.heappush(mainheap, (-np.exp(newprob), newcomp))
            #            visited.add(newcomp)
            for i in range(len(comp)):
                # If the current element can give a "unit"
                if comp[i] > 0:
                    # For each element that can receive a "unit"
                    for j in range(len(comp)):
                        if i != j:  # Cannot be the same element
                            # Copy the initial composition
                            newcomp = list(comp) #copy would be faster but its a tuple
                            # Move a "unit" from i to j
                            newcomp[i] -= 1
                            newcomp[j] += 1
                            newcomp = tuple(newcomp)
                            # Check if we have already visited this state
                            if newcomp not in visited:
                                # Calculate the new probability
                                #newprob = calculate_probability(newcomp, probarray)
                                newprob = sum(c * math.log(p) for p, c in zip(probarray, newcomp))  # compute the log of newprob
                                pn = 0
                                for nn, c in enumerate(newcomp):
                                    if nn > 0:
                                        newprob += math.log(math.comb(count - pn, c))  # use the log of comb
                                        pn += c
                                # Add the new state to the heap and visited set
                                heapq.heappush(mainheap, (-np.exp(newprob), newcomp))
                                visited.add(newcomp)
        elementaloutput[e].extend(finalprobs)
        heapq.heapify(elementaloutput[e])
    return elementaloutput


def elemental_binomial_walk(atomiccomposition, dividingthreshold):
    finaloutput = defaultdict(list)
    mainheap = []
    vectorsets = defaultdict(set) #element: set of used vectors
    nelements = len(atomiccomposition)
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
            finaloutput[e].append([-1, maxprob, m, e, nv])
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
                            newelementmass += elementalmasses[loopiso] * c
                            newelementprob *= elementalprobabilities[loopiso]**c
                            if loopiso in nonmonoisotopicelements:
                                newelementprob *= math.comb(acount-pn, c)
                                pn += c
                        heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, v.copy()])
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
            finaloutput[e].append([-1, maxprob, m, e, nv])
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
                        newelementmass += elementalmasses[loopiso] * c
                        newelementprob *= elementalprobabilities[loopiso]**c
                        if loopiso in nonmonoisotopicelements:
                            newelementprob *= math.comb(acount-pn, c)
                            pn += c
                    heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, v.copy()])
        
        cutoff = -maxprob * dividingthreshold

        r, p, m, e, v = heapq.heappop(mainheap)
        finaloutput[e].append([r, p, m, e, v])
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
                                newelementmass += elementalmasses[loopiso] * c
                                newelementprob *= elementalprobabilities[loopiso]**c
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
                            newelementmass += elementalmasses[loopiso] * c
                            newelementprob *= elementalprobabilities[loopiso]**c
                            if loopiso in nonmonoisotopicelements:
                                newelementprob *= math.comb(acount-pn, c)
                                pn += c
                        heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, nvector.copy()])
            r, p, m, e, v = heapq.heappop(mainheap)
            finaloutput[e].append([r, p, m, e, v])
    return finaloutput


dividingthreshold = 0.05 #% of the max probability that is the cutoff for other isotopomers

screwups, badmatches = [], []
times = []
for _ in range(1000):
    seq = ''.join((random.choices(list(aminoacidcomposition.keys()), k=np.random.randint(6,50))))
    atomiccomposition = Counter()
    for aa in seq:
        atomiccomposition += aminoacidcomposition[aa]
    atomiccomposition['H'] += 2
    atomiccomposition['O'] += 1
    
    ot = []
    nt = time()
    newoutput = newtest(atomiccomposition, dividingthreshold)
    st = time() - nt
    ot.append(st)
    nt = time()
    oldbased = elemental_binomial_walk(atomiccomposition, dividingthreshold)
    st = time() - nt
    ot.append(st)
    times.append(ot)
    
    for e, h in oldbased.items():
        if len(h) != len(newoutput[e]):
            screwups.append(sew)
        for n, l in enumerate(h):
            comparison = newoutput[e][n]
            if not np.isclose(-comparison[0], l[1]):
                badmatches.append(seq)

times = np.array(times)
plt.plot(times[:,0], times[:,1], '.')
plt.show()
print((times[:,0] > times[:,1]).sum(), 'newer times out of', len(times))

massesandabundances = [[], []]
for prod in itertools.product(*oldbased.values()):
    mass = 0
    prob = 1
    for r, p, m, e, v in prod:
        prob *= p
        mass += m
    massesandabundances[0].append(mass)
    massesandabundances[1].append(prob)
massesandabundances = np.array(massesandabundances)
massesandabundances = massesandabundances[:,massesandabundances[1].argsort()[::-1]][:,:20]

#what does it look like to do a single element, in descending order of isotopic-composition relevance, in terms of rate of change to total?
#is it always an increasing/decreasing rate of change of total sum abundance?

#generate each starting combination and their probs
#measure range of change from heap-based downward generation

#cutoffs are now based on maximum isotopomer abundance, the threshold is whatever fraction of max should be the lowest acceptable abundance
#this is also nested within elements
#the first to go below remains in the group, cuz its not really that important
