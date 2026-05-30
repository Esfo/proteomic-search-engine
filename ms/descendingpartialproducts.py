from collections import defaultdict, Counter
import numpy as np
import itertools
import bisect
import string
import random
import heapq
import math
import copy

maxletters = 6
maxsubs = 4
mincount = 10
maxcount = 40
dividingthreshold = 0.05

alletters = string.ascii_uppercase
#letters = random.sample(alletters, maxletters)

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

elements = {'H', 'C', 'N','O', 'S'}

isotopesbyelement = {
        'H': ('H1', 'H2'),
        'C': ('C12', 'C13'),
        'N': ('N14', 'N15'),
        'O': ('O16', 'O17', 'O18'),
        'S': ('S32', 'S34', 'S33', 'S36')} #in order of abundance

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

nonmonoisotopicelements = {'H2', 'C13', 'N15', 'O17', 'O18', 'S33', 'S34', 'S36'}

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

elementvectors = {}
nvectorpositions = {}
elementpositions = {}
elementprobabilities = {}
for e, isos in isotopesbyelement.items():
    elementvectors[e] = [0 for _ in range(len(isos))]
    nvectorpositions[e] = {k: n for n, k in enumerate(isos)}
    elementpositions[e] = {n: k for n, k in enumerate(isos)}
    elementprobabilities[e] = {n: elementalprobabilities[n] for n in isotopesbyelement[e]}

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
                            loopiso = elementpositions[e][n]
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
                            loopiso = elementpositions[e][n]
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
                        loopiso = elementpositions[e][n]
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
                        loopiso = elementpositions[e][n]
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
                                loopiso = elementpositions[e][n]
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
                            loopiso = elementpositions[e][n]
                            newelementmass += elementalmasses[loopiso] * c
                            newelementprob *= elementalprobabilities[loopiso]**c
                            if loopiso in nonmonoisotopicelements:
                                newelementprob *= math.comb(acount-pn, c)
                                pn += c
                        heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, nvector.copy()])
            r, p, m, e, v = heapq.heappop(mainheap)
            finaloutput[e].append([r, p, m, e, v])
    return finaloutput

#probcounts = {} #l: probarray
#firstheaps = {}
#for l in elements:
#    nsubs = np.random.randint(1, maxsubs)
#    count = np.random.randint(mincount, maxcount)
#    #probarray = np.random.uniform(size=nsubs)
#    #while probarray.max() < 0.94:
#    #    probarray **= 2
#    #    probarray /= probarray.sum()
#    #probarray = np.sort(probarray)[::-1]
#    #probcounts[l] = probarray
#    probarray = elementprobabilities[l]
#    initialcomp = [round(count * p) for p in probarray]
#    total = sum(initialcomp)
#    diff = total - count
#
#    if diff < 0:
#        testcomps = [[e + (i == n) for i, e in enumerate(initialcomp)] for n in range(len(initialcomp))]
#    elif diff > 0:
#        #testcomps = [[e - (i == n or e == 0) for i, e in enumerate(initialcomp)] for n in range(len(initialcomp))] #was replaced idk why
#        testcomps = [[e - (i == n and e > 0) for i, e in enumerate(initialcomp)] for n in range(len(initialcomp))]
#    else:
#        #testcomps = [initialcomp[:-1] + [initialcomp[-1]]] + [[e - (i == n) + (i == n+1) for i, e in enumerate(initialcomp)] for n in range(len(initialcomp)-1)] #stopped working on a specific example, that somehow happened, it was also a pretty basic example. this is also > 2x slower than what's below
#        testcomps = []
#        # Always include the original composition
#        testcomps.append(initialcomp.copy())
#        # If there is no difference, we need to adjust the composition
#        # For each element in the initial composition
#        for i in range(len(initialcomp)):
#            # If the current element can give a "unit"
#            if initialcomp[i] > 0:
#                # For each element that can receive a "unit"
#                for j in range(len(initialcomp)):
#                    if i != j:  # Cannot be the same element
#                        # Copy the initial composition
#                        newcomp = initialcomp.copy()
#                        # Move a "unit" from i to j
#                        newcomp[i] -= 1
#                        newcomp[j] += 1
#                        # Add the new composition to the list
#                        testcomps.append(newcomp)
#    
#    subheap = []
#    for comp in testcomps:
#        #prob = sum(c * math.log(p) for p, c in zip(probarray, comp))  #compute the log of newprob
#        prob = 1
#        pn = 0
#        for nn, c in enumerate(comp):
#            prob *= probarray[nn] ** c
#            if nn > 0:
#                #prob += math.log(math.comb(count - pn, c))  #use the log of comb
#                prob *= math.comb(count-pn, c)
#                pn += c
#        #subheap.append([np.exp(prob), l, comp])
#        subheap.append([prob, l, comp])
#    
#    maxprob = max(subheap)[0]
#    for s in subheap:
#        r = s[0] / maxprob
#        s.insert(0, -r)
#
#    firstheaps[l] = subheap

#firstheaps = {'A': [[-1, 0.9, 'A', [12, 1]], #[ratio, probability, key']
#                [-0.0555, 0.05, 'A', [12, 2]]],
#              'B': [[-1, 0.5, 'B', [15, 1, 0]],
#                [-0.8, 0.4, 'B', [14, 2, 0]],
#                [-0.1, 0.05, 'B', [13, 2, 1]]],
#              'C': [[-1, 0.7, 'C', [10, 0]],
#                [-0.2857, 0.2, 'C', [9, 1]]]}

#firstheaps = {'H': [[-0.6055502886034876, 0.0017114188653438106, 'H', [13, 10]],
#  [-1.0, 0.002826220873068467, 'H', [12, 11]]],
# 'X': [[-0.8574007193397538, 0.010636265187230692, 'X', [21, 15]],
#  [-1.0, 0.012405244067699415, 'X', [20, 16]]],
# 'L': [[-1.0, 0.27587772003428507, 'L', [33, 2]],
#  [-0.7386581231894632, 0.2037793189103132, 'L', [32, 3]],
#  [-0.8759922936140809, 0.24166675672985666, 'L', [34, 1]]]}

#firstheaps = {'B': [[-0.9945845467542896, 0.13512058752948827, 'B', [24, 13]],
#  [-0.8817997349726828, 0.11979805906062561, 'B', [23, 14]],
#  [-1.0, 0.13585631103000595, 'B', [25, 12]]],
# 'H': [[-1.0, 0.19463327108919154, 'H', [5, 21]],
#  [-0.9225732448932664, 0.17956344847294622, 'H', [4, 22]],
#  [-0.8622128918843438, 0.16781531552272128, 'H', [6, 20]]],
# 'Q': [[-1.0, 0.24596980913913094, 'Q', [3, 14]],
#  [-0.9778239838275642, 0.24051517867373062, 'Q', [2, 15]],
#  [-0.7158752613736697, 0.17608370140750704, 'Q', [4, 13]]]}

#firstheaps = {'O': [[-0.17982270064792324, 0.00022958930861979678, 'O', [4, 2]],
#  [-1.0, 0.0012767537568536026, 'O', [3, 3]]],
# 'S': [[-1.0, 0.8387344854954, 'S', [7, 5]],
#  [-0.9362133715989615, 0.7852344405419687, 'S', [8, 4]]],
# 'D': [[-0.8738460137986074, 0.01967494253897154, 'D', [13, 4]],
#  [-1.0, 0.02251534278155552, 'D', [12, 5]]],
# 'N': [[-0.45225593746691894, 0.032025130854761906, 'N', [8, 18]],
#  [-1.0, 0.07081196331912047, 'N', [7, 19]]],
# 'B': [[-0.28400270532638, 0.004115104506012636, 'B', [5, 5]],
#  [-1.0, 0.014489666573012037, 'B', [4, 6]]]}

seq = ''.join((random.choices(list(aminoacidcomposition.keys()), k=np.random.randint(6,50))))

atomiccomposition = Counter()
for aa in seq:
    atomiccomposition += aminoacidcomposition[aa]
atomiccomposition['H'] += 2
atomiccomposition['O'] += 1

firstheaps = elemental_binomial_walk(atomiccomposition, dividingthreshold)

originalheap = copy.deepcopy(firstheaps)

output = []
for prod in itertools.product(*firstheaps.values()):
    formulas = []
    prob = 1
    for r, p, m, e, v in sorted(prod, key=lambda x: x[2]):
        #formula += ''.join(([str(e) + f'{n}' + f'({i})' for n, i in enumerate(v) if i > 0]))
        formula = ''
        for n, c in enumerate(v):
            if c > 0:
                formula += f'{elementpositions[e][n]}({c})'
        formulas.append(formula)
        prob *= p
    formula = ''.join((sorted(formulas)))
    output.append((formula, prob))

output = sorted(output, key=lambda x: x[1], reverse=True)

#have a pool and a mainheap
#mainheap has products
#pool has individual sublists that are a part of things allowed into mainheap, they get popped from firstheaps
for k in firstheaps:
    heapq.heapify(firstheaps[k])

mainpool = defaultdict(list) #things already popped from firstheaps
for k in firstheaps:
    mainpool[k].append(heapq.heappop(firstheaps[k]))

formula = ''
maxprob = 1
finalprobs = {} #subformula: prob
for b in sorted(mainpool):
    for r, p, m, e, v in mainpool[b]:
        #formula += ''.join(([str(e) + f'{n}' + f'({i})' for n, i in enumerate(v) if i > 0]))
        for n, c in enumerate(v):
            if c > 0:
                formula += f'{elementpositions[e][n]}({c})'
        maxprob *= p

finalprobs[formula] = maxprob

oflen = len(formula)

cutoff = maxprob * dividingthreshold
mainheap = list(itertools.chain(*firstheaps.values()))
heapq.heapify(mainheap)

answers = {i[0]: i[1] for i in output if i[1] >= cutoff}

failures = []
maincount = 0
multinomialpath = [] #sublists not in mainpool
#multinomialpath.extend([i[0] for i in mainpool.values()])
probabilityranking = [] #representative lists of ratio probability to sort multinomialpath
#probabilityranking = [i[0] for i in multinomialpath] #representative lists of ratio probability to sort multinomialpath
#while loop here that deals with new element addition + products of all others
while mainheap:
    r, p, m, e, v = heapq.heappop(mainheap)
    baseiter = {k: v for k, v in mainpool.items() if k != e}
    baseiter[e] = [[r, p, m, e, v]]
    maincount += 1
    print('~ main iter', maincount)
    print(r, p, m, e, v)
    
    formula = ''
    prob = 1
    for b in sorted(baseiter):
        for sr, sp, sm, se, sv in baseiter[b]:
            #formula += ''.join(([str(se) + f'{n}' + f'({i})' for n, i in enumerate(sv) if i > 0]))
            for n, c in enumerate(sv):
                if c > 0:
                    formula += f'{elementpositions[se][n]}({c})'
            prob *= sp
    if len(formula) < oflen/2:
        print(formula)
        print('initialization', r, p, m, e, v, formula, prob)
        print(baseiter)
        print('~')
    print(formula)
    
    if prob < cutoff:
        break
    finalprobs[formula] = prob
    
    ind = bisect.bisect(probabilityranking, r)
    if r in probabilityranking:
        print('non multi', r, p, m, e, v, formula)
    probabilityranking.insert(ind, r)
    multinomialpath.insert(ind, [r, p, m, e, v])
    #multinomialpath.append([r, p, e, v])
    #newadditions.append([r, p, e, v])
    #print('newaddition length', len(newadditions))
    #print('newadditions', newadditions)
    #now combine all newadditions for the last bit of combinations
    #might need a while loop?
    
    #^mainpool will be automatically applied to each new incomer, check to break
    #if the mainpool applies without breaking, proceed to the looping heap where all the 2nd/3rd places can be looped through each......
    #^this looping heap should be done via different combinations that replace members of mainheap
    #so not just individual elements, but combinations
    #the combinations will apply in order to the following NEW combination that gets made with the newest incomer -> and THAT then gets added to the end of the list (or does it need to be bisected in?)
    #so this wouldn't even need to be a heap, just a list afaik -> its automatically sorted via appending/bisecting
    itern = 0
    checkedcombos = set()
    #newadditions = []
    for path in multinomialpath.copy():
    #for path in multinomialpath:
        itern += 1
        print(' ~ subiter', itern)
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
                        #sef = ''.join((f'{se}{str(n)}{(val)}' for n, val in enumerate(sv)))
                        sef = ''
                        for n, c in enumerate(sv):
                            if c > 0:
                                sef += f'{elementpositions[se][n]}({c})'
                        seformulas.append(sef)
                        multipath.append([sr, sp, sm, se, sv])
                checkformula = ''.join((sorted(seformulas)))
                print('normal - multipath', multipath)
                print('normal - sepool', sepool)
                if checkformula in checkedcombos:
                    print('multi repeated combo aborted')
                    continue
                else:
                    checkedcombos.add(checkformula)
                if len(multipath) == 0:
                    print('empty multipath aborted')
                    continue
            case _:
                sr, sp, sm, se, sv = path
                sef = ''.join((f'{se}{str(n)}{(val)}' for n, val in enumerate(sv)))
                print(se, sp, se, sv)
                if sef in checkedcombos:
                    print('single repeat combo aborted')
                    continue
                else:
                    checkedcombos.add(sef)
                if se == e:
                    print('same element failure caught')
                    continue
                nsr = sr
        newratio = nsr * r
        if newratio > 0:
            newratio *= -1
        if -newratio >= dividingthreshold:
            if multielement:
                seformula = ''
                newprob = 1
                newiter = {k: v for k, v in baseiter.items() if k not in sepool}
                newiter[e] = [[r, p, m, e, v]]
                for ir, ip, im, ie, iv in multipath:
                    newiter[ie] = [[ir, ip, im, ie, iv]]
                for b in sorted(newiter):
                    for ir, ip, im, ie, iv in newiter[b]:
                        #seformula += ''.join(([str(ie) + f'{n}' + f'({i})' for n, i in enumerate(iv) if i > 0]))
                        for n, c in enumerate(iv):
                            if c > 0:
                                seformula += f'{elementpositions[ie][n]}({c})'
                        newprob *= ip
            else:
                newiter = {k: v for k, v in baseiter.items() if k != se}
                newiter[se] = [[sr, sp, sm, se, sv]]
                seformula = ''
                newprob = 1
                for b in sorted(newiter):
                    for ir, ip, im, ie, iv in newiter[b]:
                        #seformula += ''.join(([str(ie) + f'{n}' + f'({i})' for n, i in enumerate(iv) if i > 0]))
                        for n, c in enumerate(iv):
                            if c > 0:
                                seformula += f'{elementpositions[ie][n]}({c})'
                        newprob *= ip
            if newprob >= cutoff:
                finalprobs[seformula] = newprob
                if multielement:
                    ind = bisect.bisect(probabilityranking, newratio)
                    if newratio in probabilityranking:
                        print('multi, ranking', r, p, m, e, v, seformula)
                        print(multipath)
                        print(newiter)
                        print('~')
                    probabilityranking.insert(ind, newratio)
                    multinomialpath.insert(ind, [newratio, *multipath])
                    #newadditions.append([newratio, *multipath])
                    if len(multinomialpath[ind]) == 1:
                        print('length failure muli', r, p, m, e, v, seformula, multinomialpath[ind])
                    #multinomialpath.append([newratio, *multipath])
                else:
                    ind = bisect.bisect(probabilityranking, newratio)
                    if newratio in probabilityranking:
                        print('single multi', r, p, m, e, v, seformula)
                        print(sr, sp, sm, se, sv)
                        print(newiter)
                        print('~')
                    probabilityranking.insert(ind, newratio)
                    multinomialpath.insert(ind, [newratio, [sr, sp, sm, se, sv], [r, p, m, e, v]])
                    #newadditions.append([newratio, [sr, sp, se, sv], [r, p ,e, v]])
                    if len(multinomialpath[ind]) == 1:
                        print('length failure single', r, p, m, e, v, seformula, multinomialpath[ind])
                    #multinomialpath.append([newratio, [sr, sp, se, sv], [r, p ,e, v]])
        else:
            if multielement:
                print('multi cutoff failure', multipath)
            else:
                print('single cutoff failure', sr, sp, sm, se, sv, '~', r, p, e, v)
            break

#pop mainheap
#apply base maxes
    #- if below thresh, break while loop
#loop multinomial path
    #- if below thresh, break, but maintain the while loop
#after looping finishes, append/bisect
#apply a combination of that newest addition to every existing member of the looping list
    #- if the combo has a ratio below cutoff (pre-calculate this too) then break that loop and don't combine any more

print('~~~~~~~~~~~~~~~~~~')

errors = 0
for n, (k, v) in enumerate(finalprobs.items()):
    print(k)
    print(answers[k], v)
    if not np.isclose(v, answers[k]):
        errors += 1
    print('~')

if len(finalprobs) == len(answers):
    print('length pass')
else:
    print('length failure')
print(errors, 'errors')

#so the exact descending order doesn't actually matter
#the main loop is guaranteed descent, while the subloop is adjacent descent that isn't perfect but keeps you from missing any stragglers
#so its as perfect as it can be, mission accomplished
