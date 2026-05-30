import numpy as np
from time import time
from collections import defaultdict, Counter
from scipy import special
import math
from scipy.special import comb
from operator import mul
from functools import reduce


#probarray = [0.8, 0.3, 0.2]
#count = 784
#
#print('count', count)
#print('probarray', probarray)
#print('maxvecs:')

def max_estimation(count, probarray):
    estimates = [round(count * p) for p in probarray]
    testcomps = [estimates.copy() for i in estimates]
    if sum(estimates) < count:
        for n, t in enumerate(testcomps):
            t[n] += 1
    elif sum(estimates) > count:
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
    try:
        probvectors = []
        probvals = []
        for comp in testcomps:
            for n, v in enumerate(probarray):
                pn = 0
                newprob = 1
                for nn, c in enumerate(comp):
                    newprob *= probarray[nn]**c
                    if nn > 0:
                        newprob *= math.comb(count-pn, c)
                        pn += c
                probvectors.append(comp.copy())
                probvals.append(newprob)
        maxprob = max(probvals)
        maxvec = probvectors[probvals.index(maxprob)]
        return maxvec, maxprob

    except OverflowError:
        print('overflowed')

def max_estimation_optimized(count, probarray):
    probarray = np.array(probarray)
    estimates = np.round(count * probarray).astype(int)
    testcomps = np.tile(estimates, (len(estimates), 1))
    
    testcomps[np.arange(len(estimates)), np.arange(len(estimates))] += np.where(np.sum(estimates) < count, 1, np.where(np.sum(estimates) > count, -1, 0))
    maxprob = -np.inf
    maxvec = None
    
    for comp in testcomps:
        #newprob = np.prod(np.power(probarray, comp))
        log_newprob = sum(c * math.log(p) for p, c in zip(probarray, comp))  # compute the log of newprob
        pn = 0
        for nn, c in enumerate(comp):
            if nn > 0:
                #newprob *= comb(count - pn, c)
                log_newprob += math.log(comb(count - pn, c))  # use the log of comb
                pn += c
        if log_newprob > maxprob:
            maxprob = log_newprob
            maxvec = comp
            
    return maxvec.tolist(), math.exp(maxprob)

def max_estimation_small(count, probarray):
    estimates = [round(count * p) for p in probarray]
    testcomps = [estimates.copy() for _ in estimates]
    
    diff = sum(estimates) - count
    if diff < 0:
        for n, t in enumerate(testcomps):
            t[n] += 1
    elif diff > 0:
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
    
    maxprob = -float('inf')
    maxvec = None
    
    for comp in testcomps:
        #newprob = math.prod(p**c for p, c in zip(probarray, comp))
        log_newprob = sum(c * math.log(p) for p, c in zip(probarray, comp))  # compute the log of newprob
        pn = 0
        for nn, c in enumerate(comp):
            if nn > 0:
                #newprob *= math.comb(count - pn, c)
                log_newprob += math.log(comb(count - pn, c))  # use the log of comb
                pn += c
        if log_newprob > maxprob:
            maxprob = log_newprob
            maxvec = comp
    return maxvec, math.exp(maxprob)

def max_estimation_optimized_small_v2(count, probarray):
    estimates = [round(count * p) for p in probarray]
    total = sum(estimates)
    diff = total - count

    if diff < 0:
        testcomps = [[e + (i == n) for i, e in enumerate(estimates)] for n in range(len(estimates))]
    elif diff > 0:
        #testcomps = [[e - (i == n or e == 0) for i, e in enumerate(estimates)] for n in range(len(estimates))] #was replaced idk why
        testcomps = [[e - (i == n and e > 0) for i, e in enumerate(estimates)] for n in range(len(estimates))]
    else:
        testcomps = [estimates[:-1] + [estimates[-1]]] + [[e - (i == n) + (i == n+1) for i, e in enumerate(estimates)] for n in range(len(estimates)-1)]

    maxprob = -float('inf')
    maxvec = None

    for comp in testcomps:
        log_newprob = sum(c * math.log(p) for p, c in zip(probarray, comp))  # compute the log of newprob
        pn = 0
        for nn, c in enumerate(comp):
            if nn > 0:
                log_newprob += math.log(math.comb(count - pn, c))  # use the log of comb
                pn += c
        if log_newprob > maxprob:
            maxprob = log_newprob
            maxvec = comp

    return maxvec, math.exp(maxprob)

n = 0
maxspots = 3
compnopes = []
probnopes = []
output = defaultdict(list)
while n < 1000:
    n += 1
    nprobs = np.random.randint(maxspots) + 1
    probarray = np.random.uniform(size=nprobs)
    probarray = probarray / probarray.sum()
    probarray = sorted(probarray.tolist(), reverse=True)
    #probarray = probarray.tolist() #all functions required probarray to be reverse sorted or else nothing works
    count = np.random.randint(100,1000)
    t = []
    nt = time()
    orig = max_estimation(count, probarray)
    t.append(time() - nt)
    nt = time()
    new1 = max_estimation_optimized(count, probarray)
    t.append(time() - nt)
    nt = time()
    new2 = max_estimation_small(count, probarray)
    t.append(time() - nt)
    nt = time()
    new3 = max_estimation_optimized_small_v2(count, probarray)
    t.append(time() - nt)
    output[nprobs].append(t)
    if new1[0] != orig[0]:
        compnopes.append([n, 1, new1, orig, probarray])
    if new2[0] != orig[0]:
        compnopes.append([n, 2, new2, orig, probarray])
    if new3[0] != orig[0]:
        compnopes.append([n, 3, new3, orig, probarray])
    if not np.isclose(new1[1], orig[1]):
        probnopes.append([n, 1, new1, orig, probarray])
    if not np.isclose(new2[1], orig[1]):
        probnopes.append([n, 2, new2, orig, probarray])
    if not np.isclose(new3[1], orig[1]):
        probnopes.append([n, 3, new3, orig, probarray])

for k in sorted(output):
    v = output[k]
    v = np.array(v)
    length = v.shape[1] - 1
    print('length', k)
    am = v.sum(axis=0)
    print(am.tolist())
    print('winner', am.argmin())
    print(Counter(v.argmin(axis=1)))
    print('~~~')
