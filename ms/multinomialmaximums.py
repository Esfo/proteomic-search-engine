import numpy as np
import itertools
import math

#this file was used to generate the single max abundance estimation concept for theoretical ms2 fragments

#generate 2, 3s, and groups of 4 i suppose

#but start with 2s
#titrate %'s
#check how rounding/+1 works vs exact calculation

probarray = [1, 0, 0]
modifier = 0.001
maxcount = 999

for n in range(len(probarray[1:])):
    probarray[0] -= modifier
    probarray[n+1] += modifier

results = []
while probarray[0] >= (1 / len(probarray)) - modifier: #otherwise you reach the same, symmetrical solutions to what you've already solved for
    
    try:
        addposition = np.random.randint(1, len(probarray))
    except ValueError:
        addposition = 1
    
    probarray[0] -= modifier
    probarray[addposition] += modifier
    
    countarray = [0 for _ in probarray]
    
    tn = 1
    while tn <= maxcount:
        try:
            #estimations
            estimates = [round(tn * p) for p in probarray]
            testcomps = [estimates.copy() for i in estimates]
            if sum(estimates) < tn:
                #print(tn, probarray, estimates)
                #testcomps[0][1] -= 1
                #testcomps[1][0] -= 1
                for n, t in enumerate(testcomps):
                    t[n] += 1
            elif sum(estimates) > tn:
                #print(tn, probarray, estimates)
                #testcomps[0][1] -= 1
                #testcomps[1][0] -= 1
                for n, t in enumerate(testcomps):
                    if t[n] > 0:
                        t[n] -= 1
                    else:
                        t[0] -= 1
                        t[n] += 1
            else:
                #check if there's enough to subtract?
                #testcomps[0][0] -= 1
                #testcomps[0][1] += 1
                for n, t in enumerate(testcomps[:-1]):
                    t[0] -= 1
                    t[n+1] += 1

            #^you can fit these into for-loops quite logically. the second one might need slight use of combinatorics?
            probvectors = []
            probvals = []
            for comp in testcomps:
                for n, v in enumerate(probarray):
                    pn = 0
                    newprob = 1
                    for nn, c in enumerate(comp):
                        newprob *= probarray[nn]**c
                        if nn > 0:
                            newprob *= math.comb(tn-pn, c)
                            pn += c
                    probvectors.append(comp.copy())
                    probvals.append(newprob)
            maxprob = max(probvals)
            maxvec = probvectors[probvals.index(maxprob)]
            out1 = [tn, maxprob, maxvec.copy(), probarray.copy()]

            #calculations
            probvectors = []
            probvals = []
            for n, v in enumerate(probarray):
                newvec = countarray.copy()
                newvec[n] += 1
                pn = 0
                newprob = 1
                for nn, c in enumerate(newvec):
                    newprob *= probarray[nn]**c
                    if nn > 0:
                        newprob *= math.comb(tn-pn, c)
                        pn += c
                probvectors.append(newvec.copy())
                probvals.append(newprob)
            maxprob = max(probvals)
            maxvec = probvectors[probvals.index(maxprob)]
            out2 = [tn, maxprob, maxvec.copy(), probarray.copy()]
            results.append(out1 + out2)
            countarray = maxvec
            tn += 1
        except OverflowError:
            break

nopes = []
for n, r in enumerate(results):
    if not results[0][2] == results[0][6]:
        nopes.append(n)
print(len(nopes), 'mismatches')

#next steps:
#do the rounding estimates encompass the 2nd best?
#compare the rounding output, and its 2nd best in variablefraganalysis
