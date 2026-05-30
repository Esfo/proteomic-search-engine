from collections import defaultdict, Counter
import numpy as np
from scipy import special, optimize, stats
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

elementvectors = {}
nvectorpositions = {}
nelementpositions = {}
for e, isos in isotopesbyelement.items():
    elementvectors[e] = [0 for _ in range(len(isos))]
    nvectorpositions[e] = {k: n for n, k in enumerate(isos)}
    nelementpositions[e] = {n: k for n, k in enumerate(isos)}

maxcount = 1000000

#carbon
tn = 1
carbonresults = []
carbonvector = elementvectors['C'].copy()
while tn <= maxcount:
    try:
        c12 = elementalprobabilities['C12'] * tn
        c13 = elementalprobabilities['C13'] * tn
        #c12round = np.ceil(c12)
        #c13round = np.floor(c13)
        c12round = round(c12)
        c13round = round(c13)
        c12vector = carbonvector.copy()
        c12vector[0] += 1 #C12
        #nvector[1] += 1 #C13
        pn = 0
        c12prob = 1
        for n, c in enumerate(c12vector):
            loopiso = nelementpositions['C'][n]
            eprob = elementalprobabilities[loopiso]**c
            c12prob *= eprob
            if loopiso in nonmonoisotopicelements:
                ecomb = math.comb(tn-pn, c)
                c12prob *= ecomb
                pn += c
        c13vector = carbonvector.copy()
        #nvector[0] -= 1 #C12
        c13vector[1] += 1 #C13
        pn = 0
        c13prob = 1
        for n, c in enumerate(c13vector):
            loopiso = nelementpositions['C'][n]
            eprob = elementalprobabilities[loopiso]**c
            c13prob *= eprob
            if loopiso in nonmonoisotopicelements:
                ecomb = math.comb(tn-pn, c)
                c13prob *= ecomb
                pn += c
        if c13prob > c12prob:
            carbonvector = c13vector.copy()
            fprob = c13prob
        else:
            carbonvector = c12vector.copy()
            fprob = c12prob
        carbonresults.append([tn, carbonvector[0], carbonvector[1], c12round, c13round, fprob])
        tn += 1
    except OverflowError:
        break

carbonresults = np.array(carbonresults)

#the simple estimate is always off predictably here.
#(carbonresults[:,1] - carbonresults[:,3]).min()
#Out[3]: 0.0
#(carbonresults[:,1] - carbonresults[:,3]).max()
#Out[5]: 1.0
#(carbonresults[:,2] - carbonresults[:,4]).min()
#Out[6]: -1.0
#(carbonresults[:,2] - carbonresults[:,4]).max()
#Out[7]: 0.0

#plt.plot(carbonresults[:,-1], carbonresults[:,0])
#plt.show()
#
#def curvefunc(x, a, b):
#    return 1/(a*x + b)
#
#params, cov = optimize.curve_fit(curvefunc, carbonresults[:,0], carbonresults[:,5])
#output = curvefunc(carbonresults[:,0], *params)

#plt.plot(carbonresults[:,0], carbonresults[:,1], label='N x 12')
#plt.legend()
#plt.show()
#plt.plot(carbonresults[:,0], carbonresults[:,2], label='N x 13')
#plt.legend()
#plt.show()
#plt.plot(carbonresults[:,1], carbonresults[:,2], label='12 x 13')
#plt.legend()
#plt.show()

#plt.plot(np.where(carbonresults[:,1] != carbonresults[:,3])[0], label='nonmatch')
#plt.plot(np.where(carbonresults[:,1] == carbonresults[:,3])[0], label='match')
#plt.xlabel('#th match')
#plt.ylabel('# carbons')
#plt.legend()
#plt.show()
#
#last13 = 0
#difflocations = []
#for o in carbonresults.tolist():
#    if o[2] > last13:
#        last13 = o[2]
#        difflocations.append([o[0], o[2]])
#difflocations = np.array(difflocations)
#
#def curvefunc(x, a, b):
#    return a*x + b
##params, cov = optimize.curve_fit(curvefunc, difflocations[:,0], difflocations[:,1])
#params, cov = optimize.curve_fit(curvefunc, carbonresults[:,0], carbonresults[:,2])
#output = curvefunc(difflocations[:,0], *params)
##this output is perfectly predictive, and you can basically just searchsorted difflocations to figure out how many isotopes you'll need for max abundance
##the reason i'm abandoning this is because for fragment distributions i need to work outside the bounds of natural isotopic abundance
#
#carbonparams = [elementalprobabilities['C13'], -elementalprobabilities['C13']/2]
#carbonoutput = curvefunc(difflocations[:,0], *tryparams)

#drs = []
#base = 94
#for d in difflocations:
#    val = base * d[1]
#    diff = val - d[0]
#    drs.append([val, diff])
#
#drs = np.array(drs)

#def carbon_binom(n):
#    k_values = np.arange(0, n+1)
#    probabilities = stats.binom.pmf(k_values, n, elementalprobabilities['C13'])
#    max_k = k_values[np.argmax(probabilities)]


#oxygen
tn = 2
oxygenresults = []
oxygenvector = elementvectors['O'].copy()
oxygenvector[0] += 1
while tn <= maxcount:
    try:
        o16 = elementalprobabilities['O16'] * tn
        o17 = elementalprobabilities['O17'] * tn
        o18 = elementalprobabilities['O18'] * tn
        #o16round = np.ceil(o16)
        #o18floor = np.floor(o18)
        #if o16round - o18floor < tn:
        #    o18round = np.ceil(o18)
        #else:
        #    o18round = o18floor
        #o17round = np.floor(o17)
        o16round = round(o16)
        o17round = round(o17)
        o18round = round(o18)
        o16vector = oxygenvector.copy()
        o16vector[0] += 1 #o16
        pn = 0
        o16prob = 1
        for n, c in enumerate(o16vector):
            loopiso = nelementpositions['O'][n]
            eprob = elementalprobabilities[loopiso]**c
            o16prob *= eprob
            if loopiso in nonmonoisotopicelements:
                ecomb = math.comb(tn-pn, c)
                o16prob *= ecomb
                pn += c
        o17vector = oxygenvector.copy()
        o17vector[1] += 1 #o17
        pn = 0
        o17prob = 1
        for n, c in enumerate(o17vector):
            loopiso = nelementpositions['O'][n]
            eprob = elementalprobabilities[loopiso]**c
            o17prob *= eprob
            if loopiso in nonmonoisotopicelements:
                ecomb = math.comb(tn-pn, c)
                o17prob *= ecomb
                pn += c
        o18vector = oxygenvector.copy()
        o18vector[2] += 1 #o18
        pn = 0
        o18prob = 1
        for n, c in enumerate(o18vector):
            loopiso = nelementpositions['O'][n]
            eprob = elementalprobabilities[loopiso]**c
            o18prob *= eprob
            if loopiso in nonmonoisotopicelements:
                ecomb = math.comb(tn-pn, c)
                o18prob *= ecomb
                pn += c
        if o17prob > o16prob:
            if o17prob > o18prob:
                oxygenvector = o17vector.copy()
            else:
                oxygenvector = o18vector.copy()
        else:
            if o16prob > o18prob:
                oxygenvector = o16vector.copy()
            else:
                oxygenvector = o18vector.copy()
        oxygenresults.append([tn, oxygenvector[0], oxygenvector[1], oxygenvector[2], o16round, o17round, o18round])
        tn += 1
    except OverflowError:
        break

oxygenresults = np.array(oxygenresults)

#i'm going to abandon this as it ultimately won't help much for fragment distributions
#~
#actually, i'm going to try this again by enabling a change in element % abundance
#generate an ms1 formula distribution -> make fragments, and check how many of the fragments can be estimated via the same rounding technique
#maybe by using this expanded space i'll be able to see a better pattern
