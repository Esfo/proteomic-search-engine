import random
from time import time
import numpy as np
from collections import Counter
from itertools import accumulate
import matplotlib.pyplot as plt
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

nfragmentcompositions = {}
nfragmentcompositions['a'] = Counter({'C': -1, 'O': -1})
nfragmentcompositions['b'] = Counter()
nfragmentcompositions['c'] = Counter({'N': 1, 'H': 3})

cfragmentcompositions = {}
cfragmentcompositions['x'] = Counter({'C': 1, 'O': 2})
cfragmentcompositions['y'] = Counter({'H': 2, 'O': 1})
cfragmentcompositions['z'] = Counter({'N': -1, 'H': -1, 'O': 1})

def fragmentation_compositions(seq):
    fragments = {}
    fragcomp = Counter()
    for n, aa in enumerate(seq[:-1]): #n-term
        fragcomp += aminoacidcomposition[aa]
        for ion, modcomp in nfragmentcompositions.items():
            fragments[ion + str(n + 1)] = fragcomp + modcomp
    fragcomp = Counter()
    for n, aa in enumerate(seq[::-1][:-1]): #c-term
        fragcomp += aminoacidcomposition[aa]
        for ion, modcomp in cfragmentcompositions.items():
            fragments[ion + str(n + 1)] = fragcomp + modcomp
    return fragments

def fragmentation_compositions_optimized(seq):
    fragments = {}

    # Calculate the compositions of the n-term fragments
    fragcomp_n = {}
    for n, aa in enumerate(seq[:-1]):  
        fragcomp_n = {k: fragcomp_n.get(k, 0) + aminoacidcomposition[aa].get(k, 0) 
                      for k in set(fragcomp_n).union(aminoacidcomposition[aa])}
        for ion, modcomp in nfragmentcompositions.items():
            fragments[ion + str(n + 1)] = {k: fragcomp_n.get(k, 0) + modcomp.get(k, 0) 
                                           for k in set(fragcomp_n).union(modcomp)}

    # Calculate the compositions of the c-term fragments
    fragcomp_c = {}
    for n, aa in enumerate(seq[::-1][:-1]): 
        fragcomp_c = {k: fragcomp_c.get(k, 0) + aminoacidcomposition[aa].get(k, 0) 
                      for k in set(fragcomp_c).union(aminoacidcomposition[aa])}
        for ion, modcomp in cfragmentcompositions.items():
            fragments[ion + str(n + 1)] = {k: fragcomp_c.get(k, 0) + modcomp.get(k, 0) 
                                           for k in set(fragcomp_c).union(modcomp)}

    return fragments

def fragmentation_compositions_optimized2(seq):
    fragments = {}

    # Calculate the compositions of the n-term fragments
    fragcomp_n = {}
    for n, aa in enumerate(seq[:-1]):  
        aa_composition = aminoacidcomposition[aa]
        for k in aa_composition:
            fragcomp_n[k] = fragcomp_n.get(k, 0) + aa_composition.get(k, 0)
        for ion, modcomp in nfragmentcompositions.items():
            fragment_composition = fragcomp_n.copy()
            for k in modcomp:
                fragment_composition[k] = fragment_composition.get(k, 0) + modcomp.get(k, 0)
            fragments[ion + str(n + 1)] = fragment_composition

    # Calculate the compositions of the c-term fragments
    fragcomp_c = {}
    for n, aa in enumerate(seq[::-1][:-1]): 
        aa_composition = aminoacidcomposition[aa]
        for k in aa_composition:
            fragcomp_c[k] = fragcomp_c.get(k, 0) + aa_composition.get(k, 0)
        for ion, modcomp in cfragmentcompositions.items():
            fragment_composition = fragcomp_c.copy()
            for k in modcomp:
                fragment_composition[k] = fragment_composition.get(k, 0) + modcomp.get(k, 0)
            fragments[ion + str(n + 1)] = fragment_composition

    return fragments

def fragmentation_compositions_fixed(seq):
    fragments = {}

    #calculate the compositions of the n-term fragments
    fragcomp_n = {}
    for n, aa in enumerate(seq[:-1]):  
        aa_composition = aminoacidcomposition[aa]
        for k in aa_composition:
            fragcomp_n[k] = fragcomp_n.get(k, 0) + aa_composition.get(k, 0)
        for ion, modcomp in nfragmentcompositions.items():
            fragment_composition = fragcomp_n.copy()
            for k in modcomp:
                fc = fragment_composition.get(k, 0) + modcomp.get(k, 0)
                if fc > 0:
                    fragment_composition[k] = fc
                else:
                    del fragment_composition[k]
            fragments[ion + str(n + 1)] = fragment_composition

    #calculate the compositions of the c-term fragments
    fragcomp_c = {}
    for n, aa in enumerate(seq[::-1][:-1]): 
        aa_composition = aminoacidcomposition[aa]
        for k in aa_composition:
            fragcomp_c[k] = fragcomp_c.get(k, 0) + aa_composition.get(k, 0)
        for ion, modcomp in cfragmentcompositions.items():
            fragment_composition = fragcomp_c.copy()
            for k in modcomp:
                fc = fragment_composition.get(k, 0) + modcomp.get(k, 0)
                if fc > 0:
                    fragment_composition[k] = fc
                else:
                    del fragment_composition[k]
            fragments[ion + str(n + 1)] = fragment_composition

    return fragments

#seq = 'EGKMLSPHENDYDNSPTALSRISSPNSDR'
n = 0
times = []
while n < 1000:
    n += 1
    t = []
    seq = ''.join((random.choices(list(aminoacidcomposition.keys()), k=np.random.randint(6,50))))
    nt = time()
    oldfrags = fragmentation_compositions(seq)
    t.append(time() - nt)
    nt = time()
    newfrags = fragmentation_compositions_optimized(seq)
    t.append(time() - nt)
    nt = time()
    newfrags2 = fragmentation_compositions_optimized2(seq)
    t.append(time() - nt)
    times.append(t)
    nt = time()
    newfrags3 = fragmentation_compositions_fixed(seq)
    t.append(time() - nt)
    times.append(t)

times = np.array(times)

print((times[:,0] < times[:,1]).sum())
print((times[:,0] < times[:,2]).sum())
print((times[:,1] < times[:,2]).sum())
print((times[:,1] < times[:,3]).sum())
