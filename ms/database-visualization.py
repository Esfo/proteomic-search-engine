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
import lmdb
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
librarylocation = '/home/sfo/data/proteomics/fastas/search-db/'
proteomefile = '/home/sfo/data/proteomics/fastas/proteomes/Human_Homo_sapien.fasta'
proteome = proteomefile.split('/')[-1].split('.')[0]

environment_partial = partial(lmdb.Environment, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=9999, max_spare_txns=1, lock=True)

getkeys = []
abundances = {} #formula: [[masses], [intensities]]
with environment_partial(librarylocation) as env:
    formuladb = '.'.join(('formulaidentifier', proteome))
    formulas = env.open_db(formuladb.encode())
    with env.begin(write=False) as txn:
        with txn.cursor(formulas) as cursor:
            for k, v in cursor:
                getkeys.append(k)
    fulldb = env.open_db('distributions.full'.encode())
    with env.begin(write=False) as txn:
        with txn.cursor(fulldb) as cursor:
            for k, v in cursor.getmulti(getkeys):
                out = np.frombuffer(v)
                out = out.reshape(2, out.size//2)
                abundances[k.decode()] = out

lens = []
percs = []
for k, v in abundances.items():
    percs.append(v[1].sum())
    lens.append(len(v[1]))

plt.hist(percs, bins=100)
plt.show()

plt.hist(lens, bins=100)
plt.show()

plt.plot(lens, percs, '.', alpha=0.03)
plt.show()

#these seem to match the visualization of random peptides from cache-modeling.py pretty well
