import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from colour import Color
from pyteomics import mzml
from time import time
import pandas as pd
import lmdb
from functools import partial
import gc
import concurrent.futures
import multiprocessing as mp
from multiprocessing.managers import BaseManager, DictProxy
from collections import Counter, defaultdict
from textwrap import wrap
from scipy import sparse, integrate, spatial, stats, special
from pandas.api.types import CategoricalDtype
from sklearn.neighbors import NearestNeighbors
from more_itertools import sort_together
from distinctipy import distinctipy as dp
import math
import sqlitedict as sq
import random
import itertools
import pickle
import sys
import os
import warnings
#warnings.filterwarnings("error")
np.set_printoptions(suppress=True)
#np.warnings.filterwarnings('ignore') #depricated i guess
np.testing.suppress_warnings(forwarding_rule='always')
gc.enable()

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

def boundary_stack(rbounds):
    boundarybreaks = np.unique(rbounds)
    boundarystack = np.stack((boundarybreaks[:-1], boundarybreaks[1:]), axis=1)
    rbounds = rbounds.tolist()
    rsize = len(rbounds)
    finalsum = 0
    for ls, rs in boundarystack.tolist():
        stackslice = rs - ls #destined to be smaller than boundslice, basically
        stacksum = 0
        stacklines = 0
        for lb, rb in rbounds:
            if ls < rb and rs > lb: #overlap check
                boundslice = rb - lb
                stackfrac = stackslice / boundslice
                stacksum += stackfrac
                stacklines += 1
        nonoverlaps = rsize - stacklines
        sover = stacklines - nonoverlaps
        stacknum = sover * (stacksum * stacklines) ** (sover / rsize)
        finalsum += stacknum
    return finalsum

goodstack = [[85, 90],
             [86, 89],
             [87, 88]]

betterstack = [[85, 90],
               [86, 89],
               [86, 88]]

beststack = [[85, 90],
             [86, 90],
             [86, 88]]

badstack = [[85, 90],
            [83, 86],
            [86, 92]]

worsestack = [[85, 90],
              [80, 82],
              [86, 92]]

worststack = [[85, 90],
              [80, 82],
              [89, 92]]

stacks = []
stacks.append(worststack)
stacks.append(worsestack)
stacks.append(badstack)
stacks.append(goodstack)
stacks.append(betterstack)
stacks.append(beststack)
#in order of shit -> good

for stack in stacks:
    stack = np.array(stack)
    unstackedboundaries = stack - stack[:,0,None]
    rstack = boundary_stack(stack)
    ustack = boundary_stack(unstackedboundaries)
    print(rstack, ustack)

#perfectly overlapping with decreasing sizes in some way or another -> best good
#non-perfectly overlapping with decreasing sizes in some way or another -> next best good
#non-perfectly overlapping with decreasing sizes and increasing distance away from the core -> ~good
#non-perfectly overlapping with increasing sizes -> bad
#completely misaligned -> more bad

#where decreasing sizes means along the axis of increasing mass

#~
#improving the boundary stacking to better elucidate good vs mid overlaps and so on will be key to a new distributionassembly model
#it currently does an interesting counterbalance with intensity diffs
#where it can be improved is basically in the logic of what it considers as a metric, big number good small number worse negative number bad
#i think it poorly handles non-overlaps somewhat though, doesn't it?
#i think it currently adds up points for stacks at each interval depending how many things overlap there
#maybe i should divide by intervals instead of lines? it would always be < 1
