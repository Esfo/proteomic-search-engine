import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from colour import Color
import psutil
import asyncio
import aiofiles
from pyteomics import mzml
import csv
import bisect
import heapq
import fcntl #this will need to be portalocker on other operating systems
from time import time
import pandas as pd
import gc
import concurrent.futures
import multiprocessing as mp
from collections import Counter, defaultdict
from scipy import sparse, integrate, spatial, stats, special
from pandas.api.types import CategoricalDtype
from sklearn.neighbors import NearestNeighbors
from more_itertools import sort_together
from distinctipy import distinctipy as dp
from functools import partial
from pickleshare import PickleShareDB
import math
import zlib
import lmdb
import random
import itertools
import string
import pickle
import sys
import os
import warnings
#warnings.filterwarnings("error")
np.set_printoptions(suppress=True)
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
        '#8ff6ff',
        '#ff9f9c',
        '#2ded8d',
        '#fbffb3',
        '#ea68f2',
        '#7d26ff',
        ]
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', chexes)
#n = 0
#for c in chexes:
#    plt.plot([n, n+1], [n, n+1], color=c, label=c)
#    n += 1
#plt.legend()
#plt.show()

mzmlfile = '/home/sfo/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
#mzmlfile = '/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
basefolder = '/'.join((mzmlfile.split('/')[:-2]))
basefile = mzmlfile.split('/')[-1].split('.mzML')[0]

processinglocation = '/'.join((basefolder, 'fileprocessing', basefile))
fragmentlocation = '/'.join((basefolder, 'fileprocessing', basefile, 'fragments'))
scanalytelocation = '/'.join((basefolder, 'fileprocessing', basefile, 'scanalytegroups'))
librarylocation = '/home/sfo/data/proteomics/fastas/search-db'
csvfilename = '/'.join((processinglocation, 'fragment.matches'))
proteome = 'Human_Homo_sapien'
nprocs = 8
proton = 1.007276554940804
dividingthreshold = 0.8
ppmtol = 25
ppmmod = ppmtol / 1000000

def intersection_merge(mergable_items):
    sn = 0
    itemgroups = defaultdict(set) #group: [members]
    groupsofitems = {} #member: group
    for items in mergable_items:
        locs = set()
        for i in items:
            if i in groupsofitems:
                locs.add(groupsofitems[i])
        if locs:
            joiner = min(locs)
            if len(locs) > 1:
                for oldlocs in locs.difference([joiner]):
                    for ol in itemgroups[oldlocs]:
                        groupsofitems[ol] = joiner
                    itemgroups[joiner].update(itemgroups.pop(oldlocs))
        else:
            joiner = sn
            sn += 1
        itemgroups[joiner].update(items)
        for i in items:
            groupsofitems[i] = joiner
    return list(itemgroups.values())

scanalytefile = '/'.join((processinglocation, 'scanalytes.pickle'))
with open(scanalytefile, 'rb') as pick:
    scanalytecharges = pickle.load(pick)
#scanalytecharges = defaultdict(dict) #analyteid: scan: charge

#using scanalytecharges -> scan: analyte: charge
# - intersection merge by analyteid and scan
#   > use a scanalyte index, like what you seem to have in postfragprocessing, as analytes and scans are both just integers
# - make analysisgroups -> analyteid: analysisgroup
csvindex = 0
scanalyteindex = 0
#these new indices work as placeholders/trackers for logistical purposes
analyteidbyscanalyteindex = {} #scanalyte index: analyteid
scanbyscanalyteindex = {} #scanalyte index: scan
scanalyteindexbyanalyteid = {} #analyteid: scanalyte index
scanalyteindexbyscan = {} #scan: scanalyte index
mergableindices = set() #starts off as pairs of the two above newindices
for analyteid, scans in scanalytecharges.items():
    scanalytelist = []
    if analyteid in scanalyteindexbyanalyteid:
        oldindex = scanalyteindexbyanalyteid[analyteid]
        scanalytelist.append(oldindex)
    else:
        scanalyteindexbyanalyteid[analyteid] = scanalyteindex
        analyteidbyscanalyteindex[scanalyteindex] = analyteid
        scanalytelist.append(scanalyteindex)
        scanalyteindex += 1
    for scan in scans:
        if scan in scanalyteindexbyscan:
            oldindex = scanalyteindexbyscan[scan]
            scanalytelist.append(oldindex)
        else:
            scanalyteindexbyscan[scan] = scanalyteindex
            scanbyscanalyteindex[scanalyteindex] = scan
            scanalytelist.append(scanalyteindex)
            scanalyteindex += 1
    mergableindices.add(tuple(scanalytelist))

scanalytegroups = list(map(tuple, intersection_merge(mergableindices)))
print(len(scanalytegroups), 'scanalyte groups')

scanalytegroupsfile = '/'.join((processinglocation, 'scanalytegroups.pickle'))
with open(scanalytegroupsfile, 'wb') as pick:
    pickle.dump(scanalytegroups, pick)
#scanalytecharges = defaultdict(dict) #analyteid: scan: charge
