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

#mzmlfile = '/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
mzmlfile = '/home/sfo/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
#mzmlfile = '/home/sfo/store/data/PXD035665/mzMLs/GradientAmount_HeLa_4ug_240min_R01.mzML'
msrun = mzml.MzML(mzmlfile, dtype=np.float64)

basefolder = '/'.join((mzmlfile.split('/')[:-2]))
basefile = mzmlfile.split('/')[-1].split('.mzML')[0]

processinglocation = '/'.join((basefolder, 'fileprocessing', basefile))
librarylocation = '/home/sfo/data/proteomics/fastas/search-db'
#librarylocation = '/home/sfo/data/proteomics/fastas/iodoacetamide-search/'
proteome = 'Human_Homo_sapien'
nprocs = 8
subisotopomericdepth = 0.8
proton = 1.007276554940804
dividingthreshold = 0.1

environment_partial = partial(lmdb.Environment, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=9999, max_spare_txns=1, lock=True)

linepositionsbyformulafile = '/'.join((processinglocation, 'linepositionsbyformula.pickle'))
with open(linepositionsbyformulafile, 'rb') as pick:
    linepositionsbyformula = pickle.load(pick)
#linepositionsbyformula = defaultdict(lambda: defaultdict(set)) #formula: matchable library positions of a distribution: [lines that have ms2 scans]

seqsbyformula = {} #formula: [seqs]
with environment_partial(librarylocation) as env:
    seqdb = '.'.join(('seqsbyformula', proteome))
    seqs = env.open_db(seqdb.encode())
    with env.begin(write=False) as txn:
        with txn.cursor(seqs) as cursor:
            for k, v in cursor:
                key = k.decode()
                if key in linepositionsbyformula:
                    #minimal decoy generation necessary
                    value = eval(v.decode())
                    seqsbyformula[key] = value

def shuffle_string(s):
    char_list = list(s)
    random.shuffle(char_list)
    return ''.join(char_list)

def unique_permutations_count(sequence):
    # Count the frequency of each element in the sequence
    freq = Counter(sequence)

    # Calculate the total number of permutations
    total_permutations = math.factorial(len(sequence))

    # Divide by the factorial of the frequency of each element to account for repetitions
    for count in freq.values():
        total_permutations //= math.factorial(count)

    return total_permutations

#hasn't failed yet
#aas = 'PPEREERPEIPERIGPEIGGPGGG'
#total = 0
#while True:
#    randomseq = random.choices(aas, k=np.random.randint(5,12))
#    correctlength = len(set(itertools.permutations(randomseq)))
#    functionoutput = unique_permutations_count(randomseq)
#    if correctlength != functionoutput:
#        break
#    else:
#        print('woot')
#        total += 1

formulabysortedseq = {} #sortedseq: formula
seqsbysortedseq = defaultdict(set) #sortedseq: [seqs]
for formula, seqs in seqsbyformula.items():
    for seq in seqs:
        sortedseq = ''.join((sorted(seq)))
        seqsbysortedseq[sortedseq].add(seq)
        formulabysortedseq[sortedseq] = formula

#for now this only handles trytic peptides
#but i should note if c or n terminal AAs are relevant to the digest and handle it here
#make them kwargs and default them to false, it should either be a [0] or [-1] slice, if they exist then slice those out first via case/match
fulldecoyset = set() #all decoy sequences
seqswithdecoysbyformula = defaultdict(list) #formula: [seqs + decoys]
for sortedseq, seqs in seqsbysortedseq.items():
    decoys = set()
    slen = len(seqs)
    #make seqgroups based on first/last AA depending on enzyme
    seqgroups = defaultdict(lambda: defaultdict(list)) #position (0 or -1): AA: [seqs]
    for seq in seqs:
        #this would need to be:
        #if seq.startswith/endswith and make double groups for when both AAs apply
        seqgroups[-1][seq[-1]].append(seq)
    for position, aas in seqgroups.items():
        for aa, subseqs in aas.items():
            #if len(aa) > 1: double-group i suppose?
            initialseq = subseqs[0][:-1]
            setlen = len(set(initialseq))
            if setlen > 1:
                subdecoys = set()
                sublen = len(subseqs)
                permax = unique_permutations_count(initialseq) #the -1 is considering K or R ending is consistent
                for seq in subseqs:
                    #tryptic only atm
                    endchar = seq[-1]
                    shortseq = seq[:-1]
                    while True:
                        decoy = shuffle_string(shortseq) + endchar
                        if decoy not in subdecoys and decoy not in decoys and decoy not in seqs:
                            subdecoys.add(decoy)
                            break
                        if len(subdecoys) + sublen == permax:
                            #all potential sequences already made
                            break
                decoys.update(subdecoys)
            #else: #setlen == 1 and sublen == 1
                #the sequence only has one AA, no decoys possible, whatever
                #break
    seqs.update(decoys)
    fulldecoyset.update(decoys)
    seqswithdecoysbyformula[formulabysortedseq[sortedseq]].extend(seqs.copy())

seqswithdecoysbyformula = dict(seqswithdecoysbyformula)

seqswithdecoysbyformulafile = '/'.join((processinglocation, 'seqswithdecoysbyformula.pickle'))
with open(seqswithdecoysbyformulafile, 'wb') as pick:
    pickle.dump(seqswithdecoysbyformula, pick)
#seqswithdecoysbyformula = defaultdict(list) #formula: [seqs + decoys]

fulldecoysetfile = '/'.join((processinglocation, 'fulldecoyset.pickle'))
with open(fulldecoysetfile, 'wb') as pick:
    pickle.dump(fulldecoyset, pick)
#fulldecoyset = set() #all decoy sequences
