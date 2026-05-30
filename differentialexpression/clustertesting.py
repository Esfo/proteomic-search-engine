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
import editdistance_s
from Bio import SeqIO
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

proteomefile = '/home/sfo/data/proteomics/fastas/proteomes/Human_Homo_sapien.fasta'
minsplitlength = 2

fasta = SeqIO.parse(open(proteomefile), 'fasta')
seqs = {}
for fasta in fasta:
    sequence, idn = str(fasta.seq), fasta.id
    seqs[idn] = sequence

#iterate through each length split individually
#dont need to compare seqs of a different length
#compare via levenstein
#make a cutoff for each split length
#begin to link sequences of proteins that fit within the cutoff
#map the links across split lengths
#eventually at long enough split lengths the hierarchy should form
#more sense should show up at longer split lengths
#that sense might dropoff when a gap emerges
#but maybe that in itself is visibility?
#also i don't think i'll cluster at the codon level because i want the clustering to be relevant to the biochemistry, so similar AAs should be favored over codons, not that the former is really implemented through this idea

#editdistance_s.distance(b1,b2) #levenshtein

maxsplitlength = max(len(v) for v in seqs.values()) - 1

seqsbysplit = defaultdict(list) #split seq: protein --- somehow using a set is heavier in memory?? despite massive redundancy
for splitlen in range(minsplitlength, maxsplitlength):
    for pid, seq in seqs.items():
        splits = [seq[n:n+splitlen] for n, i in enumerate(range(len(seq)-splitlen+1))]
        for split in splits:
            seqsbysplit[split].add(pid)
#this has massive memory overhead
