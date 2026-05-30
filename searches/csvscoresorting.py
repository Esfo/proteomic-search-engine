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
from functools import partial
from pickleshare import PickleShareDB
from decimal import Decimal, getcontext
import tempfile
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

import resource
softlimit, hardlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hardlimit-1, hardlimit))

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

peptiderankingsfile = '/'.join((processinglocation, 'peptiderankings.csv'))
testrankingsfile = '/'.join((processinglocation, 'testrankings.csv'))
#peptideheaders = ['sequence', 'analyteid', 'score', 'ion_coverage', 'scan_indices']

def yieldlines(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        for row in file.readlines():
            yield row

def csvsplit(inputfile, chunksize=1000000):
    #chunk csv, sort, save temp files
    with open(inputfile, 'r', encoding='utf-8') as originalcsv:
        headers = originalcsv.readline()
        tempfiles = []
        while True:
            n = 0
            rows = []
            finished = False
            try:
                while True:
                    row = originalcsv.readline()
                    sequence, analyteid, score, ion_coverage, scan_indices = row.strip().split(',')
                    score = Decimal(score)
                    rows.append([score, row])
                    n += 1
                    if n > chunksize:
                        break
            except ValueError:
                #file is finished
                finished = True
            if rows:
                #sort by score
                rows = list(zip(*sorted(rows)))[1]
                #save temp file
                tempcsv = tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8', delete=False)
                csvwriter = csv.writer(tempcsv)
                for row in rows:
                    csvwriter.writerow(row.strip().split(','))
                tempfiles.append(tempcsv.name)
                tempcsv.close()
            if finished:
                break
    return tempfiles, headers

def fileheaps(tempfiles, outputfile, headers):

    fileheap = [] #[[score, tempfile, row],...]
    fileyielders = {} #tempfile name: generator to yield its next row
    for t in tempfiles:
        #set up heaps of sorted files yielding lines
        fileyielders[t] = yieldlines(t)
        row = next(fileyielders[t])
        sequence, analyteid, score, ion_coverage, scan_indices = row.strip().split(',')
        score = Decimal(score)
        output = [score, t, row]
        heapq.heappush(fileheap, output)
        #fileheap.append(output)

    with open(outputfile, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers.strip().split(','))
        while fileheap:
            #write the latest sorted value to the csv
            score, t, row = heapq.heappop(fileheap)
            #score, t, row = fileheap.pop(fileheap.index(max(fileheap)))
            writer.writerow(row.strip().split(','))
            
            #keep track of the next latest line coming out of each file
            #yield the next value from the temp chunk that was just taken
            try:
                newrow = next(fileyielders[t])
            except StopIteration:
                #that file is done
                continue
            sequence, analyteid, score, ion_coverage, scan_indices = newrow.strip().split(',')
            score = Decimal(score)
            output = [score, t, newrow]
            heapq.heappush(fileheap, output)
            #fileheap.append(output)

tempfiles, headers = csvsplit(peptiderankingsfile)
fileheaps(tempfiles, testrankingsfile, headers)

for t in tempfiles:
    os.unlink(t)
