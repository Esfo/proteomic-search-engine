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
librarylocation = '/home/sfo/data/proteomics/fastas/search-db'
csvfilename = '/'.join((processinglocation, 'fragment.matches'))
proteome = 'Human_Homo_sapien-NoTremb'
nprocs = 8
proton = 1.007276554940804
dividingthreshold = 0.8
ppmtol = 25
ppmmod = ppmtol / 1000000

peptiderankingsfile = '/'.join((processinglocation, 'peptiderankings.csv'))
#distributionrankingsfile = '/'.join((processinglocation, 'distributionrankings.csv'))
#scanrankingsfile = '/'.join((processinglocation, 'scanrankings.csv'))

environment_partial = partial(lmdb.Environment, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=9999, max_spare_txns=1, lock=True)

#fulldecoysetfile = '/'.join((processinglocation, 'fulldecoyset.pickle'))
#with open(fulldecoysetfile, 'rb') as pick:
#    fulldecoyset = pickle.load(pick)
##fulldecoyset = set() #all decoy sequences

lineintensitiesofscansfile = '/'.join((processinglocation, 'lineintensitiesofscans.pickle'))
with open(lineintensitiesofscansfile, 'rb') as pick:
    lineintensitiesofscans = pickle.load(pick)
#lineintensitiesofscans = defaultdict(lambda: defaultdict(dict)) #scan: line: raw area of ms1 line, average of 2 flanking points

scoredscansfile = ''.join((processingdirectory, 'scored.ms2.pickle'))
with open(scoredscansfile, 'rb') as pick:
    scoredms2scans = pickle.load(pick)
#scoredms2scans = {} #scan: [[masses], [intensities], [ion scores]]

#peptideheaders = ['sequence', 'analyteid', 'score', 'ion_coverage', 'scan_indices']
#df = pd.read_csv(peptiderankingsfile)
#ddf = pd.read_csv(distributionrankingsfile)
#ldf = pd.read_csv(scanrankingsfile)

fullseqs = [i.encode() for i in set(df.loc[:,'sequence'])]

#seqsbyformula = {} #formula: [seqs]
proteinsofseqs = {} #seq: proteins
with environment_partial(librarylocation) as env:
    proteomedb = env.open_db((proteome + '.proteinsofseqs').encode())
    with env.begin(write=False) as txn:
        with txn.cursor(proteomedb) as cursor:
            for k, v in cursor.getmulti(fullseqs):
                proteinsofseqs[k.decode()] = eval(v)

fulldecoyset = set()
for seq, proteins in proteinsofseqs.items():
    isnotdecoy = any('decoy' not in i for i in proteins)
    if not isnotdecoy:
        fulldecoyset.add(seq)

cfile = '/home/sfo/store/flowcharacterizations/round3/crux-output/200901_fR_400.comet.txt'
pfile = '/home/sfo/store/flowcharacterizations/round3/crux-output/200901_fR_400.percolator.target.peptides.txt'
psmfile = '/home/sfo/store/flowcharacterizations/round3/crux-output/200901_fR_400.percolator.target.psms.txt'

cf = pd.read_csv(cfile, delimiter='\t')
pf = pd.read_csv(pfile, delimiter='\t')
psf = pd.read_csv(psmfile, delimiter='\t')

#pf.sort_values('q-value', inplace=True)
#cf.sort_values('xcorr score', inplace=True, ascending=False)
#df.sort_values('dg3', inplace=True, ascending=False)
#
#cf.loc[:,'isdecoy'] = cf.loc[:,'sequence'].apply(lambda x: x not in libraryseqs)
#pf.loc[:,'sequence'] = pf.loc[:,'peptide'].apply(lambda x: x.split('.')[1])

#DO do the combinatorics -> benefit scores of chimeric spectra via what matches together and also subtraction-benefit for whatever in the spectra isn't matched if its bad
#do the same for single-hit line-scan combos and subtract the bad things

#parameterize as much as possible then:
#DO ML, gridsearch + cross-validation for SVM, decision tree, random forest, etc, check a bunch i guess
#at all 3 levels, line/dist/analyte
#infer false positive rate from line/dists i guess?




#not everything matches perfectly
#this is because pandas is interpreting the values as floats -> precision errors
#ndf is being read as strings and im converting to floats in numpy -> more accuracy i guess

#for future analyses:
#best guess peptides
#best guess proteins
#the difference between the proteins able to be inferred from either dataset
#which ones are consistent vs different
#so you can compare peptide-derived protein probabilities vs spectra-protein probabilities i suppose
#^that's a lower and upper bound, and which direction 

#final columns as:
#number of ms1 charge states
#number of ms1 charge states sampled via ms2

#acceptances = [] #[[seq, analyteid, score, ioncoverage, scanindices, false discovery count, rank, is_decoy, decoy_first], ...]
#decoylist = []
#decoyfirst = {} #analyteid: True/False if it was ID'd as a decoy before anything else, tracking which analytes were decoys first
#
#thresholds = [] #all scores
#fdrs = []
#abovethreshold = []
#decoycount = 0
#targetcount = 0
#
#df.sort_values('score', inplace=True)
##with open(peptiderankingsfile, 'r') as rankings:
##headers = rankings.readline()
##for n, row in enumerate(rankings.readlines()):
#for n, row in enumerate(df.itertuples()):
#    #seq, analyteid, score, analytegeometry, ioncoverage, scanindices = row.strip().split(',')
#    index, seq, analyteid, ioncoverage, scanindices, score, analytegeometry, subformulashift = row
#    seq = row.sequence
#    analyteid = int(analyteid)
#    score = float(score)
#    isdecoy = False
#    if seq in fulldecoyset:
#        isdecoy = True
#        decoycount += 1
#    else:
#        targetcount += 1
#        fdrs.append(decoycount / targetcount)
#        abovethreshold.append(targetcount)
#        thresholds.append(score)
#    if analyteid not in decoyfirst:
#        decoyfirst[analyteid] = isdecoy
#    output = [seq, ioncoverage, scanindices, analyteid, score, analytegeometry, decoycount, n, isdecoy, decoyfirst[analyteid]]
#    if isdecoy:
#        decoylist.append(output)
#    else:
#        acceptances.append(output)
#
##i'll probably end up using range density in this?
#
#postacceptances = np.array([i[4:8] for i in acceptances])
#postdecoylist = np.array([i[4:8] for i in decoylist])
#
#acceptancescores = postacceptances[:,0]
#decoyscores = postdecoylist[:,0]
#
#acceptancescores = acceptancescores.astype(float)
#
#acounts = postacceptances[:,1]
#aranks = postacceptances[:,2]
#
#dlen = len(decoylist)
#pvalues = [i[4] / dlen for i in acceptances]

#score -> acts as threshold
#n false above thresh / n true above thresh

#line/dist level rankings?
#subformula abundance ratios can be done sorted, plain, and max-diff'd
#geometry can be done at the qualrank/subformula level and used as false discoveries when the qualrank isn't as good as the higher isotope
#geometry can be done at the line/scan level and checked across intensity input for best coverage
#intensity dispersion can be done at all levels pretty simply
#check average ppm error, weighted by intensity i guess

#if all of this doesn't show much difference
#then you need to score the scans and determine which ions are worth matching -> this would also speed a lot of things up!

#i also need to check this without the supersets tbh...

#ok so here's the thing
#EVERYTHING needs to discriminate
#i need a way to check for both good and bad guesses in every single metric
#and what i have isn't working
#i need more positive reinforcement for goodies
#and more punishment for bad ones
#otherwise nothing is going to work
#i need to go deeper in complexity in every area

#ppm error -> im weighting by intensity, but really i need to not pick bad ionscores
#if a subformulas ions BEST ionscores are shit, then keep them, but there needs to be an in-group threshold for each scan that's searched to push bad matches away, or the worst matches of that seq for that scan
#remember, fr-400 has only ~5k good ID's via crux
#how many good scans do i think there are by my measurements?

#what im getting wrong in sequence_geometry is that im not getting bad matches
#there should be MORE bad matches (poor ionscores) for false hits
#i don't need to hyper-differentiate the best matches, which is all i'm doing in that simulation
#time to make a simulation for fragment identification now

#a non-discriminating aspect i've implemented
#is that cross-scan score multiplication of specific ions
#those ions are always going to be there
#and if decoys match them that just makes the decoys score higher
#which then becomes a normal phenomenon
#and i can't have that

#addition-combining
#number of ions matched, single integer, always positive, > 1
#ion scores, ranging from negative to positive
#sequence geometry: remains the same, its solid
#sequence geometry needs to be the core of the score, and the ppm error and ionscores will either make or break it
#i want the ionscore to interact with ppm error in order to determine how the score geometry is affected by them
#total ppm is always turned into a negative value, then add the total ionscore
# - postive ionscore overpowers small ppm
# - negative ionscore makes small and large ppm worse off
#^meaning ionscore values should be determined based on ppm tolerance
#i want to correlate matched fragment intensities with MS1 input and apply this across the board post-collection based on how reasonable the MS2 intensities seem to be
#generate the maximum score of any match that cuts off by ppm error, descending in order of ion scores?
#   - maybe it should be processed in order of ACTUAL intensity, and cut off by ppm?

#up next:
#spit out the 3 different analytegeometries, maybe 4 if you want to do the 3rd one two ways
#from the output of the data, re-scale the geometries to match the scores? or whatever

#areas = []
#for l in ldf.itertuples():
#    areas.append(lineintensitiesofscans[l.scan][l.line])
#
##this is a shit correlation
#ldf.loc[:,'line_areas'] = areas

df.loc[:,'nions'] = df.loc[:,'ion_coverage'].apply(lambda x: x.count('/') + 1)
df.loc[:,'isdecoy'] = df.loc[:,'sequence'].apply(lambda x: x in fulldecoyset)
#check for overlap between sequences across cf/df

#ioncounts = df.loc[:,'ion_coverage'].apply(lambda x: x.count('/'))
#df.loc[:,'avg_ppm'] = df.loc[:,'analyte_ppm_error'] / ioncounts
#df.loc[:,'avg_ionscore'] = df.loc[:,'analyte_ion_score'] / ioncounts

meandiff = df.loc[:,'ag3'].mean() - df.loc[:,'analyte_ppm_error'].mean()
df.loc[:,'ag3-shift'] = df.loc[:,'ag3'] + meandiff

#ag = df.loc[:,'ag3'].to_numpy()
#z_scores = (ag - ag.mean()) / ag.std()
#df.loc[:,'fs'] = (z_scores * df.loc[:,'analyte_ppm_error'].std()) + df.loc[:,'analyte_ppm_error'].mean()

df.loc[:,'fs'] = df.loc[:,'ag3-shift'] + df.loc[:,'analyte_ion_score'] - df.loc[:,'analyte_ppm_error']
df.loc[:,'rs'] = df.loc[:,'analyte_ion_score'] / df.loc[:,'analyte_ppm_error']
#df.loc[:,'fs'] = df.loc[:,'ag3'] + df.loc[:,'avg_ionscore'] - df.loc[:,'avg_ppm']

df.loc[:,'error_per_ion'] = df.loc[:,'analyte_ppm_error'] / df.loc[:,'nions']
df.loc[:,'score_per_ion'] = df.loc[:,'analyte_ion_score'] / df.loc[:,'nions']
df.loc[:,'es'] = df.loc[:,'score_per_ion'] / df.loc[:,'error_per_ion']**2

df.sort_values('fs', inplace=True, ascending=False)

inds = df.loc[:,'sequence'].apply(lambda x: x in fulldecoyset)
de = df.loc[inds]
te = df.loc[~inds]

sorter = 'fs'
#sorter = 'ss'
#sorter = 'analyte_ion_score'
#sorter = 'ag3-shift'
#sorter = 'ag2'

#analyteiter = sorted(df.loc[:,(sorter, 'sequence')].to_numpy().tolist())
analyteiter = sorted(df.loc[:,(sorter, 'sequence')].to_numpy().tolist(), reverse=True)
#analyteiter = cf.loc[:,('xcorr score', 'sequence')].to_numpy().tolist()

decoycount = 0
targetcount = 0
acceptances = []
decoylist = []
for score, seq in analyteiter:
    isdecoy = False
    if seq in fulldecoyset:
        isdecoy = True
    if isdecoy:
        decoycount += 1
        decoylist.append([score, seq])
    else:
        targetcount += 1
        acceptances.append([decoycount / targetcount, score, seq])
fdrs = [i[0] for i in acceptances]
thresholds = [i[1] for i in acceptances]
dthresh = [i[0] for i in decoylist]

fcount = 0
acounts = []
fthresh = []
lastmatch = 0
for f in fdrs:
    if f == lastmatch:
        fcount += 1
    else:
        if lastmatch > 0:
            acounts.append(fcount)
            fthresh.append(lastmatch)
        lastmatch = f
        fcount += 1

plt.plot(fthresh, acounts,'.')
plt.show()

maximizers = ['analyte_ion_score', 'analyte_match_intensity', 'ag1', 'ag2', 'ag3', 'nions', 'ag3-shift', 'fs', 'rs', 'score_per_ion', 'es']
minimizers = ['analyte_ppm_error', 'error_per_ion']

ntrials = 0
victories = defaultdict(int) #column: count of target wins
for d, subf in df.groupby('analyteid'):
    for m in maximizers:
        subf.sort_values(m, ascending=False, inplace=True)
        isdecoy = subf.iloc[0]['isdecoy']
        if not isdecoy:
            victories[m] += 1
    for m in minimizers:
        subf.sort_values(m, inplace=True)
        isdecoy = subf.iloc[0]['isdecoy']
        if not isdecoy:
            victories[m] += 1
    ntrials += 1
