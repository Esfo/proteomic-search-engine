import pandas as pd
from time import time
import os
from collections import Counter, defaultdict
from random import sample
from Bio import SeqIO
import pickle
import multiprocessing as mp
import itertools
import concurrent
from operator import itemgetter
import random
import sys
import gc
import re
import matplotlib.pyplot as plt
import matplotlib
sys.path.append('/home/sfo/camp')
from sequencetools import seqsplit_dict
from scipy import stats
import numpy as np
from statsmodels.stats import multitest
plt.rcParams['figure.dpi'] = 300

patternfile = '/store/drosophila/PXD005713/full4.patterns.pickle'
#patternfile = '/home/sfo/data/motifs/full_drosophila2_100.patterns.pickle'

proteomefile = '/home/sfo/data/fastas/proteomes/Fly_Drosophila_melanogaster.fasta'

fasta = SeqIO.parse(open(proteomefile), 'fasta')
seqs = {}
for fasta in fasta:
    sequence, idn = str(fasta.seq), fasta.id
    seqs[idn] = sequence

proteinlabels = {o:n for o, n in enumerate(seqs.keys())}
proteinindex = list(proteinlabels.keys())
proteome = list(seqs.values())

with open(patternfile, "rb") as pick:
    motifpatterns = pickle.load(pick)
    print('loaded', patternfile)

nproteins = 1000
nsamples = 10000

def psample(proteinindex, nsamples, nproteins):
    proteinresamples = []
    for n in range(nsamples):
        proteinresamples.append(sample(proteinindex, nproteins))
    return np.asarray(proteinresamples)

def pmindex(proteomemotifs, proteinresamples, p):
        return p, proteomemotifs[proteinresamples].sum(axis=1)

def motifcounting(motifpatterns, proteome, proteinresamples, procs=3):
    with concurrent.futures.ThreadPoolExecutor(procs) as executor:
        futures = []
        for p in motifpatterns:
            proteomemotifs = np.asarray([i.count(p) for i in proteome])
            futures.append(executor.submit(pmindex, proteomemotifs, proteinresamples, p))
        motifs, resamples = [], []
        for future in concurrent.futures.as_completed(futures):
            m, r = future.result()
            motifs.append(m)
            resamples.append(r)
    return motifs, np.asarray(resamples)

motifpatterns = [i for i in motifpatterns if '.' not in i]
motifpatterns = motifpatterns[:100]

proteinresamples = psample(proteinindex, nsamples, nproteins)

t = time()
motifs, resamples = motifcounting(motifpatterns, proteome, proteinresamples)
print(time() - t)

rs = pd.DataFrame(resamples, index=motifs)

#take all motifs that are present in your sample
#have a 2d plot with one axes being p-value, and the other being some combination of either number of motifs in sample, number of motifs in proteome, or a ratio of the two, color can represent density. Potentially, the second axis could also be a visualization of how concentrated a motif is %-wise to the c- or n-terminal.
#Get skeleton motifs working in here
