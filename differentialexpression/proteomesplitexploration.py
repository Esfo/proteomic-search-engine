import pandas as pd
from time import time
import os
from collections import Counter, defaultdict
import multiprocessing as mp
from Bio import SeqIO
import pickle
import itertools
import concurrent
import random
import sys
import gc
import re
from blist import blist
import matplotlib.pyplot as plt
import matplotlib
sys.path.append('/home/sfo/camp')
from sequencetools import seqsplit_dict
from scipy import stats
import numpy as np

proteome = '/home/sfo/data/fastas/proteomes/Fly_Drosophila_melanogaster.fasta'

#patternfolder = '/store/drosophila/PXD005713/'
patternfolder = '/home/sfo/data/motifs/'
pmin = 2
pmax = 100

patternstring = 'full_drosophila' #these should probably be generated from the isoform-inclusive proteome, the more data: the better?

patternfile = ''.join((patternfolder, patternstring, '_', str(pmin), '-', str(pmax), '.patterns.pickle'))

fasta = SeqIO.parse(open(proteome), 'fasta')
seqs = {}
for fasta in fasta:
    sequence, idn = str(fasta.seq), fasta.id
    seqs[idn] = sequence

proteins = seqs.keys()
sequencelist = '|'.join((seqs.values()))

patternspace = np.linspace(pmin, pmax, pmax-pmin+1).astype(int)
per = '.'

def scount(p, sequencelist):
    splits = [sequencelist[i:i+p] for i in range(len(sequencelist)-p+1)]
    splits = [i for i in splits if '|' not in i]
    return Counter(splits)

#t = time()
#with concurrent.futures.ProcessPoolExecutor(8) as executor:
#    prs = []
#    for p in patternspace:
#        prs.append(executor.submit(scount, p, sequencelist))
#    splitcounts = {}
#    for future in concurrent.futures.as_completed(prs):
#        fp, fsl = future.result()
#        splitcounts[fp] = fsl
#print(time() - t)
#
#cf = pd.DataFrame.from_dict(splitcounts, orient='index')
#cf.columns = ['nsplits']
#cf.sort_index(inplace=True)
#
#cf.plot.line()
#based on this, going up to a pmax of 20 is acceptable enough to recover every pattern, however it would be cool to also analyze larger patterns later on once the trie structure is made in order to look at larger patterns
#because patterns should just repeat after a while, there may be an interesting way of pulling the tree structure into another form, perhaps circular, that can recognize patterns from the repeating nature of the trie



aminoacids = scount(1, sequencelist)
patterns = scount(20, sequencelist)
p2 = scount(19, sequencelist)

pf = pd.DataFrame.from_dict(patterns, orient='index')
pf.columns = [20]

#_trie = lambda: defaultdict(_trie)
#trie = _trie()
##for s in ["cat", "bat", "rat", "cam"]:
#for s, count in patterns.items():
#    curr = trie
#    for c in s:
#        curr = curr[c]
#    curr[c]['#'] = count
#    #curr.setdefault("_end")

#design a class that can hold both a dict with a new value, and a counted element
