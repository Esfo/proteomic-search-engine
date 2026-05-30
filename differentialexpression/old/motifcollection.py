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

patternstring = 'full_drosophila' #these should probably be generated from the isoform-inclusive proteome, the more data: the better

patternfile = ''.join((patternfolder, patternstring, str(pmin), '_', str(pmax), '.patterns.pickle'))

fasta = SeqIO.parse(open(proteome), 'fasta')
seqs = {}
for fasta in fasta:
    sequence, idn = str(fasta.seq), fasta.id
    seqs[idn] = sequence

proteins = seqs.keys()
sequencelist = '|'.join((seqs.values()))

patternspace = np.linspace(pmin, pmax, pmax-pmin+1).astype(int)
per = '.'
ex = '[^|]'

rt = time()
if not os.path.isfile(patternfile):
    print(f'Making {patternfile}')
    patterns = []
    for p in patternspace:
        print(f'Pattern Size: {p}')
        #func = lambda s: [s[i:i+p] for i in range(len(s)) if len(s[i:i+p]) == p]
        #splits = list(map(func, sequencelist)) #hella slow
        #splits = list(itertools.chain(*splits)) #part of ^
        #splits = re.findall(fr'(?=([^|]{{{p}}}))',s) #not as fast
        splits = [sequencelist[i:i+p] for i in range(len(sequencelist)-p+1)]
        splits = [i for i in splits if '|' not in i]
        splitcount = Counter(splits)
        mv = np.asarray(list(splitcount.values()))
        
        t = Counter(mv)
        v = np.asarray([l*i for l, i in t.items()])
        ti = np.asarray([i for i in t.keys()])
        v = v[ti.argsort()]
        ti.sort()
        cv = np.cumsum(v[::-1])[::-1]
        cd = np.diff(v) / np.diff(cv)
        try:
            ci = np.diff(cd).argmin() + 1
        except ValueError:
            ci = 1 #There's only two things in t
        commonint = (mv >= ti[ci]).sum()

        endmin = splitcount.most_common(commonint)[-1][1]
        endmax = splitcount.most_common(commonint)[0][1]
        matchend = [i[0] for i in splitcount.most_common(commonint)]
        patterns.extend(matchend)
        print(f'Full match range: {endmin} - {endmax}')
        print(f'Full match quantity: {len(matchend)} / {len(splitcount)}')

        if p > 2:
            splits = [f'{(p-2)*per}'.join((i[0], i[-1])) for i in splits]
            splitcount = Counter(splits)
            mv = np.asarray(list(splitcount.values()))

            t = Counter(mv)
            v = np.asarray([l*i for l, i in t.items()])
            ti = np.asarray([i for i in t.keys()])
            v = v[ti.argsort()]
            ti.sort()
            cv = np.cumsum(v[::-1])[::-1]
            cd = np.diff(v) / np.diff(cv)
            try:
                ci = np.diff(cd).argmin() + 1
            except ValueError:
                ci = 1 #There's only two things in t
            commonint = (mv >= ti[ci]).sum()

            endmin = splitcount.most_common(commonint)[-1][1]
            endmax = splitcount.most_common(commonint)[0][1]
            fillerend = [i[0] for i in splitcount.most_common(commonint)]
            #fillerend.extend(matchend)

            patterns.extend(fillerend)
            print(f'Skeleton match range: {endmin} - {endmax}')
            print(f'Skeleton match quantity: {len(fillerend)} / {len(splitcount)}')
        print('~')

    with open(patternfile, "wb") as pick:
        pickle.dump(patterns, pick)
print(time() - rt)


#Subsets:
#subsets can be determined via the proteome alone, then whether the subsets found in the data map to the entirety of that subset can be later realized: If all of the A,AA,AAA,AAAAA,AAAAAAAAH group is only shown as AA, then that can be spit out later as information.
#Seeing as subsets can be fuzzy, this might give a blurrier version of the analysis as an end result. It might be good to resample both a central sequence as well as the subset to compare the two.

#For AA-converted values for things like hydrophobicity:
#Subsets may be a bit trickier, but you can convert the clustered groups back into AA's, then find AA subsets to work with.
