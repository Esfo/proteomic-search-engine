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

def patternfind(p, sequencelist):
    splits = [sequencelist[i:i+p] for i in range(len(sequencelist)-p+1)]
    splits = [i for i in splits if '|' not in i]
    if p > 2:
        skeletonsplits = [f'{(p-2)*per}'.join((i[0], i[-1])) for i in splits]
        return frequencyfilter(splits), frequencyfilter(skeletonsplits)
    return frequencyfilter(splits)

def frequencyfilter(splits):
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
    #matchend = [i[0] for i in splitcount.most_common(commonint)]
    return splitcount.most_common(commonint)

if not os.path.isfile(patternfile):
    t = time()
    print(f'Making {patternfile}')
    with concurrent.futures.ProcessPoolExecutor(4) as executor:
        prs = []
        for p in patternspace:
            prs.append(executor.submit(patternfind, p, sequencelist))
        patterns = {}
        for future in concurrent.futures.as_completed(prs):
            out = future.result()
            if type(out) is tuple:
                for o in out:
                    patterns.update(o)
            else:
                patterns.update(out)
    with open(patternfile, "wb") as pick:
        pickle.dump(patterns, pick)
    print(time() - t, '- Motifs collected')
else:
    with open(patternfile, "rb") as pick:
        patterns = pickle.load(pick)
        print('loaded', patternfile)

def motifcounting(p, sequencelist):
    return p, sequencelist.count(p)

def skeletoncounting(p, sequencelist):
    finds = re.findall(p, sequencelist)
    finds = len([i for i in finds if '|' not in i])
    return p, finds

skeletonmotifs, motifs = [], []
for p in patterns.keys():
    if '.' in p:
        skeletonmotifs.append(p)
    else:
        motifs.append(p)

aminoacids = 'ACTYKNMPWSQGHFDIVERL'
aminoacids = [i for i in aminoacids]
aminoacidfrequencies = {i:sequencelist.count(i) for i in aminoacids}
aasum = sum(aminoacidfrequencies.values())
aminoacidfrequencies = {i:v/aasum for i, v in aminoacidfrequencies.items()}


mfs = motifs[:100]
skms = skeletonmotifs[:100]

t = time()
patterncount = {}
for m in mfs:
    patterncount[m] = sequencelist.count(m)
print(time() - t)


t = time()
n = 0
with concurrent.futures.ProcessPoolExecutor(4) as executor:
    futures = []
    for m in skms:
        futures.append(executor.submit(skeletoncounting, m, sequencelist))
    patterncount = {}
    for future in concurrent.futures.as_completed(futures):
        out = future.result()
        patterncount[out[0]] = out[1]
        n += 1
        if not n % 1000:
            print(n)
print(time() - t)

bones = {}
for s in reversed(skeletonmotifs):
    finds = re.findall(s, sequencelist)
    finds = [i for i in finds if '|' not in i]
    #now resample peptides at the same length as the finds to see if you find any significant sequences that don't resample here.

#Subsets:
#subsets can be determined via the proteome alone, then whether the subsets found in the data map to the entirety of that subset can be later realized: If all of the A,AA,AAA,AAAAA,AAAAAAAAH group is only shown as AA, then that can be spit out later as information.

#For AA-converted values for things like hydrophobicity:
#Subsets may be a bit trickier, but you can convert the clustered groups back into AA's, then find AA subsets to work with.
#The basis of using these is to discover AA sequences that converge with chemical patterns of other AAs without matching amino acids specifically.

#Exploration:
#How would this pattern look in terms of sequence data:
#~Examle 1
#            EE
#           AEEF
#          RAEEFP
#         IRAEEFPG
#                  etc
#As it keeps expanding, what are the amount of AEEFs that are NOT involved with RAEEFP? As in, they have different flanking AA's, and what are the amount that have at least one flanking AA belonging to that pattern?
#Is this a pattern of binding affinity?

#link AA's from the bottom up, start at the smallest and see how many other larger motifs they're in, then call that a potential 'start' of a group. It doesn't have to begint to define the entire group - but this will, by definition separate individual AA's into wide groups of AAs so that the group identity matters more than the individual. In essence, two different AA sequences can belong in different end-result motifs from this process.
#   > You would look for every 2-length combo (2l) in every n-length combo (nl), then do the same with the 3l combos, even if a layer is missed - ie something is found in 2l and 3l, then skipped at 4l and the combo reemerges in 5l - this should still be able to carry on given that you move forward with every map that works.

#The question becomes: Do you look for sequences that are there, or sequences that aren't there? In essence, would I expect a highly conserved sequence to have a lot of 'immitator sequences' around it in the pool of the proteome? Or would I expect a more 'one and only sequence' dichotomy among sequences that are strangely abundant.

#two main things need to be done atm:
# 1. Compare splits for n-length sequences, within both a shuffled-protein and resampled-proteome (based on AA frequencies), to those derived from actual sequences.
#   > Anything strange? Could this perhaps be used as a better way of picking sequences either by resampling or by frequency-comparison?
#   > 3 Ways to compare this:
#       >> Proteome Resampling via AA frequency
#           >>> Could also be done at the individual protein frequency-level
#       >> Protein Shuffling
#       >> You can also resample splits of every n-peptides (+ some leftover) to look at how every n-length sequence gets distributed across the proteome. This gives 2 things:
#           >>> The probability that these sequences are distributed across this many proteins in such a way?
#           >>> The distribution of where the motifs go in terms of %-Distance to N-/C-term
#               >>>> For the distribution of distance, you would only need to resample the proteins that have the motif, and resample by n-length splits.
# 2. Visualize a few of the majorly abundant sequences of a few n-length sequence lengths, find a way to visualize sequence similarity popularities. You can do it.
#   > You have the sequence, distance from the sequence, and every possible AA as visualizable factors.

#How many times do you REALLY have to resample an entire proteome based on an AA frequency? You're essentially resampling a combination of 20 different things, so it's not like you're underpowered when it comes to variation in that corner. Just one resampling of a proteome wouldn't create replicates of an individual hypothetical-protein, but would those be necessary to observe patterns across the entire proteome?a An exploration into just how many new proteomes you need to create is necessary in order to determine the depth of resampling needed here, if any.
#   > And conserved motifs should stick out like a sore thumb, especially if they follow the pattern in example 1.

def stringexpand(string, outer=1, minconserved=3):
    variablespots = len(string) - minconserved

    outers = outer * '.'
    mid = [''.join((i, '.?')) for i in string]
    mid = [i for i in string]

    ecs = [mid.copy()]
    tc = list(range(len(string)))
    for vs in range(1, variablespots+1):
        for mps in itertools.combinations(tc, vs):
            cs = mid.copy()
            for m in mps:
                cs[m] = cs[m].replace(string[m], '.')
            ecs.append(cs)
    for e in ecs:
        while True:
            if e[0].endswith('.'):
                del e[0]
            elif e[-1].endswith('.'):
                del e[-1]
            else:
                break

    estrings = [''.join((outers, ''.join((e)), outers)) for e in ecs]
    #estrings = [''.join((e)) for e in ecs]
    return sorted(estrings, key=lambda x: len(x), reverse=True)

estrings = stringexpand('SLAVE')
def ff(estrings, sequencelist):
    o = []
    for est in estrings:
        ecomp = re.compile(est)
        o.append(len(ecomp.findall(sequencelist)))
    return o

nc = re.compile('|'.join((estrings)))
tre = TRE(*estrings)
pattern = re.compile(tre.regex())

#need to replace a .?_ to ._? then _.? where _ is a non-substituted part of the total string. Two cycles for a single 'replacement character', which is [len(string) - match]. So 2 of those double-things in my case. here.
#^This should be a moving window, combinations might be bad. Allowing for multiple width-input, or widths up to some number would be the appropriate implementation.

#maybe just use awk/sed/grep? are these faster?? perhaps not at splitting the proteome


#first search the generated patterns list for your split group of string/trie things, then search the proteome for the string/trie things of that initial subset to get a larger subset.
#center of the visualization should be determined by AA frequency, like in 'SLAVE', where 'LA' is more frequent than 'AV', or something, and 'L' is more frequent than 'A'.



