import pandas as pd
from scipy import spatial, integrate, stats, special
import editdistance_s
from time import time
import os
from collections import Counter, defaultdict
import networkx
from networkx.algorithms.components.connected import connected_components
from Bio import SeqIO
import pickle
import itertools
import concurrent
import sys
import gc
import regex
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
plt.rcParams['figure.dpi'] = 300

#The aim of this is going to be, to show how certain peptides of the same AA composition tend to have certain orders of amino acids present within any given peptide. If they do.
#I also want to explore how much more abundant certain peptides are over others that share their same AA composition.

proteome = '/home/sfo/data/fastas/proteomes/Fly_Drosophila_melanogaster.fasta'

#patternfolder = '/store/drosophila/PXD005713/'
patternfolder = '/home/sfo/data/motifs/'
#end result prediction: as you increase pmax, you should get better predictions - but at longer computational times. The same general motifs should be found if the linkage and moving windows work the way I imagine them to, but perhaps some more specifics might be lost? Or maybe pmax has a maximum practical limit like 10 or something, where anything after is already something caught by linkage, etc.
pmin = 3 #min number of spaces between two AAs, anything less than 3 is contradictory to how the cutoff system works.
pmax = 10 #max
everyother = 1 #number of spaces to skip between the range of pmin and pmax as an array is made, using this because otherwise there's so much redundant info that it becomes a problem of scale.
addtwos = False #add NxxM sequences, with 2 spaces between each AA-pair, to the end list of sequences to consider
patternspace = range(pmin, pmax+1, everyother)

fasta = SeqIO.parse(open(proteome), 'fasta')
seqs = {}
for fasta in fasta:
    sequence, idn = str(fasta.seq), fasta.id
    seqs[idn] = sequence

proteinsbyseq = defaultdict(list)
for k, v in seqs.items():
    proteinsbyseq[v].append(k)

#eliminating duplicate protein entries
duplicates = [v for v in proteinsbyseq.values() if len(v) > 1]
for d in duplicates:
    trembs = [i for i in d if 'tr' in i]
    if len(trembs) == len(d):
        pass
    else:
        for de in trembs:
            seqs.pop(de)
            d.remove(de)
    if len(d) > 1:
        nums = [int(i.split('|')[1][-1]) for i in d]
        keeper = d[np.argmin(nums)]
        dels = [i for i in d if i != keeper]
        for de in dels:
            seqs.pop(de)

proteins = list(seqs.keys())
proteome = list(seqs.values())
joinedproteome = '|'.join((proteome))
aminoacids = list(set(itertools.chain(*proteome)))
aminoacidpairs = list(itertools.combinations_with_replacement(aminoacids, 2))

aminoacidcounts = Counter(''.join((proteome)))
countsum = sum(aminoacidcounts.values())

aminoacidfrequencies = {k:v/countsum for k, v in aminoacidcounts.items()}
aminoacidfrequencies = {k: v for k, v in sorted(aminoacidfrequencies.items(), key=lambda item: item[1], reverse=True)}

nothanks = ['|', 'X', 'U']

#def independent_skeletons(p, joinedproteome, nothanks):
#    #counts every p-length skeleton motif available in each protein
#    per = p*'.'
#    #collecting motifs in this manner assumes order of the sequence matters, ie G..T is different than T..G
#    splits = set(zip(joinedproteome[:-p], joinedproteome[p:]))
#    splits = [f'{per}'.join((i)) for i in splits if not any(r in i for r in nothanks)]
#    outlist = []
#    for s in splits:
#        skeletonmatches = Counter(regex.findall(s, joinedproteome, overlapped=True, concurrent=False))
#        for sm in tuple(skeletonmatches.keys()):
#            if any(r in sm for r in nothanks):
#                skeletonmatches.pop(sm)
#        outlist.append(skeletonmatches)
#    return outlist
#
#t = time()
#with concurrent.futures.ProcessPoolExecutor(8) as executor:
#    futures, skeletons = [], []
#    for p in patternspace:
#        futures.append(executor.submit(independent_skeletons, p, joinedproteome, nothanks))
#    for future in concurrent.futures.as_completed(futures):
#        skeletons.extend(future.result())
#print(time() - t)

def patternfind(p, joinedproteome, nothanks):
    splits = (joinedproteome[i:i+p] for i in range(len(joinedproteome)-p+1))
    splits = [i for i in splits if not any(r in i for r in nothanks)]
    return Counter(splits)


def wing_expansion(seq, proteome):
    containers = [i for i in proteome if seq in i]
    clen = len(containers)
    concounts = [len(regex.findall(seq, c, overlapped=True, concurrent=True)) for c in containers]
    if max(concounts) > 1:
        conindices = [[m.start() for m in regex.finditer(seq, i, overlapped=True)] for i in containers]
        conindices = list(itertools.chain(*conindices))
        containers = [[containers[i] for _ in range(concounts[i])] for i in range(len(containers))]
        containers = list(itertools.chain(*containers))
        clen = len(containers)
        matchinfo = np.asarray([[conindices[i], len(containers[i])] for i in range(len(containers))])
    else:
        matchinfo = np.asarray([(i.index(seq), len(i)) for i in containers])

    matchsides = np.hstack((matchinfo[:,0,None], np.diff(matchinfo)))
    matchmaxes = matchsides.max(axis=0)
    matchsum = matchmaxes.sum()
    pers = matchmaxes - matchsides
    abase = ['' for i in range(matchsum)]
    alignments = []
    for (l, r), c in zip(pers, containers):
        newbase = abase.copy()
        newbase[l:l+len(c)] = c
        alignments.append(newbase)

    seqind = matchinfo[0,0] + pers[0,0]
    seqcounts = np.asarray([len(set(i)) if all(i) else 2 for i in zip(*alignments)]) #takes ~half the time, only considering all sequences anyways
    matchingboundaries = np.argwhere(np.diff(seqcounts == 1)).flatten()
    if matchingboundaries.size > 1:
        if np.all(seqind < matchingboundaries):
            minbound = 0
        else:
            minbound = matchingboundaries[np.where(matchingboundaries <= seqind)[0].max()] + 1
        if np.all(minbound > matchingboundaries):
            maxbound = len(seqcounts)
        else:
            maxbound = matchingboundaries[np.where(matchingboundaries+1 >= seqind+len(seq))[0].min()] + 1
    else:
        if matchingboundaries[0] > seqind:
            minbound = 0
            maxbound = matchingboundaries[0] + 1
        else:
            minbound = matchingboundaries[0] + 1
            maxbound = len(seqcounts)

    trueseq = ''.join((alignments[0][minbound:maxbound]))
    return trueseq, minbound, maxbound, alignments


p = 5

splits = patternfind(p, joinedproteome, nothanks)
splitsum = sum(splits.values())


###these calculated probs here doesn't sum to 1 after p>=5, obviously if you took all the individual AAs out, they would.
#The problem essentially becomes a different beast at that point, and multiplying together individual AA frequencies becomes fruitless.
#probs, actuals = [], []
#for s, c in splits.items():
#    probs.append(np.prod([aminoacidfrequencies[i] for i in s]))
#    actuals.append(c / splitsum)
#
#probs = np.asarray(probs)
#actuals = np.asarray(actuals)
#
#plt.plot(probs, actuals, '.')
#plt.yscale('log')
#plt.xscale('log')
#plt.show()
#
#print(sum(actuals))
#print(sum(probs))
#~


#Group AAs by their probabilities, see how many peps are at each one.
probdict = defaultdict(dict)
for s, c in splits.items():
    prob = np.prod([aminoacidfrequencies[i] for i in s])
    probdict[prob][s] = c

probcounts = Counter(len(i) for i in probdict.values())

plt.bar(probcounts.keys(), probcounts.values())
plt.xlabel('# peptides at any given probability')
plt.ylabel('# of probabilities of x length')
plt.show()

problengths = {}
for pdk, pdv in probdict.items():
    problengths[pdk.round(10)] = len(pdv)

plt.scatter(problengths.keys(), problengths.values(), alpha=0.1)
plt.xlabel('probability')
plt.ylabel('length')
plt.show()

#What truly makes the case for a motif, though. Sequence? Are there motifs that rely more on composition and less on exact position?

for p, v in probdict.items():
    ranks = sorted(v.values())

trueseq, minbound, maxbound, alignments = wing_expansion(seq, proteome)

for c in alignments:
    print(''.join((c[minbound:maxbound])))

#after exploring a bunch of shit, basically: every repeated sequence found via these splits comes from an expanded sequence of some sort. They also tend to have other proteins in the mix that don't match any of the others very well outside of that sequence. This just backs up more of what my original thoughts were: There's pretty much too many sequences to handle in memory to have any type of analysis going.
#seeing as memory is such an open issue, it's definitely wise to keep track of what sequences can be found in what proteins on the fly
#what would be an interesting check is to pick out the splits that are usually filtered out and try to see if the sequences found there overlap with your typical filtering method.
