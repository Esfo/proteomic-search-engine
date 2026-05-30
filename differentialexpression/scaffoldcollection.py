from scipy import spatial, integrate, stats, special
import pandas as pd
import editdistance_s
from time import time
import os
from collections import Counter, defaultdict
import networkx
from networkx.algorithms.components.connected import connected_components
from Bio import SeqIO
import itertools
import concurrent
import sys
import gc
import regex
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
plt.rcParams['figure.dpi'] = 300

#https://github.com/life4/textdistance
#^this may be useful

#all notes and examples are based on the drosophila proteome
proteomefile = '/home/sfo/data/fastas/proteomes/Fly_Drosophila_melanogaster.fasta'

#patternfolder = '/store/drosophila/PXD005713/'
patternfolder = '/home/sfo/data/motifs/'
pname = 'drosophila'
#end result prediction: as you increase pmax, you should get better predictions - but at longer computational times. The same general motifs should be found if the linkage and moving windows work the way I imagine them to, but perhaps some more specifics might be lost? Or maybe pmax has a maximum practical limit like 10 or something, where anything after is already something caught by linkage, etc.
pmin = 3 #min number of spaces between two AAs, anything less than 3 is contradictory to how the cutoff system works.
pmax = 23 #max
everyother = 2 #number of spaces to skip between the range of pmin and pmax as an array is made, using this because otherwise there's so much redundant info that it becomes a problem of scale.
ncores = 8

patternspace = range(pmin, pmax+1, everyother)
fname = '_'.join((pname, '-'.join((list([str(i) for i in patternspace]))), 'motifs'))

fasta = SeqIO.parse(open(proteomefile), 'fasta')
seqs = {}
for fasta in fasta:
    sequence, idn = str(fasta.seq), fasta.id
    seqs[idn] = sequence

#UGH, most examples were made before I realized this, I hope they still exist?
#there is actually a need to clean up the data and remove duplicate protein entries while preserving the swissprot data over tremble. Should I do this cleanup though? Some proteins might be legitemate duplicates...
#len(seqs.values())
#Out[498]: 23450
#len(set(seqs.values()))
#Out[499]: 22326
#Need to do this because far down below, where I'm elongating sequences to their proper length, it ends up finding sequences that are only present in duplicates, then expanding to the entire length of the protein because it's only found in those duplicates. Then the entire protein is seen as a valid sequence. Which is a bit nonsensical.
proteinsbyseq = defaultdict(list)
for k, v in seqs.items():
    proteinsbyseq[v].append(k)

duplicatelibrary = {} #keeping track of these because the final motif output includes what proteins they're in
duplicates = [v for v in proteinsbyseq.values() if len(v) > 1] 
for dc in duplicates: 
    d = dc.copy()
    currentdups = set()
    trembs = [i for i in d if 'tr' in i]
    currentdups.update(trembs)
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
        currentdups.update(dels)
        for de in dels:
            seqs.pop(de)
    else:
        keeper = d[0]
    duplicatelibrary[keeper] = currentdups

proteins = list(seqs.keys())
proteome = list(seqs.values())
joinedproteome = '|'.join((proteome))
aminoacids = list(set(itertools.chain(*proteome)))
aminoacidpairs = list(itertools.combinations_with_replacement(aminoacids, 2))

aminoacidcounts = Counter(''.join((proteome)))
countsum = sum(aminoacidcounts.values())

aminoacidfrequencies = {k:v/countsum for k, v in aminoacidcounts.items()}
aminoacidfrequencies = {k: v for k, v in sorted(aminoacidfrequencies.items(), key=lambda item: item[1], reverse=True)}
aminoacidfrequencyorder = [i for i in aminoacidfrequencies.keys()]

nothanks = ['|', 'X', 'U']
#^it would be nice to instead generate all the appropriate peptide combinations that involve these rather than excluding them, I think U was one thing or another, what was X again?

#aminoacidlocationdict = defaultdict(list)
#for protein in proteome:
#    plen = len(protein)
#    for n, aa in enumerate(protein):
#       aminoacidlocationdict[aa].append(n/plen) 
#
#for aa in aminoacidlocationdict.keys():
#    plt.hist(aminoacidlocationdict[aa], bins=300)
#    plt.title(aa)
#    plt.show()

#from https://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements
def to_graph(l):
    G = networkx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G

def to_edges(l):
    """
        treat `l` as a Graph and returns it's edges
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current

#def patternfind(p, joinedproteome, nothanks):
#    splits = [joinedproteome[i:i+p] for i in range(len(joinedproteome)-p+1)]
#    splits = [i for i in splits if not any(r in i for r in nothanks)]
#    return frequencyfilter(splits)
#
#def frequencyfilter(splits):
#    #this acts as a filter for where the largest difference in the data lies
#    #plt.hist(counts, bins=20)
#    #^shows the same as:
#    #plt.hist(diffmatrixsums, bins=20)
#    #but the x-axis is shifted, anything with a count above zero here is kept
#    matches = Counter(splits)
#    matchcounts = Counter(matches.values())
#    counts, freqs = np.asarray(list(zip(*matchcounts.items())))
#    freqs = freqs[counts.argsort()]
#    counts = np.sort(counts)
#    #diffmatrixsums = (freqs - freqs.reshape(-1,1)).sum(axis=0)
#    #cutoff = counts[diffmatrixsums < 0].min()
#    cfareas = integrate.cumtrapz(counts*freqs, x=counts, initial=0)
#    diffmatrixsums = (cfareas - cfareas.reshape(-1,1)).sum(axis=0)
#    cutoff = counts[diffmatrixsums > 0].min()
#    output = [k for k, v in matches.items() if v >= cutoff]
#    return output
#
##multithreading may be better here
#t = time()
#with concurrent.futures.ProcessPoolExecutor(8) as executor:
#    futures, patterns = [], []
#    for p in patternspace:
#        futures.append(executor.submit(patternfind, p, joinedproteome, nothanks))
#    for future in concurrent.futures.as_completed(futures):
#        patterns.append(future.result())
#print(time() - t)

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
#        outlist.append(frequency_filter(skeletonmatches))
#    return outlist

#it ended up looking more advantageous to collect the proteins where the sequences originated from, so I made this slower to do that
def independent_skeletons(p, joinedproteome, seqs, nothanks):
    #counts every p-length skeleton motif available in each protein
    per = p*'.'
    #collecting motifs in this manner assumes order of the sequence matters, ie G..T is different than T..G
    splits = set(zip(joinedproteome[:-p], joinedproteome[p:]))
    splits = [f'{per}'.join((i)) for i in splits if not any(r in i for r in nothanks)]
    outlist = []
    seqindex = {}
    for s in splits:
        #skeletonmatches = Counter(regex.findall(s, joinedproteome, overlapped=True, concurrent=False))
        skeletonmatches = Counter()
        seqind = defaultdict(set)
        for k, v in seqs.items():
            seqcount = Counter(regex.findall(s, v, overlapped=True, concurrent=False))
            if seqcount:
                skeletonmatches.update(seqcount)
                for sc in seqcount:
                    seqind[sc].add(k)
        for sm in tuple(skeletonmatches.keys()):
            if any(r in sm for r in nothanks):
                skeletonmatches.pop(sm)
        freqout = frequency_filter(dict(skeletonmatches))
        seqind = {k:seqind[k] for k in freqout}
        outlist.append(freqout)
        seqindex.update(seqind)
    return outlist, seqindex

def frequency_filter(skeletonmatches):
    #this acts as a filter for where the largest difference in the data lies
    #plt.hist(counts, bins=20)
    #^shows the same as:
    #plt.hist(diffmatrixsums, bins=20)
    #but the x-axis is shifted, anything with a count above zero here is kept
    matchcounts = Counter(skeletonmatches.values())
    counts, freqs = np.asarray(list(zip(*matchcounts.items())))
    freqs = freqs[counts.argsort()]
    counts = np.sort(counts)
    #diffmatrixsums = (freqs - freqs.reshape(-1,1)).sum(axis=0)
    #cutoff = counts[diffmatrixsums < 0].min()
    cfareas = integrate.cumtrapz(counts*freqs, x=counts, initial=0)
    diffmatrixsums = (cfareas - cfareas.reshape(-1,1)).sum(axis=0)
    cutoff = counts[diffmatrixsums > 0].min()
    #output = [k for k, v in skeletonmatches.items() if v >= cutoff]
    output = {k:v for k, v in skeletonmatches.items() if v >= cutoff}
    return output

t = time()
with concurrent.futures.ProcessPoolExecutor(ncores) as executor:
    futures, skeletons, splitinds = [], [], {}
    for p in patternspace:
        futures.append(executor.submit(independent_skeletons, p, joinedproteome, seqs, nothanks))
    for future in concurrent.futures.as_completed(futures):
        sk, si = future.result()
        skeletons.extend(sk)
        splitinds.update(si)
print(time() - t)

codons = {
        'F': ['UUU', 'UUC'],
        'L': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'],
        'I': ['AUU', 'AUC', 'AUA'],
        'M': ['AUG'],
        'V': ['GUU', 'GUC', 'GUA', 'GUG'],
        'S': ['UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC'],
        'P': ['CCU', 'CCC', 'CCA', 'CCG'],
        'T': ['ACU', 'ACC', 'ACA', 'ACG'],
        'A': ['GCU', 'GCC', 'GCA', 'GCG'],
        'Y': ['UAU', 'UAC'],
        'H': ['CAU', 'CAC'],
        'Q': ['CAA', 'CAG'],
        'N': ['AAU', 'AAC'],
        'K': ['AAA', 'AAG'],
        'D': ['GAU', 'GAC'],
        'E': ['GAA', 'GAG'],
        'C': ['UGU', 'UGC'],
        'W': ['UGG'],
        'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
        'G': ['GGU', 'GGC', 'GGA', 'GGG']
        }
#it would be cool and interesting to have a nice graphic that shows the relation between aminoacidfrequencies, and the number of codons for any AA

distdict = defaultdict(dict)
#min distance, max distance, sum total distance?! mode?
for a1, a2 in itertools.combinations_with_replacement(codons.keys(), 2):
    if a1 == a2:
        distdict[a1][a2] = 0
        distdict[a2][a1] = 0
    else:
        cds = []
        for b1 in codons[a1]:
            for b2 in codons[a2]:
                cds.append(editdistance_s.distance(b1,b2))
        #mean would be sick, but it's too slow later on when generating steps. Rounding decreased 28 unique values to 16, but the median only has 6 and provides good speed, any more might not.
        #^median is the next best thing with better resolution than min/max/mode, it's also feasible
        #print(a1, a2)
        #print(cds)
        cmedian = np.median(cds)
        #cmin = np.min(cds)
        #cmax = np.max(cds)
        #cdiffmean = np.average([cmin, cmax])
        #cmode = stats.mode(cds)[0][0]
        #cmean = np.average(cds)
        #print('median:', ca)
        #print('mean:', cmean)
        #print('diff mean:', cdiffmean)
        #print('mode:', cmode)
        #print('min:', cmin)
        #print('max:', cmax)
        #print('~~~~~~~')
        distdict[a1][a2] = cmedian
        distdict[a2][a1] = cmedian

codondistances = pd.DataFrame(distdict)
distdict = codondistances.to_dict() #this improves multiprocessing later on
cmax = codondistances.max().max()
#np.fill_diagonal(codondistances.values, 0)
plt.matshow(codondistances)
plt.title('Codon Distances')
plt.xticks(range(len(codondistances)), codondistances.columns)
plt.yticks(range(len(codondistances)), codondistances.columns)
plt.show()

#cross here
#combodistdict = {}
#for k, v in distdict.items():
#    for vk, vv in v.items():
#        nk = ''.join((k, vk))
#        combodistdict[nk] = vv

#to compare distances
#def codon_distance(cdf, p1, p2):
#    return p1, p2, np.diag(cdf.loc[(i for i in p1), (i for i in p2)]).sum()
#
#t = time()
#groupings = []
#inum = 0
#print('# skellies:', len(skeletons))
#for skelly in skeletons:
#    if len(skelly) > 1:
#        motifcomparisons = defaultdict(lambda: defaultdict(int))
#        for p1, p2 in itertools.combinations_with_replacement(skelly, 2):
#            pdist = sum(distdict[i][j] for i, j in zip(p1[1:-1], p2[1:-1]) if i != j)
#            motifcomparisons[p1][p2] = pdist
#            motifcomparisons[p2][p1] = pdist
#
#        comparisons = pd.DataFrame(motifcomparisons)
#
#        stepmatrix = np.sort((comparisons).to_numpy(), axis=0)[:-1]
#        steps = np.diff(np.unique(stepmatrix.flatten()))
#        steps = np.hstack((0, steps))
#
#        movingstep = stepmatrix.min()
#        frings = []
#        for s in steps:
#            movingstep += s
#            ringlist = []
#            for idcol in comparisons.columns:
#                groupinteractions = comparisons.loc[idcol] <= movingstep
#                groupinds = comparisons.loc[groupinteractions.to_numpy()].index.tolist()
#                groupinds.append(idcol)
#                ringlist.append(groupinds)
#            rings = to_graph(ringlist)
#            rings = [sorted(i) for i in connected_components(rings)]
#            if len(rings) > 1:
#                ringsandcount = []
#                rl = len(rings)
#                for ri in rings:
#                    #normalizing to len(ri) keeps the length of the group as small as possible (promotes having more groups), while normalizing to len(rings) keeps the number of groups as small as possible (promotes less groups). These are naturally opposing forces.
#                    #but does this favor lesser interactions forming their own cluster as opposed to tacking on to clusters of greater inactions? I need a way to simulate this.
#                    groupcount = (comparisons.loc[ri,ri].sum().sum() / len(ri)) / rl
#                    ringsandcount.append([ri, groupcount])
#                if not ringsandcount in frings:
#                    frings.append(ringsandcount)
#
#        scores = []
#        for n, f in enumerate(frings):
#            nonzeros = [i[1] for i in f if i[1] > 0]
#            score = np.prod(nonzeros) * len(f) #final scoring and bang-for-buck normalization
#            scores.append([n, score, len(f), len(nonzeros)])
#
#        scores = sorted(scores, key=lambda x: x[1])
#
#        
#        groupings.extend([i[0] for i in frings[scores[0][0]]])
#    else:
#        groupings.extend(skelly)
#    inum += 1
#    print(time() - t, inum)
#print(time() - t)

#original method using a scoring to cut off groups, it's imperfect
#skelly = [i for i in skeletons if len(i[0]) == 21 and i[0].startswith('Y') and i[0].endswith('R')][0]
#
#for skelly in skeletons:
#    if len(skelly) > 1:
#        motifcomparisons = defaultdict(lambda: defaultdict(int))
#        #clevs = defaultdict(lambda: defaultdict(float))
#        #levs = defaultdict(lambda: defaultdict(int))
#        #plen = len(skelly[0])
#        for p1, p2 in itertools.combinations_with_replacement(skelly, 2):
#            if p1 == p2:
#                motifcomparisons[p1][p2] = 0
#                #clevs[p1][p2] = 0
#                #levs[p1][p2] = 0
#            else:
#                #the problem with using codon-distance is that there's an underlying assumption that there could only be the median(if using median) number of mutations occuring at any point. The number of mutations aren't important here (and it could be even more than what's implied), it's the overall similarity to the original sequence - which is independent of any number of mutations.
#                clev = sum(distdict[i][j] for i, j in zip(p1[1:-1], p2[1:-1]) if i != j)
#                #lev = editdistance_s.distance(p1, p2)
#                #ndist = lev / clev
#                #motifcomparisons[p1][p2] = ndist
#                motifcomparisons[p1][p2] = clev
#                #clevs[p1][p2] = clev
#                #levs[p1][p2] = lev
#
#        for k1, v1 in motifcomparisons.items():
#            for k2, v2 in v1.items():
#                motifcomparisons[k2][k1] = v2
#        #for k1, v1 in clevs.items():
#        #    for k2, v2 in v1.items():
#        #        clevs[k2][k1] = v2
#        #for k1, v1 in levs.items():
#        #    for k2, v2 in v1.items():
#        #        levs[k2][k1] = v2
#
#        comparisons = pd.DataFrame(motifcomparisons)
#        #from comparisons you could also make a team-level comparison of any two peptides, rather than the direct-level comparison that's held inside this via a levenshtein distance of some form. The team-level comparison would look at a correlation, of sorts, between the rest of the peptides that any two peptides interact with.
#        #clevcomparisons = pd.DataFrame(clevs)
#        #levcomparisons = pd.DataFrame(levs)
#        skellyarray = comparisons.index.values
#        #motifmatrix = np.repeat(comparisons.index.values.reshape(-1,1), len(comparisons), axis=1).astype(np.string_)
#        comparisonmatrix = comparisons.to_numpy()
#        comparisoninds = {k:n for n, k in enumerate(comparisons.index.tolist())}
#        
#        stepmatrix = np.sort((comparisons).to_numpy(), axis=0)
#        steps = np.diff(np.unique(stepmatrix.flatten()))
#        steps = np.hstack((0, steps))
#        #ns = steps.reshape(len(steps),1,1,)
#        #stepexpansion = (np.expand_dims(comparisonmatrix, axis=0) <= ns)
#
#        #sklen = len(skelly[0]) - 2 #lev?
#        sklen = (len(skelly[0]) - 2) * cmax #clev?
#
#        movingstep = stepmatrix.min()
#        scoredict, leveldict = {}, defaultdict(list)
#        #sc, cc = 0, 0
#        for stepnum, s in enumerate(steps):
#            movingstep += s
#            ringlist = []
#            
#            tm = (comparisonmatrix <= movingstep).tolist()
#            #for idcol in comparisons.columns:
#                #groupinteractions = comparisons.loc[:,idcol] <= movingstep
#                #groupinds = comparisons.loc[groupinteractions.to_numpy()].index.tolist()
#                #groupinds.append(idcol)
#            for ti in tm:
#                groupinds = skellyarray[ti].tolist()
#                ringlist.append(groupinds)
#                #print(sc, cc)
#                #cc += 1
#            rings = to_graph(ringlist)
#            rings = [sorted(i) for i in connected_components(rings)]
#            #print(sc, cc)
#            #sc += 1
#            ringsandcount = []
#            rl = len(rings)
#            #if rl > 1: #assuming there is no scenario so ridiculous that every peptide would rightfully cluster together.
#            for ri in rings:
#                ringinds = [comparisoninds[r] for r in ri]
#                #normalizing to len(ri) keeps the length of the group as small as possible (promotes having more groups), while normalizing to len(rings) keeps the number of groups as small as possible (promotes less groups). These are naturally opposing forces.
#                #but does this favor lesser interactions forming their own cluster as opposed to tacking on to clusters of greater inactions? I need a way to simulate this.
#                #cr = comparisons.loc[ri, ri].sum(axis=0).sort_values()
#                cr = comparisonmatrix[ringinds][:,ringinds].sum(axis=0)
#                #groupcount = (comparisons.loc[ri,ri].sum().sum() / len(ri)) / rl
#                #groupcount = (cr.sum() / len(ri)) / rl
#                #groupcount = (cr.sum() / len(ri))
#                #groupcount = ((cr.sum() / len(ri)) / rl) * stepnum
#                #can ignore normalizing by rl because we're looking independent of any step-generated ring set
#                #the stepnum is linear so it's only going to lead to a linear end-pattern, which is undesirable here
#                #going to leave out normalizing by len(ri) to see if this represents anything better later on
#                #depending on the metric used above:
#                #groupcount = cr.sum()
#                #groupcount = cr.sum()**2 / len(cr)
#                #this is now a % similarity across all sequences
#                groupcount = 1 - (cr.sum() / (sklen * len(ri)**2))
#                #ringset = set(cr.index)
#                ringset = {i for i, y in sorted(zip(ri, cr), key=lambda x: x[1])}
#                if tuple(ringset) not in scoredict.keys():
#                    scoredict[tuple(ringset)] = groupcount
#                    leveldict[stepnum].append(ringset)
#
#        #A cluster is a group of peptides in a specific step of a path. Paths are the growth that each cluster undergoes as the number of steps increase. Steps are the differences in levenshtein distance between peptides of this skelly that connect one peptide to another to form clusters.
#
#        groupleveldict = leveldict.copy()
#        groupleveldict.pop(0) #removing the plethora of single-peptide groups not applicable to this analysis
#        #pathdictcollector accumulates every path from any starting point in groupleveldict. It does not account for the fact that a specific path may not be unique compared to the paths its already accounted for.
#        pathdictcollector = defaultdict(lambda: defaultdict())
#        pathcount = 0
#        for k1, v1 in groupleveldict.items():
#            for group1 in v1:
#                pathcount += 1
#                pathdictcollector[pathcount][k1] = group1
#                for k2, v2 in groupleveldict.items():
#                    if k2 > k1:
#                        for group2 in v2:
#                            if group1.intersection(group2):
#                                pathdictcollector[pathcount][k2] = group2
#
#        removallist = []
#        for k1 in pathdictcollector.keys():
#            for k2 in pathdictcollector.keys():
#                if k1 != k2:
#                    if pathdictcollector[k1].items() > pathdictcollector[k2].items(): #checks if one dict is a subset of another
#                        removallist.append(k2)
#
#        #pathdict is pathdictcollector, except that only unique paths are accounted for, and they are fully accounted for.
#        pathdict = {}
#        n = 0
#        for k, v in pathdictcollector.items():
#            if k not in removallist:
#                pathdict[n] = dict(v)
#                n += 1
#
#        #clusterdict assigns a unique identifier to every cluster that exists.
#        clusterdict = {}
#        n = 0
#        for g, cl in groupleveldict.items():
#            for c in cl:
#                clusterdict[tuple(c)] = n
#                n += 1
#
#        pcols = set(j for k, i in pathdict.items() for j in i.keys())
#        #clusterpath places the unique cluster identifier at the specific step of each path where that cluster forms
#        clusterpath = pd.DataFrame(index=range(len(pathdict)), columns=pcols)
#        #scorepath places each clusters score from the ring groupings at the specific step of each path where that cluster forms.
#        scorepath = clusterpath.copy()
#        scorearrays = defaultdict(list)
#        for path, pv in pathdict.items():
#            for step, sv in pv.items():
#                tsv = tuple(sv)
#                clusterpath.loc[path, step] = clusterdict[tsv]
#                scorepath.loc[path, step] = scoredict[tsv]
#                scorearrays[path].append([step, len(tsv), scoredict[tsv]])
#
#        cparray = clusterpath.to_numpy().astype(float)
#
#        #nodecounts counts the number of earlier clusters of lower steps that were subsets of a cluster in each position on every path
#        nodecounts = {}
#        #clustercounts counts the number of peptides in each cluster
#        clustercounts = {}
#        for k, v in clusterdict.items():
#            nodecounts[v] = (cparray[(cparray == v).any(axis=1)] <= v).sum()
#            clustercounts[v] = len(k)
#
#        
#        
#        #nodeaccumulations organizes nodecounts across paths(index) and steps(columns)
#        nodeaccumulations = []
#        #clusteraccumulations organizes clustercounts across paths(index) and steps(columns)
#        clusteraccumulations = []
#        for n in clusterpath.columns:
#            nodeaccumulations.append(clusterpath.loc[:,n].map(nodecounts))
#            clusteraccumulations.append(clusterpath.loc[:,n].map(clustercounts))
#        
#        nodeaccumulations = pd.concat(nodeaccumulations, axis=1)
#        clusteraccumulations = pd.concat(clusteraccumulations, axis=1)
#
#        #score1 = (scorepath / scorepath.columns) * (nodeaccumulations * clusteraccumulations)
#        #I like score2 better because I want the # of prior nodes to count more than the current step. I might want to include that multiplication of clusteraccumulations in score2 though as well, I need to double check what this actually is...
#        #Ultimately I think some time of score cutoff of the largest reverse-path coming out of the largest ending cluster, or something, should determine a score cutoff, where any step that is less than that score gets included in the final groupings, that's something that's flexible across both the steps and paths..
#        #score2 = (scorepath * scorepath.columns) / nodeaccumulations
#        #score3 = (scorepath * scorepath.columns * clusteraccumulations) / nodeaccumulations
#        #scorevals = scores.to_numpy().flatten().astype(float)
#        #scorevals = scorevals[~np.isnan(scorevals)]
#
#        #the strength of a group is obviously it's size. Smaller groups are more likely to be false positives. Ending group size would be a great measure of this, right? Minus, perhaps, the size of the group at the very end (the lone group), then normalized to the number of motifs in skelly
#        #scores.transpose().plot.line(legend=False)
#        #plt.hist(scores.to_numpy().flatten(), bins=10) #seems to separate well
#        #plt.plot(score1, score2, '.')
#        #plt.show()
#        #determine biggest separations, there may be more than 1 big separation, so finding multiple of them is a good goal
#        #distribution of differences among both scores
#        #finding the largest ones in both directions, then drawing lines across those planes to find a group to use for combinatorics
#        #s1dist = score1.to_numpy().flatten().astype(float)
#        #s2dist = score2.to_numpy().flatten().astype(float)
#        #s1dist = s1dist[~np.isnan(s1dist)]
#        #s2dist = s2dist[~np.isnan(s2dist)]
#        ##s1diff = np.diff(np.sort(np.diff(np.sort(s1d))))
#        ##s2diff = np.diff(np.sort(np.diff(np.sort(s2d))))
#        #s1diff = np.diff(np.sort(s1dist))
#        #s2diff = np.diff(np.sort(s2dist))
#        #s1diff = np.sort(np.diff(np.sort(np.diff(np.sort(s1d)))))
#        #s2diff = np.sort(np.diff(np.sort(np.diff(np.sort(s2d)))))
#
#        #using scorepath/scoredict for the endgame:
#        #I need to find the different step-independent path combinations that lead to potential groups of non-overlapping clusters. The scores from scoredict can then be multiplied against each other and the closest product to 1 would be assigned to the best combination. 
#        #number of peptides in a cluster / number of nodes the cluster has existed for prior:
#        #(clusteraccumulations / nodeaccumulations)
#        #^this becomes the power that scorepath is raised to. When this happens, are scores > 1 better? Or are scores closest to 1 the best? These will be the numbers that get multiplied together, when considering combinations of non-overlapping clusters, to determine final score.
#        #drop in scorepath per peptides added(aka scorepath difference per clusteraccumulations)
#
#        #biggest change in slopes: find slopes, take difference -> cutoff?
#        pathcutoffs = []
#        for sk, sa in scorearrays.items():
#            scorediffs = np.diff(sa, axis=0)
#            scoreslopes = np.divide(scorediffs[:,2], scorediffs[:,0])
#            clusterslopes = np.divide(scorediffs[:,1], scorediffs[:,0])
#            scoreclusterderivative = np.divide(np.diff(scoreslopes), np.diff(clusterslopes)) #original
#            scorecutoff = scoreclusterderivative.argmin() #original
#            print(scorecutoff)
#            pathcutoffs.append([sk, scorecutoff, sa[scorecutoff][0], len(sa)])
#        #now from all related paths, the one with the largest scorecutoff yields the finalized group
#        #^ or better described: all scorecutoffs are considered, and places where a larger scorecutoff logically interferes with the existence of groups formed by smaller scorecutoffs, the dominant one is the one with the largest. And all other external logic follows this.
#        #largest cutoff wins, this helps keep the early-formed paths dominant. Smaller, uninterfering groups will also be able to 'flourish' under this, while super large end-game clusters will be logic-blocked.
#
#        #ditch the idea of scorearrays, and stop adopting the score post-cluster growth. Within each group at each step of a cluster at any point in the tree, each AA should be changed to an integer, as a unique identifier, to be able to count # of unique identifiers in each position. This can both help figure out where to put wild card AAs, and can serve as a simple system of determining whether an extra step provides a peptide that is a good fit to this group.
#        #^save old function, copy for a new one
#
#        pathcutoffs = sorted(pathcutoffs, key=lambda x: (-x[1], x[2], -x[3])) #sorted by scorecutoff (# of surviving steps), max surviving step, # of steps in the path
#
#        outgroups = {}
#        outscores = []
#        outlengths = []
#        logicblock = set()
#        for p0, p1, p2, p3 in pathcutoffs:
#            og = pathdict[p0][p2]
#            while True:
#                if logicblock.intersection(og):
#                    p1 -= 1
#                    p2 = list(pathdict[p0].keys())[p1]
#                    og = pathdict[p0][p2]
#                else:
#                    outgroups[tuple(og)] = scoredict[tuple(og)]
#                    outscores.append(scoredict[tuple(og)])
#                    outlengths.append(len(og))
#                    logicblock.update(og)
#                if p1 < 0:
#                    break


#skelly = [i for i in skeletons if len(i[0]) == 21 and i[0].startswith('Y') and i[0].endswith('R')][0] #was used for testing
#skelly = [i for i in skeletons if len(i[0]) == 4 and i[0].startswith('Y') and i[0].endswith('P')][0] #was used for testing

#t = time()
#endseqs = []
#for skelly in skeletons:
#if len(skelly) == 1:
#    endseqs.extend(skelly)
#else:
#    motifcomparisons = defaultdict(dict)
#    #clevs = defaultdict(lambda: defaultdict(float))
#    #levs = defaultdict(lambda: defaultdict(int))
#    #plen = len(skelly[0])
#    for p1, p2 in itertools.combinations_with_replacement(skelly, 2):
#        if p1 == p2:
#                motifcomparisons[p1][p2] = 0
#                #clevs[p1][p2] = 0
#                #levs[p1][p2] = 0
#            else:
#                #the problem with using codon-distance is that there's an underlying assumption that there could only be the median(if using median) number of mutations occuring at any point. The number of mutations aren't important here (and it could be even more than what's implied), it's the overall similarity to the original sequence - which is independent of any number of mutations.
#                clev = sum(distdict[i][j] for i, j in zip(p1[1:-1], p2[1:-1]) if i != j)
#                #lev = editdistance_s.distance(p1, p2)
#                #ndist = lev / clev
#                #motifcomparisons[p1][p2] = ndist
#                motifcomparisons[p1][p2] = clev
#                #clevs[p1][p2] = clev
#                #levs[p1][p2] = lev
#
#        for k1, v1 in motifcomparisons.items():
#            for k2, v2 in v1.items():
#                motifcomparisons[k2][k1] = v2
#
#        comparisons = pd.DataFrame(motifcomparisons)
#        #from comparisons you could also make a team-level comparison of any two peptides, rather than the direct-level comparison that's held inside this via a levenshtein distance of some form. The team-level comparison would look at a correlation, of sorts, between the rest of the peptides that any two peptides interact with.
#        skellyarray = comparisons.index.values
#        comparisonmatrix = comparisons.to_numpy()
#        
#        stepmatrix = np.sort((comparisons).to_numpy(), axis=0)
#        steps = np.diff(np.unique(stepmatrix.flatten()))
#        steps = np.hstack((0, steps))
#        stepbooleans = comparisonmatrix <= steps.cumsum()[:,None,None]
#
#        movingstep = stepmatrix.min()
#        groups, leveldict = set(), defaultdict(list)
#        #for stepnum, s in enumerate(steps):
#        for stepnum, tm in enumerate(stepbooleans):
#            #movingstep += s
#            ringlist = []
#            #tm = (comparisonmatrix <= movingstep).tolist()
#            for ti in tm:
#                groupinds = skellyarray[ti].tolist()
#                ringlist.append(groupinds)
#            rings = to_graph(ringlist)
#            rings = [tuple(i) for i in connected_components(rings)]
#            for ri in rings:
#                if ri not in groups:
#                    groups.add(ri)
#                    leveldict[stepnum].append(set(ri))
#
#        groupleveldict = leveldict.copy()
#        groupleveldict.pop(0) #removing the plethora of single-peptide groups not applicable to this analysis
#        #pathdictcollector accumulates every path from any starting point in groupleveldict. It does not account for the fact that a specific path may not be unique compared to the paths its already accounted for.
#        pathdictcollector = defaultdict(dict)
#        pathcount = 0
#        for k1, v1 in groupleveldict.items():
#            for group1 in v1:
#                pathcount += 1
#                pathdictcollector[pathcount][k1] = group1
#                for k2, v2 in groupleveldict.items():
#                    if k2 > k1:
#                        for group2 in v2:
#                            if group1.intersection(group2):
#                                pathdictcollector[pathcount][k2] = group2
#
#        removallist = []
#        for k1 in pathdictcollector.keys():
#            for k2 in pathdictcollector.keys():
#                if k1 != k2:
#                    if pathdictcollector[k1].items() > pathdictcollector[k2].items(): #checks if one dict is a subset of another
#                        removallist.append(k2)
#
#        #pathdict is pathdictcollector, except that only unique paths are accounted for, and they are fully accounted for.
#        pathdict = {}
#        n = 0
#        for k, v in pathdictcollector.items():
#            if k not in removallist:
#                pathdict[n] = dict(v)
#                n += 1
#
#        #this finds the greatest loss of shared amino acids across all the peptides. Shared amino acids are counted as the length of a set across the transpose of the group of peptides. When a 1 changes to another number, this is recorded across each step on the path. The step before the greatest difference in the amount of 1s is the chosen cutoff.
#        pathcutoffs = []
#        for pk, pa in pathdict.items():
#            conservationmeasure = [[len(set(i)) for i in map(list, zip(*pa[pn]))].count(1) for pn in pa]
#            if len(conservationmeasure) == 1: #no steps, it happens
#                conservationcutoff = 0
#                conservationind = list(pa.keys())[conservationcutoff]
#                pathcutoffs.append([pk, conservationcutoff, len(pa), len(pa[conservationind]), conservationind])
#            elif conservationmeasure[0] > 2: #nothing except the end-pieces are conserved
#                conservationcutoff = np.diff(conservationmeasure).argmin()
#                conservationind = list(pa.keys())[conservationcutoff]
#                pathcutoffs.append([pk, conservationcutoff, len(pa), len(pa[conservationind]), conservationind])
#        pathcutoffs = sorted(pathcutoffs, key=lambda x: (-x[1], -x[2], -x[3], x[4])) #sorted by scorecutoff (# of surviving steps), # of steps in path, number of peptides in max step, max surviving step #
#
#        candidatesequences = []
#        logicblock = set()
#        for p0, p1, p2, p3, p4 in pathcutoffs:
#            og = pathdict[p0][p4]
#            while True:
#                if logicblock.intersection(og):
#                    p1 -= 1
#                    p4 = list(pathdict[p0].keys())[p1]
#                    og = pathdict[p0][p4]
#                else:
#                    candidatesequences.append(og)
#                    logicblock.update(og)
#                    break
#                if p1 < 0:
#                    break
#        #would be nice to construct some cool plots here with candidatesequences
#        
#        takensequences = {j for i in candidatesequences for j in i}
#        finalizedsequences = []
#        for s in candidatesequences:
#            su = list(s)[0]
#            placemarks = [len(set(i)) for i in map(list, zip(*s))]
#            outsequence = ''.join(([su[i] if placemarks[i] == 1 else '.' for i in range(len(su))]))
#            finalizedsequences.append(outsequence)
#        
#        for s in leveldict[0]:
#            if not takensequences.intersection(s):
#                finalizedsequences.append(list(s)[0])
#        endseqs.extend(finalizedsequences)
#print(time() - t)

#I still like the idea of this:
#nth order rank differences
#sort -> take difference -> record percentage
#sort differences -> take next diff -> record next perc
#rinse, repeat. What %'s make the least change across these changes?
#values lost from the diff function will be counted as losses at the beginning, which should also be the smallest of the differences (which would be the least relevant place to put a divider)

#after getting clusters, the low-scoring one is always a yes, the higer scores are always a no: if there's any in between clusters, they all get compared to the auto yes/no clusters as whichever one they're most similar to(is this even possible?) is the direction they go.

# (score * current step) / (# steps this path held prior)
#(# steps held prior * # peptides in that group)
#your final cutoff could be a node distance instead of a step distance
#plt.hist(scores.to_numpy().flatten(), bins=10) #seems to separate well


#for distance function:
#use levenshtein distance as the main function.
#then, to rank things that come out the same in levenshtein distance: use the codon levenshtein distance as a secondary metric to further distinguish 

def sequenceclustering(skelly, distdict):
    motifcomparisons = defaultdict(dict)
    for p1, p2 in itertools.combinations_with_replacement(skelly, 2):
        if p1 == p2:
            motifcomparisons[p1][p2] = 0
        else:
            #the problem with using codon-distance is that there's an underlying assumption that there could only be the median(if using median) number of mutations occuring at any point. The number of mutations aren't important here (and it could be even more than what's implied), it's the overall similarity to the original sequence - which is independent of any number of mutations.
            clev = sum(distdict[i][j] for i, j in zip(p1[1:-1], p2[1:-1]) if i != j)
            #lev = editdistance_s.distance(p1, p2)
            #ndist = lev / clev
            motifcomparisons[p1][p2] = clev

    for k1, v1 in motifcomparisons.items():
        for k2, v2 in v1.items():
            motifcomparisons[k2][k1] = v2

    comparisons = pd.DataFrame(motifcomparisons)
    #from comparisons you could also make a team-level comparison of any two peptides, rather than the direct-level comparison that's held inside this via a levenshtein distance of some form. The team-level comparison would look at a correlation, of sorts, between the rest of the peptides that any two peptides interact with.
    skellyarray = comparisons.index.values
    comparisonmatrix = comparisons.to_numpy()
    
    stepmatrix = np.sort((comparisons).to_numpy(), axis=0)
    #steps = np.diff(np.unique(stepmatrix.flatten()))
    #steps = np.hstack((0, steps))
    steps = np.unique(stepmatrix.flatten())
    stepbooleans = comparisonmatrix <= steps[:,None,None]

    movingstep = stepmatrix.min()
    groups, leveldict = set(), defaultdict(list)
    for stepnum, tm in enumerate(stepbooleans):
        ringlist = []
        for ti in tm:
            groupinds = skellyarray[ti].tolist()
            ringlist.append(groupinds)
        rings = to_graph(ringlist)
        rings = [tuple(i) for i in connected_components(rings)]
        for ri in rings:
            if ri not in groups:
                groups.add(ri)
                leveldict[stepnum].append(set(ri))

    groupleveldict = leveldict.copy()
    groupleveldict.pop(0) #removing the plethora of single-peptide groups not applicable to this analysis
    #pathdictcollector accumulates every path from any starting point in groupleveldict. It does not account for the fact that a specific path may not be unique compared to the paths its already accounted for.
    pathdictcollector = defaultdict(dict)
    pathcount = 0
    for k1, v1 in groupleveldict.items():
        for group1 in v1:
            pathcount += 1
            pathdictcollector[pathcount][k1] = group1
            for k2, v2 in groupleveldict.items():
                if k2 > k1:
                    for group2 in v2:
                        if group1.intersection(group2):
                            pathdictcollector[pathcount][k2] = group2

    removallist = []
    for k1 in pathdictcollector.keys():
        for k2 in pathdictcollector.keys():
            if k1 != k2:
                if pathdictcollector[k1].items() > pathdictcollector[k2].items(): #checks if one dict is a subset of another
                    removallist.append(k2)

    #pathdict is pathdictcollector, except that only unique paths are accounted for, and they are fully accounted for.
    pathdict = {}
    n = 0
    for k, v in pathdictcollector.items():
        if k not in removallist:
            pathdict[n] = dict(v)
            n += 1

    #this finds the greatest loss of shared amino acids across all the peptides. Shared amino acids are counted as the length of a set across the transpose of the group of peptides. When a 1 changes to another number, this is recorded across each step on the path. The step before the greatest difference in the amount of 1s is the chosen cutoff.
    pathcutoffs = []
    for pk, pa in pathdict.items():
        conservationmeasure = [[len(set(i)) for i in map(list, zip(*pa[pn]))].count(1) for pn in pa]
        if len(conservationmeasure) == 1: #no steps, it happens
            conservationcutoff = 0
            conservationind = list(pa.keys())[conservationcutoff]
            pathcutoffs.append([pk, conservationcutoff, len(pa), len(pa[conservationind]), conservationind])
        elif conservationmeasure[0] > 2: #nothing except the end-pieces are conserved
            conservationcutoff = np.diff(conservationmeasure).argmin()
            conservationind = list(pa.keys())[conservationcutoff]
            pathcutoffs.append([pk, conservationcutoff, len(pa), len(pa[conservationind]), conservationind])
    pathcutoffs = sorted(pathcutoffs, key=lambda x: (-x[1], -x[2], -x[3], x[4])) #sorted by scorecutoff (# of surviving steps), # of steps in path, number of peptides in max step, max surviving step #
    
    candidatesequences = []
    logicblock = set()
    for p0, p1, p2, p3, p4 in pathcutoffs:
        og = pathdict[p0][p4]
        while True:
            if logicblock.intersection(og):
                p1 -= 1
                p4 = list(pathdict[p0].keys())[p1]
                og = pathdict[p0][p4]
            else:
                candidatesequences.append(og)
                logicblock.update(og)
                break
            if p1 < 0:
                break
    #would be nice to construct some cool plots here with candidatesequences, or maybe later as finalizedsequences
    
    takensequences = {j for i in candidatesequences for j in i}
    finalizedsequences = []
    finalizedseqlocs = {}
    for s in candidatesequences:
        su = list(s)[0]
        placemarks = [len(set(i)) for i in map(list, zip(*s))]
        outsequence = ''.join(([su[i] if placemarks[i] == 1 else '.' for i in range(len(su))]))
        finalizedsequences.append(outsequence)
        finalizedseqlocs[outsequence] = s
    
    for s in leveldict[0]:
        if not takensequences.intersection(s):
            finalizedsequences.append(''.join((s)))
    
    return finalizedsequences, finalizedseqlocs

t = time()
groupings, scaffinds = set(), {}
print('# skellies:', len(skeletons))
with concurrent.futures.ProcessPoolExecutor(ncores) as executor:
    futures = []
    for skelly in skeletons:
        if len(skelly) > 1:
            #skelly should be input as a dict in order to keep track of scaffold protein sources
            futures.append(executor.submit(sequenceclustering, list(skelly.keys()), distdict))
        else:
            groupings.add(''.join((skelly.keys())))
    for future in concurrent.futures.as_completed(futures):
        fseqs, fseqi = future.result()
        groupings.update(fseqs)
        scaffinds.update(fseqi)
print(time() - t)


skellycounts = {}
for s in skeletons:
    skellycounts.update(s)
skellycounts = Counter(skellycounts)


#Wing Expansion Search
#Making sure each sequence that's been pulled out is represented in its own entirety.
#concurrency was shit here

def wing_expansion(seq, seqs, splitinds, skellycounts):
    #containers = [i for i in proteome if seq in i]
    #containers = []
    #tc = 0
    #for p in proteome:
    #    if seq in p:
    #        containers.append(p)
    #        tc += len(regex.findall(seq, p, overlapped=True))
    #    if tc == counttarget:
    #        break
    containers = [seqs[i] for i in splitinds[seq]]
    clen = len(containers)
    if clen < skellycounts[seq]: #if the sequence shows up more than once in a protein that it's found in, this will align each index within all of those proteins individually
        conindices = [[m.start() for m in regex.finditer(seq, i, overlapped=True)] for i in containers]
        concounts = [len(i) for i in conindices]
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
    #seqcounts = np.asarray([len(set(filter(None, i))) for i in map(list, zip(*alignments))]) #allows protein runoff
    #seqcounts = np.asarray([len(set(i)) for i in zip(*alignments)])
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

    #for troubleshooting purpose
    #    if  len(regex.findall(trueseq, joinedproteome, concurrent=True, overlapped=True)) < clen: #some of these are self-overlapping sequences, and some overlap more than fucking once because they're some weird kind of forward-facing pseudo-palindrome!
    #        print(seq, 'problem')

    #for troubleshooting
    #for c in alignments:
    #    print(''.join((c[minbound:maxbound])))
    
    trueseq = ''.join((alignments[0][minbound:maxbound]))
    return trueseq, splitinds[seq]


t = time()
expansions = set()
print('Expanding', len(groupings), 'sequences')
for n, seq in enumerate(groupings):
    if '.' not in seq:
        seqexpansion, splitexpansion = wing_expansion(seq, seqs, splitinds, skellycounts)
        expansions.add(seqexpansion)
        splitinds[seqexpansion] = splitexpansion
    if not n % 10000:
        print(n, time() - t)
print(time() - t)

#No scaffolds in wing expansion:
#The thing with the scaffolds is that they get waay more hits than the normal sequences do. There's very likely to not be any expanding that's going to happen. And this would be a super slow process to even check.
#it might be better find if any are subsets of each other though and compare the output via a search in order to remove redundancy

t = time()
scaffolds = [i for i in groupings if '.' in i]
subsetscaffs = defaultdict(set)
for scaff in scaffolds:
    newscaff = scaffolds.copy()
    newscaff.remove(scaff)
    finds = [i for i in newscaff if scaff in i]
    subsetscaffs[scaff].update(finds)
print(time() - t)
subsetscaffs = {k:v for k, v in subsetscaffs.items() if v}

def scaffold_superset_minimization(k, v, joinedproteome, nothanks):
    subsetcount = len([i for i in regex.findall(k, joinedproteome, overlapped=True, concurrent=True) if not any(n in i for n in nothanks)])
    for sv in v:
        supersetcount = len([i for i in regex.findall(sv, joinedproteome, overlapped=True, concurrent=True) if not any(n in i for n in nothanks)])
        if supersetcount == subsetcount:
            return k

#This favors larger scaffolds over smaller subsets if each show up the same number of times in the proteome.
#Using the subsets listing out their own supersets, if anything in a subset's superset list matches it's proteome hits, then remove the subset. This works out rather perfectly. As subsets that are found more are kept, subsets that are found less are impossible, and supersets that are found an equal number of times to a subset are kept. If that superset is then a subset that matches to a larger superset, it's also removed and there are no issues generated along the way.
t = time()
removers = set()
with concurrent.futures.ThreadPoolExecutor(4 if ncores > 4 else ncores) as executor:
    futures = []
    for k, v in subsetscaffs.items():
        futures.append(executor.submit(scaffold_superset_minimization, k, v, joinedproteome, nothanks))
    for future in concurrent.futures.as_completed(futures):
        removers.add(future.result())
print(time() - t)

scaffolds = [i for i in scaffolds if i not in removers]

#Moving Window Search
#Might actually be a useful tool for simplifying scaffolds to become more useful on a wider scale, in order to make up for a smaller number of patternsplits?
#I think the vaguer the scaffold, the better. The algorithm continues until all output counts are equal, ie any potential new window poses as many new offers as any others

#An interesting example to look at from the initial sequence_clustering, as well as scaffold simplification:
#              'QPSPCG.N.EC...NG...C': 4,
#              'QPSPCGPNSQCRE.NEQAIC': 2,
#              'QPSPCGAN.QCR.SQGQAIC': 2,
#              'QPSPCGPN.QC.N.NGQA.C': 2,
#              'QPSPCGPNSECR..G..PSC': 2,
#there's definitely more of them too. How to simplify? Should I simplify?
#It might just be better to leave further analysis for once the time series show relevance :/
#^Problem is too complex otherwise

#hmms = sorted([i.count('.')/len(i) for i in scaffolds])
#hmmcount = Counter(hmms)
#plt.hist(hmms)
#plt.show()
##I need to have a way of catching this 20% cutoff
#
#scaffindlengths = defaultdict(dict)
#for k, v in scaffinds.items():
#    scaffindlengths[len(k)][k] = len(v)
#
##this would be sick to implement but also fucking impossible as is
#t = time()
#newseq = seq
#while True:
#    windowsize = 1
#    window = '.' * windowsize
#    outsizes = []
#    for p in range(len(newseq)-windowsize+1):
#        if newseq[p] != window:
#            nseq = ''.join((newseq[:p], window, newseq[p+windowsize:]))
#            finds = regex.findall(nseq, joinedproteome, overlapped=True, concurrent=True)
#            counts = Counter(finds)
#            outsizes.append(sum(counts.values()))
#        else:
#            outsizes.append(-1)
#    outcount = [i for i in outsizes if i != -1]
#    if len(set(outcount)) == 1:
#        break
#    else:
#        outsizes = np.asarray(outsizes)
#        if (outsizes == outsizes.max()).sum() == 1:
#            newpoint = np.argmax(outsizes)
#            newseq = '.'.join((newseq[:newpoint], newseq[newpoint+1:]))
#        else:
#            outbools = np.where(outsizes == outsizes.max())[0]
#            for newpoint in outbools:
#                newseq = '.'.join((newseq[:newpoint], newseq[newpoint+1:]))
#print(time() - t)
#test = regex.findall(nseq, joinedproteome, overlapped=True, concurrent=True)
#righteous = [len(set(i)) for i in zip(*test)]
#oldt = regex.findall(seq, joinedproteome, overlapped=True, concurrent=True)
#oldr = [len(set(i)) for i in zip(*oldt)]
##this looks pretty fucking righteous


#if all results are uniform, increase moving window size
#otherwise, pick the window that gave the greatest increase
#would be potentially useful to record window size that finally caused an effect

#'ISDIVVGKEDNVSAREALLRWARRSTARYPGVRVNDFTSSWRDGLAFSALVHRNRPDLLDWRKARNDRPRERLETAFHIVEKEYGVTRLLDPEDVDTNEPDEKSLITYISSLYDVFPEPPSIHPLFDMESQRRVHEYRDLAQQFIYWCREKTAYLQERSFPPTLIEMKRLLSDLQRFRSDEVSARKREKSKLIQIYKELERYFETVGEVDVEAELRPDAIEKAWYRMNTALQDREVILQQEIERLERLQRLADKVQREIKHVDQKLTDLEGRIGEEGRRIERLHPVDAKSIVEALETEIRHLEEPIQDMNQDCHVLNEGRYPHVSELHKKVNKLHQRWAQLRTNFHTNLVQKLSGLKYPVHETTVTRQTRMVVESRQIDTNPHFRDLQEHIEWCQNKLKQLLAADYGSDLPSVKEELDRQQHEHKIIDQFHTKILNDERQQTKFSGDELALYQQRLNQLQKVYAELLSTSTKRLSDLDSLQHFLGQASAELQWLNEKEQVEITRDWADKQLDLPSVHRYYENLMSELEKREMHFATILDRGEALLNQQHPASKCIEAHLTALQQQWAWLLQLTLCLEVHLKHATEYHQFFGEIKDAEQWLAKRDEILNSKFSQSDFGLDQGETLLRGMQDLREELNAFGETVATLQRRAQTVVPLNKRRQPVNRQGPVQAICAYKQQGQLQIEKGETVTLLDNSGRVKWRVRTAKGQEGPIPGACLLLPPPDQEAIDAAERLKRLFDRSVALWQKKHLRLRQNMIFATIRVVKGWDFDQFLAMGPEQRTAIRRALNDDADKLLSEGDPNDPQLRRLRREMDEVNRLFDEFEKRARAEEESKQASRIFTEECLAIKSKLEDMARELDQIILAPLPRDLDSLEHVLEIHSDYERRLHLLEPELKHLQETFRTIALKTPVLKKSLDNLMELWKELNTQSGLHKDRLKLLEASLAGLEDNEHVISELENELARHQDLPSTAEGLQQVFKQLNHMQDIITQQQPQMDKMNDAADQLGRMGVPTKVLGDLKRLHSNVERLNTRWSAVCNQLGERMRSCETAIGLMKNLQSSVQVEESWVDGTTERLSAMPTATSAYELD'
#^ is a very interesting example for this, it's length is 1083 but with a windowsize of up to 1076 (ie, considering every combination of 1076 adjacent inserted wildcards), the entirety of the sequence STILL only shows up at a single sequence. That's over 99% of the sequence! This speaks magnitutdes on the levels of specificity that one could expect from a lot of these sequences that are both close together, AND very far apart. This was also one of the first sequences I looked at, I imagine there are a lot of others in the same boat.
#FYDKD is another good example, it clearly has larger extending domains. But the algorithm doesn't pick up on them simply because they diverge in different proteins in different ways. Nonetheless, it provides a reason to look even further into a sequence once you realize that it's relevant in an actual dataset. Potential for more automation there as well!
#there needs to be a more advanced sequence selection, i don't know if it necessarily needs to happen at this stage, it probably could happen later with sequences found in data.
#^Take VAQIM for example, input it as seq above, if you expand around this there's way more to match - but 1 protein doesn't match the others and this prevents the rest of the sequence from showing up. The later-on sequence elucidation would probably do this justice, and potentially do justice to any of the flanking/diverging sequences if they end up being found.


#Expanding Window Search
#couldn't find much using this

#t = time()
#newseq = seq
#while True:
#    windowsize = 1
#    window = '.' * windowsize
#    for p in range(len(newseq)):
#        if newseq[p] == '.':
#            nseq = ''.join((newseq[:p], window, newseq[p:]))
#            finds = regex.findall(nseq, joinedproteome, overlapped=True, concurrent=True)
#            counts = Counter(finds)
#            outsizes.append(sum(counts.values()))



#Conclusions:
#   1. Wing expansion - Incorporated
        #this is strongly latching on to conserved regions of larger sequences - this is fine. It's not safe to assume that a short sequence found actually represents a short sequence on a protein. Thus the need for the on-the-spot analysis once the time series bit is worked out.
#   2. Moving window
        #lots of shit here, too much, too slow :/
#   3. Expanding window
        #ain't jack here
#   4. Reverse search
        #this is mostly done in aminoacidadjecencyexploration.py, ain't jack here
#   5. Reverse cutoff for splits, similar output sequences?
#what's left unclear is how to group things as varying sequences are found.
#To eliminate some bias, I think np.geomspace could be used in making patternspace. It woud be good to have more sequences of shorter length, and less of a larger length. But the larger length ones can still be useful too.
#Certain sequences, ie QQQQQ etc, actually need to be counted without overlapping. This would lead to a miscount among where these actually are. How to handle this? Should I just use .count() then to begin with?
#scaffolds can just as easily be used as a simplifcation of a sequence to represent multiple sequences until a more in-depth analysis needs to be done, as well as an actual scaffold motif.
#there's also a lot of existing overlap between sequences in expansions and scaffolds, but hey whatever.

#The on-the-spot analysis can make splits of found proteins in a quantitative group, then cluster sequences to look for similar bits. Should be pretty simple.


#End Compilation: motif, proteins it comes from, ignoring any subset-like connections here. I don't care if certain sequences are related, I just want to track them over time.

t = time()
finalmotifs = []
finalmotifs.extend(scaffolds)
finalmotifs.extend(expansions)

outframe = pd.DataFrame(finalmotifs, columns=['Motif'])
for row, mo in outframe.iterrows():
    mo = mo[0]
    if '.' in mo:
        sourceproteins = set()
        responsiblepeps = scaffinds[mo]
        for r in responsiblepeps:
            sourceproteins.update(splitinds[r])
        outframe.loc[row, 'Proteins'] = ', '.join((sourceproteins))
    else:
        outframe.loc[row, 'Proteins'] = ', '.join((splitinds[mo]))
print(time() - t)

outfile = ''.join((patternfolder, fname, '.csv'))
outframe.to_csv(outfile)

#Drosophila Examples

#patternspace: 3, 8, 13, 18
#1 hour generation process
#~50k expansions, ~60k scaffolds
#108MB db file

#patternspace: 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23
#5 hour generation process
#~55k expansions, ~162k scaffolds
#196MB db file
