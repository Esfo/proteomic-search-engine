import pandas as pd
from scipy import special
from time import time
import os
from collections import Counter, defaultdict
import networkx
from networkx.algorithms.components.connected import connected_components
import multiprocessing as mp
from Bio import SeqIO
import pickle
import itertools
import concurrent
import random
import sys
import gc
import re
import regex
from blist import blist
import matplotlib.pyplot as plt
import matplotlib
sys.path.append('/home/sfo/camp')
from sequencetools import seqsplit_dict
from scipy import stats
import numpy as np
plt.rcParams['figure.dpi'] = 100

proteome = '/home/sfo/data/fastas/proteomes/Fly_Drosophila_melanogaster.fasta'
ifsave = '/home/sfo/data/motifs/drosophila/motif-interactions.csv'

#patternfolder = '/store/drosophila/PXD005713/'
patternfolder = '/home/sfo/data/motifs/'
#end result prediction: as you increase pmax, you should get better predictions - but at longer computational times. The same general motifs should be found if the linkage and moving windows work the way I imagine them to, but perhaps some more specifics might be lost? Or maybe pmax has a maximum practical limit like 10 or something, where anything after is already something caught by linkage, etc.
pmax = 20

fasta = SeqIO.parse(open(proteome), 'fasta')
seqs = {}
for fasta in fasta:
    sequence, idn = str(fasta.seq), fasta.id
    seqs[idn] = sequence

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

patternspace = range(1, pmax)

def skeleton_search(p, proteome):
    #counts every p-length skeleton motif available in each protein
    proteomedict = defaultdict(int)
    for protein in proteome:
        for k, v in Counter(zip(protein[:-p], protein[p:])).items():
            proteomedict[k] += v
    return p, proteomedict

#This could be done with resampling results later on, that could be a bit more accurate. It's currently more stringent than I want it to be.
def frequency_filter(proteomedict):
    #this acts as a filter for where the largest difference in the data lies
    #plt.hist(counts, bins=20)
    #^shows the same as:
    #plt.hist(diffmatrix, bins=20)
    #but the x-axis is shifted, anything with a count above zero here is kept
    counts = np.sort(list(proteomedict.values())).astype(int)
    diffmatrixsums = (counts - counts.reshape(-1,1)).sum(axis=0)
    cutoff = counts[diffmatrixsums > 0].min()
    output = [k for k, v in proteomedict.items() if v >= cutoff]
    return output

def location_search(p, joinedproteome):
    psearch = regex.compile(p)
    plocs = []
    for i in psearch.finditer(joinedproteome, overlapped=True, concurrent=True):
        if '|' not in i.group():
            plocs.append(i.span())
    return np.asarray(plocs)

def location_search2(p, joinedproteome):
    psearch = regex.compile(p)
    plocs = []
    for i in psearch.finditer(joinedproteome, overlapped=True, concurrent=True):
        if '|' not in i.group():
            plocs.append(i.span()[0])
    return plocs

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

t = time()
per = '.'
patterndict = defaultdict()
with concurrent.futures.ProcessPoolExecutor(8) as executor:
    futures = []
    for p in patternspace:
        futures.append(executor.submit(skeleton_search, p, proteome))
    for future in concurrent.futures.as_completed(futures):
        pk, pn = future.result()
        pkey = (pk-1) * per
        patterndict[pkey] = pn
print(time() - t, '- pattern search finished')

t = time()
patterns = []
for pk, pn in patterndict.items():
    filteredpatterns = frequency_filter(pn)
    outstring = [f'{pk}'.join((a, b)) for a, b in filteredpatterns]
    patterns.extend(outstring)
print(time() - t, '- patterns filtered')
patterns = sorted(patterns)

#I'm going to join the proteome here to be a single string, because who cares about which proteins these overlaps are found in. I can exclude everything with a | from the matrix.
#regex module is way faster than the built in re module
#t = time()
#locationdict = defaultdict(list)
#for p in patterns:
#    psearch = regex.compile(p)
#    for i in psearch.finditer(joinedproteome, overlapped=True, concurrent=True):
#        if '|' not in i.group():
#            locationdict[p].append(i.span())
#print(time() - t)

#this will check for:
#1. Overlap
#2. Consistent (beginning - beginning) distance
#intersectiondict = defaultdict()
#subpatterns = patterns.copy()
#t = time()
#locationdict = defaultdict(list)
#for p in patterns:
#    #subpatterns.remove(p)
#    mainlocs = location_search(p, joinedproteome)
#    for subp in subpatterns:
#        sublocs = location_search(subp, joinedproteome)
#        #CALCULATE DEM OVERLAPS
#print(time() - t)

if not os.path.isfile(ifsave):
    print(f'{ifsave} not found, generating file')
    t = time()
    proteomelocationdict = defaultdict(list)
    for p in patterns:
        #subpatterns.remove(p)
        startlocs = location_search2(p, joinedproteome)
        for s in startlocs:
            proteomelocationdict[s].append(p)
    print(time() - t, '- pattern locations mapped')

#tf = pd.DataFrame.from_dict(proteomelocationdict, orient='index')

#t = time()
#proteomeinteractiondict = defaultdict(lambda: defaultdict(int))
#for pi, pl in proteomelocationdict.items():
#    for s1 in pl:
#        for s2 in pl:
#            proteomeinteractiondict[s1][s2] += 1
#            #it would be neat to keep a moving list of motifs as they extend through their length into future locations and keep adding +1 to each dict as they interact. This might give a more accurate picture too.
#            #Actually, I think this is the ONLY way it would be legitemate, since your base comparator would be a potential motif to itself.
#print(time() - t)

    t = time()
    proteomeinteractiondict = defaultdict(lambda: defaultdict(int))
    adders, rounds = [], np.array([], dtype=int)
    for pi, pl in proteomelocationdict.items():
        rounds += 1
        rounds = np.append(rounds, 0)
        adders.append(pl)
        #addround = list(itertools.chain(*adders))
        addround = set(itertools.chain(*adders)) #nothing counts as an interaction with itself when forming this as a set, and using only combinations with no replacement. It seems gently touching upon present interactions shows betters relations than overtouching interactions that have already been accounted for.
        #for s1, s2 in itertools.combinations_with_replacement(addround, 2):
        for s1, s2 in itertools.combinations(addround, 2):
            #having both of these makes the outcoming matrix symmetrical
            #this also avoids showing self-interaction, which isn't necessary for my purposes
            proteomeinteractiondict[s1][s2] += 1
            proteomeinteractiondict[s2][s1] += 1
        for n, (a, r) in enumerate(zip(adders, rounds)):
            adders[n] = [sa for sa in a if len(sa) > r]
        #adders, rounds = zip(*[(a, r) for a, r in zip(adders, rounds) if a]) #slower
        while True:
            try:
                if len(adders[0]) == 0:
                    del adders[0]
                    rounds = np.delete(rounds, 0)
                else:
                    break
            except IndexError: #nothing is in adders
                break
    print(time() - t, 'overlaps counted')

#t = time()
#interactionframe = pd.DataFrame(0, columns=patterns, index=patterns)
#adders, rounds = [], np.array([], dtype=int)
#for pi, pl in proteomelocationdict.items():
#    rounds += 1
#    rounds = np.append(rounds, 0)
#    adders.append(pl)
#    addround = itertools.chain(*adders)
#    addcounts = Counter(addround)
#    countvals = np.asarray(list(addcounts.values()))
#    ak = addcounts.keys()
#    interactionframe.loc[ak, ak] += (countvals + countvals.reshape(-1,1))
#    for n, (a, r) in enumerate(zip(adders, rounds)):
#        adders[n] = [sa for sa in a if len(sa) > r]
#    while True:
#        try:
#            if len(adders[0]) == 0:
#                del adders[0]
#                rounds = np.delete(rounds, 0)
#            else:
#                break
#        except IndexError: #nothing is in adders
#            break
#print(time() - t, ' -overlaps counted')


    interactionframe = pd.DataFrame.from_dict(proteomeinteractiondict)
    interactionframe.fillna(0, inplace=True)
    interactionframe.sort_index(inplace=True)
    interactionframe = interactionframe.loc[:,interactionframe.columns.sort_values()]
    interactionframe.index.name = 'Skeleton'
    interactionframe.to_csv(ifsave)
else:
    print(f'Loading {ifsave}')
    interactionframe = pd.read_csv(ifsave, index_col='Skeleton', na_filter=False)


#log colors on this one?
fig, ax = plt.subplots(figsize=(12,9))
im = ax.matshow(interactionframe.to_numpy())
plt.xticks(range(len(interactionframe)), interactionframe.columns.tolist(), rotation=90)
plt.yticks(range(len(interactionframe)), interactionframe.index.tolist())
fig.colorbar(im)
plt.show()

#nif = interactionframe / interactionframe.max(axis=0)
#
#fig, ax = plt.subplots(figsize=(12,9))
#im = ax.matshow(nif.to_numpy())
#plt.xticks(range(len(nif)), nif.columns.tolist(), rotation=90)
#plt.yticks(range(len(nif)), nif.index.tolist())
#fig.colorbar(im)
##plt.xlim(0, 200)
##plt.ylim(200, 0)
#plt.show()

#interactiondistances = interactionframe.max(axis=0) - interactionframe
#stepmatrix = np.sort((interactiondistances).to_numpy(), axis=0)[:-1]
#steps = np.diff(np.unique(stepmatrix.flatten()))
#steps = np.hstack((0, steps))

#def ringfunc(interactionframe, ri, rl):
#    groupcount = (interactionframe.loc[ri,ri].sum().sum() / len(ri)) / rl
##normalizing to len(ri) keeps the length of the group as small as possible (promotes having more groups), while normalizing to len(rings) keeps the number of groups as small as possible (promotes less groups). These are naturally opposing forces.
##but does this favor lesser interactions forming their own cluster as opposed to tacking on to clusters of greater inactions? I need a way to simulate this.
#    return ri, groupcount

#def colfunc(interactiondistances, idcol, movingstep):
#    groupinteractions = interactiondistances.loc[idcol] <= movingstep
#    groupinds = interactiondistances.loc[groupinteractions.to_numpy()].index.tolist()
#    groupinds.append(idcol)
#    return groupinds

#exact calculation is not possible, way too slow. Perhaps using a rankedtable while converting to a list, zipping, and transposing via zia to get individual lists -> then throwing to networkx would be a very fast solution.
#BUT BUT BUT, you don't NEED clusters, in fact there's disadvantages if you're not doing it in a heirarchical sense because(and maybe ranked table can do a heirarchical) one popular skeleton may go with multiple motifs. Maybe a ranked table is the way to discern what skeletons should be combined?
#You can find the derivative: which skeletons interact with the most other skeletons? Just a relative interaction abundance ranking should be able to represent this.


#t = time()
#with concurrent.futures.ThreadPoolExecutor(1) as executor:
#    ringlist = []
#    futures = []
#    for idcol in interactiondistances.columns:
#        futures.append(executor.submit(colfunc, interactiondistances, idcol, movingstep))
#    for future in concurrent.futures.as_completed(futures):
#        ringlist.append(future.result())
#print(time() - t)

#movingstep = stepmatrix.min()
#frings = defaultdict(list)
#n = 0
#for s in steps:
#    movingstep += s
#    ringlist = []
#    for idcol in interactiondistances.columns:
#        groupinteractions = interactiondistances.loc[idcol] <= movingstep
#        groupinds = interactiondistances.loc[groupinteractions.to_numpy()].index.tolist()
#        groupinds.append(idcol)
#        ringlist.append(groupinds)
#    print(f'step {s}')
#    rings = to_graph(ringlist)
#    rings = [sorted(i) for i in connected_components(rings)]
#    if len(rings) > 1 and len(rings) <= len(interactiondistances) / 2:
#        ringsandcount, futures = [], []
#        t = time()
#        with concurrent.futures.ThreadPoolExecutor(1) as executor:
#            for ri in rings:
#                executor.submit(ringfunc, interactionframe, ri, len(rings))
#            for future in concurrent.futures.as_completed(futures):
#                eri, egc = future.result()
#                if egc > 0:
#                    ringsandcount.append([eri, egc])
#        print(time() - t)
#        print(n, 'done')
#        n += 1
#        frings[len(rings)].append(ringsandcount)


interactionsums = interactionframe.sum(axis=0)
interactionsums.name = '# interactions'
interactionsums = interactionsums.to_frame()
interactionsums.loc[:,'length'] = [len(i) for i in interactionsums.index]
interactionsums.loc[:,'start'] = [i[0] for i in interactionsums.index.tolist()]
interactionsums.loc[:,'end'] = [i[-1] for i in interactionsums.index.tolist()]

fig, ax = plt.subplots(figsize=(12,9))
interactionsums.plot.scatter(y='# interactions', x='length', ax=ax)
plt.show()
#this plot is one of the more interesting things I've found. Is this pattern affected/made by the mechanisms of the filtering I'm doing?
#Now plot shape by starting AA and color by ending AA
#And plot # interactions vs AA and length vs AA
# - Does the amino acid frequency dictate which AAs are in more abundance motifs?
#3D plots would obviously be nice, especially the rotatable kind. That way color can be the AA not depicted on the z-axis


interactionsums.loc[:,'length'].plot.hist(bins=interactionsums.loc[:,'length'].unique().shape[0])
plt.xlabel('length')
plt.show()

interactionsums.loc[:,'# interactions'].plot.hist(bins=100)
plt.xlabel('# interactions')
plt.show()

#interactionsums.plot.scatter(y='length', x='start')
#needs color

fig, ax = plt.subplots(figsize=(12,9))
aas = interactionsums.loc[:,'start'].unique()
aas = [i for i in aminoacidfrequencyorder if i in aas]
colormap = plt.cm.tab20c_r
colors = [colormap(i) for i in np.linspace(0, 1, len(aas))]
ax.set_prop_cycle('color', colors)

for s in aas:
    ii = interactionsums.loc[:,'start'] == s
    interactionsums.loc[ii, '# interactions'].sort_values().plot.line(label=s, ax=ax)
plt.legend(bbox_to_anchor=(1,1))
plt.show()


fig, ax = plt.subplots(figsize=(12,9))
aas = interactionsums.loc[:,'end'].unique()
aas = [i for i in aminoacidfrequencyorder if i in aas]
colormap = plt.cm.tab20c_r
colors = [colormap(i) for i in np.linspace(0, 1, len(aas))]
ax.set_prop_cycle('color', colors)

for s in aas:
    ii = interactionsums.loc[:,'end'] == s
    interactionsums.loc[ii, '# interactions'].sort_values().plot.line(label=s, ax=ax)
plt.legend(bbox_to_anchor=(1,1))
plt.show()


fig, ax = plt.subplots(figsize=(12,9))
saas = interactionsums.loc[:,'start'].unique()
eaas = interactionsums.loc[:,'end'].unique()
saas = [i for i in aminoacidfrequencyorder if i in saas]
eaas = [i for i in aminoacidfrequencyorder if i in eaas]
colormap = plt.cm.tab20c_r
colors = [colormap(i) for i in np.linspace(0, 1, len(aas))]
ax.set_prop_cycle('color', colors)

for s in saas:
    for e in eaas:
        si = interactionsums.loc[:,'start'] == s
        ei = interactionsums.loc[:,'end'] == e
        ii = np.logical_and(si, ei)
        bi = interactionsums.loc[ii, '# interactions'].sort_values()
        bi.plot.line(ax=ax, label=''.join((s, e)))
#plt.legend(bbox_to_anchor=(1,1))
plt.show()



#now get the locations of every pattern and check for overlap against every other pattern, this matrix may be too big to do all at once, you could probably multiprocess this in a one-at-a-time manner
#end results can be added to a matrix of just the motifs, not including their locations. This would show raw counted overlap. Self-overlap should be included.
#I guess this matrix needs to be formed for every individual protein... So that might give and take some difficulty
#Regarding this matrix, you don't only want to see if sums of a list of distances of these two motifs across proteins is low, you want to see how long the list is.

#neat for plotting frequencies around AAs or skellies, not entirely useful [yet?!]
#for p in patternspace:
#    for a in aminoacids:
#        p = 4
#        a = 'Y'
#        seqstring = f''.join(('(?=(', a, f'.{{{p}}}))'))
#        iorder = list(aminoacidfrequencies.keys())
#        psearch = re.compile(seqstring)
#        pairmatches = []
#        for protein in proteome:
#            pairmatches.extend(psearch.findall(protein))
#        positioncounts = []
#        for zpm in zip(*pairmatches):
#            positioncounts.append(Counter(zpm))
#        countedpositions = pd.DataFrame(positioncounts)
#        iorder = [i for i in iorder if i in countedpositions.columns]
#        colormap = plt.cm.cool
#        colors = [colormap(i) for i in np.linspace(0, 1, countedpositions.shape[1])]
#        fig, ax = plt.subplots(figsize=(12,9))
#        ax.set_prop_cycle('color', colors)
#        countedpositions.loc[:,iorder].iloc[1:].plot.line(ax=ax)
#        plt.legend(bbox_to_anchor=(1,1))
#        plt.show()
#        break
#    break

#bottom up would find smaller sequences and find out if they overlap at all, one big overlap matrix of their indices
#top-down would take longer skeletons and see the frequency for certain skeletons inside it

#Do a top-down
#Is theree a way to normalize frequencies per length? It's probably not necessary but could be an improvement some day.
#

#Resample based on proteins to determine ts shit
#Resample based on AA freqs to get motifs
#   > You're resampling to see how centered certain sequences are around specific AAs.
#   > Order:
#       1. Do a first-pass direct skeleton filtering, with inner/outer AAs decided (but also including 0-space motifs), the same way you have been to get all the more frequent picks per motif length
#       2. Linkage search, and initial subset finding. Find locations of every motif you've allowed past the filter, and see how many overlap, or are within a certain proximity to each other. The locations are the only piece of info needed, and each location - even if generated from the same motif - needs to be held to the same comparison. Every location gets compared to every other location, this will allow for the 'linkage' part of the search, that can expand motifs for long sequences - especially repeats.
#       2. Do an outward search for these motifs, like you have with your 'SLAVE' example, but with only a moving window [of different sizes], not all combinations. The idea should be to make it impossible to really fit a bad match. The 1-AA-sized moving window should obviously have better (and more) matches than the 2-AA-sized moving window, this could be some type of validity check. All these moving window alterations should be generated together, so that when these wild-card searches overlap with other motif's wild-card searches: you can group these two motifs!
#           > But doing the moving window will only match motifs of ~ the same size. You need to add a linkage step for subsets that are found in proximity of each other, and the linkage step needs to happen first so that the moving window will actually be worth something.
#           > ^ I wonder if this can be looked at in a different way:
#               >> You would take every AA-pair then expand it outward, NxM, NxxM, NxxxM, etc, to see if any repeating patterns emerge. You would expect these repeats to show the same abundance (maybe minus 1 per length) at each length of motif.
#               >> You should be able to do both this, and the initial subset search on the same data.
#       3. Form the output into a large levenshtein distance matrix
#           > This also may not be the only way
#       4. Cluster similar motifs like you did in aminoacidratiodistributions.py
#           > You can make inter-cluster cutoffs to refine the clusters too, ie refine the sequences deemed similar enough to be a 'motif'
#       5. Resample a fake proteome and look to see if the patterns surrounding these motifs still exist, and if the frequencies of AAs stays the same, or how often that happens.
#           > It would be perfectly fine for there to be outrageous/poor matches for a motif in a motif-group thing. For example perhaps if something that came from a hydropathy chart looked like a bad match sequence-wise. As long as the number of really bad matches are minimal compared to the main part of the motif group in terms of both # of motifs in the group, and as well as # of sequences in the proteome, then it should be fine to allow something like that to straggle. It likely won't have any negative effects.
#           > ^ Because this becomes the case for leniency vs the case for stringency, and this should favor the side of leniency because in the case where this motif IS related, then you might be able to find it in the wild with behavior that matches the rest of the group. In a way it's a confirmable/deniable question that the database can include.
#       6. Use these motif-subsets for real proteome resampling!

#You could factor nucleotide sequence conservation into the levenshtein distance matrix - instead of amino acid edits -> it would be theminimum # of nucleotide edits to take that AA change

#When visualizing the SLAVE search, you should have a NxSLAVE grid on your plot, with dark lines to gives individual squares to each space. Then the area inside each square will be represented by a specific amino acid (whatever hits come from your searches). The color of the sub-squares will represent individual AAs. Whatever AA's color is most prominent in each square will show how likely it is to be found in that spot. Goodbye stupid letter plots!
#For the NxSLAVE part, the N is going to be the size of the moving window. This will be the y-axis, so as the moving window gets larger, you'll probably see other AAs start to creep into the grid areas.
#And this can easily spread out past SLAVE to be xxSLAVExx, to show a solid motif start/end, perhaps with some variable surrounding AAs.
#Number of hits for each moving window search should be a horizontal bar chart on the side that matches up to each vertical grid.
