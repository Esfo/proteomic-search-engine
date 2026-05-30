import gc
from random import sample, choices
from collections import Counter, defaultdict
import concurrent
import numpy as np
from Bio import SeqIO
import pandas as pd
from scipy import cluster, stats
import networkx
from networkx.algorithms.components.connected import connected_components
import itertools
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib.colors import LinearSegmentedColormap
from time import time
#plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 20

proteome = '/home/sfo/data/fastas/proteomes/Fly_Drosophila_melanogaster.fasta'
#proteome = '/home/sfo/data/fastas/proteomes/Human_Homo_sapien.fasta'

aminoacids = 'ACTYKNMPWSQGHFDIVERL'
aminoacids = [i for i in aminoacids]

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

fasta = SeqIO.parse(open(proteome), 'fasta')
seqs = {}
counts = []
freqs = []
for fasta in fasta:
    sequence, idn = str(fasta.seq), fasta.id
    seqs[idn] = sequence
    paaf = {i:sequence.count(i)/len(sequence) for i in aminoacids}
    paac = {i:sequence.count(i) for i in aminoacids}
    freqs.append(paaf)
    counts.append(paac)

sequencelist = '|'.join((seqs.values()))
aminoacidfrequencies = {i:sequencelist.count(i)/(len(sequencelist)-sequencelist.count('|')) for i in aminoacids}
proteinsizes = [len(i) for i in seqs.values()]

f = pd.DataFrame(freqs)
of = f.copy()
f.loc[:,'length'] = proteinsizes

fig, axl = plt.subplots(nrows=5, ncols=4, figsize=(15,20), sharex=True, sharey=True)
ax = [item for sublist in axl for item in sublist]
ai = 0
for aa in f.columns:
    if len(aa) == 1:
        v = f.loc[:,(aa, 'length')].sort_values(aa)
        v.loc[:,'index'] = np.arange(len(v))
        v = v.sort_values('length')
        v.plot.scatter(x=aa, y='length',ax=ax[ai])
        ax[ai].set_title(aa)
        ax[ai].set_xlabel(None)
        ax[ai].set_ylabel(None)
        ai += 1
plt.show()

afdf = of.max(axis=0)
cycleorder = afdf.sort_values(ascending=False).index.tolist()

fig, ax = plt.subplots(figsize=(16,12))
colormap = plt.cm.tab20c_r
colors = [colormap(i) for i in np.linspace(0, 1, of.shape[1])]
ax.set_prop_cycle('color', colors)

for oc in cycleorder:
    of.loc[:,oc].sort_values().reset_index(drop=True).plot.line(ax=ax, linewidth=3)
plt.title('Distibution of AA %\'s in proteins')
plt.legend(bbox_to_anchor=(1,1))
plt.yscale('log')
plt.show()

#Proteins with an abnormally high frequency for any given AA tend to be extremely small proteins

#Question: Which amino acids have the most consistent ratios?

#taking ratio of every column with every other column, minus itself. Pandas keeps this process more tidy than a numpy array with bajillions of redundant/useless columns filled with 1.
#ratios = of.to_numpy().transpose() / np.expand_dims(of, 1).transpose()
#If you take out all of the zeros, then you get a more worthwhile dataset because the extremes are kept out, and you're left with a lot of generalizations. This helps keep out proteins that are super small that would likely skew any reasonable results.
#I still need to separately assess that proteins with a 0-count are often small


#of.replace(0, np.nan, inplace=True)
#of.dropna(axis=0, inplace=True)
#these 0s don't seem to influence anything now that I'm considering the cumulative values
nf = []
for o in of.columns:
    newcols = ['/'.join((o, i)) for i in  of.columns[of.columns != o]]
    nf.append(pd.DataFrame(of.loc[:,o].to_numpy().reshape(-1,1) / of.loc[:,of.columns != o].to_numpy(), columns=newcols))

nf = pd.concat(nf, axis=1)
nf.replace([np.inf, -np.inf], np.nan, inplace=True)
sortval = nf.columns[nf.std(axis=0).argmin()]
nf.sort_values(sortval, inplace=True)
nf.reset_index(inplace=True, drop=True)

#These are neat distributions, but hard to analyze all at once.
#fr = '/home/sfo/data/motifs/drosophila/aa-ratios-plots/'
#
#for r in nf.columns:
#    fs = ''.join((fr, r.replace('/', '-'), '.png'))
#    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(16,24))
#    nf.loc[:,r].sort_values().reset_index(drop=True).plot.hist(ax=ax1, bins=200)
#    nf.loc[:,r].sort_values().reset_index(drop=True).plot.line(ax=ax2, title=r, linewidth=5)
#    fig.savefig(fs, facecolor='white', transparent=False)
#    plt.close("all")
#    gc.collect()

#sf = nf.std(axis=0).copy()
#sf.name = 'stdev'
#sf = sf.to_frame()
#sf.drop('index', axis=0, inplace=True)
#sf.loc[:,'argsort'] = sf.loc[:,'stdev'].argsort().argsort()

#fig, ax = plt.subplots(figsize=(15,20))
#for aa in f.columns:
#    if len(aa) == 1:
#        ax.scatter(range(len(f)), f.loc[:,aa].sort_values().to_numpy(), label=aa)
#plt.legend()
#plt.show()


#this is the example of what's below in the loop to make npf
steadies = nf.mean(axis=0).loc[nf.mean(axis=0).sort_values().round() == 1].index.tolist()
sl = ''.join((steadies)).replace('/', '')
scounts = Counter(sl).most_common()

#nf is the ratio of every amino acid combination across every protein (minus proteins with 0 of any amino acid)
nmeans = nf.mean(axis=0).sort_values()

#this works out, but is hard to visualize as a plot because of missing data point, and the ratios can seem different depending on the number of bins you include.
#the next idea is to make these count values cumulative, that way an increase in resolution between bins doesn't skew the values. Can easily be done by doing a cumsum across the axis

pf = []
for nu in nmeans.unique():
    inrange = nmeans.loc[nmeans == nu]
    inind = ''.join((inrange.index.tolist())).replace('/', '')
    #inind = [i[0] for i in inrange.index.tolist()] #using this one as oppose to the one above so that ratios are always representative of a foreign normalization to a single AA. Otherwise there would be inconsistencies with the cumulative values.
    #^combining bot shows one of these plots ([0]) on top of the other ([2])
    incount = Counter(inind)
    af = pd.DataFrame.from_dict(incount, orient='index', columns=[nu])
    #nc += ndiff
    pf.append(af)
pf = pd.concat(pf, axis=1)

npf = np.nancumsum(pf, axis=1)
npf = pd.DataFrame(npf, columns=pf.columns, index=pf.index)
npf.sort_index(inplace=True)
npf.columns.name = 'mean'
npf.index.name = 'AA'

fig, ax = plt.subplots(figsize=(16,12))
colormap = plt.cm.tab20c
colors = [colormap(i) for i in np.linspace(0, 1, len(npf))]
ax.set_prop_cycle('color', colors)
npf.transpose().plot.line(ax=ax)
plt.ylabel('Cumulative Count of Involvement of Individual Amino Acids in Ratios')
plt.xlabel('Means of Amino Acid Ratio Distributions across Proteins')
plt.title('Involvement of Amino Acids across Means of Distribution of Ratios in Proteins')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.xscale('log')
plt.show()

#Current question for ^, do I count all AAs? Just the numerator, denominator, or both? Should I use all of these as features for a clustering?
#Also those axes labels and title are a mess fix that shit up.

#Do a connection-based clustering from scratch:

npagg = npf.unstack().reset_index().pivot_table(columns=0, values='AA', index='mean', aggfunc=list)
npagg.columns.name = 'cumulative sum'

interactioncounts = defaultdict(lambda: defaultdict(int))
for c in npagg.columns:
    iterframe = npagg.loc[~npagg.loc[:,c].isnull(), c].copy()
    for i in iterframe:
        if type(i) is list:
            for a1 in i:
                for a2 in i:
                    interactioncounts[a1][a2] += 1

ratiointeractions = pd.DataFrame(interactioncounts)

#cross-correlating only the amino acids that aren't being directly compared to find AAs that hang out with the same other AAs.
#^Important to add to the final notebook^
#By doing this step, you improved the reproducibility of using either [0] or [2] when creating inind in the loop up above ^^^. This makes that choice a bit irrelevant and 
rc = defaultdict(lambda: defaultdict(float))
for a1 in aminoacids:
    for a2 in aminoacids:
        baselist = [a1, a2]
        rc[a1][a2] += ratiointeractions.loc[[i not in baselist for i in ratiointeractions.index], baselist].corr().to_numpy()[0,1]

interactioncorrelations = pd.DataFrame(rc)
interactioncorrelations.sort_index(inplace=True)
interactioncorrelations = interactioncorrelations.loc[:,sorted(interactioncorrelations.columns.tolist())]

fig, ax = plt.subplots(figsize=(16,12))
#cax = ax.matshow(ratiointeractions.corr())
cax = ax.matshow(interactioncorrelations)
plt.xticks(range(len(interactioncorrelations)), interactioncorrelations.columns.tolist())
plt.yticks(range(len(interactioncorrelations)), interactioncorrelations.index.tolist())
fig.colorbar(cax)
plt.show()


#d = cluster.hierarchy.distance.pdist(interactioncorrelations)
#l = cluster.hierarchy.linkage(d, method='complete')
#ind = cluster.hierarchy.fcluster(l, 0.5*d.max(), 'distance')
#neworder = interactioncorrelations.columns[ind.argsort()]
#interactioncorrelations = interactioncorrelations.loc[neworder, neworder]

#fig, ax = plt.subplots(figsize=(16,12))
##cax = ax.matshow(ratiointeractions.corr())
#cax = ax.matshow(interactioncorrelations)
#plt.xticks(range(len(interactioncorrelations)), interactioncorrelations.columns.tolist())
#plt.yticks(range(len(interactioncorrelations)), interactioncorrelations.index.tolist())
#fig.colorbar(cax)
#plt.show()

#Highest ring:
#All members of the correlation club are put into multiple rings based on their numbers, and you multiply every number in the ring by itself. Then the sum of the ring's products are taken as the metric. This makes sense because if you just multiplied everything a second time then you'd always get the same number. What do the ansers of the smallest sum vs largest sum look like?
#Each group has to have at least 2.
#Assembled rings don't come from random combinations, they come from ranked correlations/interactions.
# - As you go down the ranked table, you group everything that matches up with 

#the correlations are nice for making this table, as opposed to pure amount from ratiointeractions, because of how they're considered using the absence of baselist above. It compares whether two AA's play ball with the same team, rather than that they always play ball together.
rankedtable = pd.DataFrame(interactioncorrelations.index.to_numpy()[np.argsort(interactioncorrelations.replace(1, np.nan).rank(method='dense', axis=0, ascending=False), axis=0).to_numpy()], columns=interactioncorrelations.columns, index=interactioncorrelations.index)

#taking advantage of python python bignum ints where numpy would have an overflow
ratiointeractions = ratiointeractions.astype(object)

frings = defaultdict(list)
for mr in range(len(rankedtable)):
    ringlist = []
    for rtc in range(rankedtable.shape[1]):
        adders = rankedtable.iloc[:mr+1, rtc].tolist()
        adders.append(rankedtable.columns[rtc])
        ringlist.append(adders)
    rings = to_graph(ringlist)
    rings = [i for i in connected_components(rings)]
    ringsandcount = {}
    gcount = 0
    for ri in rings:
        #groupcount = np.unique(ratiointeractions.loc[ri, ri].to_numpy()[~np.eye(len(ri), dtype=bool)]).prod()
        #normalizers = np.arange(len(ri)).astype(object) + 1
        #normalizers = [np.math.factorial(i) for i in normalizers]
        #normalizers = np.asarray(normalizers).astype(object)
        ##implementing a degrees of freedom-like normalization for each AA that's added, obviously more numbers in a group would lead to a larger product, this helps ensure the numbers are high numbers - to best represent related interactions as a score. This isn't a score per amino acid, this is a score per amino acid added. The factorials take into consideration the number of combinations of interactions possible with each added AA.
        #groupcount = np.divide.reduce(np.hstack((groupcount, normalizers)))
        #gcount += groupcount
        #newer scoring method
        groupcount = ((ratiointeractions.loc[ri,ri].sum(axis=1) - 380) / 380).sum() / len(ri)
        ringsandcount[''.join((sorted(ri)))] = groupcount
    if len(rings) > 1:
        frings[len(rings)].append(ringsandcount)

#ef = []
#for f in frings.keys():
#    tf = pd.DataFrame.from_dict(frings[f][0], orient='index')
#    tf.index.name = 'rings'
#    tf.columns = ['score']
#    tf.loc[:,'ncolumns'] = f
#    ef.append(tf)
#ef = pd.concat(ef)
#ef = ef.reset_index().set_index(['ncolumns', 'rings'])
#ef.sort_values(['ncolumns', 'score'], inplace=True)

#for lev in ef.index.levels[0]:
#    neworder = [i for i in ''.join((ef.loc[lev].index.tolist()))]
#    interactioncorrelations = interactioncorrelations.loc[neworder,neworder]
#    
#    fig, ax = plt.subplots(figsize=(16,12))
#    cax = ax.matshow(interactioncorrelations)
#    plt.xticks(range(len(interactioncorrelations)), interactioncorrelations.columns.tolist())
#    plt.yticks(range(len(interactioncorrelations)), interactioncorrelations.index.tolist())
#    fig.colorbar(cax)
#    plt.title(f'{lev} Ranks, {len(ef.loc[lev].index)} Clusters')
#    plt.show()
#    print(ef.loc[lev])

#so is rank really the best 'distance' to use? Based on where the green and yellows are this doesn't seem to be a terrible case, but it's not the best.

#explanations of groups at each rank as a label on the y-axis, x-axis will be either the sum of correlations, or the sum of the # of interactions. OR they could be determined by which 'group of groups' remained the longest as you iterated every column

#to prove the metric, I want to use something other than a rank-based step. I'll use a distance-based one instead:

interactiondistance = 1 - interactioncorrelations
stepmatrix = np.sort((interactiondistance).replace(0, np.nan).to_numpy(), axis=0)[:-1]
steps = np.diff(np.unique(stepmatrix.flatten()))
steps = np.hstack((0, steps))

movingstep = stepmatrix.min()
frings = defaultdict(list)
for s in steps:
    movingstep += s
    ringlist = []
    for idcol in interactiondistance.columns:
        adders = interactiondistance.loc[interactiondistance.loc[idcol] <= movingstep, idcol].index.tolist()
        ringlist.append(adders)
    rings = to_graph(ringlist)
    rings = [sorted(i) for i in connected_components(rings)]
    ringsandcount = {}
    for ri in rings:
        #groupcount = np.unique(ratiointeractions.loc[ri, ri].to_numpy()[~np.eye(len(ri), dtype=bool)]).prod()
        #normalizers = np.arange(len(ri)).astype(object) + 1
        #normalizers = [np.math.factorial(i) for i in normalizers]
        #normalizers = np.asarray(normalizers).astype(object)
        #implementing a degrees of freedom-like normalization for each AA that's added, obviously more numbers in a group would lead to a larger product, this helps ensure the numbers are high numbers - to best represent related interactions as a score. This isn't a score per amino acid, this is a score per amino acid added. The factorials take into consideration the number of combinations of interactions possible with each added AA.
        #groupcount = np.divide.reduce(np.hstack((groupcount, normalizers)))
        #old thing above new thing below
        #~~~~~ trying to represent how well the number of interactions are accounted for by this group of AAs as a number. Anything above 1 is means its more than fully accounted for, higher number should be the best.
        #if len(ri) == 1:
        #    groupcount = 1 / len(rings)
        #else:
        groupcount = ((ratiointeractions.loc[ri,ri].sum(axis=1) - 380) / 380).sum() / len(ri)
        #
        #below is the same for both
        ringsandcount[''.join((sorted(ri)))] = groupcount
    if len(rings) > 1:
        frings[len(rings)].append(ringsandcount)

for k in frings.keys():
    r1b, r2b = 0, 1
    r1, r2 = r1b, r2b
    while len(frings[k]) > 1:
        if frings[k][r1] == frings[k][r2]:
            frings[k].remove(frings[k][r2])
        else:
            r2 += 1
        if r2 >= len(frings[k]):
            r1b += 1
            r2b += 1
            r1, r2 = r1b, r2b


ef = []
for fr in frings.keys():
    tf = pd.DataFrame.from_dict(frings[fr][0], orient='index')
    tf.index.name = 'rings'
    tf.columns = ['score']
    tf.loc[:,'nclusters'] = fr
    ef.append(tf)
ef = pd.concat(ef)
ef = ef.reset_index().set_index(['nclusters', 'rings'])
ef.sort_values(['nclusters', 'score'], inplace=True)

#for lev in ef.index.levels[0]:
#    neworder = [i for i in ''.join((ef.loc[lev].index.tolist()))]
#    interactioncorrelations = interactioncorrelations.loc[neworder,neworder]
#    
#    fig, ax = plt.subplots(figsize=(16,12))
#    cax = ax.matshow(interactioncorrelations)
#    plt.xticks(range(len(interactioncorrelations)), interactioncorrelations.columns.tolist())
#    plt.yticks(range(len(interactioncorrelations)), interactioncorrelations.index.tolist())
#    fig.colorbar(cax)
#    plt.title(f'{lev} Ranks, {len(ef.loc[lev].index)} Clusters')
#    plt.show()
#    print(ef.loc[lev])

finalcluster = ef.loc[ef.index.levels[0][ef.sum(level=0).to_numpy().argmax()]]

neworder = [i for i in ''.join((finalcluster.index.tolist()))]
interactioncorrelations = interactioncorrelations.loc[neworder,neworder]

fig, ax = plt.subplots(figsize=(16,12))
cax = ax.matshow(interactioncorrelations)
plt.xticks(range(len(interactioncorrelations)), interactioncorrelations.columns.tolist())
plt.yticks(range(len(interactioncorrelations)), interactioncorrelations.index.tolist())
fig.colorbar(cax)
plt.title(f'{finalcluster.shape[0]} Clusters')
plt.show()
print(finalcluster)

fig, ax = plt.subplots(figsize=(16,12))
colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'brown']
for n, clusters in enumerate(finalcluster.index.tolist()):
    clusters = [i for i in clusters]
    npf.loc[clusters].transpose().plot.line(ax=ax, color=colors[n])
    plt.ylabel('Cumulative Count of Involvement of Individual Amino Acids in Ratios')
    plt.xlabel('Means of Amino Acid Ratio Distributions across Proteins')
    plt.title('Involvement of Amino Acids across Means of Distribution of Ratios in Proteins')
plt.legend(bbox_to_anchor=(1,1), title='clusters')
plt.tight_layout()
plt.xscale('log')
plt.show()

#now i need to separate the proteome into these clusters and visualize proteins in either nicely, maybe even in the distribution plots above

clusterorder = [[j for j in i] for i in finalcluster.index.tolist()]

fig, ax = plt.subplots(figsize=(16,12))
for n, oc in enumerate(clusterorder):
    nec = pd.DataFrame(np.sort(of.loc[:,oc].to_numpy(), axis=0), columns=oc)
    nec.plot.line(ax=ax, linewidth=3, color=colors[n])
plt.title('Distibution of AA %\'s in proteins')
plt.legend(bbox_to_anchor=(1,1))
plt.yscale('log')
plt.show()


#clusterfrequency = []
#for oc in clusterorder:
#    frame = of.loc[:,oc].sum(axis=1)
#    clusterfrequency.append(frame.copy())
#clusterfrequency = pd.concat(clusterfrequency, axis=1)
#clusterfrequency.columns = [''.join((oc)) for oc in clusterorder]
#
#clusterfrequency.plot.hist(bins=1000, figsize=(16,12))
#plt.legend(bbox_to_anchor=(1,1))
#plt.show()


cycleorder = afdf.sort_values(ascending=False).index.tolist()
fig, ax = plt.subplots(figsize=(16,12))
colormap = plt.cm.tab20c_r
colors = [colormap(i) for i in np.linspace(0, 1, of.shape[1])]
ax.set_prop_cycle('color', colors)

for oc in cycleorder:
    of.loc[:,oc].plot.hist(ax=ax, bins=100, alpha=0.5)
plt.title('Distibution of AA %\'s in proteins')
plt.legend(bbox_to_anchor=(1,1))
plt.show()

cmap1 = LinearSegmentedColormap.from_list('name', ['maroon', 'lightsalmon'])
cmap2 = LinearSegmentedColormap.from_list('name', ['darkslateblue', 'cornflowerblue'])
cmap3 = LinearSegmentedColormap.from_list('name', ['seagreen', 'olive'])
cmap4 = LinearSegmentedColormap.from_list('name', ['darkmagenta', 'orchid'])
cmap5 = LinearSegmentedColormap.from_list('name', ['darkorange', 'sandybrown'])
cmap6 = LinearSegmentedColormap.from_list('name', ['gold', 'khaki'])
cmap7 = LinearSegmentedColormap.from_list('name', ['slategrey', 'lightgrey'])

colormaps = [cmap1, cmap2, cmap3, cmap4, cmap5, cmap6, cmap7]
clusterorder = [[j for j in i] for i in finalcluster.index.tolist()]
colors = [[colormaps[i](j) for j in np.linspace(0, 1, len(co))] for i, co in  enumerate(clusterorder)]
fig, ax = plt.subplots(nrows=6, figsize=(18,12), sharex=True, sharey=True)
for n, oc in enumerate(clusterorder):
    ax[n].set_prop_cycle('color', colors[n])
    for ocn in oc:
        of.loc[:,ocn].plot.hist(ax=ax[n], bins=100, alpha=0.5)
#plt.xlim(0, 0.15)
plt.suptitle('Distribution of AA %\'s in Proteins')
hls = [i.get_legend_handles_labels() for i in ax]
handles, labels = [], []
for h, l in hls:
    for sh, sl in zip(h, l):
        handles.append(sh)
        labels.append(sl)
fig.legend(handles, labels, bbox_to_anchor=(1,0.85))
plt.show()

fig, ax = plt.subplots(figsize=(16,12))
for n, oc in enumerate(clusterorder):
    ax.set_prop_cycle('color', colors[n])
    for ocn in oc:
        of.loc[:,ocn].plot.hist(ax=ax, bins=100, alpha=0.3)
plt.title('Distibution of AA %\'s in Proteins')
plt.legend(bbox_to_anchor=(1,1))
plt.show()


#now resample the AA combinations -> get their ring heat, and do 2 things:
#   > See how the ring heat metric compares to p-values when resampling
#   > See how difference of ratios of clusters from 1 (the perfect ratio) in the distributions of your clusters compare to the resampled combinations

def groupscore(ri):
    ri = set(ri)
    risd = ri.symmetric_difference(aminoacids)
    ris = [ri, risd]
    gc = []
    for r in ris:
        groupcount = np.unique(ratiointeractions.loc[r, r].to_numpy()[~np.eye(len(r), dtype=bool)]).prod()
        normalizers = [np.math.factorial(i+1) for i in range(len(r))]
        normalizers = np.asarray(normalizers).astype(object)
        gc.append(np.divide.reduce(np.hstack((groupcount, normalizers))))
    return gc


def tf(aminoacids):
    with concurrent.futures.ProcessPoolExecutor(8) as executor:
        g = 0
        futures, results = [], []
        #for n in range(len(aminoacids)-1):
        n = 0
        for ac in combinations(aminoacids, n+1):
            futures.append(executor.submit(groupscore, ac))
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results

#^above should take 17 minutes to get an exact calculation, 
#you should be resampling out of nf

#this is honestly stupid at this point, you could find all combinations that sum to 20 AAs total from the resulting ef frame, but there isn't a ton of prospect for this work for me atm...

#Find dependency-like cycles among AA's, where if one protein has more of something, it tends to have less of something else, and vice-versa. these would be your clusters.
#It would then be interesting to see to what percentile different proteins overlap in these groups, and if you can find any functional relevance that you can scrape off of uniprot in these groups, or from GO terms.

#this approach focussed on the similarity of intracluster similarity, I want the next one to focus on intercluster difference.

#so using nf you should find the most opposing dense ranks
#the largest absolute difference of two ranks should display not just difference, but opposing difference - a sort of 'one or the other' measurement.
oranks = of.rank(method='dense', axis=0)

orankdifferences = defaultdict(lambda: defaultdict(float))
nfratiodifferences = defaultdict(lambda: defaultdict(float))
for a1 in aminoacids:
    for a2 in aminoacids:
        if a1 != a2:
            baselist = [a1, a2]
            nindices = [i for i in nf.columns if sum(i.count(c) for c in baselist) == 2]
            orankdifferences[a1][a2] = oranks.loc[:,baselist].diff(axis=1).sum()[1]
            nfratiodifferences[a1][a2] = nf.loc[:,nindices].diff(axis=1).sum()[1]

orankdifferences = pd.DataFrame(orankdifferences).abs()
nfratiodifferences = pd.DataFrame(nfratiodifferences).abs()

#neworder = [i for i in ''.join((finalcluster.index.tolist()))]
neworder = [i for i in aminoacids]
orankdifferences = orankdifferences.loc[neworder,neworder]
nfratiodifferences = nfratiodifferences.loc[neworder,neworder]

fig, ax = plt.subplots(figsize=(16,12))
cax = ax.matshow(orankdifferences)
plt.xticks(range(len(orankdifferences)), orankdifferences.columns.tolist())
plt.yticks(range(len(orankdifferences)), orankdifferences.index.tolist())
fig.colorbar(cax)
plt.show()

fig, ax = plt.subplots(figsize=(16,12))
cax = ax.matshow(nfratiodifferences)
plt.xticks(range(len(nfratiodifferences)), nfratiodifferences.columns.tolist())
plt.yticks(range(len(nfratiodifferences)), nfratiodifferences.index.tolist())
fig.colorbar(cax)
plt.show()



#plot y as percentages of either 2 AAs, then the x-scale would be the difference between every proteins percent - 2 y-values both having the same x
#try with both rank difference, ratiodifference, and percentage differences from of.
#fr = '/home/sfo/data/motifs/drosophila/aa-percent-dependence-plots/'
#for a1, a2 in itertools.combinations(aminoacids, 2):
#    baselist = [a1, a2]
#    bt = ''.join((baselist))
#    fs = ''.join((fr, bt, '.png'))
#    pf = of.loc[:,baselist].copy()
#    pf.loc[:,'diff'] = pf.diff(axis=1).to_numpy()[:,1]
#    pf.sort_values('diff', inplace=True)
#    
#    fig, ax = plt.subplots(figsize=(16,12))
#    pf.plot.scatter(x='diff', y=a1, ax=ax, color='purple', alpha=0.1, label=a1)
#    pf.plot.scatter(x='diff', y=a2, ax=ax, color='green', alpha=0.1, label=a2)
#    plt.ylabel('%')
#    plt.title(bt)
#    leg = plt.legend()
#    for lh in leg.legendHandles:
#        lh.set_alpha(1)
#    fig.savefig(fs, facecolor='white', transparent=False)
#    plt.close("all")
#    gc.collect()
#    #plt.show()

experimentalclusters = ef.loc[2].index.tolist()
experimentalclusters = ['LSW', 'CHMY']
pf = []
for e in experimentalclusters:
    baselist = [i for i in e]
    pn = of.loc[:,baselist].sum(axis=1).copy()
    pf.append(pn)
pf = pd.concat(pf, axis=1)
pf.columns = experimentalclusters
pf.loc[:,'diff'] = pf.diff(axis=1).to_numpy()[:,1]
pf.loc[:,'ratio'] = np.divide.reduce(pf.loc[:,experimentalclusters], axis=1)
pf.sort_values('diff', inplace=True)

fig, ax = plt.subplots(figsize=(16,12))
pf.plot.scatter(x='diff', y=experimentalclusters[0], ax=ax, color='purple', alpha=0.1, label=experimentalclusters[0])
pf.plot.scatter(x='diff', y=experimentalclusters[1], ax=ax, color='green', alpha=0.1, label=experimentalclusters[1])
plt.ylabel('%')
plt.title('Experimental Clusters')
leg = plt.legend()
for lh in leg.legendHandles:
    lh.set_alpha(1)
plt.show()

#These are pointless endeavors because biology has evolved far past the use of just individual amino acids. Analyzing motifs is the right idea.




cf = pd.DataFrame(counts)
cfnz = cf.loc[~(cf == 0).any(axis=1)].astype(float)
#this doesn't show anything interesting, every random combination of AA counts gets either a good or great correlation like this
#clustersets = [set(i) for i in finalcluster.index]
#tf = pd.DataFrame()
#for oc in clustersets:
#    #frameinds = [len(oc.intersection(i)) for i in nf.columns]
#    #frameinds = np.asarray(frameinds)
#    #frameinds = frameinds > 1
#    #frame = of.loc[:,oc].sum(axis=1)
#    #clusterfrequency.append(frame.copy())
#    tf.loc[:,''.join((sorted(oc)))] = cf.loc[:,oc].sum(axis=1)
#tf.loc[:,'length'] = f.loc[:,'length'].copy()
#tf.loc[:,'ratio'] = tf.loc[:,'ACFHLMSWY'].to_numpy() / tf.loc[:,'DEGIKNPQRTV'].to_numpy()
#
#tf.plot.scatter(x='ACFHLMSWY', y='DEGIKNPQRTV', figsize=(16,12))
#plt.title('Count of Amino Acids by Cluster')
#plt.show()

clustersets = [set(i) for i in finalcluster.index]
ratiostats = pd.DataFrame()
for oc in clustersets:
    frameinds = [len(oc.intersection(i)) for i in nf.columns]
    frameinds = np.asarray(frameinds)
    frameinds = frameinds > 1
    ratiostats.loc[:,''.join((sorted(oc)))] = nf.loc[:,frameinds].mean(axis=1)
ratiostats.loc[:,'ratio'] = np.divide.reduce(ratiostats, axis=1)
ratiostats.loc[:,'length'] = f.loc[:,'length'].copy()

#nsamples = 1000
#for n in nsamples:
#    samp = set(np.random.choice(aminoacids, np.random.randint(low=0, high=len(aminoacids)-1)))
#    sampsd = samp.symmetric_difference(aminoacids)
#    r1 = np.divide.reduce(cfnz.loc[:,samp], axis=1)
#    r2 = np.divide.reduce(cfnz.loc[:,sampsd], axis=1)
#
#    cor = np.corrcoef(cf.loc[:,samp].sum(axis=1), cf.loc[:,sampsd].sum(axis=1))[0,1]


def independent_skeletons(p, joinedproteome, nothanks):
    #counts every p-length skeleton motif available in each protein
    per = p*'.'
    #collecting motifs in this manner assumes order of the sequence matters, ie G..T is different than T..G
    splits = set(zip(joinedproteome[:-p], joinedproteome[p:]))
    splits = [f'{per}'.join((i)) for i in splits if not any(r in i for r in nothanks)]
    outlist = []
    for s in splits:
        skeletonmatches = Counter(regex.findall(s, joinedproteome, overlapped=True, concurrent=False))
        for sm in tuple(skeletonmatches.keys()):
            if any(r in sm for r in nothanks):
                skeletonmatches.pop(sm)
        outlist.append(frequency_filter(skeletonmatches))
    return outlist

t = time()
with concurrent.futures.ProcessPoolExecutor(8) as executor:
    futures, skeletons = [], []
    for p in patternspace:
        futures.append(executor.submit(independent_skeletons, p, joinedproteome, nothanks))
    for future in concurrent.futures.as_completed(futures):
        skeletons.extend(future.result())
print(time() - t)
