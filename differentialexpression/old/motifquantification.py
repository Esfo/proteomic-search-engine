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
from iklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, LeaveOneOut
import numpy as np

folder = '/store/drosophila/PXD005713/crux-output'

#proteome = '/home/sfo/data/fastas/proteomes/Fly_Drosophila_melanogaster_Isoforms.fasta'
proteome = '/home/sfo/data/fastas/proteomes/Fly_Drosophila_melanogaster.fasta'

#outfile2 = '/home/sfo/data/motifs/Fly_Drosophila_melanogaster_distance_ps4.csv'
#outfile = '/home/sfo/data/motifs/Fly_Drosophila_melanogaster_full-filtered_distance_ps2_50.csv'
outstart = '/home/sfo/data/motifs/drosophila/by-motif/Fly_Drosophila_melanogaster_full-filtered-part_distance_ps'
#outgeneration = True
outgeneration = False

patternfile = '/store/drosophila/PXD005713/full4.patterns.pickle'

fasta = SeqIO.parse(open(proteome), 'fasta')
seqs = {}
descs = {}
for fasta in fasta:
    sequence, idn = str(fasta.seq), fasta.id
    seqs[idn] = sequence
    descs[idn] = fasta.description

ef = []
nsafs = []

files = [i for i in os.listdir(folder) if not i.startswith('full') and not i.startswith('unique') and '.' in i]
files = list(set([i.split('.')[0] for i in files]))

for f in files:
    split = f.split('_')
    timepoint = int(split[-2].split('h')[0])
    replicate = split[-1]
    
    #nsname = ''.join((folder, '/full.', f, '.spectral-counts.txt'))
    nsname = ''.join((folder, '/full.', f, '.spectral-counts.txt'))
    nfdf = pd.read_csv(nsname, delimiter='\t')
    nfdf.loc[:,'hour'] = timepoint
    nfdf.loc[:,'replicate'] = replicate
    nsafs.append(nfdf)

nsafs = pd.concat(nsafs)
npiv = nsafs.pivot_table(index='ProteinId', columns=['hour', 'replicate'], values=['NSAF', 'SAF', 'rank'])

proteins = seqs.keys()
#sequencelist = seqs.values()
sequencelist = '|'.join((seqs.values()))

with open(patternfile, "rb") as pick:
    patterns = pickle.load(pick)
    print('loaded', patternfile)

def writer1(outies):
    df = pd.DataFrame(outies)
    df.to_csv(outfile, header=False, index=False, mode='a')

def writer2(outies, plen):
    df = pd.DataFrame(outies)
    outfile = ''.join((outstart, str(plen), '.csv'))
    df.to_csv(outfile, header=False, index=False, mode='a')

#if outgeneration:
#    #for pnum in range(pmin, pmax):
#    #    startframe = pd.DataFrame(columns=['Protein', 'Motif', 'Starting Position', 'Distance from C-Term', 'Protein Length'])
#    #    outfile = ''.join((outstart, str(pnum), '.csv'))
#    #    startframe.to_csv(outfile, index=False)
#    startframe = pd.DataFrame(columns=['Protein', 'Motif', 'Starting Position', 'Distance from C-Term', 'Protein Length'])
#    startframe.to_csv(outfile, index=False)
#    pn = 0
#    with concurrent.futures.ThreadPoolExecutor(8) as executor:
#        for patternstring in patterns:
#            plen = len(patternstring)
#            outies = []
#            sys.stdout.write(f'\r{pn}/{len(patterns)}, length {plen}')
#            sys.stdout.flush()
#            for k, p in seqs.items():
#                search = re.compile(patternstring)
#                matches = search.finditer(p)
#                for m in matches:
#                    outlist = [k, patternstring, m.start(), m.endpos - m.start(), m.endpos]
#                    outies.append(outlist)
#            pn += 1
#            executor.submit(writer1, outies)
#            #executor.submit(writer2, outies, plen)
#else:
#    print(f'loading outsets')

#maxsize = 8
#minsize = 3
#chunksize = 1000000
#df = []
#for d in pd.read_csv(outfile, keep_default_na=False, na_values=['____'], chunksize=chunksize):
#    d.loc[:,'motif length'] = d.loc[:,'Motif'].apply(lambda x: len(x))
#    d = d.loc[np.logical_and(d.loc[:,'motif length'] <= maxsize, d.loc[:,'motif length'] >= minsize)]
#    df.append(d)
#    gc.collect()
#df = pd.concat(df)

psize = 4
outfile = ''.join((outstart, str(psize), '.csv'))
df = pd.read_csv(outfile, low_memory=False, keep_default_na=False, na_values=['____'])
print(f'{outfile} loaded')
df.set_index('Motif', inplace=True)
df.sort_index(inplace=True)
gc.collect()

target = 'Y..P'
targetproteins = set(df.loc[target, 'Protein'])

visibletargets = targetproteins.intersection(npiv.index)

tpiv = npiv.loc[visibletargets]

antitargets = visibletargets.symmetric_difference(npiv.index)
fpiv = npiv.loc[antitargets]
fcounts = (fpiv > 0).sum(axis=1) == fpiv.shape[1]
fcvars = fpiv.loc[fcounts, 'SAF'].var(axis=1).to_frame()
fcvars.columns = ['vars']
fcvars.loc[:,'means'] = fpiv.loc[fcounts, 'SAF'].mean(axis=1)
fcvars.loc[:,'comb'] = (1 / fcvars.loc[:,'vars']) * fcvars.loc[:,'means']
fcvars.sort_values('comb', inplace=True)
normalizer = fcvars.index[-1]

#normalizing by most consistent SAF across files
ntpiv = tpiv.loc[:,'NSAF'] / npiv.loc[normalizer, 'NSAF']
ntpiv.fillna(0, inplace=True)

nnpiv = npiv.loc[npiv.index != normalizer,'NSAF'] / npiv.loc[normalizer, 'NSAF']
nnpiv.fillna(0, inplace=True)

npw = npiv.loc[:,'NSAF'].count(axis=0)

#weighted mean of ntpiv values across replicates, weighted by overall coverage
wtnsaf = (npw * ntpiv).sum(axis=1, level=0) / npw.sum(axis=0, level=0)
wtnsaf = wtnsaf.loc[(wtnsaf > 0).sum(axis=1) > 0]
wtnsaf.sort_index(inplace=True)

wnsaf = (npw * nnpiv).sum(axis=1, level=0) / npw.sum(axis=0, level=0)
wnsaf = wnsaf.loc[(wnsaf > 0).sum(axis=1) > 0]
wnsaf.sort_index(inplace=True)
visibletargets = wnsaf.index.intersection(targetproteins)

df.loc[:,'%D from CT'] = df.loc[:,'Distance from C-Term'] / df.loc[:,'Protein Length']

targetsites = df.loc[target]
targetsites = targetsites.reset_index().set_index('Protein')
targetagg = targetsites.reset_index().groupby('Protein').agg({'Motif': 'count', '%D from CT': 'mean', 'Protein Length':'mean'})

vtsites = targetagg.loc[visibletargets]
vtsites.sort_index(inplace=True)
vtsites.loc[:,'M/AA'] = vtsites.loc[:,'Motif'] / vtsites.loc[:,'Protein Length']
vtsites.loc[:,'P/ML'] = vtsites.loc[:,'Protein Length'] / len(target)
vtsites.loc[:,'M/P/ML'] = vtsites.loc[:,'Motif'] / vtsites.loc[:,'P/ML']

#motifcount = wtnsaf.multiply(vtsites.loc[:,'Motif'].to_numpy().reshape(-1,1))
motifcount = wnsaf.loc[visibletargets].multiply(vtsites.loc[:,'Motif'].to_numpy().reshape(-1,1))
nmotifcount = wnsaf.loc[visibletargets].multiply(vtsites.loc[:,'M/AA'].to_numpy().reshape(-1,1))

motifcount = wnsaf.loc[visibletargets].multiply(vtsites.loc[:,'Motif'].to_numpy().reshape(-1,1))
nmotifcount = wnsaf.loc[visibletargets].multiply(vtsites.loc[:,'M/P/ML'].to_numpy().reshape(-1,1))

texpressiongroups = (wtnsaf > 0).sum(axis=1)
expressiongroups = (wnsaf > 0).sum(axis=1)

variableexpression = wnsaf.loc[expressiongroups < 8].index
steadyexpression = wnsaf.loc[expressiongroups > 7].index

#number of times a protein with a YxxP motif shows up
texpressiongroups.plot.hist(bins=14)
plt.title('Number of Timepoints a Protein with a YxxP Sequence is Detected')
plt.xlabel('#')
plt.show()

variabletargets = texpressiongroups.loc[texpressiongroups < 8].index
steadytargets = texpressiongroups.loc[texpressiongroups > 7].index

#number of motifs on variable groups were more enriched at the 20 hour point
motifcount.loc[variabletargets].sum(axis=0).plot.bar()
plt.title('Number of YxxP Sequences on Variable Groups Across Timepoints')
plt.show()

#bot = np.zeros(len(texpressiongroups.unique()))
#fig, ax = plt.subplots(figsize=(12,9))
#for n, g in enumerate(reversed(sorted(texpressiongroups.unique()))):
#    ginds = texpressiongroups[texpressiongroups == n].index
#    wtnsaf.loc[ginds].sum(axis=0).plot.bar(color=colors[n], bottom=bot, width=0.7)
#    bot += wtnsaf.loc[ginds].sum(axis=0)
#plt.xticks(rotation=0, fontsize=11)
#plt.show()

#number of times a protein shows up across all found proteins
groupcount = Counter(expressiongroups)
nwnsaf = wnsaf / wnsaf.sum()

x = list(groupcount.keys())
y = list(groupcount.values())
x, y = np.asarray(x), np.asarray(y)

y = y[x.argsort()]
x.sort()

data_color = [c / max(x) for c in x]
my_cmap = plt.cm.get_cmap('BrBG')
colors = my_cmap(data_color)
fromhex = matplotlib.colors.to_rgb('#690000')
#fromhex = matplotlib.colors.to_rgb('#d62020')
#fromhex = matplotlib.colors.to_rgb('#f02626')
rcol = list(fromhex)
for n, r in enumerate(rcol):
    colors[0][n] = r

frc = '#bd0000'
plt.rcParams["hatch.color"] = matplotlib.colors.to_rgb('#e6d1d1')
mhatch = '//'

motifcount = motifcount / motifcount.sum(axis=0)
nmotifcount = nmotifcount / nmotifcount.sum(axis=0)

#fig, ax = plt.subplots(nrows=3, figsize=(3.5,10))
#fig, ax = plt.subplots(nrows=4, figsize=(3.5,14))
fig, a = plt.subplots(nrows=2, ncols=2, figsize=(12,8))
ax = [a[0][0], a[0][1], a[1][0], a[1][1]]

ax[0].barh(x[0], y[0], color=colors[0], height=1, hatch=mhatch)
ax[0].barh(x[1:], y[1:], color=colors[1:], height=1)
ax[0].set_yticks(np.arange(len(groupcount))+1)
ax[0].set_xlabel('# Proteins', fontsize=12)
ax[0].set_ylabel('# Time Points', fontsize=12)
#ax[0].set_title('Protein Detection Frequencies', fontsize=15)
ax[0].tick_params(rotation=0, size=12)

#Number of identified proteins, qualitative counts
bot = np.zeros(len(expressiongroups.unique()))
for n, g in enumerate(sorted(expressiongroups.unique())):
    ginds = expressiongroups[expressiongroups == g].index
    if n == 0:
        (wnsaf.loc[ginds] > 0).sum(axis=0).plot.bar(color=colors[n], bottom=bot, width=0.7, ax=ax[1], hatch=mhatch)
    else:
        (wnsaf.loc[ginds] > 0).sum(axis=0).plot.bar(color=colors[n], bottom=bot, width=0.7, ax=ax[1])
    bot += (wnsaf.loc[ginds] > 0).sum(axis=0)
ax[1].tick_params(rotation=0, size=12)
#ax[1].set_title('Proteins Identified per Timepoint', fontsize=15)
ax[1].set_ylabel('# Proteins', fontsize=12)
ax[1].set_xlabel('Hour', fontsize=12)

#all proteins showing variable and steady expression
bot = np.zeros(len(expressiongroups.unique()))
for n, g in enumerate(sorted(expressiongroups.unique())):
    ginds = expressiongroups[expressiongroups == g].index
    if n == 0:
        nwnsaf.loc[ginds].sum(axis=0).plot.bar(color=colors[n], bottom=bot, width=0.7, ax=ax[2], hatch=mhatch)
    else:
        nwnsaf.loc[ginds].sum(axis=0).plot.bar(color=colors[n], bottom=bot, width=0.7, ax=ax[2])
    bot += nwnsaf.loc[ginds].sum(axis=0)
ax[2].tick_params(rotation=0, size=12)
#ax[2].set_title('Relative Quantity of Proteins per Frequency', fontsize=15)
ax[2].set_ylabel('Log NSAF', fontsize=12)
ax[2].set_xlabel('Hour', fontsize=12)
ax[2].set_yscale('log')

bot = np.zeros(len(expressiongroups.unique()))
for n, g in enumerate(sorted(expressiongroups.unique())):
    ginds = expressiongroups[expressiongroups == g].index.intersection(visibletargets)
    if n == 0:
        motifcount.loc[ginds].sum(axis=0).plot.bar(color=colors[n], bottom=bot, width=0.7, ax=ax[3], hatch=mhatch)
    else:
        motifcount.loc[ginds].sum(axis=0).plot.bar(color=colors[n], bottom=bot, width=0.7, ax=ax[3])
    bot += motifcount.loc[ginds].sum(axis=0)
ax[3].tick_params(rotation=0, size=11)
#ax[3].set_title('YxxP Motifs per Protein among YxxP Containing Proteins', fontsize=15)
ax[3].set_ylabel('Log NSAF', fontsize=12)
ax[3].set_xlabel('Hour', fontsize=12)
ax[3].set_yscale('log')

ax[0].text(-0.1, 1.05, 'A', fontsize=25, transform=ax[0].transAxes)
ax[1].text(-0.1, 1.05, 'B', fontsize=25, transform=ax[1].transAxes)
ax[2].text(-0.1, 1.05, 'C', fontsize=25, transform=ax[2].transAxes)
ax[3].text(-0.1, 1.05, 'D', fontsize=25, transform=ax[3].transAxes)
#fig.savefig('/home/sfo/docs/grants/fig2.png')
#plt.subplots_adjust(hspace=0.3)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(nrows=2, figsize=(6,12))

bot = np.zeros(len(expressiongroups.unique()))
for n, g in enumerate(sorted(expressiongroups.unique())):
    ginds = expressiongroups[expressiongroups == g].index.intersection(visibletargets)
    motifcount.loc[ginds].sum(axis=0).plot.bar(color=colors[n], bottom=bot, width=0.7, ax=ax[0])
    bot += motifcount.loc[ginds].sum(axis=0)
ax[0].tick_params(rotation=0, size=11)
ax[0].set_title('YxxP Motifs per Protein', fontsize=15)
ax[0].set_ylabel('NSAF', fontsize=12)
ax[0].set_xlabel('Hour', fontsize=12)
ax[0].set_yscale('log')

bot = np.zeros(len(expressiongroups.unique()))
for n, g in enumerate(sorted(expressiongroups.unique())):
    ginds = expressiongroups[expressiongroups == g].index.intersection(visibletargets)
    nmotifcount.loc[ginds].sum(axis=0).plot.bar(color=colors[n], bottom=bot, width=0.7, ax=ax[1])
    bot += nmotifcount.loc[ginds].sum(axis=0)
ax[1].tick_params(rotation=0, size=11)
ax[1].set_title('Sum of YxxP Motifs per Protein per \n Protein Length per Motif Length', fontsize=15)
ax[1].set_ylabel('NSAF', fontsize=12)
ax[1].set_xlabel('Hour', fontsize=12)
ax[1].set_yscale('log')

plt.subplots_adjust(hspace=0.3)
plt.show()

#motifcount.sum(axis=0).plot.bar()
#plt.title('Number of YxxP Sequences on Variable Groups Across Timepoints')
#plt.show()

#fig, ax = plt.subplots(figsize=(12,9))
#for n, g in enumerate(sorted(expressiongroups.unique())):
#    vtwinds = wnsaf.loc[expressiongroups == g, 20].loc[wnsaf.loc[expressiongroups == g, 20] > 0].index.intersection(visibletargets)
#    yvals = targetsites.loc[vtwinds, '%D from CT'].to_numpy()
#    yv = stats.gaussian_kde(yvals)(yvals)
#    xvals = np.repeat(g, len(yvals))
#    pf = pd.DataFrame(columns=['x', 'y', 'yd'])
#    pf.loc[:,'x'] = xvals
#    pf.loc[:,'y'] = yvals
#    pf.loc[:,'yd'] = yv
#    pf.sort_values('yd').plot.scatter(x='x', y='y', c='yd', colormap='cool', ax=ax, alpha=0.1, colorbar=False, marker='s', s=1500)
#    #make the points squares, then enlarge them
#ax.tick_params(rotation=0, size=11)
#ax.set_title('Kernel Density Estimation of Percent Distance of YxxP \n to C-Terminus at 20 Hours by Frequency of Detection', fontsize=15)
#ax.set_ylabel('% Distance to C-Terminus', fontsize=12)
#ax.set_xlabel('# Timepoints', fontsize=12)
#ax.set_xticks(np.arange(len(groupcount))+1)
#im = plt.gca().get_children()[0]
#fig.subplots_adjust(right=0.8)
#cax = fig.add_axes([0.85,0.15,0.03,0.7])
#im.set_alpha(1)
#cb = fig.colorbar(im, cax=cax)
#cb.set_label(label='Density', fontsize=12, rotation=270, labelpad=18)
#plt.show()

#fig, ax = plt.subplots(figsize=(12,9))

bandwidths = 10 ** np.logspace(-1, 1, 10)

fig, ax = plt.subplots(figsize=(7,6))
for n, c in enumerate(sorted(wnsaf.columns)):
    tinds = wnsaf.loc[wnsaf.loc[:,c] > 0].index.intersection(visibletargets)
    yvals = targetsites.loc[tinds, '%D from CT'].to_numpy()
    #yv = stats.gaussian_kde(yvals)(yvals)
    
    grid = GridSearchCV(KernelDensity(),{'bandwidth': bandwidths}, cv=5)
    grid.fit(yvals.reshape(-1,1))
    kde = grid.best_estimator_
    yu = np.unique(yvals).reshape(-1,1)
    yv = kde.score_samples(yu)
    
    xvals = np.repeat(n, len(yv))
    pf = pd.DataFrame(columns=['x', 'y', 'yd'])
    pf.loc[:,'x'] = xvals
    pf.loc[:,'y'] = yu
    pf.loc[:,'yd'] = yv
    pf.sort_values('yd').plot.scatter(x='x', y='y', c='yd', colormap='cool', ax=ax, alpha=0.1, colorbar=False, marker='s', s=470)
ax.tick_params(rotation=0, size=11)
#ax.set_title('Kernel Density Estimation of Percent Distance of YxxP \n to C-Terminus by Timepoint', fontsize=15)
ax.set_ylabel('% Distance to C-Terminus', fontsize=16)
ax.set_xlabel('Hour', fontsize=16)
xtvs = np.arange(len(wnsaf.columns))
ax.set_xticks(xtvs)
ax.set_xticklabels(wnsaf.columns)
im = plt.gca().get_children()[0]
fig.subplots_adjust(right=0.8)
#cax = fig.add_axes([0.85,0.15,0.03,0.7])
#im.set_alpha(1)
#cb = fig.colorbar(im, cax=cax)
#cb.set_label(label='Density', fontsize=18, rotation=270, labelpad=18)
for label in ax.get_xticklabels():
    label.set_fontsize(15)
for label in ax.get_yticklabels():
    label.set_fontsize(15)
#cb.ax.tick_params(labelsize=15)
plt.show()


#fig, ax = plt.subplots(figsize=(12,9))
fig, ax = plt.subplots(figsize=(7,6))
for n, c in enumerate(sorted(wnsaf.columns)):
    tinds = wnsaf.loc[wnsaf.loc[:,c] > 0].index.intersection(visibletargets)
    yvals = targetsites.loc[tinds, '%D from CT'].to_numpy()
    yv = stats.gaussian_kde(yvals, weights=wnsaf.loc[targetsites.loc[visibletargets].loc[tinds].index, c].to_numpy())(yvals)
    xvals = np.repeat(n, len(yvals))
    pf = pd.DataFrame(columns=['x', 'y', 'yd'])
    pf.loc[:,'x'] = xvals
    pf.loc[:,'y'] = yvals
    pf.loc[:,'yd'] = yv
    pf.sort_values('yd').plot.scatter(x='x', y='y', c='yd', colormap='cool', ax=ax, alpha=0.1, colorbar=False, marker='s', s=470)
ax.tick_params(rotation=0, size=11)
#ax.set_title('Quantification-Weighted Kernel Density Estimation of Percent \n Distance of YxxP to C-Terminus by Timepoint', fontsize=15)
ax.set_ylabel('% Distance to C-Terminus', fontsize=18)
ax.set_xlabel('Hour', fontsize=18)
xtvs = np.arange(len(wnsaf.columns))
ax.set_xticks(xtvs)
ax.set_xticklabels(wnsaf.columns)
im = plt.gca().get_children()[0]
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85,0.15,0.03,0.7])
im.set_alpha(1)
cb = fig.colorbar(im, cax=cax)
cb.set_label(label='Density', fontsize=18, rotation=270, labelpad=18)
for label in ax.get_xticklabels():
    label.set_fontsize(15)
for label in ax.get_yticklabels():
    label.set_fontsize(15)
cb.ax.tick_params(labelsize=15)
plt.show()


#singleexpressers = texpressiongroups.loc[texpressiongroups == 1].index
#fig, ax = plt.subplots(figsize=(12,9))
#for n, c in enumerate(sorted(wnsaf.columns)):
#    tinds = wnsaf.loc[wnsaf.loc[:,c] > 0].index.intersection(singleexpressers)
#    yvals = targetsites.loc[tinds, '%D from CT'].to_numpy()
#    yv = stats.gaussian_kde(yvals)(yvals)
#    xvals = np.repeat(n, len(yvals))
#    pf = pd.DataFrame(columns=['x', 'y', 'yd'])
#    pf.loc[:,'x'] = xvals
#    pf.loc[:,'y'] = yvals
#    pf.loc[:,'yd'] = yv
#    pf.sort_values('yd').plot.scatter(x='x', y='y', c='yd', colormap='cool', ax=ax, alpha=0.1, colorbar=False, marker='s', s=1400)
#    #make the points squares, then enlarge them
#ax.tick_params(rotation=0, size=11)
#ax.set_title('Kernel Density Estimation of Percent Distance of YxxP \n to C-Terminus of Proteins Found at Only 1 Timepoint', fontsize=15)
#ax.set_ylabel('% Distance to C-Terminus', fontsize=12)
#ax.set_xlabel('Hour', fontsize=12)
#xtvs = np.arange(len(wnsaf.columns))
#ax.set_xticks(xtvs)
#ax.set_xticklabels(wnsaf.columns)
#im = plt.gca().get_children()[0]
#fig.subplots_adjust(right=0.8)
#cax = fig.add_axes([0.85,0.15,0.03,0.7])
#im.set_alpha(1)
#cb = fig.colorbar(im, cax=cax)
#cb.set_label(label='Density', fontsize=12, rotation=270, labelpad=18)
#plt.show()


#below are non-targets, aka non YxxP containing proteins
#antitargets = visibletargets.symmetric_difference(wnsaf.index)
#awnsaf = wnsaf.loc[antitargets]
#aexpressiongroups = (awnsaf > 0).sum(axis=1)
#bot = np.zeros(len(aexpressiongroups.unique()))
#fig, ax = plt.subplots(figsize=(12,9))
#for n, g in enumerate(reversed(sorted(aexpressiongroups.unique()))):
#    ginds = aexpressiongroups[aexpressiongroups == n].index
#    awnsaf.loc[ginds].sum(axis=0).plot.bar(color=colors[n], bottom=bot, width=0.7)
#    bot += awnsaf.loc[ginds].sum(axis=0)
#plt.xticks(rotation=0, fontsize=11)
#plt.show()


#fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(18,12))
#(wnsaf.loc[variabletargets] > 0).sum(axis=0).plot.bar(ax=ax[0,0])
#(wnsaf.loc[variableexpression] > 0).sum(axis=0).plot.bar(ax=ax[1,0])
#(wnsaf.loc[steadyexpression] > 0).sum(axis=0).plot.bar(ax=ax[2,0])
#wnsaf.loc[variabletargets].sum(axis=0).plot.bar(ax=ax[0,1])
#wnsaf.loc[variableexpression].sum(axis=0).plot.bar(ax=ax[1,1])
#wnsaf.loc[steadyexpression].sum(axis=0).plot.bar(ax=ax[2,1])
#ax[0,0].set_title('Number of Identified Proteins', fontsize=15)
#ax[0,1].set_title('Summed re-Normalized NSAF of Identified Proteins', fontsize=15)
#ax[0,0].set_ylabel('Targets with Variable Expression', fontsize=12)
#ax[1,0].set_ylabel('Proteins with Variable Expression', fontsize=12)
#ax[2,0].set_ylabel('Proteins with Steady Expression', fontsize=12)
#plt.show()

variablenontargets = variabletargets.symmetric_difference(variableexpression)
ndf = df.reset_index().set_index('Protein')
ndf.sort_index(inplace=True)

######################################################################
######################################################################
#IMPORTANT
#neither of these two dataframes will give useable values for protein length, proteins giving multiple motifs will have their lengths counted multiple times
proteinmotifcount = ndf.loc[variableexpression].reset_index().groupby('Motif').agg({'Protein': 'count'})
proteomemotifcount = df.groupby('Motif').agg({'Protein': 'count'})
######################################################################
######################################################################
######################################################################
######################################################################

nproteomeproteins = df.loc[:,'Protein'].unique().shape[0]
proteomemotifcount.loc[:,'freq'] = proteomemotifcount.loc[:,'Protein'] / nproteomeproteins

nproteins = len(variableexpression)
proteinmotifcount.loc[:,'freq'] = proteinmotifcount.loc[:,'Protein'] / nproteins

proteinmotifcount.loc[:,'proteome freq'] = proteomemotifcount.loc[proteinmotifcount.index, 'freq']
proteinmotifcount.loc[:,'freq/proteomefreq'] = proteinmotifcount.loc[:,'freq'] / proteinmotifcount.loc[:, 'proteome freq']

######################################################################
######################################################################
#IMPORTANT
#enrichedproteins does not have accurate protein length information, for reasons stated above for proteinmotifcount/proteomemotifcount
enrichedinds = proteinmotifcount.loc[:,'freq'] > proteinmotifcount.loc[:,'proteome freq']
enrichedproteins = proteinmotifcount.loc[enrichedinds].sort_values('freq/proteomefreq')
######################################################################
######################################################################
######################################################################
######################################################################

#A resampling experiment from this proteome, data collections:
#Check this enrichment distribution, to see where the mean typically lies for individual motifs, and for the mean of the group.
#See if the distribution of %Distance from C-Term of YxxP motifs comes out uniform, or with a centered mean at ~50%.

#distribution of motifs from proteins identified less than 8 times, most of which show up at the 20-hour timepoint.
enrichedproteins.loc[:,'freq/proteomefreq'].plot.hist(figsize=(12,9), bins=50)
plt.title('Distribution of Potential Motifs Found in Variably Expressed Proteins over their Frequency in the Proteome', fontsize=15)
plt.show()

#number of steady targets across time
#(motifcount.loc[steadytargets] > 0).sum(axis=0).plot.bar()
#plt.show()

#summed quant of all proteins across time
wnsaf.sum(axis=0).plot.bar()
plt.show()

#distribution of YxxP across the protein, % distance from C-Terminus of all proteins
df.loc['Y..P', '%D from CT'].plot.hist(figsize=(12,9))
plt.title('Distribution of the Percent Distance of YxxP Motifs from their C-Terminus End across the Proteome', fontsize=15)
plt.show()

#same distribution as above, the ones found in this study tend to have a bias towards their YxxP site being at ~the middle of the protein.
targetsites.loc[:, '%D from CT'].plot.hist()
plt.show()

#this persists across steady and variable targets
fig, ax = plt.subplots(figsize=(12,9))
targetsites.loc[steadytargets, '%D from CT'].plot.hist(color='green', alpha=0.5, label='Steady', ax=ax)
targetsites.loc[variabletargets, '%D from CT'].plot.hist(color='purple', alpha=0.5, label='Variable', ax=ax)
plt.title('Distribution of the Percent Distance of YxxP Motifs from their C-Terminus End in Identified Proteins', fontsize=15)
plt.legend(title='Expression', fontsize=12)
plt.show()

#number of variable targets per technical replicate at 20h
npiv.loc[variabletargets, ('NSAF', 20)].notnull().sum(axis=0).plot.bar()
plt.show()

targframe = targetsites.loc[variabletargets]
targframe.loc[:,'nSAF'] = wtnsaf.loc[targframe.index, 20].tolist()
#distance to C vs %D to C, with color being quantity at 20hr
fig, ax = plt.subplots(figsize=(12,9))
targframe.sort_values('nSAF').plot.scatter(x='%D from CT', y='Distance from C-Term', c='nSAF', ax=ax, cmap='winter_r', norm=matplotlib.colors.LogNorm())
plt.yscale('log')
plt.show()

fig, ax = plt.subplots(figsize=(12,9))
targframe.sort_values('nSAF').plot.scatter(x='%D from CT', y='Distance from C-Term', c='nSAF', ax=ax, cmap='winter_r')
plt.yscale('log')
plt.show()

fig, ax = plt.subplots(figsize=(12,9))
targetsites.plot.scatter(x='%D from CT', y='Distance from C-Term', ax=ax)
plt.show()

fig, ax = plt.subplots(figsize=(12,9))
ax.plot(wtnsaf.loc[steadytargets].mean(axis=0), color='green')
ax.errorbar(wtnsaf.loc[steadytargets].mean(axis=0).index, wtnsaf.loc[steadytargets].mean(axis=0).to_numpy(), yerr=wtnsaf.loc[steadytargets].std(axis=0), color='green', alpha=0.5, capsize=4)
ax.plot(wtnsaf.loc[variabletargets].mean(axis=0), color='purple')
ax.errorbar(wtnsaf.loc[variabletargets].mean(axis=0).index, wtnsaf.loc[variabletargets].mean(axis=0).to_numpy(), yerr=wtnsaf.loc[variabletargets].std(axis=0), color='purple', alpha=0.5, capsize=4)
plt.yscale('log')
plt.show()

targetagg.loc[:,'Protein/MotifLength'] = targetagg.loc[:,'Protein Length'] / len(target)
targetagg.loc[:,'M/AA'] = targetagg.loc[:,'Motif'] / targetagg.loc[:,'Protein Length']
targetagg.loc[:,'M/nAA'] = targetagg.loc[:,'Motif'] / targetagg.loc[:,'Protein/MotifLength']
#YxxP Motifs across protein size across the proteome and expression groups
fig, ax = plt.subplots(figsize=(12,9))
targetagg.plot.scatter(x='Protein Length', y='Motif', ax=ax, color='blue', label='Neither', alpha=0.5)
targetagg.loc[steadytargets].plot.scatter(x='Protein Length', y='Motif', ax=ax, color='green', label='Steady', alpha=0.5)
targetagg.loc[variabletargets].plot.scatter(x='Protein Length', y='Motif', ax=ax, color='purple', label='Variable', alpha=0.5)
plt.legend(title='Expression', fontsize=12)
plt.show()


outfolder = '/'.join((outstart.split('/')[:-1]))
files = [i for i in os.listdir(outfolder)]
flen = len(files)
ofc = 1
taggs = {}

def aggfunc(outfolder, files):
    for of in files:
        ofn = '/'.join((outfolder, of))
        df = pd.read_csv(ofn, low_memory=False, keep_default_na=False, na_values=['____'])
        df.set_index('Motif', inplace=True)
        df.sort_index(inplace=True)
        df.loc[:,'%D from CT'] = df.loc[:,'Distance from C-Term'] / df.loc[:,'Protein Length']
        targs = df.index.unique()
        plen = len(targs)
        pn = 1
        for targ in targs:
            taggs[targ] = df.loc[targ].reset_index().groupby('Protein').agg({'Motif':'count', '%D from CT': 'mean', 'Protein Length': 'mean'})
            sys.stdout.write(f'\r{pn}/{plen}, file {ofc}/{flen}              ')
            sys.stdout.flush()
            pn += 1
        ofc += 1
        gc.collect()

countableproteins = targetagg.index
singleexpressers = expressiongroups.loc[expressiongroups == 1].index
single20hr = wnsaf.loc[singleexpressers].loc[wnsaf.loc[singleexpressers,20] > 0, 20].index
s20inds = countableproteins.intersection(single20hr)
samplefreq = (targetagg.loc[s20inds,'Motif'] / (targetagg.loc[s20inds,'Protein Length'] / motiflength)).sum() / nproteins

ndf = df.reset_index().set_index(['Motif', 'Protein'])
ndf = ndf.loc['Y..P']
yxxp20hrinds = ndf.index.intersection(single20hr)

nproteins = len(single20hr)
nsamples = 100000
motiflength = 4
motifcounts = defaultdict(list)
rs = []
fps = []
for n in range(nsamples):
    samples = random.sample(proteins, nproteins)
    sampleinds = countableproteins.intersection(samples)
    freq = (targetagg.loc[sampleinds,'Motif'] / (targetagg.loc[sampleinds,'Protein Length'] / motiflength)).sum()
    fps.append(freq)
    #nmotifs = targetagg.loc[sampleinds, 'Motif'].sum)
    #naas = targetagg.loc[sampleinds, 'Protein Length'].sum()
    #nmps = (targetagg.loc[sampleinds, 'Motif'] / targetagg.loc[sampleinds, 'Protein Length']).sum()
    #rs.append([nmotifs, naas, nmps])
    sys.stdout.write(f'\r{n}/{nsamples}')
    sys.stdout.flush()

fps = np.asarray(fps)
fpns = fps / nproteins
pvalue = (fpns >= samplefreq).sum() / nsamples

fig, ax = plt.subplots(figsize=(12,9))
ax.hist(fpns, bins=200)
ax.vlines(samplefreq, ymin=0, ymax=175, color='red')
plt.title('Resampled Frequency of YxxP Motifs in $\\it{D. melanogaster}$', fontsize=15)
plt.text(0.0093,1250, '$\\dfrac{\\sum{\\dfrac{\# Motifs}{\\left(\\dfrac{Protein\\ Length}{Motif\\ Length}\\right)}}}{\# Proteins}$', fontsize=20)
plt.text(0.009, 1950, f'n={nsamples} \n pval={pvalue}', fontsize=14)
plt.show()


#old things below
#rs = pd.DataFrame(rs)
#rs.columns = ['Motifs', '#AAs', 'Summed Motifs/AAs']
#rs.index.name = 'Trial'
#rs.loc[:,'Motifs/Proteins'] = rs.loc[:,'Motifs'] / nproteins
#rs.loc[:,'Motifs/AAs/Proteins'] = rs.loc[:,'Summed Motifs/AAs'] / nproteins
#
#tms = targetagg.loc[variabletargets, 'Motif'].sum()
#tml = targetagg.loc[variabletargets, 'Protein Length'].sum()
#tpf = tms/nproteins
#tapf = (targetagg.loc[variabletargets, 'Motif'] / targetagg.loc[variabletargets, 'Protein Length']).sum() /nproteins
#
#fig, ax = plt.subplots(ncols=2, figsize=(14,6))
#rs.loc[:,'Motifs/Proteins'].hist(ax=ax[0], bins=100)
#ax[0].set_title('$\\frac{Motifs}{Proteins}$', fontsize=20)
#ax[0].vlines(tpf, ymin=0, ymax=350, color='red')
#rs.loc[:,'Motifs/AAs/Proteins'].hist(ax=ax[1], bins=100)
#ax[1].set_title('$\\frac{\Sigma \\frac{Motifs}{Amino\\ Acids}}{Proteins}$', fontsize=20)
#ax[1].vlines(tapf, ymin=0, ymax=350, color='red')
#plt.suptitle(f'Resampled Distribution of YxxP Motifs n={nsamples}', fontsize=15)
#plt.show()
#
#firstpval = (rs.loc[:,'Motifs/Proteins'] >= tpf).sum() / nsamples
#secondpval = (rs.loc[:,'Motifs/AAs/Proteins'] >= tapf).sum() / nsamples
#print(f'first p-value: {firstpval}')
#print(f'second p-value: {secondpval}')

#fig, ax = plt.subplots(figsize=(12,9))
#for h in wtnsaf.columns:
#    targetsites.loc[wtnsaf.loc[(wtnsaf.loc[:,h] > 0)].index, 'Protein Length'].

#timequant = wtnsaf.loc[(wtnsaf > 0).sum(axis=1) == wtnsaf.shape[1]]
#timequant = wtnsaf.loc[variabletargets]
#tqproteins = timequant.index
#
#za = stats.zscore(timequant, axis=1)
#za = to_time_series_dataset(za)
#n = 9
#
#model = TimeSeriesKMeans(n_clusters=n, metric='dtw', max_iter=100, n_jobs=5, max_iter_barycenter=100)
#model.fit(za)
#out = model.predict(za)
#
#timequant.loc[:,'cluster'] = out
#
#tc = df.loc[timequant.index.tolist()]
#tc.reset_index(inplace=True)
#tc.loc[:,'cluster'] = timequant.loc[tc.loc[:,'Protein'].tolist(), 'cluster'].tolist()
#ptc = tc.pivot_table(index=['Motif', 'Protein'], columns='cluster', values='Starting Position')
#ptc.fillna(0, inplace=True)
#etc = (ptc > 0).sum(axis=0, level=0)
#
#pfw = timequant.drop('cluster', axis=1)
#fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20,14), sharex=True)
#o = 0
#for v in ax:
#    for m in v:
#        inds = out == o
#        try:
#            pfw.loc[inds].transpose().plot.line(legend=False, ax=m, alpha=0.1, fontsize=20)
#            ncount = pfw.loc[inds].shape[0]
#            ntitle = f''.join(('Cl. ', str(o), ', ' f'n={ncount}'))
#            m.set_yscale('log')
#            m.set_xlabel('hours', fontsize=20)
#            m.set_title(ntitle, fontsize=20)
#            o += 1
#        except TypeError:
#            pass
##plt.suptitle('Clustered NSAF Across Development', fontsize=40)
#plt.show()



#obviously, quantifying yxxp motifs across each file -> check whether normalizing to invariable SAF provides a different overall result
#prevalance of finding a yxxp containing protein, and their composition of AA's, averages and whatnot, compare to average AA content for each file and timepoint, are these proteins enriched?
#the clusters don't look amazing because it's made from proteins present in every file, these are likely not changing their expression levels in any crazy way - and are already highly expressed


#~~~~~~~~~~~~~~~~~~~~~~~~~~~

#You should be plotting each time point for a motif, and the entire proteome, and the unique 20h group
#Weighted by quant is fine too

fuse = 0
saveloc = '/home/sfo/data/motifs/drosophila/sk-plots/'
for target in truepos.index:
    if len(target) != fuse:
        gc.collect()
        fuse = len(target)
        sn = ''.join(('/home/sfo/data/motifs/drosophila/by-motif/Fly_Drosophila_melanogaster_full-filtered-part_distance_ps', str(fuse), '.csv'))
        df = pd.read_csv(sn)
        df.loc[:,'%D from CT'] = df.loc[:,'Distance from C-Term'] / df.loc[:,'Protein Length']
        df.set_index('Motif', inplace=True)
        df.sort_index(inplace=True)
    targetsites = df.loc[target]
    targetsites.set_index('Protein', inplace=True)
    targetproteins = set(df.loc[target, 'Protein'])
    visibletargets = wnsaf.index.intersection(targetproteins)
    figsaver = ''.join((saveloc, target.replace('.', 'x'), '.png'))

