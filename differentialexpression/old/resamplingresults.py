import pandas as pd
from time import time
import os
from collections import Counter, defaultdict
from Bio import SeqIO
import pickle
import multiprocessing as mp
import pandarallel
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

folder = '/store/drosophila/PXD005713/crux-output'

#proteome = '/home/sfo/data/fastas/proteomes/Fly_Drosophila_melanogaster_Isoforms.fasta'
proteome = '/home/sfo/data/fastas/proteomes/Fly_Drosophila_melanogaster.fasta'

pmin = 2
pmax = 50
outstart = '/home/sfo/data/motifs/drosophila/resampling/'
outgeneration = True
outgeneration = False

motifgen = True
motifgen = False

#meansabove = 2
#unique = True
#unique = False
##patternfile = '/store/drosophila/PXD005713/full4.patterns.pickle'
#patternfolder = '/store/drosophila/PXD005713/'
#
#if unique:
#    patternstring = 'unique'
#else:
#    patternstring = 'full'
#
#patternfile = ''.join((patternfolder, patternstring, str(pmin), '_', str(pmax), '.patterns.pickle'))
patternfile = '/store/drosophila/PXD005713/full4.patterns.pickle'
motiffile = ''.join((patternfolder, patternstring, str(pmin), '_', str(pmax), '.motifs.pickle'))

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
    
    nsname = ''.join((folder, '/', patternstring, '.', f, '.spectral-counts.txt'))
    nfdf = pd.read_csv(nsname, delimiter='\t')
    nfdf.loc[:,'hour'] = timepoint
    nfdf.loc[:,'replicate'] = replicate
    nsafs.append(nfdf)

nsafs = pd.concat(nsafs)
npiv = nsafs.pivot_table(index='ProteinId', columns=['hour', 'replicate'], values=['NSAF', 'SAF', 'rank'])

proteins = seqs.keys()
sequencelist = seqs.values()

patternspace = np.linspace(pmin, pmax, pmax-pmin+1).astype(int)
per = '.'

if not os.path.isfile(patternfile):
    print(f'Making {patternfile}')
    patterns = []
    for p in patternspace:
        print(p)
        func = lambda s: [s[i:i+p] for i in range(len(s)) if len(s[i:i+p]) == p]
        splits = list(map(func, nlist))
        splits = list(itertools.chain(*splits))
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
        print('match', commonint, endmin, endmax)
        
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
        fillerend.extend(matchend)
        
        patterns.extend(set(fillerend))
        print('filler', commonint, endmin, endmax) 
        print('~')
    
    with open(patternfile, "wb") as pick:
        pickle.dump(patterns, pick)
else:
    with open(patternfile, "rb") as pick:
        patterns = pickle.load(pick)
        print('loaded', patternfile)

patterns = [i for i in patterns if len(i) <= 10]

fpiv = npiv.loc[:,'SAF']
fcounts = (fpiv > 0).sum(axis=1) == fpiv.shape[1]
fcvars = fpiv.loc[fcounts].var(axis=1).to_frame()
fcvars.columns = ['vars']
fcvars.loc[:,'means'] = fpiv.loc[fcounts].mean(axis=1)
fcvars.loc[:,'comb'] = (1 / fcvars.loc[:,'vars']) * fcvars.loc[:,'means']
fcvars.sort_values('comb', inplace=True)
normalizer = fcvars.index[-1]

nnpiv = npiv.loc[:,'NSAF'] / npiv.loc[normalizer, 'NSAF']
nnpiv.fillna(0, inplace=True)

npw = npiv.loc[:,'NSAF'].count(axis=0)

wnsaf = (npw * nnpiv).sum(axis=1, level=0) / npw.sum(axis=0, level=0)
wnsaf = wnsaf.loc[(wnsaf > 0).sum(axis=1) > 0]
wnsaf.sort_index(inplace=True)

expressiongroups = (wnsaf > 0).sum(axis=1)

singleexpressers = expressiongroups.loc[expressiongroups == 1].index
single20hr = wnsaf.loc[singleexpressers].loc[wnsaf.loc[singleexpressers,20] > 0, 20].index

nproteins = len(single20hr)

files = [i for i in os.listdir(outstart)]

fn = '/home/sfo/data/motifs/drosophila/compiledsamples/rs10000.csv'
#flen = len(files)
#startframe = pd.DataFrame(index=patterns)
#startframe = startframe.transpose()
#startframe.to_csv(fn)
#for n, f in enumerate(files):
#    dn = int(''.join(([i for i in f if i.isdigit()])))
#    tfn = ''.join((outstart, f))
#    tf = pd.read_csv(tfn, keep_default_na=False, na_values=['____'])
#    tf.columns = ['motif', dn+1]
#    tf.set_index('motif', inplace=True)
#    tf = tf.transpose()
#    tf.to_csv(fn, header=False, mode='a')
#    sys.stdout.write(f'\r{n+1}/{flen}')
#    sys.stdout.flush()

df = pd.read_csv(fn, low_memory=False, keep_default_na=False, na_values=['____'])
df.drop('Unnamed: 0', axis=1, inplace=True)

newseqs = {k:v for k, v in seqs.items() if k in single20hr}

mt = time()
pmotifs = defaultdict(lambda: defaultdict(int))
pn = 0
plen = len(patterns)
for patternstring in patterns:
    search = re.compile(patternstring)
    for k, p in newseqs.items():
        matches = search.finditer(p)
        for m in matches:
            pmotifs[k][patternstring] += 1
    pn += 1
    sys.stdout.write(f'\r{pn}/{plen}')
    sys.stdout.flush()
print(f'generated motifs in {time() - mt}')

ef = pd.DataFrame(index=patterns)
ef.loc[:,'freq'] = 0
for s in single20hr:
    td = pd.DataFrame.from_dict(pmotifs[s], orient='index', columns=[s])
    td.loc[:,s] = td.loc[:,s] / (len(seqs[s]) / td.index.str.len())
    ef.loc[td.index, 'freq'] += td.loc[:,s]

ef = ef / nproteins
df = df / nproteins

ef.loc[:,'p-value'] = (df >= ef.transpose().to_numpy()).sum(axis=0) / df.shape[1]
ef = ef.loc[ef.loc[:,'freq'] > 0]
ef.sort_values('p-value', inplace=True)

alphaval = 0.01
qvals = multitest.multipletests(ef.loc[:,'p-value'].to_numpy(), alpha=alphaval, method='fdr_bh')
ef.loc[:,'q-value'] = qvals[1]

truepos = ef.loc[qvals[0]]
truepos.loc[:,'length'] = truepos.index.str.len()
truepos.sort_values('length', inplace=True)

#went with Benjamini/Hotchberg based on the distribution of p-values, it looks like an increase in true positives could lead to an increase in identified false positives next to it. A negative correlation test.
#^Because of the baseline noise between the true pos/true neg distributions.
ef.loc[:,'p-value'].plot.hist(bins=200)
plt.show()

#fuse = 0
#saveloc = '/home/sfo/data/motifs/drosophila/plots/'
#for target in truepos.index:
#    if len(target) != fuse:
#        gc.collect()
#        fuse = len(target)
#        sn = ''.join(('/home/sfo/data/motifs/drosophila/by-motif/Fly_Drosophila_melanogaster_full-filtered-part_distance_ps', str(fuse), '.csv'))
#        df = pd.read_csv(sn)
#        df.loc[:,'%D from CT'] = df.loc[:,'Distance from C-Term'] / df.loc[:,'Protein Length']
#        df.set_index('Motif', inplace=True)
#        df.sort_index(inplace=True)
#    targetsites = df.loc[target]
#    targetsites.set_index('Protein', inplace=True)
#    targetproteins = set(df.loc[target, 'Protein'])
#    visibletargets = wnsaf.index.intersection(targetproteins)
#    figsaver = ''.join((saveloc, target.replace('.', 'x'), '.png'))
#
#    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(9,16))
#    for n, c in enumerate(sorted(wnsaf.columns)):
#        tinds = wnsaf.loc[wnsaf.loc[:,c] > 0].index.intersection(visibletargets)
#        yvals = targetsites.loc[tinds, '%D from CT'].to_numpy()
#        yv = stats.gaussian_kde(yvals)(yvals)
#        xvals = np.repeat(n, len(yvals))
#        pf = pd.DataFrame(columns=['x', 'y', 'yd'])
#        pf.loc[:,'x'] = xvals
#        pf.loc[:,'y'] = yvals
#        pf.loc[:,'yd'] = yv
#        pf.sort_values('yd').plot.scatter(x='x', y='y', c='yd', colormap='cool', ax=ax1, alpha=0.1, colorbar=False, marker='s', s=470)
#    ax1.tick_params(rotation=0, size=11)
#    ax1.set_title(f'Kernel Density Estimation of Percent Distance of {target} \n to C-Terminus by Timepoint', fontsize=15)
#    ax1.set_ylabel('% Distance to C-Terminus', fontsize=16)
#    ax1.set_xlabel('Hour', fontsize=16)
#    xtvs = np.arange(len(wnsaf.columns))
#    ax1.set_xticks(xtvs)
#    ax1.set_xticklabels(wnsaf.columns)
#    #im = plt.gca().get_children()[0]
#    #fig.subplots_adjust(right=0.8)
#    #cax = fig.add_axes([0.85,0.15,0.03,0.7])
#    #im.set_alpha(1)
#    #cb = fig.colorbar(im, cax=cax)
#    #cb.set_label(label='Density', fontsize=18, rotation=270, labelpad=18)
#    #for label in ax.get_xticklabels():
#    #    label.set_fontsize(15)
#    #for label in ax.get_yticklabels():
#    #    label.set_fontsize(15)
#    #cb.ax.tick_params(labelsize=15)
#
#
##fig, ax = plt.subplots(figsize=(12,9))
#    for n, c in enumerate(sorted(wnsaf.columns)):
#        tinds = wnsaf.loc[wnsaf.loc[:,c] > 0].index.intersection(visibletargets)
#        yvals = targetsites.loc[tinds, '%D from CT'].to_numpy()
#        yv = stats.gaussian_kde(yvals, weights=wnsaf.loc[targetsites.loc[visibletargets].loc[tinds].index, c].to_numpy())(yvals)
#        xvals = np.repeat(n, len(yvals))
#        pf = pd.DataFrame(columns=['x', 'y', 'yd'])
#        pf.loc[:,'x'] = xvals
#        pf.loc[:,'y'] = yvals
#        pf.loc[:,'yd'] = yv
#        pf.sort_values('yd').plot.scatter(x='x', y='y', c='yd', colormap='cool', ax=ax2, alpha=0.1, colorbar=False, marker='s', s=470)
#    ax2.tick_params(rotation=0, size=11)
#    ax2.set_title(f'Quantification-Weighted Kernel Density Estimation of Percent \n Distance of {target} to C-Terminus by Timepoint', fontsize=15)
#    ax2.set_ylabel('% Distance to C-Terminus', fontsize=18)
#    ax2.set_xlabel('Hour', fontsize=18)
#    xtvs = np.arange(len(wnsaf.columns))
#    ax2.set_xticks(xtvs)
#    ax2.set_xticklabels(wnsaf.columns)
#    #im = plt.gca().get_children()[0]
#    #fig.subplots_adjust(right=0.8)
#    #cax = fig.add_axes([0.85,0.15,0.03,0.7])
#    #im.set_alpha(1)
#    #cb = fig.colorbar(im, cax=cax)
#    #cb.set_label(label='Density', fontsize=18, rotation=270, labelpad=18)
#    #for label in ax.get_xticklabels():
#    #    label.set_fontsize(15)
#    #for label in ax.get_yticklabels():
#    #    label.set_fontsize(15)
#    #cb.ax.tick_params(labelsize=15)
#
#    fig.savefig(figsaver)

for ev in expressiongroups.sort_values().unique():
    for ce in wnsaf.columns:
        expressers = expressiongroups.loc[expressiongroups == 1].index
        expressers = wnsaf.loc[expressers].loc[wnsaf.loc[expressers,ce] > 0, ce].index
        npe = len(expressers)
        pnorm = npe / nproteins

        pf = pd.DataFrame(index=patterns)
        pf.loc[:,'freq'] = 0
        for s in expressers:
            td = pd.DataFrame.from_dict(pmotifs[s], orient='index', columns=[s])
            td.loc[:,s] = td.loc[:,s] / (len(seqs[s]) / td.index.str.len())
            pf.loc[td.index, 'freq'] += td.loc[:,s]

        pf = pf / pnorm / nproteins

        pf.loc[:,'p-value'] = (df >= pf.transpose().to_numpy()).sum(axis=0) / df.shape[1]
        pf = pf.loc[pf.loc[:,'freq'] > 0]
        pf.sort_values('p-value', inplace=True)

        alphaval = 0.01
        try:
            eqvals = multitest.multipletests(pf.loc[:,'p-value'].to_numpy(), alpha=alphaval, method='fdr_bh')
            pf.loc[:,'q-value'] = eqvals[1]

            etruepos = pf.loc[eqvals[0]]
            etruepos.loc[:,'length'] = etruepos.index.str.len()
            etruepos.sort_values('length', inplace=True)
            print(ev, ce, etruepos.shape)
        except ZeroDivisionError:
            print(ev, ce, 'nope')
