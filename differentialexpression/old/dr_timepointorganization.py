import pandas as pd
import os
from collections import Counter, defaultdict
from multiprocessing.managers import BaseManager, DictProxy
import multiprocessing as mp
from Bio import SeqIO
import pickle
import itertools
import concurrent
import sys
import re
import matplotlib.pyplot as plt
#from dtaidistance import dtw
sys.path.append('/home/sfo/camp')
from sequencetools import seqsplit_dict
#from tslearn.clustering import TimeSeriesKMeans, KShape
#from tslearn.utils import to_time_series_dataset
#from tslearn import preprocessing as pp
from scipy import stats
import numpy as np
pd.options.display.max_rows = 3000
pd.options.display.max_columns = 3000

folder = '/store/drosophila/PXD005713/crux-output'

#proteome = '/home/sfo/data/fastas/proteomes/Fly_Drosophila_melanogaster_Isoforms.fasta'
proteome = '/home/sfo/data/fastas/proteomes/Fly_Drosophila_melanogaster.fasta'

pmin = 2
pmax = 50
patternfile = '/home/sfo/data/motifs/Fly_Drosophila_melanogaster_full-filtered_distance_ps2_50.csv'

#meansabove = 2
#unique = True
#unique = False
##patternfile = '/store/drosophila/PXD005713/full4.patterns.pickle'
#patternfolder = '/store/drosophila/PXD005713/'

if unique:
    patternstring = 'unique'
else:
    patternstring = 'full'
##patternfile = ''.join((patternfolder, patternstring, str(meansabove), '.patterns.pickle'))
#patternfile = ''.join((patternfolder, patternstring, str(pmin), '_', str(pmax), '.patterns.pickle'))

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
    nsname = ''.join((folder, '/', patternstring, '.', f, '.spectral-counts.txt'))
    nfdf = pd.read_csv(nsname, delimiter='\t')
    nfdf.loc[:,'hour'] = timepoint
    nfdf.loc[:,'replicate'] = replicate
    nsafs.append(nfdf)

nsafs = pd.concat(nsafs)
npiv = nsafs.pivot_table(index='ProteinId', columns=['hour', 'replicate'], values=['NSAF', 'SAF', 'rank'])

proteins = nfdf.loc[:,'ProteinId'].unique().tolist()

npw = npiv.loc[:,'NSAF'].count(axis=0)

fcounts = (npiv > 0).sum(axis=1) == npiv.shape[1]
fcvars = npiv.loc[fcounts, 'SAF'].var(axis=1).to_frame()
fcvars.columns = ['vars']
fcvars.loc[:,'means'] = npiv.loc[fcounts, 'SAF'].mean(axis=1)
fcvars.loc[:,'comb'] = (1 / fcvars.loc[:,'vars']) * fcvars.loc[:,'means']
fcvars.sort_values('comb', inplace=True)
#normalizer = fcvars.index[-1]

#normalized across file to the most consistent protein, more of a SAF now, this was like a de-normalization, but call it a re-normalization.
#npsafs = npiv.loc[:,'NSAF'] / npiv.loc[normalizer, 'NSAF']
#npsafs.fillna(0, inplace=True)
npiv.fillna(0, inplace=True)

#mean across time points for each protein, weighted by coverage(as count) of the file. Each nan is replaced with 0 beforehand, these values automatically will lose thier weighting. Get a standard deviation as well, non weighted? This could show good reproducibility.
#wnsaf = (npw * npsafs).sum(axis=1, level=0) / npw.sum(axis=0, level=0)
wnsaf = (npw * npiv.loc[:,'NSAF']).sum(axis=1, level=0) / npw.sum(axis=0, level=0)
wnsaf.sort_index(inplace=True)
#wnsaf.drop(normalizer, inplace=True) #score of normalizer will be nans

fw = wnsaf.copy()
#fw = wnsaf.loc[(wnsaf > 0).sum(axis=1) == wnsaf.shape[1]]
newproteins = fw.index

ta = stats.zscore(fw, axis=1)
#z-normalizing to focus on the shape rather than the absolute differences, according to https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html#dtw-between-multiple-time-series
#ta = stats.zscore(np.log(fw), axis=1)

#ds = dtw.distance_matrix_fast(ta)
#fig, ax = plt.subplots(figsize=(20,10))
#ax.imshow(di)
#plt.show()

newseqs = {i:seqs[i] for i in newproteins}
nlist = [i for i in newseqs.values()]

patternspace = np.linspace(pmin, pmax, pmax-pmin+1).astype(int)
per = '.'

#if not os.path.isfile(patternfile):
#    print(f'Making {patternfile}')
#    patterns = []
#    for p in patternspace:
#        print(p)
#        func = lambda s: [s[i:i+p] for i in range(len(s)) if len(s[i:i+p]) == p]
#        splits = list(map(func, nlist))
#        splits = list(itertools.chain(*splits))
#        splitcount = Counter(splits)
#        mv = np.asarray(list(splitcount.values()))
#        
#        t = Counter(mv)
#        v = np.asarray([l*i for l, i in t.items()])
#        ti = np.asarray([i for i in t.keys()])
#        v = v[ti.argsort()]
#        ti.sort()
#        cv = np.cumsum(v[::-1])[::-1]
#        cd = np.diff(v) / np.diff(cv)
#        try:
#            ci = np.diff(cd).argmin() + 1
#        except ValueError:
#            ci = 1 #There's only two things in t
#        commonint = (mv >= ti[ci]).sum()
#        
#        #for n in range(meansabove):
#        #    ma = np.mean(mv)
#        #    nm = mv > ma
#        #    mv = mv[nm]
#        #commonint = (np.asarray(list(splitcount.values())) > ma).sum()
#        endmin = splitcount.most_common(commonint)[-1][1]
#        endmax = splitcount.most_common(commonint)[0][1]
#        matchend = [i[0] for i in splitcount.most_common(commonint)]
#        #patterns.extend(splitcount)
#        print('match', commonint, endmin, endmax)
#        
#        splits = [f'{(p-2)*per}'.join((i[0], i[-1])) for i in splits]
#        splitcount = Counter(splits)
#        mv = np.asarray(list(splitcount.values()))
#
#        t = Counter(mv)
#        v = np.asarray([l*i for l, i in t.items()])
#        ti = np.asarray([i for i in t.keys()])
#        v = v[ti.argsort()]
#        ti.sort()
#        cv = np.cumsum(v[::-1])[::-1]
#        cd = np.diff(v) / np.diff(cv)
#        try:
#            ci = np.diff(cd).argmin() + 1
#        except ValueError:
#            ci = 1 #There's only two things in t
#        commonint = (mv >= ti[ci]).sum()
#        
#        #for n in range(meansabove):
#        #    ma = np.mean(mv)
#        #    nm = mv > ma
#        #    mv = mv[nm]
#        #commonint = (np.asarray(list(splitcount.values())) > ma).sum()
#        endmin = splitcount.most_common(commonint)[-1][1]
#        endmax = splitcount.most_common(commonint)[0][1]
#        fillerend = [i[0] for i in splitcount.most_common(commonint)]
#        fillerend.extend(matchend)
#        
#        patterns.extend(set(fillerend))
#        print('filler', commonint, endmin, endmax) 
#        print('~')
#    
#    with open(patternfile, "wb") as pick:
#        pickle.dump(patterns, pick)
#else:
#    with open(patternfile, "rb") as pick:
#        patterns = pickle.load(pick)
#        print('loaded', patternfile)
with open(patternfile, "rb") as pick:
    patterns = pickle.load(pick)
    print('loaded', patternfile)

def writer(outies):
    df = pd.DataFrame(outies)
    df.to_csv(outfile, header=False, index=False, mode='a')

#if not os.path.isfile(outfile):
#    startframe = pd.DataFrame(columns=['Protein', 'Motif', 'Starting Position', 'Distance from C-Term', 'Protein Length'])
#    startframe.to_csv(outfile, index=False)
#    pn = 0
#    with concurrent.futures.ThreadPoolExecutor(8) as executor:
#        for patternstring in patterns:
#            outies = []
#            sys.stdout.write(f'\r{pn}/{len(patterns)}')
#            sys.stdout.flush()
#            for k, p in newseqs.items():
#                search = re.compile(patternstring)
#                matches = search.finditer(p)
#                for m in matches:
#                    outlist = [k, patternstring, m.start(), m.endpos - m.start(), m.endpos]
#                    outies.append(outlist)
#            pn += 1
#            executor.submit(writer,outies)
#else:
#    print(f'loading {outfile}')
maxsizeofinterest = 5
chunksize = 1000000
df = []
for d in pd.read_csv(outfile, low_memory=False, keep_default_na=False, na_values=['____'], chunksize=chunksize):
    d.loc[:,'motif length'] = d.loc[:,'Motif'].apply(lambda x: len(x))
    d = d.loc[d.loc[:,'motif length'] <= maxsizeofinterest]
    df.append(d)
df = pd.concat(df)
df.set_index('Protein', inplace=True)
df.sort_index(inplace=True)

#yptest = df.reset_index().set_index('Motif').loc['Y..P']
#testproteins = yptest.loc[:,'Protein'].unique()
#yptest.loc[:,'%D from Ct'] = yptest.loc[:,'Distance from C-Term'] / yptest.loc[:,'Protein Length']
##yptest.loc[:,'Distance from C-Term'].plot.hist(bins=100) #looks neat, but normalizing by protein length shows uniform distribution of YxxP

fw = wnsaf.loc[testproteins]
za = stats.zscore(fw, axis=1)
ta = fw.to_numpy()
ta = to_time_series_dataset(ta)
za = to_time_series_dataset(za)

prepr = pp.TimeSeriesScalerMeanVariance()
prepr.fit(ta)

datas = {'za': za, 'ta': ta}
el = []
mets = ['euclidean', 'dtw', 'softdtw']
nr = 9
for na, da in datas.items():
    for met in mets:
        for n in range(nr):
            n += 2
            model = TimeSeriesKMeans(n_clusters=n, metric=met, max_iter=100, n_jobs=5, max_iter_barycenter=100)
#model = KShape(n_clusters=n, max_iter=100)
            model.fit(da)
            out = model.predict(da)
            
            fw.loc[:,'cluster'] = out
            fw.sort_index(inplace=True)

            tc = df.loc[fw.index.tolist()]
            tc.reset_index(inplace=True)
            tc.loc[:,'cluster'] = fw.loc[tc.loc[:,'Protein'].tolist(), 'cluster'].tolist()
            ptc = tc.pivot_table(index=['Motif', 'Protein'], columns='cluster', values='Starting Position')
            ptc.fillna(0, inplace=True)
            etc = (ptc > 0).sum(axis=0, level=0)

#etcpicks = etc.loc[(etc == 0).sum(axis=1) == n//4]
            #etcpicks = etc.loc[(etc.max(axis=1) / etc.sum(axis=1)) > 0.5]
            ol = [na, n, met, etc.loc['Y..P'].max() / etc.loc['Y..P'].sum()]
            print(ol)
            el.append(ol)
el = pd.DataFrame(el)
el.columns = ['data', 'n', 'method', '%']
#PLOT THE MEANS OF THE CLUSTERS ON A SINGLE GRAPH

edist = etc.sum(axis=0) / etc.sum(axis=0).sum()
print('abundance of clusters')
print(edist)
print(etc.sum(axis=0))

etcpicks = etc.loc[:,edist.sort_values().index]
etcpicks.loc[:,'maxes'] = etcpicks.max(axis=1)
etcpicks.sort_values('maxes', inplace=True)

#dropn = 2
#dval = edist.sort_values().iloc[-1*dropn:].index.tolist()
#ed = etcpicks.drop(dval, axis=1)
#ed = ed.drop('maxes', axis=1)
#ed.loc[:,'maxes'] = ed.max(axis=1)
#ed.sort_values('maxes', inplace=True)
motifs = etcpicks.index.tolist()

pfw = fw.drop('cluster', axis=1)
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20,14), sharex=True)
o = 0
for v in ax:
    for m in v:
        inds = out == o
        try:
            pfw.loc[inds].transpose().plot.line(legend=False, ax=m, alpha=0.1, fontsize=20)
            ncount = pfw.loc[inds].shape[0]
            ntitle = f''.join(('Cl. ', str(o), ', ' f'n={ncount}'))
            m.set_yscale('log')
            m.set_xlabel('hours', fontsize=20, rotation=45)
            m.set_title(ntitle, fontsize=20)
            o += 1
        except TypeError:
            pass
#plt.suptitle('Clustered NSAF Across Development', fontsize=40)
plt.show()

#fig, ax = plt.subplots(figsize=(20,10))
#for o in np.unique(out):
#    inds = out == o
#    dfl = ds[inds].flatten()
#    dfl = dfl[~np.isinf(dfl)]
#    ax.hist(dfl, bins=100, alpha=0.3, label=o)
#plt.legend(title='Cluster')
#plt.title('Distribution of DTW Distance')
#plt.show()

#motifs picked by looking at:
#etcpicks.sort_values(CLUSTERNUMBER).tail(somevalue)
#motifs seen mostly in single clusters were deemed legit
#when you see a group of like, 32 hits all in a row, or something, are they all the same groups of proteins?
#two most numerable clusters
tc.sort_index(inplace=True)

tc.set_index('Motif', inplace=True)
tc.sort_index(inplace=True)

tc.loc[:,'% Distance from C-Term'] = tc.loc[:,'Distance from C-Term'] / tc.loc[:,'Protein Length']

aggfunc = {'Protein': 'count', 'Starting Position': 'mean', 'Distance from C-Term': 'mean', 'Protein Length': 'mean', 'cluster': lambda x: x.nunique(), '% Distance from C-Term': 'mean'}
mc = tc.loc[motifs]

mcmeans = mc.groupby('Motif').agg(aggfunc)

stdaggfunc = {'Protein': 'count', 'Starting Position': 'std', 'Distance from C-Term': 'std', 'Protein Length': 'std', 'cluster': lambda x: x.mode()[0], '% Distance from C-Term': 'std'}

mcstds = mc.groupby('Motif').agg(stdaggfunc)


pfw = fw.drop('cluster', axis=1)
fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(20,14), sharex=True, sharey=True)
o = 0
for v in ax:
    for m in v:
        inds = out == o
        tfc = mc.loc[mc.loc[:,'cluster'] == o]
        pl = tfc.groupby('Protein').agg({'cluster': lambda x: x.mode()[0]}).index.tolist()
        try:
            pfw.loc[inds].transpose().plot.line(legend=False, ax=m, alpha=0.3)
            tpfw = pfw.loc[pl]
            tpfw.transpose().plot.line(legend=False, ax=m, alpha=1, color='red')
            ncount = pfw.loc[inds].shape[0]
            ntitle = f''.join(('Cl. ', str(o), ', ' f'n={ncount}'))
            m.set_yscale('log')
            m.set_title(ntitle)
        except TypeError:
            pass
        o += 1
plt.suptitle('Clustered Normalized Spectral Abundance Frequency')
plt.show()


#pick motifs from clusters
#add motifs to a second cluster plot in bright red on top of the rest
#figure out what proteins have the motifs
#get GO terms for these proteins

#quantify the total motifs, instead of the protein? Cluster this quantity across time
#Add in the Distance-spaced motifs from aminopairs

#asking the group for aims

mc.loc[:,'name'] = mc.loc[:,'Protein'].apply(lambda x: x.split('|')[1])
goterms = '/store/GO/Zebrafish_Danio_rerio.tsv'
gofile = pd.read_csv(goterms, delimiter='\t')

matches = [i for i in mc.loc[:,'name'].unique().tolist() if i in gofile.loc[:,'GENE PRODUCT ID'].tolist()]

nmc = mc.reset_index().set_index('name')
nmc = nmc.loc[matches]
nmc.sort_index(inplace=True)

gofile.set_index('GENE PRODUCT ID', inplace=True)
gofile.sort_index(inplace=True)

gt = gofile.loc[nmc.index, 'GO NAME']
gt.sort_index(inplace=True)

nmode = nmc.groupby('name').agg({'cluster': lambda x: x.mode()[0]})

gt = gt.to_frame()
gt.loc[:,'cluster'] = nmode.loc[gt.index, 'cluster']

ng = gt.reset_index().set_index(['GO NAME', 'cluster'])
ng.sort_index(inplace=True)

cg = ng.count(axis=1)
cg = cg.to_frame()

goframe = cg.pivot_table(index='cluster', columns='GO NAME', aggfunc='count')
goframe.fillna(0.01, inplace=True)
goframe = goframe.droplevel(axis=1, level=0)

gofilter =  10

goplot = goframe.loc[:,goframe.max() > gofilter]

goplot.sort_values(7, axis=1, inplace=True)
#sort this by the sum of the columns, so the columns with less sum, ie hopefully the columns with more specific hits, will go in a better order
#that might also work well above for finding motifs
#OR pull similarity from the peak finding process, rows with a large difference in sum to maximum might give what you want. Potentially sort by this?
#filter protein results by non-isoform database
#make some fake timepoints based on the drosophila timepoints and see if you can cluster things nicely.
#look at non-isoform only data as well, to not bias the results?

fig, ax = plt.subplots(figsize=(30,10))

im = ax.matshow(goplot.to_numpy(), norm=matplotlib.colors.LogNorm(), aspect='auto')

ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
xlabels = goplot.columns.tolist()
ax.set_xticks(np.arange(len(xlabels)))
ax.set_xticklabels(xlabels, fontsize=15)

plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")

ax.set_yticks(goplot.index.tolist())

plt.yticks(fontsize=20)
plt.ylabel('Cluster', fontsize=20)

plt.title('Gene Ontology Term Enrichment', fontsize=30)
cb = plt.colorbar(im)
cb.set_label(label='Log-Scale Enrichment', fontsize=15)
plt.show()

