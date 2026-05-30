import pandas as pd
import os
from collections import Counter, defaultdict
from multiprocessing.managers import BaseManager, DictProxy
import multiprocessing as mp
from Bio import SeqIO
import matplotlib
import itertools
import concurrent
import sys
import re
import matplotlib.pyplot as plt
from dtaidistance import dtw
sys.path.append('/home/sfo/camp')
from sequencetools import seqsplit_dict
from tslearn.clustering import TimeSeriesKMeans, KShape
from scipy import stats
import numpy as np
pd.options.display.max_rows = 3000
pd.options.display.max_columns = 3000

folder = '/store/zebrafish'
folders = [i for i in os.listdir(folder) if i.startswith('PXD')]
proteome = '/home/sfo/data/fastas/proteomes/Zebrafish_Danio_rerio_Isoforms.fasta'
motifs = '/home/sfo/data/motifs/Zebrafish_Danio_rerio_distance15.csv'
trypsin = [{'K':1, 'R':1}, {'K':'P', 'R':'P'}]

fasta = SeqIO.parse(open(proteome), 'fasta')
seqs = {}
descs = {}
for fasta in fasta:
    sequence, idn = str(fasta.seq), fasta.id
    seqs[idn] = sequence
    descs[idn] = fasta.description

cut = seqsplit_dict(seqs, trypsin, 7, 70, 2)
lcut = [i for i in cut.values()]
lcut = list(itertools.chain(*lcut))
ucut = Counter(lcut)
uniquepeptides = [i for i, v in ucut.items() if v == 1]

#class MyManager(BaseManager):
#    pass
#
#def cutfunc(cut, u, proteinswithunique):
#    for p in cut.keys():
#        if u in cut[p]:
#            proteinswithunique[p].append(u)
#    return
#
#MyManager.register('defaultdict', defaultdict, DictProxy)
#mgr = MyManager()
#mgr.start()
#proteinswithunique = mgr.defaultdict(list)
#pool = mp.Pool()
##proteinswithunique = defaultdict(list)
#for u in uniquepeptides:
#    pool.apply_async(cutfunc(cut, u, proteinswithunique))
#pool.close()
#pool.join()
##    for p in cut.keys():
##        if u in cut[p]:
##            proteinswithunique[p].append(u)

ef = []
nsafs = []
for f in folders:

    uf = '/'.join((folder, f))
    files = [i for i in os.listdir(uf)]
    extensions = [i.split('.') for i in files]
    extensions = [i for j in extensions for i in j]
    ex = Counter(extensions)
    splitter = ''.join(('.', ex.most_common(1)[0][0]))
    files = [i.split(splitter)[0] for i in files if splitter in i]
    
    timesplits = False
    if f == 'PXD010922':
        timepoint = 24
    if f == 'PXD011258':
        timepoint = 72
    if f == 'PXD006905':
        timepoint = 0
    if f == 'PXD008322':
        timepoint = 24
    if f == 'PXD005137':
        timepoint = 24
    if f == 'PXD005129':
        timepoint = 24
    if f == 'PXD001164':
        timesplits = True
    if f == 'PXD006098':
        timepoint = 2.375
    if f == 'PXD003455':
        timepoint = 120
    if f == 'PXD013173':
        timepoint = 120
    
    df = pd.DataFrame()
    df.loc[:,'file'] = files
    df.loc[:,'study'] = f
    if timesplits:
        inds = df.loc[:,'file'].str.contains('DOME')
        df.loc[inds, 'hpf'] = 5 #it's 4.3, but check this with helaina
        df.loc[~inds, 'hpf'] = 24
    else:
        df.loc[:,'hpf'] = timepoint

    ef.append(df)
    
    cf = '/'.join((folder, f, 'crux-output'))
    nsfs = [i for i in os.listdir(cf) if i.endswith('.spectral-counts.target.txt') and i.startswith('full')]
    for c in nsfs:
        nfdn = '/'.join((cf, c))
        nfdf = pd.read_csv(nfdn, delimiter='\t')
        if nfdf.size > 0:
            nfdf.loc[:,'file'] = c.split('.')[1]
            nfdf.loc[:,'study'] = f
            timesplits = False
            if f == 'PXD010922':
                timepoint = 24
            if f == 'PXD011258':
                timepoint = 72
            if f == 'PXD006905':
                timepoint = 0
            if f == 'PXD008322':
                timepoint = 24
            if f == 'PXD005137':
                timepoint = 24
            if f == 'PXD005129':
                timepoint = 24
            if f == 'PXD001164':
                timesplits = True
            if f == 'PXD006098':
                timepoint = 2.375
            if f == 'PXD003455':
                timepoint = 120
            if f == 'PXD013173':
                timepoint = 120

            if timesplits:
                inds = nfdf.loc[:,'file'].str.contains('DOME')
                nfdf.loc[inds, 'hpf'] = 5 #it's 4.3, but check this with helaina
                nfdf.loc[~inds, 'hpf'] = 24
            else:
                nfdf.loc[:,'hpf'] = timepoint

            nsafs.append(nfdf)

nsafs = pd.concat(nsafs)
npiv = nsafs.pivot_table(index='protein id', columns=['hpf', 'study', 'file'], values=['NSAF', 'parsimony rank'])

ef = pd.concat(ef)
ef.set_index('hpf', inplace=True)
ef.sort_index(inplace=True)

nf = []
for t in ef.index.unique():
    sf = ef.loc[(t)]
    sf.set_index('study', inplace=True)
    for si in sf.index.unique():
        flist = sf.loc[si, 'file'].tolist()
        for sfl in flist:
            rfile = '/'.join((folder, si, 'crux-output', ''.join((sfl, '.percolator.target.proteins.txt'))))
            rf = pd.read_csv(rfile, delimiter='\t')
            rf = rf.loc[rf.loc[:,'q-value'] <= 0.05]
            if rf.size > 0:
                tf = rf.loc[:,('ProteinId')].to_frame()
                tf.loc[:,'study'] = si
                tf.loc[:,'file'] = sfl
                tf.loc[:,'time'] = t
                nf.append(tf)
nf = pd.concat(nf)
proteins = nf.loc[:,'ProteinId'].unique().tolist()

#pull proteins from the motif file into here now

#mean across time points for each protein, weighted by coverage(as count) of the file. Each nan is replaced with 0 beforehand, these values automatically will lose thier weighting. Get a standard deviation as well, non weighted?

npw = npiv.loc[:,'NSAF'].count(axis=0)
npiv.fillna(0, inplace=True)

wnsaf = (npw * npiv.loc[:,'NSAF']).sum(axis=1, level=0) / npw.sum(axis=0, level=0)
#wnsaf.sum(axis=0) shows some level of reproducibility here

fig, ax = plt.subplots()
(wnsaf > 0).sum(axis=0).plot.bar(width=0.8)
ax.set_xticklabels(wnsaf.columns, rotation=0)
ax.set_xlabel('Hours Post-Fertilization', fontsize=12)
ax.set_ylabel('# Identified Proteins', fontsize=12)
plt.title('Currently Usable Data Across $\\it{D. rerio}$ Development', fontsize=15)
plt.show()

whits = [i for i in wnsaf.index.tolist() if i in proteins]
wnsaf = wnsaf.loc[whits]
#wnsaf = wnsaf.loc[:,[5, 24, 72]]
wnsaf = wnsaf.loc[:,[5, 24, 72, 120]]

fw = wnsaf.loc[(wnsaf > 0).sum(axis=1) == 4]
try:
    fw = fw.drop('tr|E9QCY9|E9QCY9_DANRE')
except KeyError:
    pass
newproteins = fw.index.tolist()

#chunksize = 1000000
#maxsize = 29
#ec = []
#for dc in pd.read_csv(motifs, low_memory=False, chunksize=chunksize):
#    dc.loc[:,'motif length'] = dc.loc[:,'Motif'].apply(lambda x: len(x))
#    dc = dc.loc[dc.loc[:,'motif length'] <= maxsize]
#    pl = dc.loc[:,'Protein'].unique().tolist()
#    il = [i for i in pl if i in newproteins]
#    dc.set_index('Protein', inplace=True)
#    dc.sort_index(inplace=True)
#    if any(il):
#        ec.append(dc.loc[il])
#ec = pd.concat(ec)

ta = stats.zscore(fw, axis=1) #z-normalizing to focus on the shape rather than the absolute differences, according to https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html#dtw-between-multiple-time-series
#ta = stats.zscore(np.log(fw), axis=1)

ds = dtw.distance_matrix_fast(ta)
#fig, ax = plt.subplots(figsize=(20,10))
#ax.imshow(di)
#plt.show()

newseqs = {i:seqs[i] for i in newproteins}
nlist = [i for i in newseqs.values()]


meansabove = 4
pmin = 2
pmax = 50
patternspace = np.linspace(pmin, pmax, pmax-pmin+1).astype(int)

#patterns = []
#for p in patternspace:
#    func = lambda s: [s[i:i+p] for i in range(len(s)) if len(s[i:i+p]) == p]
#    splits = list(map(func, nlist))
#    splits = list(itertools.chain(*splits))
#    splitcount = Counter(splits)
#    mv = np.asarray(list(splitcount.values()))
#    for n in range(meansabove):
#        n += 1
#        ma = np.mean(mv)
#        nm = mv > ma
#        mv = mv[nm]
#    commonint = (np.asarray(list(splitcount.values())) > ma).sum()
#    endmin = splitcount.most_common(commonint)[-1][1]
#    endmax = splitcount.most_common(commonint)[0][1]
#    splitcount = [i[0] for i in splitcount.most_common(commonint)]
#    patterns.extend(splitcount)
#    print(p, commonint, endmin, endmax)

outfile = '/home/sfo/data/motifs/Zebrafish_Danio_rerio_filtered_distance50.csv'
#out = pd.DataFrame(columns = ['Protein', 'Motif', 'Starting Position', 'Distance from C-Term', 'Protein Length'])
#out.to_csv(outfile, index=False)
#outies = []
#pn = 0
#for patternstring in patterns:
#    sys.stdout.write(f'\r{pn}/{len(patterns)}')
#    sys.stdout.flush()
#    for k, p in newseqs.items():
#        search = re.compile(patternstring)
#        matches = search.finditer(p)
#        for m in matches:
#            outlist = [k, patternstring, m.start(), m.endpos - m.start(), m.endpos]
#            outies.append(outlist)
#    pn += 1
#df = pd.DataFrame(outies)
#df.columns = ['Protein', 'Motif', 'Starting Position', 'Distance from C-Term', 'Protein Length']
#df.to_csv(outfile, index=False)

df = pd.read_csv(outfile)
df.set_index('Protein', inplace=True)
df.sort_index(inplace=True)

#outies = []
#pn = 0
#def finder(patternstring, pn):
#    for k, p in newseqs.items():
#        search = re.compile(patternstring)
#        matches = search.finditer(p)
#        for m in matches:
#            outlist = [k, patternstring, m.start(), m.endpos - m.start(), m.endpos]
#            outies.append(outlist)
#    return
#
#with concurrent.futures.ThreadPoolExecutor(2) as executor:
#    for patternstring in patterns:
#        executor.submit(finder, patternstring, pn)
#        sys.stdout.write(f'\r{pn}/{len(patterns)}')
#        sys.stdout.flush()
#        pn += 1

n = 9
model = TimeSeriesKMeans(n_clusters=n, metric="euclidean", max_iter=100, max_iter_barycenter=100, random_state=2)
#model = KShape(n_clusters=n, max_iter=100)
model.fit(ta)
out = model.predict(ta)

fw.loc[:,'cluster'] = out
fw.sort_index(inplace=True)

tc = df.loc[fw.index.tolist()]
tc.reset_index(inplace=True)
tc.loc[:,'cluster'] = fw.loc[tc.loc[:,'Protein'].tolist(), 'cluster'].tolist()
ptc = tc.pivot_table(index=['Motif', 'Protein'], columns='cluster', values='Starting Position')
ptc.fillna(0, inplace=True)
etc = (ptc > 0).sum(axis=0, level=0)

etcpicks = etc.loc[(etc == 0).sum(axis=1) == n//2]

edist = etc.sum(axis=0) / etc.sum(axis=0).sum()
print('abundance of clusters')
print(edist)
print(etc.sum(axis=0))

etcpicks = etcpicks.loc[:,edist.sort_values().index]
etcpicks.loc[:,'maxes'] = etcpicks.max(axis=1)
etcpicks.sort_values('maxes', inplace=True)

dropn = 2
dval = edist.sort_values().iloc[-1*dropn:].index.tolist()
ed = etcpicks.drop(dval, axis=1)
ed = ed.drop('maxes', axis=1)
ed.loc[:,'maxes'] = ed.max(axis=1)
ed.sort_values('maxes', inplace=True)

print(etcpicks.index.str.contains('FSDP').sum())

pfw = fw.drop('cluster', axis=1)
fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(20,14), sharex=True, sharey=True)
o = 0
for v in ax:
    for m in v:
        inds = out == o
        try:
            pfw.loc[inds].transpose().plot.line(legend=False, ax=m, alpha=0.3, fontsize=20)
            ncount = pfw.loc[inds].shape[0]
            ntitle = f''.join(('Cl. ', str(o), ', ' f'n={ncount}'))
            m.set_yscale('log')
            m.set_xticks([5, 24, 72])
            m.set_xlabel('hours post-fertilization', fontsize=20)
            m.set_title(ntitle, fontsize=20)
            o += 1
        except TypeError:
            pass
plt.suptitle('Clustered NSAF Across Development', fontsize=40)
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

#Need to automatically generate motifs here
#motifs picked by looking at:
#etcpicks.sort_values(CLUSTERNUMBER).tail(somevalue)
#motifs seen mostly in single clusters were deemed legit
#when you see a group of like, 32 hits all in a row, or something, are they all the same groups of proteins?
motifs = [
        'VFSDP',
        'FSDPC',
        'KLEGD',
        'QLRCNGVLEGIRICR',
        'QLRCNGVLEGIRIC',
        'LRCNGVLEGIRICR',
        'GKFIRI',
        'RLEEA',
        'FGKFIR',
        'LQQFF',
        'KGGKK',
        'LDEAE',
        'KPYTC',
        'GGYGG',
        'IHTGEKP',
        'GKSFT',
        'IHTGEK',
        'EESEDIKIEETFTVKHE',
        'EDIKIEETFTVKHE',
        'AFIKEESEDIKIEETFTVKHE',
        'AFIKEESEDIKIEETFTVKH',
        'EESEDIKIEETFTVKH',
        'FIKEESEDIKIEETFT',
        'KEESEDIKIEETFT',
        'FIKEESEDIKIEETF',
        'SEDIKIEETFTVKHE',
        'FIKEESEDIKIEETFTVKH',
        'FIKEESEDIKIEETFTVK',
        'FIKEESEDIKIEETFTV',
        'FIKEESEDIKIEET',
        'EESEDIKIEETFTVK',
        'AFIKEESEDIKIEETFT',
        'AFIKEESEDIKIEE',
        'IKEESEDIKIEETF',
        'IKEESEDIKIEETFT',
        'IKEESEDIKIEETFTV',
        'ESEDIKIEETFTVKHE',
        'ESEDIKIEETFTVK',
        'IKEESEDIKIEETFTVK',
        'EESEDIKIEETFTV',
        'KEESEDIKIEETFTV',
        'IKEESEDIKIEETFTVKH',
        'ESEDIKIEETFTVKH',
        'IKEESEDIKIEETFTVKHE',
        'AFIKEESEDIKIEETFTVK',
        'AFIKEESEDIKIEETF',
        'AFIKEESEDIKIEET',
        'KEESEDIKIEETFTVK',
        'AFIKEESEDIKIEETFTV',
        'MRIHT',
        'KEESEDIKIEETFTVKHE',
        'KEESEDIKIEETFTVKH',
        'SEDIKIEETFTVKH',
        'FIKEESEDIKIEETFTVKHE',
        'EETFTV',
        'IEETFTVK',
        'IEETFT',
        'KIEETFTV',
        'EETFT',
        'EETFTVK',
        'KIEETFT',
        'KIEETFTVK',
        'KIEETF',
        'ETFTVK',
        'IEETFTV',
        'IEETF',
        'ESEDIKIEETFTVKHE',
        'KEESEDIKIEETFTVKH',
        'IKEESEDIKIEETFTVKH',
        'KEESEDIKIEETFTVKHE',
        'FIKEESEDIKIEETFTVKH',
        'IKEESEDIKIEETFTVKHE',
        'SEDIKIEETFTVKHE',
        'SEDIKIEETFTVKH',
        'AFIKEESEDIKIEETFTVKH',
        'FIKEESEDIKIEETFTVKHE',
        'AFIKEESEDIKIEETFTVKHE',
        'EDIKIEETFTVKHE',
        'EESEDIKIEETFTVKH',
        'EESEDIKIEETFTVKHE',
        'RVHTGE',
        'ESEDIKIEETFTVKH',
        'HTGERP',
        'EESEDIKIEETFTV',
        'KEESEDIKIEETFTVK',
        'AFIKEESEDIKIEETFT',
        'VHTGEKP',
        'TGERP',
        'FIKEESEDIKIEET',
        'FIKEESEDIKIEETF',
        'EESEDIKIEETFTVK',
        'FIKEESEDIKIEETFT',
        'AFIKEESEDIKIEETFTV',
        'FIKEESEDIKIEETFTVK',
        'AFIKEESEDIKIEET',
        'VHTGEK',
        'AFIKEESEDIKIEETF',
        'AFIKEESEDIKIEETFTVK',
        'FIKEESEDIKIEETFTV',
        'AFIKEESEDIKIEE',
        'KEESEDIKIEETFT',
        'CGKSFSQ',
        'ESEDIKIEETFTVK',
        'RVHTG',
        'IKEESEDIKIEETF',
        'IKEESEDIKIEETFT',
        'IKEESEDIKIEETFTV',
        'IKEESEDIKIEETFTVK',
        'KEESEDIKIEETFTV',
        'GKSFSQ',
        'MRVHT',
        'HTGER',
        'TCQQCGKSF',
        'TCQQCGKS',
        'TCQQCGK',
        'RTHTGE',
        'TCQQC',
        'TCQQCG',
        'CQQCGKS',
        'QQCGKSF',
        'QQCGKS',
        'CQQCGKSF',
        'HMRTH',
        'QQCGK',
        'CQQCGK',
        'CQQCG',
        'GEKPYTC',
        'GEKPYT',
        'EKPYT',
        'EKPYTC',
        'RIHTGEKPF',
        'KIEETFTV',
        'GKSFT',
        'EETFTV',
        'EETFT',
        'EETFTVK',
        'KIEETFT',
        'IEETFTVK',
        'ETFTVK',
        'IHTGEKPF',
        'IEETFT',
        'KIEETFTVK',
        'IEETFTV',
        'KPYTC',
        'TGEKPY',
        'HTGEKPY',
        'KIEETF',
        'QCGKSFS',
        'IEETF',
        'MRIHT',
        'RIHTGEKP',
        'RIHTGEK',
        'IHTGEKP',
        'IHTGEK',
        'WDTAGQE',
        'WDTAG',
        'WDTAGQ',
        'DTAGQE',
        'DTAGQ',
    ]
#two most numerable clusters
#maybes1 = etcpicks.sort_values(10).tail(100).index.tolist()
tc.sort_index(inplace=True)
#maybes2 = etcpicks.sort_values(1).tail(100).index.tolist()

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

