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

folder = '/store/drosophila/PXD005713/crux-output'

#proteome = '/home/sfo/data/fastas/proteomes/Fly_Drosophila_melanogaster_Isoforms.fasta'
proteome = '/home/sfo/data/fastas/proteomes/Fly_Drosophila_melanogaster.fasta'

pmin = 2
pmax = 50
outstart = '/home/sfo/data/motifs/drosophila/by-motif/'
outgeneration = True
outgeneration = False

motifgen = True
#motifgen = False

#meansabove = 2
unique = True
unique = False
#patternfile = '/store/drosophila/PXD005713/full4.patterns.pickle'
patternfolder = '/store/drosophila/PXD005713/'

if unique:
    patternstring = 'unique'
else:
    patternstring = 'full'

patternfile = ''.join((patternfolder, patternstring, str(pmin), '_', str(pmax), '.patterns.pickle'))
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
        splits = list(map(func, sequencelist))
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
#patterns = patterns[:200]
#nfs = {}
#for n, (k, v) in enumerate(seqs.items()):
#    nfs[k] = v
#    if n > 100:
#        break
#seqs = nfs

if motifgen:
    mt = time()
    pmotifs = defaultdict(lambda: defaultdict(int))
    pn = 0
    plen = len(patterns)
    for patternstring in patterns:
        search = re.compile(patternstring)
        for k, p in seqs.items():
            matches = search.finditer(p)
            for m in matches:
                pmotifs[k][patternstring] += 1
        pn += 1
        sys.stdout.write(f'\r{pn}/{plen}')
        sys.stdout.flush()
    print(f'generated motifs in {time() - mt}')


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

#def aggfunc(outfolder, files):
#    taggs = {}
#    pmotifs = defaultdict(list)
#    ofc = 1
#    flen = len(files)
#    for of in files:
#        ofn = '/'.join((outfolder, of))
#        df = pd.read_csv(ofn, low_memory=False, keep_default_na=False, na_values=['____'])
#        df.set_index('Motif', inplace=True)
#        df.sort_index(inplace=True)
#        df.loc[:,'%D from CT'] = df.loc[:,'Distance from C-Term'] / df.loc[:,'Protein Length']
#        targs = df.index.unique()
#        plen = len(targs)
#        pn = 1
#        for targ in targs:
#            #taggs[targ] = df.loc[targ].reset_index().groupby('Protein').agg({'Motif':'count', '%D from CT': 'mean', 'Protein Length': 'mean'})
#            taggs[targ] = df.loc[targ].reset_index().groupby('Protein').agg({'Motif':'count', 'Protein Length': 'mean'})
#            for pro in set(df.loc[targ, 'Protein']):
#                pmotifs[pro].append(targ)
#            sys.stdout.write(f'\r{pn}/{plen}, file {ofc}/{flen}              ')
#            sys.stdout.flush()
#            pn += 1
#        ofc += 1
#        gc.collect()
#    return taggs, pmotifs

#mt = time()
#taggs, pmotifs = aggfunc(outfolder, files)
#print(time() - mt, '- finished aggregating motifs')

singleexpressers = expressiongroups.loc[expressiongroups == 1].index
single20hr = wnsaf.loc[singleexpressers].loc[wnsaf.loc[singleexpressers,20] > 0, 20].index

def samplesave(td, on):
    td.to_csv(on)

nproteins = len(single20hr)
nsamples = 10000
nt = time()
#startframe = pd.DataFrame(columns=['Motif', 'Freq'])
with concurrent.futures.ProcessPoolExecutor(8) as executor:
    for n in range(nsamples):
        mt = time()
        samples = random.sample(proteins, nproteins)
        on = ''.join(('/home/sfo/data/motifs/drosophila/resampling/rs', str(n), '.csv'))
        startframe = pd.DataFrame(index=patterns)
        startframe.loc[:,'freq'] = 0
        #startframe.to_csv(on, index=False)
        prs = []
        for s in samples:
            td = pd.DataFrame.from_dict(pmotifs[s], orient='index', columns=[s])
            td.loc[:,s] = td.loc[:,s] / (len(seqs[s]) / td.index.str.len())
            startframe.loc[td.index, 'freq'] += td.loc[:,s]
        executor.submit(samplesave, startframe, on)
        #prs.append(td)
        #prs = pd.concat(prs)
        #prs.fillna(0, inplace=True)
        #prs = prs.sum(axis=1)
        #on = ''.join(('/home/sfo/data/motifs/drosophila/resampling/rs', str(n), '.csv'))
        #prs.to_csv(on)
        #executor.submit(samplesave, list(prs), n)
        #executor.submit(samplesave, samples, n)
        sys.stdout.write(f'\r{n+1}/{nsamples} - {time() - mt}')
        sys.stdout.flush()
print(f' \n {time() - nt}')

