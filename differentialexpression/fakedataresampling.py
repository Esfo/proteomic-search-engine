from random import sample
from collections import Counter
import concurrent
import numpy as np
from time import time

aminoacids = 'ACTYKNMPWSQGHFDIVERL'
aminoacids = [i for i in aminoacids]

np.random.seed(42)
proteinsizes = np.random.randint(50,800, size=1000)

proteome = []
for k in proteinsizes:
    np.random.seed(42)
    proteome.append(''.join((np.random.choice(aminoacids, size=k))))

combinedsequences = '|'.join((proteome))

motifmin = 5
motifmax = 20
patternspace = range(motifmin, motifmax+1)
motifpatterns = []

for p in patternspace:
    splits = [combinedsequences[i:i+p] for i in range(len(combinedsequences)-p+1)]
    splits = [i for i in splits if '|' not in i]
    splitcount = Counter(splits)
    keeps = splitcount.most_common(len(splitcount)//4)
    keeps = [m for (m, v) in keeps]
    motifpatterns.extend(keeps)

proteinindex = list(range(len(proteome)))


#Resampling parts below
nproteins = 100
nsamples = 10000

def psample(proteinindex, nsamples, nproteins):
    proteinresamples = []
    for n in range(nsamples):
        proteinresamples.append(sample(proteinindex, nproteins))
    return np.asarray(proteinresamples)

def pmindex(proteomemotifs, proteinresamples):
        return proteomemotifs[proteinresamples].sum(axis=1)

def motifcounting(motifpatterns, proteome, proteinresamples, procs=6):
    with concurrent.futures.ThreadPoolExecutor(procs) as executor:
        futures = []
        for p in motifpatterns:
            proteomemotifs = np.asarray([i.count(p) for i in proteome])
            futures.append(executor.submit(pmindex, proteomemotifs, proteinresamples))
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return np.asarray(results)

proteinresamples = psample(proteinindex, nsamples, nproteins)
nproteins = 100
nsamples = 10000

def psample(proteinindex, nsamples, nproteins):
    proteinresamples = []
    for n in range(nsamples):
        proteinresamples.append(sample(proteinindex, nproteins))
    return np.asarray(proteinresamples)

def pmindex(proteomemotifs, proteinresamples):
        return proteomemotifs[proteinresamples].sum(axis=1)

def motifcounting(patterns, proteome, proteinresamples, procs=6):
    with concurrent.futures.ThreadPoolExecutor(procs) as executor:
        futures = []
        for p in patterns:
            proteomemotifs = np.asarray([i.count(p) for i in proteome])
            futures.append(executor.submit(pmindex, proteomemotifs, proteinresamples))
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return np.asarray(results)

proteinresamples = psample(proteinindex, nsamples, nproteins)

t = time()
motifresamples = motifcounting(motifpatterns, proteome, proteinresamples)
print(time() - t)

#This doesn't currently keep track of which motif is which output, a process I'm not entirely concerned with at the moment. I'm more interested in speeding up the rate at which I can accumulate the finalized form of this information, if at all possible.
