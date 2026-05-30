from pyteomics import mass
from collections import Counter, defaultdict
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from Bio import SeqIO
import sys
import re
import itertools
plt.rcParams["figure.dpi"] = 300
np.set_printoptions(suppress=True)

#element: {isotope mass: probability of isotope}
#source: http://education.expasy.org/student_projects/isotopident/htdocs/motza.html
isotopes = {
        'H': {1.0078250321:	.99984426, 2.0141017780: .00015574},
        'C': {12.0000000: .988922, 13.0033548378: .011078},
        'N': {14.0030740052: .996337, 15.0001088984: .003663},
        'O': {15.9949146221: .997628, 16.99913150: .000372, 17.9991604: .002000},
        'S': {31.97207069: .95018, 32.97145850: .00750, 33.96786683: .04215, 35.96708088: .00017}
        }

minorisotopes = {
        'H': {2.0141017780: .00015574},
        'C': {13.0033548378: .011078},
        'N': {15.0001088984: .003663},
        'O': {16.99913150: .000372, 17.9991604: .002000},
        'S': {32.97145850: .00750, 33.96786683: .04215, 35.96708088: .00017}
        }

majorisotopemasses = {
        'H': 1.0078250321,
        'C': 12.0000000,
        'N': 14.0030740052,
        'O': 15.9949146221,
        'S': 31.97207069
        }

majorisotopeprobs = {
        'H': .99984426,
        'C': .988922,
        'N': .996337,
        'O': .997628,
        'S': .95018
        }


originalelementcount = defaultdict(dict)
massadditions = defaultdict(dict)
for e, i in isotopes.items():
    maxprob = max(i.values())
    dominantisotope = [k for k, v in i.items() if v == maxprob][0]
    for m, p in i.items():
        originalelementcount[e][m-dominantisotope] = 0
        massadditions[e][m-dominantisotope] = p
originalelementcount = dict(originalelementcount)
massadditions = dict(massadditions)


#reference: https://www.sigmaaldrich.com/US/en/technical-documents/technical-article/protein-biology/protein-structural-analysis/amino-acid-reference-chart
aminoacidcomposition = {
        'A': {'C': 3, 'H': 5, 'N': 1, 'O': 1},
        'R': {'C': 6, 'H': 12, 'N': 4, 'O': 1},
        'N': {'C': 4, 'H': 6, 'N': 2, 'O': 2},
        'D': {'C': 4, 'H': 5, 'N': 1, 'O': 3},
        'C': {'C': 3, 'H': 5, 'N': 1, 'O': 1, 'S': 1},
        'Q': {'C': 5, 'H': 8, 'N': 2, 'O': 2},
        'E': {'C': 5, 'H': 7, 'N': 1, 'O': 3},
        'G': {'C': 2, 'H': 3, 'N': 1, 'O': 1},
        'H': {'C': 6, 'H': 7, 'N':3, 'O': 1},
        'I': {'C': 6, 'H': 11, 'N':1, 'O': 1},
        'L': {'C': 6, 'H': 11, 'N':1, 'O': 1},
        'K': {'C': 6, 'H': 12, 'N': 2, 'O': 1},
        'M': {'C': 5, 'H': 9, 'N':1, 'O': 1, 'S': 1},
        'F': {'C': 9, 'H': 9, 'N':1, 'O': 1},
        'P': {'C': 5, 'H': 7, 'N':1, 'O': 1},
        'S': {'C': 3, 'H': 5, 'N':1, 'O': 2},
        'T': {'C': 4, 'H': 7, 'N':1, 'O': 2},
        'W': {'C': 11, 'H': 10, 'N': 2, 'O': 1},
        'Y': {'C': 9, 'H': 9, 'N': 1, 'O': 2},
        'V': {'C': 5, 'H': 9, 'N': 1, 'O': 1}
        }

def seqsplit(sequences, cut, minlength, maxlength, missedcleavages):
    '''
    Cuts peptide sequences using two dicts, one of enzyme cut sites and the other of spots they can't hit.
    Input can be either a list, or a list of lists. Output is a list.
    '''
    cutsequences = sequences[:]
    for site in cut[0]:
        if cut[0][site] == 0:
            splitstring = r''.join(('(?=[', site, '](?!', cut[1][site], '))')) if cut[1][site] else r''.join(('(?=[', site, '])'))
        elif cut[0][site] == 1:
            splitstring = r''.join(('(?<=[', site, '](?!', cut[1][site], '))')) if cut[1][site] else r''.join(('(?<=[', site, '])'))
        cutsequences = list(map(lambda x: re.split(splitstring, x), cutsequences)) if type(cutsequences[0]) is str else [[i for j in list(map(lambda x: re.split(splitstring, x), c)) for i in j] for c in cutsequences]
    slist = []
    for s in range(len(cutsequences)):
        midlist = []
        for y in range(missedcleavages+1):
            midlist.append([''.join((cutsequences[s][i:i+y+1])) for i in range(len(cutsequences[s]))])
        if cutsequences[s][0].startswith('M'): #N-terminal cleavage, makes the cleaved and uncleaved version of the n-terminal peptide
            midlist.append([''.join((cutsequences[s][0][1:], ''.join((cutsequences[s][1:y+1])))) for y in range(missedcleavages+1)])
        midlist = list(set([i for j in midlist for i in j]))
        slist.append(midlist)
    slist = [list(filter(lambda x: maxlength >= len(x) >= minlength, i)) for i in slist]
    for i in range(len(slist)):
        slist[i].sort(key=lambda x: (sequences[i].find(x), len(x)))
    return slist

def isotope_characterization(elementcount, massadditions):
    prob = 1
    mass = 0
    for e, v in elementcount.items():
        vals = v.values()
        vsum = sum(vals)
        if vsum > 0:
            multiplier = 1
            n = 0
            eprob = 1
            csum = 0
            viters = sorted(v.items(), key=lambda x: x[1])
            for m, c in viters:
                if c > 0:
                    if m > 0:
                        multiplier *= special.comb(vsum-n, c, exact=True)
                        n += 1
                    eprob *= (massadditions[e][m]**c)
                    mass += m*c
                    csum += c
                    if csum >= vsum:
                        break
            prob *= eprob * multiplier
    return prob, mass

def isotope_elucidation(massadditions, elementcount, samplesize):
    abundanceprobs = {}
    fullprob, standingmass = isotope_characterization(elementcount, massadditions)
    if fullprob * samplesize > 1:
        abundanceprobs[standingmass] = fullprob
        for e, v in massadditions.items():
            if elementcount[e][0] > 0:
                vsum = sum(v.values())
                for m, p in v.items():
                    if m > 0:
                        if standingmass + m not in abundanceprobs:
                            elementcount[e][0] -= 1
                            elementcount[e][m] += 1
                            abundanceprobs.update(isotope_elucidation(massadditions, elementcount, samplesize))
                            elementcount[e][0] += 1
                            elementcount[e][m] -= 1
    return abundanceprobs

isotopemasses = {k:np.array(list(v.keys())) for k, v in isotopes.items()}
isotopediffs = {k:np.unique(np.abs(v - v.reshape(-1,1)))[1:] for k, v in isotopemasses.items()}
additionalisotopearray = np.unique(np.hstack(list(isotopediffs.values()))) #differences you'd expect to find +/- from a peak if there was an isotope added or subtracted
neighboringisotopearray = np.unique(np.abs(additionalisotopearray - additionalisotopearray.reshape(-1,1)))[1:] #differences you'd expecting to find +/- from an isotope peak if there were alternative isotopes from the original reference peak. One degree of separation
#^to get to the 2nd degree of separation, you'd need to take the matrix diffs of that array, the list of numbers gets longer and longer though
#^another note, this is for single-isotope addition. for 2 or more this won't have large enough numbers

proteomefile = '/home/sfo/data/fastas/proteomes/Human_Homo_sapien-NoTremb.fasta'
fasta = SeqIO.parse(open(proteomefile), 'fasta')
seqs = []
for fasta in fasta:
    sequence, idn = str(fasta.seq), fasta.id
    seqs.append(sequence)

cut = [{'K':1, 'R':1}, {'K':'P', 'R':'P'}] #trypsin
missedcleavages = 1
minlength = 6
maxlength = 30
samplesize = 100000

cutseqs = seqsplit(seqs, cut, minlength, maxlength, missedcleavages)
cutseqs = set(itertools.chain(*cutseqs))
#proteome = ''.join((seqs))

#proteomecounts = Counter(proteome)
#mincounter = min(v for k, v in proteomecounts.items() if k in aminoacidcomposition)

#set up samplesize, calculte atomicisotopeabundances, like below
#calculate possible masses up til when the number of an isotopic element is >= the number o this element in atomicisotopeabundances
#make the relative abundances derive from the amounts in atomicisotopeabundances
#^ the total number in atomicisotopeabundances can be evenly split amongst every isotope accounted for. The peptides with the most isotopes will get that +relative abundance from the number of isotopes distributed to it
#^the distributed amounts could even be put into floating form? Nah.. idk

#import goal: find a reasonable way to estimate a max # of each specific isotope based on a sample size input
#once yu have that max, you can generate combinations/permuations using itertools of the max # of each isotope for any given peptide
#from that you should be able to get a relative abundance distribution by evenly distributing what's in atomicisotopeabundances, like stated above

elementcount = originalelementcount.copy()
sn = 0
for seq in cutseqs:
    sn += 1
    if sn > 100:
        break
    for e, v in elementcount.items():
        for m, c in v.items():
            elementcount[e][m] = 0

    atomiccomposition = Counter()
    for aa in seq:
        atomiccomposition += aminoacidcomposition[aa]
#no OH loss on last residue, no H lost on first residue
    atomiccomposition['H'] += 2
    atomiccomposition['O'] += 1

    monoisotopicmass = sum(majorisotopemasses[k]*v for k, v in atomiccomposition.items())

    for e, c in atomiccomposition.items():
        elementcount[e][0] += c


    abundanceprobs = isotope_elucidation(massadditions, elementcount, samplesize)
    abundanceprobs = Counter(abundanceprobs)

    abundancedistribution = {}
    sumprob = sum(abundanceprobs.values())
    for m, p in abundanceprobs.items():
        abundancedistribution[monoisotopicmass+m] = p / sumprob
    abundancedistribution = Counter(abundancedistribution)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,4))

    ax[0].bar(abundancedistribution.keys(), abundancedistribution.values(), alpha=0.2)
    #plt.show()


#resampling below

    masses = np.zeros(shape=samplesize)
    for e, count in atomiccomposition.items():
        samples = list(isotopes[e].keys())
        weights = list(isotopes[e].values()) 
        adder = np.random.choice(samples, size=samplesize*count, p=weights).reshape(-1,count).sum(axis=1)
        masses += adder

    dist = Counter(masses.tolist())

    ndist = {}
    for m, p in dist.items():
        ndist[m] = p / samplesize
    ndist = Counter(ndist)

    ax[1].bar(ndist.keys(), ndist.values(), alpha=0.2)
    fig.suptitle(seq)
    plt.show()


#notes

#once you have all the isotope distributions of peptides in a proteome:
#see how many neighboring-mass peptides have similar looking isotope distributions.
#^you can allow a range of +/- 0.1 daltons or something, then cross-correlate everything inside these bins, allow redundant bins at a 0.05 dalton overlap.
#then plot the distribution of correlations -> tighten the dalton range and make the overlap always half of it.
#^ perhaps instead of correlation, something else might work here. And this would be a good environment to DEVELOP a metric/type of measurement for you to use on the isotope-matching component of this whole shin-dig.
#I bet it'd be a sick notebook addition too, to show a nice metric thing or whatever that can distinguish close mass matches really well. After all - that's what you'd want it to do!

#this relies on mass accumulation of differing isotopes leading to unique masses -> which they do! (I'm pretty sure, not that I've checked lmao)
#would you be able to estimate molar amount by accurately quantifying true sample size of isotopic ratios from mass spectra?
#It would also be cool to visualize isotope %'s as a time series across number of samples.
#^Maybe this would help with the process of identifying a method of separating isotopic distributions at neighboring masses. -> To have it be valid across the changing number of samples.
