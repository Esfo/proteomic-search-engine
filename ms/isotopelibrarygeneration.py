from collections import Counter, defaultdict
import numpy as np
from scipy import special, spatial, stats
import matplotlib.pyplot as plt
from Bio import SeqIO
import sys
import re
import itertools
import string
import concurrent.futures
from functools import partial
import multiprocessing as mp
import sqlitedict as sq
import lmdb
import pandas as pd
import pickle
from time import time
import os
import gc
gc.enable()

plt.rcParams["figure.dpi"] = 300
#np.set_printoptions(suppress=True)

if os.uname()[1] == 'toaster':
    plt.rcParams['figure.dpi'] = 180
elif os.uname()[1] == 'box':
    plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.facecolor'] = 'gray'
plt.rcParams['figure.facecolor'] = 'gray'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['ytick.labelcolor'] = 'white'
plt.rcParams['xtick.labelcolor'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['xtick.color'] = 'white'

saving = True
saving = False

#monokeys = False #whether to use monoisotopic or max-intensity masses as keys

#librarylocation = '/home/sfo/data/proteomics/fastas/search-db/library.sqlite'
librarylocation = '/home/sfo/data/proteomics/fastas/search-db/'

proteomefile = '/home/sfo/data/proteomics/fastas/proteomes/Human_Homo_sapien.fasta'
#proteomefile = '/home/sfo/data/proteomics/fastas/proteomes/Mouse_Mus_musculus.fasta'
proteome = proteomefile.split('/')[-1].split('.')[0]

environment_partial = partial(lmdb.Environment, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=9999, max_spare_txns=1, lock=True)


#with environment_partial(librarylocation) as env:
#    defaults = env.open_db('defaults'.encode())
#    with env.begin(write=False) as txn:
#        with txn.cursor(defaults) as cursor:
#            samplesize = int(cursor.get('samplesize'.encode()).decode())
samplesize = 100

fasta = SeqIO.parse(open(proteomefile), 'fasta')
seqs = []
for f in fasta:
    sequence, idn = str(f.seq), f.id
    seqs.append(sequence)

cut = [{'K':1, 'R':1}, {'K':'P', 'R':'P'}] #trypsin
#cut = [{'K':1, 'R':1}, {'K':'', 'R':''}] #trypsin + lys-c
missedcleavages = 2
minlength = 6
maxlength = 50
#samplesize = 100 #minimum percentage is 1 / this
maxvmods = 3 #max combination of variable modifications to be applied to any peptide

#if monokeys:
#    isotopefile = ''.join((librarylocation, proteomefile.split('/')[-1].split('_')[0].lower(), '_isotopes-', str(minlength), '-', str(maxlength), '_miss-', str(missedcleavages), '_ss', str(samplesize), '-by_mono', '.pickle'))
#    isofactorfile = ''.join((librarylocation, proteomefile.split('/')[-1].split('_')[0].lower(), '_isotopes-', str(minlength), '-', str(maxlength), '_miss-', str(missedcleavages), '_ss', str(samplesize), '-by_mono', '.factors.pickle'))
#else:

#if not librarylocation.endswith('/'):
#    librarylocation = ''.join((librarylocation, '/'))
#
#foldername = ''.join((librarylocation, proteomefile.split('/')[-1].split('_')[0].lower(), '_isotopes-', str(minlength), '-', str(maxlength), '_miss-', str(missedcleavages), '_ss', str(samplesize)))
#if not os.path.isdir(foldername):
#    os.mkdir(foldername)

#isotopefile = '/home/sfo/data/proteomics/fastas/isotope-arrays/human-isotopes-6-100_ss200.pickle'
#isotopefile = '/home/sfo/data/proteomics/fastas/isotope-arrays/human-isotopes-6-100_ss200-by_mono.pickle'

proton = 1.007276554940804

elementalmasses = {
            'H': {1: 1.007825032239, 2: 2.0141017781212},
            'C': {12: 12.000000000, 13: 13.0033548350723},
            'N': {14: 14.0030740044320, 15: 15.0001088988864},
            'O': {16: 15.9949146195717, 17: 16.9991317565069, 18: 17.9991596128676},
            'S': {32: 31.972071174414, 33: 32.971458909815, 34: 33.96786700447, 36: 35.9670807120}}

elementalprobabilities = {
        'H': {1: 0.99988570, 2: 0.00011570},
        'C': {12: 0.98938, 13: 0.01078}, 
        'N': {14: 0.9963620, 15: 0.0036420},
        'O': {16: 0.9975716, 17: 0.000381, 18: 0.0020514},
        'S': {32: 0.949926, 33: 0.00752, 34: 0.042524, 36: 0.00011}}

isotopes = {}
majorisotopemasses = {}
for k, v in elementalmasses.items():
    probs = elementalprobabilities[k]
    isotopes[k] = {}
    n = 0
    for sk, sv in v.items():
        isotopes[k][sv] = probs[sk]
        if n == 0:
            majorisotopemasses[k] = sv
            n += 1

massadditions = defaultdict(dict) #element: mass addition: probability
isotopomersbyaddition = defaultdict(dict) #element: mass addition: isotopomer
for e, i in isotopes.items():
    maxprob = max(i.values())
    dominantisotope = [k for k, v in i.items() if v == maxprob][0]
    for m, p in i.items():
        massadd = m - dominantisotope
        massadditions[e][massadd] = p
        
        elecount = [k for k, v in elementalprobabilities[e].items() if v == p][0]
        isotopomersbyaddition[e][massadd] = elecount
massadditions = dict(massadditions)
isotopomersbyaddition = dict(isotopomersbyaddition)


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

#static amino acid modifiers that are always considered present
staticmods = {
        #'C': {'H': 5, 'C': 3, 'N':1, 'O':1}, #acrylamide
        }

for aa, sad in staticmods.items():
    for saa, sav in sad.items():
        aminoacidcomposition[aa][saa] += sav

#need to modify this organization to allow more than one type of mod on the same AA
variablemods = {
        'C': {'H': 5, 'C': 3, 'N':1, 'O':1}, #acrylamide
        #H(3) C(2) N O}, #carbamidomethyl, iodoacetamide derivative
        }

#there should be more than enough in the alphabet for any reasonable number of variable mods I suppose
newmods = {} #new AA letter: its atomic composition
modifiers = defaultdict(set) #existing AA: [new AA letters]
modoriginals = {} #new AA letter: existing AA
variablecharacters = string.ascii_lowercase
for vn, (va, vad) in enumerate(variablemods.items()):
    representativecharacter = variablecharacters[vn]
    modifiers[va].add(representativecharacter)
    newmods[representativecharacter] = vad
    modoriginals[representativecharacter] = va
    aminoacidcomposition[representativecharacter] = aminoacidcomposition[va].copy()
    for vaa, vav in vad.items():
        aminoacidcomposition[representativecharacter][vaa] += vav

#for variable mods, define new AA's and make clones in cutseqs
#forced mods(mods without defined molecular components) can be added after distributions are processed, anything with a matching AA has another dist made with the +difference added to everything

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


#this is also worth a code review post
def limited_multinomial(elementcount, massadditions):
    prob = 1
    mass = 0
    for e, v in elementcount.items():
        vals = v.values()
        vsum = sum(vals)
        if vsum:
            #multiplier = 1
            #eprob = 1
            n = 0
            csum = 0
            #viters = sorted(v.items(), key=lambda x: x[1]) #this didn't seem to matter when i tested it, everything came out the same, shaved off 1/3 of the time, I think it was supposed to make the csum limiter process more efficient
            for m, c in v.items():
                if c > 0:
                    if m > 0:
                        mass += m * c
                        #multiplier *= special.comb(vsum-n, c, exact=True)
                        prob *= special.comb(vsum-n, c, exact=True)
                        n += 1
                    #eprob *= (massadditions[e][m]**c)
                    prob *= massadditions[e][m]**c
                    csum += c
                if csum >= vsum: #slight speed boost, time saver
                    break
            #prob *= eprob * multiplier
            #^speed improvement by taking this out and simplifying it above, this brough upon a super tiny calculation difference that isn't honestly relevant, it's like 10 decimal points down
    return prob, mass

#def expansion_organizer(massadditions, elementcount, samplesize):
#    abundanceprobs = {}
#    fullprob, standingmass = limited_multinomial(elementcount, massadditions)
#    if fullprob * samplesize > 1:
#        abundanceprobs[standingmass] = fullprob
#        for e, v in massadditions.items():
#            if e in elementcount:
#                if elementcount[e][0] > 0: #even though this leaves out iterations of sulfur isotopes on some iterations, it later makes up for those losses in the progenitor-recursive loops everything will eventually take from the 0 spot in a combinatoric manner in the same way that the recursive loops aim to make this hit all the spots -> i only need one strategy, not both, so this is fine. the elementcount mass additions also don't need to be ordered by their probabilities in order for this to work
#                    for m, p in v.items():
#                        if m > 0:
#                            if standingmass + m not in abundanceprobs:
#                                elementcount[e][0] -= 1
#                                elementcount[e][m] += 1
#                                abundanceprobs.update(expansion_organizer(massadditions, elementcount, samplesize))
#                                elementcount[e][0] += 1
#                                elementcount[e][m] -= 1
#    return abundanceprobs

#def expansion_organizer(massadditions, elementcount, samplesize):
#    abundanceprobs = {}
#    fullprob, standingmass = limited_multinomial(elementcount, massadditions)
#    if fullprob * samplesize > 1:
#        abundanceprobs[standingmass] = fullprob
#        for e, v in elementcount.items():
#            if v[0] > 0: #even though this leaves out iterations of sulfur isotopes on some iterations, it later makes up for those losses in the progenitor-recursive loops everything will eventually take from the 0 spot in a combinatoric manner in the same way that the recursive loops aim to make this hit all the spots -> i only need one strategy, not both, so this is fine. the elementcount mass additions also don't need to be ordered by their probabilities in order for this to work
#                for m, p in v.items():
#                    if m > 0:
#                        if standingmass + m not in abundanceprobs:
#                            elementcount[e][0] -= 1
#                            elementcount[e][m] += 1
#                            abundanceprobs.update(expansion_organizer(massadditions, elementcount, samplesize))
#                            elementcount[e][0] += 1
#                            elementcount[e][m] -= 1
#    return abundanceprobs

#def expansion_organizer(massadditions, elementcount, samplesize, isotopomersbyaddition):
#    abundanceprobs = defaultdict(dict)
#    fullprob, standingmass = limited_multinomial(elementcount, massadditions)
#    if fullprob * samplesize > 1:
#        stringformula = ''
#        for e, v in elementcount.items():
#            if v[0] > 0:
#                for m, p in v.items():
#                    if p > 0:
#                        #calculate isotopomer stringformula here, ie H1(300)H2(14) etc
#                        stringformula += ''.join((e, str(isotopomersbyaddition[e][m]), '(', str(p), ')'))
#                    if m > 0:
#                        if standingmass + m not in abundanceprobs: #relying on all combinatorics of these masses being unique, i'm pretty sure they end up that way because of how all massadditions are unique, any calculatable overlap would probably be way further away than any of the length limitations would realistically allow
#                            elementcount[e][0] -= 1
#                            elementcount[e][m] += 1
#                            abundanceprobs.update(expansion_organizer(massadditions, elementcount, samplesize, isotopomersbyaddition))
#                            elementcount[e][0] += 1
#                            elementcount[e][m] -= 1
#            #else:
#                #you don't need to worry about these because the combination of all non-0 mass addition spots get hit combinatorically like this, for as many as there are - given it passes samplesize
#        abundanceprobs[standingmass][stringformula] = fullprob
#    return abundanceprobs

def expansion_organizer(massadditions, elementcount, elementpriority, samplesize, isotopomersbyaddition):
    abundanceprobs = defaultdict(dict)
    fullprob, standingmass = limited_multinomial(elementcount, massadditions)
    if fullprob * samplesize > 1:
        stringformula = stringformula = ''.join((''.join((f'{e}{isotopomersbyaddition[e][m]}({c})' for m, c in v.items() if c > 0)) for e, v in elementcount.items()))
        for e, m, tp in elementpriority:
            if standingmass + m not in abundanceprobs: #relying on all combinatorics of these masses being unique, i'm pretty sure they end up that way because of how all massadditions are unique, any calculatable overlap would probably be way further away than any of the length limitations would realistically allow
                elementcount[e][0] -= 1
                elementcount[e][m] += 1
                #abundanceprobs.update(expansion_organizer2(massadditions, elementcount, elementpriority, samplesize, isotopomersbyaddition))
                aprobs = expansion_organizer(massadditions, elementcount, elementpriority, samplesize, isotopomersbyaddition)
                elementcount[e][0] += 1
                elementcount[e][m] -= 1
                if not aprobs:
                    break
                abundanceprobs.update(aprobs)
            #else:
                #you don't need to worry about these because the combination of all non-0 mass addition spots get hit combinatorically like this, for as many as there are - given it passes samplesize
        abundanceprobs[standingmass][stringformula] = fullprob
    return abundanceprobs


#def distribution_generation(formulastring, samplesize, monoisotopicmass, massadditions, atomiccomposition): #making way for partial
def distribution_generation(samplesize, massadditions, formulastring, monoisotopicmass, atomiccomposition):
    #atomiccomposition = Counter()
    #for aa in seq:
    #    atomiccomposition += aminoacidcomposition[aa]
    ##no OH loss on last residue, no H lost on first residue
    #atomiccomposition['H'] += 2
    #atomiccomposition['O'] += 1
    
    #elementcount = {}
    ##avoiding the need for a deepcopy, gave a ~useable speed boost
    #for k, v in massadditions.items():
    #    elementcount[k] = v.copy()
    #for e, v in elementcount.items():
    #    for m, c in v.items():
    #        elementcount[e][m] = 0
    #for e, c in atomiccomposition.items():
    #    elementcount[e][0] += c
    
    elementcount = {}
    for e, c in atomiccomposition.items():
        elementcount[e] = {0: c}
    for e, v in massadditions.items():
        if e in atomiccomposition:
            for m, c in v.items():
                if m > 0:
                    elementcount[e][m] = 0
    
    elementpriority = []
    for element, count in atomiccomposition.items():
        for m, p in massadditions[element].items():
            if m > 0:
                elementpriority.append([element, m, p*count])
    elementpriority = sorted(elementpriority, key=lambda x: -x[2])
    
    abundanceprobs = {}
    while not abundanceprobs:
        abundanceprobs.update(expansion_organizer(massadditions, elementcount, elementpriority, samplesize, isotopomersbyaddition))
        samplesize *= 2
    
    #abundancedistribution = Counter()
    massesandabundances = [[], []]
    formulas = []
    for m, fp in abundanceprobs.items():
        for f, p in fp.items(): #length of fp will always be 1 because of standingmass+m blocking in expansion_organizer
            massesandabundances[0].append(monoisotopicmass + m)
            massesandabundances[1].append(p)
            formulas.append(f)
        #abundancedistribution[monoisotopicmass+m] = p

    #mostabundantmass = massesandabundances[0][massesandabundances[1].index(max(massesandabundances[1]))]
    massesandabundances = np.array(massesandabundances)
    formulas = np.array(formulas, dtype='S')
    formulas = str(formulas[massesandabundances[0].argsort()].tolist()) #keeps the size down i suppose, i won't encode yet
    massesandabundances = massesandabundances[:,massesandabundances[0].argsort()]
    return formulastring, massesandabundances, formulas


#not sure about this anymore, I forget, isn't really needed either
#isotopemasses = {k:np.array(list(v.keys())) for k, v in isotopes.items()}
#isotopediffs = {k:np.unique(np.abs(v - v.reshape(-1,1)))[1:] for k, v in isotopemasses.items()}
#additionalisotopearray = np.unique(np.hstack(list(isotopediffs.values()))) #differences you'd expect to find +/- from a peak if there was an isotope added or subtracted
#neighboringisotopearray = np.unique(np.abs(additionalisotopearray - additionalisotopearray.reshape(-1,1))) #differences you'd expecting to find +/- from an isotope peak if there were alternative isotopes from the original reference peak. One degree of separation
##^to get to the 2nd degree of separation, you'd need to take the matrix diffs of that array, the list of numbers gets longer and longer though
##^another note, this is for single-isotope addition. for 2 or more this won't have large enough numbers

nt = time()

nothanks = ['|', 'X', 'U', 'Z', 'B']
#^i need to break these down to what they actually are, in the future, and use all possible combinations
#i should re-assess the digestion process, see if it can be made concurrent
#also i think the enzyme paradigms i use should change, it might be worthwhile to make lys-c into a RP/KP cleavage type, or make sure that i'm using it this way, and then just include the RP/KP sites as missed cleavages, this might be a wiser strategy
cutseqs = seqsplit(seqs, cut, minlength, maxlength, missedcleavages)
cutseqs = set(itertools.chain(*cutseqs))
cutseqs = [i for i in cutseqs if not any(j in i for j in nothanks)]

if variablemods:
    modseqs = set()
    #the sequence without any modifications can be excluded from this process, it already exists in cutseqs, all other combinations will be generated here
    for seq in cutseqs:
        if any(i in seq for i in modifiers):
            seqcount = Counter(seq)
            removals = set()
            for sc in seqcount:
                if sc not in modifiers:
                    removals.add(sc)
            for r in removals:
                del seqcount[r]
            
            #setup for allowing multiple variable mods of the same AA
            modlocs = {}
            modsubs = {}
            idn = 0
            for n, s in enumerate(seq):
                if s in modifiers:
                    for mod in modifiers[s]:
                        modlocs[idn] = n
                        modsubs[idn] = mod
                        idn += 1
            
            for cn in range(1, maxvmods+1):
                for modcombo in itertools.combinations(modlocs, cn):
                    if len(set(modlocs[i] for i in modcombo)) == cn:
                        newseq = seq
                        for subid in modcombo:
                            newseq = ''.join((newseq[:modlocs[subid]], modsubs[subid], newseq[modlocs[subid]+1:]))
                        modseqs.add(newseq)
    cutseqs.extend(modseqs)

print(time() - nt)

#set up samplesize, calculate atomicisotopeabundances, like below
#calculate possible masses up til when the number of an isotopic element is >= the number o this element in atomicisotopeabundances
#make the relative abundances derive from the amounts in atomicisotopeabundances
#^ the total number in atomicisotopeabundances can be evenly split amongst every isotope accounted for. The peptides with the most isotopes will get that +relative abundance from the number of isotopes distributed to it
#^the distributed amounts could even be put into floating form? Nah.. idk

#import goal: find a reasonable way to estimate a max # of each specific isotope based on a sample size input
#once yu have that max, you can generate combinations/permuations using itertools of the max # of each isotope for any given peptide
#from that you should be able to get a relative abundance distribution by evenly distributing what's in atomicisotopeabundances, like stated above


#nt = time()
#
##no concurrency worked - LIES, concurrent.futures is a scam
#atomiccompositions = {} #formula string: atomic composition dict
#seqsbyformula = defaultdict(list) #formula string: [seqs]
#massesbyformula = {} #formula string: monoisotopic mass
#for seq in cutseqs:
#    atomiccomposition = Counter()
#    for aa in seq:
#        atomiccomposition += aminoacidcomposition[aa]
#    #no OH loss on last residue, no H lost on first residue
#    atomiccomposition['H'] += 2
#    atomiccomposition['O'] += 1
#    formulastring = ''.join((''.join((k, str(v))) for k, v in atomiccomposition.items()))
#    seqsbyformula[formulastring].append(seq)
#    if formulastring not in atomiccompositions:
#        monoisotopicmass = sum(majorisotopemasses[k]*v for k, v in atomiccomposition.items())
#        atomiccompositions[formulastring] = atomiccomposition
#        massesbyformula[formulastring] = monoisotopicmass
#
#print(time() - nt)

#input: seq, aminoacidcomposition, majorisotopemasses

def formula_consolidation(aminoacidcomposition, majorisotopemasses, seq):
    atomiccomposition = Counter()
    for aa in seq:
        atomiccomposition += aminoacidcomposition[aa]
    #no OH loss on last residue, no H lost on first residue
    atomiccomposition['H'] += 2
    atomiccomposition['O'] += 1
    formulastring = ''.join((''.join((k, str(v))) for k, v in atomiccomposition.items()))
    #seqsbyformula[formulastring].append(seq)
    #if formulastring not in atomiccompositions:
    monoisotopicmass = sum(majorisotopemasses[k]*v for k, v in atomiccomposition.items())
    #atomiccompositions[formulastring] = atomiccomposition
    #massesbyformula[formulastring] = monoisotopicmass
    return formulastring, monoisotopicmass, atomiccomposition, seq



nt = time()
formula_partial = partial(formula_consolidation, aminoacidcomposition, majorisotopemasses)

atomiccompositions = {} #formula string: atomic composition dict
seqsbyformula = defaultdict(list) #formula string: [seqs]
formulasbyseq = {} #seq: formula
massesbyformula = {} #formula string: monoisotopic mass
with mp.Pool(8) as pool:
    #consolidatedformulas = tuple(pool.map(formula_partial, cutseqs))
    for formulastring, monoisotopicmass, atomiccomposition, seq in pool.map(formula_partial, cutseqs):
        seqsbyformula[formulastring].append(seq)
        formulasbyseq[seq] = formulastring
        #if formulastring not in atomiccompositions: #this slows it down
        atomiccompositions[formulastring] = atomiccomposition
        massesbyformula[formulastring] = monoisotopicmass

print(time() - nt, 'formula consolidation')


#what would speed this up is grouping sequences by composition beforehand
#and for the post-distribution fragmentation distribution generation, you can group all similar fragments before calculating distributions

#abundances = {}
#seqsbymass = defaultdict(list)
#nt = time()
#with concurrent.futures.ProcessPoolExecutor(8) as executor:
#    futures = []
#    for seq in cutseqs:
#        futures.append(executor.submit(distribution_generation, seq, samplesize, majorisotopemasses, massadditions, aminoacidcomposition))
#    for f in concurrent.futures.as_completed(futures):
#        m, s, a = f.result()
#        #abundances[s] = a
#        if m in abundances:
#            if not np.array_equal(abundances[m], a):
#                print('two distributions of the same primary mass aren\'t equal!???!?!')
#                print(s, m)
#                print(a)
#        else:
#            abundances[m] = a
#        seqsbymass[m].append(s)
#for seq in cutseqs:
#    m, s, a = distribution_generation(seq, samplesize, majorisotopemasses, massadditions, aminoacidcomposition)
#    if not m in abundances:
#        abundances[m] = a
#    seqsbymass[m].append(s)


#nt = time()
#
##manager = mp.Manager() #this slows it way down :/
#abundances = {} #formula string: [[masses], [intensities]]
##abundances = manager.dict() #formula string: [[masses], [intensities]]
#with concurrent.futures.ProcessPoolExecutor(8) as executor:
#    futures = []
#    for formula, mass in massesbyformula.items():
#        futures.append(executor.submit(distribution_generation, formula, samplesize, mass, massadditions, atomiccompositions[formula]))
#        #executor.submit(distribution_generation, formula, samplesize, mass, massadditions, atomiccompositions[formula], abundances)
#    for f in concurrent.futures.as_completed(futures):
#        outformula, dist = f.result()
#        abundances[outformula] = dist
#
#print(time() - nt)


nt = time()
distribution_generation_partial = partial(distribution_generation, samplesize, massadditions)

#pulledabundances = {}
#encodedmassesbyformula = tuple(i.encode() for i in massesbyformula)
#with environment_partial(librarylocation) as env:
#    fulls = env.open_db('distributions.full'.encode())
#    with env.begin(write=False) as txn:
#        with txn.cursor(fulls) as cursor:
#            pulledabundances.update(cursor.getmulti(encodedmassesbyformula))

abundances = {}
#for k, v in pulledabundances.items():
#    key = k.decode()
#    value = np.frombuffer(v)
#    value = value.reshape(2, value.size//2)
#    abundances[key] = value
#    del massesbyformula[key]
#print(time() - nt, 'pulled', len(abundances), 'formulas from library')

with mp.Pool(8) as pool:
    #abundances.update(dict(pool.starmap(distribution_generation_partial, ((formula, mass, atomiccompositions[formula]) for formula, mass in massesbyformula.items()))))
    formulastrings, masses, subformulas = zip(*pool.starmap(distribution_generation_partial, ((formula, mass, atomiccompositions[formula]) for formula, mass in massesbyformula.items())))

abundances = dict(zip(formulastrings, masses))
abundanceformulas = dict(zip(formulastrings, subformulas))

print(time() - nt, 'made distributions')


#there should be no integer keys so this would be fine
#nk = 0
#for k in list(seqsbymass):
#    seqsbymass[nk] = seqsbymass.pop(k)
#    abundances[nk] = abundances.pop(k)
#    nk += 1

#^when using concurrency, I think the memory from recursive functions is hanging around, might be related to this below:
#https://stackoverflow.com/questions/27664427/in-python-will-memory-for-variables-in-a-recursive-function-be-freed-if-theyre


#ain't really working?
#abundances = {}
#seqsbymass = defaultdict(list)
#nt = time()
#outlist = mp.Manager().list()
#pool = mp.Pool(8)
#for seq in cutseqs:
#    pool.apply_async(distribution_generation2(seq, samplesize, majorisotopemasses, massadditions, aminoacidcomposition, outlist, mono=monokeys))
#pool.close()
#pool.join()
#print(time() - nt)
#
#for m, s, a in outlist:
#    if m in abundances:
#        if abundances[m] != a:
#            print('two distributions of the same primary mass aren\'t equal!???!?!')
#            print(s, m)
#            print(a)
#    else:
#        abundances[m] = a
#    seqsbymass[m].append(s)
#print(time() - nt)




#this needs improvements:
#only make sum distributions if there's subisos, everything else can be in max
#include sum distributions for steplimit/newinclimit values
#concurrency???

#newincmax = 0
#fulldiffmax = 0
#subisotopomerdifferences = set() #differences of one minor/major isotopomer to its adjacent isotopomer
#maxabundancedistributions = {}
#sumabundancedistributions = {}
#distributionlinker = {} #sum/max distribution id: fulldistid
#idn = 0
#for k, v in abundances.items():
#    masses, intensities = v
#    maxmass = masses[intensities.argmax()]
#    csteps = masses - masses.min()
#    maxstep = masses.size
#    steprange = proton * np.arange(maxstep)
#    stepclasses = np.abs(csteps - steprange[:,None]).argmin(axis=0)
#    cinds, counts = np.unique(stepclasses, return_counts=True)
#    csplit = counts.cumsum().tolist()
#    majordiffs = []
#    if counts.max() > 1: #subisos exist
#        stepsplit = []
#        summedintensities = []
#        majorintensities = []
#        maxmasses = []
#        meanmasses = []
#        ci = 0
#        for cs in csplit:
#            splitmasses = masses[ci:cs]
#            splitints = intensities[ci:cs]
#            meanmass = (splitmasses * splitints).sum() / splitints.sum()
#            meanmasses.append(meanmass)
#            maxmasses.append(splitmasses.max())
#            stepsplit.append(splitmasses.tolist())
#            summedintensities.append(splitints.sum())
#            majorintensities.append(splitints.max())
#            ci = cs
#        sumabundancedistributions[idn] = np.array([meanmasses, summedintensities])
#        distributionlinker[idn] = k
#        idn += 1
#        summedintensities = np.array(summedintensities)
#        maxmajorstep = summedintensities.argmax()
#        msums = summedintensities[:-1] + summedintensities[1:]
#        majordiffs.extend((np.diff(summedintensities) / msums / 2).tolist())
#        decreasing = False
#        for ii in range(len(summedintensities)-1):
#            i1 = summedintensities[ii]
#            i2 = summedintensities[ii+1]
#            if i2 > i1:
#                if decreasing:
#                    newinc = abs(i2 - i1) / (i2 + i1) / 2
#                    if newinc > newincmax:
#                        newincmax = newinc
#            if i1 > i2:
#                decreasing = True
#    else:
#        stepsplit = []
#        majorintensities = []
#        maxmasses = []
#        ci = 0
#        for cs in csplit:
#            splitmasses = masses[ci:cs]
#            splitints = intensities[ci:cs]
#            maxmasses.append(splitmasses.max())
#            stepsplit.append(splitmasses.tolist())
#            majorintensities.append(splitints.max())
#            ci = cs
#        majorintensities = np.array(majorintensities)
#        maxmajorstep = majorintensities.argmax()
#        msums = majorintensities[:-1] + majorintensities[1:]
#        majordiffs.extend((np.diff(majorintensities) / msums / 2).tolist())
#    maxabundancedistributions[idn] = np.array([maxmasses, majorintensities])
#    distributionlinker[idn] = k
#    idn += 1
#    diffmax = np.abs(majordiffs).max()
#    if diffmax > fulldiffmax:
#        fulldiffmax = diffmax
#    decreasing = False
#    for ii in range(len(majorintensities)-1):
#        i1 = majorintensities[ii]
#        i2 = majorintensities[ii+1]
#        if i2 > i1:
#            if decreasing:
#                newinc = abs(i2 - i1) / (i2 + i1) / 2
#                if newinc > newincmax:
#                    newincmax = newinc
#        if i1 > i2:
#            decreasing = True
#    intensitypoints = []
#    masspoints = []
#    #for o in overlaps:
#    for step in stepsplit:
#        if len(step) > 1:
#    #        diffs = np.abs(masses[o].max() - masses[o]) #??? what? max intensity not max mas idiot
#    #        diffs = diffs[diffs > 0]
#            rawdiffs = np.diff(step).tolist()
#            subisotopomerdifferences.update(rawdiffs)
#            #stepints = [v[i] for i in step]
#            #maxloc = stepints.index(max(stepints))
#            #maxmass = step[maxloc]
#            #minors = [i for i in step if i != maxmass]
#            #minorints = [i for i in stepints if i != stepints[maxloc]]
#            #minorloc = minorints.index(max(minorints))
#            #minormax = minors[minorloc]
#            #distofmaxes = abs(maxmass - minormax)
#            #subisotopomerdistances.add(distofmaxes)
#            #maxdistance = max(abs(maxmass-i) for i in minors)
#            #subisotopomerranges.add(maxdistance)
#    #        subisotopomerdifferences.extend(diffs.tolist())
#    #    masspoints.append(masses[o[intensities[o].argmax()]][0])
#    #    intensitypoints.append(intensities[o].max())
#    #if len(intensitypoints) > 1:
#    #    #this should actually be the minimum adjacentratio/adjacentratio to compare across neighboring isotopomer intensities, like, what is the most drastic difference across 3 isotopomers? This is a more appropriate question/answer
#    #    intensityratios = []
#    #    intensitypoints = np.array(intensitypoints)
#    #    intensitypoints = intensitypoints[np.argsort(masspoints)]
#    #    intensityratios.extend((intensitypoints[:-1] / intensitypoints[1:]).tolist())
#    #    intensitypoints = np.flip(intensitypoints)
#    #    intensityratios.extend((intensitypoints[:-1] / intensitypoints[1:]).tolist())
#    #    intensityratios = np.array(intensityratios)
#    #    intensityratios = intensityratios[intensityratios < 1]
#    #    #intensityratios[intensityratios > 1] = 1 / intensityratios[intensityratios > 1]
#    #    adjacentratios.extend(intensityratios.tolist())
#
##dcounts = dict(Counter(subisotopomerdifferences))


#nt = time()
#
#proton = 1.00727647
#
#newincmax = 0
#fulldiffmax = 0
#subisotopomerdifferences = set() #differences of one minor/major isotopomer to its adjacent isotopomer
#maxabundancedistributions = {}
#sumabundancedistributions = {}
#distributionlinker = {} #sum/max distribution id: fulldistid
#idn = 0
#for k, v in abundances.items():
#    masses, intensities = v
#    maxmass = masses[intensities.argmax()]
#    csteps = masses - masses.min()
#    maxstep = masses.size
#    steprange = proton * np.arange(maxstep)
#    stepclasses = np.abs(csteps - steprange[:,None]).argmin(axis=0)
#    cinds, counts = np.unique(stepclasses, return_counts=True)
#    csplit = counts.cumsum().tolist()
#    majordiffs = []
#    #i'd prefer sum to be the baseline and max to be the subiso exception but this can be generated all the same using the information this already makes, if needed
#    stepsplit = []
#    summedintensities = []
#    majorintensities = []
#    maxmasses = []
#    meanmasses = []
#    ci = 0
#    if counts.max() > 1: #subisos exist
#        for cs in csplit:
#            splitmasses = masses[ci:cs]
#            splitints = intensities[ci:cs]
#            meanmass = (splitmasses * splitints).sum() / splitints.sum()
#            meanmasses.append(meanmass)
#            maxmasses.append(splitmasses.max())
#            stepsplit.append(splitmasses.tolist())
#            summedintensities.append(splitints.sum())
#            majorintensities.append(splitints.max())
#            ci = cs
#        maxabundancedistributions[idn] = np.array([maxmasses, majorintensities])
#        distributionlinker[idn] = k
#        idn += 1
#        summedintensities = np.array(summedintensities)
#        majorintensities = np.array(majorintensities)
#        msums = majorintensities[:-1] + majorintensities[1:]
#        majordiffs.extend((np.diff(majorintensities) / msums / 2).tolist())
#        decreasing = False
#        for ii in range(len(majorintensities)-1):
#            i1 = majorintensities[ii]
#            i2 = majorintensities[ii+1]
#            if i2 > i1:
#                if decreasing:
#                    newinc = abs(i2 - i1) / (i2 + i1) / 2
#                    if newinc > newincmax:
#                        newincmax = newinc
#            if i1 > i2:
#                decreasing = True
#        for step in stepsplit:
#            if len(step) > 1:
#                rawdiffs = np.diff(step).tolist()
#                subisotopomerdifferences.update(rawdiffs)
#    else:
#        summedintensities = intensities
#        meanmasses = masses
#        msums = summedintensities[:-1] + summedintensities[1:]
#        majordiffs.extend((np.diff(summedintensities) / msums / 2).tolist())
#    sumabundancedistributions[idn] = np.array([meanmasses, summedintensities])
#    distributionlinker[idn] = k
#    idn += 1
#    diffmax = np.abs(majordiffs).max()
#    if diffmax > fulldiffmax:
#        fulldiffmax = diffmax
#    decreasing = False
#    for ii in range(len(summedintensities)-1):
#        i1 = summedintensities[ii]
#        i2 = summedintensities[ii+1]
#        if i2 > i1:
#            if decreasing:
#                newinc = abs(i2 - i1) / (i2 + i1) / 2
#                if newinc > newincmax:
#                    newincmax = newinc
#        if i1 > i2:
#            decreasing = True
#print(time() - nt)


#i could probably break these down into smaller functions and get more efficiency but I've done enough for now
def distribution_management(proton, k, v):
    masses, intensities = v
    maxmass = masses[intensities.argmax()]
    csteps = masses - masses.min()
    maxstep = masses.size
    steprange = proton * np.arange(maxstep)
    stepclasses = np.abs(csteps - steprange[:,None]).argmin(axis=0)
    cinds, counts = np.unique(stepclasses, return_counts=True)
    csplit = counts.cumsum().tolist()
    subisodiffs = []
    majordiffs = []
    #i'd prefer sum to be the baseline and max to be the subiso exception but this can be generated all the same using the information this already makes, if needed
    stepsplit = []
    summedintensities = []
    majorintensities = []
    maxmasses = []
    meanmasses = []
    ci = 0
    newincmax = 0
    fulldiffmax = 0
    if counts.max() > 1: #subisos exist
        condensationcoordinates = []
        for cs in csplit:
            condensationcoordinates.append(cs-ci)
            splitmasses = masses[ci:cs]
            splitints = intensities[ci:cs]
            meanmass = (splitmasses * splitints).sum() / splitints.sum()
            meanmasses.append(meanmass)
            maxmasses.append(splitmasses.max())
            stepsplit.append(splitmasses.tolist())
            summedintensities.append(splitints.sum())
            majorintensities.append(splitints.max())
            ci = cs
        #maxabundancedistributions[idn] = np.array([maxmasses, majorintensities])
        maxabundancedistribution = np.array([maxmasses, majorintensities])
        #distributionlinker[idn] = k
        #idn += 1
        summedintensities = np.array(summedintensities)
        majorintensities = np.array(majorintensities)
        msums = majorintensities[:-1] + majorintensities[1:]
        majordiffs.extend((np.diff(majorintensities) / msums / 2).tolist())
        decreasing = False
        for ii in range(len(majorintensities)-1):
            i1 = majorintensities[ii]
            i2 = majorintensities[ii+1]
            if i2 > i1:
                if decreasing:
                    newinc = abs(i2 - i1) / (i2 + i1) / 2
                    if newinc > newincmax:
                        newincmax = newinc
            if i1 > i2:
                decreasing = True
        for step in stepsplit:
            if len(step) > 1:
                rawdiffs = np.diff(step).tolist()
                subisodiffs.extend(rawdiffs)
    else:
        condensationcoordinates = [1 for i in range(maxstep)]
        summedintensities = intensities
        meanmasses = masses
        msums = summedintensities[:-1] + summedintensities[1:]
        majordiffs.extend((np.diff(summedintensities) / msums / 2).tolist())
        maxabundancedistribution = None
    #sumabundancedistributions[idn] = np.array([meanmasses, summedintensities])
    sumabundancedistribution = np.array([meanmasses, summedintensities])
    #distributionlinker[idn] = k
    #idn += 1
    diffmax = np.abs(majordiffs).max()
    if diffmax > fulldiffmax:
        fulldiffmax = diffmax
    decreasing = False
    for ii in range(len(summedintensities)-1):
        i1 = summedintensities[ii]
        i2 = summedintensities[ii+1]
        if i2 > i1:
            if decreasing:
                newinc = abs(i2 - i1) / (i2 + i1) / 2
                if newinc > newincmax:
                    newincmax = newinc
        if i1 > i2:
            decreasing = True
    return k, maxabundancedistribution, sumabundancedistribution, subisodiffs, newincmax, fulldiffmax, np.array(condensationcoordinates).astype(int)


nt = time()

#the partial won't be needed in the cli version bc proton will be imported
distribution_management_partial = partial(distribution_management, proton)

idn = 0
newincmax = 0
fulldiffmax = 0
#distributionlinker = {} #sum/max distribution id: fulldistid
maxabundancedistributions = {}
sumabundancedistributions = {}
condensationcoordinates = {} #formula: [# isotopomers per proton-step]
subisotopomerdifferences = set() #differences of one minor/major isotopomer to its adjacent isotopomer
#output = []
with mp.Pool(8) as pool:
    #output = pool.starmap(distribution_management_partial, abundances.items())
    for k, maxes, sums, subdiffs, newinc, diffmax, coords in pool.starmap(distribution_management_partial, abundances.items()):
        match maxes:
            case np.ndarray():
                maxabundancedistributions[k] = maxes
        sumabundancedistributions[k] = sums
        subisotopomerdifferences.update(subdiffs)
        condensationcoordinates[k] = coords
        if newinc > newincmax:
            newincmax = newinc
        if diffmax > fulldiffmax:
            fulldiffmax = diffmax

print(time() - nt, 'distributions organized')


floatlen = max(len(str(i)) for i in subisotopomerdifferences)
diffarray = np.sort(list(subisotopomerdifferences))

fi = floatlen
ngroups = []
while fi > 0:
    ngroups.append(np.unique(diffarray.round(fi)).size)
    fi -= 1
cutgroups = np.array(ngroups)
cutgroups = cutgroups[cutgroups < cutgroups.max()]
equillibrium = stats.mode(cutgroups, keepdims=False)[0] #most stable equillibrium aside from the starting number of groups, seems to hold up

roundval = ngroups.index(equillibrium)

subisovals = set(diffarray.round(roundval).tolist())
print('~')
print('found subisotopic distances')
for si in sorted(subisovals):
    print(si)
print('~')

newinclimit = round(newincmax+0.05, 1)
steplimit = round(fulldiffmax+0.05, 1)
uppermass = max(v[0].max() for v in abundances.values())
rum = round(uppermass)
rlen = len(str(rum))
#ceiling of the largest place
uppermasslimit = (int(str(rum)[0]) + 1) * 10**(rlen-1)
subisomax = max(subisovals) #this is basically assuming the subisovals are too small to reasonably measure

print('newinclimit', newinclimit)
print('steplimit', steplimit)
print('upper mass limit', uppermasslimit)
print('subisomax', subisomax)
print('~')

def database_addition(env, database, inputs, proteome=None, update=True):
    db = env.open_db(database.encode())
    while True:
        try:
            with env.begin(write=True) as txn:
                with txn.cursor(db) as cursor:
                    if update:
                        encodeddict = {}
                        for k, v in inputs.items():
                            try: #string key
                                key = k.encode()
                            except AttributeError: #int key, almost could just do this alone i suppose
                                #check isdigit later i suppose
                                key = str(k).encode()
                            try: #numpy array
                                value = v.tobytes()
                            except AttributeError: #integer
                                value = str(v).encode()
                            encodeddict[key] = value
                        #cursor.putmulti(inputs.items(), dupdata=False, overwrite=True, append=False)
                        #for k, v in inputs.items():
                        #    cursor.put(k.encode(), v.tobytes())
                        cursor.putmulti(encodeddict.items(), dupdata=False, overwrite=True, append=False)
                    else:
                        encodedparameters = str(inputs).encode()
                        cursor.put(proteome.encode(), encodedparameters)
                        #decode = dict(eval(encode.decode()))
            break
        except lmdb.MapFullError:
            defaults = env.open_db('defaults'.encode())
            with env.begin(write=False) as txn:
                with txn.cursor(defaults) as cursor:
                    mapaddition = int(cursor.get('mapsize'.encode()).decode())
            newmapsize = env.info()['map_size'] + mapaddition
            env.set_mapsize(newmapsize)

nt = time()
formulaidentifier = {}
distributionidentifier = {}
n = 0
for k in abundances:
    formulaidentifier[k] = n
    distributionidentifier[n] = k
    n += 2
print(time() - nt, 'linkers made')

if saving:
    with environment_partial(librarylocation) as env:
        nt = time()
        #with sq.SqliteDict(librarylocation, tablename='parameters', flag='c', autocommit=True) as db:
        parameterdict = {'minlength': minlength, 'maxlength': maxlength, 'missedcleavages': missedcleavages, 'maxvmods': maxvmods}
        #    #leaving out enzymes + mods atm, they will be a list of sorts, maybe a separate table
        #    db[proteome] = parameterdict
        #parameters = env.open('parameters')
        #with env.begin(write=True) as txn:
        #    with txn.cursor(parameters) as cursor:
        #        encodedparameters = str(parameterdict).encode()
        #        cursor.put(proteome.encode(), encodedparameters)
        #        #decode = dict(eval(encode.decode()))
        database_addition(env, 'parameters', parameterdict, proteome=proteome, update=False)
        print(time() - nt, 'params')
        
        nt = time()
        #fullisotopefile = '/'.join((foldername, 'distributions.full.pickle'))
        #with open(fullisotopefile, 'wb') as pick:
        #    pickle.dump(abundances, pick)
        #fulldb = '.'.join((proteome, 'full'))
        #with sq.SqliteDict(librarylocation, tablename='distributions.full', flag='c', autocommit=False, outer_stack=False) as db:
        #    db.update(abundances)
        #    db.commit()
        database_addition(env, 'distributions.full', abundances, update=True)
        print(time() - nt, 'full')
        
        nt = time()
        database_addition(env, 'distributions.formulas', abundanceformulas, update=True)
        print(time() - nt, 'seqs')
        
        nt = time()
        #maxisotopefile = '/'.join((foldername, 'distributions.max.pickle'))
        #with open(maxisotopefile, 'wb') as pick:
        #    pickle.dump(maxabundancedistributions, pick)
        #maxdb = '.'.join((proteome, 'max'))
        #with sq.SqliteDict(librarylocation, tablename='distributions.max', flag='c', autocommit=False, outer_stack=False) as db:
        #    db.update(maxabundancedistributions)
        #    db.commit()
        database_addition(env, 'distributions.max', maxabundancedistributions, update=True)
        print(time() - nt, 'max')
        
        nt = time()
        #sumisotopefile = '/'.join((foldername, 'distributions.sum.pickle'))
        #with open(sumisotopefile, 'wb') as pick:
        #    pickle.dump(sumabundancedistributions, pick)
        #sumdb = '.'.join((proteome, 'sum'))
        #with sq.SqliteDict(librarylocation, tablename='distributions.sum', flag='c', autocommit=False, outer_stack=False) as db:
        #    db.update(sumabundancedistributions)
        #    db.commit()
        database_addition(env, 'distributions.sum', sumabundancedistributions, update=True)
        print(time() - nt, 'sum')
        
        nt = time()
        database_addition(env, 'distributions.condensation', condensationcoordinates, update=True)
        print(time() - nt, 'condensed')
        #not making the linker here anymore
        nt = time()
        ##linkerfile = '/'.join((foldername, 'distributions.linker.pickle'))
        ##with open(linkerfile, 'wb') as pick:
        ##    pickle.dump(distributionlinker, pick)
        #linkerdb = '.'.join((proteome, 'linker'))
        #with sq.SqliteDict(librarylocation, tablename=linkerdb, flag='w', autocommit=False, outer_stack=False) as db:
        #    db.update(distributionlinker)
        #    db.commit()
        #^actually, save distributionlinkers as formula: idn, sum is always idn, idn+1 is always max, make idn: formula to go backwards too
        formulaiddb = '.'.join(('formulaidentifier', proteome))
        distiddb = '.'.join(('distributionidentifier', proteome))
        database_addition(env, formulaiddb, formulaidentifier, update=True)
        database_addition(env, distiddb, distributionidentifier, update=True)
        print(time() - nt, 'distribution linkers')
        
        nt = time()
        #seqfile = '/'.join((foldername, 'sequences.pickle'))
        #with open(seqfile, 'wb') as pick:
        #    pickle.dump(seqsbyformula, pick)
        seqdb = '.'.join(('seqsbyformula', proteome))
        #with sq.SqliteDict(librarylocation, tablename=seqdb, flag='w', autocommit=False, outer_stack=False) as db:
        #    db.update(seqsbyformula)
        #    db.commit()
        database_addition(env, seqdb, seqsbyformula, update=True)
        print(time() - nt, 'seqs')
        
        f3 = time()
        database_addition(env, 'formulasbyseq', formulasbyseq, update=True)
        print(time() - f3, 'formulas finalized')
        
        nt = time()
        #isofactorfile = '/'.join((foldername, 'isofactors.pickle'))
        #factors = [subisovals, newinclimit, steplimit, uppermasslimit]
        #with open(isofactorfile, 'wb') as pick:
        #    pickle.dump(factors, pick)
        #with sq.SqliteDict(librarylocation, tablename='isofactors', flag='c', autocommit=True) as db:
        #    db[proteome] = {'subisomax': subisomax, 'newinclimit': newinclimit, 'steplimit': steplimit, 'uppermasslimit': uppermasslimit}
        isofactors = {'subisomax': subisomax, 'newinclimit': newinclimit, 'steplimit': steplimit, 'uppermasslimit': uppermasslimit}
        database_addition(env, 'isofactors', isofactors, proteome=proteome, update=False)
        print(time() - nt, 'isofactors')
        
        nt = time()
        #aminoacidfile = '/'.join((foldername, 'aminoacids.pickle'))
        #with open(aminoacidfile, 'wb') as pick:
        #    pickle.dump(aminoacidcomposition, pick)
        #with sq.SqliteDict(librarylocation, tablename='aminoacids', flag='c', autocommit=True) as db:
        #    db[proteome] = aminoacidcomposition
        database_addition(env, 'aminoacids', aminoacidcomposition, proteome=proteome, update=False)
        print(time() - nt, 'aminoacids')


#after a decrease in a major isotopomer, what is the greatest % increase seen afterwards?
#this barely ever happens, the amount of upper limit lost could be 1/2 the total intensity lost and you'd be in the clear.
#firsts = []
#seconds = []
#decs = []
#proton = 1.00727647
#for k, v in abundances.items():
#    masses, intensities = zip(*v.items())
#    masses = np.sort(list(masses))
#    intensities = np.array(list(intensities))
#    maxmass = masses[intensities.argmax()]
#    csteps = masses - masses.min()
#    maxstep = masses.size
#    steprange = proton * np.arange(maxstep)
#    stepclasses = np.abs(csteps - steprange[:,None]).argmin(axis=0)
#    cinds, counts = np.unique(stepclasses, return_counts=True)
#    csplit = counts.cumsum().tolist()
#    stepsplit = []
#    ci = 0
#    for cs in csplit:
#        stepsplit.append(masses.tolist()[ci:cs])
#        ci = cs
#    shapekey = maxmass in stepsplit[0]
#    orderkey = len(stepsplit)
#    intmaxes = [max(v[i] for i in j) for j in stepsplit]
#    decreasing = False
#    increasing = False
#    first = False
#    for ii in range(len(intmaxes)-1):
#        i1 = intmaxes[ii]
#        i2 = intmaxes[ii+1]
#        if i2 > i1:
#            if first:
#                seconds.append(i1 / i2)
#                first = False
#            if decreasing:
#                firsts.append(i2 / i1)
#                decs.append(lastdec)
#                decreasing = False
#                first = True
#            increasing = True
#        if i1 > i2:
#            decreasing = True
#            lastdec = i1 / i2



#how often does a distribution, of n members, have n/2 steps? or any other combination?
#stepcounter = defaultdict(lambda: defaultdict(Counter)) #shapekey: distlen: (nsisos/nsteps) : count
#
##how often is the greatest difference an increase or a decrease?
#greatestdifference = defaultdict(lambda: defaultdict(Counter)) #shapekey: distlen: Counter({-1:0, 1:0}), -1 = decrease, 1 = increase
##defaultdict(<function __main__.<lambda>()>,
##            {True: defaultdict(collections.Counter,
##                         {3: Counter({-1: 74731}),
##                          4: Counter({-1: 182321}),
##                          2: Counter({-1: 141}),
##                          5: Counter({-1: 1493})}),
##             False: defaultdict(collections.Counter,
##                         {4: Counter({-1: 106083, 1: 15}),
##                          5: Counter({1: 66303, -1: 193362}),
##                          6: Counter({1: 71304, -1: 43659}),
##                          7: Counter({-1: 5751, 1: 2751}),
##                          8: Counter({-1: 10, 1: 12})})})
##while there were a decent amount of distributions who's largest diff was on an increase, it seemed like that increase had nothing to do with the lowest intensity. I only did a rough look but the ones I looked at seemed the same. The increase in question was before the max, and the lowest point came after the max.
#examples = []
##what is the greatest decrease amongst distributions who either start with their highest or don't, this would be looking at percent difference here
#greatestdecrease = defaultdict(lambda: defaultdict(float)) #shapekey: distlen: %dec
##and count abundances of whichever post-max position it goes to
#greatestdecreaseloc = defaultdict(lambda: defaultdict(Counter)) #shapekey: distlen: position: count
#greatestincrease = defaultdict(lambda: defaultdict(float))
#greatestincreaseloc = defaultdict(lambda: defaultdict(Counter)) #shapekey: distlen: position: count
##for when an increase happens after a decrease in intensity (rare event):
#newincreases = defaultdict(lambda: defaultdict(float))
#newincreaselocs = defaultdict(lambda: defaultdict(Counter)) #shapekey: distlen: position: count
#incounting = defaultdict(lambda: defaultdict(Counter)) #shapeky: distlen: number of increasing points after any decrease: count
##these come out roughly the same when considering summed minors+major(below) vs solo majors(not shown)
##it looks like a blanket 50% diff acceptance should work fine, and when an increase happens after a decrease it should be subtle, so a 5% diff acceptable should work for this
##greatestincrease
##Out[47]: 
##defaultdict(<function __main__.<lambda>()>,
##            {False: defaultdict(float,
##                         {6: 0.24367801553293772,
##                          5: 0.18450080211158587,
##                          4: 0.09366690530990736,
##                          7: 0.2718474981301093,
##                          8: 0.27808630228959397}),
##             True: defaultdict(float,
##                         {4: 0.034352897190684205, 5: 0.030370786475781935})})
##greatestdecrease
##Out[48]: 
##defaultdict(<function __main__.<lambda>()>,
##            {False: defaultdict(float,
##                         {6: 0.3528147266970217,
##                          5: 0.3712380937193526,
##                          4: 0.2966900420465198,
##                          7: 0.32152843392791974,
##                          8: 0.3281323688501174}),
##             True: defaultdict(float,
##                         {3: 0.4103186649067978,
##                          4: 0.3798190859464008,
##                          2: 0.3400286090951099,
##                          5: 0.3279585472966023})})
##greatestincreaseloc
##Out[49]: 
##defaultdict(<function __main__.<lambda>()>,
##            {False: defaultdict(collections.Counter,
##                         {6: Counter({-1: 110318, -2: 4608, 0: 8, -3: 29}),
##                          5: Counter({0: 208225, -1: 51435, -2: 5}),
##                          4: Counter({0: 106098}),
##                          7: Counter({-2: 3783, -1: 4696, -3: 23}),
##                          8: Counter({-3: 19, -2: 3})}),
##             True: defaultdict(collections.Counter,
##                         {4: Counter({0: 42554}), 5: Counter({0: 960})})})
##greatestdecreaseloc
##Out[50]: 
##defaultdict(<function __main__.<lambda>()>,
##            {False: defaultdict(collections.Counter,
##                         {6: Counter({3: 107977, 2: 6956, 1: 29, 4: 1}),
##                          5: Counter({3: 208222, 2: 51438, 1: 5}),
##                          4: Counter({2: 106034, 1: 64}),
##                          7: Counter({3: 5361, 4: 2400, 2: 722, 1: 19}),
##                          8: Counter({3: 22})}),
##             True: defaultdict(collections.Counter,
##                         {3: Counter({2: 73246, 1: 1485}),
##                          4: Counter({3: 139760, 2: 42554, 1: 7}),
##                          2: Counter({1: 141}),
##                          5: Counter({4: 532, 3: 961})})})
##with merged minors:
##newincreases
##Out[66]:
##defaultdict(<function __main__.<lambda>()>,
##            {False: defaultdict(float,
##                         {6: 0.014983913965757706, 7: 0.029094175509686612})})
##
##newincreaselocs
##Out[67]:
##defaultdict(<function __main__.<lambda>()>,
##            {False: defaultdict(collections.Counter,
##                         {6: Counter({2: 71, 0: 7}),
##                          7: Counter({3: 3, 2: 1})})})
##no merged minors:
##newincreases
##Out[59]: 
##defaultdict(<function __main__.<lambda>()>,
##            {False: defaultdict(float,
##                         {6: 0.01490136208740914,
##                          7: 0.027295322297023954,
##                          5: 0.00010362105425723491})})
##
##newincreaselocs
##Out[60]: 
##defaultdict(<function __main__.<lambda>()>,
##            {False: defaultdict(collections.Counter,
##                         {6: Counter({2: 5}),
##                          7: Counter({2: 5}),
##                          5: Counter({2: 1})})})
##incounting
##Out[83]:
##defaultdict(<function __main__.<lambda>()>,
##            {False: defaultdict(collections.Counter,
##                         {6: Counter({0: 114885, 1: 78}),
##                          5: Counter({0: 259665}),
##                          4: Counter({0: 106098}),
##                          7: Counter({0: 8498, 1: 4}),
##                          8: Counter({0: 22})}),
##             True: defaultdict(collections.Counter,
##                         {3: Counter({0: 74731}),
##                          4: Counter({0: 182321}),
##                          2: Counter({0: 141}),
##                          5: Counter({0: 1493})})})
##^the 0s are ones that didn't have any post-decrease increase
#
##collect which # step distance from the max has the highest subiso
#highestsubisostepdiffs = defaultdict(lambda: defaultdict(Counter)) #shapekey: distlen: stepdiff: count
##^the building of this could be a nice test for the isolibrary generation as well, this is what I see right now:
##defaultdict(<function __main__.<lambda>()>,
##            {False: defaultdict(collections.Counter,
##                         {5: Counter({1: 227910, 2: 31752}),
##                          4: Counter({1: 93035, 2: 12950}),
##                          6: Counter({1: 103631, 2: 11332}),
##                          7: Counter({1: 7983, 2: 519}),
##                          8: Counter({1: 13, 2: 9})}),
##             True: defaultdict(collections.Counter,
##                         {4: Counter({1: 127397, 2: 53063}),
##                          3: Counter({2: 14500, 1: 55996}),
##                          2: Counter({1: 85}),
##                          5: Counter({2: 1493})})})
##the highest major isotopomer is never accompanied by the highest subisotopomer
##it is always 1 or 2 steps away
##so to test my library generation, if I set a higher sample size... this shouldn't change. But if all of a sudden a 3 shows up in here, then the generation isn't a 'top-down' generation, it's sort of just fucking random
#
##collect the # of subisos at each step distance from the max, I'm just going to do at each step, non-relative to the max, so that different shapes of distributions could be visible
##^doing it by max now, there seems to be a prevalence of ~2x subiso incidence at 1 step past the max, usually steps prior to the max can be between 2x or 10x less with the 10x happening a lot more often
#isocountsbystep = defaultdict(lambda: defaultdict(Counter)) #shapekey: distlen: nstep: count
#
##collect the # step distance that has the highest number of subisos, and differentiate this by # of steps
#subisoaccumulationsteps = defaultdict(lambda: defaultdict(Counter)) #shapekey: distlen: stepdist: count
##differentiate ^all this from dists where the max is the first step from ones where it's not
#
##do subisos tend to occur on two majors next to each other? I think so
##do they tend to occur on the same respective side of their majors when there's more than one?
##Is a lower probability subiso more likely to occur on the other side of a major that already had one or on another major?
##also add modifiable cysteines to this process...
#
#proton = 1.00727647
#for k, v in abundances.items():
#    masses, intensities = zip(*v.items())
#    intensities = np.array(list(intensities))
#    intensities = intensities[np.argsort(masses)]
#    masses = np.sort(list(masses))
#    maxmass = masses[intensities.argmax()]
#    #tree = spatial.KDTree(masses)
#    #overlaps = generic_meta_overlap(tree.query_ball_point(masses, r=0.5).tolist())
#    csteps = masses - masses.min()
#    #maxstep = np.ceil(csteps.max() / expdiff) + 1 #meh speed boost, not really
#    maxstep = masses.size
#    steprange = proton * np.arange(maxstep)
#    stepclasses = np.abs(csteps - steprange[:,None]).argmin(axis=0)
#    cinds, counts = np.unique(stepclasses, return_counts=True)
#    csplit = counts.cumsum().tolist()
#    #stepsplit = np.split(masses, csplit)
#    stepsplit = []
#    ci = 0
#    for cs in csplit:
#        stepsplit.append(masses.tolist()[ci:cs])
#        ci = cs
#    stepints = [[v[j] for j in i] for i in stepsplit]
#    shapekey = maxmass in stepsplit[0]
#    orderkey = len(stepsplit)
#    stepratio = len(masses) / len(stepsplit)
#    stepcounter[shapekey][orderkey][stepratio] += 1
#    maxsubiso = 0 #subiso mass identity
#    subisomax = 0 #max subiso intensity
#    subisomaxstep = 0
#    maxmajorintensity = 0
#    maxmajorstep = 0
#    majormasses = []
#    summedintensities = []
#    for n, st in enumerate(stepsplit):
#        #isocountsbystep[shapekey][orderkey][n] += len(st) - 1
#        ints = [v[i] for i in st]
#        istmax = max(ints)
#        stmax = st[ints.index(max(ints))]
#        majormasses.append(stmax)
#        summedintensities.append(istmax)
#        if istmax > maxmajorintensity:
#            maxmajorstep = n
#            maxmajorintensity = istmax
#        for ss in st:
#            if ss != stmax:
#                if v[ss] > subisomax:
#                    subisomax = v[ss]
#                    maxsubiso = ss
#                    subisomaxstep = n
#    summedintensities = [sum(i) for i in stepints]
#    maxmajorstep = summedintensities.index(max(summedintensities))
#    summedintensities = np.array(summedintensities)
#    msums = summedintensities[:-1] + summedintensities[1:]
#    diffprej = np.diff(summedintensities)
#    maxdiffloc = np.abs(diffprej).argmax()
#    if diffprej[maxdiffloc] > 0:
#        greatestdifference[shapekey][orderkey][1] += 1
#        examples.append(k)
#    else:
#        greatestdifference[shapekey][orderkey][-1] += 1
#    majordiffs = np.diff(summedintensities) / msums / 2
#    negwhere = np.where(majordiffs < 0)[0]
#    negmajors = np.abs(majordiffs[negwhere])
#    maxnegloc = negwhere[negmajors.argmax()] + 1
#    greatestdecreaseloc[shapekey][orderkey][maxnegloc-maxmajorstep] += 1
#    maxneg = negmajors.max()
#    if maxneg > greatestdecrease[shapekey][orderkey]:
#        greatestdecrease[shapekey][orderkey] = maxneg
#    poswhere = np.where(majordiffs > 0)[0]
#    if poswhere.size > 0:
#        posmajors = majordiffs[poswhere]
#        maxposloc = poswhere[posmajors.argmax()] + 1
#        greatestincreaseloc[shapekey][orderkey][maxposloc-maxmajorstep] += 1
#        maxpos = posmajors.max()
#        if maxpos > greatestincrease[shapekey][orderkey]:
#            greatestincrease[shapekey][orderkey] = maxpos
#    for n, st in enumerate(stepsplit):
#        isocountsbystep[shapekey][orderkey][n-maxmajorstep] += len(st) - 1
#    if subisomax > 0:
#        stepdiff = subisomaxstep - maxmajorstep
#        highestsubisostepdiffs[shapekey][orderkey][stepdiff] += 1
#    maxlen = max(len(i) for i in stepsplit)
#    if maxlen > 1:
#        maxlenpoints = [n for n, i in enumerate(stepsplit) if len(i) == maxlen]
#        for mlp in maxlenpoints:
#            subisoaccumulationsteps[shapekey][orderkey][mlp] += 1
#    decreasing = False
#    incount = 0
#    for ii in range(len(summedintensities)-1):
#        i1 = summedintensities[ii]
#        i2 = summedintensities[ii+1]
#        if i2 > i1:
#            if decreasing:
#                #newinc = i2 / i1
#                newinc = abs(i2 - i1) / (i2 + i1) / 2
#                newincreaselocs[shapekey][orderkey][ii+1-maxmajorstep] += 1
#                if newinc > newincreases[shapekey][orderkey]:
#                    newincreases[shapekey][orderkey] = newinc
#                incount += 1
#        if i1 > i2:
#            decreasing = True
#    incounting[shapekey][orderkey][incount] += 1
##~



#I also want to make shapekey: distlen: steppattern (as: [0, 1, 2, 3] where the index is the step and the value is the number of isotopomers): count
#takeaway - subisotopomers never accompany the lowest mass, like ever
#subisopattern = defaultdict(lambda: defaultdict(Counter))
#proton = 1.00727647
#for k, v in abundances.items():
#    masses, intensities = zip(*v.items())
#    masses = np.sort(list(masses))
#    intensities = np.array(list(intensities))
#    maxmass = masses[intensities.argmax()]
#    #tree = spatial.KDTree(masses)
#    #overlaps = generic_meta_overlap(tree.query_ball_point(masses, r=0.5).tolist())
#    csteps = masses - masses.min()
#    #maxstep = np.ceil(csteps.max() / expdiff) + 1 #meh speed boost, not really
#    maxstep = masses.size
#    steprange = proton * np.arange(maxstep)
#    stepclasses = np.abs(csteps - steprange[:,None]).argmin(axis=0)
#    cinds, counts = np.unique(stepclasses, return_counts=True)
#    csplit = counts.cumsum().tolist()
#    #stepsplit = np.split(masses, csplit)
#    stepsplit = []
#    ci = 0
#    for cs in csplit:
#        stepsplit.append(masses.tolist()[ci:cs])
#        ci = cs
#    shapekey = maxmass in stepsplit[0]
#    orderkey = len(stepsplit)
#    isopattern = tuple(len(i)-1 for i in stepsplit)
#    subisopattern[shapekey][orderkey][isopattern] += 1






#singles = np.sort([k for k, v in seqsbymass.items() if len(v) == 1])
#multiples = np.sort([k for k, v in seqsbymass.items() if len(v) > 1])

#plt.hist(multiples, bins=1000, alpha=0.2)
#plt.hist(singles, bins=1000, alpha=0.2)
#plt.show()


#this used the old sequence keys for abundances
#basically everything that shares a primary(monoisotopic as well then) mass has the exact same distribution, so I'll organize isotopeabundances by mass to save room, then have a mass lookup in seqsbymass, this would ultimately be a more useful format for the end-game search process
#for k, v in seqsbymass.items():
#    if len(v) > 1:
#        test = [abundances[i] for i in v]
#        if not np.equal.reduce(test):
#            print(k, v)
#            break





#the seqsbymass bit assumes that peptides with the same monoisotopic mass have the same isotopic distribution. Which is, from what I've seen, correct. This would be because they have the same elemental composition despite having differing AA sequences.

#abundances = {}
#seqsbymass = defaultdict(list)
#nt = time()
#for seq in cutseqs:
#    m, s, a = distribution_generation(seq, samplesize, majorisotopemasses, massadditions, aminoacidcomposition)
#    abundances[s] = a
#    seqsbymass[m].append(s)
#print(time() - nt)


        #abundanceprobs = Counter(abundanceprobs)
        
        
        #tbh this doesn't show much, the percentages don't exactly go through massive changes. The changes are extremely miniscule.
        #df = pd.DataFrame.from_dict(abundancedistribution, orient='index', columns=[max(samplesizes)])
        #df.index.name = seq
        #
        #samplesizes = np.geomspace(1,100000000, 10).tolist()
        #smax = max(samplesizes)
        #for ss in reversed(samplesizes):
        #    inds = df.loc[:,smax] * ss > 1
        #    df.loc[inds, ss] = df.loc[inds, smax] / df.loc[inds, smax].sum()
        #df.dropna(axis=1, how='all', inplace=True)
        
        
        #fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,4))

        #ax[0].bar(abundancedistribution.keys(), abundancedistribution.values(), alpha=0.2)
        #plt.show()


#resampling below

        #masses = np.zeros(shape=samplesize)
        #for e, count in atomiccomposition.items():
        #    samples = list(isotopes[e].keys())
        #    weights = list(isotopes[e].values()) 
        #    adder = np.random.choice(samples, size=samplesize*count, p=weights).reshape(-1,count).sum(axis=1)
        #    masses += adder

        #dist = Counter(masses.tolist())

        #ndist = {}
        #for m, p in dist.items():
        #    ndist[m] = p / samplesize
        #ndist = Counter(ndist)

        #ax[1].bar(ndist.keys(), ndist.values(), alpha=0.2)
        #fig.suptitle(seq)
        #plt.show()


#notes

#once you have all the isotope distributions of peptides in a proteome:
#see how many neighboring-mass peptides have similar looking isotope distributions.
#^you can allow a range of +/- 0.1 daltons or something, then cross-correlate everything inside these bins, allow redundant bins at a 0.05 dalton overlap.
#then plot the distribution of correlations -> tighten the dalton range and make the overlap always half of it.
#^ perhaps instead of correlation, something else might work here. And this would be a good environment to DEVELOP a metric/type of measurement for you to use on the isotope-matching component of this whole shin-dig.
#I bet it'd be a sick notebook addition too, to show a nice metric thing or whatever that can distinguish close mass matches really well. After all - that's what you'd want it to do!

#this relies on mass accumulation of differing isotopes leading to unique masses -> which they do! (I'm pretty sure, not that I've checked lmao)
#would you be able to estimate molar amount by accurately quantifying true sample size of isotopic ratios from mass spectra?
