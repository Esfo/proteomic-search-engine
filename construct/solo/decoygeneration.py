from database import environment

from collections import Counter, defaultdict
from random import shuffle
from math import factorial
from time import time
import pickle
import lmdb

def shuffle_string(s):
    char_list = list(s)
    shuffle(char_list)
    return ''.join(char_list)

def unique_permutations_count(sequence):
    #count the frequency of each element in the sequence
    freq = Counter(sequence)

    #calculate the total number of permutations
    total_permutations = factorial(len(sequence))

    #divide by the factorial of the frequency of each element to account for repetitions
    for count in freq.values():
        total_permutations //= factorial(count)

    return total_permutations

def decoy_generation(processingdirectory, librarylocation, proteome):
    
    linepositionsbyformulafile = ''.join((processingdirectory, 'linepositionsbyformula.pickle'))
    with open(linepositionsbyformulafile, 'rb') as pick:
        linepositionsbyformula = pickle.load(pick)
    #linepositionsbyformula = defaultdict(lambda: defaultdict(set)) #formula: matchable library positions of a distribution: [lines that have ms2 scans]
    
    seqsbyformula = {} #formula: [seqs]
    with environment(librarylocation) as env:
        seqdb = '.'.join(('seqsbyformula', proteome))
        seqs = env.open_db(seqdb.encode())
        with env.begin(write=False) as txn:
            with txn.cursor(seqs) as cursor:
                for k, v in cursor:
                    key = k.decode()
                    if key in linepositionsbyformula:
                        #minimal decoy generation necessary
                        value = eval(v.decode())
                        seqsbyformula[key] = value
    
    nt = time()
    
    formulabysortedseq = {} #sortedseq: formula
    seqsbysortedseq = defaultdict(set) #sortedseq: [seqs]
    for formula, seqs in seqsbyformula.items():
        for seq in seqs:
            sortedseq = ''.join((sorted(seq)))
            seqsbysortedseq[sortedseq].add(seq)
            formulabysortedseq[sortedseq] = formula
    
    #for now this only handles trytic peptides
    #but i should note if c or n terminal AAs are relevant to the digest and handle it here
    #make them kwargs and default them to false, it should either be a [0] or [-1] slice, if they exist then slice those out first via case/match
    fulldecoyset = set() #all decoy sequences
    seqswithdecoysbyformula = defaultdict(list) #formula: [seqs + decoys]
    for sortedseq, seqs in seqsbysortedseq.items():
        decoys = set()
        slen = len(seqs)
        #make seqgroups based on first/last AA depending on enzyme
        seqgroups = defaultdict(lambda: defaultdict(list)) #position (0 or -1): AA: [seqs]
        for seq in seqs:
            #this would need to be:
            #if seq.startswith/endswith and make double groups for when both AAs apply
            seqgroups[-1][seq[-1]].append(seq)
        for position, aas in seqgroups.items():
            for aa, subseqs in aas.items():
                #if len(aa) > 1: double-group i suppose?
                initialseq = subseqs[0][:-1]
                setlen = len(set(initialseq))
                if setlen > 1:
                    subdecoys = set()
                    sublen = len(subseqs)
                    permax = unique_permutations_count(initialseq) #the -1 is considering K or R ending is consistent
                    for seq in subseqs:
                        #tryptic only atm
                        endchar = seq[-1]
                        shortseq = seq[:-1]
                        while True:
                            decoy = shuffle_string(shortseq) + endchar
                            #decoy = shortseq[::-1] + endchar
                            if decoy not in subdecoys and decoy not in decoys and decoy not in seqs:
                                subdecoys.add(decoy)
                                break
                            #else: #for reversing
                            #    break
                            if len(subdecoys) + sublen == permax:
                                #all potential sequences already made
                                break
                    decoys.update(subdecoys)
                #else: #setlen == 1 and sublen == 1
                    #the sequence only has one AA, no decoys possible, whatever
                    #break
        seqs.update(decoys)
        fulldecoyset.update(decoys)
        seqswithdecoysbyformula[formulabysortedseq[sortedseq]].extend(seqs.copy())
    
    seqswithdecoysbyformula = dict(seqswithdecoysbyformula)
    print(time() - nt, 'decoys generated')
    
    seqswithdecoysbyformulafile = ''.join((processingdirectory, 'seqswithdecoysbyformula.pickle'))
    with open(seqswithdecoysbyformulafile, 'wb') as pick:
        pickle.dump(seqswithdecoysbyformula, pick)
    #seqswithdecoysbyformula = defaultdict(list) #formula: [seqs + decoys]
    
    fulldecoysetfile = ''.join((processingdirectory, 'fulldecoyset.pickle'))
    with open(fulldecoysetfile, 'wb') as pick:
        pickle.dump(fulldecoyset, pick)
    #fulldecoyset = set() #all decoy sequences
