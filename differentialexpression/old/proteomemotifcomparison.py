import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re
from Bio import SeqIO
import sys
sys.path.append('/home/sfo/camp')
#from sequencetools import seqsplit, seqsplit_dict
from collections import Counter
from time import time
import os
import itertools
import multiprocessing as mp
import concurrent
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

#trypsin = [{'K':1, 'R':1}, {'K':'P', 'R':'P'}]

dbfiles = {'/home/sfo/data/fastas/proteomes/Zebrafish_Danio_rerio.fasta': '/home/sfo/data/motifs/Zebrafish_Danio_rerio_distance50.csv',
        '/home/sfo/data/fastas/proteomes/Fly_Drosophila_melanogaster.fasta': '/home/sfo/data/motifs/Fly_Drosophila_melanogaster_distance50.csv'}

maxdistance = 50
#for 47000 proteins
#45 mins for max distance of 10
#3 hours for max distance of 30
#6.5 hours for max distance of 50
#20 hours for max distance of 100

aminoacids = 'ACDEFGHIKLMNPQRSTVWY'
aminoacids = [i for i in aminoacids]

aminopairs = tuple(itertools.combinations(aminoacids, 2))

for dbfile, outfile in dbfiles.items():

    db = SeqIO.parse(open(dbfile), 'fasta')

#maxn = 10
#n = 0
    dseqs = {}
    for fasta in db:
        sequence, idn = str(fasta.seq), fasta.id
        dseqs[idn] = sequence
#    n += 1
#    if n > maxn:
#        break

    lens = []
    for d in dseqs.values():
        lens.append(len(d))

#fig, ax = plt.subplots(ncols=2, figsize=(12,5))
#
#lens = sorted(lens)
#ax[0].plot(range(len(lens)), lens)
#
#lens = sorted(lens)
#ax[1].plot(range(len(lens)), lens)
#ax[1].set_yscale('log')
#
#plt.suptitle('Protein Length')
#plt.show()

    mt = time()

    out = pd.DataFrame(columns = ['Protein', 'Motif', 'Starting Position', 'Distance from C-Term', 'Protein Length'])
    out.to_csv(outfile, index=False)

#outies = []
    for k, p in dseqs.items():
        tout = []
        for l, r in aminopairs:
            for d in range(maxdistance +1):
                patternstring = r''.join((l, d*'.', r))
                search = re.compile(patternstring)
                matches = search.finditer(p)
                for m in matches:
                    outlist = [k, patternstring, m.start(), m.endpos - m.start(), m.endpos]
                    tout.append(outlist)
        df = pd.DataFrame(tout)
        df.to_csv(outfile, header=False, index=False, mode='a')

#def finder(k, p, maxdistance, aminopairs):
#    tout = []
#    #df = pd.DataFrame(columns = ['Protein', 'Motif', 'Starting Position', 'Distance from C-Term', 'Protein Length'])
#    #ind = 0
#    for l, r in aminopairs:
#        for d in range(maxdistance +1):
#            patternstring = r''.join((l, d*'.', r))
#            search = re.compile(patternstring)
#            matches = search.finditer(p)
#            for m in matches:
#                outlist = [k, patternstring, m.start(), m.endpos - m.start(), m.endpos]
#                tout.append(outlist)
#                #df.loc[ind] = outlist
#                #ind += 1
#    outies.extend(tout)
#    #df.to_csv(outfile, header=False, index=False, mode='a')
#    return

#def distancevar(k, p, maxdistance, l, r):
#    tout = []
#    for d in range(maxdistance +1):
#        patternstring = r''.join((l, d*'.', r))
#        search = re.compile(patternstring)
#        matches = search.finditer(p)
#        for m in matches:
#            outlist = [k, patternstring, m.start(), m.endpos - m.start(), m.endpos]
#            tout.append(outlist)
#    outies.extend(tout)

#def matcher(k, p, outies, d, l, r):
#    patternstring = r''.join((l, d*'.', r))
#    search = re.compile(patternstring)
#    matches = search.finditer(p)
#    for m in matches:
#        outlist = [k, patternstring, m.start(), m.endpos - m.start(), m.endpos]
#        outies.append(outlist)

#threads = []
#for k, p in dseqs.items():
#    t = th.Thread(target=finder, args=(k, p, maxdistance, aminopairs))
#    t.start()
#    threads.append(t)
#for thread in threads:
#    thread.join()

#maxproteins = 5

#with concurrent.futures.ThreadPoolExecutor(2) as executor:
#    count = 0
#    for k, p in dseqs.items():
#        executor.submit(finder, k, p, maxdistance, aminopairs)
#        count += 1
#        if count > maxproteins:
#            of = pd.DataFrame(outies)
#            of.to_csv(outfile, index=False, header=False, mode='a')
#            outies = []
#            count = 0

#count = 0
#for k, p in dseqs.items():
#    executor.submit(finder, k, p, maxdistance, aminopairs)
#    count += 1
#    if count > maxproteins:
#        of = pd.DataFrame(outies)
#        of.to_csv(outfile, index=False, header=False, mode='a')
#        outies = []
#        count = 0
#
#if outies:
#    of = pd.DataFrame(outies)
#    of.to_csv(outfile, index=False, header=False, mode='a')
#et = time() - mt
#print(et)
#print(47085/100*et/60/60, 'hours')

#out = pd.DataFrame(outies)
#out.columns = ['Protein', 'Motif', 'Starting Position', 'Distance from C-Term', 'Protein Length']
#out.loc[:,'Motif Length'] = out.loc[:,'Motif'].apply(lambda x: len(x))
#out.to_csv(outfile, index=False)
