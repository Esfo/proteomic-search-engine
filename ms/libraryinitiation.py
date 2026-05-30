from collections import Counter, defaultdict
import numpy as np
from scipy import special, spatial, stats
from Bio import SeqIO
import sys
import re
import itertools
import string
import concurrent.futures
import multiprocessing as mp
import pandas as pd
import pickle
from time import time
import sqlitedict as sq
import lmdb
from functools import partial
import os
import gc
gc.enable()

#database:
#organism.full (db name) -> formula: [seqs]
#organism.linkers
#organism.sum
#organism.max

#central isotope database -> formula: [distributions]
#internal db table -> db name: samplesize, isofactors, enzyme, etc, I can log these variables when they're used
#amino acid compositions -> dbname: composition dict

#central fragment database portion, divided up as n/c-terms:
#fragmentsbyseq -> seq: [n/c-term fragments]
#breakagesbyfragseq -> fragseq: [n/c-term frag formulas]

#desirable changes:
#proteomes should contain lists of proteomes
#tablename='proteomes', proteome: [formulas]
#only a specific seqsbyformula for each proteome
#one large {formula: distribution} library should exist
#this way when you generate existing sequences you can check the central library insead of moving around
#^for both fragmentlibrarygeneration and libraryaddition (when adding a new proteome)
#^so then what do i do about sum and max distributions because introducing that mechanism to both library-gen processes means i won't be making max dists for everything i guess


#proteomeformulas: proteome: [formulas]


#this means the whole database will basically be centered around its samplesize, this will gauge the depth of quantitative insight
#varying samplesizes would not make a compatible library, so i should separate the functionality of library creation + samplesize input and fasta file parsing and banking

#initiation parameters:
#samplesize
#map_size increment, modifiable, 2gb default
#max_dbs too, default to a high multiple of the number of databases to be made + defaults, and advice not to lower it without knowing what they're doing, link to lmb docs if anyone wants to know what they're doing


#fasta parameters:
#min/max digest length - can be different but will default to library settings
#nprocs - ^same

#dbname = 'library.sqlite'
librarylocation = '/home/sfo/data/proteomics/fastas/search-db/'

if os.path.isdir(librarylocation):
    if not librarylocation.endswith('/'):
        librarylocation = ''.join((librarylocation, '/'))
    files = os.listdir(librarylocation)
    if 'data.mdb' in files:
        datafile = ''.join((librarylocation, 'data.mdb'))
        os.remove(datafile)
    if 'lock.mdb' in files:
        lockfile = ''.join((librarylocation, 'lock.mdb'))
        os.remove(lockfile)


samplesize = 50 #minimum percentage is 1 / this
mapsize = 1024**3*2

#this will be the only one where i designate map_size, you can read it later on without this input
environment_partial = partial(lmdb.Environment, map_size=mapsize, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=99999, max_spare_txns=1, lock=True)

#cut = [{'K':1, 'R':1}, {'K':'P', 'R':'P'}] #trypsin
#cut = [{'K':1, 'R':1}, {'K':'', 'R':''}] #trypsin + lys-c
#missedcleavages = 1
#minlength = 6
#maxlength = 50
#maxvmods = 3 #max combination of variable modifications to be applied to any peptide

#if not librarylocation.endswith('/'):
#    librarylocation = ''.join((librarylocation, '/'))

#foldername = ''.join((librarylocation, proteomefile.split('/')[-1].split('_')[0].lower(), '_isotopes-', str(minlength), '-', str(maxlength), '_miss-', str(missedcleavages), '_ss', str(samplesize)))

#if not os.path.isdir(librarylocation):
#    os.mkdir(librarylocation)

#libraryname = ''.join((librarylocation, dbname))

#with sq.SqliteDict(libraryname, tablename='defaults', flag='n', autocommit=True) as db:
#    #db['enzyme'] = cut
#    #db['maxvmods'] = maxvmods
#    #db['minlength'] = minlength
#    #db['maxlength'] = maxlength
#    db['samplesize'] = samplesize
#    #db['missedcleavages'] = missedcleavages
with environment_partial(librarylocation) as env:
    defaults = env.open_db('defaults'.encode())
    with env.begin(write=True) as txn:
        with txn.cursor(defaults) as cursor:
            cursor.put('samplesize'.encode(), str(samplesize).encode())
            cursor.put('mapsize'.encode(), str(mapsize).encode())
