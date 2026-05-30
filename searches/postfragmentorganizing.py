import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from colour import Color
import psutil
import asyncio
import aiofiles
from pyteomics import mzml
import csv
import bisect
import heapq
import fcntl #this will need to be portalocker on other operating systems
from time import time
import pandas as pd
import gc
import concurrent.futures
import multiprocessing as mp
from collections import Counter, defaultdict
from scipy import sparse, integrate, spatial, stats, special
from pandas.api.types import CategoricalDtype
from sklearn.neighbors import NearestNeighbors
from more_itertools import sort_together
from distinctipy import distinctipy as dp
from functools import partial
from pickleshare import PickleShareDB
import math
import zlib
import lmdb
import random
import itertools
import string
import pickle
import sys
import os
import warnings
#warnings.filterwarnings("error")
np.set_printoptions(suppress=True)
gc.enable()

#for more https://matplotlib.org/stable/tutorials/introductory/customizing.html
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
chexes = ['#ffffff',
        '#8ff6ff',
        '#ff9f9c',
        '#2ded8d',
        '#fbffb3',
        '#ea68f2',
        '#7d26ff',
        ]
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', chexes)
#n = 0
#for c in chexes:
#    plt.plot([n, n+1], [n, n+1], color=c, label=c)
#    n += 1
#plt.legend()
#plt.show()

mzmlfile = '/home/sfo/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
#mzmlfile = '/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
basefolder = '/'.join((mzmlfile.split('/')[:-2]))
basefile = mzmlfile.split('/')[-1].split('.mzML')[0]

processinglocation = '/'.join((basefolder, 'fileprocessing', basefile))
fragmentlocation = '/'.join((basefolder, 'fileprocessing', basefile, 'fragments'))
scanalytelocation = '/'.join((basefolder, 'fileprocessing', basefile, 'scanalytegroups'))
librarylocation = '/home/sfo/data/proteomics/fastas/search-db'
csvfilename = '/'.join((processinglocation, 'fragment.matches'))
proteome = 'Human_Homo_sapien'
nprocs = 8
proton = 1.007276554940804
dividingthreshold = 0.8
ppmtol = 25
ppmmod = ppmtol / 1000000

def intersection_merge(mergable_items):
    sn = 0
    itemgroups = defaultdict(set) #group: [members]
    groupsofitems = {} #member: group
    for items in mergable_items:
        locs = set()
        for i in items:
            if i in groupsofitems:
                locs.add(groupsofitems[i])
        if locs:
            joiner = min(locs)
            if len(locs) > 1:
                for oldlocs in locs.difference([joiner]):
                    for ol in itemgroups[oldlocs]:
                        groupsofitems[ol] = joiner
                    itemgroups[joiner].update(itemgroups.pop(oldlocs))
        else:
            joiner = sn
            sn += 1
        itemgroups[joiner].update(items)
        for i in items:
            groupsofitems[i] = joiner
    return list(itemgroups.values())

scanalytefile = '/'.join((processinglocation, 'scanalytes.pickle'))
with open(scanalytefile, 'rb') as pick:
    scanalytecharges = pickle.load(pick)
#scanalytecharges = defaultdict(dict) #analyteid: scan: charge

#i dont need this yet, right?
submatchsubformulasfile = '/'.join((processinglocation, 'submatchsubformulas.pickle'))
with open(submatchsubformulasfile, 'rb') as pick:
    submatchsubformulas = pickle.load(pick)
#submatchsubformulas = {} #submatchindex: subformula

#scanmassesfile = '/'.join((processinglocation, 'scanmasses.pickle'))
#with open(scanmassesfile, 'rb') as pick:
#    scanmasses = pickle.load(pick)
##scanmasses = {} #scan: [[masses], [intensities]]

analytefile = '/'.join((processinglocation, 'analytefactors.pickle'))
with open(analytefile, 'rb') as pick:
    analytesbydistribution, distributionsoflines = pickle.load(pick)[2:4]
#analytesbydistribution = {} #distid: analyte id
#distributionsoflines = {} #lineid: distid

subformularankfile = '/'.join((processinglocation, 'subformularank.pickle'))
with open(subformularankfile, 'rb') as pick:
    subformularank = pickle.load(pick)
#subformularank = defaultdict(dict) #sequence: subformula: descending subiso rank, lower int = more relevant subiso

environment_partial = partial(lmdb.Environment, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=9999, max_spare_txns=1, lock=True)

#seqsbyformula = {} #formula: [seqs]
formulasbyseq = {}
with environment_partial(librarylocation) as env:
    seqdb = '.'.join(('seqsbyformula', proteome))
    seqs = env.open_db(seqdb.encode())
    with env.begin(write=False) as txn:
        with txn.cursor(seqs) as cursor:
            for k, v in cursor:
                key = k.decode()
                value = eval(v.decode())
                #seqsbyformula[key] = value
                for subval in value:
                    formulasbyseq[subval] = key

#using scanalytecharges -> scan: analyte: charge
# - intersection merge by analyteid and scan
#   > use a scanalyte index, like what you seem to have in postfragprocessing, as analytes and scans are both just integers
# - make analysisgroups -> analyteid: analysisgroup
csvindex = 0
scanalyteindex = 0
#these new indices work as placeholders/trackers for logistical purposes
analyteidbyscanalyteindex = {} #scanalyte index: analyteid
scanbyscanalyteindex = {} #scanalyte index: scan
scanalyteindexbyanalyteid = {} #analyteid: scanalyte index
scanalyteindexbyscan = {} #scan: scanalyte index
mergableindices = set() #starts off as pairs of the two above newindices
for analyteid, scans in scanalytecharges.items():
    scanalytelist = []
    if analyteid in scanalyteindexbyanalyteid:
        oldindex = scanalyteindexbyanalyteid[analyteid]
        scanalytelist.append(oldindex)
    else:
        scanalyteindexbyanalyteid[analyteid] = scanalyteindex
        analyteidbyscanalyteindex[scanalyteindex] = analyteid
        scanalytelist.append(scanalyteindex)
        scanalyteindex += 1
    for scan in scans:
        if scan in scanalyteindexbyscan:
            oldindex = scanalyteindexbyscan[scan]
            scanalytelist.append(oldindex)
        else:
            scanalyteindexbyscan[scan] = scanalyteindex
            scanbyscanalyteindex[scanalyteindex] = scan
            scanalytelist.append(scanalyteindex)
            scanalyteindex += 1
    mergableindices.add(tuple(scanalytelist))

scanalytegroups = list(map(tuple, intersection_merge(mergableindices)))
print(len(scanalytegroups), 'scanalyte groups')

#scanalytegroupbyanalyteid = {} #analyteid: scanalytegroup
scanalytegroupbyscan = {} #scan: scanalytegroup
for n, scanalytes in enumerate(scanalytegroups):
    #analyteids = [analyteidbyscanalyteindex[i] for i in scanalytes if i in analyteidbyscanalyteindex]
    scans = [scanbyscanalyteindex[i] for i in scanalytes if i in scanbyscanalyteindex]
    #for analyteid in analyteids:
    #    scanalytegroupbyanalyteid[analyteid] = n
    for scan in scans:
        scanalytegroupbyscan[scan] = n
    sgfilename = 'scanalytes.' + str(n) + '.csv'
    sgfileloc = '/'.join((scanalytelocation, sgfilename))
    with open(sgfileloc, 'w') as sgfn:
        #writing an empty file, they'll be appended to later and this will clear any that's in here already
        pass

#within a scan, each frag dist sould have ascending rank orders
#you'll need to pull up
#the actual ranks also need to consider overlapping frag isos of the same sequence distorting the orders
#so EVERY fragment match to specific ions can first be checked
#then maybe you actually want to make sense of the orders by determining WHICH frag dists give you that order...

#regarding break clusters
#if multiple of the recognizable patterns come from competing frag patterns within the same scan/spectra, does this make sense? should there be some kind of non-redundancy aspect to the initial discovery of these?

#breaks:
#[L|R combo as a string: "AA", fragment ion ie "y", ]

#number of matches
#totality of frag dists matched -> sum theoretical abundance?
#% intensity covered relative to MS1 expectation
#redundancy among non-competing sequences
#differences among competing sequences
#average ppm error/differences
#percent of MS2 spectra normalized by its total intensity -> confidence of an ID?
#should there be something for the consistency of distributions/charges that are visible across different scans of the same isotopomer? i might be able to do a better scan-by-scan analysis of the different fragmentation patterns of different isotopomers here than what i was thinking of doing prior just based off of raw ms2 signals

#something annoying:
#DOES this ascending fragrank actually hurt?
#why would it?
#because what if the fragment isotopomers you see are not distributed based on the multinomial probability assumed by equal fragmentation independent of isotopomer location
#WHAT IF the isotopes play a bigger role in fragmentation than they play in abundance distributions?
#in that case the distributions won't be as you expect them based on isotopomers
#and the nature of the multinomial is dependent on 2 things instead of just 1 (being which isotopomers are where are the time of fragmentation)
#and NOW you have an unpredictable layer that determines fragment distribution abundance and your rank-filtering process is actually fraudulent
#so how can i test this?
#i need to see if i can find a pattern, much like i was planning to with break clusters, of whether specific isotopes or compositions tend to be present as matches compared to the other fragment being formed?
#i'll only need to look at whether fragments have different individual isotopomers more often than their other fragments do -> and i'll need to determine whether the opposing fragment is actually identified within that spectra too
#and really this actually isn't a question of fragmentation: its a question of -- does one side of the break tend to get the charge more often than the other? and IF SO, then that might affect the fragment distributions visibility of either fragments distribution WHEN it has those isotopomers -> SO the distributions will be biased by the CHARGE depending on whether they have any specific isotopomer
#^ and if true, THIS will affect the abundance patterns of fragment distributions
#i'll need to be able to be able to generate the opposite fragment using formulasbyseq + the ion itself
#i can even go file-by-file with this no problem because each file represents a somewhat different subiso group
#BUT, realistically, i have experience with this already
#i know from PRM experiments i've done that the fragments come out the same across isotopically labeled samples
#so i don't need to worry about whether the placement of an isotope affects anything because the SILs had consistent labeling
#i'm moving forward with the fragrank filtering

#csv run:
# - make new csvs with just the groups involved?
# - but the new csvs have full frag dist matches
# - exclude non-descending rank frag dists
#   ^> this is fine because the fragment generation already solves the overlapping mass position issue this would have, so there's no theoretical masses that have the same proton location
#   > worry about whether the experimental data matches the frag dist order later

#upcoming process:
#scanalytegroups folder
#one csv per group
#group matches by seq + ion -> ascending frag dist rank requirement, keep the top at least?
#get ppm error metric per dist match
#get dist matches per seq
#break clusters

#def opposing_fragment_generation(subformula, fragformula):
#    fragcomp = {}
#    for ss in fragformula.split(')')[:-1]:
#        e, c = ss.split('(')
#        fragcomp[e] = c
#    opposingcomp = {}
#    for ss in subformula.split(')')[:-1]:
#        e, c = ss.split('(')
#        if e in fragcomp:
#            opcount = int(c) - int(fragcomp[e])
#            if opcount > 0:
#                opposingcomp[e] = str(opcount)
#        else:
#            opposingcomp[e] = c
#    return ''.join((''.join((e, '(', opposingcomp[e], ')')) for e in sorted(opposingcomp)))

#next process:
#get total ms2 ion intensity per seq

headers = ['sequence', 'fragformula', 'ion', 'submatchindex', 'theoretical_abundance', 'frag_dist_rank', 'lines', 'scan', 'theoretical_mass', 'ppm_error', 'mass_index_of_scan', 'charge']

nt = time()

files = [i for i in os.listdir(fragmentlocation) if i.endswith('.matches.csv')]

dmeasures = Counter()
dmaxes = Counter()
#linesubcounts = defaultdict(lambda: defaultdict(set)) #seq: line: [subformulas]

#sanity check test for making sure all subformulas of a line from a seq are in the same file: YES they are
#organizationchecker = defaultdict(lambda: defaultdict(lambda: defaultdict(set))) #seq: line: subformula: [files]

#there is a uniform distribution of ppm errors across all files
for f in files:
    ft = time()
    filename = '/'.join((fragmentlocation, f))
    #fragmentorganizer = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))))) #scan: analyteid: seq: line: subformula (as submatchindex): ion: charge: fragrank: ppm error
    fragmentorganizer = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))))) #scan: line: seq: subformula (as submatchindex): ion: charge: fragrank: [ppm error, ...]
    #^this works lmao
    with open(filename, 'r') as w:
        for i in w.readlines():
            row = i.split(',')
            seq = row[0]
            fragformula = row[1]
            ion = row[2]
            submatchindex = int(row[3])
            subformula = submatchsubformulas[submatchindex]
            thabundance = float(row[4])
            fragrank = int(row[5])
            scan = int(row[7])
            thmass = float(row[8])
            ppmerror = float(row[9])
            scanindex = int(row[10])
            charge = int(row[11].strip())
            #exmass, exintensity = scanmasses[scan][:,scanindex]
            if '-' in row[6]:
                lines = [int(i) for i in row[6].split('-')]
                for line in lines:
                    #analyteid = analytesbydistribution[distributionsoflines[line]]
                    fragmentorganizer[scan][line][seq][subformula][ion][charge][fragrank].append([thmass, ppmerror, thabundance, scanindex, fragformula])
                    #linesubcounts[seq][line].add(submatchsubformulas[submatchindex])
                    #fragmentorganizer[scan][analyteid][seq][line][submatchsubformulas[submatchindex]][ion][fragrank][charge].append([thmass, ppmerror, thabundance, scanindex, fragformula, submatchindex])
            else:
                line = int(row[6])
                #analyteid = analytesbydistribution[distributionsoflines[line]]
                #((experimental - theoretical) / experimental) / 1000000
                fragmentorganizer[scan][line][seq][subformula][ion][charge][fragrank].append([thmass, ppmerror, thabundance, scanindex, fragformula])
                #linesubcounts[seq][line].add(submatchsubformulas[submatchindex])
                #fragmentorganizer[scan][analyteid][seq][line][submatchsubformulas[submatchindex]][ion][fragrank][charge].append([thmass, ppmerror, thabundance, scanindex, fragformula, submatchindex])
            #analyteid = row[6]
            #if '-' in analyteid:
            #    analyteids = [int(i) for i in analyteid.split('-')]
            #    for analyteid in analyteids:
            #else:
            #    analyteid = int(analyteid)
            #b.write(i)
        #os.remove(filename)
    #for scan, analyteids in fragmentorganizer.items():
    #    alen = len(analyteids)
    #    dmeasures['analyteids'] += alen
    #    if alen > dmaxes['analyteids']:
    #        dmaxes['analyteids'] = alen
    #    for analyteid, seqs in analyteids.items():
    #        slen = len(seqs)
    #        dmeasures['seqs'] += slen
    #        if slen > dmaxes['seqs']:
    #            dmaxes['seqs'] = slen
    #        for seq, lines in seqs.items():
    #            llen = len(lines)
    #            dmeasures['lines'] += llen
    #            if llen > dmaxes['lines']:
    #                dmaxes['lines'] = llen
    #            for line, subformulas in lines.items():
    #                sublen = len(subformulas)
    #                dmeasures['subformula'] += sublen
    #                if sublen > dmaxes['subformula']:
    #                    dmaxes['subformula'] = sublen
    #                for subformula, ions in subformulas.items():
    #                    ilen = len(ions)
    #                    dmeasures['ions'] += ilen
    #                    if ilen > dmaxes['ions']:
    #                        dmaxes['ions'] = ilen
    #                    for ion, fragranks in ions.items():
    #                        flen = len(fragranks)
    #                        dmeasures['fragranks'] += flen
    #                        if flen > dmaxes['fragranks']:
    #                            dmaxes['fragranks'] = flen
    #                        for fragrank, charges in fragranks.items():
    #                            clen = len(charges)
    #                            dmeasures['charges'] += clen
    #                            if clen > dmaxes['charges']:
    #                                dmaxes['charges'] = clen
    #                            for charge, matches in charges.items():
    #                                mlen = len(matches)
    #                                dmeasures['matches'] += mlen
    #                                if mlen > dmaxes['matches']:
    #                                    dmaxes['matches'] = mlen
    distmatches = Counter()
    chargematches = Counter()
    ionmatches = Counter()
    subformulamatches = Counter()
    seqmatches = Counter()
    linematches = Counter()
    scanmatch = 0
    failscanmatch = 0
    faildistmatch = 0
    #[scan, line, seq, subformula, ion, charge, absolute avg error, average distance of errors to themselves, n frag isos]
    ascendingmatches = defaultdict(list) #scanalytegroup: [match info]
    #this process doesn't face any problems with the memory-handling subformulagrouping changes because the fragment isotopomers should always be generated together so they'll be saved together
    for scan, lines in fragmentorganizer.items():
        linematch = 0
        scanalytegroup = scanalytegroupbyscan[scan]
        for line, seqs in lines.items():
            seqmatch = 0
            for seq, subformulas in seqs.items():
                #filtering non-ascending ranked subformula matches
                subformulalist = sorted((subformularank[seq][i], i) for i in subformulas)
                count = 0
                selectedsubformulas = []
                for rank, subformula in subformulalist:
                    if rank == count:
                        selectedsubformulas.append(subformula)
                        count += 1
                    else:
                        break
                subformulamatch = 0
                for subformula in selectedsubformulas:
                    ions = subformulas[subformula]
                    #organizationchecker[seq][line][subformula].add(f)
                    ionmatch = 0
                    for ion, charges in ions.items():
                        distmatch = 0
                        for charge, fragranks in charges.items():
                            ranklist = sorted(fragranks)
                            if 0 in ranklist:
                                #fragranks work, summarize them as a distribution of this charge
                                count = 0
                                #for fragrank, ppm in fragranks.items():
                                for fragrank in ranklist:
                                    if fragrank == count:
                                        fraglist = fragranks[fragrank]
                                        newout = []
                                        for fl in fraglist:
                                            fl.extend([scan, line, seq, subformula, ion, charge, fragrank])
                                            outstring = ','.join((str(i) for i in fl)) + '\n'
                                            newout.append(outstring)
                                        #if len(fraglist) > 1:
                                            #it can happen, just not in this data at this ppm
                                        ascendingmatches[scanalytegroup].extend(newout)
                                        count += 1
                                    else:
                                        break
                                distmatches[count] += 1
                                distmatch += 1
                            else:
                                faildistmatch += 1
        #                    if len(ascendingmatches) > 3:
        #                        break
        #                if len(ascendingmatches) > 3:
        #                    break
        #            if len(ascendingmatches) > 3:
        #                break
        #        if len(ascendingmatches) > 3:
        #            break
        #    if len(ascendingmatches) > 3:
        #        break
        #if len(ascendingmatches) > 3:
        #    break
                        chargematches[distmatch] += 1
                        if distmatch > 0:
                            ionmatch += 1
                    ionmatches[ionmatch] += 1
                    if ionmatch > 0:
                        subformulamatch += 1
                subformulamatches[subformulamatch] += 1
                if subformulamatch > 0:
                    seqmatch += 1
            seqmatches[seqmatch] += 1
            if seqmatch > 0:
                linematch += 1
        linematches[linematch] += 1
        if linematch > 0:
            scanmatch += 1
        else:
            failscanmatch += 1
    print('~~~')
    print('distmatches')
    print(distmatches)
    print(faildistmatch, 'failures')
    print('chargematches')
    print(chargematches)
    print('ionmatches')
    print(ionmatches)
    print('subformulamatches')
    print(subformulamatches)
    print('seqmatches')
    print(seqmatches)
    print('linematches')
    print(linematches)
    print('scanmatches')
    print(scanmatch)
    print('scan misses')
    print(failscanmatch)
    for scanalytegroup, matches in ascendingmatches.items():
        sgfilename = 'scanalytes.' + str(scanalytegroup) + '.csv'
        sgfileloc = '/'.join((scanalytelocation, sgfilename))
        with open(sgfileloc, 'a') as sgfn:
            for m in matches:
                sgfn.write(m)
    print(time() - ft, f)
    #break

#dmeasures = Counter()
#dmaxes = Counter()
#for seq, lines in linesubcounts.items():
#    dmeasures['lines'] += len(lines)
#    if len(lines) > dmaxes['lines']:
#        dmaxes['lines'] = len(lines)
#    for line, subformulas in lines.items():
#        dmeasures['subformulas'] += len(subformulas)
#        if len(subformulas) > dmaxes['subformulas']:
#            dmaxes['subformulas'] = len(subformulas)

#newheaders = ['theoretical_mass', 'ppm_error', 'theoretical_abundance', 'scan_index', 'fragformula', 'scan', 'line', 'sequence', 'submatchindex', 'ion', 'charge', 'fragrank']

print(time() - nt, 'all csvs processed')

#just add all the metrics you can now
#even the abundance ranks matches (whatever this will be)
#what you NEED to add later is now many overlapping hits happen to specific scan indices -> how many are redundant vs competitive vs non-overlapping and whatnot
#i need to see that ppm consistency exists
#i suppose things compete over scan indices, but u haven't mentally conected this to seqs yet

#ranks will be organized OTF via heap
#why is it OTF? because i'm only going to include the top rank of any distribution in the initial list, and when that top rank gets skimmed off the top, then i average in the 2nd rank with the first rank, and push that into the heap
#^this should work out right with average errors
#^but will this work with matrix difference of errors aka consistency?
#^because the consistency of singles will be 0
#^nah no heaps, every iso has to be represented independently within any potential dist its a part of

#as the list goes down
#each sequence fills up on the total intensity of each ion
#^nah, combinatorics for the seq identifications, right?
#^first you can assess individual peptide matches with errors and consistency across all distributions it matches
#then combinatorically check their shortened individual assessments across all the non-competing seqs, to find optimal solutions
#then get all the combinatoric-solutions to compete against each other on the grander entire-file scale?
#^BUT, as for what to INCLUDE in the combinatorics... i guess i have to try every single member of every single dist to make every single kind of identification for each individual peptide, and THEN combinatorically check all of these against redundants?
#there might be some intra-seq redundancy that needs to be addressed this way

#seeing as multiple scan indices can match to the same fragmass and vice-versa, i ought to build in the redundancy needed for that in the combinatorics, i don't see it in this data but if you expanded the ppm window it would show up

#~

#so basically:
#make all potential dist id groups of any seq
#organize them in a csv as rows, with individual dist isos
#compile all seq identifications by scanalytegroup
#everything is organized into scanalyte csvs

#then:
#organize initial combinatorics by co-identified sequences among analyteids
#ie if there's 2 dists in a scan, organize combinatorics of every potential seq of both of them -> where they can't identify as the same seq, and this may be necessary to apply across charge states
#those 2 seqs (or more, including unidentifiable lines) would then undergo their combinatoric ID process against each other, as a co-identifying competition round to see what best fits the identification of that spectra, to minimize the ppm consistency and average errors and whatnot -> but as a scorable combination of IDs, rather than any individual IDs
#and if there's only one potential ID, this process is a little different but basically the same
#then those competitive combinations needs to be assessed across all relevant scans for the next OUTER layer combinations
#   > ie if scan 1 has lines 1 and 2, and scan 2 has lines 1 and 3, then all combinations of the two relevant scans based on the seqs of line 1 will need to be assessed
#   > ^but do i assess them for similarity at all? i would think they're similar enough to forego it, but i wonder if i need some kind of consistency check
#as all the scans in any group will basically be relevant -> but this will lead to massive combinatorics for large scanalytegroups

#[again] then:
#SO the initial combinations are made only inside individual scans, then the relevant intra-scan combinations scans combinations are linked across scans
#then you can asses the matches based on:
# - entropy IDs
# - MS1 intensity
#   > when a line has greater MS1 intensity, it should hold a superset of identified fragments from other scans, basically
# - ppm consistency + abs avg
# - fragment ion consistency across subisotopomers?

#and out of this, pops a ranking process
#should the ranks be done within scanalytegroups or within the entire file?
#i think ANOTHER kind of group should be made where any potential SEQUENCE redundancies are grouped together
#this will merge multiple scanalytegroups into individual sequencegroups
#and THEN starts the ranking



#BECAUSE isotopoes shouldn't affect fragmentation, when multiple lines of the same dist end up in the same scan, you should just see a +/- 1 accompanying dist, ~ish?
#not necessarily, it depends on the isotopes abundances


#~

#there may be room to compare rough stats on scans that miss vs scans with matches
#much like with the ms1 dist matches
#it may also be interesting to use a kind of sequence clustering to try and identify a bias from post-validation results of every other search engine you check to see if they bias different sequences/sequence clusters
#and if multiple engines have the same bias, it may be a datapoint of proteomics, but if they all have different biases, then it might be worthwhile trying to steer clear of these in your own engine
