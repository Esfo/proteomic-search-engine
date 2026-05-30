import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from colour import Color
from pyteomics import mzml
from time import time
import pandas as pd
import gc
import concurrent.futures
import multiprocessing as mp
from collections import Counter, defaultdict
from textwrap import wrap
from scipy import sparse, integrate, spatial, stats, special, signal
from pandas.api.types import CategoricalDtype
from sklearn.neighbors import NearestNeighbors
from more_itertools import sort_together
from distinctipy import distinctipy as dp
from functools import partial
import lmdb
import random
import itertools
import pickle
import sys
import os
import warnings
#warnings.filterwarnings("error")
np.set_printoptions(suppress=True)
#np.warnings.filterwarnings('ignore')
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

proton = 1.007276554940804
chargetolerance = 0.1
minpoints = 3
mzmlfile = '/home/sfo/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
#mzmlfile = '/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
nprocs = os.cpu_count()

basefolder = '/'.join((mzmlfile.split('/')[:-2]))
basefile = mzmlfile.split('/')[-1].split('.mzML')[0]

processinglocation = '/'.join((basefolder, 'fileprocessing', basefile))
librarylocation = '/home/sfo/data/proteomics/fastas/search-db'

proteomefile = '/home/sfo/data/proteomics/fastas/proteomes/Human_Homo_sapien.fasta'
proteome = proteomefile.split('/')[-1].split('.')[0]

environment_partial = partial(lmdb.Environment, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=99999, max_spare_txns=1, lock=True)

def coordinate_generation(scan):
    if scan['ms level'] == 2:
        precursorinfo = scan['precursorList']['precursor'][0]
        selectionwindow = precursorinfo['isolationWindow']
        precmass = selectionwindow['isolation window target m/z'].real
        lowerbound = precmass - selectionwindow['isolation window lower offset'].real
        upperbound = precmass + selectionwindow['isolation window upper offset'].real
        #trainind references index of newtrain, -> get mass -> input to trackedma -> lineuid
        scanlist = scan['scanList']['scan'][0]
        #windowbounds = scanlist['scanWindowList']['scanWindow'][0]
        #lwbound = windowbounds['scan window lower limit'].real
        #uwbound = windowbounds['scan window upper limit'].real
        rt = scanlist['scan start time'].real
        scindex = int(scan['index'])
        #bounddict = {scindex: [lwbound, uwbound]}
        #coordinates = [rt, precmass, lowerbound, upperbound, scindex]
        coordinates = [rt, lowerbound, upperbound, scindex]
        #return coordinates, bounddict
        return coordinates

nt = time()

#distributionmatchfile = '/'.join((processinglocation, 'distributionmatches.matches.pickle'))
#with open(distributionmatchfile, 'rb') as pick:
#    distributionmatches = pickle.load(pick)
##distributionmatches = defaultdict(list) #distributionkey: [librarykeys]

#roundcutfile = '/'.join((processinglocation, 'roundcutoff.pickle'))
#with open(roundcutfile, 'rb') as pick:
#    roundcutoff = pickle.load(pick)

##linking sum/max dists to their original librarykeys
#linkerfile = '/'.join((librarylocation, 'distributions.linker.pickle'))
#with open(linkerfile, 'rb') as pick:
#    distributionidentifier = pickle.load(pick)
##distributionidentifier = {} #sum/max distribution id: fulldistid
#
#
##peptide sequences by librarykey
#seqfile = '/'.join((librarylocation, 'sequences.pickle'))
#with open(seqfile, 'rb') as pick:
#    seqsbyformula = pickle.load(pick)
##seqsbyformula = defaultdict(list) #formula string: [seqs]

#seqsbyformula = {} #formula: [seqs]
##distributionidentifiers = {} #idn: formula
#with environment_partial(librarylocation) as env:
#    #distdb = '.'.join(('distributionidentifier', proteome))
#    #linkers = env.open_db(distdb.encode())
#    #with env.begin(write=False) as txn:
#    #    with txn.cursor(linkers) as cursor:
#    #        for k, v in cursor:
#    #            distributionidentifiers[int(k.decode())] = v.decode()
#    seqdb = '.'.join(('seqsbyformula', proteome))
#    seqs = env.open_db(seqdb.encode())
#    with env.begin(write=False) as txn:
#        with txn.cursor(seqs) as cursor:
#            for k, v in cursor:
#                key = k.decode()
#                value = eval(v.decode())
#                seqsbyformula[key] = value
#    parameters = env.open_db('isofactors'.encode())
#    with env.begin(write=False) as txn:
#        with txn.cursor(parameters) as cursor:
#            parameterbytes = cursor.get(proteome.encode())
#            parameterdict = dict(eval(parameterbytes.decode()))
#            #newinclimit = float(parameterdict['newinclimit'])
#            #steplimit = float(parameterdict['steplimit'])

newinclimit = 0.1
steplimit = 0.5

#formalized analyte information, summarizing all distributions across any charge states
analytefile = '/'.join((processinglocation, 'analytefactors.pickle'))
with open(analytefile, 'rb') as pick:
    analytekeys, analytedistributions, analytesbydistribution, distributionsoflines, linesofdistributions, linesofanalytes = pickle.load(pick)
#analytekeys = defaultdict(dict) #analyte id: distid: charge, this can be one or multiple distributions based on whether it has multiple charge states
#analytedistributions = {} #analyte id: [[weighted means [via intensity] across isotopomers from every charge state if there are multiple], [AUC of merged isotopomers]]
#analytesbydistribution = {} #distid: analyte id
#distributionsoflines: lineuid: distid
#linesofdistributions: distid: [lineuids ordered by mass]

regionfile = '/'.join((processinglocation, 'regions.pickle'))
with open(regionfile, 'rb') as pick:
    regions = pickle.load(pick)
#regions as [minmass, maxmass, mintime, maxtime, # datapoints, peakarea, maxintensity, wmean, lineid]
regiter = regions[regions[:,4] >= minpoints]

loaderloc = '/'.join((processinglocation, 'trackedgroups.pickle'))
with open(loaderloc, 'rb') as pick:
    trackedgroups = pickle.load(pick)

print(time() - nt, 'loaded')

msrun = mzml.MzML(mzmlfile, dtype=np.float64)

nt = time()

#this multiprocessing version was ~2x faster than the alternative, good enough reason to use it
precursorcoordinates = [] #[rt of previous ms1, lower mass bound, upper mass bound, ms2 scan index]
#scanwindowbounds = {} #scan: [lower, upper] bounds
for output in msrun.map(lambda scan: coordinate_generation(scan), processes=nprocs):
    match output:
        case list():
            coords = output
            precursorcoordinates.append(coords)
            #scanwindowbounds.update(bounds)
        #case None:
        #    pass

print(time() - nt, 'windows collected')

precursorcoordinates = sorted(precursorcoordinates, key=lambda x: x[0]) #sorted by rt
regiter = regiter[regiter[:,2].argsort()].tolist() #sorted by starting time

nt = time()

regioniter = iter(regiter)

##idk why i ever left this in
#reg = next(regioniter)
#regminrt = reg[2]
#regmaxrt = reg[3]
#regid = int(reg[8])
regminrt = -1

linesofscans = defaultdict(list) #ms2 scan index: [lineuids within window]
scansoflines = defaultdict(list) #lineuid: [ms2 scan indices]


#could i just do this with a nearest neighbors? probably, you can do literally everything with a nearest neighbors
#^yes a nn here is definitely a good idea, specifically one implemented with multiprocessing, as it reasonable here. do a radius search
#^BUT if has to iteratively cross over eac result anyways, which might end up costing at least a ~similar amount of time, it might not be worth it
#double iterating to match lines within each ms2 scan sampling window
regpool = []
for pc in precursorcoordinates:
    prt = pc[0]
    pminmass = pc[1]
    pmaxmass = pc[2]
    precid = pc[3]
    while regminrt < prt: #add more regs to regpool
        #no region rts and precursor rts are the same values, don't need <=
        try:
            reg = next(regioniter)
        except StopIteration: #regiter reached the end before precursorcoordinates, which is within the realm of expectations
            break
        regminrt = reg[2]
        #regmaxrt = reg[3]
        regid = int(reg[8])
        regpool.append(regid)
    regremovals = []
    #regcheckers = [] #didn't need, works the same without it
    for r in regpool: #this loop might be worth simple concurrency using mp.Manager().list() for regremovals
        treg = regions[r]
        #tminrt = treg[2]
        trmaxrt = treg[3]
        if trmaxrt < prt:
            regremovals.append(r)
        #elif tminrt < prt:
        #    regcheckers.append(r)
    for r in regremovals:
        regpool.remove(r)
    for r in regpool: #assess reg masses across pc masses
        treg = regions[r]
        trminmass = treg[0]
        trmaxmass = treg[1]
        if pminmass <= trmaxmass and pmaxmass >= trminmass:
            tminrt = treg[2]
            if tminrt < prt:
                linesofscans[precid].append(r)
                scansoflines[r].append(precid)

#scanmasses = {} #scan: [[masses], [intensities]]
for scan, lines in linesofscans.items():
    linesofscans[scan] = tuple(sorted(lines))
    #scanmasses[scan] = np.stack((msrun[scan]['m/z array'], msrun[scan]['intensity array']))
linesofscans = dict(linesofscans)

print(time() - nt, 'windows managed')

#it might be worth visualizing these to see if these are linemodel errors
blankscans = len(precursorcoordinates) - len(linesofscans)
if blankscans > 0:
    blankpercent = blankscans / len(precursorcoordinates)
    print('your instrument produced', blankscans, f'MS2 scans that targeted nothing within the minimum point threshhold of {minpoints} datapoints,', f'{round(blankpercent, 4)}% of all MS2 scans')

#len(set(scansoflines).difference(distributionsoflines)) #-> number of nodists within windows

#this is a more linear version of the above, the outputs match
#regminmass = regiter[:,0]
#regmaxmass = regiter[:,1]
#regmintimes = regiter[:,2]
#regmaxtimes = regiter[:,3]
#linesofscans2 = defaultdict(list) #ms2 scan index: [lineuids within window]
#scansoflines2 = defaultdict(list) #lineuid: [ms2 scan indices]
#for scan in msrun:
#    if scan['ms level'] == 2:
#        precursorinfo = scan['precursorList']['precursor'][0]
#        selectionwindow = precursorinfo['isolationWindow']
#        precmass = selectionwindow['isolation window target m/z'].real
#        lowerbound = precmass - selectionwindow['isolation window lower offset'].real
#        upperbound = precmass + selectionwindow['isolation window upper offset'].real
#        #trainind references index of newtrain, -> get mass -> input to trackedma -> lineuid
#        scanlist = scan['scanList']['scan'][0]
#        rt = scanlist['scan start time'].real
#        scindex = int(scan['index'])
#        #coordinates = [rt, precmass, lowerbound, upperbound, scindex]
#        coordinates = [rt, lowerbound, upperbound, scindex]
#        reginds = np.logical_and.reduce((regmintimes <= rt, regmaxtimes >= rt, regminmass <= upperbound, regmaxmass >= lowerbound))
#        lineuids = regiter[reginds,8].astype(int)
#        #maybe if dist is > roundcutoff, it's not kept? there are some cases where this may be a good idea bc of poor mass selection
#        if lineuids.size > 0:
#            linesofscans2[scindex] = lineuids
#            for lineuid in lineuids.tolist():
#                scansoflines2[lineuid].append(scindex)


#make precursorcoordinates into a dict here and determine MS1 line weightings and line positions (relative to their own distribution) to start taking action on
#you'll need to save the variables here and open a separate file to calculate all this with trackedgroups
#^it might be worth considering only parts of a line around the desired precursor coordinate, because of those weirdly shaped lines

precursordict = {}
for pc in precursorcoordinates:
    precursordict[pc[-1]] = pc[:-1]

nt = time()

#question: does spectralsamplings need to incorporate distance from the center of the quad's target scan as a factor in quantity?
#no, not really. it should be fairly decently governed by mathieu's equation where anything within the range is stable
#that being said, it seems likely that larger m/z ranges are more stable than smaller ones, and the fringes of either might be iffy.

#assigning relative intensity %'s of each MS1 distribution for each MS2 window
scansbyanalyte = defaultdict(list) #analyteid: [scans across all lines and charge states]
scanalytecharges = defaultdict(dict) #analyteid: scan: charge
#spectralsamplings = defaultdict(lambda: defaultdict(dict)) #scan: line: % by area of ms1 lines
lineintensitiesofscans = defaultdict(lambda: defaultdict(dict)) #scan: line: raw area of ms1 line, average of 2 flanking points
linepercentagesofscans = defaultdict(lambda: defaultdict(dict)) #scan: line: % of scan intensity input
scansums = defaultdict(float) #scan: sum area used in lineintensitiesofscans
#isotopomerpositionsofanalytes = defaultdict(set) #analyteid: isotopomer coordinate from max

#these 3 below are keeping track of which line from which distribution [at each charge] gives the most intense MS2 sampling based on MS1 intensity
maxintensitysampleofdists = defaultdict(float) #distid: max intensity
maxintensitylinesofdists = {} #distid: (line, scan)
premaxsampledistributionsoflines = {} #distid: line

for line, scans in scansoflines.items():
    try:
        distid = distributionsoflines[line]
        analyteid = analytesbydistribution[distid]
        #lines = linesofdistributions[distid]
        #maxline = regions[lines,5].argmax()
        #linecoordinate = lines.index(line) - maxline
        charge = analytekeys[analyteid][distid]
    except KeyError: #line is in nodists
        analyteid = -line
        distid = -1
        #linecoordinate = 0
        charge = 0
    scansbyanalyte[analyteid].extend(scans)
    linegroup = trackedgroups[line]
    linetimes = linegroup[:,1]
    linemasses = linegroup[:,0]
    lineintensity = linegroup[:,2]
    lmax = linemasses.max()
    lmin = linemasses.min()
    for scan in scans:
        pcoords = precursordict[scan]
        rt = pcoords[0]
        #this assumes there is a left and right intensity, i guess there would be though?
        #PROBLEM here, this assumes the time difference is the same between the two
        #^there should be a time-based extrapolation here
        #i'm leaving it for later because its simple and probably won't change much
        leftintensity = lineintensity[linetimes < rt][-1]
        rightintensity = lineintensity[linetimes > rt][0]
        sampleintensity = (leftintensity + rightintensity) / 2
        minmass = pcoords[1]
        maxmass = pcoords[2]
        if not lmin > minmass and lmax < maxmass:
            #if the overlap doesn't fully encompass all mass points, normalize by the % mass overlap
            #idc for slight mass shifts, i'm just going by the range, the shifts would be too annoying to incorporate, probably not worth my time
            #this assumes there's no realistic way for the line mass to fully encompass the scans mass window, which there shouldn't be unless the line model screws up
            if lmax > maxmass:
                percentoverlap = (maxmass - lmin) / (lmax - lmin)
            else:
                percentoverlap = (lmax - minmass) / (lmax - lmin)
            sampleintensity *= percentoverlap
        if distid >= 0:
            if sampleintensity > maxintensitysampleofdists[distid]:
                maxintensitysampleofdists[distid] = sampleintensity
                maxintensitylinesofdists[distid] = line, scan
                premaxsampledistributionsoflines[distid] = line
        lineintensitiesofscans[scan][line] = sampleintensity
        scansums[scan] += sampleintensity
        scanalytecharges[analyteid][scan] = charge
    #isotopomerpositionsofanalytes[analyteid].add(linecoordinate)
scansbyanalyte = dict(scansbyanalyte)

#turning areas into percents
for scan, lines in lineintensitiesofscans.items():
    #samplesum = 0
    #for analyteid, positions in analytes.items():
    #    for position, area in positions.items():
    #        samplesum += area
    #for analyteid, positions in analytes.items():
    #    for position, area in positions.items():
    for line in lines:
        linepercentagesofscans[scan][line] = lineintensitiesofscans[scan][line] / scansums[scan]
    linepercentagesofscans[scan] = dict(linepercentagesofscans[scan])
    lineintensitiesofscans[scan] = dict(lines) #can't pickle double default dicts
lineintensitiesofscans = dict(lineintensitiesofscans)
linepercentagesofscans = dict(linepercentagesofscans)

maxsampledistributionsoflines = {} #line: distid
for distid, line in premaxsampledistributionsoflines.items():
    maxsampledistributionsoflines[line] = distid

print(time() - nt, 'window participants quantified')

topanalyterepresentatives = {} #analyteid: charge: (scan [where the line contributes the most], [most abundant] line, [top] subformula)

#moved the generation of spectrabyformula below to distributionmatching
##i probably don't need generatedsequences
##doing this here so i don't iterate redundant sequences later
##generatedsequences = set()
#spectrabyformula = defaultdict(list) #formula: analyteid: [line-position-strings]
##for analyteid, samples in scanalytecharges.items():
#for line, librarykeys in linesbylibrarymatch.items():
#    #if analyteid in distributionmatches: #not in nodists
#    #librarykeys = distributionmatches[analyteid]
#    for libid in librarykeys.tolist(): #matches from the library
#        #if libid % 2: #don't need this anymore, only sum dists exist now
#        #    libid -= 1
#        formula = distributionidentifiers[libid]
#        #generatedsequences.update(seqsbyformula[formula])
#        #for scan in samples:
#        spectrabyformula[formula].append(analyteid)
##generatedsequences = list(generatedsequences)
#
##spectrabyformula = {}
##temp = spectrabyformula.items()
#for k, v in spectrabyformula.items():
#    for sk, sv in v.items():
#        #spectrabyformula[k] = {sk: list(sv)}
#        spectrabyformula[k][sk] = tuple(sv)
#    spectrabyformula[k] = dict(spectrabyformula[k])
#spectrabyformula = dict(spectrabyformula)

scansoflinesfile = '/'.join((processinglocation, 'scansoflines.pickle'))
with open(scansoflinesfile, 'wb') as pick:
    pickle.dump(scansoflines, pick)

linesofscansfile = '/'.join((processinglocation, 'linesofscans.pickle'))
with open(linesofscansfile, 'wb') as pick:
    pickle.dump(linesofscans, pick)

scansbyanalytefile = '/'.join((processinglocation, 'scansbyanalyte.pickle'))
with open(scansbyanalytefile, 'wb') as pick:
    pickle.dump(scansbyanalyte, pick)

scanalytefile = '/'.join((processinglocation, 'scanalytes.pickle'))
with open(scanalytefile, 'wb') as pick:
    pickle.dump(scanalytecharges, pick)

lineintensitiesofscansfile = '/'.join((processinglocation, 'lineintensitiesofscans.pickle'))
with open(lineintensitiesofscansfile, 'wb') as pick:
    pickle.dump(lineintensitiesofscans, pick)

linepercentagesofscansfile = '/'.join((processinglocation, 'linepercentagesofscans.pickle'))
with open(linepercentagesofscansfile, 'wb') as pick:
    pickle.dump(linepercentagesofscans, pick)

#scanmassesfile = '/'.join((processinglocation, 'scanmasses.pickle'))
#with open(scanmassesfile, 'wb') as pick:
#    pickle.dump(scanmasses, pick)

#scanboundsfile = '/'.join((processinglocation, 'fragmentbounds.pickle'))
#with open(scanboundsfile, 'wb') as pick:
#    pickle.dump(scanwindowbounds, pick)

#analyteboundsfile = '/'.join((processinglocation, 'analytebounds.pickle'))
#with open(analyteboundsfile, 'wb') as pick:
#    pickle.dump(analytefragmentbounds, pick)

#pepfragsfile = '/'.join((processinglocation, 'fragment.peptides.pickle'))
#with open(pepfragsfile, 'wb') as pick:
#    pickle.dump(generatedsequences, pick)

#isotopomerpositionsfile = '/'.join((processinglocation, 'isotopomersbypositions.pickle'))
#with open(isotopomerpositionsfile, 'wb') as pick:
#    pickle.dump(isotopomerpositionsofanalytes, pick)

#spectrabyformulafile = '/'.join((processinglocation, 'spectrabyformula.pickle'))
#with open(spectrabyformulafile, 'wb') as pick:
#    pickle.dump(spectrabyformula, pick)

maxintensitylinesofdistsfile = '/'.join((processinglocation, 'maxintensitylinesofdists.pickle'))
with open(maxintensitylinesofdistsfile, 'wb') as pick:
    pickle.dump(maxintensitylinesofdists, pick)
#maxintensitylinesofdists = {} #distid: (line, scan)

maxsampledistributionsoflinesfile = '/'.join((processinglocation, 'maxsampledistributionsoflines.pickle'))
with open(maxsampledistributionsoflinesfile, 'wb') as pick:
    pickle.dump(maxsampledistributionsoflines, pick)
#maxsampledistributionsoflines = {} #line: distid
