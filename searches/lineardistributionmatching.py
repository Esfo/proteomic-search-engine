import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from colour import Color
from pyteomics import mzml
from time import time
import lmdb
import pandas as pd
import gc
from functools import partial
import multiprocessing as mp
from collections import Counter, defaultdict
from scipy import sparse, integrate, spatial, stats, special
from distinctipy import distinctipy as dp
import random
import itertools
import pickle
import sys
import os
gc.enable()

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
        '#e85d58',
        '#b88cfa',
        '#f5972c',
        '#2ded8d',
        '#4bc8f2',
        '#ea68f2',
        ]
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', chexes)
#n = 0
#for c in chexes:
#    plt.plot([n, n+1], [n, n+1], color=c)
#    n += 1
#plt.show()

proton = 1.007276554940804

mzmlfile = '/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
basefolder = '/'.join((mzmlfile.split('/')[:-2]))
basefile = mzmlfile.split('/')[-1].split('.mzML')[0]

processinglocation = '/'.join((basefolder, 'fileprocessing', basefile))
#librarylocation = '/home/sfo/data/proteomics/fastas/isotope-arrays/human_isotopes-6-50_miss-1_ss50'
librarylocation = '/home/sfo/data/proteomics/fastas/search-db'
#proteome = proteomefile.split('/')[-1].split('.')[0]
#proteome = 'Human_Homo_sapien-NoTremb'
proteome = 'Human_Homo_sapien'
ppmallowance = 20
matchallowance = 2 #the allowance metric for rank disagreements between the library and the experimental data
#^only the top N (must be adjacent) of these will be applied, meaning the lowest allowances will be the limit of what's applied within the search framework

environment_partial = partial(lmdb.Environment, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=99999, max_spare_txns=1, lock=True)

#formalized analyte information, summarizing all distributions across any charge states
analytefile = '/'.join((processinglocation, 'analytefactors.pickle'))
with open(analytefile, 'rb') as pick:
    analytekeys, analytedistributions, analytesbydistribution, distributionsoflines, linesofdistributions, linesofanalytes = pickle.load(pick)
#analytekeys = defaultdict(dict) #analyte id: distid: charge, this can be one or multiple distributions based on whether it has multiple charge states
#analytedistributions = defaultdict(dict) #analyte id: ordered masses: AUC of merged isotopomers, weighted means [via intensity] across isotopomers from every charge state, isotopomer datapoints merged across each charge state - if there are any
#analytesbydistribution = {} #distid: analyte id

scansbyanalytefile = '/'.join((processinglocation, 'scansbyanalyte.pickle'))
with open(scansbyanalytefile, 'rb') as pick:
    scansbyanalyte = pickle.load(pick)

loaderloc = '/'.join((processinglocation, 'trackedgroups.pickle'))
with open(loaderloc, 'rb') as pick:
    trackedgroups = pickle.load(pick)

regionfile = '/'.join((processinglocation, 'regions.pickle'))
with open(regionfile, 'rb') as pick:
    regions = pickle.load(pick)
#regions as [minmass, maxmass, mintime, maxtime, # datapoints, peakarea, maxintensity, wmean, lineid]

chargeregionsfile = '/'.join((processinglocation, 'chargeregions.pickle'))
with open(chargeregionsfile, 'rb') as pick:
    chargeregions = pickle.load(pick)
#chargeregions = [mincharge, maxcharge, mintime, maxtime, len(v), area, maincharge, k]

scansoflinesfile = '/'.join((processinglocation, 'scansoflines.pickle'))
with open(scansoflinesfile, 'rb') as pick:
    scansoflines = pickle.load(pick)

linesofscansfile = '/'.join((processinglocation, 'linesofscans.pickle'))
with open(linesofscansfile, 'rb') as pick:
    linesofscans = pickle.load(pick)

#nodistfile = '/'.join((processinglocation, 'nodists.pickle'))
#with open(nodistfile, 'rb') as pick:
#    nodists = pickle.load(pick)

##sumdistributionfile = '/'.join((librarylocation, 'distributions.sum.pickle'))
##with open(sumdistributionfile, 'rb') as pick:
##    sumabundances = pickle.load(pick)
#sumdb = '.'.join((proteome, 'sum'))
#sumabundances = {}
#with sq.SqliteDict(librarylocation, tablename=sumdb, flag='r') as db:
#    for k, v in db.items():
#        sumabundances[k] = v
##isotopic distributions and their sequences
##seqsbymass = defaultdict(list) #mass: [sequences]
##abundances = {} #main mass: isotopomer masses: relative isotopomer intensities
#
#maxdb = '.'.join((proteome, 'max'))
#maxabundances = {}
#with sq.SqliteDict(librarylocation, tablename=maxdb, flag='r') as db:
#    for k, v in db.items():
#        maxabundances[k] = v
##maxdistributionfile = '/'.join((librarylocation, 'distributions.max.pickle'))
##with open(maxdistributionfile, 'rb') as pick:
##    maxabundances = pickle.load(pick)

def arg_coord_rectangle_overlap(rec, coords):
    tops, bottoms, lefts, rights = coords.transpose()
    c1 = rec[2] < rights
    c2 = rec[3] > lefts
    c3 = rec[0] < bottoms
    c4 = rec[1] > tops
    overlaps = np.argwhere(c1 & c2 & c3 & c4)
    return overlaps.flatten()

nt = time()

#only load one, both is too much for this file
getkeys = []
formulaidentifiers = {}
sumabundances = {}
#maxabundances = {}
formuladb = '.'.join(('formulaidentifier', proteome))
with environment_partial(librarylocation) as env:
    formulas = env.open_db(formuladb.encode())
    with env.begin(write=False) as txn:
        with txn.cursor(formulas) as cursor:
            for k, v in cursor:
                formulaidentifiers[k.decode()] = int(v.decode())
                getkeys.append(k)
    sums = env.open_db('distributions.sum'.encode())
    with env.begin(write=False) as txn:
        with txn.cursor(sums) as cursor:
            for k, v in cursor.getmulti(getkeys):
                out = np.frombuffer(v)
                out = out.reshape(2, out.size//2)
                sumabundances[k.decode()] = out
#    maxes = env.open_db('distributions.max'.encode())
#    with env.begin(write=False) as txn:
#        with txn.cursor(maxes) as cursor:
#            for k, v in cursor.getmulti(getkeys):
#                out = np.frombuffer(v)
#                out = out.reshape(2, out.size//2)
#                maxabundances[k.decode()] = out
print(time() - nt, 'loaded')

#do i need this here?
#linkerfile = '/'.join((foldername, 'distributions.linker.pickle'))
#with open(linkerfile, 'rb') as pick:
#    distributionlinker = pickle.load(pick)


#library matching is currently met with analytedistributions which might disregard incorrect charge-state matches
#the current charge-matching scheme allows for more intensity-disagreement than the below distribution matching does. Maybe this would be better to free the reigns here and match the dists to analytedists afterwards?

#so even with masses and intensities already being stored as numpy arrays within the abundance dicts, I'm still putting them into specified mass/intensity dicts below because i'd have to know which of either max or sum it came from, the library's arrays are more of a memory impact here so this is a bit of a minus, but it's not a huge deal for the distribution side because that's usually a much smaller memory impact, it's annoying that it's gotta be doe like that but oh well.

#masses,intensities (the sum norm doesnt change the adjacency ratios, right?) (also i dont need to save the intensities in a dict, just use it to generate this other stuff), intensityranks, adjacent intensity ratios -> and them argsorted, 
librarykeys = []
librarymasses = []
librarymassdict = {} #lid: [masses]
libraryintensities = {} #lid: [intensities]
libraryintensityranks = {} #lid: [intensityranks]
librarydirections = {} #lid: [increasing/decreasing, max=0]
#libraryadjacentintensityratios = {} #lid [you get it...]
#libraryadjacentintensityratioranks = {} #this can just be an argsort

#for f, (masses, intensities) in maxabundances.items():
#    k = formulaidentifiers[f] + 1
#    librarymassdict[k] = masses
#    libraryintensities[k] = intensities
#    intensityranks = intensities.argsort()[::-1]
#    libraryintensityranks[k] = intensityranks
#    intensityratios = intensities[:-1] / intensities[1:]
#    libraryadjacentintensityratios[k] = intensityratios
#    intensityratioranks = intensityratios.argsort() #directionality doesn't matter
#    libraryadjacentintensityratioranks[k] = intensityratioranks
#    librarykeys.extend(itertools.repeat(k, masses.size))
#    librarymasses.extend(masses.tolist())

for f, (masses, intensities) in sumabundances.items():
    k = formulaidentifiers[f]
    librarymassdict[k] = masses
    libraryintensities[k] = intensities
    intensityranks = np.abs(intensities.argsort().argsort() - intensities.size + 1)
    maxloc = intensities.argmax()
    leftdirections = (np.diff(intensities[:maxloc+1]) > 0).tolist()
    rightdirections = (np.diff(intensities[maxloc:]) > 0).tolist()
    directions = [1 if i else -1 for i in leftdirections] + [0] + [1 if i else -1 for i in rightdirections]
    librarydirections[k] = directions
    libraryintensityranks[k] = intensityranks
    #intensityratios = intensities[:-1] / intensities[1:]
    #libraryadjacentintensityratios[k] = intensityratios
    #intensityratioranks = intensityratios.argsort() #directionality doesn't matter
    #libraryadjacentintensityratioranks[k] = intensityratioranks
    librarykeys.extend(itertools.repeat(k, masses.size))
    librarymasses.extend(masses.tolist())

librarykeys = np.array(librarykeys)
librarymasses = np.array(librarymasses)

distributionkeys = []
distributionmasses = []
distributionmassdict = {} #did: [masses]
distributionintensities = {} #did: [intensities]
distributionintensityranks = {} #did: [intensityranks]
distributiondirections = {} #did: [+/- directions]
#distributionadjacentintensityratios = {} #did: [you get it...]
#distributionadjacentintensityratioranks = {} #this can just be an argsort
for k, (masses, intensities) in analytedistributions.items():
    distributionmassdict[k] = masses
    distributionintensities[k] = intensities
    intensityranks = np.abs(intensities.argsort().argsort() - intensities.size + 1)
    maxloc = intensities.argmax()
    leftdirections = (np.diff(intensities[:maxloc+1]) > 0).tolist()
    rightdirections = (np.diff(intensities[maxloc:]) > 0).tolist()
    directions = [1 if i else -1 for i in leftdirections] + [0] + [1 if i else -1 for i in rightdirections]
    distributiondirections[k] = directions
    distributionintensityranks[k] = intensityranks
    #intensityratios = intensities[:-1] / intensities[1:]
    #distributionadjacentintensityratios[k] = intensityratios
    #intensityratioranks = intensityratios.argsort() #directionality doesn't matter
    #distributionadjacentintensityratioranks[k] = intensityratioranks
    distributionkeys.extend(itertools.repeat(k, masses.size))
    distributionmasses.extend(masses.tolist())

distributionkeys = np.array(distributionkeys)
distributionmasses = np.array(distributionmasses)

radius = distributionmasses / 1000000 * ppmallowance

#switch the train to distmatches?
lmtree = spatial.KDTree(librarymasses[:,None])
matches = lmtree.query_ball_point(distributionmasses[:,None], radius, workers=8)

#this alone doesn't improve speed, but it prevents duplicate matches - in some ways i think, so in one sense it takes memory to make but saves some later i guess? sure
matchorganizer = defaultdict(list)
for dk, lkeys in zip(distributionkeys, matches):
    matchorganizer[dk].extend(librarykeys[lkeys])

#outlist = []
for k in list(matchorganizer):
    matchorganizer[k] = np.array(list(set(matchorganizer[k])))
#    for vs in matchorganizer[k].tolist():
#        outlist.append((k, vs))
#outlist = np.array(outlist)
#
#for this process that takes the longest out of any yet, and for which can't be made concurrent at all, i might want to reduce the process to simply matching ranks

nt = time()

lmatches = 0
dmatches = 0
#librarymatches = defaultdict(list) #librarykey: [distributionkeys]
#badmatches = defaultdict(list) #distributionkey: [librarykeys]
badmatches = {} #analyteid: [[librarykey], [allowance]]
equalmatches = {} #analyteid: [[librarykey], [allowance]]
disassociatedmatches = {} #analyteid: [[librarykey], [allowance]]
distributionmatches = {} #analyteid: [[lk], [allowance]]
#matchpairs = {} #matchid: (librarykey, distkey)
matchpairs = defaultdict(dict) #distkey: librarykey: matchid
#matchpairs = {} #pairkey: [librarykey, distkey]
#pairallowances = {} #pairkey: allowance
#pairrankdiffs = {} #pairkey: max rank diff
nonallowedpairs = {} #for plotting negatives
nonmaxpairing = {}
#lonesignalmatches = {}
#bestallowance = {} #analyteid: best aka initial allowance
failcatch = -1 #probability of preserving any given failure, it's random because there would be too many
#matchmetrics = {} #matchid: [....]
badallowances = defaultdict(list) #allowance: [[lk, dk]'s]
disassociatedallowances = defaultdict(list) #allowance: [[lk, dk]'s]
equalallowances = defaultdict(list) #allowance: [[lk, dk]'s]
matchallowances = defaultdict(list) #multiplied allowance: [matchids]
matchdirsums = {}
matchallsums = {}
output = []
mid = 0
fid = 0
npid = 0
bid = 0
bad = 0
equal = 0
disassociated = 0
#for dk, lkeys in zip(distributionkeys, matches):
for dk, lkeys in matchorganizer.items():
    #badmatches[dk] = [[], []]
    #equalmatches[dk] = [[], []]
    #disassociatedmatches[dk] = [[], []]
    distributionmatches[dk] = [[], []]
    dmasses = distributionmassdict[dk]
    dsize = dmasses.size
    tx = 0
    for lk in lkeys.tolist(): #can this loop be made concurrent?
        lmasses = librarymassdict[lk]
        lsize = lmasses.size
        
        leftoffset = int(round(lmasses.tolist()[0] - dmasses.tolist()[0]))
        if leftoffset > 0:
            li = 0
            rmax = dsize - leftoffset
            maxsize = min(lsize, rmax)
            ri = leftoffset
        elif leftoffset == 0:
            li = 0
            ri = 0
            maxsize = min(lsize, dsize)
        else: #< 0
            li = -leftoffset
            ri = 0
            lmax = lsize - li
            maxsize = min(lmax, dsize)
        
        #scattering the rest of these to only where they're needed
        le = li + maxsize
        lrange = le - li
        if lrange > 1:
            lintranks = libraryintensityranks[lk][li:le]
            if 0 in lintranks: #the top library rank is included
            #if min(lintranks.tolist()) <= matchallowance:
                #adjacentmaxsize = maxsize - 1
                
                ##it would be good to be able to snip off ends that don't work here maybe?
                #lintranks = sorted(lintranks.tolist())
                #lintdiffs = [lintranks[i+1] - lintranks[i] for i in range(adjacentmaxsize-1)] #^idk why this has the -1, but it gave matching problems for dists of only 2 masses, they didn't have any lintdiffs so they never matched
                #lintdiffs = [lintranks[i+1] - lintranks[i] for i in range(adjacentmaxsize)]
                #^i'm not convinced this is so valuable anymore
                
                #ie, it should be matching the meat of the librarymatch
                #if lintdiffs and max(lintdiffs) == 1: #those that come after the top rank are in the natural/desired sequential order, checking if lintdiffs exists means there should be more than 1 match, and prevents a valuerror
                #if lintdiffs: #match > 1
                #^you could just check for len with the above 0 in lintranks instead of doing it here
                re = ri + maxsize
                #drange = re - ri
                #if drange > 1: #already checked via libmatch, this does nothing
                ##below basically allows all adjacent intensity ratio ranks to have a maximum of 1-rank difference, and one rank can have a 2. it allows ranks to shift, as a whole, while not allowing things to lose their relative order too much.
                ##^now it just allows 1 difference
                #
                ##dorders = (dintslice[:-1] / dintslice[1:]).argsort()
                ##lorders = (lintslice[:-1] / lintslice[1:]).argsort()
                ##dorders = distributionadjacentintensityratioranks[dk][ri:reaj].tolist()
                ##lorders = libraryadjacentintensityratioranks[lk][li:leaj].tolist()
                dorders = distributionintensityranks[dk][ri:re].tolist()
                #dorders = distributionintensityranks[dk][ri:re]
                #mindorders = (dorders - dorders.min()).tolist() #realigning dist ranks for shaky dists
                #dorders = dorders.tolist()
                lorders = libraryintensityranks[lk][li:le].tolist() #this is the same as lintranks above.. technically, but I'll just keep this here for now because that one gets messed with a little
                ##^this lower try got less matches, I suppose that means its going in the wong direction in terms of flexibility
                #
                ##orderdiffs = np.abs(dorders - lorders)
                orderdiffs = [abs(i-j) for i, j in zip(dorders, lorders)]
                #minorderdiffs = [abs(i-j) for i, j in zip(mindorders, lorders)]
                ##^also make adjacent order diffs and compare those aross lib-dist matches
                ##i could introduce a ranking process for each individual dist! using a bunch of metrics
                ##
                ldirections = librarydirections[lk][li:le]
                ddirections = distributiondirections[dk][ri:re]
                directionallowance = [abs(i-j) for i, j in zip(ldirections, ddirections)]
                ###ndiffs = (orderdiffs > 0).sum()
                ###ndiffs = sum((orderdiffs > 0).tolist())
                ###ndiffs = sum(i > 0 for i in orderdiffs)
                ###^going to stop using this because matches end up being too flexible! it needs more stringency
                #missinglorders = max(lorders) - len(lorders) + 1
                #missingdorders = max(dorders) - len(dorders) + 1
                #
                ###allowance = orderdiffs.sum() - ndiffs - 1
                ###allowance = sum(orderdiffs.tolist()) - ndiffs - 1
                ###allowance = sum(orderdiffs) - ndiffs - 1
                ###allowance = sum(orderdiffs) - matchallowance
                allowance = sum(orderdiffs)
                #mallowance = sum(minorderdiffs)
                dallowance = sum(directionallowance)
                #finalallowance = allowance + dallowance + missinglorders + missingdorders #total number of errors
                floatingallowance = ((dallowance / lrange) + (allowance / lrange)) / (lrange / dsize)
                lintensities = libraryintensities[lk][li:le]
                #i think normalizing by the sum here, rather than the max, gives a fairer playing field for distributions across comparisons, you can know that a full difference of 1 means a difference of the entire distributions intensity, rather than 1 meaning simply the max while the max could be 90% or 10% of the total
                lintensities = lintensities / lintensities.sum()
                dintensities = distributionintensities[dk][ri:re]
                dintensities = dintensities / dintensities.sum()
                differencefloats = np.abs(lintensities - dintensities)
                #dmax = differencefloats.max() / lintensities.max()
                areafloat = differencefloats.sum()
                #output.append((allowance, dallowance, dsize, lrange, dmax, areafloat, mid))
                output.append((allowance, dallowance, dsize, lrange, areafloat, mid))
                #guide with integers, ie allowance, dallowance, lrange. then judge with floats, ie intensity matches
                #^then you can make distribution-based cutoffs for each level of allowance/dallowance and lrange
                #ie if an lrange of 4 has all its top allowances have a floating difference of ~0.4, then everything with a higher allowance but less of a difference can pass?
                #floating allowance can be based on normalized percentages of intensities based on lrange
                #^ the +/- can be absolute , with lower being best
                #lrange -> allowance -> floats
                #lrange -> dallowance -> floats
                #if either past via dist cutoffs, then they can be good
                #explore if normalizing by lrange equates the dists>???
                #set up free passes for both directions and ranks then view their +/- floating values, compare to others?
                #
                ##if dk not in bestallowance or bestallowance[dk] <= allowance - matchallowance:
                ##I also need to make sure that the librarymatch has its top players matched in the game - any mismatch in order of the librarymatch rank order to the distmatch means that distmatch isotopomer won't be considered, and the logic will follow suit
                ##if allowance <= 0:
                #reaj = ri + adjacentmaxsize
                #leaj = li + adjacentmaxsize
                ##consistency of proton-dists among the distmatches compared to all the others, this would include a shadow-like component towards non-matched distlines if their distances aren't consistent with what's matched
                ##but when it comes to consistency of mass distances.. do i even want this? this doesn't necessarily make sense because some distributions might be slightly close in distance or slightly farther away due to slightly different atomic compositions, but i think same-shaped distributions at similar masses might not necessarily be a better match just because it's a little closer, idk how the ppm error is going to play out yet
                ##intensity differences are definitely acceptable
                #dintslice = distributionintensities[dk][ri:re]
                #lintslice = libraryintensities[lk][li:le]
                #
                #dmslice = np.array(dmasses[ri:re])
                #lmslice = np.array(lmasses[li:le])
                #
                ##dnorm = dintslice / dintslice.sum()
                ##lnorm = lintslice / lintslice.sum()
                #dnorm = dintslice / sum(dintslice.tolist())
                #lnorm = lintslice / sum(dintslice.tolist())
                #
                #intensitydiff = dnorm - lnorm
                #idmean = sum(intensitydiff.tolist()) / maxsize
                ##intensitydiffs = np.abs(intensitydiff - idmean)
                #intensitydiffs = [abs(idmean - i) for i in intensitydiff.tolist()]
                #meanintensitydiff = sum(intensitydiffs) / maxsize
                #
                ##diaj = dnorm[:-1] / dnorm[1:]
                ##liaj = lnorm[:-1] / lnorm[1:]
                #diaj = distributionadjacentintensityratios[dk][ri:reaj]
                #liaj = libraryadjacentintensityratios[lk][li:leaj]
                #iajdiff = diaj - liaj
                #iajmean = sum(iajdiff.tolist()) / adjacentmaxsize
                ##intensityadjacentdiffs = np.abs(iajdiff - iajmean)
                #intensityadjacentdiffs = [abs(iajmean - i) for i in iajdiff.tolist()]
                #intensityadjacency = sum(intensityadjacentdiffs) / adjacentmaxsize
                #
                ##rewrites:
                ##abs function is fine
                ##change diff to [1:] - [:-1]
                ##sums to sum(tolist())
                ##generate any means as a sum(tolist()) / size
                #
                #massdiff = lmslice - dmslice
                #mdmean = sum(massdiff.tolist()) / maxsize
                ##massdiffs = np.abs(massdiff - mdmean)
                #massdiffs = [abs(mdmean - i) for i in massdiff.tolist()]
                #meanmassdiff = sum(massdiffs) / maxsize
                #
                ##these are probably fine, but could be list comprehensions, only seemed to save ~100ns, it might matter because this has bajilions of iterations
                #dadjdiffs = dmslice[1:] - dmslice[:-1]
                #ladjdiffs = lmslice[1:] - lmslice[:-1]
                #adjacentdiffs = dadjdiffs - ladjdiffs
                #adjmean = sum(adjacentdiffs.tolist()) / adjacentmaxsize
                ##massadjacenctdiffs = np.abs(adjacentdiffs - adjmean)
                #massadjacentdiffs = [abs(adjmean - i) for i in adjacentdiffs.tolist()]
                #massadjacency = sum(massadjacentdiffs) / adjacentmaxsize
                
                #number of matches is maxsize
                
                #matchpairs[mid] = tuple([lk, dk])
                #matchpairs[dk][lk] = mid
                matchpairs[mid] = tuple([lk, dk])
                #distributionmatches[dk][0].append(lk)
                #distributionmatches[dk][1].append(finalallowance)
                #distributionmatches[dk][1].append(floatingallowance)
                #matchallowances[finalallowance].append(mid)
                #matchallsums[mid] = allowance
                #matchdirsums[mid] = dallowance
                #pairallowances[mid] = allowance
                #pairrankdiffs[mid] = max(lintdiffs)
                #matchmetrics[mid] = np.array([allowance, meanintensitydiff, intensityadjacency, meanmassdiff, massadjacency, maxsize])
                #if allowance < mallowance:
                #    badmatches[dk][0].append(lk)
                #    badallowances[allowance].append(mid)
                #    badmatches[dk][1].append(allowance)
                #    bad += 1
                #elif allowance == mallowance:
                #    equalmatches[dk][0].append(lk)
                #    equalallowances[allowance].append(mid)
                #    equalmatches[dk][1].append(allowance)
                #    equal += 1
                #else:
                #    disassociatedmatches[dk][0].append(lk)
                #    disassociatedallowances[mallowance].append(mid)
                #    disassociatedmatches[dk][1].append(mallowance)
                #    disassociated += 1
                #if dk not in bestallowance:
                #    bestallowance[dk] = allowance
                #elif allowance < bestallowance[dk]:
                #    bestallowance[dk] = allowance
                mid += 1
                tx += 1
                #else:
                #    #rank orders disagreed too much
                #    if random.random() > failcatch:
                #        nonallowedpairs[fid] = tuple([lk, dk])
                #        fid += 1
                #else:
                #    #matched library orders weren't perfect
                #    if random.random() > failcatch:
                #        lonesignalmatches[bid] = tuple([lk, dk])
                #        bid += 1
                #else:
                #    #0 wasn't in lintranks
                #    if random.random() > failcatch:
                #        nonmaxpairing[npid] = tuple([lk, dk])
                #        npid += 1
    #if len(distributionmatches[dk][0]) > 0:
    #    distributionmatches[dk] = np.array(distributionmatches[dk])
    #else:
    #    del distributionmatches[dk]
    
    #if len(badmatches[dk][0]) > 0:
    #    badmatches[dk] = np.array(badmatches[dk])
    #else:
    #    del badmatches[dk]

    #if len(equalmatches[dk][0]) > 0:
    #    equalmatches[dk] = np.array(equalmatches[dk])
    #else:
    #    del equalmatches[dk]
    #
    #if len(disassociatedmatches[dk][0]) > 0:
    #    disassociatedmatches[dk] = np.array(disassociatedmatches[dk])
    #else:
    #    del disassociatedmatches[dk]
    
    if tx > 0:
        lmatches += tx
        dmatches += 1
print(time() - nt)

print('library matches:', lmatches)
print('dist matches:', dmatches)


#Exploration into the nature of the matched dists vs unmatched dists vs scans with no dists? they might have lines idk i didnt check
msrun = mzml.MzML(mzmlfile)
matcheddistkeys = np.unique(np.array(list(map(list, matchpairs.values())))).tolist()

accountedscans = set()
for ak in matcheddistkeys:
    for dist in analytekeys[ak]:
        for line in linesofdistributions[dist]:
            if line in scansoflines:
                accountedscans.update(scansoflines[line])

unaccountedscans = set(linesofscans).difference(accountedscans)

#labels = ['number of ions', 'sum abundance', 'max abundance', 'avg abundance', 'average mass', 'weighted average mass']
labels = ['number of ions', 'average mass', 'weighted average mass']

scanstats = {} #scan: [number of ions, sum abundance, max abundance, avg abundance, average mass, weighted average mass]
for scan in msrun:
    if scan['ms level'] == 2:
        index = scan['index']
        mza = scan['m/z array']
        intensities = scan['intensity array']
        nions = len(mza)
        #sumabundance = intensities.sum()
        #maxabundance = intensities.max()
        #avgabundance = intensities.mean()
        meanmasses = mza.mean()
        wmeanmass = (mza * intensities).sum() / intensities.sum()
        #scanstats[index] = [nions, sumabundance, maxabundance, avgabundance, meanmasses, wmeanmass]
        scanstats[index] = [nions, meanmasses, wmeanmass]

unknownscans = set(scanstats).difference(accountedscans.union(unaccountedscans))

for n, l in enumerate(labels):
    accountedstats = list(map(scanstats.get, accountedscans))
    unaccountedstats = list(map(scanstats.get, accountedscans))
    unknownstats = list(map(scanstats.get, unknownscans))
    fig, ax = plt.subplots()
    ax.hist(np.array(accountedstats)[:,n], bins=100, color='red', alpha=0.5)
    tx = ax.twinx()
    tx.hist(np.array(unaccountedstats)[:,n], bins=10, color='green', alpha=0.5)
    nx = tx.twinx()
    nx.hist(np.array(unknownstats)[:,n], bins=30, color='blue', alpha=0.5)
    plt.title(l)
    plt.show()

#based on the number of ions and the average fragment mass, it seems these are two different species! one group is clearly peptides, the other is something else
#if you present this you'll want to clean up the axes and also present the number of scans in each group as they're wildly different


#metricstrings = ['dmax', 'areafloat']
#outputstrings = ['allowance', 'dallowance', 'dsize']
#for mn, metstring in enumerate(metricstrings):
#    mn += len(outputstrings) + 1
#    for n, outstring in enumerate(outputstrings):
#        organizer = defaultdict(list) #output level: [areafloats]
#        for o in output:
#            organizer[o[n]].append(o)
#        alternateorganizer = list(range(len(outputstrings)))
#        alternateorganizer.remove(n)
#        #for altorg in alternateorganizer:
#        altorg = 3 #its always lrange that works best
#        #suborgstring = outputstrings[altorg]
#        suborgstring = 'lrange'
#        numberofmatchesbysize = {} #number of iso matches: number of instances
#        for size, metrics in sorted(organizer.items(), key=lambda x: -len(x[1])):
#            floatsbyaltorg = defaultdict(list) #dsize: [areafloats]
#            for m in metrics:
#                floatsbyaltorg[m[altorg]].append(m[mn])
#            metrics = np.array(metrics)
#            fig, ax = plt.subplots(figsize=(6,6), nrows=2, sharex=True)
#            for alt, afs in sorted(floatsbyaltorg.items(), key=lambda x: -len(x[1])):
#                ax[1].hist(afs, label=alt, alpha=0.3)
#            #plt.legend()
#            ax[0].hist(metrics[:,mn], bins=int(np.sqrt(metrics.shape[0])), alpha=0.2, color='yellow')
#            ax[1].set_yscale('log')
#            ax[1].legend()
#            plt.suptitle(metstring + ' of ' + str(size) + ' ' + outstring + ' by ' + suborgstring)
#            plt.show()
#            numberofmatchesbysize[size] = len(metrics)
#        
#        plt.bar(numberofmatchesbysize.keys(), numberofmatchesbysize.values())
#        plt.title(metstring + ' of ' + outstring + ' by ' + suborgstring)
#        plt.show()

#an exploration:
#the above works by taking each concept in outputstrings, as generated in the large loop above this one, and re-organizes individual float values of that initial outputstring by another output string to decipher which combination leads to more stringent selection processes
#interpret as "all the things of [output concept 1] at level [x] subdivided by [output concept 2]
#this is a pretty great method of inference for determining a "high false positive" cutoff from multiple metrics that indicate strictness via their combination
#the bisection of only 2 of the group lends itself favorably to higher false positives by making less delineations than possible but more than necessary (more than 1 way to win, nothing is over-punished)

#what works:
#dsize by lrange -> when a larger distribution matches more isos, its a better match
#anything with high lrange and low dallowance or allowance would be good, naturally
#^or just low allowance/dallowance in general
#dallowance by dsize -> when smaller direction errors match to isos with larger distributions, you see an increase in stringency - which isn't amazing if you ask me but the data looks good
#dallowance by lrange -> when smaller direction errors match to more isos, you get an increase in stringency
#allowance by dsize -> when allowance gets smaller and dsize is larger you get ~mediocre stringency improvements, some of them look sketchy but obviously the very alow allowances have good looking values
#allowance by lrange -> when smaller allowances match to more isos, you get better selections

nt = time()
arrayout = np.array(output) #(allowance, dallowance, dsize, lrange, dmax, areafloat, mid)
#so i'll work with:
#dsize by lrange -> highest equal match, in this data its at 8, where lrange == dsize
equalmatchlengths = arrayout[arrayout[:,2] == arrayout[:,3]]
maxequalmatches = equalmatchlengths[equalmatchlengths[:,3] == equalmatchlengths[:,3].max()]
#dsizebylrangedmaxcutoff = maxequalmatches[:,4].max()
dsizebylrangeareafloatcutoff = maxequalmatches[:,4].max()
#dallowance by lrange -> use the highest float of the highest lrange at the lowest dallowance
mindallowancearray = arrayout[arrayout[:,1] == arrayout[:,1].min()]
maxlrangedallowancearray = mindallowancearray[mindallowancearray[:,3] == mindallowancearray[:,3].max()]
#dallowancebylrangedmaxcutoff = maxlrangedallowancearray[:,4].max()
dallowancebylrangeareafloatcutoff = maxlrangedallowancearray[:,4].max()
#allowance by lrange -> use the highest float of the highest lrange at the lowest allowance
minallowancearray = arrayout[arrayout[:,0] == arrayout[:,0].min()]
maxlrangeallowancearray = minallowancearray[minallowancearray[:,3] == minallowancearray[:,3].max()]
#allowancebylrangedmaxcutoff = maxlrangeallowancearray[:,4].max()
allowancebylrangeareafloatcutoff = maxlrangeallowancearray[:,4].max()

#now i should order these and check how many potential IDs there are at each layer
print('rank allowance by number of matches cutoff:')
#print('difference max:', allowancebylrangedmaxcutoff)
print('floating area:', allowancebylrangeareafloatcutoff)
print('distribution size by number of matches cutoffs:')
#print('difference max:', dsizebylrangedmaxcutoff)
print('floating area:', dsizebylrangeareafloatcutoff)
print('directional allowance by number of matches cutoff:')
#print('difference max:', dallowancebylrangedmaxcutoff)
print('floating area:', dallowancebylrangeareafloatcutoff)
print(time() - nt)
#actually this is a little stupid because you can just record dmax/areafloats by mid
#then you can collect all this information OTF

#for something like allowance by lrange
# - take the max value of the highest lrange for the lowest allowance
# - then move up to the next highest allowance
#   > if the average (highest point of the density/histogram) of ALL areafloats/dmaxes of that allowance is to the left of the maximum value from above, then accept at least the maximum value of the highest lrange of all allowance groups

#allow perfect allowance and dallowance -> plot them to see what the worst areafloats look like for these
# - as backup acceptances, if something has no matches within any cutoffs, accept an allowance == 0 match, if something has none of either of those then accept a dallowance == 0 match

#bigger picture of the implementation:
#starting with the best ones, ie everything that's accepted in order
#then when a dist doesn't match all of its isos, leave the unmatched ones up for grabs
#if you find things that can grab them, indicate them as different distributions
#^i don't want to implement this, it causes a lot of complications and i think a simpler approach is what i write far below about re-working the distribution-assembly model, steplimit and newinclimit are not together complex enough
#which means now its time to assemble my cutoffs and work off of those

#dmaxcutoffs = [allowancebylrangedmaxcutoff, dsizebylrangedmaxcutoff, dallowancebylrangedmaxcutoff]
areacutoffs = [allowancebylrangeareafloatcutoff, dsizebylrangeareafloatcutoff, dallowancebylrangeareafloatcutoff]
#dmaxcutoffs = sorted(dmaxcutoffs)
areacutoffs = sorted(areacutoffs)

#mid = 0
matchallowances = defaultdict(list) #allowance: mid
#matchpairs = {} #mid: [lk, dk]
#dlens = Counter()
#maxallowances = defaultdict(list)
#for dk, lkeys in distributionmatches.items():
n = 0
matchallowances[n] = arrayout[arrayout[:,0] == 0, 5].astype(int).tolist()
n += 1
#matchallowances[n] = arrayout[arrayout[:,1] == 0, 5].astype(int).tolist()
#n += 1

#for acut, dcut in zip(areacutoffs, dmaxcutoffs):
for acut in areacutoffs:
    cutarray = arrayout[np.logical_and(arrayout[:,1] == 0, arrayout[:,4] < acut)]
#    #lkeys = lkeys[:,lkeys[1].argsort()]
#    ##minallowance = lkeys[1].min() + matchallowance
#    ##minallowance = np.unique(lkeys[1])[:matchallowance].max()
#    ##lkeys = lkeys[:,lkeys[1] <= minallowance]
#    ##distributionmatches[dk] = lkeys
#    #lset = sorted(set(lkeys[1].tolist()))
#    #lkeys = lkeys[:,lkeys[1] <= lset[0]]
#    #distributionmatches[dk] = lkeys
#    #dlens[len(lset)] += 1
#    #for n, l in enumerate(lset):
#    #    maxallowances[n].append(l)
#    matchallowances[n] = arrayout[arrayout[:,4] < acut,5].astype(int).tolist()
    matchallowances[n] = cutarray[:,5].astype(int).tolist()
    n += 1
    #matchallowances[n] = arrayout[arrayout[:,4] < dcut,6].astype(int).tolist()
    #n += 1
    #for lk, a in lkeys.transpose().tolist():
    #    matchallowances[a].append(mid)
        #matchpairs[mid] = tuple([lk, dk])
        #mid += 1
#allowancevalues = Counter(maxallowances.values())

#for n, vals in maxallowances.items():
#    plt.hist(maxallowances.values(), bins=np.sqrt(len(vals)).astype(int), label=n)
#plt.legend()
#plt.show()

#plt.bar(dlens.keys(), dlens.values())
#plt.show()


output = []
fulllibset = set()
fulldistset = set()
#for a in sorted(badallowances):
#    libset, distset = zip(*map(matchpairs.get, badallowances[a]))
#for a in sorted(equalallowances):
#    libset, distset = zip(*map(matchpairs.get, equalallowances[a]))
#for a in sorted(disassociatedallowances):
#    libset, distset = zip(*map(matchpairs.get, disassociatedallowances[a]))
for a in sorted(matchallowances):
    libset, distset = zip(*map(matchpairs.get, matchallowances[a]))
    libset = set(libset)
    distset = set(distset)
    scansum = sum(1 for i in distset if i in scansbyanalyte)
    libintersection = len(fulllibset.intersection(libset))
    distintersection = len(fulldistset.intersection(distset))
    fulllibset.update(libset)
    fulldistset.update(distset)
    analytescancounts = sum(1 for i in fulldistset if i in scansbyanalyte)
    scancounts = sum(len(scansbyanalyte[i]) for i in fulldistset if i in scansbyanalyte)
    output.append([a, len(libset), len(distset), libintersection, distintersection, scansum, len(fulllibset), len(fulldistset), analytescancounts, scancounts])

for o in output:
    print(' - '.join((map(str, o))))
print(sum(1 for i in fulldistset if i in scansbyanalyte), 'potential ids total')


#without requiring 0 in lintranks:
#0 - 754971 - 35228 - 0 - 0 - 18664 - 754971 - 35228 - 18664 - 61788
#1 - 1050927 - 39757 - 754486 - 35181 - 21616 - 1051412 - 39804 - 21642 - 77688
#2 - 871692 - 33167 - 728603 - 32256 - 18543 - 1194501 - 40715 - 22128 - 79714
#3 - 1253053 - 38372 - 1079308 - 37640 - 21062 - 1368246 - 41447 - 22377 - 80716
#4 - 1364696 - 39468 - 1291703 - 39234 - 21498 - 1441239 - 41681 - 22445 - 80841
#22445 potential ids total


#dlens = Counter()
#maxallowances = Counter()
#for dk, lkeys in badmatches.items():
#    lkeys = lkeys[:,lkeys[1].argsort()]
#    minallowance = lkeys[1].min() + matchallowance
#    lkeys = lkeys[:,lkeys[1] <= minallowance]
#    badmatches[dk] = lkeys
#    dlens[len(lkeys[0])] += 1
#    maxallowances[dk] = max(lkeys[1])
#
#plt.hist(maxallowances.values(), bins=len(set(maxallowances.values())))
#plt.title('bad')
#plt.show()
#
#plt.bar(dlens.keys(), dlens.values())
#plt.title('bad')
#plt.show()
#
#dlens = Counter()
#maxallowances = Counter()
#for dk, lkeys in equalmatches.items():
#    lkeys = lkeys[:,lkeys[1].argsort()]
#    minallowance = lkeys[1].min() + matchallowance
#    lkeys = lkeys[:,lkeys[1] <= minallowance]
#    equalmatches[dk] = lkeys
#    dlens[len(lkeys[0])] += 1
#    maxallowances[dk] = max(lkeys[1])
#
#plt.hist(maxallowances.values(), bins=len(set(maxallowances.values())))
#plt.title('equal')
#plt.show()
#
#plt.bar(dlens.keys(), dlens.values())
#plt.title('equal')
#plt.show()
#
#dlens = Counter()
#maxallowances = Counter()
#for dk, lkeys in disassociatedmatches.items():
#    lkeys = lkeys[:,lkeys[1].argsort()]
#    minallowance = lkeys[1].min() + matchallowance
#    lkeys = lkeys[:,lkeys[1] <= minallowance]
#    disassociatedmatches[dk] = lkeys
#    dlens[len(lkeys[0])] += 1
#    maxallowances[dk] = max(lkeys[1])
#
#plt.hist(maxallowances.values(), bins=len(set(maxallowances.values())))
#plt.title('disassociated')
#plt.show()
#
#plt.bar(dlens.keys(), dlens.values())
#plt.title('disassociated')
#plt.show()
#
#print('equal', equal)
#print('disassociated', disassociated)
#print('bad', bad)

n = 0
for dk, lkeys in distributionmatches.items():
    for lk, al in lkeys.transpose().tolist():
        if al >= 0:
            dmasses, dintensities = analytedistributions[dk]
            dsum = sum(dintensities)
            adjusteddistints = dintensities / dintensities.max()
            adjustedlibints = libraryintensities[lk] / libraryintensities[lk].max()
            plt.bar(dmasses, adjusteddistints, width=0.1, color='green', alpha=0.5)
            plt.bar(librarymassdict[lk], adjustedlibints, width=0.1, color='yellow', alpha=0.5)
            plt.title(' - '.join((str(lk), str(dk))))
            plt.show()
            print(lk, ',', dk)
            print(al)
            n += 1
    if n > 20:
        break

#number of libmatches should be > nonmatched lib isos
#but nonmatched is capped by number of dist isos, so if dist isos > lib isos, lib isos is the number
#elif distisos < libisos, then marched isos must be > number of non-matched dist isos?

#a majority of both -> yes
#lib in correct rank order -> yes
#dist intensity ranks can be re-maneuvered OTF, subtracting their max to re-order locally

#a minority of both -> no
#a minority of the dist -> if it matches all of the lib
#a minority of the lib -> if it matches all of the dist

#ok so i'm pretty sure when allowance < mallowance those are worse matches
#its something that had a better match before realignment, but if the realignment doesn't improve it then the matches are shit, ie a 2-1 becomes 2-0, 2-1 was "bad" here to begin with
#so ditch these, plot a lil more first

#when mallowance < allowance, these are dissasociated matches as the brunt of the experimental dist has been dissasociated with its other isos, perhaps use these differently

#for dists, favor equals over disassociations

#maxes match -> accepted +/- allowance
#
#for disassociations -> lib max matches
#disassociations only accepted when equals aren't available for that dist

#THE WHOLE PROBLEM is too difficult for just ranks

#nah i'm claiming success now with this
#BUT, the next step would be to label disassociations within dist matches now
#different library matches, basically on high allowance matches, don't necessarily share the same dist isotopomers, so there's room for these groups to be checked as independent distributions


#saverloc = '/'.join((processinglocation, 'badmatches.matches.pickle'))
#with open(saverloc, 'wb') as pick:
#    pickle.dump(badmatches, pick)
#
#saverloc = '/'.join((processinglocation, 'badmatches.pairs.pickle'))
#with open(saverloc, 'wb') as pick:
#    pickle.dump(matchpairs, pick)
#
#saverloc = '/'.join((processinglocation, 'badmatches.metrics.pickle'))
#with open(saverloc, 'wb') as pick:
#    pickle.dump(matchmetrics, pick)

#i don't think i need this
#saverloc = '/'.join((processinglocation, 'badmatches.conjoined.pickle'))
#with open(saverloc, 'wb') as pick:
#    pickle.dump([distributionmassdict, distributionintensities], pick)


#nonmatchedcharges = defaultdict(int)
#for dk in nonmatched:
    #analytekeys is  distid: charge now
#    charges = list(analytekeys[dk].keys())
#    for c in charges:
#        nonmatchedcharges[c] += 1
#


matches = set()
matchsets = {} #allowance: [all mids within their best rank of acceptance]
for k, v in matchallowances.items():
    hits = set(v).difference(matches)
    matchsets[k] = list(hits)
    matches.update(v)

batchnumber = 10
#for mid in random.sample(badallowances[0], batchnumber):
#for mid in random.sample(equalallowances[0], batchnumber):
#for mid in random.sample(disassociatedallowances[0], batchnumber):
for mid in random.sample(matchsets[0], batchnumber):
    lk, dk = matchpairs[mid]
#for dk in random.sample(list(badmatches), batchnumber):
#    lk = random.sample(badmatches[dk].tolist(), 1)[0]
#for k in random.sample(list(nonallowedpairs), batchnumber):
#    lk, dk = nonallowedpairs[k]
#for k in random.sample(list(nonmaxpairing), batchnumber):
#    lk, dk = nonmaxpairing[k]
#for k in random.sample(list(lonesignalmatches), batchnumber):
#    lk, dk = lonesignalmatches[k]
    dmasses, dintensities = analytedistributions[dk]
    dsum = sum(dintensities)
    adjusteddistints = dintensities / dintensities.sum()
    adjustedlibints = libraryintensities[lk] / libraryintensities[lk].sum()
    plt.bar(dmasses, adjusteddistints, width=0.1, color='green', alpha=0.5)
    plt.bar(librarymassdict[lk], adjustedlibints, width=0.1, color='yellow', alpha=0.5)
    plt.title(' - '.join((str(lk), str(dk))))
    plt.show()
    print(lk, ',', dk)

#msrun = mzml.MzML(mzmlfile)
#ms1scans = 0
#ms2scans = 0
#for scan in msrun:
#    if scan['ms level'] == 1:
#        ms1scans += 1
#    else:
#        ms2scans += 1

#i may be able to priority rank which analytes and scans have distributions worth identifying from this
#i don't need to care about high allowance matches to something that has top-shelf matches
#but if scans and analytes have no top-shelf picks, then take what's left i suppose
#it would be good to be able to split up a distribution by different library matches if those library matches don't overlap in isotopomers. because the entropy might get screwy, maybe this can be determined via allowance?

#but for now i'll just wrap this up by assigning ~n layers of matches to as many things as possible, and the layers must be adjacent, naturally it would start with the lowest

#mid: lk
#dk: mid
#mid: allowance
#you care most about what matches best to an experimental dist
#if prior allowance levels seen for an experimental dist leave certain isotopomer untouched, they can be grabbed by other matches as other potential dists
#and it might be easier to just designate multiple dists being in there? i guess?


#VISUALIZE NOW


#i think these might be ok, but i want to be convinced by seeing the full spectra
#from lonesignalmatches
#1220571, 32352
#29320, 29451

#these should be alright, why is it failing?
#nonallowedpairs
#107103, 32191
#1152620, 17404
#449113, 7159
#1476841, 1649
#but make sure something like this still fails:
#639024, 35820
#154102, 35681 i guess? i think this could be ok

#nonmaxpairing is pretty agreeable, visually
#that spectra would be convincing though

#passing pairs:
#1290335, 31098, kind of questionable, maybe i should know if the left-most of a library dist is monoisotopic?
#39107, 12997 not looking that hot?
#347056, 38604 - same theme, there's an experimental dist left of the library dist, it gives a strange feeling
#108501, 26413 - hypothetically questionable, but the reason i chose to match things like this is that my dist assembly might not be perfect, i'll accept this for now.


dk = 1145
lk = 648831

lk = 705747
dk = 1490
#^these two are a great indicator of why i both need better time-bin separation of longer than normal lines and to differentiate lines away from a distribution when you can pin them to a separate separate charge state distribution. and yes that's 2 separates.
#^actually i might go for the time-bin thing for individual lines, but I want to separate trailing pieces of a distribution from the main dist prior to charge-handling instead
#i think the best fix for this is a new distribution model, replacing steplimit and newinclimit - which is feasible, but i'll be doing it later
#1400-1415 m/z for charge 2
#930-946 m/z for charge 3
#retentio times 162-176

#both are based on the maximum of any dist, so smaller dists are extended more
mscale = 1.1 #percent beyond mass bounds to extend, 1.1 is 110%
tscale = 1.1 #percent beyond rt bounds

#regions as [minmass, maxmass, mintime, maxtime, # datapoints, peakarea, maxintensity, wmean, lineid]
#chargeregions = [mincharge, maxcharge, mintime, maxtime, len(v), area, maincharge, k]

cmintime = np.inf
cmaxtime = 0
chargemrange = Counter() #charge: mass range
for dist, charge in analytekeys[dk].items():
    #library factors
    libmasses = librarymasses[lk]
    libints = libraryintensities[lk]
    #
    lines = linesofdistributions[dist]
    cminmass = np.inf
    cmaxmass = 0
    for line in lines:
        minmass, maxmass, mintime, maxtime, ndp, pa, mi, wm, lid = regions[line]
        if minmass < cminmass:
            cminmass = minmass
        if maxmass > cmaxmass:
            cmaxmass = maxmass
        if mintime < cmintime:
            cmintime = mintime
        if maxtime > cmaxtime:
            cmaxtime = maxtime
    cmaxmass = cmaxmass * charge - proton * charge
    cminmass = cminmass * charge - proton * charge
    cmassrange = cmaxmass - cminmass
    chargemrange[charge] = cmassrange

maxmassradius = chargemrange.most_common(1)[0][1] * mscale / 2
maxtimeradius = (cmaxtime - cmintime) * tscale / 2
centralmass = analytedistributions[dk][0].mean()
centraltime = np.mean([cmintime, cmaxtime])

lmasses = librarymassdict[lk]
lints = libraryintensities[lk]

for dist, charge in analytekeys[dk].items():
    lplotmasses = (lmasses + (proton * charge)) / charge
    centralchargemass = (centralmass + (proton * charge)) / charge
    lines = linesofdistributions[dist]
    boundrec = [centralchargemass-maxmassradius, centralchargemass+maxmassradius, centraltime-maxtimeradius, centraltime+maxtimeradius]
    plotkeys = arg_coord_rectangle_overlap(boundrec, regions[:,:4]).tolist()
    nodistkeys = []
    dists = defaultdict(set)
    for p in plotkeys:
        try:
            dists[distributionsoflines[p]].update(linesofdistributions[distributionsoflines[p]])
        except KeyError: #line is in nodists
            nodistkeys.append(p)
    ngroups = len(dists)
    cols = dp.get_colors(ngroups)
    fig, ax = plt.subplots(nrows=3, figsize=(6,8), sharex=True)
    for n, (vdist, vlines) in enumerate(dists.items()):
        ax[1].hlines(n, regions[list(vlines)][:,0].min(), regions[list(vlines)][:,1].max(), color=cols[n], linewidth=0.6)
        for vline in vlines:
            vreg = regions[vline]
            a = trackedgroups[vline]
            ax[0].bar(vreg[7], vreg[5], width=0.02, alpha=0.4, color=cols[n])
            ax[2].scatter(a[0], a[1], marker='.', color=cols[n], s=0.3, alpha=0.3)
            ax[2].plot(a[0], a[1], '-', color=cols[n], linewidth=0.2, alpha=0.8)
            ax[1].vlines(vreg[7], n-0.1, n+0.1, color=cols[n], linewidth=0.6)
            #ax[1].text(vreg[7], n+0.1, vreg[4], fontsize=4, ha='center', color='white')
    for vline in nodistkeys:
        vreg = regions[vline]
        a = trackedgroups[vline]
        ax[0].bar(vreg[7], vreg[5], width=0.02, alpha=0.4, color='white')
        ax[2].scatter(a[0], a[1], marker='.', color='white', s=0.3, alpha=0.3)
        ax[2].plot(a[0], a[1], '-', color='white', linewidth=0.2, alpha=0.8)
    nx = ax[0].twinx()
    nx.bar(lplotmasses, lints, color='black', width=0.02, alpha=0.3)
    ax[0].plot(regions[lines][:,7], regions[lines][:,5], '*', color='white', markersize=0.8, alpha=0.8)
    ax[0].set_ylabel('intensity')
    ax[2].set_ylabel('minutes')
    ax[1].set_ylabel('distribution')
    ax[2].set_xlabel('m/z')
    plt.suptitle(charge)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)
    plt.show()
    fig.clf()
    plt.close()
#define the region around every charge state, because some charge states end up truncated erroneously
#use that to define the mass regions around each charge state
#pick up lines and make plotkeys for each charge state
#plot everything

nahs = set(scansbyanalyte).difference(fulldistset)
test = Counter(len(matchorganizer[d]) for d in nahs)
#^dk of 0 is at the top of this
#^i should plot this spectra, it looks ~fine but doesn't match what a dist of that mass should have in shape


##key plotting
#lm = librarymassdict[lk]
#dm = distributionmassdict[dk]
#
#li = libraryintensities[lk]
#di = distributionintensities[dk]
#
#lsum = sum(li)
#dsum = sum(di)
#
#li = [i/lsum for i in li]
#di = [i/dsum for i in di]
#
#plt.bar(lm, li, width=0.1, color='green', alpha=0.5)
#plt.bar(dm, di, width=0.1, color='yellow', alpha=0.5)
#plt.show()



#next i should visualize positives and negatives
#next would be finding common distribution distances to other distributions and preparing that for MS2's that I need to check later.
#   >^ if a modification is directly evident, the ms1 distributions should basically match, so that's something you can use as verification i suppose - but visualize
#visualize the analytedistributions that have no matches

#batchnumber = 10
#for k in random.sample(list(nonallowedpairs), batchnumber):
##for k in random.sample(list(matchpairs), batchnumber):
#    #lk, dk = matchpairs[k]
#    lk, dk = nonallowedpairs[k]
#    if lk in sumabundances:
#        lmasses, lintensities = sumabundances[lk]
#    else:
#        lmasses, lintensities = maxabundances[lk]
#    dmasses, dintensities = analytedistributions[dk]
#    lsum = sum(lintensities)
#    dsum = sum(dintensities)
#    lintensities = [i/lsum for i in lintensities]
#    dintensities = [i/dsum for i in dintensities]
#    plt.bar(lmasses, lintensities, width=0.1, color='yellow', alpha=0.5)
#    plt.bar(dmasses, dintensities, width=0.1, color='green', alpha=0.5)
#    plt.title(' - '.join((str(lk), str(dk))))
#    plt.show()

#what about something crazy like intensity-based matching, you would need back-up charge lists, and this could potentially help identify things as they should be...?

#instead, the below is going to be like charge matching, make a single array of every isotopicabundance distribution mass, and every single distributionregion mass, match them all, I just need a radius I suppose
#the intensityranks can work off of relative ranks - for whatever matched - instead of the full distribution hierarchy.
#any generated cutoff here could also work for the precursordistances that are off target b/c of monoisotopic bullshit

#distribution charge corrections:
#bring ranked charges from preservedranks as they pass for each line
#ie if the matched pair is the first of its mention, then it gets that charge. record all other charges for each line, as they come off of pair charges - if it's a new charge to be seen for that line so far in the ranks. these charges will also come off in a ranked order.
#^this can drag into some pretty memory-heavy baggage, i might just grab an extra charge or two, like the first two that pop up i suppose





#you can group the distributions by rank first, then use those rank-groups as a faster way of matching to found distributions
#you'll be able to do a sum-max comparison, there should be a mismatch between some of these, but i suppose just generate both metrics when the sum and max match ranks
#you should be able to minimize the badmatches the same way, matching rank orders make a group
