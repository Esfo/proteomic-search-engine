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
from multiprocessing.pool import ThreadPool
from collections import Counter, defaultdict
import concurrent.futures
import threading
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
#proton = 1.00727647

#all the memory management for shit, nothing can be made concurrent

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

environment_partial = partial(lmdb.Environment, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=99999, max_spare_txns=1, lock=True)

#312s for single linear execution
#600s for dual linear with no metrics
#769s for duallinear with metrics

def match_making(environment_partial, librarylocation, proteome, processinglocation):
    #formalized analyte information, summarizing all distributions across any charge states
    nt = time()
    analytefile = '/'.join((processinglocation, 'analytefactors.pickle'))
    with open(analytefile, 'rb') as pick:
        analytekeys, analytedistributions, analytesbydistribution, distributionsoflines, linesofdistributions = pickle.load(pick)
    #analytekeys = defaultdict(dict) #analyte id: distid: charge, this can be one or multiple distributions based on whether it has multiple charge states
    #analytedistributions = defaultdict(dict) #analyte id: ordered masses: AUC of merged isotopomers, weighted means [via intensity] across isotopomers from every charge state, isotopomer datapoints merged across each charge state - if there are any
    #analytesbydistribution = {} #distid: analyte id
    
    distributionkeys = []
    distributionmasses = []
    distributionmassdict = {} #did: [masses]
    distributionintensities = {} #did: [intensities]
    distributionintensityranks = {} #did: [intensityranks]
    distributionadjacentintensityratios = {} #did: [you get it...]
    distributionadjacentintensityratioranks = {} #this can just be an argsort
    for k, (masses, intensities) in analytedistributions.items():
        distributionmassdict[k] = masses
        distributionintensities[k] = intensities
        intensityranks = intensities.argsort()[::-1]
        distributionintensityranks[k] = intensityranks
        intensityratios = intensities[:-1] / intensities[1:]
        distributionadjacentintensityratios[k] = intensityratios
        intensityratioranks = intensityratios.argsort() #directionality doesn't matter
        distributionadjacentintensityratioranks[k] = intensityratioranks
        distributionkeys.extend(itertools.repeat(k, masses.size))
        distributionmasses.extend(masses.tolist())
    
    distributionkeys = np.array(distributionkeys)
    distributionmasses = np.array(distributionmasses)
    
    getkeys = []
    formulaidentifiers = {}
    with environment_partial(librarylocation) as env:
        formuladb = '.'.join(('formulaidentifier', proteome))
        formulas = env.open_db(formuladb.encode())
        with env.begin(write=False) as txn:
            with txn.cursor(formulas) as cursor:
                for k, v in cursor:
                    formulaidentifiers[k.decode()] = int(v.decode())
                    getkeys.append(k)
    print(time() - nt, 'initiated')
    
    matchedindices = []
    matchedindices.extend(distribution_mode_processing(environment_partial, getkeys, librarylocation, 'distributions.sum', 0, formulaidentifiers, distributionmasses, distributionmassdict, distributionintensityranks, distributionkeys, distributionadjacentintensityratios, distributionintensities))
    #matchedindices.extend(distribution_mode_processing(environment_partial, getkeys, librarylocation, 'distributions.max', 1, formulaidentifiers, distributionmasses, distributionmassdict, distributionintensityranks, distributionkeys, distributionadjacentintensityratios, distributionintensities))
    return matchedindices

def distribution_mode_processing(environment_partial, getkeys, librarylocation, distributiontype, additionfactor, formulaidentifiers, distributionmasses, distributionmassdict, distributionintensityranks, distributionkeys, distributionadjacentintensityratios, distributionintensities):
    nt = time()
    
    abundances = {}
    with environment_partial(librarylocation) as env:
        distributions = env.open_db(distributiontype.encode())
        with env.begin(write=False) as txn:
            with txn.cursor(distributions) as cursor:
                for k, v in cursor.getmulti(getkeys):
                    out = np.frombuffer(v)
                    out = out.reshape(2, out.size//2)
                    abundances[k.decode()] = out
    
    librarykeys = []
    librarymasses = []
    librarymassdict = {} #lid: [masses]
    libraryintensities = {} #lid: [intensities]
    libraryintensityranks = {} #lid: [intensityranks]
    libraryadjacentintensityratios = {} #lid [you get it...]
    libraryadjacentintensityratioranks = {} #this can just be an argsort
    
    for f, (masses, intensities) in abundances.items():
        k = formulaidentifiers[f] + additionfactor #1 for maxes
        librarymassdict[k] = masses
        libraryintensities[k] = intensities
        intensityranks = intensities.argsort()[::-1]
        libraryintensityranks[k] = intensityranks
        intensityratios = intensities[:-1] / intensities[1:]
        libraryadjacentintensityratios[k] = intensityratios
        intensityratioranks = intensityratios.argsort() #directionality doesn't matter
        libraryadjacentintensityratioranks[k] = intensityratioranks
        librarykeys.extend(itertools.repeat(k, masses.size))
        librarymasses.extend(masses.tolist())
    
    librarykeys = np.array(librarykeys)
    librarymasses = np.array(librarymasses)
    
    radius = distributionmasses / 1000000 * ppmallowance
    
    #switch the train to distmatches?
    lmtree = spatial.KDTree(librarymasses[:,None])
    matches = lmtree.query_ball_point(distributionmasses[:,None], radius, workers=8)
    
    #this alone doesn't improve speed, but it prevents duplicate matches - in some ways i think, so in one sense it takes memory to make but saves some later i guess? sure
    matchorganizer = defaultdict(list)
    for dk, lkeys in zip(distributionkeys, matches):
        matchorganizer[dk].extend(lkeys)
    
    for k in list(matchorganizer):
        matchorganizer[k] = np.array(list(set(matchorganizer[k])))
    
    print(time() - nt, 'matchorganizer assembled', len(matchorganizer))
    nt = time()

    #lmatches = 0
    #dmatches = 0
    ##librarymatches = defaultdict(list) #librarykey: [distributionkeys]
    #distributionmatches = defaultdict(list) #distributionkey: [librarykeys]
    ##matchpairs = {} #matchid: (librarykey, distkey)
    #matchpairs = defaultdict(dict) #distkey: librarykey: matchid
    #failpairs = {} #for plotting negatives
    #failcatch = 0.3 #probability of preserving any given failure, it's random because there would be too many
    #matchmetrics = {} #matchid: [....]
    #mid = 0
    #fid = 0
    #matchedindices = []
    #for dk, lkeys in matchorganizer.items():
    #    dmasses = distributionmassdict[dk]
    #    dsize = dmasses.size
    #    match_assessment_partial = partial(match_assessment, dmasses, dsize, dk)
    #    #tx = 0
    #    with concurrent.futures.ThreadPoolExecutor() as executor:
    #        futures = []
    #        for lkey in lkeys.tolist():
    #            lk = librarykeys[lkey]
    #            futures.append(executor.submit(match_assessment_partial, lk, librarymassdict[lk], libraryintensityranks[lk], distributionintensityranks[dk]))
    #        for f in concurrent.futures.as_completed(futures):
    #            out = f.result()
    #            match out:
    #                case tuple():
    #                    matchedindices.append(out)
    
    matchedindices = []
    for dk, lkeys in matchorganizer.items():
        dmasses = distributionmassdict[dk]
        dsize = dmasses.size
        match_assessment_partial = partial(match_assessment, dmasses, dsize, dk, distributionadjacentintensityratios[dk], distributionintensityranks[dk], distributionintensities[dk])
        #tx = 0
        #args = []
        #outs = []
        for lkey in lkeys.tolist():
            lk = librarykeys[lkey]
            #args.append([lk, librarymassdict[lk], libraryintensityranks[lk], distributionintensityranks[dk]])
            #out = match_assessment_partial(lk, librarymassdict[lk], libraryintensityranks[lk], libraryintensities[lk], libraryadjacentintensityratios[lk])
            #matchedindices.append(pool.apply(match_assessment_partial, args=(lk, librarymassdict[lk], libraryintensityranks[lk], libraryintensities[lk], libraryadjacentintensityratios[lk])))
            out = match_assessment_partial(lk, librarymassdict[lk], libraryintensityranks[lk], libraryintensities[lk], libraryadjacentintensityratios[lk])
            if out is not None:
                matchedindices.append(out)
                #matchedindices.append(pool.apply(match_assessment_partial, args=(dmasses, dsize, dk, distributionadjacentintensityratios[dk], distributionintensityranks[dk], distributionintensities[dk], lk, librarymassdict[lk], libraryintensityranks[lk], libraryintensities[lk], libraryadjacentintensityratios[lk])))
                #if out is not None:
                #    matchedindices.append(out)
        #for out in outs:
        #    matchedindices.append(out.get())
        
        #with mp.Pool(nprocs) as pool:
        #    for matches in pool.starmap(match_assessment_partial, args):
        #        if matches:
        #            matchedindices.append(matches)
    print(time() - nt, 'matches finished')
    return matchedindices

def match_assessment(dmasses, dsize, dk, dadjacentintensityratios, dintensityranks, dintensities, lk, lmasses, lintensityranks, lintensities, ladjacentintensityratios):
    #if lk not in dmatches: #missing the repeat hits that naturally occur from the nearestneighbor process used above, actually i'm not sure how this happens because i though matchorganizer prevented it
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
    
    lintranks = lintensityranks[li:le]
    if 0 in lintranks: #the top rank is included
        adjacentmaxsize = maxsize - 1
        
        #it would be good to be able to snip off ends that don't work here maybe?
        lintranks = sorted(lintranks.tolist())
        lintdiffs = [lintranks[i+1] - lintranks[i] for i in range(adjacentmaxsize-1)]
        
        #ie, it should be matching the meat of the librarymatch
        if lintdiffs and max(lintdiffs) == 1: #those that come after the top rank are in the natural/desired sequential order, checking if lintdiffs exists means there should be more than 1 match, and prevents a valuerror
            re = ri + maxsize
            #below basically allows all adjacent intensity ratio ranks to have a maximum of 1-rank difference, and one rank can have a 2. it allows ranks to shift, as a whole, while not allowing things to lose their relative order too much.
            #^now it just allows 1 difference
            
            #dorders = (dintslice[:-1] / dintslice[1:]).argsort()
            #lorders = (lintslice[:-1] / lintslice[1:]).argsort()
            #dorders = distributionadjacentintensityratioranks[dk][ri:reaj].tolist()
            #lorders = libraryadjacentintensityratioranks[lk][li:leaj].tolist()
            dorders = dintensityranks[ri:re].tolist()
            lorders = lintensityranks[li:le].tolist() #this is the same as lintranks above.. technically, but I'll just keep this here for now because that one gets messed with a little
            #^this lower try got less matches, I suppose that means its going in the wong direction in terms of flexibility
            
            #orderdiffs = np.abs(dorders - lorders)
            orderdiffs = [abs(i-j) for i, j in zip(dorders, lorders)]
            
            #ndiffs = (orderdiffs > 0).sum()
            #ndiffs = sum((orderdiffs > 0).tolist())
            #ndiffs = sum(i > 0 for i in orderdiffs)
            #^going to stop using this because matches end up being too flexible! it needs more stringency
            
            #allowance = orderdiffs.sum() - ndiffs - 1
            #allowance = sum(orderdiffs.tolist()) - ndiffs - 1
            #allowance = sum(orderdiffs) - ndiffs - 1
            allowance = sum(orderdiffs) - 1
            
            #I also need to make sure that the librarymatch has its top players matched in the game - any mismatch in order of the librarymatch rank order to the distmatch means that distmatch isotopomer won't be considered, and the logic will follow suit
            if allowance <= 0:
                reaj = ri + adjacentmaxsize
                leaj = li + adjacentmaxsize
                #consistency of proton-dists among the distmatches compared to all the others, this would include a shadow-like component towards non-matched distlines if their distances aren't consistent with what's matched
                #but when it comes to consistency of mass distances.. do i even want this? this doesn't necessarily make sense because some distributions might be slightly close in distance or slightly farther away due to slightly different atomic compositions, but i think same-shaped distributions at similar masses might not necessarily be a better match just because it's a little closer, idk how the ppm error is going to play out yet
                #intensity differences are definitely acceptable
                dintslice = dintensities[ri:re]
                lintslice = lintensities[li:le]
                
                dmslice = np.array(dmasses[ri:re])
                lmslice = np.array(lmasses[li:le])
                
                #dnorm = dintslice / dintslice.sum()
                #lnorm = lintslice / lintslice.sum()
                dnorm = dintslice / sum(dintslice.tolist())
                lnorm = lintslice / sum(dintslice.tolist())
                
                intensitydiff = dnorm - lnorm
                idmean = sum(intensitydiff.tolist()) / maxsize
                #intensitydiffs = np.abs(intensitydiff - idmean)
                intensitydiffs = [abs(idmean - i) for i in intensitydiff.tolist()]
                meanintensitydiff = sum(intensitydiffs) / maxsize
                
                #diaj = dnorm[:-1] / dnorm[1:]
                #liaj = lnorm[:-1] / lnorm[1:]
                diaj = dadjacentintensityratios[ri:reaj]
                liaj = ladjacentintensityratios[li:leaj]
                iajdiff = diaj - liaj
                iajmean = sum(iajdiff.tolist()) / adjacentmaxsize
                #intensityadjacentdiffs = np.abs(iajdiff - iajmean)
                intensityadjacentdiffs = [abs(iajmean - i) for i in iajdiff.tolist()]
                intensityadjacency = sum(intensityadjacentdiffs) / adjacentmaxsize
                
                #rewrites:
                #abs function is fine
                #change diff to [1:] - [:-1]
                #sums to sum(tolist())
                #generate any means as a sum(tolist()) / size
                
                massdiff = lmslice - dmslice
                mdmean = sum(massdiff.tolist()) / maxsize
                #massdiffs = np.abs(massdiff - mdmean)
                massdiffs = [abs(mdmean - i) for i in massdiff.tolist()]
                meanmassdiff = sum(massdiffs) / maxsize
                
                #these are probably fine, but could be list comprehensions, only seemed to save ~100ns, it might matter because this has bajilions of iterations
                dadjdiffs = dmslice[1:] - dmslice[:-1]
                ladjdiffs = lmslice[1:] - lmslice[:-1]
                adjacentdiffs = dadjdiffs - ladjdiffs
                adjmean = sum(adjacentdiffs.tolist()) / adjacentmaxsize
                #massadjacenctdiffs = np.abs(adjacentdiffs - adjmean)
                massadjacentdiffs = [abs(adjmean - i) for i in adjacentdiffs.tolist()]
                massadjacency = sum(massadjacentdiffs) / adjacentmaxsize
                output = np.array([dk, lk, meanintensitydiff, intensityadjacency, meanmassdiff, massadjacency, maxsize])
                return output

nt = time()

matches = np.array(match_making(environment_partial, librarylocation, proteome, processinglocation))

print(time() - nt, 'total')
nt = time()

idn = 0
matchmetrics = {} #matchid: [....]
matchpairs = defaultdict(dict) #distkey: librarykey: matchid
distributionmatches = defaultdict(list) #distributionkey: [librarykeys]
for dk, lk, meanintensitydiff, intensityadjacency, meanmassdiff, massadjacency, maxsize in matches:
    matchpairs[dk][lk] = idn
    matchmetrics[idn] = np.array([meanintensitydiff, intensityadjacency, meanmassdiff, massadjacency, maxsize])
    distributionmatches[dk].append(lk) #check for non-redundancy with sets quick
    idn += 1

for dk in list(distributionmatches):
    distributionmatches[dk] = np.array(distributionmatches[dk], dtype=int)

print(time() - nt, 'processed')
nt = time()

saverloc = '/'.join((processinglocation, 'distributionmatches.matches.pickle'))
with open(saverloc, 'wb') as pick:
    pickle.dump(distributionmatches, pick)

saverloc = '/'.join((processinglocation, 'distributionmatches.pairs.pickle'))
with open(saverloc, 'wb') as pick:
    pickle.dump(matchpairs, pick)

saverloc = '/'.join((processinglocation, 'distributionmatches.metrics.pickle'))
with open(saverloc, 'wb') as pick:
    pickle.dump(matchmetrics, pick)

print(time() - nt, 'saved')

#ultimately, i don't think ms1 match metrics are going to be helpful because i don't produce the entire distribution, nor might i produce something entirely similar to what it should look like, so this might not even be relevant
