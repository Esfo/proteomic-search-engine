import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from colour import Color
from pyteomics import mzml
from time import time
import pandas as pd
import lmdb
from functools import partial
import gc
import concurrent.futures
import multiprocessing as mp
from multiprocessing.managers import BaseManager, DictProxy
from collections import Counter, defaultdict
from textwrap import wrap
from scipy import sparse, integrate, spatial, stats, special
from pandas.api.types import CategoricalDtype
from sklearn.neighbors import NearestNeighbors
from more_itertools import sort_together
from distinctipy import distinctipy as dp
import math
import sqlitedict as sq
import random
import itertools
import pickle
import sys
import os
import warnings
#warnings.filterwarnings("error")
np.set_printoptions(suppress=True)
#np.warnings.filterwarnings('ignore') #depricated i guess
np.testing.suppress_warnings(forwarding_rule='always')
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


#loaderloc = '/store/flowcharacterizations/round3/DDAs/fileprocessing/200901_fR_400.pickle'
#with open(loaderloc, "rb") as pick:
#    regions, trackedgroups, modeltracking, timearra, roundcutoff = pickle.load(pick)

mzmlfile = '/home/sfo/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
#mzmlfile = '/home/sfo/store/data/PXD051214/mzMLs/JMM-6.mzML'
#mzmlfile = '/store/flowcharacterizations/round3/mzMLs/200901_1s-dyn-300-200_R0.mzML'
#mzmlfile = '/store/flowcharacterizations/round5/mzMLs/20210312_E5_CG_high_tw1.mzML'
#mzmlfile = '/home/sfo/store/data/PXD035665/mzMLs/GradientAmount_HeLa_4ug_240min_R01.mzML'

basefolder = '/'.join((mzmlfile.split('/')[:-2]))
basefile = mzmlfile.split('/')[-1].split('.mzML')[0]

nprocs = os.cpu_count()
processinglocation = '/'.join((basefolder, 'fileprocessing', basefile))

librarylocation = '/home/sfo/data/proteomics/fastas/search-db'
#librarylocation = '/home/sfo/data/proteomics/fastas/iodoacetamide-search'
proteome = 'Human_Homo_sapien'

environment_partial = partial(lmdb.Environment, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=99999, max_spare_txns=1, lock=True)

getkeys = []
formulaidentifiers = {} #formula: integer key
distributionidentifiers = {} #integer key: formula
sumabundances = {}
formuladb = '.'.join(('formulaidentifier', proteome))
with environment_partial(librarylocation) as env:
    parameters = env.open_db('isofactors'.encode())
    with env.begin(write=False) as txn:
        with txn.cursor(parameters) as cursor:
            parameterbytes = cursor.get(proteome.encode())
            parameterdict = dict(eval(parameterbytes.decode()))
            subisomax = float(parameterdict['subisomax'])
            uppermasslimit = float(parameterdict['uppermasslimit'])
    formulas = env.open_db(formuladb.encode())
    with env.begin(write=False) as txn:
        with txn.cursor(formulas) as cursor:
            for k, v in cursor:
                formula = k.decode()
                libkey = int(v.decode())
                formulaidentifiers[formula] = libkey
                distributionidentifiers[libkey] = formula
                getkeys.append(k)
    sums = env.open_db('distributions.sum'.encode())
    with env.begin(write=False) as txn:
        with txn.cursor(sums) as cursor:
            for k, v in cursor.getmulti(getkeys):
                out = np.frombuffer(v)
                out = out.reshape(2, out.size//2)
                sumabundances[k.decode()] = out

newinclimit = 0.1
steplimit = 0.5
ppmtolerance = 20

proton = 1.00727647

minmovinginds = 10
deadsignal = 20 #number of scans without data
#deadsignal = 0.17 #minutes, ~10 seconds
#^there might be potential to soft-code this via times taken from scan intervals

minpoints = 3
chargetolerance = 0.1 #lesson learned: these differences DO get divided across charge states, if you normalize everything back to base mass without a charge, the errors become more consistent. They're smaller errors for higher charges etc. so going by percent here is FINE!

subisomax = subisomax + subisomax * chargetolerance


#
#Peak finding parameters
#


#currently:
#subisogroups
#after, take a look at making all the ranking stuff concurent if it can be, if not, whatever.
#test out the non-overlap negative RT overlap concept to see how it plays out
#did nodists have any MS2 hits? do I account for this?
#correlate integration across normal signal vs injection-time normalized vs % of scan, which ones have the most least difference? this might be an interesting exploration.

#^save these for later, get a main.py setup that processes some files and work ms2 id's off of the standardized output

#
#Notes
#

#re-engineering ideas:
#if you want to try improving the line model, the direction to look might be to allow redundant signals to overlap on datapoints. This could potentially replace the line overlaps. When they hit the same datapoint (as nearest), they would merge, given that they have no redundant timepoints. This would just be free-flowing.
#it would be done by training the model on the new masses and matching to the live ones, if live ones all match the same point, they're put up for merge. If they fail to merge once, just implement a blocking dict that ignores the match somehow so this doesn't go over the same shit for every new scan.
#an experimental ranking, simply look at the count of the number of times a charge appears for each individual line, the sum of those counts would be that lines individual ranking. within each of its charges would need to be a sub-ranking of pairings.

#looking at a lot of this, I think all nodist matches are on the lowest charge, if so - then that's an ionization directionality I suppose


#todo:
#independent nodist-to-nodist deconvolution - low priority, less than 100 matched to actual distributions on the test file, probably wouldn't get much here
#subiso merging
#a multi-peak to single-peak reduction process for long lines
#I need to collect spatial information from the charge state data, ie distinct errors, adjacent intensity percentages, rt overlaps, etc
#make a centroiding process
#I'll need a cross-file validation scheme for MS1-predicted distributions from one file that have MS2 scans in others. Or just a scheme to match MS2 identified peptides to their untargeted counterparts in other files.
#visualize the non-matching charge-state nominees, things that dont pass closeordermatching
#^visualize the worst ppm error charge matches and worst experror dist matches
#visualize: does more intensity == more charge states? I doubt it

#potential charge-state matching:
#higher area charge matches should have more isotopomers, if not then have a strict non-close-order matching requirement, the intensity ranks must be exact matches
#dense order ranking, an intensity-rank threshhold concept for densely-ranked ions - for when two ions seem to go  back and forth in rank on different charge states

#profile a much simpler line mode loop and see if it's faster + if the line corrections solve most problems?
#cross-file distribution matching, do at the same time as MS2's I suppose

#questions:
#does a deuterated isotopomer tend to go towards a lower charge state? I keep seeing this RT shift on a weak signal that doesn't match a higher charge state, it's weird
#^in regards to the ~slight~weird rt mismatch across RT's, I don't expect ionization to be a pefectly linear process, whereby ionizing more or less of something always shows the same (especially considering background ionizers) proportion of 2+ to 3+ etc, it might be a process that spawns some 3+ in greater quantity after a certain intensity/number of ions are reached. And this could easily cross over from the early to mid-early peak shape. I think this is a fair explanation. You could also reason this could be an ordered generative process, whereby starting small and increasing [as the peak would] that you could generate differing numbers of either charge.

#what is the typical charge/length of the highest charge for something with multiple?
#what is the typical charge of the longest length distribution? maybe >2 charges?
#what is the typical charge/length of the largest peak areas?
#^these could make for decent cross-file comparisons, maybe make it more complex by assesing total 3+ area vs total 2+ area, and the ratios of those + all other groups, and their counts, how total counts / sum areas looks across files, etc

#when considering isotopomer distances in general... i wonder if it's best to relate them back to charge and include floating point charges that aren't rounded.. perhaps this would be a more useful piece of information

#cross-compare distribution diff-to-expdiffs vs mass error on charge matches
#a good comparison of peak area vs # of datapoints would be nice to see

#deuterium elutes early

#for otf peak detection i would want to use a moving mean now, when the max average is hit, that's seen as a max, as long as it decreases later, a second moving mean is implemented for the 2nd half of the peak, if a new max is achieved then this process reorders based on the new max
#^you can then keep a moving average of distance/rate increase to max like roundcutoff

#I don't think fade comes from analyzer dynamic range, or anything of the sort. I think that's an ionization phenomenon, as it always happens to the lowest intensity isotopomer of the lowest intensity charge state. Even when other charge states are at extremely low intensities for the analyzer, ie 1e5, the fade is happening only to the same exact isotopomer as you'd expect.

#charge states and fade:
#After the isotope process is done, when you're matching them to theoretical quantities, that fade may be an important feature there, it might be worth-while to determine which isotopes need that quantity corrected, and how much correction is being given. I'd wager you could systematically fix those errors while improving all fittings by either seeing if the faded quantity is 'stolen' by other charge states, or by assuming the lost/gained amount from the adjacent isotopomer ratios.
#adjacent isotopomer ratios can be used to determine which isotopomers of which charge state can be trusted as an example quantity for applying a correction by the cross-charge state ratio. This mechanism can also mark an isotopomer of a charge state as faded. If you can't determine that any of the adjacent ratios are good for a specific isotopomer, then you shouldn't accept that specific isotopomer for any of the distributions.
#^If this happens in the middle of a distribution, you'll have to split the ends I suppose. If it's in the middle for just one, cut off the rest of only that distribution and let the pieces compete at a lower priority.
#It might be worth looking for missing line-pieces on the faded isotopomers.
#in there case where there's 2 charge states and you have no idea of knowing if an isotopomer of either is a legit value, then you can just accept them as long as the masses and everything else lines up well. The order of intensities matters in this case.

#I see an interesting phenomenon, where when the intensities of the main masses are really close to 1:1, the next adjacent isotopomers of the charge state with the less intense main mass tend to be higher than those of the charge state with the main mass. And I'm wondering if this can be an observable ionization phenomenon that somehow depends on mass. And whether I can deduce that a distribution has subisotopomers to the left or right of their majors perhaps?


#MS2 plan:
# - mark which lines got MS2 hits in the line model
# - match theoretical distributions to the found ones to prepare for MS2 searching
# - match found distributions [with their inside iterations] to potential peptide distributions
#   ^ regarding isolation widths, the total AUC of [only the datapoints within the "range"] should be used as a % value of one distribution to another that relate to overall intensity/# of fragments for their respective parts in the MS2

#
#Functions
#

def intersection_merge(mergable_items):
    sn = 0
    itemgroups = defaultdict(set) #groupn: [members]
    groupsofitems = {} #line: groupn
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

def overlap_counts(rbounds):
    finalscore = 0
    for (lmin, lmax), (rmin, rmax) in itertools.combinations(rbounds, 2):
        #encompassment with small flanks -> good
        #encompassment with large flanks -> "bad"
        #asymmetrical overlap with small flank -> not too bad
        #asymmetrical overlap with large flank -> bad
        #
        #small flank == smaller than 1/2 overlap
        #larger flank -> larger than overlap

        #FIRST check if they even overlap, if not -> set up for a subtraction
        #encompassment -> flanks subtracted from each other -> this difference will be subtracted from the total overlap -> final value
        #double flank, both has a flank -> both flanks added together -> subtracted from overlap -> final value
        #single flank -> flank is subtracted from overlap -> final value
        #non-overlap -> minus the full range of both
        if rmax > lmin and lmax > rmin:
            #overlap exists
            if rmax <= lmax:
                minmax = rmax
                maxmax = lmax
                maxl = True
            else:
                minmax = lmax
                maxmax = rmax
                maxl = False
            if rmin >= lmin:
                maxmin = rmin
                minmin = lmin
                minl = True
            else:
                maxmin = lmin
                minmin = rmin
                minl = False
            if maxl and minl:
                if rmax == lmax and rmin == lmin:
                    #equal boundaries
                    overlap = rmax - rmin
                    fullrange = lmax - lmin
                    finalscore += overlap
                elif rmax == lmax:
                    #min l flank
                    overlap = lmax - rmin
                    fullrange = rmax - lmin
                    minflank = rmin - lmin
                    score = overlap - minflank
                    finalscore += score
                elif rmin == lmin:
                    #max l flank
                    overlap = rmax - rmin
                    fullrange = lmax - lmin
                    maxflank = lmax - rmax
                    score = overlap - maxflank
                    finalscore += score
                else:
                    #l encompasses r
                    overlap = rmax - rmin
                    fullrange = lmax - lmin
                    minflank = rmin - lmin
                    maxflank = lmax - rmax
                    flankdiff = abs(maxflank - minflank)
                    score = overlap - flankdiff
                    finalscore += score
            elif maxl:
                if rmax == lmax:
                    #r min flank
                    overlap = rmax - lmin
                    fullrange = lmax - rmin
                    minflank = lmin - rmin
                    score = overlap - minflank
                    finalscore += score
                else:
                    #l max flank
                    #r min flank
                    overlap = rmax - lmin
                    fullrange = lmax - rmin
                    minflank = lmin - rmin
                    maxflank = lmax - rmax
                    flankdiff = abs(maxflank - minflank)
                    score = overlap - flankdiff
                    finalscore += score
            elif minl:
                if rmin == lmin:
                    #r max flank
                    overlap = lmax - rmin
                    fullrange = rmax - lmin
                    maxflank = rmax - lmax
                    score = overlap - maxflank
                    finalscore += score
                else:
                    #r max flank
                    #l min flank
                    overlap = lmax - rmin
                    fullrange = rmax - minl
                    minflank = rmin - lmin
                    maxflank = rmax - lmax
                    flankdiff = abs(maxflank - minflank)
                    score = overlap - flankdiff
                    finalscore += score
            else:
                #r encompasses l
                overlap = lmax - lmin
                fullrange = rmax - rmin
                maxflank = rmax - lmax
                minflank = rmin - lmin
                flankdiff = abs(maxflank - minflank)
                score = overlap - flankdiff
                finalscore += score
        else:
            #there's no overlap
            loverlap = lmax - lmin
            roverlap = rmax - rmin
            #subtract both overlaps
            finalscore -= loverlap + roverlap
    return finalscore

#
#Opening the file and extracting data
#

#it would be cool if deadsignal could be soft-coded from intensity

mslevelfile = '/'.join((processinglocation, 'centroid.ms1.pickle'))
with open(mslevelfile, 'rb') as pick:
    ms1scans = pickle.load(pick)

retentiontimesbyscanfile = '/'.join((processinglocation, 'retentiontimesbyscan.pickle'))
with open(retentiontimesbyscanfile, 'rb') as pick:
    retentiontimesbyscan = pickle.load(pick)

t1 = time()
msiter = iter(ms1scans)


timearray = []
scanindex = next(msiter)
mza, intensities = ms1scans[scanindex].values()
previousdata = mza.copy()
model = spatial.KDTree(mza[:,None])
scancount = 1

#intensities = scan['intensity array']
#intensityranks = intensities.size - intensities.argsort(axis=0).argsort(axis=0) - 1
#intensitysum = intensities.sum()
#rt = scan['scanList']['scan'][0]['scan start time'].real
rt = retentiontimesbyscan[scanindex]
#timedict[rt] = scancount
timearray.append(rt)
#injectionarray.append(it)
#nintensities = intensities
#retentiontimes = np.repeat(rt, len(mza))[:,None]
retentiontimes = np.repeat(rt, mza.size)
#coords = np.hstack((mza, retentiontimes, intensities, nintensities, intensities/intensitysum, intensityranks))[:,:,None].tolist()
#coords = np.hstack((mza, retentiontimes, intensities))[:,:,None].tolist()
coords = np.stack((mza, retentiontimes, intensities), axis=1).reshape(mza.size, 1, 3).tolist()

uids = (np.arange(mza.size)).tolist()
uidcount = max(uids) + 1

#trackedgroups = {} #uniqueid: [[masses], [rt-inds], [intensities], [intensities/injection times], [percent intensities of scans], [intensity rank of scan]]
trackedgroups = {} #uniqueid: [[masses, rt-inds, intensity/injection times],[...]]
trackedma = {} #latest moving average mass of trackedgroup: lineuid
linedeletioncounter = defaultdict(int) #lineuid: notmatched count
#linedeletiontime = defaultdict(float) #lineuid: total time a line is non-matched
lastmatchtime = {} #lineuid: last time a line was matched
#trackedscancount = defaultdict(int) #lineuid: number of scans the line has existed, and has been alive, for
#trackedlength = defaultdict(int) #lineuid: number of datapoints, a moving data length
groupmovingaverages = {} #lineuid: latest moving average of line
groupdifftoma = {} #lineid: moving difference to moving average
groupranges = {} #uniqueid: [minmass, maxmass]
modeltracking = {} #scan: number of masses being [added, matched, nonmatched, removed]

modeltracker = [0, 0, 0, 0]
modeltracker[0] += mza.size
#modeltracking[scan['index']] = modeltracker
modeltracking[scanindex] = modeltracker

#flatmasslist = mza.flatten().tolist()
flatmasslist = mza.tolist()
trackedma.update(zip(flatmasslist, uids))
trackedgroups.update(zip(uids, coords))
groupmovingaverages.update(zip(uids, flatmasslist))
elen = len(uids)
groupdifftoma.update(zip(uids, np.zeros(elen).tolist()))
groupranges.update(zip(uids, np.stack((mza, mza), axis=1).tolist()))
#trackedscancount.update(zip(uids, np.ones(elen).tolist()))
#trackedlength.update(zip(uids, np.ones(elen).tolist()))
lastmatchtime.update(zip(uids, np.repeat(rt, len(uids))))

modify = False
widestmassrange = 0 #a tracked float of the widest mass range
wides = []
linecorrections = []

#collect lines at second datapoint into a proton range-linked region

precursorscanmatches = {} #ms2 scan index: lineuid
precursorlinematches = defaultdict(list) #lineuid: [ms2 scan indices]
precursorapparentcharge = defaultdict(list) #lineuid: vendor determined charge
precursordistmatches = {} #ms2 scan index: distance to k-nearest
precursorcoordinates = {} #ms2 scan index: [rt of previous ms1, precursor mass, lower mass bound, upper mass bound]
#I need to pipe these in to line corrections below to get to compare these to my measurements

p1, p2, p3, p4, p5, p6, p7, p8 = [], [], [], [], [], [], [], []

stopper = False
roundcutoff = 0
for scanindex in msiter:
    pt2 = time()
    trackedkeys = {} #latest mass in a trackedgroup: lineid
    mza, intensities = ms1scans[scanindex].values()
    rt = retentiontimesbyscan[scanindex]
    timearray.append(rt)
    
    modeltracker = [0, 0, 0, 0]
    baseind = 0
    catches = []
    massdist = []
    #a k=1 nearest neighbors for signal-processing
    #this is iterating over numpy arrays because its slower to conver them to lists and slower to index a list
    #picking whatevers closer in intensity might start to fail as a concept if the ms1 scans are more spaced out, or perhaps boxcar'd
    for fn, f in enumerate(mza.tolist()):
        mindist = np.inf
        for n, b in enumerate(previousdata[baseind:]):
            dist = abs(b-f)
            if dist < mindist:
                minind = n + baseind
                mindist = dist
            elif dist == mindist:
                #two new masses have symmetrical distances to existing moving average
                #choose whichever is within the original lines range
                currentind = trackedma[b]
                linerange = groupranges[currentind]
                othermass = previousdata[minind]
                currentmatch = b > linerange[0] and b < linerange[1]
                othermatch = othermass > linerange[0] and othermass < linerange[1]
                if currentmatch and not othermatch:
                    #new match wins
                    minind = n + baseind
                    mindist = dist
                    #no other distances will be closer
                    break
                elif othermatch and not currentmatch:
                    #old match wins
                    #no other distances will be closer
                    break
                else:
                    #either both or neither are within the range
                    #switch from comparing masses to comparing intensities
                    currentintensity = trackedgroups[currentind][-1][2]
                    otherintensity = trackedgroups[minind][-1][2]
                    massintensity = intensities[fn]
                    cabs = abs(massintensity - currentintensity)
                    oabs = abs(massintensity - otherintensity)
                    if cabs < oabs:
                        #current mass wins out
                        minind = n + baseind
                        mindist = dist
                        #no other masses will be closer
                        break
                    else:
                        #other mass wins out
                        #no other masses will be closer
                        break
            else:
                break
        catches.append(minind)
        massdist.append(mindist)
        baseind = minind
    massdist = np.array(massdist)
    catches = np.array(catches)
    
    p2.append(time() - pt2)
    pt3 = time()
    
    #when profiled this bit throughout the if redundants loop below took 11s out of 284s when profiled (on battery)
    found = previousdata[catches] #check for duplicates here
    uf, ufc = np.unique(found, return_counts=True)
    ub = ufc > 1
    redundants = np.any(ub)
    #finding redundant matches
    if redundants:
        removals = []
        for umatch in uf[ub].tolist():
            mwhere = np.where(found == umatch)[0]
            mwdists = massdist[mwhere]
            mwdargmin = mwdists.argmin()
            removals.extend(np.delete(mwhere, mwdargmin).tolist())
    
    p3.append(time() - pt3)
    pt4 = time()
    
    #removing redundant matches -> the line ended for these ones, or its skipping an index
    retentiontimes = np.repeat(rt, mza.size)
    coords = np.stack((mza, retentiontimes, intensities), axis=1).reshape(mza.size, 1, 3)
    fmassdist = massdist.copy()
    if redundants:
        #things that had a redundant match, and weren't taken, from mza are put up as new lines
        ecoords = coords[removals]
        flatmasslist = ecoords[:,0,0]
        elen = len(ecoords)
        uids = np.arange(elen) + uidcount
        uidcount += elen
        trackedgroups.update(zip(uids, ecoords.tolist()))
        trackedkeys.update(zip(flatmasslist, uids))
        trackedma.update(zip(flatmasslist, uids))
        groupmovingaverages.update(zip(uids, flatmasslist))
        groupdifftoma.update(zip(uids, np.zeros(elen)))
        groupranges.update(zip(uids, np.stack((flatmasslist, flatmasslist), axis=1).tolist()))
        modeltracker[0] += elen #newly added
        #fixing the originals
        found = np.delete(found, removals)
        fmassdist = np.delete(fmassdist, removals)
        coords = np.delete(coords, removals, axis=0)
    found = found.flatten().tolist()
    
    p4.append(time() - pt4)
    pt5 = time()
    
    #the first 3 calls here are slow as fuck
    sorteddistances = np.sort(massdist) #this moved the cutoff up a little bit, higher mass-range lines seemed to be slightly better connected, there's more mass variation up there it seems...
    #mbool = sorteddistances <= sorteddistances[:,None] #isn't there an easier way to do this?, isn't this just arange?
    #countsums = mbool.sum(axis=0) / sorteddistances.size
    ##mbool = stats.rankdata(sorteddistances, method='max')[::-1] #^yep, but it's just slightly different because this is a ranking and not counting the things beyond it - so i'll skip this for now - the last of a dense rank needs a +1?
    mbool = np.arange(sorteddistances.size)[::-1] + 1
    countsums = mbool / sorteddistances.size
    #the sumcounts below is generally the same thing, but will differ at values where mbool would have given duplicate entries, doesn't change anything major enough to change the result
    sumcounts = sorteddistances.cumsum() / sorteddistances.sum()
    mincomboind = (countsums + sumcounts).argmin()
    mincombo = sorteddistances[mincomboind]
    #moving average of average of dists under mincombo
    explicitcutoff = sorteddistances[sorteddistances <= mincombo].mean()
    roundcutoff = (roundcutoff * scancount + explicitcutoff) / (scancount + 1)
    
    p5.append(time() - pt5)
    pt6 = time()
    
    #rcos.append(roundcutoff)
    #modifying things that are already being tracked
    fmzaremovals = []
    foundremovals = []
    for c, f, d in zip(coords.tolist(), found, fmassdist): #could this loop be concurrent?
        modify = False
        nf = c[0][0]
        tid = trackedma[f]
        tgroup = trackedgroups[tid]
        tlen = len(tgroup)
        lastmass = tgroup[-1][0]
        rmin, rmax = groupranges[tid]
        grange = rmax - rmin
        rep = False
        if tid == -1: #change -1 to the trackedkey of interest
            rep = True
        #this is for when the moving decision fails for something within the existing range, ain't no thang
        rangepass = nf <= rmax and nf >= rmin
        distancepass = abs(nf - lastmass) < grange / 2
        if rangepass or distancepass:
            oldma = groupmovingaverages[tid]
            nma = (oldma * tlen + nf) / (tlen + 1)
            nmadiff = abs(oldma - nma)
            groupmovingaverages[tid] = nma
            madiff = groupdifftoma[tid]
            groupdifftoma[tid] = (madiff * (tlen - 1) + nmadiff) / tlen
            modify = True
            if rep:
                print(1, nf, nmadiff, madiff, rmax-rmin)
        #generally, this is good for on-the-fly decision making when the moving target is outside the existing mass range. This dominates later on, where it's more robust
        elif tlen >= minmovinginds:
            oldma = groupmovingaverages[tid]
            madiff = groupdifftoma[tid]
            nma = (oldma * tlen + nf) / (tlen + 1)
            nmadiff = abs(oldma - nma)
            if rep:
                print(2, nf, nmadiff, madiff, rmax-rmin)
            #if nmadiff <= np.mean(madiff): #max(madiff) + (2*np.mean(madiff)):
            #if nmadiff <= np.mean(madiff):
            if nmadiff <= madiff:
                groupmovingaverages[tid] = nma
                groupdifftoma[tid] = (madiff * (tlen - 1) + nmadiff) / tlen
                modify = True
        else:
            if tlen > 1:
                #why is this leveraging the grouprange if theres no range yet
                grange = rmax - rmin
                if d <= roundcutoff + grange:
                    oldma = groupmovingaverages[tid]
                    nma = (oldma * tlen + nf) / (tlen + 1)
                    nmadiff = abs(oldma - nma)
                    groupmovingaverages[tid] = nma
                    madiff = groupdifftoma[tid]
                    groupdifftoma[tid] = (madiff * (tlen - 1) + nmadiff) / tlen
                    modify = True
                    if rep:
                        print(3, nf, nmadiff, madiff, grange)
            else: #first one's not free, but comes at a discount
                #if (d - roundcutoff) - d * roundcutoff <= roundcutoff: #1154745
                if d <= roundcutoff * 2: #1154797 and way less lenient
                    oldma = groupmovingaverages[tid]
                    nma = (oldma * tlen + nf) / (tlen + 1)
                    nmadiff = abs(oldma - nma)
                    groupmovingaverages[tid] = nma
                    groupdifftoma[tid] = nmadiff
                    modify = True
                    if rep:
                        print(4, nf, nmadiff, madiff, grange)
        if modify:
            trackedkeys[nf] = tid
            trackedma[nma] = trackedma.pop(f)
            #trackedscancount[tid] += 1
            #trackedlength[tid] += 1
            #for n, ci in enumerate(c):
            #    trackedgroups[tid][n].append(ci)
            trackedgroups[tid].append(c[0])
            #if linedeletioncounter[tid] > 0:
            #linedeletioncounter[tid] -= 1 #1661209
            #linedeletioncounter[tid] = 0 #1593013
            #linedeletioncounter[tid] //= 2 #1606868
            lastmatchtime[uidcount] = rt
            linedeletioncounter[tid] /= 2
            #linedeletiontime[tid] /= 2
            #^these two options are very different in philosophy. The -1 doesn't allow things to be able to extend themselves very far. The =0 allows for a lifeline every time a new point is added, this one assumes that the end is not going to be hard to find. There's only a difference of 4% between the two trackedgroup lengths listed next to the numbers above^.
            if nf < rmin:
                groupranges[tid][0] = nf
            if nf > rmax:
                groupranges[tid][1] = nf
            gmin, gmax = groupranges[tid]
            grange = gmax - gmin
            if grange > widestmassrange:
                widestmassrange = grange
            modeltracker[1] += 1 #matched
        else:
            #trackedgroups[uidcount] = [[ci] for ci in c]
            trackedgroups[uidcount] = c
            trackedkeys[nf] = uidcount
            trackedma[nf] = uidcount
            groupmovingaverages[uidcount] = nf
            groupdifftoma[uidcount] = 0 #this zero won't bog down any averages, same principle new mechanics
            groupranges[uidcount] = [nf, nf]
            #trackedscancount[uidcount] += 1
            #trackedlength[uidcount] += 1
            lastmatchtime[uidcount] = rt
            uidcount += 1
            modeltracker[0] += 1 #newly added
            foundremoval = groupmovingaverages[tid]
            foundremovals.append(foundremoval)
    
    p6.append(time() - pt6)
    pt7 = time()
    
    for fr in foundremovals:
        found.remove(fr)
    #fmza = np.delete(fmza, fmzaremovals)
    #nonmatched = np.setdiff1d(previousdata, found)[:,None].flatten()
    nonmatched = np.setdiff1d(previousdata, found)
    nmlen = nonmatched.size - 1
    mzlen = mza.size
    newmodelremovals = []
    #things from previousdata not in found gets +1 to linedeletioncounter
    for n, nm in enumerate(nonmatched.tolist()): #could this loop be concurrent?
        #linekey = trackedkeys[nm]
        linekey = trackedma[nm]
        #trying to accelerate single-point loss below made the code take so long that I didn't even finish it. It's absurd, why?!
        #if trackedlength[linekey] == 1:
        #    linedeletioncounter[linekey] += 2 #accelerating the loss of lone noise points
        #elif trackedlength[linekey] == 2:
        #    #reverse the damage done by the single-scan deletion multiplier
        #    linedeletioncounter[linekey] -= trackedscancount[linekey]
        #else:
        linedeletioncounter[linekey] += 1
        #linedeletiontime[linekey] = rt - lastmatchtime[linekey]
        #trackedscancount[linekey] += 1
        if linedeletioncounter[linekey] > deadsignal:
        #if linedeletiontime[linekey] > deadsignal:
            #determine, out of all matched and nonmatched, which fall into a +/- subisomax distance to this movingma, put the lineuids together in a list to be later intersection_merged for line corrections
            newmodelremovals.append(n)
            modeltracker[3] += 1 #removed
            
            trackedgroups[linekey] = np.array(trackedgroups[linekey]) #more efficient memory storage now that it doesn't need to be appended, not much speed compromised but it is a little slower
            
            #collection subisomax-radius lines with overlapping rt's for line corrections
            correctionradius = set()
            nmoverlaps = nonmatched[np.abs(nm - nonmatched) <= subisomax].tolist()
            mzoverlaps = mza[np.abs(nm - mza) <= subisomax].tolist()
            nmkeys = list(map(trackedma.get, nmoverlaps)) #goes to trackedma
            mzkeys = list(map(trackedkeys.get, mzoverlaps)) #goes to trackedkeys
            correctionradius.update(nmkeys)
            correctionradius.update(mzkeys)
            linecorrections.append(tuple(correctionradius))
            #the below worked well for the profile data i was viewing but it just isnt great for actual line corrections, i might need another fix later
            #if n == 0:
            #    nmupper = nonmatched[n+1]
            #    nmlower = -np.inf
            #    nmupdist = abs(nmupper - nm)
            #    nmlowdist = 0
            #elif n == nmlen:
            #    nmupper = np.inf
            #    nmlower = nonmatched[n-1]
            #    nmupdist = 0
            #    nmlowdist = abs(nmlower - nm)
            #else:
            #    nmupper = nonmatched[n+1]
            #    nmlower = nonmatched[n-1]
            #    nmupdist = abs(nmupper - nm)
            #    nmlowdist = abs(nmlower - nm)
            #
            #if nmupdist > subisomax:
            #    nmupper = np.inf
            #if nmlowdist > subisomax:
            #    nmlower = -np.inf
            #
            #mzupperpoint = np.searchsorted(mza, nm) #should be greater than nm
            #if mzupperpoint == 0:
            #    mzupper = mza[mzupperpoint]
            #    mzlower = -np.inf
            #    mzupdist = abs(mzupper - nm)
            #    mzlowdist = 0
            #elif mzupperpoint == mzlen:
            #    mzupper = np.inf
            #    mzlower = mza[mzupperpoint-1]
            #    mzupdist = 0
            #    mzlowdist = abs(mzlower - nm)
            #else:
            #    mzupper = mza[mzupperpoint]
            #    mzlower = mza[mzupperpoint-1]
            #    mzupdist = abs(mzupper - nm)
            #    mzlowdist = abs(mzlower - nm)
            #
            #if mzupdist > subisomax:
            #    mzupper = np.inf
            #if mzlowdist > subisomax:
            #    mzlower = -np.inf
            #
            ##these start as np.int64
            #if nmlower < mzlower:
            #    lower = int(trackedkeys[mzlower])
            #elif mzlower < nmlower:
            #    lower = int(trackedma[nmlower])
            #else: #==, both are -np.inf
            #    lower = False
            #if nmupper > mzupper:
            #    upper = int(trackedkeys[mzupper])
            #elif mzupper > nmupper:
            #    upper = int(trackedma[nmupper])
            #else:
            #    upper = False
            #
            ##because uid can be 0
            #lowerint = type(lower) == int
            #upperint = type(upper) == int
            #if lowerint and upperint:
            #    correctionradius = lower, upper
            #elif lowerint:
            #    correctionradius = lower,
            #elif upperint:
            #    correctionradius = upper, 
            #else:
            #    #this should happen now, for things that dont fit either subisomax distance
            #    correctionradius = ()
            #linecorrections.append(correctionradius)
        else:
            modeltracker[2] += 1 #nonmatched
    
    p7.append(time() - pt7)
    pt8 = time()

    wides.append(widestmassrange)
    #nonmatched = np.delete(nonmatched, newmodelremovals)[:,None]
    #nonmatched = np.delete(nonmatched, newmodelremovals)[:,None]
    nonmatched = np.delete(nonmatched, newmodelremovals)
    #newtrain = np.append(mza, nonmatched, axis=0)
    currentmasskeys = list(map(trackedkeys.get, mza.flatten().tolist()))
    #currentmasses = np.array(list(map(groupmovingaverages.get, currentmasskeys)))[:,None]
    currentmasses = np.array(list(map(groupmovingaverages.get, currentmasskeys)))
    newtrain = np.sort(np.append(currentmasses, nonmatched, axis=0))[:,None]
    #previousdata = np.append(currentmasses, nonmatched, axis=0).flatten()
    previousdata = np.sort(np.append(currentmasses, nonmatched))
    model = spatial.KDTree(newtrain) #for ms2 tracking, this file only
    modeltracking[scanindex] = modeltracker
    scancount += 1
    
    p8.append(time() - pt8)

for cmk in currentmasskeys:
    trackedgroups[cmk] = np.array(trackedgroups[cmk])

nonmatchedkeys = list(map(trackedma.get, nonmatched.flatten().tolist()))
for nmk in nonmatchedkeys:
    trackedgroups[nmk] = np.array(trackedgroups[nmk])

timearray = np.array(timearray)
#precursorcoordinates = np.array(precursorcoordinates)
#pdistmatches = np.array(list(precursordistmatches.values()))

print(time() - t1, 'line model')

print('~')
print(sum(p1))
print(sum(p2))
print(sum(p3))
print(sum(p4))
print(sum(p5))
print(sum(p6))
print(sum(p7))
print(sum(p8))
print('~')

#line corrections:
# - subisomax radius
# - overlapping time ranges
# - no redundant timepoints - mediation process for when there is, things closest to each other will take precedence and that overlapping timepoint line forms a hard barrier that won't be crossed for lines when adopting new data points

#non-rt overlaps that are next to each other
# - use intensity rate of change to connect the pieces
# - the to-be-connected masses should be close
# - this will probably be for some stray data points and those weird curved lines


t2 = time()
regions = [] #t, b, l, r
for k, a in trackedgroups.items():
    #minmass, mintime, mii, nmii, minperc, minrank = a.min(axis=1)
    minmass, mintime, mii = a.min(axis=0)
    #maxmass, maxtime, mai, nmai, maxperc, maxrank = a.max(axis=1)
    maxmass, maxtime, mai = a.max(axis=0)
    #wmean = np.average(a[0], weights=a[2]) #better to weight by intensity or injection time? both?
    wmean = (a[:,0] * a[:,2]).sum() / a[:,2].sum() #why is this faster, wtf numpy
    regions.append([minmass, maxmass, mintime, maxtime, wmean, k])

regions = np.array(regions)
#regions = regions[regions[:,5].argsort()]
#(np.arange(regions.shape[0]) == regions[:,8]).all() #passes!

print(time() - t2, 'initial regions -', len(regions))


t3 = time()
correctiongroups = intersection_merge(linecorrections)
correctiongroups = [list(i) for i in correctiongroups if len(i) > 1]
print(time() - t3, 'correction group intersection merge')

t4 = time()
timeextension = np.diff(timearray).mean() * minpoints #i'm not a huge fan of this because of the potential for it to connect to completely different things, but there's also some shit i just need to connect...
linecorrections = []
for cg in correctiongroups:
    torder = regions[cg,4].argsort()
    ncg = np.array(cg)[torder]
    tregs = regions[ncg]
    masstable = tregs[:,:2]
    masswidths = np.diff(masstable)
    moverlaps = np.logical_and(masstable[:,0] - masswidths.flatten() <= masstable[:,1,None] + masswidths, masstable[:,1] + masswidths.flatten() >= masstable[:,0,None] - masswidths)
    timetable = tregs[:,2:4]
    toverlaps = np.logical_and(timetable[:,0] - timeextension <= timetable[:,1,None] + timeextension, timetable[:,1] + timeextension >= timetable[:,0,None] - timeextension)
    overlaps = np.logical_and(moverlaps, toverlaps)
    overwheres = np.argwhere(overlaps).tolist()
    ogroups = intersection_merge(overwheres)
    for ogs in ogroups:
        if len(ogs) > 1:
            og = list(ogs)
            tmatches = tregs[og]
            tmkeys = ncg[og]
            matchtimes = [trackedgroups[i][:,1].tolist() for i in tmkeys]
            flattimes = list(itertools.chain(*matchtimes))
            if len(flattimes) == len(set(flattimes)):
                linecorrections.append(tmkeys.tolist())
            else:
                #below is making sure only appropriately ordered links are concected and further connections aren't skipping over other lines. it's essentially assuring the intersection merge below works on the basis of this as a directional graph
                linkedpairs = {} #pair: distance of means
                uppers = {} #mkeys: upper of the pair
                uplinks = {} #mkey: closest above line
                updists = {} #mkey: distance to closest above
                downers = {} #mkeys: lower of the pair
                downlinks = {} #mkey: closest below line
                downdists = {} #mkey: distance to closest below
                for ow in overwheres:
                    if len(ogs.intersection(ow)) == 2:
                        mregs = tregs[ow]
                        mkeys = tuple(mregs[:,5].astype(int).tolist())
                        l, r = mregs[:,4]
                        lk, rk = mkeys
                        massdiff = abs(l - r)
                        if l > r:
                            upkey = lk
                            downkey = rk
                            uppers[mkeys] = lk
                            downers[mkeys] = rk
                        else:
                            upkey = rk
                            downkey = lk
                            uppers[mkeys] = rk
                            downers[mkeys] = lk
                        if upkey in downlinks:
                            if massdiff < downdists[upkey]:
                                downlinks[upkey] = downkey
                                downdists[upkey] = massdiff
                        else:
                            downlinks[upkey] = downkey
                            downdists[upkey] = massdiff
                        if downkey in uplinks:
                            if massdiff < updists[downkey]:
                                uplinks[downkey] = upkey
                                updists[downkey] = massdiff
                        else:
                            uplinks[downkey] = upkey
                            updists[downkey] = massdiff
                        #check timepoints here to make sure they're qualified for merging?
                        linkedpairs[mkeys] = massdiff
                
                sortedpairs = sorted(linkedpairs.items(), key=lambda x: x[1])
                #below is making sure no directly adjacent connections have timepoint redundancy
                passedpairs = []
                linkedtimes = {} #pair: [mergedtimes]
                for mkeys, score in sortedpairs:
                    if uplinks[downers[mkeys]] == uppers[mkeys] or downlinks[uppers[mkeys]] == downers[mkeys]:
                        mtimes = [trackedgroups[i][:,1].tolist() for i in mkeys]
                        mergedtimes = mtimes[0] + mtimes[1]
                        if len(mergedtimes) == len(set(mergedtimes)):
                            passedpairs.append(mkeys)
                            linkedtimes[mkeys] = set(mergedtimes)
                #intersection merge with non-redundancy requirement for timepoints
                #this doesn't expand mass-ranges as the signals expand, might be a flaw of this whole process I suppose
                sn = 0
                itemgroups = defaultdict(set) #groupn: [members]
                itemtimes = defaultdict(set) #groupn: [covered timepoints]
                groupsofitems = {} #line: groupn
                for items in passedpairs: 
                    locs = set()
                    for i in items: 
                        if i in groupsofitems:
                            locs.add(groupsofitems[i])
                        else:
                            otherline = i
                    combine = False
                    if locs:
                        joiner = min(locs)
                        if len(locs) > 1:
                            combine = True
                            for oldlocs in locs.difference([joiner]):
                                #the timepoint checks here are to check that non-adjacent connections aren't redundant
                                if oldlocs in itemtimes:
                                    oldtimes = itemtimes[oldlocs]
                                else:
                                    oldtimes = linkedtimes[items]
                                if not itemtimes[joiner].intersection(oldtimes):
                                    for ol in itemgroups[oldlocs]:
                                        groupsofitems[ol] = joiner
                                    itemgroups[joiner].update(itemgroups.pop(oldlocs))
                                    #there should only be one oldloc no matter the iteration, so this all operates without the need for complete prior checking of all timepoint redundancy
                                    itemtimes[joiner].update(oldtimes)
                                    if oldlocs in itemtimes:
                                        del itemtimes[oldlocs]
                                else:
                                    combine = False
                        else:
                            #check that the non-loc'd item isn't redundant for timepoints
                            oldtimes = trackedgroups[otherline][:,1].tolist()
                            if not itemtimes[joiner].intersection(oldtimes):
                                combine = True
                    else:
                        joiner = sn
                        sn += 1
                        combine = True
                    if combine:
                        itemgroups[joiner].update(items)
                        itemtimes[joiner].update(linkedtimes[items])
                        for i in items:
                            groupsofitems[i] = joiner
                linecorrections.extend([list(i) for i in itemgroups.values()])
        #else:
            #nopes.append(tmkeys.tolist())
            #add to the later < minpoint + within massrange check
            #I'll forget about anything else for now, this is good enough. Only lone datapoints < minpoints or signals that didn't pass deadsignal would be left here. minimal error if any
print(time() - t4, 'line corrections groups')

t5 = time()
for lines in linecorrections:
    linegrid = []
    for line in lines:
        if line in precursorlinematches:
            for scan in precursorlinematches[line]:
                precursorscanmatches[scan] = uidcount
            precursorlinematches[uidcount].extend(precursorlinematches.pop(line))
            precursorapparentcharge[uidcount].extend(precursorapparentcharge.pop(line))
        for c in trackedgroups[line].tolist():
            linegrid.append(list(c))
        del trackedgroups[line]
    linegrid = np.array(sorted(linegrid, key=lambda x: x[1]))
    trackedgroups[uidcount] = np.array(linegrid)
    uidcount += 1

#really fast but disconnecting later communication between dicts, I don't think I need them though
groupholder = {}

precursorlineholder = {}
precursorscanholder = {}
precursorchargeholder = {}

kl = list(trackedgroups.keys())
for k in kl:
    groupholder[k] = trackedgroups.pop(k)
    if k in precursorlinematches:
        for scan in precursorlinematches[k]:
            precursorscanholder[scan] = precursorscanmatches.pop(scan)
        precursorlineholder[k] = precursorlinematches.pop(k)
        precursorchargeholder[k] = precursorapparentcharge.pop(k)

for n, k in enumerate(kl):
    trackedgroups[n] = groupholder.pop(k)
    if k in precursorlineholder:
        for scan in precursorlineholder[k]:
            precursorscanmatches[scan] = n
        precursorlinematches[n] = precursorlineholder.pop(k)
        precursorapparentcharge[n] = precursorchargeholder.pop(k)

print(time() - t5, 'line corrections')

t6 = time()
startingpoints = defaultdict(list)
regions = [] #t, b, l, r
for k, a in trackedgroups.items():
    minmass, mintime, mii = a.min(axis=0)
    maxmass, maxtime, mai = a.max(axis=0)
    wmean = (a[:,0] * a[:,2]).sum() / a[:,2].sum() #why is this faster, wtf numpy
    peakarea = np.trapezoid(a[:,2], a[:,1])
    maxintensity = a[:,2].max()
    regions.append([minmass, maxmass, mintime, maxtime, len(a), peakarea, maxintensity, wmean, k])
    startingpoints[mintime].append(k)

regions = np.array(regions)

newwides = []
maxrange = 0
for t in sorted(startingpoints):
    for line in startingpoints[t]:
        minmass, maxmass = regions[line,:2]
        massrange = maxmass - minmass
        if massrange > maxrange:
            maxrange = massrange
    newwides.append(maxrange)

#plt.plot(wides, label='initial')
#plt.plot(newwides, label='corrected')
#plt.legend()
#plt.show()
print('old max width', max(wides))
print('new max width', max(newwides))

print(time() - t6, 'new regions -', len(regions))
t7 = time()

#st = 0
#et = np.ceil(regions[:,2:4].max())

#boundrec = [regions[:,7].min() - 1, regions[:,7].max() + 1, st, et]
#regionsample = arg_coord_rectangle_overlap(boundrec, regions[:,:4]).tolist()
#rselection = regions[regionsample]
#regiter = rselection[rselection[:,7].argsort()]
#regiter = regiter[regiter[:,4] >= minpoints]
rsorted = regions[regions[:,7].argsort()]
regiter = rsorted[rsorted[:,4] >= minpoints]

#connectionspine = {} #connectionindex: index of previous connectionindex upon which a given pair is built
#connectionpairkeys = {} #connectionindex: pairkey
pairkeys = {} #pairkey: pair
previousdecrease = {} #connectionindex: True, exists if something is increasing

di = 0 #connectionindexs
paircharges = {} #connection: charge
scoresbypair = {} #pair: [[absacdiff, datapercdiff, rtoffset],]
#pairsbyline = defaultdict(list) #mass: [pairs]
pairsbyline = defaultdict(str) #line: 'pair,pair,'

si = 0 #subisokeys
subisomasses = {} #lineuid: subisogroup
subisogroups = defaultdict(lambda: defaultdict(list)) #subiso group: max charge for pair: [pairkeys]

masswidthlimit = roundcutoff * 2

#47595459 entries into spine/cpairkeys, and this was only part of the overload
#previousdecrease held ~2.5 gigs, pretty serious, it needs to be managed
#connectionspine and connectionpairkeys had 4g each
#i could potentially seek out a memory efficient trie
#numpy is also efficient as fuck and I didn't know that, if I can turn it into a list-format then I could potentially use numpy

#pairkeys can remain the same dict
#connectionspine can be a dict of lists, converted to numpy arrays when the connection finishes, (pairkey index): [subsequent pairkey indices]
#keep a lastkeys dict {masskey: [pairkey indices]}, for when a masskey is removed, the connectionspine will turn its list to a numpy array, and previousdecrease will erase all of the values owned by that masskey
#connectionspine = defaultdict(list) #connectionindex: [pairkeys]
connectionspine = defaultdict(str) #connectionindex: 'pairkey,pairkey,'
#latestconnections = defaultdict(lambda: defaultdict(list)) #masskey: charge: [current connection indices that end in masskey], a mass is always active as long as it's in masspool, this keeps track of all the groups that end in any given mass
latestconnections = defaultdict(lambda: defaultdict(str)) #masskey: charge: [current connection indices that end in masskey], a mass is always active as long as it's in masspool, this keeps track of all the groups that end in any given mass
latestmass = {} #connectionindex: latest masskey

#^these str dicts are probably only a little slower in most places but offer a ~slight memory advantage, i came across a file with massive amounts of noise and its disheartening to see i can't actually process its distributions even with these adjustments, ahh whatever


#I'm learning that numpy arrays are actually efficient as fuck for memory storage, I should be able to implement a branchends/spineends here using dicts of lists that i turn into numpy arrays once a branch's final mass gets taken out of mass pool, i also need to remove all of that branches entries in previousdecreases
#also seeing as numpy arrays are so dope, I should change the structure of trackedgroups for lines on removal to save on memory

#i think i can break this down by overlap connections, hard-code the rt and mass limits into a consolidation process -> intersection_merge groups of overlappers -> and work on those groups independently -> rank them independently -> output dists independently
#might even be multiprocessable in the end

pi = 0 #pairkeys
masspool = []
#could this be done as two ball trees that connect to each other?
for reg in regiter.tolist():
    npoints = reg[4]
    nm = reg[7]
    nmlt = reg[2]
    nmrt = reg[3]
    nmassmin = reg[0]
    nmassmax = reg[1]
    nmwidth = nmassmax - nmassmin
    if nmwidth > masswidthlimit:
        #this is strict, but seems to help in the end, meaning that essentially the means are rather important and useful
        nmassmin = nm - masswidthlimit
        nmassmax = nm + masswidthlimit
    nkey = int(reg[8])
    nintensity = reg[5]
    masspoolremovals = []
    nlrange = nmrt - nmlt
    for okey in masspool: #could this loop be concurrent?
        oreg = regions[okey]
        om = oreg[7]
        omassmin = oreg[0]
        omassmax = oreg[1]
        omwidth = omassmax - omassmin
        if omwidth > masswidthlimit:
            omassmin = om - masswidthlimit
            omassmax = om + masswidthlimit
        if nmassmin - omassmax <= proton:
            omlt = oreg[2]
            omrt = oreg[3]
            olrange = omrt - omlt
            overpass = False
            if nmlt < omrt and nmrt > omlt: #rt's overlap
                #if you want to allow non-overlaps to match, and there are good reasons to, you need percentoverlap to become negative so that its later addition is valued worse
                overlap = min(omrt, nmrt) - max(omlt, nmlt)
                encompassed = False
                if nmlt > omlt and nmrt < omrt: #new encompassed
                    encompassed = True
                    combinedrange = nlrange + olrange
                    percentoverlap = (overlap * 2) / combinedrange
                    #using newinclimit here is appropriate because it's directly related to expected differences in adjacent isotopomer quantities, which is what the overlap is also ~somewhat assessing
                    if percentoverlap > newinclimit:
                        overpass = True
                elif omlt > nmlt and omrt < nmrt: #old encompassed
                    encompassed = True
                    combinedrange = nlrange + olrange
                    percentoverlap = (overlap * 2) / combinedrange
                    if percentoverlap > newinclimit:
                        overpass = True
                else:
                    fullrange = max(omrt, nmrt) - min(omlt, nmlt)
                    percentoverlap = overlap / fullrange
                    if percentoverlap > 0.5: #this is super lenient I think
                        overpass = True
            #else:
                #negative percentoverlap values for non-overlaps within some 1.75 range or something goes here
            if overpass:
                diff = nm - om
                #link = False
                #if encompassed or percentoverlap > 0.75: #check for distancelinks that can be used to expand the subiso range
                #    link = True
                #if diff < subisomax + roundcutoff:
                #    if link:
                #        #potential subisos
                #        maxisocharge = np.floor(subisomax / diff)
                #        if maxisocharge == 0: #it can be rounded to zero b/c of widthbuffer? not sure, but it happens
                #            maxisocharge += 1
                #        if om in subisomasses:
                #            ti = subisomasses[okey]
                #            subisogroups[ti][maxisocharge].append(pi)
                #            subisomasses[nkey] = ti
                #        else:
                #            #make a new subiso group
                #            subisogroups[si][maxisocharge].append(pi)
                #            subisomasses[okey] = si
                #            subisomasses[nkey] = si
                #            si += 1
                #        lpair = (okey, nkey)
                #        pairkeys[pi] = lpair
                #        pi += 1
                #else:
                #sometimes a charge registers as one thing but the value is way closer to +/-1
                initialcharge = round(proton / diff)
                if initialcharge > 1: #nothing close enough to care about for 1, plus the first bit would go to 0 and cause annoying zerodivision warnings
                    chargespread = np.linspace(initialcharge - 1, initialcharge + 1, 3)
                    expspread = proton / chargespread
                    minexpind = np.abs(diff - expspread).argmin()
                    charge = int(chargespread[minexpind])
                else:
                    charge = initialcharge
                actualmass = om * charge - proton * charge
                #min/max peptide mass from the isotope database, anything with a basemmass larger than that max shouldn't be considered, it would help keep out ridiculously large charges - instead having them be a part of nodists, the way they probably should
                if actualmass <= uppermasslimit:
                    expdiff = proton / charge
                    acdiff = expdiff - diff
                    diffcut = expdiff * chargetolerance
                    if acdiff > -1 * (diffcut * chargetolerance + masswidthlimit):
                    #^a lighter acceptance for distances above the charge dist - it does happen from time to time on lower-hanging straggler isotopomers
                        if acdiff <= diffcut + masswidthlimit:
                            absacdiff = abs(acdiff) * charge #normalizing -> distance to proton
                            ointensity = oreg[5]
                            ncons = 0
                            intensitypercdiff = abs(nintensity - ointensity) / (nintensity + ointensity) / 2
                            subcheck = False
                            if okey in latestconnections:
                                if charge in latestconnections[okey]:
                                    subcheck = True
                            decreasecheck = nintensity < ointensity
                            if subcheck:
                                #csubs = latestconnections[okey][charge].copy() #need copy here?
                                csubs = map(int, latestconnections[okey][charge].split(',')[:-1])
                                for adi in csubs:
                                    ratiocheck = steplimit
                                    if adi in previousdecrease:
                                        if not decreasecheck:
                                            #intensity is increasing
                                            ratiocheck = newinclimit
                                        #else:
                                            #input model decrease parameters here
                                    if intensitypercdiff <= ratiocheck:
                                        #connectionpairkeys[di] = pi
                                        #connectionspine[di] = adi
                                        #latestconnections[nkey][charge].append(di)
                                        if latestmass[adi] == okey:
                                            xdi = adi
                                        else:
                                            xdi = di
                                            #opairs = pairsbyline[okey]
                                            opairs = pairsbyline[okey].split(',')[:-1]
                                            #spinecopy = connectionspine[adi].copy()
                                            spinecopy = connectionspine[adi].split(',')[:-1]
                                            spinds = []
                                            for op in opairs:
                                                if op in spinecopy:
                                                    spinds.append(spinecopy.index(op))
                                            spi = max(spinds)
                                            spinecopy = spinecopy[:spi]
                                            #connectionspine[xdi] = spinecopy
                                            connectionspine[xdi] = ','.join((spinecopy)) + ','
                                            di += 1
                                        #connectionspine[xdi].append(pi)
                                        connectionspine[xdi] += str(pi) + ','
                                        #lm = latestmass[adi] #this is just okey
                                        #latestconnections[okey][charge].remove(adi) #i shouldn't remove this, but i should check if okey is latestmass in order to determine if a new spine should be cast out of a copy of adi up to okey
                                        latestmass[xdi] = nkey
                                        #latestconnections[nkey][charge].append(xdi)
                                        latestconnections[nkey][charge] += str(xdi) + ','
                                        if decreasecheck:
                                            #previousdecrease[di] = True
                                            previousdecrease[xdi] = True
                                        ncons += 1
                                        #di += 1
                            else:
                                #no previous subgroup 
                                if intensitypercdiff <= steplimit:
                                    #connectionpairkeys[di] = pi
                                    #connectionspine[di].append(pi)
                                    connectionspine[di] += str(pi) + ','
                                    latestmass[di] = nkey
                                    #latestconnections[nkey][charge].append(di)
                                    latestconnections[nkey][charge] += str(di) + ','
                                    if decreasecheck:
                                        previousdecrease[di] = True
                                    ncons += 1
                                    di += 1
                            if ncons > 0:
                                opoints = oreg[4]
                                #dpercdiff = abs(npoints - opoints) / (npoints + opoints) / 2
                                dpercdiff = abs(nintensity - ointensity) / (nintensity + ointensity) / 2
                                lpair = (okey, nkey)
                                pairkeys[pi] = lpair
                                paircharges[pi] = charge
                                scorelist = [absacdiff, dpercdiff, percentoverlap, ~decreasecheck]
                                for p in lpair:
                                    #pairsbyline[p].append(pi)
                                    pairsbyline[p] += str(pi) + ','
                                scoresbypair[pi] = scorelist
                                if subcheck:
                                    #connectionpairkeys[di] = pi
                                    #connectionspine[di].append(pi)
                                    connectionspine[di] += str(pi) + ','
                                    latestmass[di] = nkey
                                    #latestconnections[nkey][charge].append(di)
                                    latestconnections[nkey][charge] += str(di) + ','
                                    if decreasecheck:
                                        previousdecrease[di] = True
                                    di += 1
                                pi += 1
                #elif link:
                #    #things not really within subiso distance, but that I also want to keep track of
                #    maxisocharge = np.floor(subisomax / diff)
                #    #I'll let these ones say as 0 because they technically didn't hit
                #    #if maxisocharge == 0: #it can be rounded to zero b/c of widthbuffer? not sure, but it happens
                #    #    maxisocharge += 1
                #    if om in subisomasses:
                #        ti = subisomasses[okey]
                #        subisogroups[ti][maxisocharge].append(pi)
                #        subisomasses[nkey] = ti
                #    else:
                #        #make a new subiso group
                #        subisogroups[si][maxisocharge].append(pi)
                #        subisomasses[okey] = si
                #        subisomasses[nkey] = si
                #        si += 1
                #    lpair = (okey, nkey)
                #    pairkeys[pi] = lpair
                #    pi += 1
        else:
            #om is past proton distance, remove om from mass pool
            masspoolremovals.append(okey)
    for mpr in masspoolremovals:
        masspool.remove(mpr)
        #if mpr in pairsbyline:
        #    pairsbyline[mpr] = np.array(pairsbyline[mpr]) #memory effient storage
        if mpr in latestconnections:
            for charge, cons in latestconnections[mpr].items():
                #for con in cons:
                for con in map(int, cons.split(',')[:-1]):
                    if mpr == latestmass[con]:
                        del latestmass[con]
                        #connectionspine[con] = np.array(connectionspine[con]) #memory efficient storage
                        try:
                            del previousdecrease[con]
                        except KeyError: #no previous decrease, idc really
                            pass
            del latestconnections[mpr]
    masspool.append(nkey)

#for con, masskey in latestmass.items():
#    connectionspine[con] = np.array(connectionspine[con])
#    pairsbyline[masskey] = np.array(pairsbyline[masskey])

#there should probably be a better solution but some things were getting through as lists
#for masskey in masspool:
#    if masskey in pairsbyline:
#        pairsbyline[masskey] = np.array(pairsbyline[masskey])

del latestconnections
del previousdecrease
del latestmass

print(time() - t7, 'mass processing')
t8 = time()

flatdistgroups = set() #a tuple of every potential distribution
#minimaldistgroups = set() #only supersets of every potential distribution
#minimaldistgroupsbyline = defaultdict(set) #line: [all supersets this belongs to]

#distributionscoresbyline = defaultdict(list) #linekey: [pairkeys] -> without a set there is redundancy in here that disrupts downstream
#distributionscoredict = defaultdict(list) #pairkey: [[scores],]
for distkey, pairspine in connectionspine.items(): #this loop needs to write its massive outputs to disk and import them later on from a different function
    pairspine = map(int, pairspine.split(',')[:-1])
    maximalsuperset = []
    activepairlist = []
    scores = []
    #for pairs in pairspine.tolist():
    for pairs in pairspine:
        activepairlist.append(pairs)
        scores.append(scoresbypair[pairs])
        flatdist = []
        competinglines = set()
        for pair in activepairlist:
            pk = pairkeys[pair]
            flatdist.extend(pk)
            #flatdist.append(pair)
            #for p in pk:
            #    #there was no problems here
            #    #if p not in pairsbyline:
            #    #    print('problem child', distkey, pairs, p)
            #    #if set(pairsbyline[p].tolist()).difference(pairspine): #checking that the line is in more distributions than whats generated via this spine
            #    #if set(pairsbyline[p].tolist()).difference(activepairlist): #checking that the line is in more distributions than whats generated via this spine
            #    if set(map(int, pairsbyline[p].split(',')[:-1])).difference(activepairlist): #checking that the line is in more distributions than whats generated via this spine
            #        competinglines.add(p)
        #activelines = list(set(itertools.chain(*(pairkeys[i] for i in activepairlist))))
        #not amazing performance from this, it helped a bit but didn't offer much else
        #I'm going to keep it for now, it decreases agreed charge matches on ms2 hits but increases number of dists and multi-charge states which is interesting
        #^this goes along with taking the set difference of activepairlist above as well instead of pairspine
        rbounds = regions[flatdist,2:4]
        #rstack = boundary_stack(rbounds)
        rstack = overlap_counts(rbounds.tolist())
        if rstack > 0:
            #outputdist = tuple(flatdist)
            flatdistgroups.add(tuple(activepairlist))
            #if len(outputdist) > len(maximalsuperset):
            #    maximalsuperset = activepairlist
            #if competinglines and len(activepairlist) > 1:
            #    #unstackedboundaries = rbounds - rbounds.min(axis=1)[:,None]
            #    #ustack = boundary_stack(unstackedboundaries)
            #    #^leaving this one here as it retains the best functionality, butrstack is using a new function to allow for better distribution filtering, this seems to work pretty well so far
            #    #^actually i can just get rid of it :)
            #    #ustack = overlap_counts(unstackedboundaries)
            #    #maybe just for loop and extend a list to make ^this faster?
            #    scorearray = np.array(scores)
            #    distmean = scorearray[:,0].mean()
            #    #rtmultiplier = scorearray[:,2].prod()
            #    #rtmultiplier = rstack / ustack
            #    #rtmultiplier = rstack
            #    decreasingmultiplier = scorearray[:,3].sum() + 1
            #    #slen = scorearray.size #I did this by mistake, but it should just be a linear x4 scaling
            #    slen = len(scorearray)
            #    #if decreasingsum > 0:
            #    #    decreasingmultiplier = decreasingsum
            #    #else:
            #    #    decreasingmultiplier = 1
            #    for pair, score in zip(activepairlist, scores): 
            #        if paircharges[pair] > 1: # a lot of bad 1+ matches get high priority from this, I essentially want less 1+ than 3+ and this helps
            #            dist, ddiff, rtoffset, decs = score
            #            #meandiff = abs(distmean - dist) / (slen + 1) 
            #            meandiff = abs(distmean - dist) / slen
            #            #distdiff = meandiff - meandiff * rtoffset
            #            #distdiff = meandiff * decreasingmultiplier
            #            distdiff = meandiff * (2**decreasingmultiplier)
            #            #datadiff = ddiff - ddiff * rtoffset
            #            #datadiff = ddiff - ddiff * rtmultiplier
            #            #datadiff = ddiff / rtmultiplier
            #            datadiff = ddiff / rstack
            #            #datadiff = ddiff * decreasingmultiplier * rtmultiplier
            #            scorelist = tuple([distdiff, datadiff])
            #            #distributionscoredict[pairk] = scorelist
            #            distributionscoredict[pair].append(scorelist)
            #            for p in pairkeys[pair]:
            #                if p in competinglines:
            #                    if not pair in distributionscoresbyline[p]: #avoiding set use to save memory
            #                        distributionscoresbyline[p].append(pair)
    
    #if maximalsuperset:
    #    removals = set()
    #    maxpass = True
    #    mlen = len(maximalsuperset)
    #    maximalsuperset = set(maximalsuperset)
    #    for line in maximalsuperset:
    #        for superset in minimaldistgroupsbyline[line]:
    #            slen = len(superset)
    #            if slen > mlen:
    #                if maximalsuperset.issubset(superset):
    #                    maxpass = False
    #                    break
    #            elif slen < mlen:
    #                if maximalsuperset.issuperset(superset):
    #                    removals.add((line, superset))
    #            #else: #lengths equal
    #            #    #i actually don't have to care about this
    #            #    if superset == maximalsuperset:
    #            #        maxpass = False
    #            #        break
    #    for line, superset in removals:
    #        minimaldistgroupsbyline[line].remove(superset)
    #        try:
    #            minimaldistgroups.remove(superset)
    #        except KeyError:
    #            #a different line removed the same group, its fine
    #            pass
    #    if maxpass:
    #        maximalsuperset = tuple(maximalsuperset)
    #        minimaldistgroups.add(maximalsuperset)
    #        for line in maximalsuperset:
    #            minimaldistgroupsbyline[line].add(maximalsuperset)

print(time() - t8, 'distribution scoring')

#re-working boundary_stack into overlap_counts
#[[40.64839483 42.96961734]
# [40.66058248 42.95148015]
# [41.68309719 42.2650513 ]
# [41.41672452 42.36361071]
# [40.83700746 42.55174051]
# [41.28945501 42.44838942]
# [40.93471109 42.59416294]]
#-184.04306994618005
##input/output of boundary_stack, it needs work
#1152518 #a line
#1152732 #a line
#(196692, 1152518, 1152564, 1152641, 1152732, 1152911, 1152968) #a line linkage that should work but isn't

#print(len(pairkeys), 'pairkeys')
#print(len(subisomasses), 'subisomasses')
#print(len(subisogroups), 'subisogroups')
#print(len(pairsbyline), 'pairsbyline')
#print(len(scoresbypair), 'scoresbypair')
#print(len(connectionspine), 'connectionspine')
#print(len(connectionpairkeys), 'connectionpairkeys')
#print(len(subgroups), 'subgroups')
#print(len(distributionscoresbyline), 'distributionscoresbyline')
#print(len(distributionscoredict), 'distributionscoredict')
#
#del distributionscoresbyline
#del distributionscoredict
#del flatdistgroups
#del preservedranks
#del pairkeys
#del subisomasses
#del subisogroups
#del pairsbyline
#del scoresbypair
#del connectionspine
#del connectionpairkeys
#del subgroups
#gc.collect()

t11 = time()

dr =  0 #easier to work with dr here rather than re-index the finalized distribution just to get the max index for finaldefiniteind afterwards, the order of these aren't important
multidists = defaultdict(dict) #charge: distid: lines
#for dist in minimaldistgroups:
for dist in flatdistgroups:
    charge = paircharges[dist[0]]
    lines = itertools.chain.from_iterable(map(pairkeys.get, dist))
    multidists[charge][dr] = list(lines)
    dr += 1
finaldefiniteind = dr


#^keep all of this
#multidists becomes multidists for the non-deterministic distributions
#or rather, i can move this all, officially, to post-peptide-ID where I can dictate the dists that have been identified a-priori
#this will still be a competitive process, ie i won't do any hyper-iterations of isotopomers in a single distribution when there's 3 potential ones for a single spot
#^the best one will remain i suppose

#or, from flatdistgroups i'll superset merge to remove all smaller subsets that don't fit in anywhere, and retain a minimal set of all distributions

#superset downmerge?
#organize by length
#search downward by length to find things that are a subset -> remove them

#nt = time()
#
#distributionorganizer = defaultdict(set) #dist length: [dists]
#for f in minimaldistgroups:
#    distributionorganizer[len(f)].add(tuple(f))
#
#dsizes = sorted(distributionorganizer, reverse=True)
#for n, size in enumerate(dsizes):
#    removals = defaultdict(set) #distsize: [removals]
#    for dset in distributionorganizer[size]:
#        for subsize in dsizes[n+1:]:
#            subsets = distributionorganizer[subsize]
#            for subset in subsets:
#                if subset not in removals[subsize]:
#                    if set(dset).issuperset(subset):
#                        removals[subsize].add(subset)
#    for s, rsets in removals.items():
#        for rset in rsets:
#            distributionorganizer[s].remove(rset)
#
#print(time() - nt, 'distributions minimized')
#
#finalsets = set(itertools.chain.from_iterable(distributionorganizer.values()))
#finalsets == minimaldistgroups
#TEST PASSED!

#make dists and check for multiple charge states across all the dists now
#put them through scanmatching, decoygeneration, and subformulagrouping -> BAM


#process subisotopomer mainmass focussing here, don't allow subisos to show up as masses, but do allow their intensities to contribute to a new area value to be used for intensity ranking
distributionmasses = {} #distid: ordered masses
distributioncharges = {} #distid: charge
distributionsoflines = defaultdict(list) #line: distid
distributionsoflinemasks = {} #linemask: distid
linesofdistributions = {} #distid: mass-ordered linedkeys
linemasksofdistributions = {} #distid: mass-ordered linemasks
linemasksbylinedistributions = defaultdict(dict) #distid: mass-ordered lines: linemask
distributiontimelimits = {} #distid [starting rt, ending rt]
distributionintensities = {} #distid: mass-ordered intensities
#distributionsbycharge = defaultdict(dict) #charge: dists: mass-ordered linekeys
distributionsbycharge = defaultdict(list) #charge: dists: mass-ordered linekeys
linesbylinemask = {} #linemask: line
mask = 0
for charge, dists in multidists.items():
    for dist, lines in dists.items():
        dmasses = regions[lines,7]
        lineorder = dmasses.argsort().tolist()
        sortedlines = [lines[i] for i in lineorder]
        sortedmasses = regions[sortedlines,7]
        dintensities = regions[sortedlines,5]
        rtlimits = regions[sortedlines,2:4] 
        masses = (sortedmasses * charge) - (proton * charge)
        minrt = rtlimits.min()
        maxrt = rtlimits.max()
        distributionmasses[dist] = masses
        distributioncharges[dist] = charge
        sortedmasks = []
        for line in lines:
            linesbylinemask[mask] = line
            distributionsoflines[line].append(dist)
            distributionsoflinemasks[mask] = dist
            linemasksbylinedistributions[dist][line] = mask
            sortedmasks.append(mask)
            mask += 1
        linesofdistributions[dist] = sortedlines
        linemasksofdistributions[dist] = sortedmasks
        distributiontimelimits[dist] = [minrt, maxrt]
        distributionintensities[dist] = dintensities
        #distributionsbycharge[charge][dist] = sortedlines
        distributionsbycharge[charge].append(dist)

#subisomasses = {} #lineuid: subisogroup
#subisogroups = defaultdict(lambda: defaultdict(list)) #subiso group: max charge for pair: [pairkeys]

#subiso implementation goes here:
#passing test would be that every subiso of a major-zone would match with every other ion's respective zones to the correct charge, and RT overla > 0.75 for each zone
#the subiso radius can expand via partner radii at other zones that have been matched by some original zone
#the only problem I have is it there's like a 1, 2, and 3-charge, they don't all work out as multiples so I need a way of picking whether it goes to 2 or 3, maybe by distance, after by number of complementable zones?
#an additional line as a subiso shouldn't interupt the intensity rank order in some way, look at:
#st = 90
#et = 96
#lmb = 1533
#umb = 1539
#^for why
#so maybe a subiso shouldn't spawn a new increase

#len(subisogroups)
#Out[35]: 9854
#len(subisomasses)
#Out[37]: 17884

foundvals = []
for charge, sgd in multidists.items():
    foundvals.extend(list(itertools.chain(*sgd.values())))
specvals = regiter[:,8].astype(int)
nodists = np.setdiff1d(specvals, foundvals)

nodistmasses = regions[nodists,7]
#nodistintensities = regions[nodists,5]
#nodisttimes = regions[nodists,2:4]
#nodistkeys = np.arange(nodists.size) + di

#nodistkeys = []
#for line in nodists.tolist():
#    dreg = regions[line]
#    dmass = dreg[7] 
#    dintensity = dreg[5]
#    rtlimit = dreg[2:4] 
#    minrt = rtlimit.min()
#    maxrt = rtlimit.max()
#    distributionmasses[dr] = np.array([dmass])
#    distributioncharges[dr] = 0
#    distributionsoflines[line].append(dr)
#    linesofdistributions[dr] = [line]
#    distributiontimelimits[dr] = [minrt, maxrt]
#    distributionintensities[dr] = np.array([dintensity])
#    #distributionsbycharge[charge][dist] = sortedlines
#    distributionsbycharge[0].append(dr)
#    nodistkeys.append(dr)
#    dr += 1 #continuing from multidists count

massranges = regions[:,:2]
minmass = massranges.min() - 1
maxmass = massranges.max() + 1

#sortednodistkeys = np.array(nodistkeys)[nodistmasses.argsort()].tolist()
#sortednodistmasses = np.sort(nodistmasses)

print(time() - t11, 'distribution assembling')
print(time() - t1, 'total to distribution assembling')

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

##formalized analyte information, summarizing all distributions across any charge states
#analytefile = '/'.join((processinglocation, 'analytefactors.pickle'))
#with open(analytefile, 'rb') as pick:
#    distributionsoflines, linesofdistributions = pickle.load(pick)
##distributionsoflines: lineuid: distid
##linesofdistributions: distid: [lineuids ordered by mass]
#
#regionfile = '/'.join((processinglocation, 'regions.pickle'))
#with open(regionfile, 'rb') as pick:
#    regions = pickle.load(pick)
##regions as [minmass, maxmass, mintime, maxtime, # datapoints, peakarea, maxintensity, wmean, lineid]
#regiter = regions[regions[:,4] >= minpoints]
#
#loaderloc = '/'.join((processinglocation, 'trackedgroups.pickle'))
#with open(loaderloc, 'rb') as pick:
#    trackedgroups = pickle.load(pick)

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
regiter = regions[regions[:,4] >= minpoints]
regiter = regiter[regiter[:,2].argsort()].tolist() #sorted by starting time

nt = time()

regioniter = iter(regiter)

regminrt = -1

linesofscans = defaultdict(list) #ms2 scan index: [lineuids within window]
scansoflines = defaultdict(list) #lineuid: [ms2 scan indices]

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
        regid = int(reg[8])
        regpool.append(regid)
    regremovals = []
    for r in regpool: #this loop might be worth simple concurrency using mp.Manager().list() for regremovals
        treg = regions[r]
        trmaxrt = treg[3]
        if trmaxrt < prt:
            regremovals.append(r)
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

for scan, lines in linesofscans.items():
    linesofscans[scan] = tuple(sorted(lines))
linesofscans = dict(linesofscans)

print(time() - nt, 'windows managed')

#it might be worth visualizing these to see if these are linemodel errors
blankscans = len(precursorcoordinates) - len(linesofscans)
if blankscans > 0:
    blankpercent = blankscans / len(precursorcoordinates)
    print('your instrument produced', blankscans, f'MS2 scans that targeted nothing within the minimum point threshhold of {minpoints} datapoints,', f'{round(blankpercent, 4)}% of all MS2 scans')

precursordict = {}
for pc in precursorcoordinates:
    precursordict[pc[-1]] = pc[:-1]

nt = time()

#assigning relative intensity %'s of each MS1 distribution for each MS2 window
distributionswithscans = set() #distributions with an MS2 scan
lineintensitiesofscans = defaultdict(lambda: defaultdict(dict)) #scan: line: raw area of ms1 line, average of 2 flanking points
linepercentagesofscans = defaultdict(lambda: defaultdict(dict)) #scan: line: % of scan intensity input
scansums = defaultdict(float) #scan: sum area used in lineintensitiesofscans

#these 3 below are keeping track of which line from which distribution [at each charge] gives the most intense MS2 sampling based on MS1 intensity
maxintensitysampleofdists = defaultdict(float) #distid: max intensity
maxintensitylinesofdists = {} #distid: (line, scan)
premaxsampledistributionsoflines = {} #distid: line

for line, scans in scansoflines.items():
    distids = distributionsoflines[line]
    distributionswithscans.update(distids)
    linegroup = trackedgroups[line]
    linetimes = linegroup[:,1]
    linemasses = linegroup[:,0]
    lineintensity = linegroup[:,2]
    lmax = linemasses.max()
    lmin = linemasses.min()
    for scan in scans:
        pcoords = precursordict[scan]
        rt = pcoords[0]
        #PROBLEM here, this assumes the time difference is the same between the two
        #^there should be a time-based extrapolation here
        #i'm leaving it for later because its simple and probably won't change much
        #^upon seeing MS1/MS2 intensity correlations, this is absolutely useless anyways
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
        #if distid >= 0:
        for distid in distids:
            if sampleintensity > maxintensitysampleofdists[distid]:
                maxintensitysampleofdists[distid] = sampleintensity
                maxintensitylinesofdists[distid] = line, scan
                premaxsampledistributionsoflines[distid] = line
        lineintensitiesofscans[scan][line] = sampleintensity
        scansums[scan] += sampleintensity

#turning areas into percents
for scan, lines in lineintensitiesofscans.items():
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

#scansoflinesfile = '/'.join((processinglocation, 'scansoflines.pickle'))
#with open(scansoflinesfile, 'wb') as pick:
#    pickle.dump(scansoflines, pick)
#
#linesofscansfile = '/'.join((processinglocation, 'linesofscans.pickle'))
#with open(linesofscansfile, 'wb') as pick:
#    pickle.dump(linesofscans, pick)
#
#distributionswithscansfile = '/'.join((processinglocation, 'distributionswithscans.pickle'))
#with open(distributionswithscansfile, 'wb') as pick:
#    pickle.dump(distributionswithscans, pick)
#
#lineintensitiesofscansfile = '/'.join((processinglocation, 'lineintensitiesofscans.pickle'))
#with open(lineintensitiesofscansfile, 'wb') as pick:
#    pickle.dump(lineintensitiesofscans, pick)
#
#linepercentagesofscansfile = '/'.join((processinglocation, 'linepercentagesofscans.pickle'))
#with open(linepercentagesofscansfile, 'wb') as pick:
#    pickle.dump(linepercentagesofscans, pick)
#
#maxintensitylinesofdistsfile = '/'.join((processinglocation, 'maxintensitylinesofdists.pickle'))
#with open(maxintensitylinesofdistsfile, 'wb') as pick:
#    pickle.dump(maxintensitylinesofdists, pick)
##maxintensitylinesofdists = {} #distid: (line, scan)
#
#maxsampledistributionsoflinesfile = '/'.join((processinglocation, 'maxsampledistributionsoflines.pickle'))
#with open(maxsampledistributionsoflinesfile, 'wb') as pick:
#    pickle.dump(maxsampledistributionsoflines, pick)
##maxsampledistributionsoflines = {} #line: distid



#distributionmatching

#distributionswithscansfile = '/'.join((processinglocation, 'distributionswithscans.pickle'))
#with open(distributionswithscansfile, 'rb') as pick:
#    distributionswithscans = pickle.load(pick)
##distributionswithscans = defaultdict(list) #distid: [spectra across all lines and charge states]
#
#scansoflinesfile = '/'.join((processinglocation, 'scansoflines.pickle'))
#with open(scansoflinesfile, 'rb') as pick:
#    scansoflines = pickle.load(pick)
##scansoflines = defaultdict(list) #lineuid: [ms2 scan indices]
#
#analytefile = '/'.join((processinglocation, 'analytefactors.pickle'))
#with open(analytefile, 'rb') as pick:
#    distributionsoflines, linesofdistributions = pickle.load(pick)

def radius_neighbors(baselist, flylist, ppmmod):
    b = 0
    pool = []
    matches = {} #flylist index: [baselist indices]
    biter = enumerate(baselist)
    for fn, f in enumerate(flylist):
        ftol = f * ppmmod
        fmin = f - ftol
        fmax = f + ftol
        removals = []
        submatches = []
        for pi, pb in pool:
            if pb < fmin:
                removals.append([pi, pb])
            elif pb <= fmax:
                submatches.append(pi)
        for r in removals:
            pool.remove(r)
        while b <= fmax:
            try:
                i, b = next(biter)
                if b >= fmin:
                    pool.append([i, b])
                    if b <= fmax:
                        submatches.append(i)
            except StopIteration:
                break
        if submatches:
            matches[fn] = submatches
    return matches

nt = time()

print(time() - nt, 'formulas loaded')
nt = time()

librarykeys = []
librarymasses = []
librarymassdict = {} #lid: [masses]
librarypositions = {} #lid: [indices]
libraryintensities = {} #lid: [intensities]
libraryintensityranks = {} #lid: [intensityranks]
for f, (masses, intensities) in sumabundances.items():
    k = formulaidentifiers[f]
    librarymassdict[k] = masses
    librarypositions[k] = list(range(masses.size))
    libraryintensities[k] = intensities
    intensityranks = np.abs(intensities.argsort().argsort() - intensities.size + 1)
    libraryintensityranks[k] = intensityranks
    librarykeys.extend(itertools.repeat(k, masses.size))
    librarymasses.extend(masses.tolist())

librarykeys = np.array(librarykeys)
librarymasses = np.array(librarymasses)

librarykeys = librarykeys[librarymasses.argsort()]
librarymasses = np.sort(librarymasses)

distributionkeys = []
distributionmasslist = []
distributionintensityranks = {} #did: [intensityranks]
for k, masses in distributionmasses.items():
    charge = distributioncharges[k]
    intensities = distributionintensities[k]
    intensityranks = np.abs(intensities.argsort().argsort() - intensities.size + 1)
    distributionintensityranks[k] = intensityranks
    distributionkeys.extend(itertools.repeat(k, masses.size))
    distributionmasslist.extend(masses.tolist())

distributionkeys = np.array(distributionkeys)
distributionmasslist = np.array(distributionmasslist)

distributionkeys  = distributionkeys[distributionmasslist.argsort()]
distributionmasslist = np.sort(distributionmasslist)

ppmmod = ppmtolerance / 1000000

matches = radius_neighbors(librarymasses.tolist(), distributionmasslist.tolist(), ppmmod)

matchorganizer = defaultdict(list)
for k, lkeys in matches.items():
    dk = distributionkeys[k]
    matchorganizer[dk].extend(librarykeys[lkeys])

for k in list(matchorganizer):
    matchorganizer[k] = np.array(list(set(matchorganizer[k])))

#linepositionsbyformula = defaultdict(lambda: defaultdict(set)) #formula: matchable library positions of a distribution: [lines that have ms2 scans]
linemaskpositionsbyformula = defaultdict(lambda: defaultdict(set)) #formula: matchable library positions of a distribution: [lines that have ms2 scans]
#librarymatchesbydistribution = defaultdict(list) #distribution key: [library keys]

#i need to add a layer of complexity to make each lineid in each distribution unique -> but ultimately leading back to an actual lineid
#so that when i label the lineid's in linepositionsbyformula they always lead to the correct distribution

zeroscores = []
nonzeroscores = []
distlist = []
lmatches = 0
dmatches = 0
for dk, lkeys in matchorganizer.items():
    if dk in distributionswithscans:
        dmasses = distributionmasses[dk]
        dsize = dmasses.size
        tx = 0
        for lk in lkeys.tolist(): #can this loop be made concurrent?
            lmasses = librarymassdict[lk]
            lsize = lmasses.size
            
            leftoffset = int(round(lmasses.tolist()[0] - dmasses.tolist()[0]))
            distlist.append(leftoffset)
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
            le = li + maxsize
            lrange = le - li
            #if lrange > 1: #at least 2 matches
            lintranks = libraryintensityranks[lk][li:le]
            if 0 in lintranks: #the top library rank is included
                re = ri + maxsize
                dorders = distributionintensityranks[dk][ri:re].tolist()
                lorders = libraryintensityranks[lk][li:le].tolist()
                orderdiffs = [abs(i-j) for i, j in zip(dorders, lorders)]
                allowance = sum(orderdiffs)
                #
                #dints = distributionintensities[dk][ri:re].tolist()
                #lints = libraryintensities[lk][li:le].tolist()
                #dsum = sum(dints)
                #lsum = sum(lints)
                #dnorm = [i / dsum for i in dints]
                #lnorm = [i / lsum for i in lints]
                #intensitydiff = [d - l for d, l in zip(dnorm, lnorm)]
                #idmean = sum(intensitydiff) / maxsize
                #intensitydiffs = [abs(idmean - i) for i in intensitydiff]
                #meanintensitydiff = sum(intensitydiffs) / maxsize
                if allowance == 0: #complete heirarchical match
                    #zeroscores.append(meanintensitydiff)
                    #librarymatchesbydistribution[dk].append(lk)
                    #distlines = linesofdistributions[dk][ri:re]
                    distlinemasks = linemasksofdistributions[dk][ri:re]
                    positions = librarypositions[lk][li:le]
                    formula = distributionidentifiers[lk]
                    for linemask, pos in zip(distlinemasks, positions):
                        if linesbylinemask[linemask] in scansoflines:
                            linemaskpositionsbyformula[formula][pos].add(linemask)
                    tx += 1
                #else:
                #    #keep scores of other distributions
                #    nonzeroscores.append(meanintensitydiff)
        if tx > 0:
            lmatches += tx
            dmatches += 1

print(time() - nt, 'matches assembled')
print('library matches:', lmatches)
print('dist matches:', dmatches)

for k, v in linemaskpositionsbyformula.items():
    for sk, sv in v.items():
        v[sk] = tuple(sv)
    linemaskpositionsbyformula[k] = dict(v)
linemaskpositionsbyformula = dict(linemaskpositionsbyformula)

#linepositionsbyformulafile = '/'.join((processinglocation, 'linepositionsbyformula.pickle'))
#with open(linepositionsbyformulafile, 'wb') as pick:
#    pickle.dump(linepositionsbyformula, pick)
