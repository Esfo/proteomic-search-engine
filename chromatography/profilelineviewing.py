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
import bisect
import heapq
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

#mzmlfile = '/home/sfo/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
mzmlfile = '/home/sfo/store/data/PXD051214/mzMLs/JMM-6.mzML'
#mzmlfile = '/store/flowcharacterizations/round3/mzMLs/200901_1s-dyn-300-200_R0.mzML'
#mzmlfile = '/store/flowcharacterizations/round5/mzMLs/20210312_E5_CG_high_tw1.mzML'

#librarylocation = '/home/sfo/data/proteomics/fastas/search-db'
librarylocation = '/home/sfo/data/proteomics/fastas/iodoacetamide-search'
proteome = 'Human_Homo_sapien'

basefolder = '/'.join((mzmlfile.split('/')[:-2]))
basefile = mzmlfile.split('/')[-1].split('.mzML')[0]

processinglocation = '/'.join((basefolder, 'investigations', basefile))

#isofactorfile = '/home/sfo/data/proteomics/fastas/isotope-arrays/human_isotopes-6-50_miss-1_ss50/isofactors.pickle'
#with open(isofactorfile, "rb") as pick:
#    subisotopicdifferences, newinclimit, steplimit, uppermasslimit = pickle.load(pick)

environment_partial = partial(lmdb.Environment, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=99999, max_spare_txns=1, lock=True)

#with sq.SqliteDict(libraryfile, tablename='isofactors', flag='r') as db:
#    subisomax = db[proteome]['subisomax']
#    newinclimit = db[proteome]['newinclimit']
#    steplimit = db[proteome]['steplimit']
#    uppermasslimit = db[proteome]['uppermasslimit']
with environment_partial(librarylocation) as env:
    #defaults = env.open_db('defaults'.encode())
    #with env.begin(write=False) as txn:
    #    with txn.cursor(defaults) as cursor:
    #        samplesize = int(cursor.get('samplesize'.encode()).decode())
    parameters = env.open_db('isofactors'.encode())
    with env.begin(write=False) as txn:
        with txn.cursor(parameters) as cursor:
            parameterbytes = cursor.get(proteome.encode())
            parameterdict = dict(eval(parameterbytes.decode()))
            subisomax = float(parameterdict['subisomax'])
            #newinclimit = float(parameterdict['newinclimit'])
            #steplimit = float(parameterdict['steplimit'])
            uppermasslimit = float(parameterdict['uppermasslimit'])

newinclimit = 0.1
steplimit = 0.5

proton = 1.00727647

minmovinginds = 10
deadsignal = 20 #number of scans without data
#deadsignal = 0.17 #minutes, ~10 seconds
#^there might be potential to soft-code this via times taken from scan intervals

minpoints = 3
chargetolerance = 0.1 #lesson learned: these differences DO get divided across charge states, if you normalize everything back to base mass without a charge, the errors become more consistent. They're smaller errors for higher charges etc. so going by percent here is FINE!

#subisodiffs = np.array(list(subisotopicdifferences))[:,None]
#subisomax = subisodiffs.max()
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

#age-old classic https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def find_nearest(array, value):
    array, value = np.array(array), np.array(value).reshape(-1,1)
    idx = np.abs(array - value).argmin(axis=1)
    return idx

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

def arg_coord_rectangle_overlap(rec, coords):
    tops, bottoms, lefts, rights = coords.transpose()
    c1 = rec[2] < rights
    c2 = rec[3] > lefts
    c3 = rec[0] < bottoms
    c4 = rec[1] > tops
    overlaps = np.argwhere(c1 & c2 & c3 & c4)
    return overlaps.flatten()

def radius_neighbors_hard_tolerance(baselist, flylist, ftol):
    b = 0
    pool = []
    matches = {} #flylist index: [baselist indices]
    biter = enumerate(baselist)
    for fn, f in enumerate(flylist):
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

def boundary_finding(fmaxes, array):
    fmaxiter = fmaxes.copy().tolist()
    fmaxiter = np.append(0, fmaxiter)
    fmaxiter = np.append(fmaxiter, len(array)-1)
    peakbounds = []
    for n, l in enumerate(fmaxiter[:-1]):
        r = fmaxiter[n+1] + 1
        if n > 0:
            rightseries = array[l:r]
            rightacc = np.minimum.accumulate(rightseries)
            rtrimmer = rightseries <= rightacc
            rightestimate = np.trim_zeros(rtrimmer, trim='b').size
            nr = l + rightestimate
            rightseries = array[l:nr]
            rcutoff = np.where(rightseries == rightseries.min())[0][0]
            rightbound = l + rcutoff + 1
            peakbounds[-1].append(rightbound)
        
        if n < len(fmaxiter[:-1]) - 1:
            leftseries = array[l:r]
            leftacc = np.flip(np.minimum.accumulate(np.flip(leftseries)))
            ltrimmer = leftseries <= leftacc
            leftestimate = np.trim_zeros(ltrimmer, trim='f').size
            nl = r - leftestimate
            leftseries = array[nl:r]
            lcutoff = np.where(leftseries == leftseries.min())[0][-1]
            leftbound = nl + lcutoff
            peakbounds.append([leftbound])
    
    peakbounds = np.asarray(peakbounds)
    peakparameters = np.vstack((peakbounds[:,0], fmaxes, peakbounds[:,1])).transpose()
    peakparameters = np.unique(peakparameters, axis=0)
    return peakparameters

def minpoint_reduction(barray, mindist):
    extramaxes = set()
    mask = np.repeat(False, barray.size)
    #narray = array.copy()
    while True:
        narray = barray[~mask]

        forwardmaxcheck = np.append(narray[:-1] > narray[1:], False)
        #backwardmaxcheck = np.append(False, narray[1:] > narray[:-1])
        backwardmaxcheck = np.append(forwardmaxcheck[0], narray[1:] > narray[:-1])
        forwardmaxcheck[-1] = backwardmaxcheck[-1]

        forwardmincheck = np.append(narray[:-1] < narray[1:], False)
        #backwardmincheck = np.append(False, narray[1:] < narray[:-1])
        backwardmincheck = np.append(forwardmincheck[0], narray[1:] < narray[:-1])
        forwardmincheck[-1] = backwardmincheck[-1]

        newmask = np.logical_and(forwardmincheck, backwardmincheck)
        mins = np.where(newmask)[0]
        #mins = np.where(np.logical_and(forwardmincheck, backwardmincheck))[0]
        maxes = np.where(np.logical_and(forwardmaxcheck, backwardmaxcheck))[0]
        #extremadistances = (np.abs(maxes - mins.reshape(-1,1)) < mindist)
        #extremadistances = (np.abs(maxes - maxes.reshape(-1,1)) < mindist)
        #np.fill_diagonal(extremadistances, False)
        extremas = np.sort(np.append(mins, maxes))
        #textremas = extremas + mask.cumsum()[~mask][extremas] #using true distance didn't work out well for large peaks, over-found too many
        #extremadistances = (np.abs(np.diff(extremas)) < mindist) #brings forth incorrect distances
        extremadistances = (np.abs(extremas - extremas[:,None]) < mindist)
        np.fill_diagonal(extremadistances, False)
        
        separatedextremas = extremas[~extremadistances.any(axis=0)]
        if separatedextremas.size > 0:
            maxestomaintain = separatedextremas[np.isin(separatedextremas, maxes)]
            maxestomaintain = (maxestomaintain + mask.cumsum()[~mask][maxestomaintain]).tolist()
            extramaxes.update(maxestomaintain)
            minstomaintain = separatedextremas[np.isin(separatedextremas, mins)]
            newmask[minstomaintain] = False
            if minstomaintain.size > 0:
                mins = np.delete(mins, np.where(mins == minstomaintain[:,None])[1])
        
        adjacentextremas = extremadistances.any()
        if adjacentextremas and mins.size > 0:
            maskinds = np.argwhere(~mask)[np.argwhere(newmask)].flatten()
            mask[maskinds] = True
        else:
            break
    
    if not maxes.size:
        maxes = narray.argmax()
    
    fmaxes = maxes + mask.cumsum()[~mask][maxes]
    fmaxes = np.unique(np.append(fmaxes, list(extramaxes))).astype(int)
    return fmaxes

def axis_peaks(array, mindist):
    #boxcararray = boxcar_mean_replacement(array, boxcarlength)
    maxes = minpoint_reduction(array, mindist)
    peakparameters = boundary_finding(maxes, array) #reverted the r + 1 into this function
    peakparameters = peakparameters.tolist()
    
    #peakparameters = peakparameters + (~noisemask).cumsum()[noisemask][peakparameters]
    #peakparameters = peakparameters.tolist()
    
    #finalparameters = []
    #trimming zeros that can come from the boxcar transforms
    #for l, m, r in peakparameters:
    #    while array[l] >= array[l+1]:
    #        l += 1
    #    
    #    while array[r] >= array[r-1]:
    #        r -= 1
    #    r += 1 #setting up for slice indexing
    #    
    #    if r > l:
    #        m = array[l:r].argmax() + l
    #        finalparameters.append([l, m, r])
    
    #return finalparameters, boxcararray
    return peakparameters

t1 = time()
msrun = mzml.MzML(mzmlfile, dtype=np.float64)

timearray = []
scan = next(msrun)
mza = scan['m/z array']
previousdata = mza.copy()
model = spatial.KDTree(mza[:,None])
scancount = 1

intensities = scan['intensity array']
rt = scan['scanList']['scan'][0]['scan start time'].real
timearray.append(rt)
retentiontimes = np.repeat(rt, mza.size)
coords = np.stack((mza, retentiontimes, intensities), axis=1).reshape(mza.size, 1, 3).tolist()

uids = np.arange(mza.size).tolist()
uidcount = max(uids) + 1

trackedgroups = {} #uniqueid: [[masses, rt-inds, intensity/injection times],[...]]
trackedma = {} #latest moving average mass of trackedgroup: lineuid
linedeletioncounter = defaultdict(int) #lineuid: notmatched count
groupmovingaverages = {} #lineuid: latest moving average of line
groupdifftoma = {} #lineid: moving difference to moving average
groupranges = {} #uniqueid: [minmass, maxmass]
modeltracking = {} #scan: number of masses being [added, matched, nonmatched, removed]

modeltracker = [0, 0, 0, 0]
modeltracker[0] += mza.size
modeltracking[scan['index']] = modeltracker

flatmasslist = mza.tolist()
trackedma.update(zip(flatmasslist, uids))
trackedgroups.update(zip(uids, coords))
groupmovingaverages.update(zip(uids, flatmasslist))
elen = len(uids)
groupdifftoma.update(zip(uids, np.zeros(elen).tolist()))
groupranges.update(zip(uids, np.stack((mza, mza), axis=1).tolist()))

modify = False
widestmassrange = 0 #a tracked float of the widest mass range
wides = []
linecorrections = []

p1, p2, p3, p4, p5, p6, p7, p8 = 0, 0, 0, 0, 0, 0, 0, 0

longest = 0
longestscan = 0

roundcutoff = 0
for scan in msrun:
    if scan['ms level'] == 1:
        pt2 = time()
        trackedkeys = {} #latest mass in a trackedgroup: lineid
        scanlist = scan['scanList']['scan'][0]
        rt = scanlist['scan start time'].real
        timearray.append(rt)
        
        intensities = scan['intensity array']
        
        modeltracker = [0, 0, 0, 0]
        mza = scan['m/z array']
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
        
        p2 += time() - pt2
        pt3 = time()
        
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
        
        p3 += time() - pt3
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
        
        p4 += time() - pt4
        pt5 = time()
        
        sorteddistances = np.sort(massdist)
        mbool = np.arange(sorteddistances.size)[::-1] + 1
        countsums = mbool / sorteddistances.size
        sumcounts = sorteddistances.cumsum() / sorteddistances.sum()
        mincomboind = (countsums + sumcounts).argmin()
        mincombo = sorteddistances[mincomboind]
        #moving average of average of dists under mincombo
        explicitcutoff = sorteddistances[sorteddistances <= mincombo].mean()
        roundcutoff = (roundcutoff * scancount + explicitcutoff) / (scancount + 1)
        
        p5 += time() - pt5
        pt6 = time()
        
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
            #generally, this is good for on-the-fly decision making when the moving target is outside the existing mass range. This dominates later on, where it's more robust
            elif tlen >= minmovinginds:
                oldma = groupmovingaverages[tid]
                madiff = groupdifftoma[tid]
                nma = (oldma * tlen + nf) / (tlen + 1)
                nmadiff = abs(oldma - nma)
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
                else: #first one's not free, but comes at a discount
                    #if (d - roundcutoff) - d * roundcutoff <= roundcutoff: #1154745
                    if d <= roundcutoff * 2: #1154797 and way less lenient
                        oldma = groupmovingaverages[tid]
                        nma = (oldma * tlen + nf) / (tlen + 1)
                        nmadiff = abs(oldma - nma)
                        groupmovingaverages[tid] = nma
                        groupdifftoma[tid] = nmadiff
                        modify = True
            if modify:
                trackedkeys[nf] = tid
                trackedma[nma] = trackedma.pop(f)
                trackedgroups[tid].append(c[0])
                linedeletioncounter[tid] /= 2
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
                trackedgroups[uidcount] = c
                trackedkeys[nf] = uidcount
                trackedma[nf] = uidcount
                groupmovingaverages[uidcount] = nf
                groupdifftoma[uidcount] = 0 #this zero won't bog down any averages, same principle new mechanics
                groupranges[uidcount] = [nf, nf]
                uidcount += 1
                modeltracker[0] += 1 #newly added
                foundremoval = groupmovingaverages[tid]
                foundremovals.append(foundremoval)
        
        p6 += time() - pt6
        pt7 = time()
        
        for fr in foundremovals:
            found.remove(fr)
        nonmatched = np.setdiff1d(previousdata, found)
        nmlen = nonmatched.size - 1
        mzlen = mza.size
        newmodelremovals = []
        #things from previousdata not in found gets +1 to linedeletioncounter
        for n, nm in enumerate(nonmatched.tolist()): #could this loop be concurrent?
            #linekey = trackedkeys[nm]
            linekey = trackedma[nm]
            linedeletioncounter[linekey] += 1
            if linedeletioncounter[linekey] > deadsignal:
                #determine, out of all matched and nonmatched, which fall into a +/- subisomax distance to this movingma, put the lineuids together in a list to be later intersection_merged for line corrections
                newmodelremovals.append(n)
                modeltracker[3] += 1 #removed
                
                trackedgroups[linekey] = np.array(trackedgroups[linekey]) #more efficient memory storage now that it doesn't need to be appended, not much speed compromised but it is a little slower
                
                #collection subisomax-radius lines with overlapping rt's for line corrections
                #correctionradius = set()
                #nmoverlaps = nonmatched[np.abs(nm - nonmatched) <= subisomax].tolist()
                #mzoverlaps = mza[np.abs(nm - mza) <= subisomax].tolist()
                #trying to band-aid the current overlapping signals issue with corrections, seeing if it helps to only pick the closest ones
                
                if n == 0:
                    nmupper = nonmatched[n+1]
                    nmlower = -np.inf
                    nmupdist = abs(nmupper - nm)
                    nmlowdist = 0
                elif n == nmlen:
                    nmupper = np.inf
                    nmlower = nonmatched[n-1]
                    nmupdist = 0
                    nmlowdist = abs(nmlower - nm)
                else:
                    nmupper = nonmatched[n+1]
                    nmlower = nonmatched[n-1]
                    nmupdist = abs(nmupper - nm)
                    nmlowdist = abs(nmlower - nm)
                
                if nmupdist > subisomax:
                    nmupper = np.inf
                if nmlowdist > subisomax:
                    nmlower = -np.inf
                
                mzupperpoint = np.searchsorted(mza, nm) #should be greater than nm
                if mzupperpoint == 0:
                    mzupper = mza[mzupperpoint]
                    mzlower = -np.inf
                    mzupdist = abs(mzupper - nm)
                    mzlowdist = 0
                elif mzupperpoint == mzlen:
                    mzupper = np.inf
                    mzlower = mza[mzupperpoint-1]
                    mzupdist = 0
                    mzlowdist = abs(mzlower - nm)
                else:
                    mzupper = mza[mzupperpoint]
                    mzlower = mza[mzupperpoint-1]
                    mzupdist = abs(mzupper - nm)
                    mzlowdist = abs(mzlower - nm)
                
                if mzupdist > subisomax:
                    mzupper = np.inf
                if mzlowdist > subisomax:
                    mzlower = -np.inf
                
                #these start as np.int64
                if nmlower < mzlower:
                    lower = int(trackedkeys[mzlower])
                elif mzlower < nmlower:
                    lower = int(trackedma[nmlower])
                else: #==, both are -np.inf
                    lower = False
                if nmupper > mzupper:
                    upper = int(trackedkeys[mzupper])
                elif mzupper > nmupper:
                    upper = int(trackedma[nmupper])
                else:
                    upper = False
                
                #because uid can be 0
                lowerint = type(lower) == int
                upperint = type(upper) == int
                if lowerint and upperint:
                    correctionradius = lower, upper
                elif lowerint:
                    correctionradius = lower,
                elif upperint:
                    correctionradius = upper, 
                else:
                    #this should happen now, for things that dont fit either subisomax distance
                    correctionradius = ()
                #both nonmatched and mza are pre-sorted so this removes the need to use vectorized things as long as you only need the nearest points
                #nmkeys = list(map(trackedma.get, nmoverlaps)) #goes to trackedma
                #mzkeys = list(map(trackedkeys.get, mzoverlaps)) #goes to trackedkeys
                #correctionradius.update(nmkeys)
                #correctionradius.update(mzkeys)
                linecorrections.append(correctionradius)
            else:
                modeltracker[2] += 1 #nonmatched
        
        p7 += time() - pt7
        pt8 = time()

        wides.append(widestmassrange)
        nonmatched = np.delete(nonmatched, newmodelremovals)
        currentmasskeys = list(map(trackedkeys.get, mza.flatten().tolist()))
        currentmasses = np.array(list(map(groupmovingaverages.get, currentmasskeys)))
        newtrain = np.sort(np.append(currentmasses, nonmatched, axis=0))[:,None]
        previousdata = np.sort(np.append(currentmasses, nonmatched))
        model = spatial.KDTree(newtrain) #for ms2 tracking, this file only
        modeltracking[scan['index']] = modeltracker
        scancount += 1
        
        p8 += time() - pt8

for cmk in currentmasskeys:
    trackedgroups[cmk] = np.array(trackedgroups[cmk])

nonmatchedkeys = list(map(trackedma.get, nonmatched.flatten().tolist()))
for nmk in nonmatchedkeys:
    trackedgroups[nmk] = np.array(trackedgroups[nmk])

timearray = np.array(timearray)

print(time() - t1, 'line model')

print('~')
print(p1)
print(p2)
print(p3)
print(p4)
print(p5)
print(p6)
print(p7)
print(p8)
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

#trying to change the above linecorrections process to re-work the below isn't going to work because its being intersection_merged so it might create annoyances
#ditch linecorrections altogether, do the first loop of the corrections below as the initial step using regions
#a nearest neighbor that keeps track of a couple different distances
#multiple nn also, one for time and one for mass
#its basically going to be the set up of distributionassembly
#do a simple arg overlap test for min/max rt to every other, then find every 

#execution:
#go outward in order of mass distance at least once in each direction
#check for non-overlapping points, if it fails -> you got nothing

#maybe iterate timepoints and pull in every relevant line, then add each nearest on either side of every line to that lines set of nearest lines
#then from the nearest lines you can rank them in mass order
#you might be able to intersection merge here, then work with the group
#if a signal cant combine with the nearest line, it stops trying

#basically when you go up and down the mass scale
#if you see things that overlap in RT but have no contradicting RT datapoints, then its going to be a good match
#if you can extend a line by its own mass range, and hit another line that fits this budget -> you're good to connect

#one full diameter range outside on the above and below sides of each line will be each individual lines area of reach
#where these areas of reach overlap will be the intersection merge

#calculate the area reach first
#then sort everything by RT
#iterate retention times
#keep an active line pool
#line: [every other line within its area overlap, plus itself]
#intersection merge these dict values
#do a 1:1 comparison of every neighbor in the group, with only their direct mass neighbors

#for time-based inclusion, the mass window shouldn't be interrupted by any signals between the two
#the last [minpoint] datapoints of any line will be used to judge the distance to look outwards
#the further you go the less datapoints you should get
#if you get too many across a distance much larger than anything prior its probably a bad connection
#^so this would be density i guess, you expect density to decrease at a distance
#so that when you link 1-2 datapoints, that's all you get, which would be fine

#IF
#there's no masses within the area search
#   > is there anything within the area search mass range that is distant in time?? ugh how do i do this??
#   > this should be after mass combinations occur i guess?
#   > the less distance that it takes to bridge should mean you can have more density
#everything in the group can combine -> do it?
#   > 


t3 = time()
correctiongroups = intersection_merge(linecorrections)
correctiongroups = [list(i) for i in correctiongroups if len(i) > 1]
print(time() - t3, 'correction group intersection merge')

t4 = time()
timeextension = np.diff(timearray).mean() * minpoints #i'm not a huge fan of this because of the potential for it to connect to completely different things, but there's some shit i just need to connect also...
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
                #below is making sure only appropriately ordered links are connected and further connections aren't skipping over other lines. it's essentially assuring the intersection merge below works on the basis of this as a directional graph
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

for lines in linecorrections:
    linegrid = []
    for line in lines:
        for c in trackedgroups[line].tolist():
            linegrid.append(list(c))
        del trackedgroups[line]
    linegrid = np.array(sorted(linegrid, key=lambda x: x[1]))
    trackedgroups[uidcount] = np.array(linegrid)
    uidcount += 1

groupholder = {}

kl = list(trackedgroups.keys())
for k in kl:
    groupholder[k] = trackedgroups.pop(k)

for n, k in enumerate(kl):
    trackedgroups[n] = groupholder.pop(k)

print(time() - t3, 'line corrections')
t4 = time()

startingpoints = defaultdict(list)
regions = [] #t, b, l, r
for k, a in trackedgroups.items():
    minmass, mintime, mii = a.min(axis=0)
    maxmass, maxtime, mai = a.max(axis=0)
    wmean = (a[:,0] * a[:,2]).sum() / a[:,2].sum()
    #peakarea = np.trapezoid(a[2], a[1])
    peakarea = np.trapezoid(a[:,2], a[:,1])
    maxintensity = a[:,2].max()
    regions.append([minmass, maxmass, mintime, maxtime, len(a), peakarea, maxintensity, wmean, k])
    startingpoints[mintime].append(k)

regions = np.array(regions)

newwides = []
avgwides = []
maxrange = 0
for t in sorted(startingpoints):
    wa = 0
    wc = 0
    for line in startingpoints[t]:
        minmass, maxmass = regions[line,:2]
        massrange = maxmass - minmass
        wa += massrange
        wc += 1
        if massrange > maxrange:
            maxrange = massrange
    newwides.append(maxrange)
    avgwides.append(wa / wc)

print('old max mass width', max(wides))
print('new max mass width', max(newwides))

print(time() - t4, 'new regions -', len(regions))

regionfile = '/'.join((processinglocation, 'regions.pickle'))
with open(regionfile, 'wb') as pick:
    pickle.dump(regions, pick)

saverloc = '/'.join((processinglocation, 'trackedgroups.pickle'))
with open(saverloc, 'wb') as pick:
    pickle.dump(trackedgroups, pick)

saverloc = '/'.join((processinglocation, 'modelinfo.pickle'))
savedbits = [modeltracking, timearray]
with open(saverloc, 'wb') as pick:
    pickle.dump(savedbits, pick)

saverloc = '/'.join((processinglocation, 'roundcutoff.pickle'))
with open(saverloc, 'wb') as pick:
    pickle.dump(roundcutoff, pick)

#line 12748129 at time 5.116067

def location_plot(starttime, endtime, lowermass, uppermass, trackedgroups, regions):
    rgblow = lambda: (np.random.uniform(low=0.6, high=1), np.random.uniform(low=0.9, high=1), np.random.uniform(low=0.7, high=1))
    rgbhigh = lambda: (np.random.uniform(low=0.9, high=1), np.random.uniform(low=0, high=0.1), np.random.uniform(low=0.8, high=0.9))
    deepset = lambda: (np.random.uniform(low=0.2, high=1), np.random.uniform(low=0.4, high=1), np.random.uniform(low=0.1, high=0.8))
    boundrec = [lowermass, uppermass, starttime, endtime]

    plotkeys = arg_coord_rectangle_overlap(boundrec, regions[:,:4]).tolist()

    fig, ax = plt.subplots(figsize=(12,8), facecolor='gray', sharex=True)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_facecolor('gray')
    for k in plotkeys:
        a = trackedgroups[k].transpose()
        ainds = np.logical_and.reduce((a[1] <= endtime, a[1] >= starttime, a[0] >= lowermass, a[0] <= uppermass))
        a = a[:,ainds]
        low, high = rgblow(), rgbhigh()
        cmap = colors.LinearSegmentedColormap.from_list('rgb', [low, high])
        ax.scatter(a[1], a[0], marker='o', c=a[2], s=2, alpha=1, cmap=cmap)
        if a.size > 0:
            ax.plot(a[1], a[0], '-', color=low, linewidth=0.2, alpha=1)
    ax.set_xlabel('minutes')
    ax.set_ylabel('m/z')
