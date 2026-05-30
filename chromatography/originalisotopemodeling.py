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
import networkx
from networkx.algorithms.components.connected import connected_components
from scipy import sparse, integrate, spatial, stats, special
from pandas.api.types import CategoricalDtype
from sklearn.neighbors import NearestNeighbors
from more_itertools import sort_together
from distinctipy import distinctipy as dp
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

mzmlfile = '/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
#mzmlfile = '/store/flowcharacterizations/round3/mzMLs/200901_1s-dyn-300-200_R0.mzML'
#mzmlfile = '/store/flowcharacterizations/round5/mzMLs/20210312_E5_CG_high_tw1.mzML'

librarylocation = '/home/sfo/data/proteomics/fastas/search-db'
proteome = 'Human_Homo_sapien'

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
deadsignal = 20

minpoints = 3
chargetolerance = 0.1 #lesson learned: these differences DO get divided across charge states, if you normalize everything back to base mass without a charge, the errors become more consistent. They're smaller errors for higher charges etc. so going by percent here is FINE!

regionfile = '/store/flowcharacterizations/round3/fileprocessing/200901_fR_400.regions.pickle'
regionfile = '/store/flowcharacterizations/round5/fileprocessing/20210312_E5_CG_high_tw1.regions.pickle'
#with open(regionfile, "rb") as pick:
#    regions = pickle.load(pick)
#
#roundfile = '/store/flowcharacterizations/round5/fileprocessing/20210312_E5_CG_high_tw1.roundcutoff.pickle'
#with open(roundfile, "rb") as pick:
#    roundcutoff = pickle.load(pick)


processfile = '/store/flowcharacterizations/round3/fileprocessing/200901_fR_400.processinfo.pickle'
#with open(processfile, "rb") as pick:
#    timearray, roundcutoff = pickle.load(pick)


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

#https://stackoverflow.com/questions/24398708/slicing-a-numpy-array-along-a-dynamically-specified-axis
def array_slice(a, axis, start, end, step=1):
    return a[(slice(None),) * (axis % a.ndim) + (slice(start, end, step),)]

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

def boundary_stack(rbounds):
    boundarybreaks = np.unique(rbounds)
    boundarystack = np.stack((boundarybreaks[:-1], boundarybreaks[1:]), axis=1)
    rbounds = list(rbounds)
    rsize = len(rbounds)
    finalsum = 0
    for ls, rs in boundarystack.tolist():
        stackslice = rs - ls
        stacksum = 0
        stacklines = 0
        for lb, rb in rbounds:
            if ls < rb and rs > lb:
                boundslice = rb - lb
                stackfrac = stackslice / boundslice
                stacksum += stackfrac
                stacklines += 1
        nonoverlaps = rsize - stacklines
        sover = stacklines - nonoverlaps
        stacknum = sover * (stacksum * stacklines) ** (sover / rsize)
        finalsum += stacknum
    return finalsum


#from https://stackoverflow.com/questions/10481990/matplotlib-axis-with-two-scales-shared-origin
def align_yaxis_np(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = np.array([ax1, ax2])
    extrema = np.array([ax.get_ylim() for ax in axes])
    tops = extrema[:,1] / (extrema[:,1] - extrema[:,0])
    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
        axes, extrema, tops = [a[::-1] for a in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    extrema[0,1] = extrema[0,0] + tot_span * (extrema[0,1] - extrema[0,0])
    extrema[1,0] = extrema[1,1] + tot_span * (extrema[1,0] - extrema[1,1])
    [axes[i].set_ylim(*extrema[i]) for i in range(2)]


#
#Opening the file and extracting data
#

#it would be cool if deadsignal could be soft-coded from intensity

t1 = time()
msrun = mzml.MzML(mzmlfile, dtype=np.float64)


#timedict = {} #haven't used this for anything yet
timearray = []
#injectionarray = [] #haven't used this for anything yet
scan = next(msrun)
mza = scan['m/z array'][:,None]
model = spatial.KDTree(mza)
scancount = 1

intensities = scan['intensity array'][:,None]
#intensityranks = intensities.size - intensities.argsort(axis=0).argsort(axis=0) - 1
#intensitysum = intensities.sum()
rt = scan['scanList']['scan'][0]['scan start time'].real
#timedict[rt] = scancount
timearray.append(rt)
it = scan['scanList']['scan'][0]['ion injection time'].real
#injectionarray.append(it)
#nintensities = intensities / it
retentiontimes = np.repeat(rt, len(mza))[:,None]
#coords = np.hstack((mza, retentiontimes, intensities, nintensities, intensities/intensitysum, intensityranks))[:,:,None].tolist()
coords = np.hstack((mza, retentiontimes, intensities))[:,:,None].tolist()

uids = (np.arange(len(mza))).tolist()
uidcount = max(uids) + 1

trackedgroups = {} #uniqueid: [[masses], [rt-inds], [intensities], [intensities/injection times], [percent intensities of scans], [intensity rank of scan]]
trackedma = {} #latest moving average mass of trackedgroup: lineuid
linedeletioncounter = defaultdict(int) #lineuid: notmatched count
trackedscancount = defaultdict(int) #lineuid: number of scans the line has existed, and has been alive, for
trackedlength = defaultdict(int) #lineuid: number of datapoints, a moving data length
groupmovingaverages = {} #lineuid: latest moving average of line
groupdifftoma = {} #moving differrene to moving average
groupranges = {} #uniqueid: [minmass, maxmass]
modeltracking = {} #scan: number of masses being [added, matched, nonmatched, removed]

modeltracker = [0, 0, 0, 0]
modeltracker[0] += len(mza)
modeltracking[scan['index']] = modeltracker

flatmasslist = mza.flatten().tolist()
trackedma.update(zip(flatmasslist, uids))
trackedgroups.update(zip(uids, coords))
groupmovingaverages.update(zip(uids, flatmasslist))
elen = len(uids)
groupdifftoma.update(zip(uids, np.zeros(elen).tolist()))
groupranges.update(zip(uids, np.hstack((mza, mza)).tolist()))
trackedscancount.update(zip(uids, np.ones(elen).tolist()))
trackedlength.update(zip(uids, np.ones(elen).tolist()))

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

#rcos = []
roundcutoff = 0
for scan in msrun:
    if scan['ms level'] == 2:
        pt1 = time()
        
        precursorinfo = scan['precursorList']['precursor'][0]
        charge = int(precursorinfo['selectedIonList']['selectedIon'][0]['charge state'])
        selectionwindow = precursorinfo['isolationWindow']
        precmass = selectionwindow['isolation window target m/z'].real
        lowerbound = precmass - selectionwindow['isolation window lower offset'].real
        upperbound = precmass + selectionwindow['isolation window upper offset'].real
        dist, trainind = model.query(precmass)
        modelmass = newtrain[trainind][0]
        lineuid = trackedma[modelmass]
        #trainind references index of newtrain, -> get mass -> input to trackedma -> lineuid
        scindex = int(scan['index'])
        #maybe if dist is > roundcutoff, it's not kept? there are some cases where this may be a good idea bc of poor mass selection
        precursordistmatches[scindex] = dist
        precursorscanmatches[scindex] = lineuid
        precursorlinematches[lineuid].append(scindex)
        precursorapparentcharge[lineuid].append(charge)
        precursorcoordinates[scindex] = np.array([rt, precmass, lowerbound, upperbound])
        
        p1.append(time() - pt1)
    #if scancount >= 18777:
    #    break
    #while True:
    #    scan = next(msrun)
    #    if scan['ms level'] == 1:
    #        break
    if scan['ms level'] == 1:
        pt2 = time()
        trackedkeys = {} #latest mass in a trackedgroup
        
        modeltracker = [0, 0, 0, 0]
        previousdata = model.data
        mza = scan['m/z array'][:,None]
        massdist, catches = model.query(mza)
        
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
        
        intensities = scan['intensity array'][:,None]
        intensityranks = intensities.size - intensities.argsort(axis=0).argsort(axis=0) - 1
        #intensitysum = intensities.sum()
        scanlist = scan['scanList']['scan'][0]
        rt = scanlist['scan start time'].real
        it = scanlist['ion injection time'].real
        #timedict[rt] = scancount
        timearray.append(rt)
        #injectionarray.append(it)
        retentiontimes = np.repeat(rt, mza.size)[:,None]
        
        #removing redundant matches -> the line ended for these ones, or its skipping an index
        fmza = mza.copy()
        fint = intensities.copy()
        fintrank = intensityranks.copy()
        fmassdist = massdist.copy()
        if redundants:
            fmza = np.delete(fmza, removals)
            retentiontimes = np.delete(retentiontimes, removals)
            fint = np.delete(fint, removals)
            #fintrank = np.delete(fintrank, removals)
            #nfint = fint / it
            found = np.delete(found, removals)
            fmassdist = np.delete(fmassdist, removals)
            #coords = np.stack((fmza, retentiontimes, fint, nfint, fint/intensitysum, fintrank), axis=1).tolist()
            coords = np.stack((fmza, retentiontimes, fint), axis=1).tolist()
            #
            #things that had a redundant match, and weren't taken, from mza are put up as new lines
            excluded = mza[removals]
            flatmasslist = excluded.flatten().tolist()
            elen = len(excluded)
            retentiontimes = np.repeat(rt, elen)[:,None]
            eint = intensities[removals]
            #eintrank = intensityranks[removals]
            #neint = eint / it
            #ecoords = np.hstack((excluded, retentiontimes, eint, neint, eint/intensitysum, eintrank))[:,:,None].tolist()
            ecoords = np.hstack((excluded, retentiontimes, eint))[:,:,None].tolist()
            uids = (np.arange(elen) + uidcount).tolist()
            uidcount += elen
            trackedgroups.update(zip(uids, ecoords))
            trackedkeys.update(zip(flatmasslist, uids))
            trackedma.update(zip(flatmasslist, uids))
            groupmovingaverages.update(zip(uids, flatmasslist))
            groupdifftoma.update(zip(uids, np.zeros(elen).tolist()))
            groupranges.update(zip(uids, np.hstack((excluded, excluded)).tolist()))
            trackedscancount.update(zip(uids, np.ones(elen).tolist()))
            trackedlength.update(zip(uids, np.ones(elen).tolist()))
            modeltracker[0] += elen #newly added
        else:
            #nfint = fint / it
            #coords = np.hstack((fmza, retentiontimes, fint, nfint, fint/intensitysum, fintrank)).tolist()
            coords = np.hstack((fmza, retentiontimes, fint)).tolist()
        found = found.flatten().tolist()
        
        p4.append(time() - pt4)
        pt5 = time()
        
        sorteddistances = np.sort(massdist) #this moved the cutoff up a little bit, higher mass-range lines seemed to be slightly better connected, there's more mass variation up there it seems...
        mbool = sorteddistances <= sorteddistances[:,None] #isn't there an easier way to do this?, isn't this just arange?
        countsums = mbool.sum(axis=0) / sorteddistances.size
        #mbool = stats.rankdata(sorteddistances, method='max')[::-1] #^yep, but it's just slightly different because this is a ranking and not counting the things beyond it - so i'll skip this for now - the last of a dense rank needs a +1?
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
        for c, f, d in zip(coords, found, fmassdist): #could this loop be concurrent?
            modify = False
            nf = c[0]
            tid = trackedma[f]
            tgroup = trackedgroups[tid][0]
            tlen = len(tgroup)
            lastmass = tgroup[-1]
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
                trackedscancount[tid] += 1
                trackedlength[tid] += 1
                for n, ci in enumerate(c):
                    trackedgroups[tid][n].append(ci)
                #if linedeletioncounter[tid] > 0:
                #linedeletioncounter[tid] -= 1 #1661209
                #linedeletioncounter[tid] = 0 #1593013
                linedeletioncounter[tid] //= 2 #1606868
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
                trackedgroups[uidcount] = [[ci] for ci in c]
                trackedkeys[nf] = uidcount
                trackedma[nf] = uidcount
                groupmovingaverages[uidcount] = nf
                groupdifftoma[uidcount] = 0 #this zero won't bog down any averages, same principle new mechanics
                groupranges[uidcount] = [nf, nf]
                trackedscancount[uidcount] += 1
                trackedlength[uidcount] += 1
                uidcount += 1
                modeltracker[0] += 1 #newly added
                foundremoval = groupmovingaverages[tid]
                foundremovals.append(foundremoval)
        
        p6.append(time() - pt6)
        pt7 = time()
        
        for fr in foundremovals:
            found.remove(fr)
        #fmza = np.delete(fmza, fmzaremovals)
        nonmatched = np.setdiff1d(previousdata, found)[:,None].flatten()
        #
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
            trackedscancount[linekey] += 1
            if linedeletioncounter[linekey] > deadsignal:
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
                linecorrections.append(list(correctionradius))
            else:
                modeltracker[2] += 1 #nonmatched
        
        p7.append(time() - pt7)
        pt8 = time()

        wides.append(widestmassrange)
        nonmatched = np.delete(nonmatched, newmodelremovals)[:,None]
        #newtrain = np.append(mza, nonmatched, axis=0)
        currentmasskeys = list(map(trackedkeys.get, mza.flatten().tolist()))
        currentmasses = np.array(list(map(groupmovingaverages.get, currentmasskeys)))[:,None]
        newtrain = np.append(currentmasses, nonmatched, axis=0)
        model = spatial.KDTree(newtrain)
        modeltracking[scan['index']] = modeltracker
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
    minmass, mintime, mii = a.min(axis=1)
    #maxmass, maxtime, mai, nmai, maxperc, maxrank = a.max(axis=1)
    maxmass, maxtime, mai = a.max(axis=1)
    #wmean = np.average(a[0], weights=a[2]) #better to weight by intensity or injection time? both?
    wmean = (a[0] * a[2]).sum() / a[2].sum() #why is this faster, wtf numpy
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
            matchtimes = [trackedgroups[i][1].tolist() for i in tmkeys]
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
                        mtimes = [trackedgroups[i][1].tolist() for i in mkeys]
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
                            oldtimes = trackedgroups[otherline][1].tolist()
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

#works out pretty well, barely increased the max in wides
#limits = []
#for p in linecorrections:
#    limit = [np.inf, 0]
#    for k in p:
#        masses = trackedgroups[k][0]
#        if min(masses) < limit[0]:
#            limit[0] = min(masses)
#        if max(masses) > limit[1]:
#            limit[1] = max(masses)
#    limits.append(limit)
#limits = np.array(limits)
#np.diff(limits).max()


t5 = time()
for lines in linecorrections:
    linegrid = []
    for line in lines:
        if line in precursorlinematches:
            for scan in precursorlinematches[line]:
                precursorscanmatches[scan] = uidcount
            precursorlinematches[uidcount].extend(precursorlinematches.pop(line))
            precursorapparentcharge[uidcount].extend(precursorapparentcharge.pop(line))
        for c in list(zip(*trackedgroups[line].tolist())):
            linegrid.append(list(c))
        del trackedgroups[line]
    linegrid = sorted(linegrid, key=lambda x: x[1])
    reformattedgrid = [list(i) for i in zip(*linegrid)]
    trackedgroups[uidcount] = np.array(reformattedgrid)
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

#reversegroups = {}
for n, k in enumerate(kl):
    trackedgroups[n] = groupholder.pop(k)
    if k in precursorlineholder:
        for scan in precursorlineholder[k]:
            precursorscanmatches[scan] = n
        precursorlinematches[n] = precursorlineholder.pop(k)
        precursorapparentcharge[n] = precursorchargeholder.pop(k)
    #groupholder[n] = k
    #reversegroups[k] = n

print(time() - t5, 'line corrections')

t6 = time()
startingpoints = defaultdict(list)
regions = [] #t, b, l, r
for k, a in trackedgroups.items():
    #minmass, mintime, mii, nmii, minperc, minrank = a.min(axis=1)
    minmass, mintime, mii = a.min(axis=1)
    #maxmass, maxtime, mai, nmai, maxperc, maxrank = a.max(axis=1)
    maxmass, maxtime, mai = a.max(axis=1)
    #wmean = np.average(a[0], weights=a[2]) #better to weight by intensity or injection time? both?
    wmean = (a[0] * a[2]).sum() / a[2].sum() #why is this faster, wtf numpy
    #peakarea = np.trapezoid(a[3], a[1])
    peakarea = np.trapezoid(a[2], a[1])
    maxintensity = a[2].max()
    regions.append([minmass, maxmass, mintime, maxtime, len(a[0]), peakarea, maxintensity, wmean, k])
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

zoomplotting = False
#zoomplotting = True


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

di = 0 #connectionindexes
paircharges = {} #connection: charge
scoresbypair = {} #pair: [[absacdiff, datapercdiff, rtoffset],]
pairsbyline = defaultdict(list) #mass: [pairs]

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
connectionspine = defaultdict(list) #connectionindex: [pairkeys]
latestconnections = defaultdict(lambda: defaultdict(list)) #masskey: charge: [current connection indices that end in masskey], a mass is always active as long as it's in masspool, this keeps track of all the groups that end in any given mass
latestmass = {} #connectionindex: latest masskey

#I'm learning that numpy arrays are actually efficient as fuck for memory storage, I should be able to implement a branchends/spineends here using dicts of lists that i turn into numpy arrays once a branch's final mass gets taken out of mass pool, i also need to remove all of that branches entries in previousdecreases
#also seeing as numpy arrays are so dope, I should change the structure of trackedgroups for lines on removal to save on memory

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
                link = False
                if encompassed or percentoverlap > 0.75: #check for distancelinks that can be used to expand the subiso range
                    link = True
                if diff < subisomax + roundcutoff:
                    if link:
                        #potential subisos
                        maxisocharge = np.floor(subisomax / diff)
                        if maxisocharge == 0: #it can be rounded to zero b/c of widthbuffer? not sure, but it happens
                            maxisocharge += 1
                        if om in subisomasses:
                            ti = subisomasses[okey]
                            subisogroups[ti][maxisocharge].append(pi)
                            subisomasses[nkey] = ti
                        else:
                            #make a new subiso group
                            subisogroups[si][maxisocharge].append(pi)
                            subisomasses[okey] = si
                            subisomasses[nkey] = si
                            si += 1
                        lpair = (okey, nkey)
                        pairkeys[pi] = lpair
                        pi += 1
                else:
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
                                    csubs = latestconnections[okey][charge].copy() #need copy here?
                                    for adi in csubs:
                                        ratiocheck = steplimit
                                        if adi in previousdecrease:
                                            if not decreasecheck:
                                                #intensity is increasing
                                                ratiocheck = newinclimit
                                        if intensitypercdiff <= ratiocheck:
                                            #connectionpairkeys[di] = pi
                                            #connectionspine[di] = adi
                                            #latestconnections[nkey][charge].append(di)
                                            if latestmass[adi] == okey:
                                                xdi = adi
                                            else:
                                                xdi = di
                                                opairs = pairsbyline[okey]
                                                spinecopy = connectionspine[adi].copy()
                                                spinds = []
                                                for op in opairs:
                                                    if op in spinecopy:
                                                        spinds.append(spinecopy.index(op))
                                                spi = max(spinds)
                                                spinecopy = spinecopy[:spi]
                                                connectionspine[xdi] = spinecopy
                                                di += 1
                                            connectionspine[xdi].append(pi)
                                            #lm = latestmass[adi] #this is just okey
                                            #latestconnections[okey][charge].remove(adi) #i shouldn't remove this, but i should check if okey is latestmass in order to determine if a new spine should be cast out of a copy of adi up to okey
                                            latestmass[xdi] = nkey
                                            latestconnections[nkey][charge].append(xdi)
                                            if decreasecheck:
                                                #previousdecrease[di] = True
                                                previousdecrease[xdi] = True
                                            ncons += 1
                                            #di += 1
                                else:
                                    #no previous subgroup 
                                    if intensitypercdiff <= steplimit:
                                        #connectionpairkeys[di] = pi
                                        connectionspine[di].append(pi)
                                        latestmass[di] = nkey
                                        latestconnections[nkey][charge].append(di)
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
                                        pairsbyline[p].append(pi)
                                    scoresbypair[pi] = scorelist
                                    if subcheck:
                                        #connectionpairkeys[di] = pi
                                        connectionspine[di].append(pi)
                                        latestmass[di] = nkey
                                        latestconnections[nkey][charge].append(di)
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
        if mpr in pairsbyline:
            pairsbyline[mpr] = np.array(pairsbyline[mpr]) #memory effient storage
        if mpr in latestconnections:
            for charge, cons in latestconnections[mpr].items():
                for con in cons:
                    if mpr == latestmass[con]:
                        del latestmass[con]
                        connectionspine[con] = np.array(connectionspine[con]) #memory efficient storage
                        try:
                            del previousdecrease[con]
                        except KeyError: #no previous decrease, idc really
                            pass
            del latestconnections[mpr]
    masspool.append(nkey)

for con, masskey in latestmass.items():
    connectionspine[con] = np.array(connectionspine[con])
    pairsbyline[masskey] = np.array(pairsbyline[masskey])

#there should probably be a better solution but some things were getting through as lists
for masskey in masspool:
    if masskey in pairsbyline:
        pairsbyline[masskey] = np.array(pairsbyline[masskey])

del latestconnections
del previousdecrease
del latestmass

print(time() - t7, 'mass processing')
t8 = time()

#class MyManager(BaseManager):
#    pass
#
#rsorted = regions[regions[:,7].argsort()]
#regiter = rsorted[rsorted[:,4] >= minpoints]
#
#def distribution_connections(regiter, regions, steplimit, newinclimit, proton, uppermasslimit, roundcutoff, subisomax):
#    
#    def round_process(oreg):
#        nonlocal steplimit
#        nonlocal newinclimit
#        nonlocal proton
#        nonlocal uppermasslimit
#        nonlocal roundcutoff
#        nonlocal masswidthlimit
#        
#        nonlocal subisomax
#        nonlocal subisomasses
#        nonlocal subisogroups
#        nonlocal latestconnections
#        nonlocal latestmass
#        nonlocal connectionspine
#        nonlocal previousdecrease
#        nonlocal pairsbyline
#        nonlocal scoresbypair
#        nonlocal pairkeys
#        nonlocal paircharges
#        
#        nonlocal masspoolremovals
#        
#        #oreg = regions[okey]
#        om = oreg[7]
#        omassmin = oreg[0]
#        omassmax = oreg[1]
#        omwidth = omassmax - omassmin
#        if omwidth > masswidthlimit:
#            omassmin = om - masswidthlimit
#            omassmax = om + masswidthlimit
#        if nmassmin - omassmax <= proton:
#            omlt = oreg[2]
#            omrt = oreg[3]
#            olrange = omrt - omlt
#            overpass = False
#            if nmlt < omrt and nmrt > omlt: #rt's overlap
#                #if you want to allow non-overlaps to match, and there are good reasons to, you need percentoverlap to become negative so that its later addition is valued worse
#                overlap = min(omrt, nmrt) - max(omlt, nmlt)
#                encompassed = False
#                if nmlt > omlt and nmrt < omrt: #new encompassed
#                    encompassed = True
#                    combinedrange = nlrange + olrange
#                    percentoverlap = (overlap * 2) / combinedrange
#                    #using newinclimit here is appropriate because it's directly related to expected differences in adjacent isotopomer quantities, which is what the overlap is also ~somewhat assessing
#                    if percentoverlap > newinclimit:
#                        overpass = True
#                elif omlt > nmlt and omrt < nmrt: #old encompassed
#                    encompassed = True
#                    combinedrange = nlrange + olrange
#                    percentoverlap = (overlap * 2) / combinedrange
#                    if percentoverlap > newinclimit:
#                        overpass = True
#                else:
#                    fullrange = max(omrt, nmrt) - min(omlt, nmlt)
#                    percentoverlap = overlap / fullrange
#                    if percentoverlap > 0.5: #this is super lenient I think
#                        overpass = True
#            #else:
#                #negative percentoverlap values for non-overlaps within some 1.75 range or something goes here
#            if overpass:
#                diff = nm - om
#                link = False
#                if encompassed or percentoverlap > 0.75: #check for distancelinks that can be used to expand the subiso range
#                    link = True
#                if diff < subisomax + roundcutoff:
#                    if link:
#                        #potential subisos
#                        maxisocharge = np.floor(subisomax / diff)
#                        if maxisocharge == 0: #it can be rounded to zero b/c of widthbuffer? not sure, but it happens
#                            maxisocharge += 1
#                        if om in subisomasses:
#                            ti = subisomasses[okey]
#                            subisogroups[ti][maxisocharge].append(pi)
#                            subisomasses[nkey] = ti
#                        else:
#                            #make a new subiso group
#                            subisogroups[si][maxisocharge].append(pi)
#                            subisomasses[okey] = si
#                            subisomasses[nkey] = si
#                            si += 1
#                        lpair = (okey, nkey)
#                        pairkeys[pi] = lpair
#                        pi += 1
#                else:
#                    #sometimes a charge registers as one thing but the value is way closer to +/-1
#                    initialcharge = round(proton / diff)
#                    if initialcharge > 1: #nothing close enough to care about for 1, plus the first bit would go to 0 and cause annoying zerodivision warnings
#                        chargespread = np.linspace(initialcharge - 1, initialcharge + 1, 3)
#                        expspread = proton / chargespread
#                        minexpind = np.abs(diff - expspread).argmin()
#                        charge = int(chargespread[minexpind])
#                    else:
#                        charge = initialcharge
#                    actualmass = om * charge - proton * charge
#                    #min/max peptide mass from the isotope database, anything with a basemmass larger than that max shouldn't be considered, it would help keep out ridiculously large charges - instead having them be a part of nodists, the way they probably should
#                    if actualmass <= uppermasslimit:
#                        expdiff = proton / charge
#                        acdiff = expdiff - diff
#                        diffcut = expdiff * chargetolerance
#                        if acdiff > -1 * (diffcut * chargetolerance + masswidthlimit):
#                        #^a lighter acceptance for distances above the charge dist - it does happen from time to time on lower-hanging straggler isotopomers
#                            if acdiff <= diffcut + masswidthlimit:
#                                absacdiff = abs(acdiff) * charge #normalizing -> distance to proton
#                                ointensity = oreg[5]
#                                ncons = 0
#                                intensitypercdiff = abs(nintensity - ointensity) / (nintensity + ointensity) / 2
#                                subcheck = False
#                                if okey in latestconnections:
#                                    if charge in latestconnections[okey]:
#                                        subcheck = True
#                                decreasecheck = nintensity < ointensity
#                                if subcheck:
#                                    csubs = latestconnections[okey][charge].copy() #need copy here?
#                                    for adi in csubs:
#                                        ratiocheck = steplimit
#                                        if adi in previousdecrease:
#                                            if not decreasecheck:
#                                                #intensity is increasing
#                                                ratiocheck = newinclimit
#                                        if intensitypercdiff <= ratiocheck:
#                                            #connectionpairkeys[di] = pi
#                                            #connectionspine[di] = adi
#                                            #latestconnections[nkey][charge].append(di)
#                                            if latestmass[adi] == okey:
#                                                xdi = adi
#                                            else:
#                                                xdi = di
#                                                opairs = pairsbyline[okey]
#                                                spinecopy = connectionspine[adi].copy()
#                                                spinds = []
#                                                for op in opairs:
#                                                    if op in spinecopy:
#                                                        spinds.append(spinecopy.index(op))
#                                                spi = max(spinds)
#                                                spinecopy = spinecopy[:spi]
#                                                connectionspine[xdi] = spinecopy
#                                                di += 1
#                                            connectionspine[xdi].append(pi)
#                                            #lm = latestmass[adi] #this is just okey
#                                            #latestconnections[okey][charge].remove(adi) #i shouldn't remove this, but i should check if okey is latestmass in order to determine if a new spine should be cast out of a copy of adi up to okey
#                                            latestmass[xdi] = nkey
#                                            latestconnections[nkey][charge].append(xdi)
#                                            if decreasecheck:
#                                                #previousdecrease[di] = True
#                                                previousdecrease[xdi] = True
#                                            ncons += 1
#                                            #di += 1
#                                else:
#                                    #no previous subgroup 
#                                    if intensitypercdiff <= steplimit:
#                                        #connectionpairkeys[di] = pi
#                                        connectionspine[di].append(pi)
#                                        latestmass[di] = nkey
#                                        latestconnections[nkey][charge].append(di)
#                                        if decreasecheck:
#                                            previousdecrease[di] = True
#                                        ncons += 1
#                                        di += 1
#                                if ncons > 0:
#                                    opoints = oreg[4]
#                                    #dpercdiff = abs(npoints - opoints) / (npoints + opoints) / 2
#                                    dpercdiff = abs(nintensity - ointensity) / (nintensity + ointensity) / 2
#                                    lpair = (okey, nkey)
#                                    pairkeys[pi] = lpair
#                                    paircharges[pi] = charge
#                                    scorelist = [absacdiff, dpercdiff, percentoverlap, ~decreasecheck]
#                                    for p in lpair:
#                                        pairsbyline[p].append(pi)
#                                    scoresbypair[pi] = scorelist
#                                    if subcheck:
#                                        #connectionpairkeys[di] = pi
#                                        connectionspine[di].append(pi)
#                                        latestmass[di] = nkey
#                                        latestconnections[nkey][charge].append(di)
#                                        if decreasecheck:
#                                            previousdecrease[di] = True
#                                        di += 1
#                                    pi += 1
#        else:
#            #om is past proton distance, remove om from mass pool
#            masspoolremovals.append(okey)
#    
#    
#    class EmbeddedManager(BaseManager):
#        pass
#    emanager = EmbeddedManager()
#    emanager.register('defaultdict', defaultdict, DictProxy)
#    #emanager.register('list', list, ListProxy)
#    emanager.start()
#    manager = mp.Manager()
#    
#    pairkeys = manager.dict() #pairkey: pair
#    previousdecrease = manager.dict() #connectionindex: True, exists if something is increasing
#    
#    di = 0 #connectionindexes
#    paircharges = manager.dict() #connection: charge
#    scoresbypair = manager.dict() #pair: [[absacdiff, datapercdiff, rtoffset],]
#    pairsbyline = defaultdict(list) #mass: [pairs]
#    
#    si = 0 #subisokeys
#    subisomasses = manager.dict() #lineuid: subisogroup
#    subisogroups = defaultdict(lambda: defaultdict(list)) #subiso group: max charge for pair: [pairkeys]
#
#    masswidthlimit = roundcutoff * 2
#
#    connectionspine = defaultdict(list) #connectionindex: [pairkeys]
#    latestconnections = defaultdict(lambda: defaultdict(list)) #masskey: charge: [current connection indices that end in masskey], a mass is always active as long as it's in masspool, this keeps track of all the groups that end in any given mass
#    latestmass = manager.dict() #connectionindex: latest masskey
#
#    pi = 0 #pairkeys
#    masspool = manager.list()
#    for reg in regiter.tolist():
#        npoints = reg[4]
#        nm = reg[7]
#        nmlt = reg[2]
#        nmrt = reg[3]
#        nmassmin = reg[0]
#        nmassmax = reg[1]
#        nmwidth = nmassmax - nmassmin
#        if nmwidth > masswidthlimit:
#            #this is strict, but seems to help in the end, meaning that essentially the means are rather important and useful
#            nmassmin = nm - masswidthlimit
#            nmassmax = nm + masswidthlimit
#        nkey = int(reg[8])
#        nintensity = reg[5]
#        masspoolremovals = []
#        nlrange = nmrt - nmlt
#        for okey in masspool: #could this loop be concurrent?
#        for mpr in masspoolremovals:
#            masspool.remove(mpr)
#            if mpr in pairsbyline:
#                pairsbyline[mpr] = np.array(pairsbyline[mpr]) #memory effient storage
#            if mpr in latestconnections:
#                for charge, cons in latestconnections[mpr].items():
#                    for con in cons:
#                        if mpr == latestmass[con]:
#                            del latestmass[con]
#                            connectionspine[con] = np.array(connectionspine[con]) #memory efficient storage
#                            try:
#                                del previousdecrease[con]
#                            except KeyError: #no previous decrease, idc really
#                                pass
#                del latestconnections[mpr]
#        masspool.append(nkey)
#
#    for con, masskey in latestmass.items():
#        connectionspine[con] = np.array(connectionspine[con])
#        pairsbyline[masskey] = np.array(pairsbyline[masskey])
#
#    #there should probably be a better solution but some things were getting through as lists
#    for masskey in masspool:
#        if masskey in pairsbyline:
#            pairsbyline[masskey] = np.array(pairsbyline[masskey])



#flatdistgroups = set()
#
#distributionscoresbyline = defaultdict(set) #linekey: [pairkeys] -> without a set there is redundancy in here that disrupts downstream
#distributionscoredict = {} #pairkey: scores
#for gid, pairkey in connectionpairkeys.items():
#    if gid in connectionspine: # > 1 pair
#        #if spinecount[gid] > 2: #at least 3 pairs so the means don't default to 0 -> cheating
#        pairs = []
#        scores = []
#        loopgid = gid
#        while True:
#            loopkey = connectionpairkeys[loopgid]
#            scores.append(scoresbypair[loopkey])
#            pairs.append(loopkey)
#            if not loopgid in connectionspine:
#                break
#            loopgid = connectionspine[loopgid]
#        flatdistgroups.add(tuple(sorted(set(itertools.chain(*(pairkeys[i] for i in pairs))))))
#        scorearray = np.array(scores)
#        distmean = scorearray[:,0].mean()
#        rtmultiplier = scorearray[:,2].prod()
#        decreasingsum = scorearray[:,3].sum()
#        slen = scorearray.size
#        if decreasingsum > 0:
#            decreasingmultiplier = decreasingsum
#        else:
#            decreasingmultiplier = 1
#        for pairk, score in zip(pairs, scores): 
#            dist, ddiff, rtoffset, decs = score
#            #meandiff = abs(distmean - dist) / (slen + 1) 
#            meandiff = abs(distmean - dist) / slen
#            #distdiff = meandiff - meandiff * rtoffset
#            #distdiff = meandiff * decreasingmultiplier
#            distdiff = meandiff * (2**decreasingmultiplier)
#            #datadiff = ddiff - ddiff * rtoffset
#            datadiff = ddiff - ddiff * rtmultiplier
#            #datadiff = ddiff * decreasingmultiplier * rtmultiplier
#            scorelist = tuple([distdiff, datadiff])
#            distributionscoredict[pairk] = scorelist
#            for m in pairkeys[pairk]:
#                distributionscoresbyline[m].add(pairk)

flatdistgroups = set()

distributionscoresbyline = defaultdict(list) #linekey: [pairkeys] -> without a set there is redundancy in here that disrupts downstream
#distributionscoredict = {} #pairkey: scores
distributionscoredict = defaultdict(list) #pairkey: [[scores],]
for distkey, pairspine in connectionspine.items(): #this loop needs to write its massive outputs to disk and import them later on from a different function
    activepairlist = []
    scores = []
    for pairs in pairspine.tolist(): 
        activepairlist.append(pairs)
        scores.append(scoresbypair[pairs])
        flatdist = []
        competinglines = set()
        for pair in activepairlist:
            pk = pairkeys[pair]
            flatdist.extend(pk)
            for p in pk:
                #there was no problems here
                #if p not in pairsbyline:
                #    print('problem child', distkey, pairs, p)
                #if set(pairsbyline[p].tolist()).difference(pairspine): #checking that the line is in more distributions than whats generated via this spine
                if set(pairsbyline[p].tolist()).difference(activepairlist): #checking that the line is in more distributions than whats generated via this spine
                    competinglines.add(p)
        activelines = list(set(itertools.chain(*(pairkeys[i] for i in activepairlist))))
        #not amazing performance from this, it helped a bit but didn't offer much else
        #I'm going to keep it for now, it decreases agreed charge matches on ms2 hits but increases number of dists and multi-charge states which is interesting
        #^this goes along with taking the set difference of activepairlist above as well instead of pairspine
        rbounds = regions[activelines,2:4]
        rstack = boundary_stack(rbounds)
        if rstack > 0:
            flatdistgroups.add(tuple(sorted(set(flatdist))))
            if competinglines and len(activepairlist) > 1:
                unstackedboundaries = rbounds - rbounds.min(axis=1)[:,None]
                ustack = boundary_stack(unstackedboundaries)
                #maybe just for loop and extend a list to make ^this faster?
                scorearray = np.array(scores)
                distmean = scorearray[:,0].mean()
                #rtmultiplier = scorearray[:,2].prod()
                rtmultiplier = rstack / ustack
                decreasingmultiplier = scorearray[:,3].sum() + 1
                #slen = scorearray.size #I did this by mistake, but it should just be a linear x4 scaling
                slen = len(scorearray)
                #if decreasingsum > 0:
                #    decreasingmultiplier = decreasingsum
                #else:
                #    decreasingmultiplier = 1
                for pair, score in zip(activepairlist, scores): 
                    if paircharges[pair] > 1: # a lot of bad 1+ matches get high priority from this, I essentially want less 1+ than 3+ and this helps
                        dist, ddiff, rtoffset, decs = score
                        #meandiff = abs(distmean - dist) / (slen + 1) 
                        meandiff = abs(distmean - dist) / slen
                        #distdiff = meandiff - meandiff * rtoffset
                        #distdiff = meandiff * decreasingmultiplier
                        distdiff = meandiff * (2**decreasingmultiplier)
                        #datadiff = ddiff - ddiff * rtoffset
                        #datadiff = ddiff - ddiff * rtmultiplier
                        datadiff = ddiff / rtmultiplier
                        #datadiff = ddiff * decreasingmultiplier * rtmultiplier
                        scorelist = tuple([distdiff, datadiff])
                        #distributionscoredict[pairk] = scorelist
                        distributionscoredict[pair].append(scorelist)
                        for p in pairkeys[pair]:
                            if p in competinglines:
                                if not pair in distributionscoresbyline[p]: #avoiding set use to save memory
                                    distributionscoresbyline[p].append(pair)

print(time() - t8, 'distribution scoring')

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

t9 = time()

isodists = []
for pair, charge in paircharges.items(): #iterating paircharges and not pairkeys because pairkeys has subiso pairs that don't have charges
    dist = scoresbypair[pair][0] * charge
    isodists.append(dist)
isomean = np.mean(isodists)
#could cut some off via this, i don't see any problems with it, barely any less distributions matched, 53k -> 52k
#^it might help with memory... a little, but it's not a substantive soltuion to memory issues here, and for something to be meaningful i suppose it should reduce the number of singl-charged distributions while increasing 2/3 charged

#I think i can reduce the memory footprint of the ranks by making the blocked pairs an active process from the get-go, i should verify that forcedcompetition doesn't ever really come into play first though

preservedpairs = set()

#lines being comparison-ranked across all individual distribution scores it participates in, if it has multiple
rankedpairs = [] #[pair, minval]
for line, pairs in distributionscoresbyline.items():
    pairs = list(pairs)
    #vals = []
    #for pair in pairs:
    #    pairscores = distributionscoredict[pair]
    #    if len(pairscores) > 1:
    #        pairscores = np.array(pairscores)
    #        pairsums = pairscores.sum(axis=0)
    #        pairsums[pairsums == 0] = 1
    #        pairpercs = pairscores / pairsums
    #        percsums = pairpercs.sum(axis=1)
    #        pmin = percsums.argmin()
    #        vals.append(pairscores[pmin].tolist())
    #    else:
    #        vals.append(pairscores[0])
    vals = []
    pairexpansion = []
    for pair in pairs:
        pairscores = distributionscoredict[pair]
        vals.extend(pairscores)
        pairexpansion.extend([pair for _ in range(len(pairscores))])
    vals = np.array(vals)
    vsums = vals.sum(axis=0)
    vsums[vsums == 0] = 1 #0-sums make nans but this doesn't change the final answer of 0
    vpercs = vals / vsums
    #vpercs[np.isnan(vpercs)] = 0 #otherwise you get nans from 0 meandist matches
    sumvals = vpercs.sum(axis=1)
    #rankedpairs.extend(list(zip(pairs, sumvals.tolist())))
    rankedpairs.extend(list(zip(pairexpansion, sumvals.tolist())))
    preservedpairs.update(pairs)

secondpriorities = [] #[pair, score]
thirdpriorities = [] #[pair, score]
for line, pairs in pairsbyline.items():
    pairs = pairs.tolist()
    plen = len(pairs)
    if plen > 1:
        scores = [scoresbypair[i] for i in pairs]
        scorearray = np.array(scores)
        scorearray[:,3] += 1
        #offsetscores = scorearray[:,:2] - scorearray[:,:2] * scorearray[:,2,None]
        #normoffsets = offsetscores / offsetscores.sum(axis=0)
        #for pair, score, rtoffset in zip(pairs, normoffsets.tolist(), scorearray[:,2].tolist()):
        #    secondpriorities.append([pair, sum(score)])
        #    rtoffsets[pair] = rtoffset
        #try this? idk - actually it worked really nicely
        scoresums = scorearray.sum(axis=0)
        scoresums[scoresums == 0] = 1 #0-sums make nans but this doesn't change the final answer of 0
        normscores = scorearray / scoresums
        #offsetnorms = normscores[:,:2] - normscores[:,:2] * scorearray[:,2,None]
        #offsetnorms = normscores[:,:2] - normscores[:,:2] * scorearray[:,2,None] + normscores[:,:2] * (2**scorearray[:,3,None])
        #offsetnorms = normscores[:,:2] - normscores[:,:2] / scorearray[:,2,None] + normscores[:,:2] * (2**scorearray[:,3,None])
        offsetnorms = normscores[:,:2] * scorearray[:,3,None] / scorearray[:,2,None]
        #offsetnorms = normscores[:,:2] * normscores[:,3,None] / normscores[:,2,None]
        for pair, score in zip(pairs, offsetnorms.sum(axis=1).tolist()):
            if pair not in preservedpairs: #memory saving
                secondpriorities.append([pair, score])
    else:
        pairs = pairs[0]
        if pairs not in preservedpairs:
            #scores = scoresbypair[pairs]
            dist, ddiff, rtoffset, dec = scoresbypair[pairs]
            dec += 1 #adds 1 to things that dec'd and makes non-dec's 1 -> no change
            #scoresum = dist + ddiff
            #normalizing these won't make any difference whether they're close or not
            #s1norm = dist / scoresum
            #s1norm = dist
            #s2norm = ddiff / scoresum
            #s2norm = ddiff
            #equalizednorm = abs(s2norm - s1norm)
            equalizednorm = abs(ddiff - dist)
            #rt offset after normalization was normal for 2ndprios previously
            #outscore = equalizednorm - equalizednorm * scores[2]
            #outscore = equalizednorm * rtoffset
            outscore = equalizednorm * dec / rtoffset
            thirdpriorities.append([pairs, outscore])

#it would be interesting to mix and mingle these scores, or perhaps only a subset of them, prior to ranking and see how this affects the results
#i was also thinking it would be interesting to separate the characteristic metrics, ie rt overlap, mass dist, etc, into individual lists and co-rank them
#additionally, the extra mertics you get from the above rankedpair process could be used as BONUS rows, whereas there might be 4 rows but you only need 3 hits to pass through the co-ranking filter
firstranks = sorted(rankedpairs, key=lambda x: x[1])
secondranks = sorted(secondpriorities, key=lambda x: x[1])
thirdranks = sorted(thirdpriorities, key=lambda x: x[1])

#firstranks has a weird ass distribution!!! where do the negatives come from again?
#rankchoice = firstranks
#binsize = 0.1
#binvals = np.linspace(min(rankchoice, key=lambda x: x[1])[1], max(rankchoice, key=lambda x: x[1])[1], int((max(rankchoice, key=lambda x: x[1])[1] - min(rankchoice, key=lambda x: x[1])[1])/binsize))
#bnn = spatial.KDTree(binvals[:,None])
#dists, inds = bnn.query(np.array(firstranks)[:,1,None])
#bincounts = Counter(inds)
#plt.bar(bincounts.keys(), bincounts.values())
#plt.yscale('log')
#plt.show()

#secondrankindices = {i[0]: i[1] for i in secondranks}

#the other option is to remove the zeros as they're unreliable and probably not as important due to the fact the rt overlap being 1 probably means they're small lines
#forcedcompetition = []
#for p, s in firstranks:
#    if s == 0:
#        forcedcompetition.append([p, secondrankindices[p]])
#    else:
#        break
#
#sortedranks = sorted(forcedcompetition, key=lambda x: x[1])

if firstranks[0][1] == 0:
    print('need forced competition')

sortedranks = []
sortedranks.extend(firstranks)
sortedranks.extend(secondranks)
sortedranks.extend(thirdranks)

preservedpairs = set() #don't need the old one, this technically reduces memory while allowing the same forward functionality again i suppose
preservedranks = []
for pairkey, score in sortedranks:
    if pairkey not in preservedpairs:
        preservedranks.append([pairkey, paircharges[pairkey]])
        preservedpairs.add(pair)

print(time() - t9, 'pair ranking setup')


#revision for the final ranking below: you can use itertools.chain to make sortedranks, then determine if pairkey is in preservedpairs OTF. This should save memory I suppose, removes the need to make one large final list
#^you might also be able to rely on a generator for sorting? which you can also use in chain.from_iterable and this would lower the amount of processing dedicated to an individual instance of sorting maybe?



#no jumping ahead on this one
#simple ordered ascension, make groups on the fly and merge them when appropriate

t10 = time()

#you could intersection_merge pairs in preservedpairs to multi-process this below, by breaking things into respective ranking hierarchies that actually interact/compete with each other

sn = 0
#these are kind of a nice idea but hard to do anything with, and would take up a lot of memory in the actual implementation
#linechargeranks = defaultdict(lambda: defaultdict(int)) #lineuid: charge: number of appearances
#linechargeindex = defaultdict(dict) #lineuid: length of linechargeranks at the time of appearance of each charge
#linekeeperindex = {} #lineuid: number of appearances before being selected
distsets = defaultdict(set) #distloc: [distributions]
linelocations = {} #masskey: distloc
setcharges = defaultdict(set) #index of distsets: [charges]
for pairkey, paircharge in preservedranks:
    pair = pairkeys[pairkey]
    pairset = set(pair)
    locs = set()
    for line in pairset:
        #linechargeranks[line][paircharge] += 1
        #if not paircharge in linechargeindex[line]:
        #    linechargeindex[line][paircharge] = sum(linechargeranks[line].values())
        if line in linelocations:
            locs.add(linelocations[line])
    if locs:
        joiner = min(locs)
        if len(locs) > 1:
            dist = pairset.copy()
            for l in locs:
                dist.update(distsets[l])
            if tuple(sorted(dist)) in flatdistgroups:
                for l in locs.difference([joiner]):
                    for line in distsets[l]:
                        linelocations[line] = joiner
                    setcharges[joiner].update(setcharges.pop(l))
                    distsets[joiner].update(distsets.pop(l))
                distsets[joiner].update(pairset)
                for line in pairset:
                    linelocations[line] = joiner
                    #linekeeperindex[line] = sum(linechargeranks[line].values())
        else:
            if tuple(sorted(distsets[joiner].union(pairset))) in flatdistgroups:
                distsets[joiner].update(pairset)
                setcharges[joiner].add(paircharge)
                for line in pairset:
                    linelocations[line] = joiner
                    #linekeeperindex[line] = sum(linechargeranks[line].values())
    else:
        joiner = sn
        distsets[joiner].update(pairset)
        setcharges[joiner].add(paircharge)
        for line in pairset:
            linelocations[line] = joiner
            #linekeeperindex[line] = sum(linechargeranks[line].values())
        sn += 1

print(time() - t10, 'pair ranking')
t11 = time()

dr =  0 #easier to work with dr here rather than re-index the finalized distribution just to get the max index for finaldefiniteind afterwards, the order of these aren't important
solodists = defaultdict(dict) #charge: distid: lines
for distindex, dist in distsets.items():
    charges = setcharges[distindex]
    if dist:
        charge = max(charges)
        solodists[charge][dr] = list(dist)
        dr += 1
finaldefiniteind = dr

#process subisotopomer mainmass focussing here, don't allow subisos to show up as masses, but do allow their intensities to contribute to a new area value to be used for intensity ranking
distributionmasses = {} #distid: ordered masses
distributioncharges = {} #distid: charge
distributionsoflines = {} #line: distid
linesofdistributions = {} #distid: mass-ordered linedkeys
distributiontimelimits = {} #distid [starting rt, ending rt]
distributionintensities = {} #distid: mass-ordered intensities
#distributionsbycharge = defaultdict(dict) #charge: dists: mass-ordered linekeys
distributionsbycharge = defaultdict(list) #charge: dists: mass-ordered linekeys
for charge, dists in solodists.items():
    for dist, lines in dists.items():
        dmasses = regions[lines,7] 
        lineorder = dmasses.argsort().tolist()
        sortedlines = [lines[i] for i in lineorder]
        sortedmasses = regions[sortedlines,7].tolist()
        dintensities = regions[sortedlines,5].tolist()
        rtlimits = regions[sortedlines,2:4] 
        minrt = rtlimits.min()
        maxrt = rtlimits.max()
        distributionmasses[dist] = sortedmasses
        distributioncharges[dist] = charge
        for line in lines:
            distributionsoflines[line] = dist
        linesofdistributions[dist] = sortedlines
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
for charge, sgd in solodists.items():
    foundvals.extend(list(itertools.chain(*sgd.values())))
specvals = regiter[:,8].astype(int)
nodists = np.setdiff1d(specvals, foundvals)

nodistmasses = regions[nodists,7]
#nodistintensities = regions[nodists,5]
#nodisttimes = regions[nodists,2:4]
#nodistkeys = np.arange(nodists.size) + di

nodistkeys = []
for line in nodists.tolist():
    dreg = regions[line]
    dmass = dreg[7] 
    dintensity = dreg[5]
    rtlimit = dreg[2:4] 
    minrt = rtlimit.min()
    maxrt = rtlimit.max()
    distributionmasses[dr] = [dmass]
    distributioncharges[dr] = 0
    distributionsoflines[line] = dr
    linesofdistributions[dr] = [line]
    distributiontimelimits[dr] = [minrt, maxrt]
    distributionintensities[dr] = [dintensity]
    #distributionsbycharge[charge][dist] = sortedlines
    distributionsbycharge[0].append(dr)
    nodistkeys.append(dr)
    dr += 1 #continuing from solodists count

massranges = regions[:,:2]
minmass = massranges.min() - 1
maxmass = massranges.max() + 1



print(time() - t11, 'distribution assembling')
print(time() - t1, 'total to distribution assembling')

#this is multiprocessable - somewhat, there was issues - but set lookup times makes it irrelevant
##simple ordered ascension, make groups on the fly and merge them when appropriate
#def pair_ranking(preservedranks, flatdistgroups):
#    distsets = [] #sets of distributions
#    linelocations = {} #mass: index of distsets
#    setcharges = defaultdict(set) #index of distsets: [charges]
#    for pair, rank, paircharge in preservedranks:
#        pairset = set(pair)
#        locs = set()
#        for line in pairset:
#            if line in linelocations:
#                locs.add(linelocations[line])
#        if locs:
#            if len(locs) == 1:
#                distindex = min(locs)
#                dist = distsets[distindex]
#                if sorted(dist.union(pairset)) in flatdistgroups:
#                    distsets[distindex].update(pairset)
#                    setcharges[distindex].add(paircharge)
#                    for line in pairset:
#                        if line not in linelocations:
#                            linelocations[line] = distindex
#            else:
#                dist = pairset.copy()
#                for l in locs:
#                    dist.update(distsets[l])
#                if sorted(dist) in flatdistgroups:
#                    distindex = min(locs)
#                    distsets[distindex].update(dist)
#                    for l in locs.difference([distindex]):
#                        setcharges[distindex].update(setcharges[l])
#                        setcharges[l] = False
#                        distsets[l] = False
#                    for line in dist:
#                        linelocations[line] = distindex
#        else:
#            distsets.append(pairset)
#            distindex = len(distsets) - 1
#            setcharges[distindex].add(paircharge)
#            for line in pairset:
#                linelocations[line] = distindex
#    return distsets, setcharges, linelocations
#
##t4 = time()
##outs = []
##for rankedpairs in sortedrankgroups:
##    outs.append(pair_ranking(rankedpairs, flatdistgroups))
##print(time() - t4)
#
#t4 = time()
#print(len(preservedranks), 'preservedranks')
#print(len(sortedrankgroups), 'sortedrankgroups')
#
#dr = 0
#fulllinelocations = {}
#solodists = defaultdict(dict) #charge: distid: lines
#with concurrent.futures.ProcessPoolExecutor(2) as executor:
#    futures = []
#    for rankedpairs in sortedrankgroups:
#        futures.append(executor.submit(pair_ranking, rankedpairs, flatdistgroups))
#    for f in concurrent.futures.as_completed(futures):
#        distsets, setcharges, linelocations = f.result()
#        for dist, charges in zip(distsets, setcharges.values()):
#            if dist:
#                charge = max(charges)
#                solodists[charge][dr] = list(dist)
#                dr += 1
#        fulllinelocations.update(linelocations)
#
#print(time() - t4, 'split ranking & assembling')
#
#foundvals = []
#for charge, sgd in solodists.items():
#    foundvals.extend(list(itertools.chain(*sgd.values())))
#specvals = regiter[:,8].astype(int)
#nodists = np.setdiff1d(specvals, foundvals)


#to line corrections, visualize this surrounding area
if zoomplotting:
    zst = st
    zet = et
    zumb = umb
    zlmb = lmb
    #zst = 49.5
    #zet = 49.6
    #zlmb = 364.5
    #zumb = 367.7
    newdists = defaultdict(dict)
    for fc, fgs in solodists.items():
        for fk, pkeys in fgs.items():
            fg = regions[pkeys,:2]
            times = regions[pkeys,2:4]
            if fg.min() <= zumb and fg.max() >= zlmb:
                if times.max() >= zst and times.min() <= zet:
                    newdists[fc][fk] = pkeys
    text = False
    ngroups = sum(len(i) for i in newdists.values())
    cols = dp.get_colors(ngroups)
    cn = 0
    fig, ax = plt.subplots(nrows=3, figsize=(6,8), sharex=True)
    for fc, fgs in newdists.items():
        for fk, pkeys in fgs.items():
            col = cols[cn]
            fg = regions[pkeys,7]
            cn += 1
            for p in pkeys:
                a = np.array(trackedgroups[p])
                ax[2].plot(a[0], a[1], '-', color=col, linewidth=0.1, alpha=0.9)
                ax[2].plot(a[0], a[1], '.', color=col, markersize=0.3, alpha=0.2)
                if text:
                    ax[2].text(a[0][-1], a[1][-1], str(regions[p,7].round(2)), color='white', fontsize=4)
            fints = regions[pkeys,5]
            ax[0].bar(fg, fints, color=col, alpha=0.5, width=0.01, label=str(fc))
            if text:
                for fx, fy, pk in zip(fg.tolist(), fints.tolist(), pkeys):
                    ax[0].text(fx, fy + fy * 0.03, str(pk), color='white', fontsize=4)
            #print(fg)
            #print(fc, '-', np.diff(sorted(fg)))
            #print('~')
            ax[1].hlines(cn, fg.min(), fg.max(), color=col, linewidth=0.6)
            for vert, pl in zip(fg, pkeys):
                ax[1].vlines(vert, cn - 0.1, cn + 0.1, color=col, linewidth=0.6)
            vi = np.sort(fg)
            if vi.size > 2:
                vstack = np.stack((vi[:-1], vi[1:]), axis=1)
                editspots = np.diff(vstack) < subisomax
                if editspots.any():
                    ewheres = np.where(editspots)[0].tolist()
                    for ew in ewheres:
                        subpair = vstack[ew].tolist()
                        subints = [spectrum[i] for i in subpair]
                        winint = subints.index(max(subints))
                        winner = subpair[winint]
                        if ew > 0:
                            #edit 1 before ew
                            vstack[ew-1,1] = winner
                        if ew < len(vstack) - 1:
                            #edit 1 after ew
                            vstack[ew+1,0] = winner
                    vstack = np.delete(vstack, ewheres, axis=0)
            else:
                vstack = vi.reshape(1, -1)
            vdiffs = np.diff(vstack)
            vflat = sorted(vstack.flatten().tolist())
            labelspots = np.mean(vstack, axis=1).tolist()
            for ls, lp in zip(labelspots, vstack.tolist()):
                labeldiff = np.diff(lp)[0].round(4)
                chargedist = (proton/fc - labeldiff).round(4)
                lstring = '~'.join((str(fc), str(labeldiff)))
                if text:
                    ax[1].text(ls, cn - 0.2, lstring, fontsize=4, ha='center', color='white')

    ndmasses = regions[nodists,7]
    mdinds = np.logical_and(ndmasses >= zlmb, ndmasses <= zumb)
    ndtimes = regions[nodists,2:4]
    tdinds = np.logical_and(ndtimes.min(axis=1) <= zet, ndtimes.max(axis=1) >= zst)
    zdinds = regions[nodists,4] > 1
    finds = np.logical_and.reduce((mdinds, tdinds, zdinds))
    ndplotters = nodists[finds]
    if ndplotters.size > 0:
        ndmasses = regions[ndplotters,7].tolist()
        nints = regions[ndplotters,5].tolist()
        ax[0].bar(ndmasses, nints, alpha=0.5, color='white', width=0.01, label='N/A')
        if text:
            for fx, fy, nd in zip(ndmasses, nints, ndplotters):
                if fy > 0:
                    ax[0].text(fx, fy + fy * 0.03, str(nd), color='white', fontsize=4)
        for p in ndplotters:
            a = np.array(trackedgroups[p])
            #ax[2].scatter(a[0], a[1], marker='.', color='white', s=0.3, alpha=0.3)
            ax[2].plot(a[0], a[1], '-', linewidth=0.2, color='white', markersize=0.3, alpha=0.6)
            ax[2].plot(a[0], a[1], '.', color='white', markersize=0.3, alpha=0.3)
            if text:
                ax[2].text(a[0][-1], a[1][-1], str(regions[p,7].round(2)), color='white', fontsize=4)
    
    precarray = np.array(list(precursorcoordinates.values()))
    preckeys = np.array(list(precursorcoordinates))
    precinds = preckeys[np.logical_and.reduce((precarray[:,0] >= zst, precarray[:,0] <= zet, precarray[:,1] >= zlmb, precarray[:,1] <= zumb))]
    for p in precinds:
        pa = precursorcoordinates[p]
        ax[2].plot(pa[1], pa[0], '*', color='tan')
    ax[0].set_yscale('log')
    ax[0].set_ylabel('intensity')
    ax[2].set_ylabel('minutes')
    ax[1].set_ylabel('distribution rank')
    ax[2].set_xlabel('m/z')
    for label in ax[2].get_xticklabels():
        #label.set_ha("right")
        label.set_rotation(-45)
    ncols = 6
    if text:
        ax[0].legend(title='Charge', loc='upper left', bbox_to_anchor=(0, 1.3 + (0.1 * cn / ncols)), ncol=ncols)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)
    plt.show()
    fig.clf()
    plt.close()


t12 = time()

#^actually I'm going to implement a directional system, starting with the single-charges, and adding in the nodists as well, move up to 2-charges, and match anything that can be matched via mass/rt, if something fails the ranking/overlap check or something, maybe you can still go to one more charge before deciding, if there are no good dist matches, go to nodists? or maybe always go to nodists and cancel it as competition if something of a higher charge is found
#this will be kind of like the line model but with changing mass values to search for
#might be able to have a flexible/generative cutoff with this as well

#i need to take this opportunity to compare expdiff error on distribution vs charge-state error across the total spread
#this will help tell me if the cutoff I'm implementing at the charge-state level below is even appropriate
#^I also need to explore the RT overlap metric I'm using here, see what kind of distributions it gives

if nodists.size > 0:
    #keeping the 0-charge nodists out
    chargeiterations = sorted(distributionsbycharge)[1:]
else:
    chargeiterations = sorted(distributionsbycharge)

nodistcharges = defaultdict(dict)
chargegroups = []
oldcharge = -1
for charge in chargeiterations: #ignoring 0-charge nodists here
    if charge - oldcharge > 1: #adjacent charge searches only
        moving = False
    dists = distributionsbycharge[charge]
    distkeys = []
    massdiffs = []
    distmasses = []
    for d in dists:
        dmasses = np.array(distributionmasses[d])
        basemasses = dmasses * charge - proton * charge
        distkeys.extend(d for _ in dmasses)
        distmasses.extend(basemasses)
        massdiff = np.diff(basemasses)
        massdiffs.extend(massdiff.tolist())
    massdiffs = np.array(massdiffs)
    diffcuts = np.abs(proton - massdiffs)
    ctol = diffcuts.mean() #max for for exploration I suppose
    #print(charge, '-', ctol.round(4))
    distmasses = np.array(distmasses)[:,None]
    distkeys = np.array(distkeys)
    chargemodel = spatial.KDTree(distmasses)
    nodistupmodel = spatial.KDTree(nodistmasses[:,None] * charge - proton * charge)
    if moving:
        #train it on the new and find old masses + radii
        roundmatches = set()
        oldmatches = set()
        matches = chargemodel.query_ball_point(oldmasses, oldradii, workers=8).tolist()
        for m, o in zip(matches, oldkeys):
            if m:
                matchkeys = []
                ominrt, omaxrt = distributiontimelimits[o]
                omasses = np.array(distributionmasses[o])
                obmasses = omasses * oldcharge - proton * oldcharge
                intensities = np.array(distributionintensities[o])
                intensityranks = intensities.argsort()[::-1] #i want to see if substituting the ranked intensities for the ranked intensities diffs (ordered by mass) would be a better mechanism, this currently fucks with the mass alignment [being the basis for the rank comparison] below and throws a valueerror
                for mk in distkeys[m].tolist():
                    nminrt, nmaxrt = distributiontimelimits[mk]
                    nmasses = np.array(distributionmasses[mk])
                    nbmasses = nmasses * charge - proton * charge
                    #below is requiring a majority of the matchable masses have sufficiently overlapping retention times
                    if ominrt < nmaxrt and omaxrt > nminrt: #overlap exists
                        basemasses = [obmasses, nbmasses]
                        dlines = [[regions[j,2:4].tolist() for j in linesofdistributions[i]] for i in [o, mk]]
                        sizes = [i.size for i in basemasses]
                        maxind = sizes.index(max(sizes))
                        lineup = basemasses[maxind]
                        retentionboundaries = defaultdict(list)
                        for n, (sm, dlims) in enumerate(zip(basemasses, dlines)):
                            sdiff = np.abs(lineup - sm[:,None])
                            alignmentloc = np.argwhere(sdiff == sdiff.min())[0]
                            alignmentloc -= alignmentloc.min()
                            luind = alignmentloc[1] - alignmentloc[0]
                            outinds = [luind, luind + sm.size]
                            for ind, dlim in zip(range(*outinds), dlims):
                                retentionboundaries[ind].append(dlim)
                        overpass = 0
                        matchables = 0
                        for ind, lims in retentionboundaries.items():
                            if len(lims) > 1:
                                matchables += 1
                                (lminrt, lmaxrt), (rminrt, rmaxrt) = lims
                                if lminrt > rminrt and lmaxrt < rmaxrt: #old encompassed
                                    overpass += 1
                                elif rminrt > lminrt and rmaxrt < lmaxrt: #new encompassed
                                    overpass += 1
                                else:
                                    overlap = min(rmaxrt, lmaxrt) - max(rminrt, lminrt)
                                    fullrange = max(rmaxrt, lmaxrt) - min(rminrt, lminrt)
                                    if overlap / fullrange > 0.5:
                                        overpass += 1
                        if overpass > matchables / 2:
                            diffmatrix = np.abs(obmasses - nbmasses[:,None])
                            matchmatrix = diffmatrix < ctol
                            mmshape = matchmatrix.shape
                            majoraxis = np.argmax(mmshape)
                            minoraxis = np.argmin(mmshape)
                            matrixmatches = matchmatrix.any(axis=majoraxis)
                            if matrixmatches.sum() > matrixmatches.size / 2: #matching at least half of the smaller with >=, make this a 'majority' with >
                                alignmentloc = np.argwhere(diffmatrix == diffmatrix.min())[0]
                                minindex = min(alignmentloc)
                                alignmentloc = [i-minindex for i in alignmentloc] #for when the closest match isn't the earliest matching index
                                rmax = nbmasses.size - alignmentloc[0]
                                lmax = obmasses.size - alignmentloc[1]
                                maxsize = min(rmax, lmax)
                                matchintensities = np.array(distributionintensities[mk])
                                matchintensityranks = matchintensities.argsort()[::-1]
                                scoutrankslice = intensityranks[alignmentloc[1]:alignmentloc[1]+maxsize]
                                matchrankslice = matchintensityranks[alignmentloc[0]:alignmentloc[0]+maxsize]
                                slicesubtraction = np.abs(scoutrankslice - matchrankslice)
                                if slicesubtraction.sum() <= slicesubtraction.size - 1: #close order matching passes
                                    matchkeys.append(mk)
                                    roundmatches.add(mk)
                                    oldmatches.add(o)
                                #else:
                                    #record non-matches i suppose
                if matchkeys:
                    matchkeys.append(o)
                    chargegroups.append(matchkeys)
            #else:
                #nodist match against old, nodists as higher charge
                #nonoldmatches.add(o)
        #nodist match against new, nodists as lower charge
        #there are other nonmatches that don't pass the above, add them here I think
        nodistdownmodel = spatial.KDTree(nodistmasses[:,None] * oldcharge - proton * oldcharge)
        nonmatched = set(distkeys).difference(roundmatches)
        nonmatchedmasses = np.array([distmasses[n] for n in range(len(distkeys)) if distkeys[n] in nonmatched])
        nonmatchedkeys = [distkeys[n] for n in range(len(distkeys)) if distkeys[n] in nonmatched]
        if nonmatchedmasses.size > 0: #this is a new addition, not fully tested
            downmatches = nodistdownmodel.query_ball_point(nonmatchedmasses, oldradii, workers=8).tolist()
            for m, o in zip(downmatches, nonmatchedkeys):
                if m:
                    matchkeys = []
                    ominrt, omaxrt = distributiontimelimits[o]
                    omasses = np.array(distributionmasses[o])
                    obmasses = omasses * charge - proton * charge
                    intensities = np.array(distributionintensities[o])
                    intensityranks = intensities.argsort()[::-1]
                    for mkey in m:
                        #nodists are from m
                        mk = nodistkeys[mkey]
                        matchintensities = np.array(distributionintensities[mk])
                        if matchintensities.max() < intensities.max():
                            nminrt, nmaxrt = distributiontimelimits[mk]
                            nmasses = np.array(distributionmasses[mk])
                            nbmasses = nmasses * oldcharge - proton * oldcharge
                            #below is requiring a majority of the matchable masses have sufficiently overlapping retention times
                            if ominrt < nmaxrt and omaxrt > nminrt: #overlap exists
                                basemasses = [obmasses, nbmasses]
                                dlines = [[regions[j,2:4].tolist() for j in linesofdistributions[i]] for i in [o, mk]]
                                sizes = [i.size for i in basemasses]
                                maxind = sizes.index(max(sizes))
                                lineup = basemasses[maxind]
                                retentionboundaries = defaultdict(list)
                                for n, (sm, dlims) in enumerate(zip(basemasses, dlines)):
                                    sdiff = np.abs(lineup - sm[:,None])
                                    alignmentloc = np.argwhere(sdiff == sdiff.min())[0]
                                    alignmentloc -= alignmentloc.min()
                                    luind = alignmentloc[1] - alignmentloc[0]
                                    outinds = [luind, luind + sm.size]
                                    for ind, dlim in zip(range(*outinds), dlims):
                                        retentionboundaries[ind].append(dlim)
                                overpass = 0
                                matchables = 0
                                for ind, lims in retentionboundaries.items():
                                    if len(lims) > 1:
                                        matchables += 1
                                        (lminrt, lmaxrt), (rminrt, rmaxrt) = lims
                                        if lminrt > rminrt and lmaxrt < rmaxrt: #old encompassed
                                            overpass += 1
                                        elif rminrt > lminrt and rmaxrt < lmaxrt: #new encompassed
                                            overpass += 1
                                        else:
                                            overlap = min(rmaxrt, lmaxrt) - max(rminrt, lminrt)
                                            fullrange = max(rmaxrt, lmaxrt) - min(rminrt, lminrt)
                                            if overlap / fullrange > 0.5:
                                                overpass += 1
                                if overpass > matchables / 2:
                                    diffmatrix = np.abs(obmasses - nbmasses[:,None])
                                    matchmatrix = diffmatrix < ctol
                                    mmshape = matchmatrix.shape
                                    majoraxis = np.argmax(mmshape)
                                    minoraxis = np.argmin(mmshape)
                                    matrixmatches = matchmatrix.any(axis=majoraxis)
                                    if matrixmatches.sum() > matrixmatches.size / 2: #matching at least half of the smaller with >=, make this a 'majority' with >
                                        alignmentloc = np.argwhere(diffmatrix == diffmatrix.min())[0]
                                        minindex = min(alignmentloc)
                                        alignmentloc = [i-minindex for i in alignmentloc] #for when the closest match isn't the earliest matching index
                                        rmax = nbmasses.size - alignmentloc[0]
                                        lmax = obmasses.size - alignmentloc[1]
                                        maxsize = min(rmax, lmax)
                                        matchintensityranks = matchintensities.argsort()[::-1]
                                        scoutrankslice = intensityranks[alignmentloc[1]:alignmentloc[1]+maxsize]
                                        matchrankslice = matchintensityranks[alignmentloc[0]:alignmentloc[0]+maxsize]
                                        slicesubtraction = np.abs(scoutrankslice - matchrankslice)
                                        if slicesubtraction.sum() <= slicesubtraction.size - 1: #close order matching passes
                                            nodistcharges[o][mk] = oldcharge
                    if matchkeys:
                        matchkeys.append(o)
                        chargegroups.append(matchkeys)
                #else:
                    #nodist match against old, nodists as higher charge
                    #nonoldmatches.add(o)
        #nodist match against old, nodists as higher charge
        nonoldmatches = set(oldkeys).difference(oldmatches) #if you end up needing more competition in the mix, you can just match everything to nodists and you'll get competing matches, I've left them out via routes like this so as to lessen the complications atm, competing nodist vs matches of the same charge can be annoying when determining if a distribution should extend beyond a nodist match
        nodistupmodel = spatial.KDTree(nodistmasses[:,None] * charge - proton * charge)
        nonmatchedmasses = np.array([oldmasses[n] for n in range(len(oldkeys)) if oldkeys[n] in nonoldmatches])
        nonmatchedkeys = [oldkeys[n] for n in range(len(oldkeys)) if oldkeys[n] in nonoldmatches]
        if nonmatchedmasses.size > 0: #this is a new addition, not fully tested
            upmatches = nodistupmodel.query_ball_point(nonmatchedmasses, oldradii, workers=8).tolist()
            for m, o in zip(upmatches, nonmatchedkeys):
                if m:
                    matchkeys = []
                    ominrt, omaxrt = distributiontimelimits[o]
                    omasses = np.array(distributionmasses[o])
                    obmasses = omasses * oldcharge - proton * oldcharge
                    intensities = np.array(distributionintensities[o])
                    intensityranks = intensities.argsort()[::-1]
                    for mkey in m:
                        mk = nodistkeys[mkey]
                        matchintensities = np.array(distributionintensities[mk])
                        if matchintensities.max() < intensities.max():
                            nminrt, nmaxrt = distributiontimelimits[mk]
                            nmasses = np.array(distributionmasses[mk])
                            nbmasses = nmasses * charge - proton * charge
                            #below is requiring a majority of the matchable masses have sufficiently overlapping retention times
                            if ominrt < nmaxrt and omaxrt > nminrt: #overlap exists
                                basemasses = [obmasses, nbmasses]
                                dlines = [[regions[j,2:4].tolist() for j in linesofdistributions[i]] for i in [o, mk]]
                                sizes = [i.size for i in basemasses]
                                maxind = sizes.index(max(sizes))
                                lineup = basemasses[maxind]
                                retentionboundaries = defaultdict(list)
                                for n, (sm, dlims) in enumerate(zip(basemasses, dlines)):
                                    sdiff = np.abs(lineup - sm[:,None])
                                    alignmentloc = np.argwhere(sdiff == sdiff.min())[0]
                                    alignmentloc -= alignmentloc.min()
                                    luind = alignmentloc[1] - alignmentloc[0]
                                    outinds = [luind, luind + sm.size]
                                    for ind, dlim in zip(range(*outinds), dlims):
                                        retentionboundaries[ind].append(dlim)
                                overpass = 0
                                matchables = 0
                                for ind, lims in retentionboundaries.items():
                                    if len(lims) > 1:
                                        matchables += 1
                                        (lminrt, lmaxrt), (rminrt, rmaxrt) = lims
                                        if lminrt > rminrt and lmaxrt < rmaxrt: #old encompassed
                                            overpass += 1
                                        elif rminrt > lminrt and rmaxrt < lmaxrt: #new encompassed
                                            overpass += 1
                                        else:
                                            overlap = min(rmaxrt, lmaxrt) - max(rminrt, lminrt)
                                            fullrange = max(rmaxrt, lmaxrt) - min(rminrt, lminrt)
                                            if overlap / fullrange > 0.5:
                                                overpass += 1
                                if overpass > matchables / 2:
                                    diffmatrix = np.abs(obmasses - nbmasses[:,None])
                                    matchmatrix = diffmatrix < ctol
                                    mmshape = matchmatrix.shape
                                    majoraxis = np.argmax(mmshape)
                                    minoraxis = np.argmin(mmshape)
                                    matrixmatches = matchmatrix.any(axis=majoraxis)
                                    if matrixmatches.sum() > matrixmatches.size / 2: #matching at least half of the smaller with >=, make this a 'majority' with >
                                        alignmentloc = np.argwhere(diffmatrix == diffmatrix.min())[0]
                                        minindex = min(alignmentloc)
                                        alignmentloc = [i-minindex for i in alignmentloc] #for when the closest match isn't the earliest matching index
                                        rmax = nbmasses.size - alignmentloc[0]
                                        lmax = obmasses.size - alignmentloc[1]
                                        maxsize = min(rmax, lmax)
                                        matchintensityranks = matchintensities.argsort()[::-1]
                                        scoutrankslice = intensityranks[alignmentloc[1]:alignmentloc[1]+maxsize]
                                        matchrankslice = matchintensityranks[alignmentloc[0]:alignmentloc[0]+maxsize]
                                        slicesubtraction = np.abs(scoutrankslice - matchrankslice)
                                        if slicesubtraction.sum() <= slicesubtraction.size - 1: #close order matching passes
                                            nodistcharges[o][mk] = charge
                    if matchkeys:
                        matchkeys.append(o)
                        chargegroups.append(matchkeys)
    oldmasses = distmasses.copy()
    #oldradii = distradii.copy()
    oldradii = ctol
    oldkeys = distkeys.copy()
    oldcharge = charge
    #match masses via ball_point
    #then check rt overlaps, call distributiontimelimits within in the loop I suppose? instead of mapping
    #overlaps that pass get added to chargesets to be rank checked elsewhere
    #loop to filter by RT overlap
    moving = True

#for any nodists still left, a nodist-to-nodist match? it's a pain but it should be done I suppose

#current priority: add in nodists to the above search and distribution dict groups, and grant it a temp charge within each distribution it matches

print(time() - t12, 'charge group linkage')
t13 = time()

#combining redundant matches
chargesets = intersection_merge(chargegroups)

#i believe this sets up every shortened version of chargelayers to be used below, but i need to double check this
chargelayers = defaultdict(lambda: defaultdict(set)) #groupid: charge: [connections]
extralayercharges = defaultdict(dict) #dist: nodist: charge
distchargesofnodists = defaultdict(dict) #nodist: dists: charge
for n, cs in enumerate(chargesets):
    for c in cs:
        chargelayers[n][distributioncharges[c]].add(c)
        if c in nodistcharges:
            for nc, ncc in nodistcharges[c].items():
                chargelayers[n][ncc].add(nc)
                extralayercharges[n][nc] = ncc
                distchargesofnodists[linesofdistributions[nc][0]][c] = ncc

#I suppose I should just use ths smallest charge group's size as the basis for matching the others, but it bothers me kinda
#I suppose, if that smallest group's size fails to match things, then divide things into those that matched and those that didn't? Charge order will matter here
#doing the smallest one for now, and throwing anything that doesn't pass into a separate list to visualize the negatives
#there's also some regulatory disparity for matches that don't have smaller matches I suppose
#chargecongroups = defaultdict(dict) #connection: [all chargegroups its involved in]
##distmatchcount = defaultdict(int) #it's for plotting, I guess
#for cl in chargelayers.values():
#    for cons in itertools.product(*cl.values()):
#        #for con in cons:
#        #    distmatchcount[con] += 1
#        basemasses = [np.array(distributionmasses[i])*distributioncharges[i]-proton*distributioncharges[i] for i in cons]
#        intensities = [np.array(distributionintensities[i]) for i in cons] 
#        sizes = [i.size for i in intensities]
#        maxind = sizes.index(max(sizes))
#        lineup = basemasses[maxind]
#        distinds = []
#        for sm in basemasses:
#            sdiff = np.abs(lineup - sm[:,None])
#            alignmentloc = np.argwhere(sdiff == sdiff.min())[0]
#            alignmentloc -= alignmentloc.min()
#            luind = alignmentloc[1] - alignmentloc[0]
#            outinds = [luind, luind + sm.size]
#            distinds.append(outinds)
#        leftbound = max(i[0] for i in distinds) 
#        rightbound = min(i[1] for i in distinds)
#        alignmentsize = rightbound - leftbound
#        crossmasses = []
#        crossintensities = []
#        for n, (lc, rc) in enumerate(distinds):
#            dl = leftbound - lc
#            dr = dl + alignmentsize
#            crossmasses.append(basemasses[n][dl:dr])
#            crossintensities.append(intensities[n][dl:dr])
#        crossintensities = np.array(crossintensities)
#        crossmasses = np.array(crossmasses)
#        crossintensitysums = crossintensities.sum(axis=0)
#        massmeandiff = np.abs(crossmasses.mean(axis=0) - crossmasses).mean()
#        intensitypercs = crossintensities / crossintensitysums
#        intensitymeandiff = np.abs(intensitypercs.mean(axis=1)[:,None] - intensitypercs).mean()
#        for con in cons:
#            chargecongroups[con][cons] = [intensitymeandiff, massmeandiff]

#^this is currently reflective of the entire match, under only masses that match
#however, I want the values to be more reflective of how each mass matches its neighbors in the distribution, not the distribution as a whole.
#^this should also put me a step closer to including nodists as well...
#so I actually do want to judge the distribution as a whole, but I need to judge these values by isotopomer position rather than just what overlaps
#also i need to be able to INCLUDE each adjacent combination of all of these matches in order to flesh the logic out correctly.
#I'm not going to do every combination of dists, but I will preserve as many isotopomer position values as possible, the current alignments need to go

#actually, for nodists, it might be easier to just do a post-matching search

cid = 0
conpairs = {} #cid: adjacent charge-state matches as pairs
chargeconsets = set() #finalized charge-state groups for lookup
chargecongroups = defaultdict(dict) #connection: [all chargegroups its involved in]
#distmatchcount = defaultdict(int) #it's for plotting, I guess
for ck, cl in chargelayers.items():
    for connections in itertools.product(*cl.values()):
        connectioncharges = {}
        for con in connections:
            if distributioncharges[con] > 0:
                bc = distributioncharges[con]
            else:
                bc = extralayercharges[ck][con]
            connectioncharges[con] = bc
        connections = sorted(connections, key=lambda x: connectioncharges[x])
        clen = len(connections)
        startind = 0
        endind = 2
        coniters = []
        restart = True
        while endind <= clen:
            if restart:
                tempind = 0
                restart = False
            coniters.append(tuple(connections[startind+tempind:endind+tempind]))
            tempind += 1
            if endind + tempind > clen:
                restart = True
                endind += 1
        for cons in coniters:
            basemasses = []
            for con in cons:
                if distributioncharges[con] > 0:
                    bc = distributioncharges[con]
                else:
                    bc = extralayercharges[ck][con]
                bm = np.array(distributionmasses[con]) * bc - proton * bc
                basemasses.append(bm)
            intensities = [np.array(distributionintensities[i]) for i in cons] 
            retentiontimes = [distributiontimelimits[i] for i in cons]
            #^this wouldn't be so great, I want individual isotopomer overlaps to be assessed here
            dlines = [[regions[j,2:4].tolist() for j in linesofdistributions[i]] for i in cons]
            sizes = [i.size for i in intensities]
            maxind = sizes.index(max(sizes))
            lineup = basemasses[maxind]
            masslines = defaultdict(list)
            intensitylines = defaultdict(list)
            retentionboundaries = defaultdict(list)
            lineinds = defaultdict(list)
            for n, (sm, sints, dlims) in enumerate(zip(basemasses, intensities, dlines)):
                sdiff = np.abs(lineup - sm[:,None])
                alignmentloc = np.argwhere(sdiff == sdiff.min())[0]
                alignmentloc -= alignmentloc.min()
                luind = alignmentloc[1] - alignmentloc[0]
                outinds = [luind, luind + sm.size]
                #below allows for mean diffs to be made on unequal sized arrays
                for ind, mass, sint, dlim in zip(range(*outinds), sm.tolist(), sints, dlims):
                    masslines[ind].append(mass)
                    intensitylines[ind].append(sint)
                    retentionboundaries[ind].append(dlim)
                    lineinds[ind].append(n)
            #all isotopomers of a basemass must overlap -> the whole group must overlap with everything in that group, and > 1/2 of the groups must fulfill this
            #negative values [that form via non-overlaps] will get removed from mainoverlaps below
            matchcount = 0
            passedmatches = 0
            for k, v in retentionboundaries.items():
                if len(v) > 1:
                    matchcount += 1
                    v = np.array(v)
                    if np.logical_and(v[:,0,None] <= v[:,1], v[:,1,None] >= v[:,0]).all():
                        passedmatches += 1
            if passedmatches > matchcount / 2:
                massmeans = []
                intensitylocations = defaultdict(list)
                rtoverlaps = []
                #i need to re-examine this, it doesn't compare RT-bounds across charge states, only within a charge state? why? Wait, I think this is across matching isotopomers of different charge states, but I need to confirm then note it here
                for n, inds in lineinds.items():
                    if len(inds) > 1:
                        massline = np.array(masslines[n])
                        massmeans.extend(np.abs(massline.mean() - massline).tolist())
                        intensityline = np.array(intensitylines[n])
                        intensityperc = intensityline / intensityline.sum()
                        for p, i in zip(intensityperc.tolist(), inds):
                            intensitylocations[i].append(p)
                        rbounds = np.array(retentionboundaries[n])
                        #not doing the fully vectorized approach so that selfoverlaps can be excluded while not infringing upon any actual 100% overlaps, although i could probably just take out a diagonal somewhere
                        for ni in range(len(rbounds)-1):
                            mainrb = rbounds[ni].copy()
                            nonmainrbs = rbounds[ni+1:].copy()
                            leftinners = nonmainrbs[:,0].copy()
                            leftinners[leftinners < mainrb[0]] = mainrb[0]
                            rightinners = nonmainrbs[:,1].copy()
                            rightinners[rightinners > mainrb[1]] = mainrb[1]
                            leftouters = nonmainrbs[:,0].copy()
                            leftouters[leftouters > mainrb[0]] = mainrb[0]
                            rightouters = nonmainrbs[:,1].copy()
                            rightouters[rightouters < mainrb[1]] = mainrb[1]
                            mainoverlaps = (rightinners - leftinners) / (rightouters - leftouters)
                            mainoverlaps = mainoverlaps[mainoverlaps > 0]
                            if mainoverlaps.size > 0:
                                rtoverlaps.extend(mainoverlaps.tolist())
                intensitymeans = []
                for k, v in intensitylocations.items():
                    v = np.array(v)
                    intensitymeans.extend(np.abs(v.mean() - v).tolist())
                massmeandiff = np.mean(massmeans)
                intensitymeandiff = np.mean(intensitymeans)
                generalizedoverlap = 1 / np.mean(rtoverlaps)
                for cn in range(len(cons)-1):
                    conpair = cons[cn:cn+2]
                    conpairs[cid] = conpair
                    for con in conpair:
                        chargecongroups[con][cid] = [intensitymeandiff, massmeandiff, generalizedoverlap]
                    cid += 1
                chargeconsets.add(tuple(sorted(cons)))

#both this process and it's look-alike below for the regular pairs are not so much a normalizing, it's more like a balancing.
prioritycharges = []
secondpriorities = []
for con, congroups in chargecongroups.items():
    if len(congroups) > 1:
        csums = np.array(list(congroups.values())).sum(axis=0)
        #0s here would actually be pretty welcome
        csums[csums == 0] = 1
        n1, n2, n3 = csums
        for congroup, (s1, s2, s3) in congroups.items():
            newconscore = sum((s1/n1, s2/n2, s3/n3))
            prioritycharges.append([congroup, newconscore])
    else:
        congroup, conscore = list(congroups.items())[0]
        conscore = conscore[:-1] #going to exclude the rt overlap here as there's no competition I suppose
        secondpriorities.append([congroup, sum(conscore)])

rankedcharges = sorted(prioritycharges, key=lambda x: x[1]) #competition among matched chargegroups where matching can be done easily
secondprioritycharges = sorted(secondpriorities, key=lambda x: x[1]) #no competition happening here, the order doesn't actually matter
rankedcharges.extend(secondprioritycharges)

blocked = set()
preservedchargeranks = []
for group, score in rankedcharges:
    if group not in blocked:
        preservedchargeranks.append(conpairs[group])
        blocked.add(group)

print(time() - t13, 'charge set and priority ranking')
t14 = time()

#old ranking format
#chargeid = 0
#blocked = set()
#chargedistlines = defaultdict(lambda: defaultdict(set)) #chargegroupid: charge: [lines]
#chargedistgroups = defaultdict(dict) #chargegroupid: charge: distributionid
#chargegroupsbyline = {} #line: chargegroupid, doubles as blocking list
#chargesbyline = {} #line: charge
#for pn, (cons, score) in enumerate(preservedchargeranks):
#    #join anything that has a sorted tuple key to connectionsbykeys with the existing line infrastructure at each charge state
#    #ckeys = set(itertools.chain(*(linesofdistributions[i] for i in cons)))
#    if not any(i in blocked for i in cons):
#        #^if none are blocked, none have been used -> make new group I suppose?
#        grouplines = {}
#        groupdists = {}
#        for con in cons:
#            charge = distributioncharges[con]
#            distkeys = linesofdistributions[con]
#            chargedistlines[chargeid][charge] = distkeys
#            chargedistgroups[chargeid][charge] = con
#            for line in distkeys:
#                chargegroupsbyline[line] = chargeid
#                chargesbyline[line] = charge
#        blocked.update(cons)
#        chargeid += 1

sn = 0
groupsofdists = {} #dist: chargegroup id
chargegroups = defaultdict(set) #chargegroup id: [dist ids]
for dists in preservedchargeranks:
    locs = set()
    for i in dists:
        if i in groupsofdists:
            locs.add(groupsofdists[i])
    if locs:
        joiner = min(locs)
        if len(locs) > 1:
            tempchargegroup = set(dists)
            for l in locs:
                tempchargegroup.update(chargegroups[l])
            if tuple(sorted(tempchargegroup)) in chargeconsets:
                for oldlocs in locs.difference([joiner]):
                    for ol in chargegroups[oldlocs]:
                        groupsofdists[ol] = joiner
                    chargegroups[joiner].update(chargegroups.pop(oldlocs))
                chargegroups[joiner].update(dists)
                for i in dists:
                    groupsofdists[i] = joiner
        else:
            if tuple(sorted(chargegroups[joiner].union(dists))) in chargeconsets:
                chargegroups[joiner].update(dists)
                for i in dists:
                    groupsofdists[i] = joiner
    else:
        joiner = sn
        chargegroups[joiner].update(dists)
        for i in dists:
            groupsofdists[i] = joiner
        sn += 1

print(time() - t14, 'charge group assembly')

#chargedistlines = defaultdict(lambda: defaultdict(set)) #chargegroupid: charge: [lines]
chargedistgroups = defaultdict(dict) #chargegroupid: charge: distributionid
chargegroupsbyline = {} #line: chargegroupid, doubles as blocking list
for groupid, dists in chargegroups.items():
    for dist in dists:
        charge = distributioncharges[dist]
        chargedistgroups[groupid][charge] = dist
        distlines = linesofdistributions[dist]
        #chargedistlines[groupid][charge] = distlines
        for line in distlines:
            chargegroupsbyline[line] = groupid

print(time() - t1, 'total to charge groups')


t15 = time()

additionaldistributions = set(chargegroupsbyline).intersection(nodists) #nodists that made it into distributions will be given their own distribution
distributionchangesetup = {k:distributionsoflines[k] for k in additionaldistributions}

#removing all previous nodists from any distribution dicts
#finaldefiniteind = max(max(i.keys()) for i in solodists.values()))
indcopy = finaldefiniteind
while indcopy in distributionmasses:
    del distributionmasses[indcopy]
    del distributiontimelimits[indcopy]
    del distributionintensities[indcopy]
    distributionsbycharge[distributioncharges[indcopy]].remove(indcopy)
    for line in linesofdistributions[indcopy]:
        del distributionsoflines[line]
    del linesofdistributions[indcopy]
    indcopy += 1

distributionchanges = {}
for line in additionaldistributions:
    dreg = regions[line]
    cgroup = chargegroupsbyline[line]
    potentialcharges = distchargesofnodists[line]
    chargekey = chargegroups[cgroup].intersection(potentialcharges)
    if len(chargekey) > 1:
        print('big error -', line, 'has multiple potential nodist keys')
    ckey = list(chargekey)[0]
    charge = distchargesofnodists[line][ckey]
    dmass = dreg[7] 
    dintensity = dreg[5]
    rtlimit = dreg[2:4] 
    minrt = rtlimit.min()
    maxrt = rtlimit.max()
    distributionmasses[finaldefiniteind] = [dmass]
    distributioncharges[finaldefiniteind] = charge
    distributionsoflines[line] = finaldefiniteind
    linesofdistributions[finaldefiniteind] = [line]
    distributiontimelimits[finaldefiniteind] = [minrt, maxrt]
    distributionintensities[finaldefiniteind] = [dintensity]
    #distributionsbycharge[charge][dist] = sortedlines
    distributionsbycharge[0].append(finaldefiniteind)
    oldkey = distributionchangesetup[line]
    distributionchanges[oldkey] = finaldefiniteind
    solodists[charge][finaldefiniteind] = [line]
    finaldefiniteind += 1 #continuing from solodists count


distributionregions = []
for k, v in solodists.items():
    for sk, sv in v.items():
        masses = np.array(distributionmasses[sk])
        massmax = masses.max()
        massmin = masses.min()
        intensities = np.array(distributionintensities[sk])
        mainmass = masses[intensities.argmax()]
        mintime, maxtime = distributiontimelimits[sk]
        signalsum = defaultdict(float) #time: total intensity
        for line in sv:
            data = trackedgroups[line]
            for t, i in zip(data[1], data[2]):
                signalsum[t] += i
        signals = np.array(list(signalsum.items()))
        area = np.trapezoid(signals[:,1], signals[:,0])
        el = [massmin, massmax, mintime, maxtime, len(sv), area, k, mainmass, sk]
        distributionregions.append(el)

distributionregions = np.array(distributionregions)
distributionregions = distributionregions[distributionregions[:,8].argsort()]

print(time() - t15, 'distribution regions')

t16 = time()

chargestatelines = {} #line: charge, for things in chargegroups
chargeregions = []
for k, v in chargedistgroups.items():
    mincharge = min(v)
    maxcharge = max(v)
    times = []
    signalsum = defaultdict(float) #time: total intensity
    changers = {}
    for vc, sv in v.items():
        if sv in distributionchanges:
            changers[vc] = sv
    for vc, sv in changers.items():
        del chargedistgroups[k][vc]
        sv = distributionchanges[sv]
        vc = distributioncharges[sv]
        chargedistgroups[k][vc] = sv
    for vc, sv in v.items():
        times.extend(distributiontimelimits[sv])
        for line in linesofdistributions[sv]:
            data = trackedgroups[line]
            for t, i in zip(data[1], data[2]):
                signalsum[t] += i
            chargestatelines[line] = vc
    signals = np.array(list(signalsum.items()))
    sa = signals[:,0].argsort()
    signals = signals[sa]
    area = np.trapezoid(signals[:,1], signals[:,0])
    distinds = np.array(list(v.values()))
    charges = np.array(list(v))
    maincharge = charges[distributionregions[distinds,5].argmax()]
    mintime = min(times)
    maxtime = max(times)
    el = [mincharge, maxcharge, mintime, maxtime, len(v), area, maincharge, k]
    chargeregions.append(el)
chargeregions = np.array(chargeregions)

print(time() - t16, 'charge regions')
t17 = time()

#pack these up with distributionsoflines
#analyte id == chargegroup id if there is one, continue using sn for this count
analytekeys = defaultdict(dict) #analyteid: distid: charge, this can be one or multiple distributions based on whether it has multiple charge states
analytedistributions = {} #analyteid: [[weighted means [via intensity] across isotopomers from every charge state if there are multiple], [AUC of merged isotopomers]]
analytesbydistribution = {} #distid: analyte id
linesofanalytes = {} #analyteid: [[lines across charge states at this position], [...]]
blocked = set()
for k, v in chargedistgroups.items():
    massholder = []
    lineholder = []
    for charge, distid in v.items():
        analytekeys[k][distid] = charge
        dmasses = np.array(distributionmasses[distid]) * charge - proton * charge
        dlines = linesofdistributions[distid]
        massholder.append(dmasses)
        lineholder.append(dlines)
        analytesbydistribution[distid] = k
        blocked.add(distid)
    
    sortedlines = sorted(lineholder, key=lambda x: len(x))
    sortedmasses = sorted(massholder, key=lambda x: x.size)
    #lineup = sortedmasses[-1][:,None]
    lineup = sortedmasses[-1][:,None]
    distinds = []
    for sm in sortedmasses:
        sdiff = np.abs(lineup - sm)
        alignmentloc = np.argwhere(sdiff == sdiff.min())[0]
        alignmentloc -= alignmentloc.min()
        luind = alignmentloc[0] - alignmentloc[1]
        outinds = [luind, luind + sm.size]
        distinds.append(outinds)
    
    spaceorganizer = defaultdict(list) #position: [lineuids]
    spacemasses = defaultdict(list) #position: [basemasses]
    for di, sl, sm in zip(distinds, sortedlines, sortedmasses):
        dspaces = range(*di)
        for space, line, mass in zip(dspaces, sl, sm):
            spaceorganizer[space].append(line)
            spacemasses[space].append(mass)
    
    linesofanalytes[k] = []
    massesandintensities = [[], []]
    for sk in sorted(spaceorganizer):
        slines = spaceorganizer[sk]
        smasses = np.array(spacemasses[sk])
        areavals = regions[slines,5]
        weightedmass = (smasses * areavals).sum() / areavals.sum()
        signalsum = defaultdict(float) #time: sum intensity
        for line in slines:
            linegroup = trackedgroups[line]
            times = linegroup[1]
            intensities = linegroup[2]
            for t, i in zip(times, intensities):
                signalsum[t] += i
        signals = np.array(list(signalsum.items()))
        sa = signals[:,0].argsort()
        signals = signals[sa]
        area = np.trapezoid(signals[:,1], signals[:,0])
        #analytedistributions[k][weightedmass] = area
        massesandintensities[0].append(weightedmass)
        massesandintensities[1].append(area)
        linesofanalytes[k].append(slines)
    analytedistributions[k] = np.array(massesandintensities)

#adding any distribution without multiple charge states to analytedistributions
for distid, masses in distributionmasses.items():
    if distid not in blocked:
        charge = distributioncharges[distid]
        masses = np.array(masses)
        basemasses = masses * charge - proton * charge
        intensities = distributionintensities[distid]
        analytekeys[sn][distid] = charge
        analytesbydistribution[distid] = sn
        #for m, i in zip(basemasses.tolist(), intensities):
        #    analytedistributions[sn][m] = i
        massesandintensities = np.array([basemasses, intensities])
        analytedistributions[sn] = massesandintensities
        linesofanalytes[sn] = linesofdistributions[distid]
        sn += 1

print(time() - t17, 'summarized analyte distributions')

#charges across time for dists and charges independently
#dist length vs time
#number of charge states vs time
#number of charge states vs distribution length

#not yet done:
#visualize most common charge ranges, is it 1-2, 2-3, 2-4, etc?

starts = defaultdict(set)
ends = defaultdict(set)
for r in distributionregions:
    starts[r[2]].add(int(r[-1]))
    ends[r[3]].add(int(r[-1]))

pool = set()
timestats = []
titles = ['# distributions', '#1+', '#2+', '#3+', '#4+', '#5+', '#6+^', '#2-length', '#3-lengths', '#4-lengths', '#5-lengths', '#6^-lengths', '# unique charge states', '# unique distribution lengths', '#2-states', '#3-states', '#4-states', '#5^-states', '# total distibutions with multiple charge states', 'time']
for t in timearray.tolist():
    pool.update(starts[t])
    timetracker = np.zeros(len(titles))
    uniquecharges = set()
    uniquelengths = set()
    for p in pool:
        r = distributionregions[p]
        timetracker[0] += 1
        if r[6] < 6: #charge
            timetracker[int(r[6])] += 1
        else:
            timetracker[6] += 1
        if r[4] < 6: #distribution length
            timetracker[int(r[4])+5] += 1
        else:
            timetracker[11] += 1
        uniquecharges.add(r[6])
        uniquelengths.add(r[4])
        if p in chargestatelines:
            timetracker[18] += 1
            cbl = chargestatelines[p]
            if cbl < 5:
                timetracker[cbl+12] += 1
            else:
                timetracker[17] += 1
    timetracker[0] += len(pool)
    timetracker[12] += len(uniquecharges)
    timetracker[13] += len(uniquelengths)
    timetracker[19] += t
    timestats.append(timetracker.tolist())
    for r in ends[t]:
        pool.remove(r)
timestats = np.array(timestats)

plt.plot(timestats[:,19], timestats[:,0], '-', color='white', label='# dists')
plt.legend()
plt.show()

plt.plot(timestats[:,19], timestats[:,1], '-', alpha=0.5, color='cyan', label='1+')
plt.plot(timestats[:,19], timestats[:,2], '-', alpha=0.5, color='fuchsia', label='2+')
plt.plot(timestats[:,19], timestats[:,3], '-', alpha=0.5, color='white', label='3+')
plt.plot(timestats[:,19], timestats[:,4], '-', alpha=0.5, color='orangered', label='4+')
plt.plot(timestats[:,19], timestats[:,5], '-', alpha=0.5, color='lawngreen', label='5+')
plt.plot(timestats[:,19], timestats[:,5], '-', alpha=0.5, color='darkorange', label='6^+')
plt.legend()
plt.show()

plt.plot(timestats[:,19], timestats[:,7], '-', alpha=0.5, color='cyan', label='2L')
plt.plot(timestats[:,19], timestats[:,8], '-', alpha=0.5, color='fuchsia', label='3L')
plt.plot(timestats[:,19], timestats[:,9], '-', alpha=0.5, color='white', label='4L')
plt.plot(timestats[:,19], timestats[:,10], '-', alpha=0.5,  color='orangered', label='5L')
plt.plot(timestats[:,19], timestats[:,11], '-', alpha=0.5,  color='lawngreen', label='6^L')
plt.legend()
plt.show()

plt.plot(timestats[:,19], timestats[:,12], '-', color='orangered', label='Unique Charges')
plt.plot(timestats[:,19], timestats[:,13], '-', color='lawngreen', label='Unique Lengths')
plt.legend()
plt.show()


plt.plot(timestats[:,19], timestats[:,14], '-', color='cyan', label='2S')
plt.plot(timestats[:,19], timestats[:,15], '-', color='fuchsia', label='3S')
plt.plot(timestats[:,19], timestats[:,16], '-', color='white', label='4S')
plt.plot(timestats[:,19], timestats[:,17], '-', color='orangered', label='5^S')
plt.legend()
plt.show()

plt.plot(timestats[:,19], timestats[:,18], '-', color='orangered', label='# multiply charged')
plt.legend()
plt.show()


chargecount = defaultdict(int)
for charge, distids in solodists.items():
    chargecount[charge] += len(distids)
plt.bar(chargecount.keys(), chargecount.values())
plt.title('# of dists by charge')
plt.show()

mismatches = {} #lineuid: [mycharge, [vendor]]
agreedmatches = {} #lineuid: [mycharge, n matches / total]
for k, v in precursorapparentcharge.items():
    if k in distributionsoflines:
        dist = distributionsoflines[k]
        dcharge = distributioncharges[dist]
        if any(i == dcharge for i in v):
            agreedmatches[k] = [dcharge, v.count(dcharge) / len(v), len(v)]
        else:
            mismatches[k] = [dcharge, v]

#i also want to compare mode results from dists that have multiple vendor guesses as matching/nonmatching
#any nodist matches here?
#also draw box-like surrounding lines on targeted dists in the viewer below to make it easier to pick out the relevant dist
print('disagreed charges -', len(mismatches))
print('agreed charges -', len(agreedmatches))

#how many mismatches are in subisomasses?
#len(set(mismatches).intersection(subisomasses))
#Out[6]: 331
#len(mismatches)
#Out[7]: 2921
#len(subisomasses)
#Out[8]: 15655
#maybe I won't care about this yet

#check w/o decreasingmultiplier, and uppermasslimit

#no decreasingmultiplier:
#disagreed charges - 2940
#agreed charges - 17425

#no uppermasslimit:
#disagreed charges - 3243
#agreed charges - 17175

#^neither
#disagreed charges - 3262
#agreed charges - 17151

#both still present
#disagreed charges - 2921
#agreed charges - 17444

#plain meandiff and datadiff
#disagreed charges - 2986
#agreed charges - 17374

#~
#dropped length of activepairlist requirement to 1 in distributionscoring
#disagreed charges - 2652
#agreed charges - 17703
#adopted below

#fixed rtoffset functionality for third priorities
#disagreed charges - 2152
#agreed charges - 17933

#newinclimit to 0.2
#disagreed charges - 1992
#agreed charges - 18041

#dpercdiff measuring intensity instead of # of data points
#disagreed charges - 2057
#agreed charges - 17933

#note why the above counts don't give a consistent total, should i be concerned?

#remove newinclimit from rtoverlap section
#make the rtmultiplier something more complex than a prod, do what you do below in the chargematching section
#newinclimit to 0.2?!?!?!?! shouldn't be an improvement - that would be better, still check it



#i want to visualize the precursors with the largest distance to matched line




#to remember what these have:
#precursorscanmatches = {} #ms2 scan index: lineuid
#precursorlinematches = defaultdict(list) #lineuid: [ms2 scan indices]
#precursorapparentcharge = defaultdict(list) #lineuid: vendor determined charge
#precursordistmatches = {}

rgblow = lambda: (np.random.uniform(low=0.6, high=1), np.random.uniform(low=0.9, high=1), np.random.uniform(low=0.7, high=1))
rgbhigh = lambda: (np.random.uniform(low=0.9, high=1), np.random.uniform(low=0, high=0.1), np.random.uniform(low=0.8, high=0.9))
hitcolor = lambda: (
        np.random.uniform(low=0.4, high=0.4), #R
        np.random.uniform(low=0.2, high=1), #B
        np.random.uniform(low=0.4, high=0.5)  #G
        )

timeadd = 0.3
massadd = 1.5
ncount = 0
for line in list(mismatches)[:-10]:
    if ncount > 10:
        break
    ncount += 1
    distribution = distributionsoflines[line]
    ddreg = distributionregions[distribution]
    zlmb, zumb, zst, zet = ddreg[:4]
    zlmb -= massadd
    zumb += massadd
    zst -= timeadd
    zet += timeadd
    boundrec = [zlmb, zumb, zst, zet]
    newdists = defaultdict(dict)
    for fc, fgs in solodists.items():
        for fk, pkeys in fgs.items():
            fg = regions[pkeys,:2]
            times = regions[pkeys,2:4]
            if fg.min() <= zumb and fg.max() >= zlmb:
                if times.max() >= zst and times.min() <= zet:
                    newdists[fc][fk] = pkeys
    text = True
    ngroups = sum(len(i) for i in newdists.values())
    cols = dp.get_colors(ngroups)
    cn = 0
    #dcolor = False
    fig, ax = plt.subplots(nrows=3, figsize=(6,8), sharex=True)
    for fc, fgs in newdists.items():
        for fk, pkeys in fgs.items():
            #if fk == distribution:
            #    dmarker = 'o'
            #    dsize=0.1
            #    dcolor = True
            #else:
            #    dmarker = '.'
            #    dsize = 0.3
            #    dcolor = False
            col = cols[cn]
            fg = regions[pkeys,7]
            cn += 1
            low, high = rgblow(), rgbhigh()
            cmap = colors.LinearSegmentedColormap.from_list('rgb', [low, high])
            paint = False
            if fk == distribution:
                paint = True
            for p in pkeys:
                a = np.array(trackedgroups[p])
                #if dcolor:
                #    if p == line:
                #        dmarker = 'x'
                #        dsize = 0.5
                #        dmap = colors.LinearSegmentedColormap.from_list('rgb', [hitcolor(), hitcolor()])
                #    else:
                #        dmap = cmap
                #    ax[2].plot(a[0], a[1], '-', color=low, linewidth=0.2, alpha=0.9)
                #    ax[2].scatter(a[0], a[1], marker=dmarker, c=a[2], s=dsize, alpha=0.9, cmap=dmap)
                #else:
                if paint:
                    if p == line:
                        ax[2].plot(a[0], a[1], '.', color=col, markersize=10, alpha=0.09)
                    else:
                        ax[2].plot(a[0], a[1], '.', color=col, markersize=10, alpha=0.03)
                ax[2].plot(a[0], a[1], '-', color=col, linewidth=0.1, alpha=0.9)
                ax[2].plot(a[0], a[1], '.', color=col, markersize=0.3, alpha=0.5)
                if text:
                    if a[0][-1] > zlmb and a[0][-1] < zumb:
                        if a[1][-1] >= zst and a[1][-1] <= zet:
                            ax[2].text(a[0][-1], a[1][-1], str(regions[p,7].round(2)), color='white', fontsize=4)
            fints = regions[pkeys,5]
            ax[0].bar(fg, fints, color=col, alpha=0.5, width=0.01, label=str(fc))
            if text:
                for fx, fy, pk in zip(fg.tolist(), fints.tolist(), pkeys):
                    if fx > zlmb and fx < zumb:
                        ax[0].text(fx, fy + fy * 0.03, str(pk), color='white', fontsize=4)
            #print(fg)
            #print(fc, '-', np.diff(sorted(fg)))
            #print('~')
            ax[1].hlines(cn, fg.min(), fg.max(), color=col, linewidth=0.6)
            for vert, pl in zip(fg, pkeys):
                ax[1].vlines(vert, cn - 0.1, cn + 0.1, color=col, linewidth=0.6)
            vi = np.sort(fg)
            if vi.size > 2:
                vstack = np.stack((vi[:-1], vi[1:]), axis=1)
                editspots = np.diff(vstack) < subisomax
                if editspots.any():
                    ewheres = np.where(editspots)[0].tolist()
                    for ew in ewheres:
                        subpair = vstack[ew].tolist()
                        subints = [spectrum[i] for i in subpair]
                        winint = subints.index(max(subints))
                        winner = subpair[winint]
                        if ew > 0:
                            #edit 1 before ew
                            vstack[ew-1,1] = winner
                        if ew < len(vstack) - 1:
                            #edit 1 after ew
                            vstack[ew+1,0] = winner
                    vstack = np.delete(vstack, ewheres, axis=0)
            else:
                vstack = vi.reshape(1, -1)
            vdiffs = np.diff(vstack)
            vflat = sorted(vstack.flatten().tolist())
            labelspots = np.mean(vstack, axis=1).tolist()
            for ls, lp in zip(labelspots, vstack.tolist()):
                labeldiff = np.diff(lp)[0].round(4)
                chargedist = (proton/fc - labeldiff).round(4)
                lstring = '~'.join((str(fc), str(labeldiff)))
                if text:
                    if ls > zlmb and ls < zumb:
                        ax[1].text(ls, cn - 0.2, lstring, fontsize=4, ha='center', color='white')

    ndmasses = regions[nodists,7]
    mdinds = np.logical_and(ndmasses >= zlmb, ndmasses <= zumb)
    ndtimes = regions[nodists,2:4]
    tdinds = np.logical_and(ndtimes.min(axis=1) <= zet, ndtimes.max(axis=1) >= zst)
    zdinds = regions[nodists,4] > 1
    finds = np.logical_and.reduce((mdinds, tdinds, zdinds))
    ndplotters = nodists[finds]
    if ndplotters.size > 0:
        ndmasses = regions[ndplotters,7].tolist()
        nints = regions[ndplotters,5].tolist()
        ax[0].bar(ndmasses, nints, alpha=0.5, color='white', width=0.01, label='N/A')
        if text:
            for fx, fy, nd in zip(ndmasses, nints, ndplotters):
                if fy > 0:
                    ax[0].text(fx, fy + fy * 0.03, str(nd), color='white', fontsize=4)
        for p in ndplotters:
            a = np.array(trackedgroups[p])
            #ax[2].scatter(a[0], a[1], marker='.', color='white', s=0.3, alpha=0.3)
            ax[2].plot(a[0], a[1], '-', linewidth=0.2, color='white', markersize=0.3, alpha=0.6)
            ax[2].plot(a[0], a[1], '.', color='white', markersize=0.3, alpha=0.3)
            if text:
                ax[2].text(a[0][-1], a[1][-1], str(regions[p,7].round(2)), color='white', fontsize=4)
    ax[0].set_yscale('log')
    ax[0].set_ylabel('intensity')
    ax[2].set_ylabel('minutes')
    ax[1].set_ylabel('distribution rank')
    ax[2].set_xlabel('m/z')
    ax[2].set_xlim(zlmb, zumb)
    ax[2].set_ylim(zst, zet)
    for label in ax[2].get_xticklabels():
        #label.set_ha("right")
        label.set_rotation(-45)
    ncols = 6
    if text:
        ax[0].legend(title='Charge', loc='upper left', bbox_to_anchor=(0, 1.3 + (0.1 * cn / ncols)), ncol=ncols)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)
    plt.show()
    fig.clf()
    plt.close()
    mine, theirs = mismatches[line]
    print('mine:', mine, '-', 'theirs:', theirs)



















































#when plotting wides, for future line model corrections, break the masses down into size ranges of ~100 da or whatever. If masses around 300 are staying tight and shit at 1500 increase a bit, whatever

#maybe for some correction things? pretty fast thus far
#test = spatial.KDTree(regions[:,7,None])
#out = test.query_ball_point(regions[:,7,None], subisomax, workers=8)
#filter by RT but allow for things with less data points to be a little bit away from a larger line I suppose



#fname = ''.join((mzmlfile.split('/')[-1].split('.')[0], '.regions.pickle'))
#saverloc = '/'.join((savedir, fname))
#savedbits = regions
#with open(saverloc, "wb") as pick:
#    pickle.dump(savedbits, pick)
#
#fname = ''.join((mzmlfile.split('/')[-1].split('.')[0], '.trackedgroups.pickle'))
#saverloc = '/'.join((savedir, fname))
#savedbits = trackedgroups
#with open(saverloc, "wb") as pick:
#    pickle.dump(savedbits, pick)
#
#fname = ''.join((mzmlfile.split('/')[-1].split('.')[0], '.modelinfo.pickle'))
#saverloc = '/'.join((savedir, fname))
#savedbits = [modeltracking, timearray]
#with open(saverloc, "wb") as pick:
#    pickle.dump(savedbits, pick)
#
#fname = ''.join((mzmlfile.split('/')[-1].split('.')[0], '.roundcutoff.pickle'))
#saverloc = '/'.join((savedir, fname))
#savedbits = roundcutoff
#with open(saverloc, "wb") as pick:
#    pickle.dump(savedbits, pick)
##add timearray to this^

#savedir = '/store/flowcharacterizations/round5/fileprocessing'
#fname = ''.join((mzmlfile.split('/')[-1].split('.')[0], '.regions.pickle'))
#saverloc = '/'.join((savedir, fname))
#savedbits = regions
#with open(saverloc, "wb") as pick:
#    pickle.dump(savedbits, pick)


#saverloc = '/store/flowcharacterizations/round3/DDAs/fileprocessing/200901_fR_400.isotopes.pickle'
#savedbits = [solodists, nodists]
#with open(saverloc, "wb") as pick:
#    pickle.dump(savedbits, pick)
#
#loaderloc = '/store/flowcharacterizations/round3/DDAs/fileprocessing/200901_fR_400.isotopes.pickle'
#with open(loaderloc, "rb") as pick:
#    solodists, nodists = pickle.load(pick)




#regionplotter = regions[regions[:,4] > 3]

masswidths = regions[:,1] - regions[:,0]
timewidths = regions[:,3] - regions[:,2]
wmeans = regions[:,7]
meantime = regions[:,2:4].mean(axis=1)

gpoint = 0
twlines = []
while not np.isnan(gpoint):
    twlines.append(gpoint)
    gpoint = timewidths[timewidths > gpoint].mean()
#twlines.append(timewidths.max())
plt.hist(timewidths, color='white', bins=100)
plt.vlines(twlines, 0, 10e4, color='black')
plt.yscale('log')
plt.show()

twlines = np.array(twlines)
for n, t in enumerate(twlines[:-1]):
    nt = twlines[n+1]
    tinds = np.logical_and(timewidths > t, timewidths <= nt)
    rtws = meantime[tinds]
    plt.hist(rtws, bins=100, color='white')
    plt.title(' '.join((str(t.round(2)), '-', str(nt.round(2)))))
    plt.show()


scans = np.array(list(modeltracking.keys()))
added, matched, nonmatched, removed = np.array(list(modeltracking.values())).transpose()
#THIS SHIT is actually SUPER FUCKING INTERESTING, because you can see the ionization patterns showing where surface chemistry changes, and where the droplet formation gets stronger and edges, those points also align with times where the number of matched lines drops
#now I want to test whether than ionization drip is the actual memory issue or not, I don't think it is
#the emerging line-like patterns that you see at lower timewidths also shows if you do a scan width, it's just because of the smaller time intervals for those shorter line signals, they don't mean anything. If you go up to the ~minute width area the pattern doesn't exist anymore
fig, ax = plt.subplots(nrows=3, figsize=(7,8), sharex=True)

ax[0].plot(meantime, timewidths, '.', color='white', alpha=0.2, markersize=0.5)

ax[1].plot(meantime, timewidths, '.', color='white', alpha=0.2, markersize=0.5)
ax[1].set_yscale('log')

#this limit below is too low, visualize the 1-minute wide peaks
#also visualize widths as scans -> still have the pattern?
ax[2].plot(meantime, timewidths, '.', color='white', alpha=0.2, markersize=0.5)
ax[2].set_ylim(0.8, 1.2)

#plt.plot(scans[1:], explicitcutoffs, '-', color='white', alpha=0.5)
#plt.plot(scans[1:], rcos, '-', color='orange', alpha=0.5)
#plt.show()

#ax[3].plot(timearray, added, '-', color='yellow', alpha=0.5, label='added')
#ax[3].plot(timearray, matched, '-', color='green', alpha=0.5, label='matched')
##ax[3].plot(scans, nonmatched, '-', color='orange', alpha=0.5, label='nonmatched')
##ax[3].plot(scans, removed, '-', color='red', alpha=0.5, label='removed')
#
#switchpoints = timearray[np.where(added > matched)[0]]
#ax[3].vlines(switchpoints, 0, matched.max(), color='cyan', linewidth=0.2)
#
#ax[3].set_yscale('log')
#ax[3].legend()
plt.show()


##using scanwidths
#lscans = np.array(list(map(timetoscans.get, regionplotter[:,2].tolist())))
#rscans = np.array(list(map(timetoscans.get, regionplotter[:,3].tolist())))
#scanwidths = rscans - lscans
#
#fig, ax = plt.subplots(nrows=4, figsize=(7,8), sharex=True)
#
#ax[0].plot(meantime, scanwidths, '.', color='white', alpha=0.2, markersize=0.5)
#
#ax[1].plot(meantime, scanwidths, '.', color='white', alpha=0.2, markersize=0.5)
#ax[1].set_yscale('log')
#
##this limit below is too low, visualize the 1-minute wide peaks
##also visualize widths as scans -> still have the pattern?
#ax[2].plot(meantime, scanwidths, '.', color='white', alpha=0.2, markersize=0.5)
#ax[2].set_ylim(200, 300)
#
##plt.plot(scans[1:], explicitcutoffs, '-', color='white', alpha=0.5)
##plt.plot(scans[1:], rcos, '-', color='orange', alpha=0.5)
##plt.show()
#
#ax[3].plot(timearray, added, '-', color='yellow', alpha=0.5, label='added')
#ax[3].plot(timearray, matched, '-', color='green', alpha=0.5, label='matched')
##ax[3].plot(scans, nonmatched, '-', color='orange', alpha=0.5, label='nonmatched')
##ax[3].plot(scans, removed, '-', color='red', alpha=0.5, label='removed')
#
#switchpoints = timearray[np.where(added > matched)[0]]
#ax[3].vlines(switchpoints, 0, matched.max(), color='cyan', linewidth=0.2)
#
#ax[3].set_yscale('log')
#ax[3].legend()
#plt.show()



#would be a nice addition to a notebook:
#i forget how but this can be done much cooler with just a counter I suppose
#intensitysum = sum(np.array(v)[2].sum() for v in trackedgroups.values())
#
#sumlist = []
#lengthlist = []
#
#pointsums = defaultdict(float)
#pointlengths = defaultdict(int)
#for v in trackedgroups.values():
#    i = len(v[0])
#    linesum = sum(v[2])
#    pointsums[i] += linesum
#    pointlengths[i] += 1
#    lengthlist.append(i)
#    sumlist.append(linesum)
#
#
#sumlist = np.array(sumlist)
#plt.plot(lengthlist, sumlist, '.', color='white', alpha=0.1)
#plt.show()
#plt.plot(lengthlist, sumlist / lengthlist, '.', color='white', alpha=0.1)
#plt.show()
#^I'm thinking of deriving the cutoff from the second plot, then keeping everything to the left of it. Then I'm thinking of applying that same cutoff to everything on the first plot, where anything that's below the cutoff is removed from the data.



#plt.bar(pointlengths.keys(), pointlengths.values(), color='white')
#plt.title('number of tracked groups of x datapoints')
#plt.show()
#plt.bar(pointlengths.keys(), pointlengths.values(), color='white')
#plt.title('number of tracked groups of x datapoints')
#plt.yscale('log')
#plt.show()
#
#plt.bar(pointsums.keys(), pointsums.values(), color='white')
#plt.title('total intensity of all tracked groups of x datapoints', y=1.05)
#plt.tight_layout()
#plt.show()
#
#plt.bar(pointsums.keys(), [i/intensitysum for i in pointsums.values()], color='white')
#plt.title('total intensity percentage of all tracked groups of x datapoints')
#plt.tight_layout()
#plt.show()
#
#pointsumsperlength = {}
#for n in range(len(pointsums)):
#    n += 1
#    if pointlengths[n] == 0 and pointsums[n] == 0:
#        pointsumsperlength[n] = 0
#    else:
#        pointsumsperlength[n] = pointsums[n] / pointlengths[n]
#
#plt.bar(pointsumsperlength.keys(), pointsumsperlength.values(), color='white')
#plt.title('group sum per group length of x datapoints')
#plt.show()
#plt.bar(pointsumsperlength.keys(), pointsumsperlength.values(), color='white')
#plt.title('group sum per group length of x datapoints')
#plt.yscale('log')
#plt.show()

#Below is both a quick assessment of the algorithm, and something that shows the resolution of the instrument in this file as a ~useable/assessable numerical value
#This could also, easily, use pooled data across multiple files to show a consistency or performance measure

#this one highlights, that those of the longest length don't have shit for intensity - they're basically the continual noise that occurs
#rorder = regions[:,4].argsort()
#plt.scatter(masswidths[rorder], regions[rorder,5], c=regions[rorder,4], alpha=0.5)
#plt.colorbar()
#plt.xlabel('mass width')
#plt.ylabel('area')
#plt.show()
#
##these do a better job at showing where that noise is coming from, what mass ranges specifically.
#plt.scatter(masswidths[rorder], regions[rorder,7], c=regions[rorder,4], alpha=0.5, norm=colors.LogNorm()) #I think the lognorm coloring is more appropriate for when there's long noise signals
#plt.colorbar()
#plt.xlabel('mass width')
#plt.ylabel('weighted mean mass')
#plt.show()
#
#plt.scatter(masswidths[rorder], regions[rorder,5], c=regions[rorder,4], alpha=0.5)
#plt.colorbar()
#plt.xlabel('mass width')
#plt.ylabel('area')
#plt.show()
#
#rorder = masswidths.argsort()
#plt.scatter(regions[rorder,4], regions[rorder,5], c=masswidths[rorder], alpha=0.5)
#plt.colorbar()
#plt.xlabel('length')
#plt.ylabel('area')
#plt.show()
#
#rorder = regions[:,5].argsort()
#plt.scatter(meantime, regions[rorder,7], c=regions[rorder,5], alpha=0.5)
#plt.colorbar()
#plt.xlabel('mean time')
#plt.ylabel('mean mass')
#plt.show()


#I think what these tell me is that the greatest indicator of something being noise is how long it's collected into a single line - which so far seems to be pretty reliable. The noise has great consisteny in mass width overall, making it easy to keep as a single line on the fly.


rgblow = lambda: (np.random.uniform(low=0.6, high=1), np.random.uniform(low=0.9, high=1), np.random.uniform(low=0.7, high=1))
rgbhigh = lambda: (np.random.uniform(low=0.9, high=1), np.random.uniform(low=0, high=0.1), np.random.uniform(low=0.8, high=0.9))
deepset = lambda: (np.random.uniform(low=0.2, high=1), np.random.uniform(low=0.4, high=1), np.random.uniform(low=0.1, high=0.8))

#the goal of the next step should be to catch this one
#the co-ionizing isotopomers have datapoints collected within the same scans. You can use this to your advantage by treating all the lone datapoints as still being their own real thing. Search for nearest data points within the range of each lone thing I suppose, and when there's two lone things with something of a similar matching composition that are also the nearest lone things to each other, combine/expand the search window.
st = 173.3
et = 175
lmb = 1568
umb = 1570.5
#^the missed example

st = 20
et = 30
lmb = 400
umb = 420

st = 70
et = 80
lmb = 550
umb = 560
#^the isotopes I want to see

#so... I need to convert all these plotkey collections to massranges in case I change the line collection
#^a database could automatically update the plotkeys via plotkeys being generated

st = 25
et = 28
lmb = 290
umb = 1800

#regiment:
#change number of minpoints -> major distributions shouldn't be affected
#add in some slightly misaligned peaks to see if they mess with anything?

#clear/easily visible isotopes:
#5 charges
st = 71.7
et = 72.8
lmb = 552.8
umb = 553.8
#consistent and less

#3 charges
st = 70.3
et = 72
lmb = 557.2
umb = 558
#consistent and less

#2 charges
st = 68
et = 70.5
lmb = 551.3
umb = 551.8
#only 1 and less

#2 charges
st = 79.5
et = 81.5
lmb = 554.3
umb = 554.8
#only 1 and less

#3 charges
st = 79.2
et = 81.3
lmb = 551.9
umb = 552.6
#consistent and less

#3 charges
st = 60.6
et = 61.1
lmb = 550.2
umb = 551.6
#consistent and less

#2 charges
st = 66.4
et = 68.8
lmb = 554.3
umb = 555.8
#consistent and less

#2 charges
st = 67.6
et = 68.6
lmb = 554.7
umb = 555.7
#consistent, less

#2 charges
st = 59.1
et = 60.4
lmb = 557.7
umb = 559.7
#consistent, less

#3 charges
st = 49.5
et = 50
lmb = 524.5
umb = 525.6
#consistent, less

#3 charges, a smaller peak has a greater than normal/and/expected chargediff, weak link?
st = 57.7
et = 60.8
lmb = 531.2
umb = 532.9

#3 charges, the last one in the list is one I believe is a part of it, but has a weaker chargedistance, is that a pattern for lesser peaks?
st = 53.4
et = 54.1
lmb = 524.9
umb = 526.3

#3 charges
st = 91.4
et = 92.1
lmb = 803
umb = 804

#2 charges - HOW DO THEY NOT LINK?!
st = 173
et = 174.6
lmb = 1568.2
umb = 1570.3


#complicated distribution combinations:
st = 53.9
et = 54.8
lmb = 402.9
umb = 404.7
#^ms_deisotope is weird for this one, it inserts masses that don't exist at an intensity of 1, as if it thinks they're supposed to be there?

#development example
st = 48.7
et = 49.7
lmb = 364.5
umb = 367.4

st = 33
et = 33.8
lmb = 352.9
umb = 354.4

st = 53.4
et = 54.2
lmb = 524.5
umb = 526.5

#getting the adjacent 2-charges on this one was tricky for most of the prototypes, I got it later
st = 49.2
et = 50
lmb = 522.2
umb = 527.29

#use minpoints=2 here, can't seem to get this wider mass variation one to work well
st = 171.49
et = 172.98
lmb = 1553.78
umb = 1555.3

#hella crazy
st = 32
et = 33
lmb = 327.1
umb = 330.2


#next i should integrate line mass widths into the isotopomer tolerance bit to see if this helps with the high-mass high-variance distributions that keep giving me trouble
#I want to also integrate the subisotopomer process with the line combining process, should there be interlacing with no overlapping timepoints

#I was thinking of iterating over masses in order while collecting both potential subisotopomers (while storing information as a minimum charge of which this group makes a subiso pair, rather than information at each individual charge) and charge partners as well
#the charge tolerance of an n charge addition to any mass would just float along the iteration

#broad iso-group collection:
#same concept as line collection, deadsignal based on number of dying lines of minmovinginds, everything within a certain range is considered but it depends on their starting points
#if something starts within a box after a lot of internal lines dies, it would just make a new box

#pwf on all lines to determine which plate overlaps a line should be allowed to associate with
#

st = 50
et = 60
lmb = 520
umb = 540

st = 30
et = 40
lmb = 320
umb = 340

st = 35
et = 35.5
lmb = 321
umb = 325.2

st = 45
et = 51
lmb = 705
umb = 715

st = 91.5
et = 92.37
lmb = 802.75
umb = 804.25

st = 173.3
et = 175
lmb = 1567
umb = 1572

st = 87
et = 95
lmb = 500
umb = 503

st = 33
et = 34.25
lmb = 352.9
umb = 354.5

st = 42.82
et = 42.83
lmb = 411.1
umb = 412.3

st = 33
et = 34.25
lmb = 352.9
umb = 354.5

st = 74
et = 78
lmb = 425
umb = 435

st = 142.5
et = 144.5
lmb = 316.6
umb = 318

st = 142
et = 144
lmb = 316.6
umb = 318

st = 49.5
et = 49.6
lmb = 364.5
umb = 367.7

st = 35
et = 35.5
lmb = 321
umb = 325.2

boundrec = [lmb, umb, st, et]
plotkeys = arg_coord_rectangle_overlap(boundrec, regions[:,:4]).tolist()

boundrec = [np.inf, 0, np.inf, 0]
for k in plotkeys:
    minmass = min(trackedgroups[k][0])
    maxmass = max(trackedgroups[k][0])
    mintime = min(trackedgroups[k][1])
    maxtime = max(trackedgroups[k][1])
    if minmass < boundrec[0]:
        boundrec[0] = minmass
    if maxmass > boundrec[1]:
        boundrec[1] = maxmass
    if mintime < boundrec[2]:
        boundrec[2] = mintime
    if maxmass > boundrec[3]:
        boundrec[3] = maxtime
lmb, umb, st, et = boundrec
timeadd = 3
massadd = 1.5
boundrec[0] -= massadd
boundrec[1] += massadd
boundrec[2] -= timeadd
boundrec[3] += timeadd

plotkeys = arg_coord_rectangle_overlap(boundrec, regions[:,:4]).tolist()

fig, ax = plt.subplots(figsize=(6,4), facecolor='gray', sharex=True)
#fig, ax = plt.subplots(figsize=(12,8), facecolor='gray', sharex=True)
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.set_facecolor('gray')

#plottedpoints = timearray[np.where(np.logical_and(timearray >= st, timearray <= et))]
#ax.vlines(plottedpoints, lmb, umb, color='black', alpha=0.2, linewidth=0.2)

#a goal for these time vs mass plots might be to make a 'connect the dots' diagram of where the relevant peaks are, it would be neater than surrounding rectangles. there would be a bunch of straight lines stacked on top of each other, a rather convenient display. each line is a peak.
for k in plotkeys:
    #if len(trackedgroups[k]) < deadsignal:
    a = np.array(trackedgroups[k])
    #a = a[:,np.logical_and(np.logical_and(a[0] >= lmb, a[0] <= umb), np.logical_and(a[1] >= st, a[1] <= et))]
    low, high = rgblow(), rgbhigh()
    #cnorm = colors.lognorm()
    cmap = colors.LinearSegmentedColormap.from_list('rgb', [low, high])
    if a.size > 0:
        ax.scatter(a[1], a[0], marker='o', c=a[2], s=0.02, alpha=1, cmap=cmap)
        ax.plot(a[1], a[0], '-', color=low, linewidth=0.2, alpha=1)
        ax.text(a[1][0], a[0][0], str(k), color='white', fontsize=4)
    #if k in trackedpeaks:
    #    for peak in trackedpeaks[k]:
    #        ax.scatter(peak[1], peak[3], color='cyan', marker='*', s=0.6, alpha=0.8)
    if k in precursorlinematches:
        pm = precursorlinematches[k]
        for p in pm:
            pc = precursorcoordinates[p]
            ax.plot(pc[0], pc[1], '*', color='tan')

#ax.set_xlim(st, et)
#ax.set_ylim(lmb, umb)
ax.set_xlabel('minutes')
ax.set_ylabel('m/z')
fig.tight_layout()
plt.show()
fig.clf()
plt.close()



#for color development
colorset = lambda: (np.random.uniform(low=0.2, high=1), np.random.uniform(low=0.4, high=1), np.random.uniform(low=0.1, high=0.8))
cn = 100
x = np.arange(cn)
y = x
colorboard = [colorset() for _ in range(cn)]
plt.scatter(x,y,c=colorboard)
plt.show()



#visualizing the ion suppression from column noise
times = np.unique(regions[:,2:4])
plt.hist(times, bins=100, color='white')
plt.show()










#connection memory improvements, new paradigm, early pair charge-handling, region format


st = 49.5
et = 49.6
lmb = 364.5
umb = 367.7

st = 48
et = 52
lmb = 363
umb = 369

#this ones a pain
st = 35
et = 35.5
lmb = 321
umb = 325.2

#there's a triple in this one that's super elusive
st = 33
et = 33.8
lmb = 352.9
umb = 354.3

st = 53.4
et = 54.2
lmb = 524.5
umb = 526.5

st = 48
et = 51
lmb = 655
umb = 665

#a good line-model correction test
st = 49.8
st = 49
et = 49.9
et = 52
lmb = 522.2
umb = 528.3




#check distributions on 2494 and 3770, and check if the distribution to the left of both of these matches charge states with each other


#does charge-state matching error all come from lower intensity ions?

# chargeregions:
# [mincharge, maxcharge, mintime, maxtime, len(v), area, maincharge, k]

chargedistkeys = chargeregions[chargeregions[:,4].argsort()][-10:,7]
#chargedistkeys = chargeregions[-10:,7] #highest ranks

for distkey in chargedistkeys:
    distkey = int(distkey)
    dists = chargedistgroups[distkey]
    chargeorder = sorted(dists)
    chlen = len(dists)
    cf = pd.DataFrame()
    chargefigures = {c:n for n, c in enumerate(chargeorder)}
    fig, ax = plt.subplots(ncols=chlen, nrows=8, figsize=(6,8), sharex='col', sharey='row')
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    cg = list(dists.values())
    chargedistbounds = [np.inf, 0]
    intensitylineup = [np.array(distributionintensities[g]) for g in cg]
    flatintensities = itertools.chain(*intensitylineup)
    maxmain = max(flatintensities)
    masslineup = [np.array(distributionmasses[g])*distributioncharges[g]-proton*distributioncharges[g] for g in cg]
    masslineup = sorted(masslineup, key=lambda x: -x.size)
    intensitylineup = sorted(intensitylineup, key=lambda x: -x.size)
    arraysizes = [i.size for i in masslineup]
    if len(set(arraysizes)) == 1:
        arraymeans = np.array(masslineup).mean(axis=0)
        intensitysums = np.array(intensitylineup).sum(axis=0)
    else:
        matrixmax = max(arraysizes)
        arraysums = np.zeros(matrixmax)
        arraydividends = np.zeros(matrixmax)
        intensitysums = np.zeros(matrixmax)
        movinglineup = masslineup[0]
        for n, (a, i) in enumerate(zip(masslineup, intensitylineup)):
            basediffs = np.abs(movinglineup - a[:,None])
            alignmentloc = np.argwhere(basediffs == basediffs.min())[0].tolist()
            minindex = min(alignmentloc)
            alignmentloc = [i-minindex for i in alignmentloc] #for when the closest match isn't the earliest matching index
            cnmax = a.size - alignmentloc[0]
            bmax = movinglineup.size - alignmentloc[1]
            maxsize = min(bmax, cnmax)
            basearray = movinglineup[alignmentloc[1]:alignmentloc[1]+maxsize]
            meanarray = a[alignmentloc[0]:alignmentloc[0]+maxsize]
            iarray = i[alignmentloc[0]:alignmentloc[0]+maxsize]
            intensitysums[alignmentloc[1]:alignmentloc[1]+maxsize] += iarray
            arraysums[alignmentloc[1]:alignmentloc[1]+maxsize] += meanarray
            arraydividends[alignmentloc[1]:alignmentloc[1]+maxsize] += 1
        arraymeans = arraysums / arraydividends
    tbarwidth = 1 / len(cg)
    tspace = 0.5
    cols = dp.get_colors(len(cg))
    cmin = np.inf
    cmax = 0
    cratiolist = []
    ccbounds = [np.inf, 0]
    for n, (charge, g) in enumerate(dists.items()):
        lines = linesofdistributions[g]
        cintensities = np.array(distributionintensities[g])
        cratios = cintensities[:-1] / cintensities[1:]
        cratios[cratios < 1] = -1 / cratios[cratios < 1]
        cratiolist.append(cratios)
        abcratios = [abs(i) for i in cratios]
        cmasses = np.array(distributionmasses[g])
        if cintensities.min() < cmin:
            cmin = cintensities.min()
        if cintensities.max() > cmax:
            cmax = cintensities.max()
        concharge = distributioncharges[g]
        expdiff = proton / concharge
        basemasses = cmasses * concharge - proton * concharge
        basediffs = np.abs(basemasses - arraymeans[:,None])
        alignmentloc = np.argwhere(basediffs == basediffs.min())[0].tolist()
        minindex = min(alignmentloc)
        alignmentloc = [i-minindex for i in alignmentloc] #for when the closest match isn't the earliest matching index
        cnmax = arraymeans.size - alignmentloc[0]
        bmax = basemasses.size - alignmentloc[1]
        maxsize = min(bmax, cnmax)
        basearray = basemasses[alignmentloc[1]:alignmentloc[1]+maxsize]
        meanarray = arraymeans[alignmentloc[0]:alignmentloc[0]+maxsize]
        meanbasediff = meanarray - basearray
        meanbaseppms = (meanbasediff / basearray) * 1000000
        cmx = cmasses[alignmentloc[1]:alignmentloc[1]+maxsize]
        acdiffs = expdiff - np.diff(cmasses)
        basediffs = acdiffs * concharge
        conhax = chargefigures[concharge]
        cwidth = 0.5 *  len(lines)
        chargelengthextra = proton / charge * 2
        nst = regions[lines,2].min() - 0.5
        net = regions[lines,3].max() + 0.5
        nlmb = regions[lines,0].min() - chargelengthextra
        numb = regions[lines,1].max() + chargelengthextra
        nboundrec = [nlmb, numb, nst, net]
        nplotkeys = arg_coord_rectangle_overlap(nboundrec, regions[:,:4]).tolist()
        for p in nplotkeys:
            if p not in lines:
                a = trackedgroups[p]
                creg = regions[p]
                ax[0][conhax].plot(a[0], a[1], '.', color='white', markersize=0.8, alpha=0.3)
                ax[0][conhax].plot(a[0], a[1], '-', color='white', linewidth=0.4, alpha=1)
                ax[1][conhax].plot([creg[7], creg[7]], [0, creg[5]], '-', color='white', alpha=0.5, linewidth=cwidth)
        ax[5][conhax].hlines(0, cmasses.min(), cmasses.max(), color='black', linewidth=0.3)
        ax[7][conhax].hlines(0, cmasses.min(), cmasses.max(), color='black', linewidth=0.3)
        ax[2][conhax].hlines(proton, cmasses.min(), cmasses.max(), color='black', linewidth=0.3)
        for cline in lines:
            creg = regions[cline]
            ax[1][conhax].plot([creg[7], creg[7]], [0, creg[5]], '-', color=cols[n], alpha=1, linewidth=cwidth)
            a = trackedgroups[cline]
            ax[0][conhax].plot(a[0], a[1], '.', color=cols[n], markersize=0.8, alpha=0.3)
            ax[0][conhax].plot(a[0], a[1], '-', color=cols[n], linewidth=0.4, alpha=1)
            #ax[2][conhax].plot([creg[7], creg[7]], [0, creg[4]], '-', color=cols[n], alpha=1, linewidth=cwidth)
        basenorms = basediffs / proton
        abasediffs = np.abs(basenorms / proton - basediffs)
        diffgen = 0.05
        bn = 0
        bw = 0.02
        adjacentx = cmasses[:-1] + np.diff(cmasses) / 2
        ax[5][conhax].bar(adjacentx, cratios, width=diffgen, color=cols[n], alpha=1)
        #ax[6][conhax].bar(adjacentx, cpointratios, width=diffgen, color=cols[n], alpha=1)
        chargedists = np.diff(cmasses) * charge
        ax[2][conhax].bar(adjacentx, chargedists, width=diffgen, color=cols[n], alpha=1)
        intensitypercs = cintensities[:intensitysums.size] / intensitysums[:cintensities.size]
        for ip in chargedists:
            if ip < chargedistbounds[0]:
                chargedistbounds[0] = ip
            if ip > chargedistbounds[1]:
                chargedistbounds[1] = ip
        ax[4][conhax].bar(cmasses, intensitypercs, width=diffgen, color=cols[n], alpha=1)
        ax[6][conhax].bar(cmx, meanbaseppms, width=diffgen, color=cols[n], alpha=1)
        for nc, cn in enumerate(cg):
            cnmasses = np.array(distributionmasses[cn])
            cncharge = np.array(distributioncharges[cn])
            cnbases = cnmasses * cncharge - proton * cncharge
            cnintensities = np.array(distributionintensities[cn])
            basediffs = np.abs(basemasses - cnbases[:,None])
            alignmentloc = np.argwhere(basediffs == basediffs.min())[0].tolist() #[1, 0] here would mean the second of cnbases aligns to the first of basemasses
            minindex = min(alignmentloc)
            alignmentloc = [i-minindex for i in alignmentloc] #for when the closest match isn't the earliest matching index
            cnmax = cnbases.size - alignmentloc[0]
            bmax = basemasses.size - alignmentloc[1]
            maxsize = min(bmax, cnmax)
            basearray = cintensities[alignmentloc[1]:alignmentloc[1]+maxsize]
            cnarray = cnintensities[alignmentloc[0]:alignmentloc[0]+maxsize]
            cnratio = cnarray / basearray
            bx = cmasses[alignmentloc[1]:alignmentloc[1]+maxsize]+(diffgen*nc)
            cnratiobar = np.abs(cnratio.mean() - cnratio).mean()
            if cnratiobar > ccbounds[1]:
                ccbounds[1] = cnratiobar
            if cnratiobar < ccbounds[0] and cnratiobar > 0:
                ccbounds[0] = cnratiobar
            cf.loc[g, cn] = cnratiobar
            ax[3][conhax].bar(bx, cnratio, width=diffgen, color=cols[nc], alpha=1)
            #ax[4][conhax].bar(bx, pointratio, width=diffgen, color=cols[nc], alpha=1)
        for nc, cn in enumerate(cg):
            if cn != g:
                maincharge = np.array(distributioncharges[cn])
                mainmasses = np.array(distributionmasses[cn])
                mainbasemasses = mainmasses * maincharge - proton * maincharge
                basediffs = np.abs(basemasses - mainbasemasses[:,None])
                alignmentloc = np.argwhere(basediffs == basediffs.min())[0].tolist() #[1, 0] here would mean the second of cnbases aligns to the first of basemasses
                minindex = min(alignmentloc)
                alignmentloc = [i-minindex for i in alignmentloc] #for when the closest match isn't the earliest matching index
                cnmax = mainbasemasses.size - alignmentloc[0]
                bmax = basemasses.size - alignmentloc[1]
                maxsize = min(bmax, cnmax)
                basearray = basemasses[alignmentloc[1]:alignmentloc[1]+maxsize]
                mainarray = mainbasemasses[alignmentloc[0]:alignmentloc[0]+maxsize]
                maindiffs = mainarray - basearray
                mainppm = (maindiffs / mainarray) * 1000000
                bx = cmasses[alignmentloc[1]:alignmentloc[1]+maxsize]+(diffgen*nc)
                diffbar = np.abs(mainppm.mean() - mainppm).mean()
                ax[7][conhax].bar(bx+bn, mainppm, width=bw, color=cols[nc], alpha=1)
                bn += bw + bw / 2
        ax[0][conhax].set_title(''.join((str(concharge), '(', str(g), ')')), fontsize=12)
    ax[1][0].set_yscale('log')
    #ax[2][0].set_yscale('log')
    ax[2][0].set_ylim(chargedistbounds[0]*0.95, chargedistbounds[1]*1.05)
    ax[3][0].set_yscale('log')
    ax[5][0].set_yscale('symlog')
    ax[4][0].set_yscale('log')
    #ax[6][0].set_yscale('symlog')
    ax[1][0].set_ylim(cmin/2, cmax)
    ax[1][0].set_ylabel('peak area')
    ax[0][0].set_ylabel('retention time', fontsize=6)
    ax[3][0].set_ylabel('cross-charge', fontsize=7)
    ax[5][0].set_ylabel('adjacency')
    ax[7][0].set_ylabel('ppm error')
    ax[2][0].set_ylabel('charge distances', fontsize=6)
    ax[4][0].set_ylabel('intensity sum %', fontsize=6)
    ax[6][0].set_ylabel('ppm to mean', fontsize=7)
    for ch, hax in chargefigures.items():
        ax[-1][hax].tick_params(axis='x', labelrotation=-45)
        if hax == 0:
            #invisible right splines
            ax[0][hax].spines.right.set_visible(False)
            ax[1][hax].spines.right.set_visible(False)
        elif hax == chlen-1:
            #invisible left splines
            ax[0][hax].spines.left.set_visible(False)
            ax[1][hax].spines.left.set_visible(False)
            for tick in ax[0][hax].yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
            for tick in ax[1][hax].yaxis.get_majorticklines():
                tick.set_visible(False)
            for tick in ax[1][hax].yaxis.get_minorticklines():
                tick.set_visible(False)
        else:
            #left and right invisible
            ax[0][hax].spines.left.set_visible(False)
            ax[0][hax].spines.right.set_visible(False)
            ax[1][hax].spines.left.set_visible(False)
            ax[1][hax].spines.right.set_visible(False)
            for tick in ax[0][hax].yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
            for tick in ax[1][hax].yaxis.get_majorticklines():
                tick.set_visible(False)
            for tick in ax[1][hax].yaxis.get_minorticklines():
                tick.set_visible(False)
    plt.suptitle(distkey)
    plt.show()
    fig.clf()
    plt.close()
    gc.collect()
