import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from pyteomics import mzml
from time import time
import pandas as pd
import gc
import concurrent.futures
from collections import Counter, defaultdict
import networkx
from networkx.algorithms.components.connected import connected_components
from scipy import sparse, integrate, spatial
from pandas.api.types import CategoricalDtype
#from sklearn.neighbors import NearestNeighbors
import itertools
import pickle
import sys
import os
import warnings
warnings.filterwarnings("error")
np.set_printoptions(suppress=True)
plt.rcParams["figure.dpi"] = 300

mzmlfile = '/store/flowcharacterizations/round3/DDAs/mzMLs/200901_fR_400.mzML'
isotopefile = '/home/sfo/data/proteomics/fastas/isotope-arrays/human-isotopes-6-50_ss500.pickle'

plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.facecolor'] = 'gray'
plt.rcParams['figure.facecolor'] = 'gray'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['ytick.labelcolor'] = 'white'
plt.rcParams['xtick.labelcolor'] = 'white'

#only thing left to do for peak finding/assessing is add the boundary control for coordwindow, and to subtract the necessary amount to prevent a peak being split across 2 windows

#
#Peak finding parameters
#


#it would be good to summarize al of these parameters with a funnel diagram. Whatever parameter is at work first, affects all of the later output. If something earlier is so stringent that it filters all later results to a greater degree, then it would be smaller on the funnel diagram, and in its linear position of whenever it happens in the code.
peakfill = 0.5 #percentage of datapoints, within a considered peak range, have to be > 0, used to discriminate real from false positive peaks
nneighbors = 5 #number of data points included in the nearest neighbor calculation for peak finding, any group of indices being the center of nneighbors + 1 incoming nearest neighbor interactions is considered for peak-finding
#scanbuffer = 5 #min number of scan indices allowed between points to be considered as part of the same peak or group of peaks.
#minpoints = 7 #minimum number of data points within a selected nearest neighbor interaction group to be considered for peak finding
chrommindist = 8 #number of datapoints required between 2 peak apexes on the chromatographic axis
massmindist = 5 #number of datapoints required between 2 peak apexes on the mass axis
#chromboxcarlength = 5 #number of datapoints used in a boxcar noise reduction during the peak parameter finding process on the chromatographic axis
#massboxcarlength = 5 #number of datapoints used in a boxcar noise reduction during the peak parameter finding process on the mass axis
minmeasurements = 5 #minimum number of non-zero datapoints in any collected chromatographic peak
#^how does a bare minimum of 3 look? with peakfill set to 0 or 1 whichever is non-stringent
gridoverlapthreshold = 0.05
nsplits = 10 #if it's not even, the script will make it even - but does it even need to be even anymore?

if nsplits % 2:
    nsplits += 1


#
#Notes
#

#Main upgrades:
#Finding peaks maxes independent of intensity
#Finding left/right boundaries independent of distance


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

#need to build in a centroiding process for profile data, nothing seems to centroid my example profile data at all (wtf guys?), including msconvert
#or a workaround, 3d peak volumes might be neat? - nah, volume is preserved as area in centroiding, try surface area maybe
def scanfunc(scan):
    et = pd.DataFrame(scan['intensity array'], columns=['intensity'])
    et.loc[:,'m/z'] = scan['m/z array']
    try:
        et.loc[:,'index'] = scan['index']
    except ValueError:
        return []
    et.loc[:,'ms level'] = scan['ms level']
    et.loc[:,'time (min)'] = scan['scanList']['scan'][0]['scan start time'].real
    return [et]

#def fileread(f):
#    t = mp.Manager().list()
#    msrun = pymzml.run.Reader(f)
#    pool = mp.Pool()
#    for scan in msrun:
#        pool.apply_async(scanread(scan, t))
#    pool.close()
#    pool.join()
#    return list(t)

def boolcount(b, counts=False):
    bc = np.where(np.diff(b, prepend=True))[0]
    bc = np.append(bc, len(b))
    if counts:
        bc = np.diff(bc, prepend=0)
    if bc.size % 2:
        bc = np.append(bc, 0)
    return bc.reshape(bc.size//2,2)

def distance_placement(pile, sd):
    ed = []
    cn = 0
    for row in pile:
        sr = [0, 0]
        for nr, cv in enumerate(row):
            sr[nr] += sd[cn:cn+cv].sum()
            cn += cv
        ed.append(sr)
    return np.asarray(ed)

#from https://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements
def generic_meta_overlap(l):
    graph = networkx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        graph.add_nodes_from(part)
        # it also implies a number of edges:
        graph.add_edges_from(to_edges(part))
    return [sorted(i) for i in connected_components(graph)]

def to_edges(l):
    """
        treat `l` as a Graph and returns it's edges
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current

def first_where(check, axis):
    acheck = check.any(axis=axis)
    for n, a in enumerate(acheck):
        if a:
            v = n
            return v

#the literal bane of all this methodology is that this expandeds in a rectangular manner. It would be nice, as well as possible, to expand in a circular manner around a radius. But then modular expansion isn't possible. How might I expand both in a blob-like manner and a shape-consistent manner? The rectangles might miss a data point that's close but diagonal.
#Maybe instead of doing the checks in 1 direction at a time, I could do them all at once?
#finding a network approach that uses the distances from the nearest neighbors would prevent you from needing to do this, implying scanbuffer could be a distance cutoff in node creation for a directional graph
def ring_expansion(window, rings, distinds, newinds, scanbuffer):
    expandedrings = set()
    for r in rings:
        ringinds = newinds[distinds[r]]
        rmins = ringinds.min(axis=(0,1))
        rmaxes = ringinds.max(axis=(0,1))

        topind = rmins[0] - scanbuffer if rmins[0] > scanbuffer else 0
        topbase = rmins[0]

        bottomind = rmaxes[0] + scanbuffer + 1
        bottombase = rmaxes[0] + 1

        leftind = rmins[1] - scanbuffer if rmins[1] > scanbuffer else 0
        leftbase = rmins[1]

        rightind = rmaxes[1] + scanbuffer + 1
        rightbase = rmaxes[1] + 1

        indlist = [topind, bottomind, leftind, rightind]

        #expanding the area around the initially found peak to find all adjacent datapoints within scanbuffer range. If all 0s are found, then that's all folks!
        while True:
            #checking top
            while topind > 0:
                topcheck = window[topind:topbase,leftbase:rightbase]
                if not topcheck.any():
                    break
                #subtractor = scanbuffer - np.argwhere(topcheck.any(axis=1))[0][0]
                subtractor = scanbuffer - first_where(topcheck, 1)
                topind -= subtractor
                topbase -= subtractor
            if topind < 0:
                topind = 0

            #checking bottom
            while bottomind < len(window):
                bottomcheck = window[bottombase-1:bottomind,leftbase:rightbase]
                if not bottomcheck.any():
                    break
                #adder = np.argwhere(bottomcheck.any(axis=1))[0][0] + 1
                adder = first_where(bottomcheck, 1) + 1
                bottomind += adder
                bottombase += adder
            if bottomind > len(window) - 1:
                bottomind = len(window) - 1

            #checking left
            while leftind > 0:
                leftcheck = window[topbase:bottombase,leftind:leftbase]
                if not leftcheck.any():
                    break
                #subtractor = scanbuffer - np.argwhere(leftcheck.any(axis=0))[0][0]
                subtractor = scanbuffer - first_where(leftcheck, 0)
                leftind -= subtractor
                leftbase -= subtractor
            if leftind < 0:
                leftind = 0

            #checking right
            while rightind < window.shape[1]:
                rightcheck = window[topbase:bottombase,rightbase-1:rightind]
                if not rightcheck.any():
                    break
                #adder = np.argwhere(rightcheck.any(axis=0))[0][0] + 1
                adder = first_where(rightcheck, 0) + 1
                rightind += adder
                rightbase += adder
            if rightind > window.shape[1] - 1:
                rightind = window.shape[1] - 1

            #newindlist = [topind, bottomind, leftind, rightind] #adds space after
            #newindlist = [topbase-1, bottombase, leftbase-1, rightbase]
            newindlist = [topbase-1 if topbase > 0 else topbase, bottombase, leftbase-1 if leftbase > 0 else leftbase, rightbase] #a left index came out as -1...
            if newindlist == indlist:
                break
            indlist = newindlist.copy()
        #it would be nice to plot the different stages of the refinement process as different color rectangles
        expandedrings.add(tuple(indlist))
    return expandedrings

#def ring_definition(rings, newinds, vertmax, hormax):
#    coordlist = set()
#    for r in rings:
#        ringinds = newinds[r]
#        ringmins = ringinds.min(axis=0)
#        ringmaxes = ringinds.max(axis=0)
#        top = ringmins[0] - 1#-1 so the first index can have a flanking 0
#        bottom = ringmaxes[0] + 2 #+2 so that slicing index carries the last point, and so that you slice one point after that to have a flanking 0
#        left = ringmins[1] - 1
#        right = ringmaxes[1] + 2
#        if top < 0:
#            top = 0
#        if left < 0:
#            left = 0
#        if bottom > vertmax:
#            bottom = vertmax
#        if right > hormax:
#            right = hormax
#        ringcoords = top, bottom, left, right
#        coordlist.add(ringcoords)
#    return coordlist
    
#def old_minpoint_reduction(array, mindist):
#    #this approach fails when the maxes are at the absolute beginning or end of an array
#    extramaxes = set()
#    mask = np.repeat(False, array.size)
#    while True:
#        narray = array[~mask]
#
#        forwardmaxcheck = np.append(narray[:-1] > narray[1:], False)
#        backwardmaxcheck = np.append(False, narray[1:] > narray[:-1])
#        #backwardmaxcheck = np.append(forwardmaxcheck[0], narray[1:] > narray[:-1])
#        #forwardmaxcheck[-1] = backwardmaxcheck[-1]
#
#        forwardmincheck = np.append(narray[:-1] < narray[1:], False)
#        backwardmincheck = np.append(False, narray[1:] < narray[:-1])
#        #backwardmincheck = np.append(forwardmincheck[0], narray[1:] < narray[:-1])
#        #forwardmincheck[-1] = backwardmincheck[-1]
#
#        newmask = np.logical_and(forwardmincheck, backwardmincheck)
#        mins = np.where(newmask)[0]
#        maxes = np.where(np.logical_and(forwardmaxcheck, backwardmaxcheck))[0]
#        extremadistances = (np.abs(maxes - mins.reshape(-1,1)) < mindist)
#        
#        maxestoholdonto = maxes[~extremadistances.any(axis=0)]
#        if maxestoholdonto.size > 0:
#            maxestoholdonto = (maxestoholdonto + mask.cumsum()[~mask][maxestoholdonto]).tolist()
#            extramaxes.update(maxestoholdonto)
#        
#        adjacentextremas = extremadistances.any()
#        if adjacentextremas:
#            maskinds = np.argwhere(~mask)[np.argwhere(newmask)].flatten()
#            mask[maskinds] = True
#        else:
#            break
#    
#    if not maxes.size:
#        maxes = narray.argmax() #this seems like an easier way to allow for maxes at the first or last point?
#    
#    fmaxes = maxes + mask.cumsum()[~mask][maxes]
#    fmaxes = np.unique(np.append(fmaxes, list(extramaxes))).astype(int)
#    return fmaxes

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
        maxes = narray.argmax() #this seems like an easier way to allow for maxes at the first or last point, in case nothing is found (this case being why)
    
    fmaxes = maxes + mask.cumsum()[~mask][maxes]
    fmaxes = np.unique(np.append(fmaxes, list(extramaxes))).astype(int)
    return fmaxes

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

#def boxcar_mask(narray, boxcarlength):
#    scans = np.arange(len(narray))
#    indvector = np.linspace(scans-boxcarlength, scans+boxcarlength, num=boxcarlength*2+1, axis=1).astype(int)
#    indvector[indvector < 0] = 0
#    indvector[indvector >= len(narray) - 1] = len(narray) - 1
#    filtermeans = narray[indvector].mean(axis=1)
#    minfiltermeans = filtermeans[indvector].min(axis=1) #applying the lowest mean that each individual point is involved with
#    filterboolmap = narray < minfiltermeans
#    filteredinds = scans[filterboolmap]
#    noisemask = np.repeat(True, len(narray))
#    noisemask[filteredinds] = False
#    noisemask[0] = True
#    noisemask[-1] = True
#    return noisemask

def boxcar_mean_replacement(array, boxcarlength):
    narray = array.copy()
    scans = np.arange(len(array))
    indvector = np.linspace(scans-boxcarlength, scans+boxcarlength, num=boxcarlength*2+1, axis=1).astype(int)
    indvector[indvector < 0] = 0
    indvector[indvector >= len(array) - 1] = len(array) - 1
    filtermeans = array[indvector].mean(axis=1)
    #minfiltermeans = filtermeans[indvector].min(axis=1) #applying the lowest mean that each individual point is involved with
    #filterboolmap = narray <= minfiltermeans
    filterboolmap = narray <= filtermeans #this seems to work better than when only replacing things < minfiltermeans. I actually have no idea why.
    narray[filterboolmap] = filtermeans[filterboolmap] #this needs to be applied selectively or else the tops of the peaks can even-out when the boxcarlength considered is wider than the peak - which prevents any maxes from being found
    return narray

#def old_axis_peaks(array, boxcarlength, mindist):
#    noisemask = boxcar_mask(array, boxcarlength)
#    boxcararray = array[noisemask]
#    maxes = minpoint_reduction(boxcararray, mindist)
#    peakparameters = boundary_finding(maxes, boxcararray)
#    
#    peakparameters = peakparameters + (~noisemask).cumsum()[noisemask][peakparameters]
#    peakparameters = peakparameters.tolist()
#    
#    #tightening up just in case because noisemask can still mask 0s that may be more optimal
#    for n in range(len(peakparameters)):
#        l, m, r = peakparameters[n]
#        
#        ltrimdiff = len(array[l:m]) - len(np.trim_zeros(array[l:m], trim='f'))
#        if ltrimdiff > 1:
#            l = l + ltrimdiff - 1
#            peakparameters[n][0] = l
#        
#        rtrimdiff = len(array[m:r+1]) - len(np.trim_zeros(array[m:r+1], trim='b'))
#        if rtrimdiff > 1:
#            r = r - rtrimdiff + 1
#            peakparameters[n][2] = r
#        
#        m = l + array[l:r+1].argmax()
#        peakparameters[n][1] = m
#    
#    #peakparameters = np.asarray(peakparameters)
#    return peakparameters, noisemask

#def axis_peaks(array, boxcarlength, mindist):
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

def max_finding(array):
    forwardmaxcheck = np.append(array[:-1] > array[1:], False)
    #backwardmaxcheck = np.append(False, array[1:] > array[:-1])
    backwardmaxcheck = np.append(forwardmaxcheck[0], array[1:] > array[:-1])
    forwardmaxcheck[-1] = backwardmaxcheck[-1]
    maxes = np.where(forwardmaxcheck & backwardmaxcheck)[0]
    return maxes

def min_finding(array):
    forwardmincheck = np.append(array[:-1] < array[1:], False)
    #backwardmincheck = np.append(False, array[1:] < array[:-1])
    backwardmincheck = np.append(forwardmincheck[0], array[1:] < array[:-1])
    forwardmincheck[-1] = backwardmincheck[-1]
    mins = np.where(forwardmincheck & backwardmincheck)[0]
    return mins

def intensity_crossing(array, point):
    return np.where(np.logical_and(array[:-1] < point, array[1:] > point))[0] + 1

#this guy was helpful: https://stackoverflow.com/questions/69089508/vectorizing-a-rectangle-overlap-determination-in-numpy/69090650#69090650
def rectangle_overlap(recs):
    tops, bottoms, lefts, rights = recs.transpose()
    c1 = lefts < rights[:,None]
    c2 = rights > lefts[:,None]
    c3 = tops < bottoms[:,None]
    c4 = bottoms > tops[:,None]
    overlaps = np.argwhere(c1 & c2 & c3 & c4)
    return overlaps

def coord_rectangle_overlap(rec, coords):
    tops, bottoms, lefts, rights = coords.transpose()
    c1 = rec[2] < rights
    c2 = rec[3] > lefts
    c3 = rec[0] < bottoms
    c4 = rec[1] > tops
    overlaps = np.argwhere(c1 & c2 & c3 & c4)
    return coords[overlaps.flatten()]
#solving for the right-hand boundary in the integration of a linear equation
def integration_limit_solving(b, m, c, a):
    return ((-1*b) + np.sqrt((b**2) - (4*(0.5*m)*(-0.5*m*(c**2) - (b*c) - a)))) / m

#Current work:
#going to make both the lowest baseline subtraction function, and the linear baseline area subtraction.

def integrated_baseline_reduction(m1, m2, b1, b2, c, area):
    #m1/b1 are slope/intercept of signal
    #m2/b2 are slope/intercept of baseline
    #c is left-hand x-value, solving for right-hand x-value
    b12 = b1 - b2
    b21 = b2 - b1
    m12 = m1 - m2
    m21 = m2 - m1
    return (-1*b12) + (np.sqrt((b12**2) - (4*(0.5*m12)*((0.5*m21)*(c**2) + (b21*c) - area))) / m12)

def negligible_difference(a):
    while (np.diff(a) == 0).any():
        a[1:][np.diff(a) == 0] += 0.00000001
    return a

#def area_baseline_reduced_split(peak, xvals, nsplits):
#    workingpeak = negligible_difference(peak)
#    
#    baseline = [workingpeak[0], workingpeak[-1]]
#    baselinexvals = [xvals[0], xvals[1]]
#    baselineslope = np.diff(baseline) / np.diff(baselinevals)
#    baselineintercept = baseline[0] - (baselineslope * baselinexvals[0])
#    baselineyvals = xvals * baselineslope + baselineintercept
#    baselinecumarea = integrate.cumtrapz(baselineyvals, xvals, initial=0)
#    
#    peakslopes = np.diff(workingpeak) / np.diff(xvals)
#    peakintercepts = workingpeak[:-1] - (peakslopes * xvals[:-1])
#    peakcumarea = integrate.cumtrapz(workingpeak, xvals, initial=0) - baselinecumarea
#    peakarea = peakcumarea[-1]
#    
#    maxind = workingpeak.argmax()
#    cumdiffs = np.diff(peakcumarea, prepend=0)
#    
#    leftcdiffs = cumdiffs[1:maxind]
#    rightcdiffs = cumdiffs[maxind+1:-1]
#    
#    leftpeakdivs = np.linspace(0, peakcumarea[maxind], nsplits+1)
#    rightpeakdivs = np.linspace(0, peakcumarea[-1] - peakcumarea[maxind], nsplits+1)
#    
#    leftcumsum = leftcdiffs.cumsum()
#    rightcumsum = rightcdiffs.cumsum()
#    
#    leftinds = (leftcumsum < leftpeakdivs[1:-1][:,None]).sum(axis=1) - 1
#    rightinds = (rightcumsum < rightpeakdivs[1:-1][:,None]).sum(axis=1) + maxind - 1
#    
#    leftareas = leftpeakdivs[1:-1] - leftcumsum[leftinds]
#    upperlimit = integrated_baseline_reduction(peakslopes[leftinds], baselineslope, peakintercepts[leftinds], baselineintercept, xvals[leftinds], leftareas)
#    
#    leftsplitxvals = np.hstack((

#Doing this by drawing vertical lines along the horizontal plane. Drawing horizontal lines along a vertical plane would run into impassable challenges on noisy data.
#I don't want to do a skyline-like baseline subtraction, that problem is do-able, but I don't like that method of integration. I would much rather vertically segregate any raised basline after subtracting, directly from peak (below), the lowest baseline value.
#^maybe the other one will be in the works later on, idk
#def area_peak_split(peak, xvals, nsplits):
#    maxind = peak.argmax()
#    baseline = [peak[0], peak[-1]]
#
#    workingpeak = peak - min(baseline)
#    
#    leftspots = workingpeak[:maxind+1]
#    leftxvals = xvals[:maxind+1]
#    
#    rightspots = workingpeak[maxind:]
#    rightxvals = xvals[maxind:]
#    
#    cumarea = integrate.cumtrapz(workingpeak, xvals, initial=0)
#    areadiffs = np.diff(cumarea, prepend=0)
#    
#    slopes = np.diff(workingpeak) / np.diff(xvals)
#    
#    #For the first/last data points because for some reason MATH ITSELF was breaking for the rightsplitxvals when solving for what should have been x=xvals[-1] at y=0. The math inside the square root of the quadratic formula was returning a negative number for only this last number of rightsplitxvals. It really shouldn't be doing that... It may have been some kind of precision error... maybe. In the example that was failing, by reducing the knownarea from 132.100xx to 132, the value inside the square root became positive again. But it just seems ridiculous! Anywho, I know what the answers should be so I've inserted them, defensively. It shouldn't happen anymore.
#    #This function is also split up to depend on left/rightslopes existing so that it can handle peaks where the max is on either the first or last position.
#    
#    leftslopes = slopes[:maxind]
#    if leftslopes.size > 0:
#        leftareadiffs = areadiffs[:maxind+1]
#        leftareacumsum = leftareadiffs.cumsum()
#        leftsolverareas = np.linspace(leftareacumsum[0], leftareacumsum.max(), nsplits+1)
#        leftsolverindices = (leftareacumsum < leftsolverareas[:-1].reshape(-1,1)).sum(axis=1) - 1
#        leftsolverindices[leftsolverindices < 0] = 0
#        leftspotsofinterest = leftspots[leftsolverindices]
#        leftxvalsofinterest = leftxvals[leftsolverindices]
#        leftslopesofinterest = leftslopes[leftsolverindices]
#        leftintercepts = leftspotsofinterest - (leftxvalsofinterest * leftslopesofinterest)
#        leftareastosolve = leftsolverareas[:-1] - leftareacumsum[leftsolverindices]
#        knownareas, intercepts, lefthandxvals, slopes = leftareastosolve[1:], leftintercepts[1:], leftxvalsofinterest[1:], leftslopesofinterest[1:]
#        leftsplitxvals = (leftintercepts - np.sqrt((intercepts**2) + (slopes**2)*(lefthandxvals**2) + 2*slopes*intercepts*lefthandxvals + 2*slopes*knownareas)) / (-1*slopes)
#        #leftsplitxvals = integration_limit_solving(leftareastosolve[1:], leftintercepts[1:], leftxvalsofinterest[1:], leftslopesofinterest[1:])
#        leftsplitxvals = np.append(xvals[0], leftsplitxvals)
#        leftsplityvals = leftsplitxvals * leftslopesofinterest + leftintercepts
#    else:
#        leftsplitxvals = np.repeat(np.nan, nsplits)
#        leftsplityvals = np.repeat(np.nan, nsplits)
#    
#    rightslopes = slopes[maxind:]
#    if rightslopes.size > 0:
#        rightareadiffs = areadiffs[maxind:]
#        rightareacumsum = rightareadiffs.cumsum()
#        rightsolverareas = np.linspace(rightareacumsum[0], rightareacumsum.max(), nsplits+1)
#        rightsolverindices = (rightareacumsum < rightsolverareas[1:].reshape(-1,1)).sum(axis=1) - 1
#        rightsolverindices[rightsolverindices < 0] = 0
#        rightspotsofinterest = rightspots[rightsolverindices]
#        rightxvalsofinterest = rightxvals[rightsolverindices]
#        rightslopesofinterest = rightslopes[rightsolverindices]
#        rightintercepts = rightspotsofinterest - (rightxvalsofinterest * rightslopesofinterest)
#        rightareastosolve = rightsolverareas[1:] - rightareacumsum[rightsolverindices]
#        rightsplitxvals = integration_limit_solving(rightareastosolve[:-1], rightintercepts[:-1], rightxvalsofinterest[:-1], rightslopesofinterest[:-1])
#        rightsplitxvals = np.append(rightsplitxvals, xvals[-1])
#        rightsplityvals = rightsplitxvals * rightslopesofinterest + rightintercepts
#    else:
#        rightsplitxvals = np.repeat(np.nan, nsplits)
#        rightsplityvals = np.repeat(np.nan, nsplits)
#    
#    #to satiate curiosity
#    #plt.plot(xvals, workingpeak, '.-', color='blue')
#    #plt.plot(leftsplitxvals, leftsplityvals, '.', color='red')
#    #plt.plot(rightsplitxvals, rightsplityvals, '.', color='green')
#    #plt.show()
#    
#    return leftsplitxvals.tolist(), leftsplityvals.tolist(), rightsplitxvals.tolist(), rightsplityvals.tolist()

def peak_area_split(peak, xvals, nsplits):
    maxind = peak.argmax()
    
    leftxvals = xvals[:maxind+1]
    rightxvals = xvals[maxind:]
    
    slopes = np.diff(peak) / np.diff(xvals)
    intercepts = peak[:-1] - (xvals[:-1] * slopes)
    
    cumarea = integrate.cumtrapz(peak, xvals, initial=0)
    areadiffs = np.diff(cumarea, prepend=0)
    
    leftslopes = slopes[:maxind+1]
    rightslopes = slopes[maxind:]
    
    if leftslopes.size > 0:
        leftintercepts = intercepts[:maxind+1]
        leftspots = peak[:maxind+1]
        leftxvals = xvals[:maxind+1]
        leftareadiffs = areadiffs[:maxind+1]
        leftcumarea = leftareadiffs.cumsum()
        leftsplits = np.linspace(leftcumarea[0], leftcumarea[-1], nsplits+1)[1:-1]
        leftinds = (leftcumarea < leftsplits.reshape(-1,1)).sum(axis=1) - 1
        leftslopesofinterest = leftslopes[leftinds]
        leftinterceptsofinterest = leftintercepts[leftinds]
        leftxvalsofinterest = leftxvals[leftinds]
        leftareasofinterest = leftsplits - leftcumarea[leftinds]
        leftsplitxvals = integration_limit_solving(leftinterceptsofinterest, leftslopesofinterest, leftxvalsofinterest, leftareasofinterest)
        leftsplityvals = leftsplitxvals * leftslopesofinterest + leftinterceptsofinterest
        leftsplitxvals = np.hstack((leftxvals[0], leftsplitxvals, leftxvals[-1]))
        leftsplityvals = np.hstack((leftspots[0], leftsplityvals, leftspots[-1]))
    else:
        leftsplitxvals = np.repeat(np.nan, nsplits+1)
        leftsplityvals = np.repeat(np.nan, nsplits+1)
    
    if rightslopes.size > 0:
        rightintercepts = intercepts[maxind:]
        rightspots = peak[maxind:]
        rightxvals = xvals[maxind:]
        rightareadiffs = areadiffs[maxind+1:]
        rightcumarea = np.append(0, rightareadiffs.cumsum())
        rightsplits = np.linspace(rightcumarea[0], rightcumarea[-1], nsplits+1)[1:-1]
        rightinds = (rightcumarea < rightsplits.reshape(-1,1)).sum(axis=1) - 1
        rightslopesofinterest = rightslopes[rightinds]
        rightinterceptsofinterest = rightintercepts[rightinds]
        rightxvalsofinterest = rightxvals[rightinds]
        rightareasofinterest = rightsplits - rightcumarea[rightinds]
        rightsplitxvals = integration_limit_solving(rightinterceptsofinterest, rightslopesofinterest, rightxvalsofinterest, rightareasofinterest)
        rightsplityvals = rightsplitxvals * rightslopesofinterest + rightinterceptsofinterest
        rightsplitxvals = np.hstack((rightxvals[0], rightsplitxvals, rightxvals[-1]))
        rightsplityvals = np.hstack((rightspots[0], rightsplityvals, rightspots[-1]))
    else:
        rightsplitxvals = np.repeat(np.nan, nsplits+1)
        rightsplityvals = np.repeat(np.nan, nsplits+1)
    
    return leftsplitxvals.tolist(), leftsplityvals.tolist(), rightsplitxvals.tolist(), rightsplityvals.tolist()

def shared_peak_processing(peak, xvals, nsplits):
    npoints = len(peak)
    nzeros = (peak == 0).sum()
    nnonzeros = npoints - nzeros
    
    #these give the same answer, only need one really
    nmaxes = len(max_finding(peak))
    #nmins = len(min_finding(peak))
    maxesperpoints = nmaxes / npoints
    maxespernonzeropoints = nmaxes / nnonzeros
    
    maxintensityloc = peak.argmax()
    maxintensity = peak[maxintensityloc]
    nleftpoints = maxintensityloc - 1
    nrightpoints = npoints - nleftpoints - 1
    maxind = xvals[maxintensityloc]

    percentofmax = peak / maxintensity
    meanpercentofmax = percentofmax.mean()
    medianpercentofmax = np.median(percentofmax)
    
    scaleleft = xvals[maxintensityloc] - xvals[0]
    scaleright = xvals[-1] - xvals[maxintensityloc]
    
    leftbaseline = peak[0]
    rightbaseline = peak[-1]
    
    leftprompoint = leftbaseline
    pc = 0
    while leftprompoint == 0:
        pc += 1
        leftprompoint = peak[pc]
    rightprompoint = rightbaseline
    pc = -1
    while rightprompoint == 0:
        pc -=1
        rightprompoint = peak[pc]
    
    leftprominence = leftprompoint / maxintensity
    rightprominence = rightprompoint / maxintensity
    
    #area units = intensity * minutes
    #this baseline subtraction IS a skyline-like baseline subtraction, unlike what's currently in area_peak_splits
    baseline = [leftbaseline, rightbaseline]
    baselinescale = [xvals[0], xvals[-1]]
    trapzbaseline = integrate.trapz(baseline, baselinescale)
    simpsbaseline = integrate.simps(baseline, baselinescale)
    originaltrapzarea = integrate.trapz(peak, xvals)
    originalsimpsarea = integrate.simps(peak, xvals)
    baselinesubtractedtrapzarea = originaltrapzarea - trapzbaseline
    baselinesubtractedsimpsarea = originalsimpsarea - simpsbaseline
    
    precision = max([str(i)[::-1].find('.') for i in xvals]) #testing on just one index can be faulty!
    while True:
        try:
            leftareasplitxvals, leftareasplityvals, rightareasplitxvals, rightareasplityvals = peak_area_split(peak, xvals.round(precision), nsplits)
            break
        except RuntimeWarning: #floating precision is off, multiple problems can arise within the quadratic used to solve for area upper limits, all arising from... decimals - annoyingly enough
            precision -=1
            if precision < 1:
                leftareasplitxvals, leftareasplityvals, rightareasplitxvals, rightareasplityvals = np.repeat(np.nan, nsplits+1), np.repeat(np.nan, nsplits+1), np.repeat(np.nan, nsplits+1), np.repeat(np.nan, nsplits+1)
                break

    #I'll add the distance-based splits later I suppose.
    return (
            npoints,                        #number of data points
            nzeros,                         #number of zeros
            nmaxes,                         #number of points surrounded by 2 lower points
            maxesperpoints,                 #nmaxes / data points
            maxespernonzeropoints,          #nmaxes / non=zero data points
            maxintensityloc,                #index of the highest point
            maxintensity,                   #highest value
            nleftpoints,                    #number of data points left of maxintensity
            nrightpoints,                   #number of data points right of maxintensity
            maxind,                         #x-axis associated value at maxintensity
            meanpercentofmax,               #average of (%all data) / max
            medianpercentofmax,             #median of (%all data) / max
            scaleleft,                      #x-axis associated length left of maxintensity
            scaleright,                     #x-axis associated length right of maxintensity
            leftprominence,                 #prominence of outer-most left non-zero point
            rightprominence,                #prominence of outer-most right non-zero point
            originaltrapzarea,              #pure peak AUC, using trapezoidal rule
            originalsimpsarea,              #pure peak AUC, using simpson's rule
            baselinesubtractedtrapzarea,    #trapezoidal area - (linear area under [first,last] values)
            baselinesubtractedsimpsarea,    #simpson's area - (linear area under [first,last] values)
            precision,                      #number of decimals used in the x-axis associated values for the area split
            leftareasplitxvals,             #x-values left of max splits
            leftareasplityvals,             #y-values left of max-splits
            rightareasplitxvals,            #x-values right of max-splits
            rightareasplityvals             #y-values right of max-splits
            )

def peak_processing(peakgrid, coords, coordwindow, mp, cp, masses, scanarray, timearray, nsplits):
    #
    #Shared section
    #
    
    t, b, l, r = coords
    mp = [i + t for i in mp]
    cp = [i + l for i in cp]
    
    peakchrom = peakgrid.sum(axis=0)
    peakmass = peakgrid.sum(axis=1)

    timeinds = timearray[cp[0]:cp[2]]
    massinds = masses[mp[0]:mp[2]]
    
    nchrompoints, nchromzeros, nchrommaxes, chrommaxesperpoints, chrommaxespernonzeropoints, chrommaxintensityloc, chrommaxintensity, nchromleftpoints, nchromrightpoints, retentiontime, chrommeanpercentofmax, chrommedianpercentofmax, timeleft, timeright, chromleftprominence, chromrightprominence, chromoriginaltrapzarea, chromoriginalsimpsarea, chrombaselinesubtractedtrapzarea, chrombaselinesubtractedsimpsarea, chromxprecision, chromleftareasplitxvals, chromleftareasplityvals, chromrightareasplitxvals, chromrightareasplityvals = shared_peak_processing(peakchrom, timeinds, nsplits)
    
    nmasspoints, nmasszeros, nmassmaxes, massmaxesperpoints, massmaxespernonzeropoints, massmaxintensityloc, massmaxintensity, nmassleftpoints, nmassrightpoints, maxmass, massmeanpercentofmax, massmedianpercentofmax, massesleft, massesright, massleftprominence, massrightprominence, massoriginaltrapzarea, massoriginalsimpsarea, massbaselinesubtractedtrapzarea, massbaselinesubtractedsimpsarea, massxprecision, massleftareasplitxvals, massleftareasplityvals, massrightareasplitxvals, massrightareasplityvals = shared_peak_processing(peakmass, massinds, nsplits)
    
    mp = [i + coordwindow for i in mp]
    
    #
    #Independent features
    #
    
    leftmassscanindex = mp[0]
    rightmassscanindex = mp[2]
    leftchromscanindex = cp[0]
    rightchromscanindex = cp[2]
    
    peakinds = np.argwhere(peakgrid > 0)
    peakdata = peakgrid[peakinds[:,0], peakinds[:,1]]
    
    #
    #Chrom section
    #
    
    scaninds = scanarray[cp[0]:cp[2]]
    
    totalintensity = peakchrom.sum()
    meanintensity = peakchrom.mean()
    medianintensity = np.median(peakchrom)
    
    nmeancrosses = len(intensity_crossing(peakchrom, meanintensity))
    nmediancrosses = len(intensity_crossing(peakchrom, medianintensity))
    
    #the idea of this got tricky, because changing the time-scale, ie minutes -> seconds -> ms or whatever would give you greater areas, but there isn't a pre-defined logic to choosing one. The sum of the intensities equals the area when doing numeric integration with a distance of 1 at each point. I came to the conclusion to use the scaninds for the x-axis in this determination because you can consider the area to be changing at a rate of scan numbers. I don't currently have great confidence in the normalizedtime method below, but I want to see how it plays out. Both of these factors could play a role in quant normalization later.
    normalizedtime = np.diff(timeinds, prepend=timeinds[0]) / np.diff(timeinds).min()
    samplingratebyscan = peakchrom.sum() / np.trapezoid(peakchrom, scaninds)
    samplingratebytime = peakchrom.sum() / np.trapezoid(peakchrom, normalizedtime.cumsum())
    
    scansleft = scaninds[chrommaxintensityloc] - scaninds[0]
    scansright = scaninds[-1] - scaninds[chrommaxintensityloc]
    retentionscan = scaninds[chrommaxintensityloc]
    
    ndatapoints = len(peakinds)
    
    #
    #Mass section
    #
    #It's really important to note that I include an index for every mass found throughout the entire file. Masses that aren't picked up in a single peak can still be interpreted as zeros in these mass peaks. Might change later, idk.
    
    massesbydata = massinds[peakinds[:,0]]
    
    weightedmeanmass = np.average(massesbydata, weights=peakdata)
    geometricmeanmass = masses.mean()
    meanmass = massesbydata.mean()
    massmax = masses.max() #not to be confused with maxmass
    massmin = masses.min()
    #massatmaxchromloc = peakinds[np.where(peakinds[:,1] == chrommaxintensityloc)[0][0], 0] #why was this here again?
    massatmaxchrom = massinds[peakgrid[:,chrommaxintensityloc] > 0]
    if len(massatmaxchrom) > 1:
        massatmaxchrom = str(massatmaxchrom.tolist())[1:-1]
    else:
        massatmaxchrom = massatmaxchrom.max()
    nindividualmasses = (peakmass > 0).sum()
    
    endlist = [
            leftmassscanindex,                          #first scan index of mass peak
            rightmassscanindex,                         #last scan index of mass peak
            leftchromscanindex,                         #first scan index of chromatographic peak
            rightchromscanindex,                        #last scan index of chromatographic peak
            totalintensity,                             #total intensity
            meanintensity,                              #mean intensity
            medianintensity,                            #median intensity
            nmeancrosses,                               #number of times the chromatographic peak crosses the mean value
            nmediancrosses,                             #number of times the chromatographic peak crosses the median value
            samplingratebyscan,                         #estimated sampling rate based on intensity, scans indexes, and area
            samplingratebytime,                         #estimated sampling rate basd on intensity, time, and area
            scansleft,                                  #number of scans left of max
            scansright,                                 #number of scans right of max
            retentionscan,                                    #scan index of max value                    #does this need to have l added to it? And the above 2 values?
            ndatapoints,                                #total number of data points
            weightedmeanmass,                           #weighted mean of mass by intensity at each data point
            geometricmeanmass,                          #mean of the mass-range where each mass is represented once
            meanmass,                                   #mean of masses of each data point
            massmax,                                    #the highest mass of the peak
            massmin,                                    #the lowest mass of the peak
            massatmaxchrom,                             #mass value(s) of the highest chromatographic point
            nindividualmasses,                          #length of the mass range
            nchrompoints,                               #number of data points along the chromatographic axis
            nchromzeros,                                #number of zeros along the chromatographic axis
            nchrommaxes,                                #number of chromatographic data points surrounded by two lower values
            chrommaxesperpoints,                        #nchrommaxes / ndatapoints
            chrommaxespernonzeropoints,                 #nchrommaxes / non-zero data points
            chrommaxintensityloc,                       #index of highest chromatographic point
            chrommaxintensity,                          #value of highest chromatographic point
            nchromleftpoints,                           #number of points left of chromatographic max
            nchromrightpoints,                          #number of points right of chromatographic max
            retentiontime,                              #time at chromatographic max
            chrommeanpercentofmax,                      #average (%chromatographic data / max)
            chrommedianpercentofmax,                    #median (%chromatographic data / max)
            timeleft,                                   #total time left of chromatographic max
            timeright,                                  #total right right of chromatographic max
            chromleftprominence,                        #prominence of left-most non-zero chromatographic point
            chromrightprominence,                       #prominence of right-most non-zero chromatographic point
            chromoriginaltrapzarea,                     #pure AUC using trapezoidal rule
            chromoriginalsimpsarea,                     #pure AUC using simpson's rule
            chrombaselinesubtractedtrapzarea,           #baseline-subtracted trapezoidal area
            chrombaselinesubtractedsimpsarea,           #baseline-subtracted simpson's area
            chromxprecision,                            #number of decial points used for the x-axis associated value when making area splits
            nmasspoints,                                #number of data points along the mass axis
            nmasszeros,                                 #number of zeros along the mass axis
            nmassmaxes,                                 #number of mass data points surrounded by two lower values
            massmaxesperpoints,                         #nmassmaxes / ndatapoints
            massmaxespernonzeropoints,                  #nmassmaxes / non-zero data points
            massmaxintensityloc,                        #index of highest mass point
            massmaxintensity,                           #value of highest mass point
            nmassleftpoints,                            #number of points left of mass max
            nmassrightpoints,                           #number of points right of mass max
            maxmass,                                    #mass at max intensity of mass peak
            massmeanpercentofmax,                       #average (%mass data / max)
            massmedianpercentofmax,                     #median (%mass data / max)
            massesleft,                                 #total time left of mass max
            massesright,                                #total right right of mass max
            massleftprominence,                         #prominence of left-most non-zero mass point
            massrightprominence,                        #prominence of right-most non-zero mass point
            massoriginaltrapzarea,                      #pure AUC using trapezoidal rule
            massoriginalsimpsarea,                      #pure AUC using simpson's rule
            massbaselinesubtractedtrapzarea,            #baseline-subtracted trapezoidal area
            massbaselinesubtractedsimpsarea,            #baseline-subtracted simpson's area
            massxprecision,                             #number of decial points used for the x-axis associated value when making area splits
            ]
    
    endlist.extend(massleftareasplitxvals)              #x-values of left area splits along mass axis
    endlist.extend(massleftareasplityvals)              #y-values of left area splits along mass axis
    endlist.extend(massrightareasplitxvals)             #x-values of right area splits along mass axis
    endlist.extend(massrightareasplityvals)             #y-values of right area splits along mass axis
    endlist.extend(chromleftareasplitxvals)             #x-values of left area splits along chromatographic axis
    endlist.extend(chromleftareasplityvals)             #y-values of left area splits along chromatographic axis
    endlist.extend(chromrightareasplitxvals)            #x-values of right area splits along chromatographic axis
    endlist.extend(chromrightareasplityvals)            #y-values of right area splits along chromatographic axis
    return tuple(endlist)

def max_outer_peak(chrom):
    forwardcheck = np.append(chrom[:-1] < chrom[1:], False)
    backwardcheck = np.append(False, chrom[1:] < chrom[:-1])
    mask = np.logical_and(forwardcheck, backwardcheck)
    if mask.any():
        while True:
            #this can probably be written as a function without the if-statement, starting with a 'mask = True', then even if the whole mask is false, the loop stops and the mask is returned as unnecessary
            maskedchrom = chrom[~mask]
            forwardcheck = np.append(maskedchrom[:-1] < maskedchrom[1:], False)
            backwardcheck = np.append(False, maskedchrom[1:] < maskedchrom[:-1])
            newmask = np.logical_and(forwardcheck, backwardcheck)

            if newmask.any():
                maskinds = np.argwhere(~mask)[np.argwhere(newmask)].flatten()
                mask[maskinds] = True

                #to satiate curiosity
                #plt.plot(scaninds, chrom, '.-', color='orange')
                #plt.plot(scaninds[~mask], chrom[~mask], '.-', color='blue')
                #plt.show()
            else:
                break
    return ~mask

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

mt = time()
ef = []
msrun = mzml.MzML(mzmlfile, dtype=np.float64)
for t in msrun.map(lambda scan: scanfunc(scan)):
    ef.extend(t)

gc.collect()
print(time() - mt, '- File Extracted')


#
#Using MS1 data
#

ef = pd.concat(ef)
scantimes = ef.loc[:,('index', 'time (min)')].drop_duplicates()
scantimes.set_index('index', inplace=True)
scantimes.sort_index(inplace=True)
ef = ef.loc[ef.loc[:,'ms level'] == 1]

masses = np.unique(ef.loc[:,'m/z'])
mdiff = np.diff(masses)
res = masses[1:] / mdiff


plt.plot(res, color='white')
plt.show()

plt.plot(res, '.', color='white')
plt.show()

plt.hist(mdiff, bins=1000, color='white')
plt.yscale('log')
plt.show()








#not all these formats seem to matter for speed, for m/z values though the resolution obviously won't match what's recorded. And it seems to ~half the number of rows in the sparse matrix.
#ef.loc[:,'m/z'] = ef.loc[:,'m/z'].round(4)
#ef.loc[:,'intensity'] = ef.loc[:,'intensity'].astype(int)
#ef.loc[:,'index'] = ef.loc[:,'index'].astype(np.int16)

#
##
##Arranging original data from ef into array format
##
#
#newcol = 'index'
#newrow = 'm/z'
#values = 'intensity'
#
##this doesn't necessarily need to be done en-masse, and certainly fucks with everything once profile data comes into play, these could potentially be made on the fly for masses that fit a window
##this matrix is great in a lot of ways, and enables the nearest neighbor approach, but the downside is that it assumes uniform mass sampling across the array. Some peaks may be spread out across the matrix in an unidentifiable fashion. I'll have to make a correction for this later by going through the original mzML data.
#newcols = CategoricalDtype(sorted(ef.loc[:,newcol].unique()), ordered=True)
#newinds = CategoricalDtype(sorted(ef.loc[:,newrow].unique()), ordered=True)
#col = ef.loc[:,newcol].astype(newcols).cat.codes
#row = ef.loc[:,newrow].astype(newinds).cat.codes
#sm = sparse.csc_matrix((ef.loc[:,values], (row, col)), shape=(newinds.categories.size, newcols.categories.size))
#
##mzindex = newinds.categories.to_numpy().round(8) #rounding to the 8th decimal seems to preserve all mass uniqueness while also allowing for more appropriate precision in calculations
#mzindex = newinds.categories.to_numpy()
#
#scanarray = np.sort(ef.loc[:,'index'].unique())
#timearray = np.sort(ef.loc[:,'time (min)'].unique())
#
#print(time() - mt, 'Arrays assembled') #a sometimes confusing message


