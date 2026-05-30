import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
#from pyteomics import mzml
from time import time
import pandas as pd
import gc
import concurrent.futures
from collections import Counter, defaultdict
#import networkx
#from networkx.algorithms.components.connected import connected_components
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
            chromxprecision,                            #number of decimal points used for the x-axis associated value when making area splits
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
            massxprecision,                             #number of decimal points used for the x-axis associated value when making area splits
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
#not all these formats seem to matter for speed, for m/z values though the resolution obviously won't match what's recorded. And it seems to ~half the number of rows in the sparse matrix.
#ef.loc[:,'m/z'] = ef.loc[:,'m/z'].round(4)
#ef.loc[:,'intensity'] = ef.loc[:,'intensity'].astype(int)
#ef.loc[:,'index'] = ef.loc[:,'index'].astype(np.int16)


#
#Arranging original data from ef into array format
#

newcol = 'index'
newrow = 'm/z'
values = 'intensity'

#this doesn't necessarily need to be done en-masse, and certainly fucks with everything once profile data comes into play, these could potentially be made on the fly for masses that fit a window
#this matrix is great in a lot of ways, and enables the nearest neighbor approach, but the downside is that it assumes uniform mass sampling across the array. Some peaks may be spread out across the matrix in an unidentifiable fashion. I'll have to make a correction for this later by going through the original mzML data.
newcols = CategoricalDtype(sorted(ef.loc[:,newcol].unique()), ordered=True)
newinds = CategoricalDtype(sorted(ef.loc[:,newrow].unique()), ordered=True)
col = ef.loc[:,newcol].astype(newcols).cat.codes
row = ef.loc[:,newrow].astype(newinds).cat.codes
sm = sparse.csc_matrix((ef.loc[:,values], (row, col)), shape=(newinds.categories.size, newcols.categories.size))

#mzindex = newinds.categories.to_numpy().round(8) #rounding to the 8th decimal seems to preserve all mass uniqueness while also allowing for more appropriate precision in calculations
mzindex = newinds.categories.to_numpy()

scanarray = np.sort(ef.loc[:,'index'].unique())
timearray = np.sort(ef.loc[:,'time (min)'].unique())

print(time() - mt, 'Arrays assembled') #a sometimes confusing message


#
#Peak finding in individual mass channels
#

#Having too high a number of neighbors when accounting for noise reduction doesn't seem to reduce much noise.
#Using indices rather than mass and time/scan values here actually has a massive benefit:
#By making the x and y axis into the same type of value, the distances can be treated as equal! Otherwise mass distances would be much smaller than whatever's on the x-axis. I'd need to normalize the two to be in an equal range, which might be more or less what it's like now just with an equal # of points to make everything square. But that might not even be necessary. This seems to work nicely. And although scans are much like indices, there are gaps in the scans, which wouldn't be cool I suppose.
#The shortcoming of this approach is that it doesn't, in any way, incorporate the intensity of a datapoint. Which, depending on your point of view may be a feature and not a bug.


nt = time()
#trying redundancy to see what happens!

#Got less than below! By like 20 lol. 30879 peaks, after determining overlap (which was super slow). Just fix the slight overlaps on the fly and I think it will be good.
#nwindows = 2 #must be even
#totalsize = 20000
#windowsize = totalsize // nwindows
#coordwindow = windowsize

#how many peaks do you find if you switch the KNN distances to be mass and time? You could change the index inds based on mass/time values then do it.
#30903 peaks found
#880s at 10000
#1710s at 50000... idk, whatever
windowsize = 10000 #bigger window sizes are a bit faster when considering the whole file
coordwindow = windowsize
#basefolder = '/home/sfo/data/chromatography/masstraces/'
masschosen = 0
chromchosen = 0
noisecount = 0
itercount = 0
#areacount = 0
endpeaks = set() #will need to get rid of partial overlaps after, choose the larger one
while True:
    #add a time count for each peak next to a uniqueid of sorts, to figure out the max number of peaks found within a window, and to determine how long all those took for each window, what the window is, etc. Would work like a profiling almost.
    #window = sm[coordwindow-windowsize:coordwindow].toarray()
    window = sm[coordwindow-windowsize:coordwindow].toarray()
    #areacount += np.trapezoid(window, timearray).sum()
    if window.size == 0:
        break
    #window = window.copy() #super slow to copy a numpy array, i had no idea, slower than making it a second time. but in this case i don't need to reuse window.
    #masses = mzindex[coordwindow-windowsize:coordwindow].round(8)
    masses = mzindex[coordwindow-windowsize:coordwindow]
    inds = np.argwhere(window > 0)
    
    #need to use a different cutoff mechanism if you want this
    #signalinds = np.stack((masses[inds[:,0]], timearray[inds[:,1]]), axis=1)

    #Random noise reduction
    #nbrs = NearestNeighbors(n_neighbors=2, metric='euclidean', algorithm='auto').fit(inds)
    #dists, distinds = nbrs.kneighbors(inds)
    #911s, (30900, 173)
    
    #scipy version seems ~faster
    nbrs = spatial.KDTree(inds)
    dists, distinds = nbrs.query(inds, k=2)
    
    #tbh this is a sloppy idea imo, but I think it will work out in actuality
    #filtering at the first sorted point that dips below the mean
    dcounts = Counter(dists[:,1].round().astype(int).tolist())
    dkeys = list(dcounts.keys())
    sortorder = np.asarray(dkeys).argsort()
    dvals = np.asarray(list(dcounts.values()))
    dvals[dvals <= dvals.mean()] = 0
    cutind = np.argwhere(dvals[sortorder] <= dvals.mean())[0][0]
    cutkey = sorted(dkeys)[cutind] + 1
    
    noiseindmask = dists[:,1] > cutkey
    noiseinds = inds[noiseindmask]
    
    #distmean = dists[:,1].mean()
    #noiseindmask = dists[:,1] > distmean
    #noiseinds = inds[noiseindmask]
    
    #temp use for plotting
    #owindow = window.copy()
    noisecount += window[noiseinds[:,0], noiseinds[:,1]].sum()
    window[noiseinds[:,0], noiseinds[:,1]] = 0
    
    #newsignalinds = signalinds[~noiseindmask]
    newinds = inds[~noiseindmask]
    #newinds = np.argwhere(window > 0)
    #newsignalinds = np.stack((masses[newinds[:,0]], timearray[newinds[:,1]]), axis=1)
    
    #nbrs = NearestNeighbors(n_neighbors=nneighbors, metric='euclidean', algorithm='auto').fit(newinds)
    #dists, distinds = nbrs.kneighbors(newinds)
    nbrs = spatial.KDTree(newinds)
    dists, distinds = nbrs.query(newinds, k=nneighbors+1)
    
    #effects and distriutions can be visualized:
    #plt.bar(dcounts.keys(), dcounts.values())
    #plt.hlines(dvals.mean(), 0, max(dkeys), color='black', linewidth=0.3)
    #plt.vlines(cutkey, 0, max(dvals), color='black', linewidth=0.3)
    #plt.title('Distribution of Nearest Neighbor Distance')
    #plt.ylabel('Number of Points with Distance $x$')
    #plt.xlabel('Integer Distance from Nearest Neighbor')
    #plt.yscale('log')
    #plt.show()
    #plt.imshow(owindow, cmap='GnBu', vmin=0, vmax=1, aspect='auto')
    #plt.title('Before')
    #plt.ylabel('Mass Index')
    #plt.xlabel('Time Index')
    #plt.show()
    #plt.imshow(window, cmap='GnBu', vmin=0, vmax=1, aspect='auto')
    #plt.title('After')
    #plt.ylabel('Mass Index')
    #plt.xlabel('Time Index')
    #plt.show()
    
    #you can get slopes from intensity / distance
    #or you can ignore the slope and calculate the area of every triangle, then use the area and the sides to calculation the hypotenuse (hypotenuse = 2d distance in this case)
    #   > Using each layer of rings and starting with the highest, you would find the total distance of each interaction within nneighbors of the point that it shows up for. The top of the ring is everything that interacts with the highest point (I wonder how changing nneighbors would affect the resultant process afterwards). The next lowest ring would be everything that interacts with anything that was in the first ring, and so on and so forth. And as you go down the rings, the maximum distance between any 2 members of each ring should increase because they're encompassing both of the outer sides of the peak, they diverge from the center. And I think the total distance, and maybe distance per #interactors for each point could be summed to make a total for the level, or maybe a mean or something.
    
    #finding reciprocating nearest neighborship
    #aka finding number of nearest neighbors that also consider a given indice one of its nearest neighbor
    
    #the old method, it was slightly slower when [barely] tested.
    #distinteractioncount = Counter(distinds[:,1:].flatten().tolist())
    #distinteractioncenters = [k for k, v in distinteractioncount.items() if v > nneighbors]
    #rings = generic_meta_overlap(distinds[distinteractioncenters].tolist())
    #expandedrings = ring_expansion(window, rings, distinds, newinds, scanbuffer)
    
    #alternative to the above using a mean filter instead of a pure nneighbor cutoff. This method was just barely more thorough, in a way that might not matter, when [barely] tested. It was also faster when [barely] tested
    #dmean = dists.mean()
    #dit = zip(dists.tolist(), distinds.tolist())
    #dfinal = []
    #for d, di in dit:
    #    dz = zip(d, di)
    #    keepers = [i for v, i in dz if v < dmean]
    #    dfinal.append(keepers)
    #rings = generic_meta_overlap(dfinal)
    #frings = []
    #for r in rings:
    #    c = 0
    #    for sr in r:
    #        c += (dists[sr] < dmean).sum()
    #    if c > nneighbors:
    #        frings.append(r)
    #expandedrings = ring_expansion(window, frings, distinds, newinds, scanbuffer)
    
    #this wins out in terms of speed, high number of rings preserved (which are the 2 goals this should accomplish). And the rings aren't over-expanded by shitty groups. Either approach pretty much gets you the same result, this is mostly filtering out redundant starting points so that both generic_meta_overlap and ring_expansion takes an order of magnitude less time.
    distsums = dists.sum(axis=1)
    dmean = distsums.mean()
    filterinds = distsums < dmean
    filterdists = distinds[filterinds] #change distinds to dists for true-value based KNN
    rings = generic_meta_overlap(filterdists.tolist())
    scanbuffer = dists[:,1].mean().round().astype(int) #this approach might help later on when incorporating boxcar DDA data because this would, by its own nature, take into account the gaps there might be in sampling each mass range
    #scanbuffer = np.average(dists[:,1], weights=1/dists[:,1]).round().astype(int)
    expandedrings = ring_expansion(window, rings, distinds, newinds, scanbuffer) #scanbuffer at 5
    
    #new approach? below
    #vertmax, hormax = window.shape
    ##scanbuffer = dists.sum(axis=1).mean() / nneighbors
    ##scanbuffer = dists[:,1].mean()
    #scanbuffer = np.average(dists[:,1], weights=1/dists[:,1])
    ##scanbuffer = np.average(dists[:,1], weights=dists[:,1]) #makes everything much slower
    #linkedpoints = nbrs.query_ball_point(newinds, r=scanbuffer).tolist()
    #rings = generic_meta_overlap(linkedpoints)
    #rings = [i for i in rings if len(i) >= nneighbors]
    #expandedrings = ring_definition(rings, newinds, vertmax, hormax)

    #acceptablerings = {tuple((t, b, l, r)) for t, b, l, r in expandedrings if window[t:b,l:r].any(axis=0).sum() / (r - l) >= peakfill} #going to do this twice, here it helps prevent massive groups from merging every element, and/or from tiny groups being a bridge between everything
    #if acceptablerings:
        #rectangleoverlaps = []
        #holder = acceptablerings.copy()
        #for a in acceptablerings:
        #    midlist = []
        #    for h in holder:
        #        overlap = range_overlap(a, h)
        #        if overlap:
        #            midlist.append(h)
        #    rectangleoverlaps.append(midlist)
        #    holder.remove(a)
        #ringgroups = generic_meta_overlap(rectangleoverlaps)
    
    expandedrings = np.array(list(expandedrings))
    overlaps = rectangle_overlap(expandedrings)
    ringgroups = generic_meta_overlap(overlaps)
    
    ringregions = set()
    ringiters = []
    for rg in ringgroups:
        if len(rg) == 1:
            ringregions.add(tuple(expandedrings[rg[0]].tolist()))
        else:
            ringiters.append(rg)
    
    for ri in ringiters:
        recs = expandedrings[ri]
        rmaxes = recs.max(axis=0)
        rmins = recs.min(axis=0)
        fullrec = (rmins[0], rmaxes[1], rmins[2], rmaxes[3])
        ringregions.add(fullrec)
    
    #getting rid of anything too close to the window borders, need to add more accomodations for this later on when adding to the moving window
    #expandedrings = expandedrings[expandedrings[:,0] > scanbuffer]
    #expandedrings = expandedrings[expandedrings[:,1] < len(window) - scanbuffer]
    #^Add for future processing
    
    #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,8))
    ##fs = ''.join((basefolder, str(n), '.png'))
    #for t, b, l, r in expandedrings:
    #    #width = timearray[r-1] - timearray[l]
    #    width = r - l
    #    #height = masses[b-1] - masses[t]
    #    height = b - t
    #    #rect = patches.Rectangle((timearray[l], masses[t]), width, height, linewidth=2, edgecolor='red', facecolor='none')
    #    rect = patches.Rectangle((l, t), width, height, linewidth=2, edgecolor='red', facecolor='none')
    #    ax.add_patch(rect)
    ##ax.plot(timearray[inds[:,1]], masses[inds[:,0]], '.', markersize=0.5, color='crimson', alpha=0.5)
    ##ax.plot(timearray[newinds[:,1]], masses[newinds[:,0]], '.', markersize=0.5, color='indigo', alpha=0.5)
    ##ax.plot(newsignalinds[:,1], newsignalinds[:,0], '.', markersize=0.5, color='indigo', alpha=0.5)
    #ax.plot(newinds[:,1], newinds[:,0], '.', markersize=0.5, color='indigo', alpha=0.5)
    ##ax.plot(inds[:,1], inds[:,0], '.', markersize=0.5, color='crimson', alpha=0.5)
    ##ax.plot(newinds[:,1], newinds[:,0], '.', markersize=0.5, color='indigo', alpha=0.5)
    #plt.title(' - '.join((str(coordwindow - windowsize), str(coordwindow))))
    #plt.xlabel('minutes')
    #plt.ylabel('mass')
    ##fig.savefig(fs, facecolor='white', transparent=False)
    ##plt.close("all")
    ##gc.collect()
    #plt.show()
    
    #collectedpeaks = []
    itercount += len(ringregions)
    for coords in ringregions:
        t, b, l, r = coords
        grid = window[t:b,l:r]
        if grid.any(axis=0).sum() >= minmeasurements:
            #scaninds = scanarray[l:r]
            #massinds = masses[t:b]
            
            #I actually wonder if doing this would be a more effective method of mass-peak separation...
            gridoverlapcounts = ((grid > 0).sum(axis=0) > 1).sum() #checking for multiple separate mass peaks being in grid
            #using a straight-forward approach rather than a recursive one atm, might switch in the future, might not need to. The recursive idea has a shortcoming on where to place pre-existing confines for newer peaks, and how to expand it should it need to be expanded.
            if gridoverlapcounts / grid.shape[1] > gridoverlapthreshold: #starting from mass axis
                masschosen += 1
                fullmass = grid.sum(axis=1)
                #masspeaks = axis_peaks(fullmass, massboxcarlength, massmindist)
                masspeaks = axis_peaks(fullmass, massmindist)
                itercount += len(masspeaks) - 1
                for mp in masspeaks:
                    fullchrom = grid[mp[0]:mp[2]].sum(axis=0)
                    if (fullchrom > 0).sum() >= minmeasurements:
                        #chrompeaks = axis_peaks(fullchrom, chromboxcarlength, chrommindist)
                        chrompeaks = axis_peaks(fullchrom, chrommindist)
                        itercount += len(chrompeaks) - 1
                        for cp in chrompeaks:
                            if cp[2] - cp[0] >= minmeasurements:
                                cdpoints = (fullchrom[cp[0]:cp[2]] > 0).sum()
                                cfill = cdpoints / (cp[2] - cp[0]) >= peakfill
                                if cfill and cdpoints >= minmeasurements:
                                    peakgrid = grid[mp[0]:mp[2],cp[0]:cp[2]]
                                    pgdata = peakgrid > 0
                                    gridfail = (pgdata > 1).sum()
                                    ndatapoints = pgdata.sum()
                                    mfill = ndatapoints / (cp[2] - cp[0]) >= peakfill
                                    if mfill and ndatapoints >= minmeasurements:
                                        newfullmass = peakgrid.sum(axis=1)
                                        #trimming
                                        ml, mm, mr = mp
                                        lc = 0
                                        while newfullmass[lc] == 0 == newfullmass[lc+1]:
                                            lc += 1
                                        ml += lc
                                        rc = 0
                                        while newfullmass[rc-1] == 0 == newfullmass[rc-2]:
                                            rc -= 1
                                        mr += rc
                                        nmp = [ml, mm, mr]
                                        peakgrid = grid[nmp[0]:nmp[2],cp[0]:cp[2]]
                                        #print(coordwindow, 'm', mp, cp, f'({t}, {b}, {l}, {r})')
                                        endpeaks.add(peak_processing(peakgrid, coords, coordwindow-windowsize, nmp, cp, masses, scanarray, timearray, nsplits))
                                        #mp = [i + t + coordwindow - windowsize for i in mp]
                                        #cp = [i + l for i in cp]
                                        #collectedpeaks.append(mp + cp)
                                    else:
                                        pass
                                    if gridfail:
                                        print('mass axis failure', coordwindow-windowsize, mp, cp, f'({t}, {b}, {l}, {r})')
                                else:
                                    pass
            else: #starting from chrom axis, this route is chosen ~90-95% of the time
                chromchosen += 1
                fullchrom = grid.sum(axis=0)
                #chrompeaks = axis_peaks(fullchrom, chromboxcarlength, chrommindist)
                chrompeaks = axis_peaks(fullchrom, chrommindist)
                itercount += len(chrompeaks) - 1
                for cp in chrompeaks:
                    cdpoints = (fullchrom[cp[0]:cp[2]] > 0).sum()
                    cfill = cdpoints / (cp[2] - cp[0]) >= peakfill
                    if cfill and cdpoints >= minmeasurements:
                        #masspeaks = axis_peaks(grid[:,cp[0]:cp[2]].sum(axis=1), massboxcarlength, massmindist)
                        masspeaks = axis_peaks(grid[:,cp[0]:cp[2]].sum(axis=1), massmindist)
                        mcount = 0
                        itercount += len(masspeaks) - 1
                        for mp in masspeaks:
                            peakgrid = grid[mp[0]:mp[2],cp[0]:cp[2]]
                            ndatapoints = (peakgrid > 0).sum()
                            mfill = ndatapoints / (cp[2] - cp[0]) >= peakfill
                            if mfill and ndatapoints >= minmeasurements:
                                mcount += 1
                                #trimming
                                newfullchrom = peakgrid.sum(axis=0)
                                cl, cm, cr = cp
                                lc = 0
                                while newfullchrom[lc] == 0 == newfullchrom[lc+1]:
                                    lc += 1
                                cl += lc
                                rc = 0
                                while newfullchrom[rc-1] == 0 == newfullchrom[rc-2]:
                                    rc -= 1
                                cr += rc
                                ncp = [cl, cm, cr]
                                peakgrid = grid[mp[0]:mp[2],ncp[0]:ncp[2]]
                                #print(coordwindow, 'c', mp, cp, f'({t}, {b}, {l}, {r})')
                                endpeaks.add(peak_processing(peakgrid, coords, coordwindow-windowsize, mp, ncp, masses, scanarray, timearray, nsplits))
                                #collectedpeaks.append(mp + cp)
                            else:
                                pass
                        if mcount > 1:
                            print('chrom axis failure', coordwindow-windowsize, mp, cp, f'({t}, {b}, {l}, {r})')
                    else:
                        pass
    coordwindow += windowsize
print(time() - nt)

columntitles = [
            'leftmassscanindex',                          #first scan index of mass peak
            'rightmassscanindex',                         #last scan index of mass peak
            'leftchromscanindex',                         #first scan index of chromatographic peak
            'rightchromscanindex',                        #last scan index of chromatographic peak
            'totalintensity',                             #total intensity
            'meanintensity',                              #mean intensity
            'medianintensity',                            #median intensity
            'nmeancrosses',                               #number of times the chromatographic peak crosses the mean value
            'nmediancrosses',                             #number of times the chromatographic peak crosses the median value
            'samplingratebyscan',                         #estimated sampling rate based on intensity, scans indexes, and area
            'samplingratebytime',                         #estimated sampling rate basd on intensity, time, and area
            'scansleft',                                  #number of scans left of max
            'scansright',                                 #number of scans right of max
            'retentionscan',                              #scan index of max value
            'ndatapoints',                                #total number of data points
            'weightedmeanmass',                           #weighted mean of mass by intensity at each data point
            'geometricmeanmass',                          #mean of the mass-range where each mass is represented once
            'meanmass',                                   #mean of masses of each data point
            'massmax',                                    #the highest mass of the peak
            'massmin',                                    #the lowest mass of the peak
            'massatmaxchrom',                             #mass value(s) of the highest chromatographic point
            'nindividualmasses',                          #length of the mass range
            'nchrompoints',                               #number of data points along the chromatographic axis
            'nchromzeros',                                #number of zeros along the chromatographic axis
            'nchrommaxes',                                #number of chromatographic data points surrounded by two lower values
            'chrommaxesperpoints',                        #nchrommaxes / ndatapoints
            'chrommaxespernonzeropoints',                 #nchrommaxes / non-zero data points
            'chrommaxintensityloc',                       #index of highest chromatographic point
            'chrommaxintensity',                          #value of highest chromatographic point
            'nchromleftpoints',                           #number of points left of chromatographic max
            'nchromrightpoints',                          #number of points right of chromatographic max
            'retentiontime',                              #time at chromatographic max
            'chrommeanpercentofmax',                      #average (%chromatographic data / max)
            'chrommedianpercentofmax',                    #median (%chromatographic data / max)
            'timeleft',                                   #total time left of chromatographic max
            'timeright',                                  #total right right of chromatographic max
            'chromleftprominence',                        #prominence of left-most non-zero chromatographic point
            'chromrightprominence',                       #prominence of right-most non-zero chromatographic point
            'chromoriginaltrapzarea',                     #pure AUC using trapezoidal rule
            'chromoriginalsimpsarea',                     #pure AUC using simpson's rule
            'chrombaselinesubtractedtrapzarea',           #baseline-subtracted trapezoidal area
            'chrombaselinesubtractedsimpsarea',           #baseline-subtracted simpson's area
            'chromxprecision',                            #number of decimal points used for the x-axis associated value when making area splits
            'nmasspoints',                                #number of data points along the mass axis
            'nmasszeros',                                 #number of zeros along the mass axis
            'nmassmaxes',                                 #number of mass data points surrounded by two lower values
            'massmaxesperpoints',                         #nmassmaxes / ndatapoints
            'massmaxespernonzeropoints',                  #nmassmaxes / non-zero data points
            'massmaxintensityloc',                        #index of highest mass point
            'massmaxintensity',                           #value of highest mass point
            'nmassleftpoints',                            #number of points left of mass max
            'nmassrightpoints',                           #number of points right of mass max
            'maxmass',                                    #time at mass max
            'massmeanpercentofmax',                       #average (%mass data / max)
            'massmedianpercentofmax',                     #median (%mass data / max)
            'massesleft',                                 #number of masses left of mass max
            'massesright',                                #number of masses right of mass max
            'massleftprominence',                         #prominence of left-most non-zero mass point
            'massrightprominence',                        #prominence of right-most non-zero mass point
            'massoriginaltrapzarea',                      #pure AUC using trapezoidal rule
            'massoriginalsimpsarea',                      #pure AUC using simpson's rule
            'massbaselinesubtractedtrapzarea',            #baseline-subtracted trapezoidal area
            'massbaselinesubtractedsimpsarea',            #baseline-subtracted simpson's area
            'massxprecision',                             #number of decimal points used for the x-axis associated value when making area splits
            ]


#
#Calculatig peak widths
#


#area splits: linearly divides the area on either side of the max point of a peak into n areas. For n splits there will be n+1 bounds returned.
mlx = [f'leftmassareasplitxvals bound {n+1}' for n in range(nsplits+1)]
mly = [f'leftmassareasplityvals bound {n+1}' for n in range(nsplits+1)]
mrx = [f'rightmassareasplitxvals bound {n+1}' for n in range(nsplits+1)]
mry = [f'rightmassareasplityvals bound {n+1}' for n in range(nsplits+1)]
clx = [f'leftchromareasplitxvals bound {n+1}' for n in range(nsplits+1)]
cly = [f'leftchromareasplityvals bound {n+1}' for n in range(nsplits+1)]
crx = [f'rightchromareasplitxvals bound {n+1}' for n in range(nsplits+1)]
cry = [f'rightchromareasplityvals bound {n+1}' for n in range(nsplits+1)]

columntitles.extend(mlx)                                #x-values of left area splits along mass axis
columntitles.extend(mly)                                #y-values of left area splits along mass axis
columntitles.extend(list(reversed(mrx)))                #x-values of right area splits along mass axis
columntitles.extend(list(reversed(mry)))                #y-values of right area splits along mass axis
columntitles.extend(clx)                                #x-values of left area splits along chromatographic axis
columntitles.extend(cly)                                #y-values of left area splits along chromatographic axis
columntitles.extend(list(reversed(crx)))                #x-values of right area splits along chromatographic axis
columntitles.extend(list(reversed(cry)))                #y-values of right area splits along chromatographic axis

df = pd.DataFrame(endpeaks, columns=columntitles)

percentcovered = df.loc[:,'totalintensity'].sum() / sm.sum()
percentcoveredminusnoise = df.loc[:,'totalintensity'].sum() / (sm.sum() - noisecount)
percentnoise = noisecount / sm.sum()

print(df.shape[0], 'peak founds')
print('raw total signal accounted for:', percentcovered)
print('total non-noise signal accounted for:', percentcoveredminusnoise)
print('percent noise:', percentnoise)


#original
#30899 peaks
#raw total signal accounted for: 0.6983727750921437
#total non-noise signal accounted for: 0.7094715105544405
#percent noise: 0.01564366616162403

#signal-distance based
#39456 peak founds
#raw total signal accounted for: 0.6221903297194867
#total non-noise signal accounted for: 0.6453083410460662
#percent noise: 0.035824752069846796

#signal-distance based with x2 k=2 cutoff
#44502 peak founds
#raw total signal accounted for: 0.6211686389880313
#total non-noise signal accounted for: 0.6308774232810919
#percent noise: 0.015389335447394466

#baseline no filtering
#29550 peak founds
#raw total signal accounted for: 0.6731584579308569
#total non-noise signal accounted for: 0.6731584579308569
#percent noise: 0.0
#mean number of adjacent peaks 1.0701184433164128
#mean peak neighbor distance 33.584922321722175

#^n=20 now though, re-do all these

#ind-based
#29550 peak founds
#raw total signal accounted for: 0.6731584579308569
#mean number of adjacent peaks 1.6368866328257192
#Counter({1: 18502,
#         2: 6797,
#         3: 2386,
#         4: 1053,
#         5: 411,
#         6: 190,
#         7: 98,
#         8: 58,
#         9: 29,
#         10: 14,
#         12: 4,
#         13: 3,
#         14: 4,
#         15: 1})
#mean peak neighbor distance 33.58492232172217

#true value based
#37533 peak founds
#raw total signal accounted for: 0.6294033618137362
#mean number of adjacent peaks 1.775850584818693
#Counter({1: 21243,
#         2: 9324,
#         3: 3903,
#         4: 1668,
#         5: 711,
#         6: 368,
#         7: 149,
#         8: 67,
#         9: 53,
#         10: 18,
#         11: 10,
#         12: 4,
#         13: 2
#         14: 5,
#         15: 3,
#         16: 2,
#         17: 3})
#mean peak neighbor distance 32.46486113886733

#ind-based with static scanbuffer at 5
#21620 peak founds
#raw total signal accounted for: 0.3874648547883984
#mean number of adjacent peaks 1.6469010175763181
#Counter({1: 12905, 2: 5603, 3: 1948, 4: 703, 5: 279, 6: 66, 7: 38, 8: 23, 9: 17, 11: 14, 13: 8, 12: 7, 10: 6, 14: 2, 15: 1})
#mean peak neighbor distance 33.05377010196602

#ind-based with x2 scanbuffer
#32276 peak founds
#raw total signal accounted for: 0.7039831481635096
#mean number of adjacent peaks 1.6747428429793034
#Counter({1: 20100, 2: 7251, 3: 2660, 4: 1131, 5: 534, 6: 304, 7: 136, 8: 71, 9: 44, 10: 16, 11: 7, 12: 6, 13: 5, 14: 5, 16: 4, 15: 2})
#mean peak neighbor distance 33.83332460113166

#weighted mean for scan buffer, ind-based, performed much slower
#50763 peak founds
#raw total signal accounted for: 0.7047146650282812
#mean number of adjacent peaks 1.6435255412619925
#Counter({1: 31488, 2: 11731, 3: 4324, 4: 1801, 5: 753, 6: 360, 7: 152, 8: 90, 9: 40, 10: 13, 11: 6, 12: 3})
#mean peak neighbor distance 34.201678491040106

#indbased 0.3 peakfill
#39394 peak founds
#raw total signal accounted for: 0.6924597887499476
#mean number of adjacent peaks 1.5383053256841144
#Counter({1: 26603, 2: 8235, 3: 2574, 4: 1081, 5: 453, 6: 224, 7: 113, 8: 51, 10: 18, 9: 16, 11: 7, 16: 5, 14: 4, 12: 4, 15: 3, 13: 2, 17: 1})
#mean peak neighbor distance 37.2660596234937

#original noise-reduction functionality with scanbuffer at 5
#31208 peak founds
#raw total signal accounted for: 0.6992480211234963
#mean number of adjacent peaks 1.6412458344014356
#Counter({1: 19661, 2: 7034, 3: 2481, 4: 1086, 5: 455, 6: 247, 7: 130, 8: 63, 9: 20, 10: 14, 11: 5, 12: 4, 14: 4, 13: 3, 15: 1})
#mean peak neighbor distance 33.964905822872026

#automated scanbuffer from mean on original ind-based noise-reduced setup
#31239 peak founds
#raw total signal accounted for: 0.6986521253235491
#mean number of adjacent peaks 1.6422740804763276
#Counter({1: 19681, 2: 7047, 3: 2464, 4: 1084, 5: 462, 6: 257, 7: 131, 8: 66, 9: 18, 10: 9, 11: 5, 13: 5, 12: 5, 14: 4, 15: 1})
#mean peak neighbor distance 33.9578021107364

#same as above but inverse weighted scanbuffer mean
#30973 peak founds
#raw total signal accounted for: 0.6946864076655369
#mean number of adjacent peaks 1.6452716882446001
#Counter({1: 19470, 2: 7003, 3: 2462, 4: 1094, 5: 444, 6: 260, 7: 121, 8: 61, 9: 25, 10: 13, 13: 5, 12: 5, 11: 5, 14: 4, 15: 1})
#mean peak neighbor distance 33.92526984528731

#new expandedrings process, inverse weighted mean scanbuffer
#19096 peak founds
#raw total signal accounted for: 0.06175244161659493
#total non-noise signal accounted for: 0.06273382868965643
#percent noise: 0.01564366616162403
#mean number of adjacent peaks 1.9548596564725598
#Counter({1: 8688, 2: 5861, 3: 2698, 4: 1110, 5: 429, 6: 168, 7: 49, 8: 32, 10: 24, 9: 19, 11: 8, 14: 3, 12: 3, 13: 2, 16: 1, 15: 1})
#mean peak neighbor distance 28.350927656258673

#erase all found peaks -> find average distance between the leftovers?

#can i take the area of sm 1 row at a time? Would that be the same as taking the area under 10 rows at a time? Can I do this and sum to get the same area? Then can I use area as a metric for this? Because high intensity points that are leftover might not make much total area if they're lone points or something?

#^maybe I should return the split areas too
#bound index 0 on crx is the same split-level as index 0 on clx, etc
cwidths = [f'chrompeakwidth at {n}% area' for n in np.flip(np.linspace(0,100,nsplits+1)).tolist()]
mwidths = [f'masspeakwidth at {n}% area' for n in np.flip(np.linspace(0,100,nsplits+1)).tolist()]

df.loc[:,cwidths] = df.loc[:,crx].to_numpy() - df.loc[:,clx].to_numpy()
df.loc[:,mwidths] = df.loc[:,mrx].to_numpy() - df.loc[:,mlx].to_numpy()

#when comparing total intensity in sm to total sum intensities in df, I want to quantify what bottom portion of the intensity distribution would have to be considered noise for these results to make sense?
#but this may be more complicated. That 70% was kinda 'weak', but it might not really be, considering that 70% might be erased via w2! I should sum all of the noise that's being erased when considering this stat.

#
#Finding isotopic patterns
#


#loading isotope database
with open(isotopefile, "rb") as pick:
    isotopeabundances, seqsbymass = pickle.load(pick)
#make something to prove later on is that there are no incomplete lists in isotopeabundances

massvals = df.loc[:,'weightedmeanmass']
massaxis = massvals.rank(method='dense').astype(int) - 1

df.set_index('weightedmeanmass', inplace=True) #assumes all weightedmeanmasses are unique
#^if they're not all unique just make a negligible_difference() or something
#I actually think there's a super high chance that they're all unique, and it would be extremely difficult for them not to be
retentiondict = df.loc[:,'retentiontime'].to_dict()

coordarray = df.iloc[:,:4].to_numpy()
coorddict = dict(zip(df.index, coordarray.tolist())) #t, b, l, r

timevals = df.loc[:,'retentiontime']
peakmatrix = np.zeros(shape=(massvals.unique().size, timevals.unique().size))

timeaxis = timevals.rank(method='dense').astype(int) - 1
peakmatrix[massaxis, timeaxis] = df.loc[:,'chromoriginaltrapzarea']

#masspoints = np.sort(massvals.unique())
masspoints = massvals.to_numpy()
timepoints = np.sort(timevals.unique()) #I won't need this probably?

adjustedmassindex = pd.Series(massaxis.to_numpy(), index=massvals.to_numpy())

masscoordinates = adjustedmassindex.to_dict()
timecoordinates = timeaxis.to_dict()

peakareas = df.loc[:,'chromoriginaltrapzarea'].to_dict()

trainers = timevals.reset_index().to_numpy()
#this approach will use actual distance (not index) because finding the closest mass is the goal
nbrs = spatial.KDTree(trainers)

#using index proximity here because the time/mass scale difference fucks with finding neighbors
#it would be good to later go back and visualize the locations with a lot of adjacent peaks
inds = np.array(list(zip(masscoordinates.values(), timecoordinates.values())))
indnbrs = spatial.KDTree(inds)
nadjacents = [len(i) - 1 for i in indnbrs.query_ball_point(inds, r=20).tolist()]
dists, distinds = indnbrs.query(inds, k=2)

print('mean number of adjacent peaks', np.mean(nadjacents))
print(Counter(nadjacents))
print('mean peak neighbor distance', dists[:,1].mean())

#put here:
#a way for the nearest neighbor to make a new column in df regrding neighboring peaks that are of a ~scanbuffer distance, or something like it, away from each other. Potentially a single peak, I'd like to keep track of how this performs.
#are these useful for that?
#df.loc[:,'leftmassscanindex']
#df.loc[:,'rightmassscanindex']
#etc
#expandedrings = np.array(list(expandedrings))
#overlaps = rectangle_overlap(expandedrings)
#ringgroups = generic_meta_overlap(overlaps)

#look at the mins, means, and maxes of the mins, means, and maxes
horizontaldistance = 10 #indices, it's a max flanking distance to each side of a peak, essentially.
movingind = horizontaldistance
maxiter = peakmatrix.shape[1]
edict = defaultdict(list)
#this could be changed to not need a horizontal distance, if you make a peakmatrix copy, and fill in the width of any peak with its weightedmeanmass value along the time axis, you can keep whatever peaks are relevant in a more flexible manner, as you move across a sliding window equal to the size of the least-wide peak, or just each column I suppose
#a new setting to add would be # of nearest to check, or a pre-set minimum distance. This would help me spot-check an analytes charge state
while movingind <= maxiter:
    slicedpeaks = peakmatrix[:,movingind-horizontaldistance:movingind]
    peakinds = np.where(slicedpeaks.any(axis=1))[0]
    orderedmasses = np.sort(masspoints[peakinds])
    #if orderedmasses.size > 0:
    forwardcheck = np.diff(orderedmasses)
    forwardcheck = np.append(forwardcheck, forwardcheck[-1])
    forwardcheck[0] #for first of orderedmasses
    backwardcheck = np.flip(np.abs(np.diff(np.flip(orderedmasses))))
    backwardcheck = np.append(backwardcheck[0], backwardcheck)
    backwardcheck[-1] #for last of orderedmasses
    mindists = np.stack((forwardcheck, backwardcheck), axis=0).min(axis=0)
    for k, v in zip(orderedmasses.tolist(), mindists.tolist()):
        edict[k].append(v)
    movingind += 1

sdict = {}
for k, v in edict.items():
    sdict[k] = [min(v), np.mean(v), max(v)]

sframe = pd.DataFrame.from_dict(sdict, orient='index', columns=['min', 'mean', 'max'])


#there's going to be a dilemma with this approach: Using whatever point is nearest the expected mass will lead to incorrect isotopes being assigned to
#^so the ppm range dictates the allowed range of overlap

#regarding charges, I have a hunch that usually you'll see every charge on the way to a higher charge without generaly skipping numbers, I want to see this in the data first so I'll do the rigid min-max range
#At some point I need to show that every set of peptides with equal masses has an identical elemental composition

proton = 1.00727647

#start with the 1d KNN, then work towards exluding based on max/range ppm, then further optimize for speed

#mincharge = 2
#maxcharge = 8
#ppmrange = 10 #could also be a required max ppm for each distribution component, in order to be added to the list of matches, the 'range' would technically be 20, +/- ppm should be put in the final list instead of the absolute value
##in the case that something is at like 900 ppm error, that would just be considered as an absent peak, and treated the same as the peaks that failed to match to a different isotopomer -> sm gets searched for these before being abandoned.
##after all isotopomer candidates are collected, see how the ppm error looks across matches with only 1 candidate? Will there be any like that?
#uniqueidentifier = 0
#rcount = 0
#charges = range(mincharge, maxcharge+1)
#endlist = []
#for dn, (seqmass, s) in enumerate(seqsbymass.items()):
#    if dn > 100:
#        break
#    seqlist = []
#    dist = isotopeabundances[s[0]]
#    featuremass = dist.most_common(1)[0][0]
#    massarray = np.array(list(dist.keys()))
#    percents = np.array(list(dist.values()))
#    for c in charges:
#    #c = mincharge
#    #while True:
#        #this can also be slow, why not train a 1D KNN here?
#        chargemass = (featuremass + (c*proton)) / c
#        massdiffs = (np.abs(masspoints - chargemass) * 1000000) / chargemass
#        selectedmasses = masspoints[massdiffs < ppmrange]
#        if selectedmasses.size > 0:
#            #find the massinds for the rest of massarray now
#            chargemasses = ((massarray + (c*proton)) / c).reshape(-1,1)
#            #massarraydiffs = np.abs(masspoints - ((massarray+(proton*c))/c).reshape(-1,1))
#            massarraydiffs = (np.abs(masspoints - chargemasses) * 1000000) / chargemasses
#            #min/max peak mass boundaries should be used instead of this silly shit
#            # a 1:1 match/find -> closest match wins -> other lookups can be assigned afterwards if they fall within mass range
#            selectedarraymasses = [masspoints[i < ppmrange].tolist() if (i < ppmrange).any() else [False] for i in massarraydiffs]
#            arraymassgroups = generic_meta_overlap(selectedarraymasses)
#            massgrouporganizer = defaultdict(set)
#            for n1, mg in enumerate(arraymassgroups):
#                for n2, sa in enumerate(selectedarraymasses):
#                    for m in mg:
#                        if m in sa:
#                            massgrouporganizer[n1].add(n2)
#            combinedinds = list(massgrouporganizer.values())
#            arraylens = [len(i) for i in combinedinds]
#            arraysums = [percents[list(i)].sum() for i in combinedinds]
#            retentioninds = [timecoordinates[i] for i in selectedmasses.tolist()]
#            arraymassinds = []
#            for amg in arraymassgroups:
#                if any(amg):
#                #retentiontimes = [retentiondict[i] for i in selectedmasses.tolist()]
#                    arraymassinds.append([masscoordinates[i] for i in amg])
#            #c += 1
#            overlappedgroups = [list(i) for i in massgrouporganizer.values()]
#            retentiontimes = timepoints[retentioninds]
#            masslookups = chargemasses.flatten()
#            #for r in retentiontimes:
#            for t in retentioninds: #temp for plotting during dev
#                r = timepoints[t]
#                lookups = np.stack((masslookups, np.repeat(r, masslookups.size)), axis=1)
#                dists, distinds = nbrs.query(lookups, k=1)
#                matches = trainers[distinds]
#                uniquematches = np.unique(matches, axis=0)
#                matchschemes = []
#                for u in uniquematches:
#                    matchschemes.append(np.where((matches == u).all(axis=1))[0].tolist())
#                massmatchgroups = defaultdict(set)
#                for k, v in massgrouporganizer.items():
#                    for n, ms in enumerate(matchschemes):
#                        for m in ms:
#                            if m in v:
#                                massmatchgroups[n].add(k)
#                #if a key in massmatchgroups only has 1 value, that's a good thing. If there's multiple -> lowest distance wins out
#                massmatches = {}
#                for k, v in massmatchgroups.items():
#                    if len(v) > 1:
#                        midlist = []
#                        for i in v:
#                            midlist.append(dists[list(massgrouporganizer[i])].min())
#                        indexer = list(v)[np.argmin(midlist)]
#                    else:
#                        indexer = sum(v)
#                    massmatches[k] = indexer
#                #keys in massmatches correspond to indices of uniquematches
#                #values in massmatches correspond to keys in massgrouporganizer
#                #values in massgrouporganizer correspond to indices in matches/lookups
#                percarray = []
#                for mk, mv in massmatches.items():
#                    ma = uniquematches[mk]
#                    indexer = list(massgrouporganizer[mk])
#                    valperc = percents[indexer].sum()
#                    #matchitems = lookups[indexer][:,0].tolist()
#                    matchcount = len(indexer)
#                    #totalerror = np.abs(massarray[indexer] / c - ma[0]).sum() / ppm / c
#                    searchmass = masslookups[indexer[percents[indexer].argmax()]]
#                    mainerror = (ma[0] - searchmass) / ma[0] * 1000000
#                    area = peakareas[ma[0]]
#                    outlist = [searchmass, ma[0], mainerror, ma[1], matchcount, c, dn, uniqueidentifier, rcount, seqmass, valperc, area]
#                    uniqueidentifier += 1
#                    seqlist.append(outlist)
#                rcount += 1
#    endlist.append(seqlist)
#    break
#
#cols = [
#        'desiredmass',
#        'foundmass',
#        'mainppmerror',
#        'rt',
#        'matchcount',
#        'charge',
#        'dist #',
#        'uid',
#        'rid',
#        'dictmass',
#        'percentcomp',
#        'peakarea'
#        ]
#
#
#isf = pd.DataFrame(seqlist, columns=cols)
#
#isf.set_index('rid', inplace=True)
#isf.loc[:,'expectedperc'] = isf.loc[:,'percentcomp'] / isf.groupby(level=0).sum().loc[:,'percentcomp']
#isf.loc[:,'areaperc'] = isf.loc[:,'peakarea'] / isf.groupby(level=0).sum().loc[:,'peakarea']

#Mass corrections: ppm should be derived from the weighted mean of expected isotopomers within its range, the weights would be the percents in dist
#^This should come in being pretty neat for quantitative inferences of overlap, and when deriving isotopic percentages across datafiles

#define mass range across sm/mzindex and find dense areas
#check if there's any known peaks found along that range, also retain areas of high density somehow
#retention times that pass the above move on to the next highest isotopomer
#I also need a notebook page showing how easy/difficult it is to differentiate uniqueness across my current isotope abundance library. Ie, to what samplesize do I need to calculate to differentiate distributions of a similar size, or similar ballpark when considering charge?


massnbrs = spatial.KDTree(masspoints.reshape(-1,1))

mincharge = 2
maxcharge = 8
ppmrange = 10 #could also be a required max ppm for each distribution component, in order to be added to the list of matches, the 'range' would technically be 20, +/- ppm should be put in the final list instead of the absolute value
#in the case that something is at like 900 ppm error, that would just be considered as an absent peak, and treated the same as the peaks that failed to match to a different isotopomer -> sm gets searched for these before being abandoned.
#after all isotopomer candidates are collected, see how the ppm error looks across matches with only 1 candidate? Will there be any like that?
topcount = 3 #top n of distribution to search for
uniqueidentifier = 0
rcount = 0
charges = np.arange(maxcharge).reshape(-1,1) + 1
endlist = []
for dn, (seqmass, s) in enumerate(seqsbymass.items()):
    if dn > 100:
        break
    seqlist = []
    dist = isotopeabundances[s[0]]
    massarray = np.array(list(dist.keys()))
    percents = np.array(list(dist.values()))
    featuremasses = np.array([i for i, p in dist.most_common(topcount)]) #not limiting this to topcount will be more sensitive to smaller quantities. How feasible is that?
    #^note for the case of when an overlapping distribution may be too small to be found in the pwf, but still exists?
    featuresearches = ((featuremasses + (charges*proton)) / charges).flatten()
    chargemasses = ((massarray + (charges*proton)) / charges) #later need to incorporate min/max mass in mzindex into this
    chargepositions = np.repeat(charges, len(massarray))
    percentpositions = np.repeat(percents[None,:], len(charges), axis=0).flatten()
    flatcharges = chargemasses.flatten()
    ppmdist = featuresearches / 1000000 * ppmrange
    selectedarraymasses = massnbrs.query_ball_point(featuresearches.reshape(-1,1), r=ppmdist)
    flatselection = list(itertools.chain(*selectedarraymasses))
    massselection = masspoints[flatselection]
    retentioninds = [timecoordinates[i] for i in massselection]
    retentiontimes = timepoints[retentioninds]
    chargeindices = np.repeat(chargepositions[None,:], len(retentiontimes), axis=0).flatten()
    percentindices = np.repeat(percentpositions[None,:], len(retentiontimes), axis=0).flatten()
    masslooks = np.repeat(flatcharges[None,:], len(retentiontimes), axis=0).flatten()
    timelooks = np.repeat(retentiontimes, len(flatcharges))
    lookups = np.stack((masslooks, timelooks), axis=1)
    dists, distinds = nbrs.query(lookups, k=1)
    matches = trainers[distinds]
    matchcoords = np.array(list(map(coorddict.get, matches[:,0].tolist())))
    massbounds = mzindex[matchcoords[:,:2]]
    timebounds = timearray[matchcoords[:,2:]]
    uniquematches = np.unique(matches, axis=0)
    framestack = np.hstack((lookups, matches, massbounds, timebounds))
    matchframe = pd.DataFrame(framestack, columns=['searchmass', 'searchtime', 'matchmass', 'matchtime', 'minmass', 'maxmass', 'starttime', 'endtime'])
    matchframe.loc[:,'charge'] = chargeindices
    matchframe.loc[:,'percent'] = percentindices
    matchframe.loc[:,'ppmerror'] = (matchframe.loc[:,'searchmass'] - matchframe.loc[:,'matchmass']) / matchframe.loc[:,'searchmass'] * 1000000
    #indtester = np.logical_and(matchframe.loc[:,'minmass'] <= matchframe.loc[:,'searchmass'], matchframe.loc[:,'searchmass'] <= matchframe.loc[:,'maxmass']) #this is more stringent, but maybe it'll be better large-scale?
    indtester = matchframe.loc[:,'ppmerror'].abs() <= ppmrange
    matchframe = matchframe.loc[indtester]
    matchframe = matchframe.drop_duplicates(['searchmass', 'matchmass'])
    newtimebounds = matchframe.loc[:,('starttime', 'endtime')].to_numpy()
    overlapcheck = np.logical_and(newtimebounds[:,0] <= newtimebounds[:,1].reshape(-1,1), newtimebounds[:,1] >= newtimebounds[:,0].reshape(-1,1))
    overlaptimes = generic_meta_overlap(np.argwhere(overlapcheck).tolist())
    matchframe.reset_index(drop=True, inplace=True) #alllowing the current index to be made of the grouped indices found in the overlaps
    for n, o in enumerate(overlaptimes): #is this loop necessary? It's the same as the one below
        matchframe.loc[o,'timegroup'] = n
    #index order: matchmass -> charge -> searchtime
    #all the peaks of a distribution that fall under a mass range umbrella of a match, not just those that matched to it, will be incorporated into that match. Those that remain outside of it will be excluded, but how should the ppm it in on this? I don't think the ppm fuzzy boundary would be necessary as long as the width of the match trying to take all the other matches under its umbrella is wider than that boundary. It probably will be, because otherwise these things wouldn't overlap. The umbrella is being distributed across the rest of the non-matching distribution because there could be the scenario where the mass of something within a large isotopomer could be closer to some small peak around its outer edge than to the center of its true peak.
    #set up the check for if a lookup falls within the range of its match, for the ^above function that will only occur at individual retention times
    #np.abs(retentiontimes - uniquematches[:,None,1]).min(axis=0) can show that some retention times don't show up on any matches, so they have big mins
    #np.abs(retentiontimes - uniquematches[:,None,1]).min(axis=1) can show that some uniquematches are quite far away from desirable retentiontimes, so I'm curious how the ppm error looks on those as well, if it falls within the ppm range I think I should still consider that RT as an sm-search candidate
    matchframe.set_index('timegroup', inplace=True)
    matchframe.sort_index(inplace=True)
    for i in matchframe.index.unique():
        for c in np.unique(matchframe.loc[i, 'charge']):
            c = int(c)
            try:
                tf = matchframe.loc[i].query(f'charge == {c}')
            except AttributeError:
                tf = matchframe.loc[i]
            currentdist = chargemasses[c-1]
            smin = currentdist.min() - 1
            smax = currentdist.max() + 1
            ismin = find_nearest(mzindex, smin)[0]
            ismax = find_nearest(mzindex, smax)[0]
            if type(tf) == pd.DataFrame:
                tmin = tf.loc[:,'starttime'].min() - 1
                tmax = tf.loc[:,'endtime'].max() + 1
            else:
                tmin = tf.loc['starttime'] - 1
                tmax = tf.loc['endtime'] + 1
            itmin = find_nearest(timearray, tmin)[0]
            itmax = find_nearest(timearray, tmax)[0]
            sample = sm[ismin:ismax+1,itmin:itmax+1].toarray()
            sampleinds = np.argwhere(sample > 0)
            
            rec = np.array([ismin, ismax, itmin, itmax])
            rangeoverlaps = coord_rectangle_overlap(rec, coordarray)
            yspans = mzindex[rangeoverlaps[:,:2]]
            xspans = timearray[rangeoverlaps[:,2:]]
            yspans[yspans < smin] = smin
            yspans[yspans > smax] = smax
            xspans[xspans < tmin] = tmin
            xspans[xspans > tmax] = tmax
            spans = np.hstack((xspans, yspans))
            
            alpha=0.5
            bwidth = 0.2
            fig, ax = plt.subplots(nrows=3, figsize=(6,8), facecolor='gray')
            for t, b, l, r in spans:
                width = r - l
                height = b - t
                rect = patches.Rectangle((l, t), width, height, linewidth=2, edgecolor='gold', facecolor='gold', alpha=alpha)
                ax[0].add_patch(rect)
            ax[0].plot(mzindex[ismin:ismax+1][sampleinds[:,0]], timearray[itmin:itmax+1][sampleinds[:,1]][::-1], '.', markersize=0.3, color='white')
            ax[0].tick_params(axis='x', colors='white')
            ax[0].tick_params(axis='y', colors='white')
            ax[0].set_facecolor('gray')
            nax = ax[1].twinx()
            nax.bar(currentdist, percents, width=bwidth, color='gold', alpha=alpha)
            ax[1].plot(mzindex[ismin:ismax+1], sample.sum(axis=1), color='white')
            ax[1].tick_params(axis='x', colors='white')
            ax[1].tick_params(axis='y', colors='white')
            ax[1].set_facecolor('gray')
            align_yaxis_np(ax[1], nax)
            ax[2].plot(timearray[itmin:itmax+1], sample.sum(axis=0), color='white')
            ax[2].tick_params(axis='x', colors='white')
            ax[2].tick_params(axis='y', colors='white')
            ax[2].set_facecolor('gray')
            plt.suptitle(f'group {i}, charge {c}')
            plt.show()
            print(tf)
    break

#isotopic distribution mapping:
#moving forward, I'm going to assume that it would be too unlikely for any two distributions to overlap without any serious evidence for it.
#^I don't see any other reasonable way forward that likely wouldn't run into an unnecessary amount of true negatives
#BUT I will stick with noting things as 'possibilities', until I can show that any single evident distribution can be fully explained by a theoretical one

#for sm plotting:
#plot both the mass and chrom axis/peaks to visualize isotope distributions, does either of these hold an advantage?
#^ I think it might actually be wiser to measure area from the mass axis when looking for isotopic envelopes because the isotopes can be spread out over time in a non-neat manner. There's no reason they should form independent peaks, right?

#when mapping to sm, might is be wise to adjust peak widths if they don't initially make sense? If there are other potential minimum points to pick from, and if the widths of isotoeps seem to be a bit different, this could be a good reason for an adjustment.

#keep a defaultdict(list) of potential overlaps - it will be {weightedmeanmass: [peakidentifier,...]} Easy to find which peaks are involved in which potential distributions
#retain any potential distribution, that could be within another, if there is reasonable evidence.
#the reasonable evidence: you see a part of the distribution,
# - that could either be potentialy entitely encompassed -> and only a quantitative inference post-protein ID would give a reason to look into it
# - or a piece of a distribution is present around another, and pieces of it can't be distinguished from a larger distribution that encompasses its parts. You could infer its quantitative presence post-protein ID
#sm incorporation for an in-depth look at missing isotopes will be necessary pre df-formation

#Levels of distribution, descriptions:
# 1. Definitely there, a distribution is matched, and the percent areas match up.
#   - Extra steps about multiple sequences owning the same distribution later.
# 2. Definitely there with something else, but some of the percents are off - as if something else is overlapping with it.
# 3. Possibly there, this would be something that is the small overlap thats throwing off a more prominent distribution. It could also be two equally-size distributions that are fucking with each other. Can't deny that it's there.
# 4. Not there, period - plausible deniability, a major isotopomer is missing

#Would it be useful to scrape a database of what peptides from a proteome have and haven't been identified? Perhaps this could serve as some sort of basis for identifications. Wouldn't need to include any MS2 data at all I guess? Although a vetting process might be a good idea. Maybe a 3 of independent identifications within a database can serve as a reproducibility measure for this instead?
#^Maybe my own MS2 scoring would be useful I suppose? But it would still be missing MS1 data...

#you need to look for charge states at each retention time. charge states might need to be in the loop twice... nah but you should do the KNN all at once - every charge state and every isotope at every retention time.
#^Accumulate a list of retention times via charge state iteration...
#Every charge state at every retention time that you find for each featuremass found at each charge state

#collect, in index order: lookup mass, charge, RT, dist #, cumulative identifier #, 

#alternative approach:
#using sm as the basis for end-comparisons, and peaks found in peakmatrix as the basis for how wide a window to look for isotopes of a single charge, and expand to other charges... or something idk. Look at all charges at once - sure

#isotopic distribution considerations:
#if some mass is off by ~whatever ppm, all the isotopes should be off by that, right? It should be a consistent type of mass error I would think.
#^All that in relation to intensity too, less intense isotopes might have more error?
#

#could probably try finding best match per seqsbymass group first, then find the best match for each peak/group of peaks

#I think the most efficient method of searching this would be to look for 1 isotope group at a time across the entirety of peakmatrix. Do what you can, probably won't take too many steps to do, then just multiprocess or whatever.

#make an area-representing matrix out of locations from peak scans and masses
#can isotope search by column, with a width-buffer for columns of n scans or whatever
#maybe intensities instead of area? I wonder if the isotope ratios of either would differ... This would reveal whether area vs intensity quant is linear, which I'd expect intensity quant to be less linear.

#up next, find nearest vertical distances through that matrix you're gonna make, like nearest neighborrs but unidirectional within a +/- window.
#what's this distribution look like



##cl, cm, cr = chrompeaks.transpose()
##ml, mm, mr = masspeaks.transpose()
#cl, cm, cr = test[:,3:].transpose()
#ml, mm, mr = test[:,:3].transpose()
##grid = window[t:b,l:r]
#scaninds = scanarray[l:r]
#massinds = masses[t:b]
#fullchrom = grid.sum(axis=0)
#fullmass = grid.sum(axis=1)
#fig, ax = plt.subplots(3,1,figsize=(6,6))
#ax[0].plot(scaninds, fullchrom, '-', linewidth=0.2, color='black')
##ax[0].plot(scaninds[cnoisemask], fullchrom[cnoisemask], '.-', markersize=1, color='dimgray', linewidth=0.3)
##ax[0].plot(scaninds[~cnoisemask], fullchrom[~cnoisemask], '.', markersize=1, color='darkorange')
#ax[0].vlines(scaninds[cl], 0, fullchrom[cm], color='darkorange', linewidth=0.4)
#ax[0].vlines(scaninds[cr], 0, fullchrom[cm], color='royalblue', linewidth=0.4)
#ax[0].vlines(scaninds[cm], 0, fullchrom[cm], color='black', linewidth=0.4)
#ax[0].hlines(fullchrom[cm], scaninds[cl], scaninds[cr], linewidth=0.4, color='black')
#
#ax[1].plot(massinds, fullmass, '-', linewidth=0.2, color='black')
##ax[1].plot(massinds[mnoisemask], fullmass[mnoisemask], '.-', markersize=1, color='dimgray', linewidth=0.3)
##ax[1].plot(massinds[~mnoisemask], fullmass[~mnoisemask], '.', markersize=1, color='darkorange')
#ax[1].vlines(massinds[ml], 0, fullmass[mm], color='darkorange', linewidth=0.4)
#ax[1].vlines(massinds[mr], 0, fullmass[mm], color='royalblue', linewidth=0.4)
#ax[1].vlines(massinds[mm], 0, fullmass[mm], color='black', linewidth=0.4)
#ax[1].hlines(fullmass[mm], massinds[ml], massinds[mr], linewidth=0.4, color='black')
#ax[2].pcolor(grid, norm=LogNorm())
#for rt, rm1, rb, rl, rm2, rr in test:
#    width = rr - rl
#    height = rb - rt
#    #width = timearray[rr-1] - timearray[rl]
#    #height = masses[rb-1] - masses[rt]
#    rect = patches.Rectangle((rl, rt), width, height, linewidth=2, edgecolor='red', facecolor='none')
#    #rect = patches.Rectangle((timearray[l], masses[t]), width, height, linewidth=2, edgecolor='red', facecolor='none')
#    ax[2].add_patch(rect)
#
#ax[0].set_title('chrom')
#ax[1].set_title('mass')
#plt.tight_layout()
#plt.show()
##print('chrom vs. mass:', cpeaksperiters, 'vs.', mpeaksperiters)
#print('chrom:', len(chrompeaks), 'mass:', len(masspeaks))
#print(t,b,l,r)
#print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

#if potential peak group is within scanbuffer range of the bottom, it won't be processed in that round and the next round will have a slightly extended range that includes it. Therefore, to prevent any crossover - anything within scanbuffer range of the top is also not counted.



#
#Parameter Collections:
#

#Ideas:
#collect # mins and # maxes as done through min_finding and max_finding (they're faster than scipy's argrelmax/min functions!). This gives a description of the consistency of the signal. Also make a ratio out of (#mins or #maxes I suppose / #data points). An extra metric ratio of #mins / #maxes would be good too.
#^these give the same number, pretty much, so maybe only use 1?
#   > #maxes per n data points
#   > #maxes per n non-zero data points
#   > would be cool to see a distribution of all these ratios, and even ratios of the ratios to show comparison across all of the distribution
#these metrics together will determine whether two bordering adjacent peaks should be combined into 1
#also collect % of every datapoint to maximum, d / max
#   > get mean median max mode(and number of hits for mode) etc of this %, it will give good peak stats
#Endgame: Then try to separate peaks by shape based on combined information from the above ratios and stats and visualize the clusters. Use this as the basis for the peak cluster separation idea that plots averaged of clusters and whatnot
#Also collect how many times the data crosses the mean. This should tell you whether a peak is inhibited a lot, and (crosses / n data points)
#   > Also collect mode and median crosses, name them as such! (will mode really be useful? maybe if it's integer, or rounded to 10s or something - actually this could be another layer, round to 1s/10s/100s and collect the same info for mode???)

#Basic Info
#number of chromatographic and mass axis datapoints, and total number of datapoints
#intensity
#total intensity, the sum of grid
#"Sampling rate" -> total intensity / AUC -> This should give you an idea of how well sampled a peak was, from the basis of integration itself. There's no way the sum of intensities should ever be higher, right?
#baseline intensity on either side
#baseline subtracted areas and intensities
#^subtract area under baseline
#mass center, mass of highest intensity point along both chrom and mass axes
#geometric mass center, mean mass, weighted mean mass, mass at max intensity
#mass range
#use both simps and trapz for peak area, advertize as such, all area values should be generated only using time as an x-axis (or mass???). It would be interesting, later on, to see if quant results improve by using either scans/indices x-values, or by leaving them out during integration.
#Trapezoidal rules should outperform simpsons for peak data in general. The simpsons rule is here for a sanity check, it should always be lower, right? When I split the data for peak splits, I'll base the area splits on the trapz integration.
#signal prominence, lowest non-zero point on either side / apex
#a meta info bit: What the highest relative ranking was in any MS1 scan for a given mass. Ie, was it ever in the top 10 and therefore selectable for MS2? Will probably have to collect this after the peakfinding bit. Highest ranking for apex too.

#Peak Transformations:
# - Minpoint reduction
# - Boxcar averaging
#       > You can have an n=3 automatically, and test by the same maximum-based parameter that you usually do in minpoint reduction. If there's too many maxes, do another cycle of averaging.
# - Savgol??? hah ya right

#collect # of datapoints within X% of the maximum value, scale for every 10 %'s or so
#^these %'s and # of points to either the left or right of the maximum
#distribution of intensity by distance from center on either side, ie 90% of total intensity is within 40% distance - for left of center, idk.

#Input
#[Peak Splits] - number of parts to split the peak into when analyzing it. Splits break the peak up by intensity, surrounding the apex. Splitting into 2 parts would only collect info from half-height. Input for this must be even.
#   > Peak splits can be done both by index/scan(both??) length and by intensity, and both of these bits will give insightful information about the peak, and how the peakfinding process performed on the data.
#   > The splits along index/scan can also be used as normalization points for the end-game quantification
#   > The problem that arises with splits is what datapoint to take with some sometimes-noisy data. There might be multiple viable points at multiple locations when splitting intensity rather than distance.
#       >> One solution is simpler, to always choose the datapoint farthest from the apex. Asssuming anything that's farther away is less likely to be noise of a lesser value.
#       >> Another solution is to consider boxcar means as being those splits. Wherever the mean of n surrounding points ~= the desired intensity, that's the point to take.
#When splitting along intensity, collect the number of times that the data crosses each split. Ie, at half-height, does the data cross this 4 times? 20 times? and note (total crosses / n data points)

#Generating Peak Splits
#For space-wise splits:
# - Interpolate a number, divisible by the # of peak splits, of even linear intermediates along the peak's "route" along the chromatogram axis as time.
#For area-wise splits:
# - Just divide by n, duh
#Splits maybe shouldn't be done by max, but instead by % area. The idea of half-max can be flawed if you pass the half-max point more than once. It becomes annoying to deal with I suppose.
#for area: do the left and right sides independent of one another, each taking 10, 20 or whatever % of their own respective side.

#Peak Mechanics, to be done on raw and masked peaks
#A - number of points left/right of max
#B - ratio of the distance (time) at each [Peak Splits] on either side from the apex.
#   ^ Also collect:
#       > what % of total intensity, area, and datapoints lies in between these two points
#           ^ If I just took a prominence measurement at these same points, wouldn't this relay the same information with less hassle?
#           - a post-pwf correction can combine peaks with uniform prominence at the same mass range. And I could also use this potentially to correct other mistakes too, like 2 maxes found on 1 peak
#           - A decreasing prominence in the middle of a peak could also indicate a double-peak that didn't have both maxes found -> potential post-processing here
#   ^ In the case of noisy signals in raw data that contain more than one eligible point, it will be the farthest data points from the apex that are used for this purpose.

#collected info -> generated info
#A -> how centered the peak max is
#A + B -> differentiating truly asymetric peaks from peaks with shoulders/tails, for example if a peak has a very asymetrical number of data points on either side, but the ratio in B is 1, then the peak is assymetrical not due to a weird shape, but due to a shoulder or tail.

#The generated info can create distributions upon which to separate, via brute combinatorics, layers of a tree where each distribution is split into another distribution at each node to find a way to filter out peaks that really aren't peakish.
#The generated info would be cross-checked across clusters generated via peak shape.

#Plotting: each peak width at each intensity-split on the x-axis, number of peaks at that width on the y axis, and the chromatogram on the x-axies.
#^ not a perfect explanation, WIP, but something along these lines.

#Peak finding completion:
#Sum of intensities found in peaks / sum of intensity across the data file
#N data points found in peaks / n data points across in the data file

#Post isotope evaluation:
#Determining optimal isolation windows based on likelyhood of ms2 overlap

#For giving attitude:
#give = print, give(attitude) ta~da!
#if peakfill == 1: You think you're too good for peaks with zeros in them? I bEt YoU hAvE a PhD, please tell me all about your experience handling large-scale chromatographic data.
#^can this print out the spongebob pic?
#also insult them, then make it wait for 10 seconds to see if it will (50/50) either insult them again, or actually start the script
