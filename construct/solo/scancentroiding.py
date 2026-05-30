from collections import defaultdict
from functools import partial
from pyteomics import mzml
from time import time
import numpy as np
import pickle

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
    while True:
        narray = barray[~mask]

        forwardmaxcheck = np.append(narray[:-1] > narray[1:], False)
        backwardmaxcheck = np.append(forwardmaxcheck[0], narray[1:] > narray[:-1])
        forwardmaxcheck[-1] = backwardmaxcheck[-1]

        forwardmincheck = np.append(narray[:-1] < narray[1:], False)
        backwardmincheck = np.append(forwardmincheck[0], narray[1:] < narray[:-1])
        forwardmincheck[-1] = backwardmincheck[-1]

        newmask = np.logical_and(forwardmincheck, backwardmincheck)
        mins = np.where(newmask)[0]
        maxes = np.where(np.logical_and(forwardmaxcheck, backwardmaxcheck))[0]
        extremas = np.sort(np.append(mins, maxes))
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

def axis_peaks(array, mindist=0):
    maxes = minpoint_reduction(array, mindist)
    peakparameters = boundary_finding(maxes, array) #reverted the r + 1 into this function
    peakparameters = peakparameters.tolist()
    return peakparameters #[[left bound, max, right bound],]

def scan_centroiding(centroiding, scan, intensitytype='area', masstype='average'):
    #masstype can be average (weighted) or max
    #intensitytype can be sum, max, or area
    scanind = scan['index']
    rt = scan['scanList']['scan'][0]['scan start time'].real
    mslevel = scan['ms level']
    if mslevel in centroiding:
        mza = scan['m/z array']
        intensityarray = scan['intensity array']

        peakparameters = axis_peaks(intensityarray, mindist=0) #get rid of mindist later i guess
        masses, intensities = [], []
        for l, m, r in peakparameters:
            pm = mza[l:r]
            pi = intensityarray[l:r]
            if masstype == 'average':
                isum = 0
                msum = 0
                for i, m in zip(pi, pm):
                    isum += i
                    msum += m * i
                mass = msum / isum
            elif masstype == 'max':
                mass = pm[pi.argmax()]
            if intensitytype == 'area':
                intensity = np.trapezoid(pi, pm)
            elif intensitytype == 'max':
                intensity = pi.max()
            elif intensitytype == 'sum':
                intensity == pi.sum()
            masses.append(mass)
            intensities.append(intensity)
        centroidvals = {
                'm/z array': np.array(masses),
                'intensity array': np.array(intensities)
                 }
    else:
        centroidvals = {
                'm/z array': scan['m/z array'],
                'intensity array': scan['intensity array']
                 }
    return mslevel, rt, scanind, centroidvals

def centroid(mzmlfile, processingdirectory, centroiding, intensitytype, masstype):
    nt = time()

    msrun = mzml.MzML(mzmlfile, dtype=np.float64)

    scan_centroiding_partial = partial(scan_centroiding, centroiding, intensitytype=intensitytype, masstype=masstype)

    retentiontimesbyscan = {} #scan: rt
    datalevels = defaultdict(dict) #mslevel: scan: {masses: [], intensities: []}
    for mslevel, rt, scanind, centroidvals in msrun.map(scan_centroiding_partial):
        datalevels[mslevel][scanind] = centroidvals
        retentiontimesbyscan[scanind] = rt

    for mslevel, centroidscans in datalevels.items():
        centroidscans = dict(sorted(centroidscans.items()))
        mslevelstring = 'centroid.ms' + str(mslevel) + '.pickle'
        mslevelfile = ''.join((processingdirectory, mslevelstring))
        with open(mslevelfile, 'wb') as pick:
            pickle.dump(centroidscans, pick)

    retentiontimesbyscanfile = ''.join((processingdirectory, 'retentiontimesbyscan.pickle'))
    with open(retentiontimesbyscanfile, 'wb') as pick:
        pickle.dump(retentiontimesbyscan, pick)

    print('found', len(datalevels), 'ms levels')
    print(time() - nt, 'centroiding')
