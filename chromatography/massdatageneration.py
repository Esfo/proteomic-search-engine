import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
from scipy import signal, stats
from collections import Counter
from sklearn.neighbors import NearestNeighbors
import networkx
from networkx.algorithms.components.connected import connected_components
from time import time
plt.rcParams["figure.dpi"] = 300

nneighbors = 5 #number of data points included in the nearest neighbor calculation for peak finding, any group of indices being the center of nneighbors + 1 incoming nearest neighbor interactions is considered for peak-finding
scanbuffer = 10 #min number of scan indices allowed between points to be considered as part of the same peak or group of peaks.
minmeasurements = 8 #minimum number of non-zero datapoints in any collected chromatographic peak
npeaks = 10
rminpoints = 15
rmaxpoints = 200
noiselevel = 0.1
maxchromstd = 45
maxchrommultiplier= 1e7
#maxdisplacement = 45
#mindisplacement = 5
#lengthratiomin = 2
#lengthratiomax = 50
xdistrange = 2000
ydistrange = 500
rpeakfill = 0.9
addnoise = True
noisefill = 0.0001
noisethreshold = 0.01 #% of max in window
gridoverlapthreshold = 0.05

boundmin = 2
boundmax = 300
mumin = 0
mumax = 100
stdmin = 3
stdmax = 5

chrommindist = 5 #this is an index distance, min distance between allowable apexes
chromboxcarlength = 5 #n data points on either side of a 0 point to use in an averaging for both max and boundary finding, total points used is this number x2. Any of the encompassed data points that fall below this groups average are masked prior to any processes.
massmindist = 5
massboxcarlength = 5
minpoints = 10
peakfill = 0.7
#make fusion peaks, add more than 1 window at a specific location

starttime = time()
#notes:
#the only legitemate failures stem from the maxes not being properly elucidated in minpoiint_reduction. This can have varying causes. The major cause of this failing is actually when some of the random data generation puts to many 0s next to each other - this wouldn't happpen in real data to the best of my understanding. Other times there are non-gaussian blobs that are next to good-looking peaks that don't get picked up as easily. Meh, it happens. It's not the most realistic data in that form anyways, the blobs are not my worry.

def boolcount(b, counts=False):
    bc = np.where(np.diff(b, prepend=True))[0]
    bc = np.append(bc, len(b))
    if counts:
        bc = np.diff(bc, prepend=0)
    if bc.size % 2:
        bc = np.append(bc, 0)
    return bc.reshape(bc.size//2,2)

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

def whereless_locate(check, axis):
    acheck = check.any(axis=axis)
    for n, a in enumerate(acheck):
        if a:
            v = n
            return v

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
                subtractor = scanbuffer - whereless_locate(topcheck, 1)
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
                adder = whereless_locate(bottomcheck, 1) + 1
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
                subtractor = scanbuffer - whereless_locate(leftcheck, 0)
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
                adder = whereless_locate(rightcheck, 0) + 1
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

def minpoint_reduction(array, mindist):
    #this approach fails when the maxes are at the absolute beginning or end of an array
    extramaxes = set()
    mask = np.repeat(False, array.size)
    while True:
        narray = array[~mask]

        forwardmaxcheck = np.append(narray[:-1] > narray[1:], False)
        backwardmaxcheck = np.append(False, narray[1:] > narray[:-1])
        #backwardmaxcheck = np.append(forwardmaxcheck[0], narray[1:] > narray[:-1])
        #forwardmaxcheck[-1] = backwardmaxcheck[-1]

        forwardmincheck = np.append(narray[:-1] < narray[1:], False)
        backwardmincheck = np.append(False, narray[1:] < narray[:-1])
        #backwardmincheck = np.append(forwardmincheck[0], narray[1:] < narray[:-1])
        #forwardmincheck[-1] = backwardmincheck[-1]

        newmask = np.logical_and(forwardmincheck, backwardmincheck)
        mins = np.where(newmask)[0]
        maxes = np.where(np.logical_and(forwardmaxcheck, backwardmaxcheck))[0]
        extremadistances = (np.abs(maxes - mins.reshape(-1,1)) < mindist)
        
        maxestoholdonto = maxes[~extremadistances.any(axis=0)]
        if maxestoholdonto.size > 0:
            maxestoholdonto = (maxestoholdonto + mask.cumsum()[~mask][maxestoholdonto]).tolist()
            extramaxes.update(maxestoholdonto)
        
        adjacentextremas = extremadistances.any()
        if adjacentextremas:
            maskinds = np.argwhere(~mask)[np.argwhere(newmask)].flatten()
            mask[maskinds] = True
        else:
            break
    
    if not maxes.size:
        maxes = narray.argmax() #this seems like an easier way to allow for maxes at the first or last point
    
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

def boxcar_mask(array, boxcarlength):
    scans = np.arange(len(array))
    indvector = np.linspace(scans-boxcarlength, scans+boxcarlength, num=boxcarlength*2+1, axis=1).astype(int)
    indvector[indvector < 0] = 0
    indvector[indvector >= len(array) - 1] = len(array) - 1
    filtermeans = array[indvector].mean(axis=1)
    minfiltermeans = filtermeans[indvector].min(axis=1) #applying the lowest mean that each individual point is involved with
    filterboolmap = array < minfiltermeans
    filteredinds = scans[filterboolmap]
    noisemask = np.repeat(True, len(array))
    noisemask[filteredinds] = False
    noisemask[0] = True
    noisemask[-1] = True
    return noisemask

def boxcar_mean_replacement(array, boxcarlength):
    scans = np.arange(len(array))
    indvector = np.linspace(scans-boxcarlength, scans+boxcarlength, num=boxcarlength*2+1, axis=1).astype(int)
    indvector[indvector < 0] = 0
    indvector[indvector >= len(array) - 1] = len(array) - 1
    filtermeans = array[indvector].mean(axis=1)
    minfiltermeans = filtermeans[indvector].min(axis=1) #applying the lowest mean that each individual point is involved with
    filterboolmap = array <= minfiltermeans
    array[filterboolmap] = filtermeans[filterboolmap]
    return array

def axis_peaks(array, boxcarlength, mindist):
    boxcararray = boxcar_mean_replacement(array, boxcarlength)
    maxes = minpoint_reduction(boxcararray, mindist)
    peakparameters = boundary_finding(maxes, boxcararray)
    
    peakparameters = peakparameters.tolist()
    
    #defensive, I forget if it's necessary
    for n in range(len(peakparameters)):
        l, m, r = peakparameters[n]
        m = l + array[l:r].argmax()
        peakparameters[n][1] = m
    
    #peakparameters = np.asarray(peakparameters)
    return peakparameters

def max_finding(narray):
    #forwardmaxcheck = np.append(narray[:-1] > narray[1:], False)
    #backwardmaxcheck = np.append(False, narray[1:] > narray[:-1])
    forwardmaxcheck = np.append(narray[:-1] > narray[1:], False)
    backwardmaxcheck = np.append(forwardmaxcheck[0], narray[1:] > narray[:-1])
    forwardmaxcheck[-1] = backwardmaxcheck[-1]
    maxes = np.where(forwardmaxcheck & backwardmaxcheck)[0]
    return maxes

def min_finding(narray):
    #forwardmincheck = np.append(narray[:-1] < narray[1:], False)
    #backwardmincheck = np.append(False, narray[1:] < narray[:-1])
    forwardmincheck = np.append(narray[:-1] < narray[1:], False)
    backwardmincheck = np.append(forwardmincheck[0], narray[1:] < narray[:-1])
    forwardmincheck[-1] = backwardmincheck[-1]
    mins = np.where(forwardmincheck & backwardmincheck)[0]
    return mins

#this guy was helpful: https://stackoverflow.com/questions/69089508/vectorizing-a-rectangle-overlap-determination-in-numpy/69090650#69090650
def rectangle_overlap(recs):
    tops, bottoms, lefts, rights = recs.transpose()
    c1 = lefts < rights[:, None]
    c2 = rights > lefts[:, None]
    c3 = tops < bottoms[:, None]
    c4 = bottoms > tops[:, None]
    overlaps = np.argwhere(c1 & c2 & c3 & c4)
    return overlaps

window = np.zeros((1,1))
for p in range(npeaks):
    points = np.random.randint(rminpoints, rmaxpoints)
    chrompoints = np.random.randint(rminpoints, rmaxpoints)
    chromstd =  np.random.randint(1, maxchromstd)
    chrommultiplier = np.random.randint(1, maxchrommultiplier)
    noisemultiplier = np.random.uniform(low=1-noiselevel, high=1+noiselevel, size=points)
    chrom = signal.windows.gaussian(points, std=chromstd, sym=False) * chrommultiplier * noisemultiplier
    
    wlen = len(chrom)
    
    if rpeakfill < 1:
        zerofills = np.random.choice(np.arange(wlen), size=np.random.randint(0, round(wlen*(1-rpeakfill))))
        chrom[zerofills] = 0
    
    
    #massstd = np.random.randint(minmassstd+1, maxmassstd)
    #displacements = np.random.randint(minmassstd, massstd, size=points)
    #dmean = displacements.mean().round().astype(int)
    #basedisplacement = np.random.randint(mindisplacement, maxdisplacement) #this should actually be the -/+ versions of the same number, otherwise you're not going in the negative direction at all.
    #lengthratio = np.random.uniform(lengthratiomin, lengthratiomax)
    #widthfactor = round(points / lengthratio)
    #if widthfactor < 1:
    #    widthfactor = 1
    #the intensity of an individual datapoint will be the chance of a width distribution being smaller, the chance of that not happening will be the sum of the rest of the window
    
    upper = np.random.uniform(low=boundmin, high=boundmax)
    lower = np.random.uniform(low=boundmin*0.99, high=upper*0.99)
    mu = np.random.uniform(low=mumin, high=mumax)
    sigma = np.random.uniform(low=stdmin, high=stdmax)
    
    a = (lower - mu) / sigma
    b = (upper - mu) / sigma
    chromdisplacements = stats.truncnorm(a, b, loc=mu, scale=sigma).rvs(points)
    
    #attempt #1
    #orderedchrom = np.sort(chrom)
    #ordereddisplacements = np.flip(np.sort(displacements))
    #chromdisplacements = []
    #for w in chrom:
    #    #std = ordereddisplacements[np.argwhere(orderedchrom == w)[-1,0]]
    #    #dis = np.random.randint(dmean-std, dmean+std)
    #    std = basedisplacement + np.random.randint(0, widthfactor)
    #    chromdisplacements.append(dis)
    #chromdisplacements = np.array(chromdisplacements)
    
    #attempt #2
    #displaces = signal.windows.gaussian(points, std=lengthratio, sym=False)
    #displaces[displaces.argmax():] *= -1
    #displaces += abs(displaces.min())
    #np.random.shuffle(displaces)
    #chromdisplacements = basedisplacement + np.random.randint(0, widthfactor, size=wlen)

    #if min(chromdisplacements) < 0:
    #    chromdisplacements += abs(min(chromdisplacements))
    
    chromorder = chromdisplacements.argsort()
    #mlen = max(chromdisplacements)
    mlen = len(np.unique(chromorder))
    
    windowsection = np.zeros((mlen, wlen))
    xpos = np.arange(mlen)
    #ypos = chrom.argsort()
    #ypos = ypos.max() - ypos
    windowsection[chromorder, xpos] = chrom
    #for n, (d, i) in enumerate(zip(chromdisplacements, chrom)):
    #    windowsection[d-1,n] += i
    
    
    if window.shape[1] < xdistrange:
        xdist = np.random.randint(-xdistrange, xdistrange)
    else:
        xdist = np.random.randint(-window.shape[1] - xdistrange, xdistrange)
    if window.shape[0] < ydistrange:
        ydist = np.random.randint(-ydistrange, ydistrange)
    else:
        ydist = np.random.randint(-window.shape[0] - ydistrange, ydistrange)

    
    if xdist >= 0:
        #extend window in x-direction on the back end by wlen + xdist
        additionalpoints = wlen + xdist
        zeros = np.zeros((window.shape[0], additionalpoints))
        window = np.hstack((window, zeros))
        left = window.shape[1] - wlen
        right = window.shape[1]
    else: #xdist < 0
        if abs(xdist) > window.shape[1]:
            if wlen > abs(xdist):
                ladditionalpoints = abs(xdist + window.shape[1])
                radditionalpoints = wlen - (window.shape[1] + ladditionalpoints)
                lzeros = np.zeros((window.shape[0], ladditionalpoints))
                rzeros = np.zeros((window.shape[0], radditionalpoints))
                window = np.hstack((lzeros, window, rzeros))
                left = 0
                right = wlen
            else:
                additionalpoints = abs(xdist + window.shape[1])
                zeros = np.zeros((window.shape[0], additionalpoints))
                window = np.hstack((zeros, window))
                left = 0
                right = wlen
        else:
            if wlen > abs(xdist):
                additionalpoints = wlen + xdist
                zeros = np.zeros((window.shape[0], additionalpoints))
                window = np.hstack((window, zeros))
                left = window.shape[1] - wlen
                right = window.shape[1]
            else: #chrom is added within current window
                left = window.shape[1] + xdist
                right = left + wlen
    
    #extending along mass axis
    if ydist >= 0:
        #extend window in x-direction on the back end by mlen + ydist
        additionalpoints = mlen + ydist
        zeros = np.zeros((additionalpoints, window.shape[1]))
        window = np.vstack((window, zeros))
        top = window.shape[0] - mlen
        bottom = window.shape[0]
    else: #ydist < 0
        if abs(ydist) > window.shape[0]:
            if mlen > abs(ydist):
                ladditionalpoints = abs(ydist + window.shape[0])
                radditionalpoints = mlen - (window.shape[0] + ladditionalpoints)
                lzeros = np.zeros((ladditionalpoints, window.shape[1]))
                rzeros = np.zeros((radditionalpoints, window.shape[1]))
                window = np.vstack((lzeros, window, rzeros))
                top = 0
                bottom = mlen
            else:
                additionalpoints = abs(ydist + window.shape[0])
                zeros = np.zeros((additionalpoints, window.shape[1]))
                window = np.vstack((zeros, window))
                top = 0
                bottom = mlen
        else:
            if mlen > abs(ydist):
                additionalpoints = mlen + ydist
                zeros = np.zeros((additionalpoints, window.shape[1]))
                window = np.vstack((window, zeros))
                top = window.shape[0] - mlen
                bottom = window.shape[0]
            else: #chrom is added within current window
                top = window.shape[0] + ydist
                bottom = top + mlen
    
    window[top:bottom,left:right] += windowsection

originalwindow = window.copy() #can measure noise reduction effectiveness

if addnoise:
    nmax = window.max() * noisethreshold
    nspots = (np.prod(window.shape) * noisefill).astype(int)
    xnoise = np.random.randint(0, window.shape[1], nspots)
    ynoise = np.random.randint(0, window.shape[0], nspots)
    noiseintensity = np.random.uniform(0, nmax, nspots)
    window[ynoise,xnoise] = noiseintensity

print(time() - starttime)

inds = np.argwhere(window > 0)

#fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
#ax.plot(inds[:,1], inds[:,0], '.', markersize=0.5, color='teal')
#plt.show()

#Random noise reduction
nbrs = NearestNeighbors(n_neighbors=2, metric='euclidean', algorithm='auto').fit(inds) #ditching the brute method for auto, massive speedup when using a larger windowsize. It pro
dists, distinds = nbrs.kneighbors(inds)

#finding reciprocating nearest neighborship
#aka finding number of nearest neighbors that also consider a given indice one of its nearest neighbor
#ninteractions = (distinds[distinds] == distinds[:,0][:,None,None]).sum(axis=(1,2)) - 1

#tbh this is a sloppy idea imo, but I think it will work out in actuality
#filtering at the first point that dips below the mean
dcounts = Counter(dists[:,1].round().astype(int).tolist())
#dcounts = Counter(dists.sum(axis=1).round().astype(int))
dkeys = list(dcounts.keys())
sortorder = np.asarray(dkeys).argsort()
dvals = np.asarray(list(dcounts.values()))
dvals[dvals <= dvals.mean()] = 0
cutind = np.argwhere(dvals[sortorder] <= dvals.mean())[0][0]
cutkey = sorted(dkeys)[cutind] + 1
#scanbuffer = cutkey * 2 #a little arbitrary I suppose

#noiseinds = inds[dists[:,1] > cutkey]
noiseindmask = dists[:,1] > cutkey
noiseinds = inds[noiseindmask]
window[noiseinds[:,0], noiseinds[:,1]] = 0 #do I even need to do this???, Yea..

#newinds = np.argwhere(window > 0) #slow af
newinds = inds[~noiseindmask]
nbrs = NearestNeighbors(n_neighbors=nneighbors, metric='euclidean', algorithm='auto').fit(newinds)
dists, distinds = nbrs.kneighbors(newinds)

#make a different variable for the noise-reduced window when you decide to use this
#effects and distriutions can be visualized:
#plt.bar(dcounts.keys(), dcounts.values())
#plt.hlines(dvals.mean(), 0, max(dkeys), color='black', linewidth=0.3)
#plt.vlines(cutkey, 0, max(dvals), color='black', linewidth=0.3)
#plt.title('Distribution of Nearest Neighbor Distance')
#plt.ylabel('Number of Points with Distance $x$')
#plt.xlabel('Integer Distance from Nearest Neighbor')
#plt.yscale('log')
#plt.show()
#plt.imshow(window, cmap='GnBu', vmin=0, vmax=1, aspect='auto')
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
filterdists = distinds[distsums < dmean]
rings = generic_meta_overlap(filterdists.tolist())
expandedrings = ring_expansion(window, rings, distinds, newinds, scanbuffer)

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
#fs = ''.join((basefolder, str(n), '.png'))
#for t, b, l, r in ringregions:
#    width = timearray[r-1] - timearray[l]
#    #width = r - l
#    height = masses[b-1] - masses[t]
#    #height = b - t
#    rect = patches.Rectangle((timearray[l], masses[t]), width, height, linewidth=2, edgecolor='red', facecolor='none')
#    #rect = patches.Rectangle((l, t), width, height, linewidth=2, edgecolor='red', facecolor='none')
#    ax.add_patch(rect)
#ax.plot(timearray[inds[:,1]], masses[inds[:,0]], '.', markersize=0.5, color='crimson', alpha=0.5)
#ax.plot(timearray[newinds[:,1]], masses[newinds[:,0]], '.', markersize=0.5, color='indigo', alpha=0.5)
##ax.plot(inds[:,1], inds[:,0], '.', markersize=0.5, color='crimson', alpha=0.5)
##ax.plot(newinds[:,1], newinds[:,0], '.', markersize=0.5, color='indigo', alpha=0.5)
#plt.title(' - '.join((str(coordwindow - windowsize), str(coordwindow))))
#plt.xlabel('minutes')
#plt.ylabel('mass')
#fig.savefig(fs, facecolor='white', transparent=False)
#plt.close("all")
#gc.collect()
##plt.show()

collectedpeaks = []
for t, b, l, r in ringregions:
    grid = window[t:b,l:r]
    if grid.any(axis=0).sum() >= minpoints:
        #scaninds = scanarray[l:r]
        #massinds = masses[t:b]
        
        #I actually wonder if doing this would be a more effective method of mass-peak separation...
        gridoverlapcounts = ((grid > 0).sum(axis=0) > 1).sum() #checking for multiple separate mass peaks being in grid
        #using a straight-forward approach rather than a recursive one atm, might switch in the future, might not need to. The recursive idea has a shortcoming on where to place pre-existing confines for newer peaks, and how to expand it should it need to be expanded.
        if gridoverlapcounts / grid.shape[1] > gridoverlapthreshold: #starting from mass axis
            fullmass = grid.sum(axis=1)
            masspeaks = axis_peaks(fullmass, massboxcarlength, massmindist)
            for mp in masspeaks:
                fullchrom = grid[mp[0]:mp[2]].sum(axis=0)
                if (fullchrom > 0).sum() > minpoints:
                    chrompeaks = axis_peaks(fullchrom, chromboxcarlength, chrommindist)
                    for cp in chrompeaks:
                        if cp[2] - cp[0] > minpoints:
                            cdpoints = (fullchrom[cp[0]:cp[2]] > 0).sum()
                            cfill = cdpoints / (cp[2] - cp[0]) >= peakfill
                            if cfill and cdpoints > minpoints:
                                peakgrid = grid[mp[0]:mp[2],cp[0]:cp[2]]
                                pgdata = peakgrid > 0
                                gridfail = (pgdata > 1).sum()
                                ndatapoints = pgdata.sum()
                                mfill = ndatapoints / (cp[2] - cp[0]) >= peakfill
                                if mfill and ndatapoints > minpoints:
                                    mp = [i + t for i in mp]
                                    cp = [i + l for i in cp]
                                    collectedpeaks.append(mp + cp)
                                else:
                                    pass
                                if gridfail:
                                    print('mass axis failure', coordwindow-windowsize, mp, cp, f'({t}, {b}, {l}, {r})')
                            else:
                                pass
        else: #starting from chrom axis, this route it chosen ~90-95% of the time
            fullchrom = grid.sum(axis=0)
            chrompeaks = axis_peaks(fullchrom, chromboxcarlength, chrommindist)
            for cp in chrompeaks:
                cdpoints = (fullchrom[cp[0]:cp[2]] > 0).sum()
                cfill = cdpoints / (cp[2] - cp[0]) >= peakfill
                if cfill and cdpoints > minpoints:
                    masspeaks = axis_peaks(grid[:,cp[0]:cp[2]].sum(axis=1), massboxcarlength, massmindist)
                    mcount = 0
                    for mp in masspeaks:
                        peakgrid = grid[mp[0]:mp[2],cp[0]:cp[2]]
                        ndatapoints = (peakgrid > 0).sum()
                        mfill = ndatapoints / (cp[2] - cp[0]) >= peakfill
                        if mfill and ndatapoints > minpoints:
                            mp = [i + t for i in mp]
                            cp = [i + l for i in cp]
                            collectedpeaks.append(mp + cp)
                        else:
                            pass
                    if mcount > 1:
                        print('chrom axis failure', coordwindow-windowsize, mp, cp, f'({t}, {b}, {l}, {r})')
                else:
                    pass


print(time() - starttime, '- ringregions')
tinds = np.argwhere(originalwindow > 0)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6))

ax.plot(inds[:,1], inds[:,0], '.', markersize=0.5, color='crimson', alpha=0.3)
ax.plot(newinds[:,1], newinds[:,0], '.', markersize=0.5, color='indigo', alpha=0.3)
ax.plot(tinds[:,1], tinds[:,0], '.', markersize=0.5, color='teal', alpha=0.3)
for nt, m1, nb, nl, m2, nr in collectedpeaks:
#for nt, nb, nl, nr in ringregions:
    width = nr - nl
    height = nb - nt
    rect = patches.Rectangle((nl, nt), width, height, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

plt.show()
print(len(collectedpeaks), '/', npeaks, 'peaks found')

for winner in collectedpeaks:
    peakgrid = window[winner[0]:winner[2],winner[3]:winner[5]]
    peakchrom = peakgrid.sum(axis=0)
    peakmass = peakgrid.sum(axis=1)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6,4))
    ax[0].plot(peakchrom, '.-')
    ax[1].plot(peakmass, '.-')
    plt.show()
