import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pickle

plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.facecolor'] = 'gray'
plt.rcParams['figure.facecolor'] = 'gray'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['ytick.labelcolor'] = 'white'
plt.rcParams['xtick.labelcolor'] = 'white'

def max_finding(array):
    forwardmaxcheck = np.append(array[:-1] > array[1:], False)
    #backwardmaxcheck = np.append(False, array[1:] > array[:-1])
    backwardmaxcheck = np.append(forwardmaxcheck[0], array[1:] > array[:-1])
    forwardmaxcheck[-1] = backwardmaxcheck[-1]
    maxes = np.where(forwardmaxcheck & backwardmaxcheck)[0]
    return maxes


#with open('/home/sfo/camp/fdpick.pickle', "rb") as pick:
#    endarray = pickle.load(pick)

npeaks = 50
minpoints = 30
maxpoints = 2000
maxloc = 25000 #ends up being rough, not exact

endarray = []
for n in range(npeaks):
    xvalue = np.random.uniform(-maxloc, maxloc)
    rpoints = np.random.randint(minpoints, maxpoints)
    rstdev = np.random.randint(minpoints, rpoints)
    peak = np.random.normal(xvalue, rstdev, size=rpoints)
    endarray.extend(peak.tolist())
endarray = np.sort(endarray)

#1. order array
#2. iterate through diffs
#3. if diff is greater, density value decreases, - 1/dist
#4. if diff is lower, density value increases, + 1/dist
#5. plot values on x and density on y
#increase/decrease needs to be symmetrical in a way, I'm thinking a forward/backward pass sum
#density is derived from the distances between distances

endarray = endarray[endarray < 1.2]
endarray = np.sort(endarray)
tdiffs = np.diff(endarray, prepend=0)

plt.hist(endarray, color='white', bins=100)
plt.show()

plt.plot(endarray, tdiffs, color='white')
plt.show()

plt.plot(endarray, 1/tdiffs, color='white')
plt.show()

npoints = len(endarray)
evensplits = np.linspace(endarray.min(), endarray.max(), npoints)
splitpairs = np.stack((evensplits[:-1], evensplits[1:]), axis=1)
densitycount = []
densitymeasures = []
for l, r in splitpairs:
    spanbool = np.logical_and(endarray >= l, endarray <= r)
    spanpoints = endarray[spanbool]
    spandiffs = tdiffs[spanbool]
    spancount = spanbool.sum()
    densityestimate = spanpoints.sum() / spandiffs.sum()
    densitycount.append(spancount)
    densitymeasures.append(densityestimate)

densityvals = Counter(densitycount)
keys = np.array(list(densityvals.keys()))
values = np.array(list(densityvals.values()))
cutkey = keys[values < values.mean()][values[values < values.mean()].argmax()]

densitycount = np.array(densitycount)
densitybool = densitycount >= cutkey
dmaxes = max_finding(densitycount)

plt.bar(densityvals.keys(), densityvals.values(), color='white')
plt.show()

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

plt.plot(endarray[:-1], densitycount, '.', color='white')
plt.plot(endarray[:-1][densitybool], densitycount[densitybool], '.', color='salmon')
plt.show()
