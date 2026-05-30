import numpy as np
from scipy import integrate, signal, stats
from matplotlib import pyplot as plt
import os
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from collections import Counter, defaultdict
from time import time

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

npeaks = 100
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

#total distributed values per iteration (votes) = # points
#points / range, (optional: range * weights) = voting factor for distribution of votes
#vote sum normalized to # points each iteration -> this is the permanent value added to that range
#distribution of post-voted values within the group is determined by summing the distance of each point to every other point, each point gets its absolute distance to every other point summed, and the points with the lowest total distances get more value from the post-voted value. This is the second vote, and its normalized by the amount delegated by the first vote.
#^so this process returns a density value at each original point that was handed into the function - no expansions or extrapolated representations.
#algo only starts when a rounding combines two points
#only add density counts to things with more than one point? try it first without worrying about that I suppose
#actually it may still be wise to make an expansion/extrapolation based on the final data for display purposes, it probably can be done and is optional.

nt = time()
datalength = len(endarray)
maxdec = max(len(str(i).split('.')[1]) for i in endarray)
maxwhole = max(len(str(i).split('.')[0]) for i in endarray) #don't really care if it might include a negative sign
endarray = np.sort(endarray)

endwholes = np.unique(endarray.round())
if endwholes.size < datalength:
    #decimals have relevance
    begiter = maxdec
else:
    begiter = 0
if endwholes.size > 0:
    #whole numbers have relevance
    enditer = -maxwhole
else:
    enditer = 0

#this could probably be made cooler by assessing the distances of everything to everything initially, then calculating the would-be awarded values at each distance for each group, and you could also change the scaling metric here -> instead of a base10 rounding you could make it base2 ie every single digit matters, or would that make it a doubling?
valuedistributions = defaultdict(float) #to be used to derive exansions/extrapolations later
deciter = reversed(range(enditer, begiter))
for dec in deciter:
    movingarray = endarray.round(dec)
    umoves, ucounts = np.unique(movingarray, return_counts=True)
    if umoves.size == 1:
        break
    if umoves.size < datalength:
        #to make the total available value either ucounts[uinds].sum() or datalength?
        uinds = np.where(ucounts > 1)[0]
        distributablevotes = ucounts[uinds].sum()
        #distributablevotes = datalength
        valuedists = defaultdict(float)
        groupweights = {}
        valuecounts = 0
        for ui in uinds.tolist():
            minds = np.where(umoves[ui] == movingarray)[0]
            groupvalues = endarray[minds]
            grouprange = np.ptp(groupvalues)
            mlen = minds.size
            groupweights[tuple(groupvalues)] = mlen / grouprange #group weight
            valuecounts += mlen
            for gv in groupvalues.tolist():
                #going to avoid using matrices in case of memory errors
                valuedists[gv] += np.abs(gv - groupvalues).sum()
        vals, weights = zip(*groupweights.items())
        vlens = list(map(len, vals))
        weights = np.array(weights)
        voteassignments = weights / (weights.sum() * valuecounts)
        #voteassignments = weights / (weights.sum() * datalength)
        distributablevotes = voteassignments * vlens
        for va, dvs in zip(vals, distributablevotes):
            valueweights = dvs / np.array([valuedists[i] for i in va])
            #valuecontribution = (valueweights / valueweights.sum() * dvs).tolist()
            #valuecontribution = (valueweights / valueweights.max() * dvs).tolist()
            valuecontribution = (valueweights / valueweights.max()).tolist()
            #valuecontribution = valueweights.sum() / len(va)
            for v, vd in zip(va, valuecontribution):
            #for v in va:
                valuedistributions[v] += vd
                #valuedistributions[v] += valuecontribution
print(time() - nt, 'floating')

xv, yv = zip(*valuedistributions.items())
xv = np.array(xv)
yv = np.array(yv)
yv = yv[xv.argsort()]
xv = np.sort(xv)

lsize = endarray.size
#lsize = 500
lx = np.linspace(endarray.min(), endarray.max(), lsize)
ly = np.zeros(lsize)

for rx, ry in zip(xv.tolist(), yv.tolist()):
    xloc = np.where(np.logical_and(lx[:-1] <= rx, rx <= lx[1:]))[0]
    for xl in xloc.tolist():
        ly[xl] += ry
print(time() - nt)

kt = time()
karray = np.array(endarray)[:,None]
kplot = np.linspace(karray.min(), karray.max(), 1000)[:,None]
kde = KernelDensity(kernel='gaussian', bandwidth=100).fit(karray)
#bandwidth = np.linspace(10, 100, 10)
#grid = GridSearchCV(kde, {'bandwidth': bandwidth})
#grid.fit(karray)
#kde = grid.best_estimator_
log_density = kde.score_samples(kplot)
print(time() - kt, 'kde')

fig, ax = plt.subplots(figsize=(7,4))

tx = ax.twinx()
tx.plot(kplot, np.exp(log_density), '-', color='orangered', linewidth=0.5)

nx = tx.twinx()
nx.plot(lx, ly, '-', color='cyan', linewidth=0.5, alpha=0.5)

ax.hist(endarray, bins=500, color='white', alpha=1)

align_yaxis_np(ax, nx)
plt.show()

#treat density by decimals as a type of local heirarchical clustering
#or, with the current scheme, the smoothing would essentially only keep specific points: ones that match the direction of the following timepoint, and the optimization would be to figure out the minimum number of following timepoints to apply

#everything in ly spreads to its neighbors
#everything spreads outward by a rate determined by its % of the sum
#the amount it spreads is total, so if it spreads in 2 directions as opposed to 1, then the total is spread across the 2

#output:
#number of datapoints should be spread evenly across the range of the data
