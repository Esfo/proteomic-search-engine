import numpy as np
from scipy import spatial
from matplotlib import pyplot as plt
import os
from sklearn.neighbors import NearestNeighbors
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

npeaks = 50
minpoints = 30
maxpoints = 10000
maxloc = 25000 #ends up being rough, not exact

endarray = []
for n in range(npeaks):
    xvalue = np.random.uniform(-maxloc, maxloc)
    rpoints = np.random.randint(minpoints, maxpoints)
    rstdev = np.random.randint(minpoints, rpoints)
    peak = np.random.normal(xvalue, rstdev, size=rpoints)
    endarray.extend(peak.tolist())
endarray = np.sort(endarray)

#you would aim to do a wide-to-narrow approach, ie largest distances to small
#the 'final push' of any density apex -> when it gets to 1 data point -> is determine by it's distance to 
#don't just group by diffs, group by directional gradients -> ie groups of diffs that all increase or decrease
#^use the largest and smallest of these as your iterable distances
#you could also look at differences as a variation of sorts: +/- larger numbers is a less dense area, and you can section these areas into iterables based on gradients of the differences of rates of change per data point?
#another basis for the gradient could be to do a diff of the sorted values and split it up by increasing differences and decreasing differences

#endiffs = np.diff(endarray)
#starts = endarray[:-2]
#ends = endarray[2:]
#increasers = np.where(endiffs > 0)[0]

erange = np.ptp(endarray)
divider = erange / endarray.size
ndivs = int(erange / divider) #this is always either a whole number or a .9999 trail without the rounding

dividerpass = divider / 2 #distance used to determine whether to add to the next kernel
regionkernel = np.linspace(endarray.min(), endarray.max(), ndivs).tolist()

#density points are the median location between each divider
#for divider ranges with n data points within them, that density point gets (# of matrix combinations/sum of matrix diffs)**n as a value
#divider ranges with no data points connect the two nearest points and multiply the number of ... something

#but i want the process to be more independent of rigid dividers that may split areas of high density into two, now gerrymandered, groups
#every point gets a nearest-neighbor with divider radius
#each point can "prop" one of its neighbors up - whichever one has more points. If they both have the same number, it will prop both of them up. But either neighbors height depends on the (# of matrix combinations / sum of matrix diffs)**n, for n data points, of that points divider radius
#the # of neighbors within divider radius is a decent initial estimate of density, but i suppose the differences of the points within better elucidate the nature of the density of these points
#If point B is propped by point A, then point A's height is rested upon the base of B's total, B's total is the height it got from being propped (if it was) plus its density basis
#something can be propped from both directions, this is determied first
#if two neighbors have the same number of points within their radius, they share their other outer props
#for points that have nothing within their radius... you need to find its nearest point and do the same equation I suppose, it will give you a small number!
#so this entire thing wouldn't work great for integer data, if a matrix difference ends up being 0 -> you get nothing!

nn = NearestNeighbors(n_neighbors=2, radius=divider, n_jobs=-1)
nn.fit(endarray[:,None])
dists, inds = nn.radius_neighbors(endarray[:,None])

kernelind = 0
kernelplacement = regionkernel[kernelind]

lastleft = 0 #the number of matched points within the radius of the previous iteration
points = np.zeros(endarray.size)
kernellocations = []
for n, (e, d, i) in enumerate(zip(endarray.tolist(), dists, inds)):
    if e > kernelplacement:
        while e - kernelplacement > dividerpass: #closer to the next divider
            kernelind += 1
            kernelplacement = regionkernel[kernelind]
    #density calculations
    if i.size > 1:
        #diff matrix
        isize = i.size
        #pointdensity = ((isize**2) / np.abs(d - d[:,None]).sum())**isize
        ei = endarray[i]
        pointdensity = ((isize**2) / np.abs(ei - ei[:,None]).sum())**isize
        #^d isn't what i meant to use here, i meant to use the original datapoints and diff them, but I guess this is derivative so we'll see how it goes
    else:
        #find nearest, which would be a neighboring index
        try:
            li = i[0] - 1
            ldist = abs(endarray[li] - e)
        except IndexError: #first index
            ldist = np.inf
        try:
            ri = i[0] + 1
            rdist = abs(endarray[ri] - e)
        except IndexError: #last index reached
            rdist = np.inf
        mindist = min(ldist, rdist)
        pointdensity = (2 / mindist)**2
    #prop determination, the added 'or equal to' allows sharing props for equal match counts
    #just going to prop any direction with more matches i suppose
    #if e <= lastleft:
    #    points[n-1] += pointdensity #prop left
    #try:
    #    if e <= dists[n+1].size:
    #        points[n+1] += pointdensity #prop right
    #except IndexError: #last index reached
    #    pass
    lastleft = d.size
    points[n] += pointdensity
    kernellocations.append(kernelind)

finalkernel = np.zeros(len(regionkernel))
#for p, l in zip(points.tolist(), kernellocations):
#    finalkernel[l] += p
np.add.at(finalkernel, kernellocations, points)



#organize props via individual lookaheads/lookbacks on each iteration

#the resulting problem is that two high-density points can link over a non-dense region and not have any way of making a valley in between
#^the "kernel" in this sense will be the range (divider) that's each value is broken down into, and it will cover this concept


#this concept ultimately failed to represent anything correctly, but I think it might have potential somewhere
