import numpy as np
from scipy import integrate, signal, stats
from matplotlib import pyplot as plt
import os
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from collections import Counter, defaultdict
from time import time

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

npeaks = 1000
minpoints = 2000
maxpoints = 20000
maxloc = 2500000 #ends up being rough, not exact

endarray = []
for n in range(npeaks):
    xvalue = np.random.uniform(-maxloc, maxloc)
    rpoints = np.random.randint(minpoints+1, maxpoints)
    rstdev = np.random.randint(minpoints, rpoints)
    peak = np.random.normal(xvalue, rstdev, size=rpoints)
    endarray.extend(peak.tolist())
endarray = np.sort(endarray)

#output:
#number of datapoints should be spread evenly across the range of the data

#diff everything
#inverse diff is the value added to the linearly spaced area of the data's range
#repeat the diff process until all diffs cover a range larger than the linear space's widths
#any diff accounting for a range outside the linear spacers doesn't get included in the final count

#^does all this stuff still apply? it was a note for a different section


#original algorithm, works fine but is slower than desired
#nt = time()
#fvals = []
#mindiff = np.diff(endarray).min()
#smeans = []
#spsums = []
#for es in range(2, endarray.size):
#    emax = endarray.max()
#    linears = np.linspace(endarray.min() - mindiff, endarray.max() + mindiff, es + 1)
#    #^generated above as emin/emax, just iterate a while via looping by adding numbers, it should be faster than making this
#    leftlinears = linears[:-1]
#    rightlinears = linears[1:]
#    densitycount = np.zeros(es)
#    for n, (l, r) in enumerate(zip(leftlinears.tolist(), rightlinears.tolist())):
#        densitycount[n] += np.logical_and(endarray > l, endarray < r).sum()
#    fvals.append([densitycount.tolist(), linears.tolist()])
#    #if (densitycount == 0).any():
#    #smean = densitycount.mean()
#    smean = densitycount.sum() / densitycount.size #why is this faster lol
#    smeans.append(smean)
#    spsum = (densitycount == 0).sum()
#    spsums.append(spsum)
#    if spsum**2 > smean:
#        break

#densitycount = np.array(fvals[-1][0])
#fvn = densitycount / densitycount.sum()
#fx = np.linspace(endarray.min(), endarray.max(), fvn.size)

#plt.plot(smeans, label='mean')
#plt.plot(spsums, label='sums')
#plt.legend()
#plt.show()


#nt = time()
#
##fvals = []
#mindiff = np.diff(endarray).min()
#emin = endarray.min() - mindiff
#emax = endarray.max() + mindiff
#smeans = []
#spsums = []
#ecount = []
#meanoversums = []
#for es in range(2, endarray.size):
#    #SLOWER?!
#    #linears = np.linspace(emin, emax, location + 1)
#    #lsearch = np.searchsorted(endarray, linears)
#    #densitycount = lsearch[1:] - lsearch[:-1]
#    densitycount = np.zeros(es)
#    n = 0
#    div = (emax - emin) / es
#    l = emin
#    r = l + div
#    while r < emax:
#        lfind = np.searchsorted(endarray, l)
#        rfind = np.searchsorted(endarray, r)
#        densitycount[n] += rfind - lfind
#        l = r
#        r += div
#        n += 1
#    #fvals.append([densitycount.tolist(), linears.tolist()])
#    #smean = densitycount.mean()
#    smean = densitycount.sum() / densitycount.size #why is this faster 
#    smeans.append(smean)
#    spsum = (densitycount == 0).sum()**2
#    #spsum = 0
#    #for d in densitycount.tolist():
#    #    if d == 0:
#    #        spsum += 1
#    #    else:
#    #        break
#    #for d in densitycount[::-1].tolist():
#    #    if d == 0:
#    #        spsum += 1
#    #    else:
#    #        break
#    #spsum **= 2
#    spsums.append(spsum)
#    ecount.append(es)
#    meanoversums.append(densitycount.sum() / densitycount.mean())
#    if spsum > smean:
#        break
#print(time() - nt, '- iterated')

#plt.plot(smeans, label='mean')
#plt.plot(spsums, label='sums')
#plt.legend()
#plt.show()

nt = time()

#a shortcoming of this optimization is that the spsum might cross smean multiple times and I'm just pulling out one of the spots where it does
#the optimization process needs improvements, this is super rough, and makes no estimates based on previous data, it just goes for the midpoint between the two closest values, although this should.... be quick i would think... i think i need to visualize what values are being picked in real time to figure out if something is going wrong.

mindiff = np.diff(endarray).min()
emin = endarray.min() - mindiff
emax = endarray.max() + mindiff
ediff = emax - emin
#you could still test whether its best to start location in the middle
location = 2
previouslocations = {} #location: 1, or -1 for pos/neg diffs respectively
minpositive = endarray.size #smalest location of a positive difference
maxnegative = location #largest location of a negative difference
doubling = True
#for location in range(2, endarray.size):
while True:
    densitycount = np.zeros(location)
    n = 0
    div = ediff / location
    l = emin
    r = l + div
    while r < emax:
        lfind = np.searchsorted(endarray, l)
        rfind = np.searchsorted(endarray, r)
        densitycount[n] += rfind - lfind
        l = r
        r += div
        n += 1
    smean = densitycount.sum() / densitycount.size #why is this faster
    spsum = (densitycount == 0).sum()**2
    diff = spsum - smean #optimizing for the smallest positive difference
    #I don't formally account for an == 0 option, it's pretty unlikely imo, all the sums are integers but the means should having floating accuracy
    if diff > 0:
        previouslocations[location] = 1
        doubling = False
        if location < minpositive:
            minpositive = location
    else: #< 0, and since this technically covers the <= case, generated median locations will be rounded up i suppose? actually i didn't do this but i need to think about it
        previouslocations[location] = -1
        if location > maxnegative:
            maxnegative = location
    adjacentloc = location - previouslocations[location]
    if adjacentloc in previouslocations:
        if previouslocations[adjacentloc] != previouslocations[location]:
            #settling either for the optimized answer, or the one right before it - good enough I suppose
            break #you win
    if doubling:
        newloc = location * 2
    else:
        newloc = int(round(((minpositive + maxnegative) / 2) + 0.1)) #the 0.1 makes a .5 round to 1 which would normally go to 0, why... python?
    while newloc in previouslocations:
        newloc -= previouslocations[newloc]
    location = newloc
    #if spsum > smean:
    #    break

print(time() - nt, 'optimized')

#strategy:
#double location until you go positive, location can't be > endarray.size
#median until you hit the good spot


#^this is looking good now, the next step is to make an optimization process
#i need to figure out what the spsum/smean relationship looks like way above my target goal to see what optimization might look like

#initial step = 1, double this
#step != location, the location is the number (es in the loop version), step is the change from that number for the next iteration
#if the rate you're approaching at is less 1/2, then double the step
#if the rate you're approaching is more than 2, 1.5x the step
#manage direction
#if the stepchange ends up less than desirable, the step doubler should be able to handle this via the moving approach prediction
#ideal goal is the first value where spsum**2 > smean, aim for an absolute difference close to 0 i suppose
#keep track of the locations you've landed on and +1 until you get to a unique one if it repeats, also keep them in a dict that indicates if it was positive or negative, if location n is negative and n+1 is positive, that's you're winner


#densitycount = np.array(fvals[-1][0])
fvn = densitycount / densitycount.sum()
#fx = np.linspace(endarray.min(), endarray.max(), fvn.size)
fx = np.linspace(emin, emax, fvn.size)


##sklearn representative comparison
#nt = time()
#karray = endarray[:,None]
##kplot = np.linspace(karray.min(), karray.max(), fvn.size)[:,None]
#kplot = np.linspace(karray.min(), karray.max(), 10000)[:,None]
#kde = KernelDensity(kernel='gaussian', bandwidth=100).fit(karray)
##bandwidth = np.linspace(10, 100, 10)
##grid = GridSearchCV(kde, {'bandwidth': bandwidth})
##grid.fit(karray)
##kde = grid.best_estimator_
#log_density = kde.score_samples(kplot)
#pdens = np.exp(log_density)
#dperc = pdens / pdens.sum()
#print(time() - nt, 'kde')
#
#
##normalization for plotting
#meandiff = fvn.mean() / dperc.mean()
#dperc *= meandiff

zerospots = fx[densitycount == 0]


fig, ax = plt.subplots(figsize=(7,4))

tx = ax.twinx()
#tx.plot(kplot, dperc, '-', color='fuchsia', linewidth=0.5)

tx.plot(fx, fvn, '-', color='cyan', linewidth=0.3)
tx.vlines(zerospots, 0, fvn.max(), color='black', alpha=0.05)
#^this is showing the optimization is basically deciding to finish because the outer points are basically coming out to zero, so I changed the bounds of fx to be emin/emax again so that the 0s make sense - because those have a bounds where there is no data
#i can forsee some kind of problem with an increasing distribution being on the end of endarray, but I think this might actually not happen because the bins essentially keep getting smaller and shoving more 0s on the sides
#^actually a bigger problem is when there are more inner zeros than outer, so i'm changing it to optimize for outer zeros!

ax.hist(endarray, bins=500, color='white', alpha=0.5)

align_yaxis_np(ax, tx)
plt.show()

#plt.plot(fvn, dperc, '.', color='white')
#plt.show()
