import numpy as np
from time import time
from matplotlib import pyplot as plt
from collections import defaultdict
from distinctipy import distinctipy as dp
from scipy import stats
import itertools
import os
import concurrent.futures

#x = [10000, 30000, 50000, 100000, 200000, 500000, 1000000]
#y = [0.4, 1, 1.7, 4, 11.6, 56, 208]

nlines = 10000
nx = 2000
minlength = 1
maxlength = 100

plotting = True

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

lineparams = [] #[start, end, width, height]
lineobjects = []
for nl in range(nlines):
    width = np.random.randint(low=minlength, high=maxlength)
    pos = np.random.randint(low=0, high=nx)
    if width + pos > nx:
        pos = nx - width
    lineparams.append([pos, pos+width, width, nl])
    
    linetimes = list(range(pos, pos+width))
    lineheights = [nl for _ in range(width)]
    lineobjects.append([linetimes, lineheights])

lineregions = []
for lo in lineobjects:
    lineregions.append([min(lo[0]), max(lo[0]), max(lo[1]), len(lo[0])])
lineregions = np.array(lineregions)

#nt = time()
#uplinks = {} #minor: major
#for nlt, nrt, nkey, nd in lineregions.tolist():
#    width = nrt - nlt
#    #keeping the below without an >= or <= prevents self-referencing loops
#    leftcheck = nlt > lineregions[:,0]
#    rightcheck = nrt < lineregions[:,1]
#    check = np.logical_and(leftcheck, rightcheck)
#    check[nkey] = False
#    overlaps = lineregions[check]
#    if nd < 10:
#        widthmultiplier = width * 10
#    else:
#        widthmultiplier = width * 2
#    overinds = overlaps[:,1] - overlaps[:,0] <= widthmultiplier
#    overlaps = overlaps[overinds]
#    winind = False
#    if overlaps.size > 1:
#        olt = overlaps[:,0]
#        ort = overlaps[:,1]
#        lwing = nlt - olt
#        rwing = ort - nrt
#        wingsum = lwing + rwing
#        wingequalizer = np.abs(lwing - rwing)
#        nowidths = ort - olt
#        wingranks = list(zip(wingsum.argsort().tolist(), wingequalizer.argsort().tolist()))
#        seeninds = set()
#        for newinds in wingranks:
#            winners = []
#            for ni in newinds:
#                if ni in seeninds:
#                    ow = nowidths[ni]
#                    if ow < widthmultiplier:
#                        winners.append(ni)
#                    else:
#                        seeninds.remove(ni)
#                seeninds.add(ni)
#            wlen = len(winners)
#            if wlen > 0:
#                if wlen > 1:
#                    dpoints = overlaps[winners,3].tolist()
#                    if dpoints[0] == dpoints[1]:
#                        #higher area = winner -> randmly picking it in here though
#                        winner = winners[0]
#                        loser = winners[1]
#                    else:
#                        winner = winners[dpoints.index(max(dpoints))]
#                        loser = winners[dpoints.index(min(dpoints))]
#                else:
#                    winner = winners[0]
#                winind = overlaps[winner,2]
#                uplinks[nkey] = winind
#            if winind:
#                break
#    if not winind:
#        uplinks[nkey] = nkey
#print(time() - nt, 'uplinks')
#
#nt = time()
#downlinks = defaultdict(set)
#for k, v in uplinks.items():
#    while v in uplinks:
#        if v == uplinks[v]:
#            break
#        v = uplinks[v]
#    downlinks[v].add(k)
#print(time() - nt, 'downlinks')
#
#nt = time()
#downregions = []
#for k, v in downlinks.items():
#    lr = lineregions[k]
#    out = [lr[0], lr[1], k]
#    downregions.append(out)
#downregions = np.array(downregions)
#print(time() - nt, 'downregions')
#
#nt = time()
#newdownlinks = defaultdict(set) #major: [minors]
#for l, r, k in downregions.tolist():
#    width = r - l
#    leftcheck = l >= downregions[:,0]
#    rightcheck = r <= downregions[:,1]
#    check = np.logical_and(leftcheck, rightcheck)
#    overlaps = downregions[check]
#    overinds = overlaps[:,1] - overlaps[:,0] <= width * 2
#    overlaps = overlaps[overinds]
#    if overlaps.size > 1:
#        omajorind = (overlaps[:,1] - overlaps[:,0]).argmax()
#        omajor = overlaps[omajorind,2]
#        newdownlinks[omajor].add(k)
#        newdownlinks[omajor].update(downlinks[k])
#print(time() - nt, 'newdownlinks')
#
#if plotting:
#    nc = 20
#    cols = dp.get_colors(nc)
#    fig, ax = plt.subplots(figsize=(7,6))
#    n = 0
#    for k, v in newdownlinks.items():
#        for sv in v:
#            a = lineobjects[sv]
#            ax.plot(a[0], a[1], '-', color=cols[n], alpha=0.4, markersize=0.4)
#        a = lineobjects[k]
#        ax.plot(a[0], a[1], '*', color=cols[n], alpha=0.8, markersize=0.4)
#        n += 1
#        if n >= nc:
#            n = 0
#    plt.show()
#
#
#print('~~~~~')


#nt = time()
#uplinks = {} #minor: major
#with concurrent.futures.ThreadPoolExecutor(1) as executor:
#    futures = []
#    for lr in lineregions.tolist():
#        futures.append(executor.submit(reg_processing, lr, lineregions))
#    for f in concurrent.futures.as_completed(futures):
#        f = f.result()
#        match f:
#            case tuple():
#                uplinks[f[0]] = f[1]
#print(time() - nt, 'concurrent')

#timeorganizer = defaultdict(list) #time: inds
nt = time()
times = set()
starts = defaultdict(list) #startind: lines
ends = defaultdict(list) #endind: lines
for tn, (t, h) in enumerate(lineobjects):
    #for tv in t:
    #    timeorganizer[tv].append(tn)
    times.update(t)
    starts[min(t)].append(tn)
    ends[max(t)].append(tn)
times = sorted(times)
print(time() - nt, 'time organizing')

nt = time()
uplinks = {} #lesser line: greater line
linepool = []
for t in times:
    newstarts = starts[t]
    olt, ort = lineregions[linepool,:2].transpose()
    owidths = ort - olt
    opoints = lineregions[linepool,3]
    #okeys = lineregions[linepool,2]
    linepool = np.array(linepool)
    #for nl in newstarts:
    for nreg in lineregions[newstarts]:
        #nreg = lineregions[nl]
        nlt = nreg[0]
        nrt = nreg[1]
        nl = nreg[2]
        nwidth = nrt - nlt
        #2x uplink width limit for larger lines, 10x for tiny ones
        if nwidth < 10: #minmovinginds here I think
            nwidth *= 10
        else:
            nwidth *= 2
        #encompassers = np.logical_and(olt < nlt, ort > nrt) #you already know the left-bound checks out from the starting point
        encompassers = ort > nrt
        nolt = olt[encompassers]
        nort = ort[encompassers] 
        nowidths = owidths[encompassers]
        ndpoints = opoints[encompassers]
        lwing = nlt - nolt
        rwing = nort - nrt
        wingsum = lwing + rwing
        wingequalizer = np.abs(lwing - rwing)
        wingranks = list(zip(wingsum.argsort().tolist(), wingequalizer.argsort().tolist()))
        seeninds = set()
        winind = False
        for newinds in wingranks:
            winners = []
            for ni in newinds:
                if ni in seeninds:
                    owidth = nowidths[ni]
                    if owidth <= nwidth:
                        winners.append(ni)
                    else:
                        seeninds.remove(ni)
                seeninds.add(ni)
            wlen = len(winners)
            if wlen > 0:
                if wlen > 1:
                    #dpoints = [len(lineobjects[i][0]) for i in winners]
                    dpoints = ndpoints[winners].tolist()
                    if dpoints[0] == dpoints[1]:
                        #higher area = winner -> randmly picking it in here though
                        winner = winners[0]
                        loser = winners[1]
                    else:
                        winner = winners[dpoints.index(max(dpoints))]
                        loser = winners[dpoints.index(min(dpoints))]
                else:
                    winner = winners[0]
                winind = linepool[encompassers][winner]
                uplinks[nl] = winind
            if winind:
                break
        if not winind:
            uplinks[nl] = nl
    linepool = linepool.tolist()
    linepool.extend(newstarts)
    newends = ends[t]
    for e in newends:
        linepool.remove(e)
print(time() - nt, 'uplinks')


nt = time()
downlinks = defaultdict(set)
for k, v in uplinks.items():
    while v in uplinks:
        if v == uplinks[v]:
            downlinks[v].add(v)
            break
        v = uplinks[v]
    downlinks[v].add(k)
print(time() - nt, 'downlinks')

nt = time()
downregions = []
for k, v in downlinks.items():
    lr = lineregions[k]
    out = [lr[0], lr[1], k]
    downregions.append(out)
downregions = np.array(downregions)
print(time() - nt, 'downregions')

nt = time()
newdownlinks = defaultdict(set) #major: [minors]
for l, r, k in downregions.tolist():
    width = r - l
    leftcheck = l >= downregions[:,0]
    rightcheck = r <= downregions[:,1]
    check = np.logical_and(leftcheck, rightcheck)
    overlaps = downregions[check]
    overinds = overlaps[:,1] - overlaps[:,0] <= width * 2
    overlaps = overlaps[overinds]
    if overlaps.size > 1:
        omajorind = (overlaps[:,1] - overlaps[:,0]).argmax()
        omajor = overlaps[omajorind,2]
        newdownlinks[omajor].add(k)
        newdownlinks[omajor].update(downlinks[k])
print(time() - nt, 'newdownlinks')

if plotting:
    nc = 20
    cols = dp.get_colors(nc)
    fig, ax = plt.subplots(figsize=(7,6))
    n = 0
    for k, v in newdownlinks.items():
        for sv in v:
            a = lineobjects[sv]
            ax.plot(a[0], a[1], '-', color=cols[n], alpha=0.4, markersize=0.4)
        a = lineobjects[k]
        ax.plot(a[0], a[1], '*', color=cols[n], alpha=0.8, markersize=0.4)
        n += 1
        if n >= nc:
            n = 0
    plt.show()






##wtf is this
##timeindices = {}
##for t in sorted(timeorganizer):
##    v = timeorganizer[t]
##    for sv in v:
##        timeindices[sv] = lineobjects[sv][0].index(t)
#
##the goal should be to group lines with the most in common in terms of overlap, and allow in things that fit between, end-game is to avoid linking literally everything together
##for t, h in lineobjects:
##    plt.plot(t, h, '.-', markersize=0.5)
##plt.show()
#
#
##generate these overlaps on the fly, then iterate over all the masses whose overlaps have been checked -> once every mass within one mass's overlap list has been overlap-checked: you map the lines from smallest to largest
##a line is fully overlap-checked when it enters newends
##line mapping:
##once a line finds its closest upward match in overlap, you combine both of their line lists and move on to the next one
##what I need to develop: how to select overlap, and how to move on to the next, as well as how to cut off lines that are long noise signals
#
##selecting overlap:
##most encompassment, least overhang, co-rank both of these
##^overhang determined via co-ranking l and r overhangs
##both overlap lists are then combined and the process repeats
##so can the same line match to two different lines in the same iteration -> and link the groups? Yes
##^and moving forward in the following iteration, the group is represented as the widest line in each group, and the group can match to other groups
#
##find the most encompassing line -> it will match to the things it encompasses, 
##maybe... start by finding all the largest overlapping but not encompassing
##try going on pure/largest encompassement
##I also need to better visualize the full-spectrum overlap testing I did befire, make multiple tall ass plots if you need just get better eyes on it, group the whole spectrum and only plot a portion of it maybe
#
#overlaps = defaultdict(set)
#linepool = set()
#for t in sorted(timeorganizer):
#    newstarts = starts[t]
#    for nl in newstarts:
#        for ol in linepool:
#            overlaps[nl].add(ol)
#            overlaps[ol].add(nl)
#    linepool.update(newstarts)
#    newends = ends[t]
#    for e in newends:
#        linepool.remove(e)
#        #add to overlap-complete iteration
#    #overlap-complete iteration: check if everything something is connected to has been added to overlap-complete iteration: if so, move on to the overlap-checking
#    #for loop: iterate overlap-completes -> start with smallest line, connect to it's next highest
#
#datapoints = {} #pair: [data], explanation below
##find all overlapping pairs and their % overlap via the pool system ^above
##offsets / total range -> abs(loverhang + roverhang) / (nrange + orange)
##and -> abs(loverhang + roverhang)?!
##and -> abs(loverhang) + abs(roverhang) / (nrange + orange)
##and  -> abs(loverhang) + abs(roverhang)
##co-rank them^
#
#linepool = set()
#for t in sorted(timeorganizer):
#    newstarts = starts[t]
#    for nl in newstarts:
#        nltimes = lineobjects[nl][0]
#        nlmin = min(nltimes)
#        nlmax = max(nltimes)
#        nlrange = nlmax - nlmin
#        for ol in linepool:
#            oltimes = lineobjects[ol][0]
#            olmin = min(oltimes)
#            olmax = max(oltimes)
#            olrange = olmax - olmin
#            #
#            roverhang = olmin - nlmin
#            loverhang = olmax - nlmax
#            overlap = min((olmax, nlmax)) - max((olmin, nlmin)) + 1
#            combinedrange = nlrange + olrange
#            equalizedoverhang = abs(roverhang + loverhang)
#            totaloverhang = abs(roverhang) + abs(loverhang)
#            #percentequalizedoverhang = equalizedoverhang / combinedrange
#            percentoverhang = totaloverhang / combinedrange
#            percentoverlap = (overlap * 2) / combinedrange
#            #dpoints = [equalizedoverhang, totaloverhang, percentequalizedoverhang, percenttotaloverhang]
#            dpoints = [percentoverhang, totaloverhang, equalizedoverhang, 1/percentoverlap, 1/overlap]
#            pair = ol, nl
#            datapoints[pair] = dpoints
#    linepool.update(newstarts)
#    newends = ends[t]
#    for e in newends:
#        linepool.remove(e)
#
##% overhang -> minimize
##total overhang -> minimize
##equalized overhang -> minimize
##% overlap -> maximize
##total overlap -> maximize
#
#
##find what you follow -> if your best is worse than your followers best, stop?
##any given pair tries to get the highest position in a logical group that it can.
#
#fullmatch = len(dpoints)
#fminds = range(fullmatch)
#
#pairs, outs = zip(*datapoints.items())
##outsorts = np.argsort(outs, axis=0)
#outsorts = stats.rankdata(outs, axis=0, method='dense')
##pairsortings = [[pairs[j] for j in i] for i in outsorts]]
##^nah, you need to use scipy rank, then create a back-sorted group like timeorganizer where the dict is {rank: pairs}, this is more fair because argsort might arbitrarily take two things of the same rank and put them at different levels of this iteration
#
#fcombos = [
#        (4,),
#        (1,4,),
#        (2,4,),
#        (1,2,4,),
#        ]
#
#
#colnum = 100
#cols = dp.get_colors(colnum)
##for fm in fminds:
##    fm += 1
##    for fswitch in itertools.combinations(fminds, fm):
##for fswitch in fcombos:
##outcombo = outsorts[:,fswitch]
#pairrankings = defaultdict(list)
#for p, o in zip(pairs, outsorts):
#    for so in o:
#        pairrankings[so].append(p)
#
##fmatch = len(fswitch)
#fmatch = len(dpoints)
##for fmatch in range(1, fm+1):
#pairorders = []
#paircounts = defaultdict(int)
#pairtracks = defaultdict(list)
#for pn in sorted(pairrankings):
#    spairs = pairrankings[pn]
#    newgroups = []
#    for pair in spairs:
#        paircounts[pair] += 1
#        pairtracks[pair].append(pn)
#        if paircounts[pair] == fmatch:
#            newgroups.append(pair)
#    if newgroups:
#        if len(newgroups) > 1:
#            ##sorting here based on which has more lower/higher values in datapoints
#            #ngdata = [datapoints[i] for i in newgroups]
#            ##ngranks = np.argsort(ngdata, axis=0).argsort(axis=0)
#            #ngranks = stats.rankdata(ngdata, axis=0, method='dense')
#            ##sort by which has most 0's, then most 1's, etc. tiebreak using the next number when necessary I suppose, ie:
#            ##one with most 0's
#            ##(2 things have 1 0, 1 thing has 2): 2nd place is the 1-0's with the most 1's
#            ##things with the most 1s
#            ##it basically starts and looks at things that HAVE 0s
#            #for ranker in range(ngranks.shape[0]):
#            #    rankersums = (ngranks == ranker).sum(axis=1)
#            ngsorts = sorted((sorted(pairtracks[i]), i) for i in newgroups)
#            newpairs = [i[1] for i in ngsorts]
#            pairorders.extend(newpairs)
#        else:
#            pairorders.extend(newgroups)
#
##each line can link to one other thing, infinite number of things can link to anything
##so as long as both aren't blocked, it can link
#
#gn = 0
#groupids = {} #line: groupid
#linkedgroups = defaultdict(set) #gid: lines
#blocked = set()
#for pair in pairorders:
#    l, r = pair
#    if l not in blocked and r not in blocked:
#        linkedgroups[gn].update(pair)
#        groupids[l] = gn
#        groupids[r] = gn
#        blocked.update(pair)
#        gn += 1
#    elif r not in blocked:
#        gid = groupids[l]
#        linkedgroups[gid].add(r)
#        groupids[r] = gid
#        blocked.add(r)
#    elif l not in blocked:
#        gid = groupids[r]
#        linkedgroups[gid].add(l)
#        groupids[l] = gid
#        blocked.add(l)
#
#cc = 0
#for lg in linkedgroups.values():
#    if cc >= colnum:
#        cc = 0
#    col = cols[cc]
#    cc += 1
#    for l in lg:
#        x, y = lineobjects[l]
#        plt.plot(x, y, '-', linewidth=0.8, c=col)
##plt.title(''.join((str(fswitch), ' - ', str(fmatch))))
#plt.show()
#print(len(linkedgroups), 'groups')
#print('~')
#
#
##to summarize, this random data ain't a great match for real data, but maybe it is and I just need to use more real data to see it?
##the real data has memory issues
##and, I don't think the pair concept is overall that great, this is what needs the most replacement imo
#
##at any given timepoint, start via the shortest lines in linepool, iterate up to the biggest ones
##constantly link things that overlap
##eventually, the biggest one 'leads the pack': however, when you get to a new pack and find that your largest one is still the same, you ditch the largest for both groups and reinstate different leaders
