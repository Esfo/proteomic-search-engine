fcombos = [
        (0,3,),
        (0,4,),
        (1,3,),
        (1,4,),
        (2,3,),
        (3,4,),
        (0,1,3,),
        (0,2,3,),
        (0,3,4,),
        (1,2,3,),
        (1,3,4,),
        (2,3,4,),
        (0,1,2,3,),
        (0,1,3,4,),
        (0,2,3,4,),
        (1,2,3,4,),
        (0,1,2,3,4,),
        ]


timeorganizer = set()
starts = defaultdict(set) #startind: lines
ends = defaultdict(set) #endind: lines
for k in plotkeys:
    v = trackedgroups[k]
    a = np.array(v)
    times = a[1].tolist()
    timeorganizer.update(times)
    starts[min(times)].add(k)
    ends[max(times)].add(k)

timeorganizer = sorted(timeorganizer)

datapoints = {} #pair: [data], explanation below
#find all overlapping pairs and their % overlap via the pool system ^above
#offsets / total range -> abs(loverhang + roverhang) / (nrange + orange)
#and -> abs(loverhang + roverhang)?!
#and -> abs(loverhang) + abs(roverhang) / (nrange + orange)
#and  -> abs(loverhang) + abs(roverhang)
#co-rank them^

linepool = set()
for t in timeorganizer:
    newstarts = starts[t]
    for nl in newstarts:
        nltimes = trackedgroups[nl][1]
        nlmin = min(nltimes)
        nlmax = max(nltimes)
        nlrange = nlmax - nlmin
        for ol in linepool:
            oltimes = trackedgroups[ol][1]
            olmin = min(oltimes)
            olmax = max(oltimes)
            olrange = olmax - olmin
            #
            roverhang = olmin - nlmin
            loverhang = olmax - nlmax
            overlap = min((olmax, nlmax)) - max((olmin, nlmin)) + 1
            combinedrange = nlrange + olrange
            equalizedoverhang = abs(roverhang + loverhang)
            totaloverhang = abs(roverhang) + abs(loverhang)
            #percentequalizedoverhang = equalizedoverhang / combinedrange
            percentoverhang = totaloverhang / combinedrange
            percentoverlap = (overlap * 2) / combinedrange
            #dpoints = [equalizedoverhang, totaloverhang, percentequalizedoverhang, percenttotaloverhang]
            dpoints = [percentoverhang, totaloverhang, equalizedoverhang, 1/percentoverlap, 1/overlap]
            pair = ol, nl
            datapoints[pair] = dpoints
    linepool.update(newstarts)
    newends = ends[t]
    for e in newends:
        linepool.remove(e)

#% overhang -> minimize
#total overhang -> minimize
#equalized overhang -> minimize
#% overlap -> maximize
#total overlap -> maximize

fullmatch = len(dpoints)
fminds = range(fullmatch)

#find what you follow -> if your best is worse than your followers best, stop?
#any given pair tries to get the highest position in a logical group that it can.

pairs, outs = zip(*datapoints.items())
#outsorts = np.argsort(outs, axis=0)
outsorts = stats.rankdata(outs, axis=0, method='dense')
#pairsortings = [[pairs[j] for j in i] for i in outsorts]]
#^nah, you need to use scipy rank, then create a back-sorted group like timeorganizer where the dict is {rank: pairs}, this is more fair because argsort might arbitrarily take two things of the same rank and put them at different levels of this iteration

colnum = 100
cols = dp.get_colors(colnum)
#for fm in fminds:
#    fm += 1
    #for fswitch in itertools.combinations(fminds, fm):
for fswitch in fcombos:
    outcombo = outsorts[:,fswitch]
    pairrankings = defaultdict(list)
    for p, o in zip(pairs, outcombo):
        for so in o:
            pairrankings[so].append(p)
    
    fm = len(fswitch)
    #for fmatch in range(1, fm+1):
    pairorders = []
    paircounts = defaultdict(int)
    pairtracks = defaultdict(list)
    for pn in sorted(pairrankings):
        spairs = pairrankings[pn]
        newgroups = []
        for pair in spairs:
            paircounts[pair] += 1
            pairtracks[pair].append(pn)
            if paircounts[pair] == fm:
                newgroups.append(pair)
        if newgroups:
            if len(newgroups) > 1:
                ##sorting here based on which has more lower/higher values in datapoints
                #ngdata = [datapoints[i] for i in newgroups]
                ##ngranks = np.argsort(ngdata, axis=0).argsort(axis=0)
                #ngranks = stats.rankdata(ngdata, axis=0, method='dense')
                ##sort by which has most 0's, then most 1's, etc. tiebreak using the next number when necessary I suppose, ie:
                ##one with most 0's
                ##(2 things have 1 0, 1 thing has 2): 2nd place is the 1-0's with the most 1's
                ##things with the most 1s
                ##it basically starts and looks at things that HAVE 0s
                #for ranker in range(ngranks.shape[0]):
                #    rankersums = (ngranks == ranker).sum(axis=1)
                ngsorts = sorted((sorted(pairtracks[i]), i) for i in newgroups)
                newpairs = [i[1] for i in ngsorts]
                pairorders.extend(newpairs)
            else:
                pairorders.extend(newgroups)

    gn = 0
    groupids = {} #line: groupid
    linkedgroups = defaultdict(set) #gid: lines
    blocked = set()
    for pair in pairorders:
        l, r = pair
        if l not in blocked and r not in blocked:
            linkedgroups[gn].update(pair)
            groupids[l] = gn
            groupids[r] = gn
            blocked.update(pair)
            gn += 1
        elif r not in blocked:
            gid = groupids[l]
            linkedgroups[gid].add(r)
            groupids[r] = gid
            blocked.add(r)
        elif l not in blocked:
            gid = groupids[r]
            linkedgroups[gid].add(l)
            groupids[l] = gid
            blocked.add(l)


    cc = 0
    for lg in linkedgroups.values():
        if cc >= colnum:
            cc = 0
        col = cols[cc]
        cc += 1
        for l in lg:
            line = trackedgroups[l]
            x = line[1]
            y = line[0]
            plt.plot(x, y, '-', linewidth=0.8, c=col)
    plt.title(''.join((str(fswitch))))
    plt.xlim(24, 28.5)
    plt.ylim(400, 500)
    plt.show()
    plt.clf()
    plt.close()
    gc.collect()
