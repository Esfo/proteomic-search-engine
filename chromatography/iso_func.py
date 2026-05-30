def iso_func(spectrum, rpoints, newinclimit, steplimit, subisomax, chargetolerance, proton, masswidths):
    masses = sorted(spectrum.keys())
    di = 0
    paircharges = {} #connection: charge
    datapdiffs = {} #connection: %-diff of number of datapoints, things that can be calculated directly between two peaks and don't need to rely on the entire distribution for information
    connections = defaultdict(dict) #groupid: connection: acdiff
    subgroups = defaultdict(lambda: defaultdict(set)) #mass: charge: groups, a mass is always active as long as it's in masspool, this keeps track of all the groups that end in any given mass
    activedirection = defaultdict(int) #groupid: direction, 0 (increasing) or 1 (decreasing)
    si = 0
    subisogroups = defaultdict(lambda: defaultdict(set)) #subiso group: max charge for mass: [masses]
    subisomasses = {} #mass: subisogroup
    masspool = set()
    referenced = defaultdict(set) #mass: groupid, subgroups that have been referenced and continued, they'll be removed as unnecessary intermediates once the designated mass is removed from masspool
    for nm in masses:
        olddirections = activedirection.copy()
        #subgroupchanges = defaultdict(lambda: defaultdict(set))
        masspoolremovals = set()
        for om in masspool:
            #eliminate anything in masspool > proton distance from nm
            diff = nm - om
            widthbuffer = masswidths[nm] + masswidths[om]
            if diff < subisomax + widthbuffer:
                #potential subisos
                maxisocharge = round((subisomax / diff) - 0.5) #floor
                if om in subisomasses:
                    si = subisomasses[om]
                    subisogroups[si][maxisocharge].update((om, nm))
                    subisomasses[nm] = si
                else:
                    #make a new subiso group
                    subisogroups[si][maxisocharge].update((om, nm))
                    subisomasses[om] = si
                    subisomasses[nm] = si
                    si += 1
            elif diff <= proton + widthbuffer:
                charge = round(proton / diff)
                expdiff = proton / charge
                acdiff = expdiff - diff
                if acdiff > 0:
                    diffcut = expdiff * chargetolerance
                    if acdiff <= diffcut + widthbuffer:
                        sst = spectrum[nm]
                        sam = spectrum[om]
                        intensitypercdiff = abs(sst - sam) / (sst + sam) / 2
                        ncons = 0
                        csubs = subgroups[om][charge]
                        if csubs:
                            for adi in csubs:
                                ratiocheck = steplimit
                                if olddirections[adi] > 0:
                                    #the dist has previously begun a decrease
                                    if sst >= sam:
                                        #and intensity is increasing
                                        ratiocheck = newinclimit
                                if intensitypercdiff <= ratiocheck:
                                    lpair = (om, nm)
                                    referenced[om].add(adi)
                                    connections[di].update(connections[adi])
                                    connections[di][lpair] = acdiff
                                    subgroups[nm][charge].add(di)
                                    if sst < sam:
                                        activedirection[di] += 1
                                    ncons += 1
                                    di += 1
                        else:
                            #no previous subgroup
                            if intensitypercdiff <= steplimit:
                                lpair = (om, nm)
                                connections[di][lpair] = acdiff
                                subgroups[nm][charge].add(di)
                                if sst < sam:
                                    #decreasing
                                    activedirection[di] += 1
                                ncons += 1
                                di += 1
                        if ncons > 0:
                            npoints = rpoints[nm]
                            apoints = rpoints[om]
                            dpercdiff = abs(npoints - apoints) / (npoints + apoints) / 2
                            datapdiffs[lpair] = dpercdiff
                            paircharges[lpair] = charge
                            if csubs:
                                connections[di][lpair] = acdiff
                                subgroups[nm][charge].add(di)
                                if sst < sam:
                                    #decreasing
                                    activedirection[di] += 1
                                di += 1
            else:
                #om is past proton distance, remove om from mass pool
                masspoolremovals.add(om)
        for mpr in masspoolremovals:
            masspool.remove(mpr)
        masspool.add(nm)
    groupsbypair = defaultdict(set)
    scoresbypair = defaultdict(dict) #mass: pair: [scores]
    secondpriorities = defaultdict(dict) #essentially there's too many zeros caused by single-pair matches, they can be let in but they don't deserve to priority
    for gv, sgp in connections.items():
        slen = len(sgp)
        if slen > 1:
            sgmean = sum(sgp.values()) / slen
            for sgk, sgs in sgp.items():
                groupsbypair[sgk].add(gv)
                #scoreval = abs(sgmean - sgs)
                scoreval = abs(sgmean - sgs) / (slen + 1)
                #scoreval = abs(sgmean - sgs) / (slen + 1) / paircharges[sgk]
                #difftomeans[group][sgv][sgk] = scoreval
                scorelist = [scoreval, datapdiffs[sgk]]
                #scorelist = scoreval
                for m in sgk:
                    scoresbypair[m][sgk] = scorelist
        else:
            sgk, sgs = zip(*sgp.items())
            sgk = sgk[0]
            sgs = sgs[0]
            groupsbypair[sgk].add(gv)
            for m in sgk:
                #cscore = datapdiffs[sgk]
                #secondpriorities[m][sgk] = datapdiffs[sgk]
                secondpriorities[m][sgk] = [sgs, datapdiffs[sgk]]
                #secondpriorities[m][sgk] = [sgs/cscore, cscore]
                #secondpriorities[m][sgk] = sgs / datapdiffs[sgk]
    grouplengths = {k: len(v) for k,v in connections.items()} #count of pairs per group
    rankedpairs = [] #[pair, minval]
    for m, pg in scoresbypair.items():
        pairs, vals = zip(*pg.items())
        vals = np.array(vals)
        vpercs = vals / vals.sum(axis=0).tolist()
        sumvals = vpercs.sum(axis=1)
        minvals = vpercs.min(axis=1)
        rankedpairs.extend(list(zip(pairs, minvals)))
    secondrankedpairs = [] #[pair, minval]
    for m, pg in secondpriorities.items():
        pairs, vals = zip(*pg.items())
        vals = np.array(vals)
        vpercs = vals / vals.sum(axis=0).tolist()
        #minvals = vpercs.min(axis=1).tolist()
        #minvals = (vals[:,1] / vals[:,0]).tolist()
        minvals = np.abs(vpercs[:,1] - vpercs[:,0]).tolist()
        secondrankedpairs.extend(list(zip(pairs, minvals)))
    sortedranks = sorted(rankedpairs, key=lambda x: x[1])
    secondranks = sorted(secondrankedpairs, key=lambda x: x[1])
    sortedranks.extend(secondranks)
    preservedpairs = set()
    preservedranks = []
    for pair, score in sortedranks:
        if pair not in preservedpairs:
            preservedranks.append([pair, score])
            preservedpairs.add(pair)
    rank = 0
    groupranks = []
    paircounts = defaultdict(int)
    pairtracks = defaultdict(list)
    for pn, (pair, score) in enumerate(preservedranks):
        gvs = groupsbypair[pair]
        newgroups = []
        for gv in gvs:
            paircounts[gv] += 1
            pairtracks[gv].append(pn)
            if paircounts[gv] == grouplengths[gv]:
                #groupranks.append([gv, paircharges[pair]])
                newgroups.append(gv)
        if newgroups:
            pc = paircharges[pair]
            if len(newgroups) > 1:
                ntracks = [pairtracks[i] for i in newgroups]
                groupsort = sorted(zip(*(ntracks, newgroups)), key=lambda x: (x[0][0], len(x[0])))
                newgroups = list(zip(*groupsort))[1]
            for gv in newgroups:
                groupranks.append([gv, pc])
    groupsets = [set(itertools.chain(*connections[gv])) for gv in list(zip(*groupranks))[0]]
    dr =  0
    franks = {} #distid: rank
    blocked = set()
    solodists = defaultdict(dict)
    for ngn, (gs, (gv, gc)) in enumerate(zip(groupsets, groupranks)):
        ng = set()
        gcs = set()
        ng.update(gs)
        gcs.add(gc)
        if not blocked.intersection(gs): 
            ngit = groupsets[ngn+1:]
            for ogn, ogs in enumerate(ngit):
                if ng < ogs: #this will not partially connect two differing charges
                    if not blocked.intersection(ogs):
                        ng.update(ogs)
                        ngc = groupranks[ngn+ogn+1][1]
                        gcs.add(ngc) 
            solodists[max(gcs)][gv] = list(ng)
            blocked.update(ng) 
            franks[gv] = dr
            dr += 1
    foundvals = []
    for charge, sgd in solodists.items():
        foundvals.extend(list(itertools.chain(*sgd.values())))
    specvals = set(spectrum.keys())
    nodists = list(specvals.difference(foundvals))
    return solodists, nodists, franks
