p1 = []
p2 = []
p3 = []
px = []
#for et in [25, 30, 35, 40, 45, 50]:
for et in [60, 100, 120]:
    boundrec = [lmb, umb, st, et]
    plotkeys = arg_coord_rectangle_overlap(boundrec, regions[:,:4]).tolist()

    minpoints = 2
    minintensity = 0.4e6
    chargetolerance = 0.1
    mincharge = 0
#chargemethod = None
    chargemethod = 'step'
#stepsize = 10

    subisomax = subisodiffs.max()
    subisomax = subisomax + subisomax * chargetolerance

    spectrum = {}
    masswidths = {}
    timeranges = {}
    rlookup = {}
    rpoints = {}
#regions.append([minmass, maxmass, mintime, maxtime, len(v[0]), peakarea, npeakarea, wmean, k])
    for n, k in enumerate(plotkeys):
        pregion = regions[k]
        intensity = pregion[6]
        a = trackedgroups[k]
        mmass = pregion[7]
        klen = pregion[4]
        if klen >= minpoints:
            #if intensity > minintensity:
            spectrum[mmass] = intensity
            masswidths[mmass] = pregion[1]  - pregion[0]
            timeranges[mmass] = [pregion[2],  pregion[3]]
        rlookup[mmass] = k
        rpoints[mmass] = klen

    masses = sorted(spectrum.keys())
    di = 0
    paircharges = {} #connection: charge
    datapdiffs = {} #connection: %-diff of number of datapoints, things that can be calculated directly between two peaks and don't need to rely on the entire distribution for information
    rtoffsets = {} #connection: overlap balance, a %
    connections = defaultdict(dict) #groupid: connection: acdiff
    subgroups = defaultdict(lambda: defaultdict(set)) #mass: charge: groups, a mass is always active as long as it's in masspool, this keeps track of all the groups that end in any given mass
    activedirection = defaultdict(int) #groupid: direction, 0 (increasing) or 1 (decreasing)

    si = 0
    subisogroups = defaultdict(lambda: defaultdict(set)) #subiso group: max charge for mass: [masses]
    subisomasses = {} #mass: subisogroup

    masspool = set()
    referenced = defaultdict(set) #mass: groupid, subgroups that have been referenced and continued, they'll be removed as unnecessary intermediates once the designated mass is removed from masspool
    t1 = time()
    for nm in masses:
        olddirections = activedirection.copy()
        #subgroupchanges = defaultdict(lambda: defaultdict(set))
        masspoolremovals = set()
        nmlt, nmrt = timeranges[nm]
        nlrange = nmrt - nmlt
        for om in masspool:
            omlt, omrt = timeranges[om]
            olrange = omrt - omlt
            overpass = False
            #if nmlt < omrt and nmrt > omlt: #rt's overlap, maybe this needs to be stronger?
            if nmlt > omlt and nmrt < omrt: #new encompassed
                overpass = True
            elif omlt > nmlt and omrt < nmrt: #old encompassed
                overpass = True
            else:
                overlap = min((omrt, nmrt)) - max((omlt, nmlt))
                no = overlap / nlrange
                oo = overlap / olrange
                if no > 0.5 and oo > 0.5: #max of new/old overlap > 0.5, a majority overlap for both
                    overpass = True
            if overpass:
                #^one of them should have to majority overlap, or fully encompass
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
                                #this isn't overlap, this is overhang
                                #leftoverlap = nmlt - omlt
                                #rightoverlap = nmrt - omrt
                                #rtoffset = abs(leftoverlap + rightoverlap) / (nmrt - nmrt + omrt - omlt)
                                #rtoffset = abs(leftoverlap / rightoverlap)
                                #offset / overlapped
                                #if nmlt < omlt and omrt < nmrt: #new encompasses old
                                #    if rtoffset > 1:
                                #        rtoffset = 1 / rtoffset
                                #elif omrt < nmlt and nmlt < omrt: #old encompasses new
                                #    if rtoffset > 1:
                                #        rtoffset = 1 / rtoffset
                                #else: #imbalanced overlap
                                #    if rtoffset < 1:
                                #        rtoffset = 1 / rtoffset
                                #rtoffsets[lpair] = rtoffset
                                combinedrange = nlrange + olrange
                                #overlap = min((omrt, nmrt)) - max((omlt, nmlt)) + 1 #not necesarily sold on this +1 yet
                                overlap = min((omrt, nmrt)) - max((omlt, nmlt)) #as long as minpoints > 1
                                percentoverlap = (overlap * 2) / combinedrange
                                rtoffsets[lpair] = percentoverlap
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
    p1.append(time() - t1)

    t2 = time()
    groupsbypair = defaultdict(set)
    scoresbypair = defaultdict(dict) #mass: pair: [scores]
    secondpriorities = defaultdict(dict) #essentially there's too many zeros caused by single-pair matches, they can be let in but they don't deserve to priority
    for gv, sgp in connections.items():
        slen = len(sgp)
        if slen > 1:
            sgmean = sum(sgp.values()) / slen
            for sgk, sgs in sgp.items():
                groupsbypair[sgk].add(gv)
                offset = rtoffsets[sgk]
                #scoreval = abs(sgmean - sgs)
                scoreval = abs(sgmean - sgs) / (slen + 1)
                #scoreval = abs(sgmean - sgs) / (slen + 1) / paircharges[sgk]
                #difftomeans[group][sgv][sgk] = scoreval
                outscore = scoreval + scoreval * offset
                ddiff = datapdiffs[sgk]
                outdiff = ddiff + ddiff * offset
                #scorelist = [scoreval, datapdiffs[sgk]]
                scorelist = [outscore, outdiff]
                for m in sgk:
                    scoresbypair[m][sgk] = scorelist
        else:
            sgk, sgs = zip(*sgp.items())
            sgk = sgk[0]
            sgs = sgs[0]
            groupsbypair[sgk].add(gv)
            offset = rtoffsets[sgk]
            ddiff = datapdiffs[sgk]
            #outdiff = ddiff + ddiff * offset
            #outscore = sgs + sgs * offset
            for m in sgk:
                #cscore = datapdiffs[sgk]
                #secondpriorities[m][sgk] = datapdiffs[sgk]
                secondpriorities[m][sgk] = [sgs, ddiff]
                #secondpriorities[m][sgk] = [outscore, outdiff]
                #secondpriorities[m][sgk] = [sgs/cscore, cscore]
                #secondpriorities[m][sgk] = sgs / datapdiffs[sgk]

    grouplengths = {k: len(v) for k,v in connections.items()} #count of pairs per group

    rankedpairs = [] #[pair, minval]
    for m, pg in scoresbypair.items():
        pairs, vals = zip(*pg.items())
        #offsets = [rtoffsets[i] for i in pairs]
        vals = np.array(vals)
        vpercs = vals / vals.sum(axis=0).tolist()
        #sumvals = vpercs.sum(axis=1)
        minvals = vpercs.min(axis=1)
        #minvals = minvals + minvals * offsets
        rankedpairs.extend(list(zip(pairs, minvals)))

    secondrankedpairs = [] #[pair, minval]
    for m, pg in secondpriorities.items():
        pairs, vals = zip(*pg.items())
        offsets = [rtoffsets[i] for i in pairs]
        vals = np.array(vals)
        vpercs = vals / vals.sum(axis=0).tolist()
        minvals = np.abs(vpercs[:,1] - vpercs[:,0])
        minvals = minvals + minvals * offsets
        secondrankedpairs.extend(list(zip(pairs, minvals.tolist())))

    sortedranks = sorted(rankedpairs, key=lambda x: x[1])
#sortedranks = sorted(pairsums.items(), key=lambda x: x[1])
#sortedranks = sorted(pairsums.items(), key=lambda x: (paircharges[x[0]], x[1])) #favor lower charges and allow the supersets to assemble the appropriate higher ones? Didn't happen, more single-charges
#secondranks = sorted(secondrankedpairs)
    secondranks = sorted(secondrankedpairs, key=lambda x: x[1])
#secondranks = sorted(useableseconds.items(), key=lambda x: (paircharges[x[0]], -x[1])) #favoring lower charges here prevents a lot of 40+ charges from happening, the lone-pairs in here are prone to that occuring so this helps
    sortedranks.extend(secondranks)
#sortedranks = secondranks + sortedranks
#when a 1/2 charge takes priority over the rightful 1x
#if a 2x charge connects to the hypothetical 1/2 before the 1/2 gets extended, it converts it
    for k, v in rtoffsets.items():
        if np.isinf(v):
            rtoffsets[k] = 1
    overlapranks = sorted(rtoffsets.items(), key=lambda x: -x[1])

    preservedpairs = set()
    preservedranks = []
    for pair, score in sortedranks:
        if pair not in preservedpairs:
            preservedranks.append([pair, score])
            preservedpairs.add(pair)

#in order to address the vulnerability of distributions being constructed that don't fit the original inc/dec intensity requirements due to the free-flowing nature of this ranking,  I'm going to attempt a second ranking that ends up ranking distributions and their completion orders.
#all members of sortedranks are unique, use a non-binding ranking, the below loop that makes solodists is binding - meaning it has logic-blocking properties, the non-binding is free-flowing in the sense that it allows everything to be made.
#^I've also allowed for every combination of dists above when forming connections, so the current strategy is to allow distribution supersets to combine with groups as they 'complete' within the ranking, and their members are blocked 

    rank = 0
    groupranks = []
    initialsighting = set()
    paircounts = defaultdict(int)
    pairtracks = defaultdict(list)
    overpairtracks = defaultdict(list)
    for pn, ((pair1, score1), (pair2, score2)) in enumerate(zip(preservedranks, overlapranks)):
        pairs = [pair1, pair2]
        outpairs = []
        for pair in pairs:
            overpairtracks[pair].append(pn)
            if pair in initialsighting:
                outpairs.append(pair)
            else:
                initialsighting.add(pair)
        if len(outpairs) > 1:
            ntracks = [overpairtracks[i] for i in outpairs]
            groupsort = sorted(zip(*(ntracks, outpairs)), key=lambda x: (x[0][0], len(x[0])))
            outpairs = list(zip(*groupsort))[1]
        for pair in outpairs:
            gvs = groupsbypair[pair]
            newgroups = []
            for gv in gvs:
                paircounts[gv] += 1
                pairtracks[gv].append(pn)
                if paircounts[gv] == grouplengths[gv]:
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
    p2.append(time() - t2)

    t3 = time()
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
    p3.append(time() - t3)
    px.append(et-st)
