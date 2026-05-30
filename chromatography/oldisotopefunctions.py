

#inner space-handling

rgblow = lambda: (np.random.uniform(low=0.6, high=1), np.random.uniform(low=0.9, high=1), np.random.uniform(low=0.7, high=1))
rgbhigh = lambda: (np.random.uniform(low=0.9, high=1), np.random.uniform(low=0, high=0.1), np.random.uniform(low=0.8, high=0.9))
deepset = lambda: (np.random.uniform(low=0.2, high=1), np.random.uniform(low=0.4, high=1), np.random.uniform(low=0.1, high=0.8))

cfunc = lambda: (
        np.random.uniform(low=0.3, high=1), #R
        np.random.uniform(low=0.6, high=1), #G
        np.random.uniform(low=0.8, high=0.9) #B
        )
ndfunc = lambda: (
        np.random.uniform(low=0.9, high=1), #R
        np.random.uniform(low=0.1, high=0.6), #G
        np.random.uniform(low=0.1, high=0.6) #B
        )

with open(isotopefile, "rb") as pick:
    isotopeabundances, seqsbymass, subisotopicdifferences, newinclimit, steplimit = pickle.load(pick)

subisodiffs = np.array(list(subisotopicdifferences))[:,None]
subisotree = spatial.KDTree(subisodiffs)

st = 49.5
et = 49.6
lmb = 364.5
umb = 367.7

st = 35
et = 35.5
lmb = 321
umb = 325.2

st = 33
et = 33.8
lmb = 352.9
umb = 354.3

st = 53.4
et = 54.2
lmb = 524.5
umb = 526.5

#current example for OTF re-ranking
st = 48
et = 51
lmb = 655
umb = 665


st = 49.8
st = 49
et = 49.9
et = 52
lmb = 522.2
umb = 528.3

boundrec = [lmb, umb, st, et]
plotkeys = arg_coord_rectangle_overlap(boundrec, regions[:,:4]).tolist()

minpoints = 3
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
for nm in masses:
    olddirections = activedirection.copy()
    #subgroupchanges = defaultdict(lambda: defaultdict(set))
    masspoolremovals = set()
    nmlt, nmrt = timeranges[nm]
    nlrange = nmrt - nmlt
    for om in masspool:
        omlt, omrt = timeranges[om]
        olrange = omrt - omlt
        #overpass = False
        #if nmlt < omrt and nmrt > omlt: #rt's overlap
        #    if nmlt > omlt and nmrt < omrt: #new encompassed
        #        overpass = True
        #    elif omlt > nmlt and omrt < nmrt: #old encompassed
        #        overpass = True
        #    else:
        #        overlap = min((omrt, nmrt)) - max((omlt, nmlt))
        #        no = overlap / nlrange
        #        oo = overlap / olrange
        #        if no > 0.5 and oo > 0.5: #max of new/old overlap > 0.5, a majority overlap for both
        #            overpass = True
        overpass = False
        if nmlt < omrt and nmrt > omlt: #rt's overlap
            overlap = min(omrt, nmrt) - max(omlt, nmlt)
            if nmlt > omlt and nmrt < omrt: #new encompassed
                combinedrange = nlrange + olrange
                percentoverlap = (overlap * 2) / combinedrange
                #using newinclimit here is appropriate because it's directly related to expected differences in adjacent isotopomer quantities, which is what the overlap is also ~somewhat assessing
                if percentoverlap > newinclimit:
                    overpass = True
            elif omlt > nmlt and omrt < nmrt: #old encompassed
                combinedrange = nlrange + olrange
                percentoverlap = (overlap * 2) / combinedrange
                if percentoverlap > newinclimit:
                    overpass = True
            else:
                #no = overlap / nlrange
                #oo = overlap / olrange
                #if no > 0.5 and oo > 0.5: #max of new/old overlap > 0.5, a majority overlap for both -> 0.75 now because this shit was too lenient, this might be ok for a hard-coded value
                fullrange = max(omrt, nmrt) - min(omlt, nmlt)
                percentoverlap = overlap / fullrange
                if percentoverlap * 2 > 0.75: #this is super lenient I think
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
                diffcut = expdiff * chargetolerance
                if acdiff > -1 * (diffcut * chargetolerance + widthbuffer): #a lighter acceptance for distances above the charge dist - it does happen from time to time on lower-hanging straggler isotopomers
                    if acdiff <= diffcut + widthbuffer:
                        absacdiff = abs(acdiff) * charge
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
                                    connections[di][lpair] = absacdiff
                                    subgroups[nm][charge].add(di)
                                    if sst < sam:
                                        activedirection[di] += 1
                                    ncons += 1
                                    di += 1
                        else:
                            #no previous subgroup
                            if intensitypercdiff <= steplimit:
                                lpair = (om, nm)
                                connections[di][lpair] = absacdiff
                                subgroups[nm][charge].add(di)
                                if sst < sam:
                                    #decreasing
                                    activedirection[di] += 1
                                ncons += 1
                                di += 1
                        #version without activedirection
                        #if csubs:
                        #    for adi in csubs:
                        #        lpair = (om, nm)
                        #        referenced[om].add(adi)
                        #        connections[di].update(connections[adi])
                        #        connections[di][lpair] = absacdiff
                        #        subgroups[nm][charge].add(di)
                        #        ncons += 1
                        #        di += 1
                        #else:
                        #    #no previous subgroup
                        #    lpair = (om, nm)
                        #    connections[di][lpair] = absacdiff
                        #    subgroups[nm][charge].add(di)
                        #    ncons += 1
                        #    di += 1
                        if ncons > 0:
                            npoints = rpoints[nm]
                            apoints = rpoints[om]
                            dpercdiff = abs(npoints - apoints) / (npoints + apoints) / 2
                            datapdiffs[lpair] = dpercdiff
                            paircharges[lpair] = charge
                            #combinedrange = nlrange + olrange
                            #overlap = min((omrt, nmrt)) - max((omlt, nmlt)) +  1 #not necesarily sold on this +1 yet
                            #overlap = min((omrt, nmrt)) - max((omlt, nmlt)) #as long as minpoints > 1
                            #percentoverlap = (overlap * 2) / combinedrange
                            rtoffsets[lpair] = percentoverlap
                            #rtoffsets[lpair] = percentoverlap * percentequalizedoverhang
                            if csubs:
                                connections[di][lpair] = absacdiff
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
            offset = rtoffsets[sgk]
            #scoreval = abs(sgmean - sgs)
            scoreval = abs(sgmean - sgs) / (slen + 1) 
            #scoreval = abs(sgmean - sgs) / (slen + 1) / paircharges[sgk]
            #difftomeans[group][sgv][sgk] = scoreval
            outscore = scoreval - scoreval * offset
            ddiff = datapdiffs[sgk]
            outdiff = ddiff - ddiff * offset
            #scorelist = [scoreval, ddiff]
            scorelist = [outscore, outdiff]
            for m in sgk:
                scoresbypair[m][sgk] = scorelist
    else:
        sgk, sgs = zip(*sgp.items())
        sgk = sgk[0]
        sgs = sgs[0]
        groupsbypair[sgk].add(gv)
        #offset = rtoffsets[sgk]
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
    minvals = vpercs.sum(axis=1)
    #minvals = minvals + minvals * offsets
    rankedpairs.extend(list(zip(pairs, minvals)))

secondrankedpairs = [] #[pair, minval]
for m, pg in secondpriorities.items():
    pairs, vals = zip(*pg.items())
    offsets = [rtoffsets[i] for i in pairs]
    vals = np.array(vals)
    vpercs = vals / vals.sum(axis=0).tolist()
    minvals = np.abs(vpercs[:,1] - vpercs[:,0])
    minvals = minvals - minvals * offsets
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
finalranks = []
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
        finalranks.append(pair)
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

#original ranking system
dr =  0
franks = {} #distid: rank
blocked = set()
solodists = defaultdict(dict)
for ngn, (gs, (gv, sgc)) in enumerate(zip(groupsets, groupranks)):
    ng = set()
    gcs = set()
    ng.update(gs)
    gcs.add(sgc)
    if not blocked.intersection(gs): 
        ngit = groupsets[ngn+1:]
        for ogn, ogs in enumerate(ngit):
            if ng < ogs: #this will not partially connect two differing charges
                if not blocked.intersection(ogs):
                    ng.update(ogs)
                    ngc = groupranks[ngn+ogn+1][1]
                    gcs.add(ngc)
        mgc = max(gcs)
        solodists[mgc][gv] = list(ng)
        blocked.update(ng) 
        for g in gs:
            linelocations[g] = [mgc, gv]
        franks[gv] = dr
        dr += 1

#I'm thinking of implementing a tradeoff system whereby if giving an edge piece to another distribution allows for less total distance to expected proton distance, this would allow for the exchange. It would purely be edge pieces, and in a moving sense as well (ie dynamically considered for when an edge piece is lost), since those mostly seem to be the more problematic matches.
#^could this be the basis for ignoring the need for a chargetolerance?
#dr =  0
#franks = {} #distid: rank
#blocked = set()
#solodists = defaultdict(dict)
#linelocations = {}
#for ngn, (gs, (gv, sgc)) in enumerate(zip(groupsets, groupranks)):
#    ng = set()
#    gcs = set()
#    ng.update(gs)
#    gcs.add(sgc)
#    if not blocked.intersection(gs): 
#        ngit = groupsets[ngn+1:]
#        for ogn, ogs in enumerate(ngit):
#            if ng < ogs: #this will not partially connect two differing charges
#                blocktest = blocked.intersection(ogs)
#                if not blocktest:
#                    ng.update(ogs)
#                    ngc = groupranks[ngn+ogn+1][1]
#                    gcs.add(ngc) 
#                elif len(blocktest) == 1:
#                    bt = list(blocktest)[0]
#                    blockkeys = linelocations[list(blocktest)[0]]
#                    blockdist = solodists[blockkeys[0]][blockkeys[1]]
#                    bsdist = sorted(blockdist)
#                    if bt == bsdist[0] or bt == bsdist[-1]:
#                        bint = bsdist.index(bt) - 1
#                        if bint < 0:
#                            bint = 0
#                        ogtest = sorted(ogs)
#                        ogdiff = np.diff(ogtest)
#                        ngc = blockkeys[0]
#                        bsdiff = np.diff(bsdist)
#                        ogc = groupranks[ngn+ogn+1][1]
#                        nent = np.abs(proton - ogdiff * ogc)
#                        bent = np.abs(proton - bsdiff * ngc)
#                        if bent[bint] > np.delete(bent, bint).sum():
#                            oint = ogtest.index(bt) - 1
#                            if oint < 0:
#                                oint = 0
#                            if nent[oint] < bent[bint]:
#                                #replacement earned
#                                print(ngn, ogn)
#        mgc = max(gcs)
#        solodists[mgc][gv] = list(ng)
#        blocked.update(ng) 
#        for g in ng:
#            linelocations[g] = [mgc, gv]
#        franks[gv] = dr
#        dr += 1

##no jumping ahead on this one
##simple ordered ascension, make groups on the fly and merge them when appropriate
#distsets = [] #sets of distributions
#linelocations = {} #mass: index of distsets
#setcharges = defaultdict(set) #index of distsets: [charges]
##for ngn, (gs, (gv, sgc)) in enumerate(zip(groupsets, groupranks)):
#for ngn, gs in enumerate(finalranks):
#    sgc = paircharges[gs]
#    gs = set(gs)
#    locs = set()
#    for g in gs:
#        if g in linelocations:
#            locs.add(linelocations[g])
#    if locs:
#        if len(locs) == 1:
#            ogn = list(locs)[0]
#            ogs = distsets[ogn]
#            #if ogs < gs:
#            if ogs.union(gs) in groupsets:
#                distsets[ogn].update(gs)
#                setcharges[ogn].add(sgc)
#                for g in gs:
#                    if g not in linelocations:
#                        linelocations[g] = ogn
#        else:
#            ogt = gs.copy()
#            for l in locs:
#                ogt.update(distsets[l])
#            if ogt in groupsets:
#                ogn = min(locs)
#                distsets[ogn].update(ogt)
#                for l in locs.difference([ogn]):
#                    setcharges[ogn].update(setcharges[l])
#                    setcharges[l] = False
#                    distsets[l] = False
#                for g in ogt:
#                    linelocations[g] = ogn
#    else:
#        distsets.append(gs)
#        ogn = len(distsets) - 1
#        setcharges[ogn].add(sgc)
#        for g in gs:
#            linelocations[g] = ogn
#
#dr =  0
#franks = {} #distid: rank
#solodists = defaultdict(dict)
#for dist, charges in zip(distsets, setcharges.values()):
#    if dist:
#        charge = max(charges)
#        solodists[charge][dr] = list(dist)
#        franks[dr] = dr #just roll with it
#        dr += 1

foundvals = []
for charge, sgd in solodists.items():
    foundvals.extend(list(itertools.chain(*sgd.values())))
specvals = set(spectrum.keys())
nodists = list(specvals.difference(foundvals))

#for gs in groupsets:
#    gintersect = gs.intersection(nodists)
#    if len(gintersect) == len(gs) - 1:
##then after check if these can actually merge with other existing distributions

text = True
ngroups = sum(len(i) for i in solodists.values())
cols = dp.get_colors(ngroups)
cn = 0
fig, ax = plt.subplots(nrows=3, figsize=(6,8), sharex=True)
for fc, fgs in solodists.items():
    for fk, fg in fgs.items():
        col = cols[cn]
        cn += 1
        pkeys = [rlookup[i] for i in fg]
        for p in pkeys:
            a = np.array(trackedgroups[p])
            ax[2].scatter(a[0], a[1], marker='.', color=col, s=0.3, alpha=0.3)
            #ax[2].plot(a[0], a[1], '-', color=col, linewidth=0.2, alpha=0.8)
            #mvar = len(a[0]) / np.ptp(a[1])
            #mstring = str(np.format_float_scientific(mvar, 2))
            #mstring = str(mvar.round(4))
            if text:
                ax[2].text(a[0][-1], a[1][-1], str(regions[p,7].round(2)), color='white', fontsize=4)
        fints = [spectrum[i] for i in fg]
        #distrank = min([v for k, v in frank.items() if set(k).issubset(fg)])
        distrank = franks[fk]
        #flabel = ' - '.join((str(fc), str(distrank)))
        #flabel = str(fc)
        ax[0].bar(fg, fints, color=col, alpha=0.5, width=0.01, label=str(fc))
        if text:
            for fx, fy in zip(fg, fints):
                ax[0].text(fx, fy + fy * 0.03, str(rlookup[fx]), color='white', fontsize=4)
        print(fg)
        print(fc, '-', np.diff(sorted(fg)))
        print('~')
        ax[1].hlines(distrank, min(fg), max(fg), color=col, linewidth=0.6)
        for vert, pl in zip(fg, pkeys):
            ax[1].vlines(vert, distrank - 0.1, distrank + 0.1, color=col, linewidth=0.6)
            #if text:
                #ax[1].text(vert, distrank + 0.1, rpoints[vert], fontsize=4, ha='center', color='white')
        vi = np.sort(fg)
        if vi.size > 2:
            vstack = np.stack((vi[:-1], vi[1:]), axis=1)
            editspots = np.diff(vstack) < subisomax
            if editspots.any():
                ewheres = np.where(editspots)[0].tolist()
                for ew in ewheres:
                    subpair = vstack[ew].tolist()
                    subints = [spectrum[i] for i in subpair]
                    winint = subints.index(max(subints))
                    winner = subpair[winint]
                    if ew > 0:
                        #edit 1 before ew
                        vstack[ew-1,1] = winner
                    if ew < len(vstack) - 1:
                        #edit 1 after ew
                        vstack[ew+1,0] = winner
                vstack = np.delete(vstack, ewheres, axis=0)
        else:
            vstack = vi.reshape(1, -1)
        vdiffs = np.diff(vstack)
        vflat = sorted(vstack.flatten().tolist())
        labelspots = np.mean(vstack, axis=1).tolist()
        for ls, lp in zip(labelspots, vstack.tolist()):
            labeldiff = np.diff(lp)[0].round(4)
            chargedist = (proton/fc - labeldiff).round(4)
            lstring = ' ~ '.join((str(fc), str(labeldiff), str(chargedist)))
            if text:
                ax[1].text(ls, distrank - 0.2, lstring, fontsize=4, ha='center', color='white')
        #heightcounter += 1

if nodists:
    nints = [spectrum[i] for i in nodists]
    ax[0].bar(nodists, nints, alpha=0.5, color='white', width=0.01, label='N/A')
    if text:
        for fx, fy in zip(nodists, nints):
            ax[0].text(fx, fy + fy * 0.03, str(rlookup[fx]), color='white', fontsize=4)
    pkeys = [rlookup[i] for i in nodists]
    for p in pkeys:
        a = np.array(trackedgroups[p])
        #ainds = a[:,np.logical_and(np.logical_and(a[0] >= lmb, a[0] <= umb), np.logical_and(a[1] >= st, a[1] <= et))]
        fx = regions[p,7]
        #
        ax[2].scatter(a[0], a[1], marker='.', color='white', s=0.3, alpha=0.3)
        #ax[2].plot(a[0], a[1], '-', color='white', linewidth=0.2, alpha=0.8)
        #mvar = len(a[0]) / np.ptp(a[1])
        #mstring = str(np.format_float_scientific(mvar, 2))
        #mstring = str(mvar.round(4))
        if text:
            ax[2].text(a[0][-1], a[1][-1], str(regions[p,7].round(2)), color='white', fontsize=4)
ax[0].set_yscale('log')
ax[0].set_ylabel('intensity')
ax[2].set_ylabel('minutes')
ax[1].set_ylabel('distribution rank')
ax[2].set_xlabel('m/z')
for label in ax[2].get_xticklabels():
    #label.set_ha("right")
    label.set_rotation(-45)
ncols = 6
if text:
    ax[0].legend(title='Charge', loc='upper left', bbox_to_anchor=(0, 1.3 + (0.1 * cn / ncols)), ncol=ncols)
fig.tight_layout()
fig.subplots_adjust(hspace=0.05)
plt.show()
fig.clf()
plt.close()






#~~~~~~~~~~~~~~





#new processing paradigm
#^I will say, this one does improve final distribution selections, but when looking at narrowed time/mass ranges it might not look that way - this thrives under higher competition.
rgblow = lambda: (np.random.uniform(low=0.6, high=1), np.random.uniform(low=0.9, high=1), np.random.uniform(low=0.7, high=1))
rgbhigh = lambda: (np.random.uniform(low=0.9, high=1), np.random.uniform(low=0, high=0.1), np.random.uniform(low=0.8, high=0.9))
deepset = lambda: (np.random.uniform(low=0.2, high=1), np.random.uniform(low=0.4, high=1), np.random.uniform(low=0.1, high=0.8))

cfunc = lambda: (
        np.random.uniform(low=0.3, high=1), #R
        np.random.uniform(low=0.6, high=1), #G
        np.random.uniform(low=0.8, high=0.9) #B
        )
ndfunc = lambda: (
        np.random.uniform(low=0.9, high=1), #R
        np.random.uniform(low=0.1, high=0.6), #G
        np.random.uniform(low=0.1, high=0.6) #B
        )

with open(isotopefile, "rb") as pick:
    isotopeabundances, seqsbymass, subisotopicdifferences, newinclimit, steplimit = pickle.load(pick)

subisodiffs = np.array(list(subisotopicdifferences))[:,None]
subisotree = spatial.KDTree(subisodiffs)

st = 49.5
et = 49.6
lmb = 364.5
umb = 367.7

st = 35
et = 35.5
lmb = 321
umb = 325.2

st = 33
et = 33.8
lmb = 352.9
umb = 354.3

st = 53.4
et = 54.2
lmb = 524.5
umb = 526.5

#current example for OTF re-ranking
st = 48
et = 51
lmb = 655
umb = 665

st = 49.8
st = 49
et = 49.9
et = 52
lmb = 522.2
umb = 528.3

boundrec = [lmb, umb, st, et]
plotkeys = arg_coord_rectangle_overlap(boundrec, regions[:,:4]).tolist()

minpoints = 3
chargetolerance = 0.1

subisomax = subisodiffs.max()
subisomax = subisomax + subisomax * chargetolerance

spectrum = {}
masswidths = {}
timeranges = {}
rlookup = {}
rpoints = {}
for n, k in enumerate(plotkeys):
    pregion = regions[k]
    intensity = pregion[5]
    a = trackedgroups[k]
    mmass = pregion[7]
    klen = pregion[4]
    if klen >= minpoints:
        spectrum[mmass] = intensity
        masswidths[mmass] = pregion[1]  - pregion[0]
        timeranges[mmass] = [pregion[2],  pregion[3]]
    rlookup[mmass] = k
    rpoints[mmass] = klen

masses = sorted(spectrum.keys())

di = 0
connections = defaultdict(set) #groupid: [pairs of lines]
subgroups = defaultdict(lambda: defaultdict(set)) #mass: charge: groups, a mass is always active as long as it's in masspool, this keeps track of all the groups that end in any given mass
activedirection = defaultdict(int) #groupid: direction, 0 (increasing) or 1 (decreasing)

pairscoresbymass = defaultdict(dict) #mass: pair: [[absacdiff, datapercdiff, rtoffset],]
scoresbypair = {} #pair: [[absacdiff, datapercdiff, rtoffset],]
paircharges = {} #connection: charge

si = 0
subisogroups = defaultdict(lambda: defaultdict(set)) #subiso group: max charge for mass: [masses]
subisomasses = {} #mass: subisogroup

masspool = set()
for nm in masses:
    masspoolremovals = set()
    nmlt, nmrt = timeranges[nm]
    nlrange = nmrt - nmlt
    for om in masspool:
        omlt, omrt = timeranges[om]
        olrange = omrt - omlt
        overpass = False
        if nmlt < omrt and nmrt > omlt: #rt's overlap
            overlap = min(omrt, nmrt) - max(omlt, nmlt)
            if nmlt > omlt and nmrt < omrt: #new encompassed
                combinedrange = nlrange + olrange
                percentoverlap = (overlap * 2) / combinedrange
                #using newinclimit here is appropriate because it's directly related to expected differences in adjacent isotopomer quantities (linear-scale # data points ~= log-scale area), which is what the overlap is also ~somewhat assessing
                if percentoverlap > newinclimit:
                    overpass = True
            elif omlt > nmlt and omrt < nmrt: #old encompassed
                combinedrange = nlrange + olrange
                percentoverlap = (overlap * 2) / combinedrange
                if percentoverlap > newinclimit:
                    overpass = True
            else:
                fullrange = max(omrt, nmrt) - min(omlt, nmlt)
                percentoverlap = overlap / fullrange
                if percentoverlap * 2 > 0.75: #this is super lenient I think
                    overpass = True
        if overpass:
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
                diffcut = expdiff * chargetolerance
                if acdiff > -1 * (diffcut * chargetolerance + widthbuffer): #a lighter acceptance for distances above the charge dist - it does happen from time to time on lower-hanging straggler isotopomers
                    if acdiff <= diffcut + widthbuffer:
                        absacdiff = abs(acdiff) * charge
                        sst = spectrum[nm]
                        sam = spectrum[om]
                        intensitypercdiff = abs(sst - sam) / (sst + sam) / 2
                        ncons = 0
                        csubs = subgroups[om][charge]
                        if csubs:
                            for adi in csubs:
                                ratiocheck = steplimit
                                if activedirection[adi] > 0:
                                    #the dist has previously begun a decrease
                                    if sst >= sam:
                                        #and intensity is increasing
                                        ratiocheck = newinclimit
                                if intensitypercdiff <= ratiocheck:
                                    lpair = (om, nm)
                                    connections[di].update(connections[adi])
                                    connections[di].add(lpair)
                                    subgroups[nm][charge].add(di)
                                    if sst < sam:
                                        activedirection[di] += 1
                                    ncons += 1
                                    di += 1
                        else:
                            #no previous subgroup
                            if intensitypercdiff <= steplimit:
                                lpair = (om, nm)
                                connections[di].add(lpair)
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
                            paircharges[lpair] = charge
                            scorelist = [absacdiff, dpercdiff, percentoverlap]
                            for p in lpair:
                                pairscoresbymass[p][lpair] = scorelist
                            scoresbypair[lpair] = scorelist
                            if csubs:
                                connections[di].add(lpair)
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
distmeanscores = defaultdict(dict) #mass: pair: [scores]
for gid, pairs in connections.items():
    slen = len(pairs)
    if slen > 1:
        scores = np.array([scoresbypair[i] for i in pairs])
        distmean = scores[:,0].mean()
        for pair, score in zip(pairs, scores.tolist()): 
            groupsbypair[pair].add(gid)
            dist, ddiff, rtoffset = score
            #meandiff = abs(distmean - dist) / (slen + 1) 
            meandiff = abs(distmean - dist) / slen
            distdiff = meandiff - meandiff * rtoffset
            datadiff = ddiff - ddiff * rtoffset
            scorelist = [distdiff, datadiff]
            for m in pair:
                distmeanscores[m][pair] = scorelist
    else:
        pair = list(pairs)[0]
        groupsbypair[pair].add(gid)

rankedpairs = [] #[pair, minval]
for m, pg in distmeanscores.items():
    pairs, vals = zip(*pg.items())
    vals = np.array(vals)
    vpercs = vals / vals.sum(axis=0).tolist()
    sumvals = vpercs.sum(axis=1)
    rankedpairs.extend(list(zip(pairs, sumvals)))


#I somehow managed to do this not entirely as I had perceived previously, things are supposed to be added to pairscoresbymass here VIA their individual masses, and anything that has only a single match THEN gets added to secondpriorities. connections with only 1 match were being added to 2ndprios here without any wider assessment...
#rtoffsets = {} #pair: percent overlap
secondpriorities = [] #[pair, score]
thirdpriorities = [] #[pair, score]
for line, pairdict in pairscoresbymass.items():
    plen = len(pairdict)
    if plen > 1:
        pairs, scores = zip(*pairdict.items())
        scorearray = np.array(scores)
        #offsetscores = scorearray[:,:2] - scorearray[:,:2] * scorearray[:,2,None]
        #normoffsets = offsetscores / offsetscores.sum(axis=0)
        #for pair, score, rtoffset in zip(pairs, normoffsets.tolist(), scorearray[:,2].tolist()):
        #    secondpriorities.append([pair, sum(score)])
        #    rtoffsets[pair] = rtoffset
        #try this? idk
        normscores = scorearray / scorearray.sum(axis=0)
        offsetnorms = normscores[:,:2] - normscores[:,:2] * scorearray[:,2,None]
        for pair, score, rtoffset in zip(pairs, offsetnorms.tolist(), scorearray[:,2].tolist()):
            secondpriorities.append([pair, sum(score)])
#            rtoffsets[pair] = rtoffset
    else:
        pair, scores = zip(*pairdict.items())
        pair = pair[0]
        scores = scores[0]
        scoresum = scores[0] + scores[1]
        s1norm = scores[0] / scoresum
        s2norm = scores[1] / scoresum
        equalizednorm = abs(s2norm - s1norm)
        #rt offset after normalization was normal for 2ndprios previously
        outscore = equalizednorm - equalizednorm * scores[2]
        thirdpriorities.append([pair, outscore])
        #rtoffsets[pair] = scores[2]


grouplengths = {k: len(v) for k,v in connections.items()} #count of pairs per group
#groupsbypair = defaultdict(set) #masses: [groupids]
#for di, lines in connections.items():
#    for line in lines:
#        for pair in scoresbypair[line]:
#            if len(lines.intersection(pair)) == 2:
#                groupsbypair[pair].add(di)


sortedranks = sorted(rankedpairs, key=lambda x: x[1])
secondranks = sorted(secondpriorities, key=lambda x: x[1])
thirdranks = sorted(thirdpriorities, key=lambda x: x[1])
sortedranks.extend(secondranks)
sortedranks.extend(thirdranks)

#will this still happen?
#for k, v in rtoffsets.items():
#    if np.isinf(v):
#        rtoffsets[k] = 1

#overlapranks = sorted(rtoffsets.items(), key=lambda x: -x[1])

preservedpairs = set()
preservedranks = []
for pair, score in sortedranks:
    if pair not in preservedpairs:
        preservedranks.append([pair, score])
        preservedpairs.add(pair)

#groupranks = []
#initialsighting = set()
#paircounts = defaultdict(int)
#pairtracks = defaultdict(list)
#overpairtracks = defaultdict(list)
#for pn, ((pair1, score1), (pair2, score2)) in enumerate(zip(preservedranks, overlapranks)):
#    pairs = [pair1, pair2]
#    outpairs = []
#    for pair in pairs:
#        overpairtracks[pair].append(pn)
#        if pair in initialsighting:
#            outpairs.append(pair)
#        else:
#            initialsighting.add(pair)
#    if len(outpairs) > 1:
#        ntracks = [overpairtracks[i] for i in outpairs]
#        groupsort = sorted(zip(*(ntracks, outpairs)), key=lambda x: (x[0][0], len(x[0])))
#        outpairs = list(zip(*groupsort))[1]
#    for pair in outpairs:
#        gvs = groupsbypair[pair]
#        newgroups = []
#        for gv in gvs:
#            paircounts[gv] += 1
#            pairtracks[gv].append(pn)
#            if paircounts[gv] == grouplengths[gv]:
#                newgroups.append(gv)
#        if newgroups:
#            pc = paircharges[pair]
#            if len(newgroups) > 1:
#                ntracks = [pairtracks[i] for i in newgroups]
#                groupsort = sorted(zip(*(ntracks, newgroups)), key=lambda x: (x[0][0], len(x[0])))
#                newgroups = list(zip(*groupsort))[1]
#            for gv in newgroups:
#                groupranks.append([gv, pc])
#
#groupsets = [set(itertools.chain(*connections[g])) for g in list(zip(*groupranks))[0]]

##original ranking system
#dr =  0
#franks = {} #distid: rank
#blocked = set()
#solodists = defaultdict(dict)
#for ngn, (gs, (gv, sgc)) in enumerate(zip(groupsets, groupranks)):
#    ng = set()
#    gcs = set()
#    ng.update(gs)
#    gcs.add(sgc)
#    if not blocked.intersection(gs): 
#        ngit = groupsets[ngn+1:]
#        for ogn, ogs in enumerate(ngit):
#            if ng < ogs: #this will not partially connect two differing charges
#                if not blocked.intersection(ogs):
#                    ng.update(ogs)
#                    ngc = groupranks[ngn+ogn+1][1]
#                    gcs.add(ngc) 
#        solodists[max(gcs)][gv] = list(ng)
#        blocked.update(ng) 
#        franks[gv] = dr
#        dr += 1

##modded for finalranks - not working
#dr =  0
#franks = {} #distid: rank
#blocked = set()
#solodists = defaultdict(dict)
#for ngn, gs in enumerate(finalranks):
#    sgc = paircharges[gs]
#    gs = set(gs)
#    ng = set()
#    gcs = set()
#    ng.update(gs)
#    gcs.add(sgc)
#    if not blocked.intersection(gs): 
#        ngit = finalranks[ngn+1:]
#        for ogn, ogs in enumerate(ngit):
#            #if not blocked.intersection(ogs):
#            gu = ng.union(ogs)
#            if gu in groupsets: #this will not partially connect two differing charges
#                ng.update(ogs)
#                ngc = paircharges[ogs]
#                gcs.add(ngc) 
#        solodists[max(gcs)][gv] = list(ng)
#        #blocked.update(ng) 
#        franks[gv] = dr
#        dr += 1



#paradigm split above/below

#output final ranked list, not finalized distributions
#finalranks = []
#initialsighting = set()
#paircounts = defaultdict(int)
#pairtracks = defaultdict(list)
#overpairtracks = defaultdict(list)
#for pn, ((pair1, score1), (pair2, score2)) in enumerate(zip(preservedranks, overlapranks)):
#    pairs = [pair1, pair2]
#    outpairs = []
#    for pair in pairs:
#        overpairtracks[pair].append(pn)
#        if pair in initialsighting:
#            outpairs.append(pair)
#        else:
#            initialsighting.add(pair)
#    if len(outpairs) > 1:
#        ntracks = [overpairtracks[i] for i in outpairs]
#        groupsort = sorted(zip(*(ntracks, outpairs)), key=lambda x: (x[0][0], len(x[0])))
#        outpairs = list(zip(*groupsort))[1]
#    for pair in outpairs:
#        finalranks.append(pair)

groupsets = [set(itertools.chain(*g)) for g in connections.values()]

#no jumping ahead on this one
#simple ordered ascension, make groups on the fly and merge them when appropriate
distsets = [] #sets of distributions
linelocations = {} #mass: index of distsets
setcharges = defaultdict(set) #index of distsets: [charges]
#for ngn, (gs, (gv, sgc)) in enumerate(zip(groupsets, groupranks)):
#for ngn, gs in enumerate(finalranks):
for ngn, (gs, score) in enumerate(preservedranks):
    sgc = paircharges[gs]
    gs = set(gs)
    locs = set()
    for g in gs:
        if g in linelocations:
            locs.add(linelocations[g])
    if locs:
        if len(locs) == 1:
            ogn = list(locs)[0]
            ogs = distsets[ogn]
            #if ogs < gs:
            if ogs.union(gs) in groupsets:
                distsets[ogn].update(gs)
                setcharges[ogn].add(sgc)
                for g in gs:
                    if g not in linelocations:
                        linelocations[g] = ogn
        else:
            ogt = gs.copy()
            for l in locs:
                ogt.update(distsets[l])
            if ogt in groupsets:
                ogn = min(locs)
                distsets[ogn].update(ogt)
                for l in locs.difference([ogn]):
                    setcharges[ogn].update(setcharges[l])
                    setcharges[l] = False
                    distsets[l] = False
                for g in ogt:
                    linelocations[g] = ogn
    else:
        #All of this is completely redundant! solodists and nodists are a perfect match without this!!!
        #for ogn, ogs in enumerate(distsets):
        #    if ogs.union(gs) in groupsets: #this will not partially connect two differing charges
        #        distsets[ogn].update(gs)
        #        setcharges[ogn].add(sgc)
        #        for g in gs:
        #            linelocations[g] = ogn
        #        break
        ##
        #if not any(g in linelocations for g in gs):
        distsets.append(gs)
        ogn = len(distsets) - 1
        setcharges[ogn].add(sgc)
        for g in gs:
            linelocations[g] = ogn

dr =  0
franks = {} #distid: rank
solodists = defaultdict(dict)
for dist, charges in zip(distsets, setcharges.values()):
    if dist:
        charge = max(charges)
        solodists[charge][dr] = list(dist)
        franks[dr] = dr #just roll with it
        dr += 1

#I'm thinking of implementing a tradeoff system whereby if giving an edge piece to another distribution allows for less total distance to expected proton distance, this would allow for the exchange. It would purely be edge pieces, and in a moving sense as well (ie dynamically considered for when an edge piece is lost), since those mostly seem to be the more problematic matches.
#^could this be the basis for ignoring the need for a chargetolerance?
#dr =  0
#franks = {} #distid: rank
#blocked = set()
#solodists = defaultdict(dict)
#linelocations = {}
#for ngn, (gs, (gv, sgc)) in enumerate(zip(groupsets, groupranks)):
#    ng = set()
#    gcs = set()
#    ng.update(gs)
#    gcs.add(sgc)
#    if not blocked.intersection(gs): 
#        ngit = groupsets[ngn+1:]
#        for ogn, ogs in enumerate(ngit):
#            if ng < ogs: #this will not partially connect two differing charges
#                blocktest = blocked.intersection(ogs)
#                if not blocktest:
#                    ng.update(ogs)
#                    ngc = groupranks[ngn+ogn+1][1]
#                    gcs.add(ngc) 
#                elif len(blocktest) == 1:
#                    bt = list(blocktest)[0]
#                    blockkeys = linelocations[list(blocktest)[0]]
#                    blockdist = solodists[blockkeys[0]][blockkeys[1]]
#                    bsdist = sorted(blockdist)
#                    if bt == bsdist[0] or bt == bsdist[-1]:
#                        bint = bsdist.index(bt) - 1
#                        if bint < 0:
#                            bint = 0
#                        ogtest = sorted(ogs)
#                        ogdiff = np.diff(ogtest)
#                        ngc = blockkeys[0]
#                        bsdiff = np.diff(bsdist)
#                        ogc = groupranks[ngn+ogn+1][1]
#                        nent = np.abs(proton - ogdiff * ogc)
#                        bent = np.abs(proton - bsdiff * ngc)
#                        if bent[bint] > np.delete(bent, bint).sum():
#                            oint = ogtest.index(bt) - 1
#                            if oint < 0:
#                                oint = 0
#                            if nent[oint] < bent[bint]:
#                                #replacement earned
#                                print(ngn, ogn)
#        mgc = max(gcs)
#        solodists[mgc][gv] = list(ng)
#        blocked.update(ng) 
#        for g in ng:
#            linelocations[g] = [mgc, gv]
#        franks[gv] = dr
#        dr += 1


foundvals = []
for charge, sgd in solodists.items():
    foundvals.extend(list(itertools.chain(*sgd.values())))
specvals = set(spectrum.keys())
nodists = list(specvals.difference(foundvals))

text = True
ngroups = sum(len(i) for i in solodists.values())
cols = dp.get_colors(ngroups)
cn = 0
fig, ax = plt.subplots(nrows=3, figsize=(6,8), sharex=True)
for fc, fgs in solodists.items():
    for fk, fg in fgs.items():
        col = cols[cn]
        low, high = rgblow(), rgbhigh()
        cn += 1
        pkeys = [rlookup[i] for i in fg]
        for p in pkeys:
            a = np.array(trackedgroups[p])
            ax[2].scatter(a[0], a[1], marker='.', color=col, s=0.3, alpha=0.3)
            if text:
                ax[2].text(a[0][-1], a[1][-1], str(regions[p,7].round(2)), color='white', fontsize=4)
        fints = [spectrum[i] for i in fg]
        distrank = franks[fk]
        ax[0].bar(fg, fints, color=col, alpha=0.5, width=0.01, label=str(fc))
        if text:
            for fx, fy in zip(fg, fints):
                ax[0].text(fx, fy + fy * 0.03, str(rlookup[fx]), color='white', fontsize=4)
        print(fg)
        print(fc, '-', np.diff(sorted(fg)))
        print('~')
        ax[1].hlines(distrank, min(fg), max(fg), color=col, linewidth=0.6)
        for vert, pl in zip(fg, pkeys):
            ax[1].vlines(vert, distrank - 0.1, distrank + 0.1, color=col, linewidth=0.6)
        vi = np.sort(fg)
        if vi.size > 2:
            vstack = np.stack((vi[:-1], vi[1:]), axis=1)
            editspots = np.diff(vstack) < subisomax
            if editspots.any():
                ewheres = np.where(editspots)[0].tolist()
                for ew in ewheres:
                    subpair = vstack[ew].tolist()
                    subints = [spectrum[i] for i in subpair]
                    winint = subints.index(max(subints))
                    winner = subpair[winint]
                    if ew > 0:
                        #edit 1 before ew
                        vstack[ew-1,1] = winner
                    if ew < len(vstack) - 1:
                        #edit 1 after ew
                        vstack[ew+1,0] = winner
                vstack = np.delete(vstack, ewheres, axis=0)
        else:
            vstack = vi.reshape(1, -1)
        vdiffs = np.diff(vstack)
        vflat = sorted(vstack.flatten().tolist())
        labelspots = np.mean(vstack, axis=1).tolist()
        for ls, lp in zip(labelspots, vstack.tolist()):
            labeldiff = np.diff(lp)[0].round(4)
            chargedist = (proton/fc - labeldiff).round(4)
            lstring = ' ~ '.join((str(fc), str(labeldiff), str(chargedist)))
            if text:
                ax[1].text(ls, distrank - 0.2, lstring, fontsize=4, ha='center', color='white')

if nodists:
    nints = [spectrum[i] for i in nodists]
    ax[0].bar(nodists, nints, alpha=0.5, color='white', width=0.01, label='N/A')
    if text:
        for fx, fy in zip(nodists, nints):
            ax[0].text(fx, fy + fy * 0.03, str(rlookup[fx]), color='white', fontsize=4)
    pkeys = [rlookup[i] for i in nodists]
    for p in pkeys:
        low, high = rgblow(), rgbhigh()
        a = np.array(trackedgroups[p])
        fx = regions[p,7]
        ax[2].scatter(a[0], a[1], marker='.', color='white', s=0.3, alpha=0.3)
        if text:
            ax[2].text(a[0][-1], a[1][-1], str(regions[p,7].round(2)), color='white', fontsize=4)
ax[0].set_yscale('log')
ax[0].set_ylabel('intensity')
ax[2].set_ylabel('minutes')
ax[1].set_ylabel('distribution #')
ax[2].set_xlabel('m/z')
for label in ax[2].get_xticklabels():
    label.set_rotation(-45)
ncols = 6
if text:
    ax[0].legend(title='Charge', loc='upper left', bbox_to_anchor=(0, 1.3 + (0.1 * cn / ncols)), ncol=ncols)
fig.tight_layout()
fig.subplots_adjust(hspace=0.05)
plt.show()
fig.clf()
plt.close()




#~~~~~~~~~~~~~~~~~~

#charge-handling, region format, unfinished
rgblow = lambda: (np.random.uniform(low=0.6, high=1), np.random.uniform(low=0.9, high=1), np.random.uniform(low=0.7, high=1))
rgbhigh = lambda: (np.random.uniform(low=0.9, high=1), np.random.uniform(low=0, high=0.1), np.random.uniform(low=0.8, high=0.9))
deepset = lambda: (np.random.uniform(low=0.2, high=1), np.random.uniform(low=0.4, high=1), np.random.uniform(low=0.1, high=0.8))

cfunc = lambda: (
        np.random.uniform(low=0.3, high=1), #R
        np.random.uniform(low=0.6, high=1), #G
        np.random.uniform(low=0.8, high=0.9) #B
        )
ndfunc = lambda: (
        np.random.uniform(low=0.9, high=1), #R
        np.random.uniform(low=0.1, high=0.6), #G
        np.random.uniform(low=0.1, high=0.6) #B
        )

with open(isotopefile, "rb") as pick:
    isotopeabundances, seqsbymass, subisotopicdifferences, newinclimit, steplimit = pickle.load(pick)

subisodiffs = np.array(list(subisotopicdifferences))[:,None]
subisotree = spatial.KDTree(subisodiffs)

st = 49.5
st = 49
et = 49.6
lmb = 364.5
umb = 367.7

st = 35
et = 35.5
lmb = 321
umb = 325.2

st = 33
et = 33.8
lmb = 352.9
umb = 354.3

st = 53.4
et = 54.2
lmb = 524.5
umb = 526.5


st = 49.8
st = 49
et = 49.9
et = 52
lmb = 522.2
umb = 528.3

#boundrec = [lmb, umb, st, et]
boundrec = [regions[:,7].min() - 1, regions[:,7].max() + 1, st, et]
regionsample = arg_coord_rectangle_overlap(boundrec, regions[:,:4]).tolist()
regiter = regions[regionsample][regions[regionsample,7].argsort()]

minpoints = 2
#minintensity = 0.4e6
chargetolerance = 0.1 #lesson learned: these differences DO get divided across charge states, if you normalize everything back to base mass without a charge, the errors become more consistent. They're smaller errors for higher charges etc. so going by percent here is FINE!
#mincharge = 0
#chargemethod = None
#chargemethod = 'step' # --> There's still hope for this here???
#stepsize = 10

subisomax = subisodiffs.max()
subisomax = subisomax + subisomax * chargetolerance

di = 0
paircharges = {} #connection: charge
connectioncharges = {} #groupid: charge
datapdiffs = {} #connection: %-diff of number of datapoints, things that can be calculated directly between two peaks and don't need to rely on the entire distribution for information
rtoffsets = {} #connection: overlap balance, a %
connections = defaultdict(dict) #groupid: connection: acdiff
subgroups = defaultdict(lambda: defaultdict(set)) #mass: charge: groups, a mass is always active as long as it's in masspool, this keeps track of all the groups that end in any given mass
#activedirection = defaultdict(int) #groupid: direction, 0 (increasing) or 1 (decreasing)

si = 0
subisogroups = defaultdict(lambda: defaultdict(set)) #subiso group: max charge for mass: [masses]
subisomasses = {} #mass: subisogroup

masspool = set()
t1 = time()
for reg in regiter:
    npoints = reg[4]
    if npoints >= minpoints:
        nm = reg[7]
        nmlt = reg[2]
        nmrt = reg[3]
        nmwidth = reg[1] - reg[0]
        nkey = int(reg[8])
        sst = reg[6]
        #olddirections = activedirection.copy()
        masspoolremovals = set()
        nlrange = nmrt - nmlt
        for omr in masspool:
            oreg = regions[omr]
            omlt = oreg[2]
            omrt = oreg[3]
            olrange = omrt - omlt
            overpass = False
            if nmlt < omrt and nmrt > omlt: #rt's overlap
                overlap = min(omrt, nmrt) - max(omlt, nmlt)
                if nmlt > omlt and nmrt < omrt: #new encompassed
                    combinedrange = nlrange + olrange
                    percentoverlap = (overlap * 2) / combinedrange
                    #using newinclimit here is appropriate because it's directly related to expected differences in adjacent isotopomer quantities, which is what the overlap is also ~somewhat assessing
                    if percentoverlap > newinclimit:
                        overpass = True
                elif omlt > nmlt and omrt < nmrt: #old encompassed
                    combinedrange = nlrange + olrange
                    percentoverlap = (overlap * 2) / combinedrange
                    if percentoverlap > newinclimit:
                        overpass = True
                else:
                    #no = overlap / nlrange
                    #oo = overlap / olrange
                    #if no > 0.5 and oo > 0.5: #max of new/old overlap > 0.5, a majority overlap for both -> 0.75 now because this shit was too lenient, this might be ok for a hard-coded value
                    fullrange = max(omrt, nmrt) - min(omlt, nmlt)
                    if (overlap * 2) / fullrange > 0.75: #this is super lenient I think
                        overpass = True
            if overpass:
                #^one of them should have to majority overlap, or fully encompass
                #eliminate anything in masspool > proton distance from nm
                okey = int(oreg[8])
                om = oreg[7]
                diff = nm - om
                omwidth = oreg[1] - oreg[0]
                widthbuffer = nmwidth + omwidth
                if diff < subisomax + widthbuffer:
                    #potential subisos
                    maxisocharge = round((subisomax / diff) - 0.5) #floor
                    if om in subisomasses:
                        si = subisomasses[okey]
                        subisogroups[si][maxisocharge].update((okey, nkey))
                        subisomasses[nkey] = si
                    else:
                        #make a new subiso group
                        subisogroups[si][maxisocharge].update((okey, nkey))
                        subisomasses[okey] = si
                        subisomasses[nkey] = si
                        si += 1
                elif diff <= proton + widthbuffer:
                    charge = round(proton / diff)
                    expdiff = proton / charge
                    acdiff = expdiff - diff
                    diffcut = expdiff * chargetolerance
                    if acdiff > -1 * (diffcut * chargetolerance + widthbuffer): #a lighter acceptance for distances above the charge dist - it does happen from time to time on lower-hanging straggler isotopomers
                        if acdiff <= diffcut + widthbuffer:
                            absacdiff = abs(acdiff)
                            ##sst = spectrum[nm]
                            ##sam = spectrum[om]
                            #sam = oreg[6]
                            #ncons = 0
                            #csubs = subgroups[okey][charge]
                            #intensitypercdiff = abs(sst - sam) / (sst + sam) / 2
                            #if csubs:
                            #    for adi in csubs:
                            #        ratiocheck = steplimit
                            #        if olddirections[adi] > 0:
                            #            #the dist has previously begun a decrease
                            #            if sst >= sam:
                            #                #and intensity is increasing
                            #                ratiocheck = newinclimit
                            #        if intensitypercdiff <= ratiocheck:
                            #            lpair = (okey, nkey)
                            #            connections[di].update(connections[adi])
                            #            connections[di][lpair] = acdiff
                            #            connectioncharges[di] = charge
                            #            subgroups[nkey][charge].add(di)
                            #            if sst < sam:
                            #                activedirection[di] += 1
                            #            ncons += 1
                            #            di += 1
                            #else:
                            #    #no previous subgroup
                            #    if intensitypercdiff <= steplimit:
                            #        lpair = (okey, nkey)
                            #        connections[di][lpair] = acdiff
                            #        connectioncharges[di] = charge
                            #        subgroups[nkey][charge].add(di)
                            #        if sst < sam:
                            #            #decreasing
                            #            activedirection[di] += 1
                            #        ncons += 1
                            #        di += 1
                            sam = oreg[6]
                            ncons = 0
                            csubs = subgroups[okey][charge]
                            if csubs:
                                for adi in csubs:
                                    lpair = (okey, nkey)
                                    connections[di].update(connections[adi])
                                    connections[di][lpair] = absacdiff
                                    connectioncharges[di] = charge
                                    subgroups[nkey][charge].add(di)
                                    ncons += 1
                                    di += 1
                            else:
                                #no previous subgroup
                                lpair = (okey, nkey)
                                connections[di][lpair] = absacdiff
                                connectioncharges[di] = charge
                                subgroups[nkey][charge].add(di)
                                ncons += 1
                                di += 1
                            if ncons > 0:
                                opoints = oreg[4]
                                dpercdiff = abs(npoints - opoints) / (npoints + opoints) / 2
                                datapdiffs[lpair] = dpercdiff
                                paircharges[lpair] = charge
                                combinedrange = nlrange + olrange
                                overlap = min((omrt, nmrt)) - max((omlt, nmlt)) #as long as minpoints > 1
                                percentoverlap = (overlap * 2) / combinedrange
                                rtoffsets[lpair] = percentoverlap
                                if csubs:
                                    connections[di][lpair] = absacdiff
                                    connectioncharges[di] = charge
                                    subgroups[nm][charge].add(di)
                                    #if sst < sam:
                                    #    #decreasing
                                    #    activedirection[di] += 1
                                    di += 1
                else:
                    #om is past proton distance, remove om from mass pool
                    masspoolremovals.add(okey)
        for mpr in masspoolremovals:
            masspool.remove(mpr)
        masspool.add(int(nkey))

print(time() - t1, 'mass processing')
t2 = time()

connectionkeys = {} #groupid: lineids
connectionsbykeys = {} #lineids: groupid
connectionintensities = {} #do I need this? I think only the main intensity matters as long as the ratios line up
connectionpoints = {} #number of datapoints per line
connectionratios = {} #groupid: [adjacent intensity ratios]
connectionmasses = {} #groupid: [sorted masses]
connectionlengths = {} #groupid: connection length
#connectionmainmass = {} #groupid: mass of main isotopomer (highest intensity)
connectionfirstmass = {} #groupid: first isotopomer
mainmasspositions = {} #groupid: position of main isotopomer
connectiontimerange = {} #groupid: [min, max] bounds
connectionmainintensity = {} #groupid: intensity of main isotopomer
connectionsize = defaultdict(list) #number of isotopomers: [groups of that length]
for k, v in connections.items():
    flatconnections = list(set(itertools.chain(*v)))
    sortedconnections = sorted(flatconnections, key=lambda x: regions[x,7])
    connectionkeys[k] = sortedconnections
    connectionsbykeys[tuple(sorted(flatconnections))] = k #using flatconnections so the keys can be found without knowing masses/intensities
    sortedregions = regions[sortedconnections]
    sortedmasses = sortedregions[:,7]
    connectionmasses[k] = sortedmasses
    sortedintensities = sortedregions[:,5]
    connectionintensities[k] = sortedintensities
    connectionpoints[k] = sortedregions[:,4]
    connectiontimes = sortedregions[:,2:4]
    connectiontimerange[k] = connectiontimes.min(), connectiontimes.max()
    intensityratios = sortedintensities[:-1] / sortedintensities[1:]
    #^this has the issue that ratios < 1 are not on the same scale as ones > 1
    #^you could use the > 1 ratio and make values negative for things originally < 1
    intensityratios[intensityratios < 1] = -1 / intensityratios[intensityratios < 1]
    connectionratios[k] = intensityratios.tolist()
    mainindex = sortedintensities.argmax()
    mainmasspositions[k] = mainindex
    #connectionmainmass[k] = sortedmasses[mainindex] #main isotopomer
    connectionfirstmass[k] = sortedmasses[0] #first isotopomer
    connectionmainintensity[k] = sortedintensities[mainindex]
    flen = len(flatconnections)
    connectionsize[flen].append(k)
    connectionlengths[k] = flen

print(time() - t2, 'charge dict assembly')
#check for charge-state connections using whole groups
#sum their isotopomers at each time point and get a single area
#   ^allow linkage to groups with less isotopomers if their area is also less
#keep track of high / low intensity ratios for each adjacent pair
#overlap-check each line by RT
#initial matching by highest isotopomer? Then filter for things that contain the same number of isotopomers as what you're matching
#   ^only the things that can be directly matched will carry the charge-state priority, other things will match naturally I suppose.
#this will create a league of different priority levels that are generated from each additional charge state that's found.
#run the pairs here to be pairs of connections, if a pair is already present, skip it, only iterate upwards in charge then I suppose

#things to check:
#how charge state differences operate across different charges
#are RT's rather perfectly symmetrical across charge states? Like will a lower intensity charge state have all its RT's encompassed well enough to rely on? Then there's also the case of noisy channels - but this might be an easy y/n answer to whether or not there's noise!

#charge-state to charge-state competition can be based on RT overlap, ie that the right lines link to the right ones I suppose - whichever LINK is correct can be determined via the pair metrics already developed
#record expected mass diff across each mass and note difference/intensity
#note intensity ratio variation, is the ratio always expected to decrease at an isotopomer of lower intensity on a charge of lower intensity? Or is this more stochastic?
#note if RT's encompass
#show information about the ratio of intensities of one charge state to another below the individual plots

#make another kind of plot that log-scale shows the whole spectrum (x: m/z, y: intensity)? and show charge-state links as pointing lines up top?
#^you could also shrink the x-axis space between them and display an 'expected dist' alongside the original, but in a different color -> expected intensities too I suppose
#a 2nd plot below these would show RT overlaps (x: m/z, y: rt)

#serious ponderance, does a deuterated isotopomer tend to go towards a lower charge state? I keep seeing this RT shift on a weak signal that doesn't match a higher charge state, it's weird
#revolving 1 concept, most of these ratios, iso/iso, charge-state/charge-state, rt overlap, basdiffs?,should all essentially ~equate to a value of 1. If I take a rolling count of how far each of these are, independently, from 1, then sum/use the values the way I rank the pairs - that might be a decent ranking system for these. But it still doesn't provide me a way of cutting off bad matches.
#allow the charge-state process to break the rules of increasing/decreasing isotopomer intensities I suppose -> if it's there, the evidence is more legit than the 'model'

#^^^^^^^^^in regards to the ~slight~weird rt mismatch across RT's, I don't expect ionization to be a pefectly linear process, whereby ionizing more or less of something always shows the same (especially considering background ionizers) proportion of 2+ to 3+ etc, it might be a process that spawns some 3+ in greater quantity after a certain intensity/number of ions are reached. And this could easily cross over from the early to mid-early peak shape. I think this is a fair explanation. You could also reason this could an ordered generative process, whereby starting small and increasing [as the peak would] that you could generate differing numbers of either charge.

#so once a charge state is determined here, lower priority pairing can't add a more intense ion into the fray I suppose?

#I need to determine how I'm going to metricise the group errors and cross-charge %'s, do I connect to nearest mass/charge state? do I connect to nearest intensity? do I use all possible crosses to make a larger number of metrics that are simplified across means and more means?

t3 = time()

groupid = 0
linechargelocations = defaultdict(lambda: defaultdict(set)) #line: charge: chargegroup id
chargegroups = defaultdict(set) #groupid: [connection keys]
#intersection_merge the set of key + values here to get finalized plotting groups?
for groupsize, keys in connectionsize.items():
    keymasses = [connectionfirstmass[i] for i in keys]
    keycharges = [connectioncharges[i] for i in keys]
    keymasses = np.array(keymasses)
    keys = np.array(keys)
    lowerbound = keymasses.min()
    upperbound = keymasses.max()
    for k, kc, km in zip(keys.tolist(), keycharges, keymasses.tolist()):
        ckeys = connectionkeys[k]
        cmasses = connectionmasses[k]
        cintensities = connectionintensities[k]
        cranks = cintensities.argsort()
        cminrt, cmaxrt = connectiontimerange[k]
        charge = connectioncharges[k]
        cmainpos = mainmasspositions[k]
        expdiff = proton / charge
        meandiff = np.diff(cmasses).mean()
        ctol = (expdiff - meandiff) * charge * chargetolerance
        ctol = (expdiff - meandiff) * charge
        basemass = km * kc - proton * kc
        trial = kc + 1
        searching = True
        while searching:
            searching = False
            scoutmass = (basemass + proton * trial) / trial
            if scoutmass >= lowerbound:
                if scoutmass <= upperbound:
                    matches = np.logical_and(keymasses >= scoutmass - ctol, keymasses <= scoutmass + ctol)
                    #^check that the connection doesn't already exist, or hasn't already been checked?
                    for mk in keys[matches].tolist():
                        if connectioncharges[mk] == trial:
                            #if mainmasspositions[mk] == cmainpos:
                            mminrt, mmaxrt = connectiontimerange[mk]
                            overpass = False
                            if cminrt < mmaxrt and cmaxrt > mminrt: #rt's overlap
                                #i should only need to check that the one with a lesser intensity is encompassed, but maybe if they're close I suppose
                                if cminrt > mminrt and cmaxrt < mmaxrt: #primary encompassed
                                    overpass = True
                                elif mminrt > cminrt and mmaxrt < cmaxrt: #secondary encompassed
                                    overpass = True
                                else:
                                    #crange = cmaxrt - cminrt
                                    #mrange = mmaxrt - mminrt
                                    overlap = min(mmaxrt, cmaxrt) - max(mminrt, cminrt)
                                    #combinedrange = crange + mrange
                                    #percentoverlap = (overlap * 2) / combinedrange
                                    #co = overlap / crange
                                    #mo = overlap / mrange
                                    #if co > 0.5 and mo > 0.5: #75% overlap for both full bounds
                                    fullrange = max(mmaxrt, cmaxrt) - min(mminrt, cminrt)
                                    if (overlap * 2) / fullrange > 0.75:
                                        overpass = True
                            if overpass:
                                #if np.all(connectionintensities[mk].argsort() == cranks): #check for conserved order structure
                                chargegroups[groupid].add(mk)
                                searching = True
                                #make error and cross-charge metrics
                                #use the full interaction matrix, divide by the number of bridges for a mean
                                #the inverses for errors are the same thing b/c of the calculation, but the cross-charge %'s are different when both taken from the opposite direction AND when inverted
                                #both can be done via combinations without replacement, but you just do both calculations for the cross charge %
            trial += 1
        if groupid in chargegroups:
            chargegroups[groupid].add(k)
            groupid += 1
print(time() - t3, 'charge group linkage')
t4 = time()

#I think it would be a good move to intersection_merge the output values of chargegroups (you won't need to make linechargelocations above either) once it's done and let competitors compete within a single possible charge union - if they exist. And under the possibility that multiple multiply charged ions fly next to each other under the same neighboring charges [within their tolerances], you would be able to differentiate the two this way.

#on the other isotopomer plots, the labels can be charge(# other charge states)
chargesets = []
for k, v in chargegroups.items():
    chargesets.append(v)
#this merge guarantees connections will still be of the same size, with no super/subset merging
chargesets = intersection_merge(chargesets)

#combinatorics will be based off of this, and this allows things that didn't connect downward to branch in that direction
chargelayers = defaultdict(lambda: defaultdict(set)) #groupid: charge: [connections]
for n, cs in enumerate(chargesets):
    for c in cs:
        chargelayers[n][connectioncharges[c]].add(c)

chargecongroups = defaultdict(dict) #connection: [all chargegroups its involved in]
for cl in chargelayers.values():
    for cons in itertools.product(*cl.values()):
        #close order matching here, giving the intensities some leeway in terms of ranking without allowing them to move around too much - is the idea
        cintensities = np.array([connectionintensities[i] for i in cons])
        consorts = cintensities.argsort(axis=1)
        conranges = np.ptp(consorts, axis=0)
        if conranges.sum() < conranges.size - 1: #I like the -2 for this, it prevents 2's from switching, 3s need at least one stable position, and 4s can only switch 2 adjacents or one double, etc.. this value absorbs multi-rank jumps in the same pool as the individuals so it has its benefit i suppose
            #concheck = [0, 0]
            #cnorm = special.comb(len(cons), 2) * connectionlengths[cons[0]] #number of bridges, favors both long distribution matches as well as ions with a greater number of charge states
            #for lp, rp in itertools.combinations(cons, 2):
            #    li = connectionintensities[lp]
            #    ri = connectionintensities[rp]
            #    iratio = li / ri
            #    irmean = iratio.mean()
            #    ccmeandiff = np.abs(irmean - iratio).mean() / irmean #dividing by irmean makes it the same for whether you do li / ri or ri / li instead
            #    concheck[0] += ccmeandiff
            #    #
            #    lm = connectionmasses[lp]
            #    rm = connectionmasses[rp]
            #    lcharge = connectioncharges[lp]
            #    rcharge = connectioncharges[rp]
            #    lbase = lm * lcharge - proton * lcharge
            #    rbase = rm * rcharge - proton * rcharge
            #    basediffs = lbase - rbase
            #    massdiff = np.abs(basediffs.mean() - basediffs).mean() #already symmetrical
            #    concheck[1] += massdiff
            #concheck[0] /= cnorm
            #concheck[1] /= cnorm
            intensitysums = cintensities.sum(axis=0)
            cmasses = np.array([connectionmasses[i] for i in cons])
            charges = np.array([connectioncharges[i] for i in cons])[:,None]
            basemasses = cmasses * charges - proton * charges
            massmeandiff = np.abs(basemasses.mean(axis=0) - basemasses).mean()
            intensitypercs = cintensities / cintensities.sum(axis=0)
            intensitymeandiff = np.abs(intensitypercs.mean(axis=1)[:,None] - intensitypercs).mean()
            for con in cons:
                #chargecongroups[con][cons] = concheck
                #concheck[cons].append([ccmeandiff, massdiff])
                chargecongroups[con][cons] = [intensitymeandiff, massmeandiff]
#these two metrics are imbalanced, (953, 29) is better than (950, 29) but it wouldn't win here, the mass error is what makes it obvious but the mass metric doesn't impact the overall thing at all

#both this process and it's look-alike below for the regular pairs are not so much a normalizing, it's more like a balancing.
prioritycharges = []
#secondprioritycharges = []
secondprioritygroups = defaultdict(list)
for con, congroups in chargecongroups.items():
    if len(congroups) > 1:
        n1, n2 = np.array(list(congroups.values())).sum(axis=0)
        for congroup, (s1, s2) in congroups.items():
            lines = set(itertools.chain(*[connectionkeys[i] for i in congroup]))
            newconscore = sum((s1/n1, s2/n2))
            prioritycharges.append([congroup, newconscore])
    else:
        congroup, conscore = list(congroups.items())[0]
        lines = set(itertools.chain(*[connectionkeys[i] for i in congroup]))
        #outrank = [congroup, lines, sum(conscore)]
        nconscore = conscore.copy()
        nconscore.insert(0, congroup)
        for con in congroup:
            if nconscore not in secondprioritygroups[con]:
                secondprioritygroups[con].append(nconscore)
        #if outrank not in secondprioritycharges:
            #secondprioritycharges.append(outrank)

secondpriorities = []
thirdpriorities = [] #order won't really matter for these, everything is either already decided or not decided until it gets to this specific grouping
for line, linegroups in secondprioritygroups.items():
    if len(linegroups) > 1:
        n1, n2 = np.array(linegroups)[:,1:].sum(axis=0)
        for lg in linegroups:
            con, s1, s2 = lg
            lines = set(itertools.chain(*[connectionkeys[i] for i in con]))
            newconscore = sum((s1/n1, s2/n2))
            secondpriorities.append([con, newconscore])
    else:
        lg = linegroups[0]
        con, s1, s2 = lg
        lines = set(itertools.chain(*[connectionkeys[i] for i in con]))
        thirdpriorities.append([con, sum((s1, s2))])

rankedcharges = sorted(prioritycharges, key=lambda x: x[1]) #competition among matched chargegroups where matching can be done easily
secondprioritycharges = sorted(secondpriorities, key=lambda x: x[1]) #competition among lines where competition is needed
thirdprioritycharges = sorted(thirdpriorities, key=lambda x: x[1]) #order doesn't actually matter
rankedcharges.extend(secondprioritycharges)
rankedcharges.extend(thirdprioritycharges)
#^the connection groups take priority of the direction of what lines match to each other. There will still be a secondary pairing process for lines going on here, but each new connection group gained will have to have a superset of the existing lines for the connection group in question to grow.
#^but this has a problem with linking bad actors, there's no way to allow a subset cutoff
#I'm going to move forward accepting this for now, it could be modified via tweaking later
#^currently, it would rely on the robustness of the initial chargegroup selection to filter out poor distributions


#instead of iterating through the list at each entry, you could keep track of sets of lines made along the way, and only iterate until a superset is found - priority is given to those at the top of the list this way
#^yes but you need to do a second final sweep to make sure no existing subsets are still in the list, it's got to be free flowing, then eliminating for subsets
#^only create a blocking list for the second part, might not just be subsets but weird combinations of groups that aren't together under the prevailing logic

##what I have for the existing pair system:
#iterate pairs & score:
#    something passes the criteria for ranking:
#        orders come here
#iterate orders:
#    iterate orders again:
#        combine supersets
#
##what I propose above:
#make flylist
#iterate pairs & scores:
#    something passes the criteria for ranking:
#        list of sets kept on the fly:
#            - first check if the new passed pair is a sub/superset of anything in the existing flylist
#            - if not, its becomes its own entry
#            - if so -> merge, superset connections take priorities for charge group connection descriptions
#make blocked list
#iterate final superset list:
#    - block things already taken, in order, output should be the final list
#^this actually changes the outcome, and so would doing a post-priority blocking. You'd have to do an awkward post-priority subset removal. I also did try this previously for the pair matching process and it didn't work.


#~~~~
#multiple types of merging:
#superset merging to allow for larger distribution lengths to work out
#intersection merging to allow shorter distributions at different, faded, charges to still connect
#you can add more charge states
#things added at already possessed charge states must overlap with the existing line structure, and newer/previously unexposed lines can be added
#basically, add whatever you like as long as the existing line infrastructure at each charge is preserved
#^with some checks, the longest distribution shouldn't be on something less than the most intense I suppose
#free expansion as long as existing lines overlap
#^I'm going to skip checking that the rank of lengths equates to the rank of intensities, although this would probably be an appropriate check.


blocked = set()
preservedchargeranks = []
for group, score in rankedcharges:
    if group not in blocked:
        preservedchargeranks.append([group, score])
        blocked.add(group)
print(time() - t4, 'charge set and priority ranking')
t5 = time()

chargeid = 0
blocked = set()
chargedistlines = defaultdict(lambda: defaultdict(set)) #chargegroupid: charge: [lines]
chargedistgroups = defaultdict(dict) #chargegroupid: charge: distributionid
chargegroupsbyline = {} #line: chargegroupid, doubles as blocking list
for pn, (cons, score) in enumerate(preservedchargeranks):
    #join anything that has a sorted tuple key to connectionsbykeys with the existing line infrastructure at each charge state
    ckeys = set(itertools.chain(*(connectionkeys[i] for i in cons)))
    if not any(i in blocked for i in ckeys):
        #^if none are blocked, none have been used -> make new group I suppose?
        grouplines = {}
        groupdists = {}
        for con in cons:
            charge = connectioncharges[con]
            distkeys = connectionkeys[con]
            grouplines[charge] = distkeys
            groupdists[charge] = con
        for ncons, nscore in preservedchargeranks[pn+1:]:
            if cons != ncons:
            #if not any(i in cons for i in ncons):
                nkeys = set(itertools.chain(*(connectionkeys[i] for i in ncons)))
                if not any(i in blocked for i in nkeys):
                    if ckeys.intersection(nkeys):
                        joinlines = {}
                        joindists = {}
                        join = True
                        for ncon in ncons:
                            charge = connectioncharges[ncon]
                            nckeys = connectionkeys[ncon]
                            if charge in grouplines:
                                joinedlines = tuple(sorted(set(grouplines[charge] + nckeys)))
                                if joinedlines in connectionsbykeys:
                                    joinlines[charge] = list(joinedlines)
                                    joindists[charge] = connectionsbykeys[joinedlines]
                                else:
                                    join = False
                                    break
                            else:
                                joinlines[charge] = nckeys
                                joindists[charge] = ncon
                        if join:
                            for ch in joinlines:
                                grouplines[ch] = joinlines[ch]
                                groupdists[ch] = joindists[ch]
        chargedistlines[chargeid] = grouplines
        chargedistgroups[chargeid] = groupdists
        for ch, lines in grouplines.items():
            for line in lines:
                chargegroupsbyline[line] = chargeid
            blocked.update(lines)
        chargeid += 1
print(time() - t5, 'charge group assembly')


#sum lines of a distribution across time and take the area across charge states -> largest area should have the longest distribution -> if not, correct the longer less intense distributions to match the most intense one. This can be used as the future 'leader' to later decide whether distributions in the pairing process are allowed to expand.
#after pair-match charge states are assembled, you can look for the outer-fray lone lines that match major lines of a charge distribution

##for plotting distributions from an incomplete process
#shrinksets = defaultdict(set)
#for n, cg in enumerate(chargesets):
#    for g in cg:
#        shrinksets[n].update(connectionkeys[g])
#
#blocked = set()
#finalsets = {}
#for (k1, v1), (k2, v2) in itertools.combinations(shrinksets.items(), 2):
#    if v1.issuperset(v2):
#        blocked.add(k2)
#        if k2 in finalsets:
#            del finalsets[k2]
#        if k1 not in blocked:
#            finalsets[k1] = v1
#    elif v2.issuperset(v1):
#        blocked.add(k1)
#        if k1 in finalsets:
#            del finalsets[k1]
#        if k2 not in blocked:
#            finalsets[k2] = v2
#    elif v1 == v2:
#        if k1 not in blocked and k2 not in blocked:
#            finalsets[k1] = v1
#            if k2 in finalsets:
#                del finalsets[k2]
#        else:
#            blocked.add(k1)
#            blocked.add(k2)
#            if k2 in finalsets:
#                del finalsets[k2]
#            if k1 in finalsets:
#                del finalsets[k1]
#    else:
#        if k1 not in blocked:
#            finalsets[k1] = v1
#        if k2 not in blocked:
#            finalsets[k2] = v2
#
#plotsets = [chargesets[i] for i in finalsets]
#
#taxing = True
#for cg in plotsets:
#    cg = list(cg)
#    cg = sorted(cg, key=lambda x: connectionmainintensity[x])
#    cgcharges = [connectioncharges[i] for i in cg]
#    chargeorder = sorted(set(cgcharges))
#    chlen = len(chargeorder)
#    #if len(chargeorder) > 2:
#    #if len(cg) == chlen:
#    #if len(chargeorder) > 2:
#    if True:
#        cf = pd.DataFrame()
#        chargefigures = {c:n for n, c in enumerate(chargeorder)}
#        cgcons = [connections[i] for i in cg]
#        fig, ax = plt.subplots(ncols=chlen, nrows=5, figsize=(6,8), sharex='col', sharey='row')
#        nfig, nax = nfig, nax = plt.subplots(ncols=chlen, nrows=2, figsize=(7,4), sharey='row')
#        if taxing:
#            tfig, tax = plt.subplots(ncols=2, nrows=2, figsize=(7,4))
#        fig.subplots_adjust(hspace=0.05, wspace=0.05)
#        mainintensities = [connectionmainintensity[i] for i in cg]
#        maxmain = max(mainintensities)
#        tbarwidth = 1 / len(cg)
#        tspace = 0.5
#        cols = dp.get_colors(len(cg))
#        cmin = np.inf
#        cmax = 0
#        cratiolist = []
#        ccbounds = [np.inf, 0]
#        for n, g in enumerate(cg):
#            con = connections[g]
#            cratios = connectionratios[g]
#            cratiolist.append(cratios)
#            abcratios = [abs(i) for i in cratios]
#            cmasses = connectionmasses[g]
#            cintensities = connectionintensities[g]
#            if cintensities.min() < cmin:
#                cmin = cintensities.min()
#            if cintensities.max() > cmax:
#                cmax = cintensities.max()
#            concharge = connectioncharges[g]
#            expdiff = proton / concharge
#            basemasses = cmasses * concharge - proton * concharge
#            acdiffs = expdiff - np.diff(cmasses)
#            basediffs = acdiffs * concharge
#            conhax = chargefigures[concharge]
#            ckeys = list(set(itertools.chain(*con)))
#            cwidth = 0.5 *  len(ckeys)
#            nst = regions[ckeys,2].min()
#            net = regions[ckeys,3].max()
#            nlmb = regions[ckeys,0].min()
#            numb = regions[ckeys,1].max()
#            boundrec = [nlmb, numb, nst, net]
#            plotkeys = arg_coord_rectangle_overlap(boundrec, regions[:,:4]).tolist()
#            for p in plotkeys:
#                if p not in ckeys:
#                    a = trackedgroups[p]
#                    creg = regions[p]
#                    ax[1][conhax].plot(a[0], a[1], '.', color='white', alpha=0.2)
#                    ax[0][conhax].plot([creg[7], creg[7]], [0, creg[5]], '-', color='white', alpha=0.5, linewidth=cwidth)
#            ax[3][conhax].hlines(0, cmasses.min(), cmasses.max(), color='black', linewidth=0.3)
#            ax[4][conhax].hlines(0, cmasses.min(), cmasses.max(), color='black', linewidth=0.3)
#            for cline in ckeys:
#                creg = regions[cline]
#                ax[0][conhax].plot([creg[7], creg[7]], [0, creg[5]], '-', color=cols[n], alpha=1, linewidth=cwidth)
#                a = trackedgroups[cline]
#                ax[1][conhax].plot(a[0], a[1], '.', color=cols[n], alpha=0.2)
#            mainintensity = connectionmainintensity[g] / maxmain
#            basenorms = basediffs / proton
#            abasediffs = np.abs(basenorms / proton - basediffs)
#            if taxing:
#                tax[0][0].bar(concharge, mainintensity, tbarwidth*tspace, color=cols[n], alpha=0.5)
#                tax[0][1].bar([i+tbarwidth*concharge*tspace for i in range(len(cratios))], cratios, width=tbarwidth*tspace, color=cols[n], alpha=0.5)
#                tax[1][0].bar([i+tbarwidth*concharge*tspace for i in range(len(cratios))], basediffs, width=tbarwidth*tspace, color=cols[n], alpha=0.5)
#            diffgen = 0.05
#            for nc, cn in enumerate(cg):
#                cnmasses = connectionmasses[cn]
#                cnintensities = connectionintensities[cn]
#                cnratio = cintensities / cnintensities
#                cnratiobar = np.abs(cnratio.mean() - cnratio).mean()
#                if cnratiobar > ccbounds[1]:
#                    ccbounds[1] = cnratiobar
#                if cnratiobar < ccbounds[0] and cnratiobar > 0:
#                    ccbounds[0] = cnratiobar
#                cx = chargefigures[connectioncharges[cn]]
#                nax[0][conhax].bar(cx, cnratiobar, width=0.8, color=cols[nc], alpha=0.5)
#                cf.loc[g, cn] = cnratiobar
#                if nc >= n:
#                    ax[2][conhax].bar(cmasses+diffgen*nc, cnratio, width=diffgen/2, color=cols[nc], alpha=0.8)
#            bn = 0
#            bw = 0.02
#            for nc, cn in enumerate(cg):
#                if cn != cg:
#                    maincharge = connectioncharges[cn]
#                    mainmasses = connectionmasses[cn]
#                    mainbasemasses = mainmasses * maincharge - proton * maincharge
#                    maindiffs = mainbasemasses - basemasses
#                    mainppm = (maindiffs / mainbasemasses) * 1000000
#                    diffbar = np.abs(mainppm.mean() - mainppm).mean()
#                    cx = chargefigures[connectioncharges[cn]]
#                    nax[1][conhax].bar(cx, diffbar, width=0.8, color=cols[nc], alpha=0.5)
#                    ax[3][conhax].bar(cmasses+bn, maindiffs, width=bw, color=cols[nc], alpha=0.8)
#                    ax[4][conhax].bar(cmasses+bn, mainppm, width=bw, color=cols[nc], alpha=0.8)
#                    bn += bw + bw / 2
#            ax[0][conhax].set_title(''.join((str(concharge), '(', str(g), ')')))
#        if taxing:
#            cratiolist = np.array(cratiolist)
#            crmean = np.mean(cratiolist, axis=0)
#            crplot = np.abs(crmean - cratiolist).mean(axis=0)
#            tax[1][1].bar([i+tbarwidth*cratiolist.shape[1]*tspace for i in range(len(cratios))], crplot, width=tbarwidth, color='midnightblue', alpha=0.5)
#            tax[0][0].set_yscale('log')
#            tax[1][1].set_yscale('log')
#            tax[0][0].set_title('intensity ratios')
#            tax[0][1].set_title('isotopomer ratios')
#            tax[1][0].set_title('basediffs')
#            tax[1][1].set_title('isotopomer ratio meandiffs')
#        nax[0][0].set_yscale('log')
#        nax[0][0].set_ylim(ccbounds[0]/2, ccbounds[1])
#        #nax[0][1].set_yscale('log')
#        nax[0][0].set_ylabel('cross-charge meandiffs')
#        nax[1][0].set_ylabel('ppm meandiffs')
#        ax[0][0].set_yscale('log')
#        ax[0][0].set_ylim(cmin/2, cmax)
#        ax[0][0].set_ylabel('peak area')
#        ax[1][0].set_ylabel('retention time')
#        ax[2][0].set_ylabel('cross-charge %')
#        ax[3][0].set_ylabel('absolute error')
#        ax[4][0].set_ylabel('ppm error')
#        for ch, hax in chargefigures.items():
#            ax[-1][hax].tick_params(axis='x', labelrotation=-45)
#            if hax == 0:
#                #invisible right splines
#                ax[0][hax].spines.right.set_visible(False)
#                ax[1][hax].spines.right.set_visible(False)
#            elif hax == chlen-1:
#                #invisible left splines
#                ax[0][hax].spines.left.set_visible(False)
#                ax[1][hax].spines.left.set_visible(False)
#                for tick in ax[0][hax].yaxis.get_majorticklines():
#                    tick.set_visible(False)
#                for tick in ax[0][hax].yaxis.get_minorticklines():
#                    tick.set_visible(False)
#                for tick in ax[1][hax].yaxis.get_major_ticks():
#                    tick.tick1line.set_visible(False)
#                    tick.tick2line.set_visible(False)
#            else:
#                #left and right invisible
#                ax[0][hax].spines.right.set_visible(False)
#                ax[1][hax].spines.right.set_visible(False)
#                ax[0][hax].spines.left.set_visible(False)
#                ax[1][hax].spines.left.set_visible(False)
#                for tick in ax[0][hax].yaxis.get_majorticklines():
#                    tick.set_visible(False)
#                for tick in ax[0][hax].yaxis.get_minorticklines():
#                    tick.set_visible(False)
#                for tick in ax[1][hax].yaxis.get_major_ticks():
#                    tick.tick1line.set_visible(False)
#                    tick.tick2line.set_visible(False)
#        tfig.tight_layout()
#        nfig.tight_layout()
#        plt.show()
#        fig.clf()
#        tfig.clf()
#        nfig.clf()
#        plt.close()
#        gc.collect()

#charge-states of lesser intensity would need to be blocked, later on, from making a pair connection, that a charge-state of a higher intensity has not made

#now I need to visualize ratios, and charge distances!
#x: basemass
#y: intensity ratios
#color: charge

#horizontal bar between isotopomer intensity lines
#horizontal bar at twiny intensity ratio
#above bar, percent diff to expdiff
#below bar, absolute diff to expdiff

#basemass oriented bar chart, ordered by mainmass intensity, highest -> lowest for the sub-order? Though perhaps higher charge-states might show different behavior than lower ones when the highest intensity charge-state is in the middle somewhere.

#so, no, the intensity ratios aren't consistently ordered in any precise manner - they somewhat are, but they don't follow a strict ascent nor descent based on relative charge-state intensity

#can you perhaps explain the increase in isotopomer ratio from intensity fade via the ratio of isotopomers across charge states?
#^gotta visualize the cross charge-state isotopomer ratios as another potential avenue for a metric

#when there's a split match on the same distribution's connections, if the match with a higher intensity shows a smaller mass error, that's your winner

#I don't think the fade comes from analyzer dynamic range, or anything of the sort. I think that's an ionization phenomenon, as it always happens to the lowest intensity isotopomer of the lowest intensity charge state. Even when other charge states are at extremely low intensities for the analyzer, ie 1e5, the fade is happening only to the same exact isotopomer as you'd expect.

#I'll consider cross-charge-state error from the perspective of the most intense distribution. From there I can check whether mass error remains constant across each charge state and whether some distributions are actually a good match

#overall, the goal should be to pick which pairs to elevate, those can get plucked out of distributions that show decent matches. I think fade, even weird fade, can be allowed if the main isotopomer and it's best pair show good matching.
#all other categories of information should be used as a blocking mechanism to prevent the link from occuring.
#but I don't have a mechanism that allows a subset connection to overtake a superset should there be good reason.
#^that comes with the blocking. The lower priority (below charge-state priority) can allow extra pairs to be made so long as it's present in the highest intensity charge

#After the isotope process is done, when you're matching them to theoretical quantities, that fade may be an important feature there, it might be worth-while to determine which isotopes need that quantity corrected, and how much correction is being given. I'd wager you could systematically fix those errors while improving all fittings by either seeing if the faded quantity is 'stolen' by other charge states, or by assuming the lost/gained amount from the adjacent isotopomer ratios.
#adjacent isotopomer ratios can be used to determine which isotopomers of which charge state can be trusted as an example quantity for applying a correction by the cross-charge state ratio. This mechanism can also mark an isotopomer of a charge state as faded. If you can't determine that any of the adjacent ratios are good for a specific isotopomer, then you shouldn't accept that specific isotopomer for any of the distributions.
#^If this happens in the middle of a distribution, you'll have to split the ends I suppose. If it's in the middle for just one, cut off the rest of only that distribution and let the pieces compete at a lower priority.
#It might be worth looking for missing line-pieces on the faded isotopomers.
#in there case where there's 2 charge states and you have no idea of knowing if an isotopomer of either is a legit value, then you can just accept them as long as the masses and everything else lines up well. The order of intensities matters in this case.

#competetive processes can't be relied upon here the same way they do for basic iso distributions because there's not enough guaranteed competition over each line.
#consistency is key -> Find a conserved structure of ranks, this part must be perfect. Even if other isotopomers likely do match, they'll be left to match via the lower priority system.
#For competing charge-states, the average distance of the cross-charge %'s can be used in place of acdiff, but the acdiff goes for that distribution as opposed to any individual pairs. If a charge state with a super/sub-set of isotopomers gets outcompeted, it's not allowed to pass or connect. One distribution wins, and the contents of which are blocked.
#the second metric will be the same concept, except distance from average basemass error
#If a different charge state of a faded isotopomer can match the cross-charge %'s, that isotopomer should be accepted?
#Basically, if 2 charge states (potentially out of 3+) can agree on an isotopomer, a shittier one in a different charge state can be accepted.
#^So how would this work in the case where the most intense charge state is bullshit, but two lesser ones aren't?
#in the case of a noisy charge state, use a charge state with less competition to determine winning lines.

#summary:
#conserved structure of ranks - hard cutoff that decides what isotopomers gets to compete as a charge state. isotopomers that break ranks aren't included in any charge state - unless there's more than 1 charge state where it's consistent, then the third might be dragged along I suppose?
#^Allow for supersets to win this, then move on to the next step below for those supersets of which the ranks all play out.
#individual distributions compete via mean distance to mean charge state %'s, and mean distance to mean masss error
#mark known/visible fades by key in a fadedict of some sort

#After a brief, non-extensive search, I didn't see any reason to look for non-adjacent charge states, but this isn't the end of wondering if they're there.

#you might need to change the basemass-based connection, if two values are really close to being the basemass and if this switches back and forth across charges, you'd miss it

#seemed to have missed a line because of a slight switch in order of intensities, perhaps I should close order switching? Things can move up and down by like, 1
#^so subtract the argsorts and any absolute values > 1 would negate the process, however this might introduce some actual fuckery, I might want to make sure the actual intensity values are close in both distributions, maybe adjacent ratios here could play a part

#(1049, 494)
#^{2: [246586, 247679], 3: [246676, 248098]}
#(1205, 1126)
#^{2: [248787, 249475], 3: [249835, 251301]}
#(1111, 803), potential climb is too high? signal looks solid though
#^{4: [248824, 249347, 249797], 3: [249198, 249722, 250322]}

#bad match, coincidence:
#(2605, 1682) - I think the cross-charge % gives this away the strongest, its pattern also ~matches the ppm error pattern, the mass error isn't terrible

#needs close order switching:
#(2689, 2596)

#big mass errors:
#(2629, 87) -> has a better candidate that lost it seems, but that's not the only problem?
#(2615, 21) -> has a better candidate that lost? NO! It was just barely outside the massrange that I was slicing

#cross-charge % should have some % limit for a distance from the mean to be seem as a legit charge candidate
#(2327, 10), (2658, 2352) are good ~limits? it's a good match

#I need a way for longer distributions that have the intensity threshhold passed to swoop in and pick up things that have faded that didn't pass that threshhold -> the connection pairs are 100% always made so I just need a way to find it
#^so essentially, if the main distribution can find a key that matches the tuple(sorted(set)) of the keys, the I guess it's fine to link it to a lesser ranked charge state
#the close order switching isn't going to be straightforward, if the mainmasses switch, the distributions won't match -> I need the matching to be based on all masses rather than just the main
#I also want to exclude joining two 'different' distributions across charge states if their joining isotopomer differences are the most deviating from either of their means. It's a COINCIDENCE!
#^alternative strategy, via information processing later on: keep the distributions that don't really perfectly match glued together and search a distribution later on iteratively at it's multiple initiating sights aka adjacent isotopomer increases
#^both not dealing with the complicated bullshit upfront, and you can also make the assumption later on that they are two distributions if you decide you don't like the one. This helps deal with fade. Because otherwise... I don't really have a great way of dealing with fade. I would need massive overhead on the distribution linking process

#semi-side note, I'll need a cross-file validation scheme for MS1 predictions from one file that have MS2 scans in others. Or just a scheme to match MS2 identified peptides to their untargeted counterparts in other files.

#how well does the %'s from the sum of all isotopomers across charge states (adjacency ratio) match the adjacency ratios from the averaged charge state isotopomers? And which one matches theoretical distributions better on a large scale? Might be able to say something about the ionization events here.


#in this there are two clear bad matches, and one need for a type of close order switching
#forr the closeorderswitch match, you can make the initial match to either mainmass or monomass, that's pretty easy - and just do a logical_and to link the inds
#^and the idea of the closeorderswitch should be able to accomodate a specific number of switches based on the length I suppose
#rt mismatches are the cause of the bad matching ones, the whole distributions needs to be penalized for one bad rt match, all rt's need to be aligned under the fullmatch criteria i suppose
#although one of the bad rt matches should have been beaten out by a pair rather than the triplet that won...
#acceptable cross-charge %'s should always be either larger or smaller, never crossing, MAYBE this could only cross off a bad match when the mass error is the largest? or like 10x more than others?

#majority of isotopomers have a >= majority rt overlap of their cross charge state isotopomers in the event where something can be wrong, this can look at +/-1 charge state beyond to see if a match is there. If it is, accept the bad one in the middle. "Bad match" in this case would be something where the RTs/intensities are out of wack but the mass is good.

#TEST LATER:
#I see an interesting phenomenon, where when the intensities of the main masses are really close to 1:1, the next adjacent isotopomers of the charge state with the less intense main mass tend to be higher than those of the charge state with the main mass. And I'm wondering if this can be an observable ionization phenomenon that somehow depends on mass. And whether I can deduce that a distribution has subisotopomers to the left or right of their majors perhaps?
#(2222, 104) and (2625, 1203) are examples

#look into:
#(25555, 118), I'm not sure that 4-group should have beat the 2

#kicking something out of a charge state later on -> if a distribution wants to swoop in and claim some already claimed territory, and it offers both better alignment + a longer distribution to which the charge state can't compete AND there isn't a longer charge state chain to stand up for it, -> remove it I suppose!

#if the largest mass error and the largest rt deviation go together -> dump it I suppose, but how to determine if the largest is acceptable or not? It should also be a ~lower intensity and on the end of the distribution I suppose.
#rt overlap should basically be put into the metric process as a mean distance from the mean thing

#pretty reasonable observation:
#<1, more intense ions should have large cross-charge ratios, for ratios >1, less intense ions should have larger cross-charge ratioss
#^USUALLY, but not always, perhaps this would be a good normalizing metric to determine how a particular charge state or distribution might have been suppressed?
#For example, if the expected ratios switch, then normalize the more intense ion upwards by assigning it a greater intensity than measured? rather than normalizing the smaller one upwards.
#this adds some more justification to the close order switching process too, it seems like it's just another area where the data is extremely fragile.
#I'm also seeing larger adjacency ratios on less intense charge states - I think, investigate this more

#(3234, 1632) don't belong together because 3234 has a better distribution to be a part of that doesn't match

#the kickout process can be done after pair matching and will essentially check the close order matching of things all over again to determine if things still deserve to be linked, like a less intense distribution with more legs -> doesn't match the higher
#^or a lower intensity distribution that doesn't really match the expanded higher intensity one -> the orders are out of wack compared to how it should have faded

#a bad pattern is like (3236, 16) where the rank orders differ, and the cross charge percentages overlap -> this should be an auto-no, but how?

#new metrics:
#cross-charge order, for both intensities and dpoints
# - no need for negatives/flipped ratios, use raw ones - same effect
# - each isotopomer is normalized to the sum of that isotopomer at every charge state
# - then the you can take mean difference to mean at each isotopomer and therefore the metric will judge whether the isotopomer is a good addition
#currently, for ppm: the errors are averaged across a distribution but they should be averaged across isotopomers -> then the mean difference to this is the value to them mean across the other isotopomers -> normalize by number of isotopomers
# - average across basemass by isotopomer, then take distance to that average

#probems are stemming from the superset merging in the final ranking, bad things don't get poofed away
#looks like i'll need to allow </> 0 acdiffs or whatever, they happen

#I want to visualize how the charge distance scales back up to the ~proton length for each distribution across charge states
#I'll also put in the intensity/sum intensity across isotopomer plots
#I also want to visualize RT centering for each individual distribution, might be worth taking a ratio somewhere here
#basemass distance from average too

#check for charge distance:
#(3998, 3803) -> meh, pairmatching will fix it
#(3979, 3546, 1894) -> i want the new visualizations above
#(3966, 3185) -> their other connections don't seem to match that well because the only matches of the correct size have offset isoopomers. I'll need to look into more

#implement that each overlapping RT must have 75% fullrange, if they don't match here but do so later on via pair matching, so be it, but there's too many cases here of RT's being off for individual straggler ions, and this would be a decent place to cut off.
#for now, things that aren't distributions, yet that match perfectly, can match. For example (2870, 31) has a great match but totally isn't a distribution, they each clearly have a good distribution of their own extending in opposite directions -> when pair matching finds those pairs, you can excommunicate these two. I suppose a third charge state would override something like this? I'm not sure.
#^the ensuing matches would need to NOT have the strangest charge state distance
#can you regard fade the same at higher masses as lower? ie at lower charge states and at high? I'm not sure they act the same way

#current plan:
#re-impement activedirection, pairmatch FIRST, THEN allow for charge linkages afterwards, it seems backwards but it also seems like it would work better. The pairmatching scheme is HOT on accuracy, allowing this to take the reins beforehand would allow for much better matches
#^PLUS I can allow subset charge pairing for dists that don't form a connection coalition to pair individually pair across more than one charge state to get succesfully get the full superset link
#the resulting distribution matches can be based PURELY on the isotopomers that match, while only holding in consideration that more intense distributions should be ~longer
#^so you really only need to match the 2 highest isoptopomers, which will obviously be right next to each other, and then you can expand outwards I suppose. As any match that can't at least agree on the top 2 isn't going to be a legit match I think. The +/-1 charge state search to aid in seeing if one of them is just fucked up would help here too.
#when matching distributions, neither the mainmass or firstmass ideas are going to be able to fly alone: leader masses -> anything within 90% of the highest intensity mass? that way you can match distributions if the main mass switches across charge state, distribution structures can be filtered afterwards.

taxing = False
nfaxing = False
for dists in chargedistgroups.values():
    chargeorder = sorted(dists)
    chlen = len(dists)
    cf = pd.DataFrame()
    chargefigures = {c:n for n, c in enumerate(chargeorder)}
    fig, ax = plt.subplots(ncols=chlen, nrows=8, figsize=(6,8), sharex='col', sharey='row')
    if nfaxing:
        nfig, nax = plt.subplots(ncols=chlen, nrows=2, figsize=(7,4), sharey='row')
    if taxing:
        tfig, tax = plt.subplots(ncols=2, nrows=2, figsize=(7,4))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    cg = list(dists.values())
    mainintensities = [connectionmainintensity[i] for i in cg]
    maxmain = max(mainintensities)
    arrayintensities = np.array([connectionintensities[g] for g in cg])
    try:
        intensitysums = arrayintensities.sum(axis=0)
    except ValueError:
        amax = max(i.size for i in arrayintensities)
        arrayintensities = np.array([i for i in arrayintensities if i.size == amax])
        intensitysums = arrayintensities.sum(axis=0)
    chargedistbounds = [np.inf, 0]
    arraymasses = np.array([connectionmasses[g]*connectioncharges[g]-proton*connectioncharges[g] for g in cg])
    try:
        arraymeans = arraymasses.mean(axis=0)
    except ValueError:
        arraymasses = np.array([i for i in arraymasses if i.size == amax])
        arraymeans = arraymasses.mean(axis=0)
    tbarwidth = 1 / len(cg)
    tspace = 0.5
    cols = dp.get_colors(len(cg))
    cmin = np.inf
    cmax = 0
    cratiolist = []
    ccbounds = [np.inf, 0]
    for n, (charge, g) in enumerate(dists.items()):
        con = connections[g]
        cratios = connectionratios[g]
        cratiolist.append(cratios)
        abcratios = [abs(i) for i in cratios]
        cmasses = connectionmasses[g]
        cintensities = connectionintensities[g]
        cpoints = connectionpoints[g]
        cpointratios = cpoints[:-1] / cpoints[1:]
        cpointratios[cpointratios < 1] = -1 / cpointratios[cpointratios < 1]
        if cintensities.min() < cmin:
            cmin = cintensities.min()
        if cintensities.max() > cmax:
            cmax = cintensities.max()
        concharge = connectioncharges[g]
        expdiff = proton / concharge
        basemasses = cmasses * concharge - proton * concharge
        meanbasediff = arraymeans[:basemasses.size] - basemasses[:arraymeans.size]
        meanbaseppms = (meanbasediff / basemasses[:arraymeans.size]) * 1000000
        acdiffs = expdiff - np.diff(cmasses)
        basediffs = acdiffs * concharge
        conhax = chargefigures[concharge]
        ckeys = list(set(itertools.chain(*con)))
        cwidth = 0.5 *  len(ckeys)
        chargelengthextra = proton / charge * 2
        nst = regions[ckeys,2].min() - 0.5
        net = regions[ckeys,3].max() + 0.5
        nlmb = regions[ckeys,0].min() - chargelengthextra
        numb = regions[ckeys,1].max() + chargelengthextra
        nboundrec = [nlmb, numb, nst, net]
        nplotkeys = arg_coord_rectangle_overlap(nboundrec, regions[:,:4]).tolist()
        for p in nplotkeys:
            if p not in ckeys:
                a = trackedgroups[p]
                creg = regions[p]
                ax[0][conhax].plot(a[0], a[1], '.', color='white', alpha=0.2)
                ax[1][conhax].plot([creg[7], creg[7]], [0, creg[5]], '-', color='white', alpha=0.5, linewidth=cwidth)
        ax[5][conhax].hlines(0, cmasses.min(), cmasses.max(), color='black', linewidth=0.3)
        ax[7][conhax].hlines(0, cmasses.min(), cmasses.max(), color='black', linewidth=0.3)
        ax[2][conhax].hlines(proton, cmasses.min(), cmasses.max(), color='black', linewidth=0.3)
        for cline in ckeys:
            creg = regions[cline]
            ax[1][conhax].plot([creg[7], creg[7]], [0, creg[5]], '-', color=cols[n], alpha=1, linewidth=cwidth)
            a = trackedgroups[cline]
            ax[0][conhax].plot(a[0], a[1], '.', color=cols[n], alpha=0.2)
            #ax[2][conhax].plot([creg[7], creg[7]], [0, creg[4]], '-', color=cols[n], alpha=1, linewidth=cwidth)
        mainintensity = connectionmainintensity[g] / maxmain
        basenorms = basediffs / proton
        abasediffs = np.abs(basenorms / proton - basediffs)
        if taxing:
            tax[0][0].bar(concharge, mainintensity, tbarwidth*tspace, color=cols[n], alpha=0.5)
            tax[0][1].bar([i+tbarwidth*concharge*tspace for i in range(len(cratios))], cratios, width=tbarwidth*tspace, color=cols[n], alpha=0.5)
            tax[1][0].bar([i+tbarwidth*concharge*tspace for i in range(len(cratios))], basediffs, width=tbarwidth*tspace, color=cols[n], alpha=0.5)
        diffgen = 0.05
        bn = 0
        bw = 0.02
        adjacentx = cmasses[:-1] + np.diff(cmasses) / 2
        ax[5][conhax].bar(adjacentx, cratios, width=diffgen, color=cols[n], alpha=1)
        #ax[6][conhax].bar(adjacentx, cpointratios, width=diffgen, color=cols[n], alpha=1)
        chargedists = np.diff(cmasses) * charge
        ax[2][conhax].bar(adjacentx, chargedists, width=diffgen, color=cols[n], alpha=1)
        intensitypercs = cintensities[:intensitysums.size] / intensitysums[:cintensities.size]
        for ip in chargedists:
            if ip < chargedistbounds[0]:
                chargedistbounds[0] = ip
            if ip > chargedistbounds[1]:
                chargedistbounds[1] = ip
        ax[4][conhax].bar(cmasses, intensitypercs, width=diffgen, color=cols[n], alpha=1)
        ax[6][conhax].bar(cmasses, meanbaseppms, width=diffgen, color=cols[n], alpha=1)
        for nc, cn in enumerate(cg):
            cnmasses = connectionmasses[cn]
            cnintensities = connectionintensities[cn]
            cnpoints = connectionpoints[cn]
            #this size index mechanism won't work right for when isotopomers to the left aren't present
            cnratio = cnintensities[:cintensities.size] / cintensities[:cnintensities.size]
            pointratio = cnpoints[:cpoints.size] / cpoints[:cnpoints.size]
            cnratiobar = np.abs(cnratio.mean() - cnratio).mean()
            if cnratiobar > ccbounds[1]:
                ccbounds[1] = cnratiobar
            if cnratiobar < ccbounds[0] and cnratiobar > 0:
                ccbounds[0] = cnratiobar
            cx = chargefigures[connectioncharges[cn]]
            if nfaxing:
                nax[0][conhax].bar(cx, cnratiobar, width=0.8, color=cols[nc], alpha=0.5)
            cf.loc[g, cn] = cnratiobar
            bx = cmasses[:cnintensities.size]+(diffgen*nc)
            ax[3][conhax].bar(bx, cnratio, width=diffgen, color=cols[nc], alpha=1)
            #ax[4][conhax].bar(bx, pointratio, width=diffgen, color=cols[nc], alpha=1)
        for nc, cn in enumerate(cg):
            if cn != cg:
                maincharge = connectioncharges[cn]
                mainmasses = connectionmasses[cn]
                mainbasemasses = mainmasses * maincharge - proton * maincharge
                #this size index mechanism won't work right for when isotopomers to the left aren't present
                maindiffs = mainbasemasses[:basemasses.size] - basemasses[:mainbasemasses.size]
                mainppm = (maindiffs / mainbasemasses[:basemasses.size]) * 1000000
                diffbar = np.abs(mainppm.mean() - mainppm).mean()
                cx = chargefigures[connectioncharges[cn]]
                if nfaxing:
                    nax[1][conhax].bar(cx, diffbar, width=0.8, color=cols[nc], alpha=0.5)
                #ax[5][conhax].bar(cmasses+bn, maindiffs, width=bw, color=cols[nc], alpha=1)
                ax[7][conhax].bar(cmasses[:mainbasemasses.size]+bn, mainppm, width=bw, color=cols[nc], alpha=1)
                bn += bw + bw / 2
        ax[0][conhax].set_title(''.join((str(concharge), '(', str(g), ')')), fontsize=12)
    if taxing:
        cratiolist = np.array(cratiolist)
        crmean = np.mean(cratiolist, axis=0)
        crplot = np.abs(crmean - cratiolist).mean(axis=0)
        tax[1][1].bar([i+tbarwidth*cratiolist.shape[1]*tspace for i in range(len(cratios))], crplot, width=tbarwidth, color='midnightblue', alpha=0.5)
        tax[0][0].set_yscale('log')
        tax[1][1].set_yscale('log')
        tax[0][0].set_title('intensity ratios')
        tax[0][1].set_title('isotopomer ratios')
        tax[1][0].set_title('basediffs')
        tax[1][1].set_title('isotopomer ratio meandiffs')
    if nfaxing:
        nax[0][0].set_yscale('log')
        nax[0][0].set_ylim(ccbounds[0]/2, ccbounds[1])
        #nax[0][1].set_yscale('log')
        nax[0][0].set_ylabel('cross-charge meandiffs')
        nax[1][0].set_ylabel('ppm meandiffs')
    ax[1][0].set_yscale('log')
    #ax[2][0].set_yscale('log')
    ax[2][0].set_ylim(chargedistbounds[0]*0.95, chargedistbounds[1]*1.05)
    ax[3][0].set_yscale('log')
    ax[5][0].set_yscale('symlog')
    ax[4][0].set_yscale('log')
    #ax[6][0].set_yscale('symlog')
    ax[1][0].set_ylim(cmin/2, cmax)
    ax[1][0].set_ylabel('peak area')
    ax[0][0].set_ylabel('retention time', fontsize=6)
    ax[3][0].set_ylabel('cross-charge', fontsize=7)
    ax[5][0].set_ylabel('adjacency')
    ax[7][0].set_ylabel('ppm error')
    ax[2][0].set_ylabel('charge distances', fontsize=6)
    ax[4][0].set_ylabel('intensity sum %', fontsize=6)
    ax[6][0].set_ylabel('ppm to mean', fontsize=7)
    for ch, hax in chargefigures.items():
        ax[-1][hax].tick_params(axis='x', labelrotation=-45)
        if hax == 0:
            #invisible right splines
            ax[0][hax].spines.right.set_visible(False)
            ax[1][hax].spines.right.set_visible(False)
        elif hax == chlen-1:
            #invisible left splines
            ax[0][hax].spines.left.set_visible(False)
            ax[1][hax].spines.left.set_visible(False)
            for tick in ax[0][hax].yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
            for tick in ax[1][hax].yaxis.get_majorticklines():
                tick.set_visible(False)
            for tick in ax[1][hax].yaxis.get_minorticklines():
                tick.set_visible(False)
        else:
            #left and right invisible
            ax[0][hax].spines.left.set_visible(False)
            ax[0][hax].spines.right.set_visible(False)
            ax[1][hax].spines.left.set_visible(False)
            ax[1][hax].spines.right.set_visible(False)
            for tick in ax[0][hax].yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
            for tick in ax[1][hax].yaxis.get_majorticklines():
                tick.set_visible(False)
            for tick in ax[1][hax].yaxis.get_minorticklines():
                tick.set_visible(False)
    if taxing:
        tfig.tight_layout()
    if nfaxing:
        nfig.tight_layout()
    plt.show()
    fig.clf()
    if taxing:
        tfig.clf()
    if nfaxing:
        nfig.clf()
    plt.close()
    gc.collect()

#charge-processing based on paircharges
#pairmasses = {k: regions[list(k),7] for k in paircharges}
#pairranges = {k: [regions[list(k),2].min(), regions[list(k),3].max()] for k in paircharges}
#pairintensities = {k: regions[list(k),5].tolist() for k in paircharges}

nt = time()

pairmasses = {}
pairranges = {}
pairintensities = {}
pairratios = {}
for pair in paircharges:
    k = list(pair)
    treg = regions[k]
    pairmasses[pair] = treg[:,7]
    pairranges[pair] = [treg[:,2].min(), treg[:,3].max()]
    pairintensities[pair] = [treg[0,5],  treg[1,5]]
    pairratios[pair] = treg[0,5] / treg[1,5]

pairsbycharge = defaultdict(set)
for k, v in paircharges.items():
    pairsbycharge[v].add(k)

massranges = regions[:,:2]
maxmass = massranges.max() + 1
minmass = massranges.min() - 1


#take in each pair -> look for +1 and -1 level charges, if nothings there, quit. If you find something, move again in that direction.
#you could possibly eliminate things on the fly that get found -> include the minus direction then
chargeoverhead = defaultdict(set) #pair: charge (of npair): [npairs]
for pair, charge in paircharges.items():
    masses = pairmasses[pair]
    massdiff = masses[1] - masses[0]
    timerange = pairranges[pair]
    intensities = pairintensities[pair]
    basemasses = masses * charge - proton * charge
    #trial = charge - 1
    #while trial > 0:
    trial = charge + 1
    searching = True
    while searching:
        #test here, by switching rt overlap checks with mass overlap checks to determine which overlap happens more often, the least often one should give a faster algorithm -> analysis
        searching = False
        scoutmasses = (basemasses + proton * trial) / trial
        if scoutmasses.min() >= minmass:
            if scoutmasses.max() <= maxmass:
                #next step: immediately use this setup to figure out more about isotopomer differences across different charge states
                chargediff = massdiff * charge / trial
                ctol = chargediff * chargetolerance
                #check if trial is in pairsbycharge, if not -> break
                for npair in pairsbycharge[trial]:
                    nmasses = pairmasses[npair]
                    masscheck = np.all(np.abs(scoutmasses - nmasses) <= ctol)
                    if masscheck:
                        ntrange = pairranges[npair]
                        if timerange[0] <= ntrange[1] and ntrange[0] <= timerange[1]: #rt overlap
                            chargeoverhead[pair].add(npair)
                            searching = True
                            #check intensity ratios, and individual rt windows
                            #check that individual rt windows are porportional across charge
        trial += 1
print(time() - nt, 'charge processing')

grouplist = []
for k, v in chargeoverhead.items():
    v.add(k)
    grouplist.append(v)

outgroups = intersection_merge(grouplist)

visibleextension = 5
for og in outgroups:
    og = list(og)
    chargesort = np.argsort([paircharges[o] for o in og]).tolist()
    rcharges = [paircharges[i] for i in og]
    linecharges = {}
    for n, (l, r) in enumerate(og):
        linecharges[l] = rcharges[n]
        linecharges[r] = rcharges[n]
    ogroups = intersection_merge(og)
    olen = len(ogroups) + 1
    gcharges = [linecharges[list(i)[0]] for i in ogroups]
    fig, ax = plt.subplots(nrows=olen, ncols=2, figsize=(6,2*olen), sharex='col', sharey='row')
    cols = dp.get_colors(olen)
    osorted = sorted(zip(gcharges, ogroups), reverse=False)
    for rn, (rcharge, ro) in enumerate(osorted):
        rcol = cols[rn]
        rog = list(ro)
        reg = regions[list(rog)]
        massextension = proton / rcharge * visibleextension / 2
        rminmass = reg[:,0].min() - massextension
        rmaxmass = reg[:,1].max() + massextension
        bwidth = (rmaxmass - rminmass) / 100
        bnorm = reg[:,5].max()
        ax[rn][1].barh(reg[:,7], reg[:,5]/bnorm, color=rcol, height=bwidth, alpha=0.5)
        #ax[rn][1].barh(reg[:,7], reg[:,5], color=rcol, height=bwidth, alpha=0.5)
        for r in rog:
            a = trackedgroups[r]
            ax[rn][0].plot(a[1], a[0], '.', markersize=0.8, color=rcol, alpha=0.5)
        ax[rn][0].set_title(rcharge)
        #plottting extras
        rminrt = reg[:,2].min()
        rmaxrt = reg[:,3].max()
        brec = [rminmass, rmaxmass, rminrt, rmaxrt]
        bsample = arg_coord_rectangle_overlap(brec, regions[:,:4]).tolist()
        for bs in bsample:
            if bs not in rog:
                a = trackedgroups[bs]
                ar = regions[bs]
                ax[rn][0].plot(a[1], a[0], '.', markersize=0.4, color='white', alpha=0.5)
                ax[rn][1].barh(ar[7], ar[5]/bnorm, color='white', height=bwidth, alpha=0.5)
                #ax[rn][1].barh(ar[7], ar[5], color='white', height=bwidth, alpha=0.5)
        #ax[rn][1].set_xscale('log')
    rminmass = rminmass * rcharge - proton * rcharge
    rmaxmass = rmaxmass * rcharge + proton * rcharge
    rcharge += 1
    rminmass = (rminmass + proton * rcharge) / rcharge
    rmaxmass = (rmaxmass + proton * rcharge) / rcharge
    brec = [rminmass, rmaxmass, rminrt, rmaxrt]
    bsample = arg_coord_rectangle_overlap(brec, regions[:,:4]).tolist()
    rn += 1
    for bs in bsample:
        if bs not in rog:
            a = trackedgroups[bs]
            ar = regions[bs]
            ax[rn][0].plot(a[1], a[0], '.', markersize=0.4, color='white', alpha=0.5)
            ax[rn][1].barh(ar[7], ar[5]/bnorm, color='white', height=bwidth, alpha=0.5)
            #ax[rn][1].barh(ar[7], ar[5], color='white', height=bwidth, alpha=0.5)
    #ax[0][1].set_xscale('log')
    ax[rn][0].set_title(rcharge)
    plt.tight_layout()
    plt.show()
    fig.clf()
    plt.close()
    gc.collect()

#make the visible windows extend x number of charge-distance windows
#add in other lines within the windows
#make the charge windows extend to +/- 1 charge
#visualize mass distance and intensity ratio comparisons?

#allow charge states of lesser intensity to have a fading ratio on lower intensity isotopomers, it seems to happen that those isotopomers of lower intensity can get lost to the dynamic range
#with those intensities dropping on lower isotopomers, it makes sense to assume a 'dropping down' model of ionization. Things first gather as maner charges as possible, and the charges are lost as energy in the ionization (or ms I suppose) process 

#I need to group full distributions together and find the entire thing, rather than just pairs here


groupsbypair = defaultdict(set)
scoresbypair = defaultdict(dict) #mass: pair: [scores]
secondpriorities = defaultdict(dict) #essentially there's too many zeros caused by single-pair matches, they can be let in but they don't deserve top priority 
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
            outscore = scoreval - scoreval * offset
            ddiff = datapdiffs[sgk]
            outdiff = ddiff - ddiff * offset
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
    minvals = vpercs.sum(axis=1)
    #minvals = minvals + minvals * offsets
    rankedpairs.extend(list(zip(pairs, minvals)))

secondrankedpairs = [] #[pair, minval]
for m, pg in secondpriorities.items():
    pairs, vals = zip(*pg.items())
    offsets = [rtoffsets[i] for i in pairs]
    vals = np.array(vals)
    vpercs = vals / vals.sum(axis=0).tolist()
    minvals = np.abs(vpercs[:,1] - vpercs[:,0])
    minvals = minvals - minvals * offsets
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

print(time() - t2, 'riff raff')
t3 = time()

dr =  0
franks = {} #distid: rank
groupcharges = {} #groupid: charge
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
        mc = max(gcs)
        solodists[mc][gv] = list(ng)
        groupcharges[gv] = mc
        blocked.update(ng) 
        franks[gv] = dr
        dr += 1
print(time() - t3, 'ranking')

foundvals = []
for charge, sgd in solodists.items():
    foundvals.extend(list(itertools.chain(*sgd.values())))
specvals = regiter[:,8].astype(int)
nodists = np.setdiff1d(specvals, foundvals)

text = True
ngroups = sum(len(i) for i in solodists.values())
cols = dp.get_colors(ngroups)
heightcounter = 0
cn = 0
fig, ax = plt.subplots(nrows=3, figsize=(6,8), sharex=True)
for fc, fgs in solodists.items():
    for fk, fm in fgs.items():
        if np.logical_and(regions[fm,7] <= umb, regions[fm,7] >= lmb).any():
            ftimes = regions[fm,2:4]
            if np.logical_and(ftimes[:,0] <= et, ftimes[:,1] >= st).any():
                fg = regions[fm,7].tolist()
                col = cols[cn]
                low, high = rgblow(), rgbhigh()
                cn += 1
                for p in fm:
                    a = np.array(trackedgroups[p])
                    ax[2].scatter(a[0], a[1], marker='.', color=col, s=0.3, alpha=0.3)
                    ax[2].plot(a[0], a[1], '-', color=col, linewidth=0.2, alpha=0.8)
                    if text:
                        ax[2].text(a[0][-1], a[1][-1], str(regions[p,7].round(2)), color='white', fontsize=4)
                fints = regions[fm,6].tolist()
                specs = {}
                specs.update(zip(fm, fints))
                #distrank = franks[fk]
                ax[0].bar(fg, fints, color=col, alpha=0.5, width=0.01, label=str(fc))
                if text:
                    for fx, fy, sfm in zip(fg, fints, fm):
                        ax[0].text(fx, fy + fy * 0.03, str(sfm), color='white', fontsize=4)
                print(fg)
                print(fc, '-', np.diff(sorted(fg)))
                print('~')
                ax[1].hlines(heightcounter, min(fg), max(fg), color=col, linewidth=0.6)
                vrpoints = regions[fm,4].astype(int).tolist()
                for vert, pl in zip(fg, vrpoints):
                    ax[1].vlines(vert, heightcounter - 0.1, heightcounter + 0.1, color=col, linewidth=0.6)
                    if text:
                        ax[1].text(vert, heightcounter + 0.1, pl, fontsize=4, ha='center', color='white')
                vi = np.sort(fg)
                if vi.size > 2:
                    vstack = np.stack((vi[:-1], vi[1:]), axis=1)
                    editspots = np.diff(vstack) < subisomax
                    if editspots.any():
                        ewheres = np.where(editspots)[0].tolist()
                        for ew in ewheres:
                            subpair = vstack[ew].tolist()
                            subints = [specs[i] for i in subpair]
                            winint = subints.index(max(subints))
                            winner = subpair[winint]
                            if ew > 0:
                                #edit 1 before ew
                                vstack[ew-1,1] = winner
                            if ew < len(vstack) - 1:
                                #edit 1 after ew
                                vstack[ew+1,0] = winner
                        vstack = np.delete(vstack, ewheres, axis=0)
                else:
                    vstack = vi.reshape(1, -1)
                vdiffs = np.diff(vstack)
                vflat = sorted(vstack.flatten().tolist())
                labelspots = np.mean(vstack, axis=1).tolist()
                for ls, lp in zip(labelspots, vstack.tolist()):
                    labeldiff = np.diff(lp)[0].round(4)
                    chargedist = (proton/fc - labeldiff).round(4)
                    lstring = ' ~ '.join((str(fc), str(labeldiff), str(chargedist)))
                    if text:
                        ax[1].text(ls, heightcounter - 0.2, lstring, fontsize=4, ha='center', color='white')
                heightcounter += 1


if nodists.size > 0:
    ndmasses = regions[nodists,7]
    ndtimes = regions[nodists,2:4]
    masspass = np.logical_and(ndmasses <= umb, ndmasses >= lmb)
    timepass = np.logical_and(ndtimes[:,0] <= et, ndtimes[:,1] >= st).any()
    plotternos = nodists[np.where(np.logical_and(masspass, timepass))[0]]
    massnos = regions[plotternos,7]
    nints = regions[plotternos,6]
    ax[0].bar(massnos, nints, alpha=0.5, color='white', width=0.01, label='N/A')
    if text:
        for fx, fy, px in zip(massnos, nints, plotternos):
            if fy > 0:
                ax[0].text(fx, fy + fy * 0.03, str(px), color='white', fontsize=4)
    for p in plotternos:
        a = np.array(trackedgroups[p])
        fx = regions[p,7]
        #
        ax[2].scatter(a[0], a[1], marker='.', color='white', s=0.3, alpha=0.3)
        ax[2].plot(a[0], a[1], '-', color='white', linewidth=0.2, alpha=0.8)
        if text:
            ax[2].text(a[0][-1], a[1][-1], str(regions[p,7].round(2)), color='white', fontsize=4)
ax[0].set_yscale('log')
ax[0].set_ylabel('intensity')
ax[2].set_ylabel('minutes')
ax[1].set_ylabel('distribution rank')
ax[2].set_xlabel('m/z')
for label in ax[2].get_xticklabels():
    #label.set_ha("right")
    label.set_rotation(-45)
ncols = 6
if text:
    ax[0].legend(title='Charge', loc='upper left', bbox_to_anchor=(0, 1.3 + (0.1 * cn / ncols)), ncol=ncols)
#ax[2].set_ylim(st, et)
#ax[2].set_xlim(lmb, umb)
fig.tight_layout()
fig.subplots_adjust(hspace=0.05)
plt.show()
fig.clf()
plt.close()





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~







#new paradigm, early pair charge-handling, region format

with open(isotopefile, "rb") as pick:
    subisotopicdifferences, newinclimit, steplimit = pickle.load(pick)

dlim = abs(newinclimit - newinclimit**2) / (newinclimit + newinclimit**2) / 2

minpoints = 3
chargetolerance = 0.1 #lesson learned: these differences DO get divided across charge states, if you normalize everything back to base mass without a charge, the errors become more consistent. They're smaller errors for higher charges etc. so going by percent here is FINE!

subisodiffs = np.array(list(subisotopicdifferences))[:,None]
subisomax = subisodiffs.max()
subisomax = subisomax + subisomax * chargetolerance


st = 49.5
et = 49.6
lmb = 364.5
umb = 367.7

st = 48
et = 52
lmb = 363
umb = 369

st = 35
et = 35.5
lmb = 321
umb = 325.2

st = 33
et = 33.8
lmb = 352.9
umb = 354.3

st = 53.4
et = 54.2
lmb = 524.5
umb = 526.5

#current example for OTF re-ranking
st = 48
et = 51
lmb = 655
umb = 665

st = 49.8
st = 49
et = 49.9
et = 52
lmb = 522.2
umb = 528.3

#next check doing every ~10 minute chunk
aend = 185
incs = 10

increment = 0
wt = time()

while increment < aend:
    st = increment
    et = increment + incs
    zoomplotting = False
    #zoomplotting = True

    t1 = time()

    #fucking 120-122 takes longer than 120-130 and uses more memory???? or a 0.01 change in dlim did this???
    #boundrec = [lmb, umb, st, et]
    #tregions = regions[timewidths < 5]
    boundrec = [regions[:,7].min() - 1, regions[:,7].max() + 1, st, et]
    regionsample = arg_coord_rectangle_overlap(boundrec, regions[:,:4]).tolist()
    rselection = regions[regionsample]
    regiter = rselection[rselection[:,7].argsort()]
    regiter = regiter[regiter[:,4] >= minpoints]

    subgroups = defaultdict(lambda: defaultdict(set)) #mass: charge: groups, a mass is always active as long as it's in masspool, this keeps track of all the groups that end in any given mass
    connections = defaultdict(set) #groupid: connection: acdiff
    activedirection = defaultdict(int) #groupid: direction, 0 (increasing) or 1 (decreasing)

    di = 0
    paircharges = {} #connection: charge
    scoresbypair = {} #pair: [[absacdiff, datapercdiff, rtoffset],]
    pairscoresbymass = defaultdict(dict) #mass: pair: [[absacdiff, datapercdiff, rtoffset],]

    si = 0
    subisomasses = {} #mass: subisogroup
    subisogroups = defaultdict(lambda: defaultdict(set)) #subiso group: max charge for mass: [masses]

    masspool = set()
    for reg in regiter.tolist():
        npoints = reg[4]
        nm = reg[7]
        nmlt = reg[2]
        nmrt = reg[3]
        nmwidth = reg[1] - reg[0]
        nkey = int(reg[8])
        sst = reg[5]
        masspoolremovals = set()
        nlrange = nmrt - nmlt
        for omr in masspool:
            oreg = regions[omr]
            om = oreg[7]
            omwidth = oreg[1] - oreg[0]
            widthbuffer = nmwidth + omwidth
            diff = nm - om
            okey = int(oreg[8])
            if diff <= proton + widthbuffer:
                opoints = oreg[4]
                dpercdiff = abs(npoints - opoints) / (npoints + opoints) / 2
                #if dpercdiff < dlim: #number of datapoints aligns with newinclimit
                omlt = oreg[2]
                omrt = oreg[3]
                olrange = omrt - omlt
                overpass = False
                if nmlt < omrt and nmrt > omlt: #rt's overlap
                    #if you want to allow non-overlaps to match, and there are good reasons to, you need percentoverlap to become negative so that its later addition is valued worse
                    overlap = min(omrt, nmrt) - max(omlt, nmlt)
                    if nmlt > omlt and nmrt < omrt: #new encompassed
                        combinedrange = nlrange + olrange
                        percentoverlap = (overlap * 2) / combinedrange
                        #using newinclimit here is appropriate because it's directly related to expected differences in adjacent isotopomer quantities, which is what the overlap is also ~somewhat assessing
                        if percentoverlap > newinclimit:
                            overpass = True
                    elif omlt > nmlt and omrt < nmrt: #old encompassed
                        combinedrange = nlrange + olrange
                        percentoverlap = (overlap * 2) / combinedrange
                        if percentoverlap > newinclimit:
                            overpass = True
                    else:
                        fullrange = max(omrt, nmrt) - min(omlt, nmlt)
                        percentoverlap = overlap / fullrange
                        if percentoverlap * 2 > 0.75: #this is super lenient I think
                            overpass = True
                #else:
                    #negative percentoverlap values for non-overlaps within some 1.75 range or something goes here
                if overpass:
                    if diff < subisomax + widthbuffer:
                        #potential subisos
                        maxisocharge = round((subisomax / diff) - 0.5) #floor
                        if om in subisomasses:
                            si = subisomasses[okey]
                            subisogroups[si][maxisocharge].update((okey, nkey))
                            subisomasses[nkey] = si
                        else:
                            #make a new subiso group
                            subisogroups[si][maxisocharge].update((okey, nkey))
                            subisomasses[okey] = si
                            subisomasses[nkey] = si
                            si += 1
                    #elif diff <= proton + widthbuffer:
                    else:
                        charge = round(proton / diff)
                        expdiff = proton / charge
                        acdiff = expdiff - diff
                        diffcut = expdiff * chargetolerance
                        if acdiff > -1 * (diffcut * chargetolerance + widthbuffer): #a lighter acceptance for distances above the charge dist - it does happen from time to time on lower-hanging straggler isotopomers
                            if acdiff <= diffcut + widthbuffer: 
                                absacdiff = abs(acdiff) * charge #normalizing -> distance to proton
                                sam = oreg[5]
                                ncons = 0
                                csubs = subgroups[okey][charge]
                                intensitypercdiff = abs(sst - sam) / (sst + sam) / 2
                                if csubs:
                                    for adi in csubs:
                                        ratiocheck = steplimit
                                        if activedirection[adi] > 0:
                                            #the dist has previously begun a decrease
                                            if sst >= sam:
                                                #and intensity is increasing
                                                ratiocheck = newinclimit
                                        if intensitypercdiff <= ratiocheck:
                                            lpair = (okey, nkey)
                                            connections[di].update(connections[adi])
                                            connections[di].add(lpair)
                                            subgroups[nkey][charge].add(di)
                                            if sst < sam:
                                                activedirection[di] += 1
                                            ncons += 1
                                            di += 1
                                else:
                                    #no previous subgroup 
                                    if intensitypercdiff <= steplimit:
                                        lpair = (okey, nkey) 
                                        connections[di].add(lpair)
                                        subgroups[nkey][charge].add(di)
                                        if sst < sam:
                                            #decreasing
                                            activedirection[di] += 1
                                        ncons += 1
                                        di += 1
                                if ncons > 0:
                                    paircharges[lpair] = charge
                                    scorelist = [absacdiff, dpercdiff, percentoverlap]
                                    for p in lpair:
                                        pairscoresbymass[p][lpair] = scorelist
                                    scoresbypair[lpair] = scorelist
                                    if csubs:
                                        connections[di].add(lpair)
                                        subgroups[nkey][charge].add(di)
                                        if sst < sam:
                                            #decreasing
                                            activedirection[di] += 1
                                        di += 1
            else:
                #om is past proton distance, remove om from mass pool
                masspoolremovals.add(okey)
        for mpr in masspoolremovals:
            masspool.remove(mpr)
        masspool.add(int(nkey))

    print(time() - t1, 'mass processing')
    t2 = time()

    groupsbypair = defaultdict(set)
    distmeanscores = defaultdict(dict) #mass: pair: [scores]
    for gid, pairs in connections.items():
        slen = len(pairs)
        if slen > 1:
            scores = np.array([scoresbypair[i] for i in pairs])
            distmean = scores[:,0].mean()
            for pair, score in zip(pairs, scores.tolist()): 
                groupsbypair[pair].add(gid)
                dist, ddiff, rtoffset = score
                #meandiff = abs(distmean - dist) / (slen + 1) 
                meandiff = abs(distmean - dist) / slen
                distdiff = meandiff - meandiff * rtoffset
                datadiff = ddiff - ddiff * rtoffset
                scorelist = [distdiff, datadiff]
                for m in pair:
                    distmeanscores[m][pair] = scorelist
        else:
            pair = list(pairs)[0]
            groupsbypair[pair].add(gid)

    rankedpairs = [] #[pair, minval]
    for m, pg in distmeanscores.items():
        pairs, vals = zip(*pg.items())
        vals = np.array(vals)
        vpercs = vals / vals.sum(axis=0).tolist()
        vpercs[np.isnan(vpercs)] = 0
        sumvals = vpercs.sum(axis=1)
        rankedpairs.extend(list(zip(pairs, sumvals)))

    secondpriorities = [] #[pair, score]
    thirdpriorities = [] #[pair, score]
    for line, pairdict in pairscoresbymass.items():
        plen = len(pairdict)
        if plen > 1:
            pairs, scores = zip(*pairdict.items())
            scorearray = np.array(scores)
            #offsetscores = scorearray[:,:2] - scorearray[:,:2] * scorearray[:,2,None]
            #normoffsets = offsetscores / offsetscores.sum(axis=0)
            #for pair, score, rtoffset in zip(pairs, normoffsets.tolist(), scorearray[:,2].tolist()):
            #    secondpriorities.append([pair, sum(score)])
            #    rtoffsets[pair] = rtoffset
            #try this? idk - actually it worked really nicely
            normscores = scorearray / scorearray.sum(axis=0)
            offsetnorms = normscores[:,:2] - normscores[:,:2] * scorearray[:,2,None]
            for pair, score, rtoffset in zip(pairs, offsetnorms.tolist(), scorearray[:,2].tolist()):
                secondpriorities.append([pair, sum(score)])
        else:
            pair, scores = zip(*pairdict.items())
            pair = pair[0]
            scores = scores[0]
            scoresum = scores[0] + scores[1]
            s1norm = scores[0] / scoresum
            s2norm = scores[1] / scoresum
            equalizednorm = abs(s2norm - s1norm)
            #rt offset after normalization was normal for 2ndprios previously
            outscore = equalizednorm - equalizednorm * scores[2]
            thirdpriorities.append([pair, outscore])

    sortedranks = sorted(rankedpairs, key=lambda x: x[1])
    secondranks = sorted(secondpriorities, key=lambda x: x[1])
    thirdranks = sorted(thirdpriorities, key=lambda x: x[1])
    sortedranks.extend(secondranks)
    sortedranks.extend(thirdranks)

    preservedpairs = set()
    preservedranks = []
    for pair, score in sortedranks:
        if pair not in preservedpairs:
            preservedranks.append([pair, score])
            preservedpairs.add(pair)

    groupsets = [set(itertools.chain(*g)) for g in connections.values()]

    print(time() - t2, 'riff raff')
    t3 = time()

#no jumping ahead on this one
#simple ordered ascension, make groups on the fly and merge them when appropriate
    distsets = [] #sets of distributions
    linelocations = {} #mass: index of distsets
    setcharges = defaultdict(set) #index of distsets: [charges]
    for pair, score in preservedranks:
        paircharge = paircharges[pair]
        pairset = set(pair)
        locs = set()
        for line in pairset:
            if line in linelocations:
                locs.add(linelocations[line])
        if locs:
            if len(locs) == 1:
                distindex = min(locs)
                dist = distsets[distindex]
                if dist.union(pairset) in groupsets:
                    distsets[distindex].update(pairset)
                    setcharges[distindex].add(paircharge)
                    for line in pairset:
                        if line not in linelocations:
                            linelocations[line] = distindex
            else:
                dist = pairset.copy()
                for l in locs:
                    dist.update(distsets[l])
                if dist in groupsets:
                    distindex = min(locs)
                    distsets[distindex].update(dist)
                    for l in locs.difference([distindex]):
                        setcharges[distindex].update(setcharges[l])
                        setcharges[l] = False
                        distsets[l] = False
                    for line in dist:
                        linelocations[line] = distindex
        else:
            distsets.append(pairset)
            distindex = len(distsets) - 1
            setcharges[distindex].add(paircharge)
            for line in pairset:
                linelocations[line] = distindex

    print(time() - t3, 'ranking')
    t4 = time()

    dr =  0
    franks = {} #distid: rank
    solodists = defaultdict(dict) #charge: distid: lines
    for dist, charges in zip(distsets, setcharges.values()):
        if dist:
            charge = max(charges)
            solodists[charge][dr] = list(dist)
            franks[dr] = dr #just roll with it
            dr += 1


    foundvals = []
    for charge, sgd in solodists.items():
        foundvals.extend(list(itertools.chain(*sgd.values())))
    specvals = regiter[:,8].astype(int)
    nodists = np.setdiff1d(specvals, foundvals)

    print(time() - t4, 'assembling')

    if zoomplotting:
        zst = st
        zet = et
        zumb = umb
        zlmb = lmb
        #zst = 48
        #zet = 50
        #zlmb = 720
        #zumb = 730
        newdists = defaultdict(dict)
        for fc, fgs in solodists.items():
            for fk, pkeys in fgs.items():
                fg = regions[pkeys,7]
                times = regions[pkeys,2:4]
                if fg.min() <= zumb and fg.max() >= zlmb:
                    if times.max() >= zst and times.min() <= zet:
                        newdists[fc][fk] = pkeys
        text = False
        ngroups = sum(len(i) for i in newdists.values())
        cols = dp.get_colors(ngroups)
        cn = 0
        fig, ax = plt.subplots(nrows=3, figsize=(6,8), sharex=True)
        for fc, fgs in newdists.items():
            for fk, pkeys in fgs.items():
                col = cols[cn]
                fg = regions[pkeys,7]
                cn += 1
                for p in pkeys:
                    a = np.array(trackedgroups[p])
                    ax[2].scatter(a[0], a[1], marker='.', color=col, s=0.3, alpha=0.3)
                    if text:
                        ax[2].text(a[0][-1], a[1][-1], str(regions[p,7].round(2)), color='white', fontsize=4)
                fints = regions[pkeys,5]
                ax[0].bar(fg, fints, color=col, alpha=0.5, width=0.01, label=str(fc))
                if text:
                    for fx, fy, pk in zip(fg.tolist(), fints.tolist(), pkeys):
                        ax[0].text(fx, fy + fy * 0.03, str(pk), color='white', fontsize=4)
                print(fg)
                print(fc, '-', np.diff(sorted(fg)))
                print('~')
                ax[1].hlines(cn, fg.min(), fg.max(), color=col, linewidth=0.6)
                for vert, pl in zip(fg, pkeys):
                    ax[1].vlines(vert, cn - 0.1, cn + 0.1, color=col, linewidth=0.6)
                vi = np.sort(fg)
                if vi.size > 2:
                    vstack = np.stack((vi[:-1], vi[1:]), axis=1)
                    editspots = np.diff(vstack) < subisomax
                    if editspots.any():
                        ewheres = np.where(editspots)[0].tolist()
                        for ew in ewheres:
                            subpair = vstack[ew].tolist()
                            subints = [spectrum[i] for i in subpair]
                            winint = subints.index(max(subints))
                            winner = subpair[winint]
                            if ew > 0:
                                #edit 1 before ew
                                vstack[ew-1,1] = winner
                            if ew < len(vstack) - 1:
                                #edit 1 after ew
                                vstack[ew+1,0] = winner
                        vstack = np.delete(vstack, ewheres, axis=0)
                else:
                    vstack = vi.reshape(1, -1)
                vdiffs = np.diff(vstack)
                vflat = sorted(vstack.flatten().tolist())
                labelspots = np.mean(vstack, axis=1).tolist()
                for ls, lp in zip(labelspots, vstack.tolist()):
                    labeldiff = np.diff(lp)[0].round(4)
                    chargedist = (proton/fc - labeldiff).round(4)
                    lstring = ' ~ '.join((str(fc), str(labeldiff), str(chargedist)))
                    if text:
                        ax[1].text(ls, cn - 0.2, lstring, fontsize=4, ha='center', color='white')

        ndmasses = regions[nodists,7]
        mdinds = np.logical_and(ndmasses >= zlmb, ndmasses <= zumb)
        ndtimes = regions[nodists,2:4]
        tdinds = np.logical_and(ndtimes.min(axis=1) <= zet, ndtimes.max(axis=1) >= zst)
        zdinds = regions[nodists,4] > 1
        finds = np.logical_and.reduce((mdinds, tdinds, zdinds))
        ndplotters = nodists[finds]
        if ndplotters.size > 0:
            ndmasses = regions[ndplotters,7].tolist()
            nints = regions[ndplotters,5].tolist()
            ax[0].bar(ndmasses, nints, alpha=0.5, color='white', width=0.01, label='N/A')
            if text:
                for fx, fy, nd in zip(ndmasses, nints, ndplotters):
                    if fy > 0:
                        ax[0].text(fx, fy + fy * 0.03, str(nd), color='white', fontsize=4)
            for p in ndplotters:
                a = np.array(trackedgroups[p])
                ax[2].scatter(a[0], a[1], marker='.', color='white', s=0.3, alpha=0.3)
                if text:
                    ax[2].text(a[0][-1], a[1][-1], str(regions[p,7].round(2)), color='white', fontsize=4)
        ax[0].set_yscale('log')
        ax[0].set_ylabel('intensity')
        ax[2].set_ylabel('minutes')
        ax[1].set_ylabel('distribution rank')
        ax[2].set_xlabel('m/z')
        for label in ax[2].get_xticklabels():
            #label.set_ha("right")
            label.set_rotation(-45)
        ncols = 6
        if text:
            ax[0].legend(title='Charge', loc='upper left', bbox_to_anchor=(0, 1.3 + (0.1 * cn / ncols)), ncol=ncols)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.05)
        plt.show()
        fig.clf()
        plt.close()


    t5 = time()

    distributionmasses = {} #distid: ordered masses
    distributioncharges = {} #distid: charge
    distributionsoflines = {} #line: distid
    linesofdistributions = {} #distid: mass-ordered linedkeys
    distributiontimelimits = {} #distid [starting rt, ending rt]
    distributionintensities = {} #distid: mass-ordered intensities
    distributionsbycharge = defaultdict(dict) #charge: dists: mass-ordered linekeys
    for charge, dists in solodists.items():
        for dist, lines in dists.items():
            dmasses = regions[lines,7]
            lineorder = dmasses.argsort()
            sortedlines = np.array(lines)[lineorder]
            sortedmasses = regions[sortedlines,7]
            dintensities = regions[sortedlines,5]
            #you could rt match the top 2 I suppose
            rtlimits = regions[sortedlines,2:4]
            minrt = rtlimits.min()
            maxrt = rtlimits.max()
            #
            distributionmasses[dist] = sortedmasses
            distributioncharges[dist] = charge
            for line in lines:
                distributionsoflines[line] = dist
            linesofdistributions[dist] = sortedlines
            distributiontimelimits[dist] = [minrt, maxrt]
            distributionintensities[dist] = dintensities
            distributionsbycharge[charge][dist] = sortedlines

#check for charge-state connections using whole groups
#sum their isotopomers at each time point and get a single area
#   ^allow linkage to groups with less isotopomers if their area is also less
#keep track of high / low intensity ratios for each adjacent pair
#overlap-check each line by RT
#initial matching by highest isotopomer? Then filter for things that contain the same number of isotopomers as what you're matching
#   ^only the things that can be directly matched will carry the charge-state priority, other things will match naturally I suppose.
#this will create a league of different priority levels that are generated from each additional charge state that's found.
#run the pairs here to be pairs of connections, if a pair is already present, skip it, only iterate upwards in charge then I suppose

#things to check:
#how charge state differences operate across different charges
#are RT's rather perfectly symmetrical across charge states? Like will a lower intensity charge state have all its RT's encompassed well enough to rely on? Then there's also the case of noisy channels - but this might be an easy y/n answer to whether or not there's noise!

#charge-state to charge-state competition can be based on RT overlap, ie that the right lines link to the right ones I suppose - whichever LINK is correct can be determined via the pair metrics already developed
#record expected mass diff across each mass and note difference/intensity
#note intensity ratio variation, is the ratio always expected to decrease at an isotopomer of lower intensity on a charge of lower intensity? Or is this more stochastic?
#note if RT's encompass
#show information about the ratio of intensities of one charge state to another below the individual plots

#make another kind of plot that log-scale shows the whole spectrum (x: m/z, y: intensity)? and show charge-state links as pointing lines up top?
#^you could also shrink the x-axis space between them and display an 'expected dist' alongside the original, but in a different color -> expected intensities too I suppose
#a 2nd plot below these would show RT overlaps (x: m/z, y: rt)

#serious ponderance, does a deuterated isotopomer tend to go towards a lower charge state? I keep seeing this RT shift on a weak signal that doesn't match a higher charge state, it's weird
#revolving 1 concept, most of these ratios, iso/iso, charge-state/charge-state, rt overlap, basdiffs?,should all essentially ~equate to a value of 1. If I take a rolling count of how far each of these are, independently, from 1, then sum/use the values the way I rank the pairs - that might be a decent ranking system for these. But it still doesn't provide me a way of cutting off bad matches.
#allow the charge-state process to break the rules of increasing/decreasing isotopomer intensities I suppose -> if it's there, the evidence is more legit than the 'model'

#^^^^^^^^^in regards to the ~slight~weird rt mismatch across RT's, I don't expect ionization to be a pefectly linear process, whereby ionizing more or less of something always shows the same (especially considering background ionizers) proportion of 2+ to 3+ etc, it might be a process that spawns some 3+ in greater quantity after a certain intensity/number of ions are reached. And this could easily cross over from the early to mid-early peak shape. I think this is a fair explanation. You could also reason this could an ordered generative process, whereby starting small and increasing [as the peak would] that you could generate differing numbers of either charge.

#so once a charge state is determined here, lower priority pairing can't add a more intense ion into the fray I suppose?

#I need to determine how I'm going to metricise the group errors and cross-charge %'s, do I connect to nearest mass/charge state? do I connect to nearest intensity? do I use all possible crosses to make a larger number of metrics that are simplified across means and more means?

    massranges = regions[:,:2]
    maxmass = massranges.max() + 1
    minmass = massranges.min() - 1
#reinvent this not to iterate over groupsize and check every match but to iterate over every group then check matches to only the desired charge state
#^make it a redundant search, it has to be so that everything from every perspective is covered
#^both positive and negative direction from the initial charge, do a wide search for ANY potential charge states within the full mass range above or below the distribution
#incorporate nodists into the lone edge search for a single ion of an appropriate charge
#^for now just do a +/-1 nodist to the max/min I suppose, see if any dists outside a 1 charge difference even match first
#for matching masses, do top 2, ONE of the top 2 of either should be the same
#^nah, use everything, then keep the argsort ranks of everything, and when say... a 2-length matches a 4-length, but the 2-length matches the 2 lowest ions, it would fail because the [0,1] of the 2-length wouldn't match the [2,3] of the 4-length via close order matching.

#top version is for collecting every possible charge state, use it later to see if there are any decent patterns, I didn't find any initially but I'd like to take a deeper look once I can
#groupid = 0
#chargegroups = defaultdict(set) #groupid: [connection keys]
##intersection_merge the set of key + values here to get finalized plotting groups?
#for charge, dists in distributionsbycharge.items():
#    for dkey, lines in dists.items():
#        dmasses = distributionmasses[dkey]
#        minrt, maxrt = distributiontimelimits[dkey]
#        basemasses = dmasses * charge - proton * charge
#        trial = charge + 1
#        chargetrials = []
#        pos = True
#        searching = True
#        boundaryfault = False
#        #finding all potential charges across full mass range
#        while searching:
#            scoutmasses = (basemasses + proton * trial) / trial
#            if scoutmasses.max() >= minmass:
#                if scoutmasses.min() <= maxmass:
#                    chargetrials.append(trial)
#                else:
#                    boundaryfault = True
#            else:
#                boundaryfault = True
#            if boundaryfault:
#                if pos:
#                    trial = charge
#                    pos = False
#                    boundaryfault = False
#                else:
#                    searching = False
#            if pos:
#                trial += 1
#            else:
#                trial -= 1
#            if trial < 1:
#                searching = False
#        for trial in chargetrials:
#            if trial in distributionsbycharge:
#                scoutmasses = (basemasses + proton * trial) / trial
#                expdiff = proton / trial
#                meandiff = np.diff(scoutmasses).mean()
#                ctol = (expdiff - meandiff)
#                for mkey, mlines in distributionsbycharge[trial].items():
#                    matchminrt, matchmaxrt = distributiontimelimits[mkey]
#                    if minrt < matchmaxrt and maxrt > matchminrt: #faster if using this first? - yea, a lil, needs more profiling though
#                        matchmasses = distributionmasses[mkey][:,None]
#                        if matchmasses.max() >= scoutmasses.min() and scoutmasses.max() > matchmasses.min(): #mass ranges overlap
#                            overpass = False
#                            if minrt > matchminrt and maxrt < matchmaxrt: #primary encompassed
#                                overpass = True
#                            elif matchminrt > minrt and matchmaxrt < maxrt: #secondary encompassed
#                                overpass = True
#                            else:
#                                overlap = min(matchmaxrt, maxrt) - max(matchminrt, minrt)
#                                fullrange = max(matchmaxrt, maxrt) - min(matchminrt, minrt)
#                                if (overlap * 2) / fullrange > 0.75:
#                                    overpass = True
#                            if overpass:
#                                matchmatrix = np.abs(scoutmasses - matchmasses) < ctol
#                                mmshape = matchmatrix.shape
#                                majoraxis = np.argmax(mmshape)
#                                minoraxis = np.argmin(mmshape)
#                                matrixmatches = matchmatrix.any(axis=majoraxis)
#                                if matrixmatches.sum() >= matrixmatches.size / 2: #matching at least half I suppose, maybe make this a 'majority' be removing the =?
#                                    chargegroups[groupid].add(mkey)
#                                    #
#        if groupid in chargegroups:
#            chargegroups[groupid].add(dkey)
#            groupid += 1

    nodistmasses = regions[nodists,7]
    nodistkeys = np.arange(nodists.size) + di

    groupid = 0
    chargegroups = defaultdict(set) #groupid: [connection keys]
#intersection_merge the set of key + values here to get finalized plotting groups?
    for charge, dists in distributionsbycharge.items():
        for dkey, lines in dists.items():
            dmasses = distributionmasses[dkey]
            minrt, maxrt = distributiontimelimits[dkey]
            basemasses = dmasses * charge - proton * charge
            intensities = distributionintensities[dkey]
            intensityranks = intensities.argsort()[::-1]
            trial = charge + 1
            chargetrials = []
            pos = True
            searching = True
            boundaryfault = False
            #finding all potential charges across full mass range
            while searching:
                if trial in distributionsbycharge:
                    scoutmasses = (basemasses + proton * trial) / trial
                    if scoutmasses.max() >= minmass:
                        if scoutmasses.min() <= maxmass:
                            expdiff = proton / trial
                            scoutdiff = np.diff(scoutmasses)
                            ctol = np.abs(expdiff - scoutdiff).min() #change this to max to see if anything good is being missed // was made to abs for those > 0 acdiffs
                            ncons = 0
                            for mkey, mlines in distributionsbycharge[trial].items():
                                matchminrt, matchmaxrt = distributiontimelimits[mkey]
                                if minrt < matchmaxrt and maxrt > matchminrt: #faster if using this first? - yea, a lil, needs more profiling though
                                    matchmasses = distributionmasses[mkey][:,None]
                                    if matchmasses.max() >= scoutmasses.min() and scoutmasses.max() > matchmasses.min(): #mass ranges overlap
                                        overpass = False
                                        if minrt > matchminrt and maxrt < matchmaxrt: #primary encompassed
                                            overpass = True
                                        elif matchminrt > minrt and matchmaxrt < maxrt: #secondary encompassed
                                            overpass = True
                                        else:
                                            overlap = min(matchmaxrt, maxrt) - max(matchminrt, minrt)
                                            fullrange = max(matchmaxrt, maxrt) - min(matchminrt, minrt)
                                            if (overlap * 2) / fullrange > 0.75:
                                                overpass = True
                                        if overpass:
                                            diffmatrix = np.abs(scoutmasses - matchmasses)
                                            matchmatrix = diffmatrix < ctol
                                            mmshape = matchmatrix.shape
                                            majoraxis = np.argmax(mmshape)
                                            minoraxis = np.argmin(mmshape)
                                            matrixmatches = matchmatrix.any(axis=majoraxis)
                                            if matrixmatches.sum() >= matrixmatches.size / 2: #matching at least half I suppose, maybe make th is a 'majority' be removing the =?
                                                #I need a crossintensityrank check here, to validate things one at a time, as when doing groups of them below fails, there's no good way of separating bad matches from the good ones without senseless permutations, and that still probably wouldnt give good answers
                                                alignmentloc = np.argwhere(diffmatrix == diffmatrix.min())[0]
                                                minindex = min(alignmentloc)
                                                alignmentloc = [i-minindex for i in alignmentloc] #for when the closest match isn't the earliest matching index
                                                rmax = matchmasses.size - alignmentloc[0]
                                                lmax = scoutmasses.size - alignmentloc[1]
                                                maxsize = min(rmax, lmax)
                                                matchintensities = distributionintensities[mkey]
                                                matchintensityranks = matchintensities.argsort()[::-1]
                                                scoutrankslice = intensityranks[alignmentloc[1]:alignmentloc[1]+maxsize]
                                                matchrankslice = matchintensityranks[alignmentloc[0]:alignmentloc[0]+maxsize]
                                                slicesubtraction = np.abs(scoutrankslice - matchrankslice)
                                                if slicesubtraction.sum() < slicesubtraction.size - 1: #close order switching
                                                    chargegroups[groupid].add(mkey)
                                                    ncons += 1
                            if ncons < 1:
                                #nothing on the next charge was found
                                #check solodists for a lone ion matching the max
                                #if a matching line is found, continue to the next charge here as well
                                #make them into distributions, continue using di to label them and keep track of them on the fly - in case one gets matched twice
                                boundaryfault = True
                        else:
                            #search outside bounds
                            boundaryfault = True
                    else:
                        #search outside bounds
                        boundaryfault = True
                else:
                    #charge not present
                    boundaryfault = True
                if boundaryfault:
                    if pos:
                        trial = charge
                        pos = False
                        boundaryfault = False
                    else:
                        searching = False
                if pos:
                    trial += 1
                else:
                    trial -= 1
                if trial < 1:
                    searching = False
            if groupid in chargegroups:
                chargegroups[groupid].add(dkey)
                groupid += 1


    print(time() - t5, 'charge group linkage')
    t6 = time()

#combining redundant matches
    chargesets = intersection_merge(chargegroups.values())

#combinatorics will be based off of this, and this allows things that didn't connect downward to branch in that direction
    chargelayers = defaultdict(lambda: defaultdict(set)) #groupid: charge: [connections]
    for n, cs in enumerate(chargesets):
        for c in cs:
            chargelayers[n][distributioncharges[c]].add(c)

#I suppose I should just use ths smallest charge group's size as the basis for matching the others, but it bothers me kinda
#I suppose, if that smallest group's size fails to match things, then divide things into those that matched and those that didn't? Charge order will matter here
#doing the smallest one for now, and throwing anything that doesn't pass into a separate list to visualize the negatives
#there's also some regulatory disparity for matches that don't have smaller matches I suppose
    fid = 0
    chargecongroups = defaultdict(dict) #connection: [all chargegroups its involved in]
    failedcongroups = defaultdict(dict)
    failuresbydist = defaultdict(int)
    distmatchcount = defaultdict(int)
    for cl in chargelayers.values():
        for cons in itertools.product(*cl.values()):
            #I need a way to check combinations of less of these layers should the longest one fail
            #I also need a way of determining the closeness of two ions here for when order switching should give an allowance
            for con in cons:
                distmatchcount[con] += 1
            #close order matching here, giving the intensities some leeway in terms of ranking without allowing them to move around too much - is the idea
            basemasses = [distributionmasses[i]*distributioncharges[i]-proton*distributioncharges[i] for i in cons]
            intensities = [distributionintensities[i] for i in cons]
            matrixsizes = [i.size for i in intensities]
            sizes = [i.size for i in intensities]
            maxind = sizes.index(max(sizes))
            lineup = basemasses[maxind]
            distinds = []
            for sm in basemasses:
                sdiff = np.abs(lineup - sm[:,None])
                alignmentloc = np.argwhere(sdiff == sdiff.min())[0]
                alignmentloc -= alignmentloc.min()
                luind = alignmentloc[1] - alignmentloc[0]
                outinds = [luind, luind + sm.size]
                distinds.append(outinds)
            leftbound = max(i[0] for i in distinds)
            rightbound = min(i[1] for i in distinds)
            alignmentsize = rightbound - leftbound
            #intensityranks = [i.argsort()[::-1].argsort() for i in intensities] #the added argsort doesn't do anything, how did I come up with that?
            #intensityranks = [i.argsort()[::-1] for i in intensities]
            #crossintensityranks = []
            #for n, (lc, rc) in enumerate(distinds):
            #    dl = leftbound - lc
            #    dr = dl + alignmentsize
            #    crossintensityranks.append(intensityranks[n][dl:dr])
            #crossintensityranks = np.array(crossintensityranks)
            #conranges = np.ptp(crossintensityranks, axis=0)
            #if conranges.sum() < conranges.size - 1: #close order switching
            crossmasses = []
            crossintensities = []
            for n, (lc, rc) in enumerate(distinds):
                dl = leftbound - lc
                dr = dl + alignmentsize
                crossmasses.append(basemasses[n][dl:dr])
                crossintensities.append(intensities[n][dl:dr])
            crossintensities = np.array(crossintensities)
            crossmasses = np.array(crossmasses)
            crossintensitysums = crossintensities.sum(axis=0)
            massmeandiff = np.abs(crossmasses.mean(axis=0) - crossmasses).mean()
            intensitypercs = crossintensities / crossintensitysums
            intensitymeandiff = np.abs(intensitypercs.mean(axis=1)[:,None] - intensitypercs).mean()
            for con in cons:
                #chargecongroups[con][cons] = concheck
                #concheck[cons].append([ccmeandiff, massdiff])
                chargecongroups[con][cons] = [intensitymeandiff, massmeandiff]
        #else:
        #    for con in cons:
        #        charge = distributioncharges[con]
        #        failedcongroups[fid][charge] = con
        #        failuresbydist[con] += 1
        #    fid += 1
#these two metrics are imbalanced, (953, 29) is better than (950, 29) but it wouldn't win here, the mass error is what makes it obvious but the mass metric doesn't impact the overall thing at all

#both this process and it's look-alike below for the regular pairs are not so much a normalizing, it's more like a balancing.
    prioritycharges = []
    secondpriorities = []
    for con, congroups in chargecongroups.items():
        if len(congroups) > 1:
            n1, n2 = np.array(list(congroups.values())).sum(axis=0)
            for congroup, (s1, s2) in congroups.items():
                newconscore = sum((s1/n1, s2/n2))
                prioritycharges.append([congroup, newconscore])
        else:
            congroup, conscore = list(congroups.items())[0]
            nconscore = conscore.copy()
            nconscore.insert(0, congroup)
            for con in congroup:
                secondpriorities.append([congroup, sum(conscore)])

    rankedcharges = sorted(prioritycharges, key=lambda x: x[1]) #competition among matched chargegroups where matching can be done easily
    secondprioritycharges = sorted(secondpriorities, key=lambda x: x[1]) #no competition happening here, the order doesn't actually matter
    rankedcharges.extend(secondprioritycharges)
#^the connection groups take priority of the direction of what lines match to each other. There will still be a secondary pairing process for lines going on here, but each new connection group gained will have to have a superset of the existing lines for the connection group in question to grow.
#^but this has a problem with linking bad actors, there's no way to allow a subset cutoff
#I'm going to move forward accepting this for now, it could be modified via tweaking later
#^currently, it would rely on the robustness of the initial chargegroup selection to filter out poor distributions


#instead of iterating through the list at each entry, you could keep track of sets of lines made along the way, and only iterate until a superset is found - priority is given to those at the top of the list this way
#^yes but you need to do a second final sweep to make sure no existing subsets are still in the list, it's got to be free flowing, then eliminating for subsets
#^only create a blocking list for the second part, might not just be subsets but weird combinations of groups that aren't together under the prevailing logic

##what I have for the existing pair system:
#iterate pairs & score:
#    something passes the criteria for ranking:
#        orders come here
#iterate orders:
#    iterate orders again:
#        combine supersets
#
##what I propose above:
#make flylist
#iterate pairs & scores:
#    something passes the criteria for ranking:
#        list of sets kept on the fly:
#            - first check if the new passed pair is a sub/superset of anything in the existing flylist
#            - if not, its becomes its own entry
#            - if so -> merge, superset connections take priorities for charge group connection descriptions
#make blocked list
#iterate final superset list:
#    - block things already taken, in order, output should be the final list
#^this actually changes the outcome, and so would doing a post-priority blocking. You'd have to do an awkward post-priority subset removal. I also did try this previously for the pair matching process and it didn't work.


#~~~~
#multiple types of merging:
#superset merging to allow for larger distribution lengths to work out
#intersection merging to allow shorter distributions at different, faded, charges to still connect
#you can add more charge states
#things added at already possessed charge states must overlap with the existing line structure, and newer/previously unexposed lines can be added
#basically, add whatever you like as long as the existing line infrastructure at each charge is preserved
#^with some checks, the longest distribution shouldn't be on something less than the most intense I suppose
#free expansion as long as existing lines overlap
#^I'm going to skip checking that the rank of lengths equates to the rank of intensities, although this would probably be an appropriate check.


    blocked = set()
    preservedchargeranks = []
    for group, score in rankedcharges:
        if group not in blocked:
            preservedchargeranks.append([group, score])
            blocked.add(group)
    print(time() - t6, 'charge set and priority ranking')
    t7 = time()

    chargeid = 0
    blocked = set()
    chargedistlines = defaultdict(lambda: defaultdict(set)) #chargegroupid: charge: [lines]
    chargedistgroups = defaultdict(dict) #chargegroupid: charge: distributionid
    chargegroupsbyline = {} #line: chargegroupid, doubles as blocking list
    chargesbyline = {} #line: charge
    for pn, (cons, score) in enumerate(preservedchargeranks):
        #join anything that has a sorted tuple key to connectionsbykeys with the existing line infrastructure at each charge state
        #ckeys = set(itertools.chain(*(linesofdistributions[i] for i in cons)))
        if not any(i in blocked for i in cons):
            #^if none are blocked, none have been used -> make new group I suppose?
            grouplines = {}
            groupdists = {}
            for con in cons:
                charge = distributioncharges[con]
                distkeys = linesofdistributions[con]
                chargedistlines[chargeid][charge] = distkeys
                chargedistgroups[chargeid][charge] = con
                for line in distkeys:
                    chargegroupsbyline[line] = chargeid
                    chargesbyline[line] = charge
            blocked.update(cons)
            chargeid += 1
    print(time() - t7, 'charge group assembly')
    print(time() - t1, 'total')
    print(increment, '-', increment + incs, ':', len(chargedistgroups))
    print('~~~~~~~~~~')
    increment += incs
print(time() - wt, 'final')

#savedir = '/store/flowcharacterizations/round3/DDAs/fileprocessing'
#fname = ''.join((mzmlfile.split('/')[-1].split('.')[0], '.isotopes.pickle'))
#saverloc = '/'.join((savedir, fname))
#savedbits = [dict(solodists), nodists, linesofdistributions, distributionsoflines, distributioncharges, distributionsbycharge, dict(chargedistgroups), dict(chargedistlines), chargegroupsbyline, chargesbyline]
#with open(saverloc, "wb") as pick:
#    pickle.dump(savedbits, pick)


#first make a regions for distributions
#then make a regions for charge states

t8 = time()

distributionregions = []
for k, v in solodists.items():
    for sk, sv in v.items():
        masses = distributionmasses[sk]
        maxmass = masses.max()
        massmin = masses.min()
        intensities = distributionintensities[sk]
        mainmass = masses[intensities.argmax()]
        mintime, maxtime = distributiontimelimits[sk]
        signalsum = defaultdict(float) #time: total intensity
        for line in sv:
            data = trackedgroups[line]
            for t, i in zip(data[1], data[2]):
                signalsum[t] += i
        signals = np.array(list(signalsum.items()))
        area = np.trapezoid(signals[:,1], signals[:,0])
        el = [minmass, maxmass, mintime, maxtime, len(sv), area, k, mainmass, sk]
        distributionregions.append(el)
distributionregions = np.array(distributionregions)
distributionregions = distributionregions[distributionregions[:,8].argsort()]

print(time() - t8, 'distribution regions')

t9 = time()

chargeregions = []
for k, v in chargedistgroups.items():
    mincharge = min(v)
    maxcharge = max(v)
    times = []
    signalsum = defaultdict(float) #time: total intensity
    for sv in v.values():
        times.extend(distributiontimelimits[sv])
        for line in linesofdistributions[sv].tolist():
            data = trackedgroups[line]
            for t, i in zip(data[1], data[2]):
                signalsum[t] += i
    signals = np.array(list(signalsum.items()))
    distinds = np.array(list(v.values()))
    charges = np.array(list(v))
    maincharge = charges[distributionregions[distinds,5].argmax()]
    mintime = min(times)
    maxtime = max(times)
    el = [mincharge, maxcharge, mintime, maxtime, len(v), area, maincharge, k]
    chargeregions.append(el)
chargeregions = np.array(chargeregions)

print(time() - t9, 'charge regions')

#charges across time for dists and charges independently
#dist length vs time
#number of charge states vs time
#number of charge states vs distribution length

starts = defaultdict(set)
ends = defaultdict(set)
for r in distributionregions:
    starts[r[2]].add(int(r[-1]))
    ends[r[3]].add(int(r[-1]))

pool = set()
timestats = []
titles = ['# distributions', '#1+', '#2+', '#3+', '#4+', '#5+', '#6+^', '#2-length', '#3-lengths', '#4-lengths', '#5-lengths', '#6^-lengths', '# unique charge states', '# unique distribution lengths', '#2-states', '#3-states', '#4-states', '#5^-states', '# total distibutions with multiple charge states', 'time']
for t in timearray.tolist():
    pool.update(starts[t])
    timetracker = np.zeros(len(titles))
    uniquecharges = set()
    uniquelengths = set()
    for p in pool:
        r = distributionregions[p]
        timetracker[0] += 1
        if r[6] < 6: #charge
            timetracker[int(r[6])] += 1
        else:
            timetracker[6] += 1
        if r[4] < 6: #distribution length
            timetracker[int(r[4])+5] += 1
        else:
            timetracker[11] += 1
        uniquecharges.add(r[6])
        uniquelengths.add(r[4])
        if p in chargesbyline:
            timetracker[18] += 1
            cbl = chargesbyline[p]
            if cbl < 5:
                timetracker[cbl+12] += 1
            else:
                timetracker[17] += 1
    timetracker[0] += len(pool)
    timetracker[12] += len(uniquecharges)
    timetracker[13] += len(uniquelengths)
    timetracker[19] += t
    timestats.append(timetracker.tolist())
    for r in ends[t]:
        pool.remove(r)
timestats = np.array(timestats)

plt.plot(timestats[:,19], timestats[:,0], '-', color='white', label='# dists')
plt.legend()
plt.show()

plt.plot(timestats[:,19], timestats[:,1], '-', alpha=0.5, color='cyan', label='1+')
plt.plot(timestats[:,19], timestats[:,2], '-', alpha=0.5, color='fuchsia', label='2+')
plt.plot(timestats[:,19], timestats[:,3], '-', alpha=0.5, color='white', label='3+')
plt.plot(timestats[:,19], timestats[:,4], '-', alpha=0.5, color='orangered', label='4+')
plt.plot(timestats[:,19], timestats[:,5], '-', alpha=0.5, color='lawngreen', label='5+')
plt.plot(timestats[:,19], timestats[:,5], '-', alpha=0.5, color='darkorange', label='6^+')
plt.legend()
plt.show()

plt.plot(timestats[:,19], timestats[:,7], '-', alpha=0.5, color='cyan', label='2L')
plt.plot(timestats[:,19], timestats[:,8], '-', alpha=0.5, color='fuchsia', label='3L')
plt.plot(timestats[:,19], timestats[:,9], '-', alpha=0.5, color='white', label='4L')
plt.plot(timestats[:,19], timestats[:,10], '-', alpha=0.5,  color='orangered', label='5L')
plt.plot(timestats[:,19], timestats[:,11], '-', alpha=0.5,  color='lawngreen', label='6^L')
plt.legend()
plt.show()

plt.plot(timestats[:,19], timestats[:,12], '-', color='orangered', label='Unique Charges')
plt.plot(timestats[:,19], timestats[:,13], '-', color='lawngreen', label='Unique Lengths')
plt.legend()
plt.show()


plt.plot(timestats[:,19], timestats[:,14], '-', color='cyan', label='2S')
plt.plot(timestats[:,19], timestats[:,15], '-', color='fuchsia', label='3S')
plt.plot(timestats[:,19], timestats[:,16], '-', color='white', label='4S')
plt.plot(timestats[:,19], timestats[:,17], '-', color='orangered', label='5^S')
plt.legend()
plt.show()

plt.plot(timestats[:,19], timestats[:,18], '-', color='orangered', label='# multiply charged')
plt.legend()
plt.show()


#sum lines of a distribution across time and take the area across charge states -> largest area should have the longest distribution -> if not, correct the longer less intense distributions to match the most intense one. This can be used as the future 'leader' to later decide whether distributions in the pairing process are allowed to expand.
#after pair-match charge states are assembled, you can look for the outer-fray lone lines that match major lines of a charge distribution

#charge-states of lesser intensity would need to be blocked, later on, from making a pair connection, that a charge-state of a higher intensity has not made

#now I need to visualize ratios, and charge distances!
#x: basemass
#y: intensity ratios
#color: charge

#horizontal bar between isotopomer intensity lines
#horizontal bar at twiny intensity ratio
#above bar, percent diff to expdiff
#below bar, absolute diff to expdiff

#basemass oriented bar chart, ordered by mainmass intensity, highest -> lowest for the sub-order? Though perhaps higher charge-states might show different behavior than lower ones when the highest intensity charge-state is in the middle somewhere.

#so, no, the intensity ratios aren't consistently ordered in any precise manner - they somewhat are, but they don't follow a strict ascent nor descent based on relative charge-state intensity

#can you perhaps explain the increase in isotopomer ratio from intensity fade via the ratio of isotopomers across charge states?
#^gotta visualize the cross charge-state isotopomer ratios as another potential avenue for a metric

#when there's a split match on the same distribution's connections, if the match with a higher intensity shows a smaller mass error, that's your winner

#I don't think the fade comes from analyzer dynamic range, or anything of the sort. I think that's an ionization phenomenon, as it always happens to the lowest intensity isotopomer of the lowest intensity charge state. Even when other charge states are at extremely low intensities for the analyzer, ie 1e5, the fade is happening only to the same exact isotopomer as you'd expect.

#I'll consider cross-charge-state error from the perspective of the most intense distribution. From there I can check whether mass error remains constant across each charge state and whether some distributions are actually a good match

#overall, the goal should be to pick which pairs to elevate, those can get plucked out of distributions that show decent matches. I think fade, even weird fade, can be allowed if the main isotopomer and it's best pair show good matching.
#all other categories of information should be used as a blocking mechanism to prevent the link from occuring.
#but I don't have a mechanism that allows a subset connection to overtake a superset should there be good reason.
#^that comes with the blocking. The lower priority (below charge-state priority) can allow extra pairs to be made so long as it's present in the highest intensity charge

#After the isotope process is done, when you're matching them to theoretical quantities, that fade may be an important feature there, it might be worth-while to determine which isotopes need that quantity corrected, and how much correction is being given. I'd wager you could systematically fix those errors while improving all fittings by either seeing if the faded quantity is 'stolen' by other charge states, or by assuming the lost/gained amount from the adjacent isotopomer ratios.
#adjacent isotopomer ratios can be used to determine which isotopomers of which charge state can be trusted as an example quantity for applying a correction by the cross-charge state ratio. This mechanism can also mark an isotopomer of a charge state as faded. If you can't determine that any of the adjacent ratios are good for a specific isotopomer, then you shouldn't accept that specific isotopomer for any of the distributions.
#^If this happens in the middle of a distribution, you'll have to split the ends I suppose. If it's in the middle for just one, cut off the rest of only that distribution and let the pieces compete at a lower priority.
#It might be worth looking for missing line-pieces on the faded isotopomers.
#in there case where there's 2 charge states and you have no idea of knowing if an isotopomer of either is a legit value, then you can just accept them as long as the masses and everything else lines up well. The order of intensities matters in this case.

#competetive processes can't be relied upon here the same way they do for basic iso distributions because there's not enough guaranteed competition over each line.
#consistency is key -> Find a conserved structure of ranks, this part must be perfect. Even if other isotopomers likely do match, they'll be left to match via the lower priority system.
#For competing charge-states, the average distance of the cross-charge %'s can be used in place of acdiff, but the acdiff goes for that distribution as opposed to any individual pairs. If a charge state with a super/sub-set of isotopomers gets outcompeted, it's not allowed to pass or connect. One distribution wins, and the contents of which are blocked.
#the second metric will be the same concept, except distance from average basemass error
#If a different charge state of a faded isotopomer can match the cross-charge %'s, that isotopomer should be accepted?
#Basically, if 2 charge states (potentially out of 3+) can agree on an isotopomer, a shittier one in a different charge state can be accepted.
#^So how would this work in the case where the most intense charge state is bullshit, but two lesser ones aren't?
#in the case of a noisy charge state, use a charge state with less competition to determine winning lines.

#summary:
#conserved structure of ranks - hard cutoff that decides what isotopomers gets to compete as a charge state. isotopomers that break ranks aren't included in any charge state - unless there's more than 1 charge state where it's consistent, then the third might be dragged along I suppose?
#^Allow for supersets to win this, then move on to the next step below for those supersets of which the ranks all play out.
#individual distributions compete via mean distance to mean charge state %'s, and mean distance to mean masss error
#mark known/visible fades by key in a fadedict of some sort

#After a brief, non-extensive search, I didn't see any reason to look for non-adjacent charge states, but this isn't the end of wondering if they're there.

#you might need to change the basemass-based connection, if two values are really close to being the basemass and if this switches back and forth across charges, you'd miss it

#seemed to have missed a line because of a slight switch in order of intensities, perhaps I should close order switching? Things can move up and down by like, 1
#^so subtract the argsorts and any absolute values > 1 would negate the process, however this might introduce some actual fuckery, I might want to make sure the actual intensity values are close in both distributions, maybe adjacent ratios here could play a part

#(1049, 494)
#^{2: [246586, 247679], 3: [246676, 248098]}
#(1205, 1126)
#^{2: [248787, 249475], 3: [249835, 251301]}
#(1111, 803), potential climb is too high? signal looks solid though
#^{4: [248824, 249347, 249797], 3: [249198, 249722, 250322]}

#bad match, coincidence:
#(2605, 1682) - I think the cross-charge % gives this away the strongest, its pattern also ~matches the ppm error pattern, the mass error isn't terrible

#needs close order switching:
#(2689, 2596)

#big mass errors:
#(2629, 87) -> has a better candidate that lost it seems, but that's not the only problem?
#(2615, 21) -> has a better candidate that lost? NO! It was just barely outside the massrange that I was slicing

#cross-charge % should have some % limit for a distance from the mean to be seem as a legit charge candidate
#(2327, 10), (2658, 2352) are good ~limits? it's a good match

#I need a way for longer distributions that have the intensity threshhold passed to swoop in and pick up things that have faded that didn't pass that threshhold -> the connection pairs are 100% always made so I just need a way to find it
#^so essentially, if the main distribution can find a key that matches the tuple(sorted(set)) of the keys, the I guess it's fine to link it to a lesser ranked charge state
#the close order switching isn't going to be straightforward, if the mainmasses switch, the distributions won't match -> I need the matching to be based on all masses rather than just the main
#I also want to exclude joining two 'different' distributions across charge states if their joining isotopomer differences are the most deviating from either of their means. It's a COINCIDENCE!
#^alternative strategy, via information processing later on: keep the distributions that don't really perfectly match glued together and search a distribution later on iteratively at it's multiple initiating sights aka adjacent isotopomer increases
#^both not dealing with the complicated bullshit upfront, and you can also make the assumption later on that they are two distributions if you decide you don't like the one. This helps deal with fade. Because otherwise... I don't really have a great way of dealing with fade. I would need massive overhead on the distribution linking process

#semi-side note, I'll need a cross-file validation scheme for MS1 predictions from one file that have MS2 scans in others. Or just a scheme to match MS2 identified peptides to their untargeted counterparts in other files.

#how well does the %'s from the sum of all isotopomers across charge states (adjacency ratio) match the adjacency ratios from the averaged charge state isotopomers? And which one matches theoretical distributions better on a large scale? Might be able to say something about the ionization events here.


#in this there are two clear bad matches, and one need for a type of close order switching
#forr the closeorderswitch match, you can make the initial match to either mainmass or monomass, that's pretty easy - and just do a logical_and to link the inds
#^and the idea of the closeorderswitch should be able to accomodate a specific number of switches based on the length I suppose
#rt mismatches are the cause of the bad matching ones, the whole distributions needs to be penalized for one bad rt match, all rt's need to be aligned under the fullmatch criteria i suppose
#although one of the bad rt matches should have been beaten out by a pair rather than the triplet that won...
#acceptable cross-charge %'s should always be either larger or smaller, never crossing, MAYBE this could only cross off a bad match when the mass error is the largest? or like 10x more than others?

#majority of isotopomers have a >= majority rt overlap of their cross charge state isotopomers in the event where something can be wrong, this can look at +/-1 charge state beyond to see if a match is there. If it is, accept the bad one in the middle. "Bad match" in this case would be something where the RTs/intensities are out of wack but the mass is good.

#TEST LATER:
#I see an interesting phenomenon, where when the intensities of the main masses are really close to 1:1, the next adjacent isotopomers of the charge state with the less intense main mass tend to be higher than those of the charge state with the main mass. And I'm wondering if this can be an observable ionization phenomenon that somehow depends on mass. And whether I can deduce that a distribution has subisotopomers to the left or right of their majors perhaps?
#(2222, 104) and (2625, 1203) are examples

#look into:
#(25555, 118), I'm not sure that 4-group should have beat the 2

#kicking something out of a charge state later on -> if a distribution wants to swoop in and claim some already claimed territory, and it offers both better alignment + a longer distribution to which the charge state can't compete AND there isn't a longer charge state chain to stand up for it, -> remove it I suppose!

#if the largest mass error and the largest rt deviation go together -> dump it I suppose, but how to determine if the largest is acceptable or not? It should also be a ~lower intensity and on the end of the distribution I suppose.
#rt overlap should basically be put into the metric process as a mean distance from the mean thing

#pretty reasonable observation:
#<1, more intense ions should have large cross-charge ratios, for ratios >1, less intense ions should have larger cross-charge ratioss
#^USUALLY, but not always, perhaps this would be a good normalizing metric to determine how a particular charge state or distribution might have been suppressed?
#For example, if the expected ratios switch, then normalize the more intense ion upwards by assigning it a greater intensity than measured? rather than normalizing the smaller one upwards.
#this adds some more justification to the close order switching process too, it seems like it's just another area where the data is extremely fragile.
#I'm also seeing larger adjacency ratios on less intense charge states - I think, investigate this more

#(3234, 1632) don't belong together because 3234 has a better distribution to be a part of that doesn't match

#the kickout process can be done after pair matching and will essentially check the close order matching of things all over again to determine if things still deserve to be linked, like a less intense distribution with more legs -> doesn't match the higher
#^or a lower intensity distribution that doesn't really match the expanded higher intensity one -> the orders are out of wack compared to how it should have faded

#a bad pattern is like (3236, 16) where the rank orders differ, and the cross charge percentages overlap -> this should be an auto-no, but how?

#new metrics:
#cross-charge order, for both intensities and dpoints
# - no need for negatives/flipped ratios, use raw ones - same effect
# - each isotopomer is normalized to the sum of that isotopomer at every charge state
# - then the you can take mean difference to mean at each isotopomer and therefore the metric will judge whether the isotopomer is a good addition
#currently, for ppm: the errors are averaged across a distribution but they should be averaged across isotopomers -> then the mean difference to this is the value to them mean across the other isotopomers -> normalize by number of isotopomers
# - average across basemass by isotopomer, then take distance to that average

#probems are stemming from the superset merging in the final ranking, bad things don't get poofed away
#looks like i'll need to allow </> 0 acdiffs or whatever, they happen

#I want to visualize how the charge distance scales back up to the ~proton length for each distribution across charge states
#I'll also put in the intensity/sum intensity across isotopomer plots
#I also want to visualize RT centering for each individual distribution, might be worth taking a ratio somewhere here
#basemass distance from average too

#check for charge distance:
#(3998, 3803) -> meh, pairmatching will fix it
#(3979, 3546, 1894) -> i want the new visualizations above
#(3966, 3185) -> their other connections don't seem to match that well because the only matches of the correct size have offset isoopomers. I'll need to look into more

#implement that each overlapping RT must have 75% fullrange, if they don't match here but do so later on via pair matching, so be it, but there's too many cases here of RT's being off for individual straggler ions, and this would be a decent place to cut off.
#for now, things that aren't distributions, yet that match perfectly, can match. For example (2870, 31) has a great match but totally isn't a distribution, they each clearly have a good distribution of their own extending in opposite directions -> when pair matching finds those pairs, you can excommunicate these two. I suppose a third charge state would override something like this? I'm not sure.
#^the ensuing matches would need to NOT have the strangest charge state distance
#can you regard fade the same at higher masses as lower? ie at lower charge states and at high? I'm not sure they act the same way

#current plan:
#re-impement activedirection, pairmatch FIRST, THEN allow for charge linkages afterwards, it seems backwards but it also seems like it would work better. The pairmatching scheme is HOT on accuracy, allowing this to take the reins beforehand would allow for much better matches
#^PLUS I can allow subset charge pairing for dists that don't form a connection coalition to pair individually pair across more than one charge state to get succesfully get the full superset link
#the resulting distribution matches can be based PURELY on the isotopomers that match, while only holding in consideration that more intense distributions should be ~longer
#^so you really only need to match the 2 highest isoptopomers, which will obviously be right next to each other, and then you can expand outwards I suppose. As any match that can't at least agree on the top 2 isn't going to be a legit match I think. The +/-1 charge state search to aid in seeing if one of them is just fucked up would help here too.
#when matching distributions, neither the mainmass or firstmass ideas are going to be able to fly alone: leader masses -> anything within 90% of the highest intensity mass? that way you can match distributions if the main mass switches across charge state, distribution structures can be filtered afterwards.
plotdicts = {
        'pass': chargedistgroups,
        'fail': failedcongroups,
        }

#(2702, 62), a bad match in both pairmatches
#^but is that (2760, 569)?

for status, distgroups in plotdicts.items():
    for distkey, dists in distgroups.items():
        chargeorder = sorted(dists)
        chlen = len(dists)
        cf = pd.DataFrame()
        chargefigures = {c:n for n, c in enumerate(chargeorder)}
        fig, ax = plt.subplots(ncols=chlen, nrows=8, figsize=(6,8), sharex='col', sharey='row')
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        cg = list(dists.values())
        chargedistbounds = [np.inf, 0]
        intensitylineup = [distributionintensities[g] for g in cg]
        flatintensities = itertools.chain(*intensitylineup)
        maxmain = max(flatintensities)
        masslineup = [distributionmasses[g]*distributioncharges[g]-proton*distributioncharges[g] for g in cg]
        masslineup = sorted(masslineup, key=lambda x: -x.size)
        intensitylineup = sorted(intensitylineup, key=lambda x: -x.size)
        arraysizes = [i.size for i in masslineup]
        if len(set(arraysizes)) == 1:
            arraymeans = np.array(masslineup).mean(axis=0)
            intensitysums = np.array(intensitylineup).sum(axis=0)
        else:
            matrixmax = max(arraysizes)
            arraysums = np.zeros(matrixmax)
            arraydividends = np.zeros(matrixmax)
            intensitysums = np.zeros(matrixmax)
            movinglineup = masslineup[0]
            for n, (a, i) in enumerate(zip(masslineup, intensitylineup)):
                basediffs = np.abs(movinglineup - a[:,None])
                alignmentloc = np.argwhere(basediffs == basediffs.min())[0].tolist()
                minindex = min(alignmentloc)
                alignmentloc = [i-minindex for i in alignmentloc] #for when the closest match isn't the earliest matching index
                cnmax = a.size - alignmentloc[0]
                bmax = movinglineup.size - alignmentloc[1]
                maxsize = min(bmax, cnmax)
                basearray = movinglineup[alignmentloc[1]:alignmentloc[1]+maxsize]
                meanarray = a[alignmentloc[0]:alignmentloc[0]+maxsize]
                iarray = i[alignmentloc[0]:alignmentloc[0]+maxsize]
                intensitysums[alignmentloc[1]:alignmentloc[1]+maxsize] += iarray
                arraysums[alignmentloc[1]:alignmentloc[1]+maxsize] += meanarray
                arraydividends[alignmentloc[1]:alignmentloc[1]+maxsize] += 1
            arraymeans = arraysums / arraydividends
        tbarwidth = 1 / len(cg)
        tspace = 0.5
        cols = dp.get_colors(len(cg))
        cmin = np.inf
        cmax = 0
        cratiolist = []
        ccbounds = [np.inf, 0]
        for n, (charge, g) in enumerate(dists.items()):
            lines = linesofdistributions[g]
            cintensities = distributionintensities[g]
            cratios = cintensities[:-1] / cintensities[1:]
            cratios[cratios < 1] = -1 / cratios[cratios < 1]
            cratiolist.append(cratios)
            abcratios = [abs(i) for i in cratios]
            cmasses = distributionmasses[g]
            if cintensities.min() < cmin:
                cmin = cintensities.min()
            if cintensities.max() > cmax:
                cmax = cintensities.max()
            concharge = distributioncharges[g]
            expdiff = proton / concharge
            basemasses = cmasses * concharge - proton * concharge
            basediffs = np.abs(basemasses - arraymeans[:,None])
            alignmentloc = np.argwhere(basediffs == basediffs.min())[0].tolist()
            minindex = min(alignmentloc)
            alignmentloc = [i-minindex for i in alignmentloc] #for when the closest match isn't the earliest matching index
            cnmax = arraymeans.size - alignmentloc[0]
            bmax = basemasses.size - alignmentloc[1]
            maxsize = min(bmax, cnmax)
            basearray = basemasses[alignmentloc[1]:alignmentloc[1]+maxsize]
            meanarray = arraymeans[alignmentloc[0]:alignmentloc[0]+maxsize]
            meanbasediff = meanarray - basearray
            meanbaseppms = (meanbasediff / basearray) * 1000000
            cmx = cmasses[alignmentloc[1]:alignmentloc[1]+maxsize]
            acdiffs = expdiff - np.diff(cmasses)
            basediffs = acdiffs * concharge
            conhax = chargefigures[concharge]
            cwidth = 0.5 *  len(lines)
            chargelengthextra = proton / charge * 2
            nst = regions[lines,2].min() - 0.5
            net = regions[lines,3].max() + 0.5
            nlmb = regions[lines,0].min() - chargelengthextra
            numb = regions[lines,1].max() + chargelengthextra
            nboundrec = [nlmb, numb, nst, net]
            nplotkeys = arg_coord_rectangle_overlap(nboundrec, regions[:,:4]).tolist()
            for p in nplotkeys:
                if p not in lines:
                    a = trackedgroups[p]
                    creg = regions[p]
                    ax[0][conhax].plot(a[0], a[1], '.', color='white', markersize=0.8, alpha=0.3)
                    ax[0][conhax].plot(a[0], a[1], '-', color='white', linewidth=0.4, alpha=1)
                    ax[1][conhax].plot([creg[7], creg[7]], [0, creg[5]], '-', color='white', alpha=0.5, linewidth=cwidth)
            ax[5][conhax].hlines(0, cmasses.min(), cmasses.max(), color='black', linewidth=0.3)
            ax[7][conhax].hlines(0, cmasses.min(), cmasses.max(), color='black', linewidth=0.3)
            ax[2][conhax].hlines(proton, cmasses.min(), cmasses.max(), color='black', linewidth=0.3)
            for cline in lines:
                creg = regions[cline]
                ax[1][conhax].plot([creg[7], creg[7]], [0, creg[5]], '-', color=cols[n], alpha=1, linewidth=cwidth)
                a = trackedgroups[cline]
                ax[0][conhax].plot(a[0], a[1], '.', color=cols[n], markersize=0.8, alpha=0.3)
                ax[0][conhax].plot(a[0], a[1], '-', color=cols[n], linewidth=0.4, alpha=1)
                #ax[2][conhax].plot([creg[7], creg[7]], [0, creg[4]], '-', color=cols[n], alpha=1, linewidth=cwidth)
            basenorms = basediffs / proton
            abasediffs = np.abs(basenorms / proton - basediffs)
            diffgen = 0.05
            bn = 0
            bw = 0.02
            adjacentx = cmasses[:-1] + np.diff(cmasses) / 2
            ax[5][conhax].bar(adjacentx, cratios, width=diffgen, color=cols[n], alpha=1)
            #ax[6][conhax].bar(adjacentx, cpointratios, width=diffgen, color=cols[n], alpha=1)
            chargedists = np.diff(cmasses) * charge
            ax[2][conhax].bar(adjacentx, chargedists, width=diffgen, color=cols[n], alpha=1)
            intensitypercs = cintensities[:intensitysums.size] / intensitysums[:cintensities.size]
            for ip in chargedists:
                if ip < chargedistbounds[0]:
                    chargedistbounds[0] = ip
                if ip > chargedistbounds[1]:
                    chargedistbounds[1] = ip
            ax[4][conhax].bar(cmasses, intensitypercs, width=diffgen, color=cols[n], alpha=1)
            ax[6][conhax].bar(cmx, meanbaseppms, width=diffgen, color=cols[n], alpha=1)
            for nc, cn in enumerate(cg):
                cnmasses = distributionmasses[cn]
                cncharge = distributioncharges[cn]
                cnbases = cnmasses * cncharge - proton * cncharge
                cnintensities = distributionintensities[cn]
                basediffs = np.abs(basemasses - cnbases[:,None])
                alignmentloc = np.argwhere(basediffs == basediffs.min())[0].tolist() #[1, 0] here would mean the second of cnbases aligns to the first of basemasses
                minindex = min(alignmentloc)
                alignmentloc = [i-minindex for i in alignmentloc] #for when the closest match isn't the earliest matching index
                cnmax = cnbases.size - alignmentloc[0]
                bmax = basemasses.size - alignmentloc[1]
                maxsize = min(bmax, cnmax)
                basearray = cintensities[alignmentloc[1]:alignmentloc[1]+maxsize]
                cnarray = cnintensities[alignmentloc[0]:alignmentloc[0]+maxsize]
                cnratio = cnarray / basearray
                bx = cmasses[alignmentloc[1]:alignmentloc[1]+maxsize]+(diffgen*nc)
                cnratiobar = np.abs(cnratio.mean() - cnratio).mean()
                if cnratiobar > ccbounds[1]:
                    ccbounds[1] = cnratiobar
                if cnratiobar < ccbounds[0] and cnratiobar > 0:
                    ccbounds[0] = cnratiobar
                cf.loc[g, cn] = cnratiobar
                ax[3][conhax].bar(bx, cnratio, width=diffgen, color=cols[nc], alpha=1)
                #ax[4][conhax].bar(bx, pointratio, width=diffgen, color=cols[nc], alpha=1)
            for nc, cn in enumerate(cg):
                if cn != cg:
                    maincharge = distributioncharges[cn]
                    mainmasses = distributionmasses[cn]
                    mainbasemasses = mainmasses * maincharge - proton * maincharge
                    basediffs = np.abs(basemasses - mainbasemasses[:,None])
                    alignmentloc = np.argwhere(basediffs == basediffs.min())[0].tolist() #[1, 0] here would mean the second of cnbases aligns to the first of basemasses
                    minindex = min(alignmentloc)
                    alignmentloc = [i-minindex for i in alignmentloc] #for when the closest match isn't the earliest matching index
                    cnmax = mainbasemasses.size - alignmentloc[0]
                    bmax = basemasses.size - alignmentloc[1]
                    maxsize = min(bmax, cnmax)
                    basearray = basemasses[alignmentloc[1]:alignmentloc[1]+maxsize]
                    mainarray = mainbasemasses[alignmentloc[0]:alignmentloc[0]+maxsize]
                    maindiffs = mainarray - basearray
                    mainppm = (maindiffs / mainarray) * 1000000
                    bx = cmasses[alignmentloc[1]:alignmentloc[1]+maxsize]+(diffgen*nc)
                    diffbar = np.abs(mainppm.mean() - mainppm).mean()
                    ax[7][conhax].bar(bx+bn, mainppm, width=bw, color=cols[nc], alpha=1)
                    bn += bw + bw / 2
            ax[0][conhax].set_title(''.join((str(concharge), '(', str(g), ')', ' - ', str(failuresbydist[g]), '/', str(distmatchcount[g]))), fontsize=12)
        ax[1][0].set_yscale('log')
        #ax[2][0].set_yscale('log')
        ax[2][0].set_ylim(chargedistbounds[0]*0.95, chargedistbounds[1]*1.05)
        ax[3][0].set_yscale('log')
        ax[5][0].set_yscale('symlog')
        ax[4][0].set_yscale('log')
        #ax[6][0].set_yscale('symlog')
        ax[1][0].set_ylim(cmin/2, cmax)
        ax[1][0].set_ylabel('peak area')
        ax[0][0].set_ylabel('retention time', fontsize=6)
        ax[3][0].set_ylabel('cross-charge', fontsize=7)
        ax[5][0].set_ylabel('adjacency')
        ax[7][0].set_ylabel('ppm error')
        ax[2][0].set_ylabel('charge distances', fontsize=6)
        ax[4][0].set_ylabel('intensity sum %', fontsize=6)
        ax[6][0].set_ylabel('ppm to mean', fontsize=7)
        for ch, hax in chargefigures.items():
            ax[-1][hax].tick_params(axis='x', labelrotation=-45)
            if hax == 0:
                #invisible right splines
                ax[0][hax].spines.right.set_visible(False)
                ax[1][hax].spines.right.set_visible(False)
            elif hax == chlen-1:
                #invisible left splines
                ax[0][hax].spines.left.set_visible(False)
                ax[1][hax].spines.left.set_visible(False)
                for tick in ax[0][hax].yaxis.get_major_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
                for tick in ax[1][hax].yaxis.get_majorticklines():
                    tick.set_visible(False)
                for tick in ax[1][hax].yaxis.get_minorticklines():
                    tick.set_visible(False)
            else:
                #left and right invisible
                ax[0][hax].spines.left.set_visible(False)
                ax[0][hax].spines.right.set_visible(False)
                ax[1][hax].spines.left.set_visible(False)
                ax[1][hax].spines.right.set_visible(False)
                for tick in ax[0][hax].yaxis.get_major_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
                for tick in ax[1][hax].yaxis.get_majorticklines():
                    tick.set_visible(False)
                for tick in ax[1][hax].yaxis.get_minorticklines():
                    tick.set_visible(False)
        supt = ': '.join((str(distkey), status))
        plt.suptitle(supt, y=0.92)
        plt.show()
        fig.clf()
        plt.close()
        gc.collect()

































#early pairs charge-handling, region format
rgblow = lambda: (np.random.uniform(low=0.6, high=1), np.random.uniform(low=0.9, high=1), np.random.uniform(low=0.7, high=1))
rgbhigh = lambda: (np.random.uniform(low=0.9, high=1), np.random.uniform(low=0, high=0.1), np.random.uniform(low=0.8, high=0.9))
deepset = lambda: (np.random.uniform(low=0.2, high=1), np.random.uniform(low=0.4, high=1), np.random.uniform(low=0.1, high=0.8))

cfunc = lambda: (
        np.random.uniform(low=0.3, high=1), #R
        np.random.uniform(low=0.6, high=1), #G
        np.random.uniform(low=0.8, high=0.9) #B
        )
ndfunc = lambda: (
        np.random.uniform(low=0.9, high=1), #R
        np.random.uniform(low=0.1, high=0.6), #G
        np.random.uniform(low=0.1, high=0.6) #B
        )

with open(isotopefile, "rb") as pick:
    isotopeabundances, seqsbymass, subisotopicdifferences, newinclimit, steplimit = pickle.load(pick)

subisodiffs = np.array(list(subisotopicdifferences))[:,None]
subisotree = spatial.KDTree(subisodiffs)

st = 49.5
st = 49
et = 49.6
lmb = 364.5
umb = 367.7

st = 35
et = 35.5
lmb = 321
umb = 325.2

st = 33
et = 33.8
lmb = 352.9
umb = 354.3

st = 53.4
et = 54.2
lmb = 524.5
umb = 526.5


st = 49.8
st = 49
et = 49.9
et = 52
lmb = 522.2
umb = 528.3

#boundrec = [lmb, umb, st, et]
boundrec = [regions[:,7].min() - 1, regions[:,7].max() + 1, st, et]
regionsample = arg_coord_rectangle_overlap(boundrec, regions[:,:4]).tolist()
regiter = regions[regionsample][regions[regionsample,7].argsort()]

minpoints = 2
#minintensity = 0.4e6
chargetolerance = 0.1 #lesson learned: these differences DO get divided across charge states, if you normalize everything back to base mass without a charge, the errors become more consistent. They're smaller errors for higher charges etc. so going by percent here is FINE!

subisomax = subisodiffs.max()
subisomax = subisomax + subisomax * chargetolerance

di = 0
paircharges = {} #connection: charge
connectioncharges = {} #groupid: charge
datapdiffs = {} #connection: %-diff of number of datapoints, things that can be calculated directly between two peaks and don't need to rely on the entire distribution for information
rtoffsets = {} #connection: overlap balance, a %
connections = defaultdict(dict) #groupid: [pairs]
subgroups = defaultdict(lambda: defaultdict(set)) #mass: charge: groups, a mass is always active as long as it's in masspool, this keeps track of all the groups that end in any given mass
activedirection = defaultdict(int) #groupid: direction, 0 (increasing) or 1 (decreasing)

si = 0
subisogroups = defaultdict(lambda: defaultdict(set)) #subiso group: max charge for mass: [masses]
subisomasses = {} #mass: subisogroup

masspool = set()
t1 = time()
for reg in regiter:
    npoints = reg[4]
    if npoints >= minpoints:
        nm = reg[7]
        nmlt = reg[2]
        nmrt = reg[3]
        nmwidth = reg[1] - reg[0]
        nkey = int(reg[8])
        sst = reg[6]
        olddirections = activedirection.copy()
        masspoolremovals = set()
        nlrange = nmrt - nmlt
        for omr in masspool:
            oreg = regions[omr]
            omlt = oreg[2]
            omrt = oreg[3]
            olrange = omrt - omlt
            overpass = False
            if nmlt < omrt and nmrt > omlt: #rt's overlap
                overlap = min(omrt, nmrt) - max(omlt, nmlt)
                if nmlt > omlt and nmrt < omrt: #new encompassed
                    combinedrange = nlrange + olrange
                    percentoverlap = (overlap * 2) / combinedrange
                    #using newinclimit here is appropriate because it's directly related to expected differences in adjacent isotopomer quantities, which is what the overlap is also ~somewhat assessing
                    if percentoverlap > newinclimit:
                        overpass = True
                elif omlt > nmlt and omrt < nmrt: #old encompassed
                    combinedrange = nlrange + olrange
                    percentoverlap = (overlap * 2) / combinedrange
                    if percentoverlap > newinclimit:
                        overpass = True
                else:
                    #no = overlap / nlrange
                    #oo = overlap / olrange
                    #if no > 0.5 and oo > 0.5: #max of new/old overlap > 0.5, a majority overlap for both -> 0.75 now because this shit was too lenient, this might be ok for a hard-coded value
                    fullrange = max(omrt, nmrt) - min(omlt, nmlt)
                    percentoverlap = overlap / fullrange
                    if percentoverlap * 2 > 0.75: #this is super lenient I think
                        overpass = True
            if overpass:
                #^one of them should have to majority overlap, or fully encompass
                #eliminate anything in masspool > proton distance from nm
                okey = int(oreg[8])
                om = oreg[7]
                diff = nm - om
                omwidth = oreg[1] - oreg[0]
                widthbuffer = nmwidth + omwidth
                if diff < subisomax + widthbuffer:
                    #potential subisos
                    maxisocharge = round((subisomax / diff) - 0.5) #floor
                    if om in subisomasses:
                        si = subisomasses[okey]
                        subisogroups[si][maxisocharge].update((okey, nkey))
                        subisomasses[nkey] = si
                    else:
                        #make a new subiso group
                        subisogroups[si][maxisocharge].update((okey, nkey))
                        subisomasses[okey] = si
                        subisomasses[nkey] = si
                        si += 1
                elif diff <= proton + widthbuffer:
                    charge = round(proton / diff)
                    expdiff = proton / charge
                    acdiff = expdiff - diff
                    diffcut = expdiff * chargetolerance
                    if acdiff > -1 * (diffcut * chargetolerance + widthbuffer): #a lighter acceptance for distances above the charge dist - it does happen from time to time on lower-hanging straggler isotopomers
                        if acdiff <= diffcut + widthbuffer:
                            absacdiff = abs(acdiff)
                            #sst = spectrum[nm]
                            #sam = spectrum[om]
                            sam = oreg[6]
                            ncons = 0
                            csubs = subgroups[okey][charge]
                            intensitypercdiff = abs(sst - sam) / (sst + sam) / 2
                            if csubs:
                                for adi in csubs:
                                    ratiocheck = steplimit
                                    if olddirections[adi] > 0:
                                        #the dist has previously begun a decrease
                                        if sst >= sam:
                                            #and intensity is increasing
                                            ratiocheck = newinclimit
                                    if intensitypercdiff <= ratiocheck:
                                        lpair = (okey, nkey)
                                        connections[di].update(connections[adi])
                                        connections[di][lpair] = absacdiff
                                        connectioncharges[di] = charge
                                        subgroups[nkey][charge].add(di)
                                        if sst < sam:
                                            activedirection[di] += 1
                                        ncons += 1
                                        di += 1
                            else:
                                #no previous subgroup
                                if intensitypercdiff <= steplimit:
                                    lpair = (okey, nkey)
                                    connections[di][lpair] = absacdiff
                                    connectioncharges[di] = charge
                                    subgroups[nkey][charge].add(di)
                                    if sst < sam:
                                        #decreasing
                                        activedirection[di] += 1
                                    ncons += 1
                                    di += 1
                            if ncons > 0:
                                opoints = oreg[4]
                                dpercdiff = abs(npoints - opoints) / (npoints + opoints) / 2
                                datapdiffs[lpair] = dpercdiff
                                paircharges[lpair] = charge
                                #combinedrange = nlrange + olrange
                                #overlap = min((omrt, nmrt)) - max((omlt, nmlt)) #as long as minpoints > 1
                                #percentoverlap = (overlap * 2) / combinedrange
                                rtoffsets[lpair] = percentoverlap
                                if csubs:
                                    connections[di][lpair] = absacdiff
                                    connectioncharges[di] = charge
                                    subgroups[nm][charge].add(di)
                                    if sst < sam:
                                        #decreasing
                                        activedirection[di] += 1
                                    di += 1
                else:
                    #om is past proton distance, remove om from mass pool
                    masspoolremovals.add(okey)
        for mpr in masspoolremovals:
            masspool.remove(mpr)
        masspool.add(int(nkey))

print(time() - t1, 'mass processing')
t2 = time()

#charge-processing based on paircharges
#pairmasses = {k: regions[list(k),7] for k in paircharges}
#pairranges = {k: [regions[list(k),2].min(), regions[list(k),3].max()] for k in paircharges}
#pairintensities = {k: regions[list(k),5].tolist() for k in paircharges}

nt = time()

groupsbypair = defaultdict(set)
scoresbypair = defaultdict(dict) #mass: pair: [scores]
secondpriorities = defaultdict(dict) #essentially there's too many zeros caused by single-pair matches, they can be let in but they don't deserve top priority 
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
            outscore = scoreval - scoreval * offset
            ddiff = datapdiffs[sgk]
            outdiff = ddiff - ddiff * offset
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
    minvals = vpercs.sum(axis=1)
    #minvals = minvals + minvals * offsets
    rankedpairs.extend(list(zip(pairs, minvals)))

secondrankedpairs = [] #[pair, minval]
for m, pg in secondpriorities.items():
    pairs, vals = zip(*pg.items())
    offsets = [rtoffsets[i] for i in pairs]
    vals = np.array(vals)
    vpercs = vals / vals.sum(axis=0).tolist()
    minvals = np.abs(vpercs[:,1] - vpercs[:,0])
    minvals = minvals - minvals * offsets
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

print(time() - t2, 'riff raff')
t3 = time()

dr =  0
franks = {} #distid: rank
groupcharges = {} #groupid: charge
blocked = set()
solodists = defaultdict(dict)
for ngn, (gs, (gv, sgc)) in enumerate(zip(groupsets, groupranks)):
    ng = set()
    gcs = set()
    ng.update(gs)
    gcs.add(sgc)
    if not blocked.intersection(gs): 
        ngit = groupsets[ngn+1:]
        for ogn, ogs in enumerate(ngit):
            if ng < ogs: #this will not partially connect two differing charges
                if not blocked.intersection(ogs):
                    ng.update(ogs)
                    ngc = groupranks[ngn+ogn+1][1]
                    gcs.add(ngc) 
        mc = max(gcs)
        solodists[mc][gv] = list(ng)
        groupcharges[gv] = mc
        blocked.update(ng) 
        franks[gv] = dr
        dr += 1
print(time() - t3, 'ranking')

foundvals = []
for charge, sgd in solodists.items():
    foundvals.extend(list(itertools.chain(*sgd.values())))
specvals = regiter[:,8].astype(int)
nodists = np.setdiff1d(specvals, foundvals)

zoomplotting = False
if zoomplotting:
    zst = st
    zet = et
    zumb = umb
    zlmb = lmb
    zumb = 665
    zlmb = 660
    newdists = defaultdict(dict)
    for fc, fgs in solodists.items():
        for fk, pkeys in fgs.items():
            fg = regions[pkeys,7]
            times = regions[pkeys,2:4]
            if fg.min() <= zumb and fg.max() >= zlmb:
                if times.max() >= zst and times.min() <= zet:
                    newdists[fc][fk] = pkeys
    text = True
    ngroups = sum(len(i) for i in newdists.values())
    cols = dp.get_colors(ngroups)
    cn = 0
    fig, ax = plt.subplots(nrows=3, figsize=(6,8), sharex=True)
    for fc, fgs in newdists.items():
        for fk, pkeys in fgs.items():
            col = cols[cn]
            low, high = rgblow(), rgbhigh()
            fg = regions[pkeys,7]
            cn += 1
            for p in pkeys:
                a = np.array(trackedgroups[p])
                ax[2].scatter(a[0], a[1], marker='.', color=col, s=0.3, alpha=0.3)
                if text:
                    ax[2].text(a[0][-1], a[1][-1], str(regions[p,7].round(2)), color='white', fontsize=4)
            fints = regions[pkeys,5]
            ax[0].bar(fg, fints, color=col, alpha=0.5, width=0.01, label=str(fc))
            if text:
                for fx, fy, pk in zip(fg.tolist(), fints.tolist(), pkeys):
                    ax[0].text(fx, fy + fy * 0.03, str(pk), color='white', fontsize=4)
            print(fg)
            print(fc, '-', np.diff(sorted(fg)))
            print('~')
            ax[1].hlines(cn, fg.min(), fg.max(), color=col, linewidth=0.6)
            for vert, pl in zip(fg, pkeys):
                ax[1].vlines(vert, cn - 0.1, cn + 0.1, color=col, linewidth=0.6)
            vi = np.sort(fg)
            if vi.size > 2:
                vstack = np.stack((vi[:-1], vi[1:]), axis=1)
                editspots = np.diff(vstack) < subisomax
                if editspots.any():
                    ewheres = np.where(editspots)[0].tolist()
                    for ew in ewheres:
                        subpair = vstack[ew].tolist()
                        subints = [spectrum[i] for i in subpair]
                        winint = subints.index(max(subints))
                        winner = subpair[winint]
                        if ew > 0:
                            #edit 1 before ew
                            vstack[ew-1,1] = winner
                        if ew < len(vstack) - 1:
                            #edit 1 after ew
                            vstack[ew+1,0] = winner
                    vstack = np.delete(vstack, ewheres, axis=0)
            else:
                vstack = vi.reshape(1, -1)
            vdiffs = np.diff(vstack)
            vflat = sorted(vstack.flatten().tolist())
            labelspots = np.mean(vstack, axis=1).tolist()
            for ls, lp in zip(labelspots, vstack.tolist()):
                labeldiff = np.diff(lp)[0].round(4)
                chargedist = (proton/fc - labeldiff).round(4)
                lstring = ' ~ '.join((str(fc), str(labeldiff), str(chargedist)))
                if text:
                    ax[1].text(ls, cn - 0.2, lstring, fontsize=4, ha='center', color='white')

    ndmasses = regions[nodists,7]
    mdinds = np.logical_and(ndmasses >= zlmb, ndmasses <= zumb)
    ndtimes = regions[nodists,2:4]
    tdinds = np.logical_and(ndtimes.min(axis=1) >= zst, ndtimes.max(axis=1) <= zet)
    zdinds = regions[nodists,4] > 1
    finds = np.logical_and.reduce((mdinds, tdinds, zdinds))
    ndplotters = nodists[finds]
    if ndplotters.size > 0:
        ndmasses = regions[ndplotters,7].tolist()
        nints = regions[ndplotters,5].tolist()
        ax[0].bar(ndmasses, nints, alpha=0.5, color='white', width=0.01, label='N/A')
        if text:
            for fx, fy, nd in zip(ndmasses, nints, ndplotters):
                if fy > 0:
                    ax[0].text(fx, fy + fy * 0.03, str(nd), color='white', fontsize=4)
        for p in ndplotters:
            a = np.array(trackedgroups[p])
            ax[2].scatter(a[0], a[1], marker='.', color='white', s=0.3, alpha=0.3)
            if text:
                ax[2].text(a[0][-1], a[1][-1], str(regions[p,7].round(2)), color='white', fontsize=4)
    ax[0].set_yscale('log')
    ax[0].set_ylabel('intensity')
    ax[2].set_ylabel('minutes')
    ax[1].set_ylabel('distribution rank')
    ax[2].set_xlabel('m/z')
    for label in ax[2].get_xticklabels():
        #label.set_ha("right")
        label.set_rotation(-45)
    ncols = 6
    if text:
        ax[0].legend(title='Charge', loc='upper left', bbox_to_anchor=(0, 1.3 + (0.1 * cn / ncols)), ncol=ncols)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)
    plt.show()
    fig.clf()
    plt.close()




distributionmasses = {} #distid: ordered masses
distributioncharges = {} #distid: charge
distributionsbykeys = {} #keys: distid, to find keys without using mass or intensity
linesofdistributions = {} #distid: mass-ordered linedkeys
distributiontimelimits = {} #distid [starting rt, ending rt]
distributionintensities = {} #distid: mass-ordered intensities
distributionsbycharge = defaultdict(dict) #charge: dists: mass-ordered linekeys
for charge, dists in solodists.items():
    for dist, lines in dists.items():
        dmasses = regions[lines,7]
        lineorder = dmasses.argsort()
        sortedlines = np.array(lines)[lineorder]
        sortedmasses = regions[sortedlines,7]
        dintensities = regions[sortedlines,5]
        #you could rt match the top 2 I suppose
        rtlimits = regions[sortedlines,2:4]
        minrt = rtlimits.min()
        maxrt = rtlimits.max()
        #
        distributionmasses[dist] = sortedmasses
        distributioncharges[dist] = charge
        distributionsbykeys[tuple(sorted(sortedlines.tolist()))] = dist
        linesofdistributions[dist] = sortedlines
        distributiontimelimits[dist] = [minrt, maxrt]
        distributionintensities[dist] = dintensities
        distributionsbycharge[charge][dist] = sortedlines

#check for charge-state connections using whole groups
#sum their isotopomers at each time point and get a single area
#   ^allow linkage to groups with less isotopomers if their area is also less
#keep track of high / low intensity ratios for each adjacent pair
#overlap-check each line by RT
#initial matching by highest isotopomer? Then filter for things that contain the same number of isotopomers as what you're matching
#   ^only the things that can be directly matched will carry the charge-state priority, other things will match naturally I suppose.
#this will create a league of different priority levels that are generated from each additional charge state that's found.
#run the pairs here to be pairs of connections, if a pair is already present, skip it, only iterate upwards in charge then I suppose

#things to check:
#how charge state differences operate across different charges
#are RT's rather perfectly symmetrical across charge states? Like will a lower intensity charge state have all its RT's encompassed well enough to rely on? Then there's also the case of noisy channels - but this might be an easy y/n answer to whether or not there's noise!

#charge-state to charge-state competition can be based on RT overlap, ie that the right lines link to the right ones I suppose - whichever LINK is correct can be determined via the pair metrics already developed
#record expected mass diff across each mass and note difference/intensity
#note intensity ratio variation, is the ratio always expected to decrease at an isotopomer of lower intensity on a charge of lower intensity? Or is this more stochastic?
#note if RT's encompass
#show information about the ratio of intensities of one charge state to another below the individual plots

#make another kind of plot that log-scale shows the whole spectrum (x: m/z, y: intensity)? and show charge-state links as pointing lines up top?
#^you could also shrink the x-axis space between them and display an 'expected dist' alongside the original, but in a different color -> expected intensities too I suppose
#a 2nd plot below these would show RT overlaps (x: m/z, y: rt)

#serious ponderance, does a deuterated isotopomer tend to go towards a lower charge state? I keep seeing this RT shift on a weak signal that doesn't match a higher charge state, it's weird
#revolving 1 concept, most of these ratios, iso/iso, charge-state/charge-state, rt overlap, basdiffs?,should all essentially ~equate to a value of 1. If I take a rolling count of how far each of these are, independently, from 1, then sum/use the values the way I rank the pairs - that might be a decent ranking system for these. But it still doesn't provide me a way of cutting off bad matches.
#allow the charge-state process to break the rules of increasing/decreasing isotopomer intensities I suppose -> if it's there, the evidence is more legit than the 'model'

#^^^^^^^^^in regards to the ~slight~weird rt mismatch across RT's, I don't expect ionization to be a pefectly linear process, whereby ionizing more or less of something always shows the same (especially considering background ionizers) proportion of 2+ to 3+ etc, it might be a process that spawns some 3+ in greater quantity after a certain intensity/number of ions are reached. And this could easily cross over from the early to mid-early peak shape. I think this is a fair explanation. You could also reason this could an ordered generative process, whereby starting small and increasing [as the peak would] that you could generate differing numbers of either charge.

#so once a charge state is determined here, lower priority pairing can't add a more intense ion into the fray I suppose?

#I need to determine how I'm going to metricise the group errors and cross-charge %'s, do I connect to nearest mass/charge state? do I connect to nearest intensity? do I use all possible crosses to make a larger number of metrics that are simplified across means and more means?

t3 = time()

massranges = regions[:,:2]
maxmass = massranges.max() + 1
minmass = massranges.min() - 1
#reinvent this not to iterate over groupsize and check every match but to iterate over every group then check matches to only the desired charge state
#^make it a redundant search, it has to be so that everything from every perspective is covered
#^both positive and negative direction from the initial charge, do a wide search for ANY potential charge states within the full mass range above or below the distribution
#incorporate nodists into the lone edge search for a single ion of an appropriate charge
#^for now just do a +/-1 nodist to the max/min I suppose, see if any dists outside a 1 charge difference even match first
#for matching masses, do top 2, ONE of the top 2 of either should be the same
#^nah, use everything, then keep the argsort ranks of everything, and when say... a 2-length matches a 4-length, but the 2-length matches the 2 lowest ions, it would fail because the [0,1] of the 2-length wouldn't match the [2,3] of the 4-length via close order matching.

#top version is for collecting every possible charge state, use it later to see if there are any decent patterns, I didn't find any initially but I'd like to take a deeper look once I can
#groupid = 0
#chargegroups = defaultdict(set) #groupid: [connection keys]
##intersection_merge the set of key + values here to get finalized plotting groups?
#for charge, dists in distributionsbycharge.items():
#    for dkey, lines in dists.items():
#        dmasses = distributionmasses[dkey]
#        minrt, maxrt = distributiontimelimits[dkey]
#        basemasses = dmasses * charge - proton * charge
#        trial = charge + 1
#        chargetrials = []
#        pos = True
#        searching = True
#        boundaryfault = False
#        #finding all potential charges across full mass range
#        while searching:
#            scoutmasses = (basemasses + proton * trial) / trial
#            if scoutmasses.max() >= minmass:
#                if scoutmasses.min() <= maxmass:
#                    chargetrials.append(trial)
#                else:
#                    boundaryfault = True
#            else:
#                boundaryfault = True
#            if boundaryfault:
#                if pos:
#                    trial = charge
#                    pos = False
#                    boundaryfault = False
#                else:
#                    searching = False
#            if pos:
#                trial += 1
#            else:
#                trial -= 1
#            if trial < 1:
#                searching = False
#        for trial in chargetrials:
#            if trial in distributionsbycharge:
#                scoutmasses = (basemasses + proton * trial) / trial
#                expdiff = proton / trial
#                meandiff = np.diff(scoutmasses).mean()
#                ctol = (expdiff - meandiff)
#                for mkey, mlines in distributionsbycharge[trial].items():
#                    matchminrt, matchmaxrt = distributiontimelimits[mkey]
#                    if minrt < matchmaxrt and maxrt > matchminrt: #faster if using this first? - yea, a lil, needs more profiling though
#                        matchmasses = distributionmasses[mkey][:,None]
#                        if matchmasses.max() >= scoutmasses.min() and scoutmasses.max() > matchmasses.min(): #mass ranges overlap
#                            overpass = False
#                            if minrt > matchminrt and maxrt < matchmaxrt: #primary encompassed
#                                overpass = True
#                            elif matchminrt > minrt and matchmaxrt < maxrt: #secondary encompassed
#                                overpass = True
#                            else:
#                                overlap = min(matchmaxrt, maxrt) - max(matchminrt, minrt)
#                                fullrange = max(matchmaxrt, maxrt) - min(matchminrt, minrt)
#                                if (overlap * 2) / fullrange > 0.75:
#                                    overpass = True
#                            if overpass:
#                                matchmatrix = np.abs(scoutmasses - matchmasses) < ctol
#                                mmshape = matchmatrix.shape
#                                majoraxis = np.argmax(mmshape)
#                                minoraxis = np.argmin(mmshape)
#                                matrixmatches = matchmatrix.any(axis=majoraxis)
#                                if matrixmatches.sum() >= matrixmatches.size / 2: #matching at least half I suppose, maybe make this a 'majority' be removing the =?
#                                    chargegroups[groupid].add(mkey)
#                                    #
#        if groupid in chargegroups:
#            chargegroups[groupid].add(dkey)
#            groupid += 1

groupid = 0
chargegroups = defaultdict(set) #groupid: [connection keys]
#intersection_merge the set of key + values here to get finalized plotting groups?
for charge, dists in distributionsbycharge.items():
    for dkey, lines in dists.items():
        dmasses = distributionmasses[dkey]
        minrt, maxrt = distributiontimelimits[dkey]
        basemasses = dmasses * charge - proton * charge
        trial = charge + 1
        chargetrials = []
        pos = True
        searching = True
        boundaryfault = False
        #finding all potential charges across full mass range
        while searching:
            if trial in distributionsbycharge:
                scoutmasses = (basemasses + proton * trial) / trial
                if scoutmasses.max() >= minmass:
                    if scoutmasses.min() <= maxmass:
                        expdiff = proton / trial
                        meandiff = np.diff(scoutmasses).mean()
                        ctol = (expdiff - meandiff)
                        for mkey, mlines in distributionsbycharge[trial].items():
                            matchminrt, matchmaxrt = distributiontimelimits[mkey]
                            if minrt < matchmaxrt and maxrt > matchminrt: #faster if using this first? - yea, a lil, needs more profiling though
                                matchmasses = distributionmasses[mkey][:,None]
                                if matchmasses.max() >= scoutmasses.min() and scoutmasses.max() > matchmasses.min(): #mass ranges overlap
                                    overpass = False
                                    if minrt > matchminrt and maxrt < matchmaxrt: #primary encompassed
                                        overpass = True
                                    elif matchminrt > minrt and matchmaxrt < maxrt: #secondary encompassed
                                        overpass = True
                                    else:
                                        overlap = min(matchmaxrt, maxrt) - max(matchminrt, minrt)
                                        fullrange = max(matchmaxrt, maxrt) - min(matchminrt, minrt)
                                        if (overlap * 2) / fullrange > 0.75:
                                            overpass = True
                                    if overpass:
                                        matchmatrix = np.abs(scoutmasses - matchmasses) < ctol
                                        mmshape = matchmatrix.shape
                                        majoraxis = np.argmax(mmshape)
                                        minoraxis = np.argmin(mmshape)
                                        matrixmatches = matchmatrix.any(axis=majoraxis)
                                        if matrixmatches.sum() >= matrixmatches.size / 2: #matching at least half I suppose, maybe make this a 'majority' be removing the =?
                                            chargegroups[groupid].add(mkey)
                                            #
                    else:
                        boundaryfault = True
                else:
                    boundaryfault = True
            else:
                boundaryfault = True
            if boundaryfault:
                if pos:
                    trial = charge
                    pos = False
                    boundaryfault = False
                else:
                    searching = False
            if pos:
                trial += 1
            else:
                trial -= 1
            if trial < 1:
                searching = False
        if groupid in chargegroups:
            chargegroups[groupid].add(dkey)
            groupid += 1
        #implement nodist search here, collect max/min trials on the fly and work off of that


print(time() - t3, 'charge group linkage')
t4 = time()

#combining redundant matches
chargesets = intersection_merge(chargegroups.values())

#combinatorics will be based off of this, and this allows things that didn't connect downward to branch in that direction
chargelayers = defaultdict(lambda: defaultdict(set)) #groupid: charge: [connections]
for n, cs in enumerate(chargesets):
    for c in cs:
        chargelayers[n][connectioncharges[c]].add(c)

#I suppose I should just use ths smallest charge group's size as the basis for matching the others, but it bothers me kinda
#I suppose, if that smallest group's size fails to match things, then divide things into those that matched and those that didn't? Charge order will matter here
#doing the smallest one for now, and throwing anything that doesn't pass into a separate list to visualize the negatives
#there's also some regulatory disparity for matches that don't have smaller matches I suppose
fid = 0
chargecongroups = defaultdict(dict) #connection: [all chargegroups its involved in]
failedcongroups = defaultdict(dict)
for cl in chargelayers.values():
    for cons in itertools.product(*cl.values()):
        #close order matching here, giving the intensities some leeway in terms of ranking without allowing them to move around too much - is the idea
        basemasses = [distributionmasses[i]*distributioncharges[i]-proton*distributioncharges[i] for i in cons]
        intensities = [distributionintensities[i] for i in cons]
        matrixmin = min(i.size for i in intensities)
        sortedmasses = sorted(basemasses, key=lambda x: x.size)
        sortedintensities = sorted(intensities, key=lambda x: x.size)
        distorders = []
        for n, bm in enumerate(sortedmasses):
            if n > 0:
                orderind = np.abs(sortedmasses[0][0] - bm).argmin()
                distorders.append(orderind)
            else:
                distorders.append(0)
        #intensityranks = [i.argsort()[::-1] for i in sortedintensities]
        intensityranks = [i.argsort()[::-1].argsort() for i in sortedintensities]
        crossintensityranks = []
        for r, do in zip(intensityranks, distorders):
            crossintensityranks.append(r[do:do+matrixmin])
        crossintensityranks = np.array(crossintensityranks)
        conranges = np.ptp(crossintensityranks, axis=0)
        if conranges.sum() < conranges.size - 1: #close order switching
            #it ends up being a little janky, conceptually, because I can only metricize what can be matched across an axis
            #^so I need to probably extend this to mid-level axis that aren't the minimum but can match to more dists than is done here? I wonder if this would asymetrically hurt the intensity metric, I suppose for the mass metric it would be fine
            crossmasses = []
            crossintensities = []
            for i, m, do in zip(sortedintensities, sortedmasses, distorders):
                crossmasses.append(m[do:do+matrixmin])
                crossintensities.append(i[do:do+matrixmin])
            crossintensities = np.array(crossintensities)
            crossmasses = np.array(crossmasses)
            crossintensitysums = crossintensities.sum(axis=0)
            massmeandiff = np.abs(crossmasses.mean(axis=0) - crossmasses).mean()
            intensitypercs = crossintensities / crossintensitysums
            intensitymeandiff = np.abs(intensitypercs.mean(axis=1)[:,None] - intensitypercs).mean()
            for con in cons:
                #chargecongroups[con][cons] = concheck
                #concheck[cons].append([ccmeandiff, massdiff])
                chargecongroups[con][cons] = [intensitymeandiff, massmeandiff]
        else:
            for con in cons:
                charge = distributioncharges[con]
                failedcongroups[fid][charge] = con
            fid += 1
#these two metrics are imbalanced, (953, 29) is better than (950, 29) but it wouldn't win here, the mass error is what makes it obvious but the mass metric doesn't impact the overall thing at all

#both this process and it's look-alike below for the regular pairs are not so much a normalizing, it's more like a balancing.
prioritycharges = []
secondpriorities = []
for con, congroups in chargecongroups.items():
    if len(congroups) > 1:
        n1, n2 = np.array(list(congroups.values())).sum(axis=0)
        for congroup, (s1, s2) in congroups.items():
            newconscore = sum((s1/n1, s2/n2))
            prioritycharges.append([congroup, newconscore])
    else:
        congroup, conscore = list(congroups.items())[0]
        nconscore = conscore.copy()
        nconscore.insert(0, congroup)
        for con in congroup:
            secondpriorities.append([congroup, sum(conscore)])

rankedcharges = sorted(prioritycharges, key=lambda x: x[1]) #competition among matched chargegroups where matching can be done easily
secondprioritycharges = sorted(secondpriorities, key=lambda x: x[1]) #no competition happening here, the order doesn't actually matter
rankedcharges.extend(secondprioritycharges)
#^the connection groups take priority of the direction of what lines match to each other. There will still be a secondary pairing process for lines going on here, but each new connection group gained will have to have a superset of the existing lines for the connection group in question to grow.
#^but this has a problem with linking bad actors, there's no way to allow a subset cutoff
#I'm going to move forward accepting this for now, it could be modified via tweaking later
#^currently, it would rely on the robustness of the initial chargegroup selection to filter out poor distributions


#instead of iterating through the list at each entry, you could keep track of sets of lines made along the way, and only iterate until a superset is found - priority is given to those at the top of the list this way
#^yes but you need to do a second final sweep to make sure no existing subsets are still in the list, it's got to be free flowing, then eliminating for subsets
#^only create a blocking list for the second part, might not just be subsets but weird combinations of groups that aren't together under the prevailing logic

##what I have for the existing pair system:
#iterate pairs & score:
#    something passes the criteria for ranking:
#        orders come here
#iterate orders:
#    iterate orders again:
#        combine supersets
#
##what I propose above:
#make flylist
#iterate pairs & scores:
#    something passes the criteria for ranking:
#        list of sets kept on the fly:
#            - first check if the new passed pair is a sub/superset of anything in the existing flylist
#            - if not, its becomes its own entry
#            - if so -> merge, superset connections take priorities for charge group connection descriptions
#make blocked list
#iterate final superset list:
#    - block things already taken, in order, output should be the final list
#^this actually changes the outcome, and so would doing a post-priority blocking. You'd have to do an awkward post-priority subset removal. I also did try this previously for the pair matching process and it didn't work.


#~~~~
#multiple types of merging:
#superset merging to allow for larger distribution lengths to work out
#intersection merging to allow shorter distributions at different, faded, charges to still connect
#you can add more charge states
#things added at already possessed charge states must overlap with the existing line structure, and newer/previously unexposed lines can be added
#basically, add whatever you like as long as the existing line infrastructure at each charge is preserved
#^with some checks, the longest distribution shouldn't be on something less than the most intense I suppose
#free expansion as long as existing lines overlap
#^I'm going to skip checking that the rank of lengths equates to the rank of intensities, although this would probably be an appropriate check.


blocked = set()
preservedchargeranks = []
for group, score in rankedcharges:
    if group not in blocked:
        preservedchargeranks.append([group, score])
        blocked.add(group)
print(time() - t4, 'charge set and priority ranking')

t5 = time()
chargeid = 0
blocked = set()
chargedistlines = defaultdict(lambda: defaultdict(set)) #chargegroupid: charge: [lines]
chargedistgroups = defaultdict(dict) #chargegroupid: charge: distributionid
chargegroupsbyline = {} #line: chargegroupid, doubles as blocking list
chargesbyline = {} #line: charge
for pn, (cons, score) in enumerate(preservedchargeranks):
    #join anything that has a sorted tuple key to connectionsbykeys with the existing line infrastructure at each charge state
    #ckeys = set(itertools.chain(*(linesofdistributions[i] for i in cons)))
    if not any(i in blocked for i in cons):
        #^if none are blocked, none have been used -> make new group I suppose?
        grouplines = {}
        groupdists = {}
        for con in cons:
            charge = distributioncharges[con]
            distkeys = linesofdistributions[con]
            chargedistlines[chargeid][charge] = distkeys
            chargedistgroups[chargeid][charge] = con
            for line in distkeys:
                chargegroupsbyline[line] = chargeid
                chargesbyline[line] = charge
        blocked.update(cons)
        chargeid += 1
print(time() - t5, 'charge group assembly')


#sum lines of a distribution across time and take the area across charge states -> largest area should have the longest distribution -> if not, correct the longer less intense distributions to match the most intense one. This can be used as the future 'leader' to later decide whether distributions in the pairing process are allowed to expand.
#after pair-match charge states are assembled, you can look for the outer-fray lone lines that match major lines of a charge distribution

##for plotting distributions from an incomplete process
#shrinksets = defaultdict(set)
#for n, cg in enumerate(chargesets):
#    for g in cg:
#        shrinksets[n].update(connectionkeys[g])
#
#blocked = set()
#finalsets = {}
#for (k1, v1), (k2, v2) in itertools.combinations(shrinksets.items(), 2):
#    if v1.issuperset(v2):
#        blocked.add(k2)
#        if k2 in finalsets:
#            del finalsets[k2]
#        if k1 not in blocked:
#            finalsets[k1] = v1
#    elif v2.issuperset(v1):
#        blocked.add(k1)
#        if k1 in finalsets:
#            del finalsets[k1]
#        if k2 not in blocked:
#            finalsets[k2] = v2
#    elif v1 == v2:
#        if k1 not in blocked and k2 not in blocked:
#            finalsets[k1] = v1
#            if k2 in finalsets:
#                del finalsets[k2]
#        else:
#            blocked.add(k1)
#            blocked.add(k2)
#            if k2 in finalsets:
#                del finalsets[k2]
#            if k1 in finalsets:
#                del finalsets[k1]
#    else:
#        if k1 not in blocked:
#            finalsets[k1] = v1
#        if k2 not in blocked:
#            finalsets[k2] = v2
#
#plotsets = [chargesets[i] for i in finalsets]
#
#taxing = True
#for cg in plotsets:
#    cg = list(cg)
#    cg = sorted(cg, key=lambda x: connectionmainintensity[x])
#    cgcharges = [connectioncharges[i] for i in cg]
#    chargeorder = sorted(set(cgcharges))
#    chlen = len(chargeorder)
#    #if len(chargeorder) > 2:
#    #if len(cg) == chlen:
#    #if len(chargeorder) > 2:
#    if True:
#        cf = pd.DataFrame()
#        chargefigures = {c:n for n, c in enumerate(chargeorder)}
#        cgcons = [connections[i] for i in cg]
#        fig, ax = plt.subplots(ncols=chlen, nrows=5, figsize=(6,8), sharex='col', sharey='row')
#        nfig, nax = nfig, nax = plt.subplots(ncols=chlen, nrows=2, figsize=(7,4), sharey='row')
#        if taxing:
#            tfig, tax = plt.subplots(ncols=2, nrows=2, figsize=(7,4))
#        fig.subplots_adjust(hspace=0.05, wspace=0.05)
#        mainintensities = [connectionmainintensity[i] for i in cg]
#        maxmain = max(mainintensities)
#        tbarwidth = 1 / len(cg)
#        tspace = 0.5
#        cols = dp.get_colors(len(cg))
#        cmin = np.inf
#        cmax = 0
#        cratiolist = []
#        ccbounds = [np.inf, 0]
#        for n, g in enumerate(cg):
#            con = connections[g]
#            cratios = connectionratios[g]
#            cratiolist.append(cratios)
#            abcratios = [abs(i) for i in cratios]
#            cmasses = connectionmasses[g]
#            cintensities = connectionintensities[g]
#            if cintensities.min() < cmin:
#                cmin = cintensities.min()
#            if cintensities.max() > cmax:
#                cmax = cintensities.max()
#            concharge = connectioncharges[g]
#            expdiff = proton / concharge
#            basemasses = cmasses * concharge - proton * concharge
#            acdiffs = expdiff - np.diff(cmasses)
#            basediffs = acdiffs * concharge
#            conhax = chargefigures[concharge]
#            ckeys = list(set(itertools.chain(*con)))
#            cwidth = 0.5 *  len(ckeys)
#            nst = regions[ckeys,2].min()
#            net = regions[ckeys,3].max()
#            nlmb = regions[ckeys,0].min()
#            numb = regions[ckeys,1].max()
#            boundrec = [nlmb, numb, nst, net]
#            plotkeys = arg_coord_rectangle_overlap(boundrec, regions[:,:4]).tolist()
#            for p in plotkeys:
#                if p not in ckeys:
#                    a = trackedgroups[p]
#                    creg = regions[p]
#                    ax[1][conhax].plot(a[0], a[1], '.', color='white', alpha=0.2)
#                    ax[0][conhax].plot([creg[7], creg[7]], [0, creg[5]], '-', color='white', alpha=0.5, linewidth=cwidth)
#            ax[3][conhax].hlines(0, cmasses.min(), cmasses.max(), color='black', linewidth=0.3)
#            ax[4][conhax].hlines(0, cmasses.min(), cmasses.max(), color='black', linewidth=0.3)
#            for cline in ckeys:
#                creg = regions[cline]
#                ax[0][conhax].plot([creg[7], creg[7]], [0, creg[5]], '-', color=cols[n], alpha=1, linewidth=cwidth)
#                a = trackedgroups[cline]
#                ax[1][conhax].plot(a[0], a[1], '.', color=cols[n], alpha=0.2)
#            mainintensity = connectionmainintensity[g] / maxmain
#            basenorms = basediffs / proton
#            abasediffs = np.abs(basenorms / proton - basediffs)
#            if taxing:
#                tax[0][0].bar(concharge, mainintensity, tbarwidth*tspace, color=cols[n], alpha=0.5)
#                tax[0][1].bar([i+tbarwidth*concharge*tspace for i in range(len(cratios))], cratios, width=tbarwidth*tspace, color=cols[n], alpha=0.5)
#                tax[1][0].bar([i+tbarwidth*concharge*tspace for i in range(len(cratios))], basediffs, width=tbarwidth*tspace, color=cols[n], alpha=0.5)
#            diffgen = 0.05
#            for nc, cn in enumerate(cg):
#                cnmasses = connectionmasses[cn]
#                cnintensities = connectionintensities[cn]
#                cnratio = cintensities / cnintensities
#                cnratiobar = np.abs(cnratio.mean() - cnratio).mean()
#                if cnratiobar > ccbounds[1]:
#                    ccbounds[1] = cnratiobar
#                if cnratiobar < ccbounds[0] and cnratiobar > 0:
#                    ccbounds[0] = cnratiobar
#                cx = chargefigures[connectioncharges[cn]]
#                nax[0][conhax].bar(cx, cnratiobar, width=0.8, color=cols[nc], alpha=0.5)
#                cf.loc[g, cn] = cnratiobar
#                if nc >= n:
#                    ax[2][conhax].bar(cmasses+diffgen*nc, cnratio, width=diffgen/2, color=cols[nc], alpha=0.8)
#            bn = 0
#            bw = 0.02
#            for nc, cn in enumerate(cg):
#                if cn != cg:
#                    maincharge = connectioncharges[cn]
#                    mainmasses = connectionmasses[cn]
#                    mainbasemasses = mainmasses * maincharge - proton * maincharge
#                    maindiffs = mainbasemasses - basemasses
#                    mainppm = (maindiffs / mainbasemasses) * 1000000
#                    diffbar = np.abs(mainppm.mean() - mainppm).mean()
#                    cx = chargefigures[connectioncharges[cn]]
#                    nax[1][conhax].bar(cx, diffbar, width=0.8, color=cols[nc], alpha=0.5)
#                    ax[3][conhax].bar(cmasses+bn, maindiffs, width=bw, color=cols[nc], alpha=0.8)
#                    ax[4][conhax].bar(cmasses+bn, mainppm, width=bw, color=cols[nc], alpha=0.8)
#                    bn += bw + bw / 2
#            ax[0][conhax].set_title(''.join((str(concharge), '(', str(g), ')')))
#        if taxing:
#            cratiolist = np.array(cratiolist)
#            crmean = np.mean(cratiolist, axis=0)
#            crplot = np.abs(crmean - cratiolist).mean(axis=0)
#            tax[1][1].bar([i+tbarwidth*cratiolist.shape[1]*tspace for i in range(len(cratios))], crplot, width=tbarwidth, color='midnightblue', alpha=0.5)
#            tax[0][0].set_yscale('log')
#            tax[1][1].set_yscale('log')
#            tax[0][0].set_title('intensity ratios')
#            tax[0][1].set_title('isotopomer ratios')
#            tax[1][0].set_title('basediffs')
#            tax[1][1].set_title('isotopomer ratio meandiffs')
#        nax[0][0].set_yscale('log')
#        nax[0][0].set_ylim(ccbounds[0]/2, ccbounds[1])
#        #nax[0][1].set_yscale('log')
#        nax[0][0].set_ylabel('cross-charge meandiffs')
#        nax[1][0].set_ylabel('ppm meandiffs')
#        ax[0][0].set_yscale('log')
#        ax[0][0].set_ylim(cmin/2, cmax)
#        ax[0][0].set_ylabel('peak area')
#        ax[1][0].set_ylabel('retention time')
#        ax[2][0].set_ylabel('cross-charge %')
#        ax[3][0].set_ylabel('absolute error')
#        ax[4][0].set_ylabel('ppm error')
#        for ch, hax in chargefigures.items():
#            ax[-1][hax].tick_params(axis='x', labelrotation=-45)
#            if hax == 0:
#                #invisible right splines
#                ax[0][hax].spines.right.set_visible(False)
#                ax[1][hax].spines.right.set_visible(False)
#            elif hax == chlen-1:
#                #invisible left splines
#                ax[0][hax].spines.left.set_visible(False)
#                ax[1][hax].spines.left.set_visible(False)
#                for tick in ax[0][hax].yaxis.get_majorticklines():
#                    tick.set_visible(False)
#                for tick in ax[0][hax].yaxis.get_minorticklines():
#                    tick.set_visible(False)
#                for tick in ax[1][hax].yaxis.get_major_ticks():
#                    tick.tick1line.set_visible(False)
#                    tick.tick2line.set_visible(False)
#            else:
#                #left and right invisible
#                ax[0][hax].spines.right.set_visible(False)
#                ax[1][hax].spines.right.set_visible(False)
#                ax[0][hax].spines.left.set_visible(False)
#                ax[1][hax].spines.left.set_visible(False)
#                for tick in ax[0][hax].yaxis.get_majorticklines():
#                    tick.set_visible(False)
#                for tick in ax[0][hax].yaxis.get_minorticklines():
#                    tick.set_visible(False)
#                for tick in ax[1][hax].yaxis.get_major_ticks():
#                    tick.tick1line.set_visible(False)
#                    tick.tick2line.set_visible(False)
#        tfig.tight_layout()
#        nfig.tight_layout()
#        plt.show()
#        fig.clf()
#        tfig.clf()
#        nfig.clf()
#        plt.close()
#        gc.collect()

#charge-states of lesser intensity would need to be blocked, later on, from making a pair connection, that a charge-state of a higher intensity has not made

#now I need to visualize ratios, and charge distances!
#x: basemass
#y: intensity ratios
#color: charge

#horizontal bar between isotopomer intensity lines
#horizontal bar at twiny intensity ratio
#above bar, percent diff to expdiff
#below bar, absolute diff to expdiff

#basemass oriented bar chart, ordered by mainmass intensity, highest -> lowest for the sub-order? Though perhaps higher charge-states might show different behavior than lower ones when the highest intensity charge-state is in the middle somewhere.

#so, no, the intensity ratios aren't consistently ordered in any precise manner - they somewhat are, but they don't follow a strict ascent nor descent based on relative charge-state intensity

#can you perhaps explain the increase in isotopomer ratio from intensity fade via the ratio of isotopomers across charge states?
#^gotta visualize the cross charge-state isotopomer ratios as another potential avenue for a metric

#when there's a split match on the same distribution's connections, if the match with a higher intensity shows a smaller mass error, that's your winner

#I don't think the fade comes from analyzer dynamic range, or anything of the sort. I think that's an ionization phenomenon, as it always happens to the lowest intensity isotopomer of the lowest intensity charge state. Even when other charge states are at extremely low intensities for the analyzer, ie 1e5, the fade is happening only to the same exact isotopomer as you'd expect.

#I'll consider cross-charge-state error from the perspective of the most intense distribution. From there I can check whether mass error remains constant across each charge state and whether some distributions are actually a good match

#overall, the goal should be to pick which pairs to elevate, those can get plucked out of distributions that show decent matches. I think fade, even weird fade, can be allowed if the main isotopomer and it's best pair show good matching.
#all other categories of information should be used as a blocking mechanism to prevent the link from occuring.
#but I don't have a mechanism that allows a subset connection to overtake a superset should there be good reason.
#^that comes with the blocking. The lower priority (below charge-state priority) can allow extra pairs to be made so long as it's present in the highest intensity charge

#After the isotope process is done, when you're matching them to theoretical quantities, that fade may be an important feature there, it might be worth-while to determine which isotopes need that quantity corrected, and how much correction is being given. I'd wager you could systematically fix those errors while improving all fittings by either seeing if the faded quantity is 'stolen' by other charge states, or by assuming the lost/gained amount from the adjacent isotopomer ratios.
#adjacent isotopomer ratios can be used to determine which isotopomers of which charge state can be trusted as an example quantity for applying a correction by the cross-charge state ratio. This mechanism can also mark an isotopomer of a charge state as faded. If you can't determine that any of the adjacent ratios are good for a specific isotopomer, then you shouldn't accept that specific isotopomer for any of the distributions.
#^If this happens in the middle of a distribution, you'll have to split the ends I suppose. If it's in the middle for just one, cut off the rest of only that distribution and let the pieces compete at a lower priority.
#It might be worth looking for missing line-pieces on the faded isotopomers.
#in there case where there's 2 charge states and you have no idea of knowing if an isotopomer of either is a legit value, then you can just accept them as long as the masses and everything else lines up well. The order of intensities matters in this case.

#competetive processes can't be relied upon here the same way they do for basic iso distributions because there's not enough guaranteed competition over each line.
#consistency is key -> Find a conserved structure of ranks, this part must be perfect. Even if other isotopomers likely do match, they'll be left to match via the lower priority system.
#For competing charge-states, the average distance of the cross-charge %'s can be used in place of acdiff, but the acdiff goes for that distribution as opposed to any individual pairs. If a charge state with a super/sub-set of isotopomers gets outcompeted, it's not allowed to pass or connect. One distribution wins, and the contents of which are blocked.
#the second metric will be the same concept, except distance from average basemass error
#If a different charge state of a faded isotopomer can match the cross-charge %'s, that isotopomer should be accepted?
#Basically, if 2 charge states (potentially out of 3+) can agree on an isotopomer, a shittier one in a different charge state can be accepted.
#^So how would this work in the case where the most intense charge state is bullshit, but two lesser ones aren't?
#in the case of a noisy charge state, use a charge state with less competition to determine winning lines.

#summary:
#conserved structure of ranks - hard cutoff that decides what isotopomers gets to compete as a charge state. isotopomers that break ranks aren't included in any charge state - unless there's more than 1 charge state where it's consistent, then the third might be dragged along I suppose?
#^Allow for supersets to win this, then move on to the next step below for those supersets of which the ranks all play out.
#individual distributions compete via mean distance to mean charge state %'s, and mean distance to mean masss error
#mark known/visible fades by key in a fadedict of some sort

#After a brief, non-extensive search, I didn't see any reason to look for non-adjacent charge states, but this isn't the end of wondering if they're there.

#you might need to change the basemass-based connection, if two values are really close to being the basemass and if this switches back and forth across charges, you'd miss it

#seemed to have missed a line because of a slight switch in order of intensities, perhaps I should close order switching? Things can move up and down by like, 1
#^so subtract the argsorts and any absolute values > 1 would negate the process, however this might introduce some actual fuckery, I might want to make sure the actual intensity values are close in both distributions, maybe adjacent ratios here could play a part

#(1049, 494)
#^{2: [246586, 247679], 3: [246676, 248098]}
#(1205, 1126)
#^{2: [248787, 249475], 3: [249835, 251301]}
#(1111, 803), potential climb is too high? signal looks solid though
#^{4: [248824, 249347, 249797], 3: [249198, 249722, 250322]}

#bad match, coincidence:
#(2605, 1682) - I think the cross-charge % gives this away the strongest, its pattern also ~matches the ppm error pattern, the mass error isn't terrible

#needs close order switching:
#(2689, 2596)

#big mass errors:
#(2629, 87) -> has a better candidate that lost it seems, but that's not the only problem?
#(2615, 21) -> has a better candidate that lost? NO! It was just barely outside the massrange that I was slicing

#cross-charge % should have some % limit for a distance from the mean to be seem as a legit charge candidate
#(2327, 10), (2658, 2352) are good ~limits? it's a good match

#I need a way for longer distributions that have the intensity threshhold passed to swoop in and pick up things that have faded that didn't pass that threshhold -> the connection pairs are 100% always made so I just need a way to find it
#^so essentially, if the main distribution can find a key that matches the tuple(sorted(set)) of the keys, the I guess it's fine to link it to a lesser ranked charge state
#the close order switching isn't going to be straightforward, if the mainmasses switch, the distributions won't match -> I need the matching to be based on all masses rather than just the main
#I also want to exclude joining two 'different' distributions across charge states if their joining isotopomer differences are the most deviating from either of their means. It's a COINCIDENCE!
#^alternative strategy, via information processing later on: keep the distributions that don't really perfectly match glued together and search a distribution later on iteratively at it's multiple initiating sights aka adjacent isotopomer increases
#^both not dealing with the complicated bullshit upfront, and you can also make the assumption later on that they are two distributions if you decide you don't like the one. This helps deal with fade. Because otherwise... I don't really have a great way of dealing with fade. I would need massive overhead on the distribution linking process

#semi-side note, I'll need a cross-file validation scheme for MS1 predictions from one file that have MS2 scans in others. Or just a scheme to match MS2 identified peptides to their untargeted counterparts in other files.

#how well does the %'s from the sum of all isotopomers across charge states (adjacency ratio) match the adjacency ratios from the averaged charge state isotopomers? And which one matches theoretical distributions better on a large scale? Might be able to say something about the ionization events here.


#in this there are two clear bad matches, and one need for a type of close order switching
#forr the closeorderswitch match, you can make the initial match to either mainmass or monomass, that's pretty easy - and just do a logical_and to link the inds
#^and the idea of the closeorderswitch should be able to accomodate a specific number of switches based on the length I suppose
#rt mismatches are the cause of the bad matching ones, the whole distributions needs to be penalized for one bad rt match, all rt's need to be aligned under the fullmatch criteria i suppose
#although one of the bad rt matches should have been beaten out by a pair rather than the triplet that won...
#acceptable cross-charge %'s should always be either larger or smaller, never crossing, MAYBE this could only cross off a bad match when the mass error is the largest? or like 10x more than others?

#majority of isotopomers have a >= majority rt overlap of their cross charge state isotopomers in the event where something can be wrong, this can look at +/-1 charge state beyond to see if a match is there. If it is, accept the bad one in the middle. "Bad match" in this case would be something where the RTs/intensities are out of wack but the mass is good.

#TEST LATER:
#I see an interesting phenomenon, where when the intensities of the main masses are really close to 1:1, the next adjacent isotopomers of the charge state with the less intense main mass tend to be higher than those of the charge state with the main mass. And I'm wondering if this can be an observable ionization phenomenon that somehow depends on mass. And whether I can deduce that a distribution has subisotopomers to the left or right of their majors perhaps?
#(2222, 104) and (2625, 1203) are examples

#look into:
#(25555, 118), I'm not sure that 4-group should have beat the 2

#kicking something out of a charge state later on -> if a distribution wants to swoop in and claim some already claimed territory, and it offers both better alignment + a longer distribution to which the charge state can't compete AND there isn't a longer charge state chain to stand up for it, -> remove it I suppose!

#if the largest mass error and the largest rt deviation go together -> dump it I suppose, but how to determine if the largest is acceptable or not? It should also be a ~lower intensity and on the end of the distribution I suppose.
#rt overlap should basically be put into the metric process as a mean distance from the mean thing

#pretty reasonable observation:
#<1, more intense ions should have large cross-charge ratios, for ratios >1, less intense ions should have larger cross-charge ratioss
#^USUALLY, but not always, perhaps this would be a good normalizing metric to determine how a particular charge state or distribution might have been suppressed?
#For example, if the expected ratios switch, then normalize the more intense ion upwards by assigning it a greater intensity than measured? rather than normalizing the smaller one upwards.
#this adds some more justification to the close order switching process too, it seems like it's just another area where the data is extremely fragile.
#I'm also seeing larger adjacency ratios on less intense charge states - I think, investigate this more

#(3234, 1632) don't belong together because 3234 has a better distribution to be a part of that doesn't match

#the kickout process can be done after pair matching and will essentially check the close order matching of things all over again to determine if things still deserve to be linked, like a less intense distribution with more legs -> doesn't match the higher
#^or a lower intensity distribution that doesn't really match the expanded higher intensity one -> the orders are out of wack compared to how it should have faded

#a bad pattern is like (3236, 16) where the rank orders differ, and the cross charge percentages overlap -> this should be an auto-no, but how?

#new metrics:
#cross-charge order, for both intensities and dpoints
# - no need for negatives/flipped ratios, use raw ones - same effect
# - each isotopomer is normalized to the sum of that isotopomer at every charge state
# - then the you can take mean difference to mean at each isotopomer and therefore the metric will judge whether the isotopomer is a good addition
#currently, for ppm: the errors are averaged across a distribution but they should be averaged across isotopomers -> then the mean difference to this is the value to them mean across the other isotopomers -> normalize by number of isotopomers
# - average across basemass by isotopomer, then take distance to that average

#probems are stemming from the superset merging in the final ranking, bad things don't get poofed away
#looks like i'll need to allow </> 0 acdiffs or whatever, they happen

#I want to visualize how the charge distance scales back up to the ~proton length for each distribution across charge states
#I'll also put in the intensity/sum intensity across isotopomer plots
#I also want to visualize RT centering for each individual distribution, might be worth taking a ratio somewhere here
#basemass distance from average too

#check for charge distance:
#(3998, 3803) -> meh, pairmatching will fix it
#(3979, 3546, 1894) -> i want the new visualizations above
#(3966, 3185) -> their other connections don't seem to match that well because the only matches of the correct size have offset isoopomers. I'll need to look into more

#implement that each overlapping RT must have 75% fullrange, if they don't match here but do so later on via pair matching, so be it, but there's too many cases here of RT's being off for individual straggler ions, and this would be a decent place to cut off.
#for now, things that aren't distributions, yet that match perfectly, can match. For example (2870, 31) has a great match but totally isn't a distribution, they each clearly have a good distribution of their own extending in opposite directions -> when pair matching finds those pairs, you can excommunicate these two. I suppose a third charge state would override something like this? I'm not sure.
#^the ensuing matches would need to NOT have the strangest charge state distance
#can you regard fade the same at higher masses as lower? ie at lower charge states and at high? I'm not sure they act the same way

#current plan:
#re-impement activedirection, pairmatch FIRST, THEN allow for charge linkages afterwards, it seems backwards but it also seems like it would work better. The pairmatching scheme is HOT on accuracy, allowing this to take the reins beforehand would allow for much better matches
#^PLUS I can allow subset charge pairing for dists that don't form a connection coalition to pair individually pair across more than one charge state to get succesfully get the full superset link
#the resulting distribution matches can be based PURELY on the isotopomers that match, while only holding in consideration that more intense distributions should be ~longer
#^so you really only need to match the 2 highest isoptopomers, which will obviously be right next to each other, and then you can expand outwards I suppose. As any match that can't at least agree on the top 2 isn't going to be a legit match I think. The +/-1 charge state search to aid in seeing if one of them is just fucked up would help here too.
#when matching distributions, neither the mainmass or firstmass ideas are going to be able to fly alone: leader masses -> anything within 90% of the highest intensity mass? that way you can match distributions if the main mass switches across charge state, distribution structures can be filtered afterwards.
plotdicts = {
        'pass': chargedistgroups,
        'fail': failedcongroups,
        }

#(2702, 62), a bad match in both pairmatches
#^but is that (2760, 569)?

for status, distgroups in plotdicts.items():
    for distkey, dists in distgroups.items():
        chargeorder = sorted(dists)
        chlen = len(dists)
        cf = pd.DataFrame()
        chargefigures = {c:n for n, c in enumerate(chargeorder)}
        fig, ax = plt.subplots(ncols=chlen, nrows=8, figsize=(6,8), sharex='col', sharey='row')
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        cg = list(dists.values())
        chargedistbounds = [np.inf, 0]
        intensitylineup = [distributionintensities[g] for g in cg]
        flatintensities = itertools.chain(*intensitylineup)
        maxmain = max(flatintensities)
        masslineup = [distributionmasses[g]*distributioncharges[g]-proton*distributioncharges[g] for g in cg]
        masslineup = sorted(masslineup, key=lambda x: -x.size)
        intensitylineup = sorted(intensitylineup, key=lambda x: -x.size)
        arraysizes = [i.size for i in masslineup]
        if len(set(arraysizes)) == 1:
            arraymeans = np.array(masslineup).mean(axis=0)
            intensitysums = np.array(intensitylineup).sum(axis=0)
        else:
            matrixmax = max(arraysizes)
            arraysums = np.zeros(matrixmax)
            arraydividends = np.zeros(matrixmax)
            intensitysums = np.zeros(matrixmax)
            for n, (a, i) in enumerate(zip(masslineup, intensitylineup)):
                orderind = np.abs(masslineup[0] - a[0]).argmin()
                asize = a.size
                if asize == matrixmax:
                    arraysums += a
                    arraydividends += 1
                    intensitysums += i
                else:
                    arraysums[orderind:orderind+asize] += a
                    arraydividends[orderind:orderind+asize] += 1
                    intensitysums[orderind:orderind+asize] += i
            arraymeans = arraysums / arraydividends
        tbarwidth = 1 / len(cg)
        tspace = 0.5
        cols = dp.get_colors(len(cg))
        cmin = np.inf
        cmax = 0
        cratiolist = []
        ccbounds = [np.inf, 0]
        for n, (charge, g) in enumerate(dists.items()):
            lines = linesofdistributions[g]
            cintensities = distributionintensities[g]
            cratios = cintensities[:-1] / cintensities[1:]
            cratios[cratios < 1] = -1 / cratios[cratios < 1]
            cratiolist.append(cratios)
            abcratios = [abs(i) for i in cratios]
            cmasses = distributionmasses[g]
            if cintensities.min() < cmin:
                cmin = cintensities.min()
            if cintensities.max() > cmax:
                cmax = cintensities.max()
            concharge = distributioncharges[g]
            expdiff = proton / concharge
            basemasses = cmasses * concharge - proton * concharge
            if basemasses.size > arraymeans.size:
                orderind = np.abs(basemasses - arraymeans[0]).argmin()
                matrixmin = arraymeans.size
                meanbasediff = arraymeans - basemasses[orderind:orderind+matrixmin]
                meanbaseppms = (meanbasediff / basemasses[orderind:orderind+matrixmin]) * 1000000
                cmx = cmasses[orderind:orderind+matrixmin]
            else:
                orderind = np.abs(arraymeans - basemasses[0]).argmin()
                matrixmin = basemasses.size
                meanbasediff = arraymeans[orderind:orderind+matrixmin] - basemasses
                meanbaseppms = (meanbasediff / basemasses) * 1000000
                cmx = cmasses
            acdiffs = expdiff - np.diff(cmasses)
            basediffs = acdiffs * concharge
            conhax = chargefigures[concharge]
            cwidth = 0.5 *  len(lines)
            chargelengthextra = proton / charge * 2
            nst = regions[lines,2].min() - 0.5
            net = regions[lines,3].max() + 0.5
            nlmb = regions[lines,0].min() - chargelengthextra
            numb = regions[lines,1].max() + chargelengthextra
            nboundrec = [nlmb, numb, nst, net]
            nplotkeys = arg_coord_rectangle_overlap(nboundrec, regions[:,:4]).tolist()
            for p in nplotkeys:
                if p not in lines:
                    a = trackedgroups[p]
                    creg = regions[p]
                    ax[0][conhax].plot(a[0], a[1], '.', color='white', markersize=0.8, alpha=0.3)
                    ax[0][conhax].plot(a[0], a[1], '-', color='white', linewidth=0.4, alpha=1)
                    ax[1][conhax].plot([creg[7], creg[7]], [0, creg[5]], '-', color='white', alpha=0.5, linewidth=cwidth)
            ax[5][conhax].hlines(0, cmasses.min(), cmasses.max(), color='black', linewidth=0.3)
            ax[7][conhax].hlines(0, cmasses.min(), cmasses.max(), color='black', linewidth=0.3)
            ax[2][conhax].hlines(proton, cmasses.min(), cmasses.max(), color='black', linewidth=0.3)
            for cline in lines:
                creg = regions[cline]
                ax[1][conhax].plot([creg[7], creg[7]], [0, creg[5]], '-', color=cols[n], alpha=1, linewidth=cwidth)
                a = trackedgroups[cline]
                ax[0][conhax].plot(a[0], a[1], '.', color=cols[n], markersize=0.8, alpha=0.3)
                ax[0][conhax].plot(a[0], a[1], '-', color=cols[n], linewidth=0.4, alpha=1)
                #ax[2][conhax].plot([creg[7], creg[7]], [0, creg[4]], '-', color=cols[n], alpha=1, linewidth=cwidth)
            basenorms = basediffs / proton
            abasediffs = np.abs(basenorms / proton - basediffs)
            diffgen = 0.05
            bn = 0
            bw = 0.02
            adjacentx = cmasses[:-1] + np.diff(cmasses) / 2
            ax[5][conhax].bar(adjacentx, cratios, width=diffgen, color=cols[n], alpha=1)
            #ax[6][conhax].bar(adjacentx, cpointratios, width=diffgen, color=cols[n], alpha=1)
            chargedists = np.diff(cmasses) * charge
            ax[2][conhax].bar(adjacentx, chargedists, width=diffgen, color=cols[n], alpha=1)
            intensitypercs = cintensities[:intensitysums.size] / intensitysums[:cintensities.size]
            for ip in chargedists:
                if ip < chargedistbounds[0]:
                    chargedistbounds[0] = ip
                if ip > chargedistbounds[1]:
                    chargedistbounds[1] = ip
            ax[4][conhax].bar(cmasses, intensitypercs, width=diffgen, color=cols[n], alpha=1)
            ax[6][conhax].bar(cmx, meanbaseppms, width=diffgen, color=cols[n], alpha=1)
            for nc, cn in enumerate(cg):
                cnmasses = distributionmasses[cn]
                cncharge = distributioncharges[cn]
                cnbases = cnmasses * cncharge - proton * cncharge
                cnintensities = distributionintensities[cn]
                if basemasses.size > cnbases.size:
                    orderind = np.abs(basemasses - cnbases[0]).argmin()
                    matrixmin = cnbases.size
                    cnratio = cnintensities / cintensities[orderind:orderind+matrixmin]
                    bx = cmasses[orderind:orderind+matrixmin]+(diffgen*nc)
                else:
                    orderind = np.abs(cnbases - basemasses[0]).argmin()
                    matrixmin = basemasses.size
                    cnratio = cnintensities[orderind:orderind+matrixmin] / cintensities
                    bx = cmasses+(diffgen*nc)
                cnratiobar = np.abs(cnratio.mean() - cnratio).mean()
                if cnratiobar > ccbounds[1]:
                    ccbounds[1] = cnratiobar
                if cnratiobar < ccbounds[0] and cnratiobar > 0:
                    ccbounds[0] = cnratiobar
                cx = chargefigures[distributioncharges[cn]]
                cf.loc[g, cn] = cnratiobar
                ax[3][conhax].bar(bx, cnratio, width=diffgen, color=cols[nc], alpha=1)
                #ax[4][conhax].bar(bx, pointratio, width=diffgen, color=cols[nc], alpha=1)
            for nc, cn in enumerate(cg):
                if cn != cg:
                    maincharge = distributioncharges[cn]
                    mainmasses = distributionmasses[cn]
                    mainbasemasses = mainmasses * maincharge - proton * maincharge
                    if basemasses.size > mainbasemasses.size:
                        orderind = np.abs(basemasses - mainbasemasses[0]).argmin()
                        matrixmin = mainbasemasses.size
                        maindiffs = mainbasemasses - basemasses[orderind:orderind+matrixmin]
                        mainppm = (maindiffs / mainbasemasses) * 1000000
                        bx = cmasses[orderind:orderind+matrixmin]+(diffgen*nc)
                    else:
                        orderind = np.abs(mainbasemasses - basemasses[0]).argmin()
                        matrixmin = basemasses.size
                        maindiffs = mainbasemasses[orderind:orderind+matrixmin] - basemasses
                        mainppm = (maindiffs / mainbasemasses[orderind:orderind+matrixmin]) * 1000000
                        bx = cmasses+(diffgen*nc)
                    diffbar = np.abs(mainppm.mean() - mainppm).mean()
                    cx = chargefigures[distributioncharges[cn]]
                    #ax[5][conhax].bar(cmasses+bn, maindiffs, width=bw, color=cols[nc], alpha=1)
                    ax[7][conhax].bar(bx+bn, mainppm, width=bw, color=cols[nc], alpha=1)
                    bn += bw + bw / 2
            ax[0][conhax].set_title(''.join((str(concharge), '(', str(g), ')')), fontsize=12)
        ax[1][0].set_yscale('log')
        #ax[2][0].set_yscale('log')
        ax[2][0].set_ylim(chargedistbounds[0]*0.95, chargedistbounds[1]*1.05)
        ax[3][0].set_yscale('log')
        ax[5][0].set_yscale('symlog')
        ax[4][0].set_yscale('log')
        #ax[6][0].set_yscale('symlog')
        ax[1][0].set_ylim(cmin/2, cmax)
        ax[1][0].set_ylabel('peak area')
        ax[0][0].set_ylabel('retention time', fontsize=6)
        ax[3][0].set_ylabel('cross-charge', fontsize=7)
        ax[5][0].set_ylabel('adjacency')
        ax[7][0].set_ylabel('ppm error')
        ax[2][0].set_ylabel('charge distances', fontsize=6)
        ax[4][0].set_ylabel('intensity sum %', fontsize=6)
        ax[6][0].set_ylabel('ppm to mean', fontsize=7)
        for ch, hax in chargefigures.items():
            ax[-1][hax].tick_params(axis='x', labelrotation=-45)
            if hax == 0:
                #invisible right splines
                ax[0][hax].spines.right.set_visible(False)
                ax[1][hax].spines.right.set_visible(False)
            elif hax == chlen-1:
                #invisible left splines
                ax[0][hax].spines.left.set_visible(False)
                ax[1][hax].spines.left.set_visible(False)
                for tick in ax[0][hax].yaxis.get_major_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
                for tick in ax[1][hax].yaxis.get_majorticklines():
                    tick.set_visible(False)
                for tick in ax[1][hax].yaxis.get_minorticklines():
                    tick.set_visible(False)
            else:
                #left and right invisible
                ax[0][hax].spines.left.set_visible(False)
                ax[0][hax].spines.right.set_visible(False)
                ax[1][hax].spines.left.set_visible(False)
                ax[1][hax].spines.right.set_visible(False)
                for tick in ax[0][hax].yaxis.get_major_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
                for tick in ax[1][hax].yaxis.get_majorticklines():
                    tick.set_visible(False)
                for tick in ax[1][hax].yaxis.get_minorticklines():
                    tick.set_visible(False)
        supt = ': '.join((str(distkey), status))
        plt.suptitle(supt, y=0.92)
        plt.show()
        fig.clf()
        plt.close()
        gc.collect()




