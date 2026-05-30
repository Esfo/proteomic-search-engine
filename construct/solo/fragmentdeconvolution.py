from generalfunctions import intersection_merge
from generalfunctions import boundary_stack
from elementalcomponents import proton

import numpy as np
from pyteomics import mzml
from time import time
import multiprocessing as mp
from collections import defaultdict
from scipy import spatial
from functools import partial
from itertools import chain, product
import pickle

def charge_deconvolution(steplimit, newinclimit, chargetolerance, maxchargeofscans, scan):
    if scan['ms level'] != 2:
        return
    try:
        maxcharge = maxchargeofscans[scan['index']]
    except KeyError: #nodists and others
        maxcharge = 2
    precursorinfo = scan['precursorList']['precursor'][0]
    selectionwindow = precursorinfo['isolationWindow']
    precmass = selectionwindow['isolation window target m/z'].real
    
    masses = scan['m/z array']
    intensities = scan['intensity array']
    regiter = np.stack((masses, intensities), axis=1) #[mass, intensity]
    
    uppermasslimit = precmass * maxcharge - proton * maxcharge
    
    pairkeys = {} #pairkey: pair
    previousdecrease = {} #connectionindex: True, exists if something is increasing
    
    di = 0 #connectionindexes, keeping this incorrect plural makes it searchable
    paircharges = {} #connection: charge
    scoresbypair = {} #pair: [[absacdiff, datapercdiff, rtoffset],]
    pairsbyline = defaultdict(list) #mass: [pairs]
    
    si = 0 #subisokeys
    subisomasses = {} #mass: subisogroup
    subisogroups = defaultdict(lambda: defaultdict(list)) #subiso group: max charge for mass: [masses]
    
    #masswidthlimit = roundcutoff * 2 #this is translating the parameters across MS1/MS2 which I dislike, but I'm going to use it here for now
    #GET RID OF THIS
    masswidthlimit = proton + (proton * chargetolerance)
    
    connectionspine = defaultdict(list) #connectionindex: [pairkeys]
    latestconnections = defaultdict(lambda: defaultdict(list)) #masskey: charge: [current connection indices that end in masskey], a mass is always active as long as it's in masspool, this keeps track of all the groups that end in any given mass
    latestmass = {} #connectionindex: latest masskey
    
    pi = 0 #pairkeys
    masspool = []
    for nkey, (nm, nintensity) in enumerate(regiter.tolist()):
        masspoolremovals = []
        for okey in masspool:
            om, ointensity = regiter[okey]
            diff = nm - om
            if diff <= masswidthlimit:
                #sometimes a charge rounds as one thing but the value is way closer to +/-1
                initialcharge = round(proton / diff)
                if initialcharge > 1: #nothing close enough to care about for 1, plus the first bit would go to 0 and cause annoying zerodivision warnings
                    chargespread = np.linspace(initialcharge - 1, initialcharge + 1, 3)
                    expspread = proton / chargespread
                    minexpind = np.abs(diff - expspread).argmin()
                    charge = int(chargespread[minexpind])
                else:
                    charge = initialcharge
                if charge <= maxcharge:
                    actualmass = om * charge - proton * charge
                    if actualmass <= uppermasslimit:
                        expdiff = proton / charge
                        #acdiff = expdiff - diff
                        acdiff = abs(expdiff - diff)
                        diffcut = expdiff * chargetolerance
                        #if acdiff > -1 * (diffcut * chargetolerance + masswidthlimit):
                        #^a lighter acceptance for distances above the charge dist - it does happen from time to time on lower-hanging straggler isotopomers
                        if acdiff <= diffcut:
                            #absacdiff = abs(acdiff) * charge #normalizing -> distance to proton
                            absacdiff = acdiff * charge #normalizing -> distance to proton
                            ncons = 0
                            intensitypercdiff = abs(nintensity - ointensity) / (nintensity + ointensity) / 2
                            subcheck = False
                            if okey in latestconnections:
                                if charge in latestconnections[okey]:
                                    subcheck = True
                            decreasecheck = nintensity < ointensity
                            if subcheck:
                                csubs = latestconnections[okey][charge].copy() #need copy here?
                                for adi in csubs:
                                    ratiocheck = steplimit
                                    if adi in previousdecrease:
                                        if not decreasecheck:
                                            #intensity is increasing
                                            ratiocheck = newinclimit
                                    if intensitypercdiff <= ratiocheck:
                                        if latestmass[adi] == okey:
                                            xdi = adi
                                        else:
                                            xdi = di
                                            opairs = pairsbyline[okey]
                                            spinecopy = connectionspine[adi].copy()
                                            spinds = []
                                            for op in opairs:
                                                if op in spinecopy:
                                                    spinds.append(spinecopy.index(op))
                                            spi = max(spinds)
                                            spinecopy = spinecopy[:spi]
                                            connectionspine[xdi] = spinecopy
                                            di += 1
                                        connectionspine[xdi].append(pi)
                                        latestmass[xdi] = nkey
                                        latestconnections[nkey][charge].append(xdi)
                                        if decreasecheck:
                                            previousdecrease[xdi] = True
                                        ncons += 1
                            else:
                                #no previous subgroup 
                                if intensitypercdiff <= steplimit:
                                    connectionspine[di].append(pi)
                                    latestmass[di] = nkey
                                    latestconnections[nkey][charge].append(di)
                                    if decreasecheck:
                                        previousdecrease[di] = True
                                    ncons += 1
                                    di += 1
                            if ncons > 0:
                                dpercdiff = abs(nintensity - ointensity) / (nintensity + ointensity) / 2
                                lpair = (okey, nkey)
                                pairkeys[pi] = lpair
                                paircharges[pi] = charge
                                scorelist = [absacdiff, dpercdiff, ~decreasecheck]
                                for p in lpair:
                                    pairsbyline[p].append(pi)
                                scoresbypair[pi] = scorelist
                                if subcheck:
                                    connectionspine[di].append(pi)
                                    latestmass[di] = nkey
                                    latestconnections[nkey][charge].append(di)
                                    if decreasecheck:
                                        previousdecrease[di] = True
                                    di += 1
                                pi += 1
            else:
                #om is past proton distance, remove om from mass pool
                masspoolremovals.append(okey)
        for mpr in masspoolremovals:
            masspool.remove(mpr)
        masspool.append(nkey)
    
    flatdistgroups = set()

    distributionscoresbyline = defaultdict(list) #linekey: [pairkeys] -> without a set there is redundancy in here that disrupts downstream
    distributionscoredict = defaultdict(list) #pairkey: [[scores],]
    for distkey, pairspine in connectionspine.items():
        activepairlist = []
        scores = []
        #for pairs in pairspine.tolist():
        for pairs in pairspine:
            activepairlist.append(pairs)
            scores.append(scoresbypair[pairs])
            flatdist = []
            competinglines = set()
            for pair in activepairlist:
                pk = pairkeys[pair]
                flatdist.extend(pk)
                for p in pk:
                    if set(pairsbyline[p]).difference(activepairlist):
                        competinglines.add(p)
            activelines = list(set(chain(*(pairkeys[i] for i in activepairlist))))
            #rbounds = regions[activelines,2:4]
            rbounds = regiter[activelines,2:4]
            rstack = boundary_stack(rbounds)
            if rstack > 0:
                flatdistgroups.add(tuple(sorted(set(flatdist))))
                if competinglines and len(activepairlist) > 1:
                    unstackedboundaries = rbounds - rbounds.min(axis=1)[:,None]
                    ustack = boundary_stack(unstackedboundaries)
                    scorearray = np.array(scores)
                    distmean = scorearray[:,0].mean()
                    #rtmultiplier = scorearray[:,2].prod()
                    rtmultiplier = rstack / ustack
                    decreasingmultiplier = scorearray[:,3].sum() + 1
                    slen = len(scorearray)
                    for pair, score in zip(activepairlist, scores): 
                        if paircharges[pair] > 1: # a lot of bad 1+ matches get high priority from this, I essentially want less 1+ than 3+ and this helps
                            #dist, ddiff, rtoffset, decs = score
                            dist, ddiff, decs = score
                            meandiff = abs(distmean - dist) / slen
                            distdiff = meandiff * (2**decreasingmultiplier)
                            datadiff = ddiff / rtmultiplier
                            scorelist = tuple([distdiff, datadiff])
                            distributionscoredict[pair].append(scorelist)
                            for p in pairkeys[pair]:
                                if p in competinglines:
                                    if not pair in distributionscoresbyline[p]: #avoiding set use to save memory
                                        distributionscoresbyline[p].append(pair)
    
    preservedpairs = set()
    #lines being comparison-ranked across all individual distribution scores it participates in, if it has multiple
    rankedpairs = [] #[pair, minval]
    for line, pairs in distributionscoresbyline.items():
        vals = []
        pairexpansion = []
        for pair in pairs:
            pairscores = distributionscoredict[pair]
            vals.extend(pairscores)
            pairexpansion.extend([pair for _ in range(len(pairscores))])
        vals = np.array(vals)
        vsums = vals.sum(axis=0)
        vsums[vsums == 0] = 1 #0-sums make nans but this doesn't change the final answer of 0
        vpercs = vals / vsums
        sumvals = vpercs.sum(axis=1)
        rankedpairs.extend(list(zip(pairexpansion, sumvals.tolist())))
        preservedpairs.update(pairs)

    secondpriorities = [] #[pair, score]
    thirdpriorities = [] #[pair, score]
    for line, pairs in pairsbyline.items():
        #pairs = pairs.tolist()
        pairs = pairs
        plen = len(pairs)
        if plen > 1:
            scores = [scoresbypair[i] for i in pairs]
            scorearray = np.array(scores)
            scorearray[:,2] += 1
            scoresums = scorearray.sum(axis=0)
            scoresums[scoresums == 0] = 1 #0-sums make nans but this doesn't change the final answer of 0
            normscores = scorearray / scoresums
            #offsetnorms = normscores[:,:2] * scorearray[:,3,None] / scorearray[:,2,None]
            offsetnorms = normscores[:,:2] * scorearray[:,2,None]
            for pair, score in zip(pairs, offsetnorms.sum(axis=1).tolist()):
                if pair not in preservedpairs: #memory saving
                    secondpriorities.append([pair, score])
        else:
            pairs = pairs[0]
            if pairs not in preservedpairs:
                #dist, ddiff, rtoffset, dec = scoresbypair[pairs]
                dist, ddiff, dec = scoresbypair[pairs]
                dec += 1 #adds 1 to things that dec'd and makes non-dec's 1 -> no change
                equalizednorm = abs(ddiff - dist)
                #outscore = equalizednorm * dec / rtoffset
                outscore = equalizednorm * dec
                thirdpriorities.append([pairs, outscore])

    firstranks = sorted(rankedpairs, key=lambda x: x[1])
    secondranks = sorted(secondpriorities, key=lambda x: x[1])
    thirdranks = sorted(thirdpriorities, key=lambda x: x[1])

    sortedranks = []
    sortedranks.extend(firstranks)
    sortedranks.extend(secondranks)
    sortedranks.extend(thirdranks)

    preservedpairs = set()
    preservedranks = []
    for pairkey, score in sortedranks:
        if pairkey not in preservedpairs:
            preservedranks.append([pairkeys[pairkey], paircharges[pairkey]])
            preservedpairs.add(pair)

    sn = 0
    distsets = defaultdict(set) #distloc: [distributions]
    linelocations = {} #masskey: distloc
    setcharges = defaultdict(set) #index of distsets: [charges]
    for pair, paircharge in preservedranks:
        pairset = set(pair)
        locs = set()
        for line in pairset:
            if line in linelocations:
                locs.add(linelocations[line])
        if locs:
            joiner = min(locs)
            if len(locs) > 1:
                dist = pairset.copy()
                for l in locs:
                    dist.update(distsets[l])
                if tuple(sorted(dist)) in flatdistgroups:
                    for l in locs.difference([joiner]):
                        for line in distsets[l]:
                            linelocations[line] = joiner
                        setcharges[joiner].update(setcharges.pop(l))
                        distsets[joiner].update(distsets.pop(l))
                    distsets[joiner].update(pairset)
                    for line in pairset:
                        linelocations[line] = joiner
            else:
                if tuple(sorted(distsets[joiner].union(pairset))) in flatdistgroups:
                    distsets[joiner].update(pairset)
                    setcharges[joiner].add(paircharge)
                    for line in pairset:
                        linelocations[line] = joiner
        else:
            joiner = sn
            distsets[joiner].update(pairset)
            setcharges[joiner].add(paircharge)
            for line in pairset:
                linelocations[line] = joiner
            sn += 1
    
    dr =  0 #easier to work with dr here rather than re-index the finalized distribution just to get the max index for finaldefiniteind afterwards, the order of these aren't important
    solodists = defaultdict(dict) #charge: distid: lines
    for distindex, dist in distsets.items():
        charges = setcharges[distindex]
        if dist:
            charge = max(charges)
            solodists[charge][dr] = list(dist)
            dr += 1
    finaldefiniteind = dr
    #chargelens = {k: len(v) for k, v in solodists.items()}
    #return chargelens
    
    foundvals = []
    for charge, sgd in solodists.items():
        foundvals.extend(list(chain(*sgd.values())))
    #specvals = regiter[:,8].astype(int)
    #nodists = np.setdiff1d(specvals, foundvals)
    nodists = np.setdiff1d(np.arange(len(regiter)), foundvals)

    t1 = time()
    
    #charge-handling start
    distributionmasses = {} #distid: ordered masses
    distributioncharges = {} #distid: charge
    distributionsoflines = {} #line: distid
    linesofdistributions = {} #distid: mass-ordered linedkeys
    #distributiontimelimits = {} #distid [starting rt, ending rt]
    distributionintensities = {} #distid: mass-ordered intensities
    #distributionsbycharge = defaultdict(dict) #charge: dists: mass-ordered linekeys
    distributionsbycharge = defaultdict(list) #charge: dists: mass-ordered linekeys
    for charge, dists in solodists.items():
        for dist, lines in dists.items():
            dmasses = regiter[lines,0]
            lineorder = dmasses.argsort().tolist()
            sortedlines = [lines[i] for i in lineorder]
            #sortedmasses = regiter[sortedlines,7].tolist()
            #sortedmasses = dmasses[sortedlines].tolist()
            sortedmasses = dmasses[lineorder].tolist()
            #dintensities = regiter[sortedlines,5].tolist()
            dintensities = regiter[sortedlines,1].tolist()
            #rtlimits = regiter[sortedlines,2:4]
            #minrt = rtlimits.min()
            #maxrt = rtlimits.max()
            distributionmasses[dist] = sortedmasses
            distributioncharges[dist] = charge
            for line in lines:
                distributionsoflines[line] = dist
            linesofdistributions[dist] = sortedlines
            #distributiontimelimits[dist] = [minrt, maxrt]
            distributionintensities[dist] = dintensities
            distributionsbycharge[charge].append(dist)
    
    #nodistmasses = regiter[nodists,7]
    nodistmasses = regiter[nodists,0]
    
    nodistkeys = []
    #dr = finaldefiniteind
    for line in nodists.tolist():
        dreg = regiter[line]
        #dmass = dreg[7] 
        #dintensity = dreg[5]
        dmass = dreg[0]
        dintensity = dreg[1]
        #rtlimit = dreg[2:4] 
        #minrt = rtlimit.min()
        #maxrt = rtlimit.max()
        distributionmasses[dr] = [dmass]
        distributioncharges[dr] = 0
        distributionsoflines[line] = dr
        linesofdistributions[dr] = [line]
        #distributiontimelimits[dr] = [minrt, maxrt]
        distributionintensities[dr] = [dintensity]
        distributionsbycharge[0].append(dr)
        nodistkeys.append(dr)
        dr += 1 #continuing from solodists count
    
    #massranges = regiter[:,:2]
    massranges = regiter[:,0]
    minmass = massranges.min() - 1
    maxmass = massranges.max() + 1
    
    if nodists.size > 0:
        #keeping the 0-charge nodists out
        chargeiterations = sorted(distributionsbycharge)[1:]
    else:
        chargeiterations = sorted(distributionsbycharge)
    
    nodistcharges = defaultdict(dict)
    chargegroups = []
    oldcharge = -1
    for charge in chargeiterations: #ignoring 0-charge nodists here
        if charge - oldcharge > 1: #adjacent charge searches only
            moving = False
        dists = distributionsbycharge[charge]
        distkeys = []
        massdiffs = []
        distmasses = []
        for d in dists:
            dmasses = np.array(distributionmasses[d])
            basemasses = dmasses * charge - proton * charge
            distkeys.extend(d for _ in dmasses)
            distmasses.extend(basemasses)
            massdiff = np.diff(basemasses)
            massdiffs.extend(massdiff.tolist())
        massdiffs = np.array(massdiffs)
        diffcuts = np.abs(proton - massdiffs)
        ctol = diffcuts.mean() #use max for for exploration I suppose
        distmasses = np.array(distmasses)[:,None]
        distkeys = np.array(distkeys)
        chargemodel = spatial.KDTree(distmasses)
        nodistupmodel = spatial.KDTree(nodistmasses[:,None] * charge - proton * charge)
        if moving:
            #train it on the new and find old masses + radii
            roundmatches = set()
            oldmatches = set()
            matches = chargemodel.query_ball_point(oldmasses, oldradii, workers=1).tolist()
            for m, o in zip(matches, oldkeys):
                if m:
                    matchkeys = []
                    #ominrt, omaxrt = distributiontimelimits[o]
                    omasses = np.array(distributionmasses[o])
                    obmasses = omasses * oldcharge - proton * oldcharge
                    cintensities = np.array(distributionintensities[o])
                    intensityranks = cintensities.argsort()[::-1]
                    for mk in distkeys[m].tolist():
                        #nminrt, nmaxrt = distributiontimelimits[mk]
                        nmasses = np.array(distributionmasses[mk])
                        nbmasses = nmasses * charge - proton * charge
                        #below is requiring a majority of the matchable masses have sufficiently overlapping retention times
                        #if ominrt < nmaxrt and omaxrt > nminrt: #overlap exists
                        basemasses = [obmasses, nbmasses]
                        #dlines = [[regiter[j,2:4].tolist() for j in linesofdistributions[i]] for i in [o, mk]]
                        sizes = [i.size for i in basemasses]
                        maxind = sizes.index(max(sizes))
                        lineup = basemasses[maxind]
                        #retentionboundaries = defaultdict(list)
                        #for n, (sm, dlims) in enumerate(zip(basemasses, dlines)):
                        for n, sm in enumerate(basemasses):
                            sdiff = np.abs(lineup - sm[:,None])
                            alignmentloc = np.argwhere(sdiff == sdiff.min())[0]
                            alignmentloc -= alignmentloc.min()
                            luind = alignmentloc[1] - alignmentloc[0]
                            outinds = [luind, luind + sm.size]
                            #for ind, dlim in zip(range(*outinds), dlims):
                            #    retentionboundaries[ind].append(dlim)
                            #overpass = 0
                            #matchables = 0
                            #for ind, lims in retentionboundaries.items():
                            #    if len(lims) > 1:
                            #        matchables += 1
                            #        (lminrt, lmaxrt), (rminrt, rmaxrt) = lims
                            #        if lminrt > rminrt and lmaxrt < rmaxrt: #old encompassed
                            #            overpass += 1
                            #        elif rminrt > lminrt and rmaxrt < lmaxrt: #new encompassed
                            #            overpass += 1
                            #        else:
                            #            overlap = min(rmaxrt, lmaxrt) - max(rminrt, lminrt)
                            #            fullrange = max(rmaxrt, lmaxrt) - min(rminrt, lminrt)
                            #            if overlap / fullrange > 0.5:
                            #                overpass += 1
                            #if overpass > matchables / 2:
                            diffmatrix = np.abs(obmasses - nbmasses[:,None])
                            matchmatrix = diffmatrix < ctol
                            mmshape = matchmatrix.shape
                            majoraxis = np.argmax(mmshape)
                            minoraxis = np.argmin(mmshape)
                            matrixmatches = matchmatrix.any(axis=majoraxis)
                            if matrixmatches.sum() > matrixmatches.size / 2: #matching at least half of the smaller with >=, make this a 'majority' with >
                                alignmentloc = np.argwhere(diffmatrix == diffmatrix.min())[0]
                                minindex = min(alignmentloc)
                                alignmentloc = [i-minindex for i in alignmentloc] #for when the closest match isn't the earliest matching index
                                rmax = nbmasses.size - alignmentloc[0]
                                lmax = obmasses.size - alignmentloc[1]
                                maxsize = min(rmax, lmax)
                                matchintensities = np.array(distributionintensities[mk])
                                matchintensityranks = matchintensities.argsort()[::-1]
                                scoutrankslice = intensityranks[alignmentloc[1]:alignmentloc[1]+maxsize]
                                matchrankslice = matchintensityranks[alignmentloc[0]:alignmentloc[0]+maxsize]
                                slicesubtraction = np.abs(scoutrankslice - matchrankslice)
                                if slicesubtraction.sum() <= slicesubtraction.size - 1: #close order matching passes
                                    matchkeys.append(mk)
                                    roundmatches.add(mk)
                                    oldmatches.add(o)
                    if matchkeys:
                        matchkeys.append(o)
                        chargegroups.append(matchkeys)
            #nodist match against new, nodists as lower charge
            #there are other nonmatches that don't pass the above, add them here I think
            nodistdownmodel = spatial.KDTree(nodistmasses[:,None] * oldcharge - proton * oldcharge)
            nonmatched = set(distkeys).difference(roundmatches)
            nonmatchedmasses = np.array([distmasses[n] for n in range(len(distkeys)) if distkeys[n] in nonmatched])
            nonmatchedkeys = [distkeys[n] for n in range(len(distkeys)) if distkeys[n] in nonmatched]
            if nonmatchedmasses.size > 0: #this is a new addition, not fully tested
                downmatches = nodistdownmodel.query_ball_point(nonmatchedmasses, oldradii, workers=1).tolist()
                for m, o in zip(downmatches, nonmatchedkeys):
                    if m:
                        matchkeys = []
                        #ominrt, omaxrt = distributiontimelimits[o]
                        omasses = np.array(distributionmasses[o])
                        obmasses = omasses * charge - proton * charge
                        cintensities = np.array(distributionintensities[o])
                        intensityranks = cintensities.argsort()[::-1]
                        for mkey in m:
                            #nodists are from m
                            mk = nodistkeys[mkey]
                            matchintensities = np.array(distributionintensities[mk])
                            if matchintensities.max() < cintensities.max():
                                #nminrt, nmaxrt = distributiontimelimits[mk]
                                nmasses = np.array(distributionmasses[mk])
                                nbmasses = nmasses * oldcharge - proton * oldcharge
                                #below is requiring a majority of the matchable masses have sufficiently overlapping retention times
                                #if ominrt < nmaxrt and omaxrt > nminrt: #overlap exists
                                basemasses = [obmasses, nbmasses]
                                #dlines = [[regiter[j,2:4].tolist() for j in linesofdistributions[i]] for i in [o, mk]]
                                sizes = [i.size for i in basemasses]
                                maxind = sizes.index(max(sizes))
                                lineup = basemasses[maxind]
                                #retentionboundaries = defaultdict(list)
                                #for n, (sm, dlims) in enumerate(zip(basemasses, dlines)):
                                for n, sm in enumerate(basemasses):
                                    sdiff = np.abs(lineup - sm[:,None])
                                    alignmentloc = np.argwhere(sdiff == sdiff.min())[0]
                                    alignmentloc -= alignmentloc.min()
                                    luind = alignmentloc[1] - alignmentloc[0]
                                    outinds = [luind, luind + sm.size]
                                    #for ind, dlim in zip(range(*outinds), dlims):
                                    #    retentionboundaries[ind].append(dlim)
                                #overpass = 0
                                #matchables = 0
                                #for ind, lims in retentionboundaries.items():
                                #    if len(lims) > 1:
                                #        matchables += 1
                                #        (lminrt, lmaxrt), (rminrt, rmaxrt) = lims
                                #        if lminrt > rminrt and lmaxrt < rmaxrt: #old encompassed
                                #            overpass += 1
                                #        elif rminrt > lminrt and rmaxrt < lmaxrt: #new encompassed
                                #            overpass += 1
                                #        else:
                                #            overlap = min(rmaxrt, lmaxrt) - max(rminrt, lminrt)
                                #            fullrange = max(rmaxrt, lmaxrt) - min(rminrt, lminrt)
                                #            if overlap / fullrange > 0.5:
                                #                overpass += 1
                                #if overpass > matchables / 2:
                                diffmatrix = np.abs(obmasses - nbmasses[:,None])
                                matchmatrix = diffmatrix < ctol
                                mmshape = matchmatrix.shape
                                majoraxis = np.argmax(mmshape)
                                minoraxis = np.argmin(mmshape)
                                matrixmatches = matchmatrix.any(axis=majoraxis)
                                if matrixmatches.sum() > matrixmatches.size / 2: #matching at least half of the smaller with >=, make this a 'majority' with >
                                    alignmentloc = np.argwhere(diffmatrix == diffmatrix.min())[0]
                                    minindex = min(alignmentloc)
                                    alignmentloc = [i-minindex for i in alignmentloc] #for when the closest match isn't the earliest matching index
                                    rmax = nbmasses.size - alignmentloc[0]
                                    lmax = obmasses.size - alignmentloc[1]
                                    maxsize = min(rmax, lmax)
                                    matchintensities = np.array(distributionintensities[mk])
                                    matchintensityranks = matchintensities.argsort()[::-1]
                                    scoutrankslice = intensityranks[alignmentloc[1]:alignmentloc[1]+maxsize]
                                    matchrankslice = matchintensityranks[alignmentloc[0]:alignmentloc[0]+maxsize]
                                    slicesubtraction = np.abs(scoutrankslice - matchrankslice)
                                    if slicesubtraction.sum() <= slicesubtraction.size - 1: #close order matching passes
                                        nodistcharges[o][mk] = oldcharge
                        if matchkeys:
                            matchkeys.append(o)
                            chargegroups.append(matchkeys)
            #nodist match against old, nodists as higher charge
            nonoldmatches = set(oldkeys).difference(oldmatches) #if you end up needing more competition in the mix, you can just match everything to nodists and you'll get competing matches, I've left them out via routes like this so as to lessen the complications atm, competing nodist vs matches of the same charge can be annoying when determining if a distribution should extend beyond a nodist match
            nodistupmodel = spatial.KDTree(nodistmasses[:,None] * charge - proton * charge)
            nonmatchedmasses = np.array([oldmasses[n] for n in range(len(oldkeys)) if oldkeys[n] in nonoldmatches])
            nonmatchedkeys = [oldkeys[n] for n in range(len(oldkeys)) if oldkeys[n] in nonoldmatches]
            if nonmatchedmasses.size > 0: #this is a new addition, not fully tested
                upmatches = nodistupmodel.query_ball_point(nonmatchedmasses, oldradii, workers=1).tolist()
                for m, o in zip(upmatches, nonmatchedkeys):
                    if m:
                        matchkeys = []
                        #ominrt, omaxrt = distributiontimelimits[o]
                        omasses = np.array(distributionmasses[o])
                        obmasses = omasses * oldcharge - proton * oldcharge
                        cintensities = np.array(distributionintensities[o])
                        intensityranks = cintensities.argsort()[::-1]
                        for mkey in m:
                            mk = nodistkeys[mkey]
                            matchintensities = np.array(distributionintensities[mk])
                            if matchintensities.max() < cintensities.max():
                                #nminrt, nmaxrt = distributiontimelimits[mk]
                                nmasses = np.array(distributionmasses[mk])
                                nbmasses = nmasses * charge - proton * charge
                                #below is requiring a majority of the matchable masses have sufficiently overlapping retention times
                                #if ominrt < nmaxrt and omaxrt > nminrt: #overlap exists
                                basemasses = [obmasses, nbmasses]
                                #dlines = [[regiter[j,2:4].tolist() for j in linesofdistributions[i]] for i in [o, mk]]
                                sizes = [i.size for i in basemasses]
                                maxind = sizes.index(max(sizes))
                                lineup = basemasses[maxind]
                                #retentionboundaries = defaultdict(list)
                                #for n, (sm, dlims) in enumerate(zip(basemasses, dlines)):
                                for n, sm in enumerate(basemasses):
                                    sdiff = np.abs(lineup - sm[:,None])
                                    alignmentloc = np.argwhere(sdiff == sdiff.min())[0]
                                    alignmentloc -= alignmentloc.min()
                                    luind = alignmentloc[1] - alignmentloc[0]
                                    outinds = [luind, luind + sm.size]
                                    #for ind, dlim in zip(range(*outinds), dlims):
                                    #    retentionboundaries[ind].append(dlim)
                                #overpass = 0
                                #matchables = 0
                                #for ind, lims in retentionboundaries.items():
                                #    if len(lims) > 1:
                                #        matchables += 1
                                #        (lminrt, lmaxrt), (rminrt, rmaxrt) = lims
                                #        if lminrt > rminrt and lmaxrt < rmaxrt: #old encompassed
                                #            overpass += 1
                                #        elif rminrt > lminrt and rmaxrt < lmaxrt: #new encompassed
                                #            overpass += 1
                                #        else:
                                #            overlap = min(rmaxrt, lmaxrt) - max(rminrt, lminrt)
                                #            fullrange = max(rmaxrt, lmaxrt) - min(rminrt, lminrt)
                                #            if overlap / fullrange > 0.5:
                                #                overpass += 1
                                #if overpass > matchables / 2:
                                diffmatrix = np.abs(obmasses - nbmasses[:,None])
                                matchmatrix = diffmatrix < ctol
                                mmshape = matchmatrix.shape
                                majoraxis = np.argmax(mmshape)
                                minoraxis = np.argmin(mmshape)
                                matrixmatches = matchmatrix.any(axis=majoraxis)
                                if matrixmatches.sum() > matrixmatches.size / 2: #matching at least half of the smaller with >=, make this a 'majority' with >
                                    alignmentloc = np.argwhere(diffmatrix == diffmatrix.min())[0]
                                    minindex = min(alignmentloc)
                                    alignmentloc = [i-minindex for i in alignmentloc] #for when the closest match isn't the earliest matching index
                                    rmax = nbmasses.size - alignmentloc[0]
                                    lmax = obmasses.size - alignmentloc[1]
                                    maxsize = min(rmax, lmax)
                                    matchintensities = np.array(distributionintensities[mk])
                                    matchintensityranks = matchintensities.argsort()[::-1]
                                    scoutrankslice = intensityranks[alignmentloc[1]:alignmentloc[1]+maxsize]
                                    matchrankslice = matchintensityranks[alignmentloc[0]:alignmentloc[0]+maxsize]
                                    slicesubtraction = np.abs(scoutrankslice - matchrankslice)
                                    if slicesubtraction.sum() <= slicesubtraction.size - 1: #close order matching passes
                                        nodistcharges[o][mk] = charge
                        if matchkeys:
                            matchkeys.append(o)
                            chargegroups.append(matchkeys)
        oldmasses = distmasses.copy()
        oldradii = ctol
        oldkeys = distkeys.copy()
        oldcharge = charge
        moving = True
    
    #combining redundant matches
    chargesets = intersection_merge(chargegroups)
    
    chargelayers = defaultdict(lambda: defaultdict(set)) #groupid: charge: [connections]
    extralayercharges = defaultdict(dict) #dist: nodist: charge
    distchargesofnodists = defaultdict(dict) #nodist: dists: charge
    for n, cs in enumerate(chargesets):
        for c in cs:
            chargelayers[n][distributioncharges[c]].add(c)
            if c in nodistcharges:
                for nc, ncc in nodistcharges[c].items():
                    chargelayers[n][ncc].add(nc)
                    extralayercharges[n][nc] = ncc
                    distchargesofnodists[linesofdistributions[nc][0]][c] = ncc
    
    cid = 0
    conpairs = {} #cid: adjacent charge-state matches as pairs
    chargeconsets = set() #finalized charge-state groups for lookup
    chargecongroups = defaultdict(dict) #connection: [all chargegroups its involved in]
    #distmatchcount = defaultdict(int) #it's for plotting, I guess
    for ck, cl in chargelayers.items():
        for connections in product(*cl.values()):
            connectioncharges = {}
            for con in connections:
                if distributioncharges[con] > 0:
                    bc = distributioncharges[con]
                else:
                    bc = extralayercharges[ck][con]
                connectioncharges[con] = bc
            connections = sorted(connections, key=lambda x: connectioncharges[x])
            clen = len(connections)
            startind = 0
            endind = 2
            coniters = []
            restart = True
            while endind <= clen:
                if restart:
                    tempind = 0
                    restart = False
                coniters.append(tuple(connections[startind+tempind:endind+tempind]))
                tempind += 1
                if endind + tempind > clen:
                    restart = True
                    endind += 1
            for cons in coniters:
                basemasses = []
                for con in cons:
                    if distributioncharges[con] > 0:
                        bc = distributioncharges[con]
                    else:
                        bc = extralayercharges[ck][con]
                    bm = np.array(distributionmasses[con]) * bc - proton * bc
                    basemasses.append(bm)
                cintensities = [np.array(distributionintensities[i]) for i in cons] 
                #retentiontimes = [distributiontimelimits[i] for i in cons]
                #^this wouldn't be so great, I want individual isotopomer overlaps to be assessed here
                #dlines = [[regiter[j,2:4].tolist() for j in linesofdistributions[i]] for i in cons]
                sizes = [i.size for i in cintensities]
                maxind = sizes.index(max(sizes))
                lineup = basemasses[maxind]
                masslines = defaultdict(list)
                intensitylines = defaultdict(list)
                #retentionboundaries = defaultdict(list)
                lineinds = defaultdict(list)
                #for n, (sm, sints, dlims) in enumerate(zip(basemasses, intensities, dlines)):
                for n, (sm, sints) in enumerate(zip(basemasses, cintensities)):
                    sdiff = np.abs(lineup - sm[:,None])
                    alignmentloc = np.argwhere(sdiff == sdiff.min())[0]
                    alignmentloc -= alignmentloc.min()
                    luind = alignmentloc[1] - alignmentloc[0]
                    outinds = [luind, luind + sm.size]
                    #below allows for mean diffs to be made on unequal sized arrays
                    #for ind, mass, sint, dlim in zip(range(*outinds), sm.tolist(), sints, dlims):
                    for ind, mass, sint in zip(range(*outinds), sm.tolist(), sints):
                        masslines[ind].append(mass)
                        intensitylines[ind].append(sint)
                        #retentionboundaries[ind].append(dlim)
                        lineinds[ind].append(n)
                #all isotopomers of a basemass must overlap -> the whole group must overlap with everything in that group, and > 1/2 of the groups must fulfill this
                #negative values [that form via non-overlaps] will get removed from mainoverlaps below
                #matchcount = 0
                #passedmatches = 0
                #for k, v in retentionboundaries.items():
                #    if len(v) > 1:
                #        matchcount += 1
                #        v = np.array(v)
                #        if np.logical_and(v[:,0,None] <= v[:,1], v[:,1,None] >= v[:,0]).all():
                #            passedmatches += 1
                #if passedmatches > matchcount / 2:
                massmeans = []
                intensitylocations = defaultdict(list)
                #rtoverlaps = []
                for n, inds in lineinds.items():
                    if len(inds) > 1:
                        massline = np.array(masslines[n])
                        massmeans.extend(np.abs(massline.mean() - massline).tolist())
                        intensityline = np.array(intensitylines[n])
                        intensityperc = intensityline / intensityline.sum()
                        for p, i in zip(intensityperc.tolist(), inds):
                            intensitylocations[i].append(p)
                        #rbounds = np.array(retentionboundaries[n])
                        #not doing the fully vectorized approach so that selfoverlaps can be excluded while not infringing upon any actual 100% overlaps, although i could probably just take out a diagonal somewhere
                        #for ni in range(len(rbounds)-1):
                        #    mainrb = rbounds[ni].copy()
                        #    nonmainrbs = rbounds[ni+1:].copy()
                        #    leftinners = nonmainrbs[:,0].copy()
                        #    leftinners[leftinners < mainrb[0]] = mainrb[0]
                        #    rightinners = nonmainrbs[:,1].copy()
                        #    rightinners[rightinners > mainrb[1]] = mainrb[1]
                        #    leftouters = nonmainrbs[:,0].copy()
                        #    leftouters[leftouters > mainrb[0]] = mainrb[0]
                        #    rightouters = nonmainrbs[:,1].copy()
                        #    rightouters[rightouters < mainrb[1]] = mainrb[1]
                        #    mainoverlaps = (rightinners - leftinners) / (rightouters - leftouters)
                        #    mainoverlaps = mainoverlaps[mainoverlaps > 0]
                        #    if mainoverlaps.size > 0:
                        #        rtoverlaps.extend(mainoverlaps.tolist())
                intensitymeans = []
                for k, v in intensitylocations.items():
                    v = np.array(v)
                    intensitymeans.extend(np.abs(v.mean() - v).tolist())
                massmeandiff = np.mean(massmeans)
                intensitymeandiff = np.mean(intensitymeans)
                #generalizedoverlap = 1 / np.mean(rtoverlaps)
                for cn in range(len(cons)-1):
                    conpair = cons[cn:cn+2]
                    conpairs[cid] = conpair
                    for con in conpair:
                        #chargecongroups[con][cid] = [intensitymeandiff, massmeandiff, generalizedoverlap]
                        chargecongroups[con][cid] = [intensitymeandiff, massmeandiff]
                    cid += 1
                chargeconsets.add(tuple(sorted(cons)))
    
    #score balancing
    prioritycharges = []
    secondpriorities = []
    for con, congroups in chargecongroups.items():
        if len(congroups) > 1:
            csums = np.array(list(congroups.values())).sum(axis=0)
            #0s here would actually be pretty welcome
            csums[csums == 0] = 1
            #n1, n2, n3 = csums
            n1, n2 = csums
            #for congroup, (s1, s2, s3) in congroups.items():
            for congroup, (s1, s2) in congroups.items():
                #newconscore = sum((s1/n1, s2/n2, s3/n3))
                newconscore = sum((s1/n1, s2/n2))
                prioritycharges.append([congroup, newconscore])
        else:
            congroup, conscore = list(congroups.items())[0]
            conscore = conscore[:-1] #going to exclude the rt overlap here as there's no competition I suppose
            secondpriorities.append([congroup, sum(conscore)])
    
    rankedcharges = sorted(prioritycharges, key=lambda x: x[1]) #competition among matched chargegroups where matching can be done easily
    secondprioritycharges = sorted(secondpriorities, key=lambda x: x[1]) #no competition happening here, the order doesn't actually matter
    rankedcharges.extend(secondprioritycharges)
    
    blocked = set()
    preservedchargeranks = []
    for group, score in rankedcharges:
        if group not in blocked:
            preservedchargeranks.append(conpairs[group])
            blocked.add(group)
    
    sn = 0
    chargegroups = defaultdict(set) #analyteid: [dist ids]
    groupsofdists = {} #dist: analyteid
    for dists in preservedchargeranks:
        locs = set()
        for i in dists:
            if i in groupsofdists:
                locs.add(groupsofdists[i])
        if locs:
            joiner = min(locs)
            if len(locs) > 1:
                tempchargegroup = set(dists)
                for l in locs:
                    tempchargegroup.update(chargegroups[l])
                if tuple(sorted(tempchargegroup)) in chargeconsets:
                    for oldlocs in locs.difference([joiner]):
                        for ol in chargegroups[oldlocs]:
                            groupsofdists[ol] = joiner
                        chargegroups[joiner].update(chargegroups.pop(oldlocs))
                    chargegroups[joiner].update(dists)
                    for i in dists:
                        groupsofdists[i] = joiner
            else:
                if tuple(sorted(chargegroups[joiner].union(dists))) in chargeconsets:
                    chargegroups[joiner].update(dists)
                    for i in dists:
                        groupsofdists[i] = joiner
        else:
            joiner = sn
            sn += 1
            chargegroups[joiner].update(dists)
            for i in dists:
                groupsofdists[i] = joiner
    
    chargedistgroups = defaultdict(dict) #analyteid: charge: distributionid
    chargegroupsbyline = {} #line: analyteid, doubles as blocking list
    for groupid, dists in chargegroups.items():
        for dist in dists:
            charge = distributioncharges[dist]
            chargedistgroups[groupid][charge] = dist
            distlines = linesofdistributions[dist]
            #chargedistlines[groupid][charge] = distlines
            for line in distlines:
                chargegroupsbyline[line] = groupid
    
    additionaldistributions = set(chargegroupsbyline).intersection(nodists) #nodists that made it into distributions will be given their own distribution
    distributionchangesetup = {k:distributionsoflines[k] for k in additionaldistributions}
    
    #removing all previous nodists from any distribution dicts
    #finaldefiniteind = max(max(i.keys()) for i in solodists.values()))
    indcopy = finaldefiniteind
    while indcopy in distributionmasses:
        del distributionmasses[indcopy]
        #del distributiontimelimits[indcopy]
        del distributionintensities[indcopy]
        distributionsbycharge[distributioncharges[indcopy]].remove(indcopy)
        for line in linesofdistributions[indcopy]:
            del distributionsoflines[line]
        del linesofdistributions[indcopy]
        indcopy += 1
    
    distributionchanges = {}
    for line in additionaldistributions:
        dreg = regiter[line]
        cgroup = chargegroupsbyline[line]
        potentialcharges = distchargesofnodists[line]
        chargekey = chargegroups[cgroup].intersection(potentialcharges)
        if len(chargekey) > 1:
            print('big error -', line, 'has multiple potential nodist keys')
        ckey = list(chargekey)[0]
        charge = distchargesofnodists[line][ckey]
        #dmass = dreg[7] 
        dmass = dreg[0]
        #dintensity = dreg[5]
        dintensity = dreg[1]
        #rtlimit = dreg[2:4]
        #minrt = rtlimit.min()
        #maxrt = rtlimit.max()
        distributionmasses[finaldefiniteind] = [dmass]
        distributioncharges[finaldefiniteind] = charge
        distributionsoflines[line] = finaldefiniteind
        linesofdistributions[finaldefiniteind] = [line]
        #distributiontimelimits[finaldefiniteind] = [minrt, maxrt]
        distributionintensities[finaldefiniteind] = [dintensity]
        #distributionsbycharge[charge][dist] = sortedlines
        distributionsbycharge[0].append(finaldefiniteind)
        oldkey = distributionchangesetup[line]
        distributionchanges[oldkey] = finaldefiniteind
        solodists[charge][finaldefiniteind] = [line]
        finaldefiniteind += 1 #continuing from solodists count
    
    #distributionregions = []
    #for k, v in solodists.items():
    #    for sk, sv in v.items():
    #        dmasses = np.array(distributionmasses[sk])
    #        massmax = dmasses.max()
    #        massmin = dmasses.min()
    #        cintensities = np.array(distributionintensities[sk])
    #        mainmass = dmasses[cintensities.argmax()]
    #        #mintime, maxtime = distributiontimelimits[sk]
    #        signalsum = defaultdict(float) #time: total intensity
    #        signalsum = cintensities.sum()
    #        for line in sv:
    #            data = trackedgroups[line]
    #            for t, i in zip(data[1], data[2]):
    #                signalsum[t] += i
    #        signals = np.array(list(signalsum.items()))
    #        area = np.trapezoid(signals[:,1], signals[:,0])
    #        #el = [massmin, massmax, mintime, maxtime, len(sv), area, k, mainmass, sk]
    #        el = [massmin, massmax, len(sv), area, k, mainmass, sk]
    #        distributionregions.append(el)
    #distributionregions = np.array(distributionregions)
    ##when there's no distributions: adjust this below
    #distributionregions = distributionregions[distributionregions[:,6].argsort()]
    
    chargeregions = []
    chargestatelines = {} #needs confirmation -> lineid: chargeid
    for k, v in chargedistgroups.items():
        #mincharge = min(v)
        #maxcharge = max(v)
        #times = []
        #signalsum = defaultdict(float) #time: total intensity
        changers = {}
        for vc, sv in v.items():
            if sv in distributionchanges:
                changers[vc] = sv
        for vc, sv in changers.items():
            del chargedistgroups[k][vc]
            sv = distributionchanges[sv]
            vc = distributioncharges[sv]
            chargedistgroups[k][vc] = sv
        #for vc, sv in v.items():
        #    #times.extend(distributiontimelimits[sv])
        #    for line in linesofdistributions[sv]:
        #        data = trackedgroups[line]
        #        for t, i in zip(data[1], data[2]):
        #            signalsum[t] += i
        #        chargestatelines[line] = vc
        #signals = np.array(list(signalsum.items()))
        #sa = signals[:,0].argsort()
        #signals = signals[sa]
        #area = np.trapezoid(signals[:,1], signals[:,0])
        #distinds = np.array(list(v.values()))
        #charges = np.array(list(v))
        #maincharge = charges[distributionregions[distinds,5].argmax()]
        ##mintime = min(times)
        ##maxtime = max(times)
        ##el = [mincharge, maxcharge, mintime, maxtime, len(v), area, maincharge, k]
        #el = [mincharge, maxcharge, len(v), area, maincharge, k]
        #chargeregions.append(el)
    #chargeregions = np.array(chargeregions)

    #pack these up with distributionsoflines
    #analyte id == chargegroup id if there is one, continue using sn for this count
    analytekeys = defaultdict(dict) #analyte id: distid: charge, this can be one or multiple distributions based on whether it has multiple charge states
    analytedistributions = {} #analyte id: [[weighted means [via intensity] across isotopomers from every charge state if there are multiple], [AUC of merged isotopomers]]
    analytesbydistribution = {} #distid: analyte id
    blocked = set()
    for k, v in chargedistgroups.items():
        massholder = []
        lineholder = []
        for charge, distid in v.items():
            analytekeys[k][distid] = charge
            dmasses = np.array(distributionmasses[distid]) * charge - proton * charge
            dlines = linesofdistributions[distid]
            massholder.append(dmasses)
            lineholder.append(dlines)
            analytesbydistribution[distid] = k
            blocked.add(distid)
        
        sortedlines = sorted(lineholder, key=lambda x: len(x))
        sortedmasses = sorted(massholder, key=lambda x: x.size)
        lineup = sortedmasses[-1][:,None]
        distinds = []
        for sm in sortedmasses:
            sdiff = np.abs(lineup - sm)
            alignmentloc = np.argwhere(sdiff == sdiff.min())[0]
            alignmentloc -= alignmentloc.min()
            luind = alignmentloc[1] - alignmentloc[0]
            outinds = [luind, luind + sm.size]
            distinds.append(outinds)
        
        spaceorganizer = defaultdict(list) #position: [lineuids]
        spacemasses = defaultdict(list) #position: [basemasses]
        for di, sl, sm in zip(distinds, sortedlines, sortedmasses):
            dspaces = range(*di)
            for space, line, mass in zip(dspaces, sl, sm):
                spaceorganizer[space].append(line)
                spacemasses[space].append(mass)
        
        massesandintensities = [[], []]
        for sk in sorted(spaceorganizer):
            slines = spaceorganizer[sk]
            smasses = np.array(spacemasses[sk])
            areavals = regiter[slines,1]
            weightedmass = (smasses * areavals).sum() / areavals.sum()
            #signalsum = defaultdict(float) #time: sum intensity
            #^this was previously time-based in the original setup, for MS1 lines
            #for line in slines:
            #    linegroup = trackedgroups[line]
            #    times = linegroup[1]
            #    intensities = linegroup[2]
            #    for t, i in zip(times, intensities):
            #        signalsum[t] += i
            #signals = np.array(list(signalsum.items()))
            signal = areavals.sum()
            #sa = signals[:,0].argsort()
            #signals = signals[sa]
            #area = np.trapezoid(signals[:,1], signals[:,0])
            #analytedistributions[k][weightedmass] = area
            massesandintensities[0].append(weightedmass)
            #massesandintensities[1].append(area)
            massesandintensities[1].append(signal)
        analytedistributions[k] = np.array(massesandintensities)

    #adding any distribution without multiple charge states to analytedistributions
    for distid, distmasses in distributionmasses.items():
        if distid not in blocked:
            charge = distributioncharges[distid]
            distmasses = np.array(distmasses)
            basemasses = distmasses * charge - proton * charge
            intensities = distributionintensities[distid]
            analytekeys[sn][distid] = charge
            analytesbydistribution[distid] = sn
            #for m, i in zip(basemasses.tolist(), intensities):
            #    analytedistributions[sn][m] = i
            massesandintensities = np.array([basemasses, intensities])
            analytedistributions[sn] = massesandintensities
            sn += 1
    
    #newdistcount = max(distributionmatches) + 1
    for lineid in nodists:
        if lineid not in additionaldistributions:
            #add more if its ever necessary
            m, i = regiter[lineid]
            #assuming them as 1+ here
            #bm = m * 1 - proton * 1
            bm = m - proton
            analytedistributions[sn] = np.array([[bm], [i]])
            #analytekeys[sn][newdistcount] = 1
            #analytesbydistrbution[newdistcount] = sn
            #distributionsoflines[lineid] = newdistcount
            #linesbydistribution[newdistcount] = [lineid]
            #distributioncharges[newdistcount] = 1
            #newdistcount += 1
            sn += 1

    masslist = []
    analytelist = []
    for analyteid, analytemasses in analytedistributions.items():
        for am, ai in analytemasses.transpose().tolist():
            analytelist.append([am, ai, analyteid])
            masslist.append(am)
    analytelist = np.array(analytelist)

    #chargegroupsbydist = {} #distid: analyteid #ALREADY GOOD - analytesbydistribution
    #ms2distributions = {} #{distid: [lineuids]} #ALREADY GOOD - linesofdistributions
    #ms2chargesofdistributions = {} #{distid: charge} #ALREADY GOOD - distributioncharges
    #ms2distributionsbyanalyte = {} #{analyteid: [dist id's]} #subbing this for analytekeys
    #ms2analytedistributions = {} #{analyteid: [[masses], [intensities]]} #ALREADY GOOD - analytedistributions
    #return analytesbydistribution, linesofdistributions, distributioncharges, analytekeys, analytedistributions
    return scan['index'], analytedistributions, analytelist, masslist

def fragment_deconvolution(chargetolerance, mzmlfile, nprocs, processingdirectory):
    t1 = time()
    
    newinclimit = 0.1
    steplimit = 0.5
    
    #formalized analyte information, summarizing all distributions across any charge states
    analytefile = ''.join((processingdirectory, 'analytefactors.pickle'))
    with open(analytefile, 'rb') as pick:
        distributionsoflines = pickle.load(pick)[3]
    #distributionsoflines: lineuid: distid
    
    #using this to get charges of lines
    distributionregionsloc = ''.join((processingdirectory, 'distributionregions.pickle'))
    with open(distributionregionsloc, 'rb') as pick:
        distributionregions = pickle.load(pick)
    #distriutionregions as [massmin, massmax, mintime, maxtime, # lines, area, charge, mainmass, distributionid]
    
    linesofscansfile = ''.join((processingdirectory, 'linesofscans.pickle'))
    with open(linesofscansfile, 'rb') as pick:
        linesofscans = pickle.load(pick)


    #insert here, a maxchargeofscan
    maxchargeofscans = {} #ms2 scanid: max charge of any distribution sampled
    for scan, lines in linesofscans.items():
        maxcharge = 1 #default i guess?!
        for line in lines:
            dpass = False
            try:
                dist = distributionsoflines[line]
                dpass = True
            except KeyError:
                pass
            if dpass:
                charge = distributionregions[dist,6]
                if charge > maxcharge:
                    maxcharge = charge
        maxchargeofscans[scan] = maxcharge
    
    charge_deconvolution_partial = partial(charge_deconvolution, steplimit, newinclimit, chargetolerance, maxchargeofscans)
    
    fraganalytedistributions = {}
    fraganalytelists = {} #scan: analytelists
    fragneighbors = {} #scan: masslist
    
    msrun = mzml.MzML(mzmlfile, dtype=np.float64)
    
    print(time() - t1, 'fragment deconvolution initiated')
    t2 = time()

    #for scan in msrun:
    for output in msrun.map(lambda scan: charge_deconvolution_partial(scan), processes=nprocs):
        #output = charge_deconvolution(steplimit, newinclimit, chargetolerance, scan)
        match output:
            case tuple():
                index, analytedistributions, analytelist, masslist = output
                fraganalytedistributions[index] = analytedistributions
                fraganalytelists[index] = analytelist
                fragneighbors[index] = masslist
    print(time() - t2, 'scans processed')
    
    fraganalytedistributionsfile = ''.join((processingdirectory, 'fraganalytedistributions.pickle'))
    with open(fraganalytedistributionsfile, 'wb') as pick:
        pickle.dump(fraganalytedistributions, pick)
    
    fraganalytelistsfile = ''.join((processingdirectory, 'fraganalytelists.pickle'))
    with open(fraganalytelistsfile, 'wb') as pick:
        pickle.dump(fraganalytelists, pick)
    
    fragneighborsfile = ''.join((processingdirectory, 'fragneighbors.pickle'))
    with open(fragneighborsfile, 'wb') as pick:
        pickle.dump(fragneighbors, pick)
