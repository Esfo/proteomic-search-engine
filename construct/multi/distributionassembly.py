from elementalcomponents import proton
from database import environment

from collections import defaultdict
from itertools import chain, combinations
from time import time
import numpy as np
import pickle

def overlap_counts(rbounds):
    finalscore = 0
    for (lmin, lmax), (rmin, rmax) in combinations(rbounds, 2):
        #first check if they even overlap, if not -> set up for a subtraction
        #encompassment -> flanks subtracted from each other -> this difference will be subtracted from the total overlap -> final value
        #double flank, both has a flank -> both flanks added together -> subtracted from overlap -> final value
        #single flank -> flank is subtracted from overlap -> final value
        #non-overlap -> minus the full range of both
        if rmax > lmin and lmax > rmin:
            #overlap exists
            if rmax <= lmax:
                minmax = rmax
                maxmax = lmax
                maxl = True
            else:
                minmax = lmax
                maxmax = rmax
                maxl = False
            if rmin >= lmin:
                maxmin = rmin
                minmin = lmin
                minl = True
            else:
                maxmin = lmin
                minmin = rmin
                minl = False
            if maxl and minl:
                if rmax == lmax and rmin == lmin:
                    #equal boundaries
                    overlap = rmax - rmin
                    fullrange = lmax - lmin
                    finalscore += overlap
                elif rmax == lmax:
                    #min l flank
                    overlap = lmax - rmin
                    fullrange = rmax - lmin
                    minflank = rmin - lmin
                    score = overlap - minflank
                    finalscore += score
                elif rmin == lmin:
                    #max l flank
                    overlap = rmax - rmin
                    fullrange = lmax - lmin
                    maxflank = lmax - rmax
                    score = overlap - maxflank
                    finalscore += score
                else:
                    #l encompasses r
                    overlap = rmax - rmin
                    fullrange = lmax - lmin
                    minflank = rmin - lmin
                    maxflank = lmax - rmax
                    flankdiff = abs(maxflank - minflank)
                    score = overlap - flankdiff
                    finalscore += score
            elif maxl:
                if rmax == lmax:
                    #r min flank
                    overlap = rmax - lmin
                    fullrange = lmax - rmin
                    minflank = lmin - rmin
                    score = overlap - minflank
                    finalscore += score
                else:
                    #l max flank
                    #r min flank
                    overlap = rmax - lmin
                    fullrange = lmax - rmin
                    minflank = lmin - rmin
                    maxflank = lmax - rmax
                    flankdiff = abs(maxflank - minflank)
                    score = overlap - flankdiff
                    finalscore += score
            elif minl:
                if rmin == lmin:
                    #r max flank
                    overlap = lmax - rmin
                    fullrange = rmax - lmin
                    maxflank = rmax - lmax
                    score = overlap - maxflank
                    finalscore += score
                else:
                    #r max flank
                    #l min flank
                    overlap = lmax - rmin
                    fullrange = rmax - minl
                    minflank = rmin - lmin
                    maxflank = rmax - lmax
                    flankdiff = abs(maxflank - minflank)
                    score = overlap - flankdiff
                    finalscore += score
            else:
                #r encompasses l
                overlap = lmax - lmin
                fullrange = rmax - rmin
                maxflank = rmax - lmax
                minflank = rmin - lmin
                flankdiff = abs(maxflank - minflank)
                score = overlap - flankdiff
                finalscore += score
        else:
            #there's no overlap
            loverlap = lmax - lmin
            roverlap = rmax - rmin
            #subtract both overlaps
            finalscore -= loverlap + roverlap
    return finalscore

def distribution_assembly(minpoints, chargetolerance, librarylocation, processingdirectory, proteome):
    
    regionfile = ''.join((processingdirectory, 'regions.pickle'))
    with open(regionfile, 'rb') as pick:
        regions = pickle.load(pick)
    
    roundcutfile = ''.join((processingdirectory, 'roundcutoff.pickle'))
    with open(roundcutfile, 'rb') as pick:
        roundcutoff = pickle.load(pick)
    
    with environment(librarylocation) as env:
        proteomedb = env.open_db((proteome + '.info').encode())
        with env.begin(write=False) as txn:
            with txn.cursor(proteomedb) as cursor:
                uppermasslimit = float(cursor.get('uppermasslimit'.encode()).decode())
    
    subisomax = 0.01337851739
    newinclimit = 0.1
    steplimit = 0.5
    subisomax = subisomax + subisomax * chargetolerance
    
    t1 = time()
    
    rsorted = regions[regions[:,7].argsort()]
    regiter = rsorted[rsorted[:,4] >= minpoints]
    
    pairkeys = {} #pairkey: pair
    previousdecrease = {} #connectionindex: True, exists if something is increasing
    
    di = 0 #connectionindexs, keeping this incorrect plural makes it searchable
    paircharges = {} #connection: charge
    scoresbypair = {} #pair: [[absacdiff, datapercdiff, rtoffset],]
    pairsbyline = defaultdict(str) #line: 'pair,pair,'
    
    masswidthlimit = roundcutoff * 2
    
    connectionspine = defaultdict(str) #connectionindex: 'pairkey,pairkey,'
    latestconnections = defaultdict(lambda: defaultdict(str)) #masskey: charge: [current connection indices that end in masskey], a mass is always active as long as it's in masspool, this keeps track of all the groups that end in any given mass
    latestmass = {} #connectionindex: latest masskey
    
    pi = 0 #pairkeys
    masspool = []
    for reg in regiter.tolist():
        npoints = reg[4]
        nm = reg[7]
        nmlt = reg[2]
        nmrt = reg[3]
        nmassmin = reg[0]
        nmassmax = reg[1]
        nmwidth = nmassmax - nmassmin
        if nmwidth > masswidthlimit:
            #this is strict, but seems to help in the end, meaning that essentially the means are rather important and useful
            nmassmin = nm - masswidthlimit
            nmassmax = nm + masswidthlimit
        nkey = int(reg[8])
        nintensity = reg[5]
        masspoolremovals = []
        nlrange = nmrt - nmlt
        for okey in masspool:
            oreg = regions[okey]
            om = oreg[7]
            omassmin = oreg[0]
            omassmax = oreg[1]
            omwidth = omassmax - omassmin
            if omwidth > masswidthlimit:
                omassmin = om - masswidthlimit
                omassmax = om + masswidthlimit
            if nmassmin - omassmax <= proton:
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
                        if percentoverlap > 0.5: #this is super lenient I think
                            overpass = True
                #else:
                    #negative percentoverlap values for non-overlaps within some 1.75 range or something goes here
                if overpass:
                    diff = nm - om
                    #sometimes a charge registers as one thing but the value is way closer to +/-1
                    initialcharge = round(proton / diff)
                    if initialcharge > 1: #nothing close enough to care about for 1, plus the first bit would go to 0 and cause annoying zerodivision warnings
                        chargespread = np.linspace(initialcharge - 1, initialcharge + 1, 3)
                        expspread = proton / chargespread
                        minexpind = np.abs(diff - expspread).argmin()
                        charge = int(chargespread[minexpind])
                    else:
                        charge = initialcharge
                    actualmass = om * charge - proton * charge
                    #min/max peptide mass from the isotope database, anything with a basemmass larger than that max shouldn't be considered, it would help keep out ridiculously large charges - instead having them be a part of nodists, the way they probably should
                    if actualmass <= uppermasslimit:
                        expdiff = proton / charge
                        acdiff = expdiff - diff
                        diffcut = expdiff * chargetolerance
                        if acdiff > -1 * (diffcut * chargetolerance + masswidthlimit):
                        #^a lighter acceptance for distances above the charge dist - it does happen from time to time on lower-hanging straggler isotopomers
                            if acdiff <= diffcut + masswidthlimit:
                                absacdiff = abs(acdiff) * charge #normalizing -> distance to proton
                                ointensity = oreg[5]
                                ncons = 0
                                intensitypercdiff = abs(nintensity - ointensity) / (nintensity + ointensity) / 2
                                subcheck = False
                                if okey in latestconnections:
                                    if charge in latestconnections[okey]:
                                        subcheck = True
                                decreasecheck = nintensity < ointensity
                                if subcheck:
                                    csubs = map(int, latestconnections[okey][charge].split(',')[:-1])
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
                                                opairs = pairsbyline[okey].split(',')[:-1]
                                                spinecopy = connectionspine[adi].split(',')[:-1]
                                                spinds = []
                                                for op in opairs:
                                                    if op in spinecopy:
                                                        spinds.append(spinecopy.index(op))
                                                spi = max(spinds)
                                                spinecopy = spinecopy[:spi]
                                                connectionspine[xdi] = ','.join((spinecopy)) + ','
                                                di += 1
                                            connectionspine[xdi] += str(pi) + ','
                                            latestmass[xdi] = nkey
                                            latestconnections[nkey][charge] += str(xdi) + ','
                                            if decreasecheck:
                                                previousdecrease[xdi] = True
                                            ncons += 1
                                else:
                                    #no previous subgroup 
                                    if intensitypercdiff <= steplimit:
                                        connectionspine[di] += str(pi) + ','
                                        latestmass[di] = nkey
                                        latestconnections[nkey][charge] += str(di) + ','
                                        if decreasecheck:
                                            previousdecrease[di] = True
                                        ncons += 1
                                        di += 1
                                if ncons > 0:
                                    opoints = oreg[4]
                                    #dpercdiff = abs(npoints - opoints) / (npoints + opoints) / 2
                                    dpercdiff = abs(nintensity - ointensity) / (nintensity + ointensity) / 2
                                    lpair = (okey, nkey)
                                    pairkeys[pi] = lpair
                                    paircharges[pi] = charge
                                    scorelist = [absacdiff, dpercdiff, percentoverlap, ~decreasecheck]
                                    for p in lpair:
                                        pairsbyline[p] += str(pi) + ','
                                    scoresbypair[pi] = scorelist
                                    if subcheck:
                                        connectionspine[di] += str(pi) + ','
                                        latestmass[di] = nkey
                                        latestconnections[nkey][charge] += str(di) + ','
                                        if decreasecheck:
                                            previousdecrease[di] = True
                                        di += 1
                                    pi += 1
            else:
                #om is past proton distance, remove om from mass pool
                masspoolremovals.append(okey)
        for mpr in masspoolremovals:
            masspool.remove(mpr)
            if mpr in latestconnections:
                for charge, cons in latestconnections[mpr].items():
                    for con in map(int, cons.split(',')[:-1]):
                        if mpr == latestmass[con]:
                            del latestmass[con]
                            try:
                                del previousdecrease[con]
                            except KeyError: #no previous decrease, idc really
                                pass
                del latestconnections[mpr]
        masspool.append(nkey)
    
    del latestconnections
    del previousdecrease
    del latestmass
    
    print(time() - t1, 'mass processing')
    print('connectionspine -', len(connectionspine))
    t2 = time()
    
    flatdistgroups = set() #a tuple of every potential distribution
    for distkey, pairspine in connectionspine.items(): #this loop needs to write its massive outputs to disk and import them later on from a different function
        pairspine = map(int, pairspine.split(',')[:-1])
        maximalsuperset = []
        activepairlist = []
        scores = []
        for pairs in pairspine:
            activepairlist.append(pairs)
            scores.append(scoresbypair[pairs])
            flatdist = []
            competinglines = set()
            for pair in activepairlist:
                pk = pairkeys[pair]
                flatdist.extend(pk)
            rbounds = regions[flatdist,2:4]
            rstack = overlap_counts(rbounds.tolist())
            if rstack > 0:
                flatdistgroups.add(tuple(activepairlist))
    
    print(time() - t2, 'distribution scoring')
    print('flatdistgroups', len(flatdistgroups))
    t3  = time()
    
    dr =  0 #easier to work with dr here rather than re-index the finalized distribution just to get the max index for finaldefiniteind afterwards, the order of these aren't important
    multidists = defaultdict(dict) #charge: distid: lines
    for dist in flatdistgroups:
        charge = paircharges[dist[0]]
        lines = chain.from_iterable(map(pairkeys.get, dist))
        multidists[charge][dr] = list(lines)
        dr += 1
    finaldefiniteind = dr
    
    print(time() - t3, 'distribution assembling')
    print(sum(len(i) for i in multidists.values()), 'initial distributions')

    distributionmasses = {} #distid: ordered masses
    distributioncharges = {} #distid: charge
    distributionsoflines = defaultdict(list) #line: distid
    distributionsoflinemasks = {} #linemask: distid
    linesofdistributions = {} #distid: mass-ordered linedkeys
    linemasksofdistributions = {} #distid: mass-ordered linemasks
    linemasksbylinedistributions = defaultdict(dict) #distid: mass-ordered lines: linemask
    chargesoflinemasks = {} #linemask: charge
    distributionintensities = {} #distid: mass-ordered intensities
    linesbylinemask = {} #linemask: line
    mask = 0
    for charge, dists in multidists.items():
        for dist, lines in dists.items():
            dmasses = regions[lines,7]
            lineorder = dmasses.argsort().tolist()
            sortedlines = [lines[i] for i in lineorder]
            sortedmasses = regions[sortedlines,7]
            dintensities = regions[sortedlines,5]
            masses = (sortedmasses * charge) - (proton * charge)
            distributionmasses[dist] = masses
            distributioncharges[dist] = charge
            sortedmasks = []
            for line in lines:
                linesbylinemask[mask] = line
                distributionsoflines[line].append(dist)
                distributionsoflinemasks[mask] = dist
                linemasksbylinedistributions[dist][line] = mask
                chargesoflinemasks[mask] = charge
                sortedmasks.append(mask)
                mask += 1
            linesofdistributions[dist] = sortedlines
            linemasksofdistributions[dist] = sortedmasks
            distributionintensities[dist] = dintensities
    
    multidistfile = ''.join((processingdirectory, 'multidists.pickle'))
    with open(multidistfile, 'wb') as pick:
        pickle.dump(multidists, pick)
    
    finalindfile = ''.join((processingdirectory, 'finalindex.pickle'))
    with open(finalindfile, 'wb') as pick:
        pickle.dump(finaldefiniteind, pick)
    
    distributionmassesfile = ''.join((processingdirectory, 'distributionmasses.pickle'))
    with open(distributionmassesfile, 'wb') as pick:
        pickle.dump(distributionmasses, pick)
    #distributionmasses = {} #distid: ordered masses
    
    distributionintensitiesfile = ''.join((processingdirectory, 'distributionintensities.pickle'))
    with open(distributionintensitiesfile, 'wb') as pick:
        pickle.dump(distributionintensities, pick)
    #distributionintensities = {} #distid: mass-ordered intensities
    
    distributionchargesfile = ''.join((processingdirectory, 'distributioncharges.pickle'))
    with open(distributionchargesfile, 'wb') as pick:
        pickle.dump(distributioncharges, pick)
    #distributioncharges = {} #distid: charge
    
    distributionsoflinesfile = ''.join((processingdirectory, 'distributionsoflines.pickle'))
    with open(distributionsoflinesfile, 'wb') as pick:
        pickle.dump(distributionsoflines, pick)
    #distributionsoflines = defaultdict(list) #line: distid
    
    distributionsoflinemasksfile = ''.join((processingdirectory, 'distributionsoflinemasks.pickle'))
    with open(distributionsoflinemasksfile, 'wb') as pick:
        pickle.dump(distributionsoflinemasks, pick)
    #distributionsoflinemasks = {} #linemask: distid
    
    linesofdistributionsfile = ''.join((processingdirectory, 'linesofdistributions.pickle'))
    with open(linesofdistributionsfile, 'wb') as pick:
        pickle.dump(linesofdistributions, pick)
    #linesofdistributions = {} #distid: mass-ordered linedkeys
    
    linemasksofdistributionsfile = ''.join((processingdirectory, 'linemasksofdistributions.pickle'))
    with open(linemasksofdistributionsfile, 'wb') as pick:
        pickle.dump(linemasksofdistributions, pick)
    #linemasksofdistributions = {} #distid: mass-ordered linemasks
    
    linemasksbylinedistributionsfile = ''.join((processingdirectory, 'linemasksbylinedistributions.pickle'))
    with open(linemasksbylinedistributionsfile, 'wb') as pick:
        pickle.dump(linemasksbylinedistributions, pick)
    #linemasksbylinedistributions = defaultdict(dict) #distid: mass-ordered lines: linemask
    
    linesbylinemaskfile = ''.join((processingdirectory, 'linesbylinemask.pickle'))
    with open(linesbylinemaskfile, 'wb') as pick:
        pickle.dump(linesbylinemask, pick)
    #linesbylinemask = defaultdict(dict) #distid: mass-ordered lines: linemask
    
    chargesoflinemasksfile = ''.join((processingdirectory, 'chargesoflinemasks.pickle'))
    with open(chargesoflinemasksfile, 'wb') as pick:
        pickle.dump(chargesoflinemasks, pick)
    #chargesoflinemasks = defaultdict(dict) #distid: mass-ordered lines: linemask
