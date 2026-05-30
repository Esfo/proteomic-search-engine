from database import environment

from collections import defaultdict
from pyteomics import mzml
from time import time
import numpy as np
import pickle

def coordinate_generation(scan):
    if scan['ms level'] == 2:
        precursorinfo = scan['precursorList']['precursor'][0]
        selectionwindow = precursorinfo['isolationWindow']
        precmass = selectionwindow['isolation window target m/z'].real
        lowerbound = precmass - selectionwindow['isolation window lower offset'].real
        upperbound = precmass + selectionwindow['isolation window upper offset'].real
        scanlist = scan['scanList']['scan'][0]
        rt = scanlist['scan start time'].real
        scindex = int(scan['index'])
        coordinates = [rt, lowerbound, upperbound, scindex]
        return coordinates
 

def scan_matching(mzmlfile, minpoints, nprocs, librarylocation, processingdirectory, proteome):
    analytefile = ''.join((processingdirectory, 'analytefactors.pickle'))
    with open(analytefile, 'rb') as pick:
        analytekeys, analytedistributions, analytesbydistribution, distributionsoflines, linesofdistributions, linesofanalytes = pickle.load(pick)
    #analytekeys = defaultdict(dict) #analyte id: distid: charge, this can be one or multiple distributions based on whether it has multiple charge states
    #analytedistributions = {} #analyte id: [[weighted means [via intensity] across isotopomers from every charge state if there are multiple], [AUC of merged isotopomers]]
    #analytesbydistribution = {} #distid: analyte id
    #distributionsoflines: lineuid: distid
    #linesofdistributions: distid: [lineuids ordered by mass]

    regionfile = ''.join((processingdirectory, 'regions.pickle'))
    with open(regionfile, 'rb') as pick:
        regions = pickle.load(pick)
    #regions as [minmass, maxmass, mintime, maxtime, # datapoints, peakarea, maxintensity, wmean, lineid]
    regiter = regions[regions[:,4] >= minpoints]

    loaderloc = ''.join((processingdirectory, 'trackedgroups.pickle'))
    with open(loaderloc, 'rb') as pick:
        trackedgroups = pickle.load(pick)
    
    msrun = mzml.MzML(mzmlfile, dtype=np.float64)
    
    t1 = time()

    precursorcoordinates = [] #[rt of previous ms1, lower mass bound, upper mass bound, ms2 scan index]
    for output in msrun.map(lambda scan: coordinate_generation(scan), processes=nprocs):
        match output:
            case list():
                coords = output
                precursorcoordinates.append(coords)

    print(time() - t1, 'collected MS2 scan window data')
    t2 = time()

    precursorcoordinates = sorted(precursorcoordinates, key=lambda x: x[0]) #sorted by rt
    regiter = regiter[regiter[:,2].argsort()].tolist() #sorted by starting time

    regioniter = iter(regiter)

    regminrt = -1

    linesofscans = defaultdict(list) #ms2 scan index: [lineuids within window]
    scansoflines = defaultdict(list) #lineuid: [ms2 scan indices]
    
    regpool = []
    for pc in precursorcoordinates:
        prt = pc[0]
        pminmass = pc[1]
        pmaxmass = pc[2]
        precid = pc[3]
        while regminrt < prt: #add more regs to regpool
            #no region rts and precursor rts are the same values, don't need <=
            try:
                reg = next(regioniter)
            except StopIteration: #regiter reached the end before precursorcoordinates, which is within the realm of expectations
                break
            regminrt = reg[2]
            #regmaxrt = reg[3]
            regid = int(reg[8])
            regpool.append(regid)
        regremovals = []
        for r in regpool:
            treg = regions[r]
            trmaxrt = treg[3]
            if trmaxrt < prt:
                regremovals.append(r)
        for r in regremovals:
            regpool.remove(r)
        for r in regpool: #assess reg masses across pc masses
            treg = regions[r]
            trminmass = treg[0]
            trmaxmass = treg[1]
            if pminmass <= trmaxmass and pmaxmass >= trminmass:
                tminrt = treg[2]
                if tminrt < prt:
                    linesofscans[precid].append(r)
                    scansoflines[r].append(precid)

    for scan, lines in linesofscans.items():
        linesofscans[scan] = tuple(sorted(lines))
    linesofscans = dict(linesofscans)

    for k, v in scansoflines.items():
        scansoflines[k] = tuple(v)
    scansoflines = dict(scansoflines)

    print(time() - t2, 'determined line-window overlaps')
    print(len(linesofscans), 'scans with lines')
    print(len(scansoflines), 'lines with scans')
    print(len(set(map(distributionsoflines.get, scansoflines))), 'distributions with scans')
    t3 = time()

    #it might be worth visualizing these to see if these are linemodel errors
    blankscans = len(precursorcoordinates) - len(linesofscans)
    if blankscans > 0:
        blankpercent = blankscans / len(precursorcoordinates)
        print('your instrument produced', blankscans, f'MS2 scans that targeted nothing within the minimum point threshhold of {minpoints},', f'{round(blankpercent, 4)}% of all MS2 scans')

    precursordict = {}
    for pc in precursorcoordinates:
        precursordict[pc[-1]] = pc[:-1]

    #assigning relative intensity %'s of each MS1 distribution for each MS2 window
    scansbyanalyte = defaultdict(list) #analyteid: [scans across all lines and charge states]
    scanalytecharges = defaultdict(dict) #analyteid: scan: charge
    lineintensitiesofscans = defaultdict(lambda: defaultdict(dict)) #scan: line: raw area of ms1 line, average of 2 flanking points
    linepercentagesofscans = defaultdict(lambda: defaultdict(dict)) #scan: line: % of scan intensity input
    scansums = defaultdict(float) #scan: sum area used in spectralsamplings
    
    #these 3 below are keeping track of which line from which distribution [at each charge] gives the most intense MS2 sampling based on MS1 intensity
    maxintensitysampleofdists = defaultdict(float) #distid: max intensity
    maxintensitylinesofdists = {} #distid: (line, scan)
    premaxsampledistributionsoflines = {} #distid: line
    
    for line, scans in scansoflines.items():
        try:
            distid = distributionsoflines[line]
            analyteid = analytesbydistribution[distid]
            charge = analytekeys[analyteid][distid]
        except KeyError: #line is in nodists
            analyteid = -line
            distid = -1
            charge = 0
        scansbyanalyte[analyteid].extend(scans)
        linegroup = trackedgroups[line]
        linetimes = linegroup[:,1]
        linemasses = linegroup[:,0]
        lineintensity = linegroup[:,2]
        lmax = linemasses.max()
        lmin = linemasses.min()
        for scan in scans:
            pcoords = precursordict[scan]
            rt = pcoords[0]
            #this assumes there is a left and right intensity, i guess there would be though?
            #PROBLEM here, this assumes the time difference is the same between the two
            #^there should be a time-based extrapolation here
            #i'm leaving it for later because its simple and probably won't change much
            leftintensity = lineintensity[linetimes < rt][-1]
            rightintensity = lineintensity[linetimes > rt][0]
            sampleintensity = (leftintensity + rightintensity) / 2
            minmass = pcoords[1]
            maxmass = pcoords[2]
            if not lmin > minmass and lmax < maxmass:
                #if the overlap doesn't fully encompass all mass points, normalize by the % mass overlap
                #idc for slight mass shifts, i'm just going by the range, the shifts would be too annoying to incorporate, probably not worth my time
                #this assumes there's no realistic way for the line mass to fully encompass the scans mass window, which there shouldn't be unless the line model screws up
                if lmax > maxmass:
                    percentoverlap = (maxmass - lmin) / (lmax - lmin)
                else:
                    percentoverlap = (lmax - minmass) / (lmax - lmin)
                sampleintensity *= percentoverlap
            if distid >= 0:
                if sampleintensity > maxintensitysampleofdists[distid]:
                    maxintensitysampleofdists[distid] = sampleintensity
                    maxintensitylinesofdists[distid] = line, scan
                    premaxsampledistributionsoflines[distid] = line
            lineintensitiesofscans[scan][line] = sampleintensity
            scansums[scan] += sampleintensity
            scanalytecharges[analyteid][scan] = charge
    scansbyanalyte = dict(scansbyanalyte)
    
    #turning areas into percents
    for scan, lines in lineintensitiesofscans.items():
        for line in lines:
            linepercentagesofscans[scan][line] = lineintensitiesofscans[scan][line] / scansums[scan]
        linepercentagesofscans[scan] = dict(linepercentagesofscans[scan])
        lineintensitiesofscans[scan] = dict(lines) #can't pickle double default dicts
    lineintensitiesofscans = dict(lineintensitiesofscans)
    linepercentagesofscans = dict(linepercentagesofscans)
    
    maxsampledistributionsoflines = {} #line: distid
    for distid, line in premaxsampledistributionsoflines.items():
        maxsampledistributionsoflines[line] = distid
    
    print(time() - t3, 'weighted scan window hits by intensity')
    
    scansoflinesfile = ''.join((processingdirectory, 'scansoflines.pickle'))
    with open(scansoflinesfile, 'wb') as pick:
        pickle.dump(scansoflines, pick)
    
    linesofscansfile = ''.join((processingdirectory, 'linesofscans.pickle'))
    with open(linesofscansfile, 'wb') as pick:
        pickle.dump(linesofscans, pick)
    
    scansbyanalytefile = ''.join((processingdirectory, 'scansbyanalyte.pickle'))
    with open(scansbyanalytefile, 'wb') as pick:
        pickle.dump(scansbyanalyte, pick)
    
    scanalytefile = ''.join((processingdirectory, 'scanalytes.pickle'))
    with open(scanalytefile, 'wb') as pick:
        pickle.dump(scanalytecharges, pick)
    
    lineintensitiesofscansfile = ''.join((processingdirectory, 'lineintensitiesofscans.pickle'))
    with open(lineintensitiesofscansfile, 'wb') as pick:
        pickle.dump(lineintensitiesofscans, pick)

    linepercentagesofscansfile = ''.join((processingdirectory, 'linepercentagesofscans.pickle'))
    with open(linepercentagesofscansfile, 'wb') as pick:
        pickle.dump(linepercentagesofscans, pick)
    
    maxintensitylinesofdistsfile = ''.join((processingdirectory, 'maxintensitylinesofdists.pickle'))
    with open(maxintensitylinesofdistsfile, 'wb') as pick:
        pickle.dump(maxintensitylinesofdists, pick)
    #maxintensitylinesofdists = {} #distid: (line, scan)
    
    maxsampledistributionsoflinesfile = ''.join((processingdirectory, 'maxsampledistributionsoflines.pickle'))
    with open(maxsampledistributionsoflinesfile, 'wb') as pick:
        pickle.dump(maxsampledistributionsoflines, pick)
    #maxsampledistributionsoflines = {} #line: distid
