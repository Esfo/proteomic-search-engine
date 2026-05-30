from generalfunctions import intersection_merge

from collections import Counter, defaultdict
from itertools import chain, repeat
import multiprocessing as mp
from pyteomics import mzml
from scipy import spatial
from time import time
import numpy as np
import pickle

def boundary_calculation(group):
    mainind = 0
    primarytomainindex = {} #primary index: main index
    maintoprimaryindex = defaultdict(list) #main index: [primary indices]
    mainindicesbyscan = defaultdict(list) #scan: [main indices], previously known as scandict
    scansbymainindices = defaultdict(set)
    scans, masses, primaryinds = [], [], []
    for scan in group:
        scanmasses, primaries = massesandindices[scan]
        scanlist = repeat(scan, len(scanmasses))
        primaryinds.extend(primaries)
        masses.extend(scanmasses)
        scans.extend(scanlist)
    masses = np.array(masses)[:,None]
    radius = (masses * ppmtolerance).flatten() / 1000000
    nn = spatial.KDTree(masses)
    matches = nn.query_ball_point(masses, radius).tolist() #keeping this as the functions output is easier to work with than my dict output for this
    groupableinds = list(map(tuple, intersection_merge(matches)))
    for gi in groupableinds:
        for g in gi:
            primaryind = primaryinds[g]
            primarytomainindex[primaryind] = mainind
            maintoprimaryindex[mainind].append(primaryind)
            scan = scansbyprimaryind[primaryind]
            mainindicesbyscan[scan].append(mainind)
            scansbymainindices[mainind].add(scan)
        mainind += 1
    
    ms1entropy = defaultdict(lambda: Counter()) #ms2line: line: count
    #compare scansbymainindices to linesofscans
    for mainindex, scans in scansbymainindices.items():
        for scan in scans:
            ms1lines = linesofscans[scan]
            for line in ms1lines:
                ms1scans = set(scansoflines[line])
                ms1diffs = len(ms1scans.difference(scans))
                ms1entropy[mainindex][line] -= ms1diffs
    
    assignmentresults = {}
    for mainindex, assignablems1lines in ms1entropy.items():
        primaries = maintoprimaryindex[mainindex]
        for primary in primaries:
            #get scan: scansbyprimaryind -> get lines
            scan = scansbyprimaryind[primary]
            assessablems1lines = linesofscans[scan]
            assessabledict = Counter()
            for a in assessablems1lines:
                if a in assignablems1lines:
                    assessabledict[a] = assignablems1lines[a]
            assessablerankings = assessabledict.most_common(len(assessabledict))
            if len(assessablerankings) > 1:
                if assessablerankings[0][1] == assessablerankings[1][1]:
                    #find all top ranks
                    toprank = assessablerankings[0][1]
                    toplines = []
                    for ms1line, rank in assessablerankings:
                        if rank == toprank:
                            toplines.append(ms1line)
                        else:
                            break
                    toplines = tuple(toplines)
                    assignmentresults[primary] = toplines
                else:
                    #lone top rank
                    #assign primary to ms1 line
                    assignmentresults[primary] = assessablerankings[0][0]
            else:
                #lone top rank
                #assign primary to ms1 line
                assignmentresults[primary] = assessablerankings[0][0]
    return assignmentresults

def entropy_competition(mzmlfile, processingdirectory, ppmtol, nprocs):
    #this just WOULDN'T multiprocess right from within a class
    #it makes zero sense in my head
    #it would be as slow as a linear, non-multiprocessed approach and use 800x the memory
    #im clueless as to why, so im just using globals
    global ppmtolerance
    ppmtolerance = ppmtol
    
    nt = time()
    
    linesofscansfile = ''.join((processingdirectory, 'linesofscans.pickle'))
    with open(linesofscansfile, 'rb') as pick:
        global linesofscans
        linesofscans = pickle.load(pick)
    #linesofscans = defaultdict(list) #ms2 scan index: [lineuids within window]

    scansoflinesfile = ''.join((processingdirectory, 'scansoflines.pickle'))
    with open(scansoflinesfile, 'rb') as pick:
        global scansoflines
        scansoflines = pickle.load(pick)
    #scansoflines = defaultdict(list) #line: [ms2 scan indices]

    analytefile = ''.join((processingdirectory, 'analytefactors.pickle'))
    with open(analytefile, 'rb') as pick:
        distributionsoflines, linesofdistributions = pickle.load(pick)[3:5]
    #distributionsoflines: lineuid: distid
    #linesofdistributions: distid: [lineuids ordered by mass]

    mslevelfile = ''.join((processingdirectory, 'centroid.ms2.pickle'))
    with open(mslevelfile, 'rb') as pick:
        ms2scans = pickle.load(pick)

    mergedlines = intersection_merge(linesofscans.values())
    mergedscans = [tuple(set(chain(*[scansoflines[j] for j in i]))) for i in mergedlines]

    print(len(list(chain.from_iterable(mergedscans))), 'relevant scans')
    print(len(mergedscans), 'scan groups total')
    print(time() - nt, 'organizing scangroups')
    nt = time()

    #msrun = mzml.MzML(mzmlfile, dtype=np.float64)

    primarycount = 0
    global massesandindices
    massesandindices = {} #scan: [[masses], [primary indices]]
    global scansbyprimaryind
    scansbyprimaryind = {} #primary: scan
    indexofprimaryinds = {} #primary: mass index in scan
    primariesofscansbyindex = defaultdict(dict) #scan: index: primary
    #for scan in msrun:
    for scanindex in ms2scans:
        #scanindex = scan['index']
        if scanindex in linesofscans:
            scanmasses, intensities = ms2scans[scanindex].values()
            scanmasses = scanmasses.tolist()
            mlen = len(scanmasses)
            scanlist = repeat(scanindex, mlen)
            inds = np.arange(mlen)
            primaries = (inds + primarycount).tolist()
            inds = inds.tolist()
            primariesofscansbyindex[scanindex].update(zip(inds, primaries))
            scansbyprimaryind.update(zip(primaries, scanlist))
            indexofprimaryinds.update(zip(primaries, inds))
            massesandindices[scanindex] = [scanmasses, primaries]
            primarycount += mlen
    
    print(time() - nt, 'primary mass indexing')
    nt = time()
    
    assignmentresults = {}
    with mp.Pool(nprocs) as pool:
        for output in pool.map(boundary_calculation, mergedscans):
            assignmentresults.update(output)
    
    #print(len(assignmentresults), 'vs.', len(scansbyprimaryind))
    print(time() - nt, 'entropic boundaries')
    
    assignmentresultsfile = ''.join((processingdirectory, 'assignmentresults.pickle'))
    with open(assignmentresultsfile, 'wb') as pick:
        pickle.dump(assignmentresults, pick)
    
    primariesofscansbyindexfile = ''.join((processingdirectory, 'primariesofscansbyindex.pickle'))
    with open(primariesofscansbyindexfile, 'wb') as pick:
        pickle.dump(primariesofscansbyindex, pick)
    
    indexofprimaryindsfile = ''.join((processingdirectory, 'indexofprimaryinds.pickle'))
    with open(indexofprimaryindsfile, 'wb') as pick:
        pickle.dump(indexofprimaryinds, pick)
    
    scansbyprimaryindfile = ''.join((processingdirectory, 'scansbyprimaryind.pickle'))
    with open(scansbyprimaryindfile, 'wb') as pick:
        pickle.dump(scansbyprimaryind, pick)
