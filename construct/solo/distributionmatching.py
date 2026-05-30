from generalfunctions import radius_neighbors
from database import environment

from collections import defaultdict
from itertools import repeat
from time import time
import numpy as np
import pickle

def distribution_matching(ppmtolerance, librarylocation, proteome, processingdirectory):
    #formalized analyte information, summarizing all distributions across any charge states
    t2 = time()
    
    scansbyanalytefile = ''.join((processingdirectory, 'scansbyanalyte.pickle'))
    with open(scansbyanalytefile, 'rb') as pick:
        scansbyanalyte = pickle.load(pick)
    #scansbyanalyte = defaultdict(list) #analyteid: [spectra across all lines and charge states]
    
    scansoflinesfile = ''.join((processingdirectory, 'scansoflines.pickle'))
    with open(scansoflinesfile, 'rb') as pick:
        scansoflines = pickle.load(pick)
    #scansoflines = defaultdict(list) #lineuid: [ms2 scan indices]
    
    linesofscansfile = ''.join((processingdirectory, 'linesofscans.pickle'))
    with open(linesofscansfile, 'rb') as pick:
        linesofscans = pickle.load(pick)
    #linesofscans = defaultdict(list) #ms2 scan index: [lineuids within window]
    
    analytefile = ''.join((processingdirectory, 'analytefactors.pickle'))
    with open(analytefile, 'rb') as pick:
        analytekeys, analytedistributions, analytesbydistribution, distributionsoflines, linesofdistributions, linesofanalytes = pickle.load(pick)
    #analytekeys = defaultdict(dict) #analyte id: distid: charge, this can be one or multiple distributions based on whether it has multiple charge states
    #analytedistributions = defaultdict(dict) #analyte id: ordered masses: AUC of merged isotopomers, weighted means [via intensity] across isotopomers from every charge state, isotopomer datapoints merged across each charge state - if there are any
    #analytesbydistribution = {} #distid: analyte id

    sumabundances = {} #formula: [sum abundance dist]
    maxabundances = {} #formula: [full abundance dist]
    with environment(librarylocation) as env:
        formuladb = env.open_db(('proteomes.formulalist').encode())
        with env.begin(write=False) as txn:
            with txn.cursor(formuladb) as cursor:
                pulledformulas = eval(cursor.get(proteome.encode()).decode())
        getkeys = [i.encode() for i in pulledformulas]
        sums = env.open_db('distributions.sum'.encode())
        with env.begin(write=False) as txn:
            with txn.cursor(sums) as cursor:
                for k, v in cursor.getmulti(getkeys):
                    out = np.frombuffer(v)
                    out = out.reshape(2, out.size//2)
                    sumabundances[k.decode()] = out
        maxes = env.open_db('distributions.max'.encode())
        with env.begin(write=False) as txn:
            with txn.cursor(maxes) as cursor:
                for k, v in cursor.getmulti(getkeys):
                    out = np.frombuffer(v)
                    maxabundances[k.decode()] = out

    librarykeys = []
    librarymasses = []
    librarymassdict = {} #lid: [masses]
    librarymaxranks = {} #n: [max intensity ranks]
    librarypositions = {} #lid: [indices]
    libraryintensityranks = {} #lid: [intensityranks]
    libraryidentifiers = {} #n: formula
    for n, (f, (masses, intensities)) in enumerate(sumabundances.items()):
        libraryidentifiers[n] = f
        librarymassdict[n] = masses
        librarypositions[n] = list(range(masses.size))
        intensityranks = np.abs(intensities.argsort().argsort() - intensities.size + 1)
        libraryintensityranks[n] = intensityranks
        librarykeys.extend(repeat(n, masses.size))
        librarymasses.extend(masses.tolist())
        
        maxes = maxabundances[f]
        maxintensityranks = np.abs(maxes.argsort().argsort() - maxes.size + 1)
        librarymaxranks[n] = maxintensityranks
    
    librarykeys = np.array(librarykeys)
    librarymasses = np.array(librarymasses)
    
    librarykeys = librarykeys[librarymasses.argsort()]
    librarymasses = np.sort(librarymasses)
    
    distributionkeys = []
    distributionmasses = []
    distributionmassdict = {} #did: [masses]
    distributionintensityranks = {} #did: [intensityranks]
    for k, (masses, intensities) in analytedistributions.items():
        intensityranks = np.abs(intensities.argsort().argsort() - intensities.size + 1)
        distributionmassdict[k] = masses
        distributionintensityranks[k] = intensityranks
        distributionkeys.extend(repeat(k, masses.size))
        distributionmasses.extend(masses.tolist())
    
    distributionkeys = np.array(distributionkeys)
    distributionmasses = np.array(distributionmasses)
    
    distributionkeys  = distributionkeys[distributionmasses.argsort()]
    distributionmasses = np.sort(distributionmasses)
    
    ppmmod = ppmtolerance / 1000000
    
    matches = radius_neighbors(librarymasses.tolist(), distributionmasses.tolist(), ppmmod)

    matchorganizer = defaultdict(list)
    for k, lkeys in matches.items():
        dk = distributionkeys[k]
        matchorganizer[dk].extend(librarykeys[lkeys])

    for k in list(matchorganizer):
        matchorganizer[k] = np.array(list(set(matchorganizer[k])))
    
    linepositionsbyformula = defaultdict(lambda: defaultdict(set)) #formula: matchable library positions of a distribution: [lines that have ms2 scans]
    lmatches = 0
    dmatches = 0
    for dk, lkeys in matchorganizer.items():
        if dk in scansbyanalyte:
            dmasses = distributionmassdict[dk]
            dsize = dmasses.size
            tx = 0
            for lk in lkeys.tolist():
                lmasses = librarymassdict[lk]
                lsize = lmasses.size
                
                leftoffset = int(round(lmasses.tolist()[0] - dmasses.tolist()[0]))
                if leftoffset > 0:
                    li = 0
                    rmax = dsize - leftoffset
                    maxsize = min(lsize, rmax)
                    ri = leftoffset
                elif leftoffset == 0:
                    li = 0
                    ri = 0
                    maxsize = min(lsize, dsize)
                else: #< 0
                    li = -leftoffset
                    ri = 0
                    lmax = lsize - li
                    maxsize = min(lmax, dsize)
                le = li + maxsize
                lrange = le - li
                #if lrange > 1: #at least 2 matches
                sumlintranks = libraryintensityranks[lk][li:le]
                maxlintranks = librarymaxranks[lk][li:le]
                if 0 in sumlintranks or 0 in maxlintranks: #the top library rank is included
                    re = ri + maxsize
                    dorders = distributionintensityranks[dk][ri:re].tolist()
                    lsorders = sumlintranks.tolist()
                    lmorders = maxlintranks.tolist()
                    sdorderdiffs = [abs(i-j) for i, j in zip(dorders, lsorders)] #library sum to dist
                    mdorderdiffs = [abs(i-j) for i, j in zip(dorders, lmorders)] #dist to library max
                    lmorderdiffs = [abs(i-j) for i, j in zip(lmorders, lsorders)] #library sum to max
                    sdallowance = sum(sdorderdiffs)
                    mdallowance = sum(mdorderdiffs)
                    liballowance = sum(lmorderdiffs)
                    #if allowance == 0: #complete heirarchical match
                    if sdallowance <= liballowance or mdallowance <= liballowance:
                        distlines = linesofanalytes[dk][ri:re]
                        positions = librarypositions[lk][li:le]
                        formula = libraryidentifiers[lk]
                        for lines, pos in zip(distlines, positions):
                            for line in lines:
                                if line in scansoflines:
                                    linepositionsbyformula[formula][pos].add(line)
                        tx += 1
            if tx > 0:
                lmatches += tx
                dmatches += 1
    
    for k, v in linepositionsbyformula.items():
        for sk, sv in v.items():
            v[sk] = tuple(sv)
        linepositionsbyformula[k] = dict(v)
    linepositionsbyformula = dict(linepositionsbyformula)

    print(time() - t2, 'matches assembled')
    print('library matches:', lmatches)
    print('distribution matches:', dmatches)
    
    linepositionsbyformulafile = ''.join((processingdirectory, 'linepositionsbyformula.pickle'))
    with open(linepositionsbyformulafile, 'wb') as pick:
        pickle.dump(linepositionsbyformula, pick)
