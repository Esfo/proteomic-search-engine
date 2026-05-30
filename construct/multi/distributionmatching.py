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
    
    scansoflinesfile = ''.join((processingdirectory, 'scansoflines.pickle'))
    with open(scansoflinesfile, 'rb') as pick:
        scansoflines = pickle.load(pick)
    #scansoflines = defaultdict(list) #lineuid: [ms2 scan indices]
    
    linesbylinemaskfile = ''.join((processingdirectory, 'linesbylinemask.pickle'))
    with open(linesbylinemaskfile, 'rb') as pick:
        linesbylinemask = pickle.load(pick)
    #linesbylinemasks = defaultdict(dict) #distid: mass-ordered lines: linemask
    
    distributionmassesfile = ''.join((processingdirectory, 'distributionmasses.pickle'))
    with open(distributionmassesfile, 'rb') as pick:
        distributionmasses = pickle.load(pick)
    #distributionmasses = {} #distid: ordered masses
    
    distributionswithscansfile = ''.join((processingdirectory, 'distributionswithscans.pickle'))
    with open(distributionswithscansfile, 'rb') as pick:
        distributionswithscans = pickle.load(pick)
    #distributionswithscans = set() #distributions with an MS2 scan
    
    distributionchargesfile = ''.join((processingdirectory, 'distributioncharges.pickle'))
    with open(distributionchargesfile, 'rb') as pick:
        distributioncharges = pickle.load(pick)
    #distributioncharges = {} #distid: charge
    
    distributionintensitiesfile = ''.join((processingdirectory, 'distributionintensities.pickle'))
    with open(distributionintensitiesfile, 'rb') as pick:
        distributionintensities = pickle.load(pick)
    #distributionintensities = {} #distid: mass-ordered intensities
    
    linemasksofdistributionsfile = ''.join((processingdirectory, 'linemasksofdistributions.pickle'))
    with open(linemasksofdistributionsfile, 'rb') as pick:
        linemasksofdistributions = pickle.load(pick)
    #linemasksofdistributions = {} #distid: mass-ordered linemasks

    maxabundances = {}
    sumabundances = {}
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
    distributionmasslist = []
    distributionintensityranks = {} #did: [intensityranks]
    for k, masses in distributionmasses.items():
        charge = distributioncharges[k]
        intensities = distributionintensities[k]
        intensityranks = np.abs(intensities.argsort().argsort() - intensities.size + 1)
        distributionintensityranks[k] = intensityranks
        distributionkeys.extend(repeat(k, masses.size))
        distributionmasslist.extend(masses.tolist())
    
    distributionkeys = np.array(distributionkeys)
    distributionmasslist = np.array(distributionmasslist)
    
    distributionkeys  = distributionkeys[distributionmasslist.argsort()]
    distributionmasslist = np.sort(distributionmasslist)
    
    ppmmod = ppmtolerance / 1000000
    
    matches = radius_neighbors(librarymasses.tolist(), distributionmasslist.tolist(), ppmmod)
    
    matchorganizer = defaultdict(list) #distributionkeys: [library formulas]
    for k, lkeys in matches.items():
        dk = distributionkeys[k]
        matchorganizer[dk].extend(librarykeys[lkeys])

    for k in list(matchorganizer):
        matchorganizer[k] = np.array(list(set(matchorganizer[k])))
    
    linemaskpositionsbyformula = defaultdict(lambda: defaultdict(set)) #formula: matchable library positions of a distribution: [lines that have ms2 scans]
    lmatches = 0
    dmatches = 0
    for dk, lkeys in matchorganizer.items():
        if dk in distributionswithscans:
            dmasses = distributionmasses[dk]
            dsize = dmasses.size
            tx = 0
            for lk in lkeys.tolist(): #can this loop be made concurrent?
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
                    #lints = libraryintensities[lk][li:le].tolist()
                    #dsum = sum(dints)
                    #lsum = sum(lints)
                    #dnorm = [i / dsum for i in dints]
                    #lnorm = [i / lsum for i in lints]
                    #intensitydiff = [d - l for d, l in zip(dnorm, lnorm)]
                    #idmean = sum(intensitydiff) / maxsize
                    #intensitydiffs = [abs(idmean - i) for i in intensitydiff]
                    #meanintensitydiff = sum(intensitydiffs) / maxsize
                    #if allowance == 0: #complete heirarchical match
                    if sdallowance <= liballowance or mdallowance <= liballowance:
                        #zeroscores.append(meanintensitydiff)
                        #librarymatchesbydistribution[dk].append(lk)
                        #distlines = linesofdistributions[dk][ri:re]
                        distlinemasks = linemasksofdistributions[dk][ri:re]
                        positions = librarypositions[lk][li:le]
                        formula = libraryidentifiers[lk]
                        for linemask, pos in zip(distlinemasks, positions):
                            if linesbylinemask[linemask] in scansoflines:
                                linemaskpositionsbyformula[formula][pos].add(linemask)
                        tx += 1
                    #else:
                    #    #keep scores of other distributions
                    #    nonzeroscores.append(meanintensitydiff)
            if tx > 0:
                lmatches += tx
                dmatches += 1
    
    for k, v in linemaskpositionsbyformula.items():
        for sk, sv in v.items():
            v[sk] = tuple(sv)
        linemaskpositionsbyformula[k] = dict(v)
    linemaskpositionsbyformula = dict(linemaskpositionsbyformula)

    print(time() - t2, 'matches assembled')
    print('library matches:', lmatches)
    print('distribution matches:', dmatches)
    
    linemaskpositionsbyformulafile = ''.join((processingdirectory, 'linemaskpositionsbyformula.pickle'))
    with open(linemaskpositionsbyformulafile, 'wb') as pick:
        pickle.dump(linemaskpositionsbyformula, pick)
