from elementalcomponents import proton
from database import environment

from itertools import chain
from time import time
import numpy as np
import pickle

def scan_scoring(ppmtolerance, processingdirectory, proteome, librarylocation):
    with environment(librarylocation) as env:
        proteomedb = env.open_db((proteome + '.info').encode())
        with env.begin(write=False) as txn:
            with txn.cursor(proteomedb) as cursor:
                aminoacidcomposition = eval(cursor.get('aminoacidcomposition'.encode()).decode())
                monoisotopickeys = eval(cursor.get('monoisotopickeys'.encode()).decode())
                elementalmasses = eval(cursor.get('elementalmasses'.encode()).decode())
    
    mslevelfile = ''.join((processingdirectory, 'centroid.ms2.pickle'))
    with open(mslevelfile, 'rb') as pick:
        ms2scans = pickle.load(pick)
    #scanmasses = {} #scan: [[masses], [intensities]]
    
    linesbyscanbysubformulafile = ''.join((processingdirectory, 'linesbyscanbysubformula.pickle'))
    with open(linesbyscanbysubformulafile, 'rb') as pick:
        linesbyscanbysubformula = pickle.load(pick)
    #linesbyscanbysubformula = {} #subformula: scan: [lines]
    
    nt = time()
    
    scanlist = set()
    for subformula, scans in linesbyscanbysubformula.items():
        scanlist.update(scans)
    
    scoredms2scans = {} #scan: [[masses], [intensities]]
    for scan in list(scanlist):
        scoredms2scans[scan] = [ms2scans[scan]['m/z array'], ms2scans[scan]['intensity array']]
    
    aminomasses = {k: sum(elementalmasses[monoisotopickeys[i]] * j for i, j in comp.items()) for k, comp in aminoacidcomposition.items()}
    
    massarray = np.array(list(aminomasses.values()))
    
    #scoring ions of scans:
    #min/max mass range via amino acids for the MS2 range, any AAs can be used
    #use these AAs to then determine a potential delimiter process for dividing the ions into "groups", 100da straight might be too simple
    
    fullmaxmass = 0
    for mza, intensities in scoredms2scans.values():
        if mza.max() > fullmaxmass:
            fullmaxmass = mza.max()
    
    masslevels = []
    masslevels.append(massarray.round())
    for _ in range(np.ceil(round(fullmaxmass) / massarray.min()).astype(int) - 1):
        newlevel = (masslevels[-1] + massarray[:,None]).flatten()
        roundlevel = np.round(newlevel).astype(int)
        newlevel = np.unique(roundlevel)
        masslevels.append(newlevel)
    
    levelranges = [[i.min(), i.max()] for i in masslevels]
    flatranges = np.sort(list(chain.from_iterable(levelranges)))
    #it's going to be worth testing other matrices
    # - cutting levelranges off once it hits maxmass and taking all those indices might work too, limiting to 16 again
    # - check the raw 100 distance
    
    for scan, (mza, intensities) in scoredms2scans.items():
        maxmass = mza.max()
        minmass = mza.min()
        
        scanranges = flatranges[flatranges <= maxmass]
        
        firstind = (scanranges <= minmass).sum() - 1
        scanranges = scanranges[firstind:]
        scanranges[0] = np.floor(minmass)
        scanranges = scanranges.astype(int).tolist()
        
        #i'm hard-coding 16 to be the smallest range, its the most common difference from the matrix of differences of massarray from itself, and it seems like a reasonable minimum i suppose
        while True:
            removal = False
            for n in range(len(scanranges)-1):
                l = scanranges[n]
                r = scanranges[n+1]
                diff = r - l
                if diff < 16:
                    removal = True
                    break
            if removal:
                scanranges.remove(r)
            else:
                break
        
        maxmass = mza.max()
        if maxmass - scanranges[-1] < 16:
            scanranges[-1] = int(np.ceil(maxmass) + 1)
        else:
            scanranges.append(int(np.ceil(maxmass)) + 1)
        
        rangescores = []
        rangebounds = np.stack((scanranges[:-1], scanranges[1:]), axis=1).tolist()
        for n, (l, r) in enumerate(rangebounds):
            secintensities = intensities[np.logical_and(mza >= l, mza < r)]
            if secintensities.size == 1:
                rangescores.append(ppmtolerance / 2) #half credit i suppose
            elif secintensities.size > 1:
                secpercents = secintensities / secintensities.sum()
                #secranks = secpercents.size - secintensities.argsort().argsort()
                #secranksadj = secranks - 1
                #secratios = secranks / (secranks.size - secranksadj)
                #secscores = secratios / secpercents
                normedarray = (secpercents - secpercents.min()) / (secpercents.max() - secpercents.min())
                secscores = normedarray * (ppmtolerance * 2) - ppmtolerance #scaling between -ppmtolerance and ppmtolerance
                rangescores.extend(secscores.tolist())
        scoredms2scans[scan].append(rangescores)
        scoredms2scans[scan] = np.array(scoredms2scans[scan])
    
        #plt.bar(mza, intensities, width=2)
        #plt.vlines(scanranges, ymin=0, ymax=intensities.max(), linewidth=0.5, color='black')
        #plt.show()
    
    print(time() - nt, 'scans scored')
    
    scoredscansfile = ''.join((processingdirectory, 'scored.ms2.pickle'))
    with open(scoredscansfile, 'wb') as pick:
        pickle.dump(scoredms2scans, pick)
    #scoredms2scans = {} #scan: [[masses], [intensities], [ion scores]]
