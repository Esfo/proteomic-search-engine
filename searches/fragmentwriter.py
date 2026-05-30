from collections import Counter, defaultdict
from functools import partial
import multiprocessing as mp
from itertools import chain
from bisect import bisect
from time import time
import numpy as np
import pickle
import heapq
import lmdb
import math
import os

def ms2_loader(processingdirectory, linesbyscanbysubformula):
    #not holding every ms2 scans data if i dont need it
    mslevelfile = ''.join((processingdirectory, 'centroid.ms2.pickle'))
    with open(mslevelfile, 'rb') as pick:
        ms2scans = pickle.load(pick)
    scanlist = set()
    for subformula, scans in linesbyscanbysubformula.items():
        scanlist.update(scans)
    scanmasses = {} #scan: [masses]
    for scan in list(scanlist):
        scanmasses[scan] = ms2scans[scan]['m/z array'].tolist()
    return scanmasses

class FragmentOrganizer:
    def __init__(self, mzmlfile, ions):
        basefolder = '/'.join((mzmlfile.split('/')[:-2]))
        basefile = mzmlfile.split('/')[-1].split('.mzML')[0]
        

        processinglocation = '/'.join((basefolder, 'fileprocessing', basefile))
        fragmentlocation = '/'.join((basefolder, 'fileprocessing', basefile, 'fragments'))
        librarylocation = '/home/sfo/data/proteomics/fastas/search-db'
        self.csvfilename = '/'.join((fragmentlocation, 'fragments'))
        proteome = 'Human_Homo_sapien'
        self.proton = 1.007276554940804
        nprocs = os.cpu_count()
        self.semaphore = mp.Semaphore(nprocs)
        self.dividingthreshold = 0.1
        ppmtol = 25
        self.ppmmod = ppmtol / 1000000
        
        if not os.path.isdir(fragmentlocation):
            os.mkdir(fragmentlocation)
        
        divisionfile = '/'.join((processinglocation, 'dividedgroups.pickle'))
        with open(divisionfile, 'rb') as pick:
            self.dividedgroups = pickle.load(pick)
        
        chargesoflinesfile = '/'.join((processinglocation, 'chargesoflines.pickle'))
        with open(chargesoflinesfile, 'rb') as pick:
            self.chargesoflines = pickle.load(pick)
        #chargesoflines = line: charge
        
        #analytefile = '/'.join((processinglocation, 'analytefactors.pickle'))
        #with open(analytefile, 'rb') as pick:
        #    self.analytesbydistribution, self.distributionsoflines = pickle.load(pick)[2:4]
        #analytesbydistribution = {} #distid: analyte id
        #distributionsoflines = {} #lineid: distid
        
        linesbyscanbysubformulafile = '/'.join((processinglocation, 'linesbyscanbysubformula.pickle'))
        with open(linesbyscanbysubformulafile, 'rb') as pick:
            self.linesbyscanbysubformula = pickle.load(pick)
        #linesbyscanbysubformula = {} #subformula: scan: [lines]
        
        #scanmassesfile = '/'.join((processinglocation, 'scanmasses.pickle'))
        #with open(scanmassesfile, 'rb') as pick:
        #    self.scanmasses = pickle.load(pick)
        ##scanmasses = {} #scan: [[masses], [intensities]]
        self.scanmasses = ms2_loader(processingdirectory, self.linesbyscanbysubformula)
        
        elementsofprobindicesfile = '/'.join((processinglocation, 'elementsofprobabilityindices.pickle'))
        with open(elementsofprobindicesfile, 'rb') as pick:
            self.elementsofprobabilityindices = pickle.load(pick)
        #elementsofprobabilityindices = {} #prob index: e
    
        probabilityorganizerfile = '/'.join((processinglocation, 'probabilityorganizer.pickle'))
        with open(probabilityorganizerfile, 'rb') as pick:
            self.probabilityorganizer = pickle.load(pick)
        #probabilityorganizer = defaultdict(dict) #prob index: iso: prob
        
        matchprobfile = '/'.join((processinglocation, 'matchprobabilities.pickle'))
        with open(matchprobfile, 'rb') as pick:
            self.matchprobabilities = pickle.load(pick)
        #matchprobabilities = defaultdict(list) #subformula: [prob indices]
        
        subformulasubindsfile = '/'.join((processinglocation, 'subformulasubindices.pickle'))
        with open(subformulasubindsfile, 'rb') as pick:
            self.subformulasubindices = pickle.load(pick)
        #subformulasubindices = defaultdict(list) #subformula: [sub match indices]

        submatchsequencesfile = '/'.join((processinglocation, 'submatchsequences.pickle'))
        with open(submatchsequencesfile, 'rb') as pick:
            self.submatchsequences = pickle.load(pick)
        #submatchsequences = {} #submatchindex: sequence
        
        environment_partial = partial(lmdb.Environment, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=9999, max_spare_txns=1, lock=True)
        
        with environment_partial(librarylocation) as env:
            aas = env.open_db('aminoacids'.encode())
            with env.begin(write=False) as txn:
                with txn.cursor(aas) as cursor:
                    aaget = cursor.get(proteome.encode()).decode()
                    self.aminoacidcomposition = eval(aaget)
        
        self.elementalmasses = { #isotope: mass
                    'H1': 1.00782503223,
                    'H2': 2.01410177812,
                    'C12': 12.0000000, 
                    'C13': 13.00335483507,
                    'N14': 14.00307400443,
                    'N15': 15.00010889888,
                    'O16': 15.99491461957,
                    'O17': 16.99913175650,
                    'O18': 17.99915961286,
                    'S32': 31.9720711744,
                    'S33': 32.9714589098,
                    'S34': 33.967867004,
                    'S36': 35.96708071}

        nfragmentcompositions = {}
        nfragmentcompositions['a'] = Counter({'C': -1, 'O': -1})
        nfragmentcompositions['b'] = Counter()
        nfragmentcompositions['c'] = Counter({'N': 1, 'H': 3})
        
        cfragmentcompositions = {}
        cfragmentcompositions['x'] = Counter({'C': 1, 'O': 2})
        cfragmentcompositions['y'] = Counter({'H': 2, 'O': 1})
        cfragmentcompositions['z'] = Counter({'N': -1, 'H': -1, 'O': 1})
        #cfragmentcompositions['z•'] = Counter({'N': -1, 'O': 1})
        
        ionlist = list(ions)
        self.ndict = {k: nfragmentcompositions[k] for k in ionlist if k in nfragmentcompositions}
        self.cdict = {k: cfragmentcompositions[k] for k in ionlist if k in cfragmentcompositions}
    
    def radius_neighbors(self, baselist, flylist):
        b = 0
        pool = []
        matches = {} #flylist index: [baselist indices]
        biter = enumerate(baselist)
        for fn, fly in enumerate(flylist):
            ftol = fly * self.ppmmod
            fmin = fly - ftol
            fmax = fly + ftol
            removals = []
            submatches = []
            for pi, pb in pool:
                if pb < fmin:
                    removals.append([pi, pb])
                elif pb <= fmax:
                    submatches.append(pi)
            for r in removals:
                pool.remove(r)
            while b <= fmax:
                try:
                    i, b = next(biter)
                    if b >= fmin:
                        pool.append([i, b])
                        if b <= fmax:
                            submatches.append(i)
                except StopIteration:
                    break
            #matches.append(submatches)
            if submatches:
                matches[fn] = submatches
        return matches
    
    def fragmentation_compositions(self, seq):
        fragments = {}

        #calculate the compositions of the n-term fragments
        fragcomp_n = {}
        for n, aa in enumerate(seq[:-1]):  
            aa_composition = self.aminoacidcomposition[aa]
            for k in aa_composition:
                fragcomp_n[k] = fragcomp_n.get(k, 0) + aa_composition.get(k, 0)
            for ion, modcomp in self.ndict.items():
                fragment_composition = fragcomp_n.copy()
                for k in modcomp:
                    fc = fragment_composition.get(k, 0) + modcomp.get(k, 0)
                    if fc > 0:
                        fragment_composition[k] = fc
                    else:
                        del fragment_composition[k]
                fragments[ion + str(n + 1)] = fragment_composition
        
        #calculate the compositions of the c-term fragments
        fragcomp_c = {}
        for n, aa in enumerate(seq[::-1][:-1]): 
            aa_composition = self.aminoacidcomposition[aa]
            for k in aa_composition:
                fragcomp_c[k] = fragcomp_c.get(k, 0) + aa_composition.get(k, 0)
            for ion, modcomp in self.cdict.items():
                fragment_composition = fragcomp_c.copy()
                for k in modcomp:
                    fc = fragment_composition.get(k, 0) + modcomp.get(k, 0)
                    if fc > 0:
                        fragment_composition[k] = fc
                    else:
                        del fragment_composition[k]
                fragments[ion + str(n + 1)] = fragment_composition
        
        aa = seq[0]
        aa_composition = self.aminoacidcomposition[aa]
        for k in aa_composition:
            fragcomp_c[k] = fragcomp_c.get(k, 0) + aa_composition.get(k, 0)
        fragcomp_c['H'] += 2
        fragcomp_c['O'] += 1
        fragments['precursor'] = fragcomp_c
        
        return fragments
    
    def fragment_element_binomial_walk(self, e, acount, fragprobabilities):
        nvector = []
        fragmentvectorpositions = {} #iso: position in vector, replacing nvectorpositions
        fragmentelementpositions = {} #position: iso
        maxinitial = 0
        for n, (iso, prob) in enumerate(fragprobabilities.items()):
            nvector.append(0)
            fragmentvectorpositions[iso] = n
            fragmentelementpositions[n] = iso
            if prob > maxinitial:
                maxinitial = prob
                mk = iso
        lesserfragmentisotopes = [i for i in fragprobabilities if i != mk] #replacing nonmonoisotopicgroups
        elementlist = []
        mainheap = []
        vectorsets = defaultdict(set) #element: set of used vectors
        nvector[fragmentvectorpositions[mk]] += acount
        flen = len(fragprobabilities)
        if flen > 2:
            baseprob = fragprobabilities[mk] ** acount
            preheap = []
            preheap.append([baseprob, acount * self.elementalmasses[mk], e, nvector.copy()])
            greater = True
            lastprob = baseprob
            while greater:
                greater = False
                for iso in lesserfragmentisotopes:
                    newelementvector = nvector.copy()
                    newelementvector[fragmentvectorpositions[mk]] -= 1
                    if newelementvector[fragmentvectorpositions[mk]] > -1:
                        newelementvector[fragmentvectorpositions[iso]] += 1
                        vectorsets[e].add(tuple(newelementvector))
                        pn = 0
                        newelementmass = 0
                        newelementprob = 1
                        for n, c in enumerate(newelementvector):
                            loopiso = fragmentelementpositions[n]
                            newelementmass += self.elementalmasses[loopiso] * c
                            newelementprob *= fragprobabilities[loopiso]**c
                            if n > 0:
                                newelementprob *= math.comb(acount-pn, c)
                                pn += c
                        preheap.append([newelementprob, newelementmass, e, newelementvector.copy()])
                        if newelementprob > lastprob:
                            lastprob = newelementprob
                            greater = True
            preheap = sorted(preheap)
            maxiso = preheap[-1]
            maxprob, m, e, nv = maxiso
            elementlist.append([-1, maxprob, m, e, nv])
            maxprob *= -1
            preheap = preheap[:-1]
            for h in preheap:
                r = h[0] / maxprob
                h.insert(0, r)
                heapq.heappush(mainheap, h)
            for iso in lesserfragmentisotopes:
                v = nv.copy()
                v[fragmentvectorpositions[mk]] -= 1
                if v[fragmentvectorpositions[mk]] > -1:
                    v[fragmentvectorpositions[iso]] += 1
                    tuplevec = tuple(v)
                    if tuplevec not in vectorsets[e]:
                        vectorsets[e].add(tuplevec)
                        pn = 0
                        newelementmass = 0
                        newelementprob = 1
                        for n, c in enumerate(v):
                            loopiso = fragmentelementpositions[n]
                            newelementmass += self.elementalmasses[loopiso] * c
                            newelementprob *= fragprobabilities[loopiso]**c
                            if n > 0:
                                newelementprob *= math.comb(acount-pn, c)
                                pn += c
                        heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, v.copy()])
        else:
            preheap = []
            baseprob = fragprobabilities[mk] ** acount
            preheap.append([baseprob, acount * self.elementalmasses[mk], e, nvector.copy()])
            greater = True
            lastprob = baseprob
            iso = lesserfragmentisotopes[0]
            while greater:
                greater = False
                nvector[fragmentvectorpositions[mk]] -= 1
                if nvector[fragmentvectorpositions[mk]] > -1:
                    nvector[fragmentvectorpositions[iso]] += 1
                    vectorsets[e].add(tuple(nvector))
                    pn = 0
                    newelementmass = 0
                    newelementprob = 1
                    for n, c in enumerate(nvector):
                        loopiso = fragmentelementpositions[n]
                        newelementmass += self.elementalmasses[loopiso] * c
                        newelementprob *= fragprobabilities[loopiso]**c
                        if n > 0:
                            newelementprob *= math.comb(acount-pn, c)
                            pn += c
                    preheap.append([newelementprob, newelementmass, e, nvector.copy()])
                    if newelementprob > lastprob:
                        lastprob = newelementprob
                        greater = True
            preheap = sorted(preheap)
            maxiso = preheap[-1]
            maxprob, m, e, nv = maxiso
            elementlist.append([-1, maxprob, m, e, nv])
            maxprob *= -1
            preheap = preheap[:-1]
            for h in preheap:
                r = h[0] / maxprob
                h.insert(0, r)
                heapq.heappush(mainheap, h)
            v = nv.copy()
            v[fragmentvectorpositions[mk]] -= 1
            if v[fragmentvectorpositions[mk]] > -1:
                v[fragmentvectorpositions[iso]] += 1
                tuplevec = tuple(v)
                if tuplevec not in vectorsets[e]:
                    vectorsets[e].add(tuplevec)
                    pn = 0
                    newelementmass = 0
                    newelementprob = 1
                    for n, c in enumerate(v):
                        loopiso = fragmentelementpositions[n]
                        newelementmass += self.elementalmasses[loopiso] * c
                        newelementprob *= fragprobabilities[loopiso]**c
                        if n > 0:
                            newelementprob *= math.comb(acount-pn, c)
                            pn += c
                    heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, v.copy()])
        
        cutoff = -maxprob * self.dividingthreshold

        r, p, m, e, v = heapq.heappop(mainheap)
        elementlist.append([r, p, m, e, v])
        if flen > 2:
            while p > cutoff:
                for iso in lesserfragmentisotopes:
                    newelementvector = v.copy()
                    newelementvector[fragmentvectorpositions[mk]] -= 1
                    if newelementvector[fragmentvectorpositions[mk]] > 0:
                        newelementvector[fragmentvectorpositions[iso]] += 1
                        tuplevec = tuple(newelementvector)
                        if tuplevec not in vectorsets[e]:
                            vectorsets[e].add(tuplevec)
                            pn = 0
                            newelementmass = 0
                            newelementprob = 1
                            for n, c in enumerate(newelementvector):
                                loopiso = fragmentelementpositions[n]
                                newelementmass += self.elementalmasses[loopiso] * c
                                newelementprob *= fragprobabilities[loopiso]**c
                                if n > 0:
                                    newelementprob *= math.comb(acount-pn, c)
                                    pn += c
                            heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, newelementvector.copy()])
                r, p, m, e, v = heapq.heappop(mainheap)
                elementlist.append([r, p, m, e, v])
                try:
                    r, p, m, e, v = heapq.heappop(mainheap)
                    elementlist.append([r, p, m, e, v])
                except IndexError:
                    #mainheap is empty, this can happen when count is low and probabilities are evenly split. When this happened it was in the below loop, but I'll keep this here too just in case
                    break
        else:
            iso = lesserfragmentisotopes[0]
            while p > cutoff:
                nvector = v.copy()
                nvector[fragmentvectorpositions[mk]] -= 1
                if nvector[fragmentvectorpositions[mk]] > 0:
                    nvector[fragmentvectorpositions[iso]] += 1
                    tuplevec = tuple(nvector)
                    if tuplevec not in vectorsets[e]:
                        vectorsets[e].add(tuplevec)
                        pn = 0
                        newelementmass = 0
                        newelementprob = 1
                        for n, c in enumerate(nvector):
                            loopiso = fragmentelementpositions[n]
                            newelementmass += self.elementalmasses[loopiso] * c
                            newelementprob *= fragprobabilities[loopiso]**c
                            if n > 0:
                                newelementprob *= math.comb(acount-pn, c)
                                pn += c
                        heapq.heappush(mainheap, [newelementprob / maxprob, newelementprob, newelementmass, e, nvector.copy()])
                try:
                    r, p, m, e, v = heapq.heappop(mainheap)
                    elementlist.append([r, p, m, e, v])
                except IndexError:
                    #mainheap is empty, this can happen when count is low and probabilities are evenly split
                    break
        heapq.heapify(elementlist)
        return elementlist, fragmentelementpositions
    
    def fragment_descending_partial_products(self, elementalorganizer, fragmentpositions):
        mainpool = defaultdict(list) #things already popped from elementalorganizer
        for k in elementalorganizer:
            mainpool[k].append(heapq.heappop(elementalorganizer[k]))

        subformulas = []
        sumabundances = []
        massnumberindices = {} #mass number: index in the 2 above lists
        
        formula = ''
        maxprob = 1
        mainmass = 0
        massnumber = 0
        for b in sorted(mainpool):
            for r, p, m, e, v in mainpool[b]:
                for n, c in enumerate(v):
                    if c > 0:
                        iso = fragmentpositions[e][n]
                        massnumber += int(iso[1:]) * c
                        formula += f'{iso}({c})'
                maxprob *= p
                mainmass += m

        massnumberindices[massnumber] = 0
        subformulas.append(formula)
        sumabundances.append([mainmass * maxprob, maxprob])
        
        cutoff = maxprob * self.dividingthreshold
        mainheap = list(chain(*elementalorganizer.values()))
        heapq.heapify(mainheap)
        
        vectorpool = set()
        multinomialpath = [] #sublists not in mainpool
        probabilityranking = [] #representative lists of ratio probability to sort multinomialpath
        while mainheap:
            r, p, m, e, v = heapq.heappop(mainheap)
            baseiter = {k: v for k, v in mainpool.items() if k != e}
            baseiter[e] = [(r, p, m, e, v)]
            
            formula = ''
            prob = 1
            mass = 0
            massnumber = 0
            for b in sorted(baseiter):
                for sr, sp, sm, se, sv in baseiter[b]:
                    for n, c in enumerate(sv):
                        if c > 0:
                            iso = fragmentpositions[se][n]
                            massnumber += int(iso[1:]) * c
                            formula += f'{iso}({c})'
                    prob *= sp
                    mass += sm
            
            try:
                index = massnumberindices[massnumber]
                subformulas[index] += '-' + formula
                sumabundances[index][0] += mass * prob
                sumabundances[index][1] += prob
            except KeyError: #not in there
                index = len(massnumberindices)
                massnumberindices[massnumber] = index
                subformulas.append(formula)
                sumabundances.append([mass * prob, prob])
            if prob < cutoff:
                break
            
            tsv = tuple(v)
            if tsv not in vectorpool:
                ind = bisect(probabilityranking, r)
                probabilityranking.insert(ind, r)
                multinomialpath.insert(ind, (r, p, m, e, v))
                vectorpool.add(tsv)
            
            checkedcombos = set()
            for path in multinomialpath.copy():
                multielement = False
                match path[1]:
                    case tuple():
                        multielement = True
                        sepool = set()
                        sepool.add(e)
                        seformulas = []
                        multipath = []
                        nsr = 1
                        for sr, sp, sm, se, sv in path[1:]:
                            if se not in sepool:
                                nsr *= sr
                                sepool.add(se)
                                sef = ''
                                for n, c in enumerate(sv):
                                    if c > 0:
                                        sef += f'{fragmentpositions[se][n]}({c})'
                                seformulas.append(sef)
                                multipath.append((sr, sp, sm, se, sv))
                        checkformula = ''.join((sorted(seformulas)))
                        if checkformula in checkedcombos:
                            continue
                        else:
                            checkedcombos.add(checkformula)
                        if len(multipath) == 0:
                            continue
                    case _:
                        sr, sp, sm, se, sv = path
                        sef = ''.join((f'{se}{str(n)}{(val)}' for n, val in enumerate(sv)))
                        if sef in checkedcombos:
                            continue
                        else:
                            checkedcombos.add(sef)
                        if se == e:
                            continue
                        nsr = sr
                newratio = nsr * r
                if newratio > 0:
                    newratio *= -1
                if -newratio >= self.dividingthreshold:
                    if multielement:
                        seformula = ''
                        newprob = 1
                        newmass = 0
                        newmassnum = 0
                        newiter = {k: v for k, v in baseiter.items() if k not in sepool}
                        newiter[e] = [(r, p, m, e, v)]
                        for ir, ip, im, ie, iv in multipath:
                            newiter[ie] = [(ir, ip, im, ie, iv)]
                        for b in sorted(newiter):
                            for ir, ip, im, ie, iv in newiter[b]:
                                for n, c in enumerate(iv):
                                    if c > 0:
                                        iso = fragmentpositions[ie][n]
                                        newmassnum += int(iso[1:]) * c
                                        seformula += f'{iso}({c})'
                                newprob *= ip
                                newmass += im
                    else:
                        newiter = {k: v for k, v in baseiter.items() if k != se}
                        newiter[se] = [(sr, sp, sm, se, sv)]
                        seformula = ''
                        newprob = 1
                        newmass = 0
                        newmassnum = 0
                        for b in sorted(newiter):
                            for ir, ip, im, ie, iv in newiter[b]:
                                for n, c in enumerate(iv):
                                    if c > 0:
                                        iso = fragmentpositions[ie][n]
                                        newmassnum += int(iso[1:]) * c
                                        seformula += f'{iso}({c})'
                                newprob *= ip
                                newmass += im
                    if newprob >= cutoff:
                        try:
                            index = massnumberindices[newmassnum]
                            subformulas[index] += '-' + seformula
                            sumabundances[index][0] += newmass * newprob
                            sumabundances[index][1] += newprob
                        except KeyError: #not in there
                            index = len(massnumberindices)
                            massnumberindices[newmassnum] = index
                            subformulas.append(seformula)
                            sumabundances.append([newmass * newprob, newprob])
                        if multielement:
                            ind = bisect(probabilityranking, newratio)
                            probabilityranking.insert(ind, newratio)
                            multinomialpath.insert(ind, (newratio, *multipath))
                        else: #this is rarely ever needed, but it is needed
                            newmulti = []
                            tsv = tuple(sv)
                            #should this one be first? does it matter? i don't believe it does
                            if tsv not in vectorpool:
                                newmulti.append((sr, sp, sm, se, sv))
                                vectorpool.add(tsv)
                            tvv = tuple(v)
                            if tvv not in vectorpool:
                                newmulti.append((r, p, m, e, v))
                                vectorpool.add(tvv)
                            if newmulti:
                                ind = bisect(probabilityranking, newratio)
                                probabilityranking.insert(ind, newratio)
                                multinomialpath.insert(ind, (newratio, *newmulti))
                else:
                    break

        subformulas = np.array(subformulas, dtype='S')
        massesandabundances = np.array(sumabundances)
        massesandabundances[:,0] /= massesandabundances[:,1]
        #sorting by intensity
        subformulas = subformulas[massesandabundances[:,1].argsort()[::-1]].tolist()
        massesandabundances = massesandabundances[massesandabundances[:,1].argsort()[::-1]]
        return subformulas, massesandabundances
    
    def group_fragmentation(self, group, count):
        searchtime = 0
        fraglens = 0
        scanlens = 0
        chargeiterations = 0
        nt = time()
        positioncache = {}
        elementalcache = {}
        descentcache = {}
        finaloutput = []
        groupseqs = []
        groupsubformulas = []
        for member in group:
            if '(' in member:
                groupsubformulas.append(member)
            else:
                groupseqs.append(member)
        fragments = {}
        for seq in groupseqs:
            fragments[seq] = self.fragmentation_compositions(seq)
        for subformula in groupsubformulas:
            probindices = {self.elementsofprobabilityindices[i]: self.probabilityorganizer[i] for i in self.matchprobabilities[subformula]}
            subindices = self.subformulasubindices[subformula]
            output, fragmasses = [], []
            for submatchindex in subindices:
                seq = self.submatchsequences[submatchindex]
                for ion, fragcomp in fragments[seq].items():
                    elementalorganizer = {} #element: [[iso heaps]]
                    fragmentpositions = {} #element: position: iso
                    fragstrings = ''
                    for e, c in fragcomp.items():
                        fragprobs = probindices[e]
                        fragstring = str(c) + '/' + '/'.join(('/'.join((k, str(v))) for k, v in probindices[e].items()))
                        fragstrings += fragstring
                        if len(fragprobs) > 1:
                            #try/except if faster than an if/else, so i might as well
                            try:
                                elementlist = elementalcache[fragstring]
                                positions = positioncache[fragstring]
                            except KeyError: #not in cache
                                elementlist, positions = self.fragment_element_binomial_walk(e, c, fragprobs)
                                elementalcache[fragstring] = elementlist
                                positioncache[fragstring] = positions
                            elementalorganizer[e] = elementlist.copy()
                            fragmentpositions[e] = positions
                        else: #no need for cache, only 1 iso
                            iso = list(fragprobs)[0]
                            elementalorganizer[e] = [[-1, 1, self.elementalmasses[iso]*c, e, [c]]]
                            fragmentpositions[e] = {0: iso}
                    try:
                        fragformulas, massesandabundances = descentcache[fragstrings]
                    except KeyError: #not done prior
                        fragformulas, massesandabundances = self.fragment_descending_partial_products(elementalorganizer, fragmentpositions)
                        descentcache[fragstrings] = fragformulas, massesandabundances
                    for n, (m, i) in enumerate(massesandabundances.tolist()):
                        out = (seq, fragformulas[n].decode(), ion, submatchindex, i, n)
                        #out = (seq, fragformulas[n].decode(), ion, submatchindex, i, mainindexbysubmatchindex[submatchindex]) #do i need main index???
                        output.append(out)
                        fragmasses.append(m)
            fragmasses, output = zip(*sorted(zip(fragmasses, output)))
            fragmasses = np.array(fragmasses)
            fraglens += fragmasses.size
            st = time()
            for scan, lines in self.linesbyscanbysubformula[subformula].items():
                if len(lines) > 1:
                    #analyteid = '-'.join((str(self.analytesbydistribution[self.distributionsoflines[i]]) for i in lines))
                    maxcharge = max(self.chargesoflines[i] for i in lines)
                    lines = '-'.join((str(i) for i in lines))
                else:
                    lines = lines[0]
                    #analyteid = self.analytesbydistribution[self.distributionsoflines[lines]]
                    maxcharge = self.chargesoflines[lines]
                ms2masses = self.scanmasses[scan]
                scanlens += len(ms2masses)
                for charge in range(1, maxcharge+1):
                    chargeiterations += 1
                    chargedfragments = ((fragmasses + self.proton * charge) / charge).tolist()
                    matches = self.radius_neighbors(chargedfragments, ms2masses)
                    for scanindex, fragindices in matches.items(): #scan index: [fragmass indices]
                        #a scanmass can match to multiple generated fragment ions
                        experimentalmass = ms2masses[scanindex]
                        for fragindex in fragindices:
                            theoreticalmass = chargedfragments[fragindex]
                            ppmerror = ((experimentalmass - theoreticalmass) / experimentalmass) * 1000000
                            out = (*output[fragindex], lines, scan, theoreticalmass, ppmerror, scanindex, charge)
                            outstring = ','.join((str(i) for i in out)) + '\n' #smaller memory footprint + needs to be written like this later, only ~1-2 microsecond cost
                            finaloutput.append(outstring)
            searchtime += time() - st
        fragtime = time() - nt - searchtime
        with open(self.csvfilename + '.' + str(count) + '.matches.csv', 'w') as f:
            for piece in finaloutput:
                f.write(piece)
        print(f'{round(time() - nt,4)} - group {count} saved, fragtime: {round(fragtime, 4)}, searchtime: {round(searchtime,4)}, generated fragments: {fraglens}, scan masses: {scanlens}, charge-iterations: {chargeiterations}, matches: {len(finaloutput)}')
        self.semaphore.release()
    
    def group_processing(self):
        #with mp.Pool(processes=self.nprocs) as pool:
        #    for count, divgroup in enumerate(self.dividedgroups):
        #        #if count < 70:
        #        pool.apply_async(self.group_fragmentation, args=(divgroup, count))
        #    pool.close()
        #    pool.join()
        processes = []
        for count, divgroup in enumerate(self.dividedgroups):
            #if count < 70:
            self.semaphore.acquire()
            p = mp.Process(target=self.group_fragmentation, args=(divgroup, count))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

nt = time()

mzmlfile = '/home/sfo/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
ions = 'by'
model = FragmentOrganizer(mzmlfile, ions)
model.group_processing()

print(time() - nt, 'total')
