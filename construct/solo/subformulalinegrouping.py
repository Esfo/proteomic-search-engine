from generalfunctions import intersection_merge
from database import environment

import numpy as np
from time import time
from collections import defaultdict
import pickle

def subformula_line_grouping(librarylocation, processingdirectory, proteome, ions):
    t2 = time()
    
    linepositionsbyformulafile = ''.join((processingdirectory, 'linepositionsbyformula.pickle'))
    with open(linepositionsbyformulafile, 'rb') as pick:
        linepositionsbyformula = pickle.load(pick)
    #linepositionsbyformula = defaultdict(lambda: defaultdict(set)) #formula: matchable library positions of a distribution: [lines that have ms2 scans]
    
    scansbyanalytefile = ''.join((processingdirectory, 'scansbyanalyte.pickle'))
    with open(scansbyanalytefile, 'rb') as pick:
        scansbyanalyte = pickle.load(pick)
    #scansbyanalyte = defaultdict(list) #analyteid: [scans across all lines and charge states]
    
    scansoflinesfile = ''.join((processingdirectory, 'scansoflines.pickle'))
    with open(scansoflinesfile, 'rb') as pick:
        scansoflines = pickle.load(pick)
    #scansoflines = defaultdict(list) #lineuid: [ms2 scan indices]
    
    maxsampledistributionsoflinesfile = ''.join((processingdirectory, 'maxsampledistributionsoflines.pickle'))
    with open(maxsampledistributionsoflinesfile, 'rb') as pick:
        maxsampledistributionsoflines = pickle.load(pick)
    #maxsampledistributionsoflines = {} #line: distid
    
    encodedkeys = [i.encode() for i in linepositionsbyformula]

    seqsbyformula = {} #formula: [seqs]
    abundances = {} #formula: [[masses], [intensities]]
    abundanceformulas = {} #formula: subformulas
    condensationcoordinates = {} #formula: [# isotopomers per proton-step]
    subisodepthqualifiers = {} #formula: [[top n subisos of proton locations]]
    with environment(librarylocation) as env:
        ddb = env.open_db('distributions.formulas'.encode())
        with env.begin(write=False) as txn:
            with txn.cursor(ddb) as cursor:
                for k, v in cursor.getmulti(encodedkeys):
                    abundanceformulas[k.decode()] = eval(v.decode())
        condensationdb = env.open_db('distributions.condensationcoordinates'.encode())
        with env.begin(write=False) as txn:
            with txn.cursor(condensationdb) as cursor:
                for k, v in cursor.getmulti(encodedkeys):
                    condensationcoordinates[k.decode()] = np.frombuffer(v, dtype=int)
        subisoqualdb = env.open_db('distributions.subisodepthqualifiers'.encode())
        with env.begin(write=False) as txn:
            with txn.cursor(subisoqualdb) as cursor:
                for k, v in cursor.getmulti(encodedkeys):
                    subisodepthqualifiers[k.decode()] = eval(v.decode())
        fulldb = env.open_db('distributions.full'.encode())
        with env.begin(write=False) as txn:
            with txn.cursor(fulldb) as cursor:
                for k, v in cursor.getmulti(encodedkeys):
                    out = np.frombuffer(v)
                    out = out.reshape(2, out.size//2)
                    abundances[k.decode()] = out
        proteomedb = env.open_db((proteome + '.seqsbyformula').encode())
        with env.begin(write=False) as txn:
            with txn.cursor(proteomedb) as cursor:
                for k, v in cursor.getmulti(encodedkeys):
                    seqsbyformula[k.decode()] = eval(v.decode())
    
    probtracker = {} #prob string: prob index
    
    probabilityorganizer = defaultdict(dict) #prob index: iso: prob
    matchprobabilities = defaultdict(list) #subformula: [prob indices] #subformula here instead of match index bc the prob comp is tied to subformulas
    
    subformulasubindices = defaultdict(list) #subformula: [sub match indices]
    submatchsequences = {} #submatchindex: sequence
    elementsofprobabilityindices = {} #prob index: e
    
    linesbysubformula = defaultdict(set) #subformula: [lines that have ms2 scans]
    subformulapercent = defaultdict(dict) #subformula: sequence: (subiso abundance rank, subiso abundance)
    subformulasofsequencedistribution = defaultdict(dict) #dist: seq: subformula
    
    mergables = []
    
    probindex = 0
    submatchindex = 0
    mainmatchindex = 0
    for formula, positions in linepositionsbyformula.items():
        qualifiers = subisodepthqualifiers[formula]
        conlengths = condensationcoordinates[formula]
        conends = conlengths.cumsum()
        constarts = conends - conlengths
        subformulas = [i.decode() for i in abundanceformulas[formula]]
        massesandintensities = abundances[formula]
        theoreticalabundances = massesandintensities[1]
        for position, lines in positions.items():
            for seq in seqsbyformula[formula]:
                bi = constarts[position]
                for qualrank, sq in enumerate(qualifiers[position]):
                    subindex = bi + sq
                    sformula = subformulas[subindex]
                    subformulapercent[sformula][seq] = qualrank, theoreticalabundances[subindex]
                    linesbysubformula[sformula].update(lines)
                    for line in lines:
                        #if qualrank == 0:
                        if line in maxsampledistributionsoflines:
                            distid = maxsampledistributionsoflines[line]
                            subformulasofsequencedistribution[distid][seq] = sformula
                    mergables.append(f'{sformula}-{seq}') #trying to preserve memory here
                    subformulasubindices[sformula].append(submatchindex)
                    submatchsequences[submatchindex] = seq
                    submatchindex += 1
                    if sformula not in matchprobabilities:
                        #setting up subformula-specific probabilities
                        isocounts = set()
                        competing = set()
                        competitors = {}
                        isosums = {}
                        for ss in sformula.split(')')[:-1]:
                            iso, c = ss.split('(')
                            c = int(c)
                            splitval = 0
                            #for handling elements with multiple letters
                            while True:
                                if iso[splitval].isalpha():
                                    splitval += 1
                                else:
                                    break
                            e = iso[:splitval]
                            if e in isocounts:
                                competing.add(e)
                                competitors[e][iso] = c
                                isosums[e] += c
                            else:
                                isocounts.add(e)
                                competitors[e] = {iso: c}
                                isosums[e] = c
                        for e, v in competitors.items():
                            isoprobs = {}
                            if e in competing:
                                for iso, c in v.items():
                                    prob = c / isosums[e]
                                    isoprobs[iso] = prob
                                probstring = '/'.join(('/'.join((k, str(v))) for k, v in isoprobs.items()))
                                if probstring in probtracker:
                                    foundprobindex = probtracker[probstring]
                                    matchprobabilities[sformula].append(foundprobindex)
                                else:
                                    probtracker[probstring] = probindex
                                    probabilityorganizer[probindex] = isoprobs
                                    matchprobabilities[sformula].append(probindex)
                                    elementsofprobabilityindices[probindex] = e
                                    probindex += 1
                            else:
                                #don't need to make a new index for every time something has no competition
                                for iso in v:
                                    isoprobs[iso] = 1
                                if e not in probabilityorganizer:
                                    probstring = tuple(isoprobs.items())
                                    probtracker[probstring] = e
                                    probabilityorganizer[e] = isoprobs
                                    elementsofprobabilityindices[e] = e
                                matchprobabilities[sformula].append(e)
        mainmatchindex += 1

    probabilityorganizer = dict(probabilityorganizer)
    matchprobabilities = dict(matchprobabilities)
    subformulasubindices = dict(subformulasubindices)

    linesbyscanbysubformula = {} #subformula: scan: [lines]
    for sformula, lines in linesbysubformula.items():
        linesbyscan = defaultdict(list)
        for line in lines:
            for scan in scansoflines[line]:
                linesbyscan[scan].append(line)
        for k, v in linesbyscan.items():
            linesbyscan[k] = tuple(v)
        linesbyscan = dict(linesbyscan)
        linesbyscanbysubformula[sformula] = linesbyscan

    for subformula, seqs in subformulapercent.items():
        subformulapercent[subformula] = dict(subformulapercent[subformula])
    
    print(time() - t2, 'submatch organization completed')
    t3 = time()

    mergables = list(set(mergables))
    firstmerge = map(tuple, intersection_merge(i.split('-') for i in mergables))
    #^merging by seqs and subformulas as a first layer of redundancy reduction
    #^this yields a large, somewhat unusable number of groups that would cause a lot of pain for the high redundancy of isotopic compositions to calculate later
    #the second layer will be by isotopic composition by individual elements
    
    #custom intersection merge to limit the size of each group to whatevers written below
    #makes for good memory management later when generating fragments
    limiter = 200000 / len(ions)
    sn = 0
    groupsofitems = {} #iso-member: group
    itemgroups = defaultdict(set) #group: [members]
    subitemgroups = defaultdict(set) #group: [isotopic compositions]
    for items in firstmerge:
        locs = set()
        subitems = set()
        for i in items:
            if ')' in i:
                subgroups = defaultdict(list) #element: [subiso comps]
                for split in i.split(')')[:-1]:
                    splitval = 0
                    #for handling elements with multiple letters
                    while True:
                        if split[splitval].isalpha():
                            splitval += 1
                        else:
                            break
                    e = split[:splitval]
                    if e == 'C':
                        subgroups[e].append(split)
                for e, group in subgroups.items():
                    output = ')'.join((group)) + ')'
                    if output in groupsofitems:
                        locs.add(groupsofitems[output])
                    subitems.add(output)
        if locs:
            joiner = min(locs)
            if len(locs) > 1:
                for oldloc in locs.difference([joiner]):
                    for ol in subitemgroups[oldloc]:
                        groupsofitems[ol] = joiner
                    itemgroups[joiner].update(itemgroups.pop(oldloc))
                    subitemgroups[joiner].update(subitemgroups.pop(oldloc))
        else:
            joiner = sn
            sn += 1
        itemgroups[joiner].update(items)
        subitemgroups[joiner].update(subitems)
        for i in subitems:
            groupsofitems[i] = joiner
        if len(itemgroups[joiner]) >= limiter:
            for member in subitemgroups[joiner]:
                #by deleting the old locs it will force them incoming items into new groups
                del groupsofitems[member]
    
    dividedgroups = list((map(tuple, itemgroups.values())))
    
    print(time() - t3, 'processable fragment groups assembled')
    
    divisionfile = ''.join((processingdirectory, 'dividedgroups.pickle'))
    with open(divisionfile, 'wb') as pick:
        pickle.dump(dividedgroups, pick)
    
    elementsofprobindicesfile = ''.join((processingdirectory, 'elementsofprobabilityindices.pickle'))
    with open(elementsofprobindicesfile, 'wb') as pick:
        pickle.dump(elementsofprobabilityindices, pick)
    
    probabilityorganizerfile = ''.join((processingdirectory, 'probabilityorganizer.pickle'))
    with open(probabilityorganizerfile, 'wb') as pick:
        pickle.dump(probabilityorganizer, pick)
    
    matchprobfile = ''.join((processingdirectory, 'matchprobabilities.pickle'))
    with open(matchprobfile, 'wb') as pick:
        pickle.dump(matchprobabilities, pick)
    
    subformulasubindsfile = ''.join((processingdirectory, 'subformulasubindices.pickle'))
    with open(subformulasubindsfile, 'wb') as pick:
        pickle.dump(subformulasubindices, pick)
    
    submatchsequencesfile = ''.join((processingdirectory, 'submatchsequences.pickle'))
    with open(submatchsequencesfile, 'wb') as pick:
        pickle.dump(submatchsequences, pick)
    
    linesbyscanbysubformulafile = ''.join((processingdirectory, 'linesbyscanbysubformula.pickle'))
    with open(linesbyscanbysubformulafile, 'wb') as pick:
        pickle.dump(linesbyscanbysubformula, pick)
    
    subformulapercentfile = ''.join((processingdirectory, 'subformulapercent.pickle'))
    with open(subformulapercentfile, 'wb') as pick:
        pickle.dump(subformulapercent, pick)
    
    subformulasofsequencedistributionfile = ''.join((processingdirectory, 'subformulasofsequencedistribution.pickle'))
    with open(subformulasofsequencedistributionfile, 'wb') as pick:
        pickle.dump(subformulasofsequencedistribution, pick)
    #subformulasofsequencedistribution = {} #seq: dist: subformula
