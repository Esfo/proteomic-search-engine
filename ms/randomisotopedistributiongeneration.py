import numpy as np
import matplotlib.pyplot as plt
import pickle
from functools import reduce
import operator
import pandas as pd
from scipy import spatial, stats
from sklearn.neighbors import NearestNeighbors
import networkx
from networkx.algorithms.components.connected import connected_components
from collections import Counter, defaultdict
import gc
np.set_printoptions(suppress=True)

#for more https://matplotlib.org/stable/tutorials/introductory/customizing.html
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.facecolor'] = 'gray'
plt.rcParams['figure.facecolor'] = 'gray'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['ytick.labelcolor'] = 'white'
plt.rcParams['xtick.labelcolor'] = 'white'

def generic_meta_overlap(l):
    graph = networkx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        graph.add_nodes_from(part)
        # it also implies a number of edges:
        graph.add_edges_from(to_edges(part))
    return [sorted(i) for i in connected_components(graph)]

def to_edges(l):
    """
        treat `l` as a Graph and returns it's edges
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current

def leading_zero_count(num):
    zerosum = 0
    whole, decimals = str(num).split('.')
    if int(whole) <= 0:
        for s in decimals:
            if s == '0':
                zerosum += 1
            else:
                break
    return zerosum

def counting_sum_cutoff(array):
    mbool = array <= array[:,None]
    countsums = mbool.sum(axis=0) / array.size
    #mdsum = array.sum()
    #sumcounts = []
    #for mb in mbool:
    #    sumcounts.append(array[mb].sum() / mdsum)
    #sumcounts = np.array(sumcounts)
    #the sumcounts below is generally the same thing, but will differ at values where mbool would have given duplicate entries, doesn't change anything major enough to change the result
    sumcounts = array.cumsum() / array.sum()
    mincomboind = (countsums + sumcounts).argmin()
    mincombo = array[mincomboind]
    #moving average of average of dists under mincombo
    explicitcutoff = array[array <= mincombo].mean()
    return explicitcutoff


ngenerated = 2
maxcharge = 20
maxrandomcharge = 8
randmod = 0.0000001 #need to destroy any perfect symmetries within the theoretical distributions - luckily they don't exist in real data. This is necessary for processing uniquediffs the way I plan to
isotopefile = '/home/sfo/data/proteomics/fastas/isotope-arrays/human-isotopes-6-50_ss20.pickle'

with open(isotopefile, "rb") as pick:
    isotopeabundances, seqsbymass = pickle.load(pick)

seqmasses = np.sort(list(seqsbymass.keys()))

randomdistkeys = np.random.choice(seqmasses, size=ngenerated, replace=False).tolist()
randomcharges = np.random.randint(low=1, high=maxrandomcharge, size=ngenerated).tolist()

while True:
    randomdistkeys = np.random.choice(seqmasses, size=ngenerated, replace=False).tolist()
    randomcharges = np.random.randint(low=1, high=maxrandomcharge, size=ngenerated).tolist()
    if np.abs(np.diff(randomdistkeys)) < 1:
        break

initialrandists = [isotopeabundances[i] for i in randomdistkeys]
randists = []
for r, c in zip(initialrandists, randomcharges):
    rd = {}
    for k, v in r.items():
        rd[k/c*np.random.uniform(1-randmod, 1+randmod)] = v
    randists.append(rd)

mergeddict = Counter()
for i in randists:
    mergeddict += i

proton = 1.00727647
scope = proton * 1.2
chargearray = np.arange(maxcharge) + 1
chargevalues = proton / chargearray
chargearray = np.append(0, chargearray)
chargevalues = np.append(0, chargevalues)
chargevalues = np.append(chargevalues, scope)
upperfilter = chargevalues.size - 1


#msrun.reset()
#difftree = spatial.KDTree(chargevalues[:,None]) #sklearn class was faster
difftree = NearestNeighbors(n_neighbors=1)
difftree.fit(chargevalues[:,None])

mza = np.array(list(mergeddict.keys()))[:,None]
intensities = np.array(list(mergeddict.values()))

mzdiffmatrix = mza - mza.flatten()
mzdiffs = pd.unique(np.abs(mzdiffmatrix).flatten())
#md, mi = difftree.query(mzdiffs[:,None], k=1)
md, mi = difftree.kneighbors(mzdiffs[:,None])
md = np.sort(md[md < scope][1:])
cut = counting_sum_cutoff(md)
cut = 0.05

isotopes = defaultdict(set)

#resolutiongoal = np.diff(chargevalues[::-1]).min()

#difftree = spatial.KDTree(chargevalues[:,None])
isomodel = spatial.KDTree(mza)
scopeconnections = isomodel.query_ball_point(mza, r=scope)
isotopewindows = generic_meta_overlap(np.unique(scopeconnections).tolist())
isotopewindows = [i for i in isotopewindows if len(i) > 1]

for ni, iw in enumerate(isotopewindows):
    lasteds = []
    isomasses = mza[iw]
    #isolen = len(isomasses)
    #while isomasses.size > 0:
    mzdiffmatrix = np.abs(isomasses - isomasses.flatten())
    sorteddiffs = np.diff(np.sort(isomasses, axis=0), axis=0)
    uniquediffs = np.unique(np.abs(mzdiffmatrix))
    uniquediffs = uniquediffs[uniquediffs < scope, None]
    #isodists, chargeinds = difftree.query(uniquediffs, k=1)
    isodists, chargeinds = difftree.kneighbors(uniquediffs)
    labelsums = {}
    labelmeans = {}
    for label in np.unique(chargeinds):
        if label > 0 and label < upperfilter:
            linds = chargeinds == label
            #if linds.sum() > 1:
            lindists = isodists[linds]
            newlininds = lindists <= cut #it's good that this search is a bit wider in range because the sorted differences get figured out below
            masskeepers = uniquediffs[linds][newlininds]
            mkwhere = np.argwhere(mzdiffmatrix == masskeepers[:,None,None])
            chargegroups = generic_meta_overlap(mkwhere[:,1:].tolist())
            for cg in chargegroups:
                cval = chargevalues[label]
                mzg = isomasses[cg] #mza is already sorted
                mzgdiffs = np.diff(mzg, axis=0)
                mzgdists, mzginds = difftree.kneighbors(mzgdiffs)
                indcheck = np.logical_or(mzginds == label, mzginds == 0) #using the 0-ind to catch all the negligible difference isotopomers that only have a ~different proton - it's not perfect, can probably be made up for afterwards
                linkedindices = []
                for i in np.where(indcheck)[0]:
                    li = [i, i+1]
                    linkedindices.append(li)
                    #print(label, cg)
                    #print(mzg)
                    #print(mzgdiffs)
                    #print('~~~~')
                linkedmassinds = generic_meta_overlap(linkedindices)
                for links in linkedmassinds:
                    isotopes[label].add(tuple(mzg[links].flatten().tolist()))

matches = {}
badlocs = {}
chargefailures = []
for gn, (r, c) in enumerate(zip(randists, randomcharges)):
    setupmasses = set(list(r.keys()))
    for grc, g in isotopes.items():
        for nn, ig in enumerate(g):
            ig = set(ig)
            if setupmasses == ig:
                if c != grc:
                    chargefailures.append(gn)
                matches[gn] = 1
            elif setupmasses < ig:
                matches[gn] = len(setupmasses) / len(ig)
                badlocs[gn] = [grc, nn]
                if c != grc:
                    chargefailures.append(gn)
            elif ig < setupmasses:
                matches[gn] = len(ig) / len(setupmasses)
                badlocs[gn] = [grc, nn]
                if c != grc:
                    chargefailures.append(gn)

matchcounter = Counter(matches.values())
print(matchcounter)
marray = np.array(list(matches.values()))
print((marray == 1).sum(), '/', ngenerated, 'matched')


plt.bar(mergeddict.keys(), mergeddict.values(), color='white', width=0.01)
plt.show()

#resolutiongoal = np.diff(chargevalues[::-1]).min() / 2
#
##a boundary that better splits the means -> like a time series. Sort the differences from lowest to highest, Once you add a value that changes how many values are on one side of the mean too drastically -> that's the end?
##there's issues for when there's only two masses also, the arraydiffs/metadiffs reduce the complexity instead of increasing it.
#

#mzdiffmatrix = mza - mza.flatten()
#mzdiffs = np.unique(np.abs(mzdiffmatrix))
#
#difftree = spatial.KDTree(chargevalues[:,None])
#md, mi = difftree.query(mzdiffs[:,None], k=1)
#md = md[md < scope][1:]
#cutoff = counting_sum_cutoff(md)
#
#
##consecutivediffs = np.diff(np.unique(mzdiffs.round())) > 1
##cutoff = np.where(consecutivediffs)[0][0]
###envelopedifferences = mzdiffs[~(mzdiffs.round() > cutoff)]
#envelopedifferences = mzdiffs[mzdiffs < scope]
##envelopedifferences = envelopedifferences[1:] #removing zero
#
#mzaleftovers = mza.copy()
#massfactor = 1
#lasteds = []
##maxallowabledists = []
#linkeddifferences = []
#while envelopedifferences.size > 0: #actually, I don't like stretching this
#
#    dists, distinds = difftree.query(envelopedifferences[:,None])
#
#    #leadingzerocounts = np.array([leading_zero_count(i) for i in dists]) #maybe try base 2 later
#    #decimalcounts = Counter(leadingzerocounts.tolist())
#    ##decimaldiffs = np.diff(list(decimalcounts.values())[::-1])[::-1] #not going to prepend because those are numbers > 1, they may out number the rest at worst, and at best don't provide anything
#    #decimaldiffs = np.diff(list(decimalcounts.values())[1:])
#    #decimalcutoff = decimaldiffs.argmax() + 2
#    #decimalkey = list(decimalcounts.keys())[decimalcutoff]
#    #maxallowabledists = dists[leadingzerocounts >= decimalkey].max()
#    #mad = maxallowabledists[0] if maxallowabledists[0] < resolutiongoal else resolutiongoal
#    #mad = maxallowabledists.copy()
#
#    #cutoff = np.average(dists, weights=1/dists**2)
#    keeperbool = dists <= cutoff
#    keeperinds = distinds[keeperbool]
#    keepers = envelopedifferences[keeperbool]
#
#    linkeddifferences.extend(np.argwhere(mzdiffmatrix == keepers[:,None,None])[:,1:].tolist())
#    isotopegroups = generic_meta_overlap(linkeddifferences)
#
#    #charges = chargearray[keeperinds]
#    #isotopecharges = defaultdict(set)
#    #for i, c in zip(linkeddifferences.tolist(), charges.tolist()):
#    #    for n, ig in enumerate(isotopegroups):
#    #        if i[0] in ig:
#    #            isotopecharges[n+1].add(c)
#
#    incorporatedmasscount = sum(len(i) for i in isotopegroups)
#    if incorporatedmasscount == len(mzaleftovers):
#        break
#    elif np.all(envelopedifferences * massfactor == lasteds):
#        break
#
#    lasteds = envelopedifferences * massfactor
#    massfactor += 1
#    envelopedifferences = lasteds[~keeperbool] / massfactor
#
#ims = np.array([mza[i].mean() for i in isotopegroups])
#isotopegroups = [isotopegroups[i] for i in np.argsort(ims).tolist()]
#
#for ig in isotopegroups:
#    masses = mza[ig]
#    sorteddiffs = np.diff(np.sort(masses, axis=0), axis=0)
#    print(masses)
#    print(sorteddiffs)
#    print('~~~')

#latechargearray = np.append(0, chargearray)
#latechargevalues = np.append(0, chargevalues)
#latedifftree = spatial.KDTree(latechargevalues[:,None])
#
#mincharge = chargevalues.min()
#isotopecharges = defaultdict(set)
#for n, i in enumerate(isotopegroups):
#    group = mza[i]
#    sorteddiffs = np.diff(np.sort(group, axis=0), axis=0)
#    chargedists, chargeinds = latedifftree.query(sorteddiffs)
#    chargeinds = chargeinds[latechargearray[chargeinds] != 0]
#    
#    if np.all(chargeinds == chargeinds[0]):
#        charge = latechargearray[chargeinds[0]]
#    else:
#        charge = latechargearray[stats.mode(chargeinds)[0][0]]
#        print('dispute', n)
#        print(chargeinds)
#    isotopecharges[n] = charge
#
#matches = {}
#chargefailures = {}
#for r, c in zip(randists, randomcharges):
#    setupmasses = set(list(r.keys()))
#    for n, g in enumerate(isotopegroups):
#        grc = isotopecharges[n]
#        ig = set(mza[g].flatten().tolist())
#        if setupmasses == ig:
#            if c != grc:
#                chargefailures[n] = True
#            matches[n] = 1
#        elif setupmasses < ig:
#            matches[n] = len(ig) / len(setupmasses)
#            if c != grc:
#                chargefailures[n] = True
#        elif ig < setupmasses:
#            matches[n] = len(setupmasses) / len(ig)
#            if c != grc:
#                chargefailures[n] = True


#matchcounter = Counter(matches.values())
#print(matchcounter)
#marray = np.array(list(matches.values()))
#print((marray == 1).sum(), '/', ngenerated, 'matched')
#
#gc.collect()

#plt.bar(mergeddict.keys(), mergeddict.values(), color='white', width=0.02)
#plt.show()


#mzdiffmatrix = mza - mza.flatten()
#mzdiffs = np.unique(np.abs(mzdiffmatrix))
#consecutivediffs = np.diff(np.unique(mzdiffs.round())) > 1
#cutoff = np.where(consecutivediffs)[0][0]
##envelopedifferences = mzdiffs[~(mzdiffs.round() > cutoff)]
#envelopedifferences = mzdiffs[mzdiffs < scope]
#envelopedifferences = envelopedifferences[1:] #removing zero

#chargecutoffs = {}
#cediffs, chargeestimates = difftree.query(envelopedifferences[:,None], k=1)
#for pc in np.unique(chargeestimates):
#    workinginds = chargeestimates == pc
#    workingce = cediffs[workinginds]
#    cemean = np.average(workingce, weights=1/workingce)
#    chargecutoffs[pc] = cemean
#chargemeans = np.array(list(chargecutoffs.values()))
#cutoff = np.average(chargemeans, weights=1/chargemeans)




#proton = 1.00727647
#scope = proton * 1.2
#chargearray = np.arange(maxcharge) + 1
#chargevalues = proton / chargearray
#
##a boundary that better splits the means -> like a time series. Sort the differences from lowest to highest, Once you add a value that changes how many values are on one side of the mean too drastically -> that's the end?
##there's issues for when there's only two masses also, the arraydiffs/metadiffs reduce the complexity instead of increasing it.
#
#mza = np.array(list(mergeddict.keys()))[:,None]
#intensities = np.array(list(mergeddict.values()))
#resolutiongoal = np.diff(chargevalues[::-1]).min()
#
#isotopes = defaultdict(list)
#
#latechargearray = np.append(0, chargearray)
#latechargevalues = np.append(0, chargevalues)
#latedifftree = spatial.KDTree(latechargevalues[:,None])
#
#difftree = spatial.KDTree(chargevalues[:,None])
#isomodel = spatial.KDTree(mza)
#isotopewindows = generic_meta_overlap(i for i in isomodel.query_ball_point(mza, r=scope) if i)
#for ni, iw in enumerate(isotopewindows):
#    lasteds = []
#    isomasses = mza[iw]
#    #isolen = len(isomasses)
#    while isomasses.size > 0:
#        mzdiffmatrix = isomasses - isomasses.flatten()
#        sorteddiffs = np.diff(np.sort(isomasses, axis=0), axis=0)
#        uniquediffs = np.unique(np.abs(mzdiffmatrix))
#        uniquediffs = uniquediffs[uniquediffs < scope, None]
#        isodists, chargeinds = latedifftree.query(uniquediffs, k=1)
#        labelsums = {}
#        labelmeans = {}
#        for label in np.unique(chargeinds):
#            if label > 0:
#                linds = chargeinds == label
#                if linds.sum() > 1:
#                    lindists = isodists[linds]
#                    labelsums[label] = np.abs(lindists[:,None] - lindists).sum() / linds.sum()
#                    labelmeans[label] = uniquediffs[linds].mean()
#        #automatically defer to lowest total distance label
#        for label in np.unique(chargeinds):
#            if label > 0:
#                linds = chargeinds == label
#                if linds.sum() > 1:
#                    #sorted differences for isodiffs need to be used to elucidate separate distributions
#                    isodiffs = uniquediffs[linds]
#                    matchedinds = np.unique(np.argwhere(mzdiffmatrix == isodiffs[:,None])[:,1:])
#                    matchedgroup = isomasses[matchedinds]
#                    interisodists = np.unique(np.abs(isodiffs - isodiffs.flatten()))
#                    diststocv = isodists[linds]
#                    if interisodists.sum() < diststocv.sum(): #the more important thing to match seemed to be the consistency of the distance, rather than matching closesly to the theoretical charge distance. I may make this a mean with an n-1 base on interisodists because that always has a 0 value included.
#                        charge = latechargearray[label]
#                        isotopes[charge].append(matchedgroup.flatten().tolist())
#                    else:
#                        matchedinds = np.unique(np.argwhere(mzdiffmatrix == isodiffs[:,None])[:,1:])
#                        printints = intensities[iw][matchedinds]
#                        print('~~~~~~~~~')
#                        print(ni)
#                        for p1, p2 in zip(isodiffs, printints):
#                            print(p1, p2)
#                        plt.bar(isomasses.flatten(), intensities[iw].flatten(), color='white', width=0.01)
#                        plt.show()
#        #remove the accounted for values from isomasses so as to not waste time on charges that are 2x the correct ones
#        if np.all(isomasses == lasteds):
#            break
#        lasteds = isomasses.copy()

#sever max distance on lower and upper sides of isodiffs
#if the max increases, keep going, if the max decreases, stop I suppose


    #mzdiffmatrix = isomasses - isomasses.flatten()
    #mzdiffs = np.unique(np.abs(mzdiffmatrix))[1:]
    #mzdiffs = mzdiffs[mzdiffs < scope]
    ##maxallowabledist = np.average(mzdiffs, weights=1/mzdiffs)
    ##chargechecker = mzdiffs / chargevalues[:,None]
    ##plt.hist((np.abs(chargechecker.round() - chargechecker)).flatten(), bins=100, color='white')
    ##plt.show()

    #massfactor = 1
    #lasteds = []
    #maxallowabledists = []
    #linkeddifferences = []
    #while mzdiffs.size > 0:
    #    dists, distinds = difftree.query(mzdiffs[:,None])
    #    
    #    leadingzerocounts = np.array([leading_zero_count(i) for i in dists]) #maybe try base 2 later
    #    decimalcounts = Counter(leadingzerocounts.tolist())
#dec#imaldiffs = np.diff(list(decimalcounts.values())[::-1])[::-1] #not going to prepend because those are numbers > 1, they may out number the rest at worst, and at best don't provide anything
    #    #decimaldiffs = np.diff(list(decimalcounts.values())[1:])
    #    #decimalcutoff = decimaldiffs.argmax() + 2
    #    #decimalkey = list(decimalcounts.keys())[decimalcutoff]
    #    #maxallowabledists = dists[leadingzerocounts >= decimalkey].max()
    #    #mad = maxallowabledists[0] if maxallowabledists[0] < resolutiongoal else resolutiongoal
    #    #mad = maxallowabledists.copy()

#cut#off = np.average(dists, weights=1/dists**2)
    #    keeperbool = dists <= cutoff
    #    keeperinds = distinds[keeperbool]
    #    keepers = mzdiffs[keeperbool]

    #    linkeddifferences.extend(np.argwhere(mzdiffmatrix == keepers[:,None,None])[:,1:].tolist())
    #    isotopegroups = generic_meta_overlap(linkeddifferences)
    #    
    #    incorporatedmasscount = sum(len(i) for i in isotopegroups)
    #    if incorporatedmasscount == isolen:
    #        break
    #    #elif mzdiffs * massfactor == lasteds:
    #    elif set(mzdiffs * massfactor) == set(lasteds):
    #        break

    #    lasteds = mzdiffs * massfactor
    #    massfactor += 1
    #    mzdiffs = lasteds[~keeperbool] / massfactor
    #isotopegroups = generic_meta_overlap(linkeddifferences)
    #
    #isotopecharges = defaultdict(set)
    #for n, i in enumerate(isotopegroups):
    #    group = isomasses[i]
    #    sorteddiffs = np.diff(np.sort(group, axis=0), axis=0)
    #    chargedists, chargeinds = latedifftree.query(sorteddiffs)
    #    chargeinds = chargeinds[latechargearray[chargeinds] != 0]
    #    
    #    if np.all(chargeinds == chargeinds[0]):
    #        charge = latechargearray[chargeinds[0]]
    #    else:
    #        charge = latechargearray[stats.mode(chargeinds)[0][0]]
    #        print('dispute', sorteddiffs)
    #    isotopecharges[n] = charge

    #for (k, v), i in zip(isotopecharges.items(), isotopegroups):
    #    isotopes[v].append(isomasses[i].tolist())


#while True:
#    dists, distinds = difftree.query(envelopedifferences[:,None])
#    cutoff = np.average(dists, weights=1/dists)
#    keepers = 
#
#
#metadiffmatrix = envelopedifferences[:,None] - envelopedifferences
#metadiffs = np.abs(metadiffmatrix).reshape(-1,1)
#metadiffs = np.unique(metadiffs[metadiffs > 0,None], axis=0) #assuming complete uniqueness
#
#isomodel = spatial.KDTree(mza)
#isotopewindows = generic_meta_overlap(i for i in isomodel.query_ball_point(mza, r=scope) if i)
##isotopewindows = #filter for > len == 1?
#chargepatterns = defaultdict(list)
#meanshifter = MeanShift()
##Instead of focussing around mean centers, maximize uninterupted distance between groups?
#for iw in isotopewindows:
#    isomasses = mza[iw]
#    isomassdiffmatrix = isomasses.flatten() - isomasses
#    uniquediffs = np.unique(np.abs(isomassdiffmatrix).flatten())[1:,None]
#    workingdiffs = uniquediffs.copy()
#    #similarityorder = np.abs(workingdiffs - chargevalues).min(axis=0).argsort()
#    clusters = meanshifter.fit_predict(workingdiffs)
#    centers = meanshifter.cluster_centers_
#    multipliers = np.arange(centers.size) + 1
#    centermin = centers.min()
#    massmultiples = meanshifter.predict(centermin * multipliers[:,None])
#    charge = chargevalues[np.abs(chargevalues - centermin).argmin()]
#    metadiffmatrix = workingdiffs - workingdiffs.flatten()
#    metadiffs = np.abs(metadiffmatrix).reshape(-1,1)
#    metadiffs = np.unique(metadiffs[metadiffs > 0,None], axis=0) #assuming complete uniqueness
#    difftree = spatial.KDTree(metadiffs)
#    for c, l in zip(centers.tolist(), np.unique(clusters).tolist()):
#        group = workingdiffs[clusters == l]
#        maxgroupdist = np.abs(group - group.flatten()).max()
#        centerminmultiples = massmultiples[centermin * multipliers - centers.flatten() <= maxgroupdist]
#        removalinds = (clusters == centerminmultiples[:,None]).any(axis=0)
#        groupedmasses = generic_meta_overlap(np.argwhere(isomassdiffmatrix == workingdiffs[removalinds,None])[:,1:].tolist())
#    workingdiffs = uniquediffs.copy()
#    for s in similarityorder: #shouldn't iterate here, should only do the top match for what's unaccounted for in uniquediffs?
#        c = chargevalues[s]
#        charge = s + 1
#        metadiffmatrix = workingdiffs - workingdiffs.flatten()
#        metadiffs = np.abs(metadiffmatrix).reshape(-1,1)
#        clusters = meanshifter.fit_predict(metadiffs)
#        #^find means of clusters, then find multiples of each lowest factor
#        descendinggroups = np.unique(meanshifter.fit_predict(workingdiffs)).size
#        multiplierinds = np.arange(workingdiffs.size) + 1
#        multiplierarray = multiplierinds * c
#        workingmultipliermatrix = np.abs(workingdiffs -  multiplierarray)
#        foundmultipliers = workingmultipliermatrix.min(axis=0) / multiplierinds
#        clusters = meanshifter.fit_predict(foundmultipliers[:,None])
#        arraydiffs = workingmultipliermatrix.min(axis=1)
#        workingdiffbranchs = np.abs(workingdiffs - workingdiffs.flatten()).max(axis=0)
#        metadiffmatrix = np.abs(arraydiffs[:,None] - arraydiffs)
#        metadiffs = metadiffmatrix.flatten()
#        metadiffs = metadiffs[metadiffs > 0]
#        #aim for the largest symmetrical group you can get from the 0-side?
#        for m in np.sort(metadiffs)[::-1]:
#            iterators = np.unique(np.abs(arraydiffs[:,None] - arraydiffs) <= m, axis=0)
#            for i in iterators.tolist():
#                if i.sum() > 1:
#
#        acceptableranges = []
#        accountedindices = set()
#        while True:
#            isodists, distinds = difftree.query(c*mul, k=len(uniquediffs))
#            firstordermean = isodists.cumsum() / (np.arange(isodists.size) + 1)
#            diststomean = isodists - firstordermean
#            try:
#                cutoff = np.where(diststomean > firstordermean)[0][0]
#                acceptableranges.extend(isodists[:cutoff]/mul)
#                accountedindices.update(distinds[:cutoff])
#            except IndexError:
#                ar = np.array(acceptableranges)
#                cutoff = ar.mean() + (ar.mean() - ar).max()
#                if firstordermean[0] / mul < cutoff:
#                    accountedindices.add(distinds[0])
#                break    
#            mul += 1
#        
#        if accountedindices:
#            chargepatterninds = np.argwhere(isomassdiffmatrix == uniquediffs[list(accountedindices)][:,None])[:,1:]
#            chargepattern = [isomasses[i].flatten().tolist() for i in generic_meta_overlap(chargepatterninds.tolist())]
#            chargepatterns[charge].extend(chargepattern)
#        
#        if len(accountedindices) == uniquediffs.size:
#            break
