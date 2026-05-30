#Some dump that used to be in reverseskeletongrouping.py that didn't need to be in the mix.
#examples made using drosophila proteome

#instead of this, you can do a 'fill up to minimum'. So whatever the lowest peptide length is, everything below that can just be raw skellies: NxM, NxxM, NxxxM, etc.
if addtwos:
    twos = list('..'.join((i)) for i in itertools.combinations_with_replacement(list(codons.keys()), 2))
    groupings.extend(twos)


#find peptides with (insert n-length sequence here) on either flank. Check for overlap among peptides that have this flank. The only different parameter here would be how big n can be.
#Afterward and during, try to reduce the number of peptides in groupings info a new list. Perhaps, when it comes to even larger sequences, it would be wiser to ignore n. And just take sequences that have a low levenshtein distance to each other and see if their ends match up at any level?
#Actually, increasing n might be better than keeping it small - because if you're confirming their matches by searching the proteome to cross-examine the number of hits, then you'll get way too many hits for searching with just an n=1
#^Then the best case is to pick a max, and deal with peptides at a length of n+2 or 2+3 or something in a lesser case. Like storing everything in a trie that's limited to n for the starting piece.

scaffolds = {i for i in groupings if '.' in i}
#EY.GDPY appears twice in this when in list form, double check sequenceclustering to figure out why

groupdict = defaultdict(list)
for g in groupings:
    if '.' not in g:
        groupdict[len(g)].append(g)

patternsplits = defaultdict(set)
overlapsubtraction = 1 #I don't think there's a huge benefit to having this be greater than 1, should save time too
t = time()
for size, peps in groupdict.items():
    for pep in peps:
        patternsplits[size].update(pep[i:i+(len(pep) - overlapsubtraction)] for i in range(len(pep)-(len(pep) - overlapsubtraction)+1))
print(time() - t)

#t = time()
#n = 0
#patterngroupings = []
#for size, peps in patternsplits.items():
#    for pep in peps:
#        patterngroupings.append([i for i in groupdict[size] if pep in i])
#    n += 1
#    print(time() - t, f'- {n}/{len(patternsplits)} pattern splits')
#print(time() - t)

def pepfind(pep, gds):
    return [i for i in gds if pep in i]

patterngroupings = []
t = time()
with concurrent.futures.ThreadPoolExecutor(8) as executor:
    futures = []
    for size, peps in patternsplits.items():
        for pep in peps:
            futures.append(executor.submit(pepfind, pep, groupdict[size]))
    for future in concurrent.futures.as_completed(futures):
        patterngroupings.append(future.result())
print(time() - t, 'threaded')


rings = to_graph(patterngroupings)
rings = [tuple(i) for i in connected_components(rings)]

#trying to get rid of groups with tens of thousands of linkers, I attempted to do one without concurrency, it did not finish - but none of the assembled sequences were in the proteome. This is a hack to keep them out, I don't know if it's robust, I don't believe it is. Max-values are for losers. I also don't know what a viable max-value would be, and guessing is for losers.
#I would like to either have this be a user-input spot, or have it at least notify you that a group of 8-bajillion was left out for common decency.
ringcounts = Counter(len(i) for i in rings)
ringsizes = np.sort(list(ringcounts.keys()))
cumulativeringsizes = np.cumsum(ringsizes)
filtertest = ringsizes[1:] > cumulativeringsizes[:-1]
if filtertest[-1]:
    fn = -1
    while True:
        if filtertest[fn-1]:
            fn -= 1
        else:
            break
    rings = [i for i in rings if len(i) < ringsizes[fn]]

#for testing purposes
#out = [i for i in rings if len(i) == 433][0]

#initial exploration
#npatternsplits = []
#for o in out:
#    npatternsplits.append(o[1:])
#    npatternsplits.append(o[:-1])
#npatternsplits = Counter(npatternsplits)
#patternsbycount = defaultdict(set)
#for npk, npv in npatternsplits.items():
#    patternsbycount[npv].add(npk)
#
#problemchildren = [i for i in out if any(pi in i for pi in npatternsplits.keys())]
#
##pick the highest ones from npatternsplits, in that order
##assemble an initial skeleton with whatever interacts with those problem children
##use whats been assembled as the initial endstring for the next or final level
##loop around the 1's in patternsbycount, if these spawn multiple end strings, then those are what get output. different procedures for differents #s in patternsbycount, as they come up as o
#
#boundary = len(out) + len(out[0]) - 1
#endstring = out[0]
#alteredlist = list(out[1:])
#olen = len(endstring)
#lastlen = olen
#wn = 0
#failed = False
#while True:
#    indshift = len(out) - len(alteredlist) - 1
#    o = alteredlist[wn]
#    #if o[1:] in endstring:
#    if o[1:] == endstring[:olen-1]:
#        endstring = ''.join((o[0], endstring))
#        del alteredlist[wn]
#    #elif o[:-1] in endstring:
#    elif o[:-1] == endstring[-olen+1:]:
#        endstring = ''.join((endstring, o[-1]))
#        del alteredlist[wn]
#    else:
#        wn += 1
#    if wn >= len(alteredlist):
#        if lastlen == len(endstring):
#            failed = True
#        lastlen = len(endstring)
#        wn = 0
#    if len(endstring) >= boundary or failed:
#        break

#this produces (hopefully this is reproducible b/c of the multithreading) a sequence that's actually in the proteome!
#joinedproteome.count(endstring)
#BUT it failed, because of what's left over in alteredlist

#looking at the beginning set, there's clearly a similarity between what's left over in alteredlist, and what sequence show up multiple times!

rcounts = Counter(len(i) for i in rings)
#it wasn't this, everything was the same EXCEPT 1:63402 instead of 63403. Suspicious, but not a massive difference.. I'll have to look into it. But this points to stochasticity within the assembler.
#Counter({2: 20711,
#         1: 63403,
#         9: 207,
#         3: 8763,
#         7: 617,
#         4: 4013,
#         6: 1091,
#         5: 2093,
#         10: 133,
#         11: 85,
#         8: 332,
#         12: 45,
#         16: 9,
#         18: 5,
#         26: 1,
#         13: 30,
#         17: 2,
#         15: 6,
#         14: 13,
#         81: 1,
#         19: 3})
#out = [i for i in rings if 'YHHPH' in i][0]
#shuffling this creates an output from the below assembly process of different lengths!

#t = time()
#narrowedsequences = []
#ts = 0
#for out in rings:
#    if len(out) > 1:
#        npatternsplits = []
#        for o in out:
#            npatternsplits.append(o[1:])
#            npatternsplits.append(o[:-1])
#        npatternsplits = Counter(npatternsplits)
#        patternsbycount = defaultdict(set)
#        for npk, npv in npatternsplits.items():
#            patternsbycount[npv].add(npk)
#
#        singlets = [i for i in out if any(v in i for v in patternsbycount[1])]
#        #boundary = len(out) + len(singlets) - 1 #doesn't seem necessary, it's also hard to accurately determine this because I can't immediately tell how many singlets all work on one specific side of the sequence. ie whether having 4 singlets mean 3 of them are n-term starts and 1 c-term or 2 and 2 complicates the exact answer.
#
#        #pick the highest ones from npatternsplits, in that order
#        #assemble an initial skeleton with whatever interacts with those problem children
#        #use whats been assembled as the initial endstring for the next or final level
#        #loop around the 1's in patternsbycount, if these spawn multiple end strings, then those are what get output. different procedures for differents #s in patternsbycount, as they come up as o.
#
#        endstrings = set() #there is some redundant assembly that goes on, no way of knowing beforehand which ones will be redunant so there's just going to be wasted processing as a biproduct. There might be some way to stop this on the fly but it's pretty fast so I don't see the need.
#        for o in singlets: #I have not yet seen any sequence list that show fully circular logic. ie where nothing never matches anything only once, probably not an actual glitch in the matrix if it does happen, so much as there's just multiple possible endings, and the endings show up in all of their potential combinations.
#        #^It cold also mean, the beginning sequence is the same as the end.
#            alteredlist = list(out)
#            alteredlist.remove(o)
#            alteredlist = [alteredlist]
#            endstring = [o]
#            olen = len(o)
#            lastsum = olen
#            while True:
#                for en in range(len(endstring)):
#                    subalteredlist = alteredlist[en]
#                    if subalteredlist:
#                        es = endstring[en]
#                        lastlen = len(es)
#                        wn = 0
#                        ended = False
#                        while True:
#                            so = subalteredlist[wn]
#                            addition = False
#                            if so[1:] == es[:olen-1]:
#                                es = ''.join((so[0], es))
#                                sonum = npatternsplits[so[:-1]]
#                                left = True
#                                addition = True
#                                subalteredlist.remove(so)
#                            elif so[:-1] == es[-olen+1:]:
#                                es = ''.join((es, so[-1]))
#                                sonum = npatternsplits[so[1:]]
#                                left = False
#                                addition = True
#                                subalteredlist.remove(so)
#                            else:
#                                wn += 1
#
#                            if addition and sonum > 2:
#                                overlaps = []
#                                if left:
#                                    for ovo in out:
#                                        if so[:-1] == ovo[1:]:
#                                            overlaps.append(ovo)
#                                else:
#                                    for ovo in out:
#                                        if so[1:] == ovo[:-1]:
#                                            overlaps.append(ovo)
#                                if overlaps: #in the case that these aren't found?? It happened once on a ~60000 length out where left=True, it should have found something, no? I'm not sure atm
#                                    newstrings = []
#                                    for no, ovo in enumerate(overlaps):
#                                        if left:
#                                            novo = ''.join((ovo[0], es))
#                                        else:
#                                            novo = ''.join((es, ovo[-1]))
#                                        if no == 0:
#                                            esholder = novo
#                                        else:
#                                            endstring.append(novo)
#                                            alteredlist.append(subalteredlist)
#                                            try:
#                                                alteredlist[-1].remove(ovo)
#                                            except ValueError: #sequence already applied previously
#                                                pass
#                                    es = esholder
#                                    try:
#                                        subalteredlist.remove(overlaps[0])
#                                    except ValueError: #sequence already applied previously
#                                        pass
#
#                            if wn >= len(subalteredlist):
#                                if lastlen == len(es):
#                                    ended = True
#                                lastlen = len(es)
#                                wn = 0
#                            #if ended or len(es) >= boundary or not len(subalteredlist):
#                            if ended or not len(subalteredlist):
#                                break
#                    endstring[en] = es
#                    alteredlist[en] = subalteredlist
#                newsum = sum(len(i) for i in endstring)
#                if newsum == lastsum:
#                    break
#                lastsum = newsum
#
#            endstrings.update(i for i in endstring if i in joinedproteome)
#        #..more operations here potentially, what's faster?
#        narrowedsequences.append(endstrings)
#        ts += len(endstrings)
#    else:
#        narrowedsequences.append(set(out))
#narrowedsequences = [i for i in narrowedsequences if i] #removing empty set()
#print(time() - t, f'- {ts} successful assemblies')

#with the above, due to the removals done in alteredlist and subalteredlist, the output can be stochastic with a reshuffling of out
#without removals in alteredlist and subalteredlist, it gets caught in an infinite loop due to self-overlapping sequences

#new sequence assembly mechanism:
#iterate through out one at a time, and assemble a set each possible n+1 string
#each merged string is output into a new list, which is the target of the same process
#rinse and repeat
#build in a self, and repeating, overlap detection on this by making sure a peptide isn't contained within a string it's supposed to elongate

t = time()
narrowedsequences = set()
totalseqs, goodseqs = 0, 0
for out in rings:
    if len(out) > 1:
        npatternsplits = []
        for o in out:
            npatternsplits.append(o[1:])
            npatternsplits.append(o[:-1])
        npatternsplits = Counter(npatternsplits)
        patternsbycount = defaultdict(set)
        addtostart, addtoend = defaultdict(set), defaultdict(set)
        for ns in npatternsplits:
            for o in out:
                if o.startswith(ns):
                    addtoend[ns].add(o)
                if o.endswith(ns):
                    addtostart[ns].add(o)
        runningset = set(out)
        lastlen = len(out)
        lastsum = sum(len(i) for i in out)
        olen = len(out[0]) - 1 #assuming they're all the same length here, it tends to turn out that way but this may be a bad long-term assumption
        while True:
            newsetconstructor = set()
            for r in runningset:
                newstarts = addtostart[r[:olen]]
                newends = addtoend[r[-olen:]]
                if newends or newstarts:
                    for ne in newends:
                        if ne not in r:
                            newsetconstructor.add(''.join((r, ne[-1])))
                    for ns in newstarts:
                        if ns not in r:
                            newsetconstructor.add(''.join((ns[0], r)))
                else:
                    newsetconstructor.add(r)
            newlen = len(newsetconstructor)
            newsum = sum(len(i) for i in newsetconstructor)
            if newlen == lastlen and newsum == lastsum:
                break
            else:
                lastlen = newlen
                lastsum = newsum
                runningset = newsetconstructor.copy()
        totalseqs += len(runningset)
        outseqs = tuple(sorted(i for i in runningset if i in joinedproteome))
        if outseqs:
            narrowedsequences.add(outseqs)
            goodseqs += len(outseqs)
    else:
        narrowedsequences.add(out)
print(time() - t, f'- {goodseqs}/{totalseqs} assemblies found')




#1. sequence overlap alignment to figure out initial wildcards
#   A) Pillar equillibrium for positional alignments
#       * I'm going to avoid using the codon distance here because I can't actually say what any actual phylogenetic distance is between any of these. The larger, unshown, sequences as a whole may hold more information about that, but tiny snippets don't necessarily present accurate significance. Levenshtein distance works.
#       - Assemble using the same 1's trick from where you avoided a metric in the initial clustering. This time, reward the # of 1's present, ignore any other digit. Sum the 1's to get a final score. Going purely by levenshtein distance would favor sequences only matching the first and last of different sequences, ending in a staircase pattern where nothing is matched (probably). Maximize the # of 1s on any axis. A lack of overlap due to different sized sequences can still be seen as a 1. It's the same trick used twice now, but this time I'm trying to assemble an alignment rather than score a cutoff.
#       - ^Actually, it would be cool to just minimize the sum of unique #s along each axis considering that this is of a fixed length and isn't being compared across groups. I wonder how this would compare to the 1's approach? And likewise, using this strategy, you could implement the codon-distance.
#       - ^Actually this wouldn't accomplish that. You want to maximize 1's but minimize any other number.. sum does what?
#2. moving window search
#   - for scaffolds going into this process, it might be good to be able to return a wild-card position to an AA if it seems appropriate. In fact, this might be the proper initial process for handling scaffolds going into the moving window. Everything else would just start with the moving window, but scaffolds would start with a reversed one first, then go into the normal moving window.
#   - ^It would be good to trim wild cards at the end of sequences, this could also be used as an excess removal process for shitty sequences.
#   - This could also become a split motif search, where a sequence can deviate in like 2 directions.
#3. expanding window search
#4. Are any motifs a reverse of another motif? Can sequence reversals be used to show differences in frequency for initial motif selection? Could shuffling then also be used?


#This shows a bias stemming from the chosen spacing distances for patternspace.
#It would be interesting to know if this bias is reflected in the post time-series analysis.
#It could also come from the fact that some popular sequences were found but their appropriate peripheral sequences were less likely to be found. Actually, seeing as the leftover 5s are the least, I find it really interesting that these fan out in intervals of 5s but any 5-length sequences have clearly been assembled into larger sequences. Maybe this bias isn't going to be as important? If, after shifting sequence sizes, most of the same pattern splits are present, then this bias is meaningless and the same sequences are found regardless.
#Perhaps the expanding window will take care of this?
lencounts = Counter(len(''.join((i))) for i in narrowedsequences if len(i) == 1)
fig, ax = plt.subplots(figsize=(12,9))
ax.bar(lencounts.keys(), lencounts.values())
plt.show()



#positionaldict = {}
#maxlen = len(nout[0])
#workinglist = []
#starterspace = ' '*maxlen
#workinglist.append(starterspace)
#initialcenter = len(workinglist[0]) // 2 - maxlen // 2
#initiallength = maxlen + initialcenter
#workinglist.append(''.join((' '*initialcenter, nout[0], ' '*(maxlen-initiallength))))
#sn = 1
#
##maximize # of 1's, else minimize # of 2's/3's etc until there's only one best answer
##do more if there's multiple of the same maxes of 1's
#
#while True:
#    seq = nout[sn]
#    if len(workinglist) < len(nout):
#        workinglist.append(seq)
#
#    if len(workinglist) > 2:
#        rn = - 1
#        rw = workinglist[-1]
#        rawseq = rw.strip()
#        rlen = len(rawseq)
#        alignmentmeasures = []
#        for si in range(maxlen-rlen):
#            workinglist[rn] = ''.join((' '*si, rawseq, ' '*(maxlen-si-rlen)))
#            #alignmentmeasures.append([len(set(i))-1 for i in map(list, zip(*workinglist))].count(1))
#            #metrics = [len(set(i))-1 for i in map(list, zip(*workinglist))]
#            metrics = [len(set(i))-1 for i in map(list, itertools.zip_longest(*workinglist))]
#            alignmentmeasures.append(np.product(metrics))
#        #winner = np.abs(np.diff(np.diff(alignmentmeasures))).argmax() + 1
#        #winner = np.argmax(alignmentmeasures)
#        winner = np.argmin(alignmentmeasures)
#        workinglist[rn] = ''.join((' '*winner, rawseq, ' '*(maxlen-winner-rlen)))
#
#    sn += 1
#    if sn >= len(nout):
#        break
#for i in workinglist:
#    print(i)


#new strategy:
#sum of 1's multiplied by the length of the transpose
#so that each additional match is always favored
#final step should actually be removing individual sequences, operating like the moving window idea, to see if removing any sequences improves the output score - if it does, then that sequence can be in its own category. But more work would be needed to see if it's relevant to the proteome.. something that shows up once is useless.
#^But that brings up an interesting idea, what if I clustered for sequences that are RARELY found, as opposed to always found. Would this imply a significance? This idea actually would be more easily supported by the on-the-spot search for sequences. After the time-series protein data is being assessed. So doing a database compilation for often found sequences, then a live-data search for lesser known sequences to complement each other, would be the final strategy - WHY NOT BOTH?!
#The logic being, does a super abundant sequence signify function? Or a lackthereof? There's no way to tell, but logic says it could be either. The same reasoning applies to less abundant sequences. Determining which type of sequence makes sense to pull out a priori, is a hard ask. It actually makes more sense to try and determine it from data!

#positionranges, workinglist = [], []
#positionranges.append([0,len(nout[0])])
#workinglist.append(nout[0])
#for no in nout[1:]:
#    nlen = len(no)
#    leftpoint = np.min(positionranges, axis=0)[0]
#    rightpoint = np.max(positionranges, axis=0)[1]
#    currentrange = set(range(leftpoint, rightpoint))
#
#    startingpoints = range(leftpoint - nlen + 1, rightpoint)
#    endingpoints = range(leftpoint + 1, rightpoint + nlen)
#    tempranges = [set(range(s, e)) for s, e in zip(startingpoints, endingpoints)]
#    scores = []
#    for tr in tempranges:
#        rangeoverlaps = tr.intersection(currentrange)
#        rangevalues = sorted(list(tr))
#        score = 0
#        if rangeoverlaps:
#            for ro in rangeoverlaps:
#                #something is wrong here, I think it's not incorporating scores from that part of the sequences that don't overlap with the indices of no. Get the below mechanism with a1/a2 and emptylist into this workflow.
#                aligners = []
#                ind = [n for n, i in enumerate(rangevalues) if i == ro][0]
#                aligners.append(no[ind])
#                for w, p in zip(workinglist, positionranges):
#                    positionrange = set(range(*p))
#                    if positionrange.intersection({ro}):
#                        positionvals = sorted(list(positionrange))
#                        positioninds = [n for n, i in enumerate(positionrange) if i == ro]
#                        aligners.append(w[positioninds[0]])
#                if len(set(aligners)) == 1:
#                    score += len(aligners)
#                else:
#                    scorecount = Counter(aligners)
#                    score += sum([i for i in scorecount.values() if i > 1])
#                    #score -= sum([i for i in scorecount.values() if i == 1])
#            scores.append(score)
#    if (np.asarray(scores) == np.max(scores)).sum() > 1:
#        print(no, 'more than 1 max score')
#    chosenval = np.argmax(scores)
#    chosenrange = tempranges[chosenval]
#    workinglist.append(no)
#    positionranges.append([min(chosenrange), max(chosenrange)])
#    #positionranges = np.asarray(positionranges)
#    #positionranges += np.abs(positionranges.min(axis=0)[0])
#    #positionranges = positionranges.tolist()
#
#pmin = np.asarray(positionranges).min(axis=0)[0]
#if pmin < 0:
#    padd = np.abs(pmin)
#else:
#    padd = 0
#for pw, pv in zip(workinglist, positionranges):
#    nspace = ' '*(pv[0]+padd)
#    printstring = ''.join((nspace, pw))
#    print(printstring)
#
##make a codon-distance alignment matrix
##if the final alignment shows up less times in the proteome than any of the original pieces, it's a failure and is discarded.
##combination of every 2, highest grouping matches first, lower matches later. Check after if by removing lower matches that you get a better sequence with more proteome hits.
##not normalizing by length, to favor larger matches. But only overlapping AA's get counted for anything. Can stick to counting 1's.
#
#alignmentcomparator = defaultdict(dict)
#for l, r in itertools.combinations(nout, 2):
#    if len(l) > len(r):
#        base = l
#        mover = r
#    else:
#        base = r
#        mover = l
#    #technicalstart = 1 - len(mover)
#    mlen = len(mover)
#    technicalend = len(base) + mlen - 1
#    #technicaldistance = range(technicalstart, technicalend)
#    technicaldistance = range(technicalend+mlen-1)
#    emptylist = ['' for i in technicaldistance]
#    baselist = []
#    for mi in range(mlen-1):
#        baselist.append('')
#    for bi in base:
#        baselist.append(bi)
#    for mi in range(mlen-1):
#        baselist.append('')
#    scores = []
#    for te in range(technicalend):
#        el = emptylist.copy()
#        el[te:te+mlen] = mover
#        score = 0
#        for a1, a2 in list(zip(baselist, el)):
#            if a1 == a2:
#                score += 1
#            #elif a1 and a2 and a1 != a2:
#            #    score -= codondistances[a1][a2]
#
#        #output = [len(set([a1, a2])) for a1, a2 in list(zip(baselist, el)) if a1 and a2]
#        #output = sum([codondistances[a1][a2] for a1, a2 in map(list, zip(baselist, el)) if a1 and a2])
#        #scores.append(output)
#        scores.append(score)
#    if (np.asarray(scores) == max(scores)).sum() > 1:
#        print(l, r, 'multiple scores')
#        print(scores)
#        print('~')
#    alignmentcomparator[l][r] = np.argmax(scores)


ncounts = Counter(len(i) for i in narrowedsequences)
#Counter({1: 100519, 2: 674, 3: 60, 4: 11, 11: 1, 5: 2}) why is this fucking different every time????
#nout = sorted([i for i in narrowedsequences if len(i) == 8][0], key=lambda x: -len(x))
#nout = sorted([i for i in narrowedsequences if len(i) == 11][0], key=lambda x: -joinedproteome.count(x))
#nout = [i for i in narrowedsequences if 'TMSANK' in i][0]

for nout in narrowedsequences:
    if len(nout) > 2:
        #nout = sorted(list(nout), key=lambda x: -len(x))
        noutcount = {i:joinedproteome.count(i) for i in nout}
        nout = sorted(list(nout), key=lambda x: -noutcount[x])
        mlen = len(nout[0])
        technicalend = (mlen * 2) - 1
        endpoint = technicalend + mlen
        technicaldistance = range(endpoint-1)
        emptylist = ['' for i in technicaldistance]
        baselist = [emptylist.copy()]
        baselist[0][mlen-1:technicalend] = nout[0]

        mlen = len(nout[0])
        #initiation rounds
        for no in nout[1:]:
            nlen = len(no)
            leftpoint = mlen - nlen
            rightpoint = mlen + nlen
            currentrange = range(leftpoint, rightpoint)
            baselist.append(emptylist)

            scores = []
            for pos in currentrange:
                baselist[-1] = emptylist.copy()
                baselist[-1][pos:pos+nlen] = no
                scorearray = []
                scoreinds = []
                subtractor = 0 #this contributes to added complexity for breaking ties! argmax always picks the first member of an array where two elements are equal, without this there's a lot of that going on -> leading to poor selections of some sequences and the snowball effect that continues afterwards. This is a subtle way of subtracting small amounts of blank area from alignments that conveniently tack things on to the ends without a good reason while retaining a score equal to the top legitemate spot.
                #The subtractor basically forces the benefits of a specific alignment to outweigh small costs represented as empty spaces that otherwise wouldn't be present.
                adjacency = False
                for nz, bz in enumerate(zip(*baselist)):
                    outzip = [i for i in bz if i]
                    outlen = len(outzip)
                    if outlen > 1:
                        subtractor += len(bz) - outlen
                        zippedset = set(outzip)
                        if len(zippedset) == 1:
                            scoreinds.append(nz)
                            scorearray.append(outlen)
                        else:
                            counts = Counter(outzip)
                            countsum = sum(i for i in counts.values() if i > 1) #making this into a total sum b/c whenever something is present more than once, it gets invited to the party
                            if countsum:
                                scoreinds.append(nz)
                                scorearray.append(countsum)
                #thanks to https://stackoverflow.com/questions/67852132/multiplying-only-naturally-adjacent-integers-in-a-list
                #score = sum([*map(np.prod,np.split(scorearray,np.where(np.diff(scoreinds)!=1)[0]+1))]) - subtractor
                score = (sum([*map(np.prod,np.split(scorearray,np.where(np.diff(scoreinds)!=1)[0]+1))]) * sum(scorearray)) - subtractor #multiplying by the sum of the scorearray was added because alignments with lower overall sums looked worse, but were scoring higher in the previous schema, than alignments whose scorearrays had higher sums, but multiplied to lower values. It was a little awkward, and really close to call. But adding this in here should add a greater layer of complexity to the overall scoring machine, making ties even less likely.
                #this works well - better than everything else, but still has the same issue with multiple maxes arising
                #the multiple maxes are a mathematical phenomenon of the sequences, and not necessarily a flaw of the metric. It seems to be shorter sequences that always have trouble, for example:
                #~~~~~~~~~~~~~~~~~
                #           DLGPND
                #            NGPLQ
                #          PPLGD
                #~~~~~~~~~~~~~~~~~
                #has 2 max scores, you can either match the LG's, or the PL's.

                scores.append(score)
            winner = np.argmax(scores)
            baselist[-1] = emptylist.copy()
            baselist[-1][currentrange[winner]:currentrange[winner]+nlen] = no
            newrange = [n for n, i in enumerate(baselist[-1]) if i.isalpha()]

            woopsy = len([i for i in scores if i == max(scores)])
            if woopsy > 1:
                print('woopsy', no)

            leftadjustment = newrange[0] - mlen
            rightadjustment = newrange[0] + mlen

            if leftadjustment < 0:
                leftadd = np.abs(newrange[0] - mlen)
                for b in range(len(baselist)):
                    for l in range(leftadd):
                        baselist[b].insert(0, '')
            if rightadjustment > endpoint:
                rightadd = endpoint - rightadjustment
                endpoint = rightadjustment
                for b in range(len(baselist)):
                    for r in range(rightadd):
                        baselist[b].append('')
            emptylist = ['' for i in range(len(baselist[0]))]

        positionranges = {}
        for b in baselist:
            s = 0
            for i in b:
                if not i:
                    s += 1
                else:
                    break
            positionranges[''.join((b))] = s
        minval = min(positionranges.values())
        for k in positionranges:
            positionranges[k] -= minval

        spacepad = 10
        for pw, pv in positionranges.items():
            nspace = ' '*(pv+spacepad)
            printstring = ''.join((nspace, pw))
            print(printstring)
        print('~~~')

        #equillibration rounds
        #all of these examples were done by sorting nout via peptide length, atm I've changed it to sort by proteome counts.
        #solves problems like this
        #on the below, this also settles a problem where there's two max scores for aligning SRFGKFI in the initial setup
        #~~~~~~~~~~~~~~~~~
        #initial alignment
        #   QFGECGKYG
        #  SRFGKFI
        #      IFGKYG
        #~~~
        #realignment after initial bit is done
        #  QFGECGKYG
        #    SRFGKFI
        #     IFGKYG
        #~~~~~~~~~~~~~~~~~
        #on the below, this also settles a problem where there's two max scores for aligning SGGGGGGGSS in the initial setup. The subtractor in both initial and equillibration setups seems to add to this solution. There's multiple equillibrations for each of these 3 below if you exclude the subtractor from the scoring setup.
        #~~~~~~~~~~~~~~~~~
        #  GSSGGGGGGGGW
        #     GGGGGGGGSS
        #    SGGGGGGGSS
        #~~~
        #  GSSGGGGGGGGW
        #    GGGGGGGGSS
        #    SGGGGGGGSS
        #~~~~~~~~~~~~~~~~~
        #The subtractor addition specifically cuts out these competing max scores with IVNEY. This occurs without the subtractor:
        #~~~~~~~~~~~~~~~~~
        #    RDYEIGL
        #    RDYEII
        #  IVNEY
        #~~~
        #    RDYEIGL
        #    RDYEII
        #  IVNEY
        #~~~~~~~~~~~~~~~~~
        #The subtractor also completely solves this super annoying problem. That occurs without it:
        #~~~~~~~~~~~~~~~~~
        #      DLGPND
        #      FLGPLK
        #       NGPLK
        #       NGPLQ
        #  PPLGD
        #  PPLGF
        #~~~
        #       DLGPND
        #       FLGPLK
        #  NGPLK
        #  NGPLQ
        #   PPLGD
        #   PPLGF
        #~~~~~~~~~~~~~~~~~
        #I dislike the way it performs in this example, I feel the top one is a better match. But honestly, that first peptide doesn't really belong anyways. It seems like more of a coincidental matchup that I'm willing to overlook this.
        #~~~~~~~~~~~~~~~~~
        #  TMSANK
        #    VANAG
        #    VANAN
        #~~~
        #  TMSANK
        #  VANAG
        #  VANAN
        #~~~~~~~~~~~~~~~~~
        workinglist = baselist.copy()
        while True:
            for bn in range(len(workinglist)):
                no = nout[bn]
                nlen = len(no)
                leftpoint = mlen - nlen
                rightpoint = mlen + nlen
                currentrange = range(leftpoint, rightpoint)

                scores = []
                for pos in currentrange:
                    workinglist[bn] = emptylist.copy()
                    workinglist[bn][pos:pos+nlen] = no
                    scorearray = []
                    scoreinds = []
                    subtractor = 0 #this contributes to added complexity for breaking ties! argmax always picks the first member of an array where two elements are equal, without this there's a lot of that going on -> leading to poor selections of some sequences and the snowball effect that continues afterwards. This is a subtle way of subtracting small amounts of blank area from alignments that conveniently tack things on to the ends without a good reason while retaining a score equal to the top legitemate spot.
                    adjacency = False
                    for nz, bz in enumerate(zip(*workinglist)):
                        outzip = [i for i in bz if i]
                        outlen = len(outzip)
                        if outlen > 1:
                            subtractor += len(bz) - outlen
                            zippedset = set(outzip)
                            if len(zippedset) == 1:
                                scoreinds.append(nz)
                                scorearray.append(outlen)
                            else:
                                counts = Counter(outzip)
                                countsum = sum(i for i in counts.values() if i > 1) #making this into a total sum b/c whenever something is present more than once, it gets invited to the party
                                if countsum:
                                    scoreinds.append(nz)
                                    scorearray.append(countsum)
                    score = (sum([*map(np.prod,np.split(scorearray,np.where(np.diff(scoreinds)!=1)[0]+1))]) * sum(scorearray)) - subtractor
                    scores.append(score)
                winner = np.argmax(scores)
                workinglist[bn] = emptylist.copy()
                workinglist[bn][currentrange[winner]:currentrange[winner]+nlen] = no
                newrange = [n for n, i in enumerate(workinglist[-1]) if i.isalpha()]

                woopsy = len([i for i in scores if i == max(scores)])
                if woopsy > 1:
                    print('re-woopsy', no)

                leftadjustment = newrange[0] - mlen
                rightadjustment = newrange[0] + mlen

                if leftadjustment < 0:
                    leftadd = np.abs(newrange[0] - mlen)
                    for b in range(len(workinglist)):
                        for l in range(leftadd):
                            workinglist[b].insert(0, '')
                if rightadjustment > endpoint:
                    rightadd = endpoint - rightadjustment
                    endpoint = rightadjustment
                    for b in range(len(workinglist)):
                        for r in range(rightadd):
                            workinglist[b].append('')
                emptylist = ['' for i in range(len(workinglist[0]))]

            #Its entirely possible for the equillibrium to not be stable, and for this to go into an infinite loop. This code currently runs on prayer and lack of responsibility as fuel to prevent this from happening.
            if workinglist == baselist:
                break
            else:
                baselist = workinglist

        positionranges = {}
        for b in baselist:
            s = 0
            for i in b:
                if not i:
                    s += 1
                else:
                    break
            positionranges[''.join((b))] = s
        minval = min(positionranges.values())
        for k in positionranges:
            positionranges[k] -= minval

        spacepad = 10
        for pw, pv in positionranges.items():
            pval = str(noutcount[pw])
            nspace = ' '*(pv+spacepad)
            sspace = ' '*(spacepad-pv-(len(pw)-max(positionranges.values())))
            printstring = ''.join((nspace, pw, ' '*spacepad, sspace, pval))
            print(printstring)
        print('~~~~~~~~~~~~~~~~~')


#Determining importance:
#~~~~~~~~~~~~~~~~~
#              HTINE
#            MTVTINE
#          PQPTVTW
#~~~
#              HTINE               28
#            MTVTINE               17
#          PQPTVTW                 15
#~~~~~~~~~~~~~~~~~
#The third sequence doesn't overlap much with the first, this might be an easy way of saying it should be in a separate group.
#Take the first sequence, then do the moving window search, if the following sequences match the desired window perfectly, then you can disregard them I suppose. Anything that doesn't match well gets its own moving window process.

#faster version with matching output below
#t = time()
#discoveries = {}
#dlist = []
#multiples = 0
#for seq in narrowedsequences:
#    if len(seq) == 1:
#        seq = seq[0]
#        containers = [i for i in proteome if seq in i]
#        if len(containers) > 1:
#            if max([i.count(seq) for i in containers]) == 1:
#                matchinfo = np.asarray([(i.index(seq), len(i)) for i in containers])
#                matchsides = np.hstack((matchinfo[:,0,None], np.diff(matchinfo)))
#                matchmaxes = matchsides.max(axis=0)
#                matchsum = matchmaxes.sum()
#                pers = matchmaxes - matchsides
#                alignments = [['' for i in range(matchsum)] for i in range(len(containers))]
#                for n, ((l, r), c) in enumerate(zip(pers, containers)):
#                    alignments[n][l:l+len(c)] = c
#
#                seqind = matchinfo[0,0] + pers[0,0]
#                #seqcounts = np.asarray([len(set(filter(None, i))) for i in map(list, zip(*alignments))])
#                seqcounts = np.asarray([len(set(i)) for i in zip(*alignments)])
#                matchingboundaries = np.argwhere(np.diff(seqcounts == 1)).flatten()
#                if matchingboundaries.size > 1:
#                    if np.all(seqind < matchingboundaries):
#                        minbound = 0
#                    else:
#                        minbound = matchingboundaries[np.where(matchingboundaries <= seqind)[0].max()] + 1
#                    if np.all(minbound > matchingboundaries):
#                        maxbound = len(seqcounts)
#                    else:
#                        maxbound = matchingboundaries[np.where(matchingboundaries+1 >= seqind+len(seq))[0].min()] + 1
#                else:
#                    if matchingboundaries[0] > seqind:
#                        minbound = 0
#                        maxbound = matchingboundaries[0] + 1
#                    else:
#                        minbound = matchingboundaries[0] + 1
#                        maxbound = len(seqcounts)
#                trueseq = ''.join((alignments[0][minbound:maxbound]))
#                #if joinedproteome.count(trueseq) != len(containers):
#                #    print(seq, 'problem')
#
#                #for troubleshooting
#                #for c in alignments:
#                #    print(''.join((c[minbound:maxbound])))
#
#                discoveries[seq] = trueseq
#                dlist.append([len(seq), len(trueseq)])
#            else:
#                multiples += 1
#print(time() - t)

workingsequencelist = list(itertools.chain(*narrowedsequences)) #I'm going to flatten narrowedsequences here. Finding overlaps on short sequences might not be entirely relevant. I'll save that for after I have finished sequences.
t = time()
discoveries = {}
dlist = []
#multiples = 0
sn = 0
while True:
    #seqi = workingsequencelist[sn]
    seq = workingsequencelist[sn]
    #if len(seqi) == 1:
    #    seq = seqi[0]
    containers = [i for i in proteome if seq in i]
    clen = len(containers)
    if len(containers) > 1:
        concounts = [len(regex.findall(seq, i, overlapped=True, concurrent=True)) for i in containers]
        if max(concounts) != 1: #if the sequence shows up more than once in a protein that it's found in, this will align each index within all of those proteins individually
            conindices = [[m.start() for m in regex.finditer(seq, i, overlapped=True, concurrent=True)] for i in containers]
            conindices = list(itertools.chain(*conindices))
            containers = [[containers[i] for _ in range(concounts[i])] for i in range(len(containers))]
            containers = list(itertools.chain(*containers))
            clen = len(containers)
            matchinfo = np.asarray([[conindices[i], len(containers[i])] for i in range(len(containers))])
        else:
            matchinfo = np.asarray([(i.index(seq), len(i)) for i in containers])
        matchsides = np.hstack((matchinfo[:,0,None], np.diff(matchinfo)))
        matchmaxes = matchsides.max(axis=0)
        matchsum = matchmaxes.sum()
        pers = matchmaxes - matchsides
        alignments = [['' for i in range(matchsum)] for i in range(len(containers))]
        for n, ((l, r), c) in enumerate(zip(pers, containers)):
            alignments[n][l:l+len(c)] = c

        seqind = matchinfo[0,0] + pers[0,0]
        #seqcounts = np.asarray([len(set(filter(None, i))) for i in map(list, zip(*alignments))])
        seqcounts = np.asarray([len(set(i)) for i in zip(*alignments)])
        matchingboundaries = np.argwhere(np.diff(seqcounts == 1)).flatten()
        if matchingboundaries.size > 1:
            if np.all(seqind < matchingboundaries):
                minbound = 0
            else:
                minbound = matchingboundaries[np.where(matchingboundaries <= seqind)[0].max()] + 1
            if np.all(minbound > matchingboundaries):
                maxbound = len(seqcounts)
            else:
                maxbound = matchingboundaries[np.where(matchingboundaries+1 >= seqind+len(seq))[0].min()] + 1
        else:
            if matchingboundaries[0] > seqind:
                minbound = 0
                maxbound = matchingboundaries[0] + 1
            else:
                minbound = matchingboundaries[0] + 1
                maxbound = len(seqcounts)
        trueseq = ''.join((alignments[0][minbound:maxbound]))
        if joinedproteome.count(trueseq) != len(containers):
            if  len(regex.findall(trueseq, joinedproteome, concurrent=True, overlapped=True)) < clen: #some of these are self-overlapping sequences, and some overlap more than fucking once because they're some weird kind of forward-facing pseudo-palindrome!
                print(seq, 'problem')

        #for troubleshooting
        #for c in alignments:
        #    print(''.join((c[minbound:maxbound])))

        discoveries[seq] = trueseq
        dlist.append([len(seq), len(trueseq)])

        #if any member in workingsequencelist is present in both the newly found trueseq, and shows up the same number of times in the proteome, we discard it because it would just be redundant work.
        removers = []
        for n in workingsequencelist:
            #if len(n) == 1:
            if n in trueseq:
                if len(regex.findall(n, joinedproteome, concurrent=True, overlapped=True)) == clen:
                    removers.append(n)
        for r in removers:
            workingsequencelist.remove(r)
        #make a list of strings with spaces inserted to line up the right sequence starting points
        #simple transpose set length to figure out matching indices
        #if one protein sequence stops while everything else still matches as the other proteins go on, you could just consider all subsets of that sequence as a matching piece of that group.
    else:
    #    #if the sequence is only found once, trash that shit, why is it here?
        workingsequencelist.remove(seq)
    #else:
    #    #now here... should I aim to do all the singles first? Should I remove anything in narrowedsequences that is in a group larger than 1 that is a subet of a previously elucidated sequence? But should I add a relationship between these sequences?
    #    workingsequencelist.remove(seqi)
    if not workingsequencelist:
        break
print(time() - t)


#I think if you perform an alignment on every sequence you have, then you can also determine vertical distance and closeness between two aligned [sub]sequences. You can perhaps also determine a distance between two completely arbitrary sequences? Might be a useful thing.
