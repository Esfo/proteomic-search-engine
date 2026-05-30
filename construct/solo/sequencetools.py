import re

#functions that operate on peptide sequences

def seqsplit(sequences, cut, minlength, maxlength, missedcleavages):
    '''
    Cuts peptide sequences using two dicts, one of enzyme cut sites and the other of spots they can't hit.
    Input can be either a list, or a list of lists. Output is a list.
    '''
    cutsequences = sequences[:]
    for site in cut[0]:
        if cut[0][site] == 0:
            splitstring = r''.join(('(?=[', site, '](?!', cut[1][site], '))')) if cut[1][site] else r''.join(('(?=[', site, '])'))
        elif cut[0][site] == 1:
            splitstring = r''.join(('(?<=[', site, '](?!', cut[1][site], '))')) if cut[1][site] else r''.join(('(?<=[', site, '])'))
        cutsequences = list(map(lambda x: re.split(splitstring, x), cutsequences)) if type(cutsequences[0]) is str else [[i for j in list(map(lambda x: re.split(splitstring, x), c)) for i in j] for c in cutsequences]
    slist = []
    for s in range(len(cutsequences)):
        midlist = []
        for y in range(missedcleavages+1):
            midlist.append([''.join((cutsequences[s][i:i+y+1])) for i in range(len(cutsequences[s]))])
        if cutsequences[s][0].startswith('M'): #N-terminal cleavage, makes the cleaved and uncleaved version of the n-terminal peptide
            midlist.append([''.join((cutsequences[s][0][1:], ''.join((cutsequences[s][1:y+1])))) for y in range(missedcleavages+1)])
        midlist = list(set([i for j in midlist for i in j]))
        slist.append(midlist)
    slist = [list(filter(lambda x: maxlength >= len(x) >= minlength, i)) for i in slist]
    for i in range(len(slist)):
        slist[i].sort(key=lambda x: (sequences[i].find(x), len(x)))
    return slist

def seqsplit_dict(tseqs, cut, minlength, maxlength, missedcleavages):
    '''
    Cuts peptide sequences using two dicts, one of enzyme cut sites and the other of spots they can't hit.
    Input is {protein id: sequence}. Output is a dict as {protein id: [list of peptides]}
    '''
    cutsequences = tseqs.copy()
    for site in cut[0]:
        if cut[0][site] == 0:
            splitstring = r''.join(('(?=[', site, '](?!', cut[1][site], '))')) if cut[1][site] else r''.join(('(?=[', site, '])'))
        elif cut[0][site] == 1:
            splitstring = r''.join(('(?<=[', site, '](?!', cut[1][site], '))')) if cut[1][site] else r''.join(('(?<=[', site, '])'))
        cutsequences = {key: re.split(splitstring, cutsequences[key]) for key in cutsequences} if type(list(cutsequences.values())[0]) is str else {key: [k for j in [re.split(splitstring, i) for i in cutsequences[key]] for k in j] for key in cutsequences}
    slist = {}
    for s in cutsequences:
        slist[s] = []
        midlist = {}
        midlist[s] = []
        for y in range(missedcleavages+1):
            midlist[s].extend([''.join((cutsequences[s][i:i+y+1])) for i in range(len(cutsequences[s]))])
        if cutsequences[s][0].startswith('M'): #N-terminal cleavage, makes the cleaved and uncleaved version of the n-terminal peptide
            midlist[s].extend([''.join((cutsequences[s][0][1:], ''.join((cutsequences[s][1:y+1])))) for y in range(missedcleavages+1)])
        slist[s].extend(list(filter(lambda x: maxlength >= len(x) >= minlength, set(midlist[s]))))
        slist[s].sort(key=lambda x: (tseqs[s].find(x), len(x)))
    return slist
