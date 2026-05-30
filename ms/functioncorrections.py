def expansion_organizer(massadditions, elementcount, samplesize, isotopomersbyaddition):
    abundanceprobs = defaultdict(dict)
    fullprob, standingmass = limited_multinomial(elementcount)
    if fullprob * samplesize > 1:
        subformulastring = ''.join((''.join((f'{e}{isotopomersbyaddition[e][m]}({c})' for m, c in v.items() if c > 0)) for e, v in elementcount.items())) #this outputs a formula that describes the individual isotopes involved
        for e, v in elementcount.items():
            if v[0] > 0:
                for m, p in v.items():
                    if m > 0:
                        if standingmass + m not in abundanceprobs: #relying on all combinatorics of these masses being unique, i'm pretty sure they end up that way because of how all massadditions are unique, any calculatable overlap would probably be way further away than any of the length limitations would realistically allow
                            elementcount[e][0] -= 1
                            elementcount[e][m] += 1
                            abundanceprobs.update(expansion_organizer(massadditions, elementcount, samplesize, isotopomersbyaddition))
                            elementcount[e][0] += 1
                            elementcount[e][m] -= 1
            #else:
                #you don't need to worry about these because the combination of all non-0 mass addition spots get hit combinatorically like this, for as many as there are - given it passes samplesize
        abundanceprobs[standingmass][subformulastring] = fullprob
    return abundanceprobs

def distribution_generation(seq, samplesize):
    atomiccomposition = Counter()
    for aa in seq:
        atomiccomposition += aminoacidcomposition[aa]
    #no OH loss on last residue, no H lost on first residue
    atomiccomposition['H'] += 2
    atomiccomposition['O'] += 1
    #^make sure to +H2O to whatever sequence composition there is in your function, its just some water chemistry that gets added to either end, I don't add it to the individual pieces above because they end up linked together like a string of legos and they don't have room for the water molecule in between them.
    monoisotopicmass = sum(majorisotopemasses[k]*v for k, v in atomiccomposition.items())
    
    elementcount = {} #element: (isotopic mass - monoisotopic mass): count
    for e, c in atomiccomposition.items():
        elementcount[e] = {0: c}
    for e, v in massadditions.items():
        if e in atomiccomposition:
            for m, c in v.items():
                if m > 0:
                    elementcount[e][m] = 0
    
    #this below only lists elements present in the peptide, sometimes there's no Sulfur
    #elementpriority = [] #[element, mass addition, binomial probability]
    #for element, count in atomiccomposition.items():
    #    for m, p in massadditions[element].items():
    #        if m > 0:
    #            elementpriority.append([element, m, p*count])
    #elementpriority = sorted(elementpriority, key=lambda x: -x[2])
    #^this sorts isotopes in order of the most likely binomial to be the next highest abundance isotopomer
    #^this isn't necessary for this recursive approach because I don't re-calculate these on the fly, when in reality it would be necessary to do that if you want it to be useful
    
    abundanceprobs = {}
    #this while loop is here because of my own stupidity. This is necessary because in this function I start calculating isotopes at the monoisotopic mass. For larger molecules, the monoisotopic mass can actually be really small (and below my abundance threshold) so the functions above fail to trigger any isotopomer calculations at all
    while not abundanceprobs:
        #abundanceprobs.update(expansion_organizer(elementcount, elementpriority, samplesize, defaultdict(dict)))
        abundanceprobs.update(expansion_organizer(massadditions, elementcount, samplesize, isotopomersbyaddition))
        samplesize *= 2
    
    #below adds the monoisotopic mass to the mass additions delivered from expansion_organizer to get the final masses
    massesandabundances = [[], []]
    formulas = []
    for m, fp in abundanceprobs.items():
        for f, p in fp.items(): #length of fp will always be 1 because of currentmass+m blocking in expansion_organizer
            massesandabundances[0].append(monoisotopicmass + m)
            massesandabundances[1].append(p)
            formulas.append(f)
    
    #sorting everything by mass
    massesandabundances = np.array(massesandabundances)
    formulas = np.array(formulas, dtype='S')
    formulas = formulas[massesandabundances[0].argsort()].tolist()
    massesandabundances = massesandabundances[:,massesandabundances[0].argsort()]
    return massesandabundances, formulas


#I'll use this for profiling, to determine the speedup payment rate, #i'll use the averages of each length to take an AUC
import random

bt = time()
n = 0
times, lengths = [], []
while n < 99999: #this number took ~150 seconds for me
    seq = ''.join((random.choices(list(aminoacidcomposition.keys()), k=np.random.randint(6,50))))
    nt = time()
    abundances, subformulas = distribution_generation(seq, samplesize)
    end = time() - nt
    times.append(end)
    lengths.append(len(subformulas))
    n += 1
print(time() - bt)

plt.plot(lengths, times)
plt.show()
