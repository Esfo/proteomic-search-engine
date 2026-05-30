import pickle
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt

plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.facecolor'] = 'gray'
plt.rcParams['figure.facecolor'] = 'gray'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['ytick.labelcolor'] = 'white'
plt.rcParams['xtick.labelcolor'] = 'white'

isotopefile = '/home/sfo/data/proteomics/fastas/isotope-arrays/human_isotopes-6-100_miss-1_ss200-by_mono.pickle'

with open(isotopefile, "rb") as pick:
    isotopeabundances, seqsbymass = pickle.load(pick)
#make something to prove later on is that there are no incomplete lists in isotopeabundances

seqmasses = np.sort(list(seqsbymass.keys()))

requiredresolutions = seqmasses[:-1] / np.diff(seqmasses) #seqmasses is pre-sorted

plt.plot(requiredresolutions, '.', color='white', alpha=0.5, markersize=0.5)
plt.yscale('log')
plt.show()

diffs = {}
for k, v in isotopeabundances.items():
    keys = list(v.keys())
    diffs[k] = max(keys) - min(keys)

plt.plot(list(diffs.keys()), list(diffs.values()), '.', color='white', markersize=0.5, alpha=0.5)
plt.show()

#use the distance from 0, everything - minimum, as a displacement system for where a distributions top n isotopomers are recorded.
#look at:
#   - total distance
#   - distance of top to closest
#   - distance of lowest to next lowest
#I'm thinking n=3 is a pretty good start.

dlist = []
plist = []
for k, v in isotopeabundances.items():
    va = np.array(Counter(v).most_common(3))
    masses = va[:,0]
    percs = va[:,1]
    dlist.append((masses - masses.min()).tolist())
    plist.append(percs.tolist())

dlist, plist = np.array(dlist), np.array(plist)

plt.plot(seqmasses, plist[:,0], '.', color='white', markersize=0.5)
plt.show()
plt.plot(seqmasses, plist[:,1], '.', color='white', markersize=0.5)
plt.show()
plt.plot(seqmasses, plist[:,2], '.', color='white', markersize=0.5)
plt.show()

plt.plot(seqmasses, plist[:,0] / plist[:,1], '.', color='white', markersize=0.5)

#do a KNN of baseline mass + top 3 isotopomer distances to find the nearest similar distribution?

plt.bar(v.keys(), v.values(), width=0.01, color='white')

diffs = []
for k, v in isotopeabundances.items():
    va = np.sort(list(v.keys()))
    vd = np.diff(va)
    diffs.extend(vd[vd < 0.5].tolist())

dcounts = Counter(diffs)

#whether between peak can be lower - yes, a minor peak can also be larger than the end major peak
#analysis of mass differences of minor peaks to majors
