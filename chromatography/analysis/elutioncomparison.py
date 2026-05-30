#Round 3 pressure-generated gradients showed much higher peptide ID rates.
#This analysis should answer the question:
#Are these higher ID rates present because the linear gradients didn't have a comparable elution profile (meaning, was it cut off too early, and made too shallow to begin with? Meaning that the pressure-gradient derived method only had more peptides because a higher %ACN was reached within the same time - this would not make for a proper comparison between the two methods)? Ie a retention cut off after a certain peptide should be visible. Or are there a greater number of peptides identified across the entirety of the chromatogram for the pressure-generated gradient?

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

path = '/store/flowcharacterizations/round4/plus/crux-output/'
rtimes = '/store/flowcharacterizations/round4/plus/TICs/'

files = [i for i in os.listdir(path) if i.endswith('.percolator.target.peptides.txt') and 'blank' not in i.lower()]

ef = []
for f in files:
    fn = f.split('.')[0]
    rtn = ''.join((fn, '.tic.csv'))

    d = pd.read_csv(''.join((path, f)), delimiter='\t')
    t = pd.read_csv(''.join((rtimes, rtn)))

    d.loc[:,'scan'] -= 1
    dinds = d.loc[:,'percolator q-value'] <= 0.01
    d = d.loc[dinds]

    d.loc[:,'time (min)'] = t.loc[d.loc[:,'scan'].tolist(), 'time (min)'].to_numpy()

    of = d.loc[:,('sequence', 'time (min)')].copy()
    of.set_index('sequence', inplace=True)
    of.rename({'time (min)':fn}, axis=1, inplace=True)

    ef.append(of)

ef = pd.concat(ef, axis=1)
ef.sort_index(inplace=True)

mval = ef.max().max().round().astype(int)
tvals = np.linspace(0, mval, mval+1).astype(int)

ref = ef.round()

tf = pd.DataFrame(columns=ref.columns)
for v in tvals:
    tf.loc[v] = (ref == v).sum(axis=0)

for c in tf.columns:
    tf.loc[:,c].plot.bar(alpha=0.5, color='green', figsize=(14,10))
    plt.title(c)
    plt.show()

comps = [tf.columns[-2], tf.columns[5]]
#comps = [tf.columns[3], tf.columns[-1]]
cols = ['green', 'purple']

fig, ax = plt.subplots(figsize=(14,10))

for c, i in zip(comps, cols):
    tf.loc[:,c].plot.bar(ax=ax, alpha=0.5, label=c, color=i)
plt.legend()
plt.show()

#now from these results, it seems the increased flowrate in the beginning causes some issues for the pressure-generated method? It would imply that an experiment that removes this part would do better, BUT, how would that affect the chromatography?
#AND, this technically needs to be divied up by volume B, and not retention time

#latest eluting peptide found in ~all files, change the 0 to a 1 to make it all except 1, etc.
latestpeptide = np.argwhere((ef.sort_values(ef.columns[0]).isnull().sum(axis=1) == 0).to_numpy())[-1]
print(ef.sort_values(ef.columns[0]).iloc[latestpeptide].transpose())
#The fact that this shows a peptide eluting at ~187 mins in one of the 196 min gradients while also showing up towards the end of the 180 min gradients indicates that the elongated gradient did not have higher number of peptides due to a different population of peptides emerging at the end of this.

#Outstanding questions: How does the eluting solvent change for many of these peptides? I'd like to plot the volume B and the %B they eluted in. This might elucidate whether ionization is playing a major role in the increased numbers. If it is, you may want to try the experiment on multiple different columns with different sized needles.
#Comparison with a commercial column might also be necessary just in case what we're making is causing these effects.
