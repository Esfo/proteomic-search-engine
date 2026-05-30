import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import os

folder = '/store/flowcharacterizations/round2/DDAs/MGFs/crux-output/'

files = os.listdir(folder)
files = [i for i in files if i.endswith('.percolator.target.psms.txt')]
#files = [i for i in files if 'stat' in i]

df = []
for f in files:
    fr = ''.join((folder, f))
    td = pd.read_csv(fr, delimiter='\t')
    td = td.loc[td.loc[:,'percolator q-value'] <= 0.01]
    td.drop_duplicates('sequence', keep='first', inplace=True)
    
    nf = td.loc[:,('sequence', 'scan', 'percolator q-value')]
    nf.loc[:,'file'] = f.split('.')[0]
    df.append(nf)
df = pd.concat(df)

df = df.pivot_table(index='sequence', columns='file', values=['scan', 'percolator q-value'])
inall = df.loc[~df.loc[:,'scan'].isnull().any(axis=1)]

pi = inall.loc[:,'scan'].divide(inall.loc[:,'scan'].max(axis=1), axis=0)
pr = pi.rank(method='average', axis=1)


inall.loc[:,'meanscan'] = inall.loc[:,'scan'].mean(axis=1).astype(int)
inall.reset_index(inplace=True)
inall.sort_values(by='meanscan', inplace=True)

fig, ax = plt.subplots(figsize=(16,9))
cols = ['gray', 'firebrick', 'tan', 'darkkhaki', 'chartreuse', 'darkcyan', 'mediumpurple', 'lightcoral', 'olive', 'dodgerblue', 'y', 'blueviolet']

for i, c in enumerate(inall.loc[:,'scan'].columns):
    #inall.plot.scatter(x='sequence', y=('scan', c), c=('percolator q-value', c), colormap='summer', ax=ax, colorbar=False, alpha=1) #q-value isn't very needed here it turns out
    inall.plot.scatter(x='sequence', y=('scan', c), ax=ax, alpha=1, label=c.split('_')[1], color=cols[i]) 
plt.xlabel('Peptide')
plt.ylabel('RT')
plt.legend()
plt.show()


pi.loc[:,'meanscan'] = pi.mean(axis=1).astype(int)
pi.sort_values('meanscan', inplace=True)
pi.reset_index(inplace=True)

fig, ax = plt.subplots(figsize=(16,9))

i = 0
for c in pi.columns:
    if 'DDA' in c:
        pi.plot.scatter(x='sequence', y=c, ax=ax, alpha=1, label=c.split('_')[1], color=cols[i])
        i += 1
plt.xlabel('Peptide')
plt.ylabel('RT')
plt.legend()
plt.show()


pr.loc[:,'meanscan'] = pr.mean(axis=1).astype(int)
pr.sort_values('meanscan', inplace=True)
pr.reset_index(inplace=True)

fig, ax = plt.subplots(figsize=(16,9))

i = 0
for c in pr.columns:
    if 'DDA' in c:
        pr.plot.line(x='sequence', y=c, ax=ax, alpha=0.4, label=c.split('_')[1], color=cols[i])
        i += 1
plt.xlabel('Peptide')
plt.ylabel('RT')
plt.legend(bbox_to_anchor=(1,1))
plt.show()

#color will be q-value
#x-axis will be peptide
#y-axis will be scan #
