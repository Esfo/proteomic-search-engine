import numpy as np
import matplotlib.pyplot as plt
from pyteomics import mgf, mzml
from time import time
import pandas as pd
import gc
from scipy import sparse, signal, stats, interpolate
from pandas.api.types import CategoricalDtype
from statsmodels.nonparametric.smoothers_lowess import lowess
import itertools
import sys
import os
plt.rcParams["figure.dpi"] = 100

folder = '/store/flowcharacterizations/round1/MS1s/mzMLs/'
#mzmlfile = '/store/FE/mzMLs/test/190701_F350_yM2.mzML'
runorder = open('/store/flowcharacterizations/round1/runorder')
runorder = [i.strip() for i in runorder]
runorder = [''.join((folder, '200612_', i, '.mzML')) for i in runorder]
files = list(itertools.zip_longest(*[iter(runorder)]*2))


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def scanfunc(scan, blankfilter):
    t = pd.DataFrame()
    timedict = {}
    inds = scan['intensity array'] > blankfilter[scan['index']]
    if any(inds):
        t.loc[:,'m/z'] = scan['m/z array'][inds].round(4)
        t.loc[:,'intensity'] = scan['intensity array'][inds]
        t.loc[:,'index'] = scan['index']
    
    #t.loc[:,'m/z'] = scan['m/z array'].round(4)
    #t.loc[:,'intensity'] = scan['intensity array']
    #t.loc[:,'index'] = scan['index']
    timedict[scan['index']] = scan['scanList']['scan'][0]['scan start time']
    return t, timedict

def outfunc(scan):
    return scan['index'], scan['intensity array'].mean()

def readfunc(b, f):
    #putting the blank file to filter things below the noise line because I'm an idiot and collected 8gb data files and 7gb blanks.
    msblank = mzml.MzML(b)
    msrun = mzml.MzML(f)
    bms = []
    for i in msblank.map(lambda x: outfunc(x)):
        bms.append(i)

    bms = np.asarray(bms)
    bms = bms[bms[:,0].argsort()]
    outbms = signal.resample(bms[:,1], len(msrun))
    sms = []
    for i in msrun.map(lambda x: outfunc(x)):
        sms.append(i)
    
    sms = np.asarray(sms)
    sms = sms[sms[:,0].argsort()]
    of = pd.DataFrame()
    of.loc[:, 'noise'] = outbms
    of.loc[:, 'signal'] = sms[:,1]
    
    outname = f.split('.')[0].split('_')[-1]
    print(outname, 'done')
    return {outname: of}

rf = '/store/flowcharacterizations/round1/DDAs/MGFs/results.csv'
df = pd.read_csv(rf)
df.loc[:,'file'] = df.loc[:,'file'].apply(lambda x: x.split('.')[0].split('_')[-1])
df.set_index('file', inplace=True)

mt = time()
frames = {}
for b, f in files:
    frames.update(readfunc(b,f))

gc.collect()
print(time() - mt, '- File Extracted')
gradientpath = '/store/flowcharacterizations/round1/gradients/'

for i, f in frames.items():
    df.loc[i,'signal'] = (f.signal > f.noise * 2).sum()
    df.loc[i,'scans'] = len(f)

    if df.loc[i].signal > 4000:
        fig, ax = plt.subplots(2, 1, figsize=(8,8))
        fig.subplots_adjust(hspace=0.1)
        f.plot(ax=ax[1])
        
        gn = ''.join(([i for i in i if not i.isdigit()]))
        grad = pd.read_csv(''.join((gradientpath, gn, '.method.csv')))
        grad.fillna(0, inplace=True)
        axn = ax[0].twinx()
        grad.plot.line(x='Time', y='Flow', ax=ax[0], color='orange')
        grad.plot.line(x='Time', y='%ACN', ax=axn)
        #ax[0].legend(bbox_to_anchor=(1,1))
        #axn.legend(bbox_to_anchor=(1,1.1))
        plt.title(i)
        plt.show()
        print('Gradient max time:', grad.loc[:,'Time'].max())




#Long-term goal:
#area, mean intensity, mass count, mass count per mean intensity, 
#i want distributions of masses and intensities at each RT
#I want to see if I can compare 2 masses across files with different RT but same ~area based on my %b gradient and volume flowed
