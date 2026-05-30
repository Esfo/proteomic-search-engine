import numpy as np
import matplotlib.pyplot as plt
from pyteomics import mzml
import pymzml
import multiprocessing as mp
from time import time
import pandas as pd
import gc
from scipy import sparse, signal, stats, interpolate, integrate
from pandas.api.types import CategoricalDtype
from statsmodels.nonparametric.smoothers_lowess import lowess
import itertools
import sys
import os

def scanfunc(scan, t, tc):
    et = pd.DataFrame(scan.peaks('centroided'))
    et.columns = ['m/z', 'intensity']
    et.loc[:,'m/z'] = et.loc[:,'m/z'].round(4)
    et.loc[:,'index'] = scan.ID - 1
    et.loc[:,'ms level'] = scan.ms_level
    et.loc[:,'time (min)'] = scan.scan_time_in_minutes()
    binds = np.logical_and(et.loc[:,'m/z'].to_numpy() >= tc.loc[:,'lower limit'].to_numpy().reshape(-1,1), et.loc[:,'m/z'].to_numpy() <= tc.loc[:,'upper limit'].to_numpy().reshape(-1,1))
    et = et.loc[np.argwhere(binds)[:,1]]
    et.loc[:,'channel'] = tc.index.to_numpy()[np.argwhere(binds)[:,0]]
    t.append(et)
    return t

def readfunc(f, tc):
    t = mp.Manager().list()
    msrun = pymzml.run.Reader(f)
    pool = mp.Pool()
    for scan in msrun:
        pool.apply_async(scanfunc(scan, t, tc))
    pool.close()
    pool.join()
    return list(t)

def vsolver(a, b, c, m):
    return (b - np.sqrt((b**2) + (m**2)*(c**2) + 2*m*b*c + 2*m*a)) / (-1*m)

def volumefinder(xvals, yvals, deadvolume):
    #xvals are time, yvals are flow
    xvals, yvals = np.asarray(xvals), np.asarray(yvals)

    #cheap hack to ignore unchanging flowrates that give zero-slopes, flowrates aren't that accurate anyways
    for yn in range(len(yvals)-1):
        if yvals[yn] == yvals[yn+1]:
            yvals[yn+1] += 0.00000001

    areas = integrate.cumtrapz(yvals, xvals, initial=0)
    sfind = (areas - deadvolume).reshape(-1,1)
    sfind[sfind < 0] = 0
    slopes = np.diff(yvals) / np.diff(xvals)
    #if np.all(slopes == 0):
    #    deadtime = deadvolume / yvals[0]
    #    xpoints = xvals - deadtime
    #    xpoints[xpoints < 0] = 0
    #    return xpoints
    #else:
    adiffs = np.diff(areas)
    sfbool = areas < sfind
    sfbi = sfbool.sum(axis=1) - 1
    sfbi[sfbi < 0] = 0
    slopeofinterest = slopes[sfbi]
    priorarea = areas[sfbi]
    specificarea = sfind.flatten() - priorarea
    xvofinterest = xvals[sfbi]
    yvofinterest = yvals[sfbi]
    intercept = yvofinterest - (xvofinterest * slopeofinterest)
    return vsolver(specificarea, intercept, xvofinterest, slopeofinterest)

def timeconversion(frameobj, gr, fillval='Time', reorder=True):
    if reorder:
        try:
            tf = frameobj.loc[:,fillval].to_frame()
        except AttributeError:
            tf = frameobj.loc[:,fillval]
        tf.columns = ['Time (m)']
        gr = gr.append(tf)
    gr = gr.sort_values('Time (m)').interpolate(limit_direction='both')
    gr.loc[:,'Total Volume'] = integrate.cumtrapz(gr.loc[:,'Flow'], gr.loc[:,'Time (m)'], initial=0)
    gr.loc[:,'Column Volumes'] = gr.loc[:,'Total Volume'] / columnvolume
    gr.loc[:,'Flow B at Interval'] = gr.loc[:,'Flow'] * gr.loc[:,'%ACN'] / 100
    gr.loc[:,'Flow A at Interval'] = gr.loc[:,'Flow'] * (100 - gr.loc[:,'%ACN']) / 100
    gr.loc[:,'Cumulative Volume B'] = integrate.cumtrapz(gr.loc[:,'Flow B at Interval'], gr.loc[:,'Time (m)'], initial=0) #nL
    gr.loc[:,'Cumulative Volume A'] = integrate.cumtrapz(gr.loc[:,'Flow A at Interval'], gr.loc[:,'Time (m)'], initial=0) #nL
    
    #ic means in-column
    gr.loc[:,'ic-timepoint'] = volumefinder(gr.loc[:,'Time (m)'], gr.loc[:,'Flow'], deadvolume)
    #some values come out just below the minimum timepoint because of computational precision differences, but for the ones that do it's going to be the same conditions as the initial timepoint, so this is fixed here
    gr.loc[(gr.loc[:,'ic-timepoint'] < gr.loc[:,'Time (m)'].min()), 'ic-timepoint'] = gr.loc[:,'Time (m)'].min()
    #the volumesolver already takes the flow and volume into consideration, so simply solving for whatever was pumping at the solved timepoint is a legitemate measure for the %ACN in the column at a given time.
    percinterp = interpolate.interp1d(gr.loc[:,'Time (m)'].to_numpy(), gr.loc[:,'%ACN'].to_numpy())
    flowinterp = interpolate.interp1d(gr.loc[:,'Time (m)'], gr.loc[:,'Flow'])
    
    gr.loc[:,'ic-%ACN'] = percinterp(gr.loc[:,'ic-timepoint'])
    #ic-flow not used for actual time flow measurements, it should only be used to determine volumes
    gr.loc[:,'ic-flow'] = flowinterp(gr.loc[:,'ic-timepoint'])
    
    gr.loc[:,'ic-tvolume'] = integrate.cumtrapz(gr.loc[:,'ic-flow'], gr.loc[:,'ic-timepoint'], initial=0)
    gr.loc[:,'ic-bflow'] = gr.loc[:,'ic-flow'] * gr.loc[:,'ic-%ACN'] / 100
    gr.loc[:,'ic-aflow'] = gr.loc[:,'ic-flow'] * (100 - gr.loc[:,'ic-%ACN']) / 100
    
    gr.loc[:,'ic-bvol'] = integrate.cumtrapz(gr.loc[:,'ic-bflow'], gr.loc[:,'ic-timepoint'], initial=0) #nL
    gr.loc[:,'ic-avol'] = integrate.cumtrapz(gr.loc[:,'ic-aflow'], gr.loc[:,'ic-timepoint'], initial=0) #nL
        
    gr.drop_duplicates('Time (m)', inplace=True)
    gr.set_index('Time (m)', inplace=True)
    
    egf = gr.loc[frameobj.loc[:,fillval]]
    egf.drop_duplicates(inplace=True)
    return egf

deadvolume = 3225 #nL
columnvolume = 530 #nL

folder = '/store/flowcharacterizations/round3/DDAs/'

ddafolder = ''.join((folder, 'crux-output/'))
gradientsfolder = ''.join((folder, 'gradients/'))
dticfolder = ''.join((folder, 'TICs/'))
peakfolder = ''.join((folder, 'peaks/'))
mzfolder = ''.join((folder, 'mzMLs/'))

files = os.listdir(peakfolder)

fr = files[0]
#for fr in files:
ftitle = '_'.join((fr.split('.')[0].split('_')[1:]))
f = ''.join((peakfolder, fr))
fbase = fr.split('.')[0]
enfs = pd.read_csv(f, low_memory=False)
enfs.set_index('retention time (scan)', inplace=True)
mzmlfile = ''.join((mzfolder, fbase, '.mzML'))

pname = ''.join((ddafolder, fbase, '.percolator.target.peptides.txt'))
pfile = pd.read_csv(pname, delimiter='\t')
pfile = pfile.loc[pfile.loc[:,'percolator q-value'] < 0.01]

dname = ''.join((dticfolder, fbase, '.tic.csv'))
dtic = pd.read_csv(dname)

dtic.loc[pfile.sort_values('scan').loc[:,'scan'], 'Identified Peptide'] = 1
dtic.loc[:,'Identified Peptide'].fillna(0, inplace=True)

gname = ''.join((gradientsfolder, ''.join((fbase.split('_')[1:-1])), '.method.csv'))
try:
    gradient = pd.read_csv(gname)
except FileNotFoundError:
    gname = ''.join((gradientsfolder, '_'.join((fbase.split('_')[1:])), '.method.csv'))
    gradient = pd.read_csv(gname)

egf = timeconversion(dtic, gradient)
egf.fillna(0, inplace=True)

dtic.set_index('Time', inplace=True)
egf.loc[:,'Identified Peptide'] = dtic.loc[:,'Identified Peptide']
egf.loc[:,'Cumulative Peptides'] = egf.loc[:,'Identified Peptide'].sort_index().cumsum().tolist()
egf.loc[:,'Peptides/Column Volume'] = egf.loc[:,'Cumulative Peptides'] / egf.loc[:,'Column Volumes']

dtic.reset_index(inplace=True)
dtic.rename({'Time': 'Time (m)'}, axis=1, inplace=True)

enfs.loc[:,'Time (m)'] = dtic.loc[enfs.reset_index().loc[:,'retention time (scan)'], 'Time (m)'].tolist()
enfs.loc[:,'Column Volumes'] = egf.loc[enfs.loc[:,'Time (m)'],'Column Volumes'].tolist()
enfs.loc[:,'Total Volume'] = egf.loc[enfs.loc[:,'Time (m)'],'Total Volume'].tolist()
enfs.loc[:,'Cumulative Volume B'] = egf.loc[enfs.loc[:,'Time (m)'],'Cumulative Volume B'].tolist()
enfs.loc[:,'Column Volume B'] = enfs.loc[:,'Cumulative Volume B'] / columnvolume
dtic.loc[:,'Column Volumes'] = egf.loc[dtic.loc[:,'Time (m)'],'Column Volumes'].tolist()
dtic.loc[:,'Total Volume'] = egf.loc[dtic.loc[:,'Time (m)'],'Total Volume'].tolist()
dtic.loc[:,'Cumulative Volume B'] = egf.loc[dtic.loc[:,'Time (m)'],'Cumulative Volume B'].tolist()
dtic.loc[:,'Column Volume B'] = dtic.loc[:,'Cumulative Volume B'] / columnvolume

gradient = timeconversion(gradient, gradient, fillval='Time (m)', reorder=False)
gradient.fillna(0, inplace=True)
gradient.loc[:,'Column Volume B'] = gradient.loc[:,'Cumulative Volume B'] / columnvolume

top = 10
channelframe = enfs.groupby('mass channel').count().sort_values('r height', ascending=False).loc[:,'r height']
channels = channelframe.index.to_numpy()
chans = channels[:top]
targetchannels = enfs.set_index('mass channel').loc[chans,'+/- ppm window'].mean(level=0)
targetchannels = targetchannels.to_frame()
#targetchannels.index = targetchannels.index.to_numpy().round(5)

targetchannels.loc[:,'lower limit'] = targetchannels.index - (targetchannels.index / 1000000) * targetchannels.loc[:,'+/- ppm window']
targetchannels.loc[:,'upper limit'] = targetchannels.index + (targetchannels.index / 1000000) * targetchannels.loc[:,'+/- ppm window']

mt = time()
ef = readfunc(mzmlfile, targetchannels)

gc.collect()
print(time() - mt, '- File Extracted')

ef = pd.concat(ef)

ep = ef.pivot_table(index='time (min)', columns='channel', values='intensity', aggfunc='sum', fill_value=0)

indexmap = ef.set_index('time (min)').loc[:,'index']
indexmap.drop_duplicates(inplace=True)

ep.columns = ep.columns.to_numpy()
ep.loc[:,'scan'] = indexmap.loc[ep.index]

ecopy = ep.set_index('scan')

#channels = channels.round(4)
for c in channels:
    fs = enfs.loc[enfs.loc[:,'mass channel'] == c]
    savgol = signal.savgol_filter(ecopy.loc[:,c].to_numpy().flatten(), window_length=101, polyorder=2, mode='wrap', deriv=0)
    fig, ax = plt.subplots(figsize=(16,8))
    ecopy.loc[:,c].reset_index().plot.scatter(x='scan', y=c, ax=ax, color='orange')
    ax.plot(ecopy.index, savgol, color='green')
    ax.vlines(fs.index, 0, ecopy.loc[:,c].max(), color='black')
    #plt.xlim(20000, 30000)
    plt.show()

#channels don't look the same, if you sum from axis 1 in nomain, become combination, does it appear similar to what's in this file?

#need to make a plot using nomain for y=m/z, x=time, color=intensity to figure out wtf is going on

rs = []
for i in fs.index:
    rs.append(set(range(fs.loc[i, 'left base point'], fs.loc[i, 'right base point'])))
