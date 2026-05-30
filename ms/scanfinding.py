import numpy as np
import matplotlib.pyplot as plt
from pyteomics import mzml
import pymzml
import multiprocessing as mp
from time import time
import pandas as pd
import gc
from scipy import sparse, signal, stats, interpolate
from pandas.api.types import CategoricalDtype
from statsmodels.nonparametric.smoothers_lowess import lowess
import itertools
import sys
import os
#plt.rcParams["figure.dpi"] = 100

folder = '/store/brody/mzMLs/'
files = os.listdir(folder)
files = [i for i in files if i.endswith('.mzML')]

outfolder = '/'.join(('/'.join((folder.split('/')[:-2])), 'data'))
phn = '/'.join((outfolder, 'phe.csv'))
trn = '/'.join((outfolder, 'trp.csv'))

extracting = False

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def scanfunc(scan, t):
    try:
        et = pd.DataFrame(scan.peaks('centroided'))
        et.columns = ['m/z', 'intensity']
        et.loc[:,'m/z'] = et.loc[:,'m/z'].round(4)
        et.loc[:,'index'] = scan.ID - 1
        et.loc[:,'ms level'] = scan.ms_level
        et.loc[:,'time (min)'] = scan.scan_time_in_minutes()
        t.append(et)
        return t
    except ValueError:
        pass

#def blankfunc(scan):
#    return scan['index'], scan['intensity array'].mean()
#
#def readfunc(b, f):
#    #putting the blank file to filter things below the noise line because I'm an idiot and collected 8gb data files and 7gb blanks.
#    msblank = mzml.MzML(b)
#    msrun = mzml.MzML(f)
#    eps = []
#    for i in msblank.map(lambda x: blankfunc(x)):
#        eps.append(i)
#    eps = np.asarray(eps)
#    eps = eps[eps[:,0].argsort()]
#    lws = lowess(eps[:,1].flatten()*1.25, eps[:,0].flatten(), frac=1, it=0)
#    blankfilter = lws[:,1].flatten()
#    blankfilter = signal.resample(blankfilter, len(msrun))
#    
#    ef = []
#    times = {}
#    for sc, td in msrun.map(lambda x: scanfunc(x, blankfilter)):
#        ef.append(sc)
#        times.update(td)
#    return ef, times

def readfunc(f):
    t = mp.Manager().list()
    msrun = pymzml.run.Reader(f)
    pool = mp.Pool()
    for scan in msrun:
        pool.apply_async(scanfunc(scan, t))
    pool.close()
    pool.join()
    return list(t)

if extracting:
    phes = []
    trps = []
    for fn in files:
        if 'Blank' not in fn:
            mzmlfile = ''.join((folder, fn))
#def main(mzmlfile):
#Opening the file and extracting data
            mt = time()
            ef = readfunc(mzmlfile)
            
            gc.collect()
            print(time() - mt, '- File Extracted')
            ef = pd.concat(ef)
            
            minds = ef.loc[:,'m/z'] < 200
            
            mc1 = 176.1135 #F
            mc2 = 218.1281 #W
            mc = np.array([mc1, mc2])
            mcn = np.array(['F', 'W'])
            mcinds = np.abs(ef.loc[:,'m/z'].to_numpy() - mc.reshape(-1,1)).argmin(axis=0)
            ef.loc[:,'channel'] = mcn[mcinds]
            ef.set_index('channel', inplace=True)
            ef.sort_index(inplace=True)

            f = ef.loc[mcn[0]].set_index('index')
            f.sort_index(inplace=True)
            
            w = ef.loc[mcn[1]].set_index('index')
            w.sort_index(inplace=True)

            nf = pd.DataFrame()
            nf.loc[:,'mass wmean'] = (f.loc[:,'m/z'] * f.loc[:,'intensity']).sum(level=0) / f.loc[:,'intensity'].sum(level=0)
            nf.loc[:, 'mass error (ppm)'] = ((nf.loc[:,'mass wmean'] - mc1) / mc1) * 1000000
            nf.loc[:,'intensity'] = f.loc[:,'intensity'].sum(level=0)
            nf.loc[:,'time (min)'] = f.loc[:,'time (min)'].mean(level=0)
            
            nw = pd.DataFrame()
            nw.loc[:,'mass wmean'] = (w.loc[:,'m/z'] * w.loc[:,'intensity']).sum(level=0) / w.loc[:,'intensity'].sum(level=0)
            nw.loc[:, 'mass error (ppm)'] = ((nw.loc[:,'mass wmean'] - mc2) / mc2) * 1000000
            nw.loc[:,'intensity'] = w.loc[:,'intensity'].sum(level=0)
            nw.loc[:,'time (min)'] = w.loc[:,'time (min)'].mean(level=0)

            nf.reset_index(inplace=True)
            nw.reset_index(inplace=True)
            nf.loc[:,'file'] = fn.split('_')[1].split('.')[0]
            nw.loc[:,'file'] = fn.split('_')[1].split('.')[0]
            phes.append(nf)
            trps.append(nw)
    
    phes = pd.concat(phes)
    trps = pd.concat(trps)
    phes.set_index(['file', 'index'], inplace=True)
    trps.set_index(['file', 'index'], inplace=True)
    phes.sort_index(inplace=True)
    trps.sort_index(inplace=True)
    phes.to_csv(phn)
    trps.to_csv(trn)
else:
    phes = pd.read_csv(phn)
    trps = pd.read_csv(trn)
    phes.set_index(['file', 'index'], inplace=True)
    trps.set_index(['file', 'index'], inplace=True)
    phes.sort_index(inplace=True)
    trps.sort_index(inplace=True)

for f in phes.index.levels[0]: 
    nf = phes.loc[f]
    fig, ax = plt.subplots(figsize=(16,8))

    nf.sort_values('time (min)').plot.line(x='time (min)', y='intensity', alpha=0.4, ax=ax, color='black')
    nf.sort_values('time (min)').plot.scatter(x='time (min)', y='intensity', c='mass error (ppm)', alpha=1, colormap='cool', ax=ax, s=80, colorbar=False)
    ax.set_xlim(20,22)
    
    ax.set_title(mcn[0], fontsize=15)
    ax.set_ylabel('Intensity', fontsize=15)
    ax.set_xlabel('Time (min)', fontsize=15)
    plt.legend([])
    plt.title(' '.join(('Phenylalanine in', f)), fontsize=15)
    im = plt.gca().get_children()[0]
    cb = fig.colorbar(im)
    cb.set_label(label='Mass Error (ppm)', fontsize=15)
    plt.tight_layout()
    plt.show()

for f in trps.index.levels[0]:
    nw = trps.loc[f]
    fig, ax = plt.subplots(figsize=(16,8))
    
    nw.sort_values('time (min)').plot.line(x='time (min)', y='intensity', alpha=0.4, ax=ax, color='black')
    nw.sort_values('time (min)').plot.scatter(x='time (min)', y='intensity', c='mass error (ppm)', alpha=1, colormap='cool', ax=ax, s=80, colorbar=False)
    ax.set_xlim(20,22)

    ax.set_title(mcn[1], fontsize=15)
    ax.set_ylabel('Intensity', fontsize=15)
    ax.set_xlabel('Time (min)', fontsize=15)
    plt.legend([])
    plt.title(' '.join(('Tryptophan in', f)), fontsize=15)
    im = plt.gca().get_children()[0]
    cb = fig.colorbar(im)
    cb.set_label(label='Mass Error (ppm)', fontsize=15)
    plt.tight_layout()
    plt.show()
        
#        tn = 2
#        if 'Test' not in fn:
#            ef.loc[mcn[0]].sort_values('time (min)').plot.line(x='time (min)', y='intensity', ax=ax[0], alpha=0.4, label=label)
#            ef.loc[mcn[1]].sort_values('time (min)').plot.line(x='time (min)', y='intensity', ax=ax[1], alpha=0.4, label=label)
#        else:
#            ef.loc[mcn[0]].sort_values('time (min)').plot.line(x='time (min)', y='intensity', ax=ax[tn], alpha=0.5, label=mcn[0])
#            ef.loc[mcn[1]].sort_values('time (min)').plot.line(x='time (min)', y='intensity', ax=ax[tn], alpha=0.5, label=mcn[1])
#            tn += 1
#for a in ax:
#    a.set_yscale('log')
#plt.legend()
#plt.show()


#scantimes = ef.loc[:,('index', 'time (min)')].drop_duplicates()
#scantimes.set_index('index', inplace=True)
#ef = ef.loc[ef.loc[:,'ms level'] == 1]
##ef.loc[:,'m/z'] = ef.loc[:,'m/z'].round(4)
##ef.loc[:,'intensity'] = ef.loc[:,'intensity'].astype(int)
##ef.loc[:,'index'] = ef.loc[:,'index'].astype(np.int16)
#
##Excluding any mass channels that hold less than the mean intensity across time as a data point. These are left out for having too little data. This makes the data manageable for analysis development. It should be removed/replaced later by removing mass channels that don't  meet the minimum number of indices with contiguous data points.
##et = ef.set_index(['m/z', 'index']).sum(axis=0, level=0) - EDIT: This has been done below
##print(et.mean().to_numpy()[0], 'scans as the temporary minimum scan threshhold for inclusion')
##et = et.loc[(et > et.mean()).to_numpy()]
#et = ef.drop('intensity', axis=1)
#et = et.groupby('m/z').count()
#et = et.loc[(et > et.loc[(et > 1).to_numpy()].mean()).to_numpy()]
##print('Imposing a minimum of', et.min().to_numpy()[0], 'data points per unmerged mass channel') #this isn't actually imposing this on all of the data, it only imposes it for a shortened data-set used for finding the centers of mass.
##^Mean I was getting was 6 for simple index counts, when removing all the 1's, which was the vast vast majority, the mean shifts to ~9.6. There's ~500000 data points above 10 so this isn't bad, and 10 is a good minimum index points
##what could be good for presentation purposes is showing the histogram of et after it gets transformed into counts to show just how maany masses are only read once
##Given that the purpose here is to find centers of mass, it's acceptable to remove datapoints that don't even hold this many indices across the chromatogram
#
#en = ef.set_index('m/z')
#en.sort_index(inplace=True)
#en = en.loc[et.index.array]
##mzs = et.reset_index().to_numpy()[:,0].flatten()
##ints = et.reset_index().to_numpy()[:,1].flatten()
##lws = lowess(ints, mzs, frac=0.05, it=0)
#
#nbins = np.sqrt(en.index.unique().shape[0]).round().astype(int)
#ediff = pd.DataFrame()
#ediff.loc[:,'max'] = en.loc[:,'intensity'].max(level=0)
#ediff.loc[:,'sum'] = en.loc[:,'intensity'].sum(level=0)
#ediff.loc[:,'diff'] = ediff.loc[:,'sum'] - ediff.loc[:,'max']
#ediff.loc[:,'dbins'] = pd.cut(ediff.loc[:,'diff'], bins=nbins, labels=False)
##ediff.plot.hist(bins=nbins)
#
#ediff.loc[:,'pdiff'] = ediff.loc[:,'diff'] / ediff.loc[:,'sum']
#ediff.loc[:,'pdbins'] = pd.cut(ediff.loc[:,'pdiff'], bins=nbins, labels=False)
#
#maxfreqloc = ediff.groupby('pdbins').count().sort_values('max').index.array[-1] #in case the mode function returns a list or whatever
##maxfreqloc = ediff.loc[:,'pdbins'].mode().to_numpy()[0]
#maxbin = nbins - 1
#filterloc = maxfreqloc - (maxbin - maxfreqloc)
#
##ediff.loc[ediff.loc[:,'pdbins'] > filterloc].index.array
#
##ediff.loc[:,'diff'].plot.hist(bins=nbins)
##plt.show()
#
##orange transparent distribution below is the first bin of the plot above
##fig, ax = plt.subplots()
##ediff.loc[:,'pdiff'].plot.hist(bins=nbins, ax=ax)
##ediff.loc[ediff.loc[:,'dbins'] == 0, 'pdiff'].plot.hist(bins=nbins, ax=ax, alpha=0.5)
##plt.show()
#
##ediff.plot.scatter(x='sum', y='pdiff', alpha=0.002)
##plt.xscale('log')
##plt.show()
#
##EXPLANATION OF ABOVE
##If a mass channel's max is close to it's sum, then there's not many other data points on that mass channel - meaning this could be off-target, in terms of center of mass. It could still be within the ppm window, perhaps useful later. But for now, we're finding the center of mass.
##If a mass channels's max is far away from it's sum, then it's clear there's many other data points - this could be a good starting indicator for a center of mass. These would be good data points to keep to find the true center of mass. This allows for a filtering of peripheral points
#
#masscenters = ediff.loc[ediff.loc[:,'pdbins'] > filterloc].index.array
#ex = en.loc[masscenters]
#es = pd.DataFrame()
#es.loc[:,'intensity'] = ex.loc[:,'intensity'].sum(level=0)
##I need to make this a weighted average to verify if the plots I'm spitting out down below make any sense or not.
##Even upon plotting the weighted mean of the index based on the intensities, the [1,0] plots of m/z vs. index mean still don't make any sense, visually, for a lot of the plotted examples.
#es.loc[:,'index wmean'] = (ex.loc[:,'index'] * ex.loc[:,'intensity']).sum(level=0) / ex.loc[:,'intensity'].sum(level=0)
#es.loc[:,'index mean'] = ex.loc[:,'index'].mean(level=0)
#es.loc[:,'index std'] = ex.loc[:,'index'].std(level=0)
#
#
##lm = lws[:,0].flatten()
##li = lws[:,1].flatten()
#
#firstderivmaxes = signal.argrelextrema(es.loc[:,'intensity'].to_numpy(), np.greater)[0]
#
##In the short, biased look that I took. Using the second derivative gave me less mass channels, and more peak-filled channels. Without the 2nd deriv here, the mass channels have much narrower +- ppm windows, and there are less peaks in each plotted spectra. The difference between adjacent mass channels, in mass, seems to be smaller. I find this acceptable because it seems to make for a less complicated problem to iterate over later. Using only the first deruvatuve also seems to make my generated minimum ppm window and max ppm bridge (both below) have smaller windows.
##secondderivmaxes = signal.argrelextrema(es.loc[:,'intensity'].to_numpy()[firstderivmaxes], np.greater)[0]
##thirdderivmaxes = signal.argrelextrema(en.to_numpy()[firstderivmaxes][secondderivmaxes], np.greater)[0]
#
##Using the 2 rounds of local maximums here should show where the maximum of a jagged peak is. It seems like a decent assumption to assume the center-of-mass peaks aren't perfectly gaussian, there's more room for stochasticity to cause it to be otherwise.
#
##fig, ax = plt.subplots(8,1, figsize=(7, 20))
##d = 0.2
##n = 419.1
##w = 0.3
##
##an = 0
##while True:
##    a = ax[an]
##    ai = ax[an+1]
##    pinds = np.logical_and(es.index > n, es.index < n+w)
##    etp  = es.loc[pinds]
##    a.set_xlim(n, n+w)
##    ai.set_xlim(n, n+w)
##    etp.reset_index().plot.scatter(x='m/z', y='intensity', ax=a, s=0.1, color='purple')
##    etp.reset_index().plot.scatter(x='m/z', y='index mean', ax=ai, s=0.1, color='purple')
##    #etp.reset_index().plot.scatter(x='m/z', y='index mean', ax=ai, s=0.1, color='purple', yerr='index std', linewidth=0.05, alpha=0.4)
##    
##    a.vlines(es.index.array[firstderivmaxes], ymin=0, ymax=etp.max(), alpha=0.5, linewidth=0.1)
##    a.vlines(es.index.array[firstderivmaxes][secondderivmaxes], ymin=0, ymax=etp.max(), alpha=0.08, color='red')
##    #a.vlines(en.index.array[firstderivmaxes][secondderivmaxes][thirdderivmaxes], ymin=0, ymax=etp.max(), alpha=0.8, color='green')
##    a.set_yscale('log')
##    n += d
##    an += 2
##    if an >= len(ax):
##        break
##plt.show()
#
#fdmasses = es.index.to_numpy()[firstderivmaxes]
#
#ppmdist = np.diff(fdmasses) / fdmasses[:-1] * 1000000
#
##maxppmbridge should be a little bit lower than minppmwind? to minimize the number of max points being connected. And I'd rather there be some signal overlap rather than signal spread across too many centers due to a expanded connections. Does this make sense? It's hard to come to a good line of reasoning here as to which should be higher, I don't believe any argument for the reverse to be as straightforward as it seems.
##I seem to get fine results when they're the same thing, it also seems rather acceptable that both the minimum window and  maximum bridge size are the same value. I cannot think of better reasoning as to why one should be larger or smaller than the other.
#maxppmbridge = ppmdist.mean()
#minppmwind = ppmdist.mean()
#
##when profiled, this while loop was not a major bottle neck, it takes < 1s atm, can even be faster than that but it doesn't matter. Changing the lookahead value didn't change the length of ecoords. Output should therefore be the same.
#n = 0
#lookahead = 1000
#ecoords = []
#while True:
#    current = fdmasses[n:n+lookahead]
#    ppmmatrix = np.abs(current.reshape(-1,1) - current) / current * 1000000
#    bm = ppmmatrix < maxppmbridge
#    diag = np.argwhere(bm[1:].diagonal() == False)
#    zers = np.zeros(diag.shape).astype(int)
#    zers[1:] += diag[:-1] + 1
#    end = np.hstack((zers, diag)) + n
#    
#    if n + lookahead >= len(fdmasses):
#        extra = np.array([diag.max() + 1, len(current) - 1]) + n
#        end = np.vstack((end, extra))
#        ecoords.extend(end.tolist())
#        break
#    
#    ecoords.extend(end.tolist())
#    n += diag.max() + 1
#    
#feout = fdmasses[ecoords]
#mcppmwindows = np.abs((np.diff(feout)).flatten() / feout[:,1] * 1000000)
#masschannels = feout.mean(axis=1)
#mcppmwindows[mcppmwindows < minppmwind] = minppmwind
#gc.collect()
#
##ri = mcppmwindows < maxppmwind
##mcppmwindows = mcppmwindows[ri]
##masschannels = masschannels[ri]
#
#print('Mass Windows (ppm)')
#print(f'Count: {len(mcppmwindows)}')
#print(f'Min: {mcppmwindows.min()}')
#print(f'Max: {mcppmwindows.max()}')
#print(f'Mean: {mcppmwindows.mean()}')
#print(f'Median: {np.median(mcppmwindows)}')
#print(time() - mt)
#
#
##Now use this to filter mass channels with not enough indices, keep the # as low as possible to preserve minimum peak widths
##emasscount = ef.loc[:,['m/z', 'index']].groupby('m/z').count()
##eindecount = ef.loc[:,['m/z', 'index']].groupby('index').count()
##
#
##et = et.iloc[signalinds]
##now get peak widths of these, to determine ppm windows for individual channels, using your algorithm from spectral flowrate plotting -> Fix this also
##You'll need to make a lot of plots of different angles to see if your transforms work well
#
##tf = ef.loc[ef.loc[:,'index'] < 5]
##tfp = tf.pivot_table(values='intensity', index='m/z', columns='index')
#
#newcol = 'index'
#newrow = 'm/z'
#values = 'intensity'
#
##ef = ef.loc[ef.loc[:,'index'] < 5000]
#
##def sparse_pivot(ef, newcol, newrow, values):
#newcols = CategoricalDtype(sorted(ef.loc[:,newcol].unique()), ordered=True)
#newinds = CategoricalDtype(sorted(ef.loc[:,newrow].unique()), ordered=True)
#col = ef.loc[:,newcol].astype(newcols).cat.codes
#row = ef.loc[:,newrow].astype(newinds).cat.codes
#sm = sparse.csc_matrix((ef.loc[:,values], (row, col)), shape=(newinds.categories.size, newcols.categories.size))
#
##sm = sparse_pivot(ef, 'index', 'm/z', 'intensity')
#efs = pd.DataFrame.sparse.from_spmatrix(sm, index=newinds.categories, columns=newcols.categories)
#
#print(time() - mt)
#
#print('Assessing', sm.shape[0], 'channels for', len(masschannels), 'centers')
#
#magicnumber = 30000
#endnum = np.floor((sm.shape[0] / magicnumber) - 1).astype(int)
#mzindex = newinds.categories.to_numpy()
##
#n = 0
#sms, mzs, mzws = [], [], []
#start = magicnumber
##I remember at one point this while loop was acting funky, it didn't seem to be summing the channels together correctly, did I fix this? Also this is memory bound, multithread here, for this I might want to only consider a single mass channel at a time - currently this does whatever it can find - change to a for loop.
#
#while True:
#    ipiece = sm[start-magicnumber:start]
#    mpiece = mzindex[start-magicnumber:start]
#    
#    mci = np.logical_and(masschannels < mpiece.max(), masschannels > mpiece.min())
#    mcp = masschannels[mci]
#    mcw = mcppmwindows[mci]
#    
#    mppms = np.abs(mpiece.reshape(-1,1) - mcp) / mcp * 1000000
#    boolmatrix = mppms < mcw
#    boolmpiece = boolmatrix.any(axis=1)
#    
#    locations = np.argwhere(boolmatrix[boolmpiece])
#    mcplocs = np.unique(locations[:,1], return_inverse=True)[1]
#    
#    newpiece = pd.DataFrame(ipiece[locations[:,0]].todense(), index=mcplocs).sum(level=0).to_numpy()
#    mvals = np.unique(mcp[locations[:,1]]) 
#    mzwvals = mcw[np.argwhere(mcp == mvals)].flatten()
#     
#    sms.append(sparse.csc_matrix(newpiece))
#    mzs.append(mvals)
#    mzws.append(mzwvals)
#    sys.stdout.write(f'\r{n}/{endnum} {round(time() - mt, 4)}')
#    sys.stdout.flush()
#    n += 1
#    start += magicnumber
#    if start > sm.shape[0]:
#        break
#
#print(time() - mt)
#gc.collect()
#
#minpoints = et.min().to_numpy()[0]
#
#sms = np.asarray(sparse.vstack((sms)).todense())
#cinds = (sms > 0).sum(axis=1) > minpoints
##sms = sparse.csc_matrix(sms)
#sms = sms[cinds]
#mzs = np.asarray(list(itertools.chain(*mzs)))[cinds]
#mzws = np.asarray(list(itertools.chain(*mzws)))[cinds]
#
#medinds = np.median(sms, axis=1) == 0 #not all of these look bad, but this is a great way to determine noisy channels. Only ~120 of them were above 0 median on the file tested. You could see if any PSMs match up with these channels, it's doubtful but possible - but it would probably knock out a ton of noise.
#sms = sms[medinds]
#mzs = mzs[medinds]
#mzws = mzws[medinds]
#
#scanarray = np.sort(ef.loc[:,'index'].unique())
##scanarray = np.arange(sms.shape[1])
##savgol = signal.savgol_filter(sms, window_length=101, polyorder=2, mode='wrap', deriv=0, axis=1)
##savgol2 = signal.savgol_filter(sms, window_length=21, polyorder=0, mode='wrap', deriv=0)
#
#
##testing for correlation of neighboring mass channels. Use this to avoid having to plot a lot of different channels. Just see how many correlate > 0.5 or something. This would be a good comparison for the notebook when showing difference between first/second derivative, or different filtering steps. This is a good end-goal to optimize for.
##print(time() - mt)
##movingnum = 300
##start = movingnum
##adder = int(movingnum * (9/10))
##vals = []
##while True:
##    spiece = sms[start-movingnum:start]
##    vals.append(np.corrcoef(spiece).flatten())
##    start += adder
##    if start > len(sms):
##        break
##
##vals = np.asarray(vals)
##vals = vals.flatten()
##print(time() - mt)
##print((np.abs(vals) > 0.5).sum())
#
#
##This helps us check for signal overlap across channels. We don't want to see remnants of the same peak across two of our mass channels, that would give us the same data twice.
#
##point = find_nearest(mzs, 420.32)
##point = find_nearest(mzs, 870.6)
##point = find_nearest(mzs, 616.53)
##start = point[1] - 2
##nchans = 6
##channels = np.arange(nchans) + start
##
##l = 0
##h = sms.shape[1]
##
##for c in channels:
##    f, a = plt.subplots(2,2, figsize=(10,4), sharex='col')
##    n = mzs[c]
##    w = n * (mzws[c] / 1000000)
##    
##    finds = np.logical_and(ef.loc[:,'m/z'] > n-w, ef.loc[:,'m/z'] < n+w)
##    pinds = np.logical_and(es.index > n-w, es.index < n+w)
##    fpi = np.logical_and(es.index[firstderivmaxes] > n-w, es.index[firstderivmaxes] < n+w)
##    spi = np.logical_and(es.index[firstderivmaxes] > n-w, es.index[firstderivmaxes] < n+w)
##    
##    fp = ef.loc[finds]
##    fp = fp.groupby('m/z').agg({'intensity': 'sum', 'index': 'mean'})
##    
##    etp  = es.loc[pinds]
##    efd = es.index[firstderivmaxes][fpi]
##    esd = es.index[firstderivmaxes][spi]
##    
##    #windmean = np.arange(len(sms[c]))
##    
##    a[0,0].plot(scanarray, sms[c], '.', color='orange', alpha=0.1)
##    a[0,0].plot(scanarray, savgol[c], '-', color='green', alpha=0.5)
##    a[0,0].plot(scanarray, savgol2[c], '-', color='brown', alpha=0.5)
##    a[0,0].set_xlim(l, h)
##    #a.set_yscale('log')
##    
##    
##    fp.reset_index().plot.scatter(x='index', y='m/z', ax=a[1,0], s=0.1, color='green')
##    etp.reset_index().plot.scatter(x='index wmean', y='m/z', ax=a[1,0], s=0.2, color='purple')
##    a[1,0].set_xlim(l,h)
##    
##    
##    fp.reset_index().plot.scatter(x='m/z', y='intensity', ax=a[0,1], s=0.1, color='green')
##    etp.reset_index().plot.scatter(x='m/z', y='intensity', ax=a[0,1], s=0.2, color='purple')
##    a[0,1].vlines(efd, ymin=0, ymax=etp.loc[:,'intensity'].max(), alpha=0.5, linewidth=0.1)
##    a[0,1].vlines(esd, ymin=0, ymax=etp.loc[:,'intensity'].max(), alpha=0.08, color='red')
##    a[0,1].set_yscale('log')
##    a[0,1].yaxis.tick_right()
##    a[0,1].yaxis.set_label_position("right")
##    
##    
##    fp.reset_index().plot.scatter(x='m/z', y='index', ax=a[1,1], s=0.1, color='green')
##    etp.reset_index().plot.scatter(x='m/z', y='index wmean', ax=a[1,1], s=0.2, color='purple')
##    a[1,1].vlines(efd, ymin=0, ymax=etp.loc[:,'index wmean'].max(), alpha=0.5, linewidth=0.1)
##    a[1,1].vlines(esd, ymin=0, ymax=etp.loc[:,'index wmean'].max(), alpha=0.08, color='red')
##    a[1,1].yaxis.tick_right()
##    a[1,1].yaxis.set_label_position("right")
##    
##    plt.suptitle(''.join((str(c), ': ', str(mzs[c]), ' +/-', str(mzws[c].round(4)), 'ppm')))
##    f.subplots_adjust(hspace=0.05, wspace=0.05)
##    plt.show()
##    
##    widthbuffer = 20 #assumes peaks are at least 10 scans wide, which ain't too bad. Could probably also derive this number from the minimum scans per channels determined above - although it's softcoded. It might be wiser to keep this one hardcoded.
##    maxis = signal.argrelmax(savgol[c])[0]
##    minds = np.linspace(maxis-widthbuffer/2, maxis+widthbuffer/2, widthbuffer+1).transpose().astype(int)
##    minds[minds >= len(savgol[c])] = len(savgol[c]) - 1
##    minds[minds < 0] = 0
##    maxif = maxis[savgol[c][minds].max(axis=1) == savgol[c][maxis]]
##    maxin = maxif[savgol[c][maxif] > savgol[c][maxif].mean()]
##    
##    sgrad = np.gradient(savgol[c]) * 80 #scaling for the plot
##    sgradsgd = signal.savgol_filter(sgrad, window_length=61, polyorder=2, deriv=1, mode='wrap') * 60
##    sgdmif = signal.argrelmax(sgradsgd)[0]
##    sinds = np.linspace(sgdmif-widthbuffer/2, sgdmif+widthbuffer/2, widthbuffer+1).transpose().astype(int)
##    sinds[sinds >= len(sgrad)] = len(sgrad) - 1
##    sinds[sinds < 0] = 0
##    sgdm = sgdmif[sgradsgd[sinds].max(axis=1) == sgradsgd[sgdmif]]
##    
##    lm = np.ma.masked_array(np.repeat(sgdm.reshape(1,-1), len(maxin), axis=0), sgdm > maxin.reshape(-1,1))
##    rm = np.ma.masked_array(np.repeat(sgdm.reshape(1,-1), len(maxin), axis=0), sgdm < maxin.reshape(-1,1))
##    
##    lpoints = np.asarray(lm.max(axis=1))
##    rpoints = np.asarray(rm.min(axis=1))
##    pcaps = np.hstack((lpoints.reshape(-1,1), rpoints.reshape(-1,1)))
##    
##    f2, a2 = plt.subplots(figsize=(10,4))
##    a2.plot(scanarray, sms[c], '.', color='orange', alpha=0.1)
##    a2.plot(scanarray, savgol2[c], '-', color='brown', alpha=0.5)
##    a2.plot(scanarray, sgrad, '-', color='cyan', alpha=0.5)
##    a2.plot(scanarray, sgradsgd, '-', color='deeppink', alpha=0.5)
##    a2.plot(scanarray, savgol[c], '-', color='green', alpha=0.5)
##    a2.vlines(maxin, sms[c].max()*0.5, sms[c].max()*2, color='black', alpha=1)
##    a2.vlines(lpoints, 0, sgradsgd.max(), color='firebrick', alpha=0.5)
##    a2.vlines(rpoints, 0, sgradsgd.max(), color='navy', alpha=0.5)
##    a2.vlines(sgdm, 0, sms[c].max()*0.5, color='teal', alpha=0.5)
##    for lc, rc in pcaps:
##        a2.hlines(sgradsgd.max(), lc, rc, color='black', alpha=1)
##    a2.set_xlim(5000,7500)
##    #a2.set_ylim(0, 2000000)
##    plt.show()
#
#
#print(time() - mt)
#
widthbuffer = 5 #assumes peaks are at least 10, or this n/2, scans wide, which ain't too bad. Could probably also derive this number from the minimum scans per channels determined above - although it's softcoded. It might be wiser to keep this one hardcoded.
peakfill = 0.7 #This filters out things that seem to give on and off signal for whatever reason. Although even on major peaks I do see a lot of on and off signal, for reasons I can't explain. If it happens too much, I can kick it out via this.
prominencefilter = 5 #n times a prominence on at least one side
##maxis = signal.argrelmax(savgol, axis=1)
##sgrad = np.gradient(savgol, axis=1)
##sgradsgd = signal.savgol_filter(sgrad, window_length=61, polyorder=2, deriv=1, mode='wrap', axis=1)
##sgdmif = signal.argrelmax(sgradsgd, axis=1)
#
##use d = 5533 for testing, I guess
##d = 0 shows a good error where a single data point is transformed into a peak, this should be filtered out, due to there being no adjacency with this data.
##d = 479 for an example of the signal transforms failing to find a good peak base width, the need for corrections is there. 756?
enfs = []
for f in trps.index.levels[0]:
    channel = trps.loc[f, 'intensity'].to_numpy()
    scanarray = trps.loc[f].index.to_numpy()
    
    savgol = signal.savgol_filter(channel, window_length=101, polyorder=2, mode='wrap', deriv=0)
    #savgol2 = signal.savgol_filter(channel, window_length=21, polyorder=0, mode='wrap', deriv=0) #original
    savgol2 = signal.savgol_filter(channel, window_length=13, polyorder=0, mode='wrap', deriv=0)
    sgrad = np.gradient(savgol, scanarray)
    maxis = signal.argrelmax(savgol)[0]
    sgd = signal.savgol_filter(sgrad, window_length=61, polyorder=2, deriv=1, mode='wrap')
    sdf = signal.argrelmax(sgd)[0]
    
    #this gives the 'minimum distance' between two maxes, it's a hard sell but it's by scans so it's not really a minimum distance, it's more for making sure there aren't twin peaks screwing with the local maximum algorithm.
    minds = np.linspace(maxis-widthbuffer/2, maxis+widthbuffer/2, widthbuffer+1).transpose().astype(int)
    
    minds[minds >= len(savgol)] = len(savgol) - 1
    minds[minds < 0] = 0
    
    #this little shindig is for the case of DDAs, where the indices being used here is not a 1-to-1 relationship with the x-values in scanarray, it prevents extending the maximum peak buffer farther than it's supposed to
    migfill = scanarray[minds] > (scanarray[maxis] + widthbuffer//2).reshape(-1,1)
    milfill = scanarray[minds] < (scanarray[maxis] - widthbuffer//2).reshape(-1,1)
    
    miginds = np.ma.masked_array(minds, migfill)
    milinds = np.ma.masked_array(minds, milfill)
    
    migvals = np.asarray(miginds.max(axis=1))
    milvals = np.asarray(milinds.min(axis=1))
    
    minds[minds < milvals.reshape(-1,1)] = np.repeat(milvals, milfill.sum(axis=1), axis=0)
    minds[minds > migvals.reshape(-1,1)] = np.repeat(migvals, migfill.sum(axis=1), axis=0)
    
    maxin = maxis[savgol[minds].max(axis=1) == savgol[maxis]] #previously maxif
    #maxin = maxif[savgol[maxif] > savgol[maxif].mean()] #taking this out as a test atm
    
    #Below (using reminds) is removing peaks that are formed from a lone data point, it seems to be a common occurrence. Channels with median > 0 were already filtered in sms earlier. This doesn't need the extra steps with scanarray. This could probably happen later on, with the advantage of measuring all of the points involved in the potential peak instead of just those around the widthbuffer of the discovered maximum, but if it ain't broke don't break it.
    reminds = np.linspace(maxin-widthbuffer/2, maxin+widthbuffer/2, widthbuffer+1).transpose().astype(int)
    reminds[reminds < 0] = 0
    reminds[reminds >= len(channel)] = len(channel) - 1
    
    fininds = (channel[reminds] > np.median(channel)).sum(axis=1) >= widthbuffer * peakfill         
    if any(fininds):
        maxfin = maxin[fininds]
        
        #doing a the same filter as above, just on the 2nd derivative peaks
        sinds = np.linspace(sdf-widthbuffer/2, sdf+widthbuffer/2, widthbuffer+1).transpose().astype(int)
        sinds[sinds >= len(sgrad)] = len(sgrad) - 1
        sinds[sinds < 0] = 0
        
        sigfill = scanarray[sinds] > (scanarray[sdf] + widthbuffer//2).reshape(-1,1)
        silfill = scanarray[sinds] < (scanarray[sdf] - widthbuffer//2).reshape(-1,1)
        
        siginds = np.ma.masked_array(sinds, sigfill)
        silinds = np.ma.masked_array(sinds, silfill)
        
        sigvals = np.asarray(siginds.max(axis=1))
        silvals = np.asarray(silinds.min(axis=1))
        
        sinds[sinds < silvals.reshape(-1,1)] = np.repeat(silvals, silfill.sum(axis=1), axis=0)
        sinds[sinds > sigvals.reshape(-1,1)] = np.repeat(sigvals, sigfill.sum(axis=1), axis=0)
        
        sgdm = sdf[sgd[sinds].max(axis=1) == sgd[sdf]]
        
        lm = np.ma.masked_array(np.repeat(sgdm.reshape(1,-1), len(maxfin), axis=0), sgdm > maxfin.reshape(-1,1))
        rm = np.ma.masked_array(np.repeat(sgdm.reshape(1,-1), len(maxfin), axis=0), sgdm < maxfin.reshape(-1,1))
        
        lpoints = np.asarray(lm.max(axis=1))
        rpoints = np.asarray(rm.min(axis=1))
        
        reminds = np.logical_or(lm.mask.all(axis=1), rm.mask.all(axis=1))
        if reminds.any():
            lpoints = lpoints[~reminds]
            rpoints = rpoints[~reminds]
            maxfin = maxfin[~reminds]
        
        rpoints[rpoints > len(savgol)] = len(savgol) - 1
        
        #pcaps = np.hstack((lpoints.reshape(-1,1), rpoints.reshape(-1,1)))
        
        #trimmed things have zeros stripped off the ends
        
        peaklocs = [] #center of where the peak is located
        
        speakrs = [] #R^2 values for correlation of savitsky-golay transformation and raw data
        s2peakrs = [] #R^2 values for correlation of second savitsky-golay transformation and raw data
        
        speakpvals = [] #p-values for correlation of savitsky-golay transformations and raw data
        s2peakpvals = [] #p-values for correlation of second savitsky-golay transformations and raw data
        
        rheights = [] #maximum of the raw, untransformed data
        sheights = [] #maximums of the savitsky-golay transformation
        s2heights = [] #maximums of the second savitsky-golay transformation
        
        lefties = [] #left point of base width
        righties = [] #right point of base width
        tlefties = [] #left point of trimmed base width
        trighties = [] #right point of trimmed base width
        
        rlproms = [] #prominence of left-most point of the raw data
        rrproms = [] #prominence of the right-most point of the raw data
        
        slproms = [] #prominence of the left-most point of the original selection via third-derivative compared to the max of the savitsky-golay transform
        srproms = [] #prominence of the right-most point of the original selection via third-derivative compared to the max of the savitsky-golay transform
        
        s2lproms = [] #prominence of the left-most point of the original selection via third-derivative compared to the max of the second savitsky-golay transform
        s2rproms = [] #prominence of the right-most point of the original selection via third-derivative compared to the max of the second savitsky-golay transform
        
        s2tlproms = [] #trimmed left prominence of the second savitsky-golay transform
        s2trproms = [] #trimmed right prominence of the second savitsky-golay transform
        
        #Widths at half-max by height. Determined by counting total values greater than half-height.
        rhbwhms = [] #raw data
        shbwhms = [] #savgol transform
        s2hbwhms = [] #second savgol transform
        
        rhgoals = [] #true half-max value determined by height / 2 for raw data
        shgoals = [] #true half-max value determined by height / 2 for savgol transform
        s2hgoals = [] #true half-max value determined by height / 2 for the second savgol transform
        
        #Left and right-hand values used in calculating the width at half-max via a different method than above with the 'goals' lists. This method finds the nearest values to the desired half-max height and associates a point somewhere along the peak at the appropriate value to be used for the width at half-max measurement.
        rwhmls = [] #the left-hand point used in calculating width at half-max.
        swhmls = [] #the left-hand point used in calculating width at half-max.
        s2whmls = [] #the left-hand point used in calculating width at half-max.
        
        rwhmrs = [] #the right-hand point used in calculating width at half-max.
        swhmrs = [] #the right-hand point used in calculating width at half-max.
        s2whmrs = [] #the right-hand point used in calculating width at half-max.
        
        #Heights at the half-max values determined above via the find_nearest function
        rwhmlhs = [] #height of the left-hand point used in calculating width at half-max
        swhmlhs = [] #height of the left-hand point used in calculating width at half-max
        s2whmlhs = [] #height of the left-hand point used in calculating width at half-max
        
        rwhmrhs = [] #height of the right-hand point used in calculating width at half-max
        swhmrhs = [] #height of the right-hand point used in calculating width at half-max
        s2whmrhs = [] #height of the right-hand point used in calculating width at half-max
        
        #q1 - quadrant, area under the curves from the left-hand point to the left-hand width at half-max point
        rq1areas = []
        sq1areas = []
        s2q1areas = []
        
        #q2 - quadrant, area under the curves from the left-hand point to the left-hand width at half-max point
        rq2areas = []
        sq2areas = []
        s2q2areas = []
        
        #q3 - quadrant, area under the curves from the left-hand point to the left-hand width at half-max point
        rq3areas = []
        sq3areas = []
        s2q3areas = []
        
        #q4 - quadrant, area under the curves from the left-hand point to the left-hand width at half-max point
        rq4areas = []
        sq4areas = []
        s2q4areas = []
        
        rareas = [] #area under the curve of the raw data
        sareas = [] #area under the savitsky-golay curve used for finding peaks
        s2areas = [] #area under the second savitsky-golay curve spanning the window found via third derivative of the savitsky-golay transformation
        
        rhmareas = [] #area under the raw data curve within the half-max bounds
        shmareas = [] #area under the savitsky-golay curve within the half-max bounds
        s2hmareas = [] #area under the second savitsky-golay curve within the half-max bounds
        
        iterinds = [] #index for keeping track of peaks
        
        
        endframe = pd.DataFrame()
        for m, l, r in zip(maxfin, lpoints, rpoints):
            lprominence = savgol2[m] / savgol2[l]
            rprominence = savgol2[m] / savgol2[r]
            if np.logical_and(lprominence > prominencefilter, rprominence > prominencefilter): #using logical_and filtered out a lot of one-sided, assymetrical peaks that were really just lop-sided blobs. This helps find peak-like shapes that are clearly resolved across the relevant mass channel.
                scorr = stats.pearsonr(channel[l:r], savgol[l:r])
                s2corr = stats.pearsonr(channel[l:r], savgol2[l:r])
                
                speakrs.append(scorr[0])
                s2peakrs.append(s2corr[0])
                
                speakpvals.append(scorr[1])
                s2peakpvals.append(s2corr[1])
                
                #You need to add l and r as values to the dataframe, and convert the widths to time as well
                
                peaklocs.append(scanarray[m])
                
                rh = channel[l:r].argmax()
                sfh = savgol[l:r].argmax()
                s2fh = savgol2[l:r].argmax()
                
                rheights.append(channel[l+rh])
                sheights.append(savgol[l+s2fh])
                s2heights.append(savgol2[l+s2fh])
                
                #rbasewidths.append(scanarray[r] - scanarray[l])
                
                tl = l
                while True:
                    if savgol2[tl] == 0:
                        tl += 1
                    else:
                        break
                
                tr = r
                while True:
                    if savgol2[tr] == 0:
                        tr -= 1
                    else:
                        break
                
                lefties.append(scanarray[l])
                righties.append(scanarray[r])
                tlefties.append(scanarray[tl])
                trighties.append(scanarray[tr])
                
                tlprominence = savgol2[m] / savgol2[tl]
                trprominence = savgol2[m] / savgol2[tr]
                
                rlprominence = channel[l:r].max() / channel[l]
                slprominence = savgol[l:r].max() / savgol[l]
                
                rrprominence = channel[l:r].max() / channel[r]
                srprominence = savgol[l:r].max() / savgol[r]

                rhalfheightgoal = channel[l+rh]/2
                shalfheightgoal = savgol[l+sfh]/2
                s2halfheightgoal = savgol2[l+s2fh]/2
                

                if slprominence > 1 and srprominence > 1:
                    slhp = find_nearest(savgol[l:l+sfh], shalfheightgoal)
                    srhp = find_nearest(savgol[l+sfh:r], shalfheightgoal)
                    shh = np.mean([slhp[0], srhp[0]])
                    slhmp = scanarray[l+slhp[1]]
                    srhmp = scanarray[l+sfh+srhp[1]]
                    sq1area = np.trapezoid(savgol[l:l+slhp[1]], scanarray[l:l+slhp[1]])
                    sq2area = np.trapezoid(savgol[l+slhp[1]:l+sfh], scanarray[l+slhp[1]:l+sfh])
                    sq3area = np.trapezoid(savgol[l+sfh:l+srhp[1]], scanarray[l+sfh:l+srhp[1]])
                    sq4area = np.trapezoid(savgol[l+srhp[1]:r], scanarray[l+srhp[1]:r])
                    shmarea = np.trapezoid(savgol[l+slhp[1]:l+srhp[1]], scanarray[l+slhp[1]:l+srhp[1]])
                else:
                    slhp = [np.nan, np.nan]
                    srhp = [np.nan, np.nan]
                    shh = np.nan
                    slhmp = np.nan
                    srhmp = np.nan
                    sq1area = np.nan
                    sq2area = np.nan
                    sq3area = np.nan
                    sq4area = np.nan
                    shmarea = np.nan

                if rlprominence > 1 and rrprominence > 1:
                    rlhp = find_nearest(channel[l:l+rh], rhalfheightgoal)
                    rrhp = find_nearest(channel[l+rh:r], rhalfheightgoal)
                    rhh = np.mean([rlhp[0], rrhp[0]])
                    rlhmp = scanarray[l+rlhp[1]]
                    rrhmp = scanarray[l+rh+rrhp[1]]
                    rq1area = np.trapezoid(channel[l:l+rlhp[1]], scanarray[l:l+rlhp[1]])
                    rq2area = np.trapezoid(channel[l+rlhp[1]:l+rh], scanarray[l+rlhp[1]:l+rh])
                    rq3area = np.trapezoid(channel[l+rh:l+rrhp[1]], scanarray[l+rh:l+rrhp[1]])
                    rq4area = np.trapezoid(channel[l+rrhp[1]:r], scanarray[l+rrhp[1]:r])
                    rhmarea = np.trapezoid(channel[l+rlhp[1]:l+rrhp[1]], scanarray[l+rlhp[1]:l+rrhp[1]])
                else:
                    rlhp = [np.nan, np.nan]
                    rrhp = [np.nan, np.nan]
                    rhh = np.nan
                    rlhmp = np.nan
                    rrhmp = np.nan
                    rq1area = np.nan
                    rq2area = np.nan
                    rq3area = np.nan
                    rq4area = np.nan
                    rhmarea = np.nan

                
                rlproms.append(rlprominence)
                rrproms.append(rrprominence)
                
                slproms.append(slprominence)
                srproms.append(srprominence)
                
                s2lproms.append(lprominence)
                s2rproms.append(rprominence)
                
                s2tlproms.append(tlprominence)
                s2trproms.append(trprominence)
                #tbasewidths.append(scanarray[tr] - scanarray[tl])
                
                #rlhp = find_nearest(channel[l:l+rh], rhalfheightgoal)
                s2lhp = find_nearest(savgol2[l:l+s2fh], s2halfheightgoal)
                
                #rrhp = find_nearest(channel[l+rh:r], rhalfheightgoal)
                s2rhp = find_nearest(savgol2[l+s2fh:r], s2halfheightgoal)
                
                #rhh = np.mean([rlhp[0], rrhp[0]])
                s2hh = np.mean([s2lhp[0], s2rhp[0]])
                
                rhharray = scanarray[l:r][channel[l:r] > rhalfheightgoal]
                shharray = scanarray[l:r][savgol[l:r] > shalfheightgoal]
                s2hharray = scanarray[l:r][savgol2[l:r] > s2halfheightgoal]
                
                rhbwhms.append(rhharray.max() - rhharray.min())
                shbwhms.append(shharray.max() - shharray.min())
                s2hbwhms.append(s2hharray.max() - s2hharray.min())
                
                rhgoals.append(rhalfheightgoal)
                shgoals.append(shalfheightgoal)
                s2hgoals.append(s2halfheightgoal)
                
                #rlhmp = scanarray[l+rlhp[1]]
                s2lhmp = scanarray[l+s2lhp[1]]
                
                #rrhmp = scanarray[l+rh+rrhp[1]]
                s2rhmp = scanarray[l+s2fh+s2rhp[1]]
                
                #whms.append(scanarray[s2fh] + scanarray[s2rhp[1]] - scanarray[s2lhp[1]])
                
                rwhmls.append(rlhmp)
                swhmls.append(slhmp)
                s2whmls.append(s2lhmp)
                
                rwhmrs.append(rrhmp)
                swhmrs.append(srhmp)
                s2whmrs.append(s2rhmp)
                
                rwhmlhs.append(rlhp[0])
                swhmlhs.append(slhp[0])
                s2whmlhs.append(s2lhp[0])
                
                rwhmrhs.append(rrhp[0])
                swhmrhs.append(srhp[0])
                s2whmrhs.append(s2rhp[0])
                
                #4 quadrants of area
                #rq1areas.append(np.trapezoid(channel[l:l+rlhp[1]], scanarray[l:l+rlhp[1]]))
                rq1areas.append(rq1area)
                sq1areas.append(sq1area)
                s2q1areas.append(np.trapezoid(savgol2[l:l+s2lhp[1]], scanarray[l:l+s2lhp[1]]))
                
                #rq2areas.append(np.trapezoid(channel[l+rlhp[1]:l+rh], scanarray[l+rlhp[1]:l+rh]))
                rq2areas.append(rq2area)
                sq2areas.append(sq2area)
                s2q2areas.append(np.trapezoid(savgol2[l+s2lhp[1]:l+s2fh], scanarray[l+s2lhp[1]:l+s2fh]))
                
                #rq3areas.append(np.trapezoid(channel[l+rh:l+rrhp[1]], scanarray[l+rh:l+rrhp[1]]))
                rq3areas.append(rq3area)
                sq3areas.append(sq3area)
                s2q3areas.append(np.trapezoid(savgol2[l+s2fh:l+s2rhp[1]], scanarray[l+s2fh:l+s2rhp[1]]))
                
                #rq4areas.append(np.trapezoid(channel[l+rrhp[1]:r], scanarray[l+rrhp[1]:r]))
                rq4areas.append(rq4area)
                sq4areas.append(sq4area)
                s2q4areas.append(np.trapezoid(savgol2[l+s2rhp[1]:r], scanarray[l+s2rhp[1]:r]))
                
                rareas.append(np.trapezoid(channel[l:r], scanarray[l:r]))
                sareas.append(np.trapezoid(savgol[l:r], scanarray[l:r]))
                s2areas.append(np.trapezoid(savgol2[l:r], scanarray[l:r]))
                
                rhmareas.append(rhmarea)
                shmareas.append(shmarea)
                s2hmareas.append(np.trapezoid(savgol2[l+s2lhp[1]:l+s2rhp[1]], scanarray[l+s2lhp[1]:l+s2rhp[1]]))
                
                iterinds.append(d)
                
                #if trimming the zeros from the ends gives more than a 10% or so difference in the peak width, then use the base widths estimated from the half-height widths? If the length of the trimmed version is shorter than the estimated via half-max, then use the trimmed base width instead, but it obviously shouldn't be thinner than the two half-max width points together.
                #take a median of the channel/fit values from before, after, and in between the base width estimated via half-max width. Keep these as data points. If the medians are zero, it might be junk. Take means too?
                
                #f, ax = plt.subplots(1, 2, figsize=(10,5))
                #ax[0].plot(scanarray[l:r], channel[l:r], '.')
                #ax[0].plot(scanarray[l:r], savgol2[l:r])
                #ax[0].vlines(l+fh, ymin=0, ymax=channel[l:r].max())
                #
                #ax[0].vlines(l+fh-(fh-s2lhp[1])*2, ymin=0, ymax=s2lhp[0]*2)
                #ax[0].vlines(l+fh+s2rhp[1]*2, ymin=0, ymax=s2rhp[0]*2)
                #
                #ax[0].vlines(l+s2lhp[1], ymin=0, ymax=s2lhp[0])
                #ax[0].vlines(l+fh+s2rhp[1], ymin=0, ymax=s2rhp[0])
                #
                #
                #ax[1].plot(scanarray[l:r], channel[l:r], '.')
                #ax[1].plot(scanarray[l:r], savgol2[l:r])
                #ax[1].vlines(l+fh, ymin=0, ymax=channel[l:r].max())
                #
                #ax[1].vlines(l+fh-(fh-s2lhp[1])*2, ymin=0, ymax=s2lhp[0]*2)
                #ax[1].vlines(l+fh+s2rhp[1]*2, ymin=0, ymax=s2rhp[0]*2)
                #
                #ax[1].vlines(l+s2lhp[1], ymin=0, ymax=s2lhp[0])
                #ax[1].vlines(l+fh+s2rhp[1], ymin=0, ymax=s2rhp[0])
                #
                #ax[1].set_yscale('log')
                #plt.show()
                #print(corr)
                #print(lprominence, rprominence)
                #print('d:', d)
        if any(rheights):
            endframe.loc[:,'retention time (scan)'] = peaklocs
            endframe.loc[:,'s r-squared'] = speakrs
            endframe.loc[:,'s2 r-squared'] = s2peakrs
            endframe.loc[:,'s p-value'] = speakpvals
            endframe.loc[:,'s2 p-value'] = s2peakpvals
            
            endframe.loc[:,'raw heights'] = rheights
            endframe.loc[:,'sfit heights'] = sheights
            endframe.loc[:,'s2fit heights'] = s2heights
            
            endframe.loc[:,'left base point'] = lefties
            endframe.loc[:,'right base point'] = righties
            endframe.loc[:,'trimmed left base point'] = tlefties
            endframe.loc[:,'trimmed right base point'] = trighties
            
            endframe.loc[:,'r left prominence'] = rlproms
            endframe.loc[:,'s left prominence'] = slproms
            endframe.loc[:,'s2 left prominence'] = s2lproms
            
            endframe.loc[:,'r right prominence'] = rrproms
            endframe.loc[:,'s right prominence'] = srproms
            endframe.loc[:,'s2 right prominence'] = s2rproms
            
            endframe.loc[:,'trimmed left prominence'] = s2tlproms
            endframe.loc[:,'trimmed right prominence'] = s2trproms
            
            endframe.loc[:,'r height-based width at half-max'] = rhbwhms
            endframe.loc[:,'s height-based width at half-max'] = shbwhms
            endframe.loc[:,'s2 height-based width at half-max'] = s2hbwhms
            
            endframe.loc[:,'r true half-max height'] = rhgoals
            endframe.loc[:,'s true half-max height'] = shgoals
            endframe.loc[:,'s2 true half-max height'] = s2hgoals
            
            endframe.loc[:,'r left-hand half-max point'] = rwhmls
            endframe.loc[:,'s left-hand half-max point'] = swhmls
            endframe.loc[:,'s2 left-hand half-max point'] = s2whmls
            
            endframe.loc[:,'r right-hand half-max point'] = rwhmrs
            endframe.loc[:,'s right-hand half-max point'] = swhmrs
            endframe.loc[:,'s2 right-hand half-max point'] = s2whmrs
            
            endframe.loc[:,'r left-hand half-max height'] = rwhmlhs
            endframe.loc[:,'s left-hand half-max height'] = swhmlhs
            endframe.loc[:,'s2 left-hand half-max height'] = s2whmlhs
            
            endframe.loc[:,'r right-hand half-max height'] = rwhmrhs
            endframe.loc[:,'s right-hand half-max height'] = swhmrhs
            endframe.loc[:,'s2 right-hand half-max height'] = s2whmrhs
            
            endframe.loc[:,'r q1 area'] = rq1areas
            endframe.loc[:,'s q1 area'] = sq1areas
            endframe.loc[:,'s2 q1 area'] = s2q1areas
            
            endframe.loc[:,'r q2 area'] = rq2areas
            endframe.loc[:,'s q2 area'] = sq2areas
            endframe.loc[:,'s2 q2 area'] = s2q2areas
            
            endframe.loc[:,'r q3 area'] = rq3areas
            endframe.loc[:,'s q3 area'] = sq3areas
            endframe.loc[:,'s2 q3 area'] = s2q3areas
            
            endframe.loc[:,'r q4 area'] = rq4areas
            endframe.loc[:,'s q4 area'] = sq4areas
            endframe.loc[:,'s2 q4 area'] = s2q4areas
            
            endframe.loc[:,'r area'] = rareas
            endframe.loc[:,'s fit area'] = sareas
            endframe.loc[:,'s2 fit area'] = s2areas
            
            endframe.loc[:,'r width at half-max area'] = rhmareas
            endframe.loc[:,'s width at half-max area'] = shmareas
            endframe.loc[:,'s2 width at half-max area'] = s2hmareas
            
            endframe.loc[:,'iteration number'] = iterinds
            endframe.loc[:,'mass channel'] = mzs[d]
            endframe.loc[:,'+- ppm'] = mzws[d]
            
            enfs.append(endframe)

print(time() - mt)
enfs = pd.concat(enfs)
#
#enfs.loc[:,'raw base widths (min)'] = scantimes.loc[enfs.loc[:,'right base point']].to_numpy() - scantimes.loc[enfs.loc[:,'left base point']].to_numpy()
#enfs.loc[:,'trimmed base widths (min)'] = scantimes.loc[enfs.loc[:,'trimmed right base point']].to_numpy() - scantimes.loc[enfs.loc[:,'trimmed left base point']].to_numpy()
#
#enfs.loc[:,'r nearest-based width af half-max'] = enfs.loc[:,'r right-hand half-max point'] - enfs.loc[:,'r left-hand half-max point']
#enfs.loc[:,'s2 nearest-based width af half-max'] = enfs.loc[:,'s2 right-hand half-max point'] - enfs.loc[:,'s2 left-hand half-max point']
#
#newfolder = '/'.join(('/'.join((mzmlfile.split('/')[:-2])), 'peaks/'))
#newfile = ''.join((mzmlfile.split('/')[-1].split('.')[0], '.peaks.csv'))
#endfn = ''.join((newfolder, newfile))
#enfs.to_csv(endfn, index=False)
#
#
#
#if __name__ == '__main__':
#    main(sys.argv[1])

    #It is true that some peaks are beautiful, and others look to be sampled in a very choppy manner. Some might even question whether certain data presented as peaks are even proper peaks. I would argue that they are proper peaks. [SHOW THIS PART] It's likely these are less intense peaks. Just because a mass channels samples a lot of zero-points between some data points where the peak seems to be, it still seems to signify, visually, that something is eluting off of the column. Representing the width of this analyte to the best estimate available is still a valid data point. Whether it is struggling to ionize under competition, or unable to be consistently sampled in the mass analyzer does not take away it's validity because it is clear that something is there. The only way to legitemately remove this type of data point would be to impose a minimum intensity, or width threshhold, some of this is already done to a degree by filtering a % prominence. There is also a chance that some of this noise results from the process of joining mass windows, although if R^2 tends to align better with higher intensity peaks, this would probably not be true. I do not believe doing this would show any difference in the data other than a fine bottom line where the difference was marked. I would prefer a reason to filter this be shown with the data, rather than just assumed to be of good merit.
    #Because of the lack of any deconvolution of spectra, this analysis is more representative of signal, and less representative of analyte populations.
    
    #f2, a2 = plt.subplots(figsize=(10,4))
    #a2.plot(scanarray, channel, '.', color='orange', alpha=1)
    #a2.plot(scanarray, savgol2, '-', color='brown', alpha=1)
    #a2.plot(scanarray, sgrad*80, '-', color='cyan', alpha=0.5)
    #a2.plot(scanarray, sgd*80*60, '-', color='deeppink', alpha=0.5)
    #a2.plot(scanarray, savgol, '-', color='green', alpha=0.5)
    #a2.vlines(maxfin, channel.max()*0.5, channel.max()*2, color='black', alpha=1)
    #a2.vlines(lpoints, 0, (sgd*80*60).max(), color='firebrick', alpha=0.5)
    #a2.vlines(rpoints, 0, (sgd*80*60).max(), color='navy', alpha=0.5)
    #a2.vlines(sgdm, 0, channel.max()*0.5, color='teal', alpha=0.5)
    #for lc, rc in pcaps:
    #    a2.hlines((sgd*80*60).max(), lc, rc, color='black', alpha=1)
    #a2.set_xlim(1800,3000)
    #a2.set_ylim(0, 500000)
    #plt.show()


#widths - savgol2 hits the median, 0, on either sides
#heights - sms max within peak widths
#correlation of either savgol transform to the actual data points, use the better fit to get width at half-max, and perhaps you can look at the distribution of these R^2 after to detertmine a cutoff.
#width at half height - start at the widths, then move back until you find a value closest to the half of the sms max height
#when displaying peak width as a # of scans, show a distribution of time per scan, for every scan in this file, or show scan time as a y-value across an x-axis, so you can show scan time for every file.
#You should use the info from your peak width measurements to filter out illegitemate findings, if the distance of the max height is too assymetric to the right or left boundaries, then something might be off. You could plot a distribution of these distances individually, or the distribution of their difference to filter out outliers or errors. You could also look at WHERE the asymetric peaks seem to happen, and perhaps you could look at every other peak in that vicinity to try and figure out why it's like that? Were those analytes pushed?
#Find peak widths with ~80% overlap or something, then deconvolute these masses across the entire spectrum!
#You can still identify blobs by using the width at half height - When iterating through the left/right boundaries, if there is no curve at half the height, it ain't no peak son.

#When this is finished, you'll want to set out to re-extract the massranges +- window arrays from the file to see that what you extract independently matches what you've gotten from this file.
#You can also set the requirement that base peak boundaries are at least < 10% of the peak height on savgol or something too.

#will later need to filter out peaks that have a base width higher than width at half-max

#start = 100
#nchans = 6
#channels = np.arange(nchans) + start
#
#f, ax = plt.subplots(len(channels), 1, figsize=(7.5, 3*len(channels)))
#l = 0
#h = sms.shape[1]
#
#for a, c in zip(ax, channels):
#    a.plot(scanarray, sms[c], '.', color='orange', alpha=0.1)
#    a.plot(scanarray, savgol[c], '-', color='green', alpha=0.5)
#    a.plot(scanarray, savgol2[c], '-', color='brown', alpha=0.5)
#    a.set_title(''.join((str(c), ': ', str(mzs[c]), ' +/-', str(mzws[c].round(4)), 'ppm')))
#    a.set_xlim(l, h)
#    #a.set_yscale('log')
#plt.show()


#what these plots show me:
#I think picking by maximums is going to be played out. It's given me good insight into the data, and I think I want to write a version of Binner that 'bins of the fly', rather than taking in the entirety of the data. The input data can be the intensities and indices - it will also be able to handle weird outliers. Some of the mass traces are like the chromatographic blobs you had to deal with before, these are just mass blobs - peak width might still be completely do-able, but perhaps find a way to determine how many mass-blobs coincide with chromatographic-blobs?
#I also want to, in the future, group bins that have similar intensity profiles and similar mean/std dev of indices for charge state deconvolution. After the on-the-fly binning, you can do a binning of all the groups of RT via mean/stdev. You can weight the masses involved via intensity.

#Things I need to add in now:
#A max ppm bridge between the second derivative values (seems like it's already done)
#A max ppm bridge between the second derivative values and any neighboring points. This should prevent picking '2nd-der maximums' that aren't actually flanked on both sides, thus covering up the weakness of finding local maximums. This is necessary because the local max algorithm assumes all data to be contiguous.  - NOT YET IMPLEMENTED - because I also don't like this idea anymore

#to show that a vertical line of masses in sm center around a point:
 #- Look at the average of all masses that show up
 #- Then look at the variance of any of these sub-masses per intensity via detailed heatmap.

#start = magicnumber
#windowoverlap = 0.05
#while True:
#    ipiece = sm[start-magicnumber:start]
    

#npeaks = 10000
#maxval = 100000
#minval = 1000
#
#upperrange = np.random.normal(maxval, maxval - minval, npeaks)
#
##do the np.abs after this process too, as an alternative
#upperrange = np.abs(upperrange)
#
#stdevs = np.roll(upperrange, 1)
#
#out = np.random.normal(loc=upperrange, scale=stdevs, size=(npeaks, npeaks))
#
#out = out.flatten()
#
#plt.hist(out, bins=100)
#plt.show()
