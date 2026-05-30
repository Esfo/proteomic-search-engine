import numpy as np
import matplotlib.pyplot as plt
from pyteomics import mzml
from time import time
import pandas as pd
import gc
import concurrent.futures
from scipy import sparse, signal, stats, integrate
from pandas.api.types import CategoricalDtype
import itertools
import sys
import os
#plt.rcParams["figure.dpi"] = 100

#folder = '/store/flowcharacterizations/round2/MS1s/mzMLs/'
#mzmlfile = '/store/FE/mzMLs/test/190701_F350_yM2.mzML'
#runorder = open('/store/flowcharacterizations/runorder')
#runorder = [i.strip() for i in runorder]
#runorder = [''.join((folder, '200612_', i, '.mzML')) for i in runorder]
#files = list(itertools.zip_longest(*[iter(runorder)]*2))
#files = os.listdir(folder)
#files = [i for i in files if i.endswith('.mzML.')

def scanfunc(scan):
    et = pd.DataFrame(scan['intensity array'], columns=['intensity'])
    et.loc[:,'m/z'] = scan['m/z array'].round(4)
    try:
        et.loc[:,'index'] = scan['index']
    except ValueError:
        return []
    et.loc[:,'ms level'] = scan['ms level']
    et.loc[:,'time (min)'] = scan['scanList']['scan'][0]['scan start time'].real
    return [et]

def boolcount(b):
    m = np.append(b[0], np.diff(b))
    _, c = np.unique(m.cumsum(), return_index=True)
    out = np.diff(np.append(c, len(b)))
    if b[0] == False:
        out = np.append(0, out)
    if len(out) % 2:
        out = np.append(out, 0)
    out = out.reshape(-1, 2)
    return out

def channel_aggregation(mc, mw, mzindex, sm):
    few = np.array([mc - (mc / 1000000) * mw, mc + (mc / 1000000) * mw])
    fei = np.logical_and(mzindex >= few[0], mzindex <= few[1])
    fw = np.argwhere(fei)
    summedarray = sm[fei].sum(axis=0)[0]
    return summedarray, [fw.min(), fw.max()+1, mc, mw]

def distance_placement(pile, sd):
    ed = []
    cn = 0
    for row in pile:
        sr = [0, 0]
        for nr, cv in enumerate(row):
            sr[nr] += sd[cn:cn+cv].sum()
            cn += cv
        ed.append(sr)
    return np.asarray(ed)

def trimmer(x):
    try:
        return np.argwhere(np.diff((x == 0).cumsum()) == 0)[0][0]
    except IndexError:
        return 0

def main(mzmlfile, widthbuffer=20, peakfill=0.7, nproc=8, out=None):
    widthbuffer = int(widthbuffer) #assumes peaks are at least 10, or this n/2, scans wide, (no it doesn't) which ain't too bad. Could probably also derive this number from the minimum scans per channels determined above - although it's softcoded. It might be wiser to keep this one hardcoded.
    peakfill = float(peakfill) #This filters out things that seem to give on and off signal for whatever reason. Although even on major peaks I do see a lot of on and off signal, for reasons I can't explain. If it happens too much, I can kick it out via this.
    nproc = int(nproc) #number of processors to use for multiprocessing/multithreading.

#Opening the file and extracting data
    mt = time()
    ef = []
    msrun = mzml.MzML(mzmlfile)
    for t in msrun.map(lambda scan: scanfunc(scan), processes=nproc):
        ef.extend(t)

    gc.collect()
    print(time() - mt, '- File Extracted')

#not all these formats seem to matter for speed, for m/z values though the resolution obviously won't match what's recorded. And it seems to ~half the number of rows in the sparse matrix.
    ef = pd.concat(ef)
    scantimes = ef.loc[:,('index', 'time (min)')].drop_duplicates()
    scantimes.set_index('index', inplace=True)
    ef = ef.loc[ef.loc[:,'ms level'] == 1]
#ef.loc[:,'m/z'] = ef.loc[:,'m/z'].round(4)
#ef.loc[:,'intensity'] = ef.loc[:,'intensity'].astype(int)
#ef.loc[:,'index'] = ef.loc[:,'index'].astype(np.int16)

#Excluding any mass channels that hold less than the mean intensity across time as a data point. These are left out for having too little data. This makes the data manageable for analysis development. It should be removed/replaced later by removing mass channels that don't  meet the minimum number of indices with contiguous data points.
#et = ef.set_index(['m/z', 'index']).sum(axis=0, level=0) - EDIT: This has been done below
#print(et.mean().to_numpy()[0], 'scans as the temporary minimum scan threshhold for inclusion')
#et = et.loc[(et > et.mean()).to_numpy()]
    et = ef.drop('intensity', axis=1)
    et = et.groupby('m/z').count()
    et = et.loc[(et > et.loc[(et > 1).to_numpy()].mean()).to_numpy()]
#you can sanity check this with ((et > 1).any(axis=1) != (et > 1).all(axis=1)).sum(), it should be 0
#print('Imposing a minimum of', et.min().to_numpy()[0], 'data points per unmerged mass channel') #this isn't actually imposing this on all of the data, it only imposes it for a shortened data-set used for finding the centers of mass.
#^Mean I was getting was 6 for simple index counts, when removing all the 1's, which was the vast vast majority, the mean shifts to ~9.6. There's ~500000 data points above 10 so this isn't bad, and 10 is a good minimum index points
#what could be good for presentation purposes is showing the histogram of et after it gets transformed into counts to show just how maany masses are only read once
#Given that the purpose here is to find centers of mass, it's acceptable to remove datapoints that don't even hold this many indices across the chromatogram

    en = ef.set_index('m/z')
    en.sort_index(inplace=True)
    en = en.loc[et.index.unique().array] #this was previously non-unique, which seemed wrong, it made the array larger than when it was originally formed
#masschannels = et.reset_index().to_numpy()[:,0].flatten()
#ints = et.reset_index().to_numpy()[:,1].flatten()

    nbins = np.sqrt(en.index.unique().shape[0]).round().astype(int)
    ediff = pd.DataFrame()
    ediff.loc[:,'max'] = en.loc[:,'intensity'].max(level=0)
    ediff.loc[:,'sum'] = en.loc[:,'intensity'].sum(level=0)
    ediff.loc[:,'diff'] = ediff.loc[:,'sum'] - ediff.loc[:,'max']
#ediff.loc[:,'dbins'] = pd.cut(ediff.loc[:,'diff'], bins=nbins, labels=False)
#ediff.plot.hist(bins=nbins)

    ediff.loc[:,'pdiff'] = ediff.loc[:,'diff'] / ediff.loc[:,'sum']
    ediff.loc[:,'pdbins'] = pd.cut(ediff.loc[:,'pdiff'], bins=nbins, labels=False)

    maxfreqloc = ediff.groupby('pdbins').count().sort_values('max').index.array[-1] #in case the mode function returns a list or whatever
#maxfreqloc = ediff.loc[:,'pdbins'].mode().to_numpy()[0]
    maxbin = nbins - 1
    filterloc = maxfreqloc - (maxbin - maxfreqloc) #assumes a ~symmetrical rightward distribution, even though this is always iffy on the symmetry, hanging more towards 1 than 0, seems to always happen!


#ediff.loc[ediff.loc[:,'pdbins'] > filterloc].index.array

#ediff.loc[:,'diff'].plot.hist(bins=nbins)
#plt.show()

#orange transparent distribution below is the first bin of the plot above
#fig, ax = plt.subplots()
#ediff.loc[:,'pdiff'].plot.hist(bins=nbins, ax=ax)
#ediff.loc[ediff.loc[:,'dbins'] == 0, 'pdiff'].plot.hist(bins=nbins, ax=ax, alpha=0.5)
#plt.show()

#ediff.plot.scatter(x='sum', y='pdiff', alpha=0.002)
#plt.xscale('log')
#plt.show()

#EXPLANATION OF ABOVE
#If a mass channel's max is close to it's sum, then there's not many other data points on that mass channel - meaning this could be off-target, in terms of center of mass. It could still be within the ppm window, perhaps useful later. But for now, we're finding the center of mass.
#If a mass channels's max is far away from it's sum, then it's clear there's many other data points - this could be a good starting indicator for a center of mass. These would be good data points to keep to find the true center of mass. This allows for a filtering of peripheral points

    masscenters = ediff.loc[ediff.loc[:,'pdbins'] >= filterloc].index.array
#retaining the majority of masses
#ediff.loc[ediff.loc[:,'pdbins'] >= filterloc].reset_index().loc[:,'m/z'].plot.hist(bins=100)
#ediff.loc[ediff.loc[:,'pdbins'] <= filterloc].reset_index().loc[:,'m/z'].plot.hist(bins=100)
    ex = en.loc[masscenters]
    es = pd.DataFrame()
    es.loc[:,'intensity'] = ex.loc[:,'intensity'].sum(level=0)
#I need to make this a weighted average to verify if the plots I'm spitting out down below make any sense or not.
#Even upon plotting the weighted mean of the index based on the intensities, the [1,0] plots of m/z vs. index mean still don't make any sense, visually, for a lot of the plotted examples.
#es.loc[:,'index wmean'] = (ex.loc[:,'index'] * ex.loc[:,'intensity']).sum(level=0) / ex.loc[:,'intensity'].sum(level=0)
#es.loc[:,'index mean'] = ex.loc[:,'index'].mean(level=0)
#es.loc[:,'index std'] = ex.loc[:,'index'].std(level=0)


#lm = lws[:,0].flatten()
#li = lws[:,1].flatten()

    firstderivmaxes = signal.argrelextrema(es.loc[:,'intensity'].to_numpy(), np.greater)[0]

#In the short, biased look that I took. Using the second derivative gave me less mass channels, and more peak-filled channels. Without the 2nd deriv here, the mass channels have much narrower +- ppm windows, and there are less peaks in each plotted spectra. The difference between adjacent mass channels, in mass, seems to be smaller. I find this acceptable because it seems to make for a less complicated problem to iterate over later. Using only the first deruvatuve also seems to make my generated minimum ppm window and max ppm bridge (both below) have smaller windows.
#secondderivmaxes = signal.argrelextrema(es.loc[:,'intensity'].to_numpy()[firstderivmaxes], np.greater)[0]
#thirdderivmaxes = signal.argrelextrema(en.to_numpy()[firstderivmaxes][secondderivmaxes], np.greater)[0]

#Using the 2 rounds of local maximums here should show where the maximum of a jagged peak is. It seems like a decent assumption to assume the center-of-mass peaks aren't perfectly gaussian, there's more room for stochasticity to cause it to be otherwise.

#fig, ax = plt.subplots(8,1, figsize=(7, 20))
#d = 0.2
#n = 419.1
#w = 0.3
#
#an = 0
#while True:
#    a = ax[an]
#    ai = ax[an+1]
#    pinds = np.logical_and(es.index > n, es.index < n+w)
#    etp  = es.loc[pinds]
#    a.set_xlim(n, n+w)
#    ai.set_xlim(n, n+w)
#    etp.reset_index().plot.scatter(x='m/z', y='intensity', ax=a, s=0.1, color='purple')
#    etp.reset_index().plot.scatter(x='m/z', y='index mean', ax=ai, s=0.1, color='purple')
#    #etp.reset_index().plot.scatter(x='m/z', y='index mean', ax=ai, s=0.1, color='purple', yerr='index std', linewidth=0.05, alpha=0.4)
#    
#    a.vlines(es.index.array[firstderivmaxes], ymin=0, ymax=etp.max(), alpha=0.5, linewidth=0.1)
#    a.vlines(es.index.array[firstderivmaxes][secondderivmaxes], ymin=0, ymax=etp.max(), alpha=0.08, color='red')
#    #a.vlines(en.index.array[firstderivmaxes][secondderivmaxes][thirdderivmaxes], ymin=0, ymax=etp.max(), alpha=0.8, color='green')
#    a.set_yscale('log')
#    n += d
#    an += 2
#    if an >= len(ax):
#        break
#plt.show()

    fdmasses = es.index.to_numpy()[firstderivmaxes]

    ppmdist = np.diff(fdmasses) / fdmasses[:-1] * 1000000

#maxppmbridge should be a little bit lower than minppmwind? to minimize the number of max points being connected. And I'd rather there be some signal overlap rather than signal spread across too many centers due to a expanded connections. Does this make sense? It's hard to come to a good line of reasoning here as to which should be higher, I don't believe any argument for the reverse to be as straightforward as it seems.
#I seem to get fine results when they're the same thing, it also seems rather acceptable that both the minimum window and  maximum bridge size are the same value. I cannot think of better reasoning as to why one should be larger or smaller than the other.
    maxppmbridge = ppmdist.mean()
    minppmwind = ppmdist.mean()

#when profiled, this while loop was not a major bottle neck, it takes < 1s atm, can even be faster than that but it doesn't matter. Changing the lookahead value didn't change the length of ecoords. Output should therefore be the same.
    n = 0
    lookahead = 1000
    ecoords = []
    while True:
        current = fdmasses[n:n+lookahead]
        ppmmatrix = np.abs(current.reshape(-1,1) - current) / current * 1000000
        bm = ppmmatrix <= maxppmbridge
        diag = np.argwhere(bm[1:].diagonal() == False)
        zers = np.zeros(diag.shape).astype(int)
        zers[1:] += diag[:-1] + 1
        end = np.hstack((zers, diag)) + n
        
        if n + lookahead >= len(fdmasses):
            extra = np.array([diag.max() + 1, len(current) - 1]) + n
            end = np.vstack((end, extra))
            ecoords.extend(end.tolist())
            break
        
        ecoords.extend(end.tolist())
        n += diag.max() + 1
        
    feout = fdmasses[ecoords]
    masschannels = feout.mean(axis=1).round(4)
#this calculation of mcppmwindows it a problem, you should use masschannels to calculate the ppm error here
    mcppmwindows = np.abs((np.diff(feout)).reshape(1,-1) / feout[:,1] * 1000000)[0] / 2
    mcppmwindows[mcppmwindows < minppmwind] = minppmwind #above is just a part of choosing the centers, things can still be excluded if I take this stringently so I don't want to have actual ppm windows of 0
    gc.collect()

#ri = mcppmwindows < maxppmwind
#mcppmwindows = mcppmwindows[ri]
#masschannels = masschannels[ri]

    print('Mass Windows (ppm)')
    print(f'Count: {len(mcppmwindows)}')
    print(f'Min: {mcppmwindows.min()}')
    print(f'Max: {mcppmwindows.max()}')
    print(f'Mean: {mcppmwindows.mean()}')
    print(f'Median: {np.median(mcppmwindows)}')
    print(time() - mt)


#Now use this to filter mass channels with not enough indices, keep the # as low as possible to preserve minimum peak widths
#emasscount = ef.loc[:,['m/z', 'index']].groupby('m/z').count()
#eindecount = ef.loc[:,['m/z', 'index']].groupby('index').count()
#

#et = et.iloc[signalinds]
#now get peak widths of these, to determine ppm windows for individual channels, using your algorithm from spectral flowrate plotting -> Fix this also
#You'll need to make a lot of plots of different angles to see if your transforms work well

#tf = ef.loc[ef.loc[:,'index'] < 5]
#tfp = tf.pivot_table(values='intensity', index='m/z', columns='index')

    newcol = 'index'
    newrow = 'm/z'
    values = 'intensity'

#ef = ef.loc[ef.loc[:,'index'] < 5000]

#def sparse_pivot(ef, newcol, newrow, values):
    newcols = CategoricalDtype(sorted(ef.loc[:,newcol].unique()), ordered=True)
    newinds = CategoricalDtype(sorted(ef.loc[:,newrow].unique()), ordered=True)
    col = ef.loc[:,newcol].astype(newcols).cat.codes
    row = ef.loc[:,newrow].astype(newinds).cat.codes
    sm = sparse.csc_matrix((ef.loc[:,values], (row, col)), shape=(newinds.categories.size, newcols.categories.size))

#sm = sparse_pivot(ef, 'index', 'm/z', 'intensity')
#sm = pd.DataFrame.sparse.from_spmatrix(sm, index=newinds.categories, columns=newcols.categories)

    print(time() - mt, 'Initial arrays assembled')

    print('Assessing', sm.shape[0], 'channels for', len(masschannels), 'centers')

#magicnumber = 30000
#endnum = np.floor((sm.shape[0] / magicnumber) - 1).astype(int)
    mzindex = newinds.categories.to_numpy()
##
#n = 0
#sms, masschannels, mcppmwindows = [], [], []
#start = magicnumber
#
#while True:
#    ipiece = sm[start-magicnumber:start]
#    mpiece = mzindex[start-magicnumber:start]
#    
#    #mci = np.logical_and(masschannels <= mpiece.max(), masschannels >= mpiece.min()) #mpiece shouldn't be the deciding factor in this, it should be masschannels. you may want a mass channel that's higher than mpiece.max() because of an overlapping boundary
#    fei = np.logical_and(mpiece >= feout[:,0].reshape(-1,1), mpiece <= feout[:,1].reshape(-1,1))
#    mci = fei.any(axis=1)
#    mcp = masschannels[mci]
#    mcw = mcppmwindows[mci]
#    
#    mppms = np.abs(mpiece.reshape(-1,1) - mcp) / mcp * 1000000
#    boolmatrix = mppms < mcw
#    boolmpiece = boolmatrix.any(axis=1)
#    
#    locations = np.argwhere(boolmatrix[boolmpiece]) #This has been here forever, but seems wrong to me. It gives incorrect indices, which are later set free on ipiece, which should be a magicnumber-sized array, locations should be indexing within the magicnumber, but instead - this one indexes within which piece of boolmatrix showed true values - maxing out at much less than magicnumber
#    #Using this in conjunction with bt solves the problem I think
#    bt = np.argwhere(boolmpiece)
#    #mcplocs = np.unique(locations[:,1], return_inverse=True)[1] #didn't seem to have a purpose
#    
#    #newpiece = pd.DataFrame(ipiece[bt[locations[:,0]].reshape(1,-1)[0]].todense(), index=mcplocs).sum(level=0).to_numpy() #replaced with lack of mcplocs below
#    newpiece = pd.DataFrame(ipiece[bt[locations[:,0]].reshape(1,-1)[0]].todense(), index=locations[:,1]).sum(level=0).to_numpy()
#    mvals = np.unique(mcp[locations[:,1]]) 
#    mzwvals = mcw[np.argwhere(mcp == mvals)].flatten()
#     
#    sms.append(sparse.csc_matrix(newpiece))
#    masschannels.append(mvals)
#    mcppmwindows.append(mzwvals)
#    sys.stdout.write(f'\r{n}/{endnum} {round(time() - mt, 4)}')
#    sys.stdout.flush()
#    n += 1
#    start += magicnumber
#    if start > sm.shape[0]:
#        break


#when you put this on code review, make fake data by getting random numbers from -1 to 1, then subtract like 0.3, and make anything below 0 into a 0 to recreate the sparse matricies.
#sms, mzinwin = [], []
#endnum = len(masschannels)
#mt = time()
#for mcp, mcw in zip(masschannels, mcppmwindows):
#    few = np.array([mcp - (mcp / 1000000) * mcw, mcp + (mcp / 1000000) * mcw])
#    
#    fei = np.logical_and(mzindex >= few[0], mzindex <= few[1])
#    fw = np.argwhere(fei)
#    summedarray = sm[fei].sum(axis=0)[0]
#    
#    sms.append(summedarray)
#    mzinwin.append([fw.min(), fw.max()])
#    
#    #sys.stdout.write(f'\r{n}/{endnum} {round(time() - mt, 4)}')
#    #sys.stdout.flush()
#
#print(time() - mt)
#gc.collect()
    
    fut, mzinwin, sms = [], [], []
    mt = time()
    with concurrent.futures.ThreadPoolExecutor(nproc) as executor:
        for mc, mw in zip(masschannels, mcppmwindows):
            fut.append(executor.submit(channel_aggregation, mc, mw, mzindex, sm))
        for f in concurrent.futures.as_completed(fut):
            sa, mza = f.result()
            mzinwin.append(mza)
            sms.append(sa)
    print(time() - mt, 'Final arrays aggregated')
    
    sms = np.asarray(np.vstack((sms)))
    mzinwin = np.asarray(mzinwin)

    sms = sms[mzinwin[:,2].argsort()]
    mzinwin = mzinwin[mzinwin[:,2].argsort()]

    masschannels = mzinwin[:,2]
    mcppmwindows = mzinwin[:,3]
    mzinwin = mzinwin[:,:2].astype(int)
    gc.collect()

#minpoints = et.min().to_numpy()[0] #not entirely necessary, a defensive move that should honestly throw an error if something is wrong, rather than playing defense. But it isn't necessary and doesn't change anything either..

#sms = np.asarray(sparse.vstack((sms)).todense())
#mzinwin = np.asarray(mzinwin)
    cinds = (sms > 0).sum(axis=1) > widthbuffer

    if cinds.sum() != len(sms):
#sms = sparse.csc_matrix(sms)
        sms = sms[cinds]
#masschannels = np.asarray(list(itertools.chain(*masschannels)))[cinds]
        masschannels = masschannels[cinds]
#mcppmwindows = np.asarray(list(itertools.chain(*mcppmwindows)))[cinds]
        mcppmwindows = mcppmwindows[cinds]
        mzinwin = mzinwin[cinds]

    medinds = np.median(sms, axis=1) == 0 #not all of these look bad, but this is a great way to determine noisy channels. Only ~120 of them were above 0 median on the file tested. You could see if any PSMs match up with these channels, it's doubtful but possible - but it would probably knock out a ton of noise if turned into an exclusion list.

    if medinds.sum() != len(sms):
        sms = sms[medinds]
        masschannels = masschannels[medinds]
        mcppmwindows = mcppmwindows[medinds]
        mzinwin = mzinwin[medinds]

    scanarray = np.sort(ef.loc[:,'index'].unique())
#scanarray = np.arange(sms.shape[1])
#savgol = signal.savgol_filter(sms, window_length=101, polyorder=2, mode='wrap', deriv=0, axis=1)
#savgol2 = signal.savgol_filter(sms, window_length=21, polyorder=0, mode='wrap', deriv=0)


#testing for correlation of neighboring mass channels. Use this to avoid having to plot a lot of different channels. Just see how many correlate > 0.5 or something. This would be a good comparison for the notebook when showing difference between first/second derivative, or different filtering steps. This is a good end-goal to optimize for.
#print(time() - mt)
#movingnum = 300
#start = movingnum
#adder = int(movingnum * (9/10))
#vals = []
#while True:
#    spiece = sms[start-movingnum:start]
#    vals.append(np.corrcoef(spiece).flatten())
#    start += adder
#    if start > len(sms):
#        break
#
#vals = np.asarray(vals)
#vals = vals.flatten()
#print(time() - mt)
#print((np.abs(vals) > 0.5).sum())

#maxis = signal.argrelmax(savgol, axis=1)
#sgrad = np.gradient(savgol, axis=1)
#sgradsgd = signal.savgol_filter(sgrad, window_length=61, polyorder=2, deriv=1, mode='wrap', axis=1)
#sgdmif = signal.argrelmax(sgradsgd, axis=1)

#use d = 5533 for testing, I guess
#d = 0 shows a good error where a single data point is transformed into a peak, this should be filtered out, due to there being no adjacency with this data.
#d = 479 for an example of the signal transforms failing to find a good peak base width, the need for corrections is there. 756?
    enfs = []
    for d in range(len(sms)):
        channel = sms[d]
        
        #the savgol variable below provides an easier means of peak width estimation. The savgol2 variable would give a more accurate AUC measurement.
        savgol = signal.savgol_filter(channel, window_length=101, polyorder=2, mode='nearest', deriv=0)
        savgol2 = signal.savgol_filter(channel, window_length=21, polyorder=0, mode='nearest', deriv=0)
        sgrad = np.gradient(savgol) #not including the x-spacing for this function is alright here because the products of this variable are purely for peak-finding, it would only change scaling.
        maxis = signal.argrelmax(savgol)[0]
        sgd = signal.savgol_filter(sgrad, window_length=61, polyorder=2, deriv=1, mode='wrap')
        sgdq = signal.savgol_filter(savgol, window_length=251, polyorder=2, deriv=2, mode='wrap')
        sdf = signal.argrelmax(sgd)[0]
        sdfq = signal.argrelmax(sgdq)[0]
        
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
            
            #A lot of the below needs to be put in a function, it's currently done twice for both boundary-finding transforms
            sinds = np.linspace(sdf-widthbuffer/2, sdf+widthbuffer/2, widthbuffer+1).transpose().astype(int)
            sinds[sinds >= len(scanarray)] = len(scanarray) - 1
            sinds[sinds < 0] = 0
            
            sqinds = np.linspace(sdfq-widthbuffer/2, sdfq+widthbuffer/2, widthbuffer+1).transpose().astype(int)
            sqinds[sqinds >= len(scanarray)] = len(scanarray) - 1
            sqinds[sqinds < 0] = 0
            
            sigfill = scanarray[sinds] > (scanarray[sdf] + widthbuffer//2).reshape(-1,1)
            silfill = scanarray[sinds] < (scanarray[sdf] - widthbuffer//2).reshape(-1,1)
            
            sigqfill = scanarray[sqinds] > (scanarray[sdfq] + widthbuffer//2).reshape(-1,1)
            silqfill = scanarray[sqinds] < (scanarray[sdfq] - widthbuffer//2).reshape(-1,1)
            
            siginds = np.ma.masked_array(sinds, sigfill)
            silinds = np.ma.masked_array(sinds, silfill)
            
            sigqinds = np.ma.masked_array(sqinds, sigqfill)
            silqinds = np.ma.masked_array(sqinds, silqfill)
            
            sigvals = np.asarray(siginds.max(axis=1))
            silvals = np.asarray(silinds.min(axis=1))
            
            sigqvals = np.asarray(sigqinds.max(axis=1))
            silqvals = np.asarray(silqinds.min(axis=1))
            
            sinds[sinds < silvals.reshape(-1,1)] = np.repeat(silvals, silfill.sum(axis=1), axis=0)
            sinds[sinds > sigvals.reshape(-1,1)] = np.repeat(sigvals, sigfill.sum(axis=1), axis=0)
            
            sqinds[sqinds < silqvals.reshape(-1,1)] = np.repeat(silqvals, silqfill.sum(axis=1), axis=0)
            sqinds[sqinds > sigqvals.reshape(-1,1)] = np.repeat(sigqvals, sigqfill.sum(axis=1), axis=0)
            
            sgdm = sdf[sgd[sinds].max(axis=1) == sgd[sdf]]
            sgdmq = sdfq[sgdq[sqinds].max(axis=1) == sgdq[sdfq]]
            
            lm = np.ma.masked_array(np.repeat(sgdm.reshape(1,-1), len(maxfin), axis=0), sgdm >= maxfin.reshape(-1,1))
            rm = np.ma.masked_array(np.repeat(sgdm.reshape(1,-1), len(maxfin), axis=0), sgdm <= maxfin.reshape(-1,1))
            
            lmq = np.ma.masked_array(np.repeat(sgdmq.reshape(1,-1), len(maxfin), axis=0), sgdmq >= maxfin.reshape(-1,1))
            rmq = np.ma.masked_array(np.repeat(sgdmq.reshape(1,-1), len(maxfin), axis=0), sgdmq <= maxfin.reshape(-1,1))
            
            lpoints = lm.max(axis=1)
            rpoints = rm.min(axis=1)
            
            lqpoints = lmq.max(axis=1)
            rqpoints = rmq.min(axis=1)
            
            pb = lqpoints.mask & lpoints.mask | rqpoints.mask & rpoints.mask
            if pb.any():
                lpoints = lpoints[~pb]
                rpoints = rpoints[~pb]
                lqpoints = lqpoints[~pb]
                rqpoints = rqpoints[~pb]
                maxfin = maxfin[~pb]
            
            lpoints[lpoints.mask] = lqpoints[lpoints.mask]
            lqpoints[lqpoints.mask] = lpoints[lqpoints.mask]
            rpoints[rpoints.mask] = rqpoints[rpoints.mask]
            rqpoints[rqpoints.mask] = rpoints[rqpoints.mask]
            
            #really just a canary now
            lpoints.set_fill_value(-1)
            lqpoints.set_fill_value(-1)
            rpoints.set_fill_value(-1)
            rqpoints.set_fill_value(-1)
            
            lpoints = lpoints.filled()
            lqpoints = lqpoints.filled()
            rpoints = rpoints.filled()
            rqpoints = rqpoints.filled()
            
            rpoints[rpoints > len(savgol)] = len(savgol) - 1
            rqpoints[rqpoints > len(savgol)] = len(savgol) - 1

            if maxfin.size > 0:
                tfs = pd.DataFrame()
                tfs.loc[:,'lq'] = lqpoints
                tfs.loc[:,'l'] = lpoints
                tfs.loc[:,'m'] = maxfin
                tfs.loc[:,'r'] = rpoints
                tfs.loc[:,'rq'] = rqpoints

                #using the inner points so as to give enough of a buffer between peaks that only have slightly overlapping boundaries
                tfs.loc[:,'mxl'] = tfs.loc[:,('lq', 'l')].min(axis=1)
                tfs.loc[:,'mxr'] = tfs.loc[:,('rq', 'r')].max(axis=1)
                #tfs.loc[:,'mnl'] = tfs.loc[:,('lq', 'l')].max(axis=1)
                #tfs.loc[:,'mnr'] = tfs.loc[:,('rq', 'r')].min(axis=1)
                #tfs.loc[:,'mel'] = tfs.loc[:,('lq', 'l')].mean(axis=1).round().astype(int)
                #tfs.loc[:,'mer'] = tfs.loc[:,('rq', 'r')].mean(axis=1).round().astype(int)
                
                for t in tfs.index.unique():
                    tfs.loc[t, 'bl'] = tfs.loc[t, 'mxl'] + savgol[tfs.loc[t, 'mxl']:tfs.loc[t, 'm']].argmin()
                    tfs.loc[t,'br'] = tfs.loc[t, 'm'] + savgol[tfs.loc[t, 'm']:tfs.loc[t, 'mxr']].argmin()
                
                tfs = tfs.astype(int)
                #from https://stackoverflow.com/questions/65260750/grouping-overlapping-integer-pairs-into-a-smaller-array
                #tfs.loc[:,'group'] = (~tfs.loc[:,'mr'].shift().gt(tfs.loc[:,'ml']-widthbuffer/2)).cumsum()
                #tos = tfs.groupby('group').agg({'ml':'min', 'mr':'max', 'lq':'min', 'l':'min', 'r':'max', 'rq':'max', 'm': lambda x: channel[x].argmax()})
                tfs.loc[:,'group'] = (~tfs.loc[:,'br'].shift().gt(tfs.loc[:,'bl'])).cumsum()
                tos = tfs.groupby('group').agg({'bl':'min', 'br':'max', 'm': lambda x: savgol[x].argmax()})
                
                tfs.set_index('group', inplace=True)
                #make a SO post about this, how to aggregate based on channel argmaxes for the m value in the dataframe, simple pach below is fine for now - try/except is for groups that have only 1 value.
                for g in tfs.index.unique():
                    try:
                        tos.loc[g, 'm'] = tfs.loc[g, 'm'].tolist()[tos.loc[g, 'm']]
                    except TypeError:
                        tos.loc[g, 'm'] = tfs.loc[g, 'm']
                
                #lqpoints, lpoints, maxfin, rpoints, rqpoints = tos.loc[:,('lq', 'l', 'm', 'r', 'rq')].to_numpy().transpose()
                #lqpoints, lpoints, maxfin, rpoints, rqpoints = lqpoints.tolist(), lpoints.tolist(), maxfin.tolist(), rpoints.tolist(), rqpoints.tolist()
                
                lbs, rbs, ms = tos.to_numpy().transpose().tolist()
                #could definitely be iterating over less, but this isn't a big enough impact on performance for me to care about rn
                #for m, lo, ro, lq, rq in zip(maxfin, lpoints, rpoints, lqpoints, rqpoints):
                for lo, ro, sm in zip(lbs, rbs, ms):
                    rbase = savgol[sm:]
                    lbase = savgol[:sm+1]
                    
                    #these pretty much get me a cumulative argmin
                    rmins = np.minimum.accumulate(rbase)
                    rlinds = rbase <= rmins
                    rsd = np.diff(scanarray[sm-1:])
                    
                    lmins = np.flip(np.minimum.accumulate(np.flip(lbase)))
                    llinds = lbase <= lmins
                    lsd = np.diff(scanarray[:sm+2])
                    
                    #counts of adjacent boolean values, first value is always true
                    rpile = boolcount(rlinds)
                    lpile = boolcount(llinds)
                    
                    #gets distances between scanarray points so widthbuffer isn't applied directly to indices
                    rpc = distance_placement(rpile, rsd)
                    lpc = distance_placement(lpile, lsd)
                    
                    lrange = np.ptp(scanarray[lo:sm+1])
                    rrange = np.ptp(scanarray[sm:ro+1])
                    
                    #giving at least as much room as the initial estimates
                    rint = np.argwhere(rpc.sum(axis=1).cumsum() >= rrange)[0][0]
                    lint = np.argwhere(np.flip(lpc, axis=0).sum(axis=1).cumsum() >= lrange)[0][0]
                    
                    #this should NOT be widthbuffer/2 because there should be a widthbuffer/2 amount of space from the top of either side of a peak, considering that this should hypothetically consider there being 2 peaks next to each other, then this is the appropriate amount of space.
                    rbool = rpc[rint:, 1] >= widthbuffer
                    lbool = np.flip(lpc, axis=0)[lint:, 1] >= widthbuffer
                    if rbool.size == 0:
                        riv = 1
                    elif rbool.cumsum().max() > 0:
                        riv = np.argwhere(rbool.cumsum() > 0)[0][0] + rint
                    else:
                        riv = len(rbool)
                    
                    if lbool.size == 0:
                        liv = 1
                    elif lbool.cumsum().max() > 0:
                        liv = np.argwhere(lbool.cumsum() > 0)[0][0] + lint
                    else:
                        liv = len(lbool) + 1
                    
                    rzone = rpile[:riv+1].sum()
                    lzone = lpile[-liv:].sum()
                    
                    #taking min from left can select first occuring minimum value if the min value repeats. Not a problem for the right points.
                    lmin = savgol[len(lbase) - lzone:sm].min()
                    lms = np.argwhere(savgol[len(lbase) - lzone:sm] == lmin).max()
                    sl = len(lbase) - lzone + lms
                    sr = len(savgol) - len(rbase) + savgol[sm:sm+rzone].argmin()
                    
                    s2m = sl + savgol2[sl:sr].argmax()
                    rm = sl + channel[sl:sr].argmax()
                    
                    s2r = sr - trimmer(np.flip(savgol2[sm:sr]))
                    s2l = sl + trimmer(savgol2[sl:sm])
                    
                    rr = sr - trimmer(np.flip(channel[sm:sr]))
                    rl = sl + trimmer(channel[sl:sm])
                     
                    slbaseline = savgol[sl]
                    s2lbaseline = savgol2[s2l]
                    rlbaseline = channel[rl]
                    
                    srbaseline = savgol[sr]
                    s2rbaseline = savgol2[s2r]
                    rrbaseline = channel[rr]
                    
                    if slbaseline < 0:
                        slbaseline = 0
                    if s2lbaseline < 0:
                        s2lbaseline = 0
                    if srbaseline < 0:
                        srbaseline = 0
                    if s2rbaseline < 0:
                        s2rbaseline = 0
                        
                    smbl = np.min([slbaseline, srbaseline])
                    s2mbl = np.min([s2lbaseline, s2rbaseline])
                    rmbl = np.min([rlbaseline, rrbaseline])
                    
                    shh = scanarray[sl:sr][savgol[sl:sr] > savgol[sm] - ((savgol[sm] - smbl) / 2)]
                    s2hh = scanarray[s2l:s2r][savgol2[s2l:s2r] > savgol2[s2m] - ((savgol2[s2m] - s2mbl) / 2)]
                    rhh = scanarray[rl:rr][channel[rl:rr] > channel[rm] - ((channel[rm] - rmbl) / 2)]
                    
                    #imposing this as a minimum number of points across a peak, and as a means to avoid the strangely found super assymetrical peaks that occur on the side of larger peaks.
                    roundpass = True
                    if ((sm - sl) < (widthbuffer / 2)) & ((sr - sm) < (widthbuffer / 2)):
                        roundpass = False
                    if not shh.size & s2hh.size & rhh.size:
                        roundpass = False

                    if roundpass:
                        tosdict = {}
                        tosdict['retention time (scan index)'] = sm
                        
                        #a second test of raw signal noisiness between the l and r points needs to be done here to filter out random noise points, I wonder if this could replace the prominence filter?
                        #if np.logical_and(lprominence > prominencefilter, rprominence > prominencefilter): #using logical_and filtered out a lot of one-sided, assymetrical peaks that were really just lop-sided blobs. This helps find peak-like shapes that are clearly resolved across the relevant mass channel.
                        scorr = stats.pearsonr(channel[sl:sr], savgol[sl:sr])
                        s2corr = stats.pearsonr(channel[s2l:s2r], savgol2[s2l:s2r])
                        
                        #process failing for trailing peaks, more likely in the flush?, see masschannels == 678.6906
                        
                        tosdict['s p-value'] = scorr[1]
                        tosdict['s2 p-value'] = s2corr[1]
                        
                        tosdict['s R^2'] = scorr[0]
                        tosdict['s2 R^2'] = s2corr[0]
                        
                        tosdict['mass channel'] = masschannels[d]
                        tosdict['+/- ppm window'] = mcppmwindows[d]
                        
                        tosdict['s height'] = savgol[sm]
                        tosdict['s2 height'] = savgol2[s2m]
                        tosdict['r height'] = channel[rm]
                        
                        #the savgol areas can still have negative ends, causing a subtraction from the overall area, may want to find non-negative boundaries after determining the l and r points.
                        tosdict['s area'] = np.trapezoid(savgol[sl:sr], scanarray[sl:sr])
                        tosdict['s2 area'] = np.trapezoid(savgol2[s2l:s2r], scanarray[s2l:s2r])
                        tosdict['r area'] = np.trapezoid(channel[rl:rr], scanarray[rl:rr])
                        
                        tosdict['r peak fill'] = (channel[rl:rr] > 0).sum() / len(channel[rl:rr])
                        
                        #get non-zero points for all prominences at some point, should be taken from the lowest >0 point on either side.
                        tosdict['s left prominence'] = savgol[sm] / slbaseline if slbaseline > 0 else 0
                        tosdict['s2 left prominence'] = savgol2[s2m] / s2lbaseline if s2lbaseline > 0 else 0
                        tosdict['r left prominence'] = channel[rm] / rlbaseline if rlbaseline > 0 else 0
                        
                        tosdict['s right prominence'] = savgol[sm] / srbaseline if srbaseline > 0 else 0
                        tosdict['s2 right prominence'] = savgol2[s2m] / s2rbaseline if s2rbaseline > 0 else 0
                        tosdict['r right prominence'] = channel[rm] / rrbaseline if rrbaseline > 0 else 0
                        
                        tosdict['s baseline area subtraction'] = np.trapezoid(np.asarray([slbaseline, srbaseline]), np.array([scanarray[sl], scanarray[sr]]))
                        tosdict['s2 baseline area subtraction'] = np.trapezoid(np.asarray([s2lbaseline, s2rbaseline]), np.array([scanarray[s2l], scanarray[s2r]]))
                        tosdict['r baseline area subtraction'] = np.trapezoid(np.asarray([rlbaseline, rrbaseline]), np.array([scanarray[rl], scanarray[rr]]))
                        
                        tosdict['s left baseline'] = slbaseline
                        tosdict['s2 left baseline'] = s2lbaseline
                        tosdict['r left baseline'] = rlbaseline
                        
                        tosdict['s right baseline'] = srbaseline
                        tosdict['s2 right baseline'] = s2rbaseline
                        tosdict['r right baseline'] = rrbaseline
                        
                        tosdict['s left boundary (scan index)'] = sl
                        tosdict['s2 left boundary (scan index)'] = s2l
                        tosdict['r left boundary (scan index)'] = rl
                        
                        tosdict['s right boundary (scan index)'] = sr
                        tosdict['s2 right boundary (scan index)'] = s2r
                        tosdict['r right boundary (scan index)'] = rr
                        
                        tosdict['s width at half-max left boundary (scan)'] = shh.min()
                        tosdict['s width at half-max right boundary (scan)'] = shh.max()
                        
                        tosdict['s2 width at half-max left boundary (scan)'] = s2hh.min()
                        tosdict['s2 width at half-max right boundary (scan)'] = s2hh.max()
                        
                        tosdict['r width at half-max left boundary (scan)'] = rhh.min()
                        tosdict['r width at half-max right boundary (scan)'] = rhh.max()
                        
                        #things that can be done with the entire dataframe:
                            #width, and half-max width in both scan and time - only list indices here
                            #half-max height - but it might hinder other things, you can get the half-max baseline and subtract area below it to get area above half-max if you calculate this in the loop
                        #things to add in this loop:
                            #area and baseline estimates
                        #endcall filter, do another round of overlap testing here?
                        tosdict['iteration index'] = d
                        #tos.loc[:, 'right boundary (scan index)'] - tos.loc[:,'left boundary (scan index)'] <= widthbuffer
                        enfs.append(tosdict)
        
                #RECORD SCRATCH
                #How well do the integrated areas of peaks under fitting vs raw data match each other?? This could be my legitemacy filter.
                #you can look at the distribution of % differences.
                #perhaps you could do this on top of just a p-value filter?
                #how would this %diff area distribution compare to the R^2 distribution? Scatter plot and histogram comparison pls.
                #this would be the area that doesn't subtract the bottom part.. this might fail for that trailing peak that broke the algo? You could also dice the peak into n slices and compare area like that (while also getting your modulatable quadrants)
                #^p-value worked for it though. How would the distribution of baseline values look for that instead? Maybe this could be a legitemate filter.
                
                
                #another thought, how does the above boundary-finding bit work for the s2 and raw data? Can I just do this for both of them?
                
                #when trimming bottom area, you'll want to also collect 2 different type of width at half-maxes, half-max of the apex - baseline, then half-max of apex if it's available. There needs to be a test if the baseline is over the half-max of the apex.


    print(time() - mt)
    enfs = pd.DataFrame(enfs)

#without doing these 2 independently you'll get some bad peaks in the flush, peaks that start in different places and all end on the same left or right base point
#new edit: these shouldn't be necessary anymore because of the overlap checking, check if they end up dropping anything now
    enfs = enfs.sort_values('r height').drop_duplicates(['mass channel', 's left boundary (scan index)'], keep='last')
    enfs = enfs.sort_values('r height').drop_duplicates(['mass channel', 's right boundary (scan index)'], keep='last')

    enfs.loc[:,'s width at half-max (min)'] = scantimes.loc[enfs.loc[:,'s width at half-max right boundary (scan)']].to_numpy() - scantimes.loc[enfs.loc[:,'s width at half-max left boundary (scan)']].to_numpy()
    enfs.loc[:,'s2 width at half-max (min)'] = scantimes.loc[enfs.loc[:,'s2 width at half-max right boundary (scan)']].to_numpy() - scantimes.loc[enfs.loc[:,'s2 width at half-max left boundary (scan)']].to_numpy()
    enfs.loc[:,'r width at half-max (min)'] = scantimes.loc[enfs.loc[:,'r width at half-max right boundary (scan)']].to_numpy() - scantimes.loc[enfs.loc[:,'r width at half-max left boundary (scan)']].to_numpy()

    enfs.loc[:,'s right boundary (scan)'] = scanarray[enfs.loc[:,'s right boundary (scan index)']]
    enfs.loc[:,'s2 right boundary (scan)'] = scanarray[enfs.loc[:,'s2 right boundary (scan index)']]
    enfs.loc[:,'r right boundary (scan)'] = scanarray[enfs.loc[:,'r right boundary (scan index)']]

    enfs.loc[:,'s left boundary (scan)'] = scanarray[enfs.loc[:,'s left boundary (scan index)']]
    enfs.loc[:,'s2 left boundary (scan)'] = scanarray[enfs.loc[:,'s2 left boundary (scan index)']]
    enfs.loc[:,'r left boundary (scan)'] = scanarray[enfs.loc[:,'r left boundary (scan index)']]

    enfs.loc[:,'s base width (min)'] = scantimes.loc[enfs.loc[:,'s right boundary (scan)']].to_numpy() - scantimes.loc[enfs.loc[:,'s left boundary (scan)']].to_numpy()
    enfs.loc[:,'s2 base width (min)'] = scantimes.loc[enfs.loc[:,'s2 right boundary (scan)']].to_numpy() - scantimes.loc[enfs.loc[:,'s2 left boundary (scan)']].to_numpy()
    enfs.loc[:,'r base width (min)'] = scantimes.loc[enfs.loc[:,'r right boundary (scan)']].to_numpy() - scantimes.loc[enfs.loc[:,'r left boundary (scan)']].to_numpy()

    enfs.loc[:,'retention time (scan)'] = scanarray[enfs.loc[:,'retention time (scan index)']]
    enfs.loc[:,'retention time (min)'] = scantimes.loc[enfs.loc[:,'retention time (scan)']].to_numpy()
    
    if not out:
        newfolder = '/'.join(('/'.join((mzmlfile.split('/')[:-2])), 'peaks/'))
        newfile = ''.join((mzmlfile.split('/')[-1].split('.')[0], '.peaks.csv'))
        endfn = ''.join((newfolder, newfile))
    else:
        endfn = out
    enfs.to_csv(endfn, index=False)


if __name__ == '__main__':
    opts = sys.argv[2:]
    sd = {}
    for o in opts:
        if o.startswith('--'):
            b = o.split('=')
            sd[b[0][2:]] = b[1]

    main(sys.argv[1], **sd)

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
