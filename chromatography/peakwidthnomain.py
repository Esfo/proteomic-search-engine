import numpy as np
import matplotlib.pyplot as plt
from pyteomics import mzml
from time import time
import pandas as pd
import gc
import concurrent.futures
from scipy import sparse, signal, stats, integrate
from pandas.api.types import CategoricalDtype
from statsmodels.nonparametric.smoothers_lowess import lowess
import itertools
import sys
import os
plt.rcParams["figure.dpi"] = 300

#folder = '/store/flowcharacterizations/round2/MS1s/mzMLs/'
#mzmlfile = '/store/flowcharacterizations/round3/DDAs/mzMLs/200901_fR_300.mzML'
#mzmlfile = '/store/flowcharacterizations/round3/DDAs/mzMLs/200901_1s-dyn-300-200_B0.mzML'
mzmlfile = '/store/flowcharacterizations/round3/DDAs/mzMLs/200901_fR_400.mzML'
#mzmlfile = '/store/flowcharacterizations/round4/plus/mzMLs/20210102_fR_200-1.mzML'
#runorder = open('/store/flowcharacterizations/runorder')
#runorder = [i.strip() for i in runorder]
#runorder = [''.join((folder, '200612_', i, '.mzML')) for i in runorder]
#files = list(itertools.zip_longest(*[iter(runorder)]*2))
#files = os.listdir(folder)
#files = [i for i in files if i.endswith('.mzML.')

#from https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
#def find_nearest(array, value):
#    array = np.asarray(array)
#    idx = (np.abs(array - value)).argmin()
#    return array[idx], idx

#def scanread(scan, t):
#    et = pd.DataFrame(scan.peaks('centroided'))
#    et.columns = ['m/z', 'intensity']
#    et.loc[:,'m/z'] = et.loc[:,'m/z'].round(4)
#    et.loc[:,'index'] = scan.ID - 1
#    et.loc[:,'ms level'] = scan.ms_level
#    et.loc[:,'time (min)'] = scan.scan_time_in_minutes()
#    t.append(et)
#    return t

#need to build in a centroiding process for profile data, nothing seems to centroid my example profile data at all (wtf guys?), including msconvert
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

#def fileread(f):
#    t = mp.Manager().list()
#    msrun = pymzml.run.Reader(f)
#    pool = mp.Pool()
#    for scan in msrun:
#        pool.apply_async(scanread(scan, t))
#    pool.close()
#    pool.join()
#    return list(t)

#from https://codereview.stackexchange.com/questions/252321/counting-sequential-booleans
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

widthbuffer = 20
peakfill = 0.7
prominencefilter = 1

widthbuffer = int(widthbuffer) #assumes peaks are at least 10, or this n/2, scans wide, (no it doesn't) which ain't too bad. Could probably also derive this number from the minimum scans per channels determined above - although it's softcoded. It might be wiser to keep this one hardcoded.
peakfill = float(peakfill) #This filters out things that seem to give on and off signal for whatever reason. Although even on major peaks I do see a lot of on and off signal, for reasons I can't explain. If it happens too much, I can kick it out via this.


#
#Opening the file and extracting data
#

mt = time()
ef = []
msrun = mzml.MzML(mzmlfile)
for t in msrun.map(lambda scan: scanfunc(scan)):
    ef.extend(t)

gc.collect()
print(time() - mt, '- File Extracted')


#
#Using MS1 data
#

ef = pd.concat(ef)
scantimes = ef.loc[:,('index', 'time (min)')].drop_duplicates()
scantimes.set_index('index', inplace=True)
scantimes.sort_index(inplace=True)
ef = ef.loc[ef.loc[:,'ms level'] == 1]
#not all these formats seem to matter for speed, for m/z values though the resolution obviously won't match what's recorded. And it seems to ~half the number of rows in the sparse matrix.
#ef.loc[:,'m/z'] = ef.loc[:,'m/z'].round(4)
#ef.loc[:,'intensity'] = ef.loc[:,'intensity'].astype(int)
#ef.loc[:,'index'] = ef.loc[:,'index'].astype(np.int16)


#
#Organizing mass centers
#

#Taking the mean of the # of data points from all channels with more than 1 data point
#Any channel with less than the mean number of data points is removed.
#This process is not permanent, the original data is used later on. This part here dictates which data points will be used for an initial mass selection used to find chromatographic peaks. Ie finding centers of mass.
#et = ef.drop('intensity', axis=1)
et = ef.loc[:,('m/z', 'index')]
et = et.groupby('m/z').count()
et = et.loc[(et > et.loc[(et > 1).to_numpy()].mean()).to_numpy()]
#you can sanity check this with ((et > 1).any(axis=1) != (et > 1).all(axis=1)).sum(), it should be 0

en = ef.set_index('m/z')
en.sort_index(inplace=True)
en = en.loc[et.index.unique().array] #this index being unique values only is important else the frame will be larger than ef through pandas tomfoolery

nbins = np.sqrt(en.index.unique().size).round().astype(int)
ediff = pd.DataFrame()
ediff.loc[:,'max'] = en.loc[:,'intensity'].max(level=0)
ediff.loc[:,'sum'] = en.loc[:,'intensity'].sum(level=0)
ediff.loc[:,'diff'] = ediff.loc[:,'sum'] - ediff.loc[:,'max']

ediff.loc[:,'pdiff'] = ediff.loc[:,'diff'] / ediff.loc[:,'sum']
ediff.loc[:,'pdbins'] = pd.cut(ediff.loc[:,'pdiff'], bins=nbins, labels=False)

maxfreqloc = ediff.groupby('pdbins').count().sort_values('max').index.array[-1] #in case the mode function returns a list or whatever
#maxfreqloc = ediff.loc[:,'pdbins'].mode().to_numpy()[0]
lastbin = nbins - 1
filterloc = maxfreqloc - (lastbin - maxfreqloc) #assumes a ~symmetrical rightward distribution, even though this is always iffy on the symmetry, hanging more towards 1 than 0, seems to always happen!

#ediff.loc[:,'dbins'] = pd.cut(ediff.loc[:,'diff'], bins=nbins, labels=False)
#ediff.loc[:,'dbins'].plot.hist(bins=nbins)
#plt.show()

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

#ediff.loc[:,'pdbins'].plot.hist(bins=nbins)
#plt.yscale('log')

#EXPLANATION OF ABOVE
#If a mass channel's max is close to it's sum, then there's not many other data points on that mass channel - meaning this could be off-target, in terms of center of mass. It could still be within the ppm window, perhaps useful later. But for now, we're finding the center of mass.
#If a mass channels's max is far away from it's sum, then it's clear there's many other data points - this could be a good starting indicator for a center of mass. These would be good data points to keep to find the true center of mass. This allows for an initially useful filtering of peripheral points

#***CHECK the lower part of these ^ to visualize what kind of peaks exist
#Open question: How many of these lower bins have peaks? How many peaks in the lower bins? Ie, how many peaks are in, what's perhaps', really clean channels?
#Reverse the filter and see what you find!


#
#Using filtered data to find mass centers
#

potentialmasscenters = ediff.loc[ediff.loc[:,'pdbins'] >= filterloc].index.array
ex = en.loc[potentialmasscenters]
es = pd.DataFrame()
es.loc[:,'intensity'] = ex.loc[:,'intensity'].sum(level=0) #summed intensity along chromatographic time

#to see % of channels filtered out:
#(en.size - ex.size) / en.size #not that many

#I need to make this a weighted average to verify if the plots I'm spitting out down below make any sense or not.
#Even upon plotting the weighted mean of the index based on the intensities, the [1,0] plots of m/z vs. index mean still don't make any sense, visually, for a lot of the plotted examples.
#es.loc[:,'index wmean'] = (ex.loc[:,'index'] * ex.loc[:,'intensity']).sum(level=0) / ex.loc[:,'intensity'].sum(level=0)
#es.loc[:,'index mean'] = ex.loc[:,'index'].mean(level=0)
#es.loc[:,'index std'] = ex.loc[:,'index'].std(level=0)

firstderivmaxes = signal.argrelextrema(es.loc[:,'intensity'].to_numpy(), np.greater)[0]

#In the short, biased look that I took. Using the second derivative gave me less mass channels, and more peak-filled channels. Without the 2nd deriv here, the mass channels have much narrower +- ppm windows, and there are less peaks in each plotted spectra. The difference between adjacent mass channels, in mass, seems to be smaller. I find this acceptable because it seems to make for a less complicated problem to iterate over later. Using only the first derivative also seems to make my generated minimum ppm window and max ppm bridge (both below) have smaller windows.
#secondderivmaxes = signal.argrelextrema(es.loc[:,'intensity'].to_numpy()[firstderivmaxes], np.greater)[0]
#thirdderivmaxes = signal.argrelextrema(en.to_numpy()[firstderivmaxes][secondderivmaxes], np.greater)[0]

#Using the 2 rounds of local maximums here should show where the maximum of a jagged peak is. It seems like a decent assumption to assume the center-of-mass peaks aren't perfectly gaussian, there's more room for stochasticity to cause it to be otherwise.

#to visualize summed chromatographic intensity across potentialmasschannels and their selected maxes
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


#
#Using the mean difference across ordered mass centers to assemble reasonable +/- ppm windows around the selected mass centers
#

fdmasses = es.index.to_numpy()[firstderivmaxes]
ppmdist = np.diff(fdmasses) / fdmasses[:-1] * 1000000

#maxppmbridge should be a little bit lower than minppmwind? to minimize the number of max points being connected. And I'd rather there be some signal overlap rather than signal spread across too many centers due to a expanded connections. Does this make sense? It's hard to come to a good line of reasoning here as to which should be higher, I don't believe any argument for the reverse to be as straightforward as it seems.
#I seem to get fine results when they're the same thing, it also seems rather acceptable that both the minimum window and maximum bridge size are the same value. I cannot think of better reasoning as to why one should be larger or smaller than the other.
maxppmbridge = ppmdist.mean()
minppmwind = ppmdist.mean()


#
#Combining mass centers that overlap across each other's +/- ppm windows into mass channels to be used throughout the rest of our process
#

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
#mcppmwindows = np.abs((np.diff(feout)).reshape(1,-1) / feout[:,1] * 1000000)[0] / 2 #should use the actual mass channels instead of feout[:,1]
mcppmwindows = np.abs((np.diff(feout)).reshape(1,-1) / masschannels * 1000000)[0] / 2
mcppmwindows[mcppmwindows < minppmwind] = minppmwind #too much might be lost if this is too stringent I don't want to have actual ppm windows of 0
gc.collect()

#these are the bounds for the existing channels, these are not the bounds of the peaks to be found later.
print(f'Count: {len(mcppmwindows)}')
print('Mass Windows (ppm)')
print(f'Min: {mcppmwindows.min()}')
print(f'Max: {mcppmwindows.max()}')
print(f'Mean: {mcppmwindows.mean()}')
print(f'Median: {np.median(mcppmwindows)}')
print(time() - mt)


#
#Arranging original data from ef into array format
#

newcol = 'index'
newrow = 'm/z'
values = 'intensity'

#this doesn't necessarily need to be done en-masse, and certainly fucks with everything once profile data comes into play, these could potentially be made on the fly for masses that fit a window
newcols = CategoricalDtype(sorted(ef.loc[:,newcol].unique()), ordered=True)
newinds = CategoricalDtype(sorted(ef.loc[:,newrow].unique()), ordered=True)
col = ef.loc[:,newcol].astype(newcols).cat.codes
row = ef.loc[:,newrow].astype(newinds).cat.codes
sm = sparse.csc_matrix((ef.loc[:,values], (row, col)), shape=(newinds.categories.size, newcols.categories.size))

print(time() - mt, 'Arrays assembled') #a sometimes confusing message
print('Assessing', sm.shape[0], 'channels for', len(masschannels), 'centers')


#
#Merging arrays into the chosen mass channels
#

mzindex = newinds.categories.to_numpy()

#non-function format, concurrency helps this a lot
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
with concurrent.futures.ThreadPoolExecutor(8) as executor:
    for mc, mw in zip(masschannels, mcppmwindows):
        fut.append(executor.submit(channel_aggregation, mc, mw, mzindex, sm))
    for f in concurrent.futures.as_completed(fut):
        sa, mza = f.result()
        mzinwin.append(mza)
        sms.append(sa)
print(time() - mt)

sms = np.asarray(np.vstack((sms)))
mzinwin = np.asarray(mzinwin)

sms = sms[mzinwin[:,2].argsort()] #array of mass intensities by mass channel across chromatographic time
mzinwin = mzinwin[mzinwin[:,2].argsort()] #Original mass intensity indice range of sm now encompassed in sms.

masschannels = mzinwin[:,2] #mass centers
mcppmwindows = mzinwin[:,3] #masschannel +/- ppm windows
mzinwin = mzinwin[:,:2].astype(int)
gc.collect()

cinds = (sms > 0).sum(axis=1) > widthbuffer #channels that don't have even the required amount of datapoints between maxes
#sms = sparse.csc_matrix(sms)

if cinds.sum() != len(sms):
    sms = sms[cinds]
    masschannels = masschannels[cinds]
    mcppmwindows = mcppmwindows[cinds]
    mzinwin = mzinwin[cinds]

medinds = np.median(sms, axis=1) == 0 #not all of these look bad, but this is a great way to determine noisy channels. Only ~120 of them were above 0 median on a given file tested. You could see if any PSMs match up with these channels, it's possible
#This would probably knock out a ton of noise if turned into an exclusion list.

if medinds.sum() != len(sms):
    sms = sms[medinds]
    masschannels = masschannels[medinds]
    mcppmwindows = mcppmwindows[medinds]
    mzinwin = mzinwin[medinds]

scanarray = np.sort(ef.loc[:,'index'].unique())

#it takes too much memory to perform transforms on the entire mass intensity array, so it's done on the fly

#Correlation testing for chosen mass channels. A problem that occurred during initial design was that certain mass channels looked a lot like their neighboring channels. Which essentialy means there's undesirable data leakage. Methods have since been improved, but you can visualize how often a channel correlates with its neighbors below:
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


#
#Peak finding in individual mass channels
#

#use d = 5533 for testing, I guess
#d = 0 shows a good error where a single data point is transformed into a peak, this should be filtered out, due to there being no adjacency with this data.
#d = 479 for an example of the signal transforms failing to find a good peak base width, the need for corrections is there. 756?
# Without a direct example, ^to find large failures, find channels with small % of areas covered by found peaks
enfs = []
for d in range(len(sms)):
    channel = sms[d]
    mzin = mzinwin[d]
    ochan = sm[mzin[0]:mzin[1]].todense()
    omasses = mzindex[mzin[0]:mzin[1]]

    #the savgol variable below provides an easier means of peak width estimation. The savgol2 variable would give a more accurate AUC measurement.
    savgol = signal.savgol_filter(channel, window_length=101, polyorder=2, mode='nearest', deriv=0)
    savgol2 = signal.savgol_filter(channel, window_length=21, polyorder=0, mode='nearest', deriv=0)

    #these are smoothed derivatives
    sgrad = np.gradient(savgol) #not including the x-spacing for this function seems to improve output for this use-case
    sgd = signal.savgol_filter(sgrad, window_length=61, polyorder=2, deriv=1, mode='wrap')
    sgdq = signal.savgol_filter(savgol, window_length=251, polyorder=2, deriv=2, mode='wrap')
    sdf = signal.argrelmax(sgd)[0]
    sdfq = signal.argrelmax(sgdq)[0]
    
    maxis = signal.argrelmax(savgol)[0]

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


#newfolder = '/'.join(('/'.join((mzmlfile.split('/')[:-2])), 'peaks/'))
#newfile = ''.join((mzmlfile.split('/')[-1].split('.')[0], '.peaks.csv'))
#endfn = ''.join((newfolder, newfile))
#enfs.to_csv(endfn, index=False)


#This helps us check for signal overlap across channels. We don't want to see remnants of the same peak across two of our mass channels, that would give us the same data twice.

top = 10
channelframe = enfs.groupby('mass channel').count().sort_values('raw heights', ascending=False).loc[:,'raw heights']
channels = channelframe.index.to_numpy()
chans = channels[:top]
targetchannels = enfs.set_index('mass channel').loc[chans,'+- ppm'].mean(level=0)
targetchannels = targetchannels.to_frame()

targetchannels.loc[:,'lower limit'] = targetchannels.index - (targetchannels.index / 1000000) * targetchannels.loc[:,'+- ppm']
targetchannels.loc[:,'upper limit'] = targetchannels.index + (targetchannels.index / 1000000) * targetchannels.loc[:,'+- ppm']

selectedmass = targetchannels.index[0]

binds = enfs.loc[:,'mass channel'] == selectedmass
bc = enfs.loc[binds]

point = find_nearest(masschannels, selectedmass)
start = point[1]
nchans = 1
channels = [start]

l = 0
h = sms.shape[1]


p = targetchannels.index[0]
pw = np.argwhere(masschannels == p)[0][0]
tf = enfs.loc[enfs.loc[:,'mass channel'] == p]
ti = tf.loc[:,'r area'].argmax()
tfl = np.argwhere(scanarray == tf.loc[ti, 'left base point'])[0][0]
tfr = np.argwhere(scanarray == tf.loc[ti, 'right base point'])[0][0]

aind = 0

plt.plot(mzindex[mzinwin[d][0]-aind:mzinwin[d][1]+aind], sm[mzinwin[d][0]-aind:mzinwin[d][1]+aind,l:r].todense().sum(axis=1), '.')
plt.yscale('log')
plt.show()

plt.plot(scanarray[l:r], np.asarray(sm[mzinwin[d][0]-aind:mzinwin[d][1]+aind,l:r].todense().sum(axis=0))[0], '.')
plt.show()


c = channels[0]
#for c in channels:
f, a = plt.subplots(figsize=(16,8))
n = masschannels[c]
w = n * (mcppmwindows[c] / 1000000)
channel = sms[c]

#xl1 = scanarray[l] - 200
#xl2 = scanarray[r] + 200
xl1 = 10700
xl2 = 11600
yl = channel[find_nearest(scanarray, xl1)[1]:find_nearest(scanarray, xl2)[1]].max()

savgol = signal.savgol_filter(channel, window_length=101, polyorder=2, mode='wrap', deriv=0)
savgol2 = signal.savgol_filter(channel, window_length=21, polyorder=0, mode='wrap', deriv=0)
sgrad = np.gradient(savgol) #not including the x-spacing for this function is alright here because the products of this variable are purely for peak-finding, it would only change scaling.
maxis = signal.argrelmax(savgol)[0]
sgd = signal.savgol_filter(sgrad, window_length=61, polyorder=2, deriv=1, mode='wrap')
sgdq = signal.savgol_filter(savgol, window_length=251, polyorder=2, deriv=2, mode='wrap')
sdf = signal.argrelmax(sgdq)[0]

finds = np.logical_and(ef.loc[:,'m/z'] > n-w, ef.loc[:,'m/z'] < n+w)
pinds = np.logical_and(es.index > n-w, es.index < n+w)
fpi = np.logical_and(es.index[firstderivmaxes] > n-w, es.index[firstderivmaxes] < n+w)
spi = np.logical_and(es.index[firstderivmaxes] > n-w, es.index[firstderivmaxes] < n+w)

fp = ef.loc[finds]
fp = fp.groupby('m/z').agg({'intensity': 'sum', 'index': 'mean'})

etp  = es.loc[pinds]
efd = es.index[firstderivmaxes][fpi]
esd = es.index[firstderivmaxes][spi]

f, a = plt.subplots(figsize=(16,10))
a.plot(scanarray, channel, '.', color='orange', alpha=0.9)
a.plot(scanarray, savgol, '-', color='green', alpha=0.5)
a.vlines(scanarray[sm], 0, channel.max(), color='black')
a.vlines(scanarray[sl], 0, channel.max()/2, color='blue', alpha=0.6)
a.vlines(scanarray[sr], 0, channel.max()/2, color='gold', alpha=0.6)
#plt.xlim(scanarray.min(), scanarray.max())
#plt.ylim(0,1e6)
plt.show()

a.plot(scanarray, savgol2, '-', color='brown', alpha=0.5)
a.plot(scanarray, sgd*8000, '-', color='pink', alpha=0.9)
a.plot(scanarray, sgdq*8000, '-', color='purple', alpha=0.9)

##a.vlines(bc.loc[:,'retention time (scan)'], yl, 0, color='black')
#a.vlines(scanarray[m], yl, 0, color='black')
##a.vlines(scanarray[sdf], yl, 0, color='cyan')

#a.vlines([scanarray[r], scanarray[l]], yl, 0, color='green')
a.set_xlim(xl1, xl2)
a.set_ylim(0, yl)
plt.title(''.join((str(c), ': ', str(masschannels[c]), ' +/-', str(mcppmwindows[c].round(4)), 'ppm')))
plt.show()


#current plan:
#kernel density estimate on distribution of slopes, scipy's or sklearns with gridearch
#^Test slopes of each fitting and the raw data, whichever is best works
#get rid of the [-1] point of scanarray/channel in a new variable
#scatter plot with the channel data and use density as the color, any good way to identify peak boundaries?

#When merging channels:
#Get mean mass of channels for later description, to determine the ppm from the center on either side
#Also get median mass so you can LATER REINDEX IT when doing peak calculations to get the PEAKS ACTUAL MASS, and determine number of indices away from the median on either side is used to merge this channel

#pf = pd.DataFrame()
#pf.loc[:,'scanarray'] = scanarray[:-1]
#pf.loc[:,'channel'] = savgol[:-1]
#slopes = np.diff(savgol)/np.diff(scanarray)
#pf.loc[:,'aslopes'] = np.abs(slopes)
#pf.loc[:,'slopes'] = slopes
#pf.loc[:,'scs'] = pf.loc[:,'slopes'].cumsum()
#I like this idea of this, but I'm not sure at the moment, how to get this to work



#verification of ~how many peaks were picked up compared to the actual TIC. How can I get a measurement out of this? <-- An AUC!!!! Scanarray on X, each true/found set on Y
ssum = sms.sum(axis=0)
tsum = []
for scan in msrun:
    if scan.ms_level == 1:
        tsum.append(scan.peaks('centroided')[:,1].sum())

plt.plot(tsum, ssum, '.')
plt.show()

#For AUC! This doesn't give AUC of found peaks to chromatogram, it gives AUC of found mass channels.
plt.plot(scanarray, tsum, '.', color='purple', alpha=0.2)
plt.plot(scanarray, ssum, '.', color='green', alpha=0.2)
plt.show()

ti = integrate.cumtrapz(tsum)
si = integrate.cumtrapz(ssum)
perca = si / ti
plt.hist(perca, bins=100)
plt.show()


#for historic preservation:
def peak_accumulation(lo, ro, sm):
    tosdict = {}
    tosdict['retention time (scan index)'] = sm
    n += 1
    #ls = np.array([lo, lq])
    #rs = np.array([ro, rq])
    #rwmin = savgol2[rs].min()
    #lwmin = savgol2[ls].min()
    #rwmin = savgol[ro]
    #lwmin = savgol[lo]
    #rwa = savgol2[rs].argmin()
    #lwa = savgol2[ls].argmin()

    #Below was checking to see if all the l and r points were to the proper side of m. However, this was fixed above in a different manner. The second part of below was determining feasible limits for finding the l and r values, but this did not offer any tangible performance increase, and subsequently, searching across the whole channel didn't impact performance.
    #roundpass = True
    #if lo < m & ro > m != lq < m & rq > m:
    #    #sanity check on positioning
    #    #below problem is fixed
    #    #checking to see that positioning is right: the current setup doesn't remove certain bad indices if only one of them is a nan (from pb), some values come back as 99999 b/c of numpy masked array default fillvalue, I've changed these to -1. If either set has a bad point, the other set is chosen by default. see d = 119 in '/store/flowcharacterizations/round3/DDAs/mzMLs/200901_1s-dyn-300-250_B0.mzML' for an example, the first one in lqpoints triggers this.
    #    if lo < m & ro > m:
    #        rdist = scanarray[ro] - scanarray[m]
    #        ldist = scanarray[m] - scanarray[lo]
    #        rwmin = savgol[ro]
    #        lwmin = savgol[lo]
    #    elif lq < m & rq > m:
    #        rdist = scanarray[rq] - scanarray[m]
    #        ldist = scanarray[m] - scanarray[lq]
    #        rwmin = savgol[rq]
    #        lwmin = savgol[lq]
    #    else:
    #        #this one shouldn't be needed, they should be removed by pb if this occurs.
    #        roundpass = False
    #    rlimit = find_nearest(scanarray, scanarray[m] + 2*rdist)[1]
    #    llimit = find_nearest(scanarray, scanarray[m] - 2*ldist)[1]
    #else:
    #    #ro < rq, lo > lq - this is the expectation, this should be true
    #    rwmin = savgol[rs].min()
    #    lwmin = savgol[ls].min()
    #    if ro < rq:
    #        rdist = scanarray[rq] - scanarray[m]
    #        if savgol[rs].argmin() == 0:
    #            rlimit = rq
    #        elif savgol[rs].argmin() == 1:
    #            rlimit = find_nearest(scanarray, scanarray[m] + 2*rdist)[1]
    #    elif ro > rq:
    #        rdist = scanarray[ro] - scanarray[m]
    #        if savgol[rs].argmin() == 0:
    #            rlimit = find_nearest(scanarray, scanarray[m] + 2*rdist)[1]
    #        elif savgol[rs].argmin() == 1:
    #            rlimit = ro
    #    elif ro == rq:
    #        rdist = scanarray[m] - scanarray[ro]
    #        rlimit = find_nearest(scanarray, scanarray[m] - 2*rdist)[1]
    #
    #    if lo > lq:
    #        ldist = scanarray[m] - scanarray[lq]
    #        if savgol[ls].argmin() == 0:
    #            llimit = lq
    #        elif savgol[ls].argmin() == 1:
    #            llimit = find_nearest(scanarray, scanarray[m] - 2*ldist)[1]
    #    elif lo < lq:
    #        ldist = scanarray[m] - scanarray[lo]
    #        if savgol[ls].argmin() == 0:
    #            llimit = find_nearest(scanarray, scanarray[m] - 2*ldist)[1]
    #        elif savgol[ls].argmin() == 1:
    #            llimit = lo
    #    elif lo == lq:
    #        ldist = scanarray[m] - scanarray[lo]
    #        llimit = find_nearest(scanarray, scanarray[m] - 2*ldist)[1]

    #There is no good single transform to finding perect peak boundaries: Depending on whether the peak is massive, or is in a noisy background with other peaks - or both, these transforms can be different. So it makes more sense to use both a looser(lq/rq) and tighter(lo/ro) transform as a ranger for calibrating where to determine true peak width boundaries.
    #test 1: which is lower, pink or purple line boundaries
    #test 2: If the lower one is closer to the top, find the lowest point between the outer boundary and the top
    #alt test 2: If the lower one is farther from the top, check from the top to 2x the distance to find the lowest peak that doesn't go through any new peaks
    #do less than or equal to, or else you could actually get an increased value.
    #after all this get that > half-max value for peak width assessment working too


    #The savgols and smoothed derivatives seem to give reliable underestimates of width, so I should only need to make a simple machine that searches outward for a lower point, assuming there's always an underestimation of base width
    #^Nope, it can overestimate in the case of very small peaks that aren't prominent from a dominating neighbour peak
    #ldist = scanarray[m] - scanarray[l]
    #rdist = scanarray[r] - scanarray[m]
    #
    #llimit = find_nearest(scanarray, scanarray[m] - 2*ldist)[1]
    #rlimit = find_nearest(scanarray, scanarray[m] + 2*rdist)[1]

    #ldl = np.ceil(len(savgol[llimit:l]) / (widthbuffer  / 2)) * 10 - len(savgol[llimit:l])
    #rdl = np.ceil(len(savgol[r+1:rlimit+1]) / (widthbuffer  / 2)) * 10 - len(savgol[r+1:rlimit+1])
    #llimit -= int(ldl)
    #rlimit += int(rdl)

    #old filter for above
    #if roundpass:
    #if llimit < 0:
    #    llimit = 0
    #if rlimit > len(scanarray):
    #    rlimit = len(scanarray) - 1
    #
    #ri = np.r_[m+1:rlimit+1]
    #li = np.r_[llimit:m]
    #
    #rloc = savgol[ri]
    #lloc = savgol[li]
    #rloc = channel[ri]
    #lloc = channel[li]

    #rlinds = rloc <= rwmin
    #llinds = lloc <= lwmin
    #rlinds = savgol[m+1:] <= rwmin
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
    #rbool = rpile[1:, 1] <= widthbuffer
    rbool = rpc[rint:, 1] >= widthbuffer
    #rbool = np.logical_and(rpile[1:, 1] <= widthbuffer, rpile[:-1,0] > 0)
    #lbool = lpile[:-1, 1] <= widthbuffer
    lbool = np.flip(lpc, axis=0)[lint:, 1] >= widthbuffer
    #riv = 2 + boolcount(rbool)[0,0] if rbool.size > 0 else 1
    #liv = 1 + boolcount(lbool)[-1,0] if lbool.size > 0 else 1
    #bcr = boolcount(rbool[1:]) if rbool.size > 0 else 1
    #bcl = boolcount(lbool) if lbool.size > 0 else 1
    #rbu = rbool[bcr[0].sum():]
    #if rbu.size > 0 and rbu.cumsum().max() > 0:
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

    #r = ri[rloc[:rzone].argmin()]
    #l = len(lloc) - lzone +li[lloc[-lzone:].argmin()]
    #taking min from left can select first occuring minimum value if the min value repeats. Not a problem for the right points.
    lmin = savgol[len(lbase) - lzone:sm].min()
    lms = np.argwhere(savgol[len(lbase) - lzone:sm] == lmin).max()
    sl = len(lbase) - lzone + lms
    sr = len(savgol) - len(rbase) + savgol[sm:sm+rzone].argmin()

    #I could potentially actually have this here, and add things to a dataframe as I go. What's of current importance is the replacement for the prominence filter, need to assess signal uniformity across the l-r boundaries, and I suppose the highest average should be around m. But how to do this?
    #definitely make this into a thing, filter by p-value: It should work okday for general correlations, AND it can be powered by # of data points, helping to leave out those little shit peaks. Plot p-values by total width, and across retention time (are they mostly flush peaks?)
    s2m = sl + savgol2[sl:sr].argmax()
    rm = sl + channel[sl:sr].argmax()

    s2r = sr - trimmer(np.flip(savgol2[sm:sr]))
    s2l = sl + trimmer(savgol2[sl:sm])

    rr = sr - trimmer(np.flip(channel[sm:sr]))
    rl = sl + trimmer(channel[sl:sm])

    shh = scanarray[sl:sr][savgol[sl:sr] > savgol[sm] / 2]
    s2hh = scanarray[s2l:s2r][savgol2[s2l:s2r] > savgol2[s2m] / 2]
    rhh = scanarray[rl:rr][channel[rl:rr] > channel[rm] / 2]

    tosdict['s width at half-max left boundary (scan)'] = shh.min()
    tosdict['s width at half-max right boundary (scan)'] = shh.max()

    tosdict['s2 width at half-max left boundary (scan)'] = s2hh.min()
    tosdict['s2 width at half-max right boundary (scan)'] = s2hh.max()

    tosdict['r width at half-max left boundary (scan)'] = rhh.min()
    tosdict['r width at half-max right boundary (scan)'] = rhh.max()

    #RECORD SCRATCH
    #How well do the integrated areas of peaks under fitting vs raw data match each other?? This could be my legitemacy filter.
    #you can look at the distribution of % differences.
    #perhaps you could do this on top of just a p-value filter?
    #how would this %diff area distribution compare to the R^2 distribution? Scatter plot and histogram comparison pls.
    #this would be the area that doesn't subtract the bottom part.. this might fail for that trailing peak that broke the algo? You could also dice the peak into n slices and compare area like that (while also getting your modulatable quadrants)
    #^p-value worked for it though. How would the distribution of baseline values look for that instead? Maybe this could be a legitemate filter.


    #another thought, how does the above boundary-finding bit work for the s2 and raw data? Can I just do this for both of them?

    #when trimming bottom area, you'll want to also collect 2 different type of width at half-maxes, half-max of the apex - baseline, then half-max of apex if it's available. There needs to be a test if the baseline is over the half-max of the apex.
    
    
    
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

    #things that can be done with the entire dataframe:
        #width, and half-max width in both scan and time - only list indices here
        #half-max height - but it might hinder other things, you can get the half-max baseline and subtract area below it to get area above half-max if you calculate this in the loop
    #things to add in this loop:
        #area and baseline estimates
    #endcall filter, do another round of overlap testing here?
    tosdict['processed index'] = d
    #tos.loc[:, 'right boundary (scan index)'] - tos.loc[:,'left boundary (scan index)'] <= widthbuffer
    nfs = pd.DataFrame.from_dict(tosdict, orient='index')
    enfs.append(nfs)

