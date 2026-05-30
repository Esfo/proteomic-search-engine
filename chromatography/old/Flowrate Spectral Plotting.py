from pyteomics import mzml
import os
import numpy as np
import pandas as pd
import time
from scipy import signal
from matplotlib import pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
plt.rcParams["figure.dpi"] = 100

folder = '/home/sfo/base/MS_Files/FE/mzMLs/'
#folder = '/home/sfo/stuff/mzmls/'
#folder = '/store/FE/mzMLs/test/'
#file = 'C:/Base/MS_Files/FE/mzMLs/190701_F200_yM1.mzML'
#outfolder = 'C:/Base/data/FE/'
outfolder = '/home/sfo/stuff/mzmls/'
fn = '/home/sfo/stuff/mzmls/'

plotting = True
saving = False

n = 20
ppmwindow = 30
minmass = 400
maxmass = 600
massincreaserange = 100
massranges = np.linspace(minmass, maxmass, ((maxmass - minmass) // massincreaserange) + 1)

randomorderincrease = massincreaserange / n / 10
#randomorderincrease = 0
#inputholder = 20.3 #double peak between 9000,1000 that needs to be checked, n=1
inputholder = 0 #Constant offset in randomly generated masses

halfparamiters = 2

mindistance = 1
maxdistance = 1000
distances = np.unique(np.append(np.geomspace(mindistance, maxdistance, halfparamiters), \
np.linspace(mindistance, maxdistance, halfparamiters)))

minwidth = 1
maxwidth = 100
widths = np.unique(np.append(np.geomspace(minwidth, maxwidth, halfparamiters),\
np.linspace(minwidth, maxwidth, halfparamiters)))

# massstr = f'Mass +/-{ppmwindow}'
massstr = 'mass'
frame = pd.DataFrame(columns=[massstr, 'left half-base', 'right half-base', 'intensity', 'center', 'whm', 'n peaks'])

def tocsv(f, low, high, a, m ,d, w, peaks, peakinfo):
    frame = pd.DataFrame()
    frame.loc[:,'left bases'] = peakinfo['left_bases']
    frame.loc[:,'right bases'] = peakinfo['right_bases']
    frame.loc[:,'height'] = a[peaks]
    frame.loc[:,'scan'] = peaks
    frame.loc[:,'width at half height'] = peakinfo['widths']
    frame.loc[:,f'mass channel +/-{ppmwindow}'] = m
    fn = '.'.join(('_'.join((''.join((outfolder, f.split('.')[0])), str(int(low)), \
    str(int(high)), str(int(d)), str(int(w)))), 'csv'))
    frame.to_csv(fn, mode='a', index=False, header=False)

def scanfunc(scan, hunted):
    overlap = np.abs((scan['m/z array'] - hunted) / (hunted / 1000000))
    inds = np.where(overlap <= ppmwindow)
    return inds[0], scan['index'], scan['intensity array'], inds[1]

def avscanfunc(scan):
    return scan['index'], scan['intensity array'].mean()

for f in os.listdir(folder):
    if f.endswith('.mzML'):
        if saving:
            efn = ''.join((fn, f.split('.')[0], '.peaks.csv'))
            frame.to_csv(efn, index=False)
        st = time.time()
        #maxpeaks = 10000 #total number of masses I'd like to limit the analysis to
        #Just going to take the maxpeak of every scan - nah too noisy, and when I implemented 1000m/z as a minimum, the mean difference between every mass collected was 0.0011303946155721445, this thing catches every mass under the sun
        #Is there a way to look for the shortest lasting and highest intensity peaks?
        #Also want to quantify the noisiest signals to see which things are persisting through my spectra and how much interference they cause in a DDA -> Design an exclusion list around them.
        #subtracting off the median and anything just above, and below it should get rid of the baseline data, only the gaussian will remain - sure, but how to get rid of the 'just above' the median noise? I guess you could count the peak points themselves as outliers in some way
        #Medians seems to always be 0 pretty much, input a minimum points across a peak and then 
        #just take widths of these babies
        #And in measuring noisy channels, just take a number of times that any given channel has 
        #signal divided by the total number of scans - make a histogram here
        
        filename = ''.join((folder, f))
        print(filename)
        msrun = mzml.MzML(filename)
        nscans = len(msrun)
        mevals = np.zeros(nscans)
        for idx, av in msrun.map(lambda spec: avscanfunc(spec)):
            mevals[idx] += av

        scanarray = np.arange(nscans)
        mfilter = lowess(mevals, scanarray, is_sorted=True, frac=0.05, it=0)

        if plotting:
            plt.plot(scanarray, mevals, '-', alpha=0.5)
            plt.plot(mfilter[:,0], mfilter[:,1], '-')
            plt.title('Applied Mean Signal Filter')
            plt.show()

        frames = []
        for num in range(len(massranges)-1):
            low = massranges[num]
            high = massranges[num+1]
            hunted = (np.linspace(low, high, n) + np.random.uniform(0, \
            randomorderincrease,  size=n) + inputholder)[:,None]
            intensityarray = np.zeros(shape=(len(hunted), nscans))
            for i1, sci, scia, i2 in msrun.map(lambda spec: scanfunc(spec, hunted)):
                intensityarray[i1,sci] += scia[i2]
            
            tracker = 0
#        intensityarray = intensityarray[5][None,:]
            for ch in intensityarray:
#                y, x = np.histogram(ch, bins=100)
#                nc = 1

#                while True:
#                    ax[0].hlines(x[nc], xmin=0, xmax=nscans)
#                    ax[1].hlines(x[nc], xmin=0, xmax=nscans)
#                    nc += 1
#                    if nc > 5:
#                        break
                
                #Using the applied mean signal filter allows for the more intense (higher quality data) peaks to be recorded for that location. Lower intensity peaks for a location are likely to be ignored. Using the 95%th percentile baseline is going to allow for a baseline free of noise. The peak width at these levels should still be representative, why wouldn't they be? This also makes the modeling of lumps WAY easier. Now you can just count # of local maximums as the # of peaks in that width.
                #Use the width at half height in order to assess the heights, don't bother with baselines. It makes more sense to measure the width using the savitsky-golay filter because it's way more accurate, but model where the local maximums are via the lws filter

#                change = 30
#                rounds = 3
#                smooch = ch.copy()

#                for r in range(rounds):
#                    lowers = scanarray - change
#                    uppers = scanarray + change
#    
#                    lowers[lowers < 0] = 0
#                    uppers[uppers > scanarray.max()] = scanarray.max()
#    
#                    combs = np.linspace(lowers, uppers, change+1, axis=1).astype(np.int)
#                    smooch = smooch[np.r_[combs]].mean(axis=1)
#    
#                    change *= r + 1
#                    
#                ax[0].plot(scanarray, smooch, '-', alpha=0.5)
#    
#                ax[1].plot(scanarray, smooch, '-', alpha=0.5)


                savgol = signal.savgol_filter(ch, window_length=31, polyorder=2, mode='mirror',  deriv=0)
                savgol[savgol < 0] = 0

                frac = 0.005
                
                lws = lowess(ch, scanarray, is_sorted=True, frac=frac, it=0)

                initialmaxes = signal.argrelextrema(lws[:,1], np.greater)[0]
                # initialmaxes = initialmaxes[np.logical_and(initialmaxes > 400, initialmaxes < 15000)]
#                initialmaxes = initialmaxes[initialmaxes > 400]
                maxes = initialmaxes[ch[initialmaxes] >= mfilter[:,1][initialmaxes]]
#                sgmaxes = signal.argrelextrema(savgol, np.greater)
                
                if maxes.size > 0:
                    #This is the peak width, at half-max, machine ~
                    movingidx = np.ones(shape=(len(maxes),2)).astype(np.int)
                    movingidx[:,0] *= -1

                    halfmax = np.ceil(savgol[maxes] / 2)
                    hmidx = movingidx + maxes[:,None]

                    while True:
                        idx = savgol[hmidx] < halfmax[:,None]
                        movingidx[idx] = 0
                        hmidx += movingidx
                        
                        removers = np.logical_or(np.any(hmidx == len(savgol), axis=1), np.any(hmidx == 0, axis=1))
                        if removers.sum() > 0:
                            hmidx = hmidx[~removers]
                            halfmax = halfmax[~removers]
                            movingidx = movingidx[~removers]

                        if hmidx.size < 1:
                            break
                        
                        if np.all(movingidx == 0):
                            break
                    #peak width at half-max machine ends here ~
                    if hmidx.size > 0:
                        if plotting:
                            fig, ax = plt.subplots(2,1, figsize=(6.4,9.6))
#                ch = ch / ch.max()

                            ax[0].plot(scanarray, ch, '-', alpha=0.4, label='Raw Data')
                            ax[1].plot(scanarray, ch, '-', alpha=0.4, label='Raw Data')

                            xmin = 2000
                            xmax = 5000

                            ax[1].set_xlim(xmin, xmax)
                            ax[1].set_ylim(0, ch[xmin:xmax].max())

                            ax[0].vlines(scanarray[maxes], ymin=ch.min(), ymax=savgol[maxes], alpha=1)
                            ax[1].vlines(scanarray[maxes], ymin=ch.min(), ymax=savgol[maxes], alpha=1)

                            ax[0].plot(scanarray, lws[:,1], '-', label='LOWESS', alpha=0.5)
                            ax[1].plot(scanarray, lws[:,1], '-', label='LOWESS', alpha=0.5)

                            ax[0].plot(scanarray, savgol, label='Savitsky-Golay', alpha=0.5)
                            ax[1].plot(scanarray, savgol, label='Savitsky-Golay', alpha=0.5)
                            
                            hmplotvals = savgol[hmidx[np.arange(len(hmidx)),lws[:,1][hmidx].argmin(axis=1)]]

                            ax[0].hlines(hmplotvals, xmin=hmidx[:,0], xmax=hmidx[:,1])
                            ax[1].hlines(hmplotvals, xmin=hmidx[:,0], xmax=hmidx[:,1])

                            plt.legend()
                            plt.show()

                        if saving:
                            hmwidths = np.diff(hmidx)
                            im = np.repeat(initialmaxes, len(hmidx), axis=0)
                            ninternalpeaks = np.logical_and(hmidx[:,0][:,None] < im, hmidx[:,1][:,None] > im).sum(axis=1)

                            overlaps = np.logical_and(hmidx[:,1] >= hmidx[:,None][:,:,0], hmidx[:,0] <= hmidx[:,None][:,:,0])

                            for row in overlaps:
                                hmidx[row,0] = hmidx[row].min()
                                hmidx[row,1] = hmidx[row].max()

                            hmidx = np.unique(hmidx, axis=0)
                            pi = np.arange(len(intensityarray))[np.all(intensityarray == ch, axis=1)]
                            print('Mass:', pi, '-', hunted[pi])
                            
                            tf = frame.copy()
                            tf.loc[:,'left half-base'] = hmidx[:,0]
                            tf.loc[:,'right half-base'] = hmidx[:,1]

                            rm = np.repeat(maxes[None,:], len(hmidx), axis=0)
                            boundarymatrix = np.logical_and(hmidx[:,0][:,None] < rm, hmidx[:,1][:,None] > rm)
                            
                            peaks = []
                            intensities = []
                            for b in boundarymatrix:
                                peaks.append(maxes[b].tolist())
                                intensities.append(savgol[maxes][b].tolist())


                            tf.loc[:,'whm'] = np.diff(hmidx)
                            tf.loc[:,'n peaks'] = [len(u) for u in peaks]
                            tf.loc[:,massstr] = hunted[tracker]
                            
                            for ro, ind in tf.iterrows():
                                tf.loc[ro,'intensity'] = ', '.join(([str(round(k, 2)) for k in intensities[ro]]))
                                tf.loc[ro,'center'] = ', '.join(([str(round(k, 2)) for k in peaks[ro]]))
                            
                            
                            tf.to_csv(efn, mode='a', header=False, index=False)

                tracker += 1
#        for a, m in zip(intensityarray, hunted):
#            for d in distances:
#                for w in widths:
#            
#                    peaks, peakinfo = signal.find_peaks(a, width=30, 
#            rel_height=0.5, distance=500)
#                    if peaks.size > 0:
#                        tocsv(f, low, high, a, m, d, w, peaks, \
#                        peakinfo)
            msrun.reset()
            print(f'{low}-{high}: {time.time() - st:2f}')
#    outframe = pd.concat(frames)
#    outframe.to_csv(''.join((outfolder, filename.split('/')[-1].split('.')[0], '_processed.csv')))
        msrun.close()
#    if not os.path.exists(''.join((folder, r'Holder'))):
#        os.makedirs(''.join((folder, r'Holder')))
#    os.rename(file,''.join(('/'.join((file.split('/')[:-1])), '/Holder/', file.split('/')[-1])))
