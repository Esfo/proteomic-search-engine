import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import os
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
plt.rcParams["figure.dpi"] = 100
warnings.filterwarnings("ignore")

rf = '/store/flowcharacterizations/round3/DDAs/results.csv'
df = pd.read_csv(rf)
df.loc[:,'file'] = df.loc[:,'file'].apply(lambda x: x.split('.')[0].split('_')[-1])
df.set_index('file', inplace=True)

basefolder = '/store/flowcharacterizations/round3/DDAs/'
#basefolder = '/store/FE/'

peakfolder = ''.join((basefolder, 'peaks/'))
ticfolder = ''.join((basefolder, 'TICs/'))
files = os.listdir(peakfolder)
files = [i for i in files if i.endswith('.peaks.test.csv')]
#files = [i for i in files if 'stat' in i]

#files = [i for i in files if 'FdV' not in i]
#
#filesort = [int(i.split('_')[1].split('F')[1]) for i in files]
#filesort = np.asarray(filesort)
#
#files = np.asarray(files)
#files = files[filesort.argsort()].tolist()
#files = files[0::5]

plottervals = ['trimmed base widths (min)']


for fr in files:
#    fr = files[-4]
    f = ''.join((peakfolder, fr))
    #f = '/store/flowcharacterizations/round2/DDAs/peaks/200724_simdec-stat-70-DDA.peaks.csv'
    fbase = fr.split('.')[0]
    enfs = pd.read_csv(f, low_memory=False)

    ticfile = ''.join((ticfolder, fbase, '.tic.csv'))
    of = pd.read_csv(ticfile, low_memory=False)
    of.sort_values('scan', inplace=True)

    nrows = len(plottervals) * 2 + 1
    fig, ax = plt.subplots(nrows=nrows, figsize=(10,3*nrows), constrained_layout=True, sharex=True)
    axval = 1

    of.plot.line(x='scan', y='summed intensity', color='purple', ax=ax[0])
    of.set_index('scan', inplace=True)
    enfs.loc[:,'Time'] = of.loc[enfs.loc[:,'retention time (scan)'], 'Time'].tolist()

    for pv in plottervals:
        enfs.loc[:,'frequency'] = stats.gaussian_kde(enfs.loc[:,('retention time (scan)', pv)].to_numpy().transpose())(enfs.loc[:,('retention time (scan)', pv)].to_numpy().transpose())

        enfs.sort_values(by='frequency').plot.scatter(x='retention time (scan)', y=pv, c='frequency', colormap='winter', ax=ax[axval], alpha=0.5, norm=matplotlib.colors.LogNorm())

        enfs.sort_values(by='raw heights').plot.scatter(x='retention time (scan)', y=pv, c='raw heights', colormap='summer', ax=ax[axval+1], alpha=0.5, norm=matplotlib.colors.LogNorm())

        axval += 2

    plt.suptitle(fr)
    plt.show()


#Plotting:
#Peak widths, as is
#Distribution of peak widths along scans
#mzML file intensity
#convert everything to time!

#Q's:
#Does a higher overall TIC for a scan indicate something about it's distribution?
#If there's more masses on a scan does the highest peak seem to dictate suppression? Like, is there less observable peaks, or more peaks of decreased intensity near an ultra-large peak?
