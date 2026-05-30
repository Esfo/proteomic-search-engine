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

rf = '/home/sfo/store/flowcharacterizations/round2/DDAs/MGFs/results.csv'
df = pd.read_csv(rf)
df.loc[:,'file'] = df.loc[:,'file'].apply(lambda x: x.split('.')[0].split('_')[-1])
df.set_index('file', inplace=True)

#basefolder = '/store/flowcharacterizations/round2/MS1s/'
basefolder = '/home/sfo/store/FE/'

peakfolder = ''.join((basefolder, 'peaks/'))
ticfolder = ''.join((basefolder, 'TICs/'))
files = os.listdir(peakfolder)
files = [i for i in files if i.endswith('.peaks.csv')]
#files = [i for i in files if 'stat' in i]
files = [i for i in files if 'FdV' not in i]

filesort = [int(i.split('_')[1].split('F')[1]) for i in files]
filesort = np.asarray(filesort)

files = np.asarray(files)
files = files[filesort.argsort()].tolist()
files = files[0::5][:4]

plottervals = ['trimmed base widths']
xplotval = 'column volume'


for fr in files:
#fr = files[-1]
    f = ''.join((peakfolder, fr))
    fbase = fr.split('.')[0]
    enfs = pd.read_csv(f, low_memory=False)

    enfs.loc[:,'max prominence'] = enfs.loc[:,('left prominence', 'right prominence')].max(axis=1)
    enfs.loc[enfs.loc[:,'max prominence'] == np.inf, 'max prominence'] = np.nan

    ticfile = ''.join((ticfolder, fbase, '.tic.csv'))
    of = pd.read_csv(ticfile, low_memory=False)
    of.sort_values('scan', inplace=True)

    enfs.loc[:,'minutes'] = of.loc[enfs.loc[:,'peak location'].tolist(), 'minutes'].tolist()
    flowrate = int(fr.split('_')[1][1:])
    columnvolume = 662.679700366597 #nL
    enfs.loc[:,'nL'] = enfs.loc[:,'minutes'] * flowrate
    enfs.loc[:,'column volume'] = enfs.loc[:,'nL'] / columnvolume
    of.loc[:,'nL'] = of.loc[:,'minutes'] * flowrate
    of.loc[:,'column volume'] = of.loc[:,'nL'] / columnvolume


    nrows = len(plottervals) * 2 + 1
    fig, ax = plt.subplots(nrows=nrows, figsize=(10,3*nrows), constrained_layout=True, sharex=True)
    axval = 1

    of.sort_values(xplotval).plot.line(x=xplotval, y='summed intensity', color='purple', ax=ax[0], legend=False)

    for pv in plottervals:
        enfs.loc[:,'frequency'] = stats.gaussian_kde(enfs.loc[:,(xplotval, pv)].to_numpy().transpose())(enfs.loc[:,(xplotval, pv)].to_numpy().transpose())

        enfs.sort_values(by='frequency').plot.scatter(x=xplotval, y=pv, c='frequency', colormap='winter', ax=ax[axval], alpha=0.5, norm=matplotlib.colors.LogNorm())

        enfs.sort_values(by='raw heights').plot.scatter(x=xplotval, y=pv, c='raw heights', colormap='summer', ax=ax[axval+1], alpha=0.5, norm=matplotlib.colors.LogNorm())

        axval += 2

    plt.suptitle(flowrate)
    plt.show()


#Plotting:
#Peak widths, as is
#Distribution of peak widths along scans
#mzML file intensity
#convert everything to time!

#Q's:
#Does a higher overall TIC for a scan indicate something about it's distribution?
#If there's more masses on a scan does the highest peak seem to dictate suppression? Like, is there less observable peaks, or more peaks of decreased intensity near an ultra-large peak?
