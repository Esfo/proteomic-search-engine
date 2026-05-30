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

ms1folder = '/store/flowcharacterizations/round2/MS1s/'
ddafolder = '/store/flowcharacterizations/round2/DDAs/MGFs/crux-output/'
gradientsfolder = '/store/flowcharacterizations/round2/gradients/'
dticfolder = '/store/flowcharacterizations/round2/DDAs/TICs/'

peakfolder = ''.join((ms1folder, 'peaks/'))
mticfolder = ''.join((ms1folder, 'TICs/'))


ms1folder = '/store/flowcharacterizations/round3/DDAs/'
ddafolder = '/store/flowcharacterizations/round3/DDAs/crux-output/'
gradientsfolder = '/store/flowcharacterizations/round3/gradients/'
dticfolder = '/store/flowcharacterizations/round3/DDAs/TICs/'
peakfolder = '/store/flowcharacterizations/round3/DDAs/peaks/'
mticfolder = 'store/flowcharacterizations/round3/DDAs/TICs/'

files = os.listdir(peakfolder)
files = [i for i in files if i.endswith('.peaks.csv')]
#files = [i for i in files if 'stat' in i]

plottervals = ['trimmed base widths']
xplotval = 'Time'

columnvolume = 662.679700366597 #nL


def timeconversion(frameobj, gr):
    try:
        tf = frameobj.loc[:,'Time'].to_frame()
    except AttributeError:
        tf = frameobj.loc[:,'Time']
    tf.columns = ['Time']
    gr = gr.append(tf)
    gr.loc[gr.loc[:,'Time'] > gmaxpoint, '%ACN'] = 90
    gr = gr.sort_values('Time').interpolate()
    gr.loc[:,'Interval (m)'] = gr.sort_values('Time').loc[:,'Time'].diff()
    gr.iloc[0].fillna(0, inplace=True)

    gr.loc[:,'Volume (nL) /Time (m)'] = gr.loc[:,'Flow'] * gr.loc[:,'Interval (m)']
    gr.loc[:,'Total Volume'] = gr.loc[:,'Volume (nL) /Time (m)'].cumsum()
    gr.loc[:,'Column Volumes'] = gr.loc[:,'Total Volume'] / columnvolume
    gr.drop_duplicates('Time', inplace=True)
    gr.set_index('Time', inplace=True)
    
    egf = gr.loc[frameobj.loc[:,'Time']]
    egf.drop_duplicates(inplace=True)
    return egf


for fr in files:
#fr = files[-1]
    fr = files[1]
    f = ''.join((peakfolder, fr))
    fbase = fr.split('.')[0]
    enfs = pd.read_csv(f, low_memory=False)
    enfs.set_index('peak location', inplace=True)
    
    percolatorfile = pd.read_csv(''.join((ddafolder, ''.join(('-'.join((fr.split('.')[0].split('-')[:-1])), '-DDA', '.percolator.target.peptides.txt')))), delimiter='\t')
    percolatorfile = percolatorfile.loc[percolatorfile.loc[:,'percolator q-value'] < 0.01]
    
    dticfile = pd.read_csv(''.join((dticfolder, ''.join(('-'.join((fr.split('.')[0].split('-')[:-1])), '-DDA', '.tic.csv')))))
    
    dticfile.loc[percolatorfile.sort_values('scan').loc[:,'scan'], 'Identified Peptide'] = 1
    dticfile.loc[:,'Identified Peptide'].fillna(0, inplace=True)
    
    gradient = pd.read_csv(''.join((gradientsfolder, '-'.join((fr.split('.')[0].split('_')[1].split('-')[:-1])), '.method.csv')))
    gmaxpoint = gradient.loc[:,'Time'].max()
    gradient.loc[:,'Duration (s)'] = gradient.loc[:,'Duration'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
    gradient.loc[:,'Duration (m)'] = gradient.loc[:,'Duration (s)'] / 60
    
    egf = timeconversion(dticfile, gradient)
    
    dticfile.set_index('Time', inplace=True)
    egf.loc[:,'Identified Peptide'] = dticfile.loc[:,'Identified Peptide']
    egf.loc[:,'Cumulative Peptides'] = egf.loc[:,'Identified Peptide'].sort_index().cumsum().tolist()
    egf.loc[:,'Peptides/Column Volume'] = egf.loc[:,'Cumulative Peptides'] / egf.loc[:,'Column Volumes']
    egf.loc[:,'Binned Column Volumes'] = egf.loc[:,'Column Volumes'].apply(lambda x: np.floor(x))
    egf.loc[:,'scan'] = dticfile.loc[:,'scan']
    
    pepids = egf.set_index('Binned Column Volumes').loc[:,'Identified Peptide'].sum(level=0)
    
    mticfile = ''.join((mticfolder, fbase, '.tic.csv'))
    of = pd.read_csv(mticfile, low_memory=False)
    of.sort_values('scan', inplace=True)
    of.set_index('scan', inplace=True)
    
    enfs.loc[:,'Time'] = of.loc[enfs.reset_index().loc[:,'peak location'], 'Time'].tolist()
    
    engf = timeconversion(enfs, gradient)
    engf.sort_index(inplace=True)
    ogf = timeconversion(of, gradient)
    ogf.sort_index(inplace=True)
    dtgf = timeconversion(dticfile.reset_index(), gradient)
    dtgf.sort_index(inplace=True)
    
    of.loc[:,'Column Volumes'] = ogf.loc[of.loc[:,'Time'], 'Column Volumes'].tolist()
    enfs.loc[:,'Column Volumes'] = engf.loc[enfs.loc[:,'Time'],'Column Volumes'].tolist()
    dticfile.loc[:,'Column Volumes'] = dtgf.loc[dticfile.reset_index().loc[:,'Time'],'Column Volumes'].tolist()
    
    fig, ax = plt.subplots(nrows=3, figsize=(10,9), sharex=True)
    
    dticfile.reset_index().sort_values(xplotval).plot.line(x=xplotval, y='summed intensity', color='green', ax=ax[0], label='DDA', alpha=0.3)
    ax[0].set_ylabel('Summed Intensity')
    
    egf.reset_index().sort_values(xplotval).plot.line(x=xplotval, y='Peptides/Column Volume', ax=ax[1], legend=False, color='cornflowerblue', alpha=0.6)
    ax[1].set_ylabel('Peptides ID\'d / Column Volume')
    
    egf.reset_index().sort_values(xplotval).plot.line(x=xplotval, y='Cumulative Peptides', ax=ax[2], legend=False, color='mediumseagreen', alpha=0.6)
    ax[2].set_ylabel('Cumulative Peptide ID\'s')
    
    #axn = ax[1].twinx()
    #
    #pepids.plot.bar(color='goldenrod', alpha=0.4)
    #axn.set_ylabel('# Identified Peptides')

    plt.suptitle('-'.join((fr.split('.')[0].split('-')[:-1])).split('_')[1])

    
    gradient = timeconversion(gradient, gradient)
    gradient.fillna(0, inplace=True)

    
    fig, ax = plt.subplots(nrows=2, figsize=(10,6))
    gradient.plot.line(x='Column Volumes', y='Flow', ax=ax[0], color='green')
    axn = ax[0].twinx()
    gradient.plot.line(x='Column Volumes', y='%ACN', ax=axn, color='purple')
    
    gradient.reset_index().plot.line(x='Time', y='Column Volumes', ax=ax[1])
    
    plt.suptitle('-'.join((fr.split('.')[0].split('-')[:-1])).split('_')[1])
    plt.show()
    
    nrows = len(plottervals) * 2 + 1
    fig, ax = plt.subplots(nrows=nrows, figsize=(10,3*nrows), constrained_layout=True, sharex=True)
    axval = 1
    
    of.reset_index().sort_values(xplotval).plot.line(x=xplotval, y='summed intensity', color='purple', ax=ax[0], label='MS1', alpha=0.3)
    
    for pv in plottervals:
        enfs.loc[:,'frequency'] = stats.gaussian_kde(enfs.loc[:,(xplotval, pv)].to_numpy().transpose())(enfs.loc[:,(xplotval, pv)].to_numpy().transpose())
        
        enfs.reset_index().sort_values(by='frequency').plot.scatter(x=xplotval, y=pv, c='frequency', colormap='winter', ax=ax[axval], alpha=0.5, norm=matplotlib.colors.LogNorm())
        
        enfs.reset_index().sort_values(by='raw heights').plot.scatter(x=xplotval, y=pv, c='raw heights', colormap='summer', ax=ax[axval+1], alpha=0.5, norm=matplotlib.colors.LogNorm())
        
        axval += 2
    
    plt.suptitle('-'.join((fr.split('.')[0].split('-')[:-1])).split('_')[1])
    plt.show()

    break


#Plotting:
#Peak widths, as is
#Distribution of peak widths along scans
#mzML file intensity
#convert everything to time!

#Q's:
#Does a higher overall TIC for a scan indicate something about it's distribution?
#If there's more masses on a scan does the highest peak seem to dictate suppression? Like, is there less observable peaks, or more peaks of decreased intensity near an ultra-large peak?
