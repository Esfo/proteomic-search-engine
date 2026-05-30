import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats, integrate, interpolate
import os
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
import re
plt.rcParams["figure.dpi"] = 100
warnings.filterwarnings("ignore")

folder = '/store/flowcharacterizations/round4/fusion/'

plotting = True
#plotting = False

ddafolder = ''.join((folder, 'crux-output/'))
gradientsfolder = ''.join((folder, 'gradients/'))
dticfolder = ''.join((folder, 'TICs/'))
peakfolder = ''.join((folder, 'peaks/'))
pressurefolder = ''.join((folder, 'pressure/'))

files = os.listdir(peakfolder)
files = [i for i in files if i.endswith('.peaks.csv')]

#files = [files[0], files[3], files[6]]

plottervals = ['s base widths (min)']
xplotval = 'retention time (min)'
#xplotval = 'Total Volume'
#xplotval = 'Cumulative Volume B'
cval = 'r height'
#cval = 'r area'
ca = True #sort cval ascending on plot

deadvolume = 3225 #nL
columnvolume = 530 #nL
non_decimal = re.compile(r'[^\d.]+')

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

def timeconversion(frameobj, gr, fillval='retention time (min)', reorder=True):
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

dts = []
el = []
for fr in files:
    ftitle = '_'.join((fr.split('.')[0].split('_')[1:]))
    f = ''.join((peakfolder, fr))
    fbase = fr.split('.')[0]
    enfs = pd.read_csv(f, low_memory=False)
    enfs.set_index('retention time (min)', inplace=True)
    
    #prname = ''.join((pressurefolder, fbase, '.pressure.csv'))
    #pressure = pd.read_csv(prname)
    #pressure.loc[:,'Total Flow'] = pressure.loc[:,'Pump A Real Flow'] + pressure.loc[:,'Pump B Real Flow']
    #pressure.loc[:,'Total Desired Flow'] = pressure.loc[:,'Pump A Desired Flow'] + pressure.loc[:,'Pump B Desired Flow']
    #pressure.loc[:,'%B'] = pressure.loc[:,'Pump B Real Flow'] / pressure.loc[:,'Total Flow']
    #pressure.loc[:,'Desired %B'] = pressure.loc[:,'Pump B Desired Flow'] / pressure.loc[:,'Total Desired Flow']
    #pressure.drop_duplicates('Time (minutes)', keep='first', inplace=True)
    #pressure.fillna(0.02, inplace=True)
    #pressure.loc[:,'Time (minutes)'] = pressure.loc[:,'Time (minutes)'].apply(lambda x: non_decimal.sub('', x))
    #pressure.loc[:,'Time (minutes)'] = pressure.loc[:,'Time (minutes)'].apply(lambda x: x if x[-1].isdigit() else x[:-1])
    #pressure = pressure.astype(float)
    #pressure.sort_values('Time (minutes)', inplace=True)
    #pressure = pressure.loc[pressure.loc[:,'Time (minutes)'] <= 180]
    
    pname = ''.join((ddafolder, fbase, '.percolator.target.peptides.txt'))
    pfile = pd.read_csv(pname, delimiter='\t')
    pfile = pfile.loc[pfile.loc[:,'percolator q-value'] < 0.01]
    
    dname = ''.join((dticfolder, fbase, '.tic.csv'))
    dtic = pd.read_csv(dname)
    dtic.rename({'Time': 'retention time (min)'}, axis=1, inplace=True)
    
    dtic.loc[pfile.sort_values('scan').loc[:,'scan'], 'Identified Peptide'] = 1
    dtic.loc[:,'Identified Peptide'].fillna(0, inplace=True)
    
    #gname = ''.join((gradientsfolder, ''.join((fbase.split('_')[1:-1])), '.method.csv'))
    #try:
    #    gradient = pd.read_csv(gname)
    #except FileNotFoundError:
    #    gname = ''.join((gradientsfolder, '_'.join((fbase.split('_')[1:])), '.method.csv'))
    #    gradient = pd.read_csv(gname)
    #
    #egf = timeconversion(dtic, gradient)
    #egf.fillna(0, inplace=True)
    #egf.index.name = 'retention time (min)'
    
    #dtic.set_index('retention time (min)', inplace=True)
    #egf.loc[:,'Identified Peptide'] = dtic.loc[:,'Identified Peptide']
    #egf.loc[:,'Cumulative Peptides'] = egf.loc[:,'Identified Peptide'].sort_index().cumsum().tolist()
    #egf.loc[:,'Peptides/Column Volume'] = egf.loc[:,'Cumulative Peptides'] / egf.loc[:,'Column Volumes']
    #binname = ' '.join(('Binned', xplotval))
    #egf.loc[:,binname] = egf.reset_index().loc[:,xplotval].apply(lambda x: np.floor(x)).tolist()
    #egf.loc[:,'scan'] = dtic.loc[:,'scan']
    #
    #pepids = egf.set_index(binname).loc[:,'Identified Peptide'].sum(level=0)
    #
    #dtic.reset_index(inplace=True)
    
    #enfs.loc[:,'Time (m)'] = dtic.loc[enfs.reset_index().loc[:,'retention time (scan)'], 'Time (m)'].tolist()
    #enfs.loc[:,'Column Volumes'] = egf.loc[enfs.loc[:,'retention time (min)'],'Column Volumes'].tolist()
    #enfs.loc[:,'Total Volume'] = egf.loc[enfs.loc[:,'retention time (min)'],'Total Volume'].tolist()
    #enfs.loc[:,'Cumulative Volume B'] = egf.loc[enfs.loc[:,'retention time (min)'],'Cumulative Volume B'].tolist()
    #enfs.loc[:,'Column Volume B'] = enfs.loc[:,'Cumulative Volume B'] / columnvolume
    #dtic.loc[:,'Column Volumes'] = egf.loc[dtic.loc[:,'retention time (min)'],'Column Volumes'].tolist()
    #dtic.loc[:,'Total Volume'] = egf.loc[dtic.loc[:,'retention time (min)'],'Total Volume'].tolist()
    #dtic.loc[:,'Cumulative Volume B'] = egf.loc[dtic.loc[:,'retention time (min)'],'Cumulative Volume B'].tolist()
    #dtic.loc[:,'Column Volume B'] = dtic.loc[:,'Cumulative Volume B'] / columnvolume
    
    #gradient = timeconversion(gradient, gradient, fillval='Time (m)', reorder=False)
    #gradient.fillna(0, inplace=True)
    #gradient.loc[:,'Column Volume B'] = gradient.loc[:,'Cumulative Volume B'] / columnvolume
    #gradient.rename({'retention time (min)':'Time (m)'}, axis=1, inplace=True)

    enfs.loc[:,'min prominence'] = enfs.loc[:,('r left prominence', 'r right prominence')].min(axis=1)
    
    if plotting:
        fig, ax = plt.subplots(nrows=4, figsize=(10,10))
        #pressure.plot.line(x='Time (minutes)', y='Total Flow', ax=ax[0], label='True Flow', alpha=0.5)
        #pressure.plot.line(x='Time (minutes)', y='Total Desired Flow', color='black', ax=ax[0])
        #
        #pressure.plot.line(x='Time (minutes)', y='%B', ax=ax[1], label='True %ACN', alpha=0.5)
        #pressure.plot.line(x='Time (minutes)', y='Desired %B', ax=ax[1], alpha=0.5, color='black')
        #
        #pressure.plot.line(x='Time (minutes)', y='Pump A Pressure', ax=ax[2])
        #pressure.plot.line(x='Time (minutes)', y='Pump B Pressure', ax=ax[2])
        
        #gradient.reset_index().plot.line(x='Time (m)', y='Column Volume B', ax=ax[3])
        #gradient.reset_index().plot.line(x='Time (m)', y='Column Volumes', ax=ax[3])
        
        plt.suptitle(ftitle)
        plt.show()
    
        nrows = len(plottervals) * 5 + 2
        fig, ax = plt.subplots(nrows=nrows, figsize=(10,3*nrows), constrained_layout=True, sharex=True)
        axval = 2
        
        dtic.reset_index().sort_values(xplotval).plot.line(x=xplotval, y='summed intensity', color='green', ax=ax[0], label='TIC', alpha=0.3)
        
        #egf.reset_index().sort_values(xplotval).plot.line(x=xplotval, y='Peptides/Column Volume', ax=ax[1], color='purple', alpha=0.6, label='Peptides ID\'d / Column Volume')
        
        egf.reset_index().sort_values(xplotval).plot.line(x=xplotval, y='Cumulative Peptides', ax=ax[1], color='mediumseagreen', alpha=0.6, label='Cumulative Peptide ID\'s')
        axn = ax[1].twinx()
        pepids.plot.bar(color='goldenrod', alpha=0.4, ax=axn, label=' '.join(('Peptide ID\'s per', binname)))
        
        for pv in plottervals:
            enfs.loc[:,'frequency'] = stats.gaussian_kde(enfs.loc[:,(xplotval, pv)].to_numpy().transpose())(enfs.loc[:,(xplotval, pv)].to_numpy().transpose())
            
            enfs.reset_index().sort_values(by='frequency').plot.scatter(x=xplotval, y=pv, c='frequency', colormap='winter', ax=ax[axval], alpha=0.5, norm=matplotlib.colors.LogNorm())
            
            enfs.reset_index().sort_values(by=cval, ascending=ca).plot.scatter(x=xplotval, y=pv, c='r area', colormap='summer', ax=ax[axval+1], alpha=0.3, norm=matplotlib.colors.LogNorm())
            
            enfs.reset_index().sort_values(by=cval, ascending=ca).plot.scatter(x=xplotval, y=pv, c='r height', colormap='summer', ax=ax[axval+2], alpha=0.3, norm=matplotlib.colors.LogNorm())
            
            enfs.sort_values(by=xplotval).plot.scatter(x=xplotval, y='frequency', ax=ax[axval+3])
            
            enfs.reset_index().sort_values(by='frequency').plot.scatter(x=xplotval, y='min prominence', c='frequency', colormap='winter', ax=ax[axval+4])
            ax[axval+4].set_yscale('log')
            
            enfs.loc[:,'file'] = fbase

            axval += 2
       
        ax[-1].set_xticks([0, 30, 60, 90, 120, 150, 180])
        
        plt.suptitle(ftitle)
        plt.show()

        enfs.plot.scatter(x='r area', y='r height')
        plt.xscale('log')
        plt.yscale('log')
        plt.title(ftitle)
        plt.show()
    else:
        enfs.loc[:,'frequency'] = stats.gaussian_kde(enfs.loc[:,(xplotval, 'trimmed base widths (min)')].to_numpy().transpose())(enfs.loc[:,(xplotval, 'trimmed base widths (min)')].to_numpy().transpose())

    enfs.reset_index(inplace=True)
    enfs.set_index('retention time (min)', inplace=True)
    il = (enfs.loc[:,'frequency'] * enfs.loc[:,'r height']).sum(level=0) / enfs.loc[:,'r height'].sum(level=0)
    il = il.to_frame()
    il.columns = [fbase]
    el.append(il)
    pfile.loc[:,'file_idx'] = fbase
    dtic = dtic.reset_index().set_index('scan')
    pfile.loc[:,'retention time (min)'] = dtic.loc[pfile.loc[:,'scan'], 'retention time (min)'].tolist()
    dts.append(pfile)

el = pd.concat(el)
#el.reset_index(inplace=True)
#el.set_index('file', inplace=True)
el.sort_index(inplace=True)

if plotting:
    fig, ax = plt.subplots(figsize=(10,3))
    for c in el.columns:
        el.loc[~el.loc[:,c].isnull(), c].plot.line(ax=ax, label=c, alpha=0.4)
    plt.legend(bbox_to_anchor=(1,1))
    plt.show()

binsize = 5 #minutes
bins = round(el.index.max() / binsize)
el.interpolate(limit_direction='both', inplace=True)
el.loc[:,'Time binned'] = pd.cut(el.index, bins=bins, labels=False)

el.reset_index(inplace=True)
el.set_index('Time binned', inplace=True)

nl = el.max(level=0)
nl.drop(labels='retention time (min)', inplace=True, axis=1)

if plotting:
    nl.plot.line(figsize=(10,3))
    plt.legend(bbox_to_anchor=(1,1))
    plt.show()

areas = pd.DataFrame(index=nl.columns)
areas.loc[:,'Area'] = np.trapezoid(nl.to_numpy(), nl.index, axis=0)
areas.sort_values('Area', inplace=True)

results = pd.read_csv('/store/flowcharacterizations/round3/DDAs/results.csv')
results.loc[:,'file'] = results.loc[:,'file'].apply(lambda x: x.split('.')[0])
results.set_index('file', inplace=True)

results = pd.concat([results, areas], axis=1)
results.sort_values('Area', inplace=True)

if plotting:
    results.plot.scatter(x='peptides', y='Area')
    plt.show()

dts = pd.concat(dts)
dts = dts.pivot_table(index='sequence', columns='file_idx', values='retention time (min)')
dts.drop(['200901_fR_400', '200901_fR_500'], inplace=True, axis=1)
nts = dts.loc[dts.count(axis=1) > 4]
nts.loc[:,'avg'] = nts.mean(axis=1)

if plotting:
    fig, ax = plt.subplots(figsize=(10,3))
    nts.sort_values('avg').plot.line(ax=ax)
    plt.legend(bbox_to_anchor=(1,1))
    plt.show()


#el.loc[:,'density index'] = el.loc[:,'retention time (scan)']
#adders = el.loc[:,'retention time (scan)'].max(level=0)
#addnum = 0
#for r, a in adders.to_frame().iterrows():
#    el.loc[r, 'density index'] += addnum
#    addnum += a.to_numpy()[0]
#
#el.loc[:,'baseline frequency'] = stats.gaussian_kde(el.loc[:,('density index', pv)].to_numpy().transpose())(el.loc[:,('density index', pv)].to_numpy().transpose())
#
#
#fig, ax = plt.subplots(nrows=el.index.unique().shape[0], figsize=(10,el.index.unique().shape[0]*3), sharex=True)
#axval = 0
#for f in el.index.unique():
#    el.loc[f].reset_index().sort_values(by='frequency').plot.scatter(x=xplotval, y=pv, c='baseline frequency', colormap='winter', ax=ax[axval], alpha=0.5, norm=matplotlib.colors.LogNorm())
#    ax[axval].set_title(f)
#    axval += 1
#im = plt.gca().get_children()[0]
#cax = fig.add_axes([0.85,0.1,0.03,0.8])
#fig.colorbar(im, cax=cax)
#plt.show()

#Plotting:
#Peak widths, as is
#Distribution of peak widths along scans
#mzML file intensity
#convert everything to time!

#Q's:
#Does a higher overall TIC for a scan indicate something about it's distribution?
#If there's more masses on a scan does the highest peak seem to dictate suppression? Like, is there less observable peaks, or more peaks of decreased intensity near an ultra-large peak?

#enfs.reset_index(inplace=True)
#
#axval = 0
#
#prominences = [0]
#
#for p in prominences:
#    pnfs = enfs.loc[enfs.loc[:,'min prominence'] >= p]
#    
#    fig, ax = plt.subplots(nrows=4, figsize=(10,12), constrained_layout=True, sharex=True)
#    pnfs.loc[:,'frequency'] = stats.gaussian_kde(pnfs.loc[:,(xplotval, pv)].to_numpy().transpose())(pnfs.loc[:,(xplotval, pv)].to_numpy().transpose())
#
#    pnfs.reset_index().sort_values(by='frequency').plot.scatter(x=xplotval, y=pv, c='frequency', colormap='winter', ax=ax[axval], alpha=0.5, norm=matplotlib.colors.LogNorm())
#
#    pnfs.reset_index().sort_values(by=cval, ascending=ca).plot.scatter(x=xplotval, y=pv, c=cval, colormap='summer', ax=ax[axval+1], alpha=0.3, norm=matplotlib.colors.LogNorm())
#    
#    pnfs.sort_values(by='Time').plot.scatter(x='Time', y='frequency', ax=ax[axval+2], norm=matplotlib.colors.LogNorm())
#    
#    enfs.reset_index().sort_values(by='frequency').plot.scatter(x='Time', y='min prominence', c='frequency', colormap='winter', ax=ax[axval+3])
#    plt.yscale('log')
#
#    plt.suptitle(p)
#    plt.show()
#
#fig, ax = plt.subplots(figsize=(10,5))
#enfs.reset_index().sort_values(by='frequency').plot.scatter(x='Time (m)', y='min prominence', c='frequency', colormap='winter', ax=ax)
#plt.yscale('log')
#plt.show()
