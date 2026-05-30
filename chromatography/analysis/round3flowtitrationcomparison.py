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

folder = '/store/flowcharacterizations/round3/DDAs/'

plotting = True
plotting = False

ddafolder = ''.join((folder, 'crux-output/'))
gradientsfolder = ''.join((folder, 'gradients/'))
dticfolder = ''.join((folder, 'TICs/'))
peakfolder = ''.join((folder, 'peaks/'))
pressurefolder = ''.join((folder, 'pressure/'))

files = os.listdir(peakfolder)
files = [i for i in files if i.endswith('.peaks.csv')]
files = [i for i in files if '300' in i]

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


#files = [files[0], files[3], files[6]]

pv = 'trimmed base widths (min)'
xplotval = 'Time (m)'
xplotval = 'Column Volume B'

deadvolume = 3225 #nL
columnvolume = 530 #nL
non_decimal = re.compile(r'[^\d.]+')

#this is the version with accurate volume calculations
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
    
    #ic is in-column
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
nrows = len(files)
fig, ax = plt.subplots(nrows=nrows, figsize=(10,3*nrows), constrained_layout=True, sharex=True)
#vfig, vax = plt.subplots(figsize=(10,3))
for n, fr in enumerate(files):
    ftitle = '_'.join((fr.split('.')[0].split('_')[1:]))
    f = ''.join((peakfolder, fr))
    fbase = fr.split('.')[0]
    enfs = pd.read_csv(f, low_memory=False)
    enfs.set_index('retention time (scan)', inplace=True)
    
    prname = ''.join((pressurefolder, fbase, '.pressure.csv'))
    pressure = pd.read_csv(prname)
    pressure.loc[:,'Total Flow'] = pressure.loc[:,'Pump A Real Flow'] + pressure.loc[:,'Pump B Real Flow']
    pressure.loc[:,'Total Desired Flow'] = pressure.loc[:,'Pump A Desired Flow'] + pressure.loc[:,'Pump B Desired Flow']
    pressure.loc[:,'%B'] = pressure.loc[:,'Pump B Real Flow'] / pressure.loc[:,'Total Flow']
    pressure.loc[:,'Desired %B'] = pressure.loc[:,'Pump B Desired Flow'] / pressure.loc[:,'Total Desired Flow']
    pressure.drop_duplicates('Time (minutes)', keep='first', inplace=True)
    pressure.fillna(0.02, inplace=True)
    pressure.loc[:,'Time (minutes)'] = pressure.loc[:,'Time (minutes)'].apply(lambda x: non_decimal.sub('', x))
    pressure.loc[:,'Time (minutes)'] = pressure.loc[:,'Time (minutes)'].apply(lambda x: x if x[-1].isdigit() else x[:-1])
    pressure = pressure.astype(float)
    pressure.sort_values('Time (minutes)', inplace=True)
    pressure = pressure.loc[pressure.loc[:,'Time (minutes)'] <= 180]
    pressure.loc[:,'Total Volume'] = integrate.cumtrapz(pressure.loc[:,'Total Flow'], pressure.loc[:,'Time (minutes)'], initial=0)
    #pressure.plot.line(x='Time (minutes)', y='Total Volume', ax=vax, label=ftitle)
    
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
    binname = ' '.join(('Binned', 'Time (m)'))
    egf.loc[:,binname] = egf.reset_index().loc[:,'Time (m)'].apply(lambda x: np.floor(x)).tolist()
    egf.loc[:,'scan'] = dtic.loc[:,'scan']
    
    pepids = egf.set_index(binname).loc[:,'Identified Peptide'].sum(level=0)
    
    dtic.reset_index(inplace=True)
    dtic.rename({'Time': 'Time (m)'}, axis=1, inplace=True)
    
    enfs.loc[:,'Time (m)'] = dtic.loc[enfs.reset_index().loc[:,'retention time (scan)'], 'Time (m)'].tolist()
    enfs.loc[:,'Column Volumes'] = egf.loc[enfs.loc[:,'Time (m)'],'Column Volumes'].tolist()
    enfs.loc[:,'Total Volume'] = egf.loc[enfs.loc[:,'Time (m)'],'Total Volume'].tolist()
    enfs.loc[:,'Cumulative Volume B'] = egf.loc[enfs.loc[:,'Time (m)'],'Cumulative Volume B'].tolist()
    enfs.loc[:,'Cumulative Volume A'] = egf.loc[enfs.loc[:,'Time (m)'],'Cumulative Volume A'].tolist()
    enfs.loc[:,'Flow A at Interval'] = egf.loc[enfs.loc[:,'Time (m)'],'Flow A at Interval'].tolist()
    enfs.loc[:,'Flow B at Interval'] = egf.loc[enfs.loc[:,'Time (m)'],'Flow B at Interval'].tolist()
    enfs.loc[:,'%ACN'] = egf.loc[enfs.loc[:,'Time (m)'],'%ACN'].tolist()
    enfs.loc[:,'Flow'] = egf.loc[enfs.loc[:,'Time (m)'],'Flow'].tolist()
    enfs.loc[:,'Column Volume B'] = enfs.loc[:,'Cumulative Volume B'] / columnvolume
    
    enfs.loc[:,'ic-timepoint'] = egf.loc[enfs.loc[:,'Time (m)'],'ic-timepoint'].tolist()
    enfs.loc[:,'ic-%ACN'] = egf.loc[enfs.loc[:,'Time (m)'],'ic-%ACN'].tolist()
    enfs.loc[:,'ic-tvolume'] = egf.loc[enfs.loc[:,'Time (m)'],'ic-tvolume'].tolist()
    enfs.loc[:,'ic-flow'] = egf.loc[enfs.loc[:,'Time (m)'],'ic-flow'].tolist()
    enfs.loc[:,'ic-bflow'] = egf.loc[enfs.loc[:,'Time (m)'],'ic-bflow'].tolist()
    enfs.loc[:,'ic-aflow'] = egf.loc[enfs.loc[:,'Time (m)'],'ic-aflow'].tolist()
    enfs.loc[:,'ic-bvol'] = egf.loc[enfs.loc[:,'Time (m)'],'ic-bvol'].tolist()
    enfs.loc[:,'ic-avol'] = egf.loc[enfs.loc[:,'Time (m)'],'ic-avol'].tolist()
    enfs.loc[:,'ic-%b/flow'] = enfs.loc[:,'ic-%ACN'] / enfs.loc[:,'Flow']
    
    dtic.loc[:,'Column Volumes'] = egf.loc[dtic.loc[:,'Time (m)'],'Column Volumes'].tolist()
    dtic.loc[:,'Cumulative Volume B'] = egf.loc[dtic.loc[:,'Time (m)'],'Cumulative Volume B'].tolist()
    dtic.loc[:,'Column Volume B'] = dtic.loc[:,'Cumulative Volume B'] / columnvolume
    
    gradient = timeconversion(gradient, gradient, fillval='Time (m)', reorder=False)
    gradient.fillna(0, inplace=True)
    gradient.loc[:,'Time'] = gradient.index.to_numpy()
    gradient.loc[:,'Column Volume B'] = gradient.loc[:,'Cumulative Volume B'] / columnvolume
    
    #efilter = enfs.loc[:,'ic-%ACN'] < 33
    #enfs = enfs.loc[efilter]
    #egf.reset_index().sort_values(xplotval).plot.line(x=xplotval, y='Peptides/Column Volume', ax=ax[1], color='purple', alpha=0.6, label='Peptides ID\'d / Column Volume')
    enfs.loc[:,'frequency'] = stats.gaussian_kde(enfs.loc[:,(xplotval, pv)].to_numpy().transpose())(enfs.loc[:,(xplotval, pv)].to_numpy().transpose())
    
    enfs.reset_index().sort_values(by='frequency').plot.scatter(x=xplotval, y=pv, c='frequency', colormap='winter', ax=ax[n], alpha=0.5, norm=matplotlib.colors.LogNorm())
    #enfs.reset_index().sort_values(by='frequency').plot.scatter(x=xplotval, y=pv, c='frequency', colormap='winter', ax=ax[n], alpha=0.5)
    
    #enfs.reset_index().sort_values(by='raw heights').plot.scatter(x=xplotval, y=pv, c='raw heights', colormap='summer', ax=ax[n], alpha=0.5, norm=matplotlib.colors.LogNorm())
    ax[n].set_title(ftitle)
    
    enfs.reset_index(inplace=True)
    enfs.set_index('Time (m)', inplace=True)
    il = (enfs.loc[:,'frequency'] * enfs.loc[:,'raw heights']).sum(level=0) / enfs.loc[:,'raw heights'].sum(level=0)
    il = il.to_frame()
    il.columns = [fbase]
    el.append(il)
    pfile.loc[:,'file_idx'] = fbase
    dtic = dtic.reset_index().set_index('scan')
    pfile.loc[:,'Time (m)'] = dtic.loc[pfile.loc[:,'scan'], 'Time (m)'].tolist()
    dts.append(pfile)

plt.show()

