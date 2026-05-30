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

if os.uname()[1] == 'toaster':
    plt.rcParams['figure.dpi'] = 180
elif os.uname()[1] == 'box':
    plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.facecolor'] = 'gray'
plt.rcParams['figure.facecolor'] = 'gray'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['ytick.labelcolor'] = 'white'
plt.rcParams['xtick.labelcolor'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['xtick.color'] = 'white'

fr = '/store/flowcharacterizations/round3/DDAs/gradients/fR_400.method.csv'
deadvolume = 3225 #nL
columnvolume = 530 #nL
npoints = 1000 #number of points to interpolate
gradient = pd.read_csv(fr)

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
    gr.loc[:,'column-timepoint'] = volumefinder(gr.loc[:,'Time (m)'], gr.loc[:,'Flow'], deadvolume)
    #some values come out just below the minimum timepoint because of computational precision differences, but for the ones that do it's going to be the same conditions as the initial timepoint, so this is fixed here
    gr.loc[(gr.loc[:,'column-timepoint'] < gr.loc[:,'Time (m)'].min()), 'column-timepoint'] = gr.loc[:,'Time (m)'].min()
    #the volumesolver already takes the flow and volume into consideration, so simply solving for whatever was pumping at the solved timepoint is a legitemate measure for the %ACN in the column at a given time.
    percinterp = interpolate.interp1d(gr.loc[:,'Time (m)'].to_numpy(), gr.loc[:,'%ACN'].to_numpy())
    flowinterp = interpolate.interp1d(gr.loc[:,'Time (m)'], gr.loc[:,'Flow'])
    
    gr.loc[:,'column-%ACN'] = percinterp(gr.loc[:,'column-timepoint'])
    #column-flow not used for actual time flow measurements, it should only be used to determine volumes
    gr.loc[:,'column-flow'] = flowinterp(gr.loc[:,'column-timepoint'])
    
    gr.loc[:,'column-tvolume'] = integrate.cumtrapz(gr.loc[:,'column-flow'], gr.loc[:,'column-timepoint'], initial=0)
    gr.loc[:,'column-bflow'] = gr.loc[:,'column-flow'] * gr.loc[:,'column-%ACN'] / 100
    gr.loc[:,'column-aflow'] = gr.loc[:,'column-flow'] * (100 - gr.loc[:,'column-%ACN']) / 100
    
    gr.loc[:,'column-bvol'] = integrate.cumtrapz(gr.loc[:,'column-bflow'], gr.loc[:,'column-timepoint'], initial=0) #nL
    gr.loc[:,'column-avol'] = integrate.cumtrapz(gr.loc[:,'column-aflow'], gr.loc[:,'column-timepoint'], initial=0) #nL
        
    gr.drop_duplicates('Time (m)', inplace=True)
    gr.set_index('Time (m)', inplace=True)
    
    egf = gr.loc[frameobj.loc[:,fillval]]
    egf.drop_duplicates(inplace=True)
    return egf

def humanize_time(secs):
    mins, secs = divmod(secs, 60)
    return '%02d:%02d' % (mins, secs)

tmin = gradient.loc[:,'Time (m)'].min()
tmax = gradient.loc[:,'Time (m)'].max()

timepoints = pd.DataFrame()
timepoints.loc[:,'Time (m)'] = np.linspace(tmin, tmax, npoints)

gradient = pd.concat((gradient, timepoints))
gradient.sort_values('Time (m)', inplace=True)
gradient.drop_duplicates('Time (m)', inplace=True)
gradient = gradient.interpolate(direction='both')
gradient.reset_index(drop=True, inplace=True)
gradient.drop(['timediff', 'Duration'], inplace=True, axis=1)

gradient.loc[:,'Total Volume'] = integrate.cumtrapz(gradient.loc[:,'Flow'], gradient.loc[:,'Time (m)'], initial=0)
gradient.loc[:,'Column Volumes'] = gradient.loc[:,'Total Volume'] / columnvolume
gradient.loc[:,'Flow B at Interval'] = gradient.loc[:,'Flow'] * gradient.loc[:,'%ACN'] / 100
gradient.loc[:,'Flow A at Interval'] = gradient.loc[:,'Flow'] * (100 - gradient.loc[:,'%ACN']) / 100
gradient.loc[:,'Cumulative Volume B'] = integrate.cumtrapz(gradient.loc[:,'Flow B at Interval'], gradient.loc[:,'Time (m)'], initial=0) #nL
gradient.loc[:,'Cumulative Volume A'] = integrate.cumtrapz(gradient.loc[:,'Flow A at Interval'], gradient.loc[:,'Time (m)'], initial=0) #nL
gradient.loc[:,'column-timepoint'] = volumefinder(gradient.loc[:,'Time (m)'].to_numpy(), gradient.loc[:,'Flow'].to_numpy(), deadvolume)

#the volumesolver already takes the flow and volume into consideration, so simply solving for whatever was pumping at the solved timepoint is a legitemate measure for the %ACN in the column at a given time.
percinterp = interpolate.interp1d(gradient.loc[:,'Time (m)'].to_numpy(), gradient.loc[:,'%ACN'].to_numpy())
flowinterp = interpolate.interp1d(gradient.loc[:,'Time (m)'], gradient.loc[:,'Flow'])

gradient.loc[:,'column-%ACN'] = percinterp(gradient.loc[:,'column-timepoint'])
#column-flow not used for actual time flow measurements, it should only be used to determine volumes
gradient.loc[:,'column-flow'] = flowinterp(gradient.loc[:,'column-timepoint'])

gradient.loc[:,'column-tvolume'] = integrate.cumtrapz(gradient.loc[:,'column-flow'], gradient.loc[:,'column-timepoint'], initial=0)
gradient.loc[:,'column-bflow'] = gradient.loc[:,'column-flow'] * gradient.loc[:,'column-%ACN'] / 100
gradient.loc[:,'column-aflow'] = gradient.loc[:,'column-flow'] * (100 - gradient.loc[:,'column-%ACN']) / 100

gradient.loc[:,'column-bvol'] = integrate.cumtrapz(gradient.loc[:,'column-bflow'], gradient.loc[:,'column-timepoint'], initial=0) #nL
gradient.loc[:,'column-avol'] = integrate.cumtrapz(gradient.loc[:,'column-aflow'], gradient.loc[:,'column-timepoint'], initial=0) #nL

fig, ax = plt.subplots()
gradient.plot.line(x='Time (m)', y='%ACN', ax=ax, color='cyan')
gradient.plot.line(x='Time (m)', y='column-%ACN', ax=ax, color='fuchsia')
plt.show()
