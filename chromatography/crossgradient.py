import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import pandas as pd
import os

#for more https://matplotlib.org/stable/tutorials/introductory/customizing.html
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
chexes = ['#ffffff',
        '#e85d58',
        '#b88cfa',
        '#f5972c',
        '#2ded8d',
        '#4bc8f2',
        '#ea68f2',
        ]
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', chexes)
#n = 0
#for c in chexes:
#    plt.plot([n, n+1], [n, n+1], color=c)
#    n += 1
#plt.show()

#run plans
#make another file for expanding the volume of a certain gradient in time.
#check pressurerecording.py on the plus to see if it can add a simple +1 index column to keep every tesseract prediction in order.
#add number of steps in method to parameter output

#Parameters
save = False
save = True
folder = '/home/sfo/store/flowcharacterizations/round5/gradients/'
fileroot = 'E5_CG_high_tw1_low_pw1'
deadvolume = 3030 #nL

npoints = 1000 #number of points to interpolate

startingpressure = 350
endingpressure = 150

gradienttime = 180 #minutes
minbpercentage = 0.02
maxbpercentage = 0.4

asolventpercent = 1
bsolventpercent = 0.8

pressuredomainscheme = 'low'
pressuredomainschemeweighting = 1

timedomainscheme = 'high'
timedomainschemeweighting = 1

columnlength = 200 #mm
columninnerdiameter = 100 #um
particlediameter = 1.8 #um

epsilon = 0.458 #Interparticle porosity. Porosity is defined as the ratio of volume of pores to the total volume of the particle. Interparticle refers to the porosity between particles. This is the percentage of void volume inside the stationary phase.

temperature = 25 #celsius

flush = pd.DataFrame([
    [0.17, 85, 300],
    [10, 100, 300],
    [10.17, 2, 300],
    [21, 2, 300]],
    columns=['Time', '%B', 'Flow'])

flush.loc[:,'%ACN'] = (flush.loc[:,'%B'] / 100) * bsolventpercent + (1 - maxbpercentage) * (1 - asolventpercent)

#End parameters

def humanize_time(secs):
    mins, secs = divmod(secs, 60)
    return '%02d:%02d' % (mins, secs)

def vsolver(a, b, c, m):
    return (b - np.sqrt((b**2) + (m**2)*(c**2) + 2*m*b*c + 2*m*a)) / (-1*m)

def negligible_difference(a):
    while (np.diff(a) == 0).any():
        a[1:][np.diff(a) == 0] += 0.00000001
    return a

def rescale(array, dmin, dmax):
    omin = array.min()
    omax = array.max()
    return (dmax-dmin)/(omax-omin)*(array-omax)+dmax

def domain_weighting(npoints, descent='linear', weighting=1):
    #weighting values for linear descent behave differently, they change the slope of the line while retaining the position of the central point
    #equal descent does not consider weighting
    #other weighting values:
    #weighting values < 1 are averaged with an unweighted linear weighting
    #weighting values = 1 are unchanged
    #weighting values > 1 only work for linear gradients, however, increasing npoints gives this desired effect.

    if weighting > 1 and descent != 'linear' and descent != 'plot':
        raise ValueError('weighting values > 1 only work for linear gradients, however, increasing npoints with weighting=1 gives this desired effect.')
    
    if descent == 'equal':
        return np.ones(npoints)
    
    x = np.arange(npoints) + 1
    
    linear = -x + np.abs(x).max() + 1

    if weighting != 1:
        wlinear = linear * weighting + linear.mean() - (linear * weighting).mean()

        if descent == 'linear':
            return wlinear
    
    if descent == 'linear':
        return linear
    
    high = -1 * np.geomspace(x.min(), x.max(), len(x))
    high += np.abs(high).max() + 1
    low = np.flip(-1 * high + np.abs(high).max() + 1)
    
    if weighting < 1:
        weights = np.asarray([1 - weighting, weighting])
        whigh = np.average([linear, high], weights=weights, axis=0)
        wlow = np.average([linear, low], weights=weights, axis=0)
        
        if descent == 'high':
            return whigh

        if descent == 'low':
            return wlow
    
    if descent == 'high':
        return high

    if descent == 'low':
        return low

    if npoints % 2:
        npoints += 1
    
    halfval = npoints // 2
    
    lhhigh = -1 * np.geomspace(x[:halfval].min(), x[:halfval].max() - 1, halfval)
    lhhigh += np.abs(high).max() + 1
    lhlow = np.flip(-1 * lhhigh + np.abs(lhhigh).max())
    highlow = np.hstack((lhhigh, lhlow))
    lhlow = -1 * np.geomspace(x[:halfval].min(), x[:halfval].max() + 1, halfval)
    lhlow += np.abs(lhlow).max()
    lhhigh = np.flip(-1 * lhlow) + (np.abs(lhlow).max() - 1) * 2 + 2
    lowhigh = np.hstack((lhhigh, lhlow))

    if weighting < 1:
        whighlow = np.average([linear, highlow], weights=weights, axis=0)
        wlowhigh = np.average([linear, lowhigh], weights=weights, axis=0)

        if descent == 'highlow':
            return highlow

        if descent == 'lowhigh':
            return lowhigh

    if descent == 'highlow':
        return highlow

    if descent == 'lowhigh':
        return lowhigh

    if descent == 'plot':
        wlabel = ' '.join((str(weighting), 'weighting'))
        plt.plot(x, linear, label='linear')
        plt.plot(x, high, label='high')
        plt.plot(x, low, label='low')
        plt.plot(x, highlow, label='high-low')
        plt.plot(x, lowhigh, label='low-high')
        plt.plot(x, np.repeat(linear.mean(), len(x)), label='equal')
        if weighting != 1:
            plt.plot(x, wlinear, label=' '.join((wlabel, 'linear')))
        if weighting < 1:
            plt.plot(x, whigh, label=' '.join((wlabel, 'high')))
            plt.plot(x, wlow, label=' '.join((wlabel, 'low')))
            plt.plot(x, whighlow, label=' '.join((wlabel, 'high-low')))
            plt.plot(x, wlowhigh, label=' '.join((wlabel, 'low-high')))
        plt.legend(bbox_to_anchor=(1,1))
        plt.title(' '.join(('Interpolation of', str(npoints))))
        plt.show()
        return

    raise ValueError('Haven\'t invented that one yet')

columnlengthcm = columnlength / 10
columnradius = columninnerdiameter / 2 #um
columnradiuscm = columnradius / 1000 / 10
particlediametercm = particlediameter / 1000 / 10

#scale the domain weight to starting/end
pressuredomain = domain_weighting(npoints, descent=pressuredomainscheme, weighting=pressuredomainschemeweighting) #cannot be equal
separationpressurebar = rescale(pressuredomain, endingpressure, startingpressure) #use of this assumes the start/end are the max/mins

separationpressureubar = separationpressurebar * 1000 * 1000

percentbsolvent = np.linspace(minbpercentage, maxbpercentage, npoints)
percentacetonitrile = percentbsolvent * bsolventpercent + (1 - minbpercentage) * (1 - asolventpercent)

abstemp = temperature + 273.15 #kelvin

centipoise = np.exp(percentacetonitrile * (-3.476 + (726 / abstemp)) + (1 - percentacetonitrile) * (-5.414 + (1566 / abstemp)) + percentacetonitrile * (1 - percentacetonitrile) * (-1.762 + (929 / abstemp))) #viscosity

poise = centipoise / 100 #g/(cm-sec)

gradientflowratemlmin = (separationpressureubar * epsilon) * ((particlediametercm**2) * (epsilon**2) * np.pi * (columnradiuscm**2) * 60) / (180 * poise * columnlengthcm * (1 - epsilon)**2) #mL/min
gradientflowratenlmin = gradientflowratemlmin * 1000 * 1000 #nL/min

#These are the different types of weights I'm trying
#timedomainweight = -1 * (np.arange(len(gradientflowratenlmin)) + 1)**2 #edec.method
#timedomainweight = -1 * np.log(np.arange(len(gradientflowratenlmin)) + 1)**3 #ldec.method
#timedomainweight = np.arange(len(gradientflowratemlmin)) + 1 #simic
#timedomainweight = np.flip(np.arange(len(gradientflowratemlmin)) + 1) #simdec.method
#timedomainweight = 1 / gradientflowratenlmin #simrec.method

#this can make it weird sometimes
#timedomainweight = timedomainweight / timedomainweight.sum()


#Round >2 weights
#timedomainweight = np.flip(np.arange(len(gradientflowratenlmin)) + 1) #simdec.method

#Round >5 weights
timedomainweight = domain_weighting(npoints, descent=timedomainscheme, weighting=timedomainschemeweighting)

initialdeadtime = 1 / (gradientflowratenlmin[0] / deadvolume) #min
leftovertime = gradienttime - initialdeadtime - flush.loc[:,'Time'].max()
timebins = timedomainweight / timedomainweight.sum() * leftovertime
timedomain = np.cumsum(timebins)

volume = integrate.cumulative_trapezoid(gradientflowratenlmin, timedomain, initial=0)
#at first it seems like this plot is wrong based on the time vs flowrate plot above it, but the y-axis offset is so high that the large majority of the volume comes from the fast that the flowrate is > 300 nL/min, or whatever volume the baseline is. If the bottom of the curve was closer to zero, the plot of auc vs time would look more intuitive.


columnvolume = columnlengthcm * np.pi * columnradiuscm **2 #mL
columnvolumenl = columnvolume * 1000 * 1000 #nL


volumeportions = np.diff(volume)
volumeportions = np.insert(volumeportions, 0, volumeportions[0])

gflow = gradientflowratenlmin.copy()
ftime = timedomain.copy()
ftime += initialdeadtime

gflow = np.insert(gflow, 0, gflow[0])
ftime = np.insert(ftime, 0 ,0)

#cheap hack to ignore unchanging flowrates that give zero-slopes, flowrates aren't that accurate anyways
gflow = negligible_difference(gflow)

#areas = integrate.cumulative_trapezoid(yvals, xvals, initial=0)
gradientvolume = integrate.cumulative_trapezoid(gflow, ftime, initial=0)

#sfind = (areas - deadvolume).reshape(-1,1)
volumeshift = (gradientvolume - deadvolume).reshape(-1,1)
#sfind[sfind < 0] = 0
volumeshift[volumeshift < 0] = 0

slopes = np.diff(gflow) / np.diff(ftime)

#sfbool = areas < sfind
shiftlocation = gradientvolume < volumeshift

#sfbi = sfbool.sum(axis=1) - 1
integrationindex = shiftlocation.sum(axis=1) - 1
#sfbi[sfbi < 0] = 0
iii = integrationindex >= 0

slopeofinterest = slopes[integrationindex[iii]]
priorarea = gradientvolume[integrationindex[iii]]
specificarea = volumeshift[iii].flatten() - priorarea
xvofinterest = ftime[integrationindex[iii]]
yvofinterest = gflow[integrationindex[iii]]

intercepts = yvofinterest - (xvofinterest * slopeofinterest)
atime = vsolver(specificarea, intercepts, xvofinterest, slopeofinterest)

ff = pd.DataFrame()
ff.loc[:,'Flow'] = gflow
ff.loc[:,'Time'] = ftime

af = pd.DataFrame()
af.loc[:,'%B'] = np.insert(percentbsolvent, 0, percentbsolvent[0])
af.loc[:,'%ACN'] = np.insert(percentacetonitrile, 0, percentacetonitrile[0])
af.loc[:,'Time'] = np.insert(atime, 0, 0)

ef = pd.concat([ff, af])
ef.sort_values('Time', inplace=True)
ef.interpolate(limit_direction='both', inplace=True)
ef.drop_duplicates('Time', inplace=True)
ef.reset_index(drop=True, inplace=True)
ef.loc[:,'%B'] *= 100
ef.loc[:,'%ACN'] *= 100
ef.drop_duplicates('%B', keep='last', inplace=True)

ecs = ef.loc[:,'%B'].unique()
ecs = ecs[ecs % 1 == 0]
cvals = np.linspace(minbpercentage * 100, maxbpercentage * 100, int(maxbpercentage * 100 - minbpercentage * 100) + 1)
cvals = cvals[~(cvals == ecs.reshape(-1,1)).any(axis=0)]

comb = pd.DataFrame(cvals, columns=['%B'])
ef = pd.concat([ef, comb])
ef.sort_values('%B', inplace=True)
ef.interpolate(inplace=True)

keepinds = ef.loc[:,'%B'] % 1 == 0
nef = ef.loc[keepinds].copy()
nef.reset_index(drop=True, inplace=True)

flush.loc[:,'Time'] += initialdeadtime + leftovertime
nef = pd.concat([nef, flush])
nef.reset_index(drop=True, inplace=True)

nflow = nef.loc[:,'Flow'].to_numpy()
ntime = nef.loc[:,'Time'].to_numpy()

nflow = np.insert(nflow, 0, nflow[0])
ntime = np.insert(ntime, 0 ,0)

nflow = negligible_difference(nflow)

ngradientvolume = integrate.cumulative_trapezoid(nflow, ntime, initial=0)
nvolumeshift = (ngradientvolume - deadvolume).reshape(-1,1)
nvolumeshift[nvolumeshift < 0] = 0
nslopes = np.diff(nflow) / np.diff(ntime)
nshiftlocation = ngradientvolume < nvolumeshift
nintegrationindex = nshiftlocation.sum(axis=1) - 1
niii = nintegrationindex >= 0

nslopeofinterest = nslopes[nintegrationindex[niii]]
npriorarea = ngradientvolume[nintegrationindex[niii]]
nspecificarea = nvolumeshift[niii].flatten() - npriorarea
nxvofinterest = ntime[nintegrationindex[niii]]
nyvofinterest = nflow[nintegrationindex[niii]]

nintercepts = nyvofinterest - (nxvofinterest * nslopeofinterest)
atime = vsolver(nspecificarea, nintercepts, nxvofinterest, nslopeofinterest)
atime = atime[~np.isnan(atime)]
atime = np.insert(atime, 0, np.zeros(niii.size - atime.size - 1))

atime = negligible_difference(atime)

ref = nef.copy()
aef = pd.DataFrame(atime, columns=['Time'])
ref = pd.concat([ref, aef])
ref.sort_values('Time', inplace=True)
ref.interpolate(limit_direction='both', inplace=True)

rinds = np.argwhere(ref.loc[:,'Time'].to_numpy() == atime.reshape(-1,1))[:,1]

nef.loc[:,'ic-%B'] = ref.iloc[rinds, 2]
nef.loc[:,'ic-%ACN'] = (nef.loc[:,'ic-%B'].to_numpy() / 100) * bsolventpercent + (1 - minbpercentage) * (1 - asolventpercent)


flowrate = nef.loc[:,'Flow'].to_numpy() #nL/min
flowratemlsec = flowrate / 60 / 1000 / 1000 #mL/sec
flowratemlmin = flowrate / 1000 / 1000

percentacetonitrile = nef.loc[:,'ic-%ACN'].to_numpy() / 100

centipoise = np.exp(percentacetonitrile * (-3.476 + (726 / abstemp)) + (1 - percentacetonitrile) * (-5.414 + (1566 / abstemp)) + percentacetonitrile * (1 - percentacetonitrile) * (-1.762 + (929 / abstemp))) #viscosity

poise = centipoise / 100 #g/(cm-sec)

pressure = (flowratemlmin * 180 * poise * columnlengthcm * (1 - epsilon)**2) / ((particlediametercm**2) * (epsilon**2) * np.pi * (columnradiuscm**2) * 60) / epsilon

pressurebar = pressure / 1000 / 1000

nef.loc[:,'Pressure (bar)'] = pressurebar

nef.loc[:,'Duration'] = nef.loc[:,'Time'].diff() * 60
nef.loc[0,'Duration'] = nef.iloc[1].loc['Duration']
nef.loc[:,'Duration'] = nef.loc[:,'Duration'].apply(lambda x: humanize_time(x))
nef.loc[:,'Flow'] = nef.loc[:,'Flow'].round().to_numpy()

nef.loc[:,'aflow'] = (1 - nef.loc[:,'%B'].to_numpy() / 100) * nef.loc[:,'Flow'].to_numpy()
nef.loc[:,'bflow'] = nef.loc[:,'%B'].to_numpy() / 100 * nef.loc[:,'Flow'].to_numpy()
nef.loc[:,'avol'] = integrate.cumulative_trapezoid(nef.loc[:,'aflow'].to_numpy(), nef.loc[:,'Time'].to_numpy(), initial=0)
nef.loc[:,'bvol'] = integrate.cumulative_trapezoid(nef.loc[:,'bflow'].to_numpy(), nef.loc[:,'Time'].to_numpy(), initial=0)
nef.loc[:,'tvol'] = integrate.cumulative_trapezoid(nef.loc[:,'Flow'].to_numpy(), nef.loc[:,'Time'].to_numpy(), initial=0)

#I don't think this one is correct, see below?
#nef.loc[:,['ic-avol', 'ic-bvol', 'ic-tvol']] = nef.loc[:,['avol', 'bvol', 'tvol']].to_numpy() - deadvolume

#is this more appropriate for determining ic-volumes?
#gradient.loc[:,'ic-timepoint'] = ic_volumefinder(gradient.loc[:,'retention time (min)'], gradient.loc[:,'Flow'], deadvolume)
#gradient.loc[:,'ic-tvol'] = integrate.cumulative_trapezoid(gradient.loc[:,'flow'], gradient.loc[:,'ic-timepoint'], initial=0)
#gradient.loc[:,'ic-bflow'] = gradient.loc[:,'flow'] * gradient.loc[:,'ic-%B'] / 100
#gradient.loc[:,'ic-aflow'] = gradient.loc[:,'flow'] * (100 - gradient.loc[:,'ic-%B']) / 100
#
#gradient.loc[:,'ic-bvol'] = integrate.cumulative_trapezoid(gradient.loc[:,'ic-bflow'], gradient.loc[:,'ic-timepoint'], initial=0) #nL
#gradient.loc[:,'ic-avol'] = integrate.cumulative_trapezoid(gradient.loc[:,'ic-aflow'], gradient.loc[:,'ic-timepoint'], initial=0) #nL

if save:
    if not os.path.isdir(folder):
        os.makedirs(folder)
    gloc = ''.join((folder, fileroot, '.method.csv'))
    nef.to_csv(gloc)

    pf = pd.DataFrame(columns=['Values'])
    pf.loc['folder'] = folder
    pf.loc['file root'] = fileroot
    pf.loc['dead volume (nL)'] = deadvolume
    pf.loc['# interpolated points'] = npoints
    pf.loc['starting pressure'] = startingpressure
    pf.loc['ending pressure'] = endingpressure
    pf.loc['method length (min)'] = gradienttime
    pf.loc['starting %B'] = minbpercentage
    pf.loc['ending %B'] = maxbpercentage
    pf.loc['starting %ACN'] = minbpercentage * bsolventpercent + (1 - minbpercentage) * (1 - asolventpercent)
    pf.loc['ending %ACN'] = maxbpercentage * bsolventpercent + (1 - maxbpercentage) * (1 - asolventpercent)
    pf.loc['column length (mm)'] = columnlength
    pf.loc['column ID (um)'] = columninnerdiameter
    pf.loc['particle diameter (um)'] = particlediameter
    pf.loc['interparticle porosity'] = epsilon
    pf.loc['temperature (C)'] = temperature
    pf.loc['time domain'] = timedomainscheme
    pf.loc['pressure domain'] = pressuredomainscheme
    pf.loc['time domain weighting'] = timedomainschemeweighting
    pf.loc['pressure domain weighting'] = pressuredomainschemeweighting
    pf.loc['total volume'] = nef.loc[:,'tvol'].max()
    pf.loc['volume A'] = nef.loc[:,'avol'].max()
    pf.loc['volume B'] = nef.loc[:,'bvol'].max()
    pf.index.name = 'Parameter'

    ploc = ''.join((folder, fileroot, '.parameters.csv'))
    pf.to_csv(ploc)


fig, ax = plt.subplots()
ax.plot(timedomain, separationpressurebar+initialdeadtime)
nef.plot.line(x='Time', y='Pressure (bar)', ax=ax, style='--', dashes=(11,9))
ax.get_legend().remove()

dotted = plt.Line2D((0,1),(0,0), color='k', linestyle='--', dashes=(4,4))
straight = plt.Line2D((0,1),(0,0), color='k', linestyle='-')

ax.legend(handles=[straight, dotted], labels=['Target', 'Result'], bbox_to_anchor=(1.31,1))
plt.ylabel('Pressure (bar)')

plt.show()

fig, ax = plt.subplots()

axn = ax.twinx()
ef.plot.line(x='Time', y='Flow', color='lightpink', ax=ax, linewidth=2)
ef.plot.line(x='Time', y='%B', color='lavender', ax=axn, linewidth=2)

nef.plot.line(x='Time', y='Flow', color='lightslategray', ax=ax, style='--', dashes=(11,9), linewidth=2)
nef.plot.line(x='Time', y='%B', color='orangered', ax=axn, style='--', dashes=(11,9), linewidth=2, alpha=0.5)

ax.set_ylabel('nL/min')
axn.set_ylabel('% B')


ax.spines['left'].set_color('lightpink')
ax.spines['right'].set_color('lavender')
axn.spines['left'].set_linestyle((0, (10,17)))
axn.spines['right'].set_linestyle((0, (10,17)))
axn.spines['left'].set_color('lightslategray')
axn.spines['right'].set_color('orangered')
axn.spines['right'].set_alpha(0.5)

ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
axn.spines['right'].set_linewidth(2)
axn.spines['left'].set_linewidth(2)

ax.get_legend().remove()
axn.get_legend().remove()

dotted = plt.Line2D((0,1),(0,0), color='k', linestyle='--', dashes=(4,4))
straight = plt.Line2D((0,1),(0,0), color='k', linestyle='-')

ax.legend(handles=[straight, dotted], labels=['Target', 'Result'], bbox_to_anchor=(1.31,1))
plt.show()

fig, ax = plt.subplots()

nef.plot.line(x='Time', y='avol', color='purple', ax=ax)
nef.plot.line(x='Time', y='bvol', color='green', ax=ax)

plt.show()

