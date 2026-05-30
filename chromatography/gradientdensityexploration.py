import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats, interpolate, integrate
from statsmodels.nonparametric.smoothers_lowess import lowess
import os
import sympy as sp
import concurrent.futures
from time import time
import warnings
import re
plt.rcParams["figure.dpi"] = 100
warnings.filterwarnings("ignore")
pd.set_option('display.float_format', lambda x: '%.3f' % x)

folder = '/store/flowcharacterizations/round4/plus/'

gradientsfolder = ''.join((folder, 'gradients/'))
peakfolder = ''.join((folder, 'peaks/'))

files = os.listdir(peakfolder)
files = [i for i in files if i.endswith('.peaks.csv')]

save = False

deadvolume = 3225 #nL
columnvolume = 530 #nL
endgradientval = 30

def vsolver(a, b, c, m):
    return (b - np.sqrt((b**2) + (m**2)*(c**2) + 2*m*b*c + 2*m*a)) / (-1*m)

def ic_volumefinder(xvals, yvals, deadvolume):
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

def peripheraltransform(z, x, nx):
    zavals = np.diff(integrate.cumtrapz(z, x, initial=0))
    hvals = []
    for i, a in enumerate(zavals):
        if i == 0:
            w = nx[i+1] - nx[i]
            h = a / w
            hvals.extend([h, h])
        else:
            w = nx[i+1] - nx[i] #width
            oh = hvals[-1] #old height
            oa = w * oh #old area
            ta = oa - a #triangle area
            th = 2 * ta / w #triangle height
            h = oh - th #height
            hvals.append(h)
    return np.asarray(hvals)

def squared_solver(xvals, yvals, limit):
    #upperbound = xvals.max()
    lowerbound = xvals.min()
    valareas = integrate.cumtrapz(yvals, xvals)
    a = valareas[-1]
    g = (a * 3) / (xvals[limit]**3 - lowerbound**3) #this is specific to the equation g*x**2
    rnx = (((3 * valareas) / g) + (xvals[0] ** 3)) ** (1/3) #also specific to the equation g*x**2
    rnx = np.append(xvals[0], rnx)
    eny = g * rnx ** 2
    return rnx, eny

def area_flatten(xvals, yvals, limit):
    area = integrate.trapz(yvals, xvals)
    time = xvals[limit] - xvals.min()
    height = area / time
    avals = np.diff(integrate.cumtrapz(yvals, xvals, initial=0))
    newlengths = avals / height
    return np.append(0, np.cumsum(newlengths))

def flowrate_elucidation(xarr, yarr, windowsize=20, repeats=2):
    #xarr = cdrs.loc[:,'g*x**2 x transform (min)'].to_numpy()
    #yarr = cdrs.loc[:,'ic-bvol'].to_numpy()
    #windowsize = 20
    #repeats = 2
    nxarr = xarr.copy()
    nyarr = yarr.copy()
    arr = xarr.copy()
    if not windowsize % 2:
        windowsize += 1
#the window could be data point-based or x-value based (with or without a data point limit).
    windowsub = np.linspace(-windowsize/2, windowsize/2, windowsize).astype(int)
    warr = np.repeat(np.arange(len(arr)).reshape(-1,1), windowsize, axis=1)
    warr += windowsub
    warr[warr < 0] = 0
    warr[warr >= len(warr)] = len(warr) - 1
    
    for n in range(repeats):
        nyarr = nyarr[warr].mean(axis=1)
        nxarr = nxarr[warr].mean(axis=1)
    
    #fig, ax = plt.subplots(figsize=(10,6))
    #cdrs.plot.line(x='g*x**2 x transform (min)', y='ic-bvol', label='transformed', color='purple', ax=ax, alpha=0.3)
    #plt.plot(nxarr, nyarr, '-', label='both new', color='orange', alpha=0.3)
    #plt.legend()
    #plt.show()
    
#oflows = np.diff(yarr)/np.diff(xarr)
    flows = np.diff(nyarr)/np.diff(nxarr)
    spline = interpolate.UnivariateSpline(nxarr[1:], flows)
    sy = spline(xarr)
    sy[sy <= 0] = sy[np.argwhere(sy > 0).min()]
#lx, ly = lowess(flows, nxarr[1:], frac=0.1).transpose()
    sy = np.maximum.accumulate(sy)
    
#plt.plot(xarr[1:], oflows, '-')
    plt.plot(nxarr[1:], flows, '.')
    plt.plot(xarr, sy, '-')
    plt.show()
    
    return sy, integrate.cumtrapz(sy, xarr, initial=0)

def flowfinder(flow, yt, endbval):
    tn = 1
    while True:
        if (flow / ((tn / yt) + flow)).max() <= (endbval / 100):
            out = flow / ((tn / yt) + flow) * 100
            out[-1] = endbval
            break
        tn += 1
    return out


def humanize_time(secs):
    mins, secs = divmod(secs, 60)
    return '%02d:%02d' % (mins, secs)

for fr in files:
    #fr = files[-1]

    ftitle = '_'.join((fr.split('.')[0].split('_')[1:]))
    f = ''.join((peakfolder, fr))
    fbase = fr.split('.')[0]
    enfs = pd.read_csv(f)

    gname = ''.join((gradientsfolder, ''.join(('1s-dyn-', fbase.split('_')[1].replace('-dyn', ''))), '.method.csv'))
    try:
        gradient = pd.read_csv(gname)
    except FileNotFoundError:
        switch = True
        gname = ''.join((gradientsfolder, '_'.join((fbase.split('_')[1:])), '.method.csv'))
        gradient = pd.read_csv(gname)

    gradient.rename({'Time (m)':'retention time (min)'}, axis=1, inplace=True)

    try:
        gradient.loc[:,'retention time (min)'] = gradient.loc[:,'retention time (min)'].astype(float)
    except KeyError:
        gradient.loc[:,'Time'] = pd.to_datetime(gradient.loc[:,'Time'])
        gradient.loc[:,'retention time (min)'] = gradient.loc[:,'Time'].apply(lambda x: sum([x.second / 60, x.minute, x.hour * 60])).astype(float)

    rtimes = enfs.loc[:,'retention time (min)'].copy().to_frame()
    rtimes.index += gradient.index.max() + 1
    gradient = pd.concat([gradient, rtimes])

    gradient.drop(['Duration', 'timediff'], axis=1, inplace=True)
    gradient.sort_values('retention time (min)', inplace=True)
    gradient.interpolate(limit_direction='both', inplace=True)

    gradient.loc[:,'Total Volume'] = integrate.cumtrapz(gradient.loc[:,'Flow'], gradient.loc[:,'retention time (min)'], initial=0)
    gradient.loc[:,'Flow B at Interval'] = gradient.loc[:,'Flow'] * gradient.loc[:,'%ACN'] / 100
    gradient.loc[:,'Flow A at Interval'] = gradient.loc[:,'Flow'] * (100 - gradient.loc[:,'%ACN']) / 100
    gradient.loc[:,'Cumulative Volume B'] = integrate.cumtrapz(gradient.loc[:,'Flow B at Interval'], gradient.loc[:,'retention time (min)'], initial=0) #nL
    gradient.loc[:,'Cumulative Volume A'] = integrate.cumtrapz(gradient.loc[:,'Flow A at Interval'], gradient.loc[:,'retention time (min)'], initial=0) #nL

#ic means in-column
    gradient.loc[:,'ic-timepoint'] = ic_volumefinder(gradient.loc[:,'retention time (min)'], gradient.loc[:,'Flow'], deadvolume)
#some values come out just below the minimum timepoint because of computational precision differences, but for the ones that do it's going to be the same conditions as the initial timepoint, so this is fixed here
    gradient.loc[(gradient.loc[:,'ic-timepoint'] < gradient.loc[:,'retention time (min)'].min()), 'ic-timepoint'] = gradient.loc[:,'retention time (min)'].min()
#the volumesolver already takes the flow and volume into consideration, so simply solving for whatever was pumping at the solved timepoint is a legitemate measure for the %ACN in the column at a given time.
    percinterp = interpolate.interp1d(gradient.loc[:,'retention time (min)'].to_numpy(), gradient.loc[:,'%ACN'].to_numpy())
    flowinterp = interpolate.interp1d(gradient.loc[:,'retention time (min)'], gradient.loc[:,'Flow'])

#the tons of 0's at the initial ic values are fine, it just accumulates volume starting when the analytes hit the column
    gradient.loc[:,'ic-%ACN'] = percinterp(gradient.loc[:,'ic-timepoint'])
#flow not used for actual time flow measurements, it should only be used to determine volumes
    gradient.loc[:,'flow'] = flowinterp(gradient.loc[:,'ic-timepoint'])

    gradient.loc[:,'ic-tvol'] = integrate.cumtrapz(gradient.loc[:,'flow'], gradient.loc[:,'ic-timepoint'], initial=0)
    gradient.loc[:,'ic-bflow'] = gradient.loc[:,'flow'] * gradient.loc[:,'ic-%ACN'] / 100
    gradient.loc[:,'ic-aflow'] = gradient.loc[:,'flow'] * (100 - gradient.loc[:,'ic-%ACN']) / 100

    gradient.loc[:,'ic-bvol'] = integrate.cumtrapz(gradient.loc[:,'ic-bflow'], gradient.loc[:,'ic-timepoint'], initial=0) #nL
    gradient.loc[:,'ic-avol'] = integrate.cumtrapz(gradient.loc[:,'ic-aflow'], gradient.loc[:,'ic-timepoint'], initial=0) #nL

    gradient.drop_duplicates('retention time (min)', inplace=True)
    gradient.set_index('retention time (min)', inplace=True)

    for gc in gradient.columns:
        enfs.loc[:,gc] = gradient.loc[enfs.loc[:,'retention time (min)'], gc].tolist()

#setting up the testing infrastructure for different metrics/gridsearches is more important than any single parameter
#maybe the density estimate doesn't matter per se.. it just matters that it's the same on both sides of the prediction/goal thing
#therefore try it with scipy first

#bandwidths = 10 ** np.linspace(1, 10, 5)
#bandwidths = np.linspace(14,17)
#grid = GridSearchCV(KernelDensity(kernel='gaussian', rtol=0.5), {'bandwidth': bandwidths}, cv=20)
#gtrain = enfs.loc[:,'retention time (min)'].to_numpy().reshape(-1,1)
#grid.fit(gtrain)
#kde = grid.best_estimator_
#
#kde = KernelDensity(kernel='gaussian', rtol=0.01, bandwidth=16, leaf_size=2)
#kde.fit(gtrain)
#
#enfs.loc[:,'skldensity'] = np.exp(kde.score_samples(gtrain))
#
#enfs.loc[:,'frequency'] = stats.gaussian_kde(enfs.loc[:,'retention time (min)'].to_numpy())(enfs.loc[:,'retention time (min)'].to_numpy())
#enfs.plot.scatter(x='retention time (scan)', y='frequency')
#plt.show()
#
#x_d = np.linspace(enfs.index.min(), enfs.index.max(), 1000)
#density = sum(stats.norm(xi).pdf(x_d) for xi in enfs.index.to_numpy())
#plt.fill_between(x_d, density, alpha=0.5)
#plt.show()

#aggdict = {i:'max' for i in enfs.columns if 'ic' in i or not i.islower()}
#aggdict['frequency'] = 'count'
#etf = enfs.reset_index().groupby('Time (m)').agg(aggdict)

#should I aggregate this by rounded %ACN lol?
#For the gradient modificaitons: %ACN stays the same at respective volumes, and total volume stays constant for the gradient.
#Number of peaks per volume is the factor used to weight the time domain
#The time domain gets weighted by the volume in order to derive flowrate?
#Or - You can do the same with the number of peaks per %ACN and weight that while keeping total volume constant, once %B is applied to the time axis, then you derive the flow?

#different areas to use for reshaping peak elution:
#density
#raw counts of peaks
#raw counts of peaks resampled via time to better remove 0-points
#^I think the time frequency should be resampled until there are no points with 0 peaks - although this might be hard to put out as an open-source piece of software because people will break the hell out of that no problem. Maybe just no contiguous 0s? Still wouldn't be bulletproof.

    ecols = enfs.columns.tolist()
    ecols = {i:'max' for i in ecols if 'retention time (min)' not in i and 'mass channel' not in i}
    ecols['mass channel'] = 'count'
    cd = enfs.groupby('retention time (min)').agg(ecols)
    cd.rename({'mass channel':'peak count'}, axis=1, inplace=True)

    #dm = stats.gaussian_kde(enfs.loc[:,'retention time (min)'].to_numpy(), weights=enfs.loc[:,'s base width (min)'].to_numpy())
    #cd.loc[:,'sc-density'] = dm(cd.index.to_numpy())

    cdrs = cd.copy()
    cdrs.loc[:,'retention time copy (min)'] = cdrs.index.copy()
    cdrs.index = pd.to_datetime(cdrs.index, unit='m')
    caggdict = {i:'max' for i in cdrs.columns}
    caggdict['peak count'] = 'sum'
    cdrs.resample('1m').agg(caggdict)
    cdrs = cdrs.resample('1min').agg(caggdict)

#cdgradientlimit = np.argwhere((cd.loc[:,'%ACN'] <= endgradientval).cumsum().diff()[1:].to_numpy() < 1).min()
    roughdeadtimeestimate = deadvolume / gradient.loc[:,'Flow'].mean()
    initialgradientlimit = np.argwhere((cdrs.loc[:,'%ACN'] <= endgradientval).cumsum().diff()[1:].to_numpy() < 1).min()
    limittime = cdrs.iloc[initialgradientlimit].loc['retention time copy (min)'] + roughdeadtimeestimate
    gradientlimit = (cdrs.loc[:,'retention time copy (min)'] - limittime).abs().argmin()

#cdng = pd.DataFrame()
#cdrsng = pd.DataFrame()
    fg = pd.DataFrame()

#cdng.loc[:,'flat area time (min)'] = area_flatten(cd.index.to_numpy(), cd.loc[:,'peak count'].to_numpy(), cdgradientlimit)
#cdng.loc[:,'density flat area time (min)'] = area_flatten(cd.index.to_numpy(), cd.loc[:,'sc-density'].to_numpy(), cdgradientlimit)
#cdrsng.loc[:,'flat area time (min)'] = area_flatten(cdrs.loc[:,'retention time copy (min)'].to_numpy(), cdrs.loc[:,'peak count'].to_numpy(), gradientlimit)
    fg.loc[:,'ic-time (min)'] = area_flatten(cdrs.loc[:,'retention time copy (min)'].to_numpy(), cdrs.loc[:,'peak count'].to_numpy(), gradientlimit)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #the squared gradients were using the wrong volume in the ic_volumefinder func, it was using fg, so all those gradients are usesless.
    #backtracking here to not generate new flows
    #these were then processed the way the bottom of this loop does before it saves the csvs
    cdrs.loc[:,'ic-flat time (min)'] = area_flatten(cdrs.loc[:,'retention time copy (min)'].to_numpy(), cdrs.loc[:,'peak count'].to_numpy(), gradientlimit)
    cdrs.loc[:,'ic-flow'] = cdrs.loc[:,'ic-bflow'] + cdrs.loc[:,'ic-aflow']
    cdrs.loc[:,'pump time'] = ic_volumefinder(cdrs.loc[:,'ic-flat time (min)'], cdrs.loc[:,'ic-flow'], deadvolume)
    cdrs.loc[:,'new flow'] = cdrs.loc[:,'ic-tvol'].diff() / cdrs.loc[:,'pump time'].diff()

    nrs = cdrs.copy()
    nrs.loc[:,'pump time copy (min)'] = nrs.loc[:,'pump time'].copy()
    nrs.loc[:,'pump time'] = pd.to_datetime(nrs.loc[:,'pump time'], unit='m')
    nrs.set_index('pump time', inplace=True)
    nrsdict = {i:'max' for i in nrs.columns}
    nrsdict['peak count'] = 'sum'
    nrs.resample('1m').agg(nrsdict)
    nrs = nrs.resample('10min').agg(nrsdict)
    nrs.loc[np.isinf(nrs.loc[:,'new flow']), 'new flow'] = np.nan
    nrs.interpolate(limit_direction='both', inplace=True)
    nrs.loc[:,'timediff'] = nrs.loc[:,'pump time copy (min)'].diff() * 60
    nrs.loc[nrs.loc[:,'timediff'].isnull(), 'timediff'] = 0
    nrs.loc[:,'Duration'] = nrs.loc[:,'timediff'].apply(lambda x: humanize_time(x))

    fnm = nrs.loc[:,['pump time copy (min)', ]]




#g, x, u, A = sp.symbols('g x u A')
#
#eqs = [g * (x+1) ** 2, g / (x+1) ** 2]

#def solvefunc(v):
#    return float(sp.solveset(v, domain=sp.S.Reals).args[0])
#
#def area_transform(eq, xvals, yvals):
#    upperbound = xvals.max()
#    lowerbound = xvals.min()
#    valareas = integrate.cumtrapz(yvals, xvals)
#    
#    teq = sp.Integral(eq, (x, lowerbound, upperbound)) - valareas[-1]
#    teqsolve = sp.solveset(teq.doit())
#    gval = float(teqsolve.args[0])
#    
#    neq = eq.subs(g, gval)
#    ineq = sp.Integral(neq, (x, lowerbound, u)) - A
#    ineqlambda = sp.lambdify(A, ineq.doit())
#
#    ineqvals = ineqlambda(valareas)
#    
#    neqxvals = [xvals[0]]
#    nvs = []
#    with concurrent.futures.ProcessPoolExecutor(8) as executor:
#        for v in ineqvals:
#            nvs.append(executor.submit(solvefunc, v))
#        for f in concurrent.futures.as_completed(nvs):
#            neqxvals.append(f.result())
#    
#    neqxvals = np.sort(neqxvals)
#    eqlambda = sp.lambdify(x, eq.subs(g, gval))
#    neqyvals = eqlambda(neqxvals)
#    return neqxvals, neqyvals
#
#for n, eq in enumerate(eqs):
#    mt = time()
#    eqxstring = ' '.join((str(eq), 'x transform (min)'))
#    eqystring = ' '.join((str(eq), 'y transform'))
#    
#    cd.loc[:,eqxstring], cd.loc[:,eqystring] = area_transform(eq, cd.index.to_numpy(), cd.loc[:,'peak count'].to_numpy())
#    cd.loc[:,' '.join(('density', eqxstring))], cd.loc[:,' '.join(('density', eqystring))] = area_transform(eq, cd.index.to_numpy(), cd.loc[:,'sc-density'].to_numpy())
#    cdrs.loc[:,eqxstring], cdrs.loc[:,eqystring] = area_transform(eq, cdrs.loc[:,'retention time copy (min)'].to_numpy(), cdrs.loc[:,'peak count'].to_numpy())
#    print(f'{eq} done', round(time() - mt, 4))

#cdng.loc[:,'g*x**2 x transform (min)'], cdng.loc[:,'g*x**2 y transform'] = squared_solver(cd.index.to_numpy(), cd.loc[:,'peak count'].to_numpy(), cdgradientlimit)
#
#cdng.loc[:,'g*x**2 x density transform (min)'], cdng.loc[:,'g*x**2 y density transform'] = squared_solver(cd.index.to_numpy(), cd.loc[:,'sc-density'].to_numpy(), cdgradientlimit)

    sg = pd.DataFrame()

    sg.loc[:,'ic-x transform (min)'], sg.loc[:,'ic-y transform'] = squared_solver(cdrs.loc[:,'retention time copy (min)'].to_numpy(), cdrs.loc[:,'peak count'].to_numpy(), gradientlimit)

#cdrsng.loc[:,'g*x**2 x density transform (min)'], cdrsng.loc[:,'g*x**2 y density transform'] = squared_solver(cdrs.loc[:,'retention time copy (min)'].to_numpy(), cdrs.loc[:,'peak count'].to_numpy(), gradientlimit)

    columnsofinterest = ['ic-bvol']

#cdng.loc[:,columnsofinterest] = cd.loc[:,columnsofinterest].to_numpy()
    fg.loc[:,columnsofinterest] = cdrs.loc[:,columnsofinterest].to_numpy()
    sg.loc[:,columnsofinterest] = cdrs.loc[:,columnsofinterest].to_numpy()

#fig, ax = plt.subplots(figsize=(10,6))
##cdng.plot.scatter(x='g*x**2 x transform (min)', y='g*x**2 y transform', ax=ax, label='raw', color='purple')
#cdrsng.plot.scatter(x='g*x**2 x transform (min)', y='g*x**2 y transform', ax=ax, label='resampled', color='green')
#plt.legend()
#plt.show()

    fig, ax = plt.subplots(figsize=(10,6))
#cdng.reset_index().plot.scatter(x='g*x**2 x transform (min)', y='ic-bvol', ax=ax, label='raw', color='green', alpha=0.3)
#cdng.reset_index().plot.scatter(x='g*x**2 x density transform (min)', y='ic-bvol', ax=ax, label='density-based', color='orange', alpha=0.3)
    cd.reset_index().plot.scatter(x='retention time (min)', y='ic-bvol', ax=ax, label='original', color='purple', alpha=0.3)
    sg.plot.scatter(x='ic-x transform (min)', y='ic-bvol', ax=ax, label='resampled raw', color='grey', alpha=0.3)
    plt.legend()
    plt.show()

#Using a maximum flowrate as a constraint for volume dispersion, as another constraint for the formula being generated.
#resample cd AFTER the equations have been solved

#savgol, lowess, and univariate spline all fail to smooth properly for ic-bvol. And ic-bvol NEEDS to be smoothed, otherwise it derives flowrates above what the pump can output.

#standard deviation of differences (and maybe another order of that, to the next derivative) can be used as a measure of smoothing
#^incorporate a standard deviation of differences (or relative error) of the smoothing to the original data to keep the transform legit

#here's the place to finally use that smoothing algo you've had in your back pocket, mean of n-number of spaces around a data point.
#Incorporate it into a gradient descent? To maximize smoothness obviously..

#current issues stand with the smoothness of the ic-bvol line when resampled the time points
#fig, ax = plt.subplots(figsize=(10,6))
#cdrsng.plot.line(x='g*x**2 x transform (min)', y='ic-bvol', label='transformed', color='green', ax=ax, alpha=0.3)
#cd.loc[:,'ic-bvol'].plot.line(ax=ax, color='purple', label='original', alpha=0.3)
#plt.legend()
#plt.show()

#fig, ax = plt.subplots(figsize=(10,6))
#cdrsng.plot.line(x='g*x**2 x transform (min)', y='ic-bvol', label='transformed', color='green', ax=ax, alpha=0.3)
#cdrsng.plot.line(x='retention time copy (min)', y='ic-bvol', ax=ax, color='purple', label='original', alpha=0.3)
#plt.legend()
#plt.show()
    
    sg.loc[:,'flow B'], sg.loc[:,'ic-volume B'] = flowrate_elucidation(sg.loc[:,'ic-x transform (min)'].to_numpy(), sg.loc[:,'ic-bvol'].to_numpy(), repeats=6)

#cdrsng.loc[:,'g*x**2 density flow B'], cdrsng.loc[:,'g*x**2 density volume B'] = flowrate_elucidation(cdrsng.loc[:,'g*x**2 x density transform (min)'].to_numpy(), cdrsng.loc[:,'ic-bvol'].to_numpy(), repeats=6)

    fg.loc[:,'flow B'], fg.loc[:,'ic-volume B'] = flowrate_elucidation(fg.loc[:,'ic-time (min)'].to_numpy(), fg.loc[:,'ic-bvol'].to_numpy())

#cdrsng.loc[:,'flat density flow B'], cdrsng.loc[:,'flat density volume B'] = flowrate_elucidation(cdrsng.loc[:,'density flat area time (min)'].to_numpy(), cdrsng.loc[:,'ic-bvol'].to_numpy())


    minbval = gradient.loc[:,'%ACN'].min()
#cdrsng.loc[:,'%ACN'] = np.linspace(minbval, endgradientval, cdrsng.shape[0])
#(1 / cdrsng.loc[:,'g*x**2 y transform'])

    maxflow = 700
    endbval = 95

    sg.loc[:,'%ACN'] = flowfinder(sg.loc[:,'flow B'].to_numpy(), sg.loc[:,'ic-y transform'].to_numpy(), endbval)
    fg.loc[:,'%ACN'] = flowfinder(fg.loc[:,'flow B'].to_numpy(), fg.loc[:,'flow B'].to_numpy(), endbval)
    

    sg.loc[:,'Flow'] = sg.loc[:,'flow B'] * (100 / sg.loc[:,'%ACN'])
    sg.loc[:,'flow A'] = sg.loc[:,'Flow'] - sg.loc[:,'flow B']

    fg.loc[:,'Flow'] = fg.loc[:,'flow B'] * (100 / fg.loc[:,'%ACN'])
    fg.loc[:,'flow A'] = fg.loc[:,'Flow'] - fg.loc[:,'flow B']

    sg.loc[:,'ic-volume A'] = integrate.cumtrapz(sg.loc[:,'flow A'].to_numpy(), sg.loc[:,'ic-x transform (min)'].to_numpy(), initial=0)
    fg.loc[:,'ic-volume A'] = integrate.cumtrapz(fg.loc[:,'flow A'].to_numpy(), fg.loc[:,'ic-time (min)'].to_numpy(), initial=0)

    sg.loc[:,'ic-total volume'] = sg.loc[:,'ic-volume A'].to_numpy() + sg.loc[:,'ic-volume B'].to_numpy()
    fg.loc[:,'ic-total volume'] = fg.loc[:,'ic-volume A'].to_numpy() + fg.loc[:,'ic-volume B'].to_numpy()

    sg.loc[:,'pump time'] = ic_volumefinder(sg.loc[:,'ic-x transform (min)'], sg.loc[:,'Flow'], deadvolume)
    fg.loc[:,'pump time'] = ic_volumefinder(fg.loc[:,'ic-time (min)'], fg.loc[:,'Flow'], deadvolume)

    sg.rename({'ic-bvol':'initial B volume estimate'}, axis=1, inplace=True)
    fg.rename({'ic-bvol':'initial B volume estimate'}, axis=1, inplace=True)

    sg.rename({'pump time': 'time (min)'}, axis=1, inplace=True)
    fg.rename({'pump time': 'time (min)'}, axis=1, inplace=True)

    sg.loc[:,'timediff'] = sg.loc[:,'time (min)'].diff() * 60
    fg.loc[:,'timediff'] = fg.loc[:,'time (min)'].diff() * 60

    sg.loc[0, 'timediff'] = 0
    fg.loc[0, 'timediff'] = 0

    sg.loc[:,'Duration'] = sg.loc[:,'timediff'].apply(lambda x: humanize_time(x))
    fg.loc[:,'Duration'] = fg.loc[:,'timediff'].apply(lambda x: humanize_time(x))

    sg.loc[:,'%ACN'] = sg.loc[:,'%ACN'].round()
    fg.loc[:,'%ACN'] = fg.loc[:,'%ACN'].round()

    sg.loc[:,'Flow'] = sg.loc[:,'Flow'].round()
    fg.loc[:,'Flow'] = fg.loc[:,'Flow'].round()

    sg = sg.drop_duplicates()
    fg = fg.drop_duplicates()

    sg = sg.loc[~(sg.loc[:,'Duration'] == '00:00')]
    fg = fg.loc[~(fg.loc[:,'Duration'] == '00:00')]

    nsgm = sg.loc[:,['Flow', 'time (min)', '%ACN', 'Duration']]
    nfgm = fg.loc[:,['Flow', 'time (min)', '%ACN', 'Duration']]

    
    fileroot = f.split('/')[-1].split('.')[0].split('_')[1]
    if switch:
        fileroot = ''.join((f.split('/')[-1].split('_')[1:3])).split('.')[0]

    nsgcsvloc = ''.join(('/store/flowcharacterizations/round4/plus/gradients/generated//', fileroot, '.squared.method.csv'))
    nfgcsvloc = ''.join(('/store/flowcharacterizations/round4/plus/gradients/generated/', fileroot, '.flat.method.csv'))

    if save:
        print(fileroot)
        nsgm.to_csv(nsgcsvloc)
        nfgm.to_csv(nfgcsvloc)


#keep the total ic-avol ~ the same too, then use the inverse of the y-transformed values to weight where the most a-flow goes to

#below looks like shit
#cdng.loc[:,'g*x**2 flow B'], cdng.loc[:,'g*x**2 volume B'] = flowrate_elucidation(cdng.loc[:,'g*x**2 x transform (min)'].to_numpy(), cdng.loc[:,'ic-bvol'].to_numpy())
#
#cdng.loc[:,'flat flow B'], cdng.loc[:,'flat volume B'] = flowrate_elucidation(cdng.loc[:,'flat area time (min)'].to_numpy(), cdng.loc[:,'ic-bvol'].to_numpy())



#ncdrs = cd.copy()
#ncdrs.loc[:,'retention time copy (min)'] = ncdrs.index.copy()
#ncdrs.index = pd.to_datetime(ncdrs.index, unit='m')
#ncaggdict = {i:'max' for i in ncdrs.columns}
#ncdrs = ncdrs.resample('1min').agg(ncaggdict)


#flow2d = np.diff(flows) / np.diff(nxarr[1:])
#lowinds = flow2d < 0
#lowinds = np.append(False, lowinds)
#ly = lowess(flows, nxarr[1:], frac=0.1, xvals=nxarr[1:][lowinds]).flatten()
#nflows = flows.copy()
#nflows[lowinds] = ly

#def curvefunc(x,a,b):
#    return a*x**b

#params, cov = optimize.curve_fit(curvefunc, nxarr[1:], flows, maxfev=10000)
#nflows = curvefunc(nxarr[1:], *params)
#
#plt.plot(cd.loc[:,'g*(x + 1)**2 x transform (min)'].to_numpy(), (cd.loc[:,'ic-bvol'].diff() / cd.loc[:,'g*(x + 1)**2 x transform (min)'].diff()))

#you can't really transform the flowrate under the same time domain, a 10x change in flow would be both outside the pressure range, and flow limitations

#The gradient from new time should be considered the ic gradient, after transforming this (done, via new time), this then needs to be back-calculated into a new gradient.

#should the timedomain weight be a timedomain weight or should it be a volumedomain weight? I guess it would be time either way.

#plots below show new gradient based purely on time shifting/warping
#tfs.loc[:,'count'].plot.line()
#plt.show()

#fig, ax = plt.subplots()
#tfs.plot.line(x='Time (m)', y='%ACN', color='g', ax=ax, label='old time')
#tfs.plot.line(x='new time', y='%ACN', color='purple', ax=ax, label='new time')
#plt.show()
#
#fig, ax = plt.subplots()
#tfs.plot.line(x='Time (m)', y='Flow', color='g', ax=ax, label='old flow')
#tfs.plot.line(x='new time', y='Flow', color='purple', ax=ax, label='new time flow')
#plt.show()
#
#etf.loc[:,'count'].plot.line()
#plt.show()
#
#fig, ax = plt.subplots()
#etf.plot.line(x='Time (m)', y='%ACN', color='g', ax=ax, label='old time')
#etf.plot.line(x='new time', y='%ACN', color='purple', ax=ax, label='new time')
#plt.show()
#
#fig, ax = plt.subplots()
#etf.plot.line(x='Time (m)', y='Flow', color='g', ax=ax, label='old flow')
#etf.plot.line(x='new time', y='Flow', color='purple', ax=ax, label='new time flow')
#plt.show()

#use the densitymapping.py example for the notebook, to explain wtf is going on here

#I want to try first optimizing this methodology using the static flow, so then I can work towards keeping the total volume the same?

#keep both total volume, and total volume B when making the new gradient.

#Steps:
#1. Calculate the rectangle with the same area as the counts you have
#2. Assign the 'goal' 

#So if I want to treat something as a dense object, and stretch it out to 2x the length while watching it's height automatically halve. How can I do this?
#When I say 'object' here, my intuition tells me that this thing would likely be done as a python object, but if you have a method that doesn't use that - I'm all ears.

#there's two ways of keeping volume consistent
#1. Keeping total volume consistent - This would entail transforming 
#2. Keeping the volume between 2 points consistent - This would entail changing the flowrate
