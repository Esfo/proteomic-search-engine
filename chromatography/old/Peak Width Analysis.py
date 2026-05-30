import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyteomics import mzml
from statsmodels.nonparametric.smoothers_lowess import lowess
from math import pi
import os
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
plt.rcParams["figure.dpi"] = 100

def ecdf(data):
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x,y)

#mass is +/- 30 ppm in this dataset
# fn = '/store/FE/mzMLs/test/190701_F350_yM2.peaks.csv'
# fn = 'C:/Base/data/190701_F350_yM2.peaks.csv'
folder = 'C:/Base/MS_Files/nFE/Yeast MS1s/'
contents = os.listdir(folder)
contents = list(set([i.split('.')[0] for i in contents]))
# mf = '/store/FE/mzMLs/190701_F350_yM2.mzML'
# mf = 'C:/Base/MS_Files/FE/mzMLs/190701_F350_yM2.mzML'

for f in contents:
    print(f)
    fn = ''.join((folder, f, '.peaks.csv'))
    mf = ''.join((folder, f, '.mzML'))
    
    csv = pd.read_csv(fn)
    
    msrun = mzml.MzML(mf)
    nscans = len(msrun)
    
    # outfile = ''.join((fn.split('.')[0], '.method.csv'))
    
    ni = 1
    inds = csv.loc[:,'n peaks'] == ni
    
    nc = csv.loc[inds]
    if ni > 1:
        nc.loc[:,'intensity'] = nc.loc[:,'intensity'].apply(lambda x: np.average([float(i) for i in x.split(',')]))
        nc.loc[:,'center'] = nc.loc[:,'center'].apply(lambda x: np.average([int(round(float(i))) for i in x.split(',')]))
    
    nc.loc[:,'intensity'] = nc.loc[:,'intensity'].astype(float)
    nc.loc[:,'center'] = nc.loc[:,'center'].astype(int)
    
    nc.loc[:,'w/int'] = nc.loc[:,'whm'].values / nc.loc[:,'intensity'].values
    
    # nc = nc.loc[np.logical_and(nc.loc[:,'center'] > 400, nc.loc[:,'center'] < 15000)]
    # times = [0, 10, 50, 52, 82]
    # durations = [0, 10, 40, 2, 30]
    # flow = 350 #nL/min
    # deadvol = 3693 #nL
    # percb = [1, 7, 30, 100, 100]
    
    # method = pd.DataFrame()
    # method.loc[:,'time'] = times
    # method.loc[:,'duration'] = durations
    # method.loc[:,'%B'] = percb
    
    x, y = ecdf(nc.loc[:,'whm'].values)
    plt.plot(x,y,'.', linewidth=None)
    plt.xlim(0,300)
    plt.title('ECDF of peak width')
    plt.show()
    
    # minperscan = times[-1] / nscans
    
    # nc.loc[:,'time'] = nc.loc[:,'center'] * minperscan
    # nc.loc[:,'volume'] = nc.loc[:,'whm'] * minperscan
    
    nind = nc.loc[:,'whm'] >= nc.loc[:,'whm'].mean()
    fc = nc.loc[nind]
    
    lw = lowess(nc.loc[:,'whm'].values, nc.loc[:,'center'].values, is_sorted=False, frac=0.6, it=0)
    
    
    nc.loc[:,'center binned'] = pd.cut(nc.loc[:,'center'], bins=100, labels=False).values
    fig, ax = plt.subplots(figsize=(12,3))
    for ind, frame in nc.groupby('center binned'):
        frame.loc[:,'intensity'] = (frame.loc[:,'intensity'] / frame.loc[:,'intensity'].max())
        frame.sort_values(by='intensity').plot.scatter(x='center', y='whm', c='intensity', colormap='summer', ax=ax, colorbar=False, alpha=1)
    plt.plot(lw[:,0], lw[:,1], '-', color='black')
    plt.ylim(0,500)
    plt.xlabel('Scan # (Retention Time)')
    plt.ylabel('Width at Half-Max')
    im = plt.gca().get_children()[0]
    cb = fig.colorbar(im, label='Increasing Peak Intensity per RT ---->')
    cb.set_ticks([])
    plt.show()

# nc.sort_values(by='intensity').plot.scatter(x='center', y='whm', c='intensity', colormap='summer')
# plt.plot(lw[:,0], lw[:,1], '-', color='orange')
# plt.ylim(0,500)
# plt.title('Peak width across scans')
# plt.show()

# totalvol = times[-1] * flow
# minwidth = lw[:,1].min() * minperscan #originally done as this
# # minwidth = lw[:,1].mean() * minperscan
# minvol = minwidth * flow
# whms = lw[:,1] * minperscan
# volperscan = minperscan * flow

# oldmethod = pd.DataFrame()
# oldmethod.loc[:,'scan'] = np.arange(nscans) + 1
# oldmethod.set_index('scan', inplace=True)
# oldmethod.loc[lw[:,0], 'whm (min)'] = whms
# oldmethod.interpolate(limit_direction='both', inplace=True)

# oldmethod.loc[:,'volume'] = oldmethod.loc[:,'whm (min)'].values * flow
# oldmethod.loc[:,'new flow'] = (oldmethod.loc[:,'volume'].values / minvol) * flow
# oldmethod.loc[:,'time'] = oldmethod.index.values * minperscan


# for p, t in zip(percb, times):
#     ix = np.abs(oldmethod.loc[:,'time'] - t).argmin()
#     oldmethod.loc[ix+1,'percb'] = p
# oldmethod.loc[:,'percb'].interpolate(inplace=True)

# newflow = (oldmethod.loc[:,'new flow'].values / flow).tolist()
# newpercbmild = oldmethod.loc[:,'percb'].tolist()

# oldmethod.loc[:,'bchange'] = oldmethod.loc[:,'percb'].diff()
# oldmethod.loc[1,'bchange'] = 1
# oldmethod.loc[:,'newbrate'] = (oldmethod.loc[:,'volume'].values / minvol) * oldmethod.loc[:,'bchange'].values

# newpercbrate = oldmethod.loc[:,'newbrate'].values
# newpercb = []
# val = 0
# for v in newpercbrate:
#     val += v
#     newpercb.append(val)

# ind = 1
# store = 0
# for val, pbm, pbn in zip(newflow, newpercbmild, newpercb):
#     point = int(round(val))
#     store += -1* (point - val)

#     if store >= 1:
#         sp = int(round(store))
#         sh = -1 * (sp - store)
#         store = 0
#         point += sp
#         store += sh
    
#     if point > 1:
#         for _ in range(point):
#             newflow.remove(newflow[ind])
#             newpercbmild.remove(newpercbmild[ind])
#             newpercb.remove(newpercb[ind])
#     ind += 1

# newpercb = np.asarray(newpercb)
# newpercb[newpercb > 100] = 100

# newflow = np.asarray(newflow) * flow
# nx = np.arange(len(newflow)) 

# plt.plot(nx+1, newflow, '-')

# change = 300
# smooth = newflow.copy()

# lowers = nx - change
# uppers = nx + change

# lowers[lowers < 0] = 0
# uppers[uppers > nx.max()] = nx.max()

# combs = np.linspace(lowers, uppers, change+1, axis=1).astype(np.int)
# smooth = smooth[np.r_[combs]].mean(axis=1)

# plt.plot(nx+1, smooth, '-')
# plt.show()

# plt.plot(nx, newpercbmild, '-', label='mild')
# plt.plot(nx, newpercb, '-', label='new')
# plt.plot(oldmethod.index.values, oldmethod.loc[:,'percb'].values, '-', label='old')
# plt.legend()
# plt.show()

# newmethod = pd.DataFrame()
# newmethod.loc[:,'scan'] = nx + 1
# newmethod.loc[:,'newpercb'] = newpercb
# newmethod.loc[:,'newpercbmild'] = newpercbmild
# newmethod.loc[:,'newflow'] = smooth
# newmethod.set_index('scan', inplace=True)
# newmethod.loc[:,'time'] = newmethod.index.values * minperscan
# newmethod.loc[:,'oldflow'] = flow

# nindmax = newmethod.index.max()
# newmethod.loc[:,'oldpercb'] = oldmethod.loc[:nindmax,'percb']


# fig, ax = plt.subplots(2,2,figsize=(12,10), sharex=True, sharey=True)
# #these pressure calculations don't take into account the dead time, need to incorporate that with the volume flowed for each gradient.

# newflowpressure = pressurecalc(smooth, newmethod.loc[:,'oldpercb'].values)
# ax[0,0].plot(nx, newflowpressure)
# ax[0,0].set_title('New Flow, Old %B')

# newbothpressure = pressurecalc(smooth, newpercb)
# ax[0,1].plot(nx, newbothpressure)
# ax[0,1].set_title('New Flow, New %B')

# oldpressure = pressurecalc(newmethod.loc[:,'oldflow'].values, newmethod.loc[:,'oldpercb'].values)
# ax[1,0].plot(nx, oldpressure)
# ax[1,0].set_title('Old Flow, Old %B')

# newbothpressuremild = pressurecalc(smooth, newpercbmild)
# ax[1,1].plot(nx, newbothpressuremild)
# ax[1,1].set_title('New Flow, New Mild %B')

# plt.suptitle('Pressure in bar')
# plt.show()

# nsamples = 100
# samples = np.linspace(0, nx.max(), nsamples).round().astype(int)
# fig, ax1 = plt.subplots()
# ax1.plot(nx+1, smooth, '-')

# ax2 = ax1.twinx()
# ax2.plot(nx, newpercb, '-')
# ax2.plot(nx, newpercbmild, '-')

# ax1.vlines(samples+1, ymin=0, ymax=smooth.max())
# plt.show()

# outputmethod = newmethod.iloc[samples]
# outputmethod.loc[:,'duration'] = outputmethod.loc[:,'time'].diff().values
# outputmethod.loc[1,'duration'] = 0
# outputmethod.loc[:,'newflow'] = outputmethod.loc[:,'newflow'].round().values.astype(int)
# outputmethod.loc[:,'oldpercb'] = outputmethod.loc[:,'oldpercb'].round().values.astype(int)
# outputmethod.loc[:,'newpercb'] = outputmethod.loc[:,'newpercb'].round().values.astype(int)
# outputmethod.loc[:,'newpercbmild'] = outputmethod.loc[:,'newpercbmild'].round().values.astype(int)
# outputmethod.loc[:,'duration'] = (outputmethod.loc[:,'duration'] * 60).round()
# outputmethod.loc[:,'duration'] = outputmethod.loc[:,'duration'].apply(lambda x: humanize_time(x))

# # outputmethod.to_csv(outfile)
