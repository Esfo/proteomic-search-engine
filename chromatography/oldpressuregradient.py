import numpy as np
from math import pi
from matplotlib import pyplot as plt
from scipy import integrate
import pandas as pd
plt.rcParams["figure.dpi"] = 100

npoints = 10000

def humanize_time(secs):
    mins, secs = divmod(secs, 60)
    return '%02d:%02d' % (mins, secs)

endingpressure = 80
startingpressure = 350
csvloc = 'somefile.method.csv'
separationpressurebar = np.linspace(startingpressure, endingpressure, npoints)
#separationpressurebar = 80 #bar
gradienttime = 50 #minutes, but anything goes

separationpressureubar = separationpressurebar * 1000 * 1000

minacn = 0.02
maxacn = 0.8
#needs an inflextion point, a place where the gradient can end but the flush still comntinues afterwards, so the weighting doesn't negatively affect the flush.
percentacetonitrile = np.linspace(minacn, maxacn, npoints)

#in this file, I want to produce all 5 or whatever differently weighted gradients, then see if I can predict where the elution ends based on their output volume.

columnlength = 150 #mm
columnlengthcm = columnlength / 10
columninnerdiameter = 75 #um
columnradius = columninnerdiameter / 2 #um
columnradiuscm = columnradius / 1000 / 10
particlediameter = 3 #um
particlediametercm = particlediameter / 1000 / 10

epsilon = 0.425 #interparticle porosity
#Porosity is defined as the ratio of volume of pores to the total volume of the particle.
#Interparticle referrs to the porosity between particles

temperature = 60 #celsius
abstemp = temperature + 273.15 #kelvin

centipoise = np.exp(percentacetonitrile * (-3.476 + (726 / abstemp)) + (1 - percentacetonitrile) * (-5.414 + (1566 / abstemp)) + percentacetonitrile * (1 - percentacetonitrile) * (-1.762 + (929 / abstemp))) #viscosity

poise = centipoise / 100 #g/(cm-sec)

gradientflowratemlmin = (separationpressureubar * epsilon) * ((particlediametercm**2) * (epsilon**2) * pi * (columnradiuscm**2) * 60) / (180 * poise * columnlengthcm * (1 - epsilon)**2) #mL/min
gradientflowratenlmin = gradientflowratemlmin * 1000 * 1000 #nL/min

#These are the different types of weights I'm trying
#timedomainweight = -1 * (np.arange(len(gradientflowratenlmin)) + 1)**2 #edec.method
#timedomainweight = -1 * np.log(np.arange(len(gradientflowratenlmin)) + 1)**3 #ldec.method
#timedomainweight = np.arange(len(gradientflowratemlmin)) + 1 #simic
#timedomainweight = np.flip(np.arange(len(gradientflowratemlmin)) + 1) #simdec.method
#timedomainweight = 1 / gradientflowratenlmin #simrec.method

#this can make it weird sometimes
#timedomainweight = timedomainweight / timedomainweight.sum()


save = False
#save = False
#Round 2 weightss
timedomainweight = np.flip(np.arange(len(gradientflowratemlmin)) + 1) #simdec.method

fig, ax = plt.subplots()

axn = ax.twinx()
ax.plot(percentacetonitrile, timedomainweight, color='maroon')
axn.plot(percentacetonitrile, gradientflowratenlmin, color='gold')
ax.set_ylabel('Time Domain Weight')
axn.set_ylabel('nL/min')
ax.set_xlabel('% ACN')

plt.show()

timebins = (timedomainweight / timedomainweight.sum() * gradienttime)
timedomain = np.cumsum(timebins)

fig, ax = plt.subplots()

axn = ax.twinx()
ax.plot(timedomain, percentacetonitrile, color='black')
axn.plot(timedomain, gradientflowratenlmin, color='gold')
ax.set_ylabel('% ACN')
axn.set_ylabel('nL/min')
ax.set_xlabel('Time')

plt.show()

print('trapz:', np.trapezoid(gradientflowratenlmin, timedomain))

volume = integrate.cumtrapz(gradientflowratenlmin, timedomain)
volume = np.insert(volume, 0, 0)
#at first it seems like this plot is wrong based on the time vs flowrate plot above it, but the y-axis offset is so high that the large majority of the volume comes from the fast that the flowrate is > 300 nL/min, or whatever volume the baseline is. If the bottom of the curve was closer to zero, the plot of auc vs time would look more intuitive.


columnvolume = columnlengthcm * pi * columnradiuscm **2 #mL
columnvolumenl = columnvolume * 1000 * 1000 #nL

deadvolume = 3030 #nL

volumeportions = np.diff(volume)
volumeportions = np.insert(volumeportions, 0, volumeportions[0])

fig, ax = plt.subplots()

axn = ax.twinx()

ax.plot(timedomain, volume, color='green')
axn.plot(timedomain, volumeportions, color='orange')

ax.set_ylabel('Cumulative Volume')
axn.set_ylabel('Volume per Time')
ax.set_xlabel('Time')
plt.show()

fig, ax = plt.subplots()

axn = ax.twinx()
ax.plot(timedomain, volumeportions, color='orange')
axn.plot(timedomain, volumeportions * percentacetonitrile, color='purple')

axn.set_ylabel('Volume Acetonitrile per Time')
ax.set_ylabel('Volume per Time')
ax.set_xlabel('Time')
plt.show()

fig, ax = plt.subplots()

axn = ax.twinx()
ax.plot(timedomain, volume, color='green')
axn.plot(timedomain, volume * percentacetonitrile, color='purple')

axn.set_ylabel('% Volume Acetonitrile per Cumulative Timepoint')
ax.set_ylabel('Cumulative Volume')
ax.set_xlabel('Time')
plt.show()


initialdeadtime = deadvolume / gradientflowratenlmin[0] #min
timeextensionnumber = np.round((initialdeadtime / gradienttime) * npoints).astype(int)
timeextension = np.linspace(0,initialdeadtime, timeextensionnumber)

time = timedomain + initialdeadtime
time = np.insert(time, 0, timeextension)

flowextension = np.repeat(gradientflowratenlmin[0], timeextensionnumber)
flow = np.insert(gradientflowratenlmin, 0, flowextension)

gradientvolume = integrate.cumtrapz(flow, time)

volumeshift = gradientvolume - deadvolume
volumeshift = volumeshift[volumeshift > 0][-npoints:]


acnshiftindices = np.abs(gradientvolume.reshape(-1,1) - volumeshift).argmin(axis=0)

if acnshiftindices.size != np.unique(acnshiftindices).size:
    uinds = np.unique(acnshiftindices, return_index=True)[1]
    acnshiftindices = acnshiftindices[uinds]
    percentacetonitrile = percentacetonitrile[uinds]


ef = pd.DataFrame()
ef.loc[:,'Flow'] = flow
ef.loc[:,'Time'] = time
ef.loc[acnshiftindices, '%ACN'] = percentacetonitrile * 100
ef = ef.interpolate()

fig, ax = plt.subplots()

axn = ax.twinx()
ef.plot.line(x='Time', y='Flow', color='gold', ax=ax)
ef.plot.line(x='Time', y='%ACN', color='black', ax=axn)

ax.set_ylabel('nL/min')
axn.set_ylabel('% ACN')

ax.get_legend().remove()
axn.get_legend().remove()
plt.show()

ts = ef.loc[:,'Time'].max().round().astype(int)
samples = np.linspace(0, ts, ts+1) #giving 1 minute intervals like this
sinds = np.abs(samples.reshape(-1,1) - ef.loc[:,'Time'].values).argmin(axis=1)

nm = ef.iloc[sinds]
nm = nm.round()
nm.loc[:,'Duration'] = nm.loc[:,'Time'].diff() * 60
nm.loc[0,'Duration'] = nm.iloc[1].loc['Duration']
nm.loc[0,'%ACN'] = nm.iloc[1].loc['%ACN']
nm.loc[:,'Duration'] = nm.loc[:,'Duration'].apply(lambda x: humanize_time(x))

if save:
    nm.to_csv(csvloc)

fig, ax = plt.subplots()

axn = ax.twinx()
nm.plot.line(x='Time', y='Flow', color='gold', ax=ax)
nm.plot.line(x='Time', y='%ACN', color='black', ax=axn)

ax.set_ylabel('nL/min')
axn.set_ylabel('% ACN')

ax.get_legend().remove()
axn.get_legend().remove()
plt.show()
