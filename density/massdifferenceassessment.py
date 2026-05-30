import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from colour import Color
from pyteomics import mzml
from time import time
import pandas as pd
import gc
import concurrent.futures
from collections import Counter, defaultdict
from scipy import sparse, integrate, spatial, stats, special
from sklearn.neighbors import NearestNeighbors, KernelDensity
from distinctipy import distinctipy as dp
import random
import itertools
import pickle
import sys
import os
gc.enable()

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

mzmlfile = '/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
basefolder = '/'.join((mzmlfile.split('/')[:-2]))
basefile = mzmlfile.split('/')[-1].split('.mzML')[0]

processedroot = '/'.join((basefolder, 'fileprocessing', basefile))
librarylocation = '/home/sfo/data/proteomics/fastas/isotope-arrays/human_isotopes-6-50_miss-1_ss50'

differencefile = '/'.join((processedroot, 'distribution.differences.pickle'))
with open(differencefile, 'rb') as pick:
    differences = pickle.load(pick)

def density_by_difference(array):
    array = np.sort(array) #not needed thanks to searchsorted, but searchsorted might be a little faster if this is done i suppose
    mindiff = np.diff(array).min()
    emin = array.min() - mindiff
    emax = array.max() + mindiff
    ediff = emax - emin
    previouslocations = {} #location: 1, or -1 for pos/neg diffs respectively
    location = 2
    minpositive = array.size #smalest location of a positive difference
    #location = minpositive // 2
    #maxnegative = 0 #largest location of a negative difference
    maxnegative = location #largest location of a negative difference
    doubling = True
    while True:
        densitycount = np.zeros(location)
        n = 0
        div = ediff / location
        l = emin
        r = l + div
        while r < emax:
            lfind = np.searchsorted(array, l)
            rfind = np.searchsorted(array, r)
            densitycount[n] += rfind - lfind
            l = r
            r += div
            n += 1
        smean = densitycount.sum() / densitycount.size #why is this faster 
        spsum = (densitycount == 0).sum()**2
        diff = spsum - smean #optimizing for the smallest positive difference
        #I don't formally account for an == 0 option, it's pretty unlikely imo, all the sums are integers but the means should having floating accuracy
        if diff > 0:
            previouslocations[location] = 1
            doubling = False
            if location < minpositive:
                minpositive = location
        else: #< 0, and since this technically covers the <= case, generated median locations will be rounded up i suppose? actually i didn't do this but i need to think about it
            previouslocations[location] = -1
            if location > maxnegative:
                maxnegative = location
        adjacentloc = location - previouslocations[location]
        if adjacentloc in previouslocations:
            if previouslocations[adjacentloc] != previouslocations[location]:
                #settling either for the optimized answer, or the one right before it - good enough I suppose
                break #you win
        if doubling:
            newloc = location * 2
        else:
            newloc = int(round(((minpositive + maxnegative) / 2) + 0.1)) #the 0.1 makes a .5 round to 1 which would normally go to 0, why... python?
        while newloc in previouslocations:
            newloc -= previouslocations[newloc]
        location = newloc
    
    fvn = densitycount / densitycount.sum()
    fx = np.linspace(emin, emax, fvn.size)
    return fx, fvn

nt = time()
dx, dy = density_by_difference(differences)
print(time() - nt)

##these don't plot together for whatever reason
#plt.hist(differences, bins=1000)
#plt.show()
#
#plt.plot(dx, dy, color='cyan')
#plt.show()

saverloc = '/'.join((processedroot, 'differences.density.pickle'))
with open(saverloc, 'wb') as pick:
    pickle.dump([dx, dy], pick)

