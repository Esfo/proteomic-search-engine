import pandas as pd
from pyteomics import mgf
from time import time
import sys
import multiprocessing as mp
import numpy as np

def scanfunc(scan, ind, times):
    times[ind] = [scan['params']['rtinseconds'], msrun[0]['intensity array'].mean(), np.median(msrun[0]['intensity array']), msrun[0]['intensity array'].max(), msrun[0]['intensity array'].sum()]
    return times

def readfunc(f):
    msrun = mgf.IndexedMGF(f)
    times = mp.Manager().dict()
    pool = mp.Pool()
    for ind, scan in enumerate(msrun):
        pool.apply_async(scanfunc(scan, ind, times))
    pool.close()
    pool.join()
    return times

def main(mgffile):
    mt = time()
    times = readfunc(mgffile)
    df = pd.DataFrame.from_dict(times, orient='index')
    df.index.name = 'scan'
    df.columns = ['seconds', 'mean intensity', 'median intensity', 'max intensity', 'summed intensity']
    
    outfolder = ''.join(('/'.join((mgffile.split('/')[:-2])), '/scans/'))
    outfile = ''.join((mgffile.split('/')[-1].split('.')[0], '.scans.csv'))
    fname = ''.join((outfolder, outfile))
    df.to_csv(fname)
    print(time() - mt, '-', mgffile.split('/')[-1].split('.')[0])

if __name__ == '__main__':
    main(sys.argv[1])

