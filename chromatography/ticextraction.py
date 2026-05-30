import pandas as pd
import numpy as np
from pyteomics import mzml
from time import time
import sys
import gc

def scanfunc(scan):
    et = pd.DataFrame(scan['intensity array'], columns=['intensity'])
    et.loc[:,'m/z'] = scan['m/z array'].round(4)
    try:
        et.loc[:,'index'] = scan['index']
    except ValueError:
        return []
    et.loc[:,'ms level'] = scan['ms level']
    et.loc[:,'time (min)'] = scan['scanList']['scan'][0]['scan start time'].real
    return [et]

#this should all be incorporated into PWF eventually, where it just outputs two files, the peaks and the TICs. Alongside these would be the gradient info with a method file as optional input. Alongside all of this would be the ideal exclusion list for that instrument
def main(mzmlfile):
    mt = time()
    ef = []
    msrun = mzml.MzML(mzmlfile)
    for t in msrun.map(lambda scan: scanfunc(scan)):
        ef.extend(t)
    
    gc.collect()
    print(time() - mt, '- File Extracted')
    ef = pd.concat(ef)
    #ef = ef.loc[ef.loc[:,'ms level'] == 1]
    ef.rename({'index':'scan'}, axis=1, inplace=True)

    of = ef.groupby('scan').agg({'intensity': [np.max, np.mean, np.median, np.sum], 'm/z': [np.max, np.mean, np.median, np.sum], 'time (min)': 'mean', 'ms level': 'mean'})

    newcols = [' '.join((reversed(i))) for i in of.columns.tolist()]
    newcols = [i.replace('amax', 'max') for i in newcols]
    newcols = [i.replace('mean time (min)', 'time (min)') for i in newcols]
    newcols = [i.replace('sum', 'summed') for i in newcols]

    of.columns = newcols

    outfolder = '/'.join(('/'.join((mzmlfile.split('/')[:-2])), 'TICs/'))
    outfile = ''.join((mzmlfile.split('/')[-1].split('.')[0], '.tic.csv'))
    fname = ''.join((outfolder, outfile))
    of.to_csv(fname)
    print(time() - mt, '-', mzmlfile.split('/')[-1].split('.')[0])

if __name__ == '__main__':
    main(sys.argv[1])

