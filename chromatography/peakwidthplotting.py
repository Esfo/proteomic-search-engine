import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import os
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
plt.rcParams["figure.dpi"] = 100

warnings.filterwarnings("ignore")

rf = '/store/flowcharacterizations/round2/DDAs/MGFs/results.csv'
df = pd.read_csv(rf)
df.loc[:,'file'] = df.loc[:,'file'].apply(lambda x: x.split('.')[0].split('_')[-1])
df.set_index('file', inplace=True)


folder = '/store/flowcharacterizations/round2/MS1s/peaks/'
#folder = '/store/FE/peaks/'
files = os.listdir(folder)
files = [i for i in files if i.endswith('.peaks.csv')]
files = [i for i in files if 'stat' in i]
#files = [i for i in files if 'FdV' not in i]

#filesort = [int(i.split('_')[1].split('F')[1]) for i in files]
#filesort = np.asarray(filesort)
#
#files = np.asarray(files)
#files = files[filesort.argsort()].tolist()

plottervals = ['trimmed base widths']
xplotval = 'Time'

#fm, axm = plt.subplots(figsize=(12,3))

for fr in files:
    f = ''.join((folder, fr))
    enfs = pd.read_csv(f, low_memory=False)
    
    for pv in plottervals:
        enfs.loc[:,'frequency'] = stats.gaussian_kde(enfs.loc[:,(xplotval, pv)].to_numpy().transpose())(enfs.loc[:,(xplotval, pv)].to_numpy().transpose())
        
        enfs.reset_index().sort_values(by='frequency').plot.scatter(x=xplotval, y=pv, c='frequency', colormap='winter', ax=ax[axval], alpha=0.5, norm=matplotlib.colors.LogNorm())
        
        enfs.reset_index().sort_values(by='raw heights').plot.scatter(x=xplotval, y=pv, c='raw heights', colormap='summer', ax=ax[axval+1], alpha=0.5, norm=matplotlib.colors.LogNorm())
        
        axval += 2
    
    plt.suptitle('-'.join((fr.split('.')[0].split('-')[:-1])).split('_')[1])
    plt.show()


#axm.legend(bbox_to_anchor=(1,1))
#fm.show()
