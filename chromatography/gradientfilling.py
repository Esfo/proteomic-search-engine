import pandas as pd
import os

folder = '/store/flowcharacterizations/round4/plus/gradients/'

files = [i for i in os.listdir(folder) if i.endswith('.method.csv')]

af = pd.DataFrame()
af.loc[:,'Time'] = [160, 161, 170, 171, 180]
af.loc[:,'Time (m)'] = af.loc[:,'Time']
af.loc[:,'%ACN'] = [85, 85, 95, 2, 2]
af.loc[:,'Time'] = pd.to_datetime(af.loc[:,'Time'], unit='m')

for f in files:
    dname = ''.join((folder, f))
    nf = af.copy()
    df = pd.read_csv(dname)
    nf.loc[:,'Flow'] = df.loc[df.index.max(),'Flow']
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df.loc[:,'Time'] = pd.to_datetime(df.loc[:,'Time'])
    df = pd.concat([df, nf])

    df.loc[:,'timediff'] = df.loc[:,'Time'].diff()
    df.loc[:,'Duration'] = df.loc[:,'timediff'].apply(lambda x: ':'.join((str(x).split(' ')[-1].split(':')[1:])))
    df.loc[df.index[0], 'Duration'] = ':'.join((str(df.iloc[0].loc['Time']).split(' ')[-1].split(':')[1:]))

    df.loc[:,'Time (m)'] = df.loc[:,'Time'].dt.hour * 60 + df.loc[:,'Time'].dt.minute
    
    df.to_csv(dname)
