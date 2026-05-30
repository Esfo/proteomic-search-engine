import pandas as pd
import os
import matplotlib.pyplot as plt
#pd.options.display.max_rows = 999
#pd.options.display.max_columns = 999

rf1 = '/store/flowcharacterizations/round1/DDAs/MGFs/results.csv'
rf2 = '/store/flowcharacterizations/round2/DDAs/MGFs/results.csv'
methods = '/store/flowcharacterizations/round2/gradients/'

r1 = pd.read_csv(rf1)
r1.loc[:,'file'] = r1.loc[:,'file'].apply(lambda x: x.split('.')[0].split('_')[-1])
r1.set_index('file', inplace=True)

r2 = pd.read_csv(rf2)
r2.loc[:,'file'] = r2.loc[:,'file'].apply(lambda x: '-'.join((x.split('.')[0].split('_')[-1].split('-')[:-1])))
r2.set_index('file', inplace=True)

gradients = []

r2peaks = '/store/flowcharacterizations/round2/MS1s/peaks/'
for f in os.listdir(r2peaks):
    df = pd.read_csv(''.join((r2peaks, f)), low_memory=False)
    inloc = '-'.join((f.split('_')[1].split('.')[0].split('-')[:-1]))
    r2.loc[inloc, '# scans with peaks'] = len(df.loc[:,'peak location'].unique())
    r2.loc[inloc, '# peaks'] = len(df)
    #r2 = (r2.loc[:,'# scans with peaks'] / r2.loc[:,'peptides']).sort_values()
    r2.loc[:,'# scan with peaks/peptide'] = (r2.loc[:,'# scans with peaks'] / r2.loc[:,'peptides']).sort_values()
    
    pks = pd.read_csv(''.join((methods, inloc, '.method.csv')))
    pks.loc[:,'file'] = inloc
    gradients.append(pks)

gradients = pd.concat(gradients)
gradients.drop('Unnamed: 0', axis=1, inplace=True)

gradients = gradients.pivot_table(index='Time', values=('Flow', 'Time', '%ACN', 'Duration'), columns='file')
gradients = gradients.interpolate()
gcols = gcols = gradients.columns.levels[1].unique().tolist()
gcols = [i for i in gcols if i]

for val in gradients.columns.levels[1]:
    gradients.loc[:,('Peptides', val)] = r2.loc[val, 'peptides']
    gradients.loc[:,('Volume', val)] = gradients.loc[:,('Flow', val)].mul(gradients.index.to_numpy(), axis=0)

holder = gradients.loc[:,'%ACN'] / gradients.loc[:,'Volume']
#holder.iloc[0][holder.iloc[0] == np.inf] = 0

for val in gradients.columns.levels[0]:
    fig, ax = plt.subplots(nrows=1, figsize=(20,10))
    gradients.loc[:,val].plot.line(ax=ax)
    ax.legend(bbox_to_anchor=(1,1))
    
    #gradients.loc[:,val].plot.line(ax=ax[1], color='Peptides')
    #ax[1].legend(bbox_to_anchor=(1,1))
    #plt.colorbar()
    
    plt.show()

holder.plot.line(figsize=(20,10))
plt.show()
