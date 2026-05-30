import pandas as pd
import os
from matplotlib import pyplot as plt
from scipy import stats
import matplotlib
from itertools import chain
import matplotlib.patches as patches

default_colors = [
    # r, g, b, a
    [92, 192, 98, 0.5],
    [90, 155, 212, 0.5],
    [246, 236, 86, 0.6],
    [241, 90, 96, 0.4],
    [255, 117, 0, 0.3],
    [82, 82, 190, 0.2],
]
default_colors = [
    [i[0] / 255.0, i[1] / 255.0, i[2] / 255.0, i[3]]
    for i in default_colors
]

def draw_ellipse(fig, ax, x, y, w, h, a, fillcolor):
    e = patches.Ellipse(
        xy=(x, y),
        width=w,
        height=h,
        angle=a,
        color=fillcolor)
    ax.add_patch(e)

def draw_text(fig, ax, x, y, text, color=[0, 0, 0, 1]):
    ax.text(
        x, y, text,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=14,
        color=color)
    
def get_labels(data, fill=["number"]):
    """    
    get a dict of labels for groups in data
    
    @type data: list[Iterable]    
    @rtype: dict[str, str]
    input
      data: data to get label for
      fill: ["number"|"logic"|"percent"]
    return
      labels: a dict of labels for different sets
    example:
    In [12]: get_labels([range(10), range(5,15), range(3,8)], fill=["number"])
    Out[12]:
    {'001': '0',
     '010': '5',
     '011': '0',
     '100': '3',
     '101': '2',
     '110': '2',
     '111': '3'}
    """

    N = len(data)

    sets_data = [set(data[i]) for i in range(N)]  # sets for separate groups
    s_all = set(chain(*data))                             # union of all sets

    # bin(3) --> '0b11', so bin(3).split('0b')[-1] will remove "0b"
    set_collections = {}
    for n in range(1, 2**N):
        key = bin(n).split('0b')[-1].zfill(N)
#        value = s_all
        sets_for_intersection = [sets_data[i] for i in range(N) if  key[i] == '1']
#        sets_for_difference = [sets_data[i] for i in range(N) if  key[i] == '0']
        nv = []
        for s in sets_for_intersection:
            nv.extend(s)
        nv = list(set(nv))
#        if key.count('1') > 2:
#        for s in sets_for_difference:
#            value = value - s
        set_collections[key] =  nv

    labels = {k: "" for k in set_collections}
    if "logic" in fill:
        for k in set_collections:
            labels[k] = k + ": "
    if "number" in fill:
        for k in set_collections:
            labels[k] += str(len(set_collections[k]))
    if "percent" in fill:
        data_size = len(s_all)
        for k in set_collections:
            labels[k] += "(%.1f%%)" % (100.0 * len(set_collections[k]) / data_size)

    return labels

def venn5(labels, names=['A', 'B', 'C', 'D', 'E'], **options):
    """
    plots a 5-set Venn diagram
        
    @type labels: dict[str, str]
    @type names: list[str]
    @rtype: (Figure, AxesSubplot)
    
    input
      labels: a label dict where keys are identified via binary codes ('00001', '00010', '00100', ...),
              hence a valid set could look like: {'00001': 'text 1', '00010': 'text 2', '00100': 'text 3', ...}.
              unmentioned codes are considered as ''.
      names:  group names
      more:   colors, figsize, dpi
    return
      pyplot Figure and AxesSubplot object
    """
    colors1 = options.get('colors', [default_colors[i] for i in range(5)])
    figsize = options.get('figsize', (13, 13))
    dpi = options.get('dpi', 96)
    
    fig = plt.figure(0, figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_axis_off()
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.set_xlim(left=0.0, right=1.0)
    
    # body   
    draw_ellipse(fig, ax, 0.428, 0.449, 0.87, 0.50, 155.0, colors1[0])
    draw_ellipse(fig, ax, 0.469, 0.543, 0.87, 0.50, 82.0, colors1[1])
    draw_ellipse(fig, ax, 0.558, 0.523, 0.87, 0.50, 10.0, colors1[2])
    draw_ellipse(fig, ax, 0.578, 0.432, 0.87, 0.50, 118.0, colors1[3])
    draw_ellipse(fig, ax, 0.489, 0.383, 0.87, 0.50, 46.0, colors1[4])
    draw_text(fig, ax, 0.27, 0.11, labels.get('00001', ''))
    draw_text(fig, ax, 0.72, 0.11, labels.get('00010', ''))
    draw_text(fig, ax, 0.55, 0.13, labels.get('00011', ''))
    draw_text(fig, ax, 0.91, 0.58, labels.get('00100', ''))
    draw_text(fig, ax, 0.78, 0.64, labels.get('00101', ''))
    draw_text(fig, ax, 0.84, 0.41, labels.get('00110', ''))
    draw_text(fig, ax, 0.76, 0.55, labels.get('00111', ''))
    draw_text(fig, ax, 0.51, 0.90, labels.get('01000', ''))
    draw_text(fig, ax, 0.39, 0.15, labels.get('01001', ''))
    draw_text(fig, ax, 0.42, 0.78, labels.get('01010', ''))
    draw_text(fig, ax, 0.50, 0.15, labels.get('01011', ''))
    draw_text(fig, ax, 0.67, 0.76, labels.get('01100', ''))
    draw_text(fig, ax, 0.70, 0.71, labels.get('01101', ''))
    draw_text(fig, ax, 0.51, 0.74, labels.get('01110', ''))
    draw_text(fig, ax, 0.64, 0.67, labels.get('01111', ''))
    draw_text(fig, ax, 0.10, 0.61, labels.get('10000', ''))
    draw_text(fig, ax, 0.20, 0.31, labels.get('10001', ''))
    draw_text(fig, ax, 0.76, 0.25, labels.get('10010', ''))
    draw_text(fig, ax, 0.65, 0.23, labels.get('10011', ''))
    draw_text(fig, ax, 0.18, 0.50, labels.get('10100', ''))
    draw_text(fig, ax, 0.21, 0.37, labels.get('10101', ''))
    draw_text(fig, ax, 0.81, 0.37, labels.get('10110', ''))
    draw_text(fig, ax, 0.74, 0.40, labels.get('10111', ''))
    draw_text(fig, ax, 0.27, 0.70, labels.get('11000', ''))
    draw_text(fig, ax, 0.34, 0.25, labels.get('11001', ''))
    draw_text(fig, ax, 0.33, 0.72, labels.get('11010', ''))
    draw_text(fig, ax, 0.51, 0.22, labels.get('11011', ''))
    draw_text(fig, ax, 0.25, 0.58, labels.get('11100', ''))
    draw_text(fig, ax, 0.28, 0.39, labels.get('11101', ''))
    draw_text(fig, ax, 0.36, 0.66, labels.get('11110', ''))
    draw_text(fig, ax, 0.51, 0.47, labels.get('11111', ''))
    
    # legend
    draw_text(fig, ax, 0.02, 0.72, names[0], colors1[0])
    draw_text(fig, ax, 0.72, 0.94, names[1], colors1[1])
    draw_text(fig, ax, 0.97, 0.74, names[2], colors1[2])
    draw_text(fig, ax, 0.88, 0.05, names[3], colors1[3])
    draw_text(fig, ax, 0.12, 0.05, names[4], colors1[4])
    leg = ax.legend(names, loc='best', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    
    return fig, ax


directory = '/store/flowcharacterizations/round3/DDAs/crux-output/'
files = [i for i in os.listdir(directory) if i.endswith('.percolator.target.peptides.txt')]

nf = []
for f in files:
    frame = pd.DataFrame()
    pn = ''.join((directory, f))
    pf = pd.read_csv(pn, sep='\t')

    pf = pf.loc[pf.loc[:,'percolator q-value'] < 0.01]
    pf.loc[:,'file_idx'] = f.split('.')[0]

    nf.append(pf)

nf = pd.concat(nf)
nf.set_index('file_idx', inplace=True)
nf.sort_index(inplace=True)

nf.loc[:,'length'] = nf.loc[:,'sequence'].apply(lambda x: len(x))
#nf = nf.loc[nf.loc[:,'length'] > 6]

for i in nf.index.unique():
    nf.loc[i, 'length'].plot.hist(bins=20)
    plt.title(i)
    plt.show()

for f in nf.index.unique():
    fig, ax = plt.subplots(figsize=(10,6))
    nf.loc[f,'frequency'] = stats.gaussian_kde(nf.loc[f,('length', 'scan')].to_numpy().transpose())(nf.loc[f,('length', 'scan')].to_numpy().transpose())
    nf.loc[f].plot.scatter(x='scan', y='length', c='frequency', colormap='winter', ax=ax, alpha=0.5, norm=matplotlib.colors.LogNorm())
    plt.grid()
    plt.title(f)
    plt.show()

nf.reset_index(inplace=True)
nf.set_index(['file_idx', 'scan'], inplace=True)
nf.sort_index(inplace=True)

binsize = 100 #scans

vf = nf.loc[:,'length'].to_frame().var(axis=1, level=0)

inds = nf.index.levels[0]
inds = inds[:-2].tolist()

nf = nf.loc[inds]
peplist = [nf.loc[f, 'sequence'].tolist() for f in inds]
labels = get_labels(peplist, fill=['number'])
fig, ax = venn5(labels, names=inds)
fig.show()

