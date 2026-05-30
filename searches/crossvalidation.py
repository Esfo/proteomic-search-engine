import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from colour import Color
import psutil
import asyncio
import aiofiles
from pyteomics import mzml
import csv
import bisect
import heapq
import fcntl #this will need to be portalocker on other operating systems
from time import time
import pandas as pd
import gc
import concurrent.futures
import multiprocessing as mp
from collections import Counter, defaultdict
from scipy import sparse, integrate, spatial, stats, special
from pandas.api.types import CategoricalDtype
from sklearn.neighbors import NearestNeighbors
from more_itertools import sort_together
from functools import partial
from pickleshare import PickleShareDB
from decimal import Decimal, getcontext
import tempfile
import math
import zlib
import lmdb
import random
import itertools
import string
import pickle
import sys
import os
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#warnings.filterwarnings("error")
np.set_printoptions(suppress=True)
gc.enable()

#for more https://matplotlib.org/stable/tutorials/introductory/customizing.html
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
        '#8ff6ff',
        '#ff9f9c',
        '#2ded8d',
        '#fbffb3',
        '#ea68f2',
        '#7d26ff',
        ]
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', chexes)
#n = 0
#for c in chexes:
#    plt.plot([n, n+1], [n, n+1], color=c, label=c)
#    n += 1
#plt.legend()
#plt.show()

mzmlfile = '/home/sfo/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
#mzmlfile = '/store/flowcharacterizations/round3/mzMLs/200901_fR_400.mzML'
basefolder = '/'.join((mzmlfile.split('/')[:-2]))
basefile = mzmlfile.split('/')[-1].split('.mzML')[0]

processinglocation = '/'.join((basefolder, 'fileprocessing', basefile))
librarylocation = '/home/sfo/data/proteomics/fastas/search-db'
csvfilename = '/'.join((processinglocation, 'fragment.matches'))
proteome = 'Human_Homo_sapien'
nprocs = 8
proton = 1.007276554940804
dividingthreshold = 0.8
ppmtol = 25
ppmmod = ppmtol / 1000000

peptiderankingsfile = '/'.join((processinglocation, 'peptiderankings.csv'))
distributionrankingsfile = '/'.join((processinglocation, 'distributionrankings.csv'))
scanrankingsfile = '/'.join((processinglocation, 'scanrankings.csv'))

fulldecoysetfile = '/'.join((processinglocation, 'fulldecoyset.pickle'))
with open(fulldecoysetfile, 'rb') as pick:
    fulldecoyset = pickle.load(pick)
#fulldecoyset = set() #all decoy sequences

#peptideheaders = ['sequence', 'analyteid', 'score', 'ion_coverage', 'scan_indices']
df = pd.read_csv(peptiderankingsfile)

#DO do the combinatorics -> benefit scores of chimeric spectra via what matches together and also subtraction-benefit for whatever in the spectra isn't matched if its bad
#do the same for single-hit line-scan combos and subtract the bad things

#parameterize as much as possible then:
#DO ML, gridsearch + cross-validation for SVM, decision tree, random forest, etc, check a bunch i guess
#at all 3 levels, line/dist/analyte
#infer false positive rate from line/dists i guess?


ddf = pd.read_csv(distributionrankingsfile)
ldf = pd.read_csv(scanrankingsfile)

#x = df.drop('target', axis=1)  # Replace 'target' with your target column name
x = ldf.loc[:,('scan_ion_score', 'ppm_error', 'matched_intensity_sum', 'scan_geometry')].to_numpy()
#y = df['target']
y = ldf.loc[:,'sequence'].apply(lambda x: x in fulldecoyset).to_numpy()

# Convert boolean array to int (True -> 1, False -> 0)
y = y.astype(int)

# Split the data into known (labeled) and unknown (unlabeled) sets
X_known, X_unknown, y_known, y_unknown = train_test_split(x, y, test_size=0.3, random_state=42)

# Further split the known data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_known, y_known, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_unknown_scaled = scaler.transform(X_unknown)

# Create a Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest on the known data
rf.fit(X_train_scaled, y_train)

# Predict on validation set
y_val_pred = rf.predict(X_val_scaled)

# Calculate metrics
accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred, average='binary')
recall = recall_score(y_val, y_val_pred, average='binary')
f1 = f1_score(y_val, y_val_pred, average='binary')

print("Random Forest Results:")
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation Precision: {precision:.4f}")
print(f"Validation Recall: {recall:.4f}")
print(f"Validation F1-score: {f1:.4f}")

# Predict probabilities for unknown samples
unknown_probs = rf.predict_proba(X_unknown_scaled)

# You can set a threshold to classify the unknown samples
threshold = 0.8  # Adjust this based on your needs
unknown_predictions = (unknown_probs[:, 1] >= threshold).astype(int)

# Perform cross-validation on the known data
cv_scores = cross_val_score(rf, X_known, y_known, cv=5, scoring='accuracy')

print("\nCross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())
print("Standard deviation of CV scores:", cv_scores.std())

# Feature importance
feature_names = ['scan_ion_score', 'ppm_error', 'matched_intensity_sum', 'scan_geometry']
feature_importance = sorted(zip(feature_names, rf.feature_importances_), key=lambda x: x[1], reverse=True)
print("\nFeature Importance:")
for feature, importance in feature_importance:
    print(f"{feature}: {importance:.4f}")
