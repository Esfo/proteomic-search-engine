#!/bin/xonsh
from functools import partial
import sys
import os

proteome = 'Human_Homo_sapien'
basefolder = '/store/flowcharacterizations/round3'
librarylocation = '/home/sfo/data/proteomics/fastas/search-db'
processingdirectory = '/store/flowcharacterizations/round3/fileprocessing'

files = [i for i in $(ls @(basefolder)).split() if i.endswith('.raw')]
basenames = [i.split('.raw')[0] for i in files]

nprocs = os.cpu_count()

#setting up /mzMLs
mzfolder = '/'.join((basefolder, 'mzMLs'))
if not os.path.isdir(mzfolder):
    mkdir @(mzfolder)

#converting to mzml
#for f in files:
#    b = f.split('.raw')[0]
#    mzname = ''.join(('/'.join((mzfolder, b)), '.mzML'))
#    if not os.path.isfile(mzname):
#       docker run -it --rm -e WINEDEBUG=-all -v @(basefolder):/data chambm/pwiz-skyline-i-agree-to-the-vendor-licenses wine msconvert --mzML --outdir=mzMLs/ @(f)
def format_conversion(folder, f, centroid=False):
    dockercmd = f'docker run --rm -e WINEDEBUG=-all -v {folder}:/data chambm/pwiz-skyline-i-agree-to-the-vendor-licenses wine msconvert --mzML --outdir=mzMLs/ {f}'
    subprocess.run(dockercmd, shell=True, check=True)

partial_format_conversion = partial(format_conversion, basefolder)

with mp.Pool(nprocs) as pool:
    pool.map(partial_format_conversion, files)

#setting up /fileprocessing
processingfolder = '/'.join((basefolder, 'fileprocessing'))
if not os.path.isdir(processingfolder):
    mkdir @(processingfolder)

#running things
for b in basenames:
    mzname = ''.join(('/'.join((mzfolder, b)), '.mzML'))
    ipython -- main.py search --file @(mzname) --librarylocation @(librarylocation) --processinglocation @(processingdirectory) --proteome @(proteome)
    #^ that -- makes it work somehow, otherwise the arguments are intepreted for ipython not the script
    #https://stackoverflow.com/questions/4138145/command-line-options-to-ipython-scripts
