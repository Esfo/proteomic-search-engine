import numpy as np
import pandas as pd
import pickle


isotopefile = '/home/sfo/data/proteomics/fastas/isotope-arrays/human-isotopes-6-50.pickle'


with open(isotopefile, "rb") as pick:
    isotopeabundances, seqsbymass = pickle.load(pick)

seqmasses = np.array(list(seqsbymass.keys()))

#first make a dataframe showing how many other dominant isotopomers would overlap with each other at some ppm distance?
#^also, how much does a ppm go up as you increase mass?
#also make a separate number for charge states up to 8 or so, check for # overlaps when considering up to that many charge states? No, it should be how many of overlaps does SEQUENCE have with the n=4 charge states, that would be an individual column

#portray differences in both ppm and daltons, side by side I suppose
