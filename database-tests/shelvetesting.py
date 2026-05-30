import shelve
import numpy as np
import json

#so shelve actually interfaces directly with dbm and chooses which database to use based on OS

dbloc = '/home/sfo/test/testshelve.db'

narrays = 2000
minlength = 50
maxlength = 1000
out = {}
for o in range(narrays):
    rl = np.random.randint(minlength+1, maxlength)
    out[o] = np.random.uniform(size=(2, rl)).tobytes()

with shelve.open(dbloc) as db:
    db.update(out)

db = shelve.open(dbloc)

osave = str(out).encode()

with open(dbloc, 'x') as f:
    f.write(out)
    print(mydictionary, file=f)

import pickle

with open(dbloc, 'wb') as f:
    pickle.dump(out, f)
