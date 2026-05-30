from tinydb import TinyDB, Query
import numpy as np

dbloc = '/home/sfo/test/testbase.json'

narrays = 2000
minlength = 50
maxlength = 1000
out = {}
for o in range(narrays):
    rl = np.random.randint(minlength+1, maxlength)
    out[o+99999] = np.random.uniform(size=(2, rl)).tolist()

with TinyDB(dbloc) as db:
    db.update(out)
    db.insert(out)

db = TinyDB(dbloc)

#this seems to insert the entire dictionary as an individual entry, from the example page it seems they inert multiple small dictionaries to make a datatable
