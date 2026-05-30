from pickleshare import PickleShareDB
from functools import partial
import multiprocessing as mp
import numpy as np

dbloc = '/home/sfo/test/picklesharetest/'

narrays = 200000
minlength = 50
maxlength = 1000
out = {}
for o in range(narrays):
    rl = np.random.randint(minlength, maxlength)
    out[str(o)] = np.random.uniform(size=(2, rl)).tolist()

def dbfunc(db, k, v):
    db[k] = v

db = PickleShareDB(dbloc)
db_partial = partial(dbfunc, db)

with mp.Pool(8) as pool:
    pool.starmap(db_partial, out.items())
