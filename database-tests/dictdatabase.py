import dictdatabase as DDB
import numpy as np

#make 2 tables in the same db and be able to call up multiple elements from them that match your list

DDB.config.storage_directory = '/home/sfo/test/DDB-test/'
DDB.config.use_compression = False
DDB.config.indent = '\t'
DDB.config.use_orjson = True

narrays = 2000
minlength = 50
maxlength = 1000
out = {}
for o in range(narrays):
    rl = np.random.randint(minlength+1, maxlength)
    out[str(o)] = np.random.uniform(size=(2, rl)).tolist()

DDB.at('test').create(out, force_overwrite=True)

hey = DDB.at('test', key=str(o)).read()

#looks like this doesn't let you modify or add to an existing db
