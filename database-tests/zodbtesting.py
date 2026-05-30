import ZODB, ZODB.FileStorage
import transaction
import numpy as np

dbloc = '/home/sfo/test/testbase.fs'

storage = ZODB.FileStorage.FileStorage(dbloc)
db = ZODB.DB(storage)
connection = db.open()
root = connection.root

narrays = 200000
minlength = 50
maxlength = 1000
out = {}
for o in range(narrays):
    rl = np.random.randint(minlength+1, maxlength)
    out[str(o)] = np.random.uniform(size=(2, rl))

root.test = {}
root.test['ja'] = 'hey'

root.out = {}

for k, v in out.items():
    root.out[k] = v

root.out.update(out)

root.wat = {}
root.wat.update(out)

transaction.commit() #saved, somehow

db.close()

#modifying an existing db doesn't seem to work, and large transactions somehow take a massive memory consumption
