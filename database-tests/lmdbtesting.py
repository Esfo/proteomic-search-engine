import lmdb
import numpy as np
from time import time
import multiprocessing as mp
from functools import partial

dbname = '/home/sfo/test/example.lmdb'
ki = 1024
mi = 1024 * ki
gi = 1024 * mi
#ti = 1024 * gi
mapsize = 50 * ki

nentries = 100

testdict = {}
for i in np.random.randint(0,9999999999, size=nentries).tolist():
    t = np.random.uniform(size=(2,np.random.randint(10,100)))
    tstore = t.tobytes()
    testdict[str(i).encode()] = tstore

#nt = time()

#environment_partial = partial(lmdb.Environment, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=9999, max_spare_txns=1, lock=True)

##linear
#
##with lmdb.Environment(dbname, map_size=mapsize, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=99999, max_spare_txns=1, lock=True) as env:
#with lmdb.Environment(dbname, map_size=mapsize, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=True, meminit=True, max_readers=126, max_dbs=9999, max_spare_txns=1, lock=True) as env:
#    #with env.open_db('sub-db'.encode()) as child_db: #doesn't work???
#    child_db = env.open_db('sub-db'.encode())
#    child_db2 = env.open_db('sub-db2'.encode())
#    while True:
#        try:
#            with env.begin(write=True) as txn:
#                with txn.cursor(child_db) as cursor1:
#                    with txn.cursor(child_db2) as cursor2:
#                        cursor1.putmulti(testdict.items(), dupdata=False, overwrite=True, append=False)
#                        cursor2.putmulti(testdict.items(), dupdata=False, overwrite=True, append=False)
#                        #cursor.put(str(i).encode(), tstore)
#            break #keep adding gbs until it fits
#        except lmdb.MapFullError:
#            currentmapsize = env.info()['map_size']
#            currentmapsize += gi
#            env.set_mapsize(currentmapsize)
#    print(time() - nt, 'written')
#    child_db2 = env.open_db('sub-db2'.encode())
#    while True:
#        try:
#            with env.begin(write=True) as txn:
#                with txn.cursor(child_db2) as cursor:
#                    cursor.putmulti(testdict.items(), dupdata=False, overwrite=True, append=False)
#                    #cursor.put(str(i).encode(), tstore)
#            break #keep adding gbs until it fits
#        except lmdb.MapFullError:
#            #store gi in a database default
#            currentmapsize = env.info()['map_size']
#            currentmapsize += gi
#            env.set_mapsize(currentmapsize)
#    print(time() - nt, 'written')
#    nt = time()
#
#    remake = {}
#
#    #txn = env.begin()
#    #cursor = txn.cursor(child_db)
#    #for k, v in cursor:
#    #    key = k.decode()
#    #    out = np.frombuffer(v)
#    #    new = out.reshape(2, out.size//2)
#    #    remake[key] = new
#    #    #print(k, new)
#    #print(time() - nt, 'read')



#multiprocessing

with lmdb.Environment(dbname, map_size=mapsize, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=False, writemap=False, meminit=True, max_readers=126, max_dbs=9999, max_spare_txns=1, lock=False) as env:
    child_db = env.open_db('init-db'.encode())
    with env.begin(write=True) as txn:
        with txn.cursor(child_db) as cursor:
            cursor.put('hey'.encode(), 'yo'.encode())

def env_input(k, v):
    with lmdb.Environment(dbname, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=False, readahead=False, writemap=False, meminit=True, max_readers=126, max_dbs=9999, max_spare_txns=1, lock=False) as env:
        child_db = env.open_db('sub-db'.encode())
        with env.begin(write=True) as txn:
            with txn.cursor(child_db) as cursor:
                cursor.put(k, v, dupdata=False, overwrite=True, append=False)

nt = time()

mapsize = 50 * ki

with mp.Pool(8) as pool:
    while True:
        try:
            pool.starmap(env_input, testdict.items())
            break
        except lmdb.MapFullError:
            with lmdb.Environment(dbname, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=False, readahead=False, writemap=False, meminit=True, max_readers=126, max_dbs=9999, max_spare_txns=1, lock=False) as env:
                print('map hit')
                currentmapsize = env.info()['map_size']
                currentmapsize += gi
                env.set_mapsize(currentmapsize)
print(time() - nt, 'written')

remake = {}
with lmdb.Environment(dbname, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=False, readahead=False, writemap=False, meminit=True, max_readers=126, max_dbs=9999, max_spare_txns=1, lock=False) as env:
    child_db = env.open_db('sub-db'.encode())
    txn = env.begin(write=False)
    cursor = txn.cursor(child_db)
    for k, v in cursor:
        key = k.decode()
        out = np.frombuffer(v)
        new = out.reshape(2, out.size//2)
        remake[key] = new
        #print(k, new)
    print(time() - nt, 'read')
