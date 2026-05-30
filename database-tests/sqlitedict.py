import sqlitedict as sq
import numpy as np

#make 2 tables in the same db and be able to call up multiple elements from them that match your list

dbloc = '/home/sfo/test/testbase.sqlite'
groupname = 'distributions'
dname = 'distributionsbyformula'

narrays = 20
minlength = 5
maxlength = 10
out = {}
for o in range(narrays):
    rl = np.random.randint(minlength+1, maxlength)
    out[str(o)] = np.random.uniform(size=(2, rl))

#so this always opens a table by default, not the file itself, just clarify which table it is and work with that i suppose
db = sq.SqliteDict(dbloc)

#to list table names
#sq.SqliteDict.get_tablenames(dbloc)

dists = sq.SqliteDict(dbloc, tablename='distributions', autocommit=True)

#The `flag` parameter. Exactly one of:
#  'c': default mode, open for read/write, creating the db/table if necessary.
#  'w': open for r/w, but drop `tablename` contents first (start with empty table)
#  'r': open as read-only
#  'n': create a new database (erasing any existing tables, not just `tablename`!).

#you can add nested dictionaries, but they're immutable once they're in the db, only what the table can access through its own keys is mutable

dists.update(out)

#db.commit() #saves changes

#in works, and intersection too
set(out).intersection(dists)

yo = sq.SqliteDict(dbloc, tablename='seqs?', autocommit=True)
