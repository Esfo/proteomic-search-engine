from database import environment
import os

def library_initiation(librarylocation, dividingthreshold, subisotopomericdepth, mapsize):
    
    if os.path.isdir(librarylocation):
        if not librarylocation.endswith('/'):
            librarylocation = ''.join((librarylocation, '/'))
        files = os.listdir(librarylocation)
        if 'data.mdb' in files:
            datafile = ''.join((librarylocation, 'data.mdb'))
            os.remove(datafile)
            print('old database file removed')
        if 'lock.mdb' in files:
            lockfile = ''.join((librarylocation, 'lock.mdb'))
            os.remove(lockfile)
            print('old lock file removed')
    
    enzymes = {'trypsin': [{'K': 1, 'R': 1}, {'K': ('KP',), 'R': ('RP',)}],
               'trypsin+lys-c': [{'K': 1, 'R': 1}, {}],
               'arg-c': [{'R': 1}, {}],
               'asp-n': [{'D': 0}, {}],
               'chymotrypsin': [{'Y': 1, 'F': 1, 'W': 1}, {}],
               'glu-c': [{'E': 1}, {}],
               'lys-c': [{'K': 1}, {}]
               }
    
    compmodifications = {'acrylamide': {('C',): {'C': 3, 'H': 5, 'N':1, 'O':1}},
                     'carbamidomethylation': {('C',): {'C': 2, 'H': 3, 'N': 1, 'O': 1}},
                     'phosphorylation': {('S', 'T', 'Y'): {'O': 3, 'H': 1, 'P': 1}}
                     }
    massmodifications = {'ubiquitination': {('K', ): 383.228103}}
    
    with environment(librarylocation, map_size=mapsize) as env:
        enzymedb = env.open_db('enzymes'.encode())
        with env.begin(write=True) as txn:
            with txn.cursor(enzymedb) as cursor:
                for n, params in enumerate(enzymes.items()):
                    cursor.put(str(n).encode(), str(params).encode())
        modificationdb = env.open_db('modifications'.encode())
        modcount = 0
        with env.begin(write=True) as txn:
            with txn.cursor(modificationdb) as cursor:
                for mod, comp in compmodifications.items():
                    for aas, comp in comp.items():
                        for aa in aas:
                            param = [mod, aa, comp]
                            cursor.put(str(modcount).encode(), str(param).encode())
                            modcount += 1
                for mod, mdict in massmodifications.items():
                    for aas, mass in mdict.items():
                        for aa in aas:
                            param = [mod, aa, mass]
                            cursor.put(str(modcount).encode(), str(param).encode())
                            modcount += 1
        defaults = env.open_db('defaults'.encode())
        with env.begin(write=True) as txn:
            with txn.cursor(defaults) as cursor:
                cursor.put('dividingthreshold'.encode(), str(dividingthreshold).encode())
                cursor.put('subisotopomericdepth'.encode(), str(subisotopomericdepth).encode())
                cursor.put('mapsize'.encode(), str(mapsize).encode())
                cursor.put('enzymes'.encode(), str(len(enzymes)).encode())
                cursor.put('modifications'.encode(), str(modcount).encode())
                cursor.put('proteomes'.encode(), str(0).encode())

def enzyme_addition(librarylocation, enzyme, cutters, noncutters={}):
    params = [enzyme, [cutters, noncutters]]
    with environment(librarylocation) as env:
        while True:
            try:
                defaults = env.open_db('defaults'.encode())
                with env.begin(write=True) as txn:
                    with txn.cursor(defaults) as cursor:
                        enzymecount = int(cursor.get('enzymes'.encode()).decode())
                        cursor.put('enzymes'.encode(), str(enzymecount+1).encode())
                enzymedb = env.open_db('enzymes'.encode())
                with env.begin(write=True) as txn:
                    with txn.cursor(enzymedb) as cursor:
                        cursor.put(str(enzymecount).encode(), str(params).encode())
                break
            except lmdb.MapFullError:
                defaults = env.open_db('defaults'.encode())
                with env.begin(write=False) as txn:
                    with txn.cursor(defaults) as cursor:
                        mapaddition = int(cursor.get('mapsize'.encode()).decode())
                newmapsize = env.info()['map_size'] + mapaddition
                env.set_mapsize(newmapsize)

def composition_modification_addition(librarylocation, modstring, aas, compstring):
    compdict = {} #element: count
    countstring = ''
    elementstring = ''
    for n, s in enumerate(compstring):
        if s.isalpha():
            if n > 0 and countstring:
                #new element found, make this one and reset elementstring
                compdict[elementstring] = int(countstring)
                elementstring = s
                countstring = ''
            else:
                #element has multiple letters or initial element
                elementstring += s
        elif s.isdigit():
            countstring += s
    compdict[elementstring] = int(countstring)
    with environment(librarylocation) as env:
        while True:
            try:
                defaults = env.open_db('defaults'.encode())
                with env.begin(write=True) as txn:
                    with txn.cursor(defaults) as cursor:
                        modcount = int(cursor.get('modifications'.encode()).decode())
                        cursor.put('modifications'.encode(), str(modcount+len(aas)).encode())
                modificationdb = env.open_db('modifications'.encode())
                with env.begin(write=True) as txn:
                    with txn.cursor(modificationdb) as cursor:
                        for aa in aas:
                            param = [modstring, aa, compdict]
                            cursor.put(str(modcount).encode(), str(param).encode())
                            modcount += 1
                break
            except lmdb.MapFullError:
                defaults = env.open_db('defaults'.encode())
                with env.begin(write=False) as txn:
                    with txn.cursor(defaults) as cursor:
                        mapaddition = int(cursor.get('mapsize'.encode()).decode())
                newmapsize = env.info()['map_size'] + mapaddition
                env.set_mapsize(newmapsize)

def mass_modification_addition(librarylocation, modstring, aas, mass):
    with environment(librarylocation) as env:
        while True:
            try:
                defaults = env.open_db('defaults'.encode())
                with env.begin(write=True) as txn:
                    with txn.cursor(defaults) as cursor:
                        modcount = int(cursor.get('modifications'.encode()).decode())
                        cursor.put('modifications'.encode(), str(modcount+len(aas)).encode())
                modificationdb = env.open_db('modifications'.encode())
                with env.begin(write=True) as txn:
                    with txn.cursor(modificationdb) as cursor:
                        for aa in aas:
                            param = [modstring, aa, mass]
                            cursor.put(str(modcount).encode(), str(param).encode())
                            modcount += 1
                break
            except lmdb.MapFullError:
                defaults = env.open_db('defaults'.encode())
                with env.begin(write=False) as txn:
                    with txn.cursor(defaults) as cursor:
                        mapaddition = int(cursor.get('mapsize'.encode()).decode())
                newmapsize = env.info()['map_size'] + mapaddition
                env.set_mapsize(newmapsize)

def library_info(librarylocation):
    with environment(librarylocation) as env:
        print('library parameters:')
        defaults = env.open_db('defaults'.encode())
        with env.begin(write=False) as txn:
            with txn.cursor(defaults) as cursor:
                dividingthreshold = float(cursor.get('dividingthreshold'.encode()).decode())
                subisotopomerdepth = float(cursor.get('subisotopomericdepth'.encode()).decode())
                enzymecount = int(cursor.get('enzymes'.encode()).decode())
                modcount = int(cursor.get('modifications'.encode()).decode())
                pcount = int(cursor.get('proteomes'.encode()).decode())
                print('dividing threshold:', dividingthreshold)
                print('subisotopomeric depth:', subisotopomerdepth)
                print('enzyme count:', enzymecount)
                print('modification count:', modcount)
                print('proteome count:', pcount)
        print('~~~~~~~~~~~~~~~~~~~~~~')
        print('library enzymes:')
        enzymedb = env.open_db('enzymes'.encode())
        with env.begin(write=False) as txn:
            with txn.cursor(enzymedb) as cursor:
                for k, v in cursor:
                    n = k.decode()
                    enzyme, (cutters, noncutters) = eval(v.decode())
                    print(n, '--', f'{enzyme}:')
                    for aa, site in cutters.items():
                        if site == 0:
                            cutsite = 'n-terminus'
                        else:
                            cutsite = 'c-terminus'
                        if aa in noncutters:
                            print(aa, 'cuts at', cutsite, 'and not at', ','.join((noncutters[aa])))
                        else:
                            print(aa, 'cuts at', cutsite)
                    print('~~')
        print('~~~~~~~~~~~~~~~~~~~~~~')
        print('library modifications:')
        modificationdb = env.open_db('modifications'.encode())
        with env.begin(write=False) as txn:
            with txn.cursor(modificationdb) as cursor:
                for k, v in cursor:
                    n = k.decode()
                    mod, aa, modtype = eval(v.decode())
                    outputs = n, eval(v.decode())
                    match modtype:
                        case str():
                            print(n, '|', aa, mod, float(modtype))
                        case dict():
                            compstring = ''.join((i[0] + f'({i[1]})' for i in sorted(modtype.items())))
                            print(n, '|', aa, mod, compstring)
        print('~~~~~~~~~~~~~~~~~~~~~~')
        try:
            formuladb = env.open_db(('proteomes.formulalist').encode())
            with env.begin(write=False) as txn:
                with txn.cursor(proteomedb) as cursor:
                    proteomelist = []
                    for k, v in cursor:
                        proteomelist.appened(k)
            print('proteomes:')
            for ptm in proteomelist:
                proteomedb = env.open_db((ptm + '.info').encode())
                with env.begin(write=False) as txn:
                    with txn.cursor(proteomedb) as cursor:
                        print(ptm)
                        vdict = eval(cursor.get('variablemods'.encode()).decode())
                        sdict = eval(cursor.get('staticmods'.encode()).decode())
                        enzymename = cursor.get('enzyme'.encode()).decode()
                        nproteins = cursor.get('nproteins'.encode()).decode()
                        nformulas = cursor.get('nformulas'.encode()).decode()
                        uppermasslimit = cursor.get('uppermasslimit'.encode()).decode()
                        for aa, mods in vdict.items():
                            for mod, modtype in mods.items():
                                match modtype:
                                    case dict():
                                        compstring = ''.join((i[0] + f'({i[1]})' for i in sorted(modtype.items())))
                                        print(aa, '-', mod, 'as', compstring)
                                    case str():
                                        print(aa, '-', mod, 'as', float(modtype))
                        print('enzyme', enzymename)
                        print('number of proteins:', nproteins)
                        print('number of formulas:', nformulas)
                        print('upper peptide mass limit:', uppermasslimit)
                        print('~~~~~~')
        except UnboundLocalError:
            pass
