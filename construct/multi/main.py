from collections import defaultdict
from time import time
import sys
import os
import gc
gc.enable()

#library generation:
librarycommands = {}
#ENZYMES?!
#MODIFICATION INPUT?!

#as a precursor to library generation, it would be nice to see min/max peptide masses at different charges along an axis representative of their length, this would help advise min/max length
#and if you can take it a step further to add in vmods and the number of peptides generated from a digest i suppose... that'd be neat

#put solid explanations for how each available parameter affects memory limitations, and if you hit them what you should consider. For example minimumabundance doesn't affect memory much, but it does take the library calculation longer. maxlengths and maxvmods are the main memory consumers.
#it would be good to also have a solid comparator of memory used in the library creation process vs memory used in the search, this gets hard depending on the size of the ms file but i can measure that too
#^these measurements would be good to see from the logs, maybe use number of distributions as a measure instead of memory?

#initiation commands
librarycommands['dividingthreshold'] = 0.1
librarycommands['subisotopomericdepth'] = 0.5

#don't change aye
librarycommands['mapsize'] = 1024**3*2 #~2gb
librarycommands['maxdbs'] = 99999

librarycommands['missedcleavages'] = 2
librarycommands['minlength'] = 6
librarycommands['maxlength'] = 30
librarycommands['maxvmods'] = 3
#^perhaps it would be useful to dictate a top-down vs bottom-up madvmods, ie raw mass additions that don't rely on exact formulas, for vague/unknown modifications?
librarycommands['nprocs'] = os.cpu_count()

librarycommands['librarylocation'] = None
librarycommands['proteomefile'] = None

#file processing:
searchcommands = {}
#i need to be able to print out library parameters, or generate csv's containing them

#necessary searchcommands
searchcommands['librarylocation'] = None
searchcommands['proteome'] = None #make this case independent
searchcommands['file'] = None #mzml
searchcommands['processinglocation'] = None

#modifiable parameters
searchcommands['minpoints'] = 3
searchcommands['minmovinginds'] = 10
searchcommands['deadsignal'] = 20
searchcommands['chargetolerance'] = 0.1
searchcommands['ms1_ppmtolerance'] = 10
searchcommands['ms2_ppmtolerance'] = 15
searchcommands['nprocs'] = os.cpu_count()

#centroiding parameters
searchcommands['centroiding'] = set() #ms levels to centroid, comma-separated
searchcommands['intensitytype'] = 'area'
searchcommands['masstype'] = 'average' #weighted

#search parameters
searchcommands['ions'] = 'by'

#other ideas:
#it would be good to have a --continue flag to continue an analysis where it might have been left off for whatever reason, power/memory/crashing
#have some default enzymes/modifications/losses within the library, make a function to return them as csv's so people can input their own versions of what they want, then make a function for the library to adopt additions from this new bit, with an option to completely overwrite the list of things or to merge the existing + crafted options

#add that if nprocs == 0 it goes to max i guess

proceed = True

if __name__ == '__main__':
    t = time()
    primarycommand = sys.argv[1]
    if primarycommand == 'search':
        arguments = sys.argv[2:]
        #add a mkdir here for the processinglocation if it doesn't exist
        for i in range(len(arguments)): #can't this be above the if statement that determines main commands?
            a = arguments[i]
            if a.startswith('--'):
                a = a.strip('--')
                if a in searchcommands:
                    if len(arguments) > i + 1:
                        newarg = arguments[i+1]
                        match newarg:
                            case int():
                                print('int', newarg)
                                print(a)
                                searchcommands[a] = newarg
                            case str():
                                if a == 'centroiding':
                                    searchcommands[a] = set(map(int, newarg.split(',')))
                                else:
                                    searchcommands[a] = newarg
        
        if searchcommands['file'] == None:
            print('missing file input')
            proceed = False
        if searchcommands['processinglocation'] == None:
            print('missing output folder location')
            proceed = False
        if searchcommands['librarylocation'] == None:
            print('missing library directory')
            proceed = False
        elif not os.path.isdir(searchcommands['librarylocation']):
            print('librarylocation should be a directory')
            proceed = False
        if searchcommands['proteome'] == None:
            print('missing proteome of library')
            proceed = False
        if proceed:
            print('starting', searchcommands['file'])
            basename = searchcommands['file'].split('/')[-1].split('.mzML')[0]
            if not searchcommands['processinglocation'].endswith('/'):
                searchcommands['processinglocation'] = searchcommands['processinglocation'] + '/'
            searchcommands['processingdirectory'] = searchcommands['processinglocation'] + basename + '/'
            
            if not os.path.isdir(searchcommands['processinglocation']):
                os.mkdir(searchcommands['processinglocation'])
            
            if not os.path.isdir(searchcommands['processingdirectory']):
                os.mkdir(searchcommands['processingdirectory'])
            
            from scancentroiding import centroid
            centroid(searchcommands['file'], searchcommands['processingdirectory'], searchcommands['centroiding'], searchcommands['intensitytype'], searchcommands['masstype'])
            
            from linemodel import line_model
            line_model(searchcommands['file'], searchcommands['minpoints'], searchcommands['minmovinginds'], searchcommands['deadsignal'], searchcommands['chargetolerance'], searchcommands['librarylocation'], searchcommands['processingdirectory'], searchcommands['proteome'])
            
            from distributionassembly import distribution_assembly
            distribution_assembly(searchcommands['minpoints'], searchcommands['chargetolerance'], searchcommands['librarylocation'], searchcommands['processingdirectory'], searchcommands['proteome'])
            
            #from chargehandling import charge_handling
            #charge_handling(searchcommands['processingdirectory'], searchcommands['nprocs'])
            
            from scanmatching import scan_matching
            scan_matching(searchcommands['file'], searchcommands['minpoints'], searchcommands['nprocs'], searchcommands['librarylocation'], searchcommands['processingdirectory'], searchcommands['proteome'])
            
            from distributionmatching import distribution_matching
            distribution_matching(searchcommands['ms1_ppmtolerance'], searchcommands['librarylocation'], searchcommands['proteome'], searchcommands['processingdirectory'])
            
            from subformulalinegrouping import subformula_line_grouping
            subformula_line_grouping(searchcommands['librarylocation'], searchcommands['processingdirectory'], searchcommands['proteome'], searchcommands['ions'])
            
            from scanscoring import scan_scoring
            scan_scoring(searchcommands['ms2_ppmtolerance'], searchcommands['processingdirectory'], searchcommands['proteome'], searchcommands['librarylocation'])
            
            #from peptidefragmentscoring import fragment_writer
            #fragment_writer(searchcommands['librarylocation'], searchcommands['processingdirectory'], searchcommands['proteome'], searchcommands['nprocs'], searchcommands['ms2_ppmtolerance'], searchcommands['ions'])
            
            #from csvscoresorting import csv_score_sorting
            #csv_score_sorting(searchcommands['processingdirectory'])
            
            print(time() - t, 'total')
    
    elif primarycommand == 'library':
        #print(sys.argv)
        secondarycommand = sys.argv[2]
        arguments = sys.argv[3:]
        if secondarycommand == 'initiate':
            for i in range(len(arguments)):
                a = arguments[i]
                if a.startswith('--'):
                    a = a.strip('--')
                    if a in librarycommands:
                        if len(arguments) > i + 1:
                            newarg = arguments[i+1]
                            try:
                                if '.' in newarg:
                                    newarg = float(newarg)
                                else:
                                    newarg = int(newarg)
                            except ValueError: #it's a string
                                pass
                            librarycommands[a] = newarg
            if librarycommands['librarylocation'] == None:
                print('missing save directory input, name a folder to make the library inside')
                proceed = False
            if proceed:
                from libraryinitiation import library_initiation
                
                library_initiation(librarycommands['librarylocation'], librarycommands['dividingthreshold'], librarycommands['subisotopomericdepth'], librarycommands['mapsize'])
                
                print(time() - t, 'total')
                
        elif secondarycommand == 'proteome':
            i = 0
            staticmods = []
            variablemods = []
            while i < len(arguments):
                a = arguments[i]
                if a.startswith('--'):
                    a = a.strip('--')
                    if a in librarycommands:
                        if len(arguments) > i + 1:
                            newarg = arguments[i+1]
                            try:
                                if '.' in newarg:
                                    newarg = float(newarg)
                                else:
                                    newarg = int(newarg)
                            except ValueError: #it's a string
                                pass
                            librarycommands[a] = newarg
                            i += 2
                    elif a == 'variablemods':
                        i += 1
                        variablemods = list(map(int, arguments[i].split(',')))
                        i += 1
                    elif a == 'staticmods':
                        i += 1
                        staticmods = list(map(int, arguments[i].split(',')))
                        i += 1
                    elif a == 'enzyme':
                        i += 1
                        enzymename = int(arguments[i])
                        i += 1
                    else:
                        i += 1
                else:
                    i += 1
            if librarycommands['librarylocation'] == None:
                print('missing library file input')
                proceed = False
            if librarycommands['proteomefile'] == None:
                print('missing proteome fasta input')
                proceed = False
            if proceed:
                from libraryadditions import library_additions

                library_additions(librarycommands['librarylocation'], librarycommands['proteomefile'], librarycommands['minlength'], librarycommands['maxlength'], librarycommands['missedcleavages'], librarycommands['nprocs'], enzyme=enzymename, maxvmods=librarycommands['maxvmods'], staticmodifications=staticmods, variablemodifications=variablemods)

                print(time() - t, 'total')
        elif secondarycommand == 'enzyme':
            #python main.py library enzyme --enzyme testenzyme --cut T=0,K=1,J=1 --noncut K=TP --librarylocation ~/data/proteomics/fastas/search-db
            i = 0
            enzymename = False
            cutters = {}
            noncutters = defaultdict(list)
            while i < len(arguments):
                a = arguments[i]
                if a.startswith('--'):
                    a = a.strip('--')
                    if a in librarycommands:
                        if len(arguments) > i + 1:
                            newarg = arguments[i+1]
                            try:
                                if '.' in newarg:
                                    newarg = float(newarg)
                                else:
                                    newarg = int(newarg)
                            except ValueError: #it's a string
                                pass
                            librarycommands[a] = newarg
                            i += 2
                    elif a == 'enzyme':
                        i += 1
                        enzymename = arguments[i]
                        i += 1
                    elif a == 'cut':
                        i += 1
                        a = arguments[i]
                        sites = a.split(',')
                        for site in sites:
                            aa, loc = site.split('=')
                            cutters[aa] = int(loc)
                            #print(aa, loc)
                        i += 1
                    elif a == 'noncut':
                        i += 1
                        while True:
                            a = arguments[i]
                            if a.startswith('--'):
                                break
                            aa, sites = a.split('=')
                            noncuts = sites.split(',')
                            for site in noncuts:
                                noncutters[aa].append(site)
                                #print(aa, site)
                            i += 1
                    else:
                        #print(a)
                        i += 1
                else:
                    #print('fail', a)
                    i += 1
            if librarycommands['librarylocation'] == None:
                print('missing library file input')
                proceed = False
            if not enzymename or not cutters:
                print('missing enzyme information')
                proceed = False
            if proceed:
                from libraryinitiation import enzyme_addition
                
                cutters = dict(cutters)
                noncutters = dict(noncutters)
                enzyme_addition(librarycommands['librarylocation'], enzymename, cutters, noncutters={})
                #print(cutters)
                #print(noncutters)
                #print(librarycommands['librarylocation'])
                #print(enzymename)
        elif secondarycommand == 'modification':
            #tests:
            #python main.py library modification --modification acetylation --composition H2C2O1 --aminoacids KCSTYR --librarylocation /home/sfo/data/proteomics/fastas/search-db
            #python main.py library modification --modification acetylation --mass 42.0367 --aminoacids KCSTYR --librarylocation /home/sfo/data/proteomics/fastas/search-db
            i = 0
            aminoacids = False
            composition = False
            massaddition =  False
            modstring = False
            while i < len(arguments):
                a = arguments[i]
                if a.startswith('--'):
                    a = a.strip('--')
                    if a in librarycommands:
                        if len(arguments) > i + 1:
                            newarg = arguments[i+1]
                            try:
                                if '.' in newarg:
                                    newarg = float(newarg)
                                else:
                                    newarg = int(newarg)
                            except ValueError: #it's a string
                                pass
                            librarycommands[a] = newarg
                            i += 2
                    elif a == 'aminoacids':
                        i += 1
                        aminoacids = arguments[i]
                        i += 1
                    elif a == 'composition':
                        i += 1
                        composition = arguments[i]
                        i += 1
                    elif a == 'mass':
                        i += 1
                        massaddition = arguments[i]
                        i += 1
                    elif a == 'modification':
                        i += 1
                        modstring = arguments[i]
                        i += 1
                    else:
                        print(a)
                        i += 1
                else:
                    print('fail', a)
                    i += 1
            if librarycommands['librarylocation'] == None:
                print('missing library file input')
                proceed = False
            if not modstring:
                print('name your modification')
                proceed = False
            if massaddition and composition:
                print('you need to pick either massaddition or composition')
                proceed = False
            elif massaddition and proceed:
                from libraryinitiation import mass_modification_addition
                
                mass_modification_addition(librarycommands['librarylocation'], modstring, aminoacids, massaddition)
                #print(aminoacids)
                #print(massaddition)
            elif composition and proceed:
                from libraryinitiation import composition_modification_addition
                
                composition_modification_addition(librarycommands['librarylocation'], modstring, aminoacids, composition)
                #print(aminoacids)
                #print(composition)
        elif secondarycommand == 'list':
            for i in range(len(arguments)):
                a = arguments[i]
                if a.startswith('--'):
                    a = a.strip('--')
                    if a in librarycommands:
                        if len(arguments) > i + 1:
                            newarg = arguments[i+1]
                            try:
                                if '.' in newarg:
                                    newarg = float(newarg)
                                else:
                                    newarg = int(newarg)
                            except ValueError: #it's a string
                                pass
                            librarycommands[a] = newarg
            if librarycommands['librarylocation'] == None:
                print('missing library file input')
                proceed = False
            if proceed:
                from libraryinitiation import library_info
                
                #python main.py library list --librarylocation /home/sfo/data/proteomics/fastas/search-db
                library_info(librarycommands['librarylocation'])
    else:
        print('what do you even want')
