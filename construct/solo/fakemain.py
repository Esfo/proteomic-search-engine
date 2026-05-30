from linemodel import line_model
from distributionassembly import distribution_assembly
from chargehandling import charge_handling

from time import time
import sys
import gc
gc.enable()

#file processing:
searchcommands = {}

#necessary searchcommands
searchcommands['library'] = None
searchcommands['file'] = None
searchcommands['processinglocation'] = None

#modifiable parameters
searchcommands['minpoints'] = 3
searchcommands['minmovinginds'] = 10
searchcommands['deadsignal'] = 20
searchcommands['chargetolerance'] = 0.1
searchcommands['nprocs'] = -1


if __name__ == '__main__':
    t = time()
    primarycommand = sys.argv[1]
    arguments = sys.argv[2:]
    if primarycommand == 'search':
        #add a mkdir here for the processinglocation if it doesn't exist
        for i in range(len(arguments)):
            a = arguments[i]
            if a.startswith('--'):
                a = a.strip('--')
                if a in searchcommands:
                    if len(arguments) > i + 1:
                        newarg = arguments[i+1]
                        try:
                            if '.' in newarg:
                                newarg = float(newarg)
                            else:
                                newarg = int(newarg)
                        except ValueError: #it's a string
                            pass
                        searchcommands[a] = newarg
        
        allclear = True
        if searchcommands['file'] == None:
            print('Missing file input')
            allclear = False
        if searchcommands['processinglocation'] == None:
            print('Missing output folder')
            allclear = False
        if searchcommands['library'] == None:
            print('Missing library input')
            allclear = False
        if allclear:
            
            line_model(searchcommands['file'], searchcommands['minpoints'], searchcommands['minmovinginds'], searchcommands['deadsignal'], searchcommands['chargetolerance'], searchcommands['library'], searchcommands['processinglocation'])
            distribution_assembly(searchcommands['minpoints'], searchcommands['chargetolerance'], searchcommands['library'], searchcommands['processinglocation'])
            charge_handling(searchcommands['processinglocation'])
            
            print(time() - t, 'total')
