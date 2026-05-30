

linesofscans = {0: [0, 1, 2], #scan: [lines]
                1: [1, 2],
                2: [1, 2],
                3: [2],
                4: [3],
                5: [3, 4],
                6: [5]}


intensitiesoflines = {0: 1e2.43,
                      1: 1e3.34343,
                      2: 1e3.9834,
                      3: 1e6.88,
                      4: 1e5.62,
                      5: 1e3.56}

fragsignalsoflines = {}

ms2scans = {}

scangroups = []

#i expect more difficulty in using intensity to correctly determine which ms2 signals belong where once their intensity goes down, the lower the intensity the harder the guess
#in this light, a good metric for determining entropic coverage would be % of intensity within an ms2 scan being correctly assigned to the appropriate line
#so the small ones matter less, which is proportional

#1 line - everything is basically given to that
#2 lines - infer quantitative differences from intensities
# - i think the differences at this level can be calculated from the difference of the expected % and the % given by that of all the other identified signals
# - maybe a reason to infer individual ions would be if a single signal completes the % match better? -> i think this should be left at the assembly stage, not the setup - which is what i'm doing
#3+ lines - the constraints on the value inference should lead to more accurate boundaries

#so how does it work when an overlapping fragment mass between 2 lines is very fragmentable for one line and not for the other?
#aka the growth rate of either is different
#i guess it would be somewhat harder to model but you can also try and base the %s on how the rest of the scan adds up % wise when comparing ms1 signals
#and let the %s speak for themselves within some level of variance on exact quantity

#i need to look at data on real examples of these
#like, will you see the ms1% increase in ms2 data?
#i guess i could just model it and expect the behavior

#i want an intersection merge of overlapping mass inds, but i only want the inds to be labeled as a different mainind if they connect via actual lines
#this can be handled after the nearest neighbors by doing a second intersection merge on which lines each primaryind belongs to?
#and things would need to be within radius of each other a 2nd time seeing as the first pass links things further away by grouping more than 1 potential ion from each scan
#does this contradict the scan-scan comparisons?
