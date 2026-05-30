import numpy as np
from math import pi

columnlength = 150 #mm
columnlengthcm = columnlength / 10
columninnerdiameter = 75 #um
columnradius = columninnerdiameter / 2 #um
columnradiuscm = columnradius / 1000 / 10
particlediameter = 3 #um
particlediametercm = particlediameter / 1000 / 10

flowrate = 300 #nL/min
flowratemlsec = flowrate / 60 / 1000 / 1000 #mL/sec
flowratemlmin = flowrate / 1000 / 1000

epsilon = 0.425

temperature = 60 #celsius
abstemp = temperature + 273.15 #kelvin
percentacetonitrile = 0.05

centipoise = np.exp(percentacetonitrile * (-3.476 + (726 / abstemp)) + (1 - percentacetonitrile) * (-5.414 + (1566 / abstemp)) + percentacetonitrile * (1 - percentacetonitrile) * (-1.762 + (929 / abstemp))) #viscosity

poise = centipoise / 100 #g/(cm-sec)

linearvelocity = (flowratemlsec / (pi * columnradiuscm**2)) / epsilon #cm/sec
specificcolumnpermeability = (180 * (1 - epsilon)**2) / (particlediametercm**2 * epsilon**3)

pressure = (flowratemlmin * 180 * poise * columnlengthcm * (1 - epsilon)**2) / ((particlediametercm**2) * (epsilon**2) * pi * (columnradiuscm**2) * 60) / epsilon

pressurebar = pressure / 1000 / 1000
