import sympy as sp
from scipy import special
from collections import Counter
import numpy as np

#u, x, y = sp.symbols('u x y')
#c12, c13, h1, h2, o16, o17, o18 = sp.symbols('c12 c13 h1 h2 o16 o17 o18')
c12, c13, h1, h2, o16, o17, o18, n14, n15, s32, s33, s34, s36 = sp.symbols('c12 c13 h1 h2 o16 o17 o18 n14 n15 s32 s33 s34 s36')

h1prob = 0.999885
h2prob = 0.000115
c12prob = 0.9893
c13prob = 0.0107
n14prob = 0.99636
n15prob = 0.00364
o16prob = 0.99757
o17prob = 0.00038
o18prob = 0.00205
s32prob = 0.9499
s33prob = 0.0075
s34prob = 0.0425
s36prob = 0.0001

#eq = - y - sp.Integral(x**2, (x, 1, u))

#eq = (0.988922*c12 + 0.011078*c13)**6 * (0.99984426*h1 + 0.00015574*h2)**12 * (0.997628*o16 + 0.000372*o17 + 0.002*o18)**6

#this will take forever, > 15 minutes for this below
carbons = 0
hydrogens = 0
nitrogens = 0
oxygens = 3
sulfurs = 0

eq = (c12prob*c12 + c13prob*c13)**carbons * (h1prob*h1 + h2prob*h2)**hydrogens * (o16prob*o16 + o17prob*o17 + o18prob*o18)**oxygens * (s32prob*s32 + s33prob*s33 + s34prob*s34 + s36prob*s36)**sulfurs * (n14prob*n14 + n15prob*n15)**nitrogens



#eq2 = (0.989*c12 + 0.011*c13)**2
#ed = eq.doit()

#lam = sp.lambdify(y, ed)
#vals = lam(np.arange(5000)).tolist()

#output = [float(sp.solveset(i, domain=sp.S.Reals).args[0]) for i in vals]

#longtime = [sp.solveset(ed, u, domain=sp.S.Reals).subs(y, i) for i in np.arange(5000).tolist()]

n = 0
expansions = {}
for a in eq.expand().args:
    n += 1
    val = float(a.args[0])
    topes = ','.join((str(n), ','.join(([str(i) for i in a.args[1:]]))))
    expansions[topes] = val
expansions =  Counter(expansions)

ls = 2
lm = 1

rs = 3
rm = 2
