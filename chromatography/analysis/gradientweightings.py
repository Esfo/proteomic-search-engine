import numpy as np
from matplotlib import pyplot as plt

npoints = 100
if npoints % 2:
    npoints += 1

x = np.arange(npoints) + 1
linear = -x + np.abs(x).max() + 1

plt.plot(x,linear, label='linear')

high = -1 * np.geomspace(x.min(), x.max(), len(x))
high += np.abs(high).max() + 1

low = np.flip(-1 * high + np.abs(high).max() + 1)

halfhigh = np.mean([linear, high], axis=0)
halflow = np.mean([linear, low], axis=0)

plt.plot(x, high, label='high')
plt.plot(x, low, label='low')
plt.plot(x, halfhigh, label='half high')
plt.plot(x, halflow, label='half low')

halfval = npoints // 2

lhhigh = -1 * np.geomspace(x[:halfval].min(), x[:halfval].max() - 1, halfval)
lhhigh += np.abs(high).max() + 1

lhlow = np.flip(-1 * lhhigh + np.abs(lhhigh).max())

highlow = np.hstack((lhhigh, lhlow))
halfhighlow = np.mean([linear, highlow], axis=0)

plt.plot(x, highlow, label='high-low')
plt.plot(x, halfhighlow, label='half high-low')

lhlow = -1 * np.geomspace(x[:halfval].min(), x[:halfval].max() + 1, halfval)
lhlow += np.abs(lhlow).max()

lhhigh = np.flip(-1 * lhlow) + (np.abs(lhlow).max() - 1) * 2 + 2

lowhigh = np.hstack((lhhigh, lhlow))
halflowhigh = np.mean([linear, lowhigh], axis=0)

plt.plot(x, lowhigh, label='low-high')
plt.plot(x, halflowhigh, label='half low-high')

plt.legend(bbox_to_anchor=(1,1))
plt.title(npoints)
plt.show()

plt.plot(x, linear / 2, label='1/2X')
plt.plot(x, linear, label='1X')
plt.plot(x, linear * 2, label='2X')
plt.title(f'linear weightings {npoints}')
plt.legend(bbox_to_anchor=(1.22,1))
plt.show()
