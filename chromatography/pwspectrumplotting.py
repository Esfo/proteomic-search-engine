import matplotlib.patches as patches
#etest = ef.set_index('m/z').loc[:,'intensity']
#etest = etest.sum(level=0)
#highout = etest.sort_values().index[:20]
#
#ts = np.where(mzindex == highout[8])[0][0]
#testout = sm[ts-500:ts+500].todense()
#plt.imshow(testout, cmap='GnBu', vmin=0, vmax=1, aspect='auto')

#def find_nearest(array, value):
#    array = np.asarray(array)
#    idx = (np.abs(array - value)).argmin()
#    return idx
#
#lim1 = 6800
#lim2 = 7500
#lim1 = 0
#lim2 = len(scanarray)
#
#plotinds = np.unique(np.sort(distinds, axis=1), axis=0)
#pswaps = np.swapaxes(newinds[plotinds], 1, 2)
#plotmaxes = pswaps.max(axis=0).max(axis=1)
#plotmins = pswaps.min(axis=0).min(axis=1)
#
#fig = plt.figure(figsize=(6,8), constrained_layout=False)
#gs =  fig.add_gridspec(nrows=3, ncols=1, wspace=0.33)
#ax0 = fig.add_subplot(gs[:2])
#ax1 = fig.add_subplot(gs[2:], sharex=ax0)
#
##ax0.plot(np.arange(len(scanarray)), channel, '.', markersize=1)
##ax0.plot(np.arange(len(scanarray)), savgol2, '-', linewidth=0.5)
##ax0.vlines(ms, 0, channel.max(), color='black', linewidth=0.5)
#ax0.set_xlim(lim1,lim2)
#
##instead of getting the axes, make a scatter plot of np.where from any value > 0 in ochan
##[i for i in ax1.get_yticklabels() if int(i.get_text()) >= 0]
##ax1.set_yticklabels([omasses[ax1.get_yticks()[n].astype(int)] if i else ax1.get_yticks()[n] for n, i in enumerate(np.logical_and(ax1.get_yticks() >= 0, ax1.get_yticks() < len(omasses)))])
#
##ax1.imshow(w2, cmap='GnBu', vmin=0, vmax=1, aspect='auto', extent=[plotmins[1], plotmaxes[1], plotmaxes[0], plotmins[0]])
##for y ,x in pswaps:
##    ax1.plot(x, y, color='black', linewidth=0.2)
##ax1.vlines(ms, 0, ax1.get_yticks()[:-1].max(), color='black', linewidth=0.5)
###ax1.set_xlim(find_nearest(scanarray, lim1),find_nearest(scanarray, lim2))
##ax1.set_xlim(lim1,lim2)
###ax1.set_ylim(270,310)
#
##ax1.vlines(ms, 0, ax1.get_yticks()[:-1].max(), color='black', linewidth=0.5)
##ax1.imshow(window, cmap='GnBu', aspect='auto', vmin=0, vmax=1)
##ax1.set_ylim(305, 315)
##ax1.set_xlim(3320,3430)
#
#ax1.plot(newinds[:,1], newinds[:,0], '.', markersize=0.5)
#
##for r in rings:
##    ringinds = newinds[distinds[r]]
##    rmins = ringinds.min(axis=(0,1))
##    rmaxes = ringinds.max(axis=(0,1))
##    height, width = rmaxes - rmins
##    rect = patches.Rectangle((rmins[1], rmins[0]), width, height, linewidth=3, edgecolor='red', facecolor='none')
##    ax1.add_patch(rect)
#
##ringinds = newinds[distinds[r]]
##rmins = ringinds.min(axis=(0,1))
##rmaxes = ringinds.max(axis=(0,1))
##height, width = rmaxes - rmins
##rect = patches.Rectangle((rmins[1], rmins[0]), width, height, linewidth=3, edgecolor='red', facecolor='none')
##ax1.add_patch(rect)
#
#for er in ringregions:
##for er in ercopy:
#    t, b, l, r = er
#    width = r - l
#    height = b - t
#    rect = patches.Rectangle((l, t), width, height, linewidth=2, edgecolor='red', facecolor='none')
#    ax1.add_patch(rect)
#
##ax1.plot([rmins[1], rmaxes[1]], [rmins[0], rmaxes[0]], '*', color='purple')
#
##ax1.set_ylim(0,500)
##ax1.set_xlim(15000,15500)
#plt.show()

#ain't better tbh
#vals = window[inds[:,0], inds[:,1]]
#plt.plot(scanarray[inds[:,1]], masses[inds[:,0]], '.', markersize=0.1)
#plt.show()

#plt.bar(dcounts.keys(), dcounts.values())
#plt.hlines(dvals.mean(), 0, max(dkeys), color='black', linewidth=0.3)
#plt.vlines(cutkey, 0, max(dvals), color='black', linewidth=0.3)
#plt.title('Distribution of Nearest Neighbor Distance')
#plt.ylabel('Number of Points with Distance $x$')
#plt.xlabel('Integer Distance from Nearest Neighbor')
#plt.show()
#plt.imshow(window, cmap='GnBu', vmin=0, vmax=1, aspect='auto')
#plt.title('Before')
#plt.ylabel('Mass Index')
#plt.xlabel('Time Index')
#plt.show()
#plt.imshow(w2, cmap='GnBu', vmin=0, vmax=1, aspect='auto')
#plt.title('After')
#plt.ylabel('Mass Index')
#plt.xlabel('Time Index')
#plt.show()
#print(w2.sum()/window.sum())

#fig, ax = plt.subplots(figsize=(6,4))
#ax.plot(newinds[:,1], newinds[:,0], '.', markersize=0.5)
#
#it would be good to have different colored rectangles for different steps of the peak group refinement process as a demonstration
#for er in expandedrings:
#    t, b, l, r = er
#    width = r - l
#    height = b - t
#    rect = patches.Rectangle((l, t), width, height, linewidth=3, edgecolor='red', facecolor='none')
#    ax.add_patch(rect)
#
#plt.xlim(17000,20000)
#plt.ylim(0,600)
#plt.show()

#plt.plot(massinds, massinfo, '.-')
##plt.vlines(massinds[massmaxes], 0, massinfo[massmaxes], color='black')
#plt.show()
#plt.plot(scaninds, fullchrom, '.-')
##plt.vlines(scaninds[chrommaxes], 0, chrominfo[chrommaxes], color='black')
#plt.show()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4))

ax.plot(inds[:,1], inds[:,0], '.', markersize=0.5, color='crimson', alpha=0.5)
ax.plot(newinds[:,1], newinds[:,0], '.', markersize=0.5, color='indigo', alpha=0.5)
#ax.plot(timearray[inds[:,1]], masses[inds[:,0]], '.', markersize=0.5, color='crimson', alpha=0.5)
#ax.plot(timearray[newinds[:,1]], masses[newinds[:,0]], '.', markersize=0.5, color='indigo', alpha=0.5)
for t, b, l, r in ringregions:
    width = r - l
    height = b - t
    #width = timearray[r-1] - timearray[l]
    #height = masses[b-1] - masses[t]
    rect = patches.Rectangle((l, t), width, height, linewidth=2, edgecolor='red', facecolor='none')
    #rect = patches.Rectangle((timearray[l], masses[t]), width, height, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
plt.title(coordwindow - windowsize)
plt.show()

