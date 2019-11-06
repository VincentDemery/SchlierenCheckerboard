"""
Plots the height gradient and the curvature field.
"""

import numpy as np
from matplotlib import pyplot as plt

############################################################
#
#  Main
#
############################################################



disp = np.load('Exp26-25000-25900/Substack (25000-25900-25)0015_gradh.npy')
C = np.load('Exp26-25000-25900/Substack (25000-25900-25)0015_curv.npy')



npts = len(disp[:,0])

fig, ax = plt.subplots(1, 2, figsize=(8, 8))
#ax.scatter(disp[:,1], disp[:,0], c=range(npts))
ax[0].set_title('height gradient')
ax[0].quiver(disp[:,1], disp[:,0], disp[:,3], disp[:,2], angles='xy', 
             scale_units='xy', scale=1)

ax[1].set_title('curvature')
pcm = ax[1].pcolormesh(C[:,:,1], C[:,:,0], C[:,:,2], cmap='RdBu_r')
fig.colorbar(pcm, ax=ax, orientation='vertical')

plt.show()
