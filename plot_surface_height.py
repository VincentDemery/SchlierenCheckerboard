import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

#imid = 5
#heightfile = 'images_exemple/output-{:03d}_height.npy'.format(imid)
heightfile = 'Exp26-25000-25900/Substack (25000-25900-25)0015_height.npy'


data = np.load(heightfile)

X = data[:,:,0]
Y = data[:,:,1]
Z = data[:,:,2]

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis)

plt.show()
