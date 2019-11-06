
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors



imid = 5

data = np.load('images_exemple/output-{:03d}_height.npy'.format(imid))

X = data[:,:,0]
Y = data[:,:,1]
Z = data[:,:,2]

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

pcm = ax.pcolormesh(X, Y, Z, cmap='RdBu_r', vmin=-np.max(Z))
fig.colorbar(pcm, ax=ax, orientation='vertical')

plt.show()
