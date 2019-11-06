"""
================
Corner detection
================

Detect corner points using the Harris corner detector and determine the
subpixel position of corners ([1]_, [2]_).

.. [1] https://en.wikipedia.org/wiki/Corner_detection
.. [2] https://en.wikipedia.org/wiki/Interest_point_detection

"""

import numpy as np

from matplotlib import pyplot as plt

from skimage.util import img_as_float
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.io import imread


#imid = 1
#imfile = 'images_exemple/output-{:03d}.jpg'.format(imid)
#imfile = '26/Substack (25583).tif'
#imfile ='41/Substack (15557).tif'
imfile = 'Exp26-25000-25900/Substack (25000-25900-25)0008.tif'


match_range = 5
#ymin, ymax, xmin, xmax = 150, 450, 75, 375

############################################################
#
#  Functions
#
############################################################


def coords (img) :
    """finds the corners on an image"""
    c = corner_peaks(corner_harris(img), min_distance=8)
    c0 = corner_subpix(img, c, window_size=13)
    return c0

def closest_point_to (points_list, point) :
    """returns the closest point to a given point in a list"""
    i = np.argmin((points_list[:,0]-point[0])**2+(points_list[:,1]-point[1])**2)
    return i
    
def transfer_point (i, basecor, newcor) :
    newcor.append(basecor[i])
    basecor = np.delete(basecor, i, 0)
    return basecor, newcor

        
############################################################
#
#  Main
#
############################################################

img = img_as_float(imread(imfile, as_gray=True))

cor = coords(img)

basecor = cor

i0 = closest_point_to (cor, np.mean(cor, axis=0))

newcor = []

basecor, newcor = transfer_point(i0, basecor, newcor)

i1 = closest_point_to (basecor, newcor[0])
basecor, newcor = transfer_point(i1, basecor, newcor)

v1 = newcor[1]-newcor[0]
v2 = np.array([-v1[1], v1[0]])

i2 = closest_point_to (basecor, newcor[0] + v2)
basecor, newcor = transfer_point(i2, basecor, newcor)

newcor = np.array(newcor)

cor_lm = []
for c in cor :
    nx = np.round(sum((c - newcor[0,:])*v1)/sum(v1**2))
    ny = np.round(sum((c - newcor[0,:])*v2)/sum(v2**2))
    cor_lm.append([c[0], c[1], nx, ny])

cor_lm = np.array(cor_lm)

#print(newcor)

#-------------------------------------------
#  Plot
#-------------------------------------------
ymin, ymax, xmin, xmax = min(cor[:,1]), max(cor[:,1]), min(cor[:,0]), max(cor[:,0])

dx = max(xmax - xmin, ymax - ymin)

xm = (xmax+xmin)/2.
ym = (ymax+ymin)/2.

xmin = xm - .7*dx
xmax = xm + .7*dx
ymin = ym - .7*dx
ymax = ym + .7*dx

fig, ax = plt.subplots(1, 2)

ax[0].set_title('corners and lattice')
ax[0].imshow(img, cmap=plt.cm.gray)
nl = 7
for i in range(-nl, nl+1) :
    ax[0].plot([newcor[0,1] - nl*v1[1] + i*v2[1], newcor[0,1] + nl*v1[1] + i*v2[1]],
               [newcor[0,0] - nl*v1[0] + i*v2[0], newcor[0,0] + nl*v1[0] + i*v2[0]],
               'r-')
               
    ax[0].plot([newcor[0,1] - nl*v2[1] + i*v1[1], newcor[0,1] + nl*v2[1] + i*v1[1]],
               [newcor[0,0] - nl*v2[0] + i*v1[0], newcor[0,0] + nl*v2[0] + i*v1[0]],
               'r-')
ax[0].scatter(cor[:, 1], cor[:, 0])#, c=range(len(newcor[:,0])))
ax[0].axis((ymin, ymax, xmin, xmax))

ax[1].set_title('matching')
ax[1].imshow(img, cmap=plt.cm.gray)
ax[1].scatter(cor[:, 1], cor[:, 0], c=range(len(cor[:,0])))
ax[1].scatter(cor_lm[:, 2]*v1[1] + cor_lm[:, 3]*v2[1] + newcor[0,1],
              cor_lm[:, 2]*v1[0] + cor_lm[:, 3]*v2[0] + newcor[0,0],
              marker='v', c=range(len(cor[:,0])))
              
ax[1].axis((ymin, ymax, xmin, xmax))

plt.show()
