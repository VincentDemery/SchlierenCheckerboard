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

from scipy.interpolate import Rbf


immax = 9
match_range = 5
ymin, ymax, xmin, xmax = 150, 450, 75, 375

############################################################
#
#  Functions
#
############################################################


def coords (img) :
    """finds the corners on an image"""
    c = corner_peaks(corner_harris(img), min_distance=5)
    c0 = corner_subpix(img, c, window_size=13)
    return c0


def matching_points (c1, c2, match_range) :
    """finds the matching points between two lists of points"""
    mpts = []
    for i in range(np.shape(c1)[0]) :
        x1, y1 = c1[i, 0], c1[i, 1]
        dists2 = (c2[:,0]-x1)**2+(c2[:,1]-y1)**2
        if min(dists2) < match_range**2 :
            j = np.where(dists2 == min(dists2))[0][0]
            mpts.append([i,j])
            
    return mpts
    
def compose_mpts (m1, m2) :
    """compose two sets of matching points"""
    m = []
    for p in m1 :
        for pp in m2 :
            if p[1] == pp[0] :
                m.append([p[0],pp[1]])
    return m
    
def matching_coords_displ (c1, c2, m) :
    """coordinates and displacements of two sets points
       with the match list"""
       
    mc1 = []
    mc2 = []
    disp = []
    for match in m :
        mc1.append(c1[match[0], :])
        mc2.append(c2[match[1], :])
        disp.append(c2[match[1], :]-c1[match[0], :])
        
    return np.array(mc1), np.array(mc2), np.array(disp)
        
############################################################
#
#  Main
#
############################################################


imfiles = ['images_exemple/output-{:03d}.jpg'.format(i) for i in range(1, immax+1)]
imgs = [img_as_float(imread(imfile, as_gray=True)) for imfile in imfiles]

corners = [coords(img) for img in imgs]

mpts = []
for i, c in enumerate(corners[:-1]) :
    mpts.append(matching_points(corners[i], corners[i+1], match_range))
    
m = mpts[0]
for i, mp in enumerate(mpts[1:]) :
    m = compose_mpts(m, mp)
  
nb_matching_points = len(m)
print(nb_matching_points, 'matches found')

mc1, mc2, disp = matching_coords_displ(corners[0], corners[-1], m)


rbfx = Rbf(mc1[:,0], mc1[:,1], disp[:,0])
rbfy = Rbf(mc1[:,0], mc1[:,1], disp[:,1])

x = np.linspace(xmin, xmax)
y = np.linspace(ymin, ymax)

X, Y = np.meshgrid(x, y)

dx = rbfx(X, Y)
dy = rbfy(X, Y)


#-------------------------------------------
#  Plot
#-------------------------------------------


fig, ax = plt.subplots(1, 4)
ax[0].set_title('initial image')
ax[0].imshow(imgs[0], cmap=plt.cm.gray)
ax[0].scatter(mc1[:, 1], mc1[:, 0], c=range(nb_matching_points))
ax[0].axis((ymin, ymax, xmin, xmax))

ax[1].set_title('final image')
ax[1].imshow(imgs[-1], cmap=plt.cm.gray)
ax[1].scatter(mc2[:, 1], mc2[:, 0], c=range(nb_matching_points))
ax[1].axis((ymin, ymax, xmin, xmax))

ax[2].set_title('displacement')
ax[2].imshow(imgs[0], cmap=plt.cm.gray)
ax[2].quiver(mc1[:, 1], mc1[:, 0], disp[:, 1], disp[:, 0], angles='xy', scale_units='xy', scale=1)
#ax[2].quiver(Y, X, dy, dx, angles='xy', scale_units='xy', scale=1, color='blue')
ax[2].axis((ymin, ymax, xmin, xmax))

ax[3].set_title('interpolated displacement')
ax[3].imshow(imgs[0], cmap=plt.cm.gray)
ax[3].quiver(Y, X, dy, dx, angles='xy', scale_units='xy', scale=1)
ax[3].axis((ymin, ymax, xmin, xmax))


plt.show()
