"""
Corner detection and lattice matching
=====================================

- Detect corner points using the Harris corner detector and determine the
subpixel position of corners ([1]_, [2]_).
.. [1] https://en.wikipedia.org/wiki/Corner_detection
.. [2] https://en.wikipedia.org/wiki/Interest_point_detection

- Use the central point and its nearest neighbor to match the points to
lattice points.

- Returns: array containing the absolute and lattice coordinates of the
detected corners.

"""

import numpy as np

from skimage.util import img_as_float
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.io import imread


#imid = 1
#imfile = 'images_exemple/output-{:03d}.jpg'.format(imid)
#imfile = '26/Substack (25583).tif'
#imfile ='41/Substack (15557).tif'
imfile = 'Exp26-25000-25900/Substack (25000-25900-25)0008.tif'


############################################################
#
#  Functions
#
############################################################

def replace_file_extension (fname, ext) :
    return fname[:-1-fname[::-1].find('.')] + ext


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

    
def lattice_matching (cor) :
    """matches a set of points to a square lattice"""
    basecor = cor

    i0 = closest_point_to (cor, np.mean(cor, axis=0))

    newcor = []

    basecor, newcor = transfer_point(i0, basecor, newcor)

    i1 = closest_point_to (basecor, newcor[0])
    basecor, newcor = transfer_point(i1, basecor, newcor)

    v1 = np.array(newcor[1]-newcor[0])
    v2 = np.array([-v1[1], v1[0]])

    cor_lm = []
    for c in cor :
        nx = np.round(sum((c - newcor[0])*v1)/sum(v1**2))
        ny = np.round(sum((c - newcor[0])*v2)/sum(v2**2))
        cor_lm.append([c[0], c[1], nx, ny])
        
    return np.array(cor_lm)

############################################################
#
#  Main
#
############################################################

img = img_as_float(imread(imfile, as_gray=True))

cor = coords(img)

cor_lm = lattice_matching(cor)

cfile = replace_file_extension(imfile, '_ptslm.dat')

np.savetxt(cfile, np.array(cor_lm))
