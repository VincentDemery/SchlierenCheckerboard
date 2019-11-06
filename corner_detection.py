"""
Detects the corners on a list of images
"""

import numpy as np

from skimage.io import imread
from skimage.util import img_as_float
from skimage.feature import corner_harris, corner_subpix, corner_peaks



#immax = 9
#img_list = ['images_exemple/output-{:03d}.jpg'.format(i)\
#            for i in range(1, immax+1)]

img_list = ['Exp26-25000-25900/Substack (25000-25900-25){:04d}.tif'.format(i)\
            for i in range(8, 26)]

############################################################
#
#  Functions
#
############################################################

def replace_file_extension (fname, ext) :
    return fname[:-1-fname[::-1].find('.')] + ext


def coords (img_file) :
    """finds the corners on an image"""
    img = img_as_float(imread(img_file, as_gray=True))
    c = corner_peaks(corner_harris(img), min_distance=5)
    c0 = corner_subpix(img, c, window_size=13)
    return np.array(c0)

def coords_export (img_file) :
    corners = coords(img_file)
    print(img_file + ': ' + str(len(corners[:,0])) + ' corners found')
    cfile = replace_file_extension(img_file, '_pts.dat')
    np.savetxt(cfile, corners)
    return 0

############################################################
#
#  Main
#
############################################################

for img_file in img_list :
    coords_export(img_file)
