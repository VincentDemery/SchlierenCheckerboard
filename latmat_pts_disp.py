"""
Curvature from lattice matched points
=====================================

Takes a list of lattice matched points and returns the displacement.

"""

import numpy as np
from scipy.interpolate import Rbf

pixmm = 0.025   # Size of a pixel (mm)
sqmm = 0.5      # Size of a square (mm)

lmpts = np.loadtxt('Exp26-25000-25900/Substack (25000-25900-25)0008_ptslm.dat')


############################################################
#
#  Functions
#
############################################################

def point_from_indices(lmpts, i, j) :
    """Extracts a point with given indices"""
    i0 = set(np.where(lmpts[:,2] == i)[0]) & set(np.where(lmpts[:,3] == j)[0])
    return lmpts[list(i0)[0], :2]


def base(lmpts) :
    """Returns the central point and the base of a list of
       lattice matched points"""
    
    xy0 = point_from_indices(lmpts, 0, 0)
    xy1 = point_from_indices(lmpts, 1, 0)
    v1 = xy1 - xy0
    v2 = np.array([-v1[1], v1[0]])
    
    return xy0, v1, v2

    
def displacement(lmpts, pixmm, sqmm) :
    npts = len(lmpts[:,0])
    
    xy0, v1, v2 = base(lmpts)
    v1 = v1*sqmm/np.sqrt(sum(v1**2))
    v2 = v2*sqmm/np.sqrt(sum(v2**2))
    
    coords0 = np.tensordot(lmpts[:,2], v1, axes=0) +\
              np.tensordot(lmpts[:,3], v2, axes=0)
              
    coords = pixmm*(lmpts[:,:2] - np.tensordot(np.ones(npts), xy0, axes=0))
    
    A = np.zeros((npts, 4))
    A[:,:2] = coords0
    A[:,2:] = coords - coords0
    
    return A

    
############################################################
#
#  Main
#
############################################################

disp = displacement(lmpts, pixmm, sqmm)

# converts displacement to height gradient
#disp[:,2:] = -disp[:,2:]/(alpha*h)

np.save('Exp26-25000-25900/Substack (25000-25900-25)0008_disp.npy', disp)
