"""
Curvature from lattice matched points
=====================================

Takes a list of lattice matched points and a list of remarkable points for other
images, tracks the points and returns the height gradient.

"""

import numpy as np

match_range = 5.

pixmm = 0.025   # Size of a pixel (mm)
sqmm = 0.5      # Size of a square (mm)

clm_file = 'Exp26-25000-25900/Substack (25000-25900-25)0008_ptslm.dat'

c_files = ['Exp26-25000-25900/Substack (25000-25900-25){:04d}_pts.dat'.format(i)\
            for i in range(8, 26)]

############################################################
#
#  Functions
#
############################################################

def change_file_extension (fname, ext1, ext2) :
    return fname[:-len(ext1)] + ext2
    
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
    
    mc1, mc2, disp = np.array(mc1), np.array(mc2), np.array(disp)
    
    A = np.zeros((len(mc1[:,0]), 4))
    A[:,:2] = mc1
    A[:,2:] = disp
    
    return A


def matching_coords_displ2 (c1, c2, m, d01, pixmm) :
    """coordinates and displacements of two sets points
       with the match list, with initial displacement and dimensional
       constants"""
       
    mc1 = []
    mc2 = []
    disp = []
    for match in m :
        xy0 = d01[match[0], :2]
        
        xy1 = c1[match[0], :]
        xy2 = c2[match[1], :]
        d12 = pixmm*(xy2-xy1)

        d02 = d01[match[0], 2:] + d12
        
        disp.append([xy0[0], xy0[1], d02[0], d02[1]])
  
    return np.array(disp)
    

def track_displacement (clm_file, c_files, match_range, pixmm, sqmm):
    """Track points through a list of remarkable points and gets
       the displacement"""
    
    # Displacement of the lattice matched points (1st file)
    lmpts = np.loadtxt(clm_file)
    disp0 = displacement(lmpts, pixmm, sqmm)
    
    corners = [np.loadtxt(cfile) for cfile in c_files]
    
    mpts = []
    for i, c in enumerate(corners[:-1]) :
        mpts.append(matching_points(corners[i], corners[i+1], match_range))
    
    m = mpts[0]
    disp = matching_coords_displ2(corners[0], corners[1], m, disp0, pixmm)
#    disp[:,2:] = -disp[:,2:]/(alpha*h)
    dfile = change_file_extension(c_files[1], 'pts.dat', 'disp.npy')
    np.save(dfile, disp)

    for i in range(1, len(c_files)-1) :
        m = compose_mpts(m, mpts[i])
        disp = matching_coords_displ2(corners[0], corners[i+1], m, disp0, pixmm)
#        disp[:,2:] = -disp[:,2:]/(alpha*h)
        dfile = change_file_extension(c_files[i+1], 'pts.dat', 'disp.npy')
        np.save(dfile, disp)
        
    return 0

    
############################################################
#
#  Main
#
############################################################


track_displacement (clm_file, c_files, match_range, pixmm, sqmm)

