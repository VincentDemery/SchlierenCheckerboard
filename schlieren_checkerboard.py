"""
Schlieren method for a checkerboard pattern
===========================================

Set of functions to use the Schlieren method to find the shape of an interface
from the deformations of a checkerboard pattern.

"""

# This file includes the old files:
#   corner_detection_lattice_matching.py
#   latmat_pts_tracking_disp.py
#   disp_curvature.py
#   axisym_curvature.py

import numpy as np

from scipy.interpolate import Rbf, interp1d
from scipy.optimize import minimize

from skimage.util import img_as_float
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.io import imread


#-------------------------------------------
#  Parameters (to set somehow)
#-------------------------------------------

#match_range = 5.

#pixmm = 0.025   # Size of a pixel (mm)
#sqmm = 0.5      # Size of a square (mm)

#dx = .1     # step for the grid where to compute the curvature
#eps = .05   # step to compute the curvature

#h = 10.
#alpha = .24

############################################################
#
#  Functions
#
############################################################

def replace_file_extension (fname, ext) :
    return fname[:-1-fname[::-1].find('.')] + ext


def change_file_extension (fname, ext1, ext2) :
    return fname[:-len(ext1)] + ext2


def coords (img) :
    """Finds the corners on an image using Harris corner detector"""
    c = corner_peaks(corner_harris(img), min_distance=8)
    c0 = corner_subpix(img, c, window_size=13)
    return c0


def closest_point_to (points_list, point) :
    """Finds the closest point to a given point in a list"""
    i = np.argmin((points_list[:,0]-point[0])**2
                  +(points_list[:,1]-point[1])**2)
    return i

    
def transfer_point (i, basecor, newcor) :
    """Transfers a point (given by its index) between two lists"""
    newcor.append(basecor[i])
    basecor = np.delete(basecor, i, 0)
    return basecor, newcor

    
def lattice_matching (cor) :
    """Matches a set of points to a square lattice"""
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
    """Finds the displacement for lattice matched points
    
    Uses the square size (pixmm) and the pixel size (sqmm)
    """
    
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
    """Finds the matching points between two lists of points"""
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


def matching_coords_displ (c1, c2, m, d01, pixmm) :
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
    disp = matching_coords_displ(corners[0], corners[1], m, disp0, pixmm)
    dfile = change_file_extension(c_files[1], 'pts.dat', 'disp.npy')
    np.save(dfile, disp)

    for i in range(1, len(c_files)-1) :
        m = compose_mpts(m, mpts[i])
        disp = matching_coords_displ(corners[0], corners[i+1], m, disp0, pixmm)
        dfile = change_file_extension(c_files[i+1], 'pts.dat', 'disp.npy')
        np.save(dfile, disp)
        
    return 0


#-------------------------------------------
#  Curvature from the displacement
#-------------------------------------------


def grad_from_disp(disp, h, alpha) :
    """"Converts the displacement into height gradient"""
    disp[:,2:] = -disp[:,2:]/(alpha*h)
    return disp
    
    
def grad_from_disp_shift(disp, h, alpha) :
    """"Same as grad_from_disp, but the base points are shifted by the 
    displacement"""
    disp[:,:2] = disp[:,:2] + disp[:,2:]
    disp[:,2:] = -disp[:,2:]/(alpha*h)
    return disp

    
def curv_from_grad(grad, dx, eps) :
    """Interpolates a gradient field and compute its divergence to get
    a curvature field"""
    
    xmin = min(grad[:,0])
    xmax = max(grad[:,0])
    ymin = min(grad[:,1])
    ymax = max(grad[:,1])

    x = np.linspace(xmin, xmax, num=int(np.ceil((xmax-xmin)/dx)))
    y = np.linspace(xmin, xmax, num=int(np.ceil((ymax-ymin)/dx)))

    dx = x[1]-x[0]
    dy = y[1]-y[0]
    nx = len(x)
    ny = len(y)

    rbfx = Rbf(grad[:,0], grad[:,1], grad[:,2])
    rbfy = Rbf(grad[:,0], grad[:,1], grad[:,3])

    X, Y = np.meshgrid(x, y, indexing='ij')

    C = np.zeros((nx, ny, 3))
    C[:,:,0] = X
    C[:,:,1] = Y

    Curvx = (rbfx(X+eps, Y) - rbfx(X-eps, Y))/(2.*eps)
    Curvy = (rbfy(X, Y+eps) - rbfy(X, Y-eps))/(2.*eps)
    C[:,:,2] = (Curvx+Curvy)/2.

    return C


def curv_from_disp_export (disp_file, h, alpha, dx, eps) :
    disp = np.load(disp_file)
    grad = grad_from_disp(disp, h, alpha)
    C = curv_from_grad(grad, dx, eps)
    curv_file = change_file_extension(disp_file, 'disp.npy', 'curv.npy')
    np.save(curv_file, C)
    return 0


#-------------------------------------------
#  Axisymmetrization of the curvature field
#-------------------------------------------


def axi_score(R, Z, n, R_range=.9) :
    rmax = R_range*max(R)
    Rbins = np.linspace(0, rmax, n)
    RZm = []
    for i in range(n-1) :
        I = np.where((R-Rbins[i])*(R-Rbins[i+1])<0)[0]
        if len(I) > 0 :
            RZm.append([(Rbins[i]+Rbins[i+1])/2., np.mean(Z[I])])
    
    RZm = np.array(RZm)

    zfun = interp1d(RZm[:,0], RZm[:,1])
    
    I = np.where((R - RZm[0,0])*(R-RZm[-1,0]) < 0.)[0]
    Rcomp = R[I]
    Zcomp = Z[I]
    
    score = np.mean((Zcomp-zfun(Rcomp))**2)
    
    return score, RZm
    

def radialize(X, Y, x0, y0) :
    """Distances of points to a reference point"""
    return np.sqrt((X-x0)**2+(Y-y0)**2)
    

def to_minimize(x, X, Y, Z, n) :
    return axi_score(radialize(X, Y, x[0], x[1]), Z, n)[0]


def find_center(X, Y, Z, xguess, yguess, n) :
    m = minimize(lambda x: to_minimize(x, X, Y, Z, n), [xguess, yguess])
    return m['x'], m['fun']


def axi_function(data_file, n=50, xguess=0., yguess=0.) :
    """Fits a axisymmetric function to a 2d function
    
    Returns:
      - center
      - interpolated axisymmetric function
      - initial values as a function of the distance to the center
    """
    data = np.load(datafile)

    X = data[:,:,0]
    Y = data[:,:,1]
    Z = data[:,:,2]

    x0 = xguess
    y0 = yguess

    n = 50

    result = find_center(X.flatten(), Y.flatten(), Z.flatten(), x0, y0, n)
    x0 = result[0][0]
    y0 = result[0][1]

    s, RZm = axi_score(radialize(X, Y, x0, y0).flatten(), Z.flatten(), n)
    R = radialize(X, Y, x0, y0).flatten()
    
    return result[0], RZm, np.array(list(zip(R, Z.flatten())))


def axi_function_export(data_file, n=50) :
    center, A, B = axi_function(data_file)
    print(data_file + ': ' + str(center))

    center_file = change_file_extension(data_file, '_curv.npy', '_curv_center.npy')
    np.save(center_file, center)

    axi_file = change_file_extension(data_file, '_curv.npy', '_curv_axi.npy')
    np.save(axi_file, A)

    radial_file = change_file_extension(data_file, '_curv.npy', '_curv_rad.npy')
    np.save(radial_file, B)
    
