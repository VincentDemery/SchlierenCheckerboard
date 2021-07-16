"""
Utilities
"""

import numpy as np

from scipy.interpolate import interp1d
from scipy.optimize import minimize


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


#This could be improved a lot and maybe incorporated in schlieren_checkerboard

def axi_score(R, Z, n, ratio=.7) :
    rmax = ratio*max(R)
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
    return np.sqrt((X-x0)**2+(Y-y0)**2)


def polar_proj_vec(v, x0, y0) :
    """
    Returns the polar projection of a vector field
    
    Parameters
    ----------
    v : array
        Vector field, as a 4 columns array (x, y, vx, vy)
    x0 : float
        x-coordinate of the polar center
    y0 : float
        y-coordinate of the polar center
    
    Returns
    -------
    array
        Array with a line for each point and 3 columns (r, vr, vtheta).
    """
    npts = np.shape(v)[0]
    
    # position relative to the center, unit vector and its perpendicular
    Rr = v[:,:2] - np.tensordot(np.ones(npts), [x0, y0], axes=0)
    Rrn = Rr/np.tensordot(np.sqrt(Rr[:,0]**2 + Rr[:,1]**2), [1,1], axes=0)
        
    RPJ = np.zeros((npts, 3))
    RPJ[:,0] = radialize(v[:,0], v[:,1], x0, y0)
    RPJ[:,1] = v[:,2]*Rrn[:,0] + v[:,3]*Rrn[:,1]
    RPJ[:,2] = - v[:,2]*Rrn[:,1] + v[:,3]*Rrn[:,0]
    
    return RPJ
    
    
    

def to_minimize(x, X, Y, Z, n) :
    return axi_score(radialize(X, Y, x[0], x[1]), Z, n)[0]


def find_center(X, Y, Z, xguess, yguess, n) :
    m = minimize(lambda x: to_minimize(x, X, Y, Z, n), [xguess, yguess])
    return m['x'], m['fun']


def axi_function(X, Y, Z, n=50, **kwargs) :
    x0 = kwargs.get('xguess', np.mean(X))
    y0 = kwargs.get('yguess', np.mean(Y))
    optimize = kwargs.get('optimize', True)

    if optimize :
        result = find_center(X.flatten(), Y.flatten(), Z.flatten(), x0, y0, n)
        x0 = result[0][0]
        y0 = result[0][1]

    s, RZm = axi_score(radialize(X, Y, x0, y0).flatten(), Z.flatten(), n)
    R = radialize(X, Y, x0, y0).flatten()
    
    return [x0,y0], RZm, [R, Z.flatten()]


def fit_gradient_const_curv(grad) :
    """
    Fits a gradient field to a gradient corresponding to a constant curvature
    """
    xi, yi = grad[:,0], grad[:,1]
    ui, vi = grad[:,2], grad[:,3]
    xm, ym = np.mean(xi), np.mean(yi)
    xip, yip = xi-xm, yi-ym
    
    c = np.sum(xip*ui+yip*vi)/np.sum(xip**2+yip**2)
    xc = -np.mean(ui)/c + xm
    yc = -np.mean(vi)/c + ym

    return [c, xc, yc]
