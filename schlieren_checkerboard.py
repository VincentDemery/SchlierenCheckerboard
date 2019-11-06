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

from skimage.io import imread
from skimage.util import img_as_float
from skimage.feature import corner_harris, corner_subpix, corner_peaks

from matplotlib import pyplot as plt

import utils

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


class DeformedCheckerboard :
    """
    Deformed checkerboard image, identified points, lattice matching, etc.
    """

    def __init__(self, image_file, pixmm=0.025, sqmm=0.5, h=10., alpha=.24):
        self.image = img_as_float(imread(image_file, as_gray=True))
        self.pixmm = pixmm
        self.sqmm  = sqmm
        self.h     = h
        self.alpha = alpha
        
    
    def detect_corners (self, min_distance=5, window_size=13, verbose=True):
        """
        Detects the corners of the image using the Harris corner detector
        """
        c = corner_peaks(corner_harris(self.image), min_distance=min_distance)
        csp = corner_subpix(self.image, c, window_size=window_size)
        self.corners = self.pixmm*csp
        
        if verbose :
            print("detect_corners: {} corners detected".format(np.shape(self.corners)[0]))
        
        return np.shape(self.corners)[0]
    
    
    def lattice_matching (self) :
        """Matches the corners to a square lattice"""
        
        if not hasattr(self, 'corners'):
            self.detect_corners()
            
        center = np.mean(self.corners, axis=0)
        
        I = np.argsort(np.sum((self.corners - 
                       np.tensordot(np.ones(np.shape(self.corners)[0]), 
                                    center, axes=0))**2,
                       axis=1))
                       
        center = self.corners[I[0]]
        v1 = self.corners[I[1]] - center
        v2 = np.array([-v1[1], v1[0]])
        
        v1n = v1/np.sqrt(sum(v1**2))
        v2n = v2/np.sqrt(sum(v2**2))
        
        # This could be done without a loop
        cor_lm = []
        for c in self.corners :
            nx = np.round(sum((c - center)*v1)/sum(v1**2))
            ny = np.round(sum((c - center)*v2)/sum(v2**2))
            cor_lm.append([c[0], c[1], nx, ny, 
                           self.sqmm*(nx*v1n[0]+ny*v2n[0]), 
                           self.sqmm*(nx*v1n[1]+ny*v2n[1])])
        
        self.lm_corners = np.array(cor_lm)
    
    
    def point_from_indices(self, i, j) :
        """
        Extracts a point with given indices in the list of lattice matched
        corners
        """
        
        if not hasattr(self, 'lm_corners'):
            self.lattice_matching()
            
        i0 = set(np.where(self.lm_corners[:,2] == i)[0]) & \
             set(np.where(self.lm_corners[:,3] == j)[0])
             
        return self.lm_corners[list(i0)[0], :2]


    def base(self) :
        """Returns the central point and the base for lattice matched points"""
        
        xy0 = self.point_from_indices(0, 0)
        xy1 = self.point_from_indices(1, 0)
        v1 = xy1 - xy0
        v2 = np.array([-v1[1], v1[0]])
        
        return xy0, v1, v2
        
    
    def lattice_matching_tracking (self, ref_im, **kwargs) :
        """
        Tracks the points between the reference image and self and transfer the
        lattice matching of the reference image to self.
        """
        if not hasattr(self, 'corners'):
            self.detect_corners()
            
        if not hasattr(ref_im, 'lm_corners'):
            ref_im.lattice_matching()
            
        match_range = kwargs.get('match_range', self.sqmm/5.)
        verbose = kwargs.get('verbose', True)
         
        matches = utils.matching_points(ref_im.lm_corners, self.corners, match_range)
        
        if verbose :
            print('lattice_matching_tracking: {} matches found'.format(len(matches)))
        
        cor_lm = []
        for m in matches :
            cor_lm.append([self.corners[m[1],0], self.corners[m[1],1]] +
                           list(ref_im.lm_corners[m[0],2:]))
        
        self.lm_corners = np.array(cor_lm)


    def disp_from_lm_corners (self) :
        """
        Computes the displacement field.
        """
        if not hasattr(self, 'lm_corners'):
            self.lattice_matching()
            
        xy0 = self.base()[0]
        
        disp = self.lm_corners
        disp[:,2:4] = disp[:,:2]-disp[:,4:]-np.tensordot(np.ones(np.shape(disp)[0]), 
                                    xy0, axes=0)
        self.disp = disp[:,:4]
        
        
    def grad_from_disp(self) :
        """"Converts the displacement into height gradient"""
        if not hasattr(self, 'disp'):
            self.disp_from_lm_corners()
            
        grad = self.disp
        grad[:,2:] = -grad[:,2:]/(self.alpha*self.h)
        self.grad = grad
        
        
    def grad_from_disp_shift(self) :
        """"Same as grad_from_disp, but the base points are shifted by the 
        displacement"""
        if not hasattr(self, 'disp') :
            self.disp_from_lm_corners()
            
        grad = self.disp
        grad[:,:2] = grad[:,:2] + grad[:,2:]
        grad[:,2:] = -grad[:,2:]/(self.alpha*self.h)
        self.grad = grad

        
    def curv_from_grad(self, **kwargs) :
        """Interpolates a gradient field and compute its divergence to get
        a curvature field"""
        if not hasattr(self, 'grad') :
            self.grad_from_disp()
            
        dx  = kwargs.get('dx',  self.sqmm/4.)
        eps = kwargs.get('eps', self.sqmm/4.)
            
        xmin = min(self.grad[:,0])
        xmax = max(self.grad[:,0])
        ymin = min(self.grad[:,1])
        ymax = max(self.grad[:,1])

        x = np.linspace(xmin, xmax, num=int(np.ceil((xmax-xmin)/dx)))
        y = np.linspace(ymin, ymax, num=int(np.ceil((ymax-ymin)/dx)))

        dx = x[1]-x[0]
        dy = y[1]-y[0]
        nx = len(x)
        ny = len(y)

        rbfx = Rbf(self.grad[:,0], self.grad[:,1], self.grad[:,2])
        rbfy = Rbf(self.grad[:,0], self.grad[:,1], self.grad[:,3])

        X, Y = np.meshgrid(x, y, indexing='ij')

        C = np.zeros((nx, ny, 3))
        C[:,:,0] = X
        C[:,:,1] = Y

        Curvx = (rbfx(X+eps, Y) - rbfx(X-eps, Y))/(2.*eps)
        Curvy = (rbfy(X, Y+eps) - rbfy(X, Y-eps))/(2.*eps)
        C[:,:,2] = (Curvx+Curvy)/2.

        self.curv = C

        
    def axisym_curv(self, n=50) :
        """
        Axisymmetric interpolation of the curvature
        """
        if not hasattr(self, 'curv'):
            self.curv_from_grad()
        
        X = self.curv[:,:,0]
        Y = self.curv[:,:,1]
        Z = self.curv[:,:,2]
        
        center, axifun, axidata = utils.axi_function(X, Y, Z)
        
        self.axicurv = axifun


    def print_corners (self) :
        print(self.corners)
    
    
    def print_lm_corners (self) :
        print(self.lm_corners)
        
    
    def plot_corners (self) :
        """
        Shows the image with the corners on it
        """
        
        if not hasattr(self, 'corners'):
            self.detect_corners()
            
        xmin, xmax = min(self.corners[:,0]), max(self.corners[:,0])
        ymin, ymax = min(self.corners[:,1]), max(self.corners[:,1])
        dx = max(xmax - xmin, ymax - ymin)
        xm = (xmax+xmin)/2.
        ym = (ymax+ymin)/2.
        xmin = xm - .7*dx
        xmax = xm + .7*dx
        ymin = ym - .7*dx
        ymax = ym + .7*dx
        
        S = np.shape(self.image)
        X = self.pixmm*np.arange(S[0])
        Y = self.pixmm*np.arange(S[1])
        XX, YY = np.meshgrid(X, Y)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.axis((xmin, xmax, ymin, ymax))
        pcm = ax.pcolormesh(XX, YY, self.image.transpose(), cmap='Greys')
        ax.scatter(self.corners[:, 0], self.corners[:, 1])
        
        plt.show()
        
    
    def plot_lattice_matching (self) :
        """
        Shows the image with the matched lattice
        """
        
        if not hasattr(self, 'lm_corners'):
            self.lattice_matching()
            
        xmin, xmax = min(self.corners[:,0]), max(self.corners[:,0])
        ymin, ymax = min(self.corners[:,1]), max(self.corners[:,1])
        dx = max(xmax - xmin, ymax - ymin)
        xm = (xmax+xmin)/2.
        ym = (ymax+ymin)/2.
        xmin = xm - .7*dx
        xmax = xm + .7*dx
        ymin = ym - .7*dx
        ymax = ym + .7*dx
        
        S = np.shape(self.image)
        X = self.pixmm*np.arange(S[0])
        Y = self.pixmm*np.arange(S[1])
        XX, YY = np.meshgrid(X, Y)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.axis((xmin, xmax, ymin, ymax))
        pcm = ax.pcolormesh(XX, YY, self.image.transpose(), cmap='Greys')
        
        nl = 7
        xy0, v1, v2 = self.base()
        for i in range(-nl, nl+1):
            ax.plot([xy0[0] - nl*v1[0] + i*v2[0], xy0[0] + nl*v1[0] + i*v2[0]],
                    [xy0[1] - nl*v1[1] + i*v2[1], xy0[1] + nl*v1[1] + i*v2[1]],
                    'r-')
                    
            ax.plot([xy0[0] + i*v1[0] - nl*v2[0], xy0[0] + i*v1[0] + nl*v2[0]],
                    [xy0[1] + i*v1[1] - nl*v2[1], xy0[1] + i*v1[1] + nl*v2[1]],
                    'r-')
            

        ax.scatter(self.corners[:, 0], self.corners[:, 1])
                
        plt.show()
        
        
    def plot_disp(self) :
        """
        Plots the displacement field.
        """
        if not hasattr(self, 'disp'):
            self.disp_from_lm_corners()
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        ax.quiver(self.disp[:, 0], self.disp[:, 1], 
                  self.disp[:, 2], self.disp[:, 3], 
                  angles='xy', scale_units='xy', scale=1)
        
        plt.show()

        
    def plot_curvature (self) :
        """
        Plots the curvature field.
        """
        if not hasattr(self, 'curv'):
            self.curv_from_grad()

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        pcm = ax.pcolormesh(self.curv[:,:,0], self.curv[:,:,1], 
                            self.curv[:,:,2], cmap='RdBu_r')
        fig.colorbar(pcm, ax=ax, orientation='vertical')

        plt.show()      

