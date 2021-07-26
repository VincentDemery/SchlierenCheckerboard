"""
Copyright (C) ESPCI Paris PSL (2021)

Contributor: Vincent Démery <vincent.demery@espci.psl.eu>

This file is part of schlieren_checkerboard.

schlieren_checkerboard is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 2 as published by
the Free Software Foundation.

schlieren_checkerboard is distributed in the hope that it will be useful, but 
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
details.

You should have received a copy of the GNU General Public License
along with schlieren_checkerboard.  If not, see <http://www.gnu.org/licenses/>.
"""

"""
Main class and methods for the schlieren_checkerboard.
"""

import numpy as np
from copy import copy

from scipy.interpolate import Rbf, interp1d, RBFInterpolator, splev, splrep
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, convex_hull_plot_2d

from skimage.feature import corner_harris, corner_subpix, corner_peaks

from matplotlib import pyplot as plt
from matplotlib.path import Path

from . import utils

class DeformedCheckerboard :
    r"""
    Deformed checkerboard image, identified points, lattice matching, etc.
    
    Parameters
    ----------
    
    image
        Grayscale image (array)
    pixmm: float
        Size of a pixel (mm)
    sqmm: float
        Size of a square (mm)
    h: float
        Distance between the pattern and the interface (mm)
    alpha: float
        Parameter for the optimal indices, :math:`\alpha=1-n/n'` [Moisy et al.
        *Exp Fluids* (2009) 46:1021–1036]. For an air water interface,
        :math:`\alpha\simeq 0.25` (default value).
    
    Attributes
    ----------
    
    center : list
        Center region, circle of radius R centered at (x, y), set by [x, y, R]
    
    corners : array
        Positions of the detected corners.
    
    lm_corners : array
        Array with a line for each corner and 6 columns [x, y, index_x, index_y,
        x_reference, y_reference]. 
        Reference is the actual position of the point before displacement.
        
    disp : array
        Array with a line for each corner and 4 columns [x, y, dx, dy], where
        (x, y) is the position of the corner on the image and (dx, dy) its
        displacement from its original position. The displacement of the central
        corner is set to 0.
        
    grad : array
        Gradient of the height of the interface. Array with a line for each
        corner and 4 columns [x, y, gx, gy], where (x, y) is the position of the
        corner on the image and (gx, gy) the height gradient.
        
    rp_grad : list
        Radial projection of the gradient field. List containing the center, the
        shift, and an array with a line for each corner and 3 columns (r, 
        grad_r, grad_theta).
    
    curv : list
        Curvature field of the interface. List of three arrays: x, y, and the
        mean curvature. The curvature array is a masked array if the curvature
        has been computed with the `mask_hull` option (default).
        
    axicurv
        List of axisymmetry center ([x, y]), axisymmetric curvature (list of two
        arrays, r and curv), and curvature of the initial points (list of two
        arrays, r and curv).
        
    center_curv : list
        Curvature C of the center region, with the position (xc, yc) of the \
        curvature center (x, y), [C, x, y].
        
    """

    def __init__(self, image, pixmm=0.025, sqmm=0.5, h=10., alpha=.25):
        """
        Initializes a deformed checkerboard pattern.
        """
        self.image = image
        self.pixmm = pixmm
        self.sqmm  = sqmm
        self.h     = h
        self.alpha = alpha
        
    
    def detect_corners (self, min_distance=5, threshold_rel=0.02, 
                              window_size=13, verbose=True):
        """
        Detects the corners of the image using the Harris corner detector
        
        Sets the `corners` field to an array with a line for each corner and
        columns for x and y coordinates.
        
        Uses `corner_harris`, `corner_peaks` and `corner_subpix` from
        `skimage.feature`.
        
        If skimage.feature.corner_subpix fails (returns NaN), the initial corner
        is returned.
        
        Parameters
        ----------
        min_distance : int
            min_distance argument for `corner_peaks`.
        threshold_rel : float
            threshold_rel argument for `corner_peaks`.
        window_size : int
            window_size argument for `corner_subpix`.
        verbose : bool
            If `True` (default), displays the number of detected corners.
        
        Returns
        -------
        int
            Number of detected corners.
        """
        c = corner_peaks(corner_harris(self.image), 
                         min_distance=min_distance, threshold_rel=threshold_rel)
                         
        csp = corner_subpix(self.image, c, window_size=window_size)
        
        for i in np.argwhere(np.isnan(csp[:,0])) :
            csp[i] = c[i]
        
        self.corners = self.pixmm*csp
        
        if verbose :
            print("detect_corners: {} corners detected".format(np.shape(self.corners)[0]))
        
        return np.shape(self.corners)[0]
        

    def lattice_matching (self) :
        """
        Matches the corners to a square lattice
        
        Sets field "lm_corners" to an array with a line for each corner and 6\
        columns. Columns: [x, y, x index, y index, x reference, y reference]
        Reference is the actual position of the point before displacement.
        
        The origin of the lattice is set at the corner closest to the center of
        mass of the detected corners, or closest to the center if the center is
        defined. Then the detected corner closest to the origin is used as the 
        second basis point. The first basis vector is obtained from these two 
        points, the second basis vector is deduced by a :math:`\pi/2` rotation.
        
        The reference point of a given point is obtained by multiplying the
        normalized basis vectors by the square size and the index of the point.
        If the image is a pure magnification of the original cherckerboard, the 
        reference coordinates would be a dilatation of the detected coordinates,
        translated to the origin.
        """
        
        if not hasattr(self, 'corners'):
            self.detect_corners()
        
        # Defines the center where to choose the reference point of the
        # lattice
        if hasattr(self, 'center'):
            center = self.center[:2]
        else :
            center = np.mean(self.corners, axis=0)
        
        # Finds the reference point (closest corner to center), and the closest
        # point to get the basis of the lattice
        i0 = np.argmin(np.sum((self.corners - 
                       np.tensordot(np.ones(np.shape(self.corners)[0]), 
                                    center, axes=0))**2,
                       axis=1))
                       
        center = self.corners[i0]
        I = np.argsort(np.sum((self.corners - 
                       np.tensordot(np.ones(np.shape(self.corners)[0]), 
                                    center, axes=0))**2,
                       axis=1))
        
        # Defines the basis of the lattice from two points               
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
        
        Parameters
        ----------
        i : int
            line index of the point
        j : int
            column index of the point
            
        Returns
        -------
        array
            Coordinates of the point.
        """
        
        if not hasattr(self, 'lm_corners'):
            self.lattice_matching()
            
        i0 = set(np.where(self.lm_corners[:,2] == i)[0]) & \
             set(np.where(self.lm_corners[:,3] == j)[0])
             
        return self.lm_corners[list(i0)[0], :2]


    def base(self) :
        """
        Central point and the base for lattice matched points
        
        Returns
        -------
        array
            coordinates of the central point 
        array
            coordinates of the first base vector
        array
            coordinates of the second base vector
        """
        
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
         
        matches = utils.matching_points(ref_im.lm_corners, 
                                        self.corners,
                                        match_range)
        
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
        
        Sets the field `disp` to an array with a line per corner and 4 columns.
        Columns: [x, y, x_displacement, y_displacement]
        
        Uses the reference information in the lm_corner field. Changes \
        the reference so that the displacement of corner 0 is zero.
        """
        if not hasattr(self, 'lm_corners'):
            self.lattice_matching()
            
        xy0 = self.base()[0]
        
        disp = np.copy(self.lm_corners)
        disp[:,2:4] = disp[:,:2]-disp[:,4:]-np.tensordot(np.ones(np.shape(disp)[0]), 
                                    xy0, axes=0)
        self.disp = disp[:,:4]
        
        
    def grad_from_disp(self) :
        """Converts the displacement into height gradient
        
        Sets field grad to the gradient calculated from the displacement 
        using the Schlieren formula [Moisy et al. *Exp Fluids* (2009) 
        46:1021–1036].
        """
        if not hasattr(self, 'disp'):
            self.disp_from_lm_corners()
            
        grad = self.disp
        grad[:,2:] = -grad[:,2:]/(self.alpha*self.h)
        self.grad = grad
        
        
    def grad_from_disp_shift(self) :
        """Same as grad_from_disp, but the base points are shifted by the 
        displacement"""
        if not hasattr(self, 'disp') :
            self.disp_from_lm_corners()
            
        grad = self.disp
        grad[:,:2] = grad[:,:2] + grad[:,2:]
        grad[:,2:] = -grad[:,2:]/(self.alpha*self.h)
        self.grad = grad

        
    def curv_from_grad(self, **kwargs) :
        """
        Interpolates a gradient field and compute its divergence to get
        a curvature field.
        
        Sets the field `curv`.
        
        The step of the interpolation grid is set by the keyword argument **dx**
        (*float*), the derivative is computed with a step **eps** (*float*).
        
        If **mask_hull** (*bool*) is `True` (default), the interpolation is
        restricted to the convex hull of the detected corners; a masked array
        is used for the curvature.
        
        Parameters
        ----------
        **kwargs
            Arbitrary keyword arguments.
        
        """
        if not hasattr(self, 'grad') :
            self.grad_from_disp()
            
        dx  = kwargs.get('dx',  self.sqmm/4.)
        eps = kwargs.get('eps', self.sqmm/4.)
        mask_hull = kwargs.get('mask_hull', True)
            
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
        #rbfx = RBFInterpolator(self.grad[:,:2], self.grad[:,2])
        rbfy = Rbf(self.grad[:,0], self.grad[:,1], self.grad[:,3])

        X, Y = np.meshgrid(x, y, indexing='ij')

        curvx = (rbfx(X+eps, Y) - rbfx(X-eps, Y))/(2.*eps)
        curvy = (rbfy(X, Y+eps) - rbfy(X, Y-eps))/(2.*eps)
        curvm = (curvx+curvy)/2.
        
        if mask_hull :
            hull = ConvexHull(self.grad[:,:2])
            hull_path = Path(self.grad[hull.vertices,:2])
            mask = hull_path.contains_points(np.reshape(np.stack([X, Y],-1), (-1,2)))
            mask = np.logical_not(np.reshape(mask, np.shape(X)))
            curvm = np.ma.array(curvm, mask=mask)

        self.curv = [X, Y, curvm]
        
    
    def polar_proj_grad(self, xc, yc, shift=True) :
        """
        Computes the polar projection of the shifted gradient field.
        
        Sets the field `rp_grad`, which contains the center, the shift, and
        an array with a line for each corner and 3 columns 
        (r, grad_r, grad_theta).
        
        Parameters
        ----------
        
        xc : float
            x position of the axisymmetry center.
        yc : float
            y position of the axisymmetry center.
        shift : bool
            If True (default), adds a constant vector to the gradient field
            so that its value at (xc, yc) is zero.
        """
        
        if not hasattr(self, 'grad') :
            self.grad_from_disp()
        
        new_grad = np.copy(self.grad)
        
        rc = np.array([xc, yc])
        shiftv = np.zeros(2)
        
        if shift:
            rbfx = RBFInterpolator(self.grad[:,:2], self.grad[:,2])
            rbfy = RBFInterpolator(self.grad[:,:2], self.grad[:,3])
            
            shiftv[0] = rbfx([[xc, yc]])[0]
            shiftv[1] = rbfy([[xc, yc]])[0]
            
            npts = np.shape(self.grad)[0]
            new_grad[:,2:] -= np.tensordot(np.ones(npts), shiftv,
                                           axes=0)
        
        rp_grad = utils.polar_proj_vec(new_grad, xc, yc)
        
        self.rp_grad = [rc, shiftv, rp_grad]
        
        
    def axisym_curv(self, xguess=None, yguess=None, n=50,
                          optimize=True) :
        """
        Computes the axisymmetric interpolation of the curvature field.
        
        Sets the field axicurv.
        
        Parameters
        ----------
        
        xguess : float
            Guess for the x position of the axisymmetry center.
            If None (default) sets to the center of the curvature field.
        yguess : float
            Guess for the y position of the axisymmetry center.
        n : int
            Number of points for the axisymmetric function
        optimize : bool
            If True (default), minimizes the difference between the 2d curvature
            field and the axisymmetric function.
        """
        if not hasattr(self, 'curv'):
            self.curv_from_grad()
        
        X = self.curv[0]
        Y = self.curv[1]
        Z = self.curv[2]
        
        self.axicurv = utils.axi_function(X, Y, Z,
                                          xguess=xguess, yguess=yguess, n=n,
                                          optimize=optimize)
        
        if optimize:
            print('axisymmetry center:', self.axicurv[0])
    
    
    def compute_center_curv(self) :
        """
        Computes the average curvature in the center region
        
        Finds the least-square best approximation to the gradient in the\
        center region, defined by the field `center`.
        
        Sets the field `center_curv`.
        """
        if not hasattr(self, 'center'):
            print('center is not defined')
            return
        
        if not hasattr(self, 'grad') :
            self.grad_from_disp()
        
        xc = self.center[0]
        yc = self.center[1]
        rc = self.center[2]
        
        I = ((self.grad[:,0]-xc)**2. + (self.grad[:,1]-yc)**2. < rc**2.)
        gradr = self.grad[I,:]
        
        self.center_curv = utils.fit_gradient_const_curv(gradr)
                 
        return
    
    def compute_center_curv_errorbars(self, ncopies=100, stdev=None) :
        """
        Computes the average curvature in the center region for many copies
        with randomized points and return mean and standard deviation.
        
        Parameters
        ----------
        
        ncopies : int
            Number of copies to use.
        stdev : float
            Standard deviation used to randomize the corners. If `None`, set to
            the pixel size `pixmm`.
            
            
        Returns
        -------
        float   
            Mean of the curvature in the center.
        float
            Standard deviation.
        """
        if not hasattr(self, 'center'):
            print('center is not defined')
            return
        
        if not hasattr(self, 'grad') :
            self.grad_from_disp()
            
        if stdev == None:
            stdev = self.pixmm
        
        xc = self.center[0]
        yc = self.center[1]
        rc = self.center[2]
        
        I = ((self.grad[:,0]-xc)**2. + (self.grad[:,1]-yc)**2. < rc**2.)
        gradr = self.grad[I,:]
        
        self.center_curv = utils.fit_gradient_const_curv(gradr)
        
        npts = np.shape(gradr)[0]
        ccurvs = []
        for i in range(ncopies):
            gradr_err = np.copy(gradr)
            err = stdev*np.random.standard_normal((npts, 2))
            gradr_err[:,:2] += err
            gradr_err[:,2:] += -err/(self.alpha*self.h)
            ccurvs.append(utils.fit_gradient_const_curv(gradr_err)[0])
            
        ccurvs = np.array(ccurvs)
        return np.mean(ccurvs), np.std(ccurvs, ddof=1)
    
    
    def plot_corners (self, show_center=True) :
        """
        Shows the image with the detected corners on it.
        
        Parameters
        ----------
        show_center : bool
            If `True` (default) and the center is defined, shows it with a
            circle.
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
        pcm = ax.pcolormesh(XX, YY, self.image.transpose(), cmap='Greys', shading='nearest')
        ax.scatter(self.corners[:, 0], self.corners[:, 1])
        
        if show_center and hasattr(self, 'center') :
            x = self.center[0]
            y = self.center[1]
            R = self.center[2]
            circle = plt.Circle((x, y), R, color='r', alpha=.25)
            ax.add_artist(circle)
            
        
        plt.show()
        
    
    def plot_lattice_matching (self, show_center=True) :
        """
        Shows the image with the matched lattice
        
        Shows the image, the matched lattice, the corners (the base corners are
        highlighted), and the circle showing the center (if defined).
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
        pcm = ax.pcolormesh(XX, YY, self.image.transpose(), cmap='Greys', shading='nearest')
        
        nl = 4
        xy0, v1, v2 = self.base()
        imin = int(min(self.lm_corners[:,2]))
        imax = int(max(self.lm_corners[:,2]))
        jmin = int(min(self.lm_corners[:,3]))
        jmax = int(max(self.lm_corners[:,3]))
        for j in range(jmin, jmax+1):
            ax.plot([xy0[0] + imin*v1[0] + j*v2[0], xy0[0] + imax*v1[0] + j*v2[0]],
                    [xy0[1] + imin*v1[1] + j*v2[1], xy0[1] + imax*v1[1] + j*v2[1]],
                    'g-', zorder=1)
        
        for i in range(imin, imax+1):
            ax.plot([xy0[0] + i*v1[0] + jmin*v2[0], xy0[0] + i*v1[0] + jmax*v2[0]],
                    [xy0[1] + i*v1[1] + jmin*v2[1], xy0[1] + i*v1[1] + jmax*v2[1]],
                    'g-', zorder=1)
                    
        if show_center and hasattr(self, 'center') :
            x = self.center[0]
            y = self.center[1]
            R = self.center[2]
            circle = plt.Circle((x, y), R, color='r', alpha=.25)
            ax.add_artist(circle)


        ax.scatter(self.corners[:, 0], self.corners[:, 1], zorder=2)
        ax.scatter([xy0[0], xy0[0]+v1[0]], [xy0[1], xy0[1]+v1[1]], zorder=3)
                
        plt.show()
        
        
    def plot_disp(self) :
        """
        Plots the displacement field.
        """
        
        # Should be improved to plot also the image. zorger may be useful.
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
  
        pcm = ax.pcolormesh(self.curv[0], self.curv[1], 
                            self.curv[2], cmap='RdBu_r', 
                            shading='nearest')

                            
        fig.colorbar(pcm, ax=ax, orientation='vertical')
        
        plt.show()
        
    
    def plot_polar_proj_grad (self, spline_interp=False) :
        """
        Plots the height gradient field, together with its polar 
        projection.
        
        Parameters
        ----------
        
        spline_interp : bool
            If True, shows a spline interpolation of the radial component. \ 
            Default is false.
        """
        
        if not hasattr(self, 'rp_grad') :
            raise AttributeError('no attribute rp_grad')
            
        
        rc, shift, rp_grad = self.rp_grad
        
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        
        new_grad = np.copy(self.grad)
        npts = np.shape(self.grad)[0]
        new_grad[:,2:] -= np.tensordot(np.ones(npts), shift, axes=0)
        
        ax[0].set_xlabel('$x$ [mm]')
        ax[0].set_ylabel('$y$ [mm]')
        
        ax[0].quiver(new_grad[:, 0], new_grad[:, 1], 
                     new_grad[:, 2], new_grad[:, 3], 
                     angles='xy', scale_units='xy', scale=1)
                  
        ax[0].plot([rc[0]], [rc[1]], 'or')

        
        ax[1].set_xlabel('$r$ [mm]')
        ax[1].set_ylabel('gradient')
        
        ax[1].plot(rp_grad[:,0], rp_grad[:,1], 'ok')
        ax[1].plot(rp_grad[:,0], rp_grad[:,2], 'ok', markerfacecolor='none')
        
        if spline_interp :
            rmin = min(rp_grad[:,0])
            rmax = max(rp_grad[:,0])
            Rsplt = np.linspace(rmin, rmax, 5)[1:-1]
            isort = np.argsort(rp_grad[:,0])
            spl = splrep(rp_grad[isort,0], rp_grad[isort,1], 
                         xb=rmin, xe=rmax, t=Rsplt)
            Rspl = np.linspace(rmin, rmax, 100)
            Gspl = splev(Rspl, spl)
            ax[1].plot(Rspl, Gspl, 'r-')
        
        plt.show()
            
        
    def plot_axisym_curv (self, plot_circle=True) :
        """
        Plots the axisymmetric approximation to the curvature
        field, together with the curvature field, and indicates the
        axisymmetry center.
        
        Parameters
        ----------
        
        plot_circle : bool
            If True (default), plots a circle at the minimum of the \
            axisymmetric curvature on the curvature field.
        """
        
        if not hasattr(self, 'curv'):
            self.curv_from_grad()

        if not hasattr(self, 'axicurv'):
            self.axisym_curv()


        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        
        ax[0].set_xlabel('$x$ [mm]')
        ax[0].set_ylabel('$y$ [mm]')
        
        pcm = ax[0].pcolormesh(self.curv[0], self.curv[1], 
                            self.curv[2], cmap='RdBu_r', 
                            shading='nearest')
        
        fig.colorbar(pcm, ax=ax[0], orientation='vertical',
                     label='mean curvature [mm$^{-1}$]')
        pos = self.axicurv[0]
        ax[0].plot([pos[0]], [pos[1]], 'ok')
        
        if plot_circle :
            i  = np.argmin(self.axicurv[1][:,1])
            rc = self.axicurv[1][i,0]
            
            theta_tab = np.linspace(0., 2.*np.pi)
            ax[0].plot(pos[0]+rc*np.cos(theta_tab), 
                       pos[1]+rc*np.sin(theta_tab), '--k')

        
        ax[1].set_xlabel('$r$ [mm]')
        ax[1].set_ylabel('mean curvature [mm$^{-1}$]')
        
        ax[1].plot(self.axicurv[2][0], self.axicurv[2][1], 'ok')
        ax[1].plot(self.axicurv[1][:,0], self.axicurv[1][:,1], 'r', lw=3)
        
        plt.show()
