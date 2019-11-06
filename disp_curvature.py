"""
Interpolates a height gradient field and deduces the curvature

"""

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm

from scipy.interpolate import Rbf


#imid = 5
#grad_file = 'images_exemple/output-{:03d}_disp.dat'.format(imid)

#grad_file = 'Exp26-25000-25900/Substack (25000-25900-25)0015_gradh.npy'

disp_files = ['Exp26-25000-25900/Substack (25000-25900-25){:04d}_disp.npy'.format(i) for i in range(8, 26)]

dx = .1     # step for the grid where to compute the curvature
eps = .05   # step to compute the curvature

h = 10.
alpha = .24


############################################################
#
#  Functions
#
############################################################

def change_file_extension (fname, ext1, ext2) :
    return fname[:-len(ext1)] + ext2
    
    
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
    
############################################################
#
#  Main
#
############################################################

#curv_from_disp_export(grad_file, dx, eps)
for disp_file in disp_files :
    curv_from_disp_export(disp_file, h, alpha, dx, eps)
