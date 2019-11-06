"""
Integrates the displacement field to get the height

"""

import numpy as np

from numpy.linalg import lstsq
from scipy.sparse.linalg import lsqr

from matplotlib import pyplot as plt
from matplotlib import cm

from scipy.interpolate import Rbf


#imid = 5
#gradfile = 'images_exemple/output-{:03d}_disp.dat'.format(imid)
#heightfile = 'images_exemple/output-{:03d}_height.npy'.format(imid)
gradfile = 'Exp26-25000-25900/Substack (25000-25900-25)0015_gradh.npy'
heightfile = 'Exp26-25000-25900/Substack (25000-25900-25)0015_height.npy'

dx = .15

############################################################
#
#  Main
#
############################################################

#data_grad = np.loadtxt(gradfile)
data_grad = np.load(gradfile)

xmin = min(data_grad[:,0])
xmax = max(data_grad[:,0])
ymin = min(data_grad[:,1])
ymax = max(data_grad[:,1])



x = np.linspace(xmin, xmax, num=int(np.ceil((xmax-xmin)/dx)))
y = np.linspace(xmin, xmax, num=int(np.ceil((ymax-ymin)/dx)))

dx = x[1]-x[0]
dy = y[1]-y[0]
nx = len(x)
ny = len(y)

print(dx, dy, nx, ny)

ngx = (nx-1)*ny
ngy = nx*(ny-1)


# construction of the matrix to invert
#
# height:
#   - index of [i,j]: i + nx*j
#
# gradient:
#   - index of d/dx from [i,j] to [i+1,j]: i + (nx-1)*j
#   - index of d/dy from [i,j] to [i,j+1]: ngx + i + nx*j
#

def height_index (i, j):
    return i + nx*j

def gradx_index (i, j):
    return i + (nx-1)*j

def grady_index (i, j):
    return ngx + i + nx*j


A = np.zeros((ngx+ngy, nx*ny))

for i in range(nx) :
    for j in range(ny) :
        if i < nx-1 :
            A[gradx_index(i, j), height_index(i,j)] = -1./dx
        if i > 0 :
            A[gradx_index(i-1, j), height_index(i,j)] = 1./dx
        if j < ny-1 :
            A[grady_index(i, j), height_index(i,j)] = -1./dy
        if j > 0 :
            A[grady_index(i, j-1), height_index(i,j)] = 1./dy
            

# construction of the vector to match
B = np.zeros(ngx+ngy)

rbfx = Rbf(data_grad[:,0], data_grad[:,1], data_grad[:,2])
rbfy = Rbf(data_grad[:,0], data_grad[:,1], data_grad[:,3])

for i in range(nx) :
    for j in range(ny) :
        if i < nx-1 :
            B[gradx_index(i, j)] = rbfx((x[i]+x[i+1])/2., y[j])
        if j < ny-1 :
            B[grady_index(i, j)] = rbfy(x[i], (y[j]+y[j+1])/2.)
            

# Comparison of the time taken by
#   - numpy.linalg.lstsq
#   - scipy.sparse.linalg.lsqr 

#import timeit
#start_time = timeit.default_timer()
##H = lstsq(A, B, rcond=None)[0]
#H = lsqr(A, B)[0]
#elapsed = timeit.default_timer() - start_time
#print(elapsed)

#H = lstsq(A, B, rcond=None)[0]
H = lsqr(A, B)[0]

H = H - np.mean(H)
H = np.reshape(H, (ny, nx)).transpose()

X, Y = np.meshgrid(x, y, indexing='ij')

# export
C = np.zeros((nx, ny, 3))
C[:,:,0] = X
C[:,:,1] = Y
C[:,:,2] = H

np.save(heightfile, C)

