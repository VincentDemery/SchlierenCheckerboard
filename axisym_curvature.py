
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy.interpolate import interp1d
from scipy.optimize import minimize

#imid = 5
#datafile = 'images_exemple/output-{:03d}_height.npy'.format(imid)

datafile = 'Exp26-25000-25900/Substack (25000-25900-25)0015_curv.npy'

datafiles = ['Exp26-25000-25900/Substack (25000-25900-25){:04d}_curv.npy'.format(i) for i in range(8, 26)]

#-------------------------------------------
#  Functions
#-------------------------------------------

def change_file_extension (fname, ext1, ext2) :
    return fname[:-len(ext1)] + ext2


def axi_score(R, Z, n) :
    rmax = .9*max(R)    #/np.sqrt(2.)
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
    

def to_minimize(x, X, Y, Z, n) :
    return axi_score(radialize(X, Y, x[0], x[1]), Z, n)[0]


def find_center(X, Y, Z, xguess, yguess, n) :
    m = minimize(lambda x: to_minimize(x, X, Y, Z, n), [xguess, yguess])
    return m['x'], m['fun']


def axi_function(data_file, n=50, xguess=0., yguess=0.) :
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
    
      
#-------------------------------------------
#  Main
#-------------------------------------------

for datafile in datafiles :
    axi_function_export(datafile)

