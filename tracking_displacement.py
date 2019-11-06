"""
From a list of corners on a list of images, tracks the corners to get
the displacements

"""

import numpy as np

immax = 9
cfiles = ['images_exemple/output-{:03d}_pts.dat'.format(i) for i in range(1, immax+1)]

match_range = 5

############################################################
#
#  Functions
#
############################################################

def change_file_extension (fname, ext1, ext2) :
    return fname[:-len(ext1)] + ext2


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
   

def track_displacement (cfiles, match_range) :
    """Track points through a list of remarkable points and gets
       the displacement"""

    corners = [np.loadtxt(cfile) for cfile in cfiles]
    
    mpts = []
    for i, c in enumerate(corners[:-1]) :
        mpts.append(matching_points(corners[i], corners[i+1], match_range))
        
    m = mpts[0]
    disp = matching_coords_displ(corners[0], corners[1], m)
    dfile = change_file_extension(cfiles[1], 'pts.dat', 'disp.dat')
    np.savetxt(dfile, disp)

    for i in range(1, immax-1) :
        m = compose_mpts(m, mpts[i])
        disp = matching_coords_displ(corners[0], corners[i+1], m)
        dfile = change_file_extension(cfiles[i+1], 'pts.dat', 'disp.dat')
        np.savetxt(dfile, disp)
        
    return 0
    
############################################################
#
#  Main
#
############################################################

track_displacement (cfiles, match_range)

