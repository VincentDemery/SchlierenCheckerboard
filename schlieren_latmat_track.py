#!/usr/bin/python

from pyx import *
import numpy as np
from os.path import isfile
from scipy.interpolate import interp1d
from skimage.io import imread
from skimage.util import img_as_float


text.set(text.LatexRunner)
#text.preamble(r"\documentclass [11pt] {article}")
text.preamble(r"\usepackage{cmbright}")
#text.preamble(r"\usepackage[T1]{fontenc}")

w = 6
h = 6
dy = .5

ld = .15	#labeldist
paint = graph.axis.painter.regular(labeldist=ld, titledist=.1)

pixmm = .025

img_ext = ".tif"
pts_ext = "_pts.dat"
lmpts_ext = "_ptslm.dat"
curv_ext = "_curv.npy"
curv_ext = "_curv.npy"
curv_center_ext = "_curv_center.npy"
curv_axi_ext = "_curv_axi.npy"


############################################################
#
#	Functions
#
############################################################

def fname_base (i) :
    return "../Exp26-25000-25900/Substack (25000-25900-25){:04d}".format(i)

#-------------------------------------------
#	Scale and ticks
#-------------------------------------------


def scale (x, eps) :
	return np.sign(x)*np.log(1+np.absolute(x)/eps)
	
def mtick (x, lab) :
	return graph.axis.tick.tick(scale(x, eps), label=lab)
	
def strsigntick(s=1) :
    if s==1 :
        return ""
    else :
        return "-"

def strpowtick(n, s=1) :
    if abs(n)<=2 :
        return r"$" + strsigntick(s) + str(10**n) + "$"
    else :
	    return r"$" + strsigntick(s) + "10^{" + str(int(n)) + "}$"
	
def mtickpow (n, lab=True) :
	if lab :
		return mtick(10**n, strpowtick(n))#r"$10^{" + str(int(n)) + "}$")
	else :
		return graph.axis.tick.tick(scale(10**n, eps), ticklevel=1, label="")
	
def mtickpowm (n, lab=True) :
	if lab :
		return mtick(-10**n, strpowtick(n, -1))
	else :
		return graph.axis.tick.tick(scale(-10**n, eps), ticklevel=1, label="")
	
def myticks (xmax) :
	ticks = [mtick(0., r"$0$")]
	nmax = np.floor(np.log10(xmax))
	for n in [nmax-2, nmax-1, nmax] :
		ticks.append(mtickpow(n))
		ticks.append(mtickpowm(n))
	
	return ticks

def data_gradient (file_gradient) :
    graddat = np.loadtxt(file_gradient)

    f_r = interp1d(graddat[:,0], graddat[:,1]) 
    f_g = interp1d(graddat[:,0], graddat[:,2])
    f_b = interp1d(graddat[:,0], graddat[:,3])

    return color.functiongradient_rgb(f_r, f_g, f_b)
    

def data_gradient_reversed (file_gradient) :
    graddat = np.loadtxt(file_gradient)

    f_r = interp1d(1.-graddat[:,0], graddat[:,1]) 
    f_g = interp1d(1.-graddat[:,0], graddat[:,2])
    f_b = interp1d(1.-graddat[:,0], graddat[:,3])

    return color.functiongradient_rgb(f_r, f_g, f_b)
    
def point_from_indices(lmpts, i, j) :
    """Extracts a point with given indices"""
    i0 = set(np.where(lmpts[:,2] == i)[0]) & set(np.where(lmpts[:,3] == j)[0])
    return lmpts[list(i0)[0], :2]
    
    
def plot_for_one_image(i, pixmm, center) :
    #-------------------------------------------
    #  Image and corners
    #-------------------------------------------
    
    img_file = fname_base(i) + img_ext
    img = img_as_float(imread(img_file, as_gray=True))

    pts = np.loadtxt(fname_base(i) + pts_ext)

    y = (range(np.shape(img)[0]) - center[0])*pixmm
    x = (range(np.shape(img)[1]) - center[1])*pixmm

    X, Y = np.meshgrid(x, y)

    data = list(zip(X.flat, Y.flat, img.flat))

    pts[:,:2] = pixmm*(pts[:,:2] - np.tensordot(np.ones(len(pts[:,0])), center, axes=0))

    xmin = min(pts[:,0])
    xmax = max(pts[:,0])
    ymin = min(pts[:,1])
    ymax = max(pts[:,1])

    dx = max(xmax - xmin, ymax - ymin)

    xm = (xmax+xmin)/2.
    ym = (ymax+ymin)/2.

    xmin = xm - .7*dx
    xmax = xm + .7*dx
    ymin = ym - .7*dx
    ymax = ym + .7*dx

    c1 = canvas.canvas()

    g1 = c1.insert(graph.graphxy(height=h, width=w,
                      x=graph.axis.linear(min=xmin, max=xmax, title="$x$"),
                      y=graph.axis.linear(min=ymin, max=ymax, title="$y$")))
                      
    g1.plot(graph.data.points(data, x=1, y=2, color=3),
           [graph.style.density(gradient=color.gradient.Gray, keygraph=None)])


    g1.plot([graph.data.values(x=pts[:,1], y=pts[:,0], title=None)],
	    [graph.style.symbol(symbol=graph.style._circlesymbol, symbolattrs=[deco.filled(), color.rgb.blue], size=.1)])


    #-------------------------------------------
    #  Curvature field
    #-------------------------------------------

    c2 = canvas.canvas()

    curv = np.load(fname_base(i) + curv_ext)

    xmin = np.min(curv[:,:,0])
    xmax = np.max(curv[:,:,0])
    ymin = np.min(curv[:,:,1])
    ymax = np.max(curv[:,:,1])



    g2 = c2.insert(graph.graphxy(height=h, width=w,
                      x=graph.axis.linear(min=xmin, max=xmax, title="$x$"),
                      y=graph.axis.linear(min=ymin, max=ymax, title="$y$")))


    data = list(zip(curv[:,:,0].flat, curv[:,:,1].flat, curv[:,:,2].flat))

    g2.plot(graph.data.points(data, x=1, y=2, color=3),
           [graph.style.density(gradient=color.gradient.Jet,
                                coloraxis=graph.axis.linear(title="curvature [mm$^{-1}$]"))])
                                


    #-------------------------------------------
    #  Axisymmetric curvature
    #-------------------------------------------
    
    exist_axi = isfile(fname_base(i) + curv_center_ext)
    
    if exist_axi :
        curv_center = np.load(fname_base(i) + curv_center_ext)
        
        c3 = canvas.canvas()

        g2.plot([graph.data.values(x=[curv_center[0]], y=[curv_center[1]], title=None)],
	        [graph.style.symbol(symbol=graph.style._circlesymbol,
	                            symbolattrs=[deco.filled()], size=.2)])

        axi_curv = np.load(fname_base(i) + curv_axi_ext)

        g3 = c3.insert(graph.graphxy(height=h, width=w,
                          x=graph.axis.linear(title="$r$"),
                          y=graph.axis.linear(title="curvature")))
                          
        g3.plot(graph.data.points(axi_curv, x=1, y=2), [graph.style.line()])

    c = canvas.canvas()

    c.insert(c1)

    c.insert(c2, [trafo.translate(1.5*w, 0)])
    
    if exist_axi :
        c.insert(c3, [trafo.translate(3.5*w, 0)])
    
    c.text(-w/2, w/2, str(i), [text.halign.right])
    
#    cpic.text(dy, bpic.height()-dy, r"(a)", 
#		[text.halign.left, text.valign.top, color.rgb.white])
    
    return c
############################################################
#
#	Correlations
#
############################################################


#-------------------------------------------
#  Image and corners
#-------------------------------------------

i0 = 8

lmpts = np.loadtxt(fname_base(i0) + lmpts_ext)
center = point_from_indices(lmpts, 0, 0)

all_plots = [plot_for_one_image(i, pixmm, center) for i in range(8, 26)]

c = canvas.canvas()

for i, ci in enumerate(all_plots) :
    c.insert(ci, [trafo.translate(0, -1.5*h*i)])

c.writePDFfile()

