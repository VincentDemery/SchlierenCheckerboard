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
    

    
############################################################
#
#	Correlations
#
############################################################

c = canvas.canvas()

R = 13.8

g = c.insert(graph.graphxy(height=h, width=w,
                          x=graph.axis.linear(title="$r$ [mm]"),
                          y=graph.axis.linear(title="curvature [mm$^{-1}$]")))


data = []
for i in range(8, 26) :
    file_name = fname_base(i) + curv_axi_ext
    if isfile(file_name) :
        data.append(np.load(file_name))
    
g.plot([graph.data.points(data[i], x=1, y=2) for i in range(len(data))], 
       [graph.style.line([style.linestyle.solid, color.gradient.Jet])])

g.plot(graph.data.function("y(x)="+str(1./R)),
       [graph.style.line([style.linestyle.dashed])])
#g.plot([graph.data.points(data[i], x=1, y=2) for i in range(len(data))], 
#       [graph.style.line([style.linestyle.solid, color.gradient.Jet])])
       
#for i in range(8, 26) :
#    file_name = fname_base(i) + curv_axi_ext
#    if isfile(file_name) :
#        axi_curv = np.load(file_name)
#        g.plot(graph.data.points(axi_curv, x=1, y=2), [graph.style.line()])


c.writePDFfile()

