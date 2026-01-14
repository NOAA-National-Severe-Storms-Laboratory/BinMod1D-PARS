# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 08:16:15 2025

@author: edwin.dunnavan
"""
import numpy as np

from .bin_integrals import Pn


def Prod_kernel(xi,xj):
    
    return xi[:,None]*xj[None,:]


def Constant_kernel(xi,xj):
    
    return np.ones_like(xi[:,None]*xj[None,:])


def Golovin_kernel(xi,xj):
    
    return xi[:,None]+xj[None,:]

def hydro_kernel(vtx,vty,Ax,Ay):

    # Hydrodynamic kernel in mm^3/s. Note units eventually cancel out if
    # Nt is in #/L
    Kxy = 0.001*np.abs(vtx[:,None]-vty[None,:])*(np.sqrt(Ax[:,None])+np.sqrt(Ay[None,:]))**2.   

    return Kxy     

def long_kernel(di,dj,vi,vj,Ai,Aj):
    
    # See Long (1974) (JAS)
    # NOTE: Try Simmel's approach. 
    
    di2d,dj2d = np.meshgrid(di,dj,indexing='ij')
    Exy = np.ones_like(dj2d)
    
    # Equation 13 from Simmel et al. (2002)
    Exy[dj2d<0.1] = np.maximum(4.5e4*(dj2d[dj2d<0.1]/20)**2*(1.-3e-4/(di2d[dj2d<0.1]/20)),1e-3)
    
    Kxy = Exy*hydro_kernel(vi,vj,Ai,Aj)
    
    return Kxy

def hall_kernel(di,dj,vi,vj,Ai,Aj):
    
    from scipy.interpolate import griddata
    
    # Ecol values from Table 1 of Hall (1980)
    table = np.flipud(np.array([
    [0.97, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [0.87, 0.96, 0.98, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [0.77, 0.93, 0.97, 0.97, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [0.50, 0.79, 0.91, 0.95, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [0.20, 0.58, 0.75, 0.84, 0.88, 0.90, 0.92, 0.94, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.97, 1.0, 1.02, 1.04, 2.3, 4.0],
    [0.05, 0.43, 0.64, 0.77, 0.84, 0.87, 0.89, 0.90, 0.91, 0.91, 0.91, 0.91, 0.91, 0.92, 0.93, 0.95, 1.0, 1.03, 1.7, 3.0],
    [0.005, 0.40, 0.60, 0.70, 0.78, 0.83, 0.86, 0.88, 0.90, 0.90, 0.90, 0.90, 0.89, 0.88, 0.88, 0.89, 0.92, 1.01, 1.3, 2.3],
    [0.001, 0.07, 0.28, 0.50, 0.62, 0.68, 0.74, 0.78, 0.80, 0.80, 0.80, 0.78, 0.77, 0.76, 0.77, 0.77, 0.78, 0.79, 0.95, 1.4],
    [0.0001, 0.002, 0.02, 0.04, 0.085, 0.17, 0.27, 0.40, 0.50, 0.55, 0.58, 0.59, 0.58, 0.54, 0.51, 0.49, 0.47, 0.45, 0.47, 0.52],
    [0.0001, 0.0001, 0.005, 0.016, 0.022, 0.03, 0.043, 0.052, 0.064, 0.072, 0.079, 0.082, 0.08, 0.076, 0.067, 0.057, 0.048, 0.040, 0.033, 0.027],
    [0.0001, 0.0001, 0.0001, 0.014, 0.017, 0.019, 0.022, 0.027, 0.030, 0.033, 0.035, 0.037, 0.038, 0.038, 0.037, 0.036, 0.035, 0.032, 0.029, 0.027]
]))
    
    dratio_tab = np.arange(0.05,1.05,0.05)
    
    Dcol_tab = 0.001*2.*np.array([10.,20.,30.,40.,50.,60.,70.,100.,150.,200.,300.]) # Dcol in mm
    
    di2d,dj2d = np.meshgrid(di,dj,indexing='ij')
    
    drat2d = np.maximum(np.minimum(di2d/dj2d,dj2d/di2d),0.05)
    
    dmax2d = np.clip(np.maximum(di2d,dj2d),0.02,0.6)
    
    # Interpolate values to table
    X, Y = np.meshgrid(Dcol_tab,dratio_tab,indexing='ij')
    
    Exy = griddata((X.ravel(),Y.ravel()),table.ravel(),(dmax2d,drat2d),method='linear',rescale=True)
      
    Kxy = Exy*hydro_kernel(vi,vj,Ai,Aj)
    
    return Kxy


def long_kernel_PK(xi1,xi2,xj1,xj2,di1,di2,dj1,dj2):
    '''
    Due to the second order nature of the D<50 um kernel (i.e., the x^2 and y^2 part),
    the inversion equation to get the F(x,y) = A + B*x + C*y + D*x*y coefficients
    is numerically unstable for the largest sizes (due to muliplying/dividing by large/small
    numbers). To better estimate the correct parameters, we can simply directly match the 
    corners. This avoids dividing by small numbers.

    Parameters
    ----------
    xi1 : TYPE
        DESCRIPTION.
    xi2 : TYPE
        DESCRIPTION.
    xj1 : TYPE
        DESCRIPTION.
    xj2 : TYPE
        DESCRIPTION.
    di1 : TYPE
        DESCRIPTION.
    di2 : TYPE
        DESCRIPTION.
    dj1 : TYPE
        DESCRIPTION.
    dj2 : TYPE
        DESCRIPTION.

    Returns
    -------
    HK : TYPE
        DESCRIPTION.

    '''

    
    # See Long (1974) (JAS)
    # NOTE: Try Simmel's approach. 
    
    bins = len(xi1)
    
        
    di1_2d,dj1_2d = np.meshgrid(di1,dj1,indexing='ij')
    di2_2d,dj2_2d = np.meshgrid(di2,dj2,indexing='ij')
    
    #xi2d,xj2d = np.meshgrid(xi,xj,indexing='ij')
    
    #Kxy = 0.001*9440*(xi2d**2+xj2d**2)
    #Kxy[dj2d>0.1] = 0.000001*6.33e-3*xj2d[dj2d>0.1]
    
    cs = 0.001*9440
    cl = 0.001*5.78
    
    #cl = 0.001*1.53
    
    PKs = np.zeros((4,bins,bins))
    PKl = np.zeros((4,bins,bins))
    
    # Matching corners
    PKs[0,:,:] = -(xi1*xi2+xj1*xj2)
    PKs[1,:,:] = (xi1+xi2)
    PKs[2,:,:] = (xj1+xj2)
    
    PKs *= cs
    
    # PK[0,:,:] = 0. 
    # PK[1,:,:] = cl 
    # PK[2,:,:] = cl 
    # PK[3,:,:] = 0.
    
    il, jl = np.nonzero(dj2_2d>0.1)
    
    
    PKl[0,:,:] = 0. 
    PKl[1,:,:] = cl
    PKl[2,:,:] = cl

    PK = PKs
    
    PK[:,il,jl] = PKl[:,il,jl]

    
    return PK


# def optimize_long(Kxy,xi1,xi2,xj1,xj2):
    
#     b0 = Pn(2,xi1,xi2)+Pn(2,xj1,xj2)
#     b1 = Pn(3,xi1,xi2)+Pn(1,xi1,xi2)*Pn(2,xj1,xj2)
#     b2 = Pn(3,xj1,xj2)+Pn(1,xj1,xj2)*Pn(2,xi1,xi2)
#     b3 = Pn(3,xi1,xi2)*Pn(1,xj1,xj2)+Pn(1,xi1,xi2)*Pn(3,xj1,xj2)
    
    
    
#     return
    
    