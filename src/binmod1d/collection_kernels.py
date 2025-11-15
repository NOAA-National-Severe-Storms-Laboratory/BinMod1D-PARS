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
    
    