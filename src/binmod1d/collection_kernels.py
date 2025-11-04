# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 08:16:15 2025

@author: edwin.dunnavan
"""
import numpy as np


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