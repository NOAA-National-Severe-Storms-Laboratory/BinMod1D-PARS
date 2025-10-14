# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 10:50:41 2025

@author: edwin.dunnavan
"""
import numpy as np
from cmweather import cm as cmp

def get_cmap_vars(varname):
    
    if varname == 'Z':
        cmap = cmp.NWSRef
        levels = np.arange(-10.,80,1.)
        levels_ticks = np.arange(-10,80,5)
        clabel = r'Reflectivity [dBZ]'
        slabel = r'Z [dBZ]'
        labelpad = 45
        fontsize=32

    if varname == 'ZDR':
        
        cmap = cmp.NWSRef
        levels = [-1.,0.,0.2,0.4,0.6,0.8,1.0,1.5,2.0,2.5,3.0,4.,6.]
        levels_ticks = [-1.,0.,0.2,0.4,0.6,0.8,1.0,1.5,2.0,2.5,3.0,4.,6.]
        clabel =r'$\mathrm{Z}_{\mathrm{DR}}$ [dB]'
        slabel =r'$\mathrm{Z}_{\mathrm{DR}}$ [dB]'
        labelpad = 45
        fontsize=32

    if varname == 'KDP':
        cmap = cmp.Theodore16
        levels =[-0.1,0.,0.025,0.05,0.1,0.15,0.2,0.3,0.4,0.6,0.8,1.,1.25,1.5,1.75,2.0,3.0] # X band?
        levels_ticks = levels.copy()
        clabel = r'$\mathrm{K}_{\mathrm{dp}}$ [deg/km]'
        slabel = r'$\mathrm{K}_{\mathrm{dp}}$ [deg/km]'
        labelpad=35
        fontsize=32
         
    if varname == 'RHOHV':
        cmap = cmp.NWSRef
        levels = [0.8,0.9,0.92,0.94,0.96,0.97,0.975,0.9825,0.985,0.9875,0.99,0.995,1.]
        levels_ticks = levels.copy()
        clabel = r'Correlation Coefficient'
        slabel = r'$\rho_{\mathrm{hv}}$'
        labelpad = 30
        fontsize=32
        
    if varname == 'R':
        cmap = cmp.NWSRef
        levels = [0.,0.001,0.01,0.1,1.,5.,10.,15,20.,25.,50,100,150,200.]
        levels_ticks = levels.copy()
        clabel = r'Precip. Rate (mm/hr)'
        slabel = r'R (mm/hr)'
        labelpad = 30
        fontsize=32
        
    if varname == 'Nt':
        cmap = cmp.NWSRef
        levels = [0.,0.001,0.01,0.1,1.,5.,10.,15,20.,25.,50,100,150,200.]
        levels_ticks = levels.copy()
        clabel = r'Number Concentration (1/L)'
        slabel = r'$N_{t}$ (1/L)'
        labelpad = 30
        fontsize=22
        
    if varname == 'Dm':
        cmap = cmp.NWSRef
        levels = [0.1,0.25,0.5,0.75,1.,2., 3., 4.,5.,10.,15,20.]
        levels_ticks = levels.copy()
        clabel = r'Median Volume Diameter (mm)'
        slabel = r'$D_{0}$ (mm)'
        labelpad = 30
        fontsize=22
        
    if varname == 'WC':
        cmap = cmp.NWSRef
        levels = [0.,0.001,0.01,0.05,0.1,0.25,0.5,0.75,1.,2.5,5.,10.]
        levels_ticks = levels.copy()
        clabel = r'Water Content (g/cm$^{3}$)'
        slabel = r'WC (g/cm$^{3}$)'
        labelpad = 30
        fontsize=22
        

    
    return cmap, levels, levels_ticks, clabel, labelpad, fontsize, slabel