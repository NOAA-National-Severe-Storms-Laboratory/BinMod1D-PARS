# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 08:23:48 2026

@author: edwin.dunnavan
"""

import numpy as np



 
        
def spheroid_factors(ar):
    
    La = (1./3.)*np.ones_like(ar)
    
    kap = np.zeros_like(ar)
    
    kap[ar<1.0] = np.sqrt(ar[ar<1.0]**(-2)-1)
    kap[ar>1.0] = np.sqrt(1-ar[ar>1.0]**(-2))
    
    La[ar<1.0] = ((1+kap[ar<1.0]**2)/kap[ar<1.0]**2) *(1-np.arctan(kap[ar<1.0])/kap[ar<1.0])
    La[ar>1.0] = ((1-kap[ar>1.0]**2)/kap[ar>1.0])*((1/(2*kap[ar>1.0]))*np.log((1+kap[ar>1.0])/(1-kap[ar>1.0]))-1)
    Lc = (1.-La)/2.
    
    
    return La, Lc 
        
def angular_moments(sigma):

       # Compute angular moments from Ryzhkov et al. (2011)
       sig = (np.pi/180) * sigma
       uu = np.exp(-2.0 * sig**2)
       ang1 = 0.25 * (1 + uu)**2
       ang2 = 0.25 * (1 - uu**2)
       ang3 = (0.375 + 0.5 * uu + 0.125 * uu**4)**2
       ang4 = ((0.375 - 0.5 * uu + 0.125 * uu**4) *
               (0.375 + 0.5 * uu + 0.125 * uu**4))
       ang5 = 0.125 * (0.375 + 0.5 * uu + 0.125 * uu**4) * (1 - uu**4)
       ang6 = 0.
       ang7 = 0.5 * uu * (1 + uu)
       
       angs = np.array([ang1,ang2,ang3,ang4,ang5,ang6,ang7])
       
       return angs        
   
    
def dielectric_ice(lamda,TK):
    # From Maetzler Matlab code based on Ray 1972
    # lambda in mm
    
    f = 299.792458/lamda # Convert to GHz
    B1 = 0.0207
    B2 = 1.16e-11
    b = 335
    
    deltabeta = np.exp(-10.02 + 0.0364*(TK-273))

    betam = (B1/TK) * ( np.exp(b/TK) / ((np.exp(b/TK)-1)**2) ) + B2*f**2

    beta = betam + deltabeta

    theta = 300 / TK - 1

    alfa = (0.00504 + 0.0062*theta)*np.exp(-22.1*theta)

    ei = complex(3.1884 + 9.1e-4*(TK-273),(alfa/f)+beta*f)

    return ei    
 

def dielectric_water(t,eps_0,t0=273.15,wave=110.):
    """
    Calculate dielectric constant for fresh water at temperature T based on
    Ray (1972).

    Input:
        Temperature [K]
    Output:
        Dielectric constant
    """

    ew_eps_s = (78.54 * (1.0 - 4.579e-3 * (t - t0 - 25) +
                         1.19e-5 * (t - t0 - 25)**2 -
                          2.8e-8 * (t - t0 - 25)**3))
    ew_eps_inf = 5.27137 + 2.16474e-2 * (t - t0) - 1.31198e-3 * (t - t0)**2
    ew_alpha = (-16.8129 / t) + 6.09265e-2
    ew_lambda = 3.3836e-6 * np.exp(2513.98 / t)
    ew_sigma = 1.1117e-4
    ew_real = ew_eps_inf + (((ew_eps_s - ew_eps_inf) * (1 + (ew_lambda / (0.001 * wave))**(1 - ew_alpha) * np.sin(ew_alpha * np.pi / 2))) /
                                        (1 + 2 * (ew_lambda / (0.001 * wave))**(1 - ew_alpha) * np.sin(ew_alpha * np.pi/2) + (ew_lambda / (0.001 * wave))**(2 * (1 - ew_alpha))))
    ew_imag = (((ew_eps_s - ew_eps_inf) * ((ew_lambda / (0.001 * wave))**(1 - ew_alpha) * np.cos(ew_alpha * np.pi/2))) /
                                        (1 + 2*(ew_lambda / (0.001 * wave))**(1 - ew_alpha) * np.sin(ew_alpha * np.pi/2) + (ew_lambda / (0.001 * wave))**(2 * (1 - ew_alpha)))
                                        + (ew_sigma * (0.001 * wave)) / (2 * np.pi * 3e8 * eps_0))
    ew = complex(ew_real, ew_imag)

    return ew
