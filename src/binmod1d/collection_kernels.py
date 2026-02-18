# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 08:16:15 2025

@author: edwin.dunnavan
"""
import numpy as np

def CKE(di,dj,vi,vj):
    '''
    Collisional Kinetic Energy as defined by Low and List and others.

    '''
    
    rho_w = 1.0       # Density of water (g/cm^3)
    
    di2d,dj2d = np.meshgrid(di,dj,indexing='ij') # mm
    vi2d,vj2d = np.meshgrid(vi,vj,indexing='ij') # cm/s
    
    di2d *= 0.1 # convert to cm
    dj2d *= 0.1
    
    vi2d *= 100. 
    vj2d *= 100.
    
    return (np.pi/12.)*rho_w * (di2d**3*dj2d**3)/(di2d**3+dj2d**3)*np.abs(vj2d-vi2d)**2


def Weber_number(di,dj,vi,vj):
    
    # Physical Constants (CGS)
    rho_w = 1.0       # Density of water (g/cm^3)
    sigma = 72.8      # Surface tension of water (dyne/cm) at 20C
    
    di2d,dj2d = np.meshgrid(di,dj,indexing='ij')
    vi2d,vj2d = np.meshgrid(vi,vj,indexing='ij')
    
    di2d *= 0.1  # convert to cm
    dj2d *= 0.1 
    
    vi2d *= 100.  # convert to cm/s
    vj2d *= 100.
    
    # Ensure ds is the smaller diameter (if not already sorted)
    d_small = np.minimum(di2d, dj2d)
    
    # Convert inputs to CGS if they are in SI
    # Assuming model inputs: d [mm], vt [m/s] -> Convert to cm, cm/s
    # If your model uses different units, ADJUST THESE CONVERSIONS.
    dv_cm_s    = np.abs(vj2d - vi2d)# m/s -> cm/s
    
    # Calculate Weber Number (Inertia / Surface Tension)
    # We = (rho * d_small * dv^2) / sigma
    return (rho_w * d_small * dv_cm_s**2) / sigma

def Straub_params(di,dj,vi,vj):
    '''
    Straub et al. (2010) fragment distribution parameters

    '''
    Straub_dict = {}

    We = Weber_number(di, dj, vi, vj) # NOTE CGS units
    CW = 0.1*We*CKE(di,dj,vi,vj) # Units come out in ergs (i.e., 1e-7 Joules; 1 erg = 0.1 micro Joule), divide by 10 to get micro Joules
    
    di2d, dj2d = np.meshgrid(di,dj,indexing='ij')
    
    dl = 0.1*np.maximum(di2d,dj2d) # cm
    ds = 0.1*np.minimum(di2d,dj2d) # cm
    
    gam = dl/ds
    
    gamCW = gam*CW

    '''
    Distribution 1: Lognormal distribution
    D1_mean = 0.4 mm
    dD1 = 0.0125*CW**0.5
    Var = dD1**2/12
    
    sig2_1 = ln((Var/E**2)+1)
    mu1 = ln(E)-sig2_1/2.
    
    '''
    
    CW_thresh1 = (gamCW>7.) # Threshold in micro Joules
    
    D1_mean = 0.04  # cm
    dD_1 = 0.0125*CW**0.5
    Var1 = dD_1**2/12. # cm^2
    sig2_1 = np.log((Var1/(D1_mean**2))+1.)
    
    N1 = np.zeros_like(di2d,dtype=np.float64)
    
    N1[CW_thresh1] = 0.088*(gamCW[CW_thresh1]-7.0)
    
    mu1 = np.log(D1_mean) - 0.5 * sig2_1
    
    '''
    Distribution 2: Normal distribution
    D2_mean = 0.95 mm
    dD2 = 0.007*(CW-21.) for CW>=21 muJ
    Var = dD2**2/12
    
    sig2_1 = ln((Var/E**2)+1)
    mu1 = ln(E)-sig2_1/2.
    
    '''
    CW_thresh2 = (CW>=21.)
    D2_mean = 0.095 # cm
    dD_2 = np.zeros_like(di2d,np.float64)
    dD_2[CW_thresh2] = 0.007*(CW[CW_thresh2]-21.)
    
    sig2_2 = dD_2**2/12. 
    
    N2 = np.zeros_like(di2d,np.float64)
    N2[CW_thresh2] = 0.22*(CW[CW_thresh2]-21.)
    
    '''
    Distribution 3: Normal distribution
    D2_mean = 0.9*ds mm
    dD2 = 0.007*(CW-21.) for CW>=21 muJ
    Var = dD2**2/12
    
    sig2_1 = ln((Var/E**2)+1)
    mu1 = ln(E)-sig2_1/2.
    
    '''
    CW_thresh3 = (CW_thresh2) & (CW<=46.)
    D3_mean = 0.9*ds
    dD_3 = 0.01*(1.+0.76*CW**0.5)
    
    sig2_3 = dD_3**2/12. 
    
    N3 = np.zeros_like(di2d,np.float64)
    N3[~CW_thresh2] = 1.0 
    N3[CW_thresh3] = 0.04*(46.-CW[CW_thresh3])
    
    
    '''
    Distribution 4: Dirac function (residual)

    '''
    # Find dirac parameters through mass conservation of d1,d2 pair
    M31 = N1*np.exp(3.*mu1+9.*sig2_1/2.)
    M32 = N2*(D2_mean**3.+3.*D2_mean*sig2_2)
    M33 = N3*(D3_mean**3.+3.*D3_mean*sig2_3)

    # Residual mass (without prefactors; just need to map to original grid)
    M34 = dl**3+ds**3-(M31+M32+M33)
    
    # NOTE: Returned dictionary of distribution parameters. Converted to mm.
    Straub_dict['dist1'] = {'muf':np.log(10.*D1_mean) - 0.5 * sig2_1,
                            'sig2f':sig2_1,
                            'N':N1
                            }
    
    Straub_dict['dist2'] = {'mu':10.*D2_mean,
                            'sig2':100.*sig2_2,
                            'N':N2}
    
    Straub_dict['dist3'] = {'mu':10.*D3_mean,
                            'sig2':100.*sig2_3,
                            'N':N3}
    
    Straub_dict['dist4'] = {'x_res':M34}


    return Straub_dict
    
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


def straub_efficiency(di, dj, vi, vj, Ai, Aj):
        '''
        Calculates Coalescence Efficiency (E_coal) based on Straub et al. (2010).
        Equation: Ec = exp(-1.15 * We)
        
        Note: The parameterization is based on CGS units. 
        We must ensure the Weber number calculation uses consistent units.
        '''
        
        # Physical Constants (CGS)
        rho_w = 1.0       # Density of water (g/cm^3)
        sigma = 72.8      # Surface tension of water (dyne/cm) at 20C
        
        di2d,dj2d = np.meshgrid(di,dj,indexing='ij')
        Ai2d,Aj2d = np.meshgrid(Ai,Aj,indexing='ij')
        vi2d,vj2d = np.meshgrid(vi,vj,indexing='ij')
        
        # Ensure ds is the smaller diameter (if not already sorted)
        d_small = np.minimum(di2d, dj2d)
        
        # Convert inputs to CGS if they are in SI
        # Assuming model inputs: d [mm], vt [m/s] -> Convert to cm, cm/s
        # If your model uses different units, ADJUST THESE CONVERSIONS.
        d_small_cm = d_small * 0.1   # mm -> cm
        dv_cm_s    = np.abs(vj2d - vi2d) * 100.0 # m/s -> cm/s
        
        # Calculate Weber Number (Inertia / Surface Tension)
        # We = (rho * d_small * dv^2) / sigma
        We = (rho_w * d_small_cm * dv_cm_s**2) / sigma
        
        # Straub et al. (2010) Parameterization
        # Ec = exp(-1.15 * We)
        Ec = np.clip(np.exp(-1.15 * We),0.,1.0)
        
        large_cond = (di2d>0.06) | (dj2d>0.06)
        
        Kxy = hall_kernel(di,dj,vi,vj,Ai,Aj) # Hall kernel has Ecol * (Es=1.0) * Kxy_hydro
        
        Kxy_hydro = hydro_kernel(vi,vj,Ai,Aj)
        
        Kxy[large_cond] = Ec[large_cond]*Kxy_hydro[large_cond] # (Ecol=1.0)*Ecoal_straub * Kxy_hydro
        
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

    