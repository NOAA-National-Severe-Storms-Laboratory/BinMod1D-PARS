# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 09:40:10 2025

@author: edwin.dunnavan
"""

import numpy as np
import mpmath
from scipy.special import gamma, hyp2f1, iv, i0, i1
from scipy.optimize import fsolve

def Scott_moments(r,t,nu,E,B,kernel_type='Golovin'):
    
    T = E*t
    
    if kernel_type=='Golovin':
    
        #tau = 1 - np.exp(-0.00153*t) # normalized time
        
        tau = 1 - np.exp(-T) # normalized time
        
        gam_series = np.zeros_like(tau)
        M_rt = np.zeros_like(tau)
        
        for tt in range(len(t)):
            gam_series[tt] = mpmath.nsum(lambda k: tau[tt]**(k) *nu**(nu*(k+1))/\
                            ((tau[tt]+nu)**(r+nu+k*(nu+1))* mpmath.factorial(k+1))*\
                            (mpmath.gamma(nu+r+k*(nu+1))/\
                             mpmath.gamma(nu*(k+1))),[0,np.inf],method='direct')
    
        M_rt[tt] = (1-tau[tt]) *gam_series
        
    elif kernel_type == 'Product':
        
        #T = (7/40)*0.00959*t # ELD NOTE: Apparently 7/40 factor is necessary to get Scott's solutions in his paper.
        
        #T = (7/40)*0.00959*t
        
        #T = 0.*t
        
        #T = 0.0016*t # This is approximately the correct factor.
        
        
        #T = 0.043*t
        
        #T =  (7/48)*E*t 
        
        M_rt = np.zeros_like(tau)
        
        for tt in range(len(t)):
            M_rt[tt] = mpmath.nsum(lambda k: (T[tt]**k)*nu**((k+1)*(nu+1))*\
                                   (T[tt]+nu)**(-(nu+r+k*(nu+2)))*mpmath.gamma(r+nu+k*(nu+2))/\
                                   (mpmath.factorial(k+1)*mpmath.gamma((k+1)*(nu+1))),[0,np.inf],method='direct')
 
    elif kernel_type == 'Constant':
        
        #T =  0.0429*t
        
        #T = E*t
        
       # T = 0.0016
        
        M_rt = np.zeros_like(tau)
        
        for tt in range(len(t)):
            M_rt[tt] =  (4./(T[tt]+2)**2)*nu**(-r)*\
                        mpmath.nsum(lambda k: (mpmath.gamma(r+nu*(k+1))*((T[tt]/(T[tt]+2))**k)/\
                                              mpmath.gamma(nu*(k+1))),[0,np.inf],method='direct')
 

    return M_rt


def Feingold_moments(r,t,nu,B,gam,kernel_type='SBE'):
    
    M_rt = np.zeros_like(t)
    
    if kernel_type=='SBE':
        
        M0r = nu**(-r)*gamma(nu+r)/gamma(nu)
     
        for tt in range(len(t)):
            den = (1+(1/gam)*(np.exp(B*gam*t[tt])-1))   
            M_rt[tt] = (M0r+gam*(np.exp(B*gam*t[tt])-1)*gam**(-1-r)*gamma(1+r))/den
     
    elif kernel_type=='SCE/SBE':
    
        #E = 1 - t
        
        E = 1. - B
 
        eta = 0.5*E*gam*(2-E)
 
        M_rt = 0.5*(1-E)*gam**2*gamma(r+1)*(gam-eta)**(-r-2)*\
        (2*(gam-eta)*hyp2f1(0.5*(r+1),0.5*(r+2),1,(eta/(gam-eta))**2)-\
         eta*(r+1)*np.sign(gam-eta)*hyp2f1(0.5*(r+2), 0.5*(r+3),2,(eta/(gam-eta))**2))

    
    return M_rt


def Scott_dists(xbins,Eagg,nu,t,kernel_type='Golovin'):
    
    xt = 50
    
    #mpmath.dps = 10 
    
    bins = len(xbins)
    Tlen = len(t)
    
    #mpmath.mp.prec = 500
    
    # Distribution functions for each bin
    npbins = np.zeros([bins, Tlen])
    #mpbins = np.zeros([bins, Tlen])
    #zpbins = np.zeros([bins, Tlen])

        
    # Calculate initial grid
    npbins[:,0] = (1./gamma(nu))*(nu**nu)*xbins**(nu-1)*np.exp(-nu*xbins)
    
    
    #T = 0.00153*t # This seems right
    
    T = Eagg*t # This seems right
    
    if kernel_type=='Golovin':
        
        #tau = 1 - np.exp(-0.00153*t) #  normalized time
        
        tau = 1 - np.exp(-T) #  normalized time
    
        for xx in range(1,len(xbins)):
            
            if xbins[xx]<xt:
                
                for tt in range(1,len(t)):
                
                    npbins[xx,tt] = (1-tau[tt])*xbins[xx]**(nu-1)*np.exp(-(nu+tau[tt])*xbins[xx])*\
                    mpmath.nsum(lambda k: tau[tt]**(k) *nu**(nu*(k+1))/\
                    (mpmath.factorial(k+1)*mpmath.gamma(nu*(k+1))) * xbins[xx]**(k*(nu+1)),[0,np.inf],method='direct')
                  
                        
                        
            else: # Saddle-point approximation
            
                npbins[xx,1:] = (1-tau[1:])*np.exp(-(tau[1:]+nu)*\
                       xbins[xx]+(nu+1)*xbins[xx]*tau[1:]**(1./(nu+1)))/(xbins[xx]**(1.5)*\
                       tau[1:]**((2.*(nu-1)+3)/(2.*(nu-1)+4))*(2.*np.pi*(nu+1)/nu)**0.5)*\
                       (1-((2.*(nu-1)+3)*(nu+2))/(24.*xbins[xx]*tau[1:]**(1/(nu+1))*nu*(nu+1)))
            
    
    elif kernel_type=='Product':
        
        #T = (7/40)*0.00959*t # 7/40 factor apparently is necessary here
        
        #T = 0.00153*t # This seems right
        
        #T = 0.00959*t # 7/40 factor apparently is necessary here
        
        
        for xx in range(1,len(xbins)):
            
            if xbins[xx]<xt:
                
                for tt in range(1,len(t)):
                
                    npbins[xx,tt] =  (np.exp(-xbins[xx]*(T[tt]+nu)))*mpmath.nsum(lambda k: ((xbins[xx])**((nu-1)+\
                    k*(nu+2))*nu**((nu+1)*(k+1))/(mpmath.factorial(k+1)*mpmath.gamma((nu+1)*\
                    (k+1))))*(T[tt])**k,[0,np.inf],method='direct')
               
       
            else: # Saddle-point approximation
            
                npbins[xx,1:] = ((nu+1)*np.exp(-xbins[xx]*(T[1:]+nu)+xbins[xx]*(nu+2)*T[1:]**(1./(nu+2))*(nu/(nu+1))**((nu+1)/(nu+2)))/\
                    (xbins[xx]**(5./2)*(T[1:]*(nu+1)/nu)**((2.*(nu+2)-1)/(2.*(nu+2)))*(2.*np.pi*(nu*(nu+2)))**(1./2.)))*\
                    (1-((nu+3)*(2.*(nu+2)-1))/(24.*xbins[xx]*T[1:]**(1./(nu+2))*nu**((nu+1)/(nu+2))*(nu+1)**(1./(nu+2))*(nu+2)))
                             
        
    elif kernel_type=='Constant':
        
        #T = 0.0429*t 
        
        #T = 0.0016*t
        
        for xx in range(1,len(xbins)):
            
            if xbins[xx]<xt:
                
                for tt in range(1,len(t)):
                
                    npbins[xx,tt] =  ((4.*np.exp(-nu*xbins[xx]))/\
                                     (xbins[xx]*(T[tt]+2)**2))*mpmath.nsum(lambda k:\
                                    ((xbins[xx]*nu)**(nu*(k+1))/\
                                    mpmath.gamma(nu*(k+1)))*(T[tt]/(T[tt]+2))**k,[0,np.inf],method='direct')

            else: # Saddle-point approximation
            
            
                ys = np.zeros_like(T)

                for tt in range(1,len(t)):
                    
                    ys_eq = lambda ys: (T[tt]/(T[tt]+2))-ys**(nu-1)*(ys-1./xbins[xx])
                    
                    ys[tt] = fsolve(ys_eq,1)[0] 

                npbins[xx,1:] = 0.9221*4.*(nu)*np.exp((ys[1:]-1)*(nu)*xbins[xx])/((T[1:]+2)**2*(ys[1:]**(nu-1))*(2.*np.pi*nu*(nu-(nu-1)/(ys[1:]*xbins[xx])))**(0.5))

    
    return npbins



def Feingold_dists(xbins,t,nu,E,B,gam,kernel_type='SBE'):
    
    if kernel_type=='SBE':
        n0_xt = (1./gamma(nu))*(nu**nu)*xbins[:,None]**(nu-1)*np.exp(-nu*xbins[:,None]) # initial dist.
        npbins = (n0_xt+gam*(np.exp(B*gam*t[None,:])-1)*np.exp(-gam*xbins[:,None]))/(1+(1./gam)*(np.exp(B*gam*t[None,:])-1)) 
        
    elif kernel_type=='SCE/SBE':
        
        # NOTE: only valid for C = K*E and B = K*(1-E)
        
        
        #E_new = 1.*E
        
        E_new = 1000.*E
    
        eta = 0.5*E_new*gam*(2.-E_new)
    
        npbins = (1-E_new)*gam**2*(i0(eta*xbins)-i1(eta*xbins))*np.exp(-(gam-eta)*xbins)
    
    
        #print('kernel test')
    
    # Check total mass
   # Mt = Feingold_moments(1,t,nu,B,gam,kernel_type=kernel_type)
    
    #print('Total Mass = ',Mt)
    
    
    return npbins