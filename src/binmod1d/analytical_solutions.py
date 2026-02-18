# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 09:40:10 2025

@author: edwin.dunnavan
"""
import numpy as np
import mpmath
from scipy.special import gamma, gammaln, hyp2f1, i0, i1
from scipy.special import i0e, i1e # Import scaled Bessel functions
#from scipy.optimize import fsolve

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


def Scott_dists(xbins, Eagg, nu, t, kernel_type='Golovin', max_k=100):
    bins = len(xbins)
    Tlen = len(t)
    npbins = np.zeros([bins, Tlen])
    
    # 1. Initial State (t=0)
    # Handle the singularity at x=0 for nu < 1
    with np.errstate(divide='ignore', invalid='ignore'):
        npbins[:, 0] = (1./gamma(nu)) * (nu**nu) * xbins**(nu-1) * np.exp(-nu*xbins)
    np.nan_to_num(npbins[:, 0], copy=False) # Fix any 0^negative_power issues
    
    T = Eagg * t
    
    # --- THRESHOLD SELECTION ---
    # The series expansions for Product and Constant kernels are numerically
    # explosive for x > 10. We rely on the Saddle Point method there.
    if kernel_type == 'Constant':
        xt = 10.0  
    elif kernel_type == 'Product':
        xt = 10.0 # Switch to saddle point earlier to avoid series overflow
    else:
        xt = 50.0 # Golovin is more stable
        
    mask_sm = xbins < xt
    mask_lg = ~mask_sm
    
    X_sm = xbins[mask_sm, None]
    X_lg = xbins[mask_lg, None]
    T_vec = T[None, 1:]
    k = np.arange(max_k).reshape(-1, 1, 1)

    eps = 1e-20

    if kernel_type == 'Golovin':
        tau = 1 - np.exp(-T_vec)
        if np.any(mask_sm):
            # Log Prefactor
            ln_pre = np.log(tau + eps) + (nu - 1) * np.log(X_sm + eps) - (nu + tau) * X_sm
            
            # Log Terms
            ln_coeffs = (nu * (k+1)) * np.log(nu) - gammaln(k+2) - gammaln(nu * (k+1))
            ln_powers = k * np.log(tau + eps) + k * (nu + 1) * np.log(X_sm + eps)
            log_terms = ln_coeffs + ln_powers
            
            # Combined Log-Sum-Exp
            max_log = np.max(log_terms, axis=0)
            sum_exp = np.sum(np.exp(log_terms - max_log), axis=0)
            
            # Result: (1-tau)/tau * exp(pre + max_log) * sum_exp
            # Note: (1-tau)/tau cancels the first tau term cleanly
            term = ((1 - tau) / (tau + eps)) * np.exp(ln_pre + max_log) * sum_exp
            npbins[mask_sm, 1:] = term

    elif kernel_type == 'Product':
        if np.any(mask_sm):
            # 1. Log Prefactor: -x(T+nu)
            ln_pre = -X_sm * (T_vec + nu)
            
            # 2. Log Terms
            # ln_coeffs = (nu+1)(k+1)ln(nu) - ln(k+1)! - ln(gamma((nu+1)(k+1)))
            ln_coeffs = ((nu + 1) * (k + 1)) * np.log(nu) - gammaln(k + 2) - gammaln((nu + 1) * (k + 1))
            # ln_powers = ((nu-1) + k(nu+2))ln(x) + k*ln(T)
            ln_powers = ((nu - 1) + k * (nu + 2)) * np.log(X_sm + eps) + k * np.log(T_vec + eps)
            
            log_terms = ln_coeffs + ln_powers
            
            # 3. Combined Log-Sum-Exp (Prevents Overflow)
            max_log = np.max(log_terms, axis=0)
            sum_exp = np.sum(np.exp(log_terms - max_log), axis=0)
            
            # Direct combination avoids intermediate infinity
            npbins[mask_sm, 1:] = np.exp(ln_pre + max_log) * sum_exp

    elif kernel_type == 'Constant':
        if np.any(mask_sm):
            # Log Prefactor
            ln_pre = np.log(4.0) - (nu * X_sm) - np.log(X_sm + eps) - 2.0 * np.log(T_vec + 2.0)
            
            # Log Terms
            ln_coeffs = -gammaln(nu * (k + 1))
            ln_x_part = (nu * (k + 1)) * np.log((X_sm + eps) * nu)
            ln_t_part = k * (np.log(T_vec + eps) - np.log(T_vec + 2.0))
            
            log_terms = ln_coeffs + ln_x_part + ln_t_part
            
            # Combined Log-Sum-Exp
            max_log = np.max(log_terms, axis=0)
            sum_exp = np.sum(np.exp(log_terms - max_log), axis=0)
            
            npbins[mask_sm, 1:] = np.exp(ln_pre + max_log) * sum_exp

    # --- Saddle-point Logic (Large x) ---
    if np.any(mask_lg):
        if kernel_type == 'Golovin':
            tau = 1 - np.exp(-T_vec)
            num = (1 - tau) * np.exp(-(tau + nu) * X_lg + (nu + 1) * X_lg * tau**(1./(nu + 1)))
            den = (X_lg**1.5 * tau**((2.*(nu-1)+3)/(2.*(nu-1)+4)) * (2.*np.pi*(nu+1)/nu)**0.5)
            corr = (1 - ((2.*(nu-1)+3)*(nu+2)) / (24.*X_lg * tau**(1/(nu+1)) * nu*(nu+1)))
            npbins[mask_lg, 1:] = (num / den) * corr

        elif kernel_type == 'Product':
            phi = X_lg * (nu + 2) * T_vec**(1./(nu+2)) * (nu/(nu+1))**((nu+1)/(nu+2))
            num = (nu+1) * np.exp(-X_lg * (T_vec + nu) + phi)
            den = (X_lg**2.5 * (T_vec*(nu+1)/nu)**((2.*nu+3)/(2.*nu+4)) * (2.*np.pi*nu*(nu+2))**0.5)
            corr = (1 - ((nu+3)*(2.*nu+3)) / (24.*X_lg * T_vec**(1./(nu+2)) * nu**((nu+1)/(nu+2)) * (nu+1)**(1./(nu+2)) * (nu+2)))
            npbins[mask_lg, 1:] = (num / den) * corr

        elif kernel_type == 'Constant':
            R = T_vec / (T_vec + 2)
            ys = np.ones((X_lg.shape[0], T_vec.shape[1]))
            # Vectorized Newton-Raphson
            for _ in range(5):
                f = R - ys**(nu-1) * (ys - 1.0/X_lg)
                df = -( (nu-1) * ys**(nu-2) * (ys - 1.0/X_lg) + ys**(nu-1) )
                ys -= f/df
            
            term1 = 0.9221 * 4. * nu * np.exp((ys - 1) * nu * X_lg)
            term2 = (T_vec + 2)**2 * (ys**(nu-1))
            term3 = (2. * np.pi * nu * (nu - (nu - 1) / (ys * X_lg)))**0.5
            npbins[mask_lg, 1:] = term1 / (term2 * term3)

    return npbins

# def Scott_dists_working(xbins, Eagg, nu, t, kernel_type='Golovin', max_k=40):
#     """
#     Fully vectorized version of Scott distributions.
#     Replaces mpmath and scalar loops with NumPy tensor operations.
    
#     ELD NOTE: I used Google Gemini to optimize the old Scott_dists() function 
#     (see below; kept for posterity).
    
#     """
#     bins = len(xbins)
#     Tlen = len(t)
#     npbins = np.zeros([bins, Tlen])
    
#     # 1. Initial State (t=0)
#     npbins[:, 0] = (1./gamma(nu)) * (nu**nu) * xbins**(nu-1) * np.exp(-nu*xbins)
    
#     # Time and Bins setup
#     T = Eagg * t
#     xt = 50
#     mask_sm = xbins < xt
#     mask_lg = ~mask_sm
    
#     # Prepare 2D grid broadcast shapes
#     X_sm = xbins[mask_sm, None]  # (N_sm, 1)
#     X_lg = xbins[mask_lg, None]  # (N_lg, 1)
#     T_vec = T[None, 1:]          # (1, Tlen-1)
#     k = np.arange(max_k).reshape(-1, 1, 1) # (K, 1, 1) for series summation

#     if kernel_type == 'Golovin':
#         tau = 1 - np.exp(-T_vec)
        
#         # Series Approximation (Small x)
#         if np.any(mask_sm):
#             log_coeffs = (nu * (k+1)) * np.log(nu) - gammaln(k+2) - gammaln(nu * (k+1))
#             term_powers = (tau**k) * (X_sm**(k * (nu+1)))
#             series_sum = np.sum(np.exp(log_coeffs) * term_powers, axis=0)
            
#             prefactor = (1 - tau) * X_sm**(nu-1) * np.exp(-(nu + tau) * X_sm)
#             npbins[mask_sm, 1:] = prefactor * series_sum

#         # Saddle-point Approximation (Large x)
#         if np.any(mask_lg):
#             num = (1 - tau) * np.exp(-(tau + nu) * X_lg + (nu + 1) * X_lg * tau**(1./(nu + 1)))
#             den = (X_lg**1.5 * tau**((2.*(nu-1)+3)/(2.*(nu-1)+4)) * (2.*np.pi*(nu+1)/nu)**0.5)
#             corr = (1 - ((2.*(nu-1)+3)*(nu+2)) / (24.*X_lg * tau**(1/(nu+1)) * nu*(nu+1)))
#             npbins[mask_lg, 1:] = (num / den) * corr

#     elif kernel_type == 'Product':
#         # Series Approximation (Small x)
#         if np.any(mask_sm):
#             log_coeffs = ((nu+1)*(k+1)) * np.log(nu) - gammaln(k+2) - gammaln((nu+1)*(k+1))
#             term_powers = (X_sm**((nu-1) + k*(nu+2))) * (T_vec**k)
#             series_sum = np.sum(np.exp(log_coeffs) * term_powers, axis=0)
            
#             npbins[mask_sm, 1:] = np.exp(-X_sm * (T_vec + nu)) * series_sum

#         # Saddle-point Approximation (Large x)
#         if np.any(mask_lg):
#             phi = X_lg * (nu + 2) * T_vec**(1./(nu+2)) * (nu/(nu+1))**((nu+1)/(nu+2))
#             num = (nu+1) * np.exp(-X_lg * (T_vec + nu) + phi)
#             den = (X_lg**2.5 * (T_vec*(nu+1)/nu)**((2.*nu+3)/(2.*nu+4)) * (2.*np.pi*nu*(nu+2))**0.5)
#             corr = (1 - ((nu+3)*(2.*nu+3)) / (24.*X_lg * T_vec**(1./(nu+2)) * nu**((nu+1)/(nu+2)) * (nu+1)**(1./(nu+2)) * (nu+2)))
#             npbins[mask_lg, 1:] = (num / den) * corr

#     elif kernel_type == 'Constant':
#         # Series Approximation (Small x)
#         if np.any(mask_sm):      
#             log_pre = np.log(4.0) - (nu * X_sm) - np.log(X_sm) - 2.0 * np.log(T_vec + 2.0)
#             log_coeffs = -gammaln(nu * (k+1))
            
#             #x_term = (X_sm * nu)**(nu * (k+1))
#             #x_term = np.exp((nu*(k+1))*np.log((X_sm * nu)))
#             #t_term = (T_vec / (T_vec + 2))**k
            
#             ln_x_part = (nu * (k + 1)) * np.log(X_sm * nu)
#             ln_t_part = k * (np.log(T_vec) - np.log(T_vec + 2.0))
#             log_series_terms = log_coeffs + ln_x_part + ln_t_part # Shape (K, N_sm, T_len-1)
            
#             # 3. Log-Sum-Exp Trick to prevent overflow
#             # We find the max log term for each (bin, time) pair
#             max_log = np.max(log_series_terms, axis=0) 
#             # Sum exp(log_terms - max_log) is numerically stable (max value is 1)
#             sum_exp = np.sum(np.exp(log_series_terms - max_log), axis=0)
#             # 4. Combine: exp(log_pre + max_log) * sum_exp
#             # This effectively calculates exp(log_pre) * sum(exp(log_series_terms))
#             npbins[mask_sm, 1:] = np.exp(log_pre + max_log) * sum_exp
            
#             #series_sum = np.sum(np.exp(log_coeffs) * x_term * t_term, axis=0)
#             #prefactor = (4. * np.exp(-nu * X_sm)) / (X_sm * (T_vec + 2)**2)
#             #npbins[mask_sm, 1:] = prefactor * series_sum

#         # Vectorized Newton-Raphson & Saddle-point (Large x)
#         if np.any(mask_lg):
#             R = T_vec / (T_vec + 2)
#             ys = np.ones((X_lg.shape[0], T_vec.shape[1]))
#             for _ in range(5): # 5 iterations for convergence
#                 f = R - ys**(nu-1) * (ys - 1.0/X_lg)
#                 df = -( (nu-1) * ys**(nu-2) * (ys - 1.0/X_lg) + ys**(nu-1) )
#                 ys -= f/df
            
#             term1 = 0.9221 * 4. * nu * np.exp((ys - 1) * nu * X_lg)
#             term2 = (T_vec + 2)**2 * (ys**(nu-1))
#             term3 = (2. * np.pi * nu * (nu - (nu - 1) / (ys * X_lg)))**0.5
#             npbins[mask_lg, 1:] = term1 / (term2 * term3)

#     return npbins


# def Scott_dists_OLD(xbins,Eagg,nu,t,kernel_type='Golovin'):
    
#     xt = 50
    
#     #mpmath.dps = 10 
    
#     bins = len(xbins)
#     Tlen = len(t)
    
#     #mpmath.mp.prec = 500
    
#     # Distribution functions for each bin
#     npbins = np.zeros([bins, Tlen])
#     #mpbins = np.zeros([bins, Tlen])
#     #zpbins = np.zeros([bins, Tlen])

        
#     # Calculate initial grid
#     npbins[:,0] = (1./gamma(nu))*(nu**nu)*xbins**(nu-1)*np.exp(-nu*xbins)
    
    
#     #T = 0.00153*t # This seems right
    
#     T = Eagg*t # This seems right
    
#     if kernel_type=='Golovin':
        
#         #tau = 1 - np.exp(-0.00153*t) #  normalized time
        
#         tau = 1 - np.exp(-T) #  normalized time
    
#         for xx in range(1,len(xbins)):
            
#             if xbins[xx]<xt:
                
#                 for tt in range(1,len(t)):
                
#                     npbins[xx,tt] = (1-tau[tt])*xbins[xx]**(nu-1)*np.exp(-(nu+tau[tt])*xbins[xx])*\
#                     mpmath.nsum(lambda k: tau[tt]**(k) *nu**(nu*(k+1))/\
#                     (mpmath.factorial(k+1)*mpmath.gamma(nu*(k+1))) * xbins[xx]**(k*(nu+1)),[0,np.inf],method='direct')
                  
                        
                        
#             else: # Saddle-point approximation
            
#                 npbins[xx,1:] = (1-tau[1:])*np.exp(-(tau[1:]+nu)*\
#                        xbins[xx]+(nu+1)*xbins[xx]*tau[1:]**(1./(nu+1)))/(xbins[xx]**(1.5)*\
#                        tau[1:]**((2.*(nu-1)+3)/(2.*(nu-1)+4))*(2.*np.pi*(nu+1)/nu)**0.5)*\
#                        (1-((2.*(nu-1)+3)*(nu+2))/(24.*xbins[xx]*tau[1:]**(1/(nu+1))*nu*(nu+1)))
            
    
#     elif kernel_type=='Product':
        
#         #T = (7/40)*0.00959*t # 7/40 factor apparently is necessary here
        
#         #T = 0.00153*t # This seems right
        
#         #T = 0.00959*t # 7/40 factor apparently is necessary here
        
        
#         for xx in range(1,len(xbins)):
            
#             if xbins[xx]<xt:
                
#                 for tt in range(1,len(t)):
                
#                     npbins[xx,tt] =  (np.exp(-xbins[xx]*(T[tt]+nu)))*mpmath.nsum(lambda k: ((xbins[xx])**((nu-1)+\
#                     k*(nu+2))*nu**((nu+1)*(k+1))/(mpmath.factorial(k+1)*mpmath.gamma((nu+1)*\
#                     (k+1))))*(T[tt])**k,[0,np.inf],method='direct')
               
       
#             else: # Saddle-point approximation
            
#                 npbins[xx,1:] = ((nu+1)*np.exp(-xbins[xx]*(T[1:]+nu)+xbins[xx]*(nu+2)*T[1:]**(1./(nu+2))*(nu/(nu+1))**((nu+1)/(nu+2)))/\
#                     (xbins[xx]**(5./2)*(T[1:]*(nu+1)/nu)**((2.*(nu+2)-1)/(2.*(nu+2)))*(2.*np.pi*(nu*(nu+2)))**(1./2.)))*\
#                     (1-((nu+3)*(2.*(nu+2)-1))/(24.*xbins[xx]*T[1:]**(1./(nu+2))*nu**((nu+1)/(nu+2))*(nu+1)**(1./(nu+2))*(nu+2)))
                             
        
#     elif kernel_type=='Constant':
        
#         #T = 0.0429*t 
        
#         #T = 0.0016*t
        
#         for xx in range(1,len(xbins)):
            
#             if xbins[xx]<xt:
                
#                 for tt in range(1,len(t)):
                
#                     npbins[xx,tt] =  ((4.*np.exp(-nu*xbins[xx]))/\
#                                      (xbins[xx]*(T[tt]+2)**2))*mpmath.nsum(lambda k:\
#                                     ((xbins[xx]*nu)**(nu*(k+1))/\
#                                     mpmath.gamma(nu*(k+1)))*(T[tt]/(T[tt]+2))**k,[0,np.inf],method='direct')

#             else: # Saddle-point approximation
            
            
#                 ys = np.zeros_like(T)

#                 for tt in range(1,len(t)):
                    
#                     ys_eq = lambda ys: (T[tt]/(T[tt]+2))-ys**(nu-1)*(ys-1./xbins[xx])
                    
#                     ys[tt] = fsolve(ys_eq,1)[0] 

#                 npbins[xx,1:] = 0.9221*4.*(nu)*np.exp((ys[1:]-1)*(nu)*xbins[xx])/((T[1:]+2)**2*(ys[1:]**(nu-1))*(2.*np.pi*nu*(nu-(nu-1)/(ys[1:]*xbins[xx])))**(0.5))

    
#     return npbins


def Feingold_dists(xbins, t, nu, E, B, gam, kernel_type='SBE'):
    
    
    if kernel_type == 'SBE':
        n0_xt = (1./gamma(nu))*(nu**nu)*xbins[:,None]**(nu-1)*np.exp(-nu*xbins[:,None]) # initial dist.
        npbins = (n0_xt+gam*(np.exp(B*gam*t[None,:])-1)*np.exp(-gam*xbins[:,None]))/(1+(1./gam)*(np.exp(B*gam*t[None,:])-1)) 
        
    elif kernel_type == 'SCE/SBE':
        # E_new logic (kept as provided)
        E_new = 1000. * E 
        eta = 0.5 * E_new * gam * (2. - E_new)
        
        # --- THE FIX ---
        # Original: (i0(eta*x) - i1(eta*x)) * exp(-(gam - eta)*x)
        # Identity: i0(x) = i0e(x) * exp(x)
        # Subst:    (i0e(eta*x)*exp(eta*x) - i1e(eta*x)*exp(eta*x)) * exp(-gam*x + eta*x)
        # Factor:   exp(eta*x) * (i0e - i1e) * exp(-gam*x) * exp(eta*x) -> WRONG ALGEBRA ABOVE
        
        # Correct Algebra:
        # We want: [ I0(eta*x) - I1(eta*x) ] * exp( - (gam - eta)*x )
        #        = [ i0e(eta*x)*e^(eta*x) - i1e(eta*x)*e^(eta*x) ] * e^(-gam*x + eta*x)
        #        = e^(eta*x) * [ i0e(eta*x) - i1e(eta*x) ] * e^(-gam*x) * e^(eta*x)
        #        = e^(2*eta*x - gam*x) * [ i0e(eta*x) - i1e(eta*x) ]
        
        # Wait, usually the decaying term matches the growing term. 
        # Let's check the argument of exp.
        # Argument is -(gam - eta)*x = -gam*x + eta*x.
        # Bessel grows as exp(eta*x). 
        # Total exponent = eta*x + (-gam*x + eta*x) = 2*eta*x - gam*x.
        
        # If 2*eta - gam <= 0, this is stable. 
        # If 2*eta - gam > 0, it grows physically (and numerically).
        
        # Implementing using the scaled versions:
        arg = eta * xbins
        
        # We replace exp(-(gam-eta)x) with exp(-(gam-2*eta)x) because i0e removed one exp(eta*x)
        # Decay factor adjusted: exp( -(gam - eta)*x + eta*x ) = exp( -(gam - 2*eta)*x )
        
        decay_factor = np.exp(-(gam - 2.0 * eta) * xbins)
        bessel_diff = i0e(arg) - i1e(arg)
        
        npbins = (1 - E_new) * gam**2 * bessel_diff * decay_factor
        

    return npbins


def Feingold_dists_ORIG(xbins,t,nu,E,B,gam,kernel_type='SBE'):
    
    if kernel_type=='SBE':
        n0_xt = (1./gamma(nu))*(nu**nu)*xbins[:,None]**(nu-1)*np.exp(-nu*xbins[:,None]) # initial dist.
        npbins = (n0_xt+gam*(np.exp(B*gam*t[None,:])-1)*np.exp(-gam*xbins[:,None]))/(1+(1./gam)*(np.exp(B*gam*t[None,:])-1)) 
        
    elif kernel_type=='SCE/SBE':
        
        # NOTE: only valid for C = K*E and B = K*(1-E)
        
        
        #E_new = 1.*E
        
        E_new = 1000.*E
    
        eta = 0.5*E_new*gam*(2.-E_new)
    
        npbins = (1-E_new)*gam**2*(i0(eta*xbins)-i1(eta*xbins))*np.exp(-(gam-eta)*xbins)
    
    
    return npbins