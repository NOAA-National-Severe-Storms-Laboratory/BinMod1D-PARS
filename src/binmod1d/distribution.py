# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 08:35:18 2025

@author: edwin.dunnavan
"""

import numpy as np
import scipy.special as scip
from .bin_integrals import Pn

from .habits import habits

import matplotlib.pyplot as plt


import types

class dist():
    
    def __init__(self, sbin=4, bins=80, D1=0.01, dist_var='mass', kernel='Hydro',
                 habit_dict=None, ptype='rain', Tc=10., x0=None, mom_num=2):
        
        self.mom_num = mom_num
        
        # If no habit dict, then just use ptype here.
        if habit_dict is None:
            habit_dict = habits()[ptype] # Assuming habits() is defined elsewhere
            
        # 1. Setup the physical grid (empty bins)
        self.init_dist(sbin, bins, D1, dist_var=dist_var, kernel=kernel,
                       habit_dict=habit_dict, ptype=ptype, x0=x0, Tc=Tc, 
                       mom_num=mom_num)
        
        self.gam_norm = False
        
        init_method = habit_dict.get('init_method','gamma')
        
        # 2. Dispatcher: Populate the grid based on the chosen method
        if init_method == 'gamma':
             
            Nt0 = habit_dict.get('Nt0',1.)
            Mt0 = habit_dict.get('Mt0',1.)
            Dm0 = habit_dict.get('Dm0',2.0)
            mbar = habit_dict.get('mbar0',None)
            mu0 = habit_dict.get('mu0',3.)
            gam_norm = habit_dict.get('gam_norm',False)
            
            self.gam_norm = gam_norm
            
            self.bin_gamma_dist(Nt0=Nt0,Mt0=Mt0,mbar=mbar,mu0=mu0,Dm0=Dm0,normalize=gam_norm)
            
            # Add in parameters to habit_dict dictionary in case none are provided
            habit_dict['Nt0'] = Nt0
            habit_dict['Mt0'] = Mt0 
            habit_dict['mu0'] = mu0
            habit_dict['Dm0'] = Dm0
            habit_dict['gam_norm'] = gam_norm

        elif init_method == 'analytical':
            
            if 'func_nD' not in habit_dict:
                raise ValueError('User needs to supply func_nD as keyword argument')
            
            func_nD = habit_dict.get('func_nD',lambda D: 1000.*np.exp(-D/1.))
            
            self.bin_analytical_dist(func_nD, var=dist_var)
            
        elif init_method == 'empirical':
            
            if dist_var=='size':
                var_edges = self.d_edges 
            else:
                var_edges = self.x_edges
            
            user_edges = habit_dict.get('edges',var_edges)
            
            user_nD = habit_dict.get('nD',np.zeros((self.bins,)))
            
            self.bin_empirical_dist(user_edges, user_nD)
            
        elif init_method == 'empirical_counts':
            
            if dist_var=='size':
                var_edges = self.d_edges 
            else:
                var_edges = self.x_edges
            
            user_edges = habit_dict.get('edges',var_edges)
            
            user_N = habit_dict.get('bincounts',np.zeros((self.bins,)))
            
            self.bin_empirical_counts(user_edges, user_N)
            
        elif init_method == 'direct':
            # User directly supplied the Mbins / Nbins arrays
            
            if dist_var=='size':
                var_edges = self.d_edges 
            else:
                var_edges = self.x_edges
            
            user_edges = habit_dict.get('edges',var_edges)
            
            if ('Mbins' in list(habit_dict.keys())) and ('Nbins' not in list(habit_dict.keys())):
            
                self.Mbins = habit_dict.get('Mbins', np.zeros(self.bins))         
                self.Nbins = self.Mbins/self.xbins
                    
            if ('Mbins' not in list(habit_dict.keys())) and ('Nbins' in list(habit_dict.keys())):
            
                self.Nbins = habit_dict.get('Nbins', np.zeros(self.bins))        
                self.Mbins = self.Nbins*self.xbins
                          
        elif init_method == 'empty':
            self.Mbins = np.zeros(self.bins)
            self.Nbins = np.zeros(self.bins)
             
        else:
            raise ValueError(f"Unknown init_method: {init_method}")
                     
        # 3. Diagnostics
        if mom_num == 2:
            self.diagnose() 
        elif mom_num == 1:
            self.diagnose_1mom()

# # Number Distribution function class for arbitrary category
# class dist_OLD():
    
#     def __init__(self,sbin=4,bins=80,D1=0.01,x0=None,Nt0=1.,Mt0=1.,mbar=None,mu0=3,Dm0=2,gam_init=True,gam_norm=False,dist_var='mass',
#                  kernel='Hydro',habit_dict=None,ptype='rain',Tc=10.,mom_num=2,Mbins=None,Nbins=None):
        
#         self.mom_num = mom_num
        
#         # If no habit dict, than just use ptype here.
#         if habit_dict is None:
#             habit_dict = habits()[ptype]
        
#         self.init_dist(sbin,bins,D1,dist_var=dist_var,kernel=kernel,habit_dict=habit_dict,ptype=ptype,x0=x0,Tc=Tc,mom_num=mom_num,gam_norm=gam_norm)
        
#         if gam_init:
#             self.bin_gamma_dist(Nt0=Nt0,Mt0=Mt0,mbar=mbar,mu0=mu0,Dm0=Dm0,normalize=gam_norm)
            
#         if mom_num==2:
#             if (Mbins is not None) and (Nbins is not None):
#                 self.Mbins = Mbins 
#                 self.Nbins = Nbins
#             self.diagnose() 
            
#         elif mom_num==1:
#             if (Mbins is not None):
#                 self.Mbins = Mbins 
#             self.diagnose_1mom()
        
    def init_dist(self,sbin,bins,D1,kernel='Hydro',habit_dict=None,ptype='rain',Tc=10.,dist_var='mass',x0=None,mom_num=2):
        
        if habit_dict is None:
            habit_dict = habits()[ptype]
        
        #self.radar = radar
        self.kernel = kernel
        self.D1 = D1
        self.sbin = sbin 
        self.bins = bins
        
        self.ar = habit_dict['ar'] 
        self.br = habit_dict['br'] 
        self.arho = habit_dict['arho'] 
        self.brho = habit_dict['brho'] 
        self.av = habit_dict['av'] 
        self.bv = habit_dict['bv']
        self.sigma = habit_dict['sig']
        self.am = habit_dict['am']    # Units: g * mm^(-(3+brho)) 
        self.bm = habit_dict['bm']
        
        ar_func = habit_dict.get('ar',None)
        vt_func = habit_dict.get('vt',None)
            
        self.ptype = ptype
        self.mom_num = mom_num
        
        self.binl = np.arange(0,self.bins+1,1)
        self.rhobins = 2**(1./self.sbin) # scaling param for mass bins 
        
        if x0 is None:
            if dist_var=='size':
                self.x0= self.am*self.D1**self.bm # In grams
            else:
                self.x0 = 0.01
        else:
            self.x0 = x0
        
        self.xedges = (self.x0*self.rhobins**self.binl).astype(np.float64)
        
        self.x1 = self.xedges[:-1].copy() 
        self.x2 = self.xedges[1:].copy()
        
        self.xi1 = self.xedges[:-1].copy() 
        self.xi2 = self.xedges[1:].copy()
        
        self.xbins = 0.5*(self.xedges[:-1]+self.xedges[1:])
        self.dxbins = self.xedges[1:]-self.xedges[:-1]
        self.dxi = Pn(1,self.xi1,self.xi2)
        
        if mom_num == 1: 
            self.aki = np.zeros_like(self.xbins) 
            self.cki = np.ones_like(self.xbins)
            self.x1 = self.xi1.copy() 
            self.x2 = self.xi2.copy()
        
        self.d = (self.xbins / self.am)**(1. / self.bm)  
        self.dmax = self.ar**(-1./3.)*self.d**(1.-(self.br/3.))      
        
        self.d_edges = (self.xedges / self.am)**(1. / self.bm)  
        self.dmax_edges = self.ar**(-1./3.)*self.d_edges**(1.-(self.br/3.))   
        self.d1 = self.d_edges[:-1].copy()
        self.d2 = self.d_edges[1:].copy()
        self.dmax1 = self.dmax_edges[:-1].copy() 
        self.dmax2 = self.dmax_edges[1:].copy()
        
        rhoi = 0.92
        
        if ptype=='snow':
            
            if isinstance(ar_func,types.LambdaType):
                self.ar = ar_func(self.d)
                self.ar1 = ar_func(self.d1)
                self.ar2 = ar_func(self.d2)
                
            else:
                self.ar  = self.ar*np.ones_like(self.d1)
                self.ar1 = self.ar*np.ones_like(self.d1)
                self.ar2 = self.ar*np.ones_like(self.d2)
             
            self.rho = self.arho*self.d**(self.bm-3.)  
            self.rho[self.rho>rhoi] = rhoi
            
            self.rho1 = self.arho*self.d1**(self.bm-3.)  
            self.rho1[self.rho1>rhoi] = rhoi
            
            self.rho2 = self.arho*self.d2**(self.bm-3.)
            self.rho2[self.rho2>rhoi] = rhoi
            
        elif ptype=='rain':
                     
            if isinstance(ar_func,types.LambdaType):
                self.ar = ar_func(self.d)
                self.ar1 = ar_func(self.d1)
                self.ar2 = ar_func(self.d2)
            else: # DEFAULT for rain
                self.ar = extended_brandes(self.d)
                self.ar1 = extended_brandes(self.d1)
                self.ar2 = extended_brandes(self.d2)
                       
            self.rho = np.ones_like(self.d)  # g/cm^3
            self.rho1 = np.ones_like(self.d1) # g/cm^3
            self.rho2 = np.ones_like(self.d2) # g/cm^3
             
            #self.vt  = rain_terminal_velocity(self.d)
            #self.vt1 = rain_terminal_velocity(self.d1)
            #self.vt2 = rain_terminal_velocity(self.d2)
            #self.vt_edges = rain_terminal_velocity(self.d_edges)
        
        # if ptype=='rain':
        #     # Use Brandes (2002) relation which is a curve fit to laboratory measurements from
        #     # Gunn and Kinzer (1949) and Pruppacher and Pitter (1971)
        #     # See (https://doi.org/10.1175/1520-0450(2002)041<0674:EIREWA>2.0.CO;2)
        #     vt_brandes = lambda d: -0.1021 + 4.932*d-0.9551*d**2+0.07934*d**3-0.002362*d**4
        #     dmin_brandes = 0.1 
        #     dmax_brandes = 8.1
        #     #self.vt = -0.1021 + 4.932*self.d-0.9551*self.d**2+0.07934*self.d**3-0.002362*self.d**4
        #     #self.vt_edges = -0.1021 + 4.932*self.d_edges-0.9551*self.d_edges**2+0.07934*self.d_edges**3-0.002362*self.d_edges**4
        #     self.vt = vt_brandes(self.d)
        #     self.vt[self.d>dmax_brandes] = vt_brandes(dmax_brandes)
        #     self.vt[self.d<dmin_brandes] = vt_brandes(dmin_brandes)
            
        #     self.vt_edges = vt_brandes(self.d_edges)
        #     self.vt_edges[self.d_edges>dmax_brandes] = vt_brandes(dmax_brandes)
        #     self.vt_edges[self.d_edges<dmin_brandes] = vt_brandes(dmin_brandes)
            
            
        # else:
        
        # Set up particle properties for mass grid
        # Fall speed (m/s)
        
        #self.vt = np.clip(-0.1021 + 4.932*self.d-0.9551*self.d**2+0.07934*self.d**3-0.002362*self.d**4,0.01,10.)
        #self.vt_edges = np.clip(-0.1021 + 4.932*self.d_edges-0.9551*self.d_edges**2+0.07934*self.d_edges**3-0.002362*self.d_edges**4,0.01,10.)

        if isinstance(vt_func,types.LambdaType):
            self.vt = vt_func(self.d)
            self.vt_edges = vt_func(self.d_edges)
        else:
            self.vt = self.av*self.d**self.bv
            self.vt_edges = self.av*self.d_edges**self.bv
        
        self.vt[self.vt>10.] = 10.
        self.vt_edges[self.vt_edges>10.]=10.
        
        self.vt1 = self.vt_edges[:-1].copy() 
        self.vt2 = self.vt_edges[1:].copy()

        # ORIGINAL Atlas power-law for RAIN
        # self.vt = self.av*self.d**self.bv
        # self.vt_edges = self.av*self.d_edges**self.bv
        
        # self.vt[self.vt>10.] = 10.
        # self.vt_edges[self.vt_edges>10.]=10.
        
        # self.vt1 = self.vt_edges[:-1].copy() 
        # self.vt2 = self.vt_edges[1:].copy()

        # Midpoint Area (mm^2)
        # !!! Note, testing here
        #self.A = 0.25*np.pi*self.dmax**2.
        self.A = 0.25*np.pi*self.d**2.
        # Edge Area (mm^2)
        self.A_edges = 0.25*np.pi*self.d_edges**2.
        self.A1 = self.A_edges[:-1].copy() 
        self.A2 = self.A_edges[1:].copy()
            
        self.Mbins = np.zeros_like(self.xbins).astype(np.float64)
        self.Nbins = np.zeros_like(self.xbins).astype(np.float64)
        
        
    def bin_gamma_dist(self,Nt0=1.,Mt0=1.,mbar=None,mu0=3,Dm0=2,normalize=False):
        
        '''
        Description: Set up bins and integrals if using only mass moment
        '''
        nu = mu0+1
        #kernel = self.kernel
        
        self.Nt0 = Nt0 
        self.mu0 = mu0 
        self.Dm0 = Dm0
            
        self.Dn = Dm0/(mu0+4.)
        
        self.mn = self.am*self.Dn**self.bm

        # Number distribution function in terms of mass (n(x))
        
        if normalize: # Normalize mass distribution similar to Scott (1967) and Long (1974)
            #self.nedges = (nu)**(nu)/scip.gamma(nu)*self.xedges**(nu-1.)*np.exp(-nu*self.xedges)
        
            #self.nbins = (nu)**(nu)/scip.gamma(nu)*self.xbins**(nu-1.)*np.exp(-nu*self.xbins)
            
            if mbar is None:
                mbar = Mt0/Nt0
            
            self.nedges = (self.Nt0/mbar)*((nu**nu)/scip.gamma(nu))*(self.xedges/mbar)**(nu-1.)*np.exp(-nu*self.xedges/mbar)
        
            self.nbins = (self.Nt0/mbar)*((nu**nu)/scip.gamma(nu))*(self.xbins/mbar)**(nu-1.)*np.exp(-nu*self.xbins/mbar)
            
            
        else:
           self.nedges = (self.Nt0/self.bm)*(1./scip.gamma(self.mu0+1.))*\
               (1./self.mn)*(self.xedges/self.mn)**((nu/self.bm)-1.)*np.exp(-(self.xedges/self.mn)**(1./self.bm))
               
           self.nbins = (self.Nt0/self.bm)*(1./scip.gamma(self.mu0+1.))*\
               (1./self.mn)*(self.xbins/self.mn)**((nu/self.bm)-1.)*np.exp(-(self.xbins/self.mn)**(1./self.bm))
            
        self.Nbins = 0.5*(self.nedges[:-1]+self.nedges[1:])*(self.x2-self.x1)
        self.Mbins = (1./6.)*(self.nedges[:-1]*(2.*self.x1+self.x2)+self.nedges[1:]*(self.x1+2.*self.x2))*(self.x2-self.x1)


    def bin_analytical_dist(self, func_nD, var='size'):
        """
        Populates bins using user-defined analytical functions.
        func_N: A lambda returning number concentration (dN/dD or dN/dm)
        func_M: (Optional) A lambda returning mass concentration
        var: 'diameter' or 'mass' indicating what the lambda expects.
        """
        from scipy.integrate import quad
        
        self.Mbins = np.zeros(self.bins)
        self.Nbins = np.zeros(self.bins)
        
        # Choose the grid the function expects
        x1 = self.d1 if var == 'size' else self.xi1
        x2 = self.d2 if var == 'size' else self.xi2
        
        if var == 'size':
            bin_mass = lambda x: self.am*x**self.bm
            
        else:
            bin_mass = lambda x: x
                  
        func_nD_scaled = lambda x: bin_mass(x)*func_nD(x)
        
        for k in range(self.bins):
            
            self.Mbins[k], _ = quad(func_nD_scaled, x1[k], x2[k])
            self.Nbins[k], _ = quad(func_nD, x1[k], x2[k])

    def bin_empirical_dist(self, user_edges, user_nD):
        """
        Conservatively regrids external PSD data (density n(D)) to the model's grid.
        
        Parameters:
        -----------
        user_edges : array
            Array of length (K+1) defining the user's bin boundaries (diameter).
            *Must be in the same units as the model's internal self.d1 / self.d2*
        user_nD : array
            Array of length K defining the number density n(D).
            Assumed to be piecewise-constant across the bin.
        """
        from scipy.interpolate import interp1d
        
        # 1. Convert density n(D) to exact discrete Number (N) and Mass (M) per user bin
        D_left = user_edges[:-1]
        D_right = user_edges[1:]
        
        # Discrete Number: Integral of a constant n(D) is just n(D) * delta_D
        user_N = user_nD * (D_right - D_left)
        
        # Discrete Mass: Exact integral of n(D) * a * D^b dD
        # This prevents the underestimation that occurs if you just use the bin midpoint!
        b1 = self.bm + 1.0
        user_M = user_nD * self.am * (D_right**b1 - D_left**b1) / b1
        
        # 2. Create the Cumulative Distributions (CDF starts at 0 at the first edge)
        cdf_N = np.insert(np.cumsum(user_N), 0, 0.0)
        cdf_M = np.insert(np.cumsum(user_M), 0, 0.0)
        
        # 3. Create linear interpolators for the CDFs.
        # Everything outside the user's grid bounds gets clamped to 0 or max total.
        interp_N = interp1d(user_edges, cdf_N, kind='linear', 
                            bounds_error=False, fill_value=(0, cdf_N[-1]))
        
        interp_M = interp1d(user_edges, cdf_M, kind='linear', 
                            bounds_error=False, fill_value=(0, cdf_M[-1]))
                            
        # 4. Map to model grid by differencing the CDF at the model's diameter boundaries
        self.Nbins = interp_N(self.d2) - interp_N(self.d1)
        self.Mbins = interp_M(self.d2) - interp_M(self.d1)
        
        # Prevent any tiny negative interpolation artifacts
        self.Nbins[self.Nbins < 0] = 0.0 
        self.Mbins[self.Mbins < 0] = 0.0


    def bin_empirical_counts(self, user_edges, user_N, user_M=None):
        """
        Conservatively regrids external binned data to the model's native grid.
        user_edges: Array of length (N+1) defining the user's bin boundaries.
        user_N: Array of length N defining total number in each bin.
        """
        from scipy.interpolate import interp1d
        
        # 1. Create the Cumulative Number Distribution
        # Insert a 0 at the beginning so CDF starts at 0
        cdf_N = np.insert(np.cumsum(user_N), 0, 0.0)
        
        # 2. Create an interpolator.
        # Extrapolation is flat (0 on the left, max total number on the right)
        interp_N = interp1d(user_edges, cdf_N, kind='linear', 
                            bounds_error=False, fill_value=(0, cdf_N[-1]))
        
        # 3. Map to model grid by differencing the CDF at the model boundaries
        self.Nbins = interp_N(self.d2) - interp_N(self.d1)
        
        # Prevent any tiny negative interpolation artifacts
        self.Nbins[self.Nbins < 0] = 0.0 
        
        # 4. Handle Mass
        if user_M is not None:
            cdf_M = np.insert(np.cumsum(user_M), 0, 0.0)
            interp_M = interp1d(user_edges, cdf_M, kind='linear', 
                                bounds_error=False, fill_value=(0, cdf_M[-1]))
            self.Mbins = interp_M(self.d2) - interp_M(self.d1)
            self.Mbins[self.Mbins < 0] = 0.0
        else:
            self.Mbins = self.Nbins * self.mass


    def bin_direct_dist(self, edges, Nbins, Mbins=None, var='size'):
        """
        Conservatively regrids discrete user arrays onto the model's native grid.
        
        Parameters:
        -----------
        edges : array
            Array of length (K+1) defining the user's bin boundaries.
        Nbins : array
            Array of length K defining the absolute total number in each user bin.
        Mbins : array (Optional)
            Array of length K defining the absolute total mass in each user bin.
        var : str
            'diameter' or 'mass', indicating what physical property `edges` represents.
        """
        from scipy.interpolate import interp1d
        
        edges = np.array(edges, dtype=np.float64)
        user_N = np.array(Nbins, dtype=np.float64)
        
        # 1. Create the Cumulative Distribution (Starts at 0)
        cdf_N = np.insert(np.cumsum(user_N), 0, 0.0)
        
        # 2. Create the linear interpolator
        # Flat extrapolation ensures no particles are "invented" outside the user's bounds
        interp_N = interp1d(edges, cdf_N, kind='linear', 
                            bounds_error=False, fill_value=(0, cdf_N[-1]))
        
        # 3. Select the correct model boundaries to map onto
        if var == 'size':
            model_left = self.d1
            model_right = self.d2
        elif var == 'mass':
            model_left = self.xi1
            model_right = self.xi2
        else:
            raise ValueError("var must be 'diameter' or 'mass'")
            
        # 4. Differencing the CDF onto the model grid
        self.Nbins = interp_N(model_right) - interp_N(model_left)
        self.Nbins[self.Nbins < 0] = 0.0
        
        # 5. Handle Mass identically if provided
        if Mbins is not None:
            user_M = np.array(Mbins, dtype=np.float64)
            cdf_M = np.insert(np.cumsum(user_M), 0, 0.0)
            interp_M = interp1d(edges, cdf_M, kind='linear', 
                                bounds_error=False, fill_value=(0, cdf_M[-1]))
                                
            self.Mbins = interp_M(model_right) - interp_M(model_left)
            self.Mbins[self.Mbins < 0] = 0.0
        else:
            # Fallback: assume mass centers align with the model's native grid
            self.Mbins = self.Nbins * self.mass


    def moments(self,r):  # Units are g^n
        # Integrate to find arbitrary moments of subgrid distribution Mn = Int x^n *[n(x)=ak*x+ck]*dx
        return self.aki*Pn(r+1,self.x1,self.x2)+self.cki*Pn(r,self.x1,self.x2)

    # Function for diagnosing linear distribution function following Wang et al. (2008)
    # NOTE: Need to clip xm to left/right bin boundaries
    def diagnose_1mom(self):
         
        self.Nbins = self.Mbins/self.xbins
        
        self.n1 = self.n2 = self.cki = self.Mbins/self.dxi 
                        
        # if self.radar:
        #     self.radar_bins() 
                  
        # Diagnose mass- number-weighted bin fallspeeds and bin residence times
        self.vtm = self.vt.copy()
        self.vtn = self.vt.copy()
        
        #!!! TESTING
        #self.Mfbins =self.av*(self.am)**(-self.bv/self.bm)*self.moments((self.bm+self.bv)/self.bm)
        #self.Nfbins =self.av*(self.am)**(-self.bv/self.bm)*self.moments((self.bv)/self.bm)
        
        self.Mfbins = self.vt*self.Mbins
        self.Nfbins = self.vt*self.Nbins
        
        vt_fill = (self.Mbins>0.) & (self.Nbins>0.) & (self.vtm>0.) & (self.vtn>0.)\
                  & (self.Mfbins>0.) & (self.Nfbins>0.)
        self.vtm[vt_fill] = self.Mfbins[vt_fill]/self.Mbins[vt_fill]
        self.vtn[vt_fill] = self.Nfbins[vt_fill]/self.Nbins[vt_fill] 

        self.vtm[self.vtm>10.] = 10. 
        self.vtn[self.vtn>10.] = 10.

   
    def diagnose(self):
        
        # Google Gemini enhanced version. Needed to bug fix. (see original above)
        eps = 1e-32 # Protection against div by zero
        
        # 1. Clean up input noise
        self.Mbins[self.Mbins < eps] = 0.
        self.Nbins[self.Nbins < eps] = 0.
        
        dx = self.xi2 - self.xi1
        xm = np.zeros_like(self.Mbins)
        
        # 2. Safe mean calculation
        safe = self.Nbins > 0.
        xm[safe] = self.Mbins[safe] / self.Nbins[safe]
        
        # 3. Handle edge cases (xm must be within [xi1, xi2])
        xm = np.clip(xm, self.xi1 + eps, self.xi2 - eps)
    
        xm1 = xm / self.xi1
        cond_null = (self.Mbins <= 0.) | (self.Nbins <= 0.)
        
        # Boundary logic for Scenarios A, B, C
        bound_low = (2. + self.rhobins) / 3.
        bound_high = (1. + 2. * self.rhobins) / 3.
        
        cond_a = (bound_low <= xm1) & (xm1 <= bound_high) & (~cond_null)
        cond_b = (1. <= xm1) & (xm1 < bound_low) & (~cond_null)
        cond_c = (bound_high < xm1) & (xm1 <= self.rhobins) & (~cond_null)
    
        x1i, x2i = self.xi1.copy(), self.xi2.copy()
        n1i, n2i = np.zeros_like(xm), np.zeros_like(xm)
    
        # Scenario A: Spans full bin
        n1i[cond_a] = 2*(self.Nbins[cond_a]*(self.xi1[cond_a] + 2.*self.xi2[cond_a]) - 3.*self.Mbins[cond_a]) / (dx[cond_a]**2)
        n2i[cond_a] = 2*(-self.Nbins[cond_a]*(2.*self.xi1[cond_a] + self.xi2[cond_a]) + 3.*self.Mbins[cond_a]) / (dx[cond_a]**2)
    
        # Scenario B: Truncated at the right
        x2i[cond_b] = self.xi1[cond_b] + 3.*(xm[cond_b] - self.xi1[cond_b])
        n1i[cond_b] = 2.*self.Nbins[cond_b] / (3.*np.maximum(eps, xm[cond_b] - self.xi1[cond_b]))
        n2i[cond_b] = 0.
    
        # Scenario C: Truncated at the left
        x1i[cond_c] = self.xi2[cond_c] - 3.*(self.xi2[cond_c] - xm[cond_c])
        n1i[cond_c] = 0.
        n2i[cond_c] = 2.*self.Nbins[cond_c] / (3.*np.maximum(eps, self.xi2[cond_c] - xm[cond_c]))
    
        # 4. Correct local slope/intercept
        # Use local_dx because scenarios B/C compress the distribution
        local_dx = x2i - x1i
        self.aki = np.zeros_like(xm)
        self.cki = np.zeros_like(xm)
        
        valid = ~cond_null
        self.aki[valid] = (n2i[valid] - n1i[valid]) / np.maximum(eps, local_dx[valid])
        self.cki[valid] = (n1i[valid]*x2i[valid] - x1i[valid]*n2i[valid]) / np.maximum(eps, local_dx[valid])
        
        # Store state
        self.x1, self.x2, self.n1, self.n2 = x1i, x2i, n1i, n2i     
        
        # if self.radar:
        #     self.radar_bins() 
                  
        # Diagnose mass- number-weighted bin fallspeeds and bin residence times
        self.vtm = self.vt.copy()
        self.vtn = self.vt.copy()
        
        # !!! Testing
        #self.Mfbins =self.av*(self.am)**(-self.bv/self.bm)*self.moments((self.bm+self.bv)/self.bm)
        #self.Nfbins =self.av*(self.am)**(-self.bv/self.bm)*self.moments((self.bv)/self.bm)
        
        self.Mfbins = self.vt*self.Mbins
        self.Nfbins = self.vt*self.Nbins
        
        vt_fill = (self.Mbins>0.) & (self.Nbins>0.) & (self.vtm>0.) & (self.vtn>0.)\
                  & (self.Mfbins>0.) & (self.Nfbins>0.)
        self.vtm[vt_fill] = self.Mfbins[vt_fill]/self.Mbins[vt_fill]
        self.vtn[vt_fill] = self.Nfbins[vt_fill]/self.Nbins[vt_fill] 

        self.vtm[self.vtm>10.] = 10. 
        self.vtn[self.vtn>10.] = 10.
           
    
    def check_moments(self):
        
        Ncheck = 0.5*(self.x2-self.x1)*(self.n1+self.n2) 
        Mcheck = (1./6.)*(self.x2-self.x1)*\
              (self.n1*(2.*self.x1+self.x2)+self.n2*(self.x1+2.*self.x2))
              
        print('Ncheck = {} | Nactual = {}'.format(Ncheck.sum(),self.Nbins.sum()))
        print('Mcheck = {} | Mactual = {}'.format(Mcheck.sum(),self.Mbins.sum()))
        
        print('Ndiff = {}'.format(Ncheck-self.Nbins))
        print('Mdiff = {}'.format(Mcheck-self.Mbins))
       
        
    def radar_bins(self):
        
        ang1 = self.angs[0]
        ang2 = self.angs[1]
        ang3 = self.angs[2]
        ang4 = self.angs[3]
        ang5 = self.angs[4]
        #ang6 = angs[5]
        ang7 = self.angs[6]
        
        fhh_180_1 = fhh_0_1 = self.fscatt_pre1* (1 / (self.lb1 + self.eps1_factor))  
        fvv_180_1 = fvv_0_1 = self.fscatt_pre1* (1 / (self.la1 + self.eps1_factor))  
        
        fhh_180_2 = fhh_0_2 = self.fscatt_pre2* (1 / (self.lb2 + self.eps2_factor))  
        fvv_180_2 = fvv_0_2 = self.fscatt_pre2* (1 / (self.la2 + self.eps2_factor))
        
        
        fZh1 = self.cz * ((np.abs(fhh_180_1))**2 -
                   2.0 * ang2 * np.real(np.conj(fhh_180_1) * (fhh_180_1 - fvv_180_1)) +
                   ang4 * (np.abs(fhh_180_1 - fvv_180_1))**2)
        
        fZv1 = self.cz * ((np.abs(fhh_180_1))**2 -
                   2.0 * ang1 * np.real(np.conj(fhh_180_1) * (fhh_180_1 - fvv_180_1)) +
                   ang3 * (np.abs(fhh_180_1 - fvv_180_1))**2)
        
        fKdp1 = self.ckdp * ang7 * np.real(fhh_0_1 - fvv_0_1)
        
        fZhhvv1 = self.cz * ((np.abs(fhh_180_1))**2 +
                      ang5 * (np.abs(fhh_180_1 - fvv_180_1))**2 -
                      ang1 * (np.conj(fhh_180_1) * (fhh_180_1 - fvv_180_1)) -
                      ang2 * fhh_180_1 * np.conj(fhh_180_1 - fvv_180_1))
        
        fZh2 = self.cz * ((np.abs(fhh_180_2))**2 -
                   2.0 * ang2 * np.real(np.conj(fhh_180_2) * (fhh_180_2 - fvv_180_2)) +
                   ang4 * (np.abs(fhh_180_2 - fvv_180_2))**2)
        
        fZv2 = self.cz * ((np.abs(fhh_180_2))**2 -
                   2.0 * ang1 * np.real(np.conj(fhh_180_2) * (fhh_180_2 - fvv_180_2)) +
                   ang3 * (np.abs(fhh_180_2 - fvv_180_2))**2)
        
        fKdp2 = self.ckdp * ang7 * np.real(fhh_0_2 - fvv_0_2)
        
        fZhhvv2 = self.cz * ((np.abs(fhh_180_2))**2 +
                      ang5 * (np.abs(fhh_180_2 - fvv_180_2))**2 -
                      ang1 * (np.conj(fhh_180_2) * (fhh_180_2 - fvv_180_2)) -
                      ang2 * fhh_180_2 * np.conj(fhh_180_2 - fvv_180_2))
        
        # Find slopes/intercepts for linear interpolation formulas.
        ak_zh = (fZh2-fZh1)/self.dxbins
        ck_zh = fZh1-ak_zh*self.xi1
        
        ak_zv = (fZv2-fZv1)/self.dxbins
        ck_zv = fZv1-ak_zv*self.xi1
        
        ak_kdp = (fKdp2-fKdp1)/self.dxbins
        ck_kdp = fKdp1-ak_kdp*self.xi1
        
        ak_zhhvv = (fZhhvv2-fZhhvv1)/self.dxbins
        ck_zhhvv = fZhhvv1-ak_zhhvv*self.xi1
        
        # Linearly interpolate scattering amplitudes across each bin
        # and then integrate each term to find radar values
        # Integrations are: 1000 * int g(x) * n(x) dx = 1000 * int (ak_v * x + ck_v) * (aki*x +cki) 
        self.zh = 1000.*(ak_zh*self.moments(1.)+ck_zh*self.moments(0.))
        self.zv = 1000.*(ak_zv*self.moments(1.)+ck_zv*self.moments(0.))
        self.kdp = 1000.*(ak_kdp*self.moments(1.)+ck_kdp*self.moments(0.))
        self.zhhvv = 1000.*(ak_zhhvv*self.moments(1.)+ck_zhhvv*self.moments(0.))
        
        self.Zh = -35.*np.ones_like(self.zh)
        self.Zv = -35.*np.ones_like(self.zv)
        self.Zdr = np.zeros_like(self.zh)
        self.Kdp = np.zeros_like(self.kdp)
        
        rad_fill = (self.zh>0.)&(self.zv>0.)
        
        self.Zh[rad_fill] = 10.*np.log10(self.zh[rad_fill])
        self.Zdr[rad_fill] = 10.*np.log10(self.zh[rad_fill]/self.zv[rad_fill])
        
        zh_sum = np.nansum(self.zh)
        zv_sum = np.nansum(self.zv)
        zhhvv_sum = np.abs(np.nansum(self.zhhvv))
        
        if (zh_sum>0.) & (zv_sum>0.):
            self.ZH = 10.*np.log10(zh_sum)
            self.ZDR = 10.*np.log10(zh_sum/zv_sum)
            self.KDP = np.nansum(self.kdp)
            rhohv_denom = np.sqrt(zh_sum*zv_sum)
            if rhohv_denom>0.: # Apparently for very small zh_sum and zv_sum the denominator can still be zero.
                self.rhohv = zhhvv_sum/np.sqrt(zh_sum*zv_sum)
            else:
                self.rhohv = 1.0
            
        else:
            self.ZH = -35. 
            self.ZDR = 0. 
            self.KDP = 0.
            self.rhohv = 1.0

    def plot(self,log_switch=True,x_axis='mass',ax=None):
        '''
        Plots number and mass distributions for distribution object.

        Parameters
        ----------
        log_switch : Bool, optional
            Whether distribution scaling is log or linear. The default is True.
        x_axis : string, optional
            Whether x axis is 'mass' or 'size'. The default is 'mass'.
        ax : matplotlib.pyplot axes() object, optional
            Plots number/mass distributions in existing pyplot axes. The default is None.

        Returns
        -------
        fig : matplotlib figure object
        ax : matplotlib axes object


        '''

        if ax is None:
            ax_orig = True 
        else:
            ax_orig = False

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize=16) 
        plt.rc('ytick', labelsize=16)         

        mbins = self.xbins
        xp1 = self.x1
        xp2 = self.x2
        ap = self.aki
        cp = self.cki
        
        bm = self.bm
        am = self.am

        if x_axis=='size':
            prefactor = bm*np.log(10)
            xbins = (mbins/am)**(1./bm)
            
            ylabel_num = r'dN/dlog(D)'
            ylabel_mass = r'dM/dlog(D)'
            
            xlabel = r'log(D) [log(mm)]'
            
        elif x_axis=='mass':
            prefactor = np.log(10)
            xbins = mbins
            
            ylabel_num = r'dN/dlog(m)'
            ylabel_mass = r'dM/dlog(m)'
            
            xlabel = r'log(m) [log(g)]'
                 
        n_init = prefactor*np.heaviside(mbins-xp1,1)*np.heaviside(xp2-mbins,1)*(ap*mbins+cp)

        if ax is None:
            fig, ax = plt.subplots(2,1,figsize=((8,10)),sharex=True)
        
        # Plot m*n(m) for number (N=int m*n(m)*dln(m)) | g_n(ln(r)) = bm*m*n(m), N = int g_n(ln(r))*dln(r)
        # Plot m^2*n(m) for mass (M=int m^2*n(m)*dln(m)) | g_m(ln(r)) = bm*m^2*n(m), M = int g_m(ln(r))*dln(r) 
        
        # Initial
        ax[0].plot(np.log10(xbins),mbins*n_init,'k')
        ax[1].plot(np.log10(xbins),1000.*mbins**2*n_init,'k')
        
        ax[0].set_ylabel(ylabel_num)
        ax[1].set_ylabel(ylabel_mass)
        ax[1].set_xlabel(xlabel)
        
        #print('Initial Number = {:.2f} #/L'.format(np.nansum(mbins*n_init*(np.log10(medges[1:])-np.log10(medges[:-1])))))
        #print('Initial Mass = {:.2f} g/cm^3'.format(np.nansum(1000.*mbins**2*n_init*(np.log10(medges[1:])-np.log10(medges[:-1])))))
        
        #print('number test size=',np.nansum(mbins*n_init*(np.log10(dedges[1:])-np.log10(dedges[:-1]))))
        #print('mass test size=',np.nansum(1000.*mbins**2*n_init*(np.log10(dedges[1:])-np.log10(dedges[:-1]))))
        
 
        if ax_orig:
            return fig, ax


# Function for diagnosing linear distribution function following Wang et al. (2008)
# NOTE: Need to clip xm to left/right bin boundaries

# Mbins = (dnum,Hlen,bins,Tout)
def update_1mom(Mbins,dxi):
    
   # return Mbins/dxi[None,None,:]

    return Mbins/dxi

    

              
    
    #!!! TESTING
    #self.Mfbins =self.av*(self.am)**(-self.bv/self.bm)*self.moments((self.bm+self.bv)/self.bm)
    #self.Nfbins =self.av*(self.am)**(-self.bv/self.bm)*self.moments((self.bv)/self.bm)
    
    #self.Mfbins = self.vt*self.Mbins
    #self.Nfbins = self.vt*self.Nbins
    


   
def update_2mom(Mbins,Nbins,rhobins,bound_low,bound_high,dx,xi1,xi2):
    # Google Gemini enhanced version. Needed to bug fix. (see original above)
    
    eps = 1e-32
    
    # 1. Clean noise across the 3D block
    Mbins[Mbins < eps] = 0.
    Nbins[Nbins < eps] = 0.
    
    # 2. Vectorized Mean mass calculation
    xm = np.zeros_like(Mbins)
    safe = Nbins > 0.
    np.divide(Mbins, Nbins, out=xm, where=safe)
    
    # 3. Handle Scenario Boundaries
    xm = np.clip(xm, xi1 + eps, xi2 - eps)
    xm1 = xm / xi1
    
    # Scenario Conditions (masks of shape (dnum, Hlen, bins))
    cond_null = (Mbins <= 0.) | (Nbins <= 0.)
    cond_a = (bound_low <= xm1) & (xm1 <= bound_high) & (~cond_null)
    cond_b = (1. <= xm1) & (xm1 < bound_low) & (~cond_null)
    cond_c = (bound_high < xm1) & (xm1 <= rhobins) & (~cond_null)
    
    # Output arrays
    x1i, x2i = xi1.copy(), xi2.copy()
    n1i, n2i = np.zeros_like(xm), np.zeros_like(xm)
    
    # Scenario A Logic (Fully Vectorized)
    n1i[cond_a] = 2 * (Nbins[cond_a] * (xi1[cond_a] + 2.*xi2[cond_a]) - 3.*Mbins[cond_a]) / (dx[cond_a]**2)
    n2i[cond_a] = 2 * (-Nbins[cond_a] * (2.*xi1[cond_a] + xi2[cond_a]) + 3.*Mbins[cond_a]) / (dx[cond_a]**2)
    
    # Scenario B Logic
    x2i[cond_b] = xi1[cond_b] + 3. * (xm[cond_b] - xi1[cond_b])
    n1i[cond_b] = 2. * Nbins[cond_b] / (3. * np.maximum(eps, xm[cond_b] - xi1[cond_b]))
    n2i[cond_b] = 0.
    
    # Scenario C Logic
    x1i[cond_c] = xi2[cond_c] - 3. * (xi2[cond_c] - xm[cond_c])
    n2i[cond_c] = 2. * Nbins[cond_c] / (3. * np.maximum(eps, xi2[cond_c] - xm[cond_c]))
    n1i[cond_c] = 0.
    
    valid = ~cond_null
    
    aki = np.zeros_like(xm)
    cki = np.zeros_like(xm)
    
    # 4. Final Coefficients
    local_dx = np.maximum(eps, x2i-x1i)

    aki[valid] = (n2i[valid]-n1i[valid])/local_dx[valid]
    cki[valid] = (n1i[valid]*x2i[valid]-x1i[valid]*n2i[valid])/local_dx[valid]
    
    cond = 1*cond_a 
    cond[cond_b] = 2 
    cond[cond_c] = 3
    
    return aki, cki, x1i, x2i
 

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


def extended_brandes(d):
    """
    Modified Brandes AR fit function:
    - Unity (1.0) at d=0
    - Matches original poly between ~0.3743 and 10
    - Levels off to 0.4 for d > 10
    """
    d = np.atleast_1d(d)
    res = np.zeros_like(d)
    
    # Constants
    d_crit = 0.37426095
    p_max = 0.99966286
    v_10 = 0.4131  # P(10)
    #s_10 = -0.1096 # P'(10)
    L = 0.4        # Asymptote
    k = 8.3664     # Decay constant to match slope at d=10
    
    # 1. Start at Unity and blend to Max
    mask1 = d < d_crit
    res[mask1] = p_max + (1.0 - p_max) * (1.0 - np.sin((np.pi/2) * (d[mask1]/d_crit)))
    
    # 2. Original Polynomial region
    mask2 = (d >= d_crit) & (d <= 10)
    res[mask2] = (0.9951 + 0.0251*d[mask2] - 0.03644*d[mask2]**2 + 
                  0.005303*d[mask2]**3 - 0.0002492*d[mask2]**4)
    
    # 3. Level off to 0.4
    mask3 = d > 10
    res[mask3] = L + (v_10 - L) * np.exp(-k * (d[mask3] - 10))
    
    return res if res.size > 1 else res.item()



def rain_terminal_velocity(d_mm):
    """
    Calculates terminal velocity (m/s) from diameter (mm) using a Weibull-like fit to
    the Gunn & Kinzer (1949) measurements according to Equation 5 from Best (1950).
    """

    return 9.43*(1.-np.exp(-(0.565*d_mm)**(1.147))) # Equation 5 from Best (1950) https://doi.org/10.1002/qj.49707632905