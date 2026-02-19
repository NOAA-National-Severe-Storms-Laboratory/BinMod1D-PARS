# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 11:51:00 2025

@author: Edwin L. Dunnavan, RSII Cooperative Institute for Severe and High-Impact 
                                 Weather Research and Operations (CIWRO)

Description:
    
    This code can be used to generate model objects representing collision-coalescence 
    and collisional breakup based on the Wang et al. (2007) source-based solution
    to the Kinetic Collection Equation (KCE). This version, unlike Wang et al. (2007),
    analytically integrates all terms explicitly rather than using Gaussian-quadrature.
    Note, that if one uses enough quadrature points in Wang's approach then the integrals
    can be exact as well.
    
    Version 0.0: 
        
        ELD NOTE 01/09/2026: Try to maybe implement cupy replacement for numpy
        for gpu numpy array operations.
        
    LIST OF THINGS TO DO:
        
        SHORT TERM:
            1.) Complete Ikernel.interact_2mom_SS() and Ikernel.interact_1mom_SS() for
                utilizing parallel processing and using GPU for box and SS models.
            2.) Modify write_netcdf() method to only save necessary 1 moment variables (e.g., Mbins)
            3.) Create timing tests for box model, steady-state model, and full 1D model (see tests/ directory)
            4.) Incorporate stencils for different order advection schemes.
            5.) Incorporate output_freq for box model (and possibly something similar for SS model)
            6.) Fix issue when output_freq is incompatible with dt.
            7.) Fix Ipython_launcher bug when using Jupyter notebook (seems like the progress bar
                is messed up when it first launches but then works ok afterward.)
            8.) Fix PSD issue for multiple categories and am/bm are different.
            9.) Modify run_steady_state_1mom() and run_steady_state_2mom() to incorporate RK steps.
    
        MEDIUM TERM:
            1.) Incorporate more detailed CC/BC methods:
                a.) Phillips et al. 2015/2016 aggregation/breakup parameterizations
                b.) McFarquhar et al. rain breakup distributions
            2.) Add class/method that can be used to define limits of plots and/or
                compare different spectral_1d objects.
            
        LONG TERM: 
            1.) Add in additional microphysical processes (vapor deposition, riming, melting, etc.)
            2.) Utilize T-Matrix calculations (either explicitly or using some kind of NN)
    
"""

## Import stuff

from .distribution import dist,update_2mom
from .interaction import Interaction
from .bin_integrals import init_rk, Pn
from .analytical_solutions import Scott_dists, Feingold_dists
from .habits import habits, fragments
from .plotting_functions import get_cmap_vars
from .radar import spheroid_factors, angular_moments, dielectric_ice, dielectric_water

import numpy as np
import scipy.special as scip
import matplotlib.pyplot as plt

from matplotlib.colors import BoundaryNorm

import shutil

from datetime import datetime

from netCDF4 import Dataset

import sys

#from joblib import Parallel

if 'ipykernel_launcher.py' in sys.argv[0]:
    from tqdm.auto import tqdm
else:
    from tqdm import tqdm
       

# 1D Spectral Bin Model Class
class spectral_1d:
    
    def __init__(self,sbin=8,bins=140,dt=2,
                 tmax=800.,output_freq=1,dz=10.,ztop=0.,zbot=0.,D1=0.25,x0=0.01,Nt0=1.,Mt0=1.,Dm0=2.0,
                 mbar0=None,mu0=3.,gam_norm=False,Ecol=0.001,Es=1.0,Eb=0.,moments=2,dist_var='mass',
                 kernel='Golovin',frag_dist='exp',habit_list=['rain'],
                 ptype='rain',Tc=10.,boundary=None,dist_num=1,cc_dest=1,br_dest=1, 
                 radar=False,wavl=110.,rk_order=1,adv_order=1,gpu=False,load=None,progress=True):
        '''
        

        Parameters
        ----------
        sbin : int
            Spectral bin mass grid resolution. The default is 8.
        bins : int
            Number of bins used for each distribution. The default is 140.
        dt : float, optional
            Model time step in seconds. The default is 2.
        tmax : float, optional
            Maximum time in seconds the model is run. The default is 800.
        output_freq : int, optional
            Frequency in time with which the Full 1D column model is output. The default is 60..
        dz : float, optional
            Height grid spacing in meters. The default is 10..
        ztop : float, optional
            Top height of steady-state/1D model domain in meters. The default is 0..
        zbot : float, optional
            Bottom height of steady-state/1D model domain in meters. The default is 0..
        D1 : float, optional
            Minimum equivolume diameter bin size in mm when the 'dist_var' parameter is 'size'. The default is 0.25.
        x0 : float, optional
            Minimum bin mass in grams when the 'dist_var' parameter is 'mass'.. The default is 0.01.
        Nt0 : float, optional
            Initial gamma distribution full number concentration in 1/L. The default is 1.
        Mt0 : float, optional
            Initial gamma distribution full mass concentration in g/m^3. The default is 1.
        Dm0 : float, optional
            Initial gamma distribution full mean volume diameter in mm. The default is 2.0.
        mu0 : float, optional
            Initial gamma distribution shape parameter. The default is 3..
        gam_norm : bool, optional
            Specify whether initial gamma distribution is normalized by mass (True). The default is False.
        Ecol : float, optional
            Collision efficiency. The default is 0.001.
        Es : float, optional
            Sticking efficiency. The default is 1.0.
        Eb : float, optional
            Breakup efficiency. The default is 0..
        moments : int, optional
            Number of moments to use for spectral bin model (either 1 (mass) or 2 (mass and number)). The default is 2.
        dist_var : str, optional
            Whether to use mass or size to specify initial gamma distribution. The default is 'mass'.
        kernel : str, optional
            Type of collision kernel in collection_kernels.py to use for coalescence/breakup. The default is 'Golovin'.
        frag_dist : str, optional
            Type of fragment distribution. The default is 'exp'.
        habit_list : list, optional
            List of number of distribution (determined by len(habit_lists)) habits. The default is ['rain'].
        ptype : str, optional
            Whether particles are rain or snow. NOTE: currently not considering mixed phase or mixture of rain/show. The default is 'rain'.
        Tc : float, optional
            Layer-averaged temperature in degrees Celsius. Used in radar calculations. The default is 10..
        boundary : Nonetype or str, optional
            Upper boundary conditions for full 1D model: None   = fallout mode for 1D model
                                                        'fixed' = fixed upper boundary condition. 
                                                        The default is None.
        cc_dest : int, optional
            Distribution destination of coalesced particles. 
            Index (1 = first distribution) corresponds to the distribution in
            habit_list.
            The default is 1.
        br_dest : int, optional
            Distribution destination of breakup (fragment) particles. 
            Index (1 = first distribution) corresponds to the distribution in
            habit_list.
            DESCRIPTION. The default is 1.
        radar : bool, optional
            Whether to calculate radar variables for each bin. The default is False.
        wavl: float, optional
            Radar wavelength in mm.
        rk_order : int, optional
            Runge-Kutta time stepping order. The default is 1.
        adv_order : int, optional
            Upwind advection scheme order (CURRENTLY NOT IMPLMENTED YET). The default is 1.
        gpu : bool, optional
            Whether to use GPU (if available) (CURRENTLY NOT IMPLEMENTED YET). The default is False.
        load : str, optional
            If loading spectral_1d() object from netcdf file, this is the path string to the netcdf file.
            None means no loading.
            The default is None.

        '''   
        
        # If not loading netcdf file, then manually set up case, attributes, variables, etc.
        if load is None:
            self.setup_case(sbin=sbin,bins=bins,D1=D1,x0=x0,dt=dt,tmax=tmax,
                        output_freq=output_freq,dz=dz,ztop=ztop,zbot=zbot,Nt0=Nt0,Mt0=Mt0,Dm0=Dm0,mbar0=mbar0,mu0=mu0,gam_norm=gam_norm,Ecol=Ecol,
                        Es=Es,Eb=Eb,moments=moments,dist_var=dist_var,kernel=kernel,frag_dist=frag_dist,
                        habit_list=habit_list,ptype=ptype,Tc=Tc,radar=radar,wavl=wavl,boundary=boundary,
                        cc_dest=cc_dest,br_dest=br_dest,rk_order=rk_order,adv_order=adv_order,gpu=gpu,
                        progress=progress)
            
        else: # If load is specified then load netcdf attributes/variables as object attribute/variables
            
            with Dataset(load,'r',format='NETCDF4') as file_nc:
                
                # Read spectral_1d object attributes from netcdf file
                self.bins = file_nc.bins
                self.sbin = file_nc.sbin
                self.dnum = file_nc.dnum
                self.moments = file_nc.moments
                self.kernel = file_nc.kernel
                self.ptype  = file_nc.ptype
                self.Hlen = file_nc.Hlen
                self.Tlen = file_nc.Tlen
                self.output_freq = file_nc.output_freq
                self.Tout_len = file_nc.Tout_len
                self.dt = file_nc.dt 
                self.dz = file_nc.dz 
                self.rk_order = file_nc.rk_order
                self.adv_order = file_nc.adv_order
                self.int_type = file_nc.int_type
                self.radar =  bool(file_nc.radar)
                self.radar = file_nc.wavl
                self.br_dest = file_nc.indb
                self.cc_dest = file_nc.indc
                self.dist_var = file_nc.dist_var
                self.mu0 = file_nc.mu0
                self.Dm0 = file_nc.Dm0
                self.Nt0 = file_nc.Nt0
                self.Mt0 = file_nc.Mt0
                self.D1 = file_nc.D1
                self.lamf = file_nc.lamf
                self.x0 = file_nc.x0
                self.Tc = file_nc.Tc
                self.Ecol = file_nc.Ecol
                self.Es = file_nc.Es
                self.Eb = file_nc.Eb
                self.Eagg = file_nc.Eagg
                self.Ebr = file_nc.Ebr
                self.Ecb = file_nc.Ecb
                #self.n_jobs = file_nc.n_jobs
                self.gam_norm = bool(file_nc.gam_norm)
                if file_nc.boundary == 'None':
                    self.boundary = None
                else:
                    self.boundary = file_nc.boundary
                self.ztop = file_nc.ztop
                self.zbot = file_nc.zbot
                self.tmax = file_nc.tmax
                
                # if file_nc.parallel==0:
                #     self.parallel = False
                # else:
                #     self.parallel = True
                        
                self.rhobins = 2**(1./self.sbin) # scaling param for mass bins 
                # Boundary logic for Scenarios A, B, C
                self.bound_low = (2. + self.rhobins) / 3.
                self.bound_high = (1. + 2. * self.rhobins) / 3.
                
                self.frag_dist = file_nc.frag_dist
                
                dist_names = list(file_nc.groups.keys())
                
                self.habit_list = [file_nc.groups[dd].habit for dd in dist_names]
                self.habit_dict = [habits()[habit_list[dd]] for dd in range(self.dnum)]
                            
                if self.frag_dist is None:
                    frag_dict = fragments('exp')
                else:
                    frag_dict = fragments(self.frag_dist)
                                
                self.lamf = frag_dict['lamf']
                
                self.setup_case(sbin=self.sbin,
                                bins=self.bins,
                                D1=self.D1,
                                x0=self.x0,
                                Nt0=self.Nt0,
                                Mt0=self.Mt0,
                                mbar0=self.mbar0,
                                Dm0=self.Dm0,
                                mu0=self.mu0,
                                gam_norm=self.gam_norm,
                                dist_var=self.dist_var,
                                kernel=self.kernel,
                                Ecol=self.Ecol,
                                Es=self.Es,
                                Eb=self.Eb,
                                moments=self.moments,
                                ztop=self.ztop,
                                zbot=self.zbot, 
                                tmax=self.tmax,
                                output_freq=self.output_freq,
                                dt=self.dt,
                                dz=self.dz,
                                frag_dist=self.frag_dist,
                                habit_list=self.habit_list,
                                ptype=self.ptype,
                                Tc=self.Tc,
                                radar=self.radar,
                                wavl = self.wavl,
                                boundary=self.boundary,
                                cc_dest=self.cc_dest,
                                br_dest=self.br_dest,
                                rk_order=self.rk_order,
                                adv_order=self.adv_order,
                                gpu=False,
                                progress=True)     
                     
                # Stencils used for variable upwind advection (not currently implemented)
                self.stencils = {1: np.array([-1, 1]) / 1,
                                 2: np.array([1, -4, 3]) / 2,
                                 3: np.array([-2, 9, -18, 11]) / 6,
                                 4: np.array([3, -16, 36, -48, 25]) / 12}
                
                self.adv_base = self.stencils[adv_order]
                
                self.tout = file_nc.variables['tout'][:]
                self.z = file_nc.variables['z'][:]
                
                self.xi1 = file_nc.variables['xi1'][:]
                self.xi2 = file_nc.variables['xi2'][:]
                
                self.dxbins = self.xi2-self.xi1
                
                self.Mbins = file_nc.variables['Mbins'][:]
                
                if self.moments==2:
                    self.Nbins =  file_nc.variables['Nbins'][:]
                       
                self.diagnose_subgrid()
                self.calc_micro() 
                
                if self.radar:
                    self.calc_radar()
            
        # If running in parallel
        # if self.parallel:
        #     self._parallel_config = Parallel(n_jobs=self.n_jobs,verbose=0)
        #     self._context_stack = []
        # else:
        #     self.n_jobs=1
        #     self._parallel_config = None
            
        # self.pool = None



            
    def setup_case(self,sbin=4,bins=160,D1=0.001,x0=0.01,Nt0=1.,Mt0=1.,mbar0=None,Dm0=2.0,mu0=3,gam_norm=False,dist_var='mass',kernel='Golovin',Ecol=1.53,Es=0.001,Eb=0.,
                        moments=2,ztop=3000.0,zbot=0.,zout=None,tout=None,tmax=800.,output_freq=1,dt=10.,dz=10.,frag_dist='exp',habit_list=['rain'],ptype='rain',Tc=10.,
                        radar=False,wavl=110.,boundary=None,cc_dest=1,br_dest=1,rk_order=1,adv_order=1,gpu=False,progress=True):
        self.Tc = Tc
        self.radar = radar
        self.wavl = wavl
        self.sbin = sbin 
        self.bins = bins
        self.D1 = D1
        self.Nt0 = Nt0 
        self.Mt0 = Mt0
        self.mbar0 = mbar0
        self.Dm0 = Dm0 
        self.mu0 = mu0 
        self.x0 = x0
        self.gam_norm = gam_norm
        self.kernel = kernel
        self.Ecol = Ecol # Collision efficiency
        self.Es = Es # Sticking efficiency
        self.Eb = Eb
        self.Eagg = Ecol*Es # Collision+Coalescence efficiency
        self.Ebr = Ecol*Eb*(1-Es) # Total Breakup efficiency
        self.Ecb = self.Eagg+self.Ebr # # Total Collision/Breakup efficiency for loss term
        self.ztop = ztop 
        self.zbot = zbot 
        self.tmax = tmax
        self.output_freq = int(output_freq)
        self.dt = dt 
        self.dz = dz
        t = np.arange(0,self.tmax+self.dt,step=self.dt) # technically not needed anymore
        # If tmax = -1 then run model as steady-state in vertical

        # If zbot == ztop == const. then run as box model
        z = np.arange(self.ztop,self.zbot-self.dz,step=-self.dz)
        self.dist_var = dist_var
        self.ptype = ptype
        self.rk_order = rk_order
        self.adv_order = adv_order
        self.boundary = boundary
        self.gpu = gpu
        #self.parallel = parallel
        self.frag_dist = frag_dist
        
        # if n_jobs == -1:
        #     self.n_jobs = os.cpu_count()
        # else:
        #     self.n_jobs = n_jobs
        
        self.indc = cc_dest 
        self.indb = br_dest
        
        self.dnum = len(habit_list)
        
        self.moments = moments
        
        self.Tlen = len(t) 
        self.Hlen = len(z)
        
        self.progress = progress

        # If time array is fixed then run as steady state model
        if (self.Tlen==1) & (self.Hlen>1):
            self.int_type = 1         
            self.Hlen = 1 # Here swap Hlen and Tlen so that model is integrated in the same way as in the other two modes
            self.Tlen = len(z)
            self.dt = 1
            self.z = z
            self.t = self.z 
            self.tout =  self.z[::self.output_freq].copy()
            #self.tout = np.arange(0,self.tmax+self.output_freq,step=self.output_freq)
            self.Tout_len = len(self.tout)
            
        # If height array is fixed then run as box model   
        elif (self.Tlen>1) & (self.Hlen==1):
            self.int_type = 2
            self.Hlen = 1
            self.Tlen = len(t) 
            #self.dh = self.t
            self.z = z
            self.t = np.arange(0,self.tmax+self.dt,step=self.dt) # technically not needed anymore
            #self.tout = np.arange(0,self.tmax+self.output_freq,step=self.output_freq)
            self.tout = self.t[::self.output_freq].copy()
            self.Tout_len = len(self.tout)
            
        # If time array and height array are not fixed then run as full 1d model
        else: 
            self.int_type=0
            #self.t = np.arange(0,self.tmax+self.dt,step=self.dt) # technically not needed anymore
            self.t = t
            self.z = z
            #self.tout = np.arange(0,self.tmax+self.output_freq,step=self.output_freq)
            self.tout = self.t[::self.output_freq].copy()
            self.Tout_len = len(self.tout)
        
        self.save_mask = np.isin(self.t, self.tout)
        
        # Ensure that cc_dest and br_dest are valid     
        if (cc_dest<1) | (cc_dest>self.dnum):
            print('cc_dest needs to be between 1 and {}'.format(self.dnum))
            raise Exception()
        if (br_dest<1) | (br_dest>self.dnum):
            print('br_dest needs to be between 1 and {}'.format(self.dnum))
            raise Exception()
           
        # Initialize distribution objects
        
        if dist_var=='size':
            x0=None
        elif dist_var=='mass':
            D1=None
        
        # If dnum > habit list then just use first element for all habits
        if len(habit_list) < self.dnum:
            habit_list = [habit_list[0] for dd in range(self.dnum)]
        
        self.habit_list = habit_list
        
        self.habit_dict = [habits()[habit_list[dd]] for dd in range(self.dnum)]
        
        dists = np.empty((self.dnum,),dtype=object)
        
        self.d = np.zeros((self.dnum,self.bins)) 
        self.d1 = np.zeros((self.dnum,self.bins)) 
        self.d2 = np.zeros((self.dnum,self.bins)) 
        
        self.ar = np.zeros((self.dnum,self.bins)) 
        self.ar1 = np.zeros((self.dnum,self.bins)) 
        self.ar2 = np.zeros((self.dnum,self.bins)) 
        
        self.rho = np.zeros((self.dnum,self.bins)) 
        self.rho1 = np.zeros((self.dnum,self.bins)) 
        self.rho2 = np.zeros((self.dnum,self.bins)) 
        
        self.vt = np.zeros((self.dnum,self.bins)) 
        
        ## NEW
        for dd in range(self.dnum):
            
            if dd==0: # Original Bin distribution
                
                dists[dd] = dist(sbin=sbin,bins=bins,D1=D1,x0=x0,Nt0=Nt0,Mt0=Mt0,mbar=mbar0,mu0=mu0,Dm0=Dm0,
                               gam_init=True,gam_norm=gam_norm,dist_var=dist_var,kernel=kernel,
                               habit_dict=self.habit_dict[0],ptype=ptype,Tc=Tc,mom_num=moments)
                dist0 = dists[0]  
                
            else:
                # Coalesced or fragmented particles
                dists[dd] = dist(sbin=sbin,D1=D1,bins=bins,gam_init=False,gam_norm=gam_norm,dist_var=dist_var,
                             kernel=kernel,habit_dict=self.habit_dict[dd],ptype=ptype,x0=dist0.x0, 
                             Tc=Tc,mom_num=moments)
                    
            dists[dd].dh = self.dz/dists[dd].vt # Residence time for steady-state height calculations 
            dists[dd].dt = self.dt*np.ones_like(self.t) # Residence time for box model (i.e., equal to dt)

            self.d[dd,:]  = dists[dd].d
            self.d1[dd,:] = dists[dd].d1
            self.d2[dd,:] = dists[dd].d2
            
            self.ar[dd,:]  = dists[dd].ar
            self.ar1[dd,:] = dists[dd].ar1
            self.ar2[dd,:] = dists[dd].ar2
            
            self.rho[dd,:]  = dists[dd].rho
            self.rho1[dd,:] = dists[dd].rho1
            self.rho2[dd,:] = dists[dd].rho2
            
            self.vt[dd,:] = dists[dd].vt
            
        # Residence times 
        if self.int_type==1: # steady-state height
            self.dh = np.vstack([dist.dh for dist in dists])
        elif self.int_type==2: # box model
            self.dh = self.dt*np.ones((self.dnum,self.bins))
    
        self.dxbins = dist0.dxbins
        self.dxi = dist0.dxi 
        # self.xedges = dist0.xedges 
        self.xi1 = dist0.xi1 
        self.xi2 = dist0.xi2 
        # self.xbins = dist0.xbins 
        self.rhobins = dist0.rhobins 
        self.bound_low = (2. + self.rhobins) / 3.
        self.bound_high = (1. + 2. * self.rhobins) / 3.

        # # Set up 4D output arrays
        self.Mbins = np.zeros((self.dnum,self.Hlen,self.bins,self.Tout_len)) 
        self.Mbins[0,0,:,0] = dist0.Mbins.copy()
        
        if self.moments==2:
            self.Nbins = np.zeros_like(self.Mbins)
            self.Nbins[0,0,:,0] = dist0.Nbins.copy()          

        if frag_dist is None:
            frag_dict = fragments('exp')
        else:
            frag_dict = fragments(frag_dist)
                
        self.xbins = dist0.xbins.copy() 
        self.xedges = dist0.xedges.copy()
        
           
        # Initialize interaction kernel between each species
        # Interaction() takes a (Ndist x height) array of dist objects
        # and sets up arrays for calculating interaction (i.e., source) terms
        # in the stochastic collection/breakup equation for multiple categories
        # This is essentially the set of buffers that's updated every timestep.
        
        # ORIGINAL
        self.Ikernel = Interaction(dists,self.Hlen,cc_dest,br_dest,self.Eagg,self.Ecb,self.Ebr, 
                                   frag_dict,self.kernel,gpu=gpu,
                                   mom_num=self.moments)
        
        self.Ikernel.rhobins = self.rhobins
        self.Ikernel.bound_low = self.bound_low 
        self.Ikernel.bound_high = self.bound_high
        
        
        self.lamf = frag_dict['lamf']
        
        # Stencils used for variable upwind advection
        self.stencils = {1: np.array([-1, 1]) / 1,
                         2: np.array([1, -4, 3]) / 2,
                         3: np.array([-2, 9, -18, 11]) / 6,
                         4: np.array([3, -16, 36, -48, 25]) / 12}
        
        self.adv_base = self.stencils[adv_order]

        self.dists = dists # 3D array of distribution objects (dist_num x height x time)
            
        self.dist0 = dist0

        # Setup output text
        if self.gam_norm and (self.mbar0 is None):
            self.out_text = lambda M, Mf: 'Total Mass = {:.2f} g/m^3 | Total Mass Flux = {:.2f} g/(m^2*s)'.format(np.sum(M),np.sum(Mf))
        else:
            self.out_text = lambda M, Mf: 'Total Mass = {:.2f} g/m^3 | Total Mass Flux = {:.2f} g/(m^2*s)'.format(1000.*np.sum(M),1000.*np.sum(Mf))
 
        
    def setup_radar(self):
        # Radar stuff
        #self.wavl = 110.
        #self.sigma = 0.
        self.ew0 = complex(81.0, 23.2)  # Dielectric constant of water at 0C
        self.kw = (np.abs((self.ew0 - 1) / (self.ew0 + 2)))**2
        self.cz = (4.0 * self.wavl**4)/(np.pi**4 * self.kw)
        self.ckdp = (0.18 / np.pi) * self.wavl
        self.rhoi = 0.92

        self.sigma = np.array([self.habit_dict[dd]['sig'] for dd in range(self.dnum)]) 
        
        self.angs = np.zeros((7,self.dnum,self.bins))
               
        for dd in range(self.dnum):
            self.angs[:,dd,:] = np.tile(angular_moments(self.sigma[dd]).T,(self.bins,1)).T
        
        # Calculate scattering amplitudes and whatnot
        if self.ptype=='rain':
            
            self.eps1 = dielectric_water(self.Tc+273.15,self.ew0)
            self.eps2 = self.eps1
             
        elif self.ptype=='snow':
            
            epi = dielectric_ice(self.wavl,self.Tc+273.15)
            
            Ki = (epi-1.)/(epi+2.)
            
            self.eps1 = (1+2*(self.rho1/self.rhoi)*Ki)/(1-(self.rho1/self.rhoi)*Ki)
            self.eps2 = (1+2*(self.rho2/self.rhoi)*Ki)/(1-(self.rho2/self.rhoi)*Ki)
            
        self.la1, self.lb1 = spheroid_factors(self.ar1)
        self.la2, self.lb2 = spheroid_factors(self.ar2)
        
        self.fscatt_pre1 = ((np.pi**2 * (self.d1)**3) / (6 * self.wavl**2))
        self.fscatt_pre2 = ((np.pi**2 * (self.d2)**3) / (6 * self.wavl**2))                   
        
        self.eps1_factor = (1 / (self.eps1 - 1))
        self.eps2_factor = (1 / (self.eps2 -1))
        
    def calc_micro(self):
        ''' 
        Calculate microphysical variables
        '''
        am = np.zeros((self.dnum,self.bins))
        bm = np.zeros((self.dnum,self.bins))
        
        M3 = np.zeros((self.dnum,self.Hlen,self.Tout_len))
        M4 = np.zeros((self.dnum,self.Hlen,self.Tout_len))

        for dd in range(self.dnum):
            am = self.habit_dict[dd]['am']
            bm = self.habit_dict[dd]['bm']
            
            M3[dd,:,:] = (am**(-3./bm)*(self.calc_moments(3./bm,self.moments)).sum(axis=2)[dd,:,:])
            M4[dd,:,:] = (am**(-4./bm)*(self.calc_moments(4./bm,self.moments)).sum(axis=2)[dd,:,:])
        
        M3tot = M3.sum(axis=0)
        M4tot = M4.sum(axis=0)
        
        self.Dm = np.full_like(M3,np.nan) 
        self.Dm[M3>0.] = M4[M3>0.]/M3[M3>0.]
        
        self.Dmtot = np.full_like(M3tot,np.nan)
        self.Dmtot[M3tot>0.] = M4tot[M3tot>0.]/M3tot[M3tot>0.]
        
        self.N = self.Nbins.sum(axis=2)
        #self.M = self.Mbins.sum(axis=2)
        #self.Rm = 3.6*(self.Mbins*self.vt[:,None,:,None]).sum(axis=2)
        
        self.M = 1000.*self.Mbins.sum(axis=2)
        self.Rm = 1000.*3.6*(self.Mbins*self.vt[:,None,:,None]).sum(axis=2)
        
        #if not self.gam_norm:
        #    self.M *= 1000.
        #    self.Rm *= 1000.
        
        self.Ntot = self.N.sum(axis=0)
        self.Mtot = self.M.sum(axis=0)
        self.Rmtot = self.Rm.sum(axis=0)
        
        # NaN out invalid values. Eventually replace with numpy masked arrays
        self.N[self.N==0.] = np.nan 
        self.M[self.M==0.] = np.nan 
        self.Rm[self.Rm==0.] = np.nan
        
        self.Ntot[self.Ntot==0.] = np.nan 
        self.Mtot[self.Mtot==0.] = np.nan 
        self.Rmtot[self.Rmtot==0.] = np.nan

    def calc_radar(self):
        
        self.setup_radar()
        
        angs = self.angs
        
        ang1 = angs[0,:,:]
        ang2 = angs[1,:,:]
        ang3 = angs[2,:,:]
        ang4 = angs[3,:,:]
        ang5 = angs[4,:,:]
        #ang6 = angs[5,:,:]
        ang7 = angs[6,:,:]
        
        fhh_180_1 = fhh_0_1 = self.fscatt_pre1* (1 / (self.lb1 + self.eps1_factor))  
        fvv_180_1 = fvv_0_1 = self.fscatt_pre1* (1 / (self.la1 + self.eps1_factor))  
        
        fhh_180_2 = fhh_0_2 = self.fscatt_pre2* (1 / (self.lb2 + self.eps2_factor))  
        fvv_180_2 = fvv_0_2 = self.fscatt_pre2* (1 / (self.la2 + self.eps2_factor))
        
        #print('fhh_180_1=',fhh_180_1.shape)
        #raise Exception()
        
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
        # 
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
        # Shape = (dnum,Hlen,bins,Tout_len)
        self.zh = 1000.*(ak_zh[:,None,:,None]*self.Mbins+ck_zh[:,None,:,None]*self.Nbins)
        self.zv = 1000.*(ak_zv[:,None,:,None]*self.Mbins+ck_zv[:,None,:,None]*self.Nbins)
        self.kdp = 1000.*(ak_kdp[:,None,:,None]*self.Mbins+ck_kdp[:,None,:,None]*self.Nbins)
        self.zhhvv = 1000.*(ak_zhhvv[:,None,:,None]*self.Mbins+ck_zhhvv[:,None,:,None]*self.Nbins)
        
        # Prevents sqrt error if zh, zv, or zhhvv are slightly negative
        self.zh = np.maximum(self.zh,1e-10)
        self.zv = np.maximum(self.zv,1e-10)
        self.zhhvv = np.maximum(self.zhhvv,1e-10)
        
        # Radar variables for each habit, linear units 
        zh_tot = np.nansum(self.zh,axis=2)
        zv_tot = np.nansum(self.zv,axis=2)
        kdp_tot = np.nansum(self.kdp,axis=2)
        zhhvv_tot = np.abs(np.nansum(self.zhhvv,axis=2))
        rhohv_denom = np.sqrt(zh_tot*zv_tot)
        
        zh_full = np.nansum(zh_tot,axis=0)
        zv_full = np.nansum(zv_tot,axis=0)
        kdp_full = np.nansum(kdp_tot,axis=0)
        zhhvv_full = np.abs(np.nansum(zhhvv_tot,axis=0))
        rhohv_denom_full = np.sqrt(zh_full*zv_full)
        
        ref_min = 10.**(-3.5)
        
        # Habit specific radar variable
        self.Zh = np.full_like(zh_tot,np.nan)
        self.Zv = np.full_like(zv_tot,np.nan)
        self.Zdr = np.full_like(zh_tot,np.nan)
        self.Kdp = np.full_like(kdp_tot,np.nan)
        self.Zhhvv = np.full_like(zhhvv_tot,np.nan)
        self.Rhohv = np.full_like(zh_tot,np.nan)
        
        self.Zh[zh_tot>ref_min] = 10.*np.log10(zh_tot[zh_tot>ref_min]) 
        self.Zv[zv_tot>ref_min] = 10.*np.log10(zv_tot[zv_tot>ref_min]) 
        self.Zdr[(zh_tot>ref_min)&(zv_tot>ref_min)] = 10.*np.log10(zh_tot[(zh_tot>ref_min)&(zv_tot>ref_min)]/
                                                         zv_tot[(zh_tot>ref_min)&(zv_tot>ref_min)])
        self.Kdp   = kdp_tot
        self.Zhhvv[zhhvv_tot>ref_min] = 10.*np.log10(zhhvv_tot[zhhvv_tot>ref_min])
        self.Rhohv[rhohv_denom>ref_min] = zhhvv_tot[rhohv_denom>ref_min]/rhohv_denom[rhohv_denom>ref_min]

        # Total radar variables
        self.ZH = np.full_like(zh_full,np.nan)
        self.ZV = np.full_like(zv_full,np.nan)
        self.ZDR = np.full_like(zh_full,np.nan)
        self.KDP = np.full_like(kdp_full,np.nan)
        self.ZHHVV = np.full_like(zhhvv_full,np.nan)
        self.RHOHV = np.full_like(zh_full,np.nan)
        
        self.ZH[zh_full>ref_min] = 10.*np.log10(zh_full[zh_full>ref_min]) 
        self.ZV[zv_full>ref_min] = 10.*np.log10(zv_full[zv_full>ref_min]) 
        self.ZDR[(zh_full>ref_min)&(zv_full>ref_min)] = 10.*np.log10(zh_full[(zh_full>ref_min)&(zv_full>ref_min)]/
                                                         zv_full[(zh_full>ref_min)&(zv_full>ref_min)])
        self.KDP   = kdp_full
        self.ZHHVV[zhhvv_full>ref_min] = 10.*np.log10(zhhvv_full[zhhvv_full>ref_min])
        self.RHOHV[rhohv_denom_full>ref_min] = zhhvv_full[rhohv_denom_full>ref_min]/rhohv_denom_full[rhohv_denom_full>ref_min]
        
    # def activate_parallel(self):
        
    #     if self.parallel and self.pool is None:
    #         self.pool = self._parallel_config.__enter__()
    #         self.Ikernel.pool = self.pool

    def clean_up(self):
        
        sys.stdout.flush()
        
        #if self.parallel:
            
       #     for name, shm in self.Ikernel.shm_registry.items():
       #         shm.close()
       #         shm.unlink()
       #     self.Ikernel.shm_out.close()
       #     self.Ikernel.shm_out.unlink()
            
       # gc.collect()

    def check_init_dist(self):
              
        am = self.dist0.am 
        bm = self.dist0.bm
        av = self.dist0.av 
        bv = self.dist0.bv
        
        Dm_binned = ((am**(-4./bm)*self.dist0.moments(4./bm)).sum())/(am**(-3./bm)*self.dist0.moments(3./bm)).sum()
        
        Nt_binned = self.dist0.moments(0.).sum()
              
        WC_full = self.Nt0*am*(self.mu0+4)**(-bm)*self.Dm0**(bm)*scip.gamma(self.mu0+bm+1.)/scip.gamma(self.mu0+1)
        
        WC_binned = self.dist0.Mbins.sum()
        
        R_full = 3.6*self.Nt0*am*av*(self.mu0+4)**(-(bm+bv))*\
                 self.Dm0**(bm+bv)*scip.gamma(self.mu0+bm+bv+1.)/scip.gamma(self.mu0+1)
        R_binned = 3.6*av*(am)**(-bv/bm)*self.dist0.moments((bm+bv)/bm).sum()
        
        print('Initial Nt full = {:.2f} 1/L'.format(self.Nt0))
        print('Initial Nt binned = {:.2f} 1/L'.format(Nt_binned))
        print('------')
        print('Initial Dm full = {:.2f} mm'.format(self.Dm0))
        print('Initial Dm binned = {:.2f} mm'.format(Dm_binned))
        print('------')
        print('Initial WC full = {:.2f} g/m^3'.format(1000.*WC_full))
        print('Initial WC binned = {:.2f} g/m^3'.format(1000.*WC_binned))
        print('------')
        print('Initial R full = {:.2f} mm/hr'.format(1000.*R_full))
        print('Initial R binned = {:.2f} mm/hr'.format(1000.*R_binned))
    
    
    
    def diagnose_subgrid(self):
        
        xi1_4D = np.swapaxes(np.tile(self.xi1,(self.dnum,self.Hlen,self.Tout_len,1)),2,3)
        xi2_4D = np.swapaxes(np.tile(self.xi2,(self.dnum,self.Hlen,self.Tout_len,1)),2,3)

        if self.moments==2: # Diagnose 2 moment linear distributions
                
                        
            dx4D = np.swapaxes(np.tile(self.dxbins,(self.dnum,self.Hlen,self.Tout_len,1)),2,3)
                
            self.aki, self.cki, self.x1, self.x2 = update_2mom(self.Mbins,self.Nbins,self.rhobins,
                                                               self.bound_low,self.bound_high,
                                                               dx4D,
                                                               xi1_4D,
                                                               xi2_4D)
        elif self.moments==1: # Diagnose 1 moment uniform distributions
             
            self.Nbins = self.Mbins/self.xbins[None,None,:,None]
            
            self.cki = self.Mbins/self.dxi[None,None,:,None]
            
            self.aki = np.zeros_like(self.cki)
            
            self.x1 = xi1_4D
            self.x2 = xi2_4D

    def calc_moments(self,r,moments=2):  # Units are g^n
        # Integrate to find arbitrary moments of subgrid distribution Mn = Int x^n *[n(x)=ak*x+ck]*dx
        if moments == 2:
            return self.aki*Pn(r+1,self.x1,self.x2)+self.cki*Pn(r,self.x1,self.x2)
        elif moments == 1:
            return self.cki*Pn(r,self.x1,self.x2)


    def write_netcdf(self,filename):
        '''
        
        Parameters
        ----------
        filename : STRING
            Writes spectral_1d object attributes and variables to a netcdf file.

        '''
        
        with Dataset(filename,'w',format='NETCDF4') as file_nc:
            
            # Write attributes
            file_nc.bins = self.bins
            file_nc.sbin = self.sbin
            file_nc.dnum = self.dnum
            file_nc.moments = self.moments
            file_nc.ptype = self.ptype
            file_nc.dt = self.dt 
            file_nc.dz = self.dz 
            file_nc.Hlen = self.Hlen
            file_nc.Tlen = self.Tlen
            file_nc.output_freq = self.output_freq
            file_nc.Tout_len = self.Tout_len
            file_nc.rk_order = self.rk_order
            file_nc.adv_order = self.adv_order
            file_nc.int_type = self.int_type
            file_nc.kernel = self.kernel
            file_nc.radar = 1*self.radar
            file_nc.indb = self.indb 
            file_nc.indc = self.indc
            file_nc.dist_var = self.dist_var
            file_nc.mu0 = self.mu0
            file_nc.Dm0 = self.Dm0
            file_nc.Nt0 = self.Nt0
            file_nc.Mt0 = self.Mt0
            file_nc.D1 = self.D1 
            file_nc.lamf = self.lamf
            file_nc.x0 = self.dist0.x0
            file_nc.Tc = self.Tc
            file_nc.Ecol = self.Ecol
            file_nc.Es = self.Es
            file_nc.Eb = self.Eb 
            file_nc.Eagg = self.Eagg 
            file_nc.Ebr = self.Ebr 
            file_nc.Ecb = self.Ecb
            file_nc.gam_norm = 1*self.gam_norm
            if self.boundary is None:
                file_nc.boundary = 'None'
            else:
                file_nc.boundary = self.boundary
            file_nc.ztop = self.ztop 
            file_nc.zbot = self.zbot
            file_nc.tmax = self.tmax
            file_nc.t = self.t
            #file_nc.parallel = 1*self.parallel
            #file_nc.n_jobs = self.n_jobs
                
            file_nc.frag_dist = self.frag_dist
            
            file_nc.createDimension('time_out',self.Tout_len)
            file_nc.createDimension('height',self.Hlen)
            tout = file_nc.createVariable('tout','f4',('time_out',))
            z = file_nc.createVariable('z','f4',('height',))
        
            file_nc.createDimension('dists',self.dnum)
            file_nc.createDimension('bins',self.bins)
            
            # Put coordinates into 1D arrays
            xbins = file_nc.createVariable('xbins','f4',('bins',))
            xi1 = file_nc.createVariable('xi1','f4',('bins',))
            xi2 = file_nc.createVariable('xi2','f4',('bins',))
            
            tout[:] = self.tout
            z[:] = self.z 
            tout.units = 'seconds'
            tout.description = 'Output time'
            z.units = 'meters'
            z.description = 'Gridbox Height'
 
            xbins[:] = self.xbins
            xi1[:] = self.xedges[:-1]
            xi2[:] = self.xedges[1:]
            
            xbins.units = 'g'
            xi1.units = 'g'
            xi2.units = 'g'
            
            xbins.description = 'Mass grid midpoint'
            xi1.description = 'Mass grid left bin edge'
            xi2.description = 'Mass grid right bin edge'
                
            # Put distributions into 4D arrays
            for dd in range(self.dnum):
                # Create group
                dist_dd = file_nc.createGroup('dist{}'.format(dd+1))
                
                # Add attributes
                dist_dd.habit = self.habit_list[dd]
                dist_dd.am = self.dists[dd].am
                dist_dd.bm = self.dists[dd].bm
                dist_dd.av = self.dists[dd].av
                dist_dd.bv = self.dists[dd].bv
                dist_dd.arho = self.dists[dd].arho
                dist_dd.brho = self.dists[dd].brho
    
            # Create Array Variables
            Mbins = file_nc.createVariable('Mbins','f8',('dists','height','bins','time_out',))
            Mbins.units = 'g'
            Mbins.description = 'Total Bin Mass'
            
            Mbins[:] = self.Mbins
            
            if self.moments==2:
                Nbins = file_nc.createVariable('Nbins','f8',('dists','height','bins','time_out'))
                Nbins.units = '#'
                Nbins.description = 'Total Bin Number'
                
                Nbins[:] = self.Nbins
            

    def plot_time_height(self,var='Z'):
        '''
        For full 1D model runs, plots a time/height pcolor plot for specified input variable.
        '''
        
        if latex_check():
            plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize=22) 
        plt.rc('ytick', labelsize=22) 

        t = self.tout
        h = self.z/1000.
        
        fig, ax = plt.subplots(1,1,figsize=(14,6))

        match var:
            case 'Z':
                var_temp = self.ZH
            case 'ZDR':
                var_temp = self.ZDR
            case 'KDP':
                var_temp = self.KDP
            case 'RHOHV':
                var_temp = self.RHOHV
            case 'Nt':
                var_temp = self.Ntot
            case 'Dm':
                var_temp = self.Dmtot
            case 'WC':
                var_temp = self.Mtot
            case 'R':
                var_temp = self.Rmtot
        
        cmap, levels, levels_ticks, clabel, labelpad, fontsize, slabel = get_cmap_vars(var)
        
        Rnorm = BoundaryNorm(levels,cmap.N,extend='both') 

        cax = ax.pcolor(t,h,var_temp,norm=Rnorm,cmap=cmap)
        
        #cax = ax.pcolor(t,h,var_temp,cmap=cmap)
        
        cbar = fig.colorbar(cax,ax=ax,ticks=levels_ticks)
        
        cbar.ax.tick_params(labelsize=16)
        
        cbar.ax.set_yticklabels(levels_ticks,usetex=True)
        
        cbar.ax.minorticks_off()
        
        cbar.set_label(clabel,usetex=True,rotation=270,fontsize=fontsize,labelpad=labelpad) 

        ax.set_xlabel('Time (seconds)',fontsize=36,usetex=True)
        ax.set_ylabel('Height (km)',fontsize=36,usetex=True)
        
        ax.axes.tick_params('both',labelsize=26)

        fig.tight_layout()  
        
        return fig, ax

    def plot_moments_radar(self,ax=None,tind=-1,plot_habits=False,lstyle='-'):
        
        # If we don't have radar variables, the just calculate them here.
        if not hasattr(self,'ZH'):
            self.calc_radar()
        
        #lstyle = '-'
        
        if ax is None:
            ax_switch = True 
        else:
            ax_switch = False
        
        if latex_check():
            plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize=16) 
        plt.rc('ytick', labelsize=16) 
        
        dist_num = self.dnum

        if self.int_type==2: # Box model
        
            N  = self.N[:,0,:]
            M  = self.M[:,0,:]
            Dm = self.Dm[:,0,:]
            Rm = np.round(self.Rm[:,0,:],5)
            
            ZH = self.ZH[0,:]
            ZDR = self.ZDR[0,:]
            KDP = self.KDP[0,:]
            RHOHV = self.RHOHV[0,:]
            
            N_tot  = self.Ntot[0,:]
            M_tot  = np.round(self.Mtot[0,:],5)
            Rm_tot = np.round(self.Rmtot[0,:],5)
            Dm_tot = self.Dmtot[0,:]
            
            
            if ax is None:
                fig, ax = plt.subplots(2,4,figsize=(14,8),sharex=True)
            
            ax[0,0].plot(self.tout,N_tot,'k',linestyle=lstyle,label='total')
            ax[0,1].plot(self.tout,Dm_tot,'k',linestyle=lstyle)
            ax[0,2].plot(self.tout,M_tot,'k',linestyle=lstyle)
            ax[0,3].plot(self.tout,Rm_tot,'k',linestyle=lstyle)
            
            ax[1,0].plot(self.tout,ZH,color='k',linestyle=lstyle)
            ax[1,1].plot(self.tout,ZDR,color='k',linestyle=lstyle)
            ax[1,2].plot(self.tout,KDP,color='k',linestyle=lstyle)
            ax[1,3].plot(self.tout,RHOHV,color='k',linestyle=lstyle)
            
            ax[0,0].set_ylabel('Nt (1/L)',usetex=True,fontsize=18)
            ax[0,1].set_ylabel('Dm (mm)',usetex=True,fontsize=18)
            ax[0,2].set_ylabel(r'WC (g/m$^{3}$)',usetex=True,fontsize=18)
            ax[0,3].set_ylabel('R (mm/hr)',usetex=True,fontsize=18)
            ax[1,0].set_ylabel('Z (dBZ)',usetex=True,fontsize=18)
            ax[1,1].set_ylabel('ZDR (dB)',usetex=True,fontsize=18)
            ax[1,2].set_ylabel('Kdp (deg/km)',usetex=True,fontsize=18)
            ax[1,3].set_ylabel(r'$\rho_{\mathrm{hv}}$',usetex=True,fontsize=26)
            
            ax[1,0].set_xlabel('Time (sec)',fontsize=16,usetex=True)
            ax[1,1].set_xlabel('Time (sec)',fontsize=16,usetex=True)
            ax[1,2].set_xlabel('Time (sec)',fontsize=16,usetex=True)
            ax[1,3].set_xlabel('Time (sec)',fontsize=16,usetex=True)
            
            ax[0,0].axes.tick_params('both',labelsize=14)
            ax[0,1].axes.tick_params('both',labelsize=14)
            ax[0,2].axes.tick_params('both',labelsize=14)
            ax[0,3].axes.tick_params('both',labelsize=14)
            ax[1,0].axes.tick_params('both',labelsize=14)
            ax[1,1].axes.tick_params('both',labelsize=14)
            ax[1,2].axes.tick_params('both',labelsize=14)
            ax[1,3].axes.tick_params('both',labelsize=14)
     
            if plot_habits:
                
                for d1 in range(dist_num):
                                     
                    ax[0,0].plot(self.tout,N[d1,:],linestyle=lstyle,label='dist {}'.format(d1+1))
                    ax[0,1].plot(self.tout,Dm[d1,:],linestyle=lstyle)
                    ax[0,2].plot(self.tout,M[d1,:],linestyle=lstyle)
                    ax[0,3].plot(self.tout,Rm[d1,:],linestyle=lstyle)
   
        else:
            
            N  = self.N[:,:,tind]
            M  = self.M[:,:,tind]
            Dm = self.Dm[:,:,tind]
            Rm = np.round(self.Rm[:,:,tind],5)
            
            ZH = self.ZH[:,tind]
            ZDR = self.ZDR[:,tind]
            KDP = self.KDP[:,tind]
            RHOHV = self.RHOHV[:,tind]
            
            N_tot  = self.Ntot[:,tind]
            M_tot  = np.round(self.Mtot[:,tind],5)
            Rm_tot = np.round(self.Rmtot[:,tind],5)
            Dm_tot = self.Dmtot[:,tind]
            
            #fig, ax = plt.subplots(1,3,figsize=(12,6),sharey=True)
            
            if ax is None:
                fig, ax = plt.subplots(2,4,figsize=(14,8),sharey=True)
            
            ax[0,0].plot(N_tot,self.z/1000.,'k',linestyle=lstyle,label='total')
            ax[0,1].plot(Dm_tot,self.z/1000.,'k',linestyle=lstyle)
            ax[0,2].plot(M_tot,self.z/1000.,'k',linestyle=lstyle)
            ax[0,3].plot(Rm_tot,self.z/1000.,'k',linestyle=lstyle)
            
            ax[1,0].plot(ZH,self.z/1000.,color='k',linestyle=lstyle)
            ax[1,1].plot(ZDR,self.z/1000.,color='k',linestyle=lstyle)
            ax[1,2].plot(KDP,self.z/1000.,color='k',linestyle=lstyle)
            ax[1,3].plot(RHOHV,self.z/1000.,color='k',linestyle=lstyle)
            
            ax[0,0].set_xlabel('Nt (1/L)',usetex=True,fontsize=18)
            ax[0,1].set_xlabel('Dm (mm)',usetex=True,fontsize=18)
            ax[0,2].set_xlabel(r'WC (g/m$^{3}$)',usetex=True,fontsize=18)
            ax[0,3].set_xlabel('R (mm/hr)',usetex=True,fontsize=18)
            ax[1,0].set_xlabel('Z (dBZ)',usetex=True,fontsize=18)
            ax[1,1].set_xlabel('ZDR (dB)',usetex=True,fontsize=18)
            ax[1,2].set_xlabel('Kdp (deg/km)',usetex=True,fontsize=18)
            ax[1,3].set_xlabel(r'$\rho_{\mathrm{hv}}$',usetex=True,fontsize=26)
            
            ax[0,0].set_ylabel('Height (km)',fontsize=16,usetex=True)
            ax[1,0].set_ylabel('Height (km)',fontsize=16,usetex=True)
            
            ax[0,0].axes.tick_params('both',labelsize=14)
            ax[0,1].axes.tick_params('both',labelsize=14)
            ax[0,2].axes.tick_params('both',labelsize=14)
            ax[0,3].axes.tick_params('both',labelsize=14)
            ax[1,0].axes.tick_params('both',labelsize=14)
            ax[1,1].axes.tick_params('both',labelsize=14)
            ax[1,2].axes.tick_params('both',labelsize=14)
            ax[1,3].axes.tick_params('both',labelsize=14)
            
            if plot_habits:
                
                for d1 in range(dist_num):
                    ax[0,0].plot(N[d1,:],self.z/1000.,linestyle=lstyle,label='dist {}'.format(d1+1))
                    ax[0,1].plot(Dm[d1,:],self.z/1000.,linestyle=lstyle)
                    ax[0,2].plot(M[d1,:],self.z/1000.,linestyle=lstyle)
                    ax[0,3].plot(Rm[d1,:],self.z/1000.,linestyle=lstyle)

        if ax_switch:
            ax[0,0].legend(loc='upper center')
    
            fig.tight_layout()  
        
            return fig, ax       
 
 
    def plot_init(self,log_switch=True,x_axis='mass'):

        if latex_check():
            plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize=22) 
        plt.rc('ytick', labelsize=22)         

        mbins = self.dist0.xbins
       # medges = self.dist0.xedges.copy() 
        xp1 = self.dist0.x1
        xp2 = self.dist0.x2
        ap = self.dist0.aki
        cp = self.dist0.cki
        
        bm = self.dist0.bm
        am = self.dist0.am

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

        fig, ax = plt.subplots(2,1,figsize=((8,10)),sharex=True)
        
        # Plot m*n(m) for number (N=int m*n(m)*dln(m)) | g_n(ln(r)) = bm*m*n(m), N = int g_n(ln(r))*dln(r)
        # Plot m^2*n(m) for mass (M=int m^2*n(m)*dln(m)) | g_m(ln(r)) = bm*m^2*n(m), M = int g_m(ln(r))*dln(r) 
        
        # Initial
        ax[0].plot(np.log10(xbins),mbins*n_init,'k')
        if self.gam_norm:
            ax[1].plot(np.log10(xbins),mbins**2*n_init,'k')
        else:
            ax[1].plot(np.log10(xbins),1000.*mbins**2*n_init,'k')
        ax[0].set_ylabel(ylabel_num)
        ax[1].set_ylabel(ylabel_mass)
        ax[1].set_xlabel(xlabel)
        
        #print('Initial Number = {:.2f} #/L'.format(np.nansum(mbins*n_init*(np.log10(medges[1:])-np.log10(medges[:-1])))))
        #print('Initial Mass = {:.2f} g/cm^3'.format(np.nansum(1000.*mbins**2*n_init*(np.log10(medges[1:])-np.log10(medges[:-1])))))
        
        #print('number test size=',np.nansum(mbins*n_init*(np.log10(dedges[1:])-np.log10(dedges[:-1]))))
        #print('mass test size=',np.nansum(1000.*mbins**2*n_init*(np.log10(dedges[1:])-np.log10(dedges[:-1]))))
        
 
        return fig, ax


    def plot_dists(self,tind=-1,hind=-1,x_axis='mass',y_axis='mass',xscale='log',yscale='linear',distscale='log',normbin=False,scott_solution=False,feingold_solution=False,plot_habits=False,ax=None,lstyle='-',lcolor='k'):

        if latex_check():
            plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize=26) 
        plt.rc('ytick', labelsize=26)   
      
        if ax is None:
            ax_switch = True 
        else:
            ax_switch = False
        
        # NOTE: probably need to figure out how to deal with x_axis='size' when
        # am and bm parameters are different for each habit.
        
        if ax is None:
            fig, ax = plt.subplots(2,1,figsize=((8,10)),sharex=True)
        

        print('Plotting distributions...')
        
        
        mbins = self.xbins.copy()
        
        if normbin:
            #dxbins = np.log10(primary_init.xedges[1:]/primary_init.xedges[:-1])
            dxbins = np.log10(self.xedges[1:]/self.xedges[:-1])
        else:
            dxbins = self.dxbins.copy()
        

        # (dnum,Hlen,bins,Tout_len)
        xp1 = self.dist0.x1
        xp2 = self.dist0.x2
        ap = self.dist0.aki
        cp = self.dist0.cki
        Mbins_init = self.dist0.Mbins
        Nbins_init = self.dist0.Nbins
        
       # xbins = np.full((self.dnum,self.bins),np.nan)
        prefN = np.full((self.dnum,self.bins),np.nan)
        prefM = np.full((self.dnum,self.bins),np.nan)
        x1_final = np.full((self.dnum,self.bins),np.nan)
        x2_final = np.full((self.dnum,self.bins),np.nan)
        ak_final = np.full((self.dnum,self.bins),np.nan)
        ck_final = np.full((self.dnum,self.bins),np.nan)
        Mbins_final = np.full((self.dnum,self.bins),np.nan)
        Nbins_final = np.full((self.dnum,self.bins),np.nan)
        bm = np.full((self.dnum,),np.nan)
        am = np.full((self.dnum,),np.nan)
        
        x1_final = self.x1[:,hind,:,tind]
        x2_final = self.x2[:,hind,:,tind]
        ck_final = self.cki[:,hind,:,tind]
        ak_final = self.aki[:,hind,:,tind] 
        Mbins_final = self.Mbins[:,hind,:,tind]
        Nbins_final = self.Nbins[:,hind,:,tind]
        
        for d1 in range(self.dnum):
            
            bm[d1] = self.habit_dict[d1]['bm']
            am[d1] = self.habit_dict[d1]['am']

        if self.int_type==0:
            f_label = '{:.1f} km | {:.2f} min.'.format(self.z[hind]/1000.,self.tout[tind]/60.)
                     
        elif self.int_type==1:
            f_label = '{:.1f} km'.format(self.z[hind]/1000.)

        elif self.int_type==2:
            f_label = '{:.2f} min.'.format(self.tout[tind]/60.)

   
        # Distscale toggles between dN/dlog(m) plots and dN/dm plots, for example.
        if distscale=='log':
        
            if x_axis=='mass': # plot dN/dlog(m) and dM/dlog(m)
            
                for d1 in range(self.dnum):
                    prefN[d1,:] = mbins*np.log(10)   
                    prefM[d1,:] = mbins**2*np.log(10)
                    
                    #xbins[d1,:] = mbins.copy()
                xbins = mbins.copy()
                if normbin:
                    ylabel_num = r'd$P_{N}$/dlog(m)'
                    ylabel_mass = r'd$P_{M}$/dlog(m)'
                    
                else:
                    ylabel_num = r'dN/dlog(m)'
                    ylabel_mass = r'dM/dlog(m)'
                
                xlabel = r'log(m) [log(g)]'
                
            elif x_axis=='size':  # plot dN/dlog(D) and dM/dlog(D)
                
                for d1 in range(self.dnum):
                    
                    prefN[d1,:] = mbins*bm[d1]*np.log(10) 
                    prefM[d1,:] = mbins**2*bm[d1]*np.log(10) 
                
                    #xbins[d1,:] = (mbins/self.full[d1,0].am)**(1./self.full[d1,0].bm)
               
                xbins = (mbins/am[0])**(1./bm[0])
   
                if normbin:
                    ylabel_num = r'd$P_{N}$/dlog(D)'
                    ylabel_mass = r'd$P_{M}$/dlog(D)'
                    
                else:
                    ylabel_num = r'dN/dlog(D)'
                    ylabel_mass = r'dM/dlog(D)'
                
                xlabel = r'log(D) [log(mm)]'
                
                # Set xtickparams to something readable?
                #ax[1].set_xticks(2.**np.arange(-3,8,1))
                #ax[1].set_xticklabels(2.**np.arange(-3,8,1))
                
        else: # Linear plots of form dN/dm or dN/dD, for example.
            
            if x_axis=='mass':  # Linear plots of form dN/dm
            
                for d1 in range(self.dnum):
                    prefN[d1,:] = np.ones_like(mbins)
                    prefM[d1,:] = mbins.copy()

                    #xbins[d1,:] = mbins.copy()
                xbins = mbins.copy()
                
                if normbin:
                    ylabel_num = r'd$P_{N}$/dm'
                    ylabel_mass = r'd$P_{M}$/dm'
                    
                else:
                    ylabel_num = r'dN/dm'
                    ylabel_mass = r'dM/dm'
                
                xlabel = r'log(m) [log(g)]'
                
            elif x_axis=='size': # Linear plots of form dN/dD
                
                for d1 in range(self.dnum):
                    # !!!CHECK
                    prefN[d1,:] = am[d1]**(1./bm[d1])*bm[d1]*mbins**(1.-1./bm[d1])
                    prefM[d1,:] = am[d1]**(1./bm[d1])*bm[d1]*mbins**(2.-1./bm[d1])

                    #xbins[d1,:] = (mbins/self.full[d1,0].am)**(1./self.full[d1,0].bm)
                xbins = (mbins/am[0])**(1./bm[0])
                
                if normbin:
                    ylabel_num = r'd$P_{N}$/dD'
                    ylabel_mass = r'd$P_{M}$/dD'
                else:
                    ylabel_num = r'dN/dD'
                    ylabel_mass = r'dM/dD'
         
        if normbin:
            #Nbins_init[Nbins_init<1e-6] = np.nan
            #Mbins_init[Mbins_init<1e-6] = np.nan
            prefN_init = prefN[0,:]/np.nansum(Nbins_init)
            prefM_init = prefM[0,:]/np.nansum(Mbins_init)
            
           # Nbins_final[Nbins_final<1e-6] = np.nan
           # Mbins_final[Mbins_final<1e-6] = np.nan
            prefN_final = prefN/np.nansum(Nbins_final)
            prefM_final = prefM/np.nansum(Mbins_final)
            
        else:
            prefN_init = prefN[0,:].copy()
            prefM_init = prefM[0,:].copy()
            
            prefN_final = prefN.copy()
            prefM_final = prefM.copy()
            
        # nN_init = prefN[0,:]*np.heaviside(mbins-xp1,1)*np.heaviside(xp2-mbins,1)*(ap*mbins+cp)
        # nM_init = prefM[0,:]*np.heaviside(mbins-xp1,1)*np.heaviside(xp2-mbins,1)*(ap*mbins+cp)

        # nN_final = prefN*np.heaviside(mbins[None,:]-x1_final,1)*np.heaviside(x2_final-mbins[None,:],1)*(ak_final*mbins[None,:]+ck_final)
        # nM_final = prefM*np.heaviside(mbins[None,:]-x1_final,1)*np.heaviside(x2_final-mbins[None,:],1)*(ak_final*mbins[None,:]+ck_final)


        nN_init = prefN_init*np.heaviside(mbins-xp1,1)*np.heaviside(xp2-mbins,1)*(ap*mbins+cp)
        nM_init = prefM_init*np.heaviside(mbins-xp1,1)*np.heaviside(xp2-mbins,1)*(ap*mbins+cp)

        nN_final = prefN_final*np.heaviside(mbins[None,:]-x1_final,1)*np.heaviside(x2_final-mbins[None,:],1)*(ak_final*mbins[None,:]+ck_final)
        nM_final = prefM_final*np.heaviside(mbins[None,:]-x1_final,1)*np.heaviside(x2_final-mbins[None,:],1)*(ak_final*mbins[None,:]+ck_final)


        if self.gam_norm:
            nM_init /= 1000. 
            nM_final /= 1000.
            

        if xscale=='log':
            x = np.log10(xbins)
            
            if (x_axis=='size'):
                xlabel = r'log(D) [log(mm)]'
            elif (x_axis=='mass'):
                xlabel = r'log(m) [log(g)]'
            
        elif xscale=='linear':
            x = xbins.copy()
            ax[0].set_xlim((0.,10.))
           # ax[1].set_ylim((0.,10.))
             
            if (x_axis=='size'):
                xlabel = r'D (mm)'
            elif (x_axis=='mass'):
                xlabel = r'm (g)'
        
        if ax_switch:
            ax[0].plot(x,nN_init,':k',linewidth=2,label='initial')
        ax[0].plot(x,np.nansum(nN_final,axis=0),linestyle=lstyle,color=lcolor,linewidth=2,label=f_label)
        if plot_habits:
            for d1 in range(self.dnum):
                ax[0].plot(x,nN_final[d1,:],linewidth=2,label='dist {}'.format(d1+1))
            
        # Factor of 1000 comes from converting g to g/m^3
        if ax_switch:
            ax[1].plot(x,1000.*nM_init,':k',linewidth=2,label='initial')
        ax[1].plot(x,1000.*np.nansum(nM_final,axis=0),linestyle=lstyle,color=lcolor,linewidth=2,label=f_label)
        if plot_habits:
            for d1 in range(self.dnum):
                ax[1].plot(x,1000.*nM_final[d1,:],linewidth=2,label='dist {}'.format(d1+1))

        ax[0].set_ylabel(ylabel_num,fontsize=26)
        ax[1].set_ylabel(ylabel_mass,fontsize=26)
        
        ax[1].set_xlabel(xlabel,fontsize=26)
        
        if yscale=='log':
            ax[0].set_yscale('log')
            ax[1].set_yscale('log')
            #ax[0].set_ylim((1e-5,max(nN_init.max(),1000.*nN_final.max())))
            #ax[1].set_ylim((1e-5,max(nM_init.max(),1000.*nM_final.max())))
            
            ax[0].set_ylim(bottom=1e-5)
            ax[1].set_ylim(bottom=1e-5)

        #print('number test=',np.nansum(mbins*n_init*(np.log(medges[1:])-np.log(medges[:-1]))))
       # print('mass test=',np.nansum(mbins**2*1000.*n_init*(np.log(medges[1:])-np.log(medges[:-1]))))
        
        if (scott_solution & (self.int_type==2)):
            
            kernel_type = self.kernel
            
            #if not (hasattr(self,'n_scott')):
            self.n_scott = Scott_dists(self.xbins,self.Eagg,self.mu0+1,self.t,kernel_type=kernel_type)
        
            n_scott_new = prefN[0,:]*self.n_scott[:,tind]
            nm_scott_new = prefM[0,:]*self.n_scott[:,tind]
        
            if normbin:
                n_scott_new /= np.nansum(n_scott_new*dxbins)
                nm_scott_new /= np.nansum(nm_scott_new*dxbins)
                
                #print('n_scott_new=',np.nansum(n_scott_new*dxbins))

        
            #ax[0].plot(x,prefN[0,:]*self.n_scott[:,tind],':r',linewidth=2,label=f_label+ " analytical")
            ax[0].plot(x,n_scott_new,':r',linewidth=2,label=f_label+ " analytical")
            
            #ax[1].plot(x,1000.*prefM[0,:]*self.n_scott[:,tind],':r',label=f_label+ "analytical")
            
            #ax[1].plot(x,prefM[0,:]*self.n_scott[:,tind],':r',linewidth=2,label=f_label+ " analytical")
            ax[1].plot(x,nm_scott_new,':r',linewidth=2,label=f_label+ " analytical")
        
        if (feingold_solution & (self.int_type==2)):
            
            kernel_type = self.kernel
            
            C = self.Eagg 
            B = self.Ebr 
            
            if B>0.:
                if (C==0.):
                    kernel_type = 'SBE'
            
                elif (C>0.):
                    kernel_type = 'SCE/SBE'
                    
                #if not (hasattr(self,'n_fein')):
                self.n_fein = Feingold_dists(self.xbins,self.t,self.mu0+1,self.Eagg,self.Ebr,self.lamf,kernel_type=kernel_type)


                if kernel_type=='SBE':
                    
                    n_fein_new = prefN[0,:]*self.n_fein[:,tind]
                    nm_fein_new = prefM[0,:]*self.n_fein[:,tind]

                    if normbin:
                        n_fein_new /= np.nansum(n_fein_new*dxbins)
                        nm_fein_new /= np.nansum(nm_fein_new*dxbins)
                    
                    #ax[0].plot(x,prefN[0,:]*self.n_fein[:,tind],':r',linewidth=2,label=f_label+ " analytical")
                    #ax[1].plot(x,prefM[0,:]*self.n_fein[:,tind],':r',linewidth=2,label=f_label+ " analytical")
                    
                    ax[0].plot(x,n_fein_new,':r',linewidth=2,label=f_label+ " analytical")
                    ax[1].plot(x,nm_fein_new,':r',linewidth=2,label=f_label+ " analytical")
                    
                elif kernel_type=='SCE/SBE':
                    
                    n_fein_new = prefN[0,:]*self.n_fein
                    nm_fein_new = prefM[0,:]*self.n_fein

                    if normbin:
                        n_fein_new /= np.nansum(n_fein_new*dxbins)
                        nm_fein_new /= np.nansum(nm_fein_new*dxbins)
                        
                    #ax[0].plot(x,prefN[0,:]*self.n_fein,':r',linewidth=2,label=f_label+ " analytical")
                    #ax[1].plot(x,prefM[0,:]*self.n_fein,':r',linewidth=2,label=f_label+ " analytical")
                    
                    ax[0].plot(x,n_fein_new,':r',linewidth=2,label=f_label+ " analytical")
                    ax[1].plot(x,nm_fein_new,':r',linewidth=2,label=f_label+ " analytical")
                    
        ax[0].legend() 
            
        #plt.tight_layout()
        
        if ax_switch:
            
            fig.tight_layout()  
            
            return fig, ax    
    

    def plot_dists_height(self,tind=-1,plot_habits=False):
        
        if latex_check():
            plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize=22) 
        plt.rc('ytick', labelsize=22) 
        
        z = self.z/1000.
        
        z_lvls = np.arange(np.max(z),np.min(z)-1.,-1.)
        
        fig, ax = plt.subplots(len(z_lvls),1,figsize=(6,10),sharey=True,sharex=True)
        
        mbins = self.dist0.xbins.copy() 
        
        # if self.int_type==0:
        #     primary_init  = self.full[0,0,tind]
        # else:
        #     primary_init = self.full[0,0]
        
       # primary_init  = self.full[0,0]
        
        xp1 = self.dist0.x1
        xp2 = self.dist0.x2
        ap = self.dist0.aki
        cp = self.dist0.cki
        
        prefN_p = self.dist0.am**(1./self.dist0.bm)*self.dist0.bm*mbins**(1.-1./self.dist0.bm)
        
        #!!! NOTE: Need to change? Size here.
        xbins = (mbins/self.dist0.am)**(1./self.dist0.bm)
        
        nN_init = prefN_p*np.heaviside(mbins-xp1,1)*np.heaviside(xp2-mbins,1)*(ap*mbins+cp)
        
        ax[0].plot(xbins,nN_init,'k')

        ax[0].set_yscale('log')
        ax[0].set_xscale('linear')
        ax[0].set_ylim(bottom=0.001)
        
        ax[0].set_title('Height = {} km'.format(z_lvls[0]),fontsize=26)
        #ax[0].set_ylabel(r'Number Density (1/cm$^{3}$ 1/mm)',fontsize=16)
        ax[0].set_ylabel(r'n(D) (1/cm$^{3}$ 1/mm)',fontsize=16)
        
        ax[0].axes.tick_params(labelsize=20)
        ax[0].set_xlim((0.,5.))
        
        for hh in range(1,len(z_lvls)):
        
            zind = np.nonzero(z==z_lvls[hh])[0][0]
             
            nN_final = np.full((self.dnum,self.bins),np.nan)
            
            
            for d1 in range(self.dnum):
                
                am = self.habit_dict[d1]['am']
                bm = self.habit_dict[d1]['bm']
                
                #!!! NOTE: PICK UP HERE. NEED TO FIGURE OUT WHAT TO DO ABOUT OUTFREQ!
                #print('zind=',zind)
                #raise Exception()
                xp1_final = self.x1[d1,zind,:,tind]
                xp2_final = self.x2[d1,zind,:,tind]
                ap_final = self.aki[d1,zind,:,tind]
                cp_final = self.cki[d1,zind,:,tind]
                
                prefN = am**(1./bm)*bm*mbins**(1.-1./bm)
                
                nN_final[d1,:] = prefN*np.heaviside(mbins-xp1_final,1)*np.heaviside(xp2_final-mbins,1)*(ap_final*mbins+cp_final)

                if plot_habits:
                    ax[hh].plot(xbins,nN_final[d1,:],label='dist {}'.format(d1+1))

            if plot_habits:
                ax[hh].plot(xbins,np.nansum(nN_final,axis=0),color='k')
            else:
                ax[hh].plot(xbins,np.nansum(nN_final,axis=0),color='k',label='total')
                
            ax[hh].set_yscale('log')
            ax[hh].set_xscale('linear')
            ax[hh].set_ylim(bottom=0.001)
            
            ax[hh].set_title('Height = {} km'.format(z_lvls[hh]),fontsize=26)
            #ax[hh].set_ylabel(r'Number Density (1/cm$^{3}$ 1/mm)',fontsize=16)
            ax[hh].set_ylabel(r'n(D) (1/cm$^{3}$ 1/mm)',fontsize=16)
            
            ax[hh].axes.tick_params(labelsize=20)
            ax[hh].set_xlim((0.,5.))

        
        ax[0].set_ylim((1e-5,1e5))
        ax[-1].set_xlabel('Equivolume Diameter (mm)',fontsize=22)
        if plot_habits:
            ax[-1].legend()
        
        plt.tight_layout()
        
        return fig, ax        
 
 
    def run_steady_state_1mom(self, pbar=None):
        ''' 
        Run steady-state bin model (1-Moment) using Explicit Runge-Kutta integration.
        Uses ADAPTIVE SUB-STEPPING based on "Fraction of Residence Time".
        '''
        
        # 1. Initialize Runge-Kutta Coefficients
        RK = init_rk(self.rk_order)
        a = RK['a']
        b = RK['b']
        rklen = len(b)
        
        # Total Residence Time Vector (dnum, 1, bins)
        # self.dh contains the residence time (dt) for each bin
        total_residence_time = self.dh[:, None, :] 
        
        tf = 0
        
        # Pre-allocate RK stage storage
        dM_stages = np.zeros((self.dnum, self.Hlen, self.bins, rklen))

        # ---------------------------------------------------------------------
        # MAIN LOOP
        # ---------------------------------------------------------------------
        for tt in range(1, self.Tlen):
            
            if pbar:
                pbar.set_description(self.out_text(self.Ikernel.Mbins, self.Ikernel.Mfbins))
                pbar.update(1)

            # Start State (y_n)
            M_current = self.Ikernel.Mbins.copy()
            
            # Reset Progress (0.0 -> 1.0)
            progress = 0.0
            
            # -----------------------------------------------------------------
            # ADAPTIVE SUB-STEPPING LOOP
            # -----------------------------------------------------------------
            while progress < 1.0:
                
                # --- A. Estimate Stiffness / Safe Fraction ---
                
                # 1. Update Kernel State
                self.Ikernel.Mbins = M_current
                self.Ikernel.update_1mom_subgrid()
                
                # 2. Get current rates (dM/dt)
                # Ensure get_steady_rates_1mom calls interact_1mom... correctly
                # (You likely have a 1-moment equivalent of get_steady_rates)
                dM_dt = self.get_steady_rate_1mom(M_current)
                
                # 3. Calculate Timescale for every bin
                safe_mask_M = (M_current > 1e-15) & (np.abs(dM_dt) > 1e-20)
                
                if np.any(safe_mask_M):
                    # Time to change mass by 50%
                    t_scale_M = 0.5 * M_current[safe_mask_M] / np.abs(dM_dt[safe_mask_M])
                    
                    # Fraction of TOTAL residence time
                    # Broadcast total_residence_time to match mask shape
                    res_time_masked = np.broadcast_to(total_residence_time, M_current.shape)[safe_mask_M]
                    
                    valid_res = res_time_masked > 1e-10
                    if np.any(valid_res):
                        frac_M = t_scale_M[valid_res] / res_time_masked[valid_res]
                        d_prog_safe = np.min(frac_M)
                    else:
                        d_prog_safe = 1.0
                else:
                    d_prog_safe = 1.0

                # Clamp to remaining progress
                remaining = 1.0 - progress
                d_prog = min(d_prog_safe, remaining)
                
                # Prevent infinitesimally small steps (1e-4 minimum fraction)
                d_prog = max(d_prog, 1e-4)
                
                # -------------------------------------------------------------
                # B. CALCULATE VECTORIZED STEP SIZE
                # -------------------------------------------------------------
                # step_size is a VECTOR proportional to self.dh
                current_step_vec = d_prog * total_residence_time
                
                # -------------------------------------------------------------
                # C. RK INTEGRATION
                # -------------------------------------------------------------
                dM_stages.fill(0.)

                for ii in range(rklen):
                    if ii == 0:
                        M_stage = M_current
                    else:
                        # Sum previous stages
                        k_sum_M = np.sum(a[ii, :ii][None, None, None, :] * dM_stages[:, :, :, :ii], axis=3)
                        M_stage = np.maximum(M_current + current_step_vec * k_sum_M, 0.)

                    # Calculate Rates at stage
                    dM_dt_k = self.get_steady_rate_1mom(M_stage)
                    dM_stages[:, :, :, ii] = dM_dt_k
                
                # Final Update for this sub-step
                total_sum_M = np.sum(b[None, None, None, :] * dM_stages, axis=3)
                M_current = np.maximum(M_current + current_step_vec * total_sum_M, 0.)
                
                # Advance Progress
                progress += d_prog
            
            # Commit final state
            self.Ikernel.Mbins = M_current
            self.Ikernel.update_1mom_subgrid()

            # Save Output
            if self.save_mask[tt]:
                tf += 1 
                self.Mbins[:, :, :, tf] = self.Ikernel.Mbins.copy() 
    
        # Post-processing
        if self.int_type == 1:
            self.Mbins = np.swapaxes(self.Mbins, 1, 3)
            Tout_len = self.Hlen
            Hlen = self.Tout_len
            self.Tout_len = Tout_len
            self.Hlen = Hlen

    def get_steady_rate_1mom(self, M_curr):
        '''
        Helper function to calculate derivatives for the Steady State RK scheme.
        Updates the kernel state and returns the rate of change.
        '''
        # 1. Update the Kernel State with the RK Stage value
        self.Ikernel.Mbins = M_curr
        
        # Update physics parameters (fall speeds, diameters, etc.) based on new Mass
        self.Ikernel.update_1mom_subgrid()
        
        # 2. Calculate Rates
        # Passing 1.0 means we get the rate per unit step (mass/sec * 1.0)
        # or in this context (mass/height_step)
        M_net = self.Ikernel.interact_1mom_SS_Final(1.0)
        
        return M_net 
 
    def run_full_1mom(self, pbar=None):
        ''' 
        Run full 1D bin model using Adaptive Sub-stepping RK4.
        Matches original BC logic and optimizes CFL calculation.
        '''
        
        RK = init_rk(self.rk_order)
        a = RK['a']
        b = RK['b']
        rklen = len(b)
        
        # 1. Calculate CFL Time Step ONCE (Static Mesh)
        max_vt = np.max(self.Ikernel.vt)
        if max_vt > 0:
            # Safety factor 0.8 for RK stability
            dt_cfl = 0.8 * self.dz / max_vt
        else:
            dt_cfl = self.dt
            
        # Ensure we don't exceed the global timestep
        dt_cfl = min(dt_cfl, self.dt)
        
        tf = 0
        dM_stages = np.zeros((self.dnum, self.Hlen, self.bins, rklen))
        
        for tt in range(1, self.Tlen):
            
            if pbar:
                pbar.set_description(self.out_text(self.Ikernel.Mbins, self.Ikernel.Mfbins))
                pbar.update(1)
            
            dt_target = self.dt 
            dt_covered = 0.0
            
            # Start State (y_n)
            M_start = self.Ikernel.Mbins.copy()
            M_current = M_start.copy()
            
            # -----------------------------------------------------------------
            # 2. PRE-CALCULATE PHYSICS (Linearized Source)
            # -----------------------------------------------------------------
            # Calculate physics delta over full dt, then normalize to rate.
            self.Ikernel.Mbins = M_current
            self.Ikernel.update_1mom_subgrid() 
            
            M_delta_phys = self.Ikernel.interact_1mom_SS_Final(dt_target)
            rate_phys_frozen = M_delta_phys / dt_target
            
            # -----------------------------------------------------------------
            # 3. ADAPTIVE TRANSPORT LOOP
            # -----------------------------------------------------------------
            while dt_covered < dt_target:
                
                # --- Determine Step Size ---
                # We only need to check Depletion Stability inside the loop
                # (Since CFL is constant)
                
                # Estimate total rate for depletion check
                rate_sed_est = self.get_sed_rate_only(M_current)
                dM_dt_est = rate_phys_frozen + rate_sed_est
                
                # Depletion check: Don't remove more than 50% of mass in one step
                depletion_mask = (M_current > 1e-15) & (dM_dt_est < -1e-20)
                if np.any(depletion_mask):
                    timescale = -M_current[depletion_mask] / dM_dt_est[depletion_mask]
                    dt_dep = 0.5 * np.min(timescale)
                    dt_safe = min(dt_cfl, dt_dep)
                else:
                    dt_safe = dt_cfl
                
                # Clamp to remaining time
                remaining = dt_target - dt_covered
                dt_step = min(dt_safe, remaining)
                dt_step = max(dt_step, 1e-6) # Prevent tiny steps
                
                # --- RK Integration ---
                dM_stages.fill(0.)
                step_size_arr = np.full_like(M_current, dt_step)
                
                for ii in range(rklen):
                    if ii == 0:
                        M_stage = M_current
                    else:
                        k_sum_M = np.sum(a[ii, :ii][None, None, None, :] * dM_stages[:, :, :, :ii], axis=3)
                        M_stage = np.maximum(M_current + step_size_arr * k_sum_M, 0.)
                    
                    # Calculate Sedimentation Rate for this stage
                    # Note: We do NOT force BCs to zero here. We let the flux happen.
                    rate_sed_k = self.get_sed_rate_only(M_stage)
                    dM_stages[:, :, :, ii] = rate_phys_frozen + rate_sed_k
                
                # Final Update
                total_sum_M = np.sum(b[None, None, None, :] * dM_stages, axis=3)
                M_current = np.maximum(M_current + step_size_arr * total_sum_M, 0.)
                
                # --- APPLY FIXED BOUNDARY CONDITION ---
                # Matches your original code: Reset top boundary AFTER the step.
                if self.boundary == 'fixed':
                     M_current[:, 0, :] = M_start[:, 0, :]
                
                dt_covered += dt_step
            
            # Commit final state
            self.Ikernel.Mbins = M_current
            self.Ikernel.update_1mom_subgrid()
            
            if self.save_mask[tt]:
                tf += 1
                self.Mbins[:, :, :, tf] = self.Ikernel.Mbins.copy()

    def get_sed_rate_only(self, M_curr):
        '''
        Calculates pure sedimentation rate (dM/dt).
        Does NOT apply boundary clamping (allows flux to flow).
        '''
        # 1. Push M to Kernel
        self.Ikernel.Mbins = M_curr
        # 2. Update Mfbins
        self.Ikernel.update_1mom_subgrid()
        # 3. Get Flux
        flux = self.Ikernel.Mfbins
        
        rate_sed = np.zeros_like(M_curr)
        
        # Standard Upwind Divergence
        # Top Bin: Flux Out (Assuming 0 in)
        rate_sed[:, 0, :] = -flux[:, 0, :] / self.dz
             
        # Interior Bins
        rate_sed[:, 1:, :] = (flux[:, :-1, :] - flux[:, 1:, :]) / self.dz
        
        return rate_sed


    def run_steady_state_2mom(self, pbar=None):
        ''' 
        Run steady-state bin model using Explicit Runge-Kutta integration.
        Uses ADAPTIVE SUB-STEPPING based on "Fraction of Residence Time".
        '''
        
        # 1. Initialize Runge-Kutta Coefficients
        RK = init_rk(self.rk_order)
        a = RK['a']
        b = RK['b']
        rklen = len(b)
        
        # Total Residence Time Vector (dnum, 1, bins)
        # We perform operations on the full vector self.dh
        total_residence_time = self.dh[:, None, :] 
        
        tf = 0
        
        # Pre-allocate RK stage storage
        dM_stages = np.zeros((self.dnum, self.Hlen, self.bins, rklen))
        dN_stages = np.zeros((self.dnum, self.Hlen, self.bins, rklen))

        # ---------------------------------------------------------------------
        # MAIN LOOP
        # ---------------------------------------------------------------------
        for tt in range(1, self.Tlen):
            
            if pbar:
                pbar.set_description(self.out_text(self.Ikernel.Mbins, self.Ikernel.Mfbins))
                pbar.update(1)

            # Start State (y_n)
            M_current = self.Ikernel.Mbins.copy()
            N_current = self.Ikernel.Nbins.copy()
            
            # Reset Progress (0.0 -> 1.0)
            progress = 0.0
            
            # -----------------------------------------------------------------
            # ADAPTIVE SUB-STEPPING LOOP
            # -----------------------------------------------------------------
            while progress < 1.0:
                
                # --- A. Estimate Stiffness / Safe Fraction ---
                # 1. Get current rates (dM/dt, dN/dt)
                self.Ikernel.Mbins = M_current
                self.Ikernel.Nbins = N_current
                self.Ikernel.update_2mom_subgrid()
                
                # We use the internal rate calculator
                # Note: get_steady_rates returns (Rate), not (Rate * dt)
                # Ensure get_steady_rates calls interact_2mom... correctly
                dM_dt, dN_dt = self.get_steady_rates(M_current, N_current)
                
                # 2. Calculate Timescale for every bin
                # timescale = Mass / Rate
                # We want step_size < timescale
                # i.e., (d_prog * total_residence_time) < timescale
                # So, d_prog < timescale / total_residence_time
                
                safe_mask_M = (M_current > 1e-15) & (np.abs(dM_dt) > 1e-20)
                if np.any(safe_mask_M):
                    # How much 'time' until this bin changes significantly (e.g. 50%)
                    t_scale_M = 0.5 * M_current[safe_mask_M] / np.abs(dM_dt[safe_mask_M])
                    # Convert 'time' to 'fraction of residence time'
                    # We divide by the residence time for those specific bins
                    # Note: We must broadcast total_residence_time to match the mask if needed,
                    # but usually shapes align (dnum, Hlen, bins) vs (dnum, 1, bins).
                    # Let's assume broadcasting works naturally.
                    res_time_masked = np.broadcast_to(total_residence_time, M_current.shape)[safe_mask_M]
                    
                    # Avoid divide by zero if residence time is 0 (shouldn't happen in valid bins)
                    valid_res = res_time_masked > 1e-10
                    
                    if np.any(valid_res):
                        frac_M = t_scale_M[valid_res] / res_time_masked[valid_res]
                        max_frac_M = np.min(frac_M)
                    else:
                        max_frac_M = 1.0
                else:
                    max_frac_M = 1.0

                # Same for Number
                safe_mask_N = (N_current > 1e-15) & (np.abs(dN_dt) > 1e-20)
                if np.any(safe_mask_N):
                    t_scale_N = 0.5 * N_current[safe_mask_N] / np.abs(dN_dt[safe_mask_N])
                    res_time_masked = np.broadcast_to(total_residence_time, N_current.shape)[safe_mask_N]
                    valid_res = res_time_masked > 1e-10
                    if np.any(valid_res):
                        frac_N = t_scale_N[valid_res] / res_time_masked[valid_res]
                        max_frac_N = np.min(frac_N)
                    else:
                        max_frac_N = 1.0
                else:
                    max_frac_N = 1.0
                
                # 3. Determine Step Fraction
                d_prog_safe = min(max_frac_M, max_frac_N)
                
                # Clamp to remaining progress
                remaining = 1.0 - progress
                d_prog = min(d_prog_safe, remaining)
                
                # Prevent infinitesimally small steps (1e-4 minimum fraction)
                d_prog = max(d_prog, 1e-4)
                
                # -------------------------------------------------------------
                # B. CALCULATE VECTORIZED STEP SIZE
                # -------------------------------------------------------------
                # This is the key: step_size is a VECTOR proportional to self.dh
                current_step_vec = d_prog * total_residence_time
                
                # -------------------------------------------------------------
                # C. RK INTEGRATION
                # -------------------------------------------------------------
                dM_stages.fill(0.)
                dN_stages.fill(0.)

                for ii in range(rklen):
                    if ii == 0:
                        M_stage = M_current
                        N_stage = N_current
                    else:
                        k_sum_M = np.sum(a[ii, :ii][None, None, None, :] * dM_stages[:, :, :, :ii], axis=3)
                        k_sum_N = np.sum(a[ii, :ii][None, None, None, :] * dN_stages[:, :, :, :ii], axis=3)
                        
                        M_stage = np.maximum(M_current + current_step_vec * k_sum_M, 0.)
                        N_stage = np.maximum(N_current + current_step_vec * k_sum_N, 0.)

                    # Calculate Rates
                    dM_dt_k, dN_dt_k = self.get_steady_rates(M_stage, N_stage)
                    
                    dM_stages[:, :, :, ii] = dM_dt_k
                    dN_stages[:, :, :, ii] = dN_dt_k
                
                # Final Update for this sub-step
                total_sum_M = np.sum(b[None, None, None, :] * dM_stages, axis=3)
                total_sum_N = np.sum(b[None, None, None, :] * dN_stages, axis=3)
                
                M_current = np.maximum(M_current + current_step_vec * total_sum_M, 0.)
                N_current = np.maximum(N_current + current_step_vec * total_sum_N, 0.)
                
                # Advance Progress
                progress += d_prog
            
            # Commit final state
            self.Ikernel.Mbins = M_current
            self.Ikernel.Nbins = N_current
            self.Ikernel.update_2mom_subgrid()

            # Save Output
            if self.save_mask[tt]:
                tf += 1 
                self.Mbins[:, :, :, tf] = self.Ikernel.Mbins.copy() 
                self.Nbins[:, :, :, tf] = self.Ikernel.Nbins.copy() 
    
        # Post-processing
        if self.int_type == 1:
            self.Mbins = np.swapaxes(self.Mbins, 1, 3)
            self.Nbins = np.swapaxes(self.Nbins, 1, 3) 
            Tout_len = self.Hlen
            Hlen = self.Tout_len
            self.Tout_len = Tout_len
            self.Hlen = Hlen

    def get_steady_rates(self, M_curr, N_curr):
        '''
        Helper function to calculate derivatives for the Steady State RK scheme.
        Equivalent to 'advance_2mom' but for the steady-state loop.
        '''
        # 1. Update the Kernel State with the RK Stage values
        # We must push the 'stage' values into the kernel so it calculates 
        # the correct rates for this specific point in the Runge-Kutta cycle.
        self.Ikernel.Mbins = M_curr
        self.Ikernel.Nbins = N_curr
        
        # Update physics parameters (fall speeds, diameters) based on M_curr/N_curr
        self.Ikernel.update_2mom_subgrid() 
        
        # 2. Calculate Rates
        # Passing 1.0 means we get the rate per unit step (mass/sec * 1.0)
        M_net, N_net = self.Ikernel.interact_2mom_SS_Final(1.0)
        
        return M_net, N_net
    
    
    def run_full_2mom(self, pbar=None):
        ''' 
        Run full 1D bin model (2-Moment) using Adaptive Sub-stepping RK4.
        
        1. Physics: Calculated ONCE over full dt (Linearized Source).
        2. Transport: Calculated iteratively using update_2mom_subgrid (Accuracy).
        3. Stability: Adaptive sub-stepping handles CFL and Depletion (M & N).
        '''
        
        RK = init_rk(self.rk_order)
        a = RK['a']
        b = RK['b']
        rklen = len(b)
        
        # 1. Calculate CFL Time Step ONCE (Static Mesh assumption for max V)
        max_vt = np.max(self.Ikernel.vt)
        if max_vt > 0:
            dt_cfl = 0.8 * self.dz / max_vt
        else:
            dt_cfl = self.dt
        
        # Ensure we don't exceed the global timestep
        dt_cfl = min(dt_cfl, self.dt)
        
        tf = 0
        dM_stages = np.zeros((self.dnum, self.Hlen, self.bins, rklen))
        dN_stages = np.zeros((self.dnum, self.Hlen, self.bins, rklen))
        
        for tt in range(1, self.Tlen):
            
            if pbar:
                pbar.set_description(self.out_text(self.Ikernel.Mbins, self.Ikernel.Mfbins))
                pbar.update(1)
            
            dt_target = self.dt 
            dt_covered = 0.0
            
            # Start State (y_n)
            M_start = self.Ikernel.Mbins.copy()
            N_start = self.Ikernel.Nbins.copy()
            
            M_current = M_start.copy()
            N_current = N_start.copy()
            
            # -----------------------------------------------------------------
            # 2. PRE-CALCULATE PHYSICS (Linearized Source)
            # -----------------------------------------------------------------
            # Update state to ensure physics sees start of step conditions
            self.Ikernel.Mbins = M_current
            self.Ikernel.Nbins = N_current
            self.Ikernel.update_2mom_subgrid() 
            
            # Calculate physics delta over full dt
            M_delta_phys, N_delta_phys = self.Ikernel.interact_2mom_SS_Final(dt_target)
            
            # Convert to constant rates for the sub-steps
            rate_phys_M = M_delta_phys / dt_target
            rate_phys_N = N_delta_phys / dt_target
            
            # -----------------------------------------------------------------
            # 3. ADAPTIVE TRANSPORT LOOP
            # -----------------------------------------------------------------
            while dt_covered < dt_target:
                
                # --- A. Estimate Rates for Stability Check ---
                rate_sed_M, rate_sed_N = self.get_sed_rate_2mom_only(M_current, N_current)
                
                dM_dt_est = rate_phys_M + rate_sed_M
                dN_dt_est = rate_phys_N + rate_sed_N
                
                # --- B. Determine Safe Step (Depletion Check) ---
                
                # Check Mass Depletion
                dep_mask_M = (M_current > 1e-15) & (dM_dt_est < -1e-20)
                if np.any(dep_mask_M):
                    tau_M = -M_current[dep_mask_M] / dM_dt_est[dep_mask_M]
                    dt_dep_M = 0.5 * np.min(tau_M)
                else:
                    dt_dep_M = dt_cfl
                    
                # Check Number Depletion
                dep_mask_N = (N_current > 1e-15) & (dN_dt_est < -1e-20)
                if np.any(dep_mask_N):
                    tau_N = -N_current[dep_mask_N] / dN_dt_est[dep_mask_N]
                    dt_dep_N = 0.5 * np.min(tau_N)
                else:
                    dt_dep_N = dt_cfl
                
                # Safe step is min of CFL and Depletion limits
                dt_safe = min(dt_cfl, dt_dep_M, dt_dep_N)
                
                # Clamp to remaining time
                remaining = dt_target - dt_covered
                dt_step = min(dt_safe, remaining)
                dt_step = max(dt_step, 1e-6) 
                
                # --- C. RK Integration ---
                dM_stages.fill(0.)
                dN_stages.fill(0.)
                
                step_arr_M = np.full_like(M_current, dt_step)
                step_arr_N = np.full_like(N_current, dt_step)
                
                for ii in range(rklen):
                    if ii == 0:
                        M_stage = M_current
                        N_stage = N_current
                    else:
                        # Sum previous stages for M
                        k_sum_M = np.sum(a[ii, :ii][None, None, None, :] * dM_stages[:, :, :, :ii], axis=3)
                        M_stage = np.maximum(M_current + step_arr_M * k_sum_M, 0.)
                        
                        # Sum previous stages for N
                        k_sum_N = np.sum(a[ii, :ii][None, None, None, :] * dN_stages[:, :, :, :ii], axis=3)
                        N_stage = np.maximum(N_current + step_arr_N * k_sum_N, 0.)
                    
                    # Calculate Sedimentation Rate for this stage
                    r_sed_M_k, r_sed_N_k = self.get_sed_rate_2mom_only(M_stage, N_stage)
                    
                    # Total Rate = Frozen Physics + Dynamic Transport
                    dM_stages[:, :, :, ii] = rate_phys_M + r_sed_M_k
                    dN_stages[:, :, :, ii] = rate_phys_N + r_sed_N_k
                
                # --- D. Final Update ---
                total_sum_M = np.sum(b[None, None, None, :] * dM_stages, axis=3)
                total_sum_N = np.sum(b[None, None, None, :] * dN_stages, axis=3)
                
                M_current = np.maximum(M_current + step_arr_M * total_sum_M, 0.)
                N_current = np.maximum(N_current + step_arr_N * total_sum_N, 0.)
                
                # --- APPLY FIXED BOUNDARY CONDITION ---
                # Reset top boundary AFTER the step (Dirichlet-like behavior)
                if self.boundary == 'fixed':
                     M_current[:, 0, :] = M_start[:, 0, :]
                     N_current[:, 0, :] = N_start[:, 0, :]
                
                dt_covered += dt_step
            
            # Commit final state
            self.Ikernel.Mbins = M_current
            self.Ikernel.Nbins = N_current
            self.Ikernel.update_2mom_subgrid()
            
            if self.save_mask[tt]:
                tf += 1
                self.Mbins[:, :, :, tf] = self.Ikernel.Mbins.copy()
                self.Nbins[:, :, :, tf] = self.Ikernel.Nbins.copy()

    def get_sed_rate_2mom_only(self, M_curr, N_curr):
        '''
        Calculates pure sedimentation rate (dM/dt, dN/dt) for 2-moment.
        Does NOT apply boundary clamping (allows flux to flow).
        '''
        # 1. Push State to Kernel
        self.Ikernel.Mbins = M_curr
        self.Ikernel.Nbins = N_curr
        
        # 2. Update Fluxes (Mfbins, Nfbins)
        self.Ikernel.update_2mom_subgrid()
        
        # 3. Get Fluxes
        flux_M = self.Ikernel.Mfbins
        flux_N = self.Ikernel.Nfbins
        
        rate_sed_M = np.zeros_like(M_curr)
        rate_sed_N = np.zeros_like(N_curr)
        
        # Standard Upwind Divergence
        # Top Bin (Flux Out)       
        rate_sed_M[:, 0, :] = -flux_M[:, 0, :] / self.dz
        rate_sed_N[:, 0, :] = -flux_N[:, 0, :] / self.dz
             
        # Interior Bins (In - Out)
        rate_sed_M[:, 1:, :] = (flux_M[:, :-1, :] - flux_M[:, 1:, :]) / self.dz
        rate_sed_N[:, 1:, :] = (flux_N[:, :-1, :] - flux_N[:, 1:, :]) / self.dz
        
        return rate_sed_M, rate_sed_N
    

                
    def _run_core(self,pbar):  
        
        # If running one moment (mass) only
        if self.moments == 1:
            
            if self.int_type==0:
                self.run_full_1mom(pbar)
                
            else:
                self.run_steady_state_1mom(pbar)
                     
        # If running two moments (mass and number)
        if self.moments == 2:

            if self.int_type==0:
                self.run_full_2mom(pbar)
                
            else:
                self.run_steady_state_2mom(pbar)
        
    
    def run(self):
        ''' 
        Run bin model
        '''
        time_start = datetime.now()

        try:
                
            # Start up parallel pool if running in parallel
            # if (self.parallel) and (self.pool is None):
            #     self.activate_parallel()
                   
            if self.progress:
                
                if 'ipykernel_launcher.py' in sys.argv[0]:
                    ncols = 800
                else:
                    ncols = None

                #with tqdm(total=self.Tlen-1,position=0,leave=True,mininterval=0,miniters=1,desc="Running 1D spectral bin model") as pbar: # ORIGINAL
                with tqdm(total=self.Tlen-1,position=0,leave=True,mininterval=0.2,miniters=None,ncols=ncols,desc="Running 1D spectral bin model") as pbar:
                    self._run_core(pbar)
                    
            else:
                pbar = None
                self._run_core(pbar)

            # 4D subgrid linear (or uniform) distribution parameters
            self.diagnose_subgrid()

            # Calculate microphysical variables
            self.calc_micro()
        
            # Diagnose radar variables for final arrays
            if self.radar:
                self.calc_radar()         
            
        except Exception as E:
           print('\n{}'.format(E))
 
        finally:
            
            if pbar is not None:
                pbar.close()
            
            #Clean up temporary memory mapped files and potentially other things too.
            self.clean_up()

        time_end = datetime.now() 
        print('Model Complete! Time Elapsed = {:.2f} min'.format((time_end-time_start).total_seconds()/60.))


def latex_check():
    """
    Verifies that all executables required by Matplotlib's usetex engine are in the system PATH.
    """
    # 1. Check for the LaTeX compiler
    has_latex = shutil.which("latex") is not None
    
    # 2. Check for dvipng (required for rendering to screen / PNGs)
    has_dvipng = shutil.which("dvipng") is not None
    
    # 3. Check for Ghostscript (required for vector formats like PDF/SVG)
    # Note: Windows uses 'gswin64c' or 'gswin32c', Unix uses 'gs'
    #has_gs = any(shutil.which(cmd) for cmd in ["gs", "gswin64c", "gswin32c"])
    
    return has_latex and has_dvipng




    # def write_netcdf_ORIGINAL(self,filename):
    #     '''
        
    #     Parameters
    #     ----------
    #     filename : STRING
    #         Writes spectral_1d object attributes and variables to a netcdf file.

    #     '''
        
    #     with Dataset(filename,'w',format='NETCDF4') as file_nc:
            
    #         # Write attributes
    #         file_nc.bins = self.bins
    #         file_nc.sbin = self.sbin
    #         file_nc.dnum = self.dnum
    #         file_nc.moments = self.moments
    #         file_nc.ptype = self.ptype
    #         file_nc.dt = self.dt 
    #         file_nc.dz = self.dz 
    #         file_nc.Hlen = self.Hlen
    #         file_nc.Tlen = self.Tlen
    #         file_nc.Tout_len = self.Tout_len
    #         file_nc.rk_order = self.rk_order
    #         file_nc.adv_order = self.adv_order
    #         file_nc.int_type = self.int_type
    #         file_nc.kernel = self.kernel
    #         file_nc.radar = 1*self.radar
    #         file_nc.indb = self.indb 
    #         file_nc.indc = self.indc
    #         file_nc.dist_var = self.dist_var
    #         file_nc.mu0 = self.mu0
    #         file_nc.Dm0 = self.Dm0
    #         file_nc.Nt0 = self.Nt0
    #         file_nc.D1 = self.D1 
    #         file_nc.lamf = self.lamf
    #         file_nc.x0 = self.dist0.x0
    #         file_nc.Tc = self.Tc
    #         file_nc.Ecol = self.Ecol
    #         file_nc.Es = self.Es
    #         file_nc.Eb = self.Eb 
    #         file_nc.Eagg = self.Eagg 
    #         file_nc.Ebr = self.Ebr 
    #         file_nc.Ecb = self.Ecb
    #         file_nc.gam_norm = 1*self.gam_norm
    #         if self.boundary is None:
    #             file_nc.boundary = 'None'
    #         else:
    #             file_nc.boundary = self.boundary
    #         file_nc.ztop = self.ztop 
    #         file_nc.zbot = self.zbot
    #         file_nc.tmax = self.tmax
    #         file_nc.t = self.t
    #         file_nc.parallel = 1*self.parallel
    #         file_nc.n_jobs = self.n_jobs
                
    #         file_nc.frag_dist = self.frag_dist
 
    #         # Create dimensions
    #         if self.int_type==0:
    #             file_nc.createDimension('time_out',self.Tout_len)
    #             file_nc.createDimension('time',self.Tlen)
    #             file_nc.createDimension('height',self.Hlen)
    #             t = file_nc.createVariable('t','f4',('time',))
    #             tout = file_nc.createVariable('tout','f4',('time_out',))
    #             z = file_nc.createVariable('z','f4',('height',))
    #         elif self.int_type==1:
    #             file_nc.createDimension('height',self.Tlen)
    #             z = file_nc.createVariable('z','f4',('height',))
    #         elif self.int_type==2:
    #             file_nc.createDimension('time',self.Tlen)
    #             t = file_nc.createVariable('t','f4',('time',))

    #         file_nc.createDimension('dists',self.dnum)
    #         file_nc.createDimension('bins',self.bins)
            
    #         # Put coordinates into 1D arrays
    #         xbins = file_nc.createVariable('xbins','f4',('bins',))
    #         xi1 = file_nc.createVariable('xi1','f4',('bins',))
    #         xi2 = file_nc.createVariable('xi2','f4',('bins',))
            
    #         if self.int_type==0:
    #             tout[:] = self.tout
    #             t[:] = self.t
    #             z[:] = self.z 
    #             t.units = 'seconds'
    #             t.description = 'time'
    #             tout.units = 'seconds'
    #             tout.description = 'Output time'
    #             z.units = 'meters'
    #             z.description = 'Gridbox Height'
    #         elif self.int_type==1:
    #             z[:] = self.z 
    #             z.units = 'meters'
    #             z.description = 'Gridbox Height'            
    #         elif self.int_type==2:
    #             t[:] = self.t
    #             t.units = 'seconds'
    #             t.description = 'Output time'
            
    #         xbins[:] = self.xbins
    #         xi1[:] = self.xedges[:-1]
    #         xi2[:] = self.xedges[1:]
            
    #         xbins.units = 'g'
    #         xi1.units = 'g'
    #         xi2.units = 'g'
            
    #         xbins.description = 'Mass grid midpoint'
    #         xi1.description = 'Mass grid left bin edge'
    #         xi2.description = 'Mass grid right bin edge'
            
    #         # Put distributions into 4D arrays
    #         for dd in range(self.dnum):
    #             # Create group
    #             dist_dd = file_nc.createGroup('dist{}'.format(dd+1))
                
    #             # Add attributes
    #             dist_dd.habit = self.habit_list[dd]
    #             dist_dd.am = self.dists[dd,0].am
    #             dist_dd.bm = self.dists[dd,0].bm
    #             dist_dd.av = self.dists[dd,0].av
    #             dist_dd.bv = self.dists[dd,0].bv
    #             dist_dd.arho = self.dists[dd,0].arho
    #             dist_dd.brho = self.dists[dd,0].brho
                
    #             # Full model
    #             if self.int_type==0:
                    
    #                 # Create Array Variables
    #                 Mbins = dist_dd.createVariable('Mbins','f8',('height','time_out','bins'))
    #                 Nbins = dist_dd.createVariable('Nbins','f8',('height','time_out','bins'))
    #                 x1    = dist_dd.createVariable('x1','f8',('height','time_out','bins'))
    #                 x2    = dist_dd.createVariable('x2','f8',('height','time_out','bins'))
    #                 aki   = dist_dd.createVariable('aki','f8',('height','time_out','bins'))
    #                 cki   = dist_dd.createVariable('cki','f8',('height','time_out','bins'))
                    
    #                 Mbins[:] = np.array([[self.full[dd,hh,tt].Mbins for tt in range(self.Tout_len)] for hh in range(self.Hlen)])
    #                 Nbins[:] = np.array([[self.full[dd,hh,tt].Nbins for tt in range(self.Tout_len)] for hh in range(self.Hlen)])
    #                 aki[:]   = np.array([[self.full[dd,hh,tt].aki for tt in range(self.Tout_len)] for hh in range(self.Hlen)])
    #                 cki[:]   = np.array([[self.full[dd,hh,tt].cki for tt in range(self.Tout_len)] for hh in range(self.Hlen)])
                    
    #                 if self.radar:
    #                     Zh = dist_dd.createVariable('Zh','f8',('height','time_out','bins'))
    #                     Zv = dist_dd.createVariable('Zv','f8',('height','time_out','bins'))
    #                     Kdp = dist_dd.createVariable('Kdp','f8',('height','time_out','bins'))
    #                     Zhhvv_real = dist_dd.createVariable('Zhhvv_real','f8',('height','time_out','bins'))
    #                     Zhhvv_imag = dist_dd.createVariable('Zhhvv_imag','f8',('height','time_out','bins'))
                        
    #                     Zh.units = 'mm^6/m^3'
    #                     Zv.units = 'mm^6/m^3'
    #                     Kdp.units = 'deg/km'
    #                     Zhhvv_real.units = 'mm^6/m^3'
    #                     Zhhvv_imag.units = 'mm^6/m^3'
                        
    #                     Zh.description = 'bin radar horizontal reflectivity in linear units'
    #                     Zv.description = 'bin radar vertical reflectivity in linear units'
    #                     Kdp.description = 'bin radar specific differential phase'
    #                     Zhhvv_real.description = 'real component of bin radar cross-to-copolar reflectivity in linear units'
    #                     Zhhvv_imag.description = 'imaginary component of bin radar cross-to-copolar reflectivity in linear units'
                        
    #                     Zh[:] = np.array([[self.full[dd,hh,tt].Zh for tt in range(self.Tout_len)] for hh in range(self.Hlen)])
    #                     Zv[:] = np.array([[self.full[dd,hh,tt].Zv for tt in range(self.Tout_len)] for hh in range(self.Hlen)])
    #                     Kdp[:]   = np.array([[self.full[dd,hh,tt].Kdp for tt in range(self.Tout_len)] for hh in range(self.Hlen)])
    #                     Zhhvv_full   = np.array([[self.full[dd,hh,tt].zhhvv for tt in range(self.Tout_len)] for hh in range(self.Hlen)])
                             
    #                     Zhhvv_real[:] = np.real(Zhhvv_full)
    #                     Zhhvv_imag[:] = np.imag(Zhhvv_full)
                
    #             # Steady State model
    #             elif self.int_type==1:
    #                 # Create Array Variables
    #                 Mbins = dist_dd.createVariable('Mbins','f8',('height','bins'))
    #                 Nbins = dist_dd.createVariable('Nbins','f8',('height','bins'))
    #                 x1    = dist_dd.createVariable('x1','f8',('height','bins'))
    #                 x2    = dist_dd.createVariable('x2','f8',('height','bins'))
    #                 aki   = dist_dd.createVariable('aki','f8',('height','bins'))
    #                 cki   = dist_dd.createVariable('cki','f8',('height','bins'))
                    
    #                 Mbins[:] = np.array([self.full[dd,hh].Mbins for hh in range(self.Tlen)])
    #                 Nbins[:] = np.array([self.full[dd,hh].Nbins for hh in range(self.Tlen)])
    #                 aki[:]   = np.array([self.full[dd,hh].aki for hh in range(self.Tlen)])
    #                 cki[:]   = np.array([self.full[dd,hh].cki for hh in range(self.Tlen)])
                    
                    
    #                 if self.radar:
    #                     Zh = dist_dd.createVariable('Zh','f8',('height','bins'))
    #                     Zv = dist_dd.createVariable('Zv','f8',('height','bins'))
    #                     Kdp = dist_dd.createVariable('Kdp','f8',('height','bins'))
    #                     Zhhvv_real = dist_dd.createVariable('Zhhvv_real','f8',('height','bins'))
    #                     Zhhvv_imag = dist_dd.createVariable('Zhhvv_imag','f8',('height','bins'))
                        
    #                     Zh.units = 'mm^6/m^3'
    #                     Zv.units = 'mm^6/m^3'
    #                     Kdp.units = 'deg/km'
    #                     Zhhvv_real.units = 'mm^6/m^3'
    #                     Zhhvv_imag.units = 'mm^6/m^3'
                        
    #                     Zh.description = 'bin radar horizontal reflectivity in linear units'
    #                     Zv.description = 'bin radar vertical reflectivity in linear units'
    #                     Kdp.description = 'bin radar specific differential phase'
    #                     Zhhvv_real.description = 'real component of bin radar cross-to-copolar reflectivity in linear units'
    #                     Zhhvv_imag.description = 'imaginary component of bin radar cross-to-copolar reflectivity in linear units'
                        
    #                     Zh[:]       = np.array([self.full[dd,hh].Zh for hh in range(self.Tlen)])
    #                     Zv[:]       = np.array([self.full[dd,hh].Zv for hh in range(self.Tlen)])
    #                     Kdp[:]      = np.array([self.full[dd,hh].Kdp for hh in range(self.Tlen)])
    #                     Zhhvv_full  = np.array([self.full[dd,hh].zhhvv for hh in range(self.Tlen)])
                             
    #                     Zhhvv_real[:] = np.real(Zhhvv_full)
    #                     Zhhvv_imag[:] = np.imag(Zhhvv_full)
                    
                 
    #             # Box model
    #             elif self.int_type==2:
    #                 # Create Array Variables
    #                 Mbins = dist_dd.createVariable('Mbins','f8',('time','bins'))
    #                 Nbins = dist_dd.createVariable('Nbins','f8',('time','bins'))
    #                 x1    = dist_dd.createVariable('x1','f8',('time','bins'))
    #                 x2    = dist_dd.createVariable('x2','f8',('time','bins'))
    #                 aki   = dist_dd.createVariable('aki','f8',('time','bins'))
    #                 cki   = dist_dd.createVariable('cki','f8',('time','bins'))
                    
    #                 Mbins[:] = np.array([self.full[dd,tt].Mbins for tt in range(self.Tlen)])
    #                 Nbins[:] = np.array([self.full[dd,tt].Nbins for tt in range(self.Tlen)])
    #                 aki[:]   = np.array([self.full[dd,tt].aki for tt in range(self.Tlen)])
    #                 cki[:]   = np.array([self.full[dd,tt].cki for tt in range(self.Tlen)])
                    
    #                 if self.radar:
    #                     Zh = dist_dd.createVariable('Zh','f8',('time','bins'))
    #                     Zv = dist_dd.createVariable('Zv','f8',('time','bins'))
    #                     Kdp = dist_dd.createVariable('Kdp','f8',('time','bins'))
    #                     Zhhvv_real = dist_dd.createVariable('Zhhvv_real','f8',('time','bins'))
    #                     Zhhvv_imag = dist_dd.createVariable('Zhhvv_imag','f8',('time','bins'))
                        
    #                     Zh.units = 'mm^6/m^3'
    #                     Zv.units = 'mm^6/m^3'
    #                     Kdp.units = 'deg/km'
    #                     Zhhvv_real.units = 'mm^6/m^3'
    #                     Zhhvv_imag.units = 'mm^6/m^3'
                        
    #                     Zh.description = 'bin radar horizontal reflectivity in linear units'
    #                     Zv.description = 'bin radar vertical reflectivity in linear units'
    #                     Kdp.description = 'bin radar specific differential phase'
    #                     Zhhvv_real.description = 'real component of bin radar cross-to-copolar reflectivity in linear units'
    #                     Zhhvv_imag.description = 'imaginary component of bin radar cross-to-copolar reflectivity in linear units'
                        
    #                     Zh[:] = np.array([self.full[dd,tt].Zh for tt in range(self.Tlen)])
    #                     Zv[:] = np.array([self.full[dd,tt].Zv for tt in range(self.Tlen)])
    #                     Kdp[:]   = np.array([self.full[dd,tt].Kdp for tt in range(self.Tlen)])
    #                     Zhhvv_full   = np.array([self.full[dd,tt].zhhvv for tt in range(self.Tlen)])
                             
    #                     Zhhvv_real[:] = np.real(Zhhvv_full)
    #                     Zhhvv_imag[:] = np.imag(Zhhvv_full)

    #             else:
                    
    #                 print('int_type={} not considered'.format(self.int_type))
    #                 raise Exception()
                    
    #             Mbins.units = 'g'
    #             Nbins.units = '#'
    #             x1.units = 'g'
    #             x2.units = 'g'
    #             aki.units = '#/g?'
    #             cki.units = '#/g?'
                
    #             Mbins.description = 'Total Bin Mass'
    #             Nbins.description = 'Total Bin Number'
    #             x1.description = 'Subgrid linear distribution left bin mass edge'
    #             x2.description = 'Subgrid linear distribution right bin mass edge'
    #             aki.description = 'Subgrid linear mass distribution slope'
    #             cki.description = 'Subgrid linear mass distribution intercept'   





    # def plot_dists_height_ORIGINAL(self,tind=-1,plot_habits=False):
        
    #     #plt.rcParams['text.usetex'] = True
        
    #     plt.rc('text', usetex=True)
    #     plt.rc('font', family='serif')
    #     plt.rc('xtick', labelsize=22) 
    #     plt.rc('ytick', labelsize=22) 
        
    #     z = self.z/1000.
        
    #     z_lvls = np.arange(np.max(z),np.min(z)-1.,-1.)
        
    #     fig, ax = plt.subplots(len(z_lvls),1,figsize=(6,10),sharey=True,sharex=True)
        
    #     mbins = self.dist0.xbins.copy() 
        
    #     if self.int_type==0:
    #         primary_init  = self.full[0,0,tind]
    #     else:
    #         primary_init = self.full[0,0]
        
    #    # primary_init  = self.full[0,0]
        
    #     xp1 = primary_init.x1
    #     xp2 = primary_init.x2
    #     ap = primary_init.aki
    #     cp = primary_init.cki
        
    #     prefN_p = primary_init.am**(1./primary_init.bm)*primary_init.bm*mbins**(1.-1./primary_init.bm)

    #     xbins = (mbins/primary_init.am)**(1./primary_init.bm)
        
    #     nN_init = prefN_p*np.heaviside(mbins-xp1,1)*np.heaviside(xp2-mbins,1)*(ap*mbins+cp)
        
    #     ax[0].plot(xbins,nN_init,'k')

    #     ax[0].set_yscale('log')
    #     ax[0].set_xscale('linear')
    #     ax[0].set_ylim(bottom=0.001)
        
    #     ax[0].set_title('Height = {} km'.format(z_lvls[0]),fontsize=26)
    #     #ax[0].set_ylabel(r'Number Density (1/cm$^{3}$ 1/mm)',fontsize=16)
    #     ax[0].set_ylabel(r'n(D) (1/cm$^{3}$ 1/mm)',fontsize=16)
        
    #     ax[0].axes.tick_params(labelsize=20)
    #     ax[0].set_xlim((0.,5.))
        
    #     for hh in range(1,len(z_lvls)):
        
    #         zind = np.nonzero(z==z_lvls[hh])[0][0]
             
    #         nN_final = np.full((self.dnum,self.bins),np.nan)
            
            
    #         for d1 in range(self.dnum):

                
    #             if self.int_type==0:
    #                 xp1_final = self.full[d1,zind,tind].x1
    #                 xp2_final = self.full[d1,zind,tind].x2
    #                 ap_final  = self.full[d1,zind,tind].aki
    #                 cp_final  = self.full[d1,zind,tind].cki
    #                 prefN =self.full[d1,zind,tind].am**(1./self.full[d1,zind,tind].bm)*self.full[d1,zind,tind].bm*mbins**(1.-1./self.full[d1,zind,tind].bm)
                    
    #             else:
    #                 xp1_final = self.full[d1,zind].x1
    #                 xp2_final = self.full[d1,zind].x2
    #                 ap_final  = self.full[d1,zind].aki
    #                 cp_final  = self.full[d1,zind].cki
    #                 prefN =self.full[d1,zind].am**(1./self.full[d1,zind].bm)*self.full[d1,zind].bm*mbins**(1.-1./self.full[d1,zind].bm)
                
    #             nN_final[d1,:] = prefN*np.heaviside(mbins-xp1_final,1)*np.heaviside(xp2_final-mbins,1)*(ap_final*mbins+cp_final)

    #             if plot_habits:
    #                 ax[hh].plot(xbins,nN_final[d1,:],label='dist {}'.format(d1+1))

    #         if plot_habits:
    #             ax[hh].plot(xbins,np.nansum(nN_final,axis=0),color='k')
    #         else:
    #             ax[hh].plot(xbins,np.nansum(nN_final,axis=0),color='k',label='total')
                
    #         ax[hh].set_yscale('log')
    #         ax[hh].set_xscale('linear')
    #         ax[hh].set_ylim(bottom=0.001)
            
    #         ax[hh].set_title('Height = {} km'.format(z_lvls[hh]),fontsize=26)
    #         #ax[hh].set_ylabel(r'Number Density (1/cm$^{3}$ 1/mm)',fontsize=16)
    #         ax[hh].set_ylabel(r'n(D) (1/cm$^{3}$ 1/mm)',fontsize=16)
            
    #         ax[hh].axes.tick_params(labelsize=20)
    #         ax[hh].set_xlim((0.,5.))

        
    #     ax[0].set_ylim((1e-5,1e5))
    #     ax[-1].set_xlabel('Equivolume Diameter (mm)',fontsize=22)
    #     if plot_habits:
    #         ax[-1].legend()
        
    #     plt.tight_layout()
        
    #     return fig, ax

    # def plot_dists_ORIGINAL(self,tind=-1,hind=-1,x_axis='mass',y_axis='mass',xscale='log',yscale='linear',distscale='log',normbin=False,scott_solution=False,feingold_solution=False,plot_habits=False,ax=None,lstyle='-',lcolor='k'):

    #     plt.rc('text', usetex=True)
    #     plt.rc('font', family='serif')
    #     plt.rc('xtick', labelsize=26) 
    #     plt.rc('ytick', labelsize=26)   
      
    #     if ax is None:
    #         ax_switch = True 
    #     else:
    #         ax_switch = False
        
    #     # NOTE: probably need to figure out how to deal with x_axis='size' when
    #     # am and bm parameters are different for each habit.
        
    #     if ax is None:
    #         fig, ax = plt.subplots(2,1,figsize=((8,10)),sharex=True)
        
    #    # if (len(self.t)>1) & (len(self.z)==1):
            
        
    #     print('Plotting distributions...')
        
    #     if self.int_type==0:
    #         primary_init  = self.full[0,0,0]
    #     else:
    #         primary_init = self.full[0,0]
        
    #     mbins = primary_init.xbins.copy() 
        
    #     if normbin:
    #         dxbins = np.log10(primary_init.xedges[1:]/primary_init.xedges[:-1])
    #     else:
    #         dxbins = primary_init.dxbins.copy()
        

        
    #     xp1 = primary_init.x1
    #     xp2 = primary_init.x2
    #     ap = primary_init.aki
    #     cp = primary_init.cki
    #     Mbins_init = primary_init.Mbins 
    #     Nbins_init = primary_init.Nbins
        
    #    # xbins = np.full((self.dnum,self.bins),np.nan)
    #     prefN = np.full((self.dnum,self.bins),np.nan)
    #     prefM = np.full((self.dnum,self.bins),np.nan)
    #     x1_final = np.full((self.dnum,self.bins),np.nan)
    #     x2_final = np.full((self.dnum,self.bins),np.nan)
    #     ak_final = np.full((self.dnum,self.bins),np.nan)
    #     ck_final = np.full((self.dnum,self.bins),np.nan)
    #     Mbins_final = np.full((self.dnum,self.bins),np.nan)
    #     Nbins_final = np.full((self.dnum,self.bins),np.nan)
    #     bm = np.full((self.dnum,),np.nan)
    #     am = np.full((self.dnum,),np.nan)

    #     if self.int_type==0:
    #         f_label = '{:.1f} km | {:.2f} min.'.format(self.z[hind]/1000.,self.t[tind]/60.)
    #         for d1 in range(self.dnum):
    #             x1_final[d1,:] = self.full[d1,hind,tind].x1 
    #             x2_final[d1,:] = self.full[d1,hind,tind].x2 
    #             ak_final[d1,:] = self.full[d1,hind,tind].aki 
    #             ck_final[d1,:] = self.full[d1,hind,tind].cki
    #             Mbins_final[d1,:] = self.full[d1,hind,tind].Mbins
    #             Nbins_final[d1,:] = self.full[d1,hind,tind].Nbins
                
    #             bm[d1] = self.full[d1,hind,tind].bm 
    #             am[d1] = self.full[d1,hind,tind].am 

                
    #     elif self.int_type==1:
    #         f_label = '{:.1f} km'.format(self.z[hind]/1000.)
    #         for d1 in range(self.dnum):
    #             x1_final[d1,:] = self.full[d1,hind].x1 
    #             x2_final[d1,:] = self.full[d1,hind].x2 
    #             ak_final[d1,:] = self.full[d1,hind].aki 
    #             ck_final[d1,:] = self.full[d1,hind].cki
    #             Mbins_final[d1,:] = self.full[d1,hind].Mbins.copy()
    #             Nbins_final[d1,:] = self.full[d1,hind].Nbins.copy()
                
    #             bm[d1] = self.full[d1,hind].bm 
    #             am[d1] = self.full[d1,hind].am 
                
    #     elif self.int_type==2:
    #         f_label = '{:.2f} min.'.format(self.t[tind]/60.)
    #         for d1 in range(self.dnum):
    #             x1_final[d1,:] = self.full[d1,tind].x1 
    #             x2_final[d1,:] = self.full[d1,tind].x2 
    #             ak_final[d1,:] = self.full[d1,tind].aki 
    #             ck_final[d1,:] = self.full[d1,tind].cki
    #             Mbins_final[d1,:] = self.full[d1,tind].Mbins.copy()
    #             Nbins_final[d1,:] = self.full[d1,tind].Nbins.copy()
                
    #             bm[d1] = self.full[d1,tind].bm 
    #             am[d1] = self.full[d1,tind].am 
            
   
    #     # Distscale toggles between dN/dlog(m) plots and dN/dm plots, for example.
    #     if distscale=='log':
        
    #         if x_axis=='mass': # plot dN/dlog(m) and dM/dlog(m)
            
    #             for d1 in range(self.dnum):
    #                 prefN[d1,:] = mbins*np.log(10)   
    #                 prefM[d1,:] = mbins**2*np.log(10)
                    
    #                 #xbins[d1,:] = mbins.copy()
    #             xbins = mbins.copy()
    #             if normbin:
    #                 ylabel_num = r'd$P_{N}$/dlog(m)'
    #                 ylabel_mass = r'd$P_{M}$/dlog(m)'
                    
    #             else:
    #                 ylabel_num = r'dN/dlog(m)'
    #                 ylabel_mass = r'dM/dlog(m)'
                
    #             xlabel = r'log(m) [log(g)]'
                
    #         elif x_axis=='size':  # plot dN/dlog(D) and dM/dlog(D)
                
    #             for d1 in range(self.dnum):
                    
    #                 prefN[d1,:] = mbins*bm[d1]*np.log(10) 
    #                 prefM[d1,:] = mbins**2*bm[d1]*np.log(10) 
                
    #                 #xbins[d1,:] = (mbins/self.full[d1,0].am)**(1./self.full[d1,0].bm)
               
    #             xbins = (mbins/am[0])**(1./bm[0])
   
    #             if normbin:
    #                 ylabel_num = r'd$P_{N}$/dlog(D)'
    #                 ylabel_mass = r'd$P_{M}$/dlog(D)'
                    
    #             else:
    #                 ylabel_num = r'dN/dlog(D)'
    #                 ylabel_mass = r'dM/dlog(D)'
                
    #             xlabel = r'log(D) [log(mm)]'
                
    #             # Set xtickparams to something readable?
    #             #ax[1].set_xticks(2.**np.arange(-3,8,1))
    #             #ax[1].set_xticklabels(2.**np.arange(-3,8,1))
                
    #     else: # Linear plots of form dN/dm or dN/dD, for example.
            
    #         if x_axis=='mass':  # Linear plots of form dN/dm
            
    #             for d1 in range(self.dnum):
    #                 prefN[d1,:] = np.ones_like(mbins)
    #                 prefM[d1,:] = mbins.copy()

    #                 #xbins[d1,:] = mbins.copy()
    #             xbins = mbins.copy()
                
    #             if normbin:
    #                 ylabel_num = r'd$P_{N}$/dm'
    #                 ylabel_mass = r'd$P_{M}$/dm'
                    
    #             else:
    #                 ylabel_num = r'dN/dm'
    #                 ylabel_mass = r'dM/dm'
                
    #             xlabel = r'log(m) [log(g)]'
                
    #         elif x_axis=='size': # Linear plots of form dN/dD
                
    #             for d1 in range(self.dnum):
    #                 # !!!CHECK
    #                 prefN[d1,:] = am[d1]**(1./bm[d1])*bm[d1]*mbins**(1.-1./bm[d1])
    #                 prefM[d1,:] = am[d1]**(1./bm[d1])*bm[d1]*mbins**(2.-1./bm[d1])

    #                 #xbins[d1,:] = (mbins/self.full[d1,0].am)**(1./self.full[d1,0].bm)
    #             xbins = (mbins/am[0])**(1./bm[0])
                
    #             if normbin:
    #                 ylabel_num = r'd$P_{N}$/dD'
    #                 ylabel_mass = r'd$P_{M}$/dD'
    #             else:
    #                 ylabel_num = r'dN/dD'
    #                 ylabel_mass = r'dM/dD'
         
    #     if normbin:
    #         #Nbins_init[Nbins_init<1e-6] = np.nan
    #         #Mbins_init[Mbins_init<1e-6] = np.nan
    #         prefN_init = prefN[0,:]/np.nansum(Nbins_init)
    #         prefM_init = prefM[0,:]/np.nansum(Mbins_init)
            
    #        # Nbins_final[Nbins_final<1e-6] = np.nan
    #        # Mbins_final[Mbins_final<1e-6] = np.nan
    #         prefN_final = prefN/np.nansum(Nbins_final)
    #         prefM_final = prefM/np.nansum(Mbins_final)
            
    #     else:
    #         prefN_init = prefN[0,:].copy()
    #         prefM_init = prefM[0,:].copy()
            
    #         prefN_final = prefN.copy()
    #         prefM_final = prefM.copy()
            
    #     # nN_init = prefN[0,:]*np.heaviside(mbins-xp1,1)*np.heaviside(xp2-mbins,1)*(ap*mbins+cp)
    #     # nM_init = prefM[0,:]*np.heaviside(mbins-xp1,1)*np.heaviside(xp2-mbins,1)*(ap*mbins+cp)

    #     # nN_final = prefN*np.heaviside(mbins[None,:]-x1_final,1)*np.heaviside(x2_final-mbins[None,:],1)*(ak_final*mbins[None,:]+ck_final)
    #     # nM_final = prefM*np.heaviside(mbins[None,:]-x1_final,1)*np.heaviside(x2_final-mbins[None,:],1)*(ak_final*mbins[None,:]+ck_final)


    #     nN_init = prefN_init*np.heaviside(mbins-xp1,1)*np.heaviside(xp2-mbins,1)*(ap*mbins+cp)
    #     nM_init = prefM_init*np.heaviside(mbins-xp1,1)*np.heaviside(xp2-mbins,1)*(ap*mbins+cp)

    #     nN_final = prefN_final*np.heaviside(mbins[None,:]-x1_final,1)*np.heaviside(x2_final-mbins[None,:],1)*(ak_final*mbins[None,:]+ck_final)
    #     nM_final = prefM_final*np.heaviside(mbins[None,:]-x1_final,1)*np.heaviside(x2_final-mbins[None,:],1)*(ak_final*mbins[None,:]+ck_final)


    #     if self.gam_norm:
    #         nM_init /= 1000. 
    #         nM_final /= 1000.
            

    #     if xscale=='log':
    #         x = np.log10(xbins)
            
    #         if (x_axis=='size'):
    #             xlabel = r'log(D) [log(mm)]'
    #         elif (x_axis=='mass'):
    #             xlabel = r'log(m) [log(g)]'
            
    #     elif xscale=='linear':
    #         x = xbins.copy()
    #         ax[0].set_xlim((0.,5.))
    #        # ax[1].set_ylim((0.,10.))
             
    #         if (x_axis=='size'):
    #             xlabel = r'D (mm)'
    #         elif (x_axis=='mass'):
    #             xlabel = r'm (g)'
        
    #     if ax_switch:
    #         ax[0].plot(x,nN_init,':k',linewidth=2,label='initial')
    #     ax[0].plot(x,np.nansum(nN_final,axis=0),linestyle=lstyle,color=lcolor,linewidth=2,label=f_label)
    #     if plot_habits:
    #         for d1 in range(self.dnum):
    #             ax[0].plot(x,nN_final[d1,:],linewidth=2,label='dist {}'.format(d1+1))
            
    #     # Factor of 1000 comes from converting g to g/m^3
    #     if ax_switch:
    #         ax[1].plot(x,1000.*nM_init,':k',linewidth=2,label='initial')
    #     ax[1].plot(x,1000.*np.nansum(nM_final,axis=0),linestyle=lstyle,color=lcolor,linewidth=2,label=f_label)
    #     if plot_habits:
    #         for d1 in range(self.dnum):
    #             ax[1].plot(x,1000.*nM_final[d1,:],linewidth=2,label='dist {}'.format(d1+1))

    #     ax[0].set_ylabel(ylabel_num,fontsize=26)
    #     ax[1].set_ylabel(ylabel_mass,fontsize=26)
        
    #     ax[1].set_xlabel(xlabel,fontsize=26)
        
    #     if yscale=='log':
    #         ax[0].set_yscale('log')
    #         ax[1].set_yscale('log')
    #         ax[0].set_ylim((1e-5,max(nN_init.max(),1000.*nN_final.max())))
    #         ax[1].set_ylim((1e-5,max(nM_init.max(),1000.*nM_final.max())))

    #     #print('number test=',np.nansum(mbins*n_init*(np.log(medges[1:])-np.log(medges[:-1]))))
    #    # print('mass test=',np.nansum(mbins**2*1000.*n_init*(np.log(medges[1:])-np.log(medges[:-1]))))
        
    #     if (scott_solution & (self.int_type==2)):
            
    #         kernel_type = self.kernel
            
    #         #if not (hasattr(self,'n_scott')):
    #         self.n_scott = Scott_dists(self.xbins,self.Eagg,self.mu0+1,self.t,kernel_type=kernel_type)
        
    #         n_scott_new = prefN[0,:]*self.n_scott[:,tind]
    #         nm_scott_new = prefM[0,:]*self.n_scott[:,tind]
        
    #         if normbin:
    #             n_scott_new /= np.nansum(n_scott_new*dxbins)
    #             nm_scott_new /= np.nansum(nm_scott_new*dxbins)
                
    #             #print('n_scott_new=',np.nansum(n_scott_new*dxbins))

        
    #         #ax[0].plot(x,prefN[0,:]*self.n_scott[:,tind],':r',linewidth=2,label=f_label+ " analytical")
    #         ax[0].plot(x,n_scott_new,':r',linewidth=2,label=f_label+ " analytical")
            
    #         #ax[1].plot(x,1000.*prefM[0,:]*self.n_scott[:,tind],':r',label=f_label+ "analytical")
            
    #         #ax[1].plot(x,prefM[0,:]*self.n_scott[:,tind],':r',linewidth=2,label=f_label+ " analytical")
    #         ax[1].plot(x,nm_scott_new,':r',linewidth=2,label=f_label+ " analytical")
        
    #     if (feingold_solution & (self.int_type==2)):
            
    #         kernel_type = self.kernel
            
    #         C = self.Eagg 
    #         B = self.Ebr 
            
    #         if B>0.:
    #             if (C==0.):
    #                 kernel_type = 'SBE'
            
    #             elif (C>0.):
    #                 kernel_type = 'SCE/SBE'
                    
    #             #if not (hasattr(self,'n_fein')):
    #             self.n_fein = Feingold_dists(self.xbins,self.t,self.mu0+1,self.Eagg,self.Ebr,self.lamf,kernel_type=kernel_type)


    #             if kernel_type=='SBE':
                    
    #                 n_fein_new = prefN[0,:]*self.n_fein[:,tind]
    #                 nm_fein_new = prefM[0,:]*self.n_fein[:,tind]

    #                 if normbin:
    #                     n_fein_new /= np.nansum(n_fein_new*dxbins)
    #                     nm_fein_new /= np.nansum(nm_fein_new*dxbins)
                    
    #                 #ax[0].plot(x,prefN[0,:]*self.n_fein[:,tind],':r',linewidth=2,label=f_label+ " analytical")
    #                 #ax[1].plot(x,prefM[0,:]*self.n_fein[:,tind],':r',linewidth=2,label=f_label+ " analytical")
                    
    #                 ax[0].plot(x,n_fein_new,':r',linewidth=2,label=f_label+ " analytical")
    #                 ax[1].plot(x,nm_fein_new,':r',linewidth=2,label=f_label+ " analytical")
                    
    #             elif kernel_type=='SCE/SBE':
                    
    #                 n_fein_new = prefN[0,:]*self.n_fein
    #                 nm_fein_new = prefM[0,:]*self.n_fein

    #                 if normbin:
    #                     n_fein_new /= np.nansum(n_fein_new*dxbins)
    #                     nm_fein_new /= np.nansum(nm_fein_new*dxbins)
                        
    #                 #ax[0].plot(x,prefN[0,:]*self.n_fein,':r',linewidth=2,label=f_label+ " analytical")
    #                 #ax[1].plot(x,prefM[0,:]*self.n_fein,':r',linewidth=2,label=f_label+ " analytical")
                    
    #                 ax[0].plot(x,n_fein_new,':r',linewidth=2,label=f_label+ " analytical")
    #                 ax[1].plot(x,nm_fein_new,':r',linewidth=2,label=f_label+ " analytical")
                    
    #     ax[0].legend() 
            
    #     #plt.tight_layout()
        
    #     if ax_switch:
            
    #         fig.tight_layout()  
            
    #         return fig, ax



    ## OLD RUN METHODS
    
    
    # def run_full_2mom_ORIG(self,pbar=None):
    #     ''' 
    #     Run bin model
    #     '''
        
    #     # Use Butcher table to get rk order coefficients
    #     RK = init_rk(self.rk_order)
    #     a = RK['a']
    #     b = RK['b']
        
    #     rklen = len(b)
        
    #     tf = 0
        
    #     dM = np.zeros((self.dnum,self.Hlen,self.bins,rklen))
    #     dN = np.zeros((self.dnum,self.Hlen,self.bins,rklen))
        
        
    #     for tt in range(1,self.Tlen):

    #         if pbar:
    #             pbar.set_description(self.out_text(self.Ikernel.Mbins,self.Ikernel.Mfbins))
    #             pbar.update(1)

    #         #dM = np.zeros((self.dnum,self.Hlen,self.bins,rklen))
    #         #dN = np.zeros((self.dnum,self.Hlen,self.bins,rklen))
            
    #         M_old = self.Ikernel.Mbins.copy()
    #         N_old= self.Ikernel.Nbins.copy()
            
    #         dM.fill(0.) 
    #         dN.fill(0.)
            
    #         # Generalized Explicit Runge-Kutta time steps
    #         # Keep in mind that for stiff equations, higher
    #         # order Runge-Kutta steps might not be beneficial
    #         # due to stability issues.
    #         for ii in range(rklen):
    #             M_stage = np.maximum(M_old + self.dt*np.nansum(a[ii,:ii][None,None,None,:]*dM[:,:,:,:ii],axis=3),0.)
    #             N_stage = np.maximum(N_old + self.dt*np.nansum(a[ii,:ii][None,None,None,:]*dN[:,:,:,:ii],axis=3),0.)

    #             dM[:,:,:,ii], dN[:,:,:,ii] = self.advance_2mom(M_stage,N_stage,self.dt)
            
                    
    #         self.Ikernel.Mbins =  np.maximum(M_old + self.dt*np.nansum(b[None,None,None,:]*dM,axis=3),0.)
    #         self.Ikernel.Nbins =  np.maximum(N_old + self.dt*np.nansum(b[None,None,None,:]*dN,axis=3),0.)
            
    #         self.Ikernel.update_2mom_subgrid()
                
    #         if self.save_mask[tt]:
    #             tf += 1 
                
    #             self.Mbins[:,:,:,tf] = self.Ikernel.Mbins.copy()
    #             self.Nbins[:,:,:,tf] = self.Ikernel.Nbins.copy()
    
    
    
    
    # def run_steady_state_2mom_ORIGINAL(self,pbar=None):
             
    #         # ELD NOTE: At some point it probably will be worthwhile to do R-K timesteps
    #         if self.Ecb>0.:
                
    #             tf = 0

    #             for tt in range(1,self.Tlen):
                    
    #                 if pbar:
    #                     pbar.set_description(self.out_text(self.Ikernel.Mbins,self.Ikernel.Mfbins))
    #                     pbar.update(1)

    #                 Mbins_old = self.Ikernel.Mbins.copy() 
    #                 Nbins_old = self.Ikernel.Nbins.copy()
                    
    #                 # WORKING
    #                 #M_net, N_net = self.Ikernel.interact_2mom(1.0)
                    
    #                 # TESTING (currently seems to be working)
    #                 #M_net, N_net = self.Ikernel.interact_2mom_SS(1.0)
                    
    #                 # NUMBA TESTING
    #                 M_net, N_net = self.Ikernel.interact_2mom_SS_Final(1.0)
                    
    #                 self.Ikernel.Mbins = np.maximum(Mbins_old+M_net*self.dh[:,None,:],0.)
    #                 self.Ikernel.Nbins = np.maximum(Nbins_old+N_net*self.dh[:,None,:],0.)
                                     
    #                 self.Ikernel.update_2mom_subgrid()
                                 
    #                 if self.save_mask[tt]:
    #                     tf += 1 
    #                     self.Mbins[:,:,:,tf] = self.Ikernel.Mbins.copy() 
    #                     self.Nbins[:,:,:,tf] = self.Ikernel.Nbins.copy() 
   
    #         # NOTE: This makes sure that Mbins/Nbins are in line with other modes 
    #         # after steady-state model runs (dnum x Hlen x bins x time).
    #         if self.int_type==1:
    #             self.Mbins = np.swapaxes(self.Mbins,1,3)
    #             self.Nbins = np.swapaxes(self.Nbins,1,3) 
                
    #             Tout_len = self.Hlen
    #             Hlen = self.Tout_len
                
    #             self.Tout_len = Tout_len 
    #             self.Hlen = Hlen
    
    
    # def run_steady_state_2mom_WORKING(self, pbar=None):
    #     ''' 
    #     Run steady-state bin model using Explicit Runge-Kutta integration.
    #     Uses the Butcher table coefficients (a, b) initialized by init_rk.
    #     '''
        
    #     # 1. Initialize Runge-Kutta Coefficients
    #     RK = init_rk(self.rk_order)
    #     a = RK['a']
    #     b = RK['b']
    #     rklen = len(b)
        
    #     step_size = self.dh[:, None, :] 
        
    #     tf = 0
        
    #     # Pre-allocate RK stage storage
    #     # Shape: (dnum, Hlen, bins, rklen)
    #     dM_stages = np.zeros((self.dnum, self.Hlen, self.bins, rklen))
    #     dN_stages = np.zeros((self.dnum, self.Hlen, self.bins, rklen))

    #     # ---------------------------------------------------------------------
    #     # MAIN LOOP
    #     # ---------------------------------------------------------------------
    #     for tt in range(1, self.Tlen):
            
    #         if pbar:
    #             pbar.set_description(self.out_text(self.Ikernel.Mbins, self.Ikernel.Mfbins))
    #             pbar.update(1)

    #         # Save state at start of step (y_n)
    #         M_old = self.Ikernel.Mbins.copy() 
    #         N_old = self.Ikernel.Nbins.copy()
            
    #         dM_stages.fill(0.) 
    #         dN_stages.fill(0.)
            
    #         # -----------------------------------------------------------------
    #         # RK STAGES
    #         # -----------------------------------------------------------------
    #         for ii in range(rklen):
                
    #             # 1. Calculate Intermediate State (y_stage)
    #             # y_stage = y_n + h * sum(a_ij * k_j)
    #             if ii == 0:
    #                 M_stage = M_old
    #                 N_stage = N_old
    #             else:
    #                 # Sum previous stages weighted by 'a'
    #                 # np.nansum handles the broadcasting of the 'a' coefficients
    #                 k_sum_M = np.sum(a[ii, :ii][None, None, None, :] * dM_stages[:, :, :, :ii], axis=3)
    #                 k_sum_N = np.sum(a[ii, :ii][None, None, None, :] * dN_stages[:, :, :, :ii], axis=3)
                    
    #                 M_stage = np.maximum(M_old + step_size * k_sum_M, 0.)
    #                 N_stage = np.maximum(N_old + step_size * k_sum_N, 0.)

    #             # 2. Evaluate Derivative (k_i = f(y_stage))
    #             # The 'derivative' here is the interaction rate per unit step
    #             dM_dt, dN_dt = self.get_steady_rates(M_stage, N_stage)
                
    #             # 3. Store derivative for future stages
    #             dM_stages[:, :, :, ii] = dM_dt
    #             dN_stages[:, :, :, ii] = dN_dt
            
    #         # -----------------------------------------------------------------
    #         # FINAL UPDATE (y_n+1)
    #         # -----------------------------------------------------------------
    #         # y_n+1 = y_n + h * sum(b_i * k_i)
            
    #         total_sum_M = np.sum(b[None, None, None, :] * dM_stages, axis=3)
    #         total_sum_N = np.sum(b[None, None, None, :] * dN_stages, axis=3)
            
    #         self.Ikernel.Mbins = np.maximum(M_old + step_size * total_sum_M, 0.)
    #         self.Ikernel.Nbins = np.maximum(N_old + step_size * total_sum_N, 0.)
            
    #         # Update auxiliary physics variables for the new state
    #         self.Ikernel.update_2mom_subgrid()
            
    #         # -----------------------------------------------------------------
    #         # SAVE OUTPUT
    #         # -----------------------------------------------------------------
    #         if self.save_mask[tt]:
    #             tf += 1 
    #             self.Mbins[:, :, :, tf] = self.Ikernel.Mbins.copy() 
    #             self.Nbins[:, :, :, tf] = self.Ikernel.Nbins.copy() 
   
    #     # Post-processing: Swap axes if int_type==1 (Steady State logic)
    #     if self.int_type == 1:
    #         self.Mbins = np.swapaxes(self.Mbins, 1, 3)
    #         self.Nbins = np.swapaxes(self.Nbins, 1, 3) 
            
    #         Tout_len = self.Hlen
    #         Hlen = self.Tout_len
            
    #         self.Tout_len = Tout_len 
    #         self.Hlen = Hlen
 
    # def run_steady_state_2mom_RK_ADJ(self, pbar=None):
    #     ''' 
    #     Run steady-state model with 2 moment bin prediction.
    #     '''
        
    #     # 1. Initialize RK Coefficients
    #     RK = init_rk(self.rk_order)
    #     a = RK['a']
    #     b = RK['b']
    #     rklen = len(b)
        
    #     # Output counter
    #     tf = 0
        
    #     # Pre-allocate RK stage storage
    #     dM_stages = np.zeros((self.dnum, self.Hlen, self.bins, rklen))
    #     dN_stages = np.zeros((self.dnum, self.Hlen, self.bins, rklen)) # For 2-mom

    #     # ---------------------------------------------------------------------
    #     # MAIN LOOP (Fixed Output Grid)
    #     # ---------------------------------------------------------------------
    #     for tt in range(1, self.Tlen):
            
    #         if pbar:
    #             pbar.set_description(self.out_text(self.Ikernel.Mbins, self.Ikernel.Mfbins))
    #             pbar.update(1)

    #         # Target step
    #         # Assuming self.dh is constant or an array. Let's assume a scalar 'dh_target' for simplicity
    #         # If self.dh is an array (dnum, Hlen, bins), we take the max or relevant value.
    #         # Let's assume we are stepping 'dh' meters forward.
    #         dh_target = np.max(self.dh) 
            
    #         dh_covered = 0.0
            
    #         # Start State for this height level
    #         M_current = self.Ikernel.Mbins.copy()
    #         N_current = self.Ikernel.Nbins.copy()

    #         # -----------------------------------------------------------------
    #         # ADAPTIVE SUB-STEPPING LOOP
    #         # -----------------------------------------------------------------
    #         # We must reach dh_target, but we can take small steps to get there.
            
    #         while dh_covered < dh_target:
                
    #             # A. Estimate Stiffness / Max Safe Step
    #             # Evaluate the derivative at the current state
    #             dM_dt, dN_dt = self.get_steady_rates(M_current, N_current)
                
    #             # Stability Criterion:
    #             # We want to limit the relative change per step to ~10% (0.1)
    #             # Max Safe dh = 0.1 * (Mass / Derivative)
                
    #             # Avoid divide by zero/noise
    #             valid_mask = (M_current > 1e-15) & (np.abs(dM_dt) > 1e-20)
                
    #             if np.any(valid_mask):
    #                 scale_M = M_current[valid_mask] / (np.abs(dM_dt[valid_mask]) + 1e-30)
    #                 min_scale = np.min(scale_M)
                    
    #                 # The "10% rule" - very robust for RK4
    #                 dh_safe = 0.1 * min_scale
    #             else:
    #                 dh_safe = dh_target # Physics is quiet, take full step
                
    #             # Clamp step size
    #             remaining = dh_target - dh_covered
    #             dh_step = min(dh_safe, remaining)
                
    #             # Prevent infinitesimally small steps (stiff crash guard)
    #             dh_step = max(dh_step, 1e-4) 

    #             # B. Perform RK Integration Step with 'dh_step'
    #             # (Standard RK loop, but using M_current as base)
                
    #             dM_stages.fill(0.)
    #             dN_stages.fill(0.)
                
    #             # Store the step size for broadcasting in the RK loop
    #             step_size_arr = np.full_like(self.dh, dh_step)[:, None, :]

    #             for ii in range(rklen):
    #                 if ii == 0:
    #                     M_stage = M_current
    #                     N_stage = N_current
    #                 else:
    #                     k_sum_M = np.sum(a[ii, :ii][None, None, None, :] * dM_stages[:, :, :, :ii], axis=3)
    #                     k_sum_N = np.sum(a[ii, :ii][None, None, None, :] * dN_stages[:, :, :, :ii], axis=3)
                        
    #                     M_stage = np.maximum(M_current + step_size_arr * k_sum_M, 0.)
    #                     N_stage = np.maximum(N_current + step_size_arr * k_sum_N, 0.)

    #                 # Compute rates for this stage
    #                 dM_k, dN_k = self.get_steady_rates(M_stage, N_stage)
    #                 dM_stages[:, :, :, ii] = dM_k
    #                 dN_stages[:, :, :, ii] = dN_k
                
    #             # C. Final Update for this sub-step
    #             total_sum_M = np.sum(b[None, None, None, :] * dM_stages, axis=3)
    #             total_sum_N = np.sum(b[None, None, None, :] * dN_stages, axis=3)
                
    #             M_current = np.maximum(M_current + step_size_arr * total_sum_M, 0.)
    #             N_current = np.maximum(N_current + step_size_arr * total_sum_N, 0.)
                
    #             dh_covered += dh_step
            
    #         # -----------------------------------------------------------------
    #         # END SUB-STEPPING
    #         # -----------------------------------------------------------------
            
    #         # Commit final state to the kernel object for the next output step
    #         self.Ikernel.Mbins = M_current
    #         self.Ikernel.Nbins = N_current
    #         self.Ikernel.update_2mom_subgrid()

    #         # Save Output
    #         if self.save_mask[tt]:
    #             tf += 1 
    #             self.Mbins[:, :, :, tf] = self.Ikernel.Mbins.copy() 
    #             self.Nbins[:, :, :, tf] = self.Ikernel.Nbins.copy() 
   
    #     # Post-processing
    #     if self.int_type == 1:
    #         self.Mbins = np.swapaxes(self.Mbins, 1, 3)
    #         self.Nbins = np.swapaxes(self.Nbins, 1, 3) 

    #         Tout_len = self.Hlen
    #         Hlen = self.Tout_len
            
    #         self.Tout_len = Tout_len
    #         self.Hlen = Hlen




    # def run_full_1mom_ORIG(self,pbar=None):
    #     ''' 
    #     Run bin model
    #     '''

    #     # use Butcher table to get rk order coefficients
    #     RK = init_rk(self.rk_order)
    #     a = RK['a']
    #     b = RK['b']
        
    #     rklen = len(b)
        
    #     tf = 0
        
    #     dM = np.zeros((self.dnum,self.Hlen,self.bins,rklen))
        
    #     for tt in range(1,self.Tlen):
    #         if pbar:
    #             # if tt % 50 == 0:
    #             pbar.set_description(self.out_text(self.Ikernel.Mbins,self.Ikernel.Mfbins))
    #             pbar.update(1)
                
    #         dM.fill(0.)
            
    #         M_old = self.Ikernel.Mbins.copy()
        
    #         # Generalized Explicit Runge-Kutta time steps
    #         # Keep in mind that for stiff equations higher
    #         # order Runge-Kutta steps might not be beneficial
    #         # due to stability issues.
    #         for ii in range(rklen):
                               
    #             M_stage = np.maximum(M_old + self.dt*np.nansum(a[ii,:ii][None,None,None,:]*dM[:,:,:,:ii],axis=3),0.)

    #             dM[:,:,:,ii] = self.advance_1mom(M_stage,self.dt)
                 
    #         self.Ikernel.Mbins = np.maximum(M_old + self.dt*np.nansum(b[None,None,None,:]*dM,axis=3),0.)

    #         self.Ikernel.update_1mom_subgrid()
            
    #         if self.save_mask[tt]:
    #             tf += 1
    #             self.Mbins[:,:,:,tf] = self.Ikernel.Mbins.copy() 


    # def run_steady_state_1mom_ORIG_noadapt(self,pbar=None):
        
    #     # use Butcher table to get rk order coefficients
    #     # RK = init_rk(self.rk_order)
    #     # a = RK['a']
    #     # b = RK['b']
        
    #     # rklen = len(b)
        
    #     # ELD NOTE: At some point it probably will be worthwhile to do R-K time/height steps
    #     if self.Ecb>0.:
             
    #         tf = 0
            
    #         for tt in range(1,self.Tlen):

    #             if pbar:
    #                 pbar.set_description(self.out_text(self.Ikernel.Mbins,self.Ikernel.Mfbins))
    #                 pbar.update(1)
                
    #             Mbins_old = self.Ikernel.Mbins.copy() 
                
    #             #dM = np.zeros((self.dnum,self.Hlen,self.bins,rklen))
                
    #            # M_old = self.Ikernel.Mbins.copy()
            
    #             # Generalized Explicit Runge-Kutta time steps
    #             # Keep in mind that for stiff equations higher
    #             # order Runge-Kutta steps might not be beneficial
    #             # due to stability issues.
    #             # for ii in range(rklen):
                                   
    #             #     M_stage = np.maximum(Mbins_old + self.dt*np.sum(a[ii,:ii][None,None,None,:]*dM[:,:,:,:ii],axis=3),0.)

    #             #     Mbins = self.Ikernel.interact_1mom_SS(1.0)
                   
    #             #     M_net = (Mbins-M_stage)/self.dt
                    
    #             #     dM[:,:,:,ii] = np.maximum(Mbins_old+M_net*self.dh[:,None,:],0.)
                    
                    
    #             # self.Ikernel.Mbins = np.maximum(Mbins_old + self.dt*np.sum(b[None,None,None,:]*dM,axis=3),0.)
  

    
    #             # WORKING
    #             #M_net = self.Ikernel.interact_1mom_SS(1.0)
                
    #             M_net = self.Ikernel.interact_1mom_SS_Final(1.0)
                

    #             # WORKING NO RK
    #             self.Ikernel.Mbins = np.maximum(Mbins_old+M_net*self.dh[:,None,:],0.)
   
    #             self.Ikernel.update_1mom_subgrid()
                
    #             # Save current Ikernel Mbins in spectral_1d() Mbins array
    #             if self.save_mask[tt]:
    #                 tf += 1
    #                 self.Mbins[:,:,:,tf] = self.Ikernel.Mbins.copy()
            
              
    #         # NOTE: This makes sure that Mbins/Nbins are in line with other modes 
    #         # after steady-state model runs (dnum x Hlen x bins x time).
    #         if self.int_type==1:
    #             self.Mbins = np.swapaxes(self.Mbins,1,3)
                
    #             Tout_len = self.Hlen
    #             Hlen = self.Tout_len
                
    #             self.Tout_len = Tout_len 
    #             self.Hlen = Hlen
 
    
 
    # def run_steady_state_1mom_WORKING(self, pbar=None):
    #     '''
    #     Run steady-state 1-moment bin model using Adaptive Sub-stepping RK4.
    #     Preserves Simmel/Wang smoothness while preventing Breakup explosions.
    #     '''
        
    #     # 1. Initialize RK Coefficients
    #     RK = init_rk(self.rk_order)
    #     a = RK['a']
    #     b = RK['b']
    #     rklen = len(b)
        
    #     tf = 0
        
    #     # Pre-allocate RK stage storage
    #     dM_stages = np.zeros((self.dnum, self.Hlen, self.bins, rklen))
        
    #     # ---------------------------------------------------------------------
    #     # MAIN LOOP (Fixed Output Grid)
    #     # ---------------------------------------------------------------------
    #     for tt in range(1, self.Tlen):
            
    #         if pbar:
    #             pbar.set_description(self.out_text(self.Ikernel.Mbins, self.Ikernel.Mfbins))
    #             pbar.update(1)
            
    #         # Target Height Step for this output interval
    #         # Assuming scalar or max step across domain
    #         dh_target = np.max(self.dh)
            
    #         dh_covered = 0.0
            
    #         # Start State for this height level
    #         M_current = self.Ikernel.Mbins.copy()
            
    #         # -----------------------------------------------------------------
    #         # ADAPTIVE SUB-STEPPING LOOP
    #         # -----------------------------------------------------------------
    #         while dh_covered < dh_target:
                
    #             # A. Estimate Stiffness / Max Safe Step
    #             dM_dt = self.get_steady_rate_1mom(M_current)
                
    #             # Stability Criterion: Limit relative change to ~10% per step
    #             # Max Safe dh = 0.1 * (Mass / Derivative)
                
    #             valid_mask = (M_current > 1e-15) & (np.abs(dM_dt) > 1e-20)
                
    #             if np.any(valid_mask):
    #                 scale_M = M_current[valid_mask] / (np.abs(dM_dt[valid_mask]) + 1e-30)
    #                 min_scale = np.min(scale_M)
                    
    #                 dh_safe = 0.1 * min_scale
    #             else:
    #                 dh_safe = dh_target # Physics is quiet
                
    #             # Clamp step size
    #             remaining = dh_target - dh_covered
    #             dh_step = min(dh_safe, remaining)
                
    #             # Prevent infinitesimally small steps (stiff crash guard)
    #             dh_step = max(dh_step, 1e-4)
                
    #             # B. Perform RK Integration Step with 'dh_step'
    #             dM_stages.fill(0.)
                
    #             # Broadcast step size for the RK stages
    #             # Shape: (dnum, Hlen, bins)
    #             step_size_arr = np.full_like(self.dh, dh_step)[:, None, :]
                
    #             for ii in range(rklen):
    #                 if ii == 0:
    #                     M_stage = M_current
    #                 else:
    #                     # Sum previous stages
    #                     k_sum_M = np.sum(a[ii, :ii][None, None, None, :] * dM_stages[:, :, :, :ii], axis=3)
    #                     M_stage = np.maximum(M_current + step_size_arr * k_sum_M, 0.)
                    
    #                 # Compute rate for this stage
    #                 dM_k = self.get_steady_rate_1mom(M_stage)
    #                 dM_stages[:, :, :, ii] = dM_k
                
    #             # C. Final Update for this sub-step
    #             total_sum_M = np.sum(b[None, None, None, :] * dM_stages, axis=3)
                
    #             M_current = np.maximum(M_current + step_size_arr * total_sum_M, 0.)
                
    #             dh_covered += dh_step
            
    #         # -----------------------------------------------------------------
    #         # END SUB-STEPPING
    #         # -----------------------------------------------------------------
            
    #         # Commit final state
    #         self.Ikernel.Mbins = M_current
    #         self.Ikernel.update_1mom_subgrid()
            
    #         # Save Output
    #         if self.save_mask[tt]:
    #             tf += 1
    #             self.Mbins[:, :, :, tf] = self.Ikernel.Mbins.copy()
        
    #     # Post-processing
    #     if self.int_type == 1:
    #         self.Mbins = np.swapaxes(self.Mbins, 1, 3)
            
    #         Tout_len = self.Hlen
    #         Hlen = self.Tout_len
            
    #         self.Tout_len = Tout_len
    #         self.Hlen = Hlen
    
    # def run_steady_state_1mom_ORIG(self, pbar=None):
    #     '''
    #     Run steady-state 1-moment bin model using Explicit Runge-Kutta integration.
    #     Uses the Butcher table coefficients (a, b) initialized by init_rk.
    #     '''
        
    #     # 1. Initialize Runge-Kutta Coefficients
    #     RK = init_rk(self.rk_order)
    #     a = RK['a']
    #     b = RK['b']
    #     rklen = len(b)
        
    #     # 2. Define the "Time" Step (Here, it's Height Step dh)
    #     # Broadcasting to match (dnum, Hlen, bins)
    #     step_size = self.dh[:, None, :]
        
    #     tf = 0
        
    #     # Pre-allocate RK stage storage
    #     # Shape: (dnum, Hlen, bins, rklen)
    #     dM_stages = np.zeros((self.dnum, self.Hlen, self.bins, rklen))
        
    #     # ---------------------------------------------------------------------
    #     # MAIN LOOP
    #     # ---------------------------------------------------------------------
    #     for tt in range(1, self.Tlen):
            
    #         if pbar:
    #             pbar.set_description(self.out_text(self.Ikernel.Mbins, self.Ikernel.Mfbins))
    #             pbar.update(1)
            
    #         # Save state at start of step (y_n)
    #         M_old = self.Ikernel.Mbins.copy()
            
    #         dM_stages.fill(0.)
            
    #         # -----------------------------------------------------------------
    #         # RK STAGES
    #         # -----------------------------------------------------------------
    #         for ii in range(rklen):
                
    #             # 1. Calculate Intermediate State (y_stage)
    #             # y_stage = y_n + h * sum(a_ij * k_j)
    #             if ii == 0:
    #                 M_stage = M_old
    #             else:
    #                 # Sum previous stages weighted by 'a'
    #                 k_sum_M = np.nansum(a[ii, :ii][None, None, None, :] * dM_stages[:, :, :, :ii], axis=3)
    #                 M_stage = np.maximum(M_old + step_size * k_sum_M, 0.)
                
    #             # 2. Evaluate Derivative (k_i = f(y_stage))
    #             # The 'derivative' here is the interaction rate per unit step (M_net)
    #             dM_dt = self.get_steady_rate_1mom(M_stage)
                
    #             # 3. Store derivative for future stages
    #             dM_stages[:, :, :, ii] = dM_dt
                
    #         # -----------------------------------------------------------------
    #         # FINAL UPDATE (y_n+1)
    #         # -----------------------------------------------------------------
    #         # y_n+1 = y_n + h * sum(b_i * k_i)
            
    #         total_sum_M = np.nansum(b[None, None, None, :] * dM_stages, axis=3)
            
    #         # Apply update and clamp to positive
    #         self.Ikernel.Mbins = np.maximum(M_old + step_size * total_sum_M, 0.)
            
    #         # Update auxiliary physics variables for the new state
    #         self.Ikernel.update_1mom_subgrid()
            
    #         # -----------------------------------------------------------------
    #         # SAVE OUTPUT
    #         # -----------------------------------------------------------------
    #         if self.save_mask[tt]:
    #             tf += 1
    #             self.Mbins[:, :, :, tf] = self.Ikernel.Mbins.copy()
        
    #     # Post-processing: Swap axes if int_type==1 (Steady State logic)
    #     # Aligns output to (dnum, Hlen, bins, time)
    #     if self.int_type == 1:
    #         self.Mbins = np.swapaxes(self.Mbins, 1, 3)
            
    #         Tout_len = self.Hlen
    #         Hlen = self.Tout_len
            
    #         self.Tout_len = Tout_len
    #         self.Hlen = Hlen
    
    # def advance_1mom(self,M_old,dt):
        
    #     Mbins = np.zeros_like(M_old)
        
    #     #M_net = self.Ikernel.interact_1mom_array(dt)
        
    #     M_net = self.Ikernel.interact_1mom_SS_Final(dt)
        
    #     #M_net = self.Ikernel.interact_1mom_SS(dt)
        
    #    # M_net = self.Ikernel.interact_1mom_vectorized(dt)
        
    #     #M_net = self.Ikernel.interact_1mom_SS_NEW(dt)
        
    #     M_sed = np.zeros((self.dnum,self.Hlen,self.bins)) 
       
    #     if self.boundary is None:
    #         M_sed[:,0,:] = (dt/self.dz)*(-self.Ikernel.Mfbins[:,0,:]) 
        
    #     M_sed[:,1:,:] = (dt/self.dz)*(self.Ikernel.Mfbins[:,:-1,:]-self.Ikernel.Mfbins[:,1:,:]) 
        
    #     M_transfer = M_old+M_sed+M_net
        
    #     M_new = np.maximum(M_transfer,0.) # Should be positive if not over fragmented.
    #     Mbins[M_new>=0.] = M_new[M_new>=0.].copy()
        

    #     if self.boundary=='fixed': # If fixing top distribution. Can be helpful if trying to determine steady-state time.
    #         Mbins[:,0,:] = M_old[:,0,:].copy()
            
    #     dM = (Mbins-M_old)/dt
        
    #     return dM
    
    
    # def advance_2mom(self,M_old,N_old,dt):
        
    #     Mbins = np.zeros_like(M_old)
    #     Nbins = np.zeros_like(N_old)
        
    #     #M_net, N_net = self.Ikernel.interact_2mom_SS(dt)
        
    #     M_net, N_net = self.Ikernel.interact_2mom_SS_Final(dt)
       
    #     M_sed = np.zeros((self.dnum,self.Hlen,self.bins)) 
    #     N_sed = np.zeros((self.dnum,self.Hlen,self.bins)) 
       
    #     if self.boundary is None:
    #         M_sed[:,0,:] = (dt/self.dz)*(-self.Ikernel.Mfbins[:,0,:]) 
    #         N_sed[:,0,:] = (dt/self.dz)*(-self.Ikernel.Nfbins[:,0,:]) 
        
    #     M_sed[:,1:,:] = (dt/self.dz)*(self.Ikernel.Mfbins[:,:-1,:]-self.Ikernel.Mfbins[:,1:,:]) 
    #     N_sed[:,1:,:] = (dt/self.dz)*(self.Ikernel.Nfbins[:,:-1,:]-self.Ikernel.Nfbins[:,1:,:]) 
        
    #     M_transfer = M_old+M_sed+M_net
    #     N_transfer = N_old+N_sed+N_net   
        
    #     M_new = np.maximum(M_transfer,0.) # Should be positive if not over fragmented.
    #     Mbins[M_new>=0.] = M_new[M_new>=0.].copy()
        
    #     N_new = np.maximum(N_transfer,0.) # Should be positive if not over fragmented.
    #     Nbins[N_new>=0.] = N_new[N_new>=0.].copy()
        
    #     if self.boundary=='fixed': # If fixing top distribution. Can be helpful if trying to determine steady-state time.
    #         Mbins[:,0,:] = M_old[:,0,:].copy()
    #         Nbins[:,0,:] = N_old[:,0,:].copy()
            
    #     dM = (Mbins-M_old)/dt
    #     dN = (Nbins-N_old)/dt
        
    #     return dM, dN


    # IF USING SHARED MEMORY OR NUMPY MEMMAP ARRAYS
    
    # def __enter__(self):
        
    #     if (self.parallel) and (self.pool is None):
    #         self.pool = self._parallel_config.__enter__()
    #         self.Ikernel.pool = self.pool
    #     return self

    # def __exit__(self,exc_type,exc_val,exc_tb):
        
    #     if self.pool is not None:
    #         self._parallel_config.__exit__(exc_type, exc_val, exc_tb)
    #         self.pool = None 
    #         self.Ikernel.pool = None
        
    # def __del__(self):
        
    #     try: 
    #         self.__exit__(None,None,None)
    #     except:
    #         pass