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

"""

## Import stuff
import numpy as np
import scipy.special as scip
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
from analytical_solutions import Scott_dists, Feingold_dists
from copy import deepcopy
from distribution import dist
from interaction import Interaction
from bin_integrals import init_rk

from habits import habits, fragments

from matplotlib.colors import BoundaryNorm

from plotting_functions import get_cmap_vars

from datetime import datetime

import sys

import os 

if 'ipykernel_launcher.py' in sys.argv[0]:
    from tqdm.auto import tqdm
else:
    from tqdm import tqdm
       

# 1D Spectral Bin Model Class
class spectral_1d:
    
    def __init__(self,sbin=8,bins=140,dt=2,
                 tmax=800.,output_freq=60.,dz=10.,ztop=0.,zbot=0.,D1=0.25,x0=0.01,Nt0=1.,Dm0=2.0,
                 mu0=3.,gam_norm=False,Ecol=0.001,Es=1.0,Eb=0.,moments=2,dist_var='mass',
                 kernel='Golovin',frag_dist='exp',habit_list=['rain'],
                 ptype='rain',Tc=10.,boundary=None,dist_num=1,cc_dest=1,br_dest=1, 
                 radar=False,rk_order=1,adv_order=1,parallel=False,n_jobs=-1):
        '''
        Initialize model and PSD        
        '''       
        
        self.setup_case(sbin=sbin,bins=bins,D1=D1,x0=x0,dt=dt,tmax=tmax,
                        output_freq=output_freq,dz=dz,ztop=ztop,zbot=zbot,Nt0=Nt0,Dm0=Dm0,mu0=mu0,gam_norm=gam_norm,Ecol=Ecol,
                        Es=Es,Eb=Eb,moments=moments,dist_var=dist_var,kernel=kernel,frag_dist=frag_dist,
                        habit_list=habit_list,ptype=ptype,Tc=Tc,radar=radar,boundary=boundary,
                        dist_num=dist_num,cc_dest=cc_dest,br_dest=br_dest,rk_order=rk_order,adv_order=adv_order,
                        parallel=parallel,n_jobs=n_jobs)
        
    def setup_case(self,sbin=4,bins=160,D1=0.001,x0=0.01,Nt0=1.,Dm0=2.0,mu0=3,gam_norm=False,dist_var='mass',kernel='Golovin',Ecol=1.53,Es=0.001,Eb=0.,
                        moments=2,ztop=3000.0,zbot=0.,tmax=800.,output_freq=60.,dt=10.,dz=10.,frag_dist='exp',habit_list=['rain'],ptype='rain',Tc=10.,
                        radar=False,boundary=None,dist_num=1,cc_dest=1,br_dest=1,rk_order=1,adv_order=1,parallel=False,n_jobs=-1):
        self.Tc = Tc
        self.radar = radar
        self.sbin = sbin 
        self.bins = bins
        self.D1 = D1
        self.Nt0 = Nt0 
        self.Dm0 = Dm0 
        self.mu0 = mu0 
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
        self.output_freq = output_freq
        self.dt = dt 
        self.dz = dz
        # If tmax = -1 then run model as steady-state in vertical
        self.t = np.arange(0,self.tmax+self.dt,step=self.dt)
        self.tout = np.arange(0,self.tmax+self.output_freq,step=self.output_freq)
        self.Tout_len = len(self.tout)
        # If zbot == ztop == const. then run as box model
        self.z = np.arange(self.ztop,self.zbot-self.dz,step=-self.dz)
        self.dist_var = dist_var
        self.ptype = ptype
        self.rk_order = rk_order
        self.adv_order = adv_order
        self.boundary = boundary
        self.parallel = parallel
        
        if n_jobs == -1:
            self.n_jobs = os.cpu_count()
        else:
            self.n_jobs = n_jobs
        
        self.indc = cc_dest 
        self.indb = br_dest
        
        self.dnum = dist_num
        
        self.Tlen = len(self.t) 
        self.Hlen = len(self.z)
        self.moments = moments
        
        # If time array is fixed then run as steady state model
        if (self.Tlen==1) & (self.Hlen>1):
            self.int_type = 1         
            self.Hlen = 1
            self.Tlen = len(self.z)
            self.dt = 1
            
            # NOTE CURRENTLY NO OPTION FOR PARALLEL IN SS MODE
            self.parallel = False
            
        elif (self.Tlen>1) & (self.Hlen==1):
            self.int_type = 2
            self.Hlen = 1
            self.Tlen = len(self.t) 
            self.dh = self.t
            
            # NOTE CURRENTLY NO OPTION FOR PARALLEL IN BOX MODE
            self.parallel = False
            
        else:
            self.int_type=0
        
        # Ensure that cc_dest and br_dest are valid     
        if (cc_dest<1) | (cc_dest>dist_num):
            print('cc_dest needs to be between 1 and {}'.format(dist_num))
            raise Exception()
        if (br_dest<1) | (br_dest>dist_num):
            print('br_dest needs to be between 1 and {}'.format(dist_num))
            raise Exception()
           
        # Initialize distribution objects
        
        if dist_var=='size':
            x0=None
        elif dist_var=='mass':
            D1=None
        
        # If dnum > habit list then just use first element for all habits
        if len(habit_list) < self.dnum:
            habit_list = [habit_list[0] for dd in range(self.dnum)]
        
        habit_dict = [habits()[habit_list[dd]] for dd in range(self.dnum)]

        dists = np.empty((self.dnum,self.Hlen),dtype=object)
        
        # initial distribution
        dists[0,0] = dist(sbin=sbin,bins=bins,D1=D1,x0=x0,Nt0=Nt0,mu0=mu0,Dm0=Dm0,
                      gam_init=True,gam_norm=gam_norm,dist_var=dist_var,kernel=kernel,
                      habit_dict=habit_dict[0],ptype=ptype,Tc=Tc,radar=radar,mom_num=moments)

        self.dist0 = deepcopy(dists[0,0])
        
        for hh in range(self.Hlen):
            for dd in range(self.dnum):
                
                if not ((dd==0) & (hh==0)):
                    # Coalesced or fragmented particles
                    dists[dd,hh] = dist(sbin=sbin,D1=D1,bins=bins,gam_init=False,dist_var=dist_var,
                                 kernel=kernel,habit_dict=habit_dict[dd],ptype=ptype,x0=self.dist0.x0, 
                                 Tc=Tc,radar=radar,mom_num=moments)
                    
                dists[dd,hh].dh = self.dz/dists[dd,hh].vt   
                dists[dd,hh].dt = self.dt*np.ones_like(self.t)

        if frag_dist is None:
            frag_dict = fragments('exp')
        else:
            frag_dict = fragments(frag_dist)
                
        self.xbins = self.dist0.xbins.copy() 
        self.xedges = self.dist0.xedges.copy()
        
        # Initialize interaction kernel between each species
        # Interaction() takes a (Ndist x height) array of dist objects
        # and sets up arrays for calculating interaction (i.e., source) terms
        # in the stochastic collection/breakup equation for multiple categories
        self.Ikernel = Interaction(dists,cc_dest,br_dest,self.Eagg,self.Ecb,self.Ebr, 
                                   frag_dict,self.kernel,parallel=self.parallel,n_jobs=self.n_jobs,
                                   mom_num=self.moments)
        
        self.lamf = frag_dict['lamf']
        
        # Stencils used for variable upwind advection
        self.stencils = {1: np.array([-1, 1]) / 1,
                         2: np.array([1, -4, 3]) / 2,
                         3: np.array([-2, 9, -18, 11]) / 6,
                         4: np.array([3, -16, 36, -48, 25]) / 12}
        
        self.adv_base = self.stencils[adv_order]

        self.dists = dists # 3D array of distribution objects (dist_num x height x time)

    def clean_up(self):
        
        if self.parallel:
            
            del self.Ikernel.dMb_gain_frac
            del self.Ikernel.dNb_gain_frac
            del self.Ikernel.PK    
            del self.Ikernel.kmin
            del self.Ikernel.kmid
            del self.Ikernel.cond_1
            del self.Ikernel.self_col
            
            if self.moments==1:
                
                del self.Ikernel.dMi_loss
                del self.Ikernel.dMj_loss
                del self.Ikernel.dM_gain
        

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
    
        if self.dist0.radar:
            print('------')
            print('Initial Reflectivity = {:.2f} dBZ'.format(self.dist0.ZH))
            print('------')
            print('Initial Differential Reflectivity = {:.2f} dB'.format(self.dist0.ZDR))
            print('------')
            print('Initial Specific Differential Phase = {:.2f} deg/km'.format(self.dist0.KDP))
    
    def plot_time_height(self,var='Z'):
        
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize=16) 
        plt.rc('ytick', labelsize=16) 
        
        dist_num = len(self.dists) 

        t = self.tout
        h = self.z/1000.
             
        N  = np.full((dist_num,len(h),len(t)),np.nan)
        M  = np.full((dist_num,len(h),len(t)),np.nan)
        Dm = np.full((dist_num,len(h),len(t)),np.nan)
        Rm = np.full((dist_num,len(h),len(t)),np.nan)
        M3 = np.full((dist_num,len(h),len(t)),np.nan)
        M4 = np.full((dist_num,len(h),len(t)),np.nan)

        ZH = np.full((len(h),len(t)),np.nan)
        ZDR = np.full((len(h),len(t)),np.nan)
        KDP = np.zeros((len(h),len(t)))
        RHOHV = np.full((len(h),len(t)),np.nan)

        for tt in range(len(t)):
            
            for ff in range(len(h)):
                
                Zh = 0. 
                Zv = 0. 
                Zhhvv = 0. 
     
                for d1 in range(dist_num):
                    
                    am = self.full[d1,ff,tt].am
                    bm = self.full[d1,ff,tt].bm
                    
                    M3[d1,ff,tt] = (am**(-3./bm)*self.full[d1,ff,tt].moments(3./bm)).sum()
                    M4[d1,ff,tt] = (am**(-4./bm)*self.full[d1,ff,tt].moments(4./bm)).sum()
                      
                    N[d1,ff,tt]  = np.nansum(self.full[d1,ff,tt].Nbins)
                    M[d1,ff,tt]  = 1000.*np.nansum(self.full[d1,ff,tt].Mbins)
                    Rm[d1,ff,tt] = 1000.*3.6*np.nansum(self.full[d1,ff,tt].Mfbins)
                    
                    Zh      += np.nansum(self.full[d1,ff,tt].zh)
                    Zv      += np.nansum(self.full[d1,ff,tt].zv)
                    Zhhvv   += np.nansum(self.full[d1,ff,tt].zhhvv)
                    KDP[ff,tt] += np.nansum(self.full[d1,ff,tt].KDP)

                if Zv>0.:
                    ZDR[ff,tt] = 10.*np.log10(Zh/Zv)
                  
                if Zh>0.:    
                    ZH[ff,tt] = 10.*np.log10(Zh)    
                  
                if (Zh>0.) & (Zv>0.):
                    RHOHV[ff,tt] = np.abs(Zhhvv)/np.sqrt(Zh*Zv)
                
        M3_tot = np.nansum(M3,axis=0)
        M4_tot = np.nansum(M4,axis=0)
        
        M3_tot[M3_tot==0.] = np.nan
        
        Dm = M4_tot/M3_tot
        
        M3[M3==0.] = np.nan 
        
        #Dm = M4/M3
        
        Nt = np.nansum(N,axis=0)
        M_tot = np.nansum(M,axis=0)
        Rm_tot = np.nansum(Rm,axis=0)
        
        fig, ax = plt.subplots(1,1,figsize=(14,8))

        match var:
            case 'Z':
                var_temp = ZH.copy() 
            case 'ZDR':
                var_temp = ZDR.copy() 
            case 'KDP':
                var_temp = KDP.copy() 
            case 'RHOHV':
                var_temp = RHOHV.copy() 
            case 'Nt':
                var_temp = Nt.copy() 
            case 'Dm':
                var_temp = Dm.copy() 
            case 'WC':
                var_temp = M_tot.copy()
            case 'R':
                var_temp = Rm_tot.copy()
        
        
        cmap, levels, levels_ticks, clabel, labelpad, fontsize, slabel = get_cmap_vars(var)
        
        Rnorm = BoundaryNorm(levels,cmap.N,extend='both') 

        cax = ax.pcolor(t,h,var_temp,norm=Rnorm,cmap=cmap)
        
        #cax = ax.pcolor(t,h,var_temp,cmap=cmap)
        
        cbar = fig.colorbar(cax,ax=ax,ticks=levels_ticks)
        
        cbar.ax.tick_params(labelsize=16)
        
        cbar.ax.set_yticklabels(levels_ticks,usetex=True)
        
        cbar.ax.minorticks_off()
        
        cbar.set_label(clabel,usetex=True,rotation=270,fontsize=fontsize,labelpad=labelpad) 

        ax.set_xlabel('Time (seconds)',fontsize=16,usetex=True)
        ax.set_ylabel('Height (km)',fontsize=16,usetex=True)
        
        ax.axes.tick_params('both',labelsize=14)

        fig.tight_layout()  
        
        return fig, ax
    
    
    def plot_moments_radar(self,tind=-1,plot_habits=False):
        
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize=16) 
        plt.rc('ytick', labelsize=16) 
        
        dist_num = len(self.dists) 
        
        if self.int_type==2:
            h = self.t
        elif self.int_type==1:
            h = self.z/1000.
        elif self.int_type==0:
            #t = self.t 
            h = self.z/1000.
        
        #if len(self.z)==1:
        #    h = self.t
        #elif len(self.t)==1:
        #    h = self.z/1000.
        
        # N  = np.full_like(self.full,np.nan)
        # M  = np.full_like(self.full,np.nan)
        # Dm = np.full_like(self.full,np.nan)
        # Rm = np.full_like(self.full,np.nan)      
        # M3 = np.full_like(self.full,np.nan)  
        # M4 = np.full_like(self.full,np.nan)  
        
        N  = np.full((dist_num,len(h)),np.nan)
        M  = np.full((dist_num,len(h)),np.nan)
        Dm = np.full((dist_num,len(h)),np.nan)
        Rm = np.full((dist_num,len(h)),np.nan)    
        M3 = np.full((dist_num,len(h)),np.nan) 
        M4 = np.full((dist_num,len(h)),np.nan) 
        
        ZH = -35.*np.ones((len(h),))
        ZDR = -35.*np.ones((len(h),))
        KDP = np.zeros((len(h),))
        RHOHV = np.ones((len(h),))
        Dm_tot = np.full((len(h),),np.nan) 
        
        if self.int_type==0:
            
            for ff in range(len(h)):
                
                Zh = 0. 
                Zv = 0. 
                Zhhvv = 0. 
                
                for d1 in range(dist_num):
                    
                    am = self.full[d1,ff,tind].am
                    bm = self.full[d1,ff,tind].bm
                    
                    M3[d1,ff] = (am**(-3./bm)*self.full[d1,ff,tind].moments(3./bm)).sum()
                    M4[d1,ff] = (am**(-4./bm)*self.full[d1,ff,tind].moments(4./bm)).sum()
                      
                    N[d1,ff]  = np.nansum(self.full[d1,ff,tind].Nbins)
                    M[d1,ff]  = 1000.*np.nansum(self.full[d1,ff,tind].Mbins)
                    Rm[d1,ff] = 1000.*3.6*np.nansum(self.full[d1,ff,tind].Mfbins)
                    
                    Zh      += np.nansum(self.full[d1,ff,tind].zh)
                    Zv      += np.nansum(self.full[d1,ff,tind].zv)
                    Zhhvv   += np.nansum(self.full[d1,ff,tind].zhhvv)
                    KDP[ff] += np.nansum(self.full[d1,ff,tind].KDP)
            
    
                if Zv>0.:
                    ZDR[ff] = 10.*np.log10(Zh/Zv)
                  
                ZH[ff] = 10.*np.log10(Zh)    
                  
                RHOHV_denom = np.sqrt(Zh*Zv)
                if RHOHV_denom>0.:
                    RHOHV[ff] = np.abs(Zhhvv)/RHOHV_denom
                
            # M3_tot = np.nansum(M3,axis=0)
            # M4_tot = np.nansum(M4,axis=0)
            
            # M3_tot[M3_tot==0.] = np.nan
            
            # Dm_tot = M4_tot/M3_tot
            
            # M3[M3==0.] = np.nan 
            
            # Dm = M4/M3
            
            # N_tot = np.nansum(N,axis=0)
            # M_tot = np.nansum(M,axis=0)
            # Rm_tot = np.nansum(Rm,axis=0)
        
        else:
            for ff in range(len(h)):
                
                Zh = 0. 
                Zv = 0. 
                Zhhvv = 0. 
                
                for d1 in range(dist_num):
                    
                    am = self.full[d1,ff].am
                    bm = self.full[d1,ff].bm
                    
                    M3[d1,ff] = (am**(-3./bm)*self.full[d1,ff].moments(3./bm)).sum()
                    M4[d1,ff] = (am**(-4./bm)*self.full[d1,ff].moments(4./bm)).sum()
                      
                    N[d1,ff]  = np.nansum(self.full[d1,ff].Nbins)
                    M[d1,ff]  = 1000.*np.nansum(self.full[d1,ff].Mbins)
                    Rm[d1,ff] = 1000.*3.6*np.nansum(self.full[d1,ff].Mfbins)
                    
                    Zh      += np.nansum(self.full[d1,ff].zh)
                    Zv      += np.nansum(self.full[d1,ff].zv)
                    Zhhvv   += np.nansum(self.full[d1,ff].zhhvv)
                    KDP[ff] += np.nansum(self.full[d1,ff].KDP)
            
                if Zv>0.:
                    ZDR[ff] = 10.*np.log10(Zh/Zv)
                  
                ZH[ff] = 10.*np.log10(Zh)    
                  
                RHOHV_denom = np.sqrt(Zh*Zv)
                if RHOHV_denom>0.:
                    RHOHV[ff] = np.abs(Zhhvv)/RHOHV_denom
                
            # M3_tot = np.nansum(M3,axis=0)
            # M4_tot = np.nansum(M4,axis=0)
            
            # M3_tot[M3_tot==0.] = np.nan
            
            # Dm_tot = M4_tot/M3_tot
            
            # M3[M3==0.] = np.nan 
            
            # Dm = M4/M3
            
            # N_tot = np.nansum(N,axis=0)
            # M_tot = np.nansum(M,axis=0)
            # Rm_tot = np.nansum(Rm,axis=0)


        M3_tot = np.nansum(M3,axis=0)
        M4_tot = np.nansum(M4,axis=0)
        
        M3_tot[M3_tot==0.] = np.nan
        
        Dm_tot = M4_tot/M3_tot
        
        M3[M3==0.] = np.nan 
        
        Dm = M4/M3
        
        N_tot = np.nansum(N,axis=0)
        M_tot = np.nansum(M,axis=0)
        Rm_tot = np.nansum(Rm,axis=0)

        if self.int_type==2: # Box model
            
            fig, ax = plt.subplots(2,4,figsize=(14,8),sharey=True)
            
            ax[0,0].plot(self.t,N_tot,'k')
            ax[0,1].plot(self.t,Dm_tot,'k')
            ax[0,2].plot(self.t,M_tot,'k')
            ax[0,3].plot(self.t,Rm_tot,'k')
            
            ax[1,0].plot(self.t,ZH,color='k')
            ax[1,1].plot(self.t,ZDR,color='k')
            ax[1,2].plot(self.t,KDP,color='k')
            ax[1,3].plot(self.t,RHOHV,color='k')
            
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
                    
                    ax[0,0].plot(self.t,N[d1,:],label='dist {}'.format(d1+1))
                    ax[0,1].plot(self.t,Dm[d1,:])
                    ax[0,2].plot(self.t,M[d1,:])
                    ax[0,3].plot(self.t,Rm[d1,:])
   
        else:
            #fig, ax = plt.subplots(1,3,figsize=(12,6),sharey=True)
            
            fig, ax = plt.subplots(2,4,figsize=(14,8),sharey=True)
            
            ax[0,0].plot(N_tot,self.z/1000.,'k',label='total')
            ax[0,1].plot(Dm_tot,self.z/1000.,'k')
            ax[0,2].plot(M_tot,self.z/1000.,'k')
            ax[0,3].plot(Rm_tot,self.z/1000.,'k')
            
            ax[1,0].plot(ZH,self.z/1000.,color='k')
            ax[1,1].plot(ZDR,self.z/1000.,color='k')
            ax[1,2].plot(KDP,self.z/1000.,color='k')
            ax[1,3].plot(RHOHV,self.z/1000.,color='k')
            
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
                    ax[0,0].plot(N[d1,:],self.z/1000.,label='dist {}'.format(d1+1))
                    ax[0,1].plot(Dm[d1,:],self.z/1000.)
                    ax[0,2].plot(M[d1,:],self.z/1000.)
                    ax[0,3].plot(Rm[d1,:],self.z/1000.)

        ax[0,0].legend(loc='upper center')
    
        fig.tight_layout()  
        
        return fig, ax   
 
    def plot_init(self,log_switch=True,x_axis='mass'):

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize=16) 
        plt.rc('ytick', labelsize=16)         

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
        ax[1].plot(np.log10(xbins),1000.*mbins**2*n_init,'k')
        
        ax[0].set_ylabel(ylabel_num)
        ax[1].set_ylabel(ylabel_mass)
        ax[1].set_xlabel(xlabel)
        
        #print('Initial Number = {:.2f} #/L'.format(np.nansum(mbins*n_init*(np.log10(medges[1:])-np.log10(medges[:-1])))))
        #print('Initial Mass = {:.2f} g/cm^3'.format(np.nansum(1000.*mbins**2*n_init*(np.log10(medges[1:])-np.log10(medges[:-1])))))
        
        #print('number test size=',np.nansum(mbins*n_init*(np.log10(dedges[1:])-np.log10(dedges[:-1]))))
        #print('mass test size=',np.nansum(1000.*mbins**2*n_init*(np.log10(dedges[1:])-np.log10(dedges[:-1]))))
        
 
        return fig, ax
    
    
    def plot_dists(self,tind=-1,hind=-1,x_axis='mass',y_axis='mass',xscale='log',yscale='linear',distscale='log',scott_solution=False,feingold_solution=False,plot_habits=False):
   # def plot_dists(self,*inds,x_axis='mass',y_axis='mass',xscale='log',yscale='linear',distscale='log',scott_solution=False,feingold_solution=False,plot_habits=False):
                
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize=16) 
        plt.rc('ytick', labelsize=16) 
        
        # NOTE: probably need to figure out how to deal with x_axis='size' when
        # am and bm parameters are different for each habit.
        
        fig, ax = plt.subplots(2,1,figsize=((8,10)),sharex=True)
        
       # if (len(self.t)>1) & (len(self.z)==1):
            
        
        print('Plotting distributions...')
        
        if self.int_type==0:
            primary_init  = self.full[0,0,0]
        else:
            primary_init = self.full[0,0]
        
        mbins = primary_init.xbins.copy() 
        
        xp1 = primary_init.x1
        xp2 = primary_init.x2
        ap = primary_init.aki
        cp = primary_init.cki
        
       # xbins = np.full((self.dnum,self.bins),np.nan)
        prefN = np.full((self.dnum,self.bins),np.nan)
        prefM = np.full((self.dnum,self.bins),np.nan)
        x1_final = np.full((self.dnum,self.bins),np.nan)
        x2_final = np.full((self.dnum,self.bins),np.nan)
        ak_final = np.full((self.dnum,self.bins),np.nan)
        ck_final = np.full((self.dnum,self.bins),np.nan)
        bm = np.full((self.dnum,),np.nan)
        am = np.full((self.dnum,),np.nan)

        if self.int_type==0:
            f_label = '{} km | {} min.'.format(self.z[hind]/1000.,self.t[tind]/60.)
            for d1 in range(self.dnum):
                x1_final[d1,:] = self.full[d1,hind,tind].x1 
                x2_final[d1,:] = self.full[d1,hind,tind].x2 
                ak_final[d1,:] = self.full[d1,hind,tind].aki 
                ck_final[d1,:] = self.full[d1,hind,tind].cki
                bm[d1] = self.full[d1,hind,tind].bm 
                am[d1] = self.full[d1,hind,tind].am 
                
        elif self.int_type==1:
            f_label = '{} km'.format(self.z[hind]/1000.)
            for d1 in range(self.dnum):
                x1_final[d1,:] = self.full[d1,hind].x1 
                x2_final[d1,:] = self.full[d1,hind].x2 
                ak_final[d1,:] = self.full[d1,hind].aki 
                ck_final[d1,:] = self.full[d1,hind].cki
                bm[d1] = self.full[d1,hind].bm 
                am[d1] = self.full[d1,hind].am 
                
        elif self.int_type==2:
            f_label = '{} min.'.format(self.t[tind]/60.)
            for d1 in range(self.dnum):
                x1_final[d1,:] = self.full[d1,tind].x1 
                x2_final[d1,:] = self.full[d1,tind].x2 
                ak_final[d1,:] = self.full[d1,tind].aki 
                ck_final[d1,:] = self.full[d1,tind].cki
                bm[d1] = self.full[d1,tind].bm 
                am[d1] = self.full[d1,tind].am 
            
   
        # Distscale toggles between dN/dlog(m) plots and dN/dm plots, for example.
        if distscale=='log':
        
            if x_axis=='mass': # plot dN/dlog(m) and dM/dlog(m)
            
                for d1 in range(self.dnum):
                    prefN[d1,:] = mbins*np.log(10)   
                    prefM[d1,:] = mbins**2*np.log(10)

                    #xbins[d1,:] = mbins.copy()
                xbins = mbins.copy()
                ylabel_num = r'dN/dlog(m)'
                ylabel_mass = r'dM/dlog(m)'
                
                xlabel = r'log(m) [log(g)]'
                
            elif x_axis=='size':  # plot dN/dlog(D) and dM/dlog(D)
                
                for d1 in range(self.dnum):
                    
                    prefN[d1,:] = mbins*bm[d1]*np.log(10) 
                    prefM[d1,:] = mbins**2*bm[d1]*np.log(10) 
                
                    #xbins[d1,:] = (mbins/self.full[d1,0].am)**(1./self.full[d1,0].bm)
               
                xbins = (mbins/am[0])**(1./bm[0])
   
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
                ylabel_num = r'dN/dm'
                ylabel_mass = r'dM/dm'
                
                xlabel = r'log(m) [log(g)]'
                
            elif x_axis=='size': # Linear plots of form dN/dD
                
                for d1 in range(self.dnum):
                    # CHECK
                    prefN[d1,:] = am[d1]**(1./bm[d1])*bm[d1]*mbins**(1.-1./bm[d1])
                    prefM[d1,:] = am[d1]**(1./bm[d1])*bm[d1]*mbins**(2.-1./bm[d1])

                    #xbins[d1,:] = (mbins/self.full[d1,0].am)**(1./self.full[d1,0].bm)
                xbins = (mbins/am[0])**(1./bm[0])
                ylabel_num = r'dN/dD'
                ylabel_mass = r'dM/dD'
         
        nN_init = prefN[0,:]*np.heaviside(mbins-xp1,1)*np.heaviside(xp2-mbins,1)*(ap*mbins+cp)
        nM_init = prefM[0,:]*np.heaviside(mbins-xp1,1)*np.heaviside(xp2-mbins,1)*(ap*mbins+cp)

        nN_final = prefN*np.heaviside(mbins[None,:]-x1_final,1)*np.heaviside(x2_final-mbins[None,:],1)*(ak_final*mbins[None,:]+ck_final)
        nM_final = prefM*np.heaviside(mbins[None,:]-x1_final,1)*np.heaviside(x2_final-mbins[None,:],1)*(ak_final*mbins[None,:]+ck_final)


        if xscale=='log':
            x = np.log10(xbins)
            
            if (x_axis=='size'):
                xlabel = r'log(D) [log(mm)]'
            elif (x_axis=='mass'):
                xlabel = r'log(m) [log(g)]'
            
        elif xscale=='linear':
            x = xbins.copy()
            ax[0].set_xlim((0.,5.))
           # ax[1].set_ylim((0.,10.))
             
            if (x_axis=='size'):
                xlabel = r'D (mm)'
            elif (x_axis=='mass'):
                xlabel = r'm (g)'
        
        ax[0].plot(x,nN_init,':k',label='initial')
        ax[0].plot(x,np.nansum(nN_final,axis=0),'k',label=f_label)
        if plot_habits:
            for d1 in range(self.dnum):
                ax[0].plot(x,nN_final[d1,:],label='dist {}'.format(d1+1))
            
        # Factor of 1000 comes from converting g to g/m^3
        ax[1].plot(x,1000.*nM_init,':k',label='initial')
        ax[1].plot(x,1000.*np.nansum(nM_final,axis=0),'k',label=f_label)
        if plot_habits:
            for d1 in range(self.dnum):
                ax[1].plot(x,1000.*nM_final[d1,:],label='dist {}'.format(d1+1))

        ax[0].set_ylabel(ylabel_num,fontsize=22)
        ax[1].set_ylabel(ylabel_mass,fontsize=22)
        
        ax[1].set_xlabel(xlabel,fontsize=22)
        
        if yscale=='log':
            ax[0].set_yscale('log')
            ax[1].set_yscale('log')
            ax[0].set_ylim((1e-5,max(nN_init.max(),1000.*nN_final.max())))
            ax[1].set_ylim((1e-5,max(nM_init.max(),1000.*nM_final.max())))

        #print('number test=',np.nansum(mbins*n_init*(np.log(medges[1:])-np.log(medges[:-1]))))
       # print('mass test=',np.nansum(mbins**2*1000.*n_init*(np.log(medges[1:])-np.log(medges[:-1]))))
        
        if (scott_solution & (self.int_type==2)):
            
            kernel_type = self.kernel
            
            if not (hasattr(self,'n_scott')):
                self.n_scott = Scott_dists(self.xbins,self.Eagg,self.mu0+1,self.t,kernel_type=kernel_type)
        
            ax[0].plot(x,prefN[0,:]*self.n_scott[:,tind],':r',label=f_label+ " analytical")
            ax[1].plot(x,1000.*prefM[0,:]*self.n_scott[:,tind],':r',label=f_label+ "analytical")
        
        if (feingold_solution & (self.int_type==2)):
            
            kernel_type = self.kernel
            
            C = self.Eagg 
            B = self.Ebr 
            
            if B>0.:
                if (C==0.):
                    kernel_type = 'SBE'
            
                elif (C>0.):
                    kernel_type = 'SCE/SBE'
                    
                if not (hasattr(self,'n_fein')):
                    self.n_fein = Feingold_dists(self.xbins,self.t,self.mu0+1,self.Eagg,self.Ebr,self.lamf,kernel_type=kernel_type)

                if kernel_type=='SBE':
                    ax[0].plot(x,prefN[0,:]*self.n_fein[:,tind],':r',label=f_label+ " analytical")
                    ax[1].plot(x,1000.*prefM[0,:]*self.n_fein[:,tind],':r',label=f_label+ " analytical")
                elif kernel_type=='SCE/SBE':
                    ax[0].plot(x,prefN[0,:]*self.n_fein,':r',label=f_label+ " analytical")
                    ax[1].plot(x,1000.*prefM[0,:]*self.n_fein,':r',label=f_label+ " analytical")
                    
        ax[0].legend() 
            
        plt.tight_layout()
        
        return fig, ax
     

    def plot_dists_height(self,tind=-1,plot_habits=False):
        
        #plt.rcParams['text.usetex'] = True
        
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize=16) 
        plt.rc('ytick', labelsize=16) 
        
        z = self.z/1000.
        
        z_lvls = np.arange(np.max(z),np.min(z)-1.,-1.)
        
        fig, ax = plt.subplots(len(z_lvls),1,figsize=(8,10),sharey=True,sharex=True)
        
        mbins = self.dist0.xbins.copy() 
        
        if self.int_type==0:
            primary_init  = self.full[0,0,tind]
        else:
            primary_init = self.full[0,0]
        
       # primary_init  = self.full[0,0]
        
        xp1 = primary_init.x1
        xp2 = primary_init.x2
        ap = primary_init.aki
        cp = primary_init.cki
        
        prefN_p = primary_init.am**(1./primary_init.bm)*primary_init.bm*mbins**(1.-1./primary_init.bm)

        xbins = (mbins/primary_init.am)**(1./primary_init.bm)
        
        nN_init = prefN_p*np.heaviside(mbins-xp1,1)*np.heaviside(xp2-mbins,1)*(ap*mbins+cp)
        
        ax[0].plot(xbins,nN_init,'k')

        ax[0].set_yscale('log')
        ax[0].set_xscale('linear')
        ax[0].set_ylim(bottom=0.001)
        
        ax[0].set_title('Height = {} km'.format(z_lvls[0]),fontsize=16)
        #ax[0].set_ylabel(r'Number Density (1/cm$^{3}$ 1/mm)',fontsize=16)
        ax[0].set_ylabel(r'n(D) (1/cm$^{3}$ 1/mm)',fontsize=16)
        
        ax[0].axes.tick_params(labelsize=16)
        ax[0].set_xlim((0.,5.))
        
        for hh in range(1,len(z_lvls)):
        
            zind = np.nonzero(z==z_lvls[hh])[0][0]
            
            
            nN_final = np.full((self.dnum,self.bins),np.nan)
            
            
            for d1 in range(self.dnum):

                
                if self.int_type==0:
                    xp1_final = self.full[d1,zind,tind].x1
                    xp2_final = self.full[d1,zind,tind].x2
                    ap_final  = self.full[d1,zind,tind].aki
                    cp_final  = self.full[d1,zind,tind].cki
                    prefN =self.full[d1,zind,tind].am**(1./self.full[d1,zind,tind].bm)*self.full[d1,zind,tind].bm*mbins**(1.-1./self.full[d1,zind,tind].bm)
                    
                else:
                    xp1_final = self.full[d1,zind].x1
                    xp2_final = self.full[d1,zind].x2
                    ap_final  = self.full[d1,zind].aki
                    cp_final  = self.full[d1,zind].cki
                    prefN =self.full[d1,zind].am**(1./self.full[d1,zind].bm)*self.full[d1,zind].bm*mbins**(1.-1./self.full[d1,zind].bm)
                
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
            
            ax[hh].set_title('Height = {} km'.format(z_lvls[hh]),fontsize=16)
            #ax[hh].set_ylabel(r'Number Density (1/cm$^{3}$ 1/mm)',fontsize=16)
            ax[hh].set_ylabel(r'n(D) (1/cm$^{3}$ 1/mm)',fontsize=16)
            
            ax[hh].axes.tick_params(labelsize=16)
            ax[hh].set_xlim((0.,5.))

        
        ax[0].set_ylim((1e-5,1e5))
        ax[-1].set_xlabel('Equivolume Diameter (mm)',fontsize=16)
        if plot_habits:
            ax[-1].legend()
        
        plt.tight_layout()
        
        return fig, ax
 
    
    def advance_1mom(self,M_old,dt):
        
        Mbins = np.zeros_like(M_old)
        
        M_net = self.Ikernel.interact_1mom(dt)
        
       # print('M_net=',M_net[1,:,:].sum())
        #raise Exception()
    
        M_sed = np.zeros((self.dnum,self.Hlen,self.bins)) 
       
        if self.boundary is None:
            M_sed[:,0,:] = (dt/self.dz)*(-self.Ikernel.Mfbins[:,0,:]) 
        
        M_sed[:,1:,:] = (dt/self.dz)*(self.Ikernel.Mfbins[:,:-1,:]-self.Ikernel.Mfbins[:,1:,:]) 
              
        #M_sed = (dt/self.dz)*convolve1d(self.Ikernel.Mfbins,self.adv_base,axis=1,mode='nearest')
        
        #M_sed = (dt/self.dz)*np.apply_along_axis(np.convolve(self.Ikernel.Mfbins,self.adv_base,mode='same'),axis=1)
        
        #if self.boundary is None:
        #    M_sed[:,0,:] = (dt/self.dz)*(-self.Ikernel.Mfbins[:,0,:]) 
        
        
        M_transfer = M_old+M_sed+M_net
        
        
        M_new = np.maximum(M_transfer,0.) # Should be positive if not over fragmented.
        Mbins[M_new>=0.] = M_new[M_new>=0.].copy()
        

        if self.boundary=='fixed': # If fixing top distribution. Can be helpful if trying to determine steady-state time.
            Mbins[:,0,:] = M_old[:,0,:].copy()
            
        dM = (Mbins-M_old)/dt
        
        
        return dM
    
    
    def advance_2mom(self,M_old,N_old,dt):
        
        Mbins = np.zeros_like(M_old)
        Nbins = np.zeros_like(N_old)
            
        #M_net, N_net = self.Ikernel.interact(dt)
        
        M_net, N_net = self.Ikernel.interact_2mom(dt)
       
        M_sed = np.zeros((self.dnum,self.Hlen,self.bins)) 
        N_sed = np.zeros((self.dnum,self.Hlen,self.bins)) 
       
        if self.boundary is None:
            M_sed[:,0,:] = (dt/self.dz)*(-self.Ikernel.Mfbins[:,0,:]) 
            N_sed[:,0,:] = (dt/self.dz)*(-self.Ikernel.Nfbins[:,0,:]) 
        
        M_sed[:,1:,:] = (dt/self.dz)*(self.Ikernel.Mfbins[:,:-1,:]-self.Ikernel.Mfbins[:,1:,:]) 
        N_sed[:,1:,:] = (dt/self.dz)*(self.Ikernel.Nfbins[:,:-1,:]-self.Ikernel.Nfbins[:,1:,:]) 
                  
        M_transfer = M_old+M_sed+M_net
        N_transfer = N_old+N_sed+N_net   
        
        M_new = np.maximum(M_transfer,0.) # Should be positive if not over fragmented.
        Mbins[M_new>=0.] = M_new[M_new>=0.].copy()
        
        N_new = np.maximum(N_transfer,0.) # Should be positive if not over fragmented.
        Nbins[N_new>=0.] = N_new[N_new>=0.].copy()
        
        if self.boundary=='fixed': # If fixing top distribution. Can be helpful if trying to determine steady-state time.
            Mbins[:,0,:] = M_old[:,0,:].copy()
            Nbins[:,0,:] = N_old[:,0,:].copy()
            
        dM = (Mbins-M_old)/dt
        dN = (Nbins-N_old)/dt
        
        return dM, dN
 

    def run_steady_state_1mom(self):
        
            self.full = np.empty((self.dnum,self.Tlen),dtype=object)
            
            for d1 in range(self.dnum):
                self.full[d1,0] = deepcopy(self.dists[d1,0])
                
            if self.int_type==1:    
                dh = np.vstack([self.dists[ff,0].dh for ff in range(self.dnum)])
            elif self.int_type==2:
                dh = self.dt*np.ones((self.dnum,self.bins))
            
            # ELD NOTE: At some point it probably will be worthwhile to do R-K time/height steps
            if self.Ecb>0.:
                pbar = tqdm(position=0, leave=True, mininterval=0,miniters=1,desc="Running 1D spectral bin model\n")
                for tt in tqdm(range(1,self.Tlen)):
    
                    #print('Running 1D spectral bin model: step = {} out of {}'.format(tt,self.Tlen-1))
        
                    M  = 0.
                    Mf = 0.
                    
                    for d1 in range(self.dnum):
                        M += np.nansum(self.dists[d1,0].Mbins) 
                        Mf+= np.nansum(self.dists[d1,0].Mbins*self.dists[d1,0].vt)
    
                    #print('Total Mass = {:.2f} g/m^3 | Total Mass Flux = {:.2f} g/(m^2*s)'.format(1000.*M,1000.*Mf))
    
                    pbar.set_description('Total Mass = {:.2f} g/m^3 | Total Mass Flux = {:.2f} g/(m^2*s)'.format(1000.*M,1000.*Mf))
                    
                    Mbins_old = self.Ikernel.Mbins.copy() 
  
                    M_net = self.Ikernel.interact_1mom(1.0)
                   
                    self.Ikernel.Mbins = np.maximum(Mbins_old+M_net*dh[:,None,:],0.)
                    self.Ikernel.Nbins = self.Ikernel.Mbins/self.xbins[None,None,:]
    
                    self.Ikernel.unpack_1mom() # Unpack the interaction 3D array to each object in the (dist x height) object array
                    self.Ikernel.pack(self.Ikernel.dists) # Update moments and parameters of 2D array of distribution objects
   
                    # Save dist copies at each time/height
                    #if np.isin(self.t[tt],self.tout):
                       # tf += 1
                        #print('Saving output')
                    for d1 in range(self.dnum):
                        self.full[d1,tt] = deepcopy(self.dists[d1,0])

    
    def run_full_1mom(self):
        ''' 
        Run bin model
        '''
         
        # Full is an object array that holds
        self.full = np.empty((self.dnum,self.Hlen,self.Tout_len),dtype=object)
        
        # use Butcher table to get rk order coefficients
        RK = init_rk(self.rk_order)
        a = RK['a']
        b = RK['b']
        
        rklen = len(b)
        
        tf = 0
        
        for hh in range(self.Hlen):
           for d1 in range(self.dnum):
               self.full[d1,hh,tf] = deepcopy(self.dists[d1,hh])
        
        pbar = tqdm(position=0, leave=True, mininterval=0,miniters=1,desc="Running 1D spectral bin model\n")
        for tt in tqdm(range(1,self.Tlen)):

            #print('Running 1D spectral bin model: step = {} out of {}'.format(tt,self.Tlen-1))
            #print('Total Mass = {:.2f} g/m^3 | Total Mass Flux = {:.2f} g/(m^2*s)'.format(1000.*np.nansum(self.Ikernel.Mbins),
           #                                                                               1000.*np.nansum(self.Ikernel.Mfbins)))
            
            pbar.set_description('Total Mass = {:.2f} g/m^3 | Total Mass Flux = {:.2f} g/(m^2*s)'.format(1000.*np.nansum(self.Ikernel.Mbins),
                                                                                                         1000.*np.nansum(self.Ikernel.Mfbins)))

            dM = np.zeros((self.dnum,self.Hlen,self.bins,rklen))
            
            M_old = self.Ikernel.Mbins.copy()

            # Generalized Explicit Runge-Kutta time steps
            # Keep in mind that for stiff equations higher
            # order Runge-Kutta steps might not be beneficial
            # due to stability issues.
            for ii in range(rklen):
                M_stage = np.maximum(M_old + self.dt*np.nansum(a[ii,:ii][None,None,None,:]*dM[:,:,:,:ii],axis=3),0.)
 
                dM[:,:,:,ii] = self.advance_1mom(M_stage,self.dt)
                
            self.Ikernel.Mbins = np.maximum(M_old + self.dt*np.nansum(b[None,None,None,:]*dM,axis=3),0.)
            self.Ikernel.Nbins = self.Ikernel.Mbins/self.xbins[None,None,:]
            
            self.Ikernel.unpack_1mom() # Unpack the interaction 3D array to each object in the (dist x height) object array
            self.Ikernel.pack(self.Ikernel.dists) # Update moments and parameters of 2D array of distribution objects

            if np.isin(self.t[tt],self.tout):
                tf += 1
                #print('Saving output')
                for hh in range(self.Hlen):
                   for d1 in range(self.dnum):
                       self.full[d1,hh,tf] = deepcopy(self.dists[d1,hh])

        
        # Clean up
        
        # if self.parallel:
            
        #     del self.Ikernel.dMb_gain_frac
        #     del self.Ikernel.dNb_gain_frac
        #     del self.Ikernel.PK   
        #     del self.Ikernel.kmin
        #     del self.Ikernel.kmid
        #     del self.Ikernel.cond_1
        #     del self.Ikernel.self_col
                 
        #     del self.Ikernel.dMi_loss
        #     del self.Ikernel.dMj_loss
        #     del self.Ikernel.dM_gain

        # delattr(self,'Ikernel')


    def run_steady_state_2mom(self):
        
            self.full = np.empty((self.dnum,self.Tlen),dtype=object)
            
            for d1 in range(self.dnum):
                self.full[d1,0] = deepcopy(self.dists[d1,0])
                
            if self.int_type==1:    
                dh = np.vstack([self.dists[ff,0].dh for ff in range(self.dnum)])
            elif self.int_type==2:
                dh = self.dt*np.ones((self.dnum,self.bins))
            
            # ELD NOTE: At some point it probably will be worthwhile to do R-K timesteps
            if self.Ecb>0.:
                
                pbar = tqdm(position=0, leave=True, mininterval=0,miniters=1,desc="Running 1D spectral bin model\n")
                #with tqdm(position=0, leave=True) as pbar:
                for tt in tqdm(range(1,self.Tlen)):
    
                    #print('Running 1D spectral bin model: step = {} out of {}'.format(tt,self.Tlen-1))

                    M  = 0.
                    Mf = 0.
                    
                    for d1 in range(self.dnum):
                        M += np.nansum(self.dists[d1,0].Mbins) 
                        Mf+= np.nansum(self.dists[d1,0].Mbins*self.dists[d1,0].vt)
    
                   # print('Total Mass = {:.2f} g/m^3 | Total Mass Flux = {:.2f} g/(m^2*s)'.format(1000.*M,1000.*Mf))
    
                    #tqdm.write('Running 1D spectral bin model: step = {} out of {}\n Total Mass = {:.2f} g/m^3 | Total Mass Flux = {:.2f} g/(m^2*s)'.format(tt,self.Tlen-1,1000.*M,1000.*Mf))
                    
                    #pbar.set_description('Running 1D spectral bin model: step = {} out of {}'.format(tt,self.Tlen-1))
                    
                    pbar.set_description('Total Mass = {:.2f} g/m^3 | Total Mass Flux = {:.2f} g/(m^2*s)'.format(1000.*M,1000.*Mf))
                    
                    #time.sleep(1)
                    #tqdm.write('Total Mass = {:.2f} g/m^3 | Total Mass Flux = {:.2f} g/(m^2*s)'.format(1000.*M,1000.*Mf))
    
                    #sys.stdout.flush()  # Flush the output stream after each iteration
    
                    Mbins_old = self.Ikernel.Mbins.copy() 
                    Nbins_old = self.Ikernel.Nbins.copy()
  
                    #M_net, N_net = self.Ikernel.interact(1.0)
                    
                    M_net, N_net = self.Ikernel.interact_2mom(1.0)
                   
                    self.Ikernel.Mbins = np.maximum(Mbins_old+M_net*dh[:,None,:],0.)
                    self.Ikernel.Nbins = np.maximum(Nbins_old+N_net*dh[:,None,:],0.)

                    self.Ikernel.unpack() # Unpack the interaction 3D array to each object in the (dist x height) object array
                    self.Ikernel.pack(self.Ikernel.dists) # Update moments and parameters of 2D array of distribution objects
   
                    # Save dist copies at each time/height
                    #if np.isin(self.t[tt],self.tout):
                       # tf += 1
                        #print('Saving output')
                    for d1 in range(self.dnum):
                        self.full[d1,tt] = deepcopy(self.dists[d1,0])

    
    def run_full_2mom(self):
        ''' 
        Run bin model
        '''
        
        # Full is an object array that holds
        self.full = np.empty((self.dnum,self.Hlen,self.Tout_len),dtype=object)
        
        # use Butcher table to get rk order coefficients
        RK = init_rk(self.rk_order)
        a = RK['a']
        b = RK['b']
        
        rklen = len(b)
        
        tf = 0
        
        for hh in range(self.Hlen):
            for d1 in range(self.dnum):
                self.full[d1,hh,tf] = deepcopy(self.dists[d1,hh])
        
        # ELD NOTE: 
            # Make sure to cap min M and N and 0 to prevent numerical errors.

        pbar = tqdm(position=0, leave=True, mininterval=0,miniters=1,desc="Running 1D spectral bin model\n")
        for tt in tqdm(range(1,self.Tlen)):

            #print('Running 1D spectral bin model: step = {} out of {}'.format(tt,self.Tlen-1))
            #print('Total Mass = {:.2f} g/m^3 | Total Mass Flux = {:.2f} g/(m^2*s)'.format(1000.*np.nansum(self.Ikernel.Mbins),
            #                                                                              1000.*np.nansum(self.Ikernel.Mfbins)))
            
            pbar.set_description('Total Mass = {:.2f} g/m^3 | Total Mass Flux = {:.2f} g/(m^2*s)'.format(1000.*np.nansum(self.Ikernel.Mbins),
                                                                                                         1000.*np.nansum(self.Ikernel.Mfbins)))

            dM = np.zeros((self.dnum,self.Hlen,self.bins,rklen))
            dN = np.zeros((self.dnum,self.Hlen,self.bins,rklen))
            
            M_old = self.Ikernel.Mbins.copy()
            N_old= self.Ikernel.Nbins.copy()
            
            # Generalized Explicit Runge-Kutta time steps
            # Keep in mind that for stiff equations higher
            # order Runge-Kutta steps might not be beneficial
            # due to stability issues.
            for ii in range(rklen):
                M_stage = np.maximum(M_old + self.dt*np.nansum(a[ii,:ii][None,None,None,:]*dM[:,:,:,:ii],axis=3),0.)
                N_stage = np.maximum(N_old + self.dt*np.nansum(a[ii,:ii][None,None,None,:]*dN[:,:,:,:ii],axis=3),0.)

                dM[:,:,:,ii], dN[:,:,:,ii] = self.advance_2mom(M_stage,N_stage,self.dt)
            
            Mbins = np.maximum(M_old + self.dt*np.nansum(b[None,None,None,:]*dM,axis=3),0.)
            Nbins = np.maximum(N_old + self.dt*np.nansum(b[None,None,None,:]*dN,axis=3),0.)
                                 
            self.Ikernel.Mbins = Mbins.copy()
            self.Ikernel.Nbins = Nbins.copy()

            self.Ikernel.unpack() # Unpack the interaction 3D array to each object in the (dist x height) object array
            self.Ikernel.pack(self.Ikernel.dists) # Update moments and parameters of 2D array of distribution objects


            if np.isin(self.t[tt],self.tout):
                tf += 1
                #print('Saving output')
                for hh in range(self.Hlen):
                   for d1 in range(self.dnum):
                       self.full[d1,hh,tf] = deepcopy(self.dists[d1,hh])

        


        # if self.parallel:
            
        #     del self.Ikernel.dMb_gain_frac
        #     del self.Ikernel.dNb_gain_frac
        #     del self.Ikernel.PK    
        #     del self.Ikernel.kmin
        #     del self.Ikernel.kmid
        #     del self.Ikernel.cond_1
        #     del self.Ikernel.self_col

        # delattr(self,'Ikernel')
        

    def run(self):
        ''' 
        Run bin model
        '''
        time_start = datetime.now()

        try:

            # If running one moment (mass) only
            if self.moments == 1:
                
                if self.int_type==0:
                    self.run_full_1mom()
                    
                else:
                    self.run_steady_state_1mom()
                
            # If running two moments (mass and number)
            if self.moments == 2:
    
                if self.int_type==0:
                    self.run_full_2mom()
                    
                else:
                    self.run_steady_state_2mom()
                    
        except Exception as E:
           print('\n{}'.format(E))
 
        finally:
            #Clean up temporary memory mapped files and potentially other things too.
            self.clean_up()
    
        time_end = datetime.now() 
        print('\nModel Complete! Time Elapsed = {:.2f} min'.format((time_end-time_start).total_seconds()/60.))
