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

# Number Distribution function class for arbitrary category
class dist():
    
    def __init__(self,sbin=4,bins=80,D1=0.01,x0=None,Nt0=1.,Mt0=1.,mu0=3,Dm0=2,gam_init=True,gam_norm=False,dist_var='mass',
                 kernel='Hydro',habit_dict=None,ptype='rain',Tc=10.,radar=False,mom_num=2,Mbins=None,Nbins=None):
        
        self.mom_num = mom_num
        
        if habit_dict is None:
            habit_dict = habits()[ptype]
        
        self.init_dist(sbin,bins,D1,dist_var=dist_var,kernel=kernel,habit_dict=habit_dict,ptype=ptype,x0=x0,Tc=Tc,radar=radar,mom_num=mom_num,gam_norm=gam_norm)
        
        if gam_init:
            self.bin_gamma_dist(Nt0=Nt0,Mt0=Mt0,mu0=mu0,Dm0=Dm0,normalize=gam_norm)
            
        if mom_num==2:
            if (Mbins is not None) and (Nbins is not None):
                self.Mbins = Mbins 
                self.Nbins = Nbins
            self.diagnose() 
            
        elif mom_num==1:
            if (Mbins is not None):
                self.Mbins = Mbins 
            self.diagnose_1mom()
        
    def init_dist(self,sbin,bins,D1,kernel='Hydro',habit_dict=None,ptype='rain',Tc=10.,dist_var='mass',x0=None,radar=False,mom_num=2,gam_norm=False):
        
        if habit_dict is None:
            habit_dict = habits()[ptype]
        
        self.radar = radar
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
        
        # if gam_norm:
        #     self.am = 1.0 
        #     self.bm = 3.0
        # else:
        #     self.am = habit_dict['am']    # Units: g * mm^(-(3+brho)) 
        #     self.bm = habit_dict['bm']
            
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
        
        # Set up particle properties for mass grid
        # Fall speed (m/s)
        self.vt = self.av*self.d**self.bv
        self.vt_edges = self.av*self.d_edges**self.bv
        
        self.vt[self.vt>10.] = 10.
        self.vt_edges[self.vt_edges>10.]=10.
        
        self.vt1 = self.vt_edges[:-1].copy() 
        self.vt2 = self.vt_edges[1:].copy()

        # Midpoint Area (mm^2)
        self.A = 0.25*np.pi*self.dmax**2.
        # Edge Area (mm^2)
        self.A_edges = 0.25*np.pi*self.d_edges**2.
        self.A1 = self.A_edges[:-1].copy() 
        self.A2 = self.A_edges[1:].copy()
            
        self.Mbins = np.zeros_like(self.xbins).astype(np.float64)
        self.Nbins = np.zeros_like(self.xbins).astype(np.float64)
        
        if radar:
            # Radar stuff
            self.wavl = 110.
            #self.sigma = 0.
            self.ew0 = complex(81.0, 23.2)  # Dielectric constant of water at 0C
            self.kw = (np.abs((self.ew0 - 1) / (self.ew0 + 2)))**2
            self.cz = (4.0 * self.wavl**4)/(np.pi**4 * self.kw)
            self.ckdp = (0.18 / np.pi) * self.wavl
            self.rhoi = 0.92
            
            # Calculate scattering amplitudes and whatnot
            if self.ptype=='rain':
                
                self.rho1 = 1.0 # g/cm^3
                self.rho2 = 1.0 # g/cm^3
                
                self.eps1 = dielectric_water(Tc+273.15,self.ew0)
                self.eps2 = self.eps1
                
                self.ar = 0.9951 + 0.0251*self.d-0.03644*self.d**2+0.00503*self.d**3-0.0002492*self.d**4
                self.ar1 = 0.9951 + 0.0251*self.d1-0.03644*self.d1**2+0.00503*self.d1**3-0.0002492*self.d1**4
                self.ar2 = 0.9951 + 0.0251*self.d2-0.03644*self.d2**2+0.00503*self.d2**3-0.0002492*self.d2**4
    
                self.ar[self.ar>1.0] = 1.0 
                self.ar[self.ar<0.56] = 0.56
                
                self.ar1[self.ar1>1.0] = 1.0 
                self.ar1[self.ar1<0.56] = 0.56
                
                self.ar2[self.ar2>1.0] = 1.0 
                self.ar2[self.ar2<0.56] = 0.56
                
            elif self.ptype=='snow':
                
                self.ar1 = self.ar*np.ones_like(self.d1)
                self.ar2 = self.ar*np.ones_like(self.d2)
                
                self.rho1 = self.arho*self.d1**(self.bm-3.)  
                self.rho1[self.rho1>self.rhoi] = 0.92 
                
                self.rho2 = self.arho*self.d2**(self.bm-3.)
                self.rho2[self.rho2>self.rhoi] = 0.92 
                
                epi = dielectric_ice(self.wavl,Tc+273.15)
                
                Ki = (epi-1.)/(epi+2.)
                
                self.eps1 = (1+2*(self.rho1/self.rhoi)*Ki)/(1-(self.rho1/self.rhoi)*Ki)
                self.eps2 = (1+2*(self.rho2/self.rhoi)*Ki)/(1-(self.rho2/self.rhoi)*Ki)
            
        
    def bin_gamma_dist(self,Nt0=1.,Mt0=1.,mu0=3,Dm0=2,normalize=False):
        
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

    def moments(self,r):  # Units are g^n
        # Integrate to find arbitrary moments of subgrid distribution Mn = Int x^n *[n(x)=ak*x+ck]*dx
        return self.aki*Pn(r+1,self.x1,self.x2)+self.cki*Pn(r,self.x1,self.x2)

    # Function for diagnosing linear distribution function following Wang et al. (2008)
    # NOTE: Need to clip xm to left/right bin boundaries
    def diagnose_1mom(self):
         
            
        self.Nbins = self.Mbins/self.xbins
        
        self.n1 = self.n2 = self.cki = self.Mbins/self.dxi 
               
            
        if self.radar:
            self.radar_bins() 
                  
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


    # Function for diagnosing linear distribution function following Wang et al. (2008)
    # NOTE: Need to clip xm to left/right bin boundaries
    def diagnose(self):

        dx = self.xi2-self.xi1 #?
    
        dx2 = dx**2

        xm = self.xbins.copy()
        
        Mbins = self.Mbins.copy() # Mass will be conserved totally
        Nbins = self.Nbins.copy() # Might need to adjust Number if xm is too large or too small
        
        xm[Nbins>0.] = Mbins[Nbins>0.]/Nbins[Nbins>0.]

        xm1 = xm/self.xi1 #?

        cond_null = (Mbins==0.)|(Nbins==0.)   
        cond_a = ((2.+self.rhobins)/3.<=xm1) & (xm1<=(1.+2.*self.rhobins)/3.) & (~cond_null) 
        cond_b = (1.<= xm1) & (xm1 < (2.+self.rhobins)/3.) & (~cond_null) 
        cond_c = ((1.+2.*self.rhobins)/3. < xm1) & (xm1<=self.rhobins) & (~cond_null) 
        
        x1i = self.xi1.copy()
        x2i = self.xi2.copy()
        n1i = np.zeros_like(Mbins)
        n2i = np.zeros_like(Mbins) 
        flag = np.zeros_like(Mbins) 
        
        # Scenario a: distribution spans full bin
        #x1i[cond_a] = self.xi1[cond_a] 
        #x2i[cond_a] = self.xi2[cond_a] 
        n1i[cond_a] = 2*(Nbins[cond_a]*(self.xi1[cond_a]+2.*self.xi2[cond_a])-3.*Mbins[cond_a])/(dx2[cond_a])
        n2i[cond_a] = 2*(-Nbins[cond_a]*(2.*self.xi1[cond_a]+self.xi2[cond_a])+3.*Mbins[cond_a])/(dx2[cond_a])
        flag[cond_a] = 1

        # Scenario b: n1>0., n2==0.
        #x1i[cond_b] = self.xi1[cond_b] 
        x2i[cond_b] = self.xi1[cond_b]+3.*(xm[cond_b]-self.xi1[cond_b])
        n1i[cond_b] = 2.*Nbins[cond_b]/(3.*(xm[cond_b]-self.xi1[cond_b]))
        n2i[cond_b] = 0.
        flag[cond_b]= 2
        
        # Scenario c: n2>0., n1==0.
        x1i[cond_c] = self.xi2[cond_c]-3.*(self.xi2[cond_c]-xm[cond_c]) 
        #x2i[cond_c] = self.xi2[cond_c] 
        n1i[cond_c] = 0. 
        n2i[cond_c] = 2.*Nbins[cond_c]/(3.*(self.xi2[cond_c]-xm[cond_c]))
        flag[cond_c]= 3
        
        # Null Scenario
        flag[cond_null] = 0

        self.aki = (n2i-n1i)/dx # Slope of linear subgrid distribution
        self.cki = (n1i*x2i-x1i*n2i)/dx # intercept of linear subgrid distribution
        
        # Update parameters
        self.x1 = x1i.copy() 
        self.x2 = x2i.copy()
        self.n1 = n1i.copy() 
        self.n2 = n2i.copy() 
        self.flag = flag.copy()
            
        if self.radar:
            self.radar_bins() 
                  
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
        
        angs = angular_moments(self.sigma)
        
        ang1 = angs[0]
        ang2 = angs[1]
        ang3 = angs[2]
        ang4 = angs[3]
        ang5 = angs[4]
        #ang6 = angs[5]
        ang7 = angs[6]
        
        la1, lb1 = spheroid_factors(self.ar1)
        la2, lb2 = spheroid_factors(self.ar2)
        
        fhh_180_1 = fhh_0_1 = (((np.pi**2 * (self.d1)**3) / (6 * self.wavl**2)) * (1 / (lb1 + (1 / (self.eps1 -1)))))  
        fvv_180_1 = fvv_0_1 = (((np.pi**2 * (self.d1)**3) / (6 * self.wavl**2)) * (1 / (la1 + (1 / (self.eps1 -1)))))
        
        fhh_180_2 = fhh_0_2 = (((np.pi**2 * (self.d2)**3) / (6 * self.wavl**2)) * (1 / (lb2 + (1 / (self.eps2 -1)))))  
        fvv_180_2 = fvv_0_2 = (((np.pi**2 * (self.d2)**3) / (6 * self.wavl**2)) * (1 / (la2 + (1 / (self.eps2 -1)))))
        
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
