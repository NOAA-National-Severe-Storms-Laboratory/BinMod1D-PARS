# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 11:31:38 2025

@author: edwin.dunnavan
"""
## Import stuff
import numpy as np

from collection_kernels import Prod_kernel, Constant_kernel, Golovin_kernel, hydro_kernel
from bin_integrals import In_int, gam_int, LGN_int, integrate_rect_kernel, integrate_tri_kernel

class Interaction():
    
    '''
    interaction class initializes interaction objects that contain
    arrays in the form (Ndists x M x N) which specifies the interactions among
    all Ndists distributions.
    '''
    
    def __init__(self,dists,cc_dest,br_dest,Eagg,Ecb,Ebr,frag_dict=None,kernel='Golovin'):
        
        # cc_dest is an integer (from 1 to len(dists)) that determines the destination 
        # for coalesced particles
        
        # br_dest is an integer  (from 1 to len(dists)) that determines the destination
        # for fragments
        
        self.dists = dists
        self.frag_dict = frag_dict
        self.kernel = kernel
        self.indc = cc_dest -1
        self.indb = br_dest-1
        self.Eagg = Eagg 
        self.Ebr = Ebr 
        self.Ecb = Ecb
        
        self.dnum, self.Hlen = np.shape(self.dists)
        
        # Ensure that cc_dest and br_dest are valid     
        if (cc_dest<1) | (cc_dest>self.dnum):
            print('cc_dest needs to be between 1 and {}'.format(self.dnum))
            raise Exception()
        if (br_dest<1) | (br_dest>self.dnum):
            print('br_dest needs to be between 1 and {}'.format(self.dnum))
            raise Exception()
        
        # NOTE: Assume that ALL distributions have same bin grids
        self.bins = dists[0,0].bins
        self.sbin = dists[0,0].sbin
        
        self.xi1 = dists[0,0].xi1 
        self.xi2 = dists[0,0].xi2
        xedges   = dists[0,0].xedges
        
        self.ind_i, self.ind_j = np.meshgrid(np.arange(0,self.bins,1),np.arange(0,self.bins,1),indexing='ij')    
        
        kbin_min = self.xi1[:,None]+self.xi1[None,:]
        
        idx_min = np.searchsorted(xedges, kbin_min, side='right')-1
        
        # Clamp values to valid bin range [0, B-1]
        self.kmin = np.clip(idx_min, 0, self.bins-1) 
        self.kmid = np.minimum(self.kmin+1,self.bins-1)

        # For self-collection on mass-doubling grid, gain bounds cover exactly one bin
        kdiag = np.diag_indices(self.kmin.shape[0])
        self.kmin[kdiag] = kdiag[0]+self.sbin
        self.kmid[kdiag] = kdiag[0]+self.sbin
          
        self.kmin = np.clip(self.kmin,0,self.bins-1)
        self.kmid = np.clip(self.kmid,0,self.bins-1)
        
        #self.kmin = np.tile(np.clip(self.kmin,0,self.bins-1),(self.Hlen,1,1))
        #self.kmid = np.tile(np.clip(self.kmid,0,self.bins-1),(self.Hlen,1,1))
        
       # print('kmin_3d=',self.kmin.shape)
       # raise Exception()
         
        self.cond_1 = np.tile(((self.ind_i>=(self.bins-self.sbin)) | (self.ind_j>=(self.bins-self.sbin))),(self.Hlen,1,1))
        
        if self.Ebr>0.: # Setup fragment distribution if Ebr>0.     
            self.setup_fragments()
            
        self.PK = self.create_kernels(dists)
        
        # Create dist_num x height x bins arrays for N and M 
        self.pack(dists)
    
    def unpack(self):
        '''
        Updates 2D array (dist_num x Height) of distribution objects with new 
        distribution parameters calculated from Interaction object.

        Returns
        -------
        None.

        '''
        
        for zz in range(self.Hlen):
            for d1 in range(self.dnum):
                self.dists[d1,zz].Mbins  = self.Mbins[d1,zz,:].copy()
                self.dists[d1,zz].Nbins  = self.Nbins[d1,zz,:].copy()
                self.dists[d1,zz].diagnose()
    
    def pack(self,dists):
        '''
        Updates Ikernel with (dist_num x height x bins) array of distribution
        parameters

        Parameters
        ----------
        dists : Object
            3D Array of distribution objects

        Returns
        -------
        None.

        '''
        self.Mbins  = np.zeros((self.dnum,self.Hlen,self.bins))
        self.Nbins  = np.zeros((self.dnum,self.Hlen,self.bins))
        self.Mfbins  = np.zeros((self.dnum,self.Hlen,self.bins))
        self.Nfbins  = np.zeros((self.dnum,self.Hlen,self.bins))
        self.x1     = np.zeros((self.dnum,self.Hlen,self.bins))
        self.x2     = np.zeros((self.dnum,self.Hlen,self.bins))
        self.aki    = np.zeros((self.dnum,self.Hlen,self.bins))
        self.cki    = np.zeros((self.dnum,self.Hlen,self.bins))
        
        for zz in range(self.Hlen):
            for d1 in range(self.dnum):
                self.Mbins[d1,zz,:] = dists[d1,zz].Mbins.copy() 
                self.Nbins[d1,zz,:] = dists[d1,zz].Nbins.copy()
                self.Mfbins[d1,zz,:] = dists[d1,zz].Mfbins.copy() 
                self.Nfbins[d1,zz,:] = dists[d1,zz].Nfbins.copy()
                self.x1[d1,zz,:]    = dists[d1,zz].x1.copy() 
                self.x2[d1,zz,:]    = dists[d1,zz].x2.copy() 
                self.aki[d1,zz,:]   = dists[d1,zz].aki.copy() 
                self.cki[d1,zz,:]   = dists[d1,zz].cki.copy() 
                
           
    def create_kernels(self,dists):
        
        dlen = len(dists)
        
        # Kernel ind, x, y
        # NOTE: HK also has denominator bin width terms for x and y
        HK = np.ones((4,dlen,dlen,self.bins,self.bins))

        ## Calculate Bilinear interpolation of collection kernel
        # NOTE: fkernel form can be expressed in terms of K(x,y) = a + b*x +c*y +d*x*y
        ## ELD: Does kernel need to be evaluated at rectangle and triangle endpoints for each integral?
   
        for d1 in range(dlen):
            for d2 in range(dlen):
               
                dist1 = dists[d1,0] 
                dist2 = dists[d2,0]
   
                # Calculate corners of K(x,y), i.e. K(x0,y0), K(x1,y0), K(x0,y1), K(x1,y1)
                if self.kernel=='Hydro':
                    HK[0,d1,d2,:,:] = hydro_kernel(dist1.vt1,dist2.vt1,dist1.A1,dist2.A1)
                    HK[1,d1,d2,:,:] = hydro_kernel(dist1.vt2,dist2.vt1,dist1.A2,dist2.A1)
                    HK[2,d1,d2,:,:] = hydro_kernel(dist1.vt1,dist2.vt2,dist1.A1,dist2.A2)
                    HK[3,d1,d2,:,:] = hydro_kernel(dist1.vt2,dist2.vt2,dist1.A2,dist2.A2)
                
                elif self.kernel == 'Golovin':
                    HK[0,d1,d2,:,:] = Golovin_kernel(dist1.x1,dist2.x1)
                    HK[1,d1,d2,:,:] = Golovin_kernel(dist1.x2,dist2.x1)
                    HK[2,d1,d2,:,:] = Golovin_kernel(dist1.x1,dist2.x2)
                    HK[3,d1,d2,:,:] = Golovin_kernel(dist1.x2,dist2.x2)
                    
                elif self.kernel == 'Product':
                    HK[0,d1,d2,:,:] = Prod_kernel(dist1.x1,dist2.x1)
                    HK[1,d1,d2,:,:] = Prod_kernel(dist1.x2,dist2.x1)
                    HK[2,d1,d2,:,:] = Prod_kernel(dist1.x1,dist2.x2)
                    HK[3,d1,d2,:,:] = Prod_kernel(dist1.x2,dist2.x2)
                    
                elif self.kernel == 'Constant':
                    HK[0,d1,d2,:,:] = Constant_kernel(dist1.x1,dist2.x1)
                    HK[1,d1,d2,:,:] = Constant_kernel(dist1.x2,dist2.x1)
                    HK[2,d1,d2,:,:] = Constant_kernel(dist1.x1,dist2.x2)
                    HK[3,d1,d2,:,:] = Constant_kernel(dist1.x2,dist2.x2)
                            
        # Rearranged weights for Kernel in form: K(x,y) = a + b*x + c*y + d*x*y
        PK = np.zeros_like(HK)  
        PK[3,:,:,:,:] =  (HK[0,:,:,:,:]+HK[3,:,:,:,:]-(HK[1,:,:,:,:]+HK[2,:,:,:,:]))\
                       /(dist1.dxbins[None,None,:,None]*dist2.dxbins[None,None,None,:])
        PK[1,:,:,:,:] = ((HK[1,:,:,:,:]-HK[0,:,:,:,:])/dist1.dxbins[None,None,:,None])\
                       -PK[3,:,:,:,:]*dist2.xi2[None,None,None,:] 
        PK[2,:,:,:,:] = ((HK[2,:,:,:,:]-HK[0,:,:,:,:])/dist2.dxbins[None,None,None,:])\
                       -PK[3,:,:,:,:]*dist1.xi1[None,None,:,None] 
        PK[0,:,:,:,:] =   HK[0,:,:,:,:]\
                       -PK[1,:,:,:,:]*dist1.xi1[None,None,:,None]\
                       -PK[2,:,:,:,:]*dist2.xi1[None,None,None,:]\
                       -PK[3,:,:,:,:]*dist1.xi1[None,None,:,None]*dist2.xi1[None,None,None,:]
        
        PK[np.abs(PK)<1e-10] = 0.
        
        return PK


    def setup_fragments(self):
            self.dMb_gain_frac = np.zeros((self.bins,self.bins))
            self.dNb_gain_frac = np.zeros((self.bins,self.bins))
            
            if self.kernel=='Hydro':
            
                if self.frag_dict['dist']=='exp':
                    IF_func = lambda n,x1,x2: gam_int(n,0.,self.frag_dict['Dmf'],x1,x2)
                elif self.frag_dict['dist']=='gamma':
                    IF_func = lambda n,x1,x2: gam_int(n,self.frag_dict['muf'],self.frag_dict['Dmf'],x1,x2)   
                elif self.frag_dict['dist']=='LGN':
                    # TEMP
                    Df_med = self.frag_dict['Df_med'] # mm
                    muf = np.log(Df_med)
                    Df_mode = self.frag_dict['Df_mode']
                    sig2f = muf-np.log(Df_mode)
                    IF_func = lambda n,x1,x2:LGN_int(n,muf,sig2f,x1,x2)
            
                for xx in range(self.bins):
                    for kk in range(0,xx+1):
                        self.dMb_gain_frac[kk,xx] = self.dists[self.indb,0].am*IF_func(self.dists[self.indb,0].bm,self.dists[self.indb,0].d1[kk],self.dists[self.indb,0].d2[kk])
                        self.dNb_gain_frac[kk,xx] = IF_func(0.,self.dists[self.indb,0].d1[kk],self.dists[self.indb,0].d2[kk])
                  
            else:
                for xx in range(self.bins): # m1+m2 breakup mass
                   for kk in range(0,xx+1): # breakup gain bins
                       self.dMb_gain_frac[kk,xx] = In_int(1.,self.frag_dict['lamf'],self.xi1[kk],self.xi2[kk])
                       self.dNb_gain_frac[kk,xx] = In_int(0.,self.frag_dict['lamf'],self.xi1[kk],self.xi2[kk])   
                           
            self.dMb_gain_frac[self.dMb_gain_frac<0.]=0.
            dMb_gain_tot = np.nansum(self.dMb_gain_frac,axis=0)
            dMb_gain_tot[dMb_gain_tot==0.] = np.nan
            
            self.dMb_gain_frac = self.dMb_gain_frac/dMb_gain_tot[None,:]
            self.dNb_gain_frac = self.dNb_gain_frac/dMb_gain_tot[None,:]
            
            self.dMb_gain_frac[np.isnan(dMb_gain_tot)] = 0.
            self.dNb_gain_frac[np.isnan(dMb_gain_tot)] = 0.

      
    def calculate(self,ind1,ind2,PK,self_col=False):
        
        '''
         This function calculate mass and number transfer rates
         for collision-coalescence and collisional breakup between
         each distribution.
        '''
        
        # (height x bins)
        x11  = self.x1[ind1,:,:]
        x21  = self.x2[ind1,:,:]
        ak1  = self.aki[ind1,:,:]
        ck1  = self.cki[ind1,:,:] 
        
        x12  = self.x1[ind2,:,:]
        x22  = self.x2[ind2,:,:]
        ak2  = self.aki[ind2,:,:] 
        ck2  = self.cki[ind2,:,:] 
        
        '''
        Check edges and find integration regions in source space
        ''' 
        # (heights x bins x bins)
        x_bottom_edge = (self.xi2[self.kmin][None,:,:]-x12[:,None,:])
        x_top_edge = (self.xi2[self.kmin][None,:,:]-x22[:,None,:])
        y_left_edge = (self.xi2[self.kmin][None,:,:]-x11[:,:,None])
        y_right_edge = (self.xi2[self.kmin][None,:,:]-x21[:,:,None])
        
        check_bottom = (x11[:,:,None]<x_bottom_edge) &\
                       (x21[:,:,None]>x_bottom_edge)
         
        check_top = (x11[:,:,None]<x_top_edge) &\
                    (x21[:,:,None]>x_top_edge)
                    
        check_left = (x12[:,None,:]<y_left_edge) &\
                     (x22[:,None,:]>y_left_edge)
                    
        check_right = (x12[:,None,:]<y_right_edge) &\
                      (x22[:,None,:]>y_right_edge)            
               
        check_middle = ((0.5*(x11[:,:,None]+x21[:,:,None]))+(0.5*(x12[:,None,:]+x22[:,None,:])))<(self.xi2[self.kmin][None,:,:])
                   
        # If opposite sides check true, then integral region is rectangle + triangle
        # If adjacent sides check true, then integral region is triangle       
               
        # Check which opposite side is higher for cases where we have rectangle + triangle
        # NOTE: It SHOULD be the case that y_left_edge>y_right_edge and x_bottom_edge>x_top_edge
        # This just has to do with the geometry of the x+y mapping, i.e., the x+y lines have negative slope.
        
        '''
        Vectorized Integration Regions:
        cond_1 :  Ignore CC process for these source bins; they don't map to the largest avail bin.
        cond_2 :  k bin: Lower triangle region. Just clips BR corner.
                           Triangle = ((xi1,xj1),(xi1,y_left_edge),(xi2,xj1))
        cond_3 :  k bin: Lower triangle region. Just clips UL corner.
                           Triangle = ((xi1,xj1),(xi1,xj2),(x_bottom_edge,xj1))  
        cond_4 :  Full Rectangular source region based on self collection: ii == jj --> ii+sbin or jj+sbin
        cond_5 :  k bin: Top/Bottom clip: Rectangle on left, triangle on right
                                  Rectangle = ((xi1,xj1),(xi1,xj2),(x_top_edge,xj2),(x_top_edge,xj1))
                                  Triangle  = ((x_top_edge,xj1),(x_top_edge,xj2),(x_bottom_edge,xj1))
        cond_6 :  k bin: Left/Right clip: Rectangle on bottom, triangle on top
                                  Rectangle = ((xi1,xj1),(xi1,y_right_edge),(xi2,y_right_edge),(xi2,xj1))
                                  Triangle  = ((xi1,y_right_edge),(xi1,y_left_edge),(xi2,y_right_edge))
        cond_7 :  k+1 bin: Triangle in top right corner
                                    Triangle = ((x_top_edge,xj2),(xi2,xj2),(xi2,y_right_edge))
        cond_8 :  k bin: Triangle in lower left corner
                                    Triangle = ((xi1,xj1),(xi1,y_left_edge),(x_bottom_edge,xj1))
        '''
        
        # Initialize gain term arrays
        dMi_loss = np.zeros((self.Hlen,self.bins,self.bins))
        dMj_loss = np.zeros((self.Hlen,self.bins,self.bins))
        dNi_loss = np.zeros((self.Hlen,self.bins,self.bins))
        dM_gain  = np.zeros((self.Hlen,self.bins,self.bins,2))
        dN_gain  = np.zeros((self.Hlen,self.bins,self.bins,2))
        Mb_gain  = np.zeros((self.Hlen,self.bins))
        Nb_gain  = np.zeros((self.Hlen,self.bins))
        
        cond = np.zeros((self.Hlen,self.bins,self.bins),dtype=int)
        
        if self_col:
            sc_inds = np.tile(np.triu(np.ones((self.bins,self.bins),dtype=int),k=0),(self.Hlen,1,1))
            
        else:
            sc_inds = np.ones((self.Hlen,self.bins,self.bins),dtype=int)
            
        cond_touch = (check_bottom|check_top|check_left|check_right)
        cond_2_corner = (x21==self.xi2[None,:])[:,None,:]&(x12==self.xi1[None,:])[:,:,None]&(check_left)
        cond_3_corner = (x11==self.xi1[None,:])[:,None,:]&(x12==self.xi1[None,:])[:,:,None]&(check_bottom)
        #cond_1 = np.tile((self.ind_i>=(self.bins-self.sbin)) | (self.ind_j>=(self.bins-self.sbin)),(self.Hlen,1,1))
        cond_2 = np.eye(self.bins,k=1,dtype=bool)[None,:,:] & (cond_2_corner) & (~self.cond_1) & (cond_touch)
        cond_3 = np.eye(self.bins,k=-1,dtype=bool)[None,:,:] & (cond_3_corner) & (~self.cond_1) & (cond_touch)
        cond_4 = np.eye(self.bins,dtype=bool)[None,:,:] & (~self.cond_1)
        cond_nt = (~(self.cond_1|cond_2|cond_3|cond_4))
        cond_5 = (check_top&check_bottom)  & cond_nt
        cond_6 = (check_left&check_right)  & cond_nt
        cond_7 =  (check_right&check_top)  & cond_nt
        cond_8 = (check_left&check_bottom) & cond_nt
        cond_rect = (~cond_touch)&(~self.cond_1)&(~cond_4)&(~cond_5)&(~cond_6)&(~cond_7)&(~cond_8)
        # This occurs if source region is fully within k or k+1 bins.
        cond_9 = (cond_rect&check_middle)
       # cond_4[cond_rect&check_middle] = True  
        cond_10 = (cond_rect&(~check_middle))
        
        #print('sc_inds=',sc_inds.shape)
        #print('cond_1=',self.cond_1.shape)
        #raise Exception()
        
        k1, i1, j1  = np.nonzero((~self.cond_1)&sc_inds) # Only do loss/gain terms for 0>bins-sbin bins
        k2, i2, j2  = np.nonzero(cond_2&sc_inds)
        k3, i3, j3  = np.nonzero(cond_3&sc_inds)
        k4, i4, j4  = np.nonzero(cond_4&sc_inds)
        k5, i5, j5  = np.nonzero(cond_5&sc_inds)
        k6, i6, j6  = np.nonzero(cond_6&sc_inds)
        k7, i7, j7  = np.nonzero(cond_7&sc_inds)
        k8, i8, j8  = np.nonzero(cond_8&sc_inds)
        k9, i9, j9  = np.nonzero(cond_9&sc_inds)
        k10,i10,j10 = np.nonzero(cond_10&sc_inds)
        
        cond[k1,i1,j1]    = 1
        cond[k2,i2,j2]    = 2
        cond[k3,i3,j3]    = 3
        cond[k4,i4,j4]    = 4
        cond[k5,i5,j5]    = 5
        cond[k6,i6,j6]    = 6
        cond[k7,i7,j7]    = 7
        cond[k8,i8,j8]    = 8
        cond[k9,i9,j9]    = 9 
        cond[k10,i10,j10] = 10
        
        # Calculate transfer rates (rectangular integration, source space)
        # Collection (eqs. 23-25 in Wang et al. 2007)
        # ii collecting jj 
        dMi_loss[k1,i1,j1] = integrate_rect_kernel(x11[k1,i1],x21[k1,i1],x12[k1,j1],x22[k1,j1],1, 0, 0, PK[:,i1,j1],ak1[k1,i1],ck1[k1,i1],ak2[k1,j1],ck2[k1,j1])
        dMj_loss[k1,i1,j1] = integrate_rect_kernel(x11[k1,i1],x21[k1,i1],x12[k1,j1],x22[k1,j1],0, 1, 0, PK[:,i1,j1],ak1[k1,i1],ck1[k1,i1],ak2[k1,j1],ck2[k1,j1])
        dNi_loss[k1,i1,j1] = integrate_rect_kernel(x11[k1,i1],x21[k1,i1],x12[k1,j1],x22[k1,j1],0, 0, 0, PK[:,i1,j1],ak1[k1,i1],ck1[k1,i1],ak2[k1,j1],ck2[k1,j1])
        dNj_loss = dNi_loss.copy() # Nj loss should be same as Ni loss
        
        # Condition 4: Self collection. All Mass/Number goes into ii+sbin = jj+sbin kbin
        xi1 = x11[k4,i4].copy()
        xi2 = x21[k4,i4].copy() 
        xj1 = x12[k4,j4].copy()
        xj2 = x22[k4,j4].copy()
        
        dM_gain[k4,i4,j4,0]  = integrate_rect_kernel(xi1,xi2,xj1,xj2,0, 0, 1, PK[:,i4,j4],ak1[k4,i4],ck1[k4,i4],ak2[k4,j4],ck2[k4,j4]) 
        dN_gain[k4,i4,j4,0]  = integrate_rect_kernel(xi1,xi2,xj1,xj2,0, 0, 0, PK[:,i4,j4],ak1[k4,i4],ck1[k4,i4],ak2[k4,j4],ck2[k4,j4]) 

        # Condition 2:
        # k bin: Lower triangle region. Just clips BR corner.
        #                       Triangle = ((xi1,xj1),(xi1,y_left_edge),(xi2,xj1))
        xt1 = x11[k2,i2].copy()
        yt1 = x12[k2,j2].copy()
        xt2 = x11[k2,i2].copy()
        yt2 = y_left_edge[k2,i2,j2].copy()
        xt3 = x21[k2,i2].copy()
        yt3 = x12[k2,j2].copy()
        
        dM_gain[k2,i2,j2,0] = integrate_tri_kernel(0, 0, 1, PK[:,i2,j2], ak1[k2,i2], ck1[k2,i2], ak2[k2,j2], ck2[k2,j2], xt1, yt1, xt2, yt2, xt3, yt3)
        dM_gain[k2,i2,j2,1] = (dMi_loss[k2,i2,j2]+dMj_loss[k2,i2,j2])-dM_gain[k2,i2,j2,0]
        
        dN_gain[k2,i2,j2,0] = integrate_tri_kernel(0, 0, 0, PK[:,i2,j2], ak1[k2,i2], ck1[k2,i2], ak2[k2,j2], ck2[k2,j2], xt1, yt1, xt2, yt2, xt3, yt3)
        dN_gain[k2,i2,j2,1] = (dNi_loss[k2,i2,j2])-dN_gain[k2,i2,j2,0]
            
        # Condition 3:
        #    k bin: Lower triangle region. Just clips UL corner.
        #                      Triangle = ((xi1,xj1),(xi1,xj2),(x_bottom_edge,xj1))  
        xt1 = x11[k3,i3].copy()
        yt1 = x12[k3,j3].copy()
        xt2 = x11[k3,i3].copy()
        yt2 = x22[k3,j3].copy()
        xt3 = x_bottom_edge[k3,i3,j3].copy()
        yt3 = x12[k3,j3].copy()
        
        dM_gain[k3,i3,j3,0] = integrate_tri_kernel(0, 0, 1, PK[:,i3,j3], ak1[k3,i3], ck1[k3,i3], ak2[k3,j3], ck2[k3,j3], xt1, yt1, xt2, yt2, xt3, yt3)
        dM_gain[k3,i3,j3,1] = (dMi_loss[k3,i3,j3]+dMj_loss[k3,i3,j3])-dM_gain[k3,i3,j3,0]
            
        dN_gain[k3,i3,j3,0] = integrate_tri_kernel(0, 0, 0, PK[:,i3,j3], ak1[k3,i3], ck1[k3,i3], ak2[k3,j3], ck2[k3,j3], xt1, yt1, xt2, yt2, xt3, yt3)
        dN_gain[k3,i3,j3,1] = (dNi_loss[k3,i3,j3])-dN_gain[k3,i3,j3,0]
            
        # Condition 5: 
            
        #    k bin: Top/Bottom clip: Rectangle on left, triangle on right
        #                              Rectangle = ((xi1,xj1),(xi1,xj2),(x_top_edge,xj2),(x_top_edge,xj1))
        #                              Triangle  = ((x_top_edge,xj1),(x_top_edge,xj2),(x_bottom_edge,xj1))
       
        xr1 = x11[k5,i5].copy()
        yr1 = x12[k5,j5].copy()
        xr2 = x_top_edge[k5,i5,j5].copy()
        yr2 = x22[k5,j5].copy()
       
        xt1 = x_top_edge[k5,i5,j5].copy()
        yt1 = x12[k5,j5].copy()
        xt2 = x_top_edge[k5,i5,j5].copy()
        yt2 = x22[k5,j5].copy()
        xt3 = x_bottom_edge[k5,i5,j5].copy()
        yt3 = x12[k5,j5].copy()
        
        dM_gain[k5,i5,j5,0] = integrate_rect_kernel(xr1,xr2,yr1,yr2,0, 0, 1, PK[:,i5,j5],ak1[k5,i5],ck1[k5,i5],ak2[k5,j5],ck2[k5,j5])+\
                           integrate_tri_kernel(0, 0, 1, PK[:,i5,j5], ak1[k5,i5], ck1[k5,i5], ak2[k5,j5], ck2[k5,j5], xt1, yt1, xt2, yt2, xt3, yt3)
        
        dM_gain[k5,i5,j5,1] = (dMi_loss[k5,i5,j5]+dMj_loss[k5,i5,j5])-dM_gain[k5,i5,j5,0]
            
        dN_gain[k5,i5,j5,0] = integrate_rect_kernel(xr1,xr2,yr1,yr2,0, 0, 0, PK[:,i5,j5],ak1[k5,i5],ck1[k5,i5],ak2[k5,j5],ck2[k5,j5])+\
                           integrate_tri_kernel(0, 0, 0, PK[:,i5,j5], ak1[k5,i5], ck1[k5,i5], ak2[k5,j5], ck2[k5,j5], xt1, yt1, xt2, yt2, xt3, yt3)
                           
        dN_gain[k5,i5,j5,1] = (dNi_loss[k5,i5,j5])-dN_gain[k5,i5,j5,0]
        
        # Condition 6:
        # k bin: Left/Right clip: Rectangle on bottom, triangle on top
        #                          Rectangle = ((xi1,xj1),(xi1,y_right_edge),(xi2,y_right_edge),(xi2,xj1))
        #                          Triangle  = ((xi1,y_right_edge),(xi1,y_left_edge),(xi2,y_right_edge))
            
        xr1 = x11[k6,i6].copy()
        yr1 = x12[k6,j6].copy()
        xr2 = x21[k6,i6].copy()
        yr2 = y_right_edge[k6,i6,j6].copy()
        
        xt1 = x11[k6,i6].copy()
        yt1 = y_right_edge[k6,i6,j6].copy()
        xt2 = x11[k6,i6].copy()
        yt2 = y_left_edge[k6,i6,j6].copy()
        xt3 = x21[k6,i6].copy()
        yt3 = y_right_edge[k6,i6,j6].copy()
        
        dM_gain[k6,i6,j6,0] = integrate_rect_kernel(xr1,xr2,yr1,yr2,0, 0, 1, PK[:,i6,j6],ak1[k6,i6],ck1[k6,i6],ak2[k6,j6],ck2[k6,j6])+\
                           integrate_tri_kernel(0, 0, 1, PK[:,i6,j6], ak1[k6,i6], ck1[k6,i6], ak2[k6,j6], ck2[k6,j6], xt1, yt1, xt2, yt2, xt3, yt3)
        
        dM_gain[k6,i6,j6,1] = (dMi_loss[k6,i6,j6]+dMj_loss[k6,i6,j6])-dM_gain[k6,i6,j6,0]
            
        dN_gain[k6,i6,j6,0] = integrate_rect_kernel(xr1,xr2,yr1,yr2,0, 0, 0, PK[:,i6,j6],ak1[k6,i6],ck1[k6,i6],ak2[k6,j6],ck2[k6,j6])+\
                           integrate_tri_kernel(0, 0, 0, PK[:,i6,j6], ak1[k6,i6], ck1[k6,i6], ak2[k6,j6], ck2[k6,j6], xt1, yt1, xt2, yt2, xt3, yt3)
                           
        dN_gain[k6,i6,j6,1] = (dNi_loss[k6,i6,j6])-dN_gain[k6,i6,j6,0]
            
        # Condition 7:
        # k+1 bin: Triangle in top right corner
        #                            Triangle = ((x_top_edge,xj2),(xi2,xj2),(xi2,y_right_edge))
            
        xt1 = x_top_edge[k7,i7,j7].copy()
        yt1 = x22[k7,j7].copy()
        xt2 = x21[k7,i7].copy()
        yt2 = x22[k7,j7].copy()
        xt3 = x21[k7,i7].copy()
        yt3 = y_right_edge[k7,i7,j7].copy()
        
        dM_gain[k7,i7,j7,1] = integrate_tri_kernel(0, 0, 1, PK[:,i7,j7], ak1[k7,i7], ck1[k7,i7], ak2[k7,j7], ck2[k7,j7], xt1, yt1, xt2, yt2, xt3, yt3)
        dM_gain[k7,i7,j7,0] = (dMi_loss[k7,i7,j7]+dMj_loss[k7,i7,j7])-dM_gain[k7,i7,j7,1]
        
        dN_gain[k7,i7,j7,1] = integrate_tri_kernel(0, 0, 0, PK[:,i7,j7], ak1[k7,i7], ck1[k7,i7], ak2[k7,j7], ck2[k7,j7], xt1, yt1, xt2, yt2, xt3, yt3)
        dN_gain[k7,i7,j7,0] = (dNi_loss[k7,i7,j7])-dN_gain[k7,i7,j7,1]
            
        # Condition 8:
        #  k bin: Triangle in lower left corner
        #                                Triangle = ((xi1,xj1),(xi1,y_left_edge),(x_bottom_edge,xj1))
        xt1 = x11[k8,i8].copy()
        yt1 = x12[k8,j8].copy()
        xt2 = x11[k8,i8].copy()
        yt2 = y_left_edge[k8,i8,j8].copy()
        xt3 = x_bottom_edge[k8,i8,j8].copy()
        yt3 = x12[k8,j8].copy()
        
        dM_gain[k8,i8,j8,0] = integrate_tri_kernel(0, 0, 1, PK[:,i8,j8], ak1[k8,i8], ck1[k8,i8], ak2[k8,j8], ck2[k8,j8], xt1, yt1, xt2, yt2, xt3, yt3)
        dM_gain[k8,i8,j8,1] = (dMi_loss[k8,i8,j8]+dMj_loss[k8,i8,j8])-dM_gain[k8,i8,j8,0]
        
        dN_gain[k8,i8,j8,0] = integrate_tri_kernel(0, 0, 0, PK[:,i8,j8], ak1[k8,i8], ck1[k8,i8], ak2[k8,j8], ck2[k8,j8], xt1, yt1, xt2, yt2, xt3, yt3)
        dN_gain[k8,i8,j8,1] = (dNi_loss[k8,i8,j8])-dN_gain[k8,i8,j8,0]
            
        # Condition 9: Rectangle collection within k bin. All Mass/Number goes into kbin
        xi1 = x11[k9,i9].copy()
        xi2 = x21[k9,i9].copy() 
        xj1 = x12[k9,j9].copy()
        xj2 = x22[k9,j9].copy()
        
        dM_gain[k9,i9,j9,0]  = integrate_rect_kernel(xi1,xi2,xj1,xj2,0, 0, 1, PK[:,i9,j9],ak1[k9,i9],ck1[k9,i9],ak2[k9,j9],ck2[k9,j9]) 
        dN_gain[k9,i9,j9,0]  = integrate_rect_kernel(xi1,xi2,xj1,xj2,0, 0, 0, PK[:,i9,j9],ak1[k9,i9],ck1[k9,i9],ak2[k9,j9],ck2[k9,j9]) 

        # Condition 10: Rectangle collection within k+1 bin. All Mass/Number goes into kbin
        xi1 = x11[k10,i10].copy()
        xi2 = x21[k10,i10].copy() 
        xj1 = x12[k10,j10].copy()
        xj2 = x22[k10,j10].copy()
        
        dM_gain[k10,i10,j10,1]  = integrate_rect_kernel(xi1,xi2,xj1,xj2,0, 0, 1, PK[:,i10,j10],ak1[k10,i10],ck1[k10,i10],ak2[k10,j10],ck2[k10,j10]) 
        dN_gain[k10,i10,j10,1]  = integrate_rect_kernel(xi1,xi2,xj1,xj2,0, 0, 0, PK[:,i10,j10],ak1[k10,i10],ck1[k10,i10],ak2[k10,j10],ck2[k10,j10]) 
       
        M1_loss = np.nansum(dMi_loss,axis=2) 
        N1_loss = np.nansum(dNi_loss,axis=2) 
        
        M2_loss = np.nansum(dMj_loss,axis=1) 
        N2_loss = np.nansum(dNj_loss,axis=1)
        
        
        # ChatGPT is the GOAT for telling me about np.add.at!
        M_gain = np.zeros((self.Hlen,self.bins))
        np.add.at(M_gain, (np.arange(self.Hlen)[:,None,None],self.kmin), dM_gain[:,:,:,0])
        np.add.at(M_gain,  (np.arange(self.Hlen)[:,None,None],self.kmid), dM_gain[:,:,:,1])
        
        N_gain = np.zeros((self.Hlen,self.bins))
        np.add.at(N_gain,  (np.arange(self.Hlen)[:,None,None],self.kmin), dN_gain[:,:,:,0])
        np.add.at(N_gain,  (np.arange(self.Hlen)[:,None,None],self.kmid), dN_gain[:,:,:,1])
          
        # ELD NOTE: Breakup here can take losses from each pair and calculate gains
        # for breakup. Breakup gain arrays will be 3D.
        if self.Ebr>0.:
            
            Mij_loss = dMi_loss[k1,i1,j1]+dMj_loss[k1,i1,j1]
            
            Mb_gain = np.nansum(self.dMb_gain_frac[:,self.kmin[i1,j1]][None,:,:]*Mij_loss,axis=2) 
            Nb_gain = np.nansum(self.dNb_gain_frac[:,self.kmin[i1,j1]][None,:,:]*Mij_loss,axis=2) 
                
        return M1_loss, M2_loss, M_gain, Mb_gain, N1_loss, N2_loss, N_gain, Nb_gain
        
    # Advance PSD Mbins and Nbins by one time/height step
    def interact(self,dt):

        # Ndists x height x bins
        Mbins_old = self.Mbins.copy() 
        Nbins_old = self.Nbins.copy()
        
        Mbins = np.zeros_like(Mbins_old)
        Nbins = np.zeros_like(Nbins_old)

        M_loss = np.zeros_like(Mbins)
        N_loss = np.zeros_like(Nbins) 
        
        M_gain = np.zeros_like(Mbins)
        N_gain = np.zeros_like(Nbins)
        
        indc = self.indc
        indb = self.indb

        for d1 in range(self.dnum):
            for d2 in range(d1,self.dnum):
                if d1==d2:
                    self_col = True 
                else:
                    self_col = False
                
                # Collision-Coalescence / Breakup
                M1_loss_temp, M2_loss_temp, M_gain_temp, Mb_gain_temp,\
                N1_loss_temp, N2_loss_temp, N_gain_temp, Nb_gain_temp = self.calculate(d1,d2,np.squeeze(self.PK[:,d1,d2,:,:]),self_col)
        
                M_loss[d1,:,:]    += M1_loss_temp 
                M_loss[d2,:,:]    += M2_loss_temp
                
                M_gain[indc,:,:]  += self.Eagg*M_gain_temp
                M_gain[indb,:,:]  += self.Ebr*Mb_gain_temp
                
                N_loss[d1,:,:]    += N1_loss_temp
                N_loss[d2,:,:]    += N2_loss_temp
                 
                N_gain[indc,:,:]  += self.Eagg*N_gain_temp
                N_gain[indb,:,:]  += self.Ebr*Nb_gain_temp
                
        M_loss *= self.Ecb
        N_loss *= self.Ecb 
        
        M_net = dt*(M_gain-M_loss) 
        N_net = dt*(N_gain-N_loss)

        return M_net, N_net

        # # Update bin masses, numbers and subgrid linear distribution parameters
        # for d1 in range(self.dnum):    
        #     self.dists[d1].Mbins = Mbins[d1]
        #     self.dists[d1].Nbins = Nbins[d1]
        #     self.dists[d1].diagnose()    

 
        