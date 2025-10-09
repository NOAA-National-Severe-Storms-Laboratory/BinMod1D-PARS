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
    
    def __init__(self,dists,cc_dest,br_dest,Eagg,Ecb,Ebr,int_type='t',frag_dict=None,kernel='Golovin'):
        
        # cc_dest is an integer (from 1 to len(dists)) that determines the destination 
        # for coalesced particles
        
        # br_dest is an integer  (from 1 to len(dists)) that determines the destination
        # for fragments
        
        self.dists = dists
        self.int_type = int_type
        self.frag_dict = frag_dict
        self.kernel = kernel
        self.indc = cc_dest -1
        self.indb = br_dest-1
        self.Eagg = Eagg 
        self.Ebr = Ebr 
        self.Ecb = Ecb
        
        # Ensure that cc_dest and br_dest are valid     
        if (cc_dest<1) | (cc_dest>len(dists)):
            print('cc_dest needs to be between 1 and {}'.format(len(dists)))
            raise Exception()
        if (br_dest<1) | (br_dest>len(dists)):
            print('br_dest needs to be between 1 and {}'.format(len(dists)))
            raise Exception()
        
        # NOTE: Assume that ALL distributions have same bin grids
        self.bins = dists[0].bins
        self.sbin = dists[0].sbin
        self.xi1 = dists[0].xi1 
        self.xi2 = dists[0].xi2
        xedges = dists[0].xedges
        
        self.dnum = len(dists)
        
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
        
        if self.Ebr>0.:      
            self.setup_fragments()
            
        self.PK = self.create_kernels(dists)
        
        self.update(dists)
        
    def update(self,dists):
        self.Mbins = np.zeros((self.dnum,self.bins))
        self.Nbins = np.zeros((self.dnum,self.bins))
        
        for d1 in range(self.dnum):
            self.Mbins[d1,:] = dists[d1].Mbins.copy() 
            self.Nbins[d1,:] = dists[d1].Nbins.copy()
           
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
               
                dist1 = dists[d1] 
                dist2 = dists[d2]
   
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
                        self.dMb_gain_frac[kk,xx] = self.dists[self.indb].am*IF_func(self.dists[self.indb].bm,self.dists[self.indb].d1[kk],self.dists[self.indb].d2[kk])
                        self.dNb_gain_frac[kk,xx] = IF_func(0.,self.dists[self.indb].d1[kk],self.dists[self.indb].d2[kk])
                  
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

      
    def interact(self,dist1,dist2,PK,self_col=False):
        
        '''
         This function calculate mass and number transfer rates
         for collision-coalescence and collisional breakup between
         each distribution.
        '''
        
        # (dlen x dlen x bins)
        ak1 = dist1.aki 
        ck1 = dist1.cki 
        
        ak2 = dist2.aki 
        ck2 = dist2.cki
              
        '''
        Check edges and find integration regions in source space
        ''' 
        x_bottom_edge = (dist1.xi2[self.kmin]-dist2.x1[None,:])
        x_top_edge = (dist1.xi2[self.kmin]-dist2.x2[None,:])
        y_left_edge = (dist1.xi2[self.kmin]-dist1.x1[:,None])
        y_right_edge = (dist1.xi2[self.kmin]-dist1.x2[:,None])
        
        check_bottom = (dist1.x1[:,None]<x_bottom_edge) &\
                       (dist1.x2[:,None]>x_bottom_edge)
         
        check_top = (dist1.x1[:,None]<x_top_edge) &\
                    (dist1.x2[:,None]>x_top_edge)
                    
        check_left = (dist2.x1[None,:]<y_left_edge) &\
                    (dist2.x2[None,:]>y_left_edge)
                    
        check_right = (dist2.x1[None,:]<y_right_edge) &\
                      (dist2.x2[None,:]>y_right_edge)            
               
        check_middle = ((0.5*(dist1.x1[:,None]+dist1.x2[:,None]))+(0.5*(dist2.x1[None,:]+dist2.x2[None,:])))<dist2.xi2[self.kmin]
                   
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
        dMi_loss = np.zeros((dist1.bins,dist2.bins))
        dMj_loss = np.zeros((dist1.bins,dist2.bins))
        dNi_loss = np.zeros((dist1.bins,dist2.bins))
        dM_gain = np.zeros((dist1.bins,dist2.bins,2))
        dN_gain = np.zeros((dist1.bins,dist2.bins,2))
        Mb_gain = np.zeros((self.bins))
        Nb_gain = np.zeros((self.bins))
        
        cond = np.zeros((self.bins,self.bins),dtype=int)
        
        if self_col:
            sc_inds = np.triu(np.ones_like(cond),k=0)
            
        else:
            sc_inds = np.ones_like(cond)
        
        cond_touch = (check_bottom|check_top|check_left|check_right)
        cond_2_corner = (dist1.x2==dist1.xi2)&(dist2.x1==dist2.xi1)&(check_left)
        cond_3_corner = (dist1.x1==dist1.xi1)&(dist2.x1==dist2.xi1)&(check_bottom)
        cond_1 = (self.ind_i>=(dist1.bins-dist1.sbin)) | (self.ind_j>=(dist2.bins-dist2.sbin))
        cond_2 = np.eye(cond.shape[0],k=1,dtype=bool) & (cond_2_corner) & (~cond_1) & (cond_touch)
        cond_3 = np.eye(cond.shape[0],k=-1,dtype=bool) & (cond_3_corner) & (~cond_1) & (cond_touch)
        cond_4 = np.eye(cond.shape[0],dtype=bool) & (~cond_1)
        cond_nt = (~(cond_1|cond_2|cond_3|cond_4))
        cond_5 = (check_top&check_bottom)  & cond_nt
        cond_6 = (check_left&check_right)  & cond_nt
        cond_7 =  (check_right&check_top)  & cond_nt
        cond_8 = (check_left&check_bottom) & cond_nt
        cond_rect = (~cond_touch)&(~cond_1)&(~cond_4)&(~cond_5)&(~cond_6)&(~cond_7)&(~cond_8)
        # This occurs if source region is fully within k or k+1 bins.
        cond_9 = (cond_rect&check_middle)
       # cond_4[cond_rect&check_middle] = True  
        cond_10 = (cond_rect&(~check_middle))
        
        i1, j1 = np.nonzero((~cond_1)&sc_inds) # Only do loss/gain terms for 0>bins-sbin bins
        i2, j2 = np.nonzero(cond_2&sc_inds)
        i3, j3 = np.nonzero(cond_3&sc_inds)
        i4, j4 = np.nonzero(cond_4&sc_inds)
        i5, j5 = np.nonzero(cond_5&sc_inds)
        i6, j6 = np.nonzero(cond_6&sc_inds)
        i7, j7 = np.nonzero(cond_7&sc_inds)
        i8, j8 = np.nonzero(cond_8&sc_inds)
        i9, j9 = np.nonzero(cond_9&sc_inds)
        i10,j10 = np.nonzero(cond_10&sc_inds)
        
        cond[i1,j1] = 1
        cond[i2,j2] = 2
        cond[i3,j3] = 3
        cond[i4,j4] = 4
        cond[i5,j5] = 5
        cond[i6,j6] = 6
        cond[i7,j7] = 7
        cond[i8,j8] = 8
        cond[i9,j9] = 9 
        cond[i10,j10] = 10
        
        # Calculate transfer rates (rectangular integration, source space)
        # Collection (eqs. 23-25 in Wang et al. 2007)
        # ii collecting jj 
        dMi_loss[i1,j1] = integrate_rect_kernel(dist1.x1[i1],dist1.x2[i1],dist2.x1[j1],dist2.x2[j1],1, 0, 0, PK[:,i1,j1],ak1[i1],ck1[i1],ak2[j1],ck2[j1])
        dMj_loss[i1,j1] = integrate_rect_kernel(dist1.x1[i1],dist1.x2[i1],dist2.x1[j1],dist2.x2[j1],0, 1, 0, PK[:,i1,j1],ak1[i1],ck1[i1],ak2[j1],ck2[j1])
        dNi_loss[i1,j1] = integrate_rect_kernel(dist1.x1[i1],dist1.x2[i1],dist2.x1[j1],dist2.x2[j1],0, 0, 0, PK[:,i1,j1],ak1[i1],ck1[i1],ak2[j1],ck2[j1])
        dNj_loss = dNi_loss.copy() # Nj loss should be same as Ni loss
        
        # Condition 4: Self collection. All Mass/Number goes into ii+sbin = jj+sbin kbin
        xi1 = dist1.x1[i4].copy()
        xi2 = dist1.x2[i4].copy() 
        xj1 = dist2.x1[j4].copy()
        xj2 = dist2.x2[j4].copy()
        
        dM_gain[i4,j4,0]  = integrate_rect_kernel(xi1,xi2,xj1,xj2,0, 0, 1, PK[:,i4,j4],ak1[i4],ck1[i4],ak2[j4],ck2[j4]) 
        dN_gain[i4,j4,0]  = integrate_rect_kernel(xi1,xi2,xj1,xj2,0, 0, 0, PK[:,i4,j4],ak1[i4],ck1[i4],ak2[j4],ck2[j4]) 

        # Condition 2:
        # k bin: Lower triangle region. Just clips BR corner.
        #                       Triangle = ((xi1,xj1),(xi1,y_left_edge),(xi2,xj1))
        xt1 = dist1.x1[i2].copy()
        yt1 = dist2.x1[j2].copy()
        xt2 = dist1.x1[i2].copy()
        yt2 = y_left_edge[i2,j2].copy()
        xt3 = dist1.x2[i2].copy()
        yt3 = dist2.x1[j2].copy()
        
        dM_gain[i2,j2,0] = integrate_tri_kernel(0, 0, 1, PK[:,i2,j2], ak1[i2], ck1[i2], ak2[j2], ck2[j2], xt1, yt1, xt2, yt2, xt3, yt3)
        dM_gain[i2,j2,1] = (dMi_loss[i2,j2]+dMj_loss[i2,j2])-dM_gain[i2,j2,0]
        
        dN_gain[i2,j2,0] = integrate_tri_kernel(0, 0, 0, PK[:,i2,j2], ak1[i2], ck1[i2], ak2[j2], ck2[j2], xt1, yt1, xt2, yt2, xt3, yt3)
        dN_gain[i2,j2,1] = (dNi_loss[i2,j2])-dN_gain[i2,j2,0]
            
        # Condition 3:
        #    k bin: Lower triangle region. Just clips UL corner.
        #                      Triangle = ((xi1,xj1),(xi1,xj2),(x_bottom_edge,xj1))  
        xt1 = dist1.x1[i3].copy()
        yt1 = dist2.x1[j3].copy()
        xt2 = dist1.x1[i3].copy()
        yt2 = dist2.x2[j3].copy()
        xt3 = x_bottom_edge[i3,j3].copy()
        yt3 = dist2.x1[j3].copy()
        
        dM_gain[i3,j3,0] = integrate_tri_kernel(0, 0, 1, PK[:,i3,j3], ak1[i3], ck1[i3], ak2[j3], ck2[j3], xt1, yt1, xt2, yt2, xt3, yt3)
        dM_gain[i3,j3,1] = (dMi_loss[i3,j3]+dMj_loss[i3,j3])-dM_gain[i3,j3,0]
            
        dN_gain[i3,j3,0] = integrate_tri_kernel(0, 0, 0, PK[:,i3,j3], ak1[i3], ck1[i3], ak2[j3], ck2[j3], xt1, yt1, xt2, yt2, xt3, yt3)
        dN_gain[i3,j3,1] = (dNi_loss[i3,j3])-dN_gain[i3,j3,0]
            
        # Condition 5: 
            
        #    k bin: Top/Bottom clip: Rectangle on left, triangle on right
        #                              Rectangle = ((xi1,xj1),(xi1,xj2),(x_top_edge,xj2),(x_top_edge,xj1))
        #                              Triangle  = ((x_top_edge,xj1),(x_top_edge,xj2),(x_bottom_edge,xj1))
       
        xr1 = dist1.x1[i5].copy()
        yr1 = dist2.x1[j5].copy()
        xr2 = x_top_edge[i5,j5].copy()
        yr2 = dist2.x2[j5].copy()
       
        xt1 = x_top_edge[i5,j5].copy()
        yt1 = dist2.x1[j5].copy()
        xt2 = x_top_edge[i5,j5].copy()
        yt2 = dist2.x2[j5].copy()
        xt3 = x_bottom_edge[i5,j5].copy()
        yt3 = dist2.x1[j5].copy()
        
        dM_gain[i5,j5,0] = integrate_rect_kernel(xr1,xr2,yr1,yr2,0, 0, 1, PK[:,i5,j5],ak1[i5],ck1[i5],ak2[j5],ck2[j5])+\
                           integrate_tri_kernel(0, 0, 1, PK[:,i5,j5], ak1[i5], ck1[i5], ak2[j5], ck2[j5], xt1, yt1, xt2, yt2, xt3, yt3)
        
        dM_gain[i5,j5,1] = (dMi_loss[i5,j5]+dMj_loss[i5,j5])-dM_gain[i5,j5,0]
            
        dN_gain[i5,j5,0] = integrate_rect_kernel(xr1,xr2,yr1,yr2,0, 0, 0, PK[:,i5,j5],ak1[i5],ck1[i5],ak2[j5],ck2[j5])+\
                           integrate_tri_kernel(0, 0, 0, PK[:,i5,j5], ak1[i5], ck1[i5], ak2[j5], ck2[j5], xt1, yt1, xt2, yt2, xt3, yt3)
                           
        dN_gain[i5,j5,1] = (dNi_loss[i5,j5])-dN_gain[i5,j5,0]
        
        # Condition 6:
        # k bin: Left/Right clip: Rectangle on bottom, triangle on top
        #                          Rectangle = ((xi1,xj1),(xi1,y_right_edge),(xi2,y_right_edge),(xi2,xj1))
        #                          Triangle  = ((xi1,y_right_edge),(xi1,y_left_edge),(xi2,y_right_edge))
            
        xr1 = dist1.x1[i6].copy()
        yr1 = dist2.x1[j6].copy()
        xr2 = dist1.x2[i6].copy()
        yr2 = y_right_edge[i6,j6].copy()
        
        xt1 = dist1.x1[i6].copy()
        yt1 = y_right_edge[i6,j6].copy()
        xt2 = dist1.x1[i6].copy()
        yt2 = y_left_edge[i6,j6].copy()
        xt3 = dist1.x2[i6].copy()
        yt3 = y_right_edge[i6,j6].copy()
        
        dM_gain[i6,j6,0] = integrate_rect_kernel(xr1,xr2,yr1,yr2,0, 0, 1, PK[:,i6,j6],ak1[i6],ck1[i6],ak2[j6],ck2[j6])+\
                           integrate_tri_kernel(0, 0, 1, PK[:,i6,j6], ak1[i6], ck1[i6], ak2[j6], ck2[j6], xt1, yt1, xt2, yt2, xt3, yt3)
        
        dM_gain[i6,j6,1] = (dMi_loss[i6,j6]+dMj_loss[i6,j6])-dM_gain[i6,j6,0]
            
        dN_gain[i6,j6,0] = integrate_rect_kernel(xr1,xr2,yr1,yr2,0, 0, 0, PK[:,i6,j6],ak1[i6],ck1[i6],ak2[j6],ck2[j6])+\
                           integrate_tri_kernel(0, 0, 0, PK[:,i6,j6], ak1[i6], ck1[i6], ak2[j6], ck2[j6], xt1, yt1, xt2, yt2, xt3, yt3)
                           
        dN_gain[i6,j6,1] = (dNi_loss[i6,j6])-dN_gain[i6,j6,0]
            
        # Condition 7:
        # k+1 bin: Triangle in top right corner
        #                            Triangle = ((x_top_edge,xj2),(xi2,xj2),(xi2,y_right_edge))
            
        xt1 = x_top_edge[i7,j7].copy()
        yt1 = dist2.x2[j7].copy()
        xt2 = dist1.x2[i7].copy()
        yt2 = dist2.x2[j7].copy()
        xt3 = dist1.x2[i7].copy()
        yt3 = y_right_edge[i7,j7].copy()
        
        dM_gain[i7,j7,1] = integrate_tri_kernel(0, 0, 1, PK[:,i7,j7], ak1[i7], ck1[i7], ak2[j7], ck2[j7], xt1, yt1, xt2, yt2, xt3, yt3)
        dM_gain[i7,j7,0] = (dMi_loss[i7,j7]+dMj_loss[i7,j7])-dM_gain[i7,j7,1]
        
        dN_gain[i7,j7,1] = integrate_tri_kernel(0, 0, 0, PK[:,i7,j7], ak1[i7], ck1[i7], ak2[j7], ck2[j7], xt1, yt1, xt2, yt2, xt3, yt3)
        dN_gain[i7,j7,0] = (dNi_loss[i7,j7])-dN_gain[i7,j7,1]
            
        # Condition 8:
        #  k bin: Triangle in lower left corner
        #                                Triangle = ((xi1,xj1),(xi1,y_left_edge),(x_bottom_edge,xj1))
        xt1 = dist1.x1[i8].copy()
        yt1 = dist2.x1[j8].copy()
        xt2 = dist1.x1[i8].copy()
        yt2 = y_left_edge[i8,j8].copy()
        xt3 = x_bottom_edge[i8,j8].copy()
        yt3 = dist2.x1[j8].copy()
        
        dM_gain[i8,j8,0] = integrate_tri_kernel(0, 0, 1, PK[:,i8,j8], ak1[i8], ck1[i8], ak2[j8], ck2[j8], xt1, yt1, xt2, yt2, xt3, yt3)
        dM_gain[i8,j8,1] = (dMi_loss[i8,j8]+dMj_loss[i8,j8])-dM_gain[i8,j8,0]
        
        dN_gain[i8,j8,0] = integrate_tri_kernel(0, 0, 0, PK[:,i8,j8], ak1[i8], ck1[i8], ak2[j8], ck2[j8], xt1, yt1, xt2, yt2, xt3, yt3)
        dN_gain[i8,j8,1] = (dNi_loss[i8,j8])-dN_gain[i8,j8,0]
            
        # Condition 9: Rectangle collection within k bin. All Mass/Number goes into kbin
        xi1 = dist1.x1[i9].copy()
        xi2 = dist1.x2[i9].copy() 
        xj1 = dist2.x1[j9].copy()
        xj2 = dist2.x2[j9].copy()
        
        dM_gain[i9,j9,0]  = integrate_rect_kernel(xi1,xi2,xj1,xj2,0, 0, 1, PK[:,i9,j9],ak1[i9],ck1[i9],ak2[j9],ck2[j9]) 
        dN_gain[i9,j9,0]  = integrate_rect_kernel(xi1,xi2,xj1,xj2,0, 0, 0, PK[:,i9,j9],ak1[i9],ck1[i9],ak2[j9],ck2[j9]) 

        # Condition 10: Rectangle collection within k+1 bin. All Mass/Number goes into kbin
        xi1 = dist1.x1[i10].copy()
        xi2 = dist1.x2[i10].copy() 
        xj1 = dist2.x1[j10].copy()
        xj2 = dist2.x2[j10].copy()
        
        dM_gain[i10,j10,1]  = integrate_rect_kernel(xi1,xi2,xj1,xj2,0, 0, 1, PK[:,i10,j10],ak1[i10],ck1[i10],ak2[j10],ck2[j10]) 
        dN_gain[i10,j10,1]  = integrate_rect_kernel(xi1,xi2,xj1,xj2,0, 0, 0, PK[:,i10,j10],ak1[i10],ck1[i10],ak2[j10],ck2[j10]) 
       
        M1_loss = np.nansum(dMi_loss,axis=1) 
        N1_loss = np.nansum(dNi_loss,axis=1) 
        
        M2_loss = np.nansum(dMj_loss,axis=0) 
        N2_loss = np.nansum(dNj_loss,axis=0)

        # ChatGPT is the GOAT for telling me about np.add.at!
        M_gain = np.zeros((dist2.bins,))
        np.add.at(M_gain, self.kmin, dM_gain[:, :, 0])
        np.add.at(M_gain, self.kmid, dM_gain[:, :, 1])
        
        N_gain = np.zeros((dist2.bins,))
        np.add.at(N_gain, self.kmin, dN_gain[:, :, 0])
        np.add.at(N_gain, self.kmid, dN_gain[:, :, 1])
          
        # ELD NOTE: Breakup here can take losses from each pair and calculate gains
        # for breakup. Breakup gain arrays will be 3D.
        if self.Ebr>0.:
            
            Mij_loss = dMi_loss[i1,j1]+dMj_loss[i1,j1]
            
            Mb_gain = np.nansum(self.dMb_gain_frac[:,self.kmin[i1,j1]]*Mij_loss,axis=1) 
            Nb_gain = np.nansum(self.dNb_gain_frac[:,self.kmin[i1,j1]]*Mij_loss,axis=1) 
            
        return M1_loss, M2_loss, M_gain, Mb_gain, N1_loss, N2_loss, N_gain, Nb_gain
        
    # Advance PSDs by one time/height step
    def advance(self):

        # Ndists x bins
        Mbins_old = np.vstack([self.dists[ff].Mbins for ff in range(self.dnum)]) 
        Nbins_old = np.vstack([self.dists[ff].Nbins for ff in range(self.dnum)]) 
        
        Mbins = np.zeros_like(Mbins_old)
        Nbins = np.zeros_like(Nbins_old)
        
        dh = np.vstack([self.dists[ff].dh for ff in range(self.dnum)])

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
                N1_loss_temp, N2_loss_temp, N_gain_temp, Nb_gain_temp = self.interact(self.dists[d1],self.dists[d2],np.squeeze(self.PK[:,d1,d2,:,:]),self_col)
        
                M_loss[d1,:]    += M1_loss_temp 
                M_loss[d2,:]    += M2_loss_temp
                
                M_gain[indc,:]  += self.Eagg*M_gain_temp
                M_gain[indb,:]  += self.Ebr*Mb_gain_temp
                
                N_loss[d1,:]    += N1_loss_temp
                N_loss[d2,:]    += N2_loss_temp
                 
                N_gain[indc,:]  += self.Eagg*N_gain_temp
                N_gain[indb,:]  += self.Ebr*Nb_gain_temp
                
        M_loss *= self.Ecb*dh
        N_loss *= self.Ecb*dh 
        
        M_gain *= dh 
        N_gain *= dh
        
        M_transfer = M_gain+Mbins_old-M_loss
        N_transfer = N_gain+Nbins_old-N_loss        

        M_new = np.maximum(M_transfer,0.) # Should be positive if not over fragmented.
        Mbins[M_new>=0.] = M_new[M_new>=0.].copy()
        
        N_new = np.maximum(N_transfer,0.) # Should be positive if not over fragmented.
        Nbins[N_new>=0.] = N_new[N_new>=0.].copy()

        # Update bin masses, numbers and subgrid linear distribution parameters
        for d1 in range(self.dnum):    
            self.dists[d1].Mbins = Mbins[d1]
            self.dists[d1].Nbins = Nbins[d1]
            self.dists[d1].diagnose()    

 
        