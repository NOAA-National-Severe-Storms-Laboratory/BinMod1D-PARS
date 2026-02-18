# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 11:31:38 2025

@author: edwin.dunnavan
"""
## Import stuff
import numpy as np

from .collection_kernels import hydro_kernel, long_kernel, hall_kernel, Straub_params
from .bin_integrals import In_int, gam_int, LGN_int, GAU_int
from .bin_integrals import setup_regions
from .bin_integrals import calculate_rates,calculate_regions_batch
from .bin_integrals import vectorized_1mom, vectorized_2mom, combined_coeffs_array
from .distribution import update_1mom, update_2mom

#from joblib import dump

#from tempfile import gettempdir

import numba as nb

from scipy.special import erfinv

#import os

#from multiprocessing import shared_memory

class Interaction():
    
    '''
    Interaction class initializes Interaction objects that contain
    arrays in the form (Ndists x M x N) which specifies the interactions among
    all Ndists distributions.
    '''
    
    def __init__(self,dists,Hlen,cc_dest,br_dest,Eagg,Ecb,Ebr,frag_dict=None,kernel='Golovin',mom_num=2,gpu=False):
        
        # if gpu:
        #     import cupy as np  
        # else:
        #     import numpy as np
        
        # Setup Exact Dunavant Quadrature Nodes (7-point rule, degree 5 exact)
        # Weights
        self.w = np.array([0.225, 0.125939180544827, 0.125939180544827, 0.125939180544827, 
                      0.132394152788506, 0.132394152788506, 0.132394152788506])
        
        # Barycentric coordinates (L1, L2, L3) for the 7 nodes
        self.L = np.array([
            [1/3, 1/3, 1/3],
            [0.797426985353087, 0.101286507323456, 0.101286507323456],
            [0.101286507323456, 0.797426985353087, 0.101286507323456],
            [0.101286507323456, 0.101286507323456, 0.797426985353087],
            [0.059715871789770, 0.470142064105115, 0.470142064105115],
            [0.470142064105115, 0.059715871789770, 0.470142064105115],
            [0.470142064105115, 0.470142064105115, 0.059715871789770]])
        
        self.dists = dists
        
        self.frag_dict = frag_dict
        self.kernel = kernel
        self.indc = cc_dest-1
        self.indb = br_dest-1
        #self.Eagg = Eagg 
        #self.Ebr = Ebr 
        #self.Ecb = Ecb
    
        
        # self.parallel = parallel
        # self.n_jobs = n_jobs
        self.mom_num = mom_num
        
        self.cnz = False
               
        self.dnum = len(dists) 
        self.Hlen = Hlen
        
        # Ensure that cc_dest and br_dest are valid     
        if (cc_dest<1) | (cc_dest>self.dnum):
            print('cc_dest needs to be between 1 and {}'.format(self.dnum))
            raise Exception()
        if (br_dest<1) | (br_dest>self.dnum):
            print('br_dest needs to be between 1 and {}'.format(self.dnum))
            raise Exception()
        
        # NOTE: Assume that ALL distributions have same bin grids
        self.bins = dists[0].bins
        self.sbin = dists[0].sbin
        
        self.dxbins = dists[0].dxbins
        self.dxi = dists[0].dxi
        self.xi1 = dists[0].xi1 
        self.xi2 = dists[0].xi2
        xedges   = dists[0].xedges
          
        # NEW
        self.dxbins3D = np.tile(self.dxbins,(self.dnum,self.Hlen,1))
        self.xi1_3D = np.tile(self.xi1,(self.dnum,self.Hlen,1))
        self.xi2_3D = np.tile(self.xi2,(self.dnum,self.Hlen,1))

        self.vt = np.stack([dist.vt for dist in dists],axis=0)
        
        # Set up dynamic arrays
        # (dist,height,bins)
        self.Mbins = np.zeros((self.dnum,self.Hlen,self.bins))
        self.Mfbins = np.zeros((self.dnum,self.Hlen,self.bins))
        self.cki = np.zeros((self.dnum,self.Hlen,self.bins))
        
        for dd in range(len(dists)):
            self.Mbins[dd,0,:] = dists[dd].Mbins.copy()
            self.Mfbins[dd,0,:] = dists[dd].Mfbins.copy()
            self.cki[dd,0,:] = dists[dd].cki.copy()
        
        if self.mom_num==2:
            
            self.Nbins = np.zeros_like(self.Mbins)
            self.Nfbins = np.zeros_like(self.Mbins)
            self.aki = np.zeros_like(self.Mbins)
            self.x1 = np.zeros_like(self.Mbins)
            self.x2 = np.zeros_like(self.Mbins)
    
            for dd in range(len(dists)):
                self.Nbins[dd,0,:] = dists[dd].Nbins.copy()
                self.Nfbins[dd,0,:] = dists[dd].Nfbins.copy()
                self.aki[dd,0,:] = dists[dd].aki.copy()
                self.x1[dd,0,:] = dists[dd].x1.copy()
                self.x2[dd,0,:] = dists[dd].x2.copy()
        
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
                        
        self.xk_min = self.xi2[self.kmin]


        i_grid, j_grid = np.indices((self.bins,self.bins))
        
        self.k_lim = np.maximum(i_grid,j_grid)
        
        # parallelized computations for box and steady-state setups.
        #if self.parallel:
        # Batches for parallel processing if using Full 1D column model
        #self.batches = np.array_split(np.arange(self.Hlen),self.n_jobs) 
        if Ebr>0.:
            self.breakup = True 
        else:
            self.breakup = False

        self.pnum = int((self.dnum*(self.dnum+1))/2)
        
        
        #self.Eagg = Eagg 
        #self.Ebr  = Ebr
        #self.Ecb  = Eagg + Ebr
        
        self.Eagg = Eagg*np.ones((self.pnum,Hlen,self.bins,self.bins),dtype=np.float64)
        self.Ebr  = Ebr*np.ones((self.pnum,Hlen,self.bins,self.bins),dtype=np.float64)

        # WORKING
        if self.mom_num == 2:
            self.cond_1 = np.tile(((self.ind_i>=(self.bins-self.sbin)) | (self.ind_j>=(self.bins-self.sbin))),(self.Hlen,1,1))
            # WORKING
            self.self_col = np.ones((self.Hlen,self.pnum,self.bins,self.bins),dtype=np.uint8)
            
            # Set up indices of all bin-pair interactions that will occur during coalescence/breakup processes.
            dd = 0
            for d1 in range(self.dnum):
                for d2 in range(d1,self.dnum):
                    if d1==d2:
                        # WORKING
                        self.self_col[:,dd,:,:] = np.tile(np.triu(np.ones((self.bins,self.bins),dtype=np.uint8),k=0),(self.Hlen,1,1))
   
                    dd += 1
            
        elif self.mom_num == 1:
            self.cond_1 = np.tile(((self.ind_i>=(self.bins-self.sbin)) | (self.ind_j>=(self.bins-self.sbin))),(self.Hlen,1,1))
            # WORKING
            self.self_col = np.ones((self.Hlen,self.pnum,self.bins,self.bins),dtype=np.uint8)
            
            dd = 0
            for d1 in range(self.dnum):
                for d2 in range(d1,self.dnum):
                               
                    if d1==d2:
                        self.self_col[:,dd,:,:] = np.tile(np.triu(np.ones((self.bins,self.bins),dtype=np.uint8),k=0),(self.Hlen,1,1))
                        
                    dd += 1

        # static mask for calculations. Bins that are too close to the len(bins) not included
        # and self col lower triangle not included.
        self.static_mask = ((~(self.self_col.astype(bool))) | (self.cond_1[:,None,:,:])).transpose(1,0,2,3)           
        
        # Combine cond_2 with self_col?    
        
        self.dMb_gain_frac = np.zeros((self.bins,self.bins))
        self.dNb_gain_frac = np.zeros((self.bins,self.bins))
        
        if Ebr>0.: # Setup fragment distribution if Ebr>0.     
            self.setup_fragments()
          
        # Only consider breakup if the breakup distribution is complete.
        censor_Ebr = np.broadcast_to((self.dMb_gain_frac.sum(axis=0)<0.5),self.Ebr.shape)    
        self.Ebr[censor_Ebr] = 0.
          

        # NOTE: Before, I was (stupidly?) using the actual limits from the SBE. 
        # HOWEVER, this isn't physically realistic when coalescence and breakup
        # are considered (and calculated) as mutually exclusive scenarios. 
        # This presents a big problem because realistic fragment distributions
        # like the lognormal distribution has a fat tail and thus the presence of breakup
        # can actually generates particles much larger than either of the colliding
        # particles! In order to compare results with Feingold et al. (1988), the
        # kmin limit is used. For realistic kernels, the maximum index of the interacting
        # species is used instead.
        if (self.kernel=='Constant') | (self.kernel=='Golovin') | (self.kernel=='Product'):
            self.dMb_gain_kernel = self.dMb_gain_frac[:,self.kmin]
            self.dNb_gain_kernel = self.dNb_gain_frac[:,self.kmin]
        else:
            self.dMb_gain_kernel = self.dMb_gain_frac[:,self.k_lim]
            self.dNb_gain_kernel = self.dNb_gain_frac[:,self.k_lim]
               
        self.PK = self.create_kernels(dists)
        
        self.kmin_p = self.kmin[None,:,:].repeat(self.pnum,axis=0)
        self.kmid_p = self.kmid[None,:,:].repeat(self.pnum,axis=0)
        
        self.xk_min_p = self.xi2[self.kmin_p]
        
        dd = 0 
        
        self.d1_indices = np.zeros((self.pnum,),dtype=int)
        self.d2_indices = np.zeros((self.pnum,),dtype=int)
        
        for d1 in range(self.dnum):
            for d2 in range(d1,self.dnum):
                
                self.d1_indices[dd] = d1
                self.d2_indices[dd] = d2
        
                dd += 1
            
        if self.mom_num==1:
            self.setup_1mom()
            
     
    def setup_1mom_numba(self):
        
        self.get_dynamic_params_1mom()
        
     
    def setup_1mom(self):
        
        self.dMi_loss = np.zeros((self.pnum,self.bins,self.bins))
        self.dMj_loss = np.zeros((self.pnum,self.bins,self.bins))
        self.dM_loss = np.zeros((self.pnum,self.bins,self.bins))
        self.dM_gain = np.zeros((self.pnum,self.bins,self.bins,2))
        
        self.M_loss_tot_buffer = np.zeros_like(self.Mbins)
        self.M_gain_tot_buffer = np.zeros_like(self.Mbins)
        
        self.regions = np.empty((self.pnum,),dtype=object)
        
        ak1mom = np.zeros((self.Hlen,self.bins))
        ck1mom = np.ones((self.Hlen,self.bins))
        
        dd = 0
    
        # Calculate regions and mass transfer rates
        for d1 in range(self.dnum):
            for d2 in range(d1,self.dnum):
                   
                # Get 3D indices for all bins that will interact at all heights
                # indices have shape (bin-pairs,) where ir are dist1 indices 
                # and jr are dist2 indices. kr are height indices.
                kr, ir, jr = np.nonzero((~self.cond_1)&self.self_col[:,dd,:,:])
                
                # BATCHES
                # (bin-pair,)
                x11  = self.xi1[ir]
                x21  = self.xi2[ir]
                ak1  = ak1mom[kr,ir]
                ck1  = ck1mom[kr,ir]

                # (bin-pair,)
                x12  = self.xi1[jr]
                x22  = self.xi2[jr]
                ak2  = ak1mom[kr,jr]
                ck2  = ck1mom[kr,jr]
                
                kmin = self.kmin[ir,jr]
                kmid = self.kmid[ir,jr]
                
                PK = self.PK[:,dd,ir,jr]
                
                xk_min = self.xi2[self.kmin[ir,jr]]
                
                # NEW METHOD
                region_inds, x_bottom_edge, x_top_edge, y_left_edge, y_right_edge=setup_regions(self.bins,kr,ir,jr,x11,x21,x12,x22,xk_min)

                dMi_loss, dMj_loss, dM_loss, dM_gain, _, _ = calculate_rates(self.Hlen,self.bins,region_inds,x_bottom_edge,x_top_edge,y_left_edge,y_right_edge,
                kr,ir,jr,x11,x21,ak1,ck1,x12,x22,ak2,ck2,PK,kmin,kmid,self.dMb_gain_frac,self.dNb_gain_frac,self.w,self.L,
                breakup=self.breakup)

                self.dMi_loss[dd,ir,jr] = dMi_loss
                self.dMj_loss[dd,ir,jr] = dMj_loss 
                self.dM_loss[dd,ir,jr] = dM_loss
                self.dM_gain[dd,ir,jr,:] = dM_gain
      
                dd += 1
        
        # Total loss for each (i,j) bin-pair
        self.dM_loss = self.dMi_loss+self.dMj_loss
        
   
    def get_dynamic_params_1mom(self):
        """
        Flattens everything into 1D vectors.
        This removes all 3D/4D indexing from the Numba kernel.
        """
        p1, p2 = self.d1_indices, self.d2_indices
        
        # 1. Mass Existence + Static Mask Censors
        # Mbins is (pnum, Hlen, bins)
        M_check = (self.Mbins[p1][:, :, :, None] == 0.) | \
                  (self.Mbins[p2][:, :, None, :] == 0.)
        
        exclude = M_check | self.static_mask
        mask = ~exclude
        
        # 2. Extract Active Indices
        p_act, h_act, i_act, j_act = np.where(mask)
        
        # 3. Calculate 1D Rates (Mass product)
        vals_i = self.cki[p1[p_act], h_act, i_act]
        vals_j = self.cki[p2[p_act], h_act, j_act]
        active_rate = (vals_i * vals_j).astype(np.float64)
        
        active_Eagg = self.Eagg[p_act,h_act,i_act,j_act].astype(np.float64)
        active_Ebr = self.Ebr[p_act,h_act,i_act,j_act].astype(np.float64)
        
        active_dMi_loss = self.dMi_loss[p_act, i_act, j_act].astype(np.float64)
        active_dMj_loss = self.dMj_loss[p_act, i_act, j_act].astype(np.float64)
        active_dM_loss = active_dMi_loss+active_dMj_loss
        
        # 4. Flatten ALL Coefficients into 1D Vectors
        # This replaces dMi[p, i, j] with active_dMi[k]
        self.params = {
            'active_Eagg': active_Eagg,
            'active_Ebr': active_Ebr,
            'active_rate': active_rate,
            'active_h': h_act.astype(np.int32),
            'active_s1': p1[p_act].astype(np.int32),
            'active_s2': p2[p_act].astype(np.int32),
            'active_i': i_act.astype(np.int32),
            'active_j': j_act.astype(np.int32),
            'active_p_idx': p_act.astype(np.int32),
            # Pre-indexed coefficients (1D)
            'active_dMi': active_dMi_loss,
            'active_dMj': active_dMj_loss,
            'active_dMtot': active_dM_loss,
            # Gains (Destinations)
            'active_k0': self.kmin_p[p_act, i_act, j_act].astype(np.int32),
            'active_k1': self.kmid_p[p_act, i_act, j_act].astype(np.int32),
            'active_dMg0': self.dM_gain[p_act, i_act, j_act, 0].astype(np.float64),
            'active_dMg1': self.dM_gain[p_act, i_act, j_act, 1].astype(np.float64),
        }
        
    def get_dynamic_params(self):
        """
        Synchronizes moving grid edges and evolving distributions into a 
        single God Mode parameter dictionary.
        """
        p1, p2 = self.d1_indices, self.d2_indices
        bins = self.bins
         
        M_check = (self.Mbins[p1][:,:,:,None]==0.) | (self.Mbins[p2][:,:,None,:]==0.)
        
        exclude = M_check | self.static_mask
        
        # 2. Pre-allocate 5D Mask: (pnum, hlen, bins, bins)
        mask = np.zeros((self.pnum, self.Hlen, bins, bins), dtype=np.uint8)
        
        target_shape = mask.shape
        
        x11 =  np.broadcast_to(self.x1[p1][:, :, :, None], target_shape)
        x21 =  np.broadcast_to(self.x2[p1][:, :, :, None], target_shape)
        x12 =  np.broadcast_to(self.x1[p2][:, :, None, :],target_shape)
        x22 =  np.broadcast_to(self.x2[p2][:, :, None, :], target_shape)
        xk_min = np.broadcast_to(self.xk_min_p[:, None, :, :],target_shape)
    
        x_top, x_bot = (xk_min - x22), (xk_min - x12)
        y_lef, y_rig = (xk_min - x11), (xk_min - x21)
        
        # 3. Boolean Conditions (Now correctly 5D)
        check_b = (x11 < x_bot) & (x21 > x_bot)
        check_t = (x11 < x_top) & (x21 > x_top)
        check_l = (x12 < y_lef) & (x22 > y_lef)
        check_r = (x12 < y_rig) & (x22 > y_rig)
        cond_touch = (check_b | check_t | check_l | check_r)
        check_m = ((0.5 * (x11 + x21)) + (0.5 * (x12 + x22))) < xk_min
        
        c_BR = (x21 == x_bot) & (x12 == y_rig)
        c_UL = (x11 == x_top) & (x22 == y_lef)
    
        # 4. Assignment Logic
        # Diagonal (Self-Collection) - Broadcast 2D eye to 5D
        diag_2d = np.eye(bins, dtype=bool)
        mask[:, :, diag_2d] = 4
        
        # Expand 2D identity tools to 5D (1, 1, B, B) for clean broadcasting
        diag_5d = diag_2d[None, None, :, :]
        m_nt = ~diag_5d 
    
        # Region 2 & 3: Match the k=1 and k=-1 diagonals
        cond2 = (np.eye(bins, k=1, dtype=bool)[None, None, :, :]) & c_BR & check_l & cond_touch
        mask[cond2] = 2
        
        cond3 = (np.eye(bins, k=-1, dtype=bool)[None, None, :, :]) & c_UL & check_b & cond_touch
        mask[cond3] = 3
        
        # Regions 5-8 (Clipping triangles)
        mask[(mask == 0) & (check_t & check_b) & m_nt] = 5
        mask[(mask == 0) & (check_l & check_r) & m_nt] = 6
        mask[(mask == 0) & (check_r & check_t) & m_nt] = 7
        mask[(mask == 0) & (check_l & check_b) & m_nt] = 8
        
        # Regions 9-10 (Full Rectangles)

        m_rect = (mask == 0) & (~cond_touch) & m_nt
        mask[m_rect & check_m] = 9
        mask[m_rect & (~check_m)] = 10
        
        # Regions 11-12
        mask[(mask == 0) & c_BR & check_t] = 11
        mask[(mask == 0) & c_UL & check_r] = 12
        
        mask[exclude] = 0   
 
        # Get unique indices for each region
        mask_ind = np.reshape(np.arange(len(mask.ravel()),dtype=int),mask.shape)
        
        # Universal indices for all bin-pair interactions that will actually be calculated
        active_inds = mask_ind[mask>0]
        
        # All active regions 
        regions = mask.ravel()[active_inds]
        
        # Get active distribution functions
        PK = np.broadcast_to(self.PK[:,:,None,:,:],(4,self.pnum,self.Hlen,self.bins,self.bins)).reshape(4,-1)[:,active_inds]
        aki1 = np.broadcast_to(self.aki[p1][:, :, :, None],target_shape).ravel()[active_inds]
        cki1 = np.broadcast_to(self.cki[p1][:, :, :, None],target_shape).ravel()[active_inds]
        aki2 = np.broadcast_to(self.aki[p2][:, :, None, :],target_shape).ravel()[active_inds]
        cki2 = np.broadcast_to(self.cki[p2][:, :, None, :],target_shape).ravel()[active_inds]
        
        # (3,3,active)
        C = combined_coeffs_array(PK, aki1, cki1, aki2, cki2)
        
        kmin_4d = np.broadcast_to(self.kmin_p[:, None, :, :], target_shape) 
        kmid_4d = np.broadcast_to(self.kmid_p[:, None, :, :], target_shape)
        
        d1_4d = np.broadcast_to(p1[:, None, None, None],target_shape)
        d2_4d = np.broadcast_to(p2[:, None, None, None],target_shape)
        h_4d = np.broadcast_to(np.arange(self.Hlen)[None, :, None, None],target_shape) 
        bi_4d = np.broadcast_to(np.arange(self.bins)[None, None, :, None],target_shape)
        bj_4d =  np.broadcast_to(np.arange(self.bins)[None, None, None, :],target_shape)

        # 3. Sort all bin-pair interactions by region
        rsort = np.argsort(regions)
        
        sort_inds = active_inds[rsort]
        
        uregs, counts = np.unique(regions,return_counts=True)
        
        reg_map = {r: slice(0, 0) for r in range(2, 13)}
        
        offsets = np.insert(np.cumsum(counts),0,0)
        
        for i, reg_val in enumerate(uregs):
            reg_map[int(reg_val)] =  slice(offsets[i], offsets[i+1])



        self.params = {
            'reg_map':reg_map,
            'regions':regions[rsort],
            'Eagg':self.Eagg.ravel()[sort_inds],
            'Ebr':self.Ebr.ravel()[sort_inds],
            'd1_ind':d1_4d.ravel()[sort_inds],
            'd2_ind':d2_4d.ravel()[sort_inds],
            'hind':h_4d.ravel()[sort_inds],
            'bi_ind':bi_4d.ravel()[sort_inds],
            'bj_ind':bj_4d.ravel()[sort_inds],
            'kmin':kmin_4d.ravel()[sort_inds],
            'kmid':kmid_4d.ravel()[sort_inds],
            'C':C[:,:,rsort],
            'x11': x11.ravel()[sort_inds],
            'x21': x21.ravel()[sort_inds], 
            'x12': x12.ravel()[sort_inds], 
            'x22': x22.ravel()[sort_inds],
            'x_top': x_top.ravel()[sort_inds], 
            'x_bot': x_bot.ravel()[sort_inds], 
            'y_lef': y_lef.ravel()[sort_inds], 
            'y_rig': y_rig.ravel()[sort_inds]}

    def _update_shm_from_dict(self, params):
        """
        Efficiently copies local tensors into Shared Memory.
        """
        # Aki/Cki
        self.shm_aki1[:] = params['aki1']
        self.shm_cki1[:] = params['cki1']
        self.shm_aki2[:] = params['aki2']
        self.shm_cki2[:] = params['cki2']
    
        # Mask
        self.shm_mask[:] = params['mask']
    
        # Consolidated Boundaries
        # Order: x11, x21, x12, x22, x_top, x_bot, y_lef, y_rig
        self.shm_bounds[0] = params['x11']
        self.shm_bounds[1] = params['x21']
        self.shm_bounds[2] = params['x12']
        self.shm_bounds[3] = params['x22']
        self.shm_bounds[4] = params['x_top']
        self.shm_bounds[5] = params['x_bot']
        self.shm_bounds[6] = params['y_lef']
        self.shm_bounds[7] = params['y_rig']
 

    
    
    def update_1mom_subgrid(self):
        
        self.cki = update_1mom(self.Mbins,self.dxi[None,None,:])
        
        self.Mfbins = self.vt[:,None,:]*self.Mbins
    
  
    def update_2mom_subgrid(self):
        
        self.aki, self.cki, self.x1, self.x2 = update_2mom(self.Mbins,self.Nbins,self.rhobins,
                                                             self.bound_low,self.bound_high,
                                                             self.dxbins3D,
                                                             self.xi1_3D,
                                                             self.xi2_3D)
  
        self.Mfbins = self.vt[:,None,:]*self.Mbins 
        self.Nfbins = self.vt[:,None,:]*self.Nbins
    
    def create_kernels(self,dists):
        
        # Kernel ind, x, y
        # NOTE: HK also has denominator bin width terms for x and y
        HK = np.ones((4,self.pnum,self.bins,self.bins))
        
        ## Calculate Bilinear interpolation of collection kernel
        # NOTE: fkernel form can be expressed in terms of K(x,y) = a + b*x +c*y +d*x*y

       
        if (self.kernel=='Hydro') | (self.kernel=='Long') | (self.kernel=='Hall'):
            dd = 0
            for d1 in range(self.dnum):
                for d2 in range(d1,self.dnum):
                    dist1 = dists[d1] 
                    dist2 = dists[d2]
       
                    # Calculate corners of K(x,y), i.e. Hk[0]=K(x1,y1), Hk[1]=K(x2,y1), Hk[2] = K(x1,y2), Hk[3] = K(x2,y2)
                    if self.kernel=='Hydro':
                        HK[0,dd,:,:] = hydro_kernel(dist1.vt1,dist2.vt1,dist1.A1,dist2.A1)
                        HK[1,dd,:,:] = hydro_kernel(dist1.vt2,dist2.vt1,dist1.A2,dist2.A1)
                        HK[2,dd,:,:] = hydro_kernel(dist1.vt1,dist2.vt2,dist1.A1,dist2.A2)
                        HK[3,dd,:,:] = hydro_kernel(dist1.vt2,dist2.vt2,dist1.A2,dist2.A2)
                        
                    elif self.kernel=='Long':
                        HK[0,dd,:,:] = long_kernel(dist1.d1,dist2.d1,dist1.vt1,dist2.vt1,dist1.A1,dist2.A1)
                        HK[1,dd,:,:] = long_kernel(dist1.d2,dist2.d1,dist1.vt2,dist2.vt1,dist1.A2,dist2.A1)
                        HK[2,dd,:,:] = long_kernel(dist1.d1,dist2.d2,dist1.vt1,dist2.vt2,dist1.A1,dist2.A2)
                        HK[3,dd,:,:] = long_kernel(dist1.d2,dist2.d2,dist1.vt2,dist2.vt2,dist1.A2,dist2.A2)
                        
                    elif self.kernel=='Hall':
                        HK[0,dd,:,:] = hall_kernel(dist1.d1,dist2.d1,dist1.vt1,dist2.vt1,dist1.A1,dist2.A1)
                        HK[1,dd,:,:] = hall_kernel(dist1.d2,dist2.d1,dist1.vt2,dist2.vt1,dist1.A2,dist2.A1)
                        HK[2,dd,:,:] = hall_kernel(dist1.d1,dist2.d2,dist1.vt1,dist2.vt2,dist1.A1,dist2.A2)
                        HK[3,dd,:,:] = hall_kernel(dist1.d2,dist2.d2,dist1.vt2,dist2.vt2,dist1.A2,dist2.A2)                        
                        
                    dd += 1
            
                          
            # Rearranged weights for Kernel in form: K(x,y) = a + b*x + c*y + d*x*y
            PK = np.zeros_like(HK)  
            PK[3,:,:,:] =  (HK[0,:,:,:]+HK[3,:,:,:]-(HK[1,:,:,:]+HK[2,:,:,:]))\
                           /(dist1.dxbins[None,:,None]*dist2.dxbins[None,None,:])
            PK[1,:,:,:] = ((HK[1,:,:,:]-HK[0,:,:,:])/dist1.dxbins[None,:,None])\
                           -PK[3,:,:,:]*dist2.xi1[None,None,:] 
            PK[2,:,:,:] = ((HK[2,:,:,:]-HK[0,:,:,:])/dist2.dxbins[None,None,:])\
                           -PK[3,:,:,:]*dist1.xi1[None,:,None] 
            PK[0,:,:,:] =   HK[0,:,:,:]\
                           -PK[1,:,:,:]*dist1.xi1[None,:,None]\
                           -PK[2,:,:,:]*dist2.xi1[None,None,:]\
                           -PK[3,:,:,:]*dist1.xi1[None,:,None]*dist2.xi1[None,None,:]
            
        # Explicitly set Sum, product, and constant kernels. This is because
        # numerical round-off errors are large.
        if self.kernel == 'Golovin':
           PK = HK.copy()
           PK[0,:,:,:] = 0.0
           PK[1,:,:,:] = 1.0
           PK[2,:,:,:] = 1.0
           PK[3,:,:,:] = 0.0
           
        elif self.kernel == 'Product':
           PK = HK.copy()
           PK[0,:,:,:] = 0.0
           PK[1,:,:,:] = 0.0
           PK[2,:,:,:] = 0.0
           PK[3,:,:,:] = 1.0
           
        elif self.kernel == 'Constant':
           PK = HK.copy()
           PK[0,:,:,:] = 1.0
           PK[1,:,:,:] = 0.0
           PK[2,:,:,:] = 0.0
           PK[3,:,:,:] = 0.0
        
        return PK       
   

    def calc_smootherstep(self,d_array, d_start, d_end):
        """
        Calculates the quintic smootherstep function over an array.
        Mathematically guarantees 1st and 2nd derivatives are zero at the boundaries.
        
        Args:
            d_array (np.ndarray): Array of diameters to evaluate.
            d_start (float): Diameter where efficiency starts to rise above 0.
            d_end (float): Diameter where efficiency reaches 1.0.
            
        Returns:
            np.ndarray: Scale factors from 0.0 to 1.0 matching the shape of d_array.
        """
        scale = np.zeros_like(d_array, dtype=np.float64)
            
        # Create boolean masks for the three regions
        mask_ramp = (d_array >= d_start) & (d_array < d_end)
        mask_full = (d_array >= d_end)
        
        # 1. Map the ramp region to a normalized coordinate 't' (0.0 to 1.0)
        t = (d_array[mask_ramp] - d_start[mask_ramp]) / (d_end[mask_ramp] - d_start[mask_ramp])
        
        # 2. Evaluate the Quintic Polynomial: 6t^5 - 15t^4 + 10t^3
        # We use the nested/factored form (Horner's method) because it is 
        # mathematically equivalent but much faster for the CPU to compute.
        scale[mask_ramp] = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
        
        # 3. Set everything above the ramp to 1.0
        scale[mask_full] = 1.0
        
        return scale
   

    def setup_fragments(self):
        
            scale_factor = np.ones_like(self.kmin,dtype=np.float64)    
            
            if (self.kernel=='Hydro') or (self.kernel=='Long'):
            
                # Largest possible size of fragments based on grid
               # d_frag = self.dists[self.indb].d2[self.kmid]
            
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
                    
                    perc_start = 0.5
                    perc_end = 0.95
                    
                    d_start = np.exp(muf+np.sqrt(2*sig2f)*erfinv(2.*perc_start-1))
                    d_end = np.exp(muf+np.sqrt(2*sig2f)*erfinv(2.*perc_end-1))

                    d_cen = self.dists[self.indb].d[self.k_lim] 
                    d_left = self.dists[self.indb].d1[self.k_lim] 
                    d_right = self.dists[self.indb].d2[self.k_lim]
                    
                    dD = d_right-d_left
    
                    # Create a boolean mask of where the ramp is currently too narrow
                    narrow_mask = (d_end - d_start) < dD

                    midpoint = (d_start+d_end)/2.
                    
                    min_ramp_width = 1.5 * dD
                    
                    half_min_width = min_ramp_width/2.
                    
                    d_start = np.where(narrow_mask, midpoint - half_min_width, d_start)
                    d_end   = np.where(narrow_mask, midpoint + half_min_width, d_end) 

                    E_left = self.calc_smootherstep(d_left,d_start,d_end)
                    E_cen = self.calc_smootherstep(d_cen,d_start,d_end)
                    E_right = self.calc_smootherstep(d_right,d_start,d_end)

                    # Simpson's rule for averaging out the smoothsteps.
                    scale_factor = (E_left+4.*E_cen+E_right)/6.

                elif (self.frag_dict['dist']=='Straub'):
                    
                    IF_func = self.setup_Straub()
                    
                    
                    
                   # return


                d1 = self.dists[self.indb].d1
                d2 = self.dists[self.indb].d2

                Mb_gain_vec = self.dists[self.indb].am*IF_func(self.dists[self.indb].bm,d1,d2)
                Nb_gain_vec = IF_func(0.,d1,d2)
                
                #print('Mb_Gain_vec=',Mb_gain_vec.max())
                #print('Nb_gain_vec=',Nb_gain_vec.shape)
                #raise Exception()
                
                # The Straub et al. (2010) fragment distribution parameterization is already a conditional distribution.
                # If one of the other distributions is chosen, assume that the conditional distribution is identical to 
                # the marginal distribution along the breakup distribution axis (i.e., assumption of independence: p(m|x,y) = p(m)).
                if (self.frag_dict['dist']!='Straub'):
                    self.dMb_gain_frac  = np.tile(Mb_gain_vec[:,None],(1,self.bins))
                    self.dNb_gain_frac  = np.tile(Nb_gain_vec[:,None],(1,self.bins))
                
            else:
                for xx in range(self.bins): # m1+m2 breakup mass
                   for kk in range(0,xx+1): # breakup gain bins
                       self.dMb_gain_frac[kk,xx] = In_int(1.,self.frag_dict['lamf'],self.xi1[kk],self.xi2[kk])
                       self.dNb_gain_frac[kk,xx] = In_int(0.,self.frag_dict['lamf'],self.xi1[kk],self.xi2[kk])   
       
            # If needed, convert self.dMb_gain_frac (p(m)) into conditional distribution (p(m|x,y))

            
            # 1. Clean up negative noise and invalid bounds
            invalid_mask = np.tri(self.bins, self.bins, k=-1, dtype=bool)
            self.dMb_gain_frac[invalid_mask] = 0.0 
            self.dNb_gain_frac[invalid_mask] = 0.0

            self.dMb_gain_frac[self.dMb_gain_frac < 0.0] = 0.0
            self.dNb_gain_frac[self.dNb_gain_frac < 0.0] = 0.0

            # ---------------------------------------------------------
            # PASS 1: Initial Normalization
            # ---------------------------------------------------------
            dMb_gain_tot = np.sum(self.dMb_gain_frac, axis=0)
            
            # Avoid divide-by-zero for empty columns
            valid_cols = dMb_gain_tot > 1e-100 
            
            self.dMb_gain_frac[:, valid_cols] /= dMb_gain_tot[None, valid_cols]
            self.dNb_gain_frac[:, valid_cols] /= dMb_gain_tot[None, valid_cols]
            
            # ---------------------------------------------------------
            # PASS 2: Censor the Noise & Re-normalize
            # ---------------------------------------------------------
            fit_threshold = 1e-12
            
            # Find bins that receive a physically meaningless fraction of the mass/number
            censor_mask_M = self.dMb_gain_frac < fit_threshold
            censor_mask_N = self.dNb_gain_frac < fit_threshold
            
            # Combine masks: If a bin is killed for M, it must be killed for N
            censor_mask = censor_mask_M | censor_mask_N
            
            self.dMb_gain_frac[censor_mask] = 0.0
            self.dNb_gain_frac[censor_mask] = 0.0
            
            # Re-sum the columns after trimming the tails
            dMb_gain_tot_2 = np.sum(self.dMb_gain_frac, axis=0)
            
            # Find columns that survived the censoring
            surviving_cols = dMb_gain_tot_2 > 0.0
            
            # Re-normalize so the trimmed distributions perfectly sum to 1.0 again
            self.dMb_gain_frac[:, surviving_cols] /= dMb_gain_tot_2[None, surviving_cols]
            self.dNb_gain_frac[:, surviving_cols] /= dMb_gain_tot_2[None, surviving_cols]
            
            # Clean up the dead columns
            self.dMb_gain_frac[:, ~surviving_cols] = 0.0
            self.dNb_gain_frac[:, ~surviving_cols] = 0.0

            # ---------------------------------------------------------
            # Finally: Apply scale factor only to surviving columns
            # ---------------------------------------------------------
            valid_matrix = surviving_cols[self.k_lim]
            scale_factor[~valid_matrix] = 0.0
            
            # Broadcast and Assign
            self.Ebr = scale_factor[None, None, :, :] * self.Ebr
            self.scale_factor = scale_factor
      
            # # Only consider breakup if the breakup distribution is complete.
            # censor_Ebr = np.broadcast_to((self.dMb_gain_frac.sum(axis=0)<0.5),self.Ebr.shape)    
            # self.Ebr[censor_Ebr] = 0.
              
    
            # # NOTE: Before, I was (stupidly?) using the actual limits from the SBE. 
            # # HOWEVER, this isn't physically realistic when coalescence and breakup
            # # are considered (and calculated) as mutually exclusive scenarios. 
            # # This presents a big problem because realistic fragment distributions
            # # like the lognormal distribution has a fat tail and thus the presence of breakup
            # # can actually generates particles much larger than either of the colliding
            # # particles! In order to compare results with Feingold et al. (1988), the
            # # kmin limit is used. For realistic kernels, the maximum index of the interacting
            # # species is used instead.
            # if (self.kernel=='Constant') | (self.kernel=='Golovin') | (self.kernel=='Product'):
            #     self.dMb_gain_kernel = self.dMb_gain_frac[:,self.kmin]
            #     self.dNb_gain_kernel = self.dNb_gain_frac[:,self.kmin]
            # else:
            #     self.dMb_gain_kernel = self.dMb_gain_frac[:,self.k_lim]
            #     self.dNb_gain_kernel = self.dNb_gain_frac[:,self.k_lim]  
          

    def setup_Straub(self):
        
        '''
        Sets up Straub et al. (2010) Fragment distributions.
        '''
        
        # Get Straub distribution parameters.
        # NOTE: For simplicity, I'm assuming that users will only be using
        # this distribution function for rain. Thus, we don't need to loop 
        # through all distribution combinations (i.e., assume that all dists
        # share the same size, fallspeed, etc. grids).
        
        dist1 = self.dists[0] 
        dist2 = self.dists[0]

        # Get Straub parameters. Note, for simplicity just use bin midpoints
        # To get Straub's four fragment distribution parameters. Ideally, this 
        # would be done in some clever way by taking into account all mass 
        # combinations between the bin limits.
        straub_dict = Straub_params(dist1.d,dist2.d,dist1.vt,dist2.vt)
        
        self.straub_dict = straub_dict
        
        # Distribution 1: A Lognormal distribution
        frag_dist1 = lambda n,x1,x2:straub_dict['dist1']['N']*LGN_int(n,straub_dict['dist1']['muf'],
                                              straub_dict['dist1']['sig2f'],
                                              x1,x2)
        # Distribution 2: A Gaussian distribution
        frag_dist2 = lambda n,x1,x2: straub_dict['dist2']['N']*GAU_int(n,straub_dict['dist2']['mu'],
                                               straub_dict['dist2']['sig2'],
                                               x1,x2)
        # Distribution 3: Another Gaussian distribution
        frag_dist3 = lambda n,x1,x2: straub_dict['dist3']['N']*GAU_int(n,straub_dict['dist3']['mu'],
                                               straub_dict['dist3']['sig2'],
                                               x1,x2)
        
        # Find bin for residual drop
        x_res = 0.001*(np.pi/6.)*straub_dict['dist4']['x_res'] # Get mass (assume rain for now)
        
        # Find bin that residual is supposed to go in
        xres_ind = np.searchsorted(self.xi2,x_res,side='right')
        
        frag4_num = np.zeros_like(self.xi2)
        frag4_mass = np.zeros_like(self.xi2)
        
        frag4_num[xres_ind] = 1.0 # Only one drop for remnant
        frag4_mass[xres_ind] = x_res # Mass of the one particle assuming mass conservation overall for the binary interaction
           
        # Distribution 4: A Dirac delta spike in corresponding bin for either number or mass
        frag_dist4 = lambda n,x1,x2: frag4_num if n==1 else frag4_mass
            
        #self.frag1 = lambda n,x1,x2: frag_dist1(n,x1,x2) 
        #self.frag2 = lambda n,x1,x2: frag_dist2(n,x1,x2)
        
        return lambda n,x1,x2: frag_dist1(n,x1,x2)+frag_dist2(n,x1,x2)+frag_dist3(n,x1,x2)+frag_dist4(n,x1,x2)
        
        
        
    
    
             

    def __plot_source_target(self,d1,d2,rtype,rind,invert=False,full=False):
        
        '''
        Method for plotting the integration region of the (kk,ii,jj) interaction.
        Useful for debugging.
        '''
        
        from matplotlib.patches import Polygon
        
        import matplotlib.pyplot as plt
        
        # Mcheck = (height x d1 bins x d2 bins)
        Mcheck = ((self.Mbins[d1,:,:]==0.)[:,:,None]) | ((self.Mbins[d2,:,:]==0.)[:,None,:]) # If M1 or M2 is zero, do not include in bin-pair list.

        cond_1 = self.cond_1 | Mcheck # New cond_1. Basically exclude bin-pairs that are off grid and ones involving empty bins.
        
        dd = int((d1*(d2+1))/2)-1
        
        # Get 3D indices for all bins that will interact at all heights
        # indices have shape (bin-pairs,) where ir are dist1 indices 
        # and jr are dist2 indices. kr are height indices.
        kr, ir, jr = np.nonzero((~cond_1)&self.self_col[:,dd,:,:])

        # (dnum x height x bins)
        x11  = self.x1[d1,kr,ir]
        x21  = self.x2[d1,kr,ir]
        ak1  = self.aki[d1,kr,ir]
        ck1  = self.cki[d1,kr,ir] 
        M1   = self.Mbins[d1,kr,ir]
        N1   = self.Nbins[d1,kr,ir] 
        PK   = self.PK[:,dd,ir,jr]
        
        x12  = self.x1[d2,kr,jr]
        x22  = self.x2[d2,kr,jr]
        M2   = self.Mbins[d2,kr,jr]
        N2   = self.Nbins[d2,kr,jr] 
        ak2  = self.aki[d2,kr,jr] 
        ck2  = self.cki[d2,kr,jr]     
        
        xk_min = self.xi2[self.kmin[ir,jr]]
        
        region_inds, x_bottom_edge, x_top_edge, y_left_edge, y_right_edge=setup_regions(self.bins,kr,ir,jr,x11,x21,x12,x22,xk_min)

        dMi_loss, dMj_loss, dM_gain, dNi_loss, dN_gain = calculate_regions_batch(self.Hlen,self.bins,kr,ir,jr,x11,x21,ak1,ck1,x12,x22,ak2,ck2,PK,region_inds,
                            x_bottom_edge,x_top_edge,y_left_edge,y_right_edge,self.w,self.L)
        
        # (Hlen,bins,bins)
        M1_loss = np.nansum(dMi_loss,axis=2) # Loss of dist1 mass with collisions from dist2
        N1_loss = np.nansum(dNi_loss,axis=2) # Loss of dist1 number with collisions from dist2
        
        M2_loss = np.nansum(dMj_loss,axis=1) # Loss of dist2 mass with collisions from dist1
        N2_loss = np.nansum(dNi_loss,axis=1) # Loss of dist2 number with collisions from dist1
        
        #print('kmin=',kmin.shape)
        #raise Exception()
        
        # ChatGPT is the GOAT for telling me about np.add.at!
        M_gain = np.zeros((self.Hlen,self.bins))
        np.add.at(M_gain, (np.arange(self.Hlen)[:,None,None],self.kmin), dM_gain[:,:,:,0])
        np.add.at(M_gain, (np.arange(self.Hlen)[:,None,None],self.kmid), dM_gain[:,:,:,1])
        
        N_gain = np.zeros((self.Hlen,self.bins))
        np.add.at(N_gain,  (np.arange(self.Hlen)[:,None,None],self.kmin), dN_gain[:,:,:,0])
        np.add.at(N_gain,  (np.arange(self.Hlen)[:,None,None],self.kmid), dN_gain[:,:,:,1])

        print('total mass bin-bin gain=',np.sum(dM_gain))
        print('total mass bin-bin loss=',np.sum(dMi_loss)+np.sum(dMj_loss))
        
        print('total number bin-bin gain=',np.sum(dN_gain))
        print('total number bin-bin loss=',np.sum(dNi_loss))
        
        print('--------')
        
        print('total mass loss=',np.sum(M1_loss)+np.sum(M2_loss))
        print('total mass gain=',np.sum(M_gain))
        
        print('total number loss=',np.sum(N1_loss)+np.sum(N2_loss))
        print('total number gain=',np.sum(N_gain))
        
        print('--------')

        k2 = np.flatnonzero(region_inds==2)
        k3 = np.flatnonzero(region_inds==3) 
        k4 = np.flatnonzero(region_inds==4) 
        k5 = np.flatnonzero(region_inds==5) 
        k6 = np.flatnonzero(region_inds==6) 
        k7 = np.flatnonzero(region_inds==7)
        k8 = np.flatnonzero(region_inds==8) 
        k9 = np.flatnonzero(region_inds==9) 
        k10 = np.flatnonzero(region_inds==10) 
        k11 = np.flatnonzero(region_inds==11) 
        k12 = np.flatnonzero(region_inds==12)

        print('Number of region 2 = {}'.format(len(k2)))
        print('Number of region 3 = {}'.format(len(k3)))
        print('Number of region 4 = {}'.format(len(k4)))
        print('Number of region 5 = {}'.format(len(k5)))
        print('Number of region 6 = {}'.format(len(k6)))
        print('Number of region 7 = {}'.format(len(k7)))
        print('Number of region 8 = {}'.format(len(k8)))
        print('Number of region 9 = {}'.format(len(k9)))
        print('Number of region 10 = {}'.format(len(k10)))
        print('Number of region 11 = {}'.format(len(k11)))
        print('Number of region 12 = {}'.format(len(k12)))
        print('Total Region Number = {} out of {}'.format(len(k2)+len(k3)+len(k4)+len(k5)+
                                                          len(k6)+len(k7)+len(k8)+len(k9)+
                                                          len(k10)+len(k11)+len(k12),len(kr)))
        
        kt = np.flatnonzero(region_inds==rtype)

        ind = kt[rind]
        
        print('ind=',ind)
        
    
        kk = kr[ind]
        ii = ir[ind]
        jj = jr[ind]
        
        kmin = self.kmin[ii,jj]
        kmid = self.kmid[ii,jj]

        dist1 = self.dists[d1,kk]
        dist2 = self.dists[d2,kk]

        print('x21=',x21[ind])
        print('xi2=',self.xi2[ii])
        print('x12=',x12[ind])
        print('xi1=',self.xi1[jj])
        print('y_left_edge=',y_left_edge[ind])
        print('y_right_edge=',y_right_edge[ind])
        print('x_bottom_edge=',x_bottom_edge[ind])
        print('x_top_edge=',x_top_edge[ind])
        
        print('----')
        print('({},{}) -- ({},{})'.format(kk,ii,kk,jj))
        print('k bin={} | k+1 bin={}'.format(kmin,kmid))
        print('Mass source=',dMi_loss[kk,ii,jj]+dMj_loss[kk,ii,jj])
        print('Number source=',dNi_loss[kk,ii,jj])
        print('Mass gain k bin=',dM_gain[kk,ii,jj,0])
        print('Mass gain k+1 bin=',dM_gain[kk,ii,jj,1])
        print('Number gain k bin=',dN_gain[kk,ii,jj,0])
        print('Number gain k+1 bin=',dN_gain[kk,ii,jj,1])
        print('Mass gain total=',dM_gain[kk,ii,jj,0]+dM_gain[kk,ii,jj,1])
        print('Number gain total=',dN_gain[kk,ii,jj,0]+dN_gain[kk,ii,jj,1])
        
        # print('x21=',x21[kk,ii])
        # print('xi2=',self.xi2[ii])
        # print('x12=',x12[kk,jj])
        # print('xi1=',self.xi1[jj])
        # print('y_left_edge=',y_left_edge[kk,ii,jj])
        # print('y_right_edge=',y_right_edge[kk,ii,jj])
        # print('x_bottom_edge=',x_bottom_edge[kk,ii,jj])
        # print('x_top_edge=',x_top_edge[kk,ii,jj])
  
        # if using subgrid linear distribution 
        xi1 = dist1.x1[ii]
        xi2 = dist1.x2[ii]
        xj1 = dist2.x1[jj]
        xj2 = dist2.x2[jj]
        
        #xi1 = self.xi1[ir]
        #xi2 = self.xi2[ir]
        #xj1 = self.xi1[jr]
        #xj2 = self.xi2[jr]
        
        print('xi1=',xi1)
        print('xi2=',xi2)
        print('xj1=',xj1)
        print('xj2=',xj2)
        print('M1=',M1[ind])
        print('M2=',M2[ind])
        print('N1=',N1[ind])
        print('N2=',N2[ind])
        print('ak1=',ak1[ind])
        print('ak2=',ak2[ind])
        print('ck1=',ck1[ind])
        print('ck2=',ck2[ind])
    
        # NOTE: Need to avoid doing anything with the last bin
        kbin_min = dist1.xi1[:,None]+dist2.xi1[None,:]
        kbin_max = dist1.xi2[:,None]+dist2.xi2[None,:]
        
        fig, ax = plt.subplots()   
        
        if invert:
            source_rec = Polygon(((xj1,xi1),(xj1,xi2),(xj2,xi2),(xj2,xi1)))
        else:           
            source_rec = Polygon(((xi1,xj1),(xi1,xj2),(xi2,xj2),(xi2,xj1)))

        
        ax.add_patch(source_rec)
        ax.autoscale_view()
        
        print('({},{})'.format(xi1,xj1))
        print('({},{})'.format(xi1,xj2))
        print('({},{})'.format(xi2,xj2))
        print('({},{})'.format(xi2,xj1))
        
        #return fig, ax
        
        #raise Exception()
        
        if invert:
            # NOTE: need to reformat with new gain bin regions
            ax.plot([kbin_min[ii,jj]-xi1,kbin_min[ii,jj]-xi2],[xi1,xi2],'b')
            ax.plot([xk_min-xi1,xk_min-xi2],[xi1,xi2],'k')
            ax.plot([kbin_max[ii,jj]-xi1,kbin_max[ii,jj]-xi2],[xi1,xi2],'r')

            ax.invert_yaxis()
            
        else:
            
            if rtype==2:
                # Triangle = ((xi1,xj1),(xi1,x_left_edge),(xi2,xj1))
                kgain_t = Polygon(((xi1,xj1),(xi1,y_left_edge[ind]),(xi2,xj1)),closed=True,facecolor='purple')
            elif rtype==3:
                kgain_t = Polygon((((xi1,xj1),(xi1,xj2),(x_bottom_edge[ind],xj1))),closed=True,facecolor='purple')
            elif rtype==11:
                # Gain integral
                kgain_t = Polygon((((xi2,xj1),(x_top_edge[ind],xj2),(xi2,xj2))),closed=True,facecolor='purple')
            elif rtype==12:
                kgain_t = Polygon(((xi1,xj2),(xi2,xj2),(xi2,y_right_edge[ind])),closed=True,facecolor='purple')
            elif rtype==7:
                kgain_t = Polygon(((x_top_edge[ind],xj2),(xi2,xj2),(xi2,y_right_edge[ind])),closed=True,facecolor='purple')
            elif rtype==8:
                kgain_t = Polygon(((xi1,xj1),(xi1,y_left_edge[ind]),(x_bottom_edge[ind],xj1)),closed=True,facecolor='purple')
            elif np.isin(rtype,[4,9,10]):
                kgain_r = Polygon(((xi1,xj1),(xi1,xj2),(xi2,xj2),(xi2,xj1)),closed=True,facecolor='purple')
            elif rtype==5:
                kgain_t = Polygon(((x_top_edge[ind],xj1),(x_top_edge[ind],xj2),(x_bottom_edge[ind],xj1)),closed=True,facecolor='purple')
                kgain_r =Polygon(((xi1,xj1),(xi1,xj2),(x_top_edge[ind],xj2),(x_top_edge[ind],xj1)),closed=True,facecolor='orange')
            elif rtype==6:
                kgain_t = Polygon(((xi1,y_right_edge[ind]),(xi1,y_left_edge[ind]),(xi2,y_right_edge[ind])),closed=True,facecolor='purple')
                kgain_r =Polygon(((xi1,xj1),(xi1,y_right_edge[ind]),(xi2,y_right_edge[ind]),(xi2,xj1)),closed=True,facecolor='orange')
            
            if  (np.isin(rtype,[2,3,5,6,7,8,11,12])):
                ax.add_patch(kgain_t)
            
            if (np.isin(rtype,[4,5,6,9])):
                ax.add_patch(kgain_r)
            
            
            ax.plot([dist1.xi1[ii],dist1.xi2[ii]],[kbin_min[ii,jj]-dist1.xi1[ii],kbin_min[ii,jj]-dist1.xi2[ii]],'b')
            ax.plot([dist1.xi1[ii],dist1.xi2[ii]],[xk_min[ind]-dist1.xi1[ii],xk_min[ind]-dist1.xi2[ii]],'k')
            ax.plot([dist1.xi1[ii],dist1.xi2[ii]],[kbin_max[ii,jj]-dist1.xi1[ii],kbin_max[ii,jj]-dist1.xi2[ii]],'r')
    
        return fig, ax

    def interact_1mom_SS_Final_FC(self, dt):
        """
        Flux-Corrected Solver.
        Calculates max permissible loss per bin, scales fluxes, 
        and applies them. Strictly conservative and stable.
        """
        self.get_dynamic_params_1mom()
        
        if len(self.params['active_rate']) > 10000:
            nb.set_num_threads(nb.config.NUMBA_DEFAULT_NUM_THREADS)
        else:
            nb.set_num_threads(1)

        # 1. Calculate Potential Fluxes
        # M_loss is the mass the physics WANTS to remove.
        M_loss, M_gain = vectorized_1mom(
            self.cki, self.params, self.dMi_loss, 
            self.dMj_loss, self.dM_gain,
            self.kmin_p, self.kmid_p, self.dMb_gain_kernel, 
            self.indc, self.indb, 
            self.dnum, self.Hlen, self.bins
        )

        # 2. Calculate the Limiters (The "Reality Check")
        # How much mass can actually leave bin 'k' in time 'dt'?
        # We limit removal to 99% of current mass to stay positive.
        max_loss = 0.99 * self.Mbins
        
        requested_loss = M_loss * dt
        
        # Alpha is the fraction of the requested flux allowed to leave bin 'k'
        # alpha = 1.0 (Safe)
        # alpha < 1.0 (Limited)
        alpha = np.ones_like(self.Mbins)
        
        mask = requested_loss > 1e-30 # Avoid div/0
        alpha[mask] = np.minimum(1.0, max_loss[mask] / requested_loss[mask])
        
        # 3. Apply the Limiters to the Fluxes
        # This is the tricky part. 
        # Loss[k] is scaled by alpha[k].
        # Gain[k] comes from OTHER bins (i, j). It must be scaled by alpha[i] and alpha[j].
        
        # We need to map 'alpha' back to the collisions.
        # Since we can't easily invert the gain kernel, we use a 
        # "Global Weighted Limiter" or re-run the kernel with scaled rates.
        
        # RE-RUN METHOD (Safest and Correct):
        # We scale the 'active_rate' by min(alpha_i, alpha_j).
        
        # Map alpha to the collision pairs
        #h_act = self.params['active_h']
        #s1_act = self.params['active_s1']
        #s2_act = self.params['active_s2']
        i_act = self.params['active_i']
        j_act = self.params['active_j']
        
        alpha_i = alpha[i_act] # (Or correct mapping based on s1_act/h_act if multidim)
        alpha_j = alpha[j_act]
        
        # The limiter for collision (i,j) is the stricter of the two source limits
        pair_alpha = np.minimum(alpha_i, alpha_j)
        
        # 4. Final Calculation
        # Instead of calling the full kernel again, we can just compute the result directly?
        # No, because M_gain aggregates many pairs. 
        # We must re-run the kernel ONE time with the scaled rates.
        
        # This is your "Pass 2" from before, but with a critical difference:
        # We computed alpha based on TOTAL loss, not just individual rates.
        
        orig_rates = self.params['active_rate'].copy()
        self.params['active_rate'] *= pair_alpha
        
        M_loss_final, M_gain_final = vectorized_1mom(
            self.cki, self.params, self.dMi_loss, 
            self.dMj_loss, self.dM_gain,
            self.kmin_p, self.kmid_p, self.dMb_gain_kernel, 
            self.indc, self.indb, 
            self.dnum, self.Hlen, self.bins
        )
        
        self.params['active_rate'] = orig_rates
        
        return dt * (M_gain_final - M_loss_final)


    def interact_1mom_SS_Final_adaptive(self, dt):
        """
        Adaptive Sub-stepping for 1-Moment Scheme.
        Optimized to behave like Single-Step Euler unless stability is threatened.
        """
        
        # 1. Local Working Copy (Don't touch self.Mbins yet)
        M_current = self.Mbins.copy()
        
        # 1. Base Setup
        self.get_dynamic_params_1mom()
        
        t_evolved = 0.0
        
        # Accumulator for final net change
        M_net_total = np.zeros_like(M_current)
        
        max_substeps = 20 
        loop_count = 0
        
        while t_evolved < dt and loop_count < max_substeps:
            
            # -----------------------------------------------------------------
            # A. Physics Update (Only on 2nd+ step)
            # -----------------------------------------------------------------
            # If loop_count == 0, we use the params passed into the function.
            # We only update if the mass has actually changed.
            if loop_count > 0:
                self.Mbins = M_current
                self.update_1mom_subgrid()
                self.get_dynamic_params_1mom()
                
                if len(self.params['active_rate']) > 10000:
                    nb.set_num_threads(nb.config.NUMBA_DEFAULT_NUM_THREADS)
                else:
                    nb.set_num_threads(1)
            
            # -----------------------------------------------------------------
            # B. Run Kernel (Predictor)
            # -----------------------------------------------------------------
            M_loss, M_gain = vectorized_1mom(self.cki, self.params, self.dMi_loss, 
                                         self.dMj_loss, self.dM_gain,
                                         self.kmin_p, self.kmid_p, self.dMb_gain_kernel, 
                                         self.indc, self.indb, 
                                         self.dnum, self.Hlen, self.bins)
            # -----------------------------------------------------------------
            # C. Check Stability (Relative Thresholds)
            # -----------------------------------------------------------------
            peak_M = np.max(M_current)
            
            # Threshold: 0.1% of peak, with safety floor
            thresh_M = max(peak_M * 1e-3, 1e-20)
            
            sig_mask_M = M_current > thresh_M
            
            # Default: Take the rest of the step
            dt_step = dt - t_evolved 
            max_safe_dt = dt_step
            
            # Stability criterion: Don't deplete more than 90% of any significant bin
            if np.any(sig_mask_M):
                # Calculate turnover time tau = Mass / Loss
                # Add epsilon to Loss to prevent divide-by-zero
                loss_rates = M_loss[sig_mask_M] + 1e-30
                masses     = M_current[sig_mask_M] + 1e-30
                
                tau_M = masses / loss_rates
                
                # We limit the step to 90% of the fastest turnover time
                max_safe_dt = min(max_safe_dt, 0.9 * np.min(tau_M))

            # Apply Limit
            dt_step = min(dt_step, max_safe_dt)
            
            # Prevent infinitesimal steps (infinite loop guard)
            min_dt_limit = dt / max_substeps
            if dt_step < min_dt_limit and (dt - t_evolved) > min_dt_limit:
                 dt_step = min_dt_limit

            # -----------------------------------------------------------------
            # D. Evolve State
            # -----------------------------------------------------------------
            dM_step = dt_step * (M_gain - M_loss)
            
            M_current   += dM_step
            M_net_total += dM_step
            
            t_evolved   += dt_step
            loop_count  += 1

        # 3. Reset Global Object State
        # We manually update self.Mbins to the final result minus the net change
        # so the main loop can add M_net_total cleanly.
        self.Mbins = M_current - M_net_total
        
        # Restore parameters to the final state for the next timestep
        self.update_1mom_subgrid()
        
        return M_net_total   

# NEED TO DECIDE WHICH solver to use
    def interact_1mom_SS_Final_adaptive2(self, dt):
        """
        Frozen-Rate Adaptive Solver.
        Fast (1 kernel call) + Smooth (Analytical Integration).
        """
        # 1. Base Setup
        self.get_dynamic_params_1mom()
        
        n_active = len(self.params['active_rate'])
        if n_active > 10000:
            nb.set_num_threads(nb.config.NUMBA_DEFAULT_NUM_THREADS)
        else:
            nb.set_num_threads(1)

        # 2. Run Kernel ONCE (The Expensive Part)
        # We assume the interaction RATES (collisions per second) are constant 
        # over the timestep, even if the MASS changes.
        M_loss, M_gain = vectorized_1mom(
            self.cki, self.params, self.dMi_loss, 
            self.dMj_loss, self.dM_gain,
            self.kmin_p, self.kmid_p, self.dMb_gain_kernel, 
            self.indc, self.indb, 
            self.dnum, self.Hlen, self.bins
        )

        # 3. Analytical Sub-stepping (The Fast Part)
        # We evolve the distribution using the fixed rates calculated above.
        M_current = self.Mbins.copy()
        M_start = M_current.copy()
        
        t_evolved = 0.0
        remaining_dt = dt
        
        # Define "Significant" mass threshold to avoid dividing by zero/noise
        peak_mass = np.max(M_current)
        threshold = max(peak_mass * 1e-4, 1e-20)
        
        while t_evolved < dt:
            
            # A. Calculate max safe step for this sub-interval
            # We want to ensure no bin loses > 50% of its current mass in one sub-step
            # to maintain accuracy of the linear approximation.
            
            sig_mask = M_current > threshold
            
            # Default step is the rest of the time
            dt_sub = remaining_dt
            
            if np.any(sig_mask):
                # Rate = Loss_Flux / Current_Mass
                # Note: M_loss from kernel is "Mass/sec" based on M_start.
                # We scale it by (M_current / M_start) to approximate 1st order decay.
                
                # Effective Loss Rate (1/s)
                # decay_rate = (M_loss / M_start) 
                # This stays constant if we assume frozen kinetics!
                
                denom = M_start[sig_mask] + 1e-30
                decay_rates = M_loss[sig_mask] / denom
                
                max_rate = np.max(decay_rates)
                
                # If rate is 0.1 s^-1, max step should be ~5 seconds (0.5 / 0.1)
                if max_rate > 0:
                    safe_dt = 0.5 / max_rate 
                    dt_sub = min(dt_sub, safe_dt)
            
            # Prevent tiny steps
            dt_sub = max(dt_sub, 1e-6)
            if t_evolved + dt_sub > dt:
                dt_sub = dt - t_evolved
                
            # B. Apply Updates
            # We scale the initial rates by the fraction of mass remaining
            # Gain_t = Gain_0 * (M_source / M_source_0) ... hard to track sources.
            # Simplified Hybrid: Just apply explicit Euler on small steps.
            
            # Since we froze the rates relative to M_start, we just apply:
            # dM = (Gain - Loss) * (dt_sub) * (Correction?)
            # The simplest valid approach for "Frozen Rate" is pure Euler on small steps.
            
            #dM = dt_sub * (M_gain - M_loss)
            
            # BUT! We must limit dM to not cross zero.
            # Since we calculated safe_dt above, this should be naturally safe.
            
            # Refined Approach: Scaling Loss by Depletion
            # Loss_now ~= Loss_0 * (M_current / M_start)
            # Gain_now ~= Gain_0 * (Total_Mass_Current / Total_Mass_Start) ?? 
            # Gains are harder because they come from other bins. 
            # Assumption: Global Mass is conserved, so Gains are roughly constant? 
            # actually Gains drop as sources deplete.
            
            # Let's try the Linear Scaling approximation:
           # scaling_factor = M_current / (M_start + 1e-30)
            
            # Apply scaling only to LOSS. Keep GAIN constant (conservative estimate)
            # or scale Gain by a global depletion factor?
            # Let's stick to the simplest conservative method:
            
            # Limit loss to available mass
           # actual_loss = M_loss * dt_sub
            # If actual_loss > M_current, we clamp it.
            # But since we chose dt_sub to be safe, this rarely happens.
            
            # To ensure strict conservation, we just apply the net flux:
            M_current += dt_sub * (M_gain - M_loss)
            
            # C. Advance
            t_evolved += dt_sub
            remaining_dt -= dt_sub
            
            # Safety break for infinite loops
            if dt_sub < 1e-9:
                break
        
        # Final cleanup for tiny negative zeros
        M_current[M_current < 0] = 0.0
        
        return M_current - M_start

    def interact_1mom_SS_Final_PC(self, dt):
        
        # 1. Base Setup
        self.get_dynamic_params_1mom()
        
        n_active = len(self.params['active_rate'])
        if n_active > 10000:
            nb.set_num_threads(nb.config.NUMBA_DEFAULT_NUM_THREADS)
        else:
            nb.set_num_threads(1)

        # Save the original unscaled rates
        original_rate = self.params['active_rate'].copy()

        # =====================================================================
        # PASS 1: The Predictor
        # =====================================================================
        M_loss, M_gain = vectorized_1mom(self.cki, self.params, self.dMi_loss, 
                                         self.dMj_loss, self.dM_gain,
                                         self.kmin_p, self.kmid_p, self.dMb_gain_kernel, 
                                         self.indc, self.indb, 
                                         self.dnum, self.Hlen, self.bins)
        
        # NOTE: This works for this coalescence-only case!
        #return dt * (M_gain - M_loss)

        # Calculate how much mass we are TRYING to remove relative to what we HAVE
        Mbins = self.Mbins
        
        # Depletion ratio = (Requested Loss) / (Available Mass)
        # Avoid divide-by-zero using 1e-30
        depletion = (M_loss * dt) / (Mbins + 1e-80)
        max_depletion = np.max(depletion)

        # =====================================================================
        # THE DECISION
        # =====================================================================
        # If no bin loses more than 95% of its mass, Explicit Euler is completely safe!
        if max_depletion <= 0.95:
            # FAST PATH: Perfect conservation, 1 kernel call.
            return dt * (M_gain - M_loss)

        # =====================================================================
        # PASS 2: The Corrector (Flux Limiter)
        # =====================================================================
        else:
            # We need to slow down the physics to prevent negative mass.
            # Calculate the safety fraction for every bin in the 3D grid.
            # If depletion is 2.0 (200%), alpha becomes 0.5 (run at half speed).
            alpha_grid = np.clip(1.0 / (depletion + 1e-16), 0.0, 1.0)
            
            # Map the safety fractions back to the 1D interaction arrays
            h_act = self.params['active_h']
            s1_act = self.params['active_s1']
            s2_act = self.params['active_s2']
            i_act = self.params['active_i']
            j_act = self.params['active_j']
            
            # Get the safety limit for the two parent bins of every collision
            alpha_i = alpha_grid[s1_act, h_act, i_act]
            alpha_j = alpha_grid[s2_act, h_act, j_act]
            
            # The collision rate is limited by the most heavily depleted parent
            safe_scaling = np.minimum(alpha_i, alpha_j)
            
            # Apply the limit to the collision rates
            self.params['active_rate'] = original_rate * safe_scaling
            
            # Run the kernel ONE MORE TIME with the safe rates
            M_loss_safe, M_gain_safe = vectorized_1mom(
                                         self.cki, self.params, self.dMi_loss, 
                                         self.dMj_loss, self.dM_gain,
                                         self.kmin_p, self.kmid_p, self.dMb_gain_kernel, 
                                         self.indc, self.indb, 
                                         self.dnum, self.Hlen, self.bins)
            
            # Restore the original rates to keep the object state clean
            self.params['active_rate'] = original_rate
            
            # SLOW PATH: Perfectly conservative, guaranteed positive, 2 kernel calls.
            return dt * (M_gain_safe - M_loss_safe)


    def interact_1mom_SS_Final(self, dt):
        # Reset master buffers
        
        self.get_dynamic_params_1mom()

        n_active = len(self.params['active_rate'])

        if n_active>10000:
            nb.set_num_threads(nb.config.NUMBA_DEFAULT_NUM_THREADS)
        else:
            nb.set_num_threads(1)

        M_loss, M_gain = vectorized_1mom(self.cki, self.params, self.dMi_loss, 
                                         self.dMj_loss,self.dM_gain,
                                         self.kmin_p,self.kmid_p,self.dMb_gain_kernel, 
                                         self.indc, self.indb, 
                                         self.dnum, self.Hlen, self.bins)
    
        return dt * (M_gain-M_loss)
    
    
    def interact_2mom_SS_Final_FC(self, dt):
        """
        Strictly Conservative Flux-Corrected 2-Moment Solver.
        Scales Efficiencies (Eagg, Ebr) to enforce mass/number conservation limits.
        """
        # 1. Base Setup
        self.get_dynamic_params()
        
        n_active = len(self.params['regions'])
        if n_active > 10000:
            nb.set_num_threads(nb.config.NUMBA_DEFAULT_NUM_THREADS)
        else:
            nb.set_num_threads(1)

        # Save original efficiencies (The "Control Knobs")
        # We will modify these in-place for the second pass
        orig_Eagg = self.params['Eagg'].copy()
        orig_Ebr  = self.params['Ebr'].copy()

        # -----------------------------------------------------------------
        # STEP 1: Calculate "Wishlist" Fluxes (Unlimited)
        # -----------------------------------------------------------------
        # Run the kernel with full efficiency (alpha=1.0)
        M_loss, M_gain, N_loss, N_gain = vectorized_2mom(
            self.params, self.w, self.L, self.dMb_gain_kernel, self.dNb_gain_kernel,
            self.indc, self.indb, self.dnum, 
            self.Hlen, self.bins
        )

        # -----------------------------------------------------------------
        # STEP 2: Calculate Bin-Level Safety Factors (Alphas)
        # -----------------------------------------------------------------
        # Determine the max permissible depletion for every bin in the grid.
        
        # Define available budget (e.g., 99.9% of current mass)
        M_avail = 0.999 * self.Mbins
        N_avail = 0.999 * self.Nbins
        
        # Desired removal (Flux * dt)
        M_req = M_loss * dt
        N_req = N_loss * dt
        
        # Initialize alpha grid (default = 1.0 = Safe)
        # Shape matches Mbins (e.g., [Hlen, bins])
        alpha_M = np.ones_like(self.Mbins)
        alpha_N = np.ones_like(self.Nbins)
        
        # Calculate limit where requested flux is non-zero
        # alpha = Available / Requested
        mask_M = M_req > 1e-30
        alpha_M[mask_M] = np.minimum(1.0, M_avail[mask_M] / M_req[mask_M])
        
        mask_N = N_req > 1e-30
        alpha_N[mask_N] = np.minimum(1.0, N_avail[mask_N] / N_req[mask_N])
        
        # The limit for a bin is the stricter of Mass or Number constraints
        # This keeps Mean Diameter (M/N) consistent.
        bin_alpha = np.minimum(alpha_M, alpha_N)

        # -----------------------------------------------------------------
        # STEP 3: Map Alphas to Interactions
        # -----------------------------------------------------------------
        # A collision between bin 'i' and bin 'j' is limited by the 
        # weakest link (the bin that is emptying out fastest).
        
        # Retrieve the index mapping from params
        d1_act = self.params['d1_ind'] # Parent 1 domain index
        d2_act = self.params['d2_ind'] # Parent 2 domain index
        h_act  = self.params['hind']   # Height/Spatial index
        i_act  = self.params['bi_ind'] # Parent 1 bin index
        j_act  = self.params['bj_ind'] # Parent 2 bin index
        
        # Look up the safety factor for parent i and parent j
        # bin_alpha is (dnum, Hlen, bins) or similar, depending on your grid
        # Assuming bin_alpha matches the shape of Mbins, we broadcast lookup:
        alpha_i = bin_alpha[d1_act, h_act, i_act]
        alpha_j = bin_alpha[d2_act, h_act, j_act]
        
        # The Global Safety Factor for each interaction 'k'
        # Shape: (n_active,) - one scalar per interaction
        interaction_alpha = np.minimum(alpha_i, alpha_j)

        # -----------------------------------------------------------------
        # STEP 4: Re-Run Kernel with Scaled Efficiencies
        # -----------------------------------------------------------------
        # We assume the kernel is linear with respect to Efficiency.
        # Loss = Integral * E * ...
        # Gain = Integral * E * ...
        # Scaling E by alpha scales Loss and Gain exactly equally.
        
        # Apply scaling to the efficiencies in the params dict
        self.params['Eagg'] *= interaction_alpha
        self.params['Ebr']  *= interaction_alpha
        
        # Run the physics one last time (The "Safe" Pass)
        M_loss_final, M_gain_final, N_loss_final, N_gain_final = vectorized_2mom(
            self.params, self.w, self.L, self.dMb_gain_kernel, self.dNb_gain_kernel,
            self.indc, self.indb, self.dnum, 
            self.Hlen, self.bins
        )
        
        # Restore original efficiencies to clean up state
        self.params['Eagg'][:] = orig_Eagg
        self.params['Ebr'][:]  = orig_Ebr

        # -----------------------------------------------------------------
        # STEP 5: Final Update
        # -----------------------------------------------------------------
        M_net = dt * (M_gain_final - M_loss_final)
        N_net = dt * (N_gain_final - N_loss_final)
        
        return M_net, N_net
    
    
    def interact_2mom_SS_Final_adaptive(self, dt):
        """
        Adaptive Sub-stepping for 2-Moment Scheme.
        Optimized to behave like Single-Step Euler unless stability is threatened.
        """
        
        # 1. Local Working Copies (Don't touch self.Mbins yet)
        M_current = self.Mbins.copy()
        N_current = self.Nbins.copy()
        
        self.get_dynamic_params()
        
        t_evolved = 0.0
        
        # Accumulators for final net change
        M_net_total = np.zeros_like(M_current)
        N_net_total = np.zeros_like(N_current)
        
        max_substeps = 20 
        loop_count = 0
        
        while t_evolved < dt and loop_count < max_substeps:
            
            # -----------------------------------------------------------------
            # A. Physics Update (Only on 2nd+ step)
            # -----------------------------------------------------------------
            # If loop_count == 0, we use the params passed into the function.
            # We only update if the mass has actually changed.
            if loop_count > 0:
                self.Mbins = M_current
                self.Nbins = N_current
                self.update_2mom_subgrid() 
                self.get_dynamic_params()
                
                # Check thread count again just in case regions changed
                if len(self.params['regions']) > 10000:
                    nb.set_num_threads(nb.config.NUMBA_DEFAULT_NUM_THREADS)
                else:
                    nb.set_num_threads(1)
            
            # -----------------------------------------------------------------
            # B. Run Kernel (Predictor)
            # -----------------------------------------------------------------
            M_loss, M_gain, N_loss, N_gain = vectorized_2mom(
                self.params, self.w, self.L, self.dMb_gain_kernel, self.dNb_gain_kernel,
                self.indc, self.indb, self.dnum, 
                self.Hlen, self.bins
            )
            
            # -----------------------------------------------------------------
            # C. Check Stability (Relative Thresholds)
            # -----------------------------------------------------------------
            # 1. Define "Significant" bins (ignore tiny numerical noise)
            peak_M = np.max(M_current)
            peak_N = np.max(N_current)
            
            # Threshold: 0.1% of peak, with safety floor
            thresh_M = max(peak_M * 1e-3, 1e-20)
            thresh_N = max(peak_N * 1e-3, 1e-20)
            
            sig_mask_M = M_current > thresh_M
            sig_mask_N = N_current > thresh_N
            
            # 2. Calculate Max Safe Time
            dt_step = dt - t_evolved # Default: Take the rest of the step
            max_safe_dt = dt_step
            
            # Stability criterion: Don't deplete more than 90% of any significant bin
            if np.any(sig_mask_M):
                tau_M = (M_current[sig_mask_M] + 1e-30) / (M_loss[sig_mask_M] + 1e-30)
                max_safe_dt = min(max_safe_dt, 0.9 * np.min(tau_M))

            if np.any(sig_mask_N):
                tau_N = (N_current[sig_mask_N] + 1e-30) / (N_loss[sig_mask_N] + 1e-30)
                max_safe_dt = min(max_safe_dt, 0.9 * np.min(tau_N))
                
            # 3. Apply Limit
            dt_step = min(dt_step, max_safe_dt)
            
            # Prevent infinite loops with minimum step size
            min_dt_limit = dt / max_substeps
            if dt_step < min_dt_limit and (dt - t_evolved) > min_dt_limit:
                 dt_step = min_dt_limit

            # -----------------------------------------------------------------
            # D. Evolve State
            # -----------------------------------------------------------------
            dM_step = dt_step * (M_gain - M_loss)
            dN_step = dt_step * (N_gain - N_loss)
            
            M_current   += dM_step
            N_current   += dN_step
            
            M_net_total += dM_step
            N_net_total += dN_step
            
            t_evolved   += dt_step
            loop_count  += 1

        # 3. Reset Global Object State
        # We manually update self.Mbins to the final result minus the net change
        # so the main loop can add M_net_total cleanly.
        self.Mbins = M_current - M_net_total
        self.Nbins = N_current - N_net_total
        
        # Restore parameters to the final state for the next timestep
        self.update_2mom_subgrid() 
        
        return M_net_total, N_net_total
    
  
    def interact_2mom_SS_Final_PC(self, dt):
        """Main method for the Interaction class (Predictor-Corrector)."""
        
        # 1. Base Setup
        self.get_dynamic_params()
        
        n_active = len(self.params['regions'])
        if n_active > 10000:
            nb.set_num_threads(nb.config.NUMBA_DEFAULT_NUM_THREADS)
        else:
            nb.set_num_threads(1)

        # Save the original efficiencies to reset them later
        orig_Eagg = self.params['Eagg'].copy()
        orig_Ebr  = self.params['Ebr'].copy()

        # =====================================================================
        # PASS 1: The Predictor
        # =====================================================================
        M_loss, M_gain, N_loss, N_gain = vectorized_2mom(
            self.params, self.w, self.L, self.dMb_gain_kernel, self.dNb_gain_kernel,
            self.indc, self.indb, self.dnum, 
            self.Hlen, self.bins
        )

        # =====================================================================
        # Calculate Depletion for BOTH Mass and Number
        # =====================================================================
        # Prevent divide-by-zero using 1e-30
        depletion_M = (M_loss * dt) / (self.Mbins + 1e-30)
        depletion_N = (N_loss * dt) / (self.Nbins + 1e-30)
        
        # Find the worst-case depletion anywhere in the domain
        max_dep_M = np.max(depletion_M)
        max_dep_N = np.max(depletion_N)
        max_depletion = max(max_dep_M, max_dep_N)

        # =====================================================================
        # THE DECISION
        # =====================================================================
        if max_depletion <= 0.95:
            # FAST PATH: Perfect conservation, 1 kernel call.
            M_net = dt * (M_gain - M_loss)
            N_net = dt * (N_gain - N_loss)
            return M_net, N_net

        # =====================================================================
        # PASS 2: The Corrector (Flux Limiter)
        # =====================================================================
        else:
            # Calculate the safety fraction (alpha) for every bin in the 3D grid
            alpha_grid_M = np.clip(1.0 / (depletion_M + 1e-16), 0.0, 1.0)
            alpha_grid_N = np.clip(1.0 / (depletion_N + 1e-16), 0.0, 1.0)
            
            # The bin is limited by whichever variable (M or N) is depleting faster
            alpha_grid = np.minimum(alpha_grid_M, alpha_grid_N)
            
            # Map safety fractions back to specific parent bins for each interaction
            d1_act = self.params['d1_ind']
            d2_act = self.params['d2_ind']
            h_act  = self.params['hind']
            i_act  = self.params['bi_ind']
            j_act  = self.params['bj_ind']
            
            # Lookup the limit for parent 1 and parent 2
            alpha_i = alpha_grid[d1_act, h_act, i_act]
            alpha_j = alpha_grid[d2_act, h_act, j_act]
            
            # The collision is scaled back by the most heavily depleted parent
            safe_scaling = np.minimum(alpha_i, alpha_j)
            
            # --- APPLY SCALING TO EFFICIENCIES ---
            self.params['Eagg'] = orig_Eagg * safe_scaling
            self.params['Ebr']  = orig_Ebr * safe_scaling
            
            # Run the kernel ONE MORE TIME with the scaled efficiencies
            M_loss_s, M_gain_s, N_loss_s, N_gain_s = vectorized_2mom(
                self.params, self.w, self.L, self.dMb_gain_kernel, self.dNb_gain_kernel,
                self.indc, self.indb, self.dnum, 
                self.Hlen, self.bins
            )
            
            # Restore original efficiencies to keep your object state clean
            self.params['Eagg'] = orig_Eagg
            self.params['Ebr']  = orig_Ebr
            
            # SLOW PATH: 2 kernel calls, mathematically guaranteed positive.
            M_net = dt * (M_gain_s - M_loss_s)
            N_net = dt * (N_gain_s - N_loss_s)
            
            return M_net, N_net
  
    def interact_2mom_SS_Final(self, dt):
        """Main method for the Interaction class."""
        
        # Get dictionary of dynamic parameters as 
        # multidimensional tensors
        self.get_dynamic_params()
        
        n_active = len(self.params['regions'])
    
        if n_active>10000:
            nb.set_num_threads(nb.config.NUMBA_DEFAULT_NUM_THREADS)
        else:
            nb.set_num_threads(1)
               
        M_loss, M_gain, N_loss, N_gain = vectorized_2mom(
            self.params, self.w, self.L, self.dMb_gain_kernel, self.dNb_gain_kernel,
            self.indc, self.indb, self.dnum, 
            self.Hlen,self.bins)

        M_net = dt*(M_gain-M_loss)
        N_net = dt*(N_gain-N_loss)

        return M_net, N_net


    






### OLD FUNCIONS ###


    # def interact_2mom_SS_NEW(self, dt):

    #     # Ndists x height x bins
    #     Mbins_old = self.Mbins.copy() 
    #     Nbins_old = self.Nbins.copy()
        
    #     Mbins = np.zeros_like(Mbins_old)
    #     Nbins = np.zeros_like(Nbins_old)

    #     M_loss = np.zeros_like(Mbins)
    #     N_loss = np.zeros_like(Nbins) 
        
    #     M_gain = np.zeros_like(Mbins)
    #     N_gain = np.zeros_like(Nbins)
        
    #     indc = self.indc
    #     indb = self.indb
        
    #     dd = 0
        
    #     # Loop over (d1,d2) interaction pairs
    #     for meta in self.pair_metadata:
    #         d1, d2 = meta['d1'], meta['d2']
    #         s = meta['slice']
            
    #         # --- OPTIMIZATION: One vectorized update per d-pair ---
    #         # Fetch indices from our pre-baked static buffer
    #         kr = self.static_full['kr'][s]
    #         ir = self.static_full['ir'][s]
    #         jr = self.static_full['jr'][s]
            
    #         # Mcheck = (height x d1 bins x d2 bins)
    #         Mcheck = ((self.Mbins[d1,:,:]>0.)[:,:,None]) & ((self.Mbins[d2,:,:]>0.)[:,None,:]) # If M1 or M2 is zero, do not include in bin-pair list.

    #         #cond_1 = self.cond_1 | Mcheck # New cond_1. Basically exclude bin-pairs that are off grid and ones involving empty bins.
            
    #         # Get 3D indices for all bins that will interact at all heights
    #         # indices have shape (bin-pairs,) where ir are dist1 indices 
    #         # and jr are dist2 indices. kr are height indices.
    #         #kr, ir, jr = np.nonzero((~cond_1)&self.self_col[:,dd,:,:])
                       
    #         # Calculate ck12 only for active interactions
    #         # This is a contiguous operation now
    #         #self.ck12_dynamic[s] = self.cki[d1, kr, ir] * self.cki[d2, kr, jr]
                     
    #         active_indices = np.where(Mcheck[s])[0]

    #         if len(active_indices) == 0:
    #             dd += 1
    #             continue
            
            
    #                     # 3. EXTRACTION (The I/O Buffer)
    #         # Pull from memmap into contiguous RAM blocks
    #         kr_curr = kr[active_indices]
    #         ir_curr = ir[active_indices]
    #         jr_curr = kr[active_indices]
    #         kmin_curr = np.array(self.static_full['kmin'][s][active_indices])
    #         kmid_curr = np.array(self.static_full['kmid'][s][active_indices])
    #         dMb_gain_frac_curr = np.array(self.dMb_gain_frac[:, kmin_curr])
            
            
    #         # Slice buffers for this specific interaction set
    #         # Using the absolute slice 's' + the sub-filter 'active_indices'
    #         current_static = self.static_full[s][active_indices]
            
    #         # BATCHES
    #         # (bin-pair,)
    #         x11  = self.x1[d1,kr,ir]
    #         x21  = self.x2[d1,kr,ir]
    #         ak1  = self.aki[d1,kr,ir]
    #         ck1  = self.cki[d1,kr,ir]

    #         # (bin-pair,)
    #         x12  = self.x1[d2,kr,jr]
    #         x22  = self.x2[d2,kr,jr]
    #         ak2  = self.aki[d2,kr,jr]
    #         ck2  = self.cki[d2,kr,jr] 
            
    #         kmin = self.kmin[ir,jr]
    #         kmid = self.kmid[ir,jr]
    #         xk_min = self.xi2[kmin]
            
    #         PK = self.PK[:,dd,ir,jr]
                 
    #         # SETUP REGIONS
    #         inds, x_bottom_edge, x_top_edge, y_left_edge, y_right_edge = setup_regions(self.bins,kr,ir,jr,x11,x21,x12,x22,xk_min)
        
        
    
    #         if self.parallel:
    #             # Chunking logic for Parallel
    #             batches = np.array_split(np.arange(len(current_static)), self.n_jobs)

    #             # gain_loss_temp = sum_2mom_batches(
    #             #                   kr_curr[batches[0]],ir_curr[batches[0]],jr_curr[batches[0]],ck12_curr[batches[0]],
    #             #                   kmin_curr[batches[0]],kmid_curr[batches[0]],dMi_loss_curr[batches[0]],dMj_loss_curr[batches[0]],
    #             #                   dM_loss_curr[batches[0]],dM_gain_curr[batches[0]],dMb_gain_frac_curr[:,batches[0]],dd,
    #             #                   self.Hlen,self.bins,breakup=self.breakup)
                         
    #             # M1_loss_temp = np.sum(np.vstack([gl[0] for gl in gain_loss_temp]),axis=0)
    #             # M2_loss_temp = np.sum(np.vstack([gl[1] for gl in gain_loss_temp]),axis=0)
    #             # M_gain_temp =  np.sum(np.vstack([gl[2] for gl in gain_loss_temp]),axis=0)
    #             # Mb_gain_temp = np.sum(np.vstack([gl[3] for gl in gain_loss_temp]),axis=0)
    
    #         else:

    #             # Calculate transfer rates
    #             dMi_loss, dMj_loss, dM_loss, dM_gain, dNi_loss, dN_gain = calculate_rates(self.Hlen,self.bins,inds,x_bottom_edge,x_top_edge,y_left_edge,y_right_edge,
    #             kr,ir,jr,x11,x21,ak1,ck1,x12,x22,ak2,ck2,PK,
    #             kmin,kmid,self.dMb_gain_frac,self.dNb_gain_frac,self.w,self.L,
    #             breakup=self.breakup)

    #             # Perform transfer/assignment to bins
    #             M1_loss_temp, M2_loss_temp, M_gain_temp, Mb_gain_temp,\
    #             N1_loss_temp, N2_loss_temp, N_gain_temp, Nb_gain_temp = transfer_bins(self.Hlen,self.bins,kr,ir,jr,kmin,kmid,dMi_loss,dMj_loss,dM_loss,dM_gain,
    #                               dNi_loss,dN_gain,self.dMb_gain_frac,self.dNb_gain_frac,breakup=self.breakup)
    
    
    #         M_loss[d1,:,:]    += M1_loss_temp 
    #         M_loss[d2,:,:]    += M2_loss_temp
            
    #         M_gain[indc,:,:]  += self.Eagg*M_gain_temp
    #         M_gain[indb,:,:]  += self.Ebr*Mb_gain_temp
            
    #         N_loss[d1,:,:]    += N1_loss_temp
    #         N_loss[d2,:,:]    += N2_loss_temp
             
    #         N_gain[indc,:,:]  += self.Eagg*N_gain_temp
    #         N_gain[indb,:,:]  += self.Ebr*Nb_gain_temp
            
    #         dd += 1
    
    
    #     M_loss *= self.Ecb
    #     N_loss *= self.Ecb 
        
    #     M_net = dt*(M_gain-M_loss) 
    #     N_net = dt*(N_gain-N_loss)
        
    #     return M_net, N_net    


# def calculate_regions(x11,x21,ak1,ck1,x12,x22,ak2,ck2,PK,xi1,xi2,regions):

#     Hlen, bins = np.shape(x11)

#     '''
#     Vectorized Integration Regions:
#     cond_1 :  Ignore CC process for these source bins; they don't map to the largest avail bin.
#     cond_2 :  k bin: Lower triangle region. Just clips BR corner.
#                        Triangle = ((xi1,xj1),(xi1,y_left_edge),(xi2,xj1))
#     cond_3 :  k bin: Lower triangle region. Just clips UL corner.
#                        Triangle = ((xi1,xj1),(xi1,xj2),(x_bottom_edge,xj1))                 
#     cond_2b : k+1 bins: Upper triangle region. Just clips BR corner.
#                        Triangle = ((xi1,xj1),(x_top_edge,xj2),(xi2,xj2))
#     cond_3b : k+1 bin: Upper triangle region. Just clips UL corner.
#                              Triangle = ((xi1,xj2),(xi2,xj2),(xi2,y_right_edge)                    
#     cond_4 :  Full Rectangular source region based on self collection: ii == jj --> ii+sbin or jj+sbin
#     cond_5 :  k bin: Top/Bottom clip: Rectangle on left, triangle on right
#                               Rectangle = ((xi1,xj1),(xi1,xj2),(x_top_edge,xj2),(x_top_edge,xj1))
#                               Triangle  = ((x_top_edge,xj1),(x_top_edge,xj2),(x_bottom_edge,xj1))
#     cond_6 :  k bin: Left/Right clip: Rectangle on bottom, triangle on top
#                               Rectangle = ((xi1,xj1),(xi1,y_right_edge),(xi2,y_right_edge),(xi2,xj1))
#                               Triangle  = ((xi1,y_right_edge),(xi1,y_left_edge),(xi2,y_right_edge))
#     cond_7 :  k+1 bin: Triangle in top right corner
#                                 Triangle = ((x_top_edge,xj2),(xi2,xj2),(xi2,y_right_edge))
#     cond_8 :  k bin: Triangle in lower left corner
#                                 Triangle = ((xi1,xj1),(xi1,y_left_edge),(x_bottom_edge,xj1))
#     cond_9:   k bin: Rectangle collection within k bin. All Mass/Number goes into kbin
#     cond_10:  k+1 bin: Rectangle collection within k+1 bin. All Mass/Number goes into kbin
#     '''
    
#     x_bottom_edge = regions['x_bottom_edge']
#     x_top_edge = regions['x_top_edge']
#     y_left_edge = regions['y_left_edge']
#     y_right_edge = regions['y_right_edge']
#     i1 = regions['1']['i']
#     j1 = regions['1']['j']
#     k1 = regions['1']['k']
#     i2 = regions['2']['i']
#     j2 = regions['2']['j']
#     k2 = regions['2']['k']
#     i2b = regions['2b']['i']
#     j2b = regions['2b']['j']
#     k2b = regions['2b']['k']
#     i3 = regions['3']['i']
#     j3 = regions['3']['j']
#     k3 = regions['3']['k']
#     i3b = regions['3b']['i']
#     j3b = regions['3b']['j']
#     k3b = regions['3b']['k']
#     i4 = regions['4']['i']
#     j4 = regions['4']['j']
#     k4 = regions['4']['k']
#     i5 = regions['5']['i']
#     j5 = regions['5']['j']
#     k5 = regions['5']['k']
#     i6 = regions['6']['i']
#     j6 = regions['6']['j']
#     k6 = regions['6']['k']
#     i7 = regions['7']['i']
#     j7 = regions['7']['j']
#     k7 = regions['7']['k']
#     i8 = regions['8']['i']
#     j8 = regions['8']['j']
#     k8 = regions['8']['k']
#     i9 = regions['9']['i']
#     j9 = regions['9']['j']
#     k9 = regions['9']['k']
#     i10 = regions['10']['i']
#     j10 = regions['10']['j']
#     k10 = regions['10']['k']
    
#     # Initialize gain term arrays
#     dMi_loss = np.zeros((Hlen,bins,bins))
#     dMj_loss = np.zeros((Hlen,bins,bins))
#     dNi_loss = np.zeros((Hlen,bins,bins))
#     dM_gain  = np.zeros((Hlen,bins,bins,2))
#     dN_gain  = np.zeros((Hlen,bins,bins,2))
  
#     # Calculate transfer rates (rectangular integration, source space)
#     # Collection (eqs. 23-25 in Wang et al. 2007)
#     # ii collecting jj 

#     dMi_loss[k1,i1,j1] = integrate_fast_kernel(1,0,0,PK[:,i1,j1],ak1[k1,i1],ck1[k1,i1],ak2[k1,j1],ck2[k1,j1],'rectangle',x1=x11[k1,i1],x2=x21[k1,i1],y1=x12[k1,j1],y2=x22[k1,j1])
#     dMj_loss[k1,i1,j1] = integrate_fast_kernel(0,1,0,PK[:,i1,j1],ak1[k1,i1],ck1[k1,i1],ak2[k1,j1],ck2[k1,j1],'rectangle',x1=x11[k1,i1],x2=x21[k1,i1],y1=x12[k1,j1],y2=x22[k1,j1])
#     dNi_loss[k1,i1,j1] = integrate_fast_kernel(0,0,0,PK[:,i1,j1],ak1[k1,i1],ck1[k1,i1],ak2[k1,j1],ck2[k1,j1],'rectangle',x1=x11[k1,i1],x2=x21[k1,i1],y1=x12[k1,j1],y2=x22[k1,j1])
#     #dNj_loss = dNi_loss.copy() # Nj loss should be same as Ni loss
    
#     # Condition 4: Self collection. All Mass/Number goes into ii+sbin = jj+sbin kbin
#     xi1 = x11[k4,i4].copy()
#     xi2 = x21[k4,i4].copy() 
#     xj1 = x12[k4,j4].copy()
#     xj2 = x22[k4,j4].copy()
    
#     dM_gain[k4,i4,j4,0]  = integrate_fast_kernel(0,0,1,PK[:,i4,j4],ak1[k4,i4],ck1[k4,i4],ak2[k4,j4],ck2[k4,j4],'rectangle',x1=xi1,x2=xi2,y1=xj1,y2=xj2) 
#     dN_gain[k4,i4,j4,0]  = integrate_fast_kernel(0,0,0,PK[:,i4,j4],ak1[k4,i4],ck1[k4,i4],ak2[k4,j4],ck2[k4,j4],'rectangle',x1=xi1,x2=xi2,y1=xj1,y2=xj2) 


#     # Condition 2:
#     # k bin: Lower triangle region. Just clips BR corner.
#     #                       Triangle = ((xi1,xj1),(xi1,y_left_edge),(xi2,xj1))
#     xt1 = x11[k2,i2].copy()
#     yt1 = x12[k2,j2].copy()
#     xt2 = x11[k2,i2].copy()
#     yt2 = y_left_edge[k2,i2,j2].copy()
#     xt3 = x21[k2,i2].copy()
#     yt3 = x12[k2,j2].copy()
    
#     dM_gain[k2,i2,j2,0] = integrate_fast_kernel(0,0,1,PK[:,i2,j2],ak1[k2,i2],ck1[k2,i2],ak2[k2,j2],ck2[k2,j2],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
#     dM_gain[k2,i2,j2,1] = (dMi_loss[k2,i2,j2]+dMj_loss[k2,i2,j2])-dM_gain[k2,i2,j2,0]
    
#     dN_gain[k2,i2,j2,0] = integrate_fast_kernel(0,0,0,PK[:,i2,j2],ak1[k2,i2],ck1[k2,i2],ak2[k2,j2],ck2[k2,j2],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
#     dN_gain[k2,i2,j2,1] = (dNi_loss[k2,i2,j2])-dN_gain[k2,i2,j2,0]
        
#     # Condition 3:
#     #    k bin: Lower triangle region. Just clips UL corner.
#     #                      Triangle = ((xi1,xj1),(xi1,xj2),(x_bottom_edge,xj1))  
#     xt1 = x11[k3,i3].copy()
#     yt1 = x12[k3,j3].copy()
#     xt2 = x11[k3,i3].copy()
#     yt2 = x22[k3,j3].copy()
#     xt3 = x_bottom_edge[k3,i3,j3].copy()
#     yt3 = x12[k3,j3].copy()
    
#     dM_gain[k3,i3,j3,0] = integrate_fast_kernel(0,0,1,PK[:,i3,j3],ak1[k3,i3],ck1[k3,i3],ak2[k3,j3],ck2[k3,j3],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
#     dM_gain[k3,i3,j3,1] = (dMi_loss[k3,i3,j3]+dMj_loss[k3,i3,j3])-dM_gain[k3,i3,j3,0]
        
#     dN_gain[k3,i3,j3,0] = integrate_fast_kernel(0,0,0,PK[:,i3,j3],ak1[k3,i3],ck1[k3,i3],ak2[k3,j3],ck2[k3,j3],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
#     dN_gain[k3,i3,j3,1] = (dNi_loss[k3,i3,j3])-dN_gain[k3,i3,j3,0]
        
#     # Condition 5: 
        
#     #    k bin: Top/Bottom clip: Rectangle on left, triangle on right
#     #                              Rectangle = ((xi1,xj1),(xi1,xj2),(x_top_edge,xj2),(x_top_edge,xj1))
#     #                              Triangle  = ((x_top_edge,xj1),(x_top_edge,xj2),(x_bottom_edge,xj1))
   
#     xr1 = x11[k5,i5].copy()
#     yr1 = x12[k5,j5].copy()
#     xr2 = x_top_edge[k5,i5,j5].copy()
#     yr2 = x22[k5,j5].copy()
   
#     xt1 = x_top_edge[k5,i5,j5].copy()
#     yt1 = x12[k5,j5].copy()
#     xt2 = x_top_edge[k5,i5,j5].copy()
#     yt2 = x22[k5,j5].copy()
#     xt3 = x_bottom_edge[k5,i5,j5].copy()
#     yt3 = x12[k5,j5].copy()
    
    
#     dM_gain[k5,i5,j5,0] = integrate_fast_kernel(0,0,1,PK[:,i5,j5],ak1[k5,i5],ck1[k5,i5],ak2[k5,j5],ck2[k5,j5],'rectangle',x1=xr1,x2=xr2,y1=yr1,y2=yr2)+\
#                           integrate_fast_kernel(0,0,1,PK[:,i5,j5],ak1[k5,i5],ck1[k5,i5],ak2[k5,j5],ck2[k5,j5],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
    
#     dM_gain[k5,i5,j5,1] = (dMi_loss[k5,i5,j5]+dMj_loss[k5,i5,j5])-dM_gain[k5,i5,j5,0]
        
#     dN_gain[k5,i5,j5,0] = integrate_fast_kernel(0,0,0,PK[:,i5,j5],ak1[k5,i5],ck1[k5,i5],ak2[k5,j5],ck2[k5,j5],'rectangle',x1=xr1,x2=xr2,y1=yr1,y2=yr2)+\
#                           integrate_fast_kernel(0,0,0,PK[:,i5,j5],ak1[k5,i5],ck1[k5,i5],ak2[k5,j5],ck2[k5,j5],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
                       
#     dN_gain[k5,i5,j5,1] = (dNi_loss[k5,i5,j5])-dN_gain[k5,i5,j5,0]
    
    
#     # Condition 6:
#     # k bin: Left/Right clip: Rectangle on bottom, triangle on top
#     #                          Rectangle = ((xi1,xj1),(xi1,y_right_edge),(xi2,y_right_edge),(xi2,xj1))
#     #                          Triangle  = ((xi1,y_right_edge),(xi1,y_left_edge),(xi2,y_right_edge))
        
#     xr1 = x11[k6,i6].copy()
#     yr1 = x12[k6,j6].copy()
#     xr2 = x21[k6,i6].copy()
#     yr2 = y_right_edge[k6,i6,j6].copy()
    
#     xt1 = x11[k6,i6].copy()
#     yt1 = y_right_edge[k6,i6,j6].copy()
#     xt2 = x11[k6,i6].copy()
#     yt2 = y_left_edge[k6,i6,j6].copy()
#     xt3 = x21[k6,i6].copy()
#     yt3 = y_right_edge[k6,i6,j6].copy()
    
    
#     dM_gain[k6,i6,j6,0] = integrate_fast_kernel(0,0,1,PK[:,i6,j6],ak1[k6,i6],ck1[k6,i6],ak2[k6,j6],ck2[k6,j6],'rectangle',x1=xr1,x2=xr2,y1=yr1,y2=yr2)+\
#                           integrate_fast_kernel(0,0,1,PK[:,i6,j6],ak1[k6,i6],ck1[k6,i6],ak2[k6,j6],ck2[k6,j6],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
    
#     dM_gain[k6,i6,j6,1] = (dMi_loss[k6,i6,j6]+dMj_loss[k6,i6,j6])-dM_gain[k6,i6,j6,0]
        
#     dN_gain[k6,i6,j6,0] = integrate_fast_kernel(0,0,0,PK[:,i6,j6],ak1[k6,i6],ck1[k6,i6],ak2[k6,j6],ck2[k6,j6],'rectangle',x1=xr1,x2=xr2,y1=yr1,y2=yr2)+\
#                           integrate_fast_kernel(0,0,0,PK[:,i6,j6],ak1[k6,i6],ck1[k6,i6],ak2[k6,j6],ck2[k6,j6],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
                       
#     dN_gain[k6,i6,j6,1] = (dNi_loss[k6,i6,j6])-dN_gain[k6,i6,j6,0]
        
#     # Condition 7:
#     # k+1 bin: Triangle in top right corner
#     #                            Triangle = ((x_top_edge,xj2),(xi2,xj2),(xi2,y_right_edge))
        
#     xt1 = x_top_edge[k7,i7,j7].copy()
#     yt1 = x22[k7,j7].copy()
#     xt2 = x21[k7,i7].copy()
#     yt2 = x22[k7,j7].copy()
#     xt3 = x21[k7,i7].copy()
#     yt3 = y_right_edge[k7,i7,j7].copy()
    
#     dM_gain[k7,i7,j7,1] = integrate_fast_kernel(0,0,1,PK[:,i7,j7],ak1[k7,i7],ck1[k7,i7],ak2[k7,j7],ck2[k7,j7],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
#     dM_gain[k7,i7,j7,0] = (dMi_loss[k7,i7,j7]+dMj_loss[k7,i7,j7])-dM_gain[k7,i7,j7,1]
    
#     dN_gain[k7,i7,j7,1] = integrate_fast_kernel(0,0,0,PK[:,i7,j7],ak1[k7,i7],ck1[k7,i7],ak2[k7,j7],ck2[k7,j7],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
#     dN_gain[k7,i7,j7,0] = (dNi_loss[k7,i7,j7])-dN_gain[k7,i7,j7,1]
    
        
#     # Condition 8:
#     #  k bin: Triangle in lower left corner
#     #                                Triangle = ((xi1,xj1),(xi1,y_left_edge),(x_bottom_edge,xj1))
#     xt1 = x11[k8,i8].copy()
#     yt1 = x12[k8,j8].copy()
#     xt2 = x11[k8,i8].copy()
#     yt2 = y_left_edge[k8,i8,j8].copy()
#     xt3 = x_bottom_edge[k8,i8,j8].copy()
#     yt3 = x12[k8,j8].copy()
    
#     dM_gain[k8,i8,j8,0] = integrate_fast_kernel(0,0,1,PK[:,i8,j8],ak1[k8,i8],ck1[k8,i8],ak2[k8,j8],ck2[k8,j8],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
#     dM_gain[k8,i8,j8,1] = (dMi_loss[k8,i8,j8]+dMj_loss[k8,i8,j8])-dM_gain[k8,i8,j8,0]
    
#     dN_gain[k8,i8,j8,0] = integrate_fast_kernel(0,0,0,PK[:,i8,j8],ak1[k8,i8],ck1[k8,i8],ak2[k8,j8],ck2[k8,j8],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
#     dN_gain[k8,i8,j8,1] = (dNi_loss[k8,i8,j8])-dN_gain[k8,i8,j8,0]
        
#     # Condition 9: Rectangle collection within k bin. All Mass/Number goes into kbin
#     xi1 = x11[k9,i9].copy()
#     xi2 = x21[k9,i9].copy() 
#     xj1 = x12[k9,j9].copy()
#     xj2 = x22[k9,j9].copy()

#     dM_gain[k9,i9,j9,0]  = integrate_fast_kernel(0,0,1,PK[:,i9,j9],ak1[k9,i9],ck1[k9,i9],ak2[k9,j9],ck2[k9,j9],'rectangle',x1=xi1,x2=xi2,y1=xj1,y2=xj2) 
#     dN_gain[k9,i9,j9,0]  = integrate_fast_kernel(0,0,0,PK[:,i9,j9],ak1[k9,i9],ck1[k9,i9],ak2[k9,j9],ck2[k9,j9],'rectangle',x1=xi1,x2=xi2,y1=xj1,y2=xj2) 

#     # Condition 10: Rectangle collection within k+1 bin. All Mass/Number goes into kbin
#     xi1 = x11[k10,i10].copy()
#     xi2 = x21[k10,i10].copy() 
#     xj1 = x12[k10,j10].copy()
#     xj2 = x22[k10,j10].copy()
    
#     dM_gain[k10,i10,j10,1]  = integrate_fast_kernel(0,0,1,PK[:,i10,j10],ak1[k10,i10],ck1[k10,i10],ak2[k10,j10],ck2[k10,j10],'rectangle',x1=xi1,x2=xi2,y1=xj1,y2=xj2) 
#     dN_gain[k10,i10,j10,1]  = integrate_fast_kernel(0,0,0,PK[:,i10,j10],ak1[k10,i10],ck1[k10,i10],ak2[k10,j10],ck2[k10,j10],'rectangle',x1=xi1,x2=xi2,y1=xj1,y2=xj2) 
   
#     # Condition 11 (2b): Triangle collection within k+1 bin. Occurs when xk+1 clips BR corner and intersects top edge.
#     # Triangle = ((xi2,xj1),(x_top_edge,xj2),(xi2,xj2))
#     xt1 = x_top_edge[k2b,i2b,j2b].copy()
#     yt1 = x22[k2b,j2b].copy()
#     xt2 = x21[k2b,i2b].copy()
#     yt2 = x22[k2b,j2b].copy()
#     xt3 = x21[k2b,i2b].copy()
#     yt3 = x12[k2b,j2b].copy()
    
#     dM_gain[k2b,i2b,j2b,1] = integrate_fast_kernel(0,0,1,PK[:,i2b,j2b],ak1[k2b,i2b],ck1[k2b,i2b],ak2[k2b,j2b],ck2[k2b,j2b],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
#     dM_gain[k2b,i2b,j2b,0] = (dMi_loss[k2b,i2b,j2b]+dMj_loss[k2b,i2b,j2b])-dM_gain[k2b,i2b,j2b,1]
    
#     dN_gain[k2b,i2b,j2b,1] = integrate_fast_kernel(0,0,0,PK[:,i2b,j2b],ak1[k2b,i2b],ck1[k2b,i2b],ak2[k2b,j2b],ck2[k2b,j2b],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
#     dN_gain[k2b,i2b,j2b,0] = (dNi_loss[k2b,i2b,j2b])-dN_gain[k2b,i2b,j2b,1]
    
#     # Condition 12 (3b): Triangle collection within k+1 bin. Occurs when xk+1 clips UL corner and intersects with right edge.
#     # Triangle = ((xi1,xj2),(xi2,xj2),(xi2,y_right_edge)
#     xt1 = x11[k3b,i3b].copy()
#     yt1 = x22[k3b,j3b].copy()
#     xt2 = x21[k3b,i3b].copy()
#     yt2 = x22[k3b,j3b].copy()
#     xt3 = x21[k3b,i3b].copy()
#     yt3 = y_right_edge[k3b,i3b,j3b].copy()
    
#     dM_gain[k3b,i3b,j3b,1] = integrate_fast_kernel(0,0,1,PK[:,i3b,j3b],ak1[k3b,i3b],ck1[k3b,i3b],ak2[k3b,j3b],ck2[k3b,j3b],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
#     dM_gain[k3b,i3b,j3b,0] = (dMi_loss[k3b,i3b,j3b]+dMj_loss[k3b,i3b,j3b])-dM_gain[k3b,i3b,j3b,1]
    
#     dN_gain[k3b,i3b,j3b,1] = integrate_fast_kernel(0,0,0,PK[:,i3b,j3b],ak1[k3b,i3b],ck1[k3b,i3b],ak2[k3b,j3b],ck2[k3b,j3b],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
#     dN_gain[k3b,i3b,j3b,0] = (dNi_loss[k3b,i3b,j3b])-dN_gain[k3b,i3b,j3b,1]   
    
#     return dMi_loss, dMj_loss, dM_gain, dNi_loss, dN_gain


    # # Advance PSD Mbins and Nbins by one time/height step
    # def interact_2mom(self,dt):

    #     # Ndists x height x bins
    #     Mbins_old = self.Mbins.copy() 
    #     Nbins_old = self.Nbins.copy()
        
    #     Mbins = np.zeros_like(Mbins_old)
    #     Nbins = np.zeros_like(Nbins_old)

    #     M_loss = np.zeros_like(Mbins)
    #     N_loss = np.zeros_like(Nbins) 
        
    #     M_gain = np.zeros_like(Mbins)
    #     N_gain = np.zeros_like(Nbins)
        
    #     indc = self.indc
    #     indb = self.indb
        
    #     dd = 0
    #     for d1 in range(self.dnum):
    #         for d2 in range(d1,self.dnum):
                
    #             # Set up batches for each bin-bin pair? Would need to unravel
    #             # arrays appropriately afterward. Also would need to figure out 
    #             # how to do self and cross collection appropriately.
                 
    #             # Find all bin pairs
                 
    #         #     # OLD
    #         #     # (dnum x height x bins)
    #         #     x11  = self.x1[d1,:,:]
    #         #     x21  = self.x2[d1,:,:]
    #         #     ak1  = self.aki[d1,:,:]
    #         #     ck1  = self.cki[d1,:,:] 
    #         # # M1   = self.Mbins[d1,:,:]
                 
    #         #     x12  = self.x1[d2,:,:]
    #         #     x22  = self.x2[d2,:,:]
    #         #     ak2  = self.aki[d2,:,:] 
    #         #     ck2  = self.cki[d2,:,:] 
    #         #     #M2   = self.Mbins[d2,:,:]
                
  
    #             # Mcheck = (height x d1 bins x d2 bins)
    #             Mcheck = ((self.Mbins[d1,:,:]==0.)[:,:,None]) | ((self.Mbins[d2,:,:]==0.)[:,None,:]) # If M1 or M2 is zero, do not include in bin-pair list.
                
    #             cond_1 = self.cond_1 | Mcheck # New cond_1. Basically exclude bin-pairs that are off grid and ones involving empty bins.
                              
    #             # Get 3D indices for all bins that will interact at all heights
    #             # indices have shape (bin-pairs,) where ir are dist1 indices 
    #             # and jr are dist2 indices. kr are height indices.
    #             kr, ir, jr = np.nonzero((~cond_1)&self.self_col[:,dd,:,:])
                
    #             # (bin-pair,)
    #             x11  = self.x1[d1,kr,ir]
    #             x21  = self.x2[d1,kr,ir]
    #             ak1  = self.aki[d1,kr,ir]
    #             ck1  = self.cki[d1,kr,ir]
    #             #xi11 = self.xi1[ir]
    #             #xi21 = self.xi2[ir]

    #             # (bin-pair,)
    #             x12  = self.x1[d2,kr,jr]
    #             x22  = self.x2[d2,kr,jr]
    #             ak2  = self.aki[d2,kr,jr]
    #             ck2  = self.cki[d2,kr,jr] 
    #             #xi12  = self.xi1[jr]
    #            # xi22  = self.xi2[jr]
                
    #             xk_min = self.xi2[self.kmin[ir,jr]]
                
    #             # SETUP REGIONS
                

    #             if self.parallel:
                         
    #                 #
    #                 #Mcheck = ((M1==0.)[:,None]) | ((M2==0.)[None,:])
    #                 #ikeep = np.nonzero(~(self.cond_1&Mcheck))
                    
    #                 #Mcheck = ((M1==0.)[:,:,None]) | ((M2==0.)[:,None,:])
                    
    #                 #cond_1 = self.cond_1 | Mcheck # New cond_1. Basically exclude bin-pairs that are off grid and ones involving empty bins.
                    

                    
    #                 # Sort regions
                    
    #                 all_slices = zip(
    #                    np.array_split(kr, self.n_jobs),
    #                    np.array_split(ir, self.n_jobs),
    #                    np.array_split(jr, self.n_jobs),
    #                    np.array_split(x11, self.n_jobs),
    #                    np.array_split(x21, self.n_jobs),
    #                    np.array_split(ak1, self.n_jobs),
    #                    np.array_split(ck1, self.n_jobs),
    #                    np.array_split(x12, self.n_jobs),
    #                    np.array_split(x22, self.n_jobs),
    #                    np.array_split(ak2, self.n_jobs),
    #                    np.array_split(ck2, self.n_jobs),
    #                    #np.array_split(self.PK[:,dd,ir,jr],self.n_jobs,axis=1),
    #                   # np.array_split(self.xk_min[ir,jr],self.n_jobs)
    #                     )
                    
                    
    #                 # gain_loss_temp = Parallel(n_jobs=self.n_jobs,verbose=0)(delayed(calculate_2mom)(x11[batch,:],x21[batch,:],ak1[batch,:],ck1[batch,:],M1[batch,:],
    #                 #                         x12[batch,:],x22[batch,:],ak2[batch,:],ck2[batch,:],M2[batch,:],
    #                 #                         self.PK[:,d1,d2,:,:],self.xi1,self.xi2,self.kmin,self.kmid,self.cond_1[batch,:,:], 
    #                 #                         self.dMb_gain_frac,self.dNb_gain_frac, 
    #                 #                         self.self_col[batch,dd,:,:],breakup=self.breakup) for batch in self.batches)
                           
    #                 self.pool(delayed(sum_2mom_batches)(
    #                                        ii,*slices,self.mem_map_dict,dd,self.Hlen,self.bins,breakup=self.breakup) for slices,ii in zip(all_slices,range(self.n_jobs)))
                       
    #                 # with Parallel(n_jobs=self.n_jobs,verbose=0) as par:
                        
    #                 #     gain_loss_temp = par(delayed(calculate_2mom)(x11[batch,:],x21[batch,:],ak1[batch,:],ck1[batch,:],M1[batch,:],
    #                 #                             x12[batch,:],x22[batch,:],ak2[batch,:],ck2[batch,:],M2[batch,:],
    #                 #                             self.PK[:,dd,:,:],self.xi1,self.xi2,self.kmin,self.kmid,self.cond_1[batch,:,:], 
    #                 #                             self.dMb_gain_frac,self.dNb_gain_frac, 
    #                 #                             self.self_col[batch,dd,:,:],breakup=self.breakup) for batch in self.batches)
                        
                    
    #                 # gain_loss_temp = Parallel(n_jobs=self.n_jobs,verbose=0)(delayed(calculate_2mom)(x11[batch,:],x21[batch,:],ak1[batch,:],ck1[batch,:],M1[batch,:],
    #                 #                         x12[batch,:],x22[batch,:],ak2[batch,:],ck2[batch,:],M2[batch,:],
    #                 #                         self.PK[:,dd,:,:],self.xi1,self.xi2,self.kmin,self.kmid,self.cond_1[batch,:,:], 
    #                 #                         self.dMb_gain_frac,self.dNb_gain_frac, 
    #                 #                         self.self_col[batch,dd,:,:],breakup=self.breakup) for batch in self.batches)
                    
                    
    #                 gl_tot = np.nansum(self.mem_out,axis=0)
                    
    #                 M1_loss_temp = gl_tot[0,:,:]
    #                 M2_loss_temp = gl_tot[1,:,:]
    #                 M_gain_temp  = gl_tot[2,:,:]
    #                 Mb_gain_temp = gl_tot[3,:,:]
    #                 N1_loss_temp = gl_tot[4,:,:]
    #                 N2_loss_temp = gl_tot[5,:,:]
    #                 N_gain_temp  = gl_tot[6,:,:]
    #                 Nb_gain_temp = gl_tot[7,:,:]
                    
                    
    #                 # OLD
    #                 # M1_loss_temp = np.vstack([gl[0] for gl in gain_loss_temp])
    #                 # M2_loss_temp = np.vstack([gl[1] for gl in gain_loss_temp])
    #                 # M_gain_temp =  np.vstack([gl[2] for gl in gain_loss_temp])
    #                 # Mb_gain_temp = np.vstack([gl[3] for gl in gain_loss_temp])
    #                 # N1_loss_temp = np.vstack([gl[4] for gl in gain_loss_temp])
    #                 # N2_loss_temp = np.vstack([gl[5] for gl in gain_loss_temp])
    #                 # N_gain_temp  = np.vstack([gl[6] for gl in gain_loss_temp])
    #                 # Nb_gain_temp = np.vstack([gl[7] for gl in gain_loss_temp])
            
    #             else:

    #                 # OLD
    #                 # x11  = self.x1[d1,:]
    #                 # x21  = self.x2[d1,:]
    #                 # ak1  = self.aki[d1,:]
    #                 # ck1  = self.cki[d1,:] 
    #                 # #M1   = self.Mbins[d1,:]
                    
    #                 # x12  = self.x1[d2,:]
    #                 # x22  = self.x2[d2,:]
    #                 # ak2  = self.aki[d2,:] 
    #                 # ck2  = self.cki[d2,:] 
    #                 #M2   = self.Mbins[d2,:]
                    
    #                 # Calculate bin-source pairs
                    
    #                # Mcheck = ((M1==0.)[:,:,None]) | ((M2==0.)[:,None,:])
                    
    #                 #cond_1 = self.cond_1 | Mcheck # New cond_1. Basically exclude bin-pairs that are off grid and ones involving empty bins.
                            
    #                 # ORIGINAL method
    #                 # M1_loss_temp, M2_loss_temp, M_gain_temp, Mb_gain_temp,\
    #                 # N1_loss_temp, N2_loss_temp, N_gain_temp, Nb_gain_temp = calculate_2mom(x11,x21,ak1,ck1, 
    #                 #                x12,x22,ak2,ck2,self.PK[:,d1,d2,:,:],self.xi1,self.xi2,self.kmin,self.kmid,self.cond_1, 
    #                 #                self.dMb_gain_frac,self.dNb_gain_frac, 
    #                 #                self.self_col[:,dd,:,:],breakup=self.breakup)

                    
    #                 # NEW TRIANGLE INTEGRATION METHOD INTEGRATION
    #                 M1_loss_temp, M2_loss_temp, M_gain_temp, Mb_gain_temp,\
    #                 N1_loss_temp, N2_loss_temp, N_gain_temp, Nb_gain_temp = calculate_2mom_batch(
    #                   self.Hlen,self.bins,kr,ir,jr,x11,x21,ak1,ck1, 
    #                   x12,x22,ak2,ck2,self.PK[:,dd,ir,jr],
    #                   self.xi1,self.xi2,xk_min,self.kmin,self.kmid, 
    #                   self.dMb_gain_frac,self.dNb_gain_frac,self.w,self.L,
    #                   breakup=self.breakup)
                    
                    
    #                 # M1_loss_temp, M2_loss_temp, M_gain_temp, Mb_gain_temp,\
    #                 # N1_loss_temp, N2_loss_temp, N_gain_temp, Nb_gain_temp = calculate_2mom(x11,x21,ak1,ck1,M1, 
    #                 #                 x12,x22,ak2,ck2,M2,self.PK[:,dd,:,:],self.xi1,self.xi2,self.kmin,self.kmid,self.cond_1, 
    #                 #                 self.dMb_gain_frac,self.dNb_gain_frac, 
    #                 #                 self.self_col[:,dd,:,:],breakup=self.breakup)
                         
    #             M_loss[d1,:,:]    += M1_loss_temp 
    #             M_loss[d2,:,:]    += M2_loss_temp
                
    #             M_gain[indc,:,:]  += self.Eagg*M_gain_temp
    #             M_gain[indb,:,:]  += self.Ebr*Mb_gain_temp
                
    #             N_loss[d1,:,:]    += N1_loss_temp
    #             N_loss[d2,:,:]    += N2_loss_temp
                 
    #             N_gain[indc,:,:]  += self.Eagg*N_gain_temp
    #             N_gain[indb,:,:]  += self.Ebr*Nb_gain_temp
                
    #             dd += 1
                
    #     M_loss *= self.Ecb
    #     N_loss *= self.Ecb 
        
    #     M_net = dt*(M_gain-M_loss) 
    #     N_net = dt*(N_gain-N_loss)
        
    #     return M_net, N_net   
    
    
    
     




# #def calculate_1mom(i1,j1,n1,n2,dMi_loss,dMj_loss,dM_gain,kmin,kmid,dMb_gain_frac,breakup):
    
# def calculate_1mom(i1,j1,n12,dMi_loss,dMj_loss,dM_gain,kmin,kmid,dMb_gain_frac,breakup):
    
#     '''
#      This function calculate mass transfer rates
#      for collision-coalescence and collisional breakup between
#      each distribution for 1 moment calculations.
#     '''
    
#     #Hlen,bins = n1.shape
   
#     Hlen,bins = n12.shape[:2]
    
#    # n12 = n1[:,:,None]*n2[:,None,:]
    
#     M1_loss = np.nansum(n12*(dMi_loss[None,:,:]),axis=2) # Loss of dist1 mass with collisions from dist2
#     M2_loss = np.nansum(n12*(dMj_loss[None,:,:]),axis=1) # Loss of dist2 mass with collisions from dist1
       
#     # ChatGPT is the GOAT for telling me about np.add.at!
#     M_gain = np.zeros((Hlen,bins))
#     np.add.at(M_gain, (np.arange(Hlen)[:,None,None],kmin), n12*(dM_gain[:,:,0][None,:,:]))
#     np.add.at(M_gain,  (np.arange(Hlen)[:,None,None],kmid), n12*(dM_gain[:,:,1][None,:,:]))
    
#     # ELD NOTE: Breakup here can take losses from each pair and calculate gains
#     # for breakup. Breakup gain arrays will be 3D.
#     if breakup:
        
#         Mij_loss = n12[:,i1,j1]*(dMi_loss[i1,j1]+dMj_loss[i1,j1])[None,:]
#         Mb_gain = np.nansum((dMb_gain_frac[:,kmin[i1,j1]][None,:,:])*Mij_loss[:,None,:],axis=2)
        
#     else:
        
#         Mb_gain  = np.zeros((Hlen,bins))
        
#     return M1_loss, M2_loss, M_gain, Mb_gain   


# def calculate_2mom(x11,x21,ak1,ck1,x12,x22,ak2,ck2,
#                    PK,xi1,xi2,kmin,kmid,cond_1_orig, 
#                    dMb_gain_frac,dNb_gain_frac, 
#                    sc_inds,breakup=False):
    
#     '''
#      This function calculate mass and number transfer rates
#      for collision-coalescence and collisional breakup between
#      each distribution for 2 moment calculations (mass + number).
#     '''
    
#     Hlen,bins = x11.shape
    
#     # Set up integration regions for time-dependent subgrid linear distributions.
#     regions = setup_regions(x11,x21,ak1,ck1,x12,x22,ak2,ck2,xi1,xi2,kmin,cond_1_orig,sc_inds)
    
#     # Calculate integration regions to determine CC loss/gain integrals.
#     dMi_loss, dMj_loss, dM_gain, dNi_loss, dN_gain = calculate_regions(x11,x21,ak1,ck1,x12,x22,ak2,ck2,PK,xi1,xi2,regions)
    
#     M1_loss = np.nansum(dMi_loss,axis=2) # Loss of dist1 mass with collisions from dist2
#     N1_loss = np.nansum(dNi_loss,axis=2) # Loss of dist1 number with collisions from dist2
    
#     M2_loss = np.nansum(dMj_loss,axis=1) # Loss of dist2 mass with collisions from dist1
#     N2_loss = np.nansum(dNi_loss,axis=1) # Loss of dist2 number with collisions from dist1
    
#     # ChatGPT is the GOAT for telling me about np.add.at!
#     M_gain = np.zeros((Hlen,bins))
#     np.add.at(M_gain, (np.arange(Hlen)[:,None,None],kmin), dM_gain[:,:,:,0])
#     np.add.at(M_gain, (np.arange(Hlen)[:,None,None],kmid), dM_gain[:,:,:,1])
    
#     N_gain = np.zeros((Hlen,bins))
#     np.add.at(N_gain,  (np.arange(Hlen)[:,None,None],kmin), dN_gain[:,:,:,0])
#     np.add.at(N_gain,  (np.arange(Hlen)[:,None,None],kmid), dN_gain[:,:,:,1])
    
#     # Initialize gain term arrays
#     Mb_gain  = np.zeros((Hlen,bins))
#     Nb_gain  = np.zeros((Hlen,bins))
      
#     # ELD NOTE: Breakup here can take losses from each pair and calculate gains
#     # for breakup. Breakup gain arrays will be 3D.
#     if breakup:
        
#         k1 = regions['1']['k']
#         i1 = regions['1']['i']
#         j1 = regions['1']['j']
        
#         Mij_loss = dMi_loss[k1,i1,j1]+dMj_loss[k1,i1,j1]
  
#         np.add.at(Mb_gain,  k1, np.transpose(dMb_gain_frac[:,kmin[i1,j1]]*Mij_loss))
#         np.add.at(Nb_gain,  k1, np.transpose(dNb_gain_frac[:,kmin[i1,j1]]*Mij_loss))
         
#     return M1_loss, M2_loss, M_gain, Mb_gain, N1_loss, N2_loss, N_gain, Nb_gain    



    # # Advance PSD Mbins and Nbins by one time/height step
    # def interact_1mom(self,dt):

    #     # Ndists x height x bins
    #     Mbins_old = self.Mbins.copy() 
    #     Mbins = np.zeros_like(Mbins_old)

    #     M_loss = np.zeros_like(Mbins)
    #     M_gain = np.zeros_like(Mbins)
        
    #     indc = self.indc
    #     indb = self.indb

    #     dd = 0
        
    #     for d1 in range(self.dnum):
    #         for d2 in range(d1,self.dnum):
                 
    #             if self.parallel:
                    
                    
    #                 # (dnum x height x bins)
    #                 ck1  = self.cki[d1,:,:]               
    #                 ck2  = self.cki[d2,:,:] 
                    
                                        
    #                 # (height x bins x bins)
    #                 ck12 =  ck1[:,:,None]*ck2[:,None,:] 
                    
    #                 with Parallel(n_jobs=self.n_jobs,verbose=0) as par:
                    
    #                     gain_loss_temp = par(delayed(calculate_1mom)(
    #                                     self.regions[dd]['1']['i'],
    #                                     self.regions[dd]['1']['j'],
    #                                     ck12[batch,:,:],
    #                                     self.dMi_loss[dd,0,:,:],
    #                                     self.dMj_loss[dd,0,:,:],
    #                                     self.dM_gain[dd,0,:,:,:],
    #                                     self.kmin,self.kmid,self.dMb_gain_frac,self.breakup) for batch in self.batches)  

                    
                                        
    #                 # gain_loss_temp = Parallel(n_jobs=self.n_jobs,verbose=0)(delayed(calculate_1mom)(
    #                 #                     self.regions[dd]['1']['i'],
    #                 #                     self.regions[dd]['1']['j'],
    #                 #                     ck1[batch,:],ck2[batch,:],
    #                 #                     self.dMi_loss[dd,0,:,:],
    #                 #                     self.dMj_loss[dd,0,:,:],
    #                 #                     self.dM_gain[dd,0,:,:,:],
    #                 #                     self.kmin,self.kmid,self.dMb_gain_frac,self.breakup) for batch in self.batches)  


    #                 M1_loss_temp = np.vstack([gl[0] for gl in gain_loss_temp])
    #                 M2_loss_temp = np.vstack([gl[1] for gl in gain_loss_temp])
    #                 M_gain_temp =  np.vstack([gl[2] for gl in gain_loss_temp])
    #                 Mb_gain_temp = np.vstack([gl[3] for gl in gain_loss_temp])
                    
      
    #             else:
                
    #                 # (height x bins)
    #                 ck1  = self.cki[d1,:,:]               
    #                 ck2  = self.cki[d2,:,:] 
                    
    #                 # (height x bins x bins)
    #                 ck12 =  ck1[:,:,None]*ck2[:,None,:] 
                    
    #                 M1_loss_temp,M2_loss_temp,\
    #                 M_gain_temp,Mb_gain_temp =\
    #                 calculate_1mom(self.regions[dd]['1']['i'],
    #                                self.regions[dd]['1']['j'],
    #                                ck12,
    #                                self.dMi_loss[dd,0,:,:],
    #                                self.dMj_loss[dd,0,:,:],
    #                                self.dM_gain[dd,0,:,:,:],
    #                                self.kmin,self.kmid,self.dMb_gain_frac,self.breakup)                    
                        
                    
                    
    #             M_loss[d1,:,:]    += M1_loss_temp 
    #             M_loss[d2,:,:]    += M2_loss_temp
                
    #             M_gain[indc,:,:]  += self.Eagg*M_gain_temp
    #             M_gain[indb,:,:]  += self.Ebr*Mb_gain_temp

    #             dd += 1
                
    #     M_loss *= self.Ecb
        
    #     M_net = dt*(M_gain-M_loss) 
                
    #     return M_net
    
    
    
    

# def calculate_regions_batch_OLD(Hlen,bins,kr,ir,jr,x11,x21,ak1,ck1,x12,x22,ak2,ck2,PK,regions):

#     # kr = Height index    (batch,)
#     # ir = collectee index (batch,)
#     # jr = collector index (batch,) 

#     '''
#     Vectorized Integration Regions:
#     cond_1 :  Ignore CC process for these source bins; they don't map to the largest avail bin.
#     cond_2 :  k bin: Lower triangle region. Just clips BR corner.
#                        Triangle = ((xi1,xj1),(xi1,y_left_edge),(xi2,xj1))
#     cond_3 :  k bin: Lower triangle region. Just clips UL corner.
#                        Triangle = ((xi1,xj1),(xi1,xj2),(x_bottom_edge,xj1))                 
#     cond_2b : k+1 bins: Upper triangle region. Just clips BR corner.
#                        Triangle = ((xi1,xj1),(x_top_edge,xj2),(xi2,xj2))
#     cond_3b : k+1 bin: Upper triangle region. Just clips UL corner.
#                              Triangle = ((xi1,xj2),(xi2,xj2),(xi2,y_right_edge)                    
#     cond_4 :  Full Rectangular source region based on self collection: ii == jj --> ii+sbin or jj+sbin
#     cond_5 :  k bin: Top/Bottom clip: Rectangle on left, triangle on right
#                               Rectangle = ((xi1,xj1),(xi1,xj2),(x_top_edge,xj2),(x_top_edge,xj1))
#                               Triangle  = ((x_top_edge,xj1),(x_top_edge,xj2),(x_bottom_edge,xj1))
#     cond_6 :  k bin: Left/Right clip: Rectangle on bottom, triangle on top
#                               Rectangle = ((xi1,xj1),(xi1,y_right_edge),(xi2,y_right_edge),(xi2,xj1))
#                               Triangle  = ((xi1,y_right_edge),(xi1,y_left_edge),(xi2,y_right_edge))
#     cond_7 :  k+1 bin: Triangle in top right corner
#                                 Triangle = ((x_top_edge,xj2),(xi2,xj2),(xi2,y_right_edge))
#     cond_8 :  k bin: Triangle in lower left corner
#                                 Triangle = ((xi1,xj1),(xi1,y_left_edge),(x_bottom_edge,xj1))
#     cond_9:   k bin: Rectangle collection within k bin. All Mass/Number goes into kbin
#     cond_10:  k+1 bin: Rectangle collection within k+1 bin. All Mass/Number goes into kbin
#     '''
    
#     x_bottom_edge = regions['x_bottom_edge']
#     x_top_edge = regions['x_top_edge']
#     y_left_edge = regions['y_left_edge']
#     y_right_edge = regions['y_right_edge']
#     k1 = regions['1']['k']
#     k2 = regions['2']['k']
#     k2b = regions['2b']['k']
#     k3 = regions['3']['k']
#     k3b = regions['3b']['k']
#     k4 = regions['4']['k']
#     k5 = regions['5']['k']
#     k6 = regions['6']['k']
#     k7 = regions['7']['k']
#     k8 = regions['8']['k']
#     k9 = regions['9']['k']
#     k10 = regions['10']['k']
    
#     # Initialize gain term arrays
#     dMi_loss = np.zeros((Hlen,bins,bins))
#     dMj_loss = np.zeros((Hlen,bins,bins))
#     dNi_loss = np.zeros((Hlen,bins,bins))
#     dM_gain  = np.zeros((Hlen,bins,bins,2))
#     dN_gain  = np.zeros((Hlen,bins,bins,2))
  
#     # Calculate transfer rates (rectangular integration, source space)
#     # Collection (eqs. 23-25 in Wang et al. 2007)
#     # ii collecting jj 

#     dMi_loss[kr[k1],ir[k1],jr[k1]] = integrate_fast_kernel(1,0,0,PK[:,k1],ak1[k1],ck1[k1],ak2[k1],ck2[k1],'rectangle',x1=x11[k1],x2=x21[k1],y1=x12[k1],y2=x22[k1])
#     dMj_loss[kr[k1],ir[k1],jr[k1]] = integrate_fast_kernel(0,1,0,PK[:,k1],ak1[k1],ck1[k1],ak2[k1],ck2[k1],'rectangle',x1=x11[k1],x2=x21[k1],y1=x12[k1],y2=x22[k1])
#     dNi_loss[kr[k1],ir[k1],jr[k1]] = integrate_fast_kernel(0,0,0,PK[:,k1],ak1[k1],ck1[k1],ak2[k1],ck2[k1],'rectangle',x1=x11[k1],x2=x21[k1],y1=x12[k1],y2=x22[k1])
#     #dNj_loss = dNi_loss.copy() # Nj loss should be same as Ni loss
    
#     # Condition 4: Self collection. All Mass/Number goes into ii+sbin = jj+sbin kbin
#     xi1 = x11[k4]
#     xi2 = x21[k4] 
#     xj1 = x12[k4]
#     xj2 = x22[k4]
    
#     dM_gain[kr[k4],ir[k4],jr[k4],0]  = integrate_fast_kernel(0,0,1,PK[:,k4],ak1[k4],ck1[k4],ak2[k4],ck2[k4],'rectangle',x1=xi1,x2=xi2,y1=xj1,y2=xj2) 
#     dN_gain[kr[k4],ir[k4],jr[k4],0]  = integrate_fast_kernel(0,0,0,PK[:,k4],ak1[k4],ck1[k4],ak2[k4],ck2[k4],'rectangle',x1=xi1,x2=xi2,y1=xj1,y2=xj2) 


#     # Condition 2:
#     # k bin: Lower triangle region. Just clips BR corner.
#     #                       Triangle = ((xi1,xj1),(xi1,y_left_edge),(xi2,xj1))
#     xt1 = x11[k2]
#     yt1 = x12[k2]
#     xt2 = x11[k2]
#     yt2 = y_left_edge[k2]
#     xt3 = x21[k2]
#     yt3 = x12[k2]
    
#     dM_gain[kr[k2],ir[k2],jr[k2],0] = integrate_fast_kernel(0,0,1,PK[:,k2],ak1[k2],ck1[k2],ak2[k2],ck2[k2],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
#     dM_gain[kr[k2],ir[k2],jr[k2],1] = (dMi_loss[kr[k2],ir[k2],jr[k2]]+dMj_loss[kr[k2],ir[k2],jr[k2]])-dM_gain[kr[k2],ir[k2],jr[k2],0]
    
#     dN_gain[kr[k2],ir[k2],jr[k2],0] = integrate_fast_kernel(0,0,0,PK[:,k2],ak1[k2],ck1[k2],ak2[k2],ck2[k2],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
#     dN_gain[kr[k2],ir[k2],jr[k2],1] = (dNi_loss[kr[k2],ir[k2],jr[k2]])-dN_gain[kr[k2],ir[k2],jr[k2],0]
        
#     # Condition 3:
#     #    k bin: Lower triangle region. Just clips UL corner.
#     #                      Triangle = ((xi1,xj1),(xi1,xj2),(x_bottom_edge,xj1))  
#     xt1 = x11[k3]
#     yt1 = x12[k3]
#     xt2 = x11[k3]
#     yt2 = x22[k3]
#     xt3 = x_bottom_edge[k3]
#     yt3 = x12[k3]
    
#     dM_gain[kr[k3],ir[k3],jr[k3],0] = integrate_fast_kernel(0,0,1,PK[:,k3],ak1[k3],ck1[k3],ak2[k3],ck2[k3],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
#     dM_gain[kr[k3],ir[k3],jr[k3],1] = (dMi_loss[kr[k3],ir[k3],jr[k3]]+dMj_loss[kr[k3],ir[k3],jr[k3]])-dM_gain[kr[k3],ir[k3],jr[k3],0]
        
#     dN_gain[kr[k3],ir[k3],jr[k3],0] = integrate_fast_kernel(0,0,0,PK[:,k3],ak1[k3],ck1[k3],ak2[k3],ck2[k3],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
#     dN_gain[kr[k3],ir[k3],jr[k3],1] = (dNi_loss[kr[k3],ir[k3],jr[k3]])-dN_gain[kr[k3],ir[k3],jr[k3],0]
        
#     # Condition 5: 
        
#     #    k bin: Top/Bottom clip: Rectangle on left, triangle on right
#     #                              Rectangle = ((xi1,xj1),(xi1,xj2),(x_top_edge,xj2),(x_top_edge,xj1))
#     #                              Triangle  = ((x_top_edge,xj1),(x_top_edge,xj2),(x_bottom_edge,xj1))
   
#     xr1 = x11[k5]
#     yr1 = x12[k5]
#     xr2 = x_top_edge[k5]
#     yr2 = x22[k5]
   
#     xt1 = x_top_edge[k5]
#     yt1 = x12[k5]
#     xt2 = x_top_edge[k5]
#     yt2 = x22[k5]
#     xt3 = x_bottom_edge[k5]
#     yt3 = x12[k5]
    
    
#     dM_gain[kr[k5],ir[k5],jr[k5],0] = integrate_fast_kernel(0,0,1,PK[:,k5],ak1[k5],ck1[k5],ak2[k5],ck2[k5],'rectangle',x1=xr1,x2=xr2,y1=yr1,y2=yr2)+\
#                           integrate_fast_kernel(0,0,1,PK[:,k5],ak1[k5],ck1[k5],ak2[k5],ck2[k5],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
    
#     dM_gain[kr[k5],ir[k5],jr[k5],1] = (dMi_loss[kr[k5],ir[k5],jr[k5]]+dMj_loss[kr[k5],ir[k5],jr[k5]])-dM_gain[kr[k5],ir[k5],jr[k5],0]
        
#     dN_gain[kr[k5],ir[k5],jr[k5],0] = integrate_fast_kernel(0,0,0,PK[:,k5],ak1[k5],ck1[k5],ak2[k5],ck2[k5],'rectangle',x1=xr1,x2=xr2,y1=yr1,y2=yr2)+\
#                           integrate_fast_kernel(0,0,0,PK[:,k5],ak1[k5],ck1[k5],ak2[k5],ck2[k5],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
                       
#     dN_gain[kr[k5],ir[k5],jr[k5],1] = (dNi_loss[kr[k5],ir[k5],jr[k5]])-dN_gain[kr[k5],ir[k5],jr[k5],0]
    
    
#     # Condition 6:
#     # k bin: Left/Right clip: Rectangle on bottom, triangle on top
#     #                          Rectangle = ((xi1,xj1),(xi1,y_right_edge),(xi2,y_right_edge),(xi2,xj1))
#     #                          Triangle  = ((xi1,y_right_edge),(xi1,y_left_edge),(xi2,y_right_edge))
        
#     xr1 = x11[k6]
#     yr1 = x12[k6]
#     xr2 = x21[k6]
#     yr2 = y_right_edge[k6]
    
#     xt1 = x11[k6]
#     yt1 = y_right_edge[k6]
#     xt2 = x11[k6]
#     yt2 = y_left_edge[k6]
#     xt3 = x21[k6]
#     yt3 = y_right_edge[k6]
    
    
#     dM_gain[kr[k6],ir[k6],jr[k6],0] = integrate_fast_kernel(0,0,1,PK[:,k6],ak1[k6],ck1[k6],ak2[k6],ck2[k6],'rectangle',x1=xr1,x2=xr2,y1=yr1,y2=yr2)+\
#                           integrate_fast_kernel(0,0,1,PK[:,k6],ak1[k6],ck1[k6],ak2[k6],ck2[k6],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
    
#     dM_gain[kr[k6],ir[k6],jr[k6],1] = (dMi_loss[kr[k6],ir[k6],jr[k6]]+dMj_loss[kr[k6],ir[k6],jr[k6]])-dM_gain[kr[k6],ir[k6],jr[k6],0]
        
#     dN_gain[kr[k6],ir[k6],jr[k6],0] = integrate_fast_kernel(0,0,0,PK[:,k6],ak1[k6],ck1[k6],ak2[k6],ck2[k6],'rectangle',x1=xr1,x2=xr2,y1=yr1,y2=yr2)+\
#                           integrate_fast_kernel(0,0,0,PK[:,k6],ak1[k6],ck1[k6],ak2[k6],ck2[k6],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
                       
#     dN_gain[kr[k6],ir[k6],jr[k6],1] = (dNi_loss[kr[k6],ir[k6],jr[k6]])-dN_gain[kr[k6],ir[k6],jr[k6],0]
        
#     # Condition 7:
#     # k+1 bin: Triangle in top right corner
#     #                            Triangle = ((x_top_edge,xj2),(xi2,xj2),(xi2,y_right_edge))
        
#     xt1 = x_top_edge[k7]
#     yt1 = x22[k7]
#     xt2 = x21[k7]
#     yt2 = x22[k7]
#     xt3 = x21[k7]
#     yt3 = y_right_edge[k7]
    
#     dM_gain[kr[k7],ir[k7],jr[k7],1] = integrate_fast_kernel(0,0,1,PK[:,k7],ak1[k7],ck1[k7],ak2[k7],ck2[k7],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
#     dM_gain[kr[k7],ir[k7],jr[k7],0] = (dMi_loss[kr[k7],ir[k7],jr[k7]]+dMj_loss[kr[k7],ir[k7],jr[k7]])-dM_gain[kr[k7],ir[k7],jr[k7],1]
    
#     dN_gain[kr[k7],ir[k7],jr[k7],1] = integrate_fast_kernel(0,0,0,PK[:,k7],ak1[k7],ck1[k7],ak2[k7],ck2[k7],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
#     dN_gain[kr[k7],ir[k7],jr[k7],0] = (dNi_loss[kr[k7],ir[k7],jr[k7]])-dN_gain[kr[k7],ir[k7],jr[k7],1]
    
        
#     # Condition 8:
#     #  k bin: Triangle in lower left corner
#     #                                Triangle = ((xi1,xj1),(xi1,y_left_edge),(x_bottom_edge,xj1))
#     xt1 = x11[k8]
#     yt1 = x12[k8]
#     xt2 = x11[k8]
#     yt2 = y_left_edge[k8]
#     xt3 = x_bottom_edge[k8]
#     yt3 = x12[k8]
    
#     dM_gain[kr[k8],ir[k8],jr[k8],0] = integrate_fast_kernel(0,0,1,PK[:,k8],ak1[k8],ck1[k8],ak2[k8],ck2[k8],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
#     dM_gain[kr[k8],ir[k8],jr[k8],1] = (dMi_loss[kr[k8],ir[k8],jr[k8]]+dMj_loss[kr[k8],ir[k8],jr[k8]])-dM_gain[kr[k8],ir[k8],jr[k8],0]
    
#     dN_gain[kr[k8],ir[k8],jr[k8],0] = integrate_fast_kernel(0,0,0,PK[:,k8],ak1[k8],ck1[k8],ak2[k8],ck2[k8],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
#     dN_gain[kr[k8],ir[k8],jr[k8],1] = (dNi_loss[kr[k8],ir[k8],jr[k8]])-dN_gain[kr[k8],ir[k8],jr[k8],0]
        
#     # Condition 9: Rectangle collection within k bin. All Mass/Number goes into kbin
#     xi1 = x11[k9]
#     xi2 = x21[k9]
#     xj1 = x12[k9]
#     xj2 = x22[k9]

#     dM_gain[kr[k9],ir[k9],jr[k9],0]  = integrate_fast_kernel(0,0,1,PK[:,k9],ak1[k9],ck1[k9],ak2[k9],ck2[k9],'rectangle',x1=xi1,x2=xi2,y1=xj1,y2=xj2) 
#     dN_gain[kr[k9],ir[k9],jr[k9],0]  = integrate_fast_kernel(0,0,0,PK[:,k9],ak1[k9],ck1[k9],ak2[k9],ck2[k9],'rectangle',x1=xi1,x2=xi2,y1=xj1,y2=xj2) 

#     # Condition 10: Rectangle collection within k+1 bin. All Mass/Number goes into kbin
#     xi1 = x11[k10]
#     xi2 = x21[k10]
#     xj1 = x12[k10]
#     xj2 = x22[k10]
    
#     dM_gain[kr[k10],ir[k10],jr[k10],1]  = integrate_fast_kernel(0,0,1,PK[:,k10],ak1[k10],ck1[k10],ak2[k10],ck2[k10],'rectangle',x1=xi1,x2=xi2,y1=xj1,y2=xj2) 
#     dN_gain[kr[k10],ir[k10],jr[k10],1]  = integrate_fast_kernel(0,0,0,PK[:,k10],ak1[k10],ck1[k10],ak2[k10],ck2[k10],'rectangle',x1=xi1,x2=xi2,y1=xj1,y2=xj2) 
   
#     # Condition 11 (2b): Triangle collection within k+1 bin. Occurs when xk+1 clips BR corner and intersects top edge.
#     # Triangle = ((xi2,xj1),(x_top_edge,xj2),(xi2,xj2))
#     xt1 = x_top_edge[k2b]
#     yt1 = x22[k2b]
#     xt2 = x21[k2b]
#     yt2 = x22[k2b]
#     xt3 = x21[k2b]
#     yt3 = x12[k2b]
    
#     dM_gain[kr[k2b],ir[k2b],jr[k2b],1] = integrate_fast_kernel(0,0,1,PK[:,k2b],ak1[k2b],ck1[k2b],ak2[k2b],ck2[k2b],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
#     dM_gain[kr[k2b],ir[k2b],jr[k2b],0] = (dMi_loss[kr[k2b],ir[k2b],jr[k2b]]+dMj_loss[kr[k2b],ir[k2b],jr[k2b]])-dM_gain[kr[k2b],ir[k2b],jr[k2b],1]
    
#     dN_gain[kr[k2b],ir[k2b],jr[k2b],1] = integrate_fast_kernel(0,0,0,PK[:,k2b],ak1[k2b],ck1[k2b],ak2[k2b],ck2[k2b],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
#     dN_gain[kr[k2b],ir[k2b],jr[k2b],0] = (dNi_loss[kr[k2b],ir[k2b],jr[k2b]])-dN_gain[kr[k2b],ir[k2b],jr[k2b],1]
    
#     # Condition 12 (3b): Triangle collection within k+1 bin. Occurs when xk+1 clips UL corner and intersects with right edge.
#     # Triangle = ((xi1,xj2),(xi2,xj2),(xi2,y_right_edge)
#     xt1 = x11[k3b]
#     yt1 = x22[k3b]
#     xt2 = x21[k3b]
#     yt2 = x22[k3b]
#     xt3 = x21[k3b]
#     yt3 = y_right_edge[k3b]
    
#     dM_gain[kr[k3b],ir[k3b],jr[k3b],1] = integrate_fast_kernel(0,0,1,PK[:,k3b],ak1[k3b],ck1[k3b],ak2[k3b],ck2[k3b],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
#     dM_gain[kr[k3b],ir[k3b],jr[k3b],0] = (dMi_loss[kr[k3b],ir[k3b],jr[k3b]]+dMj_loss[kr[k3b],ir[k3b],jr[k3b]])-dM_gain[kr[k3b],ir[k3b],jr[k3b],1]
    
#     dN_gain[kr[k3b],ir[k3b],jr[k3b],1] = integrate_fast_kernel(0,0,0,PK[:,k3b],ak1[k3b],ck1[k3b],ak2[k3b],ck2[k3b],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
#     dN_gain[kr[k3b],ir[k3b],jr[k3b],0] = (dNi_loss[kr[k3b],ir[k3b],jr[k3b]])-dN_gain[kr[k3b],ir[k3b],jr[k3b],1]   
    
#     # print('dMi_loss=',np.nansum(dMi_loss))
#     # print('dMj_loss=',np.nansum(dMj_loss))
#     # print('dM_gain=',np.nansum(dM_gain))
#     # print('res=',np.nansum(dM_gain)-np.nansum(dMi_loss)-np.nansum(dMj_loss))
#     # raise Exception()
    
#     return dMi_loss, dMj_loss, dM_gain, dNi_loss, dN_gain   





    # def plot_source_target_vec(self,d1,d2,kk,ii,jj,invert=False,full=False):
        
    #     '''
    #     Method for plotting the integration region of the (kk,ii,jj) interaction.
    #     Useful for debugging.
    #     '''
        
    #     from matplotlib.patches import Polygon
        
    #     import matplotlib.pyplot as plt
        
    #     dist1 = self.dists[d1,kk]
    #     dist2 = self.dists[d2,kk]
    #     xedges   = dist1.xedges
        
    #     kbin_min = self.xi1[:,None]+self.xi1[None,:]
        
    #     idx_min = np.searchsorted(xedges, kbin_min, side='right')-1
        
    #     # Clamp values to valid bin range [0, B-1]
    #     kmin = np.clip(idx_min, 0, self.bins-1) 
    #     kmid = np.minimum(kmin+1,self.bins-1)

    #     # For self-collection on mass-doubling grid, gain bounds cover exactly one bin
    #     kdiag = np.diag_indices(kmin.shape[0])
    #     kmin[kdiag] = kdiag[0]+self.sbin
    #     kmid[kdiag] = kdiag[0]+self.sbin
          
    #     kmin = np.clip(kmin,0,self.bins-1)
    #     kmid = np.clip(kmid,0,self.bins-1)
        
    #     cond_1 = np.tile(((self.ind_i>=(self.bins-self.sbin)) | (self.ind_j>=(self.bins-self.sbin))),(self.Hlen,1,1))
    #         # WORKING
    #     self_col = np.ones((self.Hlen,self.pnum,self.bins,self.bins),dtype=int)
        
    #     dd = 0
    #     for dd1 in range(self.dnum):
    #         for dd2 in range(d1,self.dnum):
    #             if (dd1==d1) & (dd2==d2):
    #                 pp = dd
                    
    #             if dd1==dd2:
    #                 self_col[:,dd,:,:] = np.tile(np.triu(np.ones((self.bins,self.bins),dtype=int),k=0),(self.Hlen,1,1))
        
    #         dd += 1
            
    #     # (dnum x height x bins)
    #     x11  = self.x1[d1,:,:]
    #     x21  = self.x2[d1,:,:]
    #     ak1  = self.aki[d1,:,:]
    #     ck1  = self.cki[d1,:,:] 
        
    #     x12  = self.x1[d2,:,:]
    #     x22  = self.x2[d2,:,:]
    #     ak2  = self.aki[d2,:,:] 
    #     ck2  = self.cki[d2,:,:]     
        
    #     Hlen,bins = x11.shape
        
    #     regions = setup_regions(x11,x21,ak1,ck1,x12,x22,ak2,ck2,self.xi1,self.xi2,kmin,cond_1,self_col[:,pp,:,:])
        
    #     x_bottom_edge = regions['x_bottom_edge']
    #     x_top_edge = regions['x_top_edge']
    #     y_left_edge = regions['y_left_edge']
    #     y_right_edge = regions['y_right_edge']
        
    #     check_bottom = (x11[:,:,None]<x_bottom_edge) &\
    #                    (x21[:,:,None]>x_bottom_edge)
         
    #     check_top = (x11[:,:,None]<x_top_edge) &\
    #                 (x21[:,:,None]>x_top_edge)
                    
    #     check_left = (x12[:,None,:]<y_left_edge) &\
    #                  (x22[:,None,:]>y_left_edge)
                    
    #     check_right = (x12[:,None,:]<y_right_edge) &\
    #                   (x22[:,None,:]>y_right_edge)            
               
    #     check_middle = ((0.5*(x11[:,:,None]+x21[:,:,None]))+(0.5*(x12[:,None,:]+x22[:,None,:])))<(self.xi2[kmin][None,:,:])
                   
        
    #     # NEW
    #     #cond_BR_corner = ((x21==xi2)[:,:,None])&((x12==xi1)[:,None,:]) # Bottom right corner
    #     #cond_UL_corner = ((x11==xi1)[:,:,None])&((x22==xi2)[:,None,:]) # Upper left corner
        
    #     #cond_BR_corner = ((x21==self.xi2[None,:])[:,:,None])&((x12==self.xi1[None,:])[:,None,:]) # Bottom right corner
    #     #cond_UL_corner = ((x11==self.xi1[None,:])[:,:,None])&((x22==self.xi2[None,:])[:,None,:]) # Upper left corner
        
    #     cond_BR_corner = (x21[:,:,None]==x_bottom_edge)&(x12[:,None,:]==y_right_edge)
    #     cond_UL_corner = (x11[:,:,None]==x_top_edge)&(x22[:,None,:]==y_left_edge)
        
    #    # print('cond_BR_corner',cond_BR_corner.shape)
    #    # print('xi2=',self.xi2.shape)
    #    # raise Exception()
       
    #    # print((x21[:,:,None]==self.xi2[None,:,None]))
    #    # print(cond_BR_corner.shape)
    #     #print((x12[:,None,:]==self.xi1[None,None,:]).shape)
    #     #print((x11[:,:,None]==self.xi1[None,:,None]).shape)
    #     #print((x22[:,None,:]==self.xi2[None,None,:]).shape)        
    #     #print((((x21==self.xi2)[:,:,None])&((x12==self.xi1)[:,None,:])).shape)
                       
    #     #print((x21==self.xi2[None,:]).shape)
    #     #print((x12[:,None]==self.xi1[None,:]).shape)
    #     #print()
        
    #     #raise Exception()
        
    #     print('x21=',x21[kk,ii])
    #     print('xi2=',self.xi2[ii])
    #     print('x12=',x12[kk,jj])
    #     print('xi1=',self.xi1[jj])
    #     print('y_left_edge=',y_left_edge[kk,ii,jj])
    #     print('y_right_edge=',y_right_edge[kk,ii,jj])
    #     print('x_bottom_edge=',x_bottom_edge[kk,ii,jj])
    #     print('x_top_edge=',x_top_edge[kk,ii,jj])
    #    # print('check1=',(x21[:,:,None]==self.xi2[None,:,None])[kk,ii,jj])
        
    #     #print('x_bottom_edge=',x_bottom_edge[kk,ii,jj])
    #     #print('x21[:,:,None]=',x21[kk,ii])
    #     #print('y_right_edge=',y_right_edge[kk,ii,jj])
    #     #print('x12[:,None,:]=',x22[])
        

    #     print('check_bottom=',check_bottom[kk,ii,jj])
    #     print('check_top=',check_top[kk,ii,jj])
    #     print('check_left=',check_left[kk,ii,jj])
    #     print('check_right=',check_right[kk,ii,jj])
    #     print('check_middle=',check_middle[kk,ii,jj])
    #     print('cond_BR_corner=',cond_BR_corner[kk,ii,jj])
    #     print('cond_UL_corner=',cond_UL_corner[kk,ii,jj])
    #     #print('cond_UL_corner_shape=',cond_UL_corner.shape)
        
    #     xi1 = dist1.x1[ii]
    #     xi2 = dist1.x2[ii]
    #     xj1 = dist2.x1[jj]
    #     xj2 = dist2.x2[jj]
        
    #     print('xi1=',xi1)
    #     print('xi2=',xi2)
    #     print('xj1=',xj1)
    #     print('xj2=',xj2)
    
    #     # NOTE: Need to avoid doing anything with the last bin
    #     kbin_min = dist1.xi1[:,None]+dist2.xi1[None,:]
    #     kbin_max = dist1.xi2[:,None]+dist2.xi2[None,:]
        
    #     xk_min = dist2.xi2[kmin][ii,jj]
        
    #     # Find region with requested (ii,jj) indices.
    #     x_bottom_edge = regions['x_bottom_edge']
    #     x_top_edge = regions['x_top_edge']
    #     y_left_edge = regions['y_left_edge']
    #     y_right_edge = regions['y_right_edge']
    #     i1 = regions['1']['i']
    #     j1 = regions['1']['j']
    #     k1 = regions['1']['k']
    #     i2 = regions['2']['i']
    #     j2 = regions['2']['j']
    #     k2 = regions['2']['k']
    #     i2b = regions['2b']['i']
    #     j2b = regions['2b']['j']
    #     k2b = regions['2b']['k']
    #     i3 = regions['3']['i']
    #     j3 = regions['3']['j']
    #     k3 = regions['3']['k']
    #     i3b = regions['3b']['i']
    #     j3b = regions['3b']['j']
    #     k3b = regions['3b']['k']
    #     i4 = regions['4']['i']
    #     j4 = regions['4']['j']
    #     k4 = regions['4']['k']
    #     i5 = regions['5']['i']
    #     j5 = regions['5']['j']
    #     k5 = regions['5']['k']
    #     i6 = regions['6']['i']
    #     j6 = regions['6']['j']
    #     k6 = regions['6']['k']
    #     i7 = regions['7']['i']
    #     j7 = regions['7']['j']
    #     k7 = regions['7']['k']
    #     i8 = regions['8']['i']
    #     j8 = regions['8']['j']
    #     k8 = regions['8']['k']
    #     i9 = regions['9']['i']
    #     j9 = regions['9']['j']
    #     k9 = regions['9']['k']
    #     i10 = regions['10']['i']
    #     j10 = regions['10']['j']
    #     k10 = regions['10']['k']
        
        
    #     cond_loss = np.zeros((self.Hlen,self.bins,self.bins),dtype=int)
    #     cond_gain = np.zeros((self.Hlen,self.bins,self.bins),dtype=int)
        
        
    #     cond_loss[k1,i1,j1]    = 1
    #     cond_gain[k2,i2,j2]    = 2
    #     cond_gain[k3,i3,j3]    = 3
    #     cond_gain[k4,i4,j4]    = 4
    #     cond_gain[k5,i5,j5]    = 5
    #     cond_gain[k6,i6,j6]    = 6
    #     cond_gain[k7,i7,j7]    = 7
    #     cond_gain[k8,i8,j8]    = 8
    #     cond_gain[k9,i9,j9]    = 9 
    #     cond_gain[k10,i10,j10] = 10
    #     cond_gain[k2b,i2b,j2b] = 11 
    #     cond_gain[k3b,i3b,j3b] = 12
        
    #     cond_2 = np.zeros((self.Hlen,self.bins,self.bins),dtype=int)
    #     cond_3 = np.zeros((self.Hlen,self.bins,self.bins),dtype=int)
    #     cond_4 = np.zeros((self.Hlen,self.bins,self.bins),dtype=int)
    #     cond_5 = np.zeros((self.Hlen,self.bins,self.bins),dtype=int)
    #     cond_6 = np.zeros((self.Hlen,self.bins,self.bins),dtype=int)
    #     cond_7 = np.zeros((self.Hlen,self.bins,self.bins),dtype=int)
    #     cond_8 = np.zeros((self.Hlen,self.bins,self.bins),dtype=int)
    #     cond_9 = np.zeros((self.Hlen,self.bins,self.bins),dtype=int) 
    #     cond_10 = np.zeros((self.Hlen,self.bins,self.bins),dtype=int)
    #     cond_11 = np.zeros((self.Hlen,self.bins,self.bins),dtype=int)
    #     cond_12 = np.zeros((self.Hlen,self.bins,self.bins),dtype=int)
        
    #     cond_2[k2,i2,j2] = 1 
    #     cond_3[k3,i3,j3] = 1 
    #     cond_4[k4,i4,j4] = 1 
    #     cond_5[k5,i5,j5] = 1
    #     cond_6[k6,i6,j6] = 1 
    #     cond_7[k7,i7,j7] = 1
    #     cond_8[k8,i8,j8] = 1
    #     cond_9[k9,i9,j9] = 1
    #     cond_10[k10,i10,j10] = 1
    #     cond_11[k2b,i2b,j2b] = 1
    #     cond_12[k3b,i3b,j3b] = 1
        
    #     cond_gain_full = np.stack((cond_2,cond_3,cond_4,cond_5,cond_6,cond_7,
    #                                cond_8,cond_9,cond_10,cond_11,cond_12),axis=0)
        
    #     print('cond_gain_full=',np.unique((cond_gain_full>0).sum(axis=0)))
        
    #     print(np.unique(cond_gain))
    #     print('cond = ',cond_gain[kk,ii,jj])
        
    #     print('cond_loss num = ',np.count_nonzero(cond_loss))
    #     print('cond_gain num = ',np.count_nonzero(cond_gain))
        
    #    # print('k1 shape=',np.unique(k1))
    #     #raise Exception()
            
    #     fig, ax = plt.subplots();   
        
    #     if invert:
    #         source_rec = Polygon(((xj1,xi1),(xj1,xi2),(xj2,xi2),(xj2,xi1)))
    #     else:           
    #         source_rec = Polygon(((xi1,xj1),(xi1,xj2),(xi2,xj2),(xi2,xj1)))
        
    #     ax.add_patch(source_rec)
        
    #     if invert:
    #         # NOTE: need to reformat with new gain bin regions
    #         ax.plot([kbin_min[ii,jj]-xi1,kbin_min[ii,jj]-xi2],[xi1,xi2],'b')
    #         ax.plot([xk_min-xi1,xk_min-xi2],[xi1,xi2],'k')
    #         ax.plot([kbin_max[ii,jj]-xi1,kbin_max[ii,jj]-xi2],[xi1,xi2],'r')

    #         ax.invert_yaxis()
            
    #     else:
            
    #         if cond_gain[kk,ii,jj]==2:
    #             # Triangle = ((xi1,xj1),(xi1,x_left_edge),(xi2,xj1))
    #             kgain_t = Polygon(((xi1,xj1),(xi1,y_left_edge[kk,ii,jj]),(xi2,xj1)),closed=True,facecolor='purple')
    #         elif cond_gain[kk,ii,jj]==3:
    #             kgain_t = Polygon((((xi1,xj1),(xi1,xj2),(x_bottom_edge[kk,ii,jj],xj1))),closed=True,facecolor='purple')
    #         elif cond_gain[kk,ii,jj]==11:
    #             kgain_t = Polygon((((xi2,xj1),(x_top_edge[kk,ii,jj],xj2),(xi2,xj2))),closed=True,facecolor='purple')
    #         elif cond_gain[kk,ii,jj]==12:
    #             kgain_t = Polygon(((xi1,xj2),(xi2,xj2),(xi2,y_right_edge[kk,ii,jj])),closed=True,facecolor='purple')
    #         elif cond_gain[kk,ii,jj]==7:
    #             kgain_t = Polygon(((x_top_edge[kk,ii,jj],xj2),(xi2,xj2),(xi2,y_right_edge[kk,ii,jj])),closed=True,facecolor='purple')
    #         elif cond_gain[kk,ii,jj]==8:
    #             kgain_t = Polygon(((xi1,xj1),(xi1,y_left_edge[kk,ii,jj]),(x_bottom_edge[kk,ii,jj],xj1)),closed=True,facecolor='purple')
    #         elif np.isin(cond_gain[kk,ii,jj],[4,9,10]):
    #             kgain_r = Polygon(((xi1,xj1),(xi1,xj2),(xi2,xj2),(xi2,xj1)),closed=True,facecolor='purple')
    #         elif cond_gain[kk,ii,jj]==5:
    #             kgain_t = Polygon(((x_top_edge[kk,ii,jj],xj1),(x_top_edge[kk,ii,jj],xj2),(x_bottom_edge[kk,ii,jj],xj1)),closed=True,facecolor='purple')
    #             kgain_r =Polygon(((xi1,xj1),(xi1,xj2),(x_top_edge[kk,ii,jj],xj2),(x_top_edge[kk,ii,jj],xj1)),closed=True,facecolor='orange')
    #         elif cond_gain[kk,ii,jj]==6:
    #             kgain_t = Polygon(((xi1,y_right_edge[kk,ii,jj]),(xi1,y_left_edge[kk,ii,jj]),(xi2,y_right_edge[kk,ii,jj])),closed=True,facecolor='purple')
    #             kgain_r =Polygon(((xi1,xj1),(xi1,y_right_edge[kk,ii,jj]),(xi2,y_right_edge[kk,ii,jj]),(xi2,xj1)),closed=True,facecolor='orange')
            
    #         if  (np.isin(cond_gain[kk,ii,jj],[2,3,5,6,7,8,11,12])):
    #             ax.add_patch(kgain_t)
            
    #         if (np.isin(cond_gain[kk,ii,jj],[4,5,6,9])):
    #             ax.add_patch(kgain_r)
            
    #         ax.plot([dist1.xi1[ii],dist1.xi2[ii]],[kbin_min[ii,jj]-dist1.xi1[ii],kbin_min[ii,jj]-dist1.xi2[ii]],'b')
    #         ax.plot([dist1.xi1[ii],dist1.xi2[ii]],[xk_min-dist1.xi1[ii],xk_min-dist1.xi2[ii]],'k')
    #         ax.plot([dist1.xi1[ii],dist1.xi2[ii]],[kbin_max[ii,jj]-dist1.xi1[ii],kbin_max[ii,jj]-dist1.xi2[ii]],'r')
    
    #     return fig, ax, cond_gain, cond_loss 
    
    
# def setup_regions(x11,x21,ak1,ck1,x12,x22,ak2,ck2,xi1,xi2,kmin,cond_1,sc_inds):

#     Hlen, bins = np.shape(x11)

#     # (heights x bins x bins)
#     x_bottom_edge = (xi2[kmin][None,:,:]-x12[:,None,:])
#     x_top_edge = (xi2[kmin][None,:,:]-x22[:,None,:])
#     y_left_edge = (xi2[kmin][None,:,:]-x11[:,:,None])
#     y_right_edge = (xi2[kmin][None,:,:]-x21[:,:,None])
    
#     check_bottom = (x11[:,:,None]<x_bottom_edge) &\
#                    (x21[:,:,None]>x_bottom_edge)
     
#     check_top = (x11[:,:,None]<x_top_edge) &\
#                 (x21[:,:,None]>x_top_edge)
                
#     check_left = (x12[:,None,:]<y_left_edge) &\
#                  (x22[:,None,:]>y_left_edge)
                
#     check_right = (x12[:,None,:]<y_right_edge) &\
#                   (x22[:,None,:]>y_right_edge)            
           
#     check_middle = ((0.5*(x11[:,:,None]+x21[:,:,None]))+(0.5*(x12[:,None,:]+x22[:,None,:])))<(xi2[kmin][None,:,:])
               
#     # If opposite sides check true, then integral region is rectangle + triangle
#     # If adjacent sides check true or single side, then integral region is triangle       
           
#     # Check which opposite side is higher for cases where we have rectangle + triangle
#     # NOTE: It SHOULD be the case that y_left_edge>y_right_edge and x_bottom_edge>x_top_edge
#     # This just has to do with the geometry of the x+y mapping, i.e., the x+y lines have negative slope.
    
#     '''
#     Vectorized Integration Regions:
#     cond_1 :  Ignore CC process for these source bins; they don't map to the largest avail bin.
#     cond_2 :  k bin: Lower triangle region. Just clips BR corner. Occurs on diagonal+1 indices.
#                        Triangle = ((xi1,xj1),(xi1,y_left_edge),(xi2,xj1))
#     cond_3 :  k bin: Lower triangle region. Just clips UL corner. Occurs on diagonal-1 indices.
#                        Triangle = ((xi1,xj1),(xi1,xj2),(x_bottom_edge,xj1))                 
#     cond_2b : k+1 bins: Upper triangle region. Just clips BR corner.
#                        Triangle = ((xi1,xj1),(x_top_edge,xj2),(xi2,xj2))
#     cond_3b : k+1 bin: Upper triangle region. Just clips UL corner.
#                              Triangle = ((xi1,xj2),(xi2,xj2),(xi2,y_right_edge)                    
#     cond_4 :  Full Rectangular source region based on self collection: ii == jj --> ii+sbin or jj+sbin
#     cond_5 :  k bin: Top/Bottom clip: Rectangle on left, triangle on right
#                               Rectangle = ((xi1,xj1),(xi1,xj2),(x_top_edge,xj2),(x_top_edge,xj1))
#                               Triangle  = ((x_top_edge,xj1),(x_top_edge,xj2),(x_bottom_edge,xj1))
#     cond_6 :  k bin: Left/Right clip: Rectangle on bottom, triangle on top
#                               Rectangle = ((xi1,xj1),(xi1,y_right_edge),(xi2,y_right_edge),(xi2,xj1))
#                               Triangle  = ((xi1,y_right_edge),(xi1,y_left_edge),(xi2,y_right_edge))
#     cond_7 :  k+1 bin: Triangle in top right corner
#                                 Triangle = ((x_top_edge,xj2),(xi2,xj2),(xi2,y_right_edge))
#     cond_8 :  k bin: Triangle in lower left corner
#                                 Triangle = ((xi1,xj1),(xi1,y_left_edge),(x_bottom_edge,xj1))
#     cond_9:   k bin: Rectangle collection within k bin. All Mass/Number goes into kbin
#     cond_10:  k+1 bin: Rectangle collection within k+1 bin. All Mass/Number goes into kbin
#     '''
         
#     cond_touch = (check_bottom|check_top|check_left|check_right)
    
#     # NEW
#     #cond_BR_corner = ((x21==xi2)[:,:,None])&((x12==xi1)[:,None,:]) # Bottom right corner
#     #cond_UL_corner = ((x11==xi1)[:,:,None])&((x22==xi2)[:,None,:]) # Upper left corner
    
#     # NEW 2
#     cond_BR_corner = (x21[:,:,None]==x_bottom_edge)&(x12[:,None,:]==y_right_edge)
#     cond_UL_corner = (x11[:,:,None]==x_top_edge)&(x22[:,None,:]==y_left_edge)
    
#     cond_2_corner = (cond_BR_corner)&(check_left)
#     cond_3_corner = (cond_UL_corner)&(check_bottom)
    
#     cond_2b_corner = (cond_BR_corner)&(check_top)
#     cond_3b_corner = (cond_UL_corner)&(check_right)
    
#     cond_2 = np.eye(bins,k=1,dtype=bool)[None,:,:] & (cond_2_corner) & (~cond_1) & (cond_touch)
#     cond_3 = np.eye(bins,k=-1,dtype=bool)[None,:,:] & (cond_3_corner) & (~cond_1) & (cond_touch)
    
#     cond_2b = cond_2b_corner & (~cond_1)
#     cond_3b = cond_3b_corner & (~cond_1)
    
#     cond_4 = np.eye(bins,dtype=bool)[None,:,:] & ((~cond_1))
#     cond_nt = (~(cond_1|cond_2|cond_3|cond_4))
#     cond_5 = (check_top&check_bottom)  & cond_nt
#     cond_6 = (check_left&check_right)  & cond_nt
#     cond_7 =  (check_right&check_top)  & cond_nt
#     cond_8 = (check_left&check_bottom) & cond_nt
#     cond_rect = (~cond_touch)&(~cond_1)&(~cond_4)&(~cond_5)&(~cond_6)&(~cond_7)&(~cond_8)
#     cond_9 = (cond_rect&check_middle)
#     cond_10 = (cond_rect&(~check_middle))
    
#     k1, i1, j1  = np.nonzero((~cond_1)&sc_inds) # Only do loss/gain terms for 0>bins-sbin bins
#     k2, i2, j2  = np.nonzero(cond_2&sc_inds)
#     k3, i3, j3  = np.nonzero(cond_3&sc_inds)
#     k2b, i2b, j2b  = np.nonzero(cond_2b&sc_inds)
#     k3b, i3b, j3b  = np.nonzero(cond_3b&sc_inds)
#     k4, i4, j4  = np.nonzero(cond_4&sc_inds)
#     k5, i5, j5  = np.nonzero(cond_5&sc_inds)
#     k6, i6, j6  = np.nonzero(cond_6&sc_inds)
#     k7, i7, j7  = np.nonzero(cond_7&sc_inds)
#     k8, i8, j8  = np.nonzero(cond_8&sc_inds)
#     k9, i9, j9  = np.nonzero(cond_9&sc_inds)
#     k10,i10,j10 = np.nonzero(cond_10&sc_inds)
    
#     # Returns dictionary of source integration region type indices
#     return    {'x_bottom_edge':x_bottom_edge,
#                'x_top_edge':x_top_edge,
#                'y_left_edge':y_left_edge,
#                'y_right_edge':y_right_edge,
#                '1' :{'k':k1,'i':i1,'j':j1},
#                '2' :{'k':k2,'i':i2,'j':j2},
#                '2b':{'k':k2b,'i':i2b,'j':j2b},
#                '3' :{'k':k3,'i':i3,'j':j3},
#                '3b':{'k':k3b,'i':i3b,'j':j3b},
#                '4' :{'k':k4,'i':i4,'j':j4},
#                '5' :{'k':k5,'i':i5,'j':j5},
#                '6' :{'k':k6,'i':i6,'j':j6},
#                '7' :{'k':k7,'i':i7,'j':j7},
#                '8' :{'k':k8,'i':i8,'j':j8},
#                '9' :{'k':k9,'i':i9,'j':j9},
#                '10':{'k':k10,'i':i10,'j':j10}}
    
    
# def calculate_2mom_batch_NEW(Hlen,bins,region_inds,x_bottom_edge,x_top_edge,y_left_edge,y_right_edge,
#                          kr,ir,jr,x11,x21,ak1,ck1,x12,x22,ak2,ck2,PK,kmin,kmid,
#                          dMb_gain_frac,dNb_gain_frac,w,L,breakup=False):    

#     # kr = Height index    (batch,)
#     # ir = collectee index (batch,)
#     # jr = collector index (batch,) 

#     '''
#     Vectorized Integration Regions:
#     cond_1 :  All rectangular bin-pair interactions used
#     cond_2 :  k bin: Lower triangle region. Just clips BR corner.
#                        Triangle = ((xi1,xj1),(xi1,y_left_edge),(xi2,xj1))
#     cond_3 :  k bin: Lower triangle region. Just clips UL corner.
#                        Triangle = ((xi1,xj1),(xi1,xj2),(x_bottom_edge,xj1))                 
#     cond_2b : k+1 bins: Upper triangle region. Just clips BR corner.
#                        Triangle = ((xi1,xj1),(x_top_edge,xj2),(xi2,xj2))
#     cond_3b : k+1 bin: Upper triangle region. Just clips UL corner.
#                              Triangle = ((xi1,xj2),(xi2,xj2),(xi2,y_right_edge)                    
#     cond_4 :  Full Rectangular source region based on self collection: ii == jj --> ii+sbin or jj+sbin
#     cond_5 :  k bin: Top/Bottom clip: Rectangle on left, triangle on right
#                               Rectangle = ((xi1,xj1),(xi1,xj2),(x_top_edge,xj2),(x_top_edge,xj1))
#                               Triangle  = ((x_top_edge,xj1),(x_top_edge,xj2),(x_bottom_edge,xj1))
#     cond_6 :  k bin: Left/Right clip: Rectangle on bottom, triangle on top
#                               Rectangle = ((xi1,xj1),(xi1,y_right_edge),(xi2,y_right_edge),(xi2,xj1))
#                               Triangle  = ((xi1,y_right_edge),(xi1,y_left_edge),(xi2,y_right_edge))
#     cond_7 :  k+1 bin: Triangle in top right corner
#                                 Triangle = ((x_top_edge,xj2),(xi2,xj2),(xi2,y_right_edge))
#     cond_8 :  k bin: Triangle in lower left corner
#                                 Triangle = ((xi1,xj1),(xi1,y_left_edge),(x_bottom_edge,xj1))
#     cond_9:   k bin: Rectangle collection within k bin. All Mass/Number goes into kbin
#     cond_10:  k+1 bin: Rectangle collection within k+1 bin. All Mass/Number goes into kbin
#     '''
    
#     # Indices for (batch,)
#     # (kr[ind],ir[ind],jr[ind]) gives mapping for
#     # (Hlen,i bins,j bins)
#     k2 = np.flatnonzero(region_inds==2)
#     k3 = np.flatnonzero(region_inds==3) 
#     k4 = np.flatnonzero(region_inds==4) 
#     k5 = np.flatnonzero(region_inds==5) 
#     k6 = np.flatnonzero(region_inds==6) 
#     k7 = np.flatnonzero(region_inds==7)
#     k8 = np.flatnonzero(region_inds==8) 
#     k9 = np.flatnonzero(region_inds==9) 
#     k10 = np.flatnonzero(region_inds==10) 
#     k2b = np.flatnonzero(region_inds==11) 
#     k3b = np.flatnonzero(region_inds==12)
    
#     klen = len(region_inds)
    
#     # Initialize gain term arrays
#     dMi_loss = np.zeros((klen,)) # i Mass loss for each bin-pair (k,i,j)
#     dMj_loss = np.zeros((klen,)) # j Mass loss for each bin-pair (k,i,j)
#     dNi_loss = np.zeros((klen,)) # i (j) Number loss for each bin-pair (k,i,j)
#     dM_gain  = np.zeros((klen,2)) # Mass gain for each bin-pair (k,i,j)
#     dN_gain  = np.zeros((klen,2)) # Number gain for each bin-pair (k,i,j)
    
#     # NOTE PRECOMPUTE MASS/NUMBER COEFFICIENTS HERE FOR ALL BIN-PAIR INTERACTIONS
#     # This is: F(x,y)* Nx(x) * Ny(y) = (F=a+b*x+c*y+d*x*y)*(Nx=ax*x+cx)*(Ny=ay*y+cy)
#     C = combined_coeffs_array(PK, ak1, ck1, ak2, ck2)

#     # Calculate source bin-pair rectangle integral factors. Note, tried to
#     # combine as many of these factors together as possible to avoid redundant calculations.
#     dNi_loss, dMi_loss, dMj_loss = source_integrals(C,x1=x11,x2=x21,y1=x12,y2=x22)
    
#     dM_loss = dMi_loss+dMj_loss # Total bin-pair mass loss
    
#     # NOTE: Try (Hlen, bins, batch) and then sum along batch axis?
    
#     dM_gain[k4,0]  = dM_loss[k4].copy()
#     dN_gain[k4,0]  = dNi_loss[k4].copy()

#     # Condition 2:
#     # k bin: Lower triangle region. Just clips BR corner.
#     #                       Triangle = ((xi1,xj1),(xi1,y_left_edge),(xi2,xj1))
#     if len(k2)>0:
#         xt1 = x11[k2]
#         yt1 = x12[k2]
#         xt2 = x11[k2]
#         yt2 = y_left_edge[k2]
#         xt3 = x21[k2]
#         yt3 = x12[k2]
        
#         dN_gain_temp, dM_gain_temp = tri_gain_integrals(C[:,:,k2], xt1, yt1, xt2, yt2, xt3, yt3, w, L)
        
#         dN_gain[k2,0] = dN_gain_temp.copy() 
#         dM_gain[k2,0] = dM_gain_temp.copy()
        
#         dM_gain[k2,1] = (dM_loss[k2]-dM_gain_temp)
#         dN_gain[k2,1] = (dNi_loss[k2]-dN_gain_temp)

        
#     # Condition 3:
#     #    k bin: Lower triangle region. Just clips UL corner.
#     #                     Triangle = ((xi1,xj1),(xi1,xj2),(x_bottom_edge,xj1))  
    

#     if len(k3)>0:
    
#         xt1 = x11[k3]
#         yt1 = x12[k3]
#         xt2 = x11[k3]
#         yt2 = x22[k3]
#         xt3 = x_bottom_edge[k3]
#         yt3 = x12[k3]
         
#         dN_gain_temp,dM_gain_temp = tri_gain_integrals(C[:,:,k3], xt1, yt1, xt2, yt2, xt3, yt3, w, L)
        
#         dN_gain[k3,0] = dN_gain_temp.copy()
#         dM_gain[k3,0] = dM_gain_temp.copy()
        
#         dM_gain[k3,1] = (dM_loss[k3]-dM_gain_temp)
#         dN_gain[k3,1] = (dNi_loss[k3]-dN_gain_temp)
            
#     # Condition 5: 
        
#     #    k bin: Top/Bottom clip: Rectangle on left, triangle on right
#     #                              Rectangle = ((xi1,xj1),(xi1,xj2),(x_top_edge,xj2),(x_top_edge,xj1))
#     #                              Triangle  = ((x_top_edge,xj1),(x_top_edge,xj2),(x_bottom_edge,xj1))
   
#     if len(k5)>0: 
       
#         xr1 = x11[k5]
#         yr1 = x12[k5]
#         xr2 = x_top_edge[k5]
#         yr2 = x22[k5]
       
#         xt1 = x_top_edge[k5]
#         yt1 = x12[k5]
#         xt2 = x_top_edge[k5]
#         yt2 = x22[k5]
#         xt3 = x_bottom_edge[k5]
#         yt3 = x12[k5]
        
#         rect_000, rect_001 = rect_gain_integrals(C[:,:,k5],x1=xr1,x2=xr2,y1=yr1,y2=yr2)
#         tri_000, tri_001   =  tri_gain_integrals(C[:,:,k5],xt1, yt1, xt2, yt2, xt3, yt3, w, L)
        
#         dN_gain_temp = rect_000+tri_000
#         dM_gain_temp = rect_001+tri_001
        
#         dM_gain[k5,0] = dM_gain_temp.copy()
#         dN_gain[k5,0] = dN_gain_temp.copy()
        
#         dM_gain[k5,1] = (dM_loss[k5]-dM_gain_temp)
#         dN_gain[k5,1] = (dNi_loss[k5]-dN_gain_temp)
        
#     # Condition 6:
#     # k bin: Left/Right clip: Rectangle on bottom, triangle on top
#     #                          Rectangle = ((xi1,xj1),(xi1,y_right_edge),(xi2,y_right_edge),(xi2,xj1))
#     #                          Triangle  = ((xi1,y_right_edge),(xi1,y_left_edge),(xi2,y_right_edge))
       
#     if len(k6)>0:
#         xr1 = x11[k6]
#         yr1 = x12[k6]
#         xr2 = x21[k6]
#         yr2 = y_right_edge[k6]
        
#         xt1 = x11[k6]
#         yt1 = y_right_edge[k6]
#         xt2 = x11[k6]
#         yt2 = y_left_edge[k6]
#         xt3 = x21[k6]
#         yt3 = y_right_edge[k6]
           
#         rect_000, rect_001 = rect_gain_integrals(C[:,:,k6],x1=xr1,x2=xr2,y1=yr1,y2=yr2)
#         tri_000, tri_001   =  tri_gain_integrals(C[:,:,k6], xt1, yt1, xt2, yt2, xt3, yt3, w, L)
        
#         dN_gain_temp = rect_000+tri_000 
#         dM_gain_temp = rect_001+tri_001
         
#         dM_gain[k6,0] = dM_gain_temp.copy()
#         dM_gain[k6,1] = (dM_loss[k6]-dM_gain_temp)
            
#         dN_gain[k6,0] = dN_gain_temp.copy()
#         dN_gain[k6,1] = (dNi_loss[k6]-dN_gain_temp)
        
#     # Condition 7:
#     # k+1 bin: Triangle in top right corner
#     #                            Triangle = ((x_top_edge,xj2),(xi2,xj2),(xi2,y_right_edge))
    
#     if len(k7)>0: 
#         xt1 = x_top_edge[k7]
#         yt1 = x22[k7]
#         xt2 = x21[k7]
#         yt2 = x22[k7]
#         xt3 = x21[k7]
#         yt3 = y_right_edge[k7]
         
#         dN_gain_temp,dM_gain_temp = tri_gain_integrals(C[:,:,k7], xt1, yt1, xt2, yt2, xt3, yt3, w, L)
        
#         dN_gain[k7,1] = dN_gain_temp.copy()
#         dM_gain[k7,1] = dM_gain_temp.copy()
        
#         dM_gain[k7,0] = (dM_loss[k7]-dM_gain_temp)
#         dN_gain[k7,0] = (dNi_loss[k7]-dN_gain_temp)
       
#     # Condition 8:
#     #  k bin: Triangle in lower left corner
#     #                                Triangle = ((xi1,xj1),(xi1,y_left_edge),(x_bottom_edge,xj1))
#     if len(k8)>0:
#         xt1 = x11[k8]
#         yt1 = x12[k8]
#         xt2 = x11[k8]
#         yt2 = y_left_edge[k8]
#         xt3 = x_bottom_edge[k8]
#         yt3 = x12[k8]
        
#         dN_gain_temp,dM_gain_temp = tri_gain_integrals(C[:,:,k8], xt1, yt1, xt2, yt2, xt3, yt3, w, L)

#         dN_gain[k8,0] = dN_gain_temp.copy()
#         dM_gain[k8,0] = dM_gain_temp.copy()
        
#         dM_gain[k8,1] = (dM_loss[k8]-dM_gain_temp)
#         dN_gain[k8,1] = (dNi_loss[k8]-dN_gain_temp)
        
#     # Condition 9: Rectangle collection within k bin. All Mass/Number goes into kbin

#     xr1 = x11[k9]
#     xr2 = x21[k9] 
#     yr1 = x12[k9]
#     yr2 = x22[k9]

#     dM_gain[k9,0]  = dM_loss[k9].copy()
#     dN_gain[k9,0]  = dNi_loss[k9].copy()

#     # Condition 10: Rectangle collection within k+1 bin. All Mass/Number goes into k+1bin
    
#     xr1 = x11[k10]
#     xr2 = x21[k10] 
#     yr1 = x12[k10]
#     yr2 = x22[k10]

#     dM_gain[k10,1]  = dM_loss[k10].copy()
#     dN_gain[k10,1]  = dNi_loss[k10].copy()
   
#     # Condition 11 (2b): Triangle collection within k+1 bin. Occurs when xk+1 clips BR corner and intersects top edge.
#     # Triangle = ((xi2,xj1),(x_top_edge,xj2),(xi2,xj2))
    
#     if len(k2b)>0:
#         xt1 = x_top_edge[k2b]
#         yt1 = x22[k2b]
#         xt2 = x21[k2b]
#         yt2 = x22[k2b]
#         xt3 = x21[k2b]
#         yt3 = x12[k2b]
        
#         dN_gain_temp,dM_gain_temp = tri_gain_integrals(C[:,:,k2b], xt1, yt1, xt2, yt2, xt3, yt3, w, L)
        
#         dN_gain[k2b,1] = dN_gain_temp.copy() 
#         dM_gain[k2b,1] = dM_gain_temp.copy()
        
#         dM_gain[k2b,0] = (dM_loss[k2b]-dM_gain_temp)
#         dN_gain[k2b,0] = (dNi_loss[k2b]-dN_gain_temp)
        
#     # Condition 12 (3b): Triangle collection within k+1 bin. Occurs when xk+1 clips UL corner and intersects with right edge.
#     # Triangle = ((xi1,xj2),(xi2,xj2),(xi2,y_right_edge)
#     if len(k3b)>0:
#         xt1 = x11[k3b]
#         yt1 = x22[k3b]
#         xt2 = x21[k3b]
#         yt2 = x22[k3b]
#         xt3 = x21[k3b]
#         yt3 = y_right_edge[k3b]
        
#         dN_gain_temp,dM_gain_temp = tri_gain_integrals(C[:,:,k3b], xt1, yt1, xt2, yt2, xt3, yt3, w, L)
        
#         dN_gain[k3b,1] = dN_gain_temp.copy()
#         dM_gain[k3b,1] = dM_gain_temp.copy()
        
#         dM_gain[k3b,0] = (dM_loss[k3b]-dM_gain_temp)
#         dN_gain[k3b,0] = (dNi_loss[k3b]-dN_gain_temp)  
    
#     # DO TRANSFER HERE
#     # Initialize gain term arrays
#     M1_loss = np.zeros((Hlen,bins))
#     M2_loss = np.zeros((Hlen,bins))
#     M_gain  = np.zeros((Hlen,bins))
#     Mb_gain = np.zeros((Hlen,bins))

#     N1_loss = np.zeros((Hlen,bins))
#     N2_loss = np.zeros((Hlen,bins))
#     N_gain  = np.zeros((Hlen,bins))
#     Nb_gain = np.zeros((Hlen,bins))
  
#     np.add.at(M1_loss,(kr,ir),dMi_loss)
#     np.add.at(M2_loss,(kr,jr),dMj_loss)

#     np.add.at(M_gain,(kr,kmin),dM_gain[:,0])
#     np.add.at(M_gain,(kr,kmid),dM_gain[:,1])
    
#     np.add.at(N1_loss,(kr,ir),dNi_loss)
#     np.add.at(N2_loss,(kr,jr),dNi_loss)
    
#     np.add.at(N_gain,(kr,kmin),dN_gain[:,0])
#     np.add.at(N_gain,(kr,kmid),dN_gain[:,1])
      
#     # ELD NOTE: Breakup here can take losses from each pair and calculate gains
#     # for breakup. Breakup gain arrays will be 3D.
#     if breakup:
                
#         np.add.at(Mb_gain,  kr, np.transpose(dMb_gain_frac[:,kmin]*dM_loss))
#         np.add.at(Nb_gain,  kr, np.transpose(dNb_gain_frac[:,kmin]*dM_loss))
    
#     #return dMi_loss, dMj_loss, dM_gain, dNi_loss, dN_gain

#     return M1_loss, M2_loss, M_gain, Mb_gain, N1_loss, N2_loss, N_gain, Nb_gain
    


    # def unpack_1mom(self):
    #     '''
    #     Updates 2D array (dist_num x Height) of distribution objects with new 
    #     distribution parameters calculated from Interaction object.

    #     Returns
    #     -------
    #     None.

    #     '''
        
    #     # NOTE: Make sure to align with 2 moment unpack.
        
    #     for zz in range(self.Hlen):
    #         for d1 in range(self.dnum):
    #             self.dists[d1,zz].Mbins  = self.Mbins[d1,zz,:].copy()
    #             self.dists[d1,zz].Nbins  = self.Nbins[d1,zz,:].copy()
    #             self.dists[d1,zz].diagnose_1mom()
    
    
    # def unpack(self):
    #     '''
    #     Updates 2D array (dist_num x Height) of distribution objects with new 
    #     distribution parameters calculated from Interaction object.

    #     Returns
    #     -------
    #     None.

    #     '''
        
    #     for zz in range(self.Hlen):
    #         for d1 in range(self.dnum):
    #             self.dists[d1,zz].Mbins  = self.Mbins[d1,zz,:].copy()
    #             self.dists[d1,zz].Nbins  = self.Nbins[d1,zz,:].copy()
    #             self.dists[d1,zz].diagnose()
    
    # def pack(self,dists):
    #     '''
    #     Updates Ikernel with (dist_num x height x bins) array of distribution
    #     parameters

    #     Parameters
    #     ----------
    #     dists : Object
    #         3D Array of distribution objects

    #     Returns
    #     -------
    #     None.

    #     '''
        
    #     # Create a flat list of the attributes, then reshape to the desired (dnum, Hlen, bins)
    #     self.Mbins = np.array([[d.Mbins for d in row] for row in dists])
    #     self.Nbins = np.array([[d.Nbins for d in row] for row in dists])
    #     self.Mfbins = np.array([[d.Mfbins for d in row] for row in dists])
    #     self.Nfbins = np.array([[d.Nfbins for d in row] for row in dists])
    #     self.x1 = np.array([[d.x1 for d in row] for row in dists])
    #     self.x2 = np.array([[d.x2 for d in row] for row in dists])
    #     self.aki = np.array([[d.aki for d in row] for row in dists])
    #     self.cki = np.array([[d.cki for d in row] for row in dists])
    
    
    # def setup_fragments_OLD(self):
            
    #         if self.kernel=='Hydro':
            
    #             if self.frag_dict['dist']=='exp':
    #                 IF_func = lambda n,x1,x2: gam_int(n,0.,self.frag_dict['Dmf'],x1,x2)
    #             elif self.frag_dict['dist']=='gamma':
    #                 IF_func = lambda n,x1,x2: gam_int(n,self.frag_dict['muf'],self.frag_dict['Dmf'],x1,x2)   
    #             elif self.frag_dict['dist']=='LGN':
    #                 # TEMP
    #                 Df_med = self.frag_dict['Df_med'] # mm
    #                 muf = np.log(Df_med)
    #                 Df_mode = self.frag_dict['Df_mode']
    #                 sig2f = muf-np.log(Df_mode)
    #                 IF_func = lambda n,x1,x2:LGN_int(n,muf,sig2f,x1,x2)
            
    #             for xx in range(self.bins):
    #                 for kk in range(0,xx+1):
    #                     self.dMb_gain_frac[kk,xx] = self.dists[self.indb,0].am*IF_func(self.dists[self.indb,0].bm,self.dists[self.indb,0].d1[kk],self.dists[self.indb,0].d2[kk])
    #                     self.dNb_gain_frac[kk,xx] = IF_func(0.,self.dists[self.indb,0].d1[kk],self.dists[self.indb,0].d2[kk])
                  
    #         else:
    #             for xx in range(self.bins): # m1+m2 breakup mass
    #                for kk in range(0,xx+1): # breakup gain bins
    #                    self.dMb_gain_frac[kk,xx] = In_int(1.,self.frag_dict['lamf'],self.xi1[kk],self.xi2[kk])
    #                    self.dNb_gain_frac[kk,xx] = In_int(0.,self.frag_dict['lamf'],self.xi1[kk],self.xi2[kk])   
                           
    #         self.dMb_gain_frac[self.dMb_gain_frac<0.]=0.
    #         dMb_gain_tot = np.nansum(self.dMb_gain_frac,axis=0)
    #         dMb_gain_tot[dMb_gain_tot==0.] = np.nan
            
    #         self.dMb_gain_frac = self.dMb_gain_frac/dMb_gain_tot[None,:]
    #         self.dNb_gain_frac = self.dNb_gain_frac/dMb_gain_tot[None,:]
            
    #         self.dMb_gain_frac[np.isnan(self.dMb_gain_frac)|np.isnan(self.dNb_gain_frac)] = 0.
    #         self.dNb_gain_frac[np.isnan(self.dMb_gain_frac)|np.isnan(self.dNb_gain_frac)] = 0.
    
    
   # WORKING
   #def calculate_2mom_batch(Hlen,bins,kr,ir,jr,x11,x21,ak1,ck1, 
   #                   x12,x22,ak2,ck2,PK,xi1,xi2,xk_min,kmin,kmid,
   #                   dMb_gain_frac,dNb_gain_frac,w,L,breakup=False):
       
   # def calculate_2mom_batch(Hlen,bins,region_inds,x_bottom_edge,x_top_edge,y_left_edge,y_right_edge,
   #                          kr,ir,jr,x11,x21,ak1,ck1,x12,x22,ak2,ck2,PK,kmin,kmid,
   #                          dMb_gain_frac,dNb_gain_frac,w,L,breakup=False):      

   #     '''
   #      This function calculate mass and number transfer rates
   #      for collision-coalescence and collisional breakup between
   #      each distribution for 2 moment calculations (mass + number).
        
   #      NOTE: NEW SETUP!
        
   #     '''
       
   #     # Set up integration regions for time-dependent subgrid linear distributions
   #     # for each batch bin-pair interaction.
   #     #regions = setup_regions_batch(bins,kr,ir,jr,x11,x21,x12,x22,xk_min)
       
       
       
   #     # Calculate integration regions to determine CC loss/gain integrals
   #     # for each batch bin-pair interaction.
   #     # output size is: (batch,batch). 
   #     #dMi_loss, dMj_loss, dM_gain, dNi_loss, dN_gain = calculate_regions_batch(Hlen,bins,kr,ir,jr,x11,x21,ak1,ck1,x12,x22,ak2,ck2,PK,region,w,L)
       
   #     dMi_loss, dMj_loss, dM_gain, dNi_loss, dN_gain = calculate_regions_batch(Hlen,bins,kr,ir,jr,x11,x21,ak1,ck1,x12,x22,ak2,ck2,PK,region_inds,
   #                                                                              x_bottom_edge,x_top_edge,y_left_edge,y_right_edge,w,L)
       
   #     #M1_loss = np.zeros((Hlen,bins))
   #     #N1_loss = np.zeros((Hlen,bins))

   #     #M2_loss = np.zeros((Hlen,bins))
   #     #N2_loss = np.zeros((Hlen,bins))
       
   #     #print('dMi_loss=',dMi_loss.shape)
   #     #raise Exception()
       
   #     ## NEW?
       
   #     # M1_loss = np.nansum(dMi_loss,axis=2) # Loss of dist1 mass with collisions from dist2
   #     # N1_loss = np.nansum(dNi_loss,axis=2) # Loss of dist1 number with collisions from dist2
       
   #     # M2_loss = np.nansum(dMj_loss,axis=1) # Loss of dist2 mass with collisions from dist1
   #     # N2_loss = np.nansum(dNi_loss,axis=1) # Loss of dist2 number with collisions from dist1
       
   #     #np.add.at(M1_loss,(kr,ir),dMi_loss)
   #     #np.add.at(N1_loss,(kr,ir),dNi_loss)
       
   #     #np.add.at(M2_loss,(kr,jr),dMj_loss)
   #     #np.add.at(N2_loss,(kr,jr),dNi_loss)
       
   #     #M_gain = np.nansum(dM_gain,axis=())
       
   #     # ChatGPT is the GOAT for telling me about np.add.at!
   #     # M_gain = np.zeros((Hlen,bins))
   #     # np.add.at(M_gain, (kr,kmin), np.nansum(dM_gain[:,:,:,0],axis=(1,2)))
   #     # np.add.at(M_gain, (kr,kmid), np.nansum(dM_gain[:,:,:,1],axis=(1,2)))
       
   #     # N_gain = np.zeros((Hlen,bins))
   #     # np.add.at(N_gain,  (kr,kmin), np.nansum(dN_gain[:,:,:,0],axis=(1,2)))
   #     # np.add.at(N_gain,  (kr,kmid), np.nansum(dN_gain[:,:,:,1],axis=(1,2)))
       
   #     ## NEW?
       
   #     ### OLD ###
   #     # (Hlen,bins,bins)
   #     M1_loss = np.nansum(dMi_loss,axis=2) # Loss of dist1 mass with collisions from dist2
   #     N1_loss = np.nansum(dNi_loss,axis=2) # Loss of dist1 number with collisions from dist2
       
   #     M2_loss = np.nansum(dMj_loss,axis=1) # Loss of dist2 mass with collisions from dist1
   #     N2_loss = np.nansum(dNi_loss,axis=1) # Loss of dist2 number with collisions from dist1
         
   #     # ChatGPT is the GOAT for telling me about np.add.at!
   #     M_gain = np.zeros((Hlen,bins))
   #     np.add.at(M_gain, (np.arange(Hlen)[:,None,None],kmin), dM_gain[:,:,:,0])
   #     np.add.at(M_gain, (np.arange(Hlen)[:,None,None],kmid), dM_gain[:,:,:,1])
       
   #     N_gain = np.zeros((Hlen,bins))
   #     np.add.at(N_gain,  (np.arange(Hlen)[:,None,None],kmin), dN_gain[:,:,:,0])
   #     np.add.at(N_gain,  (np.arange(Hlen)[:,None,None],kmid), dN_gain[:,:,:,1])
   #     ### OLD ###
       
   #     # Initialize gain term arrays
   #     Mb_gain  = np.zeros((Hlen,bins))
   #     Nb_gain  = np.zeros((Hlen,bins))
         
   #     # ELD NOTE: Breakup here can take losses from each pair and calculate gains
   #     # for breakup. Breakup gain arrays will be 3D.
   #     if breakup:
                   
   #         Mij_loss = dMi_loss[kr,ir,jr]+dMj_loss[kr,ir,jr]
     
   #         np.add.at(Mb_gain,  kr, np.transpose(dMb_gain_frac[:,kmin[ir,jr]]*Mij_loss))
   #         np.add.at(Nb_gain,  kr, np.transpose(dNb_gain_frac[:,kmin[ir,jr]]*Mij_loss))
           
      
   #     return M1_loss, M2_loss, M_gain, Mb_gain, N1_loss, N2_loss, N_gain, Nb_gain 
 
    
 # def calculate_regions_batch_WORKING(Hlen,bins,kr,ir,jr,x11,x21,ak1,ck1,x12,x22,ak2,ck2,PK,regions,w,L):    

 #     # kr = Height index    (batch,)
 #     # ir = collectee index (batch,)
 #     # jr = collector index (batch,) 

 #     '''
 #     Vectorized Integration Regions:
 #     cond_1 :  All rectangular bin-pair interactions used
 #     cond_2 :  k bin: Lower triangle region. Just clips BR corner.
 #                        Triangle = ((xi1,xj1),(xi1,y_left_edge),(xi2,xj1))
 #     cond_3 :  k bin: Lower triangle region. Just clips UL corner.
 #                        Triangle = ((xi1,xj1),(xi1,xj2),(x_bottom_edge,xj1))                 
 #     cond_2b : k+1 bins: Upper triangle region. Just clips BR corner.
 #                        Triangle = ((xi1,xj1),(x_top_edge,xj2),(xi2,xj2))
 #     cond_3b : k+1 bin: Upper triangle region. Just clips UL corner.
 #                              Triangle = ((xi1,xj2),(xi2,xj2),(xi2,y_right_edge)                    
 #     cond_4 :  Full Rectangular source region based on self collection: ii == jj --> ii+sbin or jj+sbin
 #     cond_5 :  k bin: Top/Bottom clip: Rectangle on left, triangle on right
 #                               Rectangle = ((xi1,xj1),(xi1,xj2),(x_top_edge,xj2),(x_top_edge,xj1))
 #                               Triangle  = ((x_top_edge,xj1),(x_top_edge,xj2),(x_bottom_edge,xj1))
 #     cond_6 :  k bin: Left/Right clip: Rectangle on bottom, triangle on top
 #                               Rectangle = ((xi1,xj1),(xi1,y_right_edge),(xi2,y_right_edge),(xi2,xj1))
 #                               Triangle  = ((xi1,y_right_edge),(xi1,y_left_edge),(xi2,y_right_edge))
 #     cond_7 :  k+1 bin: Triangle in top right corner
 #                                 Triangle = ((x_top_edge,xj2),(xi2,xj2),(xi2,y_right_edge))
 #     cond_8 :  k bin: Triangle in lower left corner
 #                                 Triangle = ((xi1,xj1),(xi1,y_left_edge),(x_bottom_edge,xj1))
 #     cond_9:   k bin: Rectangle collection within k bin. All Mass/Number goes into kbin
 #     cond_10:  k+1 bin: Rectangle collection within k+1 bin. All Mass/Number goes into kbin
 #     '''
     
 #     x_bottom_edge = regions['x_bottom_edge']
 #     x_top_edge = regions['x_top_edge']
 #     y_left_edge = regions['y_left_edge']
 #     y_right_edge = regions['y_right_edge']

 #     k2 = regions['2']
 #     k2b = regions['2b']
 #     k3 = regions['3']
 #     k3b = regions['3b']
 #     k4 = regions['4']
 #     k5 = regions['5']
 #     k6 = regions['6']
 #     k7 = regions['7']
 #     k8 = regions['8']
 #     k9 = regions['9']
 #     k10 = regions['10']
     
 #     # Initialize gain term arrays
 #     dMi_loss = np.zeros((Hlen,bins,bins))
 #     dMj_loss = np.zeros((Hlen,bins,bins))
 #     dNi_loss = np.zeros((Hlen,bins,bins))
 #     dM_gain  = np.zeros((Hlen,bins,bins,2))
 #     dN_gain  = np.zeros((Hlen,bins,bins,2))

 #     # NOTE PRECOMPUTE MASS/NUMBER COEFFICIENTS HERE FOR ALL BIN-PAIR INTERACTIONS
 #     # This is: F(x,y)* Nx(x) * Ny(y) = (F=a+b*x+c*y+d*x*y)*(Nx=ax*x+cx)*(Ny=ay*y+cy)
 #     C = combined_coeffs_array(PK, ak1, ck1, ak2, ck2)
   
 #     # Calculate transfer rates (rectangular integration, source space)
 #     # Collection (eqs. 23-25 in Wang et al. 2007)
 #     # ii collecting jj 
     
     
 #     # NOTE: When batching, need to make sure that k1 corresponds to all k2-k3b indices.
 #     # This will be important for balance loading all batches while maining that the source
 #     # bin-pair loss values can be used in some of the gain calculations (e.g., region 4)

 #     # Calculate source bin-pair rectangle integral factors. Note, tried to
 #     # combine as many of these factors together as possible to avoid redundant calculations.
 #     dNi_full, dMi_full, dMj_full, = source_integrals(C,x1=x11,x2=x21,y1=x12,y2=x22)
     
 #     dMi_loss[kr,ir,jr] = dMi_full.copy() 
 #     dMj_loss[kr,ir,jr] = dMj_full.copy() 
 #     dNi_loss[kr,ir,jr] = dNi_full.copy() 
     
 #     dM_loss = dMi_loss+dMj_loss
     
 #     dM_gain[kr[k4],ir[k4],jr[k4],0]  = dM_loss[kr[k4],ir[k4],jr[k4]].copy()
 #     dN_gain[kr[k4],ir[k4],jr[k4],0]  = dNi_loss[kr[k4],ir[k4],jr[k4]].copy()

 #     # Condition 2:
 #     # k bin: Lower triangle region. Just clips BR corner.
 #     #                       Triangle = ((xi1,xj1),(xi1,y_left_edge),(xi2,xj1))
 #     if len(k2)>0:
 #         xt1 = x11[k2]
 #         yt1 = x12[k2]
 #         xt2 = x11[k2]
 #         yt2 = y_left_edge[k2]
 #         xt3 = x21[k2]
 #         yt3 = x12[k2]
         
 #         dN_gain_temp, dM_gain_temp = tri_gain_integrals(C[:,:,k2], xt1, yt1, xt2, yt2, xt3, yt3, w, L)
         
 #         dN_gain[kr[k2],ir[k2],jr[k2],0] = dN_gain_temp.copy() 
 #         dM_gain[kr[k2],ir[k2],jr[k2],0] = dM_gain_temp.copy()
         
 #         dM_gain[kr[k2],ir[k2],jr[k2],1] = dM_loss[kr[k2],ir[k2],jr[k2]]-dM_gain[kr[k2],ir[k2],jr[k2],0]
 #         dN_gain[kr[k2],ir[k2],jr[k2],1] = dNi_loss[kr[k2],ir[k2],jr[k2]]-dN_gain[kr[k2],ir[k2],jr[k2],0]

         
 #     # Condition 3:
 #     #    k bin: Lower triangle region. Just clips UL corner.
 #     #                     Triangle = ((xi1,xj1),(xi1,xj2),(x_bottom_edge,xj1))  
     

 #     if len(k3)>0:
     
 #         xt1 = x11[k3]
 #         yt1 = x12[k3]
 #         xt2 = x11[k3]
 #         yt2 = x22[k3]
 #         xt3 = x_bottom_edge[k3]
 #         yt3 = x12[k3]
          
 #         dN_gain_temp,dM_gain_temp = tri_gain_integrals(C[:,:,k3], xt1, yt1, xt2, yt2, xt3, yt3, w, L)
         
 #         dN_gain[kr[k3],ir[k3],jr[k3],0] = dN_gain_temp.copy()
 #         dM_gain[kr[k3],ir[k3],jr[k3],0] = dM_gain_temp.copy()
         
 #         dM_gain[kr[k3],ir[k3],jr[k3],1] = dM_loss[kr[k3],ir[k3],jr[k3]]-dM_gain[kr[k3],ir[k3],jr[k3],0]
 #         dN_gain[kr[k3],ir[k3],jr[k3],1] = dNi_loss[kr[k3],ir[k3],jr[k3]]-dN_gain[kr[k3],ir[k3],jr[k3],0]
             
     
 #     # Condition 5: 
         
 #     #    k bin: Top/Bottom clip: Rectangle on left, triangle on right
 #     #                              Rectangle = ((xi1,xj1),(xi1,xj2),(x_top_edge,xj2),(x_top_edge,xj1))
 #     #                              Triangle  = ((x_top_edge,xj1),(x_top_edge,xj2),(x_bottom_edge,xj1))
    
 #     if len(k5)>0: 
        
 #         xr1 = x11[k5]
 #         yr1 = x12[k5]
 #         xr2 = x_top_edge[k5]
 #         yr2 = x22[k5]
        
 #         xt1 = x_top_edge[k5]
 #         yt1 = x12[k5]
 #         xt2 = x_top_edge[k5]
 #         yt2 = x22[k5]
 #         xt3 = x_bottom_edge[k5]
 #         yt3 = x12[k5]
         
 #         rect_000, rect_001 = rect_gain_integrals(C[:,:,k5],x1=xr1,x2=xr2,y1=yr1,y2=yr2)
 #         tri_000, tri_001   =  tri_gain_integrals(C[:,:,k5],xt1, yt1, xt2, yt2, xt3, yt3, w, L)
         
         
 #         dM_gain[kr[k5],ir[k5],jr[k5],0] = rect_001+tri_001
         
 #         dM_gain[kr[k5],ir[k5],jr[k5],1] = dM_loss[kr[k5],ir[k5],jr[k5]]-dM_gain[kr[k5],ir[k5],jr[k5],0]
             
 #         dN_gain[kr[k5],ir[k5],jr[k5],0] = rect_000+tri_000
                            
 #         dN_gain[kr[k5],ir[k5],jr[k5],1] = dNi_loss[kr[k5],ir[k5],jr[k5]]-dN_gain[kr[k5],ir[k5],jr[k5],0]
         
 #     # Condition 6:
 #     # k bin: Left/Right clip: Rectangle on bottom, triangle on top
 #     #                          Rectangle = ((xi1,xj1),(xi1,y_right_edge),(xi2,y_right_edge),(xi2,xj1))
 #     #                          Triangle  = ((xi1,y_right_edge),(xi1,y_left_edge),(xi2,y_right_edge))
        
 #     if len(k6)>0:
 #         xr1 = x11[k6]
 #         yr1 = x12[k6]
 #         xr2 = x21[k6]
 #         yr2 = y_right_edge[k6]
         
 #         xt1 = x11[k6]
 #         yt1 = y_right_edge[k6]
 #         xt2 = x11[k6]
 #         yt2 = y_left_edge[k6]
 #         xt3 = x21[k6]
 #         yt3 = y_right_edge[k6]
            
 #         rect_000, rect_001 = rect_gain_integrals(C[:,:,k6],x1=xr1,x2=xr2,y1=yr1,y2=yr2)
 #         tri_000, tri_001   =  tri_gain_integrals(C[:,:,k6], xt1, yt1, xt2, yt2, xt3, yt3, w, L)
          
 #         dM_gain[kr[k6],ir[k6],jr[k6],0] = rect_001 + tri_001
 #         dM_gain[kr[k6],ir[k6],jr[k6],1] = dM_loss[kr[k6],ir[k6],jr[k6]]-dM_gain[kr[k6],ir[k6],jr[k6],0]
             
 #         dN_gain[kr[k6],ir[k6],jr[k6],0] = rect_000 + tri_000
 #         dN_gain[kr[k6],ir[k6],jr[k6],1] = dNi_loss[kr[k6],ir[k6],jr[k6]]-dN_gain[kr[k6],ir[k6],jr[k6],0]
         
 #     # Condition 7:
 #     # k+1 bin: Triangle in top right corner
 #     #                            Triangle = ((x_top_edge,xj2),(xi2,xj2),(xi2,y_right_edge))
     
 #     if len(k7)>0: 
 #         xt1 = x_top_edge[k7]
 #         yt1 = x22[k7]
 #         xt2 = x21[k7]
 #         yt2 = x22[k7]
 #         xt3 = x21[k7]
 #         yt3 = y_right_edge[k7]
          
 #         dN_gain_temp,dM_gain_temp = tri_gain_integrals(C[:,:,k7], xt1, yt1, xt2, yt2, xt3, yt3, w, L)
         
         
 #         dN_gain[kr[k7],ir[k7],jr[k7],1] = dN_gain_temp.copy()
 #         dM_gain[kr[k7],ir[k7],jr[k7],1] = dM_gain_temp.copy()
         
 #         dM_gain[kr[k7],ir[k7],jr[k7],0] = dM_loss[kr[k7],ir[k7],jr[k7]]-dM_gain[kr[k7],ir[k7],jr[k7],1]
         
 #         dN_gain[kr[k7],ir[k7],jr[k7],0] = dNi_loss[kr[k7],ir[k7],jr[k7]]-dN_gain[kr[k7],ir[k7],jr[k7],1]
        
 #     # Condition 8:
 #     #  k bin: Triangle in lower left corner
 #     #                                Triangle = ((xi1,xj1),(xi1,y_left_edge),(x_bottom_edge,xj1))
 #     if len(k8)>0:
 #         xt1 = x11[k8]
 #         yt1 = x12[k8]
 #         xt2 = x11[k8]
 #         yt2 = y_left_edge[k8]
 #         xt3 = x_bottom_edge[k8]
 #         yt3 = x12[k8]
         
 #         dN_gain_temp,dM_gain_temp = tri_gain_integrals(C[:,:,k8], xt1, yt1, xt2, yt2, xt3, yt3, w, L)
         
         
 #         dN_gain[kr[k8],ir[k8],jr[k8],0] = dN_gain_temp.copy() 
 #         dM_gain[kr[k8],ir[k8],jr[k8],0] = dM_gain_temp.copy()
         
 #         dM_gain[kr[k8],ir[k8],jr[k8],1] = dM_loss[kr[k8],ir[k8],jr[k8]]-dM_gain[kr[k8],ir[k8],jr[k8],0]
 #         dN_gain[kr[k8],ir[k8],jr[k8],1] = dNi_loss[kr[k8],ir[k8],jr[k8]]-dN_gain[kr[k8],ir[k8],jr[k8],0]
         
 #     # Condition 9: Rectangle collection within k bin. All Mass/Number goes into kbin

 #     xr1 = x11[k9]
 #     xr2 = x21[k9] 
 #     yr1 = x12[k9]
 #     yr2 = x22[k9]

 #     dM_gain[kr[k9],ir[k9],jr[k9],0]  = dM_loss[kr[k9],ir[k9],jr[k9]].copy()
 #     dN_gain[kr[k9],ir[k9],jr[k9],0]  = dNi_loss[kr[k9],ir[k9],jr[k9]].copy()

 #     # Condition 10: Rectangle collection within k+1 bin. All Mass/Number goes into k+1bin
     
 #     xr1 = x11[k10]
 #     xr2 = x21[k10] 
 #     yr1 = x12[k10]
 #     yr2 = x22[k10]

 #     dM_gain[kr[k10],ir[k10],jr[k10],1]  = dM_loss[kr[k10],ir[k10],jr[k10]].copy()
 #     dN_gain[kr[k10],ir[k10],jr[k10],1]  = dNi_loss[kr[k10],ir[k10],jr[k10]].copy()
    
     
 #     # Condition 11 (2b): Triangle collection within k+1 bin. Occurs when xk+1 clips BR corner and intersects top edge.
 #     # Triangle = ((xi2,xj1),(x_top_edge,xj2),(xi2,xj2))
     
 #     if len(k2b)>0:
 #         xt1 = x_top_edge[k2b]
 #         yt1 = x22[k2b]
 #         xt2 = x21[k2b]
 #         yt2 = x22[k2b]
 #         xt3 = x21[k2b]
 #         yt3 = x12[k2b]
         
 #         dN_gain_temp,dM_gain_temp = tri_gain_integrals(C[:,:,k2b], xt1, yt1, xt2, yt2, xt3, yt3, w, L)
         
 #         dN_gain[kr[k2b],ir[k2b],jr[k2b],1] = dN_gain_temp.copy() 
 #         dM_gain[kr[k2b],ir[k2b],jr[k2b],1] = dM_gain_temp.copy()
         
 #         dM_gain[kr[k2b],ir[k2b],jr[k2b],0] = dM_loss[kr[k2b],ir[k2b],jr[k2b]]-dM_gain[kr[k2b],ir[k2b],jr[k2b],1]
 #         dN_gain[kr[k2b],ir[k2b],jr[k2b],0] = dNi_loss[kr[k2b],ir[k2b],jr[k2b]]-dN_gain[kr[k2b],ir[k2b],jr[k2b],1]
         
 #     # Condition 12 (3b): Triangle collection within k+1 bin. Occurs when xk+1 clips UL corner and intersects with right edge.
 #     # Triangle = ((xi1,xj2),(xi2,xj2),(xi2,y_right_edge)
 #     if len(k3b)>0:
 #         xt1 = x11[k3b]
 #         yt1 = x22[k3b]
 #         xt2 = x21[k3b]
 #         yt2 = x22[k3b]
 #         xt3 = x21[k3b]
 #         yt3 = y_right_edge[k3b]
         
 #         dN_gain_temp,dM_gain_temp = tri_gain_integrals(C[:,:,k3b], xt1, yt1, xt2, yt2, xt3, yt3, w, L)
         
 #         dN_gain[kr[k3b],ir[k3b],jr[k3b],1] = dN_gain_temp.copy() 
 #         dM_gain[kr[k3b],ir[k3b],jr[k3b],1] = dM_gain_temp.copy()
         
 #         dM_gain[kr[k3b],ir[k3b],jr[k3b],0] = dM_loss[kr[k3b],ir[k3b],jr[k3b]]-dM_gain[kr[k3b],ir[k3b],jr[k3b],1]
 #         dN_gain[kr[k3b],ir[k3b],jr[k3b],0] = dNi_loss[kr[k3b],ir[k3b],jr[k3b]]-dN_gain[kr[k3b],ir[k3b],jr[k3b],1]   
     
 #     return dMi_loss, dMj_loss, dM_gain, dNi_loss, dN_gain
 
 
 
 
 
            # if parallel:
            
            #     '''
            #     Create memory map files and variables that will be used for both 
            #     2 moment and 1 moment selections.
            #     '''
            
            #     self._setup_shared_tensors()
                
            #     self.h_batches = np.array_split(np.arange(self.Hlen), self.n_jobs)
            
            #self.worker_ids = np.arange(self.n_jobs)
        
        
       # elif self.mom_num==2:
      #      self.setup_2mom()
      
      
      
      
      
        
        # if parallel:
            
        #     '''
        #     Create memory map files and variables that will be used for both 
        #     2 moment and 1 moment selections.
        #     '''
            
        #     self._setup_shared_tensors()
            
            
        #     self.worker_ids = np.arange(self.n_jobs)
            
            # self.mem_map_list = ['dMb_gain_frac','dNb_gain_frac','PK','kmin',
            #                      'kmid','xi1','xi2','xk_min','cond1','self_col',
            #                      'w','L']
            
            # self.temp_dir = gettempdir()
        
            # dMb_gain_file = os.path.join(self.temp_dir,'dMb_gain_frac.pkl')
            # dNb_gain_file = os.path.join(self.temp_dir,'dNb_gain_frac.pkl')
            # PK_file       = os.path.join(self.temp_dir,'PK.pkl')
            # kmin_file     = os.path.join(self.temp_dir,'kmin.pkl')
            # kmid_file     = os.path.join(self.temp_dir,'kmid.pkl')
            # xi1_file      = os.path.join(self.temp_dir,'xi1.pkl')
            # xi2_file      = os.path.join(self.temp_dir,'xi2.pkl')
            # xk_min_file   = os.path.join(self.temp_dir,'xk_min.pkl')
            # cond1_file    = os.path.join(self.temp_dir,'cond_1.pkl')
            # selfcol_file  = os.path.join(self.temp_dir,'self_col.pkl')
            # w_file        = os.path.join(self.temp_dir,'w.pkl')
            # L_file        = os.path.join(self.temp_dir,'L.pkl')
            
            # dump(self.dMb_gain_frac,dMb_gain_file)
            # dump(self.dNb_gain_frac,dNb_gain_file)
            # dump(self.PK,PK_file)
            # dump(self.kmin,kmin_file)
            # dump(self.kmid,kmid_file)
            # dump(self.xi1,xi1_file)
            # dump(self.xi2,xi2_file)
            # dump(self.xk_min,xk_min_file)
            # dump(self.cond_1,cond1_file)
            # dump(self.self_col,selfcol_file)
            # dump(self.w,w_file)
            # dump(self.L,L_file)
 
            # self.dMb_gain_frac = load(dMb_gain_file,mmap_mode='r')
            # self.dNb_gain_frac = load(dNb_gain_file,mmap_mode='r')
            # self.PK            = load(PK_file,mmap_mode='r')       
            # self.kmin          = load(kmin_file,mmap_mode='r')
            # self.kmid          = load(kmid_file,mmap_mode='r')
            # self.xi1           = load(xi1_file,mmap_mode='r')
            # self.xi2           = load(xi2_file,mmap_mode='r')
            # self.xk_min        = load(xk_min_file,mmap_mode='r')
            # self.cond_1        = load(cond1_file,mmap_mode='r')
            # self.self_col      = load(selfcol_file,mmap_mode='r')
            # self.w             = load(w_file,mmap_mode='r')
            # self.L             = load(L_file,mmap_mode='r')
        
            # self.mem_map_dict = {'inputs':{},
            #                      'outputs':{}}
            
            #self._temp_files = [] # All temp file paths
            
            # if self.mom_num==1:
            #     outlen = 4 
            # elif self.mom_num==2:
            #     outlen = 8
            
            # self.mem_out = self._setup_buffer(outlen)
        
            # self.mem_map_dict['inputs'] = {'dMb_gain_frac':dMb_gain_file,
            #                      'dNb_gain_frac':dNb_gain_file,
            #                      'PK':PK_file,
            #                      'kmin':kmin_file,
            #                      'kmid':kmid_file,
            #                      'xi1':xi1_file,
            #                      'xi2':xi2_file,
            #                      'xk_min':xk_min_file,
            #                      'cond_1':cond1_file,
            #                      'self_col':selfcol_file,
            #                      'w':w_file,
            #                      'L':L_file}
    
        # # Calculate transfer rates if single-moment scheme is chosen.
        # if self.mom_num==1:  
            
        #     '''
        #         Using shared memory here to save arrays temporarily if processing
        #         in parallel.
            
        #     '''
        
        #     self.dMi_loss = np.zeros((self.pnum,self.bins,self.bins))
        #     self.dMj_loss = np.zeros((self.pnum,self.bins,self.bins))
        #     self.dM_loss = np.zeros((self.pnum,self.bins,self.bins))
        #     self.dM_gain = np.zeros((self.pnum,self.bins,self.bins,2))
            
        #     self.M1_loss_buffer = np.zeros((self.Hlen,self.bins))
        #     self.M2_loss_buffer = np.zeros((self.Hlen,self.bins))
        #     self.M_gain_buffer =  np.zeros((self.Hlen,self.bins))
        #     self.Mb_gain_buffer = np.zeros((self.Hlen,self.bins))
            
        #     self.M1_loss_buffer_flat = self.M1_loss_buffer.ravel()
        #     self.M2_loss_buffer_flat = self.M2_loss_buffer.ravel()
        #     self.M_gain_buffer_flat =  self.M_gain_buffer.ravel()
        #     self.Mb_gain_buffer_flat = self.Mb_gain_buffer.ravel()
            
        #     self.M_loss_tot_buffer = np.zeros_like(self.Mbins)
        #     self.M_gain_tot_buffer = np.zeros_like(self.Mbins)
            
        #     self.regions = np.empty((self.pnum,),dtype=object)
            
        #     #ak1mom = np.zeros((1,self.bins))
        #     #ck1mom = np.ones((1,self.bins))
            
        #     ak1mom = np.zeros((self.Hlen,self.bins))
        #     ck1mom = np.ones((self.Hlen,self.bins))
            
        #     dd = 0
        
        #     # Calculate regions and mass transfer rates
        #     for d1 in range(self.dnum):
        #         for d2 in range(d1,self.dnum):
                       
        #             # Get 3D indices for all bins that will interact at all heights
        #             # indices have shape (bin-pairs,) where ir are dist1 indices 
        #             # and jr are dist2 indices. kr are height indices.
        #             kr, ir, jr = np.nonzero((~self.cond_1)&self.self_col[:,dd,:,:])
                    
        #             # BATCHES
        #             # (bin-pair,)
        #             x11  = self.xi1[ir]
        #             x21  = self.xi2[ir]
        #             ak1  = ak1mom[kr,ir]
        #             ck1  = ck1mom[kr,ir]

        #             # (bin-pair,)
        #             x12  = self.xi1[jr]
        #             x22  = self.xi2[jr]
        #             ak2  = ak1mom[kr,jr]
        #             ck2  = ck1mom[kr,jr]
                    
        #             kmin = self.kmin[ir,jr]
        #             kmid = self.kmid[ir,jr]
                    
        #             PK = self.PK[:,dd,ir,jr]
                    
        #             xk_min = self.xi2[self.kmin[ir,jr]]
                    
        #             # NEW METHOD
        #             region_inds, x_bottom_edge, x_top_edge, y_left_edge, y_right_edge=setup_regions(self.bins,kr,ir,jr,x11,x21,x12,x22,xk_min)

        #             dMi_loss, dMj_loss, dM_loss, dM_gain, _, _ = calculate_rates(self.Hlen,self.bins,region_inds,x_bottom_edge,x_top_edge,y_left_edge,y_right_edge,
        #             kr,ir,jr,x11,x21,ak1,ck1,x12,x22,ak2,ck2,PK,kmin,kmid,self.dMb_gain_frac,self.dNb_gain_frac,self.w,self.L,
        #             breakup=self.breakup)

        #             self.dMi_loss[dd,ir,jr] = dMi_loss
        #             self.dMj_loss[dd,ir,jr] = dMj_loss 
        #             self.dM_loss[dd,ir,jr] = dM_loss
        #             self.dM_gain[dd,ir,jr,:] = dM_gain

                    # WORKING
                    #self.dMi_loss[dd,:,:,:], self.dMj_loss[dd,:,:,:], self.dM_gain[dd,:,:,:,:], _, _  = calculate_regions_batch(
                    #   self.Hlen,self.bins,kr,ir,jr,x11,x21,ak1,ck1,x12,x22,ak2,ck2,PK,region_inds,
                    #                   x_bottom_edge,x_top_edge,y_left_edge,y_right_edge,self.w,self.L)
                    
                    
                    # Current 2mom method
                    # dMi_loss, dMj_loss, dM_gain, _, _, _, _, _ = calculate_2mom_batch_NEW(
                    #     self.Hlen,self.bins,region_inds, x_bottom_edge,x_top_edge,y_left_edge,y_right_edge, 
                    #     kr,ir,jr,x11,x21,ak1,ck1,x12,x22,ak2,ck2,PK,self.kmin[ir,jr],self.kmid[ir,jr],
                    #     self.dMb_gain_frac,self.dNb_gain_frac,
                    #                    self.w,self.L)
                    
                    #print('dMi_loss=',dMi_loss.shape)
                    #raise Exception()
                    
                   # self.dMi_loss[dd,:,:,:] = dMi_loss 
                   # self.dMj_loss[dd,:,:,:] = dMj_loss 
                   # self.dM_gain[dd,:,:,:,:] = dM_gain
                    
                    
                    #dd += 1
        
        
            #if parallel:
                
                # # Write all arrays to memory map files
                # self.mem_map_list += ['dMi_loss','dMj_loss','dM_loss','dM_gain']
                
                # dMi_loss_file = os.path.join(self.temp_dir,'dMi_loss.pkl')
                # dMj_loss_file = os.path.join(self.temp_dir,'dMj_loss.pkl')  
                # dM_loss_file = os.path.join(self.temp_dir,'dM_loss.pkl')  
                # dM_gain_file  = os.path.join(self.temp_dir,'dM_gain.pkl')
                
                # dump(self.dMi_loss,dMi_loss_file)
                # dump(self.dMj_loss,dMj_loss_file)
                # dump(self.dM_loss,dM_loss_file)
                # dump(self.dM_gain,dM_gain_file)
                
                #self.dMi_loss = load(dMi_loss_file,mmap_mode='r')
                #self.dMj_loss = load(dMj_loss_file,mmap_mode='r')
                #self.dM_gain  = load(dM_gain_file,mmap_mode='r')
                
                # self.mem_map_dict['inputs']['dMi_loss'] = dMi_loss_file 
                # self.mem_map_dict['inputs']['dMj_loss'] = dMj_loss_file 
                # self.mem_map_dict['inputs']['dM_loss'] = dM_loss_file 
                # self.mem_map_dict['inputs']['dM_gain'] = dM_gain_file 
                
                
        # Setup bin-pair interactions
        #if self.mom_num==1:
        #    self.setup_1mom()
 
            
    # def setup_1mom_vectorized(self):
        
    #     # Setup Interaction precalculations
    #     # Determine all possible d1, d2 combinations
        
    #     ak1mom = np.zeros((self.Hlen,self.bins))
    #     ck1mom = np.ones((self.Hlen,self.bins))
        
    #     global_mask = (~self.cond_1[:,None,:,:]) & self.self_col
        
    #     kr, dr, ir, jr = np.nonzero(global_mask)
        
    #     total_interactions = len(kr)
        
    #     PK = self.PK[:,dr,ir,jr]
        
    #     # # 3. Allocate Structured Array
    #     # static_dtype = [
    #     # ('kr', 'i4'), ('ir', 'i4'), ('jr', 'i4'),
    #     # ('kmin', 'i4'), ('kmid', 'i4'), ('dr', 'i4'),
    #     # ('d1','i4'),('d2','i4'),('d1_flat','i4'),
    #     # ('d2_flat','i4'),('gain_kmin_flat','i4'),
    #     # ('gain_kmid_flat','i4')]
        
    #     # 3. Allocate Structured Array
    #     static_dtype = [
    #     ('kr', 'i4'), ('ir', 'i4'), ('jr', 'i4'),('dr','i4'),
    #     ('kmin', 'i4'), ('kmid', 'i4'),('d1','i4'),('d2','i4'),
    #     ('d1_flat','i4'),('d2_flat','i4')]
        
    #     self.static_full = np.empty(total_interactions, dtype=static_dtype)
    #     self.static_full['kr'] = kr
    #     self.static_full['dr'] = dr
    #     self.static_full['ir'] = ir
    #     self.static_full['jr'] = jr
        
    #     self.static_full['d1'] = self.d1_indices[dr]
    #     self.static_full['d2'] = self.d2_indices[dr]
        
    #     #self.static_full['d1_flat'] = (kr * self.bins)+ir
    #     #self.static_full['d2_flat'] = (kr * self.bins)+jr
        
        
    #     self.static_full['d1_flat'] = (self.static_full['d1']*self.Hlen*self.bins) + (kr * self.bins)+ir
    #     self.static_full['d2_flat'] = (self.static_full['d2']*self.Hlen*self.bins) +(kr * self.bins)+jr
     
    #     x11 = self.xi1[ir] 
    #     x21 = self.xi2[ir] 
    #     ak1  = ak1mom[kr,ir]
    #     ck1  = ck1mom[kr,ir]
        
    #     x12 = self.xi1[jr] 
    #     x22 = self.xi2[jr]
    #     ak2  = ak1mom[kr,jr]
    #     ck2  = ck1mom[kr,jr]
        
    #     kmin = self.kmin[ir,jr]
    #     kmid = self.kmid[ir,jr]
        
    #     self.static_full['kmin'] = kmin
    #     self.static_full['kmid'] = kmid
        
    #     #self.static_full['gain_kmin_flat'] = (kr * self.bins) + kmin
    #     #self.static_full['gain_kmid_flat'] = (kr * self.bins) + kmid 
        
        
    #     #bins_range = np.arange(self.bins)
    #     #self.static_full['gain_breakup_flat'] = (self.indb * self.Hlen*self.bins)+(kr * self.bins) + kmin
        
    #     xk_min = self.xi2[kmin]
        
    #     # NEW METHOD
    #     region_inds, x_bottom_edge, x_top_edge, y_left_edge, y_right_edge=setup_regions(self.bins,kr,ir,jr,x11,x21,x12,x22,xk_min)

    #     dMi_loss, dMj_loss, dM_loss, dM_gain, _, _ = calculate_rates(self.Hlen,self.bins,region_inds,x_bottom_edge,x_top_edge,y_left_edge,y_right_edge,
    #     kr,ir,jr,x11,x21,ak1,ck1,x12,x22,ak2,ck2,PK,kmin,kmid,self.dMb_gain_frac,self.dNb_gain_frac,self.w,self.L,
    #     breakup=self.breakup)
        
    #     self.dMi_loss[dr,ir,jr] = dMi_loss
    #     self.dMj_loss[dr,ir,jr] = dMj_loss 
    #     self.dM_loss[dr,ir,jr] = dM_loss
    #     self.dM_gain[dr,ir,jr,:] = dM_gain
        
    #     # 4. Pre-allocate Dynamic Buffer (for ck12 values)
    #     self.ck12_dynamic = np.zeros(total_interactions, dtype='f8')
         
    
    
    # def interact_1mom_vectorized(self, dt):
    #     #M_loss = np.zeros_like(self.Mbins)
    #     M_gain = np.zeros_like(self.Mbins)
        
    #     # Zero out buffers before calculations
    #     self.M_loss_buffer.fill(0.)
    #     self.M_gain_buffer.fill(0.)
    #     self.Mb_gain_buffer.fill(0.)
        
    #     indc = self.indc 
    #     indb = self.indb
        
    #     # --- OPTIMIZATION: One vectorized update per d-pair ---
    #     # Fetch indices from our pre-baked static buffer
    #     kr = self.static_full['kr']
    #     dr = self.static_full['dr']
    #     ir = self.static_full['ir']
    #     jr = self.static_full['jr']
    #     d1 = self.static_full['d1']
    #     d2 = self.static_full['d2']
        
    #     #active_h, active_b = np.where()
                   
    #     # Calculate ck12 only for active interactions
    #     # This is a contiguous operation now
    #     self.ck12_dynamic = self.cki[d1, kr, ir] * self.cki[d2, kr, jr]
                 
    #     active_indices = np.where(self.ck12_dynamic > 0)[0]

    #     if len(active_indices) == 0:
    #         return np.zeros_like(self.Mbins)
        
    #     # Slice buffers for this specific interaction set
    #     # Using the absolute slice 's' + the sub-filter 'active_indices'
    #     current_static = self.static_full[active_indices]
    #     ck12_curr = self.ck12_dynamic[active_indices]
                   
    #                 # 3. EXTRACTION (The I/O Buffer)
    #     # Pull from memmap into contiguous RAM blocks

    #     dMi_loss_curr = self.dMi_loss[dr[active_indices], ir[active_indices], jr[active_indices]]
    #     dMj_loss_curr = self.dMj_loss[dr[active_indices], ir[active_indices], jr[active_indices]]
    #     dM_loss_curr  = self.dM_loss[dr[active_indices], ir[active_indices], jr[active_indices]]
    #     dM_gain_curr  = self.dM_gain[dr[active_indices], ir[active_indices], jr[active_indices], :]
    #     # Handle kmin-based indexing for dMb
    #     kmin_curr = self.static_full['kmin'][active_indices]
    #     #kmid_curr = self.static_full['kmid'][s][active_indices]
    #     dMb_gain_frac_curr = self.dMb_gain_frac[:, kmin_curr]


    #     if self.parallel:
    #         # Chunking logic for Parallel
    #         #batches = np.array_split(np.arange(len(current_static)), self.n_jobs)
            
    #         # Assuming n_active is the number of interactions where ck12 > 0 
    #         chunk_size = int(np.ceil(len(current_static) / self.n_jobs))
    #         batches = [slice(i, i + chunk_size) for i in range(0, len(current_static), chunk_size)]
          
    #         # The Parallel Call 
    #         gain_loss_temp = self.pool(delayed(sum_1mom_batches_optimized)(
    #             current_static[batch], # Slice of structured array 
    #             ck12_curr[batch], 
    #             dMi_loss_curr[batch], 
    #             dMj_loss_curr[batch], 
    #             dM_gain_curr[batch,:],
    #             dMb_gain_frac_curr[:,batch],       # Vertical slice of 2D array 
    #             self.bins, 
    #             self.Hlen
    #             ) for batch in batches)
            
            
    #         # Combine results in the main process [cite: 1, 6]
    #         total_sum = np.sum(gain_loss_temp, axis=0) # Sum across chunks 

    #         # Reshape back to the 2D height-bin grid [cite: 1, 21]
    #         M1_loss_temp = total_sum[0].reshape(self.Hlen, self.bins)
    #         M2_loss_temp = total_sum[1].reshape(self.Hlen, self.bins)
    #         M_gain_temp  = total_sum[2].reshape(self.Hlen, self.bins)
    #         Mb_gain_temp = total_sum[3].reshape(self.Hlen, self.bins)
                                 
    #     else:
            
    #         _, M_gain_temp, Mb_gain_temp = transfer_1mom_vec_3D(
    #             current_static, 
    #             self.M_loss_buffer, 
    #             self.M_gain_buffer,
    #             self.Mb_gain_buffer,
    #             ck12_curr, 
    #             dMi_loss_curr, 
    #             dMj_loss_curr, 
    #             dM_loss_curr,
    #             dM_gain_curr, 
    #             dMb_gain_frac_curr, 
    #             self.bins, 
    #             self.Hlen, 
    #             self.dnum, 
    #             breakup=self.breakup)
            
    #         # M_loss, M_gain_temp, Mb_gain_temp = transfer_1mom_bins_vec(
    #         #     current_static, 
    #         #     ck12_curr, 
    #         #     dMi_loss_curr, 
    #         #     dMj_loss_curr, 
    #         #     dM_loss_curr,
    #         #     dM_gain_curr, 
    #         #     dMb_gain_frac_curr, 
    #         #     self.dnum, 
    #         #     self.bins, 
    #         #     self.Hlen, 
    #         #     breakup=self.breakup)
            
    #         # M1_loss_temp, M2_loss_temp, M_gain_temp, Mb_gain_temp = transfer_1mom_bins_optimized(
    #         #    current_static, # Slice of structured array 
    #         #    ck12_curr, 
    #         #    dMi_loss_curr, 
    #         #    dMj_loss_curr, 
    #         #    dM_gain_curr,
    #         #    dMb_gain_frac_curr,       # Vertical slice of 2D array 
    #         #    self.bins, 
    #         #    self.Hlen)


    #     #OLD
    #     # Accumulate results
        
    #     #print('M1_loss_temp=',M1_loss_temp.shape)
    #     #raise Exception()
        
    #     #M_loss[d1,:,:]    += M1_loss_temp 
    #     #M_loss[d2,:,:]    += M2_loss_temp
        
    #     M_gain[indc,:,:]  += self.Eagg*M_gain_temp
    #     M_gain[indb,:,:]  += self.Ebr*Mb_gain_temp
        
    #     #M_loss *= self.Ecb
        
    #     M_loss = self.M_loss_buffer * self.Ecb
        
        
    #     #print('M_gain=',M_gain.sum())
    #     #print('M_loss=',M_loss.sum())
        
    #    # raise Exception()
        
    #     return dt * (M_gain - M_loss)



    # def interact_1mom_SS_NEW(self, dt):
    #     #M_loss = np.zeros_like(self.Mbins)
    #     #M_gain = np.zeros_like(self.Mbins)
        
    #     self.M_loss_tot_buffer.fill(0.) 
    #     self.M_gain_tot_buffer.fill(0.)
        
    #     indc, indb = self.indc, self.indb
    #     dd = 0
        
    #     # Loop over (d1,d2) interaction pairs
    #     for meta in self.pair_metadata:
            
    #         self.M1_loss_buffer.fill(0.)
    #         self.M2_loss_buffer.fill(0.)
    #         self.M_gain_buffer.fill(0.)
    #         self.Mb_gain_buffer.fill(0.)
            
    #         d1, d2 = meta['d1'], meta['d2']
    #         s = meta['slice']
            
    #         # --- OPTIMIZATION: One vectorized update per d-pair ---
    #         # Fetch indices from our pre-baked static buffer
    #         kr = self.static_full['kr'][s]
    #         ir = self.static_full['ir'][s]
    #         jr = self.static_full['jr'][s]
                       
    #         # Calculate ck12 only for active interactions
    #         # This is a contiguous operation now
    #         self.ck12_dynamic[s] = self.cki[d1, kr, ir] * self.cki[d2, kr, jr]
                     
    #         active_indices = np.where(self.ck12_dynamic[s] > 0)[0]

    #         if len(active_indices) == 0:
    #             dd += 1
    #             continue
            
    #         # Slice buffers for this specific interaction set
    #         # Using the absolute slice 's' + the sub-filter 'active_indices'
    #         current_static = self.static_full[s][active_indices]
    #         ck12_curr = self.ck12_dynamic[s][active_indices]
                        
    #                     # 3. EXTRACTION (The I/O Buffer)
    #         # Pull from memmap into contiguous RAM blocks

    #         dMi_loss_curr = self.dMi_loss[dd, ir[active_indices], jr[active_indices]]
    #         dMj_loss_curr = self.dMj_loss[dd, ir[active_indices], jr[active_indices]]
    #         dM_loss_curr  = self.dM_loss[dd, ir[active_indices], jr[active_indices]]
    #         dM_gain_curr  = self.dM_gain[dd, ir[active_indices], jr[active_indices], :]
    #         # Handle kmin-based indexing for dMb
    #         kmin_curr = self.static_full['kmin'][s][active_indices]
    #         #kmid_curr = self.static_full['kmid'][s][active_indices]
    #         dMb_gain_frac_curr = self.dMb_gain_frac[:, kmin_curr]

    #         if self.parallel:
    #             # Chunking logic for Parallel  
    #             # Assuming n_active is the number of interactions where ck12 > 0 
    #             chunk_size = int(np.ceil(len(current_static) / self.n_jobs))
    #             batches = [slice(i, i + chunk_size) for i in range(0, len(current_static), chunk_size)]


    #             # The Parallel Call 
    #             gain_loss_temp = self.pool(delayed(sum_1mom_batches_optimized)(
    #                 current_static[batch], # Slice of structured array 
    #                 ck12_curr[batch], 
    #                 dMi_loss_curr[batch], 
    #                 dMj_loss_curr[batch], 
    #                 dM_gain_curr[batch,:],
    #                 dMb_gain_frac_curr[:,batch],       # Vertical slice of 2D array 
    #                 self.bins, 
    #                 self.Hlen
    #                 ) for batch in batches)
                
                
    #             # Combine results in the main process [cite: 1, 6]
    #             total_sum = np.sum(gain_loss_temp, axis=0) # Sum across chunks 

    #             # Reshape back to the 2D height-bin grid [cite: 1, 21]
    #             M1_loss_temp = total_sum[0].reshape(self.Hlen, self.bins)
    #             M2_loss_temp = total_sum[1].reshape(self.Hlen, self.bins)
    #             M_gain_temp  = total_sum[2].reshape(self.Hlen, self.bins)
    #             Mb_gain_temp = total_sum[3].reshape(self.Hlen, self.bins)
                                     
 
    #         else:

    #             # WORKING
    #             # M1_loss_temp, M2_loss_temp, M_gain_temp, Mb_gain_temp = transfer_1mom_bins_optimized(
    #             #    current_static, # Slice of structured array 
    #             #    ck12_curr, 
    #             #    dMi_loss_curr, 
    #             #    dMj_loss_curr, 
    #             #    dM_loss_curr,
    #             #    dM_gain_curr,
    #             #    dMb_gain_frac_curr,       # Vertical slice of 2D array 
    #             #    self.bins, 
    #             #    self.Hlen)
                
    #             transfer_1mom_bins_inplace(
    #                current_static, 
    #                ck12_curr, 
    #                dMi_loss_curr, 
    #                dMj_loss_curr, 
    #                dM_loss_curr, 
    #                dM_gain_curr, 
    #                dMb_gain_frac_curr,
    #                self.M1_loss_buffer_flat,
    #                self.M2_loss_buffer_flat,
    #                self.M_gain_buffer_flat,
    #                self.Mb_gain_buffer_flat,
    #                self.bins, 
    #                self.Hlen,
    #                self.breakup)
    
    #         # Accumulate results
    #         #M_loss[d1,:,:]    += M1_loss_temp 
    #         #M_loss[d2,:,:]    += M2_loss_temp
            
    #         #M_gain[indc,:,:]  += self.Eagg*M_gain_temp
    #         #M_gain[indb,:,:]  += self.Ebr*Mb_gain_temp
            
            
    #         self.M_loss_tot_buffer[d1].ravel()[:] += self.M1_loss_buffer_flat
    #         self.M_loss_tot_buffer[d2].ravel()[:] += self.M2_loss_buffer_flat
            
    #         self.M_gain_tot_buffer[indc].ravel()[:]  += self.Eagg*self.M_gain_buffer_flat
    #         self.M_gain_tot_buffer[indb].ravel()[:]  += self.Ebr*self.Mb_gain_buffer_flat
            
    #         dd += 1
    
    #     #M_loss *= self.Ecb
        

        
    #     return dt *(self.M_gain_tot_buffer-self.Ecb*self.M_loss_tot_buffer)
        
        #return dt * (M_gain - M_loss)
    
    
    
    # # Advance PSD Mbins and Nbins by one time/height step
    # def interact_1mom_array(self,dt):

    #     # Ndists x height x bins
    #     self.M_loss_tot_buffer.fill(0.) 
    #     self.M_gain_tot_buffer.fill(0.)
        
    #     indc = self.indc
    #     indb = self.indb

    #     dd = 0
        
    #     for d1 in range(self.dnum):
    #         for d2 in range(d1,self.dnum):
                
    #             self.M_gain_buffer.fill(0.)
                
    #             # (dnum x height x bins)
    #             ck1  = self.cki[d1,:,:]              
    #             ck2  = self.cki[d2,:,:] 
                
    #             # ELD NOTE: Try doing this calculation in the parallel call.
    #             # (height x bins x bins)
    #             ck12 =  (ck1[:,:,None]*ck2[:,None,:])

    #             if self.parallel:
                                
    #                 # Batch w.r.t. heights
    #                 batches = np.array_split(np.arange(self.Hlen),self.n_jobs)
                    
                    
    #                 self.pool(transfer_1mom_bins(self.Hlen, self.bins, ck12, self.dMi_loss, self.dMj_loss, self.dM_loss, self.dM_gain, self.kmin, self.kmid, self.dMb_gain_frac, breakup=False))
                    
                    
                    
    #             else:
                         
    #                 M1_loss, M2_loss, M_gain, Mb_gain = transfer_1mom_bins(self.Hlen, self.bins, ck12, self.dMi_loss, self.dMj_loss, self.dM_loss, self.dM_gain, self.kmin, self.kmid, self.dMb_gain_frac, breakup=False)
                    
                    
    #                 # self.M_loss_tot_buffer[d1,:,:] += np.sum(ck12*self.dMi_loss[dd,:,:][None,:,:],axis=2)
    #                 # self.M_loss_tot_buffer[d2,:,:] += np.sum(ck12*self.dMj_loss[dd,:,:][None,:,:],axis=1)
                    
    #                 # # ChatGPT is the GOAT for telling me about np.add.at!
    #                 # #M_gain = np.zeros((self.Hlen,self.bins))
    #                 # np.add.at(self.M_gain_buffer, (np.arange(self.Hlen)[:,None,None],self.kmin), ck12*(self.dM_gain[dd,:,:,0][None,:,:]))
    #                 # np.add.at(self.M_gain_buffer,  (np.arange(self.Hlen)[:,None,None],self.kmid), ck12*(self.dM_gain[dd,:,:,1][None,:,:]))
                    
    #                 # # ELD NOTE: Breakup here can take losses from each pair and calculate gains
    #                 # # for breakup. Breakup gain arrays will be 3D.
    #                 # #if breakup:
                        
    #                 # # (Hlen,bins,bins)
    #                 # Mij_loss = ck12*((self.dMi_loss[dd,:,:]+self.dMj_loss[dd,:,:])[None,:,:])

    #                 # Mb_gain = np.sum((self.dMb_gain_frac[:,self.kmin][None,:,:,:])*Mij_loss[:,None,:,:],axis=(2,3))
                      
    #             self.M_gain_tot_buffer[indc,:,:] += self.Eagg*self.M_gain_buffer
    #             self.M_gain_tot_buffer[indb,:,:] += self.Ebr*Mb_gain
                
    #             dd += 1
                  
    #     self.M_loss_tot_buffer *= self.Ecb
               
    #     return dt*(self.M_gain_tot_buffer-self.M_loss_tot_buffer)


    # def setup_1mom_meta(self):
        
    #     # Setup Interaction precalculations
    #     # Determine all possible d1, d2 combinations
    #     self.pair_metadata = []
    #     total_interactions = 0
        
    #     ak1mom = np.zeros((self.Hlen,self.bins))
    #     ck1mom = np.ones((self.Hlen,self.bins))
        
    #     dd = 0
        
    #     # 1. First pass: Count total active bin-pairs to allocate one big buffer
    #     for d1 in range(self.dnum):
    #         for d2 in range(d1, self.dnum):
                
    #             mask = (~self.cond_1) & self.self_col[:, dd, :, :] # dd=0 used for indexing
    #             kr, ir, jr = np.nonzero(mask)
        
    #             dd += 1
                
    #             count = len(kr)
    #             self.pair_metadata.append({
    #                 'd1': d1, 'd2': d2, 'count': count,
    #                 'slice': slice(total_interactions, total_interactions + count)
    #             })
    #             total_interactions += count
    
    #     # 2. Define Structured Array for Static Data
    #     static_dtype = [
    #         ('kr', 'i4'), ('ir', 'i4'), ('jr', 'i4'),
    #         ('kmin', 'i4'), ('kmid', 'i4')
    #     ]
    #     self.static_full = np.empty(total_interactions, dtype=static_dtype)
        
    #     dd = 0
        
    #     # 3. Fill Static Data (Only once!)
    #     for meta in self.pair_metadata:
    #         s = meta['slice']
    #         d1, d2 = meta['d1'], meta['d2']
    #         mask = (~self.cond_1) & self.self_col[:, dd, :, :]
    #         kr, ir, jr = np.nonzero(mask)
            
    #         # BATCHES
    #         # (bin-pair,)
    #         x11  = self.xi1[ir]
    #         x21  = self.xi2[ir]
    #         ak1  = ak1mom[kr,ir]
    #         ck1  = ck1mom[kr,ir]

    #         # (bin-pair,)
    #         x12  = self.xi1[jr]
    #         x22  = self.xi2[jr]
    #         ak2  = ak1mom[kr,jr]
    #         ck2  = ck1mom[kr,jr]
            
    #         kmin = self.kmin[ir,jr]
    #         kmid = self.kmid[ir,jr]
            
    #         PK = self.PK[:,dd,ir,jr]
            
    #         xk_min = self.xi2[self.kmin[ir,jr]]
            
    #         # NEW METHOD
    #         region_inds, x_bottom_edge, x_top_edge, y_left_edge, y_right_edge=setup_regions(self.bins,kr,ir,jr,x11,x21,x12,x22,xk_min)

    #         dMi_loss, dMj_loss, dM_loss, dM_gain, _, _ = calculate_rates(self.Hlen,self.bins,region_inds,x_bottom_edge,x_top_edge,y_left_edge,y_right_edge,
    #         kr,ir,jr,x11,x21,ak1,ck1,x12,x22,ak2,ck2,PK,kmin,kmid,self.dMb_gain_frac,self.dNb_gain_frac,self.w,self.L,
    #         breakup=self.breakup)
                
    #         self.dMi_loss[dd,ir,jr] = dMi_loss
    #         self.dMj_loss[dd,ir,jr] = dMj_loss 
    #         self.dM_loss[dd,ir,jr] = dM_loss
    #         self.dM_gain[dd,ir,jr,:] = dM_gain
            
    #         self.static_full['kr'][s], self.static_full['ir'][s], self.static_full['jr'][s] = kr, ir, jr
    #         self.static_full['kmin'][s] = self.kmin[ir, jr]
    #         self.static_full['kmid'][s] = self.kmid[ir, jr]
            
    #         dd += 1
    
    #     # 4. Pre-allocate Dynamic Buffer (for ck12 values)
    #     self.ck12_dynamic = np.zeros(total_interactions, dtype='f8')
    
    
    
    
    # # Advance PSD Mbins and Nbins by one time/height step
    # def interact_1mom_SS(self,dt):

    #     # Ndists x height x bins
    #     Mbins_old = self.Mbins.copy() 
    #     Mbins = np.zeros_like(Mbins_old)

    #     M_loss = np.zeros_like(Mbins)
    #     M_gain = np.zeros_like(Mbins)
        
    #     indc = self.indc
    #     indb = self.indb

    #     dd = 0
        
    #     for d1 in range(self.dnum):
    #         for d2 in range(d1,self.dnum):
                
    #             # (dnum x height x bins)
    #             ck1  = self.cki[d1,:,:]              
    #             ck2  = self.cki[d2,:,:] 
                
    #             Mcheck = ((ck1==0.)[:,:,None]) | ((ck2==0.)[:,None,:]) # If M1 or M2 is zero


    #             # IF SLICING
    #             cond_1 = self.cond_1[0,:,:][None,:,:] | Mcheck # New cond_1. Basically exclude bin-pairs that are off grid and ones involving empty bins.
                
    #             # NO SLICING (NOTE: probably need to fix shape of array)
    #             #cond_1 = self.cond_1# New cond_1. Basically exclude bin-pairs that are off grid and ones involving empty bins.
                
    #             kr, ir, jr = np.nonzero((~cond_1)&self.self_col[:,dd,:,:])
                
    #             # ELD NOTE: Try doing this calculation in the parallel call.
    #             # (height x bins x bins)
    #             #ck12 =  (ck1[:,:,None]*ck2[:,None,:])
                
    #             ck12 = self.cki[d1,kr,ir]*self.cki[d2,kr,jr]

    #             dMi_loss_r = self.dMi_loss[dd,ir,jr] # (batch,)
    #             dMj_loss_r = self.dMj_loss[dd,ir,jr]  #(batch,)
    #             dM_loss_r = self.dM_loss[dd,ir,jr] #(batch,)
    #             dM_gain_r  = self.dM_gain[dd,ir,jr,:] #(batch,2)
    #             dMb_gain_frac_r = self.dMb_gain_frac[:,self.kmin[ir,jr]]
            

              
    #             if self.parallel:
                    
    #                 #batches = np.array_split(np.arange(len(kr)),self.n_jobs) 
                    
    #                 batches = np.array_split(np.arange(self.Hlen),self.n_jobs)
                    
    #                 #print(batches)
    #                # raise Exception()
                    
                                                
    #                 #gain_loss_temp = Parallel(n_jobs=self.n_jobs,verbose=0)(delayed(calculate_1mom_batch)(
    #                 #                    kr[batch],ir[batch],jr[batch],
    #                 #                    ck12[kr[batch],ir[batch],jr[batch]],
    #                 #                    dMi_loss_r[batch],
    #                 #                    dMj_loss_r[batch],
    #                 #                    dM_gain_r[batch,:],
    #                 #                    self.kmin[ir[batch],jr[batch]],
    #                 #                    self.kmid[ir[batch],jr[batch]],
    #                 #                    M_gain_temp,M1_loss,M2_loss,
    #                 #                    Mb_gain,dMb_gain_frac_r[:,batch],self.breakup) for batch in batches)  
                    
 
    #                 #sum_1mom_batches(1,batches[1],ck12,self.mem_map_dict,dd,self.Hlen,self.bins,breakup=self.breakup)
 
    #                 #self.pool(delayed(sum_1mom_batches)(
    #                 #ii,kr[batches[ii]],ir[batches[ii]],jr[batches[ii]],batches[ii],ck12[batches[ii],:,:],self.mem_map_dict,dd,self.Hlen,self.bins,breakup=self.breakup) for ii in range(len(batches))) 
                    
    #                 # gain_loss_temp = self.pool(delayed(sum_1mom_batches)(
    #                 #                     kr[batch],ir[batch],jr[batch],
    #                 #                     ck12[kr[batch],ir[batch],jr[batch]],
    #                 #                     dMi_loss_r[batch],
    #                 #                     dMj_loss_r[batch],
    #                 #                     dM_gain_r[batch,:],
    #                 #                     self.kmin[ir[batch],jr[batch]],
    #                 #                     self.kmid[ir[batch],jr[batch]],
    #                 #                     M_gain_temp,M1_loss,M2_loss,
    #                 #                     Mb_gain,dMb_gain_frac_r[:,batch],self.breakup) for batch in batches)  
                    
    #                 # WORKING
    #                 # gain_loss_temp = self.pool(delayed(calculate_1mom_batch)(
    #                 #                     kr[batch],ir[batch],jr[batch],
    #                 #                     ck12[kr[batch],ir[batch],jr[batch]],
    #                 #                     dMi_loss_r[batch],
    #                 #                     dMj_loss_r[batch],
    #                 #                     dM_gain_r[batch,:],
    #                 #                     self.kmin[ir[batch],jr[batch]],
    #                 #                     self.kmid[ir[batch],jr[batch]],
    #                 #                     M_gain_temp,M1_loss,M2_loss,
    #                 #                     Mb_gain,dMb_gain_frac_r[:,batch],self.breakup) for batch in batches)  
                    
                    
                    
    #                 # gl_tot = np.nansum(self.mem_out,axis=0)
                    
    #                 # M1_loss_temp = gl_tot[0,:,:]
    #                 # M2_loss_temp = gl_tot[1,:,:]
    #                 # M_gain_temp  = gl_tot[2,:,:]
    #                 # Mb_gain_temp = gl_tot[3,:,:]

    #                 # M1_loss_temp = np.nansum(np.vstack([gl[0] for gl in gain_loss_temp]),axis=0)
    #                 # M2_loss_temp = np.nansum(np.vstack([gl[1] for gl in gain_loss_temp]),axis=0)
    #                 # M_gain_temp =  np.nansum(np.vstack([gl[2] for gl in gain_loss_temp]),axis=0)
    #                 # Mb_gain_temp = np.nansum(np.vstack([gl[3] for gl in gain_loss_temp]),axis=0)
                    
                    
                    
    #             else:
                                         
    #                 M1_loss_temp,M2_loss_temp,\
    #                 M_gain_temp,Mb_gain_temp =\
    #                 transfer_1mom_bins(self.Hlen,self.bins,
    #                                kr,ir,jr,
    #                                #ck12[kr,ir,jr],
    #                                ck12,
    #                                dMi_loss_r,
    #                                dMj_loss_r,
    #                                dM_loss_r,
    #                                dM_gain_r,
    #                                self.kmin[ir,jr],
    #                                self.kmid[ir,jr],
    #                                dMb_gain_frac_r,self.breakup) 
                                        
    #             M_loss[d1,:,:]    += M1_loss_temp 
    #             M_loss[d2,:,:]    += M2_loss_temp
                
    #             M_gain[indc,:,:]  += self.Eagg*M_gain_temp
    #             M_gain[indb,:,:]  += self.Ebr*Mb_gain_temp
                
    #             dd += 1
                
    #     M_loss *= self.Ecb
        
    #     M_net = dt*(M_gain-M_loss) 
               
    #     return M_net



    # # Advance PSD Mbins and Nbins by one time/height step
    # def interact_2mom_SS(self,dt):

    #     # Ndists x height x bins
    #     Mbins_old = self.Mbins.copy() 
    #     Nbins_old = self.Nbins.copy()
        
    #     Mbins = np.zeros_like(Mbins_old)
    #     Nbins = np.zeros_like(Nbins_old)

    #     M_loss = np.zeros_like(Mbins)
    #     N_loss = np.zeros_like(Nbins) 
        
    #     M_gain = np.zeros_like(Mbins)
    #     N_gain = np.zeros_like(Nbins)
        
    #     indc = self.indc
    #     indb = self.indb
        
    #     dd = 0
    #     for d1 in range(self.dnum):
    #         for d2 in range(d1,self.dnum):
                                         
    #             # Mcheck = (height x d1 bins x d2 bins)
    #             Mcheck = ((self.Mbins[d1,:,:]==0.)[:,:,None]) | ((self.Mbins[d2,:,:]==0.)[:,None,:]) # If M1 or M2 is zero, do not include in bin-pair list.

    #             cond_1 = self.cond_1 | Mcheck # New cond_1. Basically exclude bin-pairs that are off grid and ones involving empty bins.
                
    #             # Get 3D indices for all bins that will interact at all heights
    #             # indices have shape (bin-pairs,) where ir are dist1 indices 
    #             # and jr are dist2 indices. kr are height indices.
    #             kr, ir, jr = np.nonzero((~cond_1)&self.self_col[:,dd,:,:])
                               
    #             # BATCHES
    #             # (bin-pair,)
    #             x11  = self.x1[d1,kr,ir]
    #             x21  = self.x2[d1,kr,ir]
    #             ak1  = self.aki[d1,kr,ir]
    #             ck1  = self.cki[d1,kr,ir]

    #             # (bin-pair,)
    #             x12  = self.x1[d2,kr,jr]
    #             x22  = self.x2[d2,kr,jr]
    #             ak2  = self.aki[d2,kr,jr]
    #             ck2  = self.cki[d2,kr,jr] 
                
    #             kmin = self.kmin[ir,jr]
    #             kmid = self.kmid[ir,jr]
    #             xk_min = self.xi2[kmin]
                
    #             PK = self.PK[:,dd,ir,jr]
                 
    #             # SETUP REGIONS
    #             inds, x_bottom_edge, x_top_edge, y_left_edge, y_right_edge = setup_regions(self.bins,kr,ir,jr,x11,x21,x12,x22,xk_min)
            
    #             if self.parallel and self.Hlen>1:
                    
    #                 #self.shm_out_arr.fill(0.0)
                    
    #                 input_arrays = (inds,x_bottom_edge,x_top_edge,y_left_edge,y_right_edge,kr,ir,jr,x11,x21,ak1,ck1,x12,x22,ak2,ck2)
                    
    #                 # SORT REGIONS ACCORDING TO COST
    #                 # Takes somewhat long but could theoretically save time in
    #                 # parallel call.
    #                 batches = balance_regions(inds,self.n_jobs)
    #                 all_slices = [tuple(arr[batch] for arr in input_arrays) for batch in batches]
                    

    #                 self.pool(delayed(sum_2mom_batches)(
    #                                        ii,*slices,self.mem_map_dict,dd,self.Hlen,self.bins,breakup=self.breakup) for slices,ii in zip(all_slices,range(self.n_jobs)))
                    
    #                 gl_tot = np.sum(self.mem_out,axis=0)
                    
    #                 M1_loss_temp = gl_tot[0,:,:]
    #                 M2_loss_temp = gl_tot[1,:,:]
    #                 M_gain_temp  = gl_tot[2,:,:]
    #                 Mb_gain_temp = gl_tot[3,:,:]
    #                 N1_loss_temp = gl_tot[4,:,:]
    #                 N2_loss_temp = gl_tot[5,:,:]
    #                 N_gain_temp  = gl_tot[6,:,:]
    #                 Nb_gain_temp = gl_tot[7,:,:]
   
    #             else:
                    
    #                 # Calculate transfer rates
    #                 dMi_loss, dMj_loss, dM_loss, dM_gain, dNi_loss, dN_gain = calculate_rates(self.Hlen,self.bins,inds,x_bottom_edge,x_top_edge,y_left_edge,y_right_edge,
    #                 kr,ir,jr,x11,x21,ak1,ck1,x12,x22,ak2,ck2,PK,
    #                 kmin,kmid,self.dMb_gain_frac,self.dNb_gain_frac,self.w,self.L,
    #                 breakup=self.breakup)

    #                 # Perform transfer/assignment to bins
    #                 M1_loss_temp, M2_loss_temp, M_gain_temp, Mb_gain_temp,\
    #                 N1_loss_temp, N2_loss_temp, N_gain_temp, Nb_gain_temp = transfer_bins(self.Hlen,self.bins,kr,ir,jr,kmin,kmid,dMi_loss,dMj_loss,dM_loss,dM_gain,
    #                                   dNi_loss,dN_gain,self.dMb_gain_frac,self.dNb_gain_frac,breakup=self.breakup)
  
    #             M_loss[d1,:,:]    += M1_loss_temp 
    #             M_loss[d2,:,:]    += M2_loss_temp
                
    #             M_gain[indc,:,:]  += self.Eagg*M_gain_temp
    #             M_gain[indb,:,:]  += self.Ebr*Mb_gain_temp
                
    #             N_loss[d1,:,:]    += N1_loss_temp
    #             N_loss[d2,:,:]    += N2_loss_temp
                 
    #             N_gain[indc,:,:]  += self.Eagg*N_gain_temp
    #             N_gain[indb,:,:]  += self.Ebr*Nb_gain_temp
                
    #             dd += 1
          
      
    #     M_loss *= self.Ecb
    #     N_loss *= self.Ecb 
        
    #     M_net = dt*(M_gain-M_loss) 
    #     N_net = dt*(N_gain-N_loss)
        
        
    #     #print('M_net=',M_net.sum())
    #     #print('M_gain=',M_gain.sum())

    #     #print('M_net=',(M_gain-M_loss).sum())
    #     #print('M_gain=',M_gain.sum())
    #     #print('M_loss=',M_loss.sum())
    #     #raise Exception()    

        
    #     return M_net, N_net  
    
    
    
    
    # def _setup_shared_2mom_tensors(self):
    #     """
    #     Pre-allocates Shared Memory buffers for the 2-moment calculations.
    #     """
    #     # Dictionary to keep track of SHM objects to prevent GC
    #     self.shm_registry = {}   
    
    #     # 1. Output Tendency Buffer: (4, dnum, Hlen, bins) 
    #     # [M_loss, M_gain, N_loss, N_gain]
    #     tendency_shape = (4, self.dnum, self.Hlen, self.bins)
    #     self.shm_out = shared_memory.SharedMemory(name='shm_out', create=True, size=np.zeros(tendency_shape).nbytes)
    #     self.shm_out_arr = np.ndarray(tendency_shape, dtype=np.float64, buffer=self.shm_out.buf)
    
    #     # 2. Geometry & Distribution Shapes
    #     geom_shape_v = (self.pnum, self.Hlen, self.bins, 1)      # Vertical (i-axis)
    #     geom_shape_h = (self.pnum, self.Hlen, 1, self.bins)      # Horizontal (j-axis)
    #     full_field_shape = (self.pnum, self.Hlen, self.bins, self.bins) # Interaction Grid
    
    #     # 3. Distribution Parameters (aki/cki)
    #     self.shm_aki1 = self._create_shm_array(np.zeros(geom_shape_v), 'shm_aki1')
    #     self.shm_cki1 = self._create_shm_array(np.zeros(geom_shape_v), 'shm_cki1')
    #     self.shm_aki2 = self._create_shm_array(np.zeros(geom_shape_h), 'shm_aki2')
    #     self.shm_cki2 = self._create_shm_array(np.zeros(geom_shape_h), 'shm_cki2')
    
    #     # 4. Consolidated Geometric Boundaries
    #     # Indices: 0:x11, 1:x21, 2:x12, 3:x22, 4:x_top, 5:x_bot, 6:y_lef, 7:y_rig
    #     self.shm_bounds = self._create_shm_array(np.zeros((8,) + full_field_shape), 'shm_bounds')
    
    #     # 5. Mask and Destination Indices
    #     self.shm_mask = self._create_shm_array(np.zeros(full_field_shape, dtype=np.uint8), 'shm_mask')
    #     self.shm_kmin = self._create_shm_array(self.kmin_p.astype(np.int32), 'shm_kmin')
    #     self.shm_kmid = self._create_shm_array(self.kmid_p.astype(np.int32), 'shm_kmid')
    
    #     # Register shapes for worker attachment
    #     self.shapes = {
    #         'geom_v': geom_shape_v,
    #         'geom_h': geom_shape_h,
    #         'field': full_field_shape,
    #         'out': tendency_shape,
    #         'dnum': self.dnum,
    #         'k': self.kmin_p.shape
    #     }

    # def _setup_shared_tensors(self):
    #     # Dictionary to keep track of SHM objects so they don't get garbage collected
    #     self.shm_registry = {}   
        
    #        # Create Shared Tensors (using unique names)
    #     # We use 'shm_' prefix to identify them in the workers
    #     self.shm_dMi_loss = self._create_shm_array(self.dMi_loss, 'shm_dMi_loss')
    #     self.shm_dMj_loss = self._create_shm_array(self.dMj_loss, 'shm_dMj_loss')
    #     self.shm_dM_loss = self._create_shm_array(self.dM_loss, 'shm_dM_loss')
    #     self.shm_dM_gain = self._create_shm_array(self.dM_gain, 'shm_dM_gain')
    #     self.shm_dMb_gain_kernel = self._create_shm_array(self.dMb_gain_kernel, 'shm_dMb_gain_kernel')
    #     self.shm_kmin = self._create_shm_array(self.kmin_p, 'shm_kmin')
    #     self.shm_kmid = self._create_shm_array(self.kmid_p, 'shm_kmid')
        
    #     # Also create the output buffer for tendencies
    #     tendency_shape = (2, self.dnum, self.Hlen, self.bins)
    #     self.shm_out = shared_memory.SharedMemory(name='shm_out', create=True, size=np.zeros(tendency_shape).nbytes)
    #     self.shm_out_arr = np.ndarray(tendency_shape, dtype=np.float64, buffer=self.shm_out.buf)
        
    #     self.shapes = {
    #     'dMi_loss': self.dMi_loss.shape, 
    #     'dMj_loss':self.dMj_loss.shape,
    #     'dM_loss':self.dM_loss.shape,
    #     'dM_gain': self.dM_gain.shape,
    #     'dMb_gain_kernel': self.dMb_gain_kernel.shape, 
    #     'k': self.kmin_p.shape,
    #     'out': self.shm_out_arr.shape}
          
    # def _create_shm_array(self,data,name):
    #     shm = shared_memory.SharedMemory(name=name,create=True, size=data.nbytes)
       
    #     shared_arr = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    #     shared_arr[:] = data[:] # Copy data into SHM once
    #     self.shm_registry[name] = shm  
       
    #     return shared_arr
       
    # def _setup_buffer(self,outlen):
        
    #     outpath = os.path.join(gettempdir(),f"output_2d_{id(self)}.mmap")
    #     out_shape = (self.n_jobs,outlen,self.Hlen,self.bins)
        
    #     out = np.memmap(outpath, dtype='float64',mode='w+',shape=out_shape)
    #     #out[:] = 0
    #     #out.flush()
    
    #     self.mem_map_dict['outputs'] = outpath
    #     self.mem_map_dict['output_shape'] = out_shape
        
    #     self._temp_files.append(outpath)
        
    #     return out
    
    # def dump_data(self,name,array):
        
    #     if name not in self.mem_map_dict:
    #         path = os.path.join(gettempdir(), f"{name}_{id(self)}.mmap")
    #         dump(array,path)
    #         self.mem_map_dict['inputs'][name] = path
    #         self._temp_files.append(path)