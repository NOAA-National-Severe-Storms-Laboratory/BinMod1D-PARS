# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 08:17:50 2025

@author: edwin.dunnavan
"""
import numpy as np

import scipy.special as scip
#from functools import lru_cache
#from heapq import heapify, heappop, heappush

#from joblib import load

#from multiprocessing import shared_memory

import numba as nb

@nb.jit(nopython=True, parallel=True, cache=False, fastmath=True)
def kernel_1mom(M_loss, M_gain, 
                Eagg, Ebr, rate, h_idx, s1_idx, s2_idx, i_idx, j_idx, 
                p_idx, dMi, dMj, k0_map, k1_map, dMg, # Full Arrays
                dMb_kernel, indc, indb):
    
    n_active = len(rate)
    n_threads = nb.get_num_threads()
    H_len, n_bins = M_gain.shape[1], M_gain.shape[2]
    n_species = M_loss.shape[0]

    L_temp = np.zeros((n_threads, n_species, H_len, n_bins))
    G_temp = np.zeros((n_threads, H_len, n_bins)) 
    B_temp = np.zeros((n_threads, H_len, n_bins)) 

    for k in nb.prange(n_active):
        tid = nb.get_thread_id()
        
        # 1. Unpack Indices
        r = rate[k]
        h = h_idx[k]
        i = i_idx[k]
        j = j_idx[k]
        p = p_idx[k] # Pair index
        
        # 3. Coalescence (Lookup 3D/4D)
        r_agg = r * Eagg[k]
        r_br  = r * Ebr[k]
        r_loss = r_agg + r_br
        
        if r_loss == 0.0: continue
        
        mass_loss_i = dMi[p,i,j]
        mass_loss_j = dMj[p,i,j]
        
        tot_mass_loss = mass_loss_i + mass_loss_j
    
        # 2. Losses (Lookup 3D)
        L_temp[tid, s1_idx[k], h, i] += r_loss * mass_loss_i
        L_temp[tid, s2_idx[k], h, j] += r_loss * mass_loss_j
          
        if r_agg > 0.:
            # Look up destination bins and split-factors on the fly
            # This eliminates the Python-side memory copy
            k_dest0 = k0_map[p, i, j]
            k_dest1 = k1_map[p, i, j]
            
            G_temp[tid, h, k_dest0] += r_agg * dMg[p, i, j, 0]
            #G_temp[tid, h, k_dest1] += r_agg * dMg[p, i, j, 1]
            
            # More exact way of getting the other bin mass partitioning done.
            # This prevents any inexactness with the dMg[:,:,:,1] calculation.
            G_temp[tid, h, k_dest1] += r_agg * (tot_mass_loss-dMg[p, i, j, 0])
            
        
        # 4. Breakup (Lookup 3D)
        if r_br > 0.0:
            
            dm_br = r_br * tot_mass_loss
            
            for b in range(n_bins):
                prob = dMb_kernel[b, i, j]
                if prob > 0:
                    B_temp[tid, h, b] += prob * dm_br

    # Reduction
    for t in range(n_threads):
        M_loss += L_temp[t]
        M_gain[indc] += G_temp[t]
        M_gain[indb] += B_temp[t]

    return M_loss, M_gain




@nb.jit(nopython=True, parallel=True,cache=True, fastmath=False)
def loss_kernel_numba(dN_out, dMi_out, dMj_out, 
                      C, x1, x2, y1, y2):
    """
    Computes rectangular source integrals matching 'source_integrals'.
    
    C: (9, N) flattened coefficient array from (3, 3, N)
       Mapping: 
       0:(0,0), 1:(0,1)[y], 2:(0,2)[y2]
       3:(1,0)[x], 4:(1,1)[xy], 5:(1,2)[xy2]
       6:(2,0)[x2], 7:(2,1)[x2y], 8:(2,2)[x2y2]
    """
    n_interactions = len(x1)
    
    for k in nb.prange(n_interactions):
        # 1. Load coordinates
        xa, xb = x1[k], x2[k]
        ya, yb = y1[k], y2[k]
        
        # 2. Calculate Basis Integrals (Exactly matching DX/DY in source_integrals)
        # x-integrals (0 to 3rd power)
        dx0 = xb - xa
        dx1 = 0.5 * (xb**2 - xa**2)
        dx2 = (1.0/3.0) * (xb**3 - xa**3)
        dx3 = 0.25 * (xb**4 - xa**4)
        
        # y-integrals (0 to 3rd power)
        dy0 = yb - ya
        dy1 = 0.5 * (yb**2 - ya**2)
        dy2 = (1.0/3.0) * (yb**3 - ya**3)
        dy3 = 0.25 * (yb**4 - ya**4)
        
        # 3. Load Coefficients with CORRECT Mapping
        # i is x-power, j is y-power. Flat index = i*3 + j
        c00 = C[0, k] # i=0, j=0
        c01 = C[1, k] # i=0, j=1 (y)
        c02 = C[2, k] # i=0, j=2 (y^2)
        c10 = C[3, k] # i=1, j=0 (x)
        c11 = C[4, k] # i=1, j=1 (xy)
        c12 = C[5, k] # i=1, j=2 (xy^2)
        c20 = C[6, k] # i=2, j=0 (x^2)
        c21 = C[7, k] # i=2, j=1 (x^2y)
        c22 = C[8, k] # i=2, j=2 (x^2y^2)
        
        # 4. Integrate Number: Int(P)
        # Sum(C_ij * Int(x^i) * Int(y^j))
        # Uses dx0..dx2 and dy0..dy2
        term_n = (c00*dx0*dy0 + c01*dx0*dy1 + c02*dx0*dy2 +
                  c10*dx1*dy0 + c11*dx1*dy1 + c12*dx1*dy2 +
                  c20*dx2*dy0 + c21*dx2*dy1 + c22*dx2*dy2)
        
        dN_out[k] = term_n
        
        # 5. Integrate Mass i: Int(x * P)
        # Shifts x-powers by +1. Uses dx1..dx3
        term_mi = (c00*dx1*dy0 + c01*dx1*dy1 + c02*dx1*dy2 +
                   c10*dx2*dy0 + c11*dx2*dy1 + c12*dx2*dy2 +
                   c20*dx3*dy0 + c21*dx3*dy1 + c22*dx3*dy2)
                   
        dMi_out[k] = term_mi

        # 6. Integrate Mass j: Int(y * P)
        # Shifts y-powers by +1. Uses dy1..dy3
        term_mj = (c00*dx0*dy1 + c01*dx0*dy2 + c02*dx0*dy3 +
                   c10*dx1*dy1 + c11*dx1*dy2 + c12*dx1*dy3 +
                   c20*dx2*dy1 + c21*dx2*dy2 + c22*dx2*dy3)
        
        dMj_out[k] = term_mj

@nb.jit(nopython=True, parallel=True,cache=True, fastmath=True)
def tri_kernel_numba(dN_out, dM_out, 
                     C, x1, y1, x2, y2, x3, y3, 
                     w, L):
    """
    Computes triangular gain integrals matching 'tri_gain_integrals'.
    """
    n_interactions = len(x1)
    n_quad = len(w)
    
    for k in nb.prange(n_interactions):
        v1x, v1y = x1[k], y1[k]
        v2x, v2y = x2[k], y2[k]
        v3x, v3y = x3[k], y3[k]
        
        # Calculate Area (matches 0.5 * abs(...))
        detJ = abs(v1x*(v2y - v3y) + v2x*(v3y - v1y) + v3x*(v1y - v2y))
        area = 0.5 * detJ
        
        # Load Coefficients (i=x_pow, j=y_pow)
        c00 = C[0, k]
        c01 = C[1, k]; c02 = C[2, k]
        c10 = C[3, k]; c11 = C[4, k]; c12 = C[5, k]
        c20 = C[6, k]; c21 = C[7, k]; c22 = C[8, k]
        
        sum_N = 0.0
        sum_M = 0.0
        
        for q in range(n_quad):
            # Quadrature Point
            l1, l2, l3 = L[q, 0], L[q, 1], L[q, 2]
            xq = l1*v1x + l2*v2x + l3*v3x
            yq = l1*v1y + l2*v2y + l3*v3y
            
            # Precompute powers
            x2_val = xq*xq
            y2_val = yq*yq
            
            # Evaluate P(x,y)
            # CAREFUL: c01 is y, c10 is x
            poly = (c00 + 
                    c01*yq + c02*y2_val + 
                    c10*xq + c11*xq*yq + c12*xq*y2_val + 
                    c20*x2_val + c21*x2_val*yq + c22*x2_val*y2_val)
            
            weight = w[q]
            
            # Accumulate
            sum_N += weight * poly
            sum_M += weight * poly * (xq + yq)
            
        dN_out[k] = sum_N * area
        dM_out[k] = sum_M * area

# 1. Enable Parallelism
@nb.jit(nopython=True, parallel=True, cache=False, fastmath=False)
def breakup_kernel(M_gain, N_gain, M_loss, 
                   dMb_kernel, dNb_kernel,Ebr, 
                   i_idx, j_idx, h_idx, k_limit_idx, 
                   indb):
    """
    Apply breakup redistribution.
    Thread-safe implementation using local reduction buffers.
    """
    
    n_interactions = len(i_idx)
    
    H_len = M_gain.shape[1]
    n_bins = M_gain.shape[2]
    
    # 2. Get active thread count (inside the parallel region this works best)
    # However, for allocation, we need it once. 
    # Numba will use the number of threads configured via set_num_threads
    n_threads = nb.get_num_threads()
    
    # 3. Allocation (Per-thread scratchpads)
    # This ensures no two threads write to the same [h, b] simultaneously
    M_temp = np.zeros((n_threads, H_len, n_bins), dtype=np.float64)
    N_temp = np.zeros((n_threads, H_len, n_bins), dtype=np.float64)
    
    # 4. Parallel Loop
    for k in nb.prange(n_interactions):
        
        tid = nb.get_thread_id()
        
        i = i_idx[k]
        j = j_idx[k]
        h = h_idx[k]
        
        # Scalar Mass Loss
        dm_eff = M_loss[k] * Ebr[k]
        
        # 5. FIX: Iterate all bins (Safety)
        # Rely on the 'if prob > 0' check to skip empty work.
        # This prevents cutting off the "upper tail" of the interpolation.
        for b in range(n_bins): 
            
            prob_m = dMb_kernel[b, i, j]
            
            # Sparse check (High Speed)
            if prob_m > 0:
                prob_n = dNb_kernel[b, i, j]
                
                # Write to thread-local buffer
                M_temp[tid, h, b] += prob_m * dm_eff
                N_temp[tid, h, b] += prob_n * dm_eff

    # 6. Reduction (Sum threads back to global)
    # This part is fast because it's a linear sum
    for t in range(n_threads):
        for h in range(H_len):
            for b in range(n_bins):
                val_m = M_temp[t, h, b]
                if val_m > 0:
                    M_gain[indb, h, b] += val_m
                    
                val_n = N_temp[t, h, b]
                if val_n > 0:
                    N_gain[indb, h, b] += val_n     

    return M_gain, N_gain

# 1. Enable Parallelism
@nb.jit(nopython=True, parallel=True, cache=False, fastmath=False)
def breakup_1mom_kernel(M_gain, M_loss,dMb_kernel, 
                   i_idx, j_idx, h_idx, k_limit_idx, 
                   indb, Ebr):
    """
    Apply breakup redistribution.
    Thread-safe implementation using local reduction buffers.
    """
    n_interactions = len(i_idx)
    
    H_len = M_gain.shape[1]
    n_bins = M_gain.shape[2]
    
    # 2. Get active thread count (inside the parallel region this works best)
    # However, for allocation, we need it once. 
    # Numba will use the number of threads configured via set_num_threads
    n_threads = nb.get_num_threads()
    
    # 3. Allocation (Per-thread scratchpads)
    # This ensures no two threads write to the same [h, b] simultaneously
    M_temp = np.zeros((n_threads, H_len, n_bins), dtype=np.float64)
    
    # 4. Parallel Loop
    for k in nb.prange(n_interactions):
        
        tid = nb.get_thread_id()
        
        i = i_idx[k]
        j = j_idx[k]
        h = h_idx[k]
        
        # Scalar Mass Loss
        dm_eff = M_loss[k] * Ebr
        
        # 5. FIX: Iterate all bins (Safety)
        # Rely on the 'if prob > 0' check to skip empty work.
        # This prevents cutting off the "upper tail" of the interpolation.
        for b in range(n_bins): 
            
            prob_m = dMb_kernel[b, i, j]
            
            # Sparse check (High Speed)
            if prob_m > 0:
                
                # Write to thread-local buffer
                M_temp[tid, h, b] += prob_m * dm_eff

    # 6. Reduction (Sum threads back to global)
    # This part is fast because it's a linear sum
    for t in range(n_threads):
        for h in range(H_len):
            for b in range(n_bins):
                val_m = M_temp[t, h, b]
                if val_m > 0:
                    M_gain[indb, h, b] += val_m
                    
    return M_gain


def combined_coeffs_tensor(f, params):
    """
    Expands: (a + bx + cy + dxy) * (aki1*x + cki1) * (aki2*y + cki2)
    f: (4, pnum, 1, bins, bins) | aki, cki: (pnum, H, bins, 1) or (pnum, H, 1, bins)
    """
    a, b, c, d = f[0][:,None,:,:], f[1][:,None,:,:], f[2][:,None,:,:], f[3][:,None,:,:]
    aki1, cki1 = params['aki1'], params['cki1']
    aki2, cki2 = params['aki2'], params['cki2']
    
    # C shape: (3, 3, pnum, H, bins, bins)
    C = np.zeros((3, 3) + aki1.shape[:2] + (aki1.shape[2], aki2.shape[3]), dtype=np.float64)

    C[0, 0] = a * cki1 * cki2
    C[1, 0] = (b * cki1 + a * aki1) * cki2
    C[0, 1] = (c * cki2 + a * aki2) * cki1
    C[2, 0] = b * aki1 * cki2
    C[0, 2] = c * cki1 * aki2
    C[1, 1] = (d * cki1 * cki2 + b * cki1 * aki2 + c * aki1 * cki2 + a * aki1 * aki2)
    C[2, 1] = (d * cki2 + b * aki2) * aki1
    C[1, 2] = (d * cki1 + c * aki1) * aki2
    C[2, 2] = aki1 * aki2 * d
    
    return C

def solve_rect(C, x1, x2, y1, y2):
    """
    C: (k, l, n, h, i, j)
    x1, x2, y1, y2: (n, h, i, j)
    """
    
    N_source, Mi_source, Mj_source = solve_source(C,x1,x2,y1,y2)
    
    M_source = Mi_source+Mj_source

    return M_source, N_source

def solve_source(C, x1, x2, y1, y2):
    """
    C: (k, l, n, h, i, j)
    x1, x2, y1, y2: (n, h, i, j)
    """
    dx, dy = x2 - x1, y2 - y1
    dsx, dsy = x2 + x1, y2 + y1
    x1x2, y1y2 = x1 * x2, y1 * y2

    # Powers 0, 1, 2, 3
    DX = np.stack([
        dx, 
        0.5 * dx * dsx, 
        (1./3.) * dx * (dsx**2 - x1x2), 
        0.25 * dx * dsx * (dsx**2 - 2.*x1x2)
    ], axis=0) # (4, n, h, i, j)

    DY = np.stack([
        dy, 
        0.5 * dy * dsy, 
        (1./3.) * dy * (dsy**2 - y1y2), 
        0.25 * dy * dsy * (dsy**2 - 2.*y1y2)
    ], axis=0) # (4, n, h, i, j)

    # Use 'k' for C's x-power, 'm' for DX's x-power
    # Use 'l' for C's y-power, 'p' for DY's y-power
    # Force alignment: k matches m, l matches p
    iP  = np.einsum('klnhij,knhij,lnhij->nhij', C, DX[:3], DY[:3])
    ixP = np.einsum('klnhij,knhij,lnhij->nhij', C, DX[1:4], DY[:3])
    iyP = np.einsum('klnhij,knhij,lnhij->nhij', C, DX[:3], DY[1:4])

    return iP, ixP, iyP

def solve_tris(C, v_coords, w, L):
    """Integrates 3x3 polynomial C over triangular field using 7-pt quadrature."""
    #v1x, v1y, v2x, v2y, v3x, v3y = v_coords
    
    v1x,v2x,v3x,v1y,v2y,v3y = v_coords
    
    area = 0.5 * np.abs(v1x*(v2y-v3y) + v2x*(v3y-v1y) + v3x*(v1y-v2y))
    
    qx = np.tensordot(L, np.stack([v1x, v2x, v3x], axis=0), axes=([1], [0]))
    qy = np.tensordot(L, np.stack([v1y, v2y, v3y], axis=0), axes=([1], [0]))
    
    x_p = np.stack([np.ones_like(qx), qx, qx**2], axis=0)
    y_p = np.stack([np.ones_like(qy), qy, qy**2], axis=0)
    monos = x_p[:, None,...] * y_p[None, :, ...] 
    
    N_t   = area * np.einsum('klnhij,klqnhij,q->nhij', C, monos, w)
    M_t = area * np.einsum('klnhij,klqnhij,qnhij,q->nhij', C, monos, qx + qy, w)
    
    return M_t, N_t

# =============================================================================
# 2. VECTORIZED CORE ENGINE
# =============================================================================



def vectorized_2mom(params, w, L, dMb_kernel, dNb_kernel, 
                    indc, indb, dnum, Hlen, bins):
    
    # 4D shape (pnum x Hlen x bins x bins)
    regions = params['regions']
    
    reg_map = params['reg_map']

    kmin_l = params['kmin']
    kmid_l = params['kmid']
    
    d1_l = params['d1_ind']
    d2_l = params['d2_ind']
    hind_l = params['hind']
    bi_ind_l = params['bi_ind']
    bj_ind_l = params['bj_ind']
    
    Eagg = params['Eagg']
    Ebr = params['Ebr']
    
    Ecb = Eagg + Ebr
    
    len_loss = len(regions)
    
    C = params['C']

    reg2 = reg_map[2]
    reg3 = reg_map[3]
    reg4 = reg_map[4]
    reg5 = reg_map[5]
    reg6 = reg_map[6]
    reg7 = reg_map[7]
    reg8 = reg_map[8]
    reg9 = reg_map[9]
    reg10 = reg_map[10]
    reg11 = reg_map[11]
    reg12 = reg_map[12]
       
    l2 = reg2.stop-reg2.start
    l3 = reg3.stop-reg3.start
    l5 = reg5.stop-reg5.start
    l6 = reg6.stop-reg6.start
    l7 = reg7.stop-reg7.start
    l8 = reg8.stop-reg8.start
    l11 = reg11.stop-reg11.start
    l12 = reg12.stop-reg12.start
    
    # Unpack boundaries for easier vertex calculation
    x11, x21, x12, x22 = params['x11'],   params['x21'],   params['x12'],   params['x22']
    x_top, x_bot, y_lef, y_rig = params['x_top'], params['x_bot'], params['y_lef'], params['y_rig']

    Cr  = np.concatenate((C,C[:,:,reg5],C[:,:,reg6]),axis=-1)
    xr1 = np.concatenate((x11,x11[reg5],x11[reg6])) 
    xr2 = np.concatenate((x21,x_top[reg5],x21[reg6]))
    yr1 = np.concatenate((x12,x12[reg5],x12[reg6]))
    yr2 = np.concatenate((x22,x22[reg5],y_rig[reg6]))
    
    # Do all triangular integrals in one go
    Ct  = np.concatenate((C[:,:,reg2],C[:,:,reg3],C[:,:,reg5],C[:,:,reg6],C[:,:,reg7],C[:,:,reg8],C[:,:,reg11],C[:,:,reg12]),axis=-1)
    xt1 = np.concatenate((x11[reg2],x11[reg3],x_top[reg5],x11[reg6],x_top[reg7],x11[reg8],x_top[reg11],x11[reg12]))
    yt1 = np.concatenate((x12[reg2],x12[reg3],x12[reg5],y_rig[reg6],x22[reg7],x12[reg8],x22[reg11],x22[reg12]))
    xt2 = np.concatenate((x11[reg2],x11[reg3],x_top[reg5],x11[reg6],x21[reg7],x11[reg8],x21[reg11],x21[reg12]))
    yt2 = np.concatenate((y_lef[reg2],x22[reg3],x22[reg5],y_lef[reg6],x22[reg7],y_lef[reg8],x22[reg11],x22[reg12]))
    xt3 = np.concatenate((x21[reg2],x_bot[reg3],x_bot[reg5],x21[reg6],x21[reg7],x_bot[reg8],x21[reg11],x21[reg12]))
    yt3 = np.concatenate((x12[reg2], x12[reg3],x12[reg5],y_rig[reg6],y_rig[reg7],x12[reg8],x12[reg11],y_rig[reg12]))
    
    # 2. Rectangular Source Integrals (C, x1, x2, y1, y2)
    # OLD
   # dNi_rect, dMi_rect, dMj_rect = source_integrals(Cr,x1=xr1,x2=xr2,y1=yr1,y2=yr2)
    
    n_rect_total = len(xr1) # The combined length of Cr
    dNi_rect = np.zeros(n_rect_total, dtype=np.float64)
    dMi_rect = np.zeros(n_rect_total, dtype=np.float64)
    dMj_rect = np.zeros(n_rect_total, dtype=np.float64)
    
    loss_kernel_numba(dNi_rect, dMi_rect, dMj_rect, 
                     Cr.reshape(9,-1), xr1, xr2, yr1, yr2)
    
    dM_rect = dMi_rect + dMj_rect  # Total mass loss for bin-pair interaction
    
    loss_inds = np.arange(len_loss)

    dN_loss  = dNi_rect[loss_inds]
    dM_loss  = dM_rect[loss_inds]
    dMi_loss = dMi_rect[loss_inds]
    dMj_loss = dMj_rect[loss_inds]
    
    n_tri_total = len(xt1)
    dN_gain_tri = np.zeros(n_tri_total, dtype=np.float64)
    dM_gain_tri = np.zeros(n_tri_total, dtype=np.float64)
    
    # OLD
    #dN_gain_tri, dM_gain_tri = tri_gain_integrals(Ct, xt1, yt1, xt2, yt2, xt3, yt3, w, L)
    
    tri_kernel_numba(dN_gain_tri, dM_gain_tri,
                     Ct.reshape(9,-1), 
                     xt1, yt1, xt2, yt2, xt3, yt3,
                     w, L)
    
    Ml2 = dM_loss[reg2]
    Ml3 = dM_loss[reg3]
    Ml4 = dM_loss[reg4]
    Ml5 = dM_loss[reg5]
    Ml6 = dM_loss[reg6]
    Ml7 = dM_loss[reg7]
    Ml8 = dM_loss[reg8]
    Ml9 = dM_loss[reg9]
    Ml10 = dM_loss[reg10]
    Ml11 = dM_loss[reg11]
    Ml12 = dM_loss[reg12]
    
    Mr5 = dM_rect[len_loss:len_loss+l5]
    Mr6 = dM_rect[len_loss+l5:]
    
    Nr5 = dNi_rect[len_loss:len_loss+l5]
    Nr6 = dNi_rect[len_loss+l5:]
    
    Nl2 = dN_loss[reg2]
    Nl3 = dN_loss[reg3]
    Nl4 = dN_loss[reg4]
    Nl5 = dN_loss[reg5]
    Nl6 = dN_loss[reg6]
    Nl7 = dN_loss[reg7]
    Nl8 = dN_loss[reg8]
    Nl9 = dN_loss[reg9]
    Nl10 = dN_loss[reg10]
    Nl11 = dN_loss[reg11]
    Nl12 = dN_loss[reg12]
    
    # Define cumulative offsets for the triangle stack
    t_lens = [l2, l3, l5, l6, l7, l8, l11, l12]
    t_offs = np.insert(np.cumsum(t_lens), 0, 0)
    
    # Use these offsets to pull Mt segments
    Mt2 = dM_gain_tri[t_offs[0]:t_offs[1]]
    Mt3 = dM_gain_tri[t_offs[1]:t_offs[2]]
    Mt5 = dM_gain_tri[t_offs[2]:t_offs[3]]
    Mt6 = dM_gain_tri[t_offs[3]:t_offs[4]]
    Mt7 = dM_gain_tri[t_offs[4]:t_offs[5]]
    Mt8 = dM_gain_tri[t_offs[5]:t_offs[6]]
    Mt11 = dM_gain_tri[t_offs[6]:t_offs[7]]
    Mt12 = dM_gain_tri[t_offs[7]:]
    
    # Use these offsets to pull Nt segments
    Nt2 = dN_gain_tri[t_offs[0]:t_offs[1]]
    Nt3 = dN_gain_tri[t_offs[1]:t_offs[2]]
    Nt5 = dN_gain_tri[t_offs[2]:t_offs[3]]
    Nt6 = dN_gain_tri[t_offs[3]:t_offs[4]]
    Nt7 = dN_gain_tri[t_offs[4]:t_offs[5]]
    Nt8 = dN_gain_tri[t_offs[5]:t_offs[6]]
    Nt11 = dN_gain_tri[t_offs[6]:t_offs[7]]
    Nt12 = dN_gain_tri[t_offs[7]:]
    
    Mrt5 = Mr5+Mt5 
    Mrt6 = Mr6+Mt6 
    
    Nrt5 = Nr5+Nt5 
    Nrt6 = Nr6+Nt6 
    
    E2, E3, E4   = Eagg[reg2], Eagg[reg3], Eagg[reg4]
    E5, E6, E7   = Eagg[reg5], Eagg[reg6], Eagg[reg7]
    E8, E9, E10  = Eagg[reg8], Eagg[reg9], Eagg[reg10]
    E11, E12     = Eagg[reg11], Eagg[reg12]
    
    # Concatenate all regions in order and do residual for bin partitioning
    # dM_gain = np.concatenate((Mt2,Ml2-Mt2,
    #                           Mt3,Ml3-Mt3,
    #                           Ml4, 
    #                           Mrt5,Ml5-Mrt5,
    #                           Mrt6,Ml6-Mrt6, 
    #                           Mt7,Ml7-Mt7, 
    #                           Mt8,Ml8-Mt8, 
    #                           Ml9, 
    #                           Ml10,
    #                           Mt11,Ml11-Mt11, 
    #                           Mt12,Ml12-Mt12))
    
    # dN_gain = np.concatenate((Nt2,Nl2-Nt2,
    #                           Nt3,Nl3-Nt3,
    #                           Nl4, 
    #                           Nrt5,Nl5-Nrt5,
    #                           Nrt6,Nl6-Nrt6, 
    #                           Nt7,Nl7-Nt7, 
    #                           Nt8,Nl8-Nt8, 
    #                           Nl9, 
    #                           Nl10,
    #                           Nt11,Nl11-Nt11, 
    #                           Nt12,Nl12-Nt12))
    
    dM_gain = np.concatenate((Mt2*E2,(Ml2-Mt2)*E2,
                              Mt3*E3,(Ml3-Mt3)*E3,
                              Ml4*E4, 
                              Mrt5*E5,(Ml5-Mrt5)*E5,
                              Mrt6*E6,(Ml6-Mrt6)*E6, 
                              Mt7*E7,(Ml7-Mt7)*E7, 
                              Mt8*E8,(Ml8-Mt8)*E8, 
                              Ml9*E9, 
                              Ml10*E10,
                              Mt11*E11,(Ml11-Mt11)*E11, 
                              Mt12*E12,(Ml12-Mt12)*E12))
    
    dN_gain = np.concatenate((Nt2*E2,(Nl2-Nt2)*E2,
                              Nt3*E3,(Nl3-Nt3)*E3,
                              Nl4*E4, 
                              Nrt5*E5,(Nl5-Nrt5)*E5,
                              Nrt6*E6,(Nl6-Nrt6)*E6, 
                              Nt7*E7,(Nl7-Nt7)*E7, 
                              Nt8*E8,(Nl8-Nt8)*E8, 
                              Nl9*E9, 
                              Nl10*E10,
                              Nt11*E11,(Nl11-Nt11)*E11, 
                              Nt12*E12,(Nl12-Nt12)*E12))   
    

    k_gain = np.concatenate((kmin_l[reg2],kmid_l[reg2],    # Region 2: kmin tri
                             kmin_l[reg3],kmid_l[reg3],    # Region 3: kmin tri
                             kmin_l[reg4],                  # Region 4: kmin rect; self collection
                             kmin_l[reg5],kmid_l[reg5],    # Region 5: kmin rect + tri
                             kmin_l[reg6],kmid_l[reg6],    # Region 6: kmin rect + tri
                             kmid_l[reg7],kmin_l[reg7],    # Region 7: kmid tri
                             kmin_l[reg8],kmid_l[reg8],    # Region 8: kmin tri
                             kmin_l[reg9],                  # Region 9:kmin rect; fully in bin
                             kmid_l[reg10],                 # Region 10: kmid rect; fully in bin
                             kmid_l[reg11],kmin_l[reg11],  # Region 11: kmid tri 
                             kmid_l[reg12],kmin_l[reg12])) # Region 12: kmid tri

    h_gain = np.concatenate((hind_l[reg2],hind_l[reg2],    # Region 2: kmin tri
                             hind_l[reg3],hind_l[reg3],    # Region 3: kmin tri
                             hind_l[reg4],                     # Region 4: kmin rect; self collection
                             hind_l[reg5],hind_l[reg5],    # Region 5: kmin rect + tri
                             hind_l[reg6],hind_l[reg6],    # Region 6: kmin rect + tri
                             hind_l[reg7],hind_l[reg7],    # Region 7: kmid tri
                             hind_l[reg8],hind_l[reg8],    # Region 8: kmin tri
                             hind_l[reg9],                     # Region 9:kmin rect; fully in bin
                             hind_l[reg10],                    # Region 10: kmid rect; fully in bin
                             hind_l[reg11],hind_l[reg11],  # Region 11: kmid tri 
                             hind_l[reg12],hind_l[reg12])) # Region 12: kmid tri

    M_loss = np.zeros((dnum, Hlen, bins))
    N_loss = np.zeros((dnum, Hlen, bins))
    
    M_gain = np.zeros((dnum, Hlen, bins))
    N_gain = np.zeros((dnum, Hlen, bins))
 
    # np.add.at(M_loss, (d1_l, hind_l, bi_ind_l), dMi_loss)
    # np.add.at(M_loss, (d2_l, hind_l, bj_ind_l), dMj_loss)
    # np.add.at(N_loss, (d1_l, hind_l, bi_ind_l), dN_loss)
    # np.add.at(N_loss, (d2_l, hind_l, bj_ind_l), dN_loss)
    
    np.add.at(M_loss, (d1_l, hind_l, bi_ind_l), Ecb * dMi_loss)
    np.add.at(M_loss, (d2_l, hind_l, bj_ind_l), Ecb * dMj_loss)
    np.add.at(N_loss, (d1_l, hind_l, bi_ind_l), Ecb * dN_loss)
    np.add.at(N_loss, (d2_l, hind_l, bj_ind_l), Ecb * dN_loss)

    # Do gains all concatenated
    np.add.at(M_gain[indc],(h_gain, k_gain), dM_gain)
    np.add.at(N_gain[indc],(h_gain, k_gain), dN_gain)
    
    #M_gain[indc] *= Eagg
    #N_gain[indc] *= Eagg
    
    # Collisional breakup
    if Ebr.any() > 0.:
        
            breakup_kernel(
            M_gain, N_gain, dM_loss.ravel(), 
            dMb_kernel, dNb_kernel, Ebr,
            params['bi_ind'], params['bj_ind'], params['hind'], 
            params['kmin'],  # Pass the limit array here
            indb)
    
    return M_loss, M_gain, N_loss, N_gain




def init_rk(rk_order):
    
    # --- Butcher tableaux (a_ij, b_i, c_i) for RK1–RK4 ---
    tableaux = {
                1: dict(a=[[0]], # RK 1st order
                        b=[1], 
                        c=[0]),
                2: dict(a=[[0, 0], # RK 2nd order
                           [0.5, 0]], 
                        b=[0, 1], 
                        c=[0, 0.5]),
                3: dict(a=[[0, 0, 0], # RK 3rd order
                           [0.5, 0, 0],
                           [-1, 2, 0]], 
                        b=[1/6, 2/3, 1/6], 
                        c=[0, 0.5, 1]),
                4: dict(a=[[0, 0, 0, 0], # RK 4th order
                           [0.5, 0, 0, 0],
                           [0, 0.5, 0, 0],
                           [0, 0, 1, 0]], 
                        b=[1/6, 1/3, 1/3, 1/6], 
                        c=[0, 0.5, 0.5, 1])
                }

    if rk_order not in tableaux:
        raise ValueError("rk_order must be 1, 2, 3, or 4")

    RK = {'a':np.array(tableaux[rk_order]['a'], dtype=float),
          'b':np.array(tableaux[rk_order]['b'], dtype=float),
          'c':np.array(tableaux[rk_order]['c'], dtype=float)}
    
    return RK 


# New coefficients: Polynomial expansions for F(x,y) * Nx(x) * Ny(y)
def combined_coeffs_array(f, ax, cx, ay, cy):
    """
    Returns C00,C10,C01,C20,C11,C02,C21,C12
    such that:
    P(x,y) = sum Cij * x^i y^j
    """
    
    a,b,c,d = _unpack_f(f)
    
    C = np.zeros((3,3,len(ax)),dtype=np.float64)

    C[0,0,:] = a*cx*cy # C00

    C[1,0,:] = (b*cx + a*ax)*cy # C10 * x
    C[0,1,:] = (c*cy + a*ay)*cx # C01 * y

    C[2,0,:] = b*ax*cy # C20 * x^2
    C[0,2,:] = c*cx*ay # C02 * y^2

    # C11 * x * y
    C[1,1,:] = (
        d*cx*cy +
        b*cx*ay +
        c*ax*cy +
        a*ax*ay
    )

    C[2,1,:] = (d*cy + b*ay)*ax    # C21 * x^2 * y
    C[1,2,:] = (d*cx + c*ax)*ay    # C12 * x * y^2
    
    C[2,2,:] = ax*ay*d

    return C


def rect_integrals(C00,C10,C01,C20,C11,C02,C21,C12,C22,x1,x2,y1,y2):
    """
    Tensor-based rectangular integration for N rectangles.
    Calculates Int(P), Int(xP), Int(yP), and Int((x+y)P).
    """
    # 1. Fundamental Differences and Sums
    dx0, dy0 = x2 - x1, y2 - y1
    dsx0, dsy0 = x2 + x1, y2 + y1
    x1x2, y1y2 = x1 * x2, y1 * y2
    
    # 2. X and Y Power Basis (The 1D Integrals)
    # DX has shape (4, N) representing integrals of [x^0, x^1, x^2, x^3]
    DX = np.stack([
        dx0,
        0.5 * dx0 * dsx0,
        (1.0/3.0) * dx0 * (dsx0**2 - x1x2),
        0.25 * dx0 * dsx0 * (dsx0**2 - 2.0 * x1x2)
    ])
    
    # DY has shape (4, N) representing integrals of [y^0, y^1, y^2, y^3]
    DY = np.stack([
        dy0,
        0.5 * dy0 * dsy0,
        (1.0/3.0) * dy0 * (dsy0**2 - y1y2),
        0.25 * dy0 * dsy0 * (dsy0**2 - 2.0 * y1y2)
    ])

    # 3. Coefficient Tensor (3, 3, N)
    # Structured as C[i, j, n] where i is x-power and j is y-power
    C_tensor = np.zeros((3, 3, x1.shape[0]))
    C_tensor[0, 0] = C00
    C_tensor[1, 0] = C10
    C_tensor[2, 0] = C20
    C_tensor[0, 1] = C01
    C_tensor[1, 1] = C11
    C_tensor[2, 1] = C21
    C_tensor[0, 2] = C02
    C_tensor[1, 2] = C12
    C_tensor[2, 2] = C22

    # 4. Tensor Contractions via einsum
    # n = rectangle index, i = x-power index, j = y-power index
    
    # Integral 1: Int(P) = Sum(C_ijn * DX_in * DY_jn)
    # DX[:3] and DY[:3] because P only goes up to x^2, y^2
    int_P = np.einsum('ijn,in,jn->n', C_tensor, DX[:3], DY[:3])

    # Integral 2: Int(xP) = Sum(C_ijn * DX_{i+1,n} * DY_jn)
    # We slice DX[1:4] to get the shifted x-powers [x^1, x^2, x^3]
    int_xP = np.einsum('ijn,in,jn->n', C_tensor, DX[1:4], DY[:3])

    # Integral 3: Int(yP) = Sum(C_ijn * DX_in * DY_{j+1,n})
    int_yP = np.einsum('ijn,in,jn->n', C_tensor, DX[:3], DY[1:4])

    return int_P,int_xP,int_yP

def source_integrals(C,x1,x2,y1,y2):
    """
    Tensor-based rectangular integration for N rectangles.
    Calculates Int(P), Int(xP), Int(yP), and Int((x+y)P).
    """
    # 1. Fundamental Differences and Sums
    dx0, dy0 = x2 - x1, y2 - y1
    dsx0, dsy0 = x2 + x1, y2 + y1
    x1x2, y1y2 = x1 * x2, y1 * y2
    
    # 2. X and Y Power Basis (The 1D Integrals)
    # DX has shape (4, N) representing integrals of [x^0, x^1, x^2, x^3]
    DX = np.stack([
        dx0,
        0.5 * dx0 * dsx0,
        (1.0/3.0) * dx0 * (dsx0**2 - x1x2),
        0.25 * dx0 * dsx0 * (dsx0**2 - 2.0 * x1x2)
    ])
    
    # DY has shape (4, N) representing integrals of [y^0, y^1, y^2, y^3]
    DY = np.stack([
        dy0,
        0.5 * dy0 * dsy0,
        (1.0/3.0) * dy0 * (dsy0**2 - y1y2),
        0.25 * dy0 * dsy0 * (dsy0**2 - 2.0 * y1y2)
    ])

    # Tensor Contractions via einsum
    # n = rectangle index, i = x-power index, j = y-power index
    
    # Integral 1: Int(P) = Sum(C_ijn * DX_in * DY_jn)
    # DX[:3] and DY[:3] because P only goes up to x^2, y^2
    #int_P = np.einsum('ijn,in,jn->n', C_tensor, DX[:3], DY[:3])
    int_P = np.einsum('ijn,in,jn->n', C, DX[:3], DY[:3])

    # Integral 2: Int(xP) = Sum(C_ijn * DX_{i+1,n} * DY_jn)
    # We slice DX[1:4] to get the shifted x-powers [x^1, x^2, x^3]
    #int_xP = np.einsum('ijn,in,jn->n', C_tensor, DX[1:4], DY[:3])
    int_xP = np.einsum('ijn,in,jn->n', C, DX[1:4], DY[:3])

    # Integral 3: Int(yP) = Sum(C_ijn * DX_in * DY_{j+1,n})
    #int_yP = np.einsum('ijn,in,jn->n', C_tensor, DX[:3], DY[1:4])
    int_yP = np.einsum('ijn,in,jn->n', C, DX[:3], DY[1:4])

    return int_P, int_xP, int_yP

#def rect_gain_integrals(C00,C10,C01,C20,C11,C02,C21,C12,C22,x1,x2,y1,y2):
def rect_gain_integrals(C,x1,x2,y1,y2):
    """
    Tensor-based rectangular integration for N rectangles.
    Calculates Int(P), Int(xP), Int(yP), and Int((x+y)P).
    """
    # 1. Fundamental Differences and Sums
    int_P, int_xP, int_yP = source_integrals(C, x1, x2, y1, y2)

    int_xyP = int_xP + int_yP

    return int_P,int_xyP


def tri_gain_integrals(C, xt1, yt1, xt2, yt2, xt3, yt3, w, L):
    """
    Direct contraction of a (3, 3, N) coefficient tensor.
    i: x-power (0,1,2), j: y-power (0,1,2), n: triangle index
    """
    
    # 1. Signed Area (N,)
    area = 0.5 * np.abs(xt1 * (yt2 - yt3) + xt2 * (yt3 - yt1) + xt3 * (yt1 - yt2))

    # 3. Map Quadrature Points (7, N)
    qx = L[:, 0, None] * xt1 + L[:, 1, None] * xt2 + L[:, 2, None] * xt3
    qy = L[:, 0, None] * yt1 + L[:, 1, None] * yt2 + L[:, 2, None] * yt3

    # 4. Construct 4D Monomial Tensor (3, 3, 7, N)
    # x_pow and y_pow represent [1, x, x^2] and [1, y, y^2] at all points
    x_pow = np.stack([np.ones_like(qx), qx, qx**2], axis=0) # (3, 7, N)
    y_pow = np.stack([np.ones_like(qy), qy, qy**2], axis=0) # (3, 7, N)
    
    # Outer product to get all i,j combinations: (3, 3, 7, N)
    # i: x index, j: y index, q: quad index, n: triangle index
    #monos = x_pow[:, None, :, :] * y_pow[None, :, :, :]
    monos = np.einsum('iqn,jqn->ijqn', x_pow, y_pow)
    #monos = np.einsum('iq...,jq...->ijq...', x_pow, y_pow)

    # 5. Tensor Contraction
    # Int(P): Multiply C[i,j,n] by Monomials[i,j,q,n] and Weights[q]
    # Sum over i, j, and q.
    avg_P = area*np.einsum('ijn,ijqn,q->n', C, monos, w)
    #avg_P = area*np.einsum('ij...,ijq...,q->...', C, monos, w)

    # Int((x+y)P): Simply add the (qx + qy) factor during contraction
    avg_xyP = area*np.einsum('ijn,ijqn,qn,q->n', C, monos, qx + qy, w)
    #avg_xyP = area*np.einsum('ij...,ijq...,q...,q->...', C, monos, qx + qy, w)

    return avg_P, avg_xyP

 
def setup_regions(bins,kr,ir,jr,x11,x21,x12,x22,xk_min):

    # (batch,)
    x_bottom_edge = (xk_min-x12)
    x_top_edge    = (xk_min-x22)
    y_left_edge   = (xk_min-x11)
    y_right_edge  = (xk_min-x21)
    
    check_bottom = (x11<x_bottom_edge) &\
                   (x21>x_bottom_edge)
     
    check_top = (x11<x_top_edge) &\
                (x21>x_top_edge)
                
    check_left = (x12<y_left_edge) &\
                 (x22>y_left_edge)
                
    check_right = (x12<y_right_edge) &\
                  (x22>y_right_edge)            
           
    check_middle = ((0.5*(x11+x21))+(0.5*(x12+x22)))<(xk_min)
    
               
    # If opposite sides check true, then integral region is rectangle + triangle
    # If adjacent sides check true or single side, then integral region is triangle       
           
    # Check which opposite side is higher for cases where we have rectangle + triangle
    # NOTE: It SHOULD be the case that y_left_edge>y_right_edge and x_bottom_edge>x_top_edge
    # This just has to do with the geometry of the x+y mapping, i.e., the x+y lines have negative slope.
    
    '''
    Vectorized Integration Regions:
    cond_1 :  Ignore CC process for these source bins; they don't map to the largest avail bin.
    cond_2 :  k bin: Lower triangle region. Just clips BR corner. Occurs on diagonal+1 indices.
                       Triangle = ((xi1,xj1),(xi1,y_left_edge),(xi2,xj1))
    cond_3 :  k bin: Lower triangle region. Just clips UL corner. Occurs on diagonal-1 indices.
                       Triangle = ((xi1,xj1),(xi1,xj2),(x_bottom_edge,xj1))                 
    cond_2b : k+1 bins: Upper triangle region. Just clips BR corner.
                       Triangle = ((xi1,xj1),(x_top_edge,xj2),(xi2,xj2))
    cond_3b : k+1 bin: Upper triangle region. Just clips UL corner.
                             Triangle = ((xi1,xj2),(xi2,xj2),(xi2,y_right_edge)                    
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
    cond_9:   k bin: Rectangle collection within k bin. All Mass/Number goes into kbin
    cond_10:  k+1 bin: Rectangle collection within k+1 bin. All Mass/Number goes into kbin
    '''
         
    # (batch,)
    cond_touch = (check_bottom|check_top|check_left|check_right)
    
    # NEW 2
    cond_BR_corner = (x21==x_bottom_edge)&(x12==y_right_edge)
    cond_UL_corner = (x11==x_top_edge)&(x22==y_left_edge)
    
    cond_2_corner = (cond_BR_corner)&(check_left)
    cond_3_corner = (cond_UL_corner)&(check_bottom)
    
    cond_2b_corner = (cond_BR_corner)&(check_top)
    cond_3b_corner = (cond_UL_corner)&(check_right)
    
    cond_2 = np.eye(bins,k=1,dtype=bool)[ir,jr] & (cond_2_corner) & (cond_touch)
    cond_3 = np.eye(bins,k=-1,dtype=bool)[ir,jr] & (cond_3_corner) & (cond_touch)
    
    cond_2b = cond_2b_corner
    cond_3b = cond_3b_corner
    
    cond_4 = np.eye(bins,dtype=bool)[ir,jr]
    cond_nt = (~(cond_2|cond_3|cond_4))
    cond_5 = (check_top&check_bottom)  & cond_nt
    cond_6 = (check_left&check_right)  & cond_nt
    cond_7 =  (check_right&check_top)  & cond_nt
    cond_8 = (check_left&check_bottom) & cond_nt
    cond_rect = (~cond_touch)&(~cond_4)&(~cond_5)&(~cond_6)&(~cond_7)&(~cond_8)
    cond_9 = (cond_rect&check_middle)
    cond_10 = (cond_rect&(~check_middle)) 
    
    # NOTE: New method. Just use integers to represent region type
    inds = np.zeros_like(kr)
    
    inds[cond_2] = 2
    inds[cond_3] = 3 
    inds[cond_4] = 4 
    inds[cond_5] = 5 
    inds[cond_6] = 6 
    inds[cond_7] = 7
    inds[cond_8] = 8 
    inds[cond_9] = 9 
    inds[cond_10] = 10 
    inds[cond_2b] = 11 
    inds[cond_3b] = 12
    
    return inds, x_bottom_edge, x_top_edge, y_left_edge, y_right_edge


def calculate_rates(Hlen,bins,region_inds,x_bottom_edge,x_top_edge,y_left_edge,y_right_edge,
                         kr,ir,jr,x11,x21,ak1,ck1,x12,x22,ak2,ck2,PK,kmin,kmid,
                         dMb_gain_frac,dNb_gain_frac,w,L,breakup=False):    

    '''
    Description: Calcualates Wang et al. (2007) integrals using numpy functions.
    '''
    
    '''
    # kr = Height index    (batch,)
    # ir = collectee index (batch,)
    # jr = collector index (batch,) 

    Vectorized Integration Regions:
    cond_1 :  All rectangular bin-pair interactions used
    cond_2 :  k bin: Lower triangle region. Just clips BR corner.
                       Triangle = ((xi1,xj1),(xi1,y_left_edge),(xi2,xj1))
    cond_3 :  k bin: Lower triangle region. Just clips UL corner.
                       Triangle = ((xi1,xj1),(xi1,xj2),(x_bottom_edge,xj1))                 
    cond_2b : k+1 bins: Upper triangle region. Just clips BR corner.
                       Triangle = ((xi1,xj1),(x_top_edge,xj2),(xi2,xj2))
    cond_3b : k+1 bin: Upper triangle region. Just clips UL corner.
                             Triangle = ((xi1,xj2),(xi2,xj2),(xi2,y_right_edge)                    
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
    cond_9:   k bin: Rectangle collection within k bin. All Mass/Number goes into kbin
    cond_10:  k+1 bin: Rectangle collection within k+1 bin. All Mass/Number goes into kbin
    '''
    
    # Indices for (batch,)
    # (kr[ind],ir[ind],jr[ind]) gives mapping for
    # (Hlen,i bins,j bins)
    k2 = np.flatnonzero(region_inds==2)
    k3 = np.flatnonzero(region_inds==3) 
    k4 = np.flatnonzero(region_inds==4) 
    k5 = np.flatnonzero(region_inds==5) 
    k6 = np.flatnonzero(region_inds==6) 
    k7 = np.flatnonzero(region_inds==7)
    k8 = np.flatnonzero(region_inds==8) 
    k9 = np.flatnonzero(region_inds==9) 
    k10 = np.flatnonzero(region_inds==10) 
    k2b = np.flatnonzero(region_inds==11) 
    k3b = np.flatnonzero(region_inds==12)
    
    klen = len(region_inds)
    
    # Initialize gain term arrays
    dMi_loss = np.zeros((klen,)) # i Mass loss for each bin-pair (k,i,j)
    dMj_loss = np.zeros((klen,)) # j Mass loss for each bin-pair (k,i,j)
    dNi_loss = np.zeros((klen,)) # i (j) Number loss for each bin-pair (k,i,j)
    dM_gain  = np.zeros((klen,2)) # Mass gain for each bin-pair (k,i,j)
    dN_gain  = np.zeros((klen,2)) # Number gain for each bin-pair (k,i,j)
    
    # NOTE PRECOMPUTE MASS/NUMBER COEFFICIENTS HERE FOR ALL BIN-PAIR INTERACTIONS
    # This is: F(x,y)* Nx(x) * Ny(y) = (F=a+b*x+c*y+d*x*y)*(Nx=ax*x+cx)*(Ny=ay*y+cy)
    C = combined_coeffs_array(PK, ak1, ck1, ak2, ck2)

    # Calculate source bin-pair rectangle integral factors. Note, tried to
    # combine as many of these factors together as possible to avoid redundant calculations.
    dNi_loss, dMi_loss, dMj_loss = source_integrals(C,x1=x11,x2=x21,y1=x12,y2=x22)
    
    dM_loss = dMi_loss+dMj_loss # Total bin-pair mass loss
    
    # NOTE: Try (Hlen, bins, batch) and then sum along batch axis?
    
    dM_gain[k4,0]  = dM_loss[k4].copy()
    dN_gain[k4,0]  = dNi_loss[k4].copy()

    # Condition 2:
    # k bin: Lower triangle region. Just clips BR corner.
    #                       Triangle = ((xi1,xj1),(xi1,y_left_edge),(xi2,xj1))
    if len(k2)>0:
        xt1 = x11[k2]
        yt1 = x12[k2]
        xt2 = x11[k2]
        yt2 = y_left_edge[k2]
        xt3 = x21[k2]
        yt3 = x12[k2]
        
        dN_gain_temp, dM_gain_temp = tri_gain_integrals(C[:,:,k2], xt1, yt1, xt2, yt2, xt3, yt3, w, L)
        
        dN_gain[k2,0] = dN_gain_temp.copy() 
        dM_gain[k2,0] = dM_gain_temp.copy()
        
        dM_gain[k2,1] = (dM_loss[k2]-dM_gain_temp)
        dN_gain[k2,1] = (dNi_loss[k2]-dN_gain_temp)


    # Condition 3:
    #    k bin: Lower triangle region. Just clips UL corner.
    #                     Triangle = ((xi1,xj1),(xi1,xj2),(x_bottom_edge,xj1))  
    

    if len(k3)>0:
    
        xt1 = x11[k3]
        yt1 = x12[k3]
        xt2 = x11[k3]
        yt2 = x22[k3]
        xt3 = x_bottom_edge[k3]
        yt3 = x12[k3]
         
        dN_gain_temp,dM_gain_temp = tri_gain_integrals(C[:,:,k3], xt1, yt1, xt2, yt2, xt3, yt3, w, L)
        
        dN_gain[k3,0] = dN_gain_temp.copy()
        dM_gain[k3,0] = dM_gain_temp.copy()
        
        dM_gain[k3,1] = (dM_loss[k3]-dM_gain_temp)
        dN_gain[k3,1] = (dNi_loss[k3]-dN_gain_temp)
        
            
    # Condition 5: 
        
    #    k bin: Top/Bottom clip: Rectangle on left, triangle on right
    #                              Rectangle = ((xi1,xj1),(xi1,xj2),(x_top_edge,xj2),(x_top_edge,xj1))
    #                              Triangle  = ((x_top_edge,xj1),(x_top_edge,xj2),(x_bottom_edge,xj1))
   
    if len(k5)>0: 
       
        xr1 = x11[k5]
        yr1 = x12[k5]
        xr2 = x_top_edge[k5]
        yr2 = x22[k5]
       
        xt1 = x_top_edge[k5]
        yt1 = x12[k5]
        xt2 = x_top_edge[k5]
        yt2 = x22[k5]
        xt3 = x_bottom_edge[k5]
        yt3 = x12[k5]
        
        rect_000, rect_001 = rect_gain_integrals(C[:,:,k5],x1=xr1,x2=xr2,y1=yr1,y2=yr2)
        tri_000, tri_001   =  tri_gain_integrals(C[:,:,k5],xt1, yt1, xt2, yt2, xt3, yt3, w, L)
        
        dN_gain_temp = rect_000+tri_000
        dM_gain_temp = rect_001+tri_001
        
        dM_gain[k5,0] = dM_gain_temp.copy()
        dN_gain[k5,0] = dN_gain_temp.copy()
        
        dM_gain[k5,1] = (dM_loss[k5]-dM_gain_temp)
        dN_gain[k5,1] = (dNi_loss[k5]-dN_gain_temp)
        
        
    # Condition 6:
    # k bin: Left/Right clip: Rectangle on bottom, triangle on top
    #                          Rectangle = ((xi1,xj1),(xi1,y_right_edge),(xi2,y_right_edge),(xi2,xj1))
    #                          Triangle  = ((xi1,y_right_edge),(xi1,y_left_edge),(xi2,y_right_edge))
       
    if len(k6)>0:
        xr1 = x11[k6]
        yr1 = x12[k6]
        xr2 = x21[k6]
        yr2 = y_right_edge[k6]
        
        xt1 = x11[k6]
        yt1 = y_right_edge[k6]
        xt2 = x11[k6]
        yt2 = y_left_edge[k6]
        xt3 = x21[k6]
        yt3 = y_right_edge[k6]
           
        rect_000, rect_001 = rect_gain_integrals(C[:,:,k6],x1=xr1,x2=xr2,y1=yr1,y2=yr2)
        tri_000, tri_001   =  tri_gain_integrals(C[:,:,k6], xt1, yt1, xt2, yt2, xt3, yt3, w, L)
        
        dN_gain_temp = rect_000+tri_000 
        dM_gain_temp = rect_001+tri_001
         
        dM_gain[k6,0] = dM_gain_temp.copy()
        dM_gain[k6,1] = (dM_loss[k6]-dM_gain_temp)
            
        dN_gain[k6,0] = dN_gain_temp.copy()
        dN_gain[k6,1] = (dNi_loss[k6]-dN_gain_temp)
        
        
    # Condition 7:
    # k+1 bin: Triangle in top right corner
    #                            Triangle = ((x_top_edge,xj2),(xi2,xj2),(xi2,y_right_edge))
    
    if len(k7)>0: 
        xt1 = x_top_edge[k7]
        yt1 = x22[k7]
        xt2 = x21[k7]
        yt2 = x22[k7]
        xt3 = x21[k7]
        yt3 = y_right_edge[k7]
         
        dN_gain_temp,dM_gain_temp = tri_gain_integrals(C[:,:,k7], xt1, yt1, xt2, yt2, xt3, yt3, w, L)
        
        dN_gain[k7,1] = dN_gain_temp.copy()
        dM_gain[k7,1] = dM_gain_temp.copy()
        
        dM_gain[k7,0] = (dM_loss[k7]-dM_gain_temp)
        dN_gain[k7,0] = (dNi_loss[k7]-dN_gain_temp)
        
       
    # Condition 8:
    #  k bin: Triangle in lower left corner
    #                                Triangle = ((xi1,xj1),(xi1,y_left_edge),(x_bottom_edge,xj1))
    if len(k8)>0:
        xt1 = x11[k8]
        yt1 = x12[k8]
        xt2 = x11[k8]
        yt2 = y_left_edge[k8]
        xt3 = x_bottom_edge[k8]
        yt3 = x12[k8]
        
        dN_gain_temp,dM_gain_temp = tri_gain_integrals(C[:,:,k8], xt1, yt1, xt2, yt2, xt3, yt3, w, L)

        dN_gain[k8,0] = dN_gain_temp.copy()
        dM_gain[k8,0] = dM_gain_temp.copy()
        
        dM_gain[k8,1] = (dM_loss[k8]-dM_gain_temp)
        dN_gain[k8,1] = (dNi_loss[k8]-dN_gain_temp)
        
        
    # Condition 9: Rectangle collection within k bin. All Mass/Number goes into kbin

    xr1 = x11[k9]
    xr2 = x21[k9] 
    yr1 = x12[k9]
    yr2 = x22[k9]

    dM_gain[k9,0]  = dM_loss[k9].copy()
    dN_gain[k9,0]  = dNi_loss[k9].copy()
    

    # Condition 10: Rectangle collection within k+1 bin. All Mass/Number goes into k+1bin
    
    xr1 = x11[k10]
    xr2 = x21[k10] 
    yr1 = x12[k10]
    yr2 = x22[k10]

    dM_gain[k10,1]  = dM_loss[k10].copy()
    dN_gain[k10,1]  = dNi_loss[k10].copy()
    
   
    # Condition 11 (2b): Triangle collection within k+1 bin. Occurs when xk+1 clips BR corner and intersects top edge.
    # Triangle = ((xi2,xj1),(x_top_edge,xj2),(xi2,xj2))
    
    if len(k2b)>0:
        xt1 = x_top_edge[k2b]
        yt1 = x22[k2b]
        xt2 = x21[k2b]
        yt2 = x22[k2b]
        xt3 = x21[k2b]
        yt3 = x12[k2b]
        
        dN_gain_temp,dM_gain_temp = tri_gain_integrals(C[:,:,k2b], xt1, yt1, xt2, yt2, xt3, yt3, w, L)
        
        dN_gain[k2b,1] = dN_gain_temp.copy() 
        dM_gain[k2b,1] = dM_gain_temp.copy()
        
        dM_gain[k2b,0] = (dM_loss[k2b]-dM_gain_temp)
        dN_gain[k2b,0] = (dNi_loss[k2b]-dN_gain_temp)
        
        
    # Condition 12 (3b): Triangle collection within k+1 bin. Occurs when xk+1 clips UL corner and intersects with right edge.
    # Triangle = ((xi1,xj2),(xi2,xj2),(xi2,y_right_edge)
    if len(k3b)>0:
        xt1 = x11[k3b]
        yt1 = x22[k3b]
        xt2 = x21[k3b]
        yt2 = x22[k3b]
        xt3 = x21[k3b]
        yt3 = y_right_edge[k3b]
        
        dN_gain_temp,dM_gain_temp = tri_gain_integrals(C[:,:,k3b], xt1, yt1, xt2, yt2, xt3, yt3, w, L)
        
        dN_gain[k3b,1] = dN_gain_temp.copy()
        dM_gain[k3b,1] = dM_gain_temp.copy()
        
        dM_gain[k3b,0] = (dM_loss[k3b]-dM_gain_temp)
        dN_gain[k3b,0] = (dNi_loss[k3b]-dN_gain_temp)  
        
    
    return dMi_loss, dMj_loss, dM_loss, dM_gain, dNi_loss, dN_gain

# vvvv Note, is this a better way? vvvv
def transfer_bins(Hlen,bins,kr,ir,jr,kmin,kmid,dMi_loss,dMj_loss,dM_loss,dM_gain,
                  dNi_loss,dN_gain,dMb_gain_frac,dNb_gain_frac,breakup=False):

    # DO TRANSFER HERE
    # Initialize gain term arrays
    M1_loss = np.zeros((Hlen,bins))
    M2_loss = np.zeros((Hlen,bins))
    M_gain  = np.zeros((Hlen,bins))
    Mb_gain = np.zeros((Hlen,bins))

    N1_loss = np.zeros((Hlen,bins))
    N2_loss = np.zeros((Hlen,bins))
    N_gain  = np.zeros((Hlen,bins))
    Nb_gain = np.zeros((Hlen,bins))
  
    np.add.at(M1_loss,(kr,ir),dMi_loss)
    np.add.at(M2_loss,(kr,jr),dMj_loss)

    np.add.at(M_gain,(kr,kmin),dM_gain[:,0])
    np.add.at(M_gain,(kr,kmid),dM_gain[:,1])
    
    np.add.at(N1_loss,(kr,ir),dNi_loss)
    np.add.at(N2_loss,(kr,jr),dNi_loss)
    
    np.add.at(N_gain,(kr,kmin),dN_gain[:,0])
    np.add.at(N_gain,(kr,kmid),dN_gain[:,1])
      
    # ELD NOTE: Breakup here can take losses from each pair and calculate gains
    # for breakup. Breakup gain arrays will be 3D.
    if breakup:
                
        np.add.at(Mb_gain,  kr, np.transpose(dMb_gain_frac[:,kmin]*dM_loss))
        np.add.at(Nb_gain,  kr, np.transpose(dNb_gain_frac[:,kmin]*dM_loss))
    
    return M1_loss, M2_loss, M_gain, Mb_gain, N1_loss, N2_loss, N_gain, Nb_gain


def calculate_regions_batch(Hlen,bins,kr,ir,jr,x11,x21,ak1,ck1,x12,x22,ak2,ck2,PK,region_inds,
                            x_bottom_edge,x_top_edge,y_left_edge,y_right_edge,w,L):    

    # kr = Height index    (batch,)
    # ir = collectee index (batch,)
    # jr = collector index (batch,) 

    '''
    Vectorized Integration Regions:
    cond_1 :  All rectangular bin-pair interactions used
    cond_2 :  k bin: Lower triangle region. Just clips BR corner.
                       Triangle = ((xi1,xj1),(xi1,y_left_edge),(xi2,xj1))
    cond_3 :  k bin: Lower triangle region. Just clips UL corner.
                       Triangle = ((xi1,xj1),(xi1,xj2),(x_bottom_edge,xj1))                 
    cond_2b : k+1 bins: Upper triangle region. Just clips BR corner.
                       Triangle = ((xi1,xj1),(x_top_edge,xj2),(xi2,xj2))
    cond_3b : k+1 bin: Upper triangle region. Just clips UL corner.
                             Triangle = ((xi1,xj2),(xi2,xj2),(xi2,y_right_edge)                    
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
    cond_9:   k bin: Rectangle collection within k bin. All Mass/Number goes into kbin
    cond_10:  k+1 bin: Rectangle collection within k+1 bin. All Mass/Number goes into kbin
    '''
    
    #x_bottom_edge = regions['x_bottom_edge']
    #x_top_edge = regions['x_top_edge']
    #y_left_edge = regions['y_left_edge']
    #y_right_edge = regions['y_right_edge']
    #inds = regions['inds'] # Array of region type described by integer value
    
    k2 = np.flatnonzero(region_inds==2)
    k3 = np.flatnonzero(region_inds==3) 
    k4 = np.flatnonzero(region_inds==4) 
    k5 = np.flatnonzero(region_inds==5) 
    k6 = np.flatnonzero(region_inds==6) 
    k7 = np.flatnonzero(region_inds==7)
    k8 = np.flatnonzero(region_inds==8) 
    k9 = np.flatnonzero(region_inds==9) 
    k10 = np.flatnonzero(region_inds==10) 
    k2b = np.flatnonzero(region_inds==11) 
    k3b = np.flatnonzero(region_inds==12)
    
    # Initialize gain term arrays
    dMi_loss = np.zeros((Hlen,bins,bins))
    dMj_loss = np.zeros((Hlen,bins,bins))
    dNi_loss = np.zeros((Hlen,bins,bins))
    dM_gain  = np.zeros((Hlen,bins,bins,2))
    dN_gain  = np.zeros((Hlen,bins,bins,2))
    
    # NEW METHOD
    
    # kr_len = len(kr)
    
    # dMi_loss = np.zeros((Hlen,kr_len))
    # dMj_loss = np.zeros((Hlen,kr_len))
    # dNi_loss = np.zeros((Hlen,kr_len))
    # dM_gain  = np.zeros((Hlen,kr_len,2))
    # dN_gain  = np.zeros((Hlen,kr_len,2))


    # NOTE PRECOMPUTE MASS/NUMBER COEFFICIENTS HERE FOR ALL BIN-PAIR INTERACTIONS
    # This is: F(x,y)* Nx(x) * Ny(y) = (F=a+b*x+c*y+d*x*y)*(Nx=ax*x+cx)*(Ny=ay*y+cy)
    C = combined_coeffs_array(PK, ak1, ck1, ak2, ck2)
  
    # Calculate transfer rates (rectangular integration, source space)
    # Collection (eqs. 23-25 in Wang et al. 2007)
    # ii collecting jj 
    
    
    # NOTE: When batching, need to make sure that k1 corresponds to all k2-k3b indices.
    # This will be important for balance loading all batches while maining that the source
    # bin-pair loss values can be used in some of the gain calculations (e.g., region 4)

    # Calculate source bin-pair rectangle integral factors. Note, tried to
    # combine as many of these factors together as possible to avoid redundant calculations.
    dNi_full, dMi_full, dMj_full = source_integrals(C,x1=x11,x2=x21,y1=x12,y2=x22)
    
    dMi_loss[kr,ir,jr] = dMi_full.copy() 
    dMj_loss[kr,ir,jr] = dMj_full.copy() 
    dNi_loss[kr,ir,jr] = dNi_full.copy() 
    
    dM_loss = dMi_loss+dMj_loss
    
    dM_gain[kr[k4],ir[k4],jr[k4],0]  = dM_loss[kr[k4],ir[k4],jr[k4]].copy()
    dN_gain[kr[k4],ir[k4],jr[k4],0]  = dNi_loss[kr[k4],ir[k4],jr[k4]].copy()

    # Condition 2:
    # k bin: Lower triangle region. Just clips BR corner.
    #                       Triangle = ((xi1,xj1),(xi1,y_left_edge),(xi2,xj1))
    if len(k2)>0:
        xt1 = x11[k2]
        yt1 = x12[k2]
        xt2 = x11[k2]
        yt2 = y_left_edge[k2]
        xt3 = x21[k2]
        yt3 = x12[k2]
        
        dN_gain_temp, dM_gain_temp = tri_gain_integrals(C[:,:,k2], xt1, yt1, xt2, yt2, xt3, yt3, w, L)
        
        dN_gain[kr[k2],ir[k2],jr[k2],0] = dN_gain_temp.copy() 
        dM_gain[kr[k2],ir[k2],jr[k2],0] = dM_gain_temp.copy()
        
        dM_gain[kr[k2],ir[k2],jr[k2],1] = dM_loss[kr[k2],ir[k2],jr[k2]]-dM_gain[kr[k2],ir[k2],jr[k2],0]
        dN_gain[kr[k2],ir[k2],jr[k2],1] = dNi_loss[kr[k2],ir[k2],jr[k2]]-dN_gain[kr[k2],ir[k2],jr[k2],0]

        
    # Condition 3:
    #    k bin: Lower triangle region. Just clips UL corner.
    #                     Triangle = ((xi1,xj1),(xi1,xj2),(x_bottom_edge,xj1))  
    

    if len(k3)>0:
    
        xt1 = x11[k3]
        yt1 = x12[k3]
        xt2 = x11[k3]
        yt2 = x22[k3]
        xt3 = x_bottom_edge[k3]
        yt3 = x12[k3]
         
        dN_gain_temp,dM_gain_temp = tri_gain_integrals(C[:,:,k3], xt1, yt1, xt2, yt2, xt3, yt3, w, L)
        
        dN_gain[kr[k3],ir[k3],jr[k3],0] = dN_gain_temp.copy()
        dM_gain[kr[k3],ir[k3],jr[k3],0] = dM_gain_temp.copy()
        
        dM_gain[kr[k3],ir[k3],jr[k3],1] = dM_loss[kr[k3],ir[k3],jr[k3]]-dM_gain[kr[k3],ir[k3],jr[k3],0]
        dN_gain[kr[k3],ir[k3],jr[k3],1] = dNi_loss[kr[k3],ir[k3],jr[k3]]-dN_gain[kr[k3],ir[k3],jr[k3],0]
            
    
    # Condition 5: 
        
    #    k bin: Top/Bottom clip: Rectangle on left, triangle on right
    #                              Rectangle = ((xi1,xj1),(xi1,xj2),(x_top_edge,xj2),(x_top_edge,xj1))
    #                              Triangle  = ((x_top_edge,xj1),(x_top_edge,xj2),(x_bottom_edge,xj1))
   
    if len(k5)>0: 
       
        xr1 = x11[k5]
        yr1 = x12[k5]
        xr2 = x_top_edge[k5]
        yr2 = x22[k5]
       
        xt1 = x_top_edge[k5]
        yt1 = x12[k5]
        xt2 = x_top_edge[k5]
        yt2 = x22[k5]
        xt3 = x_bottom_edge[k5]
        yt3 = x12[k5]
        
        rect_000, rect_001 = rect_gain_integrals(C[:,:,k5],x1=xr1,x2=xr2,y1=yr1,y2=yr2)
        tri_000, tri_001   =  tri_gain_integrals(C[:,:,k5],xt1, yt1, xt2, yt2, xt3, yt3, w, L)
        
        
        dM_gain[kr[k5],ir[k5],jr[k5],0] = rect_001+tri_001
        
        dM_gain[kr[k5],ir[k5],jr[k5],1] = dM_loss[kr[k5],ir[k5],jr[k5]]-dM_gain[kr[k5],ir[k5],jr[k5],0]
            
        dN_gain[kr[k5],ir[k5],jr[k5],0] = rect_000+tri_000
                           
        dN_gain[kr[k5],ir[k5],jr[k5],1] = dNi_loss[kr[k5],ir[k5],jr[k5]]-dN_gain[kr[k5],ir[k5],jr[k5],0]
        
    # Condition 6:
    # k bin: Left/Right clip: Rectangle on bottom, triangle on top
    #                          Rectangle = ((xi1,xj1),(xi1,y_right_edge),(xi2,y_right_edge),(xi2,xj1))
    #                          Triangle  = ((xi1,y_right_edge),(xi1,y_left_edge),(xi2,y_right_edge))
       
    if len(k6)>0:
        xr1 = x11[k6]
        yr1 = x12[k6]
        xr2 = x21[k6]
        yr2 = y_right_edge[k6]
        
        xt1 = x11[k6]
        yt1 = y_right_edge[k6]
        xt2 = x11[k6]
        yt2 = y_left_edge[k6]
        xt3 = x21[k6]
        yt3 = y_right_edge[k6]
           
        rect_000, rect_001 = rect_gain_integrals(C[:,:,k6],x1=xr1,x2=xr2,y1=yr1,y2=yr2)
        tri_000, tri_001   =  tri_gain_integrals(C[:,:,k6], xt1, yt1, xt2, yt2, xt3, yt3, w, L)
         
        dM_gain[kr[k6],ir[k6],jr[k6],0] = rect_001 + tri_001
        dM_gain[kr[k6],ir[k6],jr[k6],1] = dM_loss[kr[k6],ir[k6],jr[k6]]-dM_gain[kr[k6],ir[k6],jr[k6],0]
            
        dN_gain[kr[k6],ir[k6],jr[k6],0] = rect_000 + tri_000
        dN_gain[kr[k6],ir[k6],jr[k6],1] = dNi_loss[kr[k6],ir[k6],jr[k6]]-dN_gain[kr[k6],ir[k6],jr[k6],0]
        
    # Condition 7:
    # k+1 bin: Triangle in top right corner
    #                            Triangle = ((x_top_edge,xj2),(xi2,xj2),(xi2,y_right_edge))
    
    if len(k7)>0: 
        xt1 = x_top_edge[k7]
        yt1 = x22[k7]
        xt2 = x21[k7]
        yt2 = x22[k7]
        xt3 = x21[k7]
        yt3 = y_right_edge[k7]
         
        dN_gain_temp,dM_gain_temp = tri_gain_integrals(C[:,:,k7], xt1, yt1, xt2, yt2, xt3, yt3, w, L)
        
        
        dN_gain[kr[k7],ir[k7],jr[k7],1] = dN_gain_temp.copy()
        dM_gain[kr[k7],ir[k7],jr[k7],1] = dM_gain_temp.copy()
        
        dM_gain[kr[k7],ir[k7],jr[k7],0] = dM_loss[kr[k7],ir[k7],jr[k7]]-dM_gain[kr[k7],ir[k7],jr[k7],1]
        
        dN_gain[kr[k7],ir[k7],jr[k7],0] = dNi_loss[kr[k7],ir[k7],jr[k7]]-dN_gain[kr[k7],ir[k7],jr[k7],1]
       
    # Condition 8:
    #  k bin: Triangle in lower left corner
    #                                Triangle = ((xi1,xj1),(xi1,y_left_edge),(x_bottom_edge,xj1))
    if len(k8)>0:
        xt1 = x11[k8]
        yt1 = x12[k8]
        xt2 = x11[k8]
        yt2 = y_left_edge[k8]
        xt3 = x_bottom_edge[k8]
        yt3 = x12[k8]
        
        dN_gain_temp,dM_gain_temp = tri_gain_integrals(C[:,:,k8], xt1, yt1, xt2, yt2, xt3, yt3, w, L)

        dN_gain[kr[k8],ir[k8],jr[k8],0] = dN_gain_temp.copy() 
        dM_gain[kr[k8],ir[k8],jr[k8],0] = dM_gain_temp.copy()
        
        dM_gain[kr[k8],ir[k8],jr[k8],1] = dM_loss[kr[k8],ir[k8],jr[k8]]-dM_gain[kr[k8],ir[k8],jr[k8],0]
        dN_gain[kr[k8],ir[k8],jr[k8],1] = dNi_loss[kr[k8],ir[k8],jr[k8]]-dN_gain[kr[k8],ir[k8],jr[k8],0]
        
    # Condition 9: Rectangle collection within k bin. All Mass/Number goes into kbin

    xr1 = x11[k9]
    xr2 = x21[k9] 
    yr1 = x12[k9]
    yr2 = x22[k9]

    dM_gain[kr[k9],ir[k9],jr[k9],0]  = dM_loss[kr[k9],ir[k9],jr[k9]].copy()
    dN_gain[kr[k9],ir[k9],jr[k9],0]  = dNi_loss[kr[k9],ir[k9],jr[k9]].copy()

    # Condition 10: Rectangle collection within k+1 bin. All Mass/Number goes into k+1bin
    
    xr1 = x11[k10]
    xr2 = x21[k10] 
    yr1 = x12[k10]
    yr2 = x22[k10]

    dM_gain[kr[k10],ir[k10],jr[k10],1]  = dM_loss[kr[k10],ir[k10],jr[k10]].copy()
    dN_gain[kr[k10],ir[k10],jr[k10],1]  = dNi_loss[kr[k10],ir[k10],jr[k10]].copy()
   
    
    # Condition 11 (2b): Triangle collection within k+1 bin. Occurs when xk+1 clips BR corner and intersects top edge.
    # Triangle = ((xi2,xj1),(x_top_edge,xj2),(xi2,xj2))
    
    if len(k2b)>0:
        xt1 = x_top_edge[k2b]
        yt1 = x22[k2b]
        xt2 = x21[k2b]
        yt2 = x22[k2b]
        xt3 = x21[k2b]
        yt3 = x12[k2b]
        
        dN_gain_temp,dM_gain_temp = tri_gain_integrals(C[:,:,k2b], xt1, yt1, xt2, yt2, xt3, yt3, w, L)
        
        dN_gain[kr[k2b],ir[k2b],jr[k2b],1] = dN_gain_temp.copy() 
        dM_gain[kr[k2b],ir[k2b],jr[k2b],1] = dM_gain_temp.copy()
        
        dM_gain[kr[k2b],ir[k2b],jr[k2b],0] = dM_loss[kr[k2b],ir[k2b],jr[k2b]]-dM_gain[kr[k2b],ir[k2b],jr[k2b],1]
        dN_gain[kr[k2b],ir[k2b],jr[k2b],0] = dNi_loss[kr[k2b],ir[k2b],jr[k2b]]-dN_gain[kr[k2b],ir[k2b],jr[k2b],1]
        
    # Condition 12 (3b): Triangle collection within k+1 bin. Occurs when xk+1 clips UL corner and intersects with right edge.
    # Triangle = ((xi1,xj2),(xi2,xj2),(xi2,y_right_edge)
    if len(k3b)>0:
        xt1 = x11[k3b]
        yt1 = x22[k3b]
        xt2 = x21[k3b]
        yt2 = x22[k3b]
        xt3 = x21[k3b]
        yt3 = y_right_edge[k3b]
        
        dN_gain_temp,dM_gain_temp = tri_gain_integrals(C[:,:,k3b], xt1, yt1, xt2, yt2, xt3, yt3, w, L)
        
        dN_gain[kr[k3b],ir[k3b],jr[k3b],1] = dN_gain_temp.copy() 
        dM_gain[kr[k3b],ir[k3b],jr[k3b],1] = dM_gain_temp.copy()
        
        dM_gain[kr[k3b],ir[k3b],jr[k3b],0] = dM_loss[kr[k3b],ir[k3b],jr[k3b]]-dM_gain[kr[k3b],ir[k3b],jr[k3b],1]
        dN_gain[kr[k3b],ir[k3b],jr[k3b],0] = dNi_loss[kr[k3b],ir[k3b],jr[k3b]]-dN_gain[kr[k3b],ir[k3b],jr[k3b],1]   
    
    return dMi_loss, dMj_loss, dM_gain, dNi_loss, dN_gain


def transfer_1mom_vec_3D(static_chunk, M_loss_buffer, M_gain_buffer, Mb_gain_buffer, ck12_chunk, dMi_loss, dMj_loss, dM_loss, dM_gain, dMb_frac, bins, Hlen, dnum,breakup=False):
    """
    Worker function: Processes a contiguous 'chunk' of bin-pair interactions.
    Bypasses object overhead by using flattened 1D array math. 
    """
    # 1. Extract indices from the pre-indexed structured array 
    kr = static_chunk['kr']
    kmin = static_chunk['kmin']
    kmid = static_chunk['kmid']
    idx_d1 = static_chunk['d1_flat']
    idx_d2 = static_chunk['d2_flat']
    
    #idx_kmin = static_chunk['gain_kmin_4d']
    #idx_kmid = static_chunk['gain_kmid_4d']
    
    #idx_d1 = kr * bins + ir
    #idx_d2 = kr * bins + jr
    
    idx_kmin = kr * bins + kmin 
    idx_kmid = kr * bins + kmid
    
    # 2. Map 3D (dnum, height, bin) coordinates to a 1D flat index 
    full_grid_size = dnum * Hlen * bins
    grid_size = Hlen*bins
    
    # 3. Accumulate Losses and Coalescence Gains using bincount 
    # This is significantly faster than np.add.at for floating point weights. 
    M_loss_flat = np.bincount(idx_d1, weights=ck12_chunk * dMi_loss, minlength=full_grid_size)
    M_loss_flat += np.bincount(idx_d2, weights=ck12_chunk * dMj_loss, minlength=full_grid_size)
       
    M_loss_buffer.ravel()[:] = M_loss_flat
    
    # M1_loss_flat = np.bincount(idx_d1, weights=ck12_chunk * dMi_loss, minlength=grid_size)
    # M2_loss_flat = np.bincount(idx_d2, weights=ck12_chunk * dMj_loss, minlength=grid_size)
    
    M_gain_flat  = np.bincount(idx_kmin, weights=ck12_chunk * dM_gain[:, 0], minlength=grid_size)
    M_gain_flat += np.bincount(idx_kmid, weights=ck12_chunk * dM_gain[:, 1], minlength=grid_size)
    
    M_gain_buffer.ravel()[:] = M_gain_flat
    
    # 4. Handle Breakup Gains 
    Mb_gain_flat = np.zeros(grid_size)
    
   # if dMb_frac is not None:
    if breakup:
        # Calculate total mass lost per interaction pair 
       # Mij_loss = ck12_chunk * dM_loss
   
        # Multiply fragmentation fractions by the total mass lost 
        # weighted_frac shape: (bins, chunk_n_pairs)
        weighted_frac = dMb_frac *  ck12_chunk * dM_loss
        
        
        # Distribute fragments into height bins (kr) for each particle size bin 
        for b in range(bins):
            
            Mb_gain_flat[b::bins] += np.bincount(kr, weights=weighted_frac[b, :], minlength=Hlen)
            
            # Sum mass at each height level for this specific bin 
            #h_sum = np.bincount(kr, weights=weighted_frac[b, :], minlength=Hlen)
            # Map the height sums into the correct 1D flat grid positions 
            #Mb_gain_flat[b::bins] = h_sum 
        Mb_gain_buffer.ravel()[:] = Mb_gain_flat
            
    #M_loss_temp = M_loss_flat.reshape(dnum,Hlen, bins)
                
    #M1_loss_temp = M1_loss_flat.reshape(Hlen, bins)           
    #M2_loss_temp = M2_loss_flat.reshape(Hlen, bins)
    
    #M_gain_temp  = M_gain_flat.reshape(Hlen, bins)
    #Mb_gain_temp = Mb_gain_flat.reshape(Hlen, bins)

    # Return a single stack of flattened results to minimize IPC overhead [cite: 1, 7]
    return M_loss_buffer, M_gain_buffer, Mb_gain_buffer
    #return M_loss_temp, M_gain_temp, Mb_gain_temp
#    return M1_loss_temp, M2_loss_temp, M_gain_temp, Mb_gain_temp

def transfer_1mom_bins_optimized(static_chunk, ck12_chunk, dMi_loss, dMj_loss, dM_loss, dM_gain, dMb_frac, bins, Hlen):
    """
    Worker function: Processes a contiguous 'chunk' of bin-pair interactions.
    Bypasses object overhead by using flattened 1D array math. 
    """
    # 1. Extract indices from the pre-indexed structured array 
    kr = static_chunk['kr']
    ir = static_chunk['ir']
    jr = static_chunk['jr']
    kmin = static_chunk['kmin']
    kmid = static_chunk['kmid']
    
    # 2. Map 2D (height, bin) coordinates to a 1D flat index 
    grid_size = Hlen * bins
    idx_i = kr * bins + ir
    idx_j = kr * bins + jr
    idx_kmin = kr * bins + kmin
    idx_kmid = kr * bins + kmid
    
    # 3. Accumulate Losses and Coalescence Gains using bincount 
    # This is significantly faster than np.add.at for floating point weights. 
    M1_loss_flat = np.bincount(idx_i, weights=ck12_chunk * dMi_loss, minlength=grid_size)
    M2_loss_flat = np.bincount(idx_j, weights=ck12_chunk * dMj_loss, minlength=grid_size)
    
    M_gain_flat = np.bincount(idx_kmin, weights=ck12_chunk * dM_gain[:, 0], minlength=grid_size)
    M_gain_flat += np.bincount(idx_kmid, weights=ck12_chunk * dM_gain[:, 1], minlength=grid_size)
    
    # 4. Handle Breakup Gains 
    Mb_gain_flat = np.zeros(grid_size)
    if dMb_frac is not None:

        weighted_frac = dMb_frac * ck12_chunk * dM_loss
        
        # Distribute fragments into height bins (kr) for each particle size bin 
        for b in range(bins):
            # Sum mass at each height level for this specific bin 
            h_sum = np.bincount(kr, weights=weighted_frac[b, :], minlength=Hlen)
            # Map the height sums into the correct 1D flat grid positions 
            Mb_gain_flat[b::bins] = h_sum 
            
    M1_loss_temp = M1_loss_flat.reshape(Hlen, bins)
    M2_loss_temp = M2_loss_flat.reshape(Hlen, bins)
    M_gain_temp  = M_gain_flat.reshape(Hlen, bins)
    Mb_gain_temp = Mb_gain_flat.reshape(Hlen, bins)

    # Return a single stack of flattened results to minimize IPC overhead [cite: 1, 7]
    return M1_loss_temp, M2_loss_temp, M_gain_temp, Mb_gain_temp

def transfer_1mom_bins_inplace(static_chunk, ck12_chunk, dMi_loss, dMj_loss, dM_loss, dM_gain, dMb_frac,
                               M1_loss_buffer,M2_loss_buffer,M_gain_buffer,Mb_gain_buffer,bins, Hlen,breakup=False):
    """
    Worker function: Processes a contiguous 'chunk' of bin-pair interactions.
    Bypasses object overhead by using flattened 1D array math. 
    """
    # 1. Extract indices from the pre-indexed structured array 
    kr = static_chunk['kr']
    ir = static_chunk['ir']
    jr = static_chunk['jr']
    kmin = static_chunk['kmin']
    kmid = static_chunk['kmid']
    
    # 2. Map 2D (height, bin) coordinates to a 1D flat index 
    grid_size = Hlen * bins
    idx_i = kr * bins + ir
    idx_j = kr * bins + jr
    idx_kmin = kr * bins + kmin
    idx_kmid = kr * bins + kmid
    
    # 3. Accumulate Losses and Coalescence Gains using bincount 
    # This is significantly faster than np.add.at for floating point weights. 
    M1_loss_buffer[:] += np.bincount(idx_i, weights=ck12_chunk * dMi_loss, minlength=grid_size)
    M2_loss_buffer[:] += np.bincount(idx_j, weights=ck12_chunk * dMj_loss, minlength=grid_size)
    
    M_gain_buffer[:] += np.bincount(idx_kmin, weights=ck12_chunk * dM_gain[:, 0], minlength=grid_size)
    M_gain_buffer[:] += np.bincount(idx_kmid, weights=ck12_chunk * dM_gain[:, 1], minlength=grid_size)
    
    # 4. Handle Breakup Gains 
    if breakup:

        weighted_frac = dMb_frac * ck12_chunk * dM_loss
        
        # Distribute fragments into height bins (kr) for each particle size bin 
        for b in range(bins):
            # Sum mass at each height level for this specific bin 
            Mb_gain_buffer[b::bins] += np.bincount(kr, weights=weighted_frac[b, :], minlength=Hlen)
            
    #M1_loss_temp = M1_loss_flat.reshape(Hlen, bins)
    #M2_loss_temp = M2_loss_flat.reshape(Hlen, bins)
    #M_gain_temp  = M_gain_flat.reshape(Hlen, bins)
    #Mb_gain_temp = Mb_gain_flat.reshape(Hlen, bins)

    # Return a single stack of flattened results to minimize IPC overhead [cite: 1, 7]
    #return M1_loss_buffer, M2_loss_buffer, M_gain_buffer, Mb_gain_buffer
    #return M1_loss_temp, M2_loss_temp, M_gain_temp, Mb_gain_temp

def transfer_1mom_bins_vec(static_chunk, n12, dMi_loss, dMj_loss, dM_loss, dM_gain, dMb_gain_frac, dnum, bins, Hlen, breakup=False):

    # 1. Extract indices from the pre-indexed structured array 
    kr = static_chunk['kr']
    ir = static_chunk['ir']
    jr = static_chunk['jr']
    d1 = static_chunk['d1']
    d2 = static_chunk['d2']
    kmin = static_chunk['kmin']
    kmid = static_chunk['kmid']
    
    # DO TRANSFER HERE
    # Initialize gain term arrays
    M_loss  = np.zeros((dnum,Hlen,bins))
    M_gain  = np.zeros((Hlen,bins))
    Mb_gain = np.zeros((Hlen,bins))
    
    np.add.at(M_loss,(d1,kr,ir),n12*dMi_loss)
    np.add.at(M_loss,(d2,kr,jr),n12*dMj_loss)
    
    np.add.at(M_gain,(kr,kmin),n12*dM_gain[:,0])
    np.add.at(M_gain,(kr,kmid),n12*dM_gain[:,1])
    
    # ELD NOTE: Breakup here can take losses from each pair and calculate gains
    # for breakup. Breakup gain arrays will be 3D.
    if breakup:    
        np.add.at(Mb_gain, kr, np.transpose(dMb_gain_frac[:,kmin]*n12*dM_loss))

    return M_loss, M_gain, Mb_gain 

def worker_tensor_vec(h_slice, p_d1, p_d2, cki, dMi, dMj, dMl, dMg, dMb_kernel, 
                        kmin_p, kmid_p, indc, indb, Eagg, Ebr, dnum):
    num_h = len(h_slice)
    num_bins = cki.shape[2]
    
    # 1. 4D Interaction Tensor: (Pairs, H_slice, bins_i, bins_j)
    ck12 = cki[p_d1][:, h_slice, :, None] * cki[p_d2][:, h_slice, None, :]

    # 2. Vectorized Loss (Contract bin axes)
    loss_i = np.einsum('phij,pij->phi', ck12, dMi)
    loss_j = np.einsum('phij,pij->phj', ck12, dMj)

    # 3. Vectorized Breakup Gain: (H_slice, bin_out)
    # Contracts Pair (p), bin_i (i), and bin_j (j) axes simultaneously
    Mij_loss = ck12 * dMl[:, None, :, :]
    #Mij_loss = ck12 * (dMi + dMj)[:, None, :, :]
    
    Mb_gain_dist = np.einsum('bij,phij->hb', dMb_kernel, Mij_loss)

    # 4. Local Accumulation Buffers
    l_loss = np.zeros((dnum, num_h, num_bins))
    l_gain = np.zeros((dnum, num_h, num_bins))
    
    # --- Scatter Loss Mapping ---
    # Explicit 3D indices to avoid broadcasting errors
    p_idx_3d = p_d1[:, None, None]
    h_idx_3d = np.arange(num_h)[None, :, None]
    b_idx_3d = np.arange(num_bins)[None, None, :]
    
    np.add.at(l_loss, (p_idx_3d, h_idx_3d, b_idx_3d), loss_i)
    np.add.at(l_loss, (p_d2[:, None, None], h_idx_3d, b_idx_3d), loss_j)
    
    # --- Scatter Coalescence Gain ---
    # gain_m_full: (Pairs, H_slice, bin_i, bin_j)
    gain_m_full = ck12 * dMg[:, None, :, :, 0]
    gain_d_full = ck12 * dMg[:, None, :, :, 1]
    
    # Explicit 4D indices for height and the pair-bin destination maps
    h_idx_4d = np.arange(num_h)[None, :, None, None]
    dest_bins_min = kmin_p[:, None, :, :] # (Pairs, 1, bins, bins)
    dest_bins_mid = kmid_p[:, None, :, :] # (Pairs, 1, bins, bins)

    np.add.at(l_gain[indc], (h_idx_4d, dest_bins_min), Eagg * gain_m_full)
    np.add.at(l_gain[indc], (h_idx_4d, dest_bins_mid), Eagg * gain_d_full)

    # --- Scatter Breakup Gain ---
    # Direct additive assignment for the redistributed mass
    l_gain[indb] += Ebr * Mb_gain_dist

    return l_loss, l_gain



#def vectorized_1mom(M_loss, M_gain, cki, params,
#                          p_d1, p_d2, dMi, dMj, dMg, 
#                          kmin_p, kmid_p, 
#                          indc, Eagg, Ebr, dnum, Hlen, bins):
    
def vectorized_1mom(cki, params, dMi, dMj, dMg, kmin, kmid, 
                    dMb_kernel, indc, indb, dnum, Hlen, bins):
    """
    Fused 1-Moment Kernel.
    Calculates Coalescence Loss and Gain in a single parallel pass.
    """
    
    M_loss = np.zeros((dnum, Hlen, bins))  
    M_gain = np.zeros((dnum, Hlen, bins))

    kernel_1mom(
        M_loss, M_gain, 
        params['active_Eagg'],
        params['active_Ebr'],
        params['active_rate'], 
        params['active_h'], 
        params['active_s1'], 
        params['active_s2'], 
        params['active_i'], 
        params['active_j'],
        params['active_p_idx'],
        dMi, dMj, kmin, kmid, dMg, # Pass FULL static arrays
        dMb_kernel, 
        indc, indb)

    return M_loss, M_gain


def vectorized_1mom_OLD(cki_slice, p_d1, p_d2, dMi, dMj, dMtot, dMg, dMb, 
                         kmin_p, kmid_p, indc, indb, Eagg, Ebr, dnum):
    """
    Pure NumPy Math Engine. 
    Processes a slice of heights for all distribution pairs.
    """
    num_h = cki_slice.shape[1]
    num_bins = cki_slice.shape[2]
    
    # 1. Tensor Product: (Pairs, H_slice, bins, bins)
    ck12 = cki_slice[p_d1][:, :, :, None] * cki_slice[p_d2][:, :, None, :]

    # 2. Loss Terms (einsum is faster/cache-friendly here)
    loss_i = np.einsum('phij,pij->phi', ck12, dMi)
    loss_j = np.einsum('phij,pij->phj', ck12, dMj)

    # 3. Gain Terms
    # Coalescence
    gain_m = Eagg * (ck12 * dMg[:, None, :, :, 0])
    gain_d = Eagg * (ck12 * dMg[:, None, :, :, 1])
    
    # Breakup (Using pre-summed dMtot)
    Mb_gain_dist = np.einsum('bij,phij->hb', dMb, ck12 * dMtot[:, None, :, :])

    # 4. Local Accumulation (To be mapped to SHM)
    l_loss = np.zeros((dnum, num_h, num_bins))
    l_gain = np.zeros((dnum, num_h, num_bins))
    
    # Scatter Loss
    h_idx_3 = np.arange(num_h)[None, :, None]
    b_idx_3 = np.arange(num_bins)[None, None, :]
    
    np.add.at(l_loss, (p_d1[:, None, None], h_idx_3, b_idx_3), loss_i)
    np.add.at(l_loss, (p_d2[:, None, None], h_idx_3, b_idx_3), loss_j)
    
    # Scatter Coalescence Gain
    h_idx_4 = np.arange(num_h)[None, :, None, None]
    np.add.at(l_gain[indc], (h_idx_4, kmin_p[:, None, :, :]), gain_m)
    np.add.at(l_gain[indc], (h_idx_4, kmid_p[:, None, :, :]), gain_d)

    # Add Breakup Gain
    l_gain[indb] += Ebr * Mb_gain_dist

    return l_loss, l_gain



def transfer_1mom_bins_NEW(Hlen, bins, ck12, dMi_loss, dMj_loss, dM_loss, dM_gain, 
                       kmin, kmid, dMb_gain_frac, breakup=False):

    # (hlen x bins x bins)
    M1_loss = np.sum(ck12*dMi_loss[None,:,:],axis=2)
    M2_loss = np.sum(ck12*dMj_loss[None,:,:],axis=1)
    
    # ChatGPT is the GOAT for telling me about np.add.at!
    M_gain = np.zeros((Hlen,bins))
    np.add.at(M_gain,  (np.arange(Hlen)[:,None,None],kmin),  ck12*(dM_gain[:,:,0][None,:,:]))
    np.add.at(M_gain,  (np.arange(Hlen)[:,None,None],kmid),  ck12*(dM_gain[:,:,1][None,:,:]))
    
    # ELD NOTE: Breakup here can take losses from each pair and calculate gains
    # for breakup. Breakup gain arrays will be 3D.
    if breakup:
        
        # (Hlen,bins,bins)
        Mij_loss = ck12*(dM_loss[None,:,:])

        Mb_gain = np.sum((dMb_gain_frac[:,kmin][None,:,:,:])*Mij_loss[:,None,:,:],axis=(2,3))
    else:
        Mb_gain = np.zeros((Hlen,bins))
    
    return M1_loss, M2_loss, M_gain, Mb_gain

def transfer_1mom_bins(Hlen, bins, kr, ir, jr, n12, dMi_loss, dMj_loss, dM_loss, dM_gain, kmin, kmid, dMb_gain_frac, breakup):

    # DO TRANSFER HERE
    # Initialize gain term arrays
    M1_loss = np.zeros((Hlen,bins))
    M2_loss = np.zeros((Hlen,bins))
    M_gain  = np.zeros((Hlen,bins))
    Mb_gain = np.zeros((Hlen,bins))
    
    np.add.at(M1_loss,(kr,ir),n12*dMi_loss)
    np.add.at(M2_loss,(kr,jr),n12*dMj_loss)
    
    np.add.at(M_gain,(kr,kmin),n12*dM_gain[:,0])
    np.add.at(M_gain,(kr,kmid),n12*dM_gain[:,1])
    
    # ELD NOTE: Breakup here can take losses from each pair and calculate gains
    # for breakup. Breakup gain arrays will be 3D.
    if breakup:
       
        np.add.at(Mb_gain,  kr, np.transpose(dMb_gain_frac[:,kmin]*n12*dM_loss))

    return M1_loss, M2_loss, M_gain, Mb_gain 

def transfer_1mom_bins_Google(Hlen, bins, kr, ir, jr, n12, dMi_loss, dMj_loss, dM_gain, kmin, kmid, dMb_gain_frac, breakup):
    """
    Calculates mass transfer rates using optimized bincount accumulation.
    Indices (kr, ir, jr, kmin, kmid) must be integers.
    Weights (n12, dMi_loss, etc.) are floats.
    """
    
    # Pre-calculate the flat grid size for the 1D accumulator
    grid_size = Hlen * bins
    
    # 1. Flatten 2D indices (kr, bin) into a 1D index space
    # This allows bincount to sum across the entire Hlen x bins grid in one C-call
    idx_i = kr * bins + ir
    idx_j = kr * bins + jr
    idx_kmin = kr * bins + kmin
    idx_kmid = kr * bins + kmid

    # 2. Accumulate Losses (M1 and M2)
    # weights=n12*dMi_loss handles the floating point mass transfer rates
    M1_loss = np.bincount(idx_i, weights=n12 * dMi_loss, minlength=grid_size).reshape(Hlen, bins)
    M2_loss = np.bincount(idx_j, weights=n12 * dMj_loss, minlength=grid_size).reshape(Hlen, bins)

    # 3. Accumulate Coalescence Gains
    # We sum the primary and secondary gain bins (kmin/kmid) separately
    M_gain = np.bincount(idx_kmin, weights=n12 * dM_gain[:, 0], minlength=grid_size).reshape(Hlen, bins)
    M_gain += np.bincount(idx_kmid, weights=n12 * dM_gain[:, 1], minlength=grid_size).reshape(Hlen, bins)

    # 4. Handle Breakup Gains
    Mb_gain = np.zeros((Hlen, bins))
    if breakup:
        # Mij_loss represents total mass lost from the bin-pair to be redistributed
        Mij_loss = n12 * (dMi_loss + dMj_loss)
        
        # dMb_gain_frac is (bins, n_pairs). We scale it by the total mass lost.
        # This is a large broadcasting operation, but NumPy handles it efficiently in RAM.
        weighted_frac = dMb_gain_frac * Mij_loss  # Result shape: (bins, n_pairs)
        
        # We must sum the fragments into the correct height levels (kr).
        # We loop over bins because 'bins' is usually small (e.g. 30-100), 
        # while 'n_pairs' can be huge. bincount stays in optimized C.
        for b in range(bins):
            Mb_gain[:, b] = np.bincount(kr, weights=weighted_frac[b, :], minlength=Hlen)

    return M1_loss, M2_loss, M_gain, Mb_gain 

def _unpack_f(f):
    return f[0,:], f[1,:], f[2,:], f[3,:]




def LGN_int(n, mu, sig2, x1, x2):
    """
    Ultimate High-Precision Log-Normal integral.
    Handles both Left Tail (z < -26) and Right Tail (z > 26) using erfcx scaling.
    Calculates Integral(x^n * PDF(x) dx) from x1 to x2.
    """
    # 1. Moment parameters
    shift = n * (sig2)
    mu_prime = mu + shift
    total_moment = np.exp(n * mu + 0.5 * (n**2) * (sig2))
    
    # 2. Z-scores
    sqrt2_sig = np.sqrt(2*sig2)
    # Ensure x > 0 to avoid log(0)
    safe_x1 = np.maximum(x1, 1e-100)
    safe_x2 = np.maximum(x2, 1e-100)
    
    z1 = (np.log(safe_x1) - mu_prime) / sqrt2_sig
    z2 = (np.log(safe_x2) - mu_prime) / sqrt2_sig
    
    # 3. Calculate Interval P = 0.5 * (erfc(z1) - erfc(z2))
    prob_interval = np.zeros_like(z1)
    
    # --- REGION A: Right Tail (Both z > 0) ---
    # Use erfcx factorization: exp(-z1^2) * [erfcx(z1) - ...]
    mask_right = (z1 > 0)
    if np.any(mask_right):
        _calc_tail_interval(prob_interval, mask_right, z1, z2)

    # --- REGION B: Left Tail (Both z < 0) ---
    # Symmetry: erfc(z1) - erfc(z2) = erfc(-z2) - erfc(-z1)
    # Let u1 = -z2, u2 = -z1. Note u1 > 0, u2 > 0.
    mask_left = (z2 < 0)
    if np.any(mask_left):
        # Swap and negate to map to right tail logic
        # We pass (-z2) as the "smaller" (left) bound in positive space
        # and (-z1) as the "larger" (right) bound.
        _calc_tail_interval(prob_interval, mask_left, -z2, -z1)

    # --- REGION C: Center (Straddling 0) ---
    # z1 < 0 < z2. Standard erfc is perfectly stable here (approx 1.0 - 1.0).
    mask_center = (~mask_right) & (~mask_left)
    if np.any(mask_center):
        prob_interval[mask_center] = 0.5 * (scip.erfc(z1[mask_center]) - scip.erfc(z2[mask_center]))

    # 4. Combine
    result = total_moment * prob_interval
    return np.maximum(result, 0.0)

def _calc_tail_interval(out_array, mask, z_small, z_large):
    """
    Helper for Right-Tail logic. 
    Calculates 0.5 * (erfc(z_small) - erfc(z_large)) for positive z.
    """
    zs = z_small[mask]
    zl = z_large[mask]
    
    # We factor out exp(-z_small^2)
    diff_sq = zl**2 - zs**2
    
    # Term = 0.5 * exp(-zs^2) * [ erfcx(zs) - exp(-diff_sq)*erfcx(zl) ]
    term_bracket = scip.erfcx(zs) - np.exp(-diff_sq) * scip.erfcx(zl)
    
    # Combine safely. 
    # If zs is huge (>26), exp(-zs^2) underflows. 
    # But usually this is multiplied by total_moment later, which might be huge.
    # For now, we return the probability directly.
    out_array[mask] = 0.5 * np.exp(-zs**2) * term_bracket


def LGN_int_ORIGINAL(n,muf,sig2f,x1,x2):
    # 

    #I = 0.5*np.exp(0.5*n*(n-1)*sig2f)*\
   #     (scip.erf((np.log(x2-muf)-sig2f*(n-0.5))/(np.sqrt(2*sig2f)))-\
   #      scip.erf((np.log(x1-muf)-sig2f*(n-0.5))/(np.sqrt(2*sig2f))))

    I = 0.5*np.exp(n*muf+0.5*n**2*sig2f)*\
        (scip.erf((np.log(x2)-muf-n*sig2f)/(np.sqrt(2*sig2f)))-\
         scip.erf((np.log(x1)-muf-n*sig2f)/(np.sqrt(2*sig2f))))
            
            
    # I = np.exp(n*muf+0.5*n**2*sig2f)*\
    #     (scip.erf((np.log(x2)-muf-n*sig2f)/(np.sqrt(sig2f)))-\
    #      scip.erf((np.log(x1)-muf-n*sig2f)/(np.sqrt(sig2f))))


    return I 

def LGN_int_PB07(n,muf,sig2f,x1,x2):
    # moments of lognormal fragment mass distribution from x1 to x2
    # See Prat and Barros (2007)
    
    n += 1

    t1 = np.log(x1-muf)/np.sqrt(sig2f)
    t2 = np.log(x2-muf)/np.sqrt(sig2f)

    I = (np.sqrt(np.pi*sig2f)/2.)*np.exp(n*(muf+0.25*n*sig2f))*\
        (scip.erf(t2-0.5*n*np.sqrt(sig2f))-scip.erf(t1-0.5*n*np.sqrt(sig2f)))
         
    # NORMALIZE I so that integral of mass from x1[0] to x2[-1] evaluates to unity.
    # This is needed to preserve the total mass from each interaction.

    return I 

def GAU_int(n, mu, sig2, x1, x2):
    """
    Integrates x^n * Normal(mu, sigma^2) from x1 to x2.
    Safe and stable using error functions.
    """
    sqrt2 = np.sqrt(2.0)
    sigma = np.sqrt(sig2)
    sig_sqrt2 = np.sqrt(2.*sig2)
    
    # Standardize bounds
    z1 = (x1 - mu) / sig_sqrt2
    z2 = (x2 - mu) / sig_sqrt2
    
    # n=0: Number (Area under curve)
    if n == 0:
        return 0.5 * (scip.erf(z2) - scip.erf(z1))
    
    # n=3: Mass (Integral of x^3 * PDF)
    # Using analytic expansion of moments for Normal distribution
    if n == 3:
        # We need the indefinite integral of x^3 * exp(-(x-mu)^2 / 2sig^2)
        # It's cleaner to use a helper that evaluates the moment-generating function
        def normal_moment_3(x):
            z = (x - mu) / sigma
            # Integral part: -sigma * (x^2 + mu*x + mu^2 + 2*sig^2) * PDF(x) + mu*(mu^2 + 3sig^2)*CDF(x)
            pdf = (1.0 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * z**2)
            cdf = 0.5 * (1.0 + scip.erf(z / sqrt2))
            return -sig2 * (x**2 + mu*x + mu**2 + 2*sig2) * pdf + mu * (mu**2 + 3*sig2) * cdf
        
        return normal_moment_3(x2) - normal_moment_3(x1)
    
    return 0.0 # Extend if other moments are needed

def gam_int(n,mu,Dm,x1,x2):
    # integral of x^n *x^mu * exp(-c*x) from x1 to x2
    
    #I = c**(-n)*(scip.gammainc(n+mu+1.,c*x2)-scip.gammainc(n+mu+1.,c*x1))
    
    I = (mu+4)**(-n)*Dm**n*scip.gamma(n+mu+1)*(scip.gammainc(n+mu+1.,(mu+4)*x2/Dm)-scip.gammainc(n+mu+1.,(mu+4)*x1/Dm))
    
    
    return I    

def In_int(n,c,x1,x2):
    # integral of x^n * exp(-c*x) from x1 to x2
    
    if n==0:
        
        I = (1./c)*(np.exp(-c*x1)-np.exp(-c*x2))
        
    elif n==1:
        
        I = (1./c**2)*(np.exp(-c*x1)*(c*x1+1.)-np.exp(-c*x2)*(c*x2+1.))
        
    else:
        
        print('n needs to be either 0 or 1.')
        raise Exception()
    
    return I    
  

def moments(r,aki,cki,x1,x2):  # Units are g^n
    # Integrate to find arbitrary moments of subgrid distribution Mn = Int x^n *[n(x)=ak*x+ck]*dx
    return aki*Pn(r+1,x1,x2)+cki*Pn(r,x1,x2)

def Pn(n,x1,x2):
    # integral of x^n from x1 to x2
    
    n = n+1
    
    if n <= 1:
        P = (x2 - x1)
    else:
        P = (x2**n - x1**n)/n
    
    return P





# OLD FUNCTIONS 


# def parallel_2mom(params, w, L, dMb_kernel, dNb_kernel, 
#                   indc, indb, Eagg, Ebr, dnum, Hlen, bins, n_jobs=4):
    
#     # 4D shape (pnum x Hlen x bins x bins)
#     regions = params['regions']
    
#     reg_map = params['reg_map']

#     kmin_l = params['kmin']
#     kmid_l = params['kmid']
    
#     d1_l = params['d1_ind']
#     d2_l = params['d2_ind']
#     hind_l = params['hind']
#     bi_ind_l = params['bi_ind']
#     bj_ind_l = params['bj_ind']
    
#     len_loss = len(regions)
    
#     C = params['C']

#     reg2 = reg_map[2]
#     reg3 = reg_map[3]
#     reg4 = reg_map[4]
#     reg5 = reg_map[5]
#     reg6 = reg_map[6]
#     reg7 = reg_map[7]
#     reg8 = reg_map[8]
#     reg9 = reg_map[9]
#     reg10 = reg_map[10]
#     reg11 = reg_map[11]
#     reg12 = reg_map[12]
       
#     l2 = reg2.stop-reg2.start
#     l3 = reg3.stop-reg3.start
#     l5 = reg5.stop-reg5.start
#     l6 = reg6.stop-reg6.start
#     l7 = reg7.stop-reg7.start
#     l8 = reg8.stop-reg8.start
#     l11 = reg11.stop-reg11.start
#     l12 = reg12.stop-reg12.start
    
#     # Unpack boundaries for easier vertex calculation
#     x11, x21, x12, x22 = params['x11'],   params['x21'],   params['x12'],   params['x22']
#     x_top, x_bot, y_lef, y_rig = params['x_top'], params['x_bot'], params['y_lef'], params['y_rig']

#     Cr  = np.concatenate((C,C[:,:,reg5],C[:,:,reg6]),axis=-1)
#     xr1 = np.concatenate((x11,x11[reg5],x11[reg6])) 
#     xr2 = np.concatenate((x21,x_top[reg5],x21[reg6]))
#     yr1 = np.concatenate((x12,x12[reg5],x12[reg6]))
#     yr2 = np.concatenate((x22,x22[reg5],y_rig[reg6]))
    
#     # Do all triangular integrals in one go
#     Ct  = np.concatenate((C[:,:,reg2],C[:,:,reg3],C[:,:,reg5],C[:,:,reg6],C[:,:,reg7],C[:,:,reg8],C[:,:,reg11],C[:,:,reg12]),axis=-1)
#     xt1 = np.concatenate((x11[reg2],x11[reg3],x_top[reg5],x11[reg6],x_top[reg7],x11[reg8],x_top[reg11],x11[reg12]))
#     yt1 = np.concatenate((x12[reg2],x12[reg3],x12[reg5],y_rig[reg6],x22[reg7],x12[reg8],x22[reg11],x22[reg12]))
#     xt2 = np.concatenate((x11[reg2],x11[reg3],x_top[reg5],x11[reg6],x21[reg7],x11[reg8],x21[reg11],x21[reg12]))
#     yt2 = np.concatenate((y_lef[reg2],x22[reg3],x22[reg5],y_lef[reg6],x22[reg7],y_lef[reg8],x22[reg11],x22[reg12]))
#     xt3 = np.concatenate((x21[reg2],x_bot[reg3],x_bot[reg5],x21[reg6],x21[reg7],x_bot[reg8],x21[reg11],x21[reg12]))
#     yt3 = np.concatenate((x12[reg2], x12[reg3],x12[reg5],y_rig[reg6],y_rig[reg7],x12[reg8],x12[reg11],y_rig[reg12]))
    
#     C_rect_flat = np.ascontiguousarray(Cr.reshape(9,-1))
#     C_tri_flat = np.ascontiguousarray(Ct.reshape(9,-1))
    
#     n_rect_total = len(xr1)
#     n_tri_total = len(xt1)
    
#     # Split indices into n_jobs chunks
#     # Note: Rectangles and Triangles might have different lengths, 
#     # so we split them independently but execute in the same task if possible, 
#     # or just split the larger one.
    
#     # Simple approach: np.array_split handles uneven division automatically
#     chunks = []
    
#     # Create slices for Rectangles
#     r_slices = np.array_split(np.arange(n_rect_total), n_jobs)
#     t_slices = np.array_split(np.arange(n_tri_total), n_jobs)
    
#     for i in range(n_jobs):
#         idx_r = r_slices[i]
#         idx_t = t_slices[i]
        
#         chunk = {
#             # Slice Rectangles
#             'xr1': xr1[idx_r], 'xr2': xr2[idx_r],
#             'yr1': yr1[idx_r], 'yr2': yr2[idx_r],
#             'Cr':  C_rect_flat[:, idx_r], # Slice the (9, N) array on axis 1
            
#             # Slice Triangles
#             'xt1': xt1[idx_t], 'yt1': yt1[idx_t],
#             'xt2': xt2[idx_t], 'yt2': yt2[idx_t],
#             'xt3': xt3[idx_t], 'yt3': yt3[idx_t],
#             'Ct':  C_tri_flat[:, idx_t]
#         }
#         chunks.append(chunk)

#     # 3. Parallel Execution
#     # ---------------------------------------------------------
#     # Use 'threading' backend because Numba releases the GIL.
#     # This avoids pickling overhead entirely!
    
#     results = Parallel(n_jobs=n_jobs, backend='threading')(
#         delayed(solve_chunk_integrals)(chunk, w, L) for chunk in chunks
#     )
    
#     # 4. Reassembly
#     # ---------------------------------------------------------
#     dNi_rect = np.concatenate([r[0] for r in results])
#     dMi_rect = np.concatenate([r[1] for r in results])
#     dMj_rect = np.concatenate([r[2] for r in results])
    
#     dN_gain_tri = np.concatenate([r[3] for r in results])
#     dM_gain_tri = np.concatenate([r[4] for r in results])
    
#     dM_rect = dMi_rect+dMj_rect
    
#     loss_inds = np.arange(len_loss)

#     dN_loss  = dNi_rect[loss_inds]
#     dM_loss  = dM_rect[loss_inds]
#     dMi_loss = dMi_rect[loss_inds]
#     dMj_loss = dMj_rect[loss_inds]
    
#     # 5. Scatter & Breakup (Serial / Main Thread)
#     # ---------------------------------------------------------
#     # Proceed exactly as your serial function does from here on.
    

    
#     Ml2 = dM_loss[reg2]
#     Ml3 = dM_loss[reg3]
#     Ml4 = dM_loss[reg4]
#     Ml5 = dM_loss[reg5]
#     Ml6 = dM_loss[reg6]
#     Ml7 = dM_loss[reg7]
#     Ml8 = dM_loss[reg8]
#     Ml9 = dM_loss[reg9]
#     Ml10 = dM_loss[reg10]
#     Ml11 = dM_loss[reg11]
#     Ml12 = dM_loss[reg12]
    
#     Mr5 = dM_rect[len_loss:len_loss+l5]
#     Mr6 = dM_rect[len_loss+l5:]
    
#     Nr5 = dNi_rect[len_loss:len_loss+l5]
#     Nr6 = dNi_rect[len_loss+l5:]
    
#     Nl2 = dN_loss[reg2]
#     Nl3 = dN_loss[reg3]
#     Nl4 = dN_loss[reg4]
#     Nl5 = dN_loss[reg5]
#     Nl6 = dN_loss[reg6]
#     Nl7 = dN_loss[reg7]
#     Nl8 = dN_loss[reg8]
#     Nl9 = dN_loss[reg9]
#     Nl10 = dN_loss[reg10]
#     Nl11 = dN_loss[reg11]
#     Nl12 = dN_loss[reg12]
    
#     # Define cumulative offsets for the triangle stack
#     t_lens = [l2, l3, l5, l6, l7, l8, l11, l12]
#     t_offs = np.insert(np.cumsum(t_lens), 0, 0)
    
#     # Use these offsets to pull Mt segments
#     Mt2 = dM_gain_tri[t_offs[0]:t_offs[1]]
#     Mt3 = dM_gain_tri[t_offs[1]:t_offs[2]]
#     Mt5 = dM_gain_tri[t_offs[2]:t_offs[3]]
#     Mt6 = dM_gain_tri[t_offs[3]:t_offs[4]]
#     Mt7 = dM_gain_tri[t_offs[4]:t_offs[5]]
#     Mt8 = dM_gain_tri[t_offs[5]:t_offs[6]]
#     Mt11 = dM_gain_tri[t_offs[6]:t_offs[7]]
#     Mt12 = dM_gain_tri[t_offs[7]:]
    
#     # Use these offsets to pull Nt segments
#     Nt2 = dN_gain_tri[t_offs[0]:t_offs[1]]
#     Nt3 = dN_gain_tri[t_offs[1]:t_offs[2]]
#     Nt5 = dN_gain_tri[t_offs[2]:t_offs[3]]
#     Nt6 = dN_gain_tri[t_offs[3]:t_offs[4]]
#     Nt7 = dN_gain_tri[t_offs[4]:t_offs[5]]
#     Nt8 = dN_gain_tri[t_offs[5]:t_offs[6]]
#     Nt11 = dN_gain_tri[t_offs[6]:t_offs[7]]
#     Nt12 = dN_gain_tri[t_offs[7]:]
    
#     Mrt5 = Mr5+Mt5 
#     Mrt6 = Mr6+Mt6 
    
#     Nrt5 = Nr5+Nt5 
#     Nrt6 = Nr6+Nt6 
    
#     # Concatenate all regions in order and do residual for bin partitioning
#     dM_gain = np.concatenate((Mt2,Ml2-Mt2,
#                               Mt3,Ml3-Mt3,
#                               Ml4, 
#                               Mrt5,Ml5-Mrt5,
#                               Mrt6,Ml6-Mrt6, 
#                               Mt7,Ml7-Mt7, 
#                               Mt8,Ml8-Mt8, 
#                               Ml9, 
#                               Ml10,
#                               Mt11,Ml11-Mt11, 
#                               Mt12,Ml12-Mt12))
    
#     dN_gain = np.concatenate((Nt2,Nl2-Nt2,
#                               Nt3,Nl3-Nt3,
#                               Nl4, 
#                               Nrt5,Nl5-Nrt5,
#                               Nrt6,Nl6-Nrt6, 
#                               Nt7,Nl7-Nt7, 
#                               Nt8,Nl8-Nt8, 
#                               Nl9, 
#                               Nl10,
#                               Nt11,Nl11-Nt11, 
#                               Nt12,Nl12-Nt12))

#     k_gain = np.concatenate((kmin_l[reg2],kmid_l[reg2],    # Region 2: kmin tri
#                              kmin_l[reg3],kmid_l[reg3],    # Region 3: kmin tri
#                              kmin_l[reg4],                  # Region 4: kmin rect; self collection
#                              kmin_l[reg5],kmid_l[reg5],    # Region 5: kmin rect + tri
#                              kmin_l[reg6],kmid_l[reg6],    # Region 6: kmin rect + tri
#                              kmid_l[reg7],kmin_l[reg7],    # Region 7: kmid tri
#                              kmin_l[reg8],kmid_l[reg8],    # Region 8: kmin tri
#                              kmin_l[reg9],                  # Region 9:kmin rect; fully in bin
#                              kmid_l[reg10],                 # Region 10: kmid rect; fully in bin
#                              kmid_l[reg11],kmin_l[reg11],  # Region 11: kmid tri 
#                              kmid_l[reg12],kmin_l[reg12])) # Region 12: kmid tri

#     h_gain = np.concatenate((hind_l[reg2],hind_l[reg2],    # Region 2: kmin tri
#                              hind_l[reg3],hind_l[reg3],    # Region 3: kmin tri
#                              hind_l[reg4],                     # Region 4: kmin rect; self collection
#                              hind_l[reg5],hind_l[reg5],    # Region 5: kmin rect + tri
#                              hind_l[reg6],hind_l[reg6],    # Region 6: kmin rect + tri
#                              hind_l[reg7],hind_l[reg7],    # Region 7: kmid tri
#                              hind_l[reg8],hind_l[reg8],    # Region 8: kmin tri
#                              hind_l[reg9],                     # Region 9:kmin rect; fully in bin
#                              hind_l[reg10],                    # Region 10: kmid rect; fully in bin
#                              hind_l[reg11],hind_l[reg11],  # Region 11: kmid tri 
#                              hind_l[reg12],hind_l[reg12])) # Region 12: kmid tri

#     M_loss = np.zeros((dnum, Hlen, bins))
#     N_loss = np.zeros((dnum, Hlen, bins))
    
#     M_gain = np.zeros((dnum, Hlen, bins))
#     N_gain = np.zeros((dnum, Hlen, bins))
 
#     np.add.at(M_loss, (d1_l, hind_l, bi_ind_l), dMi_loss)
#     np.add.at(M_loss, (d2_l, hind_l, bj_ind_l), dMj_loss)
#     np.add.at(N_loss, (d1_l, hind_l, bi_ind_l), dN_loss)
#     np.add.at(N_loss, (d2_l, hind_l, bj_ind_l), dN_loss)

#     # Do gains all concatenated
#     np.add.at(M_gain[indc],(h_gain, k_gain), dM_gain)
#     np.add.at(N_gain[indc],(h_gain, k_gain), dN_gain)
    
    
#     # Collisional breakup
#     if Ebr > 0.:
        
#         k_limit = params['kmin'].astype(np.int32)
        
#         M_gain, N_gain  = breakup_kernel(
#             M_gain, N_gain, dM_loss.ravel(), 
#             dMb_kernel, dNb_kernel,
#             params['bi_ind'], params['bj_ind'], params['hind'], 
#             k_limit,  # Pass the limit array here
#             indb, Ebr)
        
        
#     # Final scaling
#     M_gain[indc] *= Eagg
#     N_gain[indc] *= Eagg
    
#     return M_loss, M_gain, N_loss, N_gain


# def balance_regions(region_inds, n_batches,cost_map={'no_calc':1,'tri':8,'rect/tri':10}):
#     """
#     Parameters
#     ----------
#     costs : array-like, shape (N,)
#         Cost per region
#     n_batches : int

#     Returns
#     -------
#     list[np.ndarray]
#         List of index arrays, one per batch
#     """
    
#     # Get costs for each region index
#     costs = np.zeros_like(region_inds)
#     costs[region_inds==2] = cost_map['tri']
#     costs[region_inds==3] = cost_map['tri'] 
#     costs[region_inds==4] = cost_map['no_calc'] 
#     costs[region_inds==5] = cost_map['rect/tri']
#     costs[region_inds==6] = cost_map['rect/tri']
#     costs[region_inds==7] = cost_map['tri'] 
#     costs[region_inds==8] = cost_map['tri'] 
#     costs[region_inds==9] = cost_map['no_calc'] 
#     costs[region_inds==10] = cost_map['no_calc'] 
#     costs[region_inds==11] = cost_map['tri'] 
#     costs[region_inds==12] = cost_map['tri'] 
    
#     #costs = np.asarray(costs)
#     idx = np.arange(costs.size)

#     order = np.argsort(costs)[::-1]  # descending cost
#     idx = idx[order]
#     costs = costs[order]

#     heap = [(0.0, i, []) for i in range(n_batches)]
#     heapify(heap)

#     for k, c in zip(idx, costs):
#         total, i, batch = heappop(heap)
#         batch.append(k)
#         heappush(heap, (total + c, i, batch))

#     #heap_tot = [hh[0] for hh in heap]
#     #print('heap_tot=',heap_tot)
#     #print('heap_len=',len(heap))
#     #raise Exception()

#     heap.sort(key=lambda x: x[1])
#     return [np.asarray(batch, dtype=int) for _, _, batch in heap]

#def sum_1mom_batches(worker_id,kr,ir,jr,ck12,mem_map_dict,dd,Hlen,bins,breakup=False):  
    
# def sum_1mom_batches(kr,ir,jr,ck12,
#                      kmin,kmid,dMi_loss,dMj_loss,dM_loss,dM_gain,dMb_gain_frac,
#                      dd,Hlen,bins,breakup=False):  
 
#     # Load memory map variables
#     #mvars = {vv: load(path, mmap_mode='r') for vv, path in mem_map_dict['inputs'].items()}
    
#     #outputs = np.memmap(mem_map_dict['outputs'],dtype='float64',mode='r+',shape=mem_map_dict['output_shape'])  

#     # kmin = mvars['kmin'][ir,jr]
#     # kmid = mvars['kmid'][ir,jr]
#     # dMi_loss = mvars['dMi_loss'][dd,ir,jr]
#     # dMj_loss = mvars['dMj_loss'][dd,ir,jr]
#     # dM_loss = mvars['dM_loss'][dd,ir,jr]
#     # dM_gain = mvars['dM_gain'][dd,ir,jr,:]
#     # dMb_gain_frac = mvars['dMb_gain_frac'][:,kmin]
    
#     M1_loss_temp, M2_loss_temp, M_gain_temp, Mb_gain_temp = transfer_1mom_bins(
#                       Hlen,bins,kr,ir,jr,
#                       ck12,
#                       dMi_loss,
#                       dMj_loss,
#                       dM_loss,
#                       dM_gain,
#                       kmin,
#                       kmid,
#                       dMb_gain_frac,breakup)
    
#     #print('M1_loss_temp=',M1_loss_temp)
    
    
#     return M1_loss_temp, M2_loss_temp, M_gain_temp, Mb_gain_temp
    
#     # Write arrays to memory map files
#     # outputs[worker_id,0,:,:] = M1_loss_temp
#     # outputs[worker_id,1,:,:] = M2_loss_temp
#     # outputs[worker_id,2,:,:] = M_gain_temp
#     # outputs[worker_id,3,:,:] = Mb_gain_temp

#     #return True
    

# def sum_2mom_batches(worker_id,inds,x_bottom_edge,x_top_edge,y_left_edge,y_right_edge, 
#                      kr,ir,jr,x11,x21,ak1,ck1,x12,x22,ak2,ck2,
#                      mem_map_dict,dd,Hlen,bins,breakup=False):  
 
#     # Load memory map variables
#     mvars = {vv: load(path, mmap_mode='r') for vv, path in mem_map_dict['inputs'].items()}
    
#     outputs = np.memmap(mem_map_dict['outputs'],dtype='float64',mode='r+',shape=mem_map_dict['output_shape'])  

#     # Get lazy loaded variables and slice accordingly
#     #xi1 = mvars['xi1']
#     #xi2 = mvars['xi2']
#     #xk_min = mvars['xk_min'][ir,jr]
#     kmin = mvars['kmin']
#     kmid = mvars['kmid']
#     dMb_gain_frac = mvars['dMb_gain_frac']
#     dNb_gain_frac = mvars['dNb_gain_frac']
#     PK = mvars['PK'][:,dd,ir,jr]
#     w = mvars['w']
#     L = mvars['L']
    
#     # Calculate transfer rates
#     dMi_loss, dMj_loss, dM_loss, dM_gain, dNi_loss, dN_gain = calculate_rates(Hlen,bins,inds,x_bottom_edge,x_top_edge,y_left_edge,y_right_edge,
#     kr,ir,jr,x11,x21,ak1,ck1,x12,x22,ak2,ck2,PK,kmin,kmid,dMb_gain_frac,dNb_gain_frac,w,L,breakup=breakup)

#     # Perform transfer/assignment to bins
#     M1_loss_temp, M2_loss_temp, M_gain_temp, Mb_gain_temp,\
#     N1_loss_temp, N2_loss_temp, N_gain_temp, Nb_gain_temp = transfer_bins(Hlen,bins,kr,ir,jr,kmin,kmid,dMi_loss,dMj_loss,dM_loss,dM_gain,
#                       dNi_loss,dN_gain,dMb_gain_frac,dNb_gain_frac,breakup=breakup)
     
#     # Write arrays to memory map files
#     outputs[worker_id,0,:,:] = M1_loss_temp
#     outputs[worker_id,1,:,:] = M2_loss_temp
#     outputs[worker_id,2,:,:] = M_gain_temp
#     outputs[worker_id,3,:,:] = Mb_gain_temp
#     outputs[worker_id,4,:,:] = N1_loss_temp
#     outputs[worker_id,5,:,:] = N2_loss_temp
#     outputs[worker_id,6,:,:] = N_gain_temp
#     outputs[worker_id,7,:,:] = Nb_gain_temp

#     return True



# def solve_chunk_integrals(chunk_data, w, L):
#     """
#     Worker function to solve a subset of interactions.
#     """
#     # Unpack the chunk dictionary
#     xr1, xr2 = chunk_data['xr1'], chunk_data['xr2']
#     yr1, yr2 = chunk_data['yr1'], chunk_data['yr2']
#     C_rect   = chunk_data['Cr']
    
#     xt1, yt1 = chunk_data['xt1'], chunk_data['yt1']
#     xt2, yt2 = chunk_data['xt2'], chunk_data['yt2']
#     xt3, yt3 = chunk_data['xt3'], chunk_data['yt3']
#     C_tri    = chunk_data['Ct']
    
#     # --- 1. Rectangular Integrals ---
#     n_rect = len(xr1)
#     dNi_r = np.zeros(n_rect, dtype=np.float64)
#     dMi_r = np.zeros(n_rect, dtype=np.float64)
#     dMj_r = np.zeros(n_rect, dtype=np.float64)
    
#     # Call Numba Kernel
#     loss_kernel_numba(dNi_r, dMi_r, dMj_r, C_rect, xr1, xr2, yr1, yr2)
    
#     # --- 2. Triangular Integrals ---
#     n_tri = len(xt1)
#     dN_t = np.zeros(n_tri, dtype=np.float64)
#     dM_t = np.zeros(n_tri, dtype=np.float64)
    
#     # Call Numba Kernel
#     tri_kernel_numba(dN_t, dM_t, C_tri, xt1, yt1, xt2, yt2, xt3, yt3, w, L)
    
#     return dNi_r, dMi_r, dMj_r, dN_t, dM_t




# def sum_1mom_batches_optimized(static_chunk, ck12_chunk, dMi_loss, dMj_loss, dM_gain, dMb_frac, bins, Hlen):
#     """
#     Worker function: Processes a contiguous 'chunk' of bin-pair interactions.
#     Bypasses object overhead by using flattened 1D array math. 
#     """
#     # 1. Extract indices from the pre-indexed structured array 
#     kr = static_chunk['kr']
#     ir = static_chunk['ir']
#     jr = static_chunk['jr']
#     kmin = static_chunk['kmin']
#     kmid = static_chunk['kmid']
    
#     # 2. Map 2D (height, bin) coordinates to a 1D flat index 
#     grid_size = Hlen * bins
#     idx_i = kr * bins + ir
#     idx_j = kr * bins + jr
#     idx_kmin = kr * bins + kmin
#     idx_kmid = kr * bins + kmid
    
#     # 3. Accumulate Losses and Coalescence Gains using bincount 
#     # This is significantly faster than np.add.at for floating point weights. 
#     m1_loss_flat = np.bincount(idx_i, weights=ck12_chunk * dMi_loss, minlength=grid_size)
#     m2_loss_flat = np.bincount(idx_j, weights=ck12_chunk * dMj_loss, minlength=grid_size)
    
#     mg_flat = np.bincount(idx_kmin, weights=ck12_chunk * dM_gain[:, 0], minlength=grid_size)
#     mg_flat += np.bincount(idx_kmid, weights=ck12_chunk * dM_gain[:, 1], minlength=grid_size)
    
#     # 4. Handle Breakup Gains 
#     mb_flat = np.zeros(grid_size)
#     if dMb_frac is not None:
#         # Calculate total mass lost per interaction pair 
#         mij_loss = ck12_chunk * (dMi_loss + dMj_loss)
        
#         # Multiply fragmentation fractions by the total mass lost 
#         # weighted_frac shape: (bins, chunk_n_pairs)
#         weighted_frac = dMb_frac * mij_loss
        
#         # Distribute fragments into height bins (kr) for each particle size bin 
#         for b in range(bins):
#             # Sum mass at each height level for this specific bin 
#             h_sum = np.bincount(kr, weights=weighted_frac[b, :], minlength=Hlen)
#             # Map the height sums into the correct 1D flat grid positions 
#             mb_flat[b::bins] = h_sum 

#     # Return a single stack of flattened results to minimize IPC overhead [cite: 1, 7]
#     return np.vstack([m1_loss_flat, m2_loss_flat, mg_flat, mb_flat])


# # Attach to Shared Memory Blocks
# def get_shm(name, shape, dtype=np.float64):
#     s = shared_memory.SharedMemory(name=name)
#     return s, np.ndarray(shape, dtype=dtype, buffer=s.buf)

# def parallel_1mom(h_slice, cki_slice, p_d1, p_d2, indc, indb, Eagg, Ebr, dnum, shapes):
    
#     # GUARD: Ensure 3D shape (Distributions, Heights, Bins)
#     if cki_slice.ndim == 2:
#         # This happens if a single height index was passed without being in a list
#         cki_slice = cki_slice[:, np.newaxis, :]

#     # Attach all segments
#     s1, dMi = get_shm('shm_dMi_loss', shapes['dMi_loss'])
#     s2, dMj = get_shm('shm_dMj_loss', shapes['dMj_loss'])
#     s3, dMl = get_shm('shm_dM_loss', shapes['dM_loss'])
#     s4, dMg = get_shm('shm_dM_gain', shapes['dM_gain'])
#     s5, dMb = get_shm('shm_dMb_gain_kernel', shapes['dMb_gain_kernel'])
#     s6, kmin = get_shm('shm_kmin', shapes['k'], np.int64)
#     s7, kmid = get_shm('shm_kmid', shapes['k'], np.int64)
#     s8, full_out = get_shm('shm_out', shapes['out'])
    
#     # Run the Math Engine
#     l_loss_res, l_gain_res = vectorized_1mom(
#         cki_slice, p_d1, p_d2, dMi, dMj, dMl, dMg, dMb, 
#         kmin, kmid, indc, indb, Eagg, Ebr, dnum)
    
#     full_out[0][:,h_slice,:] = l_loss_res 
#     full_out[1][:,h_slice,:] = l_gain_res

#     # Cleanup handles
#     for s in [s1, s2, s3, s4, s5, s6, s7, s8]:
#         s.close()
