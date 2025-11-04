# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 11:31:38 2025

@author: edwin.dunnavan
"""
## Import stuff
import numpy as np

from collection_kernels import Prod_kernel, Constant_kernel, Golovin_kernel, hydro_kernel
from bin_integrals import In_int, gam_int, LGN_int, integrate_fast_kernel

from joblib import Parallel, delayed, dump, load

from tempfile import gettempdir

import os

def setup_regions(x11,x21,ak1,ck1,x12,x22,ak2,ck2,xi1,xi2,kmin,cond_1,sc_inds):

    Hlen, bins = np.shape(x11)

    # (heights x bins x bins)
    x_bottom_edge = (xi2[kmin][None,:,:]-x12[:,None,:])
    x_top_edge = (xi2[kmin][None,:,:]-x22[:,None,:])
    y_left_edge = (xi2[kmin][None,:,:]-x11[:,:,None])
    y_right_edge = (xi2[kmin][None,:,:]-x21[:,:,None])
    
    check_bottom = (x11[:,:,None]<x_bottom_edge) &\
                   (x21[:,:,None]>x_bottom_edge)
     
    check_top = (x11[:,:,None]<x_top_edge) &\
                (x21[:,:,None]>x_top_edge)
                
    check_left = (x12[:,None,:]<y_left_edge) &\
                 (x22[:,None,:]>y_left_edge)
                
    check_right = (x12[:,None,:]<y_right_edge) &\
                  (x22[:,None,:]>y_right_edge)            
           
    check_middle = ((0.5*(x11[:,:,None]+x21[:,:,None]))+(0.5*(x12[:,None,:]+x22[:,None,:])))<(xi2[kmin][None,:,:])
               
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
         
    cond_touch = (check_bottom|check_top|check_left|check_right)
    
    # NEW
    cond_BR_corner = ((x21==xi2)[:,:,None])&((x12==xi1)[:,None,:]) # Bottom right corner
    cond_UL_corner = ((x11==xi1)[:,:,None])&((x22==xi2)[:,None,:]) # Upper left corner
    
    cond_2_corner = (cond_BR_corner)&(check_left)
    cond_3_corner = (cond_UL_corner)&(check_bottom)
    
    cond_2b_corner = (cond_BR_corner)&(check_top)
    cond_3b_corner = (cond_UL_corner)&(check_right)
    
    cond_2 = np.eye(bins,k=1,dtype=bool)[None,:,:] & (cond_2_corner) & (~cond_1) & (cond_touch)
    cond_3 = np.eye(bins,k=-1,dtype=bool)[None,:,:] & (cond_3_corner) & (~cond_1) & (cond_touch)
    
    cond_2b = cond_2b_corner & (~cond_1)
    cond_3b = cond_3b_corner & (~cond_1)
    
    cond_4 = np.eye(bins,dtype=bool)[None,:,:] & ((~cond_1))
    cond_nt = (~(cond_1|cond_2|cond_3|cond_4))
    cond_5 = (check_top&check_bottom)  & cond_nt
    cond_6 = (check_left&check_right)  & cond_nt
    cond_7 =  (check_right&check_top)  & cond_nt
    cond_8 = (check_left&check_bottom) & cond_nt
    cond_rect = (~cond_touch)&(~cond_1)&(~cond_4)&(~cond_5)&(~cond_6)&(~cond_7)&(~cond_8)
    cond_9 = (cond_rect&check_middle)
    cond_10 = (cond_rect&(~check_middle))
    
    k1, i1, j1  = np.nonzero((~cond_1)&sc_inds) # Only do loss/gain terms for 0>bins-sbin bins
    k2, i2, j2  = np.nonzero(cond_2&sc_inds)
    k3, i3, j3  = np.nonzero(cond_3&sc_inds)
    k2b, i2b, j2b  = np.nonzero(cond_2b&sc_inds)
    k3b, i3b, j3b  = np.nonzero(cond_3b&sc_inds)
    k4, i4, j4  = np.nonzero(cond_4&sc_inds)
    k5, i5, j5  = np.nonzero(cond_5&sc_inds)
    k6, i6, j6  = np.nonzero(cond_6&sc_inds)
    k7, i7, j7  = np.nonzero(cond_7&sc_inds)
    k8, i8, j8  = np.nonzero(cond_8&sc_inds)
    k9, i9, j9  = np.nonzero(cond_9&sc_inds)
    k10,i10,j10 = np.nonzero(cond_10&sc_inds)
    
 
    # Returns dictionary of source integration region type indices
    return    {'x_bottom_edge':x_bottom_edge,
               'x_top_edge':x_top_edge,
               'y_left_edge':y_left_edge,
               'y_right_edge':y_right_edge,
               '1' :{'k':k1,'i':i1,'j':j1},
               '2' :{'k':k2,'i':i2,'j':j2},
               '2b':{'k':k2b,'i':i2b,'j':j2b},
               '3' :{'k':k3,'i':i3,'j':j3},
               '3b':{'k':k3b,'i':i3b,'j':j3b},
               '4' :{'k':k4,'i':i4,'j':j4},
               '5' :{'k':k5,'i':i5,'j':j5},
               '6' :{'k':k6,'i':i6,'j':j6},
               '7' :{'k':k7,'i':i7,'j':j7},
               '8' :{'k':k8,'i':i8,'j':j8},
               '9' :{'k':k9,'i':i9,'j':j9},
               '10':{'k':k10,'i':i10,'j':j10}}




def calculate_regions(x11,x21,ak1,ck1,x12,x22,ak2,ck2,PK,xi1,xi2,regions):

    Hlen, bins = np.shape(x11)

    
    '''
    Vectorized Integration Regions:
    cond_1 :  Ignore CC process for these source bins; they don't map to the largest avail bin.
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
    
    #x_bottom_edge,x_top_edge,y_left_edge,y_right_edge,\
    #i1,j1,k1,i2,j2,k2,i2b,j2b,k2b,i3,j3,k3,i3b,j3b,k3b,\
    #i4,j4,k4,i5,j5,k5,i6,j6,k6,i7,j7,k7,i8,j8,k8,\
    #i9,j9,k9,i10,j10,k10 = setup_regions(x11,x21,ak1,ck1,x12,x22,ak2,ck2,
    #                                     xi1,xi2,kmin,cond_1,sc_inds)
    
    
    x_bottom_edge = regions['x_bottom_edge']
    x_top_edge = regions['x_top_edge']
    y_left_edge = regions['y_left_edge']
    y_right_edge = regions['y_right_edge']
    i1 = regions['1']['i']
    j1 = regions['1']['j']
    k1 = regions['1']['k']
    i2 = regions['2']['i']
    j2 = regions['2']['j']
    k2 = regions['2']['k']
    i2b = regions['2b']['i']
    j2b = regions['2b']['j']
    k2b = regions['2b']['k']
    i3 = regions['3']['i']
    j3 = regions['3']['j']
    k3 = regions['3']['k']
    i3b = regions['3b']['i']
    j3b = regions['3b']['j']
    k3b = regions['3b']['k']
    i4 = regions['4']['i']
    j4 = regions['4']['j']
    k4 = regions['4']['k']
    i5 = regions['5']['i']
    j5 = regions['5']['j']
    k5 = regions['5']['k']
    i6 = regions['6']['i']
    j6 = regions['6']['j']
    k6 = regions['6']['k']
    i7 = regions['7']['i']
    j7 = regions['7']['j']
    k7 = regions['7']['k']
    i8 = regions['8']['i']
    j8 = regions['8']['j']
    k8 = regions['8']['k']
    i9 = regions['9']['i']
    j9 = regions['9']['j']
    k9 = regions['9']['k']
    i10 = regions['10']['i']
    j10 = regions['10']['j']
    k10 = regions['10']['k']
    
    # Initialize gain term arrays
    dMi_loss = np.zeros((Hlen,bins,bins))
    dMj_loss = np.zeros((Hlen,bins,bins))
    dNi_loss = np.zeros((Hlen,bins,bins))
    dM_gain  = np.zeros((Hlen,bins,bins,2))
    dN_gain  = np.zeros((Hlen,bins,bins,2))
  
    # Calculate transfer rates (rectangular integration, source space)
    # Collection (eqs. 23-25 in Wang et al. 2007)
    # ii collecting jj 

    dMi_loss[k1,i1,j1] = integrate_fast_kernel(1,0,0,PK[:,i1,j1],ak1[k1,i1],ck1[k1,i1],ak2[k1,j1],ck2[k1,j1],'rectangle',x1=x11[k1,i1],x2=x21[k1,i1],y1=x12[k1,j1],y2=x22[k1,j1])
    dMj_loss[k1,i1,j1] = integrate_fast_kernel(0,1,0,PK[:,i1,j1],ak1[k1,i1],ck1[k1,i1],ak2[k1,j1],ck2[k1,j1],'rectangle',x1=x11[k1,i1],x2=x21[k1,i1],y1=x12[k1,j1],y2=x22[k1,j1])
    dNi_loss[k1,i1,j1] = integrate_fast_kernel(0,0,0,PK[:,i1,j1],ak1[k1,i1],ck1[k1,i1],ak2[k1,j1],ck2[k1,j1],'rectangle',x1=x11[k1,i1],x2=x21[k1,i1],y1=x12[k1,j1],y2=x22[k1,j1])
    #dNj_loss = dNi_loss.copy() # Nj loss should be same as Ni loss
    
    # Condition 4: Self collection. All Mass/Number goes into ii+sbin = jj+sbin kbin
    xi1 = x11[k4,i4].copy()
    xi2 = x21[k4,i4].copy() 
    xj1 = x12[k4,j4].copy()
    xj2 = x22[k4,j4].copy()
    
    dM_gain[k4,i4,j4,0]  = integrate_fast_kernel(0,0,1,PK[:,i4,j4],ak1[k4,i4],ck1[k4,i4],ak2[k4,j4],ck2[k4,j4],'rectangle',x1=xi1,x2=xi2,y1=xj1,y2=xj2) 
    dN_gain[k4,i4,j4,0]  = integrate_fast_kernel(0,0,0,PK[:,i4,j4],ak1[k4,i4],ck1[k4,i4],ak2[k4,j4],ck2[k4,j4],'rectangle',x1=xi1,x2=xi2,y1=xj1,y2=xj2) 


    # Condition 2:
    # k bin: Lower triangle region. Just clips BR corner.
    #                       Triangle = ((xi1,xj1),(xi1,y_left_edge),(xi2,xj1))
    xt1 = x11[k2,i2].copy()
    yt1 = x12[k2,j2].copy()
    xt2 = x11[k2,i2].copy()
    yt2 = y_left_edge[k2,i2,j2].copy()
    xt3 = x21[k2,i2].copy()
    yt3 = x12[k2,j2].copy()
    
    dM_gain[k2,i2,j2,0] = integrate_fast_kernel(0,0,1,PK[:,i2,j2],ak1[k2,i2],ck1[k2,i2],ak2[k2,j2],ck2[k2,j2],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
    dM_gain[k2,i2,j2,1] = (dMi_loss[k2,i2,j2]+dMj_loss[k2,i2,j2])-dM_gain[k2,i2,j2,0]
    
    dN_gain[k2,i2,j2,0] = integrate_fast_kernel(0,0,0,PK[:,i2,j2],ak1[k2,i2],ck1[k2,i2],ak2[k2,j2],ck2[k2,j2],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
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
    
    dM_gain[k3,i3,j3,0] = integrate_fast_kernel(0,0,1,PK[:,i3,j3],ak1[k3,i3],ck1[k3,i3],ak2[k3,j3],ck2[k3,j3],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
    dM_gain[k3,i3,j3,1] = (dMi_loss[k3,i3,j3]+dMj_loss[k3,i3,j3])-dM_gain[k3,i3,j3,0]
        
    dN_gain[k3,i3,j3,0] = integrate_fast_kernel(0,0,0,PK[:,i3,j3],ak1[k3,i3],ck1[k3,i3],ak2[k3,j3],ck2[k3,j3],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
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
    
    
    dM_gain[k5,i5,j5,0] = integrate_fast_kernel(0,0,1,PK[:,i5,j5],ak1[k5,i5],ck1[k5,i5],ak2[k5,j5],ck2[k5,j5],'rectangle',x1=xr1,x2=xr2,y1=yr1,y2=yr2)+\
                          integrate_fast_kernel(0,0,1,PK[:,i5,j5],ak1[k5,i5],ck1[k5,i5],ak2[k5,j5],ck2[k5,j5],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
    
    dM_gain[k5,i5,j5,1] = (dMi_loss[k5,i5,j5]+dMj_loss[k5,i5,j5])-dM_gain[k5,i5,j5,0]
        
    dN_gain[k5,i5,j5,0] = integrate_fast_kernel(0,0,0,PK[:,i5,j5],ak1[k5,i5],ck1[k5,i5],ak2[k5,j5],ck2[k5,j5],'rectangle',x1=xr1,x2=xr2,y1=yr1,y2=yr2)+\
                          integrate_fast_kernel(0,0,0,PK[:,i5,j5],ak1[k5,i5],ck1[k5,i5],ak2[k5,j5],ck2[k5,j5],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
                       
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
    
    
    dM_gain[k6,i6,j6,0] = integrate_fast_kernel(0,0,1,PK[:,i6,j6],ak1[k6,i6],ck1[k6,i6],ak2[k6,j6],ck2[k6,j6],'rectangle',x1=xr1,x2=xr2,y1=yr1,y2=yr2)+\
                          integrate_fast_kernel(0,0,1,PK[:,i6,j6],ak1[k6,i6],ck1[k6,i6],ak2[k6,j6],ck2[k6,j6],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
    
    dM_gain[k6,i6,j6,1] = (dMi_loss[k6,i6,j6]+dMj_loss[k6,i6,j6])-dM_gain[k6,i6,j6,0]
        
    dN_gain[k6,i6,j6,0] = integrate_fast_kernel(0,0,0,PK[:,i6,j6],ak1[k6,i6],ck1[k6,i6],ak2[k6,j6],ck2[k6,j6],'rectangle',x1=xr1,x2=xr2,y1=yr1,y2=yr2)+\
                          integrate_fast_kernel(0,0,0,PK[:,i6,j6],ak1[k6,i6],ck1[k6,i6],ak2[k6,j6],ck2[k6,j6],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
                       
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
    
    dM_gain[k7,i7,j7,1] = integrate_fast_kernel(0,0,1,PK[:,i7,j7],ak1[k7,i7],ck1[k7,i7],ak2[k7,j7],ck2[k7,j7],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
    dM_gain[k7,i7,j7,0] = (dMi_loss[k7,i7,j7]+dMj_loss[k7,i7,j7])-dM_gain[k7,i7,j7,1]
    
    dN_gain[k7,i7,j7,1] = integrate_fast_kernel(0,0,0,PK[:,i7,j7],ak1[k7,i7],ck1[k7,i7],ak2[k7,j7],ck2[k7,j7],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
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
    
    dM_gain[k8,i8,j8,0] = integrate_fast_kernel(0,0,1,PK[:,i8,j8],ak1[k8,i8],ck1[k8,i8],ak2[k8,j8],ck2[k8,j8],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
    dM_gain[k8,i8,j8,1] = (dMi_loss[k8,i8,j8]+dMj_loss[k8,i8,j8])-dM_gain[k8,i8,j8,0]
    
    dN_gain[k8,i8,j8,0] = integrate_fast_kernel(0,0,0,PK[:,i8,j8],ak1[k8,i8],ck1[k8,i8],ak2[k8,j8],ck2[k8,j8],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
    dN_gain[k8,i8,j8,1] = (dNi_loss[k8,i8,j8])-dN_gain[k8,i8,j8,0]
        
    # Condition 9: Rectangle collection within k bin. All Mass/Number goes into kbin
    xi1 = x11[k9,i9].copy()
    xi2 = x21[k9,i9].copy() 
    xj1 = x12[k9,j9].copy()
    xj2 = x22[k9,j9].copy()

    dM_gain[k9,i9,j9,0]  = integrate_fast_kernel(0,0,1,PK[:,i9,j9],ak1[k9,i9],ck1[k9,i9],ak2[k9,j9],ck2[k9,j9],'rectangle',x1=xi1,x2=xi2,y1=xj1,y2=xj2) 
    dN_gain[k9,i9,j9,0]  = integrate_fast_kernel(0,0,0,PK[:,i9,j9],ak1[k9,i9],ck1[k9,i9],ak2[k9,j9],ck2[k9,j9],'rectangle',x1=xi1,x2=xi2,y1=xj1,y2=xj2) 

    # Condition 10: Rectangle collection within k+1 bin. All Mass/Number goes into kbin
    xi1 = x11[k10,i10].copy()
    xi2 = x21[k10,i10].copy() 
    xj1 = x12[k10,j10].copy()
    xj2 = x22[k10,j10].copy()
    
    dM_gain[k10,i10,j10,1]  = integrate_fast_kernel(0,0,1,PK[:,i10,j10],ak1[k10,i10],ck1[k10,i10],ak2[k10,j10],ck2[k10,j10],'rectangle',x1=xi1,x2=xi2,y1=xj1,y2=xj2) 
    dN_gain[k10,i10,j10,1]  = integrate_fast_kernel(0,0,0,PK[:,i10,j10],ak1[k10,i10],ck1[k10,i10],ak2[k10,j10],ck2[k10,j10],'rectangle',x1=xi1,x2=xi2,y1=xj1,y2=xj2) 
   
    # Condition 11 (2b): Triangle collection within k+1 bin. Occurs when xk+1 clips BR corner and intersects top edge.
    # Triangle = ((xi2,xj1),(x_top_edge,xj2),(xi2,xj2))
    xt1 = x_top_edge[k2b,i2b,j2b].copy()
    yt1 = x22[k2b,j2b].copy()
    xt2 = x21[k2b,i2b].copy()
    yt2 = x22[k2b,j2b].copy()
    xt3 = x21[k2b,i2b].copy()
    yt3 = x12[k2b,j2b].copy()
    
    dM_gain[k2b,i2b,j2b,1] = integrate_fast_kernel(0,0,1,PK[:,i2b,j2b],ak1[k2b,i2b],ck1[k2b,i2b],ak2[k2b,j2b],ck2[k2b,j2b],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
    dM_gain[k2b,i2b,j2b,0] = (dMi_loss[k2b,i2b,j2b]+dMj_loss[k2b,i2b,j2b])-dM_gain[k2b,i2b,j2b,1]
    
    dN_gain[k2b,i2b,j2b,1] = integrate_fast_kernel(0,0,0,PK[:,i2b,j2b],ak1[k2b,i2b],ck1[k2b,i2b],ak2[k2b,j2b],ck2[k2b,j2b],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
    dN_gain[k2b,i2b,j2b,0] = (dNi_loss[k2b,i2b,j2b])-dN_gain[k2b,i2b,j2b,1]
    
    # Condition 12 (3b): Triangle collection within k+1 bin. Occurs when xk+1 clips UL corner and intersects with right edge.
    # Triangle = ((xi1,xj2),(xi2,xj2),(xi2,y_right_edge)
    xt1 = x11[k3b,i3b].copy()
    yt1 = x22[k3b,j3b].copy()
    xt2 = x21[k3b,i3b].copy()
    yt2 = x22[k3b,j3b].copy()
    xt3 = x21[k3b,i3b].copy()
    yt3 = y_right_edge[k3b,i3b,j3b].copy()
    
    dM_gain[k3b,i3b,j3b,1] = integrate_fast_kernel(0,0,1,PK[:,i3b,j3b],ak1[k3b,i3b],ck1[k3b,i3b],ak2[k3b,j3b],ck2[k3b,j3b],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
    dM_gain[k3b,i3b,j3b,0] = (dMi_loss[k3b,i3b,j3b]+dMj_loss[k3b,i3b,j3b])-dM_gain[k3b,i3b,j3b,1]
    
    dN_gain[k3b,i3b,j3b,1] = integrate_fast_kernel(0,0,0,PK[:,i3b,j3b],ak1[k3b,i3b],ck1[k3b,i3b],ak2[k3b,j3b],ck2[k3b,j3b],'triangle',xt1=xt1,yt1=yt1,xt2=xt2,yt2=yt2,xt3=xt3,yt3=yt3)
    dN_gain[k3b,i3b,j3b,0] = (dNi_loss[k3b,i3b,j3b])-dN_gain[k3b,i3b,j3b,1]   
    
    return dMi_loss, dMj_loss, dM_gain, dNi_loss, dN_gain


def calculate_1mom(i1,j1,n1,n2,dMi_loss,dMj_loss,dM_gain,kmin,kmid,dMb_gain_frac,breakup):
    
    '''
     This function calculate mass transfer rates
     for collision-coalescence and collisional breakup between
     each distribution for 1 moment calculations.
    '''
    
    Hlen,bins = n1.shape
    
    n12 = n1[:,:,None]*n2[:,None,:]
    
    M1_loss = np.nansum(n12*(dMi_loss[None,:,:]),axis=2) 
    M2_loss = np.nansum(n12*(dMj_loss[None,:,:]),axis=1) 
       
    # ChatGPT is the GOAT for telling me about np.add.at!
    M_gain = np.zeros((Hlen,bins))
    np.add.at(M_gain, (np.arange(Hlen)[:,None,None],kmin), n12*(dM_gain[:,:,0][None,:,:]))
    np.add.at(M_gain,  (np.arange(Hlen)[:,None,None],kmid), n12*(dM_gain[:,:,1][None,:,:]))
    
    # ELD NOTE: Breakup here can take losses from each pair and calculate gains
    # for breakup. Breakup gain arrays will be 3D.
    if breakup:
        
        Mij_loss = n12[:,i1,j1]*(dMi_loss[i1,j1]+dMj_loss[i1,j1])[None,:]
        Mb_gain = np.nansum((dMb_gain_frac[:,kmin[i1,j1]][None,:,:])*Mij_loss[:,None,:],axis=2)
        
    else:
        
        Mb_gain  = np.zeros((Hlen,bins))
        
        
    return M1_loss, M2_loss, M_gain, Mb_gain   


def calculate_2mom(x11,x21,ak1,ck1,M1, 
                   x12,x22,ak2,ck2,M2,
                   PK,xi1,xi2,kmin,kmid,cond_1_orig, 
                   dMb_gain_frac,dNb_gain_frac, 
                   sc_inds,breakup=False):
    
    '''
     This function calculate mass and number transfer rates
     for collision-coalescence and collisional breakup between
     each distribution for 2 moment calculations (mass + number).
    '''
    
    Hlen,bins = x11.shape
    
    regions = setup_regions(x11,x21,ak1,ck1,x12,x22,ak2,ck2,xi1,xi2,kmin,cond_1_orig,sc_inds)
    
    dMi_loss, dMj_loss, dM_gain, dNi_loss, dN_gain = calculate_regions(x11,x21,ak1,ck1,x12,x22,ak2,ck2,PK,xi1,xi2,regions)
    
    M1_loss = np.nansum(dMi_loss,axis=2) 
    N1_loss = np.nansum(dNi_loss,axis=2) 
    
    M2_loss = np.nansum(dMj_loss,axis=1) 
    N2_loss = np.nansum(dNi_loss,axis=1)
       
    # ChatGPT is the GOAT for telling me about np.add.at!
    M_gain = np.zeros((Hlen,bins))
    np.add.at(M_gain, (np.arange(Hlen)[:,None,None],kmin), dM_gain[:,:,:,0])
    np.add.at(M_gain,  (np.arange(Hlen)[:,None,None],kmid), dM_gain[:,:,:,1])
    
    N_gain = np.zeros((Hlen,bins))
    np.add.at(N_gain,  (np.arange(Hlen)[:,None,None],kmin), dN_gain[:,:,:,0])
    np.add.at(N_gain,  (np.arange(Hlen)[:,None,None],kmid), dN_gain[:,:,:,1])
    
    # Initialize gain term arrays
    Mb_gain  = np.zeros((Hlen,bins))
    Nb_gain  = np.zeros((Hlen,bins))
      
    # ELD NOTE: Breakup here can take losses from each pair and calculate gains
    # for breakup. Breakup gain arrays will be 3D.
    if breakup:
        
        k1 = regions['1']['k']
        i1 = regions['1']['i']
        j1 = regions['1']['j']
        
        Mij_loss = dMi_loss[k1,i1,j1]+dMj_loss[k1,i1,j1]
  
        np.add.at(Mb_gain,  k1, np.transpose(dMb_gain_frac[:,kmin[i1,j1]]*Mij_loss))
        np.add.at(Nb_gain,  k1, np.transpose(dNb_gain_frac[:,kmin[i1,j1]]*Mij_loss))
        
   
    return M1_loss, M2_loss, M_gain, Mb_gain, N1_loss, N2_loss, N_gain, Nb_gain    



class Interaction():
    
    '''
    interaction class initializes interaction objects that contain
    arrays in the form (Ndists x M x N) which specifies the interactions among
    all Ndists distributions.
    '''
    
    def __init__(self,dists,cc_dest,br_dest,Eagg,Ecb,Ebr,frag_dict=None,kernel='Golovin',mom_num=2,parallel=False,n_jobs=12):
        
        # cc_dest is an integer (from 1 to len(dists)) that determines the destination 
        # for coalesced particles
        
        # br_dest is an integer  (from 1 to len(dists)) that determines the destination
        # for fragments
        
        self.dists = dists
        self.frag_dict = frag_dict
        self.kernel = kernel
        self.indc = cc_dest-1
        self.indb = br_dest-1
        self.Eagg = Eagg 
        self.Ebr = Ebr 
        self.Ecb = Ecb
        self.parallel = parallel
        self.n_jobs = n_jobs
        self.mom_num = mom_num
        
        self.cnz = False
        
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
        
        # NOTE: Might be better to batch the bin-bin pairs as that would allow for
        # parallelized computations for box and steady-state setups.
        
        # self.batches = np.array_split(np.arange(self.bins))
        
        #if self.parallel:
        self.batches = np.array_split(np.arange(self.Hlen),self.n_jobs) 
        if self.Ebr>0.:
            self.breakup = True 
        else:
            self.breakup = False


        self.pnum = int((self.dnum*(self.dnum+1))/2)

        # TESTING
        #self.cond_1 = ((self.ind_i>=(self.bins-self.sbin)) | (self.ind_j>=(self.bins-self.sbin)))
        
        # WORKING
        if self.mom_num == 2:
            self.cond_1 = np.tile(((self.ind_i>=(self.bins-self.sbin)) | (self.ind_j>=(self.bins-self.sbin))),(self.Hlen,1,1))
            # WORKING
            self.self_col = np.ones((self.Hlen,self.pnum,self.bins,self.bins),dtype=int)
            self.sc_inds = np.ones((self.Hlen,self.pnum,self.bins,self.bins),dtype=int)
            
            dd = 0
            for d1 in range(self.dnum):
                for d2 in range(d1,self.dnum):
                    if d1==d2:
                        # TEST
                        #self.self_col[dd,:,:] = np.triu(np.ones((self.bins,self.bins),dtype=int),k=0)   
                        # WORKING
                        self.self_col[:,dd,:,:] = np.tile(np.triu(np.ones((self.bins,self.bins),dtype=int),k=0),(self.Hlen,1,1))
                    dd += 1
            
        elif self.mom_num == 1:
            self.cond_1 = np.tile(((self.ind_i>=(self.bins-self.sbin)) | (self.ind_j>=(self.bins-self.sbin))),(1,1,1))
            self.self_col = np.ones((1,self.pnum,self.bins,self.bins),dtype=int)
            self.sc_inds = np.ones((1,self.pnum,self.bins,self.bins),dtype=int)
            
            dd = 0
            for d1 in range(self.dnum):
                for d2 in range(d1,self.dnum):
                    if d1==d2:
                        # TEST
                        #self.self_col[dd,:,:] = np.triu(np.ones((self.bins,self.bins),dtype=int),k=0)   
                        # WORKING
                        self.self_col[:,dd,:,:] = np.tile(np.triu(np.ones((self.bins,self.bins),dtype=int),k=0),(1,1,1))
                    dd += 1
            
        
        # Combine cond_2 with self_col?    
        
        self.dMb_gain_frac = np.zeros((self.bins,self.bins))
        self.dNb_gain_frac = np.zeros((self.bins,self.bins))
        
        if self.Ebr>0.: # Setup fragment distribution if Ebr>0.     
            self.setup_fragments()
            
        self.PK = self.create_kernels(dists)
         
        if parallel:
            
            '''
            Create memory map files and variables that will be used for both 
            2 moment and 1 moment selections.
            '''
            
            self.temp_dir = gettempdir()
                
            dMb_gain_file = os.path.join(self.temp_dir,'dMb_gain_frac.pkl')
            dNb_gain_file = os.path.join(self.temp_dir,'dNb_gain_frac.pkl')
            PK_file       = os.path.join(self.temp_dir,'PK.pkl')
            kmin_file     = os.path.join(self.temp_dir,'kmin.pkl')
            kmid_file     = os.path.join(self.temp_dir,'kmid.pkl')
            cond1_file    = os.path.join(self.temp_dir,'cond_1.pkl')
            selfcol_file  = os.path.join(self.temp_dir,'self_col.pkl')
            
            dump(self.dMb_gain_frac,dMb_gain_file)
            dump(self.dNb_gain_frac,dNb_gain_file)
            dump(self.PK,PK_file)
            dump(self.kmin,kmin_file)
            dump(self.kmid,kmid_file)
            dump(self.cond_1,cond1_file)
            dump(self.self_col,selfcol_file)
 
            self.dMb_gain_frac = load(dMb_gain_file,mmap_mode='r')
            self.dNb_gain_frac = load(dNb_gain_file,mmap_mode='r')
            self.PK            = load(PK_file,mmap_mode='r')       
            self.kmin          = load(kmin_file,mmap_mode='r')
            self.kmid          = load(kmid_file,mmap_mode='r')
            self.cond_1        = load(cond1_file,mmap_mode='r')
            self.self_col      = load(selfcol_file,mmap_mode='r')
        
    
        # Calculate transfer rates if single-moment scheme is chosen.
        if self.mom_num==1:  
            
            '''
                Using memory mapping here to save arrays temporarily if processing
                in parallel.
            
            '''
        
            self.dMi_loss = np.zeros((self.pnum,1,self.bins,self.bins))
            self.dMj_loss = np.zeros((self.pnum,1,self.bins,self.bins))
            self.dM_gain = np.zeros((self.pnum,1,self.bins,self.bins,2))
            
            self.regions = np.empty((self.pnum,),dtype=object)
            
            ak1mom = np.zeros((1,self.bins))
            ck1mom = np.ones((1,self.bins))
        
            x1 = self.xi1.reshape(1,-1)
            x2 = self.xi2.reshape(1,-1)
            
        
            dd = 0
        
            # Calculate regions and mass transfer rates
            for d1 in range(self.dnum):
                for d2 in range(d1,self.dnum):
                        
                    self.regions[dd] = setup_regions(x1,x2,ak1mom,ck1mom,x1,x2,ak1mom,ck1mom,
                                      self.xi1,self.xi2,self.kmin,self.cond_1,self.self_col[:,dd,:,:])
            
                    self.dMi_loss[dd,:,:,:], self.dMj_loss[dd,:,:,:], self.dM_gain[dd,:,:,:,:], _, _ =  calculate_regions(x1,x2,ak1mom,ck1mom, 
                               x1,x2,ak1mom,ck1mom,self.PK[:,d1,d2,:,:],self.xi1,self.xi2,self.regions[dd])
                    
                    dd += 1
        
        
            if parallel:
                
                # Write all arrays to memory map files
                
                dMi_loss_file = os.path.join(self.temp_dir,'dMi_loss.pkl')
                dMj_loss_file = os.path.join(self.temp_dir,'dMj_loss.pkl')  
                dM_gain_file  = os.path.join(self.temp_dir,'dM_gain.pkl')
                
                dump(self.dMi_loss,dMi_loss_file)
                dump(self.dMj_loss,dMj_loss_file)
                dump(self.dM_gain,dM_gain_file)
                
                self.dMi_loss = load(dMi_loss_file,mmap_mode='r')
                self.dMj_loss = load(dMj_loss_file,mmap_mode='r')
                self.dM_gain  = load(dM_gain_file,mmap_mode='r')
                
                for dd in range(self.pnum):
                    i_region_file = os.path.join(self.temp_dir,'i1_region_{}.pkl'.format(dd))
                    j_region_file = os.path.join(self.temp_dir,'j1_region_{}.pkl'.format(dd))
                    dump(self.regions[dd]['1']['i'],i_region_file)
                    dump(self.regions[dd]['1']['j'],j_region_file)
                    
                    self.regions[dd]['1']['i'] = load(i_region_file,mmap_mode='r')
                    self.regions[dd]['1']['j'] = load(j_region_file,mmap_mode='r')

        # Create dist_num x height x bins arrays for N and M 
        self.pack(dists)
    
    def unpack_1mom(self):
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
                self.dists[d1,zz].diagnose_1mom()
    
    
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
        
        PK[PK<1e-5] = 0.
        #PK[np.abs(PK-1)<1e-4] = 1.
        
       # if self.kernel!='Hydro':
       #     PK = np.round(PK)
        # Explicitly set Sum, product, and constant kernels. This is because
        # numerical round-off errors are large.
        if self.kernel == 'Golovin':
           PK = HK.copy()
           PK[0,:,:,:,:] = 0.0
           PK[1,:,:,:,:] = 1.0
           PK[2,:,:,:,:] = 1.0
           PK[3,:,:,:,:] = 0.0
           
        elif self.kernel == 'Product':
           PK = HK.copy()
           PK[0,:,:,:,:] = 0.0
           PK[1,:,:,:,:] = 0.0
           PK[2,:,:,:,:] = 0.0
           PK[3,:,:,:,:] = 1.0
           
        elif self.kernel == 'Constant':
           PK = HK.copy()
           PK[0,:,:,:,:] = 1.0
           PK[1,:,:,:,:] = 0.0
           PK[2,:,:,:,:] = 0.0
           PK[3,:,:,:,:] = 0.0
        
        #print(np.unique(np.round(PK)))
        #print('PK shape=',PK.shape)
        #raise Exception()
        
        return PK


    def setup_fragments(self):
            
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
            
            self.dMb_gain_frac[np.isnan(self.dMb_gain_frac)|np.isnan(self.dNb_gain_frac)] = 0.
            self.dNb_gain_frac[np.isnan(self.dMb_gain_frac)|np.isnan(self.dNb_gain_frac)] = 0.


    def plot_source_target_vec(self,dist_ind1,dist_ind2,kk,ii,jj,invert=False,full=False):
        
        '''
        Method for plotting the integration region of the (kk,ii,jj) interaction.
        Useful for debugging.
        '''
        
        from matplotlib.patches import Polygon
        
        import matplotlib.pyplot as plt
        
        dist1 = self.dists[dist_ind1,kk]
        dist2 = self.dists[dist_ind2,kk]
        
        regions = setup_regions(dist1.x1,dist1.x2,
                                dist1.aki,dist1.cki,
                                dist2.x1,dist2.x2,
                                dist2.aki,dist2.cki,
                                self.xi1,self.xi2,self.kmin,self.cond_1,self.self_col)
        

        xi1 = dist1.x1[ii]
        xi2 = dist1.x2[ii]
        xj1 = dist2.x1[jj]
        xj2 = dist2.x2[jj]
    
        # NOTE: Need to avoid doing anything with the last bin
        kbin_min = dist1.xi1[:,None]+dist2.xi1[None,:]
        kbin_max = dist1.xi2[:,None]+dist2.xi2[None,:]
        
        xk_min = dist2.xi2[self.kmin][ii,jj]
        
        # Find region with requested (ii,jj) indices.
        x_bottom_edge = regions['x_bottom_edge']
        x_top_edge = regions['x_top_edge']
        y_left_edge = regions['y_left_edge']
        y_right_edge = regions['y_right_edge']
        i1 = regions['1']['i']
        j1 = regions['1']['j']
        k1 = regions['1']['k']
        i2 = regions['2']['i']
        j2 = regions['2']['j']
        k2 = regions['2']['k']
        i2b = regions['2b']['i']
        j2b = regions['2b']['j']
        k2b = regions['2b']['k']
        i3 = regions['3']['i']
        j3 = regions['3']['j']
        k3 = regions['3']['k']
        i3b = regions['3b']['i']
        j3b = regions['3b']['j']
        k3b = regions['3b']['k']
        i4 = regions['4']['i']
        j4 = regions['4']['j']
        k4 = regions['4']['k']
        i5 = regions['5']['i']
        j5 = regions['5']['j']
        k5 = regions['5']['k']
        i6 = regions['6']['i']
        j6 = regions['6']['j']
        k6 = regions['6']['k']
        i7 = regions['7']['i']
        j7 = regions['7']['j']
        k7 = regions['7']['k']
        i8 = regions['8']['i']
        j8 = regions['8']['j']
        k8 = regions['8']['k']
        i9 = regions['9']['i']
        j9 = regions['9']['j']
        k9 = regions['9']['k']
        i10 = regions['10']['i']
        j10 = regions['10']['j']
        k10 = regions['10']['k']
        
        cond = np.zeros((self.Hlen,self.bins,self.bins),dtype=int)
        
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
        cond[k2b,i2b,j2b] = 11 
        cond[k3b,i3b,j3b] = 12
            
        fig, ax = plt.subplots();   
        
        if invert:
            source_rec = Polygon(((xj1,xi1),(xj1,xi2),(xj2,xi2),(xj2,xi1)))
        else:           
            source_rec = Polygon(((xi1,xj1),(xi1,xj2),(xi2,xj2),(xi2,xj1)))
        
        ax.add_patch(source_rec)
        
        if invert:
            # NOTE: need to reformat with new gain bin regions
            ax.plot([kbin_min[ii,jj]-xi1,kbin_min[ii,jj]-xi2],[xi1,xi2],'b')
            ax.plot([xk_min-xi1,xk_min-xi2],[xi1,xi2],'k')
            ax.plot([kbin_max[ii,jj]-xi1,kbin_max[ii,jj]-xi2],[xi1,xi2],'r')

            ax.invert_yaxis()
            
        else:
            
            if cond[kk,ii,jj]==2:
                # Triangle = ((xi1,xj1),(xi1,x_left_edge),(xi2,xj1))
                kgain_t = Polygon(((xi1,xj1),(xi1,y_left_edge[kk,ii,jj]),(xi2,xj1)),closed=True,facecolor='purple')
            elif cond[kk,ii,jj]==3:
                kgain_t = Polygon((((xi1,xj1),(xi1,xj2),(x_bottom_edge[kk,ii,jj],xj1))),closed=True,facecolor='purple')
            elif cond[kk,ii,jj]==11:
                kgain_t = Polygon((((xi1,xj1),(x_top_edge[kk,ii,jj],xj2),(xi2,xj2))),closed=True,facecolor='purple')
            elif cond[kk,ii,jj]==12:
                kgain_t = Polygon(((xi1,xj2),(xi2,xj2),(xi2,y_right_edge[kk,ii,jj])),closed=True,facecolor='purple')
            elif cond[kk,ii,jj]==7:
                kgain_t = Polygon(((x_top_edge[kk,ii,jj],xj2),(xi2,xj2),(xi2,y_right_edge[kk,ii,jj])),closed=True,facecolor='purple')
            elif cond[kk,ii,jj]==8:
                kgain_t = Polygon(((xi1,xj1),(xi1,y_left_edge[kk,ii,jj]),(x_bottom_edge[kk,ii,jj],xj1)),closed=True,facecolor='purple')
            elif np.isin(cond[kk,ii,jj],[4,9,10]):
                kgain_r = Polygon(((xi1,xj1),(xi1,xj2),(xi2,xj2),(xi2,xj1)),closed=True,facecolor='purple')
            elif cond[kk,ii,jj]==5:
                kgain_t = Polygon(((x_top_edge[kk,ii,jj],xj1),(x_top_edge[kk,ii,jj],xj2),(x_bottom_edge[kk,ii,jj],xj1)),closed=True,facecolor='purple')
                kgain_r =Polygon(((xi1,xj1),(xi1,xj2),(x_top_edge[kk,ii,jj],xj2),(x_top_edge[kk,ii,jj],xj1)),closed=True,facecolor='orange')
            elif cond[kk,ii,jj]==6:
                kgain_t = Polygon(((xi1,y_right_edge[kk,ii,jj]),(xi1,y_left_edge[kk,ii,jj]),(xi2,y_right_edge[kk,ii,jj])),closed=True,facecolor='purple')
                kgain_r =Polygon(((xi1,xj1),(xi1,y_right_edge[kk,ii,jj]),(xi2,y_right_edge[kk,ii,jj]),(xi2,xj1)),closed=True,facecolor='orange')
            
            if  (np.isin(cond[kk,ii,jj],[2,3,5,6,7,8,11,12])):
                ax.add_patch(kgain_t)
            
            if (np.isin(cond[ii,jj],[4,5,6,9])):
                ax.add_patch(kgain_r)
            
            ax.plot([dist1.xi1[ii],dist1.xi2[ii]],[kbin_min[ii,jj]-dist1.xi1[ii],kbin_min[ii,jj]-dist1.xi2[ii]],'b')
            ax.plot([dist1.xi1[ii],dist1.xi2[ii]],[xk_min-dist1.xi1[ii],xk_min-dist1.xi2[ii]],'k')
            ax.plot([dist1.xi1[ii],dist1.xi2[ii]],[kbin_max[ii,jj]-dist1.xi1[ii],kbin_max[ii,jj]-dist1.xi2[ii]],'r')
    
        return fig, ax, cond

        
    # Advance PSD Mbins and Nbins by one time/height step
    def interact_1mom(self,dt):

        # Ndists x height x bins
        Mbins_old = self.Mbins.copy() 
        Mbins = np.zeros_like(Mbins_old)

        M_loss = np.zeros_like(Mbins)
        M_gain = np.zeros_like(Mbins)
        
        indc = self.indc
        indb = self.indb

        dd = 0
        
        for d1 in range(self.dnum):
            for d2 in range(d1,self.dnum):
                 
                if self.parallel:
                    
                    
                    # (dnum x height x bins)
                    ck1  = self.cki[d1,:,:]               
                    ck2  = self.cki[d2,:,:] 
                    
                                        
                    gain_loss_temp = Parallel(n_jobs=self.n_jobs,verbose=0)(delayed(calculate_1mom)(
                                        self.regions[dd]['1']['i'],
                                        self.regions[dd]['1']['j'],
                                        ck1[batch,:],ck2[batch,:],
                                        self.dMi_loss[dd,0,:,:],
                                        self.dMj_loss[dd,0,:,:],
                                        self.dM_gain[dd,0,:,:,:],
                                        self.kmin,self.kmid,self.dMb_gain_frac,self.breakup) for batch in self.batches)  


                    M1_loss_temp = np.vstack([gl[0] for gl in gain_loss_temp])
                    M2_loss_temp = np.vstack([gl[1] for gl in gain_loss_temp])
                    M_gain_temp =  np.vstack([gl[2] for gl in gain_loss_temp])
                    Mb_gain_temp = np.vstack([gl[3] for gl in gain_loss_temp])
                    
      
                else:
                
                    # (dnum x height x bins)
                    ck1  = self.cki[d1,:,:]               
                    ck2  = self.cki[d2,:,:] 
                    
                    
                    # Batch each (bin x bin) combination. Make sure only to include
                    # pairs that will actually be used.
                    #ck12 = ck1[:,:,None]*ck2[:,None,:].reshape(self.dnum,-1)
                    
                    # self.dMi_loss[dd,0,:,:].reshape(1,-1)
                    # self.dMj_loss[dd,0,:,:].reshape(1,-1)
                    # self.dM_gain[dd,0,:,:,:].reshape(-1,2)
                    # kmin = kmin.flatten() 
                    # kmid = kmid.flatten()
                    # self.dMb_gain_frac.flatten()
                    
                    M1_loss_temp,M2_loss_temp,\
                    M_gain_temp,Mb_gain_temp =\
                    calculate_1mom(self.regions[dd]['1']['i'],
                                    self.regions[dd]['1']['j'],
                                    ck1,ck2,
                                    self.dMi_loss[dd,0,:,:],
                                    self.dMj_loss[dd,0,:,:],
                                    self.dM_gain[dd,0,:,:,:],
                                    self.kmin,self.kmid,self.dMb_gain_frac,self.breakup)
                        
                M_loss[d1,:,:]    += M1_loss_temp 
                M_loss[d2,:,:]    += M2_loss_temp
                
                M_gain[indc,:,:]  += self.Eagg*M_gain_temp
                M_gain[indb,:,:]  += self.Ebr*Mb_gain_temp

                dd += 1
                
        M_loss *= self.Ecb
        
        M_net = dt*(M_gain-M_loss) 
                
        return M_net
    
    
    # Advance PSD Mbins and Nbins by one time/height step
    def interact_2mom(self,dt):

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
        
        dd = 0
        for d1 in range(self.dnum):
            for d2 in range(d1,self.dnum):

                if self.parallel:
                    
                    # Set up batches for each bin-bin pair? Would need to unravel
                    # arrays appropriately afterward. Also would need to figure out 
                    # how to do self and cross collection appropriately.
                    
                    # Find all bin pairs
                    
                    # (dnum x height x bins)
                    x11  = self.x1[d1,:,:]
                    x21  = self.x2[d1,:,:]
                    ak1  = self.aki[d1,:,:]
                    ck1  = self.cki[d1,:,:] 
                    M1   = self.Mbins[d1,:,:]
                    
                    x12  = self.x1[d2,:,:]
                    x22  = self.x2[d2,:,:]
                    ak2  = self.aki[d2,:,:] 
                    ck2  = self.cki[d2,:,:] 
                    M2   = self.Mbins[d2,:,:]
                    
                    #
                    #Mcheck = ((M1==0.)[:,None]) | ((M2==0.)[None,:])
                    #ikeep = np.nonzero(~(self.cond_1&Mcheck))
                    
                    #Mcheck = ((M1==0.)[:,:,None]) | ((M2==0.)[:,None,:])
                    
                    #cond_1 = self.cond_1 | Mcheck # New cond_1. Basically exclude bin-pairs that are off grid and ones involving empty bins.
                    
                    
                    gain_loss_temp = Parallel(n_jobs=self.n_jobs,verbose=0)(delayed(calculate_2mom)(x11[batch,:],x21[batch,:],ak1[batch,:],ck1[batch,:],M1[batch,:],
                                            x12[batch,:],x22[batch,:],ak2[batch,:],ck2[batch,:],M2[batch,:],
                                            self.PK[:,d1,d2,:,:],self.xi1,self.xi2,self.kmin,self.kmid,self.cond_1[batch,:,:], 
                                            self.dMb_gain_frac,self.dNb_gain_frac, 
                                            self.self_col[batch,dd,:,:],breakup=self.breakup) for batch in self.batches)
                            
                    M1_loss_temp = np.vstack([gl[0] for gl in gain_loss_temp])
                    M2_loss_temp = np.vstack([gl[1] for gl in gain_loss_temp])
                    M_gain_temp =  np.vstack([gl[2] for gl in gain_loss_temp])
                    Mb_gain_temp = np.vstack([gl[3] for gl in gain_loss_temp])
                    N1_loss_temp = np.vstack([gl[4] for gl in gain_loss_temp])
                    N2_loss_temp = np.vstack([gl[5] for gl in gain_loss_temp])
                    N_gain_temp  = np.vstack([gl[6] for gl in gain_loss_temp])
                    Nb_gain_temp = np.vstack([gl[7] for gl in gain_loss_temp])
            
                else:

                    x11  = self.x1[d1,:]
                    x21  = self.x2[d1,:]
                    ak1  = self.aki[d1,:]
                    ck1  = self.cki[d1,:] 
                    M1   = self.Mbins[d1,:]
                    
                    x12  = self.x1[d2,:]
                    x22  = self.x2[d2,:]
                    ak2  = self.aki[d2,:] 
                    ck2  = self.cki[d2,:] 
                    M2   = self.Mbins[d2,:]
                    
                    # Calculate bin-source pairs
                    
                   # Mcheck = ((M1==0.)[:,:,None]) | ((M2==0.)[:,None,:])
                    
                    #cond_1 = self.cond_1 | Mcheck # New cond_1. Basically exclude bin-pairs that are off grid and ones involving empty bins.
                            
                    M1_loss_temp, M2_loss_temp, M_gain_temp, Mb_gain_temp,\
                    N1_loss_temp, N2_loss_temp, N_gain_temp, Nb_gain_temp = calculate_2mom(x11,x21,ak1,ck1,M1, 
                                    x12,x22,ak2,ck2,M2,self.PK[:,d1,d2,:,:],self.xi1,self.xi2,self.kmin,self.kmid,self.cond_1, 
                                    self.dMb_gain_frac,self.dNb_gain_frac, 
                                    self.self_col[:,dd,:,:],breakup=self.breakup)
                    
                M_loss[d1,:,:]    += M1_loss_temp 
                M_loss[d2,:,:]    += M2_loss_temp
                
                M_gain[indc,:,:]  += self.Eagg*M_gain_temp
                M_gain[indb,:,:]  += self.Ebr*Mb_gain_temp
                
                N_loss[d1,:,:]    += N1_loss_temp
                N_loss[d2,:,:]    += N2_loss_temp
                 
                N_gain[indc,:,:]  += self.Eagg*N_gain_temp
                N_gain[indb,:,:]  += self.Ebr*Nb_gain_temp
                
                dd += 1
                
        M_loss *= self.Ecb
        N_loss *= self.Ecb 
        
        M_net = dt*(M_gain-M_loss) 
        N_net = dt*(N_gain-N_loss)
        
        return M_net, N_net    
