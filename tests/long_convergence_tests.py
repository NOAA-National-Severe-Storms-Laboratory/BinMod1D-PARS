# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 14:51:56 2025

Description:
    
    These are timing and convergence tests that are used in the GMD article 
    by Dunnavan, E. L. (2026).

@author: edwin.dunnavan
"""

from binmod1d.spectral_model import spectral_1d

import numpy as np
import os

from time import perf_counter

# Get current directory
cdir = os.getcwd()

if __name__ == "__main__":
    
    # Box model
    
    outstr = 'Long_convg_test_box_{}mom_s{}_flat.nc'
    
    output_freq = 10

    bin1 = 45 # number of bins for sbin=1
    
    sbin = [16,8,4,2,1] # resolution parameter
    
    #sbin = [8] # resolution parameter
    
    bins = bin1*np.array(sbin) # Keeps xmin and xmax consistent for each resolution.   
    
    mu0 = 0
    moms = [2,1]
    
    tmax = 3600. # s 
    dt = 1.0 # s
    r0 = 9.3e-3 # mm
    m0 = 0.001*((4./3.)*np.pi*(r0)**3) # g
    Mt0 = 1.0 # g/m^3
    x0 = 1e-12 # g
    
    Nt0 = Mt0/m0 # Put in #/m^3 for model input
    mbar0 = Mt0/Nt0
    
    rtime = np.zeros((len(moms),len(sbin)))
    
    for mm in range(len(moms)):
        for ss in range(len(sbin)):
        
            smom =  spectral_1d(sbin=sbin[ss],
                                       bins=bins[ss],
                                       tmax=tmax,
                                       output_freq=output_freq,
                                       dt=dt,
                                       Nt0=0.001*Nt0,
                                       Mt0=Mt0,
                                       mbar0=mbar0,
                                       x0=x0,mu0=mu0,
                                       kernel='Long',
                                       Ecol=0.25,
                                       Es=1.0,
                                       gam_norm=True,
                                       dist_var='mass',
                                       moments=moms[mm])
            # Run model
            start = perf_counter()
            smom.run()
            end = perf_counter()
            
            rtime[mm,ss] = end-start
            
            print('Time taken = {} sec'.format(rtime[mm,ss]))
            
            # Save model
            smom.write_netcdf(os.path.join(cdir,'Output',outstr.format(moms[mm],sbin[ss])))
    
    
            del smom
    
    #Plot at one hour
    
    import matplotlib.pyplot as plt
    
    #fig, ax = plt.subplots()
    
    lstyle = ['-',':']
    #lcolors = ['k','tab:brown','tab:blue','tab:orange','tab:red','tab:green']
    
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    lcolors = default_colors[:len(sbin)]
    
    # Now do plotting at 1 hr
    for mm in range(len(moms)):
        for ss in range(len(sbin)):
            
            smom_test = spectral_1d(load=os.path.join(cdir,'Output',outstr.format(moms[mm],sbin[ss])))
    
            if (mm==0) & (ss==0):     
                fig, ax = smom_test.plot_dists(normbin=True)
            else:
                smom_test.plot_dists(ax=ax,lstyle=lstyle[mm],lcolor=lcolors[ss],normbin=True)
                
            del smom_test
    
    
    