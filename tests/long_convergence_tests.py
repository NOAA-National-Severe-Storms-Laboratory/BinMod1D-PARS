# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 14:51:56 2025

Description:
    
    These are timing and convergence tests that are used in the GMD article 
    by Dunnavan, E. L. (2026).

@author: edwin.dunnavan
"""

from binmod1d.spectral_model import spectral_1d

from binmod1d.distribution import rain_terminal_velocity

import numpy as np
import os

from time import perf_counter

# Get current directory
cdir = os.getcwd()

if __name__ == "__main__":
    
    # Box model
    
    vt = lambda d: rain_terminal_velocity(d)
    
    outstr = 'Long_Straub_convg_test_box_{}mom_s{}.nc'
    
    output_freq = 10

    bin1 = 45 # number of bins for sbin=1
    
    #sbin = [16,8,4,2,1] # resolution parameter
    
    sbin = [16,8,4,2,1] # resolution parameter
    
    
    #sbin = [4] # resolution parameter
    
    bins = bin1*np.array(sbin) # Keeps xmin and xmax consistent for each resolution.   
    
    mu0 = 0
    moms = [2,1]
    
    #moms = [2]
    
    tmax = 3600.*3 # s 
    dt = 1.0 # s
    r0 = 9.3e-3 # mm
    m0 = 0.001*((4./3.)*np.pi*(r0)**3) # g
    Mt0 = 1.0 # g/m^3
    x0 = 1e-12 # g
    
    Nt0 = Mt0/m0 # Put in #/m^3 for model input
    mbar0 = Mt0/Nt0
    
    rtime = np.zeros((len(moms),len(sbin)))
    
    nD_func = lambda x: (Nt0/mbar0)*np.exp(-x/mbar0)

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
                                       habit_params = 'rain',
                                       kernel='Long',
                                       frag_dist = 'Straub',
                                       Ecol = 1.0, 
                                       Es = 0.6, 
                                       Eb = 1.0,
                                       init_method='analytical', 
                                       func_nD=nD_func,
                                       gam_norm=True,
                                       dist_var='mass',
                                       moments=moms[mm], 
                                       vt = vt,radar=True)
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
    
    lcolors[0] = 'k'
    
    normbin=False
    plot_init = False
    xscale='linear'
    yscale='log'
    distscale='linear'
    xlim = (0.,5.)
    ylim = (1e-6,None)
    x_axis='size'
    
    Ntot = np.zeros((len(sbin),len(moms)))
    Mtot = np.zeros((len(sbin),len(moms)))
    Dmtot = np.zeros((len(sbin),len(moms)))
    Rmtot = np.zeros((len(sbin),len(moms)))
    
    ZH = np.zeros((len(sbin),len(moms)))
    ZDR = np.zeros((len(sbin),len(moms)))
    KDP = np.zeros((len(sbin),len(moms)))
    RHOHV = np.zeros((len(sbin),len(moms)))
    
    
    # Now do plotting at 1 hr
    for mm in range(len(moms)):
        for ss in range(len(sbin)):
            
            smom_test = spectral_1d(load=os.path.join(cdir,'Output',outstr.format(moms[mm],sbin[ss])))
            
            
            print('sbin={} | mom={}'.format(sbin[ss],moms[mm]))
            print('Nt={:.3} | Dm={:.3} | WC={:.3} | R={:.3}'.format(smom_test.Ntot[0,-1],smom_test.Dmtot[0,-1],smom_test.Mtot[0,-1],smom_test.Rmtot[0,-1]))
            print('Z={:.3} | ZDR={:.3} | KDP={:.3} | RHOHV={:.3}'.format(smom_test.ZH[0,-1],smom_test.ZDR[0,-1],smom_test.KDP[0,-1],smom_test.RHOHV[0,-1]))
            print('------------')
            
            Ntot[ss,mm] = smom_test.Ntot[0,-1]
            Dmtot[ss,mm] = smom_test.Dmtot[0,-1]
            Mtot[ss,mm] = smom_test.Mtot[0,-1]
            Rmtot[ss,mm] = smom_test.Rmtot[0,-1]
            ZH[ss,mm] = smom_test.ZH[0,-1]
            ZDR[ss,mm] = smom_test.ZDR[0,-1]
            KDP[ss,mm] = smom_test.KDP[0,-1]
            RHOHV[ss,mm] = smom_test.RHOHV[0,-1]
            
            
            # if (mm==0) & (ss==0):     
            #     fig, ax = smom_test.plot_dists(normbin=normbin,plot_init=plot_init,linestyle=lstyle[mm],color=lcolors[ss],x_axis=x_axis,xscale=xscale,yscale=yscale,distscale=distscale,xlim=xlim,ylim=ylim)
            # else:
            #     smom_test.plot_dists(ax=ax,plot_init=plot_init,linestyle=lstyle[mm],color=lcolors[ss],normbin=normbin,x_axis=x_axis,xscale=xscale,yscale=yscale,distscale=distscale,xlim=xlim,ylim=ylim)
                
            if (mm==0) & (ss==0):     
                fig_mom, ax_mom = smom_test.plot_moments_radar(linestyle=lstyle[mm],color=lcolors[ss])
            else:
                smom_test.plot_moments_radar(ax=ax_mom,linestyle=lstyle[mm],color=lcolors[ss])
                
            del smom_test
    
    
    