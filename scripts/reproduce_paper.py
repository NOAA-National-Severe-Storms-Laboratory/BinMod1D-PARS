# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 13:44:58 2026

@author: edwin.dunnavan
"""

# scripts/reproduce_paper.py
import os
from binmod1d.spectral_model import spectral_1d
from binmod1d.distribution import rain_terminal_velocity
from binmod1d.plotting_functions import get_cmap_vars

import numpy as np

from time import perf_counter

from scipy.special import gamma

from matplotlib.lines import Line2D

import matplotlib.pyplot as plt

# Create output directory
os.makedirs("./Figures/", exist_ok=True)
    
def generate_figure_4():
    print("Generating Figure 4: Comparison with Analytical solutions...")
    
    sbin = 32
    bins = 14 * sbin
    dist_var = 'mass'
    
    s32_golovin = spectral_1d(sbin=sbin,tmax=60.*20.,bins=bins,dt=1,kernel='Golovin',gam_norm=True,x0=0.05,moments=2,dist_var=dist_var)
    print('Running Golovin solution')
    s32_golovin.run()
    s32_product = spectral_1d(sbin=sbin,tmax=60.*10.,bins=bins,dt=1,kernel='Product',gam_norm=True,x0=0.05,moments=2,dist_var=dist_var)
    print('Running Product solution')
    s32_product.run()
    s32_constant = spectral_1d(sbin=sbin,tmax=60.*20.,bins=bins,dt=1,kernel='Constant',gam_norm=True,x0=0.05,moments=2,dist_var=dist_var)
    print('Running Constant solution')
    s32_constant.run()

    # Plot dists from Figure 4
    fig, ax = plt.subplots(2,3,figsize=(16,8),sharey='row',sharex=True)
    
    s32_golovin.plot_dists(-1,ax=ax[:,0],plot_init=True,scott_solution=True)
    s32_product.plot_dists(-1,ax=ax[:,1],plot_init=True,scott_solution=True)
    s32_constant.plot_dists(-1,ax=ax[:,2],plot_init=True,scott_solution=True)

    ax[1,0].set_xlabel(r'$\log(m/\overline{m})$',fontsize=32)
    ax[1,1].set_xlabel(r'$\log(m/\overline{m})$',fontsize=32)
    ax[1,2].set_xlabel(r'$\log(m/\overline{m})$',fontsize=32)
    
    ax[0,0].set_ylabel('dN/dlog(m)',fontsize=32,usetex=True)
    ax[1,0].set_ylabel('dM/dlog(m)',fontsize=32,usetex=True)
    ax[0,1].set_ylabel('')
    ax[0,2].set_ylabel('')
    ax[1,1].set_ylabel('')
    ax[1,2].set_ylabel('')
    
    ax[0,0].set_yticks(np.arange(0.,1.8+0.5,0.5))
    ax[1,0].set_yticks(np.arange(0.,2.2+0.5,0.5))
    
    ax[0,0].set_title('Golovin (t=20 min.)',fontsize=36)
    ax[0,1].set_title('Product (t=10 min.)',fontsize=36)
    ax[0,2].set_title('Constant (t=20 min.)',fontsize=36)

    ax[0,0].set_ylim((0.,2.0))
    ax[1,0].set_ylim((0.,2.2))
    
    ax[1,0].set_xticks(np.arange(-2,4,1))
    ax[1,0].set_xlim((-1.2,2.5))

    # 1. Strip out the auto-generated legends from the plotting methods
    for axis in ax.flatten():
        if axis.get_legend() is not None:
            axis.get_legend().remove()

    # 2. Create the custom handles with clean names
    custom_handles = [
        Line2D([0], [0], color='k', linestyle=':', lw=2, label='Initial'),
        Line2D([0], [0], color='k', linestyle='-', lw=2, label='Numerical'),
        Line2D([0], [0], color='r', linestyle=':', lw=2, label='Analytical')
    ]

    # 3. Inject the legend specifically into the top-left panel (ax[0,0])
    ax[0,0].legend(handles=custom_handles, loc='upper right', fontsize=18, frameon=True)

    fig.tight_layout(w_pad=2.0, h_pad=0.5)
             
    objs = {} 
    objs['Figure4'] = {'fig':fig,'ax':ax}

    return objs
    
    
def generate_figure_5():
    print("Generating Figure 5: Box model Feingold comparison...")
    # Exact params from paper
    dist_var = 'mass'
    
    print('Running Breakup solution')
    s6_breakup = spectral_1d(sbin=6,bins=120,kernel='Constant',frag_dist='exp_mass',dist_var=dist_var,tmax=1800,gam_norm=True,mu0=0.,Eb=0.1,Es=0.,x0=0.0001)
    s6_breakup.run()

    s4_BC = spectral_1d(sbin=4,bins=80,kernel='Constant',frag_dist='exp_mass',dist_var=dist_var,gam_norm=True,mu0=0.,tmax=6000,Eb=1.0,Es=0.8,x0=0.0001)
    print('Running Coalescence/Breakup solution')
    s4_BC.run()

    # Plot dists from Figure 5
    fig, ax = plt.subplots(2,2,figsize=(16,10),sharey='row',sharex=True)
    
    s6_breakup.plot_dists(ax=ax[:,0],tind=-1,feingold_solution=True)
    s4_BC.plot_dists(ax=ax[:,1],tind=-1,feingold_solution=True)
    
    ax[0,0].set_ylabel('dN/dlog(m)',fontsize=32,usetex=True)
    ax[1,0].set_ylabel('dM/dlog(m)',fontsize=32,usetex=True)
    ax[0,1].set_ylabel('')
    ax[1,1].set_ylabel('')

    ax[1,0].set_xlabel(r'$\log(m/\overline{m})$',fontsize=32)
    ax[1,1].set_xlabel(r'$\log(m/\overline{m})$',fontsize=32)

    ax[0,0].set_ylim((0.,3.5))
    ax[1,0].set_ylim((0.,1.3))
    
    ax[1,0].set_xlim((-4,2))
    
    ax[1,0].set_xticks(np.arange(-4,3,1))
    
    ax[0,0].set_yticks(np.arange(0.,4,0.5))
    ax[1,0].set_yticks(np.arange(0.,1.5,0.25))
    
    # 1. Strip out the auto-generated legends from the plotting methods
    for axis in ax.flatten():
        if axis.get_legend() is not None:
            axis.get_legend().remove()
    
    # 2. Create the custom handles with clean names
    custom_handles = [
        Line2D([0], [0], color='k', linestyle=':', lw=2, label='Initial'),
        Line2D([0], [0], color='k', linestyle='-', lw=2, label='Numerical'),
        Line2D([0], [0], color='r', linestyle=':', lw=2, label='Analytical')
    ]
    
    ax[1,0].legend(handles=custom_handles, loc='upper left', fontsize=20, frameon=True)

    ax[0,0].set_title('Breakup (t=30 min.)',fontsize=32)
    ax[0,1].set_title('Coalescence-Breakup (t=100 min.)',fontsize=32)

    fig.tight_layout(w_pad=3.0, h_pad=0.5)

    objs = {} 
    objs['Figure5'] = {'fig':fig,'ax':ax}
    
    return objs

def generate_figure_6():
    print("Generating Figure 6: Steady State Rain coalescence and coalescence/breakup...")
    
    # Exact params from paper
    sbin = 3 
    bins = 160
    D1 = 0.001
    tmax = 0.
    Nt0 = 15. 
    Dm0 = 0.8 
    mu0 = 0.
    dz = 10. 
    ztop = 3000. 
    zbot = 0.
    habit_params=['rain']
    ptype='rain'
    kernel='Hydro'
    Ecol = 1.0 
    Es = 0.8
    dist_var = 'size'
    frag_dist = 'exp'
    Ecol=1.0
    Eb=0.005
    
    s3_SS = spectral_1d(sbin=sbin,
                        bins=bins,
                        D1=D1,
                        tmax=tmax,
                        Nt0=Nt0,
                        Dm0=Dm0,
                        mu0=mu0,
                        dz=dz,ztop=ztop,zbot=zbot,
                        habit_params=habit_params,ptype=ptype,
                        kernel=kernel,Ecol=Ecol,Es=Es,
                        radar=True,dist_var=dist_var)
    print('Running steady-state in height rain coalescence only...')
    s3_SS.run()
    
    s3_BC_SS = spectral_1d(sbin=sbin,
                        bins=bins,
                        D1=D1,
                        tmax=tmax,
                        Nt0=Nt0,
                        Dm0=Dm0,
                        mu0=mu0,
                        dz=dz,ztop=ztop,zbot=zbot,
                        habit_params=habit_params,ptype=ptype,
                        frag_dist=frag_dist,
                        kernel=kernel,Ecol=Ecol,Es=Es,Eb=Eb,
                        radar=True,dist_var=dist_var)
    print('Running rain steady-state in height coalescence/breakup...')
    s3_BC_SS.run()
    
    fig, ax = s3_SS.plot_dists_height(dz=1.5,fontsize=22)
    fig_moms, ax_moms = s3_SS.plot_moments_radar(fontsize=22)
    
    s3_BC_SS.plot_dists_height(dz=1.5,ax=ax,linestyle='--',fontsize=26)
    s3_BC_SS.plot_moments_radar(ax=ax_moms,linestyle='--',fontsize=26)
    
    ax[0].tick_params('both',labelsize=26)
    ax[1].tick_params('both',labelsize=26)
    ax[2].tick_params('both',labelsize=26)
    
    ax[0].set_xticks(np.arange(0,6,1))
    ax[1].set_xticks(np.arange(0,6,1))
    ax[2].set_xticks(np.arange(0,6,1))
    
    ax_moms[0,0].set_ylim((0.,3.))
    ax_moms[1,0].set_ylim((0.,3.))
    
    # Strip out any auto-generated legends from the plotting methods
    for axis in ax_moms.flatten():
        if axis.get_legend() is not None:
            axis.get_legend().remove()

    # Create the custom handles (Assuming default black color based on your methods)
    custom_handles = [
        Line2D([0], [0], color='k', linestyle='-', lw=2, label='CC only'),
        Line2D([0], [0], color='k', linestyle='--', lw=2, label='CC-BC')
    ]

    # Inject the legend specifically into the top-left panel (ax_moms[0,0])
    ax_moms[0,0].legend(handles=custom_handles, loc='upper right', fontsize=18, frameon=True)
    
    # =========================================================================
    # Modify xticks slightly
    ax_moms[0,0].set_xticks((0.,25.,50.,75.,100.))
    ax_moms[0,1].set_xticks((0.8,0.9,1.0,1.1,1.2))
    
    objs = {'Figure6':
            {'moments':{'fig':fig_moms,'ax':ax_moms},
            'dist_heights':{'fig':fig,'ax':ax}}}
        
    return objs

def generate_figure_7():
    print("Generating Figure 7: Steady State two habit snow aggregation and breakup...")
    # Exact params from paper
    sbin=3 
    bins=60
    D1 = 0.01 
    Nt0 = 50. 
    Dm0 = 1.0 
    mu0 = 3.0
    tmax = 0.
    dz=5.
    ztop=3000.
    zbot=0.
    kernel = 'Hydro'
    frag_dist='LGN'
    habit_params=['snow','fragments']
    ptype='snow'
    Ecol = 1.0
    Es = 0.6 
    Eb = 0.05
    dist_var = 'size'

    s3_snow_SS = spectral_1d(sbin=sbin,
                             bins=bins,
                             D1=D1,
                             tmax=tmax,
                             output_freq=1,
                             Nt0=Nt0,
                             Dm0=Dm0,
                             mu0=mu0,
                             dz=dz,
                             ztop=ztop,
                             zbot=zbot,
                             frag_dist=frag_dist,
                             habit_params=habit_params,
                             ptype=ptype,
                             kernel=kernel,
                             Ecol=Ecol,Es=Es,Eb=Eb,
                             radar=True,
                             dist_var=dist_var,
                             moments=2,dist_num=2,cc_dest=1,br_dest=2)
    
    s3_snow_SS.run()

    fig, ax = s3_snow_SS.plot_dists_height(dz=1.0,fontsize=22,plot_habits=True)
    fig_moms, ax_moms = s3_snow_SS.plot_moments_radar(fontsize=22,plot_habits=True)
    
    ax[0].tick_params('both',labelsize=24)
    ax[1].tick_params('both',labelsize=24)
    ax[2].tick_params('both',labelsize=24)
    ax[3].tick_params('both',labelsize=24)
    
    ax[0].set_xticks(np.arange(0,6,1))
    ax[1].set_xticks(np.arange(0,6,1))
    ax[2].set_xticks(np.arange(0,6,1))
    ax[3].set_xticks(np.arange(0,6,1))
    
    ax[0].set_yticks(10.**(np.arange(-5,7,2)))
    ax[1].set_yticks(10.**(np.arange(-5,7,2)))
    ax[2].set_yticks(10.**(np.arange(-5,7,2)))
    ax[3].set_yticks(10.**(np.arange(-5,7,2)))
    
    handles, labels = ax_moms[0,0].get_legend_handles_labels()
    
    ax_moms[0,1].legend(handles=handles,labels=labels,loc=(0.08,0.1), fontsize=18)
    
    ax_moms[0,0].get_legend().remove()
    
    ax_moms[0,0].set_ylim((0.,3.))
    
    objs = {'Figure7':
            {'moments':{'fig':fig_moms,'ax':ax_moms},
            'dist_heights':{'fig':fig,'ax':ax}}}

    return objs
    
def generate_figure_8():
    print("Generating Figure 8: 1D Time/Height Fallout Reflectivity...")
    # Exact params from paper
    sbin = 2 
    bins = 60
    D1 = 0.01
    tmax = 1500.
    output_freq = 1. 
    dt = 2.0 
    Nt0 = 15. 
    Dm0 = 0.8 
    mu0 = 0.
    ztop = 3000. 
    zbot = 0. 
    dz = 10.
    boundary = None
    Ecol = 1.0
    Es = 0.25 
    dist_var = 'size'
    moments = 2 
    rk_order = 4
    
    s2_rain = spectral_1d(sbin=sbin,
                          bins=bins,
                          D1=D1,
                          tmax=tmax,
                          output_freq=output_freq,
                          dt=dt,
                          Nt0=Nt0,Dm0=Dm0,mu0=mu0,
                          dz=dz,ztop=ztop,zbot=zbot,
                          boundary=boundary,
                          habit_list=['rain'],ptype='rain',
                          kernel='Hydro',
                          Ecol=Ecol,Es=Es,
                          radar=True,dist_var=dist_var,
                          moments=moments,rk_order=rk_order)
    s2_rain.run()
    fig_Z, ax_Z = s2_rain.plot_time_height()
    
    objs = {'Figure8':
            {'fig':fig_Z,'ax':ax_Z}}
    
    return objs

def generate_figure_9_10(fig9=True,fig10=True):
    sbin = 2 
    bins = 60 
    D1 = 0.001 
    Nt0 = 50. 
    Dm0 = 1.25 
    mu0 = 0. 
    dt = 2.0 
    tmax = 5400.
    
    #tmax = 60.
    output_freq = 6.
    ztop = 3000. 
    zbot = 0. 
    dz = 20.
    boundary = 'fixed'
    habit_params = ['snow','fragments']
    ptype = 'snow'
    kernel = 'Hydro'
    Ecol = 1.0 
    Es = 0.6 
    Eb = 0.05
    frag_dist = 'LGN'
    dist_num = 2
    cc_dest = 1
    br_dest = 2
    dist_var = 'size'
    
    s2_snow_breakup_2cat = spectral_1d(sbin=sbin,
                                       bins=bins,
                                       D1=D1,
                                       tmax=tmax,
                                       output_freq=output_freq,
                                       dt=dt,
                                       Nt0=Nt0,
                                       Dm0=Dm0,
                                       mu0=mu0,
                                       dz=dz,
                                       ztop=ztop,
                                       zbot=zbot,
                                       boundary=boundary,
                                       habit_params=habit_params,
                                       ptype=ptype,
                                       kernel=kernel,
                                       Ecol=Ecol,Es=Es,Eb=Eb,
                                       radar=True,frag_dist=frag_dist,
                                       dist_num=dist_num,cc_dest=cc_dest,br_dest=br_dest,
                                       dist_var=dist_var,moments=2,rk_order=1)
    
    s2_snow_breakup_2cat.run()
    
    if fig9:
        print("Generating Figure 9: 1D Time/Height snow examples...")
        
        s2_snow = spectral_1d(sbin=sbin,
                              bins=bins,
                              D1=D1,
                              tmax=tmax,
                              output_freq=output_freq,
                              dt=dt,
                              Nt0=Nt0,
                              Dm0=Dm0,
                              mu0=mu0,
                              dz=dz,
                              ztop=ztop,
                              zbot=zbot,
                              boundary=boundary,
                              habit_params='snow',
                              ptype='snow',
                              kernel=kernel,
                              Ecol=Ecol,Es=0.35,
                              radar=True,
                              dist_var=dist_var,moments=2,rk_order=1)
        s2_snow.run()
        
        # Consolidate all figures into a single figure (i.e., Figure 9)
        fig_9, ax_9 = plt.subplots(3,2,figsize=(12,10),layout='constrained',sharex=True,sharey=True)
        
        time_labels_orig = np.arange(0.,1.5+0.25,0.25) 
        time_ticks = 3600.*time_labels_orig
        
        height_labels_orig = np.arange(0.,3+0.5,0.5)
        
        # Wrap the numbers in LaTeX math mode!
        time_labels = [f"${t}$" for t in time_labels_orig]
        height_labels = [f'${hh}$' for hh in height_labels_orig]
        
        ax_9[0,0].set_xticks(time_ticks)
        ax_9[0,0].set_xticklabels(time_labels)
        
        ax_9[2,0].set_xlabel('Time (hours)',fontsize=32)
        ax_9[2,1].set_xlabel('Time (hours)',fontsize=32)
        
        s2_snow.plot_time_height(ax=ax_9[0,0])
        s2_snow.plot_time_height(ax=ax_9[1,0],var='ZDR')
        s2_snow.plot_time_height(ax=ax_9[2,0],var='KDP')
        
        s2_snow_breakup_2cat.plot_time_height(ax=ax_9[0,1],var='Z')
        s2_snow_breakup_2cat.plot_time_height(ax=ax_9[1,1],var='ZDR')
        s2_snow_breakup_2cat.plot_time_height(ax=ax_9[2,1],var='KDP')
        
        var_names = ['Z', 'ZDR', 'KDP']
        for row, var in enumerate(var_names):
            # Extract the 'cax' mappable from the axis using the collections trick
            cax = ax_9[row, 1].collections[0]
            
            # Get the exact labels and ticks for this variable
            cmap, levels, levels_ticks, clabel, labelpad, fontsize, slabel = get_cmap_vars(var)
            
            if var=='Z':
                #levels =np.arange(-10,80,10)
                levels_ticks=np.arange(-10,80,10)
                fontsize=26
                
            if var=='ZDR':
                fontsize=26
                    
            # Map the colorbar to the entire row (ax=ax_9[row, :])
            cbar = fig_9.colorbar(cax, ax=ax_9[row, :], ticks=levels_ticks, pad=0.01, aspect=15)
            
            # Format the colorbar to match your internal methods
            cbar.ax.tick_params(labelsize=14)
            cbar.ax.set_yticklabels(levels_ticks, usetex=True)
            cbar.ax.minorticks_off()
            cbar.set_label(clabel, usetex=True, rotation=270, fontsize=fontsize, labelpad=labelpad + 5)
        
        ax_9[2, 0].set_xticks(time_ticks)
        ax_9[2, 0].set_xticklabels(time_labels)
        ax_9[0, 0].set_yticks(height_labels_orig)
        ax_9[0, 0].set_yticklabels(height_labels)
        
        ax_9[2, 0].set_xlabel('Time (hours)', fontsize=24, usetex=True)
        ax_9[2, 1].set_xlabel('Time (hours)', fontsize=24, usetex=True)
        
        # 4. Format the Y-axis and Titles
        ax_9[0, 0].set_title('Aggregation Only', fontsize=24, usetex=True)
        ax_9[0, 1].set_title('Aggregation + Breakup', fontsize=24, usetex=True)
        
        for row in range(3):
            ax_9[row, 0].set_ylabel('Height (km)', fontsize=26, usetex=True)
            ax_9[row, 0].tick_params(axis='both', labelsize=22)
            ax_9[row, 1].tick_params(axis='both', labelsize=22)
            

    if fig10:
        lcolors = ['#000000', '#1f77b4', '#ff7f0e']
        tlabels = ['total','snow','fragments']
        slabels = ['Full 1D 1.5 h', 'Steady-state']
        lstyle = ['-','--']
         
        print("Generating Figure 10: 1D Time/Height snow example moments with Steady-State...")
        s2_snow_breakup_2cat_SS = spectral_1d(sbin=sbin,
                                              bins=bins,
                                              D1=D1,
                                              tmax=0.,
                                              output_freq=1.,
                                              Nt0=Nt0,
                                              Dm0=Dm0,
                                              mu0=mu0,
                                              dz=dz,
                                              ztop=ztop,
                                              zbot=zbot,
                                              boundary=boundary,
                                              habit_params=habit_params,
                                              ptype=ptype,kernel=kernel,
                                              Ecol=Ecol,Es=Es,Eb=Eb,
                                              radar=True,frag_dist=frag_dist,
                                              dist_num=dist_num,cc_dest=cc_dest,br_dest=br_dest,
                                              dist_var=dist_var,moments=2,rk_order=1)
        s2_snow_breakup_2cat_SS.run()
        
        fig_moms, ax_moms = s2_snow_breakup_2cat.plot_moments_radar(tind=-1,plot_habits=True)
        
        for axis in ax_moms.flatten():
            axis.set_prop_cycle(None)
        
        s2_snow_breakup_2cat_SS.plot_moments_radar(ax=ax_moms,plot_habits=True,linestyle='--')
    
        ## Redo the legends for the paper
        # Create the Custom Handles
        # Color handles for sbin (using solid lines)
        color_handles = [Line2D([0], [0], color=lcolors[i], lw=2, label=tlabels[i]) for i in range(len(tlabels))]

        # Line style handles for moments (using black color)
        style_handles = [Line2D([0], [0], color='k', linestyle=lstyle[i], lw=2, label=slabels[i]) for i in range(len(lstyle))]

        # Apply to Figure 10 (Moments/Radar timeseries)
        # First, strip out all the auto-generated legends from the subplots
        for axis in ax_moms.flatten():
            if axis.get_legend() is not None:
                axis.get_legend().remove()

        # Apply to Figure 12 (Distribution profiles)
        ax_moms[0,1].legend(handles=color_handles, loc=(0.1,0.1), fontsize=16, frameon=True)
        ax_moms[1,0].legend(handles=style_handles, loc='lower left', fontsize=16, frameon=True)

        ax_moms[0,0].set_ylim((0.,3.))
        ax_moms[0,0].set_yticks(np.arange(0.,3.5,0.5))

    objs = {} 
    if fig9:
        objs['Figure9'] = {'fig':fig_9,'ax':ax_9}
        
    if fig10:
        objs['Figure10'] = {'fig':fig_moms,'ax':ax_moms}
    
    return objs

    #!!! Note: For Figure 11 and 12, the default behavior is to plot using
    # data already generated and written as netcdf files 
    # (./scripts/Figures/Figure_11_12/Long_Straub_convg_test_box_[1,2]_s[1,2,4,8,16]_paper.nc).
    # This is because the convergence and timing tests shown in Figure 12 take some time to generate
    # (see Table 1 in the manuscript for more details on timing tests). If
    # users would like to run the model and re-generate the data used in Figure 12,
    # then they can change the "run_model" input parameter to True.
    
    # !!!CAUTION!!! Reading in the netcdf files for the s=16 model runs takes up quite a bit of memory.
    # Future versions of BinMod1D will utilize the xarray python package to better handle
    # netcdf I/O to cut down on this large memory usage.
    
def generate_figure_11_12(fig11=True,fig12=True,run_model=False):
    
    # Get current directory
    cdir = os.getcwd()
    # Box model
    
    # Set up rain velocity
    vt = lambda d: rain_terminal_velocity(d)
    
    # Output netcdf files for box model tests
    if run_model==False:
        outstr = 'Long_Straub_convg_test_box_{}mom_s{}_paper.nc'
    else:
        outstr = 'Long_Straub_convg_test_box_{}mom_s{}.nc'
    
    output_freq = 10

    bin1 = 45 # number of bins for sbin=1
    
    sbin = [16,8,4,2,1] # resolution parameter
    
    bins = bin1*np.array(sbin) # Keeps xmin and xmax consistent for each resolution.   
    
    mu0 = 0
    moms = [2,1]
    
    tmax = 3600.*3 # s 
    dt = 1.0 # s
    r0 = 9.3e-3 # mm
    m0 = 0.001*((4./3.)*np.pi*(r0)**3) # g
    Mt0 = 1.0 # g/m^3
    x0 = 1e-12 # g
    
    Nt0 = Mt0/m0 # Put in #/m^3 for model input
    mbar0 = Mt0/Nt0
    
    rtime = np.zeros((len(moms),len(sbin)))
    
    # It's easier to specify the analytical function in order
    # for the units to be correct.
    nD_func = lambda x: (Nt0/mbar0)*np.exp(-x/mbar0)

    if run_model:

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
                smom.write_netcdf(os.path.join(cdir,'Output/',outstr.format(moms[mm],sbin[ss])))
        
                del smom
    
    lstyle = ['-',':']
    
    # Colorblind-friendly colors
    lcolors = ['#000000', '#56B4E9', '#D55E00', '#009E73', '#CC79A7']
    
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
            
            # Figure 11
            if fig11:
                print("Generating Figure 11: Long-Straub cloud-rain box model transition convergence and timing moments.")
                if (mm==0) & (ss==0):     
                    fig_mom, ax_mom = smom_test.plot_moments_radar(linestyle=lstyle[mm],color=lcolors[ss])
                else:
                    smom_test.plot_moments_radar(ax=ax_mom,linestyle=lstyle[mm],color=lcolors[ss])
                
            # Figure 12
            if fig12:
                print("Generating Figure 12: Long-Straub cloud-rain box model transition convergence and timing dists.")
                if (mm==0) & (ss==0):     
                    fig, ax = smom_test.plot_dists(normbin=normbin,plot_init=plot_init,linestyle=lstyle[mm],color=lcolors[ss],x_axis=x_axis,xscale=xscale,yscale=yscale,distscale=distscale,xlim=xlim,ylim=ylim)
                else:
                    smom_test.plot_dists(ax=ax,plot_init=plot_init,linestyle=lstyle[mm],color=lcolors[ss],normbin=normbin,x_axis=x_axis,xscale=xscale,yscale=yscale,distscale=distscale,xlim=xlim,ylim=ylim)
                    
            del smom_test

    ax[0].grid()
    ax[1].grid()
    
    ax_mom[0,0].grid()
    ax_mom[0,1].grid()
    ax_mom[0,2].grid()
    ax_mom[0,3].grid()

    ax_mom[1,0].grid()
    ax_mom[1,1].grid()
    ax_mom[1,2].grid()
    ax_mom[1,3].grid()
    
    ax_mom[0,0].set_xlim((0.,tmax))
    
    time_labels = np.arange(0,3+1,1)
    time_ticks =  time_labels*3600.
    
    ax_mom[1,0].set_xticks(time_ticks)
    ax_mom[1,0].set_xticklabels(time_labels)
    
    ax_mom[0,0].set_yscale('log')
    ax_mom[1,3].set_xlabel('Time (hours)')
    ax_mom[1,0].set_xlabel('Time (hours)')
    ax_mom[1,1].set_xlabel('Time (hours)')
    ax_mom[1,2].set_xlabel('Time (hours)')
    
    ax_mom[0,1].set_ylim((0.,1.7))
    ax_mom[0,3].set_ylim((0.,20.))
    ax_mom[1,1].set_ylim((-0.1,4.))
    ax_mom[1,2].set_ylim((-0.005,0.3))
    
    ax_mom[0, 0].axes.tick_params("both", labelsize=22)
    ax_mom[0, 1].axes.tick_params("both", labelsize=22)
    ax_mom[0, 2].axes.tick_params("both", labelsize=22)
    ax_mom[0, 3].axes.tick_params("both", labelsize=22)
    ax_mom[1, 0].axes.tick_params("both", labelsize=22)
    ax_mom[1, 1].axes.tick_params("both", labelsize=22)
    ax_mom[1, 2].axes.tick_params("both", labelsize=22)
    ax_mom[1, 3].axes.tick_params("both", labelsize=22)
    
    ## Redo the legends for the paper
    # Create the Custom Handles
    # Color handles for sbin (using solid lines)
    sbin_labels = ['s$_{\mathrm{bin}}$='+str(ss) for ss in sbin]
    color_handles = [Line2D([0], [0], color=lcolors[i], lw=2, label=sbin_labels[i]) for i in range(len(sbin))]

    # Line style handles for moments (using black color)
    mom_labels = ['2-Moment', '1-Moment']
    style_handles = [Line2D([0], [0], color='k', linestyle=lstyle[i], lw=2, label=mom_labels[i]) for i in range(len(moms))]

    # Apply to Figure 12 (Distribution profiles)
    ax[1].legend(handles=color_handles, loc='lower left', fontsize=16, frameon=True)
    ax[0].legend(handles=style_handles, loc='upper right', fontsize=16, frameon=True)

    # Apply to Figure 11 (Moments/Radar timeseries)
    # First, strip out all the auto-generated legends from the subplots
    for axis in ax_mom.flatten():
        if axis.get_legend() is not None:
            axis.get_legend().remove()

    # Place the new condensed legends in the far-right column subplots
    # (You can change ax_mom[0, 3] to any other subplot if you prefer)
    ax_mom[1, 0].legend(handles=color_handles, loc='lower right', fontsize=18, frameon=True)
    ax_mom[0, 2].legend(handles=style_handles, loc=(0.125,0.125), fontsize=22, frameon=True)
    
    objs = {} 
    
    # Resize figure so that it looks nicer for the paper.
    fig_mom.set_size_inches(17,8)
    fig_mom.tight_layout()
    
    objs['Figure11'] = {'fig':fig_mom,'ax':ax_mom} 
    objs['Figure12'] = {'fig':fig,'ax':ax}    
    
    return objs
    
def generate_figure_13():
    print("Generating Figure 13: Radar retrieval application...")

    # Set up retrieval distributions
    sbin = 6
    bins = 180
    wavl = 110.8
    D1 = 0.001
    Nt0 = 50. 
    Dm0 = 1.25
    mu0 = 0.
    Es = 0.35
    # Domain variables
    ztop = 3000. 
    zbot= 0.
    dz = 10.
    # Set up snow dictionary with habit parameters
    snow_params =  {'snow': {'arho': 0.178,
                        'brho': 1.0,
                        'av': 0.81,
                        'bv': 0.15,
                        'ar': 0.65,
                        'br': 0.0,
                        'sig': 0.}}
  
    
    model = spectral_1d(sbin=sbin,bins=bins,D1=D1,tmax=0.,output_freq=1.,Nt0=Nt0,Dm0=Dm0,mu0=mu0,dz=dz,ztop=ztop,zbot=zbot,habit_params=snow_params,ptype='snow',kernel='Hydro',Ecol=1.0,Es=Es,radar=True,dist_var='size',moments=2,wavl=wavl)
    model.run()
    
    # Now, use B20 and RZ19 radar retrieval relations 
    am = model.habit_dict['snow']['am']
    d = model.d[0,:] # Just need first (and only) category
    z = model.z.copy()
    ZH  = model.ZH.copy()
    ZDR = model.ZDR.copy()
    KDP = model.KDP.copy()
    
    nD = lambda D, N, Dmv, mu: 1000.*N * (1./gamma(mu+1))* (mu+4)**(mu+1) * (1./Dmv) * (D/Dmv)**(mu) * np.exp(- (mu+4)*D/Dmv)
    
    Z_lin = 10.**(ZH/10.)
    Zdr_lin = 10.**(ZDR/10.)
    Zv_lin = Z_lin/Zdr_lin
    
    Zdp = Z_lin-Zv_lin

    # B20
    #B20_color = '#1f77b4'
    #B20_color = '#EE5396'
    B20_color = '#e41a1c'
    Nt  = 0.001*2.93e6*KDP**(4./3.)*Z_lin**(-1./3.)
    Dm  = 0.15*KDP**(-1./3.)*Z_lin**(1./3)
    IWC = 0.77*KDP**0.67*Z_lin**0.33
    R   = 1.62*KDP**0.62 * Z_lin**0.38
    # RZ19
    #RZ19_color = '#ff7f0e'
    RZ19_color = '#377eb8'
    #RZ19_color = '#1f77b4'
    Dm_rz = (0.54/(np.sqrt(110.8)))*(0.178**(-0.5))*(4./(np.sqrt(3.*2.)))*KDP**(-0.5)*Zdp**(0.5)
    Nt_rz = 0.001* 53.8*(110.8**2) * ((3.*2.)/(4.*1.))*KDP**2*Zdp**(-2)*Z_lin
    IWC_rz = 8e-3 * (110.8)*(2./4.) * KDP*Zdp**(-1)*Z_lin
    R_rz = 1000.*3.6*am*0.9*Nt_rz*Dm_rz**(2.+0.15)*(4)**(-(2+0.15))*gamma(2+0.15+1)/gamma(0+1)

    fig, ax = model.plot_moments_radar()
    
    line = ax[0,0].get_children()[0]
    line.set_label('BinMod1D')
    
    ax[0,0].plot(Nt,z/1000.,B20_color,label='B20')
    ax[0,0].plot(Nt_rz,z/1000.,RZ19_color,label='RZ19')
    
    ax[0,0].legend(loc='lower right',fontsize=18)
    
    ax[0,1].plot(Dm,z/1000.,B20_color)
    ax[0,1].plot(Dm_rz,z/1000.,RZ19_color)
    
    ax[0,2].plot(IWC,z/1000.,B20_color)
    ax[0,2].plot(IWC_rz,z/1000.,RZ19_color)
    
    ax[0,3].plot(R_rz,z/1000.,RZ19_color)
    ax[0,3].plot(R,z/1000.,B20_color)
    
    ax[0,0].set_xlim((0.,55.))
    ax[0,1].set_xlim((0.,6.))
    ax[0,2].set_xlim((0.001,1.25))
    ax[0,3].set_xlim((0.,4.5))
    
    ax[0,0].set_ylim((0.,3.))
    
    fig_p, ax_p = model.plot_dists_height(dz=1.5)
    
    nD_petar_3km = nD(d,Nt[0],Dm[0],0)
    nD_petar_1_5km = nD(d,Nt[150],Dm[150],0)
    nD_petar_0km = nD(d,Nt[-1],Dm[-1],0)
    
    nD_rz_3km = nD(d,Nt_rz[0],Dm_rz[0],0)
    nD_rz_1_5km = nD(d,Nt_rz[150],Dm_rz[150],0)
    nD_rz_0km = nD(d,Nt_rz[-1],Dm_rz[-1],0)
    
    ax_p[0].plot(d,nD_petar_3km,B20_color)
    ax_p[1].plot(d,nD_petar_1_5km,B20_color)
    ax_p[2].plot(d,nD_petar_0km,B20_color)
    
    ax_p[0].plot(d,nD_rz_3km,RZ19_color)
    ax_p[1].plot(d,nD_rz_1_5km,RZ19_color)
    ax_p[2].plot(d,nD_rz_0km,RZ19_color)
    
    ax_p[0].tick_params('both',labelsize=24)
    ax_p[1].tick_params('both',labelsize=24)
    ax_p[2].tick_params('both',labelsize=24)
    
    #ax[0,0].tick_params('both',labelsize=24)
    #ax[0,1].tick_params('both',labelsize=24)
    #ax[0,2].tick_params('both',labelsize=24)
    #ax[0,3].tick_params('both',labelsize=24)
    #ax[1,0].tick_params('both',labelsize=24)
    #ax[1,1].tick_params('both',labelsize=24)
    #ax[1,2].tick_params('both',labelsize=24)
    #ax[1,3].tick_params('both',labelsize=24)
    
    # Set x and y limits so it's easier to see the distributions
    ax_p[2].set_ylim((1e-3,5e5))
    ax_p[2].set_xlim((0.,20.))
    
    ax_p[0].set_yticks(10.**(np.arange(-5,7,2)))
    ax_p[1].set_yticks(10.**(np.arange(-5,7,2)))
    ax_p[2].set_yticks(10.**(np.arange(-5,7,2)))
    
    objs = {} 
    
    objs['Figure13'] = {} 
    
    objs['Figure13']['moments'] = {'fig':fig,'ax':ax}
    objs['Figure13']['dists_height'] = {'fig':fig_p,'ax':ax_p}               
    
            
    return objs
            

def get_ind_list(figs2gen):
    
    if isinstance(figs2gen,int):
        gen_list = [figs2gen]
        
        gen_array = np.array(gen_list)
        
        if (gen_array<4).any() | (gen_array>13).any():  
            ValueError('Please specify an integer (4-13).')
            
    elif isinstance(figs2gen,list):
        result = all(isinstance(x, int) for x in figs2gen)
        
        if not result:
            ValueError('Please ensure that the list elements are all integers')
            
        gen_list = figs2gen
        
    else: # If 'all' or anything else, then just generate all figures from the paper.
        gen_list = [4,5,6,7,8,9,10,11,12,13]
            
    
    return gen_list

def generate_figures(gen_list):
    
    ind_list = get_ind_list(gen_list)
    
    # Create a master dictionary to hold all returned figures and axes
    all_figures = {}
    
    # Flag to prevent running the expensive 11/12 function twice
    ran_9_10 = False
    ran_11_12 = False

    for ind in ind_list: 
        
        ind_int = int(ind) # ensure that it is an integer!
        
        if ind_int in [11, 12]:
            if not ran_11_12:
                objs = generate_figure_11_12()
                all_figures.update(objs)
                ran_11_12 = True
        elif ind_int in [9,10]:
            if not ran_9_10:
                objs = generate_figure_9_10()
                all_figures.update(objs)
                ran_9_10 = True
        else:
            # Use globals() to dynamically call the function and capture the return
            func_name = f"generate_figure_{ind_int}"
            
            if func_name in globals():
                # Call the function and catch the dictionary
                objs = globals()[func_name]()
                
                # Add the returned objects to our master dictionary
                if objs is not None:
                    all_figures.update(objs)
            else:
                print(f"Warning: {func_name} is not defined yet.")

    return all_figures
    

if __name__ == "__main__":
    
    #figs2gen = 'all' # If users want to reproduce all figures 4-13.
    figs2gen = [13]
    # figs2gen = [4,5,6] # Users can use a list to specify individual figures
                         # they would like to reproduce.
    # figs2gen = 4 # Otherwise, users can specify an individual integer 
                   # corresponding to the figure number (4-13) that they would 
                   # like to reproduce
    
    output_dict = generate_figures(figs2gen)

        
        
    
    
