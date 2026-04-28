# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 12:55:59 2026

@author: edwin.dunnavan
"""
from binmod1d.spectral_model import spectral_1d

import io
import pstats
import cProfile

if __name__ == '__main__':
    
    #s8_breakup_par = spectral_1d(kernel='Constant',gam_norm=True,mu0=0.,sbin=8,bins=200,Eb=0.1,Es=0.,x0=0.00001,parallel=True,n_jobs=4)

    s8_breakup_par = spectral_1d(sbin=2,bins=60,D1=0.001,tmax=3600.,
                                 output_freq=6.,dt=2.0,Nt0=50.,Dm0=1.25,
                                 mu0=0.,dz=20.0,ztop=3000.,zbot=2000.,
                                 boundary='fixed',habit_list=['snow','fragments'],
                                 ptype='snow',kernel='Hydro',Ecol=0.5,Es=0.6,
                                 Eb=0.002,radar=True,dist_num=2,cc_dest=1,br_dest=2,
                                 dist_var='size',moments=2,parallel=False,n_jobs=-1,
                                 rk_order=1,progress=True)

    # 1. Create a Profile instance
    pr = cProfile.Profile()
    
    # 2. Enable profiling, run code, and disable
    pr.enable()
    s8_breakup_par.run()
    pr.disable()
    s = io.StringIO()
    sort_by = pstats.SortKey.TIME
    ps = pstats.Stats(pr,stream=s).sort_stats(sort_by)
    ps.print_stats()
    profile_output_variable = s.getvalue()
    print(profile_output_variable)
    with open('pr_breakup_1hr_progress_2mom_full_NUMBA_NEW_par.txt','w+') as f:
        f.write(profile_output_variable)
    
    
    
    
