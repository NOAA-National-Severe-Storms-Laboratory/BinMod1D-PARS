# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 09:06:50 2025

@author: edwin.dunnavan
"""

import numpy as np

def habits():
    
    habits = {}
    
    habits['rain'] = {'arho':1.0,
                      'brho':0., 
                      'av':3.78,
                      'bv':0.67, 
                      'ar':1.0,
                      'br':0., 
                      'sig':10.}
    
    habits['rain']['am'] =  0.001*(np.pi/6.)*habits['rain']['arho'] 
    habits['rain']['bm'] =  3.-habits['rain']['brho']
    
    habits['snow'] = {'arho':0.2,
                      'brho':1.0, 
                      'av':0.8,
                      'bv':0.14, 
                      'ar':0.6,
                      'br':0., 
                      'sig':0.}
    
    habits['snow']['am'] =  0.001*(np.pi/6.)*habits['snow']['arho'] 
    habits['snow']['bm'] =  3.-habits['snow']['brho']
    
    
    habits['fragments'] = {'arho':0.6,
                      'brho':0.0, 
                      'av':0.8,
                      'bv':0.14, 
                      'ar':0.8,
                      'br':0., 
                      'sig':20.}
    
    habits['fragments']['am'] =  0.001*(np.pi/6.)*habits['fragments']['arho'] 
    habits['fragments']['bm'] =  3.-habits['fragments']['brho']
    
    return habits

def fragments(dist='exp'):
    
    if dist=='exp':
        fragments = {'dist':dist,
                     'lamf':10., 
                     'Dmf':0.25, 
                     'muf':0.}
        
    elif dist=='gamma':
        
        fragments = {'dist':dist,
                     'lamf':10.,
                     'Dmf':0.25, 
                     'muf':3.}
        
    elif dist=='LGN':
        
        fragments = {'dist':dist,
                     'lamf':10.,
                     'Df_med':0.55, 
                     'Df_mode':0.5}
        
    return fragments
