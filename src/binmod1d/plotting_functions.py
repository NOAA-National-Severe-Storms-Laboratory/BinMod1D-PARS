# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 10:50:41 2025

@author: edwin.dunnavan
"""
import numpy as np
#from cmweather import cm as cmp

import matplotlib as mpl
from matplotlib.colors import ListedColormap

def get_cmap_vars(varname):
    '''
    

    Parameters
    ----------
    varname : str
        Variable name.

    Returns
    -------
    cmap : matplotlib.color colorbar
        colorbar handle.
    levels : array or list
        Array or list of levels for contours.
    levels_ticks : array or list
        Array or list of ticks for levels.
    clabel : str
        Colorbar label.
    labelpad : int
        matplotlib colorbar labelpad parameter.
    fontsize : int
        fontsize for colorbar title.
    slabel : str
        Shorthand label for colorbar.

    '''
    # Grab NWS colormaps
    NWS_REF_COLORS, THEODORE16_COLORS, NWS_CC_COLORS, RRATE11_COLORS = get_cm_cmaps()
    
    NWSRef = ListedColormap(NWS_REF_COLORS,name='NWSRef')
    Theodore16 = ListedColormap(THEODORE16_COLORS,name='Theodore16')
    NWS_CC = ListedColormap(NWS_CC_COLORS,name='NWS_CC')
    RRate11 = ListedColormap(RRATE11_COLORS,name='RRate11')
    
    if varname == 'Z':
        #cmap = cmp.NWSRef
        cmap = NWSRef
        levels = np.arange(-10.,80,1.)
        levels_ticks = np.arange(-10,80,5)
        clabel = r'Reflectivity [dBZ]'
        slabel = r'Z [dBZ]'
        labelpad = 45
        fontsize=32

    if varname == 'ZDR':
        
        #cmap = cmp.NWSRef
        cmap = NWSRef
        #levels = [-1.,0.,0.2,0.4,0.6,0.8,1.0,1.5,2.0,2.5,3.0,4.,6.]
        levels = np.arange(-1.,6.05,0.05)
        #levels_ticks = [-1.,0.,0.2,0.4,0.6,0.8,1.0,1.5,2.0,2.5,3.0,4.,6.]
        levels_ticks = np.arange(-1.,7.,1.)
        clabel =r'$\mathrm{Z}_{\mathrm{DR}}$ [dB]'
        slabel =r'$\mathrm{Z}_{\mathrm{DR}}$ [dB]'
        labelpad = 45
        fontsize=32

    if varname == 'KDP':
        #cmap = cmp.Theodore16
        cmap = Theodore16
        levels =[-0.1,0.,0.025,0.05,0.1,0.15,0.2,0.3,0.4,0.6,0.8,1.,1.25,1.5,1.75,2.0,3.0] # X band?
        levels_ticks = levels.copy()
        clabel = r'$\mathrm{K}_{\mathrm{dp}}$ [deg/km]'
        slabel = r'$\mathrm{K}_{\mathrm{dp}}$ [deg/km]'
        labelpad=35
        fontsize=32
         
    if varname == 'RHOHV':
        #cmap = cmp.NWS_CC
        cmap = NWS_CC
        
        #levels = [0.8,0.9,0.92,0.94,0.96,0.97,0.975,0.9825,0.985,0.9875,0.99,0.995,1.]
        #levels_ticks = levels.copy()
        
        levels = np.arange(0.8,1.001,0.001)
        levels_ticks = [0.8,0.825,0.85,0.875,0.9,0.925,0.95,0.975,1.0]
        
        clabel = r'Correlation Coefficient'
        slabel = r'$\rho_{\mathrm{hv}}$'
        labelpad = 30
        fontsize=32
        
    if varname == 'R':
        #cmap = cmp.RRate11
        cmap = RRate11
        
        levels = [0.,0.001,0.01,0.1,1.,5.,10.,15,20.,25.,50,100,150,200.]
        levels_ticks = levels.copy()
        clabel = r'Precip. Rate (mm/hr)'
        slabel = r'R (mm/hr)'
        labelpad = 30
        fontsize=32
        
    if varname == 'Nt':
        cmap = mpl.cm.terrain
        levels = [0.,0.001,0.01,0.1,1.,2.5,5.,7.5,10.,12.5,15,17.50,20.,22.5,25.,50,100,150,200.]
        levels_ticks = levels.copy()
        clabel = r'Number Concentration (1/L)'
        slabel = r'$N_{t}$ (1/L)'
        labelpad = 30
        fontsize=22
        
    if varname == 'Dm':
        cmap = mpl.cm.terrain
        #levels = [0.1,0.25,0.5,0.75,1.,2., 3., 4.,5.,10.,15,20.]
        
        #levels = 2.**(np.arange(-3,6,1))
        
        levels = 2.**(np.arange(-3,5.25,0.25))
        
       # levels_ticks = levels.copy()
        
        levels_ticks =  2.**(np.arange(-3,6,1))
        
        
        clabel = r'Mean Volume Diameter (mm)'
        slabel = r'$D_{0}$ (mm)'
        labelpad = 30
        fontsize=22
        
    if varname == 'WC':
        #cmap = cmp.NWSRef
        cmap = NWSRef
        levels = [0.,0.001,0.01,0.05,0.1,0.25,0.5,0.75,1.,2.5,5.,10.]
        levels_ticks = levels.copy()
        clabel = r'Water Content (g/cm$^{3}$)'
        slabel = r'WC (g/cm$^{3}$)'
        labelpad = 30
        fontsize=22
        
    return cmap, levels, levels_ticks, clabel, labelpad, fontsize, slabel


def get_cm_cmaps():
    '''
    Generate colormap lists of NWS colormaps from cmweather python package

    Returns
    -------
    NWS_REF_COLORS : List
        NWS Ref colormap list.
    THEODORE16_COLORS : List
        THEODORE16 colormap list.
    NWS_CC_COLORS : TYPE
        NWS CC colormap list.
    RRATE11_COLORS : TYPE
        RRATE11 colormap list.

    '''
    
    NWS_REF_COLORS = ['#00ecec', '#00e8ed', '#00e4ed', '#00dfee', '#00dbee', 
                      '#00d7ef', '#00d3ef', '#00cff0', '#00cbf0', '#00c6f1', 
                      '#01c2f1', '#01bef2', '#01baf3', '#01b6f3', '#01b2f4', 
                      '#01adf4', '#01a9f5', '#01a5f5', '#01a1f6', '#0199f6', 
                      '#0190f6', '#0188f6', '#017ff6', '#0176f6', '#016df6', 
                      '#0164f6', '#015cf6', '#0153f6', '#004af6', '#0041f6', 
                      '#0038f6', '#0030f6', '#0027f6', '#001ef6', '#0015f6', 
                      '#000df6', '#0004f6', '#0008ee', '#0016e1', '#0024d3', 
                      '#0032c6', '#0040b8', '#004eab', '#005c9d', '#006a90', 
                      '#007882', '#008675', '#009467', '#00a25a', '#00b04c', 
                      '#00be3f', '#00cc31', '#00da24', '#00e816', '#00f609', 
                      '#00fe00', '#00fb00', '#00f800', '#00f500', '#00f200', 
                      '#00ef00', '#00ec00', '#00e900', '#00e600', '#00e300', 
                      '#00e000', '#00dd00', '#00da00', '#00d700', '#00d400', 
                      '#00d100', '#00ce00', '#00cb00', '#00c800', '#00c400', 
                      '#00c100', '#00be00', '#00bb00', '#00b800', '#00b500', 
                      '#00b200', '#00af00', '#00ac00', '#00a900', '#00a600', 
                      '#00a300', '#00a000', '#009d00', '#009900', '#009600', 
                      '#009300', '#009000', '#0d9600', '#1b9c00', '#29a200', 
                      '#37a800', '#45ae00', '#53b400', '#61ba00', '#6fc000', 
                      '#7dc600', '#8bcd00', '#99d300', '#a7d900', '#b5df00', 
                      '#c3e500', '#d1eb00', '#dff100', '#edf700', '#fbfd00', 
                      '#fefd00', '#fdf900', '#fbf600', '#faf200', '#f9ef00', 
                      '#f7eb00', '#f6e800', '#f5e400', '#f4e100', '#f2dd00', 
                      '#f1da00', '#f0d600', '#eed300', '#edd000', '#eccc00', 
                      '#eac900', '#e9c500', '#e8c200', '#e8bf00', '#e9bc00', 
                      '#eab900', '#ecb700', '#edb400', '#eeb200', '#f0af00', 
                      '#f1ac00', '#f2aa00', '#f4a700', '#f5a400', '#f6a200', 
                      '#f79f00', '#f99c00', '#fa9a00', '#fb9700', '#fd9500', 
                      '#fe9200', '#ff8e00', '#ff8600', '#ff7e00', '#ff7600', 
                      '#ff6e00', '#ff6600', '#ff5e00', '#ff5600', '#ff4e00', 
                      '#ff4700', '#ff3f00', '#ff3700', '#ff2f00', '#ff2700', 
                      '#ff1f00', '#ff1700', '#ff0f00', '#ff0700', '#ff0000', 
                      '#fd0000', '#fa0000', '#f80000', '#f60000', '#f40000', 
                      '#f10000', '#ef0000', '#ed0000', '#eb0000', '#e80000', 
                      '#e60000', '#e40000', '#e20000', '#df0000', '#dd0000', 
                      '#db0000', '#d90000', '#d60000', '#d50000', '#d40000', 
                      '#d30000', '#d10000', '#d00000', '#cf0000', '#ce0000', 
                      '#cd0000', '#cb0000', '#ca0000', '#c90000', '#c80000', 
                      '#c60000', '#c50000', '#c40000', '#c30000', '#c20000', 
                      '#c00000', '#c20009', '#c60017', '#c90025', '#cd0033', 
                      '#d00041', '#d4004f', '#d7005d', '#da006b', '#de0079', 
                      '#e10087', '#e50095', '#e800a3', '#ec00b1', '#ef00bf', 
                      '#f300cd', '#f600db', '#fa00e9', '#fd00f7', '#fd02fe', 
                      '#f707fb', '#f10bf8', '#ec10f5', '#e615f2', '#e119ef', 
                      '#db1eec', '#d523e9', '#d027e6', '#ca2ce3', '#c531e0', 
                      '#bf35dd', '#b93ada', '#b43fd7', '#ae43d4', '#a948d1', 
                      '#a34dce', '#9d51cb', '#9754c7', '#8f4fbc', '#864bb1', 
                      '#7e46a6', '#76419a', '#6d3d8f', '#653884', '#5c3379', 
                      '#542f6e', '#4c2a63', '#432558', '#3b214d', '#321c42', 
                      '#2a1737', '#22132c', '#190e21', '#110916', '#08050b', 
                      '#000000'] 
    
    THEODORE16_COLORS = ['#acacfd', '#aaaafd', '#a8a8fd', '#a7a7fd', '#a5a5fd', 
                         '#a3a3fd', '#a1a1fd', '#a0a0fd', '#9e9efd', '#9c9cfd', 
                         '#9a9afd', '#9999fd', '#9797fd', '#9595fd', '#9393fd', 
                         '#9292fd', '#9090fd', '#8e8efd', '#8c8cfb', '#8b8bf8', 
                         '#8989f6', '#8888f3', '#8686f1', '#8484ee', '#8383ec', 
                         '#8181e9', '#8080e7', '#7e7ee4', '#7d7de2', '#7b7bdf', 
                         '#7979dd', '#7878da', '#7676d8', '#7575d5', '#7373d3', 
                         '#7171d1', '#6f6fce', '#6c6dcc', '#6a6bca', '#6868c7', 
                         '#6666c5', '#6464c3', '#6262c0', '#5f60be', '#5d5ebb', 
                         '#5b5cb9', '#595ab7', '#5757b4', '#5555b2', '#5253b0', 
                         '#5051ad', '#4e4fab', '#4b53a1', '#495797', '#465b8d', 
                         '#435f83', '#416279', '#3e666f', '#3b6a65', '#396e5b', 
                         '#367250', '#347646', '#317a3c', '#2e7e32', '#2c8128', 
                         '#29851e', '#268914', '#248d0a', '#219100', '#259305', 
                         '#28950b', '#2c9710', '#2f9916', '#339b1b', '#369d21', 
                         '#3a9f26', '#3da12c', '#41a431', '#44a637', '#48a83c', 
                         '#4baa42', '#4fac47', '#52ae4d', '#56b052', '#59b258', 
                         '#5db45d', '#60b560', '#64b764', '#67b867', '#6bba6b', 
                         '#6ebb6e', '#71bc71', '#75be75', '#78bf78', '#7cc17c', 
                         '#7fc27f', '#83c483', '#86c586', '#89c689', '#8dc88d', 
                         '#90c990', '#94cb94', '#97cc97', '#9acd9a', '#9ecf9e', 
                         '#a1d0a1', '#a5d2a5', '#a8d3a8', '#acd4ac', '#afd6af', 
                         '#b3d7b3', '#b6d9b6', '#badaba', '#bddcbd', '#c1ddc1', 
                         '#c4dec4', '#c8e0c8', '#cbe1cb', '#cfe3cf', '#d2e4d2', 
                         '#d4e4ce', '#d5e4cb', '#d7e4c7', '#d9e4c4', '#dbe4c0', 
                         '#dce4bc', '#dee4b9', '#e0e4b5', '#e1e3b2', '#e3e3ae', 
                         '#e5e3ab', '#e6e3a7', '#e8e3a3', '#eae3a0', '#ece39c', 
                         '#ede399', '#efe395', '#efe28c', '#efe084', '#efdf7b', 
                         '#efdd72', '#efdc69', '#efdb61', '#efd958', '#efd84f', 
                         '#efd647', '#efd53e', '#efd335', '#efd22d', '#efd124', 
                         '#efcf1b', '#efce12', '#efcc0a', '#efcb01', '#efca02', 
                         '#efc803', '#efc704', '#efc505', '#efc405', '#efc306', 
                         '#efc107', '#efc008', '#efbe09', '#efbd0a', '#efbb0b', 
                         '#efba0c', '#efb90c', '#efb70d', '#efb60e', '#efb40f', 
                         '#efb310', '#ecb012', '#eaae13', '#e7ab15', '#e4a917', 
                         '#e1a619', '#dfa31a', '#dca11c', '#d99e1e', '#d79c1f', 
                         '#d49921', '#d19723', '#cf9424', '#cc9126', '#c98f28', 
                         '#c68c2a', '#c48a2b', '#c1872d', '#bf812c', '#be7b2b', 
                         '#bc742a', '#ba6e29', '#b86828', '#b76227', '#b55b26', 
                         '#b35525', '#b24f25', '#b04924', '#ae4223', '#ad3c22', 
                         '#ab3621', '#a93020', '#a7291f', '#a6231e', '#a41d1d', 
                         '#a52121', '#a72424', '#a82828', '#aa2b2b', '#ab2f2f', 
                         '#ad3332', '#ae3636', '#b03a39', '#b13d3d', '#b34140', 
                         '#b44444', '#b64847', '#b74c4b', '#b94f4e', '#ba5352', 
                         '#bc5655', '#bd5a59', '#bf5857', '#c15756', '#c25554', 
                         '#c45453', '#c65251', '#c85050', '#ca4f4e', '#cc4d4d', 
                         '#cd4c4b', '#cf4a4a', '#d14948', '#d34747', '#d54545', 
                         '#d74444', '#d84242', '#da4141', '#dc3f3f', '#de4042', 
                         '#df4046', '#e14149', '#e3424c', '#e4424f', '#e64353', 
                         '#e84456', '#e94459', '#eb455d', '#ec4560', '#ee4663', 
                         '#f04767', '#f1476a', '#f3486d', '#f54970', '#f64974', 
                         '#f84a77'] 
    
    NWS_CC_COLORS = ['#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', 
                     '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', 
                     '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', 
                     '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', 
                     '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', 
                     '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', 
                     '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', 
                     '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', 
                     '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', 
                     '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', 
                     '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', 
                     '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', 
                     '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', 
                     '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', 
                     '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', 
                     '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', 
                     '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', 
                     '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', 
                     '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', 
                     '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', 
                     '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', 
                     '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', '#0f0f8c', 
                     '#0f0f8d', '#0f0f8e', '#0f0f90', '#0e0e91', '#0e0e92', 
                     '#0e0e94', '#0e0e95', '#0e0e97', '#0e0e98', '#0e0e99', 
                     '#0e0e9b', '#0d0d9c', '#0d0d9d', '#0d0d9f', '#0d0da0', 
                     '#0d0da2', '#0d0da3', '#0d0da4', '#0c0ca6', '#0c0ca7', 
                     '#0c0ca8', '#0c0caa', '#0c0cab', '#0c0cad', '#0c0cae', 
                     '#0b0baf', '#0b0bb1', '#0b0bb2', '#0b0bb3', '#0b0bb5', 
                     '#0b0bb6', '#0b0bb8', '#0b0bb9', '#0a0aba', '#0a0abc', 
                     '#0a0abd', '#0b0bbf', '#0e0ec0', '#1111c2', '#1414c4', 
                     '#1717c6', '#1a1ac7', '#1d1dc9', '#2020cb', '#2323cd', 
                     '#2626cf', '#2929d0', '#2c2cd2', '#2f2fd4', '#3232d6', 
                     '#3535d7', '#3838d9', '#3b3bdb', '#3e3edd', '#4141df', 
                     '#4444e0', '#4747e2', '#4a4ae4', '#4d4de6', '#5050e8', 
                     '#5353e9', '#5656eb', '#5959ed', '#5c5cef', '#5f5ff0', 
                     '#6262f2', '#6565f4', '#6868f6', '#6b6bf8', '#6f6ff9', 
                     '#7272fb', '#7575fd', '#7878ff', '#7681f4', '#748be7', 
                     '#7295db', '#70a0ce', '#6eaac1', '#6cb4b4', '#6abfa7', 
                     '#68c99b', '#66d38e', '#64dd81', '#62e874', '#60f268', 
                     '#61f35f', '#65f157', '#68ee50', '#6bec48', '#6fe941', 
                     '#72e73a', '#75e432', '#78e22b', '#7cdf23', '#7fdd1c', 
                     '#82db15', '#86d80d', '#8dd90a', '#97dc09', '#a0df08', 
                     '#aae307', '#b4e606', '#bee905', '#c8ed05', '#d2f004', 
                     '#dcf303', '#e6f702', '#effa01', '#f9fd00', '#fffb00', 
                     '#fff100', '#ffe800', '#ffdf00', '#ffd500', '#ffcc00', 
                     '#ffc200', '#ffb900', '#ffaf00', '#ffa600', '#ff9c00', 
                     '#ff9300', '#fd8400', '#f76700', '#f14a00', '#eb2d00', 
                     '#e51000', '#d90307', '#c80918', '#b60f28', '#a51538', 
                     '#931b49', '#a74170', '#d67fa7', '#feb3d6', '#f7a8d1', 
                     '#ef9ecc', '#e894c6', '#e089c1', '#d97fbb', '#d175b6', 
                     '#ca6ab1', '#c360ab', '#bb56a6', '#b44ca1', '#ac419b', 
                     '#a53796']
    
    RRATE11_COLORS =  ['#800080', '#7e0281', '#7c0383', '#7a0584', '#780785', 
                       '#760986', '#740a88', '#720c89', '#700e8a', '#6e108b', 
                       '#6c118d', '#6a138e', '#68158f', '#651690', '#631892', 
                       '#611a93', '#5f1c94', '#5d1d95', '#5b1f97', '#592198', 
                       '#572399', '#55249a', '#53269c', '#51289d', '#4f299e', 
                       '#4d2b9f', '#4d2da2', '#4e2fa5', '#4f31a8', '#5033ab', 
                       '#5234ae', '#5336b1', '#5438b4', '#553ab7', '#573cba', 
                       '#583ebd', '#5940c0', '#5a42c3', '#5c44c6', '#5d45c9', 
                       '#5e47cc', '#5f49cf', '#614bd2', '#624dd6', '#634fd9', 
                       '#6451dc', '#6653df', '#6754e2', '#6856e5', '#6958e8', 
                       '#6b5aeb', '#6c5cee', '#685ee5', '#645fdc', '#6061d3', 
                       '#5c62ca', '#5964c1', '#5566b8', '#5167af', '#4d69a6', 
                       '#496a9c', '#456c93', '#416e8a', '#3d6f81', '#3a7178', 
                       '#36736f', '#327466', '#2e765d', '#2a7754', '#26794b', 
                       '#227b42', '#1e7c39', '#1a7e30', '#177f27', '#13811e', 
                       '#0f8315', '#0b840c', '#0b8609', '#0e880c', '#118b0f', 
                       '#158d12', '#188f15', '#1b9118', '#1e941c', '#22961f', 
                       '#259822', '#289a25', '#2c9c28', '#2f9f2c', '#32a12f', 
                       '#35a332', '#39a535', '#3ca838', '#3faa3b', '#43ac3f', 
                       '#46ae42', '#49b145', '#4db348', '#50b54b', '#53b74e', 
                       '#56ba52', '#5abc55', '#5dbe58', '#63c05e', '#69c264', 
                       '#6ec469', '#74c66f', '#7ac775', '#80c97b', '#85cb80', 
                       '#8bcd86', '#91cf8c', '#97d192', '#9cd397', '#a2d59d', 
                       '#a8d6a3', '#aed8a9', '#b3daae', '#b9dcb4', '#bfdeba', 
                       '#c5e0c0', '#cbe2c6', '#d0e4cb', '#d6e6d1', '#dce7d7', 
                       '#e2e9dd', '#e7ebe2', '#edede8', '#f0eee7', '#f1edde', 
                       '#f1ecd5', '#f2ebcc', '#f2eac3', '#f3eaba', '#f3e9b1', 
                       '#f4e8a8', '#f4e79f', '#f5e796', '#f5e68e', '#f6e585', 
                       '#f6e47c', '#f7e373', '#f7e36a', '#f8e261', '#f8e158', 
                       '#f9e04f', '#f9df46', '#fadf3d', '#fade35', '#fbdd2c', 
                       '#fbdc23', '#fcdc1a', '#fcdb11', '#fdda08', '#fcd809', 
                       '#fcd50a', '#fbd30c', '#fbd00d', '#face0e', '#f9cb0f', 
                       '#f9c911', '#f8c612', '#f8c413', '#f7c114', '#f7bf15', 
                       '#f6bc17', '#f5ba18', '#f5b719', '#f4b51a', '#f4b21b', 
                       '#f3b01d', '#f2ae1e', '#f2ab1f', '#f1a920', '#f1a622', 
                       '#f0a423', '#efa124', '#ef9f25', '#ee9c26', '#ec9a28', 
                       '#e99829', '#e5962b', '#e2932c', '#de912e', '#db8f2f', 
                       '#d88d31', '#d48b32', '#d18834', '#cd8636', '#ca8437', 
                       '#c68239', '#c3803a', '#bf7d3c', '#bc7b3d', '#b9793f', 
                       '#b57740', '#b27542', '#ae7243', '#ab7045', '#a76e46', 
                       '#a46c48', '#a06a49', '#9d674b', '#99654c', '#96634e', 
                       '#99634f', '#9d6351', '#a06352', '#a46353', '#a76354', 
                       '#ab6356', '#ae6357', '#b26358', '#b56359', '#b9635b', 
                       '#bc635c', '#bf635d', '#c3625e', '#c66260', '#ca6261', 
                       '#cd6262', '#d16263', '#d46265', '#d86266', '#db6267', 
                       '#de6268', '#e2626a', '#e5626b', '#e9626c', '#ec626d', 
                       '#ed606d', '#ec5d6b', '#eb5a69', '#ea5767', '#e95465', 
                       '#e85063', '#e74d61', '#e64a5f', '#e5475d', '#e4435b', 
                       '#e34059', '#e23d57', '#e13a55', '#e03754', '#df3352', 
                       '#de3050', '#dd2d4e', '#dc2a4c', '#db274a', '#da2348', 
                       '#d92046', '#d81d44', '#d71a42', '#d61640', '#d5133e', 
                       '#d4103c']
    
    return NWS_REF_COLORS, THEODORE16_COLORS, NWS_CC_COLORS, RRATE11_COLORS