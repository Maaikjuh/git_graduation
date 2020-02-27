# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:36:48 2018

Plots the simulated strains per per speed.

@author: Hermans
"""

import matplotlib.pyplot as plt
import os

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from postprocessing import (StrainAnalysis, 
                            find_simulation_curves,
                            get_rpm_linestyles,
                            get_rpm_colors,
                            remove_ticks)

plt.close('all')

dir_out = r'figures'

hc_all = [1, 3]
rpm_all = [0, 7500, 8500, 9500, 10500]
simulation_type = 'fixed_preload'  # 'lifetec_fit' or 'fixed_preload'

inputs = {
    'average_beats': True,
    'average_mode': 1,   # Include all segments (1) or optiomal segments (2)
    'average_type': 'median',  # 'mean' or 'median'
    'auto_exclude': False,  
    'plot_mode': 3,  # Plot segments and median (1), plot median + IQR (2), plot median (3)
    'transmural_weight': False,
    'reorient_time': True,
    'reorient_time_mode': 2, # Reorient such that strain == 0 at start (1), or t_act == 0 at start (2).
    'optimal_segments': {'Ecc': [4,5],
                        'Err': [4,5],
                        'Ecr': [4,5]},
    'strains_range': {'Eccc': [-0.11, 0.02],
                     'Errc': [-0.05, 0.25],
                     'Ecrc': [-0.11, 0.02]}
    #'strains_range' = 'auto'
    }
    
fontsize = 18

rpm_linestyles = get_rpm_linestyles()
rpm_colors = get_rpm_colors()

if simulation_type == 'fixed_preload':
    dir_out += '_fixed_preload' 

if not os.path.exists(dir_out):
    os.makedirs(dir_out)

for hc in hc_all:
    for i, rpm in enumerate(rpm_all):
                
        # Simulation curves file.
        filename_simulation = find_simulation_curves(hc, rpm, simulation_type=simulation_type)

        post_simulation = StrainAnalysis(filename_simulation, 
                              **inputs)
        
        # Select figure.
        filename_reference_simulation = find_simulation_curves(hc, rpm_all[0], simulation_type=simulation_type)
        figname = os.path.split(filename_reference_simulation)[0]
        
        # Plot simulation data.            
        linestyle = '-' 
        max_strains_sim_i = post_simulation.plot_strains(fontsize=fontsize, 
                                                         figname=figname, 
                                                         figure_shape='column',
                                                         label=rpm, 
                                                         linestyle=linestyle,
                                                         color=rpm_colors[rpm])[1]
        
        # Save figure to dir_out.
        if i == (len(rpm_all) - 1):
            if hc == hc_all[-1]:
                plt.figure(figname)
                legend = plt.legend(fontsize=fontsize-2, 
                                    title='LVAD speed\n(rpm)',
                                    loc=(1.04, 2.9))
                legend.get_title().set_fontsize(fontsize-2)
            
            plt.tight_layout()
            figure_filename = os.path.join(dir_out, 'hc{}_strains_timecourse_mode_{}.png'.format(hc, inputs['plot_mode']))
            plt.savefig(figure_filename, dpi=300, bbox_inches="tight")

    