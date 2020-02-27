# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 13:33:31 2018

Plot strains from the BiV simulation in a similar way as
the strain analysis of the LifeTec experiments.

@author: Hermans
"""

import matplotlib.pyplot as plt
import os
import numpy as np

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from postprocessing import (StrainAnalysis, 
                            find_simulation_curves,
                            get_rpm_linestyles,
                            get_rpm_colors, plot_global_hemodynamic_variable,
                            merge_dictionaries)

plt.close('all')

# Specify output directory for figures when compare_with_simulation == 1.
dir_out = r'figures'

rpm_all = [0, 7500, 8500, 9500, 10500, 11500, 12000]

strain_reference = 'begin_ic_ref'  # 'stress_free_ref' or 'begin_ic_ref'.

strains_range = 'auto'
average_mode = 1   # Include all segments (1) or optiomal segments (2)
average_type = 'median'  # 'mean' or 'median'
auto_exclude = False  
plane_cavity = 'SAX_LV'
plot_mode = 1  # Plot segments and median (1), plot median + IQR (2), plot median (3)
reorient_time = False
t_cycle = 700  # ms
transmural_weight = False     
trackpoints_row = None #np.arange(12, 15)
trackpoints_column = None #np.arange(10, 15)
optimal_segments = {'Elll': [3],
                    'Ettl': [3]}  

fontsize = 18

rpm_linestyles = get_rpm_linestyles()
rpm_colors = get_rpm_colors()

# Add strain reference to outdirname.
dir_out += '_{}'.format(strain_reference) 
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

max_strains_sim = {}
for rpm in rpm_all:
            
    # Simulation curves file.
    filename_simulation = find_simulation_curves(rpm=rpm, simulation_type='BiV', strain_reference=strain_reference)

    post_simulation = StrainAnalysis(filename_simulation, 
                          average_beats=True, 
                          average_mode=average_mode, 
                          average_type=average_type,
                          auto_exclude=auto_exclude,
                          optimal_segments=optimal_segments,
                          plane_cavity=plane_cavity,
                          plot_mode=plot_mode,
                          reorient_time=reorient_time,
                          t_cycle= t_cycle,
                          transmural_weight=transmural_weight,
                          strains_range=strains_range,
                          trackpoints_row=trackpoints_row,
                          trackpoints_column=trackpoints_column)
    
    if plot_mode == 3:
        # Plot all pump speeds in same figure.
        # Create/select name for figure.
        filename_reference_simulation = find_simulation_curves(rpm=rpm_all[0], simulation_type='BiV')
        figname = os.path.split(filename_reference_simulation)[0]
        
        # Plot simulation data in figure.            
        max_strains_sim_i = post_simulation.plot_strains(fontsize=fontsize, figname=figname, 
                                                     label=rpm, linestyle=rpm_linestyles[rpm],
                                                     color=rpm_colors[rpm])[1]
    else:
        # Separate plots for separate pump speeds
        figname = os.path.split(filename_simulation)[0]
        
        # Plot simulation data in figure.            
        max_strains_sim_i = post_simulation.plot_strains(fontsize=fontsize, figname=figname)[1]
        
        # Save figure to dir_out.
        figure_filename = os.path.join(dir_out, 'strains_{}_rpm_mode_{}_{}.png'.format(rpm, plot_mode, plane_cavity))
        plt.savefig(figure_filename, dpi=300, bbox_inches="tight")
  
    # Merge max_strains.
    max_strains_sim_i['rpm'] = rpm
    max_strains_sim_i['hc'] = 1
    merge_dictionaries(max_strains_sim, max_strains_sim_i)
    
# Save figure to dir_out.
legend = plt.legend(fontsize=fontsize-2, title='LVAD speed (rpm)')
legend.get_title().set_fontsize(fontsize-2)
figure_filename = os.path.join(dir_out, 'strains_timecourse_mode_{}_{}.png'.format(plot_mode, plane_cavity))
plt.savefig(figure_filename, dpi=300, bbox_inches="tight")

plt.figure()
post_simulation.plot_segments()

# Plot maximum strains per pump speed.
if plane_cavity == 'SAX_RV':
    keys_all = ['Ervccc', 'Ervrrc', 'Ervcrc']
    titles_all = ['Min $E_{cc}$', 'Max $E_{rr}$', 'Min $E_{cr}$']
    ylabels_all = ['$E_{cc}$', '$E_{rr}$', '$E_{cr}$']
    
elif plane_cavity == 'SAX_LV':
    keys_all = ['Eccc', 'Errc', 'Ecrc']
    titles_all = ['Min $E_{cc}$', 'Max $E_{rr}$', 'Min $E_{cr}$']
    ylabels_all = ['$E_{cc}$', '$E_{rr}$', '$E_{cr}$']
    
elif plane_cavity == 'LAX_BiV':
    keys_all = ['Elll', 'Ettl']
    titles_all = ['Min $E_{ll}$', 'Max $E_{tt}$']
    ylabels_all = ['$E_{ll}$', '$E_{tt}$']
    
ylimits = {'Eccc': [-0.23, -0.03],
           'Errc': [0.05, 0.4],
           'Ecrc': [-0.11, 0.01]}

plt.figure('simulation', figsize=(22, 6))
nrows = 1 #np.round(np.sqrt(len(keys_all)))
ncols = len(keys_all) #np.ceil(np.sqrt(len(keys_all)))
for i, key in enumerate(keys_all):

    plt.subplot(nrows, ncols, i+1)
    plot_global_hemodynamic_variable(max_strains_sim, key, 
                                     label_prefix='', fontsize=fontsize,
                                     linestyle='-', marker='o')

    plt.title(titles_all[i], fontsize=fontsize+2)
#        plt.ylabel(ylabels_all[i], fontsize=fontsize)
    if key in ylimits.keys():
        plt.ylim(ylimits[key])
    
    if i + ncols < len(keys_all):
        # Remove x-label for non-bottom plots.
        plt.xlabel('')

plt.legend(fontsize=fontsize-2)
plt.savefig(os.path.join(dir_out, 'strain_summary_simulation_{}.png'.format(plane_cavity)), dpi=300, bbox_inches="tight")
