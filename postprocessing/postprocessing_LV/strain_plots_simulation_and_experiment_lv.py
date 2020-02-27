# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:36:48 2018

Plots the strains of the experiments and optionally of the simulation.

@author: Hermans
"""

import numpy as np
import matplotlib.pyplot as plt
import os

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from postprocessing import (merge_dictionaries, StrainAnalysis, 
                            plot_global_hemodynamic_variable,
                            find_simulation_curves, find_experiment_curves)

plt.close('all')

# --------------------------------------------------------------------------- #
# INPUTS
# --------------------------------------------------------------------------- #
# Specify which experiment(s) to postprocess.
hc_all = [1]
rpm_all = [9500]
simulation_type = 'lifetec_fit'  # 'lifetec_fit' or 'fixed_preload'

# Specify output directory for figures when compare_with_simulation == 1.
dir_out = r'figures'

# --------------------------------------------------------------------------- #
# SETTINGS
# --------------------------------------------------------------------------- #
compare_with_simulation = True  # Also plot the simulated strains.
downsampled = False  # Use downsampled speckle tracking data.
fontsize = 20  # Fontsize in plots.

# Specify y-axis ranges.
strains_range = {'Eccc': [-0.18, 0.13],
                 'Errc': [-0.09, 0.33],
                 'Ecrc': [-0.14, 0.08]}

# Settings for the plot with experimental strains.
inputs_experiment = {
                    'average_beats': True,  # If True, for STE data the strains measured at different beats are averaged per tracking point (node).
                    'average_mode': 1,   # Include all segments (1) or optiomal segments (2)
                    'average_type': 'median',  # 'mean' or 'median'
                    'plot_mode': 2,  # Plot segments and average (1), plot average + IQR (2), plot average (3)
                    'transmural_weight': False,
                    'strains_range': strains_range
                    }

# Settings for the plot with simulated strains 
# (only required when compare_with_simulation = True).
inputs_simulation = {
                    'average_beats': False,  # If True, for STE data the strains measured at different beats are averaged per tracking point (node).
                    'average_mode': 1,   # Include all segments (1) or optiomal segments (2)
                    'average_type': 'median',  # 'mean' or 'median'
                    'plot_mode': 2,  # Plot segments and average (1), plot average + IQR (2), plot average (3)
                    'reorient_time_mode': 1,
                    'transmural_weight': False,
                    'strains_range': strains_range
                    }

# --------------------------------------------------------------------------- #

if simulation_type == 'fixed_preload':
    # Automatically add a post fix if we analyze the fixed_preload simulations.
    dir_out += '_fixed_preload' 

# Create output directory if it doesn't exists.
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

# For each requested experiment, plot the strains and keep track on max/min 
# strains per speed.
max_strains_sim = {}
max_strains_exp = {}
for hc in hc_all:
    for rpm in rpm_all:
        
        plt.close('all')
        
        # Experiment curves file.
        filename_experiment = find_experiment_curves(hc, rpm)
       
        if downsampled:
            head, tail = os.path.split(filename_experiment)
            filename_experiment = os.path.join(head, 'downsampled', tail)
        
        post = StrainAnalysis(filename_experiment, 
                              **inputs_experiment)
                
        # Plot strains for all avaiable beats/cycles.
        post.inter_beat_variability(fontsize=fontsize)
        
        # Plot strains for average/specified cycle
        max_strains_exp_i = post.plot_strains(fontsize=fontsize)[1]
     
        # Merge max_strains.
        max_strains_exp_i['rpm'] = rpm
        max_strains_exp_i['hc'] = hc
        merge_dictionaries(max_strains_exp, max_strains_exp_i)
       
        if compare_with_simulation:
            # Plot simulation results over experimental results.
            
            # Simulation curves file.
            filename_simulation = find_simulation_curves(hc, rpm, simulation_type=simulation_type)
    
            post_simulation = StrainAnalysis(filename_simulation, 
                                             **inputs_simulation)
            
            # Plot simulation data over experimental data.
            figname = os.path.split(filename_experiment)[0]
            max_strains_sim_i = post_simulation.plot_strains(fontsize=fontsize, 
                                                             figname=figname,
                                                             linestyle='--',
                                                             color='k')[1]
            plt.figure(figname)
            
            plt.subplot(1, 3, 1)
            plt.legend(['Experiment', 'Simulation'], fontsize=fontsize-2)
            
            # Save figure.
            head, tail = os.path.split(filename_simulation)
            dir_out_temp = os.path.join(head, 'figures')
            figure_filename = os.path.join(dir_out_temp, 'strains_timecourse_mode_{}.png'.format(inputs_simulation['plot_mode']))
            plt.savefig(figure_filename, dpi=300, bbox_inches="tight")
            
            # Merge max_strains.
            max_strains_sim_i['rpm'] = rpm
            max_strains_sim_i['hc'] = hc
            merge_dictionaries(max_strains_sim, max_strains_sim_i)
            
            plt.figure()
            post_simulation.plot_segments()

# Plot the max/min strains as function of pump speed.
keys_all = ['Eccc', 'Errc', 'Ecrc']
titles_all = ['Min $E_{cc}$', 'Max $E_{rr}$', 'Min $E_{cr}$']
ylabels_all = ['$E_{cc}$', '$E_{rr}$', '$E_{cr}$']
ylimits = {'Eccc': [-0.23, -0.03],
           'Errc': [0.00, 0.35],
           'Ecrc': [-0.11, 0.01]}

figsize = (8, 16) # 22, 6
fontsize = 22

nrows = len(keys_all) #1 #np.round(np.sqrt(len(keys_all)))
ncols = 1 # len(keys_all) #np.ceil(np.sqrt(len(keys_all)))
for i, key in enumerate(keys_all):
    
    # Per hert condition.
    for hc_plot in hc_all:
        plt.figure('experiment_hc{}'.format(hc_plot), figsize=figsize)
        plt.subplot(nrows, ncols, i+1)

        # EXPERIMENT
        # Per segment.    
        plot_global_hemodynamic_variable(max_strains_exp, key+'_per_segment', 
                                         label_prefix='', fontsize=fontsize, 
                                         hc_plot=hc_plot, linestyle='-')
        
        if compare_with_simulation:
            # SIMULATION
            plot_global_hemodynamic_variable(max_strains_sim, key, 
                                             label_prefix='', 
                                             fontsize=fontsize,
                                             linestyle='--', 
                                             color='k',
                                             hc_plot=hc_plot)
        else:
            # EXPERIMENT average.
            plot_global_hemodynamic_variable(max_strains_exp, key, 
                                             label_prefix='All', fontsize=fontsize, 
                                             linestyle='-', hc_plot=hc_plot)
        
        plt.title(titles_all[i], fontsize=fontsize+2)
        plt.ylabel(ylabels_all[i], fontsize=fontsize)
        plt.ylim(ylimits[key])

        if i + ncols < len(keys_all):
            # Remove x-label for non-bottom plots.
            plt.xlabel('')
            

#    # SIMULATION
#    plt.figure('simulation', figsize=figsize)
#    plt.subplot(nrows, ncols, i+1)
#    plot_global_hemodynamic_variable(max_strains_sim, key, 
#                                     label_prefix='', fontsize=fontsize,
#                                     linestyle='-')
#
#    plt.title(titles_all[i], fontsize=fontsize+2)
##        plt.ylabel(ylabels_all[i], fontsize=fontsize)
#    plt.ylim(ylimits[key])
#    
#    if i + ncols < len(keys_all):
#        # Remove x-label for non-bottom plots.
#        plt.xlabel('')

for hc_plot in hc_all:
    plt.figure('experiment_hc{}'.format(hc_plot), figsize=(22, 6))
    legend1 = plt.legend(fontsize=fontsize-2, loc=(1.04, 0.1), 
                         title='Experiment\nsegment')
    legend1.get_title().set_fontsize(fontsize-2)
    
    if compare_with_simulation: 
        axes = plt.gca()
        lines = axes.get_lines()
        legend2 = plt.legend([lines[6]], ["Simulation"], fontsize=fontsize-2, 
                             bbox_to_anchor=(1.04, 0), loc=2, borderaxespad=0.)
        axes.add_artist(legend1)
        axes.add_artist(legend2)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(dir_out, 
                             'strain_summary_experiment_hc{}.png'.format(hc_plot)), 
                             dpi=300,
                             bbox_extra_artists=(legend1,), # Does not work... cant fit the legend in saved figure.
                             bbox_inches="tight")

#plt.figure('simulation')
#plt.legend(fontsize=fontsize-2)
#plt.savefig(os.path.join(dir_out, 'strain_summary_simulation.png'), dpi=300, bbox_inches="tight")
#
