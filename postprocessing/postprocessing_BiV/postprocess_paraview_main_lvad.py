# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 13:11:40 2018

@author: Hermans
"""
import sys
sys.path.append("..") # Adds higher directory to python modules path.

from postprocessing_paraview import postprocess_paraview, find_paraview_data
from postprocessing import (get_rpm_colors,
                            merge_dictionaries,subplot_axis, remove_ticks,
                            plot_global_hemodynamic_variable, figure_make_up)

import matplotlib.pyplot as plt
import numpy as np
import os

plt.close('all')

# --------------------------------------------------------------------------- #
# INPUTS
# --------------------------------------------------------------------------- #
inputs = {'cut_off_low': -3, # Specify the low z-coordinate of the cube with to be included nodes [cm].
          'cut_off_high': 1, # Specify the high z-coordinate of the cube with to be included nodes [cm].
          'slice_thickness': 1,  # Specify the thickness of the cube (thick longitudinal slice) with to be included nodes [cm]. 
          'load_pickle_file': True}  # Reload the data from a temporary pickle file if available for fatser loading.

# Specify the pump speeds to analyze. 
# If you specify -1, it will look for the healthy simulation.
rpm_all = [-1, 0, 7500, 8500, 9500, 10500, 11500, 12000]

# Set which state to take as reference state for strains.
# Choose between 'stress_free' or 'onset_shortening'.
strain_reference = 'stress_free'  

# Plot options.
fontsize = 20
linewidth = 3

# Specify output directory.
dir_out = r'figures'

# --------------------------------------------------------------------------- #

# Check if the output directory exists.
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

rpm_colors = get_rpm_colors()
rpm_linestyles = {-1: '--', 0: "-", 7500: "-", 8500: "-", 9500:"-", 10500:'-', 11500: "-", 12000: "-"}

# Initialize figures.
fig_ss = plt.figure(figsize=(22, 5), dpi=100)
fig_tstrain = plt.figure(figsize=(22, 5), dpi=100)
fig_tstress = plt.figure(figsize=(22, 5), dpi=100)
fig_pv = plt.figure(figsize=(8, 6), dpi=80)

mechanical_summary = {}
for i, rpm in enumerate(rpm_all):
    directory_1 = find_paraview_data(rpm=rpm)
    post_1 = postprocess_paraview(directory_1, name=str(rpm), **inputs)
    
    linestyle = rpm_linestyles[rpm]
    color = rpm_colors[rpm]
    
    if strain_reference == 'stress_free':
        axlimits = [-0.1, 0.16, -2, 52]
    else:
        axlimits = [-0.2, 0.02, -2, 52]
    post_1.plot_stress_strain(color=rpm_colors[rpm], 
                              linestyle=linestyle, 
                              fig=fig_ss, 
                              label=str(rpm) if not rpm<=-1 else None, 
                              fontsize=fontsize, 
                              linewidth=linewidth, 
                              reference=strain_reference, 
                              axlimits=axlimits)
    
    post_1.plot_time_strain(color=rpm_colors[rpm], 
                            linestyle=linestyle, 
                            fig=fig_tstrain, 
                            label=str(rpm) if not rpm<=-1 else None, 
                            fontsize=fontsize, 
                            linewidth=linewidth, 
                            reference=strain_reference)
    
    post_1.plot_time_stress(color=rpm_colors[rpm], 
                            linestyle=linestyle, 
                            fig=fig_tstress, 
                            label=str(rpm) if not rpm<=-1 else None, 
                            fontsize=fontsize, 
                            linewidth=linewidth)    

#    post_1.plot_pv_loops(linestyle=linestyle, color=color, fig=fig_pv, label=str(rpm), fontsize=fontsize, linewidth=linewidth)

#    post_1.show_global_function()
    
#    post_1.show_local_function(linewidth=linewidth)
    
    loc_sum_i = post_1.return_local_function_summary(reference=strain_reference)
                    
    # Merge summaries.
    loc_sum_i['rpm'] = rpm
    loc_sum_i['hc'] = 1
    merge_dictionaries(mechanical_summary, loc_sum_i) 

    print('')
    
# Show selected nodes/regions.
post_1.show_regions(projection = '2d', fontsize=fontsize)
plt.savefig(os.path.join(dir_out, 'selected_regions.png'),  dpi=300, bbox_inches="tight")

# Save figures.
plt.figure(fig_ss.number)
if len(rpm_all) > 1:
    leg = plt.legend(title='LVAD speed\n(rpm)', fontsize=fontsize, loc=(1.04,0))
    leg.get_title().set_fontsize(fontsize)
plt.savefig(os.path.join(dir_out, 'stress_strain_loops_{}.png'.format(strain_reference)),  dpi=300, bbox_inches="tight")

plt.figure(fig_tstrain.number)
if len(rpm_all) > 1:
    leg = plt.legend(title='LVAD speed\n(rpm)', fontsize=fontsize, loc=(1.04,0))
    leg.get_title().set_fontsize(fontsize)
plt.savefig(os.path.join(dir_out, 'time_strain_loops_{}.png'.format(strain_reference)),  dpi=300, bbox_inches="tight")

plt.figure(fig_tstress.number)
if len(rpm_all) > 1:
    leg = plt.legend(title='LVAD speed\n(rpm)', fontsize=fontsize, loc=(1.04,0))
    leg.get_title().set_fontsize(fontsize)
plt.savefig(os.path.join(dir_out, 'time_stress_loops.png'),  dpi=300, bbox_inches="tight")

#plt.figure(fig_pv.number)
#if len(rpm_all) > 1:
#    leg = plt.legend(title='LVAD speed\n(rpm)', fontsize=fontsize)
#    leg.get_title().set_fontsize(fontsize)
#plt.savefig(os.path.join(dir_out, 'pv_loops.png'),  dpi=300, bbox_inches="tight")

# Subplot location.
sploc = {
        'max_strain_RV': 1,
        'max_strain_SEP': 2,
        'max_strain_LV': 3,
        'min_strain_RV': 1,
        'min_strain_SEP': 2,
        'min_strain_LV': 3,
        'range_strain_RV': 4,
        'range_strain_SEP': 5,
        'range_strain_LV': 6,
        'max_stress_RV': 7,
        'max_stress_SEP': 8, 
        'max_stress_LV': 9,
        'fiber_work_RV': 10,
        'fiber_work_SEP': 11,
        'fiber_work_LV': 12
        }

ylabels_all = {
              'max_strain': '$\epsilon_f$ [-]',
              'min_strain': '$\epsilon_f$ [-]',
              'range_strain': '$\epsilon_f$ range [-]',
              'max_stress': '$\sigma_{f, max}$ [kPa]',
              'fiber_work': '$w_f$ [mJ/mL]'
              }

num_plots = len(np.unique(list(sploc.values()))>0)
plt.figure('summary', figsize=(7*3/2, 12*4/6), dpi=100)
nrows = 4 #np.round(np.sqrt(len(keys_all)))
ncols = 3 #np.ceil(np.sqrt(len(keys_all)))
fontsize=10

axis_all = {}
keys_all = mechanical_summary.keys()
for i, key in enumerate(keys_all):
    if not key in sploc.keys():
        # Skip these keys.
        continue
    
    loc = sploc[key]
    
    if not loc in axis_all.keys():
        # First entry in subplot.
        axis_all[loc] = subplot_axis(nrows, ncols, loc)
        create_legend = False
    else:
        # Not the first entry in subplot.
        create_legend = True
        
    ax = axis_all[loc]
    
    plt.sca(ax.axis)
    plot_global_hemodynamic_variable(mechanical_summary, key, 
                                     label=key,
                                     color=ax.nextcolor(),
                                     marker=ax.nextmarker(),
                                     markersize=8*fontsize/16, fontsize=fontsize)
    
    sep_idx = key.rfind('_')
    title = key[sep_idx+1:]
    y_label = ylabels_all[key[:sep_idx]]
    figure_make_up(title=title, ylabel=y_label, 
                   fontsize=fontsize, create_legend=create_legend) 
    
    if loc + ncols <= num_plots:
        # Remove x-ticks and label from non-bottom plots.
        remove_ticks('x')
        
    if loc > ncols:
        # Remove titles from non-top plots.
        plt.title('')
        
# Additinal hardcoded make up.
for i in range(1, num_plots+1):    
    plt.sca(axis_all[i].axis)
    ylabel = axis_all[i].axis.get_ylabel()

    left_plot = int(np.ceil(i/ncols)*ncols - ncols+1)
    
    col_pos = i % ncols

    if col_pos == 1:
        # Find suitable y limits for row.
        ymin = np.inf
        ymax = -np.inf
        for ax_i in range(left_plot, left_plot+ncols):
            ax = axis_all[ax_i].axis.axis()
            if ax[2] < ymin:
                ymin = ax[2]
            if ax[3] > ymax:
                ymax = ax[3]
                    
        pad_frac = 1/15
        pad = (ymax-ymin)*pad_frac
        plt.ylim(ymin-pad, ymax+pad)
    else:
        # Take axis from first column.
        ax = axis_all[left_plot].axis.axis()
        plt.axis(ax)
        
        # Remove ylabel and ticks from non-left columns.
        remove_ticks('y')
        
# Replace titles of double plots ad add labels
ax = axis_all[1].new_make_up(fontsize=fontsize,
                             new_labels=['$\epsilon_{f, max}$',
                                         '$\epsilon_{f, min}$'])
ax = axis_all[2].new_make_up(fontsize=fontsize,
                             new_labels=['$\epsilon_{f, max}$',
                                         '$\epsilon_{f, min}$'])
ax = axis_all[3].new_make_up(fontsize=fontsize,
                             new_labels=['$\epsilon_{f, max}$',
                                         '$\epsilon_{f, min}$'])
plt.savefig(os.path.join(dir_out, 'local_summary_biv.png'), dpi=300, bbox_inches="tight")


