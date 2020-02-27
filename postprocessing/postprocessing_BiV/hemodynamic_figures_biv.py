# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 17:08:06 2018

@author: Hermans
"""

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from postprocessing import  (merge_dictionaries,
                             plot_global_hemodynamic_variable,
                             get_hc_names, find_simulation_hemodynamics,
                             load_reduced_dataset, print_hemodynamic_summary,
                             hemodynamic_summary, figure_make_up, 
                             remove_ticks, rescale_yaxis, subplot_axis)                                    

import os

plt.close('all')

# --------------------------------------------------------------------------- #
# INPUTS
# --------------------------------------------------------------------------- #
# Specify output directory.
dir_out = r'figures'

# Specify the pump speeds to analyze. 
# If you specify -1, it will look for the healthy simulation.
rpm_all = [-1, 0, 7500, 8500, 9500, 10500, 11500, 12000]

# Specify the fontsize for plotting
fontsize=16

# --------------------------------------------------------------------------- #

# Create output directory if it doen not exists.
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

# Names for the heart conditions.
hc_names = get_hc_names()

hemo_sum_sim_merged = {}
for j, j_rpm in enumerate(rpm_all):

    # Find and read results file of simulation.    
    csv = find_simulation_hemodynamics(rpm=j_rpm, simulation_type='BiV')
    results = load_reduced_dataset(filename=csv)
    
    # Print a hemodynamic summary.
    print('SIMULATION {}'.format(j_rpm))
    hemo_sum_sim = hemodynamic_summary(results)
    print_hemodynamic_summary(hemo_sum_sim)

    # Add heart condition and pump speed to summaries.
    hemo_sum_sim['hc'] = 1  # We only have one heart condition, so this number does not matter too much
    hemo_sum_sim['rpm'] = j_rpm

    # Merge the data.
    merge_dictionaries(hemo_sum_sim_merged, hemo_sum_sim)

# Subplot location.
sploc = {
          'EDV': 4, 
          'CO': 2,
          'MAP': 8, 
          'PP': 10, 
          'plv_max': 8, 
          'dpdt_max': 12, 
          'LVAD_flow': 2, 
          'CVP': 6, 
          'PCWP': 5, 
          'EDV_rv': 3, 
          'CO_rv': 1, 
          'MAP_pul': 7, 
          'PP_pul': 9, 
          'prv_max': 7, 
          'dp_rvdt_max': 11
          }


titles_all = {'HR': 'Heart rate',
          'EDV': 'LV end diastolic volume', 
          'ESV': 'LV end systoliv volume', 
          'CO': 'LV cardiac output',
          'qao': 'Aortic valve flow', 
          'SV': 'LV stroke volume', 
          'EF': 'LV ejection fraction', 
          'MAP': 'Mean systemic arterial pressure', 
          'SAP': 'Systolic systemic arterial pressure', 
          'DAP': 'Diastolic systemic arterial pressure', 
          'PP': 'Systemic pulse pressure', 
          'plv_max': 'Maximum LV pressure', 
          'dpdt_max': 'Maximum ($dp_{lv}/dt$)', 
          'W': 'LV stroke work', 
          'LVAD_flow': 'LVAD flow', 
          'LVAD_frac': 'LVAD contribution', 
          'CVP': 'Mean systemic venous pressure', 
          'PCWP': 'Mean pulmonary venous pressure', 
          'EDV_rv': 'RV end diastolic volume', 
          'ESV_rv': 'RV end systolic volume', 
          'CO_rv': 'RV cardiac output', 
          'SV_rv': 'RV stroke volume',  
          'EF_rv': 'RV ejection fraction', 
          'MAP_pul': 'Mean pulmonary arterial pressure', 
          'SAP_pul': 'Systolic pulmonary arterial pressure', 
          'DAP_pul': 'Diastolic pulmonary arterial pressure', 
          'PP_pul': 'Pulmonary pulse pressure', 
          'prv_max': 'Maximum RV pressure', 
          'dp_rvdt_max': 'Maximum ($dp_{rv}/dt$)', 
          'W_rv': 'RV stroke work'}

ylabels_all = {'HR': 'HR [bpm]',
          'EDV': 'V [ml]', 
          'ESV': 'V [ml]', 
          'CO': 'q [l/min]',
          'qao': 'q flow [l/min]', 
          'SV': 'V [ml]', 
          'EF': 'EF [-]', 
          'MAP': 'p [mmHg]', 
          'SAP': 'p [mmHg]', 
          'DAP': 'p [mmHg]', 
          'PP': 'p [mmHg]', 
          'plv_max': 'p [mmHg]', 
          'dpdt_max': '$(dp/dt)_{max}$ [mmHg/s]', 
          'W': 'W [J]', 
          'LVAD_flow': 'q [l/min]', 
          'LVAD_frac': 'LVAD contribution [%]', 
          'CVP': 'p [mmHg]', 
          'PCWP': 'p [mmHg]', 
          'EDV_rv': 'V [ml]', 
          'ESV_rv': 'V [ml]', 
          'CO_rv': 'q [l/min]', 
          'SV_rv': 'V [ml]',  
          'EF_rv': 'EF [%]', 
          'MAP_pul': 'p [mmHg]', 
          'SAP_pul': 'p [mmHg]', 
          'DAP_pul': 'p [mmHg]', 
          'PP_pul': 'p [mmHg]', 
          'prv_max': 'p [mmHg]', 
          'dp_rvdt_max': '$(dp/dt)_{max}$ [mmHg/s]', 
          'W_rv': 'W [J]'}

num_plots = len(np.unique(list(sploc.values()))>0)
fig = plt.figure(figsize=(7, 12))
nrows = 6
ncols = 2
fontsize=10

axis_all = {}
keys_all = hemo_sum_sim_merged.keys()
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
    plot_global_hemodynamic_variable(hemo_sum_sim_merged, key, 
                                     label=key,
                                     color=ax.nextcolor(),
                                     marker=ax.nextmarker(),
                                     markersize=8*fontsize/16, fontsize=fontsize)
    
    figure_make_up(title=titles_all[key], ylabel=ylabels_all[key], 
                   fontsize=fontsize, create_legend=create_legend) 
    
    if loc + ncols <= num_plots:
        # Remove x-ticks and label from non-bottom plots.
        remove_ticks('x')

# Additinal hardcoded make up.
for i in range(2, num_plots+1, 2):    
    # remove ylabel from second column.
    plt.sca(axis_all[i].axis)
    plt.ylabel('')
        
    if axis_all[i].axis.get_title() == 'Mean systemic venous pressure':
        # Rescale to left plot.
        ax1 = axis_all[i-1].axis.axis()
                
    else:
        # Rescale to right plot.        
        plt.sca(axis_all[i-1].axis)
        ax1 = axis_all[i].axis.axis()
        
    plt.ylim(rescale_yaxis(ax1, plt.axis()))
    
# Replace titles of double plots ad add labels
ax = axis_all[2].new_make_up(title='LV cardiac output', fontsize=fontsize,
                             new_labels=['$\overline{q}_{art}$',
                                         '$\overline{q}_{lvad}$']); 
ax = axis_all[8].new_make_up(title='Systemic pressures', fontsize=fontsize,
                             new_labels=['$\overline{p}_{art}$',
                                         '$p_{lv, max}$']); 
ax = axis_all[7].new_make_up(title='Pulmonary pressures', fontsize=fontsize,
                             new_labels=['$\overline{p}_{art}$',
                                         '$p_{rv, max}$']); 
plt.savefig(os.path.join(dir_out, 'hemo_summary_biv.png'), dpi=300, bbox_inches="tight")

# Plot RV EF, Stroke volume and stroke work
keys_all = ['SV_rv', 'EF_rv', 'W_rv']
plt.figure(figsize=(17, 3.3))
fontsize = 16
for i, key in enumerate(keys_all):
    plt.subplot(1, 3, i+1)    
    plot_global_hemodynamic_variable(hemo_sum_sim_merged, key, 
                                     label=key,
                                     color='C0',
                                     marker='o',
                                     markersize=8*fontsize/16, 
                                     fontsize=fontsize)
    
    figure_make_up(title=titles_all[key], ylabel=ylabels_all[key], 
                   fontsize=fontsize, create_legend=False) 
    
plt.savefig(os.path.join(dir_out, 'rv_function.png'), dpi=300, bbox_inches="tight")
