# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 12:52:01 2018

@author: Hermans
"""

import matplotlib.pyplot as plt
import numpy as np

import sys
# Add higher directory to python path (so we can import from postprocessing.py)
sys.path.append("..") 
from postprocessing import  (subplot_comparisons, merge_dictionaries,
                             plot_global_hemodynamic_variable, compute_IQ)

import os

plt.close('all')

# --------------------------------------------------------------------------- #
# INPUTS
# --------------------------------------------------------------------------- #
# Specify output directory.
dir_out = r'figures'

# Specify fontsize in plots.
fontsize = 20

# Specify 'lifetec_fit' or 'fixed_preload'.
simulation_type = 'lifetec_fit'

# Specify 'compare', 'experiment', or 'simulation'.
plot_mode = 'compare'

# --------------------------------------------------------------------------- #

if simulation_type == 'fixed_preload':
    # Automatically add a post fix if we analyze the fixed_preload simulations.
    dir_out += '_fixed_preload' 

# Create output directory if it doesn't exists.
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

# Comparison plot 1 (both 0 rpm simulations)
hc_all = [1, 3]
rpm_all = [0, 0]

hemo_sum_sim_merged_1, hemo_sum_exp_merged_1 = subplot_comparisons(hc_all, rpm_all, 
                    fontsize=fontsize, 
                    save_to_filename=os.path.join(dir_out, 'hemo_0_rpm_{}.png'.format(plot_mode)),
                    simulation_type=simulation_type, plot_mode=plot_mode)

# Comparison plot 2 (all REF simulations)
hc_all = [1, 1, 1, 1, 1]
rpm_all = [0, 7500, 8500, 9500, 10500]

hemo_sum_sim_merged_2, hemo_sum_exp_merged_2 = subplot_comparisons(hc_all, rpm_all, 
                    fontsize=fontsize, 
                    save_to_filename=os.path.join(dir_out, 'hemo_REF_{}.png'.format(plot_mode)),
                    simulation_type=simulation_type, plot_mode=plot_mode)

# Comparison plot 3 (all DEG simulations)
hc_all = [3, 3, 3, 3, 3]
rpm_all = [0, 7500, 8500, 9500, 10500]

hemo_sum_sim_merged_3, hemo_sum_exp_merged_3 = subplot_comparisons(hc_all, rpm_all, 
                    fontsize=fontsize, 
                    save_to_filename=os.path.join(dir_out, 'hemo_DEG_{}.png'.format(plot_mode)),
                    simulation_type=simulation_type, plot_mode=plot_mode)

# Merge the data of the LVAD experiments and simulation.
merge_dictionaries(hemo_sum_sim_merged_2, hemo_sum_sim_merged_3)
merge_dictionaries(hemo_sum_exp_merged_2, hemo_sum_exp_merged_3)

# Plot hemodynamic indices per pump speed.
keys_all = ['CO', 'LVAD_flow', 'LVAD_frac', 'plv_max', 'MAP', 'dpdt_max']#, 'PP']
titles_all = ['Cardiac output', 'LVAD flow', 'LVAD contribution', 
              'Maximum LV pressure', 'Mean arterial pressure', 
              '$dp/dt_{max}$']#, 'Pulse pressure']
ylabels_all = ['$\overline{q}_{art}$ [l/min]', '$\overline{q}_{lvad}$ [l/min]',
               'Flow fraction [%]', '$p_{lv,max}$ [mmHg]',
               '$\overline{p}_{art}$ [mmHg]', '$dp/dt_{max}$ [mmHg/s]']#, 'PP [mmHg]']

ylim_all = [(4.4589453441034097, 6.1446525795987741),
            (-0.33341009892648577, 6.3728599816478173),
            (-5.6252386078327747, 105.02977326703966),
            (58.699573428614222, 112.16002031292314),
            (66.536721960186426, 91.661693399888222),
            (503.3173239480152, 1178.3361970917258)]

plt.figure(figsize=(20, 10))
nrows = np.round(np.sqrt(len(keys_all)))
ncols = np.ceil(np.sqrt(len(keys_all)))
for i, key in enumerate(keys_all):
    if i == 0:
        ax1 = plt.subplot(nrows, ncols, i+1)
    elif i == 1:
        plt.subplot(nrows, ncols, i+1, sharey=ax1)
    else:
        plt.subplot(nrows, ncols, i+1)
    
    if simulation_type == 'lifetec_fit': 
        # Compare 'lifetec_fit' simulations to experiments, 
        # so plot experiment results.
        plot_global_hemodynamic_variable(hemo_sum_exp_merged_2, key, 
                                         label_prefix='Experiment ', fontsize=fontsize,
                                         linestyle='-', color='C0')
    
    plot_global_hemodynamic_variable(hemo_sum_sim_merged_2, key, 
                                     label_prefix='Simulation ', fontsize=fontsize,
                                     linestyle='--', color='C1')
    
    plt.title(titles_all[i], fontsize=fontsize+2)
    plt.ylabel(ylabels_all[i], fontsize=fontsize)
    
    plt.ylim(ylim_all[i])
    
    if i + ncols < len(keys_all):
        # Remove x-label for non-bottom plots.
        plt.xlabel('')

plt.subplot(nrows, ncols, 1)    

# Create legend.
leg = plt.legend(fontsize=fontsize*0.9)
for legobj in leg.legendHandles:
    legobj.set_linewidth(fontsize/7)

if simulation_type == 'fixed_preload':
    # Add * to labels.
    labels = leg.get_texts()
    for i in range(len(labels)):
        lab = labels[i].get_text()
        if 'Simulation' in lab:
            labels[i].set_text(lab+'*')

plt.tight_layout()
plt.savefig(os.path.join(dir_out, 'hemo_summary.png'), dpi=300, bbox_inches="tight")
