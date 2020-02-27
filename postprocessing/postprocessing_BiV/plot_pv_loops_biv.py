# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:28:29 2018

@author: Hermans
"""

import matplotlib.pyplot as plt
import numpy as np
import os

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from postprocessing import (load_reduced_dataset, kPa_to_mmHg, 
                            find_simulation_hemodynamics,
                            get_rpm_colors, figure_make_up)
plt.close('all')

# --------------------------------------------------------------------------- #
# INPUTS
# --------------------------------------------------------------------------- #
# Specify output directory.
dir_out = r'figures'

# Specify the pump speeds to analyze. 
# If you specify -1, it will look for the healthy simulation.
rpm_all = [-1, 0, 7500, 8500, 9500, 10500, 11500, 12000]

# Specify the fontsize for plotting.
fontsize = 13

# --------------------------------------------------------------------------- #

# Create output directory if it doen not exists.
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

rpm_linestyles = {-1: '--', 0: "-", 7500: "-", 8500: "-", 9500:"-", 10500:'-', 11500: "-", 12000: "-"}
rpm_colors = get_rpm_colors()

# Initialize figure.
plt.figure(figsize=(12, 6))

ax1 = plt.subplot(1, 2, 1)
figure_make_up(title='RV', xlabel='Volume [ml]', ylabel='Pressure [mmHg]', create_legend=False, fontsize=fontsize)

ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
figure_make_up(title='LV', xlabel='Volume [ml]', ylabel='Pressure [mmHg]', create_legend=False, fontsize=fontsize)

plt.axis((22.407435842222068,
          147.27891449914796,
          -6.6545480832662012,
          117.39391243593957))

for rpm in rpm_all:
            
    # Load hemodynamics file.
    filename_simulation = find_simulation_hemodynamics(rpm=rpm, simulation_type='BiV')
    results = load_reduced_dataset(filename_simulation)
    
    # RV
    v_rv = np.array(results['vcav_p'])
    p_rv = kPa_to_mmHg(np.array(results['pcav_p'])) 

    # LV
    v_lv = np.array(results['vcav_s'])
    p_lv = kPa_to_mmHg(np.array(results['pcav_s'])) 
 
    ax1.plot(v_rv, p_rv, color=rpm_colors[rpm], 
             linestyle=rpm_linestyles[rpm],
             label='{} rpm'.format(rpm) if rpm != -1 else 'Healhty', 
             linewidth=fontsize/6)  
    
    ax2.plot(v_lv, p_lv, color=rpm_colors[rpm], 
             linestyle=rpm_linestyles[rpm],
             label='{} rpm'.format(rpm) if rpm != -1 else 'Healhty', 
             linewidth=fontsize/6)

    plt.sca(ax1)
    leg = plt.legend(fontsize=fontsize*0.9)
    
    # Save figure.
    figname = 'pv_loops'
    plt.savefig(os.path.join(dir_out, '{}_{}.png'.format(figname, rpm)), dpi=300, bbox_inches="tight")
      
plt.sca(ax1)
figure_make_up(title='RV', xlabel='Volume [ml]', ylabel='Pressure [mmHg]', legend_title='LVAD speed\n(rpm)', fontsize=fontsize)
plt.sca(ax2)
figure_make_up(title='LV', xlabel='Volume [ml]', ylabel='Pressure [mmHg]', create_legend=False, fontsize=fontsize)

# Save figure.
figname = 'pv_loops'
plt.savefig(os.path.join(dir_out, figname+'.png'), dpi=300, bbox_inches="tight")