# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 21:11:27 2018

@author: Hermans
"""

import matplotlib.pyplot as plt
import numpy as np
import os

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from postprocessing import (load_reduced_dataset, kPa_to_mmHg, 
                            find_simulation_hemodynamics,
                            get_rpm_colors)
plt.close('all')

# --------------------------------------------------------------------------- #
# INPUTS
# --------------------------------------------------------------------------- #
# Specify output directory.
dir_out = r'figures'

# Specify fontsize in plots.
fontsize = 16

# Specify 'lifetec_fit' or 'fixed_preload'.
simulation_type = 'lifetec_fit'

# --------------------------------------------------------------------------- #

# Names for the heart conditions.
hc_names = {1: 'REF',
            3: 'DEG'}

# Plot pv loops of both heart conditions and all rpms.
hc_all = [1, 3]
rpm_all = [0, 7500, 8500, 9500, 10500]

if simulation_type == 'fixed_preload':
    # Automatically add a post fix if we analyze the fixed_preload simulations.
    dir_out += '_fixed_preload' 
    
# Create output directory if it doesn't exists.
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

# Load default rpm colors.
rpm_colors = get_rpm_colors()

plt.figure(figsize=(12, 6))
for i, hc in enumerate(hc_all):

    if i == 0:
        ax1 = plt.subplot(1, len(hc_all), i+1)
    else:
        plt.subplot(1, len(hc_all), i+1, sharex=ax1, sharey=ax1)

    title = hc_names[hc] + '*'*(simulation_type == 'fixed_preload')
    plt.title(title, fontsize=fontsize+2)
    plt.xlabel('LV volume [ml]', fontsize=fontsize)
    plt.ylabel('LV pressure [mmHg]', fontsize=fontsize)
    plt.tick_params(labelsize=fontsize-2)
    
    for j, rpm in enumerate(rpm_all):
                
        filename_simulation = find_simulation_hemodynamics(hc, rpm, simulation_type=simulation_type)
        
        results = load_reduced_dataset(filename_simulation)
        v_lv = np.array(results['vlv'])
        p_lv = kPa_to_mmHg(np.array(results['plv']))

        plt.plot(v_lv, p_lv, color=rpm_colors[rpm], 
                 label=rpm, linewidth=fontsize/6)
        
#        # Mark point of reference state for strains.
#        idx_ref = np.where(results['phase']==2)[0][0]
#        plt.scatter(v[idx_ref], p[idx_ref], color=rpm_colors[rpm],
#                    linewidth=fontsize/6)

leg = ax1.legend(title='LVAD speed\n(rpm)', fontsize=fontsize-2)
leg.get_title().set_fontsize(fontsize-2)

plt.axis((44.637227439832735,
          164.96095903123637,
          -5.760204485254576,
          114.81117822294604))

# Save figure.
figname = 'pv_loops'+('_fixed_preload' if simulation_type == 'fixed_preload' else '_tuned_preload')
plt.tight_layout()
plt.savefig(os.path.join(dir_out, figname+'.png'), dpi=300, bbox_inches="tight")