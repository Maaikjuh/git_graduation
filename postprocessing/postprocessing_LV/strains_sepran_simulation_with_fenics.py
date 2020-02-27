# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:01:37 2018

@author: Hermans
"""
import os
import matplotlib.pyplot as plt

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from postprocessing import (StrainAnalysis)

plt.close('all')

# --------------------------------------------------------------------------- #
# INPUTS
# --------------------------------------------------------------------------- #
# Specify the filename to the curves file containing the strains.
filename_simulation = r'E:\Graduation Project\FEniCS\PycharmProject\output\sepran_simulation\sepran_simulation_res_30\cycle_10\curves.hdf5'

# Optionally, specify the filename to the curves file containing the strains 
# for the higher resolution simulation .
# (only needed if compare_resolutions == True)
filename_simulation_high_res = r'E:\Graduation Project\FEniCS\PycharmProject\output\sepran_simulation\sepran_simulation_res_40\cycle_6\curves.hdf5'

# Specify output directory.
dir_out = os.path.join(os.path.split(filename_simulation)[0])  

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
# Bool to enable/disable comparison between low and high resolution.
compare_resolutions = False

# Strain analysis options (see StrainAnalysis())
average_beats = True  # Include all beats (True)
average_mode = 1   # Include all segments (1) or optiomal segments (2)
average_type = 'mean'  # 'mean' or 'median'
plot_mode = 3  # Plot segments and average (1), plot average + IQR (2), plot average (3)
t_cycle = 800  # Specify the cycle time.
transmural_weight = False
strains_range = 'auto'
fontsize = 20  # Fontsize in plots.

# --------------------------------------------------------------------------- #

# Initiate strain analysis.
post_simulation = StrainAnalysis(filename_simulation, 
                      average_beats=average_beats, 
                      average_mode=average_mode, 
                      average_type=average_type,
                      plot_mode=plot_mode,
                      t_cycle=t_cycle,
                      transmural_weight=transmural_weight,
                      strains_range=strains_range)

# Plot simulation.
post_simulation.plot_strains(fontsize=fontsize, color='C0', label='reference')
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.grid('on')
    
plt.tight_layout()
plt.savefig(os.path.join(dir_out, 'strains.png'), dpi=300, bbox_inches="tight")

# Plot second simulation for comparison if requested.
if compare_resolutions:
    figname = os.path.split(filename_simulation)[0]
    
    post_simulation = StrainAnalysis(filename_simulation_high_res, 
                          average_beats=average_beats, 
                          average_mode=average_mode, 
                          average_type=average_type,
                          plot_mode=plot_mode,
                          t_cycle=t_cycle,
                          transmural_weight=transmural_weight,
                          strains_range=strains_range)
    
    post_simulation.plot_strains(fontsize=fontsize, figname=figname, linestyle='--', color='C1', label='high')
    plt.figure(figname)
    leg = plt.legend(fontsize=fontsize-2, title='Resolution')
    leg.get_title().set_fontsize(fontsize-2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(dir_out, 'strains_per_resolution.png'), dpi=300, bbox_inches="tight")
