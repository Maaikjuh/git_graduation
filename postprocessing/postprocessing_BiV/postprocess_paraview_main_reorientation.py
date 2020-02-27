# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:39:27 2018

@author: Hermans
"""

from postprocessing_paraview import common_start, postprocess_paraview

import matplotlib.pyplot as plt
import os

# --------------------------------------------------------------------------- #
# INPUTS
# --------------------------------------------------------------------------- #
# Specify two directories in which the paraview data can be found for two states (e.g. initial and adapted state).
directory_1 = r'E:\Graduation Project\FEniCS\PycharmProject\output\biv_realcycle\reorientation\REF\new_mesh_reorientation_3\cycle_6\paraview_data'
directory_2 = r'E:\Graduation Project\FEniCS\PycharmProject\output\biv_realcycle\reorientation\REF\new_mesh_reorientation_4\cycle_33\paraview_data'  

# Specify corresponding labels for the two directories above.
dir_labels = ['init', 'adap']

inputs = {'slice_thickness': 1,  # Specify the thickness of the cube (thick longitudinal slice) with to be included nodes [cm].
          'load_pickle_file': True}  # Reload the data from a temporary pickle file if available for fatser loading.

# Plot options
fontsize = 13
linewidth = 2

# --------------------------------------------------------------------------- #

common_dir = common_start(directory_1, directory_2)

post_1 = postprocess_paraview(directory_1, name='init', **inputs)
post_2 = postprocess_paraview(directory_2, name='adap', **inputs)

post_1.show_regions(projection = '2d', fontsize=fontsize, skip=None)

plt.savefig(os.path.join(common_dir, 'selected_regions.png'),  dpi=300, bbox_inches="tight")

fig_ss = plt.figure(figsize=(22, 5), dpi=100)
post_1.plot_stress_strain(fig=fig_ss, label=dir_labels[0], fontsize=fontsize, linewidth=linewidth)
post_2.plot_stress_strain('--', fig=fig_ss, label=dir_labels[1], fontsize=fontsize, linewidth=linewidth)
plt.legend(frameon=False, fontsize=fontsize)
plt.savefig(os.path.join(common_dir, 'stress_strain_loops.png'),  dpi=300, bbox_inches="tight")

fig_pv = plt.figure(figsize=(8, 6), dpi=80)
post_1.plot_pv_loops(color='C0', fig=fig_pv, label=dir_labels[0], fontsize=fontsize, linewidth=linewidth)
post_2.plot_pv_loops('--', color='C1', fig=fig_pv, label=dir_labels[1], fontsize=fontsize, linewidth=linewidth)
plt.legend(frameon=False, fontsize=fontsize)
plt.savefig(os.path.join(common_dir, 'pv_loops.png'),  dpi=300, bbox_inches="tight")

post_1.show_global_function()
post_2.show_global_function()

post_1.show_local_function(linewidth=linewidth)
post_2.show_local_function(linewidth=linewidth)

fontsize = 13
post_1.plot_global_function_evolution(fontsize=fontsize, linewidth=linewidth)
plt.savefig(os.path.join(common_dir, 'global_function_evolution.png'),  dpi=300, bbox_inches="tight")

print('')