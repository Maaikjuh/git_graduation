# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:18:51 2020

@author: Maaike
"""


from postprocessing_paraview_maaike import postprocess_paraview_new

import matplotlib.pyplot as plt
import os
import math

# --------------------------------------------------------------------------- #
# INPUTS
# --------------------------------------------------------------------------- #
# Specify two directories in which the paraview data can be found for two states (e.g. initial and adapted state).
# directory_1 = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\Biventricular model\default\cycle_5_stress_free_ref\paraview_data'
# directory_1 = r'C:/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/Biventricular model/08-04_15-16_save_coord_test/cycle_5_stress_free_ref/Paraview_data'

# directory_1 = r'C:/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/24-03_14-07_infarct_xi_10/cycle_2_stress_free_ref\paraview_data'
# directory_2 = r'C:/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/31-03_16-15_infarct_droplet_tao_20_meshres_30/cycle_2_stress_free_ref/Paraview_data'

# directory_1 = r'C:/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/leftventricular model/31-03_16-15_infarct_droplet_tao_20_meshres_30/cycle_2_stress_free_ref/Paraview_data'
# directory_2 = r'C:/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/leftventricular model/20-04_10-22_infarct_no_border_meshres_30/paraview_data'

directory_1 = r'C:/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/leftventricular model/24-03_14-07_infarct_xi_10/cycle_2_stress_free_ref\paraview_data'
directory_2 = r'C:/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/leftventricular model/20-04_10-22_infarct_no_border_meshres_30/cycle_2_stress_free_ref\paraview_data'

# directory_1 = r'C:/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/leftventricular model/21-04_16-13_with_fiber_reor_30/cycle_1_stress_free_ref\paraview_data'
# directory_2 = r'C:/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/leftventricular model/21-04_16-14_no_fiber_reor_30/cycle_1_stress_free_ref/paraview_data'

# directory_2 = r'C:/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/Biventricular model/20-04_14-23_ADAPTED_BAYER_2cyc_no_infarct/cycle_2_stress_free_ref/paraview_dat'
# directory_2 = r'C:/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/Biventricular model/20-04_11-24_ADAPTED_BAYER_2cyc_with_infarct/cycle_2_stress_free_ref/paraview data'

# directory_1 = r'C:/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/leftventricular model/01-05_08-38_fiber_reorientation_meshres_20/cycle_5_stress_free_ref/paraview_data'
# directory_2 = r"C:/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/leftventricular model/01-05_10-02_fiber_no_reorientation_meshres_20/cycle_5_stress_free_ref/paraview_data"

# Specify corresponding labels for the two directories above.
dir_labels = ['ref', 'infarct']

inputs1 = {  # Specify the thickness of the cube (thick longitudinal slice) with to be included nodes [cm].
          'load_pickle_file': True,
          'AM_phi': -1/4*math.pi,
          'A_phi': 0.,
          'AL_phi': 1/4*math.pi} 
 

inputs2 = { # Specify the thickness of the cube (thick longitudinal slice) with to be included nodes [cm].
          'load_pickle_file': True, # Reload the data from a temporary pickle file if available for fatser loading.
          'AM_phi': -1/4*math.pi,
          'A_phi': 0.,
          'AL_phi': 1/4*math.pi} 


# Plot options
fontsize = 13
linewidth = 2

# --------------------------------------------------------------------------- #

# common_dir = common_start(directory_1, directory_2)

post_1 = postprocess_paraview_new(directory_1, name='init', **inputs1)
post_2 = postprocess_paraview_new(directory_2, name='adap', **inputs2)

post_1.plot_rotation()
post_2.plot_rotation()

post_1.show_regions_new(projection = '2d', segments = True, fontsize=fontsize, skip=None)
post_2.show_regions_new(projection = '2d', fontsize=fontsize, skip=None)

fig_ta_ls_loop = plt.figure(figsize=(15, 15), dpi=100)
post_1.Ta_ls_loop('--',fig=fig_ta_ls_loop,phase = False, label=dir_labels[0])
post_2.Ta_ls_loop(fig=fig_ta_ls_loop, label=dir_labels[1])

# fig_ts = plt.figure(figsize=(22, 5), dpi=100)
# post_1.plot_time_stress('--',fig=fig_ts,phase = False, label=dir_labels[0])
# post_2.plot_time_stress(fig=fig_ts,label=dir_labels[1])

# # plt.savefig(os.path.join(directory_1, 'selected_regions.png'),  dpi=300, bbox_inches="tight")

# fig_sls = plt.figure(figsize=(22, 5), dpi=100)
# post_1.plot_stress_ls_l0('--',fig=fig_sls,label=dir_labels[0])
# post_2.plot_stress_ls_l0(fig=fig_sls,label=dir_labels[1])
# # fig_ss = plt.figure(figsize=(22, 5), dpi=100)
# # post_1.plot_stress_strain_ls(fig=fig_ss, label=dir_labels[0], fontsize=fontsize, linewidth=linewidth,var='strain')
# # post_2.plot_stress_strain('--', fig=fig_ss, label=dir_labels[1], fontsize=fontsize, linewidth=linewidth)
# plt.legend(frameon=False, fontsize=fontsize)
# # plt.savefig(os.path.join(directory_1, 'stress_strain_loops.png'),  dpi=300, bbox_inches="tight")

# fig_tls = plt.figure(figsize=(22, 5), dpi=100)
# post_1.plot_time_strain_ls('--', fig=fig_tls, reference='stress_free',var ='ls',phase = False, label=dir_labels[0])
# post_2.plot_time_strain_ls(fig=fig_tls, reference='stress_free',var ='ls',label=dir_labels[1])


# fig_pv = plt.figure(figsize=(8, 6), dpi=80)
# post_1.plot_pv_loops('--', color='C1', fig=fig_pv, label=dir_labels[0], fontsize=fontsize, linewidth=linewidth)
# post_2.plot_pv_loops(color='C0', fig=fig_pv, label=dir_labels[1], fontsize=fontsize, linewidth=linewidth)

# plt.legend(frameon=False, fontsize=fontsize)
# plt.savefig(os.path.join(directory_1, 'pv_loops.png'),  dpi=300, bbox_inches="tight")

# post_1.show_global_function()
# post_2.show_global_function()

# post_1.show_local_function(linewidth=linewidth)
# post_2.show_local_function(linewidth=linewidth)

# fontsize = 13
# post_1.plot_global_function_evolution(fontsize=fontsize, linewidth=linewidth)
# plt.savefig(os.path.join(directory_1, 'global_function_evolution.png'),  dpi=300, bbox_inches="tight")

print('')