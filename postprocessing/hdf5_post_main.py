#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 14:58:02 2020

@author: maaike
"""

from hdf5_postprocess import *
import math

plt.close('all')

# directory_2 = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/Results/leftventricular model/Eikonal/23-06_13-08_eikonal_td_1node/cycle_2_begin_ic_ref'
# directory_2 = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/leftventricular model/01-07_15-48_eikonal_more_roots_mesh_20/cycle_2_begin_ic_ref'
directory_1 = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/Eikonal/06-07_15-56_ref(no_eikonal)_2_cycles_mesh_20/cycle_2_begin_ic_ref'
# directory_2 = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/Results/leftventricular model/Eikonal/15-07_10-26_eikonal_whole_endo_sig_equal_small_mesh_20/cycle_1_begin_ic_ref'
# directory_2 = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/beatit-master/beatit/' 
# directory_2 = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/Results/leftventricular model/geometry_tests/12-08_14-48_thin_wall_mesh_20/cycle_2_begin_ic_ref'
# directory_2 = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/Results/leftventricular model/geometry_tests/12-08_14-26_longer_mesh_20/cycle_2_begin_ic_ref'
# directory_2 = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/Results/leftventricular model/geometry_tests/13-08_15-20_longer_6_5_mesh_20/cycle_2_begin_ic_ref'
# directory_2 = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/Results/leftventricular model/geometry_tests/13-08_15-27_thick_wall_1_3_mesh_20/cycle_2_begin_ic_ref'
# directory_2 = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/Results/leftventricular model/geometry_tests/13-08_15-31_thin_wall_0_7_mesh_20/cycle_2_begin_ic_ref'
directory_1 = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/27_02_default_inputs/cycle_5_begin_ic_ref'
# directory_1 = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/mitral_flow_tests/02-09_10-04_res_pven_2_mesh_20/cycle_2_begin_ic_ref'
directory_1 = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/ischemic_model/09-06_13-47_ischemic_meshres_30/cycle_5_begin_ic_ref'

# directory_2 = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/ejection_time_tests/03-09_17-27_longer_act_rven_5_mesh_20/cycle_2_begin_ic_ref'
directory_2 = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ejection_time_tests/03-09_16-26_longer_act_rven_2_mesh_20/cycle_2_begin_ic_ref'

# directory_1 = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/ischemic_model/09-09_16-32_infarct_hdf5_higher_p_stress_2/cycle_1_begin_ic_ref'
# directory_2 = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/ischemic_model/09-09_16-50_infarct_hdf5_higher_p_stress_5/cycle_1_begin_ic_ref'

directory_0 = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/10-09_15-50_new_model_no_infarct_res_20/cycle_2_begin_ic_ref'
directory_1 = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/11-09_15-22_new_vars_infarct_11_a0_inf_unchanged_res_20/cycle_2_begin_ic_ref'
# directory_2 = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/14-09_09-33_new_vars_infarct_a0_inf_2_res_20/cycle_2_begin_ic_ref'
# directory_3 = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/14-09_09-31_new_vars_infarct_a0_inf_4_res_20/cycle_2_begin_ic_ref'

directory_1 = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/15-09_new_vars_ref_cyc_5_res_30/cycle_5_begin_ic_ref'

directory_2 = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/29-09_15-14_infarct_ref_a0_unchanged_res_30/cycle_5_begin_ic_ref'
directory_3 = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/07-10_08-07_infarct_ref_a0_4_res_30/cycle_5_begin_ic_ref'

directory_2 = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/07-10_08-16_infarct_endo_ref_a0_unchanged_res_30/cycle_5_begin_ic_ref'
directory_3 = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/07-10_08-20_infarct_endo_ref_a0_4_res_30/cycle_5_begin_ic_ref'

# dict_vals = {'theta_vals' : [4/10*math.pi, 5/10*math.pi, 6/10*math.pi, 7/10*math.pi, 8/10*math.pi]}
theta_vals = [9/20*math.pi, 11/20*math.pi, 13/20*math.pi, 15/20*math.pi]
theta_vals = [0.5*math.pi,  0.575*math.pi, 0.65*math.pi]
theta_vals = [0.65*math.pi]
# theta_vals = [0.5*math.pi]
dict_vals = {'theta_vals' : theta_vals, 'load_pickle': False}
# dict_vals = {'theta_vals' : [1/2*math.pi]}

label1 = 'healthy'
# label1 = 'ischemic, a0 unchanged'
label2 = 'transmural acute'
label3 = 'transmural chronic'

label2 = 'endocardial acute'
label3 = 'endocardial chronic'

# post_0 = postprocess_hdf5(directory_0, **dict_vals)
post_1 = postprocess_hdf5(directory_1, **dict_vals)
# post_2 = postprocess_hdf5(directory_2, model = 'beatit', **dict_vals)
post_2 = postprocess_hdf5(directory_2, **dict_vals)
post_3 = postprocess_hdf5(directory_3, **dict_vals)

# strain_fig, strain_plot = plt.subplots(len(theta_vals),3, sharex = True, sharey = True)
# stress_fig, stress_plot = plt.subplots(len(theta_vals),3, sharex = True, sharey = True)
# work_fig, work_plot = plt.subplots(len(theta_vals),3, sharex = True, sharey = True)

# post_1.plot_ls_lc(title = label1)
# post_0.loc_mech_ker(phi_wall = 0., strain_figs = [strain_fig, strain_plot], stress_figs = [stress_fig, stress_plot], work_figs = [work_fig, work_plot], label = label0)
# post_1.loc_mech_ker(strain_figs = [strain_fig, strain_plot], stress_figs = [stress_fig, stress_plot], work_figs = [work_fig, work_plot], label = label1)
# post_2.loc_mech_ker(phi_wall = 0.,strain_figs = [strain_fig, strain_plot], stress_figs = [stress_fig, stress_plot], work_figs = [work_fig, work_plot], label = label2)
# post_3.loc_mech_ker(phi_wall = 0.,strain_figs = [strain_fig, strain_plot], stress_figs = [stress_fig, stress_plot], work_figs = [work_fig, work_plot], label = label3)

# post_1.plot_torsion(title = label1)
# post_2.plot_torsion(title = label2)

# post_1.show_slices(wall_points=1)
# post_2.show_slices(title = 'beatit')
# post_2.show_slices(title = 'thinner')

fig, axs = plt.subplots(3,4, sharey = 'row')
post_1.loc_mech_bov_96(fig_axs = [fig, axs], label = label1)
post_2.loc_mech_bov_96(fig_axs = [fig, axs], label = label2, plot_phase = False)
post_3.loc_mech_bov_96(fig_axs = [fig, axs], label = label3, plot_phase = False)

# post_0.plot_strain(title = label0, analyse_seg = 4, variables = ['Ecc', 'Ell', 'Err',  'Ecr'])
post_1.plot_strain(title = label1, analyse_seg = 4, variables = ['Ecc', 'Ell', 'Ett', 'Err', 'Ecr'])
post_2.plot_strain(title = label2, analyse_seg = 4, variables = ['Ecc', 'Ell', 'Ett', 'Err', 'Ecr'])
post_3.plot_strain(title = label3, analyse_seg = 4, variables = ['Ecc', 'Ell', 'Ett', 'Err', 'Ecr'])
# post_2.plot_strain(title = label2)