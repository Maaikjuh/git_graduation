#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 14:58:02 2020

@author: maaike
"""

from hdf5_postprocess import *
import math

plt.close('all')

directory_2 = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/Results/leftventricular model/Eikonal/23-06_13-08_eikonal_td_1node/cycle_2_begin_ic_ref'
# directory_2 = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/leftventricular model/01-07_15-48_eikonal_more_roots_mesh_20/cycle_2_begin_ic_ref'
directory_1 = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/Results/leftventricular model/Eikonal/06-07_15-56_ref(no_eikonal)_2_cycles_mesh_20/cycle_2_begin_ic_ref'
# directory_2 = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/Results/leftventricular model/Eikonal/15-07_10-26_eikonal_whole_endo_sig_equal_small_mesh_20/cycle_1_begin_ic_ref'
# directory_2 = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/beatit-master/beatit/' 
directory_2 = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/Results/leftventricular model/geometry_tests/12-08_14-48_thin_wall_mesh_20/cycle_2_begin_ic_ref'
# directory_2 = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/Results/leftventricular model/geometry_tests/12-08_14-26_longer_mesh_20/cycle_2_begin_ic_ref'
# directory_2 = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/Results/leftventricular model/geometry_tests/13-08_15-20_longer_6_5_mesh_20/cycle_2_begin_ic_ref'
# directory_2 = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/Results/leftventricular model/geometry_tests/13-08_15-27_thick_wall_1_3_mesh_20/cycle_2_begin_ic_ref'
# directory_2 = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/Results/leftventricular model/geometry_tests/13-08_15-31_thin_wall_0_7_mesh_20/cycle_2_begin_ic_ref'


dict_vals = {'theta_vals' : [4/10*math.pi, 5/10*math.pi, 6/10*math.pi, 7/10*math.pi, 8/10*math.pi]}
dict_vals = {'theta_vals' : [9/20*math.pi, 11/20*math.pi, 13/20*math.pi, 15/20*math.pi]}
# dict_vals = {'theta_vals' : [1/2*math.pi]}

post_1 = postprocess_hdf5(directory_1, **dict_vals)
# post_2 = postprocess_hdf5(directory_2, model = 'beatit', **dict_vals)
post_2 = postprocess_hdf5(directory_2, **dict_vals)

strain_fig, strain_plot = plt.subplots(3,3, sharex = True, sharey = True)
stress_fig, stress_plot = plt.subplots(3,3, sharex = True, sharey = True)
work_fig, work_plot = plt.subplots(3,3, sharex = True, sharey = True)

post_1.loc_mech_ker(strain_figs = [strain_fig, strain_plot], stress_figs = [stress_fig, stress_plot], work_figs = [work_fig, work_plot], label = 'ref')
post_2.loc_mech_ker(strain_figs = [strain_fig, strain_plot], stress_figs = [stress_fig, stress_plot], work_figs = [work_fig, work_plot], label = 'thinner')
# post_1.show_ker_points()
post_2.show_ker_points()
# post_1.plot_torsion(title = 'ref')
post_2.plot_torsion(title = 'thinner')

# post_1.calc_strain()
post_2.calc_strain(title = 'thinner')

# post_1.show_slices(title = 'ref')
# post_2.show_slices(title = 'beatit')
post_2.show_slices(title = 'thinner')

# post_1.plot_strain()
post_2.plot_strain(title = 'thinner')