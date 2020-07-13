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


post_1 = postprocess_hdf5(directory_1)
post_2 = postprocess_hdf5(directory_2)

strain_fig, strain_plot = plt.subplots(3,3, sharex = True, sharey = True)
stress_fig, stress_plot = plt.subplots(3,3, sharex = True, sharey = True)
work_fig, work_plot = plt.subplots(3,3, sharex = True, sharey = True)

post_1.loc_mech_ker(strain_figs = [strain_fig, strain_plot], stress_figs = [stress_fig, stress_plot], work_figs = [work_fig, work_plot], label = 'ref')
post_2.loc_mech_ker(strain_figs = [strain_fig, strain_plot], stress_figs = [stress_fig, stress_plot], work_figs = [work_fig, work_plot], label = 'eikonal')
# post_1.show_ker_points()
# post_2.show_ker_points()
# post_1.plot_torsion(title = 'ref')
# post_2.plot_torsion(title = 'eikonal')
# post_1.show_slices(title = 'ref')
# post_2.show_slices(title = 'eikonal')