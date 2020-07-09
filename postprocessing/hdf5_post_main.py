#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 14:58:02 2020

@author: maaike
"""

from hdf5_postprocess import *
import math

plt.close('all')

# directory_1 = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/leftventricular model/23-06_13-08_eikonal_td_1node/cycle_2_begin_ic_ref'
directory_1 = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/leftventricular model/01-07_15-48_eikonal_more_roots_mesh_20/cycle_2_begin_ic_ref'
directory_1 = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/Results/leftventricular model/Eikonal/06-07_15-56_ref_2_cycles_mesh_20/cycle_2_begin_ic_ref'


post_1 = postprocess_hdf5(directory_1)

post_1.plot_torsion(title = 'eikonal')
post_1.show_slices(title = 'eikonal')