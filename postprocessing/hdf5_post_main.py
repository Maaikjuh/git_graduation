#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 14:58:02 2020

@author: maaike
"""

from hdf5_postprocess import *
import math

plt.close('all')

directory_1 = '/home/maaike/Documents/Graduation_project/Results/eikonal_td_1_node/cycle_2_begin_ic_ref'


post_1 = postprocess_hdf5(directory_1)

post_1.plot_torsion(title = 'eikonal')
post_1.show_slices(title = 'eikonal')