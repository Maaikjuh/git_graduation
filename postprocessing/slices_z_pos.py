# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:43:34 2020

@author: Maaike
"""
import pandas as pd

filename = r'C:\Users\Maaike\OneDrive - TU Eindhoven\Graduation_project\Results\leftventricular model\15-09_new_vars_ref_cyc_5_res_30\cycle_5_begin_ic_ref\data.pkl'
data = pd.read_pickle(filename)

max_slice = int(max(data['slice']))
for slice in range(0, max_slice + 1):
    dat = data['coords'][data['slice'] == slice]
    z = list(dat)[0][2]
    print('slice {}, z pos {}'.format(slice, z))