# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 08:27:41 2020

@author: Maaike
"""

import pandas as pd
import math
import matplotlib.pyplot as plt
plt.close('all')


slice = 2

file_ref = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/15-09_new_vars_ref_cyc_5_res_30/cycle_5_begin_ic_ref/data.pkl'
file_ref_results = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/15-09_new_vars_ref_cyc_5_res_30/results.csv'

file_isch = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/11-09_15-22_new_vars_infarct_11_a0_inf_unchanged_res_20/cycle_2_begin_ic_ref/data.pkl'
file_isch_results = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/11-09_15-22_new_vars_infarct_11_a0_inf_unchanged_res_20/results.csv'

dat_ref = pd.read_pickle(file_ref)
dat_isch = pd.read_pickle(file_isch)

res_ref = pd.read_csv(file_ref_results)
res_ref = res_ref[res_ref['cycle'] == 5]
time_ref = res_ref['t_cycle']

res_isch = pd.read_csv(file_isch_results)
res_isch = res_isch[res_isch['cycle'] == 2]
time_isch = res_isch['t_cycle']

variables = ['Ecc', 'Ell', 'Err']


wall = int(max(dat_ref['wall']))
mid_wall = wall/2

slice_ref = dat_ref[dat_ref['slice'] == slice]
slice_isch = dat_isch[dat_isch['slice'] == slice]

seg_ref = slice_ref[slice_ref['seg'] == 4]
seg_isch = slice_isch[slice_isch['seg'] == 4]

wall_ref = seg_ref[seg_ref['wall'] == mid_wall]
wall_isch = seg_isch[seg_isch['wall'] == mid_wall]

fig_strain, strain_plots = plt.subplots(3,1)
strain_plots[0].set_ylabel('strain [%]')
strain_plots[0].tick_params(axis='x', labelbottom=False)
strain_plots[0].set_ylim([-0.2, 0.25])
strain_plots[1].set_ylabel('strain [%]')
strain_plots[1].tick_params(axis='x', labelbottom=False)
strain_plots[1].set_ylim([-0.2, 0.2])
strain_plots[2].set_ylabel('strain [%]')
strain_plots[2].set_xlabel('time [ms]')
strain_plots[2].set_ylim([-0.2, 0.6])

for ii, strain_name in enumerate(variables):
    strain_ref = wall_ref[wall_ref['strain'] == strain_name]
    strain_isch = wall_isch[wall_isch['strain'] == strain_name]
    
    grouped_time_ref = strain_ref.groupby('time', as_index=False).mean()
    grouped_time_isch = strain_isch.groupby('time', as_index=False).mean()
    
    time_strain_ref = grouped_time_ref['value']/100
    time_strain_isch = grouped_time_isch['value']/100
    
    strain_plots[ii].set_title(strain_name)
    strain_plots[ii].plot(time_ref, time_strain_ref, label='control', color = 'orange')
    strain_plots[ii].plot(time_isch, time_strain_isch, label='ischemic', color = 'k')
    
strain_plots[0].legend()
plt.show()
    