# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 10:12:50 2020

@author: Maaike
"""

import pandas as pd
import math
import matplotlib.pyplot as plt
plt.close('all')

file_ref = r'C:\Users\Maaike\OneDrive - TU Eindhoven\Graduation_project\Results\leftventricular model\ejection_time_tests\03-09_16-26_longer_act_rven_2_mesh_20\cycle_2_begin_ic_ref\data.pkl'
file_ref_results = r'C:\Users\Maaike\OneDrive - TU Eindhoven\Graduation_project\Results\leftventricular model\ejection_time_tests\03-09_16-26_longer_act_rven_2_mesh_20\results.csv'

dat_ref = pd.read_pickle(file_ref)
res_ref = pd.read_csv(file_ref_results)
cycle = int(max(res_ref['cycle']) - 1)
res_ref = res_ref[res_ref['cycle'] == cycle]

t_es_cycle_ref = res_ref['t_cycle'][(res_ref['phase'] == 4).idxmax()]
t_ed_cycle_ref = res_ref['t_cycle'][(res_ref['phase'] == 2).idxmax()]

dt = 2.
index_es = int((t_es_cycle_ref- t_ed_cycle_ref)/dt)
# dat_es_ref = dat_ref[dat_ref['time'] == index_ref]

theta_vals = [9/20*math.pi, 11/20*math.pi, 13/20*math.pi, 15/20*math.pi]

Ecc = dat_ref[dat_ref['strain'] == 'Ecc']
Ecr = dat_ref[dat_ref['strain'] == 'Ecr']

fig, ax = plt.subplots() 
ax.set_xlabel('Ecc [%]', fontsize = 14)
ax.set_ylabel('Ecr [%]', fontsize = 14)
ax.set_title('Ecr vs. Ecc', fontsize = 16)
ax.axvline(x = 0., color = 'k')
ax.axhline(y = 0., color = 'k')
for slice_nr in range(0, len(theta_vals)):
    slice_Ecc = Ecc[Ecc['slice'] == slice_nr]
    slice_Ecr = Ecr[Ecr['slice'] == slice_nr]
    
    dat_Ecc = slice_Ecc.groupby('time', as_index=False).mean()
    dat_Ecr = slice_Ecr.groupby('time', as_index=False).mean()
    
    ax.plot(dat_Ecc['value'][0:index_es], dat_Ecr['value'][0:index_es], label = 'Slice ' + str(slice_nr), linewidth = 2.)
ax.legend()
plt.show()
    
    