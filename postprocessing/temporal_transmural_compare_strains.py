# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:42:52 2020

@author: Maaike
"""

import sys
sys.path.append(r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/git_graduation_project/cvbtk')
sys.path.append(r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/git_graduation_project/postprocessing')
import matplotlib.pyplot as plt
import pandas as pd
import os
from dataset import Dataset
plt.close('all')


file_healthy = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/15-09_new_vars_ref_cyc_5_res_30/cycle_5_begin_ic_ref/data.pkl'
file_isch_transmural = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/29-09_15-14_infarct_ref_a0_unchanged_res_30/cycle_5_begin_ic_ref/data.pkl'
file_isch_endocardial = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/07-10_08-16_infarct_endo_ref_a0_unchanged_res_30/cycle_5_begin_ic_ref/data.pkl'

fig_dir_out = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Thesis/Figures'

analyse_vars = [file_healthy, file_isch_transmural]
analyse_vars = [file_healthy, file_isch_endocardial]
analyse_vars = [file_healthy, file_isch_transmural, file_isch_endocardial]

label = ['healthy', 'transmural acute']
label = ['healthy', 'endocardial acute']
label = ['healthy', 'transmural acute', 'endocardial acute']

fig_save_title = 'ischemic_transm'
fig_save_title = 'ischemic_endo'
fig_save_title = 'ischemic_transm_endo'

colors = ['tab:blue', 'tab:green']
colors = ['tab:blue', 'tab:red']
colors = ['tab:blue', 'tab:green','tab:red']

variables = ['Ecc', 'Ell', 'Ett']
slice_nr = 0
analyse_seg = 4
wall_nr = 5
cycle = 5

fig_strain, strain_plots = plt.subplots(1,3, figsize=(13.42,  3.62))
strain_plots[0].set_ylabel('strain [-]')
strain_plots[0].set_xlabel('time [ms]')
# strain_plots[0].tick_params(axis='x', labelbottom=False)
strain_plots[0].set_ylim([-.2, .2])
# strain_plots[1].tick_params(axis='x', labelbottom=False)
strain_plots[1].set_ylim([-.2, .2])
# strain_plots[1].set_ylabel('strain [-]')
strain_plots[1].set_xlabel('time [ms]')
strain_plots[2].set_xlabel('time [ms]')
strain_plots[2].set_ylim([-.3, .7])
strain_plots = [strain_plots[0], strain_plots[1], strain_plots[2]]

fig_trans, trans_plots = plt.subplots(1,3, figsize=(13.42,  3.62))
trans_plots[0].set_ylabel('strain [-]')
trans_plots[0].set_xlabel('transmural depth [%]')
# trans_plots[0].tick_params(axis='x', labelbottom=False)
# trans_plots[1].tick_params(axis='x', labelbottom=False)
trans_plots[0].set_ylim([-.3, .3])
# trans_plots[1].set_ylabel('strain [-]')
trans_plots[1].set_ylim([-.3, .3])
trans_plots[1].set_xlabel('transmural depth [%]')
trans_plots[2].set_xlabel('transmural depth [%]')
trans_plots[2].set_ylim([-.2, 1.2])
# trans_plots[1,1].set_ylim([-.3, .3])
trans_plots = [trans_plots[0], trans_plots[1], trans_plots[2]]

for dat_nr, data in enumerate(analyse_vars):
    result_file = os.path.join(os.path.split(os.path.split(data)[0])[0],'results.csv') 
    full = Dataset(filename=result_file)
    results = full[full['cycle'] == cycle].copy(deep=True)
    
    t_ed_cycle = results['t_cycle'][(results['phase'] == 2).idxmax()]
    t_es_cycle = results['t_cycle'][(results['phase'] == 4).idxmax()]
    dt = 2.
    index = int((t_es_cycle- t_ed_cycle)/dt)

    time = results['t_cycle']
    phase2 = time[(results['phase'] == 1)[::-1].idxmax()] - t_ed_cycle
    phase3 = time[(results['phase'] == 2)[::-1].idxmax()] - t_ed_cycle
    phase4 = time[(results['phase'] == 3)[::-1].idxmax()] - t_ed_cycle
    phase4_end = time[(results['phase'] == 4)[::-1].idxmax()] - t_ed_cycle
    
    df = pd.read_pickle(data)
    data_es = df[df['time'] == index]
    
    for strain_count, strain_name in enumerate(variables):
        strain = df[df['strain'] == strain_name]
        strain = strain[strain['seg'] == float(analyse_seg)]
        slice = strain[strain['slice'] == slice_nr]
        wall = slice[slice['wall'] == wall_nr]
        
        grouped = wall.groupby('time', as_index=False).mean()
        time_strain = grouped['value']/100

        strain_plots[strain_count].plot(time, time_strain, color = colors[dat_nr], label = label[dat_nr])
        strain_plots[strain_count].set_title(strain_name)
        # strain_plots[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        strain_plots[len(strain_plots)-1].legend()

        if dat_nr == 0:
            for phase in [phase2, phase3, phase4, phase4_end]:
                strain_plots[strain_count].axvline(x = phase, linestyle = '--', color = 'k', linewidth = 0.5)
            # if strain_count == 0 or strain_count ==1:
            strain_plots[strain_count].annotate('ic', xy = (phase3/2 / 800 + 0.025, 0.92), xycoords='axes fraction')    
            strain_plots[strain_count].annotate('e', xy = ((phase3 + phase4)/2 / 800 + 0.01, 0.92), xycoords='axes fraction') 
            strain_plots[strain_count].annotate('ir', xy = ((phase4 + phase4_end)/2 / 800, 0.92), xycoords='axes fraction') 
            strain_plots[strain_count].annotate('d', xy = ((phase4_end + 800)/2 / 800, 0.92), xycoords='axes fraction') 

        
        strain = data_es[data_es['strain'] == strain_name]
        slice = strain[strain['slice'] == slice_nr]
        slice_wall = slice[slice['seg'] == float(analyse_seg)]
        grouped_wall = slice_wall.groupby('wall', as_index=False).mean()
        wall_strain = grouped_wall['value']/100
        
        trans_plots[strain_count].plot(range(0,101,10), wall_strain[::-1], color = colors[dat_nr], label = label[dat_nr])
        trans_plots[strain_count].set_title(strain_name)
        trans_plots[len(trans_plots)-1].legend()
fig_strain.savefig(os.path.join(fig_dir_out, 'temporal_strains_{}.png'.format(fig_save_title)), dpi=300, bbox_inches="tight")
fig_trans.savefig(os.path.join(fig_dir_out, 'transmural_strains_{}.png'.format(fig_save_title)), dpi=300, bbox_inches="tight")

        
        
        
    