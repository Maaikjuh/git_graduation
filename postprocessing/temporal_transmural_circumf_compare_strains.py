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
import numpy as np
import os
import math
from dataset import Dataset
# plt.close('all')

data = 'data_circumf'
# data = 'data_circm_fiber_stress'
# data = 'data_transm'
file_healthy = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/15-09_new_vars_ref_cyc_5_res_30/cycle_5_begin_ic_ref/{}.pkl'.format(data)
file_isch_transmural = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/29-09_15-14_infarct_ref_a0_unchanged_res_30/cycle_5_begin_ic_ref/{}.pkl'.format(data)
file_isch_transmural_chronic = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/07-10_08-07_infarct_ref_a0_4_res_30/cycle_5_begin_ic_ref/{}.pkl'.format(data)
file_isch_transmural_isotrope = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/04-11_07-57_a3_0_a0_unchanged_res_20/cycle_3_begin_ic_ref/data.pkl'
file_isch_transmural_chronic_isotrope = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/04-11_14-27_a3_0_a0_4_res_20/cycle_2_begin_ic_ref/data.pkl'

file_isch_endocardial = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/07-10_08-16_infarct_endo_ref_a0_unchanged_res_30/cycle_5_begin_ic_ref/data.pkl'
file_isch_endocardial_chronic = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/07-10_08-20_infarct_endo_ref_a0_4_res_30/cycle_5_begin_ic_ref/data.pkl'
fig_dir_out = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Thesis/Figures'

# analyse_vars = [file_healthy, file_isch_transmural]
# analyse_vars = [file_healthy, file_isch_endocardial]
# analyse_vars = [file_healthy, file_isch_transmural, file_isch_endocardial]
analyse_vars = [file_healthy, file_isch_transmural, file_isch_transmural_chronic]
# analyse_vars = [file_healthy, file_isch_transmural, file_isch_transmural_chronic, file_isch_transmural_isotrope, file_isch_transmural_chronic_isotrope]
# analyse_vars = [file_healthy, file_isch_endocardial, file_isch_endocardial_chronic]
# 
label = ['healthy', 'transmural acute']
label = ['healthy', 'endocardial acute']
label = ['healthy', 'transmural acute', 'endocardial acute']
label = ['healthy', 'transmural acute', 'transmural chronic', 'transmural acute isotrope', 'transmural chronic isotrope']
# label = ['healthy', 'endocardial acute', 'endocardial chronic']

fig_save_title = 'ischemic_transm'
fig_save_title = 'ischemic_endo'
fig_save_title = 'ischemic_transm_endo'
fig_save_title = 'ischemic_transm_acute_chronic'
# fig_save_title = 'ischemic_endo_acute_chronic'

colors = ['tab:blue', 'tab:green']
colors = ['tab:blue', 'tab:red']
colors = ['tab:blue', 'tab:green','tab:red']
colors = ['tab:blue', 'tab:green','tab:orange', 'tab:red']
# colors = ['tab:blue', 'tab:red', 'tab:orange']
colors = [None,None,None,None,None]

save_figs = False

variables = ['Ecc', 'Ell', 'Ett']
# variables = ['fiber_stress']
# variables = ['Ecc', 'Ell']
set_slice_nr = None
set_analyse_seg = None #if None, determine ischemic center segment from coordinates
set_wall_nr = None #if None, determine midwall nr from total wall nr (0: endocardial, (max): epicardial)
set_cycle = None #3
radius = 2.8029657313243685

fig_strain, strain_plots = plt.subplots(1,3, figsize=(13.42,  3.62))
strain_plots[0].set_ylabel('strain [-]')
strain_plots[0].set_xlabel('time [ms]')
strain_plots[0].set_ylim([-.2, .2])
strain_plots[1].set_ylim([-.2, .2])
strain_plots[1].set_xlabel('time [ms]')
strain_plots[2].set_xlabel('time [ms]')
strain_plots[2].set_ylim([-.3, .7])
strain_plots = [strain_plots[0], strain_plots[1], strain_plots[2]]

fig_trans, trans_plots = plt.subplots(1,3, figsize=(13.42,  3.62))
trans_plots[0].set_ylabel('strain [-]')
trans_plots[0].set_xlabel('transmural depth [%]')
trans_plots[0].set_ylim([-.3, .3])
trans_plots[1].set_ylim([-.3, .3])
trans_plots[1].set_xlabel('transmural depth [%]')
trans_plots[2].set_xlabel('transmural depth [%]')
trans_plots[2].set_ylim([-.2, 1.2])
trans_plots = [trans_plots[0], trans_plots[1], trans_plots[2]]

fig_circ, circ_plots = plt.subplots(1,3, figsize=(13.42,  3.62))
circ_plots[0].set_ylabel('strain [-]')
circ_plots[0].set_xlabel('distance from ischemic center [mm]')
circ_plots[0].set_ylim([-.2, .25])
circ_plots[1].set_ylim([-.2, .25])
circ_plots[1].set_xlabel('distance from ischemic center [mm]')
circ_plots[2].set_xlabel('distance from ischemic center [mm]')
circ_plots[2].set_ylim([-.2, .6])
circ_plots = [circ_plots[0], circ_plots[1], circ_plots[2]]

df_circumf = {}
count = 0
save_circ_dat = {'Ecc': [], 'Ell':[], 'Ett': []}
for dat_nr, data in enumerate(analyse_vars):
    #find and read results file
    result_file = os.path.join(os.path.split(os.path.split(data)[0])[0],'results.csv') 
    full = Dataset(filename=result_file)
    
    #if cycle is None, find last cycle
    if set_cycle == None:
        cycle = int(max(full['cycle']) - 1)
    else:
        cycle = set_cycle
    results = full[full['cycle'] == cycle].copy(deep=True)
    
    #find index of end diastole and end systole in the data.pkl 
    t_ed_cycle = results['t_cycle'][(results['phase'] == 2).idxmax()]
    t_es_cycle = results['t_cycle'][(results['phase'] == 4).idxmax()]
    dt = 2.
    index = int((t_es_cycle- t_ed_cycle)/dt)

    time = results['t_cycle']
    #find times of cycles, temporal begins at end diastole
    phase2 = time[(results['phase'] == 1)[::-1].idxmax()] - t_ed_cycle
    phase3 = time[(results['phase'] == 2)[::-1].idxmax()] - t_ed_cycle
    phase4 = time[(results['phase'] == 3)[::-1].idxmax()] - t_ed_cycle
    phase4_end = time[(results['phase'] == 4)[::-1].idxmax()] - t_ed_cycle
    
    #read data.pkl file
    df = pd.read_pickle(data)
    #data at end systole
    data_es = df[df['time'] == index]

    #get data at the epicardium and find mid wall point
    wall_epi = int(max(df['wall']))
    wall_df = df[df['wall'] == wall_epi]
    
    #if set_wall_nr is not set, get mid wall point
    if set_wall_nr == None:
        wall_nr = int(wall_epi/2)
    else:
        wall_nr = set_wall_nr
       
    # if set_slice_nr is not set, get last slice
    if set_slice_nr == None:
        slice_nr = int(max(df['slice']))
    else:
        slice_nr = set_slice_nr
    
    #if set_analyse_seg is not set, find segment of ischemic center at the epicardium (assume x is maximum)
    if set_analyse_seg == None:
        coords_vals = wall_df.groupby('x', as_index=False).mean()
        list_x_vals = list(coords_vals['x'])
        
        #assume center ichemic area is where x is maximum (y should be minimum)
        index = list_x_vals.index(max(coords_vals['x']))
        analyse_seg = coords_vals['seg'][index]
    else:
        analyse_seg = set_analyse_seg

    # calculate radial angle between two subsequent segments
    # to calculate the distance between all segments
    wall_seg_0_x = list(wall_df[wall_df['seg'] == 0]['x'])[0]
    wall_seg_0_y = list(wall_df[wall_df['seg'] == 0]['y'])[0]
    wall_seg_1_x = list(wall_df[wall_df['seg'] == 1]['x'])[0]
    wall_seg_1_y = list(wall_df[wall_df['seg'] == 1]['y'])[0]

        
    seg_0 = ([wall_seg_0_x, wall_seg_0_y])
    seg_1 = ([wall_seg_1_x, wall_seg_1_y])
    
    length1 =  math.sqrt(np.dot(seg_0,seg_0))
    length2 =  math.sqrt(np.dot(seg_1,seg_1))
    cos_angle = np.dot(seg_0, seg_1) / (length1 * length2)
    rad_angle = np.arccos(cos_angle)      

    omtrek = 2*np.pi*radius
    seg_length =  rad_angle/(2*np.pi) * omtrek
    dis_range = []
    for seg in range(0, int(max(df['seg']))+1):
        dis_seg = (seg - analyse_seg) * seg_length * 10
        dis_range.append(dis_seg)        
    
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
        strain_plots[len(strain_plots)-1].legend(loc='lower right')

        if dat_nr == 0:
            for phase in [phase2, phase3, phase4, phase4_end]:
                strain_plots[strain_count].axvline(x = phase, linestyle = '--', color = 'k', linewidth = 0.5)
            # if strain_count == 0 or strain_count ==1:
            strain_plots[strain_count].annotate('ic', xy = (phase3/2 / 800 + 0.025, 0.92), xycoords='axes fraction')    
            strain_plots[strain_count].annotate('e', xy = ((phase3 + phase4)/2 / 800 + 0.01, 0.92), xycoords='axes fraction') 
            strain_plots[strain_count].annotate('ir', xy = ((phase4 + phase4_end)/2 / 800, 0.92), xycoords='axes fraction') 
            strain_plots[strain_count].annotate('f', xy = ((phase4_end + 800)/2 / 800, 0.92), xycoords='axes fraction') 

        
        strain = data_es[data_es['strain'] == strain_name]
        slice = strain[strain['slice'] == slice_nr]
        slice_wall = slice[slice['seg'] == float(analyse_seg)]
        grouped_wall = slice_wall.groupby('wall', as_index=False).mean()
        wall_strain = grouped_wall['value']/100
        
        # save_trans_dat[strain_name].append(wall_strain)
        
        step = 100/len(wall_strain)
        
        trans_plots[strain_count].plot(np.arange(0.,100., step), wall_strain[::-1], color = colors[dat_nr], label = label[dat_nr])
        trans_plots[strain_count].set_title(strain_name)
        trans_plots[len(trans_plots)-1].legend()
        
        strain = strain[strain['wall'] == wall_epi]
               
        grouped_seg = strain.groupby('seg', as_index=False).mean()
        seg_strain = grouped_seg['value']/100
        # rad_angle_between_segs = 0.7853981633974481
        # plot_dis_range = dis_range[0::10]
        # plot_seg_strain = seg_strain[0::10]
        
        av_strain = []
        av_range = []
        
        for step in range(20):
            sub_strain = seg_strain[step *10 : (step+1)*10]
            sub_range = dis_range[step *10 : (step+1)*10]
            av_strain.append(np.mean(sub_strain))
            av_range.append(np.mean(sub_range))
        circ_plots[strain_count].plot(av_range, av_strain)
        
        # save_circ_dat[strain_name].append(seg_strain)
        # circ_plots[strain_count].plot(range(0, int(max(df['seg']))+1), seg_strain, color = colors[dat_nr], label = label[dat_nr])
        # circ_plots[strain_count].plot(dis_range, seg_strain, color = colors[dat_nr], label = label[dat_nr])
        circ_plots[strain_count].axvline(x = 0, linestyle = '-', color = 'k', linewidth = 0.7)
        
        if 'T0' in strain:
            T0_dict = strain[strain['T0'] < 30]
            min_T0_seg = min(T0_dict['seg'])
            max_T0_seg = max(T0_dict['seg'])
            
            dis_min_T0 = (min_T0_seg - analyse_seg) * seg_length * 10
            dis_max_T0 = (max_T0_seg - analyse_seg) * seg_length * 10
            
            circ_plots[strain_count].axvline(x = dis_min_T0, linestyle = '--', dashes=(5, 10), color = 'k', linewidth = 0.5)
            circ_plots[strain_count].axvline(x = dis_max_T0, linestyle = '--', dashes=(5, 10), color = 'k', linewidth = 0.5)
        
        circ_plots[strain_count].set_title(strain_name)
        circ_plots[len(circ_plots)-1].legend()
        
        # df_circumf[count] = {'data_nr': dat_nr, 'strain_name': strain_name, 'circ_strain': seg_strain}
        # count += 1

if save_figs == True:   
    fig_strain.savefig(os.path.join(fig_dir_out, 'temporal_strains_{}.png'.format(fig_save_title)), dpi=300, bbox_inches="tight")
    fig_trans.savefig(os.path.join(fig_dir_out, 'transmural_strains_{}.png'.format(fig_save_title)), dpi=300, bbox_inches="tight")
    fig_circ.savefig(os.path.join(fig_dir_out, 'circumferential_strains_{}.png'.format(fig_save_title)), dpi=300, bbox_inches="tight")

        
        
        
    