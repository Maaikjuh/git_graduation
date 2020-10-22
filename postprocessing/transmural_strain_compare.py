# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 16:17:18 2020

@author: Maaike
"""
import pandas as pd
import math
import matplotlib.pyplot as plt
plt.close('all')

file_ref = r'C:\Users\Maaike\OneDrive - TU Eindhoven\Results\leftventricular model\27_02_default_inputs\cycle_5_begin_ic_ref\data.pkl'
file_ref_results = r'C:\Users\Maaike\OneDrive - TU Eindhoven\Results\leftventricular model\27_02_default_inputs\results.csv'

file_isch = r'C:\Users\Maaike\OneDrive - TU Eindhoven\Results\leftventricular model\ischemic_model\09-06_13-47_ischemic_meshres_30\cycle_5_begin_ic_ref\data.pkl'
file_isch_results = r'C:\Users\Maaike\OneDrive - TU Eindhoven\Results\leftventricular model\ischemic_model\09-06_13-47_ischemic_meshres_30\results.csv'

dat_ref = pd.read_pickle(file_ref)
dat_isch = pd.read_pickle(file_isch)

res_ref = pd.read_csv(file_ref_results)
res_ref = res_ref[res_ref['cycle'] == 5]

res_isch = pd.read_csv(file_isch_results)
res_isch = res_isch[res_isch['cycle'] == 5]

variables = ['Ecc', 'Ell', 'Err',  'Ecr']
theta_vals = [9/20*math.pi, 11/20*math.pi, 13/20*math.pi, 15/20*math.pi]
wall_points = 10

dt = 2.
t_es_cycle_ref = res_ref['t_cycle'][(res_ref['phase'] == 4).idxmax()]
t_es_cycle_isch = res_isch['t_cycle'][(res_isch['phase'] == 4).idxmax()]

t_ed_cycle_ref = res_ref['t_cycle'][(res_ref['phase'] == 2).idxmax()]
t_ed_cycle_isch = res_isch['t_cycle'][(res_isch['phase'] == 2).idxmax()]

index_ref = int((t_es_cycle_ref- t_ed_cycle_ref)/dt)
index_isch = int((t_es_cycle_isch- t_ed_cycle_isch)/dt)

dat_es_ref = dat_ref[dat_ref['time'] == index_ref]
dat_es_isch = dat_isch[dat_isch['time'] == index_isch]

for strain_name in variables:
    fig = plt.figure()
    strain_ref = dat_es_ref[dat_es_ref['strain'] == strain_name]
    strain_isch = dat_es_isch[dat_es_isch['strain'] == strain_name]
    
    seg_ref = strain_ref[strain_ref['seg'] == 4]
    seg_isch = strain_isch[strain_isch['seg'] == 4]
    
    grouped_wall_ref = seg_ref.groupby('wall', as_index=False).mean()
    grouped_wall_isch = seg_isch.groupby('wall', as_index=False).mean()
    
    wall_strain_ref = grouped_wall_ref['value']
    wall_strain_isch = grouped_wall_isch['value']
    
    plt.plot(range(wall_points+1), wall_strain_ref[::-1], label='ref')
    plt.plot(range(wall_points+1), wall_strain_isch[::-1], label='ischemic')
    
    plt.title(strain_name + ' reference and ischemic segment 4')
    
    plt.xlabel('epicaridal   -    endocardial')
    plt.ylabel('strain [%]')
    plt.legend()
    plt.show()
    
    
    