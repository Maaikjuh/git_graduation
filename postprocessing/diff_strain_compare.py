# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 13:47:21 2020

@author: Maaike
"""

import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator 
import numpy as np
plt.close('all')

fontsize = 12

res_20_path = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/07-10_13-50_ref_cyc_10_res_20/strain_data.csv'
res_30_path = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/07-10_13-50_ref_cyc_10_res_30/strain_data.csv'
res_40_path = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/07-10_14-17_ref_cyc_10_res_40/strain_data.csv'

res_20 = pd.read_csv(res_20_path)
res_30 = pd.read_csv(res_30_path)
res_40 = pd.read_csv(res_40_path)

analyse_strains = ['Ecc', 'Ell', 'Err', 'Ecr']

print('Relative difference (%) at cycle 10 between res 20 and 40')
for strain in analyse_strains:
    res_20_strain = list(res_20[strain])[-1]

    res_40_strain = list(res_40[strain])[-1]
    diff = (res_20_strain-res_40_strain)
    perc = diff/res_40_strain*100
    print(strain + ' absolute: ' + str(round(diff,2)) + '     relative:'  + str(round(perc,2)))

fig_strain, strain_plots = plt.subplots(1,len(analyse_strains), figsize=(10,6))
strain_plots[0].set_ylabel('strain [%]', fontsize = fontsize)

for ax in range(0,len(analyse_strains)):
    strain_plots[ax].set_title(analyse_strains[ax], fontsize = fontsize + 2)
    strain_plots[ax].set_xlabel('Cardiac cycle', fontsize = fontsize)
    strain_plots[ax].spines['top'].set_visible(False)
    strain_plots[ax].spines['right'].set_visible(False)
    strain_plots[ax].xaxis.set_major_locator(MaxNLocator(integer=True))

colors = ['tab:blue','tab:red', 'tab:green', 'tab:orange']
for dat_file, data in enumerate([res_20, res_30, res_40]):
    for ii, strain_name in enumerate(analyse_strains):
        strain = data[strain_name]
        cycles = np.arange(1, len(strain)+1)
        strain_plots[ii].plot(cycles, strain, color = colors[dat_file])  

strain_plots[3].legend(['Simulation 1', 'Simulation 2', 'Simulation 3'], fontsize = fontsize)
# stepsize = 0.5
# start, end = strain_plots[1].get_ylim()
# strain_plots[1].yaxis.set_ticks(np.arange(start, end, stepsize))