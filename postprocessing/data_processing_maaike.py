# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 10:57:33 2020

@author: Maaike
"""

import sys
sys.path.append(r'C:\Users\Maaike\Documents\Master\Graduation_project\git_graduation_project\cvbtk')
sys.path.append(r'C:\Users\Maaike\Documents\Master\Graduation_project\git_graduation_project\postprocessing')

from dataset import Dataset


from postprocess_Maaike import (hemodynamic_summary, print_hemodynamic_summary,
                            plot_results)

import os
import matplotlib.pyplot as plt
plt.close('all')

# Specify the results.csv file (which contains the hemodynamic data) directly:
csv_normal = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\24-03_14-07_infarct_xi_10\results.csv'
csv_infarct = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\01-04_14-51_infarct_border_droplet_tao_20_meshres_20\results.csv'

ANALYZE = 'infarct'
COMPARE = True

# Load results.
results_normal = Dataset(filename=csv_normal)
results_infarct = Dataset(filename=csv_infarct)

# Take the last cycle, or specify the cycle directly
cycle = max(results_normal['cycle']) - 1

# Data from the selected cycle
if ANALYZE == 'infarct':
    data_cycle = results_infarct[results_infarct['cycle']==cycle]
else:
    data_cycle = results_normal[results_normal['cycle']==cycle]

hemo_sum = hemodynamic_summary(data_cycle)
print_hemodynamic_summary(hemo_sum,cycle)

if COMPARE == True:
    plot_results(results_normal,results_infarct, dir_out=os.path.dirname(csv_normal), cycle=cycle)
else:
    plot_results(data_cycle,None, dir_out=os.path.dirname(csv_normal), cycle=cycle)
    

