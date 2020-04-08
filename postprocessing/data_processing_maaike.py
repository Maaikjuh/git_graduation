# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 10:57:33 2020

@author: Maaike
"""

import sys
sys.path.append(r'C:\Users\Maaike\Documents\Master\Graduation_project\git_graduation_project\cvbtk')
sys.path.append(r'C:\Users\Maaike\Documents\Master\Graduation_project\git_graduation_project\postprocessing')

from dataset import Dataset


from postprocess_Maaike import *
import os
import matplotlib.pyplot as plt
plt.close('all')

# Specify the results.csv file (which contains the hemodynamic data) directly:
#csv_normal = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\24-03_14-07_infarct_xi_10\results.csv'
#csv_infarct = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\31-03_16-15_infarct_droplet_tao_20_meshres_30\results.csv'

csv_normal = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\31-03_16-15_infarct_droplet_tao_20_meshres_30\results.csv'
csv_infarct = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\07-04_11-10_infarct_border_droplet_tao_0_meshres_30\results.csv'


ANALYZE = 'infarct'
COMPARE = True

# Load results.
results_normal = Dataset(filename=csv_normal)
results_infarct = Dataset(filename=csv_infarct)

# Take the last cycle, or specify the cycle directly
cycle = max(results_normal['cycle']) - 1
cycle=1

results_normal = results_normal[results_normal['cycle']==cycle]
results_infarct = results_infarct[results_infarct['cycle']==cycle]

# Data from the selected cycle
if ANALYZE == 'infarct':
    data_cycle = results_infarct
    pathname = csv_infarct
else:
    data_cycle = results_normal
    pathname = csv_normal

hemo_sum = hemodynamic_summary(data_cycle)
print_hemodynamic_summary(hemo_sum,cycle)

if COMPARE == True:
    plot_compare_results(results_normal,results_infarct, dir_out=os.path.dirname(csv_infarct), cycle=cycle)
else:
    plot_results(data_cycle,dir_out=os.path.dirname(pathname), cycle=cycle)
    

