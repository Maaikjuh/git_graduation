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
csv = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\24-03_14-07_infarct_xi_10\results.csv'

# Load results.
results = Dataset(filename=csv)

# Take the last cycle, or specify the cycle directly
cycle = max(results['cycle']) - 1

# Data from the selected cycle
data_cycle = results[results['cycle']==cycle]

hemo_sum = hemodynamic_summary(data_cycle)
print_hemodynamic_summary(hemo_sum,cycle)
plot_results(results, dir_out=os.path.dirname(csv), cycle=cycle)