# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 12:22:36 2018

@author: Hermans
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 13:01:01 2018

@author: Hermans
"""
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from postprocessing import (load_reduced_dataset, hemodynamic_summary,
                            print_hemodynamic_summary, 
                            find_simulation_hemodynamics)

rpm_1 = 7500
rpm_2 = 12000

results_ref = find_simulation_hemodynamics(rpm=rpm_1, simulation_type='BiV')
results_dis = find_simulation_hemodynamics(rpm=rpm_2, simulation_type='BiV')

print('\n1: {} RPM'.format(rpm_1))
data_ref = load_reduced_dataset(results_ref)
hemo_sum_ref = hemodynamic_summary(data_ref)
print_hemodynamic_summary(hemo_sum_ref)

print('\n2: {} RPM'.format(rpm_2))
data_dis = load_reduced_dataset(results_dis)
hemo_sum_dis = hemodynamic_summary(data_dis)
print_hemodynamic_summary(hemo_sum_dis)

for key in hemo_sum_ref:
    v_ref = hemo_sum_ref[key]
    if abs(v_ref) < 0.000001:
        continue
    v_dis = hemo_sum_dis[key]
    change = (v_dis-v_ref)/v_ref*100
    print('{}: {:.2f} %'.format(key, change))