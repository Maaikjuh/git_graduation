# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 13:01:01 2018

@author: Hermans
"""
import sys

# Add higher directory to python path (so we can import from postprocessing.py)
sys.path.append("..") 

from postprocessing import (load_reduced_dataset, hemodynamic_summary, 
                            print_hemodynamic_summary, find_simulation_hemodynamics,
                            find_experiment_hemodynamics, Dataset)

hc = 1
rpm = 9500

filename_experiment = find_experiment_hemodynamics(hc=hc, rpm=rpm)
data_experiment = Dataset(filename=filename_experiment)

print('\nExperiment')
sum_experiment = hemodynamic_summary(data_experiment)
print_hemodynamic_summary(sum_experiment)    

filename_simulation = find_simulation_hemodynamics(hc=hc, rpm=rpm, simulation_type='lifetec_fit')
data_simulation = load_reduced_dataset(filename=filename_simulation)

print('\nSimulation')
sum_simulation = hemodynamic_summary(data_simulation)
print_hemodynamic_summary(sum_simulation)    