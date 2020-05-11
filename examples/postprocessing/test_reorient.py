# -*- coding: utf-8 -*-
"""
Created on Thu May  7 09:47:10 2020

@author: Maaike
"""
import os
ROOT_DIRECTORY = os.path.join(
        os.path.abspath(__file__).split('\\Graduation_project\\')[0],
        'Graduation_project')
CVBTK_PATH = os.path.join(ROOT_DIRECTORY,'git_graduation_project\cvbtk')
import sys
sys.path.append(CVBTK_PATH)
from dataset import Dataset





initial = 'C:/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/leftventricular model/01-05_10-02_fiber_no_reorientation_meshres_20'
cycle_initial = 1
# results_initial = os.path.join(initial, 'results_cycle_{}.hdf5'.format(cycle_initial))

dir_orientation = "C:/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/leftventricular model/01-05_08-38_fiber_reorientation_meshres_20"
results_csv = os.path.join(dir_orientation, 'results.csv') 
full_csv = Dataset(filename=results_csv)
last_cycle = int(max(full_csv['cycle']) - 1)

path = os.path.join(dir_orientation, 'reorientation_angle.csv')


# dir_1 = '/home/maaike/model/examples/systemic_circulation/realcycle/output/01-05_10-02_fiber_no_reorientation_meshres_20/'
# cycle_1 = 5
# results_1 = os.path.join(dir_1, 'results_cycle_{}.hdf5'.format(cycle_1))