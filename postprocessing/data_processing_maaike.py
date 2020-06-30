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
#plt.close('all')

# Specify the results.csv file (which contains the hemodynamic data) directly:
#csv_normal = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\24-03_14-07_infarct_xi_10\results.csv'
#csv_infarct = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\31-03_16-15_infarct_droplet_tao_20_meshres_30\results.csv'

csv_ref = r'C:/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/leftventricular model/24-03_14-07_infarct_xi_10/results.csv'
#csv_infarct = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\09-04_15-25_big_border_meshres_20\results.csv'

# csv_ref = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\leftventricular model\27_02_default_inputs\results.csv'
# csv_infarct = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\Biventricular model\default\results.csv'

#csv_ref = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\Biventricular model\default\results.csv'
#csv_infarct = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\Biventricular model\08-04_15-16_save_coord_test\results.csv'

# csv_ref = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\Biventricular model\default\results.csv'
# csv_infarct = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\Biventricular model\20-04_14-23_ADAPTED_BAYER_2cyc_no_infarct\results.csv'

# csv_ref = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\Biventricular model\20-04_14-23_ADAPTED_BAYER_2cyc_no_infarct\results.csv'
# csv_infarct = r'C:/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/Biventricular model/20-04_11-24_ADAPTED_BAYER_2cyc_with_infarct\results.csv'

# csv_ref = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\leftventricular model\21-04_16-13_with_fiber_reor_30\results.csv'
# csv_infarct = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\leftventricular model\21-04_16-14_no_fiber_reor_30\results.csv'

csv_ref = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\leftventricular model\07-05_09-26_fiber_reorientation_meshres_20\results.csv'

csv_infarct = r'C:/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/leftventricular model/19-06_14-19_ta0_map_test/results.csv'
csv_infarct = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\leftventricular model\22-06_17-33_eikonal_8node\results.csv'
# csv_infarct = r'C:/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/leftventricular model/23-06_13-08_eikonal_td_1node/results.csv'

COMPARE = True

label1 = 'eikonal'
label2 = 'reference'

title = 'hemodynamics 8 nodes'

CYCLE = 2 #'all' #'last'

#if the datasets should not be compared, select which dataset should be analyzed
if COMPARE == False:
    ANALYZE = 'normal'

# Take the last cycle, or specify the cycle directly
#cycle=10
#cycle = cycle = max(results_normal['cycle']) - 1

if CYCLE == 'last':
    # Load results
    [results_ref, cycle_ref] = load_reduced_dataset(csv_ref)
    [results_infarct, cycle_inf] = load_reduced_dataset(csv_infarct)
elif CYCLE == 'all':
    results_cycs = Dataset(filename=csv_ref)
    results_cycs_2 = Dataset(filename=csv_infarct)   
else:
    [results_ref, cycle_ref] = load_reduced_dataset(csv_ref, cycle = CYCLE)
    [results_infarct, cycle_inf] = load_reduced_dataset(csv_infarct, cycle = CYCLE) 


if COMPARE == True:
    if cycle_ref != cycle_inf:
        print('infarct and normal data not of same length, continuing with cycle {}'.format(min(cycle_ref,cycle_inf)))
        [results_normal, cycle_ref] = load_reduced_dataset(csv_ref, min(cycle_ref,cycle_inf) )
        [results_infarct, cycle_inf] = load_reduced_dataset(csv_infarct, min(cycle_ref,cycle_inf))
    hemo_sum = procentual_change_hemodynamics(results_ref,results_infarct)
    plot_compare_results(results_ref,results_infarct, dir_out=os.path.dirname(csv_infarct), cycle=None, label1= label1,label2=label2, title= title)
else:
    if CYCLE == 'last':
        # Data from the selected cycle
        if ANALYZE == 'infarct':
            data_cycle = results_infarct
            pathname = csv_infarct
            cycle = cycle_inf
        else:
            data_cycle = results_ref
            pathname = csv_ref
            cycle = cycle_ref
        hemo_sum = hemodynamic_summary(data_cycle)
        print_hemodynamic_summary(hemo_sum,cycle)
        plot_results(data_cycle,dir_out=os.path.dirname(pathname), cycle=cycle)
    elif CYCLE == 'all':
        fig = plt.figure()
        cycles_plot = Hemodynamics_all_cycles(results_cycs, fig = fig)
        cycles_plot.plot()

    
