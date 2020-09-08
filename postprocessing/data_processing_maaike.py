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

# csv_ref =  r'C:/Users/Maaike/Documents/Master/Graduation_project/Results/leftventricular model/ischemic_model/24-03_14-07_infarct_xi_10/results.csv'
#csv_infarct = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\09-04_15-25_big_border_meshres_20\results.csv'

# csv_ref = r'C:/Users/Maaike/Documents/Master/Graduation_project/Results/leftventricular model/27_02_default_inputs/results.csv'
# csv_infarct = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\Biventricular model\default\results.csv'

#csv_ref = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\Biventricular model\default\results.csv'
#csv_infarct = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\Biventricular model\08-04_15-16_save_coord_test\results.csv'

# csv_ref = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\Biventricular model\default\results.csv'
# csv_infarct = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\Biventricular model\20-04_14-23_ADAPTED_BAYER_2cyc_no_infarct\results.csv'

# csv_ref = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\Biventricular model\20-04_14-23_ADAPTED_BAYER_2cyc_no_infarct\results.csv'
# csv_infarct = r'C:/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/Biventricular model/20-04_11-24_ADAPTED_BAYER_2cyc_with_infarct\results.csv'

# csv_ref = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\leftventricular model\21-04_16-13_with_fiber_reor_30\results.csv'
# csv_infarct = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\leftventricular model\21-04_16-14_no_fiber_reor_30\results.csv'

# csv_ref =  r'C:/Users/Maaike/Documents/Master/Graduation_project/Results/leftventricular model/07-05_09-26_fiber_reorientation_meshres_20/results.csv'

# csv_var = r'C:/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/leftventricular model/19-06_14-19_ta0_map_test/results.csv'
# csv_var = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\leftventricular model\22-06_17-33_eikonal_8node\results.csv'
# csv_var = r'C:/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/leftventricular model/23-06_13-08_eikonal_td_1node/results.csv'
# csv_var = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\leftventricular model\01-07_15-48_eikonal_more_roots_mesh_20\results.csv'

# csv_var = r'C:/Users/Maaike/Documents/Master/Graduation_project/Results/leftventricular model/Eikonal/01-07_15-48_eikonal_more_roots_mesh_20/results.csv'
# csv_var = r'C:/Users/Maaike/Documents/Master/Graduation_project/Results/leftventricular model/Eikonal/16-07_10-23_bue_root_kot_sig_higher_20/results.csv'

csv_ref = r'C:/Users/Maaike/OneDrive - TU Eindhoven//Results/leftventricular model/Eikonal/06-07_15-56_ref(no_eikonal)_2_cycles_mesh_20/results.csv'
# csv_var = r'C:/Users/Maaike/Documents/Master/Graduation_project/beatit-master/beatit/pV.csv'
# csv_var = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Results/Beatit/2_cycles/pV.csv'

csv_ref = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/mitral_flow_tests/02-09_10-04_res_pven_2_mesh_20/results.csv'
# csv_var2 = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/mitral_flow_tests/02-09_10-04_res_pven_3_mesh_20/results.csv'
# csv_ref = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/mitral_flow_tests/02-09_10-04_res_pven_4_mesh_20/results.csv'

csv_var1= r'C:/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/ejection_time_tests/03-09_17-27_longer_act_rven_5_mesh_20/results.csv'
csv_var2 = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/ejection_time_tests/03-09_13-47_longer_act_rven_4_mesh_20/results.csv'
csv_var3 = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/ejection_time_tests/03-09_16-26_longer_act_rven_2_mesh_20/results.csv'


## Enter Variables ##
csv_vars = [csv_var3]

# enter the type of model for the ref and the var datasets: 'cvbtk' or 'beatit' 
model_ref = 'cvbtk'
model_var = ['cvbtk','cvbtk','cvbtk']

label_ref = ['Rven: 2, old act']
label_var = ['Rven: 2, longer act','Rven: 4, longer act','Rven: 2, longer act']

# label_ref = ['Rven: 4']
# label_var = ['Rven: 4, longer act','Rven: 2']

COMPARE = True

title = 'Varying venous resistance and ejection time'

# select cycle
# - number (same cycle is selected for all datasets (advised))
# - None (last cycle is selected for all datasets)
# - all (all the cycles are selected, cannot compare datasets)
CYCLE = None #None #'all' #None

#if the datasets should not be compared, select which dataset should be analyzed
if COMPARE == False:
    ANALYZE = 'ref' #'var'
    
## End of variable selection ##

if CYCLE == 'all':
    results_cycs = Dataset(filename=csv_ref)
    results_cycs_2 = Dataset(filename=csv_var)   
else:
    [results_ref, cycle_ref] = load_reduced_dataset(csv_ref, cycle = CYCLE, model = model_ref)
    
    results_var = []
    cycle_var = []
    for ii, csv in enumerate(csv_vars):
        [result, cycle] = load_reduced_dataset(csv, cycle = CYCLE, model = model_var[ii])
        results_var.append(result)
        cycle_var.append(cycle)

if COMPARE == True:
    # if cycle_ref != cycle_var:
    #     print('infarct and normal data not of same length, continuing with cycle {}'.format(min(cycle_ref,cycle_var)))
    #     [results_normal, cycle_ref] = load_reduced_dataset(csv_ref, min(cycle_ref,cycle_var), model = model_ref)
    #     [results_var, cycle_var] = load_reduced_dataset(csv_var, min(cycle_ref,cycle_var), model = model_var)
    hemo_sum = procentual_change_hemodynamics(results_ref,results_var[0], 
                                              title1 = label_ref[0], title2 = label_var[0], 
                                              model1 = model_ref, model2 = model_var[0])
    dir_out = os.path.split(csv_vars[0])[0]
    hemo_sum.to_csv(os.path.join(dir_out, 'hemodynamics.csv'))
    plot_compare_results(results_ref,results_var, dir_out=os.path.dirname(csv_vars[0]), 
                         cycle=cycle_ref, label_ref= label_ref, label_vars=label_var, title= title, modelref = model_ref, modelvars = model_var)
else:
    if CYCLE == 'all':
        fig = plt.figure()
        cycles_plot = Hemodynamics_all_cycles(results_cycs, fig = fig)
        cycles_plot.plot()
        
    else:
        # Data from the selected cycle
        if ANALYZE == 'var':
            data_cycle = results_var
            pathname = csv_var
            cycle = cycle_var
            model = model_var
        else:
            data_cycle = results_ref
            pathname = csv_ref
            cycle = cycle_ref
            model = model_ref
            
        hemo_sum = hemodynamic_summary(data_cycle, model)
        print_hemodynamic_summary(hemo_sum,cycle)
        plot_results(data_cycle,model = model, title = title, dir_out=os.path.dirname(pathname), cycle=cycle)


    
