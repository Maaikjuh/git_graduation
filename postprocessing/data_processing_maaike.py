# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 10:57:33 2020

@author: Maaike
"""

import sys
sys.path.append(r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/git_graduation_project/cvbtk')
sys.path.append(r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/git_graduation_project/postprocessing')

from dataset import Dataset


from postprocess_Maaike import *
import os
import matplotlib.pyplot as plt
# plt.close('all')

# Specify the results.csv file (which contains the hemodynamic data) directly:
#csv_normal = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\24-03_14-07_infarct_xi_10\results.csv'
#csv_infarct = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\31-03_16-15_infarct_droplet_tao_20_meshres_30\results.csv'

# csv_ref =  r'C:/Users/Maaike/Documents/Master/Graduation_project/Results/leftventricular model/ischemic_model/24-03_14-07_infarct_xi_10/results.csv'
#csv_infarct = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\09-04_15-25_big_border_meshres_20\results.csv'

csv_old_vars = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/27_02_default_inputs/results.csv'
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

# csv_ref = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/mitral_flow_tests/02-09_10-04_res_pven_2_mesh_20/results.csv'
# csv_var2 = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/mitral_flow_tests/02-09_10-04_res_pven_3_mesh_20/results.csv'
# csv_ref = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/mitral_flow_tests/02-09_10-04_res_pven_4_mesh_20/results.csv'

# csv_var1= r'C:/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/ejection_time_tests/03-09_17-27_longer_act_rven_5_mesh_20/results.csv'
# csv_var2 = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/ejection_time_tests/03-09_13-47_longer_act_rven_4_mesh_20/results.csv'
csv_ref = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/ejection_time_tests/03-09_16-26_longer_act_rven_2_mesh_20/results.csv'

csv_var1 = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/ischemic_model/10-09_08-37_old_vars_infarct_hdf5_p_stress_1_res_20/results.csv'
csv_var2 = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/ischemic_model/09-09_16-32_infarct_hdf5_higher_p_stress_2/results.csv'
csv_var3 = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/ischemic_model/09-09_16-50_infarct_hdf5_higher_p_stress_5/results.csv'

# csv_ref = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/ischemic_model/old_ischemic_model/09-06_13-43_ischemic_meshres_20/results.csv'

# csv_ref = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/ischemic_model/10-09_15-50_new_model_no_infarct_res_20/results.csv'
csv_ref = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/11-09_15-22_new_vars_infarct_11_a0_inf_unchanged_res_20/results.csv'
csv_var3 = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/14-09_09-31_new_vars_infarct_a0_inf_4_res_20/results.csv'
# csv_var4 = r'C:\Users\Maaike\OneDrive - TU Eindhoven\Results\leftventricular model\ischemic_model\14-09_09-33_new_vars_infarct_a0_inf_2_res_20\results.csv'
csv_var5 = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/21-09_10-03_infarct_a5_5_a0_4_20/results.csv'


# csv_var1 = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/ischemic_model/11-09_09-31_old_vars_infarct_11_a0_inf_unchanged_res_20/results.csv'
# csv_var2 = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/ischemic_model/11-09_10-36_old_vars_infarct_11_a0_inf_4_res_20/results.csv'

# # csv_ref = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/ischemic_model/11-09_15-22_new_vars_infarct_11_a0_inf_unchanged_res_20/results.csv'

# csv_var1 = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/ischemic_model/11-09_15-40_new_vars_infarct_11_a0_inf_unchanged_border_res_20/results.csv'
# csv_var2 = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Results/leftventricular model/ischemic_model/11-09_15-39_new_vars_infarct_11_a0_inf_4_border_res_20/results.csv'

csv_ref = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/15-09_new_vars_ref_cyc_5_res_30/results.csv'

# csv_ref = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/22-09_13-54_infarct_endo_a0_unchanged_res_20/results.csv'
# csv_var = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/22-09_13-37_infarct_endo_a0_4_res_20/results.csv'

csv_res_30 = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/07-10_13-50_ref_cyc_10_res_30/results.csv'

csv_res_20 = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/07-10_13-50_ref_cyc_10_res_20/results.csv'
csv_res_40 = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/07-10_14-17_ref_cyc_10_res_40/results.csv'
# csv_var = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/29-09_15-14_infarct_ref_a0_unchanged_res_30/results.csv'


csv_transm_acute = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/29-09_15-14_infarct_ref_a0_unchanged_res_30/results.csv'
csv_transm_chronic = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/07-10_08-07_infarct_ref_a0_4_res_30/results.csv'
csv_endo_acute = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/07-10_08-16_infarct_endo_ref_a0_unchanged_res_30/results.csv'
csv_endo_chronic = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/ischemic_model/07-10_08-20_infarct_endo_ref_a0_4_res_30/results.csv'

## Enter Variables ##
# csv_ref = csv_res_20
csv_vars = [csv_res_30, csv_res_40]

csv_ref = csv_res_30
csv_vars = [csv_transm_acute, csv_transm_chronic]
csv_vars = [csv_endo_acute, csv_endo_chronic]


# enter the type of model for the ref and the var datasets: 'cvbtk' or 'beatit' 
model_ref = 'cvbtk'
model_var = ['cvbtk','cvbtk','cvbtk']

# label_ref = ['Simulation 1',]
# label_var = ['Simulation 2','Simulation 3']

label_ref = ['healthy',]
label_var = ['endocardial acute','endocardial chronic']
# label_var = ['transmural acute','transmural chronic']

#if compare == False, csv_ref will be analyzed
COMPARE = True

title = ''#'Healthy and ischemic LV'
# title = 'Reference'

# select cycle
# - number (same cycle is selected for all datasets (advised))
# - None (last cycle is selected for all datasets)
# - all (all the cycles are selected, cannot compare datasets)
CYCLE = 5 #'all' #3 #None #None #'all' #None

   
## End of variable selection ##

if COMPARE == True:
    if CYCLE == 'all':
        hemodynamics = []
        results_cycs = Dataset(filename=csv_ref)
        results_var = []
        cycle_var = [] 
        cycles_plot = Hemodynamics_all_cycles(results_cycs)
        cycles_plot.plot()
        hemo = cycles_plot.hemodymanics()
        hemodynamics.append(hemo)
        colors = ['tab:red', 'tab:green', 'tab:orange']
        for ii, csv in enumerate(csv_vars):
            result = Dataset(filename = csv)
            results_var.append(result)
            cycles_plot.compare_against(label_ref, label_var, result, color = colors[ii])
            hemo = cycles_plot.hemodymanics(df = result)
            hemodynamics.append(hemo)

    else:
        [results_ref, cycle_ref] = load_reduced_dataset(csv_ref, cycle = CYCLE, model = model_ref)
        results_var = []
        cycle_var = []
        for ii, csv in enumerate(csv_vars):
            [result, cycle] = load_reduced_dataset(csv, cycle = CYCLE, model = model_var[ii])
            results_var.append(result)
            cycle_var.append(cycle)
    # if cycle_ref != cycle_var:
    #     print('infarct and normal data not of same length, continuing with cycle {}'.format(min(cycle_ref,cycle_var)))
    #     [results_normal, cycle_ref] = load_reduced_dataset(csv_ref, min(cycle_ref,cycle_var), model = model_ref)
    #     [results_var, cycle_var] = load_reduced_dataset(csv_var, min(cycle_ref,cycle_var), model = model_var)
        hemo_sum = procentual_change_hemodynamics(results_ref,results_var, 
                                                  title1 = label_ref[0], title2 = label_var, 
                                                  model1 = model_ref, model2 = model_var)
        dir_out = os.path.split(csv_vars[0])[0]
        hemo_sum.to_csv(os.path.join(dir_out, 'hemodynamics.csv'))
        plot_compare_results(results_ref,results_var, dir_out=os.path.dirname(csv_vars[0]), 
                             cycle=cycle_ref, label_ref= label_ref, label_vars=label_var, title= title, modelref = model_ref, modelvars = model_var, legend = True)
else:
    if CYCLE == 'all':
        results_cycs = Dataset(filename=csv_ref)
        fig = plt.figure()
        cycles_plot = Hemodynamics_all_cycles(results_cycs, fig = fig)
        cycles_plot.plot()
        
    else:
        [results_ref, cycle_ref] = load_reduced_dataset(csv_ref, cycle = CYCLE, model = model_ref)
        # Data from the selected cycle

        data_cycle = results_ref
        pathname = csv_ref
        cycle = cycle_ref
        model = model_ref
            
        hemo_sum = hemodynamic_summary(data_cycle, model)
        print_hemodynamic_summary(hemo_sum,cycle)
        plot_results(data_cycle,model = model, title = title, dir_out=os.path.dirname(pathname), cycle=cycle, legend = True, fontsize = 15)


    
