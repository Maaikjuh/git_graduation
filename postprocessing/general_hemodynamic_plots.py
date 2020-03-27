# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 14:14:22 2018

@author: Hermans
"""

# Add path to cvbtk directory.
import sys
sys.path.append(r'C:\Users\Maaike\Documents\Master\Graduation_project\git_graduation_project\cvbtk')
sys.path.append(r'C:\Users\Maaike\Documents\Master\Graduation_project\git_graduation_project\postprocessing')

from dataset import Dataset


from postprocessing import (postprocess, 
                            hemodynamic_summary, print_hemodynamic_summary,
                            find_simulation_hemodynamics)

import os
import matplotlib.pyplot as plt
plt.close('all')


# Specify the results.csv file (which contains the hemodynamic data) directly:
#csv = r'E:\Graduation Project\FEniCS\PycharmProject\output\sepran_simulation\sepran_simulation_res_30\results.csv'
csv = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\25-03_21-49_infarct_droplet_shape_lower_ta0\results.csv'

## OR specify heart condition and find csv file automatically:
#hc = 1
#rpm = 7500
#simulation_type = 'BiV'
#csv = find_simulation_hemodynamics(hc, rpm, simulation_type=simulation_type)

# Load results.
results = Dataset(filename=csv)

cycle = max(results['cycle']) - 1
#cycle =3

print('Analyzing cycle {}...'.format(cycle))

postprocess(results, dir_out=os.path.dirname(csv), cycle=cycle)

hemo_sum = hemodynamic_summary(results,cycle)
print_hemodynamic_summary(hemo_sum)
