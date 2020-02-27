# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 14:19:41 2019

@author: Hermans
"""

import matplotlib.pyplot as plt
import numpy as np

import sys
# Add higher directory to python path (so we can import from postprocessing.py)
sys.path.append("..") 

from postprocessing import  (Dataset, 
                             find_experiment_hemodynamics, 
                             figure_make_up)                                    

plt.close('all')

hc = 1
rpm = 9500

fontsize = 14
linewidth = 3

csv = find_experiment_hemodynamics(hc, rpm)

results = Dataset(filename=csv)

time = np.asarray(results['time']*1000)
plv = results['plv']
part = results['part']
qlvad = results['qlvad']
qart = results['qart']

plt.figure(figsize=(10,4))
plt.subplot(121)
plt.plot(time, plv, linewidth=linewidth,
         label='LV pressure')
plt.plot(time, part, linewidth=linewidth,
         label='Aortic pressure')

figure_make_up(title=None, xlabel='Time [ms]',
               ylabel='pressure [mmHg]', fontsize=fontsize)

plt.subplot(122)
plt.plot(time, qlvad, linewidth=linewidth,
         label='LVAD flow')
#plt.plot(time, qart-qlvad, linewidth=linewidth,
#         label='Aortic valve flow')
plt.plot(time, qart, linewidth=linewidth,
         label='Total flow')

figure_make_up(title=None, xlabel='Time [ms]',
               ylabel='Flow [l/min]', fontsize=fontsize)
plt.tight_layout()
