# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:21:52 2020

@author: Maaike
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
plt.close('all')
var_names = ['Healthy', 'Transmural\n acute', 'Transmural\n chronic', 'Endocardial\n acute', 'Endocardial\n chronic']
SV = [67.09, 59.7, 59.5, 64.5, 61.9]
dp_dt = [2187.1, 1978.1,2002.1,2144.5,2083.5]
W = [0.98, 0.80, 0.80, 0.92, 0.85]
EF = [60,53.1,56.6,57.6,58.6]

var_value = [SV,dp_dt,W, EF]

lims = [[57, 68], [1900, 2200], [0.75, 1.], [51, 61]]

y_label = ['SV [ml]', r'$(dp_{lv}/dt)_{max}$ [mmHg/s]', 'work [J]', 'EF [%]']
save_name = ['SV', 'dp_dt', 'work', 'EF']
save_dir = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Thesis/Figures'

font = 13

for variable in range(0,len(var_value)):
    fig, (ax, ax1) = plt.subplots(2,1,sharex=True, figsize = (7.56, 6.33), gridspec_kw={'height_ratios': [20, 1]})
    fig.subplots_adjust(hspace=0.05)
    color = ['tab:blue', 'tab:green', 'yellowgreen', 'tab:red', 'tomato']
    
    ax.bar(var_names, var_value[variable], color=color)
    ax1.bar(var_names, var_value[variable], color=color)
    
    ax.set_ylabel(y_label[variable], fontsize = font)
    plt.xticks(fontsize=font)
    
    ax1.set_ylim(0,1)
    ax.set_ylim(lims[variable])
    
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(labelbottom=False)
    
    
    d = 0.5
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax.plot([0],transform=ax.transAxes, **kwargs)
    ax1.plot([1], transform=ax1.transAxes, **kwargs)
    ax1.set_yticklabels([0])
    plt.show()
    
    fig.savefig(os.path.join(save_dir, save_name[variable] + '_staaf_diag.png'), dpi=300, bbox_inches="tight")