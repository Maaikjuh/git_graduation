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
var_names = ['Healthy', 'Acute', 'Chronic', 'Acute \n', 'Chronic\n']
SV = [67.09, 59.7, 59.5, 64.5, 61.9]
dp_dt = [2187.1, 1978.1,2002.1,2144.5,2083.5]
W = [0.98, 0.80, 0.80, 0.92, 0.85]
EF = [60,53.1,56.6,57.6,58.6]

# var_names = ['Healthy', 'td healthy', 'td isch', 'Ischemic', 'td isch LV isch']
# SV = [67.89, 67.56, 67.34, 58.78, 58.24]
# dp_dt = [2187.52, 2206.66, 2174.79, 2020.44, 1956.82]
# W = [0.98,	0.97, 0.96, 0.8, 0.78]
# EF = [60.65, 60.35, 60.14, 52.39, 51.91]

var_value = [SV,dp_dt,W, EF]

lims = [[57, 68], [1900, 2200], [0.75, 1.], [51, 61]]

y_label = ['SV [ml]', r'$(dp_{lv}/dt)_{max}$ [mmHg/s]', 'work [J]', 'EF [%]']
save_name = ['SV', 'dp_dt', 'work', 'EF']
save_dir = r'C:/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Thesis/Figures'

font = 12

for variable in range(0,len(var_value)):
    fig, (ax, ax1) = plt.subplots(2,1,sharex=True, figsize = (5.31, 4.36), gridspec_kw={'height_ratios': [20, 1]})
    fig.subplots_adjust(hspace=0.05)
    color = ['tab:blue', 'tab:green', 'yellowgreen', 'tab:red', 'tomato']
    # color = ['tab:blue', 'lightskyblue', 'tab:pink', 'tab:red', 'salmon']
    # color = ['tab:green', 'tab:blue', 'tab:purple', 'tab:red', 'tab:orange']
    
    ax.bar(var_names, var_value[variable], color=color, width=0.8)
    ax1.bar(var_names, var_value[variable], color=color, width=0.8)
    
    ax.set_ylabel(y_label[variable], fontsize = font)
    plt.xticks(fontsize=font)
    
    ax1.set_ylim(0,1)
    ax.set_ylim(lims[variable])
    
    #get ticks positions:
    x_min, x_max = ax1.get_xlim()
    ticks = [(tick - x_min)/(x_max - x_min) for tick in ax1.get_xticks()]
    
    plt.text((ticks[1] + ticks[2])/2, -2.5, 'Transmural', ha='center', va='center',  transform=ax1.transaxes, fontsize = font )
    plt.text((ticks[3] + ticks[4])/2, -2.5, 'endocardial', ha='center', va='center',  transform=ax1.transAxes, fontsize = font )
    # plt.text((ticks[1] + ticks[2])/2, -2.5, 'Transmural', ha='center', va='center', transform=ax1.transaxes, fontsize = font )
    # plt.text((ticks[3] + ticks[4])/2, -2.5, 'endocardial', ha='center', va='center',  transform=ax1.transaxes, fontsize = font )
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
    
    fig.savefig(os.path.join(save_dir, save_name[variable] + '_staaf_diag_eikonal.png'), dpi=300, bbox_inches="tight")