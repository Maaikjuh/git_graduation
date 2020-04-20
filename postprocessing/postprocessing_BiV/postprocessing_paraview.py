# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 09:08:27 2018

@author: Hermans
"""

import sys
sys.path.append("..") # Adds higher directory to python modules path.

from postprocessing import Dataset, shift_data, get_paths

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import glob
import pickle
import random
import csv

from mpl_toolkits.mplot3d import Axes3D

plt.close('all')

def kPa_to_mmHg(p_kPa):
    """
    Convert pressures from kPa to mmHg.

    Args:
        p_kPa: Pressure in kPa.

    Returns:
        Pressure in mmHg.
    """
    conversion_factor = 133.322387415
    return p_kPa*1000/conversion_factor

def mmHg_to_kPa(p_mmHg):
    """
    Convert pressures from mmHg to kPa.

    Args:
        p_mmHg: Pressure in mmHg.

    Returns:
        Pressure in kPa.
    """
    conversion_factor = 133.322387415
    return p_mmHg/1000*conversion_factor

def read_dict_from_csv(filename):
    """
    Reads a (nested) dictionary from a csv file (reverse of save_dict_to_csv).

    Args:
         filename (str): Filename of the csv file (including .csv)

         Returns:
             Nested dictionary based on the CSV file
    """

    def nested_dict(d, keys, value):
        # Set d['a']['b']['c'] = value, when keys = ['a', 'b', 'c']
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    d_out={}

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            path = line[0]

            # Convert to ints or floats if possible (otherwise, keep it as a string).
            try:
                if '.' in line[1] or 'e' in line[1]:
                    value = float(line[1])
                else:
                    value = int(line[1])
            except ValueError:
                if line[1] == 'False':
                    value = False
                elif line[1] == 'True':
                    value = True
                else:
                    value = line[1]

            if path[0]=='/':
                # Exclude first slash as it would create an empty entry when applying path.split('/')
                path = path[1:]

            nested_dict(d_out, path.split('/'), value)

    return d_out

class postprocess_paraview(object):
    def __init__(self, directory, results_csv=None, cycle=None, **kwargs):
        self.directory = directory
        self.files = self.find_files(directory)
        
        if results_csv is None:
            # Directory with results file is 2 folders down.
            cycle_dir = os.path.split(directory)[0]
            results_dir = os.path.split(cycle_dir)[0]
            results_csv = os.path.join(results_dir, 'results.csv')
            
            if cycle is None:
                # Assume it is in the cycle directory.
                cycle = float(os.path.split(cycle_dir)[1].split('_')[1])
            
        self.results = self.load_reduced_dataset(results_csv, cycle=cycle)

        self._parameters = self.default_parameters()
        self._parameters.update(kwargs)
        
        self._all_data = None
        self._idx_lv_data = None
        self._idx_rv_data = None
        self._idx_sep_data = None
        
        self._idx_ls0 = None
        
        self.column_headers = self.read_column_headers()
        
    @staticmethod
    def default_parameters():
        par = {}
        par['cut_off_low'] = -3.
        par['cut_off_high'] = 1.
        par['slice_thickness'] = 0.2
        par['R_2sep'] = 2.95675
        par['R_2'] = 3.56
        par['Z_2'] = 6.02
        par['ls0'] = 1.9
        par['name'] = ''
        par['load_pickle_file'] = True
        return par
        
    @staticmethod
    def find_files(directory):
        all_files = np.asarray(glob.glob(os.path.join(directory, '*all_points*csv')))
        all_timesteps = [float(f.split('.')[-2]) for f in all_files]
        return all_files[np.argsort(all_timesteps)].tolist()
    
    def compute_strain(self, ls, ls0=None):
        # Computes natural (true) fiber strain from fiber length with 
        # respect to the moment of first myofiber shortening.
        if len(ls.shape) == 1:    
            if ls0 is None:
                # Use sarcomere lengths at first moment of onset shortening as reference.
                ls0 = ls[self.idx_ls0]
        elif len(ls.shape) == 2:
            if ls0 is None:
                # Use sarcomere lengths at first moment of onset shortening as reference.
                ls0 = ls[:, self.idx_ls0]
            if not np.size(ls0) == 1:
                ls0 = np.tile(ls0, (ls.shape[1], 1)).transpose()
        else:
            raise ValueError('Unexpected size of ls.')

        stretch_ratio = ls/ls0
        return np.log(stretch_ratio)
    
    @property
    def idx_ls0(self):
        if self._idx_ls0 is None:
            # Find earliest moment of shortening among all selected fibers.
            regions = [self.extract_rv_idx(), 
            self.extract_sep_idx(), 
            self.extract_lv_idx()]    
            
            all_regions = np.concatenate(regions)
            
            idx_ls0 = np.min(np.argmax(
                    self.all_data[all_regions, 
                                  self.column_headers['ls_old'], :], axis=1))
            self._idx_ls0 = idx_ls0
        return self._idx_ls0
    
    @property
    def parameters(self):
        return self._parameters
        
    @property
    def all_data(self):
        if self._all_data is None:
            pickle_file = os.path.join(self.directory, 'all_data.pkl')
            if not self.parameters['load_pickle_file'] or not os.path.exists(pickle_file):
                self._all_data = self.load_all_data()
                pickle.dump(self._all_data, open(pickle_file, "wb" ))
            else:
                self._all_data = pickle.load(open(pickle_file, "rb" ))

        return self._all_data
        
    @property
    def idx_lv_data(self):
        if self._idx_lv_data is None:
            self._idx_lv_data = self.extract_lv_idx()
        return self._idx_lv_data
    
    @property
    def idx_rv_data(self):
        if self._idx_rv_data is None:
            self._idx_rv_data = self.extract_rv_idx()
        return self._idx_rv_data
    
    @property
    def idx_sep_data(self):
        if self._idx_sep_data is None:
            self._idx_sep_data = self.extract_sep_idx()
        return self._idx_sep_data
        
    @staticmethod
    def compute_work(stress, strain):
        # Add the first time point to the end of the array, closing the loop.
        stress = np.hstack((stress, stress[:,0:1]))
        strain = np.hstack((strain, strain[:,0:1]))
        return -np.trapz(stress, x=strain, axis=1)
        
    def load_all_data(self):
        # Load all data points as 3D array (point, value, time).
        data = []
        print('Loading all data into memory. This may take a while...')
        for jj, file in enumerate(self.files):
            if jj % 10 == 0:
                print('{:.2f} %'.format(jj/len(self.files)*100))
            file_data = []
            with open(file) as f:
                for i,line in enumerate(f):
                    if i == 0:
                        # Skip header.
                        continue
                    row_data = [float(x) for x in line.split(',')]
                    file_data.append(row_data)
            data.append(file_data)
        
        return np.dstack(data)
                
    def extract_lv_idx(self):
        # Extract points LV.
        h = self.column_headers
        data = self.all_data[:, :, 0]
        mask_cut_off_low = data[:, h[':2']] >= self.parameters['cut_off_low']
        mask_cut_off_high = data[:, h[':2']] <= self.parameters['cut_off_high']
        mask_cut_off_right = data[:, h[':0']] >= 0.
        mask_cut_off_slice = abs(data[:, h[':1']]) <= self.parameters['slice_thickness']/2 
        mask_tot = mask_cut_off_low*mask_cut_off_high*mask_cut_off_right*mask_cut_off_slice
        return np.where(mask_tot)[0]     
        
    def extract_rv_idx(self):
        # Extract points RV.
        # h = self.column_headers
        # data = self.all_data[:, :, 0]
        # mask_cut_off_low = data[:, h[':2']] >= self.parameters['cut_off_low']
        # mask_cut_off_high = data[:, h[':2']] <= self.parameters['cut_off_high']
        # mask_cut_off_left = np.sqrt(data[:, h[':0']]**2/self.parameters['R_2sep']**2 + data[:, h[':1']]**2/self.parameters['R_2']**2 + data[:, h[':2']]**2/self.parameters['Z_2']**2) > 1.05
        # mask_cut_off_left2 = data[:, h[':0']] < 0.
        # mask_cut_off_slice = abs(data[:, h[':1']]) <= self.parameters['slice_thickness']/2 
        # mask_tot = mask_cut_off_low*mask_cut_off_high*mask_cut_off_left*mask_cut_off_left2*mask_cut_off_slice
        
        h = self.column_headers
        data = self.all_data[:, :, 0]
        mask_cut_off_low = data[:, h[':2']] >= self.parameters['cut_off_low']
        mask_cut_off_high = data[:, h[':2']] <= self.parameters['cut_off_high']
        mask_cut_off_right = data[:, h[':0']] >= 0.
        mask_cut_off_line_left = -2.37/1.73*data[:, h[':1']] -  self.parameters['slice_thickness']/2 >=data[:, h[':0']]
        mask_cut_off_line_right = -2.37/1.73*data[:, h[':1']] +  self.parameters['slice_thickness']/2 <=data[:, h[':0']]

        mask_tot = mask_cut_off_low*mask_cut_off_high*mask_cut_off_line_left*mask_cut_off_line_right*mask_cut_off_right
         
        
        
        return np.where(mask_tot)[0]  
    
    def extract_sep_idx(self):
        # Extract points SEP.
        h = self.column_headers
        data = self.all_data[:, :, 0]
        mask_cut_off_low = data[:, h[':2']] >= self.parameters['cut_off_low']
        mask_cut_off_high = data[:, h[':2']] <= self.parameters['cut_off_high']
        mask_cut_off_left = np.sqrt(data[:, h[':0']]**2/self.parameters['R_2sep']**2 + data[:, h[':1']]**2/self.parameters['R_2']**2 + data[:, h[':2']]**2/self.parameters['Z_2']**2 ) <= 1.05
        mask_cut_off_right = data[:, h[':0']] <= 0.
        mask_cut_off_slice = abs(data[:, h[':1']]) <= self.parameters['slice_thickness']/2 
        mask_tot = mask_cut_off_low*mask_cut_off_high*mask_cut_off_left*mask_cut_off_slice*mask_cut_off_right
        return np.where(mask_tot)[0]  
    
    @staticmethod
    def load_reduced_dataset(filename, cycle=None):
        """
        Load the given CSV file and reduce it to the final cycle (default) 
        or the specified cycle..
        """
        full = Dataset(filename=filename)
        
        if cycle is None:
            # Choose final cycle.
            cycle = int(max(full['cycle']) - 1)
        
        reduced = full[full['cycle'] == cycle].copy(deep=True)
        return reduced
    
    def plot_global_function_evolution(self, *args, all_results_csv=None, 
                                       fontsize=12, **kwargs):
        """
        Plot the ventricles function in terms of stroke volume, maximum pressure,
        and work for every cycle.
        """
        if all_results_csv is None:
            # Assume that the directory with all results is one folder down.
            cycle_dir = os.path.split(self.directory)[0]
            results_dir = os.path.split(cycle_dir)[0]
            all_results_dir = os.path.split(results_dir)[0]
            all_results_csv = os.path.join(all_results_dir, 'results.csv')
        
        all_results = Dataset(filename=all_results_csv)
        
        # Find the number of the first and last cycle.
        start = min(all_results['cycle'])
        stop = max(all_results['cycle'])
        cycles = list(range(start, stop))

        # Extract the SV of the systemic and pulmonary circulation.
        strokevolumes = {'_s': [],
                         '_p': []}
        pmax = {'_s': [],
                '_p': []}
        work = {'_s': [],
                '_p': []}
        for cycle in cycles:
            df = all_results[all_results['cycle'] == int(cycle)]
            for ps in ['_s', '_p']:
                # Find begin phase 2.
                idx_ed = min(np.where(df['phase' + ps].values == 2)[0])
                # Find begin phase 4.
                idx_ee = min(np.where(df['phase' + ps].values == 4)[0])

                # Calculate SV.
                volume = df['vcav' + ps].values
                sv = volume[idx_ed] - volume[idx_ee]
                strokevolumes[ps].append(sv)

                # Maximum pressure.
                pressure = df['pcav' + ps].values
                pmax[ps].append(max(pressure))

                # Work (surface of pressure-volume loop).
                work_cycle = 0.
                for ii in range(1, len(volume)):
                    work_cycle += (pressure[ii - 1] + pressure[ii]) / 2 * (
                                volume[ii - 1] - volume[ii]) / 1000  # in Joule.
                work[ps].append(work_cycle)

        # Find begin and end of adaptation.
        cycle_dir = os.path.split(self.directory)[0]
        results_dir = os.path.split(cycle_dir)[0]
        inputs_filename = os.path.join(results_dir, 'inputs.csv')
        inputs = read_dict_from_csv(inputs_filename)
        begin_adap = inputs['model']['fiber_reorientation']['ncycles_pre'] + 1
        end_adap = inputs['model']['fiber_reorientation']['ncycles_reorient'] + begin_adap
        color_adap = 'k'

        plt.figure(figsize=(15, 5), dpi=100) 
        lv_label = mlines.Line2D([], [], color='C3', label='LV')
        rv_label = mlines.Line2D([], [], color='C0', linestyle='-', label='RV')
        
        # Stroke volume.
        ax3 = plt.subplot(1, 3, 1)
        plt.axvline(x=begin_adap, linestyle='--', color=color_adap)
        plt.axvline(x=end_adap, linestyle='--', color=color_adap)
        ax3.plot(cycles, strokevolumes['_s'], color='C3')
        ax3.set_xlabel('Cardiac cycle [-]', fontsize=fontsize)
        ax3.set_ylabel('Stroke volume [ml]', fontsize=fontsize)
        ax3.tick_params('y')
        ax3.tick_params(labelsize=fontsize-2)
        ax3.plot(cycles, strokevolumes['_p'], '-', color='C0')
        plt.title('Stroke volumes', fontsize=fontsize+2)
        plt.legend(handles=[lv_label, rv_label], loc='lower right', 
                   fontsize=fontsize)
        plt.ylim(16, 64)

        # Maximum cavity pressure.
        ax1 = plt.subplot(1, 3, 2)
        plt.axvline(x=begin_adap, linestyle='--', color=color_adap)
        plt.axvline(x=end_adap, linestyle='--', color=color_adap)
        ax1.plot(cycles, kPa_to_mmHg(np.array(pmax['_s'])), color='C3')
        ax1.set_xlabel('Cardiac cycle [-]', fontsize=fontsize)
        ax1.set_ylabel('LV pressure [mmHg]', color='C3', fontsize=fontsize)
        ax1.tick_params('y', colors='C3')
        ax1.tick_params(labelsize=fontsize-2)
        plt.ylim(96, 113)

        ax2 = ax1.twinx()
        ax2.plot(cycles, kPa_to_mmHg(np.array(pmax['_p'])), '-', color='C0')
        ax2.set_ylabel('RV pressure [mmHg]', color='C0', fontsize=fontsize)
        ax2.tick_params('y', colors='C0')
        ax2.tick_params(labelsize=fontsize-2)
        plt.title('Maximum cavity pressure', fontsize=fontsize+2)
#        plt.legend(handles=[lv_label, rv_label], loc='lower right', 
#                   fontsize=fontsize)
        plt.ylim(31.3, 33)

        # Work.
        ax5 = plt.subplot(1, 3, 3)
        plt.axvline(x=begin_adap, linestyle='--', color=color_adap)
        plt.axvline(x=end_adap, linestyle='--', color=color_adap)
        ax5.plot(cycles, work['_s'], 'C3')
        ax5.set_xlabel('Cardiac cycle [-]', fontsize=fontsize)
        ax5.set_ylabel('LV work [J]', color='C3', fontsize=fontsize)
        ax5.tick_params('y', colors='C3')
        ax5.tick_params(labelsize=fontsize-2)
        plt.ylim(0.3, 0.84)

        ax6 = ax5.twinx()
        ax6.plot(cycles, work['_p'], '-C0')
        ax6.set_ylabel('RV work [J]', color='C0', fontsize=fontsize)
        ax6.tick_params('y', colors='C0')
        ax6.tick_params(labelsize=fontsize-2)
        plt.title('Work', fontsize=fontsize+2)
#        plt.legend(handles=[lv_label, rv_label], loc='lower right', 
#                   fontsize=fontsize)
        plt.ylim(0.195, 0.228)

        
        # Increase spacing between subplots.
        plt.subplots_adjust(wspace = .5)
    
    def plot_pv_loops(self, *args, fig=None, fontsize=12, **kwargs):
        if fig is None:
            # Create new figure.
            fig = plt.figure()
        
        # Make fig current.
        plt.figure(fig.number)
        
        results = self.results
    
        for ii, p in enumerate(['_s', '_p']):
            vcav = np.array(results['vcav'+p])
            pcav = kPa_to_mmHg(np.array(results['pcav'+p]))
            plt.plot(vcav, pcav, *args, **kwargs)
            kwargs['label'] = None  # Only label the first plot. 

        plt.xlabel('Cavity volume [ml]', fontsize=fontsize)
        plt.ylabel('Cavity pressure [mmHg]', fontsize=fontsize)
        plt.tick_params(labelsize=fontsize-2)
        plt.grid('on')
        plt.axis([40, 140, 0, 120])
        plt.title('Pressure-volume loops', fontsize=fontsize+2)

    def plot_stress_strain(self, *args, fig=None, fontsize=12, 
                           reference='onset_shortening', 
                           axlimits=[-0.2, 0.02, -2, 52], 
                           **kwargs):
        
        if fig is None:
            # Create new figure.
            fig = plt.figure()
        
        # Make fig current.
        plt.figure(fig.number)

        regions = [self.extract_rv_idx(), 
                   self.extract_sep_idx(), 
                   self.extract_lv_idx()]
        region_labels = ['RV', 'SEP', 'LV']

#        all_regions = np.concatenate(regions)
#        regions.append(all_regions)
#        region_labels.append('Average')
        
        h = self.column_headers
        
        for ii, idx in enumerate(regions):
            plt.subplot(1, len(regions), ii+1)
            
            # Load strain.
            if reference == 'onset_shortening':
                strain = self.compute_strain(self.all_data[idx, h['ls_old'], :])
            elif reference == 'stress_free':
                strain = self.compute_strain(self.all_data[idx, h['ls_old'], :], self.parameters['ls0'])
            else:
                raise ValueError('Invalid strain reference.')
            
            # Load stress.
            stress = self.all_data[idx, h['active_stress'], :]
            
            # Compute mean strain and stress. Note, we do not include the 
            # first point, this one deviates from the rest.
            mean_strain = np.mean(strain, axis=0)[1:]
            mean_stress = np.mean(stress, axis=0)[1:]       
            
            plt.plot(mean_strain, mean_stress, *args, **kwargs)

            plt.xlabel('Natural myofiber strain $\epsilon_f$ [-]', fontsize=fontsize)
            plt.ylabel('Active myofiber Cauchy stress $\sigma_f$ [kPa]', fontsize=fontsize)
            plt.tick_params(labelsize=fontsize-2)
            plt.grid('on')
            plt.axis(axlimits)
            plt.title(region_labels[ii], fontsize=fontsize+2)

    def plot_time_strain(self, *args, fig=None, fontsize=12, 
                         reference='onset_shortening', reorient_data=True,
                         **kwargs):
        if fig is None:
            # Create new figure.
            fig = plt.figure()
        
        # Make fig current.
        plt.figure(fig.number)

        regions = [self.extract_rv_idx(), 
                   self.extract_sep_idx(), 
                   self.extract_lv_idx()]
        
        region_labels = ['RV', 'SEP', 'LV']

#        all_regions = np.concatenate(regions)
#        regions.append(all_regions)
#        region_labels.append('Average')
        
        h = self.column_headers

        # Extract time array.
        results = self.results
        time = results['t_cycle']

        for ii, idx in enumerate(regions):

            if ii == 0:
                ax_1 = plt.subplot(1, len(regions), ii+1)
            else:
                ax = plt.subplot(1, len(regions), ii+1, sharex=ax_1, sharey=ax_1)
            
            if reference == 'onset_shortening':
                strain = self.compute_strain(self.all_data[idx, h['ls_old'], :])
            elif reference == 'stress_free':
                strain = self.compute_strain(self.all_data[idx, h['ls_old'], :], self.parameters['ls0'])
            else:
                raise ValueError('Invalid strain reference.')
                
            # Compute mean strain.
            mean_strain = np.mean(strain, axis=0)
            
            # Reorient data.
            if reorient_data:
                mean_strain = self.reorient_data(results, mean_strain, time_axis=-1)
                
            plt.plot(time, mean_strain, *args, **kwargs)

            plt.xlabel('Time [ms]', fontsize=fontsize)
            if ii == 0:
                # Only left plot gets an y-label
                plt.ylabel('Natural myofiber strain $\epsilon_f$ [-]', fontsize=fontsize)
            else:
                # make these tick labels invisible
                plt.setp(ax.get_yticklabels(), visible=False)
                
            plt.tick_params(labelsize=fontsize-2)
            plt.grid('on')
            plt.title(region_labels[ii], fontsize=fontsize+2)
        
    def plot_time_stress(self, *args, fig=None, fontsize=12, 
                         reorient_data=True, **kwargs):
        if fig is None:
            # Create new figure.
            fig = plt.figure()
        
        # Make fig current.
        plt.figure(fig.number)

        regions = [self.extract_rv_idx(), 
                   self.extract_sep_idx(), 
                   self.extract_lv_idx()]
        
        region_labels = ['RV', 'SEP', 'LV']

#        all_regions = np.concatenate(regions)
#        regions.append(all_regions)
#        region_labels.append('Average')
        
        h = self.column_headers

        # Extract time array.
        results = self.results
        time = results['t_cycle']

        for ii, idx in enumerate(regions):

            if ii == 0:
                ax_1 = plt.subplot(1, len(regions), ii+1)
            else:
                ax = plt.subplot(1, len(regions), ii+1, sharex=ax_1, sharey=ax_1)
                
            stress = self.all_data[idx, h['active_stress'], :]
            
            # Compute mean stress. Note, we do not include the first point 
            # (this one deviates from the rest).
            mean_stress = np.mean(stress, axis=0)[1:]
            
            # Reorient data.
            if reorient_data:
                mean_stress = self.reorient_data(results, mean_stress, time_axis=-1)
                
            plt.plot(time[:-1], mean_stress, *args, **kwargs)

            plt.xlabel('Time [ms]', fontsize=fontsize)
            if ii == 0:
                # Only left plot gets an y-label
                plt.ylabel('Active myofiber Cauchy stress $\sigma_f$ [kPa]', fontsize=fontsize)
            else:
                # make these tick labels invisible
                plt.setp(ax.get_yticklabels(), visible=False)
                
            plt.tick_params(labelsize=fontsize-2)
            plt.grid('on')
            plt.title(region_labels[ii], fontsize=fontsize+2)        

    def read_column_headers(self):
        """ 
        Returns dictionary with measure as key and column index 
        (for self.all_data) as values.
        """
        # Read headers from one file.
        file = self.files[0]
        with open(file) as f:
            all_lines=[line.split() for line in f]
            headers = all_lines[0][0].replace("'", '').replace('"', '').split(',')
        
        # Create dictionary with headers and indices.
        headers_dict = {}
        for i, h in enumerate(headers):
            headers_dict[h] = i
        
        return headers_dict

    @staticmethod
    def reorient_data(results, data, time_axis=-1):
        """
        Reorients the data along the specified time axis (axis that corresponds
        to the time points), such that the first index corresponds to the 
        moment of start of activation (t_act = 0).
        """
        # Find index where t_act is zero.
        t_act = np.array(results['t_act'])
        shift = np.argmin(abs(t_act))  
        
        # Shift the data along the last axis with by an amount shift.
        data_shifted = shift_data(data, shift, axis=time_axis)[0]
        
        return data_shifted
        
    def return_local_function_summary(self, reference='onset_shortening'):
        regions = [self.extract_rv_idx(), 
                   self.extract_sep_idx(), 
                   self.extract_lv_idx()]
        
        region_labels = ['RV', 'SEP', 'LV']

        all_regions = np.concatenate(regions)
        regions.append(all_regions)
        region_labels.append('Average')
        
        h = self.column_headers

        out = {}
        for ii, idx in enumerate(regions):  
            stress = self.all_data[idx, h['active_stress'], :]
            
            if reference == 'onset_shortening':
                strain = self.compute_strain(self.all_data[idx, h['ls_old'], :])
            elif reference == 'stress_free':
                strain = self.compute_strain(self.all_data[idx, h['ls_old'], :], self.parameters['ls0'])
            else:
                raise ValueError('Invalid strain reference.')     
                
            max_stress = np.max(stress, axis=1)
            max_strain = np.max(strain, axis=1)
            min_strain = np.min(strain, axis=1)
            work = self.compute_work(stress, strain)

            label = region_labels[ii]
            out_i = {'max_strain_{}'.format(label): np.mean(max_strain),
                     'min_strain_{}'.format(label): np.mean(min_strain),
                     'range_strain_{}'.format(label): np.mean(max_strain - min_strain),
                     'max_stress_{}'.format(label): np.mean(max_stress),
                     'fiber_work_{}'.format(label): np.mean(work)}            
            out.update(out_i)
        return out

    def show_global_function(self, fontsize=12):
    
        print('\n--------------------------')
        print('Global function '+self.parameters['name'])
        print('--------------------------')

        results = self.results
        t_cycle = max(results['time']) - min(results['time'])
        heartrate = 1000/t_cycle * 60
        
        parts = ['Systemic', 'Pulmonary']
    
        for ii, p in enumerate(['_s', '_p']):
            vcav = np.array(results['vcav'+p])
            pcav = np.array(results['pcav'+p])
            part = np.array(results['part'+p])
            
            # Close the loop.
            pcav = np.hstack((pcav, pcav[0:1]))
            vcav = np.hstack((vcav, vcav[0:1]))
            work = -np.trapz(pcav, x=vcav)/1000   
            
            edv = max(vcav)
            esv = min(vcav)
            
            sv = edv - esv
                        
            co = sv/1000*heartrate
            
            ef = sv/edv
            
            part_mean = kPa_to_mmHg(np.mean(part))
            
            print('')
            print(parts[ii])
            print('--------------------------')
            print('HR: {0:1.0f} bpm'.format(heartrate))
            print('W: {0:1.3g} J'.format(work))
            print('CO: {0:1.3g} L/min'.format(co))
            print('EF: {0:1.3g} [-]'.format(ef))
            print('SV: {0:1.3g} ml'.format(sv))
            print('MAP : {0:1.3g} mmHg'.format(part_mean))
    
    def show_local_function(self, *args, fontsize=12, **kwargs):

        print('\n--------------------------')
        print('Local function '+self.parameters['name'])
        print('--------------------------')

        results = self.results
        h = self.column_headers
        
        regions = [self.extract_rv_idx(), 
                   self.extract_sep_idx(), 
                   self.extract_lv_idx()]

        all_regions = np.concatenate(regions)
        
        stress = self.all_data[all_regions, h['active_stress'], :]
        ls = self.all_data[all_regions, h['ls_old'], :]
        strain = self.compute_strain(ls)
        work = self.compute_work(stress, strain)
        
        avg_work = np.mean(work)
        std_work = np.std(work)
        
        max_stress = np.max(stress, axis=1)
        avg_max_stress = np.mean(max_stress)
        std_max_stress = np.std(max_stress)
        
        print('--------------------------')
        print('mean +- SD maximum stress : {0:1.3g} ({1:1.3g}) kPa'.format(avg_max_stress, std_max_stress))
        print('SD/mean maximum stress : {0:1.4g}'.format(std_max_stress/avg_max_stress))
        print('mean +- SD work : {0:1.3g} ({1:1.3g}) J'.format(avg_work, std_work))     
        print('SD/mean work : {0:1.4g}'.format(std_work/avg_work))
        
        # Iterate over phases.
        fig1 = plt.figure(figsize=(8, 6), dpi=100)
        fig2 = plt.figure(figsize=(8, 6), dpi=100)
        phase_labels = {1: 'Filling',
                        2: 'Isovolumic contraction',
                        3: 'Ejection',
                        4: 'Isovolumic relaxation'}
        linestyles = ['-', '--', '-.', ':']
        for ii, phase in enumerate([1, 2, 3, 4]):
            mask_phase = results['phase_s'] == phase
            ls_phase = ls[:, mask_phase]
            if not np.size(ls_phase) == 0:
                strain_phase = self.compute_strain(ls_phase, ls0=ls_phase[:, 0])
                stress_phase = stress[:, mask_phase]
                            
                avg_strain_phase = np.mean(strain_phase[:,-1])
                std_strain_phase = np.std(strain_phase[:,-1])
                
                # Print stats.
                print('---------------')
                print('{} phase'.format(phase_labels[phase]))
                print('mean +- SD strain : {0:1.3g} ({1:1.3g})'.format(avg_strain_phase, std_strain_phase))
                print('SD/mean strain : {0:1.4g}'.format(abs(std_strain_phase/avg_strain_phase)))
            
                plt.figure(fig1.number)
                # Plot mean stress strain per phase.
                plt.plot(np.mean(strain[:, mask_phase], axis=0), np.mean(stress_phase, axis=0), linestyles[ii], *args, label=phase_labels[phase], **kwargs)
                # Shade interquartile range.
                q1_stress_phase = np.percentile(stress_phase, 25, axis=0)
                q3_stress_phase = np.percentile(stress_phase, 75, axis=0)
                plt.fill_between(np.mean(strain[:, mask_phase], axis=0), q3_stress_phase, q1_stress_phase, alpha=.5)
    
                plt.figure(fig2.number)
                # Plot mean strain stress per phase.
                plt.plot(np.mean(stress_phase, axis=0), np.mean(strain[:, mask_phase], axis=0), linestyles[ii], *args, label=phase_labels[phase], **kwargs)
                # Shade interquartile range.
                q1_strain_phase = np.percentile(strain[:, mask_phase], 25, axis=0)
                q3_strain_phase = np.percentile(strain[:, mask_phase], 75, axis=0)
                plt.fill_between(np.mean(stress_phase, axis=0), q3_strain_phase, q1_strain_phase, alpha=.5)

        plt.figure(fig1.number)
        plt.legend(title='Phase', fontsize=fontsize)
        plt.grid('on')
        plt.title('Active myofiber Cauchy stress-natural strain loop per phase', 
                  fontsize=fontsize+2)
        plt.xlabel('Natural strain $\epsilon_f$ [-]', fontsize=fontsize)
        plt.ylabel('Active stress $\sigma_f$ [kPa]', fontsize=fontsize)
        plt.tick_params(labelsize=fontsize-2)
        plt.axis([-0.2, 0.01, -1, 45])
        fname = os.path.join(os.path.split(self.directory)[0], 'average_stress_strain_loop.png')
        plt.savefig(fname,  dpi=300, bbox_inches="tight")
        
        plt.figure(fig2.number)
        plt.legend(title='Phase', fontsize=fontsize)
        plt.grid('on')
        plt.title('Natural strain-active myofiber Cauchy stress loop per phase', 
                  fontsize=fontsize+2)
        plt.ylabel('Natural strain $\epsilon_f$ [-]', fontsize=fontsize)
        plt.xlabel('Active stress $\sigma_f$ [kPa]', fontsize=fontsize)
        plt.tick_params(labelsize=fontsize-2)
        plt.axis([-1, 50, -0.175, 0.01])
        fname = os.path.join(os.path.split(self.directory)[0], 'average_stain_stress_loop.png')
        plt.savefig(fname,  dpi=300, bbox_inches="tight")

    def show_regions(self, projection='2d', fontsize=12, skip=None):
        h = self.column_headers
        # Plot the regions.
        fig = plt.figure()
        if projection == '3d':
            # ax = fig.add_subplot(111, projection='3d', aspect='equal')
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
                        
        col = ['C7', 'C0', 'C1', 'C2']
        # col = ['w', 'g', 'r', 'c']
        if projection == '2d':
            # Only select nodes from slice.
            region0 = np.where(abs(self.all_data[:, h[':1'], 0]) <= self.parameters['slice_thickness']/2)[0]
        else:
            # Select all nodes.
            region0 = np.arange(len(self.all_data[:, h[':1'], 0]))
            
        regions = [region0,
                   self.extract_rv_idx(), 
                   self.extract_sep_idx(), 
                   self.extract_lv_idx()]
        
        region_labels = [None, 
                         'RV', 
                         'SEP', 
                         'LV']
        for ii, idx in enumerate(regions):
            if ii == 0 and skip is not None and skip != 0:
                # Only select a random batch of relative size 1/skip to plot excluded nodes.
                random.shuffle(idx)
                idx = idx[:int(len(idx)/skip)]

            x = self.all_data[idx, h[':0'], 0]
            y = self.all_data[idx, h[':1'], 0]
            z = self.all_data[idx, h[':2'], 0]
            if projection == '3d':
                ax.scatter3D(x, z, y, color=col[ii], label=region_labels[ii])
            else:
                ax.scatter(x, z, color=col[ii], label=region_labels[ii])
        plt.legend(frameon=False, fontsize=fontsize)
        plt.title('Nodes included in local function analysis', fontsize=fontsize+2)
        plt.axis('off')
 
        # Print some information on the number of nodes in the regions.
        N_s = [len(indices) for indices in regions]
        N_s = np.asarray(N_s)
        N_tot = len(self.all_data)
        
        for ii in range(1, len(N_s)):
            print('% of nodes in {0} section: {1:1.2f} %'.format(region_labels[ii], N_s[ii]/N_tot*100))
            
        print('% of nodes in all sections : {0:1.2f} %'.format(sum(N_s[1:])/N_tot*100))
        print('% of nodes in entire slice: {0:1.2f} %'.format(N_s[0]/N_tot*100))
        

def common_start(sa, sb):
    """ returns the longest common substring from the beginning of sa and sb """
    def _iter():
        for a, b in zip(sa.split('\\'), sb.split('\\')):
            if a == b:
                yield a+'\\'
            else:
                return

    return ''.join(_iter())

def find_paraview_data(rpm=None, cycle=None, strain_reference=None, simulation_type='BiV'):
    """
    Returns the directory with paraview data for the given pump speed.
    """
    SIMULATION_DATA_PATH = get_paths()['SIMULATION_DATA_PATH']
    if simulation_type == 'BiV':
        if rpm <= - 1:
            # Dirty fix: if rpm = -1, we load the healthy case.
            results_dir = os.path.join(SIMULATION_DATA_PATH, 
                                       r'biv_realcycle\reorientation\REF\new_mesh_reorientation_4')            
        else:
            # Directory with simulations at different pump speeds.
            hc_dir = os.path.join(SIMULATION_DATA_PATH, 
                                  r'biv_realcycle\patient_simulation')
    else:
        raise ValueError('Unknown "simulation_type".')
        
    if not rpm <= -1:
        # Find the directory with the requested pump speed.
        results_dir = glob.glob(os.path.join(hc_dir, str(rpm)+'_rpm*'))[0]
    
    # Select cycle number.
    if strain_reference is None:
        all_cycle_dirs = glob.glob(os.path.join(results_dir, 'cycle*'))
    else:
        all_cycle_dirs = glob.glob(os.path.join(results_dir, 'cycle*{}'.format(strain_reference)))
    
    c_max = -1
    cycle_dir = None
    for d in all_cycle_dirs:
        if strain_reference is not None:
            c = int(os.path.split(d)[1].split('_')[1])
        else:
            try:
                c = int(os.path.split(d)[1][6:])
            except ValueError: # cannot convert to int
                # The cycle file is specified by additional information after
                # the cycle number which is not the strain reference: 
                # ignore these directories.
                c = -1
        if cycle is None:
            # Select highest cycle.
            if c > c_max:
                c_max = c
                cycle_dir = d
                
        elif c == cycle:
            cycle_dir = d
            break
        
    if cycle_dir is None or not os.path.exists(os.path.join(cycle_dir, 'paraview_data')):
        raise RuntimeError('Paraview directory not found.')
        
    return os.path.join(cycle_dir, 'paraview_data')

