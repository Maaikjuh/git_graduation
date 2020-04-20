# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:11:24 2020

@author: Maaike
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
import math

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

def ellips_to_cartesian(focus,eps,theta,phi):
    x= focus * math.sinh(eps) * math.sin(theta) * math.cos(phi)
    y = focus * math.sinh(eps) * math.sin(theta) * math.sin(phi)
    z = focus * math.cosh(eps) * math.cos(theta)
    return x, y, z

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

class postprocess_paraview_new(object):
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
        # self._idx_rv_data = None
        # self._idx_sep_data = None
        
        self._idx_ls0 = None
        
        self.column_headers = self.read_column_headers()
        
    @staticmethod
    def default_parameters():
        #Todo change
        par = {}
        par['cut_off_low'] = -3.
        par['cut_off_high'] = -2.5
        par['slice_thickness'] = .5
        par['theta'] = 7/10*math.pi
        par['AM_phi'] = 1/5*math.pi
        par['A_phi'] = 1/2*math.pi
        par['AL_phi'] = 4/5*math.pi
        par['inner_e']= 0.934819
        par['outer_e']= 0.807075
        par['focus']= 4.3
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
        #TODO check log
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
        ls0 = self.parameters['ls0']
        stretch_ratio = ls/ls0
        return np.log(stretch_ratio)
        
    @property
    def idx_ls0(self):
        if self._idx_ls0 is None:
            # Find earliest moment of shortening among all selected fibers.
            regions = [self.extract_AM_idx(), 
            self.extract_A_idx(), 
            self.extract_AL_idx()]    
            
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
    def idx_AM_data(self):
        if self._idx_AM_data is None:
            self._idx_lv_data = self.extract_AM_idx()
        return self._idx_AM_data
    
    @property
    def idx_A_data(self):
        if self._idx_A_data is None:
            self._idx_A_data = self.extract_A_idx()
        return self._idx_A_data
    
    @property
    def idx_AL_data(self):
        if self._idx_AL_data is None:
            self._idx_AL_data = self.extract_AL_idx()
        return self._idx_AL_data
    
    @property
    def idx_T0_data(self):
        if self._idx_T0_data is None:
            self._idx_T0_data = self.extract_T0_idx()
        return self._idx_T0_data
        
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

   
        
    def extract_AM_idx(self):
        #TODO ellipsoidal to cartesian
        # Extract points AM
        h = self.column_headers
        data = self.all_data[:, :, 0]
        # x = data[:, h[':0']]
        # y = data[:, h[':1']]
        # z = data[:, h[':2']]
        # # mask_cut_off_low = data[:, h[':2']] >= self.parameters['cut_off_low']
        # mask_cut_off_high = data[:, h[':2']] <= self.parameters['cut_off_high']
        if self.parameters['A_phi'] == 0:
            mask_cut_off_right = data[:, h[':0']] >= 0.
        else:
            mask_cut_off_right = data[:, h[':1']] >= 0.
        
        x, y, z = ellips_to_cartesian(self.parameters['focus'],self.parameters['outer_e'],self.parameters['theta'], self.parameters['AM_phi'])
        
        # pt1 = np.array([0,0,0])
        # pt2 = np.array([px,py,pz])
        # vec = np.subtract(pt2,pt1)
        # contst =  self.parameters['slice_thickness']/2 * np.linalg.norm(vec)
        
        
        # x1=[tel-pt1[0] for tel in x]
        # y1=[tel-pt1[1] for tel in y]
        # z1=[tel-pt1[2] for tel in z]
        
        # x2=[tel-pt2[0] for tel in x]
        # y2=[tel-pt2[1] for tel in y]
        # z2=[tel-pt2[2] for tel in z]
        
        
        # points1 = np.array([x1,y1,z1])
        # points2 = np.array([x2,y2,z2])
        # mask_1 = np.dot(points1[:,0],vec)
        # mask_2 = np.dot(points2[:,0],vec)
        # mask_3 = np.linalg.norm(np.cross(points1[:,0], vec))
        # for i in range(1,len(data)):
        #     mask_1=np.append(mask_1,np.dot(points1[:,i],vec))
        #     mask_2=np.append(mask_2,np.dot(points2[:,i],vec))
        #     mask_3=np.append(mask_3,np.linalg.norm(np.cross(points1[:,i], vec)))
        
        # # mask_1 = [np.dot(tel,vec) for i in range(0,len(data)) for tel in points1[:,i]]

        # # mask_1 = np.dot(points1, vec) >=0
        # mask_1 = mask_1>=0
        # mask_2 = mask_2>=0
        # mask_3 = mask_3<=contst
        # mask_tot = mask_1*mask_2*mask_3
        # # mask_2 = np.dot(np.array([x,y,z])-pt2, vec) <=0
        # # mask_3 = np.linalg.norm(np.cross(np.array([x,y,z])-pt1, vec)) <= contst
        
        x2, y2, z2 = ellips_to_cartesian(self.parameters['focus'],self.parameters['inner_e'],self.parameters['theta'], self.parameters['AM_phi'])


        data_T0 = self.all_data[:, h['f_135'], 0]
        
        mask_tot_A = data_T0 >= 90
        
        
        mask_cut_off_low = data[:, h[':2']] >= z - self.parameters['slice_thickness']/2
        mask_cut_off_high = data[:, h[':2']] <= z + self.parameters['slice_thickness']/2
        mask_cut_off_line_left = data[:, h[':1']] >= (x-x2)/(y-y2)*data[:, h[':0']] -  self.parameters['slice_thickness']/2 
        mask_cut_off_line_right = data[:, h[':1']] <= (x-x2)/(y-y2)*data[:, h[':0']] +  self.parameters['slice_thickness']/2 
        # mask_cut_off_line_high = data[:, h[':2']] <= z* abs(1-(y + x/y*data[:, h[':0']]))+  self.parameters['slice_thickness']/2 
        # mask_cut_off_line_low = data[:, h[':2']] <= z* abs(1-(y + x/y*data[:, h[':0']]))-  self.parameters['slice_thickness']/2 


        # # mask_tot = mask_cut_off_line_low*mask_cut_off_line_high*mask_cut_off_line_left*mask_cut_off_line_right*mask_cut_off_right
        mask_tot = mask_tot_A*mask_cut_off_low*mask_cut_off_high*mask_cut_off_line_left*mask_cut_off_line_right*mask_cut_off_right

        # mask_tot = mask_1 * mask_2 * mask_3 * mask_cut_off_right
        return np.where(mask_tot)[0]  
        # return (np.array([2469,2345,3805]))

    def extract_A_idx(self):
        # Extract points A
        h = self.column_headers
        data = self.all_data[:, :, 0]
        x, y, z = ellips_to_cartesian(self.parameters['focus'],self.parameters['outer_e'],self.parameters['theta'], self.parameters['A_phi'])

        mask_cut_off_low = data[:, h[':2']] >= z - self.parameters['slice_thickness']/2
        mask_cut_off_high = data[:, h[':2']] <= z + self.parameters['slice_thickness']/2
        if self.parameters['A_phi'] == 0:
            mask_cut_off_right = data[:, h[':0']] >= 0.
            mask_cut_off_slice = abs(data[:, h[':1']]) <= self.parameters['slice_thickness']/2
        else:
            mask_cut_off_right = data[:, h[':1']] >= 0.
            mask_cut_off_slice = abs(data[:, h[':0']]) <= self.parameters['slice_thickness']/2        
        # mask_cut_off_line_left = data[:, h[':1']] >= x/y*data[:, h[':0']] -  self.parameters['slice_thickness']/2 
        # mask_cut_off_line_right = data[:, h[':1']] <= x/y*data[:, h[':0']] +  self.parameters['slice_thickness']/2 
        # mask_tot = mask_cut_off_low*mask_cut_off_high*mask_cut_off_right*mask_cut_off_line_left*mask_cut_off_line_right
        mask_tot = mask_cut_off_low*mask_cut_off_high*mask_cut_off_right*mask_cut_off_slice
        return np.where(mask_tot)[0]  
        # return (np.array([2006,3950,3875]))
    
    def extract_AL_idx(self):
        # Extract points AL
        h = self.column_headers
        data = self.all_data[:, :, 0]
        # mask_cut_off_low = data[:, h[':2']] >= self.parameters['cut_off_low']
        # mask_cut_off_high = data[:, h[':2']] <= self.parameters['cut_off_high']
        if self.parameters['A_phi'] == 0:
            mask_cut_off_right = data[:, h[':0']] >= 0.
        else:
            mask_cut_off_right = data[:, h[':1']] >= 0.
        
        x, y, z = ellips_to_cartesian(self.parameters['focus'],self.parameters['outer_e'],self.parameters['theta'], self.parameters['AL_phi'])
        x2, y2, z2 = ellips_to_cartesian(self.parameters['focus'],self.parameters['inner_e'],self.parameters['theta'], self.parameters['AL_phi'])
 
        data_T0 = self.all_data[:, h['f_135'], 0]
        
        mask_tot_A = data_T0 >= 90
        
        mask_cut_off_low = data[:, h[':2']] >= z - self.parameters['slice_thickness']/2
        mask_cut_off_high = data[:, h[':2']] <= z + self.parameters['slice_thickness']/2
        mask_cut_off_line_left = data[:, h[':1']] >= (x-x2)/(y-y2)*data[:, h[':0']] -  self.parameters['slice_thickness']/2 
        mask_cut_off_line_right = data[:, h[':1']] <= (x-x2)/(y-y2)*data[:, h[':0']] +  self.parameters['slice_thickness']/2 
        # mask_cut_off_line_left = 1.73/2.37*data[:, h[':0']] -  self.parameters['slice_thickness']/2 <=data[:, h[':1']]
        # mask_cut_off_line_right = 1.73/2.37*data[:, h[':0']] +  self.parameters['slice_thickness']/2 >=data[:, h[':1']]

        mask_tot = mask_tot_A*mask_cut_off_low*mask_cut_off_high*mask_cut_off_line_left*mask_cut_off_line_right*mask_cut_off_right
        return np.where(mask_tot)[0]
        # return (np.array([427,1417,1848]))
    
    def extract_T0_idx(self):
        h = self.column_headers
        data_T0 = self.all_data[:, h['f_135'], 0]
        return np.where(data_T0 <= 90)
        
    
         
    
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
    
    def plot_pv_loops(self, *args, fig=None, fontsize=12, **kwargs):
        if fig is None:
            # Create new figure.
            fig = plt.figure()
        
        # Make fig current.
        plt.figure(fig.number)
        
        results = self.results
        
        if 'vcav_s' in results.keys():
            for ii, p in enumerate(['_s', '_p']):
                vcav = np.array(results['vcav'+p])
                pcav = kPa_to_mmHg(np.array(results['pcav'+p]))
                plt.plot(vcav, pcav, *args, **kwargs)
        elif 'vlv' in results.keys():
            vcav = np.array(results['vlv'])
            pcav = kPa_to_mmHg(np.array(results["plv"]))
            plt.plot(vcav, pcav, *args, **kwargs)
        
        
        kwargs['label'] = None  # Only label the first plot. 

        plt.xlabel('Cavity volume [ml]', fontsize=fontsize)
        plt.ylabel('Cavity pressure [mmHg]', fontsize=fontsize)
        plt.tick_params(labelsize=fontsize-2)
        plt.grid('on')
        plt.axis([40, 140, 0, 120])
        plt.title('Pressure-volume loops', fontsize=fontsize+2)
        
    def plot_time_stress(self, *args, fig=None, fontsize=12,**kwargs):
        if fig is None:
            # Create new figure.
            fig = plt.figure()
        
        # Make fig current.
        plt.figure(fig.number)
        
        regions = [self.extract_AM_idx(), 
                   self.extract_A_idx(), 
                   self.extract_AL_idx()]
        region_labels = ['AM', 'A', 'AL']
        
        h = self.column_headers
                # Extract time array.
        results = self.results
        time = results['t_cycle']
        t0 = min(time)
        tend = max(time)
        
        for ii, idx in enumerate(regions):
            plt.subplot(1, len(regions), ii+1)
            stress = self.all_data[idx, h['active_stress'], :]
            mean_stress = np.mean(stress, axis=0)[:]
            
            plt.plot(time, mean_stress, *args, **kwargs)
            
            plt.xlabel('time', fontsize=fontsize)
            plt.ylabel('Ta [kPa]', fontsize=fontsize)
            plt.tick_params(labelsize=fontsize-2)
            plt.grid('on')
            plt.title(region_labels[ii], fontsize=fontsize+2) 
            plt.axis([t0, tend, 0, 80])
        
    def plot_stress_ls_l0(self, *args, fig=None, fontsize=12, 
                           reference='onset_shortening',
                            
                           **kwargs):
        if fig is None:
            # Create new figure.
            fig = plt.figure()
        
        # Make fig current.
        plt.figure(fig.number)

        regions = [self.extract_AM_idx(), 
                   self.extract_A_idx(), 
                   self.extract_AL_idx()]
        region_labels = ['AM', 'A', 'AL']

#        all_regions = np.concatenate(regions)
#        regions.append(all_regions)
#        region_labels.append('Average')
        
        h = self.column_headers
        ls0 = self.parameters['ls0']
        
        for ii, idx in enumerate(regions):
            plt.subplot(1, len(regions), ii+1)
            
            lsl0 = (self.all_data[idx, h['ls_old'], :])/ls0
            
            # Load stress.
            stress = self.all_data[idx, h['active_stress'], :]
            
            # Compute mean strain and stress. Note, we do not include the 
            # first point, this one deviates from the rest.
            mean_lsl0 = np.mean(lsl0, axis=0)[:]
            mean_stress = np.mean(stress, axis=0)[:]       
            
            plt.plot(mean_lsl0, mean_stress, *args, **kwargs)

            plt.xlabel('ls/ls0', fontsize=fontsize)
            plt.ylabel('Ta [kPa]', fontsize=fontsize)
            plt.tick_params(labelsize=fontsize-2)
            plt.grid('on')
            plt.axis([0.9, 1.2, 0., 80.])
            plt.title(region_labels[ii], fontsize=fontsize+2)  
            
            
    def plot_stress_strain_ls(self, *args, fig=None, fontsize=12, 
                           reference='onset_shortening', var='strain',
                           axlimits=[-0.2, 0.02, -2, 52], 
                           **kwargs):
        
        if fig is None:
            # Create new figure.
            fig = plt.figure()
        
        # Make fig current.
        plt.figure(fig.number)

        regions = [self.extract_AM_idx(), 
                   self.extract_A_idx(), 
                   self.extract_AL_idx()]
        region_labels = ['AM', 'A', 'AL']

#        all_regions = np.concatenate(regions)
#        regions.append(all_regions)
#        region_labels.append('Average')
        
        h = self.column_headers
        
        for ii, idx in enumerate(regions):
            plt.subplot(1, len(regions), ii+1)
            
            # Load strain.
            if var =='strain':
                if reference == 'onset_shortening':
                    strain = self.compute_strain(self.all_data[idx, h['ls_old'], :])
                elif reference == 'stress_free':
                    strain = self.compute_strain(self.all_data[idx, h['ls_old'], :], self.parameters['ls0'])
                else:
                    raise ValueError('Invalid strain reference.')
            elif var == 'ls':
                strain = self.all_data[idx, h['ls_old'], :]
            
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
            
    def plot_time_strain_ls(self, *args, fig=None, fontsize=12, 
                         reference='onset_shortening', reorient_data=True,var='strain',
                         **kwargs):
        if fig is None:
            # Create new figure.
            fig = plt.figure()
        
        # Make fig current.
        plt.figure(fig.number)

        regions = [self.extract_AM_idx(), 
                   self.extract_A_idx(), 
                   self.extract_AL_idx()]
        
        region_labels = ['AM', 'A', 'AL']

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
            
            if var == 'strain':
                if reference == 'onset_shortening':
                    strain = self.compute_strain(self.all_data[idx, h['ls_old'], :])
                elif reference == 'stress_free':
                    strain = self.compute_strain(self.all_data[idx, h['ls_old'], :], self.parameters['ls0'])
                # strain = self.all_data[idx, h['ls_old'], :]
                else:
                    raise ValueError('Invalid strain reference.')
            if var == 'ls':
                strain = self.all_data[idx, h['ls_old'], :]
                
            # Compute mean strain.
            mean_strain = np.mean(strain, axis=0)
            common_start
            # Reorient data.
            if reorient_data:
                mean_strain = self.reorient_data(results, mean_strain, time_axis=-1)
                
            plt.plot(time, mean_strain, *args, **kwargs)

            plt.xlabel('Time [ms]', fontsize=fontsize)
            if ii == 0:
                if var == 'strain':
                # Only left plot gets an y-label
                    plt.ylabel('Natural myofiber strain $\epsilon_f$ [-]', fontsize=fontsize)
                elif var =='ls':
                    plt.ylabel('ls um', fontsize=fontsize)
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
        try: 
            file = self.files[0]
        except ValueError:
            print('No submap available')  
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

    def show_regions_new(self, projection='2d', fontsize=12, skip=None):
        h = self.column_headers
        # Plot the regions.
        fig = plt.figure()
        if projection == '3d':
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
                        
        col = [ 'C7','C5', 'C0', 'C1','C2']
        # col = ['g', 'r', 'c','w']
        if projection == '2d':
            # Only select nodes from slice.
            region0 = np.where((self.all_data[:, h[':1'], 0]) >= 0.)
        else:
            # Select all nodes.
            region0 = self.all_data
            region0 = np.where((self.all_data[:, h[':1'], 0]) >= -4.)
            # region0 = np.arange(len(self.all_data[:, h[':1'], 0]))
            
        regions = [region0,
                   self.extract_T0_idx(),
                   self.extract_AM_idx(), 
                   self.extract_A_idx(), 
                   self.extract_AL_idx()]
        # regions = [region0,
        #            np.array([2469, 4259,3805]),]
        
        region_labels = [None,
                         'T0',
                        'AM', 
                         'A', 
                         'AL'
                         
                         ]
        idx_tot =[]
        idx_tot.extend(regions[1:5])

        for ii, idx in enumerate(regions):
            if ii == 0 and skip is not None and skip != 0:
                # Only select a random batch of relative size 1/skip to plot excluded nodes.
                random.shuffle(idx)
                idx = idx[:int(len(idx)/skip)]
                
            # if ii ==0:
            #     # idx =idx[0]
            #     # idx = [np.delete(idx,int(np.argwhere(idx==i))) for dat in regions[0] for i in dat for row in idx_tot if i in row ]
            #     idx =idx[0]
            #     idx_tot = idx_tot[0]
            #     for data in regions[0]:
            #         for i in data:               
            #             for row in idx_tot:
            #                 if i in row:  
            #                     idx = np.delete(idx,int(np.argwhere(idx == i)))

            x = self.all_data[idx, h[':0'], 0]
            y = self.all_data[idx, h[':1'], 0]
            z = self.all_data[idx, h[':2'], 0]
                
            if projection == '3d':
                # if ii ==0:
                #     ax.scatter3D(x, z, y, color=col[ii], label=region_labels[ii],marker='.',zdir='y')
                if ii != 0:
                    ax.scatter3D(x, z, y, color=col[ii], label=region_labels[ii],zdir='y')
            else:
                ax.scatter(x, z, color=col[ii], label=region_labels[ii])
        # if projection == '3d':
        #     ax.view_init(1, 1)
            # ax.set_aspect('equal')
        plt.legend(frameon=False, fontsize=fontsize)
        plt.title('Nodes included in local function analysis', fontsize=fontsize+2)
        # plt.axis('off')
 
        # Print some information on the number of nodes in the regions.
        regions[0] = (regions[0])[0]
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

