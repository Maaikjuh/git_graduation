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
import warnings

warnings.filterwarnings("ignore")

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
        
        if 'T0' in self.column_headers:
            if 'infarct' not in kwargs:
                up_inf = {'infarct': True}
                self._parameters.update(up_inf)
        
    @staticmethod
    def default_parameters():
        #Todo change
        par = {}
        par['infarct'] = False
        par['cut_off_low'] = -3.
        par['cut_off_high'] = -2.5
        par['slice_thickness'] = .05
        par['theta'] = 7/10*math.pi
        par['AM_phi'] = 1/5*math.pi
        par['A_phi'] = 1/2*math.pi
        par['AL_phi'] = 4/5*math.pi
        par['P_phi'] = par['A_phi'] + math.pi
        par['inner_e']= 0.3713
        par['outer_e']= 0.6784
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
    
    def compute_wall_thickness(self, height):
        h = self.column_headers
        P_epi_idx = self.extract_P_idx()[0]
        P_endo_idx = self.extract_P_idx()[-1]
        
        P_epi_x = self.all_data[P_epi_idx, h['displacement:0'], :] + self.all_data[P_epi_idx, h[':0'], 0]
        P_epi_y = self.all_data[P_epi_idx, h['displacement:1'], :] + self.all_data[P_epi_idx, h[':1'], 0]
        
        AM_epi_idx = self.extract_AM_idx()[0]
        AM_endo_idx = self.extract_AM_idx()[-1]
        
        AM_epi_idx = self.extract_AM_idx()[0]
        AM_endo_idx = self.extract_AM_idx()[-1]
        
        
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
    
    def extract_wall_idx(self, phi_val= ''):
        #select single point on epi, mid and endowall respectively
        h = self.column_headers
        data = self.all_data[:, :, 0]

        # #check on which side (x-z or y-z) of the heart the points are located
        # #assumes that if A_phi ==0 on the y-z plane (with positive x) and otherwhise on the x-z plane (with positive y)
        # if self.parameters['A_phi'] == 0:
        #     mask_cut_off_right = data[:, h[':0']] >= 0.
        #     mask_cut_off_right_P = data[:, h[':0']] <= 0.
        # else:
        #     mask_cut_off_right = data[:, h[':1']] >= 0.
        #     mask_cut_off_right_P = data[:, h[':1']] <= 0.
            
        #check if infarct is included and create mask such that points AM and AL are not located within the infarct
        if self.parameters['infarct'] == True:
            if 'T0' in h:
                T0 = 'T0'
            elif 'f_135' in h:
                T0 = 'f_135'
            elif 'f_137' in h:
                T0 = 'f_137'
            data_T0 = self.all_data[:, h[T0], 0]
            mask_T0 = data_T0 >= 90
        
        #get coordinates on the outer and inner wall for phi
        x, y, z = ellips_to_cartesian(self.parameters['focus'],self.parameters['outer_e'],self.parameters['theta'], self.parameters[phi_val])
        x2, y2, z2 = ellips_to_cartesian(self.parameters['focus'],self.parameters['inner_e'],self.parameters['theta'], self.parameters[phi_val])

        mask = {}
        mask_tot = {}
        coordname = ['x','y','z']
        loc = ['epi','mid','endo']
        mask_tot_loc = [False]
        for wall in loc:
            #creates masks for x, y and z for each point on the wall
            #points must be within the the slice thickness boundary
            #if no point within the boundary can be found, the slice thickness is increased
            #until there is/are a point(s) that is within the boundary in the x, y and z direction
            mask_tot_loc = [False]
            slice_thickness =  self.parameters['slice_thickness']/2 
            if wall == 'epi':
                coord = [x,y,z]
            elif wall == 'mid':
                coord = [(x+x2)/2,(y+y2)/2,(z+z2)/2]
            elif wall == 'endo':
                coord = [x2,y2,z2]
                
            while True not in mask_tot_loc:            
                for i in range(0,3):
                    #create mask in the x, y and z direction for a point on the wall
                    mask["mask_1_" + wall + str(coordname[i])] =  data[:, h[':'+str(i)]] >= coord[i] - slice_thickness 
                    mask["mask_2_" + wall + str(coordname[i])] =  data[:, h[':'+str(i)]] <= coord[i] + slice_thickness
                
                #create total mask to check if there is/are a point(s) that lies in all boundaries in all the directions
                mask_tot_loc_1 = mask["mask_1_" + wall + 'x'] * mask["mask_1_" + wall + 'y'] * mask["mask_1_" + wall + 'z']
                mask_tot_loc_2 = mask["mask_2_" + wall + 'x'] * mask["mask_2_" + wall + 'y'] * mask["mask_2_" + wall + 'z']
                if self.parameters['infarct'] == True and phi_val != 'A_phi':
                    mask_tot_loc = mask_tot_loc_1 * mask_tot_loc_2 * mask_T0
                else:
                    mask_tot_loc = mask_tot_loc_1 * mask_tot_loc_2
                    
                #increase boundary width
                slice_thickness = slice_thickness + 0.01
                
            mask_tot['mask_tot_' + wall] = mask_tot_loc  
            
        #only return one point per wall location (epi, mid or endo)
        return (np.array([np.where(mask_tot['mask_tot_epi'])[0][0],np.where(mask_tot['mask_tot_mid'])[0][0],np.where(mask_tot['mask_tot_endo'])[0][0]]))
        
    def extract_AM_idx(self):
        return self.extract_wall_idx(phi_val = 'AM_phi')
        # Extract points AM
        # h = self.column_headers
        # data = self.all_data[:, :, 0]
        # # x = data[:, h[':0']]
        # # y = data[:, h[':1']]
        # # z = data[:, h[':2']]
        # # # mask_cut_off_low = data[:, h[':2']] >= self.parameters['cut_off_low']
        # # mask_cut_off_high = data[:, h[':2']] <= self.parameters['cut_off_high']
        # if self.parameters['A_phi'] == 0:
        #     mask_cut_off_right = data[:, h[':0']] >= 0.
        # else:
        #     mask_cut_off_right = data[:, h[':1']] >= 0.
        
        # #cylinder method
        
        # # pt1 = np.array([0,0,0])
        # # pt2 = np.array([px,py,pz])
        # # vec = np.subtract(pt2,pt1)
        # # contst =  self.parameters['slice_thickness']/2 * np.linalg.norm(vec)
        
        
        # # x1=[tel-pt1[0] for tel in x]
        # # y1=[tel-pt1[1] for tel in y]
        # # z1=[tel-pt1[2] for tel in z]
        
        # # x2=[tel-pt2[0] for tel in x]
        # # y2=[tel-pt2[1] for tel in y]
        # # z2=[tel-pt2[2] for tel in z]
        
        
        # # points1 = np.array([x1,y1,z1])
        # # points2 = np.array([x2,y2,z2])
        # # mask_1 = np.dot(points1[:,0],vec)
        # # mask_2 = np.dot(points2[:,0],vec)
        # # mask_3 = np.linalg.norm(np.cross(points1[:,0], vec))
        # # for i in range(1,len(data)):
        # #     mask_1=np.append(mask_1,np.dot(points1[:,i],vec))
        # #     mask_2=np.append(mask_2,np.dot(points2[:,i],vec))
        # #     mask_3=np.append(mask_3,np.linalg.norm(np.cross(points1[:,i], vec)))
        
        # # # mask_1 = [np.dot(tel,vec) for i in range(0,len(data)) for tel in points1[:,i]]

        # # # mask_1 = np.dot(points1, vec) >=0
        # # mask_1 = mask_1>=0
        # # mask_2 = mask_2>=0
        # # mask_3 = mask_3<=contst
        # # mask_tot = mask_1*mask_2*mask_3
        # # # mask_2 = np.dot(np.array([x,y,z])-pt2, vec) <=0
        # # # mask_3 = np.linalg.norm(np.cross(np.array([x,y,z])-pt1, vec)) <= contst
        
        # if self.parameters['infarct'] == True:
        #     if 'T0' in h:
        #         T0 = 'T0'
        #     elif 'f_135' in h:
        #         T0 = 'f_135'
        #     elif 'f_137' in h:
        #         T0 = 'f_137'
        #     data_T0 = self.all_data[:, h[T0], 0]
        #     mask_T0 = data_T0 >= 90
    
        
        # line method
    
        # mask_cut_off_low = data[:, h[':2']] >= z - self.parameters['slice_thickness']/2
        # mask_cut_off_high = data[:, h[':2']] <= z + self.parameters['slice_thickness']/2
        # mask_cut_off_line_left = data[:, h[':1']] >= (x-x2)/(y-y2)*data[:, h[':0']] -  self.parameters['slice_thickness']/2 
        # mask_cut_off_line_right = data[:, h[':1']] <= (x-x2)/(y-y2)*data[:, h[':0']] +  self.parameters['slice_thickness']/2 
        # # mask_cut_off_line_high = data[:, h[':2']] <= z* abs(1-(y + x/y*data[:, h[':0']]))+  self.parameters['slice_thickness']/2 
        # # mask_cut_off_line_low = data[:, h[':2']] <= z* abs(1-(y + x/y*data[:, h[':0']]))-  self.parameters['slice_thickness']/2 


        # # # mask_tot = mask_cut_off_line_low*mask_cut_off_line_high*mask_cut_off_line_left*mask_cut_off_line_right*mask_cut_off_right
        
        # # mask_tot = mask_1 * mask_2 * mask_3 * mask_cut_off_right
        # if self.parameters['infarct'] == True:
        #     if 'T0' in h:
        #         T0 = 'T0'
        #     elif 'f_135' in h:
        #         T0 = 'f_135'
        #     elif 'f_137' in h:
        #         T0 = 'f_137'
        #     data_T0 = self.all_data[:, h[T0], 0]
        #     mask_T0 = data_T0 >= 90
        #     mask_tot = mask_T0*mask_cut_off_low*mask_cut_off_high*mask_cut_off_line_left*mask_cut_off_line_right*mask_cut_off_right
        # else:
        #     mask_tot =mask_cut_off_low*mask_cut_off_high*mask_cut_off_line_left*mask_cut_off_line_right*mask_cut_off_right

        # return np.where(mask_tot)[0]  
        # # return (np.array([2469,2345,3805]))

    def extract_A_idx(self):
        # Extract points A
        # h = self.column_headers
        # data = self.all_data[:, :, 0]
        # x, y, z = ellips_to_cartesian(self.parameters['focus'],self.parameters['outer_e'],self.parameters['theta'], self.parameters['A_phi'])

        # mask_cut_off_low = data[:, h[':2']] >= z - self.parameters['slice_thickness']/2
        # mask_cut_off_high = data[:, h[':2']] <= z + self.parameters['slice_thickness']/2
        # if self.parameters['A_phi'] == 0.:
        #     mask_cut_off_right = data[:, h[':0']] >= 0.
        #     mask_cut_off_slice = abs(data[:, h[':1']]) <= self.parameters['slice_thickness']/2
        # else:
        #     mask_cut_off_right = data[:, h[':1']] >= 0.
        #     mask_cut_off_slice = abs(data[:, h[':0']]) <= self.parameters['slice_thickness']/2        
        # # mask_cut_off_line_left = data[:, h[':1']] >= x/y*data[:, h[':0']] -  self.parameters['slice_thickness']/2 
        # # mask_cut_off_line_right = data[:, h[':1']] <= x/y*data[:, h[':0']] +  self.parameters['slice_thickness']/2 
        # # mask_tot = mask_cut_off_low*mask_cut_off_high*mask_cut_off_right*mask_cut_off_line_left*mask_cut_off_line_right
        # mask_tot = mask_cut_off_low*mask_cut_off_high*mask_cut_off_right*mask_cut_off_slice
        # return np.where(mask_tot)[0]  
        return self.extract_wall_idx(phi_val = 'A_phi')
        # return (np.array([2006,3950,3875]))
    
    def extract_AL_idx(self):
        # Extract points AL
        # h = self.column_headers
        # data = self.all_data[:, :, 0]
        # # mask_cut_off_low = data[:, h[':2']] >= self.parameters['cut_off_low']
        # # mask_cut_off_high = data[:, h[':2']] <= self.parameters['cut_off_high']
        # if self.parameters['A_phi'] == 0.:
        #     mask_cut_off_right = data[:, h[':0']] >= 0.
        # else:
        #     mask_cut_off_right = data[:, h[':1']] >= 0.
        
        # x, y, z = ellips_to_cartesian(self.parameters['focus'],self.parameters['outer_e'],self.parameters['theta'], self.parameters['AL_phi'])
        # x2, y2, z2 = ellips_to_cartesian(self.parameters['focus'],self.parameters['inner_e'],self.parameters['theta'], self.parameters['AL_phi'])

        
        # mask_cut_off_low = data[:, h[':2']] >= z - self.parameters['slice_thickness']/2
        # mask_cut_off_high = data[:, h[':2']] <= z + self.parameters['slice_thickness']/2
        # mask_cut_off_line_left = data[:, h[':1']] >= (x-x2)/(y-y2)*data[:, h[':0']] -  self.parameters['slice_thickness']/2 
        # mask_cut_off_line_right = data[:, h[':1']] <= (x-x2)/(y-y2)*data[:, h[':0']] +  self.parameters['slice_thickness']/2 
        # # mask_cut_off_line_left = 1.73/2.37*data[:, h[':0']] -  self.parameters['slice_thickness']/2 <=data[:, h[':1']]
        # # mask_cut_off_line_right = 1.73/2.37*data[:, h[':0']] +  self.parameters['slice_thickness']/2 >=data[:, h[':1']]

       
        # if self.parameters['infarct'] == True:
        #     if 'T0' in h:
        #         T0 = 'T0'
        #     elif 'f_135' in h:
        #         T0 = 'f_135'
        #     elif 'f_137' in h:
        #         T0 = 'f_137'
        #     data_T0 = self.all_data[:, h[T0], 0]
        #     mask_T0 = data_T0 >= 90
        #     mask_tot = mask_T0*mask_cut_off_low*mask_cut_off_high*mask_cut_off_line_left*mask_cut_off_line_right*mask_cut_off_right
        # else:
        #     mask_tot =mask_cut_off_low*mask_cut_off_high*mask_cut_off_line_left*mask_cut_off_line_right*mask_cut_off_right
        
        # return np.where(mask_tot)[0]
        return self.extract_wall_idx(phi_val = 'AL_phi')
        # return (np.array([427,1417,1848]))
    
    def extract_P_idx(self):
        self.parameters['P_phi'] = self.parameters['A_phi'] + math.pi
        return self.extract_wall_idx(phi_val = 'P_phi')
    
    def extract_T0_idx(self):
        h = self.column_headers
        if 'T0' in h:
            T0 = 'T0'
        elif 'f_135' in h:
            T0 = 'f_135'
        elif 'f_137' in h:
            T0 = 'f_137'
        data_T0 = self.all_data[:, h[T0], 0]
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
        
    def Ta_ls_loop(self,*args, fig = None, fontsize = 12, phase = True, label='', **kwargs):
        if fig is None:
            # Create new figure.
            fig = plt.figure()
        
        # Make fig current.
        plt.figure(fig.number)
        regions = [self.extract_P_idx()]
        # regions = [self.extract_P_idx(),
        #            self.extract_AM_idx(), 
        #            self.extract_A_idx(), 
        #            self.extract_AL_idx()]
        # region_labels = ['P','AM', 'A', 'AL']
        region_labels = ['']
        h = self.column_headers
        results = self.results
        
        time = results['t_cycle']
        t0 = np.amin(time)
        tend = np.amax(time)
        
        ls0 = self.parameters['ls0']
        
        for ii, idx in enumerate(regions):
            # ax1 = plt.subplot(3, len(regions), ii+1)
            plt.subplot(3, len(regions), ii+1)
            stress = self.all_data[idx, h['active_stress'], :]
            mean_stress = np.mean(stress, axis=0)[:]
            
            if len(time) != len(stress[0]):
                if len(time)>len(stress[0]):
                    print('time and stress array not of same length, shortening time array...')
                    time = time[0:len(stress[0])]


            if ii == len(regions)-1:
                plt.plot(time, mean_stress, *args, label=label)
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                # plt.legend(label, loc='center left', bbox_to_anchor=(1, 0.5
            else:
                plt.plot(time, mean_stress, *args)                
        
            
            plt.xlabel('time [ms]', fontsize=fontsize)
            if ii == 0:
                plt.ylabel('Ta [kPa]', fontsize=fontsize)
                

            
            plt.tick_params(labelsize=fontsize-2)
            plt.grid('on')
            plt.title(region_labels[ii], fontsize=fontsize+2) 
            plt.axis([t0, tend, 0, 60])
            
            # Mark the beginning of each phase
            phaselabel = ['d','ic','e','ir']
            if phase == True:
                for i in range(1,5):
                    if 'phase_s' in results.keys():
                        index = (results['phase_s'] == i).idxmax()
                    else: 
                        index = (results['phase'] == i).idxmax()
                    phase_time = results['t_cycle'][index]
                    plt.plot([phase_time, phase_time], [0, 80],'C7')
                    # axs[1][ii].plot([phase_time, phase_time], [0, 80],'C7')
                    
                    #plot phase labels
                    if i != 4:
                        #get the begin time of the next phase
                        if 'phase_s' in results.keys():
                            next_index = (results['phase_s'] == i+1).idxmax()
                        else: 
                            next_index = (results['phase'] == i+1).idxmax()
                        next_phase = results['t_cycle'][next_index]
                        #plot label between the two phases
                        plt.text((phase_time+next_phase)/2, 55, phaselabel[i-1],fontsize=13,horizontalalignment='center')
                    elif i == 4:
                        #plot the label of the last phase
                        plt.text((phase_time+max(time))/2, 55, phaselabel[i-1],fontsize=13,horizontalalignment='center')
    
                        
            ls = self.all_data[idx, h['ls_old'], :]
            mean_ls = np.mean(ls, axis=0)[:]
            
            plt.subplot(3, len(regions),  len(regions)+ii+1)
            plt.plot(time, mean_ls, *args, **kwargs)
            
            plt.xlabel('time [ms]', fontsize=fontsize)
            if ii == 0:
                plt.ylabel('ls [um]', fontsize=fontsize)
            
            plt.tick_params(labelsize=fontsize-2)
            plt.axis([min(time),max(time),1.8, 2.6])    
            plt.tick_params(labelsize=fontsize-2)
            plt.grid('on')
            
            # Mark the beginning of each phase
            phaselabel = ['d','ic','e','ir']
            if phase == True:
                for i in range(1,5):
                    if 'phase_s' in results.keys():
                        index = (results['phase_s'] == i).idxmax()
                    else: 
                        index = (results['phase'] == i).idxmax()
                    phase_time = results['t_cycle'][index]
                    plt.plot([phase_time, phase_time], [0, 80],'C7')
            
            lsl0 = ls/ls0
            
            # Load stress.
            stress = self.all_data[idx, h['active_stress'], :]
            
            # Compute mean strain and stress. 
            mean_lsl0 = np.mean(lsl0, axis=0)[:]
            mean_stress = np.mean(stress, axis=0)[:]       
            
            plt.subplot(3, len(regions),  len(regions)*2 +ii+1)
            plt.plot(mean_lsl0, mean_stress, *args, **kwargs)

            plt.xlabel('ls/ls0', fontsize=fontsize)
            if ii == 0:
                plt.ylabel('Ta [kPa]', fontsize=fontsize)
            
            plt.tick_params(labelsize=fontsize-2)
            plt.grid('on')
            plt.axis([0.95, 1.2, 0., 60.]) 
            
            
        
        
        
    def plot_time_stress(self, *args, fig=None, phase = True, fontsize=12,**kwargs):
        if fig is None:
            # Create new figure.
            fig = plt.figure()
        
        # Make fig current.
        plt.figure(fig.number)
        
        regions = [self.extract_AM_idx(), 
                   self.extract_A_idx(), 
                   self.extract_AL_idx(),
                   self.extract_P_idx()]
        region_labels = ['AM', 'A', 'AL','P']
        
        h = self.column_headers
                # Extract time array.
        results = self.results
        time = results['t_cycle']
        t0 = np.amin(time)
        tend = np.amax(time)
        
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
            
               
            # Mark the beginning of each phase
            phaselabel = ['d','ic','e','ir']
            if phase == True:
                for i in range(1,5):
                    if 'phase_s' in results.keys():
                        index = (results['phase_s'] == i).idxmax()
                    else: 
                        index = (results['phase'] == i).idxmax()
                    phase_time = results['t_cycle'][index]
                    plt.plot([phase_time, phase_time], [0, 80],'C7')
                    
                    #plot phase labels
                    if i != 4:
                        #get the begin time of the next phase
                        if 'phase_s' in results.keys():
                            next_index = (results['phase_s'] == i+1).idxmax()
                        else: 
                            next_index = (results['phase'] == i+1).idxmax()
                        next_phase = results['t_cycle'][next_index]
                        #plot label between the two phases
                        plt.text((phase_time+next_phase)/2, 70, phaselabel[i-1],fontsize=13,horizontalalignment='center')
                    elif i == 4:
                        #plot the label of the last phase
                        plt.text((phase_time+max(time))/2, 70, phaselabel[i-1],fontsize=13,horizontalalignment='center')
      

        
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
            # mean_stress = np.mean(stress, axis=0)[1:]    
            mean_stress = np.average(stress, axis=0)[1:]  
            
            plt.plot(mean_strain, mean_stress, *args, **kwargs)
            
               
            # # Mark the beginning of each phase   
            # if phase == True:
            #     for i in range(2,5):
            #         if 'phase_s' in results.keys():
            #             index = (results['phase_s'] == i).idxmax()
            #         else: 
            #             index = (results['phase'] == i).idxmax()
            #         phase_time = results['t_cycle'][index]
            #         plt.plot([phase_time, phase_time], [np.amin(self.all_data[idx, h['ls_old'], :]), np.amax(self.all_data[idx, h['ls_old'], :])],'--k')


            plt.xlabel('Natural myofiber strain $\epsilon_f$ [-]', fontsize=fontsize)
            plt.ylabel('Active myofiber Cauchy stress $\sigma_f$ [kPa]', fontsize=fontsize)
            plt.tick_params(labelsize=fontsize-2)
            plt.grid('on')
            plt.axis(axlimits)
            plt.title(region_labels[ii], fontsize=fontsize+2)    
            
    def plot_time_strain_ls(self, *args, fig=None, fontsize=12, 
                         reference='onset_shortening', phase = True, reorient_data=True,var='strain',
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
        
        # ls = self.all_data[all_regions, h['ls_old'], :]

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
            # if reorient_data:
            #     mean_strain = self.reorient_data(results, mean_strain, time_axis=-1)
                
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
                
            # Mark the beginning of each phase
            phaselabel = ['d','ic','e','ir']
            if phase == True:
                for i in range(1,5):
                    #get begin time of the phase
                    if 'phase_s' in results.keys():
                        index = (results['phase_s'] == i).idxmax()
                    else: 
                        index = (results['phase'] == i).idxmax()
                    
                    phase_time = results['t_cycle'][index]
                    plt.plot([phase_time, phase_time], [1.8, 2.6],'C7')
                
                    #plot phase labels
                    if i != 4:
                        #get the begin time of the next phase
                        if 'phase_s' in results.keys():
                            next_index = (results['phase_s'] == i+1).idxmax()
                        else: 
                            next_index = (results['phase'] == i+1).idxmax()
                        next_phase = results['t_cycle'][next_index]
                        #plot label between the two phases
                        plt.text((phase_time+next_phase)/2, 2.5, phaselabel[i-1],fontsize=13,horizontalalignment='center')
                    elif i == 4:
                        #plot the label of the last phase
                        plt.text((phase_time+max(time))/2, 2.5, phaselabel[i-1],fontsize=13,horizontalalignment='center')
                    
                    
            plt.axis([min(time),max(time),1.8, 2.6])    
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

    def show_regions_new(self, fig = None, projection='2d', fontsize=12, skip=None):
        h = self.column_headers
        # Plot the regions.
        if fig is None:
            # Create new figure.
            fig = plt.figure()
        
        # Make fig current.
        plt.figure(fig.number)
        
        if projection == '3d':
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
                        
        col = [ 'C7','C5', 'C0', 'C1','C2','C3']
        # col = ['g', 'r', 'c','w']
        if projection == '2d':
            # Only select nodes from slice.
            # region0 = np.where((self.all_data[:, h[':1'], 0]) >= 0.)
            region0 = np.where((self.all_data[:, h[':1'], 0]) >= -4.)
        else:
            # Select all nodes.
            # region0 = self.all_data
            region0 = np.where((self.all_data[:, h[':1'], 0]) >= -4.)
            # region0 = np.arange(len(self.all_data[:, h[':1'], 0]))
        
        if self.parameters['infarct'] == True:
            regions = [region0,
                       self.extract_T0_idx(),
                       self.extract_AM_idx(), 
                       self.extract_A_idx(), 
                       self.extract_AL_idx(),
                       self.extract_P_idx()]
            # regions = [region0,
            #            np.array([2469, 4259,3805]),]
            
            region_labels = [None,
                             'T0',
                            'AM', 
                             'A', 
                             'AL',
                             'P']
        else:
            regions = [region0,
                       self.extract_AM_idx(), 
                       self.extract_A_idx(), 
                       self.extract_AL_idx()]
            # regions = [region0,
            #            np.array([2469, 4259,3805]),]
            
            region_labels = [None,
                            'AM', 
                             'A', 
                             'AL']            
            
        # idx_tot =[]
        # idx_tot.extend(regions[1:5])

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
                # if ii != 0:
                ax.scatter3D(x, z, y, color=col[ii], label=region_labels[ii],zdir='y')
                
                plt.legend(frameon=False, fontsize=fontsize)
                plt.title('Nodes included in local function analysis', fontsize=fontsize+2)

            else:
                # f = plt.figure() 
                # f, axes = plt.subplots(nrows = 1, ncols = 2, sharex=True, sharey = True)
                # axes.scatter(x, z)
                ax1.scatter(x, z, color=col[ii], label=region_labels[ii])
                ax1.axis('equal')
                ax1.set_title('side view (x-z)')
                ax1.legend(frameon=False, fontsize=fontsize)
                plt.legend(frameon=False, fontsize=fontsize)
                ax2.scatter(y, z, color=col[ii], label=region_labels[ii])
                ax2.set_title('front view (y-z)')
                ax2.axis('equal')
                ax2.legend(frameon=False, fontsize=fontsize)
                fig.suptitle('Nodes included in local function analysis', fontsize=16)
                # plt.subplot(1, 2, 2)
                # ax.scatter(y, z, color=col[ii], label=region_labels[ii])
        # if projection == '3d':
        #     ax.view_init(1, 1)
            # ax.set_aspect('equal')        # plt.axis('off')
 
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

