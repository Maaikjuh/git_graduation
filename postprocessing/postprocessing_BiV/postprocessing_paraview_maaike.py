# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:03:54 2020

@author: Maaike
"""


import sys
sys.path.append("..") # Adds higher directory to python modules path.

from postprocessing import Dataset, shift_data, get_paths

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
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

def length(v):
    return math.sqrt(v[0]**2+v[1]**2)
def dot_product(v,w):
   return v[0]*w[0]+v[1]*w[1]
def determinant(v,w):
   return v[0]*w[1]-v[1]*w[0]
def inner_angle(v,w):
   cosx=dot_product(v,w)/(length(v)*length(w))
   rad=math.acos(cosx) # in radians
   return rad*180/math.pi # returns degrees

def radians_to_degrees(angle):
    """
    Converst radians to degrees.
    """
    return angle/math.pi*180

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
        par['inner_sigma']= 0.3713
        par['outer_sigma']= 0.6784
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
    
    def compute_sigma(self, x, y, z):
        """
        Ellipsoidal radial position defined such that constant values represent
        an ellipsoidal surface.
    
        Args:
            x, y, z: x, y, z coordinates.
            focus: focal length of ellipsoid.
        Returns:
            sigma
        """
        
        focus = self.parameters['focus']
        # Compute and return the sigma values.
        return 0.5 * (np.sqrt(x ** 2 + y ** 2 + (z + focus) ** 2)
                        + np.sqrt(x ** 2 + y ** 2 + (z - focus) ** 2)) / focus  
    
    
    def extract_wall_idx(self, phi_val= 0.,theta_outer = None, theta_inner= None, loc = ['epi','mid','endo']):
        #select single point on epi, mid and endowall respectively
        h = self.column_headers
        data = self.all_data[:, :, 0]
        
        if theta_outer == None:
            theta_outer = self.parameters['theta']
        if theta_inner == None:
            theta_inner = self.parameters['theta']
            
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
        x, y, z = ellips_to_cartesian(self.parameters['focus'],self.parameters['outer_sigma'],theta_outer, phi_val)
        x2, y2, z2 = ellips_to_cartesian(self.parameters['focus'],self.parameters['inner_sigma'],theta_inner, phi_val)

        mask = {}
        mask_tot = {}
        coordname = ['x','y','z']        
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
                if self.parameters['infarct'] == True and (phi_val == 'AM_phi' or phi_val == 'AL_phi'):
                    mask_tot_loc = mask_tot_loc_1 * mask_tot_loc_2 * mask_T0
                else:
                    mask_tot_loc = mask_tot_loc_1 * mask_tot_loc_2
                    
                #increase boundary width
                slice_thickness = slice_thickness + 0.01
                
            mask_tot['mask_tot_' + wall] = mask_tot_loc  
            
        if len(mask_tot.keys()) == 1:
            return np.array(np.where(mask_tot['mask_tot_' + loc[0]])[0][0])
        
        #only return one point per wall location (epi, mid or endo)
        return (np.array([np.where(mask_tot['mask_tot_epi'])[0][0],np.where(mask_tot['mask_tot_mid'])[0][0],np.where(mask_tot['mask_tot_endo'])[0][0]]))
        
    def extract_AM_idx(self):
        return self.extract_wall_idx(phi_val = self.parameters['AM_phi'])

    def extract_A_idx(self): 
        return self.extract_wall_idx(phi_val = self.parameters['A_phi'])
    
    def extract_AL_idx(self):
        return self.extract_wall_idx(phi_val = self.parameters['AL_phi'])
    
    def extract_P_idx(self):
        self.parameters['P_phi'] = self.parameters['A_phi'] + math.pi
        return self.extract_wall_idx(phi_val = self.parameters['P_phi'])
    
    def extract_segment_idx(self, nr_segments = 24, theta=None):
        if theta == None:
            theta = self.parameters['theta']
            
        h = self.column_headers
        
        # calculate all sigma values  
        # constant value represents ellipsoidal surface
        sigma = self.compute_sigma(self.all_data[:, h[':0'], 0],self.all_data[:, h[':1'], 0],self.all_data[:, h[':2'], 0])
        
        # get sigma value of epi and endo ellipsoid
        max_sigma = max(sigma)
        min_sigma = min(sigma)
        
        # calculate height (z) of the segments
        tau = math.cos(theta)
        z_epi = self.parameters['focus'] * max_sigma * tau
        
        # calculate for which theta a (segment) point on the inner ellipsoid is 
        # at the same height as the point on the outer ellipsoid
        tau = z_epi/(self.parameters['focus'] * min_sigma)
        theta_inner = math.acos(tau)
        
        # create 24 segments
        phi_int = 2*math.pi / nr_segments
        phi_range = np.arange(0., 2*math.pi, phi_int)
        phi_range = np.arange(-1*math.pi, 1*math.pi, phi_int)
        
        segments = []
        for i in phi_range:
            # for each segment, extract a point on the epi, mid and endo wall
            seg = self.extract_wall_idx(phi_val = i, theta_outer = theta, theta_inner= theta_inner)
            segments.append(seg)    
        return segments
        
        # if 'T0' in h:
        #     data_T0 = self.all_data[segments, h['T0'], 0]
        #     return segments, data_T0
        # else:      
        #     return segments   
        
    def extract_torsion_idx(self,nr_segments = 8, theta_vals = None):
        if theta_vals == None:
            theta_vals = [1.1, 4/10*math.pi, 5/10*math.pi, 6/10*math.pi, 7/10*math.pi, 8/10*math.pi]
        
        torsion_idx = []
        for i, theta in enumerate(theta_vals):
            slice_idx = self.extract_segment_idx(nr_segments,theta = theta)
            torsion_idx.append(slice_idx)
        return torsion_idx
        
    def extract_T0_idx(self):
        h = self.column_headers
        data_T0 = self.all_data[:, h['T0'], 0]
        return np.where(data_T0 <= 90)
    
    def extract_eikonal_idx(self):
        h = self.column_headers
        if 'eikonal' in h:
            td = 'eikonal'
        elif 'f_179' in h:
            td = 'f_179'
        data_td = self.all_data[:, h[td], 0]
        return np.where(data_td >= -20)
         
    def calculate_wall_thickness(self):
        h = self.column_headers
        results = self.results  
        # if 'T0' in h:
        #     segments = self.extract_segment_idx()[0]
        # else:
        segments = self.extract_segment_idx()
        
        distance = []
        
        for i, seg in enumerate(segments):
            epi = seg[0]
            endo = seg[2]
            
            x_pos = self.all_data[[epi,endo], h[':0'], :] + self.all_data[[epi,endo], h['displacement:0'], :]
            y_pos = self.all_data[[epi,endo], h[':1'], :] + self.all_data[[epi,endo], h['displacement:1'], :]
                      
            dx = x_pos[0] - x_pos[1]
            dy = y_pos[0] - y_pos[1]
            
            dist = np.sqrt(dx**2 + dy**2)
            distance.append(dist)
        
        return distance
               
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
    
    def plot_torsion(self, fig = None, fontsize=12,nr_segments = 8, theta_vals = None,title=''):
        self.show_slices_segment_idx()
        
        if fig is None:    
            fig = plt.figure()
        plt.figure(fig.number)
        
        col = ['C0', 'C1','C2','C3', 'C4','C8']
        
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2,2,3)
        ax4 = fig.add_subplot(2,2,4)
        
        h = self.column_headers
        results = self.results
  
        index_es = (results['phase'] ==4).idxmax() - results.index[0] - 1
        
        slices_idx = self.extract_torsion_idx(nr_segments) 
        nrsegments = range(1, nr_segments+1)
               
        base_idx = slices_idx[0]
       
        for ii in range(1, len(slices_idx)): 
            tor_slice = slices_idx[ii]
            
            slice_epi= []
            slice_endo = []
            shear_epi= []
            shear_endo = []
            for seg in nrsegments:    
                seg = seg -1
                seg_slice = tor_slice[seg]
                
                seg_base = base_idx[seg]
                # epi_base = seg_base[0]
                # endo_base = seg_base[2]
                
                for i in [0, 2]:
                    seg_point = seg_slice[i]
                    base_point = seg_base[i]
                    
                    x = self.all_data[[base_point, seg_point], h[':0'], 0]
                    x = x + self.all_data[[base_point, seg_point], h['displacement:0'], index_es]
                
                    y = self.all_data[[base_point, seg_point], h[':1'], 0]
                    y = y + self.all_data[[base_point, seg_point], h['displacement:1'], index_es]
                    
                    base = ([x[0],y[0]])
                    point = ([x[1],y[1]])
                    
                    vector1 = base/np.linalg.norm(base)
                    vector2 = point/np.linalg.norm(point)
                    
                    dot_product = np.dot(vector1,vector2)
                    ang_point = radians_to_degrees(np.arccos(dot_product))
                    
                    # ang_point = np.dot(base, point) / (np.linalg.norm(base) * np.linalg.norm(point))
                    # ang_point = np.arccos(ang_point)
                    # # ang_point = math.degrees(math.atan2(point[1], point[0]) - math.atan2(base[1], base[0]))
                    # ang_point = inner_angle(base,point)
                    
                    r = math.sqrt(point[0]**2 + point[1]**2)
                    height = 1 #self.all_data[base_point, h[':2'], 0] - self.all_data[seg_point, h[':2'], 0] 
                    shear = math.atan(2*r*math.sin(ang_point/2)/height)
                    if i == 0:
                        slice_epi.append(ang_point)
                        shear_epi.append(shear)
                    elif i == 2:
                        slice_endo.append(ang_point)
                        shear_endo.append(shear)
                    
            label_epi = '{}, average: {:.2f}'.format(ii, np.mean(slice_epi))
            label_endo = '{}, average: {:.2f}'.format(ii, np.mean(slice_endo))
            label_shear_epi = '{}, average: {:.2f}'.format(ii, np.mean(shear_epi))
            label_shear_endo = '{}, average: {:.2f}'.format(ii, np.mean(shear_epi))
            ax1.plot(nrsegments, slice_epi, color = col[ii-1], label = label_epi)
            ax2.plot(nrsegments, slice_endo, color = col[ii-1], label = label_endo)
            ax3.plot(nrsegments, shear_epi, color = col[ii-1], label = label_shear_epi)
            ax4.plot(nrsegments, shear_endo, color = col[ii-1], label = label_shear_endo)
        
        fig.suptitle(title,fontsize=fontsize+2)
        ax1.legend(frameon=False, fontsize=fontsize)
        ax2.legend(frameon=False, fontsize=fontsize)
        ax3.legend(frameon=False, fontsize=fontsize)
        ax4.legend(frameon=False, fontsize=fontsize)
        ax1.set_ylabel('Torsion epicardial [$^\circ$]', fontsize = fontsize)
        ax2.set_ylabel('Torsion endocardial [$^\circ$]', fontsize = fontsize)
        ax3.set_ylabel('Shear epicardial [$^\circ$]', fontsize = fontsize)
        ax4.set_ylabel('Shear endocardial [$^\circ$]', fontsize = fontsize)
        ax3.set_xlabel('Segments', fontsize = fontsize)
        ax4.set_xlabel('Segments', fontsize = fontsize)
                    
    def plot_rotation(self, fig = None, fontsize=12, title = ''):
        h = self.column_headers
        results = self.results
        
        # if 'T0' in h:
        #     segments, T0 = self.extract_segment_idx()
        # else:
        segments = self.extract_segment_idx()
        
        index_ed = (results['phase'] ==2).idxmax() - results.index[0] - 1
        index_es = (results['phase'] ==4).idxmax() - results.index[0] - 1
        
        angles = []
        strains = []
        nrsegments = []
        for ii, idx in enumerate(segments):
            idx = idx[1]
            idx_segment = np.append(idx,segments[ii-1][1])
            if ii != len(segments)-1:
                idx_segment = np.append(idx_segment,segments[ii+1][1])
            else:
                idx_segment = np.append(idx_segment,segments[0][1])

            # idx_epi = idx[0]
            x_pos = self.all_data[idx_segment, h[':0'], 0]
            y_pos = self.all_data[idx_segment, h[':1'], 0]
            
            x_ed  = x_pos + self.all_data[idx_segment, h['displacement:0'], index_ed]
            x_es  = x_pos + self.all_data[idx_segment, h['displacement:0'], index_es]
            
            y_ed  = y_pos + self.all_data[idx_segment, h['displacement:1'], index_ed]
            y_es  = y_pos + self.all_data[idx_segment, h['displacement:1'], index_es]        
            
            p_ed = np.array([x_ed, y_ed])
            p_es = np.array([x_es, y_es])
            
            ang_points_seg = []
            for i in range(0,len(p_ed)):
                ang_point = np.dot(p_ed[:,i], p_es[:,i]) / (np.linalg.norm(p_ed[:,i]) * np.linalg.norm(p_es[:,i]))
                ang_point = np.arccos(ang_point)
                # ang_point = inner_angle(p_ed[:,i],p_es[:,i])
                ang_points_seg.append(ang_point)
                
            ang_tot_seg = np.mean(ang_points_seg)
            angles.append(radians_to_degrees(ang_tot_seg))
            
            ls0 = self.all_data[idx_segment, h['ls_old'], index_ed]
            ls = self.all_data[idx_segment, h['ls_old'], index_es]
            strain = np.log( ls/ls0 )
            strains.append(np.mean(strain))
            nrsegments.append(ii+1)
            
        if fig is None:    
            fig = plt.figure()
        plt.figure(fig.number)
        
        fig.suptitle(title, fontsize = fontsize+2)
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(nrsegments, angles)
        ax1.set_ylabel('Rotation [$^\circ$]', fontsize = fontsize)
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(nrsegments, strains)
        ax2.set_ylabel('strain', fontsize = fontsize)
        ax2.set_xlabel('Segments', fontsize = fontsize)
        
        # if 'T0' in h:
        #     ax1a = ax1.twinx()
        #     ax1a.set_ylabel('T0')
        #     ax1a.plot(nrsegments, T0)
        
        ax1.axis(ymin=0.,ymax=7.)
        ax2.axis(ymin=-0.2,ymax=0.2)
   
    def plot_wall_thickness(self, fig = None, fontsize=12, nrsegments=None):
        if fig is None:
            # Create new figure.
            fig = plt.figure()
        
        # Make fig current.
        plt.figure(fig.number)
        
        distance = self.calculate_wall_thickness()    
        results = self.results
        time = results['t_cycle']
        t0 = np.amin(time)
        tend = np.amax(time)
        
        if nrsegments == None:
            nrsegments = range(0, len(segments))
        for seg, dist in enumerate(distance):
            if (seg+1) in nrsegments:
                dist = dist - dist[0]
                
                dist2 = distance[seg-1]
                dist2 = dist2 - dist2[0]
                
                dist = (dist + dist2)/2
                plt.plot(time, dist, label = str(seg+1))
        
        # Mark the beginning of each phase
        phaselabel = ['d','ic','e','ir']
        for i in range(1,5):
            if 'phase_s' in results.keys():
                index = (results['phase_s'] == i).idxmax()
            else: 
                index = (results['phase'] == i).idxmax()
            phase_time = results['t_cycle'][index]
            plt.plot([phase_time, phase_time], [-0.4, 0.2],'C7')
        
            #plot phase labels
            if i != 4:
                #get the begin time of the next phase
                if 'phase_s' in results.keys():
                    next_index = (results['phase_s'] == i+1).idxmax()
                else: 
                    next_index = (results['phase'] == i+1).idxmax()
                next_phase = results['t_cycle'][next_index]
                #plot label between the two phases
                plt.text((phase_time+next_phase)/2, 0.15, phaselabel[i-1],fontsize=13,horizontalalignment='center')
            elif i == 4:
                #plot the label of the last phase
                plt.text((phase_time+max(time))/2, 0.15, phaselabel[i-1],fontsize=13,horizontalalignment='center')
    
        
        plt.xlabel('time [ms]', fontsize=fontsize)
        plt.ylabel('Change in wall thickness [cm]', fontsize=fontsize)
        plt.tick_params(labelsize=fontsize-2)
        plt.grid('on')
        plt.axis([t0, tend, -0.4, 0.2])
        plt.legend()
        plt.title('Wall thickness', fontsize=fontsize+2)
    
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
        # regions = [self.extract_P_idx()]
        regions = [self.extract_P_idx(),
                    self.extract_AM_idx(), 
                    self.extract_A_idx(), 
                    self.extract_AL_idx()]
        region_labels = ['P','AM', 'A', 'AL']
        # region_labels = ['']
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
                           reference='onset_shortening', **kwargs):
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

   
    def show_T0_eikonal_idx(self, fig = None,fontsize=12):
        h = self.column_headers
        # Plot the regions.
        if fig is None:
            # Create new figure.
            fig = plt.figure()
        
        # Make fig current.
        plt.figure(fig.number)
                           
        col = [ 'C7','C5', 'C0', 'C1','C2','C3']
        region0 = np.where((self.all_data[:, h[':1'], 0]) >= -4.)
        
        gs = GridSpec(1,2)
        _xy2 = plt.subplot(gs[0,0])
        _xz2 = plt.subplot(gs[0,1])
  
        self._ax = {'xy2': _xy2, 'xz2': _xz2}
        
        self.lv_drawing(self._ax)
        
        regions = [region0,
                   self.extract_AM_idx(), 
                   self.extract_A_idx(), 
                   self.extract_AL_idx(),
                   self.extract_P_idx()]  
        
        region_labels = [None,
                         'AM',
                         'A',
                         'AL',
                         'P']
        if 'T0' in h:
            regions.append(self.extract_T0_idx())
            region_labels.append('T0')
        
        if 'eikonal' in h or 'f_179' in h:
            regions.append(self.extract_eikonal_idx())
            region_labels.append('eikonal')
        
        for ii, idx in enumerate(regions):
            x = self.all_data[idx, h[':0'], 0]
            y = self.all_data[idx, h[':1'], 0]
            z = self.all_data[idx, h[':2'], 0]
                    
            self._ax['xy2'].scatter(x, y, color=col[ii], label=region_labels[ii])
            self._ax['xy2'].axis('equal')
            self._ax['xy2'].set_title('top view (x-y)')
            self._ax['xy2'].legend(frameon=False, fontsize=fontsize)
            
            self._ax['xz2'].scatter(x, z, color=col[ii], label=region_labels[ii])
            self._ax['xz2'].set_title('front view (x-z)')
            self._ax['xz2'].axis('equal')
            self._ax['xz2'].legend(frameon=False, fontsize=fontsize)  
            
    def show_single_slice_segment_idx(self, fig = None, projection='2d', fontsize=12):
        h = self.column_headers
        # Plot the regions.
        if fig is None:
            # Create new figure.
            fig = plt.figure()
        
        # Make fig current.
        plt.figure(fig.number)
                           
        col = [ 'C7','C5', 'C0', 'C1','C2','C3']
        
        region0 = np.where((self.all_data[:, h[':1'], 0]) >= -4.)
        regions = [region0,
                   self.extract_segment_idx()]
        
        gs = GridSpec(1,2)
        _xy1 = plt.subplot(gs[0,0])
        _xz1 = plt.subplot(gs[0,1])
        
        self._ax = {'xy1': _xy1, 'xz1': _xz1}
        self.lv_drawing('xz1')
        
        for ii, idx in enumerate(regions):
            x = self.all_data[idx, h[':0'], 0]
            y = self.all_data[idx, h[':1'], 0]
            z = self.all_data[idx, h[':2'], 0]
                        
            if ii == 1:
                for i in range(0, len(regions[ii])):
                    self._ax['xy1'].scatter(x[i], y[i])
                    self._ax['xy1'].plot(x[i], y[i])
                    self._ax['xy1'].axis('equal')
                    nr_x = np.mean(np.append(x[i],x[i-1]))
                    nr_y = np.mean(np.append(y[i],y[i-1]))
                    self._ax['xy1'].text(nr_x,nr_y,'{}'.format(i+1), ha='center', va='center')
                    
                    self._ax['xz1'].scatter(x, z, color=col[ii])
                    self._ax['xz1'].set_title('front view (x-z)')
                    self._ax['xz1'].axis('equal')
                    self._ax['xz1'].legend(frameon=False, fontsize=fontsize)
            else:
                self._ax['xz1'].scatter(x, z, color=col[ii])
                self._ax['xz1'].axis('equal')
                self._ax['xz1'].set_title('top view (x-y)')
                self._ax['xz1'].legend(frameon=False, fontsize=fontsize)
                
    def show_slices_segment_idx(self, fig = None, projection='2d', fontsize=12):
        h = self.column_headers
        results = self.results
        # Plot the regions.
        if fig is None:
            # Create new figure.
            fig = plt.figure()
        index_es = (results['phase'] ==4).idxmax() - results.index[0]
        # Make fig current.
        plt.figure(fig.number)
        col = [ 'C7','C5', 'C0', 'C1','C2','C3', 'C4','C8']
        
        region0 = np.where((self.all_data[:, h[':1'], 0]) >= -4.)

        gs = GridSpec(2,2)
        _xy1 = plt.subplot(gs[0,0])
        _xz1 = plt.subplot(gs[0,1])
        _xy2 = plt.subplot(gs[1,0])
        _xy3 = plt.subplot(gs[1,1])   
        
        self._ax = {'xy1': _xy1, 'xz1': _xz1, 'xy2': _xy2, 'xy3': _xy3}
        
        lv_keys = {'xy1', 'xz1', 'xy3'}
        self.lv_drawing(lv_keys)
        
        regions = [region0]
        region_labels = [None]
        
        slices = self.extract_torsion_idx()
        for i, slice_idx in enumerate(slices):
            regions.append(slice_idx)
            region_labels.append('slice {}'.format(str(i)))
        
        infarct = False
        if 'T0' in h:
            infarct = True
            regions.append(self.extract_T0_idx())
            region_labels.append('T0')
            
        for ii, idx in enumerate(regions):
            x = self.all_data[idx, h[':0'], 0]
            y = self.all_data[idx, h[':1'], 0]
            z = self.all_data[idx, h[':2'], 0]
            
            xdis = x + self.all_data[idx, h['displacement:0'], index_es]
            ydis = y + self.all_data[idx, h['displacement:1'], index_es]
            
            self._ax['xy1'].scatter(x, y, color=col[ii], label=region_labels[ii])
            self._ax['xy1'].axis('equal')
            self._ax['xy1'].set_title('top view (x-y)')
            self._ax['xy1'].legend(frameon=False, fontsize=fontsize)
            
            self._ax['xz1'].scatter(x, z, color=col[ii], label=region_labels[ii])
            self._ax['xz1'].set_title('front view (x-z)')
            self._ax['xz1'].axis('equal')
            self._ax['xz1'].legend(frameon=False, fontsize=fontsize)
            
            ii_inf = len(regions)
            if infarct == True:
                ii_inf = len(regions) -1
            if ii != 0 and ii != ii_inf:       
                # self._ax['xz2'].scatter(x, z, color=col[ii], label=region_labels[ii])
                # if ii == 1:
                for i in range(0, len(slice_idx)): 
                    # self._ax['xy2'].scatter(x[i], y[i], color=col[ii])
                    # self._ax['xy2'].plot(x[i], y[i], color=col[ii])
                    
                    if ii == 1 or ii == 5:
                        self._ax['xy2'].scatter(x[i], y[i], color=col[ii])
                        self._ax['xy2'].plot(x[i], y[i], color=col[ii])
              
                        self._ax['xy3'].scatter(xdis[i], ydis[i], color=col[ii])
                        self._ax['xy3'].plot(xdis[i], ydis[i], color=col[ii])
                    if ii == 1:
                        nr_x = np.mean(np.append(x[i],x[i-1]))
                        nr_y = np.mean(np.append(y[i],y[i-1]))
                        self._ax['xy2'].text(nr_x,nr_y,'{}'.format(i+1), ha='center', va='center')
                        self._ax['xy3'].text(nr_x,nr_y,'{}'.format(i+1), ha='center', va='center')
            
              
    def lv_drawing(self, ax_keys):
        def ellips(a, b, t):
            x = a*np.cos(t)
            y = b*np.sin(t)
            return x, y
    
        def cutoff(x, y, h):
            x = x[y<=h]
            y = y[y<=h]
            return x, y
        
        R_1, y, z = ellips_to_cartesian(self.parameters['focus'],self.parameters['inner_sigma'],1/2*math.pi, 0.)
        R_2, y, z = ellips_to_cartesian(self.parameters['focus'],self.parameters['outer_sigma'],1/2*math.pi, 0.)
        
        x, y, Z1 = ellips_to_cartesian(self.parameters['focus'],self.parameters['inner_sigma'],math.pi, 0.)
        x, y, Z2 = ellips_to_cartesian(self.parameters['focus'],self.parameters['outer_sigma'],math.pi, 0.)
        
        Z_1 = abs(Z1)
        Z_2 = abs(Z2)
        
        h = 2.10  
        n = 500
        lw = 1
        
        
        # LV free wall
        t_lvfw = np.linspace(-np.pi, np.pi, n)
        x_endo, y_endo = ellips(R_1, R_1, t_lvfw)
        x_epi, y_epi = ellips(R_2, R_2, t_lvfw)
        
        if 'xy1' in ax_keys:
            self._ax['xy1'].plot(x_endo, y_endo, color='C3', linewidth=lw)
            self._ax['xy1'].plot(x_epi, y_epi, color='C3', linewidth=lw)
            self._ax['xy1'].axis('equal')
        
        if 'xy2' in ax_keys:
            self._ax['xy2'].plot(x_endo, y_endo, color='C3', linewidth=lw)
            self._ax['xy2'].plot(x_epi, y_epi, color='C3', linewidth=lw)
            self._ax['xy2'].axis('equal')
        
        if 'xy3' in ax_keys:
            self._ax['xy3'].plot(x_endo, y_endo, color='C3', linewidth=lw)
            self._ax['xy3'].plot(x_epi, y_epi, color='C3', linewidth=lw)
            self._ax['xy3'].axis('equal')
        
        # LV free wall
        t_lvfw = np.linspace(-np.pi, np.pi, n)
        x_endo, y_endo = ellips(R_1, Z_1, t_lvfw)
        x_epi, y_epi = ellips(R_2, Z_2, t_lvfw)
        x_mid, y_mid = ellips((R_1+R_2)/2, (Z_1+Z_2)/2, t_lvfw)
        
        # Cut off
        x_endo, y_endo = cutoff(x_endo, y_endo, h)
        x_epi, y_epi = cutoff(x_epi, y_epi, h)
        x_mid, y_mid = cutoff(x_mid, y_mid, h)
        
        if 'xz2' in ax_keys:
            self._ax['xz2'].plot(x_endo, y_endo, color='C3', linewidth=lw)
            self._ax['xz2'].plot(x_epi, y_epi, color='C3', linewidth=lw)
            self._ax['xz2'].plot(x_mid, y_mid, '--', color='C3', linewidth=lw/2)
            self._ax['xz2'].axis('equal')
        
        if 'xz1' in ax_keys:
            self._ax['xz1'].plot(x_endo, y_endo, color='C3', linewidth=lw)
            self._ax['xz1'].plot(x_epi, y_epi, color='C3', linewidth=lw)
            self._ax['xz1'].plot(x_mid, y_mid, '--', color='C3', linewidth=lw/2)
            self._ax['xz1'].axis('equal')


def common_start(sa, sb):
    """ returns the longest common substring from the beginning of sa and sb """
    def _iter():
        for a, b in zip(sa.split('\\'), sb.split('\\')):
            if a == b:
                yield a+'\\'
            else:
                return

    return ''.join(_iter())

