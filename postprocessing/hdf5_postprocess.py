import sys
sys.path.append('/home/maaike/Documents/Graduation_project/git_graduation/cvbtk')
sys.path.append('/home/maaike/Documents/Graduation_project/git_graduation/postprocessing')

from dataset import Dataset # shift_data, get_paths
from dolfin import *
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

def radians_to_degrees(angle):
    """
    Converst radians to degrees.
    """
    return angle/math.pi*180

def ellipsoidal_to_cartesian(focus,eps,theta,phi):
    x= focus * math.sinh(eps) * math.sin(theta) * math.cos(phi)
    y = focus * math.sinh(eps) * math.sin(theta) * math.sin(phi)
    z = focus * math.cosh(eps) * math.cos(theta)
    return x, y, z

def cartesian_to_ellipsoidal(focus, x = [0,0,0], eccentricity = None):
    if eccentricity != None:
        x[0] = focus*np.sqrt(1-eccentricity**2)/eccentricity
        x[1] = 0
        x[2] = 0

    ra = sqrt(x[0]**2+x[1]**2+(x[2]+focus)**2)
    rb = sqrt(x[0]**2+x[1]**2+(x[2]-focus)**2)

    tau= (1./(2.*focus)*(ra-rb))
    sigma=(1./(2.*focus)*(ra+rb))

    expressions_dict = {"phi": math.atan2(x[1],x[0]),
                        "eps": math.acosh(sigma),
                        "theta": math.acos(tau)} 
    return expressions_dict


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

class postprocess_hdf5(object):
    def __init__(self, directory, results_csv=None, cycle=None, **kwargs):
        self.directory = directory

        if results_csv is None:
            # Directory with results file is 1 folder down.
            results_dir = os.path.split(directory)[0]  
            results_csv = os.path.join(results_dir, 'results.csv')
            if cycle is None:
                # Assume it is in the cycle directory.
                cycle = float(os.path.split(directory)[1].split('_')[1])        

        self.mesh = self.load_mesh()
        self.vector_V = VectorFunctionSpace(self.mesh, "Lagrange", 2)
        self.function_V = FunctionSpace(self.mesh, "Lagrange", 2)

        self.cycle = cycle
        self.results = self.load_reduced_dataset(results_csv, cycle=cycle)

        self._parameters = self.default_parameters()
        self._parameters.update(kwargs)

        if 'inner_eccentricity' or 'outer_eccentricity' in kwargs:
            e_outer = self.parameters['outer_eccentricity']
            e_inner = self.parameters['inner_eccentricity']
            eps_outer = cartesian_to_ellipsoidal(self.parameters['focus'], eccentricity = e_outer)['eps']
            eps_inner = cartesian_to_ellipsoidal(self.parameters['focus'], eccentricity = e_inner)['eps']
            self.parameters['eps_outer'] = eps_outer
            self.parameters['eps_inner'] = eps_inner

    @property
    def parameters(self):
        return self._parameters

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

    @staticmethod
    def default_parameters():
        par = {}
        par['cut_off_low'] = -3.
        par['cut_off_high'] = -2.5
        par['dt'] = 2.

        par['theta'] = 7/10*math.pi
        par['AM_phi'] = 1/5*math.pi
        par['A_phi'] = 1/2*math.pi
        par['AL_phi'] = 4/5*math.pi
        par['P_phi'] = par['A_phi'] + math.pi

        par['inner_eccentricity']= 0.934819
        par['outer_eccentricity']= 0.807075
        par['eps_inner'] = 0.3713
        par['eps_outer'] = 0.6784

        par['focus']= 4.3
        par['ls0'] = 1.9
        par['name'] = ''
        return par

    def load_mesh(self):
        mesh = Mesh()
        openfile = HDF5File(mpi_comm_world(),os.path.join(self.directory,'mesh.hdf5'), 'r')
        openfile.read(mesh, 'mesh', False)

        return mesh
    
    def plot_torsion(self, fig = None, fontsize=12, nr_segments = 8, theta_vals = None, title=''):
        if fig is None:    
            fig = plt.figure()
        plt.figure(fig.number)

        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        results = self.results
#        t_start_cycle = (results['cycle'] == self.cycle).idxmax()
#        t_es = (results['phase'] == 4).idxmax()
#
#        vector = (t_es - t_start_cycle) / self.parameters['dt'] 
        
        t_es_cycle = results['t_cycle'][(results['phase'] == 4).idxmax()]
        t_es = results['time'][(results['phase'] == 4).idxmax()]
        vector = t_es_cycle/2
        
        u_vector = 'u/vector_{}'.format(int(vector))

        openfile = HDF5File(mpi_comm_world(),os.path.join(self.directory,'u.hdf5'), 'r')
        t = openfile.attributes(u_vector)['timestamp']

        if t_es == t:
            u_vals = Function(self.vector_V)
#            openfile = HDF5File(mpi_comm_world(),os.path.join(self.directory,'u.hdf5'), 'r')
            openfile.read(u_vals, u_vector)
        else:
            raise ValueError('specified time and timestamp hdf5 do not match')

        if theta_vals == None:
            theta_vals = [1.125, 4/10*math.pi, 5/10*math.pi, 6/10*math.pi, 7/10*math.pi, 8/10*math.pi]

        phi_int = 2*math.pi / nr_segments
        phi_range = np.arange(-1*math.pi, 1*math.pi, phi_int)

        focus = self.parameters['focus']
        eps_outer = self.parameters['eps_outer']
        eps_inner = self.parameters['eps_inner']

        base_epi = []
        base_endo = []

        base = {'base_epi': [],
                'base_endo': []}

        nrsegments = range(1, nr_segments+1)
        
#        parameters['allow_extrapolation'] = True

        for slice_nr, theta in enumerate(theta_vals): 
            slice_shear = {'slice_epi': [],
                        'slice_endo': [],
                        'shear_epi': [],
                        'shear_endo': []}          

            for seg, phi in enumerate(phi_range):
                x_epi, y_epi, z_epi = ellipsoidal_to_cartesian(focus,eps_outer,theta,phi)
                tau = z_epi/(focus * math.cosh(eps_inner))
                theta_inner = math.acos(tau)
                x_endo, y_endo, z_endo = ellipsoidal_to_cartesian(focus,eps_inner,theta_inner,phi)
                
                parameters['allow_extrapolation'] = True
                x,y,z = u_vals(x_epi, y_epi, z_epi)
                x_epi += x
                y_epi += y
                
                x,y,z = u_vals(x_endo, y_endo, z_endo)
                x_endo += x
                y_endo += y
#                x_epi, y_epi, z_epi = (epi + u_val for epi, u_val in zip((x_epi, y_epi, z_epi), u(x_epi, y_epi, z_epi)))
#                x_endo, y_endo, z_endo = (epi + u_val for epi, u_val in zip((x_endo, y_endo, z_endo), u(x_endo, y_endo, z_endo)))
                
                # x,y,z = u(x_epi, y_epi, z_epi)
                # [x_epi, y_epi, z_epi] += u(x_epi, y_epi, z_epi)
                # x_endo, y_endo, z_endo += u(x_endo, y_endo, z_endo)

                if slice_nr == 0:
                    base['base_epi'].append([x_epi, y_epi])
                    base['base_endo'].append([x_endo, y_endo])

                else:
                    point = {'point_epi': ([x_epi, y_epi]),
                            'point_endo': ([x_endo, y_endo])}

                    for wall in ['epi', 'endo']:
                        base_seg = base['base_' + wall][seg]

                        vector1 = base_seg/np.linalg.norm(base_seg)
                        vector2 = point['point_'+ wall]/np.linalg.norm(point['point_'+ wall])                     
                        
                        dot_product = np.dot(vector1,vector2)
                        ang_point = radians_to_degrees(np.arccos(dot_product))

                        r = math.sqrt(point['point_'+ wall][0]**2 + point['point_'+ wall][1]**2)

                        height = 1
                        shear = math.atan(2*r*math.sin(ang_point/2)/height)

                        slice_shear['slice_'+ wall].append(ang_point)
                        slice_shear['shear_'+ wall].append(shear)
                        
            if slice_nr != 0:
                label_epi = '{}, average: {:.2f}'.format(slice_nr, np.mean(slice_shear['slice_epi']))
                label_endo = '{}, average: {:.2f}'.format(slice_nr, np.mean(slice_shear['slice_endo']))
                label_shear_epi = '{}, average: {:.2f}'.format(slice_nr, np.mean(slice_shear['shear_epi']))
                label_shear_endo = '{}, average: {:.2f}'.format(slice_nr, np.mean(slice_shear['shear_endo']))
                ax1.plot(nrsegments, slice_shear['slice_epi'], label = label_epi)
                ax2.plot(nrsegments, slice_shear['slice_endo'], label = label_endo)
                ax3.plot(nrsegments, slice_shear['shear_epi'], label = label_shear_epi)
                ax4.plot(nrsegments, slice_shear['shear_endo'], label = label_shear_endo)
        
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
                    

# mesh = Mesh()
# openfile = HDF5File(mpi_comm_world(),'/mnt/c/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/leftventricular model/23-06_13-08_eikonal_td_1node/cycle_2_begin_ic_ref/mesh.hdf5', 'r')
# openfile.read(mesh, 'mesh', False)
# V = FunctionSpace(mesh, "Lagrange", 2)

# active_stress = Function(V)

# openfile = HDF5File(mpi_comm_world(), '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/leftventricular model/23-06_13-08_eikonal_td_1node/cycle_2_begin_ic_ref/active_stress.hdf5', 'r')
# openfile.read(active_stress, 'active_stress/vector_0')


# print(active_stress(0.329953, -2.33535, -0.723438))
# print(active_stress(0., 0., 0.))

directory_1 = '/home/maaike/Documents/Graduation_project/Results/eikonal_td_1_node/cycle_2_begin_ic_ref'

post_1 = postprocess_hdf5(directory_1)

post_1.plot_torsion()