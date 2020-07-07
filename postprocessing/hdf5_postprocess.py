import sys
sys.path.append('/mnt/c/Users/Maaike/Documents/Master/Graduation_project/git_graduation_project/cvbtk')
sys.path.append('/mnt/c/Users/Maaike/Documents/Master/Graduation_project/git_graduation_project/postprocessing')

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

plt.close('all')

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
        
        par['nr_segments'] = 8
        par['theta_vals'] = [1.125, 4/10*math.pi, 5/10*math.pi, 6/10*math.pi, 7/10*math.pi, 8/10*math.pi]

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
    
    def plot_torsion(self, fig = None, fontsize=12, title=''):
        if fig is None:    
            fig = plt.figure()
        plt.figure(fig.number)
        
        fig.set_size_inches(20, 13)
        
#        manager = plt.get_current_fig_manager()
#        manager.window.showMaximized()

        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        
        col = [ 'C7','C5', 'C0', 'C1','C2','C3', 'C4','C8']

        results = self.results
        
        t_es_cycle = results['t_cycle'][(results['phase'] == 4).idxmax()]
        t_es = results['time'][(results['phase'] == 4).idxmax()]
        vector = t_es_cycle/2
        
        u_vector = 'u/vector_{}'.format(int(vector))

        openfile = HDF5File(mpi_comm_world(),os.path.join(self.directory,'u.hdf5'), 'r')
        t = openfile.attributes(u_vector)['timestamp']

        if t_es == t:
            self.u_vals = Function(self.vector_V)
            openfile.read(self.u_vals, u_vector)
        else:
            raise ValueError('specified time and timestamp hdf5 do not match')

        theta_vals = self.parameters['theta_vals']
        nr_segments = self.parameters['nr_segments']

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
        
        parameters['allow_extrapolation'] = True
        self.u_vals.set_allow_extrapolation(True)

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

                x_epi, y_epi, z_epi = (epi + u_val for epi, u_val in zip((x_epi, y_epi, z_epi), self.u_vals(x_epi, y_epi, z_epi)))
                x_endo, y_endo, z_endo = (epi + u_val for epi, u_val in zip((x_endo, y_endo, z_endo), self.u_vals(x_endo, y_endo, z_endo)))
                
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
                        wall_point = point['point_'+ wall]
                        
                        ang_point = np.dot(base_seg, wall_point) / (np.linalg.norm(base_seg) * np.linalg.norm(wall_point))
                        ang_point = np.arccos(ang_point)

#                        vector1 = base_seg/np.linalg.norm(base_seg)
#                        vector2 = point['point_'+ wall]/np.linalg.norm(point['point_'+ wall])                     
#                        
#                        dot_product = np.dot(vector1,vector2)
#                        ang_point = radians_to_degrees(np.arccos(dot_product))

                        r = math.sqrt(wall_point[0]**2 + wall_point[1]**2)

                        height = 1
                        shear = math.atan(2*r*math.sin(ang_point/2)/height)

                        slice_shear['slice_'+ wall].append(ang_point)
                        slice_shear['shear_'+ wall].append(shear)
                        
            if slice_nr != 0:
                label_epi = '{}, average: {:.2f}'.format(slice_nr, np.mean(slice_shear['slice_epi']))
                label_endo = '{}, average: {:.2f}'.format(slice_nr, np.mean(slice_shear['slice_endo']))
                label_shear_epi = '{}, average: {:.2f}'.format(slice_nr, np.mean(slice_shear['shear_epi']))
                label_shear_endo = '{}, average: {:.2f}'.format(slice_nr, np.mean(slice_shear['shear_endo']))
                ax1.plot(nrsegments, slice_shear['slice_epi'], color = col[slice_nr], label = label_epi)
                ax2.plot(nrsegments, slice_shear['slice_endo'], color = col[slice_nr],label = label_endo)
                ax3.plot(nrsegments, slice_shear['shear_epi'], color = col[slice_nr],label = label_shear_epi)
                ax4.plot(nrsegments, slice_shear['shear_endo'], color = col[slice_nr],label = label_shear_endo)
        
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
        
        plt.show()
        plt.savefig(os.path.join(self.directory, 'torsion_and_shear.png'), dpi=300, bbox_inches="tight")
        
        
    def show_slices(self, fig = None, projection='2d', fontsize=12, title = ''):
        # Plot the regions.
        if fig is None:
            # Create new figure.
            fig = plt.figure()
            
        fig.set_size_inches(20, 13)
            
#        manager = plt.get_current_fig_manager()
#        manager.window.showMaximized()

        # Make fig current.
        plt.figure(fig.number)
        col = [ 'C7','C5', 'C0', 'C1','C2','C3', 'C4','C8']

        gs = GridSpec(2,2)
        _xy1 = plt.subplot(gs[0,0])
        _xz1 = plt.subplot(gs[0,1])
        _xy2 = plt.subplot(gs[1,:])
#        _xy3 = plt.subplot(gs[1,1])   
        
        self._ax = {'xy1': _xy1, 'xz1': _xz1, 'xy2': _xy2}

        self.lv_drawing(side_keys = ['xy1'], top_keys = ['xz1'])

        theta_vals = self.parameters['theta_vals']
        nr_segments = self.parameters['nr_segments']
        focus = self.parameters['focus']
        eps_outer = self.parameters['eps_outer']
        eps_inner = self.parameters['eps_inner']        
                  
        phi_int = 2*math.pi / nr_segments
        phi_range = np.arange(-1*math.pi, 1*math.pi, phi_int)
        
        for slice_nr, theta in enumerate(theta_vals): 
            x_segs = []
            y_segs = []
            z_segs = []
            
            x_segs_u = []
            y_segs_u = []
            z_segs_u = []
            
            x, y, z_epi = ellipsoidal_to_cartesian(focus,eps_outer,theta,0.0)
            tau = z_epi/(focus * math.cosh(eps_inner))
            theta_inner = math.acos(tau)
            
            thetas = [theta, theta_inner]
            for seg, phi in enumerate(phi_range):
                
                for ii, wall in enumerate([eps_outer, eps_inner]):
                    x, y, z = ellipsoidal_to_cartesian(focus,wall,thetas[ii],phi)
                    x_segs.append(x)
                    y_segs.append(y)
                    z_segs.append(z)
                    
                    x, y, z = (pos + u_val for pos, u_val in zip((x, y, z), self.u_vals(x, y, z)))
                    x_segs_u.append(x)
                    y_segs_u.append(y)
                    z_segs_u.append(z)
                    
            self._ax['xy1'].scatter(x_segs, y_segs, color=col[slice_nr], label= 'slice ' + str(slice_nr))           
            self._ax['xz1'].scatter(x_segs, z_segs, color=col[slice_nr], label= 'slice ' + str(slice_nr))
            
            if slice_nr == 0 or slice_nr == 4:
                self._ax['xy2'].scatter(x_segs_u, y_segs_u, color=col[slice_nr], label= 'slice ' + str(slice_nr))           
#            self._ax['xy3'].scatter(x_segs, y_segs, color=col[slice_nr], label= 'slice' + str(slice_nr))
   
        
        self._ax['xy1'].axis('equal')
        self._ax['xy1'].set_title('top view (x-y)')
        self._ax['xy1'].legend(frameon=False, fontsize=fontsize)
        
        self._ax['xz1'].set_title('front view (x-z)')
        self._ax['xz1'].axis('equal')
        self._ax['xz1'].legend(frameon=False, fontsize=fontsize) 
        
        self._ax['xy2'].axis('equal')
        self._ax['xy2'].set_title('top view (x-y)')
        self._ax['xy2'].legend(frameon=False, fontsize=fontsize)
        
        fig.suptitle(title,fontsize=fontsize+2)
        
        plt.savefig(os.path.join(self.directory, 'slices.png'), dpi=300, bbox_inches="tight")
#        self._ax['xy3'].set_title('front view (x-z)')
#        self._ax['xy3'].axis('equal')
#        self._ax['xy3'].legend(frameon=False, fontsize=fontsize)

    def lv_drawing(self, side_keys = '', top_keys = ''):
        def ellips(a, b, t):
            x = a*np.cos(t)
            y = b*np.sin(t)
            return x, y
    
        def cutoff(x, y, h):
            x = x[y<=h]
            y = y[y<=h]
            return x, y
        
        R_1, y, z = ellipsoidal_to_cartesian(self.parameters['focus'],self.parameters['eps_inner'],1/2*math.pi, 0.)
        R_2, y, z = ellipsoidal_to_cartesian(self.parameters['focus'],self.parameters['eps_outer'],1/2*math.pi, 0.)
        
        x, y, Z1 = ellipsoidal_to_cartesian(self.parameters['focus'],self.parameters['eps_inner'],math.pi, 0.)
        x, y, Z2 = ellipsoidal_to_cartesian(self.parameters['focus'],self.parameters['eps_outer'],math.pi, 0.)
        
        Z_1 = abs(Z1)
        Z_2 = abs(Z2)
        
        h = 2.40  
        n = 500
        lw = 1
        
        
        # LV free wall
        t_lvfw = np.linspace(-np.pi, np.pi, n)
        x_endo, y_endo = ellips(R_1, R_1, t_lvfw)
        x_epi, y_epi = ellips(R_2, R_2, t_lvfw)
        
        for key in side_keys:
            self._ax[key].plot(x_endo, y_endo, color='C3', linewidth=lw)
            self._ax[key].plot(x_epi, y_epi, color='C3', linewidth=lw)
            self._ax[key].axis('equal')  
        
        # LV free wall
        t_lvfw = np.linspace(-np.pi, np.pi, n)
        x_endo, y_endo = ellips(R_1, Z_1, t_lvfw)
        x_epi, y_epi = ellips(R_2, Z_2, t_lvfw)
        x_mid, y_mid = ellips((R_1+R_2)/2, (Z_1+Z_2)/2, t_lvfw)
        
        # Cut off
        x_endo, y_endo = cutoff(x_endo, y_endo, h)
        x_epi, y_epi = cutoff(x_epi, y_epi, h)
        x_mid, y_mid = cutoff(x_mid, y_mid, h)
        
        for key in top_keys:
            self._ax[key].plot(x_endo, y_endo, color='C3', linewidth=lw)
            self._ax[key].plot(x_epi, y_epi, color='C3', linewidth=lw)
            self._ax[key].plot(x_mid, y_mid, '--', color='C3', linewidth=lw/2)
            self._ax[key].axis('equal')
                    

# mesh = Mesh()
# openfile = HDF5File(mpi_comm_world(),'/mnt/c/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/leftventricular model/23-06_13-08_eikonal_td_1node/cycle_2_begin_ic_ref/mesh.hdf5', 'r')
# openfile.read(mesh, 'mesh', False)
# V = FunctionSpace(mesh, "Lagrange", 2)

# active_stress = Function(V)

# openfile = HDF5File(mpi_comm_world(), '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/Results_Tim/leftventricular model/23-06_13-08_eikonal_td_1node/cycle_2_begin_ic_ref/active_stress.hdf5', 'r')
# openfile.read(active_stress, 'active_stress/vector_0')


# print(active_stress(0.329953, -2.33535, -0.723438))
# print(active_stress(0., 0., 0.))
#
#directory_1 = '/home/maaike/Documents/Graduation_project/Results/eikonal_td_1_node/cycle_2_begin_ic_ref'
#
#post_1 = postprocess_hdf5(directory_1)
#
#post_1.plot_torsion()
#post_1.show_slices()