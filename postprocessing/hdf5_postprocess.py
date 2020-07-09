import sys
#ubuntu
#sys.path.append('/mnt/c/Users/Maaike/Documents/Master/Graduation_project/git_graduation_project/cvbtk')
#sys.path.append('/mnt/c/Users/Maaike/Documents/Master/Graduation_project/git_graduation_project/postprocessing')

#linux
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
        """
        calculate and plot torsion and shear
        Torsion is difference of the rotation of a base segment and a slice segment
        The rotation of these points is the angle traveled between end diastole and end systole
        Shear is calculated from the torsion, height between the base and the slice
        and the radius of the wall at the slice
        """
        if fig is None:    
            fig = plt.figure()
        plt.figure(fig.number)
        
        fig.set_size_inches(20, 13)
        
#        manager = plt.get_current_fig_manager()
#        manager.window.showMaximized()

        _t_epi  = fig.add_subplot(2, 2, 1)
        _t_endo = fig.add_subplot(2, 2, 2)
        _s_epi  = fig.add_subplot(2, 2, 3)
        _s_endo = fig.add_subplot(2, 2, 4)
        
        ax = {'torsion_epi':_t_epi, 'torsion_endo':_t_endo, 'shear_epi':_s_epi, 'shear_endo':_s_endo}
        
        col = [ 'C7','C5', 'C0', 'C1','C2','C3', 'C4','C8']

        results = self.results
        
        # find the time of end systole
        t_es_cycle = results['t_cycle'][(results['phase'] == 4).idxmax()]
        t_es = results['time'][(results['phase'] == 4).idxmax()]
        
        # find the corresponding vector name for u, assuming a dt of 2. ms
        vector = t_es_cycle/2       
        u_es_vector = 'u/vector_{}'.format(int(vector))
        
        # find the time of end diastole
        t_ed_cycle = results['t_cycle'][(results['phase'] == 2).idxmax()]
        t_ed = results['time'][(results['phase'] == 2).idxmax()]
        
        # find the corresponding vector name for u, assuming a dt of 2. ms
        vector = t_ed_cycle/2       
        u_ed_vector = 'u/vector_{}'.format(int(vector))        

        # check if the vector u corresponds with the time at end systole
        openfile = HDF5File(mpi_comm_world(),os.path.join(self.directory,'u.hdf5'), 'r')
        u_t_es = openfile.attributes(u_es_vector)['timestamp']
        u_t_ed = openfile.attributes(u_ed_vector)['timestamp']

        if t_es == u_t_es and t_ed == u_t_ed:
            # if the right vector is found, extract the displacement values at t_es
            self.u_es = Function(self.vector_V)
            self.u_ed = Function(self.vector_V)
            openfile.read(self.u_es, u_es_vector)
            openfile.read(self.u_ed, u_ed_vector)
        else:
            raise ValueError('specified time and timestamp hdf5 do not match, check dt')

        # extract the theta values of the slices
        theta_vals = self.parameters['theta_vals']
        
        # extract the number of segments and convert to a phi range
        nr_segments = self.parameters['nr_segments']

        phi_int = 2*math.pi / nr_segments
        phi_range = np.arange(-1*math.pi, 1*math.pi, phi_int)

        focus = self.parameters['focus']
        eps_outer = self.parameters['eps_outer']
        eps_inner = self.parameters['eps_inner']

        base = {'base_epi': [],
                'base_endo': []}
        
        wall = ['epi', 'endo']

        nrsegments = range(1, nr_segments+1)
        
#        parameters['allow_extrapolation'] = True
        self.u_es.set_allow_extrapolation(True)
        self.u_ed.set_allow_extrapolation(True)

        for slice_nr, theta in enumerate(theta_vals): 
            slice_shear = {'torsion_epi': [],
                        'torsion_endo': [],
                        'shear_epi': [],
                        'shear_endo': []}          

            for seg, phi in enumerate(phi_range):
                x_epi, y_epi, z_epi = ellipsoidal_to_cartesian(focus,eps_outer,theta,phi)
                tau = z_epi/(focus * math.cosh(eps_inner))
                theta_inner = math.acos(tau)
                x_endo, y_endo, z_endo = ellipsoidal_to_cartesian(focus,eps_inner,theta_inner,phi)
                
#                parameters['allow_extrapolation'] = True
                for ii, points in enumerate([[x_epi, y_epi, z_epi], [x_endo, y_endo, z_endo]]):
                    x_ed, y_ed, z_ed = (epi + u_val for epi, u_val in zip((x_epi, y_epi, z_epi), self.u_ed(points)))
                    x_es, y_es, z_es = (epi + u_val for epi, u_val in zip((x_epi, y_epi, z_epi), self.u_es(points)))
                    
                    point_ed = [x_ed, y_ed]
                    point_es = [x_es, y_es]
                                      
                    length1 =  math.sqrt(np.dot(point_ed,point_ed))
                    length2 =  math.sqrt(np.dot(point_es,point_es))
                    cos_angle = np.dot(point_ed, point_es) / (length1 * length2)
                    rad_angle = np.arccos(cos_angle)
                    rot = radians_to_degrees(rad_angle)
                    
                    if slice_nr == 0:
                        base['base_' + wall[ii]].append([rot, z_ed])
                        
                    else:
                        base_seg = base['base_' + wall[ii]][seg][0]
                        torsion = rot - base_seg
                        
                        r = math.sqrt(point_ed[0]**2 + point_ed[1]**2)
                        height = base['base_' + wall[ii]][seg][1] - z_ed
                        
                        shear = math.atan(2*r*math.sin(torsion/2)/height)
                        slice_shear['torsion_'+ wall[ii]].append(torsion)
                        slice_shear['shear_'+ wall[ii]].append(shear)
                        
#                x_es_epi, y_es_epi, z_es_epi = (epi + u_val for epi, u_val in zip((x_epi, y_epi, z_epi), self.u_es(x_epi, y_epi, z_epi)))
#                x_es_endo, y_es_endo, z_es_endo = (epi + u_val for epi, u_val in zip((x_endo, y_endo, z_endo), self.u_es(x_endo, y_endo, z_endo)))
#                
#                x_ed_epi, y_ed_epi, z_ed_epi = (epi + u_val for epi, u_val in zip((x_epi, y_epi, z_epi), self.u_ed(x_epi, y_epi, z_epi)))
#                x_ed_endo, y_ed_endo, z_ed_endo = (epi + u_val for epi, u_val in zip((x_endo, y_endo, z_endo), self.u_ed(x_endo, y_endo, z_endo)))
#                
#                point_es = [x_es_epi, y_es_epi]
#                point_ed = [x_ed_epi, y_ed_epi]
#                
#                length1 =  math.sqrt(np.dot(point_ed,point_ed))
#                length2 =  math.sqrt(np.dot(point_es,point_es))
#                cos_angle = np.dot(point_ed, point_es) / (length1 * length2)
#                rad_angle = np.arccos(cos_angle)
#                rot = radians_to_degrees(rad_angle)
                
                # x,y,z = u(x_epi, y_epi, z_epi)
                # [x_epi, y_epi, z_epi] += u(x_epi, y_epi, z_epi)
                # x_endo, y_endo, z_endo += u(x_endo, y_endo, z_endo)

#                if slice_nr == 0:
##                    base['base_epi'].append([x_epi, y_epi, z_epi])
##                    base['base_endo'].append([x_endo, y_endo, z_endo])
#                    base['base'].append([rot, z_ed_epi])
#                else:
##                    point = {'point_epi': ([x_epi, y_epi, z_epi]),
##                            'point_endo': ([x_endo, y_endo, z_endo])}
#                    
#
#                    for wall in ['epi', 'endo']:
#                        base_seg = base['base_' + wall][seg][0:2]
#                        wall_point = point['point_'+ wall][0:2]
#                        
#                        length1 =  math.sqrt(np.dot(base_seg,base_seg))
#                        length2 =  math.sqrt(np.dot(wall_point,wall_point))
#                        cos_angle = np.dot(base_seg, wall_point) / (length1 * length2)
#                        rad_angle = np.arccos(cos_angle)
#                        torsion = radians_to_degrees(rad_angle)
#
#                        r = math.sqrt(wall_point[0]**2 + wall_point[1]**2)
#                        height = base['base_' + wall][seg][2] - point['point_'+ wall][2]
#                        
#                        shear = math.atan(2*r*math.sin(torsion/2)/height)
#
#                        slice_shear['torsion_'+ wall].append(torsion)
#                        slice_shear['shear_'+ wall].append(shear)
                        
            if slice_nr != 0:
                for key in slice_shear.keys():
                    label = '{}, average: {:.2f}'.format(slice_nr, np.mean(slice_shear[key]))
                    ax[key].plot(nrsegments, slice_shear[key], color = col[slice_nr], label = label)
                    ax[key].legend(frameon=False, fontsize=fontsize)
        
        fig.suptitle(title,fontsize=fontsize+2)

        ax['torsion_epi'].set_ylabel('Torsion epicardial [$^\circ$]', fontsize = fontsize)
        ax['torsion_endo'].set_ylabel('Torsion endocardial [$^\circ$]', fontsize = fontsize)
        ax['shear_epi'].set_ylabel('Shear epicardial [$^\circ$]', fontsize = fontsize)
        ax['shear_endo'].set_ylabel('Shear endocardial [$^\circ$]', fontsize = fontsize)
        ax['shear_epi'].set_xlabel('Segments', fontsize = fontsize)
        ax['shear_endo'].set_xlabel('Segments', fontsize = fontsize)
        
#        plt.show()
        plt.savefig(os.path.join(self.directory, 'torsion_and_shear.png'), dpi=300, bbox_inches="tight")
        
    def loc_mech_ker(self, fig = None, fontsize = 12):
        
        stress_file = HDF5File(mpi_comm_world(), os.path.join(self.directory,'fiber_stress.hdf5'), 'r')
        ls_file = HDF5File(mpi_comm_world(), os.path.join(self.directory,'ls.hdf5'), 'r')
        nsteps = ls_file.attributes('ls_old')['count']
        vector_numbers = list(range(nsteps))
        
        stress_func = Function(self.function_V)
        ls_func = Function(self.function_V)
      
        phi_free_wall = 1/2*math.pi
        theta_vals = [1/2*math.pi, 0.63*math.pi, 19/25*math.pi]

        focus = self.parameters['focus']
        eps_outer = self.parameters['eps_outer']
        eps_inner = self.parameters['eps_inner']
        eps_mid = (eps_outer + eps_inner)/2
        eps_vals = [eps_inner,eps_mid,eps_outer]

        stress_cycle = {'top_endo': [],
                        'top_mid': [],
                        'top_epi': [],
                        'mid_endo': [],
                        'mid_mid': [],
                        'mid_epi': [],
                        'bot_endo': [],
                        'bot_mid': [],
                        'bot_epi': []}
        strain_cycle = {'top_endo': [],
                        'top_mid': [],
                        'top_epi': [],
                        'mid_endo': [],
                        'mid_mid': [],
                        'mid_epi': [],
                        'bot_endo': [],
                        'bot_mid': [],
                        'bot_epi': []}
        
        rad = ['endo', 'mid', 'epi']
        long = ['top', 'mid', 'bot']
        for vector in vector_numbers:
            print('{0:.2f}% '.format(vector/nsteps*100))
            stress_vector = 'fiber_stress/vector_{}'.format(vector)
            ls_vector = 'ls_old/vector_{}'.format(vector)
            
            stress_file.read(stress_func, stress_vector)
            ls_file.read(ls_func, ls_vector)
            
            stress_func.set_allow_extrapolation(True)
            ls_func.set_allow_extrapolation(True)
            
            for l, theta in enumerate(theta_vals):
                for r, eps in enumerate(eps_vals):
                    x, y, z = ellipsoidal_to_cartesian(focus,eps,theta,phi_free_wall)
                    
                    stress = stress_func(x, y, z)
                    ls = ls_func(x, y, z)
                    strain = np.log(ls/self.parameters['ls0'])
                    
                    stress_cycle[long[l] + '_' + rad[r]].append(stress)
                    strain_cycle[long[l] + '_' + rad[r]].append(strain)
        
        time = self.results['t_cycle']
        

        fig = plt.figure()
        plt.figure(fig.number)

        ii = 0
        for l in long:
            for r in rad:
                ii += 1
                strain = fig.add_subplot(3, 3, ii)
                strain.plot(time[0:len(vector_numbers)], strain_cycle[l + '_' + r])
                if ii == 1:
                    strain.set_title('endocardium', fontsize = fontsize)
                if ii == 3:
                    strain.set_title('epicardium', fontsize = fontsize)
#                    strain.yaxis.set_label_position('right')
                    strain.set_ylabel('top')
#                if ii == 9:
#                    strain.set_label_position("right")
#                    strain.set_ylabel('bottom')                    

        fig.text(0.5, 0.04, 'time [ms]', ha='center', va='center', fontsize = fontsize +1)
        fig.text(0.06, 0.5, 'strain', ha='center', va='center', rotation='vertical', fontsize = fontsize)
        fig.suptitle('Myofiber strain', fontsize = fontsize +2)

        fig = plt.figure()
        plt.figure(fig.number)                

        ii = 0
        for l in long:
            for r in rad:
                ii += 1
                stress = fig.add_subplot(3, 3, ii)
                stress.plot(time[0:len(vector_numbers)], stress_cycle[l + '_' + r])        
                if ii == 1:
                    stress.set_title('endocardium', fontsize = fontsize)
                if ii == 3:
                    stress.set_title('epicardium', fontsize = fontsize)
#                    strain.yaxis.set_label_position('right')
#                    stress.set_ylabel('top')
#                if ii == 9:
#                    strain.set_label_position("right")
#                    stress.set_ylabel('bottom')                    
                    
        fig.text(0.5, 0.04, 'time [ms]', ha='center', va='center', fontsize = fontsize +1)
        fig.text(0.06, 0.5, 'stress [kPa]', ha='center', va='center', rotation='vertical', fontsize = fontsize)
        fig.suptitle('Myofiber stress', fontsize = fontsize +2)
        
        fig = plt.figure()
        plt.figure(fig.number)                

        ii = 0
        for l in long:
            for r in rad:
                ii += 1
                work = fig.add_subplot(3, 3, ii)
                work.plot(strain_cycle[l + '_' + r], stress_cycle[l + '_' + r])  
                if ii == 1:
                    work.set_title('endocardium', fontsize = fontsize)
                if ii == 3:
                    work.set_title('epicardium', fontsize = fontsize)
#                    strain.yaxis.set_label_position('right')
                    work.set_ylabel('top')
#                if ii == 9:
#                    strain.set_label_position("right")
#                    work.set_ylabel('bottom')                    
            
        fig.text(0.5, 0.04, 'strain', ha='center', va='center', fontsize = fontsize +1)
        fig.text(0.06, 0.5, 'stress [kPa]', ha='center', va='center', rotation='vertical', fontsize = fontsize)
        fig.suptitle('Workloops', fontsize = fontsize +2)
        
    def show_slices(self, fig = None, fontsize=12, title = ''):
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
        _xy2 = plt.subplot(gs[1,0])
        _xy3 = plt.subplot(gs[1,1])
#        _xy3 = plt.subplot(gs[1,1])   
        
        self._ax = {'xy1': _xy1, 'xz1': _xz1, 'xy2': _xy2, 'xy3': _xy3}

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
            
            x_segs_es = []
            y_segs_es = []
            z_segs_es = []

            x_segs_ed = []
            y_segs_ed = []
            z_segs_ed = []
            
            x, y, z_epi = ellipsoidal_to_cartesian(focus,eps_outer,theta,0.0)
            tau = z_epi/(focus * math.cosh(eps_inner))
            theta_inner = math.acos(tau)
            
            thetas = [theta, theta_inner]
            for seg, phi in enumerate(phi_range):
                
                seg_es = []
                seg_ed = []
                
                for ii, wall in enumerate([eps_outer, eps_inner]):
                    x, y, z = ellipsoidal_to_cartesian(focus,wall,thetas[ii],phi)
                    x_segs.append(x)
                    y_segs.append(y)
                    z_segs.append(z)
                    
                    x, y, z = (pos + u_val for pos, u_val in zip((x, y, z), self.u_es(x, y, z)))
                    x_segs_es.append(x)
                    y_segs_es.append(y)
                    z_segs_es.append(z)
                    
                    seg_es.append([x,y])
                    
                    x, y, z = (pos + u_val for pos, u_val in zip((x, y, z), self.u_ed(x, y, z)))
                    x_segs_ed.append(x)
                    y_segs_ed.append(y)
                    z_segs_ed.append(z)
                    
                    seg_ed.append([x,y])
                
                if slice_nr == 0 or slice_nr == 4:
                    self._ax['xy2'].plot([i[0] for i in seg_ed], [i[1] for i in seg_ed], color=col[slice_nr])
                    self._ax['xy2'].plot([i[0] for i in seg_es], [i[1] for i in seg_es], '--', color=col[slice_nr]) 
                    self._ax['xy3'].plot([i[0] for i in seg_es], [i[1] for i in seg_es], color=col[slice_nr]) 
                    
                                        
            self._ax['xy1'].scatter(x_segs, y_segs, color=col[slice_nr], label= 'slice ' + str(slice_nr))           
            self._ax['xz1'].scatter(x_segs, z_segs, color=col[slice_nr], label= 'slice ' + str(slice_nr))
            
            if slice_nr == 0 or slice_nr == 4:
                self._ax['xy3'].scatter(x_segs_es, y_segs_es, color=col[slice_nr], label= 'es ' + str(slice_nr))    
                self._ax['xy2'].scatter(x_segs_es, y_segs_es, marker='x', color=col[slice_nr], label= 'es ' + str(slice_nr)) 
                self._ax['xy2'].scatter(x_segs_ed, y_segs_ed, color=col[slice_nr], label= 'ed ' + str(slice_nr)) 
#            self._ax['xy3'].scatter(x_segs, y_segs, color=col[slice_nr], label= 'slice' + str(slice_nr))
   
        
        self._ax['xy1'].axis('equal')
        self._ax['xy1'].set_title('top view (x-y)')
        self._ax['xy1'].legend(frameon=False, fontsize=fontsize)
        
        self._ax['xz1'].set_title('front view (x-z)')
        self._ax['xz1'].axis('equal')
        self._ax['xz1'].legend(frameon=False, fontsize=fontsize) 
        
        self._ax['xy2'].axis('equal')
        self._ax['xy2'].axis([-3.5, 3.5, -3.5, 3.5])
        self._ax['xy2'].set_title('top view (x-y) end diastole')
        self._ax['xy2'].legend(frameon=False, fontsize=fontsize)
        
        self._ax['xy3'].axis('equal')
        self._ax['xy3'].axis([-3.5, 3.5, -3.5, 3.5])
        self._ax['xy3'].set_title('top view (x-y) end sytole')
        self._ax['xy3'].legend(frameon=False, fontsize=fontsize)
        
        fig.suptitle(title,fontsize=fontsize+2)
        
        plt.savefig(os.path.join(self.directory, 'slices.png'), dpi=300, bbox_inches="tight")
#        self._ax['xy3'].set_title('front view (x-z)')
#        self._ax['xy3'].axis('equal')
#        self._ax['xy3'].legend(frameon=False, fontsize=fontsize)
        
    def show_ker_points(self):
        fig = plt.figure()
        plt.figure(fig.number)
        
        self._ax = {'fig': plt}
        self.lv_drawing(side_keys = ['fig'])
        
        phi_free_wall = 1/2*math.pi
        theta_vals = [1/2*math.pi, 0.63*math.pi, 19/25*math.pi]

        focus = self.parameters['focus']
        eps_outer = self.parameters['eps_outer']
        eps_inner = self.parameters['eps_inner']
        eps_mid = (eps_outer + eps_inner)/2
        eps_vals = [eps_inner,eps_mid,eps_outer]
        
        for theta in theta_vals:
            for eps in eps_vals:
                x, y, z = ellipsoidal_to_cartesian(focus,eps,theta,phi_free_wall)
                plt.scatter(y, z)    

    def lv_drawing(self, side_keys = None, top_keys = None):
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
        
        if top_keys != None:
            for key in top_keys:
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
        
        if side_keys != None:
            for key in side_keys:
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
directory_1 = '/home/maaike/Documents/Graduation_project/Results/eikonal_td_1_node/cycle_2_begin_ic_ref'

post_1 = postprocess_hdf5(directory_1)
post_1.loc_mech_ker()
post_1.show_ker_points()
post_1.plot_torsion()
post_1.show_slices()