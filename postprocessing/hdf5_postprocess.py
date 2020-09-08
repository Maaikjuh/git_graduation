import sys
#ubuntu
sys.path.append('/mnt/c/Users/Maaike/Documents/Master/Graduation_project/git_graduation_project/cvbtk')
sys.path.append('/mnt/c/Users/Maaike/Documents/Master/Graduation_project/git_graduation_project/postprocessing')

# #   linux
# sys.path.append('/home/maaike/Documents/Graduation_project/git_graduation/cvbtk')
# sys.path.append('/home/maaike/Documents/Graduation_project/git_graduation/postprocessing')


from dataset import Dataset # shift_data, get_paths
from utils import read_dict_from_csv
from dolfin import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
import matplotlib
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
    def __init__(self, directory, results_csv=None, inputs_csv = None, cycle=None, model = 'cvbtk', **kwargs):
        self.directory = directory
        self.model = model
        self._parameters = self.default_parameters()
        self.mesh = self.load_mesh()
        
        if model == 'cvbtk':
            if results_csv is None:
                # Directory with results file is 1 folder down.
                results_dir = os.path.split(directory)[0]
                results_csv = os.path.join(results_dir, 'results.csv')
                if cycle is None:
                    # Assume it is in the cycle directory.
                    cycle = float(os.path.split(directory)[1].split('_')[1])

            if inputs_csv is None:
                # Directory with results file is 1 folder down.
                inputs_dir = os.path.split(directory)[0]
                inputs_csv = os.path.join(inputs_dir, 'inputs.csv')

            self.inputs = read_dict_from_csv(inputs_csv)

            try:
                if self.inputs['active_stress']['eikonal'] is not None:
                    self.eikonal_dir = self.inputs['active_stress']['eikonal']['td_dir']
            except KeyError:
                self.eikonal_dir = None 

            try:
                if self.inputs['infarct']['infarct'] == True:
                    self.infarct = True
            except KeyError:
                self.infarct = False

            self._parameters['ls0'] = self.inputs['active_stress']['ls0']
            self.vector_V = VectorFunctionSpace(self.mesh, "Lagrange", 2)
            self.function_V = FunctionSpace(self.mesh, "Lagrange", 2)

        if model == 'beatit':
            if results_csv is None:
                results_csv = os.path.join(self.directory, 'pV.csv')

            self.vector_V = VectorFunctionSpace(self.mesh, "CG", 2)
            self.function_V = FunctionSpace(self.mesh, "CG", 2)
            self._parameters['focus'] = 43
            self._parameters['cut_off'] = 24

            self._parameters['phase'] = ' Phase [-]'
            self._parameters['time'] = 'Time [ms]'
            self._parameters['t_cycle'] = 'Time [ms]'
            self._parameters['cycle'] = ' Cycle [-]'
 
        self.results = self.load_reduced_dataset(results_csv, cycle=cycle)

        self._parameters.update(kwargs)

        self.hdf5_files = {}
        self.hdf5_vectors ={}
        self.functions = {}

        # calculate eps_mid:
        # get x_epi and x_endo at the equatorial point (theta = 1/2*pi)
        # calculate x_mid -> calculate eps_mid using the focus
        focus = self.parameters['focus']
        self.x_epi, self.y_epi, self.z_epi = ellipsoidal_to_cartesian(focus,self._parameters['eps_outer'],1/2*math.pi,0.)
        self.x_endo, self.y_endo, self.z_endo = ellipsoidal_to_cartesian(focus,self._parameters['eps_inner'],1/2*math.pi,0.)
        x_mid = (self.x_epi + self.x_endo)/2
        eps_mid = cartesian_to_ellipsoidal(self._parameters['focus'], x = [x_mid, 0.,0.])['eps']
        self._parameters['eps_mid'] = eps_mid

    @property
    def parameters(self):
        return self._parameters

    # @staticmethod
    def load_reduced_dataset(self, filename, cycle=None):
        """
        Load the given CSV file and reduce it to the final cycle (default)
        or the specified cycle..
        """
        full = Dataset(filename=filename)

        if self.model == 'cvbtk':
            if cycle is None:
                # Choose final cycle.
                cycle = int(max(full['cycle']) - 1)

            reduced = full[full['cycle'] == cycle].copy(deep=True)

        elif self.model == 'beatit':
            if cycle is None:
                # Choose final cycle.
                cycle = int(max(full[' Cycle [-]'])-1)

            reduced = full[full[' Cycle [-]'] == cycle].copy(deep=True)

        self.cycle = cycle
        return reduced

    @staticmethod
    def default_parameters():
        par = {}
        # par['cut_off_low'] = -3.
        par['cut_off'] = 2.4
        par['dt'] = 2.

        par['theta'] = 7/10*math.pi
        par['AM_phi'] = 1/5*math.pi
        par['A_phi'] = 1/2*math.pi
        par['AL_phi'] = 4/5*math.pi
        par['P_phi'] = par['A_phi'] + math.pi

        par['nr_segments'] = 8
        par['theta_vals'] = [1.1, 4/10*math.pi, 5/10*math.pi, 6/10*math.pi, 7/10*math.pi, 8/10*math.pi]

        par['inner_eccentricity']= 0.934819
        par['outer_eccentricity']= 0.807075
        par['eps_inner'] = 0.3713
        par['eps_mid'] = float()
        par['eps_outer'] = 0.6784

        par['tol'] = 0.01

        par['focus']= 4.3
        par['ls0'] = 1.9
        par['name'] = ''

        par['phase'] = 'phase'
        par['time'] = 'time' 
        par['t_cycle'] = 't_cycle'
        par['cycle'] = 'cycle'
        par['t_act'] = 't_act'

        par['load_pickle'] = True 
        return par

    def load_mesh(self):
        mesh = Mesh()
        if self.model == 'cvbtk':
            openfile = HDF5File(mpi_comm_world(),os.path.join(self.directory,'mesh.hdf5'), 'r')
            try:
                geometry = openfile.attributes('geometry')
                self._parameters['outer_eccentricity'] = geometry['outer_eccentricity']
                self._parameters['inner_eccentricity'] = geometry['inner_eccentricity']
                self._parameters['focus'] = geometry['focus_height']
                self._parameters['cut_off'] = geometry['truncation_height']

                eps_outer = cartesian_to_ellipsoidal(geometry['focus_height'], eccentricity = geometry['outer_eccentricity'])['eps']
                eps_inner = cartesian_to_ellipsoidal(geometry['focus_height'], eccentricity = geometry['inner_eccentricity'])['eps']

                self._parameters['eps_outer'] = eps_outer
                self._parameters['eps_inner'] = eps_inner

            except RuntimeError:
                print('No geometry parameters saved, continuing with default values...')

        elif self.model == 'beatit':
            openfile = HDF5File(mpi_comm_world(),os.path.join(self.directory,'utilities/mesh.hdf5'), 'r')
        
        openfile.read(mesh, 'mesh', False)

        return mesh

    def plot_torsion(self, torsion_fig = None, shear_fig = None, fontsize=12, title=''):
        """
        calculate and plot torsion and shear
        Torsion is difference of the rotation of a base segment and a slice segment
        The rotation of these points is the angle traveled between end diastole and end systole
        Shear is calculated from the torsion, height between the base and the slice
        and the radius of the wall at the slice
        """
        if torsion_fig is None:
            torsion_fig, torsion_plot = plt.subplots(1,3, sharex = True, sharey = True)
        # plt.figure(fig.number)
        if shear_fig is None:
            shear_fig, shear_plot = plt.subplots(1,3, sharex = True, sharey = True)

        col = [ 'C7','C5', 'C0', 'C1','C2','C3', 'C4','C8']

        results = self.results
        par = self.parameters

        # find the time of end systole
        begin_phase = results[par['phase']][results.index[0]]
        t_es_cycle = results[par['t_cycle']][(results[par['phase']] == begin_phase + 3).idxmax()]
        t_es = results[par['time']][(results[par['phase']] == begin_phase + 3).idxmax()]

        # find the corresponding vector name for u
        dt = results[par['time']][results.index[1]] - results[par['time']][results.index[0]]

        # find the time of end diastole
        t_ed_cycle = results[par['t_cycle']][(results[par['phase']] == begin_phase + 1).idxmax()]
        t_ed = results[par['time']][(results[par['phase']] == begin_phase + 1).idxmax()]

        if self.model == 'cvbtk':
            vector = t_es_cycle/dt
            u_es_vector = 'u/vector_{}'.format(int(vector))    

            # find the corresponding vector name for u
            vector = t_ed_cycle/dt
            u_ed_vector = 'u/vector_{}'.format(int(vector))   

            openfile = HDF5File(mpi_comm_world(),os.path.join(self.directory,'u.hdf5'), 'r')     

            # check if the vector u corresponds with the time at end systole   
            u_t_es = openfile.attributes(u_es_vector)['timestamp']
            u_t_ed = openfile.attributes(u_ed_vector)['timestamp']
        
        elif self.model == 'beatit':
            # displacement vector is equal to the time
            u_es_vector = 'displacement_{}/vector_0'.format(float(t_es_cycle))
            u_ed_vector = 'displacement_{}/vector_0'.format(float(t_ed_cycle))

            openfile = HDF5File(mpi_comm_world(),os.path.join(self.directory,'results.h5'), 'r')

            # the precise timestamp is already in the name 
            u_t_es = t_es
            u_t_ed = t_ed

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

        tol = self.parameters['tol']

        focus = self.parameters['focus']
        eps_outer = self.parameters['eps_outer'] - tol
        eps_mid = self.parameters['eps_mid']
        eps_inner = self.parameters['eps_inner'] + tol

        base = {'base_epi': [],
                'base_mid': [],
                'base_endo': []}

        wall = ['epi', 'mid','endo']

        # make a range of the number of segments for later plotting
        nrsegments = range(1, nr_segments+1)

        # allow extrapolation in order to define the displacement values
        # at non-nodal points
        self.u_es.set_allow_extrapolation(True)
        self.u_ed.set_allow_extrapolation(True)

        # define variables for the average torsion and shear per slice
        av_slice_torsion = {'torsion_av_epi': [],
                        'torsion_av_mid': [],
                        'torsion_av_endo': []}

        av_slice_shear = {'shear_av_epi': [],
                        'shear_av_mid': [],
                        'shear_av_endo': []}

        for slice_nr, theta in enumerate(theta_vals):
            slice_torsion = {'torsion_epi': [],
                        'torsion_mid': [],
                        'torsion_endo': []}
            slice_shear = {'shear_epi': [],
                        'shear_mid': [],
                        'shear_endo': [],
                        'shear_av_epi': [],
                        'shear_av_mid': [],
                        'shear_av_endo': []}

            for seg, phi in enumerate(phi_range):
                # get coordinates of the segments:
                #   - theta differs for each slice
                #   - phi differs for each segment
                #   - each segment has a epi, mid and endo point

                x_epi, y_epi, z_epi = ellipsoidal_to_cartesian(focus,eps_outer,theta,phi)

                # the theta values for each wall location is different
                # to get the same z value of each slice through the wall (epi, mid, endo)
                # calculate the theta value for the mid and endo point
                tau = z_epi/(focus * math.cosh(eps_mid))
                theta_mid = math.acos(tau)
                x_mid, y_mid, z_mid = ellipsoidal_to_cartesian(focus,eps_mid,theta_mid,phi)

                tau = z_epi/(focus * math.cosh(eps_inner))
                theta_inner = math.acos(tau)
                x_endo, y_endo, z_endo = ellipsoidal_to_cartesian(focus,eps_inner,theta_inner,phi)

                for ii, points in enumerate([[x_epi, y_epi, z_epi], [x_mid, y_mid, z_mid], [x_endo, y_endo, z_endo]]):
                    # extract the displacement at ed and es at a point
                    # add the displacement to the coordinate value of the point
                    xu, yu, zu = self.u_ed(points)
                    x_ed = points[0] + xu
                    y_ed = points[1] + yu
                    z_ed = points[2] + zu

                    xu, yu, zu = self.u_es(points)
                    x_es = points[0] + xu
                    y_es = points[1] + yu
                    z_es = points[2] + zu

                    # calculate the rotation:
                    # rotation of the point at ed to es
                    point_ed = [x_ed, y_ed]
                    point_es = [x_es, y_es]

                    length1 =  math.sqrt(np.dot(point_ed,point_ed))
                    length2 =  math.sqrt(np.dot(point_es,point_es))
                    cos_angle = np.dot(point_ed, point_es) / (length1 * length2)
                    rad_angle = np.arccos(cos_angle)
                    rot = radians_to_degrees(rad_angle)

                    # calculate the torsion and shear:
                    #   torsion is the rotation of a point in a slice with
                    #   respect to the rotation of a point in the base
                    #   with the same phi value and wall location:
                    #   torsion = rot_slice - rot_base
                    if slice_nr == 0:
                        # the first theta value is assumed to be the base
                        base['base_' + wall[ii]].append([rot, z_ed])

                    else:
                        # extract the rotation of the base at the same phi value and wall location
                        base_seg = base['base_' + wall[ii]][seg][0]

                        torsion = rot - base_seg

                        # calculate the shear
                        #   y = tan-1[(2r*sin(torsion/2)*h)]
                        #   - r: radius of the myocardial border (epi, mid or endo)
                        #   - h: distance between basal slice and the succeeding one
                        r = math.sqrt(point_ed[0]**2 + point_ed[1]**2)
                        height = base['base_' + wall[ii]][seg][1] - z_ed

                        shear = math.atan(2*r*math.sin(torsion/2)/height)

                        # add torsion and shear of the point to all the values of the slice
                        slice_torsion['torsion_'+ wall[ii]].append(torsion)
                        slice_shear['shear_'+ wall[ii]].append(shear)

            # plot the results of each slice independently
            if slice_nr != 0:
                for ii, key in enumerate(['epi', 'mid', 'endo']):
                    # calculate and save the average torsion of the slice
                    av_slice_torsion['torsion_av_' + key].append(np.mean(slice_torsion['torsion_' + key]))
                    # create a label to show the average value of the slice
                    label = '{}, av: {:.2f}'.format(slice_nr, np.mean(slice_torsion['torsion_' + key]))

                    # plot the torsion of each point:
                    # - for all the wall locations a different subplot has been created (epi, mid, endo)
                    # - the values are plotted for each segment number
                    torsion_plot[ii].plot(nrsegments, slice_torsion['torsion_' + key], color = col[slice_nr], label = label)
                    torsion_plot[ii].set_title(key, fontsize = fontsize)

                    # save and plot the shear of each point in the same way as the torsion
                    av_slice_shear['shear_av_' + key].append(np.mean(slice_shear['shear_' + key]))
                    label = '{}, av: {:.2f}'.format(slice_nr, np.mean(slice_shear['shear_' + key]))
                    shear_plot[ii].plot(nrsegments, slice_shear['shear_' + key], color = col[slice_nr], label = label)
                    shear_plot[ii].set_title(key, fontsize = fontsize)

                    # plot the legend under each corresponding subplot
                    torsion_plot[ii].legend(frameon=False, fontsize=fontsize, bbox_to_anchor=(-0.2, -0.45, -0.2, -0.45), loc='lower left')
                    shear_plot[ii].legend(frameon=False, fontsize=fontsize, bbox_to_anchor=(-0.2, -0.45, -0.2, -0.45), loc='lower left')

        torsion_fig.suptitle(title,fontsize=fontsize+2)
        shear_fig.suptitle(title,fontsize=fontsize+2)

        torsion_fig.text(0.5, 0.03, 'Segments', ha = 'center')
        torsion_fig.text(0.04, 0.5, 'Torsion [$^\circ$]', va='center', rotation='vertical')

        shear_fig.text(0.5, 0.03, 'Segments', ha = 'center')
        shear_fig.text(0.04, 0.5, 'Shear [$^\circ$]', va='center', rotation='vertical')

        # save the figure
        torsion_fig.savefig(os.path.join(self.directory, 'torsion.png'), dpi=300, bbox_inches="tight")
        shear_fig.savefig(os.path.join(self.directory, 'shear.png'), dpi=300, bbox_inches="tight")

    def loc_mech_ker(self, strain_figs = None, stress_figs = None, work_figs = None, fontsize = 12, label = None):
        """
        Only works for the cvbtk model (fiber_stress and ls are not (yet) saved in the beatit model)
        calculate and plot the local mechanics according to Kerchoffs 2003
        9 points:
            - endo, mid and epi
            - equatorial, mid, apical
        a) myofiber strain
        b) myofiber stress
        c) workloops
        """
        
        results = self.results
        par = self.parameters
    
        rad = ['endo', 'mid', 'epi']
        long = ['top', 'mid', 'bot']

        if not par['load_pickle'] or not os.path.exists(os.path.join(self.directory, 'loc_mech_data.pkl')):
            print('Calculating mechanics local points...')

            variables = ['fiber_stress', 'ls']

            self.open_hdf5(variables = variables)

            nsteps = self.hdf5_files['ls'].attributes('ls')['count']
            vector_numbers = list(range(nsteps))

            # define the phi value for the 9 points
            phi_free_wall = 1/2*math.pi
            # define the equatorial, mid and apical theta values
            theta_vals = [1/2*math.pi, 0.63*math.pi, 19/25*math.pi]

            focus = self.parameters['focus']

            # epsilons for the endo, mid and epicardial wall locations of the 9 points
            eps_outer = self.parameters['eps_outer']
            eps_mid = self.parameters['eps_mid']
            eps_inner = self.parameters['eps_inner'] 

            eps_vals = [eps_inner,eps_mid,eps_outer]

            data = {}
            count = 0

            # calculate and plot the mechanics over the whole cycle (all saved vector numbers)
            progress = len(vector_numbers)/10
            i = 0
            for step, vector in enumerate(vector_numbers):
                # loop over time (vector numbers)
                if i >= progress:
                    print('{0:.2f}% '.format(step/nsteps*100))
                    i = 0
                i += 1

                for vari in variables:
                    vector_time_name = self.hdf5_vectors[vari].format(vector)
                    self.hdf5_files[vari].read(self.functions[vari], vector_time_name)

                for l, theta in enumerate(theta_vals):
                    # for each point in longitudinal direction (equatorial, mid, apical)
                    for r, eps in enumerate(eps_vals):
                        # for each point in radial direction (endo, mid, epi)
                        # coordinates of the point:
                        x, y, z = ellipsoidal_to_cartesian(focus,eps,theta,phi_free_wall)

                        # stress and lenghth of that point at the timestep (vector number)
                        for vari in variables:
                            var = self.functions[vari](x, y, z)
                            if vari == 'ls':
                                var = np.log(var/self.parameters['ls0'])
                            data[count] = {'var': vari, 'time': step, 'slice': long[l], 'wall': rad[r], 'value': var}
                            count += 1

            df = pd.DataFrame.from_dict(data, "index")
            df.to_pickle(os.path.join(self.directory, 'loc_mech_data.pkl'))  

        else:
            df = pd.read_pickle(os.path.join(self.directory, 'loc_mech_data.pkl'))   

        print('Generating local mechanics plots...')

        # all the time points
        time = results['t_cycle']

        # create different figures for the strain, stress and workloops
        # 9 subplots for the 9 points are created for each figure
        if strain_figs == None:
            strain_fig, strain_plot = plt.subplots(3,3, sharex = True, sharey = True)
        else:
            strain_fig =  strain_figs[0]
            strain_plot =  strain_figs[1]
        strain_plot = strain_plot.ravel()

        if stress_figs == None:
            stress_fig, stress_plot = plt.subplots(3,3, sharex = True, sharey = True)
        else:
            stress_fig = stress_figs[0]
            stress_plot = stress_figs[1]
        stress_plot = stress_plot.ravel()

        if work_figs == None:
            work_fig, work_plot = plt.subplots(3,3, sharex = True, sharey = True)
        else:
            work_fig = work_figs[0]
            work_plot = work_figs[1]
        work_plot = work_plot.ravel()

        stress = df[df['var'] == 'fiber_stress']
        strain = df[df['var'] == 'ls']
        # loop over all subplots
        ii = 0
        for l in long:
            strain_l = strain[strain['slice']==l]
            stress_l = stress[stress['slice']==l]
            for r in rad:
                strain_val = strain_l[strain_l['wall']==r]
                stress_val = stress_l[stress_l['wall']==r]

                strain_plot[ii].plot(time[0:len(strain_val['value'])],strain_val['value'], label = label)
                stress_plot[ii].plot(time[0:len(stress_val['value'])],stress_val['value'], label = label)
                work_plot[ii].plot(strain_val['value'], stress_val['value'], label = label)

                for plot in [strain_plot[ii], stress_plot[ii], work_plot[ii]]:
                    if ii == 0:
                        plot.set_title('endocardium', fontsize = fontsize)

                    if ii == 2:
                        plot.set_title('epicardium', fontsize = fontsize)
                        plot.yaxis.set_label_position('right')
                        plot.set_ylabel('equatorial')

                    if ii == 5 and label != None:
                        plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                    if ii == 8:
                        plot.yaxis.set_label_position('right')
                        plot.set_ylabel('apical')
                ii += 1

        strain_fig.text(0.5, 0.04, 'time [ms]', ha='center', va='center', fontsize = fontsize)
        strain_fig.text(0.06, 0.5, 'strain', ha='center', va='center', rotation='vertical', fontsize = fontsize)
        strain_fig.suptitle('Myofiber strain', fontsize = fontsize +2)

        strain_fig.savefig(os.path.join(self.directory, 'myocardial_strain.png'), dpi=300, bbox_inches="tight")

        stress_fig.text(0.5, 0.04, 'time [ms]', ha='center', va='center', fontsize = fontsize)
        stress_fig.text(0.06, 0.5, 'stress [kPa]', ha='center', va='center', rotation='vertical', fontsize = fontsize)
        stress_fig.suptitle('Myofiber stress', fontsize = fontsize +2)

        stress_fig.savefig(os.path.join(self.directory, 'myocardial_stress.png'), dpi=300, bbox_inches="tight")

        work_fig.text(0.5, 0.04, 'strain', ha='center', va='center', fontsize = fontsize)
        work_fig.text(0.06, 0.5, 'stress [kPa]', ha='center', va='center', rotation='vertical', fontsize = fontsize)
        work_fig.suptitle('Workloops', fontsize = fontsize +2)

        work_fig.savefig(os.path.join(self.directory, 'workloops.png'), dpi=300, bbox_inches="tight")

        print('Plots created!')
    
    def open_hdf5(self, variables = []):
        for vari in variables:
            if self.model == 'beatit':
                if vari == 'Ell':
                    filename = os.path.join(self.directory,'postprocess/postprocess.h5')
                else:
                    filename = os.path.join(self.directory,'postprocess/postprocess_cylindrical.h5')
                vector = vari + '_rED_{}/vector_0'
            else:
                filename = os.path.join(self.directory,'{}.hdf5'.format(vari))
                vector = vari + '/vector_{}'

            self.hdf5_files[vari] = HDF5File(mpi_comm_world(), filename, 'r')
            self.hdf5_vectors[vari] = vector
            self.functions[vari] = Function(self.function_V)
            self.functions[vari].set_allow_extrapolation(True)

    def plot_ls_lc(self, title = '', variables = ['ls', 'lc']):
        results = self.results
        par = self.parameters

        if not par['load_pickle'] or not os.path.exists(os.path.join(self.directory, 'ls_lc_data.pkl')):
            self.open_hdf5(variables = variables)

            # extract the number of vectors (timesteps) in the files
            nsteps = self.hdf5_files[variables[0]].attributes(variables[0])['count']
            vector_numbers = list(range(nsteps))

            eps_epi = par['eps_outer']
            eps_mid = par['eps_mid']
            eps_endo = par['eps_inner']
            eps = [eps_endo, eps_mid, eps_epi]

            focus = par['focus']
            theta = 1/2 * math.pi
            phi = 0. 
            
            data = {}
            count = 0

            for step, vector in enumerate(vector_numbers):
                # define the vector names and extract the values of the timestep

                for wall, ep in enumerate(eps):
                    x, y, z = ellipsoidal_to_cartesian(focus,ep,theta, phi)

                    for vari in variables:
                        vector_time_name = self.hdf5_vectors[vari].format(vector)
                        self.hdf5_files[vari].read(self.functions[vari], vector_time_name)

                        length = self.functions[vari](x, y, z)

                        data[count] = {'var': vari, 'time': step, 'wall': wall, 'value': length}
                        count += 1

            # convert dictionary to dataframe and save
            df = pd.DataFrame.from_dict(data, "index")
            df.to_pickle(os.path.join(self.directory, 'ls_lc_data.pkl'))  

        else:
            df = pd.read_pickle(os.path.join(self.directory, 'ls_lc_data.pkl'))   


        time = (results[par['t_cycle']])
        # begin_phase = results[par['phase']][results.index[0]]
        # t_ed_cycle = results[par['t_cycle']][(results[par['phase']] == begin_phase + 1).idxmax()]

        phase2 = time[(results[par['phase']] == 1)[::-1].idxmax()] #- t_ed_cycle
        phase3 = time[(results[par['phase']] == 2)[::-1].idxmax()] #- t_ed_cycle
        phase4 = time[(results[par['phase']] == 3)[::-1].idxmax()] #- t_ed_cycle
        phase4_end = time[(results[par['phase']] == 4)[::-1].idxmax()] #- t_ed_cycle

        spec = GridSpec(2, 3)
        fig = plt.figure()
        endo = fig.add_subplot(spec[0,0])
        mid = fig.add_subplot(spec[0,1], sharey = endo)
        mid.tick_params(labelleft=False)
        epi = fig.add_subplot(spec[0,2], sharey = endo)
        epi.tick_params(labelleft=False)   

        av = fig.add_subplot(spec[1,:])     

        ls = df[df['var'] == 'ls']
        lc = df[df['var'] == 'lc']

        wall_loc = ['endo', 'mid', 'epi']
        plots = [endo, mid, epi]

        for wall_num, wall in enumerate(wall_loc):
            plots[wall_num].plot(time[:len(time)-1], ls['value'][ls['wall'] == wall_num], 
                    label = 'ls',
                    color = 'tab:red')
                    # linestyle = linestyle[wall_num]) 
            plots[wall_num].plot(time[:len(time)-1], lc['value'][lc['wall'] == wall_num], 
                    label = 'lc',
                    color = 'tab:green')
                    # linestyle = linestyle[wall_num]) 
            plots[wall_num].set_title(wall)
        epi.legend(loc='center left', bbox_to_anchor=(1, 0.5))            
        grouped_ls = ls.groupby('time', as_index = False).mean()
        grouped_lc = lc.groupby('time', as_index = False).mean()

        av.plot(time[:len(time)-1], grouped_ls['value'], 
                label = 'ls average',
                color = 'tab:red') 
        av.plot(time[:len(time)-1], grouped_lc['value'], 
                label = 'lc average',
                color = 'tab:green') 
        
        av.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))           

        for phase in [phase2, phase3, phase4, phase4_end]:
            for plot in [epi, mid, endo, av]:
                plot.axvline(x = phase, linestyle = '--', color = 'k', linewidth = 0.5)

        av.set_xlabel('time [ms]')
        epi.set_ylabel('length [um]')
        av.set_ylabel('length [um]')

        av2 = av.twinx()

        div_ls_lc = list(grouped_ls['value']) - grouped_lc['value']

        color = 'tab:blue'
        av2.plot(time[:len(time)-1], div_ls_lc, color = color)

        # diff.set_xlabel('time [ms]')
        av2.set_ylabel('ls - lc [um]', color = color)
        av2.tick_params(axis='y', labelcolor=color)

        fig.suptitle('ls vs lc ' + title)

        # av2.legend()
        fig.savefig(os.path.join(self.directory, 'ls_lc.png'), dpi=300, bbox_inches="tight")

        t_act = results[par['t_act']]
        p = []
        act = []
        ii = -200.
        for i, ls in enumerate(grouped_ls['value']):
            iso_cond = 0.
            if grouped_lc['value'][i] >= 1.5:
                iso_cond = 1.
            
            # iso_cond = conditional(ge(grouped_lc['value'][i], 1.5), 1, 0)

            iso_term = 140*(tanh(2.0*(grouped_lc['value'][i] - 1.5)))**2
            f_iso = iso_cond*iso_term

            t_max = 160*(ls + 1)

            twitch_term_1 = (tanh(t_act[t_act.index[0]]/ 75.0))**2
            twitch_term_2 = (tanh((t_max - t_act[t_act.index[0]])/150.0))**2
            
            twitch_cond_1 = 0.
            if i >= 0.:
                twitch_cond_1 = 1.
            # twitch_cond_1 = ge(i, 0)
            twitch_cond_2 = 0.
            if i <= t_max:
                twitch_cond_2 = 1.
            # twitch_cond_2 = le(i, t_max)
            twitch_cond = twitch_cond_1 * twitch_cond_2
            # twitch_cond = conditional(And(twitch_cond_1, twitch_cond_2), 1, 0)
            f_twitch = twitch_cond*twitch_term_1*twitch_term_2
            
            p.append(f_iso*f_twitch*20.*(ls - grouped_lc['value'][i]))
            act.append(f_twitch)
            ii += i
        
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for phase in [phase2, phase3, phase4, phase4_end]:
            ax.axvline(x = phase, linestyle = '--', color = 'k', linewidth = 0.5)

        ax.plot(time[:len(time)-1], p, label= 'p', color = 'tab:red')
        ax.tick_params(axis='y', labelcolor='tab:red')
        ax.set_ylabel('f_iso*f_twitch*Ea*(ls - lc)', color = 'tab:red')
        ax2 = ax.twinx()
        ax2.plot(time[:len(time)-1], act, label= 'act', color = 'tab:blue')
        ax2.set_ylabel('f_twitch', color = 'tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        fig.savefig(os.path.join(self.directory, 'activation.png'), dpi=300, bbox_inches="tight")

    def plot_strain(self, title = '', wall_points = 10, analyse_seg = None, variables = ['Ell', 'Ecc', 'Err', 'Ecr', 'myofiber_strain']):
        """
        Extracts and plots the strain, calculated by the postprocessing module of the model (saved as hdf5)
        For multiple slices, segments and wall locations strains are extracted from the hdf5 file.

        Args:
            - title: additional title of the plots
            - wall points: number of transmural points that will be analysed
            - variables: the types of strains that are analysed, 
              the options are all available strains, among which:
                'Ell': longitudinal strain
                'Ecc': circumferential strain
                'Err': radial strain
                'Ecr': circumferential-radial shear strain
                'Ett': transmural strain
                'myofiber_strain': myofiber strain
        
        If 'Ell', 'Ecc' and 'Err' exist, a table will be generated with the total average values of these 
        strains at end systole

        For every strain variable, the average of the strain, as well as the average of the strain for 
        the different slices and endo, mid and epi locations over the time will be generated. 
        For every strain variable, the average strain of the different slices transmurally will be plotted
        """
        print('Extracting strains from hdf5...')

        # create a figure to show which slices and transumural points will be analysed
        self.show_slices(title = title, wall_points = wall_points)

        results = self.results
        par = self.parameters

        # find the time of end systole
        begin_phase = results[par['phase']][results.index[0]]
        t_es_cycle = results[par['t_cycle']][(results[par['phase']] == begin_phase + 3).idxmax()]
        
        # find the time of end diastole
        t_ed_cycle = results[par['t_cycle']][(results[par['phase']] == begin_phase + 1).idxmax()]

        if self.model == 'cvbtk':
            # define the vector numbers containing the strain data at end systole and end diastole
            dt = results[par['time']][results.index[1]] - results[par['time']][results.index[0]]
            vector_es = t_es_cycle/dt
            vector_ed = t_ed_cycle/dt
            
            # create a handel for the strains
            self.open_hdf5(variables = variables)

            # extract the number of vectors (timesteps) in the files
            nsteps = self.hdf5_files[variables[0]].attributes(variables[0])['count']

            vector_numbers = list(range(nsteps))
            vector_numbers = list(map(int, vector_numbers))

            # for easy comparision with experimental data, the time loop starts at end diastole
            index_ed = vector_numbers.index(vector_ed)
            vector_numbers = vector_numbers[index_ed:] +  vector_numbers[:index_ed]

        elif self.model == 'beatit':
            # define the vector numbers containing the strain data at end systole and end diastole
            vector_es = t_es_cycle
            vector_ed = t_ed_cycle

            # create a handel for the strains
            self.open_hdf5(variables = variables)

            vector_numbers = results[par['time']][(results[par['cycle']] == self.cycle)]
            vector_numbers = vector_numbers.values.tolist()
            index_ed = vector_numbers.index(vector_ed)
            
            vector_numbers = vector_numbers[index_ed:] + vector_numbers[:index_ed]
            nsteps = len(vector_numbers)       

        # extract the theta values of the slices
        theta_vals = self.parameters['theta_vals']
        focus = self.parameters['focus']
        eps_outer = self.parameters['eps_outer']

        # define segments, 5 points per segment
        nr_segments = self.parameters['nr_segments']
        phi_int = 2*math.pi / nr_segments
        phi_range = np.arange(-1*math.pi, 1*math.pi, phi_int)

        # define the epsilon values of the transmural wall points
        delta_x = (self.x_epi - self.x_endo) / wall_points
        eps_vals = []
        for dx in range(wall_points + 1):
            x_pos = self.x_endo + dx * delta_x
            eps = cartesian_to_ellipsoidal(focus, x = [x_pos, 0.,0.])['eps']
            eps_vals.append(eps)

        # weighing factor, strain components are averaged in the transmural direction
        # outermost endo and epi points have a weighing factor of 0
        # the midwall point has the highest weighing factor
        r = np.arange(0,wall_points + 1)
        r = (r-r[-1]/2) / (r[-1]/2) 
        wtot = sum((1-r**2)**2)
        
        if not par['load_pickle'] or not os.path.exists(os.path.join(self.directory, 'data.pkl')):
            strain_function = {}
            # create dictionary to save results (direct save into dataframe is very slow)
            data = {}
            count = 0

            progress = nsteps/10
            i = 0  
            for steps, vector in enumerate(vector_numbers):
                # loop over time (vector numbers)
                
                if i >= progress:
                    print('{0:.2f}% '.format(steps/nsteps*100))
                    i = 0
                i += 1

                # read the strain data of the current vector number
                for vari in variables:
                    vector_time_name = self.hdf5_vectors[vari].format(vector)
                    self.hdf5_files[vari].read(self.functions[vari], vector_time_name)

                for slice_nr, theta in enumerate(theta_vals):
                    # loop over slices

                    # heigth of the epi point of the current slice
                    z_epi = ellipsoidal_to_cartesian(focus,eps_outer,theta,0.)[2]

                    for seg, phi in enumerate(phi_range):
                        # loop over segments
                        for wall, eps in enumerate(eps_vals):
                            # loop over wall locations

                            # the height of all the wall locations has to be the same, which means
                            # that for each epsilon value, the theta value has to be calculated
                            theta = math.acos(z_epi/(focus * math.cosh(eps)))

                            # coordinates of the current point
                            x, y, z = ellipsoidal_to_cartesian(focus,eps,theta, phi)

                            for vari in variables:
                                strain_function[vari] = self.functions[vari](x, y, z)

                            for vari in strain_function:
                                # strain = strain_function[vari]
                                # strain = (np.sqrt(1+ 2*strain_function[vari]) - 1) * 100 
                                strain = strain_function[vari] * 100

                                # weighing factor of the current wall point
                                w_ri = (1-r[wall]**2)**2 / wtot

                                data[count] = {'strain': vari, 'time': steps, 
                                    'slice': slice_nr, 'seg': seg, 'wall': wall, 
                                    'value': strain, 'weighted_value': strain * w_ri}

                                count += 1

            # convert dictionary to dataframe and save
            df = pd.DataFrame.from_dict(data, "index")
            df.to_pickle(os.path.join(self.directory, 'data.pkl'))

        else:
            df = pd.read_pickle(os.path.join(self.directory, 'data.pkl'))

        # make the table
        if ('Ell' and 'Ecc' and 'Err') in variables:
            # find the corresponding vector name for u
            dt = results[par['time']][results.index[1]] - results[par['time']][results.index[0]]
            index = int((t_es_cycle- t_ed_cycle)/dt)

            data_es = df[df['time'] == index]

            data_es_weight = data_es.groupby(['strain','time','slice','seg'], as_index=False).sum()

            Ell = data_es_weight[data_es_weight['strain'] == 'Ell']
            Ecc = data_es_weight[data_es_weight['strain'] == 'Ecc']
            Err = data_es_weight[data_es_weight['strain'] == 'Err']

            cell_text = []
            for slice_nr in range(0, len(theta_vals)):
                mean_Ell = round(Ell['weighted_value'][Ell['slice'] == slice_nr].mean(), 2)
                mean_Ecc = round(Ecc['weighted_value'][Ecc['slice'] == slice_nr].mean(), 2)
                mean_Err = round(Err['weighted_value'][Err['slice'] == slice_nr].mean(), 2)
                cell_text.append([mean_Ell, mean_Err, mean_Ecc])
            mean_Ell = round(Ell['weighted_value'].mean(), 2)
            mean_Ecc = round(Ecc['weighted_value'].mean(), 2)
            mean_Err = round(Err['weighted_value'].mean(), 2)
            cell_text.append([mean_Ell, mean_Err, mean_Ecc])

            rowLabels = ['slice ' + str(nr) for nr in range(0, len(theta_vals))]
            rowLabels.append('average')

            fig, ax = plt.subplots(figsize=(6,2))
            fig.patch.set_visible(False)
            ax.axis('off')
            ax.axis('tight')
            table = ax.table(  colLabels = ['Ell', 'Err', 'Ecc'],
                                rowLabels = rowLabels,
                                cellText = cell_text,
                                cellLoc = 'center',
                                loc = 'center')
            fig.suptitle('Average weighted strains at end systole \n {}'.format(title))
            fig.savefig(os.path.join(self.directory, 'weighted_strains.png'),dpi=300, bbox="tight")

        # make the figures
        spec = GridSpec(2, 3)
        time = (results[par['t_cycle']])

        # extract the times of the beginning of the end of the phases
        if self.model == 'beatit':
            phase2 = time[(results[par['phase']] == 2).idxmax() - 1] - t_ed_cycle
        else:
            phase2 = time[(results[par['phase']] == 1)[::-1].idxmax()] - t_ed_cycle
        phase3 = time[(results[par['phase']] == 2)[::-1].idxmax()] - t_ed_cycle
        phase4 = time[(results[par['phase']] == 3)[::-1].idxmax()] - t_ed_cycle
        phase4_end = time[(results[par['phase']] == 4)[::-1].idxmax()] - t_ed_cycle

        for strain_name in variables:
            fig = plt.figure()
            av = fig.add_subplot(spec[1,:])
            epi = fig.add_subplot(spec[0,0])
            mid = fig.add_subplot(spec[0,1], sharey = epi)
            mid.tick_params(labelleft=False)
            endo = fig.add_subplot(spec[0,2], sharey = epi)
            endo.tick_params(labelleft=False)

            plot = [epi, mid, endo]
            epi.set_ylabel('strain [%]')
            av.set_ylabel('weighted strain [%]')
            av.set_xlabel('time [ms]')
            walls = [0, int(len(eps_vals)/2), len(eps_vals) - 1]
            wall_names = ['endo', 'mid', 'epi']

            strain = df[df['strain'] == strain_name]
            if analyse_seg is not None:
                strain = strain[strain['seg'] == float(analyse_seg)]

            for slice_nr in range(0, len(theta_vals)):
                slice = strain[strain['slice'] == slice_nr]
                for ii, wall in enumerate(walls): 
                    wall = slice[slice['wall'] == wall]

                    grouped = wall.groupby('time', as_index=False).mean()
                    time_strain = grouped['value']

                    plot[ii].plot(time , time_strain, label = 'Slice {}'.format(slice_nr))
                    plot[ii].set_title(wall_names[ii])

                    if ii == 2:
                        plot[ii].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    
                    if slice_nr == 0:
                        for phase in [phase2, phase3, phase4, phase4_end]:
                            plot[ii].axvline(x = phase, linestyle = '--', color = 'k', linewidth = 0.5)
                            av.axvline(x = phase, linestyle = '--', color = 'k', linewidth = 0.5)

                slice = slice.groupby(['time','seg'], as_index=False).sum()
                grouped = slice.groupby('time', as_index=False).mean()
                time_strain = grouped['weighted_value']
                av.plot(time, time_strain, label = 'Slice {}'.format(slice_nr))

            strain = strain.groupby(['time','slice','seg'], as_index=False).sum()
            grouped = strain.groupby('time', as_index=False).mean()
            time_strain = grouped['weighted_value']

            av.plot(time, time_strain, '--', label = 'average')
            av.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            if analyse_seg is not None:
                fig.suptitle('{strain} segment {seg} {title}'.format(strain = strain_name, seg = analyse_seg, title = title))
                fig.savefig(os.path.join(self.directory, strain_name + '_weighted_model_segment_{}.png'.format(int(analyse_seg))), dpi=300, bbox_inches="tight")
            else:
                fig.suptitle('{strain} {title}'.format(strain = strain_name, title = title))
                fig.savefig(os.path.join(self.directory, strain_name + '_weighted_model.png'), dpi=300, bbox_inches="tight")
                 
            fig_wall, ax_wall = plt.subplots() 
            fig_seg, ax_seg = plt.subplots() 
            # data_ed = df[df['time'] == 0] 
            strain = data_es[data_es['strain'] == strain_name]
            for slice_nr in range(0, len(theta_vals)):
                slice = strain[strain['slice'] == slice_nr]

                grouped_wall = slice.groupby('wall', as_index=False).mean()
                if analyse_seg is not None:
                    slice_wall = slice[slice['seg'] == float(analyse_seg)]
                    grouped_wall = slice_wall.groupby('wall', as_index=False).mean()
      
                grouped_seg = slice.groupby('seg', as_index=False).mean()

                wall_strain = grouped_wall['value']
                seg_strain = grouped_seg['value']

                ax_wall.plot(range(len(eps_vals)), wall_strain[::-1], label = 'Slice {}'.format(slice_nr))
                ax_seg.plot(range(len(phi_range)), seg_strain, label = 'Slice {}'.format(slice_nr))

            ax_wall.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax_seg.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                
            ax_seg.set_title(strain_name + ' circumferential at end systole ' + title)

            ax_wall.set_ylabel('strain [%]')
            ax_seg.set_ylabel('strain [%]')

            ax_wall.set_xlabel('epicardial    -    endocardial')
            ax_seg.set_xlabel('segment')

            if analyse_seg is not None:
                ax_wall.set_title(strain_name + ' transmural at end systole segment {} '.format(analyse_seg) + title)
                fig_wall.savefig(os.path.join(self.directory, strain_name + '_transmural_model_es_segment_{}.png'.format(analyse_seg)), dpi=300, bbox_inches="tight")
            else:
                ax_wall.set_title(strain_name + ' transmural at end systole' + title)
                fig_wall.savefig(os.path.join(self.directory, strain_name + '_transmural_model_es.png'), dpi=300, bbox_inches="tight")
            fig_seg.savefig(os.path.join(self.directory, strain_name + '_circumferential_model_es.png'), dpi=300, bbox_inches="tight")
 

 
    def show_slices(self, fig = None, fontsize=12, title = '', wall_points = 10):
        """
        Visual representation of the slices and segments
        used in the calculations of the torsion and shear

        In addition, a plot is created showing the absolute location
        at end diastole and systole of the basel segments and the
        segments of the second to last slice
        """
        # Plot the regions.
        if fig is None:
            fig = plt.figure()

        # set size of figure
        fig.set_size_inches(20, 13)

        # Make fig current.
        plt.figure(fig.number)

        # colors of the slices
        col = [ 'C7','C5', 'C0', 'C1','C2','C3', 'C4','C8']

        try:
            self.u_es
        except AttributeError:
            u_exists = False
            gs = GridSpec(1,2)
        else:
            u_exists = True
            gs = GridSpec(2,2)

            # displacement at ed and es
            _xy2 = plt.subplot(gs[1,:])

            # also create a seperate figure for the displacement at ed and es
            ed_es_fig, ed_es_plot = plt.subplots(1,1)

        # top view slices
        _xy1 = plt.subplot(gs[0,0])
        # side view slices
        _xz1 = plt.subplot(gs[0,1])

        if u_exists:
            self._ax = {'xy1': _xy1, 'xz1': _xz1, 'xy2': _xy2}
        else:
            self._ax = {'xy1': _xy1, 'xz1': _xz1}

        if self.infarct:
            self.variables = ['T0']
            self.open_hdf5()

        # extract the theta and phi values as used in the calculations
        theta_vals = self.parameters['theta_vals']
        nr_segments = self.parameters['nr_segments']
        focus = self.parameters['focus']

        # draw the outlines of the left ventricle
        self.lv_drawing(top_keys = ['xy1'], side_keys = ['xz1'], xy_theta=theta_vals[1])

        delta_x = (self.x_epi - self.x_endo) / wall_points
        eps_vals = []
        for dx in range(wall_points + 1):
            x_pos = self.x_endo + dx * delta_x
            eps = cartesian_to_ellipsoidal(focus, x = [x_pos, 0.,0.])['eps']
            eps_vals.append(eps)

        eps_outer = self.parameters['eps_outer']

        phi_int = 2*math.pi / nr_segments
        phi_range = np.arange(-1*math.pi, 1*math.pi, phi_int)

        for slice_nr, theta in enumerate(theta_vals):
            # loop over slices
            x_segs = []
            y_segs = []
            z_segs = []

            x_segs_es = []
            y_segs_es = []
            z_segs_es = []

            x_segs_ed = []
            y_segs_ed = []
            z_segs_ed = []

            # calculate theta values for (epi), mid and endo to keep the z position
            # the same for all the points within a slice
            x, y, z_epi = ellipsoidal_to_cartesian(focus,eps_outer,theta,0.0)
            # tau = z_epi/(focus * math.cosh(eps_mid))
            # theta_mid = math.acos(tau)
            # tau = z_epi/(focus * math.cosh(eps_inner))
            # theta_inner = math.acos(tau)

            # thetas = [theta, theta_mid, theta_inner]
            for seg, phi in enumerate(phi_range):
                #loop over segments
                seg_es = []
                seg_ed = []

                for wall, eps in enumerate(eps_vals):
                    # loop over wall locations
                    theta_eps = math.acos(z_epi/(focus * math.cosh(eps)))
                    # get and save coordinates of point
                    x, y, z = ellipsoidal_to_cartesian(focus,eps,theta_eps,phi)
                    x_segs.append(x)
                    y_segs.append(y)
                    z_segs.append(z)

                    if u_exists:
                        # get displacement of the point at es and ed
                        # NOTE self.u_es and self._ed are created in def torsion.
                        # and should thus be called before this definition
                        xu, yu, zu = self.u_es(x, y, z)
                        x_es = x + xu
                        y_es = y + yu
                        z_es = z + zu

                        x_segs_es.append(x_es)
                        y_segs_es.append(y_es)
                        z_segs_es.append(z_es)

                        seg_es.append([x_es, y_es])

                        xu, yu, zu = self.u_ed(x, y, z)
                        x_ed = x + xu
                        y_ed = y + yu
                        z_ed = z + zu
                        x_segs_ed.append(x_ed)
                        y_segs_ed.append(y_ed)
                        z_segs_ed.append(z_ed)

                        seg_ed.append([x_ed,y_ed])

                if u_exists and (slice_nr == 0 or slice_nr == len(theta_vals)-2):
                    # only plot the absolute locations of points of the basel and the second to last slice
                    # connect the epi, mid and endo point of each segment for easy viewing
                    self._ax['xy2'].plot([i[0] for i in seg_ed], [i[1] for i in seg_ed], 'o-', color=col[slice_nr])
                    self._ax['xy2'].plot([i[0] for i in seg_es], [i[1] for i in seg_es], 'x--', color=col[slice_nr])
                    ed_es_plot.plot([i[0] for i in seg_ed], [i[1] for i in seg_ed], 'o-', color=col[slice_nr])
                    ed_es_plot.plot([i[0] for i in seg_es], [i[1] for i in seg_es], 'x--', color=col[slice_nr])

            if slice_nr == 1:
                self._ax['xy1'].scatter(x_segs, y_segs, color=col[slice_nr], label= 'slice ' + str(slice_nr))
            self._ax['xz1'].scatter(x_segs, z_segs, color=col[slice_nr], label= 'slice ' + str(slice_nr))

            if u_exists and (slice_nr == 0 or slice_nr == len(theta_vals)-2):
                # only plot the absolute locations of points of the basel and the second to last slice
                # this plot is technically redundant, but I'm too lazy to think of a way to correctly add the labels
                self._ax['xy2'].scatter(x_segs_ed, y_segs_ed, color=col[slice_nr], label= 'ed slice ' + str(slice_nr))
                self._ax['xy2'].scatter(x_segs_es, y_segs_es, marker='x', color=col[slice_nr], label= 'es slice ' + str(slice_nr))
                ed_es_plot.scatter(x_segs_ed, y_segs_ed, color=col[slice_nr], label= 'ed slice ' + str(slice_nr))
                ed_es_plot.scatter(x_segs_es, y_segs_es, marker='x', color=col[slice_nr], label= 'es slice ' + str(slice_nr))

        self._ax['xy1'].axis('equal')
        self._ax['xy1'].set_title('top view (x-y)')
        self._ax['xy1'].legend(frameon=False, fontsize=fontsize)

        self._ax['xz1'].set_title('front view (x-z)')
        self._ax['xz1'].axis('equal')
        self._ax['xz1'].legend(frameon=False, fontsize=fontsize)

        if u_exists:
            self._ax['xy2'].axis('equal')
            self._ax['xy2'].set_title('top view (x-y) end diastole and end systole')
            self._ax['xy2'].legend(frameon=False, fontsize=fontsize)

            ed_es_plot.axis('equal')
            ed_es_plot.set_title('top view (x-y) end diastole and end systole')
            ed_es_plot.legend(frameon=False, fontsize=fontsize)

            fig.suptitle(title,fontsize=fontsize+4)
            ed_es_fig.suptitle(title,fontsize=fontsize+2)

            # save the figures
            ed_es_fig.savefig(os.path.join(self.directory, 'ed_es.png'), dpi=300, bbox_inches="tight")

        fig.savefig(os.path.join(self.directory, 'slices.png'), dpi=300, bbox_inches="tight")

    def show_ker_points(self):
        """
        Visual representation of the 9 points used in the
        function of the local mechanics of Kerchoffs 2003
        """

        fig = plt.figure()
        plt.figure(fig.number)

        self._ax = {'fig': plt}

        # draw the outlines of the left ventricle
        self.lv_drawing(side_keys = ['fig'])

        # the same phi values used in loc_mech_ker
        # TODO could be improved to make certain same values are used
        phi_free_wall = 1/2*math.pi
        theta_vals = [1/2*math.pi, 0.63*math.pi, 19/25*math.pi]

        focus = self.parameters['focus']
        tol = self.parameters['tol']
        eps_outer = self.parameters['eps_outer'] - tol
        eps_mid = self.parameters['eps_mid']
        eps_inner = self.parameters['eps_inner'] + tol

        eps_vals = [eps_inner,eps_mid,eps_outer]

        for theta in theta_vals:
            #loop over slices
            for eps in eps_vals:
                # loop over segments
                # coordinates of point:
                x, y, z = ellipsoidal_to_cartesian(focus,eps,theta,phi_free_wall)
                plt.scatter(y, z)
        plt.xlabel('y-axis [cm]')
        plt.ylabel('z-axis [cm]')
        plt.title('Local mechanics points (Kerckhoffs 2003)')

        # save the figure
        plt.savefig(os.path.join(self.directory, 'points_ker.png'), dpi=300, bbox_inches="tight")

    def lv_drawing(self, side_keys = None, top_keys = None, xy_theta =1/2*math.pi):
        """
        Draw the outlines of the LV

        """

        def ellips(a, b, t):
            x = a*np.cos(t)
            y = b*np.sin(t)
            return x, y

        def cutoff(x, y, h):
            x = x[y<=h]
            y = y[y<=h]
            return x, y

        R_inner, y, z = ellipsoidal_to_cartesian(self.parameters['focus'],self.parameters['eps_inner'],xy_theta, 0.)
        R_outer, y, z = ellipsoidal_to_cartesian(self.parameters['focus'],self.parameters['eps_outer'],xy_theta, 0.)

        R_1, y, z = ellipsoidal_to_cartesian(self.parameters['focus'],self.parameters['eps_inner'],1/2*math.pi, 0.)
        R_2, y, z = ellipsoidal_to_cartesian(self.parameters['focus'],self.parameters['eps_outer'],1/2*math.pi, 0.)

        x, y, Z1 = ellipsoidal_to_cartesian(self.parameters['focus'],self.parameters['eps_inner'],math.pi, 0.)
        x, y, Z2 = ellipsoidal_to_cartesian(self.parameters['focus'],self.parameters['eps_outer'],math.pi, 0.)

        Z_1 = abs(Z1)
        Z_2 = abs(Z2)

        h = self.parameters['cut_off']
        n = 500
        lw = 1


        # LV above
        t_lvfw = np.linspace(-np.pi, np.pi, n)
        x_endo, y_endo = ellips(R_inner, R_inner, t_lvfw)
        x_epi, y_epi = ellips(R_outer, R_outer, t_lvfw)

        if top_keys != None:
            for key in top_keys:
                self._ax[key].plot(x_endo, y_endo, color='C3', linewidth=lw)
                self._ax[key].plot(x_epi, y_epi, color='C3', linewidth=lw)
                self._ax[key].axis('equal')

        # LV side
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
# directory_1 = '/home/maaike/Documents/Graduation_project/Results/eikonal_td_1_node/cycle_2_begin_ic_ref'
# directory_1 = '/home/maaike/Documents/Graduation_project/Results/beatit'
# directory_1 = '/home/maaike/Documents/Graduation_project/Results/ref_2_cyc_mesh_20/cycle_2_begin_ic_ref'
# dict_vals = {'theta_vals' : [9/20*math.pi, 11/20*math.pi, 13/20*math.pi, 15/20*math.pi]}
# dict_vals = {'theta_vals' : [9/20*math.pi, 11/20*math.pi, 13/20*math.pi, 15/20*math.pi]}

# post_1 = postprocess_hdf5(directory_1, **dict_vals)
# # post_1.loc_mech_ker()
# # post_1.show_ker_points()
# post_1.plot_torsion()
# post_1.plot_strain()
# post_1.show_slices()