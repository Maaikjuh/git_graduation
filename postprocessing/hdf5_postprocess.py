import sys
#ubuntu
sys.path.append('/mnt/c/Users/Maaike/Documents/Master/Graduation_project/git_graduation_project/cvbtk')
sys.path.append('/mnt/c/Users/Maaike/Documents/Master/Graduation_project/git_graduation_project/postprocessing')

#   linux
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
        self.mesh = self.load_mesh()
        self._parameters = self.default_parameters()

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

            if self.inputs['active_stress']['eikonal'] is not None:
                self.eikonal_dir = self.inputs['active_stress']['eikonal']['td_dir']

            self.parameters['ls0'] = self.inputs['active_stress']['ls0']

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

        if 'inner_eccentricity' or 'outer_eccentricity' in kwargs:
            e_outer = self.parameters['outer_eccentricity']
            e_inner = self.parameters['inner_eccentricity']
            eps_outer = cartesian_to_ellipsoidal(self.parameters['focus'], eccentricity = e_outer)['eps']
            eps_inner = cartesian_to_ellipsoidal(self.parameters['focus'], eccentricity = e_inner)['eps']
            self.parameters['eps_outer'] = eps_outer
            self.parameters['eps_inner'] = eps_inner

        # calculate eps_mid:
        # get x_epi and x_endo at the equatorial point (theta = 1/2*pi)
        # calculate x_mid -> calculate eps_mid using the focus
        focus = self.parameters['focus']
        x_epi, y_epi, z_epi = ellipsoidal_to_cartesian(focus,eps_outer,1/2*math.pi,0.)
        x_endo, y_endo, z_endo = ellipsoidal_to_cartesian(focus,eps_inner,1/2*math.pi,0.)
        x_mid = (x_epi + x_endo)/2
        eps_mid = cartesian_to_ellipsoidal(self.parameters['focus'], x = [x_mid, 0.,0.])['eps']
        self.parameters['eps_mid'] = eps_mid

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

        par['focus']= 4.3
        par['ls0'] = 1.9
        par['name'] = ''

        par['phase'] = 'phase'
        par['time'] = 'time' 
        par['t_cycle'] = 't_cycle'
        par['cycle'] = 'cycle' 
        return par

    def load_mesh(self):
        mesh = Mesh()
        if self.model == 'cvbtk':
            openfile = HDF5File(mpi_comm_world(),os.path.join(self.directory,'mesh.hdf5'), 'r')

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

        focus = self.parameters['focus']
        eps_outer = self.parameters['eps_outer']
        eps_mid = self.parameters['eps_mid']
        eps_inner = self.parameters['eps_inner']

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
                    # torsion is the rotation of a point in a slice with
                    # respect to the rotation of a point in the base
                    # with the same phi value and wall location:
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

        # create a read handel for the stress and fiber length files
        stress_file = HDF5File(mpi_comm_world(), os.path.join(self.directory,'fiber_stress.hdf5'), 'r')
        ls_file = HDF5File(mpi_comm_world(), os.path.join(self.directory,'ls.hdf5'), 'r')

        # extract the number of vectors (timesteps) in the files
        nsteps = ls_file.attributes('ls_old')['count']
        vector_numbers = list(range(nsteps))

        # initialize the variables
        stress_func = Function(self.function_V)
        ls_func = Function(self.function_V)

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

        # calculate and plot the mechanics over the whole cycle (all saved vector numbers)
        for vector in vector_numbers:
            print('{0:.2f}% '.format(vector/nsteps*100))

            # define the vector names and extract the values of the timestep
            stress_vector = 'fiber_stress/vector_{}'.format(vector)
            ls_vector = 'ls_old/vector_{}'.format(vector)

            stress_file.read(stress_func, stress_vector)
            ls_file.read(ls_func, ls_vector)

            # allow extrapolation in order to define the displacement values
            # at non-nodal points
            stress_func.set_allow_extrapolation(True)
            ls_func.set_allow_extrapolation(True)

            for l, theta in enumerate(theta_vals):
                # for each point in longitudinal direction (equatorial, mid, apical)
                for r, eps in enumerate(eps_vals):
                    # for each point in radial direction (endo, mid, epi)
                    # coordinates of the point:
                    x, y, z = ellipsoidal_to_cartesian(focus,eps,theta,phi_free_wall)

                    # stress and lenghth of that point at the timestep (vector number)
                    stress = stress_func(x, y, z)
                    ls = ls_func(x, y, z)
                    strain = np.log(ls/self.parameters['ls0'])

                    stress_cycle[long[l] + '_' + rad[r]].append(stress)
                    strain_cycle[long[l] + '_' + rad[r]].append(strain)
        stress_file.close()
        ls_file.close()
        print('Generating local mechanics plots...')

        # all the time points
        time = self.results['t_cycle']

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

        # loop over all subplots
        ii = 0
        for l in long:
            for r in rad:
                strain_plot[ii].plot(time[0:len(vector_numbers)], strain_cycle[l + '_' + r], label = label)
                stress_plot[ii].plot(time[0:len(vector_numbers)], stress_cycle[l + '_' + r], label = label)
                work_plot[ii].plot(strain_cycle[l + '_' + r], stress_cycle[l + '_' + r], label = label)

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

    def plot_strain(self):
        results = self.results

        # create a read handel for the longitudinal, circumferential and radial strains
        Ell_file = HDF5File(mpi_comm_world(), os.path.join(self.directory,'Ell.hdf5'), 'r')
        Ecc_file = HDF5File(mpi_comm_world(), os.path.join(self.directory,'Ecc.hdf5'), 'r')
        Err_file = HDF5File(mpi_comm_world(), os.path.join(self.directory,'Err.hdf5'), 'r')

        # extract the number of vectors (timesteps) in the files
        nsteps = Ell_file.attributes('Ell')['count']
        vector_numbers = list(range(nsteps))

        # initialize the variables
        Ell_func = Function(self.function_V)
        Ecc_func = Function(self.function_V)
        Err_func = Function(self.function_V)

        # allow extrapolation in order to define the displacement values
        # at non-nodal points
        Ell_func.set_allow_extrapolation(True)
        Ecc_func.set_allow_extrapolation(True)
        Err_func.set_allow_extrapolation(True)

        # extract the theta values of the slices
        theta_vals = self.parameters['theta_vals']

        # define 6 segments, 5 points per segment
        phi_int = 2*math.pi / 6
        phi_seg = phi_int/ 5
        phi_range = np.arange(-1*math.pi, 1*math.pi, phi_int)
        phi_seg = np.arange(0., phi_int, phi_seg)

        focus = self.parameters['focus']
        eps_outer = self.parameters['eps_outer']
        eps_mid = self.parameters['eps_mid']
        eps_inner = self.parameters['eps_inner']
        wall_vals = ['epi', 'mid', 'endo']

        # create dictionary to store average strain values per slice
        slices_strains = {}
        slices_strains_wall = {}
        time_strain = {}

        progress = nsteps/10
        i = 0
        for vector in vector_numbers:
            # loop over time (vector numbers)
            if i >= progress:
                print('{0:.2f}% '.format(vector/nsteps*100))
                i = 0
            i += 1

            Ell_vector = 'Ell/vector_{}'.format(vector)
            Ecc_vector = 'Ecc/vector_{}'.format(vector)
            Err_vector = 'Err/vector_{}'.format(vector)

            Ell_file.read(Ell_func, Ell_vector)
            Ecc_file.read(Ecc_func, Ecc_vector)
            Err_file.read(Err_func, Err_vector)

            all_seg = {'Ell': [],
                        'Ecc': [],
                        'Err': []}

            for slice_nr, theta in enumerate(theta_vals):
                # loop over slices

                timestep_wall = {}

                x_epi, y_epi, z_epi = ellipsoidal_to_cartesian(focus,eps_outer,theta,0.)
                tau = z_epi/(focus * math.cosh(eps_mid))
                theta_mid = math.acos(tau)

                tau = z_epi/(focus * math.cosh(eps_inner))
                theta_inner = math.acos(tau)

                thetas = [theta, theta_mid, theta_inner]

                for seg, phi in enumerate(phi_range):
                    # loop over segments
                    for phi_step in phi_seg:
                        # per segments, 5 points are used
                        for wall, eps in enumerate([eps_outer, eps_mid, eps_inner]):
                            # loop over wall location (epi, mid, endo)
                            x, y, z = ellipsoidal_to_cartesian(focus,eps,thetas[wall], phi + phi_step)

                            Ell = Ell_func(x, y, z)
                            Ecc = Ecc_func(x, y, z)
                            Err = Err_func(x, y, z)

                            strain_var = ['Ell', 'Ecc', 'Err']

                            for name, strain in enumerate([Ell, Ecc, Err]):
                                save_slice_wall = '{}_slice_{}_{}'.format(strain_var[name], slice_nr, wall_vals[wall])
                                save_slice = '{}_slice_{}'.format(strain_var[name], slice_nr)

                                all_seg[strain_var[name]].append(strain * 100)

                                if vector == 0 and seg == 0 and phi_step == 0 and wall == 0:
                                    slices_strains[save_slice] = [strain * 100]

                                    # slices_strains_wall[save_slice_wall] = [strain * 100]
                                else:
                                    slices_strains[save_slice].append(strain * 100)

                                    # slices_strains_wall[save_slice_wall].append(strain * 100)
                                if vector == 0 and seg == 0 and phi_step == 0:
                                    slices_strains_wall[save_slice_wall] = [strain * 100]
                                else:
                                    slices_strains_wall[save_slice_wall].append(strain * 100)

                                if seg == 0 and phi_step == 0:
                                    timestep_wall[strain_var[name] + '_' + wall_vals[wall]] = [strain * 100]

                                else:
                                    timestep_wall[strain_var[name] + '_' + wall_vals[wall]].append(strain*100)

                                if seg == 0 and phi_step == 0 and wall == 0:
                                    timestep_wall[save_slice] = [strain * 100]
                                else:
                                    timestep_wall[save_slice].append(strain*100)

                for strain_name in ['Ell', 'Ecc', 'Err']:
                    if vector == 0:
                        time_strain['{}_slice_{}'.format(strain_name, slice_nr)] = [np.mean(timestep_wall['{}_slice_{}'.format(strain_name, slice_nr)])]
                    else:
                        time_strain['{}_slice_{}'.format(strain_name, slice_nr)].append(np.mean(timestep_wall['{}_slice_{}'.format(strain_name, slice_nr)]))
                    for wall in wall_vals:
                        if vector == 0:
                            time_strain[strain_name + '_slice_{}_'.format(slice_nr) + wall] = [np.mean(timestep_wall[strain_name + '_' + wall])]
                        else:
                            time_strain[strain_name + '_slice_{}_'.format(slice_nr) + wall].append(np.mean(timestep_wall[strain_name + '_' + wall]))
            for strain_name in ['Ell', 'Ecc', 'Err']:
                if vector == 0:
                    time_strain[strain_name] = [np.mean(all_seg[strain_name])]
                else:
                    time_strain[strain_name].append(np.mean(all_seg[strain_name]))

        begin_phase = results['phase'][results.index[0]]
        t_es_cycle = results['t_cycle'][(results['phase'] == begin_phase + 3).idxmax()]

        # find the corresponding vector name for u
        dt = results['time'][results.index[1]] - results['time'][results.index[0]]
        vector = int(t_es_cycle/dt)

        cell_text = []
        for row in range(0, len(theta_vals)):
            # max_Ell = round(np.mean(slices_strains['Ell_slice_{}'.format(row)]), 2)
            # max_Ecc = round(np.mean(slices_strains['Ecc_slice_{}'.format(row)]), 2)
            # max_Err = round(np.mean(slices_strains['Err_slice_{}'.format(row)]), 2)
            mean_Ell = round((time_strain['Ell_slice_{}'.format(row)][vector]), 2)
            mean_Ecc = round((time_strain['Ecc_slice_{}'.format(row)][vector]), 2)
            mean_Err = round((time_strain['Err_slice_{}'.format(row)][vector]), 2)
            cell_text.append([mean_Ell, mean_Err, mean_Ecc])
        mean_Ell = round((time_strain['Ell'][vector]), 2)
        mean_Ecc = round((time_strain['Ecc'][vector]), 2)
        mean_Err = round((time_strain['Err'][vector]), 2)
        cell_text.append([mean_Ell, mean_Err, mean_Ecc])

        rowLabels = ['slice ' + str(nr) for nr in range(0, len(theta_vals))]
        rowLabels.append('average')

        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        table = ax.table(  colLabels = ['Ell', 'Err', 'Ecc'],
                            rowLabels = rowLabels,
                            cellText = cell_text,
                            loc = 'center')

        fig.suptitle('Average strains at end systole')
        fig.savefig(os.path.join(self.directory, 'strains.png'))

        spec = GridSpec(2, 3)
        time = results['t_cycle']

        for strain_name in ['Ell', 'Ecc', 'Err']:
            fig = plt.figure()
            av = fig.add_subplot(spec[1,:])
            epi = fig.add_subplot(spec[0,0])
            mid = fig.add_subplot(spec[0,1], sharey = epi)
            endo = fig.add_subplot(spec[0,2], sharey = epi)

            plot = [epi, mid, endo]
            for slice_nr in range(0, len(theta_vals)):
                for ii, wall in enumerate(wall_vals):
                    strain_dict = strain_name + '_slice_{}_{}'.format(slice_nr,  wall)
                    plot[ii].plot(time , time_strain[strain_dict], label = 'Slice {}'.format(slice_nr))
                    plot[ii].set_title(wall)

                    if ii == 2:
                        plot[ii].legend(loc='center left', bbox_to_anchor=(1, 0.5))

            av.plot(time, time_strain[strain_name], label = 'average')
            av.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            fig.suptitle(strain_name)
            fig.savefig(os.path.join(self.directory, strain_name + '_model.png'), dpi=300, bbox_inches="tight")



    def calc_strain(self):
        results = self.results
        par = self.parameters

        dt = results[par['time']][results.index[1]] - results[par['time']][results.index[0]]

        # find the time of end diastole
        begin_phase = results[par['phase']][results.index[0]]

        t_ed_cycle = results[par['t_cycle']][(results[par['phase']] == begin_phase + 1).idxmax()]
        t_es_cycle = results[par['t_cycle']][(results[par['phase']] == begin_phase + 3).idxmax()]

        if self.model == 'cvbtk':
            u_file = HDF5File(mpi_comm_world(), os.path.join(self.directory,'u.hdf5'), 'r')

            # extract the number of vectors (timesteps) in the files
            nsteps = u_file.attributes('u')['count']
            vector_numbers = list(range(nsteps))

            # find the corresponding vector name for u
            vector_ed = t_ed_cycle/dt
            u_ed_vector = 'u/vector_{}'.format(int(vector_ed))

            # find the corresponding vector name for u
            vector_es = t_es_cycle/dt
            u_es_vector = 'u/vector_{}'.format(int(vector_es))

            u_vector = 'u/vector_{}'

            vector_numbers.insert(0, vector_ed)
            vector_numbers = list(map(int, vector_numbers))
        
        elif self.model == 'beatit':
            # displacement vector is equal to the time
            u_es_vector = 'displacement_{}/vector_0'.format(float(t_es_cycle))
            u_ed_vector = 'displacement_{}/vector_0'.format(float(t_ed_cycle))
            u_vector = 'displacement_{}/vector_0'

            u_file = HDF5File(mpi_comm_world(),os.path.join(self.directory,'results.h5'), 'r')

            vector_numbers = results[par['time']][(results[par['cycle']] == self.cycle)]
            vector_numbers = vector_numbers.values.tolist()
            vector_numbers.insert(0, t_ed_cycle)
            nsteps = len(vector_numbers)

        self.u_ed = Function(self.vector_V)
        self.u_es = Function(self.vector_V)
        self.u = Function(self.vector_V)
        
        u_file.read(self.u_ed, u_ed_vector)
        u_file.read(self.u_es, u_es_vector)

        # allow extrapolation in order to define the displacement values
        # at non-nodal points
        self.u_es.set_allow_extrapolation(True)
        self.u_ed.set_allow_extrapolation(True)
        self.u.set_allow_extrapolation(True)

        # extract the theta values of the slices
        theta_vals = self.parameters['theta_vals']

        # extract the number of segments and convert to a phi range
        nr_segments = self.parameters['nr_segments']

        phi_int = 2*math.pi / nr_segments
        phi_range = np.arange(-1*math.pi, 1*math.pi, phi_int)

        focus = self.parameters['focus']
        eps_outer = self.parameters['eps_outer']
        eps_mid = self.parameters['eps_mid']
        eps_inner = self.parameters['eps_inner']        

        Err = {'tot': []}
        Ell = {'tot_epi': [],
                'tot_endo': []}
        Ecc = {'tot_epi': [],
                'tot_endo': []}  

        strains = {'Err_tot': [],
                    'Ecc_tot': [],
                    'Ell_tot': []}      

        slice_prev_ed = {}
        slice_prev_es = {}

        slice_ed = {}
        slice_seg = {}
        time_strain = {}

        wall_vals = ['epi', 'mid', 'endo']

        progress = len(vector_numbers)/10
        i = 0
        for step, vector in enumerate(vector_numbers):
            # loop over time (vector numbers)
            if i >= progress:
                print('{0:.2f}% '.format(step/nsteps*100))
                i = 0
            i += 1

            u_vector_time = u_vector.format(vector)
            u_file.read(self.u, u_vector_time)

            for slice_nr, theta in enumerate(theta_vals):
                slice_prev_ed['slice_{}'.format(slice_nr)] = []
                slice_prev_es['slice_{}'.format(slice_nr)] = []
                strains['Err_slice_{}'.format(slice_nr)] = []

                # calculate theta values for mid and inner wall
                x_epi, y_epi, z_epi = ellipsoidal_to_cartesian(focus,eps_outer,theta,0.)
                tau = z_epi/(focus * math.cosh(eps_mid))
                theta_mid = math.acos(tau)

                tau = z_epi/(focus * math.cosh(eps_inner))
                theta_inner = math.acos(tau)

                thetas = [theta, theta_mid, theta_inner]

                # iterate over the segments (different phi values per slice)
                for seg, phi in enumerate(phi_range):
                    # iterate over epi, mid and inner wall
                    for wall, eps in enumerate([eps_outer, eps_mid, eps_inner]):
                        # calulate coordinate of the point 
                        x, y, z = ellipsoidal_to_cartesian(focus,eps,thetas[wall], phi)

                        # get the displacement of the point at the point in time
                        # and add the displacement tot the coordinates
                        # to get the exact location of the point in time
                        xu, yu, zu = self.u(x, y, z)
                        x += xu
                        y += yu
                        z += zu

                        # the first vector number (step) is the time of ed
                        # save the values for reference state
                        if step == 0:
                            slice_ed['slice_{}_seg_{}_{}'.format(slice_nr, seg, wall_vals[wall])] = [x,y,z]

                        else:
                            #Err
                            if wall == 2:
                                # get locations of the point at ed for epi and endo
                                ed_epi = slice_ed['slice_{}_seg_{}_epi'.format(slice_nr, seg)]
                                ed_endo = slice_ed['slice_{}_seg_{}_endo'.format(slice_nr,seg)]

                                # get location of the point at the current time step for epi
                                seg_epi = slice_seg['slice_{}_seg_{}_epi'.format(slice_nr, seg)]

                                # calculate length between epi and endo point at ed
                                L0 = math.sqrt((ed_epi[0] - ed_endo[0])**2 + (ed_epi[1] - ed_endo[1])**2)
                                # calculate length between epi and endo point at current time step
                                L = math.sqrt((seg_epi[0] - x)**2 + (seg_epi[1] - y)**2)

                                # calculate difference between length at ed and current time step
                                Err = (L - L0)/L0

                                if seg == 0:
                                    strains['Err_slice_{}'.format(slice_nr)] = [Err]
                                # save Err per slice
                                else:
                                    strains['Err_slice_{}'.format(slice_nr)].append(Err)
                                # save to total Err
                                strains['Err_tot'].append(Err)

                            #Ecc
                            if seg != 0:
                                prev_seg_ed = slice_ed['slice_{}_seg_{}_{}'.format(slice_nr, seg - 1, wall_vals[wall])]
                                seg_ed = slice_ed['slice_{}_seg_{}_{}'.format(slice_nr, seg, wall_vals[wall])]

                                prev_seg = slice_seg['slice_{}_seg_{}_{}'.format(slice_nr, seg -1, wall_vals[wall])]

                                # calculate arc length at ED
                                length1 =  math.sqrt(np.dot(prev_seg_ed[0:2],prev_seg_ed[0:2]))
                                length2 =  math.sqrt(np.dot(seg_ed[0:2],seg_ed[0:2]))
                                cos_angle = np.dot(prev_seg_ed[0:2], seg_ed[0:2]) / (length1 * length2)
                                rad_ed = np.arccos(cos_angle)     
                                r = math.sqrt(seg_ed[0]**2 + seg_ed[1]**2)
                                length_ed = (rad_ed/(2*pi) * 2*pi * r)  

                                # calculate arc length
                                length1 =  math.sqrt(np.dot(prev_seg[0:2],prev_seg[0:2]))
                                length2 =  math.sqrt(np.dot([x,y],[x,y]))
                                cos_angle = np.dot(prev_seg[0:2], [x,y]) / (length1 * length2)
                                rad = np.arccos(cos_angle)     
                                r = math.sqrt(x**2 + y**2)
                                length = (rad/(2*pi) * 2*pi * r)  

                                Ecc = (length - length_ed) / length_ed 

                                if seg == 1:
                                    strains['Ecc_slice_{}_{}'.format(slice_nr, wall_vals[wall])] = [Ecc]
                                else:
                                    strains['Ecc_slice_{}_{}'.format(slice_nr, wall_vals[wall])].append(Ecc)                   
                                strains['Ecc_tot'].append(Ecc) 

                            # Ell
                            if slice_nr != 0:
                                prev_slice_ed = np.array(slice_ed['slice_{}_seg_{}_{}'.format(slice_nr -1, seg, wall_vals[wall])])
                                cur_slice_ed = np.array(slice_ed['slice_{}_seg_{}_{}'.format(slice_nr, seg, wall_vals[wall])])

                                prev_slice = np.array(slice_seg['slice_{}_seg_{}_{}'.format(slice_nr -1, seg, wall_vals[wall])])

                                L0 = np.sqrt(np.sum((prev_slice_ed - cur_slice_ed)**2, axis=0))
                                L = np.sqrt(np.sum((prev_slice - np.array([x,y,z]))**2, axis=0))

                                Ell = (L - L0)/L0

                                if seg == 0:
                                    strains['Ell_slice_{}-{}_{}'.format(slice_nr-1, slice_nr, wall_vals[wall])] = [Ell]
                                else:
                                    strains['Ell_slice_{}-{}_{}'.format(slice_nr-1, slice_nr, wall_vals[wall])].append(Ell)
                                strains['Ell_tot'].append(Ell)

                            slice_seg['slice_{}_seg_{}_{}'.format(slice_nr, seg, wall_vals[wall])] = [x,y,z]

                if step != 0:
                    Err_name = 'Err_slice_{}'.format(slice_nr)
                    if step == 1:              
                        time_strain[Err_name] = [np.mean(strains[Err_name])]  
                        time_strain['Err_tot'] = [np.mean(strains[Err_name])] 
                    else:
                        time_strain[Err_name].append(np.mean(strains[Err_name]))       
                        time_strain['Err_tot'].append(np.mean(strains[Err_name]))            
                    for wall in wall_vals:  
                        Ecc_name = 'Ecc_slice_{}_{}'.format(slice_nr, wall)
                        Ell_name = 'Ell_slice_{}-{}_{}'.format(slice_nr-1, slice_nr, wall)
                        if step == 1:              
                            time_strain[Ecc_name] = [np.mean(strains[Ecc_name])]  
                            time_strain['Ecc_{}'.format(wall)] = [np.mean(strains[Ecc_name])]                                  
                        else:
                            time_strain[Ecc_name].append(np.mean(strains[Ecc_name]))
                            time_strain['Ecc_{}'.format(wall)].append(np.mean(strains[Ecc_name]))
                            
                        if slice_nr != 0 and step == 1:
                            time_strain[Ell_name] = [np.mean(strains[Ell_name])] 
                            time_strain['Ell_{}'.format(wall)] = [np.mean(strains[Ell_name])] 
                        elif slice_nr != 0:
                            time_strain[Ell_name].append(np.mean(strains[Ell_name]))
                            time_strain['Ell_{}'.format(wall)].append(np.mean(strains[Ell_name]))
        
        spec = GridSpec(2, 3)
        time = results[par['t_cycle']]

        for strain_count, strain_name in enumerate(['Ell', 'Ecc', 'Err']):
            fig = plt.figure()
            av = fig.add_subplot(spec[1,:])
            epi = fig.add_subplot(spec[0,0])
            mid = fig.add_subplot(spec[0,1], sharey = epi)
            endo = fig.add_subplot(spec[0,2], sharey = epi)

            plot = [epi, mid, endo]
            for slice_nr in range(1, len(theta_vals)):
                for ii, wall in enumerate(wall_vals):
                    Err_name = 'Err_slice_{slicenr}'.format(slicenr = slice_nr)
                    Ecc_name = 'Ecc_slice_{slicenr}_{wall}'.format(slicenr = slice_nr, wall=wall)
                    Ell_name = 'Ell_slice_{slicenr_1}-{slicenr}_{wall}'.format(slicenr_1 = slice_nr-1, slicenr = slice_nr,wall=wall)
                    strain_keys = [Ell_name, Ecc_name, Err_name]

                    plot[ii].plot(time , time_strain[strain_keys[strain_count]], label = 'Slice {}'.format(slice_nr))
                    plot[ii].set_title(wall)

                    if ii == 2:
                        plot[ii].legend(loc='center left', bbox_to_anchor=(1, 0.5))

            av.plot(time, time_strain[strain_keys[strain_count]], label = 'average')
            av.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            fig.suptitle(strain_name)
            fig.savefig(os.path.join(self.directory, strain_name + '_calc.png'), dpi=300, bbox_inches="tight")


        print("\n{:<8} {:>10} {:>12} {:>14}".format('','Ell','Err', 'Ecc'))
        print("{:<8} {:>6} {:>8} {:>9} {:>8} {:>8}".format('','epi','endo', '','epi','endo'))

        for slice_nr, theta in enumerate(theta_vals):
            Err_slice = np.mean(time_strain['Err_slice_{}'.format(slice_nr)])*100
            
            Ecc_slice_epi = np.mean(time_strain['Ecc_slice_{}_epi'.format(slice_nr)])*100
            Ecc_slice_endo = np.mean(time_strain['Ecc_slice_{}_endo'.format(slice_nr)])*100
            
            print("{:<8} {:>7} {:>5} {:=10.2f} {:9.2f} {:8.2f}".format('slice {}'.format(slice_nr),'', '', Err_slice, Ecc_slice_epi, Ecc_slice_endo))

            if theta != (theta_vals[-1]):
                Ell_slice_epi = np.mean(time_strain['Ell_slice_{}-{}_epi'.format(slice_nr, slice_nr+1)]) * 100
                Ell_slice_endo = np.mean(time_strain['Ell_slice_{}-{}_endo'.format(slice_nr, slice_nr+1)]) * 100
                print("{:<8} {:6.2f} {:8.2f}".format('', Ell_slice_epi, Ell_slice_endo))
                
        print("\n{:<8} {:6.2f} {:8.2f} {:=8.2f} {:9.2f} {:8.2f}".format('average', 
                    np.mean(time_strain['Ell_epi']) * 100, np.mean(time_strain['Ell_endo']) * 100, np.mean(time_strain['Err_tot']) * 100,
                    np.mean(time_strain['Ecc_epi']) * 100, np.mean(time_strain['Ecc_endo']) * 100))

    def show_slices(self, fig = None, fontsize=12, title = ''):
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

        gs = GridSpec(2,2)
        # top view slices
        _xy1 = plt.subplot(gs[0,0])
        # side view slices
        _xz1 = plt.subplot(gs[0,1])
        # displacement at ed and es
        _xy2 = plt.subplot(gs[1,:])

        # also create a seperate figure for the displacement at ed and es
        ed_es_fig, ed_es_plot = plt.subplots(1,1)

        self._ax = {'xy1': _xy1, 'xz1': _xz1, 'xy2': _xy2}

        # draw the outlines of the left ventricle
        self.lv_drawing(top_keys = ['xy1'], side_keys = ['xz1'])

        # extract the same theta and phi values as used in the torsion and shear calculations
        theta_vals = self.parameters['theta_vals']
        nr_segments = self.parameters['nr_segments']
        focus = self.parameters['focus']

        eps_outer = self.parameters['eps_outer']
        eps_mid = self.parameters['eps_mid']
        eps_inner = self.parameters['eps_inner']

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
            tau = z_epi/(focus * math.cosh(eps_mid))
            theta_mid = math.acos(tau)
            tau = z_epi/(focus * math.cosh(eps_inner))
            theta_inner = math.acos(tau)

            thetas = [theta, theta_mid, theta_inner]
            for seg, phi in enumerate(phi_range):
                #loop over segments
                seg_es = []
                seg_ed = []

                for ii, wall in enumerate([eps_outer, eps_mid, eps_inner]):
                    # loop over wall locations

                    # get and save coordinates of point
                    x, y, z = ellipsoidal_to_cartesian(focus,wall,thetas[ii],phi)
                    x_segs.append(x)
                    y_segs.append(y)
                    z_segs.append(z)

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

                if slice_nr == 0 or slice_nr == len(theta_vals)-2:
                    # only plot the absolute locations of points of the basel and the second to last slice
                    # connect the epi, mid and endo point of each segment for easy viewing
                    self._ax['xy2'].plot([i[0] for i in seg_ed], [i[1] for i in seg_ed], 'o-', color=col[slice_nr])
                    self._ax['xy2'].plot([i[0] for i in seg_es], [i[1] for i in seg_es], 'x--', color=col[slice_nr])
                    ed_es_plot.plot([i[0] for i in seg_ed], [i[1] for i in seg_ed], 'o-', color=col[slice_nr])
                    ed_es_plot.plot([i[0] for i in seg_es], [i[1] for i in seg_es], 'x--', color=col[slice_nr])

            self._ax['xy1'].scatter(x_segs, y_segs, color=col[slice_nr], label= 'slice ' + str(slice_nr))
            self._ax['xz1'].scatter(x_segs, z_segs, color=col[slice_nr], label= 'slice ' + str(slice_nr))

            if slice_nr == 0 or slice_nr == len(theta_vals)-2:
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

        self._ax['xy2'].axis('equal')
        self._ax['xy2'].set_title('top view (x-y) end diastole and end systole')
        self._ax['xy2'].legend(frameon=False, fontsize=fontsize)

        ed_es_plot.axis('equal')
        ed_es_plot.set_title('top view (x-y) end diastole and end systole')
        ed_es_plot.legend(frameon=False, fontsize=fontsize)

        fig.suptitle(title,fontsize=fontsize+2)
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
        eps_outer = self.parameters['eps_outer']
        eps_mid = self.parameters['eps_mid']
        eps_inner = self.parameters['eps_inner']

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

    def lv_drawing(self, side_keys = None, top_keys = None):
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

        R_1, y, z = ellipsoidal_to_cartesian(self.parameters['focus'],self.parameters['eps_inner'],1/2*math.pi, 0.)
        R_2, y, z = ellipsoidal_to_cartesian(self.parameters['focus'],self.parameters['eps_outer'],1/2*math.pi, 0.)

        x, y, Z1 = ellipsoidal_to_cartesian(self.parameters['focus'],self.parameters['eps_inner'],math.pi, 0.)
        x, y, Z2 = ellipsoidal_to_cartesian(self.parameters['focus'],self.parameters['eps_outer'],math.pi, 0.)

        Z_1 = abs(Z1)
        Z_2 = abs(Z2)

        h = self.parameters['cut_off']
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
# directory_1 = '/home/maaike/Documents/Graduation_project/Results/eikonal_td_1_node/cycle_2_begin_ic_ref'
# directory_1 = '/home/maaike/Documents/Graduation_project/Results/beatit'
# dict_vals = {'theta_vals' : [9/20*math.pi, 11/20*math.pi, 13/20*math.pi, 15/20*math.pi]}
# post_1 = postprocess_hdf5(directory_1, model = 'beatit', **dict_vals)
# # post_1.loc_mech_ker()
# # post_1.show_ker_points()
# post_1.plot_torsion()
# post_1.calc_strain()
# post_1.show_slices()