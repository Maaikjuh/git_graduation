"""
This script demonstrates how two fiber fields can be compared,
by computing the angle between the vectors of the two fiber fields.

Not tested whether this works in parallel.
"""
# ROOT_DIRECTORY = os.path.join(
#         os.path.abspath(__file__).split('\\Graduation_project\\')[0],
#         'Graduation_project')
# CVBTK_PATH = os.path.join(ROOT_DIRECTORY,'git_graduation_project\cvbtk')
# import sys
# sys.path.append(CVBTK_PATH)

# from dataset import Dataset
from cvbtk import read_dict_from_csv, BiventricleGeometry, save_to_disk, save_dict_to_csv, \
     print_once, scalar_space_to_vector_space, vector_space_to_scalar_space
from cvbtk.dataset import Dataset
import os
from dolfin import acos, inner, sqrt, FunctionSpace, project, parameters, pi, conditional, lt, gt
from dolfin import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


def get_values_nodes(mesh,gamma):
    """
    Returns mean difference in angle between
    specified heights 
    """
    V = FunctionSpace(mesh, 'Lagrange', 2)

    dofmap = V.dofmap()
    dofcoors = V.tabulate_dof_coordinates().reshape((-1, 3))
    z = dofcoors[:,2]
    indices = np.where(np.logical_and(z > -2., z < 0.))[0]
    vals = gamma.vector().array()[indices]
    mean_angle = np.mean((vals))
    
    return mean_angle

def get_values_epi_mid_endo(mesh,gamma):
    focus = 4.3

    V = FunctionSpace(mesh, 'Lagrange', 2)
    dofcoors = V.tabulate_dof_coordinates().reshape((-1, 3))
    z = dofcoors[:,2]
    sigma = compute_sigma(dofcoors[:,0], dofcoors[:,1], z, focus)

    # xi = compute_coordinate_expression(3, V.ufl_element(),'xi',focus)
    max_sigma = max(sigma)
    min_sigma = min(sigma)
    mid_sigma = (max_sigma+min_sigma)/2
    # print('min sigma: {}'.format(min_sigma))
    # print('max sigma: {}'.format(max_sigma))
    # print('len sigma: {}, sigma: {}'.format(len(sigma),sigma))

    indices_epi = np.where(np.logical_and.reduce([sigma > max_sigma - 0.01, z > -2., z < 0.]))[0]
    indices_endo = np.where(np.logical_and.reduce([sigma < min_sigma + 0.01, z > -2., z < 0.]))[0]
    indices_mid = np.where(np.logical_and.reduce([sigma < mid_sigma + 0.01, sigma > mid_sigma - 0.01, z > -2., z < 0.]))[0]
    indices = np.where(np.logical_and(z > -2., z < 0.))[0]

    # print('len epi: {}, indices_epi: {}'.format(len(indices_epi),indices_epi))
    # print('len endo: {}, indices_endo: {}'.format(len(indices_endo),indices_endo))
    # print('len mid: {}, indices_mid: {}'.format(len(indices_mid),indices_mid))
    # print('len z: {}, indices_z: {}'.format(len(indices),indices))
    print('percentage of epi nodes analysed in total: {}%'.format(len(indices_epi)/len(indices)*100))
    print('percentage of endo nodes analysed in total: {}%'.format(len(indices_endo)/len(indices)*100))

    vals_epi = gamma.vector().array()[indices_epi]
    vals_mid = gamma.vector().array()[indices_mid]
    vals_endo = gamma.vector().array()[indices_endo]
    vals = gamma.vector().array()[indices]

    mean_angles = { 'total': np.mean((vals)),
                    'epi': np.mean((vals_epi)),
                    'mid': np.mean((vals_mid)),
                    'endo': np.mean((vals_endo))}
    
    return mean_angles

def compute_sigma(x, y, z, focus):
    """
    Ellipsoidal radial position defined such that constant values represent
    an ellipsoidal surface.

    Args:
        x, y, z: x, y, z coordinates.
        focus: focal length of ellipsoid.
    Returns:
        sigma
    """
    # Compute and return the sigma values.
    return 0.5 * (np.sqrt(x ** 2 + y ** 2 + (z + focus) ** 2)
                    + np.sqrt(x ** 2 + y ** 2 + (z - focus) ** 2)) / focus

def norm(v):
    """
    Returns norm of vector v (as UFL expression).
    """
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def compute_angle_between_vectors(v1, v2):
    """
    Returns the angle [radians] between vectors in v1 and v2 (as UFL expression).
    """
    arg = inner(v1, v2) / (norm(v1) * norm(v2))
    # Clip arg to [-1, 1]
    cond_low = conditional(lt(arg, -1), 1, 0)
    cond_high = conditional(gt(arg, 1), 1, 0)
    cond_arg = conditional(lt(cond_low + cond_high, 0.1), 1, 0)
    arg_ = arg*cond_arg + cond_high - cond_low
    # Return acos [radians].
    return acos(arg_)


def radians_to_degrees(angle):
    """
    Converst radians to degrees.
    """
    return angle/pi*180

def plot_angles(angle_dict, cycles,filename = 'reorientation', fontsize = 12, dpi=300, bbox_inches='tight'):
    fig = plt.figure()
    gs = GridSpec(1, 2)
    init = plt.subplot(gs[0,0])
    prev = plt.subplot(gs[0,1])
    ax = {'init': init, 'prev': prev} 
    labels = ['total', 'epi', 'mid','endo']

    ax['init'].set_xlabel('Cycle', fontsize = fontsize)
    ax['prev'].set_xlabel('Cycle', fontsize = fontsize)
    ax['init'].set_ylabel('change in orientation [$^\circ$]', fontsize = fontsize)
    ax['prev'].set_ylabel('change in orientation [$^\circ$]', fontsize = fontsize)

    for i in labels:
        if i == labels[0]:
            ax['init'].plot(cycles, angle_dict['change_init_' + i], '--', label = i)
            ax['prev'].plot(cycles, angle_dict['change_prev_' + i], '--', label = i)
        else:
            ax['init'].plot(cycles, angle_dict['change_init_' + i], label = i)
            ax['prev'].plot(cycles, angle_dict['change_prev_' + i], label = i)

    # ax['init'].tick_params(axis='y', labelcolor = color)
    # ax['prev'].tick_params(axis='y', labelcolor = color)

    # color = 'tab:blue'
    # ax1 = ax['init'].twinx()
    # ax1.set_ylabel('endo [$^\circ$]', fontsize = fontsize, color=color)
    # ax1.plot(cycles, angle_dict['change_init_endo'])

    # ax2 = ax['prev'].twinx()
    # ax2.set_ylabel('endo [$^\circ$]', fontsize = fontsize, color=color)
    # ax2.plot(cycles, angle_dict['change_prev_endo'])

    # ax1.tick_params(axis='y', labelcolor = color)
    # ax2.tick_params(axis='y', labelcolor = color)

    ax['prev'].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=fontsize -2)

    ax['init'].set_title('Change in orientation \n with initial cycle', fontsize = fontsize)
    ax['prev'].set_title('Change in orientation \n with previous cycle', fontsize = fontsize)

    plt.tight_layout()

    fig.savefig(filename, dpi=dpi, bbox_inches = bbox_inches)

def main():
    """
    Compare two fiber fields defined on the same geometry.
    """
    # --------------------------------------------------------------------------- #
    # INPUTS
    # --------------------------------------------------------------------------- #
    # Specify output directory
    dir_out = 'test_fiber_orient_epi_endo'
    directory = '.'

    # Specify directory of simulation output and cycle number to find the HDF5 file
    # with the first fiber field.
    initial = directory #'/home/maaike/model/examples/systemic_circulation/realcycle/output/01-05_10-02_fiber_no_reorientation_meshres_20/'
    cycle_initial = 1
    results_initial = os.path.join(initial, 'results_cycle_{}.hdf5'.format(cycle_initial))

    # Read inputs (we need the geometry inputs).
    inputs = read_dict_from_csv(os.path.join(initial, 'inputs.csv'))

    # Set the proper global FEniCS parameters.
    parameters.update({'form_compiler': inputs['form_compiler']})

    # Load geometry and first fiber field.
    print_once('Loading initial...')
    inputs['geometry']['load_fiber_field_from_meshfile'] = True
    geometry = BiventricleGeometry(meshfile=results_initial, **inputs['geometry'])

    # Extract first fiber vector.
    ef_init = geometry.fiber_vectors()[0].to_function(None)

    # Create scalar function space for the difference in angle between vectors.
    Q = FunctionSpace(geometry.mesh(), 'Lagrange', 2)

    # Create vector function space.
    V = scalar_space_to_vector_space(Q)

    dir_orientation = directory #"/home/maaike/model/examples/systemic_circulation/realcycle/output/01-05_08-38_fiber_reorientation_meshres_20/"
    results_csv = os.path.join(dir_orientation, 'results.csv') 
    full_csv = Dataset(filename=results_csv)
    first_cycle = int(min(full_csv['cycle']))
    last_cycle = int(max(full_csv['cycle']) - 1)

    # angle_diff_init_epi = []
    # angle_diff_init_endo = []
    # angle_diff_prev_epi = []
    # angle_diff_prev_endo = []
    angle_dict = {  'change_init_total': [],
                    'change_init_epi':[],
                    'change_init_mid': [],
                    'change_init_endo': [],
                    'change_prev_total': [],
                    'change_prev_epi': [],
                    'change_prev_mid': [],
                    'change_prev_endo': []}

    for i in range(first_cycle,last_cycle+1):
        result_cyc = os.path.join(dir_orientation, 'results_cycle_{}.hdf5'.format(i))

        # Load field of the cycle.
        print_once('Loading fiberfield cycle {}...'.format(i))
        geometry._fiber_vectors = None  # This may be a redundant statement (but I did not check if it works without).
        geometry.load_fiber_field(filepath=result_cyc)

        # Extract second fiber field.
        ef_cyc = geometry.fiber_vectors()[0].to_function(None)

        # Compute angles between vectors in first and second fiber field.
        print_once('Computing angles between init - cyc {}...'.format(i))
        gamma = compute_angle_between_vectors(ef_init, ef_cyc)  # [radians]

        save_to_disk(project(radians_to_degrees(gamma), Q), os.path.join(dir_out, 'gamma.xdmf'))

        # mean_angle = get_values_nodes(geometry.mesh(),project(radians_to_degrees(gamma), Q))
        mean_angle = get_values_epi_mid_endo(geometry.mesh(),project(radians_to_degrees(gamma), Q))

        # angle_diff_init_epi.append(mean_angle['epi'])
        # angle_diff_init_endo.append(mean_angle['endo'])

        angle_dict['change_init_total'].append(mean_angle['total'])
        angle_dict['change_init_epi'].append(mean_angle['epi'])
        angle_dict['change_init_mid'].append(mean_angle['mid'])
        angle_dict['change_init_endo'].append(mean_angle['endo'])

        if i == 1:
            ef_prev = ef_cyc
            # angle_diff_prev.append(0.)
            angle_dict['change_prev_total'].append(mean_angle['total'])
            angle_dict['change_prev_epi'].append(mean_angle['epi'])
            angle_dict['change_prev_mid'].append(mean_angle['mid'])
            angle_dict['change_prev_endo'].append(mean_angle['endo'])
        else:
            if cycle_initial != first_cycle and i == first_cycle:
                ef_prev = ef_init
            # Compute angles between vectors in first and second fiber field.
            print_once('Computing angles between cyc {} - cyc {}...'.format(i-1,i))
            gamma = compute_angle_between_vectors(ef_prev, ef_cyc)  # [radians]
            # mean_angle = get_values_nodes(geometry.mesh(),project(radians_to_degrees(gamma), Q))
            mean_angle = get_values_epi_mid_endo(geometry.mesh(),project(radians_to_degrees(gamma), Q))
            # angle_diff_prev.append(mean_angle)
            # angle_diff_prev_epi.append(mean_angle['epi'])
            # angle_diff_prev_endo.append(mean_angle['endo'])

            angle_dict['change_prev_total'].append(mean_angle['total'])
            angle_dict['change_prev_epi'].append(mean_angle['epi'])
            angle_dict['change_prev_mid'].append(mean_angle['mid'])
            angle_dict['change_prev_endo'].append(mean_angle['endo'])
            ef_prev = ef_cyc

    # angle_dict = {  'change_init':angle_diff_init,
    #                 'change_prev': angle_diff_prev}

    # angle_dict = {  'change_init_epi':angle_diff_init_epi,
    #                 'change_init_endo': angle_diff_init_endo,
    #                 'change_prev_epi': angle_diff_prev_epi,
    #                 'change_prev_endo': angle_diff_prev_endo}

    save_dict_to_csv(angle_dict, os.path.join(dir_orientation, os.path.join(dir_out,'reorientation_angle.csv')))

    save_to_disk(project(radians_to_degrees(gamma), Q), os.path.join(dir_out, 'change_fiber_vector.xdmf'))
    save_to_disk(project(ef_init, V), os.path.join(dir_out, 'ef_init.xdmf'))
    save_to_disk(project(ef_cyc, V), os.path.join(dir_out, 'ef_cycle_{}.xdmf'.format(i)))
    
    plot_angles(angle_dict, range(first_cycle,last_cycle+1), filename = os.path.join(dir_out,'reorientation'))


    # Project last field with initial (they are quadrature functions or UFL expressions,
    # so we need to project them in order to visualize them).
    print_once('Projecting...')



if __name__ == '__main__':
    main()
