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
     print_once, scalar_space_to_vector_space
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

def plot_angles(angles_init, angles_prev, cycles,filename = 'reorientation', fontsize = 12, dpi=300, bbox_inches='tight'):
    fig = plt.figure()
    gs = GridSpec(1, 2)
    init = plt.subplot(gs[0,0])
    prev = plt.subplot(gs[0,1])
    ax = {'init': init, 'prev': prev} 
    ax['init'].set_xlabel('Cycle', fontsize = fontsize)
    ax['prev'].set_xlabel('Cycle', fontsize = fontsize)
    ax['init'].set_ylabel('orientation change [$^\circ$]', fontsize = fontsize)
    ax['prev'].set_ylabel('orientation change [$^\circ$]', fontsize = fontsize)

    ax['init'].plot(cycles, angles_init)
    ax['prev'].plot(cycles, angles_prev)

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
    dir_out = 'post_fiber_reorientation_old_vs_new'
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

    angle_diff_init = []
    angle_diff_prev = []

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

        mean_angle = get_values_nodes(geometry.mesh(),project(radians_to_degrees(gamma), Q))

        angle_diff_init.append(mean_angle)

        if i == 1:
            ef_prev = ef_cyc
            angle_diff_prev.append(0.)
        else:
            # Compute angles between vectors in first and second fiber field.
            print_once('Computing angles between cyc {} - cyc {}...'.format(i-1,i))
            gamma = compute_angle_between_vectors(ef_prev, ef_cyc)  # [radians]
            mean_angle = get_values_nodes(geometry.mesh(),project(radians_to_degrees(gamma), Q))
            angle_diff_prev.append(mean_angle)
            ef_prev = ef_cyc

    angle_dict = {  'change_init':angle_diff_init,
                    'change_prev': angle_diff_prev}

    save_dict_to_csv(angle_dict, os.path.join(dir_orientation, os.path.join(dir_out,'reorientation_angle.csv')))

    save_to_disk(project(radians_to_degrees(gamma), Q), os.path.join(dir_out, 'change_fiber_vector.xdmf'))
    save_to_disk(project(ef_init, V), os.path.join(dir_out, 'ef_init.xdmf'))
    save_to_disk(project(ef_cyc, V), os.path.join(dir_out, 'ef_cycle_{}.xdmf'.format(i)))
    
    plot_angles(angle_diff_init,angle_diff_prev, range(1,last_cycle+1), filename = os.path.join(dir_out,'reorientation'))


    # Project last field with initial (they are quadrature functions or UFL expressions,
    # so we need to project them in order to visualize them).
    print_once('Projecting...')



if __name__ == '__main__':
    main()
