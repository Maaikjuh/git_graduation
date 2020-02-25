"""
This script demonstrates how two fiber fields can be compared,
by computing the angle between the vectors of the two fiber fields.

Not tested whether this works in parallel.
"""

from cvbtk import read_dict_from_csv, BiventricleGeometry, save_to_disk, \
     print_once, scalar_space_to_vector_space
import os
from dolfin import acos, inner, sqrt, FunctionSpace, project, parameters, pi, conditional, lt, gt


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


def main():
    """
    Compare two fiber fields defined on the same geometry.
    """
    # --------------------------------------------------------------------------- #
    # INPUTS
    # --------------------------------------------------------------------------- #
    # Specify directory of simulation output and cycle number to find the HDF5 file
    # with the first fiber field.
    dir_1 = 'output/biv_realcycle/reorientation/REF/new_mesh_reorientation_3'
    cycle_1 = 6
    results_1 = os.path.join(dir_1, 'results_cycle_{}.hdf5'.format(cycle_1))

    # Specify directory of simulation output and cycle number to find the HDF5 file
    # with the second fiber field.
    dir_2 = 'output/biv_realcycle/reorientation/REF/new_mesh_reorientation_4'
    cycle_2 = 34
    results_2 = os.path.join(dir_2, 'results_cycle_{}.hdf5'.format(cycle_2))

    # Specify output directory.
    dir_out = 'output/post_fiber_reorientation_old_vs_new'

    # --------------------------------------------------------------------------- #

    # Read inputs (we need the geometry inputs).
    inputs = read_dict_from_csv(os.path.join(dir_1, 'inputs.csv'))

    # Set the proper global FEniCS parameters.
    parameters.update({'form_compiler': inputs['form_compiler']})

    # Load geometry and first fiber field.
    print_once('Loading 1...')
    inputs['geometry']['load_fiber_field_from_meshfile'] = True
    geometry = BiventricleGeometry(meshfile=results_1, **inputs['geometry'])

    # Extract first fiber vector.
    ef_1 = geometry.fiber_vectors()[0].to_function(None)

    # Load second fiber field.
    print_once('Loading 2...')
    geometry._fiber_vectors = None  # This may be a redundant statement (but I did not check if it works without).
    geometry.load_fiber_field(filepath=results_2)

    # Extract second fiber field.
    ef_2 = geometry.fiber_vectors()[0].to_function(None)

    # Create scalar function space for the difference in angle between vectors.
    Q = FunctionSpace(geometry.mesh(), 'Lagrange', 2)

    # Create vector function space.
    V = scalar_space_to_vector_space(Q)

    # Compute angles between vectors in first and second fiber field.
    print_once('Computing angles...')
    gamma = compute_angle_between_vectors(ef_1, ef_2)  # [radians]

    # Project all fields (they are quadrature functions or UFL expressions,
    # so we need to project them in order to visualize them).
    print_once('Projecting...')
    save_to_disk(project(radians_to_degrees(gamma), Q), os.path.join(dir_out, 'change_fiber_vector.xdmf'))
    save_to_disk(project(ef_1, V), os.path.join(dir_out, 'ef_1.xdmf'))
    save_to_disk(project(ef_2, V), os.path.join(dir_out, 'ef_2.xdmf'))


if __name__ == '__main__':
    main()
