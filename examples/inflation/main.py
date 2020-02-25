"""
This scripts demonstrates how to do a simple passive inflation of the LV.
"""

from dolfin.cpp.common import MPI, mpi_comm_world
from dolfin import parameters
from dolfin.cpp.la import NewtonSolver

import cvbtk
from cvbtk import mmHg_to_kPa, Dataset, print_once, info_once, create_model, set_boundary_conditions, create_materials, \
    kPa_to_mmHg

import os

# Output directory
from cvbtk.routines import save_model_state_to_hdf5


# Specify output directory.
DIR_OUT = 'output/systemic_circulation/inflation/inflate_0_120_mmHg'

# Create output directory if it does not exists.
if MPI.rank(mpi_comm_world()) == 0:
    if not os.path.exists(DIR_OUT):
        os.makedirs(DIR_OUT)
# Synchronize.
MPI.barrier(mpi_comm_world())


def get_inputs():
    """
    Helper function to return a dictionary of inputs in one convenient location.
    """

    pressure = {'dp': mmHg_to_kPa(0.1), # Pressure increment.
                'p0': mmHg_to_kPa(0),   # Initial pressure.
                'p1': mmHg_to_kPa(20)}  # Maximum pressure.

    # We do not need to set all of the geometry parameters
    # as we will load a reference mesh.
    geometry = {'mesh_resolution': 30.0}

    geometry_type = 'reference_left_ventricle_pluijmert'

    material_law = 'KerckhoffsMaterial'
    material_model = {'a0': 0.5,
                      'a1': 3.0,
                      'a2': 6.0,
                      'a3': 0.01,
                      'a4': 60.0,
                      'a5': 55.0}

    # We don't need active stress.
    # Have to explicitly define this, otherwise it will load the default active stress model.
    active_stress_model = None

    form_compiler = {'quadrature_degree': 4,
                     'cpp_optimize_flags': '-O3 -march=native -mtune=native'}

    newton_solver = {'maximum_iterations': 50,
                     'absolute_tolerance': 1e-3,
                     'linear_solver': 'bicgstab',
                     'preconditioner': 'hypre_euclid',
                     'error_on_nonconvergence': True,
                     'krylov_solver': {'absolute_tolerance': 1e-7}}

    # Combine and return all input dictionaries.
    inputs = {'geometry': geometry,
              'geometry_type': geometry_type,
              'material_model': material_model,
              'material_law': material_law,
              'active_stress_model': active_stress_model,
              'form_compiler': form_compiler,
              'newton_solver': newton_solver,
              'pressure': pressure}

    return inputs


def preprocess(inputs):
    """
    Pre-processing routine for leftventricle inflation.

    Args:
        inputs(dict): Simulation inputs.

    Returns:
        lv (cvbtk Model): Initialized LeftVentricleModel.
        results (cvbtk.Dataset): Dataset for the results.
    """
    # ------------------------------------------------------------------------ #
    # Create a dataset container to store state values.                        #
    # ------------------------------------------------------------------------ #
    # Populate it with the same keys (columns) as the Sepran data.
    dataset_keys = ['plv', 'vlv', 'vector_number']

    # Create the dataset.
    results = Dataset(keys=dataset_keys)

    # ------------------------------------------------------------------------ #
    # Set the proper global FEniCS parameters.                                 #
    # ------------------------------------------------------------------------ #
    parameters.update({'form_compiler': inputs['form_compiler']})

    # ------------------------------------------------------------------------ #
    # Create the finite element model for the left ventricle.                  #
    # ------------------------------------------------------------------------ #
    # For the geometry we re-use the reference mesh.
    res = inputs['geometry']['mesh_resolution']

    # Reference mesh name is specified in inputs. Else default is 'reference_left_ventricle_pluijmert'.
    geometry_type = inputs.get('geometry_type', 'reference_left_ventricle_pluijmert')
    print_once('Loading geometry type "{}"...'.format(geometry_type))

    if geometry_type == 'reference_left_ventricle_pluijmert':
        geometry = cvbtk.resources.reference_left_ventricle_pluijmert(resolution=res, **inputs['geometry'])
    elif geometry_type == 'reference_left_ventricle':
        geometry = cvbtk.resources.reference_left_ventricle(resolution=res, **inputs['geometry'])
    else:
        raise ValueError('Unknwon geometry type.')

    # Check parameters.
    info_once(geometry.parameters, True)

    # Create model and set boundary conditions.
    lv = create_model(geometry, inputs, 'LeftVentricle')
    set_boundary_conditions(lv)

    # Add material laws to the model.
    create_materials(lv, inputs)

    # ------------------------------------------------------------------------ #
    # Set the initial conditions.                                              #
    # ------------------------------------------------------------------------ #
    lv.pressure = {'lv': 0.0}  # Redundant
    lv.volume = {'lv': lv.compute_volume()}

    return lv, results


def simulate(lv, results, inputs, dir_out='.'):

    # Create a Newton Solver.
    newton_solver = NewtonSolver()
    newton_solver.parameters.update(inputs['newton_solver'])

    # Save mesh to output HDF5.
    hdf5_filename = os.path.join(dir_out, 'results.hdf5')
    vector_number = save_model_state_to_hdf5(lv, hdf5_filename, 0, new=True, save_fiber_vector=False)

    # Save step hemodynamics.
    results.append(plv=lv.pressure['lv'],
                   vlv=lv.volume['lv'],
                   vector_number=vector_number)

    # Save the CSV data file.
    if MPI.rank(mpi_comm_world()) == 0:
        results.save(os.path.join(dir_out, 'results.csv'))

    # Extract to be prescribed pressure inputs.
    pressure = inputs['pressure']
    p = pressure['p0']
    dp = pressure['dp']
    p1 = pressure['p1']

    # Loop over pressures.
    i = 0
    while p + dp < p1:

        # Increment pressure.
        p += dp
        i += 1

        print_once('Pressure: {} mmHg...'.format(kPa_to_mmHg(p)))

        # Set boundary pressure of model.
        lv.boundary_pressure = float(p)

        # Solve for displacement using the newton solver.
        converged = newton_solver.solve(lv.problem, lv.u.vector())[1]

        # Newton solver did not fail, but reached maximum number of iterations.
        if not converged:
            print_once('Maximum number of newton iterations reached before convergence!')

        # Compute new LV volume.
        lv.volume = {'lv': lv.compute_volume()}

        # Save displacement to HDF5.
        vector_number = save_model_state_to_hdf5(lv, hdf5_filename, i, new=False, save_fiber_vector=False)

        # Save step hemodynamics.
        results.append(plv=lv.pressure['lv'],
                       vlv=lv.volume['lv'],
                       vector_number=vector_number)

        # Save the CSV data file.
        if MPI.rank(mpi_comm_world()) == 0:
            results.save(os.path.join(dir_out, 'results.csv'))

    print_once('Done!')


def main():
    # Create inputs.
    inputs = get_inputs()

    # Create lv model and results dataset.
    lv, results = preprocess(inputs)

    # Run the simulation.
    simulate(lv, results, inputs, dir_out=DIR_OUT)


if __name__ == '__main__':
    main()