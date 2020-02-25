"""
This script simulates the LifeTEc experiment using
FEniCS with Python, with some minor post-processing.

LV is modelled with a FE model, the systemic circulation is modelled as a
windkessel model. Additionally, its possible to add an LVAD to the
systemic circulation.
"""

from dolfin.cpp.common import MPI, mpi_comm_world

from cvbtk import (HemodynamicsPlot, VolumeSolver, print_once, save_dict_to_csv,
                   CustomNewtonSolver, mmHg_to_kPa, simulate, preprocess_lv, ReloadState, read_dict_from_csv)

import os
import matplotlib.pyplot as plt


# Change the number of cycles and active stress ('new' or 'old') here:
NUM_CYCLES = 5
ACT_STRESS = 'old'  # 'old' is Arts Kerckhoffs, 'new' is Arts Bovendeerd.

# Use the following options if you want to reload a saved model state and continue
# the simulation from that state (e.g. handy when a simulation crashed).
# See ReloadState.reload() for a detailed description of these options.
# Set both options to None if you don't want to reload.
DIR_RELOAD = None  # Directory with output files of the model to reload.
TIME_RELOAD = None  # The time (in ms) of the timestep to reload. Set to -1 for reloading the latest available timestep.

# Use the following option if you want to load a set of inputs and start a new simulation using those inputs.
# By specifying a path to an inputs.csv file, you can load the inputs from the file
# instead of defining them in get_inputs(). If you do not want to load inputs from a
# file and you want to define the inputs in get_inputs(), set the below path to None.
INPUTS_PATH = 'inputs.csv'

# Specify output directory.
DIR_OUT = 'output/LVAD/lifetec'

# Create directory if it doesn't exists.
if MPI.rank(mpi_comm_world()) == 0:
    if not os.path.exists(DIR_OUT):
        os.makedirs(DIR_OUT)
# Synchronize.
MPI.barrier(mpi_comm_world())


def get_inputs(number_of_cycles, active_stress):
    """
    Helper function to return a dictionary of inputs in one convenient location.

    The ``active_stress`` keyword specifies either the "old" (ArtsKerckhoffs) or
    "new" (ArtsBovendeerd) ActiveStress models.
    """
    # -------------------------------------------------------------------------- #
    # Global settings.                                                           #
    # -------------------------------------------------------------------------- #
    time = {'dt': 2.0,
            't0': 0.0,
            't1': 20000.,  # Maximum simulation time.
            'tc': 500.0}  # Maximum time for one cycle.

    # -------------------------------------------------------------------------- #
    # SYSTEMIC circulation: create a dictionary of inputs for WindkesselModel.   #
    # -------------------------------------------------------------------------- #
    # Specify type of windkessel ('WindkesselModel' (closed-loop) or
    # 'LifetecWindkesselModel' (open-loop))
    windkessel_type = 'LifetecWindkesselModel'
    windkessel_model = {'arterial_compliance':  5.02678762781376,  # [ml/kPa]
                        'arterial_resistance':  2.43592903973729,  # [kPa.ms/ml]
                        'peripheral_resistance':  119.360522947127,  # [kPa.ms/ml]
                        'venous_resistance': 1.0,  # [kPa.ms/ml]
                        'venous_pressure': mmHg_to_kPa(10)}  # [kPa]

    # -------------------------------------------------------------------------- #
    # LVAD:  create a dictionary of inputs for HeartMateII.                      #
    # -------------------------------------------------------------------------- #
    # Specify whether to add an LVAD to the windkessel model or not
    # by setting attach_lvad to True or False.
    attach_lvad = True
    lvad_model = {'frequency': float(9.5), # [krpm]
                   'lvad_volume': 66.0,
                   'alpha_slope': 0.00435981,
                   'alpha_intercept': 0.99412831,
                   'beta_slope': -0.0971561,
                   'beta_intercept': -3.34863522}

    # -------------------------------------------------------------------------- #
    # LV: create a dictionary of inputs for LeftVentricle geometry.              #
    # -------------------------------------------------------------------------- #
    # We do not need to set all of the geometry parameters
    # as we will load a reference mesh.
    # Specify reference mesh to load and resolution.
    geometry_type = 'reference_left_ventricle_pluijmert'
    geometry = {'mesh_resolution': 30.0}

    # If we do not set any fiber field parameters in geometry['fiber_field'],
    # the default (rule-based 2009) fiber field will be used.

    # NOTE that some previous options may have no effect when loading the fiber field from the meshfile.
    geometry['load_fiber_field_from_meshfile'] = True

    # -------------------------------------------------------------------------- #
    # LV: create a dictionary of inputs for LeftVentricle model.                 #
    # -------------------------------------------------------------------------- #
    # Fiber reorientation.
    # If you do not want to perform fiber reorientation, simply set ncycles_reorient to 0.
    ncycles_reorient = 0
    fiber_reorientation = {'kappa': 4.0 * time['tc'],  # Time constant of fiber reorientation model.
                           'ncycles_pre': 5,  # Number of cardiac cycles before enabling reorientation.
                           'ncycles_reorient': ncycles_reorient}  # Total number of cardiac cycles with reorientation.

    model = {'fiber_reorientation': fiber_reorientation}

    # Passive material law.
    material_law = 'KerckhoffsMaterial'
    material_model = {'a0': 0.5,
                      'a1': 3.0,
                      'a2': 6.0,
                      'a3': 0.01,
                      'a4': 60.0,
                      'a5': 55.0}

    # Active stress model.
    active_stress_ls0 = 1.9
    active_stress_beta = 0.0
    active_stress_tdep = 3*time['dt']
    # NOTE: tdep is actually the reset time: active stress is assumed zero and
    # state variables are reset when t_act exceeds tcycle-tdep.

    active_stress_arts_kerckhoffs = {'T0': 92.0,
                                     'Ea': 20.0,
                                     'al': 2.0,
                                     'lc0': 1.5,
                                     'taur': 90.,
                                     'taud': 90.,
                                     'b': 150.0,
                                     'ld': -0.4,
                                     'v0': 0.0075,
                                     'ls0': active_stress_ls0,
                                     'beta': active_stress_beta,
                                     'tdep': active_stress_tdep,
                                     'restrict_lc': True}

    active_stress_arts_bovendeerd = {'T0': 160.0,  # pg. 66: 250 kPa
                                     'ar': 100.0,
                                     'ad': 400.0,
                                     'ca': 1.2,
                                     'cv': 1.0,
                                     'lsa0': 1.5,
                                     'lsa1': 2.0,
                                     'taur1': 140.0,
                                     'taud1': 240.0,
                                     'v0': 0.01,
                                     'ls0': active_stress_ls0,
                                     'beta': active_stress_beta,
                                     'tdep': active_stress_tdep}

    # -------------------------------------------------------------------------- #
    # Initial conditions.                                                        #
    # -------------------------------------------------------------------------- #
    # Specify extra time before first depolarization (in addition to tdep)
    # to ensure sufficient initial filling.
    filling_time = 74.0

    # Initial state.
    initial_phase = {'lv': 1}
    state = {'cycle': 1,
             'phase': initial_phase,
             't_cycle': time['t0'],
             't_active_stress': time['t0'] - filling_time}

    # Initial conditions (specify pressures [in kPa]).
    initial_conditions = {'arterial_pressure': 11.5}

    # Note that venous pressure of the systemic circulation will be calculated from
    # venous volume, which will be calculated from mass conservation.

    # -------------------------------------------------------------------------- #
    # Numerical settings.                                                        #
    # -------------------------------------------------------------------------- #
    # Compiler settings.
    form_compiler = {'quadrature_degree': 4,
                     'cpp_optimize_flags': '-O3 -march=native -mtune=native'}

    # Newton solver settings (computing the displacement for a given pressure).
    newton_solver = {'maximum_iterations': 50,
                     'absolute_tolerance': 1e-3,
                     'linear_solver': 'bicgstab',
                     'preconditioner': 'hypre_euclid',
                     'error_on_nonconvergence': True,
                     'krylov_solver': {'absolute_tolerance': 1e-7}}

    # Volume solver settings (matching cavity volume with expected volumes).
    volume_solver = {'maximum_iterations': 10,
                     'absolute_tolerance': 1e-2,
                     'newton_solver': newton_solver}

    # Combine and return all input dictionaries.
    inputs = {'geometry': geometry,
              'geometry_type': geometry_type,
              'model': model,
              'material_model': material_model,
              'material_law': material_law,
              'windkessel_model': windkessel_model,
              'windkessel_type': windkessel_type,
              'attach_lvad': attach_lvad,
              'lvad_model': lvad_model,
              'form_compiler': form_compiler,
              'state': state,
              'initial_conditions': initial_conditions,
              'time': time,
              'volume_solver': volume_solver,
              'number_of_cycles': number_of_cycles}

    # Add the proper active stress parameters:
    if active_stress == 'old':
        inputs['active_stress'] = active_stress_arts_kerckhoffs
        inputs['active_stress_model'] = 'ArtsKerckhoffsActiveStress'
    elif active_stress == 'new':
        inputs['active_stress'] = active_stress_arts_bovendeerd
        inputs['active_stress_model'] = 'ArtsBovendeerdActiveStress'

    return inputs


def postprocess(results, dir_out='output'):
    """
    Postprocessing routine.
    Plots hemodynamics.

    Args:
        results: cvbtk.Dataset with results.
        dir_out (optional, str): output directory.
    """
    # Plot the results.
    plot = HemodynamicsPlot(results)
    plot.plot()
    plot.save(os.path.join(dir_out, 'results.png'))

    plot.plot_function()
    plt.savefig(os.path.join(dir_out, 'lv_function.png'))

    # Save the CSV data file.
    results.save(os.path.join(dir_out, 'results.csv'))


def main():
    # Check whether we want to reload a model, or create a new one.
    if TIME_RELOAD is not None and DIR_RELOAD is not None:
        # Reload.
        if os.path.exists(DIR_RELOAD):
            # Optionally specify input parameters to overrule loaded inputs.
            overrule_inputs = {}  # Use an empty dictionary if you do not want to overrule loaded inputs.
            wk, lv, results, inputs = ReloadState().reload(DIR_RELOAD, TIME_RELOAD, **overrule_inputs)
        else:
            raise FileNotFoundError('Path "{}" specified by DIR_RELOAD does not exist.'
                                    .format(DIR_RELOAD))
    else:
        # Create a new model.
        print_once('Creating a new model...')

        # Check if we want to load the inputs from a file or create inputs using get_inputs().
        if INPUTS_PATH is None:
            # Use the inputs specified in get_inputs().
            inputs = get_inputs(NUM_CYCLES, ACT_STRESS)
        else:
            # Load inputs from specified file.
            if type(INPUTS_PATH) == str:
                inputs = read_dict_from_csv(INPUTS_PATH)
            else:
                raise TypeError('Expected a string or None for INPUTS_PATH. Instead a {} was given.'
                                .format(type(INPUTS_PATH)))

        # Define and set up the model.
        wk, lv, results = preprocess_lv(inputs)

    # Save inputs to a csv file (as we might later wonder what inputs we have used).
    print_once('Saving inputs to {} ...'.format(os.path.join(DIR_OUT, 'inputs.csv')))
    save_dict_to_csv(inputs, os.path.join(DIR_OUT, 'inputs.csv'))

    # Setup a VolumeSolver with a custom NewtonSolver (optional).
    # Here, a custom Newton solver is used which detects divergence and saves residuals.
    solver = VolumeSolver(custom_newton_solver=CustomNewtonSolver(model=lv, dir_out=DIR_OUT),
                          **inputs['volume_solver'])

    # Call the simulation routine.
    # This try/except clause is a fail safe to write data if it crashes.
    try:
        simulate(wk, lv, results, inputs, 'LeftVentricle', solver=solver, dir_out=DIR_OUT)

    except RuntimeError as error_detail:
        print_once('Except RuntimeError: {}'.format(error_detail))
        if MPI.rank(mpi_comm_world()) == 0:
            print('Simulation failed! Saving the CSV file.')

    # Call the post-processing routine.
    postprocess(results, dir_out=DIR_OUT)


if __name__ == '__main__':
    main()