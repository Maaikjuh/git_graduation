"""
This script provides an example on how to simulate multiple cardiac cycles using
FEniCS with Python, with some minor post-processing.

The biventricle is modelled with a FE model, the systemic and pulmonary circulation
are modelled with windkessel models. Additionally, its possible to add an LVAD to the
systemic circulation.
"""

from dolfin.cpp.common import MPI, mpi_comm_world

from cvbtk import save_dict_to_csv, HemodynamicsPlotDC, \
    VolumeSolverBiV, print_once, CustomNewtonSolver, \
    simulate, preprocess_biv, ReloadState, read_dict_from_csv

import os


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
INPUTS_PATH = None #'inputs.csv'

INFARCT = False

# Specify output directory.
DIR_OUT = 'output/LVAD/biventricle'

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
    # Time.
    time = {'dt': 2.0,
            't0': 0.0,
            't1': 50000.,  # Maximum simulation time.
            'tc': 800.0}  # Maximum time for one cycle.

    # Total blood volume.
    total_volume = 5000.0

    # -------------------------------------------------------------------------- #
    # Infarct: create a dictionary of inputs for the infarct geometry.           #
    # -------------------------------------------------------------------------- #
    if INFARCT == True:
        infarct_prm = { 'phi_min': 0.,
                        'phi_max': 1.5708,
                        'theta_min': 1.5708,
                        'theta_max': 3.1416,
                        'ximin': 0., #0.5,
                        'focus': 4.3,
                        'Ta0_infarct': 20., #20.,
                        'save_T0_mesh': DIR_OUT}
    else:
        infarct_prm = None

    # Windkessel inputs from Pluijmert et al. (2017) are used.
    # -------------------------------------------------------------------------- #
    # SYSTEMIC circulation: create a dictionary of inputs for WindkesselModel.   #
    # -------------------------------------------------------------------------- #
    wk_sys = {'arterial_compliance': 15.3,  # [ml/kPa]
              'arterial_resistance': 4.46,  # [kPa.ms/ml]
              'arterial_resting_volume': 704.,  # [ml]
              'peripheral_resistance': 149.,  # [kPa.ms/ml]
              'venous_compliance': 45.9,  # [ml/kPa]
              'venous_resistance': 1.10,  # [kPa.ms/ml]
              'venous_resting_volume': 3160.}  # [ml]

    # -------------------------------------------------------------------------- #
    # PULMONARY circulation:  create a dictionary of inputs for WindkesselModel. #
    # -------------------------------------------------------------------------- #
    wk_pul = {'arterial_compliance': 45.9,  # [ml/kPa]
              'arterial_resistance': 2.48,  # [kPa.ms/ml]
              'arterial_resting_volume': 78.3,  # [ml]
              'peripheral_resistance': 17.8,  # [kPa.ms/ml]
              'venous_compliance': 15.3,  # [ml/kPa]
              'venous_resistance': 2.18,  # [kPa.ms/ml]
              'venous_resting_volume': 513.}  # [ml]

    # -------------------------------------------------------------------------- #
    # LVAD:  create a dictionary of inputs for HeartMateII.                      #
    # -------------------------------------------------------------------------- #
    # Specify whether to add an LVAD to the windkessel model or not
    # by setting attach_lvad to True or False.
    attach_lvad = False
    lvad_model = {'frequency': float(9.5), # [krpm]
                   'lvad_volume': 66.0,
                   'alpha_slope': 0.0091,
                   'alpha_intercept': 1.4,
                   'beta_slope': -0.19,
                   'beta_intercept': -1.9}

    # -------------------------------------------------------------------------- #
    # BiV: create a dictionary of inputs for Biventricle geometry.               #
    # -------------------------------------------------------------------------- #
    # Geometry. Specify resolution.
    geometry = {'mesh_resolution': 43.0  # (float)
                }

    # Specify parameters passable to the Geometry class (e.g. Bayer and fiber field pars).
    # Note that if the fiber field is loaded, these fiber parameters have no effect.
    bayer_prm = {
        'apex_bc_mode': 'line',
        'interp_mode': 3,
        'retain_mode': 'combi',
        'correct_ec': True,
        'linearize_u': True,
        'mirror': True,
        'transmural_coordinate_mode': 2}

    geometry['bayer'] = bayer_prm

    # Fiber field.
    ## Uncomment this part to get a fiber field without transverse angle:
    # fiber_prm = {'t11': 0.,
    #              't12': 0.,
    #              't21': 0.,
    #              't23': 0.,
    #              't25': 0.}
    # geometry['fiber_field'] = fiber_prm

    # NOTE that some previous options may have no effect when loading the fiber field from the meshfile.
    geometry['load_fiber_field_from_meshfile'] = True

    # -------------------------------------------------------------------------- #
    # BiV: create a dictionary of inputs for Biventricle model.                  #
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

    active_stress_arts_kerckhoffs = {'Ta0': 160.0,
                                     'Ea': 20.0,
                                     'al': 2.0,
                                     'lc0': 1.5,
                                     'taur': 75.0,
                                     'taud': 150.0,
                                     'b': 160.0,
                                     'ld': -0.5,
                                     'v0': 0.0075,
                                     'ls0': active_stress_ls0,
                                     'beta': active_stress_beta,
                                     'tdep': active_stress_tdep,
                                     'restrict_lc': True}

    active_stress_arts_bovendeerd = {'Ta0': 160.0,
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
    filling_time = 294.0

    # Initial state.
    initial_phase = {'lv': 1,
                     'rv': 1}
    state = {'cycle': 1,
             'phase': initial_phase,
             't_cycle': time['t0'],
             't_active_stress': time['t0'] - filling_time}

    # Initial conditions (specify pressures [in kPa]).
    initial_conditions = {'p_art_sys': 13.5,
                          'p_art_pul': 3.5,
                          'p_ven_pul': 1.}

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
                     'convergence_criterion': 'residual',
                     'error_on_nonconvergence': True,
                     'krylov_solver': {'absolute_tolerance': 1e-7,
                                       'error_on_nonconvergence': True}}

    # Volume solver settings (matching cavity volumes with expected volumes).
    volume_solver = {'maximum_iterations': 10,
                     'absolute_tolerance': 1e-2,
                     'newton_solver': newton_solver}

    # Combine and return all input dictionaries.
    inputs = {'wk_sys': wk_sys,
              'wk_pul': wk_pul,
              'attach_lvad': attach_lvad,
              'lvad_model': lvad_model,
              'geometry': geometry,
              'model': model,
              'material_model': material_model,
              'material_law': material_law,
              'time': time,
              'state': state,
              'initial_conditions': initial_conditions,
              'total_volume': total_volume,
              'form_compiler': form_compiler,
              'volume_solver': volume_solver,
              'number_of_cycles': number_of_cycles,
              'infarct': infarct_prm}

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

    # Plot the hemodynamic relations
    simulation_plot = HemodynamicsPlotDC(results)
    simulation_plot.plot(circulation='systemic')  # cycle=NUM_CYCLES)
    simulation_plot.save(os.path.join(dir_out, 'hemodynamics_systemic.png'))

    simulation_plot.plot(circulation='pulmonary')  # cycle=NUM_CYCLES)
    simulation_plot.save(os.path.join(dir_out, 'hemodynamics_pulmonary.png'))

    simulation_plot.plot_pvloops()
    simulation_plot.save(os.path.join(dir_out, 'pvloops.png'))

    simulation_plot.plot_function()
    simulation_plot.save(os.path.join(dir_out, 'ventricles_function.png'))

    # Save the CSV data file.
    results.save(os.path.join(dir_out, 'results.csv'))


def main():
    # Check whether we want to reload a model, or create a new one.
    if TIME_RELOAD is not None and DIR_RELOAD is not None:
        # Reload.
        if os.path.exists(DIR_RELOAD):
            # Optionally specify input parameters to overrule loaded inputs.
            overrule_inputs = {}  # Use an empty dictionary if you do not want to overrule loaded inputs.
            wk, biv, results, inputs = ReloadState().reload(DIR_RELOAD, TIME_RELOAD, **overrule_inputs)
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
        wk, biv, results = preprocess_biv(inputs)

    # Save fiber and coordinate fields for visualization in paraview.
    print_once('Saving fiber and coordinate fields to {} directory ...'.format(DIR_OUT))
    biv.geometry.save_vectors_and_coordinate_systems(biv.u.ufl_function_space(),
                                                     dir_out=os.path.join(DIR_OUT, 'geometry'))

    # Save inputs to a csv file (as we might later wonder what inputs we have used).
    print_once('Saving inputs to {} ...'.format(os.path.join(DIR_OUT, 'inputs.csv')))
    save_dict_to_csv(inputs, os.path.join(DIR_OUT, 'inputs.csv'))

    # Setup a VolumeSolver with a custom NewtonSolver (optional).
    # Here, a custom Newton solver is used which detects divergence and saves residuals.
    solver = VolumeSolverBiV(custom_newton_solver=CustomNewtonSolver(model=biv, dir_out=DIR_OUT),
                             **inputs['volume_solver'])

    # Call the simulation routine.
    # This try/except clause is a fail safe to write data if it crashes.
    try:
        simulate(wk, biv, results, inputs, 'Biventricle', solver=solver, dir_out=DIR_OUT)

    except RuntimeError as error_detail:
        print_once('Except RuntimeError: {}'.format(error_detail))
        if MPI.rank(mpi_comm_world()) == 0:
            print('Simulation failed! Plotting and saving the CSV file...')

    # Call the post-processing routine.
    postprocess(results, dir_out=DIR_OUT)


if __name__ == '__main__':
    main()
