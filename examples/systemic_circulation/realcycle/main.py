"""
This script provides an example on how to simulate multiple cardiac cycles using
FEniCS with Python, with some minor post-processing.

LV is modelled with a FE model, the systemic circulation is modelled as a
windkessel model.
"""
from dolfin.cpp.common import MPI, mpi_comm_world

from cvbtk import (HemodynamicsPlot, VolumeSolver, print_once, save_dict_to_csv,
                   CustomNewtonSolver, ReloadState, simulate, preprocess_lv, read_dict_from_csv)

import cvbtk.resources
import datetime
import os
import matplotlib.pyplot as plt

# Change the number of cycles and active stress ('new' or 'old') here:
NUM_CYCLES = 2
ACT_STRESS = 'old'  # 'old' is Arts Kerckhoffs, 'new' is Arts Bovendeerd.

# Use the following options if you want to reload a saved model state and continue
# the simulation from that state (e.g. handy when a simulation crashed).
# See ReloadState.reload() for a detailed description of these options.
# Set both options to None if you don't want to reload.
DIR_RELOAD = None  # Directory with output files of the model to reload.
TIME_RELOAD = None  # The time (in ms) of the timestep to reload. Set to -1 for reloading the latest available timestep.

# Set if Infarction should be included
INFARCT = True

# Use the following option if you want to load a set of inputs and start a new simulation using those inputs.
# By specifying a path to an inputs.csv file, you can load the inputs from the file
# instead of defining them in get_inputs(). If you do not want to load inputs from a
# file and you want to define the inputs in get_inputs(), set the below path to None.
INPUTS_PATH = None #'inputs.csv'


# Set mesh resololution. For the default mesh, chose 30, 40 or 50. 
SET_MESH_RESOLUTION = 30.0

# Use the following option if you want to load an alternative mesh (that has already been created). 
# By specifying a path to an .hdf5 file, you can load the mesh from the file
# instead of the reference mesh. If you do not want to load an alternative mesh from a
# file, but just use the reference lv mesh, set the below path to None.
LOAD_ALTERNATIVE_MESH = 'lv_maaike_seg30_res{}_mesh.hdf5'.format(int(SET_MESH_RESOLUTION))
#doofus code here
# Specify output directory.

now = datetime.datetime.now()

DIR_OUT = 'output/{}_infarct_xi_10'.format(now.strftime("%d-%m_%H-%M"))

# Create directory if it doesn't exists.
if MPI.rank(mpi_comm_world()) == 0:
    if not os.path.exists(DIR_OUT):
        os.makedirs(DIR_OUT)
    # else:
    #     DIR_OUT = DIR_OUT + "_v2"
    #     os.makedirs(DIR_OUT)
print_once('Saving to output directory: {}'.format(DIR_OUT))

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
            'tc': 800.0}  # Maximum time for one cycle.

    # -------------------------------------------------------------------------- #
    # SYSTEMIC circulation: create a dictionary of inputs for WindkesselModel.   #
    # -------------------------------------------------------------------------- #
    # Specify type of windkessel ('WindkesselModel' (closed-loop) or
    # 'LifetecWindkesselModel' (open-loop))
    windkessel_type = 'WindkesselModel'
    windkessel_model = {'arterial_compliance': 25.0,
                        'arterial_resistance': 10.0,
                        'arterial_resting_volume': 500.0,
                        'peripheral_resistance': 120.0,
                        'total_volume': 5000.0,
                        'venous_compliance': 600.0,
                        'venous_resistance': 5.0,
                        'venous_resting_volume': 3000.0}

    # TODO add option to attach LVAD.

    # -------------------------------------------------------------------------- #
    # Infarct: create a dictionary of inputs for the infarct geometry.           #
    # -------------------------------------------------------------------------- #
    if INFARCT == True:
        infarct_prm = { 'phi_min': 0.,
                        'phi_max': 1.5708,
                        'theta_min': 2.,
                        'theta_max': 2.8,
                        'ximin': 10., #0.5,
                        'focus': 4.3,
                        'Ta0_infarct': 100.,
                        'save_T0_mesh': DIR_OUT}
    else:
        infarct_prm = None

    # -------------------------------------------------------------------------- #
    # LV: create a dictionary of inputs for LeftVentricle geometry.              #
    # -------------------------------------------------------------------------- #
    # We do not need to set all of the geometry parameters
    # as we will load a reference mesh.
    # Specify reference mesh to load and resolution.

    # Specify alternative mesh, enter for geometry_type: alternative_lv_mesh
    # after the ',' enter the filename of the alternative mesh
    if LOAD_ALTERNATIVE_MESH == None:
        geometry_type = 'reference_left_ventricle', None
    else:
        geometry_type = 'alternative_lv_mesh', LOAD_ALTERNATIVE_MESH
    geometry = {'mesh_resolution': SET_MESH_RESOLUTION}

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
    material_law = 'BovendeerdMaterial'
    material_model = {'a0': 0.4,
                      'a1': 3.0,
                      'a2': 6.0,
                      'a3': 3.0,
                      'a4': 0.0,
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

    active_stress_arts_bovendeerd = {'Ta0': 160.0,  # pg. 66: 250 kPa
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
                     'absolute_tolerance': 1e-4,
                     'linear_solver': 'bicgstab',
                     'preconditioner': 'hypre_euclid',
                     'error_on_nonconvergence': True,
                     'krylov_solver': {'absolute_tolerance': 1e-7}}

    # Volume solver settings (matching cavity volumes with expected volumes).
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
              'form_compiler': form_compiler,
              'state': state,
              'initial_conditions': initial_conditions,
              'time': time,
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
    # Plot the results.
    plot = HemodynamicsPlot(results)
    plot.plot()
    plot.save(os.path.join(dir_out, 'results.png'))

    plot.plot_function()
    plt.savefig(os.path.join(dir_out, 'lv_function.png'))

    # Plot the results against the reference results.
    reference_data = cvbtk.resources.reference_hemodynamics()
    reference_plot = HemodynamicsPlot(reference_data)
    reference_plot.plot(cycle=10)
    reference_plot.compare_against(results, 'o', ms=2, markevery=2)
    reference_plot.save(os.path.join(dir_out, 'comparison.png'))

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
    #  Here, a custom Newton solver is used which detects divergence and saves residuals.
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
