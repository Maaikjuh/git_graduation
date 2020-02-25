"""
This script simulates the systemic and pulmonary circulation with a 0D model.
The left ventricle is modelled by a FEM, the right ventricle as time varying elastance
model and both circulations as 3 element windkessel models.
"""

from dolfin import DirichletBC, parameters
from dolfin.cpp.common import MPI, info, mpi_comm_world
from dolfin.cpp.io import HDF5File

import cvbtk.resources
from cvbtk import (Dataset, save_dict_to_csv,
                   ArtsBovendeerdActiveStress, ArtsKerckhoffsActiveStress,
                   BovendeerdMaterial,
                   LeftVentricleModel, VolumeSolver,
                   build_nullspace, HemodynamicsPlotDC,
                   TimeVaryingElastance, GeneralWindkesselModel, get_phase_dc)

from os.path import join
import os

# Change the number of cycles and active stress ('new' or 'old') here:
NUM_CYCLES = 10
ACT_STRESS = 'old'

# Output directory.
DIR_OUT = 'output/lv_fem_rv_tve'

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
    # Time                                                                       #
    # -------------------------------------------------------------------------- #
    time = {'dt': 2.0,
            't0': 0.0,
            't1': 8000.0,  # Maximum simulation time.
            'tc': 800.0}  # Maximum time for one cycle.

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
    # RV: create a dictionary of inputs for TimeVaryingElastance model.          #
    # -------------------------------------------------------------------------- #
    tve_rv = {'elastance_pas': 0.0024,  # [kPa/ml] (float)
                 'elastance_max': 0.089,  # [kPa/ml] (float)
                 'ventricle_resting_volume': 0.,  # [ml] (float)
                 'time_cycle': time['tc'],  # [ms] (float)
                 'time_activation': 400.,  # [ms] (float)
                 'time_depolarization': 300.} # [ms] (float) Needs to be synchronized with LV.


    # -------------------------------------------------------------------------- #
    # LV: geometry inputs for finite element model.                              #
    # -------------------------------------------------------------------------- #
    geometry = {'wall_volume': 136.0,
                'cavity_volume': 44.0,
                'focus_height': 4.3,
                'truncation_height': 2.4,
                'mesh_segments': 30,
                'mesh_resolution': 30.0}

    material_model = {'a0': 0.4,
                      'a1': 3.0,
                      'a2': 6.0,
                      'a3': 3.0,
                      'a4': 0.0,
                      'a5': 55.0}

    # Shared active stress parameters:
    active_stress_ls0 = 1.9
    active_stress_beta = 0.0
    active_stress_tdep = 300.0

    active_stress_arts_kerckhoffs = {'T0': 160.0,
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
                                     'tdep': active_stress_tdep}

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

    form_compiler = {'quadrature_degree': 4,
                     'cpp_optimize_flags': '-O3 -march=native -mtune=native'}

    newton_solver = {'maximum_iterations': 15,
                     'absolute_tolerance': 1e-4,
                     'linear_solver': 'bicgstab',
                     'preconditioner': 'hypre_euclid',
                     'error_on_nonconvergence': False,
                     'krylov_solver': {'absolute_tolerance': 1e-7}}

    volume_solver = {'maximum_iterations': 10,
                     'absolute_tolerance': 1e-2,
                     'newton_solver': newton_solver}

    # -------------------------------------------------------------------------- #
    # Specify initial conditions                                                 #
    # -------------------------------------------------------------------------- #
    # Note that venous pressure of the systemic circulation will be calculated from
    # venous volume, which will be calculated from mass conservation.
    initial_conditions = {'p_lv': 0.,
                        'p_art_sys': 15.,
                        'p_rv': 0.,
                        'p_art_pul': 3.,
                        'p_ven_pul': 5.}

    total_volume = 5000.0

    # Combine and return all input dictionaries.
    inputs = {'wk_sys': wk_sys,
              'wk_pul': wk_pul,
              'geometry': geometry,
              'material_model': material_model,
              'tve_rv': tve_rv,
              'form_compiler': form_compiler,
              'time': time,
              'initial_conditions': initial_conditions,
              'volume_solver': volume_solver,
              'number_of_cycles': number_of_cycles,
              'total_volume': total_volume}

    # Add the proper active stress parameters:
    if active_stress == 'old':
        inputs['active_stress'] = active_stress_arts_kerckhoffs
        inputs['active_stress_model'] = 'ArtsKerckhoffsActiveStress'
    elif active_stress == 'new':
        inputs['active_stress'] = active_stress_arts_bovendeerd
        inputs['active_stress_model'] = 'ArtsBovendeerdActiveStress'

    return inputs

def preprocess(inputs):
    # ------------------------------------------------------------------------ #
    # Create a dataset container to store state values.                        #
    # ------------------------------------------------------------------------ #
    dataset_keys = ['time', 'cycle',
                    'pcav_s', 'part_s', 'pven_s', 'qart_s', 'qper_s', 'qven_s', 'vcav_s', 'vart_s', 'vven_s',
                    'pcav_p', 'part_p', 'pven_p', 'qart_p', 'qper_p', 'qven_p', 'vcav_p', 'vart_p', 'vven_p']

    # Add the following additional keys.
    dataset_keys.append('t_act')  # time before/since activation
    dataset_keys.append('t_cycle')  # time in the current cycle
    dataset_keys.append('vector_number')  # corresponds to the HDF5 output
    dataset_keys.append('accuracy')
    dataset_keys.append('est')

    # Create the dataset.
    results = Dataset(keys=dataset_keys)

    # ------------------------------------------------------------------------ #
    # Set the proper global FEniCS parameters.                                 #
    # ------------------------------------------------------------------------ #
    parameters.update({'form_compiler': inputs['form_compiler']})

    # ------------------------------------------------------------------------ #
    # Create the windkessel model for the systemic circulation part.           #
    # ------------------------------------------------------------------------ #
    wk_sys = GeneralWindkesselModel('systemic_windkessel', **inputs['wk_sys'])

    # ------------------------------------------------------------------------ #
    # Create the windkessel model for the pulmonary circulation part.          #
    # ------------------------------------------------------------------------ #
    wk_pul = GeneralWindkesselModel('pulmonary_windkessel', **inputs['wk_pul'])

    # ------------------------------------------------------------------------ #
    # Create the model for the right ventricle.                                #
    # ------------------------------------------------------------------------ #
    # Use a time varying elastance model
    rv = TimeVaryingElastance('rv', **inputs['tve_rv'])

    # ------------------------------------------------------------------------ #
    # Create the model for the left ventricle.                                 #
    # ------------------------------------------------------------------------ #
    # Use a finite element model
    # The default/available parameters can be printed, which are good enough.
    if MPI.rank(mpi_comm_world()) == 0:
        info(LeftVentricleModel.default_parameters(), True)

    # For the geometry we re-use the reference mesh.
    res = inputs['geometry']['mesh_resolution']
    geometry = cvbtk.resources.reference_left_ventricle(resolution=res)

    # The finite element method model is created with the geometry.
    lv = LeftVentricleModel(geometry)

    # LeftVentricleModel defines u, from which V can be collected.
    u = lv.u
    V = u.ufl_function_space()
    fsn = geometry.fiber_vectors()

    # Material models all take u, fsn, and arbitrary inputs for arguments.
    # It needs to be set by assigning it to the ``material`` attribute.
    lv.material = BovendeerdMaterial(u, fsn, **inputs['material_model'])

    # The next step can switch between the two active stress models implemented.
    if inputs['active_stress_model'] == 'ArtsBovendeerdActiveStress':
        act = ArtsBovendeerdActiveStress(u, fsn, **inputs['active_stress'])
    elif inputs['active_stress_model'] == 'ArtsKerckhoffsActiveStress':
        act = ArtsKerckhoffsActiveStress(u, fsn, **inputs['active_stress'])
    else:
        act = None
    lv.active_stress = act

    # Dirichlet boundary conditions fix the base.
    lv.bcs = DirichletBC(V.sub(2), 0.0, geometry.tags(), geometry.base)

    # We have not fully eliminated rigid body motion yet. To do so, we will
    # define a nullspace of rigid body motions and use a iterative method which
    # can eliminate rigid body motions using this nullspace.
    lv.nullspace = build_nullspace(u, modes=['x', 'y', 'xy'])

    # ------------------------------------------------------------------------ #
    # Inspect the parameters of the windkessel and ventricle models:           #
    # ------------------------------------------------------------------------ #
    if MPI.rank(mpi_comm_world()) == 0:
        info(wk_sys.parameters, True)
        info(wk_pul.parameters, True)
        info(lv.parameters, True)
        info(rv.parameters, True)

    # ------------------------------------------------------------------------ #
    # Set the initial conditions.                                              #
    # ------------------------------------------------------------------------ #
    # The LeftVentricleModel class was designed to mimic the WindkesselModel
    # class as much as possible, so setting the current pressure and volume
    # state is done by assignment of a dictionary item.

    # The dictionary item might seem redundant, but maybe in the future we have
    # a BiventricleModel which would have a 'lv' and 'rv' volumes and pressures.

    # Determine initial state by setting initial pressures (in kPa)
    lv.pressure = {'lv': 0.0}

    rv.pressure = {'rv': inputs['initial_conditions']['p_rv']}

    wk_sys.pressure = {'art': inputs['initial_conditions']['p_art_sys']}

    wk_pul.pressure = {'art': inputs['initial_conditions']['p_art_pul'],
                       'ven': inputs['initial_conditions']['p_ven_pul']}

    t0 = inputs['time']['t0']

    # Compute initial volumes from initial pressures (or from geometry in case of FEM)
    lv.volume = {'lv': lv.geometry.compute_volume()}
    rv.volume = rv.compute_volume(t0)
    wk_sys.volume = wk_sys.compute_volume()
    wk_pul.volume = wk_pul.compute_volume()

    # Compute initial venous volume (systemic) from mass conservation
    vven_sys = inputs['total_volume'] - lv.volume['lv'] - rv.volume['rv'] - wk_sys.volume['art'] - wk_pul.volume['art'] - \
               wk_pul.volume['ven']
    wk_sys.volume = {'ven': vven_sys}

    # Compute venous pressure (systemic) from venous volume
    wk_sys.pressure = {'ven': wk_sys.compute_pressure()['ven']}

    # Compute initial flowrates from initial pressures
    p_boundary_sys = {'in': lv.pressure['lv'],
                      'out': rv.pressure['rv']}
    wk_sys.flowrate = wk_sys.compute_flowrate(p_boundary_sys)

    p_boundary_pul = {'in': rv.pressure['rv'],
                      'out': lv.pressure['lv']}
    wk_pul.flowrate = wk_pul.compute_flowrate(p_boundary_pul)

    # Print out the current (initial) state just to double check the values:
    if MPI.rank(mpi_comm_world()) == 0:
        print('The initial systemic WK state is V = {}.'.format(wk_sys.volume))
        print('The initial systemic WK state is p = {}.'.format(wk_sys.pressure))
        print('The initial systemic WK state is q = {}.'.format(wk_sys.flowrate))
        print('The initial LV state is V = {}.'.format(lv.volume))
        print('The initial LV state is p = {}.'.format(lv.pressure))

        print('The initial pulmonary WK state is V = {}.'.format(wk_pul.volume))
        print('The initial pulmonary WK state is p = {}.'.format(wk_pul.pressure))
        print('The initial pulmonary WK state is q = {}.'.format(wk_pul.flowrate))
        print('The initial RV state is V = {}.'.format(rv.volume))
        print('The initial RV state is p = {}.'.format(rv.pressure))

    return wk_sys, wk_pul, lv, rv, results

def simulate(wk_sys, wk_pul, lv, rv, results, inputs):
    # ------------------------------------------------------------------------ #
    # Set up the simulation loop.                                              #
    # ------------------------------------------------------------------------ #
    phase = 1
    cycle = 1
    dt = inputs['time']['dt']
    t0 = inputs['time']['t0']
    t1 = inputs['time']['t1']
    t = t0
    t_cycle = t0
    t_active_stress = t0

    # ------------------------------------------------------------------------ #
    # Create the output data files.                                            #
    # ------------------------------------------------------------------------ #
    write_data(t, t_cycle, phase, cycle, lv, wk_sys, rv, wk_pul, 0, 0, results, new=True)

    # ------------------------------------------------------------------------ #
    # Create the volume solver to solve the system.                            #
    # ------------------------------------------------------------------------ #
    solver = VolumeSolver(**inputs['volume_solver'])

   # ------------------------------------------------------------------------ #
    # The time loop is below.                                                  #
    # ------------------------------------------------------------------------ #
    while t < t1:
        # -------------------------------------------------------------------- #
        # t = n                                                                #
        # -------------------------------------------------------------------- #
        # Store a copy/backup of the state values and unknowns at t = n:
        u_array = lv.u.vector().array()
        u_old_array = lv.u_old.vector().array()
        ls_old_array = lv.active_stress.ls_old.vector().array()

        if isinstance(lv.active_stress, ArtsKerckhoffsActiveStress):
            lc_old_array = lv.active_stress.lc_old.vector().array()

        # -------------------------------------------------------------------- #
        # timestep                                                             #
        # -------------------------------------------------------------------- #
        # Since it's possible for the FEM solver to fail, it's better if we
        # make the rest of the time-loop its own function so that we can take
        # advantage of Python's try/except construct.

        # The idea is if the solution fails, then we'll reset u, ls, lc, etc.,
        # lower the dt, and call the time-step routine again.
        try:
            # Attempt to solve.
            accuracy = timestep(t_active_stress, t, dt, wk_sys, wk_pul, lv, rv, inputs, solver)

            # Update time states after successful solutions.
            t += dt
            t_cycle += dt
            t_active_stress += dt

        except RuntimeError:
            # Reset values from backup:
            reset_values(lv.u, u_array)
            reset_values(lv.u_old, u_old_array)
            reset_values(lv.active_stress.ls_old, ls_old_array)

            # Reset lc_old if using the proper active stress model.
            if isinstance(lv.active_stress, ArtsKerckhoffsActiveStress):
                # noinspection PyUnboundLocalVariable
                reset_values(lv.active_stress.lc_old, lc_old_array)

            # Re-attempt to solve.
            accuracy = timestep(t_active_stress, t, 0.5*dt, wk_sys, wk_pul, lv, rv, inputs, solver)

            # Update time states after successful solutions.
            t += 0.5*dt
            t_cycle += 0.5*dt
            t_active_stress += 0.5*dt

        # -------------------------------------------------------------------- #
        # t = n + 1                                                            #
        # -------------------------------------------------------------------- #
        # Check what the new phase is.
        phase_old = phase
        biv_pressure_old = {'lv': lv.pressure_old['lv'],
                            'rv': rv.pressure_old['rv']}
        biv_pressure = {'lv': lv.pressure['lv'],
                        'rv': rv.pressure['rv']}
        phase = get_phase_dc(biv_pressure_old, wk_sys.pressure, wk_pul.pressure, biv_pressure)

        # We're only interested in the phase of the LV here.
        phase = phase['lv']

        # Increment cycle count if needed.
        if phase == 1 and phase_old == 4:
            cycle += 1
            t_cycle = 0.0

        # Check if the active stress's internal time needs to be reset.
        if phase == 1 and t_active_stress >= inputs['time']['tc']:
            t_active_stress = t_active_stress - inputs['time']['tc']

        # Append selected data to the file records.
        est = solver.iteration()
        write_data(t, t_cycle, phase, cycle, lv, wk_sys, rv, wk_pul, est, accuracy, results)

        # Print some state information about the completed timestep:
        if MPI.rank(mpi_comm_world()) == 0:
            msg = ('*** [Cycle {}/{} - Phase = {}/4]:'
                   ' t = {} ms, t_cycle = {} ms,'
                   ' p_lv = {:5.2f} kPa, V_lv = {:6.2f} ml')
            print(msg.format(cycle, inputs['number_of_cycles'], phase,
                             t, t_cycle,
                             lv.pressure['lv'], lv.volume['lv']))

        # Exit if maximum cycles reached:
        #TODO check for convergence.
        if cycle > inputs['number_of_cycles']:
            if MPI.rank(mpi_comm_world()) == 0:
                print('Maximum number of cycles simulated!')
            break


def postprocess(results):
    # Plot the hemodynamic relations
    simulation_plot = HemodynamicsPlotDC(results)
    simulation_plot.plot(circulation='systemic')
    simulation_plot.save(join(DIR_OUT, 'hemodynamics_systemic.png'))

    simulation_plot.plot(circulation='pulmonary')
    simulation_plot.save(join(DIR_OUT, 'hemodynamics_pulmonary.png'))

    simulation_plot.plot_pvloops() #cycle=NUM_CYCLES)
    simulation_plot.save(join(DIR_OUT, 'pvloops.png'))

    # Save the CSV data file.
    results.save(join(DIR_OUT, 'results.csv'))

def timestep(t_active_stress, t, dt, wk_sys, wk_pul, lv, rv, inputs, solver):
    # -------------------------------------------------------------------- #
    # t = n                                                                #
    # -------------------------------------------------------------------- #
    # Store a copy/backup of the state values and unknowns at t = n:
    v_old_sys = wk_sys.volume
    q_old_sys = wk_sys.flowrate
    vlv_old = lv.volume
    u_array = lv.u.vector().array()

    v_old_pul = wk_pul.volume
    q_old_pul = wk_pul.flowrate
    vrv_old = rv.volume

    # Update the old ls (and lc) values with most recently computed values (is slow!).
    lv.active_stress.upkeep()

    # -------------------------------------------------------------------- #
    # time increment                                                       #
    # -------------------------------------------------------------------- #
    t += dt
    lv.active_stress.activation_time = t_active_stress + dt

    # -------------------------------------------------------------------- #
    # t = n + 1                                                            #
    # -------------------------------------------------------------------- #
    # Compute the new LV volume and solve for the LV pressure.
    v_target = vlv_old['lv'] + dt*(q_old_pul['ven'] - q_old_sys['art'])
    plv_new, vlv_new, accuracy, _ = solver.solve(lv, v_target)
    lv.volume = {'lv': vlv_new}
    lv.pressure = {'lv': plv_new}

    # Compute the new RV volume and solve for the RV pressure.
    vrv_new = vrv_old['rv'] + dt*(q_old_sys['ven'] - q_old_pul['art'])
    prv_new = rv.compute_pressure(t)['rv']
    rv.volume = {'rv': vrv_new}
    rv.pressure = {'rv': prv_new}

    # Compute the new windkessel model state:
    # Compute the new volumes with a simple forward Euler scheme:
    vart_new_sys = v_old_sys['art'] + dt*(q_old_sys['art'] - q_old_sys['per'])
    vart_new_pul = v_old_pul['art'] + dt*(q_old_pul['art'] - q_old_pul['per'])
    vven_new_pul = v_old_pul['ven'] + dt*(q_old_pul['per'] - q_old_pul['ven'])
    vven_new_sys = inputs['total_volume'] - vlv_new - vart_new_sys - vrv_new - vart_new_pul - vven_new_pul
    wk_sys.volume = {'art': vart_new_sys, 'ven': vven_new_sys}
    wk_pul.volume = {'art': vart_new_pul, 'ven': vven_new_pul}

    # -------------------------------------------------------------------- #
    # Compute new pressures from new volumes                               #
    # -------------------------------------------------------------------- #
    wk_sys.pressure = wk_sys.compute_pressure()
    wk_pul.pressure = wk_pul.compute_pressure()

    # -------------------------------------------------------------------- #
    # Compute new flowrates from new pressures                             #
    # -------------------------------------------------------------------- #
    # Systemic wk
    p_boundary_sys = {'in' : lv.pressure['lv'],
                      'out': rv.pressure['rv']}
    q_new_sys = wk_sys.compute_flowrate(p_boundary_sys)
    wk_sys.flowrate = q_new_sys

    # Pulmonary wk
    p_boundary_pul = {'in' : rv.pressure['rv'],
                      'out': lv.pressure['lv']}
    q_new_pul = wk_pul.compute_flowrate(p_boundary_pul)
    wk_pul.flowrate = q_new_pul

    # Update u_old with the values of u at t = n.
    lv.u_old = u_array

    return accuracy

def write_data(t, t_cycle, phase, cycle, lv, wk_sys, rv, wk_pul, est, acc, results, new=False):
    """
     Helper function to write data to the HDF5 and CSV records.
     """
    # Create a new HDF5 record if new == True, else append to existing record.
    hdf5_mode = 'w' if new else 'a'

    # This one actually writes data to disk:
    with HDF5File(mpi_comm_world(), join(DIR_OUT, 'results.hdf5'), hdf5_mode) as f:
        # Write the mesh if a new record is created.
        if new:
            f.write(lv.geometry.mesh(), 'mesh')

        # Write the primary displacement unknown.
        f.write(lv.u, 'displacement', t)

        # For the active stress:
        if isinstance(lv.active_stress, ArtsKerckhoffsActiveStress):
            f.write(lv.active_stress.lc_old, 'contractile_element', t)

        # For the CSV file:
        vector_number = f.attributes('displacement')['count'] - 1

    results.append(time=t,
                   t_cycle=t_cycle,
                   t_act=float(lv.active_stress.activation_time),

                   cycle=cycle,
                   phase=phase,
                   
                 pcav_s=lv.pressure['lv'],
                 vcav_s=lv.volume['lv'],
                   
                 vart_s=wk_sys.volume['art'],
                 vven_s=wk_sys.volume['ven'],
                   
                 part_s=wk_sys.pressure['art'],
                 pven_s=wk_sys.pressure['ven'],
                   
                 qart_s=wk_sys.flowrate['art']*1000,
                 qven_s=wk_sys.flowrate['ven']*1000,
                 qper_s=wk_sys.flowrate['per']*1000,
                   
                 pcav_p=rv.pressure['rv'],
                 vcav_p=rv.volume['rv'],
                   
                 vart_p=wk_pul.volume['art'],
                 vven_p=wk_pul.volume['ven'],
                   
                 part_p=wk_pul.pressure['art'],
                 pven_p=wk_pul.pressure['ven'],
                   
                 qart_p=wk_pul.flowrate['art']*1000,
                 qven_p=wk_pul.flowrate['ven']*1000,
                 qper_p=wk_pul.flowrate['per']*1000,

                 est=est,
                 accuracy=acc,
                 vector_number=vector_number)

    # Save the CSV data file.
    results.save(join(DIR_OUT, 'results.csv'))

def reset_values(function_to_reset, array_to_reset_from):
    """
    Helper function to reset DOLFIN quantities.
    """
    function_to_reset.vector()[:] = array_to_reset_from
    function_to_reset.vector().apply('')

def main():
    # Rather than defining inputs inline, we will define them in inputs().
    inputs = get_inputs(number_of_cycles=NUM_CYCLES, active_stress=ACT_STRESS)

    # Save inputs to a csv file (as we might later wonder what inputs we have used)
    save_dict_to_csv(inputs, join(DIR_OUT, 'inputs.csv'))

    # Define and set up the model.
    wk_sys, wk_pul, lv, rv, results = preprocess(inputs)

    # Call the simulation routine.
    # This try/except clause is a fail safe to write data if it crashes.
    try:
        simulate(wk_sys, wk_pul, lv, rv, results, inputs)

    except RuntimeError:
        if MPI.rank(mpi_comm_world()) == 0:
            print('Simulation failed! Saving the CSV file.')

    # Call the post-processing routine.
    postprocess(results)

if __name__ == '__main__':
    main()