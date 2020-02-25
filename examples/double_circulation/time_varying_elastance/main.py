"""
This script simulates the systemic and pulmonary circulation with a 0D model.
Left and right ventricles are modelled as time varying elastance models and both
circulations as 3 element windkessel models.
Its possible to include an LVAD.
"""

from dolfin.cpp.common import MPI, info, mpi_comm_world

from cvbtk import Dataset, save_dict_to_csv, GeneralWindkesselModel, \
    TimeVaryingElastance, VentriclesWrapper, HemodynamicsPlotDC, mmHg_to_kPa, HeartMateII

from os.path import join
import os

# Change the number of cycles here:
NUM_CYCLES = 10

# Output directory
DIR_OUT = 'output/time_var_elastance_DC'

# Create directory if it doesn't exists.
if MPI.rank(mpi_comm_world()) == 0:
    if not os.path.exists(DIR_OUT):
        os.makedirs(DIR_OUT)
# Synchronize.
MPI.barrier(mpi_comm_world())


def get_inputs(number_of_cycles):
    """
    Helper function to return a dictionary of inputs in one convenient location.
    """

    # -------------------------------------------------------------------------- #
    # Global settings.                                                                        #
    # -------------------------------------------------------------------------- #
    time = {'dt': 2.0,
            't0': 0.0,
            't1': 8000.,  # Maximum simulation time.
            'tc': 800.}   # Maximum time for one cycle.

    # Total blood volume.
    total_volume = 5000.0

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
    attach_lvad = True
    lvad_model = {'frequency': float(9.5), # [krpm]
                   'lvad_volume': 66.0,
                   'alpha_slope': 0.0091,
                   'alpha_intercept': 1.4,
                   'beta_slope': -0.19,
                   'beta_intercept': -1.9}

    # -------------------------------------------------------------------------- #
    # RV: create a dictionary of inputs for TimeVaryingElastance model.          #
    # -------------------------------------------------------------------------- #
    tve_rv = {'elastance_pas': mmHg_to_kPa(10/100), #0.0024,  # [kPa/ml] (float)
                 'elastance_max': mmHg_to_kPa(30/10),  # [kPa/ml] (float)
                 'ventricle_resting_volume': 40.,  # [ml] (float)
                 'time_cycle': time['tc'],  # [ms] (float)
                 'time_activation': 400.,  # [ms] (float)
                 'time_depolarization': 300.} # [ms] (float)

    # -------------------------------------------------------------------------- #
    # LV: create a dictionary of inputs for TimeVaryingElastance model.          #
    # -------------------------------------------------------------------------- #
    tve_lv = {'elastance_pas': mmHg_to_kPa(10/80), #0.0089,  # [kPa/ml] (float)
                 'elastance_max': mmHg_to_kPa(110/20), #0.24,  # [kPa/ml] (float)
                 'ventricle_resting_volume': 40.,  # [ml] (float)
                 'time_cycle': time['tc'],  # [ms] (float)
                 'time_activation': 400.,  # [ms] (float)
                 'time_depolarization': 300.}  # [ms] (float)

    # -------------------------------------------------------------------------- #
    # Specify initial conditions                                                 #
    # -------------------------------------------------------------------------- #
    initial_conditions = {'p_lv': 0.,
                        'p_art_sys': 12.5,
                        'p_rv': 0.,
                        'p_art_pul': 3.5,
                        'p_ven_pul': 3}

    # Note that venous pressure of the systemic circulation will be calculated from
    # venous volume, which will be calculated from mass conservation.

    # Combine and return all input dictionaries.
    inputs = {'wk_sys': wk_sys,
              'wk_pul': wk_pul,
              'attach_lvad': attach_lvad,
              'lvad_model': lvad_model,
              'tve_lv': tve_lv,
              'tve_rv': tve_rv,
              'time': time,
              'initial_conditions': initial_conditions,
              'number_of_cycles': number_of_cycles,
              'total_volume': total_volume}

    return inputs

def preprocess(inputs):
    # ------------------------------------------------------------------------ #
    # Create a dataset container to store state values.                        #
    # ------------------------------------------------------------------------ #
    dataset_keys = ['time', 'cycle',
                    'pcav_s', 'part_s', 'pven_s', 'qart_s', 'qper_s', 'qven_s', 'vcav_s', 'vart_s', 'vven_s',
                    'pcav_p', 'part_p', 'pven_p', 'qart_p', 'qper_p', 'qven_p', 'vcav_p', 'vart_p', 'vven_p']

    # Create the dataset.
    results = Dataset(keys=dataset_keys)

    # ------------------------------------------------------------------------ #
    # Create the windkessel model for the systemic circulation part.           #
    # ------------------------------------------------------------------------ #
    wk_sys = GeneralWindkesselModel('systemic_windkessel', **inputs['wk_sys'])

    # ------------------------------------------------------------------------ #
    # Create the windkessel model for the pulmonary circulation part.          #
    # ------------------------------------------------------------------------ #
    wk_pul = GeneralWindkesselModel('pulmonary_windkessel', **inputs['wk_pul'])

    # ------------------------------------------------------------------------ #
    # Add LVAD.                                                                #
    # ------------------------------------------------------------------------ #
    if inputs.get('attach_lvad', False):
        wk_sys.lvad = HeartMateII(**inputs['lvad_model'])

    # ------------------------------------------------------------------------ #
    # Create the biventricular model.                                          #
    # ------------------------------------------------------------------------ #
    # Create the model for the right ventricle.                                #
    # Use a time varying elastance model
    rv = TimeVaryingElastance(key='rv', **inputs['tve_rv'])

    # Create the model for the left ventricle.                                 #
    # Use a time varying elastance model
    lv = TimeVaryingElastance(key='lv', **inputs['tve_lv'])

    # Wrap the ventricle models into one object
    biv = VentriclesWrapper(lv=lv, rv=rv)

    # ------------------------------------------------------------------------ #
    # Inspect the parameters of the windkessel and ventricle models:           #
    # ------------------------------------------------------------------------ #
    if MPI.rank(mpi_comm_world()) == 0:
        info(wk_sys.parameters, True)
        info(wk_pul.parameters, True)
        info(biv.parameters['lv'], True)
        info(biv.parameters['rv'], True)

    # ------------------------------------------------------------------------ #
    # Set the initial conditions.                                              #
    # ------------------------------------------------------------------------ #
    # Determine initial state by setting initial pressures (in kPa)
    biv.pressure = {'lv': inputs['initial_conditions']['p_lv'],
                    'rv': inputs['initial_conditions']['p_rv']}

    wk_sys.pressure = {'art': inputs['initial_conditions']['p_art_sys']}

    wk_pul.pressure = {'art': inputs['initial_conditions']['p_art_pul'],
                       'ven': inputs['initial_conditions']['p_ven_pul']}

    t0 = inputs['time']['t0']

    # Compute initial volumes from initial pressures
    biv.volume = biv.compute_volume(t0)
    wk_sys.volume = wk_sys.compute_volume()
    wk_pul.volume = wk_pul.compute_volume()

    # Compute initial venous volume (systemic) from mass conservation.
    vven_sys = inputs['total_volume'] - biv.volume['lv'] - biv.volume['rv'] - wk_sys.volume['art'] - wk_pul.volume['art'] - \
               wk_pul.volume['ven'] - wk_sys.volume.get('lvad', 0)
    wk_sys.volume = {'ven': vven_sys}

    # Compute venous pressure (systemic) from venous volume.
    wk_sys.pressure = {'ven': wk_sys.compute_pressure()['ven']}

    # Compute initial flowrates from initial pressures.
    p_boundary_sys = {'in': biv.pressure['lv'],
                      'out': biv.pressure['rv']}
    wk_sys.flowrate = wk_sys.compute_flowrate(p_boundary_sys)

    p_boundary_pul = {'in': biv.pressure['rv'],
                      'out': biv.pressure['lv']}
    wk_pul.flowrate = wk_pul.compute_flowrate(p_boundary_pul)

    # Print out the current (initial) state just to double check the values:
    if MPI.rank(mpi_comm_world()) == 0:
        print('The initial systemic WK state is V = {}.'.format(wk_sys.volume))
        print('The initial systemic WK state is p = {}.'.format(wk_sys.pressure))
        print('The initial systemic WK state is q = {}.'.format(wk_sys.flowrate))

        print('The initial pulmonary WK state is V = {}.'.format(wk_pul.volume))
        print('The initial pulmonary WK state is p = {}.'.format(wk_pul.pressure))
        print('The initial pulmonary WK state is q = {}.'.format(wk_pul.flowrate))
        print('The initial BiV state is V = {}.'.format(biv.volume))
        print('The initial BiV state is p = {}.'.format(biv.pressure))

    return wk_sys, wk_pul, biv, results

def simulate(wk_sys, wk_pul, biv, results, inputs):
    # ------------------------------------------------------------------------ #
    # Set up the simulation loop.                                              #
    # ------------------------------------------------------------------------ #
    cycle = 1
    dt = inputs['time']['dt']
    t0 = inputs['time']['t0']
    t1 = inputs['time']['t1']
    tc = inputs['time']['tc']
    t = t0

    # ------------------------------------------------------------------------ #
    # Create the output data files.                                            #
    # ------------------------------------------------------------------------ #
    write_data(t, cycle, wk_sys, wk_pul, biv, results)

    # ------------------------------------------------------------------------ #
    # The time loop is below.                                                  #
    # ------------------------------------------------------------------------ #
    while t < t1:
        # -------------------------------------------------------------------- #
        # t = n                                                                #
        # -------------------------------------------------------------------- #

        # -------------------------------------------------------------------- #
        # timestep                                                             #
        # -------------------------------------------------------------------- #
        timestep(t, dt, wk_sys, wk_pul, biv, inputs)

        # Update time states after successful solutions.
        t += dt

        # -------------------------------------------------------------------- #
        # t = n + 1                                                            #
        # -------------------------------------------------------------------- #

        # Cycle number
        cycle = int(t // tc + 1)

        # Append selected data to the file records.
        write_data(t, cycle, wk_sys, wk_pul, biv, results)

        # Print some state information about the completed timestep:
        if MPI.rank(mpi_comm_world()) == 0:
            msg = ('*** [Cycle {}/{}]:'
                   ' t = {} ms,'
                   ' p_lv = {:5.2f} kPa, V_lv = {:6.2f} ml'
                   ' p_rv = {:5.2f} kPa, V_rv = {:6.2f} ml')
            print(msg.format(cycle, inputs['number_of_cycles'],
                             t,
                             biv.pressure['lv'], biv.volume['lv'],
                             biv.pressure['rv'], biv.volume['rv']))

        # Exit if maximum cycles reached:
        if cycle > inputs['number_of_cycles']:
            if MPI.rank(mpi_comm_world()) == 0:
                print('Maximum number of cycles simulated!')
            break


def postprocess(results):
    # Plot the hemodynamic relations
    simulation_plot = HemodynamicsPlotDC(results)
    simulation_plot.plot(circulation='systemic') #, cycle=NUM_CYCLES)
    simulation_plot.save(join(DIR_OUT, 'hemodynamics_systemic.png'))

    simulation_plot.plot(circulation='pulmonary') #, cycle=NUM_CYCLES)
    simulation_plot.save(join(DIR_OUT, 'hemodynamics_pulmonary.png'))

    simulation_plot.plot_pvloops(cycle=NUM_CYCLES)
    simulation_plot.save(join(DIR_OUT, 'pvloops.png'))

    # Save the CSV data file.
    results.save(join(DIR_OUT, 'results.csv'))

def timestep(t, dt, wk_sys, wk_pul, biv, inputs):
    # -------------------------------------------------------------------- #
    # t = n                                                                #
    # -------------------------------------------------------------------- #
    # Store a copy/backup of the state values and unknowns at t = n:
    v_old_sys = wk_sys.volume
    q_old_sys = wk_sys.flowrate

    v_old_pul = wk_pul.volume
    q_old_pul = wk_pul.flowrate

    vbiv_old = biv.volume

    # -------------------------------------------------------------------- #
    # time increment                                                       #
    # -------------------------------------------------------------------- #
    t += dt

    # -------------------------------------------------------------------- #
    # t = n + 1                                                            #
    # -------------------------------------------------------------------- #
    # Compute the new ventricular volumes and solve for the pressures.
    vlv_new = vbiv_old['lv'] + dt*(q_old_pul['ven'] - q_old_sys['art'] - q_old_sys.get('lvad', 0))
    vrv_new = vbiv_old['rv'] + dt*(q_old_sys['ven'] - q_old_pul['art'])
    pbiv_new = biv.compute_pressure(t)
    plv_new = pbiv_new['lv']
    prv_new = pbiv_new['rv']
    biv.volume = {'lv': vlv_new,
                  'rv': vrv_new}
    biv.pressure = {'lv': plv_new,
                    'rv': prv_new}

    # Compute the new windkessel model state:
    # Compute the new volumes with a simple forward Euler scheme:
    vart_new_sys = v_old_sys['art'] + dt*(q_old_sys['art'] - q_old_sys['per'] + q_old_sys.get('lvad', 0))
    vart_new_pul = v_old_pul['art'] + dt*(q_old_pul['art'] - q_old_pul['per'])
    vven_new_pul = v_old_pul['ven'] + dt*(q_old_pul['per'] - q_old_pul['ven'])
    vven_new_sys = inputs['total_volume'] - vlv_new - vart_new_sys - vrv_new - vart_new_pul - vven_new_pul \
                   - wk_sys.volume.get('lvad', 0)
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
    p_boundary_sys = {'in' : biv.pressure['lv'],
                      'out': biv.pressure['rv']}
    q_new_sys = wk_sys.compute_flowrate(p_boundary_sys)
    wk_sys.flowrate = q_new_sys

    # Pulmonary wk
    p_boundary_pul = {'in' : biv.pressure['rv'],
                      'out': biv.pressure['lv']}
    q_new_pul = wk_pul.compute_flowrate(p_boundary_pul)
    wk_pul.flowrate = q_new_pul

def write_data(t, cycle, wk_sys, wk_pul, biv, results):
    # This one only appends data to the record in memory.
    results.append(time=t,
                   cycle=cycle,

                 pcav_s=biv.pressure['lv'],
                 vcav_s=biv.volume['lv'],

                 vart_s=wk_sys.volume['art'],
                 vven_s=wk_sys.volume['ven'],

                 part_s=wk_sys.pressure['art'],
                 pven_s=wk_sys.pressure['ven'],

                 qart_s=wk_sys.flowrate['art']*1000,
                 qven_s=wk_sys.flowrate['ven']*1000,
                 qper_s=wk_sys.flowrate['per']*1000,

                 pcav_p=biv.pressure['rv'],
                 vcav_p=biv.volume['rv'],

                 vart_p=wk_pul.volume['art'],
                 vven_p=wk_pul.volume['ven'],

                 part_p=wk_pul.pressure['art'],
                 pven_p=wk_pul.pressure['ven'],

                 qart_p=wk_pul.flowrate['art']*1000,
                 qven_p=wk_pul.flowrate['ven']*1000,
                 qper_p=wk_pul.flowrate['per']*1000)

# def main():
# Rather than defining inputs inline, we will define them in inputs().
inputs = get_inputs(number_of_cycles=NUM_CYCLES)

# Save inputs to a csv file (as we might later wonder what inputs we have used)
save_dict_to_csv(inputs, join(DIR_OUT, 'inputs.csv'))

# Define and set up the model.
wk_sys, wk_pul, biv, results = preprocess(inputs)

# Call the simulation routine.
# This try/except clause is a fail safe to write data if it crashes.
try:
    simulate(wk_sys, wk_pul, biv, results, inputs)

except RuntimeError:
    if MPI.rank(mpi_comm_world()) == 0:
        print('Simulation failed! Saving the CSV file.')

# Call the post-processing routine.
postprocess(results)

# if __name__ == '__main__':
#     main()
