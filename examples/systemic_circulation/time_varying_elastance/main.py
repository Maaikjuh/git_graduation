"""
This script simulates the systemic circulation with a 0D model.
Left ventricle is modelled as a time varying elastance and the circulation
as a 3 element windkessel.
"""

from dolfin.cpp.common import info

from cvbtk import Dataset, TimeVaryingElastance, GeneralWindkesselModel, GeneralHemodynamicsPlot

# Load reference simulation as a Dataset
reference_file = 'model/examples/systemic_circulation/time_varying_elastance/reference_matlab_simulation.csv'
reference_data = Dataset(filename=reference_file)

# Give the number of the cycle in the reference data set to simulate and compare
cycle_to_sim = 5

# Determine the start and end times to simulate, and also the increment:
t0 = reference_data['time'][reference_data['cycle'] == cycle_to_sim].iat[0]
t = t0
dt = reference_data['time'][1] - reference_data['time'][0]
t_tot = reference_data['time'][reference_data['cycle'] == cycle_to_sim].iat[-1]
t_cycle = float(t_tot - t0 + dt)

# Plot and save the reference data to inspect that everything is working:
reference_plot = GeneralHemodynamicsPlot(reference_data)
reference_plot.plot(cycle=cycle_to_sim)
reference_plot.save('output/time_var_elastance_LV/reference_hemodynamics.png')

# Set the total blood volume
total_volume = 5000. # [ml]

# Input parameters (use same ones as for reference simulation)
# Create a dictionary of inputs for WindkesselModel.
inputs_wk = {'arterial_compliance':     18.5206,   # [ml/kPa]
             'arterial_resistance':     1.4180,    # [kPa.ms/ml]
             'arterial_resting_volume': 882.3,     # [ml]
             'peripheral_resistance':   140.3844,  # [kPa.ms/ml]
             'venous_compliance':       444.4940,  # [ml/kPa]
             'venous_resistance':       0.2836,    # [kPa.ms/ml]
             'venous_resting_volume':   3529.4}    # [ml]

# Create a dictionary of inputs for TimeVaryingElastance model for the LV
inputs_lv = {'elastance_pas':            0.0051,   # [kPa/ml] (float)
             'elastance_max':            0.2985,   # [kPa/ml] (float)
             'ventricle_resting_volume': 0.,       # [ml] (float)
             'time_cycle':               t_cycle,  # [ms] (float)
             'time_activation':          400.}      # [ms] (float)

# Create WindkesselModel object with given inputs:
wk = GeneralWindkesselModel(**inputs_wk)

# Create TimeVaryingElastance object with given inputs:
lv = TimeVaryingElastance(key='lv', **inputs_lv)

# Set windkessel initial state from reference data
wk.volume = {'ven': reference_data['vven'][reference_data['cycle'] == cycle_to_sim].iat[0],
             'art': reference_data['vart'][reference_data['cycle'] == cycle_to_sim].iat[0]}
wk.pressure = {'ven': reference_data['pven'][reference_data['cycle'] == cycle_to_sim].iat[0],
               'art': reference_data['part'][reference_data['cycle'] == cycle_to_sim].iat[0]}
wk.flowrate = {'art': reference_data['qart'][reference_data['cycle'] == cycle_to_sim].iat[0],
               'ven': reference_data['qven'][reference_data['cycle'] == cycle_to_sim].iat[0],
               'per': reference_data['qper'][reference_data['cycle'] == cycle_to_sim].iat[0]}

# LV initial state
lv.volume   = {'lv':  reference_data['vlv'][reference_data['cycle'] == cycle_to_sim].iat[0]}
lv.pressure = lv.compute_pressure(t)

# Inspect the parameters of the windkessel and lv model:
info(wk.parameters, True)
info(lv.parameters, True)

# Inspect the initialized state values of the windkessel and lv model:
print('The initial WK state is V = {}.'.format(wk.volume))
print('The initial WK state is p = {}.'.format(wk.pressure))
print('The initial WK state is q = {}.'.format(wk.flowrate))
print('The initial LV state is V = {}.'.format(lv.volume))
print('The initial LV state is p = {}.'.format(lv.pressure))

print('Simulating cycle {} from t = [{}, {}, {}] ms.'.format(cycle_to_sim, t0, dt, t_tot))

# One last step: create an output dataset and add the initial state to it:
dataset_keys = ['time', 'cycle', 'plv', 'part', 'pven', 'qart', 'qper', 'qven', 'vlv', 'vart', 'vven']
simulation_data = Dataset(keys=dataset_keys)
simulation_data.append(time=t, cycle=t//t_cycle+1,
                 plv=lv.pressure['lv'],
                 vlv=lv.volume['lv'],
                 vart=wk.volume['art'],
                 vven=wk.volume['ven'],
                 part=wk.pressure['art'],
                 pven=wk.pressure['ven'],
                 qart=wk.flowrate['art'],
                 qven=wk.flowrate['ven'],
                 qper=wk.flowrate['per'])

# The time loop is below:
while t < t_tot:
    # -------------------------------------------------------------------- #
    # t = n                                                                #
    # -------------------------------------------------------------------- #
    # Store a copy/backup of the state values at t = n:
    v_old = wk.volume
    q_old = wk.flowrate
    vlv_old = lv.volume

    # -------------------------------------------------------------------- #
    # time increment                                                       #
    # -------------------------------------------------------------------- #
    t += dt

    # -------------------------------------------------------------------- #
    # t = n + 1                                                            #
    # -------------------------------------------------------------------- #
    # Compute the new volumes with a simple forward Euler scheme:
    # noinspection PyUnresolvedReferences
    vlv_new = vlv_old['lv'] + dt*(q_old['ven'] - q_old['art'])
    vart_new = v_old['art'] + dt*(q_old['art'] - q_old['per'])
    vven_new = total_volume - vart_new - vlv_new

    # Assemble the values into a dictionary and set the new model state for the LV model
    v_new_lv = {'lv': vlv_new}
    lv.volume = v_new_lv

    # Assemble the values into a dictionary and set the new model state for the WK model
    v_new_wk = {'art': vart_new, 'ven': vven_new}
    wk.volume = v_new_wk

    # Compute new lv pressure from this new state and update the model state:
    p_new_lv = lv.compute_pressure(t)
    lv.pressure = p_new_lv

    # Compute new wk pressures from this new state and update the model state:
    p_new_wk = wk.compute_pressure()
    wk.pressure = p_new_wk

    # Compute new flow rates from WK model and update the state.
    p_boundary = {'in' : lv.pressure['lv'],
                  'out': lv.pressure['lv']}
    q_new = wk.compute_flowrate(p_boundary)
    wk.flowrate = q_new

    # Append all computed values to the output dataset:
    simulation_data.append(time=t, cycle=t//t_cycle+1,
                     plv=lv.pressure['lv'],
                     vlv=lv.volume['lv'],
                     vart=wk.volume['art'],
                     vven=wk.volume['ven'],
                     part=wk.pressure['art'],
                     pven=wk.pressure['ven'],
                     qart=wk.flowrate['art'],
                     qven=wk.flowrate['ven'],
                     qper=wk.flowrate['per'])

# Now, let's plot the hemodynamic relations of this new dataset:
simulation_plot = GeneralHemodynamicsPlot(simulation_data)
simulation_plot.plot()
simulation_plot.save('output/time_var_elastance_LV/simulation_hemodynamics.png')

# We can compare the figures using the reference plot as the reference.
reference_plot.compare_against(simulation_data, 'o', ms=2, markevery=2)
reference_plot.save('output/time_var_elastance_LV/comparison.png')

# To conclude, we can save the computed mock values to a CSV file.
simulation_data.save('output/time_var_elastance_LV/results.csv')