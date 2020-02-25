# -*- coding: utf-8 -*-
"""
This script provides an example on how to simulate and post-process global
hemodynamic quantities in a mock circulation loop using pre-computed cavity
pressures obtained from SEPRAN.
"""
from dolfin.cpp.common import info

import cvbtk.resources
from cvbtk import Dataset, HemodynamicsPlot, WindkesselModel

#def main():
# As a first step, load benchmark SEPRAN results from the resources module:
reference_data = cvbtk.resources.reference_hemodynamics()

# Plot and save the reference data to inspect that everything is working:
reference_plot = HemodynamicsPlot(reference_data)
reference_plot.plot(cycle=1)
reference_plot.save('output/reference_data.png')

# We can inspect what parameters are passable to WindkesselModel:
info(WindkesselModel.default_parameters(), True)

# Create a dictionary of inputs for WindkesselModel using the same values as
# from the reference SEPRAN results.
inputs = {'arterial_compliance': 25.0,
          'arterial_resistance': 10.0,
          'arterial_resting_volume': 500.0,
          'peripheral_resistance': 120.0,
          'total_volume': 5000.0,
          'venous_compliance': 600.0,
          'venous_resistance': 5.0,
          'venous_resting_volume': 3000.0}

# Create WindkesselModel object with these inputs:
wk = WindkesselModel(**inputs)

# Inspecting the parameters again show that they were accepted:
info(wk.parameters, True)

# Inspect the current state values of the windkessel model:
print('The current state is V = {}.'.format(wk.volume))
print('The current state is p = {}.'.format(wk.pressure))
print('The current state is q = {}.'.format(wk.flowrate))

# They are set to zero, so we can initialize them using the reference data:
wk.volume = {'ven': reference_data['vven'][0],
             'art': reference_data['vart'][0]}
wk.pressure = {'ven': reference_data['pven'][0],
               'art': reference_data['part'][0]}
wk.flowrate = {'ao': reference_data['qao'][0]/1000,
               'mv': reference_data['qmv'][0]/1000,
               'per': reference_data['qper'][0]/1000}

# Inspect the initialized state values of the windkessel model:
print('The initialized state is V = {}.'.format(wk.volume))
print('The initialized state is p = {}.'.format(wk.pressure))
print('The initialized state is q = {}.'.format(wk.flowrate))

# Determine the start and end times to simulate, and also the increment:
t = reference_data['time'][0]
t0 = t
dt = reference_data['time'][1] - t
t1 = reference_data['time'][reference_data['cycle'] == 1].iat[-1]
print('Simulating cycle from t = [{}, {}, {}] ms.'.format(t0, dt, t1))

# For this example we will prescribe cavity pressures from the reference
# data and compute the resulting windkessel state and cavity volume.
# To do this, create an iterator of the cavity pressures ...
plv = reference_data['plv'][reference_data['cycle'] == 1].iteritems()

# ... and set the initial state of our mock left ventricle model:
lv = {'p': {'lv': next(plv)[1]}, 'v': {'lv': reference_data['vlv'][0]}}
print('The initial LV state is {}.'.format(lv))

# One last step: create an output dataset and add the initial state to it:
mock_data = Dataset(keys=reference_data.keys())
mock_data.append(time=t, cycle=1,
                 plv=lv['p']['lv'],
                 vlv=lv['v']['lv'],
                 vart=wk.volume['art'],
                 vven=wk.volume['ven'],
                 part=wk.pressure['art'],
                 pven=wk.pressure['ven'],
                 qao=wk.flowrate['ao'],
                 qmv=wk.flowrate['mv'],
                 qper=wk.flowrate['per'])

# The time loop is below:
while t < t1:
    # -------------------------------------------------------------------- #
    # t = n                                                                #
    # -------------------------------------------------------------------- #
    # Store a copy/backup of the state values at t = n:
    v_old = wk.volume
    q_old = wk.flowrate
    vlv_old = lv['v'].copy()

    # -------------------------------------------------------------------- #
    # time increment                                                       #
    # -------------------------------------------------------------------- #
    t += dt

    # -------------------------------------------------------------------- #
    # t = n + 1                                                            #
    # -------------------------------------------------------------------- #
    # Compute the new volumes with a simple forward Euler scheme:
    # noinspection PyUnresolvedReferences
    vlv_new = vlv_old['lv'] + dt*(q_old['mv'] - q_old['ao'])
    vart_new = v_old['art'] + dt*(q_old['ao'] - q_old['per'])
    vven_new = inputs['total_volume'] - vart_new - vlv_new

    # Assemble the values into a dictionary and set the new model state:
    v_new = {'art': vart_new, 'ven': vven_new}
    wk.volume = v_new

    # Normally the finite element routine would be called here:
    plv_new = next(plv)[1]

    lv['p']['lv'] = plv_new
    lv['v']['lv'] = vlv_new

    # Compute new pressures from this new state and update the model state:
    p_new = wk.compute_pressure(v_new)
    wk.pressure = p_new

    # Compute new flow rates and update the state.
    q_new = wk.compute_flowrate(p_new, lv['p'])
    wk.flowrate = q_new

    # Append all computed values to the output dataset:
    mock_data.append(time=t, cycle=1,
                     plv=lv['p']['lv'],
                     vlv=lv['v']['lv'],
                     vart=wk.volume['art'],
                     vven=wk.volume['ven'],
                     part=wk.pressure['art'],
                     pven=wk.pressure['ven'],
                     qao=wk.flowrate['ao']*1000,
                     qmv=wk.flowrate['mv']*1000,
                     qper=wk.flowrate['per']*1000)

# Now, let's plot the hemodynamic relations of this new dataset:
mock_plot = HemodynamicsPlot(mock_data)
mock_plot.plot()
mock_plot.save('output/mock_data.png')

# We can compare the figures using the reference plot as the reference.
reference_plot.compare_against(mock_data, 'o', ms=2, markevery=2)
reference_plot.save('output/comparison.png')

# To conclude, we can save the computed mock values to a CSV file.
mock_data.save('output/results.csv')


#if __name__ == '__main__':
  #  main()
