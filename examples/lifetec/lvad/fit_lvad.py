"""
Iteratively find optimum LVAD model coefficients to model the LVAD flow
from the LifeTEc experiments.
"""
from cvbtk import LifetecWindkesselModel, HeartMateII, TimeVaryingElastance, Dataset, mmHg_to_kPa, kPa_to_mmHg, \
    figure_make_up
from dolfin import info
import os
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

dir_out = 'output/fit_lvad'
if not os.path.exists(dir_out):
    os.makedirs(dir_out)


# Define inputs
def get_inputs():
    # Number of cycles, starting time and timestep (ms).
    time = {'n_cycles': 10,
            't_cycle': 500,
            't': 0,
            'dt': 1}

    # Estimate/tune hemodynamics from experiment if possible.
    MAP = mmHg_to_kPa(84.57)  # Mean arterial pressure [kPa]
    CO = 5.58 / 60  # Cardiac output [ml/ms]
    RC = 800.  # RC time constant [ms]
    frac = 0.96  # Rp/(Rp + Za) [-]

    # Compute some windkessel parameters from global hemodynamic parameters.
    Rtot = MAP / CO
    Zart = Rtot * (1 - frac)
    Rp = Rtot * frac
    Cart = RC / Rp

    # Rp = MAP/CO
    # Zart = Rp*frac
    # Cart = RC/Rp

    # Input parameters (use same ones as for reference simulation)
    # Create a dictionary of inputs for WindkesselModel.
    inputs_wk = {'arterial_compliance':  5.02678762781376,  # [ml/kPa]
                 'arterial_resistance':  2.43592903973729,  # [kPa.ms/ml]
                 'peripheral_resistance':  119.360522947127,  # [kPa.ms/ml]
                 'venous_resistance': 1.0,  # [kPa.ms/ml]
                 'venous_pressure': mmHg_to_kPa(20)}  # [kPa]

    inputs_lvad = {'frequency': 0,
                   'lvad_volume': 66.0,
                   'alpha_slope': 0.013864,
                   'alpha_intercept': 2.6103,
                   'beta_slope': -0.34292,
                   'beta_intercept': -5.8354}

    # Create a dictionary of inputs for TimeVaryingElastance model for the LV
    # V0 = 60.
    # Epas = 0.76/(114.15 - V0)
    # Emax = 15.05/(65.97 - V0)
    V0 = 57.87
    Epas = 0.0134
    Emax = 8 * Epas
    inputs_lv = {'elastance_pas': Epas,  # 0.0089,  # [kPa/ml] (float)
                 'elastance_max': Emax,  # 0.24,  # [kPa/ml] (float)
                 'ventricle_resting_volume': V0,  # [ml] (float)
                 'time_cycle': float(time['t_cycle']),  # [ms] (float)
                 'time_activation': 400.,  # 410.,  # [ms] (float)
                 'time_depolarization': 80.}  # 40.}  # [ms] (float)

    inputs = {'time': time,
              'windkessel': inputs_wk,
              'lv': inputs_lv,
              'lvad': inputs_lvad}
    return inputs


def lvad_model(x, a_slope, a_intercept, b_slope, b_intercept):
    frequency = x[0]/1000
    dp = x[1]
    return (a_slope*dp + a_intercept)*frequency + b_slope*dp + b_intercept


def load_mockdata(csvfile, simulation_time_array):
    # Load csv file as Dataset.
    data = Dataset(filename=csvfile)

    # Resample data.
    time = np.array(data['time'])*1000
    plv = mmHg_to_kPa(np.array(data['plv']))
    plv_resampled = np.interp(simulation_time_array, time, plv, period=500)

    return data, plv_resampled


def preprocess(inputs, verbose=False):
    # Create WindkesselModel object with given inputs:
    wk = LifetecWindkesselModel(**inputs['windkessel'])

    # Add LVAD.
    wk.lvad = HeartMateII(**inputs['lvad'])

    # Set windkessel initial state.
    wk.volume = {'art': 0}
    wk.volume = wk.compute_pressure()

    # Create TimeVaryingElastance object with given inputs.
    lv = TimeVaryingElastance(key='lv', **inputs['lv'])

    # Initial flowrates.
    wk.flowrate = wk.compute_flowrate(wk.pressure, lv.pressure)

    if verbose:
        info(wk.lvad.parameters, True)

        # Inspect the initialized state values of the windkessel and lv model:
        print('The initial WK state is V = {}.'.format(wk.volume))
        print('The initial WK state is p = {}.'.format(wk.pressure))
        print('The initial WK state is q = {}.'.format(wk.flowrate))
        print('The initial LV state is V = {}.'.format(lv.volume))
        print('The initial LV state is p = {}.'.format(lv.pressure))

        print('Simulating {} cycles.'.format(inputs['time']['n_cycles']))

    # One last step: create an output dataset and add the initial state to it:
    dataset_keys = ['time', 'cycle', 'plv', 'part', 'pven', 'qao', 'qper', 'qmv', 'qlvad', 'vlv', 'vart', 'vven']
    simulation_data = Dataset(keys=dataset_keys)
    simulation_data.append(time=inputs['time']['t'],
                           cycle=inputs['time']['t'] // inputs['time']['t_cycle'] + 1,
                           plv=lv.pressure['lv'],
                           vlv=lv.volume['lv'],
                           vart=wk.volume['art'],
                           vven=wk.volume['ven'],
                           part=wk.pressure['art'],
                           pven=wk.pressure['ven'],
                           qao=wk.flowrate['ao']*1000,
                           qmv=wk.flowrate['mv']*1000,
                           qper=wk.flowrate['per']*1000,
                           qlvad=wk.flowrate.get('lvad', 0.0)*1000)

    return wk, lv, simulation_data


# Routine that runs a mock simulation, for given inputs, lvad parameters and experiment data.
def simulate(wk, lv, simulation_data, inputs, lifetec_data_file=None, mock_circulation=False):

    n_cycles = inputs['time']['n_cycles']
    t_cycle = inputs['time']['t_cycle']
    t = inputs['time']['t']
    dt = inputs['time']['dt']

    simulation_time_array = np.arange(t, n_cycles * t_cycle, dt)

    # Load experiment data.
    if lifetec_data_file is not None:
        mockdata, mock_plv = load_mockdata(lifetec_data_file, simulation_time_array)
    elif not mock_circulation:
        mockdata = None
    else:
        raise ValueError('No file specified with mockdata.')

    # The time loop is below:
    for idx, t in enumerate(simulation_time_array):
        # -------------------------------------------------------------------- #
        # t = n                                                                #
        # -------------------------------------------------------------------- #
        # Store a copy/backup of the state values at t = n:
        v_old = wk.volume
        q_old = wk.flowrate
        vlv_old = lv.volume

        # # -------------------------------------------------------------------- #
        # # time increment                                                       #
        # # -------------------------------------------------------------------- #
        # t += dt

        # -------------------------------------------------------------------- #
        # t = n + 1                                                            #
        # -------------------------------------------------------------------- #
        # Compute the new volumes with a simple forward Euler scheme:
        # noinspection PyUnresolvedReferences
        vlv_new = vlv_old['lv'] + dt * (q_old['mv'] - q_old['ao'] - q_old.get('lvad', 0.))
        vart_new = v_old['art'] + dt * (q_old['ao'] - q_old['per'] + q_old.get('lvad', 0.))

        # Assemble the values into a dictionary and set the new model state for the LV model
        v_new_lv = {'lv': vlv_new}
        lv.volume = v_new_lv

        # Assemble the values into a dictionary and set the new model state for the WK model
        v_new_wk = {'art': vart_new}
        wk.volume = v_new_wk

        if not mock_circulation:
            # Compute new lv pressure from this new state and update the model state:
            p_new_lv = lv.compute_pressure(t)
        else:
            p_new_lv = {'lv': mock_plv[idx]}

        lv.pressure = p_new_lv

        # Compute new wk pressures from this new state and update the model state:
        p_new_wk = wk.compute_pressure()
        wk.pressure = p_new_wk

        # Compute new flow rates from WK model and update the state.
        wk.flowrate = wk.compute_flowrate(wk.pressure, lv.pressure)

        # Append all computed values to the output dataset:
        simulation_data.append(time=t, cycle=t // t_cycle + 1,
                               plv=lv.pressure['lv'],
                               vlv=lv.volume['lv'],
                               vart=wk.volume['art'],
                               vven=wk.volume['ven'],
                               part=wk.pressure['art'],
                               pven=wk.pressure['ven'],
                               qao=wk.flowrate['ao']*1000,
                               qmv=wk.flowrate['mv']*1000,
                               qper=wk.flowrate['per']*1000,
                               qlvad=wk.flowrate.get('lvad', 0.0)*1000)

    return mockdata


# Routine that synchronizes dp with qlvad.
def synchronize(sig1, sig2, search_region_size=60, verbose=False, dir_out=''):
    """
    Shifts sig2 so that it is synchronized with sig1.
    Based on least squares.

    Args:
        sig1 (np.array): signal 1.
        sig2 (np.array): signal 2, to be shifted. Must be of same length as sig1.
        search_region_size(int, optional): number of shifts to evaluate.
        verbose (bool, optional): Plots a figure that shows the shift.
        dir_out (str, optional): If verbose is True, a figure is saved in the dir_out directory.

    Returns:
        sig2_shifted (np.array): signal 2 shifted, so that it overlaps most with signal 1.
    """
    # Normalize the 2 signals.
    sig1_norm = (sig1 - min(sig1)) / (max(sig1 - min(sig1)) + 1e-14)
    sig2_norm = (sig2 - min(sig2)) / (max(sig2 - min(sig2)) + 1e-14)

    # Shift the signal over a small domain, and minimize the sum of squared differences.
    # Initially the shift is zero.
    shift_best = 0
    ssqd_lowest = np.sum((sig2_norm - sig1_norm) ** 2)

    # The search domain for the optimum shift is located around the shift where
    # both minima (of dp and qlvad) coincide.
    shift_guess = np.argmin(sig2) - np.argmin(sig1)
    for shift in range(-int(search_region_size/2) + shift_guess,
                        int(search_region_size/2) + shift_guess):
        sig2_norm_shifted = np.concatenate((sig2_norm[shift:], sig2_norm[0:shift]))
        ssqd = np.sum((sig2_norm_shifted - sig1_norm) ** 2)
        if ssqd < ssqd_lowest:
            ssqd_lowest = ssqd * 1
            shift_best = shift * 1
    shift = shift_best * 1
    sig2_shifted = np.concatenate((sig2[shift:], sig2[0:shift]))
    sig2_norm_shifted = np.concatenate((sig2_norm[shift:], sig2_norm[0:shift]))

    if verbose:
        plt.figure()
        plt.plot(sig1_norm)
        plt.plot(sig2_norm_shifted)
        plt.legend(('sig1 (rel)', 'sig2 (rel)'))
        plt.savefig(os.path.join(dir_out, 'synchronization.png'))
    return sig2_shifted


def main():
    # 1. Load inputs.
    inputs = get_inputs()

    # 2. Choose initial LVAD fitting coefs.
    popt = [0.01386385,  2.61032402, -0.34291504, -5.83537101]
    popt = [ 0.00435981,  0.99412831, -0.0971561,  -3.34863522]

    # 3. Specify pump speeds.
    pump_speeds = [7500, 8500, 9500, 10500]

    # 4. While not converged.
    eps = 1.
    n_iter = 0
    max_iter = 20
    verbose = True
    while eps > 1e-5 and n_iter < max_iter:
        print('Iteration {}'.format(n_iter))

        n_iter += 1
        all_lvad_data = []

        # Set fitting coefs lvad model.
        lvad_parameters = {'alpha_slope': popt[0],
                           'alpha_intercept': popt[1],
                           'beta_slope': popt[2],
                           'beta_intercept': popt[3]}

        inputs['lvad'].update(lvad_parameters)

        popt_old = popt*1

        # 5. Loop over LVAD speeds.
        for speed in pump_speeds:
            # Load experiment data.
            lifetec_data_file = 'model/cvbtk/data/hemo_lifetec_hc1_{}_rpm.csv'.format(speed)

            # Set LVAD speed.
            inputs['lvad']['frequency'] = speed/1000

            # Preprocess.
            wk, lv, simulation_data = preprocess(inputs, verbose=False)

            # Run a simulation for the given speed and with the estimates coefs.
            mockdata = simulate(wk, lv, simulation_data, inputs, lifetec_data_file=lifetec_data_file, mock_circulation=True)

            # Extract results of last cycle.
            cycle = int(max(max(simulation_data['cycle']) - 1, min(simulation_data['cycle'])))
            results = simulation_data[simulation_data['cycle'] == int(cycle)].copy(deep=True)

            # Extract pressure head from simulation.
            phead = kPa_to_mmHg(np.array(results['part']) - np.array(results['plv']))

            # Resample the simulation data.
            time_mock = np.array(mockdata['time']) * 1000
            time_sim = np.array(results['time'])
            phead = np.interp(time_mock, time_sim, phead, period=inputs['time']['t_cycle'])

            # Synchronize dp with qlvad.
            qlvad = np.array(mockdata['qlvad'])
            qlvad_shifted = synchronize(-phead, qlvad, verbose=verbose, dir_out=dir_out, search_region_size=60)

            # Collect data:
            # Pump speed, pressure, flow.
            speed_array = np.ones_like(phead)*speed
            lvad_data = np.vstack([speed_array, phead, qlvad_shifted])
            all_lvad_data.append(lvad_data)

        # Collect data of all pump speeds
        all_lvad_data = np.hstack(all_lvad_data)

        # Fit the data to find new fitting coefs.
        all_speeds, all_pressures, all_flows = all_lvad_data
        xdata = np.vstack((all_speeds, all_pressures))
        ydata = np.array(all_flows)
        popt, pcov = curve_fit(lvad_model, xdata, ydata)

        # Plot the fit after each nth iteration.
        print('Fitted parameters:\n', popt)

        # Plot.
        if verbose:
            fontsize = 16
            plt.close('all')
            plt.figure()
            for ii, speed in enumerate(np.unique(all_speeds)):
                idx = np.where(all_speeds == speed)[0]
                plt.plot(all_flows[idx],
                         all_pressures[idx],
                         'C{}--'.format(ii),
                         linewidth=fontsize/8)

                idx_sort = np.argsort(xdata[1, idx])
                plt.plot(lvad_model(xdata[:, idx], *popt)[idx_sort],
                         xdata[1, idx][idx_sort],
                         'C{}'.format(ii),
                         label='{}'.format(int(speed)),
                         linewidth=fontsize/6)

            plt.grid()
            figure_make_up(title='LVAD model fit',
                           xlabel='Flow [L/min]',
                           ylabel='Pressure head [mmHg]',
                           fontsize=fontsize)

            plt.savefig(os.path.join(dir_out, 'lvad_fit.png'))

        # Compute convergence criterium.
        eps = sum(abs(popt - popt_old))
        print('Eps:', eps)


if __name__ == '__main__':
    main()