"""
This script simulates the lifetec experiment with a 0D model.
Left ventricle is modelled as a time varying elastance and the circulation
as a constant preload and a 3 element afterload.

This script was used to tune the afterload model parameters,
setting mock_circulation to True.
"""

from cvbtk import HemodynamicsPlot, kPa_to_mmHg, mmHg_to_kPa, save_dict_to_csv, figure_make_up

import numpy as np
import matplotlib.pyplot as plt
import os
from examples.lifetec.lvad.fit_lvad import preprocess, simulate, synchronize
import scipy.stats

plt.close('all')

# Specify pump speed [rpm].
pump_speed = 9500

# Specify whether we should prescribe the measured LV pressures (True)
# or use a time-varying elastance model for the LV (False).
mock_circulation = True

# If True, LVAD flow is restricted.
clamp_pump_outlet = pump_speed < 0.001

# Output directory.
dir_out = 'output/time_var_elastance_lifetec_{}_rpm_mock_{}'.format(int(pump_speed), mock_circulation)


# Define inputs
def get_inputs(pump_speed=0, clamp_pump_outlet=True):
    """
    Args:
        pump_speed: LVAD speed in rpm.
        clamp_pump_outlet (bool): If True, LVAD flow is restricted.

    Returns:
        Dictionary with inputs.
    """

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
                 'venous_resistance': float(1),  # [kPa.ms/ml]
                 'venous_pressure': mmHg_to_kPa(10)}  # [kPa]

    # Specify LVAD model fitting parameters.
    popt = [0.00435981,  0.99412831, -0.0971561,  -3.34863522]

    # popt = [0.013864, 2.6103, -0.34292, -5.8354]  # Schampaert

    if clamp_pump_outlet:
        # Restrict pump flow by setting all model parameters to zero.
        popt = [0, 0, 0, 0]

    # Set LVAD parameters.
    inputs_lvad = {'frequency': float(pump_speed/1000), # [krpm]
                   'lvad_volume': 66.0,
                   'alpha_slope': float(popt[0]),
                   'alpha_intercept': float(popt[1]),
                   'beta_slope': float(popt[2]),
                   'beta_intercept': float(popt[3])}

    # Create a dictionary of inputs for TimeVaryingElastance model for the LV
    # V0 = 60.
    # Epas = 0.76/(114.15 - V0)
    # Emax = 15.05/(65.97 - V0)
    V0 = 60.
    Epas = 0.067
    Emax = 30 * Epas
    inputs_lv = {'elastance_pas': Epas,  # 0.0089,  # [kPa/ml] (float)
                 'elastance_max': Emax,  # 0.24,  # [kPa/ml] (float)
                 'ventricle_resting_volume': V0,  # [ml] (float)
                 'time_cycle': float(time['t_cycle']),  # [ms] (float)
                 'time_activation': 380.,  # 410.,  # [ms] (float)
                 'time_depolarization': 80.}  # 40.}  # [ms] (float)

    inputs = {'time': time,
              'windkessel': inputs_wk,
              'lv': inputs_lv,
              'lvad': inputs_lvad}
    return inputs


def postprocess(simulation_data, mockdata, cycle=None, dir_out=''):
    # Load reduced dataset.
    if cycle is None:
        # Select last cycle.
        cycle = int(max(max(simulation_data['cycle']) - 1, min(simulation_data['cycle'])))
    results = simulation_data[simulation_data['cycle'] == int(cycle)].copy(deep=True)

    # Print CO and MAP
    CO_no_lvad= np.mean(results['qao']) * 60/1000  # ml/s to L/min
    CO_tot = CO_no_lvad + np.mean(results['qlvad']) * 60/1000
    MAP = np.mean(results['part']) * 7.5 # To mmHg
    print('CO (no LVAD) = {:3.2f} L/min \nCO_tot = {:3.2f} L/min \nMAP = {:4.2f} mmHg'.format(
        CO_no_lvad, CO_tot, MAP))

    # Hemodynamic plot of latest cycle.
    simulation_plot = HemodynamicsPlot(results)
    simulation_plot.plot()
    simulation_plot.save(os.path.join(dir_out, 'simulation_hemodynamics_cycle_{}.png'.format(cycle)))

    # Hemodynamic plot of all cycles.
    simulation_plot = HemodynamicsPlot(simulation_data)
    simulation_plot.plot()
    simulation_plot.save(os.path.join(dir_out, 'simulation_hemodynamics.png'))
    plt.close('all')

    # Extract some signals for further postprocessing.
    time_mock = np.array(mockdata['time']) * 1000
    part_mock = np.array(mockdata['part'])
    plv_mock = np.array(mockdata['plv'])
    time_sim = np.array(results['time'])
    qlvad_exp = np.array(mockdata['qlvad'])

    # Compare simulation and experiment pressures.
    plt.figure(figsize=(6.5 * 2, 5 * 1))
    plt.subplot(1, 2, 1)
    fontsize = 16
    plt.plot(time_mock, plv_mock, 'C0-', label='$p_{lv}$ experiment', linewidth=fontsize/6)
    plt.plot(time_mock, part_mock, 'C1-', label='$p_{art}$ experiment', linewidth=fontsize/6)
    # plt.plot(np.array(results['time']) - np.array(results['time'])[0], kPa_to_mmHg(np.array(results['plv'])), 'C0--',
    #          label='$p_{lv}$ simulation', linewidth=fontsize/6)
    plt.plot(np.array(results['time']) - np.array(results['time'])[0], kPa_to_mmHg(np.array(results['part'])), 'C1--',
             label='$p_{art}$ simulation', linewidth=fontsize/6)
    figure_make_up(title='Tuning of afterload model to REF_0', xlabel='Time [ms]', ylabel='Pressure [mmHg]',
                   fontsize=fontsize)
    plt.grid()
    plt.ylim(6.3947303091905185, 117.9297747471814)
    plt.savefig(os.path.join(dir_out, 'comparison_part.png'),  dpi=300, bbox_inches="tight")

    # Compute pressure head.
    phead = kPa_to_mmHg(np.array(results['part']) - np.array(results['plv']))

    # Plot pressure head.
    plt.figure()
    plt.plot(np.array(results['time']) - np.array(results['time'])[0], phead)
    plt.xlabel('Time [ms]')
    plt.ylabel('Pressure head [mmHg]')
    plt.title('Pressure head across LVAD')
    plt.savefig(os.path.join(dir_out, 'pressure_head.png'))

    # Fit phead ~ qlvad.
    # Resample the experiment data.
    qlvad_exp_res = np.interp(time_sim, time_mock, qlvad_exp, period=500)
    qlvad_exp_sync_phead = synchronize(-phead, qlvad_exp_res, verbose=True, dir_out=dir_out)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(phead, qlvad_exp_sync_phead)
    print('R-squared:', r_value ** 2)

    # Plot fit.
    plt.figure()
    plt.plot(phead, qlvad_exp_sync_phead)
    dp_fit = np.linspace(min(phead), max(phead), 10)
    plt.plot(dp_fit, slope * dp_fit + intercept, '--')
    plt.legend(('Data', 'Fit'))
    plt.xlabel('Pressure head [mmHg]')
    plt.ylabel('LVAD flow [L/min]')
    plt.savefig(os.path.join(dir_out, 'dp_qlvad.png'))
    plt.close('all')

    # Plot measured LVAD flow vs model flow.
    qlvad_model = np.array(results['qlvad']) * 60 / 1000  # ml/s to L/min
    qlvad_exp_sync_model = synchronize(qlvad_model, qlvad_exp_res)
    plt.figure()
    plt.plot(time_sim-time_sim[0], qlvad_exp_sync_model, label='Experiment')
    plt.plot(time_sim-time_sim[0], qlvad_model, '--', label='Model')
    plt.xlabel('Time [ms]')
    plt.ylabel('Flow [L/min]')
    plt.title('LVAD flow')
    plt.legend()
    plt.savefig(os.path.join(dir_out, 'lvad_flow.png'))


def main():
    # Load default inputs.
    inputs = get_inputs(pump_speed=pump_speed, clamp_pump_outlet=clamp_pump_outlet)

    # Save inputs to a csv file (as we might later wonder what inputs we have used).
    print('Saving inputs to {} ...'.format(os.path.join(dir_out, 'inputs.csv')))
    save_dict_to_csv(inputs, os.path.join(dir_out, 'inputs.csv'))

    # Preprocess.
    wk, lv, simulation_data = preprocess(inputs, verbose=True)

    # Lifetec data file for pump speed.
    lifetec_data_file = 'model/cvbtk/data/hemo_lifetec_hc1_{}_rpm.csv'.format(int(inputs['lvad']['frequency']*1000))

    # Run a simulation for the given speed and with the estimates coefs.
    mockdata = simulate(wk, lv, simulation_data, inputs,
                        lifetec_data_file=lifetec_data_file, mock_circulation=mock_circulation)

    # We can save the computed mock values to a CSV file.
    simulation_data.save(os.path.join(dir_out, 'results.csv'))

    # Postprocess.
    postprocess(simulation_data, mockdata, dir_out=dir_out)


if __name__ == '__main__':
    main()