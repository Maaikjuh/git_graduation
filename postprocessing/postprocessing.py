# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 12:54:49 2018

Postprocessing module.

@author: Hermans
"""
import os

# --------------------------------------------------------------------------- #
# Specify paths of relevant directories.
# --------------------------------------------------------------------------- #
# Root directory (for easily specifying paths).
#ROOT_DIRECTORY = r'E:\Graduation project'
# Automatically deduce root directory from the path to this file:
ROOT_DIRECTORY = os.path.join(
        os.path.abspath(__file__).split('\\Graduation_project\\')[0],
        'Graduation_project')

# Path to LifeTec data (curves.hdf5 and hemo.csv files)
# (e.g. E:\Graduation project\LifeTec PhysioHeart data)
#LIFETEC_DATA_PATH = os.path.join(ROOT_DIRECTORY, 'LifeTec PhysioHeart data')

# Path to the directory containing the output of the simulations. 
# (e.g. E:\Graduation project\output)
SIMULATION_DATA_PATH = os.path.join(ROOT_DIRECTORY, r'Results_Tim')

# Path to the cvbtk directory.
# (e.g. E:\Graduation project\model\cvbtk)
CVBTK_PATH = os.path.join(ROOT_DIRECTORY,'git_graduation_project\cvbtk')

# Path to the postprocessing_BiV directory (automatically)
# (is a subdirectory of the current file location).
POSTPROCESSING_BIV_PATH = os.path.join(os.path.split(
                                       os.path.abspath(__file__))[0], 
                                       'postprocessing_BiV')

# Check whether paths exist.
check_paths = [ROOT_DIRECTORY,
#               LIFETEC_DATA_PATH,
               SIMULATION_DATA_PATH,
               CVBTK_PATH,
               POSTPROCESSING_BIV_PATH]
for path in check_paths:
    if not os.path.exists(path):
        raise ValueError('Specified path "{}" does not exists. Check the specified paths in {}.'
                         .format(path, os.path.abspath(__file__)))
# --------------------------------------------------------------------------- #

# Add relevant paths to directory to make imports possible.
import sys
sys.path.append(CVBTK_PATH)
sys.path.append(POSTPROCESSING_BIV_PATH)

from dataset import Dataset  # in cvbtk direcoty

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec

import scipy.interpolate
from sklearn.cluster import KMeans
from sklearn import linear_model

import h5py
import glob
import itertools

lines = ["-","--","-.",":"]
linecycler = itertools.cycle(lines)

def get_paths():
    return {'LIFETEC_DATA_PATH': LIFETEC_DATA_PATH,
            'SIMULATION_DATA_PATH': SIMULATION_DATA_PATH,
            'CVBTK_PATH': CVBTK_PATH,
            'POSTPROCESSING_BIV_PATH': POSTPROCESSING_BIV_PATH}

def segment_markers():
    return ['v', '^', 'd','x', '+', '*']

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
    # both minima coincide.
    shift_guess = np.argmin(sig2) - np.argmin(sig1)
    for shift in range(-int(search_region_size/2) + shift_guess,
                        int(search_region_size/2) + shift_guess):
        sig2_norm_shifted = np.concatenate((sig2_norm[shift:], sig2_norm[0:shift]))
        ssqd = np.sum((sig2_norm_shifted - sig1_norm) ** 2)
        if ssqd < ssqd_lowest:
            ssqd_lowest = ssqd * 1
            shift_best = shift * 1
    shift = shift_best * 1
    if shift < 0:
        shift += len(sig2)
    i_sync = np.array(list(range(shift, len(sig2))) + list(range(0, shift)))
    sig2_shifted = sig2[i_sync] 
    sig2_norm_shifted = sig2_norm[i_sync]

    if verbose:
        plt.figure()
        plt.plot(sig1_norm)
        plt.plot(sig2_norm_shifted)
        plt.legend(('sig1 (rel)', 'sig2 (rel)'))
        plt.savefig(os.path.join(dir_out, 'synchronization.png'))
        
    return sig2_shifted, i_sync


def kPa_to_mmHg(p_kPa):
    """
    Convert pressures from kPa to mmHg.

    Args:
        p_kPa: Pressure in kPa.

    Returns:
        Pressure in mmHg.
    """
    conversion_factor = 133.322387415
    return p_kPa*1000/conversion_factor

def mmHg_to_kPa(p_mmHg):
    """
    Convert pressures from mmHg to kPa.

    Args:
        p_mmHg: Pressure in mmHg.

    Returns:
        Pressure in kPa.
    """
    conversion_factor = 133.322387415
    return p_mmHg/1000*conversion_factor

class HemodynamicsPlot(object):
    """
    High-level interface between :class:`~cvbtk.Dataset` and :mod:`matplotlib`
    that plots the pressure vs. time, volume vs. time, flowrate vs. time, and
    pressure vs. volume curves.

    Args:
        dataset: Dataset to create the figure from.
    """
    def __init__(self, dataset):
        # Store the dataset.
        self._df = dataset

        # Create an empty figure for plotting.
        self._fig = plt.figure()

        # Create a set of empty axes for plotting.
        gs = GridSpec(3, 2)
        _pt = plt.subplot(gs[0, 0])
        _vt = plt.subplot(gs[1, 0], sharex=_pt)
        _qt = plt.subplot(gs[2, 0], sharex=_pt)
        _pv = plt.subplot(gs[:, 1])
        self._ax = {'pt': _pt, 'vt': _vt, 'qt': _qt, 'pv': _pv}

        # Remove vertical spacing from the three individual axes.
        gs.update(hspace=0.15, wspace=0.30)

        # Remove x-axis labels and ticks from redundant axes.
        self._ax['pt'].xaxis.set_visible(False)
        self._ax['vt'].xaxis.set_visible(False)

        # Set axis labels.
        self._ax['qt'].set_xlabel('Time [ms]')
        self._ax['qt'].set_ylabel('Flowrate [ml/s]')
        self._ax['pt'].set_ylabel('Pressure [mmHg]')
        self._ax['vt'].set_ylabel('Volume [ml]')

        self._ax['pv'].set_xlabel('Volume [ml]')
        self._ax['pv'].set_ylabel('Pressure [mmHg]')

        # Set the global title.
        self._fig.suptitle('Hemodynamic Relations')

        # Remove the right and top spines.
        [ax.spines['top'].set_visible(False) for _, ax in self._ax.items()]
        [ax.spines['right'].set_visible(False) for _, ax in self._ax.items()]

    def compare_against(self, dataset, *args, **kwargs):
        """
        Draw additional curves on the existing figure for visual comparison.

        Args:
            dataset: Additional dataset to draw curves from.
            *args: Arbitrary positional arguments for plotting parameters.
            **kwargs: Arbitrary keyword arguments for plotting parameters.
        """
        # TODO Check that a plot has been already created.

        # Make the pressure-time plot.
        self._ax['pt'].plot(dataset['time'], kPa_to_mmHg(dataset['plv']), *args, **kwargs)
        self._ax['pt'].plot(dataset['time'], kPa_to_mmHg(dataset['pven']), *args, **kwargs)
        self._ax['pt'].plot(dataset['time'], kPa_to_mmHg(dataset['part']), *args, **kwargs)

        # Make the volume-time plot.
        self._ax['vt'].plot(dataset['time'], dataset['vlv'], *args, **kwargs)
        # self._ax['vt'].plot(dataset['time'], dataset['vven'], *args, **kwargs)
        # self._ax['vt'].plot(dataset['time'], dataset['vart'], *args, **kwargs)

        # Make the flowrate-time plot.
        self._ax['qt'].plot(dataset['time'], dataset['qmv'], *args, **kwargs)
        self._ax['qt'].plot(dataset['time'], dataset['qao'], *args, **kwargs)
        self._ax['qt'].plot(dataset['time'], dataset['qper'], *args, **kwargs)

        # Make the pressure-volume plot.
        # Each cycle (if multiple) will get its own color.
        for c in dataset['cycle'].unique():
            _df = dataset[dataset['cycle'] == int(c)]
            self._ax['pv'].plot(_df['vlv'], kPa_to_mmHg(_df['plv']), *args, **kwargs)

    def plot(self, cycle=None, legend=True):
        """
        Plot the defined hemodynamic relations for output.

        Args:
            cycle (optional): Filter and plot results for this specific cycle.
            legend (optional): Enables (default) or disables the legend.
        """
        # The cycle keyword argument can filter the results to a specific cycle.
        if cycle:
            df = self._df[self._df['cycle'] == int(cycle)]
            time = df['time'] - min(df['time'])
        else:
            df = self._df

        # Make the pressure-time plot.
        self._ax['pt'].plot(time, kPa_to_mmHg(df['plv']), label='Cavity')
        self._ax['pt'].plot(time, kPa_to_mmHg(df['pven']), label='Venous')
        self._ax['pt'].plot(time, kPa_to_mmHg(df['part']), label='Arterial')

        # Make the volume-time plot.
        self._ax['vt'].plot(time, df['vlv'], label='Cavity')
        # self._ax['vt'].plot(df['time'], df['vven'], label='Venous')
        # self._ax['vt'].plot(df['time'], df['vart'], label='Arterial')

        # Make the flowrate-time plot.
        self._ax['qt'].plot(time, df['qmv'], label='Mitral')
        self._ax['qt'].plot(time, df['qao'], label='Aortic')
        self._ax['qt'].plot(time, df['qper'], label='Peripheral')
        if 'qlvad' in df.keys():
            self._ax['qt'].plot(time, df['qlvad'], label='LVAD')

        # Make the pressure-volume plot.
        # Each cycle (if multiple) will get its own color.
        for c in df['cycle'].unique():
            _df = df[df['cycle'] == int(c)]
            self._ax['pv'].plot(_df['vlv'], kPa_to_mmHg(_df['plv']), label=str(c))

        # Add the legends for the three individual plots, if needed to.
        if legend:
            self._ax['pt'].legend(loc=2, fontsize=7) #, title='Pressure')
            self._ax['pt'].get_legend().get_title().set_fontsize('7')
            self._ax['vt'].legend(loc=2, fontsize=7) #, title='Volume')
            self._ax['vt'].get_legend().get_title().set_fontsize('7')
            self._ax['qt'].legend(loc=2, fontsize=7) #, title='Flowrate')
            self._ax['qt'].get_legend().get_title().set_fontsize('7')

        # The pressure-volume loop always has a legend if multiple are plotted.
        if not cycle and len(df['cycle'].unique()) > 1:
            self._ax['pv'].legend(loc=2, fontsize=7, title='Cycle')
            self._ax['pv'].get_legend().get_title().set_fontsize('7')

    def plot_function(self):
        """
        Plot the ventricle function in terms of stroke volume, maximum pressure,
        and work for every cycle.
        """
        # Find the number of the first and last cycle.
        start = min(self._df['cycle'])
        stop = max(self._df['cycle'])
        all_cycles = list(range(start, stop))

        # Extract SV and maximum pressure of the systemic circulation.
        strokevolumes = []
        pmax = []
        work = []
        cycles = []
        for c in all_cycles:
            df = self._df[self._df['cycle'] == int(c)]
            try:
                # Find begin phase 2.
                idx_ed = min(np.where(df['phase'].values == 2)[0])
                # Find begin phase 4.
                idx_ee = min(np.where(df['phase'].values == 4)[0])
                
            except ValueError:
                continue

            cycles.append(c)

            # Calculate SV.
            volume = df['vlv'].values
            sv = volume[idx_ed] - volume[idx_ee]
            strokevolumes.append(sv)

            # Maxmimum pressure.
            pressure = df['plv'].values
            pmax.append(max(pressure))

            # Work (surface of pressure-volume loop).
            work_cycle = 0.
            for ii in range(1, len(volume)):
                work_cycle += (pressure[ii - 1] + pressure[ii]) / 2 * (volume[ii - 1] - volume[ii]) / 1000  # in Joule.
            work.append(work_cycle)

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.plot(cycles, kPa_to_mmHg(np.array(pmax)))
        plt.title('Maximum LV pressure')
        plt.xlabel('Cardiac cycle [-]')
        plt.ylabel('Pressure [mmHg]')

        plt.subplot(1, 3, 2)
        plt.plot(cycles, strokevolumes)
        plt.title('LV stroke volume')
        plt.xlabel('Cardiac cycle [-]')
        plt.ylabel('Stroke volume [ml]')

        plt.subplot(1, 3, 3)
        plt.plot(cycles, work)
        plt.title('LV work')
        plt.xlabel('Cardiac cycle [-]')
        plt.ylabel('Work [J]')

    def save(self, filename, dpi=300, bbox_inches='tight'):
        """
        Write the currently drawn figure to file.

        Args:
            filename: Name (or path) to save the figure as/to.
            dpi (optional): Override the default dpi (300) for quality control.
            bbox_inches (optional): Override the bounding box ('tight') value.
        """
        # TODO Add check for whether or not a plot has been created.
        try:
            self._fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)

        except FileNotFoundError:
            import os
            os.mkdir('/'.join(filename.split('/')[:-1]))
            self.save(filename, dpi=dpi, bbox_inches=bbox_inches)


class GeneralHemodynamicsPlot(object):
    """
    High-level interface between :class:`~cvbtk.Dataset` and :mod:`matplotlib`
    that plots the pressure vs. time, volume vs. time, flowrate vs. time, and
    pressure vs. volume curves.

    Args:
        dataset: Dataset to create the figure from.
    """
    def __init__(self, dataset):
        # Store the dataset.
        self._df = dataset

        # Create an empty figure for plotting.
        self._fig = plt.figure()

        # Create a set of empty axes for plotting.
        gs = GridSpec(3, 2)
        _pt = plt.subplot(gs[0, 0])
        _vt = plt.subplot(gs[1, 0], sharex=_pt)
        _qt = plt.subplot(gs[2, 0], sharex=_pt)
        _pv = plt.subplot(gs[:, 1])
        self._ax = {'pt': _pt, 'vt': _vt, 'qt': _qt, 'pv': _pv}

        # Remove vertical spacing from the three individual axes.
        gs.update(hspace=0.15, wspace=0.30)

        # Remove x-axis labels and ticks from redundant axes.
        self._ax['pt'].xaxis.set_visible(False)
        self._ax['vt'].xaxis.set_visible(False)

        # Set axis labels.
        self._ax['qt'].set_xlabel('Time [ms]')
        self._ax['pv'].set_xlabel('Volume [ml]')
        self._ax['pv'].set_ylabel('Pressure [mmHg]')

        # Set the global title.
        self._fig.suptitle('Hemodynamic Relations')

        # Remove the right and top spines.
        [ax.spines['top'].set_visible(False) for _, ax in self._ax.items()]
        [ax.spines['right'].set_visible(False) for _, ax in self._ax.items()]

    def compare_against(self, dataset, *args, **kwargs):
        """
        Draw additional curves on the existing figure for visual comparison.

        Args:
            dataset: Additional dataset to draw curves from.
            *args: Arbitrary positional arguments for plotting parameters.
            **kwargs: Arbitrary keyword arguments for plotting parameters.
        """
        # TODO Check that a plot has been already created.

        # Make the pressure-time plot.
        self._ax['pt'].plot(dataset['time'], kPa_to_mmHg(dataset['plv']), *args, **kwargs)
        self._ax['pt'].plot(dataset['time'], kPa_to_mmHg(dataset['pven']), *args, **kwargs)
        self._ax['pt'].plot(dataset['time'], kPa_to_mmHg(dataset['part']), *args, **kwargs)

        # Make the volume-time plot.
        self._ax['vt'].plot(dataset['time'], dataset['vlv'], *args, **kwargs)
        # self._ax['vt'].plot(dataset['time'], dataset['vven'], *args, **kwargs)
        # self._ax['vt'].plot(dataset['time'], dataset['vart'], *args, **kwargs)

        # Make the flowrate-time plot.
        self._ax['qt'].plot(dataset['time'], dataset['qven'], *args, **kwargs)
        self._ax['qt'].plot(dataset['time'], dataset['qart'], *args, **kwargs)
        self._ax['qt'].plot(dataset['time'], dataset['qper'], *args, **kwargs)

        # Make the pressure-volume plot.
        # Each cycle (if multiple) will get its own color.
        for c in dataset['cycle'].unique():
            _df = dataset[dataset['cycle'] == int(c)]
            self._ax['pv'].plot(_df['vlv'], kPa_to_mmHg(_df['plv']), *args, **kwargs)

    def plot(self, cycle=None, legend=True):
        """
        Plot the defined hemodynamic relations for output.

        Args:
            cycle (optional): Filter and plot results for this specific cycle.
            legend (optional): Enables (default) or disables the legend.
        """
        # The cycle keyword argument can filter the results to a specific cycle.
        if cycle:
            df = self._df[self._df['cycle'] == int(cycle)]
        else:
            df = self._df

        # Make the pressure-time plot.
        self._ax['pt'].plot(df['time'], kPa_to_mmHg(df['plv']), label='Cavity')
        self._ax['pt'].plot(df['time'], kPa_to_mmHg(df['pven']), label='Venous')
        self._ax['pt'].plot(df['time'], kPa_to_mmHg(df['part']), label='Arterial')

        # Make the volume-time plot.
        self._ax['vt'].plot(df['time'], df['vlv'], label='Cavity')
        # self._ax['vt'].plot(df['time'], df['vven'], label='Venous')
        # self._ax['vt'].plot(df['time'], df['vart'], label='Arterial')

        # Make the flowrate-time plot.
        self._ax['qt'].plot(df['time'], df['qven'], label='Venous')
        self._ax['qt'].plot(df['time'], df['qart'], label='Arterial')
        self._ax['qt'].plot(df['time'], df['qper'], label='Peripheral')

        # Make the pressure-volume plot.
        # Each cycle (if multiple) will get its own color.
        for c in df['cycle'].unique():
            _df = df[df['cycle'] == int(c)]
            self._ax['pv'].plot(_df['vlv'], kPa_to_mmHg(_df['plv']), label=str(c))

        # Add the legends for the three individual plots, if needed to.
        if legend:
            self._ax['pt'].legend(loc=2, fontsize=6, title='Pressure')
            self._ax['pt'].get_legend().get_title().set_fontsize('7')
            self._ax['vt'].legend(loc=2, fontsize=6, title='Volume')
            self._ax['vt'].get_legend().get_title().set_fontsize('7')
            self._ax['qt'].legend(loc=2, fontsize=6, title='Flowrate')
            self._ax['qt'].get_legend().get_title().set_fontsize('7')

        # The pressure-volume loop always has a legend if multiple are plotted.
        if not cycle and len(df['cycle'].unique()) > 1:
            self._ax['pv'].legend(loc=2, fontsize=6, title='Cycle')
            self._ax['pv'].get_legend().get_title().set_fontsize('7')

    def save(self, filename, dpi=300, bbox_inches='tight'):
        """
        Write the currently drawn figure to file.

        Args:
            filename: Name (or path) to save the figure as/to.
            dpi (optional): Override the default dpi (300) for quality control.
            bbox_inches (optional): Override the bounding box ('tight') value.
        """
        # TODO Add check for whether or not a plot has been created.
        try:
            self._fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)

        except FileNotFoundError:
            import os
            os.mkdir('/'.join(filename.split('/')[:-1]))
            self.save(filename, dpi=dpi, bbox_inches=bbox_inches)


class HemodynamicsPlotDC(object):
    """
    High-level interface between :class:`~cvbtk.Dataset` and :mod:`matplotlib`
    that plots the pressure vs. time, volume vs. time, flowrate vs. time, and
    pressure vs. volume curves.

    Applicable to datasets with double circulation, where the keys of the systemic
    circulation have postfix '_s' and the pulmonary '_p'.

    Args:
        dataset: Dataset to create the figure from.
    """
    def __init__(self, dataset):
        # Store the dataset.
        self._df = dataset

        # Create an empty figure for plotting systemic hemodynamics.
        self._fig_s = plt.figure()

        # Create a set of empty axes for plotting systemic circulation
        gs = GridSpec(3, 2)
        _pt_s = plt.subplot(gs[0, 0])
        _vt_s = plt.subplot(gs[1, 0], sharex=_pt_s)
        _qt_s = plt.subplot(gs[2, 0], sharex=_pt_s)
        _pv_s = plt.subplot(gs[:, 1])
        self._ax_s = {'pt': _pt_s, 'vt': _vt_s, 'qt': _qt_s, 'pv': _pv_s}

        # Remove vertical spacing from the three individual axes.
        gs.update(hspace=0.15, wspace=0.30)

        # Create an second figure for plotting pulmonary hemodynamics.
        self._fig_p = plt.figure()

        # Create a set of empty axes for plotting systemic circulation
        _pt_p = plt.subplot(gs[0, 0])
        _vt_p = plt.subplot(gs[1, 0], sharex=_pt_p)
        _qt_p = plt.subplot(gs[2, 0], sharex=_pt_p)
        _pv_p = plt.subplot(gs[:, 1])
        self._ax_p = {'pt': _pt_p, 'vt': _vt_p, 'qt': _qt_p, 'pv': _pv_p}

        # Remove vertical spacing from the three individual axes.
        gs.update(hspace=0.15, wspace=0.30)

        for axi in [self._ax_s, self._ax_p]:
            # Remove x-axis labels and ticks from redundant axes.
            axi['pt'].xaxis.set_visible(False)
            axi['vt'].xaxis.set_visible(False)

            # Set axis labels.
            axi['qt'].set_xlabel('Time [ms]')
            axi['pv'].set_xlabel('Volume [ml]')
            axi['pv'].set_ylabel('Pressure [mmHg]')

            # Remove the right and top spines.
            [ax.spines['top'].set_visible(False) for _, ax in axi.items()]
            [ax.spines['right'].set_visible(False) for _, ax in axi.items()]

        # Set the global title.
        self._fig_s.suptitle('Hemodynamics Systemic Circulation')

        # Set the global title.
        self._fig_p.suptitle('Hemodynamics Pulmonary Circulation')

    def plot(self, cycle=None, legend=True, circulation='systemic'):
        """
        Plot the defined hemodynamic relations for output.

        Args:
            cycle (optional): Filter and plot results for this specific cycle.
            legend (optional): Enables (default) or disables the legend.
            circulation (optional, str): Plot 'systemic' (default) or 'pulmonary' hemodynamics
        """
        # The cycle keyword argument can filter the results to a specific cycle.
        if cycle:
            df = self._df[self._df['cycle'] == int(cycle)]
        else:
            df = self._df

        # Check which circulation to plot.
        if circulation == 'systemic':
            fig = self._fig_s
            ax = self._ax_s
            ps = '_s'
        elif circulation == 'pulmonary':
            fig = self._fig_p
            ax = self._ax_p
            ps = '_p'
        else:
            raise NameError("Unknown input "+circulation+" for argument circulation. Choose 'systemic' or 'pulmonary'.")

        plt.figure(fig.number)

        # Make the pressure-time plot.
        ax['pt'].plot(df['time'], kPa_to_mmHg(df['pcav'+ps]), label='Cavity')
        ax['pt'].plot(df['time'], kPa_to_mmHg(df['pven'+ps]), label='Venous')
        ax['pt'].plot(df['time'], kPa_to_mmHg(df['part'+ps]), label='Arterial')

        # Make the volume-time plot.
        ax['vt'].plot(df['time'], df['vcav'+ps], label='Cavity')
        # ax['vt'].plot(df['time'], df['vven'+ps], label='Venous')
        # ax['vt'].plot(df['time'], df['vart'+ps], label='Arterial')

        # Make the flowrate-time plot.
        ax['qt'].plot(df['time'], df['qven'+ps], label='Venous')
        ax['qt'].plot(df['time'], df['qart'+ps], label='Arterial')
        ax['qt'].plot(df['time'], df['qper'+ps], label='Peripheral')
        if circulation=='systemic' and 'qlvad' in df.keys():
            ax['qt'].plot(df['time'], df['qlvad'], label='LVAD')

        # Make the pressure-volume plot.
        # Each cycle (if multiple) will get its own color.
        for c in df['cycle'].unique():
            _df = df[df['cycle'] == int(c)]
            ax['pv'].plot(_df['vcav'+ps], kPa_to_mmHg(_df['pcav'+ps]), label=str(c))

        # Add the legends for the three individual plots, if needed to.
        if legend:
            ax['pt'].legend(loc=1, fontsize=6, title='Pressure')
            ax['pt'].get_legend().get_title().set_fontsize('7')
            ax['vt'].legend(loc=1, fontsize=6, title='Volume')
            ax['vt'].get_legend().get_title().set_fontsize('7')
            ax['qt'].legend(loc=1, fontsize=6, title='Flowrate')
            ax['qt'].get_legend().get_title().set_fontsize('7')

        # The pressure-volume loop always has a legend if multiple are plotted.
        if not cycle and len(df['cycle'].unique()) > 1:
            ax['pv'].legend(loc=1, fontsize=6, title='Cycle')
            ax['pv'].get_legend().get_title().set_fontsize('7')

    def plot_pvloops(self, cycle=None):
        """
        Plot the PV-loop of the LV and RV in one figure.

        Args:
            cycle (optional): Filter and plot results for this specific cycle.
        """
        # The cycle keyword argument can filter the results to a specific cycle.
        if cycle:
            df = self._df[self._df['cycle'] == int(cycle)]
        else:
            df = self._df

        # Create figure
        plt.figure()
        plt.xlabel('Volume')
        plt.ylabel('Pressure')
        plt.title('Pressure volume loop for LV (solid) and RV (dashed)')

        # Make the pressure-volume plot.
        # Each cycle (if multiple) will get its own color.
        for c in df['cycle'].unique():
            _df = df[df['cycle'] == int(c)]
            loop = plt.plot(_df['vcav_s'], kPa_to_mmHg(_df['pcav_s']), label=str(c))
            plt.plot(_df['vcav_p'], kPa_to_mmHg(_df['pcav_p']), color = loop[0].get_color(), ls = '--', label='_nolegend_')

        # The pressure-volume loop always has a legend if multiple are plotted.
        if not cycle and len(df['cycle'].unique()) > 1:
            plt.legend(loc=1, fontsize=6, title='Cycle')
            plt.legend().get_title().set_fontsize('7')

    def plot_function(self):
        """
        Plot the ventricles function in terms of stroke volume, maximum pressure,
        and work for every cycle.
        """
        # Find the number of the first and last cycle.
        start = min(self._df['cycle'])
        stop = max(self._df['cycle'])
        all_cycles = list(range(start, stop))

        # Extract the SV of the systemic and pulmonary circulation.
        strokevolumes = {'_s': [],
                         '_p': []}
        pmax = {'_s': [],
                '_p': []}
        work = {'_s': [],
                '_p': []}
        cycles = []
        for c in all_cycles:
            df = self._df[abs(self._df['cycle'] - c) <= 0.001]
            for ps in ['_s', '_p']:           
                try:
                    # Find begin phase 2.
                    idx_ed = min(np.where(df['phase' + ps].values == 2)[0])
                    # Find begin phase 4.
                    idx_ee = min(np.where(df['phase' + ps].values == 4)[0])
                    

                except ValueError:
                    continue 

                if ps == '_s':
                    cycles.append(c)
                
                # Calculate SV.
                volume = df['vcav' + ps].values
                sv = volume[idx_ed] - volume[idx_ee]
                strokevolumes[ps].append(sv)

                # Maximum pressure.
                pressure = df['pcav' + ps].values
                pmax[ps].append(max(pressure))

                # Work (surface of pressure-volume loop).
                work_cycle = 0.
                for ii in range(1, len(volume)):
                    work_cycle += (pressure[ii - 1] + pressure[ii]) / 2 * (
                                volume[ii - 1] - volume[ii]) / 1000  # in Joule.
                work[ps].append(work_cycle)

        plt.figure(figsize=(15, 5), dpi=100) 
        lv_label = mlines.Line2D([], [], color='C3', label='LV')
        rv_label = mlines.Line2D([], [], color='C0', linestyle='--', label='RV')
        
        # Stroke volume.
        ax3 = plt.subplot(1, 3, 1)
        ax3.plot(cycles, strokevolumes['_s'], color='C3')
        ax3.set_xlabel('Cardiac cycle [-]')
        ax3.set_ylabel('Stroke volume [ml]')
        ax3.tick_params('y')
#        ax4 = ax3.twinx()
        ax3.plot(cycles, strokevolumes['_p'], '--', color='C0')
#        ax4.set_ylabel('RV volume [ml]', color='C0')
#        ax4.tick_params('y', colors='C0')
        plt.title('Stroke volumes')
        plt.legend(handles=[lv_label, rv_label], loc='lower right')
#        plt.grid('on')

        # Maximum cavity pressure.
        ax1 = plt.subplot(1, 3, 2)
        ax1.plot(cycles, kPa_to_mmHg(np.array(pmax['_s'])), color='C3')
        ax1.set_xlabel('Cardiac cycle [-]')
        ax1.set_ylabel('LV pressure [mmHg]', color='C3')
        ax1.tick_params('y', colors='r')
        ax2 = ax1.twinx()
        ax2.plot(cycles, kPa_to_mmHg(np.array(pmax['_p'])), '--', color='C0')
        ax2.set_ylabel('RV pressure [mmHg]', color='C0')
        ax2.tick_params('y', colors='C0')
        plt.title('Maximum cavity pressure')
        plt.legend(handles=[lv_label, rv_label], loc='lower right')
#        plt.grid('on')

        # Work.
        ax5 = plt.subplot(1, 3, 3)
        ax5.plot(cycles, work['_s'], 'C3')
        ax5.set_xlabel('Cardiac cycle [-]')
        ax5.set_ylabel('LV work [J]', color='C3')
        ax5.tick_params('y', colors='C3')
        ax6 = ax5.twinx()
        ax6.plot(cycles, work['_p'], '--C0')
        ax6.set_ylabel('RV work [J]', color='C0')
        ax6.tick_params('y', colors='C0')
        plt.title('Work')
        plt.legend(handles=[lv_label, rv_label], loc='lower right')
#        plt.grid('on')
        
        # Increase vertical spacing between subplots.
        plt.subplots_adjust(wspace = .5)

    def save(self, filename, dpi=300, bbox_inches='tight'):
        """
        Write the current figure to file.

        Args:
            filename: Name (or path) to save the figure as/to.
            dpi (optional): Override the default dpi (300) for quality control.
            bbox_inches (optional): Override the bounding box ('tight') value.
        """
        try:
            plt.gcf().savefig(filename, dpi=dpi, bbox_inches=bbox_inches) # current figure is saved

        except FileNotFoundError:
            import os
            os.mkdir('/'.join(filename.split('/')[:-1]))
            self.save(filename, dpi=dpi, bbox_inches=bbox_inches)


def postprocess(results, dir_out='.', cycle=5):
    # Plot the hemodynamic relations
    if 'pcav_p' in results.keys():
        # BiV.
        simulation_plot = HemodynamicsPlotDC(results)
        
        simulation_plot.plot(circulation='systemic', cycle=cycle) #, cycle=NUM_CYCLES)
        simulation_plot.save(os.path.join(dir_out, 'hemodynamics_systemic.png'))
        
        simulation_plot.plot(circulation='pulmonary', cycle=cycle) #, cycle=NUM_CYCLES)
        simulation_plot.save(os.path.join(dir_out, 'hemodynamics_pulmonary.png'))
    
        simulation_plot.plot_pvloops(cycle=cycle)
        simulation_plot.save(os.path.join(dir_out, 'pvloops.png'))
        
        simulation_plot.plot_function()
        simulation_plot.save(os.path.join(dir_out, 'ventricles_function.png'))

    else:
        # LV only.
        simulation_plot = HemodynamicsPlot(results)
        
        simulation_plot.plot(cycle=cycle) #, cycle=NUM_CYCLES)
        simulation_plot.save(os.path.join(dir_out, 'hemodynamics.png'))
        
        simulation_plot.plot_function()
        plt.savefig(os.path.join(dir_out, 'lv_function.png'), dpi=300)
        
def compare_cycles(df, cycles):
    plt.figure('PV-loops compare')
    ax = plt.subplot(1, 1, 1)
    colors = ['r', 'b']
    cavity_names = ['LV', 'RV']
    for ip, p in enumerate(['_s', '_p']):
        styles = ['-', '--']
        for i, c in enumerate(cycles):
            _df = df[df['cycle'] == int(c)]
            ax.plot(_df['vcav'+p], kPa_to_mmHg(_df['pcav'+p]), linestyle=styles[i], label=cavity_names[ip]+' cycle '+str(c), color=colors[ip])
        
    ax.set_title('PV-loops')
    ax.set_xlabel('Volume [ml]')
    ax.set_ylabel('Pressure [mmHg]')
    ax.legend(loc=1, fontsize=8)
    ax.get_legend().get_title().set_fontsize('9')

def compute_IQ(hemo, hc):
    """ 
    Compute cardiac contractility index from pump hemodynamics 
    as proposed by Naiyanetr et al. 2010
    """
    idx = np.where(np.array(hemo['hc'])==hc)
    dQdt_max = np.array(hemo['dQdt_max'])[idx]
    Qp2p = np.reshape(np.array(hemo['Qp2p'])[idx], (-1, 1))
    
    # Create linear regression object
    regr = linear_model.LinearRegression()
    
    # Fit the model using the training sets
    regr.fit(Qp2p, dQdt_max)
    
    # The slope
    IQ = regr.coef_
    
    b = regr.intercept_
    
    plt.figure()
    plt.scatter(Qp2p, dQdt_max)
    plt.plot(Qp2p, IQ*Qp2p + b, '--')
    
    return IQ

def get_hc_colors():
    return {1: 'k', 3: '0.6', 4: '0.4'}
    
def get_hc_names():
    return {1: 'REF', 3: 'DEG', 4: 'DEG2'} 

def get_rpm_colors():
    return {-1: 'k', 0: "C0", 7500: "C1", 8500: "C2", 9500:"C3", 10500:'C4', 11500: 'C5', 12000: 'C6'}

def get_rpm_linestyles():
    return {-1: '--', 0: "-", 7500: "--", 8500: "-.", 9500:":", 10500:'--', 11500: "-.", 12000: ":"}

def load_mockdata(csvfile, simulation_time_array):
    # Load csv file as Dataset.
    data = Dataset(filename=csvfile)

    # Resample data.
    time = np.array(data['time'])*1000
    plv = mmHg_to_kPa(np.array(data['plv']))
    plv_resampled = np.interp(simulation_time_array, time, plv, period=500)

    return data, plv_resampled

def load_reduced_dataset(filename, cycle=None):
    """
    Load the given CSV file and reduce it to the final cycle (default).
    """
    full = Dataset(filename=filename)    
    if cycle is None:
        cycle = int(max(full['cycle']) - 1)
    
    reduced = full[full['cycle'] == cycle].copy(deep=True)
    return reduced

def hemodynamic_summary(data,cycle=None):
    
    if 'qart' in data.keys():
        # LifeTec data (arterial flow in LV simulations is 'qao').
        data = data[data['cycle']==cycle]
        time = data['time'].values*1000  # ms                  
        plv = data['plv'].values  # mmHg
        part = data['part'].values  # mmHg
        qao = data['qart'].values  # l/min
        qlvad = data['qlvad'].values  # l/min
        vlv = np.nan*np.ones_like(plv)
    
    elif 'plv' in data.keys():
        # LV results.
        data = data[data['cycle']==cycle]
        time = data['time'].values  # ms
        plv = kPa_to_mmHg(data['plv'].values)
        part = kPa_to_mmHg(data['part'].values)
        pven = kPa_to_mmHg(data['pven'].values)
        qao = data['qao'].values*60/1000  # ml/s -> l/min
        vlv = data['vlv'].values  # ml

        if 'qlvad' in data.keys():        
            qlvad = data['qlvad'].values*60/1000  # ml/s -> l/min
        else:
            qlvad = np.zeros_like(qao)
        
        # Add lvad flow to arterial flow.
        qao += qlvad
        
    elif 'pcav_s' in data.keys():
        # BiV results.
        data = data[data['cycle']==cycle]
        time = data['time'].values  # ms
        plv = kPa_to_mmHg(data['pcav_s'].values)
        part = kPa_to_mmHg(data['part_s'].values)
        pven = kPa_to_mmHg(data['pven_s'].values)
        qao = data['qart_s'].values*60/1000  # ml/s -> l/min
        vlv = data['vcav_s'].values  # ml
        
        if 'qlvad' in data.keys():        
            qlvad = data['qlvad'].values*60/1000  # ml/s -> l/min
        else:
            qlvad = np.zeros_like(qao)

        # Add lvad flow to arterial flow.
        qao += qlvad        
    else:
        raise ValueError('Unexpected keys in data dictionary.')
    data_cycle = data[data['cycle']==cycle]
    time_cycle = data_cycle['time'].values
    HR = round(60000/(max(time_cycle)-min(time_cycle)))
    EDV =  max(vlv)
    SV = np.mean(qao - qlvad)/HR * 1000
    CO = np.mean(qao)
    
    hemo = {    
            'HR': HR,
            'EDV': EDV,
            'ESV': min(vlv),
            'CO': CO,
            'qao': np.mean(qao - qlvad),
            'SV': SV,
            'EF': SV/EDV*100,
            'MAP': np.mean(part),
            'SAP': max(part),
            'DAP': min(part),
            'PP': max(part) - min(part),
            'plv_max': max(plv),
            'dpdt_max': max(np.diff(plv)/np.diff(time/1000)),
            'W': - np.trapz(mmHg_to_kPa(plv), vlv/1000),
            'LVAD_flow': np.mean(qlvad),
            'LVAD_frac': np.mean(qlvad)/CO*100,
            'dQdt_max':  max(np.diff(qlvad/60*1000)/np.diff(time/1000)), # ml/s^2
            'Qp2p': max(qlvad/60*1000) - min(qlvad/60*1000) # ml/s
            }

    if not 'qart' in data.keys():
        # Not LifeTec data (aorctic flow in LV simulations is 'qao').
        # Compute CVP (is approximately (systemic) venous pressure).
        hemo['CVP'] = np.mean(pven) 

    if 'pven_p' in data.keys():
        # BiV simulation.
        # Add pulmonary circulation data.
        prv = kPa_to_mmHg(data['pcav_p'].values)
        vrv = data['vcav_p'].values  # ml
        part_p = kPa_to_mmHg(data['part_p'].values)
        qart_p = data['qart_p'].values*60/1000  # ml/s -> l/min
        pven_p = kPa_to_mmHg(data['pven_p'].values)

        vart = data['vart_s'].values # ml
        vven = data['vven_s'].values # ml
        
        vart_p = data['vart_p'].values # ml
        vven_p = data['vven_p'].values # ml        

        EDV_rv = max(vrv)
        SV_rv = np.mean(qart_p)/HR * 1000

        hemo['MAV'] = np.mean(vart) # ml
        hemo['MVV'] = np.mean(vven) # ml

        hemo_rv = {    
                'EDV_rv': EDV_rv,
                'ESV_rv': min(vrv),
                'CO_rv': np.mean(qart_p),
                'SV_rv': SV_rv,
                'EF_rv': SV_rv/EDV_rv*100,
                'MAP_pul': np.mean(part_p),
                'SAP_pul': max(part_p),
                'DAP_pul': min(part_p),
                'PP_pul': max(part_p) - min(part_p),
                'prv_max': max(prv),
                'dp_rvdt_max': max(np.diff(prv)/np.diff(time/1000)),
                'W_rv': - np.trapz(mmHg_to_kPa(prv), vrv/1000),
                'MAV_pul': np.mean(vart_p),
                'MVV_pul': np.mean(vven_p)
                }

        # Compute PCWP (is approximately LV atrial pressure ~ pulmonary venous pressure)
        hemo['PCWP'] = np.mean(pven_p)
        
        hemo.update(hemo_rv)
        
    return hemo
    
    
def print_hemodynamic_summary(hemo):    
    print('\nSYSTEMIC CIRCULATION:')
    print(('HR: {:11.0f} bpm\n' +
           'EDV: {:10.2f} ml\n' +
           'ESV: {:10.2f} ml\n' +
          'SV: {:11.2f} ml\n' +
          'EF: {:11.2f} %\n' +
          'qao: {:10.2f} l/min \n' +
          'CO: {:11.2f} l/min\n' +
          'LVAD_frac {:.2f} %\n' +
          'MAP: {:10.2f} mmHg\n' +
          'SAP: {:10.2f} mmHg\n' +
          'DAP: {:10.2f} mmHg\n' +
          'PP: {:11.2f} mmHg\n' +
          'plv_max: {:6.2f} mmHg\n' +
          'dp/dt_max: {:.0f} mmHg/s\n' +
          'W: {:12.2f} J').format(
                  hemo['HR'], hemo['EDV'], hemo['ESV'], hemo['SV'], hemo['EF'],
                  hemo['qao'], hemo['CO'], hemo['LVAD_frac'], hemo['MAP'], 
                  hemo['SAP'], hemo['DAP'], hemo['PP'],
                  hemo['plv_max'], hemo['dpdt_max'], hemo['W']))
    
    if 'LVAD_flow' in hemo.keys():
        print('LVAD_flow: {:.2f} l/min'.format(hemo['LVAD_flow']))

    if 'CVP' in hemo.keys():
        print('CVP: {:10.2f} mmHg'.format(hemo['CVP']))        

    if 'PCWP' in hemo.keys():
        # Print BiV data.
        print('\nPULMONARY CIRCULATION:')
        print((
              'EDV: {:10.2f} ml\n' +
              'ESV: {:10.2f} ml\n' +
              'SV: {:11.2f} ml\n' +
              'EF: {:11.2f} %\n' +
              'CO: {:11.2f} l/min\n' +
              'MAP: {:10.2f} mmHg\n' +
              'SAP: {:10.2f} mmHg\n' +
              'DAP: {:10.2f} mmHg\n' +
              'PP: {:11.2f} mmHg\n' +
              'PCWP: {:9.2f} mmHg\n' + 
              'prv_max: {:6.2f} mmHg\n' +
              'dp/dt_max: {:.0f} mmHg/s\n' +
              'W: {:12.2f} J').format(
                      hemo['EDV_rv'], hemo['ESV_rv'], hemo['SV_rv'], hemo['EF_rv'],
                      hemo['CO_rv'], hemo['MAP_pul'], 
                      hemo['SAP_pul'], hemo['DAP_pul'], hemo['PP_pul'], hemo['PCWP'],
                      hemo['prv_max'], hemo['dp_rvdt_max'], hemo['W_rv']))    
        
    print('\n')
    
def figure_make_up(title=None, xlabel=None, ylabel=None, create_legend=True, legend_title=None, fontsize=13):
    if title is not None:
        plt.title(title, fontsize=fontsize*1.15)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=fontsize)
    plt.tick_params(labelsize=fontsize*0.9)
    if create_legend is True:
        leg = plt.legend(title=legend_title, fontsize=fontsize*0.9)
        leg.get_title().set_fontsize(fontsize*0.9)
    
def find_experiment_curves(hc, rpm):
    """
    Returns the curves.hdf5 file for a lifetec experiment specified by hc and rpm.
    Args:
        hc (int): heart condition. Choose from 1 (REF), 3 (DEG)
        rpm (int): pump speed in rpm (e.g. 0, 7500, 8500, etc.)
    """
    filename_experiment_target = os.path.join(LIFETEC_DATA_PATH, 
                                              'hc{0}\hc{0}_p3_n{1:d}_*_curves.hdf5'
                                              .format(hc, int(rpm/100)))
    
    # Look for files with filename_experiment_target.
    filename_experiment_list = glob.glob(filename_experiment_target)
    return filename_experiment_list[0]  # Select the first, we are averaging all beats anyways.    

def find_experiment_hemodynamics(hc, rpm):
    """
    Returns the csv file containing the hemodynamic data 
    for a lifetec experiment specified by hc and rpm.
    Args:
        hc (int): heart condition. Choose from 1 (REF), 3 (DEG)
        rpm (int): pump speed in rpm (e.g. 0, 7500, 8500, etc.)
    """   
    filename_experiment_target = os.path.join(LIFETEC_DATA_PATH, 
                                              r'hemo_lifetec_hc{}_{}_rpm.csv')
    return filename_experiment_target.format(hc, rpm)

def find_simulation_curves(hc=None, rpm=None, simulation_type='lifetec_fit', cycle=None, strain_reference=None):
    """
    Returns the curves.hdf5 file containing strain data 
    for a simulation specified by hc and rpm and simulation_type.
    Args:
        hc (int): heart condition. Choose from 1 (REF), 3 (DEG)
        rpm (int): pump speed in rpm (e.g. 0, 7500, 8500, etc.)
        simulations_type (str): specify which simulation to load:
            'lifetec_fit': LV (LifeTec) simulation with tuned preload.
            'fixed_preload': LV (LifeTec) simulation with fixed preload.
            'BiV': BiV (diseased) simulation.
        cycle (int): specify the cycle to load. By default, the last available
                     cycle is loaded.
        strain_reference (str): If the cycle directory with results contains 
                                a postfix specifying the reference state used 
                                for strains, specify this postfix here.
    """
    if simulation_type == 'lifetec_fit':
        hc_dir = os.path.join(SIMULATION_DATA_PATH, 
                              r'lifetec\hc{}'.format(hc))
    elif simulation_type == 'fixed_preload':
        hc_dir = os.path.join(SIMULATION_DATA_PATH, 
                              r'lifetec\hc{}_fixed_preload'.format(hc))       
    elif simulation_type == 'BiV':
        hc_dir = os.path.join(SIMULATION_DATA_PATH,
                              r'biv_realcycle\patient_simulation')
    else:
        raise ValueError('Unknown "simulation_type".')
        
    results_dir = glob.glob(os.path.join(hc_dir, str(rpm)+'_rpm*'))[0]
    
    # Select cycle number.
    if strain_reference is None:
        all_cycle_dirs = glob.glob(os.path.join(results_dir, 'cycle*'))
    else:
        all_cycle_dirs = glob.glob(os.path.join(results_dir, 'cycle*{}'.format(strain_reference)))
    
    c_max = -1
    cycle_dir = None
    for d in all_cycle_dirs:
        c = int(os.path.split(d)[1].split('_')[1])

        if cycle is None:
            # Select highest cycle.
            if c > c_max:
                c_max = c
                cycle_dir = d
                
        elif c == cycle:
            cycle_dir = d
            break
    if cycle_dir is None or not os.path.exists(os.path.join(cycle_dir, 'curves.hdf5')):
        raise RuntimeError('Curves file not found.')
        
    return os.path.join(cycle_dir, 'curves.hdf5')

def find_simulation_hemodynamics(hc=None, rpm=None, simulation_type='fixed_preload'):
    """
    Returns the results.csv file containing hemodynamic data 
    for a simulation specified by hc and rpm and simulation_type.
    Args:
        hc (int): heart condition. Choose from 1 (REF), 3 (DEG)
        rpm (int): pump speed in rpm (e.g. 0, 7500, 8500, etc.)
    """
    if simulation_type == 'lifetec_fit':
        hc_dir = os.path.join(SIMULATION_DATA_PATH, 
                              r'lifetec\hc{}'.format(hc))
    elif simulation_type == 'fixed_preload':
        hc_dir = os.path.join(SIMULATION_DATA_PATH, 
                              r'lifetec\hc{}_fixed_preload'
                              .format(hc))    
    elif simulation_type == 'BiV':
        # Dirty fix: if rpm == -1, we load the healthy state.
        if rpm <= -1:
            results_dir = os.path.join(SIMULATION_DATA_PATH, 
                                       r'biv_realcycle\reorientation\REF')
        else:
            hc_dir = os.path.join(SIMULATION_DATA_PATH, 
                                  r'biv_realcycle\patient_simulation')
    else:
        raise ValueError('Unknown "simulation_type". Choose from "lifetec_fit", "fixed_preload", "BiV".')
    if not rpm <= -1:
        results_dir = glob.glob(os.path.join(hc_dir, str(rpm)+'_rpm*'))[0]
    return os.path.join(results_dir, 'results.csv')


def plot_comparison(results, hemo_lt, cycle=None, fontsize=12, plot_mode='compare'):
    
    # Default to last cycle.
    if cycle is None:
        cycle = max(results['cycle']) - 1
        
    if len(np.unique(results['cycle'])) > 1:        
        print('Extracting cycle {}...'.format(cycle))
        
        # Extract results of cycle.
        results = results[results['cycle'] == int(cycle)].copy(deep=True)
    
    # Extract simulation data.
    time_sim = np.array(results['time'])
    if 'plv' in results.keys():
        # LV simulation.
        plv = kPa_to_mmHg(np.array(results['plv']))
        part = kPa_to_mmHg(np.array(results['part']))
    else:
        # BiV simulation.
        plv = kPa_to_mmHg(np.array(results['pcav_s']))
        part = kPa_to_mmHg(np.array(results['part_s']))
        
    # Extract lifetec data.
    time_lt = np.array(hemo_lt['time'])*1000  # s -> ms
    plv_lt = np.array(hemo_lt['plv'])
    part_lt = np.array(hemo_lt['part'])
    
    # Resample simulation results to lifetec data.
    plv_res = np.interp(time_lt, time_sim, plv, period=500)
    part_res = np.interp(time_lt, time_sim, part, period=500)
    
    # Synchronize results.
    plv_res_sync, i_sync = synchronize(plv_lt, plv_res)
    part_res_sync = part_res[i_sync]
    
    # Plot.
    if plot_mode in ['compare', 'experiment']:
        plt.plot(time_lt, plv_lt, 'C0-', label='$p_{lv}$ experiment', linewidth=fontsize/6)
        plt.plot(time_lt, part_lt, 'C1-', label='$p_{art}$ experiment', linewidth=fontsize/6)
    if plot_mode is 'compare':      
        plt.plot(time_lt, plv_res_sync, 'C0--', label='$p_{lv}$ simulation', linewidth=fontsize/6)
        plt.plot(time_lt, part_res_sync, 'C1--', label='$p_{art}$ simulation', linewidth=fontsize/6)
    if plot_mode is 'simulation':      
        plt.plot(time_lt, plv_res_sync, 'C0-', label='$p_{lv}$ simulation', linewidth=fontsize/6)
        plt.plot(time_lt, part_res_sync, 'C1-', label='$p_{art}$ simulation', linewidth=fontsize/6)
        
    plt.grid('on')
    
    figure_make_up(xlabel='Time [ms]', ylabel='Pressure [mmHg]',
               fontsize=fontsize, create_legend=False)


def merge_dictionaries(dict_1, dict_2):
    
    dict_1_keys = dict_1.keys()
    
    for key in dict_2.keys():
        
        # Convert to list if needed.
        if not type(dict_2[key]) is list:
            dict_2[key] = [dict_2[key]]
        
        if key in dict_1_keys:
            # Convert to list if needed.
            if not type(dict_1[key]) is list:
                dict_1[key] = [dict_1[key]]
            
            for v in dict_2[key]:
                dict_1[key].append(v)
            
        else:
            dict_1[key] = dict_2[key]
    
    return dict_1


def plot_global_hemodynamic_variable(hemo, key, label_prefix='', fontsize=13, hc_plot=None, **kwargs):
    
    # Names for the heart conditions.
    hc_names = get_hc_names()
    
    hc_colors = get_hc_colors()
    
    hc = np.array(hemo['hc'])
    rpm = np.array(hemo['rpm'])
    values = np.array(hemo[key])
    
    if hc_plot is None:
#        hc_plot = np.unique(hc)[::-1]
        hc_plot = np.unique(hc)
    elif type(hc_plot) is not list or type(hc_plot) is not np.array:
        hc_plot = [hc_plot]
        
    multiple_hc = len(hc_plot) > 1
        
    assert np.mod(len(values), len(rpm)) == 0

    num_segments = int(len(values)/len(rpm)) 
    
    values = values.reshape(-1, num_segments)

    for i_hc in hc_plot:
        idx = np.where(hc == i_hc)[0]
        if rpm[idx[0]] == -1:
            plotargs = {'color': hc_colors[i_hc],
                        'linewidth': fontsize/6,
                        'linestyle': '--'}
            plotargs.update(kwargs)
            plotargs['marker'] = None
            plotargs['label'] = None
            
            # Dirty fix: if rpm = -1, this is the healthy case which must be 
            # plotted as a horizontal dashed line
            plt.axhline(y=values[idx[0]], **plotargs)
            
            # Remove first point.
            idx = idx[1:]
            
        if num_segments > 1:             
            # Per segment.
            for j in range(num_segments):            

                plotargs = {'linestyle': ':',
                            'linewidth': fontsize/6,
                            'marker': 'o',
                            'markersize': fontsize/3}   
                plotargs.update(kwargs)

                if rpm[idx[0]] == 0:
                    # Use a dotted line for first 2 points.
                    plt.plot(rpm[idx[1:]]/1000, values[idx[1:], j], 
                                color='C{}'.format(j), 
                                label=label_prefix+hc_names[i_hc]*multiple_hc+'{}'.format(j+1),
                                **plotargs)    
                    
                    plotargs['linestyle'] = ':'
                    plt.plot(rpm[idx[:2]]/1000, values[idx[:2], j], 
                                color='C{}'.format(j), 
                                label=None,
                                **plotargs)   
                else:
                    plt.plot(rpm[idx]/1000, values[idx, j], 
                                color='C{}'.format(j), 
                                label=label_prefix+hc_names[i_hc]*multiple_hc+'{}'.format(j+1),
                                **plotargs)   
                
        else:
            plotargs = {'color': hc_colors[i_hc],
                        'label': label_prefix+hc_names[i_hc]*multiple_hc,
                        'linewidth': fontsize/6,
                        'marker': 'o',
                        'markersize': fontsize/3}
            plotargs.update(kwargs)
            
            if rpm[idx][0] == 0:
                plt.plot(rpm[idx[1:]]/1000, values[idx[1:]], **plotargs)
                
                # Use a dotted line for first 2 points.
                plotargs['linestyle'] = ':'
                plotargs['label'] = None
                plt.plot(rpm[idx[:2]]/1000, values[idx[:2]], **plotargs)
            else:
                plt.plot(rpm[idx]/1000, values[idx], **plotargs)
            
    plt.xlabel('LVAD speed [krpm]', fontsize=fontsize)
    plt.tick_params(labelsize=fontsize-2)
    plt.grid('on')
    plt.title(key)


def remove_ticks(axis, remove_label=True, **kwargs):
    """
    axis: either 'x', 'y', 'xy' or ('x', 'y')
    """
    default_kwargs = {'which': 'both',       # both major and minor ticks are affected
                      'bottom': False,      # ticks along the bottom edge are off
                      'top': False,         # ticks along the top edge are off
                      'left': False,
                      'right': False,
                      'labelbottom': False,   # labels along the bottom edge are off
                      'labelleft': False
                      }
    default_kwargs.update(kwargs)
    
    for a in axis:     
        if 'x' in a:
            # x-ticks.
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                **default_kwargs)
            # Remove x-label.
            plt.xlabel('')
        if 'y' in a:
            # y-ticks.
            plt.tick_params(
                axis='y',          # changes apply to the y-axis
                **default_kwargs)            
            # Remove y-label.
            plt.ylabel('')


def subplot_comparisons(hc_all, rpm_all, fontsize=12, save_to_filename='', 
                        simulation_type='lifetec_fit', plot_mode='compare'):
    
    # Names for the heart conditions.
    hc_names = get_hc_names()
    
    # Compute reasonable number of rows and collumns.
    nrows = np.round(np.sqrt(len(hc_all)))
    ncols = np.ceil(np.sqrt(len(hc_all)))
    
    hemo_sum_sim_merged = {}
    hemo_sum_exp_merged = {}
    
    plt.figure(figsize=(13*ncols,10*nrows))
    for i, i_hc in enumerate(hc_all):
        i_rpm = rpm_all[i]
    
        # Find and read results file of simulation.    
        csv = find_simulation_hemodynamics(i_hc, i_rpm, simulation_type=simulation_type)
        results = load_reduced_dataset(filename=csv)
        
        # Find and read results file of experiment.    
        hemo_lt_csv = find_experiment_hemodynamics(i_hc, i_rpm)
        hemo_lt = Dataset(filename=hemo_lt_csv)
        
        # Print a hemodynamic summary.
        print('SIMULATION {}_{}'.format(hc_names[i_hc], i_rpm))
        hemo_sum_sim = hemodynamic_summary(results)
        print_hemodynamic_summary(hemo_sum_sim)
            
        print('EXPERIMENT {}_{}'.format(hc_names[i_hc], i_rpm))
        hemo_sum_exp = hemodynamic_summary(hemo_lt)
        print_hemodynamic_summary(hemo_sum_exp)

        # Add heart condition and pump speed to summaries.
        hemo_sum_sim['hc'] = i_hc
        hemo_sum_sim['rpm'] = i_rpm

        hemo_sum_exp['hc'] = i_hc
        hemo_sum_exp['rpm'] = i_rpm

        # Merge the data.
        merge_dictionaries(hemo_sum_sim_merged, hemo_sum_sim)
        merge_dictionaries(hemo_sum_exp_merged, hemo_sum_exp)
        
        if i == 0:
            ax1 = plt.subplot(nrows, ncols, i+1)
        else:
             plt.subplot(nrows, ncols, i+1, sharey=ax1)
             
        plot_comparison(results, hemo_lt, fontsize=fontsize, plot_mode=plot_mode)
        
        if i + ncols < len(hc_all):
            # Remove x-label for non-bottom plots.
            plt.xlabel('')
        
        figure_make_up(title='{}_{}'.format(hc_names[i_hc], i_rpm), 
                       create_legend=False, fontsize=fontsize)
            
        if i == 0:
            # Legend at first subplot.
            figure_make_up(fontsize=fontsize)
            
    
    # Save if filename is given.
    if not len(save_to_filename) == 0:
        plt.savefig(save_to_filename,  dpi=300, bbox_inches="tight")
        
    return hemo_sum_sim_merged, hemo_sum_exp_merged


class StrainAnalysis(object):
    """
    Class for strain analysis of US experiment data or FE simulation data. 
    Args:
        filename (str): path to the file with the results. Can either be a 
        *curves.hdf5 (US experiment) or strains.hdf5 (FE simulation).
        **kwargs: optional key-word arguments for user-defined parameters
    """
    def __init__(self, filename, **kwargs):

        self.filename = filename
        self._parameters = self.default_parameters()
        self._parameters.update(**kwargs)
        
        if self.parameters['strain_vars'] is None:
            self.autofill_strain_vars()
        
        self.type = self.get_type()
        
        self._curves = None
        self._data = None
    
    @staticmethod
    def default_parameters():
        
        # Specify optimal segment numbers per strain measure.
        optimal_segments = {
                'Eccc': [3, 6],
                'Errc': [4, 5],
                'Ecrc': [3, 6]}
        
        par = {
            # If True, for STE data the strains measured at different beats are averaged per tracking point (node).
            'average_beats': True,  
            
            # option 1: include and average nodes in all segments;.
            # option 2: include and average nodes in theoretically optimal segments per strain measure. 
            'average_mode': 1,
            
            # Specify 'mean' or 'median' for computing average strain over tracking points.
            'average_type': 'median',

            # Exclude nodes that have an end strain relatively far from zero.            
            'auto_exclude': False,
            
            # Choose 'SAX_LV' , 'SAX_RV' or 'LAX_BiV' (RV only possible for BiV simulations.)
            'plane_cavity': 'SAX_LV', 
            
            # Plot segment average and global average (optionally weighted) (1), 
            # or plot average and 25th and 75th quartiles (2),
            # or plot the average (3).
            # or plot the segments
            'plot_mode': 1,
            
            # If 'reorient_time' is True, the strain data is reordered such that
            # the first strains are (close to) zero.
            'reorient_time': True,
            
            # Specify how to reorient the time if 'reorient_time' is True. 
            # 1: reorient such that Ecc equals exactly zero at start.
            # 2: reorient such that t_act equals zero at start.
            # 3: reorient such that Ecc is the first moment where Ecc is close to zero.
            'reorient_time_mode': 1,

            # Specify the strain variables (keys in curves) to plot and corresponding 
            # names for plot titles. If None, they are attomatically chosen from 'plane_cavity' parameter.
            'strain_vars': None,
            'strain_names': None,
            
            # Specify cycle time in ms.
            't_cycle': 500,
            
            # If true: weight the average transmurally accoridng to Bovendeerd et al. 2009 
            # if plot_mode == 1. If false: 'average_type' will be used as average measure.
            'transmural_weight': False,
            
            # Specify selected row(s) and/or column(s) (in the data) to track.
            # Can be an intiger or a list/array of intigers.
            'trackpoints_row': None,
            'trackpoints_column': None,
            
            # Specify optimal segment numbers per strain measure.
            'optimal_segments': optimal_segments,
            
            # Specify range for strains plots : e.g. {'Ecc': [ymin, ymax]}.
            # Default is 'auto'.
            'strains_range': 'auto'
            }  
        return par
    
    def autofill_strain_vars(self):
        plane_cavity = self.parameters['plane_cavity']
        if plane_cavity == 'SAX_LV':
            self.parameters['strain_vars'] = ['Eccc', 'Errc', 'Ecrc']
            self.parameters['strain_names'] =  ['$E_{cc}$', '$E_{rr}$', '$E_{cr}$']
                            
        elif plane_cavity == 'SAX_RV':
            self.parameters['strain_vars'] = ['Ervccc', 'Ervrrc', 'Ervcrc']
            self.parameters['strain_names'] =  ['$E_{cc}$', '$E_{rr}$', '$E_{cr}$']
        
        elif plane_cavity == 'LAX_BiV':
            self.parameters['strain_vars'] = ['Elll', 'Ettl']
            self.parameters['strain_names'] =  ['$E_{ll}$', '$E_{tt}$']
            
        else:
            raise ValueError("Invalid 'plane_cavity' parameter.")
    
    @property
    def curves(self):
        if self._curves is None:
            self._curves = self._load_curves()
        return self._curves
 
    @property
    def data(self):
        if self._data is None:
            self._data = self._extract_curves_data()
        return self._data    
    
    @property
    def parameters(self):
        return self._parameters
    
    @staticmethod
    def create_rnorm(data):
        """
        Add normalized radial coordinate to data dict. 
        """
        x = data['x_coords']
        y = data['y_coords']
        
        r_inner = np.sqrt(x[0, :]**2 + y[0, :]**2)
        r_outer = np.sqrt(x[-1, :]**2 + y[-1, :]**2)
        
        r_norm = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                r_1 = r_inner[j]
                r_2 = r_outer[j]
                r_ij = np.sqrt(x[i, j]**2 + y[i, j]**2)
                r_norm[i, j] = 2*(r_ij - r_1)/(r_2 - r_1) - 1
                        
        data['r_norm'] = r_norm
        
        return data
    
    def get_all_filenames(self):
        """
        Returns a list of all the beats of the experiment. 
        This is useful when we are analyzing US strain data 
        and we have requested to average beats (see parameters).
        Otherwise, it returns a list with only one filename (the input filename)
        """
        filename = self.filename
        if self.type == 'US' and self.parameters['average_beats']:
            # Find all filenames that contain a beat for the current experiment.
            directory, file = os.path.split(filename)
            common_file_name = file.split('Frames')[0]
            return glob.glob(os.path.join(directory, common_file_name+'*curves.hdf5'))
        else:
            return [filename]

    def get_type(self):
        """
        Returns the type of strain data ('US' or 'FE').
        """
        if 'Frames' in self.filename:
            return 'US'
        else:
            return 'FE'
        
    def inter_beat_variability(self, fontsize=12):
        
        curves = self.curves
        strain_vars = self.parameters['strain_vars']
        strain_names = self.parameters['strain_names']

        t_cycle = self.parameters['t_cycle'] 

        plt.figure(figsize=[18,6])
        
        for i, key in enumerate(curves.keys()):
            curves_i = curves[key]
 
            # Create time array
            n_frames_i = curves_i[strain_vars[0]].shape[2]
            frames_i = np.arange(n_frames_i)
            
            time_i = frames_i/max(frames_i)*t_cycle 
           
            for j, var_j in enumerate(strain_vars):
                
                # Make correct subplot current.           
                plt.subplot(1, len(strain_vars), j+1)      
                
                data_ij = curves_i[var_j]
                
                average_ij = np.mean(data_ij.reshape(-1, data_ij.shape[-1]), axis=0)
                
                plt.plot(time_i, average_ij, label=str(i+1))
        
        for j in range(len(strain_vars)):
        
            # Make correct subplot current.           
            plt.subplot(1, len(strain_vars), j+1)      
            
            figure_make_up(title='{}'.format(strain_names[j]),
                           xlabel='Time [ms]', 
                           legend_title='Beat', 
                           fontsize=fontsize)

            plt.grid('on')    
            
        plt.suptitle('Inter-beat variability', fontsize=fontsize)

    def kmeans_segments(self):
        """
        Just tried this out. Not useful, though.
        """
        data = self.data
        Ecc = data['Ecc']
        segments = data['segments']
        
        n_features = Ecc.shape[-1]

        X = np.reshape(Ecc, (-1, n_features))
        
        initial_centers = np.zeros((6, n_features))
        for i in range(6):
            data_i = Ecc[segments[:, :, i]]
            initial_centers[i, :] = np.mean(data_i, axis=0)
            
        n_clusters = 6      
        kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(X)
        
        clusters = kmeans.labels_.reshape(11, -1)
        
        segments = np.zeros((Ecc.shape[0], Ecc.shape[1], n_clusters))
        for i in range(n_clusters):
            segments[:, :, i] = clusters == i
            
        data['segments'] = segments.astype(bool)
        
    def pick_tracking_points(self, data):
        row = self.parameters['trackpoints_row']
        col = self.parameters['trackpoints_column']
        
        segments = data['segments']
        
        segments_new = np.zeros_like(segments)
        if col is not None and row is not None:
            segments_new_ = np.zeros_like(segments)
            segments_new_[row, :, :] = segments[row, :, :]
            segments_new[:, col, :] = segments_new_[:, col, :]
        
        elif row is not None:
            segments_new[row, :, :] = segments[row, :, :]
                
        elif col is not None:
            segments_new[:, col, :] = segments[:, col, :]
            
        else:
            # Include all.
            return data
            
        data['segments'] = segments_new.astype(bool)
        
        return data
        
    def plot_strains(self, fontsize=12, figname=None, figure_shape='row', 
                     **kwargs):
        """
        Plots the strains that are loaded with given parameter settings.
        
        Returns:
            variance_array (array): a variance measure of the averaged nodes per strain.
            max_strains (dict): Returns a maximum (or minimum) value per strain.
            **kwargs (optional): key worded arguments passed to plot functions.
        """
        
        data = self.data
        
        # Extract parameters.
        average_mode = self.parameters['average_mode']
        average_type = self.parameters['average_type']
        auto_exclude= self.parameters['auto_exclude']
        plot_mode = self.parameters['plot_mode']
        strain_vars = self.parameters['strain_vars']
        strain_names = self.parameters['strain_names']
        optimal_segments = self.parameters['optimal_segments']
        strains_range = self.parameters['strains_range']
        
        # Extract fixed data.
        time = data['time']
        segments = data['segments']
        
        max_strains = {}
                
        variance_array = np.zeros(len(strain_vars))
        for i, var_i in enumerate(strain_vars):
            
            # Initialize.
            max_strains[var_i+'_per_segment'] = []
            
            # Extract current strain measure.
            data_i = data[var_i]
            
            # Make correct figure and subplot current.       
            if figname is None or len(figname) == 0:
                figname = os.path.split(self.filename)[0]
                
            if figure_shape == 'column':
                plt.figure(figname, figsize=(6,13))
                plt.subplot(len(strain_vars), 1, i+1)
            
            else: 
                plt.figure(figname, figsize=[18,6])
                plt.subplot(1, len(strain_vars), i+1)
            
            if average_mode == 1:
                # Include all segments always.
                included_segments_i = list(np.arange(segments.shape[-1]) + 1)
                masks_i = segments
                
            elif average_mode == 2:
                # Include optimal segments per strain measure.
                included_segments_i = optimal_segments[var_i]
                idx_segments = np.array(included_segments_i) - 1  # segment i is located at i-1.
                masks_i = segments[:, :, idx_segments] 
            
            else:
                raise ValueError('Invalid average_mode parameter.')
                
            if auto_exclude:
                # Automatically choose which tracking points to include.
                # Outliers are excluded. An outlier is defined as by rule of thumb:
                # A tracking point that ends further than 1.5 * IQR away from Q1 or Q3.
                
                # Compute quartiles and inter quartile range.
                q1 = np.percentile(data_i[:, :, -1], 25)
                q3 = np.percentile(data_i[:, :, -1], 75)
                iqr = q3 - q1
                                
                dist_to_q1 = (q1 - data_i[:, :, -1])
                dist_to_q3 = (data_i[:, :, -1] - q3)
                                
                # Define a tolerance for the absolute distance allowed.
                tolerance = 1.5*iqr                
   
                # Apply a mask to the tracking points.
                masks_include = (dist_to_q1 < tolerance) * (dist_to_q3 < tolerance)
                nc, nr, nf = np.shape(data_i)
                masks_i *= masks_include.reshape(nc, nr, 1)  # mask_i are the masks per segment.
                
                print('{:3.1f} % of ALL tracking points included.'.format(
                        100*np.sum(masks_i)/(nc*nr)))
                
            # Collect strain per included segment.
            included_data_i = np.zeros((0, data_i.shape[-1]))
            included_r_norm_i = np.zeros((0))
            for j in range(masks_i.shape[-1]):
                mask_ij = masks_i[:, :, j]  
                data_ij = data_i[mask_ij]  # size = (npoints, nframes)
                r_norm_ij = data['r_norm'][mask_ij]
                included_data_i = np.vstack((included_data_i, data_ij))
                included_r_norm_i = np.concatenate((included_r_norm_i, r_norm_ij), axis=0)
                
                # Plot average strain in segment over time.
                segment_number = included_segments_i[j]
                
                if not self.parameters['transmural_weight']:
                    r_norm_ij = None

                avg_data_ij = self.points_average(data_ij, r_norm_ij, average_type=average_type)
                
                if plot_mode == 1:
                    plt.plot(time, avg_data_ij, ':', color='C{}'.format(segment_number-1), 
                             label='{}'.format(segment_number), linewidth=fontsize/6)
                    
                # store maximum or minimum average strain per segment.
                if var_i in ['Errc', 'Ervrrc', 'Ettl']:
                    # Maxima
                    max_strains[var_i+'_per_segment'].append(max(avg_data_ij))
                else:
                    # Minima
                    max_strains[var_i+'_per_segment'].append(min(avg_data_ij))
                
            # Compute variance.
            std_data_i_frame = np.std(included_data_i, axis=0)

            # Choose variance measure here.
            variance_measure = np.mean(std_data_i_frame)
            print('SD averaged over frames for {}: {:.4g}'.format(var_i, variance_measure))

            # store variance measure.
            variance_array[i] = variance_measure
            
            # Compute average strain.
            if not self.parameters['transmural_weight']:
                included_r_norm_i = None
            average_strain = self.points_average(included_data_i, included_r_norm_i, average_type=average_type)
                
            # store maximum or minimum average strain.
            if  var_i in ['Errc', 'Ervrrc', 'Ettl']:
                # Maxima
                max_strains[var_i] = max(average_strain)
            else:
                # Minima
                max_strains[var_i] = min(average_strain)
            
            if plot_mode == 1:
                # Default plot arguments
                plot_args = {'color': 'k',
                             'label': 'All', 
                             'linewidth': fontsize/6}
                
                # Update with user defined arguments.
                plot_args.update(kwargs)
                
                # Plot global average strain (of included segments).
                plt.plot(time, average_strain, **plot_args)
                
            elif plot_mode == 2:
                # Default plot arguments
                plot_args = {'linestyle': '-',
                             'linewidth': fontsize/6}
                
                # Update with user defined arguments.
                plot_args.update(kwargs)
                
                # Plot average and 25th and 75 quratile ranges.
                plt.plot(time, average_strain, **plot_args)
                
                # Shade interquartile range.
                fill_args = {}
                if 'color' in plot_args.keys():
                    fill_args['color'] = plot_args['color']
                q1 = np.percentile(included_data_i, 25, axis=0)
                q3 = np.percentile(included_data_i, 75, axis=0)
                plt.fill_between(time, q3, q1, alpha=.5, **fill_args)
                
            elif plot_mode == 3:
                # Default plot arguments
                plot_args = {'color': 'k',
                             'linewidth': fontsize/6}
                
                # Update with user defined arguments.
                plot_args.update(kwargs)
                
                # Plot average.
                plt.plot(time, average_strain, **plot_args)
                    
            # Make up figure.
            if plot_mode == 1 and i==len(strain_vars)-1:
                legend = plt.legend(title='Segment', fontsize=fontsize-2, loc=(1.04,0))
                legend.get_title().set_fontsize(fontsize-2)
                
            plt.xlabel('Time [ms]', fontsize=fontsize)
#            plt.ylabel('{} [-]'.format(strain_names[i]), fontsize=fontsize)
            plt.tick_params(labelsize=fontsize-2)
            plt.title('{}'.format(strain_names[i]), fontsize=fontsize+2)
            plt.grid('on')
            if not strains_range is 'auto':
                plt.ylim(strains_range[var_i])
            
#            # Create histogram of strains at end frame.
#            plt.figure(os.path.split(self.filename)[1]+'_hist', figsize=[12,6])
#            plt.subplot(1, len(strain_vars), i+1)
#            hist_data = included_data_i[:,round(3*np.shape(included_data_i)[1]/5)]
#            plt.hist(hist_data, bins=50)
#            plt.axvline(x=np.mean(hist_data), color='C1', label='mean')
#            plt.axvline(x=np.median(hist_data), color='C2', label='median')
#            plt.xlabel('{} [-]'.format(strain_names[i]), fontsize=fontsize)
#            plt.tick_params(labelsize=fontsize-2)
#            plt.title('{} at 300 ms'.format(strain_names[i]), fontsize=fontsize+2)
 
        # Save figures.
        head, tail = os.path.split(self.filename)
        dir_out = os.path.join(head, 'figures')
        
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        
        plane_cavity = self.parameters['plane_cavity']
        plt.figure(figname)
        # Specify filename. Note that it should not be too long, it will give a FileNotFoundError
        figure_filename = os.path.join(dir_out, tail.split('curves')[0]+'_strains_mode_{}_{}.png'.format(plot_mode, plane_cavity))
        plt.savefig(figure_filename,  dpi=300, bbox_inches="tight")

#        plt.figure(os.path.split(self.filename)[1]+'_hist')   
#        plt.legend(fontsize=fontsize-2)        
#        figure_filename = os.path.join(dir_out, tail.split('curves')[0]+'_strains_hist_{}.png'.format(plane_cavity))
#        plt.savefig(figure_filename,  dpi=300, bbox_inches="tight")
            
        return variance_array, max_strains
    
    @staticmethod
    def points_average(data, r_norm=None, average_type='mean'):
        
        if r_norm is None:
            # Compute average strain.
            if average_type == 'median':
                average_strain = np.median(data, axis=0)
            elif average_type == 'mean':
                average_strain = np.mean(data, axis=0)
            else:
                raise ValueError('Unknown average_type')            
            return average_strain
        else:
            tot = sum((1 - r_norm**2)**2)
            weights = (1 - r_norm**2)**2/tot
            weights = np.tile(weights.reshape(-1, 1), [1, data.shape[1]])
            return np.sum(data*weights, axis=0)
        
    def plot_segments(self):
        data = self.data
        segments = data['segments']
        x_coords = data['x_coords']
        y_coords = data['y_coords']
        
        for i in range(segments.shape[-1]):
            seg_i = segments[:, :, i]
            x_i = x_coords[seg_i]
            y_i = y_coords[seg_i]
            plt.scatter(x_i, y_i, label=i+1)
        plt.legend(title='Segment')
        plt.axis('equal')

    def _load_curves(self):
        """
        Returns a dictionary with a dictionary that holds the curves 
        information for every file in self.get_all_filenames()
        """
        curves = {}
        for filename in self.get_all_filenames():
            curves_file = {}
            with h5py.File(filename, 'r') as f:
                for dataset in f.keys():
                    # Extract data as numpy array.
                    data = f[dataset][:]
                    
                    # Convert segments to logicals.
                    if 'segments' in dataset:
                        data = data.astype(bool)
                    
                    attributes = f[dataset].attrs
                    if 'shape' in attributes:
                        # Exported from FEniCS: reshape
                        data = data.reshape(*attributes['shape'].astype(int))
                        
                    else:
                        # Exported from MATLAB: Transpose so that the array has the same shape as in MATLAB.
                        data = np.transpose(data)
                    
                    # Add data to dict.
                    curves_file[dataset] = data  
                    
            
            if self.parameters['reorient_time']:
                # Reorient time (smallest strain at t=0)
                curves_file = self._reorient_time(curves_file)
                
            curves[filename] = curves_file
            
        return curves
    
    def _extract_curves_data(self):
        """
        Returns a dictionary with for 'segments', 'x_coords', 'y_coords', and additional variables.
        The variables are averaged per tracking point over the beats if requested in the parameters.
                           
        Returns:
            data (dict): dictionary with (averaged) output variables, segment masks, and coordinates.
        """
        
        curves = self.curves
        
        # Extract curves of input file.
        curves_0 = curves[self.filename]
        
        # Initialize output variables.
        plane_cavity = self.parameters['plane_cavity']
        if plane_cavity == 'SAX_LV':
            data = {
                'segments': curves_0['segments'],
                'x_coords': curves_0['xc'],
                'y_coords': curves_0['yc']}
          
        elif plane_cavity == 'SAX_RV':
            data = {
                'segments': curves_0['segmentsrv'],
                'x_coords': curves_0['xrvc'],
                'y_coords': curves_0['yrvc']}  

        elif plane_cavity == 'LAX_BiV':
            data = {
                'segments': curves_0['segmentsl'],
                'x_coords': curves_0['xl'],
                'y_coords': curves_0['zl']}     

        else:
            raise ValueError("Invalid 'plane_cavity' parameter.")
        
        n_files = len(curves)   
        print('Averaging {} files/beats.'.format(n_files))
         
        # Specify the variables to extract for each file.
        strain_vars = self.parameters['strain_vars']
       
        # Extract each variable.
        for i, var_i in enumerate(strain_vars):
            data[var_i] = curves_0[var_i]/n_files 
            
        n_frames_ref = curves_0[strain_vars[0]].shape[2]
        frames_ref = np.arange(n_frames_ref)
        data['frames'] = frames_ref
        
        t_cycle = self.parameters['t_cycle'] 
        data['time'] = frames_ref/max(frames_ref)*t_cycle
        
        # Iterate over loaded curves files.
        for key in curves.keys():
            if key == self.filename:
                # We have this one already.
                continue
            
            # Extract curves of current file.
            curves_i = curves[key]
            
            # Extract each variable.
            for var_i in strain_vars:
                y_i = curves_i[var_i]
              
                # Number of frames for current file.
                n_frames_i = y_i.shape[2]
                frames_i = np.arange(n_frames_i)
                
                # Interpolate strains to reference frames (some beats may consists of one more
                # beat, we assume here that the first and last frame always correspond to each other)
                fy = scipy.interpolate.interp1d(frames_i, y_i, fill_value='extrapolate')      
                y_resample = fy(frames_ref)

                data[var_i] += y_resample/n_files
        
        # Add normalized radial coordinates.
        data = self.create_rnorm(data)
        
        # Add a specific tracking points
        data = self.pick_tracking_points(data)
        
        return data
    
    def _reorient_time(self, curves):
        Ecc = curves['Eccc']
        
        # Number of time points.
        n = Ecc.shape[2]
        
        if self.parameters['reorient_time_mode'] == 1:
            # Start at index where strain is closest to zero.
            idx_start = np.argmin(abs(Ecc[0,0,:]))
        elif self.parameters['reorient_time_mode'] == 2:
            # Start at indices where t_act = 0.
            # Load results file (assume it is located 2 directories up).
            results_filename = os.path.join(os.path.split(
                    os.path.split(self.filename)[0])[0], 'results.csv')
            results = load_reduced_dataset(results_filename)
            t_act = np.array(results['t_act'])
            idx_start = np.argmin(abs(t_act))   
        elif self.parameters['reorient_time_mode'] == 3:
            # Start at first index where Ecc is close to zero.
            Ecc_average = self.points_average(Ecc.reshape(-1, n), 
                                              average_type=self.parameters['average_type'])
            idx_start = np.argmax(np.diff(np.sign(Ecc_average)))
        else:
            raise ValueError('Invalid reorient_time_mode.')
        
        # Permute time for all relevant variables.
        for key in curves.keys():
            if (type(curves[key]) == np.ndarray) and (curves[key].shape[-1] == n):
                curves[key] = shift_data(curves[key], idx_start, axis=-1)[0]
        return curves
    
from itertools import cycle
def get_linecycler():
    lines = ["-","--","-.",":"]
    return cycle(lines)

def get_markercycler():
    markers = ["o","^","v","s"]
    return cycle(markers)

def get_colorcycler():
    colors = ['C{}'.format(i) for i in range(12)]
    return cycle(colors)

def rescale_yaxis(ax1, ax2):
    a = ax1[3]
    b = ax1[2]
    c = ax2[3]
    d = ax2[2]
    g = 1/20  # scale padding
    pad = c*g*(1 - b/a)
    c += pad
    d = (b*(c) + a*pad)/a - pad
    return d, c
    

def reorder_array(ar, idx, axis=0):
    """
    Reorders the array ar given by new indices idx along a specific axis.
    """
    # Swap the axis, such that the axis to reorder is the first axis.
    ar_swapped = np.swapaxes(ar, axis, 0)
            
    # Reorder array along first axis.
    ar_swapped_reordered = ar_swapped[idx]
    
    # Swap axes back to original shape.
    ar_reordered = np.swapaxes(ar_swapped_reordered, axis, 0)
    
    return ar_reordered


def shift_data(data, shift, axis=0):
    """
    Shift data leftward (if axis = 1) by an amount shift. If shift is an 
    integer between 0 and data.shape[axis], then the data is shifted along the
    specified axis such that it starts at index shift.
    """        
    len_data = np.shape(data)[axis]
    
    # Check if shift is within data length.
    if abs(shift) > len_data:
        raise ValueError('Shift exceeds data length.')
        
    if shift < 0:
        shift += len_data
        
    # Determine new indices.
    idx_shift = np.array(list(range(shift, len_data)) + list(range(0, shift)))
    
    # Shift data
    data_shifted = reorder_array(data, idx_shift, axis=axis)
    
    return data_shifted, idx_shift


class subplot_axis(object):
    def __init__(self, r, c, i, **kwargs):
        self.axis = plt.subplot(r, c, i, **kwargs)
        self._linecycler = get_linecycler()
        self._markercycler = get_markercycler()
        self._colorcycler = get_colorcycler()
    
    def nextlinestyle(self):
        return next(self._linecycler)

    def nextmarker(self):
        return next(self._markercycler)
    
    def nextcolor(self):
        return next(self._colorcycler)
    
    def new_make_up(self, title=None, fontsize=12, new_labels=None):
        ax = self.axis
        plt.sca(ax)
        
        figure_make_up(title=title, fontsize=fontsize, create_legend=False)
        
        # Replace labels with provided labels.
        try: 
            old_labels = ax.get_legend().get_texts()
        except:
            figure_make_up(fontsize=fontsize, 
            create_legend=True) 
            old_labels = ax.get_legend().get_texts()
        
        for i, new_lab in enumerate(new_labels):
            old_labels[i].set_text(new_lab)