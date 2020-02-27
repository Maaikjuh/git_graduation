# -*- coding: utf-8 -*-
"""
This module provides classes that construct common figures.
"""
import csv
import warnings

from dolfin import parameters, Function, project, det, dot, dx, sign, inv, Identity, sqrt, as_tensor, \
    VectorFunctionSpace
from dolfin.cpp.common import mpi_comm_world, MPI, mpi_comm_self
from dolfin.cpp.io import XDMFFile, HDF5File

from cvbtk.windkessel import kPa_to_mmHg
from cvbtk.models import FiberReorientation
from cvbtk.basis_vectors import GeoFunc
from cvbtk.dataset import Dataset
from cvbtk.geometries import BiventricleGeometry, LeftVentricleGeometry
from cvbtk.mechanics import deformation_gradient, ArtsKerckhoffsActiveStress
from cvbtk.resources import reference_biventricle, reference_left_ventricle_pluijmert
from cvbtk.routines import create_materials, check_heart_type_from_inputs, create_model, \
    load_model_state_from_hdf5
from cvbtk.utils import print_once, read_dict_from_csv, vector_space_to_scalar_space, \
    global_function_average, save_to_disk, quadrature_function_space

from matplotlib import pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec

import os
import numpy as np


__all__ = [
    'HemodynamicsPlot',
    'HemodynamicsPlotDC',
    'GeneralHemodynamicsPlot',
    'Export'
           ]


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
        self._ax['qt'].plot(df['time'], df['qmv'], label='Mitral')
        self._ax['qt'].plot(df['time'], df['qao'], label='Aortic')
        self._ax['qt'].plot(df['time'], df['qper'], label='Peripheral')
        if 'qlvad' in df.keys():
            self._ax['qt'].plot(df['time'], df['qlvad'], label='LVAD')

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

    def plot_function(self):
        """
        Plot the ventricle function in terms of stroke volume, maximum pressure,
        and work for every cycle.
        """
        # Find the number of the first and last cycle.
        start = min(self._df['cycle'])
        stop = max(self._df['cycle'])
        cycles = list(range(start, stop))

        # Extract SV and maximum pressure of the systemic circulation.
        strokevolumes = []
        pmax = []
        work = []
        for cycle in cycles:
            df = self._df[self._df['cycle'] == int(cycle)]
            # Find begin phase 2.
            idx_ed = min(np.where(df['phase'].values == 2)[0])
            # Find begin phase 4.
            idx_ee = min(np.where(df['phase'].values == 4)[0])

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
        cycles = list(range(start, stop))

        # Extract the SV of the systemic and pulmonary circulation.
        strokevolumes = {'_s': [],
                         '_p': []}
        pmax = {'_s': [],
                '_p': []}
        work = {'_s': [],
                '_p': []}
        for cycle in cycles:
            df = self._df[self._df['cycle'] == int(cycle)]
            for ps in ['_s', '_p']:
                # Find begin phase 2.
                idx_ed = min(np.where(df['phase' + ps].values == 2)[0])
                # Find begin phase 4.
                idx_ee = min(np.where(df['phase' + ps].values == 4)[0])

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

        # Maximum cavity pressure.
        plt.figure() 
        ax1 = plt.subplot(1, 3, 1)
        ax1.plot(cycles, kPa_to_mmHg(np.array(pmax['_s'])), 'r')
        ax1.set_xlabel('Cardiac cycle [-]')
        ax1.set_ylabel('LV pressure [mmHg]', color='r')
        ax1.tick_params('y', colors='r')
        ax2 = ax1.twinx()
        ax2.plot(cycles, kPa_to_mmHg(np.array(pmax['_p'])), '--b')
        ax2.set_ylabel('RV pressure [mmHg]', color='b')
        ax2.tick_params('y', colors='b')
        plt.title('Maximum cavity pressure')
        lv_label = mlines.Line2D([], [], color='red', label='LV')
        rv_label = mlines.Line2D([], [], color='blue', linestyle='--', label='RV')
        plt.legend(handles=[lv_label, rv_label])

        # Stroke volume.
        ax3 = plt.subplot(1, 3, 2)
        ax3.plot(cycles, strokevolumes['_s'], 'r')
        ax3.set_xlabel('Cardiac cycle [-]')
        ax3.set_ylabel('Stroke volume [ml]')
        ax3.tick_params('y')
#        ax4 = ax3.twinx()
        ax3.plot(cycles, strokevolumes['_p'], '--b')
#        ax4.set_ylabel('RV volume [ml]', color='b')
#        ax4.tick_params('y', colors='b')
        plt.title('Stroke volumes')
        plt.legend(handles=[lv_label, rv_label])

        # Work.
        ax5 = plt.subplot(1, 3, 3)
        ax5.plot(cycles, work['_s'], 'r')
        ax5.set_xlabel('Cardiac cycle [-]')
        ax5.set_ylabel('LV work [J]', color='r')
        ax5.tick_params('y', colors='r')
        ax6 = ax5.twinx()
        ax6.plot(cycles, work['_p'], '--b')
        ax6.set_ylabel('RV work [J]', color='b')
        ax6.tick_params('y', colors='b')
        plt.title('Work')
        plt.legend(handles=[lv_label, rv_label])
        
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


# The following functions may actually be collected in a new a module (e.g. postprocessing).

class Export(object):
    """
    Class for easy exporting of saved and derived variables from the results.hdf5 file to
    XDMF files, readable by ParaView. Compatible with LeftVentricle and Biventricle simulations.

    NOTE: DOES NOT RUN IN PARALLEL. Code somehow not closes the XDMF files properly and hangs at the end.

    Args:
        directory (str): path to the directory with the csv and hdf5 files.
            Expects the following files in `directory`:
                results_cycle_x.hdf5
                results.csv
                inputs.csv

        *args:  Strings that specify what to export.
                If nothing specified a default set of variables will be exported.
                Options:
                    'u', (displacement)
                    'ef', (fiber vector)
                    'ls', (sarcomere length)
                    'lc', (contractile element length)
                    'active_stress',
                    'von_mises',
                    'fiber_stress',
                    'passive_fiber_stress',
                    'J', (det(F))
                    'sign_active_stress',
                    'sign_ls_min_lc',
                    'Ecc', (circumferential strain)
                    'Err', (radial strain)
                    'Ecr', (circumferential-radial shear strain)
                    'Ell', (longitudinal strain)
                    'curves_file' (hdf5 file similar to curves file from lifetec experiments).
                 NOTE: a csv file is exported with the average fiber stress and strain for every timestep.
                 # TODO Export nodal values of each exported function field per timestep (using the HDF5 format?)

        time_interval (tuple or list, optional): Specify start and end time to load (ms).
                                                 Requires results.csv to be present in `directory`.

        cycle (int, optional): Specify which cycle to load. If time_interval is given, the specified time_interval will
                               overrule the specified cycle.

        strain_reference (str, optional): Specify which state to use as reference for the strains.
                                          Options:
                                            'begin_ic' : begin isovolumic contraction phase (of LV).
                                            'stress_free' : geometry in stress-free state.

        If no cycle or time_interval is specified, the last complete cycle is loaded by default.

        **kwargs: Optional keyword arguments that are passed to the inputs (handy if no inputs.csv file is available)

    Syntax example:
    Export('.', 'u', 'fiber_stress', 'ls', 'lc', strain_reference='begin_ic')

    """
    def __init__(self, directory, *args, time_interval=None, cycle=None, strain_reference='stress_free', **kwargs):

        # Save directory (with input files).
        self.directory = directory

        # Save the names of the variables to export (specified by *args).
        if len(args) != 0:
            self.variables = args
        else:
            # No variable specified, save default set of variables.
            self.variables = self.default_variables()
        # Remove duplicates.
        self.variables = list(set(self.variables))

        # Read inputs.
        if os.path.exists(os.path.join(self.directory, 'inputs.csv')):
            self.inputs = read_dict_from_csv(os.path.join(self.directory, 'inputs.csv'))
        else:
            if MPI.rank(mpi_comm_world()) == 0:
                warnings.warn('No inputs file found. Expecting necessary inputs passed as key-word arguments.')
            self.inputs = {}

        # Update inputs with user-defined arguments.
        self.inputs.update(kwargs)

        # Set the proper global FEniCS parameters.
        parameters.update({'form_compiler': self.inputs['form_compiler']})

        # Read csv file.
        self.data = self.load_reduced_dataset(time_interval, cycle)

        if min(self.data['cycle']) != max(self.data['cycle']):
            raise ValueError('Cannot load results from multiple cycles. Please choose a time frame within 1 cycle.')
        else:
            cycle_ = min(self.data['cycle'])

        # Directory for output files.
        if time_interval is not None:
            self.dir_out = os.path.join(directory, 'time_{}-{}'.format(time_interval[0], time_interval[1]))
        else:
            self.dir_out = os.path.join(directory, 'cycle_{}'.format(cycle_))

        # Specify the strain reference in the output directories name.
        self.dir_out += '_{}_ref'.format(strain_reference)

        # Save strain_reference.
        self.strain_reference = strain_reference

        if MPI.rank(mpi_comm_world()) == 0:
            if not os.path.exists(self.dir_out):
                os.makedirs(self.dir_out)
        # Synchronize.
        MPI.barrier(mpi_comm_world())

        print_once('Exporting to {} directory.'.format(self.dir_out))

        # Specify the results hdf5 file.
        self.results_hdf5_name = os.path.join(self.directory, 'results_cycle_{}.hdf5'.format(cycle_))

        if not os.path.exists(self.results_hdf5_name):
            raise ValueError('Results file {} does not exists.'.format(self.results_hdf5_name))

        # This will hold a handle to the open results file.
        self.results_hdf5 = None

        # Load model
        self.model = self.create_model()

        # Create attribute to access XDMF files.
        self.xdmf_files = {}

        # Create attribute to access FEniCS functions.
        self.functions = {}

        # Create attribute for active stress scalar.
        self._active_stress = None

        # Create attribute for GL strain.
        self._E_cyl = None
        self._E_car = None

        # Create an attribute for the curves dictionary, similar to the curves struct in MATLAB.
        self._curves = None

        # Create dictionary with average values for fiber stress and fiber strain.
        self.avg_fiber_stress = {}
        self.avg_fiber_strain = {}

        # Run main program.
        self.main()

    def active_stress(self):
        """
        Return the active stress.
        """
        if self._active_stress is None:
            contractility = self.model.parameters['contractility']
            p = self.model.active_stress.active_stress_scalar(self.model.u)
            f = self.model.active_stress.ls / self.inputs['active_stress']['ls0']

            # "Derive" the scalar "2nd Piola-Kirchhoff stress" value.
            # self._active_stress = contractility * p / f

            # Compute the Cauchy stress value.
            self._active_stress = contractility * p * f

        return self._active_stress

    def passive_fiber_stress(self):
        """
        Return the passive fiber stress.
        """
        # Get the PK2 stress tensor for the material model.
        S = self.model.material.piola_kirchhoff2(self.model.u)

        # Convert to Cauchy stress.
        F = deformation_gradient(self.model.u)
        return F*S*F.T/det(F)

    def cylindrical_green_lagrange_strain(self):
        """
        Returns Green-Lagrange strain wrt cylindrical basis (er, ec, -ez).
        """
        if self._E_cyl is None:
            self._E_cyl = self.compute_green_langrange_strain(basis='cylindrical')
        return self._E_cyl

    def cardiac_green_lagrange_strain(self):
        """
        Returns Green-Lagrange strain wrt cadraic basis (ec, el, et).
        """
        if self._E_car is None:
            self._E_car = self.compute_green_langrange_strain(basis='cardiac')
        return self._E_car

    @staticmethod
    def change_reference(u_ref, F, e1, e2, e3):
        # Deformation gradient tensor to new reference, wrt stress-free.
        F_ref = deformation_gradient(u_ref)

        # Deformation gradient tensor, wrt new ref.
        F_exp = F * inv(F_ref)

        # Cylindrical basis vectors wrt new reference, non-normalized.
        e1_exp_, e2_exp_, e3_exp_ = F_ref * e1, F_ref * e2, F_ref * e3

        # Cylindrical basis vectors wrt new reference, normalized.
        e1_exp = e1_exp_ / sqrt(dot(e1_exp_, e1_exp_))
        e2_exp = e2_exp_ / sqrt(dot(e2_exp_, e2_exp_))
        e3_exp = e3_exp_ / sqrt(dot(e3_exp_, e3_exp_))

        return F_exp, e1_exp, e2_exp, e3_exp

    def compute_cauchy_stress(self):
        # Get the PK2 stress tensor for the material model.
        S = self.model.material.piola_kirchhoff2(self.model.u)
        # Get the PK2 stress tensor for the active stress model if needed.
        if self.model.active_stress is not None:
            contractility = self.model.parameters['contractility']
            S = S + contractility*self.model.active_stress.piola_kirchhoff2(self.model.u)

        # Convert to Cauchy stress.
        F = deformation_gradient(self.model.u)
        return F*S*F.T/det(F)

    def compute_green_langrange_strain(self, basis='cylindrical'):
        """
        Returns Green-Lagrange strain in cylindrical basis
        with repect to begin ejection.

        Args:
            basis: Specify the basis in which to compute E.
                   Options:
                     'cylindrical' (er, ec, -ez)
                     'cardiac' (ec, el, et)
        """
        data = self.data
        model = self.model

        # Function space.
        V = model.u.ufl_function_space()

        if not self.strain_reference == 'stress_free':
            if self.strain_reference == 'begin_ic':
                # Read in the displacements at begin isovolumic contraction (phase 2) to use for a reference.
                if 'vlv' in data.keys():
                    phase_key = 'phase'
                elif 'vcav_s' in data.keys():
                    phase_key = 'phase_s'
                else:
                    raise ValueError('Cannot find LV phase in results.csv file.')
                vector_number_ref = min(data[data[phase_key] == 2]['vector_number'])
            else:
                raise ValueError('Invalid reference state for strains specified.')

            u_ref = Function(V)
            self.results_hdf5.read(u_ref, 'displacement/vector_{}'.format(str(vector_number_ref)))

        # GLOBAL displacement and deformation gradient tensor wrt REFERENCE.
        F = deformation_gradient(model.u)

        # Basis vectors wrt REFERENCE.
        if basis == 'cylindrical':
            try:
                # Define on quadrature space to prevent interpolation.
                # Unfortunately, defining it on the quadrature space takes some time.
                # Another option is to define the vectors on a Lagrangian function space and
                # normalize and orthogonalize the interpolated values by incorporating the
                # normalisation and orthogonalisation in a UFL definition of the basis on a Lagrangian space.
                V_q = quadrature_function_space(model.geometry.mesh())
                ez, er, ec = [_.to_function(V_q) for _ in model.geometry.cylindrical_vectors()]
            except AttributeError:
                ez, er, ec = model.geometry.cylindrical_vectors()
            e1, e2, e3 = er, ec, -ez

        elif basis == 'cardiac':
            try:
                V_q = quadrature_function_space(model.geometry.mesh())
                ec, el, et = [_.to_function(V_q) for _ in model.geometry.cardiac_vectors()]
            except AttributeError:
                ec, el, et = model.geometry.cardiac_vectors()
            e1, e2, e3 = ec, el, et

        else:
            raise ValueError('Invalid basis specified.')

        # Check if we should change the strain reference.
        if self.strain_reference == 'stress_free':
            F_exp = F
            M_exp = as_tensor([e1, e2, e3])
        else:
            F_exp, e1_exp, e2_exp, e3_exp = self.change_reference(u_ref, F , e1, e2, e3)
            M_exp = as_tensor([e1_exp, e2_exp, e3_exp])

        # GL strain.
        E = 0.5 * (F_exp.T * F_exp - Identity(3))

        # Finally, GL in cylindrical coordinates.
        E_exp = M_exp * E * M_exp.T

        return E_exp

    def close_xdmf_files(self):
        for vari in self.xdmf_files.keys():
            self.xdmf_files[vari].close()

    def curves(self):
        if self._curves is None:
            self._curves = self.create_curves()
        return self._curves

    def create_model(self):

        geometry_inputs = self.inputs['geometry']
        results_hdf5 = self.results_hdf5_name
        heart_type = check_heart_type_from_inputs(self.inputs)

        # Set the 'load_fiber_field_from_meshfile' parameter to True.
        geometry_inputs['load_fiber_field_from_meshfile'] = True

        # Load geometry from results file.
        try:
            print_once('Loading mesh from results file...')
            if heart_type == 'Biventricle':
                geometry = BiventricleGeometry(meshfile=results_hdf5, **geometry_inputs)
            elif heart_type == 'LeftVentricle':
                geometry = LeftVentricleGeometry(meshfile=results_hdf5, **geometry_inputs)
            else:
                raise ValueError('Unknown heart_type "{}".'.format(heart_type))

        except RuntimeError:
            print_once('Failed to load mesh from result.hdf5 file. Loading reference mesh...')
            # Load the reference geometry otherwise.
            res = self.inputs['geometry']['mesh_resolution']
            if heart_type == 'Biventricle':
                geometry = reference_biventricle(resolution=res, **geometry_inputs)
            elif heart_type == 'LeftVentricle':
                geometry = reference_left_ventricle_pluijmert(resolution=res, **geometry_inputs)
            else:
                raise ValueError('Unknown heart_type "{}".'.format(heart_type))

        # Create the model.
        model = create_model(geometry, self.inputs, heart_type)

        # Add materials.
        create_materials(model, self.inputs)

        return model

    def create_lax_track_points_biv(self):

        geo_pars = self.model.geometry.parameters['geometry']
        R_1 = geo_pars['R_1']
        R_2 = geo_pars['R_2']
        R_1sep = geo_pars['R_1sep']
        R_2sep = geo_pars['R_2sep']
        R_3 = geo_pars['R_3']
        R_4 = geo_pars['R_4']
        Z_1 = geo_pars['Z_1']
        Z_2 = geo_pars['Z_2']
        Z_3 = geo_pars['Z_3']
        Z_4 = geo_pars['Z_4']
        h = geo_pars['h']

        z_max = h
        z_min = -Z_1

        # Do not go to the boundaries exactly.
        length = z_max - z_min
        z_max -= length * 0.02
        z_min += length * 0.02

        nz = 20
        nx = 5

        # Space to stay away from the boundary of the mesh.
        edge_space = (R_4 - R_3) * 0.1

        z_sep_high = -np.sqrt((1 - (R_3 / R_2sep) ** 2) / ((1 / Z_2) ** 2 - (R_3 / Z_3 / R_2sep) ** 2))
        z_sep_low = -np.sqrt((1 - (R_4 / R_2sep) ** 2) / ((1 / Z_2) ** 2 - (R_4 / Z_4 / R_2sep) ** 2))

        x = np.zeros((nz, nx * 3))
        z = np.zeros((nz, nx * 3))
        segments = np.zeros((nz, nx * 3, 3))

        for i, zi in enumerate(np.linspace(z_min, z_max, nz)):
            # LV
            x_epi = R_2 * np.sqrt(1 - (zi / Z_2) ** 2)
            x_endo = R_1 * np.sqrt(1 - (zi / Z_1) ** 2)
            for j, xi in enumerate(np.linspace(x_endo + edge_space, x_epi - edge_space, nx)):
                x[i, j] = xi
                z[i, j] = zi
                segments[i, j, 0] = 1

            # Septum
            if zi > z_sep_low:
                x_epi = -R_2sep * np.sqrt(1 - (zi / Z_2) ** 2)
                x_endo = -R_1sep * np.sqrt(1 - (zi / Z_1) ** 2)
                for j, xi in enumerate(np.linspace(x_epi + edge_space, x_endo - edge_space, nx)):
                    x[i, j + 1 * nx] = xi
                    z[i, j + 1 * nx] = zi
                    segments[i, j + 1 * nx, 1] = 1

            # RV
            if zi > z_sep_high:
                x_epi = -R_4 * np.sqrt(1 - (zi / Z_4) ** 2)
                x_endo = -R_3 * np.sqrt(1 - (zi / Z_3) ** 2)
                for j, xi in enumerate(np.linspace(x_epi + edge_space, x_endo - edge_space, nx)):
                    x[i, j + 2 * nx] = xi
                    z[i, j + 2 * nx] = zi
                    segments[i, j + 2 * nx, 2] = 1

        segments = segments.astype('int8')

        return x, z, segments

    def create_sax_track_points_lv(self):

        # Number of tracking points in radial and circumferential direction.
        nrad = 11
        ncir = 91

        # Check if it is BiV or LV.
        heart_type = check_heart_type_from_inputs(self.inputs)
        geometry = self.model.geometry

        if heart_type == 'Biventricle':
            geo_pars = geometry.parameters['geometry']
            edge_space = geo_pars['R_1'] * 0.05
            R_1 = geo_pars['R_1'] + edge_space
            R_2 = geo_pars['R_2'] - edge_space
            R_1sep = geo_pars['R_1sep'] + edge_space
            R_2sep = geo_pars['R_2sep'] - edge_space
        elif heart_type == 'LeftVentricle':
            # LV: compute inner and outer radius at midslice.
            C = geometry.parameters['focus_height']
            e1 = geometry.parameters['inner_eccentricity']
            e2 = geometry.parameters['outer_eccentricity']
            edge_space = C * np.sqrt(1 - e1 ** 2) / e1 * 0.05
            R_1 = C * np.sqrt(1 - e1 ** 2) / e1 + edge_space
            R_2 = C * np.sqrt(1 - e2 ** 2) / e2 - edge_space
            R_1sep = R_1
            R_2sep = R_2
        else:
            raise ValueError('Unknown heart_type "{}".'.format(heart_type))

        # Define landmarks for segments.
        aivg_angle = d2r(120)
        sept_angle = aivg_angle + d2r(60)
        pivg_angle = aivg_angle + d2r(120)

        # Landmarks at midslice (z=0).
        mid_point = [0, 0]
        pivg_point = [np.cos(pivg_angle), np.sin(pivg_angle)]
        aivg_point = [np.cos(aivg_angle), np.sin(aivg_angle)]

        # Collect landmarks xy-coordinates.
        landmarks = np.vstack((mid_point, pivg_point, aivg_point))

        # Create segment angles.
        seg_angles = np.array([aivg_angle, sept_angle, pivg_angle,
                               aivg_angle + np.pi, sept_angle + np.pi, pivg_angle + np.pi])

        segments = np.zeros((nrad, ncir, 6))

        # Create coordinates for tracking points.
        dphi = 2 * np.pi / ncir
        x = np.zeros((nrad, ncir))
        y = np.zeros((nrad, ncir))
        for i in range(nrad):
            for j in range(ncir):
                phi = 0.5 * dphi + dphi * j - np.pi  # range [-pi, pi].

                # Check if septum of free wall.
                if abs(phi) > 0.5 * np.pi:
                    # Septum
                    r_inner = R_1sep
                    r_outer = R_2sep
                else:
                    # Free wall
                    r_inner = R_1
                    r_outer = R_2

                radius_x = r_inner + (r_outer - r_inner) * i / (nrad - 1)
                radius_y = R_1 + (R_2 - R_1) * i / (nrad - 1)

                x[i, j] = radius_x * np.cos(phi)
                y[i, j] = radius_y * np.sin(phi)

                # Find the corresponding segment.
                # We want the aivg_angle to be smaller than phi.
                while phi < aivg_angle:
                    phi += 2 * np.pi

                seg_ij = sum(phi > seg_angles)
                segments[i, j, seg_ij - 1] = 1

        segments = segments.astype('int8')

        return x, y, segments, landmarks

    def create_sax_track_points_rv(self):
        # RV short axis track points.

        geo_pars = self.model.geometry.parameters['geometry']
        R_2 = geo_pars['R_2']
        R_3 = geo_pars['R_3']
        R_3y = geo_pars['R_3y']
        R_4 = geo_pars['R_4']

        # Number of tracking points in radial and circumferential direction.
        nrad = 5
        ncir = 41

        # Space to stay away from the boundary of the mesh.
        edge_space = (R_4 - R_3) * 0.05

        # Compute angle in xy-plane where RV endo and septal wall meet.
        phi = GeoFunc.compute_phi_sep_endo(geo_pars, z=0)

        x = np.zeros((nrad, ncir))
        y = np.zeros((nrad, ncir))
        segments = np.zeros((nrad, ncir, 4))

        for i, phii in enumerate(np.linspace(phi, 2 * np.pi - phi, ncir)):

            # Compute radii of track points.
            r_1 = np.linspace(R_3 + edge_space, R_4 - edge_space, nrad)
            r_2 = np.linspace(R_3y + edge_space, R_2 - edge_space, nrad)

            # Determine segment of track points.
            # Divide RV into 4 segments, similar to LV segments.
            if phii < 2.0944:  # 120 degress.
                segi = 0
            elif phii < np.pi:  # 180 degrees.
                segi = 1
            elif phii < 4.18879:  # 240 degrees.
                segi = 2
            else:
                segi = 3

            # Add tracking points.
            x[:, i] = r_1 * np.cos(phii)
            y[:, i] = r_2 * np.sin(phii)
            segments[:, i, segi] = 1

        segments = segments.astype('int8')

        return x, y, segments

    def create_curves(self):
        # FIXME the naming of the keys are a bit unfortunate,
        # FIXME we might want to not be consistent with the Matlab STE code.

        # LV short axis probe points.
        xc, yc, segments, landmarks = self.create_sax_track_points_lv()

        # Create curves similar to Matlab curves struct of STE algorithm.
        curves = {'xc': xc,
                  'yc': yc,
                  'xctrack': [],  # Tracked x-coordinate.
                  'yctrack': [],  # Tracked y-coordinate.
                  'Eccc': [],  # Cirumferential strain.
                  'Errc': [],  # Radial strain.
                  'Ecrc': [],  # Shear strain.
                  'segments': segments,
                  'landmarks': landmarks}

        # TODO implement LAX tracking points for LV only model

        # RV probe points.
        heart_type = check_heart_type_from_inputs(self.inputs)
        if heart_type == 'Biventricle':

            # BiV long axis probe points.
            xl, zl, segmentsl = self.create_lax_track_points_biv()
            curves.update({'xl': xl,  # l is for longitudinal, remain consistent with Matlab STE code.
                           'zl': zl,
                           'xltrack': [],
                           'zltrack': [],
                           'Elll': [],  # Longitudinal strain.
                           'Ettl': [],  # Transmural strain.
                           'segmentsl': segmentsl})

            # RV short axis probe points.
            xcrv, ycrv, segmentsrv = self.create_sax_track_points_rv()
            curves.update({'xrvc': xcrv,
                           'yrvc': ycrv,
                           'xrvctrack': [],
                           'yrvctrack': [],
                           'Ervccc': [],  # E_rv_cc_c (last c is for circumferential view, to remain consistent).
                           'Ervrrc': [],
                           'Ervcrc': [],
                           'segmentsrv': segmentsrv})

        return curves

    @staticmethod
    def default_variables():
        # Default set of variables.
        variables = ['u',
                    'ls',
                    'lc',
                    'active_stress',
                    'von_mises',
                    'fiber_stress',
                    'sign_active_stress',
                    'Ecc',
                    'Err',
                    'Ecr',
                    'Ell',
                    'curves_file'
                     ]
        return variables

    def initialize_functions(self, V, Q):
        # Add new variables here.
        self.functions['active_stress'] = Function(Q, name='active_stress')
        self.functions['ef'] = Function(V, name='fiber_vector')
        self.functions['Ecc'] = Function(Q, name='Ecc')
        self.functions['Ell'] = Function(Q, name='Ell')
        self.functions['Ett'] = Function(Q, name='Ett')
        self.functions['Err'] = Function(Q, name='Err')
        self.functions['Ecr'] = Function(Q, name='Ecr')
        self.functions['fiber_stress'] = Function(Q, name='fiber_stress')
        self.functions['passive_fiber_stress'] = Function(Q, name='passive_fiber_stress')
        self.functions['J'] = Function(Q, name='J')
        self.functions['lc_old'] = Function(Q, name='lc_old')
        self.functions['sign_active_stress'] = Function(Q, name='sign_active_stress')
        self.functions['sign_ls_min_lc'] = Function(Q, name='sign_ls_min_lc')
        self.functions['sign_ls_old_min_lc_old'] = Function(Q, name='sign_ls_old_min_lc_old')
        self.functions['von_mises'] = Function(Q, name='von_mises')

    def load_reduced_dataset(self, time, cycle):
        # Read CSV results file if it exists.
        csv_file = os.path.join(self.directory, 'results.csv')
        if os.path.exists(csv_file):
            # Load dataframe.
            full = Dataset(filename=csv_file)

            if time is None:
                if cycle is None:
                    # Select last complete cycle.
                    cycle = int(max(max(full['cycle']) - 1, min(full['cycle'])))
                reduced = full[full['cycle'] == int(cycle)].copy(deep=True)

            else:
                # Select specified time frame.
                mask = np.array(full['time'] >= time[0]) * np.array(full['time'] <= time[1])
                reduced = full[mask].copy(deep=True)

            return reduced
        else:
            raise ValueError(
                "File '{}' does not exist. "
                "Cannot determine which vector numbers to read and cannot determine active stress time!.".format(
                    csv_file))

    def main(self):
        # Check if data was loaded from CSV.
        if self.data is None:
            print_once('Reading entire HDF5...')
            # Load entire HDF5.
            with HDF5File(mpi_comm_world(), self.results_hdf5_name, 'r') as f:
                nsteps = f.attributes('displacement')['count']
            vector_numbers = list(range(nsteps))
        else:
            vector_numbers = list(self.data['vector_number'])

        print("accesing main plotting.py")

        # Create function spaces.
        V = self.model.u.ufl_function_space()
        Q = vector_space_to_scalar_space(V)

        # Check if fiber reorientation is enabled.
        ncycles_pre = self.inputs['model']['fiber_reorientation']['ncycles_pre']
        ncycles_reorient = self.inputs['model']['fiber_reorientation']['ncycles_reorient']
        current_cycle = list(self.data['cycle'])[0]
        fiber_reorientation = FiberReorientation.check_fiber_reorientation(current_cycle, ncycles_pre, ncycles_reorient)

        # Load previous displacement and fiber vectors (for ls_old).
        if vector_numbers[0] - 1 >= 0:
            # NOTE Not every action in load_model_state_from_hdf5 is necessary,
            # setting displacement and fiber vectors would be enough.
            load_model_state_from_hdf5(self.model, self.results_hdf5_name, vector_numbers[0]-1, fiber_reorientation)

        # Create XMDF files.
        self.open_xdmf_files()

        # Create FEniCS functions.
        self.initialize_functions(V, Q)

        # Open results hdf5 file.
        self.results_hdf5 = HDF5File(mpi_comm_world(), self.results_hdf5_name, 'r')

        # Loop over timesteps.
        t_old = -1
        for idx, vector in enumerate(vector_numbers):
            print_once('{0:.2f} %'.format(idx / len(vector_numbers) * 100))

            if self.data is None:
                # Read time information from hdf5 file.
                u_vector = 'displacement/vector_{}'.format(vector)
                t = self.results_hdf5.attributes(u_vector)['timestamp']
                t_act = -1
            else:
                # Get time information from csv file.
                t = self.data['time'].values.tolist()[idx]
                t_act = self.data['t_act'].values.tolist()[idx]

            # Check if fiber reorientation is enabled.
            current_cycle = list(self.data['cycle'])[idx]
            fiber_reorientation = FiberReorientation.check_fiber_reorientation(current_cycle, ncycles_pre,
                                                                               ncycles_reorient)
            # Update model state.
            self.update_model_state(t_act, vector, fiber_reorientation)

            # Save variables.
            self.save_timestep(t=t, t_old=t_old, V=V, Q=Q)

            if 'curves_file' in self.variables:
                self.update_curves()

            t_old = t*1

        # Write average fiber stress and strain to csv.
        if MPI.rank(mpi_comm_world()) == 0:
            self.write_csv()

        if MPI.rank(mpi_comm_world()) == 0:
            if 'curves_file' in self.variables:
                self.save_curves_file()

        # Close the hdf5 file.
        self.results_hdf5.close()

        # Close the XDMF files.
        self.close_xdmf_files()

    def open_xdmf_files(self):
        for vari in self.variables:
            filename = os.path.join(self.dir_out, '{}.xdmf'.format(vari))
            self.xdmf_files[vari] = XDMFFile(filename)

    @staticmethod
    def reset_values(function_to_reset, array_to_reset_from):
        """
        Helper function to reset DOLFIN quantities.
        """
        function_to_reset.vector()[:] = array_to_reset_from
        function_to_reset.vector().apply('')

    def save_active_stress(self, **kwargs):
        t = kwargs['t']
        Q = kwargs['Q']

        active_stress = self.functions['active_stress']
        project(self.active_stress(), Q, function=active_stress)

        self.xdmf_files['active_stress'].write(active_stress, t)

    def save_curves_file(self):
        curves = self.curves()
        curves_filename = os.path.join(self.dir_out, 'curves.hdf5')
        with HDF5File(mpi_comm_self(), curves_filename, 'w') as f:
            for key in curves.keys():
                v = curves[key]
                if type(v) is list:
                    array = np.dstack(v).astype(float)
                else:
                    array = v.astype(float)

                # FEniCS HDF5File can only store 1D arrays (as far as I know), so save it as a 1D array
                # and also save the original shape as an attribute.
                f.write(array.reshape(-1), key)
                f.attributes(key)['shape'] = np.array(array.shape).astype(float)

    def save_ef(self, **kwargs):
        t = kwargs['t']
        V = kwargs['V']
        ef = self.model.geometry.fiber_vectors()[0].to_function(None)
        self.xdmf_files['ef'].write(project(ef, V), t)

        print("accesing save_ef in plotting.py")

    def save_Ecc(self, **kwargs):
        t = kwargs['t']
        Q = kwargs['Q']

        E_cyl = self.cylindrical_green_lagrange_strain()
        Ecc = self.functions['Ecc']
        project(E_cyl[1, 1], Q, function=Ecc)
        self.xdmf_files['Ecc'].write(Ecc, t)

    def save_Ell(self, **kwargs):
        t = kwargs['t']
        Q = kwargs['Q']

        E_car = self.cardiac_green_lagrange_strain()
        Ell = self.functions['Ell']
        project(E_car[1, 1], Q, function=Ell)
        self.xdmf_files['Ell'].write(Ell, t)

    def save_Ett(self, **kwargs):
        t = kwargs['t']
        Q = kwargs['Q']

        E_car = self.cardiac_green_lagrange_strain()
        Ett = self.functions['Ett']
        project(E_car[2, 2], Q, function=Ett)
        self.xdmf_files['Ett'].write(Ett, t)

    def save_Err(self, **kwargs):
        t = kwargs['t']
        Q = kwargs['Q']

        E_cyl = self.cylindrical_green_lagrange_strain()
        Err = self.functions['Err']
        project(E_cyl[0, 0], Q, function=Err)
        self.xdmf_files['Err'].write(Err, t)

    def save_Ecr(self, **kwargs):
        t = kwargs['t']
        Q = kwargs['Q']

        E_cyl = self.cylindrical_green_lagrange_strain()
        Ecr = self.functions['Ecr']
        project(E_cyl[1, 0], Q, function=Ecr)
        self.xdmf_files['Ecr'].write(Ecr, t)

    def save_fiber_stress(self, **kwargs):
        t = kwargs['t']
        Q = kwargs['Q']

        # Cauchy stress
        C = self.compute_cauchy_stress()

        # Compute stress vector along ef.
        ef = self.model.active_stress.fiber_vectors[0]
        fs = dot(ef, C*ef)

        # Project on function space.
        fiber_stress = self.functions['fiber_stress']
        project(fs, Q, function=fiber_stress)

        self.xdmf_files['fiber_stress'].write(fiber_stress, t)

        # Save the average fiber stress for this timestep.
        avg_stress = global_function_average(fiber_stress)
        self.avg_fiber_stress[t] = avg_stress

    def save_J(self, **kwargs):
        t = kwargs['t']
        Q = kwargs['Q']

        # Calculate J from displacement.
        J = self.functions['J']
        project(det(deformation_gradient(self.model.u)), Q, function=J)

        self.xdmf_files['J'].write(J, t)

    def save_lc(self, **kwargs):
        t_old = kwargs['t_old']
        Q = kwargs['Q']
        if t_old >= 0:
            lc_old = self.functions['lc_old']
            project(self.model.active_stress.lc_old, Q, function=lc_old)
            self.xdmf_files['lc'].write(lc_old, t_old)

    def save_ls(self, **kwargs):
        t_old = kwargs['t_old']
        if t_old >= 0:
            self.xdmf_files['ls'].write(self.model.active_stress.ls_old, t_old)

            # Save the average fiber strain for this timestep.
            avg_ls = global_function_average(self.model.active_stress.ls_old)
            self.avg_fiber_strain[t_old] = avg_ls/self.model.active_stress.parameters['ls0']

    def save_passive_fiber_stress(self, **kwargs):
        t = kwargs['t']
        Q = kwargs['Q']

        # Cauchy stress
        C = self.passive_fiber_stress()

        # Compute stress vector along ef.
        ef = self.model.active_stress.fiber_vectors[0]
        fs = dot(ef, C*ef)

        # Project on function space.
        passive_fiber_stress = self.functions['passive_fiber_stress']
        project(fs, Q, function=passive_fiber_stress)

        self.xdmf_files['passive_fiber_stress'].write(passive_fiber_stress, t)

    def save_sign_active_stress(self, **kwargs):
        t = kwargs['t']
        Q = kwargs['Q']

        sign_active_stress = self.functions['sign_active_stress']
        project(sign(self.active_stress() + 1e-8), Q, function=sign_active_stress)

        self.xdmf_files['sign_active_stress'].write(sign_active_stress, t)

    def save_sign_ls_min_lc(self, **kwargs):
        t = kwargs['t']
        Q = kwargs['Q']

        sign_ls_min_lc = self.functions['sign_ls_min_lc']
        project(sign(self.model.active_stress.ls - self.model.active_stress.lc + 1e-8), Q, function=sign_ls_min_lc)

        self.xdmf_files['sign_ls_min_lc'].write(sign_ls_min_lc, t)

    def save_sign_ls_old_min_lc_old(self, **kwargs):
        t = kwargs['t']
        Q = kwargs['Q']

        sign_ls_old_min_lc_old = self.functions['sign_ls_old_min_lc_old']
        ls_old_min_lc_old = self.model.active_stress.ls_old - self.model.active_stress.lc_old
        project(sign(ls_old_min_lc_old + 1e-8), Q, function=sign_ls_old_min_lc_old)

        self.xdmf_files['sign_ls_old_min_lc_old'].write(sign_ls_old_min_lc_old, t)

    def save_timestep(self, **kwargs):
        vari = self.variables

        # Add new variables here.

        if 'active_stress' in vari:
            self.save_active_stress(**kwargs)

        if 'ef' in vari:
            self.save_ef(**kwargs)

        if 'Ecc' in vari:
            self.save_Ecc(**kwargs)

        if 'Ell' in vari:
            self.save_Ell(**kwargs)

        if 'Ett' in vari:
            self.save_Ett(**kwargs)

        if 'Err' in vari:
            self.save_Err(**kwargs)

        if 'Ecr' in vari:
            self.save_Ecr(**kwargs)

        if 'fiber_stress' in vari:
            self.save_fiber_stress(**kwargs)

        if 'J' in vari:
            self.save_J(**kwargs)

        if 'lc' in vari:
            self.save_lc(**kwargs)

        if 'ls' in vari:
            self.save_ls(**kwargs)

        if 'passive_fiber_stress' in vari:
            self.save_passive_fiber_stress(**kwargs)

        if 'sign_active_stress' in vari:
            self.save_sign_active_stress(**kwargs)

        if 'sign_ls_min_lc' in vari:
            self.save_sign_ls_min_lc(**kwargs)

        if 'sign_ls_old_min_lc_old' in vari:
            self.save_sign_ls_old_min_lc_old(**kwargs)

        if 'von_mises' in vari:
            self.save_von_mises(**kwargs)

        if 'u' in vari:
            self.save_u(**kwargs)

    def save_von_mises(self, **kwargs):
        t = kwargs['t']
        Q = kwargs['Q']

        # Cauchy stress
        C = self.compute_cauchy_stress()

        # Compute Von Mises stress.
        VM = (0.5*((C[0,0] - C[1,1])**2 + (C[1,1] - C[2,2])**2 + (C[2,2] - C[0,0])**2)
                   + 3*((C[0,1])**2 + (C[1,2])**2 + (C[2,0])**2))**0.5

        # Project on function space.
        von_mises = self.functions['von_mises']
        project(VM, Q, function=von_mises)

        self.xdmf_files['von_mises'].write(von_mises, t)

    def save_u(self, **kwargs):
        t = kwargs['t']
        self.xdmf_files['u'].write(self.model.u, t)

    def update_curves(self):
        curves = self.curves()
        xc = curves['xc']
        yc = curves['yc']

        u = self.model.u
        Ecc = self.functions['Ecc']
        Err = self.functions['Err']
        Ecr = self.functions['Ecr']

        # Iterate over tracking points.
        xctrack = np.zeros_like(xc)
        yctrack = np.zeros_like(xc)
        Eccc = np.zeros_like(xc)
        Errc = np.zeros_like(xc)
        Ecrc = np.zeros_like(xc)
        for i in range(xc.shape[0]):
            for j in range(xc.shape[1]):
                x_ij = xc[i, j]
                y_ij = yc[i, j]
                z_ij = 0.0  # Midslice SAX (at equator).
                u_ij = u(x_ij, y_ij, z_ij)
                xctrack[i, j] = x_ij + u_ij[0]
                yctrack[i, j] = y_ij + u_ij[1]  # Note: ignore out of plane motion.
                Eccc[i, j] = Ecc(x_ij, y_ij, z_ij)
                Errc[i, j] = Err(x_ij, y_ij, z_ij)
                Ecrc[i, j] = Ecr(x_ij, y_ij, z_ij)

        # Save to curves dict.
        curves['xctrack'].append(xctrack)
        curves['yctrack'].append(yctrack)
        curves['Eccc'].append(Eccc)
        curves['Errc'].append(Errc)
        curves['Ecrc'].append(Ecrc)

        # Update RV short axis points if exists.
        if 'xrvc' in curves.keys():
            xrvc = curves['xrvc']
            yrvc = curves['yrvc']

            # Iterate over tracking points.
            xrvctrack = np.zeros_like(xrvc)
            yrvctrack = np.zeros_like(xrvc)
            Ervccc = np.zeros_like(xrvc)
            Ervrrc = np.zeros_like(xrvc)
            Ervcrc = np.zeros_like(xrvc)
            for i in range(xrvc.shape[0]):
                for j in range(xrvc.shape[1]):
                    x_ij = xrvc[i, j]
                    y_ij = yrvc[i, j]
                    z_ij = 0.0  # Midslice SAX (at equator).
                    u_ij = u(x_ij, y_ij, z_ij)
                    xrvctrack[i, j] = x_ij + u_ij[0]
                    yrvctrack[i, j] = y_ij + u_ij[1]  # Note: ignore out of plane motion.
                    Ervccc[i, j] = Ecc(x_ij, y_ij, z_ij)
                    Ervrrc[i, j] = Err(x_ij, y_ij, z_ij)
                    Ervcrc[i, j] = Ecr(x_ij, y_ij, z_ij)

            # Save to curves dict.
            curves['xrvctrack'].append(xrvctrack)
            curves['yrvctrack'].append(yrvctrack)
            curves['Ervccc'].append(Ervccc)
            curves['Ervrrc'].append(Ervrrc)
            curves['Ervcrc'].append(Ervcrc)

        # Update BiV long axis points if exists.
        if 'xl' in curves.keys():
            xl = curves['xl']
            zl = curves['zl']

            Ell = self.functions['Ell']  # Longitudinal strain.
            Ett = self.functions['Ett']  # Transmural strain

            # Iterate over tracking points.
            xltrack = np.zeros_like(xl)
            zltrack = np.zeros_like(xl)
            Elll = np.zeros_like(xl)
            Ettl = np.zeros_like(xl)

            # Mask for checking if a point belongs to any segment. (due to rectangular grid, there may be points falling
            # outside the domain, which is reflected by the fact that they do not belong to a segment).
            mask = np.any(curves['segmentsl'], axis=2)

            # Loop over points.
            for i in range(xl.shape[0]):
                for j in range(xl.shape[1]):
                    # Check if point belongs to any segment.
                    if mask[i, j]:
                        x_ij = xl[i, j]
                        y_ij = 0.0  # Midslice LAX
                        z_ij = zl[i, j]
                        u_ij = u(x_ij, y_ij, z_ij)
                        xltrack[i, j] = x_ij + u_ij[0]
                        zltrack[i, j] = z_ij + u_ij[2]
                        Elll[i, j] = Ell(x_ij, y_ij, z_ij)
                        Ettl[i, j] = Ett(x_ij, y_ij, z_ij)

            # Save to curves dict.
            curves['xltrack'].append(xltrack)
            curves['zltrack'].append(zltrack)
            curves['Elll'].append(Elll)
            curves['Ettl'].append(Ettl)

    def update_model_state(self, t_act, vector, fiber_reorientation):

        # Contractile element length (lc_old).
        if isinstance(self.model.active_stress, ArtsKerckhoffsActiveStress):
            # Load lc_old.
            lc_vector = 'contractile_element/vector_{}'.format(vector)
            self.results_hdf5.read(self.model.active_stress.lc_old, lc_vector)

        # Sarcomere length (ls_old).
        self.model.active_stress.ls_old = self.model.active_stress.ls

        # Activation time.
        if self.model.active_stress is not None:
            self.model.active_stress.activation_time = float(t_act + self.model.active_stress.parameters['tdep'])

        # Fiber vectors.
        if fiber_reorientation:
            # Fiber vectors are saved in results file.
            self.model.geometry.load_fiber_field(openfile=self.results_hdf5, vector_number=vector)

        # Displacement
        self.model.u_old = self.model.u.vector().array()

        u_vector = 'displacement/vector_{}'.format(vector)
        self.results_hdf5.read(self.model.u, u_vector)

    def write_csv(self):
        with open(os.path.join(self.dir_out, 'myofiber_stress_strain.csv'), 'w') as myfile:
            wr = csv.writer(myfile)
            for t in self.avg_fiber_strain.keys():
                line = [t, self.avg_fiber_stress[t], self.avg_fiber_strain[t]]
                wr.writerow(line)


def d2r(degree):
    """ Converts degrees to radians. """
    return degree/180*np.pi


