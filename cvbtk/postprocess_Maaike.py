import sys
sys.path.append(r'C:\Users\Maaike\Documents\Master\Graduation_project\git_graduation_project\cvbtk')
sys.path.append(r'C:\Users\Maaike\Documents\Master\Graduation_project\git_graduation_project\postprocessing')
import os
from dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Specify the results.csv file (which contains the hemodynamic data) directly:
csv = r'C:\Users\Maaike\Documents\Master\Graduation_project\Results_Tim\05-03_2cycles_default\results.csv'

# load results
results = Dataset(filename= csv)

# The last cycle is selected for postprocessing
# Select other cycle if you want to analyse a different cycle
cycle = max(results['cycle']) - 1
#cycle =3

# Data from the selected cycle
data_cycle = results[results['cycle']==cycle]
key = data_cycle.keys()

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

def hemodynamic_summary(data_cycle):
     time = data_cycle['time'].values  # ms
     plv = kPa_to_mmHg(data_cycle['plv'].values)
     part = kPa_to_mmHg(data_cycle['part'].values)
     pven = kPa_to_mmHg(data_cycle['pven'].values)
     qao = data_cycle['qao'].values*60/1000  # ml/s -> l/min
     vlv = data_cycle['vlv'].values  # ml
     
     
     HR = round(60000/(max(time)-min(time)))
     EDV =  max(vlv)
     SV = np.mean(qao)/HR * 1000
     CO = np.mean(qao)
     
     hemo = {    
            'HR': HR,
            'EDV': EDV,
            'ESV': min(vlv),
            'CO': CO,
            'qao': np.mean(qao),
            'SV': SV,
            'EF': SV/EDV*100,
            'MAP': np.mean(part),
            'SAP': max(part),
            'DAP': min(part),
            'PP': max(part) - min(part),
            'plv_max': max(plv),
            'W': - np.trapz(mmHg_to_kPa(plv), vlv/1000),
            'dpdt_max': max(np.diff(plv)/np.diff(time/1000)),
            }
     return hemo

def print_hemodynamic_summary(hemo,cycle):    
    print('\nSYSTEMIC CIRCULATION:')
    print('Results from cycle {}'.format(cycle))
    print(('HR: {:11.0f} bpm\n' +
           'EDV: {:10.2f} ml\n' +
           'ESV: {:10.2f} ml\n' +
          'SV: {:11.2f} ml\n' +
          'EF: {:11.2f} %\n' +
          'qao: {:10.2f} l/min \n' +
          'CO: {:11.2f} l/min\n' +
          'MAP: {:10.2f} mmHg\n' +
          'SAP: {:10.2f} mmHg\n' +
          'DAP: {:10.2f} mmHg\n' +
          'PP: {:11.2f} mmHg\n' +
          'plv_max: {:6.2f} mmHg\n' +
          'dp/dt_max: {:.0f} mmHg/s\n' +
          'W: {:12.2f} J').format(
                  hemo['HR'], hemo['EDV'], hemo['ESV'], hemo['SV'], hemo['EF'],
                  hemo['qao'], hemo['CO'], hemo['MAP'], 
                  hemo['SAP'], hemo['DAP'], hemo['PP'],
                  hemo['plv_max'], hemo['dpdt_max'], hemo['W']))
    if 'CVP' in hemo.keys():
        print('CVP: {:10.2f} mmHg'.format(hemo['CVP']))  
    print('\n')

def plot_results(results, dir_out='.', cycle=None):
            # LV only.
    simulation_plot = HemodynamicsPlot(results)
        
    simulation_plot.plot(cycle=cycle) #, cycle=NUM_CYCLES)
    simulation_plot.save(os.path.join(dir_out, 'hemodynamics.png'))
       
    #simulation_plot.plot_function()
    plt.savefig(os.path.join(dir_out, 'lv_function.png'), dpi=300)
    

hemo_sum = hemodynamic_summary(data_cycle)
print_hemodynamic_summary(hemo_sum,cycle)
plot_results(results, dir_out=os.path.dirname(csv), cycle=cycle)