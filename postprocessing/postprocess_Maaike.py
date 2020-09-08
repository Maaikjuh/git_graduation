import sys
sys.path.append(r'C:\Users\Maaike\Documents\Master\Graduation_project\git_graduation_project\cvbtk')
sys.path.append(r'C:\Users\Maaike\Documents\Master\Graduation_project\git_graduation_project\postprocessing')
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
# import plotly.graph_objects as go
import pandas as pd
from dataset import Dataset

def dict_var_names(dataset = None, model = None):
    par = {}
    
    if model == 'beatit':
        par['time'] = 'Time [ms]'
        par['plv'] = ' p_lv [kPa]'
        par['part'] = ' p_art [kPa]'
        par['pven'] = ' p_ven [kPa]'
        par['qao'] = ' Qao [mm3/ms]'
        par['qper'] = ' Qper [mm3/ms]'
        par['qmv'] = ' Qmv [mm3/ms]'
        par['vlv'] = ' V_lv [ml]'
        par['vart'] = ' Vart [ml]'
        par['vven'] = ' Vven [ml]'
        par['phase'] = ' Phase [-]'
        par['cycle'] = ' Cycle [-]'
        par['t_cycle'] = 'Time [ms]'  
        
        if not isinstance(dataset, type(None)) and ' V_lv [ml]' not in dataset.keys():
            dataset[' Phase [-]'] = dataset[' Phase [-]'] + 1
            
            dataset[' V_lv [mm3]'] = dataset[' V_lv [mm3]'] / 1000
            dataset[' Vart [mm3]'] = dataset[' Vart [mm3]'] / 1000
            dataset[' Vven [mm3]'] = dataset[' Vven [mm3]'] / 1000
            dataset[' Vtot [mm3]'] = dataset[' Vtot [mm3]'] / 1000
            
            dataset.rename(columns = {' V_lv [mm3]':  ' V_lv [ml]',
                                    ' Vart [mm3]':  ' Vart [ml]',
                                    ' Vven [mm3]':  ' Vven [ml]',
                                    ' Vtot [mm3]':  ' Vtot [ml]'}, inplace= True)
            
    else:
        par['time'] = 'time'
        par['plv'] = 'plv'
        par['part'] = 'part'
        par['pven'] = 'pven'
        par['qao'] = 'qao'
        par['qper'] = 'qper'
        par['qmv'] = 'qmv'
        par['vlv'] = 'vlv'
        par['vart'] = 'vart'
        par['vven'] = 'vven'
        par['phase'] = 'phase'
        par['cycle'] = 'cycle'
        par['t_cycle'] = 't_cycle'
    
    return dataset, par
        
def load_reduced_dataset(filename, cycle=None, model = None):
    """
    Load the given CSV file and reduce it to the final cycle (default).
    """
    full = Dataset(filename=filename)    
    dataset, var = dict_var_names( model = model)
    
    if model == 'beatit' and cycle is not None:
        cycle = cycle - 1
            
    if cycle is None:
        cycle = int(max(full[var['cycle']]) - 1)
           
    reduced = full[full[var['cycle']] == cycle].copy(deep=True)
    return reduced, cycle

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

class HemodynamicsPlot(object):
    """
    High-level interface between :class:`~cvbtk.Dataset` and :mod:`matplotlib`
    that plots the pressure vs. time, volume vs. time, flowrate vs. time, and
    pressure vs. volume curves.

    Args:
        dataset: Dataset to create the figure from.
    """
    def __init__(self, dataset, title, model = None):
        # Store the dataset.
           
        self._df, self.var = dict_var_names(dataset = dataset, model = model)
        self.model = model    

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
        self._fig.suptitle(title)

        # Remove the right and top spines.
        [ax.spines['top'].set_visible(False) for _, ax in self._ax.items()]
        [ax.spines['right'].set_visible(False) for _, ax in self._ax.items()]
        
    def plot(self,  label = '' ,cycle=None, legend=True):
        """
        Plot the defined hemodynamic relations for output.

        Args:
            cycle (optional): Filter and plot results for this specific cycle.
            legend (optional): Enables (default) or disables the legend.
        """
        # The cycle keyword argument can filter the results to a specific cycle.
        var = self.var
        
        # if cycle:
        #     df = self._df[self._df[var['cycle']] == int(cycle)]
        #     time = df[var['time']] - min(df[var['time']])
        # else:
        df = self._df
        time = df[var['time']] - min(df[var['time']])
        
        if var['plv'] in df.keys():
            #lv

            # Make the pressure-time plot.
            self._ax['pt'].plot(time, kPa_to_mmHg(df[var['plv']]), color ='#1f77b4', label='Cavity')
            self._ax['pt'].plot(time, kPa_to_mmHg(df[var['pven']]),'r', label='Venous')
            self._ax['pt'].plot(time, kPa_to_mmHg(df[var['part']]),'g', label='Arterial')
    
            # Make the volume-time plot.
            self._ax['vt'].plot(time, df[var['vlv']], color ='#1f77b4', label='Cavity')
    #        self._ax['vt'].plot(time, (df['vlv']+df['vart']+df['vven']), 'b', label='Vtot')
    #        self._ax['vt'].plot(df['time'], df['vven'], label='Venous')
    #        self._ax['vt'].plot(df['time'], df['vart'], label='Arterial')
    
            # Make the flowrate-time plot.
            self._ax['qt'].plot(time, df[var['qmv']], color ='#1f77b4', label='Mitral')
            self._ax['qt'].plot(time, df[var['qao']], 'r', label='Aortic')
            self._ax['qt'].plot(time, df[var['qper']], 'g', label='Peripheral')
                
            # Mark the beginning of each phase            
            for i in range(1,5):
                index = (df[var['phase']] == i)[::-1].idxmax()
                
                if self.model == 'beatit' and i == 1:
                    index = (df[var['phase']] == 2).idxmax()
                # index = (df[var['phase']] == i).idxmax()
                phase_time = df[var['time']][index]- min(df[var['time']])
                self._ax['pt'].axvline(x = phase_time, linestyle = '--', color = 'k')
                self._ax['vt'].axvline(x = phase_time, linestyle = '--', color = 'k')
                self._ax['qt'].axvline(x = phase_time, linestyle = '--', color = 'k')
            
            # Add label to each phase
            phase2 = df[var['time']][(df[var['phase']] == 2).idxmax()]- min(df[var['time']])
            phase3 = df[var['time']][(df[var['phase']] == 3).idxmax()]- min(df[var['time']])
            phase4 = df[var['time']][(df[var['phase']] == 4).idxmax()]- min(df[var['time']])
            phase4_end = df[var['time']][(df[var['phase']] == 4)[::-1].idxmax()]- min(df[var['time']])
            
            self._ax['pt'].text(phase2/2,kPa_to_mmHg(max(df[var['plv']]))+2,'d',fontsize=13,horizontalalignment='center')
            self._ax['pt'].text((phase2+phase3)/2,kPa_to_mmHg(max(df[var['plv']]))+2,'ic',fontsize=13,horizontalalignment='center')
            self._ax['pt'].text((phase3+phase4)/2,kPa_to_mmHg(max(df[var['plv']]))+2,'e',fontsize=13,horizontalalignment='center')
            self._ax['pt'].text((phase4+phase4_end)/2,kPa_to_mmHg(max(df[var['plv']]))+2,'ir',fontsize=13,horizontalalignment='center')
    
            # Make the pressure-volume plot.
            # Each cycle (if multiple) will get its own color.
            for c in df[var['cycle']].unique():
                _df = df[df[var['cycle']] == int(c)]
                self._ax['pv'].plot(_df[var['vlv']], kPa_to_mmHg(_df[var['plv']]), color ='#1f77b4',label=str(c) + label)
    
            # Add the legends for the three individual plots, if needed to.
            if legend:
                # self._ax['pt'].legend(loc=2, fontsize=7) #, title='Pressure')
                self._ax['pt'].legend(fontsize=8, bbox_to_anchor=(1.0, 1), loc='upper left') #, title='Pressure')
                self._ax['pt'].get_legend().get_title().set_fontsize('7')
                # self._ax['vt'].legend(loc=2, fontsize=7) #, title='Volume')
                self._ax['vt'].legend(fontsize=8, bbox_to_anchor=(1.0, 1), loc='upper left') #, title='Volume')
                self._ax['vt'].get_legend().get_title().set_fontsize('7')
                # self._ax['qt'].legend(loc=2, fontsize=7) #, title='Flowrate')
                self._ax['qt'].legend(fontsize=8, bbox_to_anchor=(1.0, 1), loc='upper left') #, title='Flowrate')
                self._ax['qt'].get_legend().get_title().set_fontsize('7')
    
            # The pressure-volume loop always has a legend if multiple are plotted.
            if not cycle and len(df[var['cycle']].unique()) > 1:
                self._ax['pv'].legend(loc=2, fontsize=7, title='Cycle')
                self._ax['pv'].get_legend().get_title().set_fontsize('7')
                
                
        elif 'pcav_s' in df.keys():
            #BiV
            
            # Make the pressure-time plot.
            self._ax['pt'].plot(time, (df['pcav_s']),color ='#1f77b4', label='Cavity')
            self._ax['pt'].plot(time, (df['pven_s']),'r', label='Venous')
            self._ax['pt'].plot(time, (df['part_s']),'g', label='Arterial')
    
            # Make the volume-time plot.
            self._ax['vt'].plot(time, df['vcav_s'], color ='#1f77b4', label='Cavity')
    #        self._ax['vt'].plot(time, (df['vlv']+df['vart']+df['vven']), 'b', label='Vtot')
    #        self._ax['vt'].plot(df['time'], df['vven'], label='Venous')
    #        self._ax['vt'].plot(df['time'], df['vart'], label='Arterial')
    
            # Make the flowrate-time plot.
            self._ax['qt'].plot(time, df['qven_s'], color ='#1f77b4', label='Mitral')
            self._ax['qt'].plot(time, df['qart_s'], 'r', label='Aortic')
            self._ax['qt'].plot(time, df['qper_s'], 'g', label='Peripheral')
                
            # Mark the beginning of each phase            
            for i in range(2,5):
                index = (df['phase_s'] == i).idxmax()
                phase_time = df['time'][index]- min(df['time'])
                self._ax['pt'].plot([phase_time, phase_time], [kPa_to_mmHg(min(df['pcav_s'])), kPa_to_mmHg(max(df['pcav_s']))],'--k')
                self._ax['vt'].plot([phase_time, phase_time], [min(df['vcav_s']), max(df['vcav_s'])],'--k')
                self._ax['qt'].plot([phase_time, phase_time], [min(df['qart_s']), max(df['qart_s'])],'--k')
            
            # Add label to each phase
            phase2 = df['time'][(df['phase_s'] == 2).idxmax()]- min(df['time'])
            phase3 = df['time'][(df['phase_s'] == 3).idxmax()]- min(df['time'])
            phase4 = df['time'][(df['phase_s'] == 4).idxmax()]- min(df['time'])
            
            self._ax['pt'].text(phase2/2,kPa_to_mmHg(max(df['pcav_s']))+2,'d',fontsize=13,horizontalalignment='center')
            self._ax['pt'].text((phase2+phase3)/2,kPa_to_mmHg(max(df['pcav_s']))+2,'ic',fontsize=13,horizontalalignment='center')
            self._ax['pt'].text((phase3+phase4)/2,kPa_to_mmHg(max(df['pcav_s']))+2,'e',fontsize=13,horizontalalignment='center')
            self._ax['pt'].text((phase4+(max(df['time'])-min(df['time'])))/2,kPa_to_mmHg(max(df['pcav_s']))+2,'ir',fontsize=13,horizontalalignment='center')
    
            # Make the pressure-volume plot.
            # Each cycle (if multiple) will get its own color.
            for c in df['cycle'].unique():
                _df = df[df['cycle'] == int(c)]
                self._ax['pv'].plot(_df['vcav_s'], kPa_to_mmHg(_df['pcav_s']), color ='#1f77b4',label=str(c))
    
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
            
            
    def compare_against(self, label_ref, label_vars, dataset,cycle=None, model = None, linestyle = '--'):
        """
        Draw additional curves on the existing figure for visual comparison.

        Args:
            dataset: Additional dataset to draw curves from.
            *args: Arbitrary positional arguments for plotting parameters.
            **kwargs: Arbitrary keyword arguments for plotting parameters.
        """
        # TODO Check that a plot has been already created.
#        cycle = max(dataset['cycle']) - 1
#        if cycle:
#            dataset = dataset[dataset['cycle']==cycle]
#            df = dataset[dataset['cycle'] == int(cycle)]
#
#        else:
#            df = dataset
        df = self._df
        var = self.var
        dataset, var_comp = dict_var_names(dataset= dataset, model = model)
            
        if var_comp['plv'] in dataset.keys():
        
            center_phase = 3
            index = (dataset[var_comp['phase']] == center_phase).idxmax()
            
            if var['plv'] in df.keys():
                indexdf = (df[var['phase']] == center_phase).idxmax()
            elif 'pcav_s' in df.keys():
                indexdf = (df[var['phase_s']] == center_phase).idxmax()
            
            phase_time = dataset[var_comp['time']][index]- min(dataset[var_comp['time']])
            phase_time_df = df[var['time']][indexdf]- min(df[var['time']])
            
            offset = phase_time_df-phase_time
            
            time = dataset[var_comp['time']]-min(dataset[var_comp['time']])+offset
            
            print('\ndata centered around phase {} at {}ms with offset {}ms'.format(center_phase,phase_time_df,offset))
            # Make the pressure-time plot.
            self._ax['pt'].plot(time, kPa_to_mmHg(dataset[var_comp['plv']]), linestyle, color ='#1f77b4')
            self._ax['pt'].plot(time, kPa_to_mmHg(dataset[var_comp['pven']]), linestyle + 'r')
            self._ax['pt'].plot(time, kPa_to_mmHg(dataset[var_comp['part']]), linestyle + 'g')
    
            # Make the volume-time plot.
            self._ax['vt'].plot(time, dataset[var_comp['vlv']], linestyle, color ='#1f77b4')
            # self._ax['vt'].plot(dataset['time'], dataset['vven'], *args, **kwargs)
            # self._ax['vt'].plot(dataset['time'], dataset['vart'], *args, **kwargs)
    
            # Make the flowrate-time plot.
            self._ax['qt'].plot(time, dataset[var_comp['qmv']], linestyle, color ='#1f77b4')
            self._ax['qt'].plot(time, dataset[var_comp['qao']], linestyle + 'r')
            self._ax['qt'].plot(time, dataset[var_comp['qper']], linestyle + 'g')
            
            for i in range(2,5):
                index = (dataset[var_comp['phase']] == i).idxmax()
                phase_time = dataset[var_comp['time']][index]- min(dataset[var_comp['time']])+offset
    
                self._ax['pt'].axvline(x = phase_time, linestyle = linestyle, color = 'y')
                self._ax['vt'].axvline(x = phase_time, linestyle = linestyle, color = 'y')
                self._ax['qt'].axvline(x = phase_time, linestyle = linestyle, color = 'y')

    
            # Make the pressure-volume plot.
            # Each cycle (if multiple) will get its own color.
            for c in dataset[var_comp['cycle']].unique():
                _df = dataset[dataset[var_comp['cycle']] == int(c)]
                self._ax['pv'].plot(_df[var_comp['vlv']], kPa_to_mmHg(_df[var_comp['plv']]), linestyle, color ='#1f77b4')
                
            if 'pcav_s' in df.keys():
                self._ax['pv'].legend(['BiV','Lv'])
            else:
                self._ax['pv'].legend(label_ref + label_vars)
                
        if 'pcav_s' in dataset.keys():
        
            center_phase = 3
            index = (dataset['phase_s'] == center_phase).idxmax()
            
            if 'plv' in df.keys():
                indexdf = (df['phase'] == center_phase).idxmax()
            elif 'pcav_s' in df.keys():
                indexdf = (df['phase_s'] == center_phase).idxmax()
            
            phase_time = dataset['time'][index]- min(dataset['time'])
            phase_time_df = df['time'][indexdf]- min(df['time'])
            
            offset = phase_time_df-phase_time
            
            time = dataset['time']-min(dataset['time'])+offset
            
            print('\n infarct data centered around phase {} at {}ms with offset {}ms'.format(center_phase,phase_time_df,offset))
            # Make the pressure-time plot.
            self._ax['pt'].plot(time, (dataset['pcav_s']), '--', color ='#1f77b4')
            self._ax['pt'].plot(time, (dataset['pven_s']), '--r')
            self._ax['pt'].plot(time, (dataset['part_s']), '--g')
    
            # Make the volume-time plot.
            self._ax['vt'].plot(time, dataset['vcav_s'], '--', color ='#1f77b4')
            # self._ax['vt'].plot(dataset['time'], dataset['vven'], *args, **kwargs)
            # self._ax['vt'].plot(dataset['time'], dataset['vart'], *args, **kwargs)
    
            # Make the flowrate-time plot.
            self._ax['qt'].plot(time, dataset['qven_s'], '--', color ='#1f77b4')
            self._ax['qt'].plot(time, dataset['qart_s'], '--r')
            self._ax['qt'].plot(time, dataset['qper_s'], '--g')
            
            for i in range(2,5):
                index = (dataset['phase_s'] == i).idxmax()
                phase_time = dataset['time'][index]- min(dataset['time'])+offset
    
                self._ax['pt'].plot([phase_time, phase_time], [min((dataset['pcav_s'])), max((dataset['pcav_s']))],'--y')
                self._ax['vt'].plot([phase_time, phase_time], [min(dataset['vcav_s']), max(dataset['vcav_s'])],'--y')
                self._ax['qt'].plot([phase_time, phase_time], [min(dataset['qart_s']), max(dataset['qart_s'])],'--y')
    
            # Make the pressure-volume plot.
            # Each cycle (if multiple) will get its own color.
            for c in dataset['cycle'].unique():
                _df = dataset[dataset['cycle'] == int(c)]
                self._ax['pv'].plot(_df['vcav_s'], kPa_to_mmHg(_df['pcav_s']), '--', color ='#1f77b4')
            
#             self._ax['pv'].text(max(dataset['vcav_s']),max(dataset['pcav_s']))
                
            if 'pcav_s' in df.keys():
                self._ax['pv'].legend(['BiV ischemic','BiV reference'])
            else:
                self._ax['pv'].legend(['Lv','BiV'])

            
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
            
class Hemodynamics_all_cycles(object):
    """
    High-level interface between :class:`~cvbtk.Dataset` and :mod:`matplotlib`
    that plots the pressure vs. time, volume vs. time, flowrate vs. time, and
    pressure vs. volume curves.

    Args:
        dataset: Dataset to create the figure from.
    """
    def __init__(self, dataset, model = None, fig=None):
        # Store the dataset.
        self._df, self.var = dict_var_names(dataset, model = model)

        if fig is None:
            # Create new figure.
            self._fig = plt.figure()
        else:
            self._fig = fig

        # Make fig current.
        plt.figure(self._fig.number)
        
        # Create a set of empty axes for plotting.
        gs = GridSpec(1, 3)
        _sv = plt.subplot(gs[0, 0])
        _pv = plt.subplot(gs[0, 1])
        _w = plt.subplot(gs[0, 2])
        self._ax = {'sv': _sv, 'pv': _pv, 'w': _w}

        # Remove vertical spacing from the three individual axes.
        gs.update(hspace=0.15, wspace=0.30)

        # Set axis labels.
        self._ax['sv'].set_xlabel('Cardiac cycle [-]')
        self._ax['pv'].set_xlabel('Cardiac cycle [-]')
        self._ax['w'].set_xlabel('Cardiac cycle [-]')
        self._ax['sv'].set_ylabel('Stroke volume [ml]')
        self._ax['pv'].set_ylabel('LV pressure [mmHg]')
        self._ax['w'].set_ylabel('LV work [J]')
        
        self._ax['sv'].set_title('Stroke Volumes')
        self._ax['pv'].set_title('Maximum cavity pressure')
        self._ax['w'].set_title('Work')

        # Set the global title.
        self._fig.suptitle('Change in hemodynamics during fiber reorientation')

        # Remove the right and top spines.
        [ax.spines['top'].set_visible(False) for _, ax in self._ax.items()]
        [ax.spines['right'].set_visible(False) for _, ax in self._ax.items()]
    

        
    def hemodymanics(self):       
        df = self._df
        var = self.var
        
        min_cyc = min(df[var['cycle']])
        max_cyc = max(df[var['cycle']])
        
        cycles = range(min_cyc, max_cyc)
        sv_cycs = []
        plv_cycs = []
        w_cycs = []
        
        for i in cycles:
            data_cycle = df[df[var['cycle']] == i]
            
            time = data_cycle[var['time']].values  # ms
            plv = kPa_to_mmHg(data_cycle[var['plv']].values)
            qao = data_cycle[var['qao']].values*60/1000  # ml/s -> l/min
            vlv = data_cycle[var['vlv']].values  # ml
            
            HR = round(60000/(max(time)-min(time)))
            EDV =  max(vlv)
            ESV = min(vlv)
            
            sv = EDV - ESV
            plv_max = max(plv)
            w = - np.trapz(mmHg_to_kPa(plv), vlv/1000)
            
            sv_cycs.append(sv)
            plv_cycs.append(plv_max)
            w_cycs.append(w)
            
        hemo_cycs = {
                    'SV': sv_cycs,
                    'plv_max': plv_cycs,
                    'W': w_cycs,
                    'cycles': cycles}
        return hemo_cycs
    
    def plot(self, *args):       
        hemo = self.hemodymanics()
        
        #make the stroke volume - cycles plot
        self._ax['sv'].plot(hemo['cycles'], hemo['SV'], *args)
        self._ax['sv'].xaxis.set_major_locator(MaxNLocator(integer=True))
        
        #make the maximum cavity pressure - cycles plot
        self._ax['pv'].plot(hemo['cycles'], hemo['plv_max'], *args)
        self._ax['pv'].xaxis.set_major_locator(MaxNLocator(integer=True))
        
        #make the work - cycles plot
        self._ax['w'].plot(hemo['cycles'], hemo['W'], *args)
        self._ax['w'].xaxis.set_major_locator(MaxNLocator(integer=True))

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

def hemodynamic_summary(data_cycle, model):
    data_cycle, var = dict_var_names(dataset = data_cycle, model = model)

    if var['plv'] in data_cycle.keys():
        #LV results
        time = data_cycle[var['time']].values  # ms
        plv = kPa_to_mmHg(data_cycle[var['plv']].values)
        part = kPa_to_mmHg(data_cycle[var['part']].values)
        pven = kPa_to_mmHg(data_cycle[var['pven']].values)
        qao = data_cycle[var['qao']].values*60/1000  # ml/s -> l/min
        vlv = data_cycle[var['vlv']].values  # ml
     
    elif 'pcav_s' in data_cycle.keys():
        # BiV results.
        time = data_cycle['time'].values  # ms
        plv = kPa_to_mmHg(data_cycle['pcav_s'].values)
        part = kPa_to_mmHg(data_cycle['part_s'].values)
        pven = kPa_to_mmHg(data_cycle['pven_s'].values)
        qao = data_cycle['qart_s'].values*60/1000  # ml/s -> l/min
        vlv = data_cycle['vcav_s'].values  # ml
        
    else:
        raise ValueError('Unexpected keys in data dictionary.')
        
    begin_cycle = data_cycle[var['time']][(data_cycle[var['phase']] == 1).idxmax()]
    end_cycle = data_cycle[var['time']][(data_cycle[var['phase']] == 4)[::-1].idxmax()]
    
    HR = round(60000/(end_cycle-begin_cycle))
    EDV =  max(vlv)
    ESV = min(vlv)
    SV = np.mean(qao)/HR * 1000
    SV_new = EDV - ESV
    CO = np.mean(qao)
     
    hemo = {    
            'HR [bpm]': HR,
            'EDV [ml]': EDV,
            'ESV [ml]':ESV,
            'CO [L/min]': CO,
            'qao [L/min]': np.mean(qao),
            # 'SV_old': SV,
            'SV [ml]': SV_new,
            # 'EF_old': SV/EDV*100,
            'EF [%]': SV_new/EDV*100,
            'MAP [mmHg]': np.mean(part),
            'SAP [mmHg]': max(part),
            'DAP [mmHg]': min(part),
            'PP [mmHg]': max(part) - min(part),
            'plv_max [mmHg]': max(plv),
            'W [J]': - np.trapz(mmHg_to_kPa(plv), vlv/1000),
            'dpdt_max [mmHg/s]': max(np.diff(plv)/np.diff(time/1000))
            }
    return hemo
 
def percentages(ref,var):
    change = (var-ref)/ref
    return change
 
def procentual_change_hemodynamics(ref, data, title1 = 'reference', title2='variation', model1= None, model2= None):
    hemo_ref = hemodynamic_summary(ref, model1)
    hemo_data = hemodynamic_summary(data, model2)
    
    perc_change = {
            'HR [bpm]': percentages(hemo_ref['HR [bpm]'],hemo_data['HR [bpm]']),
            'EDV [ml]': percentages(hemo_ref['EDV [ml]'],hemo_data['EDV [ml]']),
            'ESV [ml]':percentages(hemo_ref['ESV [ml]'],hemo_data['ESV [ml]']),
            'SV [ml]': percentages(hemo_ref['SV [ml]'],hemo_data['SV [ml]']),
            'EF [%]': percentages(hemo_ref['EF [%]'],hemo_data['EF [%]']),
            'CO [L/min]': percentages(hemo_ref['CO [L/min]'],hemo_data['CO [L/min]']),
            'MAP [mmHg]': percentages(hemo_ref['MAP [mmHg]'],hemo_data['MAP [mmHg]']),
            'SAP [mmHg]': percentages(hemo_ref['SAP [mmHg]'],hemo_data['SAP [mmHg]']),
            'DAP [mmHg]': percentages(hemo_ref['DAP [mmHg]'],hemo_data['DAP [mmHg]']),
            'PP [mmHg]': percentages(hemo_ref['PP [mmHg]'],hemo_data['PP [mmHg]']),
            'plv_max [mmHg]': percentages(hemo_ref['plv_max [mmHg]'],hemo_data['plv_max [mmHg]']),
            'dpdt_max [mmHg/s]': percentages(hemo_ref['dpdt_max [mmHg/s]'],hemo_data['dpdt_max [mmHg/s]']),
            'W [J]': percentages(hemo_ref['W [J]'],hemo_data['W [J]'])}
    
    hemo_ref = pd.DataFrame(hemo_ref,index=[0])
    hemo_data = pd.DataFrame(hemo_data,index=[0])
    perc_change = pd.DataFrame(perc_change,index=[0])
    
    print("\n{:<17} {:>10} {:>10} {:>13}".format('',title1,title2,'difference'))
    hemo_sum = pd.DataFrame(columns=[title1, title2,'difference (%)'])
    for key in perc_change.keys():
        print("{:<17} {:>10.2f} {:=10.2f} {:10.1%}".format(key, hemo_ref[key][0], hemo_data[key][0],perc_change[key][0]))
        var1 = round(hemo_ref[key][0],2)
        var2 = round(hemo_data[key][0],2)
        var3 = float(round(perc_change[key][0]*100,2))
        
        hemo_sum.loc[key] = [var1, var2, var3]
    
    return  hemo_sum    


def print_hemodynamic_summary(hemo,cycle):    
    print('\nSYSTEMIC CIRCULATION:')
    print('Results from cycle {}'.format(cycle))
    
    for key in hemo.keys():
        print("{:<17} {:=10.2f}".format(key, hemo[key]))
    # print(('HR: {:=10.0f} bpm\n' +
    #        'EDV: {:=10.2f} ml\n' +
    #        'ESV: {:=10.2f} ml\n' +
    #       'SV: {:=10.2f} ml\n' +
    #       # 'SV new: {:11.2f} ml\n' +
    #       'EF: {:=10.2f} %\n' +
    #       # 'EF_new: {:11.2f} %\n' +
    #       'qao: {:=10.2f} l/min \n' +
    #       'CO: {:=10.2f} l/min\n' +
    #       'MAP: {:=10.2f} mmHg\n' +
    #       'SAP: {:=10.2f} mmHg\n' +
    #       'DAP: {:=10.2f} mmHg\n' +
    #       'PP: {:=10.2f} mmHg\n' +
    #       'plv_max: {:=10.2f} mmHg\n' +
    #       'dp/dt_max: {:= 10.0f} mmHg/s\n' +
    #       'W: {:=10.2f} J').format(
    #               hemo['HR'], hemo['EDV'], hemo['ESV'], hemo['SV'],hemo['EF'],
    #               hemo['qao'], hemo['CO'], hemo['MAP'], 
    #               hemo['SAP'], hemo['DAP'], hemo['PP'],
    #               hemo['plv_max'], hemo['dpdt_max'], hemo['W']))
    if 'CVP' in hemo.keys():
        print('CVP: {:10.2f} mmHg'.format(hemo['CVP']))  
    print('\n')

def plot_results(results, dir_out='.', cycle=None, title = 'Hemodynamic relations', model = None):
            # LV only.
#    simulation_plot = HemodynamicsPlot(results)
#        
#    simulation_plot.plot(cycle=cycle) #, cycle=NUM_CYCLES)
    simulation_plot = HemodynamicsPlot(results, title, model = model)
    simulation_plot.plot(cycle=cycle) #, cycle=NUM_CYCLES)
    simulation_plot.save(os.path.join(dir_out, 'hemodynamics_cycle_{}.png'.format(cycle)))
       
    #simulation_plot.plot_function()
    
    
def plot_compare_results(results,results_var, dir_out='.', cycle=None, label_ref='ischemic', label_vars=['reference'], 
                         title = 'Hemodynamic relations', modelref = None, modelvars = [None]):
    linestyle = ['--', '-.', ':']
    simulation_plot = HemodynamicsPlot(results, title, model= modelref)
    simulation_plot.plot(label_ref[0], cycle=cycle) #, cycle=NUM_CYCLES)
    
    for ii, var in enumerate(results_var):
        simulation_plot.compare_against(label_ref, label_vars,var,cycle=cycle, model = modelvars[ii], linestyle = linestyle[ii])
    plt.savefig(os.path.join(dir_out, 'lv_function_compared.png'), dpi=300)
