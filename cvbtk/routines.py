"""
This module provides standard routines for LV and BiV simulations.
"""

from dolfin import parameters, DirichletBC, project
from dolfin.cpp.common import mpi_comm_world, MPI, info
from dolfin.cpp.io import HDF5File

import cvbtk
from cvbtk.utils import print_once, reset_values, read_dict_from_csv, \
    build_nullspace, info_once, vector_space_to_scalar_space, save_to_disk
from cvbtk.mechanics import ArtsKerckhoffsActiveStress, KerckhoffsMaterial, \
    BovendeerdMaterial, fiber_stretch_ratio, ArtsBovendeerdActiveStress
from cvbtk.geometries import LeftVentricleGeometry, BiventricleGeometry
from cvbtk.models import BiventricleModel, LeftVentricleModel
from cvbtk.windkessel import LifetecWindkesselModel, GeneralWindkesselModel, WindkesselModel, get_phase, get_phase_dc, \
    HeartMateII
from cvbtk.solvers import VolumeSolver, VolumeSolverBiV
from cvbtk.dataset import Dataset

from cvbtk.resources import reference_biventricle, reference_left_ventricle_pluijmert

import os
import time


__all__ = [
    'check_heart_type',
    'check_heart_type_from_inputs',
    'create_materials',
    'create_model',
    'create_windkessel_biv',
    'create_windkessel_lv',
    'load_model_state_from_hdf5',
    'preprocess_biv',
    'preprocess_lv',
    'ReloadState',
    'reset_model_state',
    'set_boundary_conditions',
    'set_initial_conditions_biv',
    'set_initial_conditions_lv',
    'simulate',
    'timestep_biv',
    'timestep_lv',
    'write_data_biv',
    'write_data_lv'
]


def check_heart_type(geometry):
    """
    Returns the heart type, based on the geometry.

    Args:
        geometry (cvbtk Geometry): geometry object (LeftVentricleGeometry or BiventricleGeometry)

    Returns:
        heart_type (str): 'LeftVentricle' or 'Biventricle'
    """
    # Test something from the geometry to figure out whether we have a
    # LV or a BiV geometry.
    if hasattr(geometry, 'rv_endocardium'):
        return 'Biventricle'
    else:
        return 'LeftVentricle'


def check_heart_type_from_inputs(inputs):
    """
    Returns the heart type, based on the simulation inputs.

    Args:
        inputs (dict): Dictionary with simulation inputs.

    Returns:
        heart_type (str): 'LeftVentricle' or 'Biventricle'
    """
    # Test something from the inputs to figure out whether we have a
    # LV or a BiV circulation.
    if 'wk_pul' in inputs.keys():
        heart_type = 'Biventricle'
    else:
        heart_type = 'LeftVentricle'
    return heart_type


def create_materials(model, inputs):
    """
    Adds passive and active materials to a model (LV or BiV).

    Args:
        model (cvbtk Model): Model object (e.g. LeftVentricleModel or BiventricleModel).
        inputs (dict): Simulation inputs.
    """
    # Extract fiber vectors.
    fsn = model.geometry.fiber_vectors()

    # Material models all take u, fsn, and arbitrary inputs for arguments.
    # It needs to be set by assigning it to the ``material`` attribute.
    u = model.u

    # Material models all take u, fsn, and arbitrary inputs for arguments.
    # It needs to be set by assigning it to the ``material`` attribute.
    law = inputs.get('material_law', 'BovendeerdMaterial')  # Default to BovendeerdMaterial.
    if law == 'KerckhoffsMaterial':
        model.material = KerckhoffsMaterial(u, fsn, **inputs['material_model'])
    elif law == 'BovendeerdMaterial':
        model.material = BovendeerdMaterial(u, fsn, **inputs['material_model'])
    else:
        raise ValueError('Unknown passive stress model specified.')

    # Active stress model.
    active_stress_model = inputs.get('active_stress_model', 'ArtsKerckhoffsActiveStress')  # Default to ArtsKerckhoffsActiveStress.
    if active_stress_model == 'ArtsBovendeerdActiveStress':
        act = ArtsBovendeerdActiveStress(u, fsn, **inputs['active_stress'])
    elif active_stress_model == 'ArtsKerckhoffsActiveStress':
        act = ArtsKerckhoffsActiveStress(u, fsn, **inputs['active_stress'])
    elif active_stress_model is None:
        act = None
    else:
        raise ValueError('Unknown active stress model specified.')

    model.active_stress = act


def create_model(geometry, inputs, heart_type):
    """
    Returns a model object (LV or BiV).

    Args:
         geometry (cvbtk Geometry): Geometry object (e.g. LeftVentricleGeometry or BiventricleGeometry).
         inputs (dict): Simulation inputs.
         heart_type (str): 'Biventricle' or 'LeftVentricle', specifying what kind of model to create.

    Returns:
        model (cvbtk Model): Model object (either BiventricleModel or LeftVentricleModel).
    """
    if heart_type == 'Biventricle':
        model = BiventricleModel(geometry, **inputs.get('model', {}))
    elif heart_type == 'LeftVentricle':
        model = LeftVentricleModel(geometry, **inputs.get('model', {}))
    else:
        raise ValueError('Unknown heart_type "{}".'.format(heart_type))

    # Check parameters.
    info_once(model.parameters, True)

    return model


def create_windkessel_biv(inputs):
    """
    Returns cvbtk Windkessel objects for the systemic and pulmonary circulations.

    Args:
         inputs (dict): Simulation inputs.

    Returns:
        Dictionary with cvbtk Windkessel objects for systemic ('sys') and pulmonary ('pul') circulation.
    """
    # Create the windkessel model for the systemic circulation part with the loaded inputs.
    wk_sys = GeneralWindkesselModel('systemic_windkessel', **inputs['wk_sys'])

    # Create the windkessel model for the pulmonary circulation part with the loaded inputs.
    wk_pul = GeneralWindkesselModel('pulmonary_windkessel', **inputs['wk_pul'])

    # Add LVAD.
    if inputs.get('attach_lvad', False):
        wk_sys.lvad = HeartMateII(**inputs['lvad_model'])

    # Print parameters.
    info_once(wk_sys.parameters, True)
    info_once(wk_pul.parameters, True)

    return {'sys': wk_sys, 'pul': wk_pul}


def create_windkessel_lv(inputs):
    """
    Returns cvbtk Windkessel object for the systemic circulation.

    Args:
         inputs (dict): Simulation inputs.

    Returns:
        Dictionary with cvbtk Windkessel object for systemic ('sys') circulation.
    """
    # Create the windkessel model for the circulation/boundary condition part.
    wk_type = inputs.get('windkessel_type', 'WindkesselModel')
    if wk_type == 'WindkesselModel':
        wk = WindkesselModel(**inputs['windkessel_model'])
    elif wk_type == 'LifetecWindkesselModel':
        wk = LifetecWindkesselModel(**inputs['windkessel_model'])
    else:
        raise ValueError('Unknown wk_type "{}"'.format(wk_type))

    # Add LVAD.
    if inputs.get('attach_lvad', False):
        wk.lvad = HeartMateII(**inputs['lvad_model'])

    # Print parameters.
    info_once(wk.parameters, True)

    return {'sys': wk}


def load_model_state_from_hdf5(model, results_hdf5, vector_number, fiber_reorientation=False):
    """
    Loads a model state into a cvbtk Model object.

    Args:
        model (cvbtk Model): Model object (e.g. LeftVentricleModel or BiventricleModel).
        results_hdf5 (str): Filename to HDF5 file containing displacement and
                            optionally contractile element length and fiber vectors.
        vector_number (int): Vector number in HDF5 file to load.
        fiber_reorientation (boolean): Specify whether fiber reorientation was on during simulation.
    """
    with HDF5File(mpi_comm_world(), results_hdf5, 'r') as f:

        # Displacement
        u_vector = 'displacement/vector_{}'.format(vector_number)
        f.read(model.u, u_vector)

        if vector_number-1 <= 0:
            raise ValueError('Cannot reload u_old from results file. Vector number = {}.'.format(vector_number-1))

        u_old_vector = 'displacement/vector_{}'.format(vector_number-1)
        f.read(model.u_old, u_old_vector)

        if vector_number-2 >= 0:
            u_old_old_vector = 'displacement/vector_{}'.format(vector_number-2)
            f.read(model.u_old_old, u_old_old_vector)
        else:
            print_once('Warning: Cannot reload u_old_old from results file. Vector number = {}'.format(vector_number-2))

        # If fiber vectors are changing (reorienting), we need the old fiber vectors (at previous timestep) to
        # compute ls_old.
        if fiber_reorientation:
            # Fiber vectors are saved in results file.
            model.geometry.load_fiber_field(openfile=f, vector_number=vector_number - 1)

        # Old sarcomere length.
        ef = model.active_stress.fiber_vectors[0]
        u_old = model.u_old
        model.active_stress.ls_old = model.active_stress.parameters['ls0']*fiber_stretch_ratio(u_old, ef)

        # Contractile element length
        if isinstance(model.active_stress, ArtsKerckhoffsActiveStress):
            # Load lc_old.
            lc_vector = 'contractile_element/vector_{}'.format(vector_number)
            f.read(model.active_stress.lc_old, lc_vector)

        # Now, load current fiber vectors.
        if fiber_reorientation:
            # Fiber vectors are saved in results file.
            model.geometry.load_fiber_field(openfile=f, vector_number=vector_number)


def preprocess_biv(inputs):
    """
    Pre-processing routine for biventricular simulations.

    Args:
        inputs(dict): Simulation inputs.

    Returns:
        wk (dict): Dictionary with initialized cvbtk Windkessel objects
                   for systemic ('sys') and pulmonary ('pul') circulation.
        biv (cvbtk Model): Initialized BiventricleModel.
        results (cvbtk.Dataset): Dataset for the results.
    """
    # ------------------------------------------------------------------------ #
    # Create a dataset container to store state values.                        #
    # ------------------------------------------------------------------------ #
    dataset_keys = ['time', 't_cycle', 't_act', 'cycle', 'phase_s', 'phase_p',
                    'pcav_s', 'part_s', 'pven_s', 'qart_s', 'qper_s', 'qven_s', 'vcav_s', 'vart_s', 'vven_s',
                    'pcav_p', 'part_p', 'pven_p', 'qart_p', 'qper_p', 'qven_p', 'vcav_p', 'vart_p', 'vven_p',
                    'a_s', 'b_s', 'c_s', 'a_p', 'b_p', 'c_p',
                    'est', 'accuracy', 'vector_number']

    if inputs.get('attach_lvad', False):
        dataset_keys.append('qlvad')  # flowrate of lvad

    # Create the dataset.
    results = Dataset(keys=dataset_keys)

    # ------------------------------------------------------------------------ #
    # Set the proper global FEniCS parameters.                                 #
    # ------------------------------------------------------------------------ #
    parameters.update({'form_compiler': inputs['form_compiler']})

    # ------------------------------------------------------------------------ #
    # Create the windkessel models for the systemic and pulmonary circulation. #
    # ------------------------------------------------------------------------ #
    wk = create_windkessel_biv(inputs)

    # ------------------------------------------------------------------------ #
    # Create the finite element biventricle.                                   #
    # ------------------------------------------------------------------------ #
    # TODO BiV geometry to load is hardcoded. Could be specified in inputs if there a more BiV geometries.
    # For the geometry we re-use the reference mesh.
    res = inputs['geometry']['mesh_resolution']
    geometry = reference_biventricle(resolution=res, **inputs['geometry'])

    # Create model and set boundary conditions.
    biv = create_model(geometry, inputs, 'Biventricle')
    set_boundary_conditions(biv)

    # Add material laws to the model.
    create_materials(biv, inputs)

    # ------------------------------------------------------------------------ #
    # Inspect the parameters of the windkessel and BiV models:                 #
    # ------------------------------------------------------------------------ #
    if MPI.rank(mpi_comm_world()) == 0:
        info(wk['sys'].parameters, True)
        info(wk['pul'].parameters, True)
        info(biv.parameters, True)
        info(biv.geometry.parameters, True)

    # ------------------------------------------------------------------------ #
    # Set the initial conditions.                                              #
    # ------------------------------------------------------------------------ #
    set_initial_conditions_biv(wk, biv, inputs)

    # Print out the current (initial) state just to double check the values:
    if MPI.rank(mpi_comm_world()) == 0:
        print('The initial systemic WK state is V = {}.'.format(wk['sys'].volume))
        print('The initial systemic WK state is p = {}.'.format(wk['sys'].pressure))
        print('The initial systemic WK state is q = {}.'.format(wk['sys'].flowrate))

        print('The initial pulmonary WK state is V = {}.'.format(wk['pul'].volume))
        print('The initial pulmonary WK state is p = {}.'.format(wk['pul'].pressure))
        print('The initial pulmonary WK state is q = {}.'.format(wk['pul'].flowrate))
        print('The initial BiV state is V = {}.'.format(biv.volume))
        print('The initial BiV state is p = {}.'.format(biv.pressure))

    return wk, biv, results


def preprocess_lv(inputs):
    """
    Pre-processing routine for leftventricle simulations.

    Args:
        inputs(dict): Simulation inputs.

    Returns:
        wk (dict): Dictionary with initialized cvbtk Windkessel object
                   for systemic ('sys') circulation.
        lv (cvbtk Model): Initialized LeftVentricleModel.
        results (cvbtk.Dataset): Dataset for the results.
    """
    # ------------------------------------------------------------------------ #
    # Create a dataset container to store state values.                        #
    # ------------------------------------------------------------------------ #
    # Populate it with the same keys (columns) as the Sepran data.
    sepran_keys = cvbtk.resources.reference_hemodynamics().keys()

    # Add the following additional keys.
    sepran_keys.append('t_act')  # time before/since activation
    sepran_keys.append('t_cycle')  # time in the current cycle
    sepran_keys.append('vector_number')  # corresponds to the HDF5 output
    if inputs.get('attach_lvad', False):
        sepran_keys.append('qlvad')  # flowrate of lvad

    # Create the dataset.
    results = Dataset(keys=sepran_keys)

    # ------------------------------------------------------------------------ #
    # Set the proper global FEniCS parameters.                                 #
    # ------------------------------------------------------------------------ #
    parameters.update({'form_compiler': inputs['form_compiler']})

    # ------------------------------------------------------------------------ #
    # Create the windkessel model for the circulation/boundary condition part. #
    # ------------------------------------------------------------------------ #
    wk = create_windkessel_lv(inputs)

    # ------------------------------------------------------------------------ #
    # Create the finite element model for the left ventricle.                  #
    # ------------------------------------------------------------------------ #
    # For the geometry we re-use the reference mesh.
    res = inputs['geometry']['mesh_resolution']

    # Reference mesh name is specified in inputs. Else default is 'reference_left_ventricle_pluijmert'.
    geometry_type = inputs.get('geometry_type', 'reference_left_ventricle_pluijmert')
    print_once('Loading geometry type "{}"...'.format(geometry_type))

    # Load the desired mesh.
    if geometry_type == 'reference_left_ventricle_pluijmert':
        geometry = cvbtk.resources.reference_left_ventricle_pluijmert(resolution=res, **inputs['geometry'])
    elif geometry_type == 'reference_left_ventricle':
        geometry = cvbtk.resources.reference_left_ventricle(resolution=res, **inputs['geometry'])
    else:
        raise ValueError('Unknwon geometry type.')

    # Check parameters.
    info_once(geometry.parameters, True)

    # Create model and set boundary conditions.
    lv = create_model(geometry, inputs, 'LeftVentricle')
    set_boundary_conditions(lv)

    # Add material laws to the model.
    create_materials(lv, inputs)

    # ------------------------------------------------------------------------ #
    # Set the initial conditions.                                              #
    # ------------------------------------------------------------------------ #
    set_initial_conditions_lv(wk, lv, inputs)

    # Print out the current (initial) state just to double check the values:
    if MPI.rank(mpi_comm_world()) == 0:
        print('Initialized state: p = {}, {}.'.format(wk['sys'].pressure, lv.pressure))
        print('Initialized state: V = {}, {}.'.format(wk['sys'].volume, lv.volume))
        print('Initialized state: q = {}.'.format(wk['sys'].flowrate))

    return wk, lv, results


class ReloadState(object):
    """
    Reloads a state to the LeftVentricleModel or BiventricleModel.

    Use the main function 'reload' to do the reloading, e.g.:
    wk, lv, results, inputs = ReloadState().reload('output/stopped_simulation', 'latest')
    See the 'reload' function for more details on inputs and outputs.
    """

    def __init__(self):
        self.inputs = None
        self.results_csv = None
        self.results_hdf5 = None
        self.heart_type = None

    def reload(self, directory, t, **kwargs):
        """
        Main function.
        Reloads a state (compatible with LeftVentricleModel and BiventricleModel).

        Args:
            directory: Directory in which the inputs.csv, results_cycle_x.hdf5 and results.csv files
                       exists from which to reload a state.
            t: The time (in ms) of the state to be reloaded, or -1.
               If the last saved timestep should be reloaded, set t to -1.
            **kwargs: Optional key-worded arguments to overrule loaded inputs
                      (by default the saved inputs from the reloaded simulation are used).

        Returns:
            wk (dict), model (cvbtk.Model), results (cvbtk.Dataset), inputs (dict)
            Using these as inputs for 'simulate' will resume the simulation.
        """
        # TODO validate for LVAD simulations.
        print_once('Reloading state from {} directory at t = {} ...'.format(directory, t))

        # Read inputs CSV.
        self.inputs = read_dict_from_csv(os.path.join(directory, 'inputs.csv'))

        # Overrule loaded inputs with key-worded input arguments.
        self.inputs.update(kwargs)

        # Read results CSV.
        self.results_csv = Dataset(filename=os.path.join(directory, 'results.csv'))

        # Default timestep to reload.
        if t == -1:
            # Select latest timestep in CSV file.
            t = self.results_csv['time'].values.tolist()[-1]

        # Find the index for the csvfile and vector number for the hdf5file of the state to reload.
        idx, cycle, vector_number = self._get_index_and_vector_number(t)

        # Store path to results HDF5 file.
        self.results_hdf5 = os.path.join(directory, 'results_cycle_{}.hdf5'.format(cycle))

        # Set the proper global FEniCS parameters.
        parameters.update({'form_compiler': self.inputs['form_compiler']})

        # Deduct type of model for the heart from inputs.
        heart_type = check_heart_type_from_inputs(self.inputs)

        # Set the 'load_fiber_field_from_meshfile' parameter to True.
        self.inputs['geometry']['load_fiber_field_from_meshfile'] = True

        # Create the windkessel model(s) and the geometry.
        # Note that wk is a dictionary with keys 'sys' (and 'pul' in case of a Biventricle).
        # Here, 'sys' contains the systemic circulation windkessel
        # and 'pul' the pulmonary circulation windkessel.
        if heart_type == 'Biventricle':
            wk = create_windkessel_biv(self.inputs)
            geometry = self._load_geometry_biv()
        elif heart_type == 'LeftVentricle':
            wk = create_windkessel_lv(self.inputs)
            geometry = self._load_geometry_lv()
        else:
            raise ValueError('Unknown heart_type "{}".'.format(heart_type))

        # Create the model and set boundary conditions.
        model = create_model(geometry, self.inputs, heart_type)
        set_boundary_conditions(model)

        # Create materials.
        create_materials(model, self.inputs)

        # Load initial state at specified timestep and set initial conditions and create a Dataset.
        if heart_type == 'Biventricle':
            # Load state.
            self._load_state_biv(model, idx, vector_number)
            set_initial_conditions_biv(wk, model, self.inputs)
        elif heart_type == 'LeftVentricle':
            # Load state.
            self._load_state_lv(model, idx, vector_number)
            set_initial_conditions_lv(wk, model, self.inputs)
        else:
            raise ValueError('Unknown heart_type "{}".'.format(heart_type))

        dataset_keys = self.results_csv.keys()

        # Create the dataset.
        results = Dataset(keys=dataset_keys)

        # Inspect the parameters of the models:
        if MPI.rank(mpi_comm_world()) == 0:
            info(wk['sys'].parameters, True)
            if 'pul' in wk.keys():
                info(wk['pul'].parameters, True)
            info(model.parameters, True)
            info(model.geometry.parameters, True)

        return wk, model, results, self.inputs

    def _get_index_and_vector_number(self, t):
        # Extract the time array.
        time = self.results_csv['time'].values.tolist()

        # Get the index of the state to be reloaded.
        idx = time.index(float(t))
        if idx < 2:
            raise RuntimeError('Cannot reload the first timestep. Consider not reloading a state.')

        cycle = list(self.results_csv['cycle'])[idx]
        vector_number = list(self.results_csv['vector_number'])[idx]

        return idx, cycle, vector_number

    def _load_geometry_biv(self):
        geometry_inputs = self.inputs['geometry']
        try:
            geometry = BiventricleGeometry(meshfile=self.results_hdf5, **geometry_inputs)
        except RuntimeError as error_detail:
            print_once('Except RuntimeError: {}'.format(error_detail))
            print_once('Failed to load mesh from result.hdf5 file. Loading reference mesh...')
            # Load the reference geometry otherwise.
            res = self.inputs['geometry']['mesh_resolution']
            geometry = reference_biventricle(resolution=res, **geometry_inputs)
        return geometry

    def _load_geometry_lv(self):
        geometry_inputs = self.inputs['geometry']
        try:
            geometry = LeftVentricleGeometry(meshfile=self.results_hdf5, **geometry_inputs)
        except RuntimeError as error_detail:
            print_once('Except RuntimeError: {}'.format(error_detail))
            print_once('Failed to load mesh from result.hdf5 file. Loading reference mesh...')
            # Load the reference geometry otherwise.
            res = self.inputs['geometry']['mesh_resolution']
            geometry = reference_left_ventricle_pluijmert(resolution=res, **geometry_inputs)
        return geometry

    def _load_state_biv(self, biv, idx, vector_number):

        inputs = self.inputs
        results_csv = self.results_csv
        results_hdf5 = self.results_hdf5

        # Extract data from CSV at timestep t and save it to the inputs for the windkessel models.
        initial_conditions = {'p_art_sys': results_csv['part_s'].values.tolist()[idx],
                              'p_art_pul': results_csv['part_p'].values.tolist()[idx],
                              'p_ven_pul': results_csv['pven_p'].values.tolist()[idx]}
        inputs['initial_conditions'] = initial_conditions

        # Extract data from CSV at timestep t and save it to the BiV model.
        # Pressure
        biv._plv = results_csv['pcav_s'].values.tolist()[idx]
        biv._prv = results_csv['pcav_p'].values.tolist()[idx]

        # Pressure old
        biv._plv_old = results_csv['pcav_s'].values.tolist()[idx-1]
        biv._prv_old = results_csv['pcav_p'].values.tolist()[idx-1]

        # Pressure old old
        biv._plv_old_old = results_csv['pcav_s'].values.tolist()[idx-2]
        biv._prv_old_old = results_csv['pcav_p'].values.tolist()[idx-2]

        # Load time variables.
        self._load_times(biv, inputs, results_csv, idx)

        # Extract data from HDF5 at timestep t.
        fiber_reorientation = True if inputs['model']['fiber_reorientation']['ncycles_reorient'] > 0 else False
        load_model_state_from_hdf5(biv, results_hdf5, vector_number, fiber_reorientation)

    def _load_state_lv(self, lv, idx, vector_number):

        inputs = self.inputs
        results_csv = self.results_csv
        results_hdf5 = self.results_hdf5

        # Extract data from CSV at timestep t and save it to the inputs for the windkessel models.
        initial_conditions = {'arterial_pressure': results_csv['part'].values.tolist()[idx]}
        inputs['initial_conditions'] = initial_conditions

        # Extract data from CSV at timestep t and save it to the LV model.
        # Pressure
        lv._plv = results_csv['plv'].values.tolist()[idx]

        # Pressure old
        lv._plv_old = results_csv['plv'].values.tolist()[idx-1]

        # Load time variables.
        self._load_times(lv, inputs, results_csv, idx)

        # Extract data from HDF5 at timestep t.
        fiber_reorientation = True if inputs['model']['fiber_reorientation']['ncycles_reorient'] > 0 else False
        load_model_state_from_hdf5(lv, results_hdf5, vector_number, fiber_reorientation)

    @staticmethod
    def _load_times(model, inputs, results_csv, idx):
        # Update the time variables of the FE model
        # dt
        time = results_csv['time'].values.tolist()
        model.dt = time[idx] - time[idx - 1]  # using the setter, the dt of the active stress model is set, too.

        # dt old
        model._dt_old = time[idx - 1] - time[idx - 2]
        model._dt_old_old = time[idx - 2] - time[idx - 3]

        # Activation time.
        if model.active_stress is not None:
            model.active_stress.activation_time = float(
                results_csv['t_act'].values.tolist()[idx] + model.active_stress.parameters['tdep'])

        # Load the global starting time, cycle time, activation time, cycle number and phase.
        heart_type = check_heart_type_from_inputs(inputs)
        if heart_type == 'Biventricle':
            reloaded_phase = {'lv': results_csv['phase_s'].values.tolist()[idx],
                              'rv': results_csv['phase_p'].values.tolist()[idx]}
        elif heart_type == 'LeftVentricle':
            reloaded_phase = {'lv': results_csv['phase'].values.tolist()[idx]}
        else:
            raise ValueError('Unknown heart_type "{}".'.format(heart_type))

        inputs['state']['phase'] = reloaded_phase
        inputs['time']['t0'] = results_csv['time'].values.tolist()[idx]
        inputs['state']['t_cycle'] = results_csv['t_cycle'].values.tolist()[idx]
        inputs['state']['cycle'] = results_csv['cycle'].values.tolist()[idx]
        if model.active_stress is not None:
            inputs['state']['t_active_stress'] = float(
                results_csv['t_act'].values.tolist()[idx] + model.active_stress.parameters['tdep'])
        else:
            inputs['state']['t_active_stress'] = 0.

        # TODO verify that we do not need to return inputs.


def reset_model_state(model, dt_old, activation_time_old, u_array, u_old_array, ls_old_array, lc_old_array=None,
                      ef_old_array=None, es_old_array=None, en_old_array=None):
    """
    Helper function to reset a model state.
    """
    # Reset time.
    model.dt = dt_old
    model.active_stress.activation_time = activation_time_old

    # Reset FEniCS functions:
    reset_values(model.u, u_array)
    reset_values(model.u_old, u_old_array)
    reset_values(model.active_stress.ls_old, ls_old_array)

    # Reset lc_old if a copy was saved.
    if lc_old_array is not None:
        reset_values(model.active_stress.lc_old, lc_old_array)

    # Reset the fiber vectors if a copy was saved.
    if ef_old_array is not None:
        model.geometry.set_fiber_vectors(ef_old_array, es_old_array, en_old_array)


def save_model_state_to_hdf5(model, hdf5_filename, t, new=False, save_fiber_vector=False):
    """
    Helper function to save a model state to a HDF5 file (works for LV and Biv).

    Args:
        model (cvbtk Model): Model object (e.g. LeftVentricleModel or BiventricleModel).
        hdf5_filename (str): Filename to HDF5 file.
        t (int, float): Time stamp.
        new (boolean): Specify whether a new file must be created (True) or if the
                       state must be appended to an existing file (False).
        save_fiber_vector (boolean): Specify whether to save the fiber vector (True) or not (False).
    """

    # Create a new HDF5 record saving the mesh and geometric parameters.
    if new:
        model.geometry.save_mesh_to_hdf5(hdf5_filename, save_fiber_vector=False)

    # Write the displacement and contractile element to disk.
    with HDF5File(mpi_comm_world(), hdf5_filename, 'a') as f:

        # Write the primary displacement unknown.
        f.write(model.u, 'displacement', t)

        if model.active_stress is not None:
            if isinstance(model.active_stress, ArtsKerckhoffsActiveStress):
                # Save contractile element length.
                f.write(model.active_stress.lc_old, 'contractile_element', t)

        # Write the fiber vectors if requested. Always save the initial fiber vector (if new==True).
        if new or save_fiber_vector:
            ef = model.geometry.fiber_vectors()[0].to_function(None)
            f.write(ef, 'fiber_vector', t)

        # For the CSV file:
        vector_number = f.attributes('displacement')['count'] - 1

        return vector_number


def set_boundary_conditions(model):
    """
    Helper function to add boundary conditions to a cvbtk Model.

    Args:
        model (cvbtk Model): Model object (e.g. LeftVentricleModel or BiventricleModel).
    """
    # Model defines u, from which V can be collected.
    u = model.u
    V = u.ufl_function_space()

    # Dirichlet boundary conditions fix the base.
    model.bcs = DirichletBC(V.sub(2), 0.0, model.geometry.tags(), model.geometry.base)

    # We have not fully eliminated rigid body motion yet. To do so, we will
    # define a nullspace of rigid body motions and use a iterative method which
    # can eliminate rigid body motions using this nullspace.
    model.nullspace = build_nullspace(u, modes=['x', 'y', 'xy'])


def set_initial_conditions_biv(wk_dict, biv, inputs):
    """
    Initializes cvbtk Windkessel and Model objects with initial conditions for biventricular simulations.

    Args:
        wk_dict (dict): Dictionary with cvbtk Windkessel objects for systemic ('sys')
                        and pulmonary ('pul') circulation.
        biv (cvbtk Model): BiventricleModel.
        inputs (dict): Simulation inputs.
    """
    # Extract the windkessel objects.
    wk_sys = wk_dict['sys']
    wk_pul = wk_dict['pul']

    # Determine initial state by setting initial pressures (in kPa).
    wk_sys.pressure = {'art': inputs['initial_conditions']['p_art_sys']}

    wk_pul.pressure = {'art': inputs['initial_conditions']['p_art_pul'],
                       'ven': inputs['initial_conditions']['p_ven_pul']}

    # Compute initial volumes from initial pressures
    biv.volume = biv.compute_volume()
    wk_sys.volume = wk_sys.compute_volume()
    wk_pul.volume = wk_pul.compute_volume()

    # Compute initial venous volume (systemic) from mass conservation.
    vven_sys = inputs['total_volume'] - biv.volume['lv'] - biv.volume['rv'] - wk_sys.volume['art'] - wk_pul.volume[
        'art'] - wk_pul.volume['ven'] - wk_sys.volume.get('lvad', 0)
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


def set_initial_conditions_lv(wk_dict, lv, inputs):
    """
    Initializes cvbtk Windkessel and Model objects with initial conditions for leftventricle simulations.

    Args:
        wk_dict (dict): Dictionary with cvbtk Windkessel object for systemic ('sys') circulation.
        lv (cvbtk Model): LeftVentricleModel.
        inputs (dict): Simulation inputs.
    """
    # Extract the windkessel object.
    wk = wk_dict['sys']

    # Set the initial conditions.
    lv.volume = {'lv': lv.compute_volume()}

    # We've defined the initial arterial pressure , from which
    # the initial arterial volume can be computed and set:
    wk.pressure = {'art': inputs['initial_conditions']['arterial_pressure']}
    wk.volume = wk.compute_volume(wk.pressure)

    # The missing volume is the venous volume, which comes from conservation:
    vven = wk.parameters['total_volume'] - wk.volume['art'] - lv.volume['lv'] \
           - wk.volume.get('lvad', 0)
    wk.volume = {'ven': vven}

    # The resulting venous pressure can be computed from the venous volume:
    # Note that for the LifetecWindkessel the venous pressure is constant
    # and not depending on venous volume.
    wk.pressure = {'ven': wk.compute_pressure(wk.volume)['ven']}

    # The resulting initial flowrate can be computed from the initial pressures:
    wk.flowrate = wk.compute_flowrate(wk.pressure, lv.pressure)


def simulate(wk, model, results, inputs, heart_type=None, solver=None, dir_out='output/'):
    """
    Routine for simulating the circulation modelled by
    0D windkessel models and the heart by a FEM model.

    Args:
        wk (dict): Dictionary with the windkessel model for the systemic circulation ('sys'),
                   and optionally a windkessel model for the pulmonary cirulation ('pul').
        model (cvbtk Model): Model object (e.g. LeftVentricleModel or BiventricleModel).
        results (cvbtk.Dataset): Dataset for the results.
        inputs (dict): Dictionary with inputs.
        heart_type (str): Either 'LeftVentricle' or 'Biventricle'.
        solver (optionally): cvbtk Volume solver.
        dir_out (str, optionally): Output directory.
    """
    if heart_type is None:
        # Deduct type of model for the heart from inputs.
        heart_type = check_heart_type_from_inputs(inputs)

    # ------------------------------------------------------------------------ #
    # Set up the simulation loop.                                              #
    # ------------------------------------------------------------------------ #
    dt = inputs['time']['dt']
    t0 = inputs['time']['t0']
    t1 = inputs['time']['t1']

    phase = inputs['state']['phase']
    cycle = inputs['state']['cycle']
    t_cycle = inputs['state']['t_cycle']
    t_active_stress = inputs['state']['t_active_stress']

    t = t0

    fiber_reorientation = True if inputs['model']['fiber_reorientation']['ncycles_reorient'] > 0 else False

    if heart_type == 'Biventricle':
        # Create the output data files.
        write_data_biv(t, t_cycle, phase, cycle, wk, model, 0, 0, results, new=True, dir_out=dir_out,
                       save_fiber_vector=fiber_reorientation)

        # Create the volume solver to solve the system if not already given.
        if solver is None:
            solver = VolumeSolverBiV(**inputs['volume_solver'])

    elif heart_type == 'LeftVentricle':
        # Create the output data files.
        write_data_lv(t, t_cycle, phase, cycle, wk, model, 0, 0, results, new=True, dir_out=dir_out,
                      save_fiber_vector=fiber_reorientation)

        # Create the volume solver to solve the system.                            #
        if solver is None:
            solver = VolumeSolver(**inputs['volume_solver'])

    # ------------------------------------------------------------------------ #
    # The time loop is below.                                                  #
    # ------------------------------------------------------------------------ #
    while t < t1:
        # -------------------------------------------------------------------- #
        # t = n                                                                #
        # t = n                                                                #
        # -------------------------------------------------------------------- #
        # Store a copy/backup of the state values and unknowns at t = n:
        u_array = model.u.vector().array()
        u_old_array = model.u_old.vector().array()
        ls_old_array = model.active_stress.ls_old.vector().array()

        if isinstance(model.active_stress, ArtsKerckhoffsActiveStress):
            lc_old_array = model.active_stress.lc_old.vector().array()
        else:
            lc_old_array = None

        if fiber_reorientation:
            ef_old_array, es_old_array, en_old_array = [_.to_array(None) for _ in model.geometry.fiber_vectors()]
        else:
            ef_old_array, es_old_array, en_old_array = None, None, None

        dt_old = model.dt*1
        activation_time_old = t_active_stress

        # Define this boolean, to be able to force to disable fiber reorientation per timestep
        # if it fails. At the start of a timestep, do not disable.
        disable_reorientation = False

        # -------------------------------------------------------------------- #
        # timestep                                                             #
        # -------------------------------------------------------------------- #
        try_again = True
        while try_again:
            # By default do not try again, assuming the simulation does not fail.
            # If it fails in the end with fiber reorientation enabled,
            # we reset this bool to True and retry without fiber reorientation.
            # This retrying with disabling fiber reorientation may not be needed anymore,
            # it was used to make older simulations run. In retro-sepct, those simulations
            # failed in the first place due to a problem with fiber vector definition.
            # Now, the fiber vectors are fixed and this retrying option may be redundant.
            try_again = False

            # Since it's possible for the FEM solver to fail, it's better if we
            # make the rest of the time-loop its own function so that we can take
            # advantage of Python's try/except construct.
            # The idea is if the solution fails, then we'll reset u, ls, lc, etc.,
            # lower the dt, and call the time-step routine again.
            try:
                # Attempt to solve.
                if heart_type == 'Biventricle':
                    accuracy = timestep_biv(t_active_stress, dt, cycle, wk, model, solver, inputs['total_volume'],
                                            fiber_reorientation, disable_reorientation=disable_reorientation)
                elif heart_type == 'LeftVentricle':
                    accuracy = timestep_lv(t_active_stress, dt, cycle, wk, model, solver,
                                           fiber_reorientation, disable_reorientation=disable_reorientation)
                else:
                    raise ValueError('Unknown heart_type "{}".'.format(heart_type))

                # Update time states after successful solutions.
                t += dt
                t_cycle += dt
                t_active_stress += dt

            except RuntimeError as error_detail:
                print_once('Except RuntimeError: {}'.format(error_detail))
                print_once('Failed to solve. Halving dt and re-attempting...')
                # Reset values from backup:
                reset_model_state(model, dt_old, activation_time_old, u_array, u_old_array, ls_old_array, lc_old_array,
                                  ef_old_array, es_old_array, en_old_array)

                # Re-attempt to solve.
                try:
                    # Attempt to solve.
                    if heart_type == 'Biventricle':
                        accuracy = timestep_biv(t_active_stress, 0.5*dt, cycle, wk, model, solver, inputs['total_volume'],
                                                fiber_reorientation, disable_reorientation=disable_reorientation)
                    elif heart_type == 'LeftVentricle':
                        accuracy = timestep_lv(t_active_stress, 0.5*dt, cycle, wk, model, solver,
                                               fiber_reorientation, disable_reorientation=disable_reorientation)
                    else:
                        raise ValueError('Unknown heart_type "{}".'.format(heart_type))

                    # Update time states after successful solutions.
                    t += 0.5*dt
                    t_cycle += 0.5*dt
                    t_active_stress += 0.5*dt

                except RuntimeError as error_detail:
                    print_once('Except RuntimeError: {}'.format(error_detail))
                    print_once('Failed to solve. Halving dt again and re-attempting...')
                    # Reset values from backup:
                    reset_model_state(model, dt_old, activation_time_old, u_array, u_old_array, ls_old_array,
                                      lc_old_array, ef_old_array, es_old_array, en_old_array)

                    # Re-attempt to solve.
                    try:
                        # Attempt to solve.
                        if heart_type == 'Biventricle':
                            accuracy = timestep_biv(t_active_stress, 0.25*dt, cycle, wk, model, solver,
                                                    inputs['total_volume'],
                                                    fiber_reorientation, disable_reorientation=disable_reorientation)
                        elif heart_type == 'LeftVentricle':
                            accuracy = timestep_lv(t_active_stress, 0.25*dt, cycle, wk, model, solver,
                                                   fiber_reorientation, disable_reorientation=disable_reorientation)
                        else:
                            raise ValueError('Unknown heart_type "{}".'.format(heart_type))

                        # Update time states after successful solutions.
                        t += 0.25*dt
                        t_cycle += 0.25*dt
                        t_active_stress += 0.25*dt

                    except RuntimeError as error_detail:
                        print_once('Except RuntimeError: {}'.format(error_detail))
                        if fiber_reorientation and not disable_reorientation:
                            # Disable reorientation and try again.
                            # Reset values from backup:
                            reset_model_state(model, dt_old, activation_time_old, u_array, u_old_array, ls_old_array,
                                              lc_old_array, ef_old_array, es_old_array, en_old_array)

                            try_again = True
                            disable_reorientation = True
                            print_once('Simulation failed. Trying without fiber reorientation.')
                        else:
                            raise RuntimeError('Simulation failed.')

        # -------------------------------------------------------------------- #
        # t = n + 1                                                            #
        # -------------------------------------------------------------------- #
        # Check what the new phases are.
        phase_old = phase
        if heart_type == 'Biventricle':
            phase = get_phase_dc(model.pressure_old, wk['sys'].pressure, wk['pul'].pressure, model.pressure)
        elif heart_type == 'LeftVentricle':
            phase = {'lv': get_phase(model.pressure_old, wk['sys'].pressure, model.pressure)}
        else:
            raise ValueError('Unknown heart_type "{}".'.format(heart_type))

        # Increment cycle count if needed.
        if phase['lv'] == 1 and phase_old['lv'] == 4:
            cycle += 1
            t_cycle = 0.0

            # For output file.
            new = True

        else:
            # For output file.
            new = False

        # Append selected data to the file records.
        est = solver.iteration()
        if heart_type == 'Biventricle':
            write_data_biv(t, t_cycle, phase, cycle, wk, model, est, accuracy, results, new=new, dir_out=dir_out,
                           save_fiber_vector=fiber_reorientation)

            # Print some state information about the completed timestep:
            if MPI.rank(mpi_comm_world()) == 0:
                msg = ('*** [Cycle {}/{} - Phase LV = {}/4 - Phase RV = {}/4]:'
                       ' t = {} ms, t_cycle = {} ms,'
                       ' p_lv = {:5.2f} kPa, V_lv = {:6.2f} ml'
                       ' p_rv = {:5.2f} kPa, V_rv = {:6.2f} ml')
                print(msg.format(cycle, inputs['number_of_cycles'], phase['lv'], phase['rv'],
                                 t, t_cycle,
                                 model.pressure['lv'], model.volume['lv'],
                                 model.pressure['rv'], model.volume['rv']))

        elif heart_type == 'LeftVentricle':
            write_data_lv(t, t_cycle, phase, cycle, wk, model, est, accuracy, results, new=new, dir_out=dir_out,
                          save_fiber_vector=fiber_reorientation)

            # Print some state information about the completed timestep:
            if MPI.rank(mpi_comm_world()) == 0:
                msg = ('*** [Cycle {}/{} - Phase = {}/4]:'
                       ' t = {} ms, t_cycle = {} ms,'
                       ' p_lv = {:5.2f} kPa, V_lv = {:6.2f} ml')
                print(msg.format(cycle, inputs['number_of_cycles'], phase['lv'],
                                 t, t_cycle,
                                 model.pressure['lv'], model.volume['lv']))
        else:
            raise ValueError('Unknown heart_type "{}".'.format(heart_type))

        # Check if the active stress's internal time needs to be reset. (Based on LV phase).
        if phase['lv'] < 3 and t_active_stress >= inputs['time']['tc']:
            t_active_stress = t_active_stress - inputs['time']['tc']

        # Exit if maximum cycles reached:
        if cycle > inputs['number_of_cycles']:
            if MPI.rank(mpi_comm_world()) == 0:
                print('Maximum number of cycles simulated!')
            break


def timestep_biv(t_active_stress, dt, cycle, wk_dict, biv, solver, total_volume, fiber_reorientation,
                 disable_reorientation=False):
    """
    Routine for a timestep for biventricular simulations.
    """
    # Extract windkessel objects.
    wk_sys = wk_dict['sys']
    wk_pul = wk_dict['pul']

    # -------------------------------------------------------------------- #
    # t = n                                                                #
    # -------------------------------------------------------------------- #
    # Store a copy/backup of the state values and unknowns at t = n:
    v_old_sys = wk_sys.volume
    q_old_sys = wk_sys.flowrate

    v_old_pul = wk_pul.volume
    q_old_pul = wk_pul.flowrate

    vbiv_old = biv.volume
    u_array = biv.u.vector().array()
    u_old_array = biv.u_old.vector().array()

    # Update the old ls (and lc) values with most recently computed values.
    biv.active_stress.upkeep()

    # -------------------------------------------------------------------- #
    # time increment                                                       #
    # -------------------------------------------------------------------- #
    biv.dt = dt
    biv.active_stress.activation_time = t_active_stress + dt

    # -------------------------------------------------------------------- #
    # t = n + 1                                                            #
    # -------------------------------------------------------------------- #
    # Compute the new ventricular volumes and solve for the pressures.
    vlv_new = vbiv_old['lv'] + dt*(q_old_pul['ven'] - q_old_sys['art'] - q_old_sys.get('lvad', 0))
    vrv_new = vbiv_old['rv'] + dt*(q_old_sys['ven'] - q_old_pul['art'])
    v_target = {'lv': vlv_new,
                'rv': vrv_new}

    # Compute new fibers in case of fiber reorientation.
    if fiber_reorientation and not disable_reorientation:
        t_fr = time.time()
        biv.reorient_fibers(cycle)
        print_once('Fibers reoriented in {} s.'.format(time.time() - t_fr))

    # Estimate new displacement field.
    # biv.u = biv.estimate_displacement()

    print_once('Target volume: {}'.format(v_target))

    # Solve to determine pressure.
    pbiv_new, vbiv_new, accuracy, _ = solver.solve(biv, v_target)

    # Save.
    biv.pressure = pbiv_new
    biv.volume = vbiv_new

    # Update u_old with the values of u at t = n.
    biv.u_old = u_array
    biv.u_old_old = u_old_array

    # Compute the new windkessel model state:
    # Compute the new volumes with a simple forward Euler scheme:
    vart_new_sys = v_old_sys['art'] + dt*(q_old_sys['art'] - q_old_sys['per'] + q_old_sys.get('lvad', 0))
    vart_new_pul = v_old_pul['art'] + dt*(q_old_pul['art'] - q_old_pul['per'])
    vven_new_pul = v_old_pul['ven'] + dt*(q_old_pul['per'] - q_old_pul['ven'])
    vven_new_sys = total_volume - vlv_new - vart_new_sys - vrv_new - vart_new_pul - vven_new_pul \
                   - wk_sys.volume.get('lvad', 0)
    wk_sys.volume = {'art': vart_new_sys, 'ven': vven_new_sys}
    wk_pul.volume = {'art': vart_new_pul, 'ven': vven_new_pul}

    # Compute new pressures from new volumes.
    wk_sys.pressure = wk_sys.compute_pressure()
    wk_pul.pressure = wk_pul.compute_pressure()

    # Compute new flowrates from new pressures.
    # Systemic wk
    p_boundary_sys = {'in': biv.pressure['lv'],
                      'out': biv.pressure['rv']}
    q_new_sys = wk_sys.compute_flowrate(p_boundary_sys)
    wk_sys.flowrate = q_new_sys

    # Pulmonary wk
    p_boundary_pul = {'in': biv.pressure['rv'],
                      'out': biv.pressure['lv']}
    q_new_pul = wk_pul.compute_flowrate(p_boundary_pul)
    wk_pul.flowrate = q_new_pul

    return accuracy


def timestep_lv(t_active_stress, dt, cycle, wk_dict, lv, solver, fiber_reorientation, disable_reorientation=False):
    """
    Routine for a timestep for left ventricular simulations.
    """
    # Extract windkessel model.
    wk = wk_dict['sys']

    # -------------------------------------------------------------------- #
    # t = n                                                                #
    # -------------------------------------------------------------------- #
    # Store a copy/backup of the state values and unknowns at t = n:
    v_old = wk.volume
    q_old = wk.flowrate
    vlv_old = lv.volume
    u_array = lv.u.vector().array()
    u_old_array = lv.u_old.vector().array()

    # Update the old ls (and lc) values with most recently computed values.
    lv.active_stress.upkeep()

    # -------------------------------------------------------------------- #
    # time increment                                                       #
    # -------------------------------------------------------------------- #
    lv.dt = dt
    lv.active_stress.activation_time = t_active_stress + dt

    # -------------------------------------------------------------------- #
    # t = n + 1                                                            #
    # -------------------------------------------------------------------- #
    # Compute the new LV volume and solve for the LV pressure.
    v_target = vlv_old['lv'] + dt*(q_old['mv'] - q_old['ao'] - q_old.get('lvad', 0))

    # Compute new fibers in case of fiber reorientation.
    if fiber_reorientation and not disable_reorientation:
        t_fr = time.time()
        lv.reorient_fibers(cycle)
        print_once('Fibers reoriented in {} s.'.format(time.time() - t_fr))

    # Estimate new displacement field.
    # lv.u = lv.estimate_displacement()

    # Solve to determine pressure.
    plv_new, vlv_new, accuracy, _ = solver.solve(lv, v_target)

    # Save.
    lv.volume = {'lv': vlv_new}
    lv.pressure = {'lv': plv_new}

    # Update u_old with the values of u at t = n.
    lv.u_old = u_array
    lv.u_old_old = u_old_array

    # Compute the new windkessel model state:
    # Compute the new volumes with a simple forward Euler scheme:
    vart_new = v_old['art'] + dt*(q_old['ao'] - q_old['per'] + q_old.get('lvad', 0))
    vven_new = wk.parameters['total_volume'] - vart_new - vlv_new - wk.volume.get('lvad', 0)
    wk.volume = {'art': vart_new, 'ven': vven_new}

    # Compute new pressures from new volumes        .
    wk.pressure = wk.compute_pressure(wk.volume)

    # Compute new flowrates from new pressures.
    wk.flowrate = wk.compute_flowrate(wk.pressure, lv.pressure)

    return accuracy


def write_data_biv(t, t_cycle, phase, cycle, wk_dict, biv, est, accuracy, results, new=False, dir_out='.',
                   save_fiber_vector=False):
    """
    Helper function to write data to the HDF5 and CSV records for biventricular simulations.
    """

    hdf5_filename = os.path.join(dir_out, 'results_cycle_{}.hdf5'.format(cycle))

    vector_number = save_model_state_to_hdf5(biv, hdf5_filename, t, new, save_fiber_vector)

    # Extract windkessel objects.
    wk_sys = wk_dict['sys']
    wk_pul = wk_dict['pul']

    data_timestep = {
        'time': t,
        't_cycle': t_cycle,
        't_act': float(biv.active_stress.activation_time),

        'cycle': cycle,
        'phase_s': phase['lv'],
        'phase_p': phase['rv'],

        'pcav_s': biv.pressure['lv'],
        'vcav_s': biv.volume['lv'],

        'vart_s': wk_sys.volume['art'],
        'vven_s': wk_sys.volume['ven'],

        'part_s': wk_sys.pressure['art'],
        'pven_s': wk_sys.pressure['ven'],

        'qart_s': wk_sys.flowrate['art']*1000,
        'qven_s': wk_sys.flowrate['ven']*1000,
        'qper_s': wk_sys.flowrate['per']*1000,

        'pcav_p': biv.pressure['rv'],
        'vcav_p': biv.volume['rv'],

        'vart_p': wk_pul.volume['art'],
        'vven_p': wk_pul.volume['ven'],

        'part_p': wk_pul.pressure['art'],
        'pven_p': wk_pul.pressure['ven'],

        'qart_p': wk_pul.flowrate['art']*1000,
        'qven_p': wk_pul.flowrate['ven']*1000,
        'qper_p': wk_pul.flowrate['per']*1000,

        'est': est,
        'accuracy': accuracy,
        'vector_number': vector_number}

    if 'qlvad' in results.keys():
        data_timestep['qlvad'] = wk_sys.flowrate['lvad']*1000

    results.append(**data_timestep)

    # Save the CSV data file.
    if MPI.rank(mpi_comm_world()) == 0:
        results.save(os.path.join(dir_out, 'results.csv'))


def write_data_lv(t, t_cycle, phase, cycle, wk_dict, lv, est, acc, results, new=False, dir_out='.',
                  save_fiber_vector=False):
    """
    Helper function to write data to the HDF5 and CSV records for left ventricular simulations.
    """

    hdf5_filename = os.path.join(dir_out, 'results_cycle_{}.hdf5'.format(cycle))

    vector_number = save_model_state_to_hdf5(lv, hdf5_filename, t, new, save_fiber_vector)

    # Extract windkessel object.
    wk = wk_dict['sys']

    data_timestep = {'time': t,
                     't_cycle': t_cycle,
                     't_act': float(lv.active_stress.activation_time),

                     'cycle': cycle,
                     'phase': phase['lv'],

                     'part': wk.pressure['art'],
                     'pven': wk.pressure['ven'],
                     'plv': lv.pressure['lv'],

                     'vart': wk.volume['art'],
                     'vven': wk.volume['ven'],
                     'vlv': lv.volume['lv'],

                     'qao': wk.flowrate['ao']*1000,
                     'qmv': wk.flowrate['mv']*1000,
                     'qper': wk.flowrate['per']*1000,

                     'est': est,
                     'accuracy': acc,
                     'vector_number': vector_number}

    if 'qlvad' in results.keys():
        data_timestep['qlvad'] = wk.flowrate['lvad']*1000

    # This one only appends data to the record in memory.
    results.append(**data_timestep)

    # Save the CSV data file.
    if MPI.rank(mpi_comm_world()) == 0:
        results.save(os.path.join(dir_out, 'results.csv'))
