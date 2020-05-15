# -*- coding: utf-8 -*-
"""
This module provides classes that glues together mathematics and FEniCS.
"""
from dolfin import (Constant, DOLFIN_EPS, FacetNormal, Function, Measure,
                    TestFunction, VectorFunctionSpace, assemble_system,
                    derivative, det, dot, dx, grad, inner, inv, project, DirichletBC, assemble, TrialFunction,
                    LocalSolver, interpolate)
from dolfin.cpp.common import Parameters
from dolfin.cpp.la import NonlinearProblem, as_backend_type

import numpy as np
import time

from cvbtk.utils import print_once, vector_space_to_tensor_space, vector_space_to_scalar_space, save_to_disk
from cvbtk.mechanics import deformation_gradient, right_cauchy_green_deformation, fiber_stretch_ratio

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

__all__ = [
    'TimeVaryingElastance',
    'VentriclesWrapper',
    'LeftVentricleModel',
    'BiventricleModel',
    'MomentumBalance',
    'FiberReorientation'
]


class TimeVaryingElastance(object):
    """
    Time varying elastance model for left or right ventricle.

    Args:
        name (str, optional): Key for the pressure and volume dictionaries (e.g. 'lv' or 'rv' or 'cav' (default)
        **kwargs: Arbitrary keyword arguments for user-defined parameters.
    """

    def __init__(self, key = 'cav', **kwargs):
        self._parameters = self.default_parameters(key+'_time_var_elastance')
        self._parameters.update(kwargs)

        self._key = key

        # (Initial default) state values:
        self._pressure = 0.0  # [kPa]
        self._pressure_old = 0.0
        self._volume = self.parameters['ventricle_resting_volume']  # [ml]

    @staticmethod
    def default_parameters(name):
        """
        Return a set of default parameters for this model.

        Args:
            name (str): Name for the parameter set
        """
        prm = Parameters(name)

        prm.add('elastance_pas', 0.005) # [kPa/ml]
        prm.add('elastance_max', 0.3) # [kPa/ml]

        prm.add('ventricle_resting_volume', 0.) # [ml]

        prm.add('time_cycle', 1000.) # [ms]
        prm.add('time_activation', 500.) # [ms]
        prm.add('time_depolarization', 0.) # [ms]

        return prm

    @property
    def parameters(self):
        """
        Return user-defined parameters for this object.
        """
        return self._parameters

    @property
    def pressure(self):
        """
        Return a dictionary of pressures for the current state.
        """
        return {self._key: self._pressure}

    @property
    def key(self):
        """
        Return a dictionary of pressures for the current state.
        """
        return self._key

    @key.setter
    def key(self, value):
        # Not permitted to change key
        raise NotImplementedError

    @pressure.setter
    def pressure(self, values):
        """
        Set the pressures of the current state from a given dictionary of
        pressures.
        """
        self._pressure_old = self._pressure
        self._pressure = float(values.get(self._key, self._pressure))

    @property
    def pressure_old(self):
        """
        Return a dictionary of pressures for the current state.
        """
        return {self._key: self._pressure_old}

    @property
    def volume(self):
        """
        Return a dictionary of volumes for the current state.
        """
        return {self._key: self._volume}

    @volume.setter
    def volume(self, values):
        """
        Set the volume of the current state from a given dictionary of volumes.
        """
        self._volume = float(values.get(self._key, self._volume))

    @staticmethod
    def activation_function(t, t_act, t_dep):
        if t > t_dep and t < (t_dep + t_act):
            act = (np.sin(np.pi*(t - t_dep)/t_act))**2
            # taur = 300
            # taud = 100
            # amp = 1/0.14
            # act = amp*(np.tanh((t-t_dep)/taur))**2 * (np.tanh((t_act - (t-t_dep))/taud))**2
        else:
            act = 0
        return act

    def compute_pressure(self, t, ventricle_volume = None):
        """
        Computes the pressure according to the elastance model, given a certain volume and time
        Args:
            t: current time in ms(in the cycle or elapsed time since first start of cycle)
            ventricle_volume (optional): volume at which to compute the pressure. Current state is used by default.
        Returns:
            dictionary with new pressure
        """
        # Extract relevant volumes
        if ventricle_volume is None:
            vcav = self.volume[self._key]
        else:
            vcav = ventricle_volume[self._key]

        # Extract relevant model parameters
        t_cycle = self.parameters['time_cycle']
        t_act = self.parameters['time_activation']
        t_dep = self.parameters['time_depolarization']
        e_pas = self.parameters['elastance_pas']
        e_max = self.parameters['elastance_max']
        vcav_rest = self.parameters['ventricle_resting_volume']

        # compute time in current cycle
        if t > t_cycle:
            t = t - t//t_cycle*t_cycle

        # compute activation function
        act = self.activation_function(t, t_act, t_dep)

        # compute elastance
        elastance = e_pas + act*(e_max - e_pas)

        # compute pressure
        p = elastance*(vcav - vcav_rest)

        return {self._key : p}

    def compute_volume(self, t, ventricle_pressure = None):
        """
        Computes the volume according to the elastance model, given a certain pressure and time
        Args:
            t: current time in ms(in the cycle or elapsed time since first start of cycle)
            ventricle_pressure (optional): pressure at which to compute the volume.  Current state is used by default.
        Returns:
            Dictionary of computed volume.
        """
        # Extract relevant pressure
        if ventricle_pressure is None:
            pcav = self.pressure[self._key]
        else:
            pcav = ventricle_pressure[self._key]

        # Extract relevant model parameters
        t_cycle = self.parameters['time_cycle']
        t_act = self.parameters['time_activation']
        t_dep = self.parameters['time_depolarization']
        e_pas = self.parameters['elastance_pas']
        e_max = self.parameters['elastance_max']
        vcav_rest = self.parameters['ventricle_resting_volume']

        # compute time in current cycle
        if t > t_cycle:
            t = t - t//t_cycle*t_cycle

        # compute activation function
        act = self.activation_function(t, t_act, t_dep)

        # compute elastance
        elastance = e_pas + act*(e_max - e_pas)

        # compute pressure
        v = pcav/elastance + vcav_rest

        return {self._key : v}

class VentriclesWrapper(object):
    """
    Class that can wrap 2 TimeVaryingElastance models (for RV and LV) into one object.
    Aimed to mimic the Biventricle FEM.

    Args:
        lv (model): TimeVaryingElastance model LV
        rv (model): TimeVaryingElastance model RV
    """

    def __init__(self, lv=None, rv=None):
        self._lv = lv
        self._rv = rv

    @property
    def parameters(self):
        """
        Return a dictionary of user-defined parameters for this object.
        """
        return {'lv': self._lv.parameters,
                'rv': self._rv.parameters}

    @property
    def pressure(self):
        """
        Return a dictionary of pressures for the current state.
        """
        return {'lv': self._lv.pressure[self._lv.key],
                'rv': self._rv.pressure[self._rv.key]}

    @pressure.setter
    def pressure(self, values):
        """
        Set the pressures of the current state from a given dictionary of
        pressures.
        """
        self._lv.pressure = values
        self._rv.pressure = values

    @property
    def pressure_old(self):
        """
        Return a dictionary of pressures for the current state.
        """
        return {'lv': self._lv.pressure_old[self._lv.key],
                'rv': self._rv.pressure_old[self._rv.key]}

    @property
    def volume(self):
        """
        Return a dictionary of volumes for the current state.
        """
        return {'lv': self._lv.volume[self._lv.key],
                'rv': self._rv.volume[self._rv.key]}

    @volume.setter
    def volume(self, values):
        """
        Set the volume of the current state from a given dictionary of volumes.
        """
        self._lv.volume = values
        self._rv.volume = values

    def compute_volume(self, t, ventricle_pressure = None):
        return {'lv': self._lv.compute_volume(t, ventricle_pressure)[self._lv.key],
                'rv': self._rv.compute_volume(t, ventricle_pressure)[self._rv.key]}

    def compute_pressure(self, t, ventricle_volume = None):
        return {'lv': self._lv.compute_pressure(t, ventricle_volume)[self._lv.key],
                'rv': self._rv.compute_pressure(t, ventricle_volume)[self._rv.key]}

class LeftVentricleModel(object):
    """
    High-level interface for simulating a left ventricle using FEniCS.

    Args:
        geometry: Geometry object, e.g., :class:`~cvbtk.LeftVentricleGeometry`.
        **kwargs: Arbitrary keyword arguments for user-defined parameters.
    """
    def __init__(self, geometry, **kwargs):
        self._parameters = self.default_parameters()
        self._parameters.update(kwargs)

        # Save the geometry to this instance.
        self._geometry = geometry

        # Create a DOLFIN FunctionSpace for the unknowns:
        V = VectorFunctionSpace(geometry.mesh(),
                                self._parameters['element_family'],
                                self._parameters['element_degree'])

        # Create the DOLFIN Function objects for the unknowns.
        self._u = Function(V, name='displacement')
        self._u_old = Function(V, name='displacement_old')
        self._u_old_old = Function(V, name='displacement_old_old')

        # Create a variable for the cavity/boundary pressure.
        self._g = Constant(0.0)

        # Create unset values for later.
        self._bcs = None
        self._problem = None
        self._material = None
        self._nullspace = None
        self._active_stress = None

        # Fiber reorientation.
        self._fiber_reorientation = None

        # State values:
        self._plv = 0.0
        self._plv_old = 0.0
        self._vlv = self.geometry.compute_volume()

        # Need to keep dt to estimate initial pressures.
        self._dt = 1.0
        self._dt_old = 1.0
        self._dt_old_old = 1.0

    @staticmethod
    def default_parameters():
        """
        Return a set of default parameters for this model.
        """
        prm = Parameters('left_ventricle')

        prm.add('contractility', 1.0)

        prm.add('element_degree', 2)
        prm.add('element_family', 'Lagrange')

        # Fiber reorientation.
        fr_prm = Parameters('fiber_reorientation')
        fr_prm.add('ncycles_pre', 0)
        fr_prm.add('ncycles_reorient', 0)
        fr_prm.add('kappa', 3200.)
        prm.add(fr_prm)

        return prm

    @property
    def parameters(self):
        """
        Return user-defined parameters for this geometry object.
        """
        return self._parameters

    @property
    def active_stress(self):
        """
        Return the defined active stress model.
        """
        return self._active_stress

    @active_stress.setter
    def active_stress(self, value):
        self._active_stress = value

    @property
    def bcs(self):
        """
        Return the defined Dirichlet boundary conditions.
        """
        return self._bcs

    @bcs.setter
    def bcs(self, value):
        self._bcs = [value] if not isinstance(value, list) else value

    @property
    def boundary_pressure(self):
        """
        Return the pressure variable used to apply boundary conditions on the
        endocardium.
        """
        return self._g

    @boundary_pressure.setter
    def boundary_pressure(self, value):
        self._g.assign(float(value))

    def compute_initial_guess(self):
        """
        Compute the initial guess for pressure values at the start of a new
        volume solver solution step.

        Returns:
            Dictionary of initial guesses for pressure.
        """
        p0 = self.pressure['lv']
        p1 = p0 + self.dt*(p0 - self.pressure_old['lv'])/self.dt_old
        return 0.1 if abs(p1) < DOLFIN_EPS else p1

    def compute_volume(self):
        """
        Helper routine to compute the geometry's volume for volume balancing.

        This method should pass in the correct arguments to the geometry's own
        compute_volume() method.
        """
        return self.geometry.compute_volume(surface='inner', du=self.u)

    @property
    def dt(self):
        """
        Return the current dt.
        """
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt_old_old = self._dt_old
        self._dt_old = self._dt
        self._dt = float(value)
        if self.active_stress is not None:
            self.active_stress.dt = float(value)

    @property
    def dt_old(self):
        """
        Return the previous dt.
        """
        return self._dt_old

    @property
    def dt_old_old(self):
        """
        Return the dt before the previous dt.
        """
        return self._dt_old_old

    def estimate_displacement(self):
        """
        Estimates the displacement for the next time step, based on extrapolation using
        2nd order Adams-Bashforth integration.

        Returns:
            u_new (array): array with new estimated displacement values.
        """
        dt = self.dt
        dt_old = self.dt_old
        dt_old_old = self.dt_old_old

        u_array = self.u.vector().array()
        u_old_array = self.u_old.vector().array()
        u_old_old_array = self.u_old_old.vector().array()

        if sum(abs(u_old_array)) > 0:
            dudt = (u_array - u_old_array) / dt_old

            if sum(abs(u_old_old_array)) > 0:
                # u_old_old has displacement values -> use 2nd order.
                dudt_old = (u_old_array - u_old_old_array) / dt_old_old
                du = dt * (dudt * (1 + 0.5 * dt / dt_old) - dudt_old * 0.5 * dt / dt_old)
            else:
                # u_old_old has no displacement values -> use forward Euler.
                du = dt * dudt
        else:
            # u_old has no displacement values -> cannot estimate an increment.
            du = 0.

        # Compute new estimate.
        u_new = u_array + du

        return u_new

    def fiber_reorientation(self):
        """
        Return the fiber reorientation object.
        """
        if self._fiber_reorientation is None:
            self._fiber_reorientation = FiberReorientation(self,
                                            self.parameters['fiber_reorientation'].to_dict())
        return self._fiber_reorientation

    @property
    def geometry(self):
        """
        Return the geometry for this specific model.
        """
        return self._geometry

    @property
    def material(self):
        """
        Return the defined material model.
        """
        return self._material

    @material.setter
    def material(self, value):
        self._material = value

    @property
    def nullspace(self):
        """
        Return the defined nullspace for this model.
        """
        return self._nullspace

    @nullspace.setter
    def nullspace(self, value):
        self._nullspace = value

    @property
    def pressure(self):
        """
        Return a dictionary of pressures for the current state.
        """
        return {'lv': self._plv}

    @pressure.setter
    def pressure(self, values):
        """
        Set the pressures of the current state from a given dictionary of
        pressures.
        """
        self._plv_old = self._plv
        self._plv = float(values.get('lv', self._plv))

    @property
    def pressure_old(self):
        """
        Return a dictionary of pressures for the current state.
        """
        return {'lv': self._plv_old}

    @property
    def problem(self):
        if not self._problem:
            self._problem = self._create_problem()
        return self._problem

    def reorient_fibers(self, current_cycle):
        """
        Performs fiber reorientation.

        Args:
            current_cycle (int): Specify the number of the current cycle to determine
                                 whether fibers should be reoriented.
        """
        self.fiber_reorientation().reorient_fibers(current_cycle)

    @property
    def u(self):
        """
        Return the current displacement unknown.
        """
        return self._u

    @u.setter
    def u(self, u):
        try:
            self.u.vector()[:] = u
            self.u.vector().apply('')
        except IndexError:
            self.u.assign(u)

    @property
    def u_old(self):
        """
        Return the previous displacement unknown.
        """
        return self._u_old

    @u_old.setter
    def u_old(self, u):
        try:
            self.u_old.vector()[:] = u
            self.u_old.vector().apply('')
        except IndexError:
            self.u_old.assign(u)

    @property
    def u_old_old(self):
        """
        Return the previous displacement unknown.
        """
        return self._u_old_old

    @u_old_old.setter
    def u_old_old(self, u):
        try:
            self.u_old_old.vector()[:] = u
            self.u_old_old.vector().apply('')
        except IndexError:
            self.u_old_old.assign(u)

    @property
    def volume(self):
        """
        Return a dictionary of volumes for the current state.
        """
        return {'lv': self._vlv}

    @volume.setter
    def volume(self, values):
        """
        Set the volume of the current state from a given dictionary of volumes.
        """
        self._vlv = float(values.get('lv', self._vlv))

    def _create_problem(self):
        """
        Create the problem object that is actually interfacing with DOLFIN.
        """
        # The FEM unknown and test functions.
        u = self.u
        v = TestFunction(u.ufl_function_space())

        # Get the PK2 stress tensor for the material model.
        S = self.material.piola_kirchhoff2(u)

        # Get the PK2 stress tensor for the active stress model if needed.
        if self.active_stress is not None:
            contractility = self.parameters['contractility']
            S = S + contractility*self.active_stress.piola_kirchhoff2(u)

        # Define the variational form, first from the constitutive model(s).
        F = deformation_gradient(u)
        L = inner(S, 0.5*(F.T*grad(v) + grad(v).T*F))*dx

        # Next, set up the Neumann boundary conditions on the endocardium.
        n0 = -det(F)*inv(F.T)*FacetNormal(self.geometry.mesh())
        ds = Measure('ds', subdomain_data=self.geometry.tags())
        L = L - dot(self.boundary_pressure*n0, v)*ds(self.geometry.endocardium)

        a = derivative(L, u)

        problem = MomentumBalance(a, L, bcs=self.bcs, nullspace=self.nullspace)
        return problem


class BiventricleModel(object):
    """
     High-level interface for simulating a biventricle using FEniCS.

     Args:
         geometry: Geometry object, e.g., :class:`~cvbtk.BiventricleGeometry`.
         **kwargs: Arbitrary keyword arguments for user-defined parameters.
     """

    def __init__(self, geometry, **kwargs):
        self._parameters = self.default_parameters()
        self._parameters.update(kwargs)

        # Save the geometry to this instance.
        self._geometry = geometry

        # Create a DOLFIN FunctionSpace for the unknowns:
        V = VectorFunctionSpace(geometry.mesh(),
                                self._parameters['element_family'],
                                self._parameters['element_degree'])

        # Create the DOLFIN Function objects for the unknowns.
        self._u = Function(V, name='displacement')
        self._u_old = Function(V, name='displacement_old')
        self._u_old_old = Function(V, name='displacement_old_old')

        # Create a variable for the cavity/boundary pressures.
        self._g = {'lv': Constant(0.0),
                   'rv': Constant(0.0)}

        # Create unset values for later.
        self._bcs = None
        self._problem = None
        self._material = None
        self._nullspace = None
        self._active_stress = None

        # Fiber reorientation.
        self._fiber_reorientation = None

        # State values:
        self._plv = 0.0
        self._prv = 0.0
        self._plv_old = 0.0
        self._prv_old = 0.0
        self._plv_old_old = 0.0
        self._prv_old_old = 0.0

        self._vlv = self.geometry.compute_volume('lv')
        self._vrv = self.geometry.compute_volume('rv')

        # Keep last 3 pressure and volume pairs.
        self._plv_pre = [None, None, None]
        self._prv_pre = [None, None, None]
        self._vlv_pre = [None, None, None]
        self._vrv_pre = [None, None, None]

        # Keep three-step linearization coefficients for post investigation.
        self.linearization_coefs = {'a': {'lv': 0, 'rv': 0},
                                    'b': {'lv': 0, 'rv': 0},
                                    'c': {'lv': 0, 'rv': 0}}

        # Need to keep dt to estimate initial pressures.
        self._dt = 2.0
        self._dt_old = 2.0
        self._dt_old_old = 2.0

    @staticmethod
    def default_parameters():
        """
        Return a set of default parameters for this model.
        """
        prm = Parameters('biventricular_parameters')

        prm.add('contractility', 1.0)

        prm.add('element_degree', 2)
        prm.add('element_family', 'Lagrange')

        # Fiber reorientation.
        fr_prm = Parameters('fiber_reorientation')
        fr_prm.add('ncycles_pre', 0)
        fr_prm.add('ncycles_reorient', 0)
        fr_prm.add('kappa', 3200.)
        prm.add(fr_prm)

        return prm

    @property
    def parameters(self):
        """
        Return user-defined parameters for this geometry object.
        """
        return self._parameters

    @property
    def active_stress(self):
        """
        Return the defined active stress model.
        """
        return self._active_stress

    @active_stress.setter
    def active_stress(self, value):
        self._active_stress = value

    @property
    def bcs(self):
        """
        Return the defined Dirichlet boundary conditions.
        """
        return self._bcs

    @bcs.setter
    def bcs(self, value):
        self._bcs = [value] if not isinstance(value, list) else value

    @property
    def boundary_pressure(self):
        """
        Return a dictionary with the pressure variables used to apply boundary conditions on the
        LV and RV endocardium.
        """
        return self._g

    @boundary_pressure.setter
    def boundary_pressure(self, value):
        glv = self._g['lv']
        glv.assign(float(value['lv']))
        grv = self._g['rv']
        grv.assign(float(value['rv']))
        self._g.update({'lv': glv,
                        'rv': grv})

    def compute_initial_guess(self, volume=None):
        """
        Compute the initial guess for pressure values at the start of a new
        volume solver solution step.

        Args:
            volume (dict): Dictionary with target volumes for 'lv' and 'rv'.

        Returns:
            Dictionary of initial guesses for pressure.
        """
        if float(self.active_stress.activation_time) >= 0 or volume is None:
            # First, reset the history: the pressure-volume pairs of the previous
            # time instant are no use due to a change in active stress.
            self.reset_history()

            # Use time integration to determine initial guess.
            return self.time_integrate()

        else:
            # # Use previous solutions to determine initial guess (p-V relations is not time dependent).
            # return self.estimate_pressure(volume)

            # Use time integration to determine initial guess.
            return self.time_integrate()


    def compute_volume_lv(self):
        """
        Helper routine to compute the geometry's LV volume for volume balancing.

        This method should pass in the correct arguments to the geometry's own
        compute_volume() method.
        """
        return self.geometry.compute_volume('lv', volume='cavity', du=self.u)

    def compute_volume_rv(self):
        """
        Helper routine to compute the geometry's RV volume for volume balancing.

        This method should pass in the correct arguments to the geometry's own
        compute_volume() method.
        """
        return self.geometry.compute_volume('rv', volume='cavity', du=self.u)

    def compute_volume(self):
        """
        Helper routine to compute the geometry's LV and RV volumes for volume balancing.

        This method should pass in the correct arguments to the geometry's own
        compute_volume() method.

        Returns a dictionary with keys 'lv' and 'rv'.
        """
        v_lv = self.compute_volume_lv()
        v_rv = self.compute_volume_rv()
        return {'lv': v_lv,
                'rv': v_rv}

    @property
    def dt(self):
        """
        Return the current dt.
        """
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt_old_old = self._dt_old
        self._dt_old = self._dt
        self._dt = float(value)
        if self.active_stress is not None:
            self.active_stress.dt = float(value)

    @property
    def dt_old(self):
        """
        Return the previous dt.
        """
        return self._dt_old

    @property
    def dt_old_old(self):
        """
        Return the dt before the previous dt.
        """
        return self._dt_old_old

    def estimate_displacement(self):
        """
        Estimates the displacement for the next time step, based on extrapolation using
        2nd order Adams-Bashforth integration.

        Returns:
            u_new (array): array with new estimated displacement values.
        """
        dt = self.dt
        dt_old = self.dt_old
        dt_old_old = self.dt_old_old

        u_array = self.u.vector().array()
        u_old_array = self.u_old.vector().array()
        u_old_old_array = self.u_old_old.vector().array()

        if sum(abs(u_old_array)) > 0:
            dudt = (u_array - u_old_array) / dt_old

            if sum(abs(u_old_old_array)) > 0:
                # u_old_old has displacement values -> use 2nd order.
                dudt_old = (u_old_array - u_old_old_array) / dt_old_old
                du = dt * (dudt * (1 + 0.5 * dt / dt_old) - dudt_old * 0.5 * dt / dt_old)
            else:
                # u_old_old has no displacement values -> use forward Euler.
                du = dt * dudt
        else:
            # u_old has no displacement values -> cannot estimate an increment.
            du = 0.

        # Compute new estimate.
        u_new = u_array + du

        return u_new

    def estimate_pressure(self, volume):
        """
        Estimates the pressure for a given target volume.

        Args:
            volume (dict): Dictionary with target volumes for 'lv' and 'rv'.

        Returns:
            Dictionary of pressure estimates.
        """
        if self.volume_history['lv'][2] is None or self.volume_history['rv'][2] is None:
            # Empty history -> Step 1: time integration.
            return self.time_integrate()

        elif self.volume_history['lv'][1] is None or self.volume_history['rv'][1] is None:
            # 1 entry in the history -> Step 2:
            p1 = {'lv': self.pressure_history['lv'][2],
                  'rv': self.pressure_history['rv'][2]}
            r1_abs = {'lv': self.volume_history['lv'][2] - volume['lv'],
                      'rv': self.volume_history['rv'][2] - volume['rv']}

            # Estimate new pressure.
            p2 = {}
            for key, factor in zip(['lv', 'rv'], [1, 1/5]):
                p2_i = p1[key] * (1 - 0.1*r1_abs[key]*factor)
                if abs(p1[key] - p2_i) < DOLFIN_EPS:
                    p2_i = p1[key] - 0.005*factor if r1_abs[key] > 0 else p1[key] + 0.005*factor
                p2[key] = p2_i
            return p2

        elif self.volume_history['lv'][0] is None or self.volume_history['rv'][0] is None:
            # 2 entries in history -> Step 3:
            # Last entry is the newest.
            p1 = {'lv': self.pressure_history['lv'][1],
                  'rv': self.pressure_history['rv'][1]}
            r1_abs = {'lv': self.volume_history['lv'][1] - volume['lv'],
                      'rv': self.volume_history['rv'][1] - volume['rv']}

            p2 = {'lv': self.pressure_history['lv'][2],
                  'rv': self.pressure_history['rv'][2]}
            r2_abs = {'lv': self.volume_history['lv'][2] - volume['lv'],
                      'rv': self.volume_history['rv'][2] - volume['rv']}

            # Estimate new pressure: ignore p_rv influence on V_lv.
            p3 = {}
            for key, factor in zip(['lv', 'rv'], [1, 1/5]):
                if abs(r2_abs[key] - r1_abs[key]) < DOLFIN_EPS or key == 'rv': # We can't ignore p_lv influence on V_rv.
                    p3_i = p2[key] - 0.005*factor if r2_abs[key] > 0 else p2[key] + 0.005*factor
                else:
                    # Estimate new pressures from the residual.
                    p3_i = p2[key] - r2_abs[key] * (p2[key] - p1[key]) / (r2_abs[key] - r1_abs[key])
                p3.update({key: p3_i})
            return p3

        else:
            # 3 entries in history -> Step 4.
            # Last entry is the newest.
            p1 = {'lv': self.pressure_history['lv'][0],
                  'rv': self.pressure_history['rv'][0]}
            r1_abs = {'lv': self.volume_history['lv'][0] - volume['lv'],
                      'rv': self.volume_history['rv'][0] - volume['rv']}

            p2 = {'lv': self.pressure_history['lv'][1],
                  'rv': self.pressure_history['rv'][1]}
            r2_abs = {'lv': self.volume_history['lv'][1] - volume['lv'],
                      'rv': self.volume_history['rv'][1] - volume['rv']}

            p3 = {'lv': self.pressure_history['lv'][2],
                  'rv': self.pressure_history['rv'][2]}
            r3_abs = {'lv': self.volume_history['lv'][2] - volume['lv'],
                      'rv': self.volume_history['rv'][2] - volume['rv']}

            # Estimate new pressure.
            return self.three_step_linearization(p1, p2, p3, r1_abs, r2_abs, r3_abs)

    def fiber_reorientation(self):
        """
        Return the fiber reorientation object.
        """
        if self._fiber_reorientation is None:
            self._fiber_reorientation = FiberReorientation(self,
                                            self.parameters['fiber_reorientation'].to_dict())
        return self._fiber_reorientation

    @property
    def geometry(self):
        """
        Return the geometry for this specific model.
        """
        return self._geometry

    @property
    def material(self):
        """
        Return the defined material model.
        """
        return self._material

    @material.setter
    def material(self, value):
        self._material = value

    @property
    def nullspace(self):
        """
        Return the defined nullspace for this model.
        """
        return self._nullspace

    @nullspace.setter
    def nullspace(self, value):
        self._nullspace = value

    @property
    def pressure(self):
        """
        Return a dictionary of pressures for the current state.
        """
        return {'lv': self._plv,
                'rv': self._prv}

    @pressure.setter
    def pressure(self, values):
        """
        Set the pressures of the current state from a given dictionary of
        pressures.
        """
        self._plv_old_old = self._plv_old
        self._prv_old_old = self._prv_old
        self._plv_old = self._plv
        self._prv_old = self._prv
        self._plv = float(values.get('lv', self._plv))
        self._prv = float(values.get('rv', self._prv))

    @property
    def pressure_history(self):
        """
        Return a dictionary of volumes for the last 3 states.
        """
        return {'lv': self._plv_pre,
                'rv': self._prv_pre}

    @pressure_history.setter
    def pressure_history(self, values):
        """
        Updates the pressure history with the latest state values.

        Args:
            values: Dictionary with pressures of latest state.
        """
        # LV.
        self._plv_pre[0] = self._plv_pre[1]
        self._plv_pre[1] = self._plv_pre[2]
        self._plv_pre[2] = float(values.get('lv', self._plv_pre[2]))

        # RV.
        self._prv_pre[0] = self._prv_pre[1]
        self._prv_pre[1] = self._prv_pre[2]
        self._prv_pre[2] = float(values.get('rv', self._prv_pre[2]))

    @property
    def pressure_old(self):
        """
        Return a dictionary of pressures for the previous state.
        """
        return {'lv': self._plv_old,
                'rv': self._prv_old}

    @pressure_old.setter
    def pressure_old(self, values):
        """
        Return a dictionary of pressures for the previous state.
        """
        self._plv_old_old = self._plv_old
        self._prv_old_old = self._prv_old
        self._plv_old = float(values.get('lv', self._plv_old))
        self._prv_old = float(values.get('rv', self._prv_old))

    @property
    def pressure_old_old(self):
        """
        Return a dictionary of pressures for the state before the previous state.
        """
        return {'lv': self._plv_old_old,
                'rv': self._prv_old_old}

    @pressure_old_old.setter
    def pressure_old_old(self, values):
        """
        Return a dictionary of pressures for the state before the previous state.
        """
        self._plv_old_old = float(values.get('lv', self._plv_old_old))
        self._prv_old_old = float(values.get('rv', self._prv_old_old))

    @property
    def problem(self):
        if not self._problem:
            self._problem = self._create_problem()
        return self._problem

    def reorient_fibers(self, current_cycle):
        """
        Performs fiber reorientation.

        Args:
            current_cycle (int): Specify the number of the current cycle to determine
                                 whether fibers should be reoriented.
        """
        self.fiber_reorientation().reorient_fibers(current_cycle)

    def reset_history(self):
        """
        Delete history entries. Used when going to a new active stress state.
        """
        self._plv_pre = [None, None, None]
        self._prv_pre = [None, None, None]
        self._vlv_pre = [None, None, None]
        self._vrv_pre = [None, None, None]

    def time_integrate(self):
        """
        Compute new pressure from time integrating previous pressures.
        Use a 1-step Euler method or a 2-step Adams-Bashforth method,
        depending on the information available.

        Returns:
            Dictionary of pressure estimates.
        """
        p = self.pressure
        p_old = self.pressure_old
        p_old_old = self.pressure_old_old
        dt = self.dt
        dt_old = self.dt_old
        dt_old_old = self.dt_old_old

        p_new = {}
        p_init = {'lv': 0.02,
                  'rv': 0.01}
        for key in ['lv', 'rv']:

            if p_old_old[key] > 0.0:
                # Use 2-step method: Adams-Bashforth.
                # Derived by using polynomial interpolation like on
                # https://en.wikiversity.org/wiki/Adams-Bashforth_and_Adams-Moulton_methods
                # where the discretized time can have different timesteps (i.e. dt does not (necessarily) equal dt_old)
                dpdt = (p[key] - p_old[key])/dt_old
                dpdt_old = (p_old[key] - p_old_old[key])/dt_old_old
                dp = dt * (dpdt * (1 + 0.5*dt/dt_old) - dpdt_old*0.5*dt/dt_old)
                p_new_ = p[key] + dp
            else:
                # Use 1-step method: Euler.
                p_new_ = p[key] + dt * (p[key] - p_old[key]) / dt_old

            # If pressure equal to zero, we are in first iteration and we do not want zero pressure.
            p_new[key] = p_init[key] if abs(p_new_) < DOLFIN_EPS else p_new_

        return p_new

    def three_step_linearization(self, p1, p2, p3, r1_abs, r2_abs, r3_abs):
        """
        Estimate new pressures from the residual.
        1:
            Determine coefficients a, b, c of assumed relation between residual and LV and RV pressures:
            r_i = a_i*plv + b_i*prv + c_i   for i in [lv, rv]
        2:
            Solve system of equations for plv and prv:
                a_lv*plv + b_lv*prv + c_lv = 0
                a_rv*plv + b_rv*prv + c_rv = 0

        Returns:
            Dictionary with new pressures for 'lv' and 'rv'.
        """
        # 1: Determine coefficients.
        a = {}
        b = {}
        c = {}
        for key in ['lv', 'rv']:
            # if abs(r2_abs[key] - r1_abs[key]) < DOLFIN_EPS:
            #     p.update({key: p2[key]}) # Use previous pressure.
            # else:
            b_i_nom = r3_abs[key] - r2_abs[key] - (r1_abs[key] - r2_abs[key]) / (p1['lv'] - p2['lv']) * (
                    p3['lv'] - p2['lv'])
            b_i_den = p3['rv'] - p2['rv'] - (p1['rv'] - p2['rv']) / (p1['lv'] - p2['lv']) * (p3['lv'] - p2['lv'])
            b.update({key: b_i_nom / b_i_den})

            a_i_nom = r1_abs[key] - r2_abs[key] - b[key] * (p1['rv'] - p2['rv'])
            a_i_den = p1['lv'] - p2['lv']
            a.update({key: a_i_nom / a_i_den})

            c_i = r3_abs[key] - a[key] * p3['lv'] - b[key] * p3['rv']
            c.update({key: c_i})

        # Save coefficients for possible postprocessing.
        self.linearization_coefs.update({'a': a,
                                         'b': b,
                                         'c': c})

        # 2: Determine new pressure estimates.
        p_rv_nom = a['rv']*c['lv'] - a['lv']*c['rv']
        p_rv_den = a['lv']*b['rv'] - a['rv']*b['lv']

        # Perform some checks.
        if abs(p_rv_den) < DOLFIN_EPS: # No division by zero
            factor = 0.005/5
            p_rv = p3['rv'] - factor if r3_abs['rv'] > 0 else p3['rv'] + factor
        else:
            p_rv = p_rv_nom/p_rv_den

        if abs(a['lv']) < DOLFIN_EPS: # No division by zero
            factor = max(0.0001, abs(0.1*r3_abs['lv']))
            p_lv = p3['lv'] - factor if r3_abs['lv'] > 0 else p3['lv'] + factor
        else:
            p_lv = -(b['lv'] * p_rv + c['lv'])/a['lv']

        # New pressure may not be the same as previous pressure.
        if abs(p_lv - p3['lv']) <= DOLFIN_EPS:
            factor = abs(0.1*r3_abs['lv'])
            p_lv = p3['lv'] - factor if r3_abs['lv'] > 0 else p3['lv'] + factor

        if abs(p_rv - p3['rv']) <= DOLFIN_EPS:
            factor = abs(0.1/5 * r3_abs['rv'])
            p_rv = p3['rv'] - factor if r3_abs['rv'] > 0 else p3['rv'] + factor

        # No zero or negative pressures:
        # p_lv = 0.02 if p_lv < DOLFIN_EPS else p_lv
        # p_rv = 0.01 if p_rv < DOLFIN_EPS else p_rv

        return {'lv': p_lv,
                'rv': p_rv}

    @property
    def u(self):
        """
        Return the current displacement unknown.
        """
        return self._u

    @u.setter
    def u(self, u):
        try:
            self.u.vector()[:] = u
            self.u.vector().apply('')
        except IndexError:
            self.u.assign(u)

    @property
    def u_old(self):
        """
        Return the previous displacement unknown.
        """
        return self._u_old

    @u_old.setter
    def u_old(self, u):
        try:
            self.u_old.vector()[:] = u
            self.u_old.vector().apply('')
        except IndexError:
            self.u_old.assign(u)

    @property
    def u_old_old(self):
        """
        Return the previous displacement unknown.
        """
        return self._u_old_old

    @u_old_old.setter
    def u_old_old(self, u):
        try:
            self.u_old_old.vector()[:] = u
            self.u_old_old.vector().apply('')
        except IndexError:
            self.u_old_old.assign(u)

    def update_history(self, pressure_new, volume_new):
        """
        Updates the volume and pressure history.

        Args:
            pressure_new: Dictionary with new pressure values (for 'lv' and 'rv')
            volume_new: Dictionary with new volume values (for 'lv' and 'rv')
        """
        # Append the new values to the history.
        self.pressure_history = pressure_new
        self.volume_history = volume_new

        # Pressure increments in history may not be equal for both ventricles
        # (will lead to division by zero in three_step_linearization).
        plv = self.pressure_history['lv']
        prv = self.pressure_history['rv']
        if plv[0] is not None and \
                abs((plv[2] - plv[1]) - (plv[1] - plv[0])) < DOLFIN_EPS and \
                abs((prv[2] - prv[1]) - (prv[1] - prv[0])) < DOLFIN_EPS:
            # Delete oldest entry from history.
            # LV.
            self._plv_pre[0] = None
            self._vlv_pre[0] = None

            # RV.
            self._prv_pre[0] = None
            self._vrv_pre[0] = None

    @property
    def volume(self):
        """
        Return a dictionary of volumes for the current state.
        """
        return {'lv': self._vlv,
                'rv': self._vrv}

    @volume.setter
    def volume(self, values):
        """
        Set the volume of the current state from a given dictionary of volumes.
        """
        self._vlv = float(values.get('lv', self._vlv))
        self._vrv = float(values.get('rv', self._vrv))

    @property
    def volume_history(self):
        """
        Return a dictionary of volumes for the last 3 states.
        """
        return {'lv': self._vlv_pre,
                'rv': self._vrv_pre}

    @volume_history.setter
    def volume_history(self, values):
        """
        Updates the volume history with the latest state values.

        Args:
            values: Dictionary with volumes of latest state.
        """
        # LV.
        self._vlv_pre[0] = self._vlv_pre[1]
        self._vlv_pre[1] = self._vlv_pre[2]
        self._vlv_pre[2] = float(values.get('lv', self._vlv_pre[2]))

        # RV.
        self._vrv_pre[0] = self._vrv_pre[1]
        self._vrv_pre[1] = self._vrv_pre[2]
        self._vrv_pre[2] = float(values.get('rv', self._vrv_pre[2]))

    def _create_problem(self):
        """
        Create the problem object that is actually interfacing with DOLFIN.
        """
        # The FEM unknown and test functions.
        u = self.u
        v = TestFunction(u.ufl_function_space())

        # Get the PK2 stress tensor for the material model.
        S = self.material.piola_kirchhoff2(u)

        # Get the PK2 stress tensor for the active stress model if needed.
        if self.active_stress is not None:
            contractility = self.parameters['contractility']
            S = S + contractility*self.active_stress.piola_kirchhoff2(u)

        # Define the variational form, first from the constitutive model(s).
        F = deformation_gradient(u)
        L = inner(S, 0.5*(F.T*grad(v) + grad(v).T*F))*dx

        # Next, set up the Neumann boundary conditions on the endocardium.
        n0 = -det(F)*inv(F.T)*FacetNormal(self.geometry.mesh())
        ds = Measure('ds', subdomain_data=self.geometry.tags())
        L = L - dot(self.boundary_pressure['lv']*n0, v)*ds(self.geometry.lv_endocardium) \
              - dot(self.boundary_pressure['rv']*n0, v)*ds(self.geometry.rv_endocardium) \
              - dot(self.boundary_pressure['rv']*n0, v)*ds(self.geometry.rv_septum)

        a = derivative(L, u)

        problem = MomentumBalance(a, L, bcs=self.bcs, nullspace=self.nullspace)
        return problem

class MomentumBalance(NonlinearProblem):
    """
    Interface to the DOLFIN solvers for nonlinear problems.

    Args:
        a: Bilinear form.
        L: Linear form.
        bcs (optional): Dirichlet boundary conditions.
        nullspace (optional): Nullspace basis vectors.
    """
    def __init__(self, a, L, bcs=None, nullspace=None):
        super(MomentumBalance, self).__init__()
        self.a = a
        self.L = L
        self.bcs = bcs
        self.nullspace = nullspace

    def F(self, b, x):
        """
        Required DOLFIN interface.
        """
        if self.nullspace is not None:
            self.nullspace.orthogonalize(b)

    def J(self, A, x):
        """
        Required DOLFIN interface.
        """
        if self.nullspace is not None:
            as_backend_type(A).set_nullspace(self.nullspace)

    def form(self, *args):
        """
        Useful DOLFIN interface.
        """
        A, P, b, x = args
        a, L = self.a, self.L
        assemble_system(a, L, bcs=self.bcs, A_tensor=A, b_tensor=b, x0=x)

class FiberReorientation(object):

    """
    Class that performs fiber reorientation.
    https://www.ncbi.nlm.nih.gov/pubmed/18701341.

    Args:
        model: cvbtk.LeftVentricleModel or cvtbk.BiventricleModel.
        inputs (dict): inputs for this routine:
                       'ncycles_pre', # Number of cycles before we apply reorientation.
                       'ncycles_reorient', # Number of cycles to apply reorientation.
                       'kappa', # Time constant of the differential equation.
    """

    def __init__(self, model, inputs):
        # Create Function spaces.
        self._V = model.u.ufl_function_space()
        self._Q = vector_space_to_scalar_space(self._V)
        self._T = vector_space_to_tensor_space(self._V)

        self._model = model
        self._inputs = inputs

        self._H = Function(self._T)

        # Store transmural cardiac vectors at all nodes.
        self._et = np.asarray(self._model.geometry.cardiac_vectors()[2].to_array(self._V))  # NOTE: has shape (3, -1)

        # Store the coordinates of the nodes.
        ef = self._model.geometry.fiber_vectors()[0]
        Q = vector_space_to_scalar_space(ef.to_function(None).ufl_function_space())
        self._X = Q.tabulate_dof_coordinates().reshape(-1, 3)

        # Identify the nodes on the epi- and endocardium to prevent fibers sticking out of these surfaces.
        on_boundary_func = self.mark_boundary()

        # TODO Check how this works when using quadrature elements (I guess no dof lies at the endo- or epicardium).
        self._on_boundary = interpolate(on_boundary_func, Q).vector().array()

    def update_fiber_vector(self, ef0, grad_u, kappa):
        """
        Routine to update the fiber vector according to Kroon et al. 2009.

        Args:
            ef0 (array): Reference fiber vector.
            grad_u (array): 3x3 matrix with the gradient of u.
            kappa: Time constant.

        Returns:
            Updated fiber vector.
        """
        # Compute deformation gradient tensor.
        F = grad_u + np.identity(3)

        # Compute right cauchy green deformation tensor.
        C = np.transpose(F) @ F

        # Calculate right stretch tensor U from C using eigenvectors and eigenvalues.
        # Use numpy's eigh, which takes a symmetric matrix as input (C is symmetric).
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        U = np.zeros((3, 3))
        for i in range(3):
            # ith eigenvalue and eigenvector.
            w_i = eigenvalues[i]
            v_i = eigenvectors[:, i]
            U += np.sqrt(w_i) * np.outer(v_i, v_i)

        # Calculate deformed fiber vector (unnormalized).
        ef_def_ = np.dot(U, ef0)

        # Normalize deformed fiber vector.
        ef_def = ef_def_ / np.linalg.norm(ef_def_)

        # Calculate def0 using forward Euler integration of differential equation.
        def0 = self._model.dt / kappa * (ef_def - ef0)

        # Add def0 to old fiber vector.
        ef_new = ef0 + def0

        return ef_new

    def grad_u_fast(self, u):
        """
        Faster (but slightly less accurate) method to compute grad(u).

        Args:
            u (Function): Displacement field.

        Returns:
            Tensor Function with the gradient of the displacement field.
        """
        # # Project (slow!).
        # return project(grad(u), self._T)

        # Local solver (results with LocalSolver are close to using project, speed is faster).
        w = TrialFunction(self._T)
        v = TestFunction(self._T)
        a = inner(w, v) * dx
        L = inner(grad(u), v) * dx
        ls = LocalSolver(a, L)
        ls.solve_local_rhs(self._H)

        return self._H

    def mark_boundary(self):
        """
        Marks the nodes that are on the endo- or epicardium.

        Returns:
            Scalar Function with 1s (on boundary) and 0s at the nodes.
        """

        geometry = self._model.geometry

        # To extract indices of nodes belonging to endo- or epicardium, we create a
        # Function and set boundary conditions at the endo- and epicardial surfaces.
        on_boundary_func = Function(self._Q)

        # Extract boundary markers.
        boundary_markers = geometry.tags()

        # Extract the endo- and epicardial markers, based on the type of model.
        if hasattr(geometry, 'rv_endocardium'):
            # Biventricle geometry.
            boundaries = [geometry.lv_epicardium,
                          geometry.rv_epicardium,
                          geometry.lv_endocardium,
                          geometry.rv_endocardium,
                          geometry.rv_septum]
        else:
            # Leftventricle geometry.
            boundaries = [geometry.epicardium,
                          geometry.endocardium]

        # Apply the boundary conditions to the Function.
        for b in boundaries:
            bc = DirichletBC(self._Q, Constant(1.0), boundary_markers, b)
            bc.apply(on_boundary_func.vector())

        # Convert on_boundary function to array.
        return on_boundary_func

    @staticmethod
    def check_fiber_reorientation(current_cycle, ncycles_pre, ncycles_reorient):
        """
        Helper-method to check whether fiber reorientation is enabled in current cycle.
        Returns True if enabled, False if not.
        """
        print('ncycles_pre: {}'.format(current_cycle))
        return not (current_cycle < 1 + ncycles_pre or current_cycle > ncycles_pre + ncycles_reorient)

    def reorient_fibers(self, current_cycle):
        """
        Fiber reorientation routine.
        Updates the undeformed fiber orientation ef0 according to the following differential equation:

        def0/dt = 1/kappa  * (ef(F) – ef0)

        Where ef(F) is the deformed fiber orientation excluding rigid body rotations. See https://www.ncbi.nlm.nih.gov/pubmed/18701341.

        Args:
            current_cycle (int): Intiger specifying the current cycle number.
        """

        # Extract inputs.
        ncycles_pre = self._inputs['ncycles_pre']
        ncycles_reorient = self._inputs['ncycles_reorient']
        if not self.check_fiber_reorientation(current_cycle, ncycles_pre, ncycles_reorient):
            # Do not reorient.
            print_once('Not reorienting fibers...')
            return

        print_once('Reorienting fibers...')
        # Extract kappa.
        kappa = self._inputs['kappa']

        # Extract the on_boundary marker.
        on_boundary = self._on_boundary

        # Extract transmural cardiac vectors at all nodes.
        et = self._et

        # Extract the coordinates of the nodes.
        X = self._X

        # Project grad(u) onto the mesh.
        grad_u_func = self.grad_u_fast(self._model.u)

        grad_u_func.set_allow_extrapolation(True)

        # Extract fiber vectors at all nodes (or quadrature points).
        fiber_vectors = self._model.geometry.fiber_vectors()
        [ef, es, en] = [np.asarray(_.to_array(None)) for _ in fiber_vectors] # NOTE: have shapes (3, -1)

        # For every node.
        for ii in range(ef.shape[1]):
            # Extract undeformed fiber vector.
            ef0 = ef[:, ii]

            # Compute grad(u) at quadrature point.
            x, y, z = X[ii, :]
            grad_u = grad_u_func(x, y, z).reshape(3,3)

            ef_new_ = self.update_fiber_vector(ef0, grad_u, kappa)

            # Correct for fibers sticking out the walls if current node is on the endocardium or epicardium.
            if on_boundary[ii] >= 0.999:
                try:
                    # Project new fiber vector onto local transmural vector.
                    ef_t = np.dot(ef_new_, et[:, ii]) * et[:, ii]

                    # Subtract projection.
                    ef_new_ -= ef_t
                except IndexError:
                    # TODO Figure out why the index is out of bounds
                    print('Error when projecting fiber vector onto local transmural vector, continuing without projection')

            # Normalize new fiber vector.
            ef_new = ef_new_/np.linalg.norm(ef_new_)

            # Create new sheet direction and enforce orthogonality between new fiber vector (ef_new)
            # and sheet vector (es).
            es_old = es[:, ii]
            es_new_ = es_old - np.dot(ef_new, es_old) * ef_new

            # Normalize new es.
            es_new = es_new_/np.linalg.norm(es_new_)

            # Compute new sheet normal vector (en) as the cross product of ef and es
            # (keep a right-handed system).
            en_new_ = np.cross(ef_new, es_new)

            # Normalize new en (maybe redundant).
            en_new = en_new_/np.linalg.norm(en_new_)

            # Replace new fiber vectors in arrays.
            ef[:, ii] = ef_new
            es[:, ii] = es_new
            en[:, ii] = en_new

        # End loop

        # Reset the fiber vectors.
        self._model.geometry.set_fiber_vectors(ef, es, en)