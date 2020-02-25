# -*- coding: utf-8 -*-
"""
This module provides high-level solver interfaces to underlying FEniCS/PETSc
implementations.
"""
import logging
import time
import math

import os
from dolfin.cpp.common import DOLFIN_EPS, MPI, Parameters, Timer, mpi_comm_world
from dolfin.cpp.io import HDF5File
from dolfin.cpp.la import NewtonSolver
from dolfin import info, Function

from cvbtk import ArtsKerckhoffsActiveStress
from cvbtk.utils import print_once, info_once, save_to_disk

__all__ = ['VolumeSolver',
           'VolumeSolverBiV',
           'CustomNewtonSolver']


class VolumeSolver(object):
    """
    This solver works by balancing the volume of the geometry until it reaches
    the target volume.

    This class was designed to mimic the same methods available as with the
    built-in DOLFIN solver classes.

    Args:
        custom_newton_solver (optional): A custom version of the NewtonSolver .
        **kwargs: Arbitrary keyword arguments for user-defined parameters.
    """
    def __init__(self, custom_newton_solver=None, **kwargs):
        self._parameters = self.default_parameters()
        self._parameters.update(kwargs)

        # noinspection PyArgumentList
        if custom_newton_solver is None:
            newton_solver = NewtonSolver()
        else:
            newton_solver = custom_newton_solver

        newton_solver.parameters.update(self._parameters['newton_solver'])
        if newton_solver.parameters['relaxation_parameter'] is None:
            newton_solver.parameters['relaxation_parameter'] = 1.
        self._newton_solver = newton_solver

        self._volume_iteration = 0
        self._newton_iteration = 0
        self._krylov_iteration = 0

        self._r0 = 0.0

        self._log = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def default_parameters():
        """
        Return a set of default parameters for this model.
        """
        prm = Parameters('volume_solver')

        prm.add('maximum_iterations', 15)

        prm.add('absolute_tolerance', 1e-2)
        prm.add('relative_tolerance', 1e-3)

        prm.add('report', True)
        prm.add('error_on_nonconvergence', True)
        prm.add('convergence_criterion', 'residual')

        prm.add('nonzero_initial_guess', True)

        # Add the default NewtonSolver parameter to this parameter set.
        # noinspection PyArgumentList
        prm.add(NewtonSolver().default_parameters())

        return prm

    @property
    def parameters(self):
        """
        Return user-defined parameters for this solver object.
        """
        return self._parameters

    def iteration(self):
        """
        Return the last known volume iteration number.
        """
        return self._volume_iteration

    def krylov_iteration(self):
        """
        Return the last known Krylov iteration number.
        """
        return self._krylov_iteration

    def newton_iteration(self):
        """
        Return the last known Newton iteration number.
        """
        return self._newton_iteration

    @property
    def newton_solver(self):
        """
        Access the NewtonSolver object for this instance.
        """
        return self._newton_solver

    def solve(self, model, volume):
        """
        Solve the system by iteratively adapting a pressure boundary condition
        until the resulting computed volume is within tolerance to the target
        volume.

        Args:
            model: Initialized model to solve.
            volume: Target volume to solve for.

        Returns:
            Tuple of pressure, volume, and absolute and relative volume errors.
        """
        # noinspection PyUnusedLocal
        timer = Timer('volume solver total')

        # Reset the iteration count.
        self._volume_iteration = 0
        self._newton_iteration = 0
        self._krylov_iteration = 0

        # The model can define compute_initial_guess() to compute the initial
        # guess for each volume solver solution step.
        if self.parameters['nonzero_initial_guess']:
            p1 = model.compute_initial_guess()
            print_once('Initial guess pressure = {}'.format(p1))
            model.boundary_pressure = p1
        else:
            p1 = float(model.boundary_pressure)

        # First solution step using, ideally, the initial guesses from above.
        try:
            v1, r1_abs, r1_rel = self._solve(model, volume)
            converged = self._check_convergence(r1_abs, r1_rel)
            if converged:
                return p1, v1, r1_abs, r1_rel
        except RuntimeError as error_detail:
            print_once('Except RuntimeError: {}'.format(error_detail))
            self._step_fail(p1, volume)
            raise

        # Compute and assign the pressure estimate for the second step.
        p2 = p1 - 0.1*p1*r1_abs
        if abs(p1 - p2) < DOLFIN_EPS:
            p2 = p1 - 0.05 if r1_abs > 0 else p1 + 0.05
        model.boundary_pressure = p2

        # The second solution step.
        try:
            v2, r2_abs, r2_rel = self._solve(model, volume)
            converged = self._check_convergence(r2_abs, r2_rel)
            if converged:
                return p2, v2, r2_abs, r2_rel
        except RuntimeError as error_detail:
            print_once('Except RuntimeError: {}'.format(error_detail))
            self._step_fail(p2, volume)
            raise

        # Now comes the Newton method loop.
        while self.iteration() <= self.parameters['maximum_iterations']:
            if abs(r2_abs - r1_abs) < DOLFIN_EPS:
                return p2, v2, r2_abs, r2_rel

            # Estimate new pressures from the residual.
            p = p2 - r2_abs*(p2 - p1)/(r2_abs - r1_abs)
            model.boundary_pressure = p

            # Solution step.
            try:
                v, r_abs, r_rel = self._solve(model, volume)
                converged = self._check_convergence(r_abs, r_rel)
                if converged:
                    return p, v, r_abs, r_rel
            except RuntimeError as error_detail:
                print_once('Except RuntimeError: {}'.format(error_detail))
                self._step_fail(p, volume)
                raise

            # Values for next iteration.
            p1 = p2
            p2 = p
            r1_abs = r2_abs
            r2_abs = r_abs

        else:
            msg = 'Volume solver did not converge after {} iterations'
            self._log.error(msg.format(self.iteration()))
            raise RuntimeError(msg.format(self.iteration()))

    def _solve(self, model, volume):
        """
        Internal solving routine.

        Args:
            model: Initialized model to solve.
            volume: Target volume to solve for.

        Returns:
            Tuple of the volume and absolute and relative volume errors.
        """
        # Store a copy/backup of the displacement field t = n:
        u_array = model.u.vector().get_local()

        # Set the relaxation parameter to 1 (may have been lowered in a previous iteration).
        self.newton_solver.parameters['relaxation_parameter'] = 1.

        # Minimum relaxation parameter.
        min_relaxation_parameter = 0.24

        converged = False
        while self.newton_solver.parameters['relaxation_parameter'] >= min_relaxation_parameter and converged == False:
            try:
                converged = self.newton_solver.solve(model.problem, model.u.vector())[1]
                # Newton solver did not fail, but reached maximum number of iterations.
                if not converged:
                    if self.newton_solver.parameters['relaxation_parameter'] >= 2*min_relaxation_parameter:
                        # Relax newton solver and try again.
                        print_once('1 Relaxing the Newton Solver to {} and re-attempting...'.format(self.newton_solver.parameters['relaxation_parameter']/2))
                        self.newton_solver.parameters['relaxation_parameter'] = self.newton_solver.parameters[
                                                                                    'relaxation_parameter'] / 2

                        if self.newton_solver.residual() > self.newton_solver.residual0():
                            # Reset values from backup:
                            reset_values(model.u, u_array)

            except RuntimeError as error_detail:
                print_once('Except RuntimeError: {}'.format(error_detail))
                if self.newton_solver.parameters['relaxation_parameter'] >= 2*min_relaxation_parameter:
                    # Relax newton solver and try again.
                    print_once('2 Relaxing the Newton Solver to {} and re-attempting...'.format(self.newton_solver.parameters['relaxation_parameter']/2))
                    self.newton_solver.parameters['relaxation_parameter'] = self.newton_solver.parameters['relaxation_parameter']/2

                    # Reset values from backup:
                    reset_values(model.u, u_array)

                else:
                    raise RuntimeError('Newton solver did not converge.')

        v = model.compute_volume()
        r_abs = v - volume

        if self.iteration() == 0:
            self._r0 = r_abs

        r_rel = r_abs/self._r0

        self._step_success(abs(r_abs), abs(r_rel))

        self._volume_iteration += 1
        self._newton_iteration += self.newton_solver.iteration()
        self._krylov_iteration += self.newton_solver.krylov_iterations()

        return v, r_abs, r_rel

    def _check_convergence(self, r_abs, r_rel):
        absolute_tolerance = self.parameters['absolute_tolerance']
        relative_tolerance = self.parameters['relative_tolerance']

        if abs(r_abs) < absolute_tolerance or abs(r_rel) < relative_tolerance:
            self._iteration_success()
            return True

        else:
            return False

    def _iteration_success(self):
        if MPI.rank(mpi_comm_world()) == 0:
            msg = ('Volume solver finished in '
                   '{} iterations, '
                   '{} nonlinear solver iterations, and '
                   '{} linear solver iterations.')
            self._log.info(msg.format(self.iteration(),
                                      self.newton_iteration(),
                                      self.krylov_iteration()))

    def _step_fail(self, p, v):
        if MPI.rank(mpi_comm_world()) == 0:
            msg = ('Volume solver failed after '
                   '{} iterations for p = {} and v = {}')
            self._log.info(msg.format(self.iteration(), p, v))

    def _step_success(self, r_abs, r_rel):
        prm = self.parameters
        if MPI.rank(mpi_comm_world()) == 0 and prm['report'] is True:
            msg = ('Volume iteration {}: '
                   'r (abs) = {:5.3e} (tol = {:5.3e}) '
                   'r (rel) = {:5.3e} (tol = {:5.3e})')
            self._log.info(msg.format(self.iteration(),
                                      r_abs, prm['absolute_tolerance'],
                                      r_rel, prm['relative_tolerance']))


class VolumeSolverBiV(object):
    """
    This solver works by balancing the LV and RV volumes of the geometry until it reaches
    the target volumes.

    This class was designed to mimic the same methods available as with the
    built-in DOLFIN solver classes.

    Args:
        custom_newton_solver (optional): A custom version of the NewtonSolver .
        **kwargs: Arbitrary keyword arguments for user-defined parameters.
    """
    def __init__(self, custom_newton_solver = None, **kwargs):
        self._parameters = self.default_parameters()
        self._parameters.update(kwargs)

        # noinspection PyArgumentList
        if custom_newton_solver is None:
            newton_solver = NewtonSolver()
        else:
            newton_solver = custom_newton_solver
        newton_solver.parameters.update(self._parameters['newton_solver'])
        if newton_solver.parameters['relaxation_parameter'] is None:
            newton_solver.parameters['relaxation_parameter'] = 1.
        self._newton_solver = newton_solver

        self._volume_iteration = 0
        self._newton_iteration = 0
        self._krylov_iteration = 0

        self._r0 = {'lv': 0.0,
                    'rv': 0.0}

        self._log = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def default_parameters():
        """
        Return a set of default parameters for this model.
        """
        prm = Parameters('volume_solver')

        prm.add('maximum_iterations', 15)

        prm.add('absolute_tolerance', 1e-4)
        prm.add('relative_tolerance', 0.05e-2)

        prm.add('report', True)
        prm.add('error_on_nonconvergence', True)
        prm.add('convergence_criterion', 'residual')

        prm.add('nonzero_initial_guess', True)

        # Add the default NewtonSolver parameter to this parameter set.
        # noinspection PyArgumentList
        prm.add(NewtonSolver().default_parameters())

        return prm

    @property
    def parameters(self):
        """
        Return user-defined parameters for this solver object.
        """
        return self._parameters

    def iteration(self):
        """
        Return the last known volume iteration number.
        """
        return self._volume_iteration

    def krylov_iteration(self):
        """
        Return the last known Krylov iteration number.
        """
        return self._krylov_iteration

    def newton_iteration(self):
        """
        Return the last known Newton iteration number.
        """
        return self._newton_iteration

    @property
    def newton_solver(self):
        """
        Access the NewtonSolver object for this instance.
        """
        return self._newton_solver

    def solve(self, model, volume):
        """
        Solve the system by iteratively adapting a pressure boundary condition
        until the resulting computed volume is within tolerance to the target
        volume.

        Args:
            model: Initialized model to solve.
            volume: Dictionary with target volumes to solve for (keys 'lv' and 'rv').

        Returns: (p, v, r_abs, r_rel)
            Tuple with dictionaries for pressure, volume, and absolute and relative volume errors (keys 'lv' and 'rv').
        """
        # noinspection PyUnusedLocal
        timer = Timer('volume solver total')

        # Reset the iteration count.
        self._volume_iteration = 0
        self._newton_iteration = 0
        self._krylov_iteration = 0

        # The model can define compute_initial_guess() to compute the initial
        # guess for each volume solver solution step.
        if self.parameters['nonzero_initial_guess']:
            p1 = model.compute_initial_guess(volume)
            model.boundary_pressure = p1
        else:
            p1 = {'lv': float(model.boundary_pressure['lv']),
                  'rv': float(model.boundary_pressure['rv'])}

        # First solution step using, ideally, the initial guesses from above.
        try:
            print_once('First step: p = {}'.format(p1))
            v1, r1_abs, r1_rel = self._solve(model, volume)
            # Save the pressure and volumes to the history.
            model.update_history(p1, v1)
            print_once('First step: p = {}, V = {}, r_abs = {}'.format(p1, v1, r1_abs))
            converged = self._check_convergence(r1_abs, r1_rel)
            if converged:
                return p1, v1, r1_abs, r1_rel
        except RuntimeError as error_detail:
            print_once('Except RuntimeError: {}'.format(error_detail))
            self._step_fail(p1, volume)
            raise

        # raise NotImplementedError
        # Next steps until convergence.
        while self.iteration() <= self.parameters['maximum_iterations']:

            # New pressure estimate.
            p = model.estimate_pressure(volume)
            print_once('Previous p: {} kPa, new p: {} kPa'.format(p1, p))

            model.boundary_pressure = p

            # Solution step.
            try:
                v, r_abs, r_rel = self._solve(model, volume)
                # Save the pressure and volumes to the history.
                model.update_history(p, v)
                print_once('Step {}: p = {}, V = {}, r_abs = {}'.format(self.iteration(), p, v, r_abs))
                converged = self._check_convergence(r_abs, r_rel)
                if converged:
                    return p, v, r_abs, r_rel
            except RuntimeError as error_detail:
                print_once('Except RuntimeError: {}'.format(error_detail))
                self._step_fail(p, volume)
                raise

            p1 = p

        else:
            msg = 'Volume solver did not converge after {} iterations.'
            self._log.error(msg.format(self.iteration()))
            raise RuntimeError(msg.format(self.iteration()))

    def _solve(self, model, volume):
        """
        Internal solving routine.

        Args:
            model: Initialized BiventricleModel to solve.
            volume: Dictionary with target volumes to solve for (keys 'lv' and 'rv').

        Returns:
            Tuple of the volume and absolute and relative volume errors.
        """
        # Store a copy/backup of the state values and unknowns at t = n:
        u_array = model.u.vector().get_local()

        # Set the relaxation parameter to 1 (may have been lowered in a previous iteration).
        self.newton_solver.parameters['relaxation_parameter'] = 1.

        # Minimum relaxation parameter.
        min_relaxation_parameter = 0.24

        converged = False
        while self.newton_solver.parameters['relaxation_parameter'] >= min_relaxation_parameter and converged == False:
            try:
                converged = self.newton_solver.solve(model.problem, model.u.vector())[1]
                if not converged:
                    if self.newton_solver.parameters['relaxation_parameter'] >= 2*min_relaxation_parameter:
                        # Relax newton solver and try again.
                        print_once('Relaxing the Newton Solver to {} and re-attempting...'.format(self.newton_solver.parameters['relaxation_parameter']/2))
                        self.newton_solver.parameters['relaxation_parameter'] = self.newton_solver.parameters[
                                                                                    'relaxation_parameter'] / 2

                        if self.newton_solver.residual() > self.newton_solver.residual0():
                            # Reset values from backup:
                            reset_values(model.u, u_array)

            except RuntimeError as error_detail:
                print_once('Except RuntimeError: {}'.format(error_detail))
                if self.newton_solver.parameters['relaxation_parameter'] >= 2*min_relaxation_parameter:
                    # Relax newton solver and try again.
                    print_once('Relaxing the Newton Solver to {} and re-attempting...'.format(self.newton_solver.parameters['relaxation_parameter']/2))
                    self.newton_solver.parameters['relaxation_parameter'] = self.newton_solver.parameters['relaxation_parameter']/2

                    # Reset values from backup:
                    reset_values(model.u, u_array)
                else:
                    raise RuntimeError('Newton solver did not converge.')

        v = model.compute_volume()

        r_abs = {'lv': v['lv'] - volume['lv'],
                 'rv': v['rv'] - volume['rv']}

        if self.iteration() == 0:
            self._r0 = r_abs

        r_rel = {'lv': r_abs['lv']/self._r0['lv'],
                 'rv': r_abs['rv']/self._r0['rv']}

        self._step_success(r_abs, r_rel)

        self._volume_iteration += 1
        self._newton_iteration += self.newton_solver.iteration()
        self._krylov_iteration += self.newton_solver.krylov_iterations()

        return v, r_abs, r_rel

    def _check_convergence(self, r_abs, r_rel):
        absolute_tolerance = self.parameters['absolute_tolerance']
        relative_tolerance = self.parameters['relative_tolerance']

        if (abs(r_abs['lv']) < absolute_tolerance or abs(r_rel['lv']) < relative_tolerance) and \
           (abs(r_abs['rv']) < absolute_tolerance or abs(r_rel['rv']) < relative_tolerance):
            self._iteration_success()
            return True

        else:
            return False

    def _iteration_success(self):
        if MPI.rank(mpi_comm_world()) == 0:
            msg = ('Volume solver finished in '
                   '{} iterations, '
                   '{} nonlinear solver iterations, and '
                   '{} linear solver iterations.')
            self._log.info(msg.format(self.iteration(),
                                      self.newton_iteration(),
                                      self.krylov_iteration()))
            print(msg.format(self.iteration(),
                                      self.newton_iteration(),
                                      self.krylov_iteration()))

    def _step_fail(self, p, v):
        if MPI.rank(mpi_comm_world()) == 0:
            msg = ('Volume solver failed after '
                   '{} iterations for p_lv = {} and v_lv = {}, and p_rv = {} and v_rv = {}')
            self._log.info(msg.format(self.iteration(), p['lv'], v['lv'], p['rv'], v['rv']))

    def _step_success(self, r_abs, r_rel):
        prm = self.parameters
        if MPI.rank(mpi_comm_world()) == 0 and prm['report'] is True:
            msg = ('Volume iteration {}: '
                   'r_lv (abs) = {:5.3e}; r_rv (abs) = {:5.3e} (tol = {:5.3e}) '
                   'r_lv (rel) = {:5.3e}; r_rv (rel) = {:5.3e} (tol = {:5.3e})')
            self._log.info(msg.format(self.iteration(),
                                      r_abs['lv'], r_abs['rv'], prm['absolute_tolerance'],
                                      r_rel['lv'], r_rel['rv'], prm['relative_tolerance']))


def reset_values(function_to_reset, array_to_reset_from):
    """
    Helper function to reset DOLFIN quantities.
    """
    function_to_reset.vector()[:] = array_to_reset_from
    function_to_reset.vector().apply('')


# Custom Newton Solver.
class CustomNewtonSolver(NewtonSolver):
    def __init__(self, model=None, dir_out='.'):
        """
        Newton solver with extended functionalities and customized converge criteria.
        Args:
            model (optional, Finite element model): E.g. BiventricleModel
                If given, the NewtonSolver saves a XDMF file of the residual and displacement at the end of an iteration.
            dir_out (optional, str): Output directory for the residual Functions if model is given.
        """
        super(CustomNewtonSolver, self).__init__()
        self._residual0 = 0.0
        self._residual_old = 0.0
        self._residual = 0.0
        self._id = 0
        self._model = model
        self._dir_out = dir_out

    def residual(self):
        return self._residual

    def residual0(self):
        return self._residual0

    def converged(self, residual, problem, newton_iteration):
        """
        Checks for divergence. Also prevents it finishing within one iteration,
        which is not good for the pressure estimation of the BiV model, due to the volume not having changed.
        """

        atol = self.parameters['absolute_tolerance']
        rtol = self.parameters['relative_tolerance']

        # Some custom convergence criterion here
        self._residual = residual.norm("l2")

        # If this is the first iteration, set initial residual.
        if newton_iteration == 0:
            self._residual0 = self._residual*1
            self._id += 1

        # Relative residual
        relative_residual = self._residual/self._residual0

        info_once("Newton iteration %d: r (abs) = %.3e (tol = %.3e) r (rel) = %.3e (tol = %.3e)" % (newton_iteration, self._residual, atol, relative_residual, rtol))

        # Check for divergence.
        # We define multiple criteria to detect divergence:
        div_bool_1 = (newton_iteration > 3 and relative_residual > self._residual_old and relative_residual > 1.)
        div_bool_2 = relative_residual > 1e2
        div_bool_3 = self._residual > 1e4
        div_bool_4 = math.isnan(self._residual)
        div_bool_5 = (newton_iteration > 8 and relative_residual > 1.)
        if div_bool_1 or div_bool_2 or div_bool_3 or div_bool_4 or div_bool_5:
            text = 'Newton solver is diverging. r (abs): %.3e, r0 (abs) : %.3e' % (self._residual, self.residual0())
            info_once(text)

            if self._model is not None:
                # Save residual.
                self.save_residual(residual, newton_iteration)
                # Save displacement.
                self.save_displacement(newton_iteration)

            raise RuntimeError(text)

        # Save residual.
        self._residual_old = self._residual*1

        # Check for convergence.
        # Also prevent it finishing with the first iteration (meaning no deformation in new step,
        # resulting in unchanged volumes, which is not good for the pressure estimation algorithm).
        if (self._residual < atol or relative_residual < rtol) and newton_iteration > 0:
            # Converged and deformed.
            return True
        else:
            return False

    def save_residual(self, residual, newton_iteration):
        filename = os.path.join(self._dir_out,
                                'newton_residuals/residual_{}_iteration_{}.xdmf'.format(self._id, newton_iteration))

        V = self._model.u.ufl_function_space()

        residual_func = Function(V, name='residual')
        residual_func.vector()[:] = residual
        residual_func.vector().apply('')

        print_once('Saving residual to {} ...'.format(filename))

        save_to_disk(residual_func, filename)

    def save_displacement(self, newton_iteration):
        # Save to hdf5, because to xdmf is not supported in parallel.
        filename = os.path.join(self._dir_out,
                                'newton_displacement/displacement_{}_iteration_{}.xdmf'.format(self._id, newton_iteration))

        print_once('Saving displacement to {} ...'.format(filename))

        save_to_disk(self._model.u, filename)
