# -*- coding: utf-8 -*-
"""
This module provides classes that define alternative coordinate systems.
"""
import numpy as np
import scipy.integrate
from dolfin import Function, interpolate
from dolfin.cpp.common import Parameters
from dolfin.cpp.fem import adapt
from dolfin.cpp.function import FunctionAssigner

from .utils import vector_space_to_scalar_space

__all__ = [
    'CoordinateSystem',
    'EllipsoidalCoordinates',
    'WallBoundedCoordinates'
]


class CoordinateSystem(object):
    """
    Coordinate systems that depend on discrete degree-of-freedom coordinates can
    be defined with this class by passing in a function that returns a NumPy
    style array with computed coordinate values.

    Args:
        func: Function to use to compute coordinate values. May also be a tuple
              with arrays containing the computed coordinate values.
        *args (optional): Additional arguments to pass to the custom function.
    """
    def __init__(self, func, *args):
        self._func = func
        self._args = args

        self._array = None
        self._function = None

    def to_array(self, V=None):
        """
        Return discrete values of the coordinate system defined in V.

        Args:
            V: :class:`~dolfin.FunctionSpace` to define the coordinates on.

        Returns:
            :class:`~numpy.array`
        """
        if self._array is None:
            if type(self._func) is tuple:
                self._array = self._func
            else:
                self._array = self._func(V, *self._args)
        return self._array

    def to_function(self, V=None):
        """
        Return a :class:`~dolfin.Function` object representing the coordinate
        values.

        Args:
            V: :class:`~dolfin.FunctionSpace` to define the coordinates on.

        Returns:
            :class:`~dolfin.Function`
        """
        if not self._function:
            u = Function(V)
            array = self.to_array(V)

            try:
                u.vector()[:] = array
                u.vector().apply('')

            except IndexError:
                Q = vector_space_to_scalar_space(V)

                u_split = u.split()
                u_scalars = [Function(Q) for _ in u_split]

                for i, _u in enumerate(u_scalars):
                    try:
                        _u.vector()[:] = array[i]

                    except TypeError:
                        _array = np.array([float(v) for v in array[i]])
                        _u.vector()[:] = _array

                    _u.vector().apply('')

                fa = [FunctionAssigner(V.sub(i), Q) for i in range(len(u))]
                [fa[i].assign(u_split[i], j) for i, j in enumerate(u_scalars)]

            self._function = u

        return self._function

class EllipsoidalCoordinates(object):
    """
    Ellipsoidal coordinate system defined by radial (σ), longitudinal (τ), and
    circumferential (φ) values.

    https://en.wikipedia.org/wiki/Prolate_spheroidal_coordinates
    """

    @staticmethod
    def sigma(V, focus):
        """
        Ellipsoidal radial position defined such that constant values represent
        an ellipsoidal surface.

        Args:
            V: :class:`~dolfin.FunctionSpace` to define the coordinates on.
            focus: Distance from the origin to the shared focus points.

        Returns:
            :class:`~numpy.array`
        """
        # Extract the coordinate locations of the degrees of freedoms.
        X = V.tabulate_dof_coordinates().reshape(-1, 3)
        # TODO Check that V is in proper space.

        # Split X into x, y, and z components.
        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]

        # Compute and return the sigma values.
        return 0.5*(np.sqrt(x**2 + y**2 + (z + focus)**2)
                    + np.sqrt(x**2 + y**2 + (z - focus)**2))/focus

    @staticmethod
    def tau(V, focus):
        """
        Ellipsoidal longitudinal position defined in [-1, 1].

        Args:
            V: :class:`~dolfin.FunctionSpace` to define the coordinates on.
            focus: Distance from the origin to the shared focus points.

        Returns:
            :class:`~numpy.array`
        """
        # Extract the coordinate locations of the degrees of freedoms.
        X = V.tabulate_dof_coordinates().reshape(-1, 3)
        # TODO Check that V is in proper space.

        # Split X into x, y, and z components.
        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]

        # Compute and return the tau values.
        return 0.5*(np.sqrt(x**2 + y**2 + (z + focus)**2)
                    - np.sqrt(x**2 + y**2 + (z - focus)**2))/focus

    @staticmethod
    def phi(V):
        """
        Ellipsoidal circumferential position defined in [-π, π].

        Args:
            V: :class:`~dolfin.FunctionSpace` to define the coordinates on.

        Returns:
            :class:`~numpy.array`
        """
        # Extract the coordinate locations of the degrees of freedoms.
        X = V.tabulate_dof_coordinates().reshape(-1, 3)
        # TODO Check that V is in proper space.

        # Split X into x and y components.
        x = X[:, 0]
        y = X[:, 1]

        # Compute and return the phi values.
        return np.arctan2(y, x)


class WallBoundedCoordinates(object):
    """
    Wall-bounded ellipsoidal coordinate system defined by longitudinal (u) and
    radial (v) values.

    https://www.ncbi.nlm.nih.gov/pubmed/19592607
    """

    @staticmethod
    def u(V, sig, tau, cutoff):
        """
        Wall-bounded normalized longitudinal position defined in [-1, 0.5].

        Args:
            V: :class:`~dolfin.FunctionSpace` to define the coordinates on.
            sig: Ellipsoidal radial position σ.
            tau: Ellipsoidal longitudinal position τ.
            cutoff: Distance above the xy-plane to truncate the ellipsoids.

        Returns:
            :class:`~numpy.array`
        """
        # Extract the z coordinate locations of the degrees of freedoms.
        z = V.tabulate_dof_coordinates().reshape(-1, 3)[:, 2]
        # TODO Check that V is in proper space.

        # Extract the coordinate values of tau in array form.
        sig_array = sig.to_array(V)
        tau_array = tau.to_array(V)

        # Create an empty array for future v values.
        u_array = np.zeros(len(sig_array))

        # Normalize values of u above origin plane in a linear manner.
        u_array[z > 0] = 0.5/cutoff*z[z > 0]

        # Define a function for SciPy's integrator to return wall distances.
        def fx(x_val, s_val):
            # Formula is scale factors from
            # https://en.wikipedia.org/wiki/Prolate_spheroidal_coordinates
            return np.sqrt(s_val**2 - x_val**2)/np.sqrt(1 - x_val**2)

        # Integrate for each point to obtain the normalized v distances.
        for i, s in enumerate(sig_array):
            if z[i] <= 0:
                AB = scipy.integrate.quad(fx, 0, -1, args=(s,))
                AP = scipy.integrate.quad(fx, 0, tau_array[i], args=(s,))
                u_array[i] = -AP[0]/AB[0]

        # Return the normalized u coordinate values.
        return u_array

    @staticmethod
    def v(V, sig, tau, sigma_inner, sigma_outer):
        """
        Wall-bounded normalized radial position defined in [-1, 1] such that
        constant values represent an ellipsoidal surface.

        Args:
            V: :class:`~dolfin.FunctionSpace` to define the coordinates on.
            sig: Ellipsoidal radial position σ.
            tau: Ellipsoidal longitudinal position τ.
            sigma_inner: Value of σ on the inner surface.
            sigma_outer: Value of σ on the outer surface.

        Returns:
            :class:`~numpy.array`
        """
        # Extract the coordinate values of sigma and tau in array form.
        sig_array = sig.to_array(V)
        tau_array = tau.to_array(V)
        # TODO Check that V is in proper space.

        # Create an empty array for future v values.
        v_array = np.zeros(len(sig_array))

        # Define a function for SciPy's integrator to return wall distances.
        def fx(x_val, t_val):
            # Formula is scale factors from
            # https://en.wikipedia.org/wiki/Prolate_spheroidal_coordinates
            return np.sqrt(x_val**2 - t_val**2)/np.sqrt(x_val**2 - 1)

        # Integrate for each point to obtain the normalized v distances.
        for i, t in enumerate(tau_array):
            AB = scipy.integrate.quad(fx, sigma_inner, sigma_outer, args=(t,))
            AP = scipy.integrate.quad(fx, sigma_inner, sig_array[i], args=(t,))
            v_array[i] = 2*AP[0]/AB[0] - 1

        # Return the normalized v coordinate values.
        return v_array
