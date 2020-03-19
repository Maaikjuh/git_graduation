# -*- coding: utf-8 -*-
"""
This module provides classes that define fiber angles.
"""
import numpy as np
import scipy.special
from dolfin import project

from .utils import scalar_space_to_vector_space, print_once

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

__all__ = ['DefaultFiberAngles',
           'ComputeFiberAngles']


class DefaultFiberAngles(object):
    """
    Fiber angles defined by helix (ah) and transmural (at) values.

    https://www.ncbi.nlm.nih.gov/pubmed/19592607
    """

    @staticmethod
    def helix(V, u, v, h10, h11, h12, h13, h14, h22, h24):
        """
        Default fiber angle specifying helical values.

        Args:
            V: :class:`~dolfin.FunctionSpace` to define the coordinates on.
            u: Wall-bounded ellipsoidal longitudinal position.
            v: Wall-bounded ellipsoidal radial position.
            h10: Helix fiber parameter.
            h11: Helix fiber parameter.
            h12: Helix fiber parameter.
            h13: Helix fiber parameter.
            h14: Helix fiber parameter.
            h22: Helix fiber parameter.
            h24: Helix fiber parameter.

        Returns:
            :class:`~numpy.array`
        """
        # Shortcut to the Legendre polynomial function from SciPy.
        L = scipy.special.eval_legendre

        # Extract the coordinate values of u and v in array form.
        u_array = u.to_array(V)
        v_array = v.to_array(V)
        # TODO Check that V is in proper space.

        # Compute the part of ah that depend on u.
        ah_u = h22*L(2, u_array) + h24*L(4, u_array) + 1

        # Compute the part of ah that depend on v.
        ah_v = (h10*L(0, v_array) + h11*L(1, v_array) + h12*L(2, v_array)
                + h13*L(3, v_array) + h14*L(4, v_array))

        # Combine and return the full ah array.
        return ah_u*ah_v

    @staticmethod
    def transverse(V, u, v, t11, t12, t21, t23, t25):
        """
        Default fiber angle specifying transverse values.

        Args:
            V: :class:`~dolfin.FunctionSpace` to define the coordinates on.
            u: Wall-bounded ellipsoidal longitudinal position.
            v: Wall-bounded ellipsoidal radial position.
            t11: Transverse fiber parameter.
            t12: Transverse fiber parameter.
            t21: Transverse fiber parameter.
            t23: Transverse fiber parameter.
            t25: Transverse fiber parameter.

        Returns:
            :class:`~numpy.array`
        """
        # Shortcut to the Legendre polynomial function from SciPy.
        L = scipy.special.eval_legendre

        # Extract the coordinate values of u and v in array form.
        u_array = u.to_array(V)
        v_array = v.to_array(V)
        # TODO Check that V is in proper space.

        # Compute the part of at that depend on u.
        at_u = t21*L(1, u_array) + t23*L(3, u_array) + t25*L(5, u_array)

        # Compute the part of at that depend on v.
        at_v = (t11*L(1, v_array) + t12*L(2, v_array) + 1)*(1 - v_array**2)

        # Combine and return the full at array.
        return at_u*at_v

    @staticmethod
    def ah_single(u, v, h10=0, h11=0, h12=0, h13=0, h14=0, h22=0, h24=0):
        """
         Default fiber angle specifying helical value. Same function as above,
         but evaluates at onlu a single point, given by wall bounded coordinates u, v.

         Args:
             u: Wall-bounded ellipsoidal longitudinal position.
             v: Wall-bounded ellipsoidal radial position.
             h10: Helix fiber parameter.
             h11: Helix fiber parameter.
             h12: Helix fiber parameter.
             h13: Helix fiber parameter.
             h14: Helix fiber parameter.
             h22: Helix fiber parameter.
             h24: Helix fiber parameter.

         Returns:
            Helix angle.
         """
        # Shortcut to the Legendre polynomial function from SciPy.
        L = scipy.special.eval_legendre

        # Compute the part of ah that depend on u.
        ah_u = h22 * L(2, u) + h24 * L(4, u) + 1

        # Compute the part of ah that depend on v.
        ah_v = (h10 * L(0, v) + h11 * L(1, v) + h12 * L(2, v)
                + h13 * L(3, v) + h14 * L(4, v))

        # Combine and return the full ah array.
        return ah_u * ah_v

    @staticmethod
    def at_single(u, v, t11=0, t12=0, t21=0, t23=0, t25=0):
        """
        Default fiber angle specifying transverse value. Same function as above,
         but evaluates at onlu a single point, given by wall bounded coordinates u, v.

        Args:
            u: Wall-bounded ellipsoidal longitudinal position.
            v: Wall-bounded ellipsoidal radial position.
            t11: Transverse fiber parameter.
            t12: Transverse fiber parameter.
            t21: Transverse fiber parameter.
            t23: Transverse fiber parameter.
            t25: Transverse fiber parameter.

        Returns:
            transverse fiber angle.
        """
        # Shortcut to the Legendre polynomial function from SciPy.
        L = scipy.special.eval_legendre

        # Compute the part of at that depend on u.
        at_u = t21 * L(1, u) + t23 * L(3, u) + t25 * L(5, u)

        # Compute the part of at that depend on v.
        at_v = (t11 * L(1, v) + t12 * L(2, v) + 1) * (1 - v ** 2)

        # Combine and return the full at array.
        return at_u * at_v


class ComputeFiberAngles(object):
    """

    """
    @staticmethod
    def ah(Q, cardiac_vectors, fiber_vectors):
        """
        Calculate effective helix angle.
        :param Q: FEniCS (scalar) function space (not a vector function).
        :param cardiac_vectors: Tuple with CoordinateSystems for the cardiac basis vectors (ec, el et).
        :param fiber_vectors: Tuple with CoordinateSystems for the cardiac fiber vectors (ef, es, en).
        :return: Helical fiber angles (in degrees)
        """

        # Extract the coordinate locations of the degrees of freedoms.
        X = Q.tabulate_dof_coordinates().reshape(-1, 3)

        # Extract unit vectors.
        V = scalar_space_to_vector_space(Q)
        ec, el, et = [np.asarray(e.to_array(V)) for e in cardiac_vectors]
        ef = np.asarray(fiber_vectors[0].to_array(V))

        ah = np.zeros(len(X))
        for ii in range(len(X)):
            ef_ii = ef[:, ii]
            # Fiber angles based on projection on cardiac basis.
            ah[ii] = np.arctan2(np.inner(ef_ii, el[:, ii]), np.inner(ef_ii, ec[:, ii]))

        return ah/np.pi*180

    @staticmethod
    def at(Q, cardiac_vectors, fiber_vectors):
        """
        Calculate effective transverse angle.
        :param Q: FEniCS (scalar) function space (not a vector function).
        :param cardiac_vectors: Tuple with CoordinateSystems for the cardiac basis vectors (ec, el et).
        :param fiber_vectors: Tuple with CoordinateSystems for the cardiac fiber vectors (ef, es, en).
        :return: Transverse fiber angles (in degrees)
        """

        # Extract the coordinate locations of the degrees of freedoms.
        X = Q.tabulate_dof_coordinates().reshape(-1, 3)

        # Extract unit vectors.
        V = scalar_space_to_vector_space(Q)
        ec, el, et = [np.asarray(e.to_array(V)) for e in cardiac_vectors]
        ef = np.asarray(fiber_vectors[0].to_array(V))

        at = np.zeros(len(X))
        for ii in range(len(X)):
            ef_ii = ef[:, ii]
            # Fiber angles based on projection on cardiac basis.
            at[ii] = np.arctan2(np.inner(ef_ii, et[:, ii]), np.inner(ef_ii, ec[:, ii]))

        return at/np.pi*180